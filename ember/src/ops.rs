//! Chapter 4: elementwise operations and reductions. Elementwise operations apply a
//! function to every element (`map`) or to every pair of elements of two broadcast
//! tensors (`zip_map`); reductions combine the elements along one axis.

use crate::broadcast::broadcast_shapes;
use crate::math;
use crate::tensor::Tensor;

impl Tensor {
    /// The elements as one slice of the buffer, in row-major order, if the tensor is
    /// contiguous; `None` otherwise.
    pub fn as_slice(&self) -> Option<&[f32]> {
        // @solution ch04 as_slice
        let end = self.offset + self.numel();
        self.is_contiguous()
            .then(|| &self.storage[self.offset..end])
        // @end
    }

    /// `f` applied to every element: a new contiguous tensor of the same shape.
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Tensor {
        // @solution ch04 map
        let data = match self.as_slice() {
            // Fast path: the elements are one run of the buffer.
            Some(run) => run.iter().map(|&x| f(x)).collect(),
            // General path: read them in order with the odometer, then apply f.
            None => {
                let mut data = self.to_vec();
                for x in &mut data {
                    *x = f(*x);
                }
                data
            }
        };
        Tensor::from_vec(data, &self.shape)
        // @end
    }

    /// `f` applied to each pair of elements of `self` and `other`, broadcast to a
    /// common shape.
    pub fn zip_map(&self, other: &Tensor, f: impl Fn(f32, f32) -> f32) -> Tensor {
        self.binary(other, "zip_map", f)
    }

    // @listing binary
    /// The engine behind `zip_map`, `add`, `sub`, `mul` and `div`. `op` names the
    /// operation in the panic message when the shapes don't broadcast.
    fn binary(&self, rhs: &Tensor, op: &str, f: impl Fn(f32, f32) -> f32) -> Tensor {
        // Fast path: same shape, and both tensors are one run of their buffers.
        if let (Some(a), Some(b)) = (self.as_slice(), rhs.as_slice())
            && self.shape == rhs.shape
        {
            let data = a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect();
            return Tensor::from_vec(data, &self.shape);
        }
        // General path: broadcast, expand both, and walk them in step.
        let (s, r) = (&self.shape, &rhs.shape);
        let Some(shape) = broadcast_shapes(s, r) else {
            panic!("{op}: shapes {s:?} and {r:?} don't broadcast");
        };
        let (a, b) = (self.expand(&shape), rhs.expand(&shape));
        Tensor::from_vec(zip_strided(&a, &b, f), &shape)
    }
    // @end

    // @listing arithmetic
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.binary(other, "add", |x, y| x + y)
    }
    // @end

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.binary(other, "sub", |x, y| x - y)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.binary(other, "mul", |x, y| x * y)
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        self.binary(other, "div", |x, y| x / y)
    }

    // @listing unary
    pub fn add_scalar(&self, s: f32) -> Tensor {
        self.map(|x| x + s)
    }

    pub fn exp(&self) -> Tensor {
        self.map(math::exp)
    }

    pub fn powf(&self, p: f32) -> Tensor {
        self.map(|x| math::powf(x, p))
    }
    // @end

    pub fn mul_scalar(&self, s: f32) -> Tensor {
        self.map(|x| x * s)
    }

    pub fn neg(&self) -> Tensor {
        self.map(|x| -x)
    }

    pub fn ln(&self) -> Tensor {
        self.map(math::ln)
    }

    pub fn sqrt(&self) -> Tensor {
        self.map(f32::sqrt)
    }

    pub fn tanh(&self) -> Tensor {
        self.map(math::tanh)
    }

    // @listing activations
    /// `max(x, 0)`, written so that a NaN stays NaN.
    pub fn relu(&self) -> Tensor {
        self.map(|x| if x < 0.0 { 0.0 } else { x })
    }

    /// `1 / (1 + e^(-x))`. For very negative x, `e^(-x)` overflows to infinity and
    /// the result is `1 / inf = 0`, which is the right answer to f32 precision.
    pub fn sigmoid(&self) -> Tensor {
        self.map(|x| 1.0 / (1.0 + math::exp(-x)))
    }
    // @end
}

// @listing zip-strided
/// `f` applied to each pair of elements of `a` and `b`, which have the same shape, in
/// row-major order. One odometer drives a storage position in each tensor.
fn zip_strided(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    let numel = a.numel();
    let mut out = Vec::with_capacity(numel);
    let mut index = vec![0; a.ndim()];
    let (mut i, mut j) = (a.offset, b.offset);
    while out.len() < numel {
        out.push(f(a.storage[i], b.storage[j]));
        for d in (0..a.ndim()).rev() {
            index[d] += 1;
            i += a.strides[d];
            j += b.strides[d];
            if index[d] < a.shape[d] {
                break;
            }
            i -= index[d] * a.strides[d];
            j -= index[d] * b.strides[d];
            index[d] = 0;
        }
    }
    out
}
// @end

// @listing operators
/// `&a + &b`, `&a - &b`, `&a * &b` and `&a / &b` for two tensors, and the same four
/// with an `f32` on the right, as in `&a * 2.0`.
macro_rules! operators {
    ($($Op:ident $op:ident),*) => {$(
        impl std::ops::$Op<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $op(self, rhs: &Tensor) -> Tensor {
                Tensor::$op(self, rhs)
            }
        }

        impl std::ops::$Op<f32> for &Tensor {
            type Output = Tensor;
            fn $op(self, rhs: f32) -> Tensor {
                self.map(|x| std::ops::$Op::$op(x, rhs))
            }
        }
    )*};
}

operators!(Add add, Sub sub, Mul mul, Div div);
// @end

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl Tensor {
    // @listing lanes
    /// Apply `f` to every *lane* along axis `dim`: the elements along `dim` for one
    /// position of all the other axes. The results come in row-major order of the
    /// other axes.
    pub(crate) fn lanes<T>(&self, dim: usize, f: impl Fn(&[f32]) -> T) -> Vec<T> {
        assert!(dim < self.ndim(), "reduce: no axis {dim}");
        // Move `dim` to the end; read in order, each lane is then a run of n values.
        let mut order: Vec<usize> = (0..self.ndim()).filter(|&d| d != dim).collect();
        order.push(dim);
        let runs = self.permute(&order).to_vec();
        let n = self.shape[dim];
        let count: usize = self.reduced_shape(dim, false).iter().product();
        (0..count).map(|k| f(&runs[k * n..(k + 1) * n])).collect()
    }

    /// The shape after reducing axis `dim`: the axis is removed, or kept with size 1.
    pub(crate) fn reduced_shape(&self, dim: usize, keepdim: bool) -> Vec<usize> {
        let mut shape = self.shape.clone();
        if keepdim {
            shape[dim] = 1;
        } else {
            shape.remove(dim);
        }
        shape
    }
    // @end
}

// @listing sum-f64
/// The sum of `xs`, added up in f64. Callers round it to f32 once, at the end.
fn sum_f64(xs: &[f32]) -> f64 {
    xs.iter().map(|&x| x as f64).sum()
}
// @end

impl Tensor {
    /// The sum along axis `dim`. With `keepdim`, the axis stays, with size 1.
    pub fn sum(&self, dim: usize, keepdim: bool) -> Tensor {
        // @solution ch04 sum
        let data = self.lanes(dim, |lane| sum_f64(lane) as f32);
        Tensor::from_vec(data, &self.reduced_shape(dim, keepdim))
        // @end
    }

    /// The mean along axis `dim`. With `keepdim`, the axis stays, with size 1.
    pub fn mean(&self, dim: usize, keepdim: bool) -> Tensor {
        // @solution ch04 mean
        let mean = |lane: &[f32]| (sum_f64(lane) / lane.len() as f64) as f32;
        let data = self.lanes(dim, mean);
        Tensor::from_vec(data, &self.reduced_shape(dim, keepdim))
        // @end
    }

    /// The largest value along axis `dim`. With `keepdim`, the axis stays.
    pub fn max(&self, dim: usize, keepdim: bool) -> Tensor {
        // @solution ch04 max
        let max = |lane: &[f32]| {
            let (&first, rest) = lane.split_first().expect("max: an empty axis");
            // `x > m` alone would skip a NaN; `is_nan` makes one NaN win.
            rest.iter()
                .fold(first, |m, &x| if x > m || x.is_nan() { x } else { m })
        };
        let data = self.lanes(dim, max);
        Tensor::from_vec(data, &self.reduced_shape(dim, keepdim))
        // @end
    }

    /// For each lane along `dim`, the position of its largest value; the first one if
    /// several are equal. In row-major order of the shape with `dim` removed.
    pub fn argmax(&self, dim: usize) -> Vec<usize> {
        // @solution ch04 argmax
        self.lanes(dim, |lane| {
            assert!(!lane.is_empty(), "argmax: an empty axis");
            // Strictly greater: on a tie, the earlier position stays.
            (1..lane.len())
                .fold(0, |best, k| if lane[k] > lane[best] { k } else { best })
        })
        // @end
    }

    /// The sum of every element.
    pub fn sum_all(&self) -> f32 {
        // @solution ch04 sum_all
        match self.as_slice() {
            Some(run) => sum_f64(run) as f32,
            None => sum_f64(&self.to_vec()) as f32,
        }
        // @end
    }

    /// The value of a tensor with exactly one element, whatever its shape.
    pub fn item(&self) -> f32 {
        // @solution ch04 item
        assert_eq!(
            self.numel(),
            1,
            "item: a tensor of shape {:?} has {} elements, not 1",
            self.shape,
            self.numel()
        );
        self.storage[self.offset]
        // @end
    }

    /// The sum that undoes `expand`: add `self` up along every axis that expanding a
    /// tensor of `shape` to `self`'s shape would add or stretch.
    pub fn sum_to(&self, shape: &[usize]) -> Tensor {
        // @solution ch04 sum_to
        assert_eq!(
            broadcast_shapes(shape, &self.shape).as_deref(),
            Some(&self.shape[..]),
            "sum_to: {:?} does not expand to {:?}",
            shape,
            self.shape
        );
        let mut t = self.clone();
        // Axes added on the left: sum them away.
        for _ in shape.len()..self.ndim() {
            t = t.sum(0, false);
        }
        // Axes stretched from 1: sum them, keeping the axis.
        for (d, &n) in shape.iter().enumerate() {
            if n == 1 && t.shape[d] != 1 {
                t = t.sum(d, true);
            }
        }
        t
        // @end
    }
}
