//! Chapter 4 challenges.

use crate::broadcast::broadcast_shapes;
use crate::tensor::{Tensor, contiguous_strides};

impl Tensor {
    // @listing min
    /// The smallest value along `dim`: the largest of the negated values, negated.
    pub fn min(&self, dim: usize, keepdim: bool) -> Tensor {
        self.neg().max(dim, keepdim).neg()
    }

    /// The position of the smallest value along `dim`, the first one on a tie.
    pub fn argmin(&self, dim: usize) -> Vec<usize> {
        self.neg().argmax(dim)
    }
    // @end

    // @listing var
    /// The variance along `dim`: the sum of squared deviations from the mean, divided
    /// by `n - correction`. `correction = 1` is Bessel's correction.
    pub fn var(&self, dim: usize, keepdim: bool, correction: usize) -> Tensor {
        let n = self.shape[dim];
        assert!(
            n > correction,
            "var: {n} values with correction {correction}"
        );
        let d = self.sub(&self.mean(dim, true)); // keepdim, so it broadcasts back
        let ss = d.mul(&d).sum(dim, keepdim);
        ss.mul_scalar(1.0 / (n - correction) as f32)
    }
    // @end

    // @listing gt
    /// 1.0 where `self > other` (after broadcasting), 0.0 elsewhere.
    pub fn gt(&self, other: &Tensor) -> Tensor {
        self.zip_map(other, |x, y| if x > y { 1.0 } else { 0.0 })
    }
    // @end

    // @listing where
    /// `a` where `cond` is nonzero and `b` where it is zero, all three broadcast to a
    /// common shape. Named after `torch.where`; `where` is a Rust keyword.
    pub fn where_(cond: &Tensor, a: &Tensor, b: &Tensor) -> Tensor {
        let shape = broadcast_shapes(&cond.shape, &a.shape)
            .and_then(|s| broadcast_shapes(&s, &b.shape))
            .unwrap_or_else(|| {
                panic!(
                    "where_: shapes {:?}, {:?} and {:?} don't broadcast",
                    cond.shape, a.shape, b.shape
                )
            });
        let [c, a, b] = [cond, a, b].map(|t| t.expand(&shape).to_vec());
        let data = (0..c.len())
            .map(|k| if c[k] != 0.0 { a[k] } else { b[k] })
            .collect();
        Tensor::from_vec(data, &shape)
    }
    // @end

    // @listing cumsum
    /// Running sums along `dim`: element k of a lane sums the lane's first k + 1.
    pub fn cumsum(&self, dim: usize) -> Tensor {
        let running = |lane: &[f32]| -> Vec<f32> {
            let mut acc = 0.0f64;
            lane.iter()
                .map(|&x| {
                    acc += x as f64;
                    acc as f32
                })
                .collect()
        };
        let data: Vec<f32> = self.lanes(dim, running).concat();
        // The data is laid out with `dim` moved to the end; move it back.
        let mut order: Vec<usize> = (0..self.ndim()).filter(|&d| d != dim).collect();
        order.push(dim);
        let moved: Vec<usize> = order.iter().map(|&d| self.shape[d]).collect();
        let mut back = vec![0; self.ndim()];
        for (k, &d) in order.iter().enumerate() {
            back[d] = k; // axis d of self is axis k of the moved tensor
        }
        Tensor::from_vec(data, &moved).permute(&back).contiguous()
    }
    // @end

    // @listing softplus
    /// `ln(1 + e^x)`, computed as `max(x, 0) + ln(1 + e^(-|x|))` so that e^x never
    /// overflows.
    pub fn softplus(&self) -> Tensor {
        self.map(|x| x.max(0.0) + (-x.abs()).exp().ln_1p())
    }
    // @end
}

impl Tensor {
    // @listing sum-strided
    /// `sum(dim, keepdim)` without moving the axis or copying the input: add each
    /// element into the slot of an output view that has stride 0 along `dim`.
    pub fn sum_strided(&self, dim: usize, keepdim: bool) -> Tensor {
        assert!(dim < self.ndim(), "sum_strided: no axis {dim}");
        let mut kept = self.shape.clone();
        kept[dim] = 1;
        let mut out_strides = contiguous_strides(&kept);
        out_strides[dim] = 0; // every position along dim lands in one slot
        let mut acc = vec![0.0f64; kept.iter().product()];
        self.add_into(&out_strides, &mut acc);
        let data = acc.into_iter().map(|x| x as f32).collect();
        Tensor::from_vec(data, &self.reduced_shape(dim, keepdim))
    }
    // @end

    // @listing add-into
    /// Walk `self` in row-major order and add each element into `acc[j]`, where `j`
    /// moves by `out_strides` beside the input's own storage position.
    fn add_into(&self, out_strides: &[usize], acc: &mut [f64]) {
        let mut index = vec![0; self.ndim()];
        let (mut i, mut j) = (self.offset, 0);
        for _ in 0..self.numel() {
            acc[j] += self.storage[i] as f64;
            for d in (0..self.ndim()).rev() {
                index[d] += 1;
                i += self.strides[d];
                j += out_strides[d];
                if index[d] < self.shape[d] {
                    break;
                }
                i -= index[d] * self.strides[d];
                j -= index[d] * out_strides[d];
                index[d] = 0;
            }
        }
    }
    // @end
}
