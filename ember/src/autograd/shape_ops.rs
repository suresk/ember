//! Chapter 9: backward rules for the operations that change shapes: `expand`, the
//! reductions `sum`, `mean` and `max`, and the views `reshape`, `transpose`,
//! `permute`, `slice` and `select`. Their rules do no arithmetic beyond adding (and
//! `mean`'s division): they move gradient elements, or add them up.

use crate::autograd::tape::Var;
use crate::tensor::Tensor;

impl<'t> Var<'t> {
    // @listing expand
    /// `self` read as `shape`, without copying (Section 3.6). An element read by
    /// several elements of the output collects all of their gradients.
    pub fn expand(self, shape: &[usize]) -> Var<'t> {
        let f = |x: &Tensor| x.expand(shape);
        self.unary("expand", f, |x, _, g| g.sum_to(x.shape()))
    }
    // @end

    // @listing sum
    /// The sum along axis `dim`; with `keepdim`, the axis stays with size 1.
    pub fn sum(self, dim: usize, keepdim: bool) -> Var<'t> {
        let f = |x: &Tensor| x.sum(dim, keepdim);
        self.unary("sum", f, move |x, _, g| spread(g, x, dim))
    }
    // @end

    // @listing mean
    /// The mean along `dim`: the sum's rule, divided by the number of elements.
    pub fn mean(self, dim: usize, keepdim: bool) -> Var<'t> {
        let f = |x: &Tensor| x.mean(dim, keepdim);
        self.unary("mean", f, move |x, _, g| {
            let n = x.shape()[dim] as f32;
            spread(&(g / n), x, dim) // divide the small tensor, then expand
        })
    }

    /// The mean of every element, as a tensor of shape `[]`.
    pub fn mean_all(self) -> Var<'t> {
        let f = |x: &Tensor| x.reshape(&[x.numel()]).mean(0, false);
        self.unary("mean_all", f, |x, _, g| {
            (g / x.numel() as f32).expand(x.shape())
        })
    }
    // @end

    // @listing max
    /// The largest value along `dim`. The gradient goes to the first largest
    /// element of each lane, the one `argmax` picks.
    pub fn max(self, dim: usize, keepdim: bool) -> Var<'t> {
        let f = |x: &Tensor| x.max(dim, keepdim);
        self.unary("max", f, move |x, _, g| to_argmax(g, x, dim))
    }
    // @end

    // @listing reshape-transpose
    pub fn reshape(self, shape: &[usize]) -> Var<'t> {
        let f = |x: &Tensor| x.reshape(shape);
        self.unary("reshape", f, |x, _, g| g.reshape(x.shape()))
    }

    pub fn transpose(self, a: usize, b: usize) -> Var<'t> {
        let f = |x: &Tensor| x.transpose(a, b);
        self.unary("transpose", f, move |_, _, g| g.transpose(a, b))
    }
    // @end

    // @listing permute

    /// Axis `d` of the result is axis `order[d]` of `self`. Going back, axis `d` of
    /// `self` is the one at the position of `d` in `order`.
    pub fn permute(self, order: &[usize]) -> Var<'t> {
        let at = |d| order.iter().position(|&o| o == d).unwrap_or(d);
        let back: Vec<usize> = (0..order.len()).map(at).collect(); // owned
        let f = |x: &Tensor| x.permute(order);
        self.unary("permute", f, move |_, _, g| g.permute(&back))
    }
    // @end

    // @listing slice-select
    /// Positions `start, start + step, …` below `end` of axis `dim`.
    pub fn slice(self, dim: usize, start: usize, end: usize, step: usize) -> Var<'t> {
        let view = move |x: &Tensor| x.slice(dim, start, end, step);
        self.unary("slice", view, move |x, _, g| unview(x.shape(), view, g))
    }

    /// Position `i` of axis `dim`, with the axis removed.
    pub fn select(self, dim: usize, i: usize) -> Var<'t> {
        let view = move |x: &Tensor| x.select(dim, i);
        self.unary("select", view, move |x, _, g| unview(x.shape(), view, g))
    }
    // @end
}

// @listing spread
/// The gradient of a reduction along `dim`, read at the input's shape: put the axis
/// back with size 1 (whether or not `keepdim` kept it), then expand along it.
fn spread(g: &Tensor, x: &Tensor, dim: usize) -> Tensor {
    g.reshape(&x.reduced_shape(dim, true)).expand(x.shape())
}
// @end

impl Tensor {
    // @listing slots
    /// The storage slot of every element, in row-major order: where `to_vec` reads.
    pub fn slots(&self) -> Vec<usize> {
        let slot = |i: usize| {
            let (mut rest, mut at) = (i, self.offset);
            for d in (0..self.ndim()).rev() {
                at += rest % self.shape[d] * self.strides[d]; // index along d
                rest /= self.shape[d];
            }
            at
        };
        (0..self.numel()).map(slot).collect()
    }
    // @end

    // @listing scatter-add

    /// A tensor of `shape`, all zeros except that element `k` of `src`, in
    /// row-major order, is added in at row-major position `at[k]`.
    pub fn scatter_add(shape: &[usize], at: &[usize], src: &Tensor) -> Tensor {
        let (m, n) = (at.len(), src.numel());
        assert_eq!(m, n, "scatter_add: {m} positions for {n} values");
        let mut data = vec![0.0; shape.iter().product()];
        for (&i, x) in at.iter().zip(src.to_vec()) {
            data[i] += x;
        }
        Tensor::from_vec(data, shape)
    }
    // @end
}

// @listing unview
/// The backward of a view of a tensor of `shape`: zeros, with `g` added in wherever
/// `view` reads. Applied to a fresh row-major tensor, the view's slots are positions.
fn unview(shape: &[usize], view: impl Fn(&Tensor) -> Tensor, g: &Tensor) -> Tensor {
    let ruler = Tensor::zeros(shape);
    Tensor::scatter_add(shape, &view(&ruler).slots(), g)
}
// @end

// @listing to-argmax
/// The gradient of `max` along `dim`: zeros of `x`'s shape, with each lane's
/// gradient at the lane's first largest element.
fn to_argmax(g: &Tensor, x: &Tensor, dim: usize) -> Tensor {
    let ruler = Tensor::zeros(x.shape());
    let starts = ruler.select(dim, 0).slots(); // each lane's first position
    let step = ruler.strides()[dim]; // from one element of a lane to the next
    let at: Vec<usize> = (starts.iter().zip(x.argmax(dim)))
        .map(|(s, k)| s + k * step)
        .collect();
    Tensor::scatter_add(x.shape(), &at, g)
}
// @end

#[cfg(test)]
mod tests {
    use crate::autograd::tape::{Tape, Var};
    use crate::tensor::Tensor;

    // @listing unsummed
    impl<'t> Var<'t> {
        /// `add` without `binary`'s `sum_to`: both operands get the gradient of
        /// the output, at the output's shape.
        fn add_unsummed(self, other: Var<'t>) -> Var<'t> {
            let y = self.value().add(&other.value());
            let parents = vec![self.id, other.id];
            let rule = |g: &Tensor| vec![g.clone(), g.clone()];
            self.tape.push("add", y, parents, rule)
        }
    }
    // @end

    #[test]
    fn forgetting_the_sum() {
        // @listing unsummed-example
        let tape = Tape::new();
        let x = tape.constant(Tensor::arange(&[2, 3]));
        let b = tape.leaf(Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]));
        let w = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let l = (x.add_unsummed(b) * tape.constant(w.clone())).sum_all();
        let gb = l.backward().get(b).unwrap().clone();
        assert_eq!(gb.shape(), [2, 3]); // a gradient of 6 numbers for 3
        assert_eq!(gb.to_vec(), w.to_vec()); // ȳ itself, never added up
        let stepped = b.value().sub(&(&gb * 0.1)); // one step of gradient descent
        assert_eq!(stepped.shape(), [2, 3]); // now a bias per row, and no panic
        // @end
        assert_eq!(stepped.to_vec(), [9.9, 19.8, 29.7, 9.6, 19.5, 29.4]);
        let right = b
            .value()
            .sub(&(&Tensor::from_vec(vec![5.0, 7.0, 9.0], &[3]) * 0.1));
        assert_eq!(right.to_vec(), [9.5, 19.3, 29.1]);
    }
}
