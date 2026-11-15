//! Chapter 9 challenges.

use crate::autograd::tape::Var;
use crate::tensor::Tensor;

impl<'t> Var<'t> {
    // @listing squeeze
    /// A new axis of size 1 at position `dim`.
    pub fn unsqueeze(self, dim: usize) -> Var<'t> {
        let mut shape = self.shape();
        assert!(dim <= shape.len(), "unsqueeze: axis {dim} of {shape:?}");
        shape.insert(dim, 1);
        self.reshape(&shape)
    }

    /// Remove axis `dim`, which must have size 1.
    pub fn squeeze(self, dim: usize) -> Var<'t> {
        let mut shape = self.shape();
        assert_eq!(shape[dim], 1, "squeeze: axis {dim} of {shape:?}");
        shape.remove(dim);
        self.reshape(&shape)
    }

    /// All the elements as one axis, in row-major order.
    pub fn flatten(self) -> Var<'t> {
        let n = self.shape().iter().product();
        self.reshape(&[n])
    }
    // @end

    // @listing var
    /// The variance along `dim`: squared deviations from the mean, summed and
    /// divided by `n - correction`. Five nodes, each with its own rule.
    ///
    /// # Panics
    /// If `correction` is not smaller than the length `n` of axis `dim`.
    pub fn var(self, dim: usize, keepdim: bool, correction: usize) -> Var<'t> {
        let n = self.shape()[dim];
        assert!(
            correction < n,
            "var: correction {correction} for {n} elements"
        );
        let d = self - self.mean(dim, true);
        (d * d).sum(dim, keepdim) / (n - correction) as f32
    }
    // @end

    // @listing cat
    /// The tensors in `vars` joined along `dim`, as one node with one parent per
    /// tensor. Its rule cuts the gradient back into the pieces.
    pub fn cat(vars: &[Var<'t>], dim: usize) -> Var<'t> {
        assert!(!vars.is_empty(), "cat: no Vars to join");
        let tape = vars[0].tape;
        let same = vars.iter().all(|v| std::ptr::eq(v.tape, tape));
        assert!(same, "cat: Vars of two tapes");
        let values: Vec<Tensor> = vars.iter().map(|v| v.value()).collect();
        let sizes: Vec<usize> = values.iter().map(|x| x.shape()[dim]).collect();
        let y = Tensor::cat(&values, dim);
        let parents = vars.iter().map(|v| v.id).collect();
        tape.push("cat", y, parents, move |g| {
            let (mut pieces, mut start) = (Vec::new(), 0);
            for &n in &sizes {
                pieces.push(g.slice(dim, start, start + n, 1));
                start += n;
            }
            pieces
        })
    }
    // @end

    // @listing repeat
    /// Challenge 3.4's `repeat`, recorded. Its backward undoes each step of the
    /// forward: reshape, sum over the copies (`expand`'s rule), reshape.
    pub fn repeat(self, counts: &[usize]) -> Var<'t> {
        let ndim = self.shape().len();
        assert!(counts.len() >= ndim, "repeat: too few counts");
        let lead = counts.len() - ndim;
        let mut shape = vec![1; lead];
        shape.extend(self.shape());
        let split: Vec<usize> = shape.iter().flat_map(|&n| [1, n]).collect();
        let tiled: Vec<usize> = (shape.iter().zip(counts))
            .flat_map(|(&n, &c)| [c, n])
            .collect();
        let back = move |g: &Tensor| g.reshape(&tiled).sum_to(&split);
        let f = |x: &Tensor| x.repeat(counts);
        self.unary("repeat", f, move |x, _, g| back(g).reshape(x.shape()))
    }
    // @end

    // @listing logsumexp
    /// `ln Σ e^x` along `dim`, computed after subtracting the lane's maximum. Its
    /// derivative is the softmax of the lane, which the rule recomputes from `x`.
    pub fn logsumexp(self, dim: usize, keepdim: bool) -> Var<'t> {
        let f = move |x: &Tensor| {
            let (m, s) = shifted_sum(x, dim);
            s.ln().add(&m).reshape(&x.reduced_shape(dim, keepdim))
        };
        self.unary("logsumexp", f, move |x, _, g| {
            let (m, s) = shifted_sum(x, dim);
            let softmax = x.sub(&m).exp().div(&s);
            softmax.mul(&g.reshape(&x.reduced_shape(dim, true)))
        })
    }
    // @end

    // @listing as-view
    /// Any view `f` of `self`, with one backward rule for all of them: add each
    /// gradient element into the slot its element was read from.
    pub fn as_view(
        self,
        name: &'static str,
        f: impl Fn(&Tensor) -> Tensor + Copy + 'static,
    ) -> Var<'t> {
        let view = move |x: &Tensor| {
            let y = f(x);
            assert!(y.shares_storage(x), "{name}: not a view"); // when recorded
            y
        };
        self.unary(name, view, move |x, _, g| {
            let ruler = Tensor::zeros(x.shape());
            Tensor::scatter_add(x.shape(), &f(&ruler).slots(), g)
        })
    }
    // @end
}

// @listing lse
/// Each lane's maximum `m` and `Σ e^(x - m)`, keeping the axis. The largest term
/// of each sum is `e^0 = 1`, so nothing overflows. A lane of all `-inf` gives NaN
/// (`-inf - -inf`); Chapter 13's `Tensor::logsumexp` handles that case.
fn shifted_sum(x: &Tensor, dim: usize) -> (Tensor, Tensor) {
    let m = x.max(dim, true);
    let s = x.sub(&m).exp().sum(dim, true);
    (m, s)
}
// @end

impl Tensor {
    // @listing tensor-cat
    /// The tensors joined along `dim`; every other axis must match.
    pub fn cat(xs: &[Tensor], dim: usize) -> Tensor {
        let mut shape = xs[0].shape().to_vec();
        shape[dim] = xs.iter().map(|x| x.shape()[dim]).sum();
        let ruler = Tensor::zeros(&shape);
        let (mut at, mut start) = (Vec::new(), 0);
        for x in xs {
            let n = x.shape()[dim];
            let part = ruler.slice(dim, start, start + n, 1);
            assert_eq!(part.shape(), x.shape(), "cat: shapes along {dim}");
            at.extend(part.slots());
            start += n;
        }
        let values: Vec<f32> = xs.iter().flat_map(|x| x.to_vec()).collect();
        let n = values.len();
        Tensor::scatter_add(&shape, &at, &Tensor::from_vec(values, &[n]))
    }
    // @end
}
