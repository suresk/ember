//! Chapter 10 challenges.

use crate::autograd::gradcheck::VarFn;
use crate::autograd::tape::{Tape, Var};
use crate::tensor::Tensor;

impl<'t> Var<'t> {
    // @listing matvec
    /// `[m, k] @ [k] -> [m]`, as one node. `self` gets the outer product of `g`
    /// and `v`; `v` gets `selfᵀ g`.
    pub fn matvec(self, v: Var<'t>) -> Var<'t> {
        self.binary(v, "matvec", Tensor::matvec, |a, v, g| {
            let (m, k) = (a.shape()[0], a.shape()[1]);
            let ga = g.reshape(&[m, 1]).mul(&v.reshape(&[1, k])); // g vᵀ
            (ga, a.transpose(0, 1).matvec(g))
        })
    }
    // @end

    // @listing matmul-broadcast
    /// `[b, m, k] @ [k, n]`: one matrix shared by a stack, through `expand`.
    pub fn matmul_shared(self, w: Var<'t>) -> Var<'t> {
        let (b, w_shape) = (self.shape()[0], w.shape());
        self.matmul(w.expand(&[b, w_shape[0], w_shape[1]]))
    }

    /// The same product as one `[b·m, k] @ [k, n]`, with no copy of `w`.
    pub fn matmul_folded(self, w: Var<'t>) -> Var<'t> {
        let (s, n) = (self.shape(), w.shape()[1]);
        let rows = self.reshape(&[s[0] * s[1], s[2]]);
        rows.matmul(w).reshape(&[s[0], s[1], n])
    }
    // @end
}

impl Tensor {
    // @listing allclose-rel
    /// Same shape, and every `|a - b| ≤ atol + rtol · |b|`, as in `torch.allclose`.
    pub fn allclose_rel(&self, other: &Tensor, atol: f32, rtol: f32) -> bool {
        let ok = |(a, b): (f32, f32)| (a - b).abs() <= atol + rtol * b.abs();
        let mut pairs = self.to_vec().into_iter().zip(other.to_vec());
        self.shape() == other.shape() && pairs.all(ok)
    }
    // @end
}

// @listing directional
/// One nudge along a direction: `Σ grad · d`, from the tape, and the slope of
/// `L = Σ f(xs + s·d)` in `s` at 0, from two evaluations.
pub fn directional_check(
    f: impl VarFn,
    xs: &[Tensor],
    d: &[Tensor],
    h: f32,
) -> (f32, f32) {
    let tape = Tape::new();
    let vars: Vec<Var> = xs.iter().map(|x| tape.leaf(x.clone())).collect();
    let grads = f(&vars).sum_all().backward();
    let dot =
        |(v, d): (&Var, &Tensor)| grads.get(*v).map_or(0.0, |g| g.mul(d).sum_all());
    let along: f32 = vars.iter().zip(d).map(dot).sum();
    let loss = |s: f32| {
        let tape = Tape::new();
        let at = |(x, d): (&Tensor, &Tensor)| tape.constant(x.add(&d.mul_scalar(s)));
        let y = f(&xs.iter().zip(d).map(at).collect::<Vec<_>>()).value();
        y.to_vec().into_iter().map(f64::from).sum::<f64>()
    };
    let slope = (loss(h) - loss(-h)) / (2.0 * h as f64);
    (along, slope as f32)
}
// @end
