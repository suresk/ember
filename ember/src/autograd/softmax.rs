//! Chapter 13: log-sum-exp, softmax and log-softmax, on tensors and on the tape. Every
//! one of them subtracts each lane's largest element before it exponentiates, so no
//! exponential is ever larger than 1.

use crate::autograd::tape::Var;
use crate::math;
use crate::tensor::Tensor;

// @listing lse-parts
/// The two parts of `ln Σ e^x` for one lane: its largest element `m`, and
/// `ln Σ e^(x - m)`, whose largest term is `e^0 = 1`. The sum is added up in f64.
fn lse_lane(lane: &[f32]) -> (f32, f32) {
    let m = lane.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
    if !m.is_finite() {
        return (m, 0.0); // every element -∞ (ln 0), or one is +∞
    }
    let s: f64 = lane.iter().map(|&x| math::exp(x - m) as f64).sum();
    (m, math::ln64(s) as f32)
}

/// Each lane's `m` and `ln Σ e^(x - m)` along `dim`, with the axis kept (size 1).
fn lse_parts(x: &Tensor, dim: usize) -> (Tensor, Tensor) {
    let (m, ln_s): (Vec<f32>, Vec<f32>) = x.lanes(dim, lse_lane).into_iter().unzip();
    let shape = x.reduced_shape(dim, true);
    (Tensor::from_vec(m, &shape), Tensor::from_vec(ln_s, &shape))
}
// @end

impl Tensor {
    // @listing tensor-lse
    /// `ln Σ e^x` along `dim`, as `m + ln Σ e^(x - m)`: it never overflows.
    pub fn logsumexp(&self, dim: usize, keepdim: bool) -> Tensor {
        let (m, ln_s) = lse_parts(self, dim);
        m.add(&ln_s).reshape(&self.reduced_shape(dim, keepdim))
    }
    // @end

    // @listing tensor-softmax
    /// `x - logsumexp(x)` along `dim`, computed as `(x - m) - ln Σ e^(x - m)`.
    pub fn log_softmax(&self, dim: usize) -> Tensor {
        let (m, ln_s) = lse_parts(self, dim);
        self.sub(&m).sub(&ln_s)
    }

    /// `e^x / Σ e^x` along `dim`: the exponential of the log-softmax, which is at
    /// most 0, so no exponential here is larger than 1 either.
    pub fn softmax(&self, dim: usize) -> Tensor {
        self.log_softmax(dim).exp()
    }
    // @end
}

impl<'t> Var<'t> {
    // @listing var-softmax
    /// Softmax along `dim`, one node. Backward is the Jacobian-vector product
    /// `y ⊙ (g - Σ g ⊙ y)`, with the sum along `dim`.
    pub fn softmax(self, dim: usize) -> Var<'t> {
        let f = move |x: &Tensor| x.softmax(dim);
        self.unary("softmax", f, move |_, y, g| {
            let dot = g.mul(y).sum(dim, true); // one number per lane
            y.mul(&g.sub(&dot))
        })
    }
    // @end

    // @listing var-log-softmax
    /// Log-softmax along `dim`, one node. Backward is `g - softmax(x) · Σ g`; the
    /// softmax is recomputed from `x`, not read from the rounded output.
    pub fn log_softmax(self, dim: usize) -> Var<'t> {
        let f = move |x: &Tensor| x.log_softmax(dim);
        self.unary("log_softmax", f, move |x, _, g| {
            g.sub(&x.softmax(dim).mul(&g.sum(dim, true)))
        })
    }
    // @end
}
