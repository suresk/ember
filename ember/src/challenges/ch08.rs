//! Chapter 8 challenges.

use crate::autograd::tape::{Tape, Var};
use crate::math;
use crate::tensor::Tensor;

impl<'t> Var<'t> {
    // @listing abs
    /// `|x|`, with derivative -1, 0 or 1: the sign of x, and 0 at x = 0.
    pub fn abs(self) -> Var<'t> {
        let d = |g: f32, x: f32| if x == 0.0 { 0.0 } else { g * x.signum() };
        self.unary("abs", |x| x.map(f32::abs), move |x, _, g| g.zip_map(x, d))
    }
    // @end

    // @listing leaky-clamp
    /// `x` for positive x, else `slope · x`.
    pub fn leaky_relu(self, slope: f32) -> Var<'t> {
        let f = move |x: f32| if x > 0.0 { x } else { slope * x };
        let d = move |g: f32, x: f32| if x > 0.0 { g } else { slope * g };
        self.unary("leaky_relu", |x| x.map(f), move |x, _, g| g.zip_map(x, d))
    }

    /// `x` limited to `[min, max]`. The gradient passes where `min ≤ x ≤ max`,
    /// the ends included, as in PyTorch.
    pub fn clamp(self, min: f32, max: f32) -> Var<'t> {
        let f = move |x: f32| x.clamp(min, max);
        let d = move |g: f32, x: f32| if min <= x && x <= max { g } else { 0.0 };
        self.unary("clamp", |x| x.map(f), move |x, _, g| g.zip_map(x, d))
    }
    // @end

    // @listing softplus
    /// `ln(1 + e^x)`, with Chapter 4's stable forward. The derivative is the
    /// sigmoid of the input.
    pub fn softplus(self) -> Var<'t> {
        self.unary("softplus", Tensor::softplus, |x, _, g| g.mul(&x.sigmoid()))
    }
    // @end

    // @listing custom
    /// A user-defined operation: forward function `f`, and `rule(x, y, g)`, the
    /// gradient with respect to `x` from the input, the output and the gradient
    /// with respect to the output.
    pub fn custom(
        self,
        name: &'static str,
        f: impl Fn(&Tensor) -> Tensor,
        rule: impl Fn(&Tensor, &Tensor, &Tensor) -> Tensor + 'static,
    ) -> Var<'t> {
        self.unary(name, f, rule)
    }
    // @end

    // @listing gelu
    /// GELU, in the tanh approximation `x/2 · (1 + tanh(k(x + 0.044715x³)))`.
    pub fn gelu(self) -> Var<'t> {
        let f = |x: &Tensor| x.map(|x| gelu_parts(x).0);
        let d = |g: f32, x: f32| g * gelu_parts(x).1;
        self.unary("gelu", f, move |x, _, g| g.zip_map(x, d))
    }
    // @end
}

// @listing gelu-parts
/// GELU's value at `x` and its derivative there.
fn gelu_parts(x: f32) -> (f32, f32) {
    const K: f32 = 0.797_884_6; // √(2/π)
    const C: f32 = 0.044_715;
    let t = math::tanh(K * (x + C * x * x * x));
    let du = K * (1.0 + 3.0 * C * x * x); // d/dx of the argument of tanh
    let y = 0.5 * x * (1.0 + t);
    let dy = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du; // product rule
    (y, dy)
}
// @end

// @listing nudge-grad
/// For an elementwise `f`, the slope of `sum(f(x))` in each element of `x`, by
/// central differences with step `h`: an estimate of `f'` at each element.
pub fn nudge_grad(
    f: impl for<'t> Fn(Var<'t>) -> Var<'t>,
    x: &Tensor,
    h: f32,
) -> Vec<f32> {
    let total = |xs: Vec<f32>| {
        let tape = Tape::new();
        let v = tape.constant(Tensor::from_vec(xs, x.shape()));
        f(v).value().sum_all()
    };
    let at = x.to_vec();
    let nudged = |i: usize, by: f32| {
        let mut xs = at.clone();
        xs[i] += by;
        total(xs)
    };
    let slope = |i| (nudged(i, h) - nudged(i, -h)) / (2.0 * h);
    (0..at.len()).map(slope).collect()
}
// @end
