//! Chapter 10: gradient checking. `gradcheck` compares the tape's gradients of a
//! function with central differences, element by element, under a tolerance that
//! has an absolute and a relative part, and reports the worst element.

use crate::autograd::tape::{Tape, Var};
use crate::tensor::Tensor;

// @listing tolerance
/// How `gradcheck` compares: a central-difference step of `eps` (times `|x|` when
/// that is larger than 1), and an allowed difference of `atol + rtol · |numeric|`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    pub eps: f32,
    pub atol: f32,
    pub rtol: f32,
}

impl Default for Tolerance {
    /// For `f32`: a step near the cube root of machine epsilon (Section 10.4).
    fn default() -> Tolerance {
        Tolerance {
            eps: 5e-3,
            atol: 1e-3,
            rtol: 1e-2,
        }
    }
}
// @end

// @listing ratio
impl Tolerance {
    /// `|analytic - numeric|` as a multiple of what is allowed: at most 1 passes.
    pub fn ratio(&self, analytic: f32, numeric: f32) -> f32 {
        let diff = (analytic - numeric).abs();
        if diff == 0.0 {
            return 0.0; // equal is close enough, even with no tolerance at all
        }
        diff / (self.atol + self.rtol * numeric.abs())
    }
}
// @end

// @listing worst
/// The element of a gradient check that came closest to failing, or failed most.
#[derive(Clone, Debug, PartialEq)]
pub struct WorstElement {
    /// Which input.
    pub input: usize,
    /// Where in it.
    pub index: Vec<usize>,
    /// The tape's gradient.
    pub analytic: f32,
    /// The central-difference estimate.
    pub numeric: f32,
    /// `|analytic - numeric|` over the allowed difference: at most 1 passes.
    pub ratio: f32,
}
// @end

// @listing var-fn
/// A function from `Var`s to a `Var` that works on any tape: the bound every
/// function here needs, under one name.
pub trait VarFn: for<'t> Fn(&[Var<'t>]) -> Var<'t> {}

impl<F> VarFn for F where F: for<'t> Fn(&[Var<'t>]) -> Var<'t> {}
// @end

// @listing weights
/// The weights of the checked loss `L = Σ w · f(inputs)`: 1, 2, 3, … in the
/// output's shape, so that a gradient in the wrong place shows (Section 9.8).
fn weights(shape: &[usize]) -> Tensor {
    Tensor::arange(shape).add_scalar(1.0)
}
// @end

// @listing loss-at
/// `L` with element `j` of input `i` set to `x`, every input a constant, and the
/// weighted sum added up in `f64`.
fn loss_at(f: &impl VarFn, inputs: &[Tensor], (i, j): (usize, usize), x: f32) -> f64 {
    let mut data = inputs[i].to_vec();
    data[j] = x;
    let tape = Tape::new();
    let mut vars: Vec<_> = inputs.iter().map(|v| tape.constant(v.clone())).collect();
    vars[i] = tape.constant(Tensor::from_vec(data, inputs[i].shape()));
    let y = f(&vars).value();
    let terms = y.to_vec().into_iter().zip(weights(y.shape()).to_vec());
    terms.map(|(y, w)| y as f64 * w as f64).sum()
}
// @end

// @listing numeric
/// The central-difference slope of `L` in element `j` of input `i`.
fn numeric(f: &impl VarFn, inputs: &[Tensor], at: (usize, usize), eps: f32) -> f32 {
    let x = inputs[at.0].to_vec()[at.1];
    let h = eps * x.abs().max(1.0); // many ulps of x, even at x = 1e9
    let (up, down) = (x + h, x - h);
    let rise = loss_at(f, inputs, at, up) - loss_at(f, inputs, at, down);
    (rise / (up as f64 - down as f64)) as f32 // the step as rounded, not 2h
}
// @end

// @listing analytic
/// The tape's gradients of `L`, one per input: zeros for an input `L` ignores.
fn analytic(f: &impl VarFn, inputs: &[Tensor]) -> Vec<Tensor> {
    let tape = Tape::new();
    let vars: Vec<Var<'_>> = inputs.iter().map(|x| tape.leaf(x.clone())).collect();
    let y = f(&vars);
    let grads = tape.backward_from(y.id, weights(&y.shape()));
    let zeros = |x: &Tensor| Tensor::zeros(x.shape());
    let grad = |v: &Var, x| grads.get(*v).cloned().unwrap_or_else(|| zeros(x));
    vars.iter().zip(inputs).map(|(v, x)| grad(v, x)).collect()
}
// @end

// @listing gradcheck
/// Compare the tape's gradient of `L = Σ w · f(inputs)` with central differences
/// in every element of every input. `Ok` if all are within tolerance; either way,
/// the element with the largest ratio. Panics if the inputs have no elements.
pub fn gradcheck(
    f: impl VarFn,
    inputs: &[Tensor],
    tol: Tolerance,
) -> Result<WorstElement, WorstElement> {
    let mut worst: Option<WorstElement> = None;
    for (i, g) in analytic(&f, inputs).into_iter().enumerate() {
        for (j, a) in g.to_vec().into_iter().enumerate() {
            let n = numeric(&f, inputs, (i, j), tol.eps);
            let ratio = tol.ratio(a, n);
            // A NaN never compares greater, so it is made the worst by hand.
            let worse = |w: &WorstElement| ratio > w.ratio || ratio.is_nan();
            if worst.as_ref().is_none_or(worse) {
                let index = unravel(j, inputs[i].shape());
                let (input, analytic, numeric) = (i, a, n);
                worst = Some(WorstElement {
                    input,
                    index,
                    analytic,
                    numeric,
                    ratio,
                });
            }
        }
    }
    let worst = worst.expect("gradcheck: the inputs have no elements to check");
    if worst.ratio <= 1.0 {
        Ok(worst)
    } else {
        Err(worst)
    }
}
// @end

// @listing unravel
/// The index of the element at row-major position `flat` in `shape`.
fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for d in (0..shape.len()).rev() {
        index[d] = flat % shape[d];
        flat /= shape[d];
    }
    index
}
// @end

#[cfg(test)]
mod tests {
    use super::{Tolerance, gradcheck};
    use crate::autograd::tape::Var;
    use crate::tensor::Tensor;

    // @listing wrong-sign
    /// 1/x with the sign of its derivative wrong: +1/x², where -1/x² is right.
    fn wrong_sign<'t>(v: &[Var<'t>]) -> Var<'t> {
        let f = |x: &Tensor| x.powf(-1.0);
        v[0].unary("reciprocal", f, |x, _, g| g.div(&x.mul(x)))
    }
    // @end

    #[test]
    fn a_tolerance_relative_to_the_gradient() {
        // @listing wrong-sign-example
        let xs = [Tensor::from_vec(vec![1e9], &[1])];
        let missed = gradcheck(wrong_sign, &xs, Tolerance::default());
        let w = missed.unwrap(); // passes: a difference of 2e-18 is below atol
        assert_eq!(w.analytic, 1e-18);
        assert!((w.numeric + 1e-18).abs() < 1e-22); // -1e-18: right, opposite sign
        let relative = Tolerance {
            atol: 0.0,
            ..Tolerance::default()
        };
        let caught = gradcheck(wrong_sign, &xs, relative).unwrap_err();
        assert!((caught.ratio - 200.0).abs() < 0.1); // 2e-18 / (0.01 · 1e-18)
        assert!(gradcheck(|v| v[0].powf(-1.0), &xs, relative).is_ok());
        // @end
        assert_eq!(1e9_f32 + 5e-3, 1e9); // a nudge of 5e-3 is no nudge at 1e9
        assert_eq!(1e9_f32 + 5e6, 1.005e9); // eps · |x| is
        assert_eq!(f32::from_bits(1e9_f32.to_bits() + 1) - 1e9, 64.0); // spacing
    }

    #[test]
    #[should_panic(expected = "gradcheck: the inputs have no elements to check")]
    fn nothing_to_check_is_not_a_pass() {
        let _ =
            gradcheck(|v| v[0].exp(), &[Tensor::zeros(&[0])], Tolerance::default());
    }

    #[test]
    fn an_empty_input_beside_a_real_one() {
        let xs = [Tensor::zeros(&[0]), Tensor::from_vec(vec![0.5], &[1])];
        let w = gradcheck(|v| v[1].exp(), &xs, Tolerance::default()).unwrap();
        assert_eq!((w.input, w.index), (1, vec![0]));
    }

    /// What the default tolerance lets through (Section 10.7): a tanh rule that
    /// is 1% too large passes, one 2% too large fails, on Section 10.7's `x`.
    #[test]
    fn the_default_passes_a_rule_one_percent_off() {
        let frac = |k: usize| (k as f32 * 0.618_034).fract();
        let x = (1..=24).map(|k| -2.0 + 4.0 * frac(k)).collect();
        let x = [Tensor::from_vec(x, &[2, 3, 4])];
        fn tanh_times<'t>(v: &[Var<'t>], s: f32) -> Var<'t> {
            let rule = move |_: &Tensor, y: &Tensor, g: &Tensor| {
                g.mul(&y.mul(y).neg().add_scalar(1.0)).mul_scalar(s)
            };
            v[0].unary("tanh", Tensor::tanh, rule)
        }
        let ratio = |s: f32| match gradcheck(
            |v| tanh_times(v, s),
            &x,
            Tolerance::default(),
        ) {
            Ok(w) | Err(w) => w.ratio,
        };
        assert!((ratio(1.005) - 0.498).abs() < 1e-3);
        assert!((ratio(1.01) - 0.995).abs() < 1e-3); // passes
        assert!((ratio(1.02) - 1.99).abs() < 1e-2); // fails
    }
}
