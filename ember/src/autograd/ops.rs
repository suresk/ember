//! Chapter 8: backward rules for the elementwise operations. Each operation is a
//! forward function from Chapter 4 and a rule, recorded with `unary` or `binary`;
//! the operators `+ - * /` on `Var` call them.

use crate::autograd::tape::Var;
use crate::math;
use crate::tensor::Tensor;

// Methods named like the `std::ops` traits, which the impls below call.
#[allow(clippy::should_implement_trait)]
impl<'t> Var<'t> {
    // @listing sub-neg
    pub fn sub(self, other: Var<'t>) -> Var<'t> {
        self.binary(other, "sub", Tensor::sub, |_, _, g| (g.clone(), g.neg()))
    }

    pub fn neg(self) -> Var<'t> {
        self.unary("neg", Tensor::neg, |_, _, g| g.neg())
    }
    // @end

    // @listing scalars
    /// `x + s` for a plain number `s`: the derivative is 1, as for `add`.
    pub fn add_scalar(self, s: f32) -> Var<'t> {
        self.unary("add_scalar", |x| x.add_scalar(s), |_, _, g| g.clone())
    }

    /// `x · s`: the derivative is `s`.
    pub fn mul_scalar(self, s: f32) -> Var<'t> {
        self.unary("mul_scalar", |x| x.mul_scalar(s), move |_, _, g| g * s)
    }

    /// `x / s`: the derivative is `1/s`.
    pub fn div_scalar(self, s: f32) -> Var<'t> {
        self.unary("div_scalar", |x| x / s, move |_, _, g| g / s)
    }
    // @end

    // @listing div
    /// `a / b`, with derivatives `1/b` with respect to `a` and `-a/b²` with respect
    /// to `b`.
    pub fn div(self, other: Var<'t>) -> Var<'t> {
        self.binary(other, "div", Tensor::div, |a, b, g| {
            let ga = g.div(b);
            let gb = g.mul(a).div(&b.mul(b)).neg();
            (ga, gb)
        })
    }
    // @end

    // @listing exp-ln
    /// `e^x`, whose derivative is `e^x` again: the output `y`.
    pub fn exp(self) -> Var<'t> {
        self.unary("exp", Tensor::exp, |_, y, g| g.mul(y))
    }

    /// The natural log, whose derivative `1/x` needs the input.
    pub fn ln(self) -> Var<'t> {
        self.unary("ln", Tensor::ln, |x, _, g| g.div(x))
    }
    // @end

    // @listing activations
    pub fn tanh(self) -> Var<'t> {
        let d = |g: f32, y: f32| g * (1.0 - y * y); // g times dy/dx
        self.unary("tanh", Tensor::tanh, move |_, y, g| g.zip_map(y, d))
    }

    /// One node, where Chapter 6's composite sigmoid made seven.
    pub fn sigmoid(self) -> Var<'t> {
        let d = |g: f32, y: f32| g * y * (1.0 - y);
        self.unary("sigmoid", Tensor::sigmoid, move |_, y, g| g.zip_map(y, d))
    }

    /// The gradient passes where the output is positive: not at x = 0.
    pub fn relu(self) -> Var<'t> {
        let d = |g: f32, y: f32| if y > 0.0 { g } else { 0.0 };
        self.unary("relu", Tensor::relu, move |_, y, g| g.zip_map(y, d))
    }
    // @end

    // @listing powf
    /// `x^p` for a fixed exponent `p`, with derivative `p·x^(p-1)`.
    pub fn powf(self, p: f32) -> Var<'t> {
        let d = move |g: f32, x: f32| g * p * math::powf(x, p - 1.0);
        self.unary("powf", move |x| x.powf(p), move |x, _, g| g.zip_map(x, d))
    }
    // @end
}

// @listing operators
/// `impl std::ops::$Op<$Rhs> for $Lhs`, whose method computes `$body` from the two
/// operands, named `$a` and `$b`.
macro_rules! operator {
    ($Op:ident $op:ident, $Lhs:ty, $Rhs:ty, |$a:ident, $b:ident| $body:expr) => {
        impl<'t> std::ops::$Op<$Rhs> for $Lhs {
            type Output = Var<'t>;
            fn $op(self, rhs: $Rhs) -> Var<'t> {
                let ($a, $b) = (self, rhs);
                $body
            }
        }
    };
}

operator!(Add add, Var<'t>, Var<'t>, |a, b| Var::add(a, b));
operator!(Sub sub, Var<'t>, Var<'t>, |a, b| Var::sub(a, b));
operator!(Mul mul, Var<'t>, Var<'t>, |a, b| Var::mul(a, b));
operator!(Div div, Var<'t>, Var<'t>, |a, b| Var::div(a, b));
operator!(Add add, Var<'t>, f32, |x, s| x.add_scalar(s));
operator!(Sub sub, Var<'t>, f32, |x, s| x.add_scalar(-s));
operator!(Mul mul, Var<'t>, f32, |x, s| x.mul_scalar(s));
operator!(Div div, Var<'t>, f32, |x, s| x.div_scalar(s));
operator!(Add add, f32, Var<'t>, |s, x| x.add_scalar(s));
operator!(Sub sub, f32, Var<'t>, |s, x| x.neg().add_scalar(s));
operator!(Mul mul, f32, Var<'t>, |s, x| x.mul_scalar(s));
operator!(Div div, f32, Var<'t>, |s, x| x.powf(-1.0).mul_scalar(s));
// @end

impl<'t> std::ops::Neg for Var<'t> {
    type Output = Var<'t>;
    fn neg(self) -> Var<'t> {
        Var::neg(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::autograd::tape::{Tape, Var};
    use crate::tensor::Tensor;

    // @listing overwrite
    /// Section 7.6's walk with one change: a gradient sent to a parent replaces the
    /// parent's gradient instead of adding to it.
    fn backward_overwriting(tape: &Tape, id: usize) -> Vec<Option<Tensor>> {
        let nodes = tape.nodes.borrow();
        let mut grads: Vec<Option<Tensor>> = vec![None; id + 1];
        grads[id] = Some(Tensor::from_vec(vec![1.0], &[]));
        for i in (0..=id).rev() {
            let to_parents = match (&grads[i], &nodes[i].backward) {
                (Some(g), Some(rule)) => rule(g),
                _ => continue,
            };
            for (&p, g) in nodes[i].parents.iter().zip(to_parents) {
                if nodes[p].requires_grad {
                    grads[p] = Some(g); // the mistake: `=`, not `accumulate`
                }
            }
        }
        grads
    }
    // @end

    #[test]
    fn overwriting_loses_contributions() {
        // @listing overwrite-example
        let tape = Tape::new();
        let x = tape.leaf(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]));
        let wrong = |l: Var| {
            let grads = backward_overwriting(&tape, l.id());
            grads[x.id()].as_ref().unwrap().to_vec()
        };
        let cube = (x * x * x).sum_all();
        assert_eq!(cube.backward().get(x).unwrap().to_vec(), [3.0, 12.0, 27.0]);
        assert_eq!(wrong(cube), [1.0, 4.0, 9.0]); // x², a third of 3x²
        let poly = (x * x + x).sum_all();
        assert_eq!(poly.backward().get(x).unwrap().to_vec(), [3.0, 5.0, 7.0]);
        assert_eq!(wrong(poly), [1.0, 2.0, 3.0]); // x, not 2x + 1
        // @end
    }
}
