//! Chapter 12: the fully connected layer, `y = x·W + b`.

use crate::autograd::tape::Var;
use crate::nn::{Module, Parameter, init};
use crate::rng::Rng;
use crate::tensor::Tensor;

// @listing linear
/// A fully connected layer: `y = x·W + b`, with `W` of shape `[in, out]` and `b` of
/// shape `[out]`, applied to the last axis of `x`.
#[derive(Debug)]
pub struct Linear {
    pub weight: Parameter,
    pub bias: Parameter,
}
// @end

// @listing linear-new
impl Linear {
    /// PyTorch's default: weights and biases uniform in ±1/√in.
    pub fn new(fan_in: usize, fan_out: usize, rng: &mut Rng) -> Linear {
        assert!(fan_in > 0, "Linear: a layer needs at least one input");
        let bound = 1.0 / (fan_in as f32).sqrt();
        let weight = init::uniform(&[fan_in, fan_out], bound, rng);
        let bias = init::uniform(&[fan_out], bound, rng);
        Linear::from_tensors(weight, bias)
    }

    /// A layer with the given weight, `[in, out]`, and bias, `[out]`.
    pub fn from_tensors(weight: Tensor, bias: Tensor) -> Linear {
        let (w, b) = (weight.shape(), bias.shape());
        let fits = w.len() == 2 && b == [w[1]];
        assert!(fits, "Linear: weight {w:?} and bias {b:?} don't fit");
        Linear {
            weight: Parameter::new(weight),
            bias: Parameter::new(bias),
        }
    }
}
// @end

// @listing forward
impl Module for Linear {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let mut shape = x.shape();
        let (k, n) = (self.weight.value.shape()[0], self.weight.value.shape()[1]);
        let fits = shape.last() == Some(&k);
        assert!(fits, "Linear: input {shape:?}, weight [{k}, {n}]");
        let rows = x.reshape(&[x.value().numel() / k, k]); // [..., k] to [B, k]
        let tape = x.tape();
        let y = rows.matmul(tape.param(&self.weight)) + tape.param(&self.bias);
        *shape.last_mut().expect("checked above") = n;
        y.reshape(&shape) // [B, n] back to [..., n]
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight, &mut self.bias]
    }
    // @hide
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)> {
        let (w, b) = (&mut self.weight, &mut self.bias); // Chapter 20
        vec![("weight".into(), w), ("bias".into(), b)]
    }
    // @show
}
// @end
