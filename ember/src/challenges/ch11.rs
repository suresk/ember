//! Chapter 11 challenges.

use crate::autograd::tape::{Tape, Var};
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

// @listing zeros
/// `zero_grad` as PyTorch did it before 2.0: every gradient a tensor of zeros.
pub fn zero_grad_to_zeros(model: &mut dyn Module) {
    for p in model.parameters() {
        p.grad = Some(Tensor::zeros(p.value.shape()));
    }
}
// @end

// @listing named
/// A module whose parameters have names, with dots for children: `hidden.w`.
pub trait NamedModule: Module {
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)>;
}

/// A child's names, each with `prefix.` in front.
pub fn prefixed<'a>(
    prefix: &str,
    named: Vec<(String, &'a mut Parameter)>,
) -> Vec<(String, &'a mut Parameter)> {
    let dotted = |(name, p)| (format!("{prefix}.{name}"), p);
    named.into_iter().map(dotted).collect()
}
// @end

// @listing micro
/// The gradient of the mean squared error over all the rows of `x` and `y`,
/// collected in micro-batches of the given sizes.
pub fn accumulate_mse(m: &mut dyn Module, x: &Tensor, y: &Tensor, sizes: &[usize]) {
    let (n, mut start) = (x.shape()[0], 0);
    assert_eq!(sizes.iter().sum::<usize>(), n, "accumulate_mse: sizes");
    for &size in sizes {
        let tape = Tape::new();
        let rows = |t: &Tensor| tape.constant(t.slice(0, start, start + size, 1));
        let loss = (m.forward(rows(x)) - rows(y)).powf(2.0).mean_all();
        let share = size as f32 / n as f32; // this micro-batch's part of the mean
        (loss * share).backward_into(&mut m.parameters());
        start += size;
    }
}
// @end

// @listing frozen
/// A module whose parameters are not trained: it hands none of them over.
pub struct Frozen<M: Module>(pub M);

impl<M: Module> Module for Frozen<M> {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        self.0.forward(x)
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        Vec::new() // backward_into finds no one to give these gradients to
    }
    fn train(&mut self, on: bool) {
        self.0.train(on) // Chapter 21: frozen is not the same as evaluating
    }
}
// @end

// @listing residual
/// `x + m(x)`: the module's output added to its input, of the same shape.
pub struct Residual<M: Module>(pub M);
impl<M: Module> Module for Residual<M> {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        x + self.0.forward(x)
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        self.0.parameters()
    }
    fn train(&mut self, on: bool) {
        self.0.train(on) // Chapter 21
    }
}
// @end

// @listing tied
/// Encode with `w`, decode with its transpose: `x·w·wᵀ`, one matrix read twice.
pub struct Tied {
    pub w: Parameter,
    pub one_node: bool, // put w on the tape once, or once per use
}

impl Module for Tied {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let tape = x.tape();
        let w = tape.param(&self.w);
        let w2 = if self.one_node {
            w
        } else {
            tape.param(&self.w)
        };
        x.matmul(w).matmul(w2.transpose(0, 1))
    }

    fn parameters(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.w]
    }
}
// @end
