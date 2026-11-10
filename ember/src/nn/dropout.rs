//! Chapter 21: inverted dropout, the first module with a training mode.

use crate::autograd::tape::Var;
use crate::nn::{Module, Parameter};
use crate::rng::Rng;
use crate::tensor::Tensor;
use std::cell::RefCell;

// @listing dropout
/// Inverted dropout: in training, each element is kept with probability
/// `1 - p` and scaled by `1 / (1 - p)`, or zeroed; in evaluation, the identity.
#[derive(Debug)]
pub struct Dropout {
    pub p: f32,
    training: bool,
    rng: RefCell<Rng>, // `forward` takes `&self`, and drawing needs `&mut`
}

impl Dropout {
    /// A dropout layer in training mode, with a generator of its own.
    pub fn new(p: f32, seed: u64) -> Dropout {
        assert!((0.0..1.0).contains(&p), "Dropout: p = {p}, needs 0 ≤ p < 1");
        let (training, rng) = (true, RefCell::new(Rng::new(seed)));
        Dropout { p, training, rng }
    }
    /// A fresh mask: `1 / (1 - p)` with probability `1 - p`, else 0.
    pub fn mask(&self, shape: &[usize]) -> Tensor {
        let (keep, n) = (1.0 - self.p, shape.iter().product());
        let scale = 1.0 / keep;
        let mut rng = self.rng.borrow_mut(); // released at the end of `mask`
        let draw = |_| if rng.uniform() < keep { scale } else { 0.0 };
        Tensor::from_vec((0..n).map(draw).collect(), shape)
    }
    pub fn is_training(&self) -> bool {
        self.training
    }
}
// @end

// @listing dropout-module
impl Module for Dropout {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        if !self.training || self.p == 0.0 {
            return x; // evaluation: every unit, unscaled, and no node
        }
        let mask = x.tape().constant(self.mask(&x.shape()));
        x * mask // the backward rule multiplies the gradient by the same mask
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        Vec::new()
    }
    fn train(&mut self, on: bool) {
        self.training = on;
    }
}
// @end
