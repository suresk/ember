//! Chapter 23: normalization layers. `LayerNorm` normalizes each example over its
//! features; `BatchNorm1d` normalizes each feature over the batch, and keeps
//! running statistics for evaluation.

use crate::autograd::tape::Var;
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;
use std::cell::RefCell;

/// A gain of ones and a shift of zeros, `[features]` each: the identity.
fn identity_affine(features: usize) -> (Parameter, Parameter) {
    let ones = Tensor::from_vec(vec![1.0; features], &[features]);
    (
        Parameter::new(ones),
        Parameter::new(Tensor::zeros(&[features])),
    )
}

// @listing layer-norm
/// Layer normalization: each example's last axis normalized to mean 0 and
/// variance 1, then scaled by `weight` (γ) and shifted by `bias` (β).
#[derive(Debug)]
pub struct LayerNorm {
    pub weight: Parameter,
    pub bias: Parameter,
    pub eps: f32,
}

impl Module for LayerNorm {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let (tape, shape, n) = (x.tape(), x.shape(), self.weight.value.numel());
        let fits = shape.last() == Some(&n);
        assert!(fits, "LayerNorm: input {shape:?}, {n} features");
        let x_hat = x.normalize(shape.len() - 1, self.eps);
        x_hat * tape.param(&self.weight) + tape.param(&self.bias)
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight, &mut self.bias]
    }
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)> {
        let (w, b) = (&mut self.weight, &mut self.bias);
        vec![("weight".into(), w), ("bias".into(), b)]
    }
}
// @end

impl LayerNorm {
    /// For a last axis of `features`, starting as the identity, ε = 1e-5.
    pub fn new(features: usize) -> LayerNorm {
        let (weight, bias) = identity_affine(features);
        let eps = 1e-5;
        LayerNorm { weight, bias, eps }
    }
}

// @listing running
/// The running statistics of a `BatchNorm1d`: what evaluation normalizes with.
#[derive(Clone, Debug)]
pub struct Running {
    pub mean: Tensor,    // [features], starts at 0
    pub var: Tensor,     // [features], starts at 1
    pub batches: Tensor, // shape [], the number of batches seen
}
// @end

// @listing batch-norm

/// Batch normalization of `[N, features]`: in training, each feature is
/// normalized over the batch; in evaluation, with the running statistics.
#[derive(Debug)]
pub struct BatchNorm1d {
    pub weight: Parameter,
    pub bias: Parameter,
    pub eps: f32,
    pub momentum: f32,
    training: bool,
    running: RefCell<Running>, // updated by `forward(&self)` in training
}
// @end

impl BatchNorm1d {
    /// For `features` features: the identity, ε = 1e-5, momentum 0.1, in
    /// training mode, with running mean 0 and variance 1.
    pub fn new(features: usize) -> BatchNorm1d {
        let (weight, bias) = identity_affine(features);
        let mean = Tensor::zeros(&[features]);
        let var = Tensor::from_vec(vec![1.0; features], &[features]);
        let batches = Tensor::zeros(&[]);
        let running = RefCell::new(Running { mean, var, batches });
        let (eps, momentum, training) = (1e-5, 0.1, true);
        BatchNorm1d {
            weight,
            bias,
            eps,
            momentum,
            training,
            running,
        }
    }

    /// A copy of the running statistics.
    pub fn running(&self) -> Running {
        self.running.borrow().clone()
    }

    pub fn is_training(&self) -> bool {
        self.training
    }
}

// @listing update
impl BatchNorm1d {
    /// Fold one batch `[N, C]` into the running statistics, PyTorch's way: the
    /// mean, and the *unbiased* variance, each weighted by `momentum`.
    fn update(&self, x: &Tensor) {
        let (n, c, m) = (x.shape()[0] as f32, x.shape()[1], self.momentum);
        let (mean, var) = x.mean_var(0); // var divides by N
        let (mean, var) = (mean.reshape(&[c]), var.reshape(&[c]));
        let unbiased = var.mul_scalar(n / (n - 1.0)); // divides by N - 1
        let mut r = self.running.borrow_mut(); // released when `update` returns
        r.mean = r.mean.mul_scalar(1.0 - m).add(&mean.mul_scalar(m));
        r.var = r.var.mul_scalar(1.0 - m).add(&unbiased.mul_scalar(m));
        r.batches = r.batches.add_scalar(1.0);
    }
}
// @end

// @listing batch-norm-module
impl Module for BatchNorm1d {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let (tape, shape, c) = (x.tape(), x.shape(), self.weight.value.numel());
        let fits = shape.len() == 2 && shape[1] == c;
        assert!(fits, "BatchNorm1d: input {shape:?}, {c} features");
        let x_hat = if self.training {
            let more = shape[0] > 1; // PyTorch's message:
            assert!(more, "Expected more than 1 value per channel when training");
            self.update(&x.value());
            x.normalize(0, self.eps) // this batch's statistics
        } else {
            let r = self.running.borrow();
            let inv_std = r.var.add_scalar(self.eps).powf(-0.5);
            (x - tape.constant(r.mean.clone())) * tape.constant(inv_std)
        };
        x_hat * tape.param(&self.weight) + tape.param(&self.bias)
    }
    fn train(&mut self, on: bool) {
        self.training = on;
    }
    fn named_buffers(&mut self) -> Vec<(String, &mut Tensor)> {
        let r = self.running.get_mut(); // `&mut self`: no run-time check
        let names = ["running_mean", "running_var", "num_batches_tracked"];
        let buffers = [&mut r.mean, &mut r.var, &mut r.batches];
        names.into_iter().map(String::from).zip(buffers).collect()
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight, &mut self.bias]
    }
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)> {
        let (w, b) = (&mut self.weight, &mut self.bias);
        vec![("weight".into(), w), ("bias".into(), b)]
    }
}
// @end
