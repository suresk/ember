//! Chapter 23 challenges.

use crate::autograd::tape::{Tape, Var};
use crate::nn::{BatchNorm1d, LayerNorm, Linear, Module, Parameter};
use crate::tensor::Tensor;

// @listing rms-norm
/// RMS normalization: each example's last axis divided by its root mean
/// square, then scaled by `weight`. No mean is subtracted, and no shift added.
#[derive(Debug)]
pub struct RmsNorm {
    pub weight: Parameter,
    pub eps: f32,
}

impl Module for RmsNorm {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let last = x.shape().len() - 1;
        let ms = (x * x).mean(last, true); // the mean square, one per example
        x / (ms + self.eps).powf(0.5) * x.tape().param(&self.weight)
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight]
    }
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)> {
        vec![("weight".into(), &mut self.weight)]
    }
}
// @end

impl RmsNorm {
    /// For a last axis of `features`: gain 1, and PyTorch's default ε, the
    /// machine epsilon of f32.
    pub fn new(features: usize) -> RmsNorm {
        let ones = Tensor::from_vec(vec![1.0; features], &[features]);
        let (weight, eps) = (Parameter::new(ones), f32::EPSILON);
        RmsNorm { weight, eps }
    }
}

// @listing batch-norm-2d
/// Batch normalization of images `[N, C, H, W]`: each channel normalized over
/// the batch and every pixel, `N·H·W` values, with one gain and shift each.
#[derive(Debug)]
pub struct BatchNorm2d(pub BatchNorm1d);

impl Module for BatchNorm2d {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let s = x.shape(); // [N, C, H, W]
        let (n, c, h, w) = (s[0], s[1], s[2], s[3]);
        let rows = x.permute(&[0, 2, 3, 1]).reshape(&[n * h * w, c]);
        let y = self.0.forward(rows); // one row per pixel, one column per channel
        y.reshape(&[n, h, w, c]).permute(&[0, 3, 1, 2])
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        self.0.parameters()
    }
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)> {
        self.0.named_parameters()
    }
    fn train(&mut self, on: bool) {
        self.0.train(on)
    }
    fn named_buffers(&mut self) -> Vec<(String, &mut Tensor)> {
        self.0.named_buffers()
    }
}
// @end

// @listing group-norm
/// Group normalization of `[N, C, H, W]`: the channels split into `groups`
/// groups, each normalized over its channels and pixels, per example.
pub fn group_norm(x: Var<'_>, groups: usize, eps: f32) -> Var<'_> {
    let shape = x.shape();
    let (n, c) = (shape[0], shape[1]);
    assert!(c % groups == 0, "group_norm: {c} channels, {groups} groups");
    let per_group = x.value().numel() / (n * groups);
    let grouped = x.reshape(&[n, groups, per_group]);
    grouped.normalize(2, eps).reshape(&shape)
}
// @end

// @listing cumulative
/// Run `batches` through `bn` in training mode with PyTorch's
/// `momentum=None`: each batch weighs `1 / (batches seen + 1)`, so the running
/// statistics are the plain averages of every batch's.
pub fn cumulative(bn: &mut BatchNorm1d, batches: &[Tensor]) {
    let tape = Tape::new();
    for batch in batches {
        let seen = bn.running().batches.item();
        bn.momentum = 1.0 / (seen + 1.0);
        bn.forward(tape.constant(batch.clone()));
    }
}
// @end

// @listing fold
/// The `Linear` layer that computes `bn(linear(x))` in evaluation mode: the
/// normalization's scale `γ/√(var + ε)` folded into the weight's columns,
/// and everything else into the bias.
pub fn fold_batch_norm(linear: &Linear, bn: &BatchNorm1d) -> Linear {
    let r = bn.running();
    let scale = r.var.add_scalar(bn.eps).powf(-0.5).mul(&bn.weight.value);
    let weight = linear.weight.value.mul(&scale); // [in, out] · [out]
    let shifted = linear.bias.value.sub(&r.mean).mul(&scale);
    Linear::from_tensors(weight, shifted.add(&bn.bias.value))
}
// @end

// @listing residual-norm
/// A residual block with layer normalization, before the branch (pre-norm,
/// `x + f(norm(x))`) or after the sum (post-norm, `norm(x + f(x))`).
pub struct NormResidual<M: Module> {
    pub f: M,
    pub norm: LayerNorm,
    pub pre: bool,
}

impl<M: Module> Module for NormResidual<M> {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        if self.pre {
            x + self.f.forward(self.norm.forward(x))
        } else {
            self.norm.forward(x + self.f.forward(x))
        }
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        let mut params = self.f.parameters();
        params.extend(self.norm.parameters());
        params
    }
    fn train(&mut self, on: bool) {
        self.f.train(on)
    }
}
// @end
