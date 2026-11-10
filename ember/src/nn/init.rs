//! Chapter 12: initial weights at a scale that keeps a deep network's signals, and
//! its gradients, near 1 from layer to layer.

use crate::rng::Rng;
use crate::tensor::Tensor;

// @listing init-sample
/// Normal samples with standard deviation `std`.
pub fn normal(shape: &[usize], std: f32, rng: &mut Rng) -> Tensor {
    Tensor::randn(shape, rng).mul_scalar(std)
}

/// Uniform samples between `-bound` and `bound`; their variance is `bound² / 3`.
pub fn uniform(shape: &[usize], bound: f32, rng: &mut Rng) -> Tensor {
    let unit = Tensor::rand(shape, rng); // in [0, 1)
    unit.mul_scalar(2.0 * bound).add_scalar(-bound)
}
// @end

// @listing init-std

/// Glorot and Bengio (2010), for tanh: variance `2 / (fan_in + fan_out)`.
pub fn xavier_std(fan_in: usize, fan_out: usize) -> f32 {
    (2.0 / (fan_in + fan_out) as f32).sqrt()
}

/// He, Zhang, Ren and Sun (2015), for ReLU: variance `2 / fan_in`.
pub fn he_std(fan_in: usize) -> f32 {
    (2.0 / fan_in as f32).sqrt()
}
// @end
