//! Chapter 15 challenges.

use crate::nn::{Module, ParamId, Parameter};
use crate::optim::{Optimizer, ParamGroup, Sgd};

// @listing decay-groups
/// `Sgd` with `weight_decay` on every parameter of two or more dimensions (the
/// weight matrices) and none on the rest (biases), both at the rate `lr`.
pub fn decay_groups(model: &mut dyn Module, lr: f32, weight_decay: f32) -> Sgd {
    let ids: Vec<ParamId> = model
        .parameters()
        .iter()
        .filter(|p| p.value.ndim() < 2)
        .map(|p| p.id())
        .collect();
    let no_decay = ParamGroup {
        ids,
        lr,
        weight_decay: 0.0,
    };
    Sgd::new(lr).weight_decay(weight_decay).group(no_decay)
}
// @end

// @listing sign-sgd
/// signSGD: every number moves by exactly `lr`, against the sign of its gradient.
pub struct SignSgd {
    groups: Vec<ParamGroup>,
}

impl SignSgd {
    pub fn new(lr: f32) -> SignSgd {
        let all = ParamGroup {
            ids: vec![],
            lr,
            weight_decay: 0.0,
        };
        SignSgd { groups: vec![all] }
    }
}

impl Optimizer for SignSgd {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        for p in params.iter_mut() {
            let Some(g) = &p.grad else { continue };
            let lr = ParamGroup::find(&self.groups, p.id()).lr;
            let g = g.to_vec();
            for (w, g) in p.value.data_mut().iter_mut().zip(g) {
                *w -= lr * if g == 0.0 { 0.0 } else { g.signum() };
            }
        }
    }
    fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }
    fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }
}
// @end

// @listing max-norm
/// Any optimizer, then each column of a weight matrix scaled down to length `max`.
pub struct MaxNorm<O: Optimizer> {
    pub inner: O,
    pub max: f32,
}

impl<O: Optimizer> Optimizer for MaxNorm<O> {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        self.inner.step(params);
        for p in params.iter_mut().filter(|p| p.value.ndim() == 2) {
            let cols = p.value.shape()[1];
            let w = p.value.data_mut(); // row-major [in, out]
            for j in 0..cols {
                let norm = w[j..].iter().step_by(cols).map(|x| x * x).sum::<f32>();
                let scale = self.max / norm.sqrt();
                if scale < 1.0 {
                    w[j..].iter_mut().step_by(cols).for_each(|x| *x *= scale);
                }
            }
        }
    }
    fn groups(&self) -> &[ParamGroup] {
        self.inner.groups()
    }
    fn groups_mut(&mut self) -> &mut [ParamGroup] {
        self.inner.groups_mut()
    }
}
// @end
