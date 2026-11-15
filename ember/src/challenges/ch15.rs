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
/// signSGD: every number moves by exactly `lr`, against the sign of its gradient
/// (with the group's weight decay added to the gradient, as in `Sgd`).
#[derive(Clone, Debug)]
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
            let Some(grad) = &p.grad else { continue };
            let group = ParamGroup::find(&self.groups, p.id());
            let (lr, wd) = (group.lr, group.weight_decay);
            let g = grad.as_slice().expect("SignSgd: a contiguous gradient");
            for (w, &gi) in p.value.data_mut().iter_mut().zip(g) {
                let d = gi + wd * *w; // the sign of the decayed gradient
                *w -= lr * if d == 0.0 { 0.0 } else { d.signum() };
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
#[derive(Clone, Debug)]
pub struct MaxNorm<O: Optimizer> {
    pub inner: O,
    pub max: f32,
}

impl<O: Optimizer> Optimizer for MaxNorm<O> {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        self.inner.step(params);
        // A parameter the inner step skipped (no gradient) is left alone too.
        let matrices = params.iter_mut().filter(|p| p.value.ndim() == 2);
        for p in matrices.filter(|p| p.grad.is_some()) {
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
