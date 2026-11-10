//! Chapter 17: AdamW, Adam with decoupled weight decay.

use super::{Adam, AdamState, Optimizer, ParamGroup, Stateful};
use crate::nn::{ParamId, Parameter};
use crate::train::{CheckpointError, StateDict};

// @listing adamw
/// `torch.optim.AdamW` (Loshchilov and Hutter): Adam whose weight decay shrinks
/// the weights beside the step, w ← (1 − lr·λ)w, with λ = 0.01 unless set.
#[derive(Clone, Debug)]
pub struct AdamW(Adam);

impl AdamW {
    pub fn new(lr: f32) -> AdamW {
        let adam = Adam::new(lr).weight_decay(0.01);
        AdamW(adam.decoupled_weight_decay(true))
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        self.0.step(params)
    }
    fn groups(&self) -> &[ParamGroup] {
        self.0.groups()
    }
    fn groups_mut(&mut self) -> &mut [ParamGroup] {
        self.0.groups_mut()
    }
}
// @end

/// Adam's builder methods, for AdamW.
impl AdamW {
    pub fn betas(self, betas: (f32, f32)) -> AdamW {
        AdamW(self.0.betas(betas))
    }
    pub fn eps(self, eps: f32) -> AdamW {
        AdamW(self.0.eps(eps))
    }
    pub fn weight_decay(self, weight_decay: f32) -> AdamW {
        AdamW(self.0.weight_decay(weight_decay))
    }
    pub fn group(self, group: ParamGroup) -> AdamW {
        AdamW(self.0.group(group))
    }
    pub fn state(&self, id: ParamId) -> Option<&AdamState> {
        self.0.state(id)
    }
}

/// Chapter 20: AdamW's state is Adam's.
impl Stateful for AdamW {
    fn state_dict(&self, params: &[&mut Parameter]) -> StateDict {
        self.0.state_dict(params)
    }

    fn load_state_dict(
        &mut self,
        params: &[&mut Parameter],
        state: &StateDict,
    ) -> Result<(), CheckpointError> {
        self.0.load_state_dict(params, state)
    }
}
