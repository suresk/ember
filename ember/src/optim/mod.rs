//! Part IV: optimizers. Chapter 15 defines what an optimizer is, the `Optimizer`
//! trait with parameter groups, and the first one, `Sgd`, with weight decay.
//! Chapter 17 adds the adaptive ones: `RmsProp`, `Adam` and `AdamW`. Chapter 18
//! adds learning-rate schedules and gradient clipping. Chapter 20 saves their
//! state with `Stateful`, and Chapter 23 adds a group without weight decay.

mod adam;
mod adamw;
mod clip;
mod rmsprop;
mod schedule;
mod sgd;

pub use adam::{Adam, AdamState};
pub use adamw::AdamW;
pub use clip::clip_grad_norm;
pub use rmsprop::{RmsProp, RmsPropState};
pub use schedule::{
    ConstantLr, CosineLr, LinearWarmup, LrScheduler, StepLr, base_lrs,
};
pub use sgd::{Momentum, Sgd, SgdState};

use crate::nn::{Module, ParamId, Parameter};
use crate::tensor::Tensor;
use crate::train::{CheckpointError, StateDict};

// @listing optimizer
/// A rule for updating parameters from their gradients.
pub trait Optimizer {
    /// Update every parameter that has a gradient, once the tape is dropped.
    fn step(&mut self, params: &mut [&mut Parameter]);
    /// The parameter groups. Group 0 holds every parameter no other group lists.
    fn groups(&self) -> &[ParamGroup];
    /// The groups, to change a learning rate or a weight decay between steps.
    fn groups_mut(&mut self) -> &mut [ParamGroup];
}
// @end

// @listing stateful
/// Chapter 20: an optimizer whose per-parameter state can be saved and put
/// back. Names are positions in `params`, since a `ParamId` lasts only as long
/// as the process.
pub trait Stateful: Optimizer {
    fn state_dict(&self, params: &[&mut Parameter]) -> StateDict;
    /// Put saved state back. Nothing changes on an error.
    fn load_state_dict(
        &mut self,
        params: &[&mut Parameter],
        state: &StateDict,
    ) -> Result<(), CheckpointError>;
}
// @end

// @listing param-group
/// Hyperparameters shared by a set of parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamGroup {
    pub ids: Vec<ParamId>,
    pub lr: f32,
    pub weight_decay: f32,
}

impl ParamGroup {
    /// The first group that lists `id`, or group 0 if none does.
    pub fn find(groups: &[ParamGroup], id: ParamId) -> &ParamGroup {
        let listed = groups.iter().find(|g| g.ids.contains(&id));
        listed.unwrap_or(&groups[0])
    }
}
// @end

// @listing no-decay
/// Chapter 23: a group with every parameter of fewer than two dimensions
/// (biases, and the gains and shifts of normalization layers), without decay.
pub fn no_decay(model: &mut dyn Module, lr: f32) -> ParamGroup {
    let params = model.parameters().into_iter();
    let flat = params.filter(|p| p.value.ndim() < 2);
    let ids = flat.map(|p| p.id()).collect();
    let weight_decay = 0.0;
    ParamGroup {
        ids,
        lr,
        weight_decay,
    }
}
// @end

// @listing update-each
/// `f(w, g, s)` for every number `w` of a parameter, its gradient `g`, and `s`,
/// one number from each of the `N` state buffers. Writes `w` and `s` in place.
pub(crate) fn update_each<const N: usize>(
    w: &mut Tensor,
    g: &[f32],
    state: [&mut Tensor; N],
    mut f: impl FnMut(&mut f32, f32, [&mut f32; N]),
) {
    assert!(state.iter().all(|s| s.numel() == g.len()));
    let mut bufs = state.map(|s| s.data_mut().iter_mut());
    for (w, &g) in w.data_mut().iter_mut().zip(g) {
        let s = bufs.each_mut().map(|b| b.next().unwrap());
        f(w, g, s);
    }
}
// @end
