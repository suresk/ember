//! Chapter 17: RMSProp, a step size per number from a running mean of squared
//! gradients.

use std::collections::HashMap;

use super::{Optimizer, ParamGroup, Stateful, update_each};
use crate::nn::{ParamId, Parameter};
use crate::tensor::Tensor;
use crate::train::{CheckpointError, StateDict, StateReader};

// @listing rmsprop-state
/// What `RmsProp` remembers about one parameter between steps.
#[derive(Clone, Debug)]
pub struct RmsPropState {
    /// How many updates this parameter has had.
    pub step: usize,
    /// The running mean of the squared gradients, v, one per number.
    pub square_avg: Tensor,
}
// @end

// @listing rmsprop
/// `torch.optim.RMSprop` without momentum or centering: v ← αv + (1 − α)g²,
/// then w ← w − lr·g / (√v + eps), where g = w̄ + weight_decay·w.
#[derive(Clone, Debug)]
pub struct RmsProp {
    groups: Vec<ParamGroup>,
    state: HashMap<ParamId, RmsPropState>,
    alpha: f32,
    eps: f32,
}
// @end

impl RmsProp {
    /// One group with no weight decay; PyTorch's alpha 0.99 and eps 1e-8.
    pub fn new(lr: f32) -> RmsProp {
        let all = ParamGroup {
            ids: vec![],
            lr,
            weight_decay: 0.0,
        };
        let (state, alpha, eps) = (HashMap::new(), 0.99, 1e-8);
        RmsProp {
            groups: vec![all],
            state,
            alpha,
            eps,
        }
    }
}

// @listing rmsprop-builder
impl RmsProp {
    /// The smoothing constant α of the running mean.
    pub fn alpha(mut self, alpha: f32) -> RmsProp {
        self.alpha = alpha;
        self
    }
    /// The number added to √v, so that a zero gradient divides by eps, not 0.
    pub fn eps(mut self, eps: f32) -> RmsProp {
        self.eps = eps;
        self
    }
    /// Set group 0's weight decay (added to the gradient, like `Sgd`'s).
    pub fn weight_decay(mut self, weight_decay: f32) -> RmsProp {
        self.groups[0].weight_decay = weight_decay;
        self
    }
    /// Add a group, with its own learning rate and weight decay.
    pub fn group(mut self, group: ParamGroup) -> RmsProp {
        self.groups.push(group);
        self
    }
    /// The state of one parameter, if it has ever been updated.
    pub fn state(&self, id: ParamId) -> Option<&RmsPropState> {
        self.state.get(&id)
    }
}
// @end

// @listing rmsprop-step
impl Optimizer for RmsProp {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        let (alpha, eps) = (self.alpha, self.eps);
        for p in params.iter_mut() {
            let Some(g) = &p.grad else { continue };
            assert_eq!(g.shape(), p.value.shape(), "RmsProp: gradient shape");
            let group = ParamGroup::find(&self.groups, p.id());
            let (lr, wd) = (group.lr, group.weight_decay);
            let g = g.as_slice().expect("RmsProp: a contiguous gradient");
            let zeros = || Tensor::zeros(p.value.shape());
            let state = self.state.entry(p.id()).or_insert_with(|| RmsPropState {
                step: 0,
                square_avg: zeros(),
            });
            state.step += 1;
            update_each(&mut p.value, g, [&mut state.square_avg], |w, g, [v]| {
                let g = g + wd * *w;
                *v = alpha * *v + (1.0 - alpha) * g * g;
                *w -= lr * g / (v.sqrt() + eps);
            });
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

/// Chapter 20: "i.step" and "i.square_avg" for each `params[i]` updated.
impl Stateful for RmsProp {
    fn state_dict(&self, params: &[&mut Parameter]) -> StateDict {
        let mut out = StateDict::new();
        for (i, p) in params.iter().enumerate() {
            let Some(s) = self.state.get(&p.id()) else {
                continue;
            };
            let step = Tensor::from_vec(vec![s.step as f32], &[]);
            out.insert(format!("{i}.step"), step);
            out.insert(format!("{i}.square_avg"), s.square_avg.clone());
        }
        out
    }

    fn load_state_dict(
        &mut self,
        params: &[&mut Parameter],
        state: &StateDict,
    ) -> Result<(), CheckpointError> {
        let (mut reader, mut loaded) = (StateReader::new(state), HashMap::new());
        for (i, p) in params.iter().enumerate() {
            let Some(step) = reader.get(&format!("{i}.step"), &[])? else {
                continue;
            };
            let name = format!("{i}.square_avg");
            let square_avg = reader.need(&name, p.value.shape())?;
            let step = step.item() as usize;
            loaded.insert(p.id(), RmsPropState { step, square_avg });
        }
        reader.finish()?;
        self.state = loaded;
        Ok(())
    }
}
