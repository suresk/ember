//! Chapter 17: Adam, momentum and RMSProp together, with bias correction.

use std::collections::HashMap;

use super::{Optimizer, ParamGroup, Stateful, update_each};
use crate::math;
use crate::nn::{ParamId, Parameter};
use crate::tensor::Tensor;
use crate::train::{CheckpointError, StateDict, StateReader};

// @listing adam-state
/// What `Adam` remembers about one parameter between steps.
#[derive(Clone, Debug)]
pub struct AdamState {
    /// How many updates this parameter has had: t in the bias corrections.
    pub step: usize,
    /// The running mean of the gradients, m (the momentum).
    pub exp_avg: Tensor,
    /// The running mean of the squared gradients, v (RMSProp's).
    pub exp_avg_sq: Tensor,
}

impl AdamState {
    /// Step 0, with both moments zero in the parameter's shape.
    pub fn new(shape: &[usize]) -> AdamState {
        let (exp_avg, exp_avg_sq) = (Tensor::zeros(shape), Tensor::zeros(shape));
        AdamState {
            step: 0,
            exp_avg,
            exp_avg_sq,
        }
    }
}
// @end

// @listing adam
/// `torch.optim.Adam`: momentum and RMSProp with bias correction, and weight
/// decay added to the gradient (L2), or applied beside the step (decoupled).
#[derive(Clone, Debug)]
pub struct Adam {
    groups: Vec<ParamGroup>,
    state: HashMap<ParamId, AdamState>,
    betas: (f32, f32),
    eps: f32,
    decoupled: bool,
}
// @end

impl Adam {
    /// One group with no weight decay; betas (0.9, 0.999) and eps 1e-8.
    pub fn new(lr: f32) -> Adam {
        let all = ParamGroup {
            ids: vec![],
            lr,
            weight_decay: 0.0,
        };
        let (betas, eps) = ((0.9, 0.999), 1e-8);
        let (state, decoupled) = (HashMap::new(), false);
        Adam {
            groups: vec![all],
            state,
            betas,
            eps,
            decoupled,
        }
    }
}

// @listing adam-builder
impl Adam {
    /// β₁ for the mean of the gradients, β₂ for the mean of their squares.
    pub fn betas(mut self, betas: (f32, f32)) -> Adam {
        self.betas = betas;
        self
    }
    /// The number added to √v_hat.
    pub fn eps(mut self, eps: f32) -> Adam {
        self.eps = eps;
        self
    }
    /// Set group 0's weight decay.
    pub fn weight_decay(mut self, weight_decay: f32) -> Adam {
        self.groups[0].weight_decay = weight_decay;
        self
    }
    /// Decay the weights beside the step, w ← (1 − lr·λ)w, instead of adding
    /// λw to the gradient: AdamW.
    pub fn decoupled_weight_decay(mut self, decoupled: bool) -> Adam {
        self.decoupled = decoupled;
        self
    }
    /// Add a group, with its own learning rate and weight decay.
    pub fn group(mut self, group: ParamGroup) -> Adam {
        self.groups.push(group);
        self
    }
    /// The state of one parameter, if it has ever been updated.
    pub fn state(&self, id: ParamId) -> Option<&AdamState> {
        self.state.get(&id)
    }
}
// @end

// @listing adam-step
impl Optimizer for Adam {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        let ((b1, b2), eps) = (self.betas, self.eps);
        for p in params.iter_mut() {
            let Some(g) = &p.grad else { continue };
            assert_eq!(g.shape(), p.value.shape(), "Adam: gradient shape");
            let group = ParamGroup::find(&self.groups, p.id());
            let (lr, wd) = (group.lr, group.weight_decay);
            let g = g.as_slice().expect("Adam: a contiguous gradient");
            let new = || AdamState::new(p.value.shape());
            let s = self.state.entry(p.id()).or_insert_with(new);
            s.step += 1;
            let t = s.step as i32;
            let (c1, c2) = (1.0 - math::powi(b1, t), 1.0 - math::powi(b2, t)); // bias corrections
            let (step_size, c2_sqrt) = (lr / c1, c2.sqrt());
            let l2 = if self.decoupled { 0.0 } else { wd }; // λ in the gradient
            let decay = lr * (wd - l2); // or lr·λ beside it
            let buffers = [&mut s.exp_avg, &mut s.exp_avg_sq];
            update_each(&mut p.value, g, buffers, |w, g, [m, v]| {
                *w -= decay * *w; // AdamW: w ← (1 − lr·λ)w
                let g = g + l2 * *w; // Adam: w̄ + λw
                *m = b1 * *m + (1.0 - b1) * g;
                *v = b2 * *v + (1.0 - b2) * g * g;
                *w -= step_size * *m / (v.sqrt() / c2_sqrt + eps);
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

// @listing adam-save
impl Stateful for Adam {
    /// "i.step", "i.exp_avg" and "i.exp_avg_sq" for each `params[i]` updated.
    fn state_dict(&self, params: &[&mut Parameter]) -> StateDict {
        let mut out = StateDict::new();
        for (i, p) in params.iter().enumerate() {
            let Some(s) = self.state.get(&p.id()) else {
                continue;
            };
            let step = Tensor::from_vec(vec![s.step as f32], &[]); // exact to 2²⁴
            out.insert(format!("{i}.step"), step);
            out.insert(format!("{i}.exp_avg"), s.exp_avg.clone());
            out.insert(format!("{i}.exp_avg_sq"), s.exp_avg_sq.clone());
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
            let (name, shape) = (|what| format!("{i}.{what}"), p.value.shape());
            let Some(step) = reader.get(&name("step"), &[])? else {
                continue;
            };
            let mut s = AdamState::new(shape);
            s.step = step.item() as usize;
            s.exp_avg = reader.need(&name("exp_avg"), shape)?;
            s.exp_avg_sq = reader.need(&name("exp_avg_sq"), shape)?;
            loaded.insert(p.id(), s);
        }
        reader.finish()?;
        self.state = loaded; // all or nothing
        Ok(())
    }
}
// @end
