//! Chapter 15: stochastic gradient descent, with weight decay and parameter groups.
//! Chapter 16: momentum and Nesterov's look-ahead.

use std::collections::HashMap;

use super::{Optimizer, ParamGroup, Stateful};
use crate::nn::{ParamId, Parameter};
use crate::tensor::Tensor;
use crate::train::{CheckpointError, StateDict, StateReader};

// @listing sgd-state
/// What `Sgd` remembers about one parameter between steps.
#[derive(Clone, Debug, Default)]
pub struct SgdState {
    /// How many updates this parameter has had.
    pub steps: usize,
    /// The velocity, made on the first step with momentum (Chapter 16).
    pub momentum_buffer: Option<Tensor>,
}
// @end

// @listing sgd
/// Plain gradient descent: w ← w − lr·(w̄ + weight_decay·w).
#[derive(Clone, Debug)]
pub struct Sgd {
    groups: Vec<ParamGroup>,
    state: HashMap<ParamId, SgdState>,
    momentum: Momentum, // Chapter 16
}

impl Sgd {
    /// One group, holding every parameter, with no weight decay.
    pub fn new(lr: f32) -> Sgd {
        let all = ParamGroup {
            ids: vec![],
            lr,
            weight_decay: 0.0,
        };
        Sgd {
            groups: vec![all],
            state: HashMap::new(),
            momentum: Momentum::default(),
        }
    }
}
// @end

// @listing sgd-builder
impl Sgd {
    /// Set group 0's weight decay.
    pub fn weight_decay(mut self, weight_decay: f32) -> Sgd {
        self.groups[0].weight_decay = weight_decay;
        self
    }
    /// Add a group, with its own learning rate and weight decay.
    pub fn group(mut self, group: ParamGroup) -> Sgd {
        self.groups.push(group);
        self
    }
    /// The state of one parameter, if it has ever been updated.
    pub fn state(&self, id: ParamId) -> Option<&SgdState> {
        self.state.get(&id)
    }
}
// @end

// @listing sgd-step
impl Optimizer for Sgd {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        for p in params.iter_mut() {
            let Some(g) = &p.grad else { continue }; // not reached: leave it alone
            assert_eq!(g.shape(), p.value.shape(), "Sgd: gradient shape");
            let group = ParamGroup::find(&self.groups, p.id());
            let (lr, wd) = (group.lr, group.weight_decay);
            let g = g.as_slice().expect("Sgd: a contiguous gradient");
            let state = self.state.entry(p.id()).or_default();
            state.steps += 1;
            if self.momentum.mu == 0.0 {
                for (w, &g) in p.value.data_mut().iter_mut().zip(g) {
                    *w -= lr * (g + wd * *w); // in place
                }
            } else {
                let v = &mut state.momentum_buffer; // Chapter 16
                self.momentum.update(&mut p.value, g, (lr, wd), v);
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

// @listing momentum
/// `torch.optim.SGD`'s momentum (here `mu`), dampening and nesterov.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Momentum {
    pub mu: f32,
    pub dampening: f32,
    pub nesterov: bool,
}
// @end

// @listing momentum-update
impl Momentum {
    /// One step for one parameter: v ← μv + (1 − τ)d, with v = d on the first,
    /// where d = w̄ + wd·w; then w ← w − lr·v, or w ← w − lr·(d + μv) for Nesterov.
    pub fn update(
        &self,
        w: &mut Tensor,
        g: &[f32],
        (lr, wd): (f32, f32),
        buffer: &mut Option<Tensor>,
    ) {
        let (mu, keep) = (self.mu, 1.0 - self.dampening);
        let first = buffer.is_none(); // PyTorch starts the velocity at d
        let v = buffer.get_or_insert_with(|| Tensor::zeros(w.shape()));
        let (w, v) = (w.data_mut(), v.data_mut());
        for ((w, v), &g) in w.iter_mut().zip(v.iter_mut()).zip(g) {
            let d = g + wd * *w; // Chapter 15's gradient with decay
            *v = if first { d } else { mu * *v + keep * d };
            *w -= lr * if self.nesterov { d + mu * *v } else { *v };
        }
    }
}
// @end

// @listing momentum-builder
impl Sgd {
    /// Momentum μ: a velocity per parameter, v ← μv + w̄, and w ← w − lr·v.
    pub fn momentum(mut self, momentum: f32) -> Sgd {
        self.momentum.mu = momentum;
        self
    }
    /// Dampening τ: v ← μv + (1 − τ)w̄ after the first step.
    pub fn dampening(mut self, dampening: f32) -> Sgd {
        self.momentum.dampening = dampening;
        self
    }
    /// Nesterov's look-ahead: w ← w − lr·(w̄ + μv). Needs momentum, no dampening.
    pub fn nesterov(mut self, nesterov: bool) -> Sgd {
        self.momentum.nesterov = nesterov;
        let Momentum { mu, dampening, .. } = self.momentum;
        let ok = !nesterov || (mu > 0.0 && dampening == 0.0);
        assert!(ok, "Sgd: Nesterov needs momentum and no dampening");
        self
    }
}
// @end

/// Chapter 20: "i.step" and, with momentum, "i.momentum_buffer" for each
/// `params[i]` that has been updated.
impl Stateful for Sgd {
    fn state_dict(&self, params: &[&mut Parameter]) -> StateDict {
        let mut out = StateDict::new();
        for (i, p) in params.iter().enumerate() {
            let Some(s) = self.state.get(&p.id()) else {
                continue;
            };
            let step = Tensor::from_vec(vec![s.steps as f32], &[]);
            out.insert(format!("{i}.step"), step);
            if let Some(v) = &s.momentum_buffer {
                out.insert(format!("{i}.momentum_buffer"), v.clone());
            }
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
            let name = format!("{i}.momentum_buffer");
            let momentum_buffer = reader.get(&name, p.value.shape())?;
            let steps = step.item() as usize;
            loaded.insert(
                p.id(),
                SgdState {
                    steps,
                    momentum_buffer,
                },
            );
        }
        reader.finish()?;
        self.state = loaded;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Tape, Tensor};
    use std::rc::Rc;

    /// Section 16.3: an empty momentum buffer costs no space.
    #[test]
    fn an_optional_buffer_is_free() {
        use std::mem::size_of;
        assert_eq!(size_of::<Option<Tensor>>(), size_of::<Tensor>());
        assert_eq!(size_of::<SgdState>(), 72);
    }

    /// Section 15.5: one owner during the update, so `make_mut` writes in place.
    #[test]
    fn the_update_has_the_buffer_to_itself() {
        let mut p = Parameter::new(Tensor::arange(&[2, 3]));
        {
            let tape = Tape::new();
            tape.param(&p).sum_all().backward_into(&mut [&mut p]);
            assert_eq!(Rc::strong_count(&p.value.storage), 3); // p, the node, a rule
        }
        assert_eq!(Rc::strong_count(&p.value.storage), 1);
        let before = Rc::as_ptr(&p.value.storage);
        Sgd::new(0.5).step(&mut [&mut p]);
        assert_eq!(Rc::as_ptr(&p.value.storage), before); // the same buffer
        assert_eq!(Rc::strong_count(&p.value.storage), 1);
    }
}
