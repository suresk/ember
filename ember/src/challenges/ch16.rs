//! Chapter 16 challenges.

use std::collections::HashMap;

use crate::nn::{ParamId, Parameter};
use crate::optim::{Optimizer, ParamGroup};
use crate::tensor::Tensor;

// @listing classical
/// Momentum in Sutskever et al.'s form: v ← μv + lr·w̄, then w ← w − v. The
/// learning rate is inside the velocity.
pub struct ClassicalSgd {
    groups: Vec<ParamGroup>,
    mu: f32,
    velocity: HashMap<ParamId, Tensor>,
}

impl ClassicalSgd {
    pub fn new(lr: f32, mu: f32) -> ClassicalSgd {
        let all = ParamGroup {
            ids: vec![],
            lr,
            weight_decay: 0.0,
        };
        let velocity = HashMap::new();
        ClassicalSgd {
            groups: vec![all],
            mu,
            velocity,
        }
    }
}
// @end

// @listing classical-step
impl Optimizer for ClassicalSgd {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        for p in params.iter_mut() {
            let Some(g) = &p.grad else { continue };
            let lr = ParamGroup::find(&self.groups, p.id()).lr;
            let first = !self.velocity.contains_key(&p.id());
            let zeros = || Tensor::zeros(g.shape());
            let v = self.velocity.entry(p.id()).or_insert_with(zeros);
            let (g, v) = (g.to_vec(), v.data_mut());
            for ((w, v), g) in p.value.data_mut().iter_mut().zip(v).zip(g) {
                *v = if first { lr * g } else { self.mu * *v + lr * g };
                *w -= *v;
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

// @listing tuning
/// The learning rate and momentum that make the heavy ball fastest on the bowl
/// ½(a·x² + b·y²), a < b: both directions shrink by (√κ − 1)/(√κ + 1) per step.
pub fn heavy_ball_tuning(a: f32, b: f32) -> (f32, f32) {
    let (ra, rb) = (a.sqrt(), b.sqrt());
    let lr = 4.0 / (ra + rb).powi(2);
    let mu = ((rb - ra) / (rb + ra)).powi(2); // κ = b/a
    (lr, mu)
}
// @end

// @listing averaged
/// Any optimizer, plus the running mean of every parameter's values after each
/// of its steps (Polyak averaging).
pub struct Averaged<O: Optimizer> {
    pub inner: O,
    means: HashMap<ParamId, (usize, Tensor)>,
}

impl<O: Optimizer> Averaged<O> {
    pub fn new(inner: O) -> Averaged<O> {
        let means = HashMap::new();
        Averaged { inner, means }
    }
    /// The mean of a parameter's values so far.
    pub fn mean(&self, id: ParamId) -> Option<&Tensor> {
        self.means.get(&id).map(|(_, mean)| mean)
    }
}
// @end

// @listing averaged-step
impl<O: Optimizer> Optimizer for Averaged<O> {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        self.inner.step(params);
        for p in params.iter() {
            let zeros = || (0, Tensor::zeros(p.value.shape()));
            let (n, mean) = self.means.entry(p.id()).or_insert_with(zeros);
            *n += 1;
            let w = p.value.to_vec();
            for (m, w) in mean.data_mut().iter_mut().zip(w) {
                *m += (w - *m) / *n as f32;
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

// @listing lookahead
/// Lookahead (Zhang et al., 2019): `inner` moves the "fast" weights for `k`
/// steps; then the slow weights move `alpha` of the way toward them, and the fast
/// weights restart from there.
pub struct Lookahead<O: Optimizer> {
    pub inner: O,
    pub k: usize,
    pub alpha: f32,
    slow: HashMap<ParamId, Tensor>,
    steps: usize,
}

impl<O: Optimizer> Lookahead<O> {
    pub fn new(inner: O, k: usize, alpha: f32) -> Lookahead<O> {
        let (slow, steps) = (HashMap::new(), 0);
        Lookahead {
            inner,
            k,
            alpha,
            slow,
            steps,
        }
    }
}

// @end

// @listing lookahead-step
impl<O: Optimizer> Optimizer for Lookahead<O> {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        for p in params.iter() {
            let start = || p.value.clone(); // shared; the first write copies
            self.slow.entry(p.id()).or_insert_with(start);
        }
        self.inner.step(params);
        self.steps += 1;
        if !self.steps.is_multiple_of(self.k) {
            return;
        }
        for p in params.iter_mut() {
            let slow = self.slow.get_mut(&p.id()).unwrap().data_mut();
            for (s, w) in slow.iter_mut().zip(p.value.data_mut()) {
                *s += self.alpha * (*w - *s);
                *w = *s;
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
