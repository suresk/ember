//! Chapter 17 challenges.

use crate::math;
use std::collections::HashMap;

use crate::nn::{ParamId, Parameter};
use crate::optim::{Optimizer, ParamGroup, update_each};
use crate::tensor::Tensor;

// @listing buffers
/// What the optimizers of these challenges share: one group (no weight decay),
/// and per parameter a step count and `N` buffers in its shape.
pub struct Buffers<const N: usize> {
    pub groups: Vec<ParamGroup>,
    pub state: HashMap<ParamId, (usize, [Tensor; N])>,
}

impl<const N: usize> Buffers<N> {
    /// `f(t, lr, w, g, s)` for every number of every parameter with a gradient.
    pub fn step<F>(&mut self, params: &mut [&mut Parameter], mut f: F)
    where
        F: FnMut(usize, f32, &mut f32, f32, [&mut f32; N]),
    {
        for p in params.iter_mut() {
            let Some(g) = &p.grad else { continue };
            let lr = ParamGroup::find(&self.groups, p.id()).lr;
            let g = g.as_slice().expect("a contiguous gradient");
            let zeros = || (0, [(); N].map(|_| Tensor::zeros(p.value.shape())));
            let (t, bufs) = self.state.entry(p.id()).or_insert_with(zeros);
            *t += 1;
            let t = *t;
            let bufs = bufs.each_mut();
            update_each(&mut p.value, g, bufs, |w, g, s| f(t, lr, w, g, s));
        }
    }
}
// @end

impl<const N: usize> Buffers<N> {
    /// One group, holding every parameter, and no state yet.
    pub fn new(lr: f32) -> Buffers<N> {
        let all = ParamGroup {
            ids: vec![],
            lr,
            weight_decay: 0.0,
        };
        let state = HashMap::new();
        Buffers {
            groups: vec![all],
            state,
        }
    }
}

/// `groups` and `groups_mut` from a field `buffers: Buffers<N>`.
macro_rules! buffered_groups {
    () => {
        fn groups(&self) -> &[ParamGroup] {
            &self.buffers.groups
        }
        fn groups_mut(&mut self) -> &mut [ParamGroup] {
            &mut self.buffers.groups
        }
    };
}

// @listing centered
/// RMSProp with `centered=True`: divide by the running standard deviation of
/// the gradient, √(v − ḡ²), instead of its root mean square √v.
pub struct CenteredRmsProp {
    pub alpha: f32,
    pub eps: f32,
    pub buffers: Buffers<2>,
}

impl CenteredRmsProp {
    pub fn new(lr: f32) -> CenteredRmsProp {
        let buffers = Buffers::new(lr);
        CenteredRmsProp {
            alpha: 0.99,
            eps: 1e-8,
            buffers,
        }
    }
}

impl Optimizer for CenteredRmsProp {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        let (alpha, eps) = (self.alpha, self.eps);
        self.buffers.step(params, |_, lr, w, g, [v, mean]| {
            *v = alpha * *v + (1.0 - alpha) * g * g;
            *mean = alpha * *mean + (1.0 - alpha) * g;
            let var = (*v - *mean * *mean).max(0.0); // rounding can make it < 0
            *w -= lr * g / (var.sqrt() + eps);
        });
    }
    buffered_groups!();
}
// @end

// @listing amsgrad
/// Adam with `amsgrad=True`: the step divides by the largest v seen so far, so
/// a burst of large gradients is never forgotten.
pub struct AmsGrad {
    pub buffers: Buffers<3>,
}

impl AmsGrad {
    pub fn new(lr: f32) -> AmsGrad {
        AmsGrad {
            buffers: Buffers::new(lr),
        }
    }
}

impl Optimizer for AmsGrad {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8);
        self.buffers.step(params, |t, lr, w, g, [m, v, v_max]| {
            let (c1, c2) = (
                1.0 - math::powi(b1, t as i32),
                1.0 - math::powi(b2, t as i32),
            );
            *m = b1 * *m + (1.0 - b1) * g;
            *v = b2 * *v + (1.0 - b2) * g * g;
            *v_max = v_max.max(*v);
            *w -= lr / c1 * *m / ((*v_max / c2).sqrt() + eps);
        });
    }
    buffered_groups!();
}
// @end

// @listing adagrad
/// Adagrad (Duchi, Hazan and Singer, 2011): divide by the root of the *sum* of
/// all squared gradients so far. PyTorch's defaults: lr 0.01, eps 1e-10.
pub struct Adagrad {
    pub eps: f32,
    pub buffers: Buffers<1>,
}

impl Adagrad {
    pub fn new(lr: f32) -> Adagrad {
        let buffers = Buffers::new(lr);
        Adagrad {
            eps: 1e-10,
            buffers,
        }
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        let eps = self.eps;
        self.buffers.step(params, |_, lr, w, g, [sum]| {
            *sum += g * g;
            *w -= lr * g / (sum.sqrt() + eps);
        });
    }
    buffered_groups!();
}
// @end

// @listing lion
/// Lion (Chen et al., 2023): step by the sign of a blend of the momentum and the
/// gradient, w ← w − lr·(sign(β₁m + (1 − β₁)g) + λw), then m ← β₂m + (1 − β₂)g.
pub struct Lion {
    pub betas: (f32, f32),
    pub weight_decay: f32,
    pub buffers: Buffers<1>,
}

impl Lion {
    pub fn new(lr: f32) -> Lion {
        let buffers = Buffers::new(lr);
        let (betas, weight_decay) = ((0.9, 0.99), 0.0);
        Lion {
            betas,
            weight_decay,
            buffers,
        }
    }
}

impl Optimizer for Lion {
    fn step(&mut self, params: &mut [&mut Parameter]) {
        let ((b1, b2), wd) = (self.betas, self.weight_decay);
        self.buffers.step(params, |_, lr, w, g, [m]| {
            let c = b1 * *m + (1.0 - b1) * g;
            let sign = if c == 0.0 { 0.0 } else { c.signum() }; // signum(0) is 1
            *w -= lr * (sign + wd * *w);
            *m = b2 * *m + (1.0 - b2) * g;
        });
    }
    buffered_groups!();
}
// @end
