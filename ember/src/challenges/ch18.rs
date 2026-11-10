//! Chapter 18 challenges.

use std::f64::consts::PI;

use crate::nn::Parameter;
use crate::optim::{LrScheduler, Optimizer};

// @listing clip-value
/// `torch.nn.utils.clip_grad_value_`: clamp every number of every gradient to
/// [-clip, clip], each on its own.
pub fn clip_grad_value(params: &mut [&mut Parameter], clip: f32) {
    for p in params.iter_mut() {
        let Some(g) = &mut p.grad else { continue };
        g.data_mut()
            .iter_mut()
            .for_each(|x| *x = x.clamp(-clip, clip));
    }
}
// @end

// @listing inverse-sqrt
/// The schedule of "Attention Is All You Need", scaled so that its peak is the
/// base rate: a linear warmup over `warmup` steps, then base·√(warmup/(t + 1)).
#[derive(Clone, Copy, Debug)]
pub struct InverseSqrt {
    pub warmup: usize,
}
impl LrScheduler for InverseSqrt {
    fn lr(&self, base: f32, step: usize) -> f32 {
        let (t, w) = ((step + 1) as f64, self.warmup as f64);
        (base as f64 * (t / w).min((w / t).sqrt())) as f32
    }
}
// @end

// @listing restarts
/// SGDR (`CosineAnnealingWarmRestarts`): a cosine from the base rate to
/// `min_lr` over `first` steps, then again over `mult` times as many, and so on.
#[derive(Clone, Copy, Debug)]
pub struct CosineRestarts {
    pub first: usize,
    pub mult: usize,
    pub min_lr: f32,
}
impl LrScheduler for CosineRestarts {
    fn lr(&self, base: f32, step: usize) -> f32 {
        let (mut t, mut len) = (step, self.first);
        while t >= len {
            t -= len; // skip whole cycles
            len *= self.mult;
        }
        let (base, min) = (base as f64, self.min_lr as f64);
        let done = t as f64 / len as f64;
        (min + (base - min) * (1.0 + (PI * done).cos()) / 2.0) as f32
    }
}
// @end

// @listing one-cycle
/// `OneCycleLR` with its defaults: a cosine rise from base/25 to the base rate
/// over the first 30% of `steps`, then a cosine fall to base/25/10⁴ at the end.
#[derive(Clone, Copy, Debug)]
pub struct OneCycle {
    pub steps: usize,
}
impl LrScheduler for OneCycle {
    fn lr(&self, base: f32, step: usize) -> f32 {
        let base = base as f64;
        let (start, end) = (base / 25.0, base / 25.0 / 1e4);
        let peak = 0.3 * self.steps as f64 - 1.0; // the step at the top
        let last = (self.steps - 1) as f64;
        let t = (step as f64).min(last);
        let anneal = |from: f64, to: f64, pct: f64| {
            to + (from - to) * (1.0 + (PI * pct).cos()) / 2.0
        };
        let lr = if t <= peak {
            anneal(start, base, t / peak)
        } else {
            anneal(base, end, (t - peak) / (last - peak))
        };
        lr as f32
    }
}
// @end

// @listing plateau
/// `ReduceLROnPlateau` (mode "min", relative threshold): multiply every
/// group's rate by `factor` when the metric has not improved on its best by
/// more than `threshold` for more than `patience` checks in a row.
#[derive(Clone, Debug)]
pub struct ReduceLrOnPlateau {
    pub factor: f32,
    pub patience: usize,
    pub threshold: f32,
    best: f32,
    bad: usize,
}
impl ReduceLrOnPlateau {
    pub fn new(factor: f32, patience: usize) -> ReduceLrOnPlateau {
        let (best, bad, threshold) = (f32::INFINITY, 0, 1e-4);
        ReduceLrOnPlateau {
            factor,
            patience,
            threshold,
            best,
            bad,
        }
    }
    /// Record one value of the metric; returns whether the rates were cut.
    pub fn step(&mut self, opt: &mut dyn Optimizer, metric: f32) -> bool {
        if metric < self.best * (1.0 - self.threshold) {
            (self.best, self.bad) = (metric, 0);
            return false;
        }
        self.bad += 1;
        if self.bad <= self.patience {
            return false;
        }
        opt.groups_mut()
            .iter_mut()
            .for_each(|g| g.lr *= self.factor);
        self.bad = 0;
        true
    }
}
// @end

// @listing norm-watch
/// Watches the gradient's norm, step by step, and flags a step whose norm is
/// not finite, or more than `factor` times (or less than 1/`factor` of) the
/// median of the last `window` norms.
#[derive(Clone, Debug)]
pub struct NormWatch {
    pub factor: f32,
    pub window: usize,
    recent: Vec<f32>,
}
impl NormWatch {
    pub fn new(factor: f32, window: usize) -> NormWatch {
        let recent = Vec::with_capacity(window);
        NormWatch {
            factor,
            window,
            recent,
        }
    }
    /// Record a norm; true if it is out of line with the recent ones.
    pub fn check(&mut self, norm: f32) -> bool {
        if !norm.is_finite() {
            return true; // and not recorded
        }
        let mut sorted = self.recent.clone();
        sorted.sort_by(f32::total_cmp);
        let flagged = match sorted.get(sorted.len() / 2) {
            Some(&m) => norm > self.factor * m || norm * self.factor < m,
            None => false, // nothing to compare with yet
        };
        if self.recent.len() == self.window {
            self.recent.remove(0);
        }
        self.recent.push(norm);
        flagged
    }
}
// @end
