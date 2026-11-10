//! Chapter 18: learning-rate schedules. A schedule is a pure function from a
//! group's base learning rate and the step number to the rate for that step.

use std::f64::consts::PI;

use super::Optimizer;

// @listing scheduler
/// A learning rate that changes over training.
pub trait LrScheduler {
    /// The rate for update number `step` (0 for the first) of a group whose
    /// learning rate was `base` when training started.
    fn lr(&self, base: f32, step: usize) -> f32;
    /// Set every group's learning rate for update `step`, from its `base` rate.
    fn apply(&self, opt: &mut dyn Optimizer, base: &[f32], step: usize) {
        let groups = opt.groups_mut();
        assert_eq!(groups.len(), base.len(), "apply: one base rate per group");
        for (group, &b) in groups.iter_mut().zip(base) {
            group.lr = self.lr(b, step);
        }
    }
}

/// Every group's learning rate, read once before training: the base rates.
pub fn base_lrs(opt: &dyn Optimizer) -> Vec<f32> {
    opt.groups().iter().map(|g| g.lr).collect()
}
// @end

// @listing constant
/// The base rate at every step.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConstantLr;
impl LrScheduler for ConstantLr {
    fn lr(&self, base: f32, _step: usize) -> f32 {
        base
    }
}
// @end

// @listing step-lr
/// `torch.optim.lr_scheduler.StepLR`: multiply the rate by `gamma` every
/// `every` steps.
#[derive(Clone, Copy, Debug)]
pub struct StepLr {
    pub every: usize,
    pub gamma: f32,
}
impl LrScheduler for StepLr {
    fn lr(&self, base: f32, step: usize) -> f32 {
        let drops = (step / self.every) as i32; // whole periods so far
        (base as f64 * (self.gamma as f64).powi(drops)) as f32
    }
}
// @end

// @listing cosine
/// `torch.optim.lr_scheduler.CosineAnnealingLR`: half a cosine from the base
/// rate down to `min_lr` over `steps` steps, then `min_lr`.
#[derive(Clone, Copy, Debug)]
pub struct CosineLr {
    pub steps: usize,
    pub min_lr: f32,
}
impl LrScheduler for CosineLr {
    fn lr(&self, base: f32, step: usize) -> f32 {
        let done = step.min(self.steps) as f64 / self.steps as f64; // 0 to 1
        let (base, min) = (base as f64, self.min_lr as f64);
        (min + (base - min) * (1.0 + (PI * done).cos()) / 2.0) as f32
    }
}
// @end

impl StepLr {
    pub fn new(every: usize, gamma: f32) -> StepLr {
        StepLr { every, gamma }
    }
}

impl CosineLr {
    pub fn new(steps: usize, min_lr: f32) -> CosineLr {
        CosineLr { steps, min_lr }
    }
}

// @listing warmup
/// A linear rise from base/`steps` to the base rate over the first `steps`
/// steps, then `then`, started from its own step 0.
pub struct LinearWarmup {
    pub steps: usize,
    pub then: Box<dyn LrScheduler>,
}
impl LinearWarmup {
    pub fn new(steps: usize, then: impl LrScheduler + 'static) -> LinearWarmup {
        let then = Box::new(then);
        LinearWarmup { steps, then }
    }
}
impl LrScheduler for LinearWarmup {
    fn lr(&self, base: f32, step: usize) -> f32 {
        if step < self.steps {
            base * (step + 1) as f32 / self.steps as f32
        } else {
            self.then.lr(base, step - self.steps)
        }
    }
}
// @end

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_are_trait_objects() {
        let all: Vec<Box<dyn LrScheduler>> = vec![
            Box::new(ConstantLr),
            Box::new(StepLr::new(2, 0.5)),
            Box::new(LinearWarmup::new(2, ConstantLr)),
        ];
        let at3: Vec<f32> = all.iter().map(|s| s.lr(1.0, 3)).collect();
        assert_eq!(at3, [1.0, 0.5, 1.0]);
        assert_eq!(size_of::<Box<dyn LrScheduler>>(), 16);
    }
}
