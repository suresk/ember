//! Chapter 14 challenges.

use crate::autograd::tape::{Tape, Var};
use crate::math;
use crate::nn::{Module, cross_entropy, mse_loss};
use crate::rng::Rng;
use crate::tensor::Tensor;
use crate::toy::Labeled;
use std::f32::consts::{PI, TAU};

// @listing gd-step
/// One step of gradient descent on `loss(model(x))`, from a fresh tape. Returns
/// the loss before the step. Parameters without a gradient are left alone.
pub fn gd_step(
    model: &mut dyn Module,
    x: &Tensor,
    lr: f32,
    loss: impl for<'t> Fn(Var<'t>) -> Var<'t>,
) -> f32 {
    model.zero_grad();
    let value = {
        let tape = Tape::new();
        let l = loss(model.forward(tape.constant(x.clone())));
        l.backward_into(&mut model.parameters());
        l.value().item()
    };
    for p in model.parameters() {
        if let Some(g) = &p.grad {
            p.value = p.value.sub(&g.mul_scalar(lr));
        }
    }
    value
}
// @end

// @listing spirals-k
/// `classes` interleaved spirals: arm c starts at angle 2π·c/classes. For two
/// classes this is `toy::spirals`, point for point.
pub fn spirals_k(
    per_class: usize,
    classes: usize,
    turns: f32,
    noise: f32,
    rng: &mut Rng,
) -> Labeled {
    let mut point = |arm: usize| {
        let r = rng.uniform();
        let angle = TAU * turns * r + TAU * arm as f32 / classes as f32;
        let x = r * math::cos(angle) + noise * rng.normal();
        let y = r * math::sin(angle) + noise * rng.normal();
        [x, y]
    };
    let (mut data, mut labels) = (Vec::new(), Vec::new());
    for arm in 0..classes {
        for _ in 0..per_class {
            data.extend(point(arm));
            labels.push(arm);
        }
    }
    (Tensor::from_vec(data, &[classes * per_class, 2]), labels)
}
// @end

// @listing moons
/// Two interleaved half circles, as scikit-learn's `make_moons` draws them: class
/// 0 on the upper half of the unit circle, class 1 on the lower half of a circle
/// centered at (1, 0.5); each at a uniform angle, plus normal noise.
pub fn moons(per_class: usize, noise: f32, rng: &mut Rng) -> Labeled {
    let (mut data, mut labels) = (Vec::new(), Vec::new());
    for class in 0..2 {
        for _ in 0..per_class {
            let t = PI * rng.uniform();
            let (x, y) = match class {
                0 => (math::cos(t), math::sin(t)),
                _ => (1.0 - math::cos(t), 0.5 - math::sin(t)),
            };
            data.extend([x + noise * rng.normal(), y + noise * rng.normal()]);
            labels.push(class);
        }
    }
    (Tensor::from_vec(data, &[2 * per_class, 2]), labels)
}
// @end

// @listing l2
/// `lambda` times the sum of the squares of every weight matrix (biases are left
/// out), put on `tape` as a loss term of its own.
pub fn l2_penalty<'t>(
    tape: &'t Tape,
    model: &mut dyn Module,
    lambda: f32,
) -> Var<'t> {
    let mut total = tape.constant(Tensor::zeros(&[]));
    for p in model.parameters() {
        if p.value.ndim() == 2 {
            total = total + tape.param(p).powf(2.0).sum_all();
        }
    }
    total * lambda
}

/// A step on the mean squared error plus the penalty.
pub fn l2_step(
    model: &mut dyn Module,
    x: &Tensor,
    y: &Tensor,
    lr: f32,
    lambda: f32,
) -> f32 {
    model.zero_grad();
    let value = {
        let tape = Tape::new();
        let pred = model.forward(tape.constant(x.clone()));
        let fit = mse_loss(pred, tape.constant(y.clone()));
        let loss = fit + l2_penalty(&tape, model, lambda);
        loss.backward_into(&mut model.parameters());
        fit.value().item()
    };
    for p in model.parameters() {
        let g = p.grad.as_ref().expect("a gradient");
        p.value = p.value.sub(&g.mul_scalar(lr));
    }
    value
}
// @end

// @listing early
/// What `fit_early` found: the step with the lowest held-out loss, that loss, and
/// how many steps it ran in all.
#[derive(Clone, Debug, PartialEq)]
pub struct Stopped {
    pub best_step: usize,
    pub best_loss: f32,
    pub steps: usize,
}

/// The cross-entropy of `model` on `data`, with nothing recorded for a backward.
pub fn held_out_loss(model: &dyn Module, (x, labels): &Labeled) -> f32 {
    let tape = Tape::new();
    tape.no_grad(|| cross_entropy(model.forward(tape.constant(x.clone())), labels))
        .value()
        .item()
}
// @end

// @listing early-loop
/// Full-batch descent on `train` that checks the held-out loss every `every` steps
/// and stops after `patience` checks without a new best, or at `max_steps`. The
/// model is left with the best parameters seen.
pub fn fit_early(
    model: &mut dyn Module,
    (train, held): (&Labeled, &Labeled),
    lr: f32,
    (every, patience, max_steps): (usize, usize, usize),
) -> Stopped {
    let save = |m: &mut dyn Module| -> Vec<Tensor> {
        m.parameters().iter().map(|p| p.value.clone()).collect()
    };
    let mut best = (0, held_out_loss(model, held), save(model));
    let (mut since, mut steps) = (0, 0);
    while steps < max_steps && since < patience {
        for _ in 0..every {
            gd_step(model, &train.0, lr, |z| cross_entropy(z, &train.1));
        }
        steps += every;
        let loss = held_out_loss(model, held);
        if loss < best.1 {
            (best, since) = ((steps, loss, save(model)), 0);
        } else {
            since += 1;
        }
    }
    for (p, value) in model.parameters().into_iter().zip(best.2) {
        p.value = value; // back to the best parameters
    }
    Stopped {
        best_step: best.0,
        best_loss: best.1,
        steps,
    }
}
// @end

// @listing lr-range
/// The learning-rate range test: one step at each of `steps` learning rates rising
/// geometrically from `lo` to `hi`, continuing from wherever the last step left
/// the model. Returns (learning rate, loss before its step) pairs.
pub fn lr_range(
    model: &mut dyn Module,
    x: &Tensor,
    loss: impl for<'t> Fn(Var<'t>) -> Var<'t>,
    (lo, hi): (f32, f32),
    steps: usize,
) -> Vec<(f32, f32)> {
    (0..steps)
        .map(|k| {
            let lr = lo * math::powf(hi / lo, k as f32 / (steps - 1) as f32);
            (lr, gd_step(model, x, lr, &loss))
        })
        .collect()
}
// @end
