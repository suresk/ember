//! Tests for the Chapter 16 challenges.
use ember::{
    Averaged, ClassicalSgd, Lookahead, Optimizer, Parameter, Rng, Sgd, Tensor,
    heavy_ball_tuning,
};

fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

fn scalar(w: f32) -> Parameter {
    Parameter::new(Tensor::from_vec(vec![w], &[1]))
}

/// The velocity `opt` keeps for a one-number parameter.
fn velocity(opt: &Sgd, w: &Parameter) -> f32 {
    let state = opt.state(w.id()).unwrap();
    state.momentum_buffer.as_ref().unwrap().item()
}

/// One step of `opt` with the gradient `g(w)`.
fn step(w: &mut Parameter, opt: &mut dyn Optimizer, g: impl Fn(f32) -> f32) {
    w.grad = Some(Tensor::from_vec(vec![g(w.value.item())], &[1]));
    opt.step(&mut [&mut *w]);
}

#[test]
fn three_momentum_steps() {
    // @listing predict-test
    let (mut plain, mut heavy) = (scalar(1.0), scalar(1.0));
    let (mut sgd, mut momentum) = (Sgd::new(0.1), Sgd::new(0.1).momentum(0.5));
    let (mut ws, mut hs) = (vec![], vec![]);
    for _ in 0..3 {
        step(&mut plain, &mut sgd, |w| 2.0 * w); // f(w) = w²
        step(&mut heavy, &mut momentum, |w| 2.0 * w);
        ws.push(plain.value.item());
        hs.push(heavy.value.item());
    }
    assert!(close(&ws, &[0.8, 0.64, 0.512], 1e-6));
    assert!(close(&hs, &[0.8, 0.54, 0.302], 1e-6));
    // @end
}

#[test]
fn dampening_and_a_cold_start() {
    // @listing dampening-test
    let mut w = scalar(0.0);
    let mut opt = Sgd::new(0.1).momentum(0.9).dampening(0.9);
    let mut vs = vec![];
    for _ in 0..10 {
        step(&mut w, &mut opt, |_| 1.0); // a steady slope
        vs.push(velocity(&opt, &w));
    }
    assert!(close(&vs, &[1.0; 10], 1e-6)); // an average of 1s: the step is lr
    // A velocity started at 0 instead of at the first gradient:
    let mut cold = vec![];
    let mut v = 0.0f32;
    for _ in 0..10 {
        v = 0.9 * v + 0.1 * 1.0;
        cold.push(v);
    }
    let at = [cold[0], cold[1], cold[9]];
    assert!(close(&at, &[0.1, 0.19, 0.6513], 1e-4));
    assert!((cold[9] / (1.0 - 0.9f32.powi(10)) - 1.0).abs() < 1e-5); // corrected
    // @end
}

#[test]
fn classical_momentum() {
    // @listing classical-test
    let (mut a, mut b) = (scalar(1.0), scalar(1.0));
    let mut torch = Sgd::new(0.1).momentum(0.9);
    let mut classical = ClassicalSgd::new(0.1, 0.9);
    let mut moves = vec![];
    for t in 1..=6 {
        if t == 6 {
            torch.groups_mut()[0].lr = 0.01; // a schedule cuts the rate
            classical.groups_mut()[0].lr = 0.01;
        }
        let before = (a.value.item(), b.value.item());
        step(&mut a, &mut torch, |w| w);
        step(&mut b, &mut classical, |w| w);
        moves.push((before.0 - a.value.item(), before.1 - b.value.item()));
    }
    assert!(moves[..5].iter().all(|(x, y)| (x - y).abs() < 1e-6)); // the same
    let (m4, m5) = (moves[4], moves[5]);
    assert!(close(&[m4.0, m5.0, m5.1], &[0.256, 0.0227, 0.2301], 1e-4));
    // @end
}

/// The mean and variance of `xs`, in f64.
fn moments(xs: &[f32]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean, var)
}

#[test]
fn momentum_averages_noise() {
    // @listing noise-test
    let directions = |mut opt: Sgd| {
        let (mut rng, mut w) = (Rng::new(16), scalar(0.0));
        let mut out = vec![];
        for _ in 0..10_100 {
            let (before, noise) = (w.value.item(), rng.normal());
            step(&mut w, &mut opt, |_| 1.0 + noise); // slope 1, plus noise
            out.push((before - w.value.item()) / 0.1); // the step, divided by lr
        }
        moments(&out[100..]) // after the velocity has warmed up
    };
    let (m0, v0) = directions(Sgd::new(0.1));
    let (m1, v1) = directions(Sgd::new(0.1).momentum(0.9).dampening(0.9));
    let (m2, v2) = directions(Sgd::new(0.1).momentum(0.9));
    assert!((m0 - 1.0049).abs() < 1e-4 && (v0 - 0.9896).abs() < 1e-4);
    assert!((m1 - 1.0051).abs() < 1e-4 && (v1 - 0.0489).abs() < 1e-4);
    assert!((m2 - 10.0513).abs() < 1e-3 && (v2 - 4.8862).abs() < 1e-3);
    // @end
}

#[test]
fn tuned_heavy_ball() {
    // @listing tuning-test
    let (lr, mu) = heavy_ball_tuning(1.0, 100.0);
    assert!((lr - 4.0 / 121.0).abs() < 1e-7 && (mu - 0.66942).abs() < 1e-5);
    let settles = |mut opt: Sgd| {
        let mut w = Parameter::new(Tensor::from_vec(vec![1.0, 1.0], &[2]));
        let mut last_above = 0;
        for t in 1..=1000 {
            let (x, y) = (w.value.get(&[0]), w.value.get(&[1]));
            w.grad = Some(Tensor::from_vec(vec![x, 100.0 * y], &[2]));
            opt.step(&mut [&mut w]);
            let (x, y) = (w.value.get(&[0]), w.value.get(&[1]));
            let loss = 0.5 * x * x + 50.0 * y * y;
            if loss.is_nan() || loss >= 1e-6 {
                last_above = t;
            }
        }
        last_above + 1
    };
    assert_eq!(settles(Sgd::new(lr).momentum(mu)), 69);
    assert_eq!(settles(Sgd::new(0.0195).momentum(0.9)), 172); // Section 16.6
    let r = 0.9f32.sqrt(); // μ = 0.9 is the tuning for κ = ((1 + r)/(1 - r))²
    assert!((((1.0 + r) / (1.0 - r)).powi(2) - 1441.0).abs() < 1.0);
    // @end
}

#[test]
fn polyak_averaging() {
    // @listing averaged-test
    let mut rng = Rng::new(16);
    let mut w = scalar(0.0); // start at the minimum of ½w²
    let mut opt = Averaged::new(Sgd::new(0.1));
    let mut squares = 0.0f32;
    for t in 0..2000 {
        let noise = rng.normal();
        step(&mut w, &mut opt, |w| w + noise); // a noisy gradient
        if t >= 1000 {
            squares += w.value.item().powi(2) / 1000.0;
        }
    }
    let mean = opt.mean(w.id()).unwrap().item();
    assert!((squares.sqrt() - 0.2443).abs() < 1e-3); // the iterates wander
    assert!((mean - 0.0307).abs() < 1e-3); // their mean is 8 times closer
    // @end
}

#[test]
fn lookahead_calms_momentum() {
    // @listing lookahead-test
    let run = |opt: &mut dyn Optimizer| {
        let mut w = scalar(1.0);
        let mut losses = vec![];
        for _ in 0..50 {
            step(&mut w, opt, |w| w); // f(w) = ½w²
            losses.push(0.5 * w.value.item().powi(2));
        }
        let worst = losses[5..].iter().cloned().fold(0.0, f32::max);
        (worst, losses[49])
    };
    let (worst, last) = run(&mut Sgd::new(0.1).momentum(0.9));
    assert!((worst - 0.1822).abs() < 1e-4 && (last - 2.2231e-3).abs() < 1e-7);
    let mut look = Lookahead::new(Sgd::new(0.1).momentum(0.9), 5, 0.5);
    let (worst, last) = run(&mut look);
    assert!((worst - 0.1186).abs() < 1e-4 && (last - 1.0427e-7).abs() < 1e-10);
    // @end
}
