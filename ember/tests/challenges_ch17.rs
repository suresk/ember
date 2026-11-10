//! Tests for the Chapter 17 challenges.
use ember::{
    Adagrad, Adam, AmsGrad, CenteredRmsProp, Lion, Optimizer, Parameter, RmsProp,
    Rng, Tensor, toy,
};

#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;

fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

/// `steps` steps of `opt` on a one-number parameter from `start`, with the
/// gradient `grad(t)` at step t = 1, 2, … Returns how far each step moved it
/// (positive = downhill for a positive gradient).
fn moves(
    opt: &mut dyn Optimizer,
    start: f32,
    grad: impl Fn(usize) -> f32,
    steps: usize,
) -> Vec<f32> {
    let mut w = Parameter::new(Tensor::from_vec(vec![start], &[1]));
    (1..=steps)
        .map(|t| {
            let before = w.value.item();
            w.grad = Some(Tensor::from_vec(vec![grad(t)], &[1]));
            opt.step(&mut [&mut w]);
            before - w.value.item()
        })
        .collect()
}

#[test]
fn two_adam_steps_by_hand() {
    // @listing predict-test
    let mut opt = Adam::new(0.1);
    let mut w = Parameter::new(Tensor::from_vec(vec![0.0], &[1]));
    let mut ws = vec![];
    for g in [2.0, -1.0] {
        w.grad = Some(Tensor::from_vec(vec![g], &[1]));
        opt.step(&mut [&mut w]);
        ws.push(w.value.item());
    }
    let state = opt.state(w.id()).unwrap();
    let (m, v) = (state.exp_avg.item(), state.exp_avg_sq.item());
    assert!(close(&[m, v], &[0.08, 0.004996], 1e-7));
    assert!(close(&ws, &[-0.1, -0.126634], 1e-6)); // still going down
    // @end
}

#[test]
fn epsilon_and_tiny_gradients() {
    // @listing eps-test
    let first = |g: f32| moves(&mut Adam::new(0.1), 0.0, |_| g, 1)[0];
    let steps = [1e-6, 1e-8, 1e-10].map(first);
    assert!(close(&steps, &[0.0990099, 0.05, 0.000990099], 1e-7));
    // @end
}

#[test]
fn adam_without_momentum() {
    // @listing beta1-test
    let rms = moves(&mut RmsProp::new(0.01), 0.0, |_| 1.0, 100);
    let adam = moves(&mut Adam::new(0.01).betas((0.0, 0.99)), 0.0, |_| 1.0, 100);
    let at = |m: &[f32]| [1, 2, 3, 10, 100].map(|t| m[t - 1]);
    let want = [0.1, 0.0708882, 0.0580249, 0.0323393, 0.0125594];
    assert!(close(&at(&rms), &want, 1e-6)); // 0.01 / √(1 − 0.99^t)
    assert!(close(&at(&adam), &[0.01; 5], 1e-8));
    // @end
}

#[test]
fn centered_rmsprop() {
    // @listing centered-test
    let mut rng = Rng::new(17);
    let noisy: Vec<f32> = (0..2000).map(|_| 1.0 + 0.1 * rng.normal()).collect();
    let mean_step = |opt: &mut dyn Optimizer| {
        let m = moves(opt, 0.0, |t| noisy[t - 1], 2000);
        m[1000..].iter().sum::<f32>() / 1000.0 / 0.001 // in units of lr
    };
    let plain = mean_step(&mut RmsProp::new(0.001));
    let centered = mean_step(&mut CenteredRmsProp::new(0.001));
    assert!(close(&[plain, centered], &[0.9932, 10.1436], 1e-3));
    // A steady gradient of 1: the variance goes to 0, and the step grows.
    let steady = moves(&mut CenteredRmsProp::new(0.001), 0.0, |_| 1.0, 1000);
    let at = [1, 10, 100, 1000].map(|t| steady[t - 1] / 0.001);
    assert!(close(&at, &[10.0504, 3.4006, 2.0759, 152.12], 1e-2));
    // @end
}

#[test]
fn amsgrad_never_forgets() {
    // @listing amsgrad-test
    let grad = |t| if t <= 10 { 1.0 } else { 0.01 }; // a burst, then small
    let at = |m: Vec<f32>| [1, 11, 100, 1000, 3000, 5000].map(|t| m[t - 1] / 0.001);
    let adam = at(moves(&mut Adam::new(0.001), 0.0, grad, 5000));
    let ams = at(moves(&mut AmsGrad::new(0.001), 0.0, grad, 5000));
    let want_adam = [1.0, 0.8977, 0.0325, 0.1297, 0.3997, 0.7714];
    let want_ams = [1.0, 0.8973, 0.0311, 0.0797, 0.0977, 0.0999];
    assert!(close(&adam, &want_adam, 1e-4) && close(&ams, &want_ams, 1e-4));
    // @end
}

#[test]
fn adagrad_slows_down() {
    // @listing adagrad-test
    let steady = moves(&mut Adagrad::new(0.01), 0.0, |_| 1.0, 100);
    let at = [1, 2, 4, 100].map(|t| steady[t - 1]);
    assert!(close(&at, &[0.01, 0.0070711, 0.005, 0.001], 1e-6)); // 0.01/√t
    // @end
}

#[test]
fn lion_takes_sign_steps() {
    // @listing lion-test
    // 20 gradients of +1, then −1s: every step is exactly ±lr.
    let grad = |t| if t <= 20 { 1.0 } else { -1.0 };
    let steps = moves(&mut Lion::new(0.01), 0.0, grad, 30);
    assert!(steps[..27].iter().all(|&s| (s - 0.01).abs() < 1e-7)); // down to step 27
    assert!(steps[27..].iter().all(|&s| (s + 0.01).abs() < 1e-7)); // turned
    let run = |lr: f32, steps: usize| {
        let mut rng = Rng::new(14);
        let small = toy::spirals(25, 1.5, 0.05, &mut rng); // Section 14.6's short run
        let mut model = spirals::mlp(16, &mut rng);
        let mut opt = Lion::new(lr);
        let losses: Vec<f32> = (0..steps)
            .map(|_| spirals::step_with(&mut model, &small, &mut opt))
            .collect();
        let state: usize = opt.buffers.state.values().map(|(_, [m])| m.numel()).sum();
        (losses.iter().position(|&l| l < 0.01).map(|i| i + 1), state)
    };
    assert_eq!(run(0.01, 191), (Some(191), 354)); // one buffer: 354 numbers
    // @end
}
