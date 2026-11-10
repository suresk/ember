//! Tests for the Chapter 18 challenges.
use ember::{
    Adam, ConstantLr, CosineLr, CosineRestarts, InverseSqrt, LinearWarmup,
    LrScheduler, NormWatch, OneCycle, Optimizer, Parameter, ReduceLrOnPlateau, Rng,
    Sgd, Tensor, clip_grad_norm, clip_grad_value, toy,
};

#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;

fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

fn tensor(xs: &[f32]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), &[xs.len()])
}

#[test]
fn warmup_then_cosine_by_hand() {
    // @listing predict-test
    let schedule = LinearWarmup::new(10, CosineLr::new(90, 0.001));
    let at = [0, 9, 10, 55, 99].map(|t| schedule.lr(0.1, t));
    assert!(close(&at, &[0.01, 0.1, 0.1, 0.0505, 0.00103], 1e-5));
    // @end
}

#[test]
fn clip_by_value() {
    // @listing value-test
    let mut a = Parameter::new(tensor(&[0.0, 0.0]));
    let mut b = Parameter::new(tensor(&[0.0]));
    a.grad = Some(tensor(&[3.0, 4.0]));
    b.grad = Some(tensor(&[12.0]));
    clip_grad_value(&mut [&mut a, &mut b], 1.0);
    let (ga, gb) = (a.grad.as_ref().unwrap(), b.grad.as_ref().unwrap());
    assert_eq!((ga.to_vec(), gb.to_vec()), (vec![1.0, 1.0], vec![1.0]));
    // The angle between [3, 4, 12] and [1, 1, 1]: cos = 19 / (13·√3).
    let cos = 19.0 / (13.0 * 3f32.sqrt());
    assert!(
        (cos - 0.8438).abs() < 1e-4 && (cos.acos().to_degrees() - 32.46).abs() < 0.01
    );
    // @end
    let norm = clip_grad_norm(&mut [&mut a, &mut b], f32::INFINITY);
    assert!((norm - 3f32.sqrt()).abs() < 1e-6);
}

#[test]
fn inverse_square_root() {
    // @listing isqrt-test
    let schedule = InverseSqrt { warmup: 100 };
    let at = [0, 49, 99, 399, 9999].map(|t| schedule.lr(0.1, t));
    assert!(close(&at, &[0.001, 0.05, 0.1, 0.05, 0.01], 1e-7));
    // @end
}

#[test]
fn cosine_with_restarts() {
    // @listing restarts-test
    let schedule = CosineRestarts {
        first: 10,
        mult: 2,
        min_lr: 0.0,
    };
    let at = [0, 5, 9, 10, 20, 29, 30, 50, 69, 70].map(|t| schedule.lr(0.1, t));
    let want = [
        0.1, 0.05, 0.00245, 0.1, 0.05, 0.00062, 0.1, 0.05, 0.00015, 0.1,
    ];
    assert!(close(&at, &want, 1e-5));
    // @end
}

#[test]
fn one_cycle() {
    // @listing one-cycle-test
    let schedule = OneCycle { steps: 100 };
    let at = [0, 14, 29, 64, 99].map(|t| schedule.lr(0.1, t));
    assert!(close(&at, &[0.004, 0.0494, 0.1, 0.05, 4e-7], 1e-4));
    assert!((at[4] - 4e-7).abs() < 1e-12);
    let mut rng = Rng::new(14);
    let small = toy::spirals(25, 1.5, 0.05, &mut rng);
    let mut model = spirals::mlp(16, &mut rng);
    let mut opt = Adam::new(0.3);
    let (losses, _) =
        spirals::train_with(&mut model, &small, &mut opt, &schedule, (100, f32::MAX));
    let below = losses.iter().position(|&l| l < 0.01).map(|i| i + 1);
    let worst = losses.iter().cloned().fold(0.0, f32::max);
    assert_eq!((below, (worst * 1000.0).round()), (Some(52), 695.0));
    // @end
    assert!((losses[99] - 0.00194).abs() < 2e-5);
}

#[test]
fn reduce_on_plateau() {
    // @listing plateau-test
    let mut opt = Sgd::new(0.1);
    let mut plateau = ReduceLrOnPlateau::new(0.1, 2);
    let metrics = [
        1.0, 0.9, 0.9, 0.95, 0.9, 0.89995, 0.8, 0.85, 0.81, 0.8, 0.79,
    ];
    let mut cuts = vec![];
    for (k, &m) in metrics.iter().enumerate() {
        if plateau.step(&mut opt, m) {
            cuts.push(k + 1);
        }
    }
    assert_eq!(cuts, [5, 10]); // the 5th and 10th values
    assert!((opt.groups()[0].lr - 0.001).abs() < 1e-9);
    // @end
}

#[test]
fn watching_the_norm() {
    // @listing watch-test
    let flags = |warmup: usize| {
        let mut rng = Rng::new(14);
        let small = toy::spirals(25, 1.5, 0.05, &mut rng);
        let mut model = spirals::mlp(16, &mut rng);
        let mut opt = Adam::new(0.3);
        let schedule = LinearWarmup::new(warmup, ConstantLr);
        let run = (30, f32::MAX);
        let (_, norms) =
            spirals::train_with(&mut model, &small, &mut opt, &schedule, run);
        let mut watch = NormWatch::new(10.0, 5);
        let flagged = norms.iter().map(|&n| watch.check(n));
        let steps: Vec<usize> = flagged
            .enumerate()
            .filter(|f| f.1)
            .map(|f| f.0 + 1)
            .collect();
        (steps, norms[0], norms[1])
    };
    let (cold, n1, n2) = flags(1);
    assert!(close(&[n1, n2], &[0.0275, 1.21], 0.01)); // 44 times larger
    assert_eq!((cold, flags(20).0), (vec![2], vec![]));
    // @end
    let mut watch = NormWatch::new(10.0, 3);
    let seen: Vec<bool> = [1.0, 1.2, 0.9, f32::NAN, 1.1, 0.05]
        .map(|n| watch.check(n))
        .to_vec();
    assert_eq!(seen, [false, false, false, true, false, true]);
}
