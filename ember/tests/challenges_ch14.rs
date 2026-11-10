//! Tests for the Chapter 14 challenges.
#![allow(clippy::approx_constant)] // ln 2 and ln 3 to four places
use ember::{
    Linear, Module, ReLU, Rng, Sequential, Tanh, Tape, cross_entropy, fit_early,
    gd_step, held_out_loss, l2_penalty, l2_step, lr_range, moons, mse_loss,
    spirals_k, toy,
};

// The chapter's examples, as modules (their `main`s go unused).
#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;
#[path = "../examples/xor.rs"]
#[allow(dead_code)]
mod xor;

fn count(model: &mut dyn Module) -> usize {
    model.parameters().iter().map(|p| p.value.numel()).sum()
}

#[test]
fn predict_the_numbers() {
    let mut rng = Rng::new(14);
    let (train, _) = spirals::data(&mut rng);
    let mut model = spirals::mlp(16, &mut rng);
    assert_eq!(count(&mut model), 16 * 16 + 6 * 16 + 2); // 354
    let tape = Tape::new();
    let loss = cross_entropy(model.forward(tape.constant(train.0.clone())), &train.1);
    assert_eq!(tape.len(), 22);
    assert!((loss.value().item() - 0.6965).abs() < 1e-4); // ln 2 = 0.6931
    let mut wide = spirals::mlp(64, &mut rng);
    let tape = Tape::new();
    wide.forward(tape.constant(train.0.clone()));
    assert_eq!(tape.len(), 21); // the same graph, without the loss
    assert_eq!(count(&mut wide), 4482);
    assert_eq!(200 * (2 * 16 + 16 * 16 + 16 * 2), 64_000); // multiply-adds
    assert_eq!((200usize.div_ceil(32), 200 / 32), (7, 6)); // batches of 32
}

// @listing three-test
#[test]
fn three_spirals() {
    let two = spirals_k(100, 2, 1.5, 0.05, &mut Rng::new(14));
    let toy = toy::spirals(100, 1.5, 0.05, &mut Rng::new(14));
    assert_eq!(two.0.to_vec(), toy.0.to_vec()); // the same points
    let mut rng = Rng::new(14);
    let data = spirals_k(25, 3, 1.0, 0.05, &mut rng);
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(2, 16, &mut rng)),
        Box::new(Tanh),
        Box::new(Linear::new(16, 16, &mut rng)),
        Box::new(Tanh),
        Box::new(Linear::new(16, 3, &mut rng)),
    ]);
    assert_eq!(count(&mut model), 371);
    let mut step =
        || gd_step(&mut model, &data.0, 1.0, |z| cross_entropy(z, &data.1));
    let losses: Vec<f32> = (0..500).map(|_| step()).collect();
    assert!((losses[0] - 1.1113).abs() < 1e-4); // ln 3 = 1.0986
    assert!((losses[499] - 0.2052).abs() < 1e-4);
    assert!((spirals::accuracy(&model, &data) - 0.8933).abs() < 1e-4); // 67 of 75
}
// @end

#[test]
fn two_moons() {
    let mut rng = Rng::new(14);
    let data = moons(50, 0.1, &mut rng);
    let fit = |model: &mut dyn Module| -> Vec<f32> {
        let (x, labels) = &data;
        (0..500)
            .map(|_| gd_step(model, x, 1.0, |z| cross_entropy(z, labels)))
            .collect()
    };
    let mut line = Linear::new(2, 2, &mut rng);
    let losses = fit(&mut line);
    assert!((losses[499] - 0.2759).abs() < 1e-4);
    assert_eq!(spirals::accuracy(&line, &data), 0.85);
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(2, 8, &mut rng)),
        Box::new(Tanh),
        Box::new(Linear::new(8, 2, &mut rng)),
    ]);
    let losses = fit(&mut model);
    assert!((losses[499] - 0.0188).abs() < 1e-4);
    assert_eq!(spirals::accuracy(&model, &data), 0.99);
}

// @listing act-test
/// The loss after 500 steps of a 2-2-1 network on XOR, with tanh or ReLU.
fn xor_loss(relu: bool, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let hidden = Linear::new(2, 2, &mut rng);
    let act: Box<dyn Module> = if relu { Box::new(ReLU) } else { Box::new(Tanh) };
    let out = Linear::new(2, 1, &mut rng);
    let mut model = Sequential::new(vec![Box::new(hidden), act, Box::new(out)]);
    let (x, y) = toy::xor();
    (0..500)
        .map(|_| xor::step(&mut model, &x, &y, 0.3))
        .last()
        .unwrap()
}

#[test]
fn tanh_or_relu() {
    let fits = |relu| (0..20).filter(|&seed| xor_loss(relu, seed) < 0.01).count();
    assert_eq!((fits(false), fits(true)), (20, 3));
    let stuck: Vec<f32> = (0..4).map(|seed| xor_loss(true, seed)).collect();
    let want = [1.0 / 6.0, 0.25, 0.25, 1.0 / 6.0];
    assert!(stuck.iter().zip(want).all(|(l, w)| (l - w).abs() < 1e-3));
}
// @end

// @listing l2-test
#[test]
fn an_l2_penalty() {
    // The penalty's gradient is 2·lambda·w, and biases get none.
    let mut model = xor::mlp(4, &mut Rng::new(0));
    let tape = Tape::new();
    l2_penalty(&tape, &mut model, 0.1).backward_into(&mut model.parameters());
    let ps = model.parameters();
    let want = ps[0].value.mul_scalar(0.2).to_vec();
    assert_eq!(ps[0].grad.as_ref().unwrap().to_vec(), want);
    assert!(ps[1].grad.is_none());
    // Training: a larger lambda, smaller weights and a worse fit.
    let (x, y) = toy::xor();
    let run = |lambda| {
        let mut model = xor::mlp(4, &mut Rng::new(0));
        let fit = (0..1000)
            .map(|_| l2_step(&mut model, &x, &y, 0.1, lambda))
            .last();
        let ps = model.parameters();
        let squares = ps.iter().filter(|p| p.value.ndim() == 2);
        (
            fit.unwrap(),
            squares.map(|p| p.value.powf(2.0).sum_all()).sum::<f32>(),
        )
    };
    let near = |(a, b): (f32, f32), (c, d): (f32, f32)| {
        (a - c).abs() < 1e-4 && (b - d).abs() < 1e-3
    };
    assert!(near(run(0.0), (0.0, 13.122)));
    assert!(near(run(0.001), (0.0001, 12.135)));
    assert!(near(run(0.01), (0.0073, 8.425)));
    assert!(near(run(0.1), (0.25, 0.0)));
}
// @end

// @listing early-test
#[test]
fn early_stopping() {
    let mut rng = Rng::new(14);
    let train = moons(15, 0.3, &mut rng); // 30 points to learn from
    let held = moons(100, 0.3, &mut rng);
    let mut model = spirals::mlp(32, &mut rng);
    let stopped = fit_early(&mut model, (&train, &held), 1.0, (25, 4, 2000));
    assert_eq!((stopped.best_step, stopped.steps), (200, 300));
    assert!((stopped.best_loss - 0.2985).abs() < 1e-4);
    assert!((held_out_loss(&model, &held) - 0.2985).abs() < 1e-4); // restored
    assert_eq!(spirals::accuracy(&model, &held), 0.905);
}
// @end

#[test]
fn learning_rate_range_test() {
    let (x, y) = toy::xor();
    let mut model = xor::mlp(4, &mut Rng::new(0));
    let lrs = (1e-3, 10.0);
    let curve = lr_range(
        &mut model,
        &x,
        |p| mse_loss(p, p.tape().constant(y.clone())),
        lrs,
        100,
    );
    let lowest = (0..100)
        .min_by(|&a, &b| curve[a].1.total_cmp(&curve[b].1))
        .unwrap();
    assert_eq!(lowest, 81);
    let (lr, loss) = curve[81];
    assert!((lr - 1.874).abs() < 1e-3 && (loss - 0.1373).abs() < 1e-4);
    assert!(curve[82].1 > 1.2 && curve[83].1 > 18.0); // and then it blows up
    assert!(curve[99].1.is_nan());
}

#[test]
fn moons_shape() {
    let (x, labels) = moons(10, 0.0, &mut Rng::new(1));
    assert_eq!(x.shape(), [20, 2]);
    assert!(labels[..10].iter().all(|&c| c == 0));
    let row = |i: usize| (x.get(&[i, 0]), x.get(&[i, 1]));
    let (a, b) = row(0);
    assert!((a.hypot(b) - 1.0).abs() < 1e-5 && b >= 0.0); // on the upper unit circle
    let (a, b) = row(10);
    assert!(((a - 1.0).hypot(b - 0.5) - 1.0).abs() < 1e-5 && b <= 0.5);
}
