//! The examples printed in Chapter 21 of the guided edition: inverted dropout, its
//! mask on the tape, training and evaluation modes, `evaluate`'s guard, the two
//! classic mode bugs, and dropout narrowing the gap on a small spirals problem.
use ember::{
    Adam, DataLoader, Dropout, EvalMode, Module, Parameter, Rng, Sequential, Tape,
    Tensor, TensorDataset, evaluate, gradcheck, toy, toy::Labeled, train_step,
};
use std::cell::Cell;
use std::rc::Rc;

#[path = "../examples/mnist_train.rs"]
#[allow(dead_code)]
mod mnist_train;

use mnist_train::mlp_dropout;

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn ones(n: usize) -> Tensor {
    Tensor::from_vec(vec![1.0; n], &[n])
}

fn bits(model: &mut dyn Module) -> Vec<u32> {
    let params = model.parameters();
    params
        .iter()
        .flat_map(|p| p.value.to_vec())
        .map(f32::to_bits)
        .collect()
}

fn mean(x: &Tensor) -> f32 {
    x.sum_all() / x.numel() as f32
}

#[test]
fn inverted_dropout_keeps_the_mean() {
    // @listing mean-test
    let drop = Dropout::new(0.5, 21);
    let tape = Tape::new();
    let y = drop.forward(tape.constant(ones(10))).value();
    assert_eq!(y.to_vec(), [2., 0., 0., 0., 2., 2., 0., 2., 2., 0.]);
    assert_eq!(mean(&y), 1.0); // five of ten kept, twice as large
    let y = drop.forward(tape.constant(ones(10_000))).value();
    let kept = y.to_vec().iter().filter(|&&v| v > 0.0).count();
    assert_eq!((kept, mean(&y)), (5_060, 1.012)); // the stream goes on
    // @end
}

#[test]
fn the_mask_is_a_constant_on_the_tape() {
    let mut rng = Rng::new(21); // the draws behind the mask: keep where u < 0.5
    let u: Vec<f32> = (0..6)
        .map(|_| (rng.uniform() * 100.0).round() / 100.0)
        .collect();
    assert_eq!(u, [0.03, 0.92, 0.52, 0.65, 0.09, 0.46]);
    // @listing backward-test
    let drop = Dropout::new(0.5, 21);
    let tape = Tape::new();
    let x = tape.leaf(t(&[1., 2., 3., 4., 5., 6.], &[6]));
    let c = tape.constant(t(&[10., 20., 30., 40., 50., 60.], &[6]));
    let y = drop.forward(x);
    let loss = (y * c).sum_all();
    assert_eq!(y.value().to_vec(), [2., 0., 0., 0., 10., 12.]); // x ⊙ mask
    assert_eq!((loss.value().item(), tape.len()), (1240.0, 6));
    let x_bar = loss.backward().get(x).unwrap().to_vec();
    assert_eq!(x_bar, [20., 0., 0., 0., 100., 120.]); // c ⊙ mask
    // @end
    // A random function fails a gradient check: every evaluation draws a new mask.
    let x = t(&[1., 2., 3., 4., 5., 6.], &[6]);
    let tol = ember::Tolerance::default();
    assert!(
        gradcheck(|xs| drop.forward(xs[0]), std::slice::from_ref(&x), tol).is_err()
    );
    let mask = drop.mask(&[6]); // a fixed mask is an ordinary product
    let fixed =
        gradcheck(|xs| xs[0] * xs[0].tape().constant(mask.clone()), &[x], tol);
    assert!(fixed.is_ok());
}

#[test]
fn same_seed_same_masks() {
    // @listing seed-test
    let (a, b) = (Dropout::new(0.5, 21), Dropout::new(0.5, 21));
    let first = a.mask(&[10]).to_vec();
    assert_eq!(first, b.mask(&[10]).to_vec()); // the same seed, the same mask
    assert_ne!(first, a.mask(&[10]).to_vec()); // a's next one differs
    assert_ne!(first, Dropout::new(0.5, 22).mask(&[10]).to_vec());
    // @end
}

/// A module without parameters that remembers the last mode it was given.
struct Probe(Rc<Cell<bool>>);

impl Module for Probe {
    fn forward<'t>(&self, x: ember::Var<'t>) -> ember::Var<'t> {
        x
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        vec![]
    }
    fn train(&mut self, on: bool) {
        self.0.set(on);
    }
}

#[test]
fn eval_mode_is_the_identity() {
    // @listing modes-test
    let mut rng = Rng::new(21);
    let mut model = mlp_dropout(&[3, 4, 2], 0.5, &mut rng); // 3 → 4 → 2
    let x = Tensor::randn(&[2, 3], &mut rng);
    let run = |model: &Sequential| {
        let tape = Tape::new();
        let y = model.forward(tape.constant(x.clone())).value().to_vec();
        (y, tape.len())
    };
    let (a, nodes) = run(&model);
    assert!(a != run(&model).0 && nodes == 16); // a new mask each call
    model.train(false); // Sequential passes it on to the Dropout
    let (a, nodes) = run(&model);
    assert!(a == run(&model).0 && nodes == 14); // no mask, no product
    model.train(true);
    assert_ne!(run(&model).0, a);
    // @end
}

#[test]
fn evaluate_switches_and_restores() {
    let mut rng = Rng::new(21);
    let mut model = mlp_dropout(&[2, 8, 2], 0.5, &mut rng);
    let (x, labels) = toy::spirals(20, 1.0, 0.05, &mut rng);
    let ds = TensorDataset::new(x, labels);
    // @listing evaluate-test
    let first = evaluate(&mut model, &mut DataLoader::new(&ds, 10));
    assert_eq!(first, evaluate(&mut model, &mut DataLoader::new(&ds, 10)));
    let mode = Rc::new(Cell::new(true));
    model.layers.push(Box::new(Probe(mode.clone())));
    {
        let guarded = EvalMode::new(&mut model);
        assert!(!mode.get()); // evaluation mode inside
        let _ = guarded.forward(Tape::new().constant(Tensor::zeros(&[1, 2])));
    }
    assert!(mode.get()); // training mode again
    // @end
    let bad = TensorDataset::new(Tensor::zeros(&[4, 3]), vec![0; 4]); // 3 ≠ 2 inputs
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        evaluate(&mut model, &mut DataLoader::new(&bad, 4))
    }));
    assert!(result.is_err() && mode.get()); // restored while the panic unwound
}

/// Section 21.7's problem: 50 spiral points per arm to train on, 200 per arm to
/// validate on, and a 2–16–16–2 ReLU network with dropout `p` after each hidden
/// layer, trained for `steps` full-batch steps of Adam at 0.03.
fn spirals_run(p: f32, steps: usize) -> (Sequential, Labeled, Labeled) {
    let mut rng = Rng::new(21);
    let train = toy::spirals(50, 1.5, 0.05, &mut rng);
    let val = toy::spirals(200, 1.5, 0.05, &mut rng);
    let mut model = mlp_dropout(&[2, 16, 16, 2], p, &mut rng);
    let mut opt = Adam::new(0.03);
    for _ in 0..steps {
        train_step(&mut model, &train, &mut opt, f32::INFINITY);
    }
    (model, train, val)
}

/// `evaluate` on a labeled set, rounded to 3 decimals.
fn measure(model: &mut Sequential, (x, labels): &Labeled) -> (f32, f32) {
    let ds = TensorDataset::new(x.clone(), labels.clone());
    let (loss, accuracy) = evaluate(model, &mut DataLoader::new(&ds, 400));
    let r = |v: f32| (v * 1000.0).round() / 1000.0;
    (r(loss), r(accuracy))
}

/// The accuracy of `model` as it is, in whatever mode it is in.
fn accuracy_as_is(model: &Sequential, (x, labels): &Labeled) -> f32 {
    let tape = Tape::new();
    let logits = model.forward(tape.constant(x.clone())).value();
    ember::accuracy(&logits, labels)
}

#[test]
fn the_gap_without_dropout() {
    // @listing gap-test
    let (mut plain, train, val) = spirals_run(0.0, 300);
    assert_eq!(measure(&mut plain, &train), (0.032, 0.99)); // (loss, accuracy)
    assert_eq!(measure(&mut plain, &val), (0.595, 0.905)); // loss gap 0.563
    // @end
}

#[test]
fn dropout_narrows_the_gap() {
    // @listing dropout-gap-test
    let (mut dropped, train, val) = spirals_run(0.1, 300);
    assert_eq!(measure(&mut dropped, &train), (0.106, 0.98));
    assert_eq!(measure(&mut dropped, &val), (0.235, 0.94)); // loss gap 0.129
    // @end
    // @listing train-mode-test
    let a = accuracy_as_is(&dropped, &val); // dropout still on: a random network
    let b = accuracy_as_is(&dropped, &val);
    assert_eq!((a, b), (0.9, 0.8925)); // 360 and 357 of 400; evaluate: 376
    // @end
}

#[test]
fn forgetting_to_switch_back() {
    // @listing forgot-test
    let mut rng = Rng::new(21);
    let train = toy::spirals(50, 1.5, 0.05, &mut rng);
    let mut model = mlp_dropout(&[2, 16, 16, 2], 0.1, &mut rng.clone());
    let mut plain = mlp_dropout(&[2, 16, 16, 2], 0.0, &mut rng);
    model.train(false); // an evaluation that never switched back
    let (mut a, mut b) = (Adam::new(0.03), Adam::new(0.03));
    for _ in 0..50 {
        train_step(&mut model, &train, &mut a, f32::INFINITY);
        train_step(&mut plain, &train, &mut b, f32::INFINITY);
    }
    assert_eq!(bits(&mut model), bits(&mut plain)); // no dropout at all
    // @end
}

#[test]
fn dropout_leaves_the_weights_alone() {
    let mut plain = mnist_train::mlp(&mut Rng::new(20));
    let mut same = mlp_dropout(&[784, 128, 10], 0.0, &mut Rng::new(20));
    let mut dropped = mlp_dropout(&[784, 128, 10], 0.2, &mut Rng::new(20));
    assert_eq!(bits(&mut plain), bits(&mut same));
    assert_eq!(bits(&mut plain), bits(&mut dropped));
    let names: Vec<String> = dropped
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(names, ["0.weight", "0.bias", "3.weight", "3.bias"]);
}
