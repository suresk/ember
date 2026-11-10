//! The examples printed in Chapter 15 of the guided edition: an `Optimizer` trait,
//! parameter groups, SGD and weight decay.
use ember::{
    Linear, Module, Optimizer, ParamGroup, ParamId, Parameter, Rng, Sgd, Tape,
    Tensor, l2_step, mse_loss, toy,
};

// The examples are compiled into this test as modules (their `main`s go unused).
#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;
#[path = "../examples/xor.rs"]
#[allow(dead_code)]
mod xor;

fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

/// Section 14.2's step on the mean squared error, with the update handed to `opt`.
fn mse_step(model: &mut impl Module, x: &Tensor, y: &Tensor, opt: &mut Sgd) -> f32 {
    model.zero_grad();
    let loss = {
        let tape = Tape::new();
        let pred = model.forward(tape.constant(x.clone()));
        let loss = mse_loss(pred, tape.constant(y.clone()));
        loss.backward_into(&mut model.parameters());
        loss.value().item()
    };
    opt.step(&mut model.parameters());
    loss
}

/// Every number of a model, in `parameters()` order.
fn numbers(model: &mut impl Module) -> Vec<f32> {
    model
        .parameters()
        .iter()
        .flat_map(|p| p.value.to_vec())
        .collect()
}

#[test]
fn sgd_reproduces_section_11_6() {
    // @listing affine
    // Section 11.6's model and data: y = 2x + 1 at x = 1, 2, from w = b = 0.
    let mut model = Linear::from_tensors(Tensor::zeros(&[1, 1]), Tensor::zeros(&[1]));
    let x = Tensor::from_vec(vec![1.0, 2.0], &[2, 1]);
    let y = Tensor::from_vec(vec![3.0, 5.0], &[2, 1]);
    let mut opt = Sgd::new(0.1);
    let losses: Vec<f32> = (0..4)
        .map(|_| mse_step(&mut model, &x, &y, &mut opt))
        .collect();
    assert!(close(&losses, &[17.0, 1.685, 0.16925, 0.019169], 1e-5));
    let (w, b) = (model.weight.value.item(), model.bias.value.item());
    assert!(close(&[w, b], &[1.8819, 1.1496], 1e-4));
    assert_eq!(opt.state(model.weight.id()).unwrap().steps, 4);
    // @end
}

#[test]
fn a_parameter_without_a_gradient_is_skipped() {
    // @listing skip
    let mut used = Parameter::new(Tensor::from_vec(vec![1.0, 2.0], &[2]));
    let mut spare = Parameter::new(Tensor::from_vec(vec![1.0, 2.0], &[2]));
    let tape = Tape::new();
    let loss = tape.param(&used).sum_all(); // the gradient is [1, 1]
    loss.backward_into(&mut [&mut used, &mut spare]);
    drop(tape);
    let mut opt = Sgd::new(0.1).weight_decay(0.5);
    opt.step(&mut [&mut used, &mut spare]);
    assert!(close(&used.value.to_vec(), &[0.85, 1.8], 1e-6)); // w - 0.1(1 + w/2)
    assert_eq!(spare.value.to_vec(), [1.0, 2.0]); // grad None: untouched
    assert!(opt.state(spare.id()).is_none());
    spare.grad = Some(Tensor::zeros(&[2])); // Challenge 11.2's zeros instead
    opt.step(&mut [&mut spare]);
    assert!(close(&spare.value.to_vec(), &[0.95, 1.9], 1e-6)); // decayed
    // @end
}

#[test]
fn the_update_writes_in_place() {
    // @listing in-place
    let mut p = Parameter::new(Tensor::arange(&[2, 2]));
    p.grad = Some(Tensor::from_vec(vec![1.0; 4], &[2, 2]));
    let buffer = |p: &Parameter| p.value.as_slice().unwrap().as_ptr();

    let before = buffer(&p);
    p.value = p.value.sub(&p.grad.as_ref().unwrap().mul_scalar(0.5));
    assert_ne!(buffer(&p), before); // Section 14.2's update: a new buffer

    let before = buffer(&p);
    let mut opt = Sgd::new(0.5);
    opt.step(&mut [&mut p]);
    assert_eq!(buffer(&p), before); // nothing shares it: written in place
    assert_eq!(p.value.to_vec(), [-1.0, 0.0, 1.0, 2.0]);

    let saved = p.value.clone(); // a second owner
    opt.step(&mut [&mut p]);
    assert_ne!(buffer(&p), before); // so make_mut copied first
    assert_eq!(saved.to_vec(), [-1.0, 0.0, 1.0, 2.0]); // and the copy kept these
    assert_eq!(p.value.to_vec(), [-1.5, -0.5, 0.5, 1.5]);
    // @end
}

#[test]
fn data_mut_on_a_view_copies() {
    let x = Tensor::arange(&[2, 3]);
    let mut t = x.transpose(0, 1);
    t.data_mut()[1] = 100.0; // t is [[0, 3], [1, 4], [2, 5]]
    assert_eq!(t.to_vec(), [0.0, 100.0, 1.0, 4.0, 2.0, 5.0]);
    assert_eq!(x.to_vec(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]); // untouched
    assert!(t.is_contiguous() && !t.shares_storage(&x));
    // A contiguous slice of a buffer it alone owns is written where it lies.
    let mut rows = Tensor::arange(&[3, 2]).slice(0, 1, 3, 1);
    rows.data_mut()[0] = -1.0;
    assert_eq!(
        (rows.offset(), rows.to_vec()),
        (2, vec![-1.0, 3.0, 4.0, 5.0])
    );
}

#[test]
fn weight_decay_by_hand() {
    // @listing decay
    let mut w = Parameter::new(Tensor::from_vec(vec![1.0, -2.0], &[2]));
    {
        let tape = Tape::new();
        let c = tape.constant(Tensor::from_vec(vec![0.5, 0.5], &[2]));
        let loss = (tape.param(&w) * c).sum_all(); // the gradient is c
        loss.backward_into(&mut [&mut w]);
    }
    Sgd::new(0.1).weight_decay(0.5).step(&mut [&mut w]);
    assert!(close(&w.value.to_vec(), &[0.9, -1.95], 1e-6));
    // @end
}

/// One step on the mean squared error plus `lambda/2` times the sum of the squares
/// of the parameters `decays` picks, put on the tape with a second `param`.
fn penalized_step(
    model: &mut impl Module,
    (x, y): (&Tensor, &Tensor),
    lambda: f32,
    decays: fn(&Parameter) -> bool,
    opt: &mut Sgd,
) -> f32 {
    model.zero_grad();
    let fit = {
        let tape = Tape::new();
        let pred = model.forward(tape.constant(x.clone()));
        let fit = mse_loss(pred, tape.constant(y.clone()));
        let mut loss = fit;
        for p in model.parameters().into_iter().filter(|p| decays(p)) {
            loss = loss + tape.param(p).powf(2.0).sum_all() * (lambda / 2.0);
        }
        loss.backward_into(&mut model.parameters());
        fit.value().item()
    };
    opt.step(&mut model.parameters());
    fit
}

#[test]
fn decay_is_the_gradient_of_a_penalty() {
    // @listing penalty
    let (x, y) = toy::xor();
    let mut a = xor::mlp(4, &mut Rng::new(0));
    let mut b = xor::mlp(4, &mut Rng::new(0)); // the same network
    let (mut decay, mut plain) = (Sgd::new(0.1).weight_decay(0.01), Sgd::new(0.1));
    for _ in 0..100 {
        let la = mse_step(&mut a, &x, &y, &mut decay);
        let lb = penalized_step(&mut b, (&x, &y), 0.01, |_| true, &mut plain);
        assert!((la - lb).abs() < 1e-6);
    }
    assert!(close(&numbers(&mut a), &numbers(&mut b), 1e-6));
    // @end
}

#[test]
fn no_decay_on_biases() {
    // @listing groups
    let (x, y) = toy::xor();
    let mut model = xor::mlp(4, &mut Rng::new(0));
    let biases: Vec<ParamId> = model
        .parameters()
        .iter()
        .filter(|p| p.value.ndim() == 1)
        .map(|p| p.id())
        .collect();
    let no_decay = ParamGroup {
        ids: biases.clone(),
        lr: 0.1,
        weight_decay: 0.0,
    };
    let mut opt = Sgd::new(0.1).weight_decay(0.01).group(no_decay);
    for _ in 0..100 {
        mse_step(&mut model, &x, &y, &mut opt);
    }
    // @end
    assert_eq!(opt.groups().len(), 2);
    assert_eq!(ParamGroup::find(opt.groups(), biases[1]).weight_decay, 0.0);
    let weight = model.parameters()[0].id();
    assert_eq!(ParamGroup::find(opt.groups(), weight).weight_decay, 0.01);

    // The same as Challenge 14.5's penalty, lambda·Σw² on weight matrices only,
    // with lambda half the decay.
    let mut again = xor::mlp(4, &mut Rng::new(0));
    for _ in 0..100 {
        l2_step(&mut again, &x, &y, 0.1, 0.005);
    }
    assert!(close(&numbers(&mut model), &numbers(&mut again), 1e-6));

    // And the test-local penalty with the same choice of parameters.
    let mut third = xor::mlp(4, &mut Rng::new(0));
    let mut plain = Sgd::new(0.1);
    let weights = |p: &Parameter| p.value.ndim() == 2;
    for _ in 0..100 {
        penalized_step(&mut third, (&x, &y), 0.01, weights, &mut plain);
    }
    assert!(close(&numbers(&mut model), &numbers(&mut third), 1e-6));
}

#[test]
fn a_learning_rate_can_change_between_steps() {
    let mut opt = Sgd::new(0.1);
    opt.groups_mut()[0].lr = 0.05; // what Chapter 18's schedules will do
    let mut p = Parameter::new(Tensor::from_vec(vec![1.0], &[1]));
    p.grad = Some(Tensor::from_vec(vec![2.0], &[1]));
    opt.step(&mut [&mut p]);
    assert_eq!(p.value.to_vec(), [0.9]);
}

#[test]
fn the_spirals_step_with_sgd() {
    // @listing spirals
    let mut rng = Rng::new(14);
    let small = toy::spirals(25, 1.5, 0.05, &mut rng); // Section 14.6's short run
    let mut by_hand = spirals::mlp(16, &mut rng.clone()); // two equal models
    let mut with_sgd = spirals::mlp(16, &mut rng);
    let mut opt = Sgd::new(2.0);
    for _ in 0..100 {
        let a = spirals::step(&mut by_hand, &small, 2.0);
        let b = spirals::step_with(&mut with_sgd, &small, &mut opt);
        assert_eq!(a, b); // bit for bit
    }
    assert_eq!(numbers(&mut by_hand), numbers(&mut with_sgd));
    // @end
}
