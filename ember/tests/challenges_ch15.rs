//! Tests for the Chapter 15 challenges.
use ember::{
    MaxNorm, Module, Optimizer, ParamGroup, Parameter, Rng, SignSgd, Tape, Tensor,
    decay_groups, mse_loss, toy,
};

#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;
#[path = "../examples/xor.rs"]
#[allow(dead_code)]
mod xor;

fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

fn param(values: &[f32]) -> Parameter {
    Parameter::new(Tensor::from_vec(values.to_vec(), &[values.len()]))
}

#[test]
fn two_steps_with_decay() {
    // @listing predict-test
    let (mut w, mut b) = (param(&[2.0, -1.0]), param(&[1.0]));
    let bias_group = ParamGroup {
        ids: vec![b.id()],
        lr: 0.2,
        weight_decay: 0.0,
    };
    let mut opt = ember::Sgd::new(0.1).weight_decay(0.5).group(bias_group);
    let mut step = |w: &mut Parameter, b: &mut Parameter| {
        let tape = Tape::new();
        let c = tape.constant(Tensor::from_vec(vec![1.0, -1.0], &[2]));
        let loss = (tape.param(w) * c).sum_all() + tape.param(b).sum_all();
        (w.grad, b.grad) = (None, None);
        loss.backward_into(&mut [&mut *w, &mut *b]);
        drop(tape);
        opt.step(&mut [w, b]);
    };
    step(&mut w, &mut b);
    assert!(close(&w.value.to_vec(), &[1.8, -0.85], 1e-6));
    assert!(close(&b.value.to_vec(), &[0.8], 1e-6));
    step(&mut w, &mut b);
    assert!(close(&w.value.to_vec(), &[1.61, -0.7075], 1e-6));
    assert!(close(&b.value.to_vec(), &[0.6], 1e-6));
    for _ in 0..500 {
        step(&mut w, &mut b);
    }
    assert!(close(&w.value.to_vec(), &[-2.0, 2.0], 1e-4)); // -g / λ
    // @end
}

#[test]
fn decay_groups_skip_biases() {
    let mut model = spirals::mlp(16, &mut Rng::new(14));
    let opt = decay_groups(&mut model, 1.0, 1e-3);
    let decayed: Vec<usize> = model
        .parameters()
        .iter()
        .filter(|p| ParamGroup::find(opt.groups(), p.id()).weight_decay > 0.0)
        .map(|p| p.value.numel())
        .collect();
    assert_eq!(decayed, [32, 256, 32]); // 320 of 354 numbers
    assert_eq!(opt.groups()[1].ids.len(), 3); // the three biases
}

#[test]
fn sign_sgd() {
    let mut w = param(&[1.0, -2.0, 3.0]);
    w.grad = Some(Tensor::from_vec(vec![0.5, -0.001, 0.0], &[3]));
    SignSgd::new(0.1).step(&mut [&mut w]);
    assert!(close(&w.value.to_vec(), &[0.9, -1.9, 3.0], 1e-6));

    // @listing sign-test
    let (x, y) = toy::xor();
    let optimizers: Vec<Box<dyn Optimizer>> =
        vec![Box::new(ember::Sgd::new(0.1)), Box::new(SignSgd::new(0.01))];
    let mut at = vec![]; // the loss at steps 100, 200 and 300
    for mut opt in optimizers {
        let mut model = xor::mlp(4, &mut Rng::new(0));
        for i in 1..=300 {
            model.zero_grad();
            let loss = {
                let tape = Tape::new();
                let pred = model.forward(tape.constant(x.clone()));
                let l = mse_loss(pred, tape.constant(y.clone()));
                l.backward_into(&mut model.parameters());
                l.value().item()
            };
            opt.step(&mut model.parameters());
            if i % 100 == 0 {
                at.push(loss);
            }
        }
    }
    let want = [0.210592, 0.110705, 0.013136, 0.004597, 0.002830, 0.002830];
    assert!(close(&at, &want, 1e-6));
    // @end

    // The floor rises with the step: 0.03 and 0.1 end at about 0.0144 and 0.191.
    for (lr, floor) in [(0.03, 0.014395), (0.1, 0.191265)] {
        let mut opt = SignSgd::new(lr);
        let mut model = xor::mlp(4, &mut Rng::new(0));
        let mut loss = 0.0;
        for _ in 0..300 {
            loss = xor::step(&mut model, &x, &y, 0.0); // gradients only
            opt.step(&mut model.parameters());
        }
        assert!((loss - floor).abs() < 1e-5);
    }
}

#[test]
fn saved_parameters_survive() {
    // @listing saved-test
    let mut rng = Rng::new(14);
    let small = toy::spirals(25, 1.5, 0.05, &mut rng);
    let mut model = spirals::mlp(16, &mut rng);
    let mut opt = ember::Sgd::new(2.0);
    let buffers = |m: &mut dyn Module| -> Vec<*const f32> {
        m.parameters()
            .iter()
            .map(|p| p.value.as_slice().unwrap().as_ptr())
            .collect()
    };
    let saved: Vec<Tensor> =
        model.parameters().iter().map(|p| p.value.clone()).collect();
    let copy: Vec<Vec<f32>> = saved.iter().map(|t| t.to_vec()).collect();
    let mut moved = vec![];
    for _ in 0..3 {
        let before = buffers(&mut model);
        spirals::step_with(&mut model, &small, &mut opt);
        let after = buffers(&mut model);
        moved.push(before.iter().zip(&after).filter(|(a, b)| a != b).count());
    }
    assert_eq!(moved, [6, 0, 0]); // one copy per parameter, on the first step only
    assert!(saved.iter().zip(&copy).all(|(t, v)| t.to_vec() == *v));
    // @end
}

#[test]
fn max_norm() {
    // @listing max-norm-test
    let mut w = Parameter::new(Tensor::from_vec(vec![3.0, 0.0, 4.0, 1.0], &[2, 2]));
    let mut b = param(&[5.0, 5.0]);
    w.grad = Some(Tensor::zeros(&[2, 2]));
    b.grad = Some(Tensor::zeros(&[2]));
    let mut opt = MaxNorm {
        inner: ember::Sgd::new(0.1),
        max: 1.0,
    };
    opt.step(&mut [&mut w, &mut b]);
    assert!(close(&w.value.to_vec(), &[0.6, 0.0, 0.8, 1.0], 1e-6)); // 5 → 1
    assert_eq!(b.value.to_vec(), [5.0, 5.0]); // biases are not constrained
    assert_eq!(opt.groups()[0].lr, 0.1); // the inner optimizer's groups
    // @end
}
