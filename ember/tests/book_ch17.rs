//! The examples printed in Chapter 17 of the guided edition: RMSProp, Adam and
//! AdamW.
use ember::{
    Adam, AdamW, Linear, Module, Optimizer, Parameter, ReLU, RmsProp, Rng,
    Sequential, Sgd, Tape, Tensor, cross_entropy, toy,
};

#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;

fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

// @listing walk
/// `steps` steps of `opt` on one parameter starting at `start`, with the
/// gradient `grad(t, w)` set by hand at step t = 1, 2, … Returns the parameter
/// after each step.
fn walk(
    opt: &mut dyn Optimizer,
    start: &[f32],
    mut grad: impl FnMut(usize, &[f32]) -> Vec<f32>,
    steps: usize,
) -> Vec<Vec<f32>> {
    let n = start.len();
    let mut w = Parameter::new(Tensor::from_vec(start.to_vec(), &[n]));
    let mut path = Vec::with_capacity(steps);
    for t in 1..=steps {
        let g = grad(t, &w.value.to_vec());
        w.grad = Some(Tensor::from_vec(g, &[n]));
        opt.step(&mut [&mut w]);
        path.push(w.value.to_vec());
    }
    path
}
// @end

#[test]
fn rmsprop_by_hand() {
    // @listing rms-trace
    let mut opt = RmsProp::new(0.01);
    let mut w = Parameter::new(Tensor::from_vec(vec![1.0], &[1]));
    let (mut vs, mut ws) = (vec![], vec![]);
    for _ in 0..3 {
        w.grad = Some(w.value.clone()); // f(w) = ½w², so w̄ = w
        opt.step(&mut [&mut w]);
        vs.push(opt.state(w.id()).unwrap().square_avg.item());
        ws.push(w.value.item());
    }
    assert!(close(&vs, &[0.01, 0.018, 0.0247575], 1e-7));
    assert!(close(&ws, &[0.9, 0.832918, 0.779982], 1e-6));
    // @end
}

#[test]
fn adam_by_hand() {
    // @listing adam-trace
    let mut opt = Adam::new(0.1);
    let mut w = Parameter::new(Tensor::from_vec(vec![1.0], &[1]));
    let (mut ms, mut vs, mut ws) = (vec![], vec![], vec![]);
    for _ in 0..3 {
        w.grad = Some(w.value.clone()); // f(w) = ½w² again
        opt.step(&mut [&mut w]);
        let state = opt.state(w.id()).unwrap();
        ms.push(state.exp_avg.item());
        vs.push(state.exp_avg_sq.item());
        ws.push(w.value.item());
    }
    assert!(close(&ms, &[0.1, 0.18, 0.2420412], 1e-6));
    assert!(close(&vs, &[0.001, 0.001809, 0.00244785], 1e-7));
    assert!(close(&ws, &[0.9, 0.800412, 0.701587], 1e-6));
    // @end
    // The steps, and the corrected moments m_hat = m/(1 − 0.9^t), v_hat = v/(1 − 0.999^t).
    let steps = [1.0 - ws[0], ws[0] - ws[1], ws[1] - ws[2]];
    assert!(close(&steps, &[0.1, 0.0995878, 0.098826], 1e-6));
    let m_hat = [0, 1, 2].map(|i| ms[i] / (1.0 - 0.9f32.powi(i as i32 + 1)));
    let v_hat = [0, 1, 2].map(|i| vs[i] / (1.0 - 0.999f32.powi(i as i32 + 1)));
    assert!(close(&m_hat, &[1.0, 0.947368, 0.893141], 1e-5));
    assert!(close(&v_hat, &[1.0, 0.904952, 0.816767], 1e-4));
}

#[test]
fn the_first_step_is_a_sign() {
    // @listing first-step
    let first = |g: f32| walk(&mut Adam::new(0.1), &[0.0], |_, _| vec![g], 1)[0][0];
    assert_eq!([1.0, 1000.0, -5.0].map(first), [-0.1, -0.1, 0.1]);
    assert!((first(0.001) + 0.099999).abs() < 1e-6); // eps shows at 0.001
    // @end
}

#[test]
fn without_bias_correction() {
    // @listing uncorrected
    // A constant gradient g = 1: Adam's moments without the corrections.
    let (mut m, mut v) = (0.0f64, 0.0f64);
    let mut ratio = vec![]; // m / √v: the step in units of lr
    for _ in 0..1000 {
        m = 0.9 * m + 0.1;
        v = 0.999 * v + 0.001;
        ratio.push(m / v.sqrt());
    }
    let at = [1, 2, 3, 12, 100, 1000].map(|t| ratio[t - 1] as f32);
    let want = [3.1623, 4.2496, 4.9502, 6.5685, 3.2408, 1.2576];
    assert!(close(&at, &want, 1e-4));
    let peak = ratio.iter().cloned().fold(0.0, f64::max);
    assert_eq!(peak, ratio[11]); // the largest step is step 12's
    // @end
    // With the corrections, every step of a constant gradient is exactly lr.
    let path = walk(&mut Adam::new(0.1), &[0.0], |_, _| vec![1.0], 100);
    let mut before = 0.0;
    for w in &path {
        assert!((before - w[0] - 0.1).abs() < 1e-5);
        before = w[0];
    }
}

#[test]
fn a_step_size_per_number() {
    // @listing scale
    // f(w) = ½(0.001·w₀² + w₁²): gradients 0.001·w₀ and w₁, 1000× apart.
    let grad = |_, w: &[f32]| vec![0.001 * w[0], w[1]];
    let sgd = walk(&mut Sgd::new(0.1), &[1.0, 1.0], grad, 3);
    let adam = walk(&mut Adam::new(0.1), &[1.0, 1.0], grad, 3);
    assert!(close(&sgd[2], &[0.9997, 0.729], 1e-6));
    assert!(close(&adam[2], &[0.70159, 0.70159], 1e-5));
    // @end
    let rms = walk(&mut RmsProp::new(0.01), &[1.0, 1.0], grad, 3);
    assert!(close(&rms[2], &[0.78, 0.78], 1e-4));
}

#[test]
fn two_kinds_of_decay() {
    // @listing two-decays
    // Section 15.6's example: w = [1, -2], w̄ = [0.5, 0.5], lr 0.1, λ 0.5.
    let grad = |_, _: &[f32]| vec![0.5, 0.5];
    let (start, steps) = ([1.0, -2.0], 1);
    let mut adam = Adam::new(0.1).weight_decay(0.5);
    let mut adamw = AdamW::new(0.1).weight_decay(0.5);
    let l2 = walk(&mut adam, &start, grad, steps);
    let decoupled = walk(&mut adamw, &start, grad, steps);
    assert!(close(&l2[0], &[0.9, -1.9], 1e-6)); // −0.1·sign(w̄ + λw)
    assert!(close(&decoupled[0], &[0.85, -2.0], 1e-6)); // 0.95w, then −0.1·sign(w̄)
    // @end
}

/// What is left of a weight of 1 after 1000 steps of `opt` whose gradient
/// alternates +s, −s: its value, divided by the value without decay.
fn kept(with: &mut dyn Optimizer, without: &mut dyn Optimizer, s: f32) -> f32 {
    let grad = |t: usize, _: &[f32]| vec![if t % 2 == 1 { s } else { -s }];
    let a = walk(with, &[1.0], grad, 1000)[999][0];
    let b = walk(without, &[1.0], grad, 1000)[999][0];
    a / b
}

#[test]
fn decay_and_gradient_size() {
    // @listing decay-table
    let (lr, wd) = (0.001, 0.1);
    let rows = [0.01, 0.1, 1.0, 10.0, 100.0].map(|s| {
        let sgd = kept(&mut Sgd::new(lr).weight_decay(wd), &mut Sgd::new(lr), s);
        let adam = kept(&mut Adam::new(lr).weight_decay(wd), &mut Adam::new(lr), s);
        let adamw = AdamW::new(lr).weight_decay(wd);
        let adamw = kept(&mut { adamw }, &mut Adam::new(lr), s);
        [sgd, adam, adamw]
    });
    let sgd = [0.9048, 0.9048, 0.9049, 0.9053, 0.9096];
    let adam = [0.2623, 0.4618, 0.9053, 0.9901, 0.9990];
    assert!(close(&rows.map(|r| r[0]), &sgd, 1e-4));
    assert!(close(&rows.map(|r| r[1]), &adam, 1e-4));
    assert!(close(&rows.map(|r| r[2]), &[0.9048; 5], 1e-4)); // (1 − 0.0001)^1000
    // @end
}

#[test]
fn the_spirals_with_adam() {
    // @listing spirals
    let run = |mut opt: Box<dyn Optimizer>, steps| {
        let mut rng = Rng::new(14);
        let small = toy::spirals(25, 1.5, 0.05, &mut rng); // Section 14.6's short run
        let mut model = spirals::mlp(16, &mut rng);
        let losses: Vec<f32> = (0..steps)
            .map(|_| spirals::step_with(&mut model, &small, &mut *opt))
            .collect();
        let below = losses.iter().position(|&l| l < 0.01).map(|i| i + 1);
        (below, spirals::accuracy(&model, &small))
    };
    assert_eq!(run(Box::new(Adam::new(0.03)), 83), (Some(83), 1.0));
    assert_eq!(run(Box::new(RmsProp::new(0.01)), 345), (Some(345), 1.0));
    // @end
}

#[test]
fn what_adam_remembers() {
    // @listing memory
    let mut rng = Rng::new(42);
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(784, 128, &mut rng)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, &mut rng)),
    ]);
    let x = Tensor::rand(&[2, 784], &mut rng);
    {
        let tape = Tape::new();
        let loss = cross_entropy(model.forward(tape.constant(x)), &[3, 7]);
        loss.backward_into(&mut model.parameters());
    }
    let mut opt = Adam::new(0.001);
    opt.step(&mut model.parameters());
    let params = model.parameters();
    let weights: usize = params.iter().map(|p| p.value.numel()).sum();
    let state: usize = params
        .iter()
        .map(|p| opt.state(p.id()).unwrap())
        .map(|s| s.exp_avg.numel() + s.exp_avg_sq.numel())
        .sum();
    assert_eq!((weights, state), (101_770, 203_540));
    assert_eq!(state * size_of::<f32>(), 814_160); // bytes
    // @end
    assert_eq!(weights * size_of::<f32>(), 407_080);
}
