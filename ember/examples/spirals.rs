//! Chapter 14: a multilayer perceptron learns two interleaved spirals.
//!
//! cargo run --release --example spirals             train, then draw the result
//! cargo run --release --example spirals -- widths   hidden widths 4, 16 and 64
//! cargo run --release --example spirals -- time     where a step's time goes
//! cargo run --release --example spirals -- decay    weight decay (Chapter 15)
//! cargo run --release --example spirals -- momentum momentum (Chapter 16)
//! cargo run --release --example spirals -- adam     RMSProp, Adam, AdamW (Ch. 17)
//! cargo run --release --example spirals -- schedule schedules, warmup (Ch. 18)

use ember::{
    Adam, AdamW, ConstantLr, CosineLr, Linear, LinearWarmup, LrScheduler, Module,
    Optimizer, RmsProp, Rng, Sequential, Sgd, Tanh, Tape, Tensor, base_lrs,
    clip_grad_norm, cross_entropy, plot, toy, toy::Labeled,
};
use std::time::{Duration, Instant};

// @listing data
/// 100 points per arm, 1.5 turns, noise 0.05: a training set, and a held-out set
/// drawn from the same generator.
pub fn data(rng: &mut Rng) -> (Labeled, Labeled) {
    let train = toy::spirals(100, 1.5, 0.05, rng);
    let held_out = toy::spirals(100, 1.5, 0.05, rng);
    (train, held_out)
}

/// 2 inputs, two hidden layers of `width` tanh units, 2 logits.
pub fn mlp(width: usize, rng: &mut Rng) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(2, width, rng)),
        Box::new(Tanh),
        Box::new(Linear::new(width, width, rng)),
        Box::new(Tanh),
        Box::new(Linear::new(width, 2, rng)),
    ])
}
// @end

// @listing step
/// One step of full-batch gradient descent on the cross-entropy. Returns the loss
/// before the step.
pub fn step(model: &mut impl Module, (x, labels): &Labeled, lr: f32) -> f32 {
    model.zero_grad();
    let loss = {
        let tape = Tape::new();
        let logits = model.forward(tape.constant(x.clone()));
        let loss = cross_entropy(logits, labels);
        loss.backward_into(&mut model.parameters());
        loss.value().item()
    };
    for p in model.parameters() {
        let g = p.grad.as_ref().expect("a gradient");
        p.value = p.value.sub(&g.mul_scalar(lr));
    }
    loss
}
// @end

// @listing step-with
/// Section 14.6's step with the update handed to an optimizer (Chapter 15).
pub fn step_with(
    model: &mut impl Module,
    (x, labels): &Labeled,
    opt: &mut dyn Optimizer,
) -> f32 {
    model.zero_grad();
    let loss = {
        let tape = Tape::new();
        let logits = model.forward(tape.constant(x.clone()));
        let loss = cross_entropy(logits, labels);
        loss.backward_into(&mut model.parameters());
        loss.value().item()
    }; // the tape is dropped: every parameter owns its buffer
    opt.step(&mut model.parameters());
    loss
}
// @end

// @listing train-with
/// Chapter 18's training loop: before update t, the schedule sets every group's
/// learning rate; after the backward pass, the gradient is clipped to
/// `max_norm`. Returns every loss and every gradient norm before clipping.
pub fn train_with(
    model: &mut impl Module,
    (x, labels): &Labeled,
    opt: &mut dyn Optimizer,
    schedule: &dyn LrScheduler,
    (steps, max_norm): (usize, f32),
) -> (Vec<f32>, Vec<f32>) {
    let base = base_lrs(opt); // the rates the optimizer was built with
    let (mut losses, mut norms) = (vec![], vec![]);
    for t in 0..steps {
        schedule.apply(opt, &base, t); // 1. this step's learning rate
        model.zero_grad();
        let loss = {
            let tape = Tape::new(); // 2. loss and gradients
            let loss = cross_entropy(model.forward(tape.constant(x.clone())), labels);
            loss.backward_into(&mut model.parameters());
            loss.value().item()
        };
        norms.push(clip_grad_norm(&mut model.parameters(), max_norm)); // 3. clip
        opt.step(&mut model.parameters()); // 4. update, in place
        losses.push(loss);
    }
    (losses, norms)
}
// @end

// @listing accuracy
/// The class with the largest logit, for every row of `x`.
pub fn predict(model: &impl Module, x: &Tensor) -> Vec<usize> {
    let tape = Tape::new();
    let logits = tape.no_grad(|| model.forward(tape.constant(x.clone())));
    logits.value().argmax(1)
}

/// The fraction of rows whose predicted class is the label.
pub fn accuracy(model: &impl Module, (x, labels): &Labeled) -> f32 {
    let predicted = predict(model, x);
    let right = predicted.iter().zip(labels).filter(|(p, l)| p == l).count();
    right as f32 / labels.len() as f32
}
// @end

/// `steps` steps; prints a line every `every` steps, and returns every loss.
pub fn train(
    model: &mut impl Module,
    data: &(Labeled, Labeled),
    lr: f32,
    steps: usize,
    every: usize,
) -> Vec<f32> {
    let (train, held_out) = data;
    let mut losses = Vec::with_capacity(steps);
    for i in 1..=steps {
        losses.push(step(model, train, lr));
        if every > 0 && (i == 1 || i % every == 0) {
            let (a, b) = (accuracy(model, train), accuracy(model, held_out));
            println!("{i:>6} {:>8.4} {a:>8.3} {b:>8.3}", losses[i - 1]);
        }
    }
    losses
}

/// One step, timed in three parts: forward and loss, backward, update.
fn timed_step(
    model: &mut impl Module,
    (x, labels): &Labeled,
    lr: f32,
) -> [Duration; 3] {
    model.zero_grad();
    let t0 = Instant::now();
    let tape = Tape::new();
    let loss = cross_entropy(model.forward(tape.constant(x.clone())), labels);
    let t1 = Instant::now();
    loss.backward_into(&mut model.parameters());
    drop(tape);
    let t2 = Instant::now();
    for p in model.parameters() {
        let g = p.grad.as_ref().expect("a gradient");
        p.value = p.value.sub(&g.mul_scalar(lr));
    }
    [t1 - t0, t2 - t1, t2.elapsed()]
}

fn widths() {
    println!(
        "{:>6} {:>7} {:>8} {:>8} {:>8} {:>10}",
        "width", "params", "loss", "train", "held", "per step"
    );
    for width in [4, 16, 64] {
        let mut rng = Rng::new(14);
        let data = data(&mut rng);
        let mut model = mlp(width, &mut rng);
        let count: usize = model.parameters().iter().map(|p| p.value.numel()).sum();
        let start = Instant::now();
        let losses = train(&mut model, &data, 1.0, 10_000, 0);
        let per_step = start.elapsed() / 10_000;
        let (a, b) = (accuracy(&model, &data.0), accuracy(&model, &data.1));
        println!(
            "{width:>6} {count:>7} {:>8.4} {a:>8.3} {b:>8.3} {per_step:>10.1?}",
            losses[9_999]
        );
    }
}

fn time() {
    let mut rng = Rng::new(14);
    let (train, _) = data(&mut rng);
    let mut model = mlp(16, &mut rng);
    let mut total = [Duration::ZERO; 3];
    for _ in 0..1000 {
        let parts = timed_step(&mut model, &train, 1.0);
        for k in 0..3 {
            total[k] += parts[k];
        }
    }
    for (name, t) in ["forward and loss", "backward", "update"].iter().zip(total) {
        println!("{name:>16} {:>8.1?}", t / 1000);
    }
    // The forward pass's three products alone, on tensors of the same shapes.
    let (a, w1) = (Tensor::zeros(&[200, 2]), Tensor::zeros(&[2, 16]));
    let (h, w2, w3) = (
        Tensor::zeros(&[200, 16]),
        Tensor::zeros(&[16, 16]),
        Tensor::zeros(&[16, 2]),
    );
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = (a.matmul(&w1), h.matmul(&w2), h.matmul(&w3));
    }
    println!(
        "{:>16} {:>8.1?}",
        "forward products",
        start.elapsed() / 1000
    );
    let h = Tensor::randn(&[200, 16], &mut rng);
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = (h.tanh(), h.tanh());
    }
    println!("{:>16} {:>8.1?}", "forward tanh", start.elapsed() / 1000);
    let b = Tensor::zeros(&[16]);
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = (h.add(&b), h.add(&b), a.add(&b.slice(0, 0, 2, 1)));
    }
    println!("{:>16} {:>8.1?}", "forward biases", start.elapsed() / 1000);
}

/// 10,000 steps with weight decay 0, 1e-4, 1e-3 and 1e-2 on every parameter.
fn decay() {
    println!(
        "{:>8} {:>8} {:>8} {:>8} {:>8}",
        "decay", "loss", "train", "held", "sum w²"
    );
    for wd in [0.0, 1e-4, 1e-3, 1e-2] {
        let mut rng = Rng::new(14);
        let data = data(&mut rng);
        let mut model = mlp(16, &mut rng);
        let mut opt = Sgd::new(1.0).weight_decay(wd);
        let mut loss = 0.0;
        for _ in 0..10_000 {
            loss = step_with(&mut model, &data.0, &mut opt);
        }
        let (a, b) = (accuracy(&model, &data.0), accuracy(&model, &data.1));
        let sq: f32 = model
            .parameters()
            .iter()
            .map(|p| p.value.powf(2.0).sum_all())
            .sum();
        println!("{wd:>8} {loss:>8.4} {a:>8.3} {b:>8.3} {sq:>8.1}");
    }
}

// @listing momentum-runs
/// Plain SGD, momentum and Nesterov (Chapter 16), 10,000 steps each: the first
/// step with a loss below 0.05 and below 0.01, the last loss, and the accuracies.
fn momentum() {
    let runs = [
        ("sgd", Sgd::new(1.0)),
        ("sgd", Sgd::new(3.0)),
        ("sgd", Sgd::new(5.0)),
        ("momentum", Sgd::new(0.1).momentum(0.9)),
        ("momentum", Sgd::new(0.5).momentum(0.9)),
        ("nesterov", Sgd::new(0.5).momentum(0.9).nesterov(true)),
        ("momentum", Sgd::new(1.0).momentum(0.9)),
        ("nesterov", Sgd::new(1.0).momentum(0.9).nesterov(true)),
    ];
    header();
    for (name, mut opt) in runs {
        report(name, &mut opt);
    }
}
// @end

/// The column names of `report`'s lines.
fn header() {
    let head = ["", "lr", "<0.05", "<0.01", "loss", "train", "held"];
    println!(
        "{:>8} {:>6} {:>6} {:>6} {:>8} {:>6} {:>6}",
        head[0], head[1], head[2], head[3], head[4], head[5], head[6]
    );
}

/// Train Section 14.6's model for 10,000 steps with `opt` and print one line.
fn report(name: &str, opt: &mut dyn Optimizer) {
    let mut rng = Rng::new(14);
    let data = data(&mut rng);
    let mut model = mlp(16, &mut rng);
    let losses: Vec<f32> = (0..10_000)
        .map(|_| step_with(&mut model, &data.0, opt))
        .collect();
    let below = |t: f32| match losses.iter().position(|&l| l < t) {
        Some(i) => format!("{}", i + 1),
        None => "never".into(),
    };
    let (a, b) = (accuracy(&model, &data.0), accuracy(&model, &data.1));
    let (lr, last) = (opt.groups()[0].lr, losses[9_999]);
    let (b05, b01) = (below(0.05), below(0.01));
    println!("{name:>8} {lr:>6} {b05:>6} {b01:>6} {last:>8.4} {a:>6.3} {b:>6.3}");
}

/// Chapter 17: RMSProp, Adam and AdamW beside SGD, over a sweep of learning rates.
fn adaptive() {
    header();
    for lr in [0.1, 0.3, 1.0, 3.0, 10.0] {
        report("sgd", &mut Sgd::new(lr));
    }
    report("momentum", &mut Sgd::new(1.0).momentum(0.9));
    for lr in [0.001, 0.003, 0.01, 0.03] {
        report("rmsprop", &mut RmsProp::new(lr));
    }
    for lr in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3] {
        report("adam", &mut Adam::new(lr));
    }
    for lr in [0.01, 0.03] {
        report("adamw", &mut AdamW::new(lr));
    }
}

/// Chapter 18: 10,000 steps of `opt` under `schedule`, clipped to `max_norm`;
/// prints the first step below 0.01, the last loss, the largest loss of the
/// second half, and the accuracies.
fn report_with(name: &str, opt: &mut dyn Optimizer, schedule: &dyn LrScheduler) {
    let mut rng = Rng::new(14);
    let data = data(&mut rng);
    let mut model = mlp(16, &mut rng);
    let lr = opt.groups()[0].lr;
    let (losses, _) =
        train_with(&mut model, &data.0, opt, schedule, (10_000, f32::INFINITY));
    let below = match losses.iter().position(|&l| l < 0.01) {
        Some(i) => format!("{}", i + 1),
        None => "never".into(),
    };
    let worst = losses[5_000..].iter().cloned().fold(0.0, f32::max);
    let (a, b) = (accuracy(&model, &data.0), accuracy(&model, &data.1));
    let last = losses[9_999];
    println!(
        "{name:>14} {lr:>6} {below:>6} {last:>8.1e} {worst:>8.1e} {a:>6.3} {b:>6.3}"
    );
}

/// Chapter 18: constant rates against cosine decay and a warmup of 100 steps.
fn schedules() {
    println!(
        "{:>14} {:>6} {:>6} {:>8} {:>8} {:>6} {:>6}",
        "", "lr", "<0.01", "loss", "worst", "train", "held"
    );
    let cosine = CosineLr::new(10_000, 0.0);
    let warm = LinearWarmup::new(100, ConstantLr);
    let warm_cosine = LinearWarmup::new(100, CosineLr::new(9_900, 0.0));
    for lr in [0.03, 0.1, 0.3] {
        report_with("adam", &mut Adam::new(lr), &ConstantLr);
        report_with("adam cosine", &mut Adam::new(lr), &cosine);
        report_with("adam warmup", &mut Adam::new(lr), &warm);
        report_with("warmup cosine", &mut Adam::new(lr), &warm_cosine);
    }
    report_with("momentum", &mut Sgd::new(1.0).momentum(0.9), &ConstantLr);
    report_with("mom. cosine", &mut Sgd::new(1.0).momentum(0.9), &cosine);
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("schedule") => return schedules(),
        Some("momentum") => return momentum(),
        Some("adam") => return adaptive(),
        Some("widths") => return widths(),
        Some("decay") => return decay(),
        Some("time") => return time(),
        _ => {}
    }
    let mut rng = Rng::new(14);
    let data = data(&mut rng);
    let mut model = mlp(16, &mut rng);
    println!("{:>6} {:>8} {:>8} {:>8}", "step", "loss", "train", "held");
    let start = Instant::now();
    let losses = train(&mut model, &data, 1.0, 10_000, 1000);
    println!("{:.2?} per step", start.elapsed() / 10_000);
    print!("{}", plot::curve(&losses, 72, 10));
    let (x, labels) = &data.0;
    let classify = |grid: &Tensor| predict(&model, grid);
    print!("{}", plot::regions(classify, 1.1, (64, 24), (x, labels)));
}
