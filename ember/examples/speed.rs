//! Chapter 24: where a training step spends its time, and what threads and
//! reused buffers buy.
//!
//! cargo run --release --example speed               profile MLP and CNN steps
//! cargo run --release --example speed -- epochs     one epoch of each, timed
//! cargo run --release --example speed -- passes     one pass over the CNN's
//!                                                    tensors, five ways
//! cargo run --release --example speed -- challenges the challenges, timed

#[path = "mnist_cnn.rs"]
#[allow(dead_code)]
pub mod mnist_cnn;

use ember::data::{self, DataError};
use ember::profile::{self, Profile, mark};
use ember::{
    AdamW, Batch, DataLoader, Module, Optimizer, Rng, Sequential, Tape, Tensor,
    Window, clip_grad_norm, cross_entropy, set_threads, train_epoch, train_step,
};
use mnist_cnn::cnn;
use mnist_cnn::mnist_train::mlp;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

// @listing marked-step
/// Section 20.3's `train_step`, with a mark after each of its parts.
pub fn marked_step(model: &mut dyn Module, batch: &Batch, opt: &mut dyn Optimizer) {
    let (x, labels) = batch;
    model.zero_grad();
    mark("step", "zero_grad");
    {
        let tape = Tape::new();
        let loss = cross_entropy(model.forward(tape.constant(x.clone())), labels);
        loss.backward_into(&mut model.parameters());
        mark("step", "add_grad"); // after the walk's last mark
    } // the tape is dropped
    mark("step", "drop tape");
    clip_grad_norm(&mut model.parameters(), 1.0);
    mark("step", "clip");
    opt.step(&mut model.parameters());
    mark("step", "optimizer");
}
// @end

/// `steps` steps of `model` on MNIST batches of 64, after 10 to warm up.
fn profile_steps(model: &mut Sequential, batches: &[Batch], steps: usize) -> Profile {
    let mut opt = AdamW::new(1e-3);
    for batch in &batches[..10] {
        marked_step(model, batch, &mut opt);
    }
    let ((), profile) = profile::profile(|| {
        for batch in batches[10..].iter().cycle().take(steps) {
            marked_step(model, batch, &mut opt);
        }
    });
    profile
}

fn report(name: &str, profile: &Profile, steps: u32) {
    println!(
        "\n{name}, per step (averaged over {steps}):\n{}",
        profile.per(steps)
    );
}

fn profiles(dir: &Path) -> Result<(), DataError> {
    let train = data::mnist(dir, true)?;
    let mut rng = Rng::new(24);
    let batches: Vec<Batch> = DataLoader::new(&train, 64)
        .shuffle(Rng::new(24))
        .epoch()
        .take(110)
        .collect();
    let mut model = mlp(&mut rng);
    report("MLP", &profile_steps(&mut model, &batches, 1000), 1000);
    let mut model = cnn(&mut rng);
    report("CNN", &profile_steps(&mut model, &batches, 100), 100);
    Ok(())
}

// @listing epochs
/// One epoch of Section 20.3's `train_step` on MNIST, batches of 64, for the
/// MLP and the CNN: the time, and the mean loss, which threads don't change.
fn epochs(dir: &Path) -> Result<(), DataError> {
    let train = data::mnist(dir, true)?;
    let nets = [("MLP", mlp as fn(&mut Rng) -> Sequential), ("CNN", cnn)];
    for (name, build) in nets {
        let mut rng = Rng::new(24);
        let mut model = build(&mut rng);
        let mut opt = AdamW::new(1e-3);
        let mut loader = DataLoader::new(&train, 64).shuffle(rng).drop_last(true);
        let start = Instant::now();
        let step = |m: &mut dyn Module, b: &Batch| train_step(m, b, &mut opt, 1.0);
        let loss = train_epoch(&mut model, &mut loader, step);
        let secs = start.elapsed().as_secs_f64();
        println!("{name}: one epoch in {secs:.2} s, mean loss {loss:.6}");
    }
    Ok(())
}
// @end

/// The fastest of 50 runs of `f`, in microseconds, after one to warm up.
fn micros(f: impl Fn() -> Tensor) -> f64 {
    drop(f());
    let time = |_| {
        let start = Instant::now();
        drop(black_box(f()));
        start.elapsed().as_secs_f64() * 1e6
    };
    (0..50).map(time).fold(f64::INFINITY, f64::min)
}

// @listing passes
/// Passes over the CNN's tensors: its first activation, `[64, 8, 28, 28]`,
/// four ways, and the second convolution's im2col matrix, made and transposed.
/// The time, and the time per number written (or read, for a sum).
fn passes() -> Result<(), DataError> {
    set_threads(1);
    let x = Tensor::arange(&[64, 8, 28, 28]);
    let conv = Tensor::arange(&[64, 28, 28, 8]).permute(&[0, 3, 1, 2]); // a view
    let (bias, cols) = (Tensor::arange(&[8, 1, 1]), Tensor::arange(&[12_544, 72]));
    let (pooled, same) = (Tensor::arange(&[64, 8, 14, 14]), Window::new(3, 1, 1));
    let cases: [(&str, &dyn Fn() -> Tensor); 6] = [
        ("relu, contiguous", &|| x.relu()),
        ("relu, permuted view", &|| conv.relu()),
        ("+ bias, broadcast", &|| x.add(&bias)),
        ("bias gradient, sum_to", &|| x.sum_to(&[8, 1, 1])),
        ("im2col of [64, 8, 14, 14]", &|| pooled.im2col(same)),
        ("im2col matrix, transposed", &|| {
            cols.transpose(0, 1).contiguous()
        }),
    ];
    for (name, f) in cases {
        let n = f().numel().max(x.numel()) as f64; // numbers written, or read
        let us = micros(f);
        println!("{name:<26} {us:>7.1} µs {:>6.2} ns", us * 1e3 / n);
    }
    Ok(())
}
// @end

/// Timings for the answers to Chapter 24's challenges.
fn challenges(dir: &Path) -> Result<(), DataError> {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let counts = [1, 2, 6, cores];
    let x = Tensor::randn(&[4096, 1024], &mut Rng::new(24)); // 4M numbers
    let row = |name: &str, time: &dyn Fn(usize) -> f64| {
        let times: Vec<String> =
            counts.iter().map(|&t| format!("{:.0}", time(t))).collect();
        println!(
            "par_map {name} on {counts:?} threads: {} µs",
            times.join(", ")
        );
    };
    let relu = |v: f32| if v < 0.0 { 0.0 } else { v };
    row("relu", &|t| micros(|| ember::par_map(&x, relu, t)));
    row("exp", &|t| {
        micros(|| ember::par_map(&x, ember::math::exp, t))
    });
    set_threads(1);
    for (k, m, n) in [(12_544, 72, 16), (50_176, 9, 8)] {
        let (a, g) = (Tensor::arange(&[k, m]), Tensor::arange(&[k, n]));
        let copied = micros(|| a.transpose(0, 1).matmul(&g));
        let tn = micros(|| a.matmul_tn(&g));
        println!(
            "[{k}, {m}]ᵀ @ [{k}, {n}]: copy + blocked {copied:.0} µs, tn {tn:.0} µs"
        );
    }
    let pooled = Tensor::arange(&[64, 8, 14, 14]);
    let same = Window::new(3, 1, 1);
    let times: Vec<String> = (counts.iter())
        .map(|&t| {
            format!("{:.0}", micros(|| ember::im2col_parallel(&pooled, same, t)))
        })
        .collect();
    let one = micros(|| pooled.im2col(same));
    println!(
        "im2col {one:.0} µs; parallel on {counts:?}: {} µs",
        times.join(", ")
    );
    let (batch, m, k, n) = (256, 32, 16, 32);
    let (a, b) = (
        Tensor::arange(&[batch, m, k]),
        Tensor::arange(&[batch, k, n]),
    );
    let plain = micros(|| a.matmul(&b));
    let (sa, sb) = (a.as_slice().unwrap(), b.as_slice().unwrap());
    let times: Vec<String> = (counts.iter())
        .map(|&t| {
            let f = || {
                let mut c = vec![0.0; batch * m * n];
                ember::matmul_stacks(sa, sb, &mut c, (batch, (m, k, n)), t);
                Tensor::from_vec(c, &[batch, m, n])
            };
            format!("{:.0}", micros(f))
        })
        .collect();
    println!(
        "[256, 32, 16] @ [256, 16, 32]: matmul {plain:.0} µs, stacks {} µs",
        times.join(", ")
    );
    set_threads(cores);
    let train = data::mnist(dir, true)?;
    for prefetch in [false, true] {
        let mut model = mlp(&mut Rng::new(24));
        let mut opt = AdamW::new(1e-3);
        let start = Instant::now();
        let mut step = |b: Batch| {
            train_step(&mut model, &b, &mut opt, 1.0);
        };
        if prefetch {
            ember::prefetched_epoch(&train, 64, &mut Rng::new(24), &mut step);
        } else {
            let mut loader = DataLoader::new(&train, 64)
                .shuffle(Rng::new(24))
                .drop_last(true);
            loader.epoch().for_each(&mut step);
        }
        println!(
            "MLP epoch, prefetch {prefetch}: {:.3} s",
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let default = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../data"));
    let dir = args.iter().find(|a| a.contains('/')).map(PathBuf::from);
    let dir = dir.unwrap_or(default);
    let result = match args.first().map(String::as_str) {
        Some("epochs") => epochs(&dir),
        Some("passes") => passes(),
        Some("challenges") => challenges(&dir),
        _ => profiles(&dir),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
