//! Chapter 19: a multilayer perceptron learns MNIST's handwritten digits.
//!
//! Download the four files and uncompress them into `data/` at the workspace root
//! (Section 19.6), or pass another folder after `--`:
//!
//! cargo run --release --example mnist_mlp              train for five epochs
//! cargo run --release --example mnist_mlp -- stats     the pixels' mean and std
//! cargo run --release --example mnist_mlp -- loader    get vs get_batch, timed

use ember::data::{self, DataError};
use ember::{
    AdamW, Batch, CosineLr, DataLoader, Dataset, Linear, LinearWarmup, LrScheduler,
    Module, Optimizer, ReLU, Rng, Sequential, Tape, Tensor, TensorDataset, base_lrs,
    clip_grad_norm, cross_entropy,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// 784 pixels, 128 ReLU units, 10 logits: 101,770 numbers.
pub fn mlp(rng: &mut Rng) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(784, 128, rng)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, rng)),
    ])
}

/// Chapter 18's step on one batch, clipped at a norm of 1. Returns the loss.
pub fn step(
    model: &mut impl Module,
    (x, labels): &Batch,
    opt: &mut dyn Optimizer,
) -> f32 {
    model.zero_grad();
    let loss = {
        let tape = Tape::new();
        let loss = cross_entropy(model.forward(tape.constant(x.clone())), labels);
        loss.backward_into(&mut model.parameters());
        loss.value().item()
    };
    clip_grad_norm(&mut model.parameters(), 1.0);
    opt.step(&mut model.parameters());
    loss
}

// @listing accuracy
/// The fraction of examples whose largest logit is their label, in batches of
/// 1,000 without a tape's rules (Chapter 20 makes this a proper metric).
pub fn accuracy(model: &impl Module, data: &impl Dataset) -> f32 {
    let mut right = 0;
    for (x, labels) in DataLoader::new(data, 1000).epoch() {
        let tape = Tape::new();
        let logits = tape.no_grad(|| model.forward(tape.constant(x)));
        let predicted = logits.value().argmax(1);
        let hits = predicted.iter().zip(&labels).filter(|(p, l)| p == l);
        right += hits.count();
    }
    right as f32 / data.len() as f32
}
// @end

/// Five epochs of batches of 64, AdamW at 1e-3 with a warmup and a cosine.
fn train(dir: &Path) -> Result<(), DataError> {
    let start = Instant::now();
    let (train, test) = (data::mnist(dir, true)?, data::mnist(dir, false)?);
    println!("loaded in {:.2?}", start.elapsed());
    let mut rng = Rng::new(19);
    let mut model = mlp(&mut rng);
    println!("epoch     loss    train     test     time");
    // @listing loop
    let mut loader = DataLoader::new(&train, 64).shuffle(rng).drop_last(true);
    let steps = 5 * loader.len(); // 5 epochs of 937 batches
    let schedule = LinearWarmup::new(100, CosineLr::new(steps - 100, 0.0));
    let mut opt = AdamW::new(1e-3);
    let base = base_lrs(&opt);
    let mut t = 0; // updates so far, across epochs
    for epoch in 1..=5 {
        let (start, mut total) = (Instant::now(), 0.0);
        for batch in loader.epoch() {
            schedule.apply(&mut opt, &base, t);
            total += step(&mut model, &batch, &mut opt);
            t += 1;
        }
        let (loss, secs) = (total / loader.len() as f32, start.elapsed());
        let (a, b) = (accuracy(&model, &train), accuracy(&model, &test));
        println!("{epoch:>5} {loss:>8.4} {a:>8.4} {b:>8.4} {secs:>8.2?}");
    }
    // @end
    Ok(())
}

/// The mean and standard deviation of `xs`, added up in `f64`.
fn mean_std(xs: &[f32]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = xs.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// The training pixels' mean and standard deviation, before and after
/// normalizing, and how many examples each class has.
fn stats(dir: &Path) -> Result<(), DataError> {
    let path = dir.join("train-images-idx3-ubyte");
    let bytes =
        std::fs::read(&path).map_err(|source| DataError::Io { path, source })?;
    let (mean, std) = mean_std(data::mnist_images(&bytes)?.as_slice().unwrap());
    println!("pixels in [0, 1]: mean {mean:.5}, std {std:.5}");
    let train = data::mnist(dir, true)?;
    let (mean, std) = mean_std(train.tensors().0.as_slice().unwrap());
    println!("normalized: mean {mean:.4}, std {std:.4}");
    let mut counts = [0; 10];
    train.tensors().1.iter().for_each(|&l| counts[l] += 1);
    println!("examples per class: {counts:?}");
    Ok(())
}

/// A dataset with only `get`, so batches are made by the default `get_batch`.
struct PerExample<'a>(&'a TensorDataset);

impl Dataset for PerExample<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(&self, i: usize) -> (Tensor, usize) {
        self.0.get(i)
    }
}

/// One shuffled epoch of batches of 64, both ways.
fn loader(dir: &Path) -> Result<(), DataError> {
    let train = data::mnist(dir, true)?;
    let time = |d: &dyn Dataset| {
        let start = Instant::now();
        let mut loader = DataLoader::new(d, 64).shuffle(Rng::new(19));
        let pixels: usize = loader.epoch().map(|(x, _)| x.numel()).sum();
        (pixels, start.elapsed())
    };
    let (n, fast) = time(&train);
    let (m, slow) = time(&PerExample(&train));
    assert_eq!(n, m);
    println!("an epoch of batches: get_batch {fast:.2?}, get and stack {slow:.2?}");
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let default = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../data"));
    let dir = args.iter().find(|a| a.contains('/')).map(PathBuf::from);
    let dir = dir.unwrap_or(default);
    let result = match args.first().map(String::as_str) {
        Some("stats") => stats(&dir),
        Some("loader") => loader(&dir),
        _ => train(&dir),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        eprintln!(
            "(download and gunzip the MNIST files into {})",
            dir.display()
        );
        std::process::exit(1);
    }
}
