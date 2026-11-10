//! Chapter 23: MNIST with and without normalization layers, trained with plain
//! SGD and momentum over a range of learning rates.
//!
//! cargo run --release --example mnist_norm           the learning-rate sweep
//! cargo run --release --example mnist_norm -- 5      the same, 5 epochs per run

use ember::data::{self, DataError};
use ember::{
    BatchNorm1d, DataLoader, Dataset, LayerNorm, Linear, Module, ReLU, Rng,
    Sequential, Sgd, evaluate, random_split, train_epoch, train_step,
};
use std::path::PathBuf;
use std::time::Instant;

/// Which normalization layer follows each hidden `Linear` layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Norm {
    None,
    Batch,
    Layer,
}

// @listing norm-mlp
/// Linear layers between the `sizes`; each hidden one is followed by the
/// normalization layer `norm` (if any) and a ReLU.
pub fn mlp_norm(sizes: &[usize], norm: Norm, rng: &mut Rng) -> Sequential {
    let mut layers: Vec<Box<dyn Module>> = vec![];
    for (i, pair) in sizes.windows(2).enumerate() {
        let n = pair[1];
        layers.push(Box::new(Linear::new(pair[0], n, rng)));
        if i + 2 < sizes.len() {
            match norm {
                Norm::None => {}
                Norm::Batch => layers.push(Box::new(BatchNorm1d::new(n))),
                Norm::Layer => layers.push(Box::new(LayerNorm::new(n))),
            }
            layers.push(Box::new(ReLU));
        }
    }
    Sequential::new(layers)
}
// @end

/// The network of the sweep: three hidden layers of 256.
pub const SIZES: [usize; 5] = [784, 256, 256, 256, 10];

/// Train `epochs` epochs of SGD with momentum 0.9 at a constant rate `lr`, no
/// clipping; the validation accuracy after each epoch, or NaN once diverged.
fn run<D: Dataset + ?Sized>(
    (train, val): (&D, &D),
    norm: Norm,
    lr: f32,
    epochs: usize,
) -> Vec<f32> {
    let mut rng = Rng::new(23);
    let mut model = mlp_norm(&SIZES, norm, &mut rng);
    let mut opt = Sgd::new(lr).momentum(0.9);
    let loader = DataLoader::new(train, 64).drop_last(true);
    let mut loader = loader.shuffle(rng);
    let mut row = vec![];
    for _ in 0..epochs {
        let loss = train_epoch(&mut model, &mut loader, |m, batch| {
            train_step(m, batch, &mut opt, f32::INFINITY)
        });
        let (_, acc) = evaluate(&mut model, &mut DataLoader::new(val, 1000));
        row.push(if loss.is_finite() { acc } else { f32::NAN });
    }
    row
}

fn sweep(dir: &PathBuf, epochs: usize) -> Result<(), DataError> {
    let full = data::mnist(dir, true)?;
    let parts = random_split(&full, &[55_000, 5_000], &mut Rng::new(20));
    let data = (&parts[0], &parts[1]);
    println!("validation accuracy after {epochs} epochs ({SIZES:?})");
    println!(
        "{:<6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "lr", 0.01, 0.03, 0.1, 0.3, 1.0
    );
    for norm in [Norm::None, Norm::Batch, Norm::Layer] {
        let start = Instant::now();
        let mut cells = vec![];
        for lr in [0.01, 0.03, 0.1, 0.3, 1.0] {
            let row = run(data, norm, lr, epochs);
            let last = *row.last().unwrap();
            cells.push(if last.is_nan() {
                "  diverged".into()
            } else {
                format!("{last:>8.4}")
            });
        }
        let secs = start.elapsed().as_secs_f64();
        println!(
            "{:<6} {}  ({secs:.1}s)",
            format!("{norm:?}"),
            cells.join(" ")
        );
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let default = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../data"));
    let dir = args.iter().find(|a| a.contains('/')).map(PathBuf::from);
    let dir = dir.unwrap_or(default);
    let epochs = args
        .iter()
        .find_map(|a| a.parse::<usize>().ok())
        .unwrap_or(3);
    if let Err(e) = sweep(&dir, epochs) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
