//! Chapter 22: a small convolutional network on MNIST, trained with Chapter 20's
//! loop, and compared with Chapter 19's MLP on digits moved a few pixels.
//!
//! cargo run --release --example mnist_cnn             train the CNN, 10 epochs
//! cargo run --release --example mnist_cnn -- shifts   the MLP and the CNN on
//!                                                     shifted test digits

#[path = "mnist_train.rs"]
#[allow(dead_code)]
pub mod mnist_train;

use ember::data::{self, DataError};
use ember::{
    AdamW, CheckpointError, Conv2d, DataLoader, Flatten, Linear, MaxPool2d, Module,
    ReLU, Rng, Sequential, Tensor, TensorDataset, Unflatten, Window, evaluate,
    random_split,
};
use mnist_train::{Plan, Progress, fit, mlp};
use std::path::{Path, PathBuf};
use std::time::Instant;

// @listing cnn
/// Two blocks of a 3 × 3 convolution, a ReLU and 2 × 2 max pooling, then a
/// `Linear` layer on the 16 maps of 7 × 7 that remain.
pub fn cnn(rng: &mut Rng) -> Sequential {
    let same = Window::new(3, 1, 1); // 3 × 3, stride 1, padding 1: same size
    Sequential::new(vec![
        Box::new(Unflatten(vec![1, 28, 28])), // [N, 784] → [N, 1, 28, 28]
        Box::new(Conv2d::new(1, 8, same, rng)), // [N, 8, 28, 28]
        Box::new(ReLU),
        Box::new(MaxPool2d::new(2)), // [N, 8, 14, 14]
        Box::new(Conv2d::new(8, 16, same, rng)), // [N, 16, 14, 14]
        Box::new(ReLU),
        Box::new(MaxPool2d::new(2)), // [N, 16, 7, 7]
        Box::new(Flatten),           // [N, 784]
        Box::new(Linear::new(16 * 7 * 7, 10, rng)),
    ])
}
// @end

/// The digits of `ds` moved `dx` pixels right (left if negative) and `dy`
/// down, the pixels moved in filled with the background, `fill`.
pub fn shifted(
    ds: &TensorDataset,
    (dy, dx): (isize, isize),
    fill: f32,
) -> TensorDataset {
    let (x, labels) = ds.tensors();
    let src = x.as_slice().unwrap();
    let mut out = vec![fill; src.len()];
    let (images, _) = src.as_chunks::<784>(); // whole images, as arrays
    for (image, from) in out.as_chunks_mut::<784>().0.iter_mut().zip(images) {
        for (i, v) in image.iter_mut().enumerate() {
            let (y, x) = ((i / 28) as isize - dy, (i % 28) as isize - dx);
            if (0..28).contains(&y) && (0..28).contains(&x) {
                *v = from[(y * 28 + x) as usize];
            }
        }
    }
    TensorDataset::new(Tensor::from_vec(out, x.shape()), labels.to_vec())
}

/// Any error this example reports.
#[derive(Debug)]
enum Error {
    Data(DataError),
    Checkpoint(CheckpointError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Data(e) => write!(f, "{e}"),
            Error::Checkpoint(e) => write!(f, "{e}"),
        }
    }
}

impl From<DataError> for Error {
    fn from(e: DataError) -> Error {
        Error::Data(e)
    }
}

impl From<CheckpointError> for Error {
    fn from(e: CheckpointError) -> Error {
        Error::Checkpoint(e)
    }
}

/// Section 20.8's run, ten epochs, for the network `build` makes from seed 20;
/// returns the trained model and its test set.
fn train(
    dir: &Path,
    name: &str,
    build: fn(&mut Rng) -> Sequential,
) -> Result<(Sequential, TensorDataset), Error> {
    let (full, test) = (data::mnist(dir, true)?, data::mnist(dir, false)?);
    let dir = dir.join("checkpoints").join(name);
    std::fs::create_dir_all(&dir).map_err(|source| {
        let path = dir.clone();
        CheckpointError::Io { path, source }
    })?;
    let plan = Plan {
        epochs: 10,
        stop: 10,
        batch: 64,
        warmup: 100,
        dir,
    };
    let mut rng = Rng::new(20);
    let parts = random_split(&full, &[55_000, 5_000], &mut rng);
    let mut model = build(&mut rng);
    let count: usize = model.parameters().iter().map(|p| p.value.numel()).sum();
    println!("{name}: {count} parameters\nepoch     loss  val loss  val acc");
    let mut opt = AdamW::new(1e-3);
    let mut at = Progress::new(rng);
    let start = Instant::now();
    fit(
        (&mut model, &mut opt),
        (&parts[0], &parts[1]),
        &mut at,
        &plan,
    )?;
    let secs = start.elapsed().as_secs_f64();
    let (loss, accuracy) = evaluate(&mut model, &mut DataLoader::new(&test, 1000));
    println!("last model on the test set: loss {loss:.4}, accuracy {accuracy:.4}");
    println!("trained in {secs:.1}s, {:.2}s per epoch", secs / 10.0);
    Ok((model, test))
}

/// The test accuracy with every digit moved by 0 to 4 pixels to the right.
fn shift_row(model: &mut Sequential, test: &TensorDataset) -> Vec<f32> {
    let fill = (0.0 - data::MNIST_MEAN) / data::MNIST_STD; // the background
    let accuracy = |model: &mut Sequential, d| {
        let moved = shifted(test, (0, d), fill);
        evaluate(model, &mut DataLoader::new(&moved, 1000)).1
    };
    (0..5).map(|d| accuracy(model, d)).collect()
}

fn shifts(dir: &Path) -> Result<(), Error> {
    let (mut mlp, test) = train(dir, "mlp", mlp)?;
    let (mut cnn, _) = train(dir, "cnn", cnn)?;
    println!("\nshift      0      1      2      3      4");
    for (name, model) in [("MLP", &mut mlp), ("CNN", &mut cnn)] {
        let row = shift_row(model, &test);
        let cells: Vec<String> = row.iter().map(|a| format!("{a:.4}")).collect();
        println!("{name:<5} {}", cells.join(" "));
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let default = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../data"));
    let dir = args.iter().find(|a| a.contains('/')).map(PathBuf::from);
    let dir = dir.unwrap_or(default);
    let result = match args.first().map(String::as_str) {
        Some("shifts") => shifts(&dir),
        _ => train(&dir, "cnn", cnn).map(|_| ()),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
