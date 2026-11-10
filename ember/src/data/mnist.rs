//! Chapter 19: reading MNIST's IDX files with the standard library.

use super::TensorDataset;
use crate::tensor::Tensor;
use std::path::{Path, PathBuf};
use std::{fmt, io};

// @listing error
/// Everything that can go wrong while loading a dataset from files: a file that
/// can't be read, a magic number or a size that is not the expected one, a label
/// that is not a digit, and image and label files of different lengths.
#[derive(Debug)]
pub enum DataError {
    Io { path: PathBuf, source: io::Error },
    BadMagic { expected: u32, found: u32 },
    WrongSize { expected: usize, found: usize },
    BadLabel { index: usize, label: u8 },
    Mismatch { images: usize, labels: usize },
}
// @end

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use DataError::*;
        match self {
            Io { path, source } => write!(f, "{}: {source}", path.display()),
            BadMagic { expected, found } => {
                write!(f, "magic number {found:#010x}, expected {expected:#010x}")
            }
            WrongSize { expected, found } => {
                write!(f, "{found} bytes, but the header says {expected}")
            }
            BadLabel { index, label } => write!(f, "label {index} is {label}"),
            Mismatch { images, labels } => {
                write!(f, "{images} images, {labels} labels")
            }
        }
    }
}

impl std::error::Error for DataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DataError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// An IDX file's sizes, and its data.
pub type Idx<'a> = (Vec<usize>, &'a [u8]);

// @listing parse-idx
/// An IDX file of unsigned bytes with `ndim` dimensions, its data borrowed from
/// `bytes`. The header is the magic number `0x0000_08nn` (type 0x08, unsigned
/// byte; nn dimensions), then each size, all big-endian `u32`s.
pub fn parse_idx(bytes: &[u8], ndim: usize) -> Result<Idx<'_>, DataError> {
    let word = |k: usize| match bytes.get(4 * k..4 * k + 4) {
        Some(four) => u32::from_be_bytes(four.try_into().unwrap()),
        None => 0, // past the end: the size check below reports it
    };
    let (expected, found) = (0x0800 | ndim as u32, word(0));
    if found != expected {
        return Err(DataError::BadMagic { expected, found });
    }
    let dims: Vec<usize> = (1..=ndim).map(|k| word(k) as usize).collect();
    let header = 4 * (1 + ndim);
    let (expected, found) = (header + dims.iter().product::<usize>(), bytes.len());
    if found != expected {
        return Err(DataError::WrongSize { expected, found });
    }
    Ok((dims, &bytes[header..]))
}
// @end

/// The mean and standard deviation of every pixel of MNIST's 60,000 training
/// images, scaled to [0, 1].
pub const MNIST_MEAN: f32 = 0.1307;
pub const MNIST_STD: f32 = 0.3081;

// @listing images
/// An images file as rows of `[n, rows · cols]`, each pixel scaled to [0, 1].
pub fn mnist_images(bytes: &[u8]) -> Result<Tensor, DataError> {
    let (dims, pixels) = parse_idx(bytes, 3)?;
    let x = pixels.iter().map(|&p| p as f32 / 255.0).collect();
    Ok(Tensor::from_vec(x, &[dims[0], dims[1] * dims[2]]))
}

/// A labels file, each label checked to be a digit.
pub fn mnist_labels(bytes: &[u8]) -> Result<Vec<usize>, DataError> {
    let (_, labels) = parse_idx(bytes, 1)?;
    if let Some(index) = labels.iter().position(|&l| l > 9) {
        let label = labels[index];
        return Err(DataError::BadLabel { index, label });
    }
    Ok(labels.iter().map(|&l| l as usize).collect())
}

/// `(x − mean) / std`, elementwise.
pub fn normalize(x: &Tensor, mean: f32, std: f32) -> Tensor {
    x.map(|v| (v - mean) / std)
}
// @end

// @listing mnist
/// The training set (60,000 digits) or the test set (10,000) from the four
/// uncompressed files in `dir`, normalized with the training set's mean and
/// standard deviation.
pub fn mnist(dir: impl AsRef<Path>, train: bool) -> Result<TensorDataset, DataError> {
    let prefix = if train { "train" } else { "t10k" };
    let read = |name: &str| {
        let path = dir.as_ref().join(format!("{prefix}-{name}"));
        std::fs::read(&path).map_err(|source| DataError::Io { path, source })
    };
    let x = mnist_images(&read("images-idx3-ubyte")?)?;
    let labels = mnist_labels(&read("labels-idx1-ubyte")?)?;
    if x.shape()[0] != labels.len() {
        let (images, labels) = (x.shape()[0], labels.len());
        return Err(DataError::Mismatch { images, labels });
    }
    let x = normalize(&x, MNIST_MEAN, MNIST_STD);
    Ok(TensorDataset::new(x, labels))
}
// @end
