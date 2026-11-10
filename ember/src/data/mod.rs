//! Chapter 19: datasets, batches, and the MNIST files.

mod loader;
mod mnist;
mod split;

pub use loader::{Batch, Batches, DataLoader};
pub use mnist::{
    DataError, Idx, MNIST_MEAN, MNIST_STD, mnist, mnist_images, mnist_labels,
    normalize, parse_idx,
};
pub use split::{Subset, random_split};

use crate::tensor::Tensor;

// @listing dataset
/// A numbered collection of examples, each an input and a class.
pub trait Dataset {
    /// How many examples there are.
    fn len(&self) -> usize;

    /// Example `i`: its input, and its class.
    fn get(&self, i: usize) -> (Tensor, usize);

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The examples at `indices`, their inputs stacked along a new first axis.
    /// This default calls `get` for each one and copies the inputs together; a
    /// dataset that keeps its examples in one buffer can do better.
    fn get_batch(&self, indices: &[usize]) -> (Tensor, Vec<usize>) {
        let (mut data, mut labels, mut shape) = (vec![], vec![], vec![0]);
        for &i in indices {
            let (x, y) = self.get(i);
            if shape.len() == 1 {
                shape.extend(x.shape()); // the first example sets the shape
            }
            data.extend(x.to_vec());
            labels.push(y);
        }
        shape[0] = indices.len();
        (Tensor::from_vec(data, &shape), labels)
    }
}
// @end

// @listing tensor-dataset
/// Examples held in memory: row `i` of `x` is example `i`'s input.
#[derive(Clone, Debug)]
pub struct TensorDataset {
    x: Tensor,
    labels: Vec<usize>,
}

impl TensorDataset {
    pub fn new(x: Tensor, labels: Vec<usize>) -> TensorDataset {
        let (rows, n) = (x.shape().first().copied().unwrap_or(0), labels.len());
        assert!(rows == n, "TensorDataset: {rows} rows but {n} labels");
        let x = x.contiguous();
        TensorDataset { x, labels }
    }

    /// All the inputs, `[n, …]`, and all the labels.
    pub fn tensors(&self) -> (&Tensor, &[usize]) {
        (&self.x, &self.labels)
    }
}
// @end

// @listing tensor-dataset-impl
impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.labels.len()
    }

    fn get(&self, i: usize) -> (Tensor, usize) {
        (self.x.select(0, i), self.labels[i]) // a view: nothing is copied
    }

    /// One copy per row, straight from the buffer into the batch.
    fn get_batch(&self, indices: &[usize]) -> (Tensor, Vec<usize>) {
        let all = self.x.as_slice().expect("contiguous, made so by `new`");
        let row = self.x.numel() / self.len().max(1);
        let mut data = Vec::with_capacity(indices.len() * row);
        for &i in indices {
            data.extend_from_slice(&all[i * row..(i + 1) * row]);
        }
        let mut shape = self.x.shape().to_vec();
        shape[0] = indices.len();
        let labels = indices.iter().map(|&i| self.labels[i]).collect();
        (Tensor::from_vec(data, &shape), labels)
    }
}
// @end
