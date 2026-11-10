//! Chapter 20: what a training loop measures.

use crate::tensor::Tensor;

// @listing accuracy
/// The fraction of rows of `logits`, `[N, C]`, whose largest entry is at the
/// row's label.
pub fn accuracy(logits: &Tensor, labels: &[usize]) -> f32 {
    let (shape, n) = (logits.shape(), labels.len());
    let fits = shape.len() == 2 && shape[0] == n;
    assert!(fits, "accuracy: {shape:?}, {n} labels");
    let predicted = logits.argmax(1); // a class per row, as usize
    let right = predicted.iter().zip(labels).filter(|(p, l)| p == l).count();
    right as f32 / n as f32
}
// @end

// @listing average
/// A mean over examples, from the means of batches of any size.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Average {
    sum: f64,     // of mean × examples, over the batches so far
    count: usize, // examples so far
}

impl Average {
    pub fn add(&mut self, mean: f32, n: usize) {
        self.sum += mean as f64 * n as f64;
        self.count += n;
    }
    pub fn mean(&self) -> f32 {
        (self.sum / self.count as f64) as f32 // NaN before the first batch
    }
}
// @end

impl Average {
    pub fn new() -> Average {
        Average::default()
    }

    pub fn count(&self) -> usize {
        self.count
    }
}
