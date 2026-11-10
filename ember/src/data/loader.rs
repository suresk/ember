//! Chapter 19: a data loader cuts a dataset into batches, in a new random order
//! every epoch.

use super::Dataset;
use crate::rng::Rng;
use crate::tensor::Tensor;

/// A batch: the inputs stacked along a first axis, and their classes.
pub type Batch = (Tensor, Vec<usize>);

// @listing loader
/// Batches of `batch_size` examples from a borrowed dataset. With a generator
/// the order is shuffled at the start of every epoch; `drop_last` skips a short
/// final batch.
pub struct DataLoader<'d, D: Dataset + ?Sized> {
    dataset: &'d D,
    batch_size: usize,
    rng: Option<Rng>,
    drop_last: bool,
}

impl<'d, D: Dataset + ?Sized> DataLoader<'d, D> {
    /// One pass over the dataset, in a new order if there is a generator.
    pub fn epoch(&mut self) -> Batches<'d, D> {
        let mut order: Vec<usize> = (0..self.dataset.len()).collect();
        if let Some(rng) = &mut self.rng {
            rng.shuffle(&mut order);
        }
        order.truncate(self.len() * self.batch_size); // drops a short batch
        let batch_size = self.batch_size;
        Batches {
            dataset: self.dataset,
            order,
            batch_size,
            at: 0,
        }
    }
}
// @end

impl<'d, D: Dataset + ?Sized> DataLoader<'d, D> {
    pub fn new(dataset: &'d D, batch_size: usize) -> DataLoader<'d, D> {
        assert!(batch_size > 0, "DataLoader: a batch size of 0");
        DataLoader {
            dataset,
            batch_size,
            rng: None,
            drop_last: false,
        }
    }

    /// Shuffle every epoch, drawing from `rng`.
    pub fn shuffle(mut self, rng: Rng) -> DataLoader<'d, D> {
        self.rng = Some(rng);
        self
    }

    pub fn drop_last(mut self, on: bool) -> DataLoader<'d, D> {
        self.drop_last = on;
        self
    }

    /// Batches per epoch: ⌈n / b⌉, or ⌊n / b⌋ when the short one is dropped.
    pub fn len(&self) -> usize {
        let (n, b) = (self.dataset.len(), self.batch_size);
        if self.drop_last { n / b } else { n.div_ceil(b) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The shuffling generator, if there is one: Chapter 20 saves its state, so
    /// that a resumed run draws the same orders.
    pub fn rng(&self) -> Option<&Rng> {
        self.rng.as_ref()
    }
}

// @listing batches
/// The batches of one epoch, in order.
pub struct Batches<'d, D: Dataset + ?Sized> {
    dataset: &'d D,
    order: Vec<usize>,
    batch_size: usize,
    at: usize, // where the next batch starts in `order`
}

impl<D: Dataset + ?Sized> Iterator for Batches<'_, D> {
    type Item = Batch;

    fn next(&mut self) -> Option<Batch> {
        if self.at == self.order.len() {
            return None;
        }
        let to = (self.at + self.batch_size).min(self.order.len());
        let batch = self.dataset.get_batch(&self.order[self.at..to]);
        self.at = to;
        Some(batch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = (self.order.len() - self.at).div_ceil(self.batch_size);
        (left, Some(left))
    }
}

impl<D: Dataset + ?Sized> ExactSizeIterator for Batches<'_, D> {}
// @end
