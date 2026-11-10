//! Challenge 19.2, moved here in Chapter 20: some of a dataset's examples, and a
//! seeded split into disjoint parts, for a validation set.

use super::{Batch, Dataset};
use crate::rng::Rng;
use crate::tensor::Tensor;

// @listing subset
/// The examples of `dataset` at `indices`, renumbered from 0.
pub struct Subset<'d, D: Dataset + ?Sized> {
    pub dataset: &'d D,
    pub indices: Vec<usize>,
}

impl<D: Dataset + ?Sized> Dataset for Subset<'_, D> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, i: usize) -> (Tensor, usize) {
        self.dataset.get(self.indices[i])
    }

    fn get_batch(&self, indices: &[usize]) -> Batch {
        let inner: Vec<usize> = indices.iter().map(|&i| self.indices[i]).collect();
        self.dataset.get_batch(&inner) // keeps the dataset's fast path
    }
}
// @end

// @listing random-split
/// Disjoint random subsets of the given sizes, which must add up to the
/// dataset's length: one shuffle, cut into pieces.
pub fn random_split<'d, D: Dataset + ?Sized>(
    dataset: &'d D,
    sizes: &[usize],
    rng: &mut Rng,
) -> Vec<Subset<'d, D>> {
    let total: usize = sizes.iter().sum();
    assert_eq!(total, dataset.len(), "random_split: sizes {sizes:?}");
    let mut order: Vec<usize> = (0..total).collect();
    rng.shuffle(&mut order);
    let mut rest = &order[..];
    let mut pieces = vec![];
    for &n in sizes {
        let (piece, tail) = rest.split_at(n);
        let indices = piece.to_vec();
        pieces.push(Subset { dataset, indices });
        rest = tail;
    }
    pieces
}
// @end
