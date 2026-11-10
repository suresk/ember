//! Chapter 2 challenges.

use crate::tensor::{Tensor, contiguous_strides};
use std::rc::Rc;

impl Tensor {
    // @listing squeeze
    /// A new axis of size 1 at position `dim`. Its stride is never used; this one
    /// keeps a contiguous tensor contiguous.
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        assert!(
            dim <= self.ndim(),
            "unsqueeze: position {dim} of {}",
            self.ndim()
        );
        let stride = if dim < self.ndim() {
            self.strides[dim] * self.shape[dim]
        } else {
            1
        };
        let mut t = self.clone();
        t.shape.insert(dim, 1);
        t.strides.insert(dim, stride);
        t
    }

    /// Drop axis `dim`, which must have size 1.
    pub fn squeeze(&self, dim: usize) -> Tensor {
        assert_eq!(
            self.shape[dim], 1,
            "squeeze: axis {dim} has size {}",
            self.shape[dim]
        );
        let mut t = self.clone();
        t.shape.remove(dim);
        t.strides.remove(dim);
        t
    }
    // @end

    // @listing diagonal
    /// The main diagonal of a matrix, as a view: one step along it is one step down
    /// and one step right, so its stride is the sum of the two strides.
    pub fn diagonal(&self) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "diagonal: shape {:?} is not a matrix",
            self.shape
        );
        Tensor {
            storage: Rc::clone(&self.storage),
            shape: vec![self.shape[0].min(self.shape[1])],
            strides: vec![self.strides[0] + self.strides[1]],
            offset: self.offset,
        }
    }
    // @end

    // @listing chunk
    /// Split axis `dim` into pieces of `ceil(len / n)`; the last may be shorter.
    pub fn chunk(&self, dim: usize, n: usize) -> Vec<Tensor> {
        assert!(n >= 1, "chunk: n must be at least 1");
        let len = self.shape[dim];
        let size = len.div_ceil(n).max(1);
        (0..len)
            .step_by(size)
            .map(|start| self.slice(dim, start, (start + size).min(len), 1))
            .collect()
    }
    // @end

    // @listing view
    /// `self` read in a new shape without copying, or `None` when no strides can do
    /// it. Walk both shapes from the right in chunks: runs of old axes that are laid
    /// out back to back in storage, and that the new shape must split or merge
    /// exactly.
    pub fn view(&self, shape: &[usize]) -> Option<Tensor> {
        let numel: usize = shape.iter().product();
        assert_eq!(numel, self.numel(), "view: {:?} to {:?}", self.shape, shape);
        let make = |strides| Tensor {
            storage: Rc::clone(&self.storage),
            shape: shape.to_vec(),
            strides,
            offset: self.offset,
        };
        if numel == 0 || self.ndim() == 0 {
            return Some(make(contiguous_strides(shape)));
        }
        let mut strides = vec![0; shape.len()];
        let mut v = shape.len(); // new axes still to place are 0..v
        // The stride of the current chunk's last axis.
        let mut base = self.strides[self.ndim() - 1];
        let (mut old_n, mut new_n) = (1, 1);
        for d in (0..self.ndim()).rev() {
            old_n *= self.shape[d];
            // The chunk ends here unless axis d - 1 continues it: it is one block of
            // axes d.. apart in storage.
            let ends = d == 0
                || (self.shape[d - 1] != 1 && self.strides[d - 1] != old_n * base);
            if !ends {
                continue;
            }
            while v > 0 && (new_n < old_n || shape[v - 1] == 1) {
                strides[v - 1] = new_n * base;
                new_n *= shape[v - 1];
                v -= 1;
            }
            if new_n != old_n {
                return None; // a new axis straddles two chunks
            }
            if d > 0 {
                base = self.strides[d - 1];
                (old_n, new_n) = (1, 1);
            }
        }
        (v == 0).then(|| make(strides))
    }
    // @end
}
