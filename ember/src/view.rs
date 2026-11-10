//! Chapter 2: strides and views. Every method here except `contiguous` (and `reshape`
//! when it has to fall back on it) returns a tensor that shares storage with `self`:
//! only the shape, strides and offset change.

use crate::tensor::{Tensor, contiguous_strides};
use std::rc::Rc;

impl Tensor {
    /// Swap axes `a` and `b`. No data moves.
    pub fn transpose(&self, a: usize, b: usize) -> Tensor {
        // @solution ch02 transpose
        assert!(
            a < self.ndim() && b < self.ndim(),
            "transpose({a}, {b}) on {} axes",
            self.ndim()
        );
        let mut t = self.clone();
        t.shape.swap(a, b);
        t.strides.swap(a, b);
        t
        // @end
    }

    /// Reorder the axes: axis `d` of the result is axis `order[d]` of `self`.
    pub fn permute(&self, order: &[usize]) -> Tensor {
        // @solution ch02 permute
        let n = self.ndim();
        let mut seen = vec![false; n];
        assert_eq!(order.len(), n, "permute: {order:?} for {n} axes");
        for &d in order {
            assert!(
                d < n && !seen[d],
                "permute: {order:?} is not a permutation of 0..{n}"
            );
            seen[d] = true;
        }
        Tensor {
            storage: Rc::clone(&self.storage),
            shape: order.iter().map(|&d| self.shape[d]).collect(),
            strides: order.iter().map(|&d| self.strides[d]).collect(),
            offset: self.offset,
        }
        // @end
    }

    /// Keep positions `start, start + step, …` below `end` on axis `dim`, like
    /// `x[..., start:end:step, ...]` in NumPy.
    pub fn slice(&self, dim: usize, start: usize, end: usize, step: usize) -> Tensor {
        // @solution ch02 slice
        assert!(dim < self.ndim(), "slice: axis {dim} of {}", self.ndim());
        assert!(step >= 1, "slice: step must be at least 1");
        assert!(
            start <= end && end <= self.shape[dim],
            "slice: {start}..{end} on an axis of size {}",
            self.shape[dim]
        );
        let mut t = self.clone();
        t.offset += start * self.strides[dim];
        t.shape[dim] = (end - start).div_ceil(step);
        t.strides[dim] *= step;
        t
        // @end
    }

    /// Fix axis `dim` at position `i` and drop it, like `x[..., i, ...]`.
    pub fn select(&self, dim: usize, i: usize) -> Tensor {
        // @solution ch02 select
        assert!(
            i < self.shape[dim],
            "select: index {i} on an axis of size {}",
            self.shape[dim]
        );
        let mut t = self.clone();
        t.offset += i * self.strides[dim];
        t.shape.remove(dim);
        t.strides.remove(dim);
        t
        // @end
    }

    /// True if walking the elements in row-major order visits consecutive storage
    /// slots. Axes of size 1 are never stepped along, so their stride doesn't matter.
    pub fn is_contiguous(&self) -> bool {
        // @solution ch02 is_contiguous
        let mut want = 1; // the row-major stride, built up from the last axis
        for (&n, &s) in self.shape.iter().zip(&self.strides).rev() {
            if n != 1 && s != want {
                return false;
            }
            want *= n;
        }
        true
        // @end
    }

    /// The elements in row-major order of `self`'s shape, wherever they are stored.
    pub fn to_vec(&self) -> Vec<f32> {
        // @solution ch02 to_vec
        let mut out = Vec::with_capacity(self.numel());
        if self.numel() == 0 {
            return out;
        }
        // An odometer over the index: bump the last axis, carry into earlier ones.
        let mut index = vec![0; self.ndim()];
        let mut at = self.offset;
        loop {
            out.push(self.storage[at]);
            let mut d = self.ndim();
            loop {
                if d == 0 {
                    return out;
                }
                d -= 1;
                index[d] += 1;
                at += self.strides[d];
                if index[d] < self.shape[d] {
                    break;
                }
                at -= index[d] * self.strides[d];
                index[d] = 0;
            }
        }
        // @end
    }

    /// The same elements laid out row-major: `self` itself (a shared view) if it
    /// already is, otherwise a copy.
    pub fn contiguous(&self) -> Tensor {
        // @solution ch02 contiguous
        if self.is_contiguous() {
            self.clone()
        } else {
            Tensor::from_vec(self.to_vec(), &self.shape)
        }
        // @end
    }

    /// The same elements, read in row-major order into a new shape. A view when
    /// `self` is contiguous, a copy otherwise.
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        // @solution ch02 reshape
        let numel: usize = shape.iter().product();
        assert_eq!(
            numel,
            self.numel(),
            "reshape: {:?} to {:?}",
            self.shape,
            shape
        );
        let src = self.contiguous();
        Tensor {
            storage: src.storage,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: src.offset,
        }
        // @end
    }
}

impl Tensor {
    // @listing unfold
    /// Every window of `size` along `dim`, `step` apart, as a view with one more
    /// axis: axis `dim` counts windows and the new last axis runs along a window.
    pub fn unfold(&self, dim: usize, size: usize, step: usize) -> Tensor {
        let fits = size >= 1 && step >= 1 && size <= self.shape[dim];
        assert!(fits, "unfold: window {size}, step {step}, {:?}", self.shape);
        let mut t = self.clone();
        t.shape[dim] = (self.shape[dim] - size) / step + 1;
        t.strides[dim] = self.strides[dim] * step;
        t.shape.push(size);
        t.strides.push(self.strides[dim]);
        t
    }
    // @end
}
