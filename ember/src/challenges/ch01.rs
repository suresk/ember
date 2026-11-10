//! Chapter 1 challenges.

use crate::tensor::Tensor;
use std::fmt;
use std::rc::Rc;

// @listing unravel-index
/// The index whose row-major position in `shape` is `flat`: the inverse of
/// `storage_index` for a fresh tensor.
pub fn unravel_index(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let numel: usize = shape.iter().product();
    assert!(
        flat < numel,
        "position {flat} out of range for shape {shape:?}"
    );
    let mut index = vec![0; shape.len()];
    for d in (0..shape.len()).rev() {
        index[d] = flat % shape[d];
        flat /= shape[d];
    }
    index
}
// @end

impl Tensor {
    // @listing full
    pub fn full(shape: &[usize], value: f32) -> Tensor {
        Tensor::from_vec(vec![value; shape.iter().product()], shape)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }
    // @end

    // @listing eye
    pub fn eye(n: usize) -> Tensor {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0; // row i, column i
        }
        Tensor::from_vec(data, &[n, n])
    }
    // @end

    // @listing from-fn
    pub fn from_fn(shape: &[usize], f: impl Fn(&[usize]) -> f32) -> Tensor {
        let numel: usize = shape.iter().product();
        let data = (0..numel).map(|i| f(&unravel_index(i, shape))).collect();
        Tensor::from_vec(data, shape)
    }
    // @end

    // @listing set
    /// Write `value` at `index`. If another tensor shares the storage, this tensor
    /// first gets its own copy, so the other one never sees the write.
    pub fn set(&mut self, index: &[usize], value: f32) {
        let at = self.storage_index(index);
        Rc::make_mut(&mut self.storage)[at] = value;
    }
    // @end

    // @listing allclose
    pub fn allclose(&self, other: &Tensor, tol: f32) -> bool {
        self.shape == other.shape
            && (0..self.numel()).all(|i| {
                let index = unravel_index(i, &self.shape);
                (self.get(&index) - other.get(&index)).abs() <= tol
            })
    }
    // @end

    // @listing fmt-axis
    /// Write axis `d` and everything inside it, with `index[..d]` already fixed.
    fn fmt_axis(
        &self,
        f: &mut fmt::Formatter,
        index: &mut [usize],
        d: usize,
    ) -> fmt::Result {
        if d == self.ndim() {
            return write!(f, "{}", self.get(index));
        }
        write!(f, "[")?;
        for i in 0..self.shape[d] {
            if i > 0 && d + 1 == self.ndim() {
                write!(f, ", ")?;
            } else if i > 0 {
                write!(f, ",\n{}", " ".repeat(d + 1))?; // line up under the [
            }
            index[d] = i;
            self.fmt_axis(f, index, d + 1)?;
        }
        write!(f, "]")
    }
    // @end
}

// @listing display
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut index = vec![0; self.ndim()];
        self.fmt_axis(f, &mut index, 0)
    }
}
// @end
