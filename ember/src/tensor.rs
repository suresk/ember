//! Chapter 1: a tensor is a buffer and a shape.

use std::rc::Rc;

// @listing tensor-struct
/// An n-dimensional array of `f32`.
///
/// The numbers live in `storage`, a flat buffer that several tensors may share.
/// `shape`, `strides` and `offset` say how to find element `[i0, i1, …]` in it:
/// `offset + i0 * strides[0] + i1 * strides[1] + …`.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) storage: Rc<Vec<f32>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
}
// @end

/// Row-major (C order) strides for `shape`: the last axis has stride 1, and each
/// earlier axis steps over one whole copy of everything after it.
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    // @solution ch01 contiguous_strides
    let mut strides = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
    // @end
}

impl Tensor {
    // @listing constructors
    /// A tensor that owns `data`, laid out row-major with the given shape.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Tensor {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "from_vec: {} values for shape {:?}",
            data.len(),
            shape
        );
        Tensor {
            storage: Rc::new(data),
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: 0,
        }
    }

    /// The values 0, 1, 2, … laid out in `shape`. Handy in tests: each value is its
    /// own row-major position.
    pub fn arange(shape: &[usize]) -> Tensor {
        let numel: usize = shape.iter().product();
        Tensor::from_vec((0..numel).map(|i| i as f32).collect(), shape)
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::from_vec(vec![0.0; shape.iter().product()], shape)
    }
    // @end

    // @listing accessors
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// True if the two tensors read from the same buffer.
    pub fn shares_storage(&self, other: &Tensor) -> bool {
        Rc::ptr_eq(&self.storage, &other.storage)
    }
    // @end

    /// Where element `index` lives in the storage buffer.
    pub fn storage_index(&self, index: &[usize]) -> usize {
        // @solution ch01 storage_index
        assert_eq!(
            index.len(),
            self.ndim(),
            "index {:?} for a tensor of shape {:?}",
            index,
            self.shape
        );
        let mut at = self.offset;
        for (d, (&i, &n)) in index.iter().zip(&self.shape).enumerate() {
            assert!(i < n, "index {i} out of range for axis {d} of size {n}");
            at += i * self.strides[d];
        }
        at
        // @end
    }

    // @listing get
    pub fn get(&self, index: &[usize]) -> f32 {
        self.storage[self.storage_index(index)]
    }
    // @end
}

// @listing data-mut
impl Tensor {
    /// The tensor's numbers in row-major order, to write in place. A view that
    /// isn't one run of its buffer is first copied into a buffer of its own, and a
    /// buffer that anything else shares is copied by `Rc::make_mut` (clone on
    /// write), so a write never shows through another tensor.
    pub fn data_mut(&mut self) -> &mut [f32] {
        if !self.is_contiguous() {
            *self = Tensor::from_vec(self.to_vec(), &self.shape);
        }
        let (start, end) = (self.offset, self.offset + self.numel());
        &mut Rc::make_mut(&mut self.storage)[start..end] // copies only if shared
    }
}
// @end
