//! The examples printed in Chapter 1 of the guided edition. Each `@listing` region is
//! a listing in the book, so every number the chapter quotes is checked here.
use ember::{Tensor, contiguous_strides};

#[test]
fn row_major_strides() {
    // @listing strides-3d
    assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
    assert_eq!(contiguous_strides(&[5]), vec![1]);
    assert_eq!(contiguous_strides(&[]), Vec::<usize>::new()); // a scalar
    // @end
}

#[test]
fn first_tensor() {
    // @listing first-tensor
    let x = Tensor::arange(&[2, 3]);
    assert_eq!(x.shape(), &[2, 3]);
    assert_eq!(x.strides(), &[3, 1]);
    assert_eq!((x.ndim(), x.numel(), x.offset()), (2, 6, 0));

    let s = Tensor::from_vec(vec![4.0], &[]); // a scalar
    assert_eq!((s.ndim(), s.numel()), (0, 1));
    // @end
}

// @listing bad-length
#[test]
#[should_panic(expected = "from_vec: 5 values for shape [2, 3]")]
fn from_vec_checks_the_length() {
    Tensor::from_vec(vec![0.0; 5], &[2, 3]);
}
// @end

#[test]
#[should_panic(expected = "has more than usize::MAX elements")]
fn a_shape_too_big_for_usize_panics() {
    // 2^32 · 2^32 = 2^64 wraps to 0 in a release build, which would match vec![].
    Tensor::from_vec(vec![], &[1 << 32, 1 << 32]);
}

#[test]
fn clones_share_storage() {
    // @listing clone-shares
    let x = Tensor::arange(&[2, 3]);
    let y = x.clone(); // bumps a count; copies shape, strides and offset
    assert!(x.shares_storage(&y));

    let z = Tensor::arange(&[2, 3]); // same values, its own buffer
    assert!(!x.shares_storage(&z));
    // @end
}

#[test]
fn reading_elements() {
    // @listing read
    let x = Tensor::arange(&[2, 3]);
    assert_eq!(x.storage_index(&[1, 2]), 5); // 1 * 3 + 2 * 1
    assert_eq!(x.get(&[1, 2]), 5.0);

    let x = Tensor::arange(&[2, 3, 4]);
    assert_eq!(x.storage_index(&[1, 2, 3]), 23); // 1 * 12 + 2 * 4 + 3 * 1
    assert_eq!(x.get(&[1, 2, 3]), 23.0);
    // @end
}

// @listing bad-index
#[test]
#[should_panic(expected = "index 2 out of range for axis 0 of size 2")]
fn index_past_the_end_panics() {
    let x = Tensor::arange(&[2, 3]);
    x.get(&[2, 0]);
}
// @end
