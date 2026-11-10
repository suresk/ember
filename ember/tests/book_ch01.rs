//! The examples printed in Chapter 1 of the guided edition. Each `@listing` region is
//! a listing in the book, so every number the chapter quotes is checked here.
use ember::{Tensor, contiguous_strides};

#[test]
fn first_tensor() {
    // @listing first-tensor
    let x = Tensor::arange(&[2, 3]);
    assert_eq!(x.shape(), &[2, 3]);
    assert_eq!(x.strides(), &[3, 1]);
    assert_eq!(x.storage_index(&[1, 2]), 5); // 1 * 3 + 2 * 1
    assert_eq!(x.get(&[1, 2]), 5.0);
    // @end
}

#[test]
fn strides_in_three_dimensions() {
    // @listing strides-3d
    assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
    assert_eq!(contiguous_strides(&[5]), vec![1]);
    assert_eq!(contiguous_strides(&[]), Vec::<usize>::new()); // a scalar

    let x = Tensor::arange(&[2, 3, 4]);
    assert_eq!(x.storage_index(&[1, 2, 3]), 23); // 1 * 12 + 2 * 4 + 3 * 1
    // @end
}

#[test]
fn clones_share_storage() {
    // @listing clone-shares
    let x = Tensor::arange(&[2, 3]);
    let y = x.clone(); // copies one pointer and three small vectors
    assert!(x.shares_storage(&y));

    let z = Tensor::arange(&[2, 3]); // same values, its own buffer
    assert!(!x.shares_storage(&z));
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
