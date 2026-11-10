//! The examples printed in Chapter 2 of the guided edition.
use ember::Tensor;

#[test]
fn transpose_example() {
    // @listing transpose-example
    let x = Tensor::arange(&[3, 4]);
    let y = x.transpose(0, 1);
    assert_eq!(y.shape(), &[4, 3]);
    assert_eq!(y.strides(), &[1, 4]);
    assert_eq!(y.get(&[2, 1]), 6.0); // the same slot as x[1, 2]
    assert!(y.shares_storage(&x));
    // @end
}

#[test]
fn permute_example() {
    // @listing permute-example
    let x = Tensor::arange(&[2, 3, 4]);
    let y = x.permute(&[2, 0, 1]);
    assert_eq!(y.shape(), &[4, 2, 3]);
    assert_eq!(y.strides(), &[1, 12, 4]);
    assert_eq!(y.get(&[3, 1, 2]), x.get(&[1, 2, 3])); // y[i, j, k] = x[j, k, i]
    // @end
}

#[test]
fn slice_example() {
    // @listing slice-example
    let x = Tensor::arange(&[4, 5]);
    let y = x.slice(0, 1, 3, 1).slice(1, 2, 5, 1); // x[1:3, 2:5]
    assert_eq!(y.shape(), &[2, 3]);
    assert_eq!(y.strides(), &[5, 1]);
    assert_eq!(y.offset(), 7); // 1 * 5 + 2 * 1
    assert_eq!(y.to_vec(), vec![7.0, 8.0, 9.0, 12.0, 13.0, 14.0]);
    // @end
}

#[test]
fn stepped_slice_example() {
    // @listing step-example
    let x = Tensor::arange(&[10]);
    let y = x.slice(0, 0, 10, 3); // x[0:10:3]
    assert_eq!(y.shape(), &[4]); // 0, 3, 6, 9: ceil(10 / 3)
    assert_eq!(y.strides(), &[3]);
    assert_eq!(y.to_vec(), vec![0.0, 3.0, 6.0, 9.0]);
    // @end
}

#[test]
fn select_example() {
    // @listing select-example
    let x = Tensor::arange(&[3, 4]);
    let row = x.select(0, 1); // x[1]
    assert_eq!(
        (row.shape(), row.strides(), row.offset()),
        (&[4][..], &[1][..], 4)
    );
    let col = x.select(1, 2); // x[:, 2]
    assert_eq!(
        (col.shape(), col.strides(), col.offset()),
        (&[3][..], &[4][..], 2)
    );
    assert_eq!(col.to_vec(), vec![2.0, 6.0, 10.0]);
    // @end
}

#[test]
fn contiguity_example() {
    // @listing contiguity-example
    let x = Tensor::arange(&[3, 4]);
    assert!(x.is_contiguous());
    assert!(!x.transpose(0, 1).is_contiguous()); // strides [1, 4]
    assert!(x.select(0, 1).is_contiguous()); // a row: stride 1
    assert!(!x.select(1, 2).is_contiguous()); // a column: stride 4
    assert!(x.slice(0, 1, 3, 1).is_contiguous()); // whole rows
    assert!(!x.slice(1, 0, 2, 1).is_contiguous()); // part of each row
    // Shape [4, 1], strides [1, 4]: the size-1 axis is never stepped along.
    assert!(x.transpose(0, 1).slice(1, 1, 2, 1).is_contiguous());
    // @end
}

#[test]
fn reshape_example() {
    // @listing reshape-example
    let x = Tensor::arange(&[2, 3]);
    let v = x.reshape(&[3, 2]); // x is contiguous: a view
    assert!(v.shares_storage(&x));

    let t = x.transpose(0, 1); // shape [3, 2], strides [1, 3]
    let r = t.reshape(&[6]); // no strides can read t in this order: a copy
    assert!(!r.shares_storage(&x));
    assert_eq!(r.to_vec(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    // @end
}
