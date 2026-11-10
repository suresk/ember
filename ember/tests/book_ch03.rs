//! The examples printed in Chapter 3 of the guided edition.
use ember::{Tensor, broadcast_shapes};

// @listing helpers
/// Integers as `f32`s, to compare with `to_vec`.
fn floats(xs: &[usize]) -> Vec<f32> {
    xs.iter().map(|&x| x as f32).collect()
}

/// Two tensors of the same shape added element by element, in row-major order.
fn add_in_order(a: &Tensor, b: &Tensor) -> Vec<f32> {
    assert_eq!(a.shape(), b.shape());
    let (a, b) = (a.to_vec(), b.to_vec());
    a.iter().zip(&b).map(|(p, q)| p + q).collect()
}
// @end

#[test]
fn broadcast_rule_examples() {
    // @listing rule-examples
    // [3, 1] and [4]: pad [4] to [1, 4], then stretch each 1.
    assert_eq!(broadcast_shapes(&[3, 1], &[4]), Some(vec![3, 4]));
    // A bias of shape [3] against a batch of shape [2, 3].
    assert_eq!(broadcast_shapes(&[2, 3], &[3]), Some(vec![2, 3]));
    assert_eq!(
        broadcast_shapes(&[8, 1, 6, 1], &[7, 1, 5]),
        Some(vec![8, 7, 6, 5])
    );
    assert_eq!(broadcast_shapes(&[], &[2, 3]), Some(vec![2, 3])); // a scalar
    assert_eq!(broadcast_shapes(&[5, 1], &[1, 0]), Some(vec![5, 0])); // 1 stretches to 0
    // @end
}

#[test]
fn shapes_that_do_not_broadcast() {
    // @listing no-broadcast
    // One value per row of a [2, 3] matrix: a [2] lines up with the columns.
    assert_eq!(broadcast_shapes(&[2, 3], &[2]), None);
    assert_eq!(broadcast_shapes(&[2, 3], &[2, 1]), Some(vec![2, 3]));
    // Same number of elements, still incompatible: 3 against 2, 2 against 3.
    assert_eq!(broadcast_shapes(&[2, 3], &[3, 2]), None);
    // @end
}

#[test]
fn expand_a_bias() {
    // @listing expand-bias
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]);
    let e = b.expand(&[2, 3]);
    assert_eq!(e.shape(), &[2, 3]);
    assert_eq!(e.strides(), &[0, 1]);
    assert!(e.shares_storage(&b));
    assert_eq!(e.storage_index(&[0, 2]), 2);
    assert_eq!(e.storage_index(&[1, 2]), 2); // row 1 reads the same slot
    assert_eq!(e.to_vec(), floats(&[10, 20, 30, 10, 20, 30]));
    // @end

    // @listing bias-sum
    let x = Tensor::arange(&[2, 3]);
    assert_eq!(add_in_order(&x, &e), floats(&[10, 21, 32, 13, 24, 35]));
    // @end
}

#[test]
fn expand_a_column_and_a_scalar() {
    // @listing expand-column
    let c = Tensor::arange(&[3, 1]); // a column: shape [3, 1], strides [1, 1]
    let e = c.expand(&[3, 4]);
    assert_eq!(e.strides(), &[1, 0]);
    assert_eq!(e.to_vec(), floats(&[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]));

    let s = Tensor::from_vec(vec![7.0], &[]); // a scalar: no axes at all
    let e = s.expand(&[2, 3]);
    assert_eq!(e.strides(), &[0, 0]);
    assert_eq!(e.to_vec(), vec![7.0; 6]);
    // @end
}

#[test]
fn expand_keeps_offset_and_strides() {
    // @listing expand-view
    let x = Tensor::arange(&[3, 4]);
    let col = x.slice(1, 2, 3, 1); // x[:, 2:3]: shape [3, 1], strides [4, 1]
    let e = col.expand(&[2, 3, 5]);
    assert_eq!((e.strides(), e.offset()), (&[0, 4, 0][..], 2));
    assert_eq!(e.to_vec()[..6], floats(&[2, 2, 2, 2, 2, 6]));
    // @end
}

// @listing expand-panics
#[test]
#[should_panic(expected = "expand: [2] to [2, 3]: axis 0 has size 2, not 1 or 3")]
fn expand_rejects_a_mismatched_axis() {
    Tensor::arange(&[2]).expand(&[2, 3]);
}

#[test]
#[should_panic(expected = "expand: [2, 3] to [3] would drop axes")]
fn expand_cannot_remove_axes() {
    Tensor::arange(&[2, 3]).expand(&[3]);
}
// @end

#[test]
fn the_cost_of_a_copy() {
    // @listing bias-bytes
    let bias = Tensor::zeros(&[4096]); // one value per feature
    let e = bias.expand(&[512, 4096]); // one row per example in the batch
    assert_eq!(e.numel(), 2_097_152);
    assert!(e.shares_storage(&bias));
    let f32_bytes = std::mem::size_of::<f32>();
    assert_eq!(bias.numel() * f32_bytes, 16_384); // what the view reads: 16 KiB
    assert_eq!(e.numel() * f32_bytes, 8_388_608); // what a copy holds: 8 MiB

    let m = e.contiguous(); // materialize: 2,097,152 writes
    assert!(!m.shares_storage(&bias));
    assert_eq!(m.strides(), &[4096, 1]);
    // @end
}

#[test]
fn expanded_tensors_are_not_contiguous() {
    // @listing expand-contiguity
    let b = Tensor::arange(&[3]);
    assert!(!b.expand(&[2, 3]).is_contiguous()); // strides [0, 1]
    assert!(b.expand(&[1, 3]).is_contiguous()); // stride 0, but on a size-1 axis

    let m = b.expand(&[2, 3]).contiguous();
    assert_eq!(m.strides(), &[3, 1]);
    assert_eq!(m.to_vec(), floats(&[0, 1, 2, 0, 1, 2]));
    // @end
}

#[test]
fn broadcasting_both_operands() {
    // @listing both-sides
    let x = Tensor::from_vec(vec![0.0, 10.0, 20.0], &[3, 1]);
    let y = Tensor::arange(&[4]);
    let shape = broadcast_shapes(x.shape(), y.shape()).unwrap();
    let (a, b) = (x.expand(&shape), y.expand(&shape));
    assert_eq!((a.strides(), b.strides()), (&[1, 0][..], &[0, 1][..]));

    let want = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23];
    assert_eq!(add_in_order(&a, &b), floats(&want));
    // @end
}

#[test]
fn every_slot_is_read_once_per_row() {
    // @listing reads-per-slot
    let b = Tensor::arange(&[3]);
    let e = b.expand(&[2, 3]).to_vec(); // each value is its slot
    for slot in 0..3 {
        let reads = e.iter().filter(|&&v| v == slot as f32).count();
        assert_eq!(reads, 2); // one read per row of the batch
    }
    // @end
}
