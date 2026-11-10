//! Tests for the Chapter 3 challenges: `cargo test --test challenges_ch03`.
use ember::{Tensor, broadcast_all, broadcast_axes, broadcast_shapes, zip_slots};

fn floats(xs: &[usize]) -> Vec<f32> {
    xs.iter().map(|&x| x as f32).collect()
}

#[test]
fn predict_broadcasts() {
    assert_eq!(broadcast_shapes(&[5, 1, 4], &[3, 1]), Some(vec![5, 3, 4]));
    assert_eq!(broadcast_shapes(&[4, 3], &[4]), None);
    assert_eq!(broadcast_shapes(&[1], &[6, 1]), Some(vec![6, 1]));

    let a = Tensor::arange(&[3, 1]).expand(&[2, 3, 4]);
    assert_eq!(a.strides(), &[0, 1, 0]);
    assert_eq!(a.numel(), 24);

    let t = Tensor::arange(&[4, 6]).transpose(0, 1).slice(1, 2, 3, 1);
    assert_eq!(
        (t.shape(), t.strides(), t.offset()),
        (&[6, 1][..], &[1, 6][..], 12)
    );
    let e = t.expand(&[6, 5]);
    assert_eq!((e.strides(), e.offset()), (&[1, 0][..], 12));
    assert_eq!(e.to_vec()[..7], floats(&[12, 12, 12, 12, 12, 13, 13]));
    assert!(!e.is_contiguous());
}

#[test]
fn broadcast_all_folds_the_rule() {
    let shapes: [&[usize]; 3] = [&[3, 1], &[4], &[2, 1, 1]];
    assert_eq!(broadcast_all(&shapes), Some(vec![2, 3, 4]));
    assert_eq!(broadcast_all(&[&[2, 3]]), Some(vec![2, 3]));
    assert_eq!(broadcast_all(&[]), Some(vec![]));
    assert_eq!(broadcast_all(&[&[3], &[1], &[2, 1], &[4, 1, 2]]), None);
}

#[test]
fn outer_product_by_expand() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
    let b = Tensor::from_vec(vec![10.0, 20.0], &[2]);
    let o = a.outer(&b);
    assert_eq!(o.shape(), &[3, 2]);
    assert_eq!(o.to_vec(), floats(&[10, 20, 20, 40, 30, 60]));
    assert_eq!(b.outer(&a).to_vec(), floats(&[10, 20, 30, 20, 40, 60]));
}

#[test]
fn repeat_copies_tiles() {
    let x = Tensor::arange(&[2, 3]);
    let r = x.repeat(&[2, 2]);
    assert_eq!(r.shape(), &[4, 6]);
    assert_eq!(
        r.to_vec(),
        floats(&[
            0, 1, 2, 0, 1, 2, //
            3, 4, 5, 3, 4, 5, //
            0, 1, 2, 0, 1, 2, //
            3, 4, 5, 3, 4, 5,
        ])
    );
    assert!(!r.shares_storage(&x) && r.is_contiguous());

    let b = Tensor::arange(&[3]);
    let r = b.repeat(&[2, 1]);
    assert_eq!(r.shape(), &[2, 3]);
    assert_eq!(r.to_vec(), b.expand(&[2, 3]).to_vec());
    assert_eq!(r.strides(), &[3, 1]);
    assert_eq!(
        x.transpose(0, 1).repeat(&[1, 2]).to_vec(),
        floats(&[0, 3, 0, 3, 1, 4, 1, 4, 2, 5, 2, 5])
    );
}

#[test]
fn writing_into_a_broadcast_view() {
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]);
    let e = b.expand(&[2, 3]);
    assert!(e.is_broadcast());
    assert!(!b.expand(&[1, 3]).is_broadcast());
    assert!(!Tensor::arange(&[3, 4]).transpose(0, 1).is_broadcast());

    let mut w = e.clone();
    w.set(&[0, 1], 100.0); // one slot, read by both rows
    assert_eq!(w.to_vec(), floats(&[10, 100, 30, 10, 100, 30]));
    assert_eq!(b.to_vec(), floats(&[10, 20, 30])); // set copied the shared buffer

    let mut w = e.clone();
    w.make_writable();
    assert_eq!(w.strides(), &[3, 1]);
    w.set(&[0, 1], 100.0);
    assert_eq!(w.to_vec(), floats(&[10, 100, 30, 10, 20, 30]));

    let mut t = Tensor::arange(&[3, 4]).transpose(0, 1);
    let before = t.clone();
    t.make_writable(); // not broadcast: no copy
    assert!(t.shares_storage(&before));
}

#[test]
fn broadcast_axes_lists_the_stride_zero_axes() {
    assert_eq!(broadcast_axes(&[3], &[2, 3]), vec![0]);
    assert_eq!(broadcast_axes(&[3, 1], &[2, 3, 4]), vec![0, 2]);
    assert_eq!(broadcast_axes(&[2, 3], &[2, 3]), Vec::<usize>::new());
    assert_eq!(broadcast_axes(&[1, 1], &[5, 1]), vec![0]);
    assert_eq!(broadcast_axes(&[3], &[1, 3]), vec![0]);
    let e = Tensor::arange(&[3, 1]).expand(&[2, 3, 4]);
    for d in broadcast_axes(&[3, 1], &[2, 3, 4]) {
        assert_eq!(e.strides()[d], 0);
    }
}

#[test]
#[should_panic(expected = "broadcast_axes: [2] does not expand to [2, 3]")]
fn broadcast_axes_rejects_a_shape_that_does_not_expand() {
    broadcast_axes(&[2], &[2, 3]);
}

#[test]
fn zip_slots_walks_two_tensors_at_once() {
    let x = Tensor::arange(&[3, 1]);
    let y = Tensor::arange(&[4]);
    let pairs = zip_slots(&x, &y);
    assert_eq!(pairs.len(), 12);
    assert_eq!(&pairs[..5], &[(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]);
    assert_eq!(pairs[11], (2, 3));

    // Views work too: their offsets and strides carry through.
    let t = Tensor::arange(&[2, 3]).transpose(0, 1); // shape [3, 2], strides [1, 3]
    let col = Tensor::arange(&[4, 2]).slice(0, 1, 4, 1).slice(1, 1, 2, 1); // 3, 5, 7
    let pairs = zip_slots(&t, &col);
    assert_eq!(pairs, vec![(0, 3), (3, 3), (1, 5), (4, 5), (2, 7), (5, 7)]);

    // A scalar pairs its one slot with every slot of the other.
    let s = Tensor::from_vec(vec![7.0], &[]);
    assert_eq!(zip_slots(&s, &x), vec![(0, 0), (0, 1), (0, 2)]);
}

#[test]
#[should_panic(expected = "zip_slots: [2, 3] and [2] don't broadcast")]
fn zip_slots_rejects_incompatible_shapes() {
    zip_slots(&Tensor::arange(&[2, 3]), &Tensor::arange(&[2]));
}
