//! Tests for the Chapter 1 challenges: `cargo test --test challenges_ch01`.
use ember::{Tensor, contiguous_strides, unravel_index};

#[test]
fn predict_strides_and_an_index() {
    let x = Tensor::arange(&[3, 2, 5]);
    assert_eq!(x.strides(), &[10, 5, 1]);
    assert_eq!(x.storage_index(&[2, 1, 4]), 29);
    assert_eq!(x.storage_index(&[1, 0, 3]), 13);
    assert_eq!(contiguous_strides(&[4, 1, 3]), vec![3, 3, 1]);
}

#[test]
fn full_and_ones() {
    let x = Tensor::full(&[2, 3], 7.5);
    assert_eq!(x.shape(), &[2, 3]);
    assert_eq!(x.get(&[1, 2]), 7.5);
    assert_eq!(Tensor::ones(&[4]).get(&[3]), 1.0);
}

#[test]
fn eye_has_ones_on_the_diagonal() {
    let i = Tensor::eye(3);
    assert_eq!(i.shape(), &[3, 3]);
    for r in 0..3 {
        for c in 0..3 {
            assert_eq!(
                i.get(&[r, c]),
                if r == c { 1.0 } else { 0.0 },
                "eye(3)[{r}, {c}]"
            );
        }
    }
}

#[test]
fn unravel_index_inverts_storage_index() {
    assert_eq!(unravel_index(23, &[2, 3, 4]), vec![1, 2, 3]);
    assert_eq!(unravel_index(29, &[3, 2, 5]), vec![2, 1, 4]);
    let x = Tensor::arange(&[2, 3, 4]);
    for i in 0..24 {
        assert_eq!(x.storage_index(&unravel_index(i, &[2, 3, 4])), i);
    }
}

#[test]
#[should_panic(expected = "out of range")]
fn unravel_index_rejects_a_position_past_the_end() {
    unravel_index(6, &[2, 3]);
}

#[test]
fn from_fn_calls_f_with_each_index() {
    let x = Tensor::from_fn(&[2, 3], |i| (10 * i[0] + i[1]) as f32);
    assert_eq!(x.get(&[0, 0]), 0.0);
    assert_eq!(x.get(&[1, 2]), 12.0);
    assert_eq!(x.get(&[0, 1]), 1.0);
}

#[test]
fn set_copies_on_write() {
    let x = Tensor::arange(&[2, 3]);
    let mut y = x.clone();
    y.set(&[0, 0], 100.0);
    assert_eq!(y.get(&[0, 0]), 100.0);
    assert_eq!(x.get(&[0, 0]), 0.0, "the clone's write must not reach x");
    assert!(!x.shares_storage(&y));

    let mut z = Tensor::zeros(&[2]);
    z.set(&[1], 3.0);
    assert_eq!(z.get(&[1]), 3.0);
}

#[test]
fn allclose_compares_shape_and_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
    let b = Tensor::from_vec(vec![1.0, 2.0005, 3.0], &[3]);
    assert!(a.allclose(&b, 1e-3));
    assert!(!a.allclose(&b, 1e-5));
    assert!(!a.allclose(&Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]), 1e-3));
}

#[test]
fn display_nests_brackets() {
    assert_eq!(
        format!("{}", Tensor::arange(&[2, 3])),
        "[[0, 1, 2],\n [3, 4, 5]]"
    );
    assert_eq!(
        format!("{}", Tensor::arange(&[2, 2, 2])),
        "[[[0, 1],\n  [2, 3]],\n [[4, 5],\n  [6, 7]]]"
    );
    assert_eq!(
        format!("{}", Tensor::from_vec(vec![1.5, -2.0], &[2])),
        "[1.5, -2]"
    );
    assert_eq!(format!("{}", Tensor::from_vec(vec![4.0], &[])), "4");
}
