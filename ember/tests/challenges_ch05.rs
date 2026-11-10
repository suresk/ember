//! Tests for the Chapter 5 challenges: `cargo test --test challenges_ch05`.
use ember::{Tensor, matmul_blocked_with, matmul_naive, matmul_nt_kernel};

fn floats(xs: &[i32]) -> Vec<f32> {
    xs.iter().map(|&x| x as f32).collect()
}

fn fractions(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 37 + seed * 11) % 101) as f32 / 50.5 - 1.0)
        .collect()
}

#[test]
fn predict_matmul() {
    let x = Tensor::arange(&[3, 3]);
    let xx = x.matmul(&x);
    assert_eq!(xx.to_vec()[..3], floats(&[15, 18, 21]));
    assert_eq!(xx.to_vec()[6..], floats(&[69, 90, 111]));
    assert_eq!(x.transpose(0, 1).matmul(&x).get(&[2, 2]), 93.0);
    let y = Tensor::arange(&[4, 2, 3]).matmul(&Tensor::arange(&[4, 3, 5]));
    assert_eq!(y.shape(), &[4, 2, 5]);
    let flops = |b: usize, m: usize, k: usize, n: usize| 2 * b * m * k * n;
    assert_eq!(flops(4, 2, 3, 5), 240);
    assert_eq!(flops(1, 32, 100, 10), 64_000);
    // A 1024 × 1024 product at 60 GFLOP/s, in milliseconds.
    let ms = flops(1, 1024, 1024, 1024) as f64 / 60e9 * 1e3;
    assert_eq!(flops(1, 1024, 1024, 1024), 2_147_483_648);
    assert!((ms - 35.8).abs() < 0.05, "{ms}");
}

#[test]
fn dot_and_trace() {
    let u = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
    let v = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]);
    assert_eq!(u.dot(&v), 32.0);
    let (a, b) = (Tensor::arange(&[2, 3]), Tensor::arange(&[3, 2]));
    assert_eq!(a.matmul(&b).trace(), 50.0); // 10 + 40
    assert_eq!(
        b.matmul(&a).to_vec(),
        floats(&[3, 4, 5, 9, 14, 19, 15, 24, 33])
    );
    assert_eq!(b.matmul(&a).trace(), 50.0); // 3 + 14 + 33
    // Entry (i, j) of a @ b is the dot of row i of a with column j of b.
    assert_eq!(a.select(0, 1).dot(&b.select(1, 0)), 28.0);
}

#[test]
#[should_panic(expected = "dot: shapes [3] and [2] are not one vector length")]
fn dot_wants_equal_lengths() {
    Tensor::arange(&[3]).dot(&Tensor::arange(&[2]));
}

#[test]
fn matvec_is_a_product_with_a_column() {
    let a = Tensor::arange(&[2, 3]);
    let ones = Tensor::from_vec(vec![1.0; 3], &[3]);
    assert_eq!(a.matvec(&ones).to_vec(), floats(&[3, 12]));
    let d = Tensor::from_vec(vec![1.0, 0.0, -1.0], &[3]);
    assert_eq!(a.matvec(&d).to_vec(), floats(&[-2, -2]));
    // Bit for bit what matmul gives with v as a [k, 1] column.
    let (m, k) = (5, 37);
    let a = Tensor::from_vec(fractions(m * k, 1), &[m, k]);
    let v = Tensor::from_vec(fractions(k, 2), &[k]);
    let col = a.matmul(&v.reshape(&[k, 1]));
    assert_eq!(a.matvec(&v).to_vec(), col.to_vec());
    assert_eq!(a.matvec(&v).shape(), &[5]);
    let empty = Tensor::zeros(&[2, 0]).matvec(&Tensor::zeros(&[0]));
    assert_eq!(empty.to_vec(), vec![0.0, 0.0]);
}

#[test]
fn matmul_nt_needs_no_transpose() {
    let a = Tensor::arange(&[2, 3]);
    assert_eq!(a.matmul_nt(&a).to_vec(), floats(&[5, 14, 14, 50]));
    let (m, k, n) = (6, 41, 9);
    let x = Tensor::from_vec(fractions(m * k, 1), &[m, k]);
    let y = Tensor::from_vec(fractions(n * k, 2), &[n, k]);
    let want = x.matmul(&y.transpose(0, 1));
    assert_eq!(x.matmul_nt(&y).to_vec(), want.to_vec()); // same sums, same order
    assert_eq!(x.matmul_nt(&y).shape(), &[6, 9]);
    let mut c = vec![0.0; m * n];
    matmul_nt_kernel(
        x.as_slice().unwrap(),
        y.as_slice().unwrap(),
        &mut c,
        (m, k, n),
    );
    assert_eq!(c, want.to_vec());
}

#[test]
fn matmul_broadcasts_batch_axes() {
    // Shared weights: [4, 2, 3] @ [3, 2], the shape of a batch through a layer.
    let x = Tensor::arange(&[4, 2, 3]);
    let w = Tensor::arange(&[3, 2]);
    let y = x.matmul_broadcast(&w);
    assert_eq!(y.shape(), &[4, 2, 2]);
    assert_eq!(y.to_vec()[..4], floats(&[10, 13, 28, 40]));
    for t in 0..4 {
        let want = x.select(0, t).matmul(&w).to_vec();
        assert_eq!(y.select(0, t).to_vec(), want);
    }
    // Batch axes [2, 1] and [3] broadcast to [2, 3].
    let a = Tensor::arange(&[2, 1, 2, 3]);
    let b = Tensor::arange(&[3, 3, 2]);
    let c = a.matmul_broadcast(&b);
    assert_eq!(c.shape(), &[2, 3, 2, 2]);
    let one = a.select(0, 1).select(0, 0).matmul(&b.select(0, 2));
    assert_eq!(c.select(0, 1).select(0, 2).to_vec(), one.to_vec());
    // Two plain matrices: the same as matmul.
    let (p, q) = (Tensor::arange(&[2, 3]), Tensor::arange(&[3, 2]));
    assert_eq!(p.matmul_broadcast(&q).to_vec(), p.matmul(&q).to_vec());
}

#[test]
#[should_panic(
    expected = "matmul: [2, 2, 3] @ [3, 3, 2]: batch axes don't broadcast"
)]
fn matmul_broadcast_rejects_batches() {
    Tensor::arange(&[2, 2, 3]).matmul_broadcast(&Tensor::arange(&[3, 3, 2]));
}

#[test]
fn blocked_with_any_panel() {
    let (m, k, n) = (9, 70, 17);
    let (a, b) = (fractions(m * k, 1), fractions(k * n, 2));
    let mut want = vec![0.0; m * n];
    matmul_naive(&a, &b, &mut want, (m, k, n));
    for panel in [1, 2, 16, 69, 70, 71, 4096] {
        let mut c = vec![0.0; m * n];
        matmul_blocked_with(&a, &b, &mut c, (m, k, n), panel);
        assert_eq!(c, want, "panel {panel}");
    }
}

#[test]
fn matmul_through_strides() {
    let x = Tensor::arange(&[4, 5]);
    let a = x.slice(1, 1, 4, 1); // [4, 3], strides [5, 1], offset 1
    let b = x.transpose(0, 1).slice(0, 0, 3, 1); // [3, 4], strides [1, 5]
    let want = a.contiguous().matmul(&b.contiguous());
    assert_eq!(a.matmul_strided(&b).to_vec(), want.to_vec());
    let y = Tensor::from_vec(fractions(6 * 8, 3), &[6, 8]);
    let yt = y.transpose(0, 1); // [8, 6]
    assert_eq!(yt.matmul_strided(&y).to_vec(), yt.matmul(&y).to_vec());
    let e = Tensor::arange(&[3]).expand(&[2, 3]); // stride 0 works too
    assert_eq!(e.matmul_strided(&yt.slice(0, 0, 3, 1)).shape(), &[2, 6]);
}
