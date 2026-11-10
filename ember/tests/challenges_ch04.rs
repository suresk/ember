//! Tests for the Chapter 4 challenges: `cargo test --test challenges_ch04`.
use ember::Tensor;

fn floats(xs: &[i32]) -> Vec<f32> {
    xs.iter().map(|&x| x as f32).collect()
}

#[test]
fn predict_reductions() {
    let x = Tensor::arange(&[2, 3, 4]);
    let s = x.sum(1, false);
    assert_eq!(s.shape(), &[2, 4]);
    assert_eq!(s.to_vec()[..4], floats(&[12, 15, 18, 21]));
    assert_eq!(x.sum(1, true).shape(), &[2, 1, 4]);
    assert_eq!(x.max(2, false).to_vec(), floats(&[3, 7, 11, 15, 19, 23]));
    assert_eq!(x.argmax(0), vec![1; 12]);
    let m = x.mean(2, true);
    assert_eq!((m.shape(), m.get(&[0, 0, 0])), (&[2, 3, 1][..], 1.5));
    assert_eq!(x.sum_all(), 276.0);
    let t = x.sum_to(&[3, 1]);
    assert_eq!(
        (t.shape(), t.to_vec()),
        (&[3, 1][..], floats(&[60, 92, 124]))
    );
    assert_eq!(x.transpose(1, 2).as_slice(), None);
    assert_eq!((&x - &m).get(&[1, 2, 3]), 1.5);
}

#[test]
fn min_and_argmin() {
    let x = Tensor::from_vec(vec![3.0, 1.0, 1.0, 0.0, 5.0, 0.0], &[2, 3]);
    assert_eq!(x.min(1, false).to_vec(), vec![1.0, 0.0]);
    assert_eq!(x.argmin(1), vec![1, 0]); // ties: the first position
    assert_eq!(x.min(0, true).shape(), &[1, 3]);
    assert_eq!(x.min(0, true).to_vec(), vec![0.0, 1.0, 0.0]);
    assert_eq!(x.argmin(0), vec![1, 0, 1]);
}

#[test]
fn var_with_bessels_correction() {
    let x = Tensor::from_vec(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &[8]);
    assert_eq!(x.var(0, false, 0).item(), 4.0);
    assert_eq!(x.var(0, false, 1).item(), 32.0 / 7.0);

    let m = Tensor::arange(&[2, 3]);
    let v = m.var(1, true, 1);
    assert_eq!((v.shape(), v.to_vec()), (&[2, 1][..], vec![1.0, 1.0]));

    // One pass, mean(x²) - mean(x)², cancels catastrophically.
    let y = Tensor::from_vec(vec![10_000.0, 10_001.0, 10_002.0], &[3]);
    let mean = y.mean(0, false);
    let one_pass = y.mul(&y).mean(0, false).sub(&mean.mul(&mean));
    assert_eq!(one_pass.item(), 0.0);
    assert_eq!(y.var(0, false, 0).item(), 2.0 / 3.0);
    assert_eq!(y.var(0, false, 1).item(), 1.0);
}

#[test]
fn where_selects_and_masks() {
    let x = Tensor::from_vec(vec![-2.0, 3.0, -1.0, 4.0], &[4]);
    let zero = Tensor::zeros(&[]);
    let mask = x.gt(&zero);
    assert_eq!(mask.to_vec(), floats(&[0, 1, 0, 1]));
    assert_eq!(Tensor::where_(&mask, &x, &zero).to_vec(), x.relu().to_vec());

    // Arithmetic masking breaks on infinity: 0 * inf is NaN.
    let cond = Tensor::from_vec(vec![1.0, 0.0], &[2]);
    let a = Tensor::from_vec(vec![1.0, f32::INFINITY], &[2]);
    let b = Tensor::from_vec(vec![5.0, 6.0], &[2]);
    assert_eq!(Tensor::where_(&cond, &a, &b).to_vec(), vec![1.0, 6.0]);
    let mixed = &(&cond * &a) + &(&cond.neg().add_scalar(1.0) * &b);
    assert_eq!(mixed.get(&[0]), 1.0);
    assert!(mixed.get(&[1]).is_nan());

    // All three operands broadcast.
    let rows = Tensor::from_vec(vec![1.0, 0.0], &[2, 1]);
    let minus = Tensor::from_vec(vec![-1.0], &[]);
    let w = Tensor::where_(&rows, &Tensor::arange(&[3]), &minus);
    assert_eq!(w.shape(), &[2, 3]);
    assert_eq!(w.to_vec(), floats(&[0, 1, 2, -1, -1, -1]));
}

#[test]
#[should_panic(expected = "where_: shapes [2], [3] and [] don't broadcast")]
fn where_rejects_incompatible_shapes() {
    let s = Tensor::zeros(&[]);
    Tensor::where_(&Tensor::arange(&[2]), &Tensor::arange(&[3]), &s);
}

#[test]
fn cumsum_runs_along_an_axis() {
    let x = Tensor::arange(&[2, 3]);
    let c1 = x.cumsum(1);
    assert_eq!(c1.to_vec(), floats(&[0, 1, 3, 3, 7, 12]));
    let c0 = x.cumsum(0);
    assert_eq!(
        (c0.shape(), c0.to_vec()),
        (&[2, 3][..], floats(&[0, 1, 2, 3, 5, 7]))
    );
    // The last element of each lane is the lane's sum.
    let y = Tensor::arange(&[2, 3, 4]);
    let c = y.cumsum(1);
    assert_eq!(c.select(1, 2).to_vec(), y.sum(1, false).to_vec());
    assert_eq!(c.get(&[1, 1, 2]), 14.0 + 18.0);
}

#[test]
fn softplus_without_overflow() {
    let naive = |x: f32| (1.0 + x.exp()).ln();
    assert_eq!(naive(89.0), f32::INFINITY); // e^89 overflows f32
    let x = Tensor::from_vec(vec![-100.0, 0.0, 89.0, 100.0], &[4]);
    let s = x.softplus().to_vec();
    assert_eq!(s[1], 2f32.ln());
    assert_eq!((s[2], s[3]), (89.0, 100.0));
    assert!(s[0] > 0.0 && s[0] < 1e-43); // e^-100 ≈ 3.7e-44, a subnormal f32
    // It agrees with the formula where the formula is safe.
    let one = Tensor::from_vec(vec![1.0], &[1]).softplus().item();
    assert!((one - naive(1.0)).abs() < 1e-6); // both about 1.3132616
}

#[test]
fn sum_strided_matches_sum() {
    let x = Tensor::arange(&[2, 3, 4]);
    for dim in 0..3 {
        for keepdim in [false, true] {
            let (a, b) = (x.sum_strided(dim, keepdim), x.sum(dim, keepdim));
            assert_eq!(a.shape(), b.shape());
            assert_eq!(a.to_vec(), b.to_vec());
        }
    }
    let t = Tensor::arange(&[4, 6]).transpose(0, 1).slice(0, 1, 6, 2); // a view
    assert_eq!(t.sum_strided(1, false).to_vec(), t.sum(1, false).to_vec());
    assert_eq!(t.sum_strided(0, true).to_vec(), floats(&[9, 27, 45, 63]));
    let e = Tensor::arange(&[3]).expand(&[2, 3]); // stride 0 in the input
    assert_eq!(e.sum_strided(0, false).to_vec(), floats(&[0, 2, 4]));
}
