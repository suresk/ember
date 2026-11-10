//! The examples printed in Chapter 24 of the guided edition: a profile of one
//! step, Amdahl's law, the gradient nobody asked for, matrix products on
//! several threads (and the same numbers from any number of them), and an
//! `is_contiguous` that builds nothing.
use ember::profile::{mark, profile};
use ember::{
    AdamW, MIN_FLOPS_PER_THREAD, Rng, Tape, Tensor, contiguous_strides,
    matmul_blocked, matmul_parallel, set_threads, threads, threads_for,
};

#[path = "../examples/speed.rs"]
#[allow(dead_code)]
mod speed;

use speed::marked_step;
use speed::mnist_cnn::mnist_train::mlp;

#[test]
fn profiling_a_step() {
    // @listing profile-test
    let mut rng = Rng::new(24);
    let mut model = mlp(&mut rng); // Chapter 19's 784–128–10 network
    let x = Tensor::rand(&[4, 784], &mut rng);
    let batch = (x, vec![3, 1, 4, 1]);
    let mut opt = AdamW::new(1e-3);
    let ((), p) = profile(|| marked_step(&mut model, &batch, &mut opt));
    let forward = ["const", "param", "reshape", "matmul", "add", "relu"];
    let forward = forward.map(|op| p.calls("forward", op));
    assert_eq!(forward, [1, 4, 4, 2, 2, 1]); // 14 nodes, then the loss:
    assert_eq!(p.calls("forward", "cross_entropy"), 1);
    let backward = ["cross_entropy", "reshape", "add", "matmul", "relu"];
    let backward = backward.map(|op| p.calls("backward", op));
    assert_eq!(backward, [1, 3, 2, 2, 1]); // 9 rules
    assert_eq!(p.calls("step", "optimizer"), 1);
    assert_eq!(p.total(None), p.rows.iter().map(|r| r.3).sum());
    // @end
    mark("step", "nothing"); // no profile running: does nothing
    assert_eq!(p.calls("backward", "param"), 0); // leaves have no rule to mark
}

/// The best speedup of a program that spends a fraction `p` of its time in a
/// part made `s` times faster.
fn amdahl(p: f64, s: f64) -> f64 {
    1.0 / ((1.0 - p) + p / s)
}

#[test]
fn amdahls_law() {
    let r = |x: f64| (x * 100.0).round() / 100.0;
    let mlp = 869.8 / 1151.6; // matmul's share, Section 24.2's table
    let cnn = 3803.0 / 10297.0;
    assert_eq!((r(mlp), r(cnn)), (0.76, 0.37));
    assert_eq!(r(amdahl(mlp, f64::INFINITY)), 4.09);
    assert_eq!(r(amdahl(mlp, 6.0)), 2.70);
    assert_eq!(r(amdahl(mlp, 2.0)), 1.61);
    assert_eq!(r(amdahl(cnn, f64::INFINITY)), 1.59);
    assert_eq!(r(amdahl(cnn, 6.0)), 1.44);
}

#[test]
fn a_constant_gets_no_product() {
    // @listing unneeded-test
    let (x, w) = (Tensor::arange(&[4, 3]), Tensor::arange(&[3, 2]));
    let tape = Tape::new();
    let (xc, wv) = (tape.constant(x.clone()), tape.leaf(w.clone()));
    let grads = xc.matmul(wv).sum_all().backward();
    assert!(grads.get(xc).is_none()); // nothing for the constant
    let both = Tape::new(); // the same product with both operands leaves
    let (xl, wl) = (both.leaf(x), both.leaf(w));
    let all = xl.matmul(wl).sum_all().backward();
    let bits = |t: &Tensor| t.to_vec().iter().map(|v| v.to_bits()).collect();
    let w1: Vec<u32> = bits(grads.get(wv).unwrap());
    assert_eq!(w1, bits(all.get(wl).unwrap())); // the weight's gradient, to the bit
    assert_eq!(all.get(xl).unwrap().to_vec()[..2], [1.0, 5.0]); // x̄ = 1·Wᵀ
    // @end
}

#[test]
fn threads_change_no_bits() {
    // @listing same-bits
    let (m, k, n) = (37, 300, 19); // edges, two panels, and odd bands
    let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 / 7.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 / 3.0).collect();
    let mut one = vec![0.0; m * n];
    matmul_blocked(&a, &b, &mut one, (m, k, n));
    for threads in 1..=10 {
        let mut c = vec![0.0; m * n];
        matmul_parallel(&a, &b, &mut c, (m, k, n), threads);
        let same = c.iter().zip(&one).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same, "{threads} threads");
    }
    // @end
}

#[test]
fn how_many_threads() {
    // @listing threads-for
    let flops = 2 * 64 * 784 * 128; // Chapter 19's first layer, forward
    assert_eq!(flops / MIN_FLOPS_PER_THREAD, 6); // 12,845,056 / 2,097,152
    set_threads(18); // as many as the author's machine has
    assert_eq!(threads_for((64, 784, 128)), 6);
    assert_eq!(threads_for((64, 128, 10)), 1); // 163,840 FLOPs: one thread
    assert_eq!(threads_for((1024, 1024, 1024)), 18); // all of them
    set_threads(4);
    assert_eq!((threads(), threads_for((64, 784, 128))), (4, 4));
    // @end
}

/// Chapter 2's test: compare with the row-major strides, size-1 axes aside.
fn by_definition(t: &Tensor) -> bool {
    let want = contiguous_strides(t.shape());
    let axes = t.shape().iter().zip(t.strides()).zip(&want);
    axes.into_iter().all(|((&n, &s), &w)| n == 1 || s == w)
}

#[test]
fn is_contiguous_as_defined() {
    let x = Tensor::arange(&[3, 4, 5]);
    let views = [
        x.clone(),
        x.transpose(0, 2),
        x.select(1, 2),
        x.slice(0, 1, 3, 1),
        x.slice(2, 0, 5, 2),
        x.slice(1, 1, 2, 1).transpose(0, 1), // a size-1 axis moved
        x.reshape(&[60]),
        Tensor::arange(&[5]).expand(&[3, 5]),
        Tensor::arange(&[5]).expand(&[1, 5]),
        Tensor::zeros(&[]),
        Tensor::zeros(&[0, 3]),
    ];
    for v in &views {
        assert_eq!(v.is_contiguous(), by_definition(v), "{:?}", v.strides());
    }
}
