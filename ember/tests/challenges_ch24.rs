//! Tests for the challenges of Chapter 24.
use ember::{Batch, DataLoader, Rng, Tensor, TensorDataset, Window, matmul_blocked};

/// Every number of `t`, as bits, so that comparisons are exact.
fn bits(t: &Tensor) -> Vec<u32> {
    t.to_vec().iter().map(|v| v.to_bits()).collect()
}

// @challenge 1
#[test]
fn predict_step_flops_and_amdahl() {
    // @listing predict-test
    let flops = |m: usize, k: usize, n: usize| 2 * m * k * n;
    let forward = flops(64, 784, 128) + flops(64, 128, 10);
    let backward = flops(784, 64, 128) + flops(128, 64, 10) + flops(64, 10, 128);
    assert_eq!((forward, backward), (13_008_896, 13_172_736));
    let ms = (forward + backward) as f64 / 64e9 * 1e3; // at 64 GFLOP/s
    assert_eq!((ms * 1000.0).round() / 1000.0, 0.409);
    let p = 545.2 / 670.2; // the products' share after Section 24.4
    let amdahl = |s: f64| 1.0 / ((1.0 - p) + p / s);
    let r = |x: f64| (x * 100.0).round() / 100.0;
    assert_eq!((r(p), r(amdahl(6.0))), (0.81, 3.10));
    assert_eq!(r(amdahl(f64::INFINITY)), 5.36);
    // @end
}

// @challenge 2
use ember::par_map;

#[test]
fn par_map_matches_map() {
    // @listing par-map-test
    let mut rng = Rng::new(24);
    let x = Tensor::randn(&[300, 1000], &mut rng); // 300,000 numbers
    let one = x.map(f32::exp);
    for threads in [1, 2, 3, 4, 7, 18] {
        assert_eq!(bits(&par_map(&x, f32::exp, threads)), bits(&one));
    }
    let small = Tensor::arange(&[1000]); // under the threshold: no threads
    assert_eq!(par_map(&small, |v| v * 2.0, 8).to_vec()[999], 1998.0);
    let view = x.transpose(0, 1); // a view is copied first
    let expected = bits(&view.map(f32::exp));
    assert_eq!(bits(&par_map(&view, f32::exp, 4)), expected);
    // @end
}

// @challenge 3
#[test]
fn transposed_product_without_a_copy() {
    // @listing tn-test
    let mut rng = Rng::new(24);
    let a = Tensor::randn(&[301, 9], &mut rng); // like im2col's columns
    let g = Tensor::randn(&[301, 8], &mut rng); // and the product's gradient
    let copied = a.transpose(0, 1).matmul(&g); // Chapter 10's rule: [9, 8]
    assert_eq!(bits(&a.matmul_tn(&g)), bits(&copied));
    let x = Tensor::arange(&[2, 3]); // xᵀ x by hand: [[9, 12, 15], ...]
    assert_eq!(x.matmul_tn(&x).to_vec()[..3], [9.0, 12.0, 15.0]);
    // @end
}

// @challenge 4
use ember::im2col_parallel;

#[test]
fn parallel_im2col_is_im2col() {
    // @listing im2col-test
    let mut rng = Rng::new(24);
    let x = Tensor::randn(&[5, 3, 6, 7], &mut rng);
    let (same, pool) = (Window::new(3, 1, 1), Window::new(2, 2, 0));
    for win in [same, pool, Window::new(3, 2, 2)] {
        let one = x.im2col(win);
        for threads in [1, 2, 3, 5, 8] {
            assert_eq!(bits(&im2col_parallel(&x, win, threads)), bits(&one));
        }
    }
    // @end
}

#[test]
fn parallel_im2col_of_no_images() {
    let x = Tensor::zeros(&[0, 2, 4, 4]);
    let win = Window::new(3, 1, 1);
    let m = im2col_parallel(&x, win, 4);
    assert_eq!(m.shape(), x.im2col(win).shape());
}

// @challenge 5
use ember::matmul_stacks;

#[test]
fn stacks_on_threads() {
    // @listing stacks-test
    let (batch, m, k, n) = (7, 5, 6, 9);
    let a: Vec<f32> = (0..batch * m * k).map(|i| (i % 13) as f32 / 7.0).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|i| (i % 11) as f32 / 3.0).collect();
    let mut one = vec![0.0; batch * m * n];
    let pairs = a.chunks(m * k).zip(b.chunks(k * n));
    for ((a, b), c) in pairs.zip(one.chunks_mut(m * n)) {
        matmul_blocked(a, b, c, (m, k, n)); // one matrix at a time
    }
    for threads in 1..=8 {
        let mut c = vec![0.0; batch * m * n];
        matmul_stacks(&a, &b, &mut c, batch, (m, k, n), threads);
        assert!(c.iter().zip(&one).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
    // @end
}

#[test]
fn an_empty_stack() {
    let mut c: Vec<f32> = vec![];
    matmul_stacks(&[], &[], &mut c, 0, (2, 3, 4), 4); // used to panic in chunks(0)
}

// @challenge 6
use ember::prefetched_epoch;

#[test]
fn prefetched_batches_match_the_loader() {
    // @listing prefetch-test
    let mut rng = Rng::new(24);
    let x = Tensor::randn(&[50, 6], &mut rng);
    let ds = TensorDataset::new(x, (0..50).map(|i| i % 10).collect());
    let mut loader = DataLoader::new(&ds, 8).shuffle(Rng::new(7)).drop_last(true);
    let served: Vec<Batch> = loader.epoch().collect(); // 6 batches of 8
    let mut fetched: Vec<Batch> = vec![];
    prefetched_epoch(&ds, 8, &mut Rng::new(7), |batch| fetched.push(batch));
    assert_eq!(fetched.len(), 6);
    for ((x1, y1), (x2, y2)) in served.iter().zip(&fetched) {
        assert_eq!((bits(x1), y1, x1.shape()), (bits(x2), y2, x2.shape()));
    }
    // @end
}

#[test]
fn prefetching_no_examples() {
    let ds = TensorDataset::new(Tensor::zeros(&[0, 6]), vec![]);
    let mut batches = 0;
    prefetched_epoch(&ds, 8, &mut Rng::new(7), |_| batches += 1);
    assert_eq!(batches, 0);
}

// @challenge 7
use ember::Pool;

#[test]
fn pool_reuses_buffers() {
    // @listing pool-test
    let mut pool = Pool::default();
    let mut a = pool.take(1000);
    a[3] = 7.0;
    let at = a.as_ptr();
    pool.give(a);
    assert_eq!(pool.held(), 1);
    let b = pool.take(1000); // the same buffer, zeroed again
    assert_eq!((b.as_ptr(), b.iter().all(|&v| v == 0.0)), (at, true));
    let c = pool.take(500); // no buffer of that length: a new one
    assert_eq!((c.len(), pool.held()), (500, 0));
    // @end
}
