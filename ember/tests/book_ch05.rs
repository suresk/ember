//! The examples printed in Chapter 5 of the guided edition. Timings are not tested
//! here (they belong to `cargo xtask bench`); everything deterministic is.
use ember::{
    Kernel, PANEL, TILE_M, TILE_N, Tensor, matmul_blocked, matmul_ikj, matmul_naive,
};

fn floats(xs: &[i32]) -> Vec<f32> {
    xs.iter().map(|&x| x as f32).collect()
}

/// `len` values in [-1, 1) that are not small integers, so that rounding happens.
fn fractions(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 37 + seed * 11) % 101) as f32 / 50.5 - 1.0)
        .collect()
}

#[test]
fn counting_flops() {
    // @listing flops
    let flops = |m: usize, k: usize, n: usize| 2 * m * k * n;
    assert_eq!(flops(2, 3, 2), 24);
    assert_eq!(flops(512, 512, 512), 268_435_456);
    assert_eq!(flops(64, 784, 128), 12_845_056); // a batch through a layer
    assert_eq!(64 * 128, 8192); // the bias adds on its output
    // Doubling n multiplies the work by 8.
    assert_eq!(flops(1024, 1024, 1024), 8 * flops(512, 512, 512));
    // @end
}

#[test]
fn a_kernel_on_slices() {
    // @listing kernel-example
    let a = floats(&[0, 1, 2, 3, 4, 5]); // 2 × 3
    let b = floats(&[0, 1, 2, 3, 4, 5]); // 3 × 2
    let mut c = vec![0.0; 4];
    matmul_naive(&a, &b, &mut c, (2, 3, 2));
    assert_eq!(c, floats(&[10, 13, 28, 40]));
    matmul_naive(&a, &b, &mut c, (2, 3, 2)); // kernels add into c
    assert_eq!(c, floats(&[20, 26, 56, 80]));
    // @end
}

// @listing kernel-panics
#[test]
#[should_panic(expected = "kernel: slices for (m, k, n) = (2, 2, 2)")]
fn kernels_check_their_slices() {
    let mut c = vec![0.0; 4];
    matmul_naive(&[0.0; 6], &[0.0; 4], &mut c, (2, 2, 2)); // a is 2 × 3
}
// @end

#[test]
fn matmul_on_tensors() {
    // @listing matmul-example
    let a = Tensor::arange(&[2, 3]); // [[0, 1, 2], [3, 4, 5]]
    let b = Tensor::arange(&[3, 2]); // [[0, 1], [2, 3], [4, 5]]
    let c = a.matmul(&b);
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.to_vec(), floats(&[10, 13, 28, 40]));
    assert_eq!(c.get(&[1, 0]), 3.0 * 0.0 + 4.0 * 2.0 + 5.0 * 4.0);

    let x = Tensor::arange(&[2, 2, 3]); // two 2 × 3 matrices
    let y = Tensor::arange(&[2, 3, 2]); // two 3 × 2 matrices
    let z = x.matmul(&y);
    assert_eq!(z.shape(), &[2, 2, 2]);
    assert_eq!(z.to_vec(), floats(&[10, 13, 28, 40, 172, 193, 244, 274]));

    let at = a.transpose(0, 1); // a view: strides [1, 3]
    assert_eq!(at.as_slice(), None);
    let g = a.matmul(&at); // at is copied before the kernel runs
    assert_eq!(g.to_vec(), floats(&[5, 14, 14, 50]));
    assert_eq!(g.to_vec(), a.matmul(&at.contiguous()).to_vec());
    // @end
}

// @listing matmul-panics
#[test]
#[should_panic(expected = "matmul: [2, 3] @ [2, 3]: inner sizes differ")]
fn matmul_checks_the_inner_sizes() {
    Tensor::arange(&[2, 3]).matmul(&Tensor::arange(&[2, 3]));
}
// @end

#[test]
#[should_panic(
    expected = "matmul: [2, 2, 3] @ [3, 2]: need two 2-D or two 3-D tensors"
)]
fn matmul_wants_matching_ranks() {
    Tensor::arange(&[2, 2, 3]).matmul(&Tensor::arange(&[3, 2]));
}

#[test]
#[should_panic(expected = "matmul: [2, 2, 3] @ [3, 3, 2]: batch sizes differ")]
fn matmul_checks_the_batch() {
    Tensor::arange(&[2, 2, 3]).matmul(&Tensor::arange(&[3, 3, 2]));
}

#[test]
fn empty_products() {
    // k = 0: every entry is a sum of nothing.
    let c = Tensor::zeros(&[2, 0]).matmul(&Tensor::zeros(&[0, 3]));
    assert_eq!(c.to_vec(), vec![0.0; 6]);
    assert_eq!(
        Tensor::zeros(&[0, 4])
            .matmul(&Tensor::zeros(&[4, 2]))
            .shape(),
        &[0, 2]
    );
}

#[test]
fn the_inner_loops_read() {
    // @listing inner-loops
    let b = Tensor::arange(&[3, 4]);
    let column = b.select(1, 1); // what ijk's inner loop reads from b
    assert_eq!((column.strides(), column.offset()), (&[4][..], 1));
    assert_eq!(column.to_vec(), floats(&[1, 5, 9]));
    let row = b.select(0, 1); // what ikj's inner loop reads from b
    assert_eq!((row.strides(), row.offset()), (&[1][..], 4));
    assert!(row.as_slice().is_some());
    // @end
}

#[test]
fn ikj_one_row_at_a_time() {
    // @listing ikj-trace
    // After steps 0..p, row i of c holds a[i, :p] @ b[:p, :].
    let (a, b) = (Tensor::arange(&[2, 3]), Tensor::arange(&[3, 2]));
    let after = |p| a.slice(1, 0, p, 1).matmul(&b.slice(0, 0, p, 1)).to_vec();
    assert_eq!(after(1), floats(&[0, 0, 0, 3]));
    assert_eq!(after(2), floats(&[2, 3, 8, 15]));
    let (sa, sb) = (a.as_slice().unwrap(), b.as_slice().unwrap());
    let mut c = vec![0.0; 4];
    matmul_ikj(sa, sb, &mut c, (2, 3, 2));
    assert_eq!(c, floats(&[10, 13, 28, 40]));
    // @end
}

#[test]
fn the_kernels_agree_exactly() {
    // @listing kernels-agree
    // Sizes that leave edges: 7 = 4 + 3 rows, 10 = 8 + 2 columns, and k
    // spans two panels.
    let (m, k, n) = (7, 300, 10);
    let (a, b) = (fractions(m * k, 1), fractions(k * n, 2));
    let run = |kernel: Kernel| {
        let mut c = vec![0.0; m * n];
        kernel(&a, &b, &mut c, (m, k, n));
        c
    };
    let want = run(matmul_naive);
    assert_eq!(run(matmul_ikj), want); // equal to the last bit
    assert_eq!(run(matmul_blocked), want);
    // Close to, but not exactly, the same products summed in f64.
    let exact = |ij: usize| -> f64 {
        let (i, j) = (ij / n, ij % n);
        let terms = (0..k).map(|p| a[i * k + p] as f64 * b[p * n + j] as f64);
        terms.sum()
    };
    let errors = (0..m * n).map(|ij| (want[ij] as f64 - exact(ij)).abs());
    let err = errors.fold(0.0, f64::max);
    assert!(err > 0.0 && err < 1e-5, "largest error {err}");
    // @end
}

#[test]
fn what_a_tile_saves() {
    // A 128-byte cache line holds 32 f32s; a row of 512 fills 16 lines.
    let per_line = 128 / size_of::<f32>();
    assert_eq!((per_line, 512 / per_line), (32, 16));
    // @listing tile-traffic
    // One step of p for a 4 × 8 tile: 32 multiply-adds, and memory operations.
    let fmas = TILE_M * TILE_N;
    let (ikj, tile) = (3 * fmas, TILE_M + TILE_N); // ikj: load b, load c, store c
    assert_eq!((fmas, ikj, tile, ikj / tile), (32, 96, 12, 8));
    // A panel of a 2048-wide b in bytes; one strip's cache lines with and
    // without panels.
    let mib = PANEL * 2048 * 4 / 1024 / 1024;
    let kib = (PANEL * 128 / 1024, 2048 * 128 / 1024);
    assert_eq!((mib, kib), (2, (32, 256)));
    // @end
}

#[test]
fn layouts_for_add() {
    // @listing add-layouts
    // The three layouts of `cargo xtask bench add`, at n = 4 instead of 1024.
    let n = 4;
    let gapped = Tensor::arange(&[n, n + 1]).slice(1, 0, n, 1);
    assert_eq!((gapped.strides(), gapped.as_slice()), (&[5, 1][..], None));
    let transposed = Tensor::arange(&[n, n]).transpose(0, 1);
    assert_eq!(transposed.strides(), &[1, 4]);
    assert_eq!(transposed.as_slice(), None);
    // @end
}
