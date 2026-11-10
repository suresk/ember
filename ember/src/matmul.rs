//! Chapter 5: matrix multiplication. The kernels work on plain slices holding
//! row-major matrices; `Tensor::matmul` checks shapes, makes its operands
//! contiguous, and hands their slices to a kernel, once per matrix in a batch.
//!
//! Chapter 24: the kernel runs on several threads, each on a band of rows of
//! the result. Threads share slices, never tensors: a `Tensor` holds an `Rc`,
//! which is neither `Send` nor `Sync`, so this doesn't compile:
//!
//! ```compile_fail
//! use ember::Tensor;
//!
//! let x = Tensor::arange(&[4]);
//! std::thread::scope(|s| {
//!     s.spawn(|| x.sum_all()); // `Rc<Vec<f32>>` cannot be shared between threads
//! });
//! ```

use crate::tensor::Tensor;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::thread;

// @listing kernel-type
/// The sizes of a matrix product, `(m, k, n)`: `a` is m × k, `b` is k × n, and
/// `c` is m × n.
pub type Dims = (usize, usize, usize);

/// A matrix-multiply kernel: `kernel(a, b, c, dims)` adds the product of `a` and
/// `b` into `c`. All three are row-major slices.
pub type Kernel = fn(&[f32], &[f32], &mut [f32], Dims);

/// Every kernel starts by checking that the slices fit the sizes.
pub(crate) fn check_lens(a: &[f32], b: &[f32], c: &[f32], (m, k, n): Dims) {
    let (have, want) = ([a.len(), b.len(), c.len()], [m * k, k * n, m * n]);
    assert_eq!(have, want, "kernel: slices for (m, k, n) = ({m}, {k}, {n})");
}
// @end

// @listing naive
/// The definition, loop for loop: for each `(i, j)`, the dot product of row `i`
/// of `a` with column `j` of `b`.
pub fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], (m, k, n): Dims) {
    check_lens(a, b, c, (m, k, n));
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += acc;
        }
    }
}
// @end

// @listing ikj
/// The same sums with the two inner loops swapped: row `i` of `c` collects
/// `a[i][p]` times row `p` of `b`, for each `p` in turn.
pub fn matmul_ikj(a: &[f32], b: &[f32], c: &mut [f32], (m, k, n): Dims) {
    check_lens(a, b, c, (m, k, n));
    if k == 0 || n == 0 {
        return; // nothing to add; and chunks_exact(0) would panic
    }
    for (a_row, c_row) in a.chunks_exact(k).zip(c.chunks_exact_mut(n)) {
        for (&a_ip, b_row) in a_row.iter().zip(b.chunks_exact(n)) {
            for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                *c_ij += a_ip * b_pj;
            }
        }
    }
}
// @end

// @listing blocked
/// The tile of `c` that `matmul_blocked` holds in local variables, 4 rows by 8
/// columns, and the rows of `b` it walks before moving to the next tile: a panel.
pub const TILE_M: usize = 4;
pub const TILE_N: usize = 8;
pub const PANEL: usize = 256;

/// The same sums again, 4 × 8 entries of `c` at a time, one panel of `PANEL`
/// rows of `b` at a time.
pub fn matmul_blocked(a: &[f32], b: &[f32], c: &mut [f32], (m, k, n): Dims) {
    check_lens(a, b, c, (m, k, n));
    let (m4, n8) = (m - m % TILE_M, n - n % TILE_N);
    for p0 in (0..k).step_by(PANEL) {
        let ps = p0..k.min(p0 + PANEL);
        for i in (0..m4).step_by(TILE_M) {
            for j in (0..n8).step_by(TILE_N) {
                tile(a, b, c, (i, j), ps.clone(), (k, n));
            }
        }
    }
    // What the tiles miss: the last m % 4 rows, and the last n % 8 columns.
    edge(a, b, c, m4..m, 0..n, (k, n));
    edge(a, b, c, 0..m4, n8..n, (k, n));
}
// @end

// @listing tile
/// Adds `a[i..i+4, ps] · b[ps, j..j+8]` into the tile of `c` at `(i, j)`. The
/// tile lives in `acc`, a local array, for the whole loop over `ps`.
#[inline]
pub(crate) fn tile(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    (i, j): (usize, usize),
    ps: Range<usize>,
    (k, n): (usize, usize),
) {
    let mut acc = [[0.0; TILE_N]; TILE_M];
    for (r, acc_r) in acc.iter_mut().enumerate() {
        acc_r.copy_from_slice(&c[(i + r) * n + j..][..TILE_N]);
    }
    for p in ps {
        let b_row = &b[p * n + j..][..TILE_N];
        for (r, acc_r) in acc.iter_mut().enumerate() {
            let a_rp = a[(i + r) * k + p];
            for (c_rj, &b_pj) in acc_r.iter_mut().zip(b_row) {
                *c_rj += a_rp * b_pj;
            }
        }
    }
    for (r, acc_r) in acc.iter().enumerate() {
        c[(i + r) * n + j..][..TILE_N].copy_from_slice(acc_r);
    }
}
// @end

/// The entries of `c` in `rows` × `cols` that no tile covers, one at a time.
pub(crate) fn edge(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    rows: Range<usize>,
    cols: Range<usize>,
    (k, n): (usize, usize),
) {
    for i in rows {
        for j in cols.clone() {
            for p in 0..k {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
}

// @listing dims
/// `(batch, m, k, n)` for `a @ b`: two matrices, `[m, k]` and `[k, n]`, or two
/// stacks of them, `[batch, m, k]` and `[batch, k, n]`.
fn matmul_dims(a: &[usize], b: &[usize]) -> (usize, usize, usize, usize) {
    let (batch, (m, k), (k2, n)) = match (a, b) {
        (&[m, k], &[k2, n]) => (1, (m, k), (k2, n)),
        (&[b1, m, k], &[b2, k2, n]) => {
            assert_eq!(b1, b2, "matmul: {a:?} @ {b:?}: batch sizes differ");
            (b1, (m, k), (k2, n))
        }
        _ => panic!("matmul: {a:?} @ {b:?}: need two 2-D or two 3-D tensors"),
    };
    assert_eq!(k, k2, "matmul: {a:?} @ {b:?}: inner sizes differ");
    (batch, m, k, n)
}
// @end

impl Tensor {
    /// The matrix product `self @ other`, for `[m, k] @ [k, n]` or, matrix by
    /// matrix, for `[batch, m, k] @ [batch, k, n]`. A new contiguous tensor.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // @solution ch05 matmul
        let (batch, m, k, n) = matmul_dims(&self.shape, &other.shape);
        // Kernels read slices, so strided operands are copied first.
        let (a, b) = (self.contiguous(), other.contiguous());
        let (a, b) = (a.as_slice().unwrap(), b.as_slice().unwrap());
        let mut c = vec![0.0; batch * m * n];
        let threads = threads_for((m, k, n)); // Chapter 24
        for t in 0..batch {
            let (a, b) = (&a[t * m * k..][..m * k], &b[t * k * n..][..k * n]);
            let c = &mut c[t * m * n..][..m * n];
            matmul_parallel(a, b, c, (m, k, n), threads);
        }
        let mut shape = self.shape.clone();
        *shape.last_mut().unwrap() = n;
        Tensor::from_vec(c, &shape)
        // @end
    }
}

// @listing threads
/// The most threads a matrix product may use; 0 until it is first asked for.
static THREADS: AtomicUsize = AtomicUsize::new(0);

/// Let matrix products use up to `n` threads; 1 keeps them on the caller's.
pub fn set_threads(n: usize) {
    assert!(n > 0, "set_threads: 0 threads");
    THREADS.store(n, Relaxed);
}

/// The limit: what `set_threads` set, or else the `EMBER_THREADS` environment
/// variable, or else the number of threads the machine can run at once.
pub fn threads() -> usize {
    match THREADS.load(Relaxed) {
        0 => {
            let var = std::env::var("EMBER_THREADS").ok();
            let n = var.and_then(|v| v.parse().ok()).filter(|&n| n > 0);
            let cores = || thread::available_parallelism().map_or(1, |n| n.get());
            THREADS.store(n.unwrap_or_else(cores), Relaxed);
            threads()
        }
        n => n,
    }
}
// @end

// @listing parallel
/// `matmul_blocked` with the rows of `c` cut into bands, one per thread. Each
/// entry is still computed by one thread, adding the same products in the same
/// order, so the result is the same, to the bit, for any number of threads.
pub fn matmul_parallel(a: &[f32], b: &[f32], c: &mut [f32], d: Dims, threads: usize) {
    let (m, k, n) = d;
    check_lens(a, b, c, d);
    let band = m.div_ceil(threads.max(1)).next_multiple_of(TILE_M); // rows each
    if band >= m || k == 0 || n == 0 {
        return matmul_blocked(a, b, c, d); // one band: no thread to start
    }
    let mut bands = a.chunks(band * k).zip(c.chunks_mut(band * n));
    let (a0, c0) = bands.next().unwrap(); // this thread's own band
    thread::scope(|scope| {
        for (a, c) in bands {
            let rows = c.len() / n; // `band`, or fewer in the last band
            scope.spawn(move || matmul_blocked(a, b, c, (rows, k, n)));
        }
        matmul_blocked(a0, b, c0, (band, k, n));
    }); // every thread has finished here
}
// @end

// @listing threshold
/// A thread costs microseconds to start and stop, so each one should get at
/// least this much work (Section 24.7 measures it).
pub const MIN_FLOPS_PER_THREAD: usize = 1 << 21;

/// How many threads a product of `dims` is worth: at most `threads()`.
pub fn threads_for((m, k, n): Dims) -> usize {
    (2 * m * k * n / MIN_FLOPS_PER_THREAD).clamp(1, threads())
}
// @end
