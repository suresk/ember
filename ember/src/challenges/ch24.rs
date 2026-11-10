//! Chapter 24 challenges.

use std::sync::mpsc;
use std::thread;

use crate::data::{Batch, TensorDataset};
use crate::matmul::{Dims, check_lens, matmul_blocked};
use crate::nn::Window;
use crate::rng::Rng;
use crate::tensor::Tensor;

// @listing par-map
/// Numbers per thread below which `par_map` stays on one thread.
pub const MIN_PER_THREAD: usize = 1 << 16;

/// `x.map(f)` on up to `threads` threads, each mapping one run of the output.
pub fn par_map(x: &Tensor, f: impl Fn(f32) -> f32 + Sync, threads: usize) -> Tensor {
    let threads = (x.numel() / MIN_PER_THREAD).clamp(1, threads);
    if threads == 1 {
        return x.map(f); // not worth a thread
    }
    let x = x.contiguous();
    let src = x.as_slice().unwrap();
    let mut out = vec![0.0; src.len()];
    let run = src.len().div_ceil(threads);
    let f = &f; // each thread borrows the one closure
    thread::scope(|s| {
        for (out, src) in out.chunks_mut(run).zip(src.chunks(run)) {
            s.spawn(move || {
                out.iter_mut().zip(src).for_each(|(y, &v)| *y = f(v));
            });
        }
    });
    Tensor::from_vec(out, x.shape())
}
// @end

// @listing tn-kernel
/// Adds `aᵀ · b` into `c`, for `a` row-major `k × m` and `b` row-major `k × n`:
/// row `p` of each gives `c[i][..] += a[p][i] · b[p][..]`, an ikj loop that
/// reads both operands in order, so `aᵀ` is never copied.
pub fn matmul_tn_kernel(a: &[f32], b: &[f32], c: &mut [f32], (m, k, n): Dims) {
    check_lens(a, b, c, (m, k, n)); // a has k · m numbers, like m · k
    if m == 0 || n == 0 {
        return;
    }
    for (a_row, b_row) in a.chunks_exact(m).zip(b.chunks_exact(n)) {
        for (&a_pi, c_row) in a_row.iter().zip(c.chunks_exact_mut(n)) {
            for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                *c_ij += a_pi * b_pj;
            }
        }
    }
}
// @end

impl Tensor {
    // @listing matmul-tn
    /// `selfᵀ @ other` for `self` `[k, m]` and `other` `[k, n]`, without
    /// copying the transpose: the same numbers as `self.transpose(0, 1)
    /// .matmul(other)`, bit for bit.
    pub fn matmul_tn(&self, other: &Tensor) -> Tensor {
        let (s, o) = (self.shape(), other.shape());
        let (&[k, m], &[k2, n]) = (s, o) else {
            panic!("matmul_tn: {s:?}, {o:?}")
        };
        assert_eq!(k, k2, "matmul_tn: {s:?}ᵀ @ {o:?}");
        let (a, b) = (self.contiguous(), other.contiguous());
        let mut c = vec![0.0; m * n];
        let (a, b) = (a.as_slice().unwrap(), b.as_slice().unwrap());
        matmul_tn_kernel(a, b, &mut c, (m, k, n));
        Tensor::from_vec(c, &[m, n])
    }
    // @end
}

// @listing im2col-parallel
/// `x.im2col(win)` with the images in bands, one band per thread. A thread
/// can't share `x`, which holds an `Rc`, so it gets its images as a slice and
/// builds a tensor of its own from them.
pub fn im2col_parallel(x: &Tensor, win: Window, threads: usize) -> Tensor {
    let &[n, c, h, w] = x.shape() else {
        panic!("im2col_parallel: shape {:?}", x.shape());
    };
    let x = x.contiguous();
    let [rows, cols] = win.matrix([n, c, h, w]);
    let mut m = vec![0.0; rows * cols];
    let per = n.div_ceil(threads.max(1)); // images per band
    let (image, fields) = (c * h * w, rows / n * cols); // numbers per image
    let bands = x.as_slice().unwrap().chunks(per * image);
    thread::scope(|s| {
        for (src, out) in bands.zip(m.chunks_mut(per * fields)) {
            s.spawn(move || {
                let shape = [src.len() / image, c, h, w];
                let band = Tensor::from_vec(src.to_vec(), &shape); // this thread's
                out.copy_from_slice(band.im2col(win).as_slice().unwrap());
            });
        }
    });
    Tensor::from_vec(m, &[rows, cols])
}
// @end

// @listing stacks
/// A stack of `batch` products `[m, k] @ [k, n]` with whole matrices handed to
/// threads: thread `t` computes every matrix of its band, start to finish.
pub fn matmul_stacks(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    (batch, dims): (usize, Dims),
    threads: usize,
) {
    let (m, k, n) = dims;
    assert_eq!([a.len(), b.len()], [batch * m * k, batch * k * n]);
    if m * k * n == 0 {
        return; // nothing to add into `c`
    }
    let per = batch.div_ceil(threads.max(1)); // matrices per thread
    let bands = a.chunks(per * m * k).zip(b.chunks(per * k * n));
    thread::scope(|s| {
        for ((a, b), c) in bands.zip(c.chunks_mut(per * m * n)) {
            s.spawn(move || {
                let pairs = a.chunks(m * k).zip(b.chunks(k * n));
                for ((a, b), c) in pairs.zip(c.chunks_mut(m * n)) {
                    matmul_blocked(a, b, c, dims);
                }
            });
        }
    });
}
// @end

// @listing prefetch
/// One epoch of shuffled batches of `ds`, as `DataLoader::new(ds, size)
/// .shuffle(..).drop_last(true)` would serve them, with each batch gathered on
/// a second thread while `step` runs on the one before. Batches cross between
/// the threads as plain vectors, since a `Tensor` can't.
pub fn prefetched_epoch(
    ds: &TensorDataset,
    size: usize,
    rng: &mut Rng,
    mut step: impl FnMut(Batch),
) {
    let (x, labels) = ds.tensors();
    let (pixels, row) = (x.as_slice().unwrap(), x.numel() / labels.len());
    let mut order: Vec<usize> = (0..labels.len()).collect();
    rng.shuffle(&mut order);
    let (send, receive) = mpsc::sync_channel::<(Vec<f32>, Vec<usize>)>(2);
    thread::scope(|s| {
        s.spawn(move || {
            for chunk in order.chunks_exact(size) {
                let gather = chunk.iter().flat_map(|&i| &pixels[i * row..][..row]);
                let ys = chunk.iter().map(|&i| labels[i]).collect();
                if send.send((gather.copied().collect(), ys)).is_err() {
                    return; // the receiver is gone
                }
            }
        });
        for (xs, ys) in receive {
            step((Tensor::from_vec(xs, &[size, row]), ys));
        }
    });
}
// @end
