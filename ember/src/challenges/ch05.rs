//! Chapter 5 challenges.

use crate::broadcast::broadcast_shapes;
use crate::matmul::{Dims, TILE_M, TILE_N, check_lens, edge, tile};
use crate::tensor::Tensor;

impl Tensor {
    // @listing dot
    /// The dot product of two vectors of the same length, added up in order in f32,
    /// as the matmul kernels do.
    pub fn dot(&self, other: &Tensor) -> f32 {
        let (a, b) = (&self.shape, &other.shape);
        assert!(
            a.len() == 1 && a == b,
            "dot: shapes {a:?} and {b:?} are not one vector length"
        );
        let (x, y) = (self.to_vec(), other.to_vec());
        x.iter().zip(&y).fold(0.0, |acc, (&p, &q)| acc + p * q)
    }

    /// The sum of the main diagonal of a square matrix.
    pub fn trace(&self) -> f32 {
        let s = &self.shape;
        assert!(s.len() == 2 && s[0] == s[1], "trace: {s:?} is not square");
        self.diagonal().sum_all() // Challenge 2.3's view
    }
    // @end

    // @listing matvec
    /// The matrix-vector product `[m, k] @ [k] -> [m]`: one dot product per row.
    pub fn matvec(&self, v: &Tensor) -> Tensor {
        let (s, t) = (&self.shape, &v.shape);
        assert!(
            s.len() == 2 && t.len() == 1 && s[1] == t[0],
            "matvec: {s:?} @ {t:?}"
        );
        let (a, v) = (self.contiguous(), v.contiguous());
        let (a, v) = (a.as_slice().unwrap(), v.as_slice().unwrap());
        let dot =
            |row: &[f32]| row.iter().zip(v).fold(0.0, |acc, (&x, &y)| acc + x * y);
        let data = match s[1] {
            0 => vec![0.0; s[0]], // chunks_exact(0) would panic
            k => a.chunks_exact(k).map(dot).collect(),
        };
        Tensor::from_vec(data, &[s[0]])
    }
    // @end

    // @listing matmul-nt
    /// `self @ other.T` for `self` of shape `[m, k]` and `other` of shape `[n, k]`,
    /// without making the transpose: every entry is a dot product of two rows.
    pub fn matmul_nt(&self, other: &Tensor) -> Tensor {
        let (s, t) = (&self.shape, &other.shape);
        assert!(
            s.len() == 2 && t.len() == 2 && s[1] == t[1],
            "matmul_nt: {s:?} @ {t:?}.T"
        );
        let (a, b) = (self.contiguous(), other.contiguous());
        let mut c = vec![0.0; s[0] * t[0]];
        let dims = (s[0], s[1], t[0]);
        matmul_nt_kernel(a.as_slice().unwrap(), b.as_slice().unwrap(), &mut c, dims);
        Tensor::from_vec(c, &[s[0], t[0]])
    }
    // @end

    // @listing matmul-broadcast
    /// `self @ other` for `[..., m, k] @ [..., k, n]`, broadcasting the batch axes
    /// (everything before the last two), as `torch.matmul` does.
    pub fn matmul_broadcast(&self, other: &Tensor) -> Tensor {
        let (s, t) = (&self.shape, &other.shape);
        assert!(s.len() >= 2 && t.len() >= 2, "matmul: {s:?} @ {t:?}");
        let (sb, tb) = (&s[..s.len() - 2], &t[..t.len() - 2]);
        let Some(batch) = broadcast_shapes(sb, tb) else {
            panic!("matmul: {s:?} @ {t:?}: batch axes don't broadcast");
        };
        let count = batch.iter().product();
        // Expand each operand to the full batch, then fold the batch into one axis.
        let fold = |x: &Tensor, rows: usize, cols: usize| {
            let full = [&batch[..], &[rows, cols]].concat();
            x.expand(&full).reshape(&[count, rows, cols])
        };
        let (m, k, n) = (s[s.len() - 2], s[s.len() - 1], t[t.len() - 1]);
        let c = fold(self, m, k).matmul(&fold(other, k, n));
        c.reshape(&[&batch[..], &[m, n]].concat())
    }
    // @end

    // @listing matmul-strided
    /// `self @ other` for two matrices read through their strides, with no copy.
    pub fn matmul_strided(&self, other: &Tensor) -> Tensor {
        let (s, t) = (&self.shape, &other.shape);
        assert!(
            s.len() == 2 && t.len() == 2 && s[1] == t[0],
            "matmul_strided: {s:?} @ {t:?}"
        );
        let (m, k, n) = (s[0], s[1], t[1]);
        let (a, b) = (&self.storage, &other.storage);
        let (sa, sb) = (&self.strides, &other.strides);
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_ip = a[self.offset + i * sa[0] + p * sa[1]];
                let b_row = other.offset + p * sb[0];
                for (j, c_ij) in c[i * n..(i + 1) * n].iter_mut().enumerate() {
                    *c_ij += a_ip * b[b_row + j * sb[1]];
                }
            }
        }
        Tensor::from_vec(c, &[m, n])
    }
    // @end
}

// @listing nt-kernel
/// Adds `a @ bt.T` into `c`, where `a` is m × k and `bt` is n × k, both row-major.
pub fn matmul_nt_kernel(a: &[f32], bt: &[f32], c: &mut [f32], (m, k, n): Dims) {
    let (have, want) = ([a.len(), bt.len(), c.len()], [m * k, n * k, m * n]);
    assert_eq!(have, want, "matmul_nt_kernel: slices for ({m}, {k}, {n})");
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &bt[j * k..(j + 1) * k];
            let dot = a_row.iter().zip(b_row).fold(0.0, |s, (&x, &y)| s + x * y);
            c[i * n + j] += dot;
        }
    }
}
// @end

// @listing blocked-with
/// `matmul_blocked` with panels of `panel` rows of `b`, for Challenge 5.6.
pub fn matmul_blocked_with(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    (m, k, n): Dims,
    panel: usize,
) {
    check_lens(a, b, c, (m, k, n));
    let (m4, n8) = (m - m % TILE_M, n - n % TILE_N);
    for p0 in (0..k).step_by(panel) {
        let ps = p0..k.min(p0 + panel);
        for i in (0..m4).step_by(TILE_M) {
            for j in (0..n8).step_by(TILE_N) {
                tile(a, b, c, (i, j), ps.clone(), (k, n));
            }
        }
    }
    edge(a, b, c, m4..m, 0..n, (k, n));
    edge(a, b, c, 0..m4, n8..n, (k, n));
}
// @end
