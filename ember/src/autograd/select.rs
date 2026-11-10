//! Chapter 25: the operations attention needs that pick elements rather than
//! compute them: rows of a table by index (`index_select`, an embedding's lookup),
//! a real select that overwrites masked positions (`masked_fill`, the causal
//! mask), and equal pieces of an axis as views (`chunk`, Challenge 2.4's).

use crate::autograd::tape::Var;
use crate::broadcast::broadcast_shapes;
use crate::tensor::Tensor;

impl Tensor {
    // @listing tensor-index-select
    /// Rows `ids` of axis 0, in that order: shape `[ids.len(), …rest]`. A copy.
    pub fn index_select(&self, ids: &[usize]) -> Tensor {
        let x = self.contiguous();
        let (rows, data) = (self.shape[0], x.as_slice().unwrap());
        let row = self.numel().checked_div(rows).unwrap_or(0); // per row
        let mut out = Vec::with_capacity(ids.len() * row);
        for &i in ids {
            assert!(i < rows, "index_select: id {i}, {rows} rows");
            out.extend_from_slice(&data[i * row..][..row]);
        }
        let mut shape = self.shape.clone();
        shape[0] = ids.len();
        Tensor::from_vec(out, &shape)
    }
    // @end

    // @listing tensor-masked-fill
    /// `value` where `mask` (broadcast to `self`'s shape) is not 0, else `self`:
    /// a real select, so the value is never multiplied or added to anything.
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Tensor {
        let fits =
            broadcast_shapes(&self.shape, mask.shape()) == Some(self.shape.clone());
        assert!(fits, "masked_fill: mask {:?}, {:?}", mask.shape, self.shape);
        self.zip_map(mask, |x, m| if m != 0.0 { value } else { x })
    }
    // @end
}

impl<'t> Var<'t> {
    // @listing var-index-select
    /// Rows `ids` of axis 0. Backward: each row of `g` is added into row `ids[k]`
    /// of zeros, so a row picked twice collects both gradients.
    pub fn index_select(self, ids: &[usize]) -> Var<'t> {
        let picked = ids.to_vec(); // the rule outlives the caller's slice
        let f = |x: &Tensor| x.index_select(ids);
        self.unary("index_select", f, move |x, _, g| {
            let row = x.numel() / x.shape()[0];
            let rows = picked.iter().flat_map(|&i| i * row..(i + 1) * row);
            Tensor::scatter_add(x.shape(), &rows.collect::<Vec<_>>(), g)
        })
    }
    // @end

    // @listing var-masked-fill
    /// `value` where `mask` is not 0. Backward: the gradient passes where `x` was
    /// kept and is 0 where it was overwritten, since `value` doesn't depend on `x`.
    pub fn masked_fill(self, mask: &Tensor, value: f32) -> Var<'t> {
        let kept = mask.clone(); // shares the buffer; the rule owns it
        let f = |x: &Tensor| x.masked_fill(mask, value);
        self.unary("masked_fill", f, move |_, _, g| g.masked_fill(&kept, 0.0))
    }
    // @end

    // @listing var-chunk
    /// Axis `dim` cut into pieces of `⌈len / n⌉`, the last one shorter if need
    /// be: Challenge 2.4's `chunk`, on the tape, as `slice` views.
    pub fn chunk(self, dim: usize, n: usize) -> Vec<Var<'t>> {
        assert!(n >= 1, "chunk: n must be at least 1");
        let len = self.shape()[dim];
        let size = len.div_ceil(n).max(1);
        let piece = |start: usize| self.slice(dim, start, (start + size).min(len), 1);
        (0..len).step_by(size).map(piece).collect()
    }
    // @end
}
