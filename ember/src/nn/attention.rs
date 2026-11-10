//! Chapter 25: scaled dot-product attention with a causal mask, and the
//! multi-head layer that runs several attentions side by side.

use crate::autograd::tape::Var;
use crate::nn::{Linear, Module};
use crate::rng::Rng;
use crate::tensor::Tensor;

// @listing causal-mask
/// `[t, t]`: 1 where key `j` comes after query `i` (`j > i`), 0 elsewhere. The
/// diagonal is never masked, so no row is masked entirely.
pub fn causal_mask(t: usize) -> Tensor {
    let future = (0..t * t).map(|k| if k % t > k / t { 1.0 } else { 0.0 });
    Tensor::from_vec(future.collect(), &[t, t])
}
// @end

// @listing attention
/// For stacks `q`, `k` of shape `[N, T, d]`: each query's weights over the keys,
/// `softmax(q·kᵀ / √d)`, with every key after the query masked out if `causal`.
pub fn attention_weights<'t>(q: Var<'t>, k: Var<'t>, causal: bool) -> Var<'t> {
    let (t, d) = (q.shape()[1], q.shape()[2]);
    let scores = q.matmul(k.transpose(1, 2)).div_scalar((d as f32).sqrt());
    let scores = match causal {
        true => scores.masked_fill(&causal_mask(t), f32::NEG_INFINITY),
        false => scores,
    };
    scores.softmax(2) // each row sums to 1
}

/// Scaled dot-product attention: each query's weighted average of the values.
pub fn attention<'t>(q: Var<'t>, k: Var<'t>, v: Var<'t>, causal: bool) -> Var<'t> {
    attention_weights(q, k, causal).matmul(v)
}
// @end

// @listing multi-head
/// Causal self-attention with `heads` heads of `dim / heads` features each, from
/// one fused projection to queries, keys and values, `qkv`, and one back, `proj`.
#[derive(Debug)]
pub struct MultiHeadAttention {
    pub qkv: Linear,  // [dim, 3·dim]
    pub proj: Linear, // [dim, dim]
    pub heads: usize,
}

impl MultiHeadAttention {
    /// `[B, T, dim]` to `[B·heads, T, dim / heads]`: one stack entry per head.
    pub fn split<'t>(&self, x: Var<'t>) -> Var<'t> {
        let (s, h) = (x.shape(), self.heads);
        let x = x.reshape(&[s[0], s[1], h, s[2] / h]); // [B, T, h, d_h]
        let x = x.permute(&[0, 2, 1, 3]); // a view: [B, h, T, d_h]
        x.reshape(&[s[0] * h, s[1], s[2] / h]) // a copy: the view is strided
    }
    /// The inverse of `split`: `[B·heads, T, d_h]` back to `[B, T, dim]`.
    pub fn merge<'t>(&self, y: Var<'t>) -> Var<'t> {
        let (s, h) = (y.shape(), self.heads);
        let y = y.reshape(&[s[0] / h, h, s[1], s[2]]).permute(&[0, 2, 1, 3]);
        y.reshape(&[s[0] / h, s[1], h * s[2]])
    }
    /// Queries, keys and values for every head, each `[B·heads, T, d_h]`.
    pub fn qkv<'t>(&self, x: Var<'t>) -> [Var<'t>; 3] {
        let parts = self.qkv.forward(x).chunk(2, 3); // three [B, T, dim] views
        [0, 1, 2].map(|i| self.split(parts[i]))
    }
}
// @end

// @listing multi-head-module
impl Module for MultiHeadAttention {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let [q, k, v] = self.qkv(x);
        let heads = attention(q, k, v, true); // [B·heads, T, d_h]
        self.proj.forward(self.merge(heads))
    }
    children!(qkv, proj);
}
// @end

impl MultiHeadAttention {
    /// PyTorch's default `Linear` initialization for both projections.
    pub fn new(dim: usize, heads: usize, rng: &mut Rng) -> MultiHeadAttention {
        assert_eq!(
            dim % heads,
            0,
            "MultiHeadAttention: {dim} features, {heads} heads"
        );
        let qkv = Linear::new(dim, 3 * dim, rng);
        let proj = Linear::new(dim, dim, rng);
        MultiHeadAttention { qkv, proj, heads }
    }

    /// The attention weights of every head, `[B·heads, T, T]`: what `forward`
    /// averages the values with. For looking at, not for training.
    pub fn weights<'t>(&self, x: Var<'t>) -> Var<'t> {
        let [q, k, _] = self.qkv(x);
        attention_weights(q, k, true)
    }
}
