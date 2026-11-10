//! Chapter 25: a pre-norm transformer block and a tiny decoder-only transformer
//! made of them.

use crate::autograd::tape::Var;
use crate::nn::{
    Dropout, Embedding, LayerNorm, Linear, Module, MultiHeadAttention, ReLU,
};
use crate::rng::Rng;
use crate::tensor::Tensor;

// @listing block
/// A pre-norm transformer block: `x + Attn(LN(x))`, then `x + MLP(LN(x))`, with
/// dropout on each branch before it is added back.
#[derive(Debug)]
pub struct Block {
    pub ln1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub drop1: Dropout,
    pub ln2: LayerNorm,
    pub mlp: Mlp,
    pub drop2: Dropout,
}

impl Module for Block {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let x = x + self.drop1.forward(self.attn.forward(self.ln1.forward(x)));
        x + self.drop2.forward(self.mlp.forward(self.ln2.forward(x)))
    }
    children!(ln1, attn, drop1, ln2, mlp, drop2);
}
// @end

// @listing mlp
/// The block's feed-forward part, applied to each position on its own:
/// `dim → 4·dim`, ReLU, `4·dim → dim`.
#[derive(Debug)]
pub struct Mlp {
    pub up: Linear,
    pub down: Linear,
}

impl Module for Mlp {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        self.down.forward(ReLU.forward(self.up.forward(x)))
    }
    children!(up, down);
}
// @end

impl Block {
    /// A block of width `dim`; the two dropout layers draw their seeds from `rng`.
    pub fn new(dim: usize, heads: usize, p: f32, rng: &mut Rng) -> Block {
        let (ln1, ln2) = (LayerNorm::new(dim), LayerNorm::new(dim));
        let attn = MultiHeadAttention::new(dim, heads, rng);
        let up = Linear::new(dim, 4 * dim, rng);
        let mlp = Mlp {
            up,
            down: Linear::new(4 * dim, dim, rng),
        };
        let drop1 = Dropout::new(p, rng.next_u64());
        let drop2 = Dropout::new(p, rng.next_u64());
        Block {
            ln1,
            attn,
            drop1,
            ln2,
            mlp,
            drop2,
        }
    }
}

// @listing config
/// The sizes of a `TinyTransformer`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub vocab: usize,   // number of distinct tokens
    pub context: usize, // longest sequence it reads
    pub dim: usize,     // features per position
    pub heads: usize,   // attention heads per block
    pub layers: usize,  // blocks
    pub dropout: f32,   // in every block
}
// @end

// @listing tiny-transformer
/// A decoder-only transformer: token and position embeddings, `layers` blocks, a
/// final `LayerNorm`, and a linear head to one logit per token of the vocabulary.
#[derive(Debug)]
pub struct TinyTransformer {
    pub tok: Embedding,
    pub pos: Embedding,
    pub blocks: Vec<Block>,
    pub ln: LayerNorm,
    pub head: Linear,
}

impl Module for TinyTransformer {
    /// Token ids `[B, T]` to logits `[B, T, vocab]`; the logits at position `i`
    /// predict token `i + 1` from tokens `0..=i`.
    fn forward<'t>(&self, ids: Var<'t>) -> Var<'t> {
        let t = ids.shape()[1];
        let at = ids.tape().constant(Tensor::arange(&[t])); // positions 0..t
        let x = self.tok.forward(ids) + self.pos.forward(at); // [B, T, dim]
        self.head.forward(self.ln.forward(self.blocks.forward(x)))
    }
    children!(tok, pos, blocks, ln, head);
}
// @end

impl TinyTransformer {
    /// Embeddings, then each block, then the head, all drawn from `rng` in order.
    pub fn new(c: Config, rng: &mut Rng) -> TinyTransformer {
        let tok = Embedding::new(c.vocab, c.dim, rng);
        let pos = Embedding::new(c.context, c.dim, rng);
        let block = |_| Block::new(c.dim, c.heads, c.dropout, rng);
        let blocks: Vec<Block> = (0..c.layers).map(block).collect();
        let ln = LayerNorm::new(c.dim);
        let head = Linear::new(c.dim, c.vocab, rng);
        TinyTransformer {
            tok,
            pos,
            blocks,
            ln,
            head,
        }
    }
}
