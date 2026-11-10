//! Chapter 25's challenges: sampling, sinusoidal positions, padding masks, a
//! key-value cache for decoding, and rotary position embeddings.

use crate::autograd::tape::{Tape, Var};
use crate::nn::{Module, TinyTransformer, attention, causal_mask};
use crate::rng::Rng;
use crate::tensor::Tensor;

// @listing probabilities
/// The next token's probabilities from one row of logits `[vocab]`: the `k`
/// largest kept (the rest set to −∞ by `masked_fill`), divided by `temperature`.
pub fn probabilities(logits: &Tensor, temperature: f32, k: usize) -> Tensor {
    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a)); // largest first
    let kth = sorted[k.min(sorted.len()) - 1];
    let below = logits.map(|x| if x < kth { 1.0 } else { 0.0 });
    let kept = logits.masked_fill(&below, f32::NEG_INFINITY);
    kept.mul_scalar(1.0 / temperature).softmax(0)
}

/// A token drawn from `probabilities`: the first whose running total passes a
/// uniform number.
pub fn sample(logits: &Tensor, temperature: f32, k: usize, rng: &mut Rng) -> usize {
    let p = probabilities(logits, temperature, k).to_vec();
    let (u, mut total) = (rng.uniform(), 0.0);
    for (i, &pi) in p.iter().enumerate() {
        total += pi;
        if u < total {
            return i;
        }
    }
    p.iter().rposition(|&pi| pi > 0.0).unwrap() // rounding left u above the total
}
// @end

// @listing sinusoidal
/// Vaswani et al.'s fixed positions `[context, dim]`: for pair `i`, the sine
/// and cosine of `p·ω_i` with `ω_i = 10000^(-2i/dim)`.
pub fn sinusoidal(context: usize, dim: usize) -> Tensor {
    let mut out = Vec::with_capacity(context * dim);
    for p in 0..context {
        for j in 0..dim {
            let omega = 10000f64.powf(-((j / 2 * 2) as f64) / dim as f64);
            let angle = p as f64 * omega;
            out.push(if j % 2 == 0 { angle.sin() } else { angle.cos() } as f32);
        }
    }
    Tensor::from_vec(out, &[context, dim])
}
// @end

// @listing padding
/// `[N, T, T]`: 1 where query `i` of sequence `s` may not read key `j`, because
/// `j` is in its future or is padding. Sequence `s` has `lengths[s]` tokens, at
/// the end of the `T` positions if `left`, else at the start.
pub fn padding_mask(lengths: &[usize], t: usize, left: bool) -> Tensor {
    let future = causal_mask(t).to_vec();
    let mut out = Vec::with_capacity(lengths.len() * t * t);
    for &len in lengths {
        let pad = |j: usize| if left { j < t - len } else { j >= len };
        for (k, &f) in future.iter().enumerate() {
            out.push(if f != 0.0 || pad(k % t) { 1.0 } else { 0.0 });
        }
    }
    Tensor::from_vec(out, &[lengths.len(), t, t])
}

/// Section 25.2's attention with any mask of 1s (masked) and 0s.
pub fn masked_attention<'t>([q, k, v]: [Var<'t>; 3], mask: &Tensor) -> Var<'t> {
    let d = q.shape()[2] as f32;
    let scores = q.matmul(k.transpose(1, 2)).div_scalar(d.sqrt());
    let weights = scores.masked_fill(mask, f32::NEG_INFINITY).softmax(2);
    weights.matmul(v)
}
// @end

// @listing kv-cache
/// Each block's keys and values so far, one `Vec` per head of `[t, d_h]` rows.
#[derive(Clone, Debug, Default)]
pub struct KvCache {
    pub keys: Vec<Vec<Vec<f32>>>, // [block][head] → t·d_h numbers
    pub values: Vec<Vec<Vec<f32>>>, // the same for the values
}

/// `rows`, one `[1, d_h]` matrix per head (`[heads, 1, d_h]`), appended to
/// `cache`; then all of it as `[heads, t, d_h]`.
fn grow(cache: &mut Vec<Vec<f32>>, rows: &Tensor) -> Tensor {
    let (heads, dh) = (rows.shape()[0], rows.shape()[2]);
    cache.resize(heads, vec![]);
    for (h, head) in cache.iter_mut().enumerate() {
        head.extend(rows.select(0, h).to_vec());
    }
    let t = cache[0].len() / dh;
    Tensor::from_vec(cache.concat(), &[heads, t, dh])
}
// @end

// @listing cached-step
impl KvCache {
    /// The logits `[vocab]` for one more token at position `p`, reading every
    /// earlier position's keys and values from the cache and adding its own.
    pub fn step(&mut self, m: &TinyTransformer, token: usize, p: usize) -> Tensor {
        let tape = Tape::new();
        let one = |x: usize| tape.constant(Tensor::from_vec(vec![x as f32], &[1, 1]));
        tape.no_grad(|| {
            let layers = m.blocks.len();
            self.keys.resize(layers, vec![]);
            self.values.resize(layers, vec![]);
            let mut x = m.tok.forward(one(token)) + m.pos.forward(one(p));
            for (l, b) in m.blocks.iter().enumerate() {
                let [q, k, v] = b.attn.qkv(b.ln1.forward(x)); // [heads, 1, d_h] each
                let k = tape.constant(grow(&mut self.keys[l], &k.value()));
                let v = tape.constant(grow(&mut self.values[l], &v.value()));
                let heads = attention(q, k, v, false); // no future in the cache
                x = x + b.attn.proj.forward(b.attn.merge(heads));
                x = x + b.mlp.forward(b.ln2.forward(x)); // dropout is off
            }
            let logits = m.head.forward(m.ln.forward(x)).value();
            logits.reshape(&[logits.numel()])
        })
    }
}

/// `n` tokens after the prompt `ids`, each the most likely one, one position
/// per step through a cache: Section 25.8's `greedy` without recomputing.
pub fn generate_cached(m: &TinyTransformer, ids: &[usize], n: usize) -> Vec<usize> {
    let (mut cache, mut out) = (KvCache::default(), vec![]);
    let mut logits = Tensor::zeros(&[1]);
    for (p, &id) in ids.iter().enumerate() {
        logits = cache.step(m, id, p); // the prompt
    }
    for p in ids.len()..ids.len() + n {
        out.push(logits.argmax(0)[0]);
        if out.len() < n {
            logits = cache.step(m, out[out.len() - 1], p);
        }
    }
    out
}
// @end

// @listing rope
/// Rotary position embedding of `x`, `[N, T, d]` with `d` even: features `2i`
/// and `2i + 1` at position `p` are turned by the angle `p·base^(-2i/d)`.
pub fn rope(x: Var<'_>, base: f32) -> Var<'_> {
    let (t, d) = (x.shape()[1], x.shape()[2]);
    let (mut cos, mut sin) = (vec![], vec![]);
    for p in 0..t {
        for j in 0..d {
            let angle =
                p as f64 * (base as f64).powf(-((j / 2 * 2) as f64) / d as f64);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    let mut turn = vec![0.0; d * d]; // x·turn = [-x1, x0, -x3, x2, …]
    for i in (0..d).step_by(2) {
        (turn[(i + 1) * d + i], turn[i * d + i + 1]) = (-1.0, 1.0);
    }
    let tape = x.tape();
    let c = |v: Vec<f32>, s: &[usize]| tape.constant(Tensor::from_vec(v, s));
    let rows = x
        .reshape(&[x.value().numel() / d, d])
        .matmul(c(turn, &[d, d]));
    x * c(cos, &[t, d]) + rows.reshape(&x.shape()) * c(sin, &[t, d])
}
// @end
