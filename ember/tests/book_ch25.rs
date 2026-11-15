//! The examples printed in Chapter 25 of the guided edition: embeddings as a
//! lookup, attention traced on three positions, the causal mask, why scores are
//! divided by √d, many heads at once, the transformer block, the tiny
//! transformer and its parameter count, and a short run of the sorting task.
use ember::{
    AdamW, Block, Config, Embedding, Module, MultiHeadAttention, Rng, Tape, Tensor,
    TinyTransformer, Tolerance, Var, attention, attention_weights, causal_mask,
    gradcheck, one_hot, train_step,
};

#[path = "../examples/tiny_transformer.rs"]
#[allow(dead_code)]
mod tiny_transformer;

use tiny_transformer::{Sorter, config, greedy, sort_batch};

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn close(got: &[f32], want: &[f32], tol: f32) -> bool {
    let ok = got.len() == want.len();
    let ok = ok && got.iter().zip(want).all(|(&g, &w)| (g - w).abs() <= tol);
    if !ok {
        println!("got {got:?}\nwant {want:?}");
    }
    ok
}

#[test]
fn an_embedding_is_a_lookup() {
    // @listing lookup-test
    let mut table = Embedding::from_tensor(Tensor::arange(&[4, 3])); // 4 tokens
    let tape = Tape::new();
    let ids = tape.constant(t(&[2., 0., 2.], &[3])); // token 2 appears twice
    let rows = table.forward(ids);
    assert_eq!(rows.value().to_vec(), [6., 7., 8., 0., 1., 2., 6., 7., 8.]);
    let w = tape.constant(t(&[1., 2., 3., 10., 20., 30., 100., 200., 300.], &[3, 3]));
    (rows * w).sum_all().backward_into(&mut table.parameters());
    let grad = table.weight.grad.as_ref().unwrap().to_vec();
    assert_eq!(
        grad,
        [10., 20., 30., 0., 0., 0., 101., 202., 303., 0., 0., 0.]
    );
    // @end
}

#[test]
fn a_lookup_is_a_one_hot_product() {
    let table = Tensor::arange(&[4, 3]);
    let ids = [2, 0, 2];
    let tape = Tape::new();
    let a = tape.leaf(table.clone());
    let w = tape.constant(t(&[1., 2., 3., 10., 20., 30., 100., 200., 300.], &[3, 3]));
    let picked = a.index_select(&ids);
    let product = tape.constant(one_hot(&ids, 4)).matmul(a);
    assert_eq!(picked.value().to_vec(), product.value().to_vec());
    let g1 = (picked * w).sum_all().backward().get(a).unwrap().to_vec();
    let g2 = (product * w).sum_all().backward().get(a).unwrap().to_vec();
    assert_eq!(g1, g2);
}

/// Section 25.2's queries, keys and values: three positions, two features.
fn qkv(tape: &Tape) -> [Var<'_>; 3] {
    let q = t(&[1., 0., 0., 2., 1., 1.], &[1, 3, 2]);
    let k = t(&[1., 0., 0., 1., 2., 1.], &[1, 3, 2]);
    let v = t(&[10., 0., 0., 10., 4., 6.], &[1, 3, 2]);
    [q, k, v].map(|x| tape.leaf(x))
}

#[test]
fn attention_by_hand() {
    // @listing trace-test
    let tape = Tape::new();
    let [q, k, v] = qkv(&tape);
    let scores = q.matmul(k.transpose(1, 2)).value();
    assert_eq!(scores.to_vec(), [1., 0., 2., 0., 2., 2., 1., 1., 3.]);
    let w = attention_weights(q, k, true).value(); // scaled, masked, softmax
    let want = [1., 0., 0., 0.1956, 0.8044, 0., 0.1636, 0.1636, 0.6728];
    assert!(close(&w.to_vec(), &want, 1e-4));
    let out = attention(q, k, v, true).value();
    let want = [10., 0., 1.9557, 8.0443, 4.3272, 5.6728];
    assert!(close(&out.to_vec(), &want, 1e-4));
    // @end
    // Without the mask, the first query looks mostly at the last key.
    let w = attention_weights(q, k, false).value().to_vec();
    assert!(close(&w[..3], &[0.2840, 0.1400, 0.5760], 1e-4));
}

#[test]
fn the_causal_mask() {
    // @listing mask-test
    assert_eq!(
        causal_mask(3).to_vec(),
        [0., 1., 1., 0., 0., 1., 0., 0., 0.]
    );
    let tape = Tape::new();
    let s = tape.leaf(t(&[1., 2., 3., 4.], &[2, 2]));
    let masked = s.masked_fill(&causal_mask(2), f32::NEG_INFINITY);
    assert_eq!(masked.value().to_vec(), [1., f32::NEG_INFINITY, 3., 4.]);
    let w = tape.constant(t(&[10., 20., 30., 40.], &[2, 2]));
    let grad = (masked.softmax(1) * w).sum_all().backward();
    let g = grad.get(s).unwrap().to_vec();
    assert_eq!(g[1], 0.0); // the masked score gets no gradient, and no NaN
    assert!(g.iter().all(|x| x.is_finite()));
    // @end
}

#[test]
fn arithmetic_masks_fail() {
    // @listing nan-test
    let s = t(&[1., 2., 3., 4.], &[2, 2]);
    let inf = f32::NEG_INFINITY;
    let added = s.add(&causal_mask(2).mul_scalar(inf)); // 0·(−∞) is NaN
    let added = added.to_vec();
    assert!(added[0].is_nan() && added[2].is_nan() && added[3].is_nan());
    assert_eq!(added[1], inf); // only the masked place is right
    let all = t(&[inf, inf], &[1, 2]).softmax(1); // a row with nothing left
    assert!(all.to_vec().iter().all(|x| x.is_nan()));
    assert!((inf - inf).is_nan() && (0.0 * inf).is_nan() && inf.exp() == 0.0);
    // @end
}

#[test]
fn masking_is_a_real_select() {
    // gradcheck passes through masked_fill with −∞ and the softmax after it.
    let tol = Tolerance::default();
    let x = Tensor::randn(&[2, 3, 3], &mut Rng::new(25));
    let mask = causal_mask(3);
    let worst = gradcheck(
        |v| v[0].masked_fill(&mask, f32::NEG_INFINITY).softmax(2),
        &[x],
        tol,
    );
    assert!(worst.is_ok());
}

/// For `d` features, 100 stacks of 16 random queries and keys from seed 25: the
/// standard deviation of the scores, the mean largest weight of a row, and the
/// number of rows (of 1,600) that put more than 99% on one key.
fn peaks(d: usize, scaled: bool) -> (f32, f32, usize) {
    let mut rng = Rng::new(25);
    let q = Tensor::randn(&[100, 16, d], &mut rng);
    let k = Tensor::randn(&[100, 16, d], &mut rng);
    let mut s = q.matmul(&k.transpose(1, 2));
    if scaled {
        s = s.mul_scalar(1.0 / (d as f32).sqrt());
    }
    let (_, var) = s.reshape(&[s.numel()]).mean_var(0);
    let top = s.softmax(2).max(2, false).to_vec();
    let mean = top.iter().sum::<f32>() / top.len() as f32;
    let peaked = top.iter().filter(|&&w| w > 0.99).count();
    (var.item().sqrt(), mean, peaked)
}

#[test]
fn why_divide_by_root_d() {
    // (d, standard deviation, mean largest weight, rows over 0.99), unscaled
    let raw = [
        (2, 1.418, 0.2913, 1),
        (16, 4.036, 0.6906, 94),
        (64, 8.018, 0.8435, 463),
        (256, 15.980, 0.9202, 898),
    ];
    // and divided by √d: (standard deviation, mean largest weight), no row over 0.99
    let scaled = [
        (1.003, 0.2179),
        (1.009, 0.2454),
        (1.002, 0.2441),
        (0.999, 0.2481),
    ];
    for ((d, std, top, peaked), (std_s, top_s)) in raw.into_iter().zip(scaled) {
        let (s, m, p) = peaks(d, false);
        assert!(close(&[s, m], &[std, top], 1e-3) && p == peaked, "d {d}");
        let (s, m, p) = peaks(d, true);
        assert!(
            close(&[s, m], &[std_s, top_s], 1e-3) && p == 0,
            "d {d} scaled"
        );
    }
}

#[test]
fn heads_are_separate_attentions() {
    // Two heads of two features: each head is Section 25.2's attention on its
    // own columns of q, k and v, and the projection adds up the heads' parts.
    let mut rng = Rng::new(25);
    let mha = MultiHeadAttention::new(4, 2, &mut rng);
    let x = Tensor::randn(&[2, 3, 4], &mut rng);
    let tape = Tape::new();
    let y = mha.forward(tape.constant(x.clone())).value();
    let qkv = mha.qkv.forward(tape.constant(x)); // [2, 3, 12]
    let proj = tape.constant(mha.proj.weight.value.clone()); // [4, 4]
    let mut sum = tape.constant(mha.proj.bias.value.clone());
    for h in 0..2 {
        let cols = |at: usize| qkv.slice(2, at + 2 * h, at + 2 * h + 2, 1);
        let head = attention(cols(0), cols(4), cols(8), true); // [2, 3, 2]
        let rows = head
            .reshape(&[6, 2])
            .matmul(proj.slice(0, 2 * h, 2 * h + 2, 1));
        sum = sum + rows.reshape(&[2, 3, 4]);
    }
    assert!(close(&y.to_vec(), &sum.value().to_vec(), 1e-5));
}

#[test]
fn attention_passes_gradcheck() {
    let mut rng = Rng::new(25);
    let inputs = [0, 1, 2].map(|_| Tensor::randn(&[2, 3, 2], &mut rng));
    let tol = Tolerance::default();
    let worst = gradcheck(|v| attention(v[0], v[1], v[2], true), &inputs, tol);
    assert!(worst.is_ok_and(|w| w.ratio < 0.12));
    let mha = MultiHeadAttention::new(4, 2, &mut rng);
    let x = Tensor::randn(&[2, 3, 4], &mut rng);
    let worst = gradcheck(|v| mha.forward(v[0]), &[x], tol);
    assert!(worst.is_ok_and(|w| w.ratio < 0.2));
}

#[test]
fn a_block_passes_gradcheck() {
    let mut rng = Rng::new(25);
    let mut block = Block::new(4, 2, 0.5, &mut rng);
    block.train(false); // dropout's masks are random: check the evaluation mode
    let x = Tensor::randn(&[2, 3, 4], &mut rng);
    let worst = gradcheck(|v| block.forward(v[0]), &[x], Tolerance::default());
    assert!(worst.is_ok_and(|w| w.ratio < 0.06));
}

#[test]
fn counting_parameters() {
    // @listing count-test
    let mut model = TinyTransformer::new(config(8, 0.1), &mut Rng::new(25));
    let named = model.named_parameters();
    let sizes: Vec<usize> = named.iter().map(|(_, p)| p.value.numel()).collect();
    assert_eq!(named.len(), 30);
    assert_eq!(sizes[..2], [640, 960]); // tok.weight [10, 64], pos.weight [15, 64]
    assert_eq!(named[4].0, "blocks.0.attn.qkv.weight");
    assert_eq!(sizes[2..14].iter().sum::<usize>(), 49_984); // one block
    assert_eq!(sizes.iter().sum::<usize>(), 102_346);
    // @end
    let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names[..4],
        [
            "tok.weight",
            "pos.weight",
            "blocks.0.ln1.weight",
            "blocks.0.ln1.bias"
        ]
    );
    assert_eq!(
        names[10..14],
        [
            "blocks.0.mlp.up.weight",
            "blocks.0.mlp.up.bias",
            "blocks.0.mlp.down.weight",
            "blocks.0.mlp.down.bias"
        ]
    );
    assert_eq!(
        names[26..],
        ["ln.weight", "ln.bias", "head.weight", "head.bias"]
    );
    // The same count for GPT-2 small, whose head shares the token table.
    let (d, block) = (768, 12 * 768 * 768 + 13 * 768);
    assert_eq!(50_257 * d + 1024 * d + 12 * block + 2 * d, 124_439_808);
}

#[test]
fn train_and_eval_reach_every_dropout() {
    let mut model = TinyTransformer::new(config(8, 0.1), &mut Rng::new(25));
    let ids = sort_batch(8, 2, &mut Rng::new(1)).0;
    let forward = |model: &TinyTransformer| {
        let tape = Tape::new();
        let y = model.forward(tape.constant(ids.clone()));
        (y.shape(), tape.len())
    };
    assert_eq!(forward(&model), (vec![2, 15, 10], 144));
    model.train(false);
    assert_eq!(forward(&model), (vec![2, 15, 10], 136)); // 4 dropouts, 2 nodes each
}

#[test]
fn a_short_run() {
    // @listing short-run
    let c = Config {
        vocab: 10,
        context: 7,
        dim: 16,
        heads: 2,
        layers: 1,
        dropout: 0.1,
    };
    let mut rng = Rng::new(25);
    let mut sorter = Sorter {
        model: TinyTransformer::new(c, &mut rng),
        n: 4,
    };
    let mut opt = AdamW::new(1e-2);
    let mut losses = vec![];
    for _ in 0..30 {
        let batch = sort_batch(4, 16, &mut rng); // fresh sequences every step
        losses.push(train_step(&mut sorter, &batch, &mut opt, 1.0));
    }
    let at = [losses[0], losses[9], losses[19], losses[29]];
    assert!(close(&at, &[2.4639, 1.9790, 1.6326, 1.5244], 1e-4));
    // @end
    assert_eq!(greedy(&mut sorter.model, &[3, 1, 2, 0]), [0, 1, 2, 2]); // not yet
}

#[test]
fn greedy_switches_to_evaluation() {
    // In training mode, half the features dropped, every forward pass differs.
    // `greedy` switches dropout off, so it writes the same digits every time, and
    // the model is back in its earlier mode afterwards (2 nodes per dropout layer).
    let mut model = TinyTransformer::new(config(4, 0.5), &mut Rng::new(25));
    let first = greedy(&mut model, &[3, 1, 2, 0]);
    for _ in 0..4 {
        assert_eq!(greedy(&mut model, &[3, 1, 2, 0]), first);
    }
    let ids = sort_batch(4, 1, &mut Rng::new(1)).0;
    let nodes = |model: &TinyTransformer| {
        let tape = Tape::new();
        model.forward(tape.constant(ids.clone()));
        tape.len()
    };
    let training = nodes(&model);
    model.train(false);
    assert_eq!(training, nodes(&model) + 8); // 2 blocks, 2 dropout layers each
    greedy(&mut model, &[3, 1, 2, 0]);
    assert!(!model.is_training()); // evaluation mode stays evaluation mode
}

#[test]
fn training_sequences_rarely_repeat() {
    // `train`'s draws from seed 25: the model first, then 1,000 batches of 64.
    let mut rng = Rng::new(25);
    TinyTransformer::new(config(8, 0.1), &mut rng);
    let mut seen = std::collections::HashSet::new();
    let mut repeats = 0;
    for _ in 0..1000 {
        let ids = sort_batch(8, 64, &mut rng).0.to_vec();
        for row in ids.chunks(15) {
            let digits: Vec<u32> = row[..8].iter().map(|&d| d as u32).collect();
            repeats += !seen.insert(digits) as usize;
        }
    }
    assert_eq!(repeats, 23); // of 64,000, from 10⁸ possible sequences
    // The example's 1,000 test sequences, from `Rng::new(2025)`: how many were
    // also training sequences (0.64 expected)?
    let mut rng = Rng::new(2025);
    let mut seen_in_training = 0;
    for _ in 0..1000 {
        let digits: Vec<u32> = (0..8).map(|_| rng.below(10) as u32).collect();
        seen_in_training += seen.contains(&digits) as usize;
    }
    assert_eq!(seen_in_training, 0);
}

#[test]
fn causal_attention_sees_order() {
    // Unmasked attention is permutation-equivariant: shuffle the positions and
    // the outputs shuffle with them. Causal attention is not, because position i
    // reads i + 1 positions.
    let x = Tensor::randn(&[1, 4, 3], &mut Rng::new(25));
    let order = [2, 0, 3, 1];
    let shuffle = |y: &Tensor| {
        let rows: Vec<Tensor> = order.iter().map(|&i| y.select(1, i)).collect();
        rows.iter().flat_map(|r| r.to_vec()).collect::<Vec<f32>>()
    };
    let shuffled = t(&shuffle(&x), &[1, 4, 3]);
    let tape = Tape::new();
    for causal in [false, true] {
        let run = |x: &Tensor| {
            let x = tape.constant(x.clone());
            attention(x, x, x, causal).value()
        };
        let (before, after) = (shuffle(&run(&x)), run(&shuffled).to_vec());
        let gap = before.iter().zip(&after).map(|(a, b)| (a - b).abs());
        let gap = gap.fold(0.0f32, f32::max);
        assert_eq!(gap < 1e-6, !causal, "causal {causal}: {gap}");
    }
}

#[test]
#[should_panic(expected = "causal needs as many keys as queries")]
fn a_causal_mask_needs_square_scores() {
    let tape = Tape::new();
    let q = tape.constant(Tensor::zeros(&[1, 1, 2])); // one query, three keys
    let k = tape.constant(Tensor::zeros(&[1, 3, 2]));
    attention_weights(q, k, true);
}

#[test]
#[should_panic(expected = "TinyTransformer: 16 positions, context 15")]
fn a_sequence_longer_than_the_context() {
    let model = TinyTransformer::new(config(8, 0.1), &mut Rng::new(25));
    let tape = Tape::new();
    model.forward(tape.constant(Tensor::zeros(&[1, 16])));
}
