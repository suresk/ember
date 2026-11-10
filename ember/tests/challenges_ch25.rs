//! Tests for Chapter 25's challenges.
use ember::{
    Config, KvCache, Module, Rng, Tape, Tensor, TinyTransformer, Tolerance,
    attention, generate_cached, gradcheck, masked_attention, padding_mask,
    probabilities, rope, sample, sinusoidal,
};

#[path = "../examples/tiny_transformer.rs"]
#[allow(dead_code)]
mod tiny_transformer;

use tiny_transformer::{config, greedy};

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

fn small() -> TinyTransformer {
    let c = Config {
        vocab: 10,
        context: 7,
        dim: 8,
        heads: 2,
        layers: 2,
        dropout: 0.1,
    };
    let mut model = TinyTransformer::new(c, &mut Rng::new(25));
    model.train(false);
    model
}

#[test]
fn predict_parameters_and_a_running_mean() {
    // @listing predict-test
    let mut c = config(8, 0.1); // 10 tokens, 15 positions, 4 heads
    (c.dim, c.layers) = (32, 3);
    let mut model = TinyTransformer::new(c, &mut Rng::new(25));
    let count: usize = model.parameters().iter().map(|p| p.value.numel()).sum();
    let block = 12 * 32 * 32 + 13 * 32;
    assert_eq!(count, 10 * 32 + 15 * 32 + 3 * block + 2 * 32 + 32 * 10 + 10);
    assert_eq!(count, 39_306);
    let tape = Tape::new();
    let q = tape.constant(Tensor::arange(&[1, 4, 2])); // any queries
    let k = tape.constant(Tensor::zeros(&[1, 4, 2])); // every score is 0
    let v = tape.constant(t(&[1., 2., 3., 4.], &[1, 4, 1]));
    let y = attention(q, k, v, true).value();
    assert!(close(&y.to_vec(), &[1.0, 1.5, 2.0, 2.5], 1e-6)); // the prefix's mean
    // @end
}

#[test]
fn temperature_and_top_k() {
    // @listing sample-test
    let logits = t(&[2., 1., 0.], &[3]);
    let p = |temp, k| probabilities(&logits, temp, k).to_vec();
    assert!(close(&p(1.0, 3), &[0.6652, 0.2447, 0.0900], 1e-4));
    assert!(close(&p(0.5, 3), &[0.8668, 0.1173, 0.0159], 1e-4));
    assert!(close(&p(2.0, 3), &[0.5065, 0.3072, 0.1863], 1e-4));
    assert!(close(&p(1.0, 2), &[0.7311, 0.2689, 0.0], 1e-4));
    assert!(close(&p(0.01, 3), &[1.0, 0.0, 0.0], 1e-30)); // greedy, in the limit
    let mut rng = Rng::new(25);
    let mut counts = [0; 3];
    for _ in 0..10_000 {
        counts[sample(&logits, 1.0, 3, &mut rng)] += 1;
    }
    assert_eq!(counts, [6730, 2400, 870]);
    // @end
}

#[test]
fn sinusoidal_positions() {
    // @listing sinusoidal-test
    let pe = sinusoidal(16, 8);
    let want = [0.8415, 0.5403, 0.0998, 0.9950, 0.0100, 1.0, 0.0010, 1.0];
    assert!(close(&pe.select(0, 1).to_vec(), &want, 1e-4));
    let dot = |a: usize, b: usize| {
        let (x, y) = (pe.select(0, a).to_vec(), pe.select(0, b).to_vec());
        x.iter().zip(&y).map(|(x, y)| x * y).sum::<f32>()
    };
    assert!((dot(3, 5) - dot(10, 12)).abs() < 1e-5); // only the distance counts
    assert!((dot(3, 5) - 2.5637).abs() < 1e-4 && (dot(3, 3) - 4.0).abs() < 1e-5);
    // @end
}

#[test]
fn padding_masks() {
    // @listing padding-test
    let mut rng = Rng::new(25);
    let tape = Tape::new();
    let [q, k, v] =
        [0; 3].map(|_| tape.constant(Tensor::randn(&[2, 4, 2], &mut rng)));
    let short = [q, k, v].map(|x| x.slice(0, 1, 2, 1).slice(1, 0, 2, 1));
    let [qs, ks, vs] = short; // the second sequence's first 2 tokens, alone
    let alone = attention(qs, ks, vs, true).value();
    let right = masked_attention([q, k, v], &padding_mask(&[4, 2], 4, false)).value();
    let real = right.select(0, 1).slice(0, 0, 2, 1).to_vec();
    assert!(close(&real, &alone.to_vec(), 1e-6)); // padding at the end: no change
    let left = masked_attention([q, k, v], &padding_mask(&[4, 2], 4, true)).value();
    let rows = left.select(0, 1).to_vec();
    assert!(rows[..4].iter().all(|x| x.is_nan())); // two rows with no key at all
    // @end
}

#[test]
fn a_key_value_cache() {
    // @listing cache-test
    let model = small();
    for digits in [[3, 1, 2, 0], [9, 9, 0, 5], [4, 4, 4, 4]] {
        assert_eq!(generate_cached(&model, &digits, 4), greedy(&model, &digits));
    }
    // @end
    let (seq, tape) = ([3., 1., 2., 0., 0., 1.], Tape::new());
    let full = model.forward(tape.constant(t(&seq, &[1, 6]))).value();
    let mut cache = KvCache::default();
    for (p, &d) in seq.iter().enumerate() {
        let one = cache.step(&model, d as usize, p).to_vec();
        assert!(close(&one, &full.select(1, p).to_vec(), 1e-5));
    }
}

#[test]
fn rotary_positions() {
    // @listing rope-test
    let tape = Tape::new();
    let q = Tensor::randn(&[1, 1, 4], &mut Rng::new(25));
    let same = tape.constant(q.expand(&[1, 12, 4])); // one vector at 12 positions
    let r = rope(same, 10_000.0).value();
    let at = |p: usize| r.select(1, p).to_vec();
    let dot = |a: usize, b: usize| at(a).iter().zip(at(b)).map(|(x, y)| x * y).sum();
    let (shifted, same_gap) = ([dot(5, 8), dot(9, 2)], [dot(0, 3), dot(7, 0)]);
    assert!(close(&shifted, &same_gap, 1e-5)); // the score depends on the gap only
    let length = q.to_vec().iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((dot(7, 7).sqrt() - length).abs() < 1e-5); // a rotation keeps lengths
    // @end
    let tol = Tolerance::default();
    let x = Tensor::randn(&[2, 3, 4], &mut Rng::new(26));
    assert!(gradcheck(|v| rope(v[0], 100.0), &[x], tol).is_ok());
}
