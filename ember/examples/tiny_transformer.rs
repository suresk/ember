//! Chapter 25: a tiny transformer learns to sort digits.
//!
//! cargo run --release --example tiny_transformer           train, then sort
//! cargo run --release --example tiny_transformer -- 3000    train 3000 steps
//! cargo run --release --example tiny_transformer -- baseline no blocks at all
//! cargo run --release --example tiny_transformer -- challenges  caching, sampling

use ember::{
    AdamW, Batch, Config, CosineLr, LinearWarmup, LrScheduler, Module, Rng, Tape,
    Tensor, TinyTransformer, Var, base_lrs, generate_cached, no_decay, sample,
    train_step,
};
use std::time::Instant;

/// Digits per sequence.
pub const N: usize = 8;

// @listing task
/// `batch` random sequences of `n` digits, each followed by the same digits
/// sorted: inputs `[batch, 2n - 1]` (the last sorted digit is never read) and,
/// as labels, the `n` sorted digits of every sequence.
pub fn sort_batch(n: usize, batch: usize, rng: &mut Rng) -> Batch {
    let (mut ids, mut labels) = (vec![], vec![]);
    for _ in 0..batch {
        let digits: Vec<usize> = (0..n).map(|_| rng.below(10)).collect();
        let mut sorted = digits.clone();
        sorted.sort();
        let seq = digits.iter().chain(&sorted[..n - 1]);
        ids.extend(seq.map(|&d| d as f32));
        labels.extend(sorted);
    }
    (Tensor::from_vec(ids, &[batch, 2 * n - 1]), labels)
}
// @end

// @listing sorter
/// The transformer, read only where it predicts the answer: positions `n - 1`
/// to `2n - 2`, as `[batch·n, vocab]`, the shape `cross_entropy` takes.
pub struct Sorter {
    pub model: TinyTransformer,
    pub n: usize,
}

impl Module for Sorter {
    fn forward<'t>(&self, ids: Var<'t>) -> Var<'t> {
        let logits = self.model.forward(ids); // [B, 2n - 1, vocab]
        let (b, v, n) = (logits.shape()[0], logits.shape()[2], self.n);
        logits.slice(1, n - 1, 2 * n - 1, 1).reshape(&[b * n, v])
    }
    ember::children!(model);
}
// @end

/// The model of the chapter: 2 blocks of width 64 with 4 heads.
pub fn config(n: usize, dropout: f32) -> Config {
    let context = 2 * n - 1;
    Config {
        vocab: 10,
        context,
        dim: 64,
        heads: 4,
        layers: 2,
        dropout,
    }
}

// @listing greedy
/// Sort `digits` by asking the model for one digit at a time: each step reads
/// everything so far and appends the most likely next digit.
pub fn greedy(model: &TinyTransformer, digits: &[usize]) -> Vec<usize> {
    let mut seq: Vec<f32> = digits.iter().map(|&d| d as f32).collect();
    for _ in 0..digits.len() {
        let tape = Tape::new();
        let ids = tape.constant(Tensor::from_vec(seq.clone(), &[1, seq.len()]));
        let logits = tape.no_grad(|| model.forward(ids)).value(); // [1, T, 10]
        let last = logits.select(1, seq.len() - 1); // [1, 10]
        seq.push(last.argmax(1)[0] as f32);
    }
    seq[digits.len()..].iter().map(|&d| d as usize).collect()
}
// @end

/// The fraction of `count` new sequences that `greedy` sorts exactly right.
pub fn solved(model: &TinyTransformer, n: usize, count: usize, rng: &mut Rng) -> f32 {
    let mut right = 0;
    for _ in 0..count {
        let digits: Vec<usize> = (0..n).map(|_| rng.below(10)).collect();
        let mut sorted = digits.clone();
        sorted.sort();
        right += (greedy(model, &digits) == sorted) as usize;
    }
    right as f32 / count as f32
}

// @listing train
/// `steps` steps of AdamW on fresh batches of 64, with warmup and a cosine
/// decay, from `seed`: the trained model and every step's loss.
pub fn train(c: Config, n: usize, steps: usize, seed: u64) -> (Sorter, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let model = TinyTransformer::new(c, &mut rng);
    let mut sorter = Sorter { model, n };
    let lr = 3e-3;
    let mut opt = AdamW::new(lr).group(no_decay(&mut sorter, lr));
    let warmup = steps / 10;
    let schedule = LinearWarmup::new(warmup, CosineLr::new(steps - warmup, 0.0));
    let base = base_lrs(&opt);
    let mut losses = vec![];
    for t in 0..steps {
        schedule.apply(&mut opt, &base, t);
        let batch = sort_batch(n, 64, &mut rng);
        losses.push(train_step(&mut sorter, &batch, &mut opt, 1.0)); // Section 20.3
    }
    (sorter, losses)
}
// @end

// @listing weights
/// Block `layer`'s attention weights for one sequence, `[heads, T, T]`: the
/// blocks below it run as usual, then its attention reports instead of mixing.
pub fn weights(model: &TinyTransformer, seq: &[usize], layer: usize) -> Tensor {
    let tape = Tape::new();
    let ids: Vec<f32> = seq.iter().map(|&d| d as f32).collect();
    let ids = tape.constant(Tensor::from_vec(ids, &[1, seq.len()]));
    let at = tape.constant(Tensor::arange(&[seq.len()]));
    let x = model.tok.forward(ids) + model.pos.forward(at);
    let x = model.blocks[..layer].iter().fold(x, |x, b| b.forward(x));
    let block = &model.blocks[layer];
    block.attn.weights(block.ln1.forward(x)).value()
}
// @end

/// One head's weights as text, for `digits` and their `sorted` answer: a row
/// per answer digit (the position that writes it), a column per position read,
/// each cell shaded by its weight: blank below 0.1, then `. : + # @`.
pub fn attention_map(
    w: &Tensor,
    digits: &[usize],
    sorted: &[usize],
    head: usize,
) -> String {
    let (n, shades) = (digits.len(), [' ', '.', ':', '+', '#', '@']);
    let read: Vec<usize> = digits.iter().chain(&sorted[..n - 1]).copied().collect();
    let mut out = String::from("writes");
    for (j, d) in read.iter().enumerate() {
        out += &format!("{}{d}", if j == n { " | " } else { " " });
    }
    for (row, d) in sorted.iter().enumerate() {
        out += &format!("\n{d:>6}");
        for j in 0..read.len() {
            let a = w.get(&[head, n - 1 + row, j]);
            let k = ((a * 5.0).round() as usize).min(5);
            out += &format!("{}{}", if j == n { " | " } else { " " }, shades[k]);
        }
    }
    out + "\n"
}

/// Two heads' maps side by side, with a header naming them.
fn side_by_side(left: &str, right: &str, names: (&str, &str)) -> String {
    let mut out = format!("{:<40}{}\n", names.0, names.1);
    for (l, r) in left.lines().zip(right.lines()) {
        out += &format!("{l:<40}{r}\n");
    }
    out
}

/// Train, report, sort five sequences and show what two heads look at.
fn run(steps: usize, layers: usize) {
    let c = Config {
        layers,
        ..config(N, 0.1)
    };
    let start = Instant::now();
    let (mut sorter, losses) = train(c, N, steps, 25);
    let secs = start.elapsed().as_secs_f32();
    sorter.train(false); // dropout off for sorting
    let per = steps / 10;
    for t in (0..steps).step_by(per).chain([steps - 1]) {
        println!("step {:>5}  loss {:.4}", t + 1, losses[t]);
    }
    let ms = 1e3 * secs / steps as f32;
    println!("{steps} steps in {secs:.1} s, {ms:.1} ms per step");
    let model = &sorter.model;
    let mut rng = Rng::new(2025);
    let right = solved(model, N, 1000, &mut rng) * 1000.0;
    println!("sorted {right} of 1,000 new sequences");
    for _ in 0..5 {
        let digits: Vec<usize> = (0..N).map(|_| rng.below(10)).collect();
        println!("{digits:?} -> {:?}", greedy(model, &digits));
    }
    if layers < 2 {
        return;
    }
    let digits = [6, 9, 7, 0, 2, 4, 1, 9];
    let sorted = greedy(model, &digits);
    let seq = [&digits[..], &sorted[..N - 1]].concat(); // what the model reads
    let w = weights(model, &seq, 1);
    for (a, b) in [(0, 2), (1, 3)] {
        let map = |h| attention_map(&w, &digits, &sorted, h);
        let (l, r) = (map(a), map(b));
        let names = (format!("block 1, head {a}"), format!("block 1, head {b}"));
        println!("{}", side_by_side(&l, &r, (&names.0, &names.1)));
    }
}

/// Challenges 25.2 and 25.5: decoding with and without a cache, and sampling.
fn challenges() {
    let (mut sorter, _) = train(config(N, 0.1), N, 1000, 25);
    sorter.train(false);
    let model = &sorter.model;
    let mut rng = Rng::new(2025);
    let seqs: Vec<Vec<usize>> = (0..1000)
        .map(|_| (0..N).map(|_| rng.below(10)).collect())
        .collect();
    for cached in [false, true] {
        let start = Instant::now();
        let mut same = 0;
        for digits in &seqs {
            let out = match cached {
                false => greedy(model, digits),
                true => generate_cached(model, digits, N),
            };
            let mut sorted = digits.clone();
            sorted.sort();
            same += (out == sorted) as usize;
        }
        let ms = start.elapsed().as_secs_f32() * 1e3 / seqs.len() as f32;
        println!("cached {cached}: {ms:.3} ms per sequence, {same} sorted");
    }
    let long = Config {
        context: 256,
        ..config(N, 0.0)
    }; // untrained: for timing
    let model = &TinyTransformer::new(long, &mut Rng::new(25));
    let prompt = [3, 1, 4, 1, 5, 9, 2, 6];
    for count in [8, 56, 248] {
        let start = Instant::now();
        let mut seq = prompt.to_vec();
        for _ in 0..count {
            let tape = Tape::new();
            let ids: Vec<f32> = seq.iter().map(|&d| d as f32).collect();
            let ids = tape.constant(Tensor::from_vec(ids, &[1, seq.len()]));
            let logits = tape.no_grad(|| model.forward(ids)).value();
            seq.push(logits.select(1, seq.len() - 1).argmax(1)[0]);
        }
        let slow = start.elapsed().as_secs_f32() * 1e3;
        let start = Instant::now();
        let fast = generate_cached(model, &prompt, count);
        let quick = start.elapsed().as_secs_f32() * 1e3;
        assert_eq!(fast, seq[8..]);
        println!(
            "{count} tokens: {slow:.1} ms without the cache, {quick:.1} ms with it"
        );
    }
    let model = &sorter.model;
    for temperature in [0.5, 1.0, 2.0] {
        let mut right = 0;
        for digits in &seqs {
            let mut seq: Vec<f32> = digits.iter().map(|&d| d as f32).collect();
            for _ in 0..N {
                let tape = Tape::new();
                let ids =
                    tape.constant(Tensor::from_vec(seq.clone(), &[1, seq.len()]));
                let logits = tape.no_grad(|| model.forward(ids)).value();
                let last = logits.select(1, seq.len() - 1).reshape(&[10]);
                seq.push(sample(&last, temperature, 10, &mut rng) as f32);
            }
            let mut sorted = digits.clone();
            sorted.sort();
            let out: Vec<usize> = seq[N..].iter().map(|&d| d as usize).collect();
            right += (out == sorted) as usize;
        }
        println!("temperature {temperature}: {right} of 1000 sorted");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("baseline") => run(1000, 0),
        Some("challenges") => challenges(),
        Some(steps) => run(steps.parse().expect("a number of steps"), 2),
        None => run(1000, 2),
    }
}
