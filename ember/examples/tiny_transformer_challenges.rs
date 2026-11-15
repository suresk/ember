//! Chapter 25, Challenges 25.2 and 25.5: the trained sorter decoded with and
//! without a key-value cache, and with sampling at three temperatures.
//!
//! cargo run --release --example tiny_transformer_challenges

use ember::{
    Config, Module, Rng, Tape, Tensor, TinyTransformer, generate_cached, sample,
};
use std::time::Instant;

#[path = "tiny_transformer.rs"]
#[allow(dead_code)]
mod tiny_transformer;

use tiny_transformer::{N, config, greedy, train};

/// The digits `model` writes after `prompt`, `count` of them, one full forward
/// pass per digit (`greedy` without its stop after `prompt.len()` digits).
fn uncached(model: &TinyTransformer, prompt: &[usize], count: usize) -> Vec<usize> {
    let mut seq = prompt.to_vec();
    for _ in 0..count {
        let tape = Tape::new();
        let ids = seq.iter().map(|&d| d as f32).collect();
        let ids = tape.constant(Tensor::from_vec(ids, &[1, seq.len()]));
        let logits = tape.no_grad(|| model.forward(ids)).value();
        seq.push(logits.select(1, seq.len() - 1).argmax(1)[0]);
    }
    seq.split_off(prompt.len())
}

fn main() {
    let (mut sorter, _) = train(config(N, 0.1), N, 1000, 25);
    let model = &mut sorter.model;
    let mut rng = Rng::new(2025);
    let seqs: Vec<Vec<usize>> = (0..1000)
        .map(|_| (0..N).map(|_| rng.below(10)).collect())
        .collect();
    for cached in [false, true] {
        let start = Instant::now();
        let mut same = 0;
        for digits in &seqs {
            let out = if cached {
                generate_cached(model, digits, N) // no dropout layer on this path
            } else {
                greedy(model, digits)
            };
            let mut sorted = digits.clone();
            sorted.sort_unstable();
            same += (out == sorted) as usize;
        }
        let ms = start.elapsed().as_secs_f32() * 1e3 / seqs.len() as f32;
        println!("cached {cached}: {ms:.3} ms per sequence, {same} sorted");
    }
    let long = Config {
        context: 256,
        ..config(N, 0.0)
    };
    let long = TinyTransformer::new(long, &mut Rng::new(25)); // untrained: timing
    let prompt = [3, 1, 4, 1, 5, 9, 2, 6];
    for count in [8, 56, 248] {
        let start = Instant::now();
        let slow_out = uncached(&long, &prompt, count);
        let slow = start.elapsed().as_secs_f32() * 1e3;
        let start = Instant::now();
        let fast = generate_cached(&long, &prompt, count);
        let quick = start.elapsed().as_secs_f32() * 1e3;
        assert_eq!(fast, slow_out);
        println!(
            "{count} tokens: {slow:.1} ms without the cache, {quick:.1} ms with it"
        );
    }
    model.train(false); // dropout off for sampling
    for temperature in [0.5, 1.0, 2.0] {
        let mut right = 0;
        for digits in &seqs {
            let mut seq = digits.clone();
            for _ in 0..N {
                let tape = Tape::new();
                let ids = seq.iter().map(|&d| d as f32).collect();
                let ids = tape.constant(Tensor::from_vec(ids, &[1, seq.len()]));
                let logits = tape.no_grad(|| model.forward(ids)).value();
                let last = logits.select(1, seq.len() - 1).reshape(&[10]);
                seq.push(sample(&last, temperature, 10, &mut rng));
            }
            let mut sorted = digits.clone();
            sorted.sort_unstable();
            right += (seq[N..] == sorted[..]) as usize;
        }
        println!("temperature {temperature}: {right} of 1000 sorted");
    }
}
