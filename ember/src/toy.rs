//! Chapter 14: small datasets, made up on the spot, to train on before real data
//! (Part V's `src/data/` loads files; these are generated).

use crate::rng::Rng;
use crate::tensor::Tensor;
use std::f32::consts::{PI, TAU};

/// Points as rows of `[n, 2]`, and a class for each.
pub type Labeled = (Tensor, Vec<usize>);

// @listing xor
/// The four points of XOR as rows of `[4, 2]`, and their targets as `[4, 1]`: 1
/// when exactly one input is 1.
pub fn xor() -> (Tensor, Tensor) {
    let x = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y = [0.0, 1.0, 1.0, 0.0];
    (
        Tensor::from_vec(x.to_vec(), &[4, 2]),
        Tensor::from_vec(y.to_vec(), &[4, 1]),
    )
}
// @end

// @listing spirals
/// Two interleaved spirals of `per_class` points each, as rows of `[2n, 2]`, and
/// their labels: n zeros, then n ones. Each point of arm c is at a uniform
/// random radius r in [0, 1) and angle 2π·turns·r + π·c, then moved by normal
/// noise of standard deviation `noise` in x and in y.
pub fn spirals(per_class: usize, turns: f32, noise: f32, rng: &mut Rng) -> Labeled {
    let n = per_class;
    let mut point = |arm: usize| {
        let r = rng.uniform();
        let angle = TAU * turns * r + PI * arm as f32;
        let x = r * angle.cos() + noise * rng.normal();
        let y = r * angle.sin() + noise * rng.normal();
        [x, y]
    };
    let (mut data, mut labels) = (Vec::with_capacity(4 * n), Vec::new());
    for arm in 0..2 {
        for _ in 0..n {
            data.extend(point(arm));
            labels.push(arm);
        }
    }
    (Tensor::from_vec(data, &[2 * n, 2]), labels)
}
// @end
