//! Tests for the Chapter 19 challenges.
use ember::{
    Adam, DataLoader, Dataset, Rng, Shifted, Tensor, TensorDataset, noise_scale,
    random_split, shift, stratified_order, toy,
};
use std::cell::RefCell;

#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;

#[test]
fn predict_the_mnist_numbers() {
    // @listing predict-test
    let ds = TensorDataset::new(Tensor::zeros(&[60_000, 1]), vec![0; 60_000]);
    let mut loader = DataLoader::new(&ds, 64);
    let last = loader.epoch().last().unwrap().0;
    assert_eq!((loader.len(), last.shape()[0]), (938, 32));
    assert_eq!(DataLoader::new(&ds, 64).drop_last(true).len(), 937);
    assert_eq!(DataLoader::new(&ds, 1000).len(), 60); // and 10 of the test set
    assert_eq!(60_000 * 784, 47_040_000); // pixels: one byte each in the file
    assert_eq!(47_040_000 * 4, 188_160_000); // as f32: 179.4 MiB
    assert_eq!(70_000 * 784 * 4, 219_520_000); // with the test set
    // @end
    assert_eq!(47_040_000 + 16, 47_040_016); // the file's size, header included
    assert!((188_160_000.0f64 / 1048576.0 - 179.44).abs() < 0.01);
}

#[test]
fn a_seeded_split() {
    // @listing split-test
    let ds = TensorDataset::new(Tensor::arange(&[10, 1]), (0..10).collect());
    let parts = random_split(&ds, &[7, 3], &mut Rng::new(19));
    assert_eq!(parts[0].indices, [9, 6, 3, 8, 5, 4, 0]);
    assert_eq!(parts[1].indices, [2, 1, 7]); // Section 19.4's first epoch, cut
    let (x, labels) = parts[1].get_batch(&[0, 2]);
    assert_eq!((x.to_vec(), labels), (vec![2.0, 7.0], vec![2, 7]));
    let mut loader = DataLoader::new(&parts[1], 2);
    assert_eq!(loader.epoch().len(), 2);
    // @end
    let big = TensorDataset::new(Tensor::zeros(&[60_000, 1]), vec![0; 60_000]);
    let parts = random_split(&big, &[55_000, 5_000], &mut Rng::new(19));
    let mut all: Vec<usize> = parts.iter().flat_map(|p| p.indices.clone()).collect();
    all.sort();
    assert_eq!(all, (0..60_000).collect::<Vec<_>>()); // disjoint, and everything
}

#[test]
fn shifted_digits() {
    // @listing shift-test
    let image = Tensor::arange(&[16]).to_vec(); // a 4 × 4 image, 0 to 15
    let moved = shift(&image, 4, (1, -2), -1.0); // down 1, left 2
    assert_eq!(moved[..8], [-1.0, -1.0, -1.0, -1.0, 2.0, 3.0, -1.0, -1.0]);
    assert_eq!(moved[12..], [10.0, 11.0, -1.0, -1.0]);
    let ds = TensorDataset::new(Tensor::arange(&[1, 16]), vec![3]);
    let rng = RefCell::new(Rng::new(19));
    let shifted = Shifted {
        dataset: &ds,
        max: 2,
        fill: 0.0,
        rng,
    };
    let (a, b) = (shifted.get(0).0.to_vec(), shifted.get(0).0.to_vec());
    assert_ne!(a, b); // a new shift at every look
    // @end
    let mut seen = std::collections::HashSet::new();
    for _ in 0..500 {
        let x = shifted.get(0).0.to_vec();
        seen.insert(x.iter().map(|v| *v as i32).collect::<Vec<_>>());
    }
    assert_eq!(seen.len(), 25); // 5 × 5 shifts, all different on this image
}

#[test]
fn steps_not_epochs() {
    // @listing forever-test
    let ds = TensorDataset::new(Tensor::arange(&[10, 1]), vec![0; 10]);
    let loader = DataLoader::new(&ds, 4)
        .shuffle(Rng::new(19))
        .drop_last(true);
    let firsts: Vec<f32> = loader
        .forever()
        .take(5)
        .map(|(x, _)| x.get(&[0, 0]))
        .collect();
    assert_eq!(firsts, [9.0, 5.0, 0.0, 3.0, 8.0]); // two per epoch
    // @end
}

/// Batches of `size` in `order`: how many miss some class, and the fewest and
/// most of class 0 in one batch.
fn census(order: &[usize], labels: &[usize], size: usize) -> (usize, [usize; 2]) {
    let (mut missing, mut zeros) = (0, [size, 0]);
    for batch in order.chunks_exact(size) {
        let mut count = [0; 10];
        batch.iter().for_each(|&i| count[labels[i]] += 1);
        missing += count.contains(&0) as usize;
        zeros = [zeros[0].min(count[0]), zeros[1].max(count[0])];
    }
    (missing, zeros)
}

#[test]
fn balanced_batches() {
    // @listing balanced-test
    let labels: Vec<usize> = (0..60_000).map(|i| i % 10).collect();
    let mut rng = Rng::new(19);
    let mut shuffled: Vec<usize> = (0..60_000).collect();
    rng.shuffle(&mut shuffled);
    let even = stratified_order(&labels, &mut rng);
    let mut sorted = even.clone();
    sorted.sort();
    assert_eq!(sorted, (0..60_000).collect::<Vec<_>>()); // a permutation
    assert_eq!(census(&shuffled, &labels, 64), (10, [1, 16]));
    assert_eq!(census(&even, &labels, 64), (0, [6, 7]));
    // @end
    // Inclusion–exclusion: P(some class missing) = Σ (−1)^(k+1) C(10,k) (1 − k/10)^64.
    let p: f64 = (1..=10)
        .map(|k| {
            let c = (0..k).fold(1.0, |c, j| c * (10 - j) as f64 / (j + 1) as f64);
            let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
            sign * c * (1.0 - k as f64 / 10.0).powi(64)
        })
        .sum();
    assert!((p - 0.011762).abs() < 1e-6 && (p * 937.0 - 11.02).abs() < 0.01);
    assert!((10.0 * 0.9f64.powi(64) - 0.011790).abs() < 1e-6); // the first term
    assert!(((64.0f64 * 0.1 * 0.9).sqrt() - 2.4).abs() < 1e-9);
}

#[test]
fn the_noise_scale() {
    // @listing noise-scale-test
    let mut rng = Rng::new(19);
    let data = toy::spirals(25, 1.5, 0.05, &mut rng); // 50 points
    let mut model = spirals::mlp(16, &mut rng);
    let ds = TensorDataset::new(data.0.clone(), data.1.clone());
    let at_start = noise_scale(&mut model, &ds);
    let mut opt = Adam::new(0.03);
    let losses: Vec<f32> = (0..200)
        .map(|_| spirals::step_with(&mut model, &data, &mut opt))
        .collect();
    let late = noise_scale(&mut model, &ds);
    assert_eq!([at_start, late].map(f64::round), [64.0, 932.0]);
    assert!((losses[199] - 0.00237).abs() < 1e-5);
    // @end
    assert!((at_start - 63.558).abs() < 1e-3 && (late - 931.95).abs() < 0.01);
    // A batch of 8's relative error at the start, by the formula: 2.6.
    let error = (at_start / 8.0 * (50.0 - 8.0) / 49.0).sqrt();
    assert!((error - 2.610).abs() < 1e-3);
}
