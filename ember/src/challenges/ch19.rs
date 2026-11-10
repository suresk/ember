//! Chapter 19 challenges.

use crate::autograd::tape::Tape;
use crate::data::{Batch, DataLoader, Dataset};
use crate::nn::{Module, cross_entropy};
use crate::rng::Rng;
use crate::tensor::Tensor;
use std::cell::RefCell;

// @listing shift
/// A `side × side` image moved down by `dy` and right by `dx` pixels; the pixels
/// moved in from outside are `fill`.
pub fn shift(
    image: &[f32],
    side: usize,
    (dy, dx): (i32, i32),
    fill: f32,
) -> Vec<f32> {
    let mut out = vec![fill; side * side];
    let n = side as i32;
    for r in 0..n {
        for c in 0..n {
            let (from_r, from_c) = (r - dy, c - dx);
            if (0..n).contains(&from_r) && (0..n).contains(&from_c) {
                out[(r * n + c) as usize] = image[(from_r * n + from_c) as usize];
            }
        }
    }
    out
}

/// Every example of `dataset`, a square image, shifted by a random number of
/// pixels in `-max..=max` along each axis, drawn anew at each `get`.
pub struct Shifted<'d, D: Dataset + ?Sized> {
    pub dataset: &'d D,
    pub max: i32,
    pub fill: f32,
    pub rng: RefCell<Rng>,
}

impl<D: Dataset + ?Sized> Dataset for Shifted<'_, D> {
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, i: usize) -> (Tensor, usize) {
        let (x, label) = self.dataset.get(i);
        let side = (x.numel() as f64).sqrt() as usize;
        let mut rng = self.rng.borrow_mut();
        let mut draw = || rng.below(2 * self.max as usize + 1) as i32 - self.max;
        let by = (draw(), draw());
        let moved = shift(&x.to_vec(), side, by, self.fill);
        (Tensor::from_vec(moved, x.shape()), label)
    }
}
// @end

// @listing forever
impl<'d, D: Dataset + ?Sized> DataLoader<'d, D> {
    /// Batches without end: epoch after epoch, each in a new order. Training
    /// counted in steps takes as many as it needs with `.take(steps)`.
    pub fn forever(mut self) -> impl Iterator<Item = Batch> + 'd {
        assert!(!self.is_empty(), "forever: no batches in an epoch");
        std::iter::repeat_with(move || self.epoch()).flatten()
    }
}
// @end

// @listing stratified
/// An order of the examples in which every class is spread evenly: the k-th of
/// a class's n members (in a shuffled order) sorts at (k + u) / n, with u drawn
/// once per class in [0, 1). A run of B examples then holds B·n/N of the class,
/// rounded up or down.
pub fn stratified_order(labels: &[usize], rng: &mut Rng) -> Vec<usize> {
    let classes = labels.iter().max().map_or(0, |&c| c + 1);
    let mut members = vec![vec![]; classes];
    for (i, &label) in labels.iter().enumerate() {
        members[label].push(i);
    }
    let mut keyed = vec![];
    for class in &mut members {
        rng.shuffle(class);
        let (n, u) = (class.len() as f64, rng.uniform() as f64);
        for (k, &i) in class.iter().enumerate() {
            keyed.push(((k as f64 + u) / n, i));
        }
    }
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
    keyed.into_iter().map(|(_, i)| i).collect()
}
// @end

// @listing noise-scale
/// The gradient noise scale `tr Σ / |G|²` over a whole dataset: the variance of
/// the per-example gradients, summed over all parameters, over the squared
/// length of their mean. One backward pass per example.
pub fn noise_scale(model: &mut dyn Module, data: &dyn Dataset) -> f64 {
    let n = data.len();
    let grads: Vec<Vec<f64>> = (0..n).map(|i| example_grad(model, data, i)).collect();
    let mean: Vec<f64> = (0..grads[0].len())
        .map(|k| grads.iter().map(|g| g[k]).sum::<f64>() / n as f64)
        .collect();
    let spread = |g: &Vec<f64>| -> f64 {
        g.iter().zip(&mean).map(|(a, m)| (a - m).powi(2)).sum()
    };
    let trace = grads.iter().map(spread).sum::<f64>() / n as f64;
    trace / mean.iter().map(|m| m * m).sum::<f64>()
}

/// Example `i`'s gradient, every parameter's in one vector.
fn example_grad(model: &mut dyn Module, data: &dyn Dataset, i: usize) -> Vec<f64> {
    let (x, label) = data.get_batch(&[i]);
    model.zero_grad();
    let tape = Tape::new();
    let loss = cross_entropy(model.forward(tape.constant(x)), &label);
    loss.backward_into(&mut model.parameters());
    let params = model.parameters();
    let grads = params.iter().map(|p| p.grad.as_ref().expect("a gradient"));
    grads.flat_map(|g| g.to_vec()).map(|v| v as f64).collect()
}
// @end
