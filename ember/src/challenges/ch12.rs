//! Chapter 12 challenges.

use crate::autograd::gradcheck::{Tolerance, WorstElement};
use crate::autograd::tape::{Tape, Var};
use crate::challenges::ch01::unravel_index;
use crate::nn::{Module, Parameter, Sequential};
use crate::rng::Rng;
use crate::tensor::Tensor;

// @listing moments
/// The mean and the variance of `xs`, added up in `f64`.
pub fn moments(xs: &[f32]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean, var)
}
// @end

// @listing naive-shuffle
/// The tempting mistake: every position swaps with *any* position.
pub fn naive_shuffle<T>(rng: &mut Rng, xs: &mut [T]) {
    for i in 0..xs.len() {
        let j = rng.below(xs.len());
        xs.swap(i, j);
    }
}
// @end

// @listing leaky
/// Leaky ReLU as a module: `x` for positive `x`, else `slope · x`.
#[derive(Clone, Copy, Debug)]
pub struct LeakyReLU(pub f32);

impl Module for LeakyReLU {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        x.leaky_relu(self.0)
    }

    fn parameters(&mut self) -> Vec<&mut Parameter> {
        Vec::new()
    }
}

/// He's standard deviation for a leaky ReLU: variance `2 / ((1 + slope²) fan_in)`.
pub fn he_std_leaky(fan_in: usize, slope: f32) -> f32 {
    (2.0 / ((1.0 + slope * slope) * fan_in as f32)).sqrt()
}
// @end

// @listing builder
/// The module that changes nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl Module for Identity {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        x
    }

    fn parameters(&mut self) -> Vec<&mut Parameter> {
        Vec::new()
    }
}

impl Sequential {
    /// The same list with `layer` at the end, boxed here instead of by the caller.
    pub fn with(mut self, layer: impl Module + 'static) -> Sequential {
        self.layers.push(Box::new(layer));
        self
    }
}
// @end

// @listing truncated
/// Normal samples with standard deviation `std`, each redrawn until it is
/// within two standard deviations of 0.
pub fn truncated_normal(shape: &[usize], std: f32, rng: &mut Rng) -> Tensor {
    let n: usize = shape.iter().product();
    let mut draw = || loop {
        let z = rng.normal();
        if z.abs() <= 2.0 {
            return z * std;
        }
    };
    Tensor::from_vec((0..n).map(|_| draw()).collect(), shape)
}
// @end

// @listing param-loss
/// `L = Σ (1, 2, 3, …) · model(x)`, added up in `f64`, recorded on no tape.
fn weighted_loss(model: &dyn Module, x: &Tensor) -> f64 {
    let tape = Tape::new();
    let y = tape.no_grad(|| model.forward(tape.constant(x.clone())).value());
    let terms = y.to_vec().into_iter().enumerate();
    terms.map(|(i, v)| (i + 1) as f64 * v as f64).sum()
}

/// The gradient `backward_into` leaves for the same loss, one per parameter.
fn param_grads(model: &mut dyn Module, x: &Tensor) -> Vec<Tensor> {
    model.zero_grad();
    let tape = Tape::new();
    let y = model.forward(tape.constant(x.clone()));
    let weights = Tensor::arange(&y.shape()).add_scalar(1.0);
    let loss = (y * tape.constant(weights)).sum_all();
    loss.backward_into(&mut model.parameters());
    let zeros = |p: &&mut Parameter| Tensor::zeros(p.value.shape());
    let grad = |p: &&mut Parameter| p.grad.clone().unwrap_or_else(|| zeros(p));
    model.parameters().iter().map(grad).collect()
}
// @end

// @listing param-slope
/// The central difference of the loss in element `j` of parameter `i`. The
/// parameter is put back as it was, buffer and all.
fn slope(model: &mut dyn Module, x: &Tensor, ij: (usize, usize), eps: f32) -> f32 {
    let (i, j) = ij;
    let value = model.parameters()[i].value.clone();
    let x0 = value.to_vec()[j];
    let (up, down) = (x0 + eps * x0.abs().max(1.0), x0 - eps * x0.abs().max(1.0));
    let mut loss_with = |xj: f32| {
        let mut v = value.to_vec();
        v[j] = xj;
        model.parameters()[i].value = Tensor::from_vec(v, value.shape());
        weighted_loss(model, x)
    };
    let rise = loss_with(up) - loss_with(down);
    model.parameters()[i].value = value;
    (rise / (up - down) as f64) as f32
}
// @end

// @listing param-check
/// `gradcheck` for a module's parameters, with Chapter 10's tolerance.
pub fn gradcheck_params(
    model: &mut dyn Module,
    x: &Tensor,
    tol: Tolerance,
) -> Result<WorstElement, WorstElement> {
    let mut worst = (f32::NEG_INFINITY, 0, 0, 0.0, 0.0); // ratio, i, j, a, n
    for (i, g) in param_grads(model, x).into_iter().enumerate() {
        for (j, a) in g.to_vec().into_iter().enumerate() {
            let n = slope(model, x, (i, j), tol.eps);
            let ratio = tol.ratio(a, n);
            if ratio > worst.0 || ratio.is_nan() {
                worst = (ratio, i, j, a, n);
            }
        }
    }
    let (ratio, input, j, analytic, numeric) = worst;
    let index = unravel_index(j, model.parameters()[input].value.shape());
    let worst = WorstElement {
        input,
        index,
        analytic,
        numeric,
        ratio,
    };
    if ratio <= 1.0 { Ok(worst) } else { Err(worst) }
}
// @end

// @listing orthogonal
/// A random `n × n` orthogonal matrix: Gram–Schmidt on the rows of a Gaussian one.
pub fn orthogonal(n: usize, rng: &mut Rng) -> Tensor {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let gauss = Tensor::randn(&[n, n], rng).to_vec();
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    for row in gauss.chunks(n) {
        let mut v: Vec<f64> = row.iter().map(|&x| x as f64).collect();
        for q in &rows {
            let d = dot(&v, q); // v's component along an earlier row...
            v.iter_mut().zip(q).for_each(|(a, b)| *a -= d * b); // ...removed
        }
        let norm = dot(&v, &v).sqrt();
        rows.push(v.iter().map(|a| a / norm).collect());
    }
    Tensor::from_vec(rows.concat().iter().map(|&a| a as f32).collect(), &[n, n])
}
// @end
