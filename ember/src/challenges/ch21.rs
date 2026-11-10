//! Chapter 21 challenges.

use crate::autograd::tape::{Tape, Var};
use crate::nn::{Dropout, Linear, Module, Parameter};
use crate::tensor::Tensor;

// @listing mc-dropout
/// Monte Carlo dropout: `t` forward passes of `x` with dropout left on, under
/// `no_grad`; the mean of the softmax probabilities and their standard
/// deviation, element by element. The model ends in training mode.
pub fn mc_dropout(model: &mut dyn Module, x: &Tensor, t: usize) -> (Tensor, Tensor) {
    model.train(true);
    let pass = |_| {
        let tape = Tape::new();
        let logits = tape.no_grad(|| model.forward(tape.constant(x.clone())));
        logits.value().softmax(1)
    };
    let probs: Vec<Tensor> = (0..t).map(pass).collect();
    let average = |ts: &mut dyn Iterator<Item = Tensor>| {
        let sum = ts.reduce(|s, p| s.add(&p)).expect("t > 0");
        sum.mul_scalar(1.0 / t as f32)
    };
    let mean = average(&mut probs.iter().cloned());
    let var = average(&mut probs.iter().map(|p| p.sub(&mean).powf(2.0)));
    (mean, var.sqrt())
}
// @end

// @listing drop-connect
/// A `Linear` layer whose *weights* are dropped in training, by a mask from
/// `drop`, new at every call. In evaluation it is the plain layer.
pub struct DropConnect {
    pub linear: Linear,
    pub drop: Dropout,
}

impl Module for DropConnect {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        if !self.drop.is_training() {
            return self.linear.forward(x);
        }
        let tape = x.tape();
        let mask = self.drop.mask(self.linear.weight.value.shape());
        let w = tape.param(&self.linear.weight) * tape.constant(mask);
        x.matmul(w) + tape.param(&self.linear.bias) // 2-D inputs only
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        self.linear.parameters()
    }
    fn train(&mut self, on: bool) {
        self.drop.train(on);
    }
}
// @end

// @listing deterministic
/// Does `model` compute the same thing twice in evaluation mode? Leaves the
/// model in training mode.
pub fn eval_is_deterministic(model: &mut dyn Module, x: &Tensor) -> bool {
    model.train(false);
    let run = |model: &dyn Module| {
        let tape = Tape::new();
        let y = tape.no_grad(|| model.forward(tape.constant(x.clone())));
        let y = y.value().to_vec();
        y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    };

    let same = run(model) == run(model);
    model.train(true);
    same
}
// @end

// @listing channel-dropout
/// Dropout of whole channels: for an input `[N, C, …]`, each of the N·C
/// channels is zeroed or scaled as one, by one draw.
pub struct ChannelDropout(pub Dropout);

impl Module for ChannelDropout {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        if !self.0.is_training() || self.0.p == 0.0 {
            return x;
        }
        let mut shape = x.shape();
        shape[2..].fill(1); // [N, C, 1, 1]: broadcast over each channel
        x * x.tape().constant(self.0.mask(&shape))
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        Vec::new()
    }
    fn train(&mut self, on: bool) {
        self.0.train(on);
    }
}
// @end

// @listing every-mask
/// All 2ⁿ dropout masks of `n` units, with their probabilities: a unit is kept,
/// as `1 / (1 - p)`, with probability `1 - p`, and dropped, as 0, with `p`.
pub fn every_mask(n: usize, p: f32) -> Vec<(f64, Tensor)> {
    let keep = 1.0 - p as f64;
    (0..1usize << n)
        .map(|bits| {
            let kept = |i: usize| bits >> i & 1 == 1;
            let prob = (0..n).map(|i| if kept(i) { keep } else { 1.0 - keep });
            let mask = (0..n).map(|i| if kept(i) { 1.0 / keep as f32 } else { 0.0 });
            (prob.product(), Tensor::from_vec(mask.collect(), &[n]))
        })
        .collect()
}
// @end
