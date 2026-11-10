//! Chapter 13 challenges.

use crate::autograd::tape::Var;
use crate::nn::one_hot;
use crate::tensor::Tensor;

// @listing nll
/// The mean over the rows of `-log_probs[i, targets[i]]`, one node. Its backward
/// is `-onehot / N`: a gradient only where a target was read.
pub fn nll_loss<'t>(log_probs: Var<'t>, targets: &[usize]) -> Var<'t> {
    let lp = log_probs.value();
    let (n, classes) = (targets.len(), lp.shape()[1]);
    assert_eq!(lp.shape()[0], n, "nll_loss: {n} targets");
    let picked = targets.iter().enumerate();
    let sum: f64 = picked.map(|(i, &c)| lp.get(&[i, c]) as f64).sum();
    let loss = Tensor::from_vec(vec![(-sum / n as f64) as f32], &[]);
    let onehot = one_hot(targets, classes);
    let rule = move |g: &Tensor| vec![onehot.mul_scalar(-g.item() / n as f32)];
    let tape = log_probs.tape();
    tape.push("nll_loss", loss, vec![log_probs.id()], rule)
}
// @end

// @listing smoothed
/// Cross-entropy against the target `q = (1 - eps)·onehot + eps / C`, one node,
/// with backward `(softmax - q) / N`.
pub fn cross_entropy_smoothed<'t>(
    logits: Var<'t>,
    targets: &[usize],
    eps: f32,
) -> Var<'t> {
    let z = logits.value();
    let (n, classes) = (targets.len(), z.shape()[1]);
    let q = one_hot(targets, classes)
        .mul_scalar(1.0 - eps)
        .add_scalar(eps / classes as f32);
    let log_p = z.log_softmax(1);
    let loss = log_p.mul(&q).sum_all() / -(n as f32);
    let loss = Tensor::from_vec(vec![loss], &[]);
    let rule = move |g: &Tensor| {
        let p = log_p.exp();
        vec![p.sub(&q).mul_scalar(g.item() / n as f32)]
    };
    let tape = logits.tape();
    tape.push("cross_entropy", loss, vec![logits.id()], rule)
}
// @end

// @listing masked
/// Cross-entropy over the rows whose target is `Some`, averaged over those rows
/// only; the other rows get a gradient of 0.
pub fn cross_entropy_masked<'t>(
    logits: Var<'t>,
    targets: &[Option<usize>],
) -> Var<'t> {
    let z = logits.value();
    let (rows, classes) = (z.shape()[0], z.shape()[1]);
    assert_eq!(targets.len(), rows, "cross_entropy_masked: targets");
    let kept: Vec<(usize, usize)> = (0..rows)
        .filter_map(|i| targets[i].map(|c| (i, c)))
        .collect();
    let n = kept.len() as f32; // 0 if every row is ignored: the loss is NaN
    let log_p = z.log_softmax(1);
    let sum: f32 = kept.iter().map(|&(i, c)| log_p.get(&[i, c])).sum();
    let loss = Tensor::from_vec(vec![-sum / n], &[]);
    let (mut onehot, mut mask) = (vec![0.0; rows * classes], vec![0.0; rows]);
    for &(i, c) in &kept {
        onehot[i * classes + c] = 1.0;
        mask[i] = 1.0;
    }
    let onehot = Tensor::from_vec(onehot, &[rows, classes]);
    let mask = Tensor::from_vec(mask, &[rows, 1]); // broadcast along each row
    let rule = move |g: &Tensor| {
        let diff = log_p.exp().sub(&onehot).mul(&mask);
        vec![diff.mul_scalar(g.item() / n)]
    };
    let tape = logits.tape();
    tape.push("cross_entropy", loss, vec![logits.id()], rule)
}
// @end

// @listing bce
/// The mean of `-[y ln σ(x) + (1 - y) ln(1 - σ(x))]` over every element, computed
/// as `max(x, 0) - x·y + ln(1 + e^-|x|)`, one node with backward `(σ(x) - y) / n`.
pub fn bce_with_logits<'t>(logits: Var<'t>, targets: &Tensor) -> Var<'t> {
    let x = logits.value();
    assert_eq!(x.shape(), targets.shape(), "bce_with_logits: shapes");
    let n = x.numel() as f32;
    let term = |x: f32, y: f32| x.max(0.0) - x * y + (-x.abs()).exp().ln_1p();
    let loss = x.zip_map(targets, term).sum_all() / n;
    let loss = Tensor::from_vec(vec![loss], &[]);
    let y = targets.clone();
    let rule = move |g: &Tensor| {
        let p = x.sigmoid();
        vec![p.sub(&y).mul_scalar(g.item() / n)]
    };
    let tape = logits.tape();
    tape.push("bce_with_logits", loss, vec![logits.id()], rule)
}
// @end

// @listing temperature
/// Softmax of `logits / temp` along the last axis.
pub fn softmax_t(logits: &Tensor, temp: f32) -> Tensor {
    logits.mul_scalar(1.0 / temp).softmax(logits.ndim() - 1)
}
// @end

// @listing jacobian
/// The full `[C, C]` Jacobian of the softmax of a vector: `diag(p) - p pᵀ`.
pub fn softmax_jacobian(x: &Tensor) -> Tensor {
    let c = x.numel();
    let p = x.softmax(0);
    let column = p.reshape(&[c, 1]);
    let outer = column.matmul(&p.reshape(&[1, c])); // p pᵀ
    let mut diag = vec![0.0; c * c];
    for (i, &pi) in p.to_vec().iter().enumerate() {
        diag[i * c + i] = pi;
    }
    Tensor::from_vec(diag, &[c, c]).sub(&outer)
}
// @end
