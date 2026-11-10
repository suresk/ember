//! Chapter 13: the loss functions. `cross_entropy` for classes, recorded as one node
//! with the fused backward `(softmax - onehot) / N`, and `mse_loss` for numbers.

use crate::autograd::tape::Var;
use crate::tensor::Tensor;

// @listing check-targets
/// Panics unless `logits` is `[N, C]` and `targets` holds `N` classes below `C`.
fn check_targets(logits: &Tensor, targets: &[usize]) {
    let shape = logits.shape();
    assert!(shape.len() == 2, "cross_entropy: logits {shape:?}");
    let (n, classes) = (shape[0], shape[1]);
    let count = targets.len();
    assert!(count == n, "cross_entropy: {count} targets for {n} rows");
    for &c in targets {
        assert!(c < classes, "cross_entropy: target {c}, {classes} classes");
    }
}

/// A `[N, classes]` tensor with a 1 at `[i, targets[i]]` and 0 everywhere else.
pub fn one_hot(targets: &[usize], classes: usize) -> Tensor {
    let mut data = vec![0.0; targets.len() * classes];
    for (row, &c) in targets.iter().enumerate() {
        data[row * classes + c] = 1.0;
    }
    Tensor::from_vec(data, &[targets.len(), classes])
}
// @end

// @listing cross-entropy
/// The mean over the rows of `-ln softmax(logits)[target]`, as one tape node whose
/// backward is `(softmax(logits) - onehot) / N`.
pub fn cross_entropy<'t>(logits: Var<'t>, targets: &[usize]) -> Var<'t> {
    let z = logits.value();
    check_targets(&z, targets);
    let n = targets.len();
    let log_p = z.log_softmax(1); // [N, C], every entry finite and at most 0
    let picked = targets.iter().enumerate();
    let sum: f64 = picked.map(|(i, &c)| log_p.get(&[i, c]) as f64).sum();
    let loss = Tensor::from_vec(vec![(-sum / n as f64) as f32], &[]);
    let onehot = one_hot(targets, z.shape()[1]);
    let rule = move |g: &Tensor| {
        let p = log_p.exp(); // the softmax, from the stored log-probabilities
        vec![p.sub(&onehot).mul_scalar(g.item() / n as f32)]
    };
    let tape = logits.tape();
    tape.push("cross_entropy", loss, vec![logits.id()], rule)
}
// @end

// @listing mse
/// The mean of `(pred - target)²` over every element. The shapes must be equal:
/// broadcasting a `[N]` target against a `[N, 1]` prediction would compare every
/// prediction with every target.
pub fn mse_loss<'t>(pred: Var<'t>, target: Var<'t>) -> Var<'t> {
    let (p, t) = (pred.shape(), target.shape());
    assert!(p == t, "mse_loss: prediction {p:?}, target {t:?}");
    (pred - target).powf(2.0).mean_all()
}
// @end
