//! Tests for the Chapter 13 challenges.
#![allow(clippy::approx_constant)] // numbers printed to four places
use ember::{
    Rng, Tape, Tensor, Tolerance, accuracy, bce_with_logits, cross_entropy,
    cross_entropy_masked, cross_entropy_smoothed, gradcheck, nll_loss,
    softmax_jacobian, softmax_t,
};

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn close(x: &Tensor, want: &[f32], tol: f32) -> bool {
    let x = x.to_vec();
    x.len() == want.len() && x.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

#[test]
fn predict_softmax_and_loss() {
    let logits = [1f32.ln(), 2f32.ln(), 3f32.ln(), 4f32.ln()];
    let tape = Tape::new();
    let z = tape.leaf(t(&logits, &[1, 4]));
    assert!(close(&z.value().softmax(1), &[0.1, 0.2, 0.3, 0.4], 1e-6));
    let loss = cross_entropy(z, &[3]);
    assert!((loss.value().item() - 0.9163).abs() < 1e-4); // -ln 0.4
    let grad = loss.backward().get(z).unwrap().clone();
    assert!(close(&grad, &[0.1, 0.2, 0.3, -0.6], 1e-6));
    // Adding 100 to every logit changes nothing.
    let up = tape.leaf(t(&logits, &[1, 4]).add_scalar(100.0));
    assert!((cross_entropy(up, &[3]).value().item() - 0.9163).abs() < 1e-4);
    // Doubling squares the ratios: p ∝ [1, 4, 9, 16].
    let double = tape.leaf(t(&logits, &[1, 4]).mul_scalar(2.0));
    let want = [1.0 / 30.0, 4.0 / 30.0, 9.0 / 30.0, 16.0 / 30.0];
    assert!(close(&double.value().softmax(1), &want, 1e-6));
    assert!((cross_entropy(double, &[3]).value().item() - 0.6286).abs() < 1e-4);
    assert!((cross_entropy(double, &[0]).value().item() - 3.4012).abs() < 1e-4);
}

#[test]
fn nll_after_log_softmax() {
    // @listing nll-test
    let logits = Tensor::randn(&[4, 5], &mut Rng::new(7)).mul_scalar(3.0);
    let targets = [3, 0, 4, 0];
    let tape = Tape::new();
    let z = tape.leaf(logits.clone());
    let fused = cross_entropy(z, &targets);
    let start = tape.len();
    let split = nll_loss(z.log_softmax(1), &targets);
    assert_eq!(tape.len() - start, 2); // log_softmax, nll_loss
    assert!((fused.value().item() - split.value().item()).abs() < 1e-6);
    let (a, b) = (fused.backward(), split.backward());
    assert!(close(a.get(z).unwrap(), &b.get(z).unwrap().to_vec(), 1e-6));
    let tol = Tolerance::default();
    gradcheck(|v| nll_loss(v[0], &targets), &[logits], tol).unwrap();
    // @end
}

#[test]
fn label_smoothing() {
    let tape = Tape::new();
    // Uniform logits: the smoothed loss is still ln 3.
    let uniform = tape.leaf(Tensor::zeros(&[1, 3]));
    let loss = cross_entropy_smoothed(uniform, &[0], 0.1).value().item();
    assert!((loss - 1.0986).abs() < 1e-4);
    // Confident logits: plain cross-entropy goes to 0, the smoothed one grows.
    let ten = tape.leaf(t(&[10.0, 0.0, 0.0], &[1, 3]));
    assert!(cross_entropy(ten, &[0]).value().item() < 1e-4); // 9.1e-5
    let smoothed = cross_entropy_smoothed(ten, &[0], 0.1).value().item();
    assert!((smoothed - 0.6668).abs() < 1e-4);
    let twenty = tape.leaf(t(&[20.0, 0.0, 0.0], &[1, 3]));
    let smoothed = cross_entropy_smoothed(twenty, &[0], 0.1).value().item();
    assert!((smoothed - 1.3333).abs() < 1e-4);
    // The minimum is at p = q = [0.9333, 0.0333, 0.0333], a gap of ln 28.
    let best = tape.leaf(t(&[28f32.ln(), 0.0, 0.0], &[1, 3]));
    let loss = cross_entropy_smoothed(best, &[0], 0.1);
    assert!((loss.value().item() - 0.2911).abs() < 1e-4); // the entropy of q
    let grad = loss.backward().get(best).unwrap().clone();
    assert!(close(&grad, &[0.0, 0.0, 0.0], 1e-6));
    let logits = Tensor::randn(&[3, 4], &mut Rng::new(1));
    let tol = Tolerance::default();
    gradcheck(
        |v| cross_entropy_smoothed(v[0], &[1, 3, 0], 0.2),
        &[logits],
        tol,
    )
    .unwrap();
}

#[test]
fn ignored_rows() {
    let tape = Tape::new();
    let z = tape.leaf(t(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0], &[3, 3]));
    let loss = cross_entropy_masked(z, &[Some(2), Some(1), None]);
    assert!((loss.value().item() - 0.7531).abs() < 1e-4); // Section 13.7's batch
    let grad = loss.backward().get(z).unwrap().to_vec();
    let want = [
        0.0450, 0.1224, -0.1674, 0.1667, -0.3333, 0.1667, 0.0, 0.0, 0.0,
    ];
    assert!(close(&t(&grad, &[9]), &want, 5e-5));
    let none = cross_entropy_masked(z, &[None, None, None]);
    assert!(none.value().item().is_nan()); // 0 / 0, as in PyTorch
    let logits = Tensor::randn(&[4, 3], &mut Rng::new(3));
    let targets = [Some(0), None, Some(2), Some(2)];
    let tol = Tolerance::default();
    gradcheck(|v| cross_entropy_masked(v[0], &targets), &[logits], tol).unwrap();
}

#[test]
fn binary_cross_entropy_with_logits() {
    let tape = Tape::new();
    let x = tape.leaf(t(&[100.0, -100.0, 0.0], &[3]));
    let y = t(&[0.0, 0.0, 1.0], &[3]);
    // Naive: through the probability, σ(100) = 1 and ln(1 - 1) = -∞.
    let p0 = x.value().sigmoid().get(&[0]);
    assert_eq!(p0, 1.0); // σ(100) rounds to 1
    assert_eq!(-(1.0 - p0).ln(), f32::INFINITY); // the y = 0 term, ln(1 - 1)
    let loss = bce_with_logits(x, &y);
    assert!((loss.value().item() - 33.5644).abs() < 1e-3); // (100 + 0 + ln 2) / 3
    let grad = loss.backward().get(x).unwrap().to_vec();
    assert!((grad[0] - 1.0 / 3.0).abs() < 1e-6 && grad[1].abs() < 1e-30);
    assert!((grad[2] + 1.0 / 6.0).abs() < 1e-6); // (0.5 - 1) / 3
    // It is two-class cross-entropy with logits [0, x].
    let z = tape.leaf(t(&[0.0, 3.0], &[1, 2]));
    let two = cross_entropy(z, &[1]).value().item();
    let one = bce_with_logits(tape.leaf(t(&[3.0], &[1])), &t(&[1.0], &[1]));
    assert!((two - one.value().item()).abs() < 1e-6);
    let logits = Tensor::randn(&[2, 3], &mut Rng::new(5)).mul_scalar(4.0);
    let targets = t(&[1.0, 0.0, 0.5, 0.0, 1.0, 1.0], &[2, 3]);
    let tol = Tolerance::default();
    gradcheck(|v| bce_with_logits(v[0], &targets), &[logits], tol).unwrap();
}

#[test]
fn temperature() {
    let logits = t(&[1.0, 2.0, 3.0], &[3]);
    let at = |temp: f32| softmax_t(&logits, temp);
    assert!(close(&at(0.5), &[0.0159, 0.1173, 0.8668], 5e-5));
    assert!(close(&at(1.0), &[0.0900, 0.2447, 0.6652], 5e-5));
    assert!(close(&at(2.0), &[0.1863, 0.3072, 0.5065], 5e-5));
    assert!(close(&at(100.0), &[0.3300, 0.3333, 0.3367], 5e-5));
    assert!(close(&at(0.01), &[0.0, 0.0, 1.0], 1e-30)); // T → 0: the argmax
    // Temperature changes the loss but never the accuracy.
    let batch = t(&[1.0, 2.0, 3.0, 3.0, 2.0, 1.0], &[2, 3]);
    let loss = |temp: f32| {
        let tape = Tape::new();
        let z = tape.constant(batch.mul_scalar(1.0 / temp));
        cross_entropy(z, &[2, 1]).value().item()
    };
    assert!((loss(0.5) - 1.1429).abs() < 1e-4);
    assert!((loss(1.0) - 0.9076).abs() < 1e-4);
    assert!((loss(2.0) - 0.9303).abs() < 1e-4);
    assert_eq!(accuracy(&batch, &[2, 1]), 0.5);
    assert_eq!(accuracy(&batch.mul_scalar(2.0), &[2, 1]), 0.5);
}

#[test]
fn the_jacobian_vector_product() {
    let mut rng = Rng::new(11);
    let x = Tensor::randn(&[6], &mut rng);
    let g = Tensor::randn(&[6], &mut rng);
    let jacobian = softmax_jacobian(&x);
    let rows = jacobian.sum(1, false); // every row sums to 0
    assert!(close(&rows, &[0.0; 6], 1e-6));
    assert!(close(&jacobian, &jacobian.transpose(0, 1).to_vec(), 0.0)); // symmetric
    let full = g.reshape(&[1, 6]).matmul(&jacobian).reshape(&[6]); // gᵀJ
    let tape = Tape::new();
    let leaf = tape.leaf(x);
    let y = leaf.softmax(0);
    let weighted = (y * tape.constant(g)).sum_all();
    let rule = weighted.backward().get(leaf).unwrap().clone();
    assert!(close(&rule, &full.to_vec(), 1e-6));
}
