//! The examples printed in Chapter 13 of the guided edition: softmax, log-sum-exp and
//! cross-entropy.
#![allow(clippy::approx_constant)] // the printed numbers are ln 2, ln 10 to 4 places
use ember::{
    Linear, Module, ReLU, Rng, Sequential, Tape, Tensor, Tolerance, Var,
    cross_entropy, gradcheck, mse_loss, one_hot,
};

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

/// Every element of `x` is within `tol` of `want`.
fn close(x: &Tensor, want: &[f32], tol: f32) -> bool {
    let x = x.to_vec();
    x.len() == want.len() && x.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

// @listing naive
/// Softmax straight from the definition, `e^x / Σ e^x`, along the last axis.
fn naive_softmax(x: &Tensor) -> Tensor {
    let e = x.exp();
    e.div(&e.sum(x.ndim() - 1, true))
}
// @end

#[test]
fn softmax_from_the_definition() {
    // @listing naive-example
    let p = naive_softmax(&t(&[1.0, 2.0, 3.0], &[3]));
    assert!(close(&p, &[0.0900, 0.2447, 0.6652], 5e-5));
    assert!((p.sum_all() - 1.0).abs() < 1e-6);
    let shifted = naive_softmax(&t(&[11.0, 12.0, 13.0], &[3]));
    assert!(close(&shifted, &[0.0900, 0.2447, 0.6652], 5e-5)); // the same
    assert!((f32::MAX.ln() - 88.7228).abs() < 1e-4); // the largest exponent
    assert!(88.7f32.exp().is_finite()); // 3.3e38, near f32::MAX
    assert_eq!(89f32.exp(), f32::INFINITY);
    let big = naive_softmax(&t(&[1000.0, 1001.0], &[2])); // inf / inf
    assert!(big.to_vec().iter().all(|p| p.is_nan()));
    // @end
}

#[test]
fn log_sum_exp() {
    // @listing lse-example
    let big = t(&[1000.0, 1001.0], &[2]);
    assert_eq!(big.exp().sum(0, false).ln().item(), f32::INFINITY); // naive
    assert!((big.logsumexp(0, false).item() - 1001.3133).abs() < 1e-4);
    let small = t(&[-1000.0, -1001.0], &[2]);
    let naive = small.exp().sum(0, false).ln().item(); // ln(0 + 0)
    assert_eq!(naive, f32::NEG_INFINITY);
    assert!((small.logsumexp(0, false).item() + 999.6867).abs() < 1e-4);
    let rows = t(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3]);
    let lse = rows.logsumexp(1, true); // one per row, the axis kept
    assert_eq!(lse.shape(), [2, 1]);
    assert!(close(&lse, &[3.4076, 1.0986], 1e-4)); // the second is ln 3
    // @end
}

#[test]
fn summing_in_f64() {
    // @listing vocabulary
    // One logit of 0 and 100,000 of ln(1e-8): ln(1 + 100,000 · 1e-8) = 0.0009995.
    let mut logits = vec![(1e-8f32).ln(); 100_001];
    logits[0] = 0.0;
    let lse = t(&logits, &[100_001]).logsumexp(0, false).item();
    assert!((lse - 0.0009995).abs() < 1e-7);
    let in_f32: f32 = logits.iter().map(|x| x.exp()).sum(); // 1 + 1e-8 rounds to 1
    assert_eq!(in_f32.ln(), 0.0);
    // @end
}

#[test]
fn the_log_of_a_softmax() {
    // @listing log-softmax-example
    let z = t(&[0.0, 200.0], &[2]);
    assert!(((-200f64).exp() / 1.383e-87 - 1.0).abs() < 1e-3); // in f64
    assert_eq!(z.softmax(0).to_vec(), [0.0, 1.0]); // e^-200 underflows in f32
    assert_eq!(f32::from_bits(1), 1.4e-45); // the smallest positive f32
    assert_eq!(z.softmax(0).ln().to_vec(), [f32::NEG_INFINITY, 0.0]);
    assert_eq!(z.log_softmax(0).to_vec(), [-200.0, 0.0]);
    let tie = t(&[1000.0, 1000.0], &[2]);
    let subtracted = tie.sub(&tie.logsumexp(0, true)); // x - (m + ln s)
    assert!((subtracted.get(&[0]) + 0.693_176).abs() < 1e-6); // 1000.6931 rounded
    assert_eq!(tie.log_softmax(0).to_vec(), [-0.693_147_2; 2]); // (x - m) - ln s
    // @end
}

#[test]
fn the_softmax_jacobian() {
    // @listing jacobian
    let p = t(&[1.0, 2.0, 3.0], &[3]).softmax(0).to_vec();
    let jacobian: Vec<f32> = (0..9)
        .map(|k| {
            let (i, j) = (k / 3, k % 3);
            let diagonal = if i == j { p[i] } else { 0.0 };
            diagonal - p[i] * p[j] // ∂p_i/∂x_j
        })
        .collect();
    let want = [
        [0.0819, -0.0220, -0.0599],
        [-0.0220, 0.1848, -0.1628],
        [-0.0599, -0.1628, 0.2227],
    ];
    assert!(close(&t(&jacobian, &[3, 3]), want.as_flattened(), 5e-5));
    // The backward pass for g = [1, 0, 0] is row 0, without building the matrix.
    let tape = Tape::new();
    let x = tape.leaf(t(&[1.0, 2.0, 3.0], &[3]));
    let p0 = x.softmax(0).select(0, 0); // the gradient of p_0 alone
    let grads = p0.backward();
    assert!(close(grads.get(x).unwrap(), &want[0], 5e-5));
    // @end
}

#[test]
fn softmax_rules_pass_gradcheck() {
    // @listing softmax-gradcheck
    let x = [Tensor::randn(&[2, 3, 4], &mut Rng::new(42)).mul_scalar(3.0)];
    let tol = Tolerance::default();
    for dim in 0..3 {
        gradcheck(|v| v[0].softmax(dim), &x, tol).unwrap();
        gradcheck(|v| v[0].log_softmax(dim), &x, tol).unwrap();
    }
    // @end
}

#[test]
fn a_batch_traced() {
    // @listing batch
    let tape = Tape::new();
    let logits = tape.leaf(t(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3]));
    let loss = cross_entropy(logits, &[2, 1]);
    assert_eq!(tape.len(), 2); // the leaf and one node
    assert!((loss.value().item() - 0.7531).abs() < 1e-4); // (0.4076 + 1.0986) / 2
    let grads = loss.backward();
    let p = logits.value().softmax(1);
    let want = p.sub(&one_hot(&[2, 1], 3)).mul_scalar(0.5); // (p - onehot) / N
    assert_eq!(grads.get(logits).unwrap().to_vec(), want.to_vec());
    let printed = [0.0450, 0.1224, -0.1674, 0.1667, -0.3333, 0.1667];
    assert!(close(&want, &printed, 5e-5));
    // @end
}

#[test]
fn cross_entropy_passes_gradcheck() {
    // @listing ce-gradcheck
    let logits = Tensor::randn(&[4, 5], &mut Rng::new(7)).mul_scalar(3.0);
    let targets = [3, 0, 4, 0];
    let tol = Tolerance::default();
    gradcheck(|v| cross_entropy(v[0], &targets), &[logits], tol).unwrap();
    // @end
}

// @listing composed
/// Cross-entropy from separate nodes: `-Σ onehot ⊙ ln(softmax(z)) / N`.
fn composed<'t>(logits: Var<'t>, targets: &[usize]) -> Var<'t> {
    let n = targets.len() as f32;
    let onehot = logits.tape().constant(one_hot(targets, logits.shape()[1]));
    let log_p = logits.softmax(1).ln();
    (log_p * onehot).sum_all() * (-1.0 / n)
}
// @end

#[test]
fn fused_against_composed() {
    // @listing fused-vs-composed
    let tape = Tape::new();
    let z = tape.leaf(t(&[0.0, 200.0], &[1, 2])); // confidently wrong
    let fused = cross_entropy(z, &[0]);
    assert_eq!(fused.value().item(), 200.0);
    assert_eq!(fused.backward().get(z).unwrap().to_vec(), [-1.0, 1.0]);
    let start = tape.len();
    let naive = composed(z, &[0]);
    assert_eq!(tape.len() - start, 6); // const, softmax, ln, mul, sum, scale
    assert_eq!(naive.value().item(), f32::INFINITY);
    let g = naive.backward().get(z).unwrap().to_vec();
    assert!(g.iter().all(|x| x.is_nan()));
    // @end
}

#[test]
fn the_gradients_agree_when_nothing_overflows() {
    let tape = Tape::new();
    let z = tape.leaf(t(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3]));
    let a = cross_entropy(z, &[2, 1]);
    let b = composed(z, &[2, 1]);
    assert!((a.value().item() - b.value().item()).abs() < 1e-6);
    let (ga, gb) = (a.backward(), b.backward());
    assert!(close(
        ga.get(z).unwrap(),
        &gb.get(z).unwrap().to_vec(),
        1e-6
    ));
}

#[test]
fn the_loss_of_a_guess() {
    // @listing ln-c
    let mut rng = Rng::new(42);
    let model = Sequential::new(vec![
        Box::new(Linear::new(784, 128, &mut rng)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, &mut rng)),
    ]);
    let images = Tensor::rand(&[64, 784], &mut rng);
    let labels: Vec<usize> = (0..64).map(|i| i % 10).collect();
    let tape = Tape::new();
    let loss = cross_entropy(model.forward(tape.constant(images)), &labels);
    assert!((10f32.ln() - 2.3026).abs() < 1e-4); // a uniform guess's loss
    assert!((loss.value().item() - 2.3234).abs() < 1e-4); // 0.9% above it
    // @end
}

// @listing targets
#[test]
#[should_panic(expected = "cross_entropy: 1 targets for 2 rows")]
fn a_target_per_row() {
    let tape = Tape::new();
    cross_entropy(tape.leaf(Tensor::zeros(&[2, 3])), &[0]);
}

#[test]
#[should_panic(expected = "cross_entropy: target 3, 3 classes")]
fn targets_below_the_class_count() {
    let tape = Tape::new();
    cross_entropy(tape.leaf(Tensor::zeros(&[2, 3])), &[0, 3]);
}
// @end

#[test]
fn mean_squared_error() {
    // @listing mse-example
    let tape = Tape::new();
    let pred = tape.leaf(t(&[0.0, 0.0], &[2, 1]));
    let loss = mse_loss(pred, tape.constant(t(&[3.0, 5.0], &[2, 1])));
    assert_eq!(loss.value().item(), 17.0); // Section 11.4's first step
    assert_eq!(loss.backward().get(pred).unwrap().to_vec(), [-3.0, -5.0]);
    // A [3, 1] prediction against a [3] target broadcasts to [3, 3]:
    let p = tape.constant(t(&[1.0, 2.0, 3.0], &[3, 1]));
    let y = tape.constant(t(&[1.0, 2.0, 3.0], &[3]));
    let wrong = (p - y).powf(2.0).mean_all(); // every p against every y
    assert!((wrong.value().item() - 12.0 / 9.0).abs() < 1e-6); // not 0
    // @end
}

// @listing mse-panic
#[test]
#[should_panic(expected = "mse_loss: prediction [3, 1], target [3]")]
fn mse_loss_wants_equal_shapes() {
    let tape = Tape::new();
    let p = tape.constant(t(&[1.0, 2.0, 3.0], &[3, 1]));
    mse_loss(p, tape.constant(t(&[1.0, 2.0, 3.0], &[3])));
}
// @end

// @listing classifier
/// One step of gradient descent on the cross-entropy; returns the loss before it.
fn step(model: &mut Linear, x: &Tensor, labels: &[usize], lr: f32) -> f32 {
    model.zero_grad();
    let loss = {
        let tape = Tape::new();
        let loss = cross_entropy(model.forward(tape.constant(x.clone())), labels);
        loss.backward_into(&mut model.parameters());
        loss.value().item()
    };
    for p in model.parameters() {
        let g = p.grad.as_ref().expect("a gradient");
        p.value = p.value.sub(&g.mul_scalar(lr));
    }
    loss
}
// @end

#[test]
fn a_tiny_classifier() {
    // @listing classifier-example
    let xy = [
        2.0, 0.0, 1.5, 0.5, -1.0, 1.5, -1.5, 1.0, -1.0, -1.5, 0.0, -2.0,
    ];
    let (x, labels) = (t(&xy, &[6, 2]), [0, 0, 1, 1, 2, 2]); // two per class
    let mut model = Linear::new(2, 3, &mut Rng::new(42));
    let losses: Vec<f32> =
        (0..5).map(|_| step(&mut model, &x, &labels, 0.5)).collect();
    let want = [1.5977, 0.9441, 0.5652, 0.3689, 0.2673];
    assert!(close(&t(&losses, &[5]), &want, 1e-4));
    let tape = Tape::new();
    let logits = model.forward(tape.constant(x)).value();
    assert_eq!(logits.argmax(1), labels); // every point in its class
    // @end
}
