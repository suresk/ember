//! The examples printed in Chapter 23 of the guided edition: activations that drift,
//! layer normalization composed and fused, batch normalization's running
//! statistics in training and evaluation, buffers in checkpoints, and what
//! normalization buys on the spirals.
use ember::nn::init;
use ember::train::{read_state, to_bytes};
use ember::{
    AdamW, BatchNorm1d, CheckpointError, DataLoader, LayerNorm, Linear, Module,
    Optimizer, ReLU, Rng, Sequential, Sgd, Tape, Tensor, TensorDataset, Tolerance,
    Var, evaluate, gradcheck, load_state_dict, no_decay, state_dict, toy, train_step,
};

#[path = "../examples/mnist_norm.rs"]
#[allow(dead_code)]
mod mnist_norm;

use mnist_norm::{Norm, mlp_norm};

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn near(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

fn close(got: &[f32], want: &[f32], tol: f32) -> bool {
    let ok = got.len() == want.len();
    let ok = ok && got.iter().zip(want).all(|(&g, &w)| near(g, w, tol));
    if !ok {
        println!("got {got:?}\nwant {want:?}");
    }
    ok
}

fn rms(x: &Tensor) -> f32 {
    let v = x.to_vec();
    let sum: f64 = v.iter().map(|&a| (a as f64) * (a as f64)).sum();
    (sum / v.len() as f64).sqrt() as f32
}

// @listing through
/// Ten ReLU layers of width 100 with weights of standard deviation `std`, on 50
/// random inputs, each layer's product normalized first if `norm`: the RMS of
/// every layer's output, and of the input's gradient for `L = Σ y`.
fn through(std: f32, norm: bool) -> (Vec<f32>, f32) {
    let mut rng = Rng::new(23);
    let tape = Tape::new();
    let input = tape.leaf(Tensor::randn(&[50, 100], &mut rng));
    let (mut x, mut sizes) = (input, Vec::new());
    for _ in 0..10 {
        let w = tape.constant(init::normal(&[100, 100], std, &mut rng));
        let z = x.matmul(w);
        let z = if norm { z.normalize(1, 1e-5) } else { z }; // Section 23.4
        x = z.relu();
        sizes.push(rms(&x.value()));
    }
    let grads = x.sum_all().backward();
    (sizes, rms(grads.get(input).unwrap()))
}
// @end

/// Layers 1, 2, 5 and 10 and the gradient, as the section's table prints them.
fn row(std: f32, norm: bool) -> [f32; 5] {
    let (s, g) = through(std, norm);
    [s[0], s[1], s[4], s[9], g]
}

/// Each of `got` within 0.5% of `want`.
fn digits(got: [f32; 5], want: [f32; 5]) -> bool {
    let ok = got
        .iter()
        .zip(want)
        .all(|(&g, w)| (g - w).abs() <= 5e-3 * w.abs());
    if !ok {
        println!("got {got:?}");
    }
    ok
}

/// PyTorch's default variance for 100 inputs, He's, and twice He's.
fn three_scales() -> [f32; 3] {
    let (default, he) = (0.1 / 3f32.sqrt(), init::he_std(100));
    assert!(near(default, 0.0577, 1e-4) && near(he, 0.1414, 1e-4));
    [default, he, 2.0 * he]
}

#[test]
fn drifting_activations() {
    let [default, he, twice] = three_scales();
    assert!(digits(
        row(default, false),
        [0.398, 0.158, 0.0102, 1.12e-4, 3.10e-4]
    ));
    assert!(digits(row(he, false), [0.975, 0.949, 0.900, 0.869, 2.41]));
    assert!(digits(row(twice, false), [1.95, 3.79, 28.8, 890.0, 2464.0]));
}

#[test]
fn normalized_activations() {
    for std in three_scales() {
        // Normalized, the three rows are the same to four digits.
        assert!(digits(
            row(std, true),
            [0.7037, 0.7059, 0.6904, 0.7037, 0.4607]
        ));
    }
}

#[test]
fn normalizing_two_examples() {
    // @listing trace-test
    let x = t(&[1., 2., 3., 6., 12., 14., 16., 22.], &[2, 4]); // row 2 = 2·row 1 + 10
    let (mean, var) = x.mean_var(1);
    assert_eq!([mean.to_vec(), var.to_vec()], [[3., 16.], [3.5, 14.]]);
    let tape = Tape::new();
    let y = tape.constant(x).normalize(1, 1e-5).value();
    let want = [-1.069_043, -0.534_522, 0.0, 1.603_565];
    assert!(close(&y.select(0, 0).to_vec(), &want, 1e-6));
    assert!(close(&y.select(0, 1).to_vec(), &want, 3e-6)); // the same row
    let (mean, var) = y.mean_var(1); // each row now has mean 0, variance 1
    assert!(close(&mean.to_vec(), &[0.0, 0.0], 1e-7));
    assert!(close(&var.to_vec(), &[1.0, 1.0], 3e-6)); // 3.5/3.50001
    // @end
    let (_, var) = t(&[10000., 10001., 10002.], &[3]).mean_var(0);
    assert_eq!(var.to_vec(), [2.0 / 3.0]); // two passes (Challenge 4.4 got 0)
}

// @listing composed
/// `(x - mean) / √(var + eps)` along the last axis, from Chapters 8 and 9's
/// operations: seven nodes, each with a rule the tape already knows.
fn normalize_composed(x: Var<'_>, eps: f32) -> Var<'_> {
    let last = x.shape().len() - 1;
    let d = x - x.mean(last, true); // mean, sub
    let var = (d * d).mean(last, true); // mul, mean: the biased variance
    d / (var + eps).powf(0.5) // add_scalar, powf, div
}
// @end

#[test]
fn composed_and_fused() {
    // @listing fused-test
    let x = t(&[1., 2., 3., 6., 12., 14., 16., 22.], &[2, 4]);
    let w = t(&[1., 2., 3., 4., 5., 6., 7., 8.], &[2, 4]); // L = Σ w·x_hat
    let grad = |f: fn(Var<'_>) -> Var<'_>| {
        let tape = Tape::new();
        let x = tape.leaf(x.clone());
        let y = f(x);
        let nodes = tape.len() - 1; // without the input
        let grads = (y * tape.constant(w.clone())).sum_all().backward();
        (nodes, grads.get(x).unwrap().to_vec())
    };
    let (n_composed, composed) = grad(|x| normalize_composed(x, 1e-5));
    let (n_fused, fused) = grad(|x| x.normalize(1, 1e-5));
    assert_eq!((n_composed, n_fused), (7, 1));
    let row = [-0.190_902, 0.038_179, 0.267_261, -0.114_538]; // (−5, 1, 7, −3)/14σ
    assert!(close(&fused[..4], &row, 1e-6) && close(&composed[..4], &row, 1e-6));
    let half: Vec<f32> = row.iter().map(|g| g / 2.0).collect(); // σ is twice
    assert!(close(&fused[4..], &half, 1e-6));
    // @end
    let mut rng = Rng::new(23);
    let (xr, tol) = (Tensor::randn(&[3, 5], &mut rng), Tolerance::default());
    let check = |f: fn(Var<'_>) -> Var<'_>, x: Tensor| {
        let worst = gradcheck(|v| f(v[0]), &[x], tol);
        assert!(worst.is_ok(), "{worst:?}");
    };
    check(|x| x.normalize(1, 1e-5), xr.clone());
    check(|x| x.normalize(0, 1e-5), xr.clone());
    check(|x| normalize_composed(x, 1e-5), xr);
    check(|x| x.normalize(1, 1e-5), x);
}

#[test]
fn the_layer_norm_module() {
    // @listing layer-norm-test
    let mut ln = LayerNorm::new(4);
    assert_eq!(ln.weight.value.to_vec(), [1., 1., 1., 1.]); // γ = 1, β = 0
    ln.weight.value = t(&[1., 2., 1., 0.5], &[4]);
    ln.bias.value = t(&[0., 0., 10., 0.], &[4]);
    let tape = Tape::new();
    let x = tape.constant(t(&[1., 2., 3., 6., 12., 14., 16., 22.], &[2, 4]));
    let y = ln.forward(x).value();
    let want = [-1.069_043, -1.069_044, 10.0, 0.801_783]; // x_hat · γ + β
    assert!(close(&y.select(0, 0).to_vec(), &want, 1e-6));
    assert_eq!(tape.len(), 6); // x, normalize, γ, mul, β, add
    let names: Vec<String> = ln.named_parameters().into_iter().map(|n| n.0).collect();
    assert_eq!(names, ["weight", "bias"]);
    // @end
    let tol = Tolerance::default();
    let mut rng = Rng::new(23);
    let inputs = [
        Tensor::randn(&[3, 4], &mut rng),
        Tensor::randn(&[4], &mut rng),
    ];
    let affine = gradcheck(|v| v[0].normalize(1, 1e-5) * v[1], &inputs, tol);
    assert!(affine.is_ok());
}

#[test]
fn running_statistics() {
    // @listing running-test
    let mut bn = BatchNorm1d::new(1); // one feature: running mean 0, variance 1
    let tape = Tape::new();
    let batches = [[1., 2., 3., 6.], [2., 4., 6., 8.], [0., 0., 4., 4.]];
    let means = [0.3, 0.77, 0.893];
    let vars = [1.366_667, 1.896_667, 2.240_333];
    for (i, b) in batches.into_iter().enumerate() {
        let y = bn.forward(tape.constant(t(&b, &[4, 1]))).value();
        let r = bn.running();
        let got = [r.mean.item(), r.var.item()];
        assert!(close(&got, &[means[i], vars[i]], 1e-6));
        assert_eq!(r.batches.item(), (i + 1) as f32);
        if i == 2 {
            assert!(close(&y.to_vec(), &[-1., -1., 1., 1.], 2e-6)); // (x − 2)/2
        }
    }
    bn.train(false); // evaluation: the running statistics, and nothing changes
    let x = tape.constant(t(&[1., 2., 3., 6.], &[4, 1])); // batch 1 again
    let y = bn.forward(x).value();
    let want = [0.071_487, 0.739_589, 1.407_691, 3.411_996]; // (x − 0.893)/1.4968
    assert!(close(&y.to_vec(), &want, 1e-5));
    assert_eq!(bn.running().batches.item(), 3.0);
    // @end
    let rest: f32 = 0.9f32.powi(3);
    assert!(near(rest, 0.729, 1e-6));
    assert!(near(0.9f32.powi(50), 0.005_154, 1e-6));
    assert!(near(0.9f32.powi(100), 2.66e-5, 1e-7));
}

#[test]
fn a_batch_of_one() {
    let bn = BatchNorm1d::new(3);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tape = Tape::new();
        bn.forward(tape.constant(t(&[1., 2., 3.], &[1, 3]))).value()
    }));
    let message = *caught.unwrap_err().downcast::<&str>().unwrap();
    assert_eq!(
        message,
        "Expected more than 1 value per channel when training"
    );
    let tape = Tape::new();
    // In evaluation, one example is fine: it doesn't change the statistics.
    let mut bn = BatchNorm1d::new(3);
    bn.train(false);
    let y = bn.forward(tape.constant(t(&[1., 2., 3.], &[1, 3]))).value();
    assert!(close(&y.to_vec(), &[1., 2., 3.], 1e-4)); // (x − 0)/√(1 + ε)
    // Two examples train, and normalize to ±1 whatever they are.
    let mut bn = BatchNorm1d::new(1);
    let y = bn.forward(tape.constant(t(&[5., 7.], &[2, 1]))).value();
    assert!(close(&y.to_vec(), &[-1., 1.], 1e-5));
    assert!(close(&bn.running().var.to_vec(), &[1.1], 1e-6)); // 0.9 + 0.1·2
    let _ = bn.named_buffers();
    // 65 examples in batches of 64 leave a batch of one, unless dropped.
    let ds = TensorDataset::new(Tensor::zeros(&[65, 3]), vec![0; 65]);
    let sizes = |drop| {
        let mut loader = DataLoader::new(&ds, 64).drop_last(drop);
        loader.epoch().map(|b| b.1.len()).collect::<Vec<_>>()
    };
    assert_eq!((sizes(false), sizes(true)), (vec![64, 1], vec![64]));
}

/// A small network with batch normalization: 3 → 4, BatchNorm1d, ReLU, → 2.
fn small(rng: &mut Rng) -> Sequential {
    mlp_norm(&[3, 4, 2], Norm::Batch, rng)
}

#[test]
fn buffers_in_a_checkpoint() {
    // @listing buffers-test
    let mut rng = Rng::new(23);
    let mut model = small(&mut rng); // 3 → 4, BatchNorm1d(4), ReLU, → 2
    let x = Tensor::randn(&[8, 3], &mut rng);
    let batch = (x, vec![0, 1, 0, 1, 1, 0, 0, 1]);
    let mut opt = AdamW::new(0.01);
    for _ in 0..3 {
        train_step(&mut model, &batch, &mut opt, 1.0);
    }
    let state = state_dict(&mut model);
    let names: Vec<&str> = state.keys().map(String::as_str).collect();
    let want = "0.bias 0.weight 1.bias 1.num_batches_tracked 1.running_mean \
                1.running_var 1.weight 3.bias 3.weight";
    assert_eq!(names.join(" "), want);
    assert_eq!(state["1.num_batches_tracked"].item(), 3.0);
    let bytes = to_bytes(&state); // a checkpoint file's bytes
    let mut copy = small(&mut Rng::new(99)); // other weights, fresh statistics
    load_state_dict(&mut copy, &read_state(&bytes).unwrap()).unwrap();
    let tape = Tape::new();
    let outputs = |m: &mut Sequential| {
        m.train(false); // evaluation: the running statistics
        m.forward(tape.constant(batch.0.clone())).value().to_vec()
    };
    assert_eq!(outputs(&mut model), outputs(&mut copy)); // bit for bit
    // @end
    // A checkpoint saved before Chapter 23 has no buffers: an error by name.
    let mut old = state.clone();
    old.retain(|k, _| !k.contains("running") && !k.contains("batches"));
    let err = load_state_dict(&mut copy, &old).unwrap_err();
    let CheckpointError::Keys {
        missing,
        unexpected,
    } = err
    else {
        panic!()
    };
    assert_eq!(
        missing,
        ["1.running_mean", "1.running_var", "1.num_batches_tracked"]
    );
    assert!(unexpected.is_empty());
}

#[test]
fn no_decay_on_gains_and_shifts() {
    let mut rng = Rng::new(23);
    let mut model = mlp_norm(&[784, 256, 256, 10], Norm::Layer, &mut rng);
    // @listing no-decay-test
    let group = no_decay(&mut model, 1e-3);
    let opt = AdamW::new(1e-3).group(group); // group 0 decays, group 1 doesn't
    let sizes = |g: usize, model: &mut Sequential| {
        let ids = &opt.groups()[1].ids;
        let params = model.parameters().into_iter();
        let mine = params.filter(|p| ids.contains(&p.id()) == (g == 1));
        mine.map(|p| p.value.numel()).sum::<usize>()
    };
    assert_eq!(opt.groups()[1].ids.len(), 7); // 3 biases, 2 gains, 2 shifts
    assert_eq!(sizes(1, &mut model), 3 * 256 + 3 * 256 + 10); // 1,546 numbers
    assert_eq!(sizes(0, &mut model), 784 * 256 + 256 * 256 + 256 * 10);
    // @end
}

#[test]
fn the_norm_mlp() {
    let mut rng = Rng::new(23);
    let mut model = mlp_norm(&[784, 256, 256, 10], Norm::Batch, &mut rng);
    let names: Vec<String> = state_dict(&mut model).into_keys().collect();
    assert_eq!(names.len(), 4 + 4 + 6 + 2);
    assert!(names.contains(&"4.running_var".to_string()));
    let mut plain = mlp_norm(&[784, 256, 256, 10], Norm::None, &mut Rng::new(23));
    let numbers =
        |m: &mut Sequential| m.parameters().iter().map(|p| p.value.numel()).sum();
    let n: usize = numbers(&mut plain);
    assert_eq!(numbers(&mut model), n + 4 * 256);
    // The same weights: normalization layers draw no random numbers.
    assert_eq!(
        model.parameters()[0].value.to_vec(),
        plain.parameters()[0].value.to_vec()
    );
    let _ = ReLU;
    let _ = Linear::new(1, 1, &mut rng);
}

// @listing spirals-run
/// Section 14.6's spirals (25 points per arm) and a 2–16–16–16–2 ReLU network
/// with or without batch normalization, 150 full-batch steps of SGD with
/// momentum 0.9 at `lr`: the last training loss, and the accuracy in evaluation.
fn spirals_run(norm: Norm, lr: f32) -> (f32, f32) {
    let mut rng = Rng::new(23);
    let (x, labels) = toy::spirals(25, 1.5, 0.05, &mut rng);
    let mut model = mlp_norm(&[2, 16, 16, 16, 2], norm, &mut rng);
    let mut opt = Sgd::new(lr).momentum(0.9);
    let batch = (x, labels);
    let losses: Vec<f32> = (0..150)
        .map(|_| train_step(&mut model, &batch, &mut opt, f32::INFINITY))
        .collect();
    let data = TensorDataset::new(batch.0, batch.1);
    let (_, accuracy) = evaluate(&mut model, &mut DataLoader::new(&data, 50));
    (losses[149], accuracy)
}
// @end

#[test]
fn normalization_on_the_spirals() {
    let (plain, plain_acc) = spirals_run(Norm::None, 0.3);
    let (batch, batch_acc) = spirals_run(Norm::Batch, 0.3);
    assert!(near(plain, 0.5749, 1e-4) && near(plain_acc, 0.64, 1e-6));
    assert!(batch < 2e-4 && batch_acc == 1.0);
}
