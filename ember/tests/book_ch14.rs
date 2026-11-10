//! The examples printed in Chapter 14 of the guided edition: training an MLP end to
//! end, on XOR and on two spirals.
#![allow(clippy::approx_constant)] // ln 2 to four places
use ember::{
    Linear, Module, Rng, Sequential, Tanh, Tape, Tensor, Tolerance, cross_entropy,
    gradcheck, plot, toy,
};

// The examples are compiled into this test as modules, so the tests run the code
// the chapter prints. Their `main` functions go unused here.
#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;
#[path = "../examples/xor.rs"]
#[allow(dead_code)]
mod xor;

fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

#[test]
fn the_xor_points() {
    let (x, y) = toy::xor();
    assert_eq!(x.to_vec(), [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    assert_eq!(y.to_vec(), [0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn a_line_cannot_fit_xor() {
    // @listing line
    let (x, y) = toy::xor();
    let mut line = Linear::new(2, 1, &mut Rng::new(0));
    let losses: Vec<f32> = (0..1000)
        .map(|_| xor::step(&mut line, &x, &y, 0.5))
        .collect();
    assert!((losses[0] - 1.2228).abs() < 1e-4);
    assert!((losses[999] - 0.25).abs() < 1e-6); // and no lower
    let w = line.weight.value.to_vec();
    assert!(w[0].abs() < 1e-6 && w[1].abs() < 1e-6); // both weights 0
    assert!((line.bias.value.item() - 0.5).abs() < 1e-6); // predict 0.5 for all
    // @end
}

#[test]
fn a_hidden_layer_by_hand() {
    // @listing by-hand
    // Unit 0 is OR, unit 1 is AND, as ±1; the output is (OR - AND) / 2.
    let w1 = Tensor::from_vec(vec![10.0, 10.0, 10.0, 10.0], &[2, 2]);
    let b1 = Tensor::from_vec(vec![-5.0, -15.0], &[2]);
    let w2 = Tensor::from_vec(vec![0.5, -0.5], &[2, 1]);
    let model = Sequential::new(vec![
        Box::new(Linear::from_tensors(w1, b1)),
        Box::new(Tanh),
        Box::new(Linear::from_tensors(w2, Tensor::zeros(&[1]))),
    ]);
    let (x, y) = toy::xor();
    let out = xor::outputs(&model, &x);
    assert!(close(&out, &y.to_vec(), 1e-4));
    // @end
}

#[test]
fn a_hidden_layer_learns_xor() {
    let (x, y) = toy::xor();
    let mut model = xor::mlp(4, &mut Rng::new(0));
    let losses: Vec<f32> = (0..500)
        .map(|_| xor::step(&mut model, &x, &y, 0.1))
        .collect();
    let at = [1, 100, 200, 300, 400, 500].map(|i| losses[i - 1]);
    let want = [0.619676, 0.210592, 0.110705, 0.013136, 0.000421, 0.000009];
    assert!(close(&at, &want, 1e-6));
    let out = xor::outputs(&model, &x);
    assert!(close(&out, &[0.002, 0.996, 0.998, 0.004], 5e-4));
}

#[test]
fn learning_rates_on_xor() {
    // @listing rates
    let (x, y) = toy::xor();
    let run = |lr: f32| -> Vec<f32> {
        let mut model = xor::mlp(4, &mut Rng::new(0)); // the same start every time
        (0..500)
            .map(|_| xor::step(&mut model, &x, &y, lr))
            .collect()
    };
    let (slow, good, fast, spiky) = (run(0.01), run(0.1), run(0.3), run(0.5));
    // @end
    let at = |l: &[f32]| [l[9], l[99]]; // steps 10 and 100
    let peak = |l: &[f32]| (0..500).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap();
    assert!(close(&at(&slow), &[0.4189, 0.2411], 1e-4));
    assert!(close(&at(&good), &[0.2413, 0.2106], 1e-4));
    assert!(close(&at(&fast), &[0.2359, 0.0142], 1e-4));
    assert!(close(&at(&spiky), &[0.2385, 0.1021], 1e-4));
    assert!((slow[499] - 0.2297).abs() < 1e-4 && good[499] < 1e-5);
    assert!(fast[499] < 1e-13 && spiky[499] < 1e-8);
    assert_eq!(
        [&slow, &good, &fast, &spiky].map(|l| peak(l)),
        [0, 0, 127, 66]
    );
    assert!(close(
        &[slow[0], fast[127], spiky[66]],
        &[0.6197, 1.3709, 2.4789],
        1e-4
    ));
    let wild = run(1.0);
    let first = [0.6197, 3.7378, 1.3915, 43.3998, 65.0059, 3349.0];
    assert!(
        close(&wild[..6], &first, 1e-3) && (wild[9] / 1.8612e10 - 1.0).abs() < 1e-4
    );
    assert!(wild[24] == f32::INFINITY && wild[99].is_nan());
}

#[test]
fn two_spirals() {
    let mut rng = Rng::new(14);
    let (x, labels) = toy::spirals(100, 1.5, 0.05, &mut rng);
    assert_eq!(x.shape(), [200, 2]);
    assert!(
        labels[..100].iter().all(|&c| c == 0)
            && labels[100..].iter().all(|&c| c == 1)
    );
    let (again, _) = toy::spirals(100, 1.5, 0.05, &mut Rng::new(14));
    assert_eq!(again.to_vec(), x.to_vec()); // the same seed, the same points
    let (held_out, _) = toy::spirals(100, 1.5, 0.05, &mut rng); // the stream goes on
    assert_ne!(held_out.to_vec(), x.to_vec());
    let radius = |i: usize| x.get(&[i, 0]).hypot(x.get(&[i, 1]));
    let farthest = (0..200).map(radius).fold(0.0, f32::max);
    assert!((farthest - 1.0473).abs() < 1e-4); // r < 1, plus noise
}

#[test]
fn the_first_loss_is_ln_2() {
    // @listing first-loss
    let mut rng = Rng::new(14);
    let (train, _) = spirals::data(&mut rng);
    let mut model = spirals::mlp(16, &mut rng);
    let sizes: Vec<usize> =
        model.parameters().iter().map(|p| p.value.numel()).collect();
    assert_eq!(sizes, [32, 16, 256, 16, 32, 2]); // 354 numbers
    let tape = Tape::new();
    let logits = model.forward(tape.constant(train.0.clone()));
    let loss = cross_entropy(logits, &train.1);
    assert_eq!(tape.len(), 22); // 1 + 3 · 6 + 2 + 1
    assert!((loss.value().item() - 0.6965).abs() < 1e-4);
    assert!((2f32.ln() - 0.6931).abs() < 1e-4); // ln C, for C = 2 classes
    // @end
}

#[test]
fn the_parameters_of_three_widths() {
    let count = |width| {
        let mut model = spirals::mlp(width, &mut Rng::new(14));
        model
            .parameters()
            .iter()
            .map(|p| p.value.numel())
            .sum::<usize>()
    };
    assert_eq!([4, 16, 64].map(count), [42, 354, 4482]);
}

#[test]
fn a_short_spirals_run() {
    let mut rng = Rng::new(14);
    let small = toy::spirals(25, 1.5, 0.05, &mut rng); // 50 points
    let mut model = spirals::mlp(16, &mut rng);
    let losses: Vec<f32> = (0..750)
        .map(|_| spirals::step(&mut model, &small, 2.0))
        .collect();
    let at = [1, 250, 500, 750].map(|i| losses[i - 1]);
    assert!(close(&at, &[0.6940, 0.4656, 0.1503, 0.0076], 1e-4));
    assert_eq!(spirals::accuracy(&model, &small), 1.0);
}

#[test]
fn overfit_a_tiny_batch() {
    // @listing tiny
    let mut rng = Rng::new(14);
    let ((x, labels), _) = spirals::data(&mut rng);
    let x = x.slice(0, 0, 200, 25).contiguous(); // 8 points, 4 of each arm
    let labels: Vec<usize> = labels.iter().step_by(25).copied().collect();
    let tiny = (x, labels);
    let mut model = spirals::mlp(16, &mut rng);
    let losses: Vec<f32> = (0..300)
        .map(|_| spirals::step(&mut model, &tiny, 1.0))
        .collect();
    assert!((losses[0] - 0.6927).abs() < 1e-4); // ln 2 again
    assert_eq!(losses.iter().position(|&l| l < 0.01), Some(295)); // step 296
    assert!((losses[99] - 0.2794).abs() < 1e-4 && losses[299] < 0.01);
    // @end
}

#[test]
fn gradcheck_the_whole_model() {
    // @listing gradcheck
    let mut rng = Rng::new(14);
    let ((x, labels), _) = spirals::data(&mut rng);
    let model = spirals::mlp(16, &mut rng);
    let x = x.slice(0, 0, 200, 25).contiguous(); // 8 points, 4 of each arm
    let labels: Vec<usize> = labels.iter().step_by(25).copied().collect();
    let tol = Tolerance::default();
    let worst = gradcheck(|v| cross_entropy(model.forward(v[0]), &labels), &[x], tol);
    assert!((worst.unwrap().ratio - 0.0030).abs() < 1e-4); // 0.3% of the tolerance
    // @end
}

#[test]
fn text_plots() {
    let curve = plot::curve(&[0.8, 0.4, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05], 8, 4);
    let want = [
        "  0.8000 |*       ",
        "         | *      ",
        "         |  **    ",
        "  0.0500 |    ****",
        "         +--------",
        "         0       8",
    ];
    assert_eq!(curve.lines().collect::<Vec<_>>(), want);
    // @listing plots
    let points = Tensor::from_vec(vec![0.5, 0.5, -0.5, -0.5, 0.6, -0.6], &[3, 2]);
    let right = |grid: &Tensor| {
        (0..grid.shape()[0])
            .map(|i| (grid.get(&[i, 0]) > 0.0) as usize)
            .collect()
    };
    let map = plot::regions(right, 1.0, (8, 4), (&points, &[1, 0, 0]));
    let want = ["....    ", "....  x ", "....    ", "..o.  o "];
    assert_eq!(map.lines().collect::<Vec<_>>(), want);
    // @end
}
