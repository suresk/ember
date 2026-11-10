//! Tests for the Chapter 23 challenges.
use ember::{
    BatchNorm1d, BatchNorm2d, LayerNorm, Linear, Module, NormResidual, ReLU, RmsNorm,
    Rng, Sequential, Tape, Tensor, Tolerance, cumulative, fold_batch_norm, gradcheck,
    group_norm, state_dict,
};

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn close(got: &[f32], want: &[f32], tol: f32) -> bool {
    let ok = got.len() == want.len();
    let ok = ok && got.iter().zip(want).all(|(&g, &w)| (g - w).abs() <= tol);
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

#[test]
fn predict_layer_norm_and_running_mean() {
    // @listing predict-test
    let row = t(&[2., 4., 4., 4., 5., 5., 7., 9.], &[1, 8]); // mean 5, σ² 4
    let mut ln = LayerNorm::new(8);
    ln.weight.value = t(&[2.0; 8], &[8]); // γ = 2
    ln.bias.value = t(&[1.0; 8], &[8]); // β = 1
    let tape = Tape::new();
    let y = ln.forward(tape.constant(row)).value().to_vec();
    assert!(close(&y, &[-2., 0., 0., 0., 1., 1., 3., 5.], 1e-5)); // 2x_hat + 1
    let bn = BatchNorm1d::new(1);
    bn.forward(tape.constant(t(&[2., 4., 4., 4.], &[4, 1])));
    bn.forward(tape.constant(t(&[5., 5., 7., 9.], &[4, 1])));
    let r = bn.running();
    let got = [r.mean.item(), r.var.item()];
    assert!(close(&got, &[0.965, 1.266_667], 1e-6));
    // @end
}

#[test]
fn rms_normalization() {
    // @listing rms-test
    let norm = RmsNorm::new(4);
    let tape = Tape::new();
    let x = t(&[1., 2., 3., 6., 11., 12., 13., 16.], &[2, 4]); // row 2 = row 1 + 10
    let y = norm.forward(tape.constant(x.clone())).value();
    let want = [0.282_843, 0.565_685, 0.848_528, 1.697_056]; // x/√12.5
    assert!(close(&y.select(0, 0).to_vec(), &want, 1e-6));
    let shifted = [0.837_526, 0.913_664, 0.989_803, 1.218_219]; // not the same
    assert!(close(&y.select(0, 1).to_vec(), &shifted, 1e-6));
    assert_eq!(tape.len(), 8); // x, mul, mean, add_scalar, powf, div, γ, mul
    // @end
    let tol = Tolerance::default();
    let r = gradcheck(
        |v| {
            let ms = (v[0] * v[0]).mean(1, true);
            v[0] / (ms + f32::EPSILON).powf(0.5)
        },
        &[x],
        tol,
    );
    assert!(r.is_ok());
}

#[test]
fn batch_norm_for_images() {
    // @listing bn2d-test
    let mut bn = BatchNorm2d(BatchNorm1d::new(3));
    let x = Tensor::arange(&[2, 3, 2, 2]); // channel c: 4c..4c+3 and 4c+12..4c+15
    let tape = Tape::new();
    let y = bn.forward(tape.constant(x)).value();
    assert_eq!(y.shape(), [2, 3, 2, 2]);
    let channel0: Vec<f32> = [y.select(1, 0).to_vec()].concat();
    let s = 0.163_846; // 1/√37.25: the 8 values are 0..3 and 12..15
    let want = [-7.5, -6.5, -5.5, -4.5, 4.5, 5.5, 6.5, 7.5].map(|d| d * s);
    assert!(close(&channel0, &want, 1e-5));
    let r = bn.0.running();
    assert!(close(&r.mean.to_vec(), &[0.75, 1.15, 1.55], 1e-6)); // 0.1 · 7.5, …
    assert!(close(&r.var.to_vec(), &[5.157_143; 3], 1e-5)); // 0.9 + 0.1 · 298/7
    let names: Vec<String> = state_dict(&mut bn).into_keys().collect();
    let want = "bias num_batches_tracked running_mean running_var weight";
    assert_eq!(names.join(" "), want);
    // @end
}

#[test]
fn group_normalization() {
    // @listing group-test
    let mut rng = Rng::new(23);
    let x = Tensor::randn(&[2, 4, 3, 3], &mut rng);
    let tape = Tape::new();
    let v = tape.constant(x.clone());
    let same = |a: ember::Var<'_>, b: ember::Var<'_>| {
        close(&a.value().to_vec(), &b.value().to_vec(), 1e-6)
    };
    let rows = |r: usize| v.reshape(&[r, 72 / r]).normalize(1, 1e-5);
    assert!(same(group_norm(v, 1, 1e-5), rows(2))); // one group: layer norm
    assert!(same(group_norm(v, 4, 1e-5), rows(8))); // a group per channel
    let alone = group_norm(tape.constant(x.slice(0, 0, 1, 1)), 2, 1e-5);
    let in_batch = group_norm(v, 2, 1e-5).slice(0, 0, 1, 1);
    assert!(same(alone, in_batch)); // no statistics across the batch
    // @end
    let tol = Tolerance::default();
    let small = Tensor::randn(&[2, 4, 2, 2], &mut rng);
    assert!(gradcheck(|v| group_norm(v[0], 2, 1e-5), &[small], tol).is_ok());
}

#[test]
fn a_cumulative_average() {
    // @listing cumulative-test
    let batches = [[2., 4., 4., 4.], [5., 5., 7., 9.], [0., 0., 4., 4.]];
    let batches: Vec<Tensor> = batches.iter().map(|b| t(b, &[4, 1])).collect();
    let mut bn = BatchNorm1d::new(1);
    cumulative(&mut bn, &batches);
    let r = bn.running();
    let want = [4.0, (1.0 + 11.0 / 3.0 + 16.0 / 3.0) / 3.0]; // (3.5 + 6.5 + 2)/3
    assert!(close(&[r.mean.item(), r.var.item()], &want, 1e-6));
    assert_eq!((r.batches.item(), bn.momentum), (3.0, 1.0 / 3.0));
    // @end
}

#[test]
fn folding_batch_norm() {
    // @listing fold-test
    let mut rng = Rng::new(23);
    let linear = Linear::new(5, 3, &mut rng);
    let mut bn = BatchNorm1d::new(3);
    bn.weight.value = Tensor::randn(&[3], &mut rng);
    bn.bias.value = Tensor::randn(&[3], &mut rng);
    let tape = Tape::new();
    for _ in 0..4 {
        let x = Tensor::randn(&[16, 5], &mut rng);
        let x = x.mul_scalar(3.0).add_scalar(1.0); // mean 1, spread 3
        bn.forward(linear.forward(tape.constant(x))); // running statistics
    }
    bn.train(false);
    let folded = fold_batch_norm(&linear, &bn);
    let x = tape.constant(Tensor::randn(&[8, 5], &mut rng));
    let two = bn.forward(linear.forward(x)).value().to_vec();
    let one = folded.forward(x).value().to_vec();
    assert!(close(&one, &two, 2e-6));
    // @end
}

/// Twelve blocks `f` = Linear·ReLU·Linear of width 64 with layer norm, before
/// or after the sum; the RMS after blocks 1, 3, 6 and 12, and of the input's
/// gradient for Σ w·y, w random.
fn he(rng: &mut Rng) -> Linear {
    let w = ember::nn::init::normal(&[64, 64], ember::nn::init::he_std(64), rng);
    Linear::from_tensors(w, Tensor::zeros(&[64]))
}

fn stack(pre: bool) -> ([f32; 4], f32) {
    let mut rng = Rng::new(23);
    let blocks: Vec<NormResidual<Sequential>> = (0..12)
        .map(|_| NormResidual {
            f: Sequential::new(vec![
                Box::new(he(&mut rng)),
                Box::new(ReLU),
                Box::new(he(&mut rng)),
            ]),
            norm: LayerNorm::new(64),
            pre,
        })
        .collect();
    let tape = Tape::new();
    let input = tape.leaf(Tensor::randn(&[32, 64], &mut rng));
    let mut x = input;
    let mut sizes = vec![];
    for b in &blocks {
        x = b.forward(x);
        sizes.push(rms(&x.value()));
    }
    let w = tape.constant(Tensor::randn(&[32, 64], &mut rng));
    let g = (x * w).sum_all().backward();
    (
        [sizes[0], sizes[2], sizes[5], sizes[11]],
        rms(g.get(input).unwrap()),
    )
}

#[test]
fn pre_norm_and_post_norm() {
    // @listing residual-test
    let (pre, pre_grad) = stack(true); // x + f(norm(x))
    let (post, post_grad) = stack(false); // norm(x + f(x))
    assert!(close(&pre, &[1.859, 2.679, 3.698, 4.968], 1e-3)); // ≈ √(1 + 2k)
    assert!(close(&post, &[1.0; 4], 1e-5));
    assert!(close(&[pre_grad, post_grad], &[4.705, 0.934], 1e-3));
    // @end
}
