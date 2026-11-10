//! Tests for the Chapter 12 challenges.
use ember::nn::init;
use ember::{
    Identity, LeakyReLU, Linear, Module, Parameter, ReLU, Rng, Sequential, Tanh,
    Tape, Tensor, Tolerance, Var, gradcheck_params, he_std_leaky, moments,
    naive_shuffle, orthogonal, truncated_normal,
};
use std::collections::BTreeMap;

fn near(x: f64, want: f64, rel: f64) -> bool {
    (x - want).abs() <= rel * want.abs()
}

fn rms(x: &Tensor) -> f32 {
    let v = x.to_vec();
    let sum: f64 = v.iter().map(|&a| (a as f64) * (a as f64)).sum();
    (sum / v.len() as f64).sqrt() as f32
}

#[test]
fn predict_shapes_and_scales() {
    let mut rng = Rng::new(42);
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(3, 4, &mut rng)),
        Box::new(ReLU),
        Box::new(Linear::new(4, 2, &mut rng)),
    ]);
    let tape = Tape::new();
    let y = model.forward(tape.constant(Tensor::randn(&[5, 7, 3], &mut rng)));
    assert_eq!(y.shape(), [5, 7, 2]);
    assert_eq!(tape.len(), 14); // const, 6 per Linear, relu
    let count: usize = model.parameters().iter().map(|p| p.value.numel()).sum();
    assert_eq!((count, model.parameters().len()), (26, 4));
    // The scales for a 512 → 256 layer.
    let bound = 1.0 / 512f64.sqrt();
    assert!(near(bound, 0.04419, 1e-4));
    assert!(near(bound / 3f64.sqrt(), 0.02551, 1e-3)); // PyTorch default's std
    assert!(near(init::he_std(512) as f64, 0.0625, 1e-6));
    assert!(near(init::xavier_std(512, 256) as f64, 0.05103, 1e-3));
    let layer = Linear::new(512, 256, &mut rng);
    assert!(near(rms(&layer.weight.value) as f64, 0.02551, 1e-2));
}

#[test]
fn samples_have_the_right_moments() {
    // @listing moments-test
    let mut rng = Rng::new(42);
    let u: Vec<f32> = (0..100_000).map(|_| rng.uniform()).collect();
    let (mean, var) = moments(&u);
    assert!(near(mean, 0.5, 3e-3) && near(var, 1.0 / 12.0, 1e-2)); // 0.4989, 0.08285
    let z: Vec<f32> = (0..100_000).map(|_| rng.normal()).collect();
    let (mean, var) = moments(&z);
    assert!(mean.abs() < 0.01 && near(var, 1.0, 1e-2)); // -0.0058, 1.0035
    let tails = z.iter().filter(|x| x.abs() > 2.0).count();
    assert_eq!(tails, 4601); // 4.55% of 100,000 expected: 4,550
    let ones: u32 = (0..100_000).map(|_| rng.next_u64().count_ones()).sum();
    assert!(near(ones as f64 / 6_400_000.0, 0.5, 1e-3)); // half the bits set
    // @end
    assert!(near(moments(&u).0, 0.49893, 1e-4) && near(moments(&z).1, 1.00350, 1e-4));
    assert!(
        near(moments(&u).1, 0.08285, 1e-3) && near(moments(&z).0, -0.00579, 1e-2)
    );
}

/// How often each order of `[0, 1, 2]` comes out of 60,000 shuffles.
fn orders(shuffle: impl Fn(&mut Rng, &mut [usize; 3])) -> Vec<usize> {
    let mut rng = Rng::new(42);
    let mut counts = BTreeMap::new();
    for _ in 0..60_000 {
        let mut xs = [0, 1, 2];
        shuffle(&mut rng, &mut xs);
        *counts.entry(xs).or_insert(0) += 1;
    }
    counts.into_values().collect() // in order: 012, 021, 102, 120, 201, 210
}

#[test]
fn fisher_yates_is_uniform() {
    let counts = orders(|rng, xs| rng.shuffle(xs));
    assert_eq!(counts, [10026, 10098, 9777, 10242, 9913, 9944]);
    assert!(counts.iter().all(|&c| (9_700..10_300).contains(&c))); // 10,000 each
    let mut rng = Rng::new(7);
    assert!((0..1000).all(|_| rng.below(10) < 10));
}

#[test]
fn naive_shuffle_is_biased() {
    let counts = orders(|rng, xs| naive_shuffle(rng, xs));
    assert_eq!(counts, [8884, 11018, 11125, 11154, 8921, 8898]);
    let expected = [4, 5, 5, 5, 4, 4].map(|k| k * 60_000 / 27); // 8,888 and 11,111
    assert!(
        counts
            .iter()
            .zip(expected)
            .all(|(&c, e)| c.abs_diff(e) < 300)
    );
}

/// The RMS after each of ten layers of leaky ReLU with slope 0.5.
fn leaky_layers(std: f32) -> Vec<f32> {
    let mut rng = Rng::new(42);
    let tape = Tape::new();
    let mut x = tape.constant(Tensor::randn(&[50, 100], &mut rng));
    let mut sizes = Vec::new();
    for _ in 0..10 {
        let w = tape.constant(init::normal(&[100, 100], std, &mut rng));
        x = LeakyReLU(0.5).forward(x.matmul(w));
        sizes.push(rms(&x.value()));
    }
    sizes
}

#[test]
fn leaky_relu_he_gain() {
    assert!(near(
        he_std_leaky(100, 0.5) as f64,
        (2.0f64 / 125.0).sqrt(),
        1e-6
    ));
    let plain = leaky_layers(init::he_std(100)); // grows by about √1.25 a layer
    assert!(near(plain[9] as f64, 2.97, 1e-2)); // √1.25¹⁰ = 3.05
    let gained = leaky_layers(he_std_leaky(100, 0.5));
    assert!(gained.iter().all(|&s| (0.89..0.98).contains(&s)));
    assert_eq!(LeakyReLU(0.5).parameters().len(), 0);
}

#[test]
fn building_a_sequential() {
    let mut rng = Rng::new(42);
    let mut model = Sequential::new(vec![])
        .with(Linear::new(3, 4, &mut rng))
        .with(ReLU)
        .with(Identity)
        .with(Linear::new(4, 2, &mut rng));
    assert_eq!(model.layers.len(), 4);
    let count: usize = model.parameters().iter().map(|p| p.value.numel()).sum();
    assert_eq!(count, 26);
    let tape = Tape::new();
    let x = tape.constant(Tensor::arange(&[2, 3]));
    let empty = Sequential::new(vec![]);
    assert_eq!(empty.forward(x).id(), x.id()); // no layers: x itself
    assert_eq!(Identity.forward(x).id(), x.id());
    assert_eq!(model.forward(x).shape(), [2, 2]);
}

#[test]
fn truncated_normal_std() {
    let t = truncated_normal(&[100_000], 0.1, &mut Rng::new(42));
    assert!(t.to_vec().iter().all(|x| x.abs() <= 0.2));
    let (mean, var) = moments(&t.to_vec());
    assert!(mean.abs() < 1e-3);
    assert!(near(var.sqrt(), 0.087_96, 2e-3)); // 0.8796 · 0.1
}

/// A layer whose author put the weight on the tape as a constant.
struct Forgetful(Linear);

impl Module for Forgetful {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let tape = x.tape();
        let w = tape.constant(self.0.weight.value.clone()); // should be param
        x.matmul(w) + tape.param(&self.0.bias)
    }

    fn parameters(&mut self) -> Vec<&mut Parameter> {
        self.0.parameters()
    }
}

#[test]
fn checking_parameters() {
    let mut rng = Rng::new(42);
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(3, 4, &mut rng)),
        Box::new(Tanh),
        Box::new(Linear::new(4, 2, &mut rng)),
    ]);
    let x = Tensor::randn(&[2, 3, 3], &mut rng);
    let before: Vec<Vec<f32>> = model
        .parameters()
        .iter()
        .map(|p| p.value.to_vec())
        .collect();
    let worst = gradcheck_params(&mut model, &x, Tolerance::default()).unwrap();
    assert_eq!((worst.input, worst.index), (0, vec![0, 1])); // W₁[0][1]
    assert!(near(worst.ratio as f64, 0.0113, 1e-2)); // -0.75209 vs -0.75199
    let after: Vec<Vec<f32>> = model
        .parameters()
        .iter()
        .map(|p| p.value.to_vec())
        .collect();
    assert_eq!(before, after); // every parameter put back
    let mut bug = Forgetful(Linear::new(3, 2, &mut rng));
    let rows = Tensor::randn(&[4, 3], &mut rng); // 2-D: Forgetful has no fold
    let worst = gradcheck_params(&mut bug, &rows, Tolerance::default()).unwrap_err();
    assert_eq!((worst.input, worst.analytic), (0, 0.0)); // the weight got nothing
    assert!(near(worst.ratio as f64, 99.2, 1e-2)); // where nudges say -13.03
}

#[test]
fn orthogonal_init() {
    let mut rng = Rng::new(42);
    let q = orthogonal(64, &mut rng);
    let identity = q.transpose(0, 1).matmul(&q); // Qᵀ Q
    for i in 0..64 {
        for j in 0..64 {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((identity.get(&[i, j]) - want).abs() < 1e-6);
        }
    }
    let x = Tensor::randn(&[8, 64], &mut rng);
    let (mut a, mut b) = (x.clone(), x.clone());
    for _ in 0..50 {
        a = a.matmul(&orthogonal(64, &mut rng));
        b = b.matmul(&init::normal(&[64, 64], 0.125, &mut rng)); // 1/√64
    }
    assert!(near(rms(&a) as f64, rms(&x) as f64, 1e-5)); // 0.98548 both
    assert!(near(rms(&b) as f64, 0.517, 1e-2)); // a random walk, halved
}

#[test]
fn the_numbers_in_the_answers() {
    // Challenge 12.2: standard errors.
    let se_uniform = (1.0f64 / 12.0).sqrt() / 100_000f64.sqrt();
    assert!(near(se_uniform, 0.000_913, 1e-3));
    assert!(near((0.5 - 0.498_93) / se_uniform, 1.2, 0.05));
    assert!(near(0.005_79 / (1.0 / 100_000f64.sqrt()), 1.8, 0.05));
    assert!(near((100_000.0f64 * 0.0455 * 0.9545).sqrt(), 66.0, 0.01));
    // Challenge 12.3: Fisher–Yates' spread, and the naive shuffle's 27 paths.
    assert!(near((60_000.0f64 / 6.0 * 5.0 / 6.0).sqrt(), 91.3, 0.01));
    assert_eq!((60_000 * 5 / 27, 60_000 * 4 / 27), (11_111, 8_888));
    // Challenge 12.6: the truncated normal's variance.
    let density = (-2.0f64).exp() / (2.0 * std::f64::consts::PI).sqrt(); // φ(2)
    assert!(near(density, 0.053_99, 1e-3));
    let var = 1.0 - 2.0 * 2.0 * density / 0.9545;
    assert!(near(var, 0.7737, 1e-3) && near(var.sqrt(), 0.8796, 1e-3));
}
