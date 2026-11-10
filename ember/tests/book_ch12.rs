//! The examples printed in Chapter 12 of the guided edition: linear layers,
//! activations and initialization.
use ember::nn::init;
use ember::{
    Linear, Module, Parameter, ReLU, Rng, Sequential, Tanh, Tape, Tensor, Tolerance,
    Var, gradcheck,
};

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

/// `x` is within a fraction `rel` of `want`.
fn near(x: f32, want: f32, rel: f32) -> bool {
    (x - want).abs() <= rel * want.abs()
}

/// The root mean square of the elements, `√(Σx² / n)`: the typical size of one.
fn rms(x: &Tensor) -> f32 {
    let v = x.to_vec();
    let sum: f64 = v.iter().map(|&a| (a as f64) * (a as f64)).sum();
    (sum / v.len() as f64).sqrt() as f32
}

#[test]
fn a_seeded_stream() {
    // @listing rng-example
    let mut rng = Rng::new(42);
    assert_eq!(rng.next_u64(), 0xbdd7_3226_2feb_6e95);
    assert_eq!(rng.next_u64(), 0x28ef_e333_b266_f103);
    let mut copy = rng.clone(); // the same state, so the same future
    assert_eq!(rng.next_u64(), 0x4752_6757_130f_9f52);
    assert_eq!(copy.next_u64(), 0x4752_6757_130f_9f52);
    assert_eq!(Rng::new(42).next_u64(), 0xbdd7_3226_2feb_6e95); // from the top
    assert_eq!(Rng::new(0).next_u64(), 0xe220_a839_7b1d_cdaf); // the reference's
    // @end
}

#[test]
fn uniform_floats() {
    // @listing uniform-example
    let mut rng = Rng::new(42);
    let u: Vec<f32> = (0..3).map(|_| rng.uniform()).collect();
    assert_eq!(u, [0.741_564_87, 0.159_910_38, 0.278_601_1]);
    assert_eq!(0xbdd732 as f32 / 16_777_216.0, u[0]); // the top 24 bits / 2²⁴
    let top = (u64::MAX >> 40) as f32 / 16_777_216.0; // the largest uniform()
    assert_eq!(top, 1.0 - 2f32.powi(-24)); // 0.99999994
    let naive = |bits: u64| bits as f32 / u64::MAX as f32; // all 64 bits
    assert_eq!(naive(u64::MAX - 1000), 1.0); // rounds up to 1
    assert_eq!(naive(u64::MAX - (1 << 39) + 1), 1.0); // so do the top 2³⁹ values
    // @end
}

#[test]
fn normal_samples() {
    // @listing normal-example
    let radius = |u1: f64| (-2.0 * u1.ln()).sqrt();
    assert_eq!(radius(0.0), f64::INFINITY); // why u1 must not be 0
    assert!(near(radius(2f64.powi(-53)) as f32, 8.5717, 1e-4)); // the largest
    let z = Tensor::randn(&[10_000], &mut Rng::new(42));
    assert!(near(z.sum_all() / 10_000.0, -0.006_909, 1e-3)); // mean near 0
    assert!(near(rms(&z), 0.983_84, 1e-4)); // spread near 1
    // @end
    let u = Tensor::rand(&[10_000], &mut Rng::new(7));
    assert!(u.to_vec().iter().all(|&x| (0.0..1.0).contains(&x)));
}

#[test]
fn one_layer() {
    // @listing one-layer
    let mut rng = Rng::new(42);
    let x = Tensor::randn(&[50, 100], &mut rng); // 50 rows of 100 inputs
    assert!(near(rms(&x), 0.986, 1e-3));
    let big = x.matmul(&init::normal(&[100, 100], 1.0, &mut rng));
    assert!(near(rms(&big), 9.85, 1e-3)); // √(100 · 1 · 1) = 10
    let outputs = big.tanh().to_vec();
    let saturated = outputs.iter().filter(|y| y.abs() > 0.99).count();
    assert_eq!(saturated, 3949); // of 5,000: 79%
    let right = x.matmul(&init::normal(&[100, 100], 0.1, &mut rng));
    assert!(near(rms(&right), 0.975, 1e-3)); // √(100 · 0.01 · 1) = 1
    assert!(near(rms(&right.relu()), 0.691, 1e-3)); // 0.975 / √2 = 0.690
    // @end
}

/// An activation, as a function of a `Var` on any tape.
type Act = fn(Var<'_>) -> Var<'_>;
const TANH: Act = |y| y.tanh();
const RELU: Act = |y| y.relu();

// @listing depth
/// The RMS after each layer `act(x·W)`, and of the input's gradient for `Σy`.
fn through_layers(std: f32, act: Act, depth: usize) -> (Vec<f32>, f32) {
    let mut rng = Rng::new(42);
    let tape = Tape::new();
    let input = tape.leaf(Tensor::randn(&[50, 100], &mut rng));
    let (mut x, mut sizes) = (input, Vec::new());
    for _ in 0..depth {
        let w = tape.constant(init::normal(&[100, 100], std, &mut rng));
        x = act(x.matmul(w));
        sizes.push(rms(&x.value()));
    }
    let grads = x.sum_all().backward();
    (sizes, rms(grads.get(input).unwrap()))
}
// @end

/// One row of the section's table: the RMS after layers 1, 2, 5 and 10 of ten, and
/// the gradient's.
fn row(std: f32, act: Act) -> [f32; 5] {
    let (sizes, grad) = through_layers(std, act, 10);
    [sizes[0], sizes[1], sizes[4], sizes[9], grad]
}

/// `got` is `want` to the three digits printed.
fn digits(got: [f32; 5], want: [f32; 5]) -> bool {
    let ok = got.iter().zip(want).all(|(&g, w)| near(g, w, 5e-3));
    if !ok {
        eprintln!("got {got:?}, want {want:?}");
    }
    ok
}

#[test]
fn ten_layers_too_small_and_too_large() {
    // @listing depth-tanh
    let small = [0.0975, 0.00992, 9.65e-6, 9.62e-11, 7.26e-11]; // the table's rows
    assert!(digits(row(0.01, TANH), small));
    let large = [0.960, 0.958, 0.958, 0.959, 4490.0];
    assert!(digits(row(1.0, TANH), large));
    // @end
}

#[test]
fn ten_layers_initialized_to_scale() {
    let xavier = row(init::xavier_std(100, 100), TANH); // 0.1
    assert!(digits(xavier, [0.624, 0.486, 0.3182, 0.230, 0.216]));
    let xavier = row(init::xavier_std(100, 100), RELU);
    assert!(digits(xavier, [0.685, 0.424, 0.127, 0.0255, 0.0648]));
    let he = row(init::he_std(100), RELU); // 0.141
    assert!(digits(he, [0.969, 0.847, 0.720, 0.817, 2.07]));
    let (every, _) = through_layers(init::he_std(100), RELU, 10);
    assert!(every.iter().all(|&s| (0.72..0.97).contains(&s))); // layers 1 to 10
    // PyTorch's default, uniform in ±1/√100: standard deviation 1/√300.
    let torch = row(1.0 / 300f32.sqrt(), TANH);
    assert!(digits(torch, [0.456, 0.250, 0.0455, 0.00291, 0.00229]));
    let torch = row(1.0 / 300f32.sqrt(), RELU);
    assert!(digits(torch, [0.395, 0.141, 0.00817, 1.05e-4, 2.67e-4]));
    // Xavier with PyTorch's gain for tanh, 5/3.
    let gain = row(init::xavier_std(100, 100) * 5.0 / 3.0, TANH);
    assert!(digits(gain, [0.756, 0.691, 0.654, 0.657, 1.88]));
}

#[test]
fn fan_in_and_fan_out() {
    let var = |std: f32| std * std;
    let xavier = var(init::xavier_std(784, 128)); // 2 / 912
    assert!(near(784.0 * xavier, 1.72, 1e-3)); // forward: fan_in · Var(w)
    assert!(near(128.0 * xavier, 0.2807, 1e-3)); // backward: fan_out · Var(w)
    assert!(near(784.0 * var(init::he_std(784)), 2.0, 1e-6)); // ReLU halves it
    let pytorch = var(1.0 / 784f32.sqrt()) / 3.0; // uniform in ±1/√784
    assert!(near(784.0 * pytorch, 1.0 / 3.0, 1e-6)); // a third
}

/// The mean squared error, as in Chapter 11 (Chapter 13 adds it to ember).
fn mse<'t>(pred: Var<'t>, y: Var<'t>) -> Var<'t> {
    (pred - y).powf(2.0).mean_all()
}

/// XOR: four points, and whether exactly one input is 1.
fn xor() -> (Tensor, Tensor) {
    let x = t(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    (x, t(&[0.0, 1.0, 1.0, 0.0], &[4, 1]))
}

/// `steps` of gradient descent on XOR with learning rate 0.5, as in Section 11.6.
fn train(model: &mut Sequential, steps: usize) {
    let (x, y) = xor();
    for _ in 0..steps {
        model.zero_grad();
        let tape = Tape::new();
        let loss = mse(
            model.forward(tape.constant(x.clone())),
            tape.constant(y.clone()),
        );
        loss.backward_into(&mut model.parameters());
        drop(tape);
        for p in model.parameters() {
            let g = p.grad.as_ref().expect("a gradient");
            p.value = p.value.sub(&g.mul_scalar(0.5));
        }
    }
}

// @listing filled
/// Two inputs, three tanh units, one output; every weight `c`, every bias 0.
fn filled(c: f32) -> Sequential {
    let full = |shape: &[usize]| Tensor::zeros(shape).add_scalar(c);
    Sequential::new(vec![
        Box::new(Linear::from_tensors(full(&[2, 3]), Tensor::zeros(&[3]))),
        Box::new(Tanh),
        Box::new(Linear::from_tensors(full(&[3, 1]), Tensor::zeros(&[1]))),
    ])
}
// @end

#[test]
fn zero_and_constant_init() {
    // @listing symmetry
    let mut zero = filled(0.0);
    train(&mut zero, 1);
    let grads: Vec<f32> = zero.parameters().iter().map(|p| grad_rms(p)).collect();
    assert_eq!(grads[..3], [0.0, 0.0, 0.0]); // W₁, b₁ and W₂ get nothing
    assert!(grads[3] > 0.0); // only the output's bias learns
    let mut same = filled(0.5);
    train(&mut same, 100);
    let w1 = same.parameters()[0].value.to_vec(); // [2, 3], one column per unit
    assert_ne!(w1, [0.5; 6]); // it learned...
    assert!(w1[0] == w1[1] && w1[1] == w1[2]); // ...but its three units
    assert!(w1[3] == w1[4] && w1[4] == w1[5]); // are still one unit
    // @end
    // Random weights break the tie.
    let mut rng = Rng::new(42);
    let mut random = Sequential::new(vec![
        Box::new(Linear::new(2, 3, &mut rng)),
        Box::new(Tanh),
        Box::new(Linear::new(3, 1, &mut rng)),
    ]);
    train(&mut random, 100);
    let w1 = random.parameters()[0].value.to_vec();
    assert!(w1[0] != w1[1] && w1[1] != w1[2]);
}

fn grad_rms(p: &Parameter) -> f32 {
    rms(p.grad.as_ref().expect("a gradient"))
}

#[test]
fn pytorch_default_init() {
    // @listing linear-new
    let mut rng = Rng::new(42);
    let layer = Linear::new(784, 128, &mut rng);
    assert_eq!(layer.weight.value.shape(), [784, 128]); // [in, out]
    let largest = |t: &Tensor| t.to_vec().iter().fold(0f32, |m, w| m.max(w.abs()));
    assert!(largest(&layer.weight.value) <= 1.0 / 28.0); // 1/√784 = 0.0357
    assert!(largest(&layer.bias.value) <= 1.0 / 28.0); // the bias too
    assert!(near(rms(&layer.weight.value), 0.020_56, 1e-3)); // 1/√(3 · 784) = 0.0206
    // @end
    assert!(near(largest(&layer.weight.value), 0.035_714, 1e-4));
}

#[test]
fn batches_of_any_rank() {
    // @listing shapes
    let layer = Linear::from_tensors(Tensor::arange(&[3, 2]), t(&[10.0, 20.0], &[2]));
    let tape = Tape::new();
    let one = layer.forward(tape.constant(t(&[1.0, 0.0, -1.0], &[3]))); // [3]
    assert_eq!(one.shape(), [2]);
    assert_eq!(one.value().to_vec(), [6.0, 16.0]);
    let stack = layer.forward(tape.constant(Tensor::arange(&[2, 4, 3]))); // [2, 4, 3]
    assert_eq!(stack.shape(), [2, 4, 2]);
    let second = tape.constant(Tensor::arange(&[2, 4, 3]).select(0, 1).contiguous());
    let alone = layer.forward(second); // matrix 1 of the stack, as a [4, 3] batch
    assert_eq!(alone.value().to_vec(), stack.value().select(0, 1).to_vec());
    assert_eq!(alone.value().to_vec()[..2], [92.0, 141.0]); // row [12, 13, 14]
    // @end
}

#[test]
#[should_panic(expected = "Linear: input [4, 5], weight [3, 2]")]
fn the_wrong_number_of_features() {
    let layer = Linear::from_tensors(Tensor::arange(&[3, 2]), Tensor::zeros(&[2]));
    let tape = Tape::new();
    layer.forward(tape.constant(Tensor::zeros(&[4, 5])));
}

#[test]
fn a_layer_checked_in_one_line() {
    // @listing gradcheck
    let mut rng = Rng::new(42);
    let layer = Linear::new(4, 3, &mut rng);
    let x = Tensor::randn(&[2, 5, 4], &mut rng); // two stacks of five rows
    gradcheck(|v| layer.forward(v[0]).tanh(), &[x], Tolerance::default()).unwrap();
    // @end
}

#[test]
fn the_layer_of_chapter_10() {
    // Section 10.9's tanh(xW + b), as a Linear followed by Tanh.
    let golden = |shape: &[usize], lo: f32, hi: f32| {
        let n: usize = shape.iter().product();
        let frac = |k: usize| (k as f32 * 0.618_034).fract();
        Tensor::from_vec((1..=n).map(|k| lo + (hi - lo) * frac(k)).collect(), shape)
    };
    let (x, w, b) = (
        golden(&[4, 3], -1.0, 1.0),
        golden(&[3, 2], -0.5, 0.5),
        golden(&[2], -0.1, 0.1),
    );
    let tape = Tape::new();
    let want = (tape.constant(x.clone()).matmul(tape.constant(w.clone()))
        + tape.constant(b.clone()))
    .tanh();
    let model =
        Sequential::new(vec![Box::new(Linear::from_tensors(w, b)), Box::new(Tanh)]);
    let got = model.forward(tape.constant(x));
    assert_eq!(got.value().to_vec(), want.value().to_vec());
}

#[test]
fn a_sequential_mlp() {
    // @listing mlp
    let mut rng = Rng::new(42);
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(784, 128, &mut rng)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, &mut rng)),
    ]);
    let count: usize = model.parameters().iter().map(|p| p.value.numel()).sum();
    assert_eq!((count, model.parameters().len()), (101_770, 4)); // Section 11.7's
    let tape = Tape::new();
    let images = tape.constant(Tensor::rand(&[64, 784], &mut rng)); // a fake batch
    let logits = model.forward(images);
    assert_eq!(logits.shape(), [64, 10]); // ten scores for each of 64 images
    assert_eq!(tape.len(), 14); // the batch, 6 nodes per Linear, 1 relu
    // @end
}

/// `P(|z| > x)` for a standard normal, by Simpson's rule on `[x, x + 20]`.
fn two_tails(x: f64) -> f64 {
    let n = 20_000;
    let h = 20.0 / n as f64;
    let f = |t: f64| (-t * t / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let inner: f64 = (1..n)
        .map(|i| f(x + i as f64 * h) * [2.0, 4.0][i % 2])
        .sum();
    2.0 * h / 3.0 * (f(x) + inner + f(x + 20.0))
}

#[test]
fn the_numbers_in_the_prose() {
    // Section 12.2: the 1-in-34-million draw, and the tails Box–Muller cuts off.
    assert_eq!(1u64 << 25, 33_554_432);
    assert_eq!((1u64 << 25) / 100_000, 335); // initializations of 100,000 weights
    let radius = |u1: f64| (-2.0 * u1.ln()).sqrt();
    assert!(near(radius(2f64.powi(-24)) as f32, 5.77, 1e-3));
    assert!(near((1.0 / two_tails(5.768)) as f32, 1.25e8, 0.02));
    assert!(near(two_tails(8.5717) as f32, 1.03e-17, 0.02));
    // Section 12.3: tanh is within 0.01 of ±1 past 2.65, a normal passes 0.265σ 79%.
    assert!(near(0.99f32.atanh(), 2.647, 1e-3));
    assert!(near(two_tails(0.265) as f32, 0.791, 1e-3));
    // Section 12.5: Xavier with ReLU loses √2 a layer; PyTorch's default √6.
    assert!(near(0.685 / 2f32.sqrt().powi(9), 0.0303, 1e-2));
    assert!(near(6f32.sqrt(), 2.45, 1e-3));
    assert!(near(6f32.powi(-5), 1.29e-4, 1e-2));
    // The Rust box: sizes.
    assert_eq!(size_of::<Linear>(), 272); // two 136-byte parameters
    assert_eq!((size_of::<ReLU>(), size_of::<Box<dyn Module>>()), (0, 16));
}
