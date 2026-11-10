//! Tests for the Chapter 21 challenges.
use ember::data::{MNIST_MEAN, MNIST_STD};
use ember::{
    Adam, ChannelDropout, DropConnect, Dropout, Frozen, Linear, Module, Parameter,
    Residual, Rng, Sequential, Tape, Tensor, Var, eval_is_deterministic, every_mask,
    mc_dropout, toy, train_step,
};

#[path = "../examples/mnist_train.rs"]
#[allow(dead_code)]
mod mnist_train;

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn zeros() -> Tensor {
    Tensor::zeros(&[1, 3])
}

fn round(x: f32, places: i32) -> f32 {
    let s = 10f32.powi(places);
    (x * s).round() / s
}

#[test]
fn predict_dropout_statistics() {
    // @listing predict-test
    let (x, p) = (3.0f32, 0.5f32);
    let (mean, var) = (x, x * x * p / (1.0 - p)); // E = x, Var = x²p/(1 − p)
    assert_eq!((mean, var), (3.0, 9.0));
    let y = Dropout::new(p, 21).mask(&[100_000]).mul_scalar(x).to_vec();
    let m = y.iter().sum::<f32>() / 1e5;
    let v = y.iter().map(|y| (y - m) * (y - m)).sum::<f32>() / 1e5;
    assert_eq!((round(m, 3), round(v, 2)), (3.017, 9.0)); // 100,000 draws
    let mask = Dropout::new(p, 21).mask(&[64, 128]).to_vec();
    let kept = mask.iter().filter(|&&k| k > 0.0).count();
    assert_eq!(kept, 4_117); // expected 4,096 ± 45.25 of 8,192
    // @end
}

#[test]
fn dropout_on_the_pixels() {
    // @listing pixels-test
    let norm = |v: f32| (v - MNIST_MEAN) / MNIST_STD; // a dropped pixel: 0
    let r = |v: f32| round(v, 4);
    assert_eq!((r(norm(0.0)), r(norm(1.0))), (-0.4242, 2.8215)); // background, ink
    assert_eq!((r(norm(0.0) / 0.8), r(norm(1.0) / 0.8)), (-0.5303, 3.5269)); // kept
    let mut model = mnist_train::mlp(&mut Rng::new(20));
    model.layers.insert(0, Box::new(Dropout::new(0.2, 20)));
    let named = model.named_parameters();
    let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["1.weight", "1.bias", "3.weight", "3.bias"]);
    // @end
}

#[test]
fn monte_carlo_dropout() {
    let mut rng = Rng::new(21);
    let train = toy::spirals(50, 1.5, 0.05, &mut rng);
    let mut model = mnist_train::mlp_dropout(&[2, 16, 16, 2], 0.5, &mut rng);
    let mut opt = Adam::new(0.03);
    for _ in 0..100 {
        train_step(&mut model, &train, &mut opt, f32::INFINITY);
    }
    // @listing mc-test
    let x = t(&[0.0, 0.0, 0.5, 0.0, 2.0, 2.0], &[3, 2]); // (2, 2) is far outside
    let (mean, std) = mc_dropout(&mut model, &x, 1000);
    model.train(false);
    let tape = Tape::new();
    let eval = model.forward(tape.constant(x.clone())).value().softmax(1);
    let class0 = |p: &Tensor| -> Vec<f32> {
        p.to_vec().iter().step_by(2).map(|&v| round(v, 3)).collect()
    };
    assert_eq!(class0(&eval), [0.576, 0.564, 0.955]); // P(class 0), dropout off
    assert_eq!(class0(&mean), [0.572, 0.558, 0.808]); // the mean of 1,000 passes
    assert_eq!(class0(&std), [0.042, 0.059, 0.26]); // and their spread
    // @end
}

#[test]
fn drop_connect() {
    // @listing drop-connect-test
    let linear = Linear::from_tensors(Tensor::arange(&[3, 2]), t(&[10., 20.], &[2]));
    let drop = Dropout::new(0.5, 21);
    let mut layer = DropConnect { linear, drop };
    let tape = Tape::new();
    let x = tape.constant(t(&[1., 1., 1.], &[1, 3]));
    let y = layer.forward(x);
    assert_eq!(y.value().to_vec(), [18., 30.]); // W ⊙ M = [[0, 0], [0, 0], [8, 10]]
    y.sum_all().backward_into(&mut layer.parameters());
    let grad = layer.linear.weight.grad.as_ref().unwrap().to_vec();
    assert_eq!(grad, [2., 0., 0., 0., 2., 2.]); // xᵀ·1 ⊙ M: the mask itself
    layer.train(false);
    assert_eq!(layer.forward(x).value().to_vec(), [16., 29.]); // the plain layer
    // @end
}

/// A composite that forgets to pass `train` on to its child.
struct Forgetful(Sequential);

impl Module for Forgetful {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        self.0.forward(x)
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        self.0.parameters()
    }
}

#[test]
fn a_module_that_ignores_train() {
    // @listing deterministic-test
    let mut rng = Rng::new(21);
    let x = Tensor::randn(&[4, 2], &mut rng);
    let net = |rng: &mut Rng| mnist_train::mlp_dropout(&[2, 8, 2], 0.5, rng);
    assert!(eval_is_deterministic(&mut net(&mut rng), &x));
    assert!(!eval_is_deterministic(&mut Forgetful(net(&mut rng)), &x));
    assert!(eval_is_deterministic(&mut Residual(net(&mut rng)), &x));
    assert!(eval_is_deterministic(&mut Frozen(net(&mut rng)), &x));
    // @end
}

#[test]
fn channel_dropout() {
    // @listing channel-test
    let drop = ChannelDropout(Dropout::new(0.5, 21));
    let tape = Tape::new();
    let x = tape.constant(Tensor::arange(&[2, 3, 2, 2]));
    let y = drop.forward(x).value();
    let planes: Vec<Vec<f32>> = y.to_vec().chunks(4).map(|c| c.to_vec()).collect();
    assert_eq!(planes[0], [0., 2., 4., 6.]); // channel (0, 0) kept, doubled
    assert_eq!(planes[1..4], [[0.; 4], [0.; 4], [0.; 4]]); // three dropped whole
    assert_eq!(planes[4..], [[32., 34., 36., 38.], [40., 42., 44., 46.]]);
    // @end
}

#[test]
fn the_average_over_every_mask() {
    // @listing every-mask-test
    let mut rng = Rng::new(21);
    let hidden = Tensor::rand(&[1, 8], &mut rng); // 8 units after a ReLU
    let head = Linear::new(8, 3, &mut rng);
    let logits = |h: Tensor| head.forward(Tape::new().constant(h)).value();
    let (mut z, mut p, mut log_p) = (zeros(), zeros(), zeros());
    for (prob, mask) in every_mask(8, 0.5) {
        let l = logits(hidden.mul(&mask)); // one of 256 thinned networks
        let weighted = |t: Tensor| t.mul_scalar(prob as f32);
        z = z.add(&weighted(l.clone()));
        p = p.add(&weighted(l.softmax(1)));
        log_p = log_p.add(&weighted(l.log_softmax(1)));
    }
    let eval = logits(hidden.clone()); // dropout off: no mask, no scaling
    assert!(z.allclose(&eval, 1e-6)); // the mean of the logits, exactly
    let geometric = log_p.softmax(1); // exp, then normalize to a sum of 1
    assert!(geometric.allclose(&eval.softmax(1), 1e-6)); // exactly, too
    assert_eq!(round(p.get(&[0, 0]), 4), 0.3509); // the arithmetic mean differs:
    assert_eq!(round(eval.softmax(1).get(&[0, 0]), 4), 0.3523); // softmax of eval
    // @end
}
