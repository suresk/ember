//! Tests for the Chapter 8 challenges: `cargo test --test challenges_ch08`.
use ember::{Gradients, Tape, Tensor, Var, nudge_grad};

fn vector(xs: &[f32]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), &[xs.len()])
}

fn grad(grads: &Gradients, v: Var) -> Vec<f32> {
    grads.get(v).expect("a gradient").to_vec()
}

/// `d sum(y) / dx`.
fn d(y: Var, x: Var) -> Vec<f32> {
    grad(&y.sum_all().backward(), x)
}

fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(x, w)| (x - w).abs() <= tol)
}

#[test]
fn predict_the_gradient() {
    let tape = Tape::new();
    let x = tape.leaf(vector(&[-1.0, 0.0, 2.0]));
    let l = (x * x * x - 3.0 * x + x.relu()).sum_all();
    assert_eq!(tape.len(), 8);
    assert_eq!(l.value().item(), 6.0); // 2 + 0 + 4
    assert_eq!(grad(&l.backward(), x), [0.0, -3.0, 10.0]); // 3x² - 3 + relu'
}

#[test]
fn abs_has_a_corner() {
    let tape = Tape::new();
    let x = tape.leaf(vector(&[-2.0, 0.0, 3.0]));
    let y = x.abs();
    assert_eq!(y.value().to_vec(), [2.0, 0.0, 3.0]);
    assert_eq!(d(y, x), [-1.0, 0.0, 1.0]);
    let h = 1e-3f32;
    assert_eq!((h.abs() - (-h).abs()) / (2.0 * h), 0.0); // the nudge at 0 sees 0
}

#[test]
fn leaky_relu_and_clamp() {
    let tape = Tape::new();
    let x = tape.leaf(vector(&[-2.0, 0.0, 0.5, 1.0, 3.0]));
    let l = x.leaky_relu(0.1);
    assert_eq!(l.value().to_vec(), [-0.2, 0.0, 0.5, 1.0, 3.0]);
    assert_eq!(d(l, x), [0.1, 0.1, 1.0, 1.0, 1.0]); // slope at 0, as in PyTorch
    let c = x.clamp(0.0, 1.0);
    assert_eq!(c.value().to_vec(), [0.0, 0.0, 0.5, 1.0, 1.0]);
    assert_eq!(d(c, x), [0.0, 1.0, 1.0, 1.0, 0.0]); // the ends pass
}

#[test]
fn softplus_stays_finite() {
    let tape = Tape::new();
    let x = tape.leaf(vector(&[-30.0, 0.0, 100.0]));
    let s = x.softplus();
    let ds = d(s, x);
    assert!((s.value().get(&[0]) - 9.357623e-14).abs() < 1e-19);
    assert!(close(
        &s.value().to_vec()[1..],
        &[std::f32::consts::LN_2, 100.0],
        1e-6
    ));
    assert!((ds[0] - 9.357623e-14).abs() < 1e-19); // σ(-30) = e^-30, about
    assert_eq!(ds[1..], [0.5, 1.0]);
    let y = s.value().get(&[0]);
    assert_eq!(1.0 - (-y).exp(), 0.0); // σ(x) = 1 - e^-y, but not in f32
    // Composed from exp and ln: the value at -30 is lost, and at 100 everything is.
    let naive = (x.exp() + 1.0).ln();
    let dn = d(naive, x);
    assert_eq!(
        naive.value().to_vec(),
        [0.0, std::f32::consts::LN_2, f32::INFINITY]
    );
    assert!((dn[0] - 9.357623e-14).abs() < 1e-19); // the gradient survives
    assert!(dn[2].is_nan()); // 1/∞ · ∞
}

#[test]
fn a_user_defined_operation() {
    let tape = Tape::new();
    let x = tape.leaf(vector(&[0.25, 1.0, 4.0]));
    let square = x.custom("square", |x| x.mul(x), |x, _, g| g.mul(x).mul_scalar(2.0));
    assert_eq!(square.value().to_vec(), [0.0625, 1.0, 16.0]);
    assert_eq!((tape.len(), square.op()), (2, "square"));
    assert_eq!(d(square, x), [0.5, 2.0, 8.0]); // 2x
    // 1/√x, whose derivative -x^(-3/2)/2 is -y³/2: the output alone.
    let rsqrt = |x: &Tensor| x.powf(-0.5);
    let r = x.custom("rsqrt", rsqrt, |_, y, g| {
        g.zip_map(y, |g, y| -0.5 * g * y * y * y)
    });
    assert_eq!(r.value().to_vec(), [2.0, 1.0, 0.5]);
    assert_eq!(d(r, x), [-4.0, -0.5, -0.0625]);
    assert_eq!(r.op(), "rsqrt");
}

/// Three rules with a bug each, and the points where a nudge check catches them.
#[test]
fn find_the_bug() {
    fn sigmoid_x(v: Var) -> Var {
        let rule = |x: &Tensor, _: &Tensor, g: &Tensor| {
            g.zip_map(x, |g, x| g * x * (1.0 - x))
        };
        v.custom("sigmoid_x", Tensor::sigmoid, rule)
    }
    fn tanh_1y(v: Var) -> Var {
        v.custom("tanh_1y", Tensor::tanh, |_, y, g| {
            g.zip_map(y, |g, y| g * (1.0 - y))
        })
    }
    fn recip_sign(v: Var) -> Var {
        // 1/x, with the sign of the derivative -1/x² dropped.
        v.custom("recip", |x| x.powf(-1.0), |x, _, g| g.div(&x.mul(x)))
    }
    let checks = |f: fn(Var) -> Var, at: &[f32]| {
        let tape = Tape::new();
        let x = tape.leaf(vector(at));
        let exact = d(f(x), x);
        let nudged = nudge_grad(f, &x.value(), 1e-3);
        exact
            .iter()
            .zip(&nudged)
            .map(|(a, b)| (a - b).abs() < 1e-2)
            .collect::<Vec<_>>()
    };
    assert_eq!(checks(sigmoid_x, &[0.0, 1.0]), [false, false]); // 0 vs 0.25
    assert_eq!(checks(tanh_1y, &[0.0, 1.0]), [true, false]); // 1 - 0 = 1 - 0²
    assert_eq!(checks(recip_sign, &[1.0, -2.0]), [false, false]);
    assert_eq!(checks(recip_sign, &[1e9]), [true]); // ±1e-18: both about 0
}

/// GELU from the operations of Section 8: nine nodes after the input.
fn gelu_composite(x: Var<'_>) -> Var<'_> {
    let k = (2.0f32 / std::f32::consts::PI).sqrt();
    let inner = (x + x * x * x * 0.044715) * k;
    x * 0.5 * (1.0 + inner.tanh())
}

#[test]
fn gelu_by_hand() {
    let at = [-1.0, 0.0, 1.0, 2.0, 3.0];
    let tape = Tape::new();
    let x = tape.leaf(vector(&at));
    let y = x.gelu();
    assert_eq!(tape.len(), 2);
    let want = [-0.158808, 0.0, 0.841192, 1.9545977, 2.9963627];
    assert!(close(&y.value().to_vec(), &want, 1e-6));
    let dy = d(y, x);
    let want = [-0.0829641, 0.5, 1.0829641, 1.0860993, 1.0115843];
    assert!(close(&dy, &want, 1e-6));
    assert!(close(
        &dy,
        &nudge_grad(|v| v.gelu(), &x.value(), 1e-3),
        5e-4
    ));
    let u = (2.0f32 / std::f32::consts::PI).sqrt() * (1.0 + 0.044715); // at x = 1
    assert!((u - 0.8336).abs() < 1e-4 && (u.tanh() - 0.6823).abs() < 1e-4);

    let tape = Tape::new();
    let x = tape.leaf(vector(&at));
    let c = gelu_composite(x);
    assert_eq!(tape.len(), 10);
    assert!(close(&c.value().to_vec(), &y.value().to_vec(), 1e-6));
    assert!(close(&d(c, x), &dy, 1e-6));
}
