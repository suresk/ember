//! Tests for the Chapter 9 challenges: `cargo test --test challenges_ch09`.
use ember::{Gradients, Tape, Tensor, Var};

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn grad(grads: &Gradients, v: Var) -> Vec<f32> {
    grads.get(v).expect("a gradient").to_vec()
}

fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(x, w)| (x - w).abs() <= tol)
}

/// The gradient with respect to `x` of `Σ seed · y`.
fn seeded(y: Var, x: Var, seed: &[f32]) -> Vec<f32> {
    let s = y.tape().constant(t(seed, &y.shape()));
    grad(&(y * s).sum_all().backward(), x)
}

#[test]
fn predict_the_reductions() {
    let tape = Tape::new();
    let x = tape.leaf(t(&[3.0, 1.0, 3.0, 0.0, 2.0, 1.0], &[2, 3]));
    let l = x.max(1, false).sum_all() + x.sum(0, false).mean_all();
    assert_eq!(tape.len(), 6);
    assert!(close(&l.value().to_vec(), &[25.0 / 3.0], 1e-6)); // 5 + 10/3
    let (a, b) = (4.0 / 3.0, 1.0 / 3.0);
    assert!(close(&grad(&l.backward(), x), &[a, b, b, b, a, b], 1e-6));
    // A central difference splits the tie in row 0: 1/2 + 1/3 = 5/6 for x00.
    let loss = |x00: f32| {
        let x = t(&[x00, 1.0, 3.0, 0.0, 2.0, 1.0], &[2, 3]);
        x.max(1, false).sum_all() + x.sum(0, false).sum_all() / 3.0
    };
    assert!(close(
        &[(loss(3.25) - loss(2.75)) / 0.5],
        &[5.0 / 6.0],
        1e-6
    ));
}

#[test]
fn squeeze_unsqueeze_flatten() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::arange(&[2, 3]));
    let u = x.unsqueeze(1);
    assert_eq!(u.shape(), [2, 1, 3]);
    assert_eq!(u.squeeze(1).shape(), [2, 3]);
    assert_eq!(u.op(), "reshape");
    let f = x.transpose(0, 1).flatten(); // a copy: the transpose isn't contiguous
    assert_eq!(f.value().to_vec(), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    let seed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(seeded(f, x, &seed), [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    assert_eq!(seeded(u, x, &seed), seed);
}

#[test]
#[should_panic(expected = "squeeze: axis 0 of [2, 3]")]
fn squeeze_wants_size_one() {
    let tape = Tape::new();
    tape.leaf(Tensor::arange(&[2, 3])).squeeze(0);
}

#[test]
fn variance_backward() {
    let tape = Tape::new();
    let xs = t(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &[8]);
    let x = tape.leaf(xs.clone());
    let v = x.var(0, false, 0);
    assert_eq!(v.value().to_vec(), [4.0]);
    assert_eq!(v.value().to_vec(), xs.var(0, false, 0).to_vec());
    let g = grad(&v.backward(), x); // 2(x - 5)/8
    assert_eq!(g, [-0.75, -0.25, -0.25, -0.25, 0.0, 0.0, 0.5, 1.0]);
    let start = tape.len();
    let b = x.var(0, false, 1);
    assert_eq!(tape.len() - start, 5); // mean, sub, mul, sum, div_scalar
    let gb = grad(&b.backward(), x); // 2(x - 5)/7
    let want: Vec<f32> = xs.to_vec().iter().map(|x| 2.0 * (x - 5.0) / 7.0).collect();
    assert!(close(&gb, &want, 1e-6));
}

#[test]
fn cat_backward() {
    let tape = Tape::new();
    let a = tape.leaf(Tensor::arange(&[2, 2]));
    let b = tape.leaf(t(&[10.0, 20.0], &[2, 1]));
    let y = Var::cat(&[a, b], 1);
    assert_eq!(y.value().to_vec(), [0.0, 1.0, 10.0, 2.0, 3.0, 20.0]);
    assert_eq!((y.parents(), y.shape()), (vec![0, 1], vec![2, 3]));
    let s = tape.constant(t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let grads = (y * s).sum_all().backward();
    assert_eq!(grad(&grads, a), [1.0, 2.0, 4.0, 5.0]);
    assert_eq!(grad(&grads, b), [3.0, 6.0]);
    let twice = Var::cat(&[a, a], 0); // [4, 2]
    let seed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(seeded(twice, a, &seed), [6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn repeat_and_expand_backward() {
    let tape = Tape::new();
    let x = tape.leaf(t(&[1.0, 2.0, 3.0], &[3]));
    let r = x.repeat(&[2]);
    assert_eq!(r.value().to_vec(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    let seed = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(seeded(r, x, &seed), [3.0, 5.0, 7.0]); // 0 + 3, 1 + 4, 2 + 5
    let e = x.expand(&[2, 3]);
    assert_eq!(seeded(e, x, &seed), [3.0, 5.0, 7.0]);
    let m = tape.leaf(Tensor::arange(&[2, 2]));
    let r2 = m.repeat(&[2, 3]); // [4, 6]
    let ones = [1.0; 24];
    assert_eq!(seeded(r2, m, &ones), [6.0; 4]); // 2 · 3 copies of each
    let seed24: Vec<f32> = (0..24).map(|i| i as f32).collect();
    // m[0][0] sits at rows 0, 2 and columns 0, 2, 4: 0+2+4 + 12+14+16.
    assert_eq!(seeded(r2, m, &seed24)[0], 48.0);
}

#[test]
fn logsumexp_backward() {
    let tape = Tape::new();
    let x = tape.leaf(t(&[0.0, 2f32.ln(), 3f32.ln()], &[3]));
    let y = x.logsumexp(0, false);
    assert!(close(&y.value().to_vec(), &[6f32.ln()], 1e-6));
    let g = grad(&y.backward(), x);
    assert!(close(&g, &[1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0], 1e-6));
    let big = tape.leaf(t(&[1000.0, 1000.0], &[1, 2]));
    let y = big.logsumexp(1, true);
    assert_eq!(y.shape(), [1, 1]);
    assert!(close(&y.value().to_vec(), &[1000.6931], 1e-3));
    assert_eq!(grad(&y.backward(), big), [0.5, 0.5]);
    let from_y = big.value().sub(&y.value()).exp(); // e^(x - y), reading y
    assert_eq!(from_y.to_vec(), [0.49998546, 0.49998546]);
    let naive = big.exp().sum(1, true).ln();
    assert_eq!(naive.value().to_vec(), [f32::INFINITY]);
    assert!(grad(&naive.backward(), big).iter().all(|g| g.is_nan()));
}

#[test]
fn one_rule_for_every_view() {
    let x =
        Tensor::from_vec((0..24).map(|i| (i * 7 % 24) as f32).collect(), &[2, 3, 4]);
    type View = fn(&Tensor) -> Tensor;
    type Op = for<'t> fn(Var<'t>) -> Var<'t>;
    let pairs: [(View, Op); 6] = [
        (|x| x.expand(&[3, 2, 3, 4]), |v| v.expand(&[3, 2, 3, 4])),
        (|x| x.transpose(0, 2), |v| v.transpose(0, 2)),
        (|x| x.permute(&[1, 2, 0]), |v| v.permute(&[1, 2, 0])),
        (|x| x.slice(2, 1, 4, 2), |v| v.slice(2, 1, 4, 2)),
        (|x| x.select(1, 2), |v| v.select(1, 2)),
        (|x| x.reshape(&[4, 6]), |v| v.reshape(&[4, 6])),
    ];
    for (view, op) in pairs {
        let tape = Tape::new();
        let a = tape.leaf(x.clone());
        let (y1, y2) = (a.as_view("view", view), op(a));
        assert_eq!(y1.value().to_vec(), y2.value().to_vec());
        let n: usize = y1.shape().iter().product();
        let seed: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        assert_eq!(seeded(y1, a, &seed), seeded(y2, a, &seed));
    }
}

#[test]
#[should_panic(expected = "copy: not a view")]
fn as_view_rejects_a_copy() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::arange(&[2, 3]));
    x.as_view("copy", |x| x.transpose(0, 1).contiguous())
        .sum_all()
        .backward();
}
