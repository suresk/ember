//! Tests for the Chapter 10 challenges: `cargo test --test challenges_ch10`.
use ember::{Gradients, Tape, Tensor, Tolerance, Var, directional_check, gradcheck};

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn grad(grads: &Gradients, v: Var) -> Vec<f32> {
    grads.get(v).expect("a gradient").to_vec()
}

fn golden(shape: &[usize], lo: f32, hi: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let frac = |k: usize| (k as f32 * 0.618_034).fract();
    Tensor::from_vec((1..=n).map(|k| lo + (hi - lo) * frac(k)).collect(), shape)
}

#[test]
fn predict_matmul_gradients() {
    let tape = Tape::new();
    let a = tape.leaf(Tensor::arange(&[2, 3]));
    let b = tape.leaf(Tensor::arange(&[3, 4]));
    let grads = a.matmul(b).sum_all().backward();
    assert_eq!(grads.get(a).unwrap().shape(), [2, 3]);
    assert_eq!(grad(&grads, a), [6.0, 22.0, 38.0, 6.0, 22.0, 38.0]); // rows of B
    let col = [3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0];
    assert_eq!(grads.get(b).unwrap().shape(), [3, 4]);
    assert_eq!(grad(&grads, b), col); // columns of A, summed
    let x = tape.leaf(Tensor::arange(&[2, 3]));
    let gram = x.matmul(x.transpose(0, 1)).sum_all().backward();
    assert_eq!(grad(&gram, x), [6.0, 10.0, 14.0, 6.0, 10.0, 14.0]);
}

#[test]
fn matvec_backward() {
    let tape = Tape::new();
    let a = tape.leaf(Tensor::arange(&[2, 3]));
    let v = tape.leaf(t(&[1.0, 0.0, -1.0], &[3]));
    let y = a.matvec(v);
    assert_eq!(y.value().to_vec(), [-2.0, -2.0]);
    let grads = (y * tape.constant(t(&[1.0, 2.0], &[2])))
        .sum_all()
        .backward();
    assert_eq!(grad(&grads, a), [1.0, 0.0, -1.0, 2.0, 0.0, -2.0]);
    assert_eq!(grad(&grads, v), [6.0, 9.0, 12.0]);
    let (m, v) = (golden(&[3, 4], -1.0, 1.0), golden(&[4], -1.0, 1.0));
    let r = gradcheck(|x| x[0].matvec(x[1]), &[m, v], Tolerance::default());
    assert!(r.is_ok(), "{r:?}");
}

#[test]
fn quadratic_form_gradient() {
    // @listing quadratic
    let tape = Tape::new();
    let a = tape.constant(t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let x = tape.leaf(t(&[1.0, -1.0], &[2]));
    let row = x.reshape(&[1, 2]);
    let q = row.matmul(a).matmul(row.transpose(0, 1)).sum_all(); // xᵀ A x
    assert_eq!(q.value().item(), 0.0); // 1 - 2 - 3 + 4
    assert_eq!(grad(&q.backward(), x), [-3.0, -3.0]); // (A + Aᵀ) x
    // @end
    let r = gradcheck(
        |v| {
            let row = v[0].reshape(&[1, 3]);
            row.matmul(v[1]).matmul(row.transpose(0, 1))
        },
        &[golden(&[3], -1.0, 1.0), golden(&[3, 3], -1.0, 1.0)],
        Tolerance::default(),
    );
    assert!(r.is_ok(), "{r:?}");
}

#[test]
fn allclose_rel() {
    let (a, b) = (t(&[1.0, 1000.0], &[2]), t(&[1.001, 1001.0], &[2]));
    assert!(!a.allclose_rel(&b, 1e-3, 0.0)); // 1000 vs 1001: off by 1
    assert!(a.allclose_rel(&b, 1e-3, 1e-3)); // 1 ≤ 0.001 + 1.001
    let (one, big) = (t(&[1.0], &[1]), t(&[1.1], &[1]));
    assert!(one.allclose_rel(&big, 0.0, 0.0913)); // 0.1 ≤ 0.1004
    assert!(!big.allclose_rel(&one, 0.0, 0.0913)); // 0.1 > 0.0913
    assert!(!a.allclose_rel(&t(&[1.0, 1000.0, 0.0], &[3]), 1.0, 1.0));
}

#[test]
fn broadcast_matmul_backward() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::arange(&[2, 2, 3]));
    let w = tape.leaf(Tensor::arange(&[3, 2]));
    let shared = x.matmul_shared(w).sum_all().backward();
    let folded = x.matmul_folded(w).sum_all().backward();
    assert_eq!(grad(&shared, w), [18.0, 18.0, 22.0, 22.0, 26.0, 26.0]); // Σ over rows
    assert_eq!(grad(&shared, w), grad(&folded, w));
    assert_eq!(grad(&shared, x), grad(&folded, x));
    let inputs = [golden(&[3, 2, 4], -1.0, 1.0), golden(&[4, 2], -1.0, 1.0)];
    let r = gradcheck(|v| v[0].matmul_shared(v[1]), &inputs, Tolerance::default());
    assert!(r.is_ok(), "{r:?}");
}

#[test]
fn directional_derivative_check() {
    let x = [golden(&[2, 3], -2.0, 2.0)];
    let d = [golden(&[2, 3], -1.0, 1.0)]; // the direction
    let (a, n) = directional_check(|v| v[0].tanh(), &x, &d, 5e-3);
    assert!((a - n).abs() <= 1e-3 + 1e-2 * n.abs(), "{a} {n}");
    // tanh with 1 - y for 1 - y²: the directional check catches it.
    let (a, n) = directional_check(
        |v| {
            v[0].custom("tanh", Tensor::tanh, |_, y, g| {
                g.mul(&y.neg().add_scalar(1.0))
            })
        },
        &x,
        &d,
        5e-3,
    );
    assert!(
        (a - -2.2401).abs() < 1e-3 && (n - 0.1887).abs() < 1e-3,
        "{a} {n}"
    );
}

#[test]
fn gram_matrix_bugs() {
    let x = [golden(&[3, 2], -1.0, 1.0)];
    let d = [golden(&[3, 2], -1.0, 1.0)]; // a direction, for L = Σ y
    // @listing gram
    type Rule = fn(&Tensor, &Tensor, &Tensor) -> Tensor;
    let rules: [Rule; 3] = [
        |x, _, g| g.matmul(x).add(&g.transpose(0, 1).matmul(x)), // right
        |x, _, g| g.matmul(x).mul_scalar(2.0), // right only for a symmetric g
        |x, _, g| g.matmul(x),                 // one of the two reads
    ];
    let gram = |x: &Tensor| x.matmul(&x.transpose(0, 1)); // x xᵀ
    let (tol, mut verdicts) = (Tolerance::default(), Vec::new());
    for rule in rules {
        let weighted = gradcheck(|v| v[0].custom("gram", gram, rule), &x, tol);
        let (a, n) =
            directional_check(|v| v[0].custom("gram", gram, rule), &x, &d, 5e-3);
        verdicts.push((weighted.is_ok(), (a - n).abs() <= 1e-3 + 1e-2 * n.abs()));
    }
    assert_eq!(verdicts, [(true, true), (false, true), (false, false)]);
    // @end
}
