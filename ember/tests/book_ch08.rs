//! The examples printed in Chapter 8 of the guided edition: backward rules for the
//! elementwise operations.
use ember::{Gradients, Tape, Tensor, Var};
use std::f32::consts::{E, LN_2};

/// A tensor of shape `[]` holding `x`.
fn scalar(x: f32) -> Tensor {
    Tensor::from_vec(vec![x], &[])
}

fn vector(xs: &[f32]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), &[xs.len()])
}

fn grad(grads: &Gradients, v: Var) -> Vec<f32> {
    grads.get(v).expect("a gradient").to_vec()
}

/// Every element of `xs` within `tol` of the matching element of `want`.
fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(x, w)| (x - w).abs() <= tol)
}

#[test]
fn subtraction_and_scalars() {
    // @listing sub-example
    let tape = Tape::new();
    let a = tape.leaf(vector(&[5.0, 1.0]));
    let b = tape.leaf(vector(&[2.0, 3.0]));
    let y = a.sub(b).mul_scalar(3.0).add_scalar(1.0); // 3(a - b) + 1
    assert_eq!(y.value().to_vec(), [10.0, -5.0]);
    let grads = y.sum_all().backward();
    assert_eq!(grad(&grads, a), [3.0, 3.0]); // 1 · 3 · 1
    assert_eq!(grad(&grads, b), [-3.0, -3.0]); // -1 · 3 · 1
    let n = a.neg().div_scalar(2.0).sum_all(); // -a/2
    assert_eq!(grad(&n.backward(), a), [-0.5, -0.5]);
    // @end
}

#[test]
fn division() {
    // @listing div-example
    let tape = Tape::new();
    let a = tape.leaf(vector(&[3.0, 1.0]));
    let b = tape.leaf(vector(&[2.0, -4.0]));
    let q = a.div(b);
    assert_eq!(q.value().to_vec(), [1.5, -0.25]);
    let grads = q.sum_all().backward();
    assert_eq!(grad(&grads, a), [0.5, -0.25]); // 1/b
    assert_eq!(grad(&grads, b), [-0.75, -0.0625]); // -a/b²
    // @end
}

#[test]
fn exp_and_ln() {
    // @listing exp-ln-example
    let tape = Tape::new();
    let x = tape.leaf(vector(&[0.5, 1.0, 4.0]));
    let e = x.exp();
    assert!(close(&e.value().to_vec(), &[1.6487213, E, 54.59815], 1e-5));
    let de = grad(&e.sum_all().backward(), x);
    assert_eq!(de, e.value().to_vec()); // the output itself
    let l = x.ln();
    assert!(close(&l.value().to_vec(), &[-LN_2, 0.0, 2.0 * LN_2], 1e-6));
    assert_eq!(grad(&l.sum_all().backward(), x), [2.0, 1.0, 0.25]); // 1/x
    let back = x.ln().exp(); // x again, so its gradient is 1
    assert!(close(&grad(&back.sum_all().backward(), x), &[1.0; 3], 1e-6));
    // @end
}

#[test]
fn activations() {
    // @listing activations-example
    let tape = Tape::new();
    let ln3 = 3f32.ln();
    let x = tape.leaf(vector(&[-1.0, 0.0, ln3 / 2.0, ln3]));
    let t = x.tanh();
    assert!(close(
        &t.value().to_vec(),
        &[-0.7615942, 0.0, 0.5, 0.8],
        1e-6
    ));
    let dt = grad(&t.sum_all().backward(), x); // 1 - y²
    assert!(close(&dt, &[0.4199743, 1.0, 0.75, 0.36], 1e-6));
    let s = x.sigmoid();
    assert!(close(
        &s.value().to_vec(),
        &[0.2689414, 0.5, 0.6339746, 0.75],
        1e-6
    ));
    let ds = grad(&s.sum_all().backward(), x); // y(1 - y)
    assert!(close(&ds, &[0.1966119, 0.25, 0.2320508, 0.1875], 1e-6));
    let r = x.relu();
    assert_eq!(grad(&r.sum_all().backward(), x), [0.0, 0.0, 1.0, 1.0]);
    // @end
}

#[test]
fn a_primitive_sigmoid_is_one_node() {
    // @listing sigmoid-nodes
    let tape = Tape::new();
    let x = tape.leaf(scalar(3f32.ln()));
    let s = x.sigmoid();
    assert_eq!((tape.len(), s.op(), s.parents()), (2, "sigmoid", vec![0]));
    assert!(close(&s.value().to_vec(), &[0.75], 1e-7));
    assert!(close(&grad(&s.backward(), x), &[0.1875], 1e-7));
    // @end
}

#[test]
fn powers() {
    // @listing powf-example
    let tape = Tape::new();
    let x = tape.leaf(vector(&[0.0, 1.0, 2.0, 4.0]));
    let sq = x.powf(2.0).sum_all();
    assert_eq!(grad(&sq.backward(), x), [0.0, 2.0, 4.0, 8.0]); // 2x
    let root = x.powf(0.5).sum_all();
    let d = grad(&root.backward(), x); // 0.5 / √x
    assert_eq!(d, [f32::INFINITY, 0.5, 0.35355338, 0.25]); // ∞ at 0
    let inv = x.powf(-1.0).sum_all();
    assert_eq!(grad(&inv.backward(), x)[1..], [-1.0, -0.25, -0.0625]); // -1/x²
    // @end
}

#[test]
fn operators_on_vars() {
    // @listing operators-example
    let tape = Tape::new();
    let x = tape.leaf(vector(&[1.0, 2.0]));
    let w = tape.leaf(vector(&[3.0, -1.0]));
    let y = (w * x - 1.0) / 2.0 + 1.0 / x; // no &, no clone
    assert_eq!(y.value().to_vec(), [2.0, -1.0]); // 1 + 1, -1.5 + 0.5
    let grads = y.sum_all().backward();
    assert_eq!(grad(&grads, x), [0.5, -0.75]); // w/2 - 1/x²
    assert_eq!(grad(&grads, w), [0.5, 1.0]); // x/2
    // @end
    let z = 1.0 - -x; // neg, then add_scalar
    assert_eq!((z.value().to_vec(), z.op()), (vec![2.0, 3.0], "add_scalar"));
}

#[test]
fn used_more_than_once() {
    // @listing fan-out-example
    let tape = Tape::new();
    let x = tape.leaf(vector(&[1.0, 2.0, 3.0]));
    let d = |y: Var| grad(&y.sum_all().backward(), x);
    assert_eq!(d(x + x), [2.0, 2.0, 2.0]); // 1 + 1
    assert_eq!(d(x * x), [2.0, 4.0, 6.0]); // x + x
    assert_eq!(d(x * x * x), [3.0, 12.0, 27.0]); // x² + x·x + x·x
    assert_eq!(d(x.powf(3.0)), [3.0, 12.0, 27.0]); // 3x², from one rule
    // @end
}

/// The two-layer computation of Section 8.7, with its four leaves.
fn two_layers(tape: &Tape, p: [f32; 4]) -> (Var<'_>, [Var<'_>; 4]) {
    let x = tape.constant(scalar(1.5));
    let [w1, b1, w2, b2] = p.map(|v| tape.leaf(scalar(v)));
    let h = (w1 * x + b1).relu();
    let s = (w2 * h + b2).sigmoid();
    let l = (s - 1.0).powf(2.0);
    (l, [w1, b1, w2, b2])
}

#[test]
fn the_chain_rule_as_code() {
    // @listing chain-forward
    let tape = Tape::new();
    let x = tape.constant(scalar(1.5));
    let [w1, b1, w2, b2] = [2.0, -1.0, -0.5, 1.0].map(|v| tape.leaf(scalar(v)));
    let h = (w1 * x + b1).relu(); // layer 1
    let s = (w2 * h + b2).sigmoid(); // layer 2
    let l = (s - 1.0).powf(2.0); // squared error from a target of 1
    assert_eq!((tape.len(), h.id(), s.id(), l.id()), (13, 7, 10, 12));
    let at = |v: Var| v.value().item();
    assert_eq!([at(h), at(s), at(l)], [2.0, 0.5, 0.25]);
    // @end

    // @listing chain-backward
    let grads = l.backward();
    let g = |v: Var| grads.get(v).unwrap().item();
    assert_eq!([g(s), g(h)], [-1.0, 0.125]);
    assert_eq!([g(w1), g(b1), g(w2), g(b2)], [0.1875, 0.125, -0.5, -0.25]);
    assert!(grads.get(x).is_none()); // a constant
    // @end
}

/// Section 8.7's two tables, every row: the same computation with every
/// intermediate named.
#[test]
fn the_chain_rule_node_by_node() {
    let tape = Tape::new();
    let x = tape.constant(scalar(1.5));
    let [w1, b1, w2, b2] = [2.0, -1.0, -0.5, 1.0].map(|v| tape.leaf(scalar(v)));
    let u1 = w1 * x;
    let z1 = u1 + b1;
    let h = z1.relu();
    let u2 = w2 * h;
    let z2 = u2 + b2;
    let s = z2.sigmoid();
    let e = s - 1.0;
    let l = e.powf(2.0);
    let nodes = [x, w1, b1, w2, b2, u1, z1, h, u2, z2, s, e, l];
    let ids: Vec<usize> = nodes.iter().map(|v| v.id()).collect();
    assert_eq!(ids, (0..13).collect::<Vec<_>>());
    let values: Vec<f32> = nodes.iter().map(|v| v.value().item()).collect();
    let want = [
        1.5, 2.0, -1.0, -0.5, 1.0, 3.0, 2.0, 2.0, -1.0, 0.0, 0.5, -0.5, 0.25,
    ];
    assert_eq!(values, want);
    let parents: Vec<Vec<usize>> = nodes[5..].iter().map(|v| v.parents()).collect();
    let want: [&[usize]; 8] =
        [&[1, 0], &[5, 2], &[6], &[3, 7], &[8, 4], &[9], &[10], &[11]];
    assert_eq!(parents, want);
    let ops: Vec<&str> = nodes[5..].iter().map(|v| v.op()).collect();
    let want = [
        "mul",
        "add",
        "relu",
        "mul",
        "add",
        "sigmoid",
        "add_scalar",
        "powf",
    ];
    assert_eq!(ops, want);
    let grads = l.backward();
    let g: Vec<f32> = nodes[1..]
        .iter()
        .map(|&v| grads.get(v).unwrap().item())
        .collect();
    let want = [
        0.1875, 0.125, -0.5, -0.25, 0.125, 0.125, 0.125, -0.25, -0.25, -1.0, -1.0,
        1.0,
    ];
    assert_eq!(g, want);
    assert!(grads.get(x).is_none());
}

#[test]
fn nudges_agree_with_the_tape() {
    // @listing chain-nudge
    let p = [2.0, -1.0, -0.5, 1.0]; // w1, b1, w2, b2
    let loss = |i: usize, by: f32| {
        let mut q = p;
        q[i] += by; // nudge parameter i
        let tape = Tape::new();
        two_layers(&tape, q).0.value().item()
    };
    let h = 1e-3;
    let slope = |i| (loss(i, h) - loss(i, -h)) / (2.0 * h);
    let slopes: Vec<f32> = (0..4).map(slope).collect();
    assert!(close(&slopes, &[0.1875, 0.125, -0.5, -0.25], 1e-4));
    // @end
    let tape = Tape::new();
    let (l, leaves) = two_layers(&tape, p);
    let grads = l.backward();
    let exact: Vec<f32> = leaves
        .iter()
        .map(|&v| grads.get(v).unwrap().item())
        .collect();
    assert!(close(&slopes, &exact, 1e-4));
}

#[test]
fn ln_near_zero() {
    // @listing ln-zero
    let tape = Tape::new();
    let x = tape.leaf(vector(&[0.0, 1e-6, 1.0]));
    let y = x.ln();
    assert_eq!(y.value().get(&[0]), f32::NEG_INFINITY);
    assert!(close(&y.value().to_vec()[1..], &[-13.815511, 0.0], 1e-5));
    let d = grad(&y.sum_all().backward(), x); // 1/x
    assert_eq!(d[0], f32::INFINITY);
    assert!(close(&d[1..], &[1e6, 1.0], 0.1));
    let xlnx = (x * x.ln()).sum_all(); // 0 · -∞ in the forward pass
    assert!(xlnx.value().item().is_nan());
    assert!(grad(&xlnx.backward(), x)[0].is_nan());
    // @end
}

#[test]
fn saturation() {
    // @listing saturation
    let tape = Tape::new();
    let x = tape.leaf(vector(&[-20.0, -10.0, 0.0, 10.0, 20.0]));
    let s = x.sigmoid();
    let want = [2.0611537e-9, 4.5397872e-5, 0.5, 0.9999546, 1.0];
    assert!(close(&s.value().to_vec(), &want, 1e-7)); // σ(20) rounds to 1
    let ds = grad(&s.sum_all().backward(), x); // y(1 - y)
    assert!((ds[0] - 2.0611537e-9).abs() < 1e-15);
    assert_eq!(ds[2], 0.25);
    assert_eq!((0.25 / ds[1]).round(), 5507.0); // 4.5396e-5, right
    assert!((ds[3] - 4.5417e-5).abs() < 1e-9); // 4.5417e-5, a little off
    assert_eq!(ds[4], 0.0); // 1 · (1 - 1): the true slope is 2.06e-9
    let dt = grad(&x.tanh().sum_all().backward(), x);
    assert_eq!(dt, [0.0, 0.0, 1.0, 0.0, 0.0]); // tanh(±10) rounds to ±1
    // @end
    assert_eq!(1.0f32.next_down(), 1.0 - 5.9604645e-8); // the f32 spacing below 1
    assert!((0.25 / ds[3] - 5505.0).abs() < 1.0); // from the rounded y
}
