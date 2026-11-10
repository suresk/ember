//! Tests for the Chapter 6 challenges: `cargo test --test challenges_ch06`.
use ember::autograd::scalar::Value;
use ember::{neuron, numeric_grads};

fn close(x: f32, y: f32, tol: f32) -> bool {
    (x - y).abs() <= tol
}

/// Section 6.7's `backward` with `=` in place of `+=`.
fn backward_overwriting(output: &Value) {
    let order = output.topo_order();
    output.set_grad(1.0);
    for node in order.iter().rev() {
        let upstream = node.grad();
        for (input, local) in node.local_grads() {
            input.set_grad(local * upstream);
        }
    }
}

fn predict_graph() -> [Value; 4] {
    let (x, y, z) = (Value::new(2.0), Value::new(3.0), Value::new(-1.0));
    let l = (&x + &y) * (&y * &z);
    [x, y, z, l]
}

#[test]
fn predict_gradients() {
    let [x, y, z, l] = predict_graph();
    assert_eq!(l.data(), -15.0); // 5 · -3
    assert_eq!(l.topo_order().len(), 6);
    l.backward();
    assert_eq!([x.grad(), y.grad(), z.grad()], [-3.0, -8.0, 15.0]);
    let [x, y, z, l] = predict_graph();
    backward_overwriting(&l);
    assert_eq!([x.grad(), y.grad(), z.grad()], [-3.0, -3.0, 15.0]);
}

#[test]
fn zero_grad_resets_the_graph() {
    let (a, b, c, f) = (
        Value::new(3.0),
        Value::new(-2.0),
        Value::new(8.0),
        Value::new(4.0),
    );
    let d = &a * &b + &c;
    let l = &d * &f;
    l.backward();
    l.backward(); // without zero_grad: d starts the second pass at 4, not 0
    assert_eq!([l.grad(), d.grad()], [1.0, 8.0]);
    assert_eq!([a.grad(), b.grad()], [-32.0, 48.0]); // four times, not twice
    l.zero_grad();
    assert!(l.topo_order().iter().all(|v| v.grad() == 0.0));
    l.backward();
    assert_eq!([a.grad(), b.grad()], [-8.0, 12.0]);
}

#[test]
fn div_and_sigmoid() {
    let (a, b) = (Value::new(3.0), Value::new(2.0));
    let q = a.div(&b);
    assert_eq!(q.data(), 1.5);
    q.backward();
    assert_eq!([a.grad(), b.grad()], [0.5, -0.75]); // 1/b and -a/b²

    let x = Value::new(3.0_f32.ln()); // e⁻ˣ = 1/3
    let s = x.sigmoid();
    assert!(close(s.data(), 0.75, 1e-6));
    assert_eq!(s.topo_order().len(), 7);
    s.backward();
    assert!(close(x.grad(), 0.1875, 1e-6)); // s(1 - s) = 3/16
}

#[test]
fn a_two_input_neuron() {
    let xs = [Value::new(2.0), Value::new(-1.0)];
    let ws = [Value::new(0.5), Value::new(-0.25)];
    let b = Value::new(-1.0);
    let o = neuron(&xs, &ws, &b); // tanh(1 + 0.25 - 1) = tanh(0.25)
    assert!(close(o.data(), 0.2449, 1e-4));
    o.backward();
    let dn = 1.0 - o.data() * o.data(); // do/dn = 0.9400
    assert!(close(dn, 0.9400, 1e-4));
    assert!(close(ws[0].grad(), 1.880, 1e-3)); // dn · x₁
    assert!(close(ws[1].grad(), -0.940, 1e-3)); // dn · x₂
    assert!(close(b.grad(), 0.940, 1e-3));
    assert!(close(xs[0].grad(), 0.470, 1e-3)); // dn · w₁
    assert!(close(xs[1].grad(), -0.235, 1e-3)); // dn · w₂
}

#[test]
#[should_panic(expected = "neuron: 2 inputs, 1 weights")]
fn neuron_wants_a_weight_per_input() {
    neuron(
        &[Value::new(1.0), Value::new(2.0)],
        &[Value::new(1.0)],
        &Value::new(0.0),
    );
}

#[test]
fn numeric_grads_agree_with_backward() {
    let f = |v: &[Value]| neuron(&v[..2], &v[2..4], &v[4]);
    let at = [2.0, -1.0, 0.5, -0.25, -1.0];
    let inputs: Vec<Value> = at.iter().map(|&x| Value::new(x)).collect();
    f(&inputs).backward();
    let numeric = numeric_grads(f, &at, 0.01);
    for (v, n) in inputs.iter().zip(&numeric) {
        assert!(close(v.grad(), *n, 1e-3), "{} vs {n}", v.grad());
    }

    // Central differences against one-sided ones, for exp at 1 (exact slope e).
    let e = 1.0_f32.exp();
    let central = numeric_grads(|v| v[0].exp(), &[1.0], 0.01)[0];
    let one_sided = ((1.01_f32).exp() - e) / 0.01;
    assert!((central - e).abs() < 1e-4);
    assert!(close(one_sided - e, 0.0136, 1e-3));
}

#[test]
fn a_number_used_three_times() {
    let x = Value::new(3.0);
    let y = &x * &x * &x; // (x·x)·x
    assert_eq!(y.topo_order().len(), 3);
    y.backward();
    assert_eq!(x.grad(), 27.0); // 3x²: 9 from the outer product, 9 + 9 inner

    let x = Value::new(3.0);
    backward_overwriting(&(&x * &x * &x));
    assert_eq!(x.grad(), 9.0);

    let x = Value::new(3.0);
    let y = x.powf(3.0);
    assert_eq!(y.topo_order().len(), 2);
    y.backward();
    assert_eq!(x.grad(), 27.0);
}

#[test]
fn the_graph_as_dot() {
    let (a, b) = (Value::new(3.0), Value::new(-2.0));
    let l = &a * &b + &a;
    l.backward();
    let dot = "digraph {
  n0 [label=\"3, grad -1\"];
  n1 [label=\"-2, grad 3\"];
  n2 [label=\"* = -6, grad 1\"];
  n3 [label=\"+ = -3, grad 1\"];
  n0 -> n2;
  n1 -> n2;
  n2 -> n3;
  n0 -> n3;
}
";
    assert_eq!(l.to_dot(), dot);
}
