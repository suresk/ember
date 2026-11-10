//! The examples printed in Chapter 6 of the guided edition: a scalar autograd engine,
//! traced by hand.
use ember::autograd::scalar::Value;
use std::cell::RefCell;

fn close(x: f32, y: f32, tol: f32) -> bool {
    (x - y).abs() <= tol
}

fn data(values: &[Value]) -> Vec<f32> {
    values.iter().map(Value::data).collect()
}

#[test]
fn nudging_an_input() {
    // @listing nudge
    let loss = |a: f32, b: f32, c: f32, f: f32| (a * b + c) * f;
    let (a, b, c, f) = (3.0, -2.0, 8.0, 4.0);
    assert_eq!(loss(a, b, c, f), 8.0);
    let h = 0.001;
    assert!(close(loss(a + h, b, c, f), 7.992, 1e-5)); // L falls by 0.008

    // The slope: change in L per unit change in the input.
    let l = loss(a, b, c, f);
    let slope = |nudged: f32| (nudged - l) / h;
    assert!(close(slope(loss(a + h, b, c, f)), -8.0, 0.001)); // -7.9994
    assert!(close(slope(loss(a, b + h, c, f)), 12.0, 0.002)); // 12.0010
    assert!(close(slope(loss(a, b, c + h, f)), 4.0, 0.002)); // 4.0016
    assert!(close(slope(loss(a, b, c, f + h)), 2.0, 0.001)); // 1.9999
    // @end
}

#[test]
fn a_smaller_nudge_is_not_better() {
    // @listing nudge-rounding
    let loss = |a: f32| (a * -2.0 + 8.0) * 4.0;
    let slope = |h: f32| (loss(3.0 + h) - loss(3.0)) / h;
    assert!(close(slope(0.1), -8.0, 1e-5)); // -7.99999
    assert!(close(slope(0.001), -8.0, 0.001)); // -7.9994
    assert!(close(slope(0.00001), -8.011, 0.001)); // -8.0109
    // @end
}

/// The graph of Sections 6.2 and 6.3: L = (a·b + c)·f.
fn graph() -> [Value; 7] {
    let (a, b) = (Value::new(3.0), Value::new(-2.0));
    let (c, f) = (Value::new(8.0), Value::new(4.0));
    let e = &a * &b;
    let d = &e + &c;
    let l = &d * &f;
    [a, b, c, f, e, d, l]
}

#[test]
fn the_forward_pass() {
    // @listing forward
    let a = Value::new(3.0);
    let b = Value::new(-2.0);
    let c = Value::new(8.0);
    let f = Value::new(4.0);
    let e = &a * &b;
    let d = &e + &c;
    let l = &d * &f;
    assert_eq!([e.data(), d.data(), l.data()], [-6.0, 2.0, 8.0]);
    assert_eq!(a.grad(), 0.0); // nothing has run backward yet
    assert_eq!(l.local_grads().len(), 2); // its inputs, d and f
    // @end
}

#[test]
fn local_derivatives() {
    let [.., l] = graph();
    let inputs = l.local_grads(); // L = d·f
    assert_eq!(inputs[0].0.data(), 2.0); // d
    assert_eq!(inputs[0].1, 4.0); // dL/dd = f
    assert_eq!(inputs[1].1, 2.0); // dL/df = d

    // @listing local-example
    let x = Value::new(3.0_f32.ln() / 2.0); // 0.5493
    assert!(close(x.tanh().data(), 0.5, 1e-6));
    assert!(close(x.tanh().local_grads()[0].1, 0.75, 1e-6)); // 1 - 0.5²
    assert_eq!(Value::new(0.0).exp().local_grads()[0].1, 1.0);
    assert_eq!(Value::new(-1.0).relu().local_grads()[0].1, 0.0);
    assert_eq!(Value::new(3.0).powf(2.0).local_grads()[0].1, 6.0); // 2·3
    // @end
}

#[test]
fn topological_order() {
    // @listing topo-example
    let [.., l] = graph();
    let order = l.topo_order();
    // a, b, e, c, d, f, L: each node after its inputs.
    assert_eq!(data(&order), [3.0, -2.0, -6.0, 8.0, 2.0, 4.0, 8.0]);
    // @end
}

#[test]
fn backward_on_the_graph() {
    // @listing backward-example
    let [a, b, c, f, e, d, l] = graph();
    l.backward();
    assert_eq!(l.grad(), 1.0);
    assert_eq!([d.grad(), f.grad()], [4.0, 2.0]);
    assert_eq!([e.grad(), c.grad()], [4.0, 4.0]);
    assert_eq!([a.grad(), b.grad()], [-8.0, 12.0]); // 4·b and 4·a
    // @end
}

// @listing overwrite
/// `backward` with one character changed: `=` where it had `+=`.
fn backward_overwriting(output: &Value) {
    let order = output.topo_order();
    output.set_grad(1.0);
    for node in order.iter().rev() {
        let upstream = node.grad();
        for (input, local) in node.local_grads() {
            input.set_grad(local * upstream); // overwrites
        }
    }
}
// @end

#[test]
fn gradients_add_on_fan_out() {
    // @listing fan-out
    let (a, b) = (Value::new(3.0), Value::new(-2.0));
    let l = &a * &b + &a; // a is used twice
    assert_eq!(l.data(), -3.0);
    l.backward();
    assert_eq!([a.grad(), b.grad()], [-1.0, 3.0]); // dL/da = b + 1

    let (a, b) = (Value::new(3.0), Value::new(-2.0));
    let l = &a * &b + &a;
    backward_overwriting(&l);
    assert_eq!([a.grad(), b.grad()], [-2.0, 3.0]); // the +1 is lost
    // @end
}

#[test]
fn a_square_is_a_node_with_one_input_twice() {
    // @listing square
    let x = Value::new(3.0);
    let y = &x * &x;
    y.backward();
    assert_eq!(x.grad(), 6.0); // 2x
    let x = Value::new(3.0);
    backward_overwriting(&(&x * &x));
    assert_eq!(x.grad(), 3.0); // half of it
    // @end
}

#[test]
fn a_shared_node_is_visited_once() {
    let (a, b) = (Value::new(3.0), Value::new(-2.0));
    let l = &a * &b + &a;
    assert_eq!(data(&l.topo_order()), [3.0, -2.0, -6.0, -3.0]);
}

/// The loop of `backward`, from zero gradients, over an order the caller chooses.
fn backward_in(order: &[&Value]) {
    for node in order {
        node.set_grad(0.0);
    }
    order[0].set_grad(1.0);
    for node in order {
        let upstream = node.grad();
        for (input, local) in node.local_grads() {
            input.add_grad(local * upstream);
        }
    }
}

#[test]
fn the_order_matters() {
    // @listing wrong-order
    let (a, b, c) = (Value::new(3.0), Value::new(-2.0), Value::new(4.0));
    let d = &a * &b; // -6
    let m = &d * &c; // -24
    let l = &d + &m; // -30: d is used by m and by L
    l.backward();
    assert_eq!([d.grad(), a.grad(), b.grad()], [5.0, -10.0, 15.0]);

    // Breadth first: L, then its inputs d and m, then theirs.
    backward_in(&[&l, &d, &m, &a, &b, &c]);
    assert_eq!(d.grad(), 5.0); // d's own gradient ends up right...
    assert_eq!([a.grad(), b.grad()], [-2.0, 3.0]); // ...after d used 1
    // @end
    assert_eq!(c.grad(), -6.0);

    // Written m + d, the same breadth-first rule visits m first and is right.
    let l = &m + &d;
    backward_in(&[&l, &m, &d, &c, &a, &b]);
    assert_eq!([a.grad(), b.grad()], [-10.0, 15.0]);
}

#[test]
fn a_sum_of_a_thousand_numbers() {
    // @listing chain
    let xs: Vec<Value> = (0..1000).map(|i| Value::new(i as f32)).collect();
    let mut total = Value::new(0.0);
    for x in &xs {
        total = total + x; // a chain: each sum is an input of the next
    }
    assert_eq!(total.topo_order().len(), 2001); // 1001 leaves, 1000 sums
    total.backward();
    assert!(xs.iter().all(|x| x.grad() == 1.0));
    // @end
    assert_eq!(total.data(), 499_500.0);
}

// @listing refcell-panic
#[test]
#[should_panic(expected = "RefCell already mutably borrowed")]
fn two_borrows_of_one_node() {
    let grad = RefCell::new(3.0_f32); // x's gradient, in y = x * x
    let mut into_a = grad.borrow_mut(); // a is x...
    *into_a += 2.0 * *grad.borrow(); // ...and so is b
}
// @end
