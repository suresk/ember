//! Chapter 6 challenges.

use std::collections::HashMap;
use std::fmt::Write;

use crate::autograd::scalar::{Op, Value};

impl Value {
    // @listing zero-grad
    /// Set the gradient of every node the output depends on back to 0.
    pub fn zero_grad(&self) {
        for node in self.topo_order() {
            node.set_grad(0.0);
        }
    }
    // @end

    // @listing div-sigmoid
    /// `self / other`, as `self · other⁻¹`.
    pub fn div(&self, other: &Value) -> Value {
        self * other.powf(-1.0)
    }

    /// `1 / (1 + e⁻ˣ)`, built from operations the engine already has.
    pub fn sigmoid(&self) -> Value {
        let e = (Value::new(0.0) - self).exp();
        (Value::new(1.0) + e).powf(-1.0)
    }
    // @end

    // @listing to-dot
    /// The graph in Graphviz's DOT language: one line per node, numbered in
    /// topological order, then one line per edge, from input to output.
    pub fn to_dot(&self) -> String {
        let order = self.topo_order();
        let index: HashMap<usize, usize> =
            order.iter().enumerate().map(|(i, v)| (v.id(), i)).collect();
        let mut out = String::from("digraph {\n");
        for (i, v) in order.iter().enumerate() {
            let (x, g) = (v.data(), v.grad());
            let label = match op_name(&v.0.borrow().op) {
                "" => format!("{x}, grad {g}"),
                op => format!("{op} = {x}, grad {g}"),
            };
            writeln!(out, "  n{i} [label=\"{label}\"];").unwrap();
        }
        for (i, v) in order.iter().enumerate() {
            for (input, _) in v.local_grads() {
                writeln!(out, "  n{} -> n{i};", index[&input.id()]).unwrap();
            }
        }
        out + "}\n"
    }
    // @end
}

// @listing op-name
/// A short name for an operation; empty for a leaf.
fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Leaf => "",
        Op::Add(..) => "+",
        Op::Sub(..) => "-",
        Op::Mul(..) => "*",
        Op::Tanh(_) => "tanh",
        Op::Exp(_) => "exp",
        Op::Relu(_) => "relu",
        Op::Powf(..) => "powf",
    }
}
// @end

// @listing neuron
/// One neuron: `tanh(w · x + b)`, for inputs `xs` and weights `ws` of one length.
pub fn neuron(xs: &[Value], ws: &[Value], b: &Value) -> Value {
    let (nx, nw) = (xs.len(), ws.len());
    assert_eq!(nx, nw, "neuron: {nx} inputs, {nw} weights");
    let mut n = b.clone();
    for (x, w) in xs.iter().zip(ws) {
        n = n + w * x;
    }
    n.tanh()
}
// @end

// @listing numeric-grads
/// The derivative of `f` with respect to each of its inputs at `at`, by central
/// differences: `(f(x + h) - f(x - h)) / 2h`, one input at a time.
pub fn numeric_grads(f: impl Fn(&[Value]) -> Value, at: &[f32], h: f32) -> Vec<f32> {
    let eval = |i: usize, dx: f32| {
        let xs: Vec<Value> = (0..at.len())
            .map(|j| Value::new(if j == i { at[j] + dx } else { at[j] }))
            .collect();
        f(&xs).data()
    };
    (0..at.len())
        .map(|i| (eval(i, h) - eval(i, -h)) / (2.0 * h))
        .collect()
}
// @end
