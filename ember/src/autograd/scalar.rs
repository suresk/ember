//! Chapter 6: a scalar autograd engine in the style of micrograd. Every number is a
//! node on the heap that remembers the operation that made it, and `backward` walks
//! the graph from the output to the leaves, applying the chain rule at each node.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

// @listing node
/// A shared handle to a node. Cloning a `Value` copies the handle, not the node.
#[derive(Clone)]
pub struct Value(pub(crate) Rc<RefCell<Node>>);

/// One number in the graph: its value, the gradient of the output with respect to
/// it, and the operation that produced it.
pub(crate) struct Node {
    pub(crate) data: f32,
    pub(crate) grad: f32,
    pub(crate) op: Op,
}
// @end

// @listing op
/// How a node was made, holding handles to its inputs. A leaf has no inputs.
pub(crate) enum Op {
    Leaf,
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    Tanh(Value),
    Exp(Value),
    Relu(Value),
    Powf(Value, f32),
}
// @end

impl Value {
    // @listing accessors
    /// A leaf: a number that no operation produced, such as an input or a weight.
    pub fn new(data: f32) -> Value {
        Value::from_op(data, Op::Leaf)
    }

    pub(crate) fn from_op(data: f32, op: Op) -> Value {
        let node = Node {
            data,
            grad: 0.0,
            op,
        };
        Value(Rc::new(RefCell::new(node)))
    }

    pub fn data(&self) -> f32 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f32 {
        self.0.borrow().grad
    }

    pub fn set_grad(&self, grad: f32) {
        self.0.borrow_mut().grad = grad;
    }

    pub fn add_grad(&self, grad: f32) {
        self.0.borrow_mut().grad += grad;
    }
    // @end

    /// The node's address, which identifies it while any handle to it is alive.
    pub(crate) fn id(&self) -> usize {
        Rc::as_ptr(&self.0).addr()
    }

    // @listing functions
    pub fn tanh(&self) -> Value {
        Value::from_op(self.data().tanh(), Op::Tanh(self.clone()))
    }

    pub fn exp(&self) -> Value {
        Value::from_op(self.data().exp(), Op::Exp(self.clone()))
    }

    pub fn relu(&self) -> Value {
        Value::from_op(self.data().max(0.0), Op::Relu(self.clone()))
    }

    pub fn powf(&self, p: f32) -> Value {
        Value::from_op(self.data().powf(p), Op::Powf(self.clone(), p))
    }
    // @end

    // @listing local-grads
    /// This node's inputs, each with the derivative of this node with respect to it.
    pub fn local_grads(&self) -> Vec<(Value, f32)> {
        let node = self.0.borrow();
        let y = node.data; // the output, which some derivatives reuse
        match &node.op {
            Op::Leaf => vec![],
            Op::Add(a, b) => vec![(a.clone(), 1.0), (b.clone(), 1.0)],
            Op::Sub(a, b) => vec![(a.clone(), 1.0), (b.clone(), -1.0)],
            Op::Mul(a, b) => vec![(a.clone(), b.data()), (b.clone(), a.data())],
            Op::Tanh(a) => vec![(a.clone(), 1.0 - y * y)],
            Op::Exp(a) => vec![(a.clone(), y)],
            Op::Relu(a) => vec![(a.clone(), if y > 0.0 { 1.0 } else { 0.0 })],
            Op::Powf(a, p) => vec![(a.clone(), p * a.data().powf(p - 1.0))],
        }
    }
    // @end

    // @listing topo
    /// Every node `self` depends on, each after all of its inputs; `self` is last.
    pub fn topo_order(&self) -> Vec<Value> {
        let mut seen = HashSet::new();
        let mut order = Vec::new();
        self.visit(&mut seen, &mut order);
        order
    }

    fn visit(&self, seen: &mut HashSet<usize>, order: &mut Vec<Value>) {
        if !seen.insert(self.id()) {
            return; // reached before, by another path
        }
        for (input, _) in self.local_grads() {
            input.visit(seen, order);
        }
        order.push(self.clone()); // after every input
    }
    // @end

    // @listing backward
    /// Fill in `grad` for every node the output depends on: the derivative of `self`
    /// with respect to that node. Gradients add to whatever `grad` already holds.
    pub fn backward(&self) {
        let order = self.topo_order();
        self.set_grad(1.0); // dL/dL
        for node in order.iter().rev() {
            let upstream = node.grad();
            for (input, local) in node.local_grads() {
                input.add_grad(local * upstream); // the chain rule
            }
        }
    }
    // @end
}

/// `+`, `-` and `*` on any mix of `Value` and `&Value`.
// @listing operators
macro_rules! operators {
    ($($Op:ident $op:ident),*) => {$(
        impl<R: std::borrow::Borrow<Value>> std::ops::$Op<R> for &Value {
            type Output = Value;
            fn $op(self, rhs: R) -> Value {
                let rhs: &Value = rhs.borrow();
                let data = std::ops::$Op::$op(self.data(), rhs.data());
                Value::from_op(data, Op::$Op(self.clone(), rhs.clone()))
            }
        }

        impl<R: std::borrow::Borrow<Value>> std::ops::$Op<R> for Value {
            type Output = Value;
            fn $op(self, rhs: R) -> Value {
                std::ops::$Op::$op(&self, rhs)
            }
        }
    )*};
}

operators!(Add add, Sub sub, Mul mul);
// @end

/// Prints the number and its gradient, not the graph behind it.
impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={}, grad={})", self.data(), self.grad())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Section 6.10's count: what one number costs in this engine, on a 64-bit target.
    #[test]
    fn one_number_takes_56_bytes() {
        assert_eq!(size_of::<Value>(), 8); // a pointer
        assert_eq!(size_of::<Op>(), 24);
        assert_eq!(size_of::<RefCell<Node>>(), 40);
        // Rc's heap block: the strong and weak counts, then the RefCell.
        assert_eq!(2 * size_of::<usize>() + size_of::<RefCell<Node>>(), 56);
    }
}
