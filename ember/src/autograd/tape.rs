//! Chapter 7: a tape of tensor operations. Every value a computation produces is
//! appended to one `Vec`, in the order it ran, so walking that `Vec` backward visits
//! each node after everything computed from it. A `Var` is an index into the tape.
//!
//! A `Var` borrows its tape, so it can't outlive it. This doesn't compile:
//!
//! ```compile_fail
//! use ember::{Tape, Tensor};
//!
//! let x = {
//!     let tape = Tape::new();
//!     tape.leaf(Tensor::zeros(&[3]))
//! };
//! println!("{:?}", x.shape());
//! ```

use std::cell::{Cell, RefCell};
use std::fmt;

use crate::nn::{ParamId, Parameter};
use crate::tensor::Tensor;

// @listing node
/// A node's backward rule: from the gradient with respect to the node's value, one
/// gradient for each of its parents, in order.
pub(crate) type BackwardFn = Box<dyn Fn(&Tensor) -> Vec<Tensor>>;

/// One value on the tape, with where it came from and how to send a gradient back.
pub(crate) struct Node {
    pub(crate) value: Tensor,
    pub(crate) op: &'static str, // "leaf", "const", "add", ...
    pub(crate) parents: Vec<usize>,
    pub(crate) requires_grad: bool,
    pub(crate) backward: Option<BackwardFn>,
}
// @end

// @listing tape
/// The record of a computation: every value it produced, in the order it ran.
pub struct Tape {
    pub(crate) nodes: RefCell<Vec<Node>>,
    grad_enabled: Cell<bool>,
    // @hide
    params: RefCell<Vec<(ParamId, usize)>>,
    // @show
}

/// A handle to one value on a tape: the tape, and the value's index in it.
#[derive(Clone, Copy)]
pub struct Var<'t> {
    pub(crate) tape: &'t Tape,
    pub(crate) id: usize,
}
// @end

impl Tape {
    pub fn new() -> Tape {
        let nodes = RefCell::new(Vec::new());
        Tape {
            nodes,
            grad_enabled: Cell::new(true),
            params: RefCell::new(Vec::new()),
        }
    }

    // @listing inputs
    /// A value we want the gradient for, such as a weight.
    pub fn leaf(&self, value: Tensor) -> Var<'_> {
        self.input(value, "leaf", true)
    }

    /// A value we don't: an input batch, a target, a fixed number.
    pub fn constant(&self, value: Tensor) -> Var<'_> {
        self.input(value, "const", false)
    }

    fn input(&self, value: Tensor, op: &'static str, requires_grad: bool) -> Var<'_> {
        self.record(Node {
            value,
            op,
            parents: Vec::new(),
            requires_grad,
            backward: None,
        })
    }
    // @end

    // @listing record
    /// Append `node` to the tape. The borrow of the `Vec` ends with the call.
    fn record(&self, node: Node) -> Var<'_> {
        // @hide
        crate::profile::mark("forward", node.op); // Chapter 24
        // @show
        let mut nodes = self.nodes.borrow_mut();
        nodes.push(node);
        let id = nodes.len() - 1;
        Var { tape: self, id }
    }
    // @end

    /// The number of nodes recorded so far.
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // @listing push
    /// Record `value`, computed from `parents`. `backward` maps its gradient to one
    /// gradient per parent; it is kept only if the node requires a gradient.
    pub(crate) fn push(
        &self,
        op: &'static str,
        value: Tensor,
        parents: Vec<usize>,
        backward: impl Fn(&Tensor) -> Vec<Tensor> + 'static,
    ) -> Var<'_> {
        let wanted = |p: &usize| self.nodes.borrow()[*p].requires_grad;
        let requires_grad = self.grad_enabled.get() && parents.iter().any(wanted);
        let backward = requires_grad.then(|| Box::new(backward) as BackwardFn);
        self.record(Node {
            value,
            op,
            parents,
            requires_grad,
            backward,
        })
    }
    // @end

    // @listing walk
    /// The backward walk from node `id`, whose gradient is `seed`: every node from
    /// `id` down to 0, each passing its finished gradient to its parents.
    pub(crate) fn backward_from(&self, id: usize, seed: Tensor) -> Gradients<'_> {
        let nodes = self.nodes.borrow();
        let mut grads: Vec<Option<Tensor>> = vec![None; id + 1];
        grads[id] = Some(seed);
        for i in (0..=id).rev() {
            let to_parents = match (&grads[i], &nodes[i].backward) {
                (Some(g), Some(rule)) => rule(g), // the chain rule, for one node
                _ => continue, // not on a path to the output, or nothing to pass
            };
            // @hide
            check_gradients(&nodes, i, &to_parents); // Chapter 9
            // @show
            for (&p, g) in nodes[i].parents.iter().zip(to_parents) {
                if nodes[p].requires_grad {
                    accumulate(&mut grads[p], g);
                }
            }
            // @hide
            crate::profile::mark("backward", nodes[i].op); // Chapter 24
            // @show
        }
        Gradients { tape: self, grads }
    }
    // @end

    pub fn is_grad_enabled(&self) -> bool {
        self.grad_enabled.get()
    }
}

// @listing no-grad
impl Tape {
    /// Run `f` with recording off: operations inside still compute their values,
    /// but keep no backward rules, and nothing they produce requires a gradient.
    pub fn no_grad<R>(&self, f: impl FnOnce() -> R) -> R {
        let was = self.grad_enabled.replace(false);
        let _restore = Restore {
            flag: &self.grad_enabled,
            was,
        }; // dropped when `no_grad` returns or unwinds
        f()
    }
}

/// Puts `was` back into `flag` when dropped.
struct Restore<'a> {
    flag: &'a Cell<bool>,
    was: bool,
}

impl Drop for Restore<'_> {
    fn drop(&mut self) {
        self.flag.set(self.was);
    }
}
// @end

impl Default for Tape {
    fn default() -> Tape {
        Tape::new()
    }
}

// @listing check-gradients
/// Panic unless node `i`'s rule returned one gradient per parent, each in its
/// parent's shape. A wrong shape would otherwise pass unnoticed: `accumulate`
/// stores a slot's first gradient as it is, and broadcasts any later one.
fn check_gradients(nodes: &[Node], i: usize, grads: &[Tensor]) {
    let (op, parents) = (nodes[i].op, &nodes[i].parents);
    let (m, n) = (grads.len(), parents.len());
    assert!(
        m == n,
        "{op}: the rule returned {m} gradients for {n} parents"
    );
    for (k, (&p, g)) in parents.iter().zip(grads).enumerate() {
        let (got, want) = (g.shape(), nodes[p].value.shape());
        assert!(
            got == want,
            "{op}: the gradient for parent {k} has shape {got:?}, not {want:?}"
        );
    }
}
// @end

// @listing accumulate
/// Add `g` into a gradient slot: gradients from several consumers add up.
pub(crate) fn accumulate(slot: &mut Option<Tensor>, g: Tensor) {
    *slot = Some(match slot.take() {
        Some(sum) => sum.add(&g),
        None => g,
    });
}
// @end

impl<'t> Var<'t> {
    // @listing var-accessors
    /// The value, sharing its buffer with the tape: a clone of the `Rc`.
    pub fn value(self) -> Tensor {
        self.tape.nodes.borrow()[self.id].value.clone()
    }

    pub fn shape(self) -> Vec<usize> {
        self.tape.nodes.borrow()[self.id].value.shape().to_vec()
    }
    // @end

    pub fn requires_grad(self) -> bool {
        self.tape.nodes.borrow()[self.id].requires_grad
    }

    /// The position on the tape: the number of nodes recorded before this one.
    pub fn id(self) -> usize {
        self.id
    }

    /// The operation that made this node: "leaf", "const", "add", "mul", ...
    pub fn op(self) -> &'static str {
        self.tape.nodes.borrow()[self.id].op
    }

    /// The ids of the nodes this one was computed from.
    pub fn parents(self) -> Vec<usize> {
        self.tape.nodes.borrow()[self.id].parents.clone()
    }

    pub fn tape(self) -> &'t Tape {
        self.tape
    }

    // @listing unary
    /// Record `y = f(x)` for `x = self`. `rule(x, y, g)` is the gradient with
    /// respect to `x`, given the gradient `g` with respect to `y`.
    pub(crate) fn unary(
        self,
        op: &'static str,
        f: impl Fn(&Tensor) -> Tensor,
        rule: impl Fn(&Tensor, &Tensor, &Tensor) -> Tensor + 'static,
    ) -> Var<'t> {
        let x = self.value();
        let y = f(&x);
        let out = y.clone(); // shares y's buffer
        self.tape
            .push(op, y, vec![self.id], move |g| vec![rule(&x, &out, g)])
    }
    // @end

    // @listing binary
    /// Record `f(a, b)` for `a = self`, `b = other`. `rule(a, b, g)` returns the
    /// gradients for `a` and `b`, summed down to their shapes (see Chapter 9).
    pub(crate) fn binary(
        self,
        other: Var<'t>,
        op: &'static str,
        f: impl Fn(&Tensor, &Tensor) -> Tensor,
        rule: impl Fn(&Tensor, &Tensor, &Tensor) -> (Tensor, Tensor) + 'static,
    ) -> Var<'t> {
        let same_tape = std::ptr::eq(self.tape, other.tape);
        assert!(same_tape, "{op}: Vars of two tapes");
        let (a, b) = (self.value(), other.value());
        let y = f(&a, &b);
        let parents = vec![self.id, other.id];
        self.tape.push(op, y, parents, move |g| {
            let (ga, gb) = rule(&a, &b, g);
            vec![ga.sum_to(a.shape()), gb.sum_to(b.shape())]
        })
    }
    // @end
}

// Chapter 8 adds `+` and `*` on `Var`, which call these.
#[allow(clippy::should_implement_trait)]
impl<'t> Var<'t> {
    // @listing ops
    pub fn add(self, other: Var<'t>) -> Var<'t> {
        self.binary(other, "add", Tensor::add, |_, _, g| (g.clone(), g.clone()))
    }

    pub fn mul(self, other: Var<'t>) -> Var<'t> {
        self.binary(other, "mul", Tensor::mul, |a, b, g| (g.mul(b), g.mul(a)))
    }

    /// The sum of every element, as a tensor of shape `[]`.
    pub fn sum_all(self) -> Var<'t> {
        let sum = |x: &Tensor| Tensor::from_vec(vec![x.sum_all()], &[]);
        self.unary("sum_all", sum, |x, _, g| g.expand(x.shape()))
    }
    // @end

    // @listing backward
    /// The gradient of `self`, which must hold exactly one number, with respect to
    /// every node it depends on that requires a gradient.
    pub fn backward(self) -> Gradients<'t> {
        let shape = self.shape();
        let n: usize = shape.iter().product();
        assert_eq!(n, 1, "backward: the output has shape {shape:?}");
        assert!(
            self.requires_grad(),
            "backward: the output requires no gradient"
        );
        let seed = Tensor::from_vec(vec![1.0], &shape); // dL/dL = 1
        self.tape.backward_from(self.id, seed)
    }
    // @end
}

// @listing gradients
/// The result of a backward pass on `tape`: one gradient slot per node, empty for
/// nodes that don't require a gradient or don't lead to the output.
pub struct Gradients<'t> {
    pub(crate) tape: &'t Tape,
    pub(crate) grads: Vec<Option<Tensor>>,
}

impl<'t> Gradients<'t> {
    /// The gradient of `var`, which must be on the tape these gradients came from.
    pub fn get(&self, var: Var<'t>) -> Option<&Tensor> {
        let same_tape = std::ptr::eq(self.tape, var.tape);
        assert!(same_tape, "Gradients::get: a Var of another tape");
        self.grads.get(var.id).and_then(Option::as_ref)
    }
}
// @end

// Chapter 11: parameters on the tape.
impl Tape {
    // @listing param
    /// Put a parameter's value on the tape as a leaf, and record which node
    /// holds it. The value is shared, not copied, and the parameter isn't borrowed.
    pub fn param(&self, p: &Parameter) -> Var<'_> {
        let var = self.input(p.value.clone(), "param", true);
        self.params.borrow_mut().push((p.id(), var.id));
        var
    }
    // @end
}

impl Var<'_> {
    // @listing backward-into
    /// Run the backward pass from `self` and add each parameter's gradient into
    /// its `grad`. A parameter the loss doesn't reach is left as it was.
    pub fn backward_into(self, params: &mut [&mut Parameter]) {
        let mut grads = self.backward();
        for &(id, node) in self.tape.params.borrow().iter() {
            let Some(g) = grads.grads.get_mut(node).and_then(Option::take) else {
                continue; // the loss doesn't depend on this node
            };
            if let Some(p) = params.iter_mut().find(|p| p.id() == id) {
                p.add_grad(g);
            }
        }
    }
    // @end
}

/// Prints the node's position and shape, not the graph behind it.
impl fmt::Debug for Var<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Var(id={}, shape={:?})", self.id, self.shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Section 7.8's count: what one node costs, on a 64-bit target.
    #[test]
    fn a_node_takes_128_bytes() {
        assert_eq!(size_of::<Tensor>(), 64); // Rc, shape, strides, offset
        assert_eq!(size_of::<Vec<usize>>(), 24);
        assert_eq!(size_of::<Option<BackwardFn>>(), 16); // data and vtable pointers
        assert_eq!(size_of::<Node>(), 128);
        assert_eq!(size_of::<Var>(), 16); // a reference and an index
    }
}
