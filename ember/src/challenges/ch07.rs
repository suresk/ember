//! Chapter 7 challenges.

use std::collections::HashSet;
use std::rc::Rc;

use crate::autograd::tape::{Gradients, Tape, Var, accumulate};
use crate::tensor::Tensor;

impl<'t> Var<'t> {
    // @listing detach
    /// The same value as a new node that doesn't require a gradient, so nothing
    /// flows back through it.
    pub fn detach(self) -> Var<'t> {
        self.tape.constant(self.value())
    }
    // @end

    // @listing backward-with
    /// The backward pass from an output of any shape, starting from `seed` as the
    /// gradient with respect to `self`.
    pub fn backward_with(self, seed: Tensor) -> Gradients {
        let (have, want) = (seed.shape(), self.shape());
        assert_eq!(have, want, "backward_with: seed {have:?}, output {want:?}");
        self.tape.backward_from(self.id, seed)
    }
    // @end

    // @listing backward-leaves
    /// `backward`, keeping only the leaves' gradients: every other node's gradient
    /// is dropped as soon as it has been passed to its parents.
    pub fn backward_leaves(self) -> Gradients {
        assert_eq!(self.value().numel(), 1, "backward_leaves: not 1 element");
        let nodes = self.tape.nodes.borrow();
        let mut grads: Vec<Option<Tensor>> = vec![None; self.id + 1];
        grads[self.id] = Some(Tensor::from_vec(vec![1.0], &self.shape()));
        for i in (0..=self.id).rev() {
            let Some(rule) = &nodes[i].backward else {
                continue; // a leaf, or a node that needs nothing: keep its slot
            };
            let Some(g) = grads[i].take() else { continue }; // freed here
            for (&p, gp) in nodes[i].parents.iter().zip(rule(&g)) {
                if nodes[p].requires_grad {
                    accumulate(&mut grads[p], gp);
                }
            }
        }
        Gradients { grads }
    }
    // @end
}

impl Gradients {
    // @listing merge
    /// The gradients of the sum of two outputs of one tape: slot by slot, add the
    /// two gradients, or keep whichever one exists.
    pub fn merge(self, other: Gradients) -> Gradients {
        let n = self.grads.len().max(other.grads.len());
        let (mut a, mut b) = (self.grads, other.grads);
        a.resize(n, None);
        b.resize(n, None);
        let sum = |slots| match slots {
            (Some(x), Some(y)) => Some(Tensor::add(&x, &y)),
            (x, y) => x.or(y),
        };
        let grads = a.into_iter().zip(b).map(sum).collect();
        Gradients { grads }
    }
    // @end

    /// The number of nodes that have a gradient.
    pub fn count(&self) -> usize {
        self.grads.iter().filter(|g| g.is_some()).count()
    }
}

impl Tape {
    // @listing add-n
    /// The sum of several tensors of one shape, recorded as a single node with one
    /// parent per input.
    pub fn add_n<'t>(&'t self, vars: &[Var<'t>]) -> Var<'t> {
        assert!(!vars.is_empty(), "add_n: no inputs");
        let shape = vars[0].shape();
        let mut sum = Tensor::zeros(&shape);
        for v in vars {
            assert_eq!(v.shape(), shape, "add_n: shapes differ");
            sum = sum.add(&v.value());
        }
        let parents = vars.iter().map(|v| v.id).collect();
        let n = vars.len();
        self.push("add_n", sum, parents, move |g| vec![g.clone(); n])
    }
    // @end

    // @listing value-bytes
    /// The bytes of tensor data the tape keeps alive, counting each storage buffer
    /// once however many nodes share it.
    pub fn value_bytes(&self) -> usize {
        let nodes = self.nodes.borrow();
        let mut seen = HashSet::new();
        let mut bytes = 0;
        for node in nodes.iter() {
            let storage = &node.value.storage;
            if seen.insert(Rc::as_ptr(storage)) {
                bytes += storage.len() * size_of::<f32>();
            }
        }
        bytes
    }
    // @end
}
