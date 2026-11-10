//! Chapter 11: a parameter is a tensor with an identity that lasts from one training
//! step to the next, and the gradient collected for it.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::tensor::Tensor;

// @listing param-id
/// Which parameter a gradient belongs to: a number no other parameter in the
/// process has, whichever thread made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamId(usize);

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

impl ParamId {
    fn next() -> ParamId {
        ParamId(NEXT_ID.fetch_add(1, Ordering::Relaxed)) // read and add, as one step
    }
}
// @end

// @listing parameter
/// A weight of a model: its value, owned, and the gradient collected for it so
/// far. `grad` is `None` until a backward pass reaches it.
#[derive(Debug)]
pub struct Parameter {
    id: ParamId,
    pub value: Tensor,
    pub grad: Option<Tensor>,
}

impl Parameter {
    pub fn new(value: Tensor) -> Parameter {
        let (id, grad) = (ParamId::next(), None);
        Parameter { id, value, grad }
    }

    pub fn id(&self) -> ParamId {
        self.id
    }
}
// @end

// @listing add-grad
impl Parameter {
    /// Add `g` into `grad`, keeping `grad` a contiguous tensor.
    pub(crate) fn add_grad(&mut self, g: Tensor) {
        let sum = match self.grad.take() {
            Some(sum) => sum.add(&g), // a new tensor
            None => g.contiguous(),   // kept as it is, or copied if strided
        };
        self.grad = Some(sum);
    }
}
// @end

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;
    use std::rc::Rc;

    /// Section 11.2: the tape shares the parameter's buffer while it lives, and only
    /// while it lives.
    #[test]
    fn the_tape_shares_the_buffer_for_one_step() {
        let p = Parameter::new(Tensor::arange(&[2, 3]));
        assert_eq!(Rc::strong_count(&p.value.storage), 1);
        {
            let tape = Tape::new();
            let w = tape.param(&p);
            assert!(w.value().shares_storage(&p.value));
            assert_eq!(Rc::strong_count(&p.value.storage), 2); // p and the node
        }
        assert_eq!(Rc::strong_count(&p.value.storage), 1); // p's own again
    }

    #[test]
    fn a_parameter_is_136_bytes() {
        assert_eq!(size_of::<ParamId>(), 8);
        assert_eq!(size_of::<Option<Tensor>>(), 64); // the Rc can't be null
        assert_eq!(size_of::<Parameter>(), 136);
    }
}
