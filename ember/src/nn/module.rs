//! Chapter 11: the trait every layer, loss-free model and composite implements.

use crate::autograd::tape::Var;
use crate::nn::Parameter;
use crate::tensor::Tensor;

// @listing module
/// A piece of a model: a computation, and the parameters it reads.
pub trait Module {
    /// Run the computation on `x`'s tape. The result borrows the tape, not the
    /// module, so the module is free again as soon as `forward` returns.
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t>;

    /// Every parameter, each exactly once, in a fixed order.
    fn parameters(&mut self) -> Vec<&mut Parameter>;

    /// Forget the gradients collected so far.
    fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.grad = None;
        }
    }

    /// Chapter 20: every parameter with a name, by default its position.
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)> {
        let numbered = self.parameters().into_iter().enumerate();
        numbered.map(|(i, p)| (i.to_string(), p)).collect()
    }

    /// Chapter 21: switch to training (`true`) or evaluation mode. Only modules
    /// that behave differently in the two, and composites, override it.
    fn train(&mut self, _on: bool) {}

    /// Chapter 23: state that is saved but not trained, such as batch norm's
    /// running statistics, with names. Most modules have none.
    fn named_buffers(&mut self) -> Vec<(String, &mut Tensor)> {
        Vec::new()
    }
}
// @end
