//! Chapter 12: a list of modules, run one after another.

use crate::autograd::tape::Var;
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

// @listing sequential
/// Modules applied in order, each to the output of the one before.
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Sequential {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        self.layers.iter().fold(x, |x, layer| layer.forward(x))
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        let layers = self.layers.iter_mut(); // one `&mut Box` at a time
        layers.flat_map(|m| m.parameters()).collect()
    }
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)> {
        let mut named = vec![]; // Chapter 20: "0.weight", "0.bias", "2.weight", …
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let dotted = |(name, p)| (format!("{i}.{name}"), p);
            named.extend(layer.named_parameters().into_iter().map(dotted));
        }
        named
    }
    fn train(&mut self, on: bool) {
        self.layers.iter_mut().for_each(|layer| layer.train(on)); // Chapter 21
    }
    fn named_buffers(&mut self) -> Vec<(String, &mut Tensor)> {
        let mut named = vec![]; // Chapter 23: "1.running_mean", …
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let dotted = |(name, t)| (format!("{i}.{name}"), t);
            named.extend(layer.named_buffers().into_iter().map(dotted));
        }
        named
    }
}
// @end
