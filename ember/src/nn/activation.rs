//! Chapter 12: the activations as modules, so that they can sit in a `Sequential`.

use crate::autograd::tape::Var;
use crate::nn::{Module, Parameter};

// @listing activation
/// A module with no parameters that applies one of `Var`'s activations.
macro_rules! activation {
    ($name:ident, $method:ident) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl Module for $name {
            fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
                x.$method()
            }

            fn parameters(&mut self) -> Vec<&mut Parameter> {
                Vec::new() // nothing to train
            }
        }
    };
}

activation!(ReLU, relu);
activation!(Tanh, tanh);
activation!(Sigmoid, sigmoid);
// @end
