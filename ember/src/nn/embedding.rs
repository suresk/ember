//! Chapter 25: a table of learned vectors, one row per token, looked up by index.

use crate::autograd::tape::Var;
use crate::nn::{Module, Parameter};
use crate::rng::Rng;
use crate::tensor::Tensor;

// @listing embedding
/// A lookup table of `[vocab, dim]` learned vectors. Its input holds token ids,
/// whole numbers stored as `f32`; its output has one more axis, of size `dim`.
#[derive(Debug)]
pub struct Embedding {
    pub weight: Parameter,
}

impl Module for Embedding {
    fn forward<'t>(&self, ids: Var<'t>) -> Var<'t> {
        let (vocab, dim) = (self.weight.value.shape()[0], self.dim());
        let picked = token_ids(&ids.value(), vocab); // whole numbers below `vocab`
        let rows = ids.tape().param(&self.weight).index_select(&picked);
        rows.reshape(&[ids.shape(), vec![dim]].concat()) // [..., dim]
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight]
    }
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)> {
        vec![("weight".into(), &mut self.weight)]
    }
}
// @end

/// The ids in `x` as indices, checked to be whole numbers below `vocab`.
fn token_ids(x: &Tensor, vocab: usize) -> Vec<usize> {
    let id = |x: f32| {
        let ok = x >= 0.0 && x.fract() == 0.0 && (x as usize) < vocab;
        assert!(ok, "Embedding: id {x}, {vocab} rows");
        x as usize
    };
    x.to_vec().into_iter().map(id).collect()
}

impl Embedding {
    /// PyTorch's default: every entry drawn from N(0, 1).
    pub fn new(vocab: usize, dim: usize, rng: &mut Rng) -> Embedding {
        Embedding::from_tensor(Tensor::randn(&[vocab, dim], rng))
    }

    /// The number of features of each token's vector.
    pub fn dim(&self) -> usize {
        self.weight.value.shape()[1]
    }

    /// A table with the given `[vocab, dim]` rows.
    pub fn from_tensor(weight: Tensor) -> Embedding {
        assert_eq!(weight.ndim(), 2, "Embedding: weight {:?}", weight.shape());
        Embedding {
            weight: Parameter::new(weight),
        }
    }
}
