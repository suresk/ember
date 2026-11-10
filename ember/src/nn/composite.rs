//! Chapter 25: one macro for the four methods a module made of named child
//! modules must pass on to its children, and lists of modules of one type.

use crate::autograd::tape::Var;
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

// @listing children
/// Inside `impl Module for S`: `parameters`, `named_parameters`, `train` and
/// `named_buffers` for a struct whose fields `a, b, …` are modules, each
/// passed on to every field in order, names prefixed with `"a."`, `"b."`, ….
#[macro_export]
macro_rules! children {
    ($($field:ident),+) => {
        fn parameters(&mut self) -> Vec<&mut $crate::Parameter> {
            let mut all = Vec::new();
            $(all.extend(self.$field.parameters());)+
            all
        }
        fn named_parameters(&mut self) -> Vec<(String, &mut $crate::Parameter)> {
            let mut all = Vec::new();
            $(let dot = |(n, p)| (format!("{}.{n}", stringify!($field)), p);
            all.extend(self.$field.named_parameters().into_iter().map(dot));)+
            all
        }
        fn train(&mut self, on: bool) {
            $(self.$field.train(on);)+
        }
        fn named_buffers(&mut self) -> Vec<(String, &mut $crate::Tensor)> {
            let mut all = Vec::new();
            $(let dot = |(n, t)| (format!("{}.{n}", stringify!($field)), t);
            all.extend(self.$field.named_buffers().into_iter().map(dot));)+
            all
        }
    };
}
// @end

// @listing vec-module
/// A list of modules of one type, applied in order: `Sequential` without the
/// boxes, so each element keeps its type. Names are `"0.…"`, `"1.…"`, ….
impl<M: Module> Module for Vec<M> {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        self.iter().fold(x, |x, m| m.forward(x))
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        self.iter_mut().flat_map(|m| m.parameters()).collect()
    }
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)> {
        let mut all = vec![];
        for (i, m) in self.iter_mut().enumerate() {
            let named = m.named_parameters().into_iter();
            all.extend(named.map(|(n, p)| (format!("{i}.{n}"), p)));
        }
        all
    }
    fn train(&mut self, on: bool) {
        self.iter_mut().for_each(|m| m.train(on));
    }
    fn named_buffers(&mut self) -> Vec<(String, &mut Tensor)> {
        let mut all = vec![];
        for (i, m) in self.iter_mut().enumerate() {
            let named = m.named_buffers().into_iter();
            all.extend(named.map(|(n, t)| (format!("{i}.{n}"), t)));
        }
        all
    }
}
// @end
