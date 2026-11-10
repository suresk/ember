//! Chapter 22: max pooling, and the two reshapes that put a convolutional network
//! between flat MNIST rows and a `Linear` head.

use crate::autograd::tape::Var;
use crate::nn::conv::{Window, four};
use crate::nn::{Module, Parameter};

// @listing max-pool
/// The largest value in each `size` × `size` window of each channel, the
/// windows `size` apart: max pooling, which halves the height and width for 2.
#[derive(Clone, Copy, Debug)]
pub struct MaxPool2d {
    pub window: Window,
}

impl Module for MaxPool2d {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let ([n, c, h, w], win) = (four(&x.shape(), "MaxPool2d"), self.window);
        let planes = x.reshape(&[n * c, 1, h, w]); // each channel on its own
        let fields = planes.im2col(win); // [N·C·Ho·Wo, k·k]
        let y = fields.max(1, false); // Chapter 9: the gradient goes to argmax
        y.reshape(&[n, c, win.out(h), win.out(w)])
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        Vec::new()
    }
}
// @end

impl MaxPool2d {
    pub fn new(size: usize) -> MaxPool2d {
        let window = Window::new(size, size, 0); // zeros could beat negative inputs
        MaxPool2d { window }
    }
}

// @listing flatten
/// `[N, …]` to `[N, rest]`: each example's values in one row, for a `Linear`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Flatten;

/// `[N, rest]` to `[N, …shape]`: each row read as a tensor of `shape`.
#[derive(Clone, Debug)]
pub struct Unflatten(pub Vec<usize>);

impl Module for Flatten {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let n = x.shape()[0];
        x.reshape(&[n, x.value().numel() / n])
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        Vec::new()
    }
}

impl Module for Unflatten {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let shape = [&[x.shape()[0]], &self.0[..]].concat();
        x.reshape(&shape)
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        Vec::new()
    }
}
// @end
