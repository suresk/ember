//! Chapter 22 challenges.

use crate::autograd::tape::Var;
use crate::challenges::ChannelDropout;
use crate::nn::{Dropout, Linear, Module, Parameter, Sequential, Window};

// @listing pointwise
/// A `Linear` layer applied at every pixel of `x`, `[N, C, H, W]`: the channels
/// of each pixel are its inputs. This is a 1 × 1 convolution.
pub fn pointwise<'t>(x: Var<'t>, linear: &Linear) -> Var<'t> {
    let pixels = x.permute(&[0, 2, 3, 1]); // [N, H, W, C]: channels last
    let y = linear.forward(pixels); // Linear folds the leading axes
    y.permute(&[0, 3, 1, 2]) // [N, C_out, H, W]
}
// @end

// @listing avg-pool
/// The mean of each `size` × `size` window of each channel, the windows `size`
/// apart.
#[derive(Clone, Copy, Debug)]
pub struct AvgPool2d {
    pub window: Window,
}

/// The mean of each channel over all of its pixels: `[N, C, H, W]` to `[N, C]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlobalAvgPool;

impl Module for AvgPool2d {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let (s, win) = (x.shape(), self.window);
        let planes = x.reshape(&[s[0] * s[1], 1, s[2], s[3]]);
        let y = planes.im2col(win).mean(1, false); // the gradient: 1/k² each
        y.reshape(&[s[0], s[1], win.out(s[2]), win.out(s[3])])
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        Vec::new()
    }
}

impl Module for GlobalAvgPool {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let s = x.shape();
        x.reshape(&[s[0], s[1], s[2] * s[3]]).mean(2, false)
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        Vec::new()
    }
}
// @end

impl AvgPool2d {
    pub fn new(size: usize) -> AvgPool2d {
        AvgPool2d {
            window: Window::new(size, size, 0),
        }
    }
}

// @listing receptive-field
/// The side of the square of input pixels that one output pixel of a stack of
/// windows can see: each window widens it by `k − 1` steps of the layers
/// below, and each stride multiplies the step.
pub fn receptive_field(stack: &[Window]) -> usize {
    let (mut side, mut step) = (1, 1); // in input pixels
    for w in stack {
        side += (w.size - 1) * step;
        step *= w.stride;
    }
    side
}
// @end

// @listing channel-dropout-cnn
/// `model` with a `ChannelDropout(p)` inserted after each of the layers at
/// `after`, each with its own seed, `seed`, `seed + 1`, …
pub fn with_channel_dropout(
    mut model: Sequential,
    after: &[usize],
    p: f32,
    seed: u64,
) -> Sequential {
    for (i, &at) in after.iter().enumerate().rev() {
        let drop = ChannelDropout(Dropout::new(p, seed + i as u64));
        model.layers.insert(at + 1, Box::new(drop)); // last first: no shifting
    }
    model
}
// @end

// @listing depthwise
/// Depthwise convolution: channel `c` of `x`, `[N, C, H, W]`, convolved with
/// its own k × k kernel, row `c` of `weight`, `[C, k·k]`, and nothing else.
pub fn depthwise_conv2d<'t>(x: Var<'t>, weight: Var<'t>, win: Window) -> Var<'t> {
    let [n, c, h, w] = <[usize; 4]>::try_from(x.shape()).expect("[N, C, H, W]");
    let (ho, wo, kk) = (win.out(h), win.out(w), win.size * win.size);
    let fields = x.reshape(&[n * c, 1, h, w]).im2col(win); // [N·C·Ho·Wo, k·k]
    let fields = fields.reshape(&[n, c, ho * wo, kk]);
    let products = fields * weight.reshape(&[c, 1, kk]); // each channel's kernel
    products.sum(3, false).reshape(&[n, c, ho, wo])
}
// @end

// @listing transposed
impl<'t> Var<'t> {
    /// `col2im` on the tape, for an image of `shape`; its backward is `im2col`.
    pub fn col2im(self, shape: &[usize], win: Window) -> Var<'t> {
        let f = |m: &crate::Tensor| m.col2im(shape, win);
        self.unary("col2im", f, move |_, _, g| g.im2col(win))
    }
}

/// Transposed convolution: every pixel of `x`, `[N, C, H, W]`, times `weight`,
/// `[C, C_out·k·k]`, is a k × k patch of `C_out` channels, and the patches are
/// added into the output where they overlap: the adjoint of `conv2d`.
pub fn conv_transpose2d<'t>(x: Var<'t>, weight: Var<'t>, win: Window) -> Var<'t> {
    let [n, c, h, w] = <[usize; 4]>::try_from(x.shape()).expect("[N, C, H, W]");
    let c_out = weight.shape()[1] / (win.size * win.size);
    let size = |n: usize| (n - 1) * win.stride + win.size - 2 * win.padding;
    let rows = x.permute(&[0, 2, 3, 1]).reshape(&[n * h * w, c]);
    let patches = rows.matmul(weight); // [N·H·W, C_out·k·k]
    patches.col2im(&[n, c_out, size(h), size(w)], win)
}
// @end
