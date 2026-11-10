//! Chapter 22: convolution by im2col. Every receptive field of the input is copied
//! into a row of a matrix, so that the convolution is one matrix product; the
//! backward pass scatters the rows back with col2im, adding where windows overlap.

use crate::autograd::tape::Var;
use crate::nn::{Module, Parameter, init};
use crate::rng::Rng;
use crate::tensor::Tensor;

// @listing window
/// A square window sliding over an image: `size` × `size`, moved `stride`
/// pixels at a time, over the image framed by `padding` pixels of zeros.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Window {
    /// How many positions fit along an axis of `n` pixels: ⌊(n + 2p − k)/s⌋ + 1.
    pub fn out(&self, n: usize) -> usize {
        let padded = n + 2 * self.padding;
        assert!(padded >= self.size, "Window: {} pixels for {self:?}", n);
        (padded - self.size) / self.stride + 1
    }
    /// The input pixel that output position `o` reads at offset `i` of the
    /// window, or `None` if it is in the padding.
    pub fn input(&self, o: usize, i: usize, n: usize) -> Option<usize> {
        let padded = o * self.stride + i; // a position in the padded image
        padded.checked_sub(self.padding).filter(|&at| at < n)
    }
}
// @end

impl Window {
    pub fn new(size: usize, stride: usize, padding: usize) -> Window {
        assert!(
            size > 0 && stride > 0,
            "Window: size {size}, stride {stride}"
        );
        Window {
            size,
            stride,
            padding,
        }
    }
}

/// The four sizes of an `[N, C, H, W]` shape.
pub(crate) fn four(shape: &[usize], op: &str) -> [usize; 4] {
    match *shape {
        [n, c, h, w] => [n, c, h, w],
        _ => panic!("{op}: shape {shape:?}, expected [N, C, H, W]"),
    }
}

impl Window {
    /// The shape of the im2col matrix of an input of `shape`: one row per
    /// receptive field, `[N·Ho·Wo, C·k·k]`.
    pub fn matrix(&self, [n, c, h, w]: [usize; 4]) -> [usize; 2] {
        [n * self.out(h) * self.out(w), c * self.size * self.size]
    }
}

// @listing walk
/// Calls `f(m, x)` for every entry of the im2col matrix of an input of `shape`
/// that reads a pixel: `m` is the entry's row-major position in the matrix,
/// `x` the pixel's in the input. Entries in the padding are skipped.
fn each_read(shape: [usize; 4], win: Window, mut f: impl FnMut(usize, usize)) {
    let [n, c, h, w] = shape;
    let (ho, wo, k) = (win.out(h), win.out(w), win.size);
    let mut m = 0; // the entries come in row-major order
    for row in 0..n * ho * wo {
        let (image, oy, ox) = (row / (ho * wo), row / wo % ho, row % wo);
        for channel in 0..c {
            let plane = (image * c + channel) * h * w; // where the channel starts
            for ky in 0..k {
                let y = win.input(oy, ky, h);
                for kx in 0..k {
                    if let (Some(y), Some(x)) = (y, win.input(ox, kx, w)) {
                        f(m, plane + y * w + x);
                    }
                    m += 1;
                }
            }
        }
    }
}
// @end

// @listing im2col
impl Tensor {
    /// Every receptive field of `self`, `[N, C, H, W]`, as a row of a matrix
    /// `[N·Ho·Wo, C·k·k]`: channel by channel, each one's k × k pixels row by row.
    pub fn im2col(&self, win: Window) -> Tensor {
        let shape = four(self.shape(), "im2col");
        let x = self.contiguous(); // a copy only if `self` is strided
        let x = x.as_slice().unwrap();
        let [rows, cols] = win.matrix(shape);
        let mut m = vec![0.0; rows * cols]; // the padding stays 0
        each_read(shape, win, |at, from| m[at] = x[from]);
        Tensor::from_vec(m, &[rows, cols])
    }
    /// The adjoint of `im2col`: every entry of `self` added into the pixel of an
    /// input of `shape` that it was read from.
    pub fn col2im(&self, shape: &[usize], win: Window) -> Tensor {
        let dims = four(shape, "col2im");
        assert_eq!(self.shape(), win.matrix(dims), "col2im: for {shape:?}");
        let m = self.contiguous();
        let m = m.as_slice().unwrap();
        let mut x = vec![0.0; shape.iter().product()];
        each_read(dims, win, |from, at| x[at] += m[from]); // overlaps add up
        Tensor::from_vec(x, shape)
    }
}
// @end

// @listing conv-var
impl<'t> Var<'t> {
    /// `im2col` on the tape. Its backward is `col2im`: a pixel read by several
    /// windows collects the gradients of all of their entries.
    pub fn im2col(self, win: Window) -> Var<'t> {
        let f = move |x: &Tensor| x.im2col(win);
        self.unary("im2col", f, move |x, _, g| g.col2im(x.shape(), win))
    }
    /// The convolution of `self`, `[N, C, H, W]`, with `weight`, `[C·k·k,
    /// C_out]`: one matrix product, read back as `[N, C_out, Ho, Wo]`.
    pub fn conv2d(self, weight: Var<'t>, win: Window) -> Var<'t> {
        let [n, _, h, w] = four(&self.shape(), "conv2d");
        let c_out = weight.shape()[1];
        let rows = self.im2col(win).matmul(weight); // [N·Ho·Wo, C_out]
        let y = rows.reshape(&[n, win.out(h), win.out(w), c_out]);
        y.permute(&[0, 3, 1, 2]) // channels first, as a view
    }
}
// @end

// @listing conv2d
/// A convolution layer: `c_out` kernels of `c_in` × k × k weights, stored as
/// the columns of `weight`, `[c_in·k·k, c_out]`, and one bias per kernel.
#[derive(Debug)]
pub struct Conv2d {
    pub weight: Parameter,
    pub bias: Parameter,
    pub window: Window,
}

impl Conv2d {
    /// He initialization: each kernel reads `fan_in = c_in·k·k` inputs.
    pub fn new(c_in: usize, c_out: usize, window: Window, rng: &mut Rng) -> Conv2d {
        let fan_in = c_in * window.size * window.size;
        let weight = init::normal(&[fan_in, c_out], init::he_std(fan_in), rng);
        Conv2d::from_tensors(weight, Tensor::zeros(&[c_out]), window)
    }
}

impl Module for Conv2d {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let tape = x.tape();
        let c_out = self.bias.value.numel();
        let bias = tape.param(&self.bias).reshape(&[c_out, 1, 1]); // per channel
        x.conv2d(tape.param(&self.weight), self.window) + bias
    }
    fn parameters(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight, &mut self.bias]
    }
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)> {
        let (w, b) = (&mut self.weight, &mut self.bias);
        vec![("weight".into(), w), ("bias".into(), b)]
    }
}
// @end

impl Conv2d {
    /// A layer with the given weight, `[c_in·k·k, c_out]`, and bias, `[c_out]`.
    pub fn from_tensors(weight: Tensor, bias: Tensor, window: Window) -> Conv2d {
        assert!(weight.ndim() == 2 && bias.shape() == [weight.shape()[1]]);
        let (weight, bias) = (Parameter::new(weight), Parameter::new(bias));
        Conv2d {
            weight,
            bias,
            window,
        }
    }
}
