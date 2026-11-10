//! The examples printed in Chapter 22 of the guided edition: convolution defined by
//! its loops, output sizes, im2col and its adjoint col2im, the layer, max pooling,
//! shifts, and a small CNN.
use ember::{
    AdamW, Conv2d, MaxPool2d, Module, Rng, Tape, Tensor, Tolerance, Window, accuracy,
    gradcheck, train_step,
};

#[path = "../examples/mnist_cnn.rs"]
#[allow(dead_code)]
mod mnist_cnn;

use mnist_cnn::cnn;

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

// @listing definition
/// The definition, loop by loop: `x` is `[N, C, H, W]` and `w` holds `C_out`
/// kernels as PyTorch stores them, `[C_out, C, k, k]`. Output pixel `(i, j)` of
/// channel `o` is the sum of a window of `x` times kernel `o`, entry by entry.
fn correlate(x: &Tensor, w: &Tensor, win: Window) -> Tensor {
    let ([n, c, h, wd], [c_out, k]) = (dims(x), [w.shape()[0], win.size]);
    let (ho, wo) = (win.out(h), win.out(wd));
    let mut y = vec![0.0; n * c_out * ho * wo];
    let mut at = 0;
    for b in 0..n {
        for o in 0..c_out {
            for (i, j) in (0..ho).flat_map(|i| (0..wo).map(move |j| (i, j))) {
                for (ch, dy, dx) in kernel_offsets(c, k) {
                    let p = (win.input(i, dy, h), win.input(j, dx, wd));
                    if let (Some(py), Some(px)) = p {
                        y[at] += x.get(&[b, ch, py, px]) * w.get(&[o, ch, dy, dx]);
                    }
                }
                at += 1;
            }
        }
    }
    Tensor::from_vec(y, &[n, c_out, ho, wo])
}
// @end

fn dims(x: &Tensor) -> [usize; 4] {
    x.shape().try_into().expect("[N, C, H, W]")
}

/// Every (channel, row, column) of a `c` × `k` × `k` kernel, row-major.
fn kernel_offsets(c: usize, k: usize) -> impl Iterator<Item = (usize, usize, usize)> {
    (0..c * k * k).map(move |i| (i / (k * k), i / k % k, i % k))
}

/// The tiny example: a 4 × 4 `arange` image and the 2 × 2 kernel [[1, 2], [3, 4]].
fn tiny() -> (Tensor, Tensor) {
    let x = Tensor::arange(&[1, 1, 4, 4]);
    (x, t(&[1., 2., 3., 4.], &[1, 1, 2, 2]))
}

#[test]
fn sliding_a_kernel() {
    // @listing slide-test
    let (x, w) = tiny();
    let y = correlate(&x, &w, Window::new(2, 1, 0)); // stride 1, no padding
    assert_eq!(y.shape(), [1, 1, 3, 3]);
    let rows = [[34., 44., 54.], [74., 84., 94.], [114., 124., 134.]];
    assert_eq!(y.to_vec(), rows.concat()); // 40i + 10j + 34
    let flipped = t(&[4., 3., 2., 1.], &[1, 1, 2, 2]); // a true convolution
    let z = correlate(&x, &flipped, Window::new(2, 1, 0));
    assert_eq!(z.to_vec()[..3], [16., 26., 36.]); // 40i + 10j + 16
    // @end
}

#[test]
fn output_sizes() {
    // @listing sizes-test
    let out = |k, s, p, n| Window::new(k, s, p).out(n);
    assert_eq!(out(3, 1, 0, 28), 26); // no padding: the map shrinks by k − 1
    assert_eq!(out(3, 1, 1, 28), 28); // padding (k − 1)/2: "same" size
    assert_eq!(out(5, 1, 2, 28), 28);
    assert_eq!(out(2, 2, 0, 28), 14); // 2 × 2 windows, 2 apart: half
    assert_eq!(out(2, 2, 0, 7), 3); // ⌊5/2⌋ + 1: the last row is dropped
    assert_eq!(out(3, 2, 1, 7), 4); // ⌊6/2⌋ + 1
    // @end
    let w = Window::new(3, 1, 1);
    assert_eq!((w.input(0, 0, 28), w.input(0, 1, 28)), (None, Some(0)));
    assert_eq!(w.input(27, 2, 28), None); // 27 + 2 − 1 = 28, past the edge
}

#[test]
fn the_unfold_view() {
    // @listing unfold-test
    let x = Tensor::arange(&[4, 4]);
    let fields = x.unfold(0, 2, 1).unfold(1, 2, 1); // [3, 3, 2, 2]: no copy
    assert_eq!(fields.strides(), [4, 1, 4, 1]);
    let window = fields.select(0, 1).select(0, 2); // output pixel [1, 2]
    assert_eq!(window.to_vec(), [6., 7., 10., 11.]);
    assert!(fields.shares_storage(&x) && !fields.is_contiguous());
    let rows = fields.reshape(&[9, 4]); // one window per row: now a copy
    assert!(!rows.shares_storage(&x));
    // @end
}

#[test]
fn the_im2col_matrix() {
    // @listing im2col-test
    let (x, w) = tiny();
    let cols = x.im2col(Window::new(2, 1, 0));
    assert_eq!(cols.shape(), [9, 4]); // 9 windows of 4 pixels
    assert_eq!(cols.select(0, 0).to_vec(), [0., 1., 4., 5.]);
    assert_eq!(cols.select(0, 4).to_vec(), [5., 6., 9., 10.]);
    let y = cols.matmul(&w.reshape(&[4, 1])); // the kernel as a column
    assert_eq!(y.to_vec(), correlate(&x, &w, Window::new(2, 1, 0)).to_vec());
    let padded = x.im2col(Window::new(3, 1, 1)); // 3 × 3, padding 1
    assert_eq!(padded.shape(), [16, 9]);
    let corner = padded.select(0, 0).to_vec(); // the window at the top left
    assert_eq!(corner, [0., 0., 0., 0., 0., 1., 0., 4., 5.]);
    // @end
}

#[test]
fn im2col_by_views() {
    // im2col without padding is unfold twice, a permute and a copy.
    let mut rng = Rng::new(22);
    let x = Tensor::randn(&[2, 3, 7, 6], &mut rng);
    for (k, s) in [(3, 1), (2, 2), (3, 2)] {
        let v = x.unfold(2, k, s).unfold(3, k, s); // [N, C, Ho, Wo, k, k]
        let v = v.permute(&[0, 2, 3, 1, 4, 5]); // [N, Ho, Wo, C, k, k]
        let rows = v
            .contiguous()
            .reshape(&[v.numel() / (3 * k * k), 3 * k * k]);
        assert_eq!(rows.to_vec(), x.im2col(Window::new(k, s, 0)).to_vec());
    }
    let x = Tensor::arange(&[4, 4]);
    let rows = x.unfold(0, 2, 1).unfold(1, 2, 1).reshape(&[9, 4]);
    let x = x.reshape(&[1, 1, 4, 4]);
    assert_eq!(rows.to_vec(), x.im2col(Window::new(2, 1, 0)).to_vec());
}

#[test]
fn convolution_as_one_product() {
    // @listing layer-test
    let mut rng = Rng::new(22);
    let x = Tensor::randn(&[2, 3, 5, 6], &mut rng); // N = 2, C = 3, 5 × 6
    let torch_w = Tensor::randn(&[4, 3, 3, 3], &mut rng); // 4 kernels, PyTorch's way
    let w = torch_w.reshape(&[4, 27]).transpose(0, 1); // ember's: [27, 4]
    let (bias, win) = (t(&[1., 2., 3., 4.], &[4]), Window::new(3, 2, 1));
    let layer = Conv2d::from_tensors(w.contiguous(), bias.clone(), win);
    let tape = Tape::new();
    let y = layer.forward(tape.constant(x.clone())).value();
    assert_eq!(y.shape(), [2, 4, 3, 3]); // ⌊(5 + 2 − 3)/2⌋ + 1, ⌊(6 + 2 − 3)/2⌋ + 1
    let want = correlate(&x, &torch_w, win).add(&bias.reshape(&[4, 1, 1]));
    assert!(y.allclose_rel(&want, 1e-5, 1e-5));
    assert_eq!(tape.len(), 9); // 4 for the product, 3 to reshape, 2 for the bias
    // @end
}

#[test]
fn what_im2col_costs() {
    // @listing cost-test
    let mb = |floats: usize| floats as f64 * 4.0 / 1e6; // f32: 4 bytes each
    let (input, win) = ([64, 32, 28, 28], Window::new(3, 1, 1));
    let x: usize = input.iter().product();
    let [rows, cols] = win.matrix(input);
    assert_eq!((x, mb(x)), (1_605_632, 6.422528)); // 64 maps of 32 × 28 × 28
    assert_eq!((rows, cols), (50_176, 288)); // 64·28·28 windows, 32·3·3 each
    assert_eq!((rows * cols, mb(rows * cols)), (14_450_688, 57.802752)); // 9×
    let macs = rows * cols * 32; // the product with [288, 32]
    assert_eq!(macs, 462_422_016);
    // @end
    let small = [64, 1, 28, 28];
    assert_eq!(win.matrix(small), [50_176, 9]);
}

#[test]
fn col2im_adds_the_overlaps() {
    let win = Window::new(2, 1, 0);
    let ones = Tensor::from_vec(vec![1.0; 36], &[9, 4]); // 9 windows × 4 reads
    let reads = ones.col2im(&[1, 1, 4, 4], win);
    let count = [
        1., 2., 2., 1., 2., 4., 4., 2., 2., 4., 4., 2., 1., 2., 2., 1.,
    ];
    assert_eq!(reads.to_vec(), count); // how many windows read each pixel
    // @listing col2im-test
    let (x, w) = tiny();
    let tape = Tape::new();
    let (xv, wv) = (tape.leaf(x), tape.leaf(w.reshape(&[4, 1])));
    let grads = xv.conv2d(wv, Window::new(2, 1, 0)).sum_all().backward();
    let x_bar = grads.get(xv).unwrap(); // [1, 1, 4, 4]
    assert_eq!(x_bar.select(2, 1).to_vec(), [4., 10., 10., 6.]); // its row 1
    assert_eq!(x_bar.sum_all(), 90.0); // 9 windows, each passing 1 + 2 + 3 + 4
    assert_eq!(grads.get(wv).unwrap().to_vec(), [45., 54., 81., 90.]);
    // @end
    let all = [
        1., 3., 3., 2., 4., 10., 10., 6., 4., 10., 10., 6., 3., 7., 7., 4.,
    ];
    assert_eq!(x_bar.to_vec(), all); // Σ w over the reads
}

#[test]
fn col2im_is_the_adjoint() {
    // ⟨im2col(x), m⟩ = ⟨x, col2im(m)⟩ for any x and m, padding and stride included.
    let mut rng = Rng::new(22);
    let x = Tensor::randn(&[2, 3, 6, 5], &mut rng);
    let win = Window::new(3, 2, 1);
    let cols = x.im2col(win);
    let m = Tensor::randn(cols.shape(), &mut rng);
    let left = cols.mul(&m).sum_all();
    let right = x.mul(&m.col2im(x.shape(), win)).sum_all();
    assert!((left - right).abs() < 1e-4, "{left} vs {right}");
}

#[test]
fn convolution_passes_gradcheck() {
    // @listing gradcheck-test
    let mut rng = Rng::new(22);
    let x = Tensor::randn(&[2, 2, 5, 4], &mut rng);
    let w = Tensor::randn(&[2 * 3 * 3, 3], &mut rng); // 2 channels in, 3 out
    let tol = Tolerance::default();
    for (stride, padding) in [(1, 1), (2, 0), (2, 2)] {
        let win = Window::new(3, stride, padding);
        let im2col = gradcheck(|v| v[0].im2col(win), std::slice::from_ref(&x), tol);
        let xw = [x.clone(), w.clone()];
        let conv = gradcheck(|v| v[0].conv2d(v[1], win), &xw, tol);
        assert!(im2col.is_ok() && conv.is_ok(), "{im2col:?} {conv:?}");
    }
    // @end
}

#[test]
fn max_pooling_routes_to_the_argmax() {
    // @listing pool-test
    let x = [
        1., 5., 2., 0., 3., 4., 8., 6., 7., 0., 1., 2., 9., 3., 4., 4.,
    ];
    let tape = Tape::new();
    let xv = tape.leaf(t(&x, &[1, 1, 4, 4])); // the 4 × 4 image row by row
    let y = MaxPool2d::new(2).forward(xv);
    assert_eq!(y.value().to_vec(), [5., 8., 9., 4.]);
    let g = tape.constant(t(&[10., 20., 30., 40.], &[1, 1, 2, 2]));
    let grads = (y * g).sum_all().backward();
    let x_bar = grads.get(xv).unwrap().to_vec();
    let at: Vec<usize> = (0..16).filter(|&i| x_bar[i] != 0.0).collect();
    assert_eq!(at, [1, 6, 12, 14]); // [0, 1], [1, 2], [3, 0] and the first 4
    assert_eq!(
        [x_bar[1], x_bar[6], x_bar[12], x_bar[14]],
        [10., 20., 30., 40.]
    );
    // @end
}

/// A 6 × 8 image, all background, with a bright L-shaped stroke at `(row, col)`.
fn stroke(row: usize, col: usize) -> Tensor {
    let mut x = Tensor::zeros(&[1, 1, 6, 8]);
    for (dy, dx) in [(0, 0), (1, 0), (2, 0), (2, 1)] {
        x.data_mut()[(row + dy) * 8 + col + dx] = 1.0;
    }
    x
}

#[test]
fn a_shift_moves_the_map() {
    // @listing shift-test
    let mut rng = Rng::new(22);
    let conv = Conv2d::new(1, 4, Window::new(3, 1, 1), &mut rng);
    let maps = |x: Tensor| {
        let tape = Tape::new();
        conv.forward(tape.constant(x)).value() // [1, 4, 6, 8]
    };
    let (a, b) = (maps(stroke(1, 2)), maps(stroke(1, 3))); // one pixel right
    let moved = b.slice(3, 1, 8, 1).to_vec(); // b's columns 1..8
    assert_eq!(moved, a.slice(3, 0, 7, 1).to_vec()); // a's columns 0..7: equal
    // @end
    // Max pooling: a shift by 2 moves the pooled map by 1; a shift by 1 doesn't.
    let pooled = |x: Tensor| {
        let tape = Tape::new();
        let y = conv.forward(tape.constant(x));
        MaxPool2d::new(2).forward(y).value()
    };
    let (p, q) = (pooled(stroke(1, 2)), pooled(stroke(1, 4)));
    assert_eq!(q.slice(3, 1, 4, 1).to_vec(), p.slice(3, 0, 3, 1).to_vec());
}

#[test]
fn the_cnn() {
    // @listing cnn-test
    let mut model = cnn(&mut Rng::new(20));
    let params = model.named_parameters();
    let sizes: Vec<usize> = params.iter().map(|(_, p)| p.value.numel()).collect();
    assert_eq!(sizes, [72, 8, 1152, 16, 7840, 10]);
    assert_eq!(sizes.iter().sum::<usize>(), 9_098); // the MLP: 101,770
    let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "1.weight", "1.bias", "4.weight", "4.bias", "8.weight", "8.bias"
        ]
    );
    let tape = Tape::new();
    let x = tape.constant(Tensor::zeros(&[2, 784])); // two flat MNIST rows
    let shapes: Vec<Vec<usize>> = (model.layers.iter())
        .scan(x, |x, layer| {
            *x = layer.forward(*x);
            Some(x.shape())
        })
        .collect();
    assert_eq!(shapes[3], [2, 8, 14, 14]); // after the first pooling
    assert_eq!(shapes[6], [2, 16, 7, 7]);
    assert_eq!(shapes[8], [2, 10]);
    // @end
    // Multiply-adds per image: each output pixel of each map reads C·k·k inputs.
    let conv1 = 28 * 28 * 9 * 8;
    let conv2 = 14 * 14 * 72 * 16;
    assert_eq!(
        (conv1, conv2, conv1 + conv2 + 7840),
        (56_448, 225_792, 290_080)
    );
    assert_eq!(784 * 128 + 128 * 10, 101_632); // the MLP's
}

/// `n` synthetic 28 × 28 digits: class 0 a horizontal bar, class 1 a vertical
/// one, each 3 pixels wide and 14 long, at a random place.
fn bars(n: usize, rng: &mut Rng) -> (Tensor, Vec<usize>) {
    let mut x = vec![0.0; n * 784];
    let labels: Vec<usize> = (0..n).map(|i| i % 2).collect();
    for (image, &label) in x.as_chunks_mut::<784>().0.iter_mut().zip(&labels) {
        let (a, b) = (rng.below(12) + 2, rng.below(23) + 1); // where the bar starts
        for (along, across) in (0..14).flat_map(|i| (0..3).map(move |j| (i, j))) {
            let (y, x) = if label == 0 {
                (b + across, a + along)
            } else {
                (a + along, b + across)
            };
            image[y * 28 + x] = 1.0;
        }
    }
    (Tensor::from_vec(x, &[n, 784]), labels)
}

#[test]
fn the_cnn_learns_bars() {
    // @listing bars-test
    let mut rng = Rng::new(22);
    let mut model = cnn(&mut rng);
    let batch = bars(8, &mut rng); // four of each, at random places
    let mut opt = AdamW::new(3e-3);
    let mut step = || train_step(&mut model, &batch, &mut opt, 1.0);
    let losses: Vec<f32> = (0..5).map(|_| (step() * 1e4).round() / 1e4).collect();
    assert_eq!(losses, [2.3518, 1.6824, 1.1197, 0.6804, 0.3859]);
    let (x, labels) = bars(8, &mut rng); // eight more, at other places
    let tape = Tape::new();
    let logits = tape.no_grad(|| model.forward(tape.constant(x)));
    assert_eq!(accuracy(&logits.value(), &labels), 0.875); // seven of eight
    // @end
}

#[test]
#[should_panic(expected = "im2col: shape [3, 4], expected [N, C, H, W]")]
fn im2col_needs_four_axes() {
    Tensor::arange(&[3, 4]).im2col(Window::new(2, 1, 0));
}
