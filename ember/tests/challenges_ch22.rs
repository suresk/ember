//! Tests for the Chapter 22 challenges.
use ember::{
    AvgPool2d, Conv2d, Flatten, GlobalAvgPool, Linear, MaxPool2d, Module, ReLU, Rng,
    Sequential, Tape, Tensor, Tolerance, Window, conv_transpose2d, depthwise_conv2d,
    eval_is_deterministic, gradcheck, pointwise, receptive_field,
    with_channel_dropout,
};

#[path = "../examples/mnist_cnn.rs"]
#[allow(dead_code)]
mod mnist_cnn;

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn count(model: &mut dyn Module) -> usize {
    model.parameters().iter().map(|p| p.value.numel()).sum()
}

#[test]
fn predict_conv_shapes() {
    // @listing predict-test
    let mut rng = Rng::new(22);
    let mut model = Sequential::new(vec![
        Box::new(Conv2d::new(3, 16, Window::new(5, 2, 2), &mut rng)),
        Box::new(MaxPool2d::new(2)),
        Box::new(Conv2d::new(16, 32, Window::new(3, 1, 0), &mut rng)),
        Box::new(Flatten),
        Box::new(Linear::new(32 * 6 * 6, 10, &mut rng)),
    ]);
    let sizes: Vec<_> = model.parameters().iter().map(|p| p.value.numel()).collect();
    assert_eq!(sizes, [1200, 16, 4608, 32, 11_520, 10]); // 17,386 in all
    let tape = Tape::new();
    let mut x = tape.constant(Tensor::zeros(&[8, 3, 32, 32]));
    let mut shapes = vec![];
    for layer in &model.layers {
        x = layer.forward(x);
        shapes.push(x.shape());
    }
    assert_eq!(shapes[0], [8, 16, 16, 16]); // ⌊(32 + 4 − 5)/2⌋ + 1 = 16
    assert_eq!(shapes[1], [8, 16, 8, 8]);
    assert_eq!(shapes[2], [8, 32, 6, 6]); // no padding: 8 − 2
    assert_eq!(shapes[3], [8, 1152]);
    assert_eq!(shapes[4], [8, 10]);
    let first = Tensor::zeros(&[8, 3, 32, 32]).im2col(Window::new(5, 2, 2));
    assert_eq!(first.shape(), [2048, 75]); // 8·16·16 windows of 3·5·5
    // @end
}

#[test]
fn one_by_one_is_linear() {
    // @listing pointwise-test
    let mut rng = Rng::new(22);
    let linear = Linear::new(3, 4, &mut rng); // 3 channels in, 4 out
    let (w, b) = (linear.weight.value.clone(), linear.bias.value.clone());
    let conv = Conv2d::from_tensors(w, b, Window::new(1, 1, 0)); // the same tensors
    let x = Tensor::randn(&[2, 3, 5, 5], &mut rng);
    let tape = Tape::new();
    let a = pointwise(tape.constant(x.clone()), &linear).value();
    let c = conv.forward(tape.constant(x)).value();
    assert_eq!(a.shape(), [2, 4, 5, 5]);
    assert!(a.allclose_rel(&c, 1e-6, 1e-5)); // one map, two routes
    // @end
}

#[test]
fn average_pooling() {
    // @listing avg-test
    let tape = Tape::new();
    let x = tape.leaf(Tensor::arange(&[1, 1, 4, 4]));
    let y = AvgPool2d::new(2).forward(x);
    assert_eq!(y.value().to_vec(), [2.5, 4.5, 10.5, 12.5]);
    let grad = y.sum_all().backward().get(x).unwrap().to_vec();
    assert_eq!(grad, [0.25; 16]); // every pixel in one window, a quarter each
    let g = GlobalAvgPool.forward(tape.constant(Tensor::arange(&[2, 3, 2, 2])));
    assert_eq!(g.value().to_vec(), [1.5, 5.5, 9.5, 13.5, 17.5, 21.5]); // [2, 3]
    // @end
    let mut rng = Rng::new(22);
    let same = Window::new(3, 1, 1);
    let mut gap = Sequential::new(vec![
        Box::new(Conv2d::new(1, 8, same, &mut rng)),
        Box::new(ReLU),
        Box::new(MaxPool2d::new(2)),
        Box::new(Conv2d::new(8, 16, same, &mut rng)),
        Box::new(ReLU),
        Box::new(GlobalAvgPool), // [N, 16]
        Box::new(Linear::new(16, 10, &mut rng)),
    ]);
    assert_eq!(count(&mut gap), 80 + 1168 + 170); // 1,418 against 9,098
}

#[test]
fn receptive_field_of_the_cnn() {
    // @listing field-test
    let (conv, pool) = (Window::new(3, 1, 1), Window::new(2, 2, 0));
    assert_eq!(receptive_field(&[conv, pool, conv, pool]), 10);
    assert_eq!(receptive_field(&[conv, conv]), 5);
    assert_eq!(receptive_field(&[conv; 13]), 27); // 13 layers to see a digit
    let mut rng = Rng::new(22);
    let two = Sequential::new(vec![
        Box::new(Conv2d::new(1, 2, conv, &mut rng)),
        Box::new(Conv2d::new(2, 1, conv, &mut rng)),
    ]);
    let tape = Tape::new();
    let x = tape.leaf(Tensor::zeros(&[1, 1, 9, 9]));
    let y = two.forward(x).select(2, 4).select(2, 4); // the center pixel
    let grad = y.sum_all().backward().get(x).unwrap().to_vec();
    let seen: Vec<usize> = (0..81).filter(|&i| grad[i] != 0.0).collect();
    let inside = |i: &usize| (2..=6).contains(&(i / 9)) && (2..=6).contains(&(i % 9));
    assert!(seen.len() == 25 && seen.iter().all(inside)); // rows, columns 2..=6
    // @end
}

#[test]
fn channel_dropout_in_the_cnn() {
    // @listing cnn-dropout-test
    let model = mnist_cnn::cnn(&mut Rng::new(20));
    let mut model = with_channel_dropout(model, &[3, 6], 0.25, 22);
    let names = model.named_parameters().into_iter().map(|(n, _)| n);
    let names = names.collect::<Vec<_>>().join(" ");
    assert_eq!(names, "1.weight 1.bias 5.weight 5.bias 10.weight 10.bias");
    let x = Tensor::randn(&[4, 784], &mut Rng::new(23));
    assert!(eval_is_deterministic(&mut model, &x)); // Sequential passes train on
    let first = |model: &Sequential| {
        let tape = Tape::new();
        let mut h = tape.constant(x.clone());
        for layer in &model.layers[..5] {
            h = layer.forward(h); // through the first ChannelDropout
        }
        h.value().reshape(&[32, 196]) // 4 images × 8 maps of 14 × 14
    };
    let trained = first(&model);
    model.train(false);
    let all = first(&model);
    let zero = |t: &Tensor, m| t.select(0, m).to_vec().iter().all(|&v| v == 0.0);
    let dropped = |&m: &usize| zero(&trained, m) && !zero(&all, m);
    let gone = (0..32).filter(dropped).count();
    assert_eq!(gone, 11); // of 32 maps; 8 ± 2.4 expected
    // @end
}

#[test]
fn depthwise() {
    // @listing depthwise-test
    let mut rng = Rng::new(22);
    let (c, win) = (3, Window::new(3, 1, 1));
    let x = Tensor::randn(&[2, c, 5, 5], &mut rng);
    let k = Tensor::randn(&[c, 9], &mut rng);
    let mut full = vec![0.0; c * 9 * c]; // [C·9, C]: kernel c reads only channel c
    for ch in 0..c {
        for j in 0..9 {
            full[(ch * 9 + j) * c + ch] = k.get(&[ch, j]);
        }
    }
    let tape = Tape::new();
    let (xv, kv) = (tape.constant(x.clone()), tape.constant(k.clone()));
    let a = depthwise_conv2d(xv, kv, win).value();
    let b = xv.conv2d(tape.constant(t(&full, &[c * 9, c])), win).value();
    assert!(a.allclose_rel(&b, 1e-6, 1e-5));
    let tol = Tolerance::default();
    let check = gradcheck(|v| depthwise_conv2d(v[0], v[1], win), &[x, k], tol);
    assert!(check.is_ok());
    let (dw, pw, full) = (32 * 9, 32 * 64, 32 * 9 * 64); // 32 → 64 channels
    assert_eq!((dw + pw, full), (2_336, 18_432)); // separable: 7.9 times fewer
    // @end
}

#[test]
fn transposed_convolution() {
    // @listing transposed-test
    let tape = Tape::new();
    let x = tape.constant(t(&[1., 2., 3., 4.], &[1, 1, 2, 2]));
    let ones = tape.constant(t(&[1.; 4], &[1, 4])); // [1, 1·2·2]
    let up = conv_transpose2d(x, ones, Window::new(2, 2, 0)).value(); // [1, 1, 4, 4]
    assert_eq!(up.select(2, 0).to_vec(), [1., 1., 2., 2.]); // rows 0 and 1
    assert_eq!(up.select(2, 3).to_vec(), [3., 3., 4., 4.]); // rows 2 and 3
    let mut rng = Rng::new(22); // the adjoint: ⟨conv(x), y⟩ = ⟨x, convᵀ(y)⟩
    let (win, x) = (Window::new(3, 2, 1), Tensor::randn(&[2, 3, 7, 7], &mut rng));
    let w = Tensor::randn(&[27, 4], &mut rng);
    let y = Tensor::randn(&[2, 4, 4, 4], &mut rng);
    let (xv, wv) = (tape.constant(x.clone()), tape.constant(w.clone()));
    let conv = xv.conv2d(wv, win).value();
    let wt = tape.constant(w.transpose(0, 1).contiguous()); // [4, 27]
    let back = conv_transpose2d(tape.constant(y.clone()), wt, win).value();
    let (left, right) = (conv.mul(&y).sum_all(), x.mul(&back).sum_all());
    assert!((left - right).abs() < 1e-3 * left.abs().max(1.0));
    // @end
    assert_eq!(
        (up.shape(), back.shape()),
        (&[1, 1, 4, 4][..], &[2, 3, 7, 7][..])
    );
}
