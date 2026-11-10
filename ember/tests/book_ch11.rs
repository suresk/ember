//! The examples printed in Chapter 11 of the guided edition: parameters and the
//! `Module` trait.
use ember::{Module, ParamId, Parameter, Tape, Tensor, Var};
use std::cell::Cell;
use std::thread;

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

/// A parameter's gradient as a `Vec`.
fn grad(p: &Parameter) -> Vec<f32> {
    p.grad.as_ref().expect("a gradient").to_vec()
}

/// `x` is within `1e-5` of `want`, relative to `want` (absolute below 1).
fn near(x: f32, want: f32) -> bool {
    (x - want).abs() <= 1e-5 * want.abs().max(1.0)
}

// @listing affine
/// `y = x·w + b` for a batch of rows: `x` is `[B, k]`, `w` is `[k, n]`, `b` is `[n]`.
struct Affine {
    w: Parameter,
    b: Parameter,
}

impl Module for Affine {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let tape = x.tape(); // the tape this step records on
        x.matmul(tape.param(&self.w)) + tape.param(&self.b)
    }

    fn parameters(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.w, &mut self.b]
    }
}
// @end

impl Affine {
    fn new(w: Tensor, b: Tensor) -> Affine {
        let (w, b) = (Parameter::new(w), Parameter::new(b));
        Affine { w, b }
    }

    /// `k` inputs to `n` outputs, every number 0.
    fn zeros(k: usize, n: usize) -> Affine {
        Affine::new(Tensor::zeros(&[k, n]), Tensor::zeros(&[n]))
    }
}

/// The section's model, `w = 0`, `b = 0`, and its data: two points on `y = 2x + 1`.
fn line() -> (Affine, Tensor, Tensor) {
    let model = Affine::new(t(&[0.0], &[1, 1]), t(&[0.0], &[1]));
    (model, t(&[1.0, 2.0], &[2, 1]), t(&[3.0, 5.0], &[2, 1]))
}

// @listing mse
/// The mean squared error, `mean((pred - y)²)`.
fn mse<'t>(pred: Var<'t>, y: Var<'t>) -> Var<'t> {
    (pred - y).powf(2.0).mean_all()
}
// @end

// @listing step
/// The loss on `(x, y)`, with its gradients added into the parameters' `grad`.
fn loss_and_grads(model: &mut impl Module, x: &Tensor, y: &Tensor) -> f32 {
    let tape = Tape::new(); // one tape per step (Section 7.7)
    let pred = model.forward(tape.constant(x.clone()));
    let loss = mse(pred, tape.constant(y.clone()));
    loss.backward_into(&mut model.parameters());
    loss.value().item()
} // the tape is dropped here, and each parameter owns its buffer again
// @end

// @listing sgd
/// Gradient descent: move every parameter `lr` times its gradient downhill.
fn sgd(model: &mut impl Module, lr: f32) {
    for p in model.parameters() {
        let g = p.grad.as_ref().expect("a gradient");
        p.value = p.value.sub(&g.mul_scalar(lr));
    }
}
// @end

#[test]
fn ids_are_unique() {
    // @listing ids
    let one = || Parameter::new(t(&[1.0], &[1]));
    let (a, b) = (one(), one());
    assert_ne!(a.id(), b.id()); // same value, different parameters
    assert!(a.id() < b.id()); // on one thread, numbered in order
    let made = || Parameter::new(Tensor::zeros(&[1])).id();
    let mut ids: Vec<ParamId> = (0..4)
        .map(|_| thread::spawn(made).join().unwrap()) // one parameter per thread
        .collect();
    ids.dedup();
    assert_eq!(ids.len(), 4); // four different ids
    // @end
}

#[test]
fn a_thread_local_counter_repeats_itself() {
    // @listing thread-local
    thread_local! {
        static NEXT: Cell<usize> = const { Cell::new(0) };
    }
    let next = || NEXT.with(|n| n.replace(n.get() + 1)); // one counter per thread
    let ids: Vec<usize> = (0..4)
        .map(|_| thread::spawn(next).join().unwrap())
        .collect();
    assert_eq!(ids, [0, 0, 0, 0]); // every thread starts its own count at 0
    // @end
}

#[test]
fn a_parameter_on_the_tape() {
    // @listing param-example
    let w = Parameter::new(Tensor::arange(&[2, 3]));
    let tape = Tape::new();
    let v = tape.param(&w);
    assert_eq!((v.op(), v.requires_grad()), ("param", true));
    assert!(v.value().shares_storage(&w.value)); // the buffer, not a copy
    let again = tape.param(&w); // a second use is a second node
    assert_eq!((v.id(), again.id(), tape.len()), (0, 1, 2));
    // @end
}

#[test]
fn one_step_by_hand() {
    // @listing step-example
    let (mut model, xs, ys) = line(); // w = 0, b = 0; y = 2x + 1 at x = 1, 2
    let tape = Tape::new();
    let (x, y) = (tape.constant(xs), tape.constant(ys));
    let pred = model.forward(x);
    let loss = mse(pred, y);
    assert_eq!((loss.value().item(), tape.len()), (17.0, 9));
    loss.backward_into(&mut model.parameters());
    assert_eq!((grad(&model.w), grad(&model.b)), (vec![-13.0], vec![-8.0]));
    // @end
    // The table: x, y, param w, matmul, param b, add, sub, powf, mean_all.
    assert_eq!((x.id(), y.id(), pred.id(), loss.id()), (0, 1, 5, 8));
    assert_eq!((pred.op(), pred.parents()), ("add", vec![3, 4]));
    assert_eq!(loss.parents(), [7]);
    assert_eq!(pred.value().to_vec(), [0.0, 0.0]);
}

#[test]
fn the_table_of_one_step() {
    // Section 11.4's table, node by node, on a tape built by hand.
    let (model, xs, ys) = line();
    let tape = Tape::new();
    let (x, y) = (tape.constant(xs), tape.constant(ys));
    let w = tape.param(&model.w);
    let m = x.matmul(w);
    let b = tape.param(&model.b);
    let pred = m + b;
    let d = pred - y;
    let sq = d.powf(2.0);
    let loss = sq.mean_all();
    let vars = [x, y, w, m, b, pred, d, sq, loss];
    let ids: Vec<usize> = vars.iter().map(|v| v.id()).collect();
    assert_eq!(ids, (0..9).collect::<Vec<_>>());
    let ops: Vec<&str> = vars.iter().map(|v| v.op()).collect();
    let want = ["const", "const", "param", "matmul", "param", "add", "sub"];
    assert_eq!(ops[..7], want);
    assert_eq!((ops[7], ops[8]), ("powf", "mean_all"));
    assert_eq!((m.parents(), pred.parents()), (vec![0, 2], vec![3, 4]));
    assert_eq!((d.parents(), sq.parents()), (vec![5, 1], vec![6]));
    assert_eq!(
        (d.value().to_vec(), sq.value().to_vec()),
        (vec![-3.0, -5.0], vec![9.0, 25.0])
    );
    let grads = loss.backward();
    let g = |v: Var| grads.get(v).unwrap().to_vec();
    assert_eq!((g(loss), g(sq)), (vec![1.0], vec![0.5, 0.5]));
    for v in [d, pred, m] {
        assert_eq!(g(v), [-3.0, -5.0]);
    }
    assert_eq!((g(w), g(b)), (vec![-13.0], vec![-8.0]));
    assert!(grads.get(x).is_none() && grads.get(y).is_none());
}

#[test]
fn two_backward_passes_add() {
    let (mut model, xs, ys) = line();
    // @listing twice
    let tape = Tape::new();
    let loss = mse(model.forward(tape.constant(xs)), tape.constant(ys));
    loss.backward_into(&mut model.parameters());
    loss.backward_into(&mut model.parameters()); // the same loss again
    assert_eq!((grad(&model.w), grad(&model.b)), (vec![-26.0], vec![-16.0]));
    model.zero_grad();
    assert!(model.w.grad.is_none() && model.b.grad.is_none());
    // @end
}

#[test]
fn micro_batches_add_up_to_the_batch() {
    let (mut model, _, _) = line();
    // @listing micro-batches
    for (x, y) in [(1.0, 3.0), (2.0, 5.0)] {
        let tape = Tape::new(); // a tape per micro-batch: one example each
        let pred = model.forward(tape.constant(t(&[x], &[1, 1])));
        let loss = mse(pred, tape.constant(t(&[y], &[1, 1])));
        (loss * 0.5).backward_into(&mut model.parameters()); // 1 of 2 examples
    }
    assert_eq!((grad(&model.w), grad(&model.b)), (vec![-13.0], vec![-8.0]));
    // @end
    // Each example alone, unscaled: (-6, -6) and (-20, -10).
    for ((x, y), want) in [((1.0, 3.0), [-6.0, -6.0]), ((2.0, 5.0), [-20.0, -10.0])] {
        model.zero_grad();
        let tape = Tape::new();
        let pred = model.forward(tape.constant(t(&[x], &[1, 1])));
        let loss = mse(pred, tape.constant(t(&[y], &[1, 1])));
        loss.backward_into(&mut model.parameters());
        assert_eq!([grad(&model.w)[0], grad(&model.b)[0]], want);
    }
}

#[test]
fn four_steps_of_gradient_descent() {
    // @listing training
    let (mut model, x, y) = line();
    let mut losses = vec![];
    for _ in 0..4 {
        model.zero_grad();
        losses.push(loss_and_grads(&mut model, &x, &y));
        sgd(&mut model, 0.1);
    }
    let want = [17.0, 1.685, 0.16925, 0.019169];
    assert!(losses.iter().zip(want).all(|(&l, w)| near(l, w)));
    let (w, b) = (model.w.value.item(), model.b.value.item());
    assert!(near(w, 1.8819) && near(b, 1.1496)); // heading for 2 and 1
    // @end
    assert!(near(loss_and_grads(&mut model, &x, &y), 0.0042459));
}

#[test]
fn the_trace_of_the_first_steps() {
    // The table in Section 11.6: w, b and both gradients at each step.
    let (mut model, x, y) = line();
    let rows = [
        (0.0, 0.0, -13.0, -8.0),
        (1.3, 0.8, -4.1, -2.5),
        (1.71, 1.05, -1.3, -0.77),
        (1.84, 1.127, -0.419, -0.226),
    ];
    for (w, b, gw, gb) in rows {
        assert!(near(model.w.value.item(), w) && near(model.b.value.item(), b));
        model.zero_grad();
        loss_and_grads(&mut model, &x, &y);
        assert!(near(grad(&model.w)[0], gw) && near(grad(&model.b)[0], gb));
        sgd(&mut model, 0.1);
    }
}

#[test]
fn forgetting_zero_grad() {
    // @listing forgetting
    let (mut model, x, y) = line();
    let mut losses = vec![];
    for _ in 0..2 {
        losses.push(loss_and_grads(&mut model, &x, &y)); // no zero_grad
        sgd(&mut model, 0.1);
    }
    assert!(near(grad(&model.w)[0], -17.1)); // -13 from step 1, -4.1 from step 2
    assert!(near(grad(&model.b)[0], -10.5)); // -8 and -2.5
    assert!(near(losses[1], 1.685)); // step 2 starts where it should...
    let (w, b) = (model.w.value.item(), model.b.value.item());
    assert!(near(w, 3.01) && near(b, 1.85)); // ...but takes step 1 again
    assert!(near(loss_and_grads(&mut model, &x, &y), 5.84825)); // 1.685 before
    // @end
}

// @listing two-layers
/// Two affine maps with a `tanh` between them.
struct TwoLayers {
    hidden: Affine,
    out: Affine,
}

impl Module for TwoLayers {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        self.out.forward(self.hidden.forward(x).tanh())
    }

    fn parameters(&mut self) -> Vec<&mut Parameter> {
        let mut ps = self.hidden.parameters(); // borrows self.hidden...
        ps.extend(self.out.parameters()); // ...and self.out: two fields
        ps
    }
}
// @end

impl TwoLayers {
    fn zeros(k: usize, h: usize, n: usize) -> TwoLayers {
        let (hidden, out) = (Affine::zeros(k, h), Affine::zeros(h, n));
        TwoLayers { hidden, out }
    }
}

// @listing count
/// The number of trainable numbers in a model.
fn count(model: &mut dyn Module) -> usize {
    model.parameters().iter().map(|p| p.value.numel()).sum()
}
// @end

#[test]
fn modules_of_modules() {
    // @listing two-layers-example
    let mut model = TwoLayers::zeros(2, 3, 1);
    let tape = Tape::new();
    let x = tape.constant(t(&[1.0, 2.0, -1.0, 0.5], &[2, 2]));
    let loss = model.forward(x).sum_all();
    loss.backward_into(&mut model.parameters());
    let shape = |p: &&mut Parameter| p.grad.as_ref().unwrap().shape().to_vec();
    let shapes: Vec<Vec<usize>> = model.parameters().iter().map(shape).collect();
    assert_eq!(shapes, [vec![2, 3], vec![3], vec![3, 1], vec![1]]);
    assert_eq!(count(&mut model), 13); // 2·3 + 3 + 3·1 + 1
    model.zero_grad(); // the default method, through both children
    assert!(model.parameters().iter().all(|p| p.grad.is_none()));
    // @end
}

#[test]
fn counting_an_mnist_model() {
    // @listing mnist-count
    let mut model = TwoLayers::zeros(784, 128, 10);
    assert_eq!(count(&mut model), 101_770); // 784·128 + 128 + 128·10 + 10
    assert_eq!(model.parameters().len(), 4); // in four tensors
    // @end
}

#[test]
fn boxed_modules() {
    // @listing boxed
    let mut models: Vec<Box<dyn Module>> = vec![
        Box::new(Affine::zeros(2, 1)),
        Box::new(TwoLayers::zeros(2, 3, 1)),
    ];
    let tape = Tape::new();
    let x = tape.constant(Tensor::arange(&[4, 2]));
    for m in &models {
        assert_eq!(m.forward(x).shape(), [4, 1]); // the same call for both
    }
    let counts: Vec<usize> = models.iter_mut().map(|m| count(m.as_mut())).collect();
    assert_eq!(counts, [3, 13]);
    // @end
}
