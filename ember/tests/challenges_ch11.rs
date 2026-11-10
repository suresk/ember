//! Tests for the Chapter 11 challenges.
use ember::{
    Frozen, Module, NamedModule, Parameter, Residual, Tape, Tensor, Tied, Var,
    accumulate_mse, prefixed, zero_grad_to_zeros,
};

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn grad(p: &Parameter) -> Vec<f32> {
    p.grad.as_ref().expect("a gradient").to_vec()
}

fn near(x: f32, want: f32) -> bool {
    (x - want).abs() <= 1e-5 * want.abs().max(1.0)
}

/// Section 11.6's module.
struct Affine {
    w: Parameter,
    b: Parameter,
}

impl Affine {
    fn new(w: Tensor, b: Tensor) -> Affine {
        let (w, b) = (Parameter::new(w), Parameter::new(b));
        Affine { w, b }
    }
}

impl Module for Affine {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        let tape = x.tape();
        x.matmul(tape.param(&self.w)) + tape.param(&self.b)
    }

    fn parameters(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.w, &mut self.b]
    }
}

/// Section 11.7's module.
struct TwoLayers {
    hidden: Affine,
    out: Affine,
}

impl Module for TwoLayers {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        self.out.forward(self.hidden.forward(x).tanh())
    }

    fn parameters(&mut self) -> Vec<&mut Parameter> {
        let mut ps = self.hidden.parameters();
        ps.extend(self.out.parameters());
        ps
    }
}

fn zeros(k: usize, n: usize) -> Affine {
    Affine::new(Tensor::zeros(&[k, n]), Tensor::zeros(&[n]))
}

#[test]
fn predict_accumulation() {
    let mut model = Affine::new(t(&[2.0, -1.0], &[2, 1]), t(&[0.5], &[1]));
    let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tape = Tape::new();
    let loss = model.forward(tape.constant(x)).sum_all();
    assert_eq!((loss.value().item(), tape.len()), (3.0, 6));
    loss.backward_into(&mut model.parameters());
    assert_eq!(
        (grad(&model.w), grad(&model.b)),
        (vec![4.0, 6.0], vec![2.0])
    );
    loss.backward_into(&mut model.parameters());
    assert_eq!(
        (grad(&model.w), grad(&model.b)),
        (vec![8.0, 12.0], vec![4.0])
    );
    model.zero_grad();
    loss.backward_into(&mut model.parameters());
    assert_eq!(grad(&model.w), [4.0, 6.0]);
}

/// A module with a parameter its forward never reads.
struct Spare {
    used: Affine,
    spare: Parameter,
}

impl Module for Spare {
    fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
        self.used.forward(x)
    }

    fn parameters(&mut self) -> Vec<&mut Parameter> {
        let mut ps = self.used.parameters();
        ps.push(&mut self.spare);
        ps
    }
}

#[test]
fn zeros_or_none() {
    let used = Affine::new(t(&[2.0, -1.0], &[2, 1]), t(&[0.5], &[1]));
    let mut model = Spare {
        used,
        spare: Parameter::new(Tensor::zeros(&[3])),
    };
    let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let step = |model: &mut Spare| {
        let tape = Tape::new();
        let loss = model.forward(tape.constant(x.clone())).sum_all();
        loss.backward_into(&mut model.parameters());
    };
    zero_grad_to_zeros(&mut model);
    step(&mut model);
    assert_eq!(grad(&model.used.w), [4.0, 6.0]); // 0 + the gradient
    assert_eq!(grad(&model.spare), [0.0, 0.0, 0.0]); // zeros, never touched
    model.zero_grad();
    step(&mut model);
    assert_eq!(grad(&model.used.w), [4.0, 6.0]); // the same numbers
    assert!(model.spare.grad.is_none()); // "no gradient", not "a gradient of 0"
}

// @listing named-impls
impl NamedModule for Affine {
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)> {
        vec![("w".into(), &mut self.w), ("b".into(), &mut self.b)]
    }
}

impl NamedModule for TwoLayers {
    fn named_parameters(&mut self) -> Vec<(String, &mut Parameter)> {
        let hidden = NamedModule::named_parameters(&mut self.hidden);
        let mut named = prefixed("hidden", hidden);
        named.extend(prefixed(
            "out",
            NamedModule::named_parameters(&mut self.out),
        ));
        named
    }
}
// @end

#[test]
fn dotted_names() {
    let mut model = TwoLayers {
        hidden: zeros(2, 3),
        out: zeros(3, 1),
    };
    let named = NamedModule::named_parameters(&mut model);
    let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["hidden.w", "hidden.b", "out.w", "out.b"]);
    let sizes: Vec<usize> = named.iter().map(|(_, p)| p.value.numel()).collect();
    assert_eq!(sizes, [6, 3, 3, 1]);
    // The same parameters, in the same order, as parameters().
    let by_name: Vec<_> = NamedModule::named_parameters(&mut model)
        .iter()
        .map(|(_, p)| p.id())
        .collect();
    let ids: Vec<_> = model.parameters().iter().map(|p| p.id()).collect();
    assert_eq!(by_name, ids);
}

#[test]
fn uneven_micro_batches() {
    let x = t(&[1.0, 2.0, 3.0, 4.0], &[4, 1]);
    let y = t(&[3.0, 5.0, 7.0, 9.0], &[4, 1]); // y = 2x + 1
    let line = || Affine::new(t(&[0.0], &[1, 1]), t(&[0.0], &[1]));
    for sizes in [vec![4], vec![1, 3], vec![2, 2], vec![1, 1, 1, 1]] {
        let mut model = line();
        accumulate_mse(&mut model, &x, &y, &sizes);
        assert!(near(grad(&model.w)[0], -35.0), "{sizes:?}");
        assert!(near(grad(&model.b)[0], -12.0), "{sizes:?}");
    }
    // Unweighted, the means of 1 and 3 rows add to something else.
    let mut model = line();
    let parts = [((0, 1), -6.0, -6.0), ((1, 4), -134.0 / 3.0, -14.0)];
    for ((lo, hi), gw, gb) in parts {
        let mut alone = line();
        for m in [&mut model, &mut alone] {
            let tape = Tape::new();
            let rows = |t: &Tensor| tape.constant(t.slice(0, lo, hi, 1));
            let loss = (m.forward(rows(&x)) - rows(&y)).powf(2.0).mean_all();
            loss.backward_into(&mut m.parameters());
        }
        assert!(near(grad(&alone.w)[0], gw) && near(grad(&alone.b)[0], gb));
    }
    assert!(near(grad(&model.w)[0], -6.0 - 134.0 / 3.0)); // -50.67
    assert!(near(grad(&model.b)[0], -20.0));
}

#[test]
fn a_frozen_module() {
    struct Head {
        body: Frozen<Affine>,
        head: Affine,
    }
    impl Module for Head {
        fn forward<'t>(&self, x: Var<'t>) -> Var<'t> {
            self.head.forward(self.body.forward(x))
        }
        fn parameters(&mut self) -> Vec<&mut Parameter> {
            let mut ps = self.body.parameters();
            ps.extend(self.head.parameters());
            ps
        }
    }
    let body = Frozen(Affine::new(t(&[2.0], &[1, 1]), t(&[1.0], &[1])));
    let head = Affine::new(t(&[0.5], &[1, 1]), t(&[0.0], &[1]));
    let mut model = Head { body, head };
    assert_eq!(model.parameters().len(), 2); // only the head's

    let tape = Tape::new();
    let loss = model
        .forward(tape.constant(t(&[1.0, 2.0], &[2, 1])))
        .sum_all();
    loss.backward_into(&mut model.parameters());
    assert!(model.body.0.w.grad.is_none() && model.body.0.b.grad.is_none());
    assert_eq!(grad(&model.head.w), [8.0]); // body's output: 3 + 5
    assert_eq!(grad(&model.head.b), [2.0]);
}

#[test]
fn frozen_gradients_are_still_computed() {
    let frozen = Frozen(Affine::new(t(&[2.0], &[1, 1]), t(&[1.0], &[1])));
    let tape = Tape::new();
    let y = frozen.forward(tape.constant(t(&[1.0, 2.0], &[2, 1])));
    assert!(y.requires_grad()); // its matmul and add keep their rules
    assert_eq!(y.parents(), [2, 3]); // the matmul, and param b
    assert!(frozen.0.w.grad.is_none());
}

#[test]
fn a_residual_block() {
    let mut block = Residual(Affine::new(t(&[0.5], &[1, 1]), t(&[1.0], &[1])));
    let tape = Tape::new();
    let x = tape.leaf(t(&[1.0, 2.0, 3.0], &[3, 1]));
    let loss = block.forward(x).sum_all();
    assert_eq!(loss.value().item(), 12.0); // 2.5 + 4 + 5.5
    assert_eq!(loss.backward().get(x).unwrap().to_vec(), [1.5, 1.5, 1.5]);
    loss.backward_into(&mut block.parameters());
    assert_eq!((grad(&block.0.w), grad(&block.0.b)), (vec![6.0], vec![3.0]));
    // With w = 0 the block is the identity, and x still gets a gradient of 1.
    block.0.w.value = t(&[0.0], &[1, 1]);
    let tape = Tape::new();
    let x = tape.leaf(t(&[1.0, 2.0, 3.0], &[3, 1]));
    let grads = block.forward(x).sum_all().backward();
    assert_eq!(grads.get(x).unwrap().to_vec(), [1.0, 1.0, 1.0]);
}

#[test]
fn tied_weights() {
    for one_node in [true, false] {
        let mut model = Tied {
            w: Parameter::new(t(&[1.0, 2.0], &[2, 1])),
            one_node,
        };
        let tape = Tape::new();
        let y = model.forward(tape.constant(t(&[1.0, 1.0], &[1, 2])));
        assert_eq!(y.value().to_vec(), [3.0, 6.0]);
        y.sum_all().backward_into(&mut model.parameters());
        assert_eq!(grad(&model.w), [6.0, 6.0]);
        // const x, then w once or twice, two products, a transpose, the sum.
        assert_eq!(tape.len(), if one_node { 6 } else { 7 });
    }
    // Each use alone: the encoder's share and the decoder's.
    let w = t(&[1.0, 2.0], &[2, 1]);
    let tape = Tape::new();
    let (enc, dec) = (tape.leaf(w.clone()), tape.leaf(w));
    let x = tape.constant(t(&[1.0, 1.0], &[1, 2]));
    let grads = x
        .matmul(enc)
        .matmul(dec.transpose(0, 1))
        .sum_all()
        .backward();
    assert_eq!(grads.get(enc).unwrap().to_vec(), [3.0, 3.0]);
    assert_eq!(grads.get(dec).unwrap().to_vec(), [3.0, 3.0]);
}
