//! The examples printed in Chapter 9 of the guided edition: backward through
//! broadcasting, reductions and views.
use ember::{Gradients, Tape, Tensor, Var};

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn grad(grads: &Gradients, v: Var) -> Vec<f32> {
    grads.get(v).expect("a gradient").to_vec()
}

fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(x, w)| (x - w).abs() <= tol)
}

/// The weights 1, 2, …, 6 of Section 9.1's loss, as a 2 × 3 matrix.
fn w23() -> Tensor {
    t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
}

// @listing nudge
/// The slope of `L = Σ seed · f(x)` in each element of `x`, by central differences
/// with step `h` (Section 8.7's check, for any function of one Var).
fn nudges(
    f: impl for<'t> Fn(Var<'t>) -> Var<'t>,
    x: &Tensor,
    seed: &Tensor,
    h: f32,
) -> Vec<f32> {
    let loss = |xs: Vec<f32>| {
        let tape = Tape::new();
        let y = f(tape.constant(Tensor::from_vec(xs, x.shape())));
        y.value().mul(seed).sum_all()
    };
    let at = x.to_vec();
    let slope = |i: usize| {
        let (mut up, mut down) = (at.clone(), at.clone());
        up[i] += h;
        down[i] -= h;
        (loss(up) - loss(down)) / (2.0 * h)
    };
    (0..at.len()).map(slope).collect()
}
// @end

/// The tape's gradient of the same loss, with respect to `x`.
fn tape_grad(
    f: impl for<'t> Fn(Var<'t>) -> Var<'t>,
    x: &Tensor,
    seed: &Tensor,
) -> Vec<f32> {
    let tape = Tape::new();
    let v = tape.leaf(x.clone());
    let l = (f(v) * tape.constant(seed.clone())).sum_all();
    grad(&l.backward(), v)
}

/// Section 9.1's batch plus a bias, as a function of the bias.
fn add_to_batch(b: Var) -> Var {
    b.tape().constant(Tensor::arange(&[2, 3])) + b
}

#[test]
fn a_bias_read_twice() {
    // @listing bias
    let tape = Tape::new();
    let x = tape.leaf(Tensor::arange(&[2, 3])); // a batch of two rows
    let b = tape.leaf(t(&[10.0, 20.0, 30.0], &[3])); // one bias for both
    let w = tape.constant(w23()); // weights, so that ȳ = w
    let y = x + b;
    assert_eq!(y.value().to_vec(), [10.0, 21.0, 32.0, 13.0, 24.0, 35.0]);
    let grads = (y * w).sum_all().backward();
    assert_eq!(grad(&grads, x), w23().to_vec()); // ȳ, one element each
    assert_eq!(grad(&grads, b), [5.0, 7.0, 9.0]); // 1 + 4, 2 + 5, 3 + 6
    let ones = (x + b).sum_all().backward(); // every ȳ is 1:
    assert_eq!(grad(&ones, b), [2.0, 2.0, 2.0]); // the number of reads
    // @end
}

#[test]
fn nudging_the_bias() {
    // @listing bias-nudge
    let b = t(&[10.0, 20.0, 30.0], &[3]);
    assert_eq!(nudges(add_to_batch, &b, &w23(), 0.25), [5.0, 7.0, 9.0]);
    assert_eq!(tape_grad(add_to_batch, &b, &w23()), [5.0, 7.0, 9.0]);
    // @end
}

#[test]
fn both_operands_broadcast() {
    // @listing outer
    let tape = Tape::new();
    let c = tape.leaf(t(&[1.0, 2.0], &[2, 1])); // a column
    let r = tape.leaf(t(&[10.0, 20.0, 30.0], &[3])); // a row
    let y = c * r; // [2, 3]: y[i][j] = c[i] · r[j]
    assert_eq!(y.value().to_vec(), [10.0, 20.0, 30.0, 20.0, 40.0, 60.0]);
    let grads = (y * tape.constant(w23())).sum_all().backward();
    assert_eq!(grad(&grads, c), [140.0, 320.0]); // Σ over j of w[i][j] · r[j]
    assert_eq!(grads.get(c).unwrap().shape(), [2, 1]);
    assert_eq!(grad(&grads, r), [9.0, 12.0, 15.0]); // Σ over i of w[i][j] · c[i]
    // @end
}

#[test]
fn expand_backward_is_sum_to() {
    // @listing expand-example
    let tape = Tape::new();
    let b = tape.leaf(t(&[10.0, 20.0, 30.0], &[3]));
    let e = b.expand(&[2, 3]);
    assert!(e.value().shares_storage(&b.value())); // a view on the tape, too
    let grads = (e * tape.constant(w23())).sum_all().backward();
    assert_eq!(grad(&grads, b), [5.0, 7.0, 9.0]); // as in Section 9.1
    let col = tape.leaf(t(&[1.0, 2.0], &[2, 1]));
    let l = (col.expand(&[2, 3]) * tape.constant(w23())).sum_all();
    assert_eq!(grad(&l.backward(), col), [6.0, 15.0]); // the rows of w, summed
    // @end
}

#[test]
fn sum_and_mean() {
    // @listing sum-example
    let tape = Tape::new();
    let x = tape.leaf(Tensor::arange(&[2, 3]));
    let d = |y: Var, seed: &[f32]| {
        let s = tape.constant(t(seed, &y.shape()));
        grad(&(y * s).sum_all().backward(), x)
    };
    let rows = x.sum(1, false);
    assert_eq!(rows.value().to_vec(), [3.0, 12.0]);
    let want = [10.0, 10.0, 10.0, 100.0, 100.0, 100.0];
    assert_eq!(d(rows, &[10.0, 100.0]), want); // each ȳ copied along its row
    assert_eq!(d(x.sum(1, true), &[10.0, 100.0]), want); // keepdim: the same
    let cols = x.mean(0, false);
    assert_eq!(cols.value().to_vec(), [1.5, 2.5, 3.5]);
    let half = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    assert_eq!(d(cols, &[2.0, 4.0, 6.0]), half); // copied down, then halved
    let all = grad(&x.mean_all().backward(), x);
    assert!(close(&all, &[1.0 / 6.0; 6], 1e-7)); // ȳ / 6 for each element
    // @end
}

#[test]
fn spreading_without_the_axis() {
    // @listing spread-mistake
    let g = t(&[1.0, 2.0, 3.0], &[3]); // ȳ for the row sums of a 3 × 3 matrix
    let right = g.reshape(&[3, 1]).expand(&[3, 3]); // row i holds ȳ[i]
    let wrong = g.expand(&[3, 3]); // column j holds ȳ[j]: lined up the wrong way
    assert_eq!(right.to_vec()[..6], [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    assert_eq!(wrong.to_vec()[..6], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_eq!(wrong.to_vec(), right.transpose(0, 1).to_vec());
    let x = Tensor::arange(&[3, 3]);
    let nudged = nudges(|x| x.sum(1, false), &x, &g, 0.25);
    assert_eq!(nudged, right.to_vec());
    // @end
}

#[test]
#[should_panic(expected = "expand: [2] to [2, 3]: axis 0 has size 2, not 1 or 3")]
fn spreading_a_non_square_gradient_panics() {
    t(&[1.0, 2.0], &[2]).expand(&[2, 3]);
}

#[test]
fn max_backward() {
    // @listing max-example
    let tape = Tape::new();
    let xs = t(&[2.0, 7.0, 7.0, 4.0, 1.0, 3.0], &[2, 3]); // a tie in row 0
    let x = tape.leaf(xs.clone());
    let m = x.max(1, false);
    assert_eq!(m.value().to_vec(), [7.0, 4.0]);
    assert_eq!(xs.argmax(1), [1, 0]); // first wins
    let seed = t(&[10.0, 20.0], &[2]);
    let grads = (m * tape.constant(seed.clone())).sum_all().backward();
    assert_eq!(grad(&grads, x), [0.0, 10.0, 0.0, 20.0, 0.0, 0.0]); // to argmax
    let nudged = nudges(|x| x.max(1, false), &xs, &seed, 0.25);
    assert_eq!(nudged, [0.0, 5.0, 5.0, 20.0, 0.0, 0.0]); // the tie, split
    // @end
}

#[test]
fn views_backward() {
    // @listing views-example
    let tape = Tape::new();
    let x = tape.leaf(Tensor::arange(&[2, 3]));
    let d = |y: Var| {
        let seed = tape.constant(Tensor::arange(&y.shape()));
        grad(&(y * seed).sum_all().backward(), x)
    };
    let y = x.transpose(0, 1);
    assert!(y.value().shares_storage(&x.value()));
    assert_eq!(d(y), [0.0, 2.0, 4.0, 1.0, 3.0, 5.0]); // ȳ, transposed back
    assert_eq!(d(x.reshape(&[3, 2])), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let x3 = tape.leaf(Tensor::arange(&[2, 3, 4]));
    let y3 = x3.permute(&[2, 0, 1]); // shape [4, 2, 3]
    let seed = tape.constant(Tensor::arange(&[4, 2, 3]));
    let g = (y3 * seed).sum_all().backward().get(x3).unwrap().clone();
    assert_eq!((g.get(&[0, 0, 1]), g.get(&[1, 2, 3])), (6.0, 23.0)); // 6k + 3i + j
    // @end
    assert_eq!(
        g.to_vec(),
        Tensor::arange(&[4, 2, 3]).permute(&[1, 2, 0]).to_vec()
    );
}

#[test]
fn slots_and_scatter() {
    // @listing slots-example
    let x = Tensor::arange(&[3, 4]);
    assert_eq!(x.slice(1, 0, 4, 2).slots(), [0, 2, 4, 6, 8, 10]); // columns 0, 2
    assert_eq!(x.select(0, 1).slots(), [4, 5, 6, 7]); // row 1
    let b = t(&[10.0, 20.0, 30.0], &[3]);
    assert_eq!(b.expand(&[2, 3]).slots(), [0, 1, 2, 0, 1, 2]); // each slot twice
    let into = Tensor::scatter_add(&[3], &[0, 1, 2, 0, 1, 2], &w23());
    assert_eq!(into.to_vec(), [5.0, 7.0, 9.0]); // Section 9.1's sum, again
    // @end
}

#[test]
fn slice_and_select_backward() {
    // @listing slice-example
    let tape = Tape::new();
    let x = tape.leaf(Tensor::arange(&[3, 4]));
    let even = x.slice(1, 0, 4, 2); // columns 0 and 2
    let seed = tape.constant(t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
    let g = grad(&(even * seed).sum_all().backward(), x);
    let want = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0];
    assert_eq!(g, want); // columns 1 and 3 get zeros
    let row = x.select(0, 1);
    let seed = tape.constant(t(&[1.0, 2.0, 3.0, 4.0], &[4]));
    let g = grad(&(row * seed).sum_all().backward(), x);
    assert_eq!(g[4..8], [1.0, 2.0, 3.0, 4.0]); // row 1
    assert_eq!(g.iter().filter(|&&v| v == 0.0).count(), 8); // zeros elsewhere
    // @end
}

/// Distinct values in a scrambled order: 7i mod 24.
fn scrambled() -> Tensor {
    let xs: Vec<f32> = (0..24).map(|i| (i * 7 % 24) as f32).collect();
    Tensor::from_vec(xs, &[2, 3, 4])
}

#[test]
fn every_rule_against_nudges() {
    // @listing check-all
    type Op = fn(Var) -> Var;
    let ops: [(&str, Op); 12] = [
        ("expand", |x| x.expand(&[3, 2, 3, 4])),
        ("sum", |x| x.sum(1, false)),
        ("sum keepdim", |x| x.sum(2, true)),
        ("mean", |x| x.mean(1, true)),
        ("mean_all", |x| x.mean_all()),
        ("max", |x| x.max(2, false)),
        ("max dim 0", |x| x.max(0, true)),
        ("reshape", |x| x.reshape(&[4, 6])),
        ("transpose", |x| x.transpose(0, 2)),
        ("permute", |x| x.permute(&[1, 2, 0])),
        ("slice", |x| x.slice(2, 1, 4, 2)),
        ("select", |x| x.select(1, 2)),
    ];
    let x = scrambled(); // shape [2, 3, 4], no ties
    for (name, f) in ops {
        let shape = f(Tape::new().constant(x.clone())).shape();
        let seed = Tensor::arange(&shape).add_scalar(1.0); // 1, 2, 3, ...
        let (tape, nudged) = (tape_grad(f, &x, &seed), nudges(f, &x, &seed, 0.25));
        assert!(
            close(&tape, &nudged, 1e-4),
            "{name}: {tape:?} vs {nudged:?}"
        );
    }
    // @end
}

#[test]
fn centering_rows() {
    // @listing center
    let tape = Tape::new();
    let x = tape.leaf(Tensor::arange(&[2, 3]));
    let mean = x.mean(1, true); // [[1], [4]]
    let c = x - mean; // [2, 3] minus [2, 1]
    assert_eq!(c.value().to_vec(), [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]);
    let l = (c * tape.constant(w23())).sum_all();
    assert_eq!(tape.len(), 6);
    let grads = l.backward();
    assert_eq!(grad(&grads, c), w23().to_vec());
    assert_eq!(grad(&grads, mean), [-6.0, -15.0]); // -w, summed along rows
    assert_eq!(grad(&grads, x), [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]); // w - mean(w)
    // @end
    let from_mean = grads.get(x).unwrap().sub(&w23()); // the second contribution
    assert_eq!(from_mean.to_vec(), [-2.0, -2.0, -2.0, -5.0, -5.0, -5.0]);
    let x = Tensor::arange(&[2, 3]);
    let nudged = nudges(|x| x - x.mean(1, true), &x, &w23(), 0.25);
    assert!(close(&nudged, &[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0], 1e-5));
}
