//! Tests for the Chapter 7 challenges: `cargo test --test challenges_ch07`.
use ember::{Gradients, Tape, Tensor, Var};

fn scalar(x: f32) -> Tensor {
    Tensor::from_vec(vec![x], &[])
}

fn grad(grads: &Gradients, v: Var) -> Vec<f32> {
    grads.get(v).expect("a gradient").to_vec()
}

// @challenge 1
#[test]
fn predict_the_tape() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]));
    let w = tape.leaf(Tensor::from_vec(vec![2.0, -1.0, 0.5], &[3]));
    let c = tape.constant(Tensor::from_vec(vec![1.0; 3], &[3]));
    let u = x.mul(w);
    let v = u.add(c);
    let p = v.mul(x);
    let l = p.sum_all(); // the sum of w·x² + c·x
    assert_eq!(tape.len(), 7);
    assert_eq!(l.id(), 6);
    assert_eq!(l.value().item(), 8.5); // 3 - 2 + 7.5
    let grads = l.backward();
    assert_eq!(grad(&grads, x), [5.0, -3.0, 4.0]); // 2wx + c
    assert_eq!(grad(&grads, w), [1.0, 4.0, 9.0]); // x²
    assert!(grads.get(c).is_none());
    // Six gradients: x, w and the four operations.
    assert!([u, v, p, l].iter().all(|&n| grads.get(n).is_some()));
}

// @challenge 2
#[test]
fn detach_stops_the_gradient() {
    let tape = Tape::new();
    let x = tape.leaf(scalar(3.0));
    let y = x.mul(x.detach());
    assert_eq!(y.value().item(), 9.0);
    assert_eq!(grad(&y.backward(), x), [3.0]); // not 2x = 6
    let d = x.detach();
    assert!(!d.requires_grad());
    assert!(d.value().shares_storage(&x.value()));
    assert_eq!(tape.len(), 4); // x, a detached copy, y, another copy
}

// @challenge 3
#[test]
fn a_seed_for_a_vector_output() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]));
    let w = tape.leaf(Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]));
    let y = x.mul(w);
    let seed = Tensor::from_vec(vec![1.0, 0.0, -1.0], &[3]);
    let grads = y.backward_with(seed.clone());
    assert_eq!(grad(&grads, x), [4.0, 0.0, -6.0]); // w · seed
    assert_eq!(grad(&grads, w), [1.0, 0.0, -3.0]); // x · seed

    // The same as the ordinary backward of the sum of y · seed.
    let l = y.mul(tape.constant(seed)).sum_all();
    let same = l.backward();
    assert_eq!(grad(&same, x), [4.0, 0.0, -6.0]);
    assert_eq!(grad(&same, w), [1.0, 0.0, -3.0]);
}

#[test]
#[should_panic(expected = "backward_with: seed [2], output [3]")]
fn a_seed_has_the_output_shape() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::arange(&[3]));
    x.mul(x).backward_with(Tensor::zeros(&[2]));
}

// @challenge 4
#[test]
fn two_backward_passes() {
    let tape = Tape::new();
    let a = tape.leaf(Tensor::from_vec(vec![1.0, 2.0], &[2]));
    let b = tape.leaf(Tensor::from_vec(vec![3.0, -1.0], &[2]));
    let l1 = a.mul(b).sum_all();
    let l2 = a.mul(a).sum_all();
    let g1 = l1.backward();
    assert_eq!(grad(&l1.backward(), a), grad(&g1, a)); // no compounding
    let g2 = l2.backward();
    assert_eq!(grad(&g1, a), [3.0, -1.0]); // b
    assert_eq!(grad(&g2, a), [2.0, 4.0]); // 2a
    assert!(g2.get(b).is_none()); // l2 doesn't use b

    let merged = g1.merge(g2);
    let both = l1.add(l2).backward();
    assert_eq!(grad(&merged, a), [5.0, 3.0]);
    assert_eq!(grad(&both, a), [5.0, 3.0]);
    assert_eq!(grad(&merged, b), grad(&both, b)); // a: [1, 2]
    assert_eq!(grad(&merged, b), [1.0, 2.0]);
}

#[test]
#[should_panic(expected = "merge: Gradients of two tapes")]
fn merge_wants_one_tape() {
    let (t1, t2) = (Tape::new(), Tape::new());
    let a = t1.leaf(scalar(1.0));
    let b = t2.leaf(scalar(2.0));
    let _ = a.sum_all().backward().merge(b.sum_all().backward());
}

// @challenge 5
#[test]
fn one_node_for_many_inputs() {
    let tape = Tape::new();
    let a = tape.leaf(Tensor::from_vec(vec![1.0, 2.0], &[2]));
    let b = tape.leaf(Tensor::from_vec(vec![10.0, 20.0], &[2]));
    let s = tape.add_n(&[a, b, a]);
    assert_eq!(s.value().to_vec(), [12.0, 24.0]);
    assert_eq!(s.parents(), [0, 1, 0]);
    assert_eq!(tape.len(), 3); // one node, not two adds
    let grads = s.sum_all().backward();
    assert_eq!(grad(&grads, a), [2.0, 2.0]); // used twice
    assert_eq!(grad(&grads, b), [1.0, 1.0]);
}

#[test]
#[should_panic(expected = "add_n: shapes differ")]
fn add_n_wants_one_shape() {
    let tape = Tape::new();
    let a = tape.leaf(Tensor::arange(&[2]));
    let b = tape.leaf(Tensor::arange(&[3]));
    tape.add_n(&[a, b]);
}

#[test]
#[should_panic(expected = "add_n: Vars of two tapes")]
fn add_n_wants_one_tape() {
    let (t1, t2) = (Tape::new(), Tape::new());
    t1.leaf(scalar(5.0)); // id 0 on t1
    let w = t2.leaf(scalar(3.0)); // id 0 on t2
    t1.add_n(&[w]);
}

// @challenge 6
#[test]
fn bytes_on_the_tape() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::arange(&[1000]));
    let y = x.mul(x);
    y.sum_all();
    assert_eq!(tape.value_bytes(), 8_004); // x, y, and one number
    tape.constant(x.value()); // what Challenge 7.2's detach records
    assert_eq!(tape.value_bytes(), 8_004);

    // Section 7.8's chain: 100,001 numbers, one per node.
    let tape = Tape::new();
    let x = tape.leaf(scalar(2.0));
    let mut y = x;
    for _ in 0..100_000 {
        y = y.add(x);
    }
    assert_eq!(tape.value_bytes(), 400_004);
    assert_eq!(tape.len() * 128, 12_800_128); // the nodes themselves
    assert_eq!(12_800_128 / 400_004, 32);
}

// @challenge 7
#[test]
fn only_the_leaves_keep_gradients() {
    let tape = Tape::new();
    let x = tape.leaf(scalar(2.0));
    let mut y = x;
    for _ in 0..1000 {
        y = y.add(x);
    }
    assert_eq!(y.backward().count(), 1001);
    let leaves = y.backward_leaves();
    assert_eq!(leaves.count(), 1);
    assert_eq!(grad(&leaves, x), [1001.0]);
    assert!(leaves.get(y).is_none());
}
