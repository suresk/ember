//! The examples printed in Chapter 7 of the guided edition: a tape of tensor
//! operations.
use ember::autograd::scalar::Value;
use ember::{Tape, Tensor, Var};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// A tensor of shape `[]` holding `x`.
fn scalar(x: f32) -> Tensor {
    Tensor::from_vec(vec![x], &[])
}

fn grad(grads: &ember::Gradients, v: Var) -> Vec<f32> {
    grads.get(v).expect("a gradient").to_vec()
}

#[test]
fn leaves_and_constants() {
    // @listing inputs-example
    let tape = Tape::new();
    let w = tape.leaf(Tensor::arange(&[2, 3]));
    let x = tape.constant(Tensor::from_vec(vec![1.0, 0.5, -1.0], &[3]));
    assert_eq!((w.id(), x.id(), tape.len()), (0, 1, 2));
    assert!(w.requires_grad() && !x.requires_grad());
    assert!(w.value().shares_storage(&w.value())); // a pointer, not the numbers
    let w2 = w; // Var is Copy: w is still usable
    assert_eq!((w2.id(), w.shape()), (0, vec![2, 3]));
    // @end
}

/// The graph of Chapter 6, on a tape: L = (a·b + c)·f.
fn graph(tape: &Tape) -> [Var<'_>; 7] {
    let a = tape.leaf(scalar(3.0));
    let b = tape.leaf(scalar(-2.0));
    let c = tape.leaf(scalar(8.0));
    let f = tape.leaf(scalar(4.0));
    let e = a.mul(b);
    let d = e.add(c);
    let l = d.mul(f);
    [a, b, c, f, e, d, l]
}

#[test]
fn the_forward_pass_on_a_tape() {
    // @listing forward
    let tape = Tape::new();
    let a = tape.leaf(scalar(3.0));
    let b = tape.leaf(scalar(-2.0));
    let c = tape.leaf(scalar(8.0));
    let f = tape.leaf(scalar(4.0));
    let e = a.mul(b);
    let d = e.add(c);
    let l = d.mul(f);
    assert_eq!(tape.len(), 7);
    assert_eq!([e.id(), d.id(), l.id()], [4, 5, 6]);
    assert_eq!((d.op(), d.parents()), ("add", vec![4, 2]));
    assert_eq!(l.value().item(), 8.0);
    assert_eq!(l.shape(), [0usize; 0]); // shape [], one number
    // @end
    assert_eq!((a.op(), a.parents()), ("leaf", vec![]));
    assert_eq!((e.op(), e.parents()), ("mul", vec![0, 1]));
    assert_eq!((l.op(), l.parents()), ("mul", vec![5, 3]));
    let values: Vec<f32> = graph(&tape).iter().map(|v| v.value().item()).collect();
    assert_eq!(values, [3.0, -2.0, 8.0, 4.0, -6.0, 2.0, 8.0]);
}

#[test]
fn backward_on_the_tape() {
    // @listing backward-example
    let tape = Tape::new();
    let [a, b, c, f, e, d, l] = graph(&tape);
    let grads = l.backward();
    assert_eq!(grad(&grads, l), [1.0]);
    assert_eq!([grad(&grads, d), grad(&grads, f)], [[4.0], [2.0]]);
    assert_eq!([grad(&grads, e), grad(&grads, c)], [[4.0], [4.0]]);
    assert_eq!([grad(&grads, a), grad(&grads, b)], [[-8.0], [12.0]]);
    // @end

    // A second pass gives the same numbers: nothing was stored in the nodes.
    let again = l.backward();
    assert_eq!(grad(&again, a), [-8.0]);
}

#[test]
fn gradients_of_tensors() {
    // @listing tensor-example
    let tape = Tape::new();
    let a = tape.leaf(Tensor::arange(&[2, 3])); // 0..6
    let bs = vec![1.0, -1.0, 2.0, 0.5, 3.0, -2.0];
    let b = tape.leaf(Tensor::from_vec(bs, &[2, 3]));
    let k = tape.constant(Tensor::from_vec(vec![10.0; 6], &[2, 3]));
    let l = a.mul(b).add(k).sum_all(); // sum of a·b + 10
    assert_eq!(l.value().item(), 66.5); // 0 - 1 + 4 + 1.5 + 12 - 10, plus 60
    let grads = l.backward();
    assert_eq!(grad(&grads, a), b.value().to_vec()); // dL/da = b
    assert_eq!(grad(&grads, b), a.value().to_vec()); // dL/db = a
    assert!(grads.get(k).is_none()); // a constant gets no gradient
    assert_eq!(grads.get(a).unwrap().shape(), [2, 3]);
    // @end
}

#[test]
fn a_tensor_used_twice() {
    // @listing fan-out
    let tape = Tape::new();
    let x = tape.leaf(Tensor::from_vec(vec![-1.0, 0.0, 2.0], &[3]));
    let l = x.mul(x).add(x).sum_all(); // the sum of x² + x
    assert_eq!(l.value().item(), 6.0);
    let grads = l.backward();
    assert_eq!(grad(&grads, x), [-1.0, 1.0, 5.0]); // 2x + 1, from three uses
    // @end
}

#[test]
fn nodes_past_the_output_are_skipped() {
    let tape = Tape::new();
    let x = tape.leaf(scalar(3.0));
    let y = x.mul(x);
    let z = y.mul(x); // recorded after y
    let grads = y.backward();
    assert_eq!(grad(&grads, x), [6.0]);
    assert!(grads.get(z).is_none());
}

#[test]
#[should_panic(expected = "backward: the output has shape [3]")]
fn backward_needs_one_number() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::arange(&[3]));
    x.mul(x).backward();
}

#[test]
#[should_panic(expected = "add: Vars of two tapes")]
fn two_tapes_dont_mix() {
    let (t1, t2) = (Tape::new(), Tape::new());
    let x = t1.leaf(scalar(1.0));
    let y = t2.leaf(scalar(2.0));
    x.add(y);
}

#[test]
fn recording_off() {
    // @listing no-grad-example
    let tape = Tape::new();
    let w = tape.leaf(Tensor::from_vec(vec![2.0, -1.0], &[2]));
    let x = tape.constant(Tensor::from_vec(vec![3.0, 4.0], &[2]));
    let y = tape.no_grad(|| w.mul(x).sum_all()); // a prediction, say
    assert_eq!(y.value().item(), 2.0);
    assert!(!y.requires_grad()); // although w requires a gradient
    assert_eq!(tape.len(), 4); // the nodes are still recorded
    assert!(tape.is_grad_enabled()); // and recording is back on
    assert!(w.mul(x).requires_grad());
    // @end
    let z = w.mul(x).sum_all();
    assert!(z.requires_grad());
    assert_eq!(grad(&z.backward(), w), [3.0, 4.0]);
    assert!(!x.mul(x).requires_grad()); // constants only
    assert!(w.requires_grad() && !x.requires_grad());
}

#[test]
fn a_chain_of_100_000_nodes() {
    // @listing chain
    let tape = Tape::new();
    let x = tape.leaf(scalar(2.0));
    let mut y = x;
    for _ in 0..100_000 {
        y = y.add(x); // each sum is an input of the next, as in Section 6.9
    }
    assert_eq!(tape.len(), 100_001);
    assert_eq!(y.value().item(), 200_002.0); // 100,001 · x
    let grads = y.backward(); // a loop over 100,001 ids
    assert_eq!(grad(&grads, x), [100_001.0]);
    drop(tape); // a loop too: dropping a Vec
    // @end
}

#[test]
fn what_a_handle_costs() {
    // @listing sizes
    assert_eq!(size_of::<Var>(), 16); // &Tape and usize
    assert_eq!(size_of::<Value>(), 8); // Chapter 6's handle, an Rc
    assert_eq!(size_of::<Tensor>(), 64); // the value in each node
    // @end
}

// @listing counting
/// The system allocator, counting the allocations made on each thread.
struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// How many allocations `f` makes on this thread.
fn allocations(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.get();
    f();
    ALLOCATIONS.get() - before
}
// @end

#[test]
fn allocations_for_a_sum() {
    // @listing allocations
    let scalar_engine = allocations(|| {
        let mut total = Value::new(0.0);
        for i in 0..1000 {
            total = total + Value::new(i as f32);
        }
        total.backward();
    });
    let tape_sum = |n: usize| {
        allocations(|| {
            let tape = Tape::new();
            tape.leaf(Tensor::arange(&[n])).sum_all().backward();
        })
    };
    assert_eq!(scalar_engine, 4_022); // about 4 per number
    assert_eq!((tape_sum(1000), tape_sum(1_000_000)), (17, 17)); // 18 until Ch. 24
    // @end
}
