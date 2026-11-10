//! Chapter 20: the bytes a tape holds during evaluation, with and without
//! `no_grad`, counted by a global allocator of its own (a separate test binary,
//! like Chapter 7's and 19's, so no other test is counted).
use ember::{Linear, Module, ReLU, Rng, Sequential, Tape, Tensor, cross_entropy};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.set(LIVE.get() + layout.size() as isize);
        unsafe { System.alloc(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        LIVE.set(LIVE.get() + size as isize - layout.size() as isize);
        unsafe { System.realloc(ptr, layout, size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.set(LIVE.get() - layout.size() as isize);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn mlp(rng: &mut Rng) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(784, 128, rng)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, rng)),
    ])
}

/// The bytes still allocated by a forward pass and its loss, on a batch of
/// `n` rows, while the tape lives; and the tape's length.
fn held(
    model: &dyn Module,
    x: &Tensor,
    labels: &[usize],
    no_grad: bool,
) -> (isize, usize) {
    let before = LIVE.get();
    let tape = Tape::new();
    let forward = || {
        let logits = model.forward(tape.constant(x.clone()));
        cross_entropy(logits, labels).value().item()
    };
    let _loss = if no_grad {
        tape.no_grad(forward)
    } else {
        forward()
    };
    (LIVE.get() - before, tape.len())
}

#[test]
fn what_a_tape_holds() {
    ember::set_threads(1); // Chapter 24: starting threads allocates too
    let mut rng = Rng::new(20);
    let model = mlp(&mut rng);
    let x = Tensor::rand(&[100, 784], &mut rng);
    let labels: Vec<usize> = (0..100).map(|i| i % 10).collect();
    let with = held(&model, &x, &labels, false);
    let without = held(&model, &x, &labels, true);
    assert_eq!(with, (174_284, 15)); // bytes, nodes
    assert_eq!(without, (164_484, 15)); // 9,800 fewer bytes, the same nodes
}
