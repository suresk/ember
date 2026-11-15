//! Chapter 20: the bytes a tape holds during evaluation, with and without
//! `no_grad`, counted by a global allocator of its own (a separate test binary,
//! like Chapter 7's and 19's, so no other test is counted). The count is one
//! for the whole process, not one per thread: from Chapter 24 on, a matrix
//! product's threads free memory that this thread allocated (the closures they
//! run), and a per-thread count would never see it come back.
use ember::{Linear, Module, ReLU, Rng, Sequential, Tape, Tensor, cross_entropy};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering::Relaxed};

struct Counting;

/// The bytes allocated and not yet freed, by any thread.
static LIVE: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        LIVE.fetch_add(size as isize - layout.size() as isize, Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Relaxed);
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
    let before = LIVE.load(Relaxed);
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
    (LIVE.load(Relaxed) - before, tape.len())
}

#[test]
fn what_a_tape_holds() {
    let mut rng = Rng::new(20);
    let model = mlp(&mut rng);
    let x = Tensor::rand(&[100, 784], &mut rng);
    let labels: Vec<usize> = (0..100).map(|i| i % 10).collect();
    let with = held(&model, &x, &labels, false);
    let without = held(&model, &x, &labels, true);
    // From Chapter 24 on, each of the two matmul rules also holds two flags, 8
    // bytes more in each boxed closure (Section 24.4): 174,284.
    let flags = if with.0 == 174_268 { 0 } else { 16 };
    assert_eq!(with, (174_268 + flags, 15)); // bytes, nodes
    assert_eq!(without, (164_484, 15)); // 9,800 fewer bytes, the same nodes
}
