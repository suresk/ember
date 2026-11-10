//! Chapter 24: allocations in a training step, counted by a global allocator of
//! its own (a separate test binary, like Chapters 7, 15, 19 and 20's, so no
//! other test is counted).
use ember::{
    AdamW, Linear, Module, ReLU, Rng, Sequential, Tensor, contiguous_strides,
    set_threads, train_step,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        BYTES.set(BYTES.get() + layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// How many allocations `f` makes on this thread, and how many bytes in all.
fn allocations<R>(f: impl FnOnce() -> R) -> (usize, usize) {
    let (count, bytes) = (ALLOCATIONS.get(), BYTES.get());
    drop(f());
    (ALLOCATIONS.get() - count, BYTES.get() - bytes)
}

fn mlp(rng: &mut Rng) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(784, 128, rng)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, rng)),
    ])
}

#[test]
fn allocations_per_step() {
    set_threads(1); // starting threads allocates too
    // @listing alloc-test
    let x = Tensor::arange(&[64, 8, 28, 28]).permute(&[0, 2, 3, 1]);
    assert_eq!(allocations(|| x.is_contiguous()), (0, 0)); // now
    assert_eq!(allocations(|| contiguous_strides(x.shape())), (1, 32)); // before
    let mut rng = Rng::new(24);
    let mut model = mlp(&mut rng);
    let batch = (Tensor::rand(&[64, 784], &mut rng), vec![7; 64]);
    let mut opt = AdamW::new(1e-3);
    train_step(&mut model, &batch, &mut opt, 1.0); // fills Adam's state
    let step = allocations(|| train_step(&mut model, &batch, &mut opt, 1.0));
    assert_eq!(step, (356, 880_858)); // one step of Chapter 19's network
    // What Chapter 23's step allocated besides: a copy of every gradient, ...
    let params = model.parameters();
    let copy = |t: &Tensor| allocations(|| Tensor::from_vec(t.to_vec(), t.shape()));
    let copies: usize = params.iter().map(|p| copy(&p.value).1).sum();
    assert_eq!(copies, 101_770 * 4 + 304); // the numbers, and 304 bytes besides
    // ... and the input's gradient, after a copy of the transposed weight.
    let (g, w) = (Tensor::zeros(&[64, 128]), &params[0].value);
    let product = allocations(|| g.matmul(&w.transpose(0, 1))).1;
    assert_eq!(product, (128 * 784 + 64 * 784) * 4 + 240);
    // @end
}
