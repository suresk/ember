//! Challenge 15.5: allocations per update, counted by a global allocator of its
//! own (a separate test binary, like Chapter 7's, so no other test is counted).
use ember::{Module, Optimizer, Rng, Sgd, Tape, cross_entropy, toy};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;

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
fn allocations(f: impl FnOnce()) -> (usize, usize) {
    let (count, bytes) = (ALLOCATIONS.get(), BYTES.get());
    f();
    (ALLOCATIONS.get() - count, BYTES.get() - bytes)
}

// @challenge 5
#[test]
fn allocations_per_update() {
    // @listing alloc-test
    let mut rng = Rng::new(14);
    let (x, labels) = toy::spirals(25, 1.5, 0.05, &mut rng);
    let mut model = spirals::mlp(16, &mut rng);
    let grads = |m: &mut dyn Module| {
        m.zero_grad();
        let tape = Tape::new();
        let loss = cross_entropy(m.forward(tape.constant(x.clone())), &labels);
        loss.backward_into(&mut m.parameters());
    };
    grads(&mut model);
    let by_hand = allocations(|| {
        for p in model.parameters() {
            let g = p.grad.as_ref().unwrap();
            p.value = p.value.sub(&g.mul_scalar(0.1));
        }
    });
    let mut opt = Sgd::new(0.1);
    let first = allocations(|| opt.step(&mut model.parameters()));
    grads(&mut model);
    let later = allocations(|| opt.step(&mut model.parameters()));
    let list = allocations(|| drop(model.parameters()));
    // @hide
    // Chapter 24's `is_contiguous` builds no vector of strides, which saves 18
    // allocations (216 bytes) by hand and 12 (144 bytes) per `Sgd` step. Add them
    // back, so that this test asserts Chapter 15's counts before Chapter 24 and
    // after it; Chapter 24's `earlier_counts_now` asserts the new ones.
    let x = ember::Tensor::zeros(&[2, 3]);
    let saved = allocations(|| assert!(x.is_contiguous())).0 == 0;
    let back = |(n, bytes), (dn, db)| {
        if saved {
            (n + dn, bytes + db)
        } else {
            (n, bytes)
        }
    };
    let (by_hand, first) = (back(by_hand, (18, 216)), back(first, (12, 144)));
    let later = back(later, (12, 144));
    // @show
    assert_eq!(by_hand, (71, 3960)); // 2 · 354 · 4 bytes are buffers
    // The first step also fills the map. A HashMap's size depends on the CPU (its
    // control bytes come in groups of 16 with SSE2 and 8 without), so count only.
    assert_eq!(first.0, 19);
    assert!(first.1 > later.1);
    assert_eq!(later, (17, 288));
    assert_eq!(list, (5, 144)); // model.parameters() alone
    // @end
}
