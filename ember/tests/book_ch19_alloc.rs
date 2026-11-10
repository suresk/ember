//! Chapter 19: allocations per batch, counted by a global allocator of its own (a
//! separate test binary, like Chapter 7's, so no other test is counted).
use ember::{Dataset, Tensor, TensorDataset};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        unsafe { System.alloc(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        unsafe { System.realloc(ptr, layout, size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// How many allocations (and reallocations) `f` makes on this thread.
fn allocations(f: impl FnOnce()) -> usize {
    let count = ALLOCATIONS.get();
    f();
    ALLOCATIONS.get() - count
}

/// A dataset with only `get`, so its batches come from the default `get_batch`.
struct PerExample<'a>(&'a TensorDataset);

impl Dataset for PerExample<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(&self, i: usize) -> (Tensor, usize) {
        self.0.get(i)
    }
}

#[test]
fn allocations_per_batch() {
    // @listing alloc-test
    let ds = TensorDataset::new(Tensor::zeros(&[1000, 784]), vec![0; 1000]);
    let indices: Vec<usize> = (0..64).map(|i| i * 15).collect();
    let batched = allocations(|| drop(ds.get_batch(&indices)));
    let stacked = allocations(|| drop(PerExample(&ds).get_batch(&indices)));
    assert_eq!((batched, stacked), (6, 273)); // for one batch of 64 (7 until Ch. 24)
    // @end
}
