//! Part II: automatic differentiation. Chapter 6 builds `scalar`, a small engine on
//! single numbers that exists to show the mechanics; Chapter 7 replaces it with a tape
//! of tensors, Chapter 8 gives it its elementwise operations, Chapter 9 the
//! operations that change shapes, and Chapter 10 the matrix product and `gradcheck`,
//! Chapter 13 softmax and log-softmax, Chapter 23 normalization, and Chapter 25 the
//! operations that pick elements: `index_select`, `masked_fill` and `chunk`.

pub mod gradcheck;
mod matmul;
mod norm;
mod ops;
pub mod scalar;
mod select;
mod shape_ops;
mod softmax;
pub mod tape;
