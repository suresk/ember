//! Chapter 3 challenges.

use crate::broadcast::broadcast_shapes;
use crate::tensor::Tensor;

// @listing broadcast-all
/// The shape that all of `shapes` broadcast to together, or `None` if there is none.
pub fn broadcast_all(shapes: &[&[usize]]) -> Option<Vec<usize>> {
    shapes
        .iter()
        .try_fold(Vec::new(), |acc, s| broadcast_shapes(&acc, s))
}
// @end

// @listing broadcast-axes
/// The axes of `to` along which `expand` from `from` gives stride 0: every axis it
/// adds on the left, and every axis it stretches from size 1.
pub fn broadcast_axes(from: &[usize], to: &[usize]) -> Vec<usize> {
    assert_eq!(
        broadcast_shapes(from, to).as_deref(),
        Some(to),
        "broadcast_axes: {from:?} does not expand to {to:?}"
    );
    let lead = to.len() - from.len();
    (0..to.len())
        .filter(|&d| d < lead || (from[d - lead] == 1 && to[d] != 1))
        .collect()
}
// @end

// @listing zip-slots
/// For each element of the shape `a` and `b` broadcast to, in row-major order, the
/// storage slots it reads in `a` and in `b`. One odometer drives both positions.
pub fn zip_slots(a: &Tensor, b: &Tensor) -> Vec<(usize, usize)> {
    let shape = broadcast_shapes(&a.shape, &b.shape).unwrap_or_else(|| {
        panic!("zip_slots: {:?} and {:?} don't broadcast", a.shape, b.shape)
    });
    let (a, b) = (a.expand(&shape), b.expand(&shape));
    let numel: usize = shape.iter().product();
    let mut out = Vec::with_capacity(numel);
    let mut index = vec![0; shape.len()];
    let (mut at_a, mut at_b) = (a.offset, b.offset);
    while out.len() < numel {
        out.push((at_a, at_b));
        for d in (0..shape.len()).rev() {
            index[d] += 1;
            at_a += a.strides[d];
            at_b += b.strides[d];
            if index[d] < shape[d] {
                break;
            }
            at_a -= index[d] * a.strides[d];
            at_b -= index[d] * b.strides[d];
            index[d] = 0;
        }
    }
    out
}
// @end

impl Tensor {
    // @listing outer
    /// The outer product of two vectors: `out[i, j] = self[i] * other[j]`.
    pub fn outer(&self, other: &Tensor) -> Tensor {
        assert!(
            self.ndim() == 1 && other.ndim() == 1,
            "outer: two vectors only"
        );
        let (m, n) = (self.shape[0], other.shape[0]);
        let rows = self.reshape(&[m, 1]).expand(&[m, n]).to_vec();
        let cols = other.expand(&[m, n]).to_vec();
        let data = rows.iter().zip(&cols).map(|(x, y)| x * y).collect();
        Tensor::from_vec(data, &[m, n])
    }
    // @end

    // @listing repeat
    /// Tile `self` `counts[d]` times along each axis, into a new buffer. `counts` may
    /// be longer than the shape, which adds axes on the left, as in `expand`.
    pub fn repeat(&self, counts: &[usize]) -> Tensor {
        assert!(counts.len() >= self.ndim(), "repeat: too few counts");
        let lead = counts.len() - self.ndim();
        let mut shape = vec![1; lead];
        shape.extend_from_slice(&self.shape);
        // [a, b] becomes [1, a, 1, b], stretched to [c0, a, c1, b], copied, and
        // regrouped as [c0 * a, c1 * b].
        let pairs = shape.iter().zip(counts);
        let split: Vec<usize> = shape.iter().flat_map(|&n| [1, n]).collect();
        let tiled: Vec<usize> = pairs.clone().flat_map(|(&n, &c)| [c, n]).collect();
        let out: Vec<usize> = pairs.map(|(&n, &c)| n * c).collect();
        let copy = self.reshape(&split).expand(&tiled).contiguous();
        copy.reshape(&out)
    }
    // @end

    // @listing writable
    /// True if some axis of size more than 1 has stride 0, so that several elements
    /// read the same slot.
    pub fn is_broadcast(&self) -> bool {
        self.shape
            .iter()
            .zip(&self.strides)
            .any(|(&n, &s)| n > 1 && s == 0)
    }

    /// Give a broadcast tensor a buffer of its own, one slot per element, so that
    /// writing one element changes only that element. Other tensors are left alone.
    pub fn make_writable(&mut self) {
        if self.is_broadcast() {
            *self = self.contiguous();
        }
    }
    // @end
}
