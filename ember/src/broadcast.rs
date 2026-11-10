//! Chapter 3: broadcasting. `broadcast_shapes` decides which shape two tensors
//! combine to; `expand` reads a tensor as that shape without copying, by giving every
//! stretched axis a stride of 0.

use crate::tensor::Tensor;

/// The shape that tensors of shapes `a` and `b` broadcast to, or `None` if there is
/// none. The shapes are aligned on the right, a missing axis counts as size 1, and an
/// axis of size 1 stretches to match the other.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    // @solution ch03 broadcast_shapes
    let n = a.len().max(b.len());
    // Axis `d` of `s` padded with 1s on the left to n axes.
    let size =
        |s: &[usize], d: usize| (d + s.len()).checked_sub(n).map_or(1, |i| s[i]);
    let mut out = Vec::with_capacity(n);
    for d in 0..n {
        let (x, y) = (size(a, d), size(b, d));
        out.push(match (x, y) {
            _ if x == y => x,
            (1, _) => y,
            (_, 1) => x,
            _ => return None,
        });
    }
    Some(out)
    // @end
}

impl Tensor {
    /// `self` read as a tensor of `shape`, without copying. `shape` may add axes on
    /// the left and stretch axes of size 1; those axes get stride 0, so every
    /// position along them reads the same slots.
    pub fn expand(&self, shape: &[usize]) -> Tensor {
        // @solution ch03 expand
        assert!(
            shape.len() >= self.ndim(),
            "expand: {:?} to {:?} would drop axes",
            self.shape,
            shape
        );
        let lead = shape.len() - self.ndim();
        let mut strides = vec![0; shape.len()]; // new axes: stride 0
        for (k, &have) in self.shape.iter().enumerate() {
            let want = shape[lead + k];
            if have == want {
                strides[lead + k] = self.strides[k];
            } else {
                // A stretched axis keeps stride 0; only size 1 can stretch.
                assert!(
                    have == 1,
                    "expand: {:?} to {:?}: axis {k} has size {have}, not 1 or {want}",
                    self.shape,
                    shape
                );
            }
        }
        let mut t = self.clone();
        t.shape = shape.to_vec();
        t.strides = strides;
        t
        // @end
    }
}
