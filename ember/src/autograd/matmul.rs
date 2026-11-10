//! Chapter 10: the backward rule of the matrix product. Both gradients are matrix
//! products again, each with one operand transposed.

use crate::autograd::tape::Var;
use crate::tensor::Tensor;

// @listing matmul
/// A matrix, or a stack of them, with its last two axes swapped: a view.
fn t(x: &Tensor) -> Tensor {
    let n = x.ndim();
    x.transpose(n - 2, n - 1)
}

impl<'t> Var<'t> {
    /// `self @ other`, for `[m, k] @ [k, n]` or `[b, m, k] @ [b, k, n]`. With
    /// `g` the gradient of the result, `self` gets `g @ otherᵀ` and `other` gets
    /// `selfᵀ @ g`.
    pub fn matmul(self, other: Var<'t>) -> Var<'t> {
        // Chapter 24: a gradient no parent wants is not computed.
        let (need_a, need_b) = (self.requires_grad(), other.requires_grad());
        let rule = move |a: &Tensor, b: &Tensor, g: &Tensor| {
            let ga = if need_a { g.matmul(&t(b)) } else { unneeded(a) };
            let gb = if need_b { t(a).matmul(g) } else { unneeded(b) };
            (ga, gb)
        };
        self.binary(other, "matmul", Tensor::matmul, rule)
    }
}

/// A stand-in for a gradient the walk will drop: a 0 expanded to `x`'s shape,
/// one number, which `binary`'s `sum_to` passes through untouched.
fn unneeded(x: &Tensor) -> Tensor {
    Tensor::zeros(&[]).expand(x.shape())
}
// @end

#[cfg(test)]
mod tests {
    use super::t;
    use crate::autograd::gradcheck::{Tolerance, gradcheck};
    use crate::autograd::tape::Var;
    use crate::tensor::Tensor;

    // @listing untransposed
    impl<'t> Var<'t> {
        /// `matmul` with one transpose forgotten: `g @ b` for `a`.
        fn matmul_untransposed(self, other: Var<'t>) -> Var<'t> {
            let rule =
                |a: &Tensor, b: &Tensor, g: &Tensor| (g.matmul(b), t(a).matmul(g));
            self.binary(other, "matmul", Tensor::matmul, rule)
        }
    }
    // @end

    #[test]
    fn a_bug_with_the_right_shape() {
        fn bug<'t>(v: &[Var<'t>]) -> Var<'t> {
            v[0].matmul_untransposed(v[1])
        }
        // @listing untransposed-example
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let worst = gradcheck(bug, &[a.clone(), b.clone()], Tolerance::default())
            .unwrap_err(); // it fails, and says where:
        assert_eq!((worst.input, worst.index.clone()), (0, vec![0, 0]));
        assert_eq!(worst.analytic, 19.0); // (W B)[0][0] = 1·5 + 2·7
        assert!((worst.numeric - 17.0).abs() < 1e-3); // (W Bᵀ)[0][0] = 1·5 + 2·6
        // @end
        assert!((worst.ratio - 11.70).abs() < 0.01, "{worst:?}"); // 2 / 0.171
        let ok =
            gradcheck(|v| v[0].matmul(v[1]), &[a.clone(), b], Tolerance::default());
        assert!(ok.is_ok(), "{ok:?}");
        // A symmetric b hides the bug: b equals its transpose.
        let sym = Tensor::from_vec(vec![5.0, 6.0, 6.0, 8.0], &[2, 2]);
        assert!(gradcheck(bug, &[a, sym], Tolerance::default()).is_ok());
    }

    #[test]
    #[should_panic(expected = "matmul: [2, 2] @ [3, 2]: inner sizes differ")]
    fn the_bug_panics_on_a_rectangle() {
        let tape = crate::autograd::tape::Tape::new();
        let a = tape.leaf(Tensor::arange(&[2, 3]));
        let b = tape.leaf(Tensor::arange(&[3, 2]));
        a.matmul_untransposed(b).sum_all().backward();
    }
}
