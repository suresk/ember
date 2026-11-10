//! Chapter 23: normalizing along one axis, on tensors and on the tape. The
//! statistics are computed in two passes, added up in f64, as Chapter 4 promised.

use crate::autograd::tape::Var;
use crate::tensor::Tensor;

// @listing mean-var
/// One lane's mean and biased variance (divided by `n`), in two passes: the
/// mean first, then the squared deviations from it, both added up in f64.
fn mean_var_lane(lane: &[f32]) -> (f32, f32) {
    let n = lane.len() as f64;
    let mean = lane.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = lane.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean as f32, var as f32)
}

impl Tensor {
    /// The mean and the biased variance along `dim`, each with the axis kept.
    pub fn mean_var(&self, dim: usize) -> (Tensor, Tensor) {
        let lanes = self.lanes(dim, mean_var_lane).into_iter();
        let (mean, var): (Vec<f32>, Vec<f32>) = lanes.unzip();
        let shape = self.reduced_shape(dim, true);
        let tensor = |data| Tensor::from_vec(data, &shape);
        (tensor(mean), tensor(var))
    }
}
// @end

impl<'t> Var<'t> {
    // @listing var-normalize
    /// `x_hat = (x - mean) / σ` along `dim`, with `σ = √(var + eps)` and the biased
    /// variance, as one node. Backward: `(g - mean(g) - x_hat·mean(g·x_hat)) / σ`.
    pub fn normalize(self, dim: usize, eps: f32) -> Var<'t> {
        let x = self.value();
        let (mean, var) = x.mean_var(dim);
        let inv_std = var.add_scalar(eps).powf(-0.5); // 1/σ, one per lane
        let x_hat = x.sub(&mean).mul(&inv_std);
        let y = x_hat.clone(); // the rule keeps x_hat and 1/σ, not x
        self.tape.push("normalize", y, vec![self.id], move |g| {
            let g_mean = g.mean(dim, true);
            let gx_mean = g.mul(&x_hat).mean(dim, true);
            let centered = g.sub(&g_mean).sub(&x_hat.mul(&gx_mean));
            vec![centered.mul(&inv_std)]
        })
    }
    // @end
}
