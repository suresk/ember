//! Chapter 18: gradient clipping. Scale every gradient by one factor so that the
//! length of all of them together is at most a limit.

use crate::nn::Parameter;

// @listing clip
/// `torch.nn.utils.clip_grad_norm_`: if the L2 norm of all the gradients
/// together is above `max_norm`, scale every gradient by max_norm/(norm + 1e-6).
/// Returns the norm before clipping. Parameters without a gradient are skipped.
pub fn clip_grad_norm(params: &mut [&mut Parameter], max_norm: f32) -> f32 {
    let mut squares = 0.0f64;
    for p in params.iter() {
        let Some(g) = &p.grad else { continue };
        let g = g.as_slice().expect("clip_grad_norm: a contiguous gradient");
        squares += g.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
    }
    let norm = squares.sqrt() as f32;
    let scale = max_norm / (norm + 1e-6);
    if scale < 1.0 {
        for p in params.iter_mut() {
            let Some(g) = &mut p.grad else { continue };
            g.data_mut().iter_mut().for_each(|x| *x *= scale);
        }
    }
    norm
}
// @end
