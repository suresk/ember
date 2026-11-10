//! Reference solutions to the end-of-chapter challenges in the guided edition. Each
//! one is written with only what its chapter has built.

mod ch01;
mod ch02;
mod ch03;
mod ch04;
mod ch05;
mod ch06;
mod ch07;
mod ch08;
mod ch09;
mod ch10;
mod ch11;
mod ch12;
mod ch13;
mod ch14;
mod ch15;
mod ch16;
mod ch17;
mod ch18;
mod ch19;
mod ch20;
mod ch21;
mod ch22;
mod ch23;
mod ch24;
mod ch25;

pub use ch01::unravel_index;
pub use ch03::{broadcast_all, broadcast_axes, zip_slots};
pub use ch05::{matmul_blocked_with, matmul_nt_kernel};
pub use ch06::{neuron, numeric_grads};
pub use ch08::nudge_grad;
pub use ch10::directional_check;
pub use ch11::{
    Frozen, NamedModule, Residual, Tied, accumulate_mse, prefixed, zero_grad_to_zeros,
};
pub use ch12::{
    Identity, LeakyReLU, gradcheck_params, he_std_leaky, moments, naive_shuffle,
    orthogonal, truncated_normal,
};
pub use ch13::{
    bce_with_logits, cross_entropy_masked, cross_entropy_smoothed, nll_loss,
    softmax_jacobian, softmax_t,
};
pub use ch14::{
    Stopped, fit_early, gd_step, held_out_loss, l2_penalty, l2_step, lr_range, moons,
    spirals_k,
};
pub use ch15::{MaxNorm, SignSgd, decay_groups};
pub use ch16::{Averaged, ClassicalSgd, Lookahead, heavy_ball_tuning};
pub use ch17::{Adagrad, AmsGrad, CenteredRmsProp, Lion};
pub use ch18::{
    CosineRestarts, InverseSqrt, NormWatch, OneCycle, ReduceLrOnPlateau,
    clip_grad_value,
};
pub use ch19::{Shifted, noise_scale, shift, stratified_order};
pub use ch20::{
    BestK, EarlyStopping, confusion, fnv1a, load_matching, seal, to_safetensors,
    top_k_accuracy, unseal,
};
pub use ch21::{
    ChannelDropout, DropConnect, eval_is_deterministic, every_mask, mc_dropout,
};
pub use ch22::{
    AvgPool2d, GlobalAvgPool, conv_transpose2d, depthwise_conv2d, pointwise,
    receptive_field, with_channel_dropout,
};
pub use ch23::{
    BatchNorm2d, NormResidual, RmsNorm, cumulative, fold_batch_norm, group_norm,
};
pub use ch24::{
    MIN_PER_THREAD, im2col_parallel, matmul_stacks, matmul_tn_kernel, par_map,
    prefetched_epoch,
};
pub use ch25::{
    KvCache, generate_cached, masked_attention, padding_mask, probabilities, rope,
    sample, sinusoidal,
};
