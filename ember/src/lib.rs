//! ember: the deep learning framework built in *Build a Deep Learning Framework in
//! Rust*.
//!
//! Each chapter of the book adds a module, in the order the README lists them. The
//! comments `// @listing name` and `// @solution chNN name` mark the regions that
//! the book prints as listings; they have no effect on the code.

pub mod autograd;
mod broadcast;
mod challenges;
pub mod data;
mod matmul;
pub mod nn;
mod ops;
pub mod optim;
pub mod plot;
pub mod profile;
mod rng;
mod tensor;
pub mod toy;
pub mod train;
mod view;

pub use autograd::gradcheck::{Tolerance, VarFn, WorstElement, gradcheck};
pub use autograd::tape::{Gradients, Tape, Var};
pub use broadcast::broadcast_shapes;
pub use challenges::{
    Adagrad, AmsGrad, Averaged, AvgPool2d, BatchNorm2d, BestK, CenteredRmsProp,
    ChannelDropout, ClassicalSgd, CosineRestarts, DropConnect, EarlyStopping, Frozen,
    GlobalAvgPool, Identity, InverseSqrt, KvCache, LeakyReLU, Lion, Lookahead,
    MIN_PER_THREAD, MaxNorm, NamedModule, NormResidual, NormWatch, OneCycle,
    ReduceLrOnPlateau, Residual, RmsNorm, Shifted, SignSgd, Stopped, Tied,
    accumulate_mse, bce_with_logits, broadcast_all, broadcast_axes, clip_grad_value,
    confusion, conv_transpose2d, cross_entropy_masked, cross_entropy_smoothed,
    cumulative, decay_groups, depthwise_conv2d, directional_check,
    eval_is_deterministic, every_mask, fit_early, fnv1a, fold_batch_norm, gd_step,
    generate_cached, gradcheck_params, group_norm, he_std_leaky, heavy_ball_tuning,
    held_out_loss, im2col_parallel, l2_penalty, l2_step, load_matching, lr_range,
    masked_attention, matmul_blocked_with, matmul_nt_kernel, matmul_stacks,
    matmul_tn_kernel, mc_dropout, moments, moons, naive_shuffle, neuron, nll_loss,
    noise_scale, nudge_grad, numeric_grads, orthogonal, padding_mask, par_map,
    pointwise, prefetched_epoch, prefixed, probabilities, receptive_field, rope,
    sample, seal, shift, sinusoidal, softmax_jacobian, softmax_t, spirals_k,
    stratified_order, to_safetensors, top_k_accuracy, truncated_normal,
    unravel_index, unseal, with_channel_dropout, zero_grad_to_zeros, zip_slots,
};
pub use data::{
    Batch, Batches, DataLoader, Dataset, Subset, TensorDataset, random_split,
};
pub use matmul::{
    Dims, Kernel, MIN_FLOPS_PER_THREAD, PANEL, TILE_M, TILE_N, matmul_blocked,
    matmul_ikj, matmul_naive, matmul_parallel, set_threads, threads, threads_for,
};
pub use nn::{
    BatchNorm1d, Block, Config, Conv2d, Dropout, Embedding, Flatten, LayerNorm,
    Linear, MaxPool2d, Mlp, Module, MultiHeadAttention, ParamId, Parameter, ReLU,
    Running, Sequential, Sigmoid, Tanh, TinyTransformer, Unflatten, Window,
    attention, attention_weights, causal_mask, cross_entropy, mse_loss, one_hot,
};
pub use optim::{
    Adam, AdamState, AdamW, ConstantLr, CosineLr, LinearWarmup, LrScheduler,
    Momentum, Optimizer, ParamGroup, RmsProp, RmsPropState, Sgd, SgdState, Stateful,
    StepLr, base_lrs, clip_grad_norm, no_decay,
};
pub use rng::Rng;
pub use tensor::{Tensor, contiguous_strides};
pub use train::{
    Average, CheckpointError, EvalMode, StateDict, accuracy, evaluate,
    load_state_dict, state_dict, train_epoch, train_step,
};
