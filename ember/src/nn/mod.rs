//! Part III: layers and losses. Chapter 11 gives a model's weights a home that
//! outlives each step's tape, `Parameter`, and a trait for the things that hold them,
//! `Module`. Chapter 12 adds initialization and the layers networks are made of,
//! Chapter 13 the losses, Chapter 21 dropout, the first module that behaves
//! differently in training and in evaluation, Chapter 22 convolution and pooling,
//! Chapter 23 normalization, and Chapter 25 embeddings, attention and a tiny
//! transformer.

#[macro_use]
mod composite;

mod activation;
mod attention;
mod conv;
mod dropout;
mod embedding;
pub mod init;
mod linear;
mod loss;
mod module;
mod norm;
mod parameter;
mod pool;
mod sequential;
mod transformer;

pub use activation::{ReLU, Sigmoid, Tanh};
pub use attention::{MultiHeadAttention, attention, attention_weights, causal_mask};
pub use conv::{Conv2d, Window};
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use linear::Linear;
pub use loss::{cross_entropy, mse_loss, one_hot};
pub use module::Module;
pub use norm::{BatchNorm1d, LayerNorm, Running};
pub use parameter::{ParamId, Parameter};
pub use pool::{Flatten, MaxPool2d, Unflatten};
pub use sequential::Sequential;
pub use transformer::{Block, Config, Mlp, TinyTransformer};
