//! Part V, Chapter 20: the training loop. Metrics (`accuracy`, `Average`),
//! evaluation without gradients, the epoch loop, state dicts that name every
//! parameter, and checkpoint files in a small documented format. Chapter 21 makes
//! `evaluate` switch the model to evaluation mode, with an `EvalMode` guard.

mod file;
mod metrics;
mod run;
mod state;

pub use file::{MAGIC, VERSION, load, read_state, save, to_bytes, write_state};
pub use metrics::{Average, accuracy};
pub use run::{EvalMode, evaluate, train_epoch, train_step};
pub use state::{
    CheckpointError, StateDict, StateReader, load_state_dict, state_dict,
    strip_prefix, tensor_to_u64, u64_to_tensor, with_prefix,
};
