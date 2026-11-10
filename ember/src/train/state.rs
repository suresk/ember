//! Chapter 20: a model's (or an optimizer's) state as named tensors, and loading
//! it back with errors that say what doesn't fit.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::{error, fmt, io};

use crate::nn::Module;
use crate::tensor::Tensor;

use CheckpointError as Error; // in this file

// @listing state-dict
/// Named tensors, kept sorted by name, so that a file written from one has the
/// same bytes every time.
pub type StateDict = BTreeMap<String, Tensor>;

/// Every parameter's value, and every buffer, by its dotted name. The tensors
/// share their buffers with the parameters until an update writes to them.
pub fn state_dict(model: &mut dyn Module) -> StateDict {
    let named = model.named_parameters().into_iter();
    let mut state: StateDict = named.map(|(n, p)| (n, p.value.clone())).collect();
    let buffers = model.named_buffers().into_iter(); // Chapter 23
    state.extend(buffers.map(|(n, t)| (n, t.clone())));
    state
}
// @end

// @listing error
/// What can go wrong reading a checkpoint, or fitting one to a model.
#[derive(Debug)]
pub enum CheckpointError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    NotACheckpoint {
        found: Vec<u8>,
    },
    Version {
        found: u32,
    },
    Truncated {
        at: usize,
        need: usize,
        len: usize,
    },
    Corrupt {
        at: usize,
        what: &'static str,
    },
    Keys {
        missing: Vec<String>,
        unexpected: Vec<String>,
    },
    Shape {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
}
// @end

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use CheckpointError::*;
        match self {
            Io { path, source } => write!(f, "{}: {source}", path.display()),
            NotACheckpoint { found } => {
                write!(f, "not an ember checkpoint (starts with {found:02x?})")
            }
            Version { found } => write!(
                f,
                "checkpoint version {found}; this ember reads version {}",
                super::VERSION
            ),
            Truncated { at, need, len } => write!(
                f,
                "truncated: {need} bytes needed at byte {at}, but the file has {len}"
            ),
            Corrupt { at, what } => write!(f, "corrupt at byte {at}: {what}"),
            Keys {
                missing,
                unexpected,
            } => write!(
                f,
                "missing keys {missing:?}, unexpected keys {unexpected:?}"
            ),
            Shape {
                name,
                expected,
                found,
            } => write!(f, "{name}: shape {found:?}, expected {expected:?}"),
        }
    }
}

impl error::Error for CheckpointError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            CheckpointError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

// @listing load-state-dict
/// Set every parameter of `model` from `state`. Nothing changes unless every
/// name matches and every shape fits.
pub fn load_state_dict(
    model: &mut dyn Module,
    state: &StateDict,
) -> Result<(), CheckpointError> {
    let mut reader = StateReader::new(state);
    let mut values = vec![];
    for (name, p) in model.named_parameters() {
        values.push(reader.need(&name, p.value.shape())?); // check them all...
    }
    for (name, t) in model.named_buffers() {
        values.push(reader.need(&name, t.shape())?); // Chapter 23
    }
    reader.finish()?;
    let mut values = values.into_iter();
    for (_, p) in model.named_parameters() {
        p.value = values.next().unwrap(); // ...before changing any
    }
    for (_, t) in model.named_buffers() {
        *t = values.next().unwrap(); // Chapter 23
    }
    Ok(())
}
// @end

// @listing reader
/// Reads entries of a state dict by name, checking their shapes, and keeps
/// track, so that `finish` can report the names nobody asked for.
pub struct StateReader<'s> {
    state: &'s StateDict,
    read: BTreeSet<&'s str>, // asked for and found
    missing: Vec<String>,    // asked for and not found
}

impl StateReader<'_> {
    /// The entry `name`, if there is one, after checking its shape.
    pub fn get(
        &mut self,
        name: &str,
        shape: &[usize],
    ) -> Result<Option<Tensor>, Error> {
        let Some((key, t)) = self.state.get_key_value(name) else {
            return Ok(None);
        };
        self.read.insert(key);
        if t.shape() != shape {
            let (expected, found) = (shape.to_vec(), t.shape().to_vec());
            let name = name.to_string();
            return Err(Error::Shape {
                name,
                expected,
                found,
            });
        }
        Ok(Some(t.clone()))
    }
}
// @end

impl<'s> StateReader<'s> {
    pub fn new(state: &'s StateDict) -> StateReader<'s> {
        let (read, missing) = (BTreeSet::new(), vec![]);
        StateReader {
            state,
            read,
            missing,
        }
    }
}

// @listing reader-need
impl StateReader<'_> {
    /// The entry `name`, which must be there, or `finish` reports it missing.
    pub fn need(&mut self, name: &str, shape: &[usize]) -> Result<Tensor, Error> {
        let t = self.get(name, shape)?;
        Ok(t.unwrap_or_else(|| {
            self.missing.push(name.to_string());
            Tensor::zeros(shape) // a stand-in until `finish`
        }))
    }

    /// Ok if every name asked for was there, and every entry was asked for.
    pub fn finish(self) -> Result<(), Error> {
        let unread = |k: &&String| !self.read.contains(k.as_str());
        let unexpected: Vec<String> =
            self.state.keys().filter(unread).cloned().collect();
        match (self.missing, unexpected) {
            (m, u) if m.is_empty() && u.is_empty() => Ok(()),
            (missing, unexpected) => Err(Error::Keys {
                missing,
                unexpected,
            }),
        }
    }
}
// @end

// @listing prefix
/// `state`'s entries with `prefix.` in front of every name.
pub fn with_prefix(prefix: &str, state: StateDict) -> StateDict {
    let dotted = |(name, t)| (format!("{prefix}.{name}"), t);
    state.into_iter().map(dotted).collect()
}

/// The entries whose names start with `prefix.`, without it.
pub fn strip_prefix(prefix: &str, state: &StateDict) -> StateDict {
    let start = format!("{prefix}.");
    let inside = state.iter().filter_map(|(name, t)| {
        let rest = name.strip_prefix(&start)?;
        Some((rest.to_string(), t.clone()))
    });
    inside.collect()
}
// @end

// @listing u64
/// A `u64` as four 16-bit pieces, low first: every piece is an integer below
/// 2¹⁶, which an `f32` holds exactly.
pub fn u64_to_tensor(x: u64) -> Tensor {
    let pieces = (0..4).map(|i| (x >> (16 * i) & 0xFFFF) as f32);
    Tensor::from_vec(pieces.collect(), &[4])
}

pub fn tensor_to_u64(t: &Tensor) -> u64 {
    let pieces = t.to_vec().into_iter().enumerate();
    pieces.map(|(i, p)| (p as u64) << (16 * i)).sum()
}
// @end
