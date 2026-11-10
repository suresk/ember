//! Chapter 20: ember's checkpoint file, byte by byte.
//!
//! ```text
//! "EMBR"            4 bytes, the magic
//! version           u32, 1
//! count             u32, how many tensors
//! then per tensor, in name order:
//!   name length     u32, in bytes
//!   name            UTF-8
//!   ndim            u32
//!   dims            ndim × u32
//!   data            numel × f32, row-major
//! ```
//!
//! Every number is little-endian.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use super::{CheckpointError, StateDict};
use crate::tensor::Tensor;

// @listing write
pub const MAGIC: &[u8; 4] = b"EMBR";
pub const VERSION: u32 = 1;

/// Write `state` in ember's checkpoint format.
pub fn write_state(out: &mut impl Write, state: &StateDict) -> io::Result<()> {
    out.write_all(MAGIC)?;
    put(out, VERSION as usize)?;
    put(out, state.len())?;
    for (name, t) in state {
        put(out, name.len())?; // bytes, not characters
        out.write_all(name.as_bytes())?;
        put(out, t.ndim())?;
        for &d in t.shape() {
            put(out, d)?;
        }
        for x in t.to_vec() {
            out.write_all(&x.to_le_bytes())?;
        }
    }
    Ok(())
}
// @end

/// A size as a little-endian `u32`.
fn put(out: &mut impl Write, x: usize) -> io::Result<()> {
    let x = u32::try_from(x).expect("checkpoint: a size over u32::MAX");
    out.write_all(&x.to_le_bytes())
}

/// The checkpoint's bytes, in memory.
pub fn to_bytes(state: &StateDict) -> Vec<u8> {
    let mut bytes = vec![];
    write_state(&mut bytes, state).expect("a Vec never fails to write");
    bytes
}

// @listing save
/// Write `state` to `path`, through a temporary file that is renamed over
/// `path` only when complete, so a crash never leaves half a checkpoint.
pub fn save(
    path: impl AsRef<Path>,
    state: &StateDict,
) -> Result<(), CheckpointError> {
    let path = path.as_ref();
    let partial = path.with_extension("partial");
    let io = |source| CheckpointError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut out = BufWriter::new(File::create(&partial).map_err(io)?);
    write_state(&mut out, state).map_err(io)?;
    out.flush().map_err(io)?; // BufWriter's last bytes, or their error
    drop(out);
    fs::rename(&partial, path).map_err(io)
}
// @end

/// Read a checkpoint written by `save`.
pub fn load(path: impl AsRef<Path>) -> Result<StateDict, CheckpointError> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|source| CheckpointError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    read_state(&bytes)
}

// @listing cursor
/// A position in a checkpoint's bytes; every read checks the length first.
struct Cursor<'b> {
    bytes: &'b [u8],
    at: usize,
}

impl<'b> Cursor<'b> {
    fn take(&mut self, need: usize) -> Result<&'b [u8], CheckpointError> {
        let (at, len) = (self.at, self.bytes.len());
        if need > len - at {
            return Err(CheckpointError::Truncated { at, need, len });
        }
        self.at += need;
        Ok(&self.bytes[at..at + need])
    }

    fn u32(&mut self) -> Result<usize, CheckpointError> {
        let four = self.take(4)?.try_into().unwrap(); // 4 bytes by construction
        Ok(u32::from_le_bytes(four) as usize)
    }
}

fn corrupt(at: usize, what: &'static str) -> CheckpointError {
    CheckpointError::Corrupt { at, what }
}
// @end

// @listing read
/// Parse a checkpoint's bytes.
pub fn read_state(bytes: &[u8]) -> Result<StateDict, CheckpointError> {
    if !bytes.starts_with(MAGIC) {
        let found = bytes[..bytes.len().min(4)].to_vec();
        return Err(CheckpointError::NotACheckpoint { found });
    }
    let mut c = Cursor { bytes, at: 4 };
    let found = c.u32()? as u32;
    if found != VERSION {
        return Err(CheckpointError::Version { found });
    }
    let mut state = StateDict::new();
    for _ in 0..c.u32()? {
        let (at, len) = (c.at, c.u32()?);
        let name = std::str::from_utf8(c.take(len)?);
        let name = name.map_err(|_| corrupt(at, "a name not in UTF-8"))?;
        let t = read_tensor(&mut c)?;
        if state.insert(name.to_string(), t).is_some() {
            return Err(corrupt(at, "a name used twice"));
        }
    }
    match c.at == bytes.len() {
        true => Ok(state),
        false => Err(corrupt(c.at, "bytes after the last tensor")),
    }
}
// @end

// @listing read-tensor
/// One tensor's ndim, dims and data. The data's size is checked against what
/// is left before anything is allocated for it.
fn read_tensor(c: &mut Cursor) -> Result<Tensor, CheckpointError> {
    let ndim = c.u32()?;
    let shape: Vec<usize> = (0..ndim).map(|_| c.u32()).collect::<Result<_, _>>()?;
    let at = c.at;
    let size = shape.iter().try_fold(4usize, |n, &d| n.checked_mul(d));
    let size = size.ok_or(corrupt(at, "dims too large"))?;
    let (words, _) = c.take(size)?.as_chunks::<4>(); // [u8; 4]s, no rest
    let values = words.iter().map(|&w| f32::from_le_bytes(w));
    Ok(Tensor::from_vec(values.collect(), &shape))
}
// @end
