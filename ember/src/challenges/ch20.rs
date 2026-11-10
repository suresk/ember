//! Chapter 20 challenges.

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::nn::Module;
use crate::tensor::Tensor;
use crate::train::{CheckpointError, StateDict, save};

// @listing confusion
/// `m[label][predicted]`: how often each class was taken for each other one.
pub fn confusion(logits: &Tensor, labels: &[usize], n: usize) -> Vec<Vec<usize>> {
    let mut m = vec![vec![0; n]; n];
    for (&p, &l) in logits.argmax(1).iter().zip(labels) {
        m[l][p] += 1;
    }
    m
}

/// The fraction of rows whose label is among their `k` largest logits. A tie
/// goes to the earlier class, as in `argmax`, so `k = 1` is `accuracy`.
pub fn top_k_accuracy(logits: &Tensor, labels: &[usize], k: usize) -> f32 {
    let c = logits.shape()[1];
    let x = logits.contiguous().to_vec();
    let hits = labels.iter().enumerate().filter(|&(i, &l)| {
        let row = &x[i * c..(i + 1) * c];
        let ahead = |(j, &v): (usize, &f32)| v > row[l] || (v == row[l] && j < l);
        row.iter().enumerate().filter(|&p| ahead(p)).count() < k
    });
    hits.count() as f32 / labels.len() as f32
}
// @end

// @listing early-stopping
/// Stop when the validation loss hasn't improved for `patience` epochs.
#[derive(Clone, Debug)]
pub struct EarlyStopping {
    pub patience: usize,
    pub best: f32,
    pub best_epoch: usize,
    waited: usize, // epochs since the best
}

impl EarlyStopping {
    /// Record an epoch's validation loss; true when training should stop.
    pub fn update(&mut self, epoch: usize, loss: f32) -> bool {
        if loss < self.best {
            (self.best, self.best_epoch, self.waited) = (loss, epoch, 0);
        } else {
            self.waited += 1;
        }
        self.waited >= self.patience
    }
}
// @end

impl EarlyStopping {
    pub fn new(patience: usize) -> EarlyStopping {
        let (best, best_epoch, waited) = (f32::INFINITY, 0, 0);
        EarlyStopping {
            patience,
            best,
            best_epoch,
            waited,
        }
    }
}

// @listing best-k
/// Keep the `k` checkpoints with the lowest validation loss, as
/// `epoch-{n}.embr` in `dir`, deleting the rest.
pub struct BestK {
    pub k: usize,
    pub dir: PathBuf,
    pub kept: Vec<(f32, usize)>, // (loss, epoch), best first
}

impl BestK {
    /// Save `state` if it is among the best `k` so far. Returns whether it was.
    pub fn offer(
        &mut self,
        epoch: usize,
        loss: f32,
        state: &StateDict,
    ) -> Result<bool, CheckpointError> {
        let full = self.kept.len() == self.k;
        if full && self.kept.last().is_none_or(|w| loss >= w.0) {
            return Ok(false); // no better than the worst kept
        }
        save(self.path(epoch), state)?;
        let at = self.kept.partition_point(|&(l, _)| l <= loss);
        self.kept.insert(at, (loss, epoch));
        if self.kept.len() > self.k {
            let (_, worst) = self.kept.pop().unwrap();
            let path = self.path(worst);
            let io = |source| CheckpointError::Io {
                path: path.clone(),
                source,
            };
            std::fs::remove_file(&path).map_err(io)?;
        }
        Ok(true)
    }

    pub fn path(&self, epoch: usize) -> PathBuf {
        self.dir.join(format!("epoch-{epoch}.embr"))
    }
}
// @end

impl BestK {
    /// Keep the best `k`, in `dir`; nothing kept yet.
    pub fn new(k: usize, dir: PathBuf) -> BestK {
        BestK {
            k,
            dir,
            kept: vec![],
        }
    }
}

// @listing checksum
/// FNV-1a, 32 bits: for each byte, xor it in, then multiply by a prime.
pub fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h = (h ^ b as u32).wrapping_mul(0x0100_0193);
    }
    h
}

/// The bytes with their checksum appended, little-endian.
pub fn seal(mut bytes: Vec<u8>) -> Vec<u8> {
    let sum = fnv1a(&bytes);
    bytes.extend(sum.to_le_bytes());
    bytes
}

/// The bytes before the checksum, if it matches them.
pub fn unseal(bytes: &[u8]) -> Result<&[u8], CheckpointError> {
    let corrupt = |at, what| CheckpointError::Corrupt { at, what };
    let at = bytes
        .len()
        .checked_sub(4)
        .ok_or(corrupt(0, "no checksum"))?;
    let (body, sum) = bytes.split_at(at);
    match fnv1a(body).to_le_bytes() == sum {
        true => Ok(body),
        false => Err(corrupt(at, "a wrong checksum")),
    }
}
// @end

// @listing load-matching
/// PyTorch's `strict=False`: load the entries whose names the model has, and
/// return the names the model lacks (missing) and the state's extras
/// (unexpected). A shape mismatch is still an error, and changes nothing.
pub fn load_matching(
    model: &mut dyn Module,
    state: &StateDict,
) -> Result<(Vec<String>, Vec<String>), CheckpointError> {
    let mut found = vec![];
    for (name, p) in model.named_parameters() {
        let Some(t) = state.get(&name) else { continue };
        if t.shape() != p.value.shape() {
            let (expected, found) = (p.value.shape().to_vec(), t.shape().to_vec());
            return Err(CheckpointError::Shape {
                name,
                expected,
                found,
            });
        }
        found.push(name);
    }
    let (mut missing, used) = (vec![], BTreeSet::from_iter(found.iter()));
    for (name, p) in model.named_parameters() {
        match state.get(&name) {
            Some(t) => p.value = t.clone(),
            None => missing.push(name),
        }
    }
    let unexpected = state.keys().filter(|k| !used.contains(k)).cloned();
    Ok((missing, unexpected.collect()))
}
// @end

// @listing safetensors
/// `state` in the safetensors format: an 8-byte little-endian header length,
/// a JSON header padded with spaces to a multiple of 8, then the raw data.
pub fn to_safetensors(state: &StateDict) -> Vec<u8> {
    let (mut entries, mut data) = (vec![], vec![]);
    for (name, t) in state {
        let start = data.len();
        t.to_vec().iter().for_each(|x| data.extend(x.to_le_bytes()));
        let dims: Vec<String> = t.shape().iter().map(|d| d.to_string()).collect();
        let (dims, end) = (dims.join(","), data.len());
        let info = format!(
            r#"{{"dtype":"F32","shape":[{dims}],"data_offsets":[{start},{end}]}}"#
        );
        entries.push(format!("{}:{info}", json_string(name)));
    }
    let mut header = format!("{{{}}}", entries.join(","));
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header.as_bytes());
    out.extend(data);
    out
}

/// A JSON string: quotes, with `"`, `\` and control characters escaped.
fn json_string(s: &str) -> String {
    let mut out = String::from('"');
    for c in s.chars() {
        match c {
            '"' | '\\' => out.extend(['\\', c]),
            c if (c as u32) < 0x20 => out += &format!("\\u{:04x}", c as u32),
            c => out.push(c),
        }
    }
    out + "\""
}
// @end
