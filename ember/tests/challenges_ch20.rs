//! Tests for the Chapter 20 challenges.
use ember::train::{read_state, to_bytes};
use ember::{
    AdamW, BestK, CheckpointError, EarlyStopping, Linear, Module, Rng, Tensor,
    accuracy, confusion, load_matching, seal, state_dict, to_safetensors,
    top_k_accuracy, train_step, unseal,
};
use std::path::PathBuf;

#[path = "../examples/mnist_train.rs"]
#[allow(dead_code)]
mod mnist_train;

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

/// Section 20.6's tiny layer: weight [[1], [-2]], bias [0.5].
fn tiny() -> Linear {
    Linear::from_tensors(t(&[1., -2.], &[2, 1]), t(&[0.5], &[1]))
}

#[test]
fn predict_the_checkpoint_sizes() {
    // @listing predict-test
    let mut rng = Rng::new(20);
    let mut model = mnist_train::mlp(&mut rng);
    let mut opt = AdamW::new(1e-3);
    let batch = (Tensor::rand(&[2, 784], &mut rng), vec![3, 7]);
    train_step(&mut model, &batch, &mut opt, 1.0); // so that Adam has state
    let best = to_bytes(&state_dict(&mut model)).len();
    let names = 8 + 6 + 8 + 6; // "0.weight", "0.bias", "2.weight", "2.bias"
    assert_eq!(best, 12 + 4 * (4 + 4) + names + 4 * 6 + 4 * 101_770);
    assert_eq!(best, 407_176);
    let at = mnist_train::Progress::new(rng);
    let last = mnist_train::checkpoint(&mut model, &opt, &at);
    assert_eq!((last.len(), to_bytes(&last).len()), (20, 1_221_780));
    // The last of 938 batches, 32 examples, in a mean of batch means:
    let weight: f64 = 1.0 / 938.0 / 32.0; // per example, against 1/60,000
    assert!((weight * 60_000.0 - 1.9989).abs() < 1e-4); // twice its share
    // @end
}

#[test]
fn a_confusion_matrix() {
    // @listing confusion-test
    let logits = t(&[1., 2., 3., 3., 2., 1., 0., 5., 1., 2., 2., 0.], &[4, 3]);
    let labels = [2, 0, 1, 1];
    let m = confusion(&logits, &labels, 3);
    assert_eq!(m, [[1, 0, 0], [1, 1, 0], [0, 0, 1]]); // a 1 taken for a 0
    assert_eq!(
        top_k_accuracy(&logits, &labels, 1),
        accuracy(&logits, &labels)
    );
    assert_eq!(top_k_accuracy(&logits, &labels, 2), 1.0);
    // @end
}

#[test]
fn early_stopping_with_patience() {
    // @listing early-test
    let losses = [1.0, 0.8, 0.85, 0.79, 0.9, 0.95, 0.7];
    let mut stop = EarlyStopping::new(2);
    let mut last = 0;
    for (i, &loss) in losses.iter().enumerate() {
        last = i + 1; // epochs count from 1
        if stop.update(last, loss) {
            break;
        }
    }
    assert_eq!((last, stop.best_epoch, stop.best), (6, 4, 0.79)); // 0.7 unseen
    // @end
}

/// A fresh folder in the system's temporary directory, removed by `Drop`.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn the_best_two() {
    let id = std::process::id();
    let scratch = Scratch(std::env::temp_dir().join(format!("ember-ch20-best-{id}")));
    std::fs::create_dir_all(&scratch.0).unwrap();
    let dir = scratch.0.clone();
    // @listing best-k-test
    let state = state_dict(&mut tiny());
    let mut best = BestK::new(2, dir.clone());
    let losses = [0.5, 0.4, 0.45, 0.3, 0.6];
    let saved = (1..=5).map(|e| best.offer(e, losses[e - 1], &state).unwrap());
    assert_eq!(saved.collect::<Vec<_>>(), [true, true, true, true, false]);
    assert_eq!(best.kept, [(0.3, 4), (0.4, 2)]); // (loss, epoch)
    let files = std::fs::read_dir(&dir)
        .unwrap()
        .map(|f| f.unwrap().file_name());
    let mut files: Vec<_> = files.collect();
    files.sort();
    assert_eq!(files, ["epoch-2.embr", "epoch-4.embr"]);
    // @end
}

#[test]
fn corrupted_checkpoints() {
    // @listing corrupt-test
    let bytes = to_bytes(&state_dict(&mut tiny()));
    for n in 0..bytes.len() {
        assert!(read_state(&bytes[..n]).is_err()); // every truncation fails
    }
    let mut flipped = bytes.clone();
    flipped[61] ^= 0x01; // the top byte of -2.0: c0 becomes c1
    let read = read_state(&flipped).unwrap(); // reads fine, wrongly
    assert_eq!(read["weight"].to_vec(), [1.0, -8.0]);
    let sealed = seal(bytes.clone());
    assert_eq!(unseal(&sealed).unwrap(), &bytes[..]);
    let mut bad = sealed.clone();
    bad[61] ^= 0x01;
    let e = unseal(&bad).unwrap_err().to_string();
    assert_eq!(e, "corrupt at byte 62: a wrong checksum");
    // @end
}

#[test]
fn a_new_head() {
    // @listing new-head-test
    let mut rng = Rng::new(20);
    let mut pretrained = mnist_train::mlp(&mut rng);
    let mut state = state_dict(&mut pretrained);
    let mut model = ember::Sequential::new(vec![
        Box::new(Linear::new(784, 128, &mut rng)),
        Box::new(ember::ReLU),
        Box::new(Linear::new(128, 2, &mut rng)), // two classes, not ten
    ]);
    let e = load_matching(&mut model, &state).unwrap_err();
    assert_eq!(
        e.to_string(),
        "2.weight: shape [128, 10], expected [128, 2]"
    );
    state.retain(|name, _| name.starts_with("0.")); // keep the body only
    let (missing, unexpected) = load_matching(&mut model, &state).unwrap();
    assert_eq!(
        (missing, unexpected.len()),
        (vec!["2.weight".into(), "2.bias".into()], 0)
    );
    let first = |m: &mut ember::Sequential| m.parameters()[0].value.to_vec();
    assert_eq!(first(&mut model), first(&mut pretrained));
    // @end
    assert!(matches!(
        ember::load_state_dict(&mut model, &state),
        Err(CheckpointError::Keys { .. })
    ));
}

#[test]
fn safetensors_bytes() {
    // @listing safetensors-test
    let bytes = to_safetensors(&state_dict(&mut tiny()));
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + n]).unwrap();
    assert_eq!(
        header.trim_end(),
        concat!(
            r#"{"bias":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"#,
            r#""weight":{"dtype":"F32","shape":[2,1],"data_offsets":[4,12]}}"#
        )
    );
    assert_eq!((header.trim_end().len(), n), (118, 120)); // two spaces
    assert_eq!(bytes.len(), 8 + 120 + 12);
    assert_eq!(
        bytes[8 + n..],
        [0.5f32, 1.0, -2.0].map(f32::to_le_bytes).concat()
    );
    // @end
}
