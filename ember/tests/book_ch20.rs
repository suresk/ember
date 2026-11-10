//! The examples printed in Chapter 20 of the guided edition: metrics, evaluation,
//! state dicts, checkpoint files, and a run resumed bit for bit. No test needs the
//! MNIST files; checkpoints go to a temporary folder that is removed.
use ember::train::{self, read_state, tensor_to_u64, to_bytes, u64_to_tensor};
use ember::{
    Adam, AdamW, Average, CheckpointError, DataLoader, Linear, Module, Optimizer,
    Rng, Sequential, Sgd, StateDict, Stateful, Tape, Tensor, TensorDataset, accuracy,
    evaluate, load_state_dict, random_split, state_dict, toy, train_step,
};
use std::path::PathBuf;

#[path = "../examples/mnist_train.rs"]
#[allow(dead_code)]
mod mnist_train;
#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;

use mnist_train::{Plan, Progress, fit, restore};

/// A fresh folder in the system's temporary directory, removed by `Drop`.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let id = std::process::id();
        let dir = std::env::temp_dir().join(format!("ember-ch20-{name}-{id}"));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn bits(model: &mut dyn Module) -> Vec<u32> {
    let params = model.parameters();
    params
        .iter()
        .flat_map(|p| p.value.to_vec())
        .map(f32::to_bits)
        .collect()
}

#[test]
fn a_validation_split() {
    // @listing split-test
    let ds = TensorDataset::new(Tensor::arange(&[10, 1]), (0..10).collect());
    let parts = random_split(&ds, &[7, 3], &mut Rng::new(19));
    assert_eq!(parts[0].indices, [9, 6, 3, 8, 5, 4, 0]); // Section 19.4's order
    assert_eq!(parts[1].indices, [2, 1, 7]);
    let (x, labels) = DataLoader::new(&parts[1], 3).epoch().next().unwrap();
    assert_eq!((x.to_vec(), labels), (vec![2.0, 1.0, 7.0], vec![2, 1, 7]));
    // @end
}

#[test]
fn accuracy_of_a_batch() {
    // @listing accuracy-test
    let logits = t(&[1., 2., 3., 3., 2., 1., 0., 5., 1., 2., 2., 0.], &[4, 3]);
    assert_eq!(logits.argmax(1), [2, 0, 1, 0]); // a tie goes to the first
    assert_eq!(accuracy(&logits, &[2, 0, 1, 1]), 0.75);
    // @end
}

#[test]
#[should_panic(expected = "accuracy: [4, 3], 3 labels")]
fn a_label_per_row() {
    accuracy(&Tensor::zeros(&[4, 3]), &[0, 1, 2]);
}

#[test]
fn a_weighted_mean() {
    // @listing average-test
    // Ten examples in batches of 4, 4 and 2: losses 1 (eight times), then 4, 4.
    let batches = [(1.0, 4), (1.0, 4), (4.0, 2)]; // (the batch's mean, its size)
    let mut loss = Average::new();
    for (mean, n) in batches {
        loss.add(mean, n);
    }
    assert_eq!((loss.mean(), loss.count()), (1.6, 10)); // (8 + 8) / 10
    let means: f32 = batches.iter().map(|b| b.0).sum::<f32>() / 3.0;
    assert_eq!(means, 2.0); // the last two examples count double
    // @end
}

#[test]
fn evaluate_matches_one_big_batch() {
    // @listing evaluate-test
    let mut rng = Rng::new(20);
    let (x, labels) = toy::spirals(50, 1.0, 0.05, &mut rng);
    let mut model = spirals::mlp(16, &mut rng);
    let ds = TensorDataset::new(x.clone(), labels.clone());
    let (loss, right) = evaluate(&mut model, &mut DataLoader::new(&ds, 100));
    let tape = Tape::new();
    let logits = model.forward(tape.constant(x));
    let all = ember::cross_entropy(logits, &labels).value().item();
    assert!((loss - all).abs() < 1e-6 && right == accuracy(&logits.value(), &labels));
    let (by_32, again) = evaluate(&mut model, &mut DataLoader::new(&ds, 32));
    assert!((by_32 - loss).abs() < 1e-6 && again == right); // 32, 32, 32, 4
    // @end
    println!("{loss} {right}");
}

#[test]
fn no_grad_keeps_the_nodes_but_not_the_rules() {
    // @listing no-grad-test
    let model = mnist_train::mlp(&mut Rng::new(20));
    let (x, labels) = (Tensor::zeros(&[5, 784]), [0, 1, 2, 3, 4]);
    let tape = Tape::new();
    let loss = |x| ember::cross_entropy(model.forward(tape.constant(x)), &labels);
    let y = loss(x.clone());
    let n = tape.len();
    let z = tape.no_grad(|| loss(x));
    assert_eq!((n, tape.len()), (15, 30)); // as many nodes again...
    assert!(y.requires_grad() && !z.requires_grad()); // ...without rules
    // @end
}

#[test]
fn names_for_parameters() {
    // @listing named-test
    let mut model = mnist_train::mlp(&mut Rng::new(20));
    let named = model.named_parameters();
    let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["0.weight", "0.bias", "2.weight", "2.bias"]);
    let state = state_dict(&mut model); // sorted: bias before weight
    let keys: Vec<&String> = state.keys().collect();
    assert_eq!(keys, ["0.bias", "0.weight", "2.bias", "2.weight"]);
    // @end
    let named = model.named_parameters();
    let shapes: Vec<&[usize]> = named.iter().map(|(_, p)| p.value.shape()).collect();
    assert_eq!(shapes, [&[784, 128][..], &[128], &[128, 10], &[10]]);
}

/// A module that keeps the default, positional names.
struct Pair(Linear);

impl Module for Pair {
    fn forward<'t>(&self, x: ember::Var<'t>) -> ember::Var<'t> {
        self.0.forward(x)
    }

    fn parameters(&mut self) -> Vec<&mut ember::Parameter> {
        self.0.parameters()
    }
}

#[test]
fn default_names_are_positions() {
    let mut pair = Pair(Linear::new(2, 3, &mut Rng::new(20)));
    let names: Vec<String> =
        pair.named_parameters().into_iter().map(|n| n.0).collect();
    assert_eq!(names, ["0", "1"]);
}

#[test]
fn a_snapshot_survives_the_update() {
    // @listing snapshot-test
    let mut layer = Linear::from_tensors(t(&[1., 2.], &[2, 1]), t(&[0.5], &[1]));
    let snapshot = state_dict(&mut layer);
    assert!(snapshot["weight"].shares_storage(&layer.weight.value)); // no copy
    layer.weight.grad = Some(t(&[1., 1.], &[2, 1]));
    Sgd::new(0.5).step(&mut layer.parameters()); // in place...
    assert_eq!(layer.weight.value.to_vec(), [0.5, 1.5]);
    assert_eq!(snapshot["weight"].to_vec(), [1.0, 2.0]); // ...but not here
    assert!(!snapshot["weight"].shares_storage(&layer.weight.value));
    // @end
}

#[test]
fn loading_checks_names_and_shapes() {
    // @listing load-errors-test
    let mut rng = Rng::new(20);
    let mut model = mnist_train::mlp(&mut rng);
    let mut other = mnist_train::mlp(&mut rng);
    let (before, mut state) = (bits(&mut other), state_dict(&mut model));
    let w = state.remove("2.weight").unwrap();
    state.insert("2.w".into(), w); // a renamed field
    let mut err =
        |state| load_state_dict(&mut other, &state).unwrap_err().to_string();
    let keys = r#"missing keys ["2.weight"], unexpected keys ["2.w"]"#;
    assert_eq!(err(state), keys);
    let mut state = state_dict(&mut model);
    let w = state["0.weight"].transpose(0, 1); // PyTorch's layout, [out, in]
    state.insert("0.weight".into(), w);
    let shape = "0.weight: shape [128, 784], expected [784, 128]";
    assert_eq!(err(state), shape);
    assert_eq!(bits(&mut other), before); // nothing changed
    // @end
    let state = state_dict(&mut model);
    load_state_dict(&mut other, &state).unwrap();
    assert_eq!(bits(&mut other), bits(&mut model));
}

#[test]
fn pytorch_weights_are_transposed() {
    // @listing pytorch-test
    // nn.Linear(3, 2) in PyTorch: weight [out, in] = [2, 3], y = x·Wᵀ + b.
    let torch_w = t(&[1., 0., -1., 2., 1., 0.], &[2, 3]);
    let mut state = StateDict::new();
    state.insert("weight".into(), torch_w.transpose(0, 1).contiguous());
    state.insert("bias".into(), t(&[10., 20.], &[2]));
    let mut layer = Linear::new(3, 2, &mut Rng::new(20));
    load_state_dict(&mut layer, &state).unwrap();
    let tape = Tape::new();
    let y = layer.forward(tape.constant(t(&[3., 4., 5.], &[1, 3])));
    assert_eq!(y.value().to_vec(), [8.0, 30.0]); // 3 - 5 + 10, 6 + 4 + 20
    // @end
}

/// `xxd`'s layout: offset, 16 bytes in hex, the same bytes as text.
fn hexdump(bytes: &[u8]) -> String {
    let mut out = String::new();
    for (row, chunk) in bytes.chunks(16).enumerate() {
        let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
        let text: String = chunk
            .iter()
            .map(|&b| if b.is_ascii_graphic() { b as char } else { '.' })
            .collect();
        out += &format!("{:04x}  {:<47}  {text}\n", row * 16, hex.join(" "));
    }
    out
}

#[test]
fn a_tiny_checkpoint_byte_by_byte() {
    // @listing bytes-test
    let mut layer = Linear::from_tensors(t(&[1., -2.], &[2, 1]), t(&[0.5], &[1]));
    let bytes = to_bytes(&state_dict(&mut layer));
    assert_eq!(bytes.len(), 12 + (4 + 4 + 4 + 4 + 4) + (4 + 6 + 4 + 8 + 8)); // 62
    assert_eq!(&bytes[..4], b"EMBR");
    assert_eq!(bytes[58..], (-2.0f32).to_le_bytes()); // 00 00 00 c0
    assert_eq!(read_state(&bytes).unwrap()["weight"].to_vec(), [1.0, -2.0]);
    // @end
    let want = "\
0000  45 4d 42 52 01 00 00 00 02 00 00 00 04 00 00 00  EMBR............
0010  62 69 61 73 01 00 00 00 01 00 00 00 00 00 00 3f  bias...........?
0020  06 00 00 00 77 65 69 67 68 74 02 00 00 00 02 00  ....weight......
0030  00 00 01 00 00 00 00 00 80 3f 00 00 00 c0        .........?....
";
    assert_eq!(hexdump(&bytes), want);
}

#[test]
fn a_file_round_trip_is_bit_exact() {
    // @listing round-trip-test
    let dir = Scratch::new("round-trip");
    let mut state = state_dict(&mut mnist_train::mlp(&mut Rng::new(20)));
    let odd = [-0.0, f32::INFINITY, f32::NAN, 1e-45, f32::MAX, 0.1];
    state.insert("odd".into(), t(&odd, &[2, 3]));
    let path = dir.0.join("model.embr");
    train::save(&path, &state).unwrap();
    let back = train::load(&path).unwrap();
    assert_eq!(
        back.keys().collect::<Vec<_>>(),
        state.keys().collect::<Vec<_>>()
    );
    for (name, x) in &state {
        let same = |t: &Tensor| t.to_vec().into_iter().map(f32::to_bits);
        assert!(same(x).eq(same(&back[name])) && x.shape() == back[name].shape());
    }
    // @end
    assert_eq!(std::fs::read(&path).unwrap(), to_bytes(&state)); // same bytes
    assert!(!path.with_extension("partial").exists());
}

#[test]
fn bad_files_say_what_is_wrong() {
    // @listing file-errors-test
    let mut layer = Linear::from_tensors(t(&[1., -2.], &[2, 1]), t(&[0.5], &[1]));
    let bytes = to_bytes(&state_dict(&mut layer));
    let err = |b: &[u8]| read_state(b).unwrap_err().to_string();
    let idx = [0x00, 0x00, 0x08, 0x03, 0x00, 0x00, 0xea, 0x60]; // an MNIST file
    let magic = "not an ember checkpoint (starts with [00, 00, 08, 03])";
    assert_eq!(err(&idx), magic);
    let mut v2 = bytes.clone();
    v2[4] = 2;
    let version = "checkpoint version 2; this ember reads version 1";
    assert_eq!(err(&v2), version);
    let short = "truncated: 8 bytes needed at byte 54, but the file has 60";
    assert_eq!(err(&bytes[..60]), short);
    // @end
    let missing = train::load("no/such/dir/x.embr").unwrap_err().to_string();
    assert!(missing.starts_with("no/such/dir/x.embr: "));
    assert!(matches!(
        train::load("no/such/dir/x.embr"),
        Err(CheckpointError::Io { .. })
    ));
}

#[test]
fn an_rng_in_four_pieces() {
    // @listing u64-test
    let x = u64_to_tensor(0x0123_4567_89ab_cdef);
    assert_eq!(x.to_vec(), [52719.0, 35243.0, 17767.0, 291.0]); // cdef, 89ab, …
    assert_eq!(tensor_to_u64(&x), 0x0123_4567_89ab_cdef);
    let mut rng = Rng::new(42);
    rng.next_u64();
    let mut copy = Rng::new(tensor_to_u64(&u64_to_tensor(rng.state())));
    assert_eq!(copy.next_u64(), 0x28ef_e333_b266_f103); // seed 42's second number
    // @end
    assert_eq!(rng.next_u64(), 0x28ef_e333_b266_f103);
}

#[test]
fn adam_state_by_position() {
    // @listing optim-state-test
    let mut model = spirals::mlp(4, &mut Rng::new(20));
    let mut opt = Adam::new(0.01);
    let (x, labels) = toy::spirals(5, 1.5, 0.05, &mut Rng::new(1));
    train_step(&mut model, &(x, labels), &mut opt, 1.0);
    let state = opt.state_dict(&model.parameters());
    assert_eq!(state.len(), 6 * 3); // six parameters, three entries each
    let first: Vec<&String> = state.keys().take(3).collect();
    assert_eq!(first, ["0.exp_avg", "0.exp_avg_sq", "0.step"]);
    assert_eq!(state["5.step"].item(), 1.0);
    let mut sgd = Sgd::new(0.1); // has no exp_avg to put them in
    let e = sgd.load_state_dict(&model.parameters(), &state);
    let e = e.unwrap_err().to_string();
    assert!(e.starts_with(r#"missing keys [], unexpected keys ["0.exp_avg", "#));
    assert_eq!(e.matches(".exp_avg").count(), 12); // the moments, not the steps
    // @end
    let mut again = Adam::new(0.01);
    again.load_state_dict(&model.parameters(), &state).unwrap();
    assert_eq!(again.state_dict(&model.parameters()).len(), 18);
}

/// Section 20.8's run, on the spirals: 100 training points, 100 for validation.
fn spiral_run() -> (TensorDataset, TensorDataset) {
    let mut rng = Rng::new(20);
    let (x, labels) = toy::spirals(50, 1.0, 0.05, &mut rng);
    let (vx, vlabels) = toy::spirals(50, 1.0, 0.05, &mut rng);
    (
        TensorDataset::new(x, labels),
        TensorDataset::new(vx, vlabels),
    )
}

/// A model, an optimizer and a starting point, all from one seed.
fn fresh(seed: u64) -> (Sequential, AdamW, Progress) {
    let mut rng = Rng::new(seed);
    let model = spirals::mlp(16, &mut rng);
    let at = Progress {
        step: 0,
        epoch: 0,
        best: f32::INFINITY,
        rng,
    };
    (model, AdamW::new(0.1), at)
}

#[test]
fn a_resumed_run_is_the_same_run() {
    let scratch = Scratch::new("resume");
    let (train, val) = spiral_run();
    let plan = |stop| Plan {
        epochs: 20,
        stop,
        batch: 10,
        warmup: 5,
        dir: scratch.0.clone(),
    };
    // @listing resume-test
    let (mut model, mut opt, mut at) = fresh(20); // twenty epochs in one go
    let whole = fit((&mut model, &mut opt), (&train, &val), &mut at, &plan(20));
    let (mut m, mut o, mut a) = fresh(20); // ten, then stop
    fit((&mut m, &mut o), (&train, &val), &mut a, &plan(10)).unwrap();
    drop((m, o, a));
    let (mut m, mut o, _) = fresh(99); // a new process, a new seed
    let last = train::load(scratch.0.join("last.embr")).unwrap();
    let mut a = restore(&mut m, &mut o, &last).unwrap();
    let rest = fit((&mut m, &mut o), (&train, &val), &mut a, &plan(20)).unwrap();
    assert_eq!(whole.unwrap()[10..], rest); // epochs 11 to 20, the same rows
    assert_eq!(bits(&mut m), bits(&mut model)); // and the same weights
    // @end
    println!("{rest:?}");
}

#[test]
fn a_resumed_run_needs_everything() {
    let scratch = Scratch::new("resume-parts");
    let (train, val) = spiral_run();
    let plan = |stop| Plan {
        epochs: 20,
        stop,
        batch: 10,
        warmup: 5,
        dir: scratch.0.clone(),
    };
    let (mut model, mut opt, mut at) = fresh(20);
    let whole =
        fit((&mut model, &mut opt), (&train, &val), &mut at, &plan(20)).unwrap();
    let (mut m, mut o, mut a) = fresh(20);
    fit((&mut m, &mut o), (&train, &val), &mut a, &plan(10)).unwrap();
    let last = train::load(scratch.0.join("last.embr")).unwrap();
    // Everything but the optimizer's state.
    let (mut m, _, _) = fresh(99);
    let mut a = restore(&mut m, &mut AdamW::new(0.1), &last).unwrap();
    let mut o = AdamW::new(0.1);
    let no_moments =
        fit((&mut m, &mut o), (&train, &val), &mut a, &plan(20)).unwrap();
    // Everything but the loader's generator.
    let (mut m, mut o, _) = fresh(99);
    let mut a = restore(&mut m, &mut o, &last).unwrap();
    a.rng = Rng::new(20);
    let new_order = fit((&mut m, &mut o), (&train, &val), &mut a, &plan(20)).unwrap();
    let val =
        |rows: &[mnist_train::Row]| (rows.last().unwrap().val_loss * 1e4).round();
    assert_eq!(val(&whole), 2529.0); // the run, resumed or not: 0.2529
    assert_eq!(val(&no_moments), 2375.0); // fresh moments: 0.2375
    assert_eq!(val(&new_order), 2608.0); // other batches: 0.2608
}
