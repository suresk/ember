//! Chapter 20: MNIST with a validation set, a table per epoch, checkpoints, the
//! best model kept, and a run that can be stopped and resumed.
//!
//! Download the four files and uncompress them into `data/` at the workspace root
//! (Section 19.6), or pass another folder after `--`:
//!
//! cargo run --release --example mnist_train             train for 10 epochs
//! cargo run --release --example mnist_train -- stop 4   stop after epoch 4
//! cargo run --release --example mnist_train -- resume   continue from last.embr
//! cargo run --release --example mnist_train -- grad     evaluation, timed
//! cargo run --release --example mnist_train -- dropout  with and without dropout
//! cargo run --release --example mnist_train -- dropout 30  the same, 30 epochs

use ember::data::{self, DataError};
use ember::train::{self, strip_prefix, tensor_to_u64, u64_to_tensor, with_prefix};
use ember::{
    AdamW, CheckpointError, CosineLr, DataLoader, Dataset, Dropout, Linear,
    LinearWarmup, LrScheduler, Module, ReLU, Rng, Sequential, StateDict, Stateful,
    Tape, Tensor, base_lrs, cross_entropy, evaluate, load_state_dict, random_split,
    state_dict, train_epoch, train_step,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// 784 pixels, 128 ReLU units, 10 logits: Chapter 19's network.
pub fn mlp(rng: &mut Rng) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(784, 128, rng)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, rng)),
    ])
}

// @listing dropout-mlp
/// Linear layers between the `sizes`, [784, 128, 10] for Chapter 19's network;
/// each hidden layer is followed by a ReLU and dropout with probability `p`.
pub fn mlp_dropout(sizes: &[usize], p: f32, rng: &mut Rng) -> Sequential {
    let mut layers: Vec<Box<dyn Module>> = vec![];
    for (i, pair) in sizes.windows(2).enumerate() {
        layers.push(Box::new(Linear::new(pair[0], pair[1], rng)));
        if i + 2 < sizes.len() {
            layers.push(Box::new(ReLU)); // a hidden layer
            if p > 0.0 {
                layers.push(Box::new(Dropout::new(p, 21 + i as u64))); // own seeds
            }
        }
    }
    Sequential::new(layers)
}
// @end

/// Where a run is, beside its model and optimizer: everything else a resumed
/// run needs to continue exactly as if it had never stopped.
// @listing progress
#[derive(Clone, Debug)]
pub struct Progress {
    pub step: usize,  // updates so far, for the schedule
    pub epoch: usize, // epochs finished
    pub best: f32,    // the lowest validation loss so far
    pub rng: Rng,     // the loader's generator, for the next epochs' orders
}
// @end

// @listing checkpoint
/// The whole run as one state dict.
pub fn checkpoint(
    model: &mut dyn Module,
    opt: &dyn Stateful,
    at: &Progress,
) -> StateDict {
    let mut state = with_prefix("model", state_dict(model));
    state.extend(with_prefix("optim", opt.state_dict(&model.parameters())));
    let scalar = |x: f32| Tensor::from_vec(vec![x], &[]);
    state.insert("step".into(), scalar(at.step as f32));
    state.insert("epoch".into(), scalar(at.epoch as f32));
    state.insert("best".into(), scalar(at.best));
    state.insert("rng".into(), u64_to_tensor(at.rng.state()));
    state
}
// @end

// @listing restore
/// Load a checkpoint made by `checkpoint` into a model and an optimizer built
/// the same way, and return the progress it saved.
pub fn restore(
    model: &mut dyn Module,
    opt: &mut dyn Stateful,
    state: &StateDict,
) -> Result<Progress, CheckpointError> {
    load_state_dict(model, &strip_prefix("model", state))?;
    opt.load_state_dict(&model.parameters(), &strip_prefix("optim", state))?;
    let get = |name: &str| {
        let missing = vec![name.to_string()];
        let keys = CheckpointError::Keys {
            missing,
            unexpected: vec![],
        };
        state.get(name).ok_or(keys)
    };
    Ok(Progress {
        step: get("step")?.item() as usize,
        epoch: get("epoch")?.item() as usize,
        best: get("best")?.item(),
        rng: Rng::new(tensor_to_u64(get("rng")?)),
    })
}
// @end

/// How long to train, in batches of what size, and where checkpoints go.
pub struct Plan {
    pub epochs: usize, // the schedule's length
    pub stop: usize,   // the last epoch to run now (`epochs` for all)
    pub batch: usize,  // training batch size
    pub warmup: usize, // steps of linear warmup before the cosine
    pub dir: PathBuf,  // where best.embr and last.embr go
}

/// One epoch's line of the table.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub epoch: usize,
    pub loss: f32,
    pub val_loss: f32,
    pub val_accuracy: f32,
    pub best: bool,
}

// @listing fit
/// Train from `at` to epoch `plan.stop`, evaluating on `val` after every epoch,
/// keeping the best model in best.embr and the whole run in last.embr.
pub fn fit<D: Dataset + ?Sized>(
    (model, opt): (&mut Sequential, &mut AdamW),
    (train, val): (&D, &D),
    at: &mut Progress,
    plan: &Plan,
) -> Result<Vec<Row>, CheckpointError> {
    let loader = DataLoader::new(train, plan.batch).drop_last(true);
    let mut loader = loader.shuffle(at.rng.clone()); // where the run left off
    let cosine = CosineLr::new(plan.epochs * loader.len() - plan.warmup, 0.0);
    let schedule = LinearWarmup::new(plan.warmup, cosine);
    let (base, mut rows) = (base_lrs(opt), vec![]);
    for epoch in at.epoch + 1..=plan.stop {
        let loss = train_epoch(model, &mut loader, |model, batch| {
            schedule.apply(opt, &base, at.step);
            at.step += 1;
            train_step(model, batch, opt, 1.0)
        });
        let (val_loss, val_accuracy) =
            evaluate(model, &mut DataLoader::new(val, 1000));
        let best = val_loss < at.best;
        if best {
            at.best = val_loss;
            train::save(plan.dir.join("best.embr"), &state_dict(model))?;
        }
        (at.epoch, at.rng) = (epoch, loader.rng().unwrap().clone());
        train::save(plan.dir.join("last.embr"), &checkpoint(model, opt, at))?;
        rows.push(Row::new(epoch, loss, (val_loss, val_accuracy), best));
        log(rows.last().unwrap());
    }
    Ok(rows)
}
// @end

impl Row {
    fn new(
        epoch: usize,
        loss: f32,
        (val_loss, val_accuracy): (f32, f32),
        best: bool,
    ) -> Row {
        Row {
            epoch,
            loss,
            val_loss,
            val_accuracy,
            best,
        }
    }
}

impl Progress {
    /// The start of a run: nothing done yet, the loader's generator `rng`.
    pub fn new(rng: Rng) -> Progress {
        Progress {
            step: 0,
            epoch: 0,
            best: f32::INFINITY,
            rng,
        }
    }
}

/// Print a line of the table.
fn log(r: &Row) {
    let mark = if r.best { "  saved best" } else { "" };
    let (e, l, v, a) = (r.epoch, r.loss, r.val_loss, r.val_accuracy);
    println!("{e:>5} {l:>8.4} {v:>9.4} {a:>8.4}{mark}");
}

/// Any error this example reports: a data file's or a checkpoint's.
#[derive(Debug)]
enum Error {
    Data(DataError),
    Checkpoint(CheckpointError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Data(e) => write!(f, "{e}"),
            Error::Checkpoint(e) => write!(f, "{e}"),
        }
    }
}

impl From<DataError> for Error {
    fn from(e: DataError) -> Error {
        Error::Data(e)
    }
}

impl From<CheckpointError> for Error {
    fn from(e: CheckpointError) -> Error {
        Error::Checkpoint(e)
    }
}

/// Ten epochs on 55,000 training images, validated on the other 5,000; then the
/// best model on the test set. With `stop`, end early; with `resume`, continue.
fn run(dir: &Path, stop: usize, resume: bool) -> Result<(), Error> {
    let (full, test) = (data::mnist(dir, true)?, data::mnist(dir, false)?);
    let dir = dir.join("checkpoints");
    std::fs::create_dir_all(&dir).map_err(|source| {
        let path = dir.clone();
        CheckpointError::Io { path, source }
    })?;
    let (epochs, batch, warmup) = (10, 64, 100);
    let plan = Plan {
        epochs,
        stop,
        batch,
        warmup,
        dir,
    };
    // @listing main
    let mut rng = Rng::new(20);
    let parts = random_split(&full, &[55_000, 5_000], &mut rng);
    let (training, validation) = (&parts[0], &parts[1]);
    let mut model = mlp(&mut rng);
    let mut opt = AdamW::new(1e-3);
    let mut at = Progress::new(rng); // step 0, epoch 0, best ∞
    if resume {
        let last = train::load(plan.dir.join("last.embr"))?;
        at = restore(&mut model, &mut opt, &last)?; // replaces all of the above
    }
    println!("epoch     loss  val loss  val acc");
    let start = Instant::now();
    let data = (training, validation);
    fit((&mut model, &mut opt), data, &mut at, &plan)?;
    let secs = start.elapsed().as_secs_f64();
    let best = train::load(plan.dir.join("best.embr"))?;
    load_state_dict(&mut model, &best)?;
    let (loss, accuracy) = evaluate(&mut model, &mut DataLoader::new(&test, 1000));
    println!("best model on the test set: loss {loss:.4}, accuracy {accuracy:.4}");
    // @end
    println!("trained in {secs:.1}s");
    Ok(())
}

/// The test set's loss and accuracy, as `evaluate` computes them but with the
/// backward rules recorded: the same numbers, for timing.
fn evaluate_recording(model: &dyn Module, data: &dyn Dataset) -> (f32, f32) {
    let (mut loss, mut right) = (train::Average::new(), train::Average::new());
    for (x, labels) in DataLoader::new(data, 1000).epoch() {
        let tape = Tape::new();
        let logits = model.forward(tape.constant(x));
        let l = cross_entropy(logits, &labels).value().item();
        loss.add(l, labels.len());
        right.add(train::accuracy(&logits.value(), &labels), labels.len());
    }
    (loss.mean(), right.mean())
}

/// Evaluation on the test set with and without `no_grad`, timed.
fn grad(dir: &Path) -> Result<(), Error> {
    let test = data::mnist(dir, false)?;
    let mut model = mlp(&mut Rng::new(20));
    for _ in 0..2 {
        let t = Instant::now();
        let a = evaluate(&mut model, &mut DataLoader::new(&test, 1000));
        let fast = t.elapsed();
        let t = Instant::now();
        let b = evaluate_recording(&model, &test);
        let slow = t.elapsed();
        assert_eq!(a, b);
        println!("evaluate: no_grad {fast:.2?}, recording rules {slow:.2?}");
    }
    Ok(())
}

/// `epochs` epochs of each network, with and without dropout; then each final
/// model on its training images, in evaluation mode, and on the test set.
fn dropout(dir: &Path, epochs: usize) -> Result<(), Error> {
    let (full, test) = (data::mnist(dir, true)?, data::mnist(dir, false)?);
    let runs: [(&[usize], f32); 6] = [
        (&[784, 128, 10], 0.0),
        (&[784, 128, 10], 0.2),
        (&[784, 128, 10], 0.5),
        (&[784, 512, 512, 10], 0.0),
        (&[784, 512, 512, 10], 0.2),
        (&[784, 512, 512, 10], 0.5),
    ];
    let mut summary = vec![];
    for (sizes, p) in runs {
        let dir = dir.join("checkpoints").join("dropout");
        std::fs::create_dir_all(&dir).map_err(|source| {
            let path = dir.clone();
            CheckpointError::Io { path, source }
        })?;
        let plan = Plan {
            epochs,
            stop: epochs,
            batch: 64,
            warmup: 100,
            dir,
        };
        println!("{sizes:?}, p = {p}\nepoch     loss  val loss  val acc");
        let start = Instant::now();
        let mut rng = Rng::new(20);
        let parts = random_split(&full, &[55_000, 5_000], &mut rng);
        let mut model = mlp_dropout(sizes, p, &mut rng);
        let mut opt = AdamW::new(1e-3);
        let mut at = Progress::new(rng);
        let data = (&parts[0], &parts[1]);
        let rows = fit((&mut model, &mut opt), data, &mut at, &plan)?;
        let secs = start.elapsed().as_secs_f64();
        let last = rows.last().unwrap();
        let (tl, ta) = evaluate(&mut model, &mut DataLoader::new(&parts[0], 1000));
        let (_, test_acc) = evaluate(&mut model, &mut DataLoader::new(&test, 1000));
        let (vl, va) = (last.val_loss, last.val_accuracy);
        let (hidden, gap) = (format!("{:?}", &sizes[1..sizes.len() - 1]), vl - tl);
        let train = format!("{hidden:>12} {p:>4} {tl:>8.4} {ta:>7.4}");
        let val = format!("{vl:>8.4} {va:>7.4} {gap:>7.4}");
        summary.push(format!("{train} {val} {test_acc:>7.4} {secs:>6.1}s"));
    }
    println!(
        "\n      hidden    p    train    acc      val     acc     gap    test   time"
    );
    for line in summary {
        println!("{line}");
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let default = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../data"));
    let dir = args.iter().find(|a| a.contains('/')).map(PathBuf::from);
    let dir = dir.unwrap_or(default);
    let number = args.iter().find_map(|a| a.parse::<usize>().ok());
    let result = match args.first().map(String::as_str) {
        Some("stop") => run(&dir, number.unwrap_or(4), false),
        Some("resume") => run(&dir, 10, true),
        Some("grad") => grad(&dir),
        Some("dropout") => dropout(&dir, number.unwrap_or(10)),
        _ => run(&dir, 10, false),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
