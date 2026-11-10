# ember

ember is a small deep learning framework written in plain Rust, organized the way PyTorch is:
strided tensors that share memory, broadcasting, a tape-based autograd engine, modules,
optimizers and a training loop. It has no dependencies, runs on the CPU, and computes in `f32`.

It is the code for the book *Build a Deep Learning Framework in Rust* by Spencer Uresk. The
book builds ember one chapter at a time, and every listing in it is printed from this
repository, so the code here is the code on the page.

ember is for learning. It trains a multilayer perceptron and a small CNN on MNIST and a tiny
transformer on a toy task, and it is tested thoroughly, but it is not meant to compete with
PyTorch, Candle or Burn.

## Running it

You need Rust 1.99 or later. The repository pins the toolchain in `rust-toolchain.toml`, so
`rustup` installs the right one the first time you run `cargo`.

```text
$ cargo test                       # every example and challenge in the book, about ten seconds
$ cargo run --example xor          # the first network that trains (Chapter 14)
$ cargo run --release --example spirals
$ cargo run --release --example tiny_transformer
```

The MNIST examples need the dataset, about 55 MB unpacked. A script downloads it into `data/`,
checks it against the checksums torchvision uses, and uncompresses it:

```text
$ scripts/get-mnist.sh
$ cargo run --release --example mnist_mlp     # Chapter 19: an MLP, a second or so per epoch
$ cargo run --release --example mnist_train   # Chapter 20: the full loop, with checkpoints
$ cargo run --release --example mnist_cnn     # Chapter 22
```

Use `--release` for anything that trains. A debug build is many times slower.

## Following the book

The repository is a Cargo workspace with two crates. `ember/` is the library. `xtask/` holds
the benchmarks and a command that runs one chapter's tests:

```text
$ cargo xtask check 7              # Chapter 7's tests: what passes, what fails, what isn't written
$ cargo xtask bench matmul 512     # Chapter 5's timings (always in a release build)
```

Each chapter has two test files in `ember/tests/`:

- `book_chNN.rs` holds the examples the chapter prints. Every number the text quotes is an
  assertion there.
- `challenges_chNN.rs` checks the end-of-chapter challenges. The solutions are in
  `ember/src/challenges/chNN.rs`, so don't open that directory until you have tried them.

This is where each chapter's code lives:

| Chapter | | Main files |
|---|---|---|
| 1 | A tensor is a buffer and a shape | `src/tensor.rs` |
| 2 | Strides and views | `src/view.rs` |
| 3 | Broadcasting | `src/broadcast.rs` |
| 4 | Elementwise operations and reductions | `src/ops.rs` |
| 5 | Matrix multiplication | `src/matmul.rs`, `xtask/src/bench.rs` |
| 6 | The computational graph by hand | `src/autograd/scalar.rs` |
| 7 | A tape and a node arena | `src/autograd/tape.rs` |
| 8 | Backward for elementwise operations | `src/autograd/ops.rs` |
| 9 | Backward through broadcasting, reductions and views | `src/autograd/shape_ops.rs` |
| 10 | Matmul backward and gradient checking | `src/autograd/matmul.rs`, `src/autograd/gradcheck.rs` |
| 11 | Parameters and the Module trait | `src/nn/parameter.rs`, `src/nn/module.rs` |
| 12 | Linear layers, activations and initialization | `src/rng.rs`, `src/nn/init.rs`, `src/nn/linear.rs`, `src/nn/activation.rs`, `src/nn/sequential.rs` |
| 13 | Softmax, log-sum-exp and cross-entropy | `src/autograd/softmax.rs`, `src/nn/loss.rs` |
| 14 | Training an MLP end to end | `src/toy.rs`, `src/plot.rs`, `examples/xor.rs`, `examples/spirals.rs` |
| 15 | An Optimizer trait: parameter groups, SGD and weight decay | `src/optim/mod.rs`, `src/optim/sgd.rs` |
| 16 | Momentum and Nesterov | `src/optim/sgd.rs` |
| 17 | RMSProp, Adam and AdamW | `src/optim/rmsprop.rs`, `src/optim/adam.rs`, `src/optim/adamw.rs` |
| 18 | Learning-rate schedules and gradient clipping | `src/optim/schedule.rs`, `src/optim/clip.rs` |
| 19 | Datasets, batching and a seeded shuffle | `src/data/`, `examples/mnist_mlp.rs` |
| 20 | The training loop: evaluation, metrics and checkpoints | `src/train/`, `examples/mnist_train.rs` |
| 21 | Dropout and train/eval behavior | `src/nn/dropout.rs` |
| 22 | Convolution by im2col | `src/nn/conv.rs`, `src/nn/pool.rs`, `examples/mnist_cnn.rs` |
| 23 | Normalization layers | `src/autograd/norm.rs`, `src/nn/norm.rs`, `examples/mnist_norm.rs` |
| 24 | Speed: profiling, threads and copies | `src/profile.rs`, `src/matmul.rs`, `examples/speed.rs` |
| 25 | Attention and a tiny transformer | `src/autograd/select.rs`, `src/nn/embedding.rs`, `src/nn/attention.rs`, `src/nn/transformer.rs`, `examples/tiny_transformer.rs` |

Paths in the table are inside `ember/`, except `xtask/`.

Two things to know when you compare your code with this one:

- **This is the finished framework.** A file shows its final state, so a function from an early
  chapter may include lines that a later chapter added. The book says so where it happens
  (Chapter 24, for instance, revisits code from Chapters 2, 7, 10 and 11).
- **The `// @listing` and `// @solution` comments** mark the regions the book prints. They are
  how the book's build finds its listings, and they have no effect on the code.

## A taste

```rust
use ember::{Tape, Tensor};

let tape = Tape::new();
let x = tape.leaf(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]));
let w = tape.leaf(Tensor::from_vec(vec![0.5, -1.0, 2.0], &[3]));
let loss = x.mul(w).sum_all();          // 0.5 - 2.0 + 6.0 = 4.5
let grads = loss.backward();
assert_eq!(grads.get(w).unwrap().to_vec(), vec![1.0, 2.0, 3.0]); // d loss / d w = x
```

## Found a mistake?

Please open an issue, whether it is a bug in the code or a place where the book and the
repository disagree. Pull requests that fix bugs are welcome. Because the book prints this
code, changes that reshape a listing have to wait for the next printing.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option. The license covers the code in this repository.
The text of the book is not part of it.
