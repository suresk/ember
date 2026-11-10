//! The examples printed in Chapter 19 of the guided edition: datasets, batches, a
//! seeded shuffle, and MNIST's IDX files (built here in memory: no test needs the
//! real files).
use ember::data::{self, DataError};
use ember::{
    Adam, Batch, DataLoader, Dataset, Module, Rng, Sequential, Tape, Tensor,
    TensorDataset, cross_entropy, toy,
};

#[path = "../examples/mnist_mlp.rs"]
#[allow(dead_code)]
mod mnist_mlp;
#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;

/// A dataset with only `get`, so its batches come from the default `get_batch`.
struct PerExample<'a>(&'a TensorDataset);

impl Dataset for PerExample<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(&self, i: usize) -> (Tensor, usize) {
        self.0.get(i)
    }
}

#[test]
fn a_tensor_dataset() {
    // @listing dataset-test
    let ds = TensorDataset::new(Tensor::arange(&[5, 3]), vec![7, 8, 9, 7, 8]);
    let (x, label) = ds.get(1);
    assert_eq!((x.to_vec(), label), (vec![3.0, 4.0, 5.0], 8)); // shape [3]
    let (x, labels) = ds.get_batch(&[4, 0]);
    assert_eq!((x.shape(), &labels[..]), (&[2, 3][..], &[8, 7][..]));
    assert_eq!(x.to_vec(), [12.0, 13.0, 14.0, 0.0, 1.0, 2.0]);
    let (y, same) = PerExample(&ds).get_batch(&[4, 0]); // get, then stack
    assert_eq!((y.to_vec(), same), (x.to_vec(), labels)); // the same batch
    // @end
    assert_eq!(y.shape(), x.shape());
    let (x, _) = ds.get(1);
    assert!(x.shape() == [3] && x.shares_storage(ds.tensors().0)); // a view
}

#[test]
#[should_panic(expected = "TensorDataset: 5 rows but 4 labels")]
fn a_label_per_row() {
    TensorDataset::new(Tensor::arange(&[5, 3]), vec![0; 4]);
}

#[test]
fn batches_per_epoch() {
    // @listing epochs-test
    let ds = TensorDataset::new(Tensor::arange(&[10, 1]), vec![0; 10]);
    let mut keep = DataLoader::new(&ds, 4);
    let sizes: Vec<usize> = keep.epoch().map(|(x, _)| x.shape()[0]).collect();
    assert_eq!((keep.len(), sizes), (3, vec![4, 4, 2]));
    let mut cut = DataLoader::new(&ds, 4).drop_last(true);
    let sizes: Vec<usize> = cut.epoch().map(|(x, _)| x.shape()[0]).collect();
    assert_eq!((cut.len(), sizes), (2, vec![4, 4]));
    // MNIST's 60,000 training digits in batches of 64:
    assert_eq!((60_000 / 64, 60_000 % 64), (937, 32));
    assert_eq!((60_000usize.div_ceil(64), 5 * 937), (938, 4685));
    // @end
    let first = keep.epoch().next().unwrap().0; // no shuffle: in order
    assert_eq!(first.to_vec(), [0.0, 1.0, 2.0, 3.0]);
    let mut epoch = keep.epoch();
    assert_eq!(epoch.len(), 3);
    epoch.next();
    assert_eq!(epoch.size_hint(), (2, Some(2)));
    assert_eq!(cut.epoch().len(), 2);
    // A full-batch forward pass through 784→128: 6.0 billion multiply-adds.
    assert_eq!(60_000u64 * 784 * 128, 6_021_120_000);
    assert_eq!(64 * 784 * 128, 6_422_528);
}

/// The order a loader visits the examples in, epoch by epoch.
fn orders(loader: &mut DataLoader<TensorDataset>, epochs: usize) -> Vec<Vec<usize>> {
    let order = |batches: ember::Batches<'_, TensorDataset>| {
        let rows = batches.flat_map(|(x, _)| x.to_vec());
        rows.map(|v| v as usize).collect::<Vec<usize>>()
    };
    (0..epochs).map(|_| order(loader.epoch())).collect()
}

#[test]
fn a_seeded_shuffle() {
    // @listing shuffle-test
    let ds = TensorDataset::new(Tensor::arange(&[10, 1]), vec![0; 10]);
    let mut loader = DataLoader::new(&ds, 4).shuffle(Rng::new(19));
    let epochs = orders(&mut loader, 3); // each input is its own index
    assert_eq!(epochs[0], [9, 6, 3, 8, 5, 4, 0, 2, 1, 7]);
    assert_eq!(epochs[1], [0, 4, 1, 2, 3, 6, 9, 7, 8, 5]);
    for order in &epochs {
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>()); // a permutation
    }
    assert_ne!(epochs[1], epochs[2]); // a new order every epoch
    let mut again = DataLoader::new(&ds, 4).shuffle(Rng::new(19));
    assert_eq!(orders(&mut again, 3), epochs); // the same seed, the same orders
    // @end
}

#[test]
fn shuffled_batches_carry_their_labels() {
    let x = Tensor::arange(&[100, 2]);
    let labels: Vec<usize> = (0..100).map(|i| i % 7).collect();
    let ds = TensorDataset::new(x, labels);
    let mut loader = DataLoader::new(&ds, 16).shuffle(Rng::new(3));
    let mut seen = 0;
    for (x, labels) in loader.epoch() {
        for (row, &label) in labels.iter().enumerate() {
            let i = x.get(&[row, 0]) as usize / 2; // row i is [2i, 2i + 1]
            assert_eq!(label, i % 7);
            seen += 1;
        }
    }
    assert_eq!(seen, 100);
}

/// An IDX file of unsigned bytes: magic `0x0000_08nn`, the sizes, the data.
fn idx(dims: &[u32], data: &[u8]) -> Vec<u8> {
    let mut bytes = (0x0800 | dims.len() as u32).to_be_bytes().to_vec();
    for d in dims {
        bytes.extend(d.to_be_bytes());
    }
    bytes.extend(data);
    bytes
}

#[test]
fn reading_idx_bytes() {
    // @listing idx-test
    let images = idx(&[2, 2, 3], &[0, 51, 102, 153, 204, 255, 0, 0, 0, 0, 0, 255]);
    assert_eq!(images[..4], [0x00, 0x00, 0x08, 0x03]); // then 2, 2 and 3
    assert_eq!(images.len(), 16 + 12); // a header of four u32s, then 2·2·3 bytes
    let x = data::mnist_images(&images).unwrap(); // 2 images of 2 × 3, flattened
    assert_eq!(x.shape(), [2, 6]);
    assert_eq!(x.to_vec()[..6], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    let z = data::normalize(&x, data::MNIST_MEAN, data::MNIST_STD);
    assert!((z.get(&[0, 0]) + 0.4242).abs() < 1e-4); // black: (0 − 0.1307)/0.3081
    assert!((z.get(&[0, 5]) - 2.8215).abs() < 1e-4); // white
    // @end
    let (dims, pixels) = data::parse_idx(&images, 3).unwrap();
    assert_eq!((dims, pixels.len()), (vec![2, 2, 3], 12));
    let labels = data::mnist_labels(&idx(&[3], &[7, 0, 9])).unwrap();
    assert_eq!(labels, [7, 0, 9]);
}

#[test]
fn bad_files_say_what_is_wrong() {
    // @listing errors-test
    let images = |b: &[u8]| data::mnist_images(b).unwrap_err().to_string();
    let labels = |b: &[u8]| data::mnist_labels(b).unwrap_err().to_string();
    let file = idx(&[3], &[7, 0, 9]); // a good labels file
    let wrong = "magic number 0x00000801, expected 0x00000803"; // as images
    assert_eq!(images(&file), wrong);
    assert_eq!(labels(&file[..10]), "10 bytes, but the header says 11");
    assert_eq!(labels(&file[..6]), "6 bytes, but the header says 8");
    assert_eq!(labels(&idx(&[3], &[7, 10, 9])), "label 1 is 10");
    // @end
    let err = data::mnist_labels(&idx(&[3], &[7, 10, 9])).unwrap_err();
    assert!(matches!(
        err,
        DataError::BadLabel {
            index: 1,
            label: 10
        }
    ));
    assert!(std::error::Error::source(&err).is_none());
}

/// A folder of its own in the system's temporary directory.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("ember-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn mnist_from_a_folder() {
    // @listing folder-test
    let dir = scratch("mnist");
    let write = |name: &str, bytes| std::fs::write(dir.join(name), bytes).unwrap();
    write(
        "train-images-idx3-ubyte",
        idx(&[2, 1, 2], &[0, 255, 255, 0]),
    );
    write("train-labels-idx1-ubyte", idx(&[2], &[3, 4]));
    let train = data::mnist(&dir, true).unwrap();
    let (x, labels) = train.tensors();
    assert_eq!((x.shape(), labels), (&[2, 2][..], &[3, 4][..]));
    assert!((x.get(&[0, 1]) - 2.8215).abs() < 1e-4); // white, normalized
    let err = data::mnist(&dir, false).unwrap_err(); // no t10k files here
    let missing = dir.join("t10k-images-idx3-ubyte");
    assert!(
        err.to_string()
            .starts_with(&format!("{}: ", missing.display()))
    );
    let DataError::Io { source, .. } = &err else {
        panic!("{err}")
    };
    assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
    write("train-labels-idx1-ubyte", idx(&[3], &[3, 4, 5]));
    let err = data::mnist(&dir, true).unwrap_err();
    assert_eq!(err.to_string(), "2 images, 3 labels");
    // @end
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The gradient of the mean loss over one batch, every parameter's in one vector.
fn gradient(model: &mut Sequential, (x, labels): &Batch) -> Vec<f32> {
    model.zero_grad();
    let tape = Tape::new();
    let loss = cross_entropy(model.forward(tape.constant(x.clone())), labels);
    loss.backward_into(&mut model.parameters());
    let params = model.parameters();
    let grads = params.iter().map(|p| p.grad.as_ref().unwrap());
    grads.flat_map(|g| g.to_vec()).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

/// For 50 batches of `size`: the mean cosine between each batch's gradient
/// and `full`, and the RMS of their difference relative to `|full|`.
fn noise(
    model: &mut Sequential,
    ds: &TensorDataset,
    size: usize,
    full: &[f32],
) -> [f64; 2] {
    let rng = Rng::new(19);
    let mut loader = DataLoader::new(ds, size).shuffle(rng).drop_last(true);
    let batches = std::iter::repeat_with(|| loader.epoch()).flatten().take(50);
    let (mut cosine, mut error, n) = (0.0, 0.0, dot(full, full));
    for batch in batches {
        let g = gradient(model, &batch);
        let diff: Vec<f32> = g.iter().zip(full).map(|(a, b)| a - b).collect();
        cosine += dot(&g, full) / (dot(&g, &g) * n).sqrt() / 50.0;
        error += dot(&diff, &diff) / n / 50.0;
    }
    [cosine, error.sqrt()]
}

fn round(x: f64, places: i32) -> f64 {
    let scale = 10f64.powi(places);
    (x * scale).round() / scale
}

#[test]
fn minibatch_gradients_are_noisy() {
    // @listing noise-test
    let mut rng = Rng::new(19);
    let (x, labels) = toy::spirals(1000, 1.5, 0.05, &mut rng); // 2,000 points
    let mut model = spirals::mlp(16, &mut rng);
    let ds = TensorDataset::new(x.clone(), labels.clone());
    let full = gradient(&mut model, &(x, labels));
    let table = [1, 8, 64, 512].map(|b| noise(&mut model, &ds, b, &full));
    let [cosines, errors] = [0, 1].map(|k| table.map(|row| round(row[k], 3)));
    assert_eq!(cosines, [-0.216, 0.264, 0.807, 0.998]);
    assert_eq!(errors, [4.645, 1.795, 0.634, 0.173]);
    // @end
    let fpc = |b: f64| ((2000.0 - b) / 1999.0).sqrt(); // without replacement
    let predicted =
        [8.0, 64.0, 512.0f64].map(|b| round(4.645 / b.sqrt() * fpc(b), 3));
    assert_eq!(predicted, [1.639, 0.571, 0.177]);
}

#[test]
fn the_mnist_example_step() {
    // The example's model, step and accuracy on 80 made-up "digits": noise,
    // with a bright block of 78 pixels whose place is the label.
    let mut rng = Rng::new(19);
    let mut x = Tensor::rand(&[80, 784], &mut rng);
    let labels: Vec<usize> = (0..80).map(|i| i % 10).collect();
    for (i, &label) in labels.iter().enumerate() {
        for p in 78 * label..78 * (label + 1) {
            x.data_mut()[i * 784 + p] += 1.0;
        }
    }
    let ds = TensorDataset::new(x, labels);
    let mut model = mnist_mlp::mlp(&mut rng);
    let mut opt = Adam::new(1e-3);
    let mut loader = DataLoader::new(&ds, 8).shuffle(rng);
    let losses: Vec<f32> = loader
        .epoch()
        .map(|batch| mnist_mlp::step(&mut model, &batch, &mut opt))
        .collect();
    assert_eq!(losses.len(), 10);
    assert!((losses[0] - 2.3198).abs() < 1e-4); // ln 10 = 2.3026
    assert!((losses[9] - 1.8591).abs() < 1e-4);
    assert_eq!(mnist_mlp::accuracy(&model, &ds), 0.95);
}
