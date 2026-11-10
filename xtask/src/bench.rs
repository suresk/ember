//! `cargo xtask bench matmul`, `cargo xtask bench layout` and friends: timings of ember's
//! kernels, always in a release build.

use ember::{
    Kernel, Tensor, matmul_blocked, matmul_blocked_with, matmul_ikj, matmul_naive,
    matmul_nt_kernel, matmul_parallel,
};
use std::hint::black_box;
use std::time::{Duration, Instant};

// @listing best-of
/// The fastest of `runs` calls to `f`, after one call to warm up. The fastest run
/// is the one least disturbed by everything else the machine was doing.
fn best_of(runs: usize, mut f: impl FnMut()) -> Duration {
    f();
    let mut best = Duration::MAX;
    for _ in 0..runs {
        let start = Instant::now();
        f();
        best = best.min(start.elapsed());
    }
    best
}
// @end

/// `len` values in [-1, 1], the same on every run; `seed` makes two inputs differ.
fn data(len: usize, seed: usize) -> Vec<f32> {
    let v = |i: usize| ((i * 7919 + seed * 104_729) % 2001) as f32;
    (0..len).map(|i| v(i) / 1000.0 - 1.0).collect()
}

// @listing bench-matmul
const KERNELS: [(&str, Kernel); 3] = [
    ("naive", matmul_naive),
    ("ikj", matmul_ikj),
    ("blocked", matmul_blocked),
];

/// Each kernel on two n × n matrices: milliseconds and GFLOP/s.
pub fn matmul(sizes: &[usize]) {
    println!("{:>5}  {:<8} {:>9} {:>8}", "n", "kernel", "ms", "GFLOP/s");
    for &n in sizes {
        let (a, b) = (data(n * n, 1), data(n * n, 2));
        let mut c = vec![0.0; n * n];
        let flops = 2.0 * (n as f64).powi(3);
        for (name, kernel) in KERNELS {
            let t = best_of(3, || {
                c.fill(0.0); // kernels add into c
                kernel(black_box(&a), black_box(&b), &mut c, (n, n, n));
            });
            let (ms, gflops) = (t.as_secs_f64() * 1e3, flops / t.as_secs_f64() / 1e9);
            println!("{n:>5}  {name:<8} {ms:>9.2} {gflops:>8.2}");
        }
    }
}
// @end

// @listing bench-layout
/// `x + y` and `x.matmul(y)` for 1024 × 1024 tensors held three ways: contiguous,
/// rows with a gap after each, and transposed.
pub fn layout() {
    let n = 1024;
    let t = |seed, cols| Tensor::from_vec(data(n * cols, seed), &[n, cols]);
    let gapped = |seed| t(seed, n + 1).slice(1, 0, n, 1);
    let transposed = |seed| t(seed, n).transpose(0, 1);
    let cases = [
        ("contiguous", t(1, n), t(2, n)),
        ("gapped rows", gapped(1), gapped(2)),
        ("transposed", transposed(1), transposed(2)),
    ];
    let ms = |t: Duration| t.as_secs_f64() * 1e3;
    println!("layout           add ms   ns/elem   matmul ms");
    for (name, x, y) in &cases {
        let add = ms(best_of(20, || drop(black_box(x.add(y)))));
        let matmul = ms(best_of(3, || drop(black_box(x.matmul(y)))));
        let ns = add * 1e6 / (n * n) as f64;
        println!("{name:<12} {add:>9.3} {ns:>9.2} {matmul:>11.1}");
    }
}
// @end

// @listing bench-panel
/// Challenge 5.6: `matmul_blocked_with` on two n × n matrices, for panels of 16
/// rows up to n rows (which is no panel at all).
pub fn panel(n: usize) {
    let (a, b) = (data(n * n, 1), data(n * n, 2));
    let mut c = vec![0.0; n * n];
    let flops = 2.0 * (n as f64).powi(3);
    println!("{:>5}  {:>6} {:>9} {:>8}", "n", "panel", "ms", "GFLOP/s");
    let panels = (4..).map(|e| 1 << e).take_while(|&p| p < n).chain([n]);
    for panel in panels {
        let t = best_of(3, || {
            c.fill(0.0);
            matmul_blocked_with(black_box(&a), &b, &mut c, (n, n, n), panel);
        });
        let (ms, gflops) = (t.as_secs_f64() * 1e3, flops / t.as_secs_f64() / 1e9);
        println!("{n:>5}  {panel:>6} {ms:>9.2} {gflops:>8.2}");
    }
}
// @end

// @listing bench-nt
/// Challenge 5.4: `x @ y.T` for n × n matrices, three ways.
pub fn nt(n: usize) {
    let (a, b) = (data(n * n, 1), data(n * n, 2));
    let x = Tensor::from_vec(a.clone(), &[n, n]);
    let yt = Tensor::from_vec(b.clone(), &[n, n]).transpose(0, 1); // a view
    let flops = 2.0 * (n as f64).powi(3);
    let show = |name: &str, t: Duration| {
        let (ms, gflops) = (t.as_secs_f64() * 1e3, flops / t.as_secs_f64() / 1e9);
        println!("{n:>5}  {name:<15} {ms:>9.2} {gflops:>8.2}");
    };
    let copy = best_of(3, || drop(black_box(x.matmul(&yt))));
    show("copy + blocked", copy);
    let strided = best_of(3, || drop(black_box(x.matmul_strided(&yt))));
    show("strided", strided);
    let mut c = vec![0.0; n * n];
    let nt = || {
        c.fill(0.0);
        matmul_nt_kernel(black_box(&a), black_box(&b), &mut c, (n, n, n));
    };
    show("nt kernel", best_of(3, nt));
}
// @end

// @listing bench-threads
/// `matmul_parallel` on the products of an MNIST step and on n × n matrices,
/// for 1, 2, 4, … threads: milliseconds, and the speedup over one thread.
pub fn threads(n: usize) {
    let shapes = [
        ("[64,784]x[784,128]", (64, 784, 128)),
        ("[784,64]x[64,128]", (784, 64, 128)),
        ("[50176,9]x[9,8]", (50_176, 9, 8)),
        ("[12544,72]x[72,16]", (12_544, 72, 16)),
        ("n x n", (n, n, n)),
    ];
    let max = std::thread::available_parallelism().map_or(1, |n| n.get());
    let few = [1, 2, 4, 6, 8, 12, 16, 24, 32]
        .into_iter()
        .filter(|&t| t < max);
    let counts: Vec<usize> = few.chain([max]).collect();
    print!("{:<20}", "product");
    counts.iter().for_each(|t| print!(" {t:>8}"));
    println!();
    for (name, (m, k, p)) in shapes {
        let (a, b) = (data(m * k, 1), data(k * p, 2));
        let mut c = vec![0.0; m * p];
        let mut one = 0.0;
        print!("{name:<20}");
        for &t in &counts {
            let time = best_of(20, || {
                c.fill(0.0);
                matmul_parallel(black_box(&a), black_box(&b), &mut c, (m, k, p), t);
            });
            let ms = time.as_secs_f64() * 1e3;
            if t == 1 {
                one = ms;
            }
            print!(" {ms:>8.3}");
        }
        println!("   (1 thread {one:.3} ms)");
    }
}

/// What starting and joining `t` scoped threads that do nothing costs.
pub fn spawn() {
    for t in [1, 2, 4, 8, 16] {
        let time = best_of(200, || {
            std::thread::scope(|s| {
                for _ in 0..t {
                    s.spawn(|| black_box(0));
                }
            })
        });
        println!("{t:>3} threads: {:>8.1} µs", time.as_secs_f64() * 1e6);
    }
}
// @end
