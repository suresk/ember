//! The examples printed in Chapter 10 of the guided edition: matmul backward and
//! gradient checking.
use ember::{Gradients, Tape, Tensor, Tolerance, Var, gradcheck};

fn t(xs: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), shape)
}

fn grad(grads: &Gradients, v: Var) -> Vec<f32> {
    grads.get(v).expect("a gradient").to_vec()
}

/// `x` is within a fraction `rel` of `want`.
fn near(x: f64, want: f64, rel: f64) -> bool {
    (x - want).abs() <= rel * want.abs()
}

/// The section's weights on the four entries of C.
fn w22() -> Tensor {
    t(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
}

#[test]
fn the_gradient_by_hand() {
    // @listing by-hand
    let (a, b, w) = (Tensor::arange(&[2, 3]), Tensor::arange(&[3, 2]), w22());
    assert_eq!(a.matmul(&b).to_vec(), [10.0, 13.0, 28.0, 40.0]);
    let da = w.matmul(&b.transpose(0, 1)); // [2, 2] @ [2, 3]
    let db = a.transpose(0, 1).matmul(&w); // [3, 2] @ [2, 2]
    assert_eq!(da.to_vec(), [2.0, 8.0, 14.0, 4.0, 18.0, 32.0]);
    assert_eq!(db.to_vec(), [9.0, 12.0, 13.0, 18.0, 17.0, 24.0]);
    // @end
    // ā01 by its two reads: w00·b10 + w01·b11.
    assert_eq!(1.0 * 2.0 + 2.0 * 3.0, da.get(&[0, 1]));
    // The loss moves by 8h when a01 moves by h (exact for h = 0.25).
    let loss = |a: &Tensor| a.matmul(&b).mul(&w).sum_all();
    let mut up = a.to_vec();
    up[1] += 0.25;
    assert_eq!((loss(&t(&up, &[2, 3])) - loss(&a)) / 0.25, 8.0);
}

#[test]
fn matmul_backward() {
    // @listing matmul-example
    let tape = Tape::new();
    let a = tape.leaf(Tensor::arange(&[2, 3]));
    let b = tape.leaf(Tensor::arange(&[3, 2]));
    let c = a.matmul(b);
    assert_eq!(c.value().to_vec(), [10.0, 13.0, 28.0, 40.0]);
    let grads = (c * tape.constant(w22())).sum_all().backward();
    assert_eq!(grad(&grads, a), [2.0, 8.0, 14.0, 4.0, 18.0, 32.0]); // W Bᵀ
    assert_eq!(grad(&grads, b), [9.0, 12.0, 13.0, 18.0, 17.0, 24.0]); // Aᵀ W
    assert_eq!(grads.get(a).unwrap().shape(), [2, 3]); // a's shape, b's shape
    assert_eq!(grads.get(b).unwrap().shape(), [3, 2]);
    // @end
}

#[test]
fn batched_matmul_backward() {
    // @listing batched
    let tape = Tape::new();
    let a = tape.leaf(Tensor::arange(&[2, 2, 3])); // two 2 × 3 matrices
    let b = tape.leaf(Tensor::arange(&[2, 3, 2])); // two 3 × 2 matrices
    let w = t(&[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], &[2, 2, 2]);
    let grads = (a.matmul(b) * tape.constant(w)).sum_all().backward();
    let (ga, gb) = (grads.get(a).unwrap(), grads.get(b).unwrap());
    let mat = |g: &Tensor, i| g.select(0, i).to_vec(); // matrix i of a stack
    assert_eq!(mat(ga, 0), [2.0, 8.0, 14.0, 4.0, 18.0, 32.0]); // Section 10.1's
    assert_eq!(mat(ga, 1), [20.0, 26.0, 32.0, 46.0, 60.0, 74.0]);
    assert_eq!(mat(gb, 0), [9.0, 12.0, 13.0, 18.0, 17.0, 24.0]);
    assert_eq!(mat(gb, 1), [33.0, 48.0, 37.0, 54.0, 41.0, 60.0]);
    // @end
    // Matrix 1 on its own gives the same gradients.
    let tape = Tape::new();
    let a1 = tape.leaf(Tensor::arange(&[2, 2, 3]).select(0, 1).contiguous());
    let b1 = tape.leaf(Tensor::arange(&[2, 3, 2]).select(0, 1).contiguous());
    let g = (a1.matmul(b1) * tape.constant(w22())).sum_all().backward();
    assert_eq!(grad(&g, a1), mat(ga, 1));
    assert_eq!(grad(&g, b1), mat(gb, 1));
}

#[test]
fn backward_costs_two_products() {
    // @listing flops
    let flops = |(m, k, n): (u64, u64, u64)| 2 * m * k * n;
    let (m, k, n) = (64, 784, 128); // a batch of 64 MNIST images, 128 features
    let forward = flops((m, k, n)); // [m, k] @ [k, n]
    let da = flops((m, n, k)); // dC @ Bᵀ: [m, n] @ [n, k]
    let db = flops((k, m, n)); // Aᵀ @ dC: [k, m] @ [m, n]
    assert_eq!(forward, 12_845_056);
    assert_eq!((da, db), (forward, forward));
    assert_eq!(da + db, 25_690_112); // twice the forward pass
    // @end
}

#[test]
fn the_size_of_a_nudge() {
    // @listing nudge-size
    let e = 1f64.exp(); // the slope of e^x at 1
    let exp = |x: f32| Tensor::from_vec(vec![x], &[1]).exp().item() as f64;
    let error = |h: f32| {
        let slope = (exp(1.0 + h) - exp(1.0 - h)) / (2.0 * h as f64);
        (slope - e).abs()
    };
    let hs = [1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5, 1e-6];
    let errors: Vec<f64> = hs.iter().map(|&h| error(h)).collect();
    let best = (0..7).min_by(|&i, &j| errors[i].total_cmp(&errors[j]));
    assert_eq!(best.map(|i| hs[i]), Some(5e-3)); // near ε^(1/3)
    // @end
    let want = [
        4.53e-3, 3.57e-5, 2.38e-5, 4.75e-5, 8.82e-4, 1.16e-2, 9.57e-2,
    ];
    for (got, want) in errors.iter().zip(want) {
        assert!(near(*got, want, 0.01), "{errors:?}");
    }
    // The model: truncation e·h²/6 and rounding at most ε·e/h.
    let eps = f32::EPSILON as f64;
    assert!(near(eps, 1.19e-7, 0.01));
    assert!(near(eps.cbrt(), 4.92e-3, 0.01));
    assert!(near(eps.powf(2.0 / 3.0), 2.42e-5, 0.01)); // the error at the best h
    let truncation = [4.53e-3, 4.53e-5, 1.13e-5, 4.53e-7, 4.53e-9];
    let rounding = [3.24e-6, 3.24e-5, 6.48e-5, 3.24e-4, 3.24e-3, 3.24e-2, 0.324];
    for (i, &h) in hs.iter().enumerate() {
        let h = h as f64;
        if i < 5 {
            assert!(near(e * h * h / 6.0, truncation[i], 0.01));
        }
        assert!(near(eps * e / h, rounding[i], 0.01));
    }
    assert!(near((3.0 * eps).cbrt(), 7.09e-3, 0.01)); // the best h for e^x
    for (&h, got) in hs.iter().zip(&errors) {
        let h = h as f64;
        assert!(*got <= e * h * h / 6.0 + eps * e / h); // within the two terms
    }
}

#[test]
fn an_absolute_tolerance_fails_large_gradients() {
    // @listing scales
    let x = [t(&[10.0, -10.0], &[2])]; // e^10 = 22026.5, e^-10 = 0.0000454
    let absolute = Tolerance {
        rtol: 0.0,
        ..Tolerance::default()
    }; // atol only
    let failed = gradcheck(|v| v[0].exp(), &x, absolute).unwrap_err();
    assert_eq!((failed.index, failed.analytic), (vec![0], 22026.465));
    assert!(near(failed.numeric as f64, 22035.64, 1e-6)); // off by 9.2
    assert!(gradcheck(|v| v[0].exp(), &x, Tolerance::default()).is_ok());
    // @end
    assert!(near(failed.ratio as f64, 9173.8, 1e-4));
    let passed = gradcheck(|v| v[0].exp(), &x, Tolerance::default()).unwrap();
    assert!(near(passed.ratio as f64, 0.0416, 0.01)); // 9.17 / (0.001 + 220.4)
    assert!(near(0.01 * failed.numeric as f64, 220.4, 1e-3));
    // The error is truncation: e^x h²/6 with h = 0.05.
    assert!(near(22026.465 * 0.05 * 0.05 / 6.0, 9.18, 1e-3));
    assert!(near(9.17 / 22026.465, 4.2e-4, 0.01));
    assert!(near(0.05 * 0.05 / 6.0, 4.2e-4, 0.01));
}

// @listing golden
/// `n` numbers spread evenly over `[lo, hi)`, no two close: `lo + (hi - lo)` times
/// the fractional part of `0.618034 k` (the golden ratio), for `k = 1, 2, …`.
fn golden(shape: &[usize], lo: f32, hi: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let frac = |k: usize| (k as f32 * 0.618_034).fract();
    Tensor::from_vec((1..=n).map(|k| lo + (hi - lo) * frac(k)).collect(), shape)
}
// @end

type Check = for<'t> fn(&[Var<'t>]) -> Var<'t>;

/// Every rule of Chapters 7 to 10; `v[0]` has either sign, `v[1]` is positive.
const RULES: [(&str, Check); 27] = [
    ("add", |v| v[0] + v[1]),
    ("mul", |v| v[0] * v[1]),
    ("sum_all", |v| v[0].sum_all()),
    ("sub", |v| v[0] - v[1]),
    ("div", |v| v[0] / v[1]),
    ("neg", |v| -v[0]),
    ("add_scalar", |v| v[0] + 2.0),
    ("mul_scalar", |v| v[0] * 3.0),
    ("div_scalar", |v| v[0] / 3.0),
    ("exp", |v| v[0].exp()),
    ("ln", |v| v[1].ln()),
    ("powf", |v| v[1].powf(1.5)),
    ("tanh", |v| v[0].tanh()),
    ("sigmoid", |v| v[0].sigmoid()),
    ("relu", |v| v[0].relu()),
    ("expand", |v| v[0].expand(&[3, 2, 3, 4])),
    ("sum", |v| v[0].sum(1, false)),
    ("mean", |v| v[0].mean(2, true)),
    ("mean_all", |v| v[0].mean_all()),
    ("max", |v| v[0].max(2, false)),
    ("reshape", |v| v[0].reshape(&[4, 6])),
    ("transpose", |v| v[0].transpose(0, 2)),
    ("permute", |v| v[0].permute(&[1, 2, 0])),
    ("slice", |v| v[0].slice(2, 1, 4, 2)),
    ("select", |v| v[0].select(1, 2)),
    ("matmul", |v| {
        v[0].select(0, 0).matmul(v[1].select(0, 1).transpose(0, 1))
    }),
    ("batched matmul", |v| v[0].matmul(v[1].transpose(1, 2))),
];

#[test]
fn every_rule_against_gradcheck() {
    // @listing every-rule
    let x = golden(&[2, 3, 4], -2.0, 2.0); // either sign, no ties, no zeros
    let p = golden(&[2, 3, 4], 0.5, 2.5); // positive, for ln, powf and div
    let mut table = Vec::new();
    for (name, f) in RULES {
        let w = gradcheck(f, &[x.clone(), p.clone()], Tolerance::default())
            .unwrap_or_else(|w| panic!("{name}: {w:?}"));
        let rel = (w.analytic - w.numeric).abs() / w.numeric.abs().max(1e-30);
        table.push((name, rel, w.ratio));
    }
    // @end
    let want = [
        ("add", 2.4e-5),
        ("mul", 8.3e-6),
        ("sum_all", 1.5e-6),
        ("sub", 2.4e-5),
        ("div", 7.7e-5),
        ("add_scalar", 2.4e-5),
        ("mul_scalar", 9.7e-6),
        ("div_scalar", 7.3e-6),
        ("exp", 1.4e-5),
        ("ln", 2.6e-5),
        ("powf", 6.1e-6),
        ("tanh", 7.0e-5),
        ("sigmoid", 2.8e-5),
        ("sum", 2.4e-5),
        ("mean", 2.4e-5),
        ("mean_all", 1.1e-5),
        ("matmul", 1.3e-5),
        ("batched matmul", 9.4e-5),
    ];
    let exact = ["neg", "relu", "expand", "max", "reshape", "transpose"];
    let exact = [&exact[..], &["permute", "slice", "select"]].concat();
    assert_eq!(want.len() + exact.len(), 27);
    for (name, rel, ratio) in &table {
        match want.iter().find(|(n, _)| n == name) {
            Some((_, r)) => assert!(near(*rel as f64, *r, 0.05), "{name}: {rel}"),
            None => assert!(exact.contains(name) && *rel == 0.0, "{name}: {rel}"),
        }
        assert!(*ratio < 0.01, "{name}: {ratio}"); // under 1% of the tolerance
    }
    let largest = table.iter().map(|r| r.2).fold(0.0, f32::max);
    assert!(near(largest as f64, 0.0093, 0.01)); // batched matmul
}

#[test]
fn a_layer_checked() {
    // @listing layer
    let x = golden(&[4, 3], -1.0, 1.0); // a batch of four inputs, three features
    let w = golden(&[3, 2], -0.5, 0.5); // weights from three features to two
    let b = golden(&[2], -0.1, 0.1); // one bias per output feature
    let layer = gradcheck(
        |v| (v[0].matmul(v[1]) + v[2]).tanh(), // tanh(x W + b)
        &[x, w, b],
        Tolerance::default(),
    );
    let worst = layer.unwrap(); // all 20 elements of x, W and b pass
    assert_eq!((worst.input, worst.index), (1, vec![2, 0])); // W[2][0]
    assert!(near(worst.analytic as f64, 1.10134, 1e-5));
    // @end
    assert!(near(worst.numeric as f64, 1.10137, 1e-5));
    assert!(worst.ratio < 0.002);
}
