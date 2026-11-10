//! The examples printed in Chapter 18 of the guided edition: learning-rate
//! schedules and gradient clipping.
use ember::{
    Adam, AdamW, ConstantLr, CosineLr, LinearWarmup, LrScheduler, Module, Optimizer,
    ParamGroup, Parameter, Rng, Sgd, StepLr, Tape, Tensor, base_lrs, clip_grad_norm,
    mse_loss, plot, toy,
};

#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;

const NO_CLIP: f32 = f32::INFINITY;

fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

fn tensor(xs: &[f32]) -> Tensor {
    Tensor::from_vec(xs.to_vec(), &[xs.len()])
}

/// The first step (counting from 1) whose loss is below `tol`.
fn below(losses: &[f32], tol: f32) -> Option<usize> {
    losses.iter().position(|&l| l < tol).map(|i| i + 1)
}

#[test]
fn adam_on_a_clean_bowl() {
    // ½w² from w = 1, Adam at 0.1, gradient w exactly: the swings die down.
    let mut opt = Adam::new(0.1);
    let mut w = Parameter::new(tensor(&[1.0]));
    let mut path = vec![];
    for _ in 0..60 {
        w.grad = Some(w.value.clone());
        opt.step(&mut [&mut w]);
        path.push(w.value.item());
    }
    let first_below = path.iter().position(|&w| w < 0.0).unwrap() + 1;
    let low =
        |a: usize, b: usize| path[a..b].iter().cloned().fold(f32::MAX, f32::min);
    let high =
        |a: usize, b: usize| path[a..b].iter().cloned().fold(f32::MIN, f32::max);
    assert_eq!(first_below, 12);
    let swings = [low(10, 30), high(30, 45), low(45, 60)];
    assert!(close(&swings, &[-0.273, 0.102, -0.042], 1e-3));
    assert_eq!(path[18], low(10, 30)); // the first swing's far end is step 19
}

// @listing noisy
/// The bowl ½w² from w = 1, but each gradient is w plus a standard normal
/// draw, as a minibatch's would be. The schedule `s` sets the rate before each
/// step. Returns w after each of 1000 steps.
fn noisy(opt: &mut dyn Optimizer, s: &dyn LrScheduler, rng: &mut Rng) -> Vec<f32> {
    let base = base_lrs(opt);
    let mut w = Parameter::new(tensor(&[1.0]));
    let mut path = Vec::with_capacity(1000);
    for t in 0..1000 {
        s.apply(opt, &base, t);
        w.grad = Some(tensor(&[w.value.item() + rng.normal()]));
        opt.step(&mut [&mut w]);
        path.push(w.value.item());
    }
    path
}
// @end

/// Adam at `lr` under `schedule` on the noisy bowl, 20 runs of 1000 steps: the
/// root mean square of w over steps 51–100, 451–500 and 951–1000.
fn floor(lr: f32, schedule: &dyn LrScheduler) -> [f32; 3] {
    let windows = [(50, 100), (450, 500), (950, 1000)];
    let (mut rng, mut sums) = (Rng::new(18), [0.0f64; 3]);
    for _ in 0..20 {
        let path = noisy(&mut Adam::new(lr), schedule, &mut rng);
        for (sum, &(a, b)) in sums.iter_mut().zip(&windows) {
            *sum += path[a..b].iter().map(|&w| (w * w) as f64).sum::<f64>();
        }
    }
    sums.map(|sum| (sum / (20 * 50) as f64).sqrt() as f32)
}

#[test]
fn the_noise_floor() {
    // @listing floor
    let rows = [
        floor(0.1, &ConstantLr),
        floor(0.01, &ConstantLr),
        floor(0.001, &ConstantLr),
        floor(0.1, &StepLr::new(250, 0.1)), // ÷10 at steps 250, 500, 750
        floor(0.1, &CosineLr::new(1000, 0.0)), // from 0.1 to 0
    ];
    let want = [
        [0.2138, 0.2192, 0.2576],
        [0.5536, 0.0778, 0.0853],
        [0.9472, 0.7060, 0.4706],
        [0.2138, 0.0752, 0.0522],
        [0.2124, 0.1524, 0.0642],
    ];
    // @end
    for (row, want) in rows.iter().zip(&want) {
        assert!(close(row, want, 1e-4), "{row:?}");
    }
}

#[test]
fn three_schedules() {
    // @listing tables
    let (step, cosine) = (StepLr::new(4, 0.5), CosineLr::new(10, 0.0));
    let warm = LinearWarmup::new(4, CosineLr::new(6, 0.0));
    let rates =
        |s: &dyn LrScheduler| (0..=10).map(|t| s.lr(0.1, t)).collect::<Vec<_>>();
    let a = [
        0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025,
    ];
    let b = [
        0.1, 0.0976, 0.0905, 0.0794, 0.0655, 0.05, 0.0345, 0.0206, 0.0095,
    ];
    let c = [
        0.025, 0.05, 0.075, 0.1, 0.1, 0.0933, 0.075, 0.05, 0.025, 0.0067, 0.0,
    ];
    assert!(close(&rates(&step), &a, 1e-8));
    assert!(close(&rates(&cosine)[..9], &b, 5e-5));
    assert!(close(&rates(&cosine)[9..], &[0.0024, 0.0], 5e-5));
    assert!(close(&rates(&warm), &c, 5e-5));
    assert_eq!(cosine.lr(0.1, 5), 0.05); // exactly: computed in f64
    // @end
    assert_eq!(cosine.lr(0.1, 25), 0.0); // past the end: the minimum
}

#[test]
fn every_group_from_its_base() {
    // @listing groups
    let head = Parameter::new(tensor(&[0.0]));
    let (ids, lr, weight_decay) = (vec![head.id()], 0.01, 0.0); // a slower group
    let mut opt = Sgd::new(0.1).group(ParamGroup {
        ids,
        lr,
        weight_decay,
    });
    let base = base_lrs(&opt);
    CosineLr::new(10, 0.0).apply(&mut opt, &base, 5); // halfway
    let lrs: Vec<f32> = opt.groups().iter().map(|g| g.lr).collect();
    assert_eq!((base, lrs), (vec![0.1, 0.01], vec![0.05, 0.005]));
    // @end
}

#[test]
fn cosine_in_f32() {
    // @listing f32-cosine
    use std::f32::consts::PI;
    let half = 0.1 * (1.0 + (PI * 5.0 / 10.0).cos()) / 2.0; // all f32
    assert_eq!((PI / 2.0).cos(), -4.371139e-8); // not 0: PI is not π
    assert_eq!(half, 0.049999997);
    // @end
}

#[test]
fn a_schedule_drawn() {
    // @listing plot
    let schedule = LinearWarmup::new(10, CosineLr::new(90, 0.0));
    let lrs: Vec<f32> = (0..100).map(|t| schedule.lr(0.1, t)).collect();
    let picture = plot::curve_linear(&lrs, 50, 10);
    // @end
    let want = concat!(
        "  0.1000 |    ********                                      \n",
        "         |            *****                                 \n",
        "         |   *             ****                             \n",
        "         |                     ****                         \n",
        "         |  *                      ***                      \n",
        "         |                            ***                   \n",
        "         | *                             ***                \n",
        "         |                                  ****            \n",
        "         |*                                     *****       \n",
        "  0.0001 |                                           *******\n",
        "         +--------------------------------------------------\n",
        "         0                                               100\n",
    );
    assert_eq!(picture, want, "\n{picture}");
}

// @listing short
/// Section 14.6's short spirals (25 points per arm, seed 14) trained by
/// `train_with`: the losses, the gradient norms and the final accuracy.
fn short(
    opt: &mut dyn Optimizer,
    schedule: &dyn LrScheduler,
    (steps, max_norm): (usize, f32),
) -> (Vec<f32>, Vec<f32>, f32) {
    let mut rng = Rng::new(14);
    let small = toy::spirals(25, 1.5, 0.05, &mut rng);
    let mut m = spirals::mlp(16, &mut rng);
    let (losses, norms) =
        spirals::train_with(&mut m, &small, opt, schedule, (steps, max_norm));
    (losses, norms, spirals::accuracy(&m, &small))
}
// @end

#[test]
fn warmup_tames_adam() {
    // @listing warmup
    let (cold, _, _) = short(&mut Adam::new(0.3), &ConstantLr, (100, NO_CLIP));
    let first = [0.694, 0.891, 3.658, 2.469, 1.315];
    assert!(close(&cold[..5], &first, 1e-3) && (cold[99] - 0.270).abs() < 1e-3);
    let warmup = LinearWarmup::new(20, ConstantLr);
    let (warm, _, _) = short(&mut Adam::new(0.3), &warmup, (60, NO_CLIP));
    let worst = warm.iter().cloned().fold(0.0, f32::max);
    assert_eq!(below(&warm, 0.01), Some(56)); // Adam at 0.03: 83 (Chapter 17)
    assert!((worst - 0.711).abs() < 5e-4);
    // @end
}

#[test]
fn adam_steps_after_one_gradient() {
    // The mean |Δw|/lr over the 354 numbers, for the first four steps at 0.3.
    let mut rng = Rng::new(14);
    let small = toy::spirals(25, 1.5, 0.05, &mut rng);
    let mut model = spirals::mlp(16, &mut rng);
    let mut opt = Adam::new(0.3);
    let numbers = |m: &mut ember::Sequential| -> Vec<f32> {
        m.parameters()
            .iter()
            .flat_map(|p| p.value.to_vec())
            .collect()
    };
    let mut ratios = vec![];
    for _ in 0..4 {
        let before = numbers(&mut model);
        spirals::step_with(&mut model, &small, &mut opt);
        let after = numbers(&mut model);
        let moved: f32 = before.iter().zip(&after).map(|(a, b)| (a - b).abs()).sum();
        ratios.push(moved / before.len() as f32 / 0.3);
    }
    assert!(close(&ratios, &[1.0, 0.736, 0.480, 0.314], 1e-3));
}

#[test]
fn cosine_on_the_spirals() {
    // @listing spirals
    let (constant, _, _) = short(&mut Adam::new(0.03), &ConstantLr, (200, NO_CLIP));
    let cosine = CosineLr::new(200, 0.0);
    let (decayed, _, _) = short(&mut Adam::new(0.03), &cosine, (200, NO_CLIP));
    let firsts = (below(&constant, 0.01), below(&decayed, 0.01));
    assert_eq!(firsts, (Some(83), Some(98)));
    let lasts = [constant[199], decayed[199]];
    assert!(close(&lasts, &[0.00112, 0.00486], 1e-5));
    // @end
}

// @listing chain
/// 20 weights in a chain, y = w₁·w₂·…·w₂₀·x, all starting at 1.2, trained
/// toward y = 1 at x = 1 with `opt`, clipping at `max_norm`. Returns the loss
/// and the gradient's norm (before clipping) at every step.
fn chain(opt: &mut dyn Optimizer, max_norm: f32, steps: usize) -> Vec<(f32, f32)> {
    let mut ws: Vec<Parameter> =
        (0..20).map(|_| Parameter::new(tensor(&[1.2]))).collect();
    let mut out = vec![];
    for _ in 0..steps {
        let mut params: Vec<&mut Parameter> = ws.iter_mut().collect();
        params.iter_mut().for_each(|p| p.grad = None);
        let loss = {
            let tape = Tape::new();
            let x = tape.constant(tensor(&[1.0]));
            let y = params.iter().fold(x, |h, w| h * tape.param(w));
            let loss = mse_loss(y, tape.constant(tensor(&[1.0])));
            loss.backward_into(&mut params);
            loss.value().item()
        };
        let norm = clip_grad_norm(&mut params, max_norm);
        opt.step(&mut params);
        out.push((loss, norm));
    }
    out
}
// @end

#[test]
fn a_cliff() {
    // @listing cliff
    let sgd = |lr, max_norm, steps| chain(&mut Sgd::new(lr), max_norm, steps);
    let big = sgd(0.01, NO_CLIP, 4); // NO_CLIP = ∞: the norm is only measured
    assert!(close(&[big[0].0, big[0].1], &[1394.1, 10669.3], 0.05));
    assert!(big[1].0.is_infinite() && big[3].0.is_nan());
    let small = sgd(0.001, NO_CLIP, 300);
    assert!(close(&[small[2].0, small[299].0], &[1.0, 1.0], 1e-6)); // stuck
    assert!(small[299].1 < 1e-8); // on a plateau
    let tiny = sgd(0.0001, NO_CLIP, 300);
    assert!(close(&[tiny[1].0, tiny[299].0], &[0.2967, 0.1425], 1e-4));
    let losses = |run: Vec<(f32, f32)>| run.iter().map(|r| r.0).collect::<Vec<_>>();
    let done = |max_norm| below(&losses(sgd(0.01, max_norm, 100)), 1e-6);
    assert_eq!((done(1.0), done(10.0)), (Some(97), Some(19)));
    // @end
    let adam = chain(&mut Adam::new(0.01), NO_CLIP, 300);
    assert!((adam[299].0 - 0.1674).abs() < 1e-3);
}

#[test]
fn clipping_by_hand() {
    // @listing clip-trace
    let mut a = Parameter::new(tensor(&[0.0, 0.0]));
    let mut b = Parameter::new(tensor(&[0.0]));
    let mut spare = Parameter::new(tensor(&[0.0])); // no gradient: skipped
    (a.grad, b.grad) = (Some(tensor(&[3.0, 4.0])), Some(tensor(&[12.0])));
    let norm = clip_grad_norm(&mut [&mut a, &mut b, &mut spare], 20.0);
    assert_eq!(norm, 13.0); // √(9 + 16 + 144), below 20: nothing changes
    let norm = clip_grad_norm(&mut [&mut a, &mut b, &mut spare], 1.0);
    assert_eq!(norm, 13.0); // the norm before clipping
    let (ga, gb) = (a.grad.unwrap().to_vec(), b.grad.unwrap().to_vec());
    assert!(close(&ga, &[0.230769, 0.307692], 1e-6)); // 3/13, 4/13
    assert!(close(&gb, &[0.923077], 1e-6) && spare.grad.is_none()); // 12/13
    // @end
}

#[test]
fn the_full_step() {
    // @listing canonical
    let mut opt = AdamW::new(0.3);
    let schedule = LinearWarmup::new(20, CosineLr::new(130, 0.0)); // 150 in all
    let (losses, norms, accuracy) = short(&mut opt, &schedule, (150, 1.0));
    assert_eq!((below(&losses, 0.01), accuracy), (Some(50), 1.0));
    assert!((losses[149] - 0.0015).abs() < 1e-4);
    assert!(norms.iter().all(|&n| n < 0.6)); // the clip never acted
    // @end
}
