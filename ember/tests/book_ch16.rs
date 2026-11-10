//! The examples printed in Chapter 16 of the guided edition: momentum and Nesterov.
use ember::{Optimizer, Parameter, Rng, Sgd, Tensor, plot, toy};

#[path = "../examples/spirals.rs"]
#[allow(dead_code)]
mod spirals;

fn close(xs: &[f32], want: &[f32], tol: f32) -> bool {
    xs.len() == want.len() && xs.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

/// One step of `opt` on f(w) = ½w², whose gradient is w itself. Returns the
/// velocity `opt` holds for `w` afterward, or 0 if it holds none.
fn bowl_step(w: &mut Parameter, opt: &mut Sgd) -> f32 {
    w.grad = Some(w.value.clone()); // shares w's buffer: make_mut copies
    opt.step(&mut [&mut *w]);
    let state = opt.state(w.id()).unwrap();
    state.momentum_buffer.as_ref().map_or(0.0, |v| v.item())
}

#[test]
fn momentum_by_hand() {
    // @listing trace
    let start = || Parameter::new(Tensor::from_vec(vec![1.0], &[1]));
    let (mut plain, mut heavy) = (start(), start());
    let (mut sgd, mut momentum) = (Sgd::new(0.1), Sgd::new(0.1).momentum(0.9));
    let (mut ws, mut vs, mut hs) = (vec![], vec![], vec![]);
    for _ in 0..5 {
        bowl_step(&mut plain, &mut sgd);
        vs.push(bowl_step(&mut heavy, &mut momentum));
        ws.push(plain.value.item());
        hs.push(heavy.value.item());
    }
    assert!(close(&ws, &[0.9, 0.81, 0.729, 0.6561, 0.59049], 1e-6));
    assert!(close(&vs, &[1.0, 1.8, 2.34, 2.592, 2.5596], 1e-6));
    assert!(close(&hs, &[0.9, 0.72, 0.486, 0.2268, -0.02916], 1e-6));
    // @end
    // The overshoot: on to w = -0.6037 at step 9, the loss rising from step 6.
    for _ in 5..9 {
        bowl_step(&mut heavy, &mut momentum);
    }
    assert!((heavy.value.item() + 0.6037).abs() < 1e-4);
    assert!((0.5 * heavy.value.item().powi(2) - 0.1822).abs() < 1e-4);
}

#[test]
fn nesterov_by_hand() {
    // @listing trace-nesterov
    let mut w = Parameter::new(Tensor::from_vec(vec![1.0], &[1]));
    let mut nesterov = Sgd::new(0.1).momentum(0.9).nesterov(true);
    let (mut gs, mut vs, mut ws) = (vec![], vec![], vec![]);
    for _ in 0..5 {
        gs.push(w.value.item()); // the gradient of ½w² is w
        vs.push(bowl_step(&mut w, &mut nesterov));
        ws.push(w.value.item());
    }
    assert!(close(&gs, &[1.0, 0.81, 0.5751, 0.327321, 0.093888], 1e-6));
    assert!(close(&vs, &[1.0, 1.71, 2.1141, 2.230011, 2.100898], 1e-6));
    let want = [0.81, 0.5751, 0.327321, 0.093888, -0.104582];
    assert!(close(&ws, &want, 1e-6));
    // @end
}

#[test]
fn the_velocity_on_a_steady_slope() {
    // @listing slope
    let mut w = Parameter::new(Tensor::from_vec(vec![0.0], &[1]));
    let mut opt = Sgd::new(0.1).momentum(0.9);
    let mut moves = vec![];
    for _ in 0..50 {
        let before = w.value.item();
        w.grad = Some(Tensor::from_vec(vec![1.0], &[1])); // f(w) = w
        opt.step(&mut [&mut w]);
        moves.push(before - w.value.item()); // lr · v
    }
    let at = [1, 2, 10, 50].map(|t| moves[t - 1]);
    assert!(close(&at, &[0.1, 0.19, 0.65132, 0.99485], 1e-5)); // toward 0.1/(1 - 0.9)
    // @end
}

// @listing bowl
/// The weights and the loss after each step.
type Path = Vec<(Vec<f32>, f32)>;

/// `steps` steps of `opt` on the bowl f(w) = ½ Σ c_i w_i², from `start`, with the
/// gradient c_i w_i set by hand.
fn bowl(c: &[f32], start: &[f32], opt: &mut Sgd, steps: usize) -> Path {
    let n = start.len();
    let mut w = Parameter::new(Tensor::from_vec(start.to_vec(), &[n]));
    let mut path = Vec::with_capacity(steps);
    for _ in 0..steps {
        let g = w.value.to_vec().iter().zip(c).map(|(w, c)| c * w).collect();
        w.grad = Some(Tensor::from_vec(g, &[n]));
        opt.step(&mut [&mut w]);
        let now = w.value.to_vec();
        let loss = now.iter().zip(c).map(|(w, c)| 0.5 * c * w * w).sum();
        path.push((now, loss));
    }
    path
}

/// The step from which the loss stays below `tol`, if it does by the end.
fn settles(path: &Path, tol: f32) -> Option<usize> {
    let above = |(_, loss): &(Vec<f32>, f32)| loss.is_nan() || *loss >= tol;
    match path.iter().rposition(above) {
        None => Some(1),
        Some(i) if i + 1 < path.len() => Some(i + 2),
        Some(_) => None, // still above it at the end, or diverged
    }
}
// @end

fn losses(path: &Path) -> Vec<f32> {
    path.iter().map(|(_, l)| *l).collect()
}

const VALLEY: [f32; 2] = [1.0, 100.0];

#[test]
fn plain_sgd_in_a_valley() {
    // @listing valley
    let path = bowl(&VALLEY, &[1.0, 1.0], &mut Sgd::new(0.0195), 1000);
    let xs: Vec<f32> = path[..4].iter().map(|(w, _)| w[0]).collect();
    let ys: Vec<f32> = path[..4].iter().map(|(w, _)| w[1]).collect();
    assert!(close(&xs, &[0.9805, 0.96138, 0.94263, 0.92425], 1e-5));
    assert!(close(&ys, &[-0.95, 0.9025, -0.85738, 0.81451], 1e-5));
    assert_eq!(settles(&path, 1e-6), Some(334));
    // The best of the learning rates 0.0005, 0.001, …, 0.02:
    let best = (1..=40)
        .map(|k| k as f32 * 0.0005)
        .filter_map(|lr| {
            let path = bowl(&VALLEY, &[1.0, 1.0], &mut Sgd::new(lr), 1000);
            settles(&path, 1e-6).map(|steps| (steps, lr))
        })
        .min_by_key(|&(steps, _)| steps);
    assert_eq!(best, Some((334, 0.0195)));
    // @end
}

#[test]
fn momentum_in_a_valley() {
    // @listing valley-momentum
    let settle =
        |opt: Sgd| settles(&bowl(&VALLEY, &[1.0, 1.0], &mut { opt }, 3000), 1e-6);
    let momentum = |lr| Sgd::new(lr).momentum(0.9);
    let nesterov = |lr| Sgd::new(lr).momentum(0.9).nesterov(true);
    assert_eq!(settle(Sgd::new(0.008)), Some(817));
    assert_eq!(settle(momentum(0.008)), Some(169));
    assert_eq!(settle(nesterov(0.008)), Some(109));
    assert_eq!(settle(Sgd::new(0.0195)), Some(334));
    assert_eq!(settle(momentum(0.0195)), Some(172));
    assert_eq!(settle(nesterov(0.0195)), None); // diverges
    assert_eq!(settle(momentum(0.005)), Some(166));
    assert_eq!(settle(momentum(0.035)), Some(192));
    // @end
}

/// The figure in Section 16.6: three runs of 240 steps on the valley, side by side
/// on one log scale.
#[test]
fn the_valley_figure() {
    let mut all = vec![];
    for mut opt in [
        Sgd::new(0.0195),
        Sgd::new(0.0195).momentum(0.9),
        Sgd::new(0.008).momentum(0.9).nesterov(true),
    ] {
        all.extend(losses(&bowl(&VALLEY, &[1.0, 1.0], &mut opt, 240)));
    }
    let figure = plot::curve(&all, 72, 12);
    let trimmed: Vec<&str> = figure.lines().map(str::trim_end).collect();
    assert_eq!(
        trimmed,
        FIGURE.trim_matches('\n').lines().collect::<Vec<_>>()
    );
    let ends = [239, 479, 719].map(|i| all[i]); // the loss after 240 steps
    assert!((ends[0] - 3.925e-5).abs() < 1e-8);
    assert!((ends[1] - 4.84e-10).abs() < 1e-12);
    assert!((ends[2] - 1.887e-13).abs() < 1e-15);
}

const FIGURE: &str = r"
 30.0958 |**                      **
         |  ***                     ***                   *
         |     ***                     **                  *
         |        *******                ***                ***
         |               *******            **                 **
         |                      **            ***                ***
         |                                       ***                *
         |                                          **               ****
         |                                            ***                * *
         |                                               *                * *
         |                                                                   * **
  0.0000 |                                                                    *  *
         +------------------------------------------------------------------------
         0                                                                     720
";

#[test]
fn stability_limits() {
    // @listing limits
    let last = |opt: Sgd| bowl(&[1.0], &[1.0], &mut { opt }, 500)[499].1;
    let momentum = |lr| Sgd::new(lr).momentum(0.9);
    let nesterov = |lr| Sgd::new(lr).momentum(0.9).nesterov(true);
    assert!(last(Sgd::new(1.9)) < 1e-6 && last(Sgd::new(2.1)).is_infinite());
    assert!(last(momentum(3.7)) < 1e-6 && last(momentum(3.9)).is_nan());
    assert!(last(nesterov(1.3)) < 1e-6 && last(nesterov(1.4)) > 1e37);
    // @end
    // After 50 steps at lr 0.1 on the round bowl: momentum is far behind.
    let at50 = |opt: Sgd| bowl(&[1.0], &[1.0], &mut { opt }, 50)[49].1;
    let l = [
        at50(Sgd::new(0.1)),
        at50(momentum(0.1)),
        at50(nesterov(0.1)),
    ];
    assert!(close(&l, &[1.3281e-5, 2.2231e-3, 1.2201e-5], 1e-7));
}

#[test]
fn the_spirals_with_momentum() {
    // @listing spirals
    let run = |mut opt: Sgd, steps| {
        let mut rng = Rng::new(14);
        let small = toy::spirals(25, 1.5, 0.05, &mut rng); // Section 14.6's short run
        let mut model = spirals::mlp(16, &mut rng);
        let losses: Vec<f32> = (0..steps)
            .map(|_| spirals::step_with(&mut model, &small, &mut opt))
            .collect();
        let below = losses.iter().position(|&l| l < 0.01).map(|i| i + 1);
        (below, spirals::accuracy(&model, &small))
    };
    assert_eq!(run(Sgd::new(2.0), 1000), (Some(881), 1.0));
    assert_eq!(run(Sgd::new(0.5).momentum(0.9), 206), (Some(206), 1.0));
    let nesterov = Sgd::new(0.5).momentum(0.9).nesterov(true);
    assert_eq!(run(nesterov, 203), (Some(203), 1.0));
    // @end
}
