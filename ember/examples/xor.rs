//! Chapter 14: a straight line can't fit XOR; one hidden layer can.
//!
//! cargo run --release --example xor

use ember::{Linear, Module, Rng, Sequential, Tanh, Tape, Tensor, mse_loss, toy};

// @listing step
/// One step of gradient descent on the mean squared error, from a fresh tape.
/// Returns the loss before the step.
pub fn step(model: &mut impl Module, x: &Tensor, y: &Tensor, lr: f32) -> f32 {
    model.zero_grad(); // 1. forget the last step's gradients
    let loss = {
        let tape = Tape::new();
        let pred = model.forward(tape.constant(x.clone())); // 2. forward
        let loss = mse_loss(pred, tape.constant(y.clone())); // 3. loss
        loss.backward_into(&mut model.parameters()); // 4. backward
        loss.value().item()
    }; // the tape is dropped here
    sgd(model, lr); // 5. update
    loss
}

/// Move every parameter against its gradient: w ← w - lr·w̄.
pub fn sgd(model: &mut impl Module, lr: f32) {
    for p in model.parameters() {
        let g = p.grad.as_ref().expect("a gradient");
        p.value = p.value.sub(&g.mul_scalar(lr));
    }
}
// @end

// @listing mlp
/// 2 inputs, `hidden` tanh units, 1 output.
pub fn mlp(hidden: usize, rng: &mut Rng) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(2, hidden, rng)),
        Box::new(Tanh),
        Box::new(Linear::new(hidden, 1, rng)),
    ])
}
// @end

/// The model's outputs for `x`, one per row.
pub fn outputs(model: &impl Module, x: &Tensor) -> Vec<f32> {
    let tape = Tape::new();
    model.forward(tape.constant(x.clone())).value().to_vec()
}

// @listing main
fn main() {
    let (x, y) = toy::xor();
    let mut line = Linear::new(2, 1, &mut Rng::new(0));
    let losses: Vec<f32> = (0..1000).map(|_| step(&mut line, &x, &y, 0.5)).collect();
    println!("linear model: loss {:.4} after 1000 steps", losses[999]);

    let mut model = mlp(4, &mut Rng::new(0));
    println!("{:>6} {:>10}", "step", "loss");
    for i in 1..=500 {
        let loss = step(&mut model, &x, &y, 0.1);
        if i == 1 || i % 100 == 0 {
            println!("{i:>6} {loss:>10.6}");
        }
    }
    println!("outputs: {:.3?}", outputs(&model, &x));
}
// @end
