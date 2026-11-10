//! Chapter 20: a training step, an epoch of them, and an evaluation.

use super::{Average, accuracy};
use crate::autograd::tape::Tape;
use crate::data::{Batch, DataLoader, Dataset};
use crate::nn::{Module, cross_entropy};
use crate::optim::{Optimizer, clip_grad_norm};
use std::ops::Deref;

// @listing train-step
/// Chapter 18's step on one batch: gradients of the cross-entropy, clipped to
/// a norm of at most `max_norm`, then an update. Returns the batch's loss.
pub fn train_step(
    model: &mut dyn Module,
    (x, labels): &Batch,
    opt: &mut dyn Optimizer,
    max_norm: f32,
) -> f32 {
    model.zero_grad();
    let loss = {
        let tape = Tape::new();
        let loss = cross_entropy(model.forward(tape.constant(x.clone())), labels);
        loss.backward_into(&mut model.parameters());
        loss.value().item()
    }; // the tape is dropped
    clip_grad_norm(&mut model.parameters(), max_norm);
    opt.step(&mut model.parameters());
    loss
}
// @end

// @listing train-epoch
/// One pass over `loader`, calling `step` on every batch. Returns the mean of
/// the batches' losses, weighted by their sizes: the mean loss per example.
pub fn train_epoch<D: Dataset + ?Sized>(
    model: &mut dyn Module,
    loader: &mut DataLoader<'_, D>,
    mut step: impl FnMut(&mut dyn Module, &Batch) -> f32,
) -> f32 {
    let mut loss = Average::new();
    for batch in loader.epoch() {
        loss.add(step(model, &batch), batch.1.len());
    }
    loss.mean()
}
// @end

// @listing evaluate
/// The mean cross-entropy and the accuracy of `model` over one pass of
/// `loader`, in evaluation mode, with no backward rules recorded.
pub fn evaluate<D: Dataset + ?Sized>(
    model: &mut dyn Module,
    loader: &mut DataLoader<'_, D>,
) -> (f32, f32) {
    let model = EvalMode::new(model); // back in training mode when dropped
    let (mut loss, mut right) = (Average::new(), Average::new());
    for (x, labels) in loader.epoch() {
        let tape = Tape::new(); // one per batch, dropped at the end of it
        let logits = tape.no_grad(|| model.forward(tape.constant(x)));
        let l = tape.no_grad(|| cross_entropy(logits, &labels).value().item());
        loss.add(l, labels.len());
        right.add(accuracy(&logits.value(), &labels), labels.len());
    }
    (loss.mean(), right.mean())
}
// @end

// @listing eval-mode
/// A model switched to evaluation mode for as long as this guard lives. It
/// derefs to the model, and switches it back to training mode when dropped.
pub struct EvalMode<'m>(&'m mut dyn Module);
impl<'m> EvalMode<'m> {
    pub fn new(model: &'m mut dyn Module) -> EvalMode<'m> {
        model.train(false);
        EvalMode(model)
    }
}
impl<'m> Deref for EvalMode<'m> {
    type Target = dyn Module + 'm;
    fn deref(&self) -> &(dyn Module + 'm) {
        self.0
    }
}
impl Drop for EvalMode<'_> {
    fn drop(&mut self) {
        self.0.train(true); // also when a panic unwinds through the guard
    }
}
// @end
