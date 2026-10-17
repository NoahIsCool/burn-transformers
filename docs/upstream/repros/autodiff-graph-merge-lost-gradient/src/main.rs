//! Burn 0.21 autodiff: backward silently drops gradients after graph merges.
//!
//! `GraphLocator::merge_two` (burn-autodiff/src/runtime/graph.rs) moves the
//! merged graph's steps into the main graph and re-maps the merged graph's
//! *keys*, but the merged graph's *origin* is never added to the main graph's
//! keys. When the main graph is later merged into a third graph, every key is
//! re-mapped again but that origin still points at the emptied graph. An op on
//! the origin tensor then registers its step in the dead graph, and a backward
//! pass started there cannot reach the origin's step: the gradient of every
//! leaf behind it is silently missing (`grad` returns `None`).
//!
//! A tensor becomes an origin when none of its parents is still registered,
//! for example the first op on a parameter after a previous backward pass
//! consumed the parameter's leaf step. In practice: a router whose logits come
//! from `module::linear(x, w, Some(b))` on every training step after the
//! first, with the routed dispatch merging the logits' graph with the experts'.
//! Which graph survives a merge depends on `HashMap` iteration order, so the
//! loss is nondeterministic.

use burn::backend::{Autodiff, Flex};
use burn::tensor::{Distribution, Tensor};

type B = Autodiff<Flex>;

/// One trial; returns whether `w` received a gradient.
fn trial() -> bool {
    let device = Default::default();
    let w = Tensor::<B, 2>::random([4, 4], Distribution::Default, &device).require_grad();
    // A first backward pass consumes `w`'s leaf step.
    let _ = w.clone().sum().backward();

    // `a` has no registered parent graph left, so it becomes the origin of a new graph.
    let a = w.clone() * 2.0;
    let p = Tensor::<B, 2>::random([4, 4], Distribution::Default, &device).require_grad();
    let q = Tensor::<B, 2>::random([4, 4], Distribution::Default, &device).require_grad();
    // Two merges: `a`'s graph is merged into another graph, which is merged again.
    let _side = (a.clone() * p) * q;

    let grads = a.sum().backward();
    w.grad(&grads).is_some()
}

/// Runs the trials and prints how many lost the gradient.
fn main() {
    let trials = 1000;
    let lost = (0..trials).filter(|_| !trial()).count();
    println!("{lost} of {trials} backward passes returned no gradient for `w`");
}
