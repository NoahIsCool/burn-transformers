//! Burn 0.22 autodiff: a valid backward panics with "graph tape has already been
//! consumed" after two graph merges (Burn 0.21 silently returned no gradient).
//!
//! `GraphLocator::merge_two` (burn-autodiff/src/runtime/graph.rs) moves the
//! merged graph's steps into the main graph and re-maps the merged graph's
//! *keys*, but the merged graph's *origin* is never added to the main graph's
//! keys. When the main graph is later merged into a third graph, every key is
//! re-mapped again but that origin still points at the emptied graph. An op on
//! the origin tensor then registers its step in the dead graph, so the backward
//! pass started there cannot reach the origin's step.
//!
//! A tensor becomes an origin when none of its parents is still registered,
//! for example the first op on a parameter after a previous backward pass
//! consumed the parameter's leaf step. Burn 0.22 keeps the graph with the
//! smallest origin node id in a merge (a `BTreeMap`), so the outcome depends
//! only on creation order: with the side leaves created after `a`, `a`'s graph
//! stays the main graph and nothing is lost; with them created before `a`
//! (`q`, then `p`), `a`'s graph is merged into `p`'s, which is merged into
//! `q`'s, and every trial panics. The panic comes from the tape check Burn 0.22
//! added; in 0.21 the same bookkeeping error returned `grad == None`.

use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::tensor::{Distribution, Tensor};

/// One trial; returns whether `w` received a gradient.
///
/// `side_leaves_first` creates the two side leaves `q`, `p` (in that order)
/// before `a`, so their graph origins have smaller node ids than `a`'s.
fn trial(side_leaves_first: bool) -> bool {
    let device = burn::tensor::Device::flex().autodiff();
    let leaf = || Tensor::<2>::random([4, 4], Distribution::Default, &device).require_grad();
    let w = leaf();
    let early = side_leaves_first.then(|| {
        let q = leaf();
        let p = leaf();
        (p, q)
    });
    // A first backward pass consumes `w`'s leaf step.
    let _ = w.clone().sum().backward();

    // `a` has no registered parent graph left, so it becomes the origin of a new graph.
    let a = w.clone() * 2.0;
    let (p, q) = early.unwrap_or_else(|| (leaf(), leaf()));
    // Two merges: `a`'s graph is merged into another graph, which is merged again.
    let _side = (a.clone() * p) * q;

    let grads = a.sum().backward();
    w.grad(&grads).is_some()
}

/// Runs the trials for both creation orders and prints how many lost the gradient.
fn main() {
    let trials = 1000;
    std::panic::set_hook(Box::new(|info| {
        let mut slot = first_panic_slot().lock().unwrap();
        slot.get_or_insert_with(|| info.to_string());
    }));
    for (side_leaves_first, label) in [
        (false, "side leaves created after `a`"),
        (true, "side leaves created before `a`"),
    ] {
        let (mut lost, mut panicked) = (0, 0);
        for _ in 0..trials {
            match catch_unwind(AssertUnwindSafe(|| trial(side_leaves_first))) {
                Ok(true) => {}
                Ok(false) => lost += 1,
                Err(_) => panicked += 1,
            }
        }
        println!(
            "{label}: {lost} of {trials} backward passes returned no gradient for `w`, {panicked} panicked"
        );
    }
    if let Some(message) = first_panic_slot().lock().unwrap().take() {
        println!("first panic: {message}");
    }
}

/// The first panic message, shared with the panic hook.
fn first_panic_slot() -> &'static std::sync::Mutex<Option<String>> {
    /// Storage for the first panic message.
    static SLOT: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    &SLOT
}
