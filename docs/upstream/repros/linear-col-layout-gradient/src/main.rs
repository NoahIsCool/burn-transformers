//! Burn 0.21: a `LinearLayout::Col` weight is not an autodiff leaf.
//!
//! `LinearConfig::init` draws a column-layout weight as `[out, in]` and adds
//! an `init_mapper` that transposes it. `Param::init_mapper` applies that
//! transform *after* the initializer has marked the tensor as requiring a
//! gradient, so the stored parameter is the transpose node, not a leaf:
//!
//! 1. The first backward pass consumes that node's gradient, `Param::grad`
//!    returns `None`, and `GradientsParams::from_grads` hands the optimizer
//!    nothing for the weight: the first optimizer step silently skips every
//!    column-layout weight. (Later passes happen to work only because the
//!    transpose step was consumed by the first one.)
//! 2. `load_record` of a record taken from a live autodiff module applies the
//!    `load_mapper` transpose to a tracked tensor and then calls
//!    `set_require_grad(true)` on the resulting non-leaf, which panics with
//!    "Can't convert a non leaf tensor into a tracked tensor".

use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::backend::{Autodiff, Flex};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, LinearLayout};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::{Distribution, Tensor};

type B = Autodiff<Flex>;

/// Whether the weight has a gradient on each of three consecutive passes.
fn weight_gradient_per_pass(linear: &Linear<B>) -> Vec<bool> {
    let device = Default::default();
    (0..3)
        .map(|_| {
            let x = Tensor::<B, 2>::random([2, 4], Distribution::Default, &device);
            let grads = linear.forward(x).sum().backward();
            linear.weight.grad(&grads).is_some()
        })
        .collect()
}

/// Whether the first SGD step changes the weight.
fn first_step_changes_weight(linear: Linear<B>) -> bool {
    let device = Default::default();
    let x = Tensor::<B, 2>::random([2, 4], Distribution::Default, &device);
    let before = linear.weight.val().into_data();
    let grads = GradientsParams::from_grads(linear.forward(x).sum().backward(), &linear);
    let mut optimizer = SgdConfig::new().init::<B, Linear<B>>();
    optimizer.step(0.1, linear, grads).weight.val().into_data() != before
}

/// Reports gradient, optimizer and load behavior for both layouts.
fn main() {
    std::panic::set_hook(Box::new(|info| {
        let message = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .unwrap_or("?");
        println!("    panic: {message}");
    }));
    let device = Default::default();
    for layout in [LinearLayout::Row, LinearLayout::Col] {
        let config = LinearConfig::new(4, 3).with_layout(layout);
        println!("{layout:?} layout");
        println!(
            "  first SGD step changes the weight: {}",
            first_step_changes_weight(config.init::<B>(&device))
        );
        println!(
            "  weight gradient on passes 1..3: {:?}",
            weight_gradient_per_pass(&config.init::<B>(&device))
        );
        let record = config.init::<B>(&device).into_record();
        let loaded = catch_unwind(AssertUnwindSafe(|| {
            config.init::<B>(&device).load_record(record)
        }));
        match loaded {
            Ok(loaded) => println!(
                "  load_record from a live module: ok, gradients {:?}",
                weight_gradient_per_pass(&loaded)
            ),
            Err(_) => println!("  load_record from a live module: panicked"),
        }
    }
}
