//! Burn 0.21: `AutodiffModule::valid()` and `from_inner()` drop every
//! parameter's save/load mapper.
//!
//! A `LinearLayout::Col` weight is held as `[in, out]` and saved as
//! `[out, in]` through its save mapper; loading transposes it back. After
//! `valid()` (evaluation) or `from_inner()` the mapper is gone, so the module
//! saves `[in, out]`, and loading that file into a column-layout module
//! transposes it again: a square projection silently computes with `Wᵀ`, a
//! non-square one is saved with the wrong shape.

use burn::backend::{Autodiff, Flex};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig, LinearLayout};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Distribution, Tensor, backend::Backend};

/// The backend under test: Flex with autodiff.
type B = Autodiff<Flex>;
/// In-memory full-precision recorder used for the save/load round trip.
type Bytes = BinBytesRecorder<FullPrecisionSettings>;

/// Saves a `Linear<S>` record to bytes and loads it into a fresh inference module.
fn round_trip<S: Backend>(
    record: <Linear<S> as Module<S>>::Record,
    config: &LinearConfig,
) -> Linear<Flex> {
    let device = Default::default();
    let recorder = Bytes::default();
    let bytes = Recorder::<S>::record(&recorder, record, ()).unwrap();
    let loaded: <Linear<Flex> as Module<Flex>>::Record =
        Recorder::<Flex>::load(&recorder, bytes, &device).unwrap();
    config.init::<Flex>(&device).load_record(loaded)
}

/// Largest absolute difference between two tensors.
fn max_diff(a: Tensor<Flex, 2>, b: Tensor<Flex, 2>) -> f32 {
    let (a, b): (Vec<f32>, Vec<f32>) = (
        a.into_data().to_vec().unwrap(),
        b.into_data().to_vec().unwrap(),
    );
    a.iter().zip(&b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

/// Saves and reloads through each route and reports outputs and saved shapes.
fn main() {
    let device = Default::default();

    // Square projection: values after a save/load round trip.
    let config = LinearConfig::new(4, 4).with_layout(LinearLayout::Col);
    let owner = config.init::<B>(&device);
    let x = Tensor::<Flex, 2>::random([3, 4], Distribution::Default, &device);
    let expected = owner.valid().forward(x.clone());
    println!("Col-layout 4 -> 4 projection, saved and reloaded into a fresh inference module:");
    for (route, reloaded) in [
        (
            "training owner (Autodiff) record",
            round_trip::<B>(owner.clone().into_record(), &config),
        ),
        (
            "owner.valid() record",
            round_trip::<Flex>(owner.valid().into_record(), &config),
        ),
        (
            "Linear::from_inner(owner.valid()) record",
            round_trip::<B>(
                Linear::<B>::from_inner(owner.valid()).into_record(),
                &config,
            ),
        ),
    ] {
        println!(
            "  {route:<42} max |y - expected| = {:.2e}",
            max_diff(reloaded.forward(x.clone()), expected.clone())
        );
    }

    // Non-square projection: the saved shape is the checkpoint-facing [out, in] only for the owner.
    let config = LinearConfig::new(4, 3).with_layout(LinearLayout::Col);
    let owner = config.init::<B>(&device);
    println!(
        "Col-layout 4 -> 3 projection, saved weight shape (checkpoint layout is [out, in] = [3, 4]):"
    );
    println!(
        "  {:<42} {:?}",
        "training owner record",
        owner.clone().into_record().weight.val().dims()
    );
    println!(
        "  {:<42} {:?}",
        "owner.valid() record",
        owner.valid().into_record().weight.val().dims()
    );
    println!(
        "  {:<42} {:?}",
        "Linear::from_inner(owner.valid()) record",
        Linear::<B>::from_inner(owner.valid())
            .into_record()
            .weight
            .val()
            .dims()
    );
}
