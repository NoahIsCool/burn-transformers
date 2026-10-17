//! Burn Flex 0.21: `slice_assign` and `scatter` copy their whole destination.
//!
//! A decode cache writes one token `[1, 8, 1, 128]` into a preallocated
//! `[1, 8, capacity, 128]` F32 plane that the caller owns uniquely (the plane
//! is moved into `slice_assign` and the result replaces it). The work is
//! `O(region)`, so the time per write should not depend on `capacity`. On
//! Burn Flex 0.21 it grows linearly with `capacity`, because every call copies
//! the complete plane. The second table does the same for a one-element
//! `scatter` (`IndexingUpdateOp::Add`) into a `[1, vocabulary]` count buffer,
//! and the third for a one-coordinate `scatter_nd` into a `[1, vocabulary]` Int
//! (`Assign`) and F32 (`Add`) buffer, the shape of a generation step's
//! vocabulary mask. The last check confirms copy-on-write semantics, which must
//! hold before and after a fix: a write into a destination that is still shared
//! never changes the other handle.

use std::time::Instant;

use burn::backend::Flex;
use burn::tensor::{ElementConversion, IndexingUpdateOp, Int, Tensor};

/// The backend under test.
type B = Flex;

/// Median of `samples` timings of `body`, in microseconds.
fn median_us(samples: usize, mut body: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..samples)
        .map(|_| {
            let start = Instant::now();
            body();
            start.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// Times one-token writes, one-count scatters and one-coordinate `scatter_nd`s,
/// then checks copy-on-write.
fn main() {
    let device = Default::default();
    let token = Tensor::<B, 4>::ones([1, 8, 1, 128], &device);

    println!(
        "slice_assign of one [1, 8, 1, 128] token into a uniquely owned [1, 8, capacity, 128] F32 plane"
    );
    println!(
        "{:>10} {:>12} {:>14} {:>16}",
        "capacity", "plane MiB", "us per write", "plane bytes/time"
    );
    for capacity in [1024usize, 4096, 16384] {
        let mut plane = Some(Tensor::<B, 4>::zeros([1, 8, capacity, 128], &device));
        let mut position = 0;
        let us = median_us(41, || {
            let destination = plane.take().unwrap();
            let slot = position % capacity;
            plane =
                Some(destination.slice_assign([0..1, 0..8, slot..slot + 1, 0..128], token.clone()));
            position += 1;
        });
        let bytes = (8 * capacity * 128 * 4) as f64;
        println!(
            "{capacity:>10} {:>12.1} {us:>14.1} {:>11.1} GB/s",
            bytes / (1024.0 * 1024.0),
            bytes / (us * 1e3)
        );
        // Keep the result observable so nothing is optimized away.
        let check: f32 = plane.unwrap().slice([0..1, 0..1, 0..1, 0..1]).into_scalar();
        assert_eq!(check, 1.0);
    }

    println!();
    println!("scatter(Add) of one count into a uniquely owned [1, vocabulary] F32 buffer");
    println!(
        "{:>10} {:>12} {:>14}",
        "vocabulary", "buffer MiB", "us per write"
    );
    let index = Tensor::<B, 2, Int>::from_ints([[3]], &device);
    let one = Tensor::<B, 2>::ones([1, 1], &device);
    for vocabulary in [32_768usize, 262_144, 2_097_152] {
        let mut counts = Some(Tensor::<B, 2>::zeros([1, vocabulary], &device));
        let us = median_us(41, || {
            let destination = counts.take().unwrap();
            counts =
                Some(destination.scatter(1, index.clone(), one.clone(), IndexingUpdateOp::Add));
        });
        println!(
            "{vocabulary:>10} {:>12.1} {us:>14.1}",
            (vocabulary * 4) as f64 / (1024.0 * 1024.0)
        );
        let check: f32 = counts.unwrap().slice([0..1, 3..4]).into_scalar();
        assert_eq!(check, 41.0);
    }

    println!();
    println!(
        "scatter_nd of one coordinate into a uniquely owned [1, vocabulary] buffer (Int Assign, F32 Add)"
    );
    println!(
        "{:>10} {:>12} {:>14} {:>14}",
        "vocabulary", "buffer MiB", "Int us/write", "F32 us/write"
    );
    let coordinate = Tensor::<B, 2, Int>::from_ints([[0, 3]], &device);
    for vocabulary in [32_768usize, 262_144, 2_097_152] {
        let mut mask = Some(Tensor::<B, 2, Int>::zeros([1, vocabulary], &device));
        let int_us = median_us(41, || {
            let destination = mask.take().unwrap();
            mask = Some(destination.scatter_nd(
                coordinate.clone(),
                Tensor::<B, 1, Int>::from_ints([7], &device),
                IndexingUpdateOp::Assign,
            ));
        });
        let mut counts = Some(Tensor::<B, 2>::zeros([1, vocabulary], &device));
        let float_us = median_us(41, || {
            let destination = counts.take().unwrap();
            counts = Some(destination.scatter_nd(
                coordinate.clone(),
                Tensor::<B, 1>::ones([1], &device),
                IndexingUpdateOp::Add,
            ));
        });
        println!(
            "{vocabulary:>10} {:>12.1} {int_us:>14.1} {float_us:>14.1}",
            (vocabulary * 4) as f64 / (1024.0 * 1024.0)
        );
        let assigned: i64 = mask.unwrap().slice([0..1, 3..4]).into_scalar().elem();
        let added: f32 = counts.unwrap().slice([0..1, 3..4]).into_scalar();
        assert_eq!((assigned, added), (7, 41.0));
    }

    // Copy-on-write: a shared destination is copied, never written in place.
    let original = Tensor::<B, 2>::zeros([4, 4], &device);
    let shared = original.clone();
    let written = shared.slice_assign([1..2, 0..4], Tensor::<B, 2>::ones([1, 4], &device));
    let scattered = original.clone().scatter(
        1,
        Tensor::<B, 2, Int>::zeros([4, 1], &device),
        Tensor::<B, 2>::ones([4, 1], &device),
        IndexingUpdateOp::Add,
    );
    let scattered_nd = original.clone().scatter_nd(
        Tensor::<B, 2, Int>::from_ints([[1, 2], [3, 0]], &device),
        Tensor::<B, 1>::ones([2], &device),
        IndexingUpdateOp::Add,
    );
    let untouched: f32 = original.sum().into_scalar();
    let (written, scattered, scattered_nd): (f32, f32, f32) = (
        written.sum().into_scalar(),
        scattered.sum().into_scalar(),
        scattered_nd.sum().into_scalar(),
    );
    println!();
    println!(
        "copy-on-write: shared original sums to {untouched} (expected 0), writes sum to {written}, {scattered} and {scattered_nd} (expected 4, 4 and 2)"
    );
    assert_eq!(
        (untouched, written, scattered, scattered_nd),
        (0.0, 4.0, 4.0, 2.0)
    );
}
