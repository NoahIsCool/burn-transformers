//! Burn Flex 0.21: F32 binary ops accelerate only two broadcast shapes.
//!
//! `binary_op_f32` has SIMD paths for (a) same-shape contiguous operands,
//! (b) one contiguous row shared by every leading position, and (c) one scalar
//! per row. Two common transformer broadcasts match neither and fall back to a
//! scalar `StridedIter` walk (after first copying a strided left operand):
//!
//! - RoPE: a `[b, h, s, d]` heads-swapped view times `[b, 1, s, d]` cos/sin
//!   tables broadcast over heads;
//! - an outer product `[b, h, 1, n] * [b, h, p, 1]` (a Mamba-2 state update).
//!
//! Each case is timed against the same arithmetic on operands that were
//! first materialized to the full shape (the "dense" column), which takes the
//! existing SIMD path. Results are checked to be bit-identical.

use std::time::Instant;

use burn::backend::Flex;
use burn::tensor::{Distribution, Tensor};

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
    times[samples / 2]
}

/// Times `lhs * rhs` with broadcasting and against dense copies of both operands.
fn case(label: &str, lhs: Tensor<B, 4>, rhs: Tensor<B, 4>) {
    let shape = (lhs.clone() * rhs.clone()).dims();
    // Dense operands: contiguous copies with the full broadcast shape.
    let dense_lhs = Tensor::<B, 4>::from_data(lhs.clone().expand(shape).into_data(), &lhs.device());
    let dense_rhs = Tensor::<B, 4>::from_data(rhs.clone().expand(shape).into_data(), &rhs.device());
    let broadcast = (lhs.clone() * rhs.clone()).into_data();
    let dense = (dense_lhs.clone() * dense_rhs.clone()).into_data();
    assert_eq!(broadcast, dense, "{label}: results differ");

    let samples = 51;
    let us_broadcast = median_us(samples, || {
        std::hint::black_box(lhs.clone() * rhs.clone());
    });
    let us_dense = median_us(samples, || {
        std::hint::black_box(dense_lhs.clone() * dense_rhs.clone());
    });
    println!(
        "{label:<52} {:>10.1} {:>10.1} {:>8.1}x",
        us_broadcast,
        us_dense,
        us_broadcast / us_dense
    );
}

/// Times the RoPE and outer-product broadcasts against dense operands.
fn main() {
    let device = Default::default();
    println!(
        "{:<52} {:>10} {:>10} {:>9}",
        "F32 multiply, median us", "broadcast", "dense", "ratio"
    );

    // RoPE: queries projected as [b, s, h, d] and viewed as [b, h, s, d].
    let (b, s, h, d) = (1, 512, 32, 128);
    let q = Tensor::<B, 4>::random([b, s, h, d], Distribution::Default, &device).swap_dims(1, 2);
    let cos = Tensor::<B, 4>::random([b, 1, s, d], Distribution::Default, &device);
    case(
        "RoPE  [1,32,512,128] (swapped view) * [1,1,512,128]",
        q,
        cos,
    );

    // The same RoPE table on a contiguous [b, h, s, d] operand.
    let q = Tensor::<B, 4>::random([b, h, s, d], Distribution::Default, &device);
    let cos = Tensor::<B, 4>::random([b, 1, s, d], Distribution::Default, &device);
    case(
        "RoPE  [1,32,512,128] (contiguous)   * [1,1,512,128]",
        q,
        cos,
    );

    // Outer product: [b, h, 1, n] * [b, h, p, 1].
    let (b, h, p, n) = (1, 64, 64, 128);
    let x = Tensor::<B, 4>::random([b, h, 1, n], Distribution::Default, &device);
    let y = Tensor::<B, 4>::random([b, h, p, 1], Distribution::Default, &device);
    case("outer [1,64,1,128] * [1,64,64,1]", x, y);
}
