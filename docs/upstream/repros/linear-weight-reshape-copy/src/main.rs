//! Burn 0.21 `module::linear` reshapes its weight `[in, out]` to
//! `[1, .., 1, in, out]` before a batched matmul. On Flex, a reshape of a
//! non-contiguous tensor copies it, so a weight stored as a transposed view
//! (a tied embedding readout `h · Eᵀ`, any `LinearLayout::Col` weight loaded
//! from an `[out, in]` checkpoint) is copied in full on every call.
//!
//! The same product with the activation flattened to `[rows, in]` reads the
//! strided weight directly. The two agree to F32 rounding (the GEMM may
//! accumulate a strided and a contiguous operand in a different order).

use std::time::Instant;

use burn::backend::{Autodiff, Flex};
use burn::tensor::{Distribution, Tensor, module::linear};

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

/// Checks `module::linear` values and gradients against an explicit broadcast matmul.
fn gradient_parity() {
    /// Flex with autodiff, for the gradient check.
    type A = Autodiff<Flex>;
    let device = Default::default();
    let mut worst = 0.0f32;
    for transposed in [false, true] {
        let w = if transposed {
            Tensor::<A, 2>::random([6, 5], Distribution::Default, &device).transpose()
        } else {
            Tensor::<A, 2>::random([5, 6], Distribution::Default, &device)
        }
        .detach()
        .require_grad();
        let b = Tensor::<A, 1>::random([6], Distribution::Default, &device).require_grad();
        let x = Tensor::<A, 4>::random([2, 3, 4, 5], Distribution::Default, &device).require_grad();
        let r = Tensor::<A, 4>::random([2, 3, 4, 6], Distribution::Default, &device);
        let ours = (linear(x.clone(), w.clone(), Some(b.clone())) * r.clone())
            .sum()
            .backward();
        let explicit =
            ((x.clone().matmul(w.clone().unsqueeze::<4>()) + b.clone().unsqueeze::<4>()) * r)
                .sum()
                .backward();
        let diff = |a: Tensor<Flex, 4>, b: Tensor<Flex, 4>| {
            let (a, b): (Vec<f32>, Vec<f32>) = (
                a.into_data().to_vec().unwrap(),
                b.into_data().to_vec().unwrap(),
            );
            a.iter()
                .zip(&b)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        };
        worst = worst
            .max(diff(x.grad(&ours).unwrap(), x.grad(&explicit).unwrap()))
            .max(diff(
                w.grad(&ours).unwrap().unsqueeze(),
                w.grad(&explicit).unwrap().unsqueeze(),
            ))
            .max(diff(
                b.grad(&ours).unwrap().unsqueeze(),
                b.grad(&explicit).unwrap().unsqueeze(),
            ));
    }
    println!(
        "gradient parity (x, weight, bias; contiguous and transposed weight): max |difference| {worst:.2e}"
    );
    assert!(worst <= 1e-5);
}

/// Checks gradients, then times the decode readout three ways.
fn main() {
    gradient_parity();
    let device = Default::default();
    let (hidden, vocabulary) = (1024, 151_936);
    // A tied readout: the [vocabulary, hidden] embedding table used as Eᵀ.
    let table = Tensor::<B, 2>::random([vocabulary, hidden], Distribution::Default, &device);
    let weight = table.clone().transpose(); // [hidden, vocabulary], strided view
    let contiguous = Tensor::<B, 2>::from_data(weight.clone().into_data(), &device);

    println!(
        "decode readout [1, 1, {hidden}] x [{hidden}, {vocabulary}] F32 ({:.0} MiB weight), median us",
        (hidden * vocabulary * 4) as f64 / 1048576.0
    );
    for tokens in [1usize, 16] {
        let x = Tensor::<B, 3>::random([1, tokens, hidden], Distribution::Default, &device);
        let flattened = |w: &Tensor<B, 2>| {
            x.clone()
                .reshape([tokens, hidden])
                .matmul(w.clone())
                .reshape([1, tokens, vocabulary])
        };
        let reference: Vec<f32> = linear(x.clone(), contiguous.clone(), None)
            .into_data()
            .to_vec()
            .unwrap();
        let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for candidate in [linear(x.clone(), weight.clone(), None), flattened(&weight)] {
            let values: Vec<f32> = candidate.into_data().to_vec().unwrap();
            let error = values
                .iter()
                .zip(&reference)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                error <= 1e-5 * scale,
                "results differ by {error} (scale {scale})"
            );
        }
        let samples = 21;
        let linear_view = median_us(samples, || {
            std::hint::black_box(linear(x.clone(), weight.clone(), None));
        });
        let linear_contiguous = median_us(samples, || {
            std::hint::black_box(linear(x.clone(), contiguous.clone(), None));
        });
        let flat_view = median_us(samples, || {
            std::hint::black_box(flattened(&weight));
        });
        println!(
            "  {tokens:>2} token(s): module::linear, transposed view {linear_view:>9.0} | module::linear, contiguous {linear_contiguous:>9.0} | flattened, transposed view {flat_view:>9.0}"
        );
    }
}
