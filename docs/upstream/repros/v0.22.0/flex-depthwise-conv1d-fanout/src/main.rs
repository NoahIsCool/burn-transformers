//! Burn Flex 0.22 (unchanged since 0.21): the depthwise `conv1d` kernel fans every `(batch, channel)`
//! plane out to rayon, however little work each plane holds.
//!
//! A recurrent mixer's one-token decode step convolves `[1, channels, kernel]`
//! with a depthwise `[channels, 1, kernel]` filter: one output column,
//! `channels * kernel` multiply-adds (24,576 here). Flex nevertheless spawns
//! one rayon task per channel. Idle, that already costs more than the
//! arithmetic; when other threads keep the cores busy (a data loader, a second
//! model, a test harness), the calling thread waits in rayon's latch for
//! descheduled workers and the step time explodes.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use burn::tensor::{Distribution, Tensor, module::conv1d, ops::ConvOptions};

/// Channels of the depthwise convolution.
const CHANNELS: usize = 6144;
/// Taps of the causal convolution.
const KERNEL: usize = 4;

/// Median and 99th percentile of `samples` timings of `body`, in microseconds.
fn percentiles_us(samples: usize, mut body: impl FnMut()) -> (f64, f64) {
    let mut times: Vec<f64> = (0..samples)
        .map(|_| {
            let start = Instant::now();
            body();
            start.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    times.sort_by(f64::total_cmp);
    (times[samples / 2], times[samples * 99 / 100])
}

/// Times the one-column convolution with idle and with busy cores.
fn main() {
    let device = burn::tensor::Device::flex();
    let input = Tensor::<3>::random([1, CHANNELS, KERNEL], Distribution::Default, &device);
    let weight = Tensor::<3>::random([CHANNELS, 1, KERNEL], Distribution::Default, &device);
    let bias = Tensor::<1>::random([CHANNELS], Distribution::Default, &device);
    let options = ConvOptions::new([1], [0], [1], CHANNELS);

    // Host reference: the same single output column as a plain sequential loop.
    let x: Vec<f32> = input.to_data().try_to_vec().unwrap();
    let w: Vec<f32> = weight.to_data().try_to_vec().unwrap();
    let b: Vec<f32> = bias.to_data().try_to_vec().unwrap();
    let reference = |out: &mut [f32]| {
        for c in 0..CHANNELS {
            let mut acc = 0.0f32;
            for k in 0..KERNEL {
                acc += w[c * KERNEL + k] * x[c * KERNEL + k];
            }
            out[c] = acc + b[c];
        }
    };
    let mut expected = vec![0.0f32; CHANNELS];
    reference(&mut expected);
    let got: Vec<f32> = conv1d(
        input.clone(),
        weight.clone(),
        Some(bias.clone()),
        options.clone(),
    )
    .into_data()
    .try_to_vec()
    .unwrap();
    assert_eq!(
        got, expected,
        "Flex result differs from the sequential loop"
    );

    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    println!(
        "depthwise conv1d, one output column: [1, {CHANNELS}, {KERNEL}] * [{CHANNELS}, 1, {KERNEL}] ({} multiply-adds), {threads} hardware threads",
        CHANNELS * KERNEL
    );
    println!("{:<44} {:>12} {:>12}", "", "median us", "p99 us");
    for contended in [false, true] {
        let stop = Arc::new(AtomicBool::new(false));
        let spinners: Vec<_> = (0..if contended { threads } else { 0 })
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut n = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        n = n.wrapping_mul(6364136223846793005).wrapping_add(1);
                        std::hint::black_box(n);
                    }
                })
            })
            .collect();
        let label = if contended {
            "busy cores"
        } else {
            "idle cores"
        };
        let mut out = vec![0.0f32; CHANNELS];
        let (median, p99) = percentiles_us(2001, || {
            reference(&mut out);
            std::hint::black_box(&out);
        });
        println!(
            "{:<44} {median:>12.1} {p99:>12.1}",
            format!("sequential host loop ({label})")
        );
        let (median, p99) = percentiles_us(2001, || {
            let y = conv1d(
                input.clone(),
                weight.clone(),
                Some(bias.clone()),
                options.clone(),
            );
            std::hint::black_box(y);
        });
        println!(
            "{:<44} {median:>12.1} {p99:>12.1}",
            format!("burn-flex conv1d ({label})")
        );
        stop.store(true, Ordering::Relaxed);
        for spinner in spinners {
            spinner.join().unwrap();
        }
    }
}
