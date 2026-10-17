//! Burn 0.22 / cubek-matmul 0.3 on Metal: matmul with a packed block-quantized
//! right-hand side panics or returns wrong values for some kernels.
//!
//! Part 1 of the 0.21 repro, ported: `x[m, k] · W` through the public
//! `Tensor::matmul` on Metal, with `W` packed as `QuantStore::PackedU32`,
//! symmetric Q8S/Q4S/Q2S, `[1, g]` blocks, and F32 or F16 activations. `W` is
//! either the transposed view of a packed `[n, k]` checkpoint matrix (blocks
//! along `k`, a column-major right-hand side) or packed `[k, n]` in place. The
//! reference is the host F64 product with the same packed bytes dequantized
//! independently on Flex. A product that panics counts as wrong.
//!   cargo run --release                     # no autotuning
//!   cargo run --release --features autotune # Burn's default for GPU backends
//! Without autotuning the products 0.21 got wrong now fail at launch (shared
//! memory budget exceeded, invalid vector size). With autotuning the selected
//! candidate varies between runs and is occasionally wrong.

use burn::tensor::quantization::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype};
use burn::tensor::{DType, Device, DeviceKind, Distribution, Tensor};

/// Symmetric packed `value` with `group`-element blocks along the last axis.
fn scheme(value: QuantValue, group: u8) -> QuantScheme {
    QuantScheme::default()
        .with_value(value)
        .with_store(QuantStore::PackedU32(0))
        .per_block([1, group], ScaleDtype::F32)
        .with_mode(QuantMode::Symmetric)
}

/// A packed `W[k, n]` on Metal and the host values of its dequantization.
fn packed_weight(s: &QuantScheme, k: usize, n: usize, transposed: bool) -> (Tensor<2>, Vec<f32>) {
    let metal = Device::metal(DeviceKind::DefaultDevice);
    let stored = if transposed { [n, k] } else { [k, n] };
    let packed = Tensor::<2>::random(stored, Distribution::Default, &metal).quantize_dynamic(s);
    let reference =
        Tensor::<2>::from_data(packed.clone().into_data(), &Device::flex()).dequantize();
    let reference = if transposed {
        reference.transpose()
    } else {
        reference
    };
    let values = reference.into_data().try_to_vec::<f32>().unwrap();
    (
        if transposed {
            packed.transpose()
        } else {
            packed
        },
        values,
    )
}

/// Host F64 product `x · w` of row-major `[m, k]` and `[k, n]` values.
fn host_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let a = x[i * k + p] as f64;
            for j in 0..n {
                out[i * n + j] += a * w[p * n + j] as f64;
            }
        }
    }
    out
}

/// Largest error relative to the largest reference magnitude; non-finite values count as infinite.
fn relative_error(values: &[f32], reference: &[f64]) -> f64 {
    let scale = reference.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    values.iter().zip(reference).fold(0.0f64, |a, (v, r)| {
        if v.is_finite() {
            a.max((*v as f64 - r).abs())
        } else {
            f64::INFINITY
        }
    }) / scale
}

/// One row of products for each `m`; returns the cells and the count above `wrong_above`.
fn row(
    w: &Tensor<2>,
    wv: &[f32],
    activation: DType,
    rows: &[usize],
    wrong_above: f64,
) -> (String, usize) {
    let [k, n] = w.dims();
    let metal = Device::metal(DeviceKind::DefaultDevice);
    let mut wrong = 0;
    let cells: Vec<String> = rows
        .iter()
        .map(|&m| {
            let cell = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let x = Tensor::<2>::random([m, k], Distribution::Default, &metal)
                    .cast(DType::F16)
                    .cast(activation);
                let xv = x
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap();
                let reference = host_matmul(&xv, wv, m, k, n);
                let _ = x.clone().matmul(w.clone()).into_data();
                let values = x
                    .matmul(w.clone())
                    .cast(DType::F32)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap();
                relative_error(&values, &reference)
            }));
            match cell {
                Ok(error) => {
                    wrong += (error > wrong_above) as usize;
                    format!("m={m}: {error:.1e}")
                }
                Err(_) => {
                    wrong += 1;
                    format!("m={m}: panic")
                }
            }
        })
        .collect();
    (cells.join(", "), wrong)
}

/// Prints the public-matmul table.
fn main() {
    std::panic::set_hook(Box::new(|info| {
        /// Panic messages already printed, so each distinct one appears once.
        static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        let message = info
            .payload()
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| info.payload().downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        let first = message
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(300)
            .collect::<String>();
        let mut seen = SEEN.lock().unwrap();
        if !seen.contains(&first) {
            println!("      panic: {first}");
            seen.push(first);
        }
    }));
    let (k, n) = (1024, 1024);
    println!(
        "Tensor::matmul on Metal (burn 0.22.0), CubeCL autotuning {}; max relative error vs the host reference",
        if cfg!(feature = "autotune") {
            "on"
        } else {
            "off"
        }
    );
    println!("  (wrong: above 1e-4 with F32 activations, above 1e-2 with F16 activations)");
    let rows = [1usize, 2, 4, 8, 32, 64];
    let mut wrong = [0usize; 2];
    let mut total = [0usize; 2];
    for (transposed, group) in [(true, 64u8), (false, 32u8)] {
        let layout = if transposed {
            "col-major, 64-element groups"
        } else {
            "row-major, 32-element groups"
        };
        println!("  W[{k}, {n}] {layout}:");
        for value in [QuantValue::Q8S, QuantValue::Q4S, QuantValue::Q2S] {
            let (w, wv) = packed_weight(&scheme(value, group), k, n, transposed);
            for (index, (activation, limit)) in [(DType::F32, 1e-4), (DType::F16, 1e-2)]
                .into_iter()
                .enumerate()
            {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    row(&w, &wv, activation, &rows, limit)
                }));
                match result {
                    Ok((cells, bad)) => {
                        wrong[index] += bad;
                        total[index] += rows.len();
                        println!("    {:<3} x {activation:?}: {cells}", format!("{value:?}"));
                    }
                    Err(_) => {
                        println!("    {:<3} x {activation:?}: panicked", format!("{value:?}"))
                    }
                }
            }
        }
    }
    println!(
        "  {} of {} F32-activation and {} of {} F16-activation products are wrong",
        wrong[0], total[0], wrong[1], total[1]
    );
    println!("  Narrow output, W[{k}, 16] col-major, 64-element groups, F32 activations:");
    for value in [QuantValue::Q4S, QuantValue::Q2S] {
        let (w, wv) = packed_weight(&scheme(value, 64), k, 16, true);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            row(&w, &wv, DType::F32, &[1, 2, 8, 32, 33, 64], 1e-4)
        })) {
            Ok((cells, _)) => println!("    {:<3}: {cells}", format!("{value:?}")),
            Err(_) => println!("    {:<3}: panicked", format!("{value:?}")),
        }
    }
}
