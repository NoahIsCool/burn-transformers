//! Burn 0.21 / cubek-matmul 0.2 on Metal: matmul with a packed block-quantized
//! right-hand side returns wrong values for some kernels.
//!
//! `x[m, k] · W` with `W` packed as `QuantStore::PackedU32`, symmetric
//! `Q8S`/`Q4S`/`Q2S`, `QuantLevel::block([1, g])`. The reference is the host
//! F64 product with the same packed bytes dequantized independently on Flex
//! (and, for F16 activations, the F16-rounded activations).
//!
//! Part 1 goes through Burn's public `Tensor::matmul` on the Metal backend
//! with F32 and F16 activations: `W` is either the transposed view of a packed
//! `[n, k]` checkpoint matrix (blocks along `k`, a column-major right-hand
//! side, as a `LinearLayout::Col` weight loaded from a checkpoint) or packed
//! `[k, n]` in place. A narrow `W[1024, 16]` shows that no row threshold is
//! safe. Run it twice:
//!   cargo run --release                     # no autotuning (Strategy::Auto)
//!   cargo run --release --features autotune # Burn's default for GPU backends
//! Part 2 calls each cubek-matmul strategy explicitly, with the operands bound
//! exactly as `burn_cubecl::kernel::matmul::launch_matmul` binds them, to show
//! which kernels are wrong. Strategies that decline a problem are omitted.

use burn::backend::wgpu::{CubeBackend, WgpuRuntime};
use burn::backend::{Flex, Metal};
use burn::tensor::quantization::{
    QTensorPrimitive, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
use burn::tensor::{DType, Distribution, Tensor, TensorPrimitive, backend::Backend};
use cubek::matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek::matmul::launch::{Strategy, launch_ref};
use cubek::matmul::routines::BlueprintStrategy;
use cubek::std::InputBinding;

/// The Metal backend without fusion, whose primitives are `CubeTensor`s.
type Raw = CubeBackend<WgpuRuntime, f32, i32, u8>;

/// Symmetric packed `value` with `group`-element blocks along the last axis.
fn scheme(value: QuantValue, group: u8) -> QuantScheme {
    QuantScheme::default()
        .with_value(value)
        .with_param(QuantParam::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_level(QuantLevel::block([1, group]))
        .with_mode(QuantMode::Symmetric)
}

/// A packed weight `W[k, n]` on backend `B` and the host values of its dequantization.
///
/// `transposed` builds `W` as the transposed view of a packed `[n, k]` matrix
/// whose blocks run along `k`; otherwise `W` is packed `[k, n]` with blocks
/// along `n`. The reference values are dequantized on Flex from the same
/// packed bytes, before any transpose.
fn packed_weight<B: Backend>(
    scheme: &QuantScheme,
    k: usize,
    n: usize,
    transposed: bool,
) -> (Tensor<B, 2>, Vec<f32>) {
    let device = Default::default();
    let stored = if transposed { [n, k] } else { [k, n] };
    let packed =
        Tensor::<B, 2>::random(stored, Distribution::Default, &device).quantize_dynamic(scheme);
    let reference =
        Tensor::<Flex, 2>::from_data(packed.clone().into_data(), &Default::default()).dequantize();
    let reference = if transposed {
        reference.transpose()
    } else {
        reference
    };
    let values = reference.into_data().to_vec().unwrap();
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

/// Largest error of `values` relative to the largest reference magnitude.
///
/// A non-finite value counts as an infinite error (`f64::max` would skip NaN).
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

/// `x · W` with one explicit cubek strategy and a packed `W`, bound as
/// `burn_cubecl::kernel::matmul::launch_matmul` binds them.
fn run_strategy(
    strategy: &Strategy,
    x: Tensor<Raw, 2>,
    w: Tensor<Raw, 2>,
) -> Result<Vec<f32>, String> {
    let [m, _] = x.dims();
    let [_, n] = w.dims();
    let out = Tensor::<Raw, 2>::zeros([m, n], &x.device())
        .into_primitive()
        .tensor();
    let lhs = x.into_primitive().tensor();
    let TensorPrimitive::QFloat(rhs) = w.into_primitive() else {
        unreachable!("packed weight")
    };
    let scheme = *rhs.scheme();
    let (data, scales) = rhs.quantized_handles().expect("packed tensor");
    let (data_dtype, scale_dtype) = (data.dtype, scales.dtype);
    let rhs = InputBinding::quantized(
        data.binding(),
        scales.binding(),
        rhs.meta.shape().clone(),
        scheme,
        data_dtype.into(),
        scale_dtype.into(),
    );
    let lhs = InputBinding::new(lhs.binding(), DType::F32.into());
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: DType::F32.into(),
        rhs: DType::F32.into(),
        out: DType::F32.into(),
    });
    let client = out.client.clone();
    launch_ref(
        strategy,
        &client,
        lhs,
        rhs,
        out.clone().binding(),
        &mut dtypes,
    )
    .map_err(|error| {
        format!("{error:?}")
            .lines()
            .next()
            .unwrap_or("")
            .to_string()
    })?;
    Ok(
        Tensor::<Raw, 2>::from_primitive(TensorPrimitive::Float(out))
            .into_data()
            .to_vec()
            .unwrap(),
    )
}

/// The cubek-matmul strategies the autotuner and `Strategy::Auto` can run for these shapes.
fn strategies() -> Vec<(&'static str, Strategy)> {
    vec![
        ("Auto", Strategy::Auto),
        (
            "SimpleUnit",
            Strategy::SimpleUnit(BlueprintStrategy::Inferred(Default::default())),
        ),
        (
            "DoubleUnit",
            Strategy::DoubleUnit(BlueprintStrategy::Inferred(Default::default())),
        ),
        (
            "SimpleVecMat",
            Strategy::SimpleVecMat(BlueprintStrategy::Inferred(().into())),
        ),
        (
            "DoubleVecMat",
            Strategy::DoubleVecMat(BlueprintStrategy::Inferred(().into())),
        ),
        (
            "GemvUnitPerpendicular",
            Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(Default::default())),
        ),
        (
            "GemvPlaneParallel",
            Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(Default::default())),
        ),
    ]
}

/// One Part 1 row: `x[m, k] · W` through the public `matmul` for each `m`, with
/// `activation` (F32 or F16) left-hand values; returns the cells and the number
/// of products whose error exceeds `wrong_above`.
fn part1_row(
    w: &Tensor<Metal, 2>,
    w_values: &[f32],
    activation: DType,
    rows: &[usize],
    wrong_above: f64,
) -> (String, usize) {
    let [k, n] = w.dims();
    let mut wrong = 0;
    let cells: Vec<String> = rows
        .iter()
        .map(|&m| {
            // F16-representable activations, so both dtypes share one reference.
            let x = Tensor::<Metal, 2>::random([m, k], Distribution::Default, &Default::default())
                .cast(DType::F16)
                .cast(activation);
            let x_values: Vec<f32> = x.clone().cast(DType::F32).into_data().to_vec().unwrap();
            let reference = host_matmul(&x_values, w_values, m, k, n);
            // The first call autotunes when autotuning is on; report the second.
            let _ = x.clone().matmul(w.clone()).into_data();
            let values: Vec<f32> = x
                .matmul(w.clone())
                .cast(DType::F32)
                .into_data()
                .to_vec()
                .unwrap();
            let error = relative_error(&values, &reference);
            wrong += (error > wrong_above) as usize;
            format!("m={m}: {error:.1e}")
        })
        .collect();
    (cells.join(", "), wrong)
}

/// Prints Part 1 (public `matmul`) and Part 2 (explicit strategies).
fn main() {
    // Autotuning's own kernel-expansion failures panic inside caught tasks; keep them quiet.
    std::panic::set_hook(Box::new(|_| {}));
    let (k, n) = (1024, 1024);

    println!(
        "Part 1: Tensor::matmul on Metal, CubeCL autotuning {}; max relative error vs the host reference",
        if cfg!(feature = "autotune") {
            "on"
        } else {
            "off"
        }
    );
    println!("  (wrong: above 1e-4 with F32 activations, above 1e-2 with F16 activations)");
    let values = [QuantValue::Q8S, QuantValue::Q4S, QuantValue::Q2S];
    let rows = [1usize, 2, 4, 8, 32, 64];
    let mut wrong = [0usize; 2];
    let mut total = [0usize; 2];
    // (column-major blocks along k, group 64): a column-layout checkpoint weight;
    // (row-major blocks along n, group 32): a weight quantized in place as [k, n].
    for (transposed, group) in [(true, 64u8), (false, 32u8)] {
        let layout = if transposed {
            "col-major, 64-element groups"
        } else {
            "row-major, 32-element groups"
        };
        println!("  W[{k}, {n}] {layout}:");
        for value in values {
            let (w, w_values) = packed_weight::<Metal>(&scheme(value, group), k, n, transposed);
            for (index, (activation, limit)) in [(DType::F32, 1e-4), (DType::F16, 1e-2)]
                .into_iter()
                .enumerate()
            {
                let (cells, bad) = part1_row(&w, &w_values, activation, &rows, limit);
                wrong[index] += bad;
                total[index] += rows.len();
                println!("    {:<3} x {activation:?}: {cells}", format!("{value:?}"));
            }
        }
    }
    println!(
        "  {} of {} F32-activation and {} of {} F16-activation products are wrong",
        wrong[0], total[0], wrong[1], total[1]
    );
    println!("  Narrow output, W[{k}, 16] col-major, 64-element groups, F32 activations:");
    for value in [QuantValue::Q4S, QuantValue::Q2S] {
        let (w, w_values) = packed_weight::<Metal>(&scheme(value, 64), k, 16, true);
        let (cells, _) = part1_row(&w, &w_values, DType::F32, &[1, 2, 8, 32, 33, 64], 1e-4);
        println!("    {:<3}: {cells}", format!("{value:?}"));
    }
    for value in [QuantValue::Q8S, QuantValue::Q4S] {
        let (w, w_values) = packed_weight::<Metal>(&scheme(value, 64), k, n, true);
        let values: Vec<f32> = w.dequantize().into_data().to_vec().unwrap();
        let reference: Vec<f64> = w_values.iter().map(|v| *v as f64).collect();
        println!(
            "  dequantize() of the col-major (transposed) packed {value:?} view: max relative error {:.1e}",
            relative_error(&values, &reference)
        );
    }

    println!();
    println!("Part 2: explicit cubek-matmul strategies, max relative error vs the host reference:");
    for value in [QuantValue::Q8S, QuantValue::Q4S, QuantValue::Q2S] {
        for transposed in [false, true] {
            let layout = if transposed { "col-major" } else { "row-major" };
            let (w, w_values) = packed_weight::<Raw>(&scheme(value, 32), k, n, transposed);
            for m in [1usize, 4, 64] {
                let x =
                    Tensor::<Raw, 2>::random([m, k], Distribution::Default, &Default::default());
                let x_values: Vec<f32> = x.clone().into_data().to_vec().unwrap();
                let reference = host_matmul(&x_values, &w_values, m, k, n);
                let cells: Vec<String> = strategies()
                    .into_iter()
                    .filter_map(|(name, strategy)| {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            run_strategy(&strategy, x.clone(), w.clone())
                        }));
                        match result {
                            Ok(Ok(values)) => Some(format!(
                                "{name} {:.0e}",
                                relative_error(&values, &reference)
                            )),
                            Ok(Err(_)) => None,
                            Err(_) => Some(format!("{name} panicked")),
                        }
                    })
                    .collect();
                println!(
                    "  {:<4} {layout} m={m:<3} {}",
                    format!("{value:?}"),
                    cells.join(", ")
                );
            }
        }
    }
}
