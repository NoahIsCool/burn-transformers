//! Opt-in micro-benchmark of projection weight storage order on one backend.
//!
//! A checkpoint stores a projection as `C` with shape `[output, input]`, while
//! [`transformers::operations::linear`] multiplies `x · W` with the logical
//! `[input, output]` weight `W = Cᵀ`. Burn's `LinearLayout::Col` mapper produces
//! `W` as a *transposed view* of the checkpoint buffer, and a tied language-model
//! head multiplies by the transposed view of the `[vocabulary, hidden]` embedding
//! table in the same way. For each shape this harness times four computations of
//! the identical logical product:
//!
//! - `strided`: a column-layout `Linear` loaded through Burn's plain record path,
//!   whose runtime weight is a transposed view (the behaviour before the crate's
//!   installer chose storage order, and the tied-head product);
//! - `contiguous`: a row-layout `Linear` holding a contiguous `[input, output]`
//!   buffer;
//! - `installed`: a column-layout `Linear` filled by
//!   [`transformers::checkpoint::install_float_parameter`], i.e. what every family
//!   loader produces on this backend;
//! - `checkpoint_lhs`: `(C · xᵀ)ᵀ`, which keeps the contiguous checkpoint buffer as
//!   the *left* operand and returns a transposed view of the `[output, rows]`
//!   product (diagnostic only; no model uses it).
//!
//! Each sample times `repetitions` back-to-back products followed by one
//! `Backend::sync`, so asynchronous backends are measured to completion and launch
//! and synchronization latency is amortized; accelerators need far more
//! repetitions than the synchronous CPU backend for that amortization. Variants
//! are interleaved within every sample so that host load drift affects them
//! alike. Before timing, every variant's output is checked against `contiguous`
//! with a tolerance of `1e-5` (F32) or `1e-2` (half precision) relative to the
//! largest reference magnitude, which admits only matmul reassociation.
//!
//! Each report's `provenance` field records the compiled harness sources,
//! executable, toolchain, features, build tree and host load (see
//! `benchmark_identity.rs`); `TRANSFORMERS_BENCH_SOURCE_LABEL` names the source
//! revision of a prebuilt binary.

use std::time::Instant;

use burn::{
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{DType, Tensor, TensorData, backend::Backend},
};
use serde_json::{Value, json};
use transformers::{checkpoint::install_float_parameter, operations::linear};

use super::benchmark_identity::{host_load, source_identity};

/// Untimed calls per variant and shape before sampling.
const WARMUP_CALLS: usize = 3;
/// Interleaved samples per variant and shape.
const SAMPLES: usize = 15;
/// Variant names in report order; see the module documentation.
const VARIANTS: [&str; 4] = ["strided", "contiguous", "installed", "checkpoint_lhs"];

/// One projection geometry taken from a supported checkpoint family.
struct ProjectionShape {
    /// Human-readable source of the geometry.
    name: &'static str,
    /// Input width `I` (the contracted axis).
    input: usize,
    /// Output width `O`.
    output: usize,
    /// Activation row counts timed for this shape.
    rows: &'static [usize],
}

/// Single-token decode, an 8-row batched decode, and two prefill widths.
const PROJECTION_ROWS: &[usize] = &[1, 8, 128, 512];
/// Decode-shaped rows only, for the large vocabulary heads.
const HEAD_ROWS: &[usize] = &[1, 8];

/// Qwen3.5-0.8B and Llama-style projection geometries plus a tied vocabulary head.
const SHAPES: [ProjectionShape; 6] = [
    ProjectionShape {
        name: "qwen3.5-0.8b gated q_proj 1024->4096",
        input: 1024,
        output: 4096,
        rows: PROJECTION_ROWS,
    },
    ProjectionShape {
        name: "square attention 1024->1024",
        input: 1024,
        output: 1024,
        rows: PROJECTION_ROWS,
    },
    ProjectionShape {
        name: "qwen3.5-0.8b mlp up 1024->3584",
        input: 1024,
        output: 3584,
        rows: PROJECTION_ROWS,
    },
    ProjectionShape {
        name: "qwen3.5-0.8b mlp down 3584->1024",
        input: 3584,
        output: 1024,
        rows: PROJECTION_ROWS,
    },
    ProjectionShape {
        name: "qwen3.5-0.8b o_proj 2048->1024",
        input: 2048,
        output: 1024,
        rows: PROJECTION_ROWS,
    },
    ProjectionShape {
        name: "smollm-135m tied head 576->49152",
        input: 576,
        output: 49_152,
        rows: HEAD_ROWS,
    },
];

/// The runtime operands of every variant for one shape.
struct Operands<B: Backend> {
    /// Column-layout projection with a strided runtime weight.
    strided: Linear<B>,
    /// Row-layout projection with contiguous `[input, output]` storage.
    contiguous: Linear<B>,
    /// Column-layout projection filled by the crate's installer.
    installed: Linear<B>,
    /// The contiguous `[output, input]` checkpoint matrix `C`.
    checkpoint: Tensor<B, 2>,
}

impl<B: Backend> Operands<B> {
    /// Computes variant `index` (in [`VARIANTS`] order) for activations `x`.
    fn product(&self, index: usize, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match index {
            0 => linear(x, &self.strided).expect("strided projection"),
            1 => linear(x, &self.contiguous).expect("contiguous projection"),
            2 => linear(x, &self.installed).expect("installed projection"),
            _ => self.checkpoint.clone().matmul(x.transpose()).transpose(),
        }
    }
}

/// Deterministic, sign-varying values with no repeated structure along either axis.
fn pattern(count: usize, salt: f32) -> Vec<f32> {
    (0..count)
        .map(|index| (index as f32 * 0.618_034 + salt).sin() * 0.05)
        .collect()
}

/// Transposes a row-major `[rows, columns]` host matrix.
fn transpose_host(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            transposed[column * rows + row] = values[row * columns + column];
        }
    }
    transposed
}

/// Builds every variant's operands from one deterministic checkpoint matrix.
fn operands<B: Backend>(shape: &ProjectionShape, dtype: DType, device: &B::Device) -> Operands<B> {
    let values = pattern(shape.output * shape.input, 0.25);
    let checkpoint_data =
        || TensorData::new(values.clone(), [shape.output, shape.input]).convert_dtype(dtype);
    let column = LinearConfig::new(shape.input, shape.output)
        .with_bias(false)
        .with_layout(LinearLayout::Col);

    let checkpoint = Tensor::<B, 2>::from_data(checkpoint_data(), (device, dtype));
    let mut strided = column.init::<B>(device);
    strided.weight = strided
        .weight
        .clone()
        .load_record(Param::initialized(ParamId::new(), checkpoint.clone()));

    let mut contiguous = LinearConfig::new(shape.input, shape.output)
        .with_bias(false)
        .init::<B>(device);
    let row_major = transpose_host(&values, shape.output, shape.input);
    contiguous.weight = Param::from_tensor(Tensor::<B, 2>::from_data(
        TensorData::new(row_major, [shape.input, shape.output]).convert_dtype(dtype),
        (device, dtype),
    ));

    let mut installed = column.init::<B>(device);
    install_float_parameter(&mut installed.weight, checkpoint_data())
        .expect("install a checkpoint-shaped column-layout weight");

    Operands {
        strided,
        contiguous,
        installed,
        checkpoint,
    }
}

/// Median, minimum and maximum of microsecond samples.
fn summary(mut samples: Vec<f64>) -> Value {
    samples.sort_by(f64::total_cmp);
    let median = if samples.len() % 2 == 1 {
        samples[samples.len() / 2]
    } else {
        (samples[samples.len() / 2 - 1] + samples[samples.len() / 2]) / 2.0
    };
    json!({
        "median_us": median,
        "min_us": samples[0],
        "max_us": samples[samples.len() - 1],
    })
}

/// Checks a variant against the contiguous reference before any timing.
fn assert_matches(name: &str, actual: Tensor<impl Backend, 2>, reference: &[f32], dtype: DType) {
    let actual = actual
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .expect("F32 projection output");
    assert_eq!(actual.len(), reference.len(), "{name} output length");
    let tolerance = if dtype == DType::F32 { 1e-5 } else { 1e-2 };
    let scale = reference
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    for (index, (actual, expected)) in actual.iter().zip(reference).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance * scale.max(1.0),
            "{name} output {index}: {actual} vs {expected}"
        );
    }
}

/// Runs every shape and row count and returns a JSON report for one backend and dtype.
///
/// `repetitions` back-to-back products form one sample, followed by one
/// synchronization; the reported time is the sample divided by `repetitions`.
pub fn run<B: Backend>(
    device: &B::Device,
    dtype: DType,
    backend_profile: &str,
    repetitions: usize,
) -> Value {
    let provenance = source_identity(&[
        (
            "tests/small_model_benchmarks.rs",
            include_bytes!("../small_model_benchmarks.rs"),
        ),
        (
            "tests/support/linear_layout_benchmark.rs",
            include_bytes!("linear_layout_benchmark.rs"),
        ),
        (
            "tests/support/benchmark_identity.rs",
            include_bytes!("benchmark_identity.rs"),
        ),
    ]);
    let mut results = Vec::new();
    for shape in &SHAPES {
        let operands = operands::<B>(shape, dtype, device);
        for &rows in shape.rows {
            let input = Tensor::<B, 2>::from_data(
                TensorData::new(pattern(rows * shape.input, 1.5), [rows, shape.input])
                    .convert_dtype(dtype),
                (device, dtype),
            );
            let reference = operands
                .product(1, input.clone())
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()
                .expect("F32 reference");
            for (index, name) in VARIANTS.iter().enumerate() {
                assert_matches(
                    name,
                    operands.product(index, input.clone()),
                    &reference,
                    dtype,
                );
                for _ in 0..WARMUP_CALLS {
                    drop(operands.product(index, input.clone()));
                }
            }
            B::sync(device).expect("synchronize after warmup");
            let mut samples = vec![Vec::with_capacity(SAMPLES); VARIANTS.len()];
            for _ in 0..SAMPLES {
                for (index, slot) in samples.iter_mut().enumerate() {
                    let started = Instant::now();
                    for _ in 0..repetitions {
                        drop(operands.product(index, input.clone()));
                    }
                    B::sync(device).expect("synchronize timed products");
                    slot.push(started.elapsed().as_secs_f64() * 1e6 / repetitions as f64);
                }
            }
            let timings = VARIANTS
                .iter()
                .zip(samples)
                .map(|(name, samples)| ((*name).to_owned(), summary(samples)))
                .collect::<serde_json::Map<_, _>>();
            results.push(json!({
                "shape": shape.name,
                "input": shape.input,
                "output": shape.output,
                "rows": rows,
                "per_call": timings,
            }));
        }
    }
    json!({
        "benchmark": "linear weight storage order",
        "provenance": provenance,
        "load_average_after": host_load(),
        "backend_profile": backend_profile,
        "backend": B::name(device),
        "dtype": format!("{dtype:?}"),
        "warmup_calls": WARMUP_CALLS,
        "samples": SAMPLES,
        "repetitions_per_sample": repetitions,
        "results": results,
    })
}
