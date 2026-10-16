//! Opt-in warm prefill and cached-decode timing for any local `AutoCausalLm` checkpoint.
//!
//! The runner loads the checkpoint strictly with F32 casts (exact for BF16 and
//! F16 sources) and times three shapes with the repository protocol: three
//! unmeasured warmups and ten measured repetitions, each on fresh cache state,
//! with 128 teacher-forced cached decode steps so EOS cannot change the
//! denominator.
//!
//! | Shape | Rows | Valid source tokens per row |
//! | --- | ---: | ---: |
//! | `batch1_source32` | 1 | 32 |
//! | `batch1_source128` | 1 | 128 |
//! | `batch8_source32` | 8 | 32 |
//!
//! Source and forced tokens are deterministic functions of their position, so
//! repeated runs and different source revisions execute identical work. Every
//! timed phase is bracketed by `Backend::sync`; decode is model-only (no host
//! token selection), and each repetition's final logits are reduced to one host
//! checksum outside the timed interval so silent graph changes between runs are
//! visible in the report. Prefill throughput counts valid tokens (`rows ·
//! source`); decode throughput counts generated tokens (`rows · 128`).
//!
//! Because the checkpoint carries no committed acceptance record, the report
//! identifies it by the SHA-256 of `config.json` and of every `*.safetensors`
//! file instead of authenticating it against an allowlist. Its `provenance`
//! field records the compiled harness sources, executable, toolchain, features,
//! build tree and host load (see `benchmark_identity.rs`); set
//! `TRANSFORMERS_BENCH_SOURCE_LABEL` to name the source revision of a prebuilt
//! binary and `TRANSFORMERS_BENCH_METADATA_JSON` to attach an operator
//! `RunEnvironment` record.

use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use serde_json::{Value, json};
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, DecodeInput, ForwardOptions,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
};

use super::benchmark_identity::{file_sha256, host_load, source_identity};
use super::benchmark_metrics::{DECODE_STEPS, MEASURED_REPETITIONS, WARMUP_REPETITIONS, summarize};

/// Timed `(label, rows, source tokens per row)` shapes; see the module documentation.
const SHAPES: [(&str, usize, usize); 3] = [
    ("batch1_source32", 1, 32),
    ("batch1_source128", 1, 128),
    ("batch8_source32", 8, 32),
];

/// Hashes `config.json` and every SafeTensors shard in name order.
fn identity(root: &Path) -> Value {
    let mut names = fs::read_dir(root)
        .expect("list checkpoint directory")
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name == "config.json" || name.ends_with(".safetensors"))
        .collect::<Vec<_>>();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let hash = file_sha256(&root.join(&name));
            (name, Value::from(hash))
        })
        .collect::<serde_json::Map<_, _>>()
        .into()
}

/// Deterministic token ID for row `row` at position `position`.
fn token(row: usize, position: usize, vocabulary: usize) -> i64 {
    // 7919 is prime, so consecutive positions visit well-spread vocabulary IDs.
    ((row * 104_729 + position * 7_919 + 13) % vocabulary) as i64
}

/// Builds a `[rows, width]` token tensor starting at `offset` outside timed regions.
fn tokens<B: Backend>(
    rows: usize,
    width: usize,
    offset: usize,
    vocabulary: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let values = (0..rows)
        .flat_map(|row| (0..width).map(move |position| token(row, offset + position, vocabulary)))
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [rows, width]), device)
}

/// One fresh-state repetition of one shape.
struct Sample {
    /// Synchronized prefill seconds.
    prefill: f64,
    /// Synchronized seconds for all forced decode steps.
    decode: f64,
    /// Sum of the final step's logits, read back after timing.
    checksum: f64,
}

/// Prefills `source`, then decodes every forced step on fresh cache state.
fn sample<B: Backend>(
    model: &AutoCausalLm<B>,
    source: &Tensor<B, 2, Int>,
    forced: &[Tensor<B, 2, Int>],
    device: &B::Device,
) -> Sample {
    let [rows, width] = source.dims();
    let mut state = model
        .new_decode_state(
            CacheConfig::new(rows, width + forced.len()).expect("cache capacity"),
            rows,
            device,
        )
        .expect("decode state");
    let options = ForwardOptions::default();
    B::sync(device).expect("synchronize before prefill");
    let started = Instant::now();
    let prefill = model
        .prefill(
            CausalLmInput {
                token_ids: source.clone(),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .expect("prefill");
    B::sync(device).expect("synchronize after prefill");
    let prefill_seconds = started.elapsed().as_secs_f64();
    drop(prefill);
    let started = Instant::now();
    let mut last = None;
    for step in forced {
        last = Some(
            model
                .decode(
                    DecodeInput {
                        token_ids: step.clone(),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &options,
                )
                .expect("decode")
                .logits,
        );
    }
    B::sync(device).expect("synchronize after decode");
    let decode_seconds = started.elapsed().as_secs_f64();
    let checksum = last
        .expect("at least one forced step")
        .into_data()
        .convert::<f64>()
        .to_vec::<f64>()
        .expect("F64 logits")
        .iter()
        .sum();
    Sample {
        prefill: prefill_seconds,
        decode: decode_seconds,
        checksum,
    }
}

/// Times every shape for the checkpoint in `TRANSFORMERS_CAUSAL_LM_BENCH_DIR`.
pub fn run<B: Backend>(device: &B::Device, backend_profile: &str) -> Value {
    let root: PathBuf = std::env::var_os("TRANSFORMERS_CAUSAL_LM_BENCH_DIR")
        .map(PathBuf::from)
        .expect("set TRANSFORMERS_CAUSAL_LM_BENCH_DIR to a local causal LM checkpoint");
    let provenance = source_identity(&[
        (
            "tests/small_model_benchmarks.rs",
            include_bytes!("../small_model_benchmarks.rs"),
        ),
        (
            "tests/support/causal_decode_benchmark.rs",
            include_bytes!("causal_decode_benchmark.rs"),
        ),
        (
            "tests/support/benchmark_metrics.rs",
            include_bytes!("benchmark_metrics.rs"),
        ),
        (
            "tests/support/benchmark_identity.rs",
            include_bytes!("benchmark_identity.rs"),
        ),
    ]);
    let identity = identity(&root);
    let options = AutoLoadOptions {
        checkpoint: LoadOptions {
            dtype_policy: DTypePolicy::Cast(TensorDType::F32),
            ..LoadOptions::default()
        },
        load_tokenizer: false,
        ..AutoLoadOptions::default()
    };
    B::sync(device).expect("synchronize before load");
    let started = Instant::now();
    let model = AutoCausalLm::<B>::load_local(&root, &options, device)
        .expect("strict load")
        .into_model();
    B::sync(device).expect("synchronize after load");
    let load_seconds = started.elapsed().as_secs_f64();
    let vocabulary = model.config().vocabulary_size();

    let mut series = Vec::new();
    for (label, rows, width) in SHAPES {
        let source = tokens::<B>(rows, width, 0, vocabulary, device);
        let forced = (0..DECODE_STEPS)
            .map(|step| tokens::<B>(rows, 1, width + step, vocabulary, device))
            .collect::<Vec<_>>();
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        let mut checksums = Vec::new();
        for repetition in 0..WARMUP_REPETITIONS + MEASURED_REPETITIONS {
            let sample = sample(&model, &source, &forced, device);
            if repetition >= WARMUP_REPETITIONS {
                prefill.push(sample.prefill);
                decode.push(sample.decode);
                checksums.push(sample.checksum);
            }
        }
        series.push(json!({
            "shape": label,
            "rows": rows,
            "valid_source_tokens_per_row": width,
            "decode_steps": DECODE_STEPS,
            "prefill": summarize(&prefill, Some(rows * width), MEASURED_REPETITIONS)
                .expect("prefill summary"),
            "decode_model_only": summarize(&decode, Some(rows * DECODE_STEPS), MEASURED_REPETITIONS)
                .expect("decode summary"),
            "final_logit_checksums": checksums,
        }));
    }
    json!({
        "benchmark": "causal LM warm prefill and cached decode",
        "provenance": provenance,
        "load_average_after": host_load(),
        "backend_profile": backend_profile,
        "backend": B::name(device),
        "parameter_dtype": "F32",
        "checkpoint_identity": identity,
        "strict_load_seconds": load_seconds,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "series": series,
    })
}
