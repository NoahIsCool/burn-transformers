//! Reproducible Qwen3.5-0.8B Metal/F16 timing with authenticated local artifacts.
//!
//! `qwen::qwen35_metal_baseline_benchmark` measures batch-one sources of 32, 128 and
//! 512 valid tokens followed by 128 forced cached steps. Every shape receives three
//! complete warmups and ten measured repetitions. An independent ignored test,
//! `qwen::qwen35_metal_long_prefill_benchmark`, measures only 2K/8K/32K prefill in
//! 256-token chunks. It never decodes beyond a 32K source or allocates above 32K.
//! `qwen::qwen35_metal_public_causal_greedy_benchmark` is a third, separate series:
//! it measures the complete public `generate_causal` coordinator on the pinned
//! natural prompt, with three warmups and ten measured 32-token continuations.
//! Those end-to-end intervals include cache allocation, host selection, probability
//! readback and stopping, but exclude tokenization and reference validation. They
//! are not replacements for the baseline's historical greedy model-API loop.
//!
//! Set `TRANSFORMERS_QWEN35_DIR`, `TRANSFORMERS_QWEN35_REFERENCE_DIR` (the bundle's
//! parent), `TRANSFORMERS_BENCH_METADATA_JSON` and `TRANSFORMERS_BENCH_OUTPUT_JSON`.
//! Run one already-built release test executable with `--ignored --exact
//! --nocapture --test-threads=1`, supervised by `/usr/bin/time -l`. The report is
//! create-only. Hash authentication precedes timed loading and therefore can warm
//! the filesystem cache: strict-load latency is not a cold-disk measurement.
//!
//! Model intervals end at `Backend::sync`; enclosing readback intervals additionally
//! include argmax, one-token transfer and host bookkeeping. KV payload grows as
//! `L_full * 2 * H_kv * D * capacity * sizeof(F16)`, while DeltaNet retains a fixed
//! `L_delta * H_v * D_k * D_v` F32 association matrix and F16 convolution history.
//! These analytical tensor payloads exclude allocation overhead and are checked
//! against the exposed cache tensor shapes and dtypes. RSS boundary snapshots are
//! not peaks; an external supervisor supplies whole-process memory high-water.
//!
//! A report is one sample of a campaign, not a verdict. The campaign protocol in
//! `BENCHMARKS.md` (implemented by `scripts/qwen35_benchmark_protocol.py`) runs
//! the baseline and public tests alternately for two binaries, takes medians over
//! its steady-state rounds, and judges the warm, natural-loop and strict-load
//! medians against the reference in
//! `acceptance/measurements/qwen3.5-0.8b-metal-rebaseline-20261018.json`. Cold
//! first token is one observation per process that includes lazy kernel and Metal
//! pipeline compilation, so it also depends on the state of the operating system's
//! shader cache: the report marks it `"gated": false`, and campaigns report its
//! median and range without gating it.

#[allow(
    dead_code,
    reason = "shared metrics also contains FLAN-specific accounting"
)]
#[path = "support/benchmark_metrics.rs"]
mod benchmark_metrics;

#[allow(dead_code, reason = "the public coordinator runner is Metal-only")]
#[path = "support/public_generation_benchmark.rs"]
mod public_generation_benchmark;

use serde::Serialize;
use transformers::models::qwen3_5::{Qwen3_5LayerType, Qwen3_5TextConfig};

/// Maximum total indexed context in this machine's accepted benchmark protocol.
const MAX_CONTEXT: usize = 32_768;
/// Chunk width of the explicitly separate long-context prefill series.
const PREFILL_CHUNK: usize = 256;

/// Separately recorded growing and fixed batch-one cache tensor payloads.
#[derive(Debug, Serialize)]
struct QwenCacheAccounting {
    /// Full-attention block count.
    full_attention_layers: usize,
    /// Recurrent DeltaNet block count.
    recurrent_layers: usize,
    /// F16 key/value payload added by each indexed token.
    kv_bytes_per_token: u64,
    /// Preallocated full-attention key/value payload.
    kv_allocated_bytes: u64,
    /// Fixed F32 association matrices across all recurrent blocks.
    recurrent_matrix_bytes: u64,
    /// Fixed F16 depthwise-convolution history across all recurrent blocks.
    convolution_history_bytes: u64,
    /// Checked sum; excludes allocator reservations and temporary working tensors.
    total_tensor_payload_bytes: u64,
}

/// Multiplies nonzero dimensions without allowing wraparound to understate memory.
fn product(dimensions: &[u64]) -> Result<u64, String> {
    if dimensions.is_empty() || dimensions.contains(&0) {
        return Err("nonzero cache dimensions are required".into());
    }
    dimensions.iter().try_fold(1_u64, |total, factor| {
        total
            .checked_mul(*factor)
            .ok_or_else(|| "cache product overflow".into())
    })
}

/// Computes real Qwen hybrid cache geometry without initializing model tensors.
fn cache_accounting(
    config: &Qwen3_5TextConfig,
    capacity: usize,
) -> Result<QwenCacheAccounting, String> {
    if capacity == 0 || capacity > MAX_CONTEXT {
        return Err("benchmark capacity must be in 1..=32768".into());
    }
    let schedule = config.resolved_layer_types();
    let full = schedule
        .iter()
        .filter(|&&layer| layer == Qwen3_5LayerType::FullAttention)
        .count();
    let recurrent = schedule.len() - full;
    let kv_per_token = product(&[
        full as u64,
        2,
        config.num_key_value_heads as u64,
        config.head_dim as u64,
        2,
    ])?;
    let kv_allocated = product(&[kv_per_token, capacity as u64])?;
    let matrix = product(&[
        recurrent as u64,
        config.linear_num_value_heads as u64,
        config.linear_key_head_dim as u64,
        config.linear_value_head_dim as u64,
        4,
    ])?;
    let key_channels = product(&[
        2,
        config.linear_num_key_heads as u64,
        config.linear_key_head_dim as u64,
    ])?;
    let value_channels = product(&[
        config.linear_num_value_heads as u64,
        config.linear_value_head_dim as u64,
    ])?;
    let channels = key_channels
        .checked_add(value_channels)
        .ok_or("convolution channel sum overflow")?;
    let width = config
        .linear_conv_kernel_dim
        .checked_sub(1)
        .ok_or("invalid convolution width")?;
    let convolution = product(&[recurrent as u64, channels, width as u64, 2])?;
    let total = kv_allocated
        .checked_add(matrix)
        .and_then(|sum| sum.checked_add(convolution))
        .ok_or("cache sum overflow")?;
    Ok(QwenCacheAccounting {
        full_attention_layers: full,
        recurrent_layers: recurrent,
        kv_bytes_per_token: kv_per_token,
        kv_allocated_bytes: kv_allocated,
        recurrent_matrix_bytes: matrix,
        convolution_history_bytes: convolution,
        total_tensor_payload_bytes: total,
    })
}

/// Validates the fixed-work baseline or separate chunked-prefill shape before allocation.
fn capacity_for(source: usize, prefill_only: bool) -> Result<usize, String> {
    if prefill_only {
        if ![2048, 8192, MAX_CONTEXT].contains(&source) || !source.is_multiple_of(PREFILL_CHUNK) {
            return Err("long prefill requires exactly 2048, 8192 or 32768 source tokens".into());
        }
        Ok(source)
    } else {
        if ![32, 128, 512].contains(&source) {
            return Err("baseline requires exactly 32, 128 or 512 source tokens".into());
        }
        source
            .checked_add(benchmark_metrics::DECODE_STEPS)
            .filter(|&capacity| capacity <= MAX_CONTEXT)
            .ok_or_else(|| "decode would exceed 32K".into())
    }
}

/// Computes the pinned F16 full-vector cosine gate with finite, nonempty inputs.
fn cosine(actual: &[f32], expected: &[f32]) -> Result<f64, String> {
    if actual.is_empty() || actual.len() != expected.len() {
        return Err("cosine requires equal nonempty vectors".into());
    }
    let (mut dot, mut left, mut right) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("non-finite reference vector".into());
        }
        let (actual, expected) = (f64::from(actual), f64::from(expected));
        dot += actual * expected;
        left += actual * actual;
        right += expected * expected;
    }
    if left == 0.0 || right == 0.0 {
        return Err("zero reference norm".into());
    }
    Ok(dot / (left.sqrt() * right.sqrt()))
}

#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    target_os = "macos",
    not(feature = "rocm")
))]
#[path = "support/qwen35_release_reference.rs"]
mod qwen35_release_reference;

/// Streams exactly the recorded bytes from a non-symlink regular Darwin descriptor.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    target_os = "macos",
    not(feature = "rocm")
))]
fn sha256_regular_file(path: &std::path::Path, expected_bytes: u64) -> String {
    use sha2::{Digest, Sha256};
    use std::{io::Read, os::unix::fs::OpenOptionsExt};
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(0x100 | 0x4)
        .open(path)
        .expect("O_NOFOLLOW|O_NONBLOCK artifact open");
    let metadata = file.metadata().unwrap();
    assert!(metadata.is_file());
    assert_eq!(metadata.len(), expected_bytes);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut read_bytes = 0_u64;
    loop {
        let count = file.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }
        read_bytes = read_bytes.checked_add(count as u64).unwrap();
        assert!(read_bytes <= expected_bytes);
        digest.update(&buffer[..count]);
    }
    assert_eq!(read_bytes, expected_bytes);
    format!("{:x}", digest.finalize())
}

#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    target_os = "macos",
    not(feature = "rocm")
))]
mod qwen {
    //! Opt-in authenticated Metal timing for Qwen prefill, cached decode, and public generation.

    use super::benchmark_metrics::{
        DECODE_STEPS, MEASURED_REPETITIONS, MemoryPhase, RunEnvironment, WARMUP_REPETITIONS,
        memory_profiling_enabled, summarize, token_hash,
    };
    use super::{
        MAX_CONTEXT, PREFILL_CHUNK, cache_accounting, capacity_for, cosine, product,
        public_generation_benchmark, qwen35_release_reference, sha256_regular_file,
    };
    use burn::tensor::{
        DType, Int, Tensor, TensorData,
        backend::{Backend, BackendTypes},
    };
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        io::Write,
        path::{Path, PathBuf},
        process::Command,
        time::Instant,
    };
    use transformers::{
        AutoCausalLm, AutoLoadOptions, CausalGenerationRequest, CausalLanguageModel, CausalLmInput,
        DecodeInput, DecodeState, EncodeOptions, ForwardOptions, GenerationConfig, LogitsSelection,
        cache::CacheConfig,
        checkpoint::{DTypePolicy, LoadOptions, TensorDType},
        generate_causal,
        models::qwen3_5::{Qwen3_5DecodeState, Qwen3_5ForCausalLm},
    };

    /// Concrete Metal backend; loaded parameters and activations are explicitly cast to F16.
    type Metal = burn::backend::Metal;

    /// Exact immutable small-model checkpoint identity.
    const REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
    /// Repository label accompanying the content authentication.
    const REPOSITORY: &str = "Qwen/Qwen3.5-0.8B";
    /// Real generated EOS ID; forced-work loops deliberately do not stop here.
    const EOS: u32 = 248_044;

    /// Disjoint timing protocols; selecting one never executes either other warm series.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum BenchmarkSeries {
        /// Historical fixed-source prefill/decode plus natural model-API loop.
        Baseline,
        /// Optional bounded long-context prefill, with no decode suffix.
        LongPrefill,
        /// Actual public generation coordinator on the authenticated natural prompt.
        PublicCausalGreedy,
    }

    /// Requires an explicit local argument rather than discovering network artifacts.
    fn required_path(name: &str) -> PathBuf {
        std::env::var_os(name)
            .map(PathBuf::from)
            .unwrap_or_else(|| panic!("set {name}"))
    }

    /// Hashes a regular local provenance file without loading it all into RAM.
    fn file_hash(path: &Path) -> String {
        sha256_regular_file(path, fs::symlink_metadata(path).unwrap().len())
    }

    /// Executes a metadata probe while preserving unsupported observations explicitly.
    fn command_text(program: &str, arguments: &[&str]) -> Value {
        match Command::new(program)
            .args(arguments)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
        {
            Ok(output) if output.status.success() => {
                json!({"value":String::from_utf8_lossy(&output.stdout).trim(),"error":null})
            }
            Ok(output) => {
                json!({"value":null,"error":String::from_utf8_lossy(&output.stderr).trim()})
            }
            Err(error) => json!({"value":null,"error":error.to_string()}),
        }
    }

    /// Records a boundary RSS observation, explicitly not a device or unified-memory peak.
    fn rss_snapshot(phase: &str) -> Value {
        let observation =
            command_text("ps", &["-o", "rss=", "-p", &std::process::id().to_string()]);
        let bytes = observation["value"]
            .as_str()
            .and_then(|value| value.parse::<u64>().ok())
            .and_then(|value| value.checked_mul(1024));
        json!({"phase":phase,"resident_bytes":bytes,"is_peak":false,"measurement":"ps RSS boundary snapshot","probe":observation})
    }

    /// Completes queued Metal kernels and fusion work at a timing boundary.
    fn synchronize(device: &<Metal as BackendTypes>::Device) {
        <Metal as Backend>::sync(device).expect("synchronized benchmark boundary");
    }

    /// Creates valid batch-one IDs outside timed inference; positions follow cache offsets.
    fn input(ids: &[u32], device: &<Metal as BackendTypes>::Device) -> CausalLmInput<Metal> {
        CausalLmInput {
            token_ids: Tensor::<Metal, 2, Int>::from_data(
                TensorData::new(
                    ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>(),
                    [1, ids.len()],
                ),
                device,
            ),
            token_mask: None,
            position_ids: None,
        }
    }

    /// Reads only one greedy ID from the accelerator.
    fn select(logits: Tensor<Metal, 3>) -> u32 {
        let data = logits.argmax(2).reshape([1]).into_data().convert::<i64>();
        u32::try_from(data.to_vec::<i64>().unwrap()[0]).unwrap()
    }

    /// Materializes F16 logits as F32 solely for untimed reference validation.
    fn values(logits: Tensor<Metal, 3>) -> Vec<f32> {
        assert_eq!(logits.dtype(), DType::F16);
        logits.cast(DType::F32).into_data().to_vec::<f32>().unwrap()
    }

    /// Selects the last valid row without allocating a full sequence vocabulary output.
    fn options() -> ForwardOptions {
        ForwardOptions {
            logits: LogitsSelection::Last,
            ..ForwardOptions::default()
        }
    }

    /// Allocates a production full-KV plus recurrent cache with bounded capacity.
    fn new_state(
        model: &Qwen3_5ForCausalLm<Metal>,
        capacity: usize,
        device: &<Metal as BackendTypes>::Device,
    ) -> Qwen3_5DecodeState<Metal> {
        assert!((1..=MAX_CONTEXT).contains(&capacity));
        model
            .new_decode_state(CacheConfig::new(1, capacity).unwrap(), 1, device)
            .unwrap()
    }

    /// Validates actual committed tensor payloads against independent analytical factors.
    fn observe_cache(
        model: &Qwen3_5ForCausalLm<Metal>,
        capacity: usize,
        device: &<Metal as BackendTypes>::Device,
    ) -> Value {
        let expected = cache_accounting(model.config(), capacity).unwrap();
        assert_eq!(
            (expected.full_attention_layers, expected.recurrent_layers),
            (6, 18)
        );
        let state = new_state(model, capacity, device);
        assert_eq!(state.mtp_layer_count(), 0);
        assert_eq!(state.staging_capacity(), 0);
        let kv = state.tensor_state().main().native_kv(0).unwrap();
        let recurrent = state.tensor_state().main().recurrent(0).unwrap();
        let (mut kv_bytes, mut matrix_bytes, mut convolution_bytes) = (0_u64, 0_u64, 0_u64);
        for layer in 0..kv.spec().layers() {
            for tensor in [kv.keys(layer).unwrap(), kv.values(layer).unwrap()] {
                assert_eq!(tensor.dtype(), DType::F16);
                kv_bytes = kv_bytes
                    .checked_add(
                        product(&tensor.dims().map(|dim| dim as u64))
                            .unwrap()
                            .checked_mul(2)
                            .unwrap(),
                    )
                    .unwrap();
            }
        }
        for layer in 0..recurrent.spec().layers() {
            let matrix = recurrent.matrix(layer).unwrap();
            let convolution = recurrent.convolution(layer).unwrap();
            assert_eq!(matrix.dtype(), DType::F32);
            assert_eq!(convolution.dtype(), DType::F16);
            matrix_bytes = matrix_bytes
                .checked_add(
                    product(&matrix.dims().map(|dim| dim as u64))
                        .unwrap()
                        .checked_mul(4)
                        .unwrap(),
                )
                .unwrap();
            convolution_bytes = convolution_bytes
                .checked_add(
                    product(&convolution.dims().map(|dim| dim as u64))
                        .unwrap()
                        .checked_mul(2)
                        .unwrap(),
                )
                .unwrap();
        }
        assert_eq!(kv_bytes, expected.kv_allocated_bytes);
        assert_eq!(matrix_bytes, expected.recurrent_matrix_bytes);
        assert_eq!(convolution_bytes, expected.convolution_history_bytes);
        json!({"analytical":expected,"observed_kv_bytes":kv_bytes,"observed_recurrent_matrix_bytes":matrix_bytes,"observed_convolution_history_bytes":convolution_bytes,"allocator_overhead_included":false,"temporary_working_tensors_included":false})
    }

    /// Executes the first prefill, then checks all 32 authenticated cached reference rows.
    ///
    /// The returned cold first-token time is a diagnostic (`"gated": false`): it
    /// includes lazy kernel and Metal pipeline compilation, whose cost also depends
    /// on the operating system's shader cache, not only on this crate's graph.
    fn cold_and_reference(
        model: &Qwen3_5ForCausalLm<Metal>,
        reference: &qwen35_release_reference::ReferenceBundle,
        device: &<Metal as BackendTypes>::Device,
    ) -> Value {
        let ids = reference
            .input_ids()
            .iter()
            .map(|&id| u32::try_from(id).unwrap())
            .collect::<Vec<_>>();
        let source = input(&ids, device);
        let mut state = new_state(model, ids.len() + 32, device);
        synchronize(device);
        let phase = MemoryPhase::begin("cold_prefill");
        let start = Instant::now();
        let mut logits = model
            .prefill(source, &mut state, &options())
            .unwrap()
            .logits;
        let first = select(logits.clone());
        synchronize(device);
        let seconds = start.elapsed().as_secs_f64();
        phase.finish();
        let prompt_cosine = cosine(&values(logits.clone()), reference.prompt_logits()).unwrap();
        assert!(prompt_cosine >= 0.999);
        let mut minimum_cached_cosine = 1.0_f64;
        let mut minimum_teacher_cosine = 1.0_f64;
        let mut predicted = Vec::new();
        for step in 0..32 {
            let actual = values(logits.clone());
            minimum_cached_cosine =
                minimum_cached_cosine.min(cosine(&actual, reference.cached_row(step)).unwrap());
            minimum_teacher_cosine =
                minimum_teacher_cosine.min(cosine(&actual, reference.teacher_row(step)).unwrap());
            let id = select(logits);
            assert_eq!(i64::from(id), reference.greedy_token_ids()[step]);
            predicted.push(id);
            if step == 31 {
                break;
            }
            let token = input(&[id], device);
            logits = model
                .decode(
                    DecodeInput {
                        token_ids: token.token_ids,
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &options(),
                )
                .unwrap()
                .logits;
        }
        assert_eq!(first, predicted[0]);
        assert!(minimum_cached_cosine >= 0.999 && minimum_teacher_cosine >= 0.999);
        json!({"seconds":seconds,"gated":false,"source_token_ids":ids,"includes_cache_allocation":false,"includes_token_selection_and_readback":true,"includes_lazy_kernel_compilation":true,"prompt_reference_cosine":prompt_cosine,"minimum_cached_reference_cosine":minimum_cached_cosine,"minimum_teacher_reference_cosine":minimum_teacher_cosine,"reference_cached_decisions":32,"token_ids":predicted,"tokens_u32le_sha256":token_hash(&predicted)})
    }

    /// One baseline or chunked-prefill sample with honest work denominators.
    struct Sample {
        /// Model prefill time excludes final selection/readback.
        prefill: f64,
        /// Prefill plus final greedy selection and one-token readback.
        first_token: f64,
        /// Sum of synchronized model-only forced decode intervals.
        decode_model: f64,
        /// Enclosing forced loop including selection and host bookkeeping.
        decode_inclusive: f64,
        /// First prediction and optional 128 predictions at forced prefixes.
        predictions: Vec<u32>,
    }

    /// Processes bounded pre-created input chunks and optional fixed-work decode tokens.
    fn sample(
        model: &Qwen3_5ForCausalLm<Metal>,
        chunks: &[CausalLmInput<Metal>],
        forced: &[CausalLmInput<Metal>],
        capacity: usize,
        device: &<Metal as BackendTypes>::Device,
    ) -> Sample {
        let mut state = new_state(model, capacity, device);
        synchronize(device);
        let source_length = capacity - forced.len();
        let prefill_phase = MemoryPhase::begin(&format!("warm_source_{source_length}_prefill"));
        let start = Instant::now();
        let mut last = None;
        for chunk in chunks {
            last = Some(
                model
                    .prefill(chunk.clone(), &mut state, &options())
                    .unwrap()
                    .logits,
            );
            synchronize(device);
        }
        let prefill = start.elapsed().as_secs_f64();
        let mut final_logits = last.unwrap();
        let first = select(final_logits.clone());
        synchronize(device);
        let first_token = start.elapsed().as_secs_f64();
        prefill_phase.finish();
        let mut predictions = vec![first];
        let mut decode_model = 0.0;
        let decode_phase = (!forced.is_empty())
            .then(|| MemoryPhase::begin(&format!("warm_source_{source_length}_decode_128")));
        let start = Instant::now();
        for token in forced {
            synchronize(device);
            let step = Instant::now();
            let output = model
                .decode(
                    DecodeInput {
                        token_ids: token.token_ids.clone(),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &options(),
                )
                .unwrap();
            synchronize(device);
            decode_model += step.elapsed().as_secs_f64();
            final_logits = output.logits;
            predictions.push(select(final_logits.clone()));
        }
        synchronize(device);
        let decode_inclusive = start.elapsed().as_secs_f64();
        if let Some(phase) = decode_phase {
            phase.finish();
        }
        // Inspect the complete terminal vocabulary outside timed intervals. A
        // repeatable argmax alone cannot prove long-context logits stayed finite.
        assert!(values(final_logits).iter().all(|value| value.is_finite()));
        assert_eq!(state.lengths(), [capacity]);
        assert_eq!(predictions.len(), forced.len() + 1);
        assert!(first_token >= prefill && decode_inclusive >= decode_model);
        Sample {
            prefill,
            first_token,
            decode_model,
            decode_inclusive,
            predictions,
        }
    }

    /// Separately times actual greedy feedback and EOS handling on the accepted prompt.
    fn natural_greedy(
        model: &Qwen3_5ForCausalLm<Metal>,
        source_ids: &[u32],
        device: &<Metal as BackendTypes>::Device,
    ) -> Value {
        let source = input(source_ids, device);
        let mut times = Vec::new();
        let mut expected: Option<Vec<u32>> = None;
        let mut finish = "length";
        for repetition in 0..WARMUP_REPETITIONS + MEASURED_REPETITIONS {
            synchronize(device);
            let phase = MemoryPhase::begin("natural_greedy");
            let start = Instant::now();
            let mut state = new_state(model, source_ids.len() + 32, device);
            let mut logits = model
                .prefill(source.clone(), &mut state, &options())
                .unwrap()
                .logits;
            let mut ids = Vec::new();
            finish = "length";
            for step in 0..32 {
                let id = select(logits);
                ids.push(id);
                if id == EOS {
                    finish = "eos";
                    break;
                }
                if step == 31 {
                    break;
                }
                let token = input(&[id], device);
                logits = model
                    .decode(
                        DecodeInput {
                            token_ids: token.token_ids,
                            token_mask: None,
                            position_ids: None,
                        },
                        &mut state,
                        &options(),
                    )
                    .unwrap()
                    .logits;
            }
            synchronize(device);
            let seconds = start.elapsed().as_secs_f64();
            phase.finish();
            if let Some(expected) = &expected {
                assert_eq!(&ids, expected);
            }
            expected = Some(ids);
            if repetition >= WARMUP_REPETITIONS {
                times.push(seconds);
            }
        }
        let ids = expected.unwrap();
        json!({"source_token_ids":source_ids,"source_tokens_u32le_sha256":token_hash(source_ids),"max_new_tokens":32,"eos_token_ids":[EOS],"mode":"greedy_model_api_loop","cache_allocation_included":true,"token_ids":ids,"tokens_u32le_sha256":token_hash(&ids),"finish_reason":finish,"total_time":summarize(&times,None,MEASURED_REPETITIONS).unwrap()})
    }

    /// Times complete public coordinator calls, validating every warmup and measured result.
    ///
    /// Request/tensor preparation is outside each timer. `generate_causal` owns its
    /// cache lifetime, token processing, readback and EOS/stop tracking; all of those
    /// costs are inside. The final sync completes queued work before elapsed time is
    /// observed. We report no invented model-only or first-token sub-intervals here.
    fn public_causal_greedy(
        model: &AutoCausalLm<Metal>,
        defaults: &GenerationConfig,
        reference: &qwen35_release_reference::ReferenceBundle,
        device: &<Metal as BackendTypes>::Device,
    ) -> Value {
        let source_ids = reference
            .input_ids()
            .iter()
            .map(|&id| u32::try_from(id).unwrap())
            .collect::<Vec<_>>();
        let expected_ids = reference
            .greedy_token_ids()
            .iter()
            .map(|&id| u32::try_from(id).unwrap())
            .collect::<Vec<_>>();
        let generation = public_generation_benchmark::generation_config(
            defaults,
            model.config().vocabulary_size(),
        )
        .unwrap();
        assert_eq!(generation.eos_token_ids, [EOS]);
        let capacity = public_generation_benchmark::cache_capacity(source_ids.len()).unwrap();
        let source = input(&source_ids, device);
        let mut times = Vec::with_capacity(MEASURED_REPETITIONS);
        let mut outputs = Vec::with_capacity(WARMUP_REPETITIONS + MEASURED_REPETITIONS);
        for repetition in 0..WARMUP_REPETITIONS + MEASURED_REPETITIONS {
            let request = CausalGenerationRequest {
                prompt: source.clone(),
                cache: CacheConfig::new(1, capacity).unwrap(),
                generation: generation.clone(),
                prefill_chunk_size: None,
                vocabulary_size: model.config().vocabulary_size(),
            };
            synchronize(device);
            let phase = MemoryPhase::begin("public_causal_greedy");
            let start = Instant::now();
            let output = generate_causal(model, request).unwrap();
            synchronize(device);
            let seconds = start.elapsed().as_secs_f64();
            phase.finish();
            let canonical = public_generation_benchmark::validate_output(
                &output,
                source_ids.len(),
                &expected_ids,
            )
            .unwrap();
            let measured = repetition >= WARMUP_REPETITIONS;
            if measured {
                times.push(seconds);
            }
            outputs.push(json!({
                "repetition": repetition,
                "partition": if measured { "measured" } else { "warmup" },
                "output": canonical,
            }));
        }
        json!({
            "mode": public_generation_benchmark::MODE,
            "source_token_ids": source_ids,
            "source_tokens_u32le_sha256": token_hash(&source_ids),
            "expected_token_ids": expected_ids,
            "expected_tokens_u32le_sha256": token_hash(&expected_ids),
            "protocol": public_generation_benchmark::protocol(&generation, source_ids.len()).unwrap(),
            "per_repetition_outputs": outputs,
            "total_time": summarize(&times, None, MEASURED_REPETITIONS).unwrap(),
            "model_only_time": null,
            "first_token_latency": null,
            "sub_timing_limitation": "the public non-streaming coordinator returns only the complete continuation; this series measures one synchronized end-to-end interval"
        })
    }

    /// Rejects accidentally executing a debug benchmark while allowing compile-only CI.
    #[allow(
        clippy::assertions_on_constants,
        reason = "ignored tests are compiled in debug but measured only in release"
    )]
    fn require_release() {
        assert!(!cfg!(debug_assertions), "benchmark requires --release");
    }

    /// Authenticates, validates and measures one explicitly selected Qwen benchmark series.
    fn run(kind: BenchmarkSeries) {
        let prefill_only = kind == BenchmarkSeries::LongPrefill;
        let public_causal = kind == BenchmarkSeries::PublicCausalGreedy;
        require_release();
        let environment: RunEnvironment = serde_json::from_slice(
            &fs::read(required_path("TRANSFORMERS_BENCH_METADATA_JSON")).unwrap(),
        )
        .unwrap();
        environment.validate().unwrap();
        assert!(
            environment.accelerator_exclusive,
            "reserve Metal before measuring"
        );
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let source_hash = format!(
            "{:x}",
            Sha256::digest(include_bytes!("qwen35_benchmarks.rs"))
        );
        let helper_hash = format!(
            "{:x}",
            Sha256::digest(include_bytes!("support/benchmark_metrics.rs"))
        );
        let reference_helper_hash = format!(
            "{:x}",
            Sha256::digest(include_bytes!("support/qwen35_release_reference.rs"))
        );
        assert_eq!(
            file_hash(&manifest_dir.join("tests/qwen35_benchmarks.rs")),
            source_hash
        );
        assert_eq!(
            file_hash(&manifest_dir.join("tests/support/benchmark_metrics.rs")),
            helper_hash
        );
        assert_eq!(
            file_hash(&manifest_dir.join("tests/support/qwen35_release_reference.rs")),
            reference_helper_hash
        );
        let public_helper_hash = format!(
            "{:x}",
            Sha256::digest(include_bytes!("support/public_generation_benchmark.rs"))
        );
        assert_eq!(
            file_hash(&manifest_dir.join("tests/support/public_generation_benchmark.rs")),
            public_helper_hash
        );
        let authentication = Instant::now();
        let reference_parent = required_path("TRANSFORMERS_QWEN35_REFERENCE_DIR");
        let reference = qwen35_release_reference::load(&reference_parent);
        let root = required_path("TRANSFORMERS_QWEN35_DIR")
            .canonicalize()
            .unwrap();
        reference.validate_checkpoint(&root);
        let authentication_seconds = authentication.elapsed().as_secs_f64();
        let reference_json = reference_parent
            .join("qwen3.5-0.8b-2fc06364-reference.bundle/qwen3.5-0.8b-2fc06364-reference.json");
        let reference_record: Value =
            serde_json::from_slice(&fs::read(&reference_json).unwrap()).unwrap();
        let device = Default::default();
        let mut memory = vec![rss_snapshot("before_load")];
        synchronize(&device);
        let phase = MemoryPhase::begin("strict_load");
        let start = Instant::now();
        let loaded = AutoCausalLm::<Metal>::load_local(
            &root,
            &AutoLoadOptions {
                checkpoint: LoadOptions {
                    dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                    ..LoadOptions::default()
                },
                repository: Some(REPOSITORY.into()),
                revision: Some(REVISION.into()),
                qwen_mtp: false,
                ..AutoLoadOptions::default()
            },
            &device,
        )
        .unwrap();
        synchronize(&device);
        let load_seconds = start.elapsed().as_secs_f64();
        phase.finish();
        let report = loaded.load_report();
        assert!(report.is_strictly_valid());
        assert_eq!(report.applied.len(), 321);
        assert_eq!(report.allowlisted_ignored.len(), 168);
        assert_eq!(report.dtype_converted.len(), 320);
        assert!(report.quantized.is_empty() && report.allowlisted_missing.is_empty());
        let AutoCausalLm::Qwen3_5(model) = loaded.model() else {
            panic!("wrong Qwen model family");
        };
        let tokenizer = loaded.tokenizer().unwrap();
        let encoded = tokenizer
            .encode(
                reference.prompt(),
                &EncodeOptions {
                    add_special_tokens: false,
                    ..EncodeOptions::default()
                },
            )
            .unwrap();
        let expected_ids = reference
            .input_ids()
            .iter()
            .map(|&id| u32::try_from(id).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(encoded.token_ids(), std::slice::from_ref(&expected_ids));
        memory.push(rss_snapshot("after_load"));
        let cold = cold_and_reference(model, &reference, &device);
        memory.push(rss_snapshot("after_cold_and_reference"));
        let forced_ids = if prefill_only || public_causal {
            Vec::new()
        } else {
            (0..DECODE_STEPS)
                .map(|index| [760_u32, 8964, 579][index % 3])
                .collect::<Vec<_>>()
        };
        let forced = forced_ids
            .iter()
            .map(|&id| input(&[id], &device))
            .collect::<Vec<_>>();
        let lengths = if public_causal {
            Vec::new()
        } else if prefill_only {
            vec![2048, 8192, MAX_CONTEXT]
        } else {
            vec![32, 128, 512]
        };
        let mut series = Vec::new();
        for length in lengths {
            let capacity = capacity_for(length, prefill_only).unwrap();
            let ids = (0..length)
                .map(|index| [760_u32, 8964, 579][index % 3])
                .collect::<Vec<_>>();
            let chunk_size = if prefill_only { PREFILL_CHUNK } else { length };
            let chunks = ids
                .chunks(chunk_size)
                .map(|chunk| input(chunk, &device))
                .collect::<Vec<_>>();
            for _ in 0..WARMUP_REPETITIONS {
                std::hint::black_box(sample(model, &chunks, &forced, capacity, &device));
            }
            let samples = (0..MEASURED_REPETITIONS)
                .map(|_| sample(model, &chunks, &forced, capacity, &device))
                .collect::<Vec<_>>();
            for sample in &samples {
                assert_eq!(sample.predictions, samples[0].predictions);
            }
            let decode_model = (!prefill_only).then(|| {
                summarize(
                    &samples
                        .iter()
                        .map(|sample| sample.decode_model)
                        .collect::<Vec<_>>(),
                    Some(DECODE_STEPS),
                    MEASURED_REPETITIONS,
                )
                .unwrap()
            });
            let decode_inclusive = (!prefill_only).then(|| {
                summarize(
                    &samples
                        .iter()
                        .map(|sample| sample.decode_inclusive)
                        .collect::<Vec<_>>(),
                    Some(DECODE_STEPS),
                    MEASURED_REPETITIONS,
                )
                .unwrap()
            });
            series.push(json!({"source_length":length,"source_token_ids":ids,"source_tokens_u32le_sha256":token_hash(&ids),"batch_size":1,"valid_source_tokens":length,"padding":"none","cache_capacity":capacity,"cache_window":null,"cache":observe_cache(model,capacity,&device),"prefill_chunk_size":chunk_size,"prefill_chunks":chunks.len(),"decode_steps":forced.len(),
                "prefill":summarize(&samples.iter().map(|sample|sample.prefill).collect::<Vec<_>>(),Some(length),MEASURED_REPETITIONS).unwrap(),
                "first_token_latency":summarize(&samples.iter().map(|sample|sample.first_token).collect::<Vec<_>>(),None,MEASURED_REPETITIONS).unwrap(),
                "decode_model_only":decode_model,"decode_including_greedy_readback":decode_inclusive,
                "predicted_token_ids":samples[0].predictions,"predicted_tokens_u32le_sha256":token_hash(&samples[0].predictions),"per_repetition_prediction_hashes":samples.iter().map(|sample|token_hash(&sample.predictions)).collect::<Vec<_>>() }));
            memory.push(rss_snapshot(&format!("after_source_{length}")));
        }
        let natural = if public_causal {
            Some(public_causal_greedy(
                loaded.model(),
                loaded.generation_defaults(),
                &reference,
                &device,
            ))
        } else {
            (!prefill_only).then(|| natural_greedy(model, &expected_ids, &device))
        };
        memory.push(rss_snapshot("after_measured_series"));
        let final_authentication = Instant::now();
        reference.validate_checkpoint(&root);
        let final_authentication_seconds = final_authentication.elapsed().as_secs_f64();
        let toolchain = option_env!("RUSTUP_TOOLCHAIN");
        let compiler = toolchain.map_or_else(
            || json!({"value":null,"error":"compile-time RUSTUP_TOOLCHAIN unavailable"}),
            |toolchain| {
                command_text(
                    "rustc",
                    &[&format!("+{toolchain}"), "--version", "--verbose"],
                )
            },
        );
        let benchmark_name = match kind {
            BenchmarkSeries::Baseline => "qwen35_08b_f16_baseline",
            BenchmarkSeries::LongPrefill => "qwen35_08b_f16_long_prefill_only",
            BenchmarkSeries::PublicCausalGreedy => "qwen35_08b_f16_public_causal_greedy",
        };
        let mut result = json!({"schema_version":1,"benchmark":benchmark_name,
            "process_memory_instrumentation_enabled":memory_profiling_enabled(),"timing_comparable_to_uninstrumented_series":!memory_profiling_enabled(),
            "provenance":{"repository":REPOSITORY,"revision":REVISION,"checkpoint_artifacts":reference_record["provenance"]["artifacts"],"reference_manifest_sha256":file_hash(&reference_json),"reference_payloads":reference_record["payloads"],"crate_commit":command_text("git",&["rev-parse","HEAD"]),"crate_worktree_status":command_text("git",&["status","--porcelain","--untracked-files=all","--","."]),"rustc":compiler,"compiled_rustup_toolchain":toolchain,"cargo_lock_sha256":file_hash(&manifest_dir.join("Cargo.lock")),"benchmark_source_sha256":source_hash,"metric_helper_sha256":helper_hash,"reference_helper_sha256":reference_helper_hash,"benchmark_executable_sha256":file_hash(&std::env::current_exe().unwrap()),"burn_version":"0.21.0","profile":"release (debug assertions disabled)","backend_profile":"Metal/F16","backend_name":Metal::name(&device),"parameter_dtype":"F16","recurrent_matrix_dtype":"F32","quantization":null,"cargo_features":features(),"environment":environment},
            "protocol":{"warmup_repetitions":WARMUP_REPETITIONS,"measured_repetitions":MEASURED_REPETITIONS,"decode_mode":if prefill_only {"none: prefill-only"} else {"teacher_forced_with_greedy_predictions"},"decode_steps":forced.len(),"forced_decode_token_ids":forced_ids,"forced_decode_tokens_u32le_sha256":token_hash(&forced_ids),"maximum_total_context":MAX_CONTEXT,"seed":0,"tokenization_included":false,"prefill_cache_allocation_included":false,"synchronization":"Backend::sync at timing boundaries and every prefill chunk; readback included only in selection-inclusive intervals"},
            "artifact_authentication_seconds_separate_from_load":authentication_seconds,"post_run_artifact_authentication_seconds":final_authentication_seconds,"strict_load_seconds":load_seconds,"strict_load_report":report,"cold_first_token":cold,"warm_series":series,"natural_greedy":natural,
            "memory":{"boundary_snapshots":memory,"phase_peak_host_bytes":null,"phase_peak_device_bytes":null,"phase_peak_unified_bytes":null,"external_high_water_evidence":"supervise the already-built executable; snapshots are not peaks"},"publishable_without_external_memory_and_full_acceptance_evidence":false});
        result["provenance"]["public_generation_helper_sha256"] = json!(public_helper_hash);
        if public_causal {
            result["protocol"] = result["natural_greedy"]["protocol"].clone();
            result["public_causal_cache"] = observe_cache(
                model,
                public_generation_benchmark::cache_capacity(expected_ids.len()).unwrap(),
                &device,
            );
        }
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(required_path("TRANSFORMERS_BENCH_OUTPUT_JSON"))
            .expect("create-only report");
        output
            .write_all(&serde_json::to_vec(&result).unwrap())
            .unwrap();
        output.write_all(b"\n").unwrap();
        output.sync_all().unwrap();
    }

    /// Records all backend and optional runtime features that could affect performance.
    fn features() -> Vec<&'static str> {
        [
            ("std", cfg!(feature = "std")),
            ("metal", cfg!(feature = "metal")),
            ("flex", cfg!(feature = "flex")),
            ("cpu", cfg!(feature = "cpu")),
            ("cuda", cfg!(feature = "cuda")),
            ("rocm", cfg!(feature = "rocm")),
            ("autodiff", cfg!(feature = "autodiff")),
            ("tokenizers", cfg!(feature = "tokenizers")),
            ("chat-templates", cfg!(feature = "chat-templates")),
            ("kernel-tracing", cfg!(feature = "kernel-tracing")),
            (
                "attention-kernel-tracing",
                cfg!(feature = "attention-kernel-tracing"),
            ),
        ]
        .into_iter()
        .filter_map(|(name, enabled)| enabled.then_some(name))
        .collect()
    }

    /// Runs only short-source F16 Metal prefill and fixed-work cached decode baselines.
    #[test]
    #[ignore = "requires authenticated Qwen artifacts, exclusive Metal, environment metadata and release profile"]
    fn qwen35_metal_baseline_benchmark() {
        run(BenchmarkSeries::Baseline);
    }

    /// Runs an independent, potentially expensive 2K/8K/32K chunked-prefill-only series.
    #[test]
    #[ignore = "long prefill only: requires explicit selection, authenticated Qwen artifacts, exclusive Metal and release profile"]
    fn qwen35_metal_long_prefill_benchmark() {
        run(BenchmarkSeries::LongPrefill);
    }

    /// Runs natural greedy feedback through the public API, independent of model-loop baselines.
    #[test]
    #[ignore = "public coordinator: requires authenticated Qwen artifacts, exclusive Metal, environment metadata and release profile"]
    fn qwen35_metal_public_causal_greedy_benchmark() {
        run(BenchmarkSeries::PublicCausalGreedy);
    }
}

#[cfg(test)]
mod tests {
    //! Host-only checks for hybrid cache accounting and overflow rejection.

    use super::*;
    use transformers::models::config::ModelConfig;

    /// Returns official tiny-family geometry without constructing model parameters.
    fn config() -> Qwen3_5TextConfig {
        Qwen3_5TextConfig::from_json_str(include_str!("fixtures/qwen3_5_0_8b.json")).unwrap()
    }

    /// Six growing F16 KV blocks and eighteen fixed recurrent blocks are accounted separately.
    #[test]
    fn cache_geometry_matches_pinned_qwen35() {
        let actual = cache_accounting(&config(), 640).unwrap();
        assert_eq!(
            (actual.full_attention_layers, actual.recurrent_layers),
            (6, 18)
        );
        assert_eq!(actual.kv_bytes_per_token, 12_288);
        assert_eq!(actual.kv_allocated_bytes, 7_864_320);
        assert_eq!(actual.recurrent_matrix_bytes, 18_874_368);
        assert_eq!(actual.convolution_history_bytes, 663_552);
        assert_eq!(actual.total_tensor_payload_bytes, 27_402_240);
        assert_eq!(
            cache_accounting(&config(), MAX_CONTEXT)
                .unwrap()
                .kv_allocated_bytes,
            402_653_184
        );
    }

    /// Invalid and overflowing dimensions cannot produce deceptively small cache estimates.
    #[test]
    fn cache_arithmetic_fails_closed() {
        for capacity in [0, MAX_CONTEXT + 1, usize::MAX] {
            assert!(cache_accounting(&config(), capacity).is_err());
        }
        let mut invalid = config();
        invalid.head_dim = usize::MAX;
        assert!(cache_accounting(&invalid, 1).is_err());
        invalid = config();
        invalid.linear_num_key_heads = usize::MAX;
        assert!(cache_accounting(&invalid, 1).is_err());
        for width in [0, 1] {
            invalid = config();
            invalid.linear_conv_kernel_dim = width;
            assert!(cache_accounting(&invalid, 1).is_err());
        }
        assert!(product(&[]).is_err());
        assert!(product(&[0]).is_err());
        assert!(product(&[u64::MAX, 2]).is_err());
    }

    /// A 32K source belongs only to prefill-only; no 128-step suffix can be implied.
    #[test]
    fn protocol_shapes_never_exceed_total_32k() {
        for (source, capacity) in [(32, 160), (128, 256), (512, 640)] {
            assert_eq!(capacity_for(source, false).unwrap(), capacity);
        }
        for source in [2048, 8192, MAX_CONTEXT] {
            assert_eq!(capacity_for(source, true).unwrap(), source);
            assert!(capacity_for(source, false).is_err());
        }
        for source in [0, 1, MAX_CONTEXT + 1, usize::MAX] {
            assert!(capacity_for(source, false).is_err());
            assert!(capacity_for(source, true).is_err());
        }
        assert!(capacity_for(32, true).is_err());
    }

    /// Full-vector numerical validation rejects missing, nonfinite and zero-norm evidence.
    #[test]
    fn cosine_gate_is_finite_and_sensitive_to_direction() {
        assert_eq!(cosine(&[1.0, 0.0], &[1.0, 0.0]).unwrap(), 1.0);
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]).unwrap(), 0.0);
        assert!(cosine(&[], &[]).is_err());
        assert!(cosine(&[1.0], &[1.0, 2.0]).is_err());
        assert!(cosine(&[0.0], &[1.0]).is_err());
        assert!(cosine(&[f32::NAN], &[1.0]).is_err());
        assert!(cosine(&[1.0], &[f32::INFINITY]).is_err());
    }
}
