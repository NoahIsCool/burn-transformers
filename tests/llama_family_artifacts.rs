//! Real Llama-family checkpoints against locally generated Transformers references.
//!
//! The reference bundle comes from `scripts/generate_causal_lm_reference.py`
//! (Transformers `AutoModelForCausalLM`, F32 or F64, CPU, eager attention): token IDs
//! of a few fixed prompts, every prompt position's full-vocabulary logits as
//! raw little-endian F32, and a 32-token cached greedy continuation of the
//! first prompt with each step's top-2 margin. These tests are ignored by
//! default and need two local directories:
//!
//! ```text
//! TRANSFORMERS_LLAMA_FAMILY_DIR=/path/to/checkpoint \
//! TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR=/path/to/reference-bundle \
//! cargo test --release --test llama_family_artifacts -- --ignored --nocapture --test-threads 1
//! ```
//!
//! Weights are loaded with the strict automatic loader and cast to F32 while
//! streaming (BF16 to F32 is exact, as in the reference), unless
//! `TRANSFORMERS_LLAMA_FAMILY_DTYPE` selects half precision (see
//! [Half precision](#half-precision)). Each test runs on Flex; with the
//! `metal` feature the same checks also run on Metal, and with the `cuda`
//! feature on CUDA (see [CUDA](#cuda)). The pairing of checkpoint and bundle
//! is verified through the bundle's SHA-256 of `config.json`.
//!
//! # Tolerance
//!
//! Both sides compute in F32 with different GEMM blockings, so logits drift
//! by accumulated rounding through every layer. Measured on Flex/F32 over
//! every prompt position (logits of magnitude 10 to 30): SmolLM-135M at
//! `1d461723` deviates by at most `9.1e-5`, Qwen3-4B at `531c80e2` by at most
//! `1.4e-4`, and both reproduce all 32 greedy tokens. The default absolute
//! tolerance, [`DEFAULT_TOLERANCE`] = `1e-3`, leaves about seven times that
//! drift for deeper checkpoints; set `TRANSFORMERS_LLAMA_FAMILY_TOLERANCE` to
//! override it for other checkpoints or backends. The greedy check feeds the
//! reference's tokens back, so every step continues the reference's prefix,
//! and compares each step whose reference top-2 margin exceeds twice the
//! tolerance; a closer race is a tie at that tolerance and is skipped rather
//! than ending the comparison (`tests/support/backend_numerics.rs`, "Greedy
//! ties"). At least [`DEFAULT_MIN_GREEDY`] = 2 tokens must be compared, so a
//! bundle whose continuation is all ties fails instead of passing vacuously;
//! `TRANSFORMERS_LLAMA_FAMILY_MIN_GREEDY` overrides the minimum.
//!
//! # Checkpoints with massive activations
//!
//! An F32 reference carries its own rounding error, so a deviation measures
//! the sum of two implementations' errors. Generate the bundle with
//! `--dtype float64` to measure this crate's error alone. Qwen2-7B at
//! `453ed157` is the case that needs it: its first-position hidden states
//! reach magnitudes near 7,000, where one F32 unit in the last place is about
//! `5e-4`. Against an F64 bundle (checked against an independent
//! layer-streamed F64 forward, agreeing within `2.1e-5`), the largest
//! first-position deviations on its third prompt are `1.6e-3` for F32
//! Transformers itself, `1.8e-4` for Flex, and `1.9e-3` for Metal; every other
//! position stays below `3e-4` on both backends, and both reproduce all 32
//! greedy tokens. The Metal figure is ordinary F32 accumulation order,
//! comparable to (slightly above) the reference implementation's own `1.6e-3`,
//! so that checkpoint's Metal run uses
//! `TRANSFORMERS_LLAMA_FAMILY_TOLERANCE=2.5e-3`. `meta-llama/Llama-3.1-8B` at
//! `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` (Llama 3 RoPE scaling; bundle
//! `config.json` SHA-256
//! `54acfad3cffe057640904ca8a1e83525e6551c70c7a04c641f5a9eda0bbf64bd`) stays
//! within `7.6e-5` on both backends against an F32 bundle.
//!
//! # Half precision
//!
//! `TRANSFORMERS_LLAMA_FAMILY_DTYPE` selects the stored dtype: `f32` (the
//! default; everything above), `bf16` (the Flex profile), or `f16` (the Metal
//! profile); any other value panics. A half-precision run is judged against
//! the same F32 bundle (the F64 one for Qwen2-7B): against a half-precision
//! bundle the two implementations' roundings would add, as the gpt-oss gate
//! found. Its bounds are not chosen here but measured from the official
//! implementation, by `tests/support/half_precision_calibration.rs`:
//! `TRANSFORMERS_LLAMA_FAMILY_OFFICIAL_DIRS` lists (separated as in `PATH`)
//! the four bundles in which Transformers evaluates the same checkpoint,
//! prompts, and greedy length in the run's dtype on the CPU and on MPS, with
//! eager and SDPA attention, and `TRANSFORMERS_LLAMA_FAMILY_DECODE_TRUTH_DIR`
//! names the decode truth, the F32 (F64) CPU-eager bundle that adds the
//! reference's own stateless logits at the greedy positions. That module
//! validates the pairing fail closed before the checkpoint loads. Per prompt,
//! this crate's logits must stay within `s_max E_max` at every position and
//! `s_mean E_mean` on average, where `E` is the largest deviation of the four
//! official graphs from the truth and the slacks are that module's
//! leave-one-out constants (`1.5` and `1.3` in BF16). If any official graph of
//! the dtype is non-finite (Qwen2-7B in F16, whose official CPU and MPS eager
//! F16 logits are NaN on every prompt), the run fails as uncalibrated before
//! the checkpoint loads. `TRANSFORMERS_LLAMA_FAMILY_BF16_REFERENCE_DIR`, which
//! named one official BF16 bundle under the previous rule, is refused.
//!
//! The cached greedy test also checks the decode path whatever the margins:
//! its cached decode rows, and the stateless pass's rows at the same
//! positions, must each meet the bounds of the official graphs' decode rows
//! against the decode truth; the cached-versus-stateless gap is printed as a
//! diagnostic only. The longest prompt (the first of equal ones), prefilled for
//! its first quarter and then decoded token by token, is judged against the
//! envelope of the official graphs' own cached decode under that protocol. The
//! cached prefill and each batched row must lie within their prompt's
//! `s_max E_max` of the bundle and, provisionally, within `0.5` of the
//! stateless pass. The greedy continuation is judged by the same
//! teacher-forced rule as an F32 run, at the larger of prompt 0's and the
//! decode rows' maximum bounds. `TRANSFORMERS_LLAMA_FAMILY_TOLERANCE` is
//! refused, because it would replace the calibration;
//! `TRANSFORMERS_LLAMA_FAMILY_MIN_GREEDY` still applies, and where no greedy
//! step is decisive the cached greedy test needs it set to `0`, which drops
//! only the argmax floor, not the decode checks. An F32 run's cached greedy
//! test reads the decode truth too and compares its cached decode rows with
//! it at the F32 tolerance.
//!
//! The official bundles, their digests, the envelopes, and the derivation of
//! the slacks are recorded in
//! `acceptance/measurements/half-precision-envelope-20261018.json`. To run the
//! Flex/BF16 gate (set `f16`, `--features metal`, and the `metal::` tests for
//! Metal/F16, with the `f16` bundles):
//!
//! ```text
//! R=/Volumes/T9/transformers-references/Llama-3.1-8B-d04e592b
//! TRANSFORMERS_LLAMA_FAMILY_DTYPE=bf16 \
//! TRANSFORMERS_LLAMA_FAMILY_DIR=/Volumes/T9/models/meta-llama/Llama-3.1-8B \
//! TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR=$R \
//! TRANSFORMERS_LLAMA_FAMILY_DECODE_TRUTH_DIR=$R-f32-cpu-eager \
//! TRANSFORMERS_LLAMA_FAMILY_OFFICIAL_DIRS=$R-bf16-cpu-eager-7a50e44:$R-bf16-cpu-sdpa-7a50e44:$R-bf16-mps-eager-7a50e44:$R-bf16-mps-sdpa-7a50e44 \
//! cargo test --release --test llama_family_artifacts -- --ignored --nocapture --test-threads 1
//! ```
//!
//! Each official bundle takes the truth's prompts (`--prompt`, repeated in
//! order) and greedy length; for the default prompts:
//!
//! ```text
//! python3 -I scripts/generate_causal_lm_reference.py --dtype bfloat16 \
//!   --device mps --attn-implementation sdpa \
//!   --decode-tokens-from /Volumes/T9/transformers-references/Llama-3.1-8B-d04e592b --prompt-decode \
//!   --model-dir /Volumes/T9/models/meta-llama/Llama-3.1-8B \
//!   --output /Volumes/T9/transformers-references/Llama-3.1-8B-d04e592b-bf16-mps-sdpa-7a50e44 \
//!   --repository meta-llama/Llama-3.1-8B --revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
//! ```
//!
//! The decode truth is the same command with the truth's `--dtype` (`float32`,
//! or `float64` for Qwen2-7B), the default CPU eager graph, and no
//! `--prompt-decode`. Two bundle-only tests read the same variables but load
//! no checkpoint and run no Burn model, so their passes qualify no Burn run:
//! `official_graphs_calibrate_the_half_precision_envelope` prints every
//! official deviation, envelope, bound, and the decisive greedy steps, and
//! `official_graphs_admit_each_other_leave_one_out` fails if an official graph
//! lies outside the envelope of the other three under the compiled slacks,
//! which means they must be re-derived.
//!
//! # CUDA
//!
//! The `cuda` module repeats the three checks on Burn's `Cuda` backend
//! (device 0; `CUDA_VISIBLE_DEVICES` selects the GPU), built and run on the
//! GPU host with the CUDA toolkit available (`nvcc` at build time,
//! `$CUDA_PATH/include` for run-time kernel compilation):
//!
//! ```text
//! TRANSFORMERS_LLAMA_FAMILY_DIR=/path/to/checkpoint \
//! TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR=/path/to/reference-bundle \
//! cargo test --release --features cuda --test llama_family_artifacts cuda_ -- --ignored --nocapture --test-threads 1
//! ```
//!
//! Each load first probes how the device multiplies F32 matrices
//! (`tests/support/backend_numerics.rs`). A GPU below compute capability 8.0
//! multiplies in F32, and the defaults above apply. From compute capability
//! 8.0 Burn 0.21 rounds F32 matrix inputs to TF32 (10-bit mantissas), which
//! no measured default covers: an F32 run then requires
//! `TRANSFORMERS_LLAMA_FAMILY_TOLERANCE` and qualifies TF32 execution, not
//! F32 numerics.
//!
//! `TRANSFORMERS_LLAMA_FAMILY_DTYPE` and the official envelope select and
//! judge a half-precision run on CUDA exactly as on Flex and Metal
//! ([Half precision](#half-precision)): each dtype is held to the envelope of
//! the four official graphs in its own precision, which were evaluated on the
//! Mac's CPU and MPS (no official CUDA graph is part of the envelope). Such a
//! run refuses `TRANSFORMERS_LLAMA_FAMILY_TOLERANCE` on
//! every device and prints the probe's finding without acting on it: TF32
//! rounds F32 operands, while the matrix products of these graphs take the
//! stored F16 or BF16 tensors as operands (F32 appears only in elementwise
//! steps and reductions such as normalization, softmax, and activation
//! functions). That is read from the source, not measured: no CUDA run, in
//! any dtype, has been recorded yet.

#![cfg(feature = "flex")]

#[path = "support/backend_numerics.rs"]
mod backend_numerics;
#[path = "support/half_precision_calibration.rs"]
mod half_precision_calibration;

use std::path::{Path, PathBuf};

use backend_numerics::{Gap, MatmulPrecision, check_teacher_forced_greedy, checkpoint_tolerance};
use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use half_precision_calibration::{
    DecodeTruth, Envelope, HalfPrecisionGate, is_half_precision, refuse_retired, reject_overrides,
    required_directories, required_directory, run_dtype,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, DecodeInput, ForwardOptions, LogitsSelection,
    TokenMask,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, LoadOptions},
    models::AutoCausalDecodeState,
};

/// Default absolute logit tolerance; see the module documentation.
const DEFAULT_TOLERANCE: f32 = 1e-3;

/// Default minimum number of decisive greedy tokens that must be compared.
const DEFAULT_MIN_GREEDY: usize = 2;

/// One prompt of the reference bundle.
struct Prompt {
    /// Token IDs of the unpadded prompt.
    tokens: Vec<i64>,
    /// Row-major `[tokens, vocabulary]` logits.
    logits: Vec<f32>,
}

/// A parsed reference bundle.
struct Reference {
    /// Vocabulary size `V`.
    vocabulary: usize,
    /// Prompts in bundle order.
    prompts: Vec<Prompt>,
    /// Greedy continuation of prompt 0 as `(token, margin)`.
    greedy: Vec<(i64, f32)>,
    /// SHA-256 of the configuration the bundle was generated from.
    config_sha256: String,
}

impl Reference {
    /// Reads and checks a bundle directory.
    fn read(root: &Path) -> Self {
        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("reference.json")).unwrap())
                .unwrap();
        let vocabulary = manifest["vocabulary_size"].as_u64().unwrap() as usize;
        let prompts = manifest["prompts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|prompt| {
                let tokens: Vec<i64> = serde_json::from_value(prompt["token_ids"].clone()).unwrap();
                let file = root.join(prompt["logits_file"].as_str().unwrap());
                let bytes = std::fs::read(&file).unwrap();
                assert_eq!(
                    hex(&Sha256::digest(&bytes)),
                    prompt["logits_sha256"].as_str().unwrap(),
                    "{} is corrupted",
                    file.display()
                );
                let logits: Vec<f32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect();
                assert_eq!(logits.len(), tokens.len() * vocabulary);
                Prompt { tokens, logits }
            })
            .collect();
        let greedy = manifest["greedy"]["steps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|step| {
                (
                    step["token"].as_i64().unwrap(),
                    step["margin"].as_f64().unwrap() as f32,
                )
            })
            .collect();
        Self {
            vocabulary,
            prompts,
            greedy,
            config_sha256: manifest["provenance"]["config_sha256"]
                .as_str()
                .unwrap()
                .into(),
        }
    }
}

/// Lower-case hexadecimal digest.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Variables of the previous single-bundle half-precision rule, and what replaces each.
const RETIRED: [(&str, &str); 1] = [(
    "TRANSFORMERS_LLAMA_FAMILY_BF16_REFERENCE_DIR",
    "TRANSFORMERS_LLAMA_FAMILY_OFFICIAL_DIRS and TRANSFORMERS_LLAMA_FAMILY_DECODE_TRUTH_DIR",
)];

/// Returns the checkpoint directory and the bundle directory.
fn directories() -> (PathBuf, PathBuf) {
    (
        required_directory("TRANSFORMERS_LLAMA_FAMILY_DIR"),
        required_directory("TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR"),
    )
}

/// Returns an F32 run's absolute tolerance: `TRANSFORMERS_LLAMA_FAMILY_TOLERANCE` or [`DEFAULT_TOLERANCE`].
///
/// [`DEFAULT_TOLERANCE`] applies only to a device of F32 `precision`; a TF32
/// device requires `TRANSFORMERS_LLAMA_FAMILY_TOLERANCE`.
fn f32_tolerance(precision: MatmulPrecision) -> f32 {
    checkpoint_tolerance(
        "TRANSFORMERS_LLAMA_FAMILY_TOLERANCE",
        DEFAULT_TOLERANCE,
        precision,
    )
}

/// Returns how many decisive greedy tokens a run must compare.
fn minimum_greedy() -> usize {
    std::env::var("TRANSFORMERS_LLAMA_FAMILY_MIN_GREEDY")
        .map(|value| value.parse().expect("minimum must be an integer"))
        .unwrap_or(DEFAULT_MIN_GREEDY)
}

/// A loaded checkpoint, its bundle, its greedy tolerance, and the gate of an F16 or BF16 run.
///
/// The tolerance decides greedy steps ([`check_teacher_forced_greedy`]). In
/// an F32 run it is the absolute logit tolerance and also judges every
/// logit; in a half-precision run it is prompt 0's maximum bound
/// ([`HalfPrecisionGate::greedy_tolerance`]), and the gate judges the logits.
type Loaded<B> = (AutoCausalLm<B>, Reference, f32, Option<HalfPrecisionGate>);

/// Loads the checkpoint in `TRANSFORMERS_LLAMA_FAMILY_DTYPE` after checking it matches the bundle.
///
/// The device is probed first. An F32 run (the default) resolves its
/// tolerance from the probe ([`f32_tolerance`]), so a TF32 device without an
/// explicit `TRANSFORMERS_LLAMA_FAMILY_TOLERANCE` fails before loading. An
/// F16 or BF16 run refuses that variable on every device and is judged by
/// the official envelope of `TRANSFORMERS_LLAMA_FAMILY_OFFICIAL_DIRS` and
/// `TRANSFORMERS_LLAMA_FAMILY_DECODE_TRUTH_DIR`, which is validated before the
/// checkpoint loads. The retired single-bundle variable is refused in either
/// dtype.
fn load<B: Backend>(device: &B::Device) -> Loaded<B> {
    refuse_retired(&RETIRED);
    let dtype = run_dtype("TRANSFORMERS_LLAMA_FAMILY_DTYPE");
    let precision = MatmulPrecision::probe::<B>(device);
    let (model_dir, reference_dir) = directories();
    let (tolerance, half) = if is_half_precision(dtype) {
        reject_overrides(&["TRANSFORMERS_LLAMA_FAMILY_TOLERANCE"]);
        let half = HalfPrecisionGate::new(
            dtype,
            &reference_dir,
            &required_directory("TRANSFORMERS_LLAMA_FAMILY_DECODE_TRUTH_DIR"),
            &required_directories("TRANSFORMERS_LLAMA_FAMILY_OFFICIAL_DIRS"),
        );
        let tolerance = half.greedy_tolerance();
        eprintln!(
            "F32 matrix products use {precision:?} inputs; the {dtype:?} run is judged by the official envelope, greedy steps at {tolerance:e}"
        );
        (tolerance, Some(half))
    } else {
        let tolerance = f32_tolerance(precision);
        eprintln!("F32 matrix products use {precision:?} inputs; tolerance {tolerance:e}");
        (tolerance, None)
    };
    let reference = Reference::read(&reference_dir);
    let config = std::fs::read(model_dir.join("config.json")).unwrap();
    assert_eq!(
        hex(&Sha256::digest(&config)),
        reference.config_sha256,
        "the bundle was generated from another config.json"
    );
    let options = AutoLoadOptions {
        checkpoint: LoadOptions {
            dtype_policy: DTypePolicy::Cast(dtype),
            ..LoadOptions::default()
        },
        load_tokenizer: false,
        ..AutoLoadOptions::default()
    };
    let model = AutoCausalLm::<B>::load_local(&model_dir, &options, device)
        .expect("strict load")
        .into_model();
    assert_eq!(model.config().vocabulary_size(), reference.vocabulary);
    (model, reference, tolerance, half)
}

/// Uploads one row of token IDs.
fn row<B: Backend>(tokens: &[i64], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(tokens.to_vec(), [1, tokens.len()]), device)
}

/// Reads logits to the host as F32.
fn host<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor
        .cast(burn::tensor::DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

/// Returns the stateless last-position logits of one unpadded prompt.
fn last_logits<B: Backend>(
    model: &AutoCausalLm<B>,
    tokens: &[i64],
    device: &B::Device,
) -> Vec<f32> {
    let input = CausalLmInput {
        token_ids: row::<B>(tokens, device),
        token_mask: None,
        position_ids: None,
    };
    host(
        model
            .forward(input, &ForwardOptions::default())
            .unwrap()
            .logits,
    )
}

/// Returns the stateless logits of one unpadded sequence at `positions`, one vocabulary-wide row per position.
fn logits_at<B: Backend>(
    model: &AutoCausalLm<B>,
    tokens: &[i64],
    positions: &[usize],
    device: &B::Device,
) -> Vec<f32> {
    let input = CausalLmInput {
        token_ids: row::<B>(tokens, device),
        token_mask: None,
        position_ids: None,
    };
    let options = ForwardOptions {
        logits: LogitsSelection::Positions(positions.to_vec()),
        ..ForwardOptions::default()
    };
    host(model.forward(input, &options).unwrap().logits)
}

/// Returns prompt 0 followed by the bundle's teacher-forced greedy tokens.
fn continued_prompt(reference: &Reference) -> Vec<i64> {
    reference.prompts[0]
        .tokens
        .iter()
        .copied()
        .chain(reference.greedy.iter().map(|&(token, _)| token))
        .collect()
}

/// Appends one token to `state` and returns the logits of its position.
fn decode_one<B: Backend>(
    model: &AutoCausalLm<B>,
    state: &mut AutoCausalDecodeState<B>,
    token: i64,
    device: &B::Device,
) -> Vec<f32> {
    let next = DecodeInput {
        token_ids: row::<B>(&[token], device),
        position_ids: None,
        token_mask: None,
    };
    host(
        model
            .decode(next, state, &ForwardOptions::default())
            .unwrap()
            .logits,
    )
}

/// Decodes the longest prompt token by token after a prefill of its first quarter and judges it by the official prompt-decode envelope.
///
/// The prompt and the prefill length are the ones the official graphs used
/// ([`HalfPrecisionGate::prompt_decode_plan`]). The prefill returns the
/// logits of each of its positions and every later position comes from one
/// cached decode step, so the assembled `[tokens, V]` logits cover the whole
/// prompt and must meet both bounds of the official graphs' own cached decode
/// under the same protocol, as gpt-oss's `check_cached_long_prompt` bounds its
/// decode path against the bundle itself.
fn check_cached_prompt_decode<B: Backend>(
    model: &AutoCausalLm<B>,
    reference: &Reference,
    half: &HalfPrecisionGate,
    device: &B::Device,
) {
    let (index, prefill) = half.prompt_decode_plan();
    let prompt = &reference.prompts[index];
    let length = prompt.tokens.len();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, length).unwrap(), 1, device)
        .unwrap();
    let every_position = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let input = CausalLmInput {
        token_ids: row::<B>(&prompt.tokens[..prefill], device),
        token_mask: None,
        position_ids: None,
    };
    let mut logits = host(
        model
            .prefill(input, &mut state, &every_position)
            .unwrap()
            .logits,
    );
    for &token in &prompt.tokens[prefill..] {
        logits.extend(decode_one(model, &mut state, token, device));
    }
    half.check_prompt_decode(
        &format!("prompt {index} decoded after a {prefill}-token prefill"),
        &logits,
        &prompt.logits,
    );
}

/// Returns the largest absolute difference of two equally long sets of logits.
///
/// A non-finite value on either side counts as an infinite difference, so a
/// NaN or infinite logit fails every tolerance (see [`Gap::measure`], which
/// also rejects a length mismatch).
fn gap(actual: &[f32], expected: &[f32]) -> f32 {
    Gap::measure(actual, expected).deviation
}

/// Stateless logits at every prompt position match the reference.
fn check_prompt_logits<B: Backend>(device: &B::Device) {
    let (model, reference, tolerance, half) = load::<B>(device);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    for (index, prompt) in reference.prompts.iter().enumerate() {
        let input = CausalLmInput {
            token_ids: row::<B>(&prompt.tokens, device),
            token_mask: None,
            position_ids: None,
        };
        let logits = host(model.forward(input, &options).unwrap().logits);
        if let Some(half) = &half {
            half.check_prompt(&format!("prompt {index}"), index, &logits, &prompt.logits);
            continue;
        }
        let largest = gap(&logits, &prompt.logits);
        eprintln!(
            "prompt {index}: {} tokens, max |Δ| = {largest:e}",
            prompt.tokens.len()
        );
        assert!(
            largest <= tolerance,
            "prompt {index}: max |Δ| = {largest:e}"
        );
    }
}

/// The cached greedy continuation of prompt 0 matches at every decisive step, and its decode rows match the decode truth.
///
/// The reference's tokens are fed back, so ties are skipped rather than
/// ending the comparison; see [`check_teacher_forced_greedy`], which decides
/// an F32 run's steps at its tolerance and a half-precision run's at the
/// gate's greedy bound. Whatever the margins, every decode row is also judged
/// against the decode truth in `TRANSFORMERS_LLAMA_FAMILY_DECODE_TRUTH_DIR`:
/// an F32 run's cached rows at its tolerance ([`DecodeTruth::check_f32`]), a
/// half-precision run's cached rows and its stateless rows at the same
/// positions by the official decode envelope
/// ([`HalfPrecisionGate::check_decode`]). A half-precision run also checks
/// the prefill against the stateless pass (provisionally) and the longest
/// prompt decoded token by token ([`check_cached_prompt_decode`]).
fn check_cached_greedy<B: Backend>(device: &B::Device) {
    let (model, reference, tolerance, half) = load::<B>(device);
    let f32_truth = half.is_none().then(|| {
        DecodeTruth::read(
            &directories().1,
            &required_directory("TRANSFORMERS_LLAMA_FAMILY_DECODE_TRUTH_DIR"),
        )
    });
    let prompt = &reference.prompts[0];
    let capacity = prompt.tokens.len() + reference.greedy.len();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, capacity).unwrap(), 1, device)
        .unwrap();
    let input = CausalLmInput {
        token_ids: row::<B>(&prompt.tokens, device),
        token_mask: None,
        position_ids: None,
    };
    let logits = host(
        model
            .prefill(input, &mut state, &ForwardOptions::default())
            .unwrap()
            .logits,
    );
    let last = &prompt.logits[(prompt.tokens.len() - 1) * reference.vocabulary..];
    match &half {
        None => assert!(gap(&logits, last) <= tolerance, "cached prefill logits"),
        Some(half) => {
            half.check_rows("cached prefill", 0, &logits, last);
            let stateless = last_logits(&model, &prompt.tokens, device);
            half.check_consistency("cached prefill", &logits, &stateless);
        }
    }
    // Every decode step's logits are kept for the decode-row checks below.
    let mut decoded = Vec::new();
    let mut decode = |token| {
        let logits = decode_one(&model, &mut state, token, device);
        decoded.push(logits.clone());
        logits
    };
    check_teacher_forced_greedy(
        "greedy",
        &reference.greedy,
        tolerance,
        minimum_greedy(),
        "TRANSFORMERS_LLAMA_FAMILY_MIN_GREEDY",
        logits,
        &mut decode,
    );
    // The comparison decodes no token after its last step; decoding the last
    // greedy token too gives every token of the continuation one decode step,
    // and decode step s, fed greedy token s at position T + s, returns the
    // logits of that position, the decode truth's row s.
    let Some(&(token, _)) = reference.greedy.last() else {
        panic!("the bundle's greedy continuation is empty, so no decode step can be checked");
    };
    decode(token);
    let Some(half) = &half else {
        f32_truth
            .expect("an F32 run reads the decode truth")
            .check_f32("cached decode", &decoded, tolerance);
        return;
    };
    let continued = continued_prompt(&reference);
    let positions: Vec<usize> = (prompt.tokens.len()..continued.len()).collect();
    let stateless = logits_at(&model, &continued, &positions, device);
    half.check_decode("cached decode", prompt.tokens.len(), &decoded, &stateless);
    check_cached_prompt_decode(&model, &reference, half, device);
}

/// All prompts in one left-padded batch reproduce each prompt's last logits.
///
/// A half-precision run also checks each row against the stateless pass.
fn check_batched_prefill<B: Backend>(device: &B::Device) {
    let (model, reference, tolerance, half) = load::<B>(device);
    let width = reference
        .prompts
        .iter()
        .map(|prompt| prompt.tokens.len())
        .max()
        .unwrap();
    let rows = reference.prompts.len();
    let mut tokens = Vec::with_capacity(rows * width);
    let mut valid = Vec::with_capacity(rows * width);
    for prompt in &reference.prompts {
        let padding = width - prompt.tokens.len();
        tokens.extend(std::iter::repeat_n(0, padding).chain(prompt.tokens.iter().copied()));
        valid.extend(
            std::iter::repeat_n(0_i64, padding).chain(std::iter::repeat_n(1, prompt.tokens.len())),
        );
    }
    let token_mask =
        Tensor::<B, 2, Int>::from_data(TensorData::new(valid, [rows, width]), device).equal_elem(1);
    let mut state = model
        .new_decode_state(CacheConfig::new(rows, width).unwrap(), rows, device)
        .unwrap();
    let input = CausalLmInput {
        token_ids: Tensor::from_data(TensorData::new(tokens, [rows, width]), device),
        token_mask: Some(TokenMask::new(token_mask)),
        position_ids: None,
    };
    let logits = host(
        model
            .prefill(input, &mut state, &ForwardOptions::default())
            .unwrap()
            .logits,
    );
    for (index, prompt) in reference.prompts.iter().enumerate() {
        let actual = &logits[index * reference.vocabulary..(index + 1) * reference.vocabulary];
        let expected = &prompt.logits[(prompt.tokens.len() - 1) * reference.vocabulary..];
        if let Some(half) = &half {
            let label = format!("batched row {index}");
            half.check_consistency(&label, actual, &last_logits(&model, &prompt.tokens, device));
            half.check_rows(&label, index, actual, expected);
            continue;
        }
        let largest = gap(actual, expected);
        eprintln!("batched row {index}: max |Δ| = {largest:e}");
        assert!(largest <= tolerance, "batched row {index}");
    }
}

/// Reads the official envelope of `TRANSFORMERS_LLAMA_FAMILY_DTYPE` from the bundles alone.
///
/// # Panics
///
/// If the dtype is not half precision or [`Envelope::read`] refuses the bundles.
fn official_envelope() -> Envelope {
    refuse_retired(&RETIRED);
    let dtype = run_dtype("TRANSFORMERS_LLAMA_FAMILY_DTYPE");
    assert!(
        is_half_precision(dtype),
        "set TRANSFORMERS_LLAMA_FAMILY_DTYPE to bf16 or f16 to read the official envelope of that dtype"
    );
    Envelope::read(
        dtype,
        &required_directory("TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR"),
        &required_directory("TRANSFORMERS_LLAMA_FAMILY_DECODE_TRUTH_DIR"),
        &required_directories("TRANSFORMERS_LLAMA_FAMILY_OFFICIAL_DIRS"),
    )
}

/// The official graphs' distance from the truth, which sets the half-precision bounds of `TRANSFORMERS_LLAMA_FAMILY_DTYPE`.
///
/// Reads only the bundles, validates their pairing, and prints each
/// manifest's SHA-256, each official graph's maximum (with its position and
/// vocabulary entry) and mean deviation of every prompt, the decode rows, and
/// the decoded prompt, the envelopes and bounds, and the greedy steps a
/// half-precision run compares. No Burn model runs, so a pass qualifies no
/// Burn run; an uncalibrated truth fails.
#[test]
#[ignore = "needs TRANSFORMERS_LLAMA_FAMILY_DTYPE, _REFERENCE_DIR, _DECODE_TRUTH_DIR, and _OFFICIAL_DIRS (bundles only)"]
fn official_graphs_calibrate_the_half_precision_envelope() {
    official_envelope().report(&required_directory(
        "TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR",
    ));
}

/// Every official graph lies within the envelope of the other three under the compiled slacks.
///
/// A failure means the slacks must be re-derived with
/// `scripts/derive_half_precision_envelope.py`, never tuned; see the support
/// module's Slack section. Bundles only.
#[test]
#[ignore = "needs TRANSFORMERS_LLAMA_FAMILY_DTYPE, _REFERENCE_DIR, _DECODE_TRUTH_DIR, and _OFFICIAL_DIRS (bundles only)"]
fn official_graphs_admit_each_other_leave_one_out() {
    official_envelope().assert_leave_one_out();
}

/// Flex (F32, or `TRANSFORMERS_LLAMA_FAMILY_DTYPE`): every prompt position's logits.
#[test]
#[ignore = "needs TRANSFORMERS_LLAMA_FAMILY_DIR and _REFERENCE_DIR"]
fn flex_prompt_logits_match_the_reference() {
    check_prompt_logits::<Flex<f32, i32>>(&Default::default());
}

/// Flex (F32, or `TRANSFORMERS_LLAMA_FAMILY_DTYPE`): the cached greedy continuation.
#[test]
#[ignore = "needs TRANSFORMERS_LLAMA_FAMILY_DIR and _REFERENCE_DIR"]
fn flex_cached_greedy_matches_the_reference() {
    check_cached_greedy::<Flex<f32, i32>>(&Default::default());
}

/// Flex (F32, or `TRANSFORMERS_LLAMA_FAMILY_DTYPE`): a ragged left-padded batch of every prompt.
#[test]
#[ignore = "needs TRANSFORMERS_LLAMA_FAMILY_DIR and _REFERENCE_DIR"]
fn flex_batched_prefill_matches_the_reference() {
    check_batched_prefill::<Flex<f32, i32>>(&Default::default());
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    //! Metal counterparts of the Flex checks, run serially by the integrator.
    //!
    //! They run in F32 unless `TRANSFORMERS_LLAMA_FAMILY_DTYPE` selects the
    //! F16 profile (`f16`).

    use burn::backend::Metal;

    /// Metal: every prompt position's logits.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_LLAMA_FAMILY_DIR, and _REFERENCE_DIR"]
    fn metal_prompt_logits_match_the_reference() {
        super::check_prompt_logits::<Metal>(&Default::default());
    }

    /// Metal: the cached greedy continuation.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_LLAMA_FAMILY_DIR, and _REFERENCE_DIR"]
    fn metal_cached_greedy_matches_the_reference() {
        super::check_cached_greedy::<Metal>(&Default::default());
    }

    /// Metal: a ragged left-padded batch of every prompt.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_LLAMA_FAMILY_DIR, and _REFERENCE_DIR"]
    fn metal_batched_prefill_matches_the_reference() {
        super::check_batched_prefill::<Metal>(&Default::default());
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    //! CUDA counterparts of the Flex checks, run serially on the GPU host.
    //!
    //! They run in F32 unless `TRANSFORMERS_LLAMA_FAMILY_DTYPE` selects half
    //! precision (`bf16` or `f16`). An F32 run on a GPU that rounds F32
    //! matrix inputs to TF32 needs an explicit
    //! `TRANSFORMERS_LLAMA_FAMILY_TOLERANCE`; see the [module
    //! documentation](super#cuda).

    use burn::backend::Cuda;

    /// CUDA: every prompt position's logits.
    #[test]
    #[ignore = "needs CUDA, TRANSFORMERS_LLAMA_FAMILY_DIR, and _REFERENCE_DIR"]
    fn cuda_prompt_logits_match_the_reference() {
        super::check_prompt_logits::<Cuda>(&Default::default());
    }

    /// CUDA: the cached greedy continuation.
    #[test]
    #[ignore = "needs CUDA, TRANSFORMERS_LLAMA_FAMILY_DIR, and _REFERENCE_DIR"]
    fn cuda_cached_greedy_matches_the_reference() {
        super::check_cached_greedy::<Cuda>(&Default::default());
    }

    /// CUDA: a ragged left-padded batch of every prompt.
    #[test]
    #[ignore = "needs CUDA, TRANSFORMERS_LLAMA_FAMILY_DIR, and _REFERENCE_DIR"]
    fn cuda_batched_prefill_matches_the_reference() {
        super::check_batched_prefill::<Cuda>(&Default::default());
    }
}
