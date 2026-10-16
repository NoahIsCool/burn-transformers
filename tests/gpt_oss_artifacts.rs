//! Real gpt-oss checkpoints against locally generated Transformers references.
//!
//! Reference bundles come from `scripts/generate_causal_lm_reference.py`
//! (Transformers `AutoModelForCausalLM`, CPU, eager attention). Without CUDA
//! kernels Transformers decodes the MXFP4 experts to BF16 (its
//! `convert_moe_packed_tensors`, which this crate reproduces bit for bit). Two
//! kinds of bundle are useful:
//!
//! * a **full** BF16 reference of gpt-oss-20b, the release gate (the F32 model
//!   does not fit a 96 GiB host):
//!
//!   ```text
//!   python3 scripts/generate_causal_lm_reference.py --dtype bfloat16 \
//!     --model-dir /Volumes/T9/models/openai/gpt-oss-20b \
//!     --output /path/to/gpt-oss-20b-6cee5e81-bf16-reference.bundle \
//!     --repository openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee
//!   ```
//!
//! * a **truncated** F32 reference (`--num-hidden-layers 1`): the real
//!   embedding, first block (sliding attention with real sinks, the real
//!   router, 32 MXFP4-decoded experts), final norm, and head, in about 9 GiB.
//!   It checks real weights end to end at F32 precision on Flex. This test
//!   reads the truncation from the bundle's provenance and loads only those
//!   blocks, allowlisting the checkpoint's other blocks as unread.
//!
//! The checks cover every prompt position's logits and decisive argmaxes,
//! the cached greedy continuation, a ragged left-padded batched prefill
//! (also against the crate's own stateless pass), and the longest prompt
//! decoded token by token through the caches; include a prompt longer than
//! the 128-token window (`--prompt`, repeatable) so sliding layers wrap
//! their ring.
//!
//! Each bundle records token IDs of a few fixed prompts, every prompt
//! position's full-vocabulary logits (as F32), and a 32-token cached greedy
//! continuation with each step's top-2 margin. The tests are ignored by
//! default and need two local directories:
//!
//! ```text
//! TRANSFORMERS_GPT_OSS_DIR=/Volumes/T9/models/openai/gpt-oss-20b \
//! TRANSFORMERS_GPT_OSS_REFERENCE_DIR=/path/to/reference.bundle \
//! cargo test --release [--features metal] --test gpt_oss_artifacts -- --ignored --nocapture --test-threads 1
//! ```
//!
//! A full bundle loads through the strict automatic loader; a truncated one
//! through the family loader. Either way no tensor may be missing or
//! unexpectedly present, and every expert projection of the loaded blocks
//! must be decoded from MXFP4. The pairing of checkpoint and bundle is
//! verified through the bundle's SHA-256 of `config.json`.
//!
//! # Memory
//!
//! gpt-oss-20b has 20.9 B parameters, 19.1 B of them in routed experts.
//! Loaded in F16 or BF16 the model needs about 39 GiB, plus at most about
//! 1.4 GiB of load staging (the largest expert tensor's 265 MB of blocks,
//! 17 MB of scales, and its 1.06 GB decoded F16 output). F32 needs about
//! 78 GiB and does not fit a 96 GiB host beside the operating system. The
//! reference generator holds the BF16 model (about 39 GiB) plus the
//! checkpoint's MXFP4 tensors while converting, so it needs about 55 GiB. A
//! one-block truncation needs about 9 GiB in F32 on either side.
//!
//! # Dtype and tolerance
//!
//! `TRANSFORMERS_GPT_OSS_DTYPE` selects the stored dtype (`f16`, `bf16`,
//! `f32`, or `f64`); it defaults to the bundle's own dtype, so an F32 bundle
//! runs in F32 and a BF16 bundle in BF16 (set `f16` for the Metal F16
//! profile). An F32 run against an F32 bundle differs only by reduction
//! order: every logit must lie within [`F32_TOLERANCE`] and their mean
//! deviation within [`F32_MEAN_TOLERANCE`]. Whenever the run or the bundle is
//! half precision, rounding dominates and [`HALF_TOLERANCE`] and
//! [`HALF_MEAN_TOLERANCE`] apply instead. `TRANSFORMERS_GPT_OSS_TOLERANCE`
//! and `TRANSFORMERS_GPT_OSS_MEAN_TOLERANCE` override them.
//!
//! Matching storage dtypes does not make a half-precision comparison like for
//! like: Transformers' BF16 eager path also rounds its attention and router
//! softmaxes to BF16, while this crate normalizes in F32. On four real blocks
//! (Flex, the four prompts below), the crate's BF16 run is no closer to the
//! BF16 reference than to the F32 one, and the BF16 reference itself is about
//! as far from the F32 reference:
//!
//! | run | reference | max \|Δ\| | mean \|Δ\| per prompt |
//! | --- | --- | --- | --- |
//! | crate F32 | F32 | `1.3e-4` | below `5e-6` |
//! | crate BF16 | F32 | `1.37` | at most `0.030` |
//! | crate F16 | F32 | `1.82` | at most `0.062` |
//! | crate BF16 | BF16 | `1.71` | at most `0.054` |
//! | crate F16 | BF16 | `1.85` | at most `0.060` |
//! | Transformers BF16 | F32 | `1.73` | at most `0.055` |
//!
//! So a half-precision run (Metal F16 included) is best qualified against a
//! truncated **F32** bundle, which measures its own rounding; a full BF16
//! bundle then checks the whole depth at the BF16 reference's resolution.
//! The half-precision defaults cover the four-block measurements with some
//! headroom; they are provisional for the full 24-block model, whose
//! deviation the integrator records.
//!
//! Independently of the reference, cached and batched passes must agree with
//! this crate's own stateless pass within [`SELF_CONSISTENCY_TOLERANCE`] for
//! the dtype. Greedy tokens are compared while the reference's top-2 margin
//! exceeds twice the tolerance, and at least [`DEFAULT_MIN_GREEDY`] tokens
//! must be compared (`TRANSFORMERS_GPT_OSS_MIN_GREEDY` overrides it). A
//! truncated model's greedy margins are mostly below one logit, so a
//! half-precision run against a truncated bundle compares no greedy tokens
//! and needs `TRANSFORMERS_GPT_OSS_MIN_GREEDY=0`.
//!
//! # Measured
//!
//! On Flex/F32 against truncated F32 bundles of gpt-oss-20b at `6cee5e8`
//! (Transformers 5.2.0, PyTorch 2.10.0, generated on this host on
//! 2026-10-15): one block with the default prompts deviates by at most
//! `7.6e-5` (mean `3.4e-6`); four blocks (two sliding, two full), with a
//! 183-token fourth prompt, by at most `1.3e-4` (mean below `5e-6`), the
//! token-by-token cached decode of that prompt (wrapping the 128-slot ring)
//! by `7.6e-5`, and all 32 greedy tokens match. The half-precision rows of
//! the table above come from the same four-block F32 bundle and a four-block
//! BF16 bundle generated from the same prompts
//! (`gpt-oss-20b-6cee5e81-4layer-{f32,bf16}-reference.bundle`). The
//! full-model BF16 bundle and the Metal runs are the integrator's.

#![cfg(feature = "flex")]

use std::path::{Path, PathBuf};

use burn::{
    backend::Flex,
    tensor::{DType, Int, Tensor, TensorData, backend::Backend},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, DecodeInput, ForwardOptions, LogitsSelection,
    TokenMask,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, KeyMatcher, LoadOptions, LoadReport, TensorDType},
    models::{
        config::ModelConfig,
        gpt_oss::{GptOssConfig, GptOssForCausalLm},
    },
};

/// Default absolute logit tolerance of an F32 (or F64) run against an F32 bundle.
const F32_TOLERANCE: f32 = 1e-3;

/// Default bound on one prompt's mean absolute logit deviation, F32 run against an F32 bundle.
const F32_MEAN_TOLERANCE: f32 = 1e-4;

/// Default absolute logit tolerance when the run or the bundle is half precision.
///
/// Four real blocks deviate by at most `1.85` in every half-precision pairing
/// (see the module documentation); provisional for the full model.
const HALF_TOLERANCE: f32 = 2.5;

/// Default bound on one prompt's mean absolute logit deviation in half precision.
///
/// Four real blocks stay at or below `0.062`; an architectural error moves
/// most logits of most positions, so the mean separates it from rounding
/// noise far better than the maximum. Provisional for the full model.
const HALF_MEAN_TOLERANCE: f32 = 0.15;

/// Absolute agreement of cached/batched passes with this crate's stateless pass, per dtype.
///
/// The batched check compares a left-padded batch with each prompt run
/// alone, so it also absorbs any shape dependence of half-precision kernels.
/// On four real blocks on Flex the largest gaps were `3.6e-5` (F32), `0.125`
/// (BF16), and `0.41` (F16): Flex's F16 matmul accumulates in F16, so F16
/// runs there are noisier than BF16 ones.
const SELF_CONSISTENCY_TOLERANCE: [(TensorDType, f32); 3] = [
    (TensorDType::F32, 1e-3),
    (TensorDType::F16, 0.5),
    (TensorDType::BF16, 0.5),
];

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
    /// Evaluation dtype of the reference (`float32` or `bfloat16`).
    dtype: String,
    /// Blocks kept by a truncated reference, or `None` for the whole model.
    layers: Option<usize>,
}

impl Reference {
    /// Reads and checks a bundle directory.
    fn read(root: &Path) -> Self {
        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("reference.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["model_type"], "gpt_oss", "not a gpt-oss bundle");
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
        let provenance = &manifest["provenance"];
        Self {
            vocabulary,
            prompts,
            greedy,
            config_sha256: provenance["config_sha256"].as_str().unwrap().into(),
            dtype: provenance["dtype"].as_str().unwrap_or("float32").into(),
            layers: provenance["num_hidden_layers"]
                .as_u64()
                .map(|layers| layers as usize),
        }
    }
}

/// Lower-case hexadecimal digest.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Run settings read from the environment and the bundle.
struct Settings {
    /// Checkpoint directory.
    model_dir: PathBuf,
    /// Stored dtype of every floating parameter.
    dtype: TensorDType,
    /// Absolute logit tolerance against the reference.
    tolerance: f32,
    /// Bound on each prompt's mean absolute logit deviation from the reference.
    mean_tolerance: f32,
    /// Minimum number of decisive greedy tokens.
    minimum_greedy: usize,
}

/// Returns the dtype a bundle was evaluated in, from its provenance name.
fn bundle_dtype(name: &str) -> TensorDType {
    match name {
        "float32" => TensorDType::F32,
        "bfloat16" => TensorDType::BF16,
        other => panic!("unsupported reference dtype {other}"),
    }
}

/// Returns the default `(maximum, mean)` logit tolerances of running `run` against a `bundle` reference.
///
/// Only a full-precision run against an F32 bundle is held to reduction-order
/// accuracy; any half-precision side is held to rounding accuracy.
fn default_tolerances(bundle: TensorDType, run: TensorDType) -> (f32, f32) {
    let full_precision =
        bundle == TensorDType::F32 && matches!(run, TensorDType::F32 | TensorDType::F64);
    if full_precision {
        (F32_TOLERANCE, F32_MEAN_TOLERANCE)
    } else {
        (HALF_TOLERANCE, HALF_MEAN_TOLERANCE)
    }
}

impl Settings {
    /// Reads the settings; dtype and tolerance defaults follow the bundle's dtype.
    fn new(model_dir: PathBuf, reference: &Reference) -> Self {
        let bundle = bundle_dtype(&reference.dtype);
        let dtype = match std::env::var("TRANSFORMERS_GPT_OSS_DTYPE").as_deref() {
            Ok("f64") => TensorDType::F64,
            Ok("f32") => TensorDType::F32,
            Ok("bf16") => TensorDType::BF16,
            Ok("f16") => TensorDType::F16,
            Err(_) => bundle,
            Ok(other) => {
                panic!("TRANSFORMERS_GPT_OSS_DTYPE must be f16, bf16, f32, or f64, not {other}")
            }
        };
        let (default_tolerance, default_mean) = default_tolerances(bundle, dtype);
        let parse = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|value| value.parse::<f32>().expect(name))
        };
        Self {
            model_dir,
            dtype,
            tolerance: parse("TRANSFORMERS_GPT_OSS_TOLERANCE").unwrap_or(default_tolerance),
            mean_tolerance: parse("TRANSFORMERS_GPT_OSS_MEAN_TOLERANCE").unwrap_or(default_mean),
            minimum_greedy: parse("TRANSFORMERS_GPT_OSS_MIN_GREEDY")
                .map_or(DEFAULT_MIN_GREEDY, |value| value as usize),
        }
    }

    /// Returns the self-consistency tolerance of the selected dtype.
    fn consistency(&self) -> f32 {
        SELF_CONSISTENCY_TOLERANCE
            .iter()
            .find(|(dtype, _)| *dtype == self.dtype)
            .map_or(1.0, |&(_, tolerance)| tolerance)
    }
}

/// Returns a required directory from the environment.
fn directory(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name}; see the module documentation"))
}

/// Checks the strict report: nothing missing or unexpected, and `layers` blocks of MXFP4 experts.
fn check_report(report: &LoadReport, layers: usize) {
    assert!(
        report.missing.is_empty() && report.unexpected.is_empty(),
        "missing {:?}, unexpected {:?}",
        report.missing,
        report.unexpected
    );
    let decoded = report
        .transformed
        .iter()
        .filter(|record| record.operation == "dequantize_mxfp4")
        .count();
    assert_eq!(
        decoded,
        2 * layers,
        "every expert projection decodes from MXFP4"
    );
}

/// Loads the checkpoint (truncated as the bundle says) after checking it matches the bundle.
fn load<B: Backend>(device: &B::Device) -> (AutoCausalLm<B>, Reference, Settings) {
    let model_dir = directory("TRANSFORMERS_GPT_OSS_DIR");
    let reference = Reference::read(&directory("TRANSFORMERS_GPT_OSS_REFERENCE_DIR"));
    let settings = Settings::new(model_dir, &reference);
    let text = std::fs::read_to_string(settings.model_dir.join("config.json")).unwrap();
    assert_eq!(
        hex(&Sha256::digest(text.as_bytes())),
        reference.config_sha256,
        "the bundle was generated from another config.json"
    );
    let checkpoint = LoadOptions {
        dtype_policy: DTypePolicy::Cast(settings.dtype),
        ..LoadOptions::default()
    };
    let started = std::time::Instant::now();
    let model = match reference.layers {
        None => {
            let options = AutoLoadOptions {
                checkpoint,
                load_tokenizer: false,
                ..AutoLoadOptions::default()
            };
            let loaded = AutoCausalLm::<B>::load_local(&settings.model_dir, &options, device)
                .expect("strict load");
            let layers = GptOssConfig::from_json_str(&text)
                .unwrap()
                .num_hidden_layers;
            check_report(loaded.load_report(), layers);
            loaded.into_model()
        }
        Some(layers) => {
            let mut config = GptOssConfig::from_json_str(&text).unwrap();
            let total = config.num_hidden_layers;
            config.num_hidden_layers = layers;
            config.layer_types.truncate(layers);
            let options = LoadOptions {
                allow_unexpected: (layers..total)
                    .map(|layer| KeyMatcher::prefix(format!("model.layers.{layer}.")))
                    .collect(),
                ..checkpoint
            };
            let mut model = GptOssForCausalLm::<B>::init(config, device).unwrap();
            let outcome = model
                .load_checkpoint(&settings.model_dir, &options)
                .expect("strict truncated load");
            check_report(&outcome.report, layers);
            AutoCausalLm::from(model)
        }
    };
    eprintln!(
        "loaded {} blocks as {:?} in {:?}",
        reference
            .layers
            .map_or("all".into(), |layers| layers.to_string()),
        settings.dtype,
        started.elapsed()
    );
    assert_eq!(model.config().vocabulary_size(), reference.vocabulary);
    (model, reference, settings)
}

/// Uploads one row of token IDs.
fn row<B: Backend>(tokens: &[i64], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(tokens.to_vec(), [1, tokens.len()]), device)
}

/// Reads logits to the host as F32.
fn host<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.cast(DType::F32).into_data().to_vec::<f32>().unwrap()
}

/// Returns the largest absolute difference.
fn gap(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max)
}

/// Returns the argmax, preferring the lowest index on ties.
fn argmax(logits: &[f32]) -> usize {
    (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then(b.cmp(&a)))
        .unwrap()
}

/// Returns the reference's top-2 margin of one position.
fn margin(logits: &[f32]) -> f32 {
    let best = argmax(logits);
    let runner_up = (0..logits.len())
        .filter(|&index| index != best)
        .map(|index| logits[index])
        .fold(f32::NEG_INFINITY, f32::max);
    logits[best] - runner_up
}

/// Stateless logits at every prompt position match the reference; decisive argmaxes agree.
fn check_prompt_logits<B: Backend>(device: &B::Device) {
    let (model, reference, settings) = load::<B>(device);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let vocabulary = reference.vocabulary;
    for (index, prompt) in reference.prompts.iter().enumerate() {
        let input = CausalLmInput {
            token_ids: row::<B>(&prompt.tokens, device),
            token_mask: None,
            position_ids: None,
        };
        let logits = host(model.forward(input, &options).unwrap().logits);
        let largest = gap(&logits, &prompt.logits);
        let mean = logits
            .iter()
            .zip(&prompt.logits)
            .map(|(a, e)| f64::from((a - e).abs()))
            .sum::<f64>()
            / logits.len() as f64;
        eprintln!(
            "prompt {index}: {} tokens, max |Δ| = {largest:e}, mean |Δ| = {mean:e}",
            prompt.tokens.len()
        );
        for position in 0..prompt.tokens.len() {
            let expected = &prompt.logits[position * vocabulary..(position + 1) * vocabulary];
            if margin(expected) > 2.0 * settings.tolerance {
                let actual = &logits[position * vocabulary..(position + 1) * vocabulary];
                assert_eq!(
                    argmax(actual),
                    argmax(expected),
                    "prompt {index}, position {position}: decisive argmax"
                );
            }
        }
        assert!(
            largest <= settings.tolerance,
            "prompt {index}: max |Δ| = {largest:e}"
        );
        assert!(
            mean <= f64::from(settings.mean_tolerance),
            "prompt {index}: mean |Δ| = {mean:e}"
        );
    }
}

/// The cached greedy continuation of prompt 0 matches while it is decisive.
fn check_cached_greedy<B: Backend>(device: &B::Device) {
    let (model, reference, settings) = load::<B>(device);
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
    let mut logits = host(
        model
            .prefill(input, &mut state, &ForwardOptions::default())
            .unwrap()
            .logits,
    );
    let last = &prompt.logits[(prompt.tokens.len() - 1) * reference.vocabulary..];
    assert!(
        gap(&logits, last) <= settings.tolerance,
        "cached prefill logits"
    );
    let mut compared = 0;
    for (step, &(expected, step_margin)) in reference.greedy.iter().enumerate() {
        if step_margin <= 2.0 * settings.tolerance {
            eprintln!("greedy step {step}: reference margin {step_margin:e} is a tie; stopping");
            break;
        }
        compared += 1;
        assert_eq!(argmax(&logits) as i64, expected, "greedy step {step}");
        let next = DecodeInput {
            token_ids: row::<B>(&[expected], device),
            position_ids: None,
            token_mask: None,
        };
        logits = host(
            model
                .decode(next, &mut state, &ForwardOptions::default())
                .unwrap()
                .logits,
        );
    }
    eprintln!(
        "greedy: {compared} of {} tokens match",
        reference.greedy.len()
    );
    let minimum = settings.minimum_greedy.min(reference.greedy.len());
    assert!(
        compared >= minimum,
        "only {compared} decisive greedy tokens were compared (minimum {minimum})"
    );
}

/// All prompts in one left-padded batch reproduce the stateless pass and the reference.
fn check_batched_prefill<B: Backend>(device: &B::Device) {
    let (model, reference, settings) = load::<B>(device);
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
        let alone = host(
            model
                .forward(
                    CausalLmInput {
                        token_ids: row::<B>(&prompt.tokens, device),
                        token_mask: None,
                        position_ids: None,
                    },
                    &ForwardOptions::default(),
                )
                .unwrap()
                .logits,
        );
        let consistency = gap(actual, &alone);
        let expected = &prompt.logits[(prompt.tokens.len() - 1) * reference.vocabulary..];
        let largest = gap(actual, expected);
        eprintln!(
            "batched row {index}: max |Δ| = {largest:e} to the reference, {consistency:e} to the stateless pass"
        );
        assert!(
            consistency <= settings.consistency(),
            "batched row {index} vs stateless"
        );
        assert!(largest <= settings.tolerance, "batched row {index}");
    }
}

/// The longest prompt decodes token by token through the caches, past the sliding window.
///
/// A quarter of the prompt is prefilled into a state whose capacity exceeds
/// gpt-oss's 128-token window (so sliding layers use their ring), and every
/// remaining token is decoded alone and compared with the reference logits
/// at its position. A prompt longer than the window wraps the ring.
fn check_cached_long_prompt<B: Backend>(device: &B::Device) {
    let (model, reference, settings) = load::<B>(device);
    let prompt = reference
        .prompts
        .iter()
        .max_by_key(|prompt| prompt.tokens.len())
        .unwrap();
    let length = prompt.tokens.len();
    let prefill = (length / 4).max(1);
    let capacity = length.max(256);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, capacity).unwrap(), 1, device)
        .unwrap();
    let vocabulary = reference.vocabulary;
    let expected =
        |position: usize| &prompt.logits[position * vocabulary..(position + 1) * vocabulary];
    let logits = host(
        model
            .prefill(
                CausalLmInput {
                    token_ids: row::<B>(&prompt.tokens[..prefill], device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits,
    );
    let mut largest = gap(&logits, expected(prefill - 1));
    for position in prefill..length {
        let logits = host(
            model
                .decode(
                    DecodeInput {
                        token_ids: row::<B>(&prompt.tokens[position..=position], device),
                        position_ids: None,
                        token_mask: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap()
                .logits,
        );
        largest = largest.max(gap(&logits, expected(position)));
    }
    eprintln!(
        "cached decode of a {length}-token prompt (prefill {prefill}): max |Δ| = {largest:e}"
    );
    assert!(largest <= settings.tolerance, "cached long prompt");
}

/// Runs default to the bundle's dtype; only full precision on both sides gets the tight tolerances.
#[test]
fn tolerance_defaults_follow_the_bundle_and_run_dtypes() {
    use TensorDType::{BF16, F16, F32, F64};
    assert_eq!(bundle_dtype("float32"), F32);
    assert_eq!(bundle_dtype("bfloat16"), BF16);
    assert!(std::panic::catch_unwind(|| bundle_dtype("float16")).is_err());
    let tight = (F32_TOLERANCE, F32_MEAN_TOLERANCE);
    let loose = (HALF_TOLERANCE, HALF_MEAN_TOLERANCE);
    assert_eq!(default_tolerances(F32, F32), tight);
    assert_eq!(default_tolerances(F32, F64), tight);
    for (bundle, run) in [
        (F32, F16),
        (F32, BF16),
        (BF16, BF16),
        (BF16, F16),
        (BF16, F32),
    ] {
        assert_eq!(
            default_tolerances(bundle, run),
            loose,
            "{bundle:?} bundle, {run:?} run"
        );
    }
    // The mean bound is the sharper half-precision check, and both cover the measurements.
    const { assert!(HALF_MEAN_TOLERANCE < HALF_TOLERANCE / 10.0) };
    const { assert!(HALF_TOLERANCE > 1.85 && HALF_MEAN_TOLERANCE > 0.062) };
}

/// Flex: every prompt position's logits.
#[test]
#[ignore = "needs TRANSFORMERS_GPT_OSS_DIR, _REFERENCE_DIR, and 9-80 GiB of memory"]
fn flex_prompt_logits_match_the_reference() {
    check_prompt_logits::<Flex<f32, i32>>(&Default::default());
}

/// Flex: the cached greedy continuation.
#[test]
#[ignore = "needs TRANSFORMERS_GPT_OSS_DIR, _REFERENCE_DIR, and 9-80 GiB of memory"]
fn flex_cached_greedy_matches_the_reference() {
    check_cached_greedy::<Flex<f32, i32>>(&Default::default());
}

/// Flex: a ragged left-padded batch of every prompt.
#[test]
#[ignore = "needs TRANSFORMERS_GPT_OSS_DIR, _REFERENCE_DIR, and 9-80 GiB of memory"]
fn flex_batched_prefill_matches_the_reference() {
    check_batched_prefill::<Flex<f32, i32>>(&Default::default());
}

/// Flex: the longest prompt decoded token by token, past the window.
#[test]
#[ignore = "needs TRANSFORMERS_GPT_OSS_DIR, _REFERENCE_DIR, and 9-80 GiB of memory"]
fn flex_cached_long_prompt_matches_the_reference() {
    check_cached_long_prompt::<Flex<f32, i32>>(&Default::default());
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    //! Metal counterparts of the Flex checks, run serially by the integrator.
    //!
    //! The stored dtype follows the bundle unless `TRANSFORMERS_GPT_OSS_DTYPE`
    //! is set; the F16 profile sets `f16` and is best qualified against a
    //! truncated F32 bundle (see this file's module documentation).

    use burn::backend::Metal;

    /// Metal: every prompt position's logits.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn metal_prompt_logits_match_the_reference() {
        super::check_prompt_logits::<Metal>(&Default::default());
    }

    /// Metal: the cached greedy continuation.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn metal_cached_greedy_matches_the_reference() {
        super::check_cached_greedy::<Metal>(&Default::default());
    }

    /// Metal: the longest prompt decoded token by token, past the window.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn metal_cached_long_prompt_matches_the_reference() {
        super::check_cached_long_prompt::<Metal>(&Default::default());
    }

    /// Metal: a ragged left-padded batch of every prompt.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn metal_batched_prefill_matches_the_reference() {
        super::check_batched_prefill::<Metal>(&Default::default());
    }
}
