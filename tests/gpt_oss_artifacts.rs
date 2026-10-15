//! Real gpt-oss checkpoints against locally generated Transformers references.
//!
//! The reference bundle comes from `scripts/generate_causal_lm_reference.py`
//! (Transformers `AutoModelForCausalLM`, CPU, eager attention), which for
//! gpt-oss must run in BF16:
//!
//! ```text
//! python3 scripts/generate_causal_lm_reference.py --dtype bfloat16 \
//!   --model-dir /Volumes/T9/models/openai/gpt-oss-20b \
//!   --output /path/to/gpt-oss-20b-6cee5e81-bf16-reference.bundle \
//!   --repository openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee
//! ```
//!
//! Without CUDA kernels Transformers decodes the MXFP4 experts to BF16 (its
//! `convert_moe_packed_tensors`, which this crate reproduces bit for bit) and
//! evaluates the whole model in BF16. The bundle records token IDs of a few
//! fixed prompts, every prompt position's full-vocabulary logits (as F32), and
//! a 32-token cached greedy continuation with each step's top-2 margin.
//!
//! These tests are ignored by default and need two local directories:
//!
//! ```text
//! TRANSFORMERS_GPT_OSS_DIR=/Volumes/T9/models/openai/gpt-oss-20b \
//! TRANSFORMERS_GPT_OSS_REFERENCE_DIR=/path/to/reference.bundle \
//! cargo test --release --features metal --test gpt_oss_artifacts -- --ignored --nocapture --test-threads 1
//! ```
//!
//! The checkpoint is loaded with the strict automatic loader: every tensor
//! must be consumed (no missing or unexpected names) and all 48 expert
//! projections must be decoded from MXFP4. The pairing of checkpoint and
//! bundle is verified through the bundle's SHA-256 of `config.json`.
//!
//! # Memory
//!
//! gpt-oss-20b has 20.9 B parameters, 19.1 B of them in routed experts.
//! Loaded in F16 or BF16 the model needs about 39 GiB, plus at most about
//! 1.4 GiB of load staging (the largest expert tensor's 265 MB of blocks,
//! 17 MB of scales, and its 1.06 GB decoded F16 output). F32 needs about
//! 78 GiB and does not fit a 96 GiB host beside the operating system. The
//! reference generator holds the BF16 model (about 39 GiB) plus the
//! checkpoint's MXFP4 tensors while converting, so it needs about 55 GiB.
//!
//! # Dtype and tolerance
//!
//! `TRANSFORMERS_GPT_OSS_DTYPE` selects the stored dtype: `f16` (default),
//! `bf16`, or `f32`. The reference itself runs in BF16, including its eager
//! attention softmax, so logits differ from any of these by BF16 rounding
//! accumulated over 24 blocks, and a near-tie in routing may select a
//! different expert. The default absolute tolerance,
//! [`DEFAULT_TOLERANCE`] = `1.0` logit, is provisional until the integrator
//! records the measured deviation on Metal; set `TRANSFORMERS_GPT_OSS_TOLERANCE`
//! to tighten it. Independently of the reference, the cached and batched
//! paths must agree with this crate's own stateless pass within
//! [`SELF_CONSISTENCY_TOLERANCE`] scaled to the dtype. Greedy tokens are
//! compared while the reference's top-2 margin exceeds twice the tolerance,
//! and at least [`DEFAULT_MIN_GREEDY`] tokens must be compared.

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
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
};

/// Default absolute logit tolerance against the BF16 reference; see the module documentation.
const DEFAULT_TOLERANCE: f32 = 1.0;

/// Absolute agreement of cached/batched passes with this crate's stateless pass, per dtype.
const SELF_CONSISTENCY_TOLERANCE: [(TensorDType, f32); 3] = [
    (TensorDType::F32, 1e-3),
    (TensorDType::F16, 0.25),
    (TensorDType::BF16, 0.5),
];

/// Default minimum number of decisive greedy tokens that must be compared.
const DEFAULT_MIN_GREEDY: usize = 2;

/// Number of expert projections gpt-oss-20b stores as MXFP4 (two per layer).
const MXFP4_PROJECTIONS: usize = 48;

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
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
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

/// Run settings read from the environment.
struct Settings {
    /// Checkpoint directory.
    model_dir: PathBuf,
    /// Reference bundle directory.
    reference_dir: PathBuf,
    /// Stored dtype of every floating parameter.
    dtype: TensorDType,
    /// Absolute logit tolerance against the reference.
    tolerance: f32,
    /// Minimum number of decisive greedy tokens.
    minimum_greedy: usize,
}

impl Settings {
    /// Reads the settings; see the module documentation.
    fn from_environment() -> Self {
        let directory = |name: &str| {
            std::env::var_os(name)
                .map(PathBuf::from)
                .unwrap_or_else(|| panic!("set {name}; see the module documentation"))
        };
        let dtype = match std::env::var("TRANSFORMERS_GPT_OSS_DTYPE").as_deref() {
            Ok("f32") => TensorDType::F32,
            Ok("bf16") => TensorDType::BF16,
            Ok("f16") | Err(_) => TensorDType::F16,
            Ok(other) => {
                panic!("TRANSFORMERS_GPT_OSS_DTYPE must be f16, bf16, or f32, not {other}")
            }
        };
        let parse = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|value| value.parse().expect(name))
        };
        Self {
            model_dir: directory("TRANSFORMERS_GPT_OSS_DIR"),
            reference_dir: directory("TRANSFORMERS_GPT_OSS_REFERENCE_DIR"),
            dtype,
            tolerance: parse("TRANSFORMERS_GPT_OSS_TOLERANCE").unwrap_or(DEFAULT_TOLERANCE),
            minimum_greedy: parse("TRANSFORMERS_GPT_OSS_MIN_GREEDY")
                .map(|value: f32| value as usize)
                .unwrap_or(DEFAULT_MIN_GREEDY),
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

/// Loads the checkpoint strictly after checking it matches the bundle.
fn load<B: Backend>(device: &B::Device) -> (AutoCausalLm<B>, Reference, Settings) {
    let settings = Settings::from_environment();
    let reference = Reference::read(&settings.reference_dir);
    let config = std::fs::read(settings.model_dir.join("config.json")).unwrap();
    assert_eq!(
        hex(&Sha256::digest(&config)),
        reference.config_sha256,
        "the bundle was generated from another config.json"
    );
    let options = AutoLoadOptions {
        checkpoint: LoadOptions {
            dtype_policy: DTypePolicy::Cast(settings.dtype),
            ..LoadOptions::default()
        },
        load_tokenizer: false,
        ..AutoLoadOptions::default()
    };
    let started = std::time::Instant::now();
    let loaded =
        AutoCausalLm::<B>::load_local(&settings.model_dir, &options, device).expect("strict load");
    eprintln!("loaded in {:?}", started.elapsed());
    let report = loaded.load_report();
    assert!(report.missing.is_empty() && report.unexpected.is_empty());
    let decoded = report
        .transformed
        .iter()
        .filter(|record| record.operation == "dequantize_mxfp4")
        .count();
    assert_eq!(
        decoded, MXFP4_PROJECTIONS,
        "every expert projection decodes from MXFP4"
    );
    let model = loaded.into_model();
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

/// Flex: every prompt position's logits.
#[test]
#[ignore = "needs TRANSFORMERS_GPT_OSS_DIR, _REFERENCE_DIR, and 40-80 GiB of memory"]
fn flex_prompt_logits_match_the_reference() {
    check_prompt_logits::<Flex<f32, i32>>(&Default::default());
}

/// Flex: the cached greedy continuation.
#[test]
#[ignore = "needs TRANSFORMERS_GPT_OSS_DIR, _REFERENCE_DIR, and 40-80 GiB of memory"]
fn flex_cached_greedy_matches_the_reference() {
    check_cached_greedy::<Flex<f32, i32>>(&Default::default());
}

/// Flex: a ragged left-padded batch of every prompt.
#[test]
#[ignore = "needs TRANSFORMERS_GPT_OSS_DIR, _REFERENCE_DIR, and 40-80 GiB of memory"]
fn flex_batched_prefill_matches_the_reference() {
    check_batched_prefill::<Flex<f32, i32>>(&Default::default());
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    //! Metal counterparts of the Flex checks (F16 by default), run serially by the integrator.

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

    /// Metal: a ragged left-padded batch of every prompt.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn metal_batched_prefill_matches_the_reference() {
        super::check_batched_prefill::<Metal>(&Default::default());
    }
}
