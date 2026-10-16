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
//! streaming (BF16 to F32 is exact, as in the reference). Each test runs on
//! Flex; with the `metal` feature the same checks also run on Metal in F32.
//! The pairing of checkpoint and bundle is verified through the bundle's
//! SHA-256 of `config.json`.
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
//! override it for other checkpoints or backends. Greedy tokens are compared
//! while the reference's top-2 margin exceeds twice the tolerance; a closer
//! race is an F32 tie. At least [`DEFAULT_MIN_GREEDY`] = 2 tokens (the
//! prefill's argmax and one cached decode step) must be compared, so a
//! bundle whose continuation opens with a tie fails instead of passing
//! vacuously; `TRANSFORMERS_LLAMA_FAMILY_MIN_GREEDY` overrides the minimum.
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
//! greedy tokens. The Metal figure is ordinary F32 accumulation order, as
//! large as the reference implementation's own, so that checkpoint's Metal
//! run uses `TRANSFORMERS_LLAMA_FAMILY_TOLERANCE=2.5e-3`. Llama-3.1-8B at
//! `d04e592b` (Llama 3 RoPE scaling) stays within `7.6e-5` on both backends
//! against an F32 bundle.

#![cfg(feature = "flex")]

use std::path::{Path, PathBuf};

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, DecodeInput, ForwardOptions, LogitsSelection,
    TokenMask,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
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

/// Returns the checkpoint directory, the bundle directory, and the tolerance.
fn environment() -> (PathBuf, PathBuf, f32) {
    let variable = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .unwrap_or_else(|| panic!("set {name}; see the module documentation"))
    };
    let tolerance = std::env::var("TRANSFORMERS_LLAMA_FAMILY_TOLERANCE")
        .map(|value| value.parse().expect("tolerance must be a float"))
        .unwrap_or(DEFAULT_TOLERANCE);
    (
        variable("TRANSFORMERS_LLAMA_FAMILY_DIR"),
        variable("TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR"),
        tolerance,
    )
}

/// Returns how many decisive greedy tokens a run must compare.
fn minimum_greedy() -> usize {
    std::env::var("TRANSFORMERS_LLAMA_FAMILY_MIN_GREEDY")
        .map(|value| value.parse().expect("minimum must be an integer"))
        .unwrap_or(DEFAULT_MIN_GREEDY)
}

/// Loads the checkpoint with F32 casts after checking it matches the bundle.
fn load<B: Backend>(device: &B::Device) -> (AutoCausalLm<B>, Reference, f32) {
    let (model_dir, reference_dir, tolerance) = environment();
    let reference = Reference::read(&reference_dir);
    let config = std::fs::read(model_dir.join("config.json")).unwrap();
    assert_eq!(
        hex(&Sha256::digest(&config)),
        reference.config_sha256,
        "the bundle was generated from another config.json"
    );
    let options = AutoLoadOptions {
        checkpoint: LoadOptions {
            dtype_policy: DTypePolicy::Cast(TensorDType::F32),
            ..LoadOptions::default()
        },
        load_tokenizer: false,
        ..AutoLoadOptions::default()
    };
    let model = AutoCausalLm::<B>::load_local(&model_dir, &options, device)
        .expect("strict load")
        .into_model();
    assert_eq!(model.config().vocabulary_size(), reference.vocabulary);
    (model, reference, tolerance)
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

/// Returns the largest absolute difference.
fn gap(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max)
}

/// Returns the argmax.
fn argmax(logits: &[f32]) -> usize {
    (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then(b.cmp(&a)))
        .unwrap()
}

/// Stateless logits at every prompt position match the reference.
fn check_prompt_logits<B: Backend>(device: &B::Device) {
    let (model, reference, tolerance) = load::<B>(device);
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

/// The cached greedy continuation of prompt 0 matches while it is decisive.
fn check_cached_greedy<B: Backend>(device: &B::Device) {
    let (model, reference, tolerance) = load::<B>(device);
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
    assert!(gap(&logits, last) <= tolerance, "cached prefill logits");
    let mut compared = 0;
    for (step, &(expected, margin)) in reference.greedy.iter().enumerate() {
        if margin <= 2.0 * tolerance {
            eprintln!("greedy step {step}: reference margin {margin:e} is a tie; stopping");
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
    let minimum = minimum_greedy().min(reference.greedy.len());
    assert!(
        compared >= minimum,
        "only {compared} decisive greedy tokens were compared (minimum {minimum}); \
         regenerate the bundle with another first prompt or lower TRANSFORMERS_LLAMA_FAMILY_MIN_GREEDY"
    );
}

/// All prompts in one left-padded batch reproduce each prompt's last logits.
fn check_batched_prefill<B: Backend>(device: &B::Device) {
    let (model, reference, tolerance) = load::<B>(device);
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
        let largest = gap(actual, expected);
        eprintln!("batched row {index}: max |Δ| = {largest:e}");
        assert!(largest <= tolerance, "batched row {index}");
    }
}

/// Flex/F32: every prompt position's logits.
#[test]
#[ignore = "needs TRANSFORMERS_LLAMA_FAMILY_DIR and _REFERENCE_DIR"]
fn flex_prompt_logits_match_the_reference() {
    check_prompt_logits::<Flex<f32, i32>>(&Default::default());
}

/// Flex/F32: the cached greedy continuation.
#[test]
#[ignore = "needs TRANSFORMERS_LLAMA_FAMILY_DIR and _REFERENCE_DIR"]
fn flex_cached_greedy_matches_the_reference() {
    check_cached_greedy::<Flex<f32, i32>>(&Default::default());
}

/// Flex/F32: a ragged left-padded batch of every prompt.
#[test]
#[ignore = "needs TRANSFORMERS_LLAMA_FAMILY_DIR and _REFERENCE_DIR"]
fn flex_batched_prefill_matches_the_reference() {
    check_batched_prefill::<Flex<f32, i32>>(&Default::default());
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    //! Metal/F32 counterparts of the Flex checks, run serially by the integrator.

    use burn::backend::Metal;

    /// Metal/F32: every prompt position's logits.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_LLAMA_FAMILY_DIR, and _REFERENCE_DIR"]
    fn metal_prompt_logits_match_the_reference() {
        super::check_prompt_logits::<Metal>(&Default::default());
    }

    /// Metal/F32: the cached greedy continuation.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_LLAMA_FAMILY_DIR, and _REFERENCE_DIR"]
    fn metal_cached_greedy_matches_the_reference() {
        super::check_cached_greedy::<Metal>(&Default::default());
    }

    /// Metal/F32: a ragged left-padded batch of every prompt.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_LLAMA_FAMILY_DIR, and _REFERENCE_DIR"]
    fn metal_batched_prefill_matches_the_reference() {
        super::check_batched_prefill::<Metal>(&Default::default());
    }
}
