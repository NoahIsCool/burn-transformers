//! Opt-in tests for large or externally downloaded model artifacts.
//!
//! Ordinary CI never assumes a model directory exists. Maintainers can point
//! `TRANSFORMERS_QWEN35_DIR` at an immutable local checkout and explicitly run
//! ignored tests when certifying a checkpoint revision.

#![cfg(feature = "tokenizers")]

use std::{fs, path::PathBuf};

#[cfg(feature = "chat-templates")]
use transformers::{ChatMessage, ChatTemplateOptions};
use transformers::{
    EncodeOptions, LocalTokenizer,
    checkpoint::{CheckpointManifest, LoadOptions},
    models::{config::ModelConfig, qwen3_5::Qwen3_5TextConfig},
};

#[cfg(feature = "metal")]
use burn::{
    backend::Metal,
    tensor::{DType, Int, Tensor, TensorData},
};
#[cfg(feature = "metal")]
use transformers::{
    CausalLanguageModel, DecodeInput, DecodeState, ForwardOptions, LogitsSelection,
    architecture::CausalLmInput,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, TensorDType, execute_local_load_plan},
    models::qwen3_5::{Qwen3_5CheckpointOptions, Qwen3_5ForCausalLm},
};

/// Parses the official nested multimodal config and exercises text-only tokenization.
#[test]
#[ignore = "requires a pinned external Qwen3.5 artifact directory"]
fn validates_pinned_qwen35_local_artifacts() {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR")
            .expect("set TRANSFORMERS_QWEN35_DIR to the pinned local checkout"),
    );
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("official nested text configuration must validate");
    assert_eq!(config.model_type, "qwen3_5_text");
    assert_eq!(config.vocab_size, 248_320);

    let manifest = CheckpointManifest::discover(&root, &LoadOptions::default())
        .expect("official SafeTensors index and header must pass strict discovery");
    assert_eq!(manifest.tensors.len(), 488);
    assert_eq!(manifest.tensor_bytes, 1_746_882_752);
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.starts_with("model.language_model."))
            .count(),
        320
    );
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.starts_with("model.visual."))
            .count(),
        153
    );
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.starts_with("mtp."))
            .count(),
        15
    );

    let tokenizer = LocalTokenizer::from_directory(&root)
        .expect("official tokenizer artifacts must load locally");
    let prompt = "Explain why cached decoding is faster in one sentence.";
    let encoding = tokenizer
        .encode(
            prompt,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("official tokenizer must encode text");
    assert!(!encoding.token_ids()[0].is_empty());

    #[cfg(feature = "chat-templates")]
    {
        let rendered = tokenizer
            .render_chat(
                &[ChatMessage::text("user", prompt)],
                &ChatTemplateOptions {
                    add_generation_prompt: true,
                    ..ChatTemplateOptions::default()
                },
            )
            .expect("official text-only chat template must render");
        assert!(rendered.contains("<|im_start|>user"));
        assert!(rendered.ends_with("<think>\n\n</think>\n\n"));
    }
}

/// Streams the pinned language weights into Metal F16 and executes one prompt.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a pinned external Qwen3.5 artifact directory and Metal GPU"]
fn loads_pinned_qwen35_weights_and_runs_metal_inference() {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR")
            .expect("set TRANSFORMERS_QWEN35_DIR to the pinned local checkout"),
    );
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("official nested text configuration must validate");
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        ..LoadOptions::default()
    };
    let manifest = CheckpointManifest::discover(&root, &options)
        .expect("official SafeTensors headers must pass strict discovery");
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions::default())
        .expect("official topology must produce a checkpoint contract");
    let plan = contract
        .build_load_plan(&manifest, &options)
        .expect("every language tensor must map strictly");
    assert_eq!(plan.tensors.len(), 320);
    assert_eq!(plan.aliases.len(), 1);
    assert_eq!(plan.report.allowlisted_ignored.len(), 168);

    let device = Default::default();
    let mut model = Qwen3_5ForCausalLm::<Metal>::init(config, &device)
        .expect("official Qwen graph must initialize lazily");
    let outcome = execute_local_load_plan(&manifest, &plan, &mut model)
        .unwrap_or_else(|error| panic!("streamed model load failed: {error}"));
    assert!(outcome.report.is_strictly_valid());
    assert_eq!(outcome.report.applied.len(), 321);
    assert_eq!(outcome.stats.tensors_applied, 320);
    assert_eq!(outcome.stats.aliases_applied, 1);

    let tokenizer = LocalTokenizer::from_directory(&root)
        .expect("official tokenizer artifacts must load locally");
    let encoding = tokenizer
        .encode(
            "Hello",
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("prompt must tokenize");
    let tokens = encoding.token_ids()[0]
        .iter()
        .map(|token| i64::from(*token))
        .collect::<Vec<_>>();
    let sequence = tokens.len();
    let input_ids =
        Tensor::<Metal, 2, Int>::from_data(TensorData::new(tokens.clone(), [1, sequence]), &device);
    let output = model
        .forward(
            CausalLmInput {
                token_ids: input_ids.clone(),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect("loaded model must execute inference");
    assert_eq!(output.logits.dims(), [1, 1, 248_320]);
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/qwen3_5_checkpoint/qwen3_5_0_8b_reference.json"
    ))
    .expect("committed reference logits must be valid JSON");
    let expected_top_ids = reference["top_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let (_, top_ids) = output.logits.clone().topk_with_indices(10, 2);
    let top_ids = top_ids
        .to_data()
        .to_vec::<i32>()
        .expect("Metal top-k must return i32 token IDs");
    assert_eq!(top_ids, expected_top_ids);

    let sample_indices = reference["sample_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap())
        .collect::<Vec<_>>();
    let sample_count = sample_indices.len();
    let sample_indices = Tensor::<Metal, 1, Int>::from_data(
        TensorData::new(sample_indices, [sample_count]),
        &device,
    );
    let sample_logits = output
        .logits
        .clone()
        .select(2, sample_indices)
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("selected Metal logits must convert to f32");
    let expected_sample_logits = reference["sample_logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    for (actual, expected) in sample_logits.iter().zip(expected_sample_logits) {
        assert!(
            (actual - expected).abs() <= 0.25,
            "reference logit mismatch: expected {expected}, found {actual}"
        );
    }

    if let Some(reference_path) = std::env::var_os("TRANSFORMERS_QWEN35_REFERENCE_LOGITS") {
        let reference_path = PathBuf::from(reference_path);
        let bytes = fs::read(&reference_path).unwrap_or_else(|error| {
            panic!(
                "failed to read raw little-endian f32 reference logits from {}: {error}",
                reference_path.display()
            )
        });
        let expected_bytes = 248_320_usize
            .checked_mul(size_of::<f32>())
            .expect("the fixed reference vocabulary byte length must fit usize");
        assert_eq!(
            bytes.len(),
            expected_bytes,
            "reference logit file must contain exactly one little-endian f32 vocabulary vector"
        );
        let reference_logits = bytes
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let actual_logits = output
            .logits
            .clone()
            .cast(DType::F32)
            .to_data()
            .to_vec::<f32>()
            .expect("Metal logits must convert to an f32 host vector");
        assert_eq!(actual_logits.len(), reference_logits.len());
        let cosine = cosine_similarity(&actual_logits, &reference_logits);
        eprintln!("full-vocabulary reference-logit cosine similarity: {cosine:.9}");
        assert!(
            cosine >= 0.999,
            "full-vocabulary cosine similarity {cosine} is below the 0.999 acceptance gate"
        );
    }

    let top_token = top_ids
        .first()
        .copied()
        .expect("top-k returns one result for a non-empty vocabulary");
    assert!((0..248_320).contains(&(top_token as usize)));

    let expected_greedy = reference["greedy_token_ids"]
        .as_array()
        .expect("reference must contain a greedy continuation")
        .iter()
        .map(|value| value.as_i64().expect("greedy token IDs must be integers") as i32)
        .collect::<Vec<_>>();
    assert_eq!(
        expected_greedy.len(),
        reference["greedy_max_new_tokens"]
            .as_u64()
            .expect("reference generation length must be an integer") as usize
    );
    let cache_config = CacheConfig::new(1, sequence + expected_greedy.len())
        .expect("reference generation cache geometry must be valid");
    let mut state = model
        .new_decode_state(cache_config, 1, &device)
        .expect("official Qwen cache geometry must initialize");
    let mut cached = model
        .prefill(
            CausalLmInput {
                token_ids: input_ids.clone(),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("cached prompt prefill must execute");
    let prompt_stateless = output
        .logits
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("stateless prompt logits must convert to f32");
    let prompt_cached = cached
        .logits
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("cached prompt logits must convert to f32");
    let prompt_cosine = cosine_similarity(&prompt_stateless, &prompt_cached);
    assert!(
        prompt_cosine >= 0.999,
        "cached prefill cosine similarity {prompt_cosine} is below the 0.999 gate"
    );

    let mut generated = Vec::with_capacity(expected_greedy.len());
    for (step, expected) in expected_greedy.iter().copied().enumerate() {
        let selected = cached
            .logits
            .clone()
            .argmax(2)
            .to_data()
            .to_vec::<i32>()
            .expect("cached greedy IDs must use i32 storage");
        let token = *selected
            .first()
            .expect("one cached logit row must select one token");
        assert_eq!(
            token, expected,
            "cached continuation differs from the official reference at step {step}"
        );
        generated.push(token);
        if step + 1 < expected_greedy.len() {
            cached = model
                .decode(
                    DecodeInput {
                        token_ids: Tensor::<Metal, 2, Int>::from_data(
                            TensorData::new(vec![i64::from(token)], [1, 1]),
                            &device,
                        ),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap_or_else(|error| panic!("cached decode step {step} failed: {error}"));
        }
    }
    assert_eq!(generated, expected_greedy);
    assert_eq!(state.lengths(), &[sequence + expected_greedy.len() - 1]);

    let mut teacher_tokens = tokens;
    teacher_tokens.extend(
        expected_greedy
            .iter()
            .take(expected_greedy.len() - 1)
            .map(|token| i64::from(*token)),
    );
    let teacher_sequence = teacher_tokens.len();
    let teacher_output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(teacher_tokens, [1, teacher_sequence]),
                    &device,
                ),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .expect("teacher-forced stateless forward must execute");
    let stateless_greedy = teacher_output
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("stateless greedy IDs must use i32 storage");
    assert_eq!(
        stateless_greedy, expected_greedy,
        "cached and stateless top-1 continuations must match at every step"
    );
}

/// Computes cosine similarity with f64 accumulation for acceptance diagnostics.
#[cfg(feature = "metal")]
fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len());
    let (dot, left_square, right_square) =
        left.iter()
            .zip(right)
            .fold((0.0_f64, 0.0_f64, 0.0_f64), |acc, (left, right)| {
                let left = f64::from(*left);
                let right = f64::from(*right);
                (
                    acc.0 + left * right,
                    acc.1 + left * left,
                    acc.2 + right * right,
                )
            });
    dot / (left_square.sqrt() * right_square.sqrt())
}
