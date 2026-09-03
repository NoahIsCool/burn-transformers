//! Opt-in tests for large or externally downloaded model artifacts.
//!
//! Ordinary CI never assumes a model directory exists. Maintainers can point
//! `TRANSFORMERS_QWEN35_DIR` at an immutable local checkout and explicitly run
//! ignored tests when certifying a checkpoint revision.

#![cfg(feature = "tokenizers")]

use std::{fs, path::PathBuf};

use transformers::{
    ChatMessage, ChatTemplateOptions, EncodeOptions, LocalTokenizer,
    checkpoint::{CheckpointManifest, LoadOptions},
    models::{config::ModelConfig, qwen3_5::Qwen3_5TextConfig},
};

#[cfg(feature = "metal")]
use burn::{
    backend::Metal,
    tensor::{Int, Tensor, TensorData},
};
#[cfg(feature = "metal")]
use transformers::{
    ForwardOptions,
    architecture::CausalLmInput,
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
        Tensor::<Metal, 2, Int>::from_data(TensorData::new(tokens, [1, sequence]), &device);
    let output = model
        .forward(
            CausalLmInput {
                token_ids: input_ids,
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect("loaded model must execute inference");
    assert_eq!(output.logits.dims(), [1, 1, 248_320]);
    let top_token = output
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("Metal argmax must return i32 token IDs");
    assert_eq!(top_token.len(), 1);
    assert!((0..248_320).contains(&(top_token[0] as usize)));
}
