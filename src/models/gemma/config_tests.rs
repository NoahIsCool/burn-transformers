//! Configuration schema: per-`model_type` defaults, null handling, the
//! multimodal wrapper, round trips, and ignored keys.

use proptest::prelude::*;

use super::*;
use crate::models::{config::ConfigLoadError, gemma::GemmaCheckpointOptions};

/// Parses and validates a config document.
fn parse(json: &str) -> Result<GemmaConfig, ConfigLoadError> {
    GemmaConfig::from_json_str(json)
}

/// Absent fields take each reference's own constructor defaults.
#[test]
fn absent_fields_take_the_reference_defaults() {
    let gemma2 = parse(r#"{"model_type": "gemma2"}"#).unwrap();
    assert_eq!(gemma2, GemmaConfig::new(GemmaVariant::Gemma2));
    assert_eq!(gemma2.vocab_size, 256_000);
    assert_eq!(gemma2.max_position_embeddings, 8192);
    assert_eq!(gemma2.attn_logit_softcapping, Some(50.0));
    assert_eq!(gemma2.final_logit_softcapping, Some(30.0));
    let gemma3 = parse(r#"{"model_type": "gemma3_text"}"#).unwrap();
    assert_eq!(gemma3, GemmaConfig::new(GemmaVariant::Gemma3Text));
    assert_eq!(gemma3.vocab_size, 262_208);
    assert_eq!(gemma3.max_position_embeddings, 131_072);
    assert_eq!(gemma3.attn_logit_softcapping, None);
    assert_eq!(gemma3.final_logit_softcapping, None);
    for config in [&gemma2, &gemma3] {
        assert_eq!(
            (
                config.hidden_size,
                config.intermediate_size,
                config.num_hidden_layers
            ),
            (2304, 9216, 26)
        );
        assert_eq!(
            (
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim
            ),
            (8, 4, 256)
        );
        assert_eq!(config.hidden_activation, "gelu_pytorch_tanh");
        assert_eq!(config.query_pre_attn_scalar, 256.0);
        assert_eq!(config.sliding_window, 4096);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert!(config.tie_word_embeddings && !config.attention_bias);
        assert!(!config.use_bidirectional_attention);
        assert_eq!(config.pad_token_id, Some(TokenIds::One(0)));
        assert_eq!(config.eos_token_id, Some(TokenIds::One(1)));
        assert_eq!(config.bos_token_id, Some(TokenIds::One(2)));
    }
}

/// `null` disables soft-caps and token IDs but is rejected for required geometry.
#[test]
fn null_values_follow_the_reference() {
    let disabled = parse(
        r#"{"model_type": "gemma2", "attn_logit_softcapping": null,
            "final_logit_softcapping": null, "pad_token_id": null,
            "eos_token_id": [1, 107], "layer_types": null,
            "use_bidirectional_attention": null}"#,
    )
    .unwrap();
    assert_eq!(disabled.attn_logit_softcapping, None);
    assert_eq!(disabled.final_logit_softcapping, None);
    assert_eq!(disabled.pad_token_id, None);
    assert_eq!(disabled.eos_token_id, Some(TokenIds::Many(vec![1, 107])));
    assert_eq!(disabled.layer_types, None);
    assert!(!disabled.use_bidirectional_attention);
    for field in [
        "head_dim",
        "num_key_value_heads",
        "sliding_window",
        "query_pre_attn_scalar",
    ] {
        let json = format!(r#"{{"model_type": "gemma3_text", "{field}": null}}"#);
        let error = parse(&json).unwrap_err();
        assert!(
            matches!(error, ConfigLoadError::Json(_)),
            "{field}: {error}"
        );
        assert!(error.to_string().contains(field), "{field}: {error}");
    }
}

/// A multimodal Gemma 3 config is read through its `text_config`.
#[test]
fn the_multimodal_wrapper_is_read_through_its_text_config() {
    // The pinned google/gemma-3-4b-it config.json (revision 093f9f38).
    let wrapper = r#"{
      "architectures": ["Gemma3ForConditionalGeneration"],
      "boi_token_index": 255999, "eoi_token_index": 256000,
      "eos_token_id": [1, 106], "image_token_index": 262144,
      "initializer_range": 0.02, "mm_tokens_per_image": 256, "model_type": "gemma3",
      "text_config": {
        "hidden_size": 2560, "intermediate_size": 10240, "model_type": "gemma3_text",
        "num_hidden_layers": 34, "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
        "sliding_window": 1024
      },
      "torch_dtype": "bfloat16", "transformers_version": "4.50.0.dev0",
      "vision_config": {"hidden_size": 1152, "image_size": 896, "intermediate_size": 4304,
        "model_type": "siglip_vision_model", "num_attention_heads": 16,
        "num_hidden_layers": 27, "patch_size": 14, "vision_use_head": false}
    }"#;
    let config = parse(wrapper).unwrap();
    assert_eq!(config.variant(), Some(GemmaVariant::Gemma3Text));
    assert_eq!(config.hidden_size, 2560);
    assert_eq!(config.num_hidden_layers, 34);
    assert_eq!(config.sliding_window, 1024);
    assert_eq!(config.vocab_size, 262_208);
    assert_eq!(config.rope_scaling.as_ref().unwrap().factor, Some(8.0));
    assert!(config.rope_parameters.is_none() && config.layer_types.is_none());
}

/// Serialization writes every resolved value and parses back to the same config.
#[test]
fn configs_round_trip_through_json() {
    let fixtures = [
        include_str!("../../../tests/fixtures/tiny_goldens/gemma2/config.json"),
        include_str!("../../../tests/fixtures/tiny_goldens/gemma3-text/config.json"),
    ];
    for json in fixtures {
        let config = parse(json).unwrap();
        let written = serde_json::to_string(&config).unwrap();
        assert_eq!(parse(&written).unwrap(), config, "{written}");
    }
    let legacy = GemmaConfig {
        rope_theta: Some(1e6),
        rope_local_base_freq: Some(1e4),
        sliding_window_pattern: Some(6),
        ..GemmaConfig::new(GemmaVariant::Gemma3Text)
    };
    let written = serde_json::to_string(&legacy).unwrap();
    assert_eq!(parse(&written).unwrap(), legacy);
}

/// Keys the references never read are ignored; unknown model types fail validation.
#[test]
fn ignored_keys_and_unknown_model_types() {
    let ignored = parse(
        r#"{"model_type": "gemma2", "hidden_act": "gelu_pytorch_tanh",
            "cache_implementation": "hybrid", "torch_dtype": "bfloat16",
            "_sliding_window_pattern": 6, "use_cache": true}"#,
    )
    .unwrap();
    assert_eq!(ignored, GemmaConfig::new(GemmaVariant::Gemma2));
    let error = parse(r#"{"model_type": "gemma"}"#).unwrap_err();
    assert!(error.to_string().contains("model_type"), "{error}");
    assert_eq!(
        GemmaConfig {
            model_type: "gemma3".into(),
            ..GemmaConfig::new(GemmaVariant::Gemma3Text)
        }
        .variant(),
        None
    );
}

/// Variants round-trip through their `model_type` strings.
#[test]
fn variants_name_their_model_types() {
    for (variant, model_type) in GemmaVariant::ALL.into_iter().zip(GemmaVariant::MODEL_TYPES) {
        assert_eq!(variant.model_type(), model_type);
        assert_eq!(GemmaVariant::from_model_type(model_type), Some(variant));
        assert_eq!(
            serde_json::to_string(&variant).unwrap(),
            format!("\"{model_type}\"")
        );
    }
    assert_eq!(GemmaVariant::from_model_type("gemma3"), None);
    assert_eq!(
        <GemmaConfig as ModelConfig>::MODEL_TYPES,
        ["gemma2", "gemma3_text"]
    );
}

/// A quantization directive is kept for validation, and `null` means none.
#[test]
fn quantization_directives_are_kept_for_validation() {
    let packed = parse(r#"{"model_type": "gemma2", "quantization_config": {"bits": 4}}"#);
    assert!(
        packed
            .unwrap_err()
            .to_string()
            .contains("quantization_config")
    );
    let plain = parse(r#"{"model_type": "gemma2", "quantization_config": null}"#).unwrap();
    assert!(plain.quantization_config.is_none());
}

proptest! {
    /// Adversarial documents never panic: parsing, validation, and the
    /// checkpoint contract either succeed together or report an error.
    #[test]
    fn adversarial_documents_never_panic(
        model_type in prop::sample::select(vec!["gemma2", "gemma3_text", "gemma3", ""]),
        widths in prop::collection::vec(
            prop::option::of(prop_oneof![
                Just(0_u64),
                1_u64..64,
                Just(u64::from(u32::MAX)),
                Just(u64::MAX),
            ]),
            7,
        ),
        scalar in prop::option::of(prop_oneof![Just(-1.0_f64), Just(0.0), 0.5_f64..512.0]),
        softcap in prop::option::of(prop::option::of(prop_oneof![Just(0.0_f64), 0.5_f64..64.0])),
        pattern in prop::option::of(0_usize..8),
        layer_types in prop::option::of(prop::collection::vec(
            prop::sample::select(vec!["sliding_attention", "full_attention", "chunked_attention"]),
            0..6,
        )),
    ) {
        let mut document = serde_json::json!({ "model_type": model_type });
        let names = [
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ];
        for (name, width) in names.iter().zip(&widths) {
            if let Some(width) = width {
                document[*name] = serde_json::json!(width);
            }
        }
        if let Some(scalar) = scalar {
            document["query_pre_attn_scalar"] = serde_json::json!(scalar);
        }
        if let Some(softcap) = softcap {
            document["final_logit_softcapping"] = serde_json::json!(softcap);
        }
        if let Some(pattern) = pattern {
            document["sliding_window_pattern"] = serde_json::json!(pattern);
        }
        if let Some(layer_types) = layer_types {
            document["layer_types"] = serde_json::json!(layer_types);
        }
        if let Ok(config) = parse(&document.to_string()) {
            let architecture = config.architecture().expect("parsing validated the config");
            prop_assert_eq!(architecture.num_layers(), config.num_hidden_layers);
            // Every validated geometry also has a representable tensor contract.
            prop_assert!(config.checkpoint_contract(GemmaCheckpointOptions::default()).is_ok());
        }
    }
}
