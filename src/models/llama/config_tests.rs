//! Serialized schema: per-`model_type` defaults, null versus absent keys,
//! round trips, real checkpoint configs, and rejected directives.

use super::*;
use crate::models::config::RopeType;

/// Parses and validates a config document.
fn parse(
    json: serde_json::Value,
) -> Result<LlamaFamilyConfig, crate::models::config::ConfigLoadError> {
    LlamaFamilyConfig::from_json_str(&json.to_string())
}

/// SmolLM-135M at revision 1d461723 (a `llama` checkpoint written by Transformers 4.41).
const SMOLLM_135M: &str = r#"{
  "architectures": ["LlamaForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "bos_token_id": 0, "eos_token_id": 0, "hidden_act": "silu", "hidden_size": 576,
  "initializer_range": 0.02, "intermediate_size": 1536, "max_position_embeddings": 2048,
  "mlp_bias": false, "model_type": "llama", "num_attention_heads": 9, "num_hidden_layers": 30,
  "num_key_value_heads": 3, "pretraining_tp": 1, "rms_norm_eps": 1e-05, "rope_scaling": null,
  "rope_theta": 10000.0, "tie_word_embeddings": true, "torch_dtype": "bfloat16",
  "transformers_version": "4.41.2", "use_cache": true, "vocab_size": 49152
}"#;

/// Qwen3-4B at revision 531c80e2 (a `qwen3` checkpoint written by Transformers 4.51).
const QWEN3_4B: &str = r#"{
  "architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "bos_token_id": 151643, "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu",
  "hidden_size": 2560, "initializer_range": 0.02, "intermediate_size": 9728,
  "max_position_embeddings": 40960, "max_window_layers": 36, "model_type": "qwen3",
  "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null,
  "tie_word_embeddings": true, "torch_dtype": "bfloat16", "transformers_version": "4.51.0",
  "use_cache": true, "use_sliding_window": false, "vocab_size": 151936
}"#;

/// The RoPE block of `meta-llama/Llama-3.1-8B` at `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
const LLAMA_31_ROPE: &str = r#"{
  "factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
  "original_max_position_embeddings": 8192, "rope_type": "llama3"
}"#;

/// Absent fields take the defaults of the document's own `model_type`.
#[test]
fn defaults_follow_the_model_type() {
    let expectations = [
        ("llama", 32_000, 32, 128, 1e-6, false, None, 10_000.0),
        ("mistral", 32_000, 8, 128, 1e-6, false, Some(4096), 10_000.0),
        ("qwen2", 151_936, 32, 128, 1e-6, false, Some(4096), 10_000.0),
        ("qwen3", 151_936, 32, 128, 1e-6, false, Some(4096), 10_000.0),
        ("smollm3", 128_256, 4, 128, 1e-6, true, None, 2_000_000.0),
        ("olmo2", 50_304, 32, 128, 1e-5, false, None, 10_000.0),
    ];
    for (model_type, vocab, kv, head, eps, tied, window, theta) in expectations {
        let config: LlamaFamilyConfig =
            serde_json::from_value(serde_json::json!({ "model_type": model_type })).unwrap();
        assert_eq!(config.vocab_size, vocab, "{model_type}");
        assert_eq!(config.num_key_value_heads, kv, "{model_type}");
        assert_eq!(config.head_dim, head, "{model_type}");
        assert_eq!(config.rms_norm_eps, eps, "{model_type}");
        assert_eq!(config.tie_word_embeddings, tied, "{model_type}");
        assert_eq!(config.sliding_window, window, "{model_type}");
        assert_eq!(config.default_rope_theta, theta, "{model_type}");
        assert_eq!(config, LlamaFamilyConfig::new(config.variant().unwrap()));
        assert_eq!(
            config.variant().map(LlamaVariant::model_type),
            Some(model_type)
        );
    }
}

/// An explicit `null` derives the value instead of taking the type's default.
#[test]
fn explicit_nulls_differ_from_absent_keys() {
    let absent: LlamaFamilyConfig =
        serde_json::from_value(serde_json::json!({ "model_type": "mistral" })).unwrap();
    let nulls: LlamaFamilyConfig = serde_json::from_value(serde_json::json!({
        "model_type": "mistral", "num_key_value_heads": null, "sliding_window": null,
        "head_dim": null, "eos_token_id": null,
    }))
    .unwrap();
    assert_eq!(absent.num_key_value_heads, 8);
    assert_eq!(nulls.num_key_value_heads, 32);
    assert_eq!(absent.sliding_window, Some(4096));
    assert_eq!(nulls.sliding_window, None);
    assert_eq!(absent.eos_token_id, Some(TokenIds::One(2)));
    assert_eq!(nulls.eos_token_id, None);
    let qwen3: LlamaFamilyConfig = serde_json::from_value(serde_json::json!({
        "model_type": "qwen3", "head_dim": null, "hidden_size": 64, "num_attention_heads": 4,
    }))
    .unwrap();
    assert_eq!(qwen3.head_dim, 16);
}

/// Serialization writes every resolved field, so a round trip is the identity.
#[test]
fn serialization_round_trips() {
    for variant in LlamaVariant::ALL {
        let config = LlamaFamilyConfig::new(variant);
        let json = serde_json::to_string(&config).unwrap();
        let back: LlamaFamilyConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, config, "{variant:?}");
    }
    let config = LlamaFamilyConfig::from_json_str(QWEN3_4B).unwrap();
    let back: LlamaFamilyConfig =
        serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
    assert_eq!(back, config);
}

/// Real checkpoint configs parse with their exact fields.
#[test]
fn real_checkpoint_configs_parse() {
    let smollm = LlamaFamilyConfig::from_json_str(SMOLLM_135M).unwrap();
    assert_eq!(smollm.variant(), Some(LlamaVariant::Llama));
    assert_eq!(
        (smollm.num_attention_heads, smollm.num_key_value_heads),
        (9, 3)
    );
    assert_eq!(smollm.head_dim, 64);
    assert_eq!(smollm.rms_norm_eps, 1e-5);
    assert!(smollm.tie_word_embeddings && smollm.rope_parameters.is_none());
    assert_eq!(smollm.rope_theta, Some(10_000.0));

    let qwen = LlamaFamilyConfig::from_json_str(QWEN3_4B).unwrap();
    assert_eq!(qwen.variant(), Some(LlamaVariant::Qwen3));
    assert_eq!(qwen.head_dim, 128);
    assert_eq!(qwen.sliding_window, None);
    assert_eq!(qwen.rope_theta, Some(1_000_000.0));

    let rope: RopeParameters = serde_json::from_str(LLAMA_31_ROPE).unwrap();
    assert_eq!(rope.rope_type, RopeType::Llama3);
    let llama31 = parse(serde_json::json!({
        "model_type": "llama", "rope_scaling": serde_json::from_str::<serde_json::Value>(LLAMA_31_ROPE).unwrap(),
        "rope_theta": 500000.0, "max_position_embeddings": 131072, "vocab_size": 128256,
        "num_key_value_heads": 8, "eos_token_id": [128001, 128008, 128009],
    }))
    .unwrap();
    assert_eq!(llama31.rope_parameters, Some(rope));
}

/// Keys no reference reads are ignored; a packed-weight directive is rejected.
#[test]
fn unread_keys_are_ignored_and_quantization_is_rejected() {
    let config = parse(serde_json::json!({
        "model_type": "llama", "pretraining_tp": 4, "torch_dtype": "bfloat16",
        "use_cache": false, "architectures": ["X"], "quantization_config": null,
    }))
    .unwrap();
    assert_eq!(config, LlamaFamilyConfig::new(LlamaVariant::Llama));
    let error = parse(serde_json::json!({
        "model_type": "llama", "quantization_config": {"quant_method": "fp8"},
    }))
    .unwrap_err();
    assert!(error.to_string().contains("quantization_config"), "{error}");
}

/// Unknown model types and malformed fields fail with contextual errors.
#[test]
fn unsupported_documents_are_rejected() {
    let error = parse(serde_json::json!({ "model_type": "gemma2" })).unwrap_err();
    assert!(error.to_string().contains("model_type"), "{error}");
    assert!(parse(serde_json::json!({ "model_type": "llama", "vocab_size": -1 })).is_err());
    let error =
        parse(serde_json::json!({ "model_type": "llama", "eos_token_id": 40000 })).unwrap_err();
    assert!(error.to_string().contains("eos_token_id"), "{error}");
    assert_eq!(LlamaVariant::from_model_type("llama4"), None);
    assert_eq!(LlamaVariant::MODEL_TYPES.len(), LlamaVariant::ALL.len());
}

/// Nested wrappers select the text config as for every family.
#[test]
fn wrapped_text_configs_are_selected() {
    let wrapped = serde_json::json!({ "text_config": {
        "model_type": "olmo2", "vocab_size": 64, "pad_token_id": null, "eos_token_id": null,
    } });
    let config = parse(wrapped).unwrap();
    assert_eq!(config.variant(), Some(LlamaVariant::Olmo2));
    assert_eq!(config.vocab_size, 64);
}
