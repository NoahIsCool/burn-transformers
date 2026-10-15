//! Serialized schema: reference defaults, null versus absent keys, the
//! released gpt-oss-20b config, and round trips.

use super::*;
use crate::models::config::{ConfigLoadError, RopeType};

/// openai/gpt-oss-20b `config.json` at revision 6cee5e81ee83917806bbde320786a8fb61efebee.
pub(in crate::models::gpt_oss) const GPT_OSS_20B: &str = r#"{
  "architectures": ["GptOssForCausalLM"], "attention_bias": true, "attention_dropout": 0.0,
  "eos_token_id": 200002, "experts_per_token": 4, "head_dim": 64, "hidden_act": "silu",
  "hidden_size": 2880, "initial_context_length": 4096, "initializer_range": 0.02,
  "intermediate_size": 2880,
  "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention",
    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
    "sliding_attention", "full_attention", "sliding_attention", "full_attention"],
  "max_position_embeddings": 131072, "model_type": "gpt_oss", "num_attention_heads": 64,
  "num_experts_per_tok": 4, "num_hidden_layers": 24, "num_key_value_heads": 8,
  "num_local_experts": 32, "output_router_logits": false, "pad_token_id": 199999,
  "quantization_config": {
    "modules_to_not_convert": ["model.layers.*.self_attn", "model.layers.*.mlp.router",
      "model.embed_tokens", "lm_head"],
    "quant_method": "mxfp4"
  },
  "rms_norm_eps": 1e-05,
  "rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0,
    "original_max_position_embeddings": 4096, "rope_type": "yarn", "truncate": false},
  "rope_theta": 150000, "router_aux_loss_coef": 0.9, "sliding_window": 128,
  "swiglu_limit": 7.0, "tie_word_embeddings": false, "transformers_version": "4.55.0.dev0",
  "use_cache": true, "vocab_size": 201088
}"#;

/// Parses and validates a config document.
fn parse(json: serde_json::Value) -> Result<GptOssConfig, ConfigLoadError> {
    GptOssConfig::from_json_str(&json.to_string())
}

/// Absent fields take the reference constructor's defaults.
#[test]
fn absent_fields_take_the_reference_defaults() {
    let config: GptOssConfig =
        serde_json::from_value(serde_json::json!({ "model_type": "gpt_oss" })).unwrap();
    assert_eq!(config, GptOssConfig::default());
    assert_eq!(config.vocab_size, 201_088);
    assert_eq!(config.hidden_size, 2880);
    assert_eq!(config.intermediate_size, 2880);
    assert_eq!(config.num_hidden_layers, 36);
    assert_eq!(config.num_attention_heads, 64);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.head_dim, 64);
    assert_eq!(config.num_local_experts, 128);
    assert_eq!(config.num_experts_per_tok, 4);
    assert_eq!(config.sliding_window, Some(128));
    assert_eq!(config.layer_types, GptOssConfig::default_layer_types(36));
    assert!(!config.tie_word_embeddings);
    assert_eq!(config.hidden_act, "silu");
    assert_eq!(config.max_position_embeddings, 131_072);
    assert_eq!(config.rms_norm_eps, 1e-5);
    assert_eq!(
        config.rope_parameters,
        GptOssConfig::default_rope_parameters()
    );
    assert_eq!(config.rope_theta, None);
    assert_eq!(config.attention_dropout, 0.0);
    assert!(config.attention_bias);
    assert_eq!(config.router_aux_loss_coef, 0.9);
    assert!(!config.output_router_logits);
    assert_eq!(config.swiglu_limit, None);
    assert_eq!(config.pad_token_id, None);
    assert_eq!(config.bos_token_id, None);
    assert_eq!(config.eos_token_id, None);
    assert_eq!(config.quantization_config, None);
    assert!(config.architecture().is_ok());
}

/// The default schedule slides even layers and attends globally on odd ones.
#[test]
fn default_layer_types_alternate_sliding_and_full() {
    assert_eq!(
        GptOssConfig::default_layer_types(4),
        [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
        ]
    );
    assert!(GptOssConfig::default_layer_types(0).is_empty());
}

/// The default RoPE parameters are the reference's untruncated YaRN.
#[test]
fn default_rope_parameters_are_untruncated_yarn() {
    let rope = GptOssConfig::default_rope_parameters();
    assert_eq!(rope.rope_type, RopeType::Yarn);
    assert_eq!(rope.factor, Some(32.0));
    assert_eq!(rope.beta_fast, Some(32.0));
    assert_eq!(rope.beta_slow, Some(1.0));
    assert_eq!(rope.truncate, Some(false));
    assert_eq!(rope.original_max_position_embeddings, Some(4096));
    assert_eq!(rope.rope_theta, None);
    assert_eq!(GptOssConfig::default_rope_theta(), 150_000.0);
}

/// The released gpt-oss-20b configuration parses, keeping every serialized value.
#[test]
fn released_20b_config_parses() {
    let config = GptOssConfig::from_json_str(GPT_OSS_20B).unwrap();
    assert_eq!(config.num_hidden_layers, 24);
    assert_eq!(config.num_local_experts, 32);
    assert_eq!(config.num_experts_per_tok, 4);
    assert_eq!(config.layer_types, GptOssConfig::default_layer_types(24));
    assert_eq!(config.rope_theta, Some(150_000.0));
    assert_eq!(config.rope_parameters.rope_type, RopeType::Yarn);
    assert_eq!(config.rope_parameters.truncate, Some(false));
    assert_eq!(config.swiglu_limit, Some(7.0));
    assert_eq!(config.pad_token_id, Some(TokenIds::One(199_999)));
    assert_eq!(config.eos_token_id, Some(TokenIds::One(200_002)));
    assert_eq!(
        config.quantization_config.as_ref().unwrap()["quant_method"],
        "mxfp4"
    );
}

/// JSON `null` derives KV heads and head width, and disables the window.
#[test]
fn null_fields_follow_the_constructor() {
    let config: GptOssConfig = serde_json::from_value(serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 64,
        "num_attention_heads": 8,
        "num_key_value_heads": null,
        "head_dim": null,
        "sliding_window": null,
        "layer_types": null,
        "num_hidden_layers": 2,
        "quantization_config": null,
    }))
    .unwrap();
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.head_dim, 8);
    assert_eq!(config.sliding_window, None);
    assert_eq!(config.layer_types, GptOssConfig::default_layer_types(2));
    assert_eq!(config.quantization_config, None);
    // Without a window the default schedule's sliding layer is invalid.
    assert!(config.architecture().is_err());
    // Zero heads make the derived head width zero, which validation rejects.
    let zero: GptOssConfig = serde_json::from_value(serde_json::json!({
        "model_type": "gpt_oss", "num_attention_heads": 0, "head_dim": null,
    }))
    .unwrap();
    assert_eq!(zero.head_dim, 0);
    assert!(zero.architecture().is_err());
}

/// `rope_parameters` and the legacy `rope_scaling` key are interchangeable.
#[test]
fn rope_scaling_is_an_alias_of_rope_parameters() {
    let nested = serde_json::json!({"rope_type": "yarn", "factor": 8.0,
        "original_max_position_embeddings": 4096, "rope_theta": 10000.0});
    let modern: GptOssConfig = serde_json::from_value(
        serde_json::json!({"model_type": "gpt_oss", "rope_parameters": nested}),
    )
    .unwrap();
    let legacy: GptOssConfig = serde_json::from_value(
        serde_json::json!({"model_type": "gpt_oss", "rope_scaling": nested}),
    )
    .unwrap();
    assert_eq!(modern, legacy);
    assert_eq!(modern.rope_parameters.factor, Some(8.0));
}

/// Serialization writes every value and parses back to an equal configuration.
#[test]
fn configs_round_trip_through_json() {
    for config in [
        GptOssConfig::default(),
        GptOssConfig::from_json_str(GPT_OSS_20B).unwrap(),
    ] {
        let text = serde_json::to_string(&config).unwrap();
        let again: GptOssConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(config, again);
    }
}

/// Parsing validates: semantic errors surface as configuration errors.
#[test]
fn parsing_validates_the_architecture() {
    assert!(parse(serde_json::json!({ "model_type": "gpt_oss" })).is_ok());
    assert!(matches!(
        parse(serde_json::json!({ "model_type": "llama" })),
        Err(ConfigLoadError::Validation(_))
    ));
    assert!(matches!(
        parse(serde_json::json!({ "model_type": "gpt_oss", "hidden_size": "wide" })),
        Err(ConfigLoadError::Json(_))
    ));
}
