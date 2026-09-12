//! Integration tests for official-family JSON compatibility and validation.

use transformers::models::config::{ModelConfig, Validate};
use transformers::models::deepseek_v2::DeepseekV2Config;
use transformers::models::muse_glimmer::{MuseGlimmerLayerType, MuseGlimmerTextConfig};
use transformers::models::nemotron_h::{NemotronHBlockType, NemotronHConfig};
use transformers::models::qwen3_5::{Qwen3_5LayerType, Qwen38TextConfig};
use transformers::models::qwen3_moe::Qwen3MoeConfig;
use transformers::models::t5::{T5Config, T5FeedForwardProjection};

/// Serializes a normalized config, parses it again, and compares every field.
fn assert_normalized_round_trip<T>(config: &T)
where
    T: ModelConfig + serde::Serialize + std::fmt::Debug + PartialEq,
{
    let json = serde_json::to_string(config).unwrap();
    let reparsed = T::from_json_str(&json).unwrap();
    assert_eq!(&reparsed, config);
}

/// Nested Qwen3.8 JSON preserves layer count, projection width, full attention, and MTP depth.
#[test]
fn parses_nested_qwen38_text_topology() {
    let config = Qwen38TextConfig::from_json_str(include_str!("fixtures/qwen3_8_27b.json"))
        .expect("official-family Qwen3.8 config should parse");
    assert_eq!(config.num_hidden_layers, 64);
    assert_eq!(config.query_projection_width(), 12_288);
    assert_eq!(
        config.resolved_layer_types()[3],
        Qwen3_5LayerType::FullAttention
    );
    assert_eq!(config.mtp_num_hidden_layers, 1);
}

/// Qwen3 A3B JSON preserves expert count, top-k routing, and the first MoE layer.
#[test]
fn parses_qwen3_a3b_routing_topology() {
    let config =
        Qwen3MoeConfig::from_json_str(include_str!("fixtures/qwen3_30b_a3b.json")).unwrap();
    assert_eq!(config.num_experts, 128);
    assert_eq!(config.num_experts_per_tok, 8);
    assert!(config.is_moe_layer(0));
}

/// DeepSeek V2 JSON preserves the MLA head width, latent KV rank, and MoE schedule.
#[test]
fn parses_deepseek_v2_mla_and_yarn() {
    let config =
        DeepseekV2Config::from_json_str(include_str!("fixtures/deepseek_v2.json")).unwrap();
    assert_eq!(config.qk_head_dim(), 192);
    assert_eq!(config.kv_lora_rank, 512);
    assert!(config.is_moe_layer(1));
}

/// Nested Muse text JSON derives the final full-attention layer with RoPE disabled.
#[test]
fn parses_nested_muse_text_and_derives_nope_schedule() {
    let config =
        MuseGlimmerTextConfig::from_json_str(include_str!("fixtures/muse_glimmer.json")).unwrap();
    assert_eq!(config.resolved_layer_types().len(), 52);
    assert_eq!(
        config.resolved_layer_types()[51],
        MuseGlimmerLayerType::FullAttention
    );
    assert_eq!(config.resolved_layer_rope_theta()[51], 0.0);
}

/// FLAN and UL2 JSON retain their distinct gated projections and attention dimensions.
#[test]
fn parses_flan_and_ul2_as_t5_topologies() {
    let flan = T5Config::from_json_str(include_str!("fixtures/flan_t5_xl.json")).unwrap();
    assert_eq!(flan.feed_forward_proj, T5FeedForwardProjection::GatedGelu);
    assert_eq!(flan.attention_inner_dim(), 2048);

    let ul2 = T5Config::from_json_str(include_str!("fixtures/ul2.json")).unwrap();
    assert_eq!(ul2.feed_forward_proj, T5FeedForwardProjection::GatedSilu);
    assert_eq!(ul2.decoder_layers(), 32);
}

/// Legacy Nemotron syntax expands the block pattern and accepts an unbounded time-step limit.
#[test]
fn parses_legacy_nemotron_pattern_and_nonstandard_infinity() {
    let config =
        NemotronHConfig::from_json_str(include_str!("fixtures/nemotron_h_8b.json")).unwrap();
    assert_eq!(config.block_count(), 52);
    assert!(config.time_step_limit[1].is_none());
    assert!(
        config
            .layers_block_type
            .contains(&NemotronHBlockType::FullAttention)
    );
}

/// Validation rejects inconsistent attention widths, routing, positions, buckets, and schedules.
#[test]
fn rejects_invalid_dimensions_schedules_routing_and_positions() {
    let mut qwen =
        Qwen38TextConfig::from_json_str(include_str!("fixtures/qwen3_8_27b.json")).unwrap();
    qwen.num_key_value_heads = 5;
    assert!(qwen.validate().is_err());

    let mut qwen_moe =
        Qwen3MoeConfig::from_json_str(include_str!("fixtures/qwen3_30b_a3b.json")).unwrap();
    qwen_moe.num_experts_per_tok = 129;
    assert!(qwen_moe.validate().is_err());

    let mut deepseek =
        DeepseekV2Config::from_json_str(include_str!("fixtures/deepseek_v2.json")).unwrap();
    deepseek.qk_rope_head_dim = 63;
    assert!(deepseek.validate().is_err());

    let mut muse =
        MuseGlimmerTextConfig::from_json_str(include_str!("fixtures/muse_glimmer.json")).unwrap();
    muse.sliding_window = Some(muse.max_position_embeddings + 1);
    assert!(muse.validate().is_err());

    let mut t5 = T5Config::from_json_str(include_str!("fixtures/flan_t5_xl.json")).unwrap();
    t5.relative_attention_num_buckets = 31;
    assert!(t5.validate().is_err());

    let mut nemotron =
        NemotronHConfig::from_json_str(include_str!("fixtures/nemotron_h_8b.json")).unwrap();
    nemotron.num_hidden_layers = Some(51);
    assert!(nemotron.validate().is_err());
}

/// Deserialization rejects unsupported Nemotron layer names.
#[test]
fn rejects_unknown_layer_names_during_deserialization() {
    let json = r#"{
        "model_type":"nemotron_h",
        "layers_block_type":["linear_attention","magic_attention"]
    }"#;
    assert!(NemotronHConfig::from_json_str(json).is_err());
}

/// Normalized family configs survive JSON round trips without changing any fields.
#[test]
fn normalized_schemas_round_trip_without_losing_topology() {
    let qwen = Qwen38TextConfig::from_json_str(include_str!("fixtures/qwen3_8_27b.json")).unwrap();
    assert_normalized_round_trip(&qwen);

    let qwen_moe =
        Qwen3MoeConfig::from_json_str(include_str!("fixtures/qwen3_30b_a3b.json")).unwrap();
    assert_normalized_round_trip(&qwen_moe);

    let deepseek =
        DeepseekV2Config::from_json_str(include_str!("fixtures/deepseek_v2.json")).unwrap();
    assert_normalized_round_trip(&deepseek);

    let muse =
        MuseGlimmerTextConfig::from_json_str(include_str!("fixtures/muse_glimmer.json")).unwrap();
    assert_normalized_round_trip(&muse);

    let t5 = T5Config::from_json_str(include_str!("fixtures/ul2.json")).unwrap();
    assert_normalized_round_trip(&t5);

    let nemotron =
        NemotronHConfig::from_json_str(include_str!("fixtures/nemotron_h_8b.json")).unwrap();
    assert_normalized_round_trip(&nemotron);
}
