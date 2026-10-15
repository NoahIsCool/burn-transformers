//! The translation table: the released checkpoint's topology, the shared
//! components it selects, YaRN parity with the reference, and every rejection.

use super::super::config::tests::GPT_OSS_20B;
use super::*;
use crate::{
    attention::QueryKeyNormScope,
    models::config::{ModelConfig, RopeType, TokenIds, Validate},
    position::RotaryFrequencies,
};

/// Returns the released gpt-oss-20b configuration.
fn released() -> GptOssConfig {
    GptOssConfig::from_json_str(GPT_OSS_20B).unwrap()
}

/// Asserts that `config` is rejected at `field`.
fn rejects(config: &GptOssConfig, field: &str) {
    let error = config.architecture().unwrap_err();
    assert_eq!(error.architecture, FAMILY);
    assert_eq!(error.field, field, "{error}");
}

/// The released checkpoint translates to its documented topology.
#[test]
fn released_20b_topology() {
    let architecture = released().architecture().unwrap();
    assert_eq!(architecture.num_layers(), 24);
    assert_eq!(architecture.vocab_size, 201_088);
    assert_eq!(architecture.hidden_size, 2880);
    assert_eq!(architecture.intermediate_size, 2880);
    assert_eq!(
        (
            architecture.query_heads,
            architecture.key_value_heads,
            architecture.head_dim
        ),
        (64, 8, 64)
    );
    assert!(architecture.attention_bias);
    assert_eq!(architecture.weight_format, GptOssWeightFormat::Mxfp4);
    assert_eq!(architecture.sliding_window(), Some(128));
    for (layer, connectivity) in architecture.layers.iter().enumerate() {
        let expected = if layer % 2 == 0 {
            AttentionConnectivity::SlidingWindow { window: 128 }
        } else {
            AttentionConnectivity::Global
        };
        assert_eq!(*connectivity, expected, "layer {layer}");
    }
    assert_eq!(
        (architecture.experts, architecture.experts_per_token),
        (32, 4)
    );
    assert_eq!(
        architecture.glu,
        ClampedSwiGlu::new(7.0)
            .with_alpha(1.702)
            .with_up_offset(1.0)
    );
    assert_eq!(architecture.norm_epsilon, 1e-5);
}

/// Attention has biases on all four projections, sinks, and no Q/K norms.
#[test]
fn attention_config_selects_sinks_and_biases() {
    let architecture = released().architecture().unwrap();
    let attention = architecture.attention_config();
    assert!(attention.attention_sinks);
    assert_eq!(
        attention.effective_projection_bias(),
        ProjectionBias::uniform(true)
    );
    assert_eq!(attention.query_key_norm_epsilon, None);
    assert_eq!(attention.query_key_norm_scope, QueryKeyNormScope::PerHead);
    assert_eq!(attention.scale, None);
    let unbiased = GptOssConfig {
        attention_bias: false,
        ..released()
    }
    .architecture()
    .unwrap();
    assert_eq!(
        unbiased.attention_config().effective_projection_bias(),
        ProjectionBias::uniform(false)
    );
}

/// The router ranks raw biased logits and normalizes only the selected ones.
#[test]
fn router_config_is_top_k_then_softmax() {
    let router = released().architecture().unwrap().router_config();
    assert_eq!(router.experts, 32);
    assert_eq!(router.experts_per_token, 4);
    assert_eq!(router.policy.scoring, RouterScoring::Softmax);
    assert_eq!(router.policy.order, ScoringOrder::TopKThenScore);
    assert!(router.policy.router_bias);
    assert!(!router.policy.correction_bias);
    assert_eq!(
        router.policy.projection,
        RouterProjectionPrecision::ModelDtype
    );
    assert!(!router.renormalize_selected);
    assert_eq!(router.routed_scale, 1.0);
    assert!(router.validate().is_ok());
}

/// RMSNorm scales in F32 before one rounding and multiplies by the reciprocal root.
#[test]
fn rms_norm_matches_the_reference_operation_order() {
    let norm = released().architecture().unwrap().rms_norm(2880);
    assert!(!norm.round_before_scale);
    assert_eq!(norm.root_mode, RmsNormRootMode::MultiplyByReciprocalSqrt);
    assert_eq!(norm.epsilon, 1e-5);
}

/// YaRN frequencies and amplitude match Transformers 5.2 `GptOssRotaryEmbedding`.
///
/// Reference values were computed from the released config with the
/// pinned Transformers; F32 frequencies agree to a relative `1e-6`.
#[test]
fn yarn_frequencies_match_the_reference() {
    /// Transformers 5.2.0 `inv_freq` of the released gpt-oss-20b config.
    const REFERENCE: [f64; 32] = [
        1.0,
        0.689_044_296_741_485_6,
        0.474_782_049_655_914_3,
        0.327_145_874_500_274_66,
        0.225_418_001_413_345_34,
        0.155_322_983_860_969_54,
        0.107_024_423_778_057_1,
        0.073_744_565_248_489_38,
        0.050_813_272_595_405_58,
        0.031_705_696_135_759_354,
        0.019_334_999_844_431_877,
        0.011_592_049_151_659_012,
        0.006_794_959_306_716_919,
        0.003_860_359_080_135_822_3,
        0.002_093_792_660_161_853,
        0.001_052_602_194_249_63,
        0.000_456_483_918_242_156_5,
        0.000_129_318_694_234_825_67,
        3.830_881_178_146_228e-5,
        2.639_646_845_636_889_3e-5,
        1.818_833_698_052_913e-5,
        1.253_256_959_898_863e-5,
        8.635_495_760_245_249e-6,
        5.950_239_483_354_38e-6,
        4.099_978_468_730_114e-6,
        2.825_066_758_305_183_6e-6,
        1.946_596_285_051_782_6e-6,
        1.341_290_953_860_152_5e-6,
        9.242_089_618_055_616e-7,
        6.368_209_142_237_902e-7,
        4.387_978_549_402_76e-7,
        3.023_511_396_804_679e-7,
    ];
    let architecture = released().architecture().unwrap();
    assert_eq!(architecture.rope.rotary_dimension(), 64);
    let frequencies = RotaryFrequencies::new(architecture.rope, 1).unwrap();
    assert_eq!(frequencies.inverse().len(), REFERENCE.len());
    for (index, (&ours, &reference)) in frequencies.inverse().iter().zip(&REFERENCE).enumerate() {
        let relative = ((f64::from(ours) - reference) / reference).abs();
        assert!(relative <= 1e-6, "inv_freq[{index}]: {ours} vs {reference}");
    }
    assert!((frequencies.amplitude() - 1.346_573_590_279_972_7).abs() <= 1e-12);
}

/// The default top-level theta applies when neither level serializes one.
#[test]
fn rope_theta_defaults_to_150000() {
    let explicit = GptOssConfig {
        rope_theta: Some(150_000.0),
        ..GptOssConfig::default()
    }
    .architecture()
    .unwrap();
    let implicit = GptOssConfig::default().architecture().unwrap();
    assert_eq!(explicit.rope, implicit.rope);
    let other = GptOssConfig {
        rope_theta: Some(10_000.0),
        ..GptOssConfig::default()
    }
    .architecture()
    .unwrap();
    assert_ne!(other.rope, implicit.rope);
}

/// A dense checkpoint has no quantization directive; only MXFP4 is accepted otherwise.
#[test]
fn weight_format_follows_quantization_config() {
    let dense = GptOssConfig {
        quantization_config: None,
        ..released()
    };
    assert_eq!(
        dense.architecture().unwrap().weight_format,
        GptOssWeightFormat::Dense
    );
    for method in [
        serde_json::json!({"quant_method": "fp8"}),
        serde_json::json!({}),
    ] {
        rejects(
            &GptOssConfig {
                quantization_config: Some(method),
                ..released()
            },
            "quantization_config",
        );
    }
}

/// Settings the reference ignores, or the graph cannot honor, are rejected.
#[test]
fn unsupported_settings_are_rejected() {
    rejects(
        &GptOssConfig {
            model_type: "llama".into(),
            ..released()
        },
        "model_type",
    );
    rejects(
        &GptOssConfig {
            attention_dropout: 0.1,
            ..released()
        },
        "attention_dropout",
    );
    rejects(
        &GptOssConfig {
            hidden_act: "gelu".into(),
            ..released()
        },
        "hidden_act",
    );
    rejects(
        &GptOssConfig {
            swiglu_limit: Some(10.0),
            ..released()
        },
        "swiglu_limit",
    );
    rejects(
        &GptOssConfig {
            rms_norm_eps: 0.0,
            ..released()
        },
        "rms_norm_eps",
    );
}

/// Malformed schedules are rejected with the offending field.
#[test]
fn malformed_layer_schedules_are_rejected() {
    rejects(
        &GptOssConfig {
            layer_types: GptOssConfig::default_layer_types(23),
            ..released()
        },
        "layer_types",
    );
    let mut types = GptOssConfig::default_layer_types(24);
    types[3] = "chunked_attention".into();
    rejects(
        &GptOssConfig {
            layer_types: types,
            ..released()
        },
        "layer_types",
    );
    rejects(
        &GptOssConfig {
            sliding_window: None,
            ..released()
        },
        "sliding_window",
    );
    rejects(
        &GptOssConfig {
            sliding_window: Some(0),
            ..released()
        },
        "sliding_window",
    );
    // A schedule without sliding layers needs no window.
    let global = GptOssConfig {
        sliding_window: None,
        layer_types: vec!["full_attention".into(); 24],
        ..released()
    }
    .architecture()
    .unwrap();
    assert_eq!(global.sliding_window(), None);
}

/// Impossible geometries and routing are rejected.
#[test]
fn impossible_geometry_is_rejected() {
    for (config, field) in [
        (
            GptOssConfig {
                hidden_size: 0,
                ..released()
            },
            "hidden_size",
        ),
        (
            GptOssConfig {
                num_local_experts: 0,
                ..released()
            },
            "num_local_experts",
        ),
        (
            GptOssConfig {
                num_experts_per_tok: 0,
                ..released()
            },
            "num_experts_per_tok",
        ),
        (
            GptOssConfig {
                num_experts_per_tok: 33,
                ..released()
            },
            "num_experts_per_tok",
        ),
        (
            GptOssConfig {
                num_key_value_heads: 7,
                ..released()
            },
            "num_attention_heads",
        ),
        (
            GptOssConfig {
                intermediate_size: usize::MAX / 2,
                ..released()
            },
            "intermediate_size",
        ),
        (
            GptOssConfig {
                pad_token_id: Some(TokenIds::One(201_088)),
                ..released()
            },
            "pad_token_id",
        ),
        (
            GptOssConfig {
                head_dim: 63,
                ..released()
            },
            "rope_parameters",
        ),
    ] {
        rejects(&config, field);
    }
}

/// Validation and the model-config trait agree.
#[test]
fn validate_derives_the_architecture() {
    assert!(released().validate().is_ok());
    assert!(
        GptOssConfig {
            hidden_size: 0,
            ..released()
        }
        .validate()
        .is_err()
    );
    assert_eq!(<GptOssConfig as ModelConfig>::MODEL_TYPES, ["gpt_oss"]);
    assert_eq!(
        released().rope_parameters.rope_type,
        RopeType::Yarn,
        "the fixture is YaRN"
    );
}
