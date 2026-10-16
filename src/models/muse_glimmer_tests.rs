//! Configuration tests for Muse schedules, rotary parameters, and local-window requirements.

use super::*;

/// Default layers repeat three theta-500,000 local blocks followed by a zero-theta global block.
#[test]
fn default_schedule_is_three_rotary_local_layers_then_one_nope_global_layer() {
    let config = MuseGlimmerTextConfig::default();
    let schedule = config.resolved_layer_types();
    let theta = config.resolved_layer_rope_theta();
    assert_eq!(schedule.len(), 52);
    assert_eq!(theta.len(), 52);
    for (index, (layer_type, theta)) in schedule.into_iter().zip(theta).enumerate() {
        if index % 4 == 3 {
            assert_eq!(layer_type, MuseGlimmerLayerType::FullAttention);
            assert_eq!(theta, 0.0);
        } else {
            assert_eq!(layer_type, MuseGlimmerLayerType::SlidingAttention);
            assert_eq!(theta, 500_000.0);
        }
    }
}

/// Explicit head dimensions and full-only schedules remain independent of a window.
#[test]
fn explicit_schedule_and_head_width_allow_architecture_experiments() {
    let config = MuseGlimmerTextConfig {
        hidden_size: 10,
        num_hidden_layers: 4,
        num_attention_heads: 3,
        num_key_value_heads: 1,
        head_dim: 4,
        sliding_window: None,
        layer_types: vec![MuseGlimmerLayerType::FullAttention; 4],
        layer_rope_theta: vec![500_000.0; 4],
        ..MuseGlimmerTextConfig::default()
    };
    config.validate().unwrap();
    assert_eq!(config.resolved_sliding_window(), None);

    let unused_window = MuseGlimmerTextConfig {
        sliding_window: Some(0),
        ..config
    };
    unused_window.validate().unwrap();
    assert_eq!(unused_window.resolved_sliding_window(), None);
}

/// A local layer still requires a nonzero cache and mask window.
#[test]
fn sliding_schedule_requires_a_valid_window() {
    let config = MuseGlimmerTextConfig {
        num_hidden_layers: 1,
        sliding_window: None,
        layer_types: vec![MuseGlimmerLayerType::SlidingAttention],
        layer_rope_theta: vec![500_000.0],
        ..MuseGlimmerTextConfig::default()
    };
    assert!(config.validate().is_err());

    let config = MuseGlimmerTextConfig {
        sliding_window: Some(0),
        ..config
    };
    assert!(config.validate().is_err());
}

/// Negative per-layer theta is rejected even when the schedule length is valid.
#[test]
fn rejects_a_negative_per_layer_rope_theta() {
    let config = MuseGlimmerTextConfig {
        num_hidden_layers: 4,
        layer_rope_theta: vec![500_000.0, 500_000.0, 500_000.0, -1.0],
        ..MuseGlimmerTextConfig::default()
    };
    assert!(config.validate().is_err());
}

/// Asserts that `config` fails validation at exactly `field` with exactly `reason`.
fn assert_rejected(config: &MuseGlimmerTextConfig, field: &str, reason: &str) {
    let error = config
        .validate()
        .expect_err("the configuration must be rejected");
    assert_eq!(
        (error.architecture, error.field, error.reason.as_str()),
        ("muse_glimmer", field, reason),
        "unexpected diagnostic for {config:?}"
    );
}

/// Returns the published defaults with one edit applied.
fn edited(edit: impl FnOnce(&mut MuseGlimmerTextConfig)) -> MuseGlimmerTextConfig {
    let mut config = MuseGlimmerTextConfig::default();
    edit(&mut config);
    config
}

/// Every extent, activation, scale, schedule, window, theta, and token ID fails by field.
#[test]
fn every_invalid_field_is_rejected_with_its_exact_diagnostic() {
    let nonzero = "must be nonzero";
    let positive = "must be finite and greater than zero";
    let cases: Vec<(MuseGlimmerTextConfig, &str, &str)> = vec![
        (
            edited(|c| c.model_type = "muse_glimmer".into()),
            "model_type",
            r#"expected one of ["muse_glimmer_text"], found "muse_glimmer""#,
        ),
        (edited(|c| c.vocab_size = 0), "vocab_size", nonzero),
        (edited(|c| c.hidden_size = 0), "hidden_size", nonzero),
        (
            edited(|c| c.intermediate_size = 0),
            "intermediate_size",
            nonzero,
        ),
        (
            edited(|c| c.num_hidden_layers = 0),
            "num_hidden_layers",
            nonzero,
        ),
        (edited(|c| c.head_dim = 0), "head_dim", nonzero),
        (
            edited(|c| c.max_position_embeddings = 0),
            "max_position_embeddings",
            nonzero,
        ),
        (
            edited(|c| c.num_attention_heads = 3),
            "num_attention_heads",
            "must be divisible by num_key_value_heads",
        ),
        (
            edited(|c| c.head_dim = 127),
            "head_dim",
            "RoPE rotates pairs, so head_dim must be even",
        ),
        (
            edited(|c| c.hidden_activation = "gelu".into()),
            "hidden_activation",
            "released Muse Glimmer text blocks require SiLU/SwiGLU",
        ),
        (
            edited(|c| c.initializer_range = 0.0),
            "initializer_range",
            positive,
        ),
        (edited(|c| c.rms_norm_eps = -1.0), "rms_norm_eps", positive),
        (
            edited(|c| c.post_norm_eps = f64::NAN),
            "post_norm_eps",
            positive,
        ),
        (
            edited(|c| c.final_logit_softcapping = 0.0),
            "final_logit_softcapping",
            positive,
        ),
        (
            edited(|c| c.qk_scale_factor = f64::INFINITY),
            "qk_scale_factor",
            positive,
        ),
        (
            edited(|c| c.output_multiplier = -0.5),
            "output_multiplier",
            positive,
        ),
        (
            edited(|c| c.attention_dropout = 2.0),
            "attention_dropout",
            "must be finite and in [0, 1]",
        ),
        (
            edited(|c| c.layer_types = vec![MuseGlimmerLayerType::FullAttention; 3]),
            "layer_types",
            "layer_types and layer_rope_theta must match num_hidden_layers",
        ),
        (
            edited(|c| c.layer_rope_theta = vec![0.0; 51]),
            "layer_types",
            "layer_types and layer_rope_theta must match num_hidden_layers",
        ),
        (
            edited(|c| c.sliding_window = Some(131_073)),
            "sliding_window",
            "cannot exceed max_position_embeddings",
        ),
        (
            edited(|c| {
                c.layer_rope_theta = vec![500_000.0; 52];
                c.layer_rope_theta[7] = f64::NAN;
            }),
            "layer_rope_theta",
            "values must be finite and non-negative; zero denotes NoPE",
        ),
        (
            edited(|c| c.pad_token_id = Some(TokenIds::One(202_048))),
            "pad_token_id",
            "contains an ID outside the vocabulary",
        ),
        (
            edited(|c| c.bos_token_id = Some(TokenIds::Many(Vec::new()))),
            "bos_token_id",
            "must contain at least one token ID",
        ),
        (
            edited(|c| c.eos_token_id = Some(TokenIds::Many(vec![1, 202_048]))),
            "eos_token_id",
            "contains an ID outside the vocabulary",
        ),
    ];
    for (config, field, reason) in &cases {
        assert_rejected(config, field, reason);
    }
    // Invalid RoPE metadata is reported by the shared RoPE validator.
    assert_rejected(
        &edited(|c| {
            c.rope_parameters = Some(RopeParameters {
                rope_type: crate::models::config::RopeType::Linear,
                rope_theta: Some(500_000.0),
                ..RopeParameters::default()
            });
        }),
        "rope.factor",
        "is required by the selected scaling algorithm",
    );
    MuseGlimmerTextConfig::default().validate().unwrap();
    assert_eq!(
        <MuseGlimmerTextConfig as ModelConfig>::MODEL_TYPES,
        &["muse_glimmer_text"]
    );
    assert_eq!(
        <MuseGlimmerTextConfig as ModelConfig>::FAMILY,
        "muse_glimmer"
    );
}

/// Without `rope_parameters` the global theta falls back to Transformers' default.
#[test]
fn absent_rope_parameters_use_the_transformers_default_theta() {
    let config = MuseGlimmerTextConfig {
        num_hidden_layers: 4,
        rope_parameters: None,
        ..MuseGlimmerTextConfig::default()
    };
    config.validate().unwrap();
    assert_eq!(config.global_rope_theta(), 10_000.0);
    assert_eq!(
        config.resolved_layer_rope_theta(),
        [10_000.0, 10_000.0, 10_000.0, 0.0]
    );
    assert_eq!(config.resolved_sliding_window(), Some(2048));
}
