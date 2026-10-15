//! Translation table: schedules, RoPE sets, scales, soft-caps, norms, the
//! embedding multiplier, and every rejected configuration.

use super::*;
use crate::models::config::{ModelConfig, RopeType, TokenIds};
use crate::position::{LinearRopeScaling, RopeScaling};

/// A small valid configuration of `variant` with `L` layers.
fn small(variant: GemmaVariant, layers: usize) -> GemmaConfig {
    GemmaConfig {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 24,
        num_hidden_layers: layers,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        max_position_embeddings: 64,
        sliding_window: 3,
        ..GemmaConfig::new(variant)
    }
}

/// Returns the layer types of an architecture.
fn types(architecture: &GemmaArchitecture) -> Vec<GemmaLayerType> {
    architecture
        .layers
        .iter()
        .map(|layer| layer.layer_type)
        .collect()
}

/// Asserts that `config` is rejected at `field`.
fn rejects(config: &GemmaConfig, field: &str) {
    let error = config.architecture().unwrap_err();
    assert_eq!(error.field, field, "{error}");
}

/// Without `layer_types`, Gemma 2 alternates and Gemma 3 follows its pattern.
#[test]
fn default_schedules_follow_each_reference() {
    use GemmaLayerType::{FullAttention as F, SlidingAttention as S};
    let gemma2 = small(GemmaVariant::Gemma2, 5).architecture().unwrap();
    assert_eq!(types(&gemma2), [S, F, S, F, S]);
    let gemma3 = small(GemmaVariant::Gemma3Text, 13).architecture().unwrap();
    assert_eq!(types(&gemma3), [S, S, S, S, S, F, S, S, S, S, S, F, S]);
    let patterned = GemmaConfig {
        sliding_window_pattern: Some(2),
        ..small(GemmaVariant::Gemma3Text, 4)
    };
    assert_eq!(types(&patterned.architecture().unwrap()), [S, F, S, F]);
    let explicit = GemmaConfig {
        layer_types: Some(vec![
            "full_attention".into(),
            "sliding_attention".into(),
            "full_attention".into(),
        ]),
        ..small(GemmaVariant::Gemma2, 3)
    };
    let explicit = explicit.architecture().unwrap();
    assert_eq!(types(&explicit), [F, S, F]);
    assert_eq!(
        explicit.layers[1].connectivity,
        AttentionConnectivity::SlidingWindow { window: 3 }
    );
    assert_eq!(
        explicit.layers[0].connectivity,
        AttentionConnectivity::Global
    );
    assert_eq!(explicit.sliding_window(), Some(3));
    let global_only = GemmaConfig {
        layer_types: Some(vec!["full_attention".into(); 2]),
        ..small(GemmaVariant::Gemma2, 2)
    };
    assert_eq!(global_only.architecture().unwrap().sliding_window(), None);
    assert_eq!(
        GemmaLayerType::SlidingAttention.as_str(),
        "sliding_attention"
    );
    assert_eq!(GemmaLayerType::FullAttention.as_str(), "full_attention");
}

/// The real Gemma 3 4B text config translates to the published topology.
#[test]
fn gemma_3_4b_translates_to_its_published_topology() {
    let json = r#"{"model_type": "gemma3", "text_config": {
        "hidden_size": 2560, "intermediate_size": 10240, "model_type": "gemma3_text",
        "num_hidden_layers": 34, "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
        "sliding_window": 1024}}"#;
    let architecture = GemmaConfig::from_json_str(json)
        .unwrap()
        .architecture()
        .unwrap();
    let globals: Vec<usize> = (0..34)
        .filter(|&layer| architecture.layers[layer].layer_type == GemmaLayerType::FullAttention)
        .collect();
    assert_eq!(globals, [5, 11, 17, 23, 29]);
    assert_eq!(architecture.sliding_window(), Some(1024));
    assert_eq!(architecture.attention_scale, 1.0 / 16.0);
    assert_eq!(architecture.attention_softcap, None);
    assert_eq!(architecture.final_logit_softcap, None);
    assert!(architecture.query_key_norm && architecture.tie_word_embeddings);
    assert_eq!(architecture.sliding_rope.theta(), 10_000.0);
    assert_eq!(architecture.sliding_rope.scaling(), RopeScaling::None);
    assert_eq!(architecture.full_rope.theta(), 1_000_000.0);
    assert_eq!(
        architecture.full_rope.scaling(),
        RopeScaling::Linear(LinearRopeScaling::new(8.0).unwrap())
    );
    assert_eq!(architecture.full_rope.rotary_dimension(), 256);
    let schedule = architecture.rotary_schedule();
    assert_eq!(
        schedule.sets,
        [architecture.sliding_rope, architecture.full_rope]
    );
    assert_eq!(schedule.layers[5], Some(1));
    assert_eq!(schedule.layers[4], Some(0));
    assert_eq!(schedule.precision, RotaryPrecision::ActivationDtype);
}

/// Gemma 2 rotates every layer with one set; its theta defaults to 1e4.
#[test]
fn gemma_2_shares_one_rope_set() {
    let architecture = small(GemmaVariant::Gemma2, 4).architecture().unwrap();
    assert_eq!(architecture.sliding_rope, architecture.full_rope);
    assert_eq!(architecture.full_rope.theta(), 10_000.0);
    let schedule = architecture.rotary_schedule();
    assert_eq!(schedule.sets.len(), 1);
    assert!(schedule.layers.iter().all(|set| *set == Some(0)));
    let legacy = GemmaConfig {
        rope_theta: Some(500.0),
        rope_scaling: Some(RopeParameters {
            rope_type: RopeType::Linear,
            factor: Some(4.0),
            ..RopeParameters::default()
        }),
        ..small(GemmaVariant::Gemma2, 2)
    };
    let legacy = legacy.architecture().unwrap();
    assert_eq!(legacy.full_rope.theta(), 500.0);
    assert_eq!(
        legacy.full_rope.scaling(),
        RopeScaling::Linear(LinearRopeScaling::new(4.0).unwrap())
    );
    let nested: GemmaConfig = GemmaConfig::from_json_str(
        r#"{"model_type": "gemma2", "num_hidden_layers": 2, "hidden_size": 16,
            "intermediate_size": 24, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 4, "vocab_size": 32,
            "rope_parameters": {"rope_type": "default", "rope_theta": 250.0}}"#,
    )
    .unwrap();
    assert_eq!(nested.architecture().unwrap().full_rope.theta(), 250.0);
}

/// Gemma 3's legacy fields and per-layer mapping produce the local and global sets.
#[test]
fn gemma_3_rope_sets_follow_the_legacy_conversion() {
    let defaults = small(GemmaVariant::Gemma3Text, 6).architecture().unwrap();
    assert_eq!(defaults.sliding_rope.theta(), 10_000.0);
    assert_eq!(defaults.full_rope.theta(), 1_000_000.0);
    let legacy = GemmaConfig {
        rope_theta: Some(2e5),
        rope_local_base_freq: Some(50.0),
        ..small(GemmaVariant::Gemma3Text, 6)
    };
    let legacy = legacy.architecture().unwrap();
    assert_eq!(legacy.sliding_rope.theta(), 50.0);
    assert_eq!(legacy.full_rope.theta(), 2e5);
    let fixture = GemmaConfig::from_json_str(include_str!(
        "../../../tests/fixtures/tiny_goldens/gemma3-text/config.json"
    ))
    .unwrap()
    .architecture()
    .unwrap();
    assert_eq!(fixture.sliding_rope.theta(), 10_000.0);
    assert_eq!(fixture.full_rope.theta(), 1_000_000.0);
    assert_eq!(
        fixture.full_rope.scaling(),
        RopeScaling::Linear(LinearRopeScaling::new(8.0).unwrap())
    );
    // A per-layer mapping with only the global entry defaults the local one.
    let partial: GemmaConfig = GemmaConfig::from_json_str(
        r#"{"model_type": "gemma3_text", "num_hidden_layers": 2, "hidden_size": 16,
            "intermediate_size": 24, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 4, "vocab_size": 32, "rope_local_base_freq": 77.0,
            "rope_parameters": {"full_attention": {"rope_type": "linear", "factor": 2.0}}}"#,
    )
    .unwrap();
    let partial = partial.architecture().unwrap();
    assert_eq!(partial.sliding_rope.theta(), 77.0);
    assert_eq!(partial.full_rope.theta(), 1_000_000.0);
}

/// Ambiguous or silently ignored RoPE fields are rejected per variant.
#[test]
fn rope_fields_the_reference_ignores_are_rejected() {
    let scaling = RopeParameters {
        rope_type: RopeType::Linear,
        factor: Some(2.0),
        ..RopeParameters::default()
    };
    let both = GemmaConfig {
        rope_parameters: Some(RopeParameterSet::Uniform(RopeParameters::default())),
        rope_scaling: Some(scaling.clone()),
        ..small(GemmaVariant::Gemma2, 2)
    };
    rejects(&both, "rope_scaling");
    let by_type = RopeParameterSet::ByLayerType(
        RopeParametersByLayerType::new(BTreeMap::from([(
            "full_attention".to_owned(),
            scaling.clone(),
        )]))
        .unwrap(),
    );
    let gemma2_by_type = GemmaConfig {
        rope_parameters: Some(by_type),
        ..small(GemmaVariant::Gemma2, 2)
    };
    rejects(&gemma2_by_type, "rope_parameters");
    let flat = GemmaConfig {
        rope_parameters: Some(RopeParameterSet::Uniform(scaling.clone())),
        ..small(GemmaVariant::Gemma3Text, 2)
    };
    rejects(&flat, "rope_parameters");
    let empty_flat = GemmaConfig {
        rope_parameters: Some(RopeParameterSet::Uniform(RopeParameters::default())),
        ..small(GemmaVariant::Gemma3Text, 2)
    };
    assert_eq!(
        empty_flat.architecture().unwrap(),
        small(GemmaVariant::Gemma3Text, 2).architecture().unwrap()
    );
    let stray = RopeParameterSet::ByLayerType(
        RopeParametersByLayerType::new(BTreeMap::from([("chunked_attention".to_owned(), scaling)]))
            .unwrap(),
    );
    let stray = GemmaConfig {
        rope_parameters: Some(stray),
        ..small(GemmaVariant::Gemma3Text, 2)
    };
    rejects(&stray, "rope_parameters");
    let local_on_gemma2 = GemmaConfig {
        rope_local_base_freq: Some(10.0),
        ..small(GemmaVariant::Gemma2, 2)
    };
    rejects(&local_on_gemma2, "rope_local_base_freq");
    let partial = GemmaConfig {
        rope_scaling: Some(RopeParameters {
            partial_rotary_factor: Some(0.5),
            ..RopeParameters::default()
        }),
        ..small(GemmaVariant::Gemma2, 2)
    };
    rejects(&partial, "rope_parameters");
}

/// Scale, soft-caps, and Q/K norms reach the shared attention configuration.
#[test]
fn attention_follows_the_scalar_and_soft_caps() {
    let gemma2 = GemmaConfig {
        query_pre_attn_scalar: 144.0,
        attention_bias: true,
        ..small(GemmaVariant::Gemma2, 2)
    }
    .architecture()
    .unwrap();
    let config = gemma2.attention_config();
    assert_eq!(config.scale, Some(1.0 / 12.0));
    assert_eq!(config.softcap, Some(50.0));
    assert_eq!(config.query_key_norm_epsilon, None);
    assert_eq!(
        config.effective_projection_bias(),
        ProjectionBias::uniform(true)
    );
    assert_eq!(gemma2.final_logit_softcap, Some(30.0));
    let gemma3 = small(GemmaVariant::Gemma3Text, 2).architecture().unwrap();
    let config = gemma3.attention_config();
    assert_eq!(config.scale, Some(1.0 / 16.0));
    assert_eq!(config.softcap, None);
    assert_eq!(config.query_key_norm_epsilon, Some(1e-6));
    assert_eq!(config.query_key_norm_scope, QueryKeyNormScope::PerHead);
    assert!(config.query_key_norm_zero_centered);
    assert_eq!(
        config.effective_projection_bias(),
        ProjectionBias::uniform(false)
    );
}

/// Norms are zero-centered, F32, scale-then-cast, with the reciprocal-root graph.
#[test]
fn norms_and_mlp_follow_the_reference_conventions() {
    let architecture = small(GemmaVariant::Gemma2, 2).architecture().unwrap();
    let norm = architecture.rms_norm(16);
    assert!(norm.zero_centered && !norm.round_before_scale);
    assert_eq!(norm.root_mode, RmsNormRootMode::MultiplyByReciprocalSqrt);
    assert_eq!(norm.epsilon, 1e-6);
    let mlp = architecture.mlp_config();
    assert_eq!(mlp.activation, Activation::GeluTanh);
    assert!(!mlp.bias);
    assert_eq!((mlp.hidden_size, mlp.intermediate_size), (16, 24));
    for (name, expected) in [
        ("gelu_new", Activation::GeluTanh),
        ("gelu", Activation::Gelu),
        ("silu", Activation::Silu),
        ("swish", Activation::Silu),
        ("relu", Activation::Relu),
    ] {
        let config = GemmaConfig {
            hidden_activation: name.into(),
            ..small(GemmaVariant::Gemma3Text, 1)
        };
        assert_eq!(
            config.architecture().unwrap().activation,
            expected,
            "{name}"
        );
    }
}

/// The embedding multiplier is `sqrt(d)` rounded to the activation dtype.
#[test]
fn embedding_scale_rounds_like_the_reference() {
    let scale = |hidden_size, dtype| {
        GemmaArchitecture {
            hidden_size,
            ..small(GemmaVariant::Gemma2, 1).architecture().unwrap()
        }
        .embedding_scale(dtype)
    };
    // The reference's own comment: sqrt(3072) = 55.4256 becomes 55.5 in BF16.
    assert_eq!(scale(3072, DType::BF16), 55.5);
    assert_eq!(scale(3072, DType::F16), 55.4375);
    assert_eq!(scale(3072, DType::F32), f64::from(3072.0_f64.sqrt() as f32));
    assert_eq!(scale(2560, DType::BF16), 50.5);
    assert_eq!(scale(16, DType::F32), 4.0);
    assert_eq!(scale(16, DType::F64), 4.0);
}

/// Every directive the graph cannot honor is rejected at its field.
#[test]
fn unsupported_configurations_are_rejected() {
    let base = || small(GemmaVariant::Gemma3Text, 2);
    for (config, field) in [
        (
            GemmaConfig {
                model_type: "gemma3".into(),
                ..base()
            },
            "model_type",
        ),
        (
            GemmaConfig {
                attention_dropout: 0.1,
                ..base()
            },
            "attention_dropout",
        ),
        (
            GemmaConfig {
                quantization_config: Some(serde_json::json!({"bits": 4})),
                ..base()
            },
            "quantization_config",
        ),
        (
            GemmaConfig {
                use_bidirectional_attention: true,
                ..base()
            },
            "use_bidirectional_attention",
        ),
        (
            GemmaConfig {
                attn_logit_softcapping: Some(50.0),
                ..base()
            },
            "attn_logit_softcapping",
        ),
        (
            GemmaConfig {
                final_logit_softcapping: Some(0.0),
                ..base()
            },
            "final_logit_softcapping",
        ),
        (
            GemmaConfig {
                sliding_window_pattern: Some(0),
                ..base()
            },
            "sliding_window_pattern",
        ),
        (
            GemmaConfig {
                sliding_window_pattern: Some(2),
                ..small(GemmaVariant::Gemma2, 2)
            },
            "sliding_window_pattern",
        ),
        (
            GemmaConfig {
                sliding_window: 0,
                ..base()
            },
            "sliding_window",
        ),
        (
            GemmaConfig {
                query_pre_attn_scalar: -1.0,
                ..base()
            },
            "query_pre_attn_scalar",
        ),
        (
            GemmaConfig {
                rms_norm_eps: 0.0,
                ..base()
            },
            "rms_norm_eps",
        ),
        (
            GemmaConfig {
                num_key_value_heads: 3,
                ..base()
            },
            "num_attention_heads",
        ),
        (
            GemmaConfig {
                head_dim: 0,
                ..base()
            },
            "head_dim",
        ),
        (
            GemmaConfig {
                hidden_activation: "tanh".into(),
                ..base()
            },
            "hidden_activation",
        ),
        (
            GemmaConfig {
                layer_types: Some(vec!["full_attention".into()]),
                ..base()
            },
            "layer_types",
        ),
        (
            GemmaConfig {
                layer_types: Some(vec!["chunked_attention".into(); 2]),
                ..base()
            },
            "layer_types",
        ),
        (
            GemmaConfig {
                eos_token_id: Some(TokenIds::One(32)),
                ..base()
            },
            "eos_token_id",
        ),
    ] {
        rejects(&config, field);
    }
}

/// An absurd layer count is rejected before any per-layer allocation.
///
/// Regression: `adversarial_documents_never_panic` found that a
/// `num_hidden_layers` of `u32::MAX` made validation allocate a 4-billion
/// entry schedule (about 100 GiB) and `usize::MAX` overflow the vector's
/// capacity, aborting instead of reporting an error.
#[test]
fn absurd_layer_counts_are_rejected_without_allocating() {
    for layers in [usize::MAX, u32::MAX as usize, MAX_LAYERS + 1] {
        rejects(&small(GemmaVariant::Gemma2, layers), "num_hidden_layers");
    }
    assert_eq!(
        small(GemmaVariant::Gemma3Text, MAX_LAYERS)
            .architecture()
            .unwrap()
            .num_layers(),
        MAX_LAYERS
    );
}
