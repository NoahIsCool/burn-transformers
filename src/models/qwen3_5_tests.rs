//! Configuration checks for Qwen hybrid schedules, projection widths, and unsupported topology.

use super::*;

/// An implicit eight-layer schedule places full attention at every fourth block.
#[test]
fn derives_the_documented_hybrid_schedule() {
    let config = Qwen3_5TextConfig {
        num_hidden_layers: 8,
        layer_types: Vec::new(),
        ..Qwen3_5TextConfig::default()
    };
    assert_eq!(
        config.resolved_layer_types(),
        vec![
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ]
    );
}

/// An explicit all-attention schedule validates independently of the default hybrid cadence.
#[test]
fn explicit_schedule_overrides_the_derived_interval() {
    let config = Qwen3_5TextConfig {
        num_hidden_layers: 4,
        layer_types: vec![Qwen3_5LayerType::FullAttention; 4],
        ..Qwen3_5TextConfig::default()
    };
    config.validate().unwrap();
}

/// An explicit three-block schedule cannot satisfy a declared depth of four.
#[test]
fn rejects_an_explicit_schedule_with_the_wrong_length() {
    let config = Qwen3_5TextConfig {
        num_hidden_layers: 4,
        layer_types: vec![Qwen3_5LayerType::FullAttention; 3],
        ..Qwen3_5TextConfig::default()
    };
    assert!(config.validate().is_err());
}

/// Twenty-four 256-wide query heads require 12,288 channels when output gating is enabled.
#[test]
fn gated_query_projection_has_twice_the_attention_width() {
    let config = Qwen3_5TextConfig {
        num_attention_heads: 24,
        head_dim: 256,
        attn_output_gate: true,
        ..Qwen3_5TextConfig::default()
    };
    assert_eq!(config.query_projection_width(), 12_288);
}

/// A gated query width beyond `usize` fails validation and its accessor saturates.
#[test]
fn rejects_an_unrepresentable_gated_query_width() {
    // `2^(bits - 9)` heads of width 256 need `2^(bits - 1)` query channels,
    // which fits; the gate doubles that to `2^bits`, one past `usize::MAX`.
    let heads = (usize::MAX >> 9) + 1;
    let overflowing = Qwen3_5TextConfig {
        num_attention_heads: heads,
        head_dim: 256,
        attn_output_gate: true,
        ..Qwen3_5TextConfig::default()
    };
    assert_eq!(overflowing.query_projection_width(), usize::MAX);
    let error = overflowing
        .validate()
        .expect_err("an overflowing gated width must not validate");
    assert_eq!(error.field, "num_attention_heads");

    // Without the gate the same heads are representable and exact.
    let ungated = Qwen3_5TextConfig {
        attn_output_gate: false,
        ..overflowing
    };
    assert_eq!(ungated.query_projection_width(), heads * 256);
    ungated
        .validate()
        .expect("a representable ungated width validates");
}

/// Serialized aliases for the implemented SiLU DeltaNet gate remain compatible.
#[test]
fn accepts_only_implemented_delta_gate_spellings() {
    for gate in [None, Some("silu"), Some("swish")] {
        let config = Qwen3_5TextConfig {
            output_gate_type: gate.map(str::to_owned),
            ..Qwen3_5TextConfig::default()
        };
        config.validate().expect("SiLU spelling must validate");
    }

    let config = Qwen3_5TextConfig {
        output_gate_type: Some("sigmoid".into()),
        ..Qwen3_5TextConfig::default()
    };
    let error = config
        .validate()
        .expect_err("an unimplemented gate must fail closed");
    assert_eq!(error.field, "output_gate_type");
}

/// Runtime-affecting legacy fields cannot be accepted and then ignored.
#[test]
fn rejects_unimplemented_recurrent_and_layer_topologies() {
    let cases = [
        Qwen3_5TextConfig {
            mamba_ssm_dtype: Some("float16".into()),
            ..Qwen3_5TextConfig::default()
        },
        Qwen3_5TextConfig {
            mlp_only_layers: vec![0],
            ..Qwen3_5TextConfig::default()
        },
        Qwen3_5TextConfig {
            mtp_use_dedicated_embeddings: true,
            ..Qwen3_5TextConfig::default()
        },
    ];
    let expected_fields = [
        "mamba_ssm_dtype",
        "mlp_only_layers",
        "mtp_use_dedicated_embeddings",
    ];

    for (config, expected_field) in cases.into_iter().zip(expected_fields) {
        let error = config
            .validate()
            .expect_err("unimplemented topology must fail closed");
        assert_eq!(error.field, expected_field);
    }
}

/// Builds the nested RoPE object of a config that rotates `factor` of each head.
fn nested_partial(factor: Option<f64>) -> Option<RopeParameters> {
    Some(RopeParameters {
        partial_rotary_factor: factor,
        ..RopeParameters::default()
    })
}

/// A nested partial factor without a top-level value is honoured (QW-13).
///
/// The Qwen default of 0.25 is Transformers' `setdefault`, the weakest
/// source. It previously filled the top-level field at deserialization and
/// was then compared against the nested value as if a user had written it,
/// rejecting `rope_parameters.partial_rotary_factor = 0.5`.
#[test]
fn nested_partial_rotary_factor_wins_over_the_family_default() {
    let config = Qwen3_5TextConfig {
        head_dim: 16,
        rope_parameters: nested_partial(Some(0.5)),
        ..Qwen3_5TextConfig::default()
    };
    config
        .validate()
        .expect("a nested factor alone must validate");
    let rope = config.rope_config().expect("nested factor translates");
    assert_eq!(rope.rotary_dimension(), 8);

    let parsed: Qwen3_5TextConfig = serde_json::from_str(
        r#"{"head_dim": 16, "rope_parameters": {"rope_type": "default", "partial_rotary_factor": 0.5}}"#,
    )
    .expect("config JSON without a top-level factor parses");
    assert_eq!(parsed.partial_rotary_factor, None);
    parsed.validate().expect("parsed nested factor validates");
    assert_eq!(parsed.rope_config().unwrap().rotary_dimension(), 8);
}

/// The partial factor resolves nested, then top level, then the family default.
#[test]
fn partial_rotary_factor_resolution_follows_transformers_precedence() {
    let resolve = |nested: Option<f64>, top_level: Option<f64>| {
        let config = Qwen3_5TextConfig {
            head_dim: 16,
            rope_parameters: nested_partial(nested),
            partial_rotary_factor: top_level,
            ..Qwen3_5TextConfig::default()
        };
        config.validate().map(|()| config.rope_config())
    };
    // Neither source: the Qwen default rotates a quarter of each head.
    assert_eq!(
        resolve(None, None).unwrap().unwrap().rotary_dimension(),
        4,
        "default 0.25 of 16 channels"
    );
    assert_eq!(
        Qwen3_5TextConfig {
            head_dim: 16,
            ..Qwen3_5TextConfig::default()
        }
        .rope_config()
        .unwrap()
        .rotary_dimension(),
        4,
        "an absent rope_parameters object also uses the default"
    );
    // A top-level value alone, and agreeing values, are accepted.
    assert_eq!(
        resolve(None, Some(1.0))
            .unwrap()
            .unwrap()
            .rotary_dimension(),
        16
    );
    assert_eq!(
        resolve(Some(0.5), Some(0.5))
            .unwrap()
            .unwrap()
            .rotary_dimension(),
        8
    );
    // Two serialized values that disagree are rejected rather than guessed.
    let error = resolve(Some(0.5), Some(0.25)).expect_err("serialized disagreement");
    assert_eq!(error.field, "partial_rotary_factor");
}

/// Every independent topology rule reports its own serialized field and reason.
///
/// Each case perturbs the valid default topology in one place, so the error
/// proves that rule (not an earlier one) rejected it.
#[test]
fn each_topology_rule_names_its_field_and_reason() {
    let base = Qwen3_5TextConfig::default;
    base().validate().expect("the default topology validates");
    let cases: Vec<(Qwen3_5TextConfig, &str, &str)> = vec![
        (
            Qwen3_5TextConfig {
                model_type: "qwen3_5".into(),
                ..base()
            },
            "model_type",
            "expected one of [\"qwen3_5_text\"], found \"qwen3_5\"",
        ),
        (
            Qwen3_5TextConfig {
                linear_conv_kernel_dim: 0,
                ..base()
            },
            "linear_conv_kernel_dim",
            "must be nonzero",
        ),
        (
            Qwen3_5TextConfig {
                num_attention_heads: 6,
                ..base()
            },
            "num_attention_heads",
            "must be divisible by num_key_value_heads",
        ),
        (
            Qwen3_5TextConfig {
                linear_num_value_heads: 24,
                linear_num_key_heads: 16,
                ..base()
            },
            "linear_num_value_heads",
            "must be divisible by linear_num_key_heads",
        ),
        (
            Qwen3_5TextConfig {
                initializer_range: f64::NAN,
                ..base()
            },
            "initializer_range",
            "must be finite and greater than zero",
        ),
        (
            Qwen3_5TextConfig {
                hidden_act: "gelu".into(),
                ..base()
            },
            "hidden_act",
            "only the released SiLU/SwiGLU topology is supported",
        ),
        (
            Qwen3_5TextConfig {
                rope_theta: Some(0.5),
                ..base()
            },
            "rope_theta",
            "must be finite and at least one",
        ),
        (
            Qwen3_5TextConfig {
                rope_parameters: Some(RopeParameters {
                    mrope_section: Some(vec![16, 16]),
                    ..RopeParameters::default()
                }),
                ..base()
            },
            "mrope_section",
            "Qwen multimodal RoPE requires temporal, height, and width sections",
        ),
        (
            Qwen3_5TextConfig {
                num_hidden_layers: 4,
                mtp_num_hidden_layers: 5,
                ..base()
            },
            "mtp_num_hidden_layers",
            "cannot exceed the target decoder depth",
        ),
        (
            Qwen3_5TextConfig {
                pad_token_id: Some(TokenIds::One(248_320)),
                ..base()
            },
            "pad_token_id",
            "contains an ID outside the vocabulary",
        ),
        (
            Qwen3_5TextConfig {
                bos_token_id: Some(TokenIds::Many(Vec::new())),
                ..base()
            },
            "bos_token_id",
            "must contain at least one token ID",
        ),
        (
            Qwen3_5TextConfig {
                eos_token_id: Some(TokenIds::Many(vec![0, 248_320])),
                ..base()
            },
            "eos_token_id",
            "contains an ID outside the vocabulary",
        ),
    ];
    for (config, field, reason) in cases {
        let error = config
            .validate()
            .expect_err("the mutated configuration must fail validation");
        assert_eq!(error.architecture, "qwen3_5");
        assert_eq!((error.field, error.reason), (field, reason.to_owned()));
    }
}

/// Three multimodal sections whose doubled sum is the rotary width validate as text RoPE.
#[test]
fn three_rotary_sections_and_in_vocabulary_tokens_validate() {
    let config = Qwen3_5TextConfig {
        rope_parameters: Some(RopeParameters {
            mrope_section: Some(vec![11, 11, 10]),
            ..RopeParameters::default()
        }),
        num_hidden_layers: 4,
        mtp_num_hidden_layers: 4,
        pad_token_id: Some(TokenIds::One(248_319)),
        eos_token_id: Some(TokenIds::Many(vec![0, 248_319])),
        ..Qwen3_5TextConfig::default()
    };
    config.validate().expect("boundary values validate");
    assert_eq!(config.rope_config().unwrap().rotary_dimension(), 64);
}
