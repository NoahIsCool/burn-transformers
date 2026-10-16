//! Configuration checks for sparse-layer cadence, dense overrides, and router bounds.

use super::*;

/// Sparse cadence selects every second block, while an explicit dense override takes precedence.
#[test]
fn sparse_step_and_dense_overrides_define_layer_kind() {
    let config = Qwen3MoeConfig {
        num_hidden_layers: 4,
        decoder_sparse_step: 2,
        mlp_only_layers: vec![3],
        ..Qwen3MoeConfig::default()
    };
    assert!(!config.is_moe_layer(0));
    assert!(config.is_moe_layer(1));
    assert!(!config.is_moe_layer(3));
}

/// Selecting five experts from a four-expert bank must fail configuration validation.
#[test]
fn rejects_more_selected_experts_than_exist() {
    let config = Qwen3MoeConfig {
        num_experts: 4,
        num_experts_per_tok: 5,
        ..Qwen3MoeConfig::default()
    };
    assert!(config.validate().is_err());
}

/// A small valid topology: four routed blocks of eight experts with grouped-query attention.
fn valid() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        max_position_embeddings: 64,
        max_window_layers: 4,
        moe_intermediate_size: 6,
        num_experts: 8,
        num_experts_per_tok: 2,
        ..Qwen3MoeConfig::default()
    }
}

/// Returns the field and reason of a configuration that must fail validation.
fn rejection(config: &Qwen3MoeConfig) -> (&'static str, String) {
    let error = config
        .validate()
        .expect_err("the mutated configuration must fail validation");
    assert_eq!(error.architecture, "qwen3_moe");
    (error.field, error.reason)
}

/// Every independent topology rule reports its own serialized field and reason.
///
/// Each case perturbs one field of a valid configuration, so the error proves
/// that rule (not an earlier one) rejected it; the base case validates.
#[test]
fn each_topology_rule_names_its_field_and_reason() {
    valid().validate().expect("the base topology validates");
    let cases: Vec<(Qwen3MoeConfig, &str, &str)> = vec![
        (
            Qwen3MoeConfig {
                model_type: "qwen3".into(),
                ..valid()
            },
            "model_type",
            "expected one of [\"qwen3_moe\"], found \"qwen3\"",
        ),
        (
            Qwen3MoeConfig {
                num_experts: 0,
                ..valid()
            },
            "num_experts",
            "must be nonzero",
        ),
        (
            Qwen3MoeConfig {
                num_attention_heads: 3,
                ..valid()
            },
            "num_attention_heads",
            "must be divisible by num_key_value_heads",
        ),
        (
            Qwen3MoeConfig {
                num_experts_per_tok: 9,
                ..valid()
            },
            "num_experts_per_tok",
            "cannot exceed num_experts",
        ),
        (
            Qwen3MoeConfig {
                hidden_act: "gelu".into(),
                ..valid()
            },
            "hidden_act",
            "released Qwen3 expert and dense blocks require SiLU",
        ),
        (
            Qwen3MoeConfig {
                rms_norm_eps: 0.0,
                ..valid()
            },
            "rms_norm_eps",
            "must be finite and greater than zero",
        ),
        (
            Qwen3MoeConfig {
                attention_dropout: 1.5,
                ..valid()
            },
            "attention_dropout",
            "must be finite and in [0, 1]",
        ),
        (
            Qwen3MoeConfig {
                router_aux_loss_coef: -0.001,
                ..valid()
            },
            "router_aux_loss_coef",
            "must be finite and non-negative",
        ),
        (
            Qwen3MoeConfig {
                router_aux_loss_coef: f64::NAN,
                ..valid()
            },
            "router_aux_loss_coef",
            "must be finite and non-negative",
        ),
        (
            Qwen3MoeConfig {
                use_sliding_window: true,
                sliding_window: None,
                ..valid()
            },
            "sliding_window",
            "is required when use_sliding_window is true",
        ),
        (
            Qwen3MoeConfig {
                use_sliding_window: true,
                sliding_window: Some(0),
                ..valid()
            },
            "sliding_window",
            "must be nonzero",
        ),
        (
            Qwen3MoeConfig {
                use_sliding_window: true,
                sliding_window: Some(65),
                ..valid()
            },
            "sliding_window",
            "cannot exceed max_position_embeddings",
        ),
        (
            Qwen3MoeConfig {
                mlp_only_layers: vec![4],
                ..valid()
            },
            "mlp_only_layers",
            "indices must be unique and less than num_hidden_layers",
        ),
        (
            Qwen3MoeConfig {
                mlp_only_layers: vec![1, 1],
                ..valid()
            },
            "mlp_only_layers",
            "indices must be unique and less than num_hidden_layers",
        ),
        (
            Qwen3MoeConfig {
                mlp_only_layers: vec![0, 1, 2, 3],
                ..valid()
            },
            "decoder_sparse_step",
            "schedule does not contain an MoE layer",
        ),
        (
            Qwen3MoeConfig {
                decoder_sparse_step: 5,
                ..valid()
            },
            "decoder_sparse_step",
            "schedule does not contain an MoE layer",
        ),
        (
            Qwen3MoeConfig {
                rope_theta: Some(0.5),
                ..valid()
            },
            "rope_theta",
            "must be finite and at least one",
        ),
        (
            Qwen3MoeConfig {
                pad_token_id: Some(TokenIds::One(32)),
                ..valid()
            },
            "pad_token_id",
            "contains an ID outside the vocabulary",
        ),
        (
            Qwen3MoeConfig {
                bos_token_id: Some(TokenIds::Many(Vec::new())),
                ..valid()
            },
            "bos_token_id",
            "must contain at least one token ID",
        ),
        (
            Qwen3MoeConfig {
                eos_token_id: Some(TokenIds::Many(vec![1, 32])),
                ..valid()
            },
            "eos_token_id",
            "contains an ID outside the vocabulary",
        ),
    ];
    for (config, field, reason) in cases {
        assert_eq!(rejection(&config), (field, reason.to_owned()));
    }
}

/// An enabled window at exactly the position limit, and in-vocabulary token IDs, validate.
#[test]
fn boundary_window_and_token_ids_validate() {
    let config = Qwen3MoeConfig {
        use_sliding_window: true,
        sliding_window: Some(64),
        pad_token_id: Some(TokenIds::One(31)),
        bos_token_id: Some(TokenIds::Many(vec![0, 31])),
        eos_token_id: Some(TokenIds::One(0)),
        mlp_only_layers: vec![0, 1, 2],
        ..valid()
    };
    config.validate().expect("boundary values validate");
    assert!(!config.is_moe_layer(2));
    assert!(config.is_moe_layer(3));
}

/// The RoPE translator reports a non-unit partial rotary factor at its RoPE field.
#[test]
fn rope_config_rejects_partial_rotation() {
    let config = Qwen3MoeConfig {
        rope_parameters: Some(RopeParameters {
            partial_rotary_factor: Some(0.5),
            ..RopeParameters::default()
        }),
        ..valid()
    };
    let error = config
        .rope_config()
        .expect_err("Qwen3-MoE rotates whole heads");
    assert!(
        error.to_string().contains("qwen3_moe"),
        "unexpected diagnostic: {error}"
    );
    assert_eq!(valid().rope_config().unwrap().head_dimension(), 4);
}
