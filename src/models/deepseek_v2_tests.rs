//! DeepSeek-V2 configuration defaults, sparse routing constraints, and checked dimension boundaries.

use super::*;

/// The default V2-Lite dimensions, expert counts, direct queries, and YaRN policy validate together.
#[test]
fn defaults_match_the_pinned_v2_lite_topology() {
    let config = DeepseekV2Config::default();
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_hidden_layers, 27);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.n_routed_experts, 64);
    assert_eq!(config.num_experts_per_tok, 6);
    assert_eq!(config.n_shared_experts, 2);
    assert_eq!(config.q_lora_rank, None);
    assert_eq!(
        config.rope_scaling.as_ref().map(|rope| rope.rope_type),
        Some(super::super::config::RopeType::Yarn)
    );
    config.validate().unwrap();
}

/// A direct query projection remains valid when the optional query bottleneck is absent.
#[test]
fn lite_checkpoint_may_omit_query_lora() {
    let config = DeepseekV2Config {
        q_lora_rank: None,
        ..DeepseekV2Config::default()
    };
    config.validate().unwrap();
}

/// Selecting three experts from a group containing only two is rejected during validation.
#[test]
fn routing_groups_must_offer_enough_experts() {
    let config = DeepseekV2Config {
        n_routed_experts: 8,
        n_group: 4,
        topk_group: 1,
        num_experts_per_tok: 3,
        ..DeepseekV2Config::default()
    };
    assert!(config.validate().is_err());
}

/// Zero query rank, an invalid dense-prefix schedule, and infinite routing scale are rejected.
#[test]
fn invalid_lora_schedule_and_router_scalars_fail_validation() {
    let config = DeepseekV2Config {
        q_lora_rank: Some(0),
        ..DeepseekV2Config::default()
    };
    assert!(config.validate().is_err());

    let config = DeepseekV2Config {
        first_k_dense_replace: DeepseekV2Config::default().num_hidden_layers,
        ..DeepseekV2Config::default()
    };
    assert!(config.validate().is_err());

    let config = DeepseekV2Config {
        routed_scaling_factor: f64::INFINITY,
        ..DeepseekV2Config::default()
    };
    assert!(config.validate().is_err());
}

/// Combined query/key width saturates for inspection and fails validation when addition would overflow.
#[test]
fn overflowing_head_dimensions_fail_without_panicking() {
    let config = DeepseekV2Config {
        qk_nope_head_dim: usize::MAX,
        qk_rope_head_dim: 2,
        ..DeepseekV2Config::default()
    };
    assert_eq!(config.qk_head_dim(), usize::MAX);
    assert!(config.validate().is_err());
}

/// Returns the field and reason of a configuration that must fail validation.
fn rejection(config: &DeepseekV2Config) -> (&'static str, String) {
    let error = config
        .validate()
        .expect_err("the mutated configuration must fail validation");
    assert_eq!(error.architecture, "deepseek_v2");
    (error.field, error.reason)
}

/// Every independent topology rule reports its own serialized field and reason.
///
/// Each case perturbs the valid V2-Lite defaults in one place, so the error
/// proves that rule (not an earlier one) rejected it.
#[test]
fn each_topology_rule_names_its_field_and_reason() {
    let base = DeepseekV2Config::default;
    let cases: Vec<(DeepseekV2Config, &str, &str)> = vec![
        (
            DeepseekV2Config {
                model_type: "deepseek_v3".into(),
                ..base()
            },
            "model_type",
            "expected one of [\"deepseek_v2\"], found \"deepseek_v3\"",
        ),
        (
            DeepseekV2Config {
                moe_layer_freq: 0,
                ..base()
            },
            "moe_layer_freq",
            "must be nonzero",
        ),
        (
            DeepseekV2Config {
                qk_rope_head_dim: 63,
                ..base()
            },
            "qk_rope_head_dim",
            "RoPE rotates channel pairs, so this width must be even",
        ),
        (
            DeepseekV2Config {
                kv_lora_rank: 0,
                ..base()
            },
            "kv_lora_rank",
            "must be nonzero",
        ),
        (
            DeepseekV2Config {
                use_mla: false,
                num_key_value_heads: 3,
                ..base()
            },
            "num_attention_heads",
            "must be divisible by num_key_value_heads",
        ),
        (
            DeepseekV2Config {
                ep_size: 0,
                ..base()
            },
            "ep_size",
            "must be nonzero",
        ),
        (
            DeepseekV2Config {
                n_group: 5,
                ..base()
            },
            "n_routed_experts",
            "must be divisible by n_group",
        ),
        (
            DeepseekV2Config {
                num_experts_per_tok: 65,
                ..base()
            },
            "num_experts_per_tok",
            "cannot exceed n_routed_experts",
        ),
        (
            DeepseekV2Config {
                topk_group: 2,
                ..base()
            },
            "topk_group",
            "cannot exceed n_group",
        ),
        (
            DeepseekV2Config {
                moe_layer_freq: 64,
                ..base()
            },
            "moe_layer_freq",
            "schedule does not contain an MoE layer",
        ),
        (
            DeepseekV2Config {
                hidden_act: "gelu".into(),
                ..base()
            },
            "hidden_act",
            "released DeepSeek-V2 blocks require SiLU/SwiGLU",
        ),
        (
            DeepseekV2Config {
                attention_dropout: -0.1,
                ..base()
            },
            "attention_dropout",
            "must be finite and in [0, 1]",
        ),
        (
            DeepseekV2Config {
                aux_loss_alpha: -0.001,
                ..base()
            },
            "aux_loss_alpha",
            "must be finite and non-negative",
        ),
        (
            DeepseekV2Config {
                aux_loss_alpha: f64::INFINITY,
                ..base()
            },
            "aux_loss_alpha",
            "must be finite and non-negative",
        ),
        (
            DeepseekV2Config {
                rope_theta: 0.5,
                ..base()
            },
            "rope_theta",
            "must be finite and at least one",
        ),
        (
            DeepseekV2Config {
                pad_token_id: Some(TokenIds::One(102_400)),
                ..base()
            },
            "pad_token_id",
            "contains an ID outside the vocabulary",
        ),
        (
            DeepseekV2Config {
                bos_token_id: Some(TokenIds::Many(Vec::new())),
                ..base()
            },
            "bos_token_id",
            "must contain at least one token ID",
        ),
        (
            DeepseekV2Config {
                eos_token_id: Some(TokenIds::Many(vec![0, 102_400])),
                ..base()
            },
            "eos_token_id",
            "contains an ID outside the vocabulary",
        ),
    ];
    for (config, field, reason) in cases {
        assert_eq!(rejection(&config), (field, reason.to_owned()));
    }
}

/// Grouped-query attention without MLA validates when heads divide evenly.
#[test]
fn non_latent_attention_requires_only_divisible_heads() {
    let config = DeepseekV2Config {
        use_mla: false,
        kv_lora_rank: 0,
        num_key_value_heads: 4,
        ..DeepseekV2Config::default()
    };
    config.validate().expect("16 query heads share 4 KV heads");
}

/// Sparse layers start at the dense prefix and follow the MoE cadence.
#[test]
fn sparse_cadence_starts_after_the_dense_prefix() {
    let config = DeepseekV2Config {
        num_hidden_layers: 6,
        first_k_dense_replace: 1,
        moe_layer_freq: 2,
        ..DeepseekV2Config::default()
    };
    config.validate().unwrap();
    let sparse = (0..6)
        .filter(|&layer| config.is_moe_layer(layer))
        .collect::<Vec<_>>();
    assert_eq!(sparse, [2, 4]);
}
