//! Nemotron-H schedule normalization, configuration rejection, and pinned schema tests.

use super::*;

/// Published 52-block Nemotron 3/3.5 topology: 23 Mamba, 23 MoE, 6 GQA.
const OFFICIAL_30B_PATTERN: &str = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";

/// Legacy characters deserialize to ordered block types and reconstruct unchanged.
#[test]
fn legacy_pattern_round_trips_to_normalized_blocks() {
    let json = r#"{
            "model_type":"nemotron_h",
            "hybrid_override_pattern":"M-*E",
            "num_hidden_layers":4
        }"#;
    let config: NemotronHConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.hybrid_override_pattern(), "M-*E");
    config.validate().unwrap();
}

/// A declared depth that disagrees with the configured block schedule is rejected.
#[test]
fn rejects_declared_depth_that_conflicts_with_pattern() {
    let config = NemotronHConfig {
        num_hidden_layers: Some(99),
        ..NemotronHConfig::default()
    };
    assert!(config.validate().is_err());
}

/// Runtime-specific recurrence and routing constraints fail during config validation.
#[test]
fn rejects_unsupported_recurrent_and_shared_expert_edges() {
    let config = NemotronHConfig {
        conv_kernel: 1,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    let config = NemotronHConfig {
        n_shared_experts: 2,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    let config = NemotronHConfig {
        n_routed_experts: 2,
        n_group: 2,
        topk_group: 2,
        num_experts_per_tok: 2,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

/// The pinned Nemotron 3 Nano legacy-pattern config remains schema compatible.
#[test]
fn nemotron_3_nano_official_shape_config_is_compatible() {
    let json = format!(
        r#"{{
                "model_type":"nemotron_h",
                "vocab_size":131072,
                "hidden_size":2688,
                "hybrid_override_pattern":"{OFFICIAL_30B_PATTERN}",
                "num_hidden_layers":52,
                "num_attention_heads":32,
                "num_key_value_heads":2,
                "head_dim":128,
                "max_position_embeddings":262144,
                "intermediate_size":1856,
                "ssm_state_size":128,
                "mamba_num_heads":64,
                "mamba_head_dim":64,
                "n_groups":8,
                "conv_kernel":4,
                "chunk_size":128,
                "n_routed_experts":128,
                "n_shared_experts":1,
                "moe_intermediate_size":1856,
                "moe_shared_expert_intermediate_size":3712,
                "num_experts_per_tok":6,
                "routed_scaling_factor":2.5,
                "n_group":1,
                "topk_group":1,
                "norm_eps":0.00001
            }}"#
    );
    let config: NemotronHConfig = serde_json::from_str(&json).unwrap();
    config.validate().unwrap();
    assert_eq!(config.block_count(), 52);
    assert_eq!(
        config
            .layers_block_type
            .iter()
            .filter(|&&kind| kind == NemotronHBlockType::LinearAttention)
            .count(),
        23
    );
    assert_eq!(
        config
            .layers_block_type
            .iter()
            .filter(|&&kind| kind == NemotronHBlockType::Moe)
            .count(),
        23
    );
}

/// The 3.5 Lightning list-form schema and checkpoint-declared MTP schedule parse.
#[test]
fn nemotron_3_5_lightning_list_config_is_compatible() {
    let layers = OFFICIAL_30B_PATTERN
        .chars()
        .map(|character| match character {
            'M' => "mamba",
            'E' => "moe",
            '*' => "attention",
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    let config: NemotronHConfig = serde_json::from_value(serde_json::json!({
        "model_type": "nemotron_h",
        "vocab_size": 131072,
        "hidden_size": 2688,
        "layers_block_type": layers,
        "num_hidden_layers": 52,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "max_position_embeddings": 262144,
        "intermediate_size": 1856,
        "ssm_state_size": 128,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "n_groups": 8,
        "conv_kernel": 4,
        "chunk_size": 128,
        "n_routed_experts": 128,
        "n_shared_experts": 1,
        "moe_intermediate_size": 1856,
        "moe_shared_expert_intermediate_size": 3712,
        "num_experts_per_tok": 6,
        "routed_scaling_factor": 2.5,
        "n_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
        "norm_eps": 0.00001
    }))
    .unwrap();
    config.validate().unwrap();
    assert_eq!(config.block_count(), 52);
    assert_eq!(config.num_nextn_predict_layers, 1);
    assert_eq!(
        config.mtp_layers_block_type,
        [NemotronHBlockType::FullAttention, NemotronHBlockType::Moe]
    );
}

/// Asserts that `config` fails validation at exactly `field` with exactly `reason`.
fn assert_rejected(config: &NemotronHConfig, field: &str, reason: &str) {
    let error = config
        .validate()
        .expect_err("the configuration must be rejected");
    assert_eq!(
        (error.architecture, error.field, error.reason.as_str()),
        ("nemotron_h", field, reason),
        "unexpected diagnostic for {config:?}"
    );
}

/// Returns the upstream defaults with one edit applied.
fn edited(edit: impl FnOnce(&mut NemotronHConfig)) -> NemotronHConfig {
    let mut config = NemotronHConfig::default();
    edit(&mut config);
    config
}

/// The default heterogeneous schedule is valid and every block kind round-trips.
#[test]
fn default_schedule_validates_and_every_pattern_character_round_trips() {
    let config = NemotronHConfig::default();
    config.validate().unwrap();
    assert_eq!(config.hybrid_override_pattern(), "ME*-");
    for character in ['M', '*', '-', 'E'] {
        let kind = NemotronHBlockType::from_pattern(character).unwrap();
        assert_eq!(kind.pattern_character(), character);
    }
    assert_eq!(NemotronHBlockType::from_pattern('x'), None);
    assert_eq!(
        <NemotronHConfig as ModelConfig>::MODEL_TYPES,
        &["nemotron_h"]
    );
    assert_eq!(<NemotronHConfig as ModelConfig>::FAMILY, "nemotron_h");
}

/// Unknown pattern characters name the character and its index; lists accept aliases.
#[test]
fn schedule_wire_forms_reject_unknown_characters_and_accept_aliases() {
    let error = serde_json::from_str::<NemotronHConfig>(r#"{"hybrid_override_pattern":"M*x"}"#)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("invalid Nemotron-H pattern character 'x' at index 2"),
        "{error}"
    );
    let config: NemotronHConfig = serde_json::from_str(
        r#"{"layer_types":["mamba","attention","mlp","moe","linear_attention","full_attention"]}"#,
    )
    .unwrap();
    assert_eq!(config.hybrid_override_pattern(), "M*-EM*");
    assert!(serde_json::from_str::<NemotronHConfig>(r#"{"layer_types":["conv"]}"#).is_err());
    assert!(serde_json::from_str::<NemotronHConfig>(r#"{"layer_types":7}"#).is_err());
}

/// Global extents, declared depth, regularization, epsilons, and token IDs fail by field.
#[test]
fn global_fields_are_rejected_with_their_exact_diagnostics() {
    let cases: Vec<(NemotronHConfig, &str, &str)> = vec![
        (
            edited(|c| c.model_type = "llama".into()),
            "model_type",
            r#"expected one of ["nemotron_h"], found "llama""#,
        ),
        (
            edited(|c| c.vocab_size = 0),
            "vocab_size",
            "must be nonzero",
        ),
        (
            edited(|c| c.hidden_size = 0),
            "hidden_size",
            "must be nonzero",
        ),
        (
            edited(|c| c.layers_block_type.clear()),
            "layers_block_type",
            "must be nonzero",
        ),
        (
            edited(|c| c.max_position_embeddings = 0),
            "max_position_embeddings",
            "must be nonzero",
        ),
        (
            edited(|c| c.num_logits_to_keep = 0),
            "num_logits_to_keep",
            "must be nonzero",
        ),
        (
            edited(|c| c.num_hidden_layers = Some(3)),
            "num_hidden_layers",
            "must match the authoritative block schedule length",
        ),
        (
            edited(|c| c.initializer_range = 0.0),
            "initializer_range",
            "must be finite and greater than zero",
        ),
        (
            edited(|c| c.layer_norm_epsilon = f64::NAN),
            "layer_norm_epsilon",
            "must be finite and greater than zero",
        ),
        (
            edited(|c| c.attention_dropout = 1.5),
            "attention_dropout",
            "must be finite and in [0, 1]",
        ),
        (
            edited(|c| c.hidden_dropout = -0.1),
            "hidden_dropout",
            "must be finite and in [0, 1]",
        ),
        (
            edited(|c| c.rms_norm_eps = Some(-1.0)),
            "rms_norm_eps",
            "must be finite and greater than zero",
        ),
        (
            edited(|c| c.rms_norm_eps = Some(1e-6)),
            "rms_norm_eps",
            "must agree with layer_norm_epsilon",
        ),
        (
            edited(|c| c.norm_eps = Some(2e-5)),
            "norm_eps",
            "must agree with layer_norm_epsilon",
        ),
        (
            edited(|c| c.pad_token_id = Some(TokenIds::One(131_072))),
            "pad_token_id",
            "contains an ID outside the vocabulary",
        ),
        (
            edited(|c| c.bos_token_id = Some(TokenIds::Many(Vec::new()))),
            "bos_token_id",
            "must contain at least one token ID",
        ),
        (
            edited(|c| c.eos_token_id = Some(TokenIds::Many(vec![2, 131_072]))),
            "eos_token_id",
            "contains an ID outside the vocabulary",
        ),
    ];
    for (config, field, reason) in &cases {
        assert_rejected(config, field, reason);
    }
    // Redundant epsilons that agree with `layer_norm_epsilon` are accepted.
    edited(|c| {
        c.rms_norm_eps = Some(1e-5);
        c.norm_eps = Some(1e-5);
    })
    .validate()
    .unwrap();
}

/// Attention geometry, rotary metadata, and the window are checked only with attention.
#[test]
fn attention_fields_are_rejected_only_when_attention_is_scheduled() {
    let cases: Vec<(NemotronHConfig, &str, &str)> = vec![
        (edited(|c| c.head_dim = 0), "head_dim", "must be nonzero"),
        (
            edited(|c| c.num_attention_heads = 12),
            "num_attention_heads",
            "must be divisible by num_key_value_heads",
        ),
        (
            edited(|c| c.num_key_value_heads = 0),
            "num_key_value_heads",
            "must be nonzero",
        ),
        (
            edited(|c| c.partial_rotary_factor = Some(0.0)),
            "partial_rotary_factor",
            "must be finite and in (0, 1]",
        ),
        (
            edited(|c| c.partial_rotary_factor = Some(1.5)),
            "partial_rotary_factor",
            "must be finite and in (0, 1]",
        ),
        (
            edited(|c| c.partial_rotary_factor = Some(f64::NAN)),
            "partial_rotary_factor",
            "must be finite and in (0, 1]",
        ),
        (
            // 128 * 0.3 = 38.4 is not an integral width.
            edited(|c| c.partial_rotary_factor = Some(0.3)),
            "partial_rotary_factor",
            "must produce an even integral rotary width",
        ),
        (
            // 128 * 3/128 = 3 is integral but odd.
            edited(|c| c.partial_rotary_factor = Some(3.0 / 128.0)),
            "partial_rotary_factor",
            "must produce an even integral rotary width",
        ),
        (
            edited(|c| c.rope_theta = Some(0.0)),
            "rope_theta",
            "must be finite and greater than zero",
        ),
        (
            edited(|c| c.sliding_window = Some(0)),
            "sliding_window",
            "must be nonzero",
        ),
        (
            edited(|c| c.sliding_window = Some(4097)),
            "sliding_window",
            "cannot exceed max_position_embeddings",
        ),
    ];
    for (config, field, reason) in &cases {
        assert_rejected(config, field, reason);
    }
    // A partial factor yielding an even integral width and a full window are valid.
    edited(|c| {
        c.partial_rotary_factor = Some(0.5);
        c.sliding_window = Some(4096);
    })
    .validate()
    .unwrap();
    // Without any attention block (and no MTP stage) the same fields are unused.
    for (config, _, _) in cases {
        let mut config = config;
        config.layers_block_type = vec![NemotronHBlockType::LinearAttention];
        config.validate().unwrap();
    }
    // An enabled MTP stage's attention block re-enables the attention checks.
    let mtp_attention = edited(|c| {
        c.layers_block_type = vec![NemotronHBlockType::LinearAttention];
        c.num_nextn_predict_layers = 1;
        c.head_dim = 0;
    });
    assert_rejected(&mtp_attention, "head_dim", "must be nonzero");
}

/// Dense MLP widths and the shared MLP/MoE activation are checked where they run.
#[test]
fn mlp_and_activation_fields_are_rejected_only_when_scheduled() {
    assert_rejected(
        &edited(|c| c.intermediate_size = 0),
        "intermediate_size",
        "must be nonzero",
    );
    assert_rejected(
        &edited(|c| c.mlp_hidden_act = "gelu".into()),
        "mlp_hidden_act",
        "must be `relu2` or `silu`",
    );
    edited(|c| c.mlp_hidden_act = "silu".into())
        .validate()
        .unwrap();
    // A MoE-only schedule still checks the activation but not the dense width.
    let moe_only = edited(|c| {
        c.layers_block_type = vec![NemotronHBlockType::Moe];
        c.intermediate_size = 0;
    });
    moe_only.validate().unwrap();
    let mut gelu = moe_only.clone();
    gelu.mlp_hidden_act = "gelu".into();
    assert_rejected(&gelu, "mlp_hidden_act", "must be `relu2` or `silu`");
    // A mixer-only schedule ignores both.
    let mut mamba_only = gelu;
    mamba_only.layers_block_type = vec![NemotronHBlockType::LinearAttention];
    mamba_only.validate().unwrap();
}

/// Every Mamba-2 extent, discretization bound, clamp, and cache dtype fails by field.
#[test]
fn mamba_fields_are_rejected_with_their_exact_diagnostics() {
    let mut cases: Vec<(NemotronHConfig, &str, &str)> = [
        "ssm_state_size",
        "mamba_num_heads",
        "mamba_head_dim",
        "n_groups",
        "conv_kernel",
        "expand",
        "chunk_size",
    ]
    .into_iter()
    .map(|field| {
        let config = edited(|c| match field {
            "ssm_state_size" => c.ssm_state_size = 0,
            "mamba_num_heads" => c.mamba_num_heads = 0,
            "mamba_head_dim" => c.mamba_head_dim = 0,
            "n_groups" => c.n_groups = 0,
            "conv_kernel" => c.conv_kernel = 0,
            "expand" => c.expand = 0,
            _ => c.chunk_size = 0,
        });
        (config, field, "must be nonzero")
    })
    .collect();
    cases.extend([
        (
            edited(|c| c.conv_kernel = 1),
            "conv_kernel",
            "must be at least two so the recurrent convolution has history",
        ),
        (
            edited(|c| c.mamba_num_heads = 12),
            "mamba_num_heads",
            "must be divisible by n_groups",
        ),
        (
            edited(|c| c.mamba_hidden_act = "relu".into()),
            "mamba_hidden_act",
            "released Nemotron-H Mamba blocks require SiLU",
        ),
        (
            edited(|c| c.time_step_rank = Some(0)),
            "time_step_rank",
            "must be nonzero",
        ),
        (
            edited(|c| c.time_step_min = 0.0),
            "time_step_min",
            "must be finite and greater than zero",
        ),
        (
            edited(|c| c.time_step_max = f64::INFINITY),
            "time_step_max",
            "must be finite and greater than zero",
        ),
        (
            edited(|c| c.time_step_floor = -1.0),
            "time_step_floor",
            "must be finite and greater than zero",
        ),
        (
            edited(|c| c.time_step_min = 0.5),
            "time_step_min",
            "requires time_step_floor <= time_step_min <= time_step_max",
        ),
        (
            edited(|c| c.time_step_floor = 0.01),
            "time_step_min",
            "requires time_step_floor <= time_step_min <= time_step_max",
        ),
        (
            edited(|c| c.time_step_limit = vec![Some(0.0)]),
            "time_step_limit",
            "must contain exactly [lower, upper]",
        ),
        (
            edited(|c| c.time_step_limit = vec![None, None]),
            "time_step_limit",
            "lower bound must be finite",
        ),
        (
            edited(|c| c.time_step_limit = vec![Some(-0.5), None]),
            "time_step_limit",
            "lower bound must be finite and non-negative",
        ),
        (
            edited(|c| c.time_step_limit = vec![Some(f64::INFINITY), None]),
            "time_step_limit",
            "lower bound must be finite and non-negative",
        ),
        (
            edited(|c| c.time_step_limit = vec![Some(0.5), Some(0.25)]),
            "time_step_limit",
            "finite upper bound must be at least the lower bound",
        ),
        (
            edited(|c| c.time_step_limit = vec![Some(0.0), Some(f64::INFINITY)]),
            "time_step_limit",
            "finite upper bound must be at least the lower bound",
        ),
        (
            edited(|c| c.mamba_ssm_cache_dtype = "int8".into()),
            "mamba_ssm_cache_dtype",
            "must be float32, float16, or bfloat16",
        ),
    ]);
    for (config, field, reason) in &cases {
        assert_rejected(config, field, reason);
    }
    for dtype in ["float16", "bfloat16"] {
        edited(|c| {
            c.mamba_ssm_cache_dtype = dtype.into();
            c.time_step_rank = Some(4);
            c.time_step_limit = vec![Some(0.0), Some(0.5)];
        })
        .validate()
        .unwrap();
    }
    // Recurrent metadata is unused by a schedule without Mamba-2 blocks.
    for (config, _, _) in cases {
        let mut config = config;
        config.layers_block_type = vec![NemotronHBlockType::FullAttention];
        config.validate().unwrap();
    }
}

/// Expert counts, groups, the router feasibility check, and scaling fail by field.
#[test]
fn moe_fields_are_rejected_with_their_exact_diagnostics() {
    let mut cases: Vec<(NemotronHConfig, &str, &str)> = [
        "n_routed_experts",
        "n_shared_experts",
        "moe_intermediate_size",
        "moe_shared_expert_intermediate_size",
        "num_experts_per_tok",
        "n_group",
        "topk_group",
    ]
    .into_iter()
    .map(|field| {
        let config = edited(|c| match field {
            "n_routed_experts" => c.n_routed_experts = 0,
            "n_shared_experts" => c.n_shared_experts = 0,
            "moe_intermediate_size" => c.moe_intermediate_size = 0,
            "moe_shared_expert_intermediate_size" => c.moe_shared_expert_intermediate_size = 0,
            "num_experts_per_tok" => c.num_experts_per_tok = 0,
            "n_group" => c.n_group = 0,
            _ => c.topk_group = 0,
        });
        (config, field, "must be nonzero")
    })
    .collect();
    cases.extend([
        (
            edited(|c| c.n_shared_experts = 2),
            "n_shared_experts",
            "v1 supports the released single shared-expert topology",
        ),
        (
            edited(|c| c.moe_latent_size = Some(0)),
            "moe_latent_size",
            "must be nonzero",
        ),
        (
            edited(|c| c.n_group = 3),
            "n_routed_experts",
            "must be divisible by n_group",
        ),
        (
            // Eight experts in eight groups leave one expert per group.
            edited(|c| c.n_group = 8),
            "n_group",
            "each router group must contain at least two experts for top-2 group scoring",
        ),
        (
            // (usize::MAX - 1) experts in one group, two groups kept.
            edited(|c| {
                c.n_routed_experts = usize::MAX - 1;
                c.topk_group = 2;
            }),
            "topk_group",
            "eligible expert count overflows usize",
        ),
        (
            // Two groups of four keep one group: four eligible experts < five.
            edited(|c| {
                c.n_group = 2;
                c.num_experts_per_tok = 5;
            }),
            "num_experts_per_tok",
            "routing groups do not contain enough eligible experts",
        ),
        (
            edited(|c| c.routed_scaling_factor = f64::NAN),
            "routed_scaling_factor",
            "must be finite and greater than zero",
        ),
    ]);
    for (config, field, reason) in &cases {
        assert_rejected(config, field, reason);
    }
    edited(|c| c.moe_latent_size = Some(1024))
        .validate()
        .unwrap();
    // Expert metadata is unused without a MoE block or an MTP stage.
    for (config, _, _) in &cases {
        let mut config = config.clone();
        config.layers_block_type = vec![NemotronHBlockType::Mlp];
        config.validate().unwrap();
    }
    // An enabled MTP stage's MoE block re-enables the expert checks.
    for (config, field, reason) in cases {
        let mut config = config;
        config.layers_block_type = vec![NemotronHBlockType::Mlp];
        config.num_nextn_predict_layers = 1;
        assert_rejected(&config, field, reason);
    }
}

/// Only the released single `[attention, moe]` MTP stage of a representable width is accepted.
#[test]
fn mtp_stage_shape_is_restricted_to_the_released_topology() {
    let unsupported = "v1 supports one checkpoint-declared [attention, moe] MTP stage";
    assert_rejected(
        &edited(|c| c.num_nextn_predict_layers = 2),
        "mtp_layers_block_type",
        unsupported,
    );
    assert_rejected(
        &edited(|c| {
            c.num_nextn_predict_layers = 1;
            c.mtp_layers_block_type =
                vec![NemotronHBlockType::Moe, NemotronHBlockType::FullAttention];
        }),
        "mtp_layers_block_type",
        unsupported,
    );
    assert_rejected(
        &edited(|c| {
            c.num_nextn_predict_layers = 1;
            c.hidden_size = usize::MAX;
        }),
        "hidden_size",
        "MTP concatenation width overflows usize",
    );
    // A disabled stage ignores its declared schedule entirely.
    edited(|c| c.mtp_layers_block_type = vec![NemotronHBlockType::Mlp])
        .validate()
        .unwrap();
    edited(|c| c.num_nextn_predict_layers = 1)
        .validate()
        .unwrap();
}
