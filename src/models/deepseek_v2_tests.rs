use super::*;

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

#[test]
fn lite_checkpoint_may_omit_query_lora() {
    let config = DeepseekV2Config {
        q_lora_rank: None,
        ..DeepseekV2Config::default()
    };
    config.validate().unwrap();
}

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
