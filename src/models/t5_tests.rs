//! Tests T5 configuration widths, historical activation aliases, and allocation bounds.

use super::*;

/// Valid attention projections may be narrower than the residual stream.
#[test]
fn attention_inner_width_is_independent_of_residual_width() {
    let config = T5Config {
        d_model: 512,
        d_kv: 64,
        num_heads: 6,
        ..T5Config::default()
    };
    config.validate().unwrap();
    assert_eq!(config.attention_inner_dim(), 384);
}

/// UL2's derived activation must agree with its selected feed-forward projection.
#[test]
fn derived_ul2_fields_must_match_projection() {
    let config = T5Config {
        feed_forward_proj: T5FeedForwardProjection::GatedSilu,
        dense_act_fn: Some("gelu".into()),
        is_gated_act: Some(true),
        ..T5Config::default()
    };
    assert!(config.validate().is_err());
}

/// Gated GELU accepts the historical `gelu_new` activation and gating metadata.
#[test]
fn gated_gelu_uses_t5s_historical_gelu_new_activation() {
    assert_eq!(T5FeedForwardProjection::GatedGelu.activation(), "gelu_new");
    let config = T5Config {
        feed_forward_proj: T5FeedForwardProjection::GatedGelu,
        dense_act_fn: Some("gelu_new".into()),
        is_gated_act: Some(true),
        ..T5Config::default()
    };
    config.validate().unwrap();
}

/// Projection and relative-bias element-count overflow produce contextual errors.
#[test]
fn allocation_products_are_checked_before_backend_initialization() {
    let attention_overflow = T5Config {
        d_model: usize::MAX,
        d_kv: 2,
        d_ff: 1,
        vocab_size: 1,
        num_heads: 1,
        ..T5Config::default()
    };
    assert!(
        attention_overflow
            .validate()
            .unwrap_err()
            .to_string()
            .contains("attention_projection_elements")
    );

    let bias_overflow = T5Config {
        d_model: 1,
        d_kv: 1,
        d_ff: 1,
        vocab_size: 1,
        num_heads: usize::MAX,
        ..T5Config::default()
    };
    assert!(
        bias_overflow
            .validate()
            .unwrap_err()
            .to_string()
            .contains("relative_attention_bias_elements")
    );
}
