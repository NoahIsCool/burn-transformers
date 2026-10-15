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

/// The causal decoder table needs `max_distance > buckets / 2`, a stricter bound than the encoder's.
///
/// Buckets 32 with max distance 12 passes the encoder bound (12 > 8) but not
/// the decoder bound (12 <= 16): full validation and the checkpoint contract
/// must reject it with a field error, while encoder-only validation accepts it.
#[test]
fn decoder_bucket_bound_is_validated_with_a_field_error() {
    let config = T5Config {
        relative_attention_num_buckets: 32,
        relative_attention_max_distance: 12,
        ..T5Config::default()
    };
    let error = config.validate().unwrap_err();
    assert_eq!(error.field, "relative_attention_max_distance");
    assert!(error.reason.contains("causal decoder"), "{error}");
    let error = config.checkpoint_contract().unwrap_err();
    assert_eq!(error.field, "relative_attention_max_distance");
    config
        .validate_stacks(T5Stacks::EncoderOnly)
        .expect("the encoder bound alone holds");
    for max_distance in [17, 128] {
        T5Config {
            relative_attention_max_distance: max_distance,
            ..config.clone()
        }
        .validate()
        .expect("both bucket tables are valid");
    }
    // The boundary itself is rejected: the logarithmic region needs max > exact.
    assert!(
        T5Config {
            relative_attention_max_distance: 16,
            ..config
        }
        .validate()
        .is_err()
    );
}

proptest::proptest! {
    /// Full validation accepts a bucket geometry exactly when both stack tables can be built.
    #[test]
    fn full_validation_agrees_with_both_bucket_constructors(
        num_buckets in 1_usize..80,
        max_distance in 1_usize..80,
    ) {
        use crate::position::T5RelativePositionBuckets;

        let config = T5Config {
            relative_attention_num_buckets: num_buckets,
            relative_attention_max_distance: max_distance,
            ..T5Config::default()
        };
        let encoder = T5RelativePositionBuckets::new(num_buckets, max_distance, true).is_ok();
        let decoder = T5RelativePositionBuckets::new(num_buckets, max_distance, false).is_ok();
        proptest::prop_assert_eq!(config.validate().is_ok(), encoder && decoder);
        proptest::prop_assert_eq!(
            config.validate_stacks(T5Stacks::EncoderOnly).is_ok(),
            encoder
        );
    }
}
