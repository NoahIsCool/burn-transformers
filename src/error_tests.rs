//! Tests for contextual configuration and tensor-shape diagnostics.

use super::{Error, Result};
use crate::{
    cache::{CacheConfig, CacheConfigError},
    position::{RopeConfig, RopeConfigError, RopeScaling},
};

/// Verifies configuration errors retain both machine-localizing and human context.
#[test]
fn configuration_error_is_contextual() {
    let error = Error::configuration("attention.head_dim", "must be non-zero");
    assert_eq!(
        error.to_string(),
        "invalid configuration at attention.head_dim: must be non-zero"
    );
}

/// Verifies shape errors format symbolic dimensions without tensor allocation.
#[test]
fn shape_error_is_contextual() {
    let error = Error::shape("query", "[B, S, H]", "[2, 8]");
    assert_eq!(
        error.to_string(),
        "shape mismatch for query: expected [B, S, H], found [2, 8]"
    );
}

/// Intervention errors name the point, include the cause's text, and keep the cause matchable.
#[test]
fn intervention_error_keeps_point_reason_and_matchable_cause() {
    let plain = Error::intervention("decoder.layers.0.mlp.output", "selection out of range");
    assert_eq!(
        plain.to_string(),
        "intervention at decoder.layers.0.mlp.output failed: selection out of range"
    );
    let caused = Error::intervention_caused_by(
        "encoder.layers.4.mlp.output",
        "edit callback returned an error",
        Error::shape("adapter", "[1, 2]", "[1, 3]"),
    );
    assert_eq!(
        caused.to_string(),
        "intervention at encoder.layers.4.mlp.output failed: edit callback returned an error: shape mismatch for adapter: expected [1, 2], found [1, 3]"
    );
    match &caused {
        Error::Intervention {
            point,
            cause: Some(cause),
            ..
        } => {
            assert_eq!(point, "encoder.layers.4.mlp.output");
            assert!(matches!(cause.as_ref(), Error::Shape { tensor, .. } if tensor == "adapter"));
        }
        other => panic!("unexpected error {other:?}"),
    }
    // The cause is printed once: it is not also reported as an error source.
    assert!(std::error::Error::source(&caused).is_none());
}

/// Cache dimensions compose with `?` and keep their typed error.
#[test]
fn cache_configuration_errors_convert_with_question_mark() {
    /// Builds a windowed cache inside a function returning the crate result.
    fn windowed(sequences: usize, capacity: usize, window: usize) -> Result<CacheConfig> {
        Ok(CacheConfig::new(sequences, capacity)?.with_sliding_window(window)?)
    }
    assert_eq!(
        windowed(2, 8, 4).unwrap().sliding_window().map(usize::from),
        Some(4)
    );
    for (sequences, capacity, window, expected) in [
        (0, 8, 4, CacheConfigError::ZeroSequences),
        (2, 0, 4, CacheConfigError::ZeroCapacity),
        (2, 8, 0, CacheConfigError::ZeroSlidingWindow),
        (
            2,
            8,
            9,
            CacheConfigError::SlidingWindowExceedsCapacity {
                window: 9,
                capacity: 8,
            },
        ),
    ] {
        let error = windowed(sequences, capacity, window).unwrap_err();
        assert!(matches!(error, Error::CacheConfig(found) if found == expected));
        assert_eq!(
            error.to_string(),
            format!("cache configuration error: {expected}")
        );
    }
}

/// Rotary configuration composes with `?` and keeps its typed error.
#[test]
fn rope_configuration_errors_convert_with_question_mark() {
    /// Builds an unscaled rotary configuration inside a function returning the crate result.
    fn rope(head_dim: usize, rotary_dim: usize) -> Result<RopeConfig> {
        Ok(RopeConfig::new(
            head_dim,
            rotary_dim,
            10_000.0,
            RopeScaling::None,
        )?)
    }
    assert!(rope(4, 4).is_ok());
    let expected = RopeConfig::new(0, 4, 10_000.0, RopeScaling::None).unwrap_err();
    let error = rope(0, 4).unwrap_err();
    assert!(matches!(error, Error::Rope(found) if found == expected));
    assert_eq!(
        error.to_string(),
        format!("rope configuration error: {expected}")
    );
    assert!(matches!(
        Error::from(RopeConfigError::ZeroHeadDimension),
        Error::Rope(RopeConfigError::ZeroHeadDimension)
    ));
}
