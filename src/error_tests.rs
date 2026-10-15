//! Tests for contextual configuration and tensor-shape diagnostics.

use super::Error;

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
