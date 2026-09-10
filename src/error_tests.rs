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
