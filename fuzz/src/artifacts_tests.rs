//! Deterministic regression seeds for the artifacts fuzz harness.

use super::*;

/// Seeds exercise valid discovery, traversal, duplicate keys and corrupt envelopes.
#[test]
fn artifact_modes_have_bounded_success_and_failure_paths() {
    assert!(exercise(b"\x01a.safetensors"));
    assert!(!exercise(b"\x01../a.safetensors"));
    assert!(!exercise(b"\x01/etc/passwd"));
    assert!(!exercise(b"\0{\"weight_map\":{},\"weight_map\":{}}"));
    assert!(!exercise(b"\x02\xff\xff\xff\xff\xff\xff\xff\xff"));
    assert!(exercise(
        b"\x03{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}"
    ));
}
