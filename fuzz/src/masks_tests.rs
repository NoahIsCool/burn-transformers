//! Deterministic regression seeds for the masks fuzz harness.

use super::*;

/// Zero-length fuzz input still tests fully masked rows; all-one bytes cover valid tokens.
#[test]
fn covers_padding_and_position_overflow() {
    exercise(&[]);
    exercise(&[1; 96]);
    exercise(&[255; 96]);
    exercise(&[254; 96]);
}
