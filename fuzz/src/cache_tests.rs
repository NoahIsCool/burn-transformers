//! Deterministic regression seeds for the cache fuzz harness.

use super::*;

/// Appends, failed mutation under a mark, rejected commit, rollback and stale marks interact.
#[test]
fn transactional_sequence_restores_reference_state() {
    exercise(&[
        0, 7, 0, 0, 3, 0, 6, 0, 0, 0, 0, 0, 2, 0, 7, 0, 3, 0, 8, 0, 0, 0, 9, 0, 0, 0,
    ]);
    exercise(&[255; 128]);
    for start in 0..12 {
        for count in 0..12 {
            check_ring(start, count, 4);
        }
    }
    check_ring(usize::MAX, 1, 4);
}
