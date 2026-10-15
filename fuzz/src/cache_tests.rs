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

/// Two-layer, two-head write-ahead steps through every ending (see `fuzz/seeds`).
///
/// Header: rows 3, capacity 12, windows 2 and 5, staging 3, recurrent, two
/// layers, width 2. Each operation is `[operation, count x3, selector]`:
/// write-ahead (7) published (selector % 4 < 2), abandoned (2), or
/// interfered with (3: begin, rollback, or an empty step), mixed with
/// ordinary steps, begin, and commit.
const WRITE_AHEAD_SCRIPT: &[u8] = include_bytes!("../seeds/cache_transactions/tensor_write_ahead");

/// Write-ahead endings match the oracle and stale steps are refused.
#[test]
fn tensor_write_ahead_sequences_match_the_history_oracle() {
    exercise_tensor_state(WRITE_AHEAD_SCRIPT);
    // The same script with one layer and width 1 takes the scalar paths.
    let mut scalar = WRITE_AHEAD_SCRIPT.to_vec();
    scalar[6] = 0;
    scalar[7] = 0;
    exercise_tensor_state(&scalar);
}

/// Steps, transactions, rewinds, and resets over every group kind agree with the oracle.
#[test]
fn tensor_state_sequences_match_the_history_oracle() {
    exercise_tensor_state(&[]);
    exercise_tensor_state(&[255; 384]);
    exercise_tensor_state(&(0..=255).collect::<Vec<_>>());
    // rows 3, capacity 12, windows 2 and 5, staging 3, recurrent, one layer,
    // width 1; then step, begin, step, commit, step, rollback, reorder,
    // truncate, reset.
    let mut script = vec![2, 10, 1, 4, 2, 0, 0, 0];
    for operation in [0, 1, 0, 2, 0, 3, 4, 5, 6, 0, 1, 0, 0, 2] {
        script.extend([operation, 2, 1, 3, operation % 2]);
    }
    exercise_tensor_state(&script);
    exercise_tensor_state(WRITE_AHEAD_SCRIPT);
    for seed in 0..64_u8 {
        let input: Vec<u8> = (0..=255_u8)
            .chain(0..44)
            .map(|index| seed.wrapping_mul(31).wrapping_add(index.wrapping_mul(7)))
            .collect();
        exercise_tensor_state(&input);
    }
    // A linear congruential generator covers many more operation mixes.
    let mut seed = 0x2545_f491_u32;
    for _ in 0..256 {
        let input: Vec<u8> = (0..384)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                seed.to_le_bytes()[3]
            })
            .collect();
        exercise_tensor_state(&input);
    }
}
