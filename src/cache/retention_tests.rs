//! Retention predicates and working windows against brute-force ring simulation.

use super::*;

/// Simulates which absolute positions a ring of width `window` still stores.
fn stored_positions(written: usize, window: usize) -> Vec<bool> {
    let mut slots = vec![None; window];
    for position in 0..written {
        slots[position % window] = Some(position);
    }
    (0..written)
        .map(|position| slots[position % window] == Some(position))
        .collect()
}

/// The closed-form rewind predicate equals "every newly visible position survived".
#[test]
fn ring_rewind_predicate_matches_eviction_simulation_exhaustively() {
    for window in 1..=6 {
        for from in 0..=14 {
            let stored = stored_positions(from, window);
            for to in 0..=from {
                let visible = to.saturating_sub(window)..to;
                let expected = visible.clone().all(|position| stored[position]);
                assert_eq!(
                    ring_rewind_is_safe(from, to, window),
                    expected,
                    "window {window}, rewind {from} -> {to}, visible {visible:?}"
                );
            }
        }
    }
}

/// Recurrent state can only stay put or restart from zero.
#[test]
fn recurrent_rewind_allows_only_identity_or_restart() {
    assert!(recurrent_rewind_is_exact(5, 5));
    assert!(recurrent_rewind_is_exact(5, 0));
    assert!(recurrent_rewind_is_exact(0, 0));
    assert!(!recurrent_rewind_is_exact(5, 4));
    assert!(!recurrent_rewind_is_exact(5, 1));
}

/// Row scans report the lowest offending row and accept fully safe batches.
#[test]
fn first_unsafe_rows_are_reported_in_batch_order() {
    assert_eq!(first_unsafe_ring_rewind(&[9, 9, 9], &[9, 0, 9], 4), None);
    assert_eq!(
        first_unsafe_ring_rewind(&[9, 3, 9, 9], &[9, 1, 5, 2], 4),
        Some(2)
    );
    assert_eq!(first_unsafe_ring_rewind(&[], &[], 1), None);
    assert_eq!(first_inexact_recurrent_rewind(&[4, 4, 4], &[4, 0, 4]), None);
    assert_eq!(
        first_inexact_recurrent_rewind(&[4, 4, 4], &[4, 3, 2]),
        Some(1)
    );
}

/// The working window keeps the newest proposals, then the newest committed tokens.
#[test]
fn working_window_partitions_the_visible_suffix_exhaustively() {
    for window in 1..=5 {
        for committed in 0..=12 {
            for staged in 0..=8 {
                let plan = ring_working_window(committed, staged, window);
                let working = committed + staged;
                let visible = working.saturating_sub(window)..working;
                assert_eq!(plan.retained_start, visible.start);
                // Reassemble the visible positions from the two planned parts.
                let committed_part = committed - plan.committed_keep..committed;
                let staged_part = committed + plan.staged_skip..working;
                let assembled: Vec<usize> = committed_part.clone().chain(staged_part).collect();
                assert_eq!(assembled, visible.collect::<Vec<_>>());
                // The committed part must be readable from the ring itself.
                let stored = stored_positions(committed, window);
                assert!(committed_part.clone().all(|position| stored[position]));
            }
        }
    }
}
