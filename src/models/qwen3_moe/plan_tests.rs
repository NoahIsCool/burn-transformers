//! Qwen3-MoE schedule tests against the reference `(i + 1) mod step` formulation.

use proptest::prelude::*;

use super::{layer_window, sparse_layer};

/// The reference predicate as Transformers writes it, in wide arithmetic.
fn reference_sparse(index: usize, step: usize, experts: usize, dense: bool) -> bool {
    step != 0 && !dense && experts > 0 && (index as u128 + 1).is_multiple_of(step as u128)
}

/// Small schedules match the reference exactly, including step one and zero.
#[test]
fn sparse_schedule_matches_the_reference_exhaustively() {
    for step in 0..7 {
        for index in 0..64 {
            for experts in [0, 4] {
                for dense in [false, true] {
                    assert_eq!(
                        sparse_layer(index, step, experts, dense),
                        reference_sparse(index, step, experts, dense),
                        "index={index} step={step} experts={experts} dense={dense}"
                    );
                }
            }
        }
    }
    // The overflow-free form also covers the largest index.
    assert!(sparse_layer(usize::MAX, 1, 4, false));
}

/// Windows start exactly at `max_window_layers` and need enablement and a width.
#[test]
fn windows_follow_the_boundary_and_both_guards() {
    assert_eq!(layer_window(3, true, 3, Some(8)), Some(8));
    assert_eq!(layer_window(2, true, 3, Some(8)), None);
    assert_eq!(layer_window(5, false, 3, Some(8)), None);
    assert_eq!(layer_window(5, true, 3, None), None);
}

proptest! {
    /// Arbitrary indices and steps agree with the reference predicate.
    #[test]
    fn sparse_schedule_matches_the_reference(
        index in any::<usize>(),
        step in 0usize..1_000,
        experts in 0usize..3,
        dense in any::<bool>(),
    ) {
        prop_assert_eq!(
            sparse_layer(index, step, experts, dense),
            reference_sparse(index, step, experts, dense)
        );
    }
}
