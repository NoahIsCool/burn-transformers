//! Qwen3-MoE schedule tests against the reference `(i + 1) mod step` formulation.

use proptest::prelude::*;

use super::{any_sparse_layer, listed, sparse_layer};

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

/// Membership and schedule existence on hand-picked schedules.
#[test]
fn sparse_schedule_existence_honors_dense_overrides() {
    assert!(listed(&[3, 1], 1));
    assert!(!listed(&[3, 1], 2));
    assert!(!listed(&[], 0));
    // Step one makes every block sparse unless overridden.
    assert!(any_sparse_layer(2, 1, 4, &[0]));
    assert!(!any_sparse_layer(2, 1, 4, &[0, 1]));
    // Step two makes only odd blocks sparse.
    assert!(!any_sparse_layer(1, 2, 4, &[]));
    assert!(any_sparse_layer(2, 2, 4, &[]));
    assert!(!any_sparse_layer(4, 2, 4, &[1, 3]));
    assert!(!any_sparse_layer(4, 0, 4, &[]));
    assert!(!any_sparse_layer(4, 1, 0, &[]));
    assert!(!any_sparse_layer(0, 1, 4, &[]));
}

proptest! {
    /// Schedule existence equals the iterator definition over every block.
    #[test]
    fn sparse_schedule_existence_matches_the_iterator_definition(
        layers in 0usize..12,
        step in 0usize..5,
        experts in 0usize..3,
        dense in proptest::collection::vec(0usize..12, 0..6),
    ) {
        let reference = (0..layers)
            .any(|index| sparse_layer(index, step, experts, dense.contains(&index)));
        prop_assert_eq!(any_sparse_layer(layers, step, experts, &dense), reference);
    }

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
