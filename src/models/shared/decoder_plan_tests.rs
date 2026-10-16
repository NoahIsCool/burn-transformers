//! Shared decoder planning against direct reference formulations.
//!
//! The Verus harnesses prove these postconditions for all inputs; the tests
//! pin concrete values and cross-check each function against a naive
//! implementation of its specification.

use proptest::prelude::*;

use super::*;

/// Layers are numbered densely inside their own group.
#[test]
fn cache_slots_number_each_group_densely() {
    let plan = cache_slots(&[false, true, true, false, true]);
    assert_eq!(plan.slots, [0, 0, 1, 1, 2]);
    assert_eq!((plan.full_layers, plan.ring_layers), (2, 3));
    let empty = cache_slots(&[]);
    assert!(empty.slots.is_empty());
    assert_eq!((empty.full_layers, empty.ring_layers), (0, 0));
}

/// Allocated groups are numbered densely, full storage first.
#[test]
fn group_indices_number_allocated_groups_densely() {
    let both = group_indices(2, 3);
    assert_eq!((both.full, both.ring), (Some(0), Some(1)));
    let ring_only = group_indices(0, 3);
    assert_eq!((ring_only.full, ring_only.ring), (None, Some(0)));
    let full_only = group_indices(2, 0);
    assert_eq!((full_only.full, full_only.ring), (Some(0), None));
    let none = group_indices(0, 0);
    assert_eq!((none.full, none.ring), (None, None));
}

/// Only a truncation to a shorter nonzero length leaves stale bytes.
#[test]
fn truncation_leaves_bytes_only_when_shortening_to_nonzero() {
    assert!(truncation_leaves_bytes(&[5, 3], &[5, 2]));
    assert!(!truncation_leaves_bytes(&[5, 3], &[0, 3]));
    assert!(!truncation_leaves_bytes(&[5, 3], &[5, 4]));
    assert!(!truncation_leaves_bytes(&[5], &[]));
    assert!(truncation_leaves_bytes(&[5, 3], &[4]));
}

/// The widest context is the maximum, or 1 without active rows.
#[test]
fn widest_context_is_the_maximum_or_one() {
    assert_eq!(widest_context(&[]), 1);
    assert_eq!(widest_context(&[3, 9, 4]), 9);
    assert_eq!(widest_context(&[0]), 0);
}

/// Projection widths are exact, or name the overflow; `u64` extents are exact.
#[test]
fn projection_extents_are_exact_or_report_overflow() {
    assert_eq!(projection_extent(8, 128), Ok(1024));
    assert_eq!(projection_extent(0, usize::MAX), Ok(0));
    assert_eq!(projection_extent(usize::MAX, 2), Err(ExtentError::Overflow));
    assert_eq!(
        checkpoint_extent(usize::MAX),
        u64::try_from(usize::MAX).ok()
    );
    assert_eq!(checkpoint_extent(7), Some(7));
}

/// A missing head width is `hidden / heads`, rounded down, or zero without heads.
#[test]
fn derived_head_dim_divides_or_is_zero() {
    assert_eq!(derived_head_dim(4096, 32), 128);
    assert_eq!(derived_head_dim(10, 3), 3);
    assert_eq!(derived_head_dim(64, 0), 0);
}

proptest! {
    /// Cache slots equal per-group prefix counts.
    #[test]
    fn cache_slots_equal_prefix_counts(ring in prop::collection::vec(any::<bool>(), 0..24)) {
        let plan = cache_slots(&ring);
        for (layer, &in_ring) in ring.iter().enumerate() {
            let earlier = ring[..layer].iter().filter(|&&other| other == in_ring).count();
            prop_assert_eq!(plan.slots[layer], earlier);
        }
        prop_assert_eq!(plan.ring_layers, ring.iter().filter(|&&flag| flag).count());
        prop_assert_eq!(plan.full_layers + plan.ring_layers, ring.len());
    }

    /// Each group's slots are an order-preserving bijection onto `0..size`.
    #[test]
    fn cache_slots_biject_each_group_in_order(ring in prop::collection::vec(any::<bool>(), 0..24)) {
        let plan = cache_slots(&ring);
        for flag in [false, true] {
            let slots: Vec<usize> = ring
                .iter()
                .zip(&plan.slots)
                .filter(|&(&in_ring, _)| in_ring == flag)
                .map(|(_, &slot)| slot)
                .collect();
            let size = if flag { plan.ring_layers } else { plan.full_layers };
            prop_assert_eq!(slots, (0..size).collect::<Vec<_>>());
        }
    }

    /// Truncation leaves bytes exactly when some common row shrinks to a nonzero length.
    #[test]
    fn truncation_matches_its_definition(
        from in prop::collection::vec(0usize..6, 0..6),
        to in prop::collection::vec(0usize..6, 0..6),
    ) {
        let expected = from
            .iter()
            .zip(&to)
            .any(|(&before, &after)| after != 0 && after < before);
        prop_assert_eq!(truncation_leaves_bytes(&from, &to), expected);
    }

    /// The widest context is the maximum of a nonempty list.
    #[test]
    fn widest_context_is_the_list_maximum(
        contexts in prop::collection::vec(0usize..1000, 1..12),
    ) {
        prop_assert_eq!(widest_context(&contexts), *contexts.iter().max().unwrap());
    }

    /// Projection extents equal the checked product.
    #[test]
    fn projection_extent_is_the_checked_product(heads in any::<usize>(), head_dim in 0usize..1 << 20) {
        let expected = heads
            .checked_mul(head_dim)
            .map(|width| u64::try_from(width).unwrap())
            .ok_or(ExtentError::Overflow);
        prop_assert_eq!(projection_extent(heads, head_dim), expected);
    }
}
