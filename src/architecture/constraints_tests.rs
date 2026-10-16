//! Specification-constraint tests against direct reference definitions.

use super::*;

/// Zero dimensions take precedence over indivisible grouped-query heads.
#[test]
fn attention_heads_classify_in_precedence_order() {
    assert_eq!(attention_heads(0, 0, 0), HeadCheck::ZeroDimension);
    assert_eq!(attention_heads(8, 0, 4), HeadCheck::ZeroDimension);
    assert_eq!(attention_heads(8, 3, 0), HeadCheck::ZeroDimension);
    assert_eq!(attention_heads(8, 3, 4), HeadCheck::IndivisibleGroups);
    assert_eq!(attention_heads(8, 2, 4), HeadCheck::Valid);
    assert_eq!(attention_heads(8, 8, 1), HeadCheck::Valid);
}

/// DeltaNet value heads must form equal groups over nonzero key heads.
#[test]
fn delta_net_heads_require_value_head_multiples() {
    assert_eq!(delta_net_heads(2, 4, 8, 8, 4), HeadCheck::Valid);
    assert_eq!(delta_net_heads(3, 4, 8, 8, 4), HeadCheck::IndivisibleGroups);
    assert_eq!(delta_net_heads(2, 4, 8, 8, 0), HeadCheck::ZeroDimension);
    assert_eq!(delta_net_heads(0, 4, 8, 8, 4), HeadCheck::ZeroDimension);
}

/// Mamba2 group sizes must divide the head count.
#[test]
fn mamba2_group_size_divides_heads() {
    assert!(mamba2_dimensions(8, 64, 16, 4, 2));
    assert!(!mamba2_dimensions(8, 64, 16, 4, 3));
    assert!(!mamba2_dimensions(8, 64, 16, 4, 0));
    assert!(!mamba2_dimensions(0, 64, 16, 4, 1));
}

/// Expert counts need experts, width, and a selection in `1..=experts`.
#[test]
fn expert_counts_bound_the_selection() {
    assert_eq!(expert_counts(0, 1, 8), ExpertCheck::Empty);
    assert_eq!(expert_counts(4, 1, 0), ExpertCheck::Empty);
    assert_eq!(expert_counts(4, 0, 8), ExpertCheck::SelectionOutOfRange);
    assert_eq!(expert_counts(4, 5, 8), ExpertCheck::SelectionOutOfRange);
    assert_eq!(expert_counts(4, 4, 8), ExpertCheck::Valid);
}

/// MLA rejects zeros, then an overflowing query-key width, and saturates the builder width.
#[test]
fn mla_dimensions_reject_zero_then_overflow() {
    assert_eq!(mla_dimensions(2, 4, 2, 4, 8, None), MlaCheck::Valid);
    assert_eq!(
        mla_dimensions(2, 4, 2, 4, 8, Some(0)),
        MlaCheck::ZeroDimension
    );
    assert_eq!(
        mla_dimensions(2, 4, 2, 4, 0, Some(3)),
        MlaCheck::ZeroDimension
    );
    assert_eq!(
        mla_dimensions(2, usize::MAX, 1, 4, 8, None),
        MlaCheck::HeadDimOverflow
    );
    assert_eq!(
        mla_dimensions(2, usize::MAX - 1, 1, 4, 8, None),
        MlaCheck::Valid
    );
    assert_eq!(mla_qk_head_dim(4, 2), 6);
    assert_eq!(mla_qk_head_dim(usize::MAX, 1), usize::MAX);
}

/// Cross-attention projections need nonzero heads and representable sizes.
#[test]
fn cross_projection_sizes_must_fit() {
    assert!(cross_projection_fits(4, 8, 16));
    assert!(cross_projection_fits(4, 8, 0));
    assert!(!cross_projection_fits(0, 8, 16));
    assert!(!cross_projection_fits(4, 0, 16));
    assert!(!cross_projection_fits(usize::MAX, 2, 0));
    assert!(!cross_projection_fits(usize::MAX / 2, 2, 2));
}

/// Stack checks report the first zero dimension in declaration order.
#[test]
fn stack_dimensions_report_the_first_zero() {
    assert_eq!(stack_dimensions(0, 0, 0, 0), StackCheck::ZeroVocabulary);
    assert_eq!(stack_dimensions(1, 0, 0, 0), StackCheck::ZeroHidden);
    assert_eq!(stack_dimensions(1, 1, 0, 0), StackCheck::NoLayers);
    assert_eq!(stack_dimensions(1, 1, 1, 0), StackCheck::ZeroPositions);
    assert_eq!(stack_dimensions(1, 1, 1, 1), StackCheck::Valid);
}

/// Group-preserving reorders reject bad group sizes, then the first escaping row.
#[test]
fn group_reorders_stay_inside_their_groups() {
    assert_eq!(group_reorder(&[0, 1], 0, 2), GroupReorder::InvalidGroupSize);
    assert_eq!(
        group_reorder(&[0, 1, 2], 2, 3),
        GroupReorder::InvalidGroupSize
    );
    assert_eq!(group_reorder(&[1, 0, 3, 3], 2, 4), GroupReorder::Preserved);
    assert_eq!(
        group_reorder(&[1, 2, 3, 3], 2, 4),
        GroupReorder::LeavesGroup { row: 1 }
    );
    assert_eq!(group_reorder(&[], 2, 4), GroupReorder::Preserved);
}

proptest::proptest! {
    /// Head classification matches the direct definition, and valid groups cover every head.
    #[test]
    fn head_checks_match_their_definition(
        heads in 0usize..17,
        groups in 0usize..9,
        width in 0usize..3,
    ) {
        let expected = if heads == 0 || groups == 0 || width == 0 {
            HeadCheck::ZeroDimension
        } else if !heads.is_multiple_of(groups) {
            HeadCheck::IndivisibleGroups
        } else {
            HeadCheck::Valid
        };
        proptest::prop_assert_eq!(attention_heads(heads, groups, width), expected);
        proptest::prop_assert_eq!(delta_net_heads(groups, heads, width, 1, 1), expected);
        if expected == HeadCheck::Valid {
            let size = heads / groups;
            proptest::prop_assert_eq!(size * groups, heads);
            proptest::prop_assert!((0..heads).all(|head| head / size < groups));
            for group in 0..groups {
                let members = (0..heads).filter(|head| head / size == group).count();
                proptest::prop_assert_eq!(members, size);
            }
        }
    }

    /// Overflow-sensitive checks agree with exact 128-bit arithmetic.
    #[test]
    fn overflow_checks_match_wide_arithmetic(
        a in proptest::num::usize::ANY,
        b in proptest::num::usize::ANY,
        c in 0usize..1 << 20,
    ) {
        let (wide_a, wide_b, wide_c) = (a as u128, b as u128, c as u128);
        let max = usize::MAX as u128;
        let inner = wide_a.checked_mul(wide_b);
        let fits = a > 0
            && b > 0
            && inner.is_some_and(|inner| inner <= max && inner.checked_mul(wide_c).is_some_and(|all| all <= max));
        proptest::prop_assert_eq!(cross_projection_fits(a, b, c), fits);
        let sum = wide_a + wide_b;
        proptest::prop_assert_eq!(mla_qk_head_dim(a, b) as u128, sum.min(max));
    }

    /// Group reorders agree with the iterator reference and preserved full batches stay in bounds.
    #[test]
    fn group_reorder_matches_its_definition(
        indices in proptest::collection::vec(0usize..12, 0..10),
        group_size in 0usize..5,
        batch in 0usize..12,
    ) {
        let expected = if group_size == 0 || !batch.is_multiple_of(group_size) {
            GroupReorder::InvalidGroupSize
        } else {
            match indices
                .iter()
                .enumerate()
                .position(|(row, source)| source / group_size != row / group_size)
            {
                Some(row) => GroupReorder::LeavesGroup { row },
                None => GroupReorder::Preserved,
            }
        };
        let actual = group_reorder(&indices, group_size, batch);
        proptest::prop_assert_eq!(actual, expected);
        if actual == GroupReorder::Preserved && indices.len() == batch {
            proptest::prop_assert!(indices.iter().all(|&source| source < batch));
        }
    }
}
