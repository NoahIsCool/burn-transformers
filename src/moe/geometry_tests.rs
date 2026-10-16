//! Expert-bank, router, and group geometry checks: worked examples, violation order, and properties.

use super::*;

/// Zero widths, overflowing matrices, and bad top-k are reported in validation order.
#[test]
fn expert_bank_reports_the_first_violation() {
    assert_eq!(check_expert_bank(4, 8, 3, 2), Ok(()));
    assert_eq!(check_expert_bank(4, 8, 3, 3), Ok(()));
    for (hidden, intermediate, experts) in [(0, 8, 3), (4, 0, 3), (4, 8, 0)] {
        assert_eq!(
            check_expert_bank(hidden, intermediate, experts, 0),
            Err(ExpertBankViolation::ZeroWidth)
        );
    }
    assert_eq!(
        check_expert_bank(usize::MAX, 2, 2, 0),
        Err(ExpertBankViolation::ExpertMatrixOverflow)
    );
    assert_eq!(
        check_expert_bank(usize::MAX / 2, 2, 3, 0),
        Err(ExpertBankViolation::RouterMatrixOverflow)
    );
    assert_eq!(
        check_expert_bank(4, 8, 3, 0),
        Err(ExpertBankViolation::ExpertsPerToken)
    );
    assert_eq!(
        check_expert_bank(4, 8, 3, 4),
        Err(ExpertBankViolation::ExpertsPerToken)
    );
}

/// An empty router is reported before an invalid top-k.
#[test]
fn router_cardinality_accepts_one_through_all_experts() {
    assert_eq!(
        check_router_cardinality(0, 0),
        Err(RouterCardinalityViolation::NoExperts)
    );
    assert_eq!(
        check_router_cardinality(4, 0),
        Err(RouterCardinalityViolation::ExpertsPerToken)
    );
    assert_eq!(
        check_router_cardinality(4, 5),
        Err(RouterCardinalityViolation::ExpertsPerToken)
    );
    assert_eq!(check_router_cardinality(4, 1), Ok(()));
    assert_eq!(check_router_cardinality(4, 4), Ok(()));
}

/// DeepSeek-V3-shaped groups pass; each invalid geometry names its first failure.
#[test]
fn group_geometry_reports_width_or_first_violation() {
    // 256 experts in 8 groups of 32, keeping 4 groups for top-8 routing.
    assert_eq!(check_group_geometry(256, 8, 8, 4), Ok(32));
    // One group keeps every expert.
    assert_eq!(check_group_geometry(6, 6, 1, 1), Ok(6));
    assert_eq!(
        check_group_geometry(6, 2, 0, 1),
        Err(GroupGeometryViolation::Groups)
    );
    assert_eq!(
        check_group_geometry(6, 2, 4, 1),
        Err(GroupGeometryViolation::Groups)
    );
    assert_eq!(
        check_group_geometry(6, 2, 3, 0),
        Err(GroupGeometryViolation::TopGroups)
    );
    assert_eq!(
        check_group_geometry(6, 2, 3, 4),
        Err(GroupGeometryViolation::TopGroups)
    );
    assert_eq!(
        check_group_geometry(6, 5, 3, 2),
        Err(GroupGeometryViolation::TooFewEligible { eligible: 4 })
    );
    // The eligible count never overflows, even for the largest expert count.
    assert_eq!(
        check_group_geometry(usize::MAX, 1, usize::MAX, usize::MAX),
        Ok(1)
    );
}

/// Group widths tile the experts and rank at most two scores, never more than a group holds.
#[test]
fn group_widths_tile_the_experts() {
    assert_eq!(
        group_widths(256, 8),
        GroupWidths {
            per_group: 32,
            ranked: 2
        }
    );
    assert_eq!(
        group_widths(6, 6),
        GroupWidths {
            per_group: 1,
            ranked: 1
        }
    );
    assert_eq!(
        group_widths(0, 3),
        GroupWidths {
            per_group: 0,
            ranked: 0
        }
    );
    for groups in [1, 2, 4, 8] {
        let widths = group_widths(64, groups);
        assert_eq!(widths.per_group * groups, 64);
        assert!(widths.ranked <= widths.per_group);
    }
}

/// The expert-ID range check is exact at both ends.
#[test]
fn expert_id_range_is_exact() {
    assert!(expert_ids_within(0, 3, 4));
    assert!(!expert_ids_within(0, 4, 4));
    assert!(!expert_ids_within(-1, 2, 4));
    assert!(!expert_ids_within(0, -1, 4));
    assert!(!expert_ids_within(0, 0, 0));
    assert!(expert_ids_within(i64::MAX - 1, i64::MAX - 1, usize::MAX));
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(256))]

    /// Group geometry matches wide integer arithmetic for every small configuration.
    #[test]
    fn group_geometry_matches_wide_arithmetic(
        experts in 0usize..64,
        experts_per_token in 0usize..64,
        groups in 0usize..16,
        top_groups in 0usize..18,
    ) {
        let expected = if groups == 0 || experts % groups != 0 {
            Err(GroupGeometryViolation::Groups)
        } else if top_groups == 0 || top_groups > groups {
            Err(GroupGeometryViolation::TopGroups)
        } else {
            let eligible = top_groups as u128 * (experts / groups) as u128;
            if eligible < experts_per_token as u128 {
                Err(GroupGeometryViolation::TooFewEligible {
                    eligible: eligible as usize,
                })
            } else {
                Ok(experts / groups)
            }
        };
        proptest::prop_assert_eq!(
            check_group_geometry(experts, experts_per_token, groups, top_groups),
            expected
        );
    }

    /// Expert-bank checks match wide integer arithmetic near the overflow boundary.
    #[test]
    fn expert_bank_matches_wide_arithmetic(
        hidden in proptest::prop_oneof![0usize..8, (usize::MAX / 4)..=usize::MAX],
        intermediate in 0usize..8,
        experts in 0usize..8,
        experts_per_token in 0usize..9,
    ) {
        let fits = |width: usize| hidden as u128 * width as u128 <= usize::MAX as u128;
        let expected = if hidden == 0 || intermediate == 0 || experts == 0 {
            Err(ExpertBankViolation::ZeroWidth)
        } else if !fits(intermediate) {
            Err(ExpertBankViolation::ExpertMatrixOverflow)
        } else if !fits(experts) {
            Err(ExpertBankViolation::RouterMatrixOverflow)
        } else if experts_per_token == 0 || experts_per_token > experts {
            Err(ExpertBankViolation::ExpertsPerToken)
        } else {
            Ok(())
        };
        proptest::prop_assert_eq!(
            check_expert_bank(hidden, intermediate, experts, experts_per_token),
            expected
        );
    }
}
