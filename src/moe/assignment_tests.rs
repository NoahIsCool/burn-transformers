//! Counting-sort dispatch plans: worked examples, invalid IDs, and a naive-grouping property.

use super::*;

/// A small routing table groups assignments expert-major and token-major.
#[test]
fn groups_assignments_stably_by_expert() {
    // Tokens x slots = 3 x 2: [[0, 2], [2, 0], [0, 2]].
    let plan = plan_expert_assignments(&[0, 2, 2, 0, 0, 2], 4).expect("valid IDs");
    assert_eq!(plan.offsets, [0, 3, 3, 6, 6]);
    assert_eq!(plan.order, [0, 3, 4, 1, 2, 5]);
}

/// Empty routing and unused experts produce empty ranges.
#[test]
fn empty_routing_and_unused_experts_have_empty_ranges() {
    let plan = plan_expert_assignments(&[], 3).expect("no assignments");
    assert_eq!(plan.offsets, [0, 0, 0, 0]);
    assert!(plan.order.is_empty());
    let plan = plan_expert_assignments(&[], 0).expect("no experts and no assignments");
    assert_eq!(plan.offsets, [0]);
}

/// The first out-of-range ID is reported with its flattened position.
#[test]
fn reports_the_first_invalid_expert_id() {
    for (ids, assignment, expert) in [
        (vec![0, -1, 5], 1, -1),
        (vec![0, 1, 3], 2, 3),
        (vec![i64::MIN], 0, i64::MIN),
        (vec![1, i64::MAX], 1, i64::MAX),
    ] {
        assert_eq!(
            plan_expert_assignments(&ids, 3),
            Err(InvalidExpertId { assignment, expert })
        );
    }
    assert_eq!(
        plan_expert_assignments(&[0], 0),
        Err(InvalidExpertId {
            assignment: 0,
            expert: 0
        })
    );
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(96))]

    /// Each range equals the naive per-expert scan of assignment indices.
    #[test]
    fn plan_matches_naive_grouping(
        experts in 1usize..10,
        seeds in proptest::collection::vec(0usize..1000, 0..60),
    ) {
        let ids = seeds.iter().map(|&seed| (seed % experts) as i64).collect::<Vec<_>>();
        let plan = plan_expert_assignments(&ids, experts).expect("in-range IDs");
        proptest::prop_assert_eq!(plan.offsets.len(), experts + 1);
        proptest::prop_assert_eq!(plan.order.len(), ids.len());
        for expert in 0..experts {
            let naive = (0..ids.len())
                .filter(|&index| ids[index] == expert as i64)
                .collect::<Vec<_>>();
            proptest::prop_assert_eq!(
                &plan.order[plan.offsets[expert]..plan.offsets[expert + 1]],
                naive.as_slice()
            );
        }
    }
}
