//! Counting-sort dispatch plans and dispatch blocks: worked examples, invalid IDs, and properties.

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

/// Splits experts into maximal runs within the budget; an oversized expert stands alone.
#[test]
fn dispatch_blocks_respect_the_budget_without_splitting_experts() {
    // Expert counts [3, 0, 4, 1, 6, 0]: offsets are their prefix sums.
    let offsets = [0, 3, 3, 7, 8, 14, 14];
    let blocks = |budget| {
        let mut ends = Vec::new();
        let mut first = 0;
        while first + 1 < offsets.len() {
            first = next_dispatch_block(&offsets, first, budget);
            ends.push(first);
        }
        ends
    };
    // Experts 0..2 hold 3 rows (adding expert 2 would make 7 > 4); expert 2
    // fills the budget alone; expert 4's 6 rows exceed it and stand alone.
    assert_eq!(blocks(4), [2, 3, 4, 5, 6]);
    assert_eq!(blocks(7), [3, 6]);
    assert_eq!(blocks(14), [6]);
    assert_eq!(blocks(usize::MAX), [6]);
    // A zero budget still makes progress one expert at a time.
    assert_eq!(blocks(0), [1, 2, 3, 4, 5, 6]);
    // A single expert is always one block.
    assert_eq!(next_dispatch_block(&[0, 9], 0, 1), 1);
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(96))]

    /// Blocks partition the experts into maximal runs that fit the budget.
    #[test]
    fn dispatch_blocks_partition_experts_maximally(
        counts in proptest::collection::vec(0usize..20, 1..12),
        budget in 0usize..60,
    ) {
        let mut offsets = vec![0];
        for count in &counts {
            offsets.push(offsets.last().unwrap() + count);
        }
        let experts = counts.len();
        let mut first = 0;
        while first < experts {
            let end = next_dispatch_block(&offsets, first, budget);
            proptest::prop_assert!(first < end && end <= experts);
            let rows = offsets[end] - offsets[first];
            proptest::prop_assert!(end == first + 1 || rows <= budget);
            if end < experts {
                proptest::prop_assert!(offsets[end + 1] - offsets[first] > budget);
            }
            first = end;
        }
    }

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
