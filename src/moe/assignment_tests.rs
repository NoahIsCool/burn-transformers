//! Counting-sort dispatch plans, dispatch blocks, and routed-ID flattening: worked examples,
//! invalid inputs, and properties.

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

/// Block boundaries partition the experts exactly as repeated `next_dispatch_block` calls do.
#[test]
fn dispatch_block_boundaries_partition_the_experts() {
    let offsets = [0, 3, 3, 7, 8, 14, 14];
    assert_eq!(dispatch_blocks(&offsets, 4), [0, 2, 3, 4, 5, 6]);
    assert_eq!(dispatch_blocks(&offsets, 7), [0, 3, 6]);
    assert_eq!(dispatch_blocks(&offsets, usize::MAX), [0, 6]);
    assert_eq!(dispatch_blocks(&offsets, 0), [0, 1, 2, 3, 4, 5, 6]);
    // Without experts there are no blocks.
    assert_eq!(dispatch_blocks(&[0], 5), [0]);
}

/// Range lengths are the per-expert counts; degenerate offset lists have none.
#[test]
fn range_lengths_are_per_expert_counts() {
    assert_eq!(range_lengths(&[0, 3, 3, 7, 8]), [3, 0, 4, 1]);
    assert!(range_lengths(&[5]).is_empty());
    assert!(range_lengths(&[]).is_empty());
}

/// An expert's rows inside a block are its offset relative to the block start.
#[test]
fn block_expert_rows_locate_each_expert_inside_its_block() {
    let offsets = [0, 3, 3, 7, 8];
    assert_eq!(block_expert_rows(&offsets, 0, 0), (0, 3));
    assert_eq!(block_expert_rows(&offsets, 0, 1), (3, 0));
    assert_eq!(block_expert_rows(&offsets, 0, 2), (3, 4));
    assert_eq!(block_expert_rows(&offsets, 2, 3), (4, 1));
    assert_eq!(block_expert_rows(&offsets, 3, 3), (0, 1));
}

/// Assignment counts must be addressable by `i64` indices.
#[test]
fn routed_assignment_count_is_exact_or_rejected() {
    assert_eq!(routed_assignment_count(5, 3), Some(15));
    assert_eq!(routed_assignment_count(0, 3), Some(0));
    assert_eq!(
        routed_assignment_count(i64::MAX as usize, 1),
        Some(i64::MAX as usize)
    );
    assert_eq!(routed_assignment_count(i64::MAX as usize, 2), None);
    assert_eq!(routed_assignment_count(usize::MAX, 2), None);
}

/// Host IDs flatten token-major; flags are dropped, checked, and reported in order.
#[test]
fn routed_ids_flatten_and_check_flags() {
    // Two tokens, two slots, and a flag column of zeros.
    assert_eq!(
        routed_ids(&[3, 1, 0, 2, 0, 0], 2, 2, true),
        Ok(vec![3, 1, 2, 0])
    );
    assert_eq!(routed_ids(&[3, 1, 2, 0], 2, 2, false), Ok(vec![3, 1, 2, 0]));
    // The first flagged token is reported even when a later one is flagged too.
    assert_eq!(
        routed_ids(&[3, 1, 0, 2, 0, 1, 4, 4, 1], 3, 2, true),
        Err(RoutedIdsError::NonFinite { token: 1 })
    );
    for (values, flagged) in [(&[1, 2, 3][..], false), (&[1, 2, 0, 3][..], true)] {
        assert_eq!(
            routed_ids(values, 2, 2, flagged),
            Err(RoutedIdsError::Length {
                found: values.len()
            })
        );
    }
    assert_eq!(
        routed_ids(&[], 1, 1, false),
        Err(RoutedIdsError::Length { found: 0 })
    );
    assert_eq!(routed_ids(&[], 0, 4, true), Ok(vec![]));
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(96))]

    /// Every assignment sits at its expert's offset plus its earlier same-expert assignments.
    #[test]
    fn plan_places_each_assignment_at_its_slot(
        experts in 1usize..10,
        seeds in proptest::collection::vec(0usize..1000, 0..60),
    ) {
        let ids = seeds.iter().map(|&seed| (seed % experts) as i64).collect::<Vec<_>>();
        let plan = plan_expert_assignments(&ids, experts).expect("in-range IDs");
        for (assignment, &id) in ids.iter().enumerate() {
            let earlier = ids[..assignment].iter().filter(|&&other| other == id).count();
            proptest::prop_assert_eq!(
                plan.order[plan.offsets[id as usize] + earlier],
                assignment
            );
        }
    }

    /// Block boundaries agree with the iterated single-block planner.
    #[test]
    fn dispatch_blocks_match_iterated_planner(
        counts in proptest::collection::vec(0usize..20, 0..12),
        budget in 0usize..60,
    ) {
        let mut offsets = vec![0];
        for count in &counts {
            offsets.push(offsets.last().unwrap() + count);
        }
        let mut expected = vec![0];
        while *expected.last().unwrap() < counts.len() {
            expected.push(next_dispatch_block(&offsets, *expected.last().unwrap(), budget));
        }
        proptest::prop_assert_eq!(dispatch_blocks(&offsets, budget), expected);
        proptest::prop_assert_eq!(range_lengths(&offsets), counts);
    }

    /// Flattening drops the flag column and keeps every ID in row-major order.
    #[test]
    fn routed_ids_match_row_major_columns(
        rows in proptest::collection::vec(proptest::collection::vec(-3i64..9, 1..5), 1..6),
        flagged in proptest::bool::ANY,
    ) {
        let slots = rows[0].len();
        let rows = rows.into_iter().map(|mut row| { row.resize(slots, 0); row }).collect::<Vec<_>>();
        let mut values = Vec::new();
        for row in &rows {
            values.extend_from_slice(row);
            if flagged {
                values.push(0);
            }
        }
        let expected = rows.concat();
        proptest::prop_assert_eq!(routed_ids(&values, rows.len(), slots, flagged), Ok(expected));
    }

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
