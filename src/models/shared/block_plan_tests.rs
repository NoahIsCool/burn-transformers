//! Batch planning against brute-force oracles, edge cases, and property tests.

use proptest::prelude::*;

use super::*;

/// Gather and scatter round-trip every valid column and zero every padding column.
#[test]
fn row_maps_pack_left_and_restore_columns() {
    let maps = row_maps(&[1, 3], 5, 4);
    assert_eq!(maps.gather, [1, 3, 1, 1]);
    assert_eq!(maps.scatter, [4, 0, 4, 1, 4]);
    let full = row_maps(&[0, 1, 2], 3, 3);
    assert_eq!(full.gather, [0, 1, 2]);
    assert_eq!(full.scatter, [0, 1, 2]);
    let empty = row_maps(&[], 2, 1);
    assert_eq!(empty.gather, [0]);
    assert_eq!(empty.scatter, [1, 1]);
}

/// Extents are the widest row and the furthest cache end; overflow and mismatches fail.
#[test]
fn batch_extents_report_width_and_cache_end() {
    assert_eq!(batch_extents(&[3, 0, 5], &[1, 4, 0]), Some((4, 5)));
    assert_eq!(batch_extents(&[], &[]), Some((0, 0)));
    assert_eq!(batch_extents(&[usize::MAX], &[1]), None);
    assert_eq!(batch_extents(&[0, 1], &[1]), None);
}

/// Limits are prefix maxima of each active row's need and never exceed the extent.
#[test]
fn key_limits_cover_every_active_row() {
    // Row 0 is long but has one token; row 1 is short and has three.
    assert_eq!(key_limits(&[10, 0], &[1, 3], 3, 11), [11, 11, 11]);
    assert_eq!(key_limits(&[0, 0], &[3, 3], 3, 3), [1, 2, 3]);
    // An inactive row needs nothing; the limit is still at least one.
    assert_eq!(key_limits(&[0, 7], &[1, 0], 1, 7), [1]);
}

/// Active entries sit at `L_b + k`; padding reports the row's baseline.
#[test]
fn query_slots_follow_each_row_cursor() {
    assert_eq!(
        query_slots(&[4, 0], &[2, 1], 2, 6),
        [vec![4, 5], vec![0, 0]]
    );
}

/// Identity detection, validity, implicit positions, and index conversion are exact.
#[test]
fn block_helpers_match_their_definitions() {
    assert!(identity_layout(&[3, 3], 3, 3));
    assert!(!identity_layout(&[3, 2], 3, 3));
    assert!(!identity_layout(&[2, 2], 2, 3));
    assert_eq!(query_validity(&[2, 2], 2), None);
    assert_eq!(
        query_validity(&[2, 0], 2),
        Some(vec![vec![true, true], vec![false, false]])
    );
    assert_eq!(
        implicit_positions(&[4, 0, 9], &[2, 1, 0], 2),
        [vec![4, 5], vec![0, 0], vec![0, 0]]
    );
    assert_eq!(clamp_limits(&[1, 3, 9], 4), [1, 3, 4]);
    assert_eq!(clamp_limits(&[], 4), Vec::<usize>::new());
}

/// Rotary contexts take the first active row, raised to one, and per-row fallbacks.
#[test]
fn rotary_contexts_follow_the_first_active_row() {
    let plan = rotary_contexts(&[0, 2, 1], &[9, 5, 5]);
    assert_eq!(
        plan,
        RotaryContexts {
            first: 5,
            uniform: true,
            rows: vec![5, 5, 5]
        }
    );
    let plan = rotary_contexts(&[1, 0, 1], &[0, 9, 4]);
    assert_eq!(
        plan,
        RotaryContexts {
            first: 1,
            uniform: false,
            rows: vec![1, 1, 4]
        }
    );
    let plan = rotary_contexts(&[0, 0], &[3, 4]);
    assert_eq!(
        plan,
        RotaryContexts {
            first: 1,
            uniform: true,
            rows: vec![1, 1]
        }
    );
}

/// The cached-rotary preflight reports the first violation, rows before configurations.
#[test]
fn uncacheable_context_reports_the_first_violation_in_order() {
    let limits = [None, Some(8), Some(4)];
    // Row 0 is fresh, row 1 finished; row 2 exceeds the limit-4 configuration only.
    assert_eq!(
        first_uncacheable_context(&[0, 3, 2, 1], &[1, 0, 1, 1], &[20, 20, 6, 9], &limits),
        Some((6, 4))
    );
    // Row 3 exceeds both limits; the first configuration in order wins.
    assert_eq!(
        first_uncacheable_context(&[0, 3, 2, 1], &[1, 0, 1, 1], &[20, 20, 4, 9], &limits),
        Some((9, 8))
    );
    assert_eq!(
        first_uncacheable_context(&[1, 1], &[1, 1], &[4, 3], &limits),
        None
    );
    assert_eq!(first_uncacheable_context(&[1], &[1], &[99], &[None]), None);
    assert_eq!(first_uncacheable_context(&[1], &[1], &[99], &[]), None);
}

/// Strategy for one row: valid columns of a mask over `sequence` columns.
fn row_columns(sequence: usize) -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(any::<bool>(), sequence)
        .prop_map(|mask| (0..mask.len()).filter(|&column| mask[column]).collect())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Gather/scatter compose to the masked identity for any mask and block width.
    #[test]
    fn row_maps_are_a_masked_identity(
        (sequence, columns, extra) in (1usize..12).prop_flat_map(|sequence| {
            (Just(sequence), row_columns(sequence), 0usize..4)
        }),
    ) {
        let block = columns.len() + extra;
        let maps = row_maps(&columns, sequence, block);
        prop_assert_eq!(maps.gather.len(), block);
        prop_assert_eq!(maps.scatter.len(), sequence);
        for column in 0..sequence {
            let restored = maps.scatter[column];
            match columns.iter().position(|&valid| valid == column) {
                Some(k) => {
                    prop_assert_eq!(restored, k);
                    prop_assert_eq!(maps.gather[k], column);
                }
                None => prop_assert_eq!(restored, block),
            }
        }
        prop_assert!(maps.gather.iter().all(|&column| column < sequence));
    }

    /// Block helpers equal the iterator definitions they replaced.
    #[test]
    fn block_helpers_match_iterator_definitions(
        rows in proptest::collection::vec((0usize..20, 0usize..6), 1..5),
        sequence in 1usize..7,
        contexts in proptest::collection::vec(0usize..9, 4),
    ) {
        let baselines = rows.iter().map(|&(length, _)| length).collect::<Vec<_>>();
        let counts = rows.iter().map(|&(_, count)| count).collect::<Vec<_>>();
        let (block, extent) = batch_extents(&baselines, &counts).unwrap();
        prop_assert_eq!(
            identity_layout(&counts, block, sequence),
            block == sequence && counts.iter().all(|&count| count == sequence)
        );
        prop_assume!(block > 0);
        let slots = query_slots(&baselines, &counts, block, extent).concat();
        let valid = |entry: usize| entry % block < counts[entry / block];
        let implicit = slots
            .iter()
            .enumerate()
            .map(|(entry, &slot)| if valid(entry) { slot } else { 0 })
            .collect::<Vec<_>>();
        prop_assert_eq!(implicit_positions(&baselines, &counts, block).concat(), implicit);
        let flags = counts
            .iter()
            .any(|&count| count != block)
            .then(|| (0..counts.len() * block).map(valid).collect::<Vec<_>>());
        prop_assert_eq!(query_validity(&counts, block).map(|rows| rows.concat()), flags);

        let contexts = &contexts[..counts.len()];
        let active = contexts
            .iter()
            .zip(&counts)
            .filter(|(_, count)| **count > 0)
            .map(|(&context, _)| context)
            .collect::<Vec<_>>();
        let first = active.first().copied().unwrap_or(1).max(1);
        let plan = rotary_contexts(&counts, contexts);
        prop_assert_eq!(plan.first, first);
        prop_assert_eq!(plan.uniform, active.iter().all(|&context| context == first));
        for (row, &count) in counts.iter().enumerate() {
            let expected = match count {
                0 => first,
                _ => contexts[row].max(1),
            };
            prop_assert_eq!(plan.rows[row], expected);
        }
    }

    /// The cached-rotary preflight equals the nested loop it replaced.
    #[test]
    fn uncacheable_context_matches_the_nested_loop(
        rows in proptest::collection::vec((0usize..3, 0usize..3, 0usize..12), 0..5),
        limits in proptest::collection::vec(proptest::option::of(0usize..12), 0..4),
    ) {
        let lengths = rows.iter().map(|&(length, _, _)| length).collect::<Vec<_>>();
        let counts = rows.iter().map(|&(_, count, _)| count).collect::<Vec<_>>();
        let contexts = rows.iter().map(|&(_, _, context)| context).collect::<Vec<_>>();
        let mut reference = None;
        'rows: for (row, (&context, &count)) in contexts.iter().zip(&counts).enumerate() {
            if count == 0 || lengths[row] == 0 {
                continue;
            }
            for limit in limits.iter().flatten() {
                if context > *limit {
                    reference = Some((context, *limit));
                    break 'rows;
                }
            }
        }
        prop_assert_eq!(
            first_uncacheable_context(&lengths, &counts, &contexts, &limits),
            reference
        );
    }

    /// Key limits equal the brute-force prefix maxima, clamped below by one.
    #[test]
    fn key_limits_match_brute_force(
        rows in proptest::collection::vec((0usize..20, 0usize..6), 1..5),
    ) {
        let baselines = rows.iter().map(|&(length, _)| length).collect::<Vec<_>>();
        let counts = rows.iter().map(|&(_, count)| count).collect::<Vec<_>>();
        let (block, extent) = batch_extents(&baselines, &counts).unwrap();
        prop_assume!(extent >= 1);
        let limits = key_limits(&baselines, &counts, block, extent);
        let mut running = 1;
        for (index, &limit) in limits.iter().enumerate() {
            for (&length, &count) in baselines.iter().zip(&counts) {
                if count > index {
                    running = running.max(length + index + 1);
                }
            }
            prop_assert_eq!(limit, running);
            prop_assert!(limit <= extent);
        }
        let slots = query_slots(&baselines, &counts, block, extent);
        for (row, (&length, &count)) in baselines.iter().zip(&counts).enumerate() {
            prop_assert_eq!(slots[row].len(), block);
            for k in 0..count {
                prop_assert_eq!(slots[row][k], length + k);
                prop_assert!(slots[row][k] < limits[k]);
            }
        }
    }
}
