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
