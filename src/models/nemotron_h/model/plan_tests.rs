//! Compact-batch layouts, execution policies, and cursor arithmetic.

use proptest::prelude::*;

use super::*;

/// Lists the valid columns of every row of a row-major mask.
fn columns_of(mask: &[Vec<bool>]) -> Vec<Vec<usize>> {
    mask.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter_map(|(column, &valid)| valid.then_some(column))
                .collect()
        })
        .collect()
}

/// Checks every map of a plan against its defining equations.
fn assert_plan_is_consistent(columns: &[Vec<usize>], sequence: usize, plan: &CompactPlan) {
    let batch = columns.len();
    let width = columns.iter().map(Vec::len).max().unwrap_or(0);
    let tokens = columns.iter().map(Vec::len).sum::<usize>();
    assert_eq!(plan.width, width);
    assert_eq!(plan.batch(), batch);
    assert_eq!(plan.tokens(), tokens);
    assert_eq!(plan.gather.len(), batch * width);
    assert_eq!(plan.scatter.len(), batch * sequence);
    assert_eq!(plan.unpack.len(), batch * width);
    let mut packed = 0;
    for (row, valid) in columns.iter().enumerate() {
        assert_eq!(plan.counts[row], valid.len());
        for slot in 0..width {
            let compact = row * width + slot;
            if slot < valid.len() {
                assert_eq!(plan.gather[compact], row * sequence + valid[slot]);
                assert_eq!(plan.scatter[row * sequence + valid[slot]], compact);
                assert_eq!(plan.packed[packed], compact);
                assert_eq!(plan.unpack[compact], packed);
                packed += 1;
            } else {
                assert_eq!(plan.gather[compact], row * sequence);
                assert_eq!(plan.unpack[compact], tokens);
            }
        }
        for column in 0..sequence {
            if !valid.contains(&column) {
                assert_eq!(plan.scatter[row * sequence + column], batch * width);
            }
        }
        let expected_last = valid
            .len()
            .checked_sub(1)
            .map_or(batch * width, |last| row * width + last);
        assert_eq!(plan.last[row], expected_last);
    }
    assert_eq!(packed, tokens);
}

/// Left padding, right padding, interior holes, and an inactive row in one batch.
#[test]
fn ragged_batch_plans_every_map() {
    let mask = vec![
        vec![false, false, true, true],
        vec![true, true, false, false],
        vec![true, false, true, false],
        vec![false, false, false, false],
    ];
    let columns = columns_of(&mask);
    let plan = plan_compaction(&columns, 4).unwrap();
    assert_plan_is_consistent(&columns, 4, &plan);
    assert_eq!(plan.width, 2);
    assert_eq!(plan.gather, [2, 3, 4, 5, 8, 10, 12, 12]);
    assert_eq!(plan.packed, [0, 1, 2, 3, 4, 5]);
    assert_eq!(plan.unpack, [0, 1, 2, 3, 4, 5, 6, 6]);
    assert_eq!(plan.last, [1, 3, 5, 8]);
    assert!(!plan.is_dense());
    assert!(!plan.is_identity());
}

/// A fully valid batch is the identity layout.
#[test]
fn dense_batch_is_identity() {
    let columns = vec![vec![0, 1, 2]; 2];
    let plan = plan_compaction(&columns, 3).unwrap();
    assert_plan_is_consistent(&columns, 3, &plan);
    assert!(plan.is_dense());
    assert!(plan.is_identity());
    assert_eq!(plan.scatter, [0, 1, 2, 3, 4, 5]);
}

/// One-token decode with an inactive row is dense in width but not in tokens.
#[test]
fn decode_with_finished_row_is_not_dense() {
    let columns = vec![vec![0], vec![]];
    let plan = plan_compaction(&columns, 1).unwrap();
    assert_plan_is_consistent(&columns, 1, &plan);
    assert_eq!(plan.width, 1);
    assert!(!plan.is_dense());
    assert_eq!(plan.scatter, [0, 2]);
}

/// Every row inactive yields an empty compact batch.
#[test]
fn all_inactive_rows_have_zero_width() {
    let columns = vec![vec![], vec![]];
    let plan = plan_compaction(&columns, 3).unwrap();
    assert_plan_is_consistent(&columns, 3, &plan);
    assert_eq!(plan.width, 0);
    assert!(plan.gather.is_empty());
    assert_eq!(plan.last, [0, 0]);
}

/// Unsorted, duplicated, and out-of-range columns are rejected.
#[test]
fn invalid_columns_are_rejected() {
    assert!(plan_compaction(&[vec![1, 0]], 2).is_none());
    assert!(plan_compaction(&[vec![1, 1]], 2).is_none());
    assert!(plan_compaction(&[vec![2]], 2).is_none());
}

/// Overflowing flat extents are rejected instead of wrapping.
#[cfg(target_pointer_width = "64")]
#[test]
fn overflowing_extents_are_rejected() {
    let columns = vec![vec![usize::MAX - 1]; 2];
    assert!(plan_compaction(&columns, usize::MAX).is_none());
}

/// Valid slots continue every row's numbering; padded slots repeat its last readable position.
#[test]
fn query_positions_follow_working_lengths() {
    assert_eq!(query_positions(&[3, 0], &[2, 2], 2).unwrap(), [3, 4, 0, 1]);
    // Row 0 pads after one token, row 1 is a left-padded prompt, row 2 is a
    // finished row with history, and row 3 has neither tokens nor history.
    assert_eq!(
        query_positions(&[3, 0, 5, 0], &[1, 2, 0, 0], 3).unwrap(),
        [3, 3, 3, 0, 1, 1, 4, 4, 4, 0, 0, 0]
    );
    assert!(query_positions(&[3], &[1, 1], 2).is_none());
    assert!(query_positions(&[usize::MAX], &[2], 2).is_none());
    // Every row's `W + n` must be representable, even when each valid slot's
    // own position fits; a zero-width call reads no slot and needs no check.
    assert!(query_positions(&[1], &[usize::MAX], 1).is_none());
    assert_eq!(query_positions(&[1], &[usize::MAX], 0), Some(Vec::new()));
    // A padded slot repeats the last position instead of computing `W + j`.
    assert_eq!(slot_position(usize::MAX - 1, 1, 5), Some(usize::MAX - 1));
    assert_eq!(slot_position(usize::MAX, 1, 0), None);
    assert_eq!(slot_position(0, 0, 3), Some(0));
    #[cfg(target_pointer_width = "64")]
    assert!(query_positions(&[i64::MAX as usize + 1], &[1], 1).is_none());
}

/// Regression for review NEMR-6: a padded query keeps a readable key inside the window.
///
/// With the former `W_b + j` numbering, the left-padded row of a `[4, 2]`
/// prefill under a two-token window put its padded slot 3 at position 3,
/// beyond the window of both valid keys `{0, 1}`, so every key was excluded.
#[test]
fn padded_queries_keep_a_key_inside_a_sliding_window() {
    let positions = query_positions(&[0, 0], &[4, 2], 4).unwrap();
    assert_eq!(positions, [0, 1, 2, 3, 0, 1, 1, 1]);
    let window = 2;
    for (row, length) in [(0usize, 4i64), (1, 2)] {
        for &query in &positions[row * 4..row * 4 + 4] {
            assert!((0..length).any(|key| key <= query && key + window > query));
        }
    }
}

/// The recurrent update needs exactly one token and existing state.
#[test]
fn mamba_step_follows_the_reference_condition() {
    assert_eq!(mamba_step(1, 3), MambaStep::Recurrent);
    assert_eq!(mamba_step(1, 0), MambaStep::Chunked);
    assert_eq!(mamba_step(2, 3), MambaStep::Chunked);
    assert_eq!(mamba_step(0, 3), MambaStep::Chunked);
}

/// Transactional steps keep one state per single token or snapshots per token.
#[test]
fn transactional_mamba_policy() {
    assert_eq!(
        transactional_mamba(&[1, 0, 1], &[0, 0, 2]),
        Ok(TransactionalMamba::Terminal)
    );
    assert_eq!(
        transactional_mamba(&[3, 1, 0], &[4, 2, 0]),
        Ok(TransactionalMamba::Snapshots)
    );
    assert_eq!(transactional_mamba(&[3, 1], &[4, 0]), Err(1));
    assert_eq!(transactional_mamba(&[2], &[]), Err(0));
    // The first active row without state is reported.
    assert_eq!(transactional_mamba(&[2, 1, 1], &[3, 0, 0]), Err(1));
}

/// MTP truncation only rewinds and keeps the one-token lag.
#[test]
fn mtp_truncation_keeps_the_lag() {
    assert_eq!(mtp_length_after_truncate(5, 4), 4);
    assert_eq!(mtp_length_after_truncate(3, 4), 2);
    assert_eq!(mtp_length_after_truncate(0, 4), 0);
    assert_eq!(mtp_length_after_truncate(1, 0), 0);
}

/// An EOS draft is never evaluated by the target.
#[test]
fn verification_skips_a_terminal_draft() {
    assert_eq!(verification_tokens(3, false), Some(4));
    assert_eq!(verification_tokens(3, true), Some(3));
    assert_eq!(verification_tokens(0, true), Some(1));
    assert_eq!(verification_tokens(usize::MAX, false), None);
}

/// Attention and Mamba-2 blocks are numbered separately in schedule order.
#[test]
fn cache_slots_number_each_kind_separately() {
    use CacheKind::{Attention, Recurrent, Stateless};
    let plan = plan_cache_slots(&[
        Recurrent, Stateless, Attention, Recurrent, Stateless, Attention, Recurrent,
    ]);
    assert_eq!(plan.slots, [0, 0, 0, 1, 0, 1, 2]);
    assert_eq!(plan.attention, 2);
    assert_eq!(plan.recurrent, 3);
    let empty = plan_cache_slots(&[]);
    assert!(empty.slots.is_empty());
    assert_eq!((empty.attention, empty.recurrent), (0, 0));
}

/// A window applies only when it is nonzero and shorter than the longest history.
#[test]
fn active_window_needs_a_history_beyond_it() {
    assert_eq!(longest_end(&[3, 7, 1], &[2, 0, 9]), 10);
    assert_eq!(longest_end(&[3, 7, 1], &[2]), 5);
    assert_eq!(longest_end(&[], &[4]), 0);
    assert_eq!(longest_end(&[usize::MAX], &[1]), usize::MAX);
    assert_eq!(active_window(Some(4), 5), Some(4));
    assert_eq!(active_window(Some(5), 5), None);
    assert_eq!(active_window(Some(0), 5), None);
    assert_eq!(active_window(None, 5), None);
}

/// Uniform baselines, context limits, MTP cursors, and convolution widths.
#[test]
fn decode_bookkeeping_helpers() {
    assert!(all_equal(&[]));
    assert!(all_equal(&[4]));
    assert!(all_equal(&[4, 4, 4]));
    assert!(!all_equal(&[4, 4, 5]));
    assert!(exceeds_limit(&[3, 9], &[1, 2], 10));
    assert!(!exceeds_limit(&[3, 8], &[1, 2], 10));
    assert!(!exceeds_limit(&[3, 9], &[1], 10));
    assert!(exceeds_limit(&[usize::MAX], &[1], usize::MAX));
    assert!(exceeds_limit(&[11], &[0], 10));
    assert_eq!(
        mtp_lengths_after_truncate(&[5, 3, 0], &[4, 4, 4]),
        [4, 2, 0]
    );
    assert_eq!(mtp_lengths_after_truncate(&[5, 3], &[4]), [4]);
    assert_eq!(
        convolution_channels(64, 64, 8, 128),
        Some(64 * 64 + 2 * 8 * 128)
    );
    assert_eq!(convolution_channels(usize::MAX, 2, 0, 0), None);
    assert_eq!(convolution_channels(0, 0, usize::MAX, 2), None);
    assert_eq!(convolution_channels(1, usize::MAX, 0, 0), Some(usize::MAX));
    assert_eq!(convolution_channels(1, usize::MAX, 1, 1), None);
}

proptest! {
    /// Cache slots equal the counters the cached pass used to keep.
    #[test]
    fn cache_slots_match_running_counters(kinds in proptest::collection::vec(0_u8..3, 0..12)) {
        let kinds = kinds
            .into_iter()
            .map(|kind| match kind {
                0 => CacheKind::Attention,
                1 => CacheKind::Recurrent,
                _ => CacheKind::Stateless,
            })
            .collect::<Vec<_>>();
        let plan = plan_cache_slots(&kinds);
        let (mut attention, mut recurrent) = (0, 0);
        for (&kind, &slot) in kinds.iter().zip(&plan.slots) {
            match kind {
                CacheKind::Attention => {
                    prop_assert_eq!(slot, attention);
                    attention += 1;
                }
                CacheKind::Recurrent => {
                    prop_assert_eq!(slot, recurrent);
                    recurrent += 1;
                }
                CacheKind::Stateless => prop_assert_eq!(slot, 0),
            }
        }
        prop_assert_eq!((plan.attention, plan.recurrent), (attention, recurrent));
    }

    /// The decode helpers equal the iterator code they replaced.
    #[test]
    fn decode_helpers_match_iterator_references(
        baselines in proptest::collection::vec(0_usize..20, 0..5),
        counts in proptest::collection::vec(0_usize..20, 0..5),
        window in proptest::option::of(0_usize..40),
        limit in 0_usize..40,
    ) {
        let longest = baselines
            .iter()
            .zip(&counts)
            .map(|(&baseline, &count)| baseline + count)
            .max()
            .unwrap_or(0);
        prop_assert_eq!(longest_end(&baselines, &counts), longest);
        prop_assert_eq!(
            active_window(window, longest),
            window.filter(|&window| window < longest).filter(|&window| window > 0)
        );
        prop_assert_eq!(
            all_equal(&baselines),
            baselines.windows(2).all(|pair| pair[0] == pair[1])
        );
        prop_assert_eq!(
            exceeds_limit(&baselines, &counts, limit),
            baselines.iter().zip(&counts).any(|(&length, &count)| length + count > limit)
        );
        let expected = baselines
            .iter()
            .zip(&counts)
            .map(|(&target, &mtp)| mtp_length_after_truncate(target, mtp))
            .collect::<Vec<_>>();
        prop_assert_eq!(mtp_lengths_after_truncate(&baselines, &counts), expected);
    }

    /// Convolution widths equal the former checked chain on arbitrary extents.
    #[test]
    fn convolution_channels_match_the_checked_chain(
        heads in prop_oneof![0_usize..9, Just(usize::MAX / 2), Just(usize::MAX)],
        head_dim in prop_oneof![0_usize..9, Just(usize::MAX / 3), Just(usize::MAX)],
        groups in prop_oneof![0_usize..9, Just(usize::MAX / 4)],
        state in prop_oneof![0_usize..9, Just(usize::MAX / 2)],
    ) {
        let expected = heads.checked_mul(head_dim).and_then(|intermediate| {
            groups
                .checked_mul(state)
                .and_then(|state| state.checked_mul(2))
                .and_then(|state| intermediate.checked_add(state))
        });
        prop_assert_eq!(convolution_channels(heads, head_dim, groups, state), expected);
    }

    /// Every random mask produces maps satisfying their defining equations.
    #[test]
    fn random_masks_plan_consistently(
        mask in proptest::collection::vec(proptest::collection::vec(any::<bool>(), 5), 1..5),
    ) {
        let columns = columns_of(&mask);
        let plan = plan_compaction(&columns, 5).unwrap();
        assert_plan_is_consistent(&columns, 5, &plan);
        // The tensor wrappers skip the maps of dense and identity plans.
        if plan.is_dense() {
            prop_assert!(plan.packed.iter().enumerate().all(|(i, &source)| source == i));
            prop_assert!(plan.unpack.iter().enumerate().all(|(i, &source)| source == i));
            if plan.width > 0 {
                for (row, &last) in plan.last.iter().enumerate() {
                    prop_assert_eq!(last, row * plan.width + plan.width - 1);
                }
            }
        }
        if plan.is_identity() {
            prop_assert!(plan.gather.iter().enumerate().all(|(i, &source)| source == i));
            prop_assert!(plan.scatter.iter().enumerate().all(|(i, &source)| source == i));
        }
    }

    /// Every slot position follows its defining rule and sees a readable key.
    ///
    /// For a row with `W + n > 0` the position `q` satisfies `q < W + n`, so
    /// the key at `q` is readable and lies inside every window `w >= 1`.
    #[test]
    fn query_positions_follow_slot_rules(
        rows in proptest::collection::vec((0usize..6, 0usize..4), 1..5),
        extra in 0usize..3,
    ) {
        let baselines = rows.iter().map(|row| row.0).collect::<Vec<_>>();
        let counts = rows.iter().map(|row| row.1).collect::<Vec<_>>();
        let width = counts.iter().copied().max().unwrap_or(0) + extra;
        let positions = query_positions(&baselines, &counts, width).unwrap();
        prop_assert_eq!(positions.len(), rows.len() * width);
        for (row, (&baseline, &count)) in baselines.iter().zip(&counts).enumerate() {
            let end = baseline + count;
            for slot in 0..width {
                let position = usize::try_from(positions[row * width + slot]).unwrap();
                prop_assert_eq!(Some(position), slot_position(baseline, count, slot));
                if slot < count {
                    prop_assert_eq!(position, baseline + slot);
                } else {
                    prop_assert_eq!(position, end.max(1) - 1);
                }
                prop_assert!(position <= baseline + slot);
                if end > 0 {
                    prop_assert!(position < end);
                }
            }
        }
    }

    /// The MTP mapping never advances and preserves `mtp + 1 <= max(target, 1)`.
    #[test]
    fn mtp_truncation_invariant(target in 0usize..64, mtp in 0usize..64) {
        let result = mtp_length_after_truncate(target, mtp);
        prop_assert!(result <= mtp);
        prop_assert!(result < target.max(1));
    }
}
