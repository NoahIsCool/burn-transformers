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

/// Query positions continue every row's numbering, including padded slots.
#[test]
fn query_positions_follow_working_lengths() {
    assert_eq!(query_positions(&[3, 0], 2).unwrap(), [3, 4, 0, 1]);
    assert!(query_positions(&[usize::MAX], 2).is_none());
    #[cfg(target_pointer_width = "64")]
    assert!(query_positions(&[i64::MAX as usize + 1], 1).is_none());
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

proptest! {
    /// Every random mask produces maps satisfying their defining equations.
    #[test]
    fn random_masks_plan_consistently(
        mask in proptest::collection::vec(proptest::collection::vec(any::<bool>(), 5), 1..5),
    ) {
        let columns = columns_of(&mask);
        let plan = plan_compaction(&columns, 5).unwrap();
        assert_plan_is_consistent(&columns, 5, &plan);
    }

    /// The MTP mapping never advances and preserves `mtp + 1 <= max(target, 1)`.
    #[test]
    fn mtp_truncation_invariant(target in 0usize..64, mtp in 0usize..64) {
        let result = mtp_length_after_truncate(target, mtp);
        prop_assert!(result <= mtp);
        prop_assert!(result < target.max(1));
    }
}
