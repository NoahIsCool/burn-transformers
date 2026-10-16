//! Executable checks of the Verus-proved metadata layouts.
//!
//! The proofs run outside Cargo (see `verus/README.md`); these tests compare
//! each layout with the iterator expression it replaced, so a regression is
//! caught in every ordinary test run even where Verus is not installed.

use proptest::prelude::*;

use super::{
    ChunkCall, any_valid, chunk_activity, chunk_call, count_valid, deinterleave, fill_row_lengths,
    history_token_id, last_valid, push_constraint_row, push_history_row, record_scatter,
    repeat_row,
};

/// Mask scans handle empty spans, single slots, and spans without valid entries.
#[test]
fn mask_scans_cover_span_boundaries() {
    let valid = [false, true, true, false, true, false];
    assert_eq!(count_valid(&valid, 0, 6), 3);
    assert_eq!(count_valid(&valid, 2, 2), 0);
    assert_eq!(count_valid(&valid, 5, 6), 0);
    assert_eq!(last_valid(&valid, 0, 6), Some(4));
    assert_eq!(last_valid(&valid, 0, 4), Some(2));
    assert_eq!(last_valid(&valid, 3, 4), None);
    assert_eq!(last_valid(&valid, 3, 3), None);
    assert!(any_valid(&valid, 3, 5));
    assert!(!any_valid(&valid, 5, 6));
    assert!(!any_valid(&valid, 1, 1));
}

/// Row lengths count each row and name the first empty one.
#[test]
fn row_lengths_report_the_first_empty_row() {
    let valid = [true, false, true, true, false, false, false, true];
    let mut lengths = Vec::new();
    assert_eq!(fill_row_lengths(&mut lengths, &valid, 4, 2), Err(2));
    assert_eq!(lengths, [1, 2]);
    let mut lengths = Vec::new();
    assert_eq!(fill_row_lengths(&mut lengths, &valid[..4], 2, 2), Ok(()));
    assert_eq!(lengths, [1, 2]);
    let mut lengths = Vec::new();
    assert_eq!(fill_row_lengths(&mut lengths, &[], 0, 7), Ok(()));
    assert!(lengths.is_empty());
    let mut lengths = Vec::new();
    assert_eq!(fill_row_lengths(&mut lengths, &[], 3, 0), Err(0));
}

/// Chunks are skipped, prefilled, or decoded by their rows' activity.
#[test]
fn chunk_calls_follow_row_activity() {
    // Row 0 is valid in columns 0..2, row 1 only in column 3.
    let valid = [true, true, false, false, false, false, false, true];
    assert_eq!(chunk_activity(&valid, 2, 4, 0, 2), [true, false]);
    assert_eq!(chunk_activity(&valid, 2, 4, 2, 3), [false, false]);
    assert_eq!(chunk_activity(&valid, 2, 4, 1, 4), [true, true]);
    assert_eq!(chunk_call(&[true, false]), ChunkCall::Decode);
    assert_eq!(chunk_call(&[false, false]), ChunkCall::Skip);
    assert_eq!(chunk_call(&[true, true]), ChunkCall::Prefill);
    assert_eq!(chunk_call(&[]), ChunkCall::Skip);
}

/// Scatter rows are laid out in order and padded with no-op entries.
#[test]
fn scatter_rows_are_ordered_and_padded() {
    let mut indices = vec![-7];
    assert_eq!(push_constraint_row(&mut indices, &[9, 8], true, &[3], 5), 3);
    assert_eq!(indices, [-7, 9, 8, 3, 0, 0]);
    assert_eq!(
        push_constraint_row(&mut indices, &[9, 8], false, &[3], 1),
        1
    );
    assert_eq!(indices, [-7, 9, 8, 3, 0, 0, 3]);
    assert_eq!(push_constraint_row(&mut indices, &[], true, &[], 0), 0);
    assert_eq!(indices.len(), 7);

    let (mut ids, mut observed, mut generated) = (Vec::new(), Vec::new(), Vec::new());
    push_history_row(&mut ids, &mut observed, &mut generated, &[5, 6, 7], 1, 4);
    push_history_row(&mut ids, &mut observed, &mut generated, &[], 0, 4);
    assert_eq!(ids, [5, 6, 7, 0, 0, 0, 0, 0]);
    assert_eq!(observed, [1, 1, 1, 0, 0, 0, 0, 0]);
    assert_eq!(generated, [0, 1, 1, 0, 0, 0, 0, 0]);

    assert_eq!(
        record_scatter(&[4, 5, 6], &[true, false, true]),
        (vec![4, 0, 6], vec![1, 0, 1])
    );
    assert_eq!(repeat_row(&[1_i64, 2], 3), [1, 2, 1, 2, 1, 2]);
    assert_eq!(repeat_row(&[0.5_f64], 2), [0.5, 0.5]);
    assert!(repeat_row(&[1_u8], 0).is_empty());
}

/// Packed readbacks split into IDs and scores; history values must be vocabulary IDs.
#[test]
fn readback_and_token_conversion_boundaries() {
    assert_eq!(
        deinterleave(&[1.0, -0.5, 2.0, -1.5]),
        (vec![1.0, 2.0], vec![-0.5, -1.5])
    );
    assert_eq!(deinterleave(&[1, 2, 3]), (vec![1, 3], vec![2]));
    assert_eq!(deinterleave::<u8>(&[]), (vec![], vec![]));
    assert_eq!(history_token_id(0, 1), Some(0));
    assert_eq!(history_token_id(1, 1), None);
    assert_eq!(history_token_id(-1, 8), None);
    assert_eq!(
        history_token_id(i64::from(u32::MAX), usize::MAX),
        Some(u32::MAX)
    );
    assert_eq!(history_token_id(i64::from(u32::MAX) + 1, usize::MAX), None);
}

proptest! {
    /// Mask scans agree with the iterator expressions they replaced.
    #[test]
    fn mask_scans_match_iterators(
        valid in proptest::collection::vec(any::<bool>(), 0..24),
        a in 0_usize..25,
        b in 0_usize..25,
    ) {
        let (start, end) = (a.min(b).min(valid.len()), a.max(b).min(valid.len()));
        let span = &valid[start..end];
        prop_assert_eq!(count_valid(&valid, start, end), span.iter().filter(|&&v| v).count());
        prop_assert_eq!(
            last_valid(&valid, start, end),
            span.iter().rposition(|&v| v).map(|offset| start + offset)
        );
        prop_assert_eq!(any_valid(&valid, start, end), span.iter().any(|&v| v));
    }

    /// Per-row scans and chunk calls agree with the original row-by-row loops.
    #[test]
    fn row_scans_match_iterators(
        batch in 0_usize..5,
        sequence in 0_usize..6,
        seed in proptest::collection::vec(any::<bool>(), 30),
        a in 0_usize..7,
        b in 0_usize..7,
    ) {
        let valid = &seed[..batch * sequence];
        let mut lengths = Vec::new();
        let filled = fill_row_lengths(&mut lengths, valid, batch, sequence);
        let mut expected = Vec::new();
        let mut first_empty = None;
        for row in 0..batch {
            let length = valid[row * sequence..(row + 1) * sequence].iter().filter(|&&v| v).count();
            if length == 0 {
                first_empty = Some(row);
                break;
            }
            expected.push(length);
        }
        prop_assert_eq!(filled, first_empty.map_or(Ok(()), Err));
        prop_assert_eq!(lengths, expected);

        let (start, end) = (a.min(b).min(sequence), a.max(b).min(sequence));
        let active = chunk_activity(valid, batch, sequence, start, end);
        let reference = (0..batch)
            .map(|row| valid[row * sequence + start..row * sequence + end].iter().any(|&v| v))
            .collect::<Vec<_>>();
        prop_assert_eq!(&active, &reference);
        let expected_call = if reference.iter().all(|&row| !row) {
            ChunkCall::Skip
        } else if reference.iter().all(|&row| row) {
            ChunkCall::Prefill
        } else {
            ChunkCall::Decode
        };
        prop_assert_eq!(chunk_call(&active), expected_call);
    }

    /// Scatter layouts agree with the iterator constructions they replaced.
    #[test]
    fn scatter_layouts_match_iterators(
        tokens in proptest::collection::vec(0_u32..50, 0..8),
        active_seed in proptest::collection::vec(any::<bool>(), 8),
        eos in proptest::collection::vec(0_u32..50, 0..3),
        suppress in any::<bool>(),
        extra in 0_usize..3,
        prompt_length in 0_usize..9,
        batch in 0_usize..4,
    ) {
        let active = &active_seed[..tokens.len()];
        let (ids, increments) = record_scatter(&tokens, active);
        let expected_ids = tokens
            .iter()
            .zip(active)
            .map(|(&token, &is_active)| if is_active { i64::from(token) } else { 0 })
            .collect::<Vec<_>>();
        let expected_increments = active.iter().map(|&is_active| i64::from(is_active)).collect::<Vec<_>>();
        prop_assert_eq!(ids, expected_ids);
        prop_assert_eq!(increments, expected_increments);

        // Hard-constraint rows: push, then resize to the padded width.
        let real = if suppress { eos.len() } else { 0 } + tokens.len();
        let width = real + extra;
        let mut indices = vec![1, 2];
        prop_assert_eq!(push_constraint_row(&mut indices, &eos, suppress, &tokens, width), real);
        let mut expected = vec![1, 2];
        if suppress {
            expected.extend(eos.iter().map(|&token| i64::from(token)));
        }
        expected.extend(tokens.iter().map(|&token| i64::from(token)));
        expected.resize(2 + width, 0);
        prop_assert_eq!(indices, expected);

        // History-count rows over a width that may exceed the row.
        let width = tokens.len() + extra;
        let (mut ids, mut observed, mut generated) = (vec![9], vec![9], vec![9]);
        push_history_row(&mut ids, &mut observed, &mut generated, &tokens, prompt_length, width);
        let (mut expected_ids, mut expected_observed, mut expected_generated) = (vec![9], vec![9], vec![9]);
        for column in 0..width {
            let token = tokens.get(column).copied();
            expected_ids.push(token.map_or(0, i64::from));
            expected_observed.push(i64::from(token.is_some()));
            expected_generated.push(i64::from(token.is_some() && column >= prompt_length));
        }
        prop_assert_eq!(ids, expected_ids);
        prop_assert_eq!(observed, expected_observed);
        prop_assert_eq!(generated, expected_generated);

        // Repeated rows equal the original `cycle().take()` construction.
        if !tokens.is_empty() {
            let expected = tokens.iter().copied().cycle().take(batch * tokens.len()).collect::<Vec<_>>();
            prop_assert_eq!(repeat_row(&tokens, batch), expected);
        }
    }

    /// Readback splitting and token conversion agree with the original iterators.
    #[test]
    fn readback_and_tokens_match_iterators(
        values in proptest::collection::vec(any::<i64>(), 0..12),
        value in any::<i64>(),
        vocabulary in 0_usize..1 << 34,
    ) {
        let (evens, odds) = deinterleave(&values);
        prop_assert_eq!(evens, values.iter().step_by(2).copied().collect::<Vec<_>>());
        prop_assert_eq!(odds, values.iter().skip(1).step_by(2).copied().collect::<Vec<_>>());
        for candidate in [value, value.rem_euclid(1 << 33), value.rem_euclid(64)] {
            let expected = u32::try_from(candidate)
                .ok()
                .filter(|&token| (token as usize) < vocabulary);
            prop_assert_eq!(history_token_id(candidate, vocabulary), expected);
        }
    }
}
