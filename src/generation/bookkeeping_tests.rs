//! Executable checks of the Verus-proved bookkeeping contracts.
//!
//! The proofs run outside Cargo (see `verus/README.md`); these property tests
//! exercise the same contracts against naive references in every ordinary
//! test run, so a regression is caught even where Verus is not installed.

use proptest::prelude::*;

use super::{
    FrontierPlan, PositionError, RowSpanError, SpeculativeCommit, SpeculativeOutcome, StopDecision,
    advance_active_positions, beam_row, ends_with_stop, fill_beam_expansion, frontier_capacity,
    longest_stop_suffix, matrix_row_span, next_position_after, partition_frontier,
    prefill_chunk_end, prompt_lookup_draft_span, prompt_lookup_span, required_cache_capacity,
    row_span, speculative_commit, stop_decision,
};

/// Naive longest-suffix search: maximum length, then smallest index.
fn reference_longest(generated: &[u32], stops: &[Vec<u32>]) -> Option<(usize, usize)> {
    stops
        .iter()
        .enumerate()
        .filter(|(_, stop)| generated.ends_with(stop))
        .map(|(index, stop)| (index, stop.len()))
        .max_by_key(|&(index, length)| (length, usize::MAX - index))
}

/// Stop matching handles empty, longer-than-history, and tied stops.
#[test]
fn stop_matching_boundaries() {
    assert!(ends_with_stop(&[1, 2], &[]));
    assert!(!ends_with_stop(&[2], &[1, 2]));
    assert!(ends_with_stop(&[1, 2], &[1, 2]));
    let stops = vec![vec![2], vec![1, 2], vec![9, 2], vec![1, 2]];
    assert_eq!(longest_stop_suffix(&[1, 2], &stops), Some((1, 2)));
    assert_eq!(longest_stop_suffix(&[3], &stops), None);
    assert_eq!(longest_stop_suffix(&[], &[]), None);
}

/// EOS outranks stops, stops wait for `min_new_tokens`, and length is the fallback.
#[test]
fn stop_decisions_follow_documented_precedence() {
    let stops = vec![vec![5]];
    assert_eq!(
        stop_decision(true, &[5], &stops, 9, 1),
        StopDecision::EndOfSequence
    );
    assert_eq!(
        stop_decision(false, &[5], &stops, 0, 1),
        StopDecision::StopSequence {
            index: 0,
            length: 1
        }
    );
    assert_eq!(
        stop_decision(false, &[5], &stops, 2, 1),
        StopDecision::Length
    );
    assert_eq!(
        stop_decision(false, &[5], &stops, 2, 3),
        StopDecision::Continue
    );
    assert_eq!(
        stop_decision(false, &[4], &stops, 0, 3),
        StopDecision::Continue
    );
}

/// Speculative commits follow the documented table and reject inconsistent outcomes.
#[test]
fn speculative_commit_table() {
    let commit = |target| SpeculativeCommit {
        emitted_tokens: target + 1,
        target_cache_tokens: target,
        draft_cache_tokens: target,
    };
    assert_eq!(
        speculative_commit(3, SpeculativeOutcome::Rejected { accepted: 0 }),
        Some(commit(1))
    );
    assert_eq!(
        speculative_commit(3, SpeculativeOutcome::Rejected { accepted: 2 }),
        Some(commit(3))
    );
    assert_eq!(
        speculative_commit(3, SpeculativeOutcome::Rejected { accepted: 3 }),
        None
    );
    assert_eq!(
        speculative_commit(3, SpeculativeOutcome::AcceptedEndOfSequence { accepted: 1 }),
        Some(commit(1))
    );
    assert_eq!(
        speculative_commit(3, SpeculativeOutcome::AcceptedEndOfSequence { accepted: 0 }),
        None
    );
    assert_eq!(
        speculative_commit(3, SpeculativeOutcome::AcceptedEndOfSequence { accepted: 4 }),
        None
    );
    assert_eq!(
        speculative_commit(3, SpeculativeOutcome::AllAccepted),
        Some(commit(4))
    );
    assert_eq!(speculative_commit(0, SpeculativeOutcome::AllAccepted), None);
    assert_eq!(
        speculative_commit(usize::MAX, SpeculativeOutcome::AllAccepted),
        None
    );
    assert_eq!(
        speculative_commit(
            usize::MAX,
            SpeculativeOutcome::AcceptedEndOfSequence {
                accepted: usize::MAX
            }
        ),
        None
    );
}

/// Explicit positions accept exactly the range that fits `i32` after feedback.
#[test]
fn next_position_boundaries() {
    let limit = i64::from(i32::MAX);
    assert_eq!(next_position_after(-1, 1), Err(PositionError::Negative));
    assert_eq!(next_position_after(limit, 1), Ok(limit + 1));
    assert_eq!(next_position_after(limit, 0), Ok(limit + 1));
    assert_eq!(
        next_position_after(limit, 2),
        Err(PositionError::ExceedsSignedIndex)
    );
    assert_eq!(next_position_after(limit - 1, 2), Ok(limit));
    assert_eq!(
        next_position_after(i64::MAX, 1),
        Err(PositionError::ExceedsSignedIndex)
    );
    assert_eq!(
        next_position_after(0, usize::MAX),
        Err(PositionError::ExceedsSignedIndex)
    );
}

/// The cache holds the prompt plus every fed-back token, never the final one.
#[test]
fn cache_capacity_excludes_the_final_token() {
    assert_eq!(required_cache_capacity(5, 1), Some(5));
    assert_eq!(required_cache_capacity(5, 3), Some(7));
    assert_eq!(required_cache_capacity(5, 0), Some(5));
    assert_eq!(required_cache_capacity(usize::MAX, 1), Some(usize::MAX));
    assert_eq!(required_cache_capacity(usize::MAX, 2), None);
}

/// Active rows advance by one, inactive rows stay, and overflow names the first row.
#[test]
fn active_positions_advance_and_report_the_first_overflow() {
    let mut positions = [3, 7, 9];
    assert_eq!(
        advance_active_positions(&mut positions, &[true, false, true]),
        Ok(())
    );
    assert_eq!(positions, [4, 7, 10]);
    let mut positions = [1, i64::MAX, i64::MAX, 5];
    assert_eq!(
        advance_active_positions(&mut positions, &[true, false, true, true]),
        Err(2)
    );
    // Rows before the failure advanced; rows from it on are unchanged.
    assert_eq!(positions, [2, i64::MAX, i64::MAX, 5]);
    let mut empty: [i64; 0] = [];
    assert_eq!(advance_active_positions(&mut empty, &[]), Ok(()));
}

/// Matrix rows are exact spans, and each failure reports its first cause.
#[test]
fn matrix_row_spans_report_overflow_and_bounds() {
    assert_eq!(matrix_row_span(0, 4, 8), Ok((0, 4)));
    assert_eq!(matrix_row_span(1, 4, 8), Ok((4, 8)));
    assert_eq!(matrix_row_span(2, 4, 8), Err(RowSpanError::OutOfBounds));
    assert_eq!(matrix_row_span(3, 0, 0), Ok((0, 0)));
    assert_eq!(
        matrix_row_span(usize::MAX, 2, usize::MAX),
        Err(RowSpanError::OffsetOverflow)
    );
    assert_eq!(
        matrix_row_span(1, usize::MAX / 2 + 1, usize::MAX),
        Err(RowSpanError::EndOverflow)
    );
}

/// The frontier partition keeps in-width finished ranks and the best live ranks.
#[test]
fn frontier_partition_examples() {
    let finished = [true, false, true, false, false, true, false];
    assert_eq!(
        partition_frontier(&finished, 2),
        FrontierPlan {
            finalized: vec![0],
            selected: vec![1, 3],
        }
    );
    assert_eq!(
        partition_frontier(&finished, 3),
        FrontierPlan {
            finalized: vec![0, 2],
            selected: vec![1, 3, 4],
        }
    );
    assert_eq!(
        partition_frontier(&[true, true], 4),
        FrontierPlan {
            finalized: vec![0, 1],
            selected: vec![],
        }
    );
    assert_eq!(
        partition_frontier(&[], 4),
        FrontierPlan {
            finalized: vec![],
            selected: vec![],
        }
    );
    assert_eq!(
        partition_frontier(&[false, true], 0),
        FrontierPlan {
            finalized: vec![],
            selected: vec![],
        }
    );
}

/// Prompt lookup across sizes prefers the largest matching n-gram.
#[test]
fn prompt_lookup_draft_prefers_the_largest_size() {
    let history = [7, 1, 2, 3, 4, 1, 2, 5, 1, 2];
    // Size 3 ("5 1 2") has no earlier match; size 2 ("1 2") does.
    assert_eq!(prompt_lookup_draft_span(&history, 1, 3, 3), Some((2, 3, 6)));
    assert_eq!(prompt_lookup_draft_span(&history, 3, 3, 3), None);
    // A zero minimum behaves like one; an empty range proposes nothing.
    assert_eq!(prompt_lookup_draft_span(&[4, 4], 0, 1, 2), Some((1, 1, 2)));
    assert_eq!(prompt_lookup_draft_span(&history, 3, 2, 3), None);
    assert_eq!(prompt_lookup_draft_span(&history, 0, 0, 3), None);
    assert_eq!(prompt_lookup_draft_span(&history, 1, 3, 0), None);
}

/// Prompt lookup proposes the continuation of the earliest earlier occurrence.
#[test]
fn prompt_lookup_examples() {
    let history = [7, 1, 2, 3, 4, 1, 2, 5, 1, 2];
    // "1 2" first occurs at 1, followed by 3 4 1 ...
    assert_eq!(prompt_lookup_span(&history, 2, 3), Some((3, 6)));
    assert_eq!(prompt_lookup_span(&history, 3, 3), None);
    assert_eq!(prompt_lookup_span(&history, 0, 3), None);
    assert_eq!(prompt_lookup_span(&history, 2, 0), None);
    // A match whose continuation reaches the end is truncated there.
    assert_eq!(prompt_lookup_span(&[1, 1], 1, 5), Some((1, 2)));
    assert_eq!(prompt_lookup_span(&[1], 1, 5), None);
}

proptest! {
    /// Longest-suffix matching agrees with a naive search.
    #[test]
    fn longest_stop_matches_reference(
        generated in proptest::collection::vec(0_u32..3, 0..8),
        stops in proptest::collection::vec(proptest::collection::vec(0_u32..3, 1..4), 0..5),
    ) {
        prop_assert_eq!(longest_stop_suffix(&generated, &stops), reference_longest(&generated, &stops));
        for stop in &stops {
            prop_assert_eq!(ends_with_stop(&generated, stop), generated.ends_with(stop));
        }
    }

    /// Prefill chunks partition the prompt into nonempty, bounded, ordered pieces.
    #[test]
    fn prefill_chunks_partition_the_sequence(sequence in 1_usize..64, chunk in 1_usize..80) {
        let mut start = 0;
        let mut covered = 0;
        while start < sequence {
            let end = prefill_chunk_end(start, chunk, sequence);
            prop_assert!(start < end && end <= sequence && end - start <= chunk);
            covered += end - start;
            start = end;
        }
        prop_assert_eq!(covered, sequence);
    }

    /// Row spans address exactly row `r`'s slice of a flattened rectangle.
    #[test]
    fn row_spans_stay_inside_the_rectangle(
        batch in 1_usize..8,
        sequence in 1_usize..8,
        row_seed in 0_usize..64,
        a in 0_usize..8,
        b in 0_usize..8,
    ) {
        let row = row_seed % batch;
        let (start, end) = (a.min(b) % (sequence + 1), a.max(b) % (sequence + 1));
        let (start, end) = (start.min(end), start.max(end));
        let (first, last) = row_span(row, batch, sequence, start, end);
        prop_assert_eq!(first, row * sequence + start);
        prop_assert_eq!(last, row * sequence + end);
        prop_assert!(last <= batch * sequence);
    }

    /// Beam rows belong to their source group, and expansion maps row `i` to `i / beams`.
    #[test]
    fn beam_rows_preserve_source_groups(source in 0_usize..1000, beams in 1_usize..16, slot_seed in 0_usize..16) {
        let slot = slot_seed % beams;
        let row = beam_row(source, beams, slot).expect("small values fit");
        prop_assert_eq!(row / beams, source);
        prop_assert_eq!(row % beams, slot);
        let mut indices = Vec::new();
        fill_beam_expansion(&mut indices, 3 * beams, beams);
        prop_assert_eq!(indices.len(), 3 * beams);
        for (row, &source) in indices.iter().enumerate() {
            prop_assert_eq!(source, row / beams);
        }
    }

    /// The candidate frontier is `beams * max(2, eos + 1)` or reports overflow.
    #[test]
    fn frontier_capacity_matches_formula(beams in 0_usize..1 << 20, eos in 0_usize..64) {
        prop_assert_eq!(frontier_capacity(beams, eos), Some(beams * (eos + 1).max(2)));
    }

    /// The frontier partition agrees with the original single-pass selection loop.
    #[test]
    fn frontier_partition_matches_reference(
        finished in proptest::collection::vec(any::<bool>(), 0..24),
        beams in 0_usize..8,
    ) {
        let mut finalized = Vec::new();
        let mut selected = Vec::new();
        for (rank, &is_finished) in finished.iter().enumerate() {
            if is_finished {
                if rank < beams {
                    finalized.push(rank);
                }
            } else if selected.len() < beams {
                selected.push(rank);
            }
        }
        prop_assert_eq!(partition_frontier(&finished, beams), FrontierPlan { finalized, selected });
    }

    /// Multi-size prompt lookup agrees with trying each size from the largest down.
    #[test]
    fn prompt_lookup_draft_matches_reference(
        history in proptest::collection::vec(0_u32..3, 0..12),
        min_ngram in 0_usize..4,
        max_ngram in 0_usize..5,
        budget in 0_usize..4,
    ) {
        let expected = (min_ngram.max(1)..=max_ngram).rev().find_map(|ngram| {
            prompt_lookup_span(&history, ngram, budget).map(|(start, end)| (ngram, start, end))
        });
        prop_assert_eq!(prompt_lookup_draft_span(&history, min_ngram, max_ngram, budget), expected);
    }

    /// Prompt lookup agrees with a naive earliest-match scan.
    #[test]
    fn prompt_lookup_matches_reference(
        history in proptest::collection::vec(0_u32..3, 0..12),
        ngram in 1_usize..4,
        budget in 1_usize..5,
    ) {
        let length = history.len();
        let expected = (length > ngram)
            .then(|| {
                let suffix = &history[length - ngram..];
                (0..length - ngram).find(|&candidate| &history[candidate..candidate + ngram] == suffix)
            })
            .flatten()
            .map(|candidate| {
                let start = candidate + ngram;
                (start, (start + budget).min(length))
            });
        prop_assert_eq!(prompt_lookup_span(&history, ngram, budget), expected);
    }
}

/// Overflowing beam rows and frontiers are reported instead of wrapping.
#[test]
fn beam_arithmetic_reports_overflow() {
    assert_eq!(beam_row(usize::MAX, 2, 1), None);
    assert_eq!(beam_row(usize::MAX / 2, 2, 1), Some(usize::MAX));
    assert_eq!(frontier_capacity(usize::MAX, 0), None);
    assert_eq!(frontier_capacity(2, usize::MAX), None);
}
