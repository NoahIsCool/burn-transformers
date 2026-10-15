//! Executable checks of the Verus-proved bookkeeping contracts.
//!
//! The proofs run outside Cargo (see `verus/README.md`); these property tests
//! exercise the same contracts against naive references in every ordinary
//! test run, so a regression is caught even where Verus is not installed.

use proptest::prelude::*;

use super::{
    PositionError, SpeculativeCommit, SpeculativeOutcome, StopDecision, beam_row, ends_with_stop,
    fill_beam_expansion, frontier_capacity, longest_stop_suffix, next_position_after,
    prefill_chunk_end, prompt_lookup_span, row_span, speculative_commit, stop_decision,
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
