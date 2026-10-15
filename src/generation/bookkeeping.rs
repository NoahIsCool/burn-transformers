//! Verified integer bookkeeping shared by the generation coordinators.
//!
//! Generation mixes backend tensor math with small host-side decisions: which
//! stop sequence a row just completed, which prompt columns a prefill chunk
//! covers, which expanded batch row holds beam `k` of source `s`, and how many
//! speculative tokens a cache may publish. Those decisions are pure integer
//! logic, so they live here, free of Burn, floating point, and crate errors.
//! Callers wrap them with contextual [`crate::Error`] values.
//!
//! # Verified properties
//!
//! Every function below is checked in place by Verus through
//! `verus/generation.rs` (see `verus/README.md`). For all admissible inputs
//! the proofs establish that machine arithmetic cannot overflow or underflow
//! and that:
//!
//! - stop matching returns exactly the longest configured suffix, breaking
//!   equal lengths by the smallest index, and the finish decision applies the
//!   precedence EOS, then stop sequence (only after `min_new_tokens`), then
//!   length;
//! - prefill chunks partition `0..sequence` into nonempty pieces of at most
//!   `chunk_size` columns, and each row's slice of a flattened rectangular mask
//!   stays inside `0..batch * sequence`;
//! - beam row `source * beams + slot` always belongs to group `source`, so a
//!   reorder built from these rows never moves a hypothesis to another source;
//!   the initial expansion maps row `i` to source `i / beams`;
//! - a speculative cycle publishes exactly one more emitted token than target
//!   cache entries and keeps the draft cache aligned with the target cache;
//! - prompt-lookup drafting returns the continuation of the earliest earlier
//!   occurrence of the trailing n-gram, inside the history and within the
//!   draft budget.
//!
//! The proofs do not cover tensor kernels or the coordinators that call these
//! functions; the Rust test suite covers those.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// `stop` occurs at the end of `generated`.
pub open spec fn is_suffix(generated: Seq<u32>, stop: Seq<u32>) -> bool {
    &&& stop.len() <= generated.len()
    &&& forall|k: int| 0 <= k < stop.len()
        ==> #[trigger] stop[k] == generated[generated.len() - stop.len() + k]
}

/// Some configured stop sequence is a suffix of `generated`.
pub open spec fn any_stop_matches(generated: Seq<u32>, stops: Seq<Vec<u32>>) -> bool {
    exists|i: int| 0 <= i < stops.len() && is_suffix(generated, #[trigger] stops[i]@)
}

/// `(index, length)` names the longest matching stop, with the smallest index among ties.
pub open spec fn is_longest_stop(
    generated: Seq<u32>,
    stops: Seq<Vec<u32>>,
    index: int,
    length: int,
) -> bool {
    &&& 0 <= index < stops.len()
    &&& length == stops[index]@.len()
    &&& is_suffix(generated, stops[index]@)
    &&& forall|i: int| 0 <= i < stops.len() && is_suffix(generated, #[trigger] stops[i]@)
        ==> stops[i]@.len() <= length
    &&& forall|i: int| 0 <= i < index && is_suffix(generated, #[trigger] stops[i]@)
        ==> stops[i]@.len() < length
}

/// The `n` tokens starting at `a` equal the `n` tokens starting at `b`.
pub open spec fn windows_match(history: Seq<u32>, a: int, b: int, n: int) -> bool {
    forall|k: int| 0 <= k < n ==> #[trigger] history[a + k] == history[b + k]
}

} // verus!

// ---------------------------------------------------------------------------
// Stop sequences and finish decisions
// ---------------------------------------------------------------------------

/// Returns whether `stop` is a suffix of `generated`.
///
/// An empty `stop` is trivially a suffix; configuration validation rejects
/// empty stop sequences before any tracker can use one.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(matched =>
    ensures
        matched == is_suffix(generated@, stop@),
))]
pub(crate) fn ends_with_stop(generated: &[u32], stop: &[u32]) -> bool {
    if stop.len() > generated.len() {
        return false;
    }
    // Compare `stop[k]` with the aligned generated token `generated[offset + k]`.
    let offset = generated.len() - stop.len();
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            offset == generated.len() - stop.len(),
            index <= stop.len(),
            forall|k: int| 0 <= k < index ==> #[trigger] stop@[k] == generated@[offset + k],
        decreases stop.len() - index,
    ))]
    while index < stop.len() {
        if generated[offset + index] != stop[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// Finds the longest stop sequence that `generated` ends with.
///
/// Returns `(index, length)` into `stops`. Equal lengths resolve to the
/// smallest index, so configuration order is a deterministic tie breaker.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
        match found {
            None => !any_stop_matches(generated@, stops@),
            Some((index, length)) => is_longest_stop(generated@, stops@, index as int, length as int),
        }
))]
pub(crate) fn longest_stop_suffix(generated: &[u32], stops: &[Vec<u32>]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= stops.len(),
            match best {
                None => forall|i: int| 0 <= i < index ==> !is_suffix(generated@, #[trigger] stops@[i]@),
                Some((chosen, length)) => {
                    &&& chosen < index
                    &&& length == stops@[chosen as int]@.len()
                    &&& is_suffix(generated@, stops@[chosen as int]@)
                    &&& forall|i: int| 0 <= i < index && is_suffix(generated@, #[trigger] stops@[i]@)
                        ==> stops@[i]@.len() <= length
                    &&& forall|i: int| 0 <= i < chosen && is_suffix(generated@, #[trigger] stops@[i]@)
                        ==> stops@[i]@.len() < length
                },
            },
        decreases stops.len() - index,
    ))]
    while index < stops.len() {
        let stop = stops[index].as_slice();
        if ends_with_stop(generated, stop) {
            // Strictly longer matches replace the current choice; equal
            // lengths keep the earlier index.
            let improves = match best {
                None => true,
                Some((_, best_length)) => stop.len() > best_length,
            };
            if improves {
                best = Some((index, stop.len()));
            }
        }
        index += 1;
    }
    best
}

/// Host-side outcome of appending one token to a generated row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StopDecision {
    /// The row remains active.
    Continue,
    /// The appended token is a configured end-of-sequence ID.
    EndOfSequence,
    /// A configured stop sequence now ends the row.
    StopSequence {
        /// Index of the matching sequence in the configuration.
        index: usize,
        /// Number of tokens in the matching sequence.
        length: usize,
    },
    /// The row reached its new-token budget.
    Length,
}

/// Decides whether a row finishes after `generated` received its newest token.
///
/// Precedence is EOS, then the longest stop sequence, then the length budget.
/// Stop sequences only count once at least `min_new_tokens` tokens exist,
/// matching vLLM's `min_tokens`; EOS suppression before that point is the
/// logits processor's job, so an EOS that still arrives here always finishes.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(decision =>
    ensures
        is_eos ==> decision == StopDecision::EndOfSequence,
        !is_eos ==> match decision {
            StopDecision::EndOfSequence => false,
            StopDecision::StopSequence { index, length } => {
                &&& generated.len() >= min_new_tokens
                &&& is_longest_stop(generated@, stops@, index as int, length as int)
            },
            StopDecision::Length => {
                &&& generated.len() >= max_new_tokens
                &&& (generated.len() < min_new_tokens || !any_stop_matches(generated@, stops@))
            },
            StopDecision::Continue => {
                &&& generated.len() < max_new_tokens
                &&& (generated.len() < min_new_tokens || !any_stop_matches(generated@, stops@))
            },
        },
))]
pub(crate) fn stop_decision(
    is_eos: bool,
    generated: &[u32],
    stops: &[Vec<u32>],
    min_new_tokens: usize,
    max_new_tokens: usize,
) -> StopDecision {
    if is_eos {
        return StopDecision::EndOfSequence;
    }
    // Stop sequences only count once the row has its minimum length.
    let stop = if generated.len() >= min_new_tokens {
        longest_stop_suffix(generated, stops)
    } else {
        None
    };
    if let Some((index, length)) = stop {
        return StopDecision::StopSequence { index, length };
    }
    if generated.len() >= max_new_tokens {
        StopDecision::Length
    } else {
        StopDecision::Continue
    }
}

// ---------------------------------------------------------------------------
// Prompt chunking and rectangular metadata
// ---------------------------------------------------------------------------

/// Returns the exclusive end of the prefill chunk that starts at `start`.
///
/// Repeatedly setting `start = end` therefore visits `0..sequence` in order,
/// in nonempty chunks of at most `chunk_size` columns, and terminates.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(end =>
    requires
        start < sequence,
        chunk_size > 0,
    ensures
        start < end,
        end <= sequence,
        end - start <= chunk_size,
        end < sequence ==> end - start == chunk_size,
))]
pub(crate) fn prefill_chunk_end(start: usize, chunk_size: usize, sequence: usize) -> usize {
    // `sequence - start` cannot underflow, and comparing against it avoids
    // computing `start + chunk_size`, which could overflow.
    if chunk_size >= sequence - start {
        sequence
    } else {
        start + chunk_size
    }
}

/// Flattened offsets of columns `start..end` of `row` in a `[batch, sequence]` buffer.
///
/// The precondition `batch * sequence <= usize::MAX` holds for every buffer
/// that exists, so the returned span never overflows and stays in bounds.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(span =>
    requires
        row < batch,
        start <= end,
        end <= sequence,
        batch * sequence <= usize::MAX,
    ensures
        span.0 == row * sequence + start,
        span.1 == row * sequence + end,
        span.1 <= batch * sequence,
))]
pub(crate) fn row_span(
    row: usize,
    batch: usize,
    sequence: usize,
    start: usize,
    end: usize,
) -> (usize, usize) {
    #[cfg(verus_keep_ghost)]
    proof! {
        // row < batch, so (row + 1) * sequence <= batch * sequence.
        assert(row * sequence + sequence <= batch * sequence) by (nonlinear_arith)
            requires
                row < batch,
        ;
    }
    // `batch` only bounds the result; it appears in the specification above.
    let _ = batch;
    let offset = row * sequence;
    (offset + start, offset + end)
}

// ---------------------------------------------------------------------------
// Beam rows
// ---------------------------------------------------------------------------

/// Expanded-batch row holding beam `slot` of source row `source`.
///
/// Beam search lays out `beams` consecutive rows per source. The proof shows
/// the row's group `row / beams` is `source`, so any reorder whose parents
/// and destinations come from this function stays within each source group.
/// `None` reports `source * beams + slot` overflowing `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(row =>
    requires
        slot < beams,
    ensures
        match row {
            Some(row) => {
                &&& row == source * beams + slot
                &&& row / beams == source
                &&& row % beams == slot
            },
            None => source * beams + slot > usize::MAX,
        },
))]
pub(crate) fn beam_row(source: usize, beams: usize, slot: usize) -> Option<usize> {
    let row = source.checked_mul(beams)?.checked_add(slot)?;
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
            row as int,
            beams as int,
            source as int,
            slot as int,
        );
    }
    Some(row)
}

/// Appends the initial source index of every expanded beam row.
///
/// Row `i` of a `[batch * beams]` beam batch starts as a copy of source
/// `i / beams`; `count` must equal `batch * beams`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(
    requires
        beams > 0,
        old(indices).len() == 0,
    ensures
        final(indices).len() == count,
        forall|i: int| 0 <= i < count ==> #[trigger] final(indices)@[i] == i / (beams as int),
))]
pub(crate) fn fill_beam_expansion(indices: &mut Vec<usize>, count: usize, beams: usize) {
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            beams > 0,
            row <= count,
            indices.len() == row,
            forall|i: int| 0 <= i < row ==> #[trigger] indices@[i] == i / (beams as int),
        decreases count - row,
    ))]
    while row < count {
        indices.push(row / beams);
        row += 1;
    }
}

/// Number of best candidates examined per source in one beam step.
///
/// One beam width supplies live successors and each distinct EOS ID may
/// consume another width of finished candidates, with a floor of two widths
/// (the conventional single-EOS frontier). `None` reports overflow.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(frontier =>
    ensures
        match frontier {
            Some(frontier) => {
                let multiplier = if eos_count + 1 >= 2 { eos_count + 1 } else { 2 };
                frontier == beams * multiplier
            },
            None => {
                let multiplier = if eos_count + 1 >= 2 { eos_count + 1 } else { 2 };
                eos_count + 1 > usize::MAX || beams * multiplier > usize::MAX
            },
        },
))]
pub(crate) fn frontier_capacity(beams: usize, eos_count: usize) -> Option<usize> {
    let widths = eos_count.checked_add(1)?;
    let multiplier = if widths >= 2 { widths } else { 2 };
    beams.checked_mul(multiplier)
}

// ---------------------------------------------------------------------------
// Explicit positions
// ---------------------------------------------------------------------------

/// Why an explicit prompt position cannot be continued.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PositionError {
    /// The last valid prompt position is negative.
    Negative,
    /// The last position fed to the model would exceed `i32::MAX`.
    ExceedsSignedIndex,
}

/// Next explicit position after a prompt whose last valid position is `last`.
///
/// A coordinator feeds at most `max_new_tokens - 1` generated tokens back to
/// the model (the final token is never decoded), at positions
/// `last + 1 ..= last + max_new_tokens - 1`. Accepting the request therefore
/// requires that range to fit Burn's portable signed 32-bit index domain.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(next =>
    ensures
        ({
            let fed_back = if max_new_tokens == 0 { 0 } else { max_new_tokens - 1 };
            match next {
                Ok(next) => {
                    &&& last >= 0
                    &&& next == last + 1
                    &&& last + fed_back <= i32::MAX
                },
                Err(PositionError::Negative) => last < 0,
                Err(PositionError::ExceedsSignedIndex) => last >= 0 && last + fed_back > i32::MAX,
            }
        }),
))]
pub(crate) fn next_position_after(last: i64, max_new_tokens: usize) -> Result<i64, PositionError> {
    if last < 0 {
        return Err(PositionError::Negative);
    }
    let limit = i32::MAX as i64;
    if last > limit {
        return Err(PositionError::ExceedsSignedIndex);
    }
    let fed_back = if max_new_tokens == 0 {
        0
    } else {
        max_new_tokens - 1
    };
    // `0 <= last <= limit`, so the headroom is a small non-negative value and
    // both sides of the comparison convert to `u64` without wrapping.
    let headroom = (limit - last) as u64;
    if fed_back as u64 > headroom {
        return Err(PositionError::ExceedsSignedIndex);
    }
    Ok(last + 1)
}

// ---------------------------------------------------------------------------
// Speculative decoding
// ---------------------------------------------------------------------------

/// Returns whether `history[a..a + n]` equals `history[b..b + n]`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(matched =>
    requires
        a + n <= history.len(),
        b + n <= history.len(),
    ensures
        matched == windows_match(history@, a as int, b as int, n as int),
))]
fn windows_equal(history: &[u32], a: usize, b: usize, n: usize) -> bool {
    let mut offset = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            a + n <= history.len(),
            b + n <= history.len(),
            offset <= n,
            forall|k: int| 0 <= k < offset ==> #[trigger] history@[a + k] == history@[b + k],
        decreases n - offset,
    ))]
    while offset < n {
        if history[a + offset] != history[b + offset] {
            return false;
        }
        offset += 1;
    }
    true
}

/// Draft span proposed by prompt lookup for one n-gram size.
///
/// Prompt-lookup ("n-gram") drafting predicts that text repeats: when the last
/// `ngram` tokens of `history` occurred earlier, the tokens that followed that
/// earlier occurrence are proposed as drafts. Like Hugging Face's
/// `PromptLookupCandidateGenerator`, the earliest occurrence with a nonempty
/// continuation wins. Returns `[start, end)` into `history`, at most
/// `max_tokens` long, or `None` without a usable match.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
        match found {
            Some((start, end)) => {
                &&& ngram > 0
                &&& ngram <= start
                &&& start < end
                &&& end <= history.len()
                &&& end - start <= max_tokens
                &&& (end == history.len() || end - start == max_tokens)
                &&& windows_match(
                    history@,
                    start - ngram,
                    history.len() - ngram,
                    ngram as int,
                )
                &&& forall|candidate: int| 0 <= candidate < start - ngram ==> !#[trigger] windows_match(
                    history@,
                    candidate,
                    history.len() - ngram,
                    ngram as int,
                )
            },
            None => ngram == 0 || max_tokens == 0 || history.len() <= ngram || forall|candidate: int|
                0 <= candidate < history.len() - ngram ==> !#[trigger] windows_match(
                    history@,
                    candidate,
                    history.len() - ngram,
                    ngram as int,
                ),
        },
))]
pub(crate) fn prompt_lookup_span(
    history: &[u32],
    ngram: usize,
    max_tokens: usize,
) -> Option<(usize, usize)> {
    let length = history.len();
    if ngram == 0 || max_tokens == 0 || length <= ngram {
        return None;
    }
    // The trailing n-gram starts at `suffix`; a match at `candidate < suffix`
    // has at least one following token, at `candidate + ngram < length`.
    let suffix = length - ngram;
    let mut candidate = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            0 < ngram < length,
            max_tokens > 0,
            suffix == length - ngram,
            length == history.len(),
            candidate <= suffix,
            forall|earlier: int| 0 <= earlier < candidate ==> !#[trigger] windows_match(
                history@,
                earlier,
                suffix as int,
                ngram as int,
            ),
        decreases suffix - candidate,
    ))]
    while candidate < suffix {
        if windows_equal(history, candidate, suffix, ngram) {
            let start = candidate + ngram;
            let end = if max_tokens >= length - start {
                length
            } else {
                start + max_tokens
            };
            return Some((start, end));
        }
        candidate += 1;
    }
    None
}

/// How target verification ended one speculative cycle.
///
/// A cycle starts from a target-selected lead token and `drafted` proposals.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeOutcome {
    /// Proposal `accepted` (zero-based) was rejected and replaced by a target correction.
    Rejected {
        /// Proposals accepted before the rejection.
        accepted: usize,
    },
    /// The `accepted`-th accepted proposal was an end-of-sequence token.
    AcceptedEndOfSequence {
        /// Proposals accepted, including the terminal EOS proposal.
        accepted: usize,
    },
    /// Every proposal was accepted and the target emitted one bonus token.
    AllAccepted,
}

/// Token counts a speculative cycle publishes relative to its transaction mark.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeCommit {
    /// Tokens returned to the caller, starting with the lead token.
    pub emitted_tokens: usize,
    /// Target-cache entries to keep: the lead plus every accepted non-EOS proposal.
    pub target_cache_tokens: usize,
    /// Draft-cache entries to keep, equal to the target count so a shifted
    /// MTP cache stays exactly one position behind the target.
    pub draft_cache_tokens: usize,
}

/// Computes the accepted prefix a speculative cycle may publish.
///
/// With `a` accepted proposals out of `d`:
///
/// | outcome | emitted | target cache | draft cache |
/// | --- | --- | --- | --- |
/// | rejected at `a < d` | `a + 2` (lead, `a` drafts, correction) | `a + 1` | `a + 1` |
/// | EOS accepted, `1 <= a <= d` | `a + 1` | `a` | `a` |
/// | all accepted, `d >= 1` | `d + 2` (lead, `d` drafts, bonus) | `d + 1` | `d + 1` |
///
/// The last emitted token is never in the target cache: the next cycle (or
/// nothing, after EOS) consumes it. `None` reports an outcome inconsistent
/// with `drafted`, or arithmetic overflow.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(commit =>
    ensures
        match commit {
            Some(commit) => {
                &&& commit.draft_cache_tokens == commit.target_cache_tokens
                &&& commit.emitted_tokens == commit.target_cache_tokens + 1
                &&& 1 <= commit.target_cache_tokens <= drafted + 1
                &&& match outcome {
                    SpeculativeOutcome::Rejected { accepted } => {
                        &&& accepted < drafted
                        &&& commit.target_cache_tokens == accepted + 1
                    },
                    SpeculativeOutcome::AcceptedEndOfSequence { accepted } => {
                        &&& 1 <= accepted <= drafted
                        &&& commit.target_cache_tokens == accepted
                    },
                    SpeculativeOutcome::AllAccepted => {
                        &&& drafted >= 1
                        &&& commit.target_cache_tokens == drafted + 1
                    },
                }
            },
            None => match outcome {
                SpeculativeOutcome::Rejected { accepted } =>
                    accepted >= drafted || accepted + 2 > usize::MAX,
                SpeculativeOutcome::AcceptedEndOfSequence { accepted } =>
                    accepted == 0 || accepted > drafted || accepted + 1 > usize::MAX,
                SpeculativeOutcome::AllAccepted => drafted == 0 || drafted + 2 > usize::MAX,
            },
        },
))]
pub fn speculative_commit(
    drafted: usize,
    outcome: SpeculativeOutcome,
) -> Option<SpeculativeCommit> {
    let target_cache_tokens = match outcome {
        SpeculativeOutcome::Rejected { accepted } => {
            if accepted >= drafted {
                return None;
            }
            // accepted < drafted <= usize::MAX, so this cannot overflow.
            accepted + 1
        }
        SpeculativeOutcome::AcceptedEndOfSequence { accepted } => {
            if accepted == 0 || accepted > drafted {
                return None;
            }
            accepted
        }
        SpeculativeOutcome::AllAccepted => {
            if drafted == 0 {
                return None;
            }
            drafted.checked_add(1)?
        }
    };
    let emitted_tokens = target_cache_tokens.checked_add(1)?;
    Some(SpeculativeCommit {
        emitted_tokens,
        target_cache_tokens,
        draft_cache_tokens: target_cache_tokens,
    })
}

#[cfg(test)]
#[path = "bookkeeping_tests.rs"]
mod tests;
