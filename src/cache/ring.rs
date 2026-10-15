//! Pure plans for standard, mirrored, and speculative sliding-window storage.
//!
//! Absolute token position `p` maps to physical slot `p % capacity`. A retained
//! suffix no wider than capacity crosses that boundary at most once, so reads
//! and writes need at most two contiguous segments. Plans use checked integer
//! arithmetic and do not allocate backend tensors or depend on floating dtype.
//!
//! Mirrored storage doubles physical slots to make chronological reads contiguous.
//! Speculative proposals instead remain in separate linear storage until acceptance:
//! rolling back a logical length cannot recover baseline values overwritten in a
//! full ring. This trades temporary proposal memory for reversible execution.
//!
//! # Verified planning
//!
//! The write and read planners are checked in place by Verus through
//! `verus/cache.rs` (see `verus/README.md`). For every admissible input, the
//! proofs establish that all machine arithmetic is free of overflow, that the
//! returned segments partition the retained suffix in order, that every segment
//! fits inside the ring, and that each token lands in, or is read from, physical
//! slot `absolute_position mod capacity`. Writes of a full ring touch every
//! slot exactly once. Tensor copies driven by these plans remain covered by
//! the cache test suite.

use core::fmt;
use core::num::NonZeroUsize;
use core::ops::Range;

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Physical slot that holds absolute position `absolute_start + k`.
///
/// This is the ring invariant `slot(p) = p mod capacity` written relative to
/// the first position of a contiguous run.
pub open spec fn ring_slot(absolute_start: int, k: int, capacity: int) -> int {
    (absolute_start + k) % capacity
}

/// Advancing by `k` from `a` stays before the ring end or wraps exactly once.
///
/// With `r = a mod c`, the slot of `a + k` is `r + k` while `r + k < c`, and
/// `r + k - c` once the run crosses the end, provided it crosses at most once.
proof fn lemma_slot_after_offset(a: int, k: int, c: int)
    requires
        c > 0,
        a >= 0,
        k >= 0,
        a % c + k < 2 * c,
    ensures
        a % c + k < c ==> (a + k) % c == a % c + k,
        a % c + k >= c ==> (a + k) % c == a % c + k - c,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(a, c);
    let q = a / c;
    let r = a % c;
    assert(a + k == c * q + (r + k));
    if r + k < c {
        vstd::arithmetic::div_mod::lemma_mod_multiples_vanish(q, r + k, c);
        vstd::arithmetic::div_mod::lemma_small_mod((r + k) as nat, c as nat);
    } else {
        assert(a + k == c * (q + 1) + (r + k - c)) by (nonlinear_arith)
            requires
                a + k == c * q + (r + k),
        ;
        vstd::arithmetic::div_mod::lemma_mod_multiples_vanish(q + 1, r + k - c, c);
        vstd::arithmetic::div_mod::lemma_small_mod((r + k - c) as nat, c as nat);
    }
}

} // verus!

/// One contiguous piece of an incoming chunk written into a standard ring.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingWriteSegment {
    /// Contiguous interval within the incoming token chunk.
    input: Range<usize>,
    /// Equally sized destination interval inside the physical ring.
    physical: Range<usize>,
}

impl RingWriteSegment {
    /// Returns the range within the incoming token chunk.
    #[must_use]
    pub fn input(&self) -> Range<usize> {
        self.input.clone()
    }

    /// Returns the matching destination range in physical ring storage.
    #[must_use]
    pub fn physical(&self) -> Range<usize> {
        self.physical.clone()
    }
}

/// Complete write plan for appending a chunk to a standard ring buffer.
///
/// When a chunk is longer than the ring, only its newest `capacity` tokens are
/// present after the write. `retained_input` identifies that suffix. At most
/// two segments are needed because the retained suffix wraps at most once.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingWritePlan {
    /// Absolute positions covered by the entire incoming chunk.
    appended_positions: Range<usize>,
    /// Newest incoming suffix that survives ring eviction.
    retained_input: Range<usize>,
    /// Contiguous destination writes in incoming-token order.
    segments: Vec<RingWriteSegment>,
}

/// Safe commit planning for speculative writes targeting a wrapped ring.
///
/// Proposed K/V tensors must first be written to separate, append-only staging
/// storage indexed by [`staged_input`](Self::staged_input). After verification,
/// [`commit_prefix`](Self::commit_prefix) returns the only writes that should
/// touch the committed ring. Rolling back requires no ring write at all: the
/// staging allocation is simply discarded.
///
/// This staging rule is necessary once a ring is full. Writing proposal
/// position `p` directly to `p mod capacity` may destroy committed position
/// `p - capacity`; restoring only a logical length cannot reconstruct it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedRingTransactionPlan {
    /// Absolute proposal positions held outside committed storage.
    staged_positions: Range<usize>,
    /// Linear staging interval beginning at proposal offset zero.
    staged_input: Range<usize>,
}

impl StagedRingTransactionPlan {
    /// Returns the tentative absolute positions held outside the ring.
    #[must_use]
    pub fn staged_positions(&self) -> Range<usize> {
        self.staged_positions.clone()
    }

    /// Returns the required linear staging-buffer range.
    #[must_use]
    pub fn staged_input(&self) -> Range<usize> {
        self.staged_input.clone()
    }

    /// Plans copying an accepted staging prefix into the committed ring.
    ///
    /// The returned write segments address the staging buffer as their input.
    /// An accepted prefix longer than the ring is valid: only its newest
    /// `capacity` entries need to remain in the final physical state.
    ///
    /// # Errors
    ///
    /// Returns [`RingPlanError::CommitPrefixExceedsStaged`] if `accepted_tokens`
    /// is longer than the proposal, or propagates ring position errors.
    pub fn commit_prefix(
        &self,
        accepted_tokens: usize,
        capacity: NonZeroUsize,
    ) -> Result<RingWritePlan, RingPlanError> {
        let staged_tokens = self.staged_input.len();
        if accepted_tokens > staged_tokens {
            return Err(RingPlanError::CommitPrefixExceedsStaged {
                accepted: accepted_tokens,
                staged: staged_tokens,
            });
        }
        plan_ring_write(self.staged_positions.start, accepted_tokens, capacity)
    }
}

impl RingWritePlan {
    /// Returns the absolute positions represented by the full incoming chunk.
    #[must_use]
    pub fn appended_positions(&self) -> Range<usize> {
        self.appended_positions.clone()
    }

    /// Returns the suffix of the incoming chunk retained after the write.
    #[must_use]
    pub fn retained_input(&self) -> Range<usize> {
        self.retained_input.clone()
    }

    /// Returns the contiguous physical writes in input order.
    #[must_use]
    pub fn segments(&self) -> &[RingWriteSegment] {
        &self.segments
    }
}

/// One chronological piece read from a wrapped standard ring.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingReadSegment {
    /// Absolute token positions represented by this slice.
    absolute: Range<usize>,
    /// Equally sized interval inside physical ring storage.
    physical: Range<usize>,
}

impl RingReadSegment {
    /// Returns the absolute token positions represented by this segment.
    #[must_use]
    pub fn absolute(&self) -> Range<usize> {
        self.absolute.clone()
    }

    /// Returns the corresponding physical ring range.
    #[must_use]
    pub fn physical(&self) -> Range<usize> {
        self.physical.clone()
    }
}

/// Chronological read plan for the retained suffix of a standard ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingReadPlan {
    /// Absolute suffix still visible under the retention policy.
    retained_positions: Range<usize>,
    /// Physical slices ordered from oldest to newest retained token.
    segments: Vec<RingReadSegment>,
}

impl RingReadPlan {
    /// Returns the absolute range visible to local attention.
    #[must_use]
    pub fn retained_positions(&self) -> Range<usize> {
        self.retained_positions.clone()
    }

    /// Returns one or two physical slices in chronological order.
    #[must_use]
    pub fn segments(&self) -> &[RingReadSegment] {
        &self.segments
    }
}

/// One contiguous write into both halves of a mirrored ring.
///
/// Mirroring slot `s` at `s + capacity` makes every suffix of at most
/// `capacity` tokens readable as one contiguous physical slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirroredRingWriteSegment {
    /// Contiguous interval within the incoming token chunk.
    input: Range<usize>,
    /// Destination interval in the first ring half.
    primary: Range<usize>,
    /// Duplicate destination offset by one ring capacity.
    mirror: Range<usize>,
}

impl MirroredRingWriteSegment {
    /// Returns the range within the incoming token chunk.
    #[must_use]
    pub fn input(&self) -> Range<usize> {
        self.input.clone()
    }

    /// Returns the destination range in the first ring half.
    #[must_use]
    pub fn primary(&self) -> Range<usize> {
        self.primary.clone()
    }

    /// Returns the identical destination range in the mirrored half.
    #[must_use]
    pub fn mirror(&self) -> Range<usize> {
        self.mirror.clone()
    }
}

/// Single-slice read plan for a mirrored ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirroredRingReadPlan {
    /// Absolute suffix visible to sliding attention.
    retained_positions: Range<usize>,
    /// Contiguous interval spanning at most two mirrored halves.
    physical: Range<usize>,
}

impl MirroredRingReadPlan {
    /// Returns the absolute range visible to local attention.
    #[must_use]
    pub fn retained_positions(&self) -> Range<usize> {
        self.retained_positions.clone()
    }

    /// Returns the contiguous physical range in the doubled allocation.
    #[must_use]
    pub fn physical(&self) -> Range<usize> {
        self.physical.clone()
    }
}

/// Computes the newest absolute positions retained by a sliding window.
///
/// For an exclusive logical sequence end `L` and window `W`, the returned
/// interval is `[max(0, L - W), L)`. This is the exact key interval needed by
/// one-token local-attention decode at position `L - 1`.
#[must_use]
pub fn sliding_window_range(sequence_length: usize, window: NonZeroUsize) -> Range<usize> {
    sequence_length.saturating_sub(window.get())..sequence_length
}

/// Plans physical writes for an appended token chunk in a standard ring.
///
/// Do not use this function to write uncommitted speculative tokens into an
/// existing ring. Use [`plan_staged_ring_transaction`] and commit only the
/// accepted prefix, otherwise wraparound can destroy data needed by rollback.
///
/// # Errors
///
/// Returns [`RingPlanError::PositionOverflow`] when `start_position +
/// token_count` cannot be represented by `usize`.
pub fn plan_ring_write(
    start_position: usize,
    token_count: usize,
    capacity: NonZeroUsize,
) -> Result<RingWritePlan, RingPlanError> {
    plan_ring_write_slots(start_position, token_count, capacity.get())
}

/// Verified core of [`plan_ring_write`] over a raw, positive capacity.
///
/// Input index `j` of the chunk has absolute position `start_position + j`.
/// Only the newest `min(token_count, capacity)` inputs survive, and each lands
/// in physical slot `(start_position + j) mod capacity`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    requires
        capacity > 0,
    ensures
        match plan {
            Err(error) => {
                &&& start_position + token_count > usize::MAX
                &&& error == (RingPlanError::PositionOverflow { start_position, token_count })
            },
            Ok(plan) => {
                let retained = if token_count <= capacity { token_count } else { capacity };
                let segments = plan.segments;
                &&& start_position + token_count <= usize::MAX
                &&& plan.appended_positions.start == start_position
                &&& plan.appended_positions.end == start_position + token_count
                &&& plan.retained_input.start == token_count - retained
                &&& plan.retained_input.end == token_count
                &&& segments.len() <= 2
                &&& (segments.len() == 0) == (token_count == 0)
                &&& segments.len() >= 1 ==> segments[0].input.start == token_count - retained
                &&& segments.len() >= 1 ==> segments[segments.len() - 1].input.end == token_count
                &&& segments.len() == 2 ==> segments[1].input.start == segments[0].input.end
                &&& forall|i: int| #![auto] 0 <= i < segments.len() ==> {
                    &&& segments[i].input.start < segments[i].input.end
                    &&& segments[i].physical.start < segments[i].physical.end
                    &&& segments[i].physical.end <= capacity
                    &&& segments[i].input.end - segments[i].input.start
                        == segments[i].physical.end - segments[i].physical.start
                    &&& forall|j: int| segments[i].input.start <= j < segments[i].input.end ==>
                        #[trigger] ring_slot(start_position as int, j, capacity as int)
                            == segments[i].physical.start + (j - segments[i].input.start)
                }
                &&& segments.len() == 2 ==> segments[1].physical.end <= segments[0].physical.start
                &&& retained == capacity && segments.len() == 1 ==>
                    segments[0].physical.start == 0 && segments[0].physical.end == capacity
                &&& retained == capacity && segments.len() == 2 ==>
                    segments[0].physical.end == capacity
                        && segments[1].physical.end == segments[0].physical.start
            },
        }
))]
fn plan_ring_write_slots(
    start_position: usize,
    token_count: usize,
    capacity: usize,
) -> Result<RingWritePlan, RingPlanError> {
    let Some(end_position) = start_position.checked_add(token_count) else {
        return Err(RingPlanError::PositionOverflow {
            start_position,
            token_count,
        });
    };
    let retained_count = token_count.min(capacity);
    let retained_input_start = token_count - retained_count;
    let retained_input = retained_input_start..token_count;
    let retained_absolute_start = end_position - retained_count;
    let segments = write_segments(
        retained_input_start,
        retained_absolute_start,
        retained_count,
        capacity,
    );
    // Input index `j` has absolute position `start_position + j`, which equals
    // the retained run's start plus its offset `j - retained_input_start`.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|j: int| retained_input_start <= j < token_count implies
            #[trigger] ring_slot(start_position as int, j, capacity as int)
                == ring_slot(
                    retained_absolute_start as int,
                    j - retained_input_start,
                    capacity as int,
                )
        by {
            assert(start_position as int + j
                == retained_absolute_start as int + (j - retained_input_start));
        }
    }

    Ok(RingWritePlan {
        appended_positions: start_position..end_position,
        retained_input,
        segments,
    })
}

/// Plans append-only staging for a speculative ring-cache transaction.
///
/// The plan itself never writes the committed ring. Store the proposed values
/// in a separate linear tensor of `token_count` entries, then call
/// [`StagedRingTransactionPlan::commit_prefix`] after verification. Dropping
/// the staging tensor implements physical rollback exactly.
///
/// # Errors
///
/// Returns [`RingPlanError::PositionOverflow`] when the tentative absolute end
/// cannot be represented by `usize`.
pub fn plan_staged_ring_transaction(
    start_position: usize,
    token_count: usize,
) -> Result<StagedRingTransactionPlan, RingPlanError> {
    let end_position =
        start_position
            .checked_add(token_count)
            .ok_or(RingPlanError::PositionOverflow {
                start_position,
                token_count,
            })?;
    Ok(StagedRingTransactionPlan {
        staged_positions: start_position..end_position,
        staged_input: 0..token_count,
    })
}

/// Plans chronological slices for the newest positions in a standard ring.
///
/// `window` may be smaller than physical `capacity`; only
/// `min(sequence_length, window)` tokens are returned.
///
/// # Errors
///
/// Returns [`RingPlanError::WindowExceedsCapacity`] when local retention cannot
/// fit in the ring.
pub fn plan_ring_read(
    sequence_length: usize,
    window: NonZeroUsize,
    capacity: NonZeroUsize,
) -> Result<RingReadPlan, RingPlanError> {
    validate_window(window, capacity)?;
    let retained_positions = sliding_window_range(sequence_length, window);
    Ok(RingReadPlan {
        segments: read_segments(
            retained_positions.start,
            retained_positions.end,
            capacity.get(),
        ),
        retained_positions,
    })
}

/// Verified chronological split of a retained absolute interval.
///
/// The caller guarantees that the retained interval is no wider than the ring.
/// Absolute position `p` is read from physical slot `p mod capacity`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(segments =>
    requires
        capacity > 0,
        retained_start <= retained_end,
        retained_end - retained_start <= capacity,
    ensures
        segments.len() <= 2,
        (segments.len() == 0) == (retained_start == retained_end),
        segments.len() >= 1 ==> segments[0].absolute.start == retained_start,
        segments.len() >= 1 ==> segments[segments.len() - 1].absolute.end == retained_end,
        segments.len() == 2 ==> segments[1].absolute.start == segments[0].absolute.end,
        forall|i: int| #![auto] 0 <= i < segments.len() ==> {
            &&& segments[i].absolute.start < segments[i].absolute.end
            &&& segments[i].physical.start < segments[i].physical.end
            &&& segments[i].physical.end <= capacity
            &&& segments[i].absolute.end - segments[i].absolute.start
                == segments[i].physical.end - segments[i].physical.start
            &&& forall|p: int| segments[i].absolute.start <= p < segments[i].absolute.end ==>
                #[trigger] ring_slot(p, 0, capacity as int)
                    == segments[i].physical.start + (p - segments[i].absolute.start)
        },
))]
fn read_segments(
    retained_start: usize,
    retained_end: usize,
    capacity: usize,
) -> Vec<RingReadSegment> {
    let retained_count = retained_end - retained_start;
    let physical_start = retained_start % capacity;
    let first_count = retained_count.min(capacity - physical_start);
    let second_count = retained_count - first_count;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|p: int| retained_start <= p < retained_end implies
            #[trigger] ring_slot(p, 0, capacity as int)
                == if p - retained_start < first_count {
                    physical_start + (p - retained_start)
                } else {
                    p - retained_start - first_count
                }
        by {
            lemma_slot_after_offset(retained_start as int, p - retained_start, capacity as int);
        }
    }

    let mut segments = Vec::with_capacity(if second_count > 0 { 2 } else { 1 });
    if first_count > 0 {
        segments.push(RingReadSegment {
            absolute: retained_start..retained_start + first_count,
            physical: physical_start..physical_start + first_count,
        });
    }
    if second_count > 0 {
        segments.push(RingReadSegment {
            absolute: retained_start + first_count..retained_end,
            physical: 0..second_count,
        });
    }
    segments
}

/// Plans writes into both halves of a mirrored ring allocation.
///
/// # Errors
///
/// Returns an error when the absolute end or doubled allocation indices
/// overflow `usize`.
pub fn plan_mirrored_ring_write(
    start_position: usize,
    token_count: usize,
    capacity: NonZeroUsize,
) -> Result<Vec<MirroredRingWriteSegment>, RingPlanError> {
    capacity
        .get()
        .checked_mul(2)
        .ok_or(RingPlanError::DoubledCapacityOverflow {
            capacity: capacity.get(),
        })?;
    let standard = plan_ring_write(start_position, token_count, capacity)?;
    Ok(standard
        .segments
        .iter()
        .map(|segment| MirroredRingWriteSegment {
            input: segment.input.clone(),
            primary: segment.physical.clone(),
            mirror: segment.physical.start + capacity.get()..segment.physical.end + capacity.get(),
        })
        .collect())
}

/// Plans the single contiguous read enabled by a mirrored ring.
///
/// # Errors
///
/// Returns an error when the doubled capacity overflows `usize`.
pub fn plan_mirrored_ring_read(
    sequence_length: usize,
    capacity: NonZeroUsize,
) -> Result<MirroredRingReadPlan, RingPlanError> {
    capacity
        .get()
        .checked_mul(2)
        .ok_or(RingPlanError::DoubledCapacityOverflow {
            capacity: capacity.get(),
        })?;
    let retained_positions = sliding_window_range(sequence_length, capacity);
    let retained_count = retained_positions.end - retained_positions.start;
    let physical_start = retained_positions.start % capacity.get();

    Ok(MirroredRingReadPlan {
        retained_positions,
        physical: physical_start..physical_start + retained_count,
    })
}

/// Splits at most one ring wrap into ordered contiguous input/destination ranges.
///
/// The caller ensures `count <= capacity` and nonzero capacity; the first range
/// starts at `absolute_start % capacity`, and any remainder starts at slot zero.
/// It also ensures `input_start <= usize::MAX - count`, so both input endpoints fit.
/// Input `input_start + k` is written to slot `(absolute_start + k) mod capacity`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(segments =>
    requires
        capacity > 0,
        count <= capacity,
        input_start + count <= usize::MAX,
    ensures
        segments.len() <= 2,
        (segments.len() == 0) == (count == 0),
        segments.len() >= 1 ==> segments[0].input.start == input_start,
        segments.len() >= 1 ==> segments[segments.len() - 1].input.end == input_start + count,
        segments.len() == 2 ==> segments[1].input.start == segments[0].input.end,
        forall|i: int| #![auto] 0 <= i < segments.len() ==> {
            &&& segments[i].input.start < segments[i].input.end
            &&& segments[i].physical.start < segments[i].physical.end
            &&& segments[i].physical.end <= capacity
            &&& segments[i].input.end - segments[i].input.start
                == segments[i].physical.end - segments[i].physical.start
            &&& forall|j: int| segments[i].input.start <= j < segments[i].input.end ==>
                #[trigger] ring_slot(absolute_start as int, j - input_start, capacity as int)
                    == segments[i].physical.start + (j - segments[i].input.start)
        },
        segments.len() == 2 ==> segments[1].physical.end <= segments[0].physical.start,
        count == capacity && segments.len() == 1 ==>
            segments[0].physical.start == 0 && segments[0].physical.end == capacity,
        count == capacity && segments.len() == 2 ==>
            segments[0].physical.end == capacity
                && segments[1].physical.end == segments[0].physical.start,
))]
fn write_segments(
    input_start: usize,
    absolute_start: usize,
    count: usize,
    capacity: usize,
) -> Vec<RingWriteSegment> {
    let physical_start = absolute_start % capacity;
    let first_count = count.min(capacity - physical_start);
    let second_count = count - first_count;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|k: int| 0 <= k < count implies
            #[trigger] ring_slot(absolute_start as int, k, capacity as int)
                == if k < first_count { physical_start + k } else { k - first_count }
        by {
            lemma_slot_after_offset(absolute_start as int, k, capacity as int);
        }
    }

    let mut segments = Vec::with_capacity(if second_count > 0 { 2 } else { 1 });
    if first_count > 0 {
        segments.push(RingWriteSegment {
            input: input_start..input_start + first_count,
            physical: physical_start..physical_start + first_count,
        });
    }
    if second_count > 0 {
        segments.push(RingWriteSegment {
            input: input_start + first_count..input_start + count,
            physical: 0..second_count,
        });
    }
    segments
}

/// Rejects a retained window wider than the nonzero physical ring capacity.
fn validate_window(window: NonZeroUsize, capacity: NonZeroUsize) -> Result<(), RingPlanError> {
    if window <= capacity {
        Ok(())
    } else {
        Err(RingPlanError::WindowExceedsCapacity {
            window: window.get(),
            capacity: capacity.get(),
        })
    }
}

/// Invalid dimensions or arithmetic in ring-buffer planning.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingPlanError {
    /// Absolute chunk-end calculation overflowed.
    PositionOverflow {
        /// First absolute position of the chunk.
        start_position: usize,
        /// Number of appended tokens.
        token_count: usize,
    },
    /// The requested read window cannot fit in physical storage.
    WindowExceedsCapacity {
        /// Requested retained width.
        window: usize,
        /// Physical ring width.
        capacity: usize,
    },
    /// A mirrored allocation of `2 * capacity` cannot be represented.
    DoubledCapacityOverflow {
        /// Undoubled ring capacity.
        capacity: usize,
    },
    /// Verification attempted to accept more entries than were staged.
    CommitPrefixExceedsStaged {
        /// Invalid accepted-prefix length.
        accepted: usize,
        /// Available proposal length.
        staged: usize,
    },
}

impl fmt::Display for RingPlanError {
    /// Formats the failed ring arithmetic or proposal boundary with the offending counts.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositionOverflow {
                start_position,
                token_count,
            } => write!(
                formatter,
                "absolute position {start_position} overflows when appending {token_count} tokens"
            ),
            Self::WindowExceedsCapacity { window, capacity } => write!(
                formatter,
                "sliding window {window} exceeds physical ring capacity {capacity}"
            ),
            Self::DoubledCapacityOverflow { capacity } => write!(
                formatter,
                "mirrored ring capacity 2 * {capacity} overflows usize"
            ),
            Self::CommitPrefixExceedsStaged { accepted, staged } => write!(
                formatter,
                "cannot commit {accepted} ring entries from a staging buffer of length {staged}"
            ),
        }
    }
}

impl std::error::Error for RingPlanError {}

#[cfg(test)]
#[path = "ring_tests.rs"]
mod tests;
