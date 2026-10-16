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
//! slot exactly once. Mirrored writes cover the same retained suffix as the
//! standard plan, stay below `2 * capacity` with the mirror exactly one
//! capacity after the primary, a mirrored read's offset `j` reads the slot of
//! position `start + j`, and a staged commit succeeds exactly for a staged,
//! representable prefix and plans only that prefix. Staging plans, sliding
//! window starts (`max(0, L - W)`), and window-capacity validation of reads
//! are exact. Every error is returned exactly when its condition holds.
//! Tensor copies driven by these plans remain covered by the cache test suite.

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

/// `plan` is the exact write plan for `token_count` inputs starting at absolute
/// `start_position` in a ring of `capacity` slots.
///
/// Only the newest `retained = min(token_count, capacity)` inputs survive.
/// Their at most two segments are non-empty, ordered, contiguous, and cover
/// exactly the retained suffix `[token_count - retained, token_count)` of the
/// input; input `j` lands in slot `(start_position + j) mod capacity`; and the
/// two physical ranges never overlap, tiling the whole ring when it is full.
pub(crate) open spec fn write_plan_correct(
    plan: RingWritePlan,
    start_position: usize,
    token_count: usize,
    capacity: usize,
) -> bool {
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
}

/// First absolute position of the sliding window ending at `length`: `max(0, length - window)`.
pub open spec fn window_start(length: int, window: int) -> int {
    if length >= window { length - window } else { 0 }
}

} // verus!

/// One contiguous piece of an incoming chunk written into a standard ring.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingWriteSegment {
    /// Contiguous interval within the incoming token chunk.
    pub(crate) input: Range<usize>,
    /// Equally sized destination interval inside the physical ring.
    pub(crate) physical: Range<usize>,
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
    pub(crate) appended_positions: Range<usize>,
    /// Newest incoming suffix that survives ring eviction.
    pub(crate) retained_input: Range<usize>,
    /// Contiguous destination writes in incoming-token order.
    pub(crate) segments: Vec<RingWriteSegment>,
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
#[cfg_attr(verus_keep_ghost, verus_verify)]
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
        commit_prefix_slots(
            self.staged_positions.start,
            self.staged_input.len(),
            accepted_tokens,
            capacity.get(),
        )
    }
}

/// Verified core of [`StagedRingTransactionPlan::commit_prefix`].
///
/// Accepting `accepted` of `staged` proposals at absolute `staged_start`
/// succeeds exactly when `accepted <= staged` and the accepted end is
/// representable, and then plans exactly the accepted prefix: the segments
/// cover the newest `min(accepted, capacity)` accepted entries in order, all
/// inside `[0, accepted)` and hence inside the staging buffer, and proposal
/// `j` lands in slot `(staged_start + j) mod capacity`. Rejected proposals are
/// never planned, so rollback needs no ring write at all.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        capacity > 0,
    ensures
        result is Ok <==> (accepted <= staged && staged_start + accepted <= usize::MAX),
        accepted > staged ==> result == Err::<RingWritePlan, RingPlanError>(
            RingPlanError::CommitPrefixExceedsStaged { accepted, staged }),
        accepted <= staged && staged_start + accepted > usize::MAX ==>
            result == Err::<RingWritePlan, RingPlanError>(RingPlanError::PositionOverflow {
                start_position: staged_start,
                token_count: accepted,
            }),
        result matches Ok(plan) ==> write_plan_correct(plan, staged_start, accepted, capacity),
))]
fn commit_prefix_slots(
    staged_start: usize,
    staged: usize,
    accepted: usize,
    capacity: usize,
) -> Result<RingWritePlan, RingPlanError> {
    if accepted > staged {
        return Err(RingPlanError::CommitPrefixExceedsStaged { accepted, staged });
    }
    plan_ring_write_slots(staged_start, accepted, capacity)
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
    pub(crate) absolute: Range<usize>,
    /// Equally sized interval inside physical ring storage.
    pub(crate) physical: Range<usize>,
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
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingReadPlan {
    /// Absolute suffix still visible under the retention policy.
    pub(crate) retained_positions: Range<usize>,
    /// Physical slices ordered from oldest to newest retained token.
    pub(crate) segments: Vec<RingReadSegment>,
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
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirroredRingWriteSegment {
    /// Contiguous interval within the incoming token chunk.
    pub(crate) input: Range<usize>,
    /// Destination interval in the first ring half.
    pub(crate) primary: Range<usize>,
    /// Duplicate destination offset by one ring capacity.
    pub(crate) mirror: Range<usize>,
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
#[cfg_attr(verus_keep_ghost, verus_verify)]
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
    sliding_window_start(sequence_length, window.get())..sequence_length
}

/// Verified start of [`sliding_window_range`]: exactly `max(0, length - window)`.
///
/// The returned start never exceeds `sequence_length`, and the window it
/// opens holds exactly `min(sequence_length, window)` positions.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(start =>
    ensures
        start as int == window_start(sequence_length as int, window as int),
        start <= sequence_length,
        sequence_length - start
            == if sequence_length <= window { sequence_length } else { window },
))]
fn sliding_window_start(sequence_length: usize, window: usize) -> usize {
    sequence_length.saturating_sub(window)
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
            Ok(plan) => write_plan_correct(plan, start_position, token_count, capacity),
        }
))]
pub(crate) fn plan_ring_write_slots(
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
    staged_transaction_slots(start_position, token_count)
}

/// Verified core of [`plan_staged_ring_transaction`].
///
/// The plan exists exactly when `start_position + token_count` fits, and then
/// stages the absolute positions `[start_position, start_position +
/// token_count)` at linear offsets `[0, token_count)`, one offset per proposal.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Err(error) => {
                &&& start_position + token_count > usize::MAX
                &&& error == (RingPlanError::PositionOverflow { start_position, token_count })
            },
            Ok(plan) => {
                &&& start_position + token_count <= usize::MAX
                &&& plan.staged_positions.start == start_position
                &&& plan.staged_positions.end == start_position + token_count
                &&& plan.staged_input.start == 0
                &&& plan.staged_input.end == token_count
            },
        }
))]
fn staged_transaction_slots(
    start_position: usize,
    token_count: usize,
) -> Result<StagedRingTransactionPlan, RingPlanError> {
    let Some(end_position) = start_position.checked_add(token_count) else {
        return Err(RingPlanError::PositionOverflow {
            start_position,
            token_count,
        });
    };
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
    ring_read_slots(sequence_length, window.get(), capacity.get())
}

/// Verified core of [`plan_ring_read`] over raw, positive widths.
///
/// The plan exists exactly when `window <= capacity`. It then retains the
/// absolute window `[max(0, L - window), L)` and reads it in at most two
/// chronological slices that partition it in order, with position `p` read
/// from physical slot `p mod capacity`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        window > 0,
        capacity > 0,
    ensures
        match result {
            Err(error) => {
                &&& window > capacity
                &&& error == (RingPlanError::WindowExceedsCapacity { window, capacity })
            },
            Ok(plan) => {
                let segments = plan.segments;
                &&& window <= capacity
                &&& plan.retained_positions.start as int
                    == window_start(sequence_length as int, window as int)
                &&& plan.retained_positions.end == sequence_length
                &&& segments.len() <= 2
                &&& (segments.len() == 0) == (plan.retained_positions.start == sequence_length)
                &&& segments.len() >= 1 ==>
                    segments[0].absolute.start == plan.retained_positions.start
                &&& segments.len() >= 1 ==>
                    segments[segments.len() - 1].absolute.end == sequence_length
                &&& segments.len() == 2 ==> segments[1].absolute.start == segments[0].absolute.end
                &&& forall|i: int| #![auto] 0 <= i < segments.len() ==> {
                    &&& segments[i].absolute.start < segments[i].absolute.end
                    &&& segments[i].physical.start < segments[i].physical.end
                    &&& segments[i].physical.end <= capacity
                    &&& segments[i].absolute.end - segments[i].absolute.start
                        == segments[i].physical.end - segments[i].physical.start
                    &&& forall|p: int| segments[i].absolute.start <= p < segments[i].absolute.end
                        ==> #[trigger] ring_slot(p, 0, capacity as int)
                            == segments[i].physical.start + (p - segments[i].absolute.start)
                }
            },
        }
))]
pub(crate) fn ring_read_slots(
    sequence_length: usize,
    window: usize,
    capacity: usize,
) -> Result<RingReadPlan, RingPlanError> {
    if window > capacity {
        return Err(RingPlanError::WindowExceedsCapacity { window, capacity });
    }
    let retained_start = sliding_window_start(sequence_length, window);
    Ok(RingReadPlan {
        segments: read_segments(retained_start, sequence_length, capacity),
        retained_positions: retained_start..sequence_length,
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
    mirrored_write_slots(start_position, token_count, capacity.get())
}

/// Verified core of [`plan_mirrored_ring_write`] over a raw, positive capacity.
///
/// The plan exists exactly when `2 * capacity` and the absolute chunk end are
/// representable. Its segments are the standard ring plan's segments, each
/// written twice: at its primary slots and at the same slots shifted by
/// `capacity`. They therefore cover exactly the retained input suffix
/// `[token_count - min(token_count, capacity), token_count)` in order, both
/// copies stay below `2 * capacity`, the primary ranges never overlap, and
/// every token lands in slot `(start_position + j) mod capacity`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        capacity > 0,
    ensures
        result is Ok <==> (2 * capacity <= usize::MAX && start_position + token_count <= usize::MAX),
        2 * capacity > usize::MAX ==> result
            == Err::<Vec<MirroredRingWriteSegment>, RingPlanError>(
                RingPlanError::DoubledCapacityOverflow { capacity }),
        2 * capacity <= usize::MAX && start_position + token_count > usize::MAX ==> result
            == Err::<Vec<MirroredRingWriteSegment>, RingPlanError>(
                RingPlanError::PositionOverflow { start_position, token_count }),
        result matches Ok(segments) ==> {
            let retained = if token_count <= capacity { token_count } else { capacity };
            &&& segments.len() <= 2
            &&& (segments.len() == 0) == (token_count == 0)
            &&& segments.len() >= 1 ==> segments[0].input.start == token_count - retained
            &&& segments.len() >= 1 ==> segments[segments.len() - 1].input.end == token_count
            &&& segments.len() == 2 ==> segments[1].input.start == segments[0].input.end
            &&& segments.len() == 2 ==> segments[1].primary.end <= segments[0].primary.start
            &&& forall|i: int| #![auto] 0 <= i < segments.len() ==> {
                &&& segments[i].input.start < segments[i].input.end <= token_count
                &&& segments[i].primary.start < segments[i].primary.end
                &&& segments[i].primary.end <= capacity
                &&& segments[i].input.end - segments[i].input.start
                    == segments[i].primary.end - segments[i].primary.start
                &&& segments[i].mirror.start == segments[i].primary.start + capacity
                &&& segments[i].mirror.end == segments[i].primary.end + capacity
                &&& segments[i].mirror.end <= 2 * capacity
                &&& forall|j: int| segments[i].input.start <= j < segments[i].input.end ==>
                    #[trigger] ring_slot(start_position as int, j, capacity as int)
                        == segments[i].primary.start + (j - segments[i].input.start)
            }
        },
))]
pub(crate) fn mirrored_write_slots(
    start_position: usize,
    token_count: usize,
    capacity: usize,
) -> Result<Vec<MirroredRingWriteSegment>, RingPlanError> {
    if capacity.checked_mul(2).is_none() {
        return Err(RingPlanError::DoubledCapacityOverflow { capacity });
    }
    let standard = plan_ring_write_slots(start_position, token_count, capacity)?;
    let mut mirrored: Vec<MirroredRingWriteSegment> = Vec::with_capacity(standard.segments.len());
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= standard.segments.len(),
            capacity > 0,
            2 * capacity <= usize::MAX,
            write_plan_correct(standard, start_position, token_count, capacity),
            mirrored.len() == index,
            forall|i: int| #![auto] 0 <= i < index ==> {
                &&& mirrored[i].input == standard.segments[i].input
                &&& mirrored[i].primary == standard.segments[i].physical
                &&& mirrored[i].mirror.start == standard.segments[i].physical.start + capacity
                &&& mirrored[i].mirror.end == standard.segments[i].physical.end + capacity
            },
        decreases standard.segments.len() - index,
    ))]
    while index < standard.segments.len() {
        let segment = &standard.segments[index];
        mirrored.push(MirroredRingWriteSegment {
            input: segment.input.start..segment.input.end,
            primary: segment.physical.start..segment.physical.end,
            mirror: segment.physical.start + capacity..segment.physical.end + capacity,
        });
        index += 1;
    }
    Ok(mirrored)
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
    mirrored_read_slots(sequence_length, capacity.get())
}

/// Verified core of [`plan_mirrored_ring_read`] over a raw, positive capacity.
///
/// A mirrored allocation holds ring slot `s` at physical indices `s` and
/// `s + capacity`, so physical index `q < 2 * capacity` holds slot
/// `q mod capacity`. Starting at `start mod capacity`, the retained run of at
/// most `capacity` positions is therefore one contiguous range, and offset
/// `j` reads exactly the slot of absolute position `start + j`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        capacity > 0,
    ensures
        result is Err <==> 2 * capacity > usize::MAX,
        result matches Ok(plan) ==> {
            let retained = if sequence_length <= capacity { sequence_length } else { capacity };
            &&& plan.retained_positions.start == sequence_length - retained
            &&& plan.retained_positions.end == sequence_length
            &&& plan.physical.start == plan.retained_positions.start % capacity
            &&& plan.physical.end == plan.physical.start + retained
            &&& plan.physical.end <= 2 * capacity
            &&& forall|j: int| 0 <= j < retained ==>
                #[trigger] ring_slot(plan.physical.start as int, j, capacity as int)
                    == ring_slot(plan.retained_positions.start as int, j, capacity as int)
        },
))]
fn mirrored_read_slots(
    sequence_length: usize,
    capacity: usize,
) -> Result<MirroredRingReadPlan, RingPlanError> {
    if capacity.checked_mul(2).is_none() {
        return Err(RingPlanError::DoubledCapacityOverflow { capacity });
    }
    let retained = if sequence_length <= capacity {
        sequence_length
    } else {
        capacity
    };
    let retained_start = sequence_length - retained;
    let physical_start = retained_start % capacity;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|j: int| 0 <= j < retained implies
            #[trigger] ring_slot(physical_start as int, j, capacity as int)
                == ring_slot(retained_start as int, j, capacity as int)
        by {
            // (a mod c + j) mod c == (a + j) mod c for 0 <= j < c.
            vstd::arithmetic::div_mod::lemma_add_mod_noop(retained_start as int, j, capacity as int);
            vstd::arithmetic::div_mod::lemma_add_mod_noop(physical_start as int, j, capacity as int);
            vstd::arithmetic::div_mod::lemma_small_mod(j as nat, capacity as nat);
            vstd::arithmetic::div_mod::lemma_mod_twice(retained_start as int, capacity as int);
        }
    }
    Ok(MirroredRingReadPlan {
        retained_positions: retained_start..sequence_length,
        physical: physical_start..physical_start + retained,
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
