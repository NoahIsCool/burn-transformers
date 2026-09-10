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

use core::fmt;
use core::num::NonZeroUsize;
use core::ops::Range;

/// One contiguous piece of an incoming chunk written into a standard ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingWriteSegment {
    input: Range<usize>,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingWritePlan {
    appended_positions: Range<usize>,
    retained_input: Range<usize>,
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
    staged_positions: Range<usize>,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingReadSegment {
    absolute: Range<usize>,
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
    retained_positions: Range<usize>,
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
    input: Range<usize>,
    primary: Range<usize>,
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
    retained_positions: Range<usize>,
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
    let end_position =
        start_position
            .checked_add(token_count)
            .ok_or(RingPlanError::PositionOverflow {
                start_position,
                token_count,
            })?;
    let retained_count = token_count.min(capacity.get());
    let retained_input_start = token_count - retained_count;
    let retained_input = retained_input_start..token_count;
    let retained_absolute_start = end_position - retained_count;
    let segments = write_segments(
        retained_input_start,
        retained_absolute_start,
        retained_count,
        capacity.get(),
    );

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
    let retained_count = retained_positions.end - retained_positions.start;
    let physical_start = retained_positions.start % capacity.get();
    let first_count = retained_count.min(capacity.get() - physical_start);

    let mut segments = Vec::with_capacity(usize::from(retained_count > first_count) + 1);
    if first_count > 0 {
        segments.push(RingReadSegment {
            absolute: retained_positions.start..retained_positions.start + first_count,
            physical: physical_start..physical_start + first_count,
        });
    }
    let second_count = retained_count - first_count;
    if second_count > 0 {
        segments.push(RingReadSegment {
            absolute: retained_positions.start + first_count..retained_positions.end,
            physical: 0..second_count,
        });
    }

    Ok(RingReadPlan {
        retained_positions,
        segments,
    })
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
fn write_segments(
    input_start: usize,
    absolute_start: usize,
    count: usize,
    capacity: usize,
) -> Vec<RingWriteSegment> {
    let physical_start = absolute_start % capacity;
    let first_count = count.min(capacity - physical_start);
    let mut segments = Vec::with_capacity(usize::from(count > first_count) + 1);
    if first_count > 0 {
        segments.push(RingWriteSegment {
            input: input_start..input_start + first_count,
            physical: physical_start..physical_start + first_count,
        });
    }
    let second_count = count - first_count;
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
