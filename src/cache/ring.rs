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
}

impl fmt::Display for RingPlanError {
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
        }
    }
}

impl std::error::Error for RingPlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test values are non-zero")
    }

    #[test]
    fn sliding_window_is_absolute_and_saturating() {
        assert_eq!(sliding_window_range(0, nonzero(4)), 0..0);
        assert_eq!(sliding_window_range(3, nonzero(4)), 0..3);
        assert_eq!(sliding_window_range(4, nonzero(4)), 0..4);
        assert_eq!(sliding_window_range(9, nonzero(4)), 5..9);
    }

    #[test]
    fn write_plan_handles_empty_contiguous_wrapped_and_oversized_chunks() {
        let empty = plan_ring_write(3, 0, nonzero(4)).expect("valid plan");
        assert_eq!(empty.appended_positions(), 3..3);
        assert!(empty.segments().is_empty());

        let contiguous = plan_ring_write(1, 2, nonzero(4)).expect("valid plan");
        assert_eq!(contiguous.retained_input(), 0..2);
        assert_eq!(contiguous.segments()[0].input(), 0..2);
        assert_eq!(contiguous.segments()[0].physical(), 1..3);

        let wrapped = plan_ring_write(3, 3, nonzero(4)).expect("valid plan");
        assert_eq!(wrapped.segments().len(), 2);
        assert_eq!(wrapped.segments()[0].physical(), 3..4);
        assert_eq!(wrapped.segments()[1].physical(), 0..2);

        let oversized = plan_ring_write(2, 7, nonzero(4)).expect("valid plan");
        assert_eq!(oversized.appended_positions(), 2..9);
        assert_eq!(oversized.retained_input(), 3..7);
        assert_eq!(oversized.segments()[0].input(), 3..6);
        assert_eq!(oversized.segments()[0].physical(), 1..4);
        assert_eq!(oversized.segments()[1].input(), 6..7);
        assert_eq!(oversized.segments()[1].physical(), 0..1);
    }

    #[test]
    fn read_plan_returns_chronological_slices() {
        let plan = plan_ring_read(10, nonzero(4), nonzero(4)).expect("valid plan");
        assert_eq!(plan.retained_positions(), 6..10);
        assert_eq!(plan.segments().len(), 2);
        assert_eq!(plan.segments()[0].absolute(), 6..8);
        assert_eq!(plan.segments()[0].physical(), 2..4);
        assert_eq!(plan.segments()[1].absolute(), 8..10);
        assert_eq!(plan.segments()[1].physical(), 0..2);

        let partial = plan_ring_read(2, nonzero(3), nonzero(8)).expect("valid plan");
        assert_eq!(partial.retained_positions(), 0..2);
        assert_eq!(partial.segments().len(), 1);
        assert_eq!(partial.segments()[0].physical(), 0..2);
    }

    #[test]
    fn read_plan_rejects_impossible_window() {
        assert_eq!(
            plan_ring_read(4, nonzero(5), nonzero(4)),
            Err(RingPlanError::WindowExceedsCapacity {
                window: 5,
                capacity: 4,
            })
        );
    }

    #[test]
    fn mirrored_ring_writes_both_halves_and_reads_one_slice() {
        let writes = plan_mirrored_ring_write(3, 3, nonzero(4)).expect("valid plan");
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].primary(), 3..4);
        assert_eq!(writes[0].mirror(), 7..8);
        assert_eq!(writes[1].primary(), 0..2);
        assert_eq!(writes[1].mirror(), 4..6);

        let read = plan_mirrored_ring_read(6, nonzero(4)).expect("valid plan");
        assert_eq!(read.retained_positions(), 2..6);
        assert_eq!(read.physical(), 2..6);
    }

    #[test]
    fn arithmetic_errors_are_reported() {
        assert_eq!(
            plan_ring_write(usize::MAX, 1, nonzero(4)),
            Err(RingPlanError::PositionOverflow {
                start_position: usize::MAX,
                token_count: 1,
            })
        );
        let too_large = nonzero(usize::MAX / 2 + 1);
        assert_eq!(
            plan_mirrored_ring_read(0, too_large),
            Err(RingPlanError::DoubledCapacityOverflow {
                capacity: usize::MAX / 2 + 1,
            })
        );
    }

    #[test]
    fn exhaustive_write_plans_match_ring_simulation() {
        for capacity in 1..=8 {
            for start in 0..=20 {
                for count in 0..=20 {
                    let plan = plan_ring_write(start, count, nonzero(capacity))
                        .expect("small positions do not overflow");
                    let mut ring = vec![None; capacity];
                    for input_index in 0..count {
                        ring[(start + input_index) % capacity] = Some(input_index);
                    }

                    let planned = plan
                        .segments()
                        .iter()
                        .flat_map(|segment| segment.input().zip(segment.physical()))
                        .collect::<Vec<_>>();
                    let first_retained = count.saturating_sub(capacity);
                    let expected = (first_retained..count)
                        .map(|input| (input, (start + input) % capacity))
                        .collect::<Vec<_>>();
                    assert_eq!(planned, expected);
                    for (input, slot) in planned {
                        assert_eq!(ring[slot], Some(input));
                    }
                }
            }
        }
    }

    #[test]
    fn exhaustive_read_plans_reconstruct_absolute_order() {
        for capacity in 1..=8 {
            for window in 1..=capacity {
                for length in 0..=24 {
                    let plan = plan_ring_read(length, nonzero(window), nonzero(capacity))
                        .expect("window fits");
                    let mappings = plan
                        .segments()
                        .iter()
                        .flat_map(|segment| segment.absolute().zip(segment.physical()))
                        .collect::<Vec<_>>();
                    let expected = sliding_window_range(length, nonzero(window))
                        .map(|absolute| (absolute, absolute % capacity))
                        .collect::<Vec<_>>();
                    assert_eq!(mappings, expected);
                }
            }
        }
    }

    #[test]
    fn exhaustive_mirrored_reads_are_contiguous() {
        for capacity in 1..=16 {
            for length in 0..=64 {
                let plan = plan_mirrored_ring_read(length, nonzero(capacity))
                    .expect("small capacity can be doubled");
                assert_eq!(
                    plan.physical().len(),
                    plan.retained_positions().len(),
                    "capacity={capacity}, length={length}"
                );
                for (absolute, physical) in plan.retained_positions().zip(plan.physical()) {
                    assert_eq!(physical % capacity, absolute % capacity);
                }
            }
        }
    }
}
