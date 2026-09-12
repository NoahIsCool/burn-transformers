//! Exact machine-integer model of the native cache ring-write partition.
//!
//! The formulas mirror `src/cache/ring.rs::write_segments`; this standalone
//! proof is not compiled into the library. See `cache_ring.md` for the source
//! correspondence, verification command, and unproved runtime boundaries.

use vstd::prelude::*;

verus! {

/// Scalar endpoints for the two possible contiguous ring-write segments.
pub struct RingWriteSplit {
    /// First physical slot, equal to the absolute starting position modulo capacity.
    pub physical_start: usize,
    /// Number of retained tokens written before the physical end of the ring.
    pub first_count: usize,
    /// Number of retained tokens written from physical slot zero after wrapping.
    pub second_count: usize,
    /// Exclusive input endpoint of the first segment and start of the second.
    pub input_boundary: usize,
    /// Exclusive input endpoint after both segments.
    pub input_end: usize,
    /// Exclusive physical endpoint of the first segment.
    pub physical_end: usize,
}

/// Splits a retained input suffix into bounded, disjoint physical intervals.
///
/// `count` is at most one ring capacity. The caller supplies an input suffix
/// whose exclusive endpoint fits `usize`; `absolute_start` itself is arbitrary.
/// Zero-count intervals are omitted by the production vector-building code.
pub fn split_ring_write(
    input_start: usize,
    absolute_start: usize,
    count: usize,
    capacity: usize,
) -> (split: RingWriteSplit)
    requires
        capacity > 0,
        count <= capacity,
        input_start as int + count as int <= usize::MAX as int,
    ensures
        split.physical_start == absolute_start % capacity,
        split.physical_start < capacity,
        split.first_count <= count,
        split.first_count <= capacity - split.physical_start,
        split.first_count as int == if count <= capacity - split.physical_start {
            count as int
        } else {
            capacity - split.physical_start
        },
        split.first_count as int + split.second_count as int == count,
        split.second_count <= split.physical_start,
        split.input_boundary as int == input_start as int + split.first_count as int,
        split.input_end as int == input_start as int + count as int,
        input_start <= split.input_boundary <= split.input_end,
        split.input_end as int - split.input_boundary as int == split.second_count,
        split.physical_end as int == split.physical_start as int + split.first_count as int,
        split.physical_end <= capacity,
        split.physical_end as int - split.physical_start as int == split.first_count,
        count == 0 ==> split.first_count == 0 && split.second_count == 0,
        count > 0 ==> split.first_count > 0,
        split.second_count > 0 ==> split.physical_end == capacity,
        count == capacity ==> split.physical_end == capacity
            && split.second_count == split.physical_start,
{
    let physical_start = absolute_start % capacity;
    let available = capacity - physical_start;
    // The explicit branch is the same minimum used by `usize::min` at runtime.
    let first_count = if count <= available { count } else { available };
    let second_count = count - first_count;
    let input_boundary = input_start + first_count;
    let input_end = input_start + count;
    let physical_end = physical_start + first_count;
    RingWriteSplit {
        physical_start,
        first_count,
        second_count,
        input_boundary,
        input_end,
        physical_end,
    }
}

/// Scalar input suffix selected by the public checked ring-write planner.
pub struct RetainedWriteInput {
    /// First retained index in the incoming token chunk.
    pub input_start: usize,
    /// Absolute position of that first retained token.
    pub absolute_start: usize,
    /// Number of newest input tokens retained by the ring.
    pub count: usize,
    /// Exclusive absolute endpoint after the complete append.
    pub absolute_end: usize,
}

/// Reproduces the checked arithmetic that establishes the split preconditions.
///
/// `None` models the production `PositionOverflow` error. Successful results
/// retain exactly `min(token_count, capacity)` newest tokens, including an empty
/// suffix when `token_count` is zero.
pub fn prepare_retained_input(
    start_position: usize,
    token_count: usize,
    capacity: usize,
) -> (result: Option<RetainedWriteInput>)
    requires
        capacity > 0,
    ensures
        result.is_none() <==> start_position as int + token_count as int > usize::MAX as int,
        match result {
            Some(retained) => {
                &&& retained.count == if token_count <= capacity { token_count } else { capacity }
                &&& retained.count <= capacity
                &&& retained.input_start as int + retained.count as int == token_count
                &&& retained.input_start as int + retained.count as int <= usize::MAX as int
                &&& retained.absolute_start as int + retained.count as int == retained.absolute_end
                &&& retained.absolute_end as int == start_position as int + token_count as int
                &&& retained.absolute_start as int == start_position as int + retained.input_start as int
            },
            None => true,
        },
{
    let absolute_end = match start_position.checked_add(token_count) {
        Some(end) => end,
        None => return None,
    };
    let count = if token_count <= capacity { token_count } else { capacity };
    let input_start = token_count - count;
    let absolute_start = absolute_end - count;
    Some(RetainedWriteInput { input_start, absolute_start, count, absolute_end })
}

/// Verifies composition of the public planner's arithmetic and segment split.
///
/// Every non-overflowing append produces a complete retained input partition
/// ending at `token_count`, with each physical destination inside the ring.
pub fn checked_write_partition(
    start_position: usize,
    token_count: usize,
    capacity: usize,
) -> (result: Option<RingWriteSplit>)
    requires
        capacity > 0,
    ensures
        result.is_none() <==> start_position as int + token_count as int > usize::MAX as int,
        match result {
            Some(split) => {
                &&& split.input_end == token_count
                &&& split.physical_end <= capacity
                &&& split.second_count <= split.physical_start < capacity
                &&& split.first_count as int + split.second_count as int
                    == if token_count <= capacity { token_count as int } else { capacity as int }
            },
            None => true,
        },
{
    match prepare_retained_input(start_position, token_count, capacity) {
        Some(retained) => Some(split_ring_write(
            retained.input_start,
            retained.absolute_start,
            retained.count,
            capacity,
        )),
        None => None,
    }
}

}
