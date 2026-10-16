//! KV planning against hand-picked configurations and naive oracles.

use core::ops::Range;

use proptest::prelude::*;

use super::*;

/// Builds the expected extent of a valid group.
fn extent(slots: usize, window: Option<usize>) -> Result<TimeExtent, TimeSlotViolation> {
    Ok(TimeExtent { slots, window })
}

/// Every storage mode reports its exact extent and width or its first invalid rule.
#[test]
fn committed_time_slots_cover_every_storage_rule() {
    use KvCacheStorage::{Full, MirroredSlidingRing, SlidingRing};
    assert_eq!(
        committed_time_slots(Full, false, None, Some(4), 16),
        extent(16, None)
    );
    assert_eq!(
        committed_time_slots(Full, true, None, None, 16),
        extent(16, None)
    );
    assert_eq!(
        committed_time_slots(SlidingRing, false, None, Some(4), 16),
        extent(4, Some(4))
    );
    assert_eq!(
        committed_time_slots(SlidingRing, false, Some(8), Some(4), 16),
        extent(8, Some(8))
    );
    assert_eq!(
        committed_time_slots(MirroredSlidingRing, false, None, Some(4), 16),
        extent(8, Some(4))
    );
    assert_eq!(
        committed_time_slots(MirroredSlidingRing, false, Some(3), Some(4), 16),
        extent(6, Some(3))
    );
    assert_eq!(
        committed_time_slots(SlidingRing, true, Some(32), None, 16),
        Err(TimeSlotViolation::InPlaceStagingRequiresFullStorage)
    );
    assert_eq!(
        committed_time_slots(Full, false, Some(4), None, 16),
        Err(TimeSlotViolation::WindowOnFullStorage { window: 4 })
    );
    assert_eq!(
        committed_time_slots(SlidingRing, false, Some(32), Some(4), 16),
        Err(TimeSlotViolation::WindowExceedsCapacity { window: 32 })
    );
    assert_eq!(
        committed_time_slots(MirroredSlidingRing, false, None, None, 16),
        Err(TimeSlotViolation::WindowRequired)
    );
    let huge = usize::MAX / 2 + 1;
    assert_eq!(
        committed_time_slots(MirroredSlidingRing, false, None, Some(huge), usize::MAX),
        Err(TimeSlotViolation::MirroredOverflow { window: huge })
    );
}

/// The effective width is the group's own window, else the bank's, and none for full storage.
#[test]
fn effective_window_prefers_the_group_window() {
    use KvCacheStorage::{Full, MirroredSlidingRing, SlidingRing};
    assert_eq!(effective_window_of(Full, Some(3), Some(4)), None);
    assert_eq!(effective_window_of(Full, None, Some(4)), None);
    for ring in [SlidingRing, MirroredSlidingRing] {
        assert_eq!(effective_window_of(ring, Some(3), Some(4)), Some(3));
        assert_eq!(effective_window_of(ring, None, Some(4)), Some(4));
        assert_eq!(effective_window_of(ring, None, None), None);
    }
}

/// Committed reads keep `min(L, W)` positions and retain the newest suffix.
#[test]
fn committed_reads_keep_the_newest_window() {
    assert_eq!(visible_count(3, Some(4)), 3);
    assert_eq!(visible_count(9, Some(4)), 4);
    assert_eq!(visible_count(9, None), 9);
    let rows = committed_read_rows(&[0, 3, 9], Some(4));
    assert_eq!(rows.keep, [0, 3, 4]);
    assert_eq!(rows.retained, [0..0, 0..3, 5..9]);
    let full = committed_read_rows(&[2, 7], None);
    assert_eq!(full.keep, [2, 7]);
    assert_eq!(full.retained, [0..2, 0..7]);
}

/// Working reads split each row's window and reject the first overflowing row.
#[test]
fn working_reads_split_windows_and_reject_first_bad_row() {
    let rows = working_read_rows(&[6, 1], &[3, 1], Some(4), 16).expect("within capacity");
    assert_eq!(rows.ends, [9, 2]);
    // Row 0 keeps one committed token and all three proposals; row 1 keeps both parts.
    assert_eq!(rows.keep, [1, 1]);
    assert_eq!(rows.retained, [5..9, 0..2]);
    assert_eq!(rows.windows[0].staged_skip, 0);
    let full = working_read_rows(&[6], &[3], None, 16).expect("within capacity");
    assert_eq!(full.retained.len(), 1);
    assert_eq!(full.retained[0], 0..9);
    assert_eq!(full.keep, [6]);
    assert_eq!(
        working_read_rows(&[1, 15, 20], &[1, 2, 0], Some(4), 16),
        Err((1, WriteEndViolation::ExceedsCapacity { end: 17 }))
    );
    assert_eq!(
        working_read_rows(&[usize::MAX], &[1], None, usize::MAX),
        Err((0, WriteEndViolation::Overflow))
    );
    // A zipped scan plans only the shorter input.
    assert_eq!(
        working_read_rows(&[1, 2], &[0], None, 4)
            .expect("within capacity")
            .ends,
        [1]
    );
}

/// Mirrored runs start at `start mod W` and stay below `2W`.
#[test]
fn mirrored_slots_are_one_contiguous_run() {
    assert_eq!(mirrored_window_slots(6, 4, 4), 2..6);
    assert_eq!(mirrored_window_slots(3, 0, 4), 3..3);
    assert_eq!(mirrored_window_slots(7, 1, 4), 3..4);
}

/// Staged read ranges cover proposals `[skip, P)` at positions `[C + skip, C + P)`.
#[test]
fn staged_read_ranges_align_offsets_with_positions() {
    let plan = working_window(6, 3, Some(2));
    assert_eq!(plan.staged_skip, 1);
    assert_eq!(staged_read_ranges(6, 3, plan), (7..9, 1..3));
    let full = working_window(6, 3, None);
    assert_eq!(staged_read_ranges(6, 3, full), (6..9, 0..3));
}

/// Write-ahead is admitted only for full storage, in place inside a transaction.
#[test]
fn write_ahead_requires_full_storage_and_in_place_staging() {
    use KvCacheStorage::{Full, MirroredSlidingRing, SlidingRing};
    assert_eq!(write_ahead_allowed(Full, false, false), Ok(()));
    assert_eq!(write_ahead_allowed(Full, true, true), Ok(()));
    assert_eq!(
        write_ahead_allowed(Full, false, true),
        Err(WriteAheadViolation::SeparateStaging)
    );
    for ring in [SlidingRing, MirroredSlidingRing] {
        assert_eq!(
            write_ahead_allowed(ring, true, false),
            Err(WriteAheadViolation::RingStorage)
        );
    }
}

proptest! {
    /// Working windows hold exactly the newest `min(C + P, W)` positions.
    #[test]
    fn working_windows_match_visible_suffix(
        committed in 0_usize..20,
        staged in 0_usize..8,
        window in 1_usize..10,
    ) {
        let plan = working_window(committed, staged, Some(window));
        let working = committed + staged;
        let visible = working.min(window);
        let staged_keep = staged - plan.staged_skip;
        prop_assert_eq!(plan.retained_start, working - visible);
        prop_assert_eq!(plan.committed_keep + staged_keep, visible);
        prop_assert_eq!(staged_keep, staged.min(window));
    }

    /// Mirrored offsets read the slot of the matching absolute position.
    #[test]
    fn mirrored_offsets_read_matching_slots(
        start in 0_usize..50,
        window in 1_usize..9,
        keep in 0_usize..9,
    ) {
        let keep = keep.min(window);
        let slots = mirrored_window_slots(start, keep, window);
        prop_assert_eq!(slots.len(), keep);
        prop_assert!(slots.end <= 2 * window);
        for (offset, physical) in slots.enumerate() {
            prop_assert_eq!(physical % window, (start + offset) % window);
        }
    }
}

/// Builds a direct segment for expectations.
fn segment(
    input: Range<usize>,
    primary: Range<usize>,
    mirror: Option<Range<usize>>,
) -> DirectSegment {
    DirectSegment {
        input,
        primary,
        mirror,
    }
}

/// Direct appends write full slots, wrapped ring slots, or both mirrored halves.
#[test]
fn direct_writes_address_each_storage_mode() {
    use KvCacheStorage::{Full, MirroredSlidingRing, SlidingRing};
    assert_eq!(
        direct_write_segments(Full, 5, 3, 2, None),
        Ok(vec![segment(2..5, 5..8, None)])
    );
    // Positions 6..10 in a ring of 4 wrap after slot 3; columns start at 1.
    assert_eq!(
        direct_write_segments(SlidingRing, 6, 4, 1, Some(4)),
        Ok(vec![segment(1..3, 2..4, None), segment(3..5, 0..2, None)])
    );
    assert_eq!(
        direct_write_segments(MirroredSlidingRing, 6, 4, 0, Some(4)),
        Ok(vec![
            segment(0..2, 2..4, Some(6..8)),
            segment(2..4, 0..2, Some(4..6)),
        ])
    );
    // An oversized chunk keeps only its newest `W` tokens.
    assert_eq!(
        direct_write_segments(SlidingRing, 0, 6, 0, Some(4)),
        Ok(vec![segment(2..4, 2..4, None), segment(4..6, 0..2, None)])
    );
    // An empty append checks nothing, even a missing width.
    assert_eq!(
        direct_write_segments(SlidingRing, 3, 0, 0, None),
        Ok(vec![])
    );
}

/// Direct appends report the first failed check.
#[test]
fn direct_writes_reject_the_first_failed_check() {
    use KvCacheStorage::{Full, MirroredSlidingRing, SlidingRing};
    for ring in [SlidingRing, MirroredSlidingRing] {
        assert_eq!(
            direct_write_segments(ring, 0, 1, 0, None),
            Err(DirectWriteViolation::WindowRequired)
        );
    }
    let overflow = DirectWriteViolation::Ring(RingPlanError::PositionOverflow {
        start_position: usize::MAX,
        token_count: 1,
    });
    assert_eq!(
        direct_write_segments(Full, usize::MAX, 1, 0, None),
        Err(overflow)
    );
    assert_eq!(
        direct_write_segments(SlidingRing, usize::MAX, 1, 0, Some(4)),
        Err(overflow)
    );
    let huge = usize::MAX / 2 + 1;
    assert_eq!(
        direct_write_segments(MirroredSlidingRing, usize::MAX, 1, 0, Some(huge)),
        Err(DirectWriteViolation::Ring(
            RingPlanError::DoubledCapacityOverflow { capacity: huge }
        ))
    );
}

/// A run writes its rows from input rows shifted down by the source offset.
#[test]
fn direct_runs_shift_source_rows() {
    let run = RowRun {
        first_row: 2,
        end_row: 4,
        start: 7,
        column: 1,
        count: 2,
    };
    let plan = plan_direct_run(run, 2, KvCacheStorage::Full, None).expect("representable");
    assert_eq!(plan.rows, 2..4);
    assert_eq!(plan.source_rows, 0..2);
    assert_eq!(plan.segments, [segment(1..3, 7..9, None)]);
    assert_eq!(
        plan_direct_run(run, 0, KvCacheStorage::SlidingRing, None),
        Err(DirectWriteViolation::WindowRequired)
    );
}

/// Commits check the proposal end, then capacity or width, prefix, and extent.
#[test]
fn commit_writes_reject_the_first_failed_check() {
    use KvCacheStorage::{Full, MirroredSlidingRing, SlidingRing};
    assert_eq!(
        commit_write_segments(Full, 4, 3, 2, None, 16),
        Ok(vec![segment(0..2, 4..6, None)])
    );
    assert_eq!(commit_write_segments(Full, 4, 3, 0, None, 16), Ok(vec![]));
    assert_eq!(
        commit_write_segments(SlidingRing, 3, 3, 3, Some(4), 16),
        Ok(vec![segment(0..1, 3..4, None), segment(1..3, 0..2, None)])
    );
    assert_eq!(
        commit_write_segments(Full, usize::MAX, 1, 0, None, 16),
        Err(DirectWriteViolation::Ring(
            RingPlanError::PositionOverflow {
                start_position: usize::MAX,
                token_count: 1,
            }
        ))
    );
    assert_eq!(
        commit_write_segments(Full, 15, 3, 2, None, 16),
        Err(DirectWriteViolation::WriteEnd(
            WriteEndViolation::ExceedsCapacity { end: 17 }
        ))
    );
    // Ring storage checks the width before the accepted prefix.
    assert_eq!(
        commit_write_segments(SlidingRing, 0, 1, 2, None, 16),
        Err(DirectWriteViolation::WindowRequired)
    );
    assert_eq!(
        commit_write_segments(MirroredSlidingRing, 0, 1, 2, Some(4), 16),
        Err(DirectWriteViolation::Ring(
            RingPlanError::CommitPrefixExceedsStaged {
                accepted: 2,
                staged: 1,
            }
        ))
    );
    let huge = usize::MAX / 2 + 1;
    assert_eq!(
        commit_write_segments(MirroredSlidingRing, 0, 1, 0, Some(huge), 16),
        Err(DirectWriteViolation::Ring(
            RingPlanError::DoubledCapacityOverflow { capacity: huge }
        ))
    );
}

/// Simulates writing `count` tokens (labelled by input column) into `slots` physical slots.
fn apply_segments(segments: &[DirectSegment], slots: usize) -> Vec<Option<usize>> {
    let mut storage = vec![None; slots];
    for segment in segments {
        let mut destinations = vec![segment.primary.clone()];
        destinations.extend(segment.mirror.clone());
        for destination in destinations {
            assert_eq!(destination.len(), segment.input.len());
            for (column, slot) in segment.input.clone().zip(destination) {
                storage[slot] = Some(column);
            }
        }
    }
    storage
}

proptest! {
    /// Ring and mirrored writes leave every slot holding its newest token.
    #[test]
    fn direct_ring_writes_match_sequential_simulation(
        start in 0_usize..40,
        count in 0_usize..12,
        column in 0_usize..5,
        width in 1_usize..7,
        mirrored in any::<bool>(),
    ) {
        let storage = if mirrored {
            KvCacheStorage::MirroredSlidingRing
        } else {
            KvCacheStorage::SlidingRing
        };
        let segments = direct_write_segments(storage, start, count, column, Some(width))
            .expect("representable");
        let slots = if mirrored { 2 * width } else { width };
        let written = apply_segments(&segments, slots);
        // Sequentially writing column `column + j` at position `start + j`.
        let mut expected = vec![None; slots];
        for j in 0..count {
            let slot = (start + j) % width;
            expected[slot] = Some(column + j);
            if mirrored {
                expected[slot + width] = Some(column + j);
            }
        }
        prop_assert_eq!(written, expected);
    }

    /// Commits plan exactly what the public staged-commit and mirrored planners plan.
    #[test]
    fn commit_writes_match_public_ring_planners(
        baseline in 0_usize..40,
        proposed in 0_usize..10,
        accepted in 0_usize..10,
        width in 1_usize..7,
        mirrored in any::<bool>(),
    ) {
        let window = core::num::NonZeroUsize::new(width).expect("positive");
        let storage = if mirrored {
            KvCacheStorage::MirroredSlidingRing
        } else {
            KvCacheStorage::SlidingRing
        };
        let planned = commit_write_segments(storage, baseline, proposed, accepted, Some(width), 64);
        let staging = crate::cache::plan_staged_ring_transaction(baseline, proposed)
            .expect("representable");
        let expected = staging.commit_prefix(accepted, window).map(|plan| {
            if mirrored {
                crate::cache::plan_mirrored_ring_write(baseline, accepted, window)
                    .expect("representable")
                    .into_iter()
                    .map(|part| segment(part.input(), part.primary(), Some(part.mirror())))
                    .collect::<Vec<_>>()
            } else {
                plan.segments()
                    .iter()
                    .map(|part| segment(part.input(), part.physical(), None))
                    .collect()
            }
        });
        prop_assert_eq!(planned, expected.map_err(DirectWriteViolation::Ring));
    }
}
