//! KV planning against hand-picked configurations and naive oracles.

use proptest::prelude::*;

use super::*;

/// Every storage mode reports its exact extent or its first invalid rule.
#[test]
fn committed_time_slots_cover_every_storage_rule() {
    use KvCacheStorage::{Full, MirroredSlidingRing, SlidingRing};
    assert_eq!(committed_time_slots(Full, false, None, Some(4), 16), Ok(16));
    assert_eq!(committed_time_slots(Full, true, None, None, 16), Ok(16));
    assert_eq!(
        committed_time_slots(SlidingRing, false, None, Some(4), 16),
        Ok(4)
    );
    assert_eq!(
        committed_time_slots(SlidingRing, false, Some(8), Some(4), 16),
        Ok(8)
    );
    assert_eq!(
        committed_time_slots(MirroredSlidingRing, false, None, Some(4), 16),
        Ok(8)
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
