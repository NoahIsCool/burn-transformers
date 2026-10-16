//! Extent and alignment checks of MTP speculation.

use proptest::prelude::*;

use super::*;

/// Appends end at `start + count` when it fits the limit and report each failure.
#[test]
fn bounded_end_reports_each_failure() {
    assert_eq!(bounded_end(3, 4, 7), Ok(7));
    assert_eq!(bounded_end(3, 4, 6), Err(ExtentError::Exceeds(7)));
    assert_eq!(
        bounded_end(usize::MAX, 1, usize::MAX),
        Err(ExtentError::Overflow)
    );
    assert_eq!(bounded_end(0, 0, 0), Ok(0));
}

/// Alignment requires one row in each bank with the MTP cursor one behind.
#[test]
fn shifted_alignment_requires_a_one_position_lag() {
    assert!(shifted_alignment(&[5], &[4]));
    assert!(!shifted_alignment(&[5], &[5]));
    assert!(!shifted_alignment(&[5], &[3]));
    assert!(!shifted_alignment(&[0], &[usize::MAX]));
    assert!(!shifted_alignment(&[5, 5], &[4, 4]));
    assert!(!shifted_alignment(&[], &[]));
    assert!(shifted_alignment(&[usize::MAX], &[usize::MAX - 1]));
}

proptest! {
    /// Both checks equal the checked arithmetic they replaced.
    #[test]
    fn checks_match_checked_arithmetic(
        start in prop_oneof![0_usize..16, Just(usize::MAX - 1), Just(usize::MAX)],
        count in prop_oneof![0_usize..16, Just(usize::MAX)],
        limit in prop_oneof![0_usize..32, Just(usize::MAX)],
        target in proptest::collection::vec(0_usize..6, 0..3),
        mtp in proptest::collection::vec(0_usize..6, 0..3),
    ) {
        let expected = match start.checked_add(count) {
            None => Err(ExtentError::Overflow),
            Some(end) if end > limit => Err(ExtentError::Exceeds(end)),
            Some(end) => Ok(end),
        };
        prop_assert_eq!(bounded_end(start, count, limit), expected);
        prop_assert_eq!(
            shifted_alignment(&target, &mtp),
            target.len() == 1 && mtp.len() == 1 && mtp[0].checked_add(1) == Some(target[0])
        );
    }
}

/// Trailing query positions are the last `q` positions before `end`.
#[test]
fn trailing_positions_end_at_the_step_end() {
    assert_eq!(trailing_positions(7, 3), Some(vec![4, 5, 6]));
    assert_eq!(trailing_positions(7, 0), Some(Vec::new()));
    assert_eq!(trailing_positions(2, 2), Some(vec![0, 1]));
    #[cfg(target_pointer_width = "64")]
    {
        let limit = i64::MAX as usize;
        assert_eq!(trailing_positions(limit + 1, 1), Some(vec![i64::MAX]));
        assert_eq!(trailing_positions(limit + 2, 1), None);
        assert_eq!(trailing_positions(limit + 2, 0), Some(Vec::new()));
    }
}

proptest! {
    /// Trailing positions equal the checked range conversion they replaced.
    #[test]
    fn trailing_positions_match_the_range_conversion(
        end in prop_oneof![0_usize..64, Just(i64::MAX as usize), Just(usize::MAX)],
        queries in 0_usize..8,
    ) {
        let queries = queries.min(end);
        let expected = (end - queries..end)
            .map(|position| i64::try_from(position).ok())
            .collect::<Option<Vec<_>>>();
        prop_assert_eq!(trailing_positions(end, queries), expected);
    }
}
