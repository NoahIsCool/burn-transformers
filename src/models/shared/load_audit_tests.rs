//! Load-audit counter arithmetic: exact sums, maxima, and doubled read counts.

use proptest::prelude::*;

use super::*;

/// Totals are exact sums with attained maxima; overflow is reported.
#[test]
fn payload_totals_sum_and_maximize_exactly() {
    assert_eq!(payload_totals(&[]), Some((0, 0)));
    assert_eq!(payload_totals(&[5]), Some((5, 5)));
    assert_eq!(payload_totals(&[3, 9, 0, 9]), Some((21, 9)));
    assert_eq!(payload_totals(&[u64::MAX, 0]), Some((u64::MAX, u64::MAX)));
    assert_eq!(payload_totals(&[u64::MAX, 1]), None);
}

/// Equivalence reads double the verified pair count and report overflow.
#[test]
fn equivalence_reads_double_the_pair_count() {
    assert_eq!(equivalence_reads(0), Some(0));
    assert_eq!(equivalence_reads(7), Some(14));
    let half = usize::try_from(u64::MAX / 2).expect("64-bit usize");
    assert_eq!(equivalence_reads(half), Some(u64::MAX - 1));
    assert_eq!(equivalence_reads(half + 1), None);
}

proptest! {
    /// Totals agree with checked iterator folds over the same records.
    #[test]
    fn payload_totals_match_iterator_folds(
        bytes in prop::collection::vec(prop_oneof![0_u64..1_000, u64::MAX - 10..=u64::MAX], 0..8),
    ) {
        let total = bytes.iter().try_fold(0_u64, |total, &value| total.checked_add(value));
        let largest = bytes.iter().copied().max().unwrap_or(0);
        prop_assert_eq!(payload_totals(&bytes), total.map(|total| (total, largest)));
    }
}
