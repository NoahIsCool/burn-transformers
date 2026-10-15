//! Integer-bound tests against exact 128-bit reference arithmetic.

use super::*;

/// Positional extents are `high + 1` exactly when representable.
#[test]
fn extents_follow_the_maximum_position() {
    assert_eq!(extent_after(0), Some(1));
    assert_eq!(extent_after(41), Some(42));
    assert_eq!(extent_after(-1), None);
    assert_eq!(extent_after(i64::MIN), None);
    if usize::BITS <= 63 {
        assert_eq!(extent_after(i64::MAX), None);
    } else {
        assert_eq!(extent_after(i64::MAX), Some(i64::MAX as usize + 1));
    }
}

/// Broadcasting accepts singleton or equal axes of equal rank only.
#[test]
fn broadcast_compatibility_is_axiswise() {
    assert!(broadcasts_to(&[1, 3, 1, 5], &[2, 3, 4, 5]));
    assert!(broadcasts_to(&[], &[]));
    assert!(!broadcasts_to(&[2, 3], &[4, 3]));
    assert!(!broadcasts_to(&[1, 3], &[1, 3, 1]));
    assert!(!broadcasts_to(&[0], &[1]));
    assert!(broadcasts_to(&[0], &[0]));
}

/// Staged ranges must resume at the cursor and fit the stack; consecutive ranges tile it.
#[test]
fn staged_ranges_resume_at_the_cursor() {
    assert!(follows_cursor(0, 0, 0, 3));
    assert!(follows_cursor(2, 2, 3, 3));
    assert!(!follows_cursor(2, 1, 3, 3), "repeat");
    assert!(!follows_cursor(1, 2, 3, 3), "skip");
    assert!(!follows_cursor(2, 2, 4, 3), "overrun");
    assert!(!follows_cursor(2, 3, 2, 3), "reversed");
    let mut cursor = 0;
    for end in [1, 1, 3, 4] {
        assert!(follows_cursor(cursor, cursor, end, 4));
        cursor = end;
    }
    assert_eq!(cursor, 4);
}

/// Fold levels halve, carry exactly the odd middle column, and strictly shrink.
#[test]
fn fold_levels_halve_and_carry_odd_columns() {
    let level = |pairs, partner_start, next| FoldLevel {
        pairs,
        partner_start,
        next,
    };
    assert_eq!(fold_level(2), level(1, 1, 1));
    assert_eq!(fold_level(3), level(1, 2, 2));
    assert_eq!(fold_level(151_936), level(75_968, 75_968, 75_968));
    assert_eq!(
        fold_level(usize::MAX),
        level(usize::MAX / 2, usize::MAX / 2 + 1, usize::MAX / 2 + 1)
    );
    let mut width = 151_937;
    let mut levels = 0;
    while width > 1 {
        let level = fold_level(width);
        assert_eq!(2 * level.pairs + width % 2, width);
        assert_eq!(level.partner_start + level.pairs, width);
        assert!(level.next >= 1 && level.next < width);
        width = level.next;
        levels += 1;
    }
    // ceil(log2(151_937)) = 18 levels.
    assert_eq!(levels, 18);
}

proptest::proptest! {
    /// Extents and range checks agree with exact 128-bit integer arithmetic.
    #[test]
    fn bounds_match_a_wide_integer_reference(
        low in proptest::num::i64::ANY,
        high in proptest::num::i64::ANY,
        extent in proptest::num::usize::ANY,
    ) {
        let expected_extent = (high >= 0 && i128::from(high) < usize::MAX as i128)
            .then(|| (i128::from(high) + 1) as usize);
        proptest::prop_assert_eq!(extent_after(high), expected_extent);
        let expected_within = low >= 0 && low <= high && i128::from(high) < extent as i128;
        proptest::prop_assert_eq!(range_within(low, high, extent), expected_within);
    }

    /// Broadcast acceptance matches its axiswise definition for random rank-4 shapes.
    #[test]
    fn broadcast_matches_its_definition(
        actual in proptest::array::uniform4(0usize..4),
        expected in proptest::array::uniform4(0usize..4),
    ) {
        let reference = actual.iter().zip(expected).all(|(a, e)| *a == 1 || *a == e);
        proptest::prop_assert_eq!(broadcasts_to(&actual, &expected), reference);
    }

    /// Every column of a folding level is read by exactly one block, and the depth is ceil(log2 w).
    #[test]
    fn fold_levels_partition_columns(width in 2usize..300) {
        let level = fold_level(width);
        let mut reads = vec![0usize; width];
        for j in 0..level.pairs {
            reads[j] += 1;
            reads[level.partner_start + j] += 1;
        }
        if level.next != level.pairs {
            reads[level.pairs] += 1;
        }
        proptest::prop_assert!(reads.iter().all(|&count| count == 1));
        let mut depth = 0;
        let mut current = width;
        while current > 1 {
            current = fold_level(current).next;
            depth += 1;
        }
        proptest::prop_assert_eq!(depth, width.next_power_of_two().trailing_zeros());
    }
}
