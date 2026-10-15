//! Shared GPT dimension checks.

use proptest::prelude::*;

use super::*;

/// The first extent outside `1..=i32::MAX` is reported by position.
#[test]
fn first_invalid_extent_reports_the_first_offender() {
    assert_eq!(first_invalid_extent(&[1, 768, 12]), None);
    assert_eq!(first_invalid_extent(&[1, 0, 12]), Some(1));
    assert_eq!(first_invalid_extent(&[i32::MAX as usize, 0]), Some(1));
    assert_eq!(first_invalid_extent(&[i32::MAX as usize + 1, 0]), Some(0));
    assert_eq!(first_invalid_extent(&[]), None);
}

/// An explicit width wins; the default is `4 d`; zero widths are refused.
#[test]
fn intermediate_width_follows_the_gpt2_default() {
    assert_eq!(intermediate_width(Some(24), 12), Some(24));
    assert_eq!(intermediate_width(None, 12), Some(48));
    assert_eq!(intermediate_width(Some(0), 12), None);
    assert_eq!(intermediate_width(None, 0), None);
    assert_eq!(intermediate_width(None, usize::MAX), None);
    assert_eq!(intermediate_width(Some(7), usize::MAX), Some(7));
}

/// Products are accepted exactly when every running product fits.
#[test]
fn product_fits_follows_the_checked_chain() {
    assert!(product_fits(&[]));
    assert!(product_fits(&[768, 50_257]));
    assert!(!product_fits(&[usize::MAX, 2]));
    // An overflowing prefix fails even though a later zero would cancel it.
    assert!(!product_fits(&[usize::MAX, 2, 0]));
    assert!(product_fits(&[usize::MAX, 1, 0]));
}

proptest! {
    /// The checks equal the iterator and checked-arithmetic code they replaced.
    #[test]
    fn checks_match_the_former_code(
        values in proptest::collection::vec(
            prop_oneof![0_usize..4, Just(i32::MAX as usize), Just(i32::MAX as usize + 1)],
            0..5,
        ),
        explicit in proptest::option::of(prop_oneof![0_usize..9, Just(usize::MAX)]),
        width in prop_oneof![0_usize..9, Just(usize::MAX / 4), Just(usize::MAX / 4 + 1)],
        factors in proptest::collection::vec(
            prop_oneof![0_usize..5, Just(usize::MAX / 3), Just(usize::MAX)],
            0..4,
        ),
    ) {
        prop_assert_eq!(
            first_invalid_extent(&values),
            values.iter().position(|&value| value == 0 || value > i32::MAX as usize)
        );
        prop_assert_eq!(
            intermediate_width(explicit, width),
            explicit.or_else(|| width.checked_mul(4)).filter(|inner| *inner != 0)
        );
        let chained = factors
            .iter()
            .try_fold(1_usize, |product, &factor| product.checked_mul(factor))
            .is_some();
        prop_assert_eq!(product_fits(&factors), chained);
    }
}
