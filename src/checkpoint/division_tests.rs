//! Edge-value and property checks for the shared rounded-up division.

use proptest::prelude::*;

use super::*;

/// Ceilings match the overflow-free `div_ceil` on edge values.
#[test]
fn ceilings_match_div_ceil_at_the_edges() {
    for value in [0_u64, 1, 2, 31, 32, 33, u64::MAX - 1, u64::MAX] {
        for divisor in [1_u64, 2, 32, 64, 128, u64::MAX] {
            assert_eq!(ceil_div_u64(value, divisor), value.div_ceil(divisor));
            if let (Ok(value), Ok(divisor)) = (usize::try_from(value), usize::try_from(divisor)) {
                assert_eq!(ceil_div_usize(value, divisor), value.div_ceil(divisor));
            }
        }
    }
}

proptest! {
    /// Ceilings agree with `div_ceil` and are the least sufficient multiple count.
    #[test]
    fn ceilings_are_the_least_covering_quotient(value in any::<u64>(), divisor in 1_u64..) {
        let quotient = ceil_div_u64(value, divisor);
        prop_assert_eq!(quotient, value.div_ceil(divisor));
        prop_assert!(quotient <= value);
        prop_assert!(u128::from(value) <= u128::from(quotient) * u128::from(divisor));
        if quotient > 0 {
            prop_assert!(u128::from(quotient - 1) * u128::from(divisor) < u128::from(value));
        }
        if let (Ok(value), Ok(divisor)) = (usize::try_from(value), usize::try_from(divisor)) {
            prop_assert_eq!(ceil_div_usize(value, divisor), value.div_ceil(divisor));
        }
    }
}
