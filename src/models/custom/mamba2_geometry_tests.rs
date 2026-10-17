//! Mamba-2 head grouping: exact division or rejection.

use super::mamba2_groups;

/// Divisible groupings translate exactly; zero or non-dividing group sizes are rejected.
#[test]
fn groups_divide_heads_exactly_or_are_rejected() {
    assert_eq!(mamba2_groups(4, 2), Some(2));
    assert_eq!(mamba2_groups(4, 1), Some(4));
    assert_eq!(mamba2_groups(4, 4), Some(1));
    assert_eq!(mamba2_groups(6, 4), None);
    assert_eq!(mamba2_groups(4, 0), None);
    assert_eq!(mamba2_groups(0, 3), Some(0));
}

proptest::proptest! {
    /// A grouping exists exactly for positive divisors, and then multiplies back to the heads.
    #[test]
    fn groups_multiply_back_to_the_heads(heads in 0_usize..512, group_size in 0_usize..64) {
        let groups = mamba2_groups(heads, group_size);
        proptest::prop_assert_eq!(
            groups.is_some(),
            group_size > 0 && heads % group_size == 0
        );
        if let Some(groups) = groups {
            proptest::prop_assert_eq!(groups * group_size, heads);
        }
    }
}
