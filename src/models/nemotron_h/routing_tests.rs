//! Group-limited routing feasibility.

use proptest::prelude::*;

use super::*;

/// The released Lightning router and each infeasible configuration.
#[test]
fn routing_groups_accepts_feasible_configurations_only() {
    // 128 experts in one group, top-6 per token.
    assert_eq!(routing_groups(128, 1, 1, 6), Ok(128));
    assert_eq!(routing_groups(64, 8, 4, 8), Ok(32));
    assert_eq!(routing_groups(8, 0, 1, 1), Err(RoutingError::SmallGroups));
    assert_eq!(routing_groups(8, 8, 1, 1), Err(RoutingError::SmallGroups));
    assert_eq!(routing_groups(64, 8, 9, 8), Err(RoutingError::Insufficient));
    assert_eq!(
        routing_groups(64, 8, 4, 33),
        Err(RoutingError::Insufficient)
    );
    assert_eq!(
        routing_groups(usize::MAX, 1, 2, 1),
        Err(RoutingError::EligibleOverflow)
    );
}

proptest! {
    /// The check equals the configuration validator's former inline arithmetic.
    #[test]
    fn routing_groups_matches_the_inline_checks(
        experts in 0_usize..64,
        groups in 1_usize..9,
        top_groups in 0_usize..10,
        per_token in 0_usize..40,
    ) {
        let expected = if experts / groups < 2 {
            Err(RoutingError::SmallGroups)
        } else {
            match (experts / groups).checked_mul(top_groups) {
                None => Err(RoutingError::EligibleOverflow),
                Some(eligible) if top_groups > groups || per_token > eligible => {
                    Err(RoutingError::Insufficient)
                }
                Some(eligible) => Ok(eligible),
            }
        };
        prop_assert_eq!(routing_groups(experts, groups, top_groups, per_token), expected);
    }
}
