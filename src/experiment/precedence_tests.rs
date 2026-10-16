//! Precedence tests against the iterator reference.

use super::*;

/// The last matching flag wins; no match yields `None`.
#[test]
fn the_last_matching_override_wins() {
    assert_eq!(last_match(&[]), None);
    assert_eq!(last_match(&[false, false]), None);
    assert_eq!(last_match(&[true, false, true, false]), Some(2));
    assert_eq!(last_match(&[true]), Some(0));
}

proptest::proptest! {
    /// `last_match` agrees with `iter().rposition`.
    #[test]
    fn last_match_matches_rposition(matches in proptest::collection::vec(proptest::bool::ANY, 0..12)) {
        proptest::prop_assert_eq!(last_match(&matches), matches.iter().rposition(|&matched| matched));
    }
}
