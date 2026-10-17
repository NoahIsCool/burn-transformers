//! Transition checks of the target claim ledger.

use super::*;

/// Each target accepts one claim; later claims collide and change nothing.
#[test]
fn targets_accept_exactly_one_claim() {
    let mut ledger = open_ledger(3);
    assert!(ledger.claim(0, TargetState::Primary));
    assert!(!ledger.claim(0, TargetState::Alias));
    assert!(!ledger.claim(0, TargetState::Primary));
    assert!(ledger.claim(1, TargetState::Alias));
    assert!(!ledger.claim(1, TargetState::Primary));
    assert_eq!(
        ledger.states,
        [TargetState::Primary, TargetState::Alias, TargetState::Open]
    );
    assert!(ledger.is_produced(0));
    assert!(!ledger.is_produced(1));
    assert!(!ledger.resolve(0));
    assert!(ledger.resolve(1));
    assert!(!ledger.resolve(1));
    assert!(ledger.is_produced(1));
    assert!(!ledger.is_produced(2));
    assert!(!ledger.resolve(2));
}
