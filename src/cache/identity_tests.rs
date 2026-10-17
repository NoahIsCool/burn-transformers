//! Identity tokens equal exactly their own clones.

use super::IdentityToken;

/// Clones share an identity; independently created tokens never do.
#[test]
fn tokens_equal_only_their_clones() {
    let first = IdentityToken::new();
    let clone = first.clone();
    let second = IdentityToken::new();
    assert_eq!(first, clone);
    assert_ne!(first, second);
    assert_ne!(clone, second);
}

/// A token created after another is dropped still differs from live tokens.
#[test]
fn fresh_tokens_differ_from_every_live_token() {
    let live: Vec<IdentityToken> = (0..64).map(|_| IdentityToken::new()).collect();
    drop(IdentityToken::new());
    let fresh = IdentityToken::new();
    assert!(live.iter().all(|token| *token != fresh));
}

/// Debug output hides the allocation address.
#[test]
fn debug_output_is_address_free() {
    assert_eq!(format!("{:?}", IdentityToken::new()), "IdentityToken(..)");
}
