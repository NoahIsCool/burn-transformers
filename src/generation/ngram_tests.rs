//! Executable checks of the Verus-proved n-gram index arithmetic.
//!
//! The proofs run outside Cargo (see `verus/README.md`); these tests compare
//! each core with the expression it replaced in `history.rs`.

use proptest::prelude::*;

use super::{completed_ngram_prefix, insert_follower, ngram_lookup_start};

/// Prefixes and lookup keys appear exactly once a full n-gram fits.
#[test]
fn ngram_ranges_cover_size_boundaries() {
    assert_eq!(completed_ngram_prefix(0, 1), None);
    assert_eq!(completed_ngram_prefix(1, 1), Some((0, 0)));
    assert_eq!(completed_ngram_prefix(2, 3), None);
    assert_eq!(completed_ngram_prefix(5, 3), Some((2, 4)));
    assert_eq!(ngram_lookup_start(0, 1), Some(0));
    assert_eq!(ngram_lookup_start(0, 2), None);
    assert_eq!(ngram_lookup_start(1, 2), Some(0));
    assert_eq!(ngram_lookup_start(4, 3), Some(2));
    assert_eq!(ngram_lookup_start(usize::MAX, 1), Some(usize::MAX));
    assert_eq!(ngram_lookup_start(usize::MAX, usize::MAX), Some(1));
}

/// Followers stay sorted and deduplicated at the front, middle, and end.
#[test]
fn followers_insert_in_order_once() {
    let mut followers = Vec::new();
    for token in [5, 1, 9, 5, 3, 9, 0, u32::MAX] {
        insert_follower(&mut followers, token);
    }
    assert_eq!(followers, [0, 1, 3, 5, 9, u32::MAX]);
}

proptest! {
    /// N-gram ranges agree with the original inline arithmetic.
    #[test]
    fn ngram_ranges_match_reference(length in 0_usize..64, size in 1_usize..8) {
        let expected = (length >= size).then(|| (length - size, length - 1));
        prop_assert_eq!(completed_ngram_prefix(length, size), expected);
        let lookup = (length + 1 >= size).then(|| length + 1 - size);
        prop_assert_eq!(ngram_lookup_start(length, size), lookup);
        // The lookup key is the prefix the next token is recorded under.
        prop_assert_eq!(lookup, completed_ngram_prefix(length + 1, size).map(|(start, _)| start));
    }

    /// Follower insertion agrees with `binary_search` plus `insert`.
    #[test]
    fn follower_insertion_matches_binary_search(tokens in proptest::collection::vec(0_u32..20, 0..32)) {
        let mut followers = Vec::new();
        let mut expected: Vec<u32> = Vec::new();
        for token in tokens {
            insert_follower(&mut followers, token);
            if let Err(position) = expected.binary_search(&token) {
                expected.insert(position, token);
            }
            prop_assert_eq!(&followers, &expected);
        }
    }
}
