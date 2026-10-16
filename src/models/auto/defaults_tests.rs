//! Hugging Face generation-default arithmetic: top-k, minimum length, and token lists.

use super::{
    HUGGING_FACE_DEFAULT_TOP_K, TokenIdError, canonical_token_ids, clamped_min_new_tokens,
    sampling_top_k,
};

/// Zero disables top-k, an unset value means 50, and every value is clamped to the vocabulary.
#[test]
fn sampling_top_k_follows_hugging_face() {
    assert_eq!(sampling_top_k(Some(0), 64), None);
    assert_eq!(sampling_top_k(None, 64), Some(HUGGING_FACE_DEFAULT_TOP_K));
    assert_eq!(sampling_top_k(None, 16), Some(16));
    assert_eq!(sampling_top_k(Some(9), 8), Some(8));
    assert_eq!(sampling_top_k(Some(usize::MAX), 3), Some(3));
    // An empty vocabulary has no valid cutoff; validation rejects it separately.
    assert_eq!(sampling_top_k(Some(4), 0), Some(0));
}

/// The minimum length is clamped to the length bound and lowered only when it exceeds it.
#[test]
fn min_new_tokens_are_clamped_to_the_length_bound() {
    assert_eq!(clamped_min_new_tokens(3, 8), 3);
    assert_eq!(clamped_min_new_tokens(8, 8), 8);
    assert_eq!(clamped_min_new_tokens(9, 8), 8);
    assert_eq!(clamped_min_new_tokens(0, 0), 0);
}

/// Token lists are checked in order and canonicalized to sorted distinct `u32` IDs.
#[test]
fn token_lists_are_checked_in_order_and_canonicalized() {
    assert_eq!(canonical_token_ids(&[], 8), Ok(Vec::new()));
    assert_eq!(canonical_token_ids(&[7, 2, 7, 0], 8), Ok(vec![0, 2, 7]));
    assert_eq!(
        canonical_token_ids(&[1, 9, 10], 9),
        Err(TokenIdError::OutsideVocabulary { id: 9 })
    );
    let oversized = usize::try_from(u32::MAX).expect("64-bit usize") + 1;
    assert_eq!(
        canonical_token_ids(&[1, oversized, usize::MAX], usize::MAX),
        Err(TokenIdError::ExceedsU32 { id: oversized })
    );
    // The vocabulary check precedes the `u32` check for the same ID.
    assert_eq!(
        canonical_token_ids(&[oversized], oversized),
        Err(TokenIdError::OutsideVocabulary { id: oversized })
    );
}

/// Long adversarial lists canonicalize in `O(n log n)`, keeping each distinct ID once.
#[test]
fn long_token_lists_canonicalize_quickly() {
    let ids = (0..200_000_usize)
        .rev()
        .chain(0..200_000)
        .collect::<Vec<_>>();
    let canonical = canonical_token_ids(&ids, 200_000).unwrap();
    assert_eq!(canonical.len(), 200_000);
    assert!(canonical.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!((canonical[0], canonical[199_999]), (0, 199_999));
}

proptest::proptest! {
    /// Canonical lists equal the former sort-and-deduplicate normalization.
    #[test]
    fn canonical_token_ids_match_sort_and_dedup(
        ids in proptest::collection::vec(0_usize..40, 0..12),
        vocabulary_size in 1_usize..40,
    ) {
        let expected = match ids.iter().find(|&&id| id >= vocabulary_size) {
            Some(&id) => Err(TokenIdError::OutsideVocabulary { id }),
            None => {
                let mut tokens = ids.iter().map(|&id| id as u32).collect::<Vec<_>>();
                tokens.sort_unstable();
                tokens.dedup();
                Ok(tokens)
            }
        };
        proptest::prop_assert_eq!(canonical_token_ids(&ids, vocabulary_size), expected);
    }

    /// The sampling top-k is disabled exactly by zero and otherwise lies in `1..=vocabulary`.
    #[test]
    fn sampling_top_k_is_disabled_by_zero_and_clamped_otherwise(
        top_k in proptest::option::of(0_usize..200),
        vocabulary_size in 1_usize..100,
    ) {
        let resolved = sampling_top_k(top_k, vocabulary_size);
        proptest::prop_assert_eq!(resolved.is_none(), top_k == Some(0));
        if let Some(resolved) = resolved {
            proptest::prop_assert!((1..=vocabulary_size).contains(&resolved));
            proptest::prop_assert_eq!(resolved, top_k.unwrap_or(50).min(vocabulary_size));
        }
    }
}
