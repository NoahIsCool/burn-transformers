//! Hugging Face generation-default arithmetic: the top-k resolution rule.

use super::{HUGGING_FACE_DEFAULT_TOP_K, sampling_top_k};

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

proptest::proptest! {
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
