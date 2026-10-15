//! Planning functions against direct reference formulations.
//!
//! The Verus harness proves these postconditions for all inputs; the tests
//! pin concrete values and cross-check each function against a naive
//! implementation of its specification.

use proptest::prelude::*;

use super::*;

/// Gemma 2 alternates from a sliding layer; Gemma 3's period makes every sixth layer global.
#[test]
fn patterns_match_the_published_schedules() {
    let alternate: Vec<bool> = (0..6).map(|layer| sliding_by_pattern(layer, 2)).collect();
    assert_eq!(alternate, [true, false, true, false, true, false]);
    let six: Vec<usize> = (0..34)
        .filter(|&layer| !sliding_by_pattern(layer, 6))
        .collect();
    assert_eq!(six, [5, 11, 17, 23, 29]);
    assert!(
        !sliding_by_pattern(0, 1),
        "pattern 1 makes every layer global"
    );
    // At the top of the range the reference's `l + 1` would overflow.
    assert!(sliding_by_pattern(usize::MAX - 1, usize::MAX - 1));
    assert!(
        !sliding_by_pattern(usize::MAX, 2),
        "2^64 is even, so layer 2^64 - 1 is global"
    );
}

/// Shared sets map every layer to zero; split sets map sliding to zero and global to one.
#[test]
fn rotary_sets_follow_the_layer_types() {
    assert_eq!(rotary_sets(&[true, false, true], false), [0, 1, 0]);
    assert_eq!(rotary_sets(&[true, false, true], true), [0, 0, 0]);
    assert!(rotary_sets(&[], false).is_empty());
}

proptest! {
    /// The overflow-free test equals the reference's `(l + 1) mod p != 0`.
    #[test]
    fn sliding_by_pattern_matches_the_reference_formula(
        layer in 0_usize..100_000,
        pattern in 1_usize..64,
    ) {
        prop_assert_eq!(sliding_by_pattern(layer, pattern), (layer + 1) % pattern != 0);
    }

    /// Every index is in range and names the layer type's set.
    #[test]
    fn rotary_sets_stay_in_range(
        sliding in proptest::collection::vec(any::<bool>(), 0..64),
        shared in any::<bool>(),
    ) {
        let sets = rotary_sets(&sliding, shared);
        prop_assert_eq!(sets.len(), sliding.len());
        let sets_used = if shared { 1 } else { 2 };
        for (set, sliding) in sets.into_iter().zip(sliding) {
            prop_assert!(set < sets_used);
            prop_assert_eq!(set, usize::from(!shared && !sliding));
        }
    }
}
