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
    assert_eq!(
        rotary_sets(&[true, false, true], false),
        [Some(0), Some(1), Some(0)]
    );
    assert_eq!(
        rotary_sets(&[true, false, true], true),
        [Some(0), Some(0), Some(0)]
    );
    assert!(rotary_sets(&[], false).is_empty());
    assert_eq!((rotary_set_count(true), rotary_set_count(false)), (1, 2));
}

/// The table holds one entry per set: sliding first, then global unless shared.
#[test]
fn rotary_set_tables_hold_each_type_at_its_index() {
    assert_eq!(rotary_set_table('s', 'g', false), ['s', 'g']);
    assert_eq!(rotary_set_table('s', 's', true), ['s']);
    for shared in [false, true] {
        let table = rotary_set_table(10, 20, shared);
        assert_eq!(table.len(), rotary_set_count(shared));
        for (sliding, set) in [true, false]
            .into_iter()
            .zip(rotary_sets(&[true, false], shared))
        {
            let expected = if sliding || shared { 10 } else { 20 };
            assert_eq!(table[set.unwrap()], expected);
        }
    }
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
        for (set, &sliding) in sets.iter().zip(&sliding) {
            let set = set.expect("every Gemma layer rotates");
            prop_assert!(set < rotary_set_count(shared));
            prop_assert_eq!(set, usize::from(!shared && !sliding));
        }
        for (i, first) in sets.iter().enumerate() {
            for (j, second) in sets.iter().enumerate() {
                prop_assert_eq!(first == second, shared || sliding[i] == sliding[j]);
            }
        }
    }
}

/// The default periods: Gemma 2 alternates, Gemma 3 reads its pattern (default 6).
#[test]
fn default_periods_follow_the_variant() {
    assert_eq!(default_period(true, Some(9)), 2);
    assert_eq!(default_period(false, None), 6);
    assert_eq!(default_period(false, Some(4)), 4);
}

/// Gemma 3's 26-layer default schedule has its four global layers at 5, 11, 17, 23.
#[test]
fn default_schedules_place_global_layers_at_period_ends() {
    let flags = sliding_schedule(26, 6);
    let global: Vec<usize> = (0..26).filter(|&layer| !flags[layer]).collect();
    assert_eq!(global, [5, 11, 17, 23]);
    assert_eq!(sliding_schedule(4, 2), [true, false, true, false]);
    assert!(sliding_schedule(0, 3).is_empty());
}

/// Exactly one set flag selects its index; none or several are reported.
#[test]
fn unique_flags_select_exactly_one_namespace() {
    assert_eq!(unique_flag(&[false, true, false]), UniqueFlag::Unique(1));
    assert_eq!(unique_flag(&[false, false, false]), UniqueFlag::Absent);
    assert_eq!(unique_flag(&[true, false, true]), UniqueFlag::Ambiguous);
    assert_eq!(unique_flag(&[]), UniqueFlag::Absent);
}

proptest! {
    /// Default schedules match the reference formula and have `⌊L / p⌋` global layers.
    #[test]
    fn default_schedules_count_their_global_layers(layers in 0usize..200, pattern in 1usize..40) {
        let flags = sliding_schedule(layers, pattern);
        prop_assert_eq!(flags.len(), layers);
        for (layer, &sliding) in flags.iter().enumerate() {
            prop_assert_eq!(sliding, (layer + 1) % pattern != 0);
        }
        prop_assert_eq!(flags.iter().filter(|&&sliding| !sliding).count(), layers / pattern);
    }

    /// The selection equals a direct count of set flags.
    #[test]
    fn unique_flags_match_a_count(found in proptest::collection::vec(any::<bool>(), 0..6)) {
        let set: Vec<usize> = (0..found.len()).filter(|&index| found[index]).collect();
        let expected = match set.as_slice() {
            [] => UniqueFlag::Absent,
            [index] => UniqueFlag::Unique(*index),
            _ => UniqueFlag::Ambiguous,
        };
        prop_assert_eq!(unique_flag(&found), expected);
    }
}

/// Projection widths are exact checkpoint extents, or report usize overflow.
#[test]
fn projection_extents_are_exact_or_report_overflow() {
    assert_eq!(projection_extent(8, 128), Ok(1024));
    assert_eq!(projection_extent(0, usize::MAX), Ok(0));
    assert_eq!(projection_extent(usize::MAX, 2), Err(ExtentError::Overflow));
    assert_eq!(
        checkpoint_extent(usize::MAX),
        u64::try_from(usize::MAX).ok()
    );
    assert_eq!(checkpoint_extent(7), Some(7));
}
