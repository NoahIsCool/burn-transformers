//! Muse layer-schedule and cache-slot planning against brute-force oracles.

use proptest::prelude::*;

use super::*;

/// The released 52-layer schedule has 13 global layers ending at the last layer.
#[test]
fn released_schedule_counts_backward_from_the_last_layer() {
    let global = (0..52)
        .map(|index| is_global_layer(52, index))
        .collect::<Vec<_>>();
    assert_eq!(global.iter().filter(|&&global| global).count(), 13);
    assert!(global[51] && global[3] && !global[50] && !global[0]);
    assert_eq!(group_slots(&global)[51], 12);
    assert_eq!(group_slots(&global)[50], 38);
    assert!(is_global_layer(1, 0));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Slots count earlier layers of the same kind, so each group is dense in layer order.
    #[test]
    fn group_slots_are_dense_per_kind(global in proptest::collection::vec(any::<bool>(), 0..40)) {
        let slots = group_slots(&global);
        prop_assert_eq!(slots.len(), global.len());
        for (index, &kind) in global.iter().enumerate() {
            let earlier = global[..index].iter().filter(|&&other| other == kind).count();
            prop_assert_eq!(slots[index], earlier);
        }
        for kind in [true, false] {
            let group = slots
                .iter()
                .zip(&global)
                .filter(|&(_, &other)| other == kind)
                .map(|(&slot, _)| slot)
                .collect::<Vec<_>>();
            prop_assert_eq!(group, (0..global.iter().filter(|&&other| other == kind).count()).collect::<Vec<_>>());
        }
    }
}

/// Hand-checked ring slices: whole ring before row `W - 1`, window unions after.
#[test]
fn ring_chunk_keys_read_history_only_while_visible() {
    // H = 3, W = 3: rows 0..8 still see history (rows 0 and 1 do).
    assert_eq!(ring_chunk_keys(3, 0, 8, 3), Some((0, 11)));
    // Rows 8..16 see fresh entries 6..16 only.
    assert_eq!(ring_chunk_keys(3, 8, 16, 3), Some((9, 19)));
    // Row 1 with W = 3 still sees one history slot; row 2 does not.
    assert_eq!(ring_chunk_keys(5, 1, 2, 3), Some((0, 7)));
    assert_eq!(ring_chunk_keys(5, 2, 3, 3), Some((5, 8)));
    // W = 1 sees only itself; an empty ring needs no history.
    assert_eq!(ring_chunk_keys(4, 0, 1, 1), Some((4, 5)));
    assert_eq!(ring_chunk_keys(0, 0, 4, 2), Some((0, 4)));
    assert_eq!(ring_chunk_keys(usize::MAX, 0, 1, 2), None);
}

/// Brute-force oracle of the keys block queries `start..end` may see over `[history | block]`.
///
/// Returns the smallest contiguous range containing every visible fresh
/// column `history + j` (`k - W < j <= k`) and, while some row `k + 1 < W`,
/// every ring slot (their order is physical, so all of them).
fn visible_hull(history: usize, start: usize, end: usize, window: usize) -> (usize, usize) {
    let mut needed = Vec::new();
    for k in start..end {
        if k + 1 < window {
            needed.extend(0..history);
        }
        needed.extend((k.saturating_sub(window - 1)..=k).map(|j| history + j));
    }
    (
        *needed.iter().min().unwrap(),
        needed.iter().max().unwrap() + 1,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Ring slices equal the hull of the visible keys and respect the planning bound.
    #[test]
    fn ring_chunk_keys_equal_the_visible_hull(
        history in 0usize..40,
        start in 0usize..40,
        rows in 1usize..20,
        window in 1usize..45,
    ) {
        let end = start + rows;
        let (lo, hi) = ring_chunk_keys(history, start, end, window).unwrap();
        // An empty ring with a history-seeing row starts at fresh entry 0 = column 0.
        prop_assert_eq!((lo, hi), visible_hull(history, start, end, window));
        prop_assert!(hi - lo < history + window + rows);
    }
}
