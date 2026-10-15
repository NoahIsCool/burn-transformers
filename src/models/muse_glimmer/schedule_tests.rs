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

/// The derived schedule equals the per-layer rule and holds `ceil(n / 4)` global layers.
#[test]
fn derived_schedule_has_one_global_layer_per_four() {
    assert_eq!(derived_global_layers(5), [true, false, false, false, true]);
    assert!(derived_global_layers(0).is_empty());
    for layers in 0..60 {
        let global = derived_global_layers(layers);
        let reference = (0..layers)
            .map(|index| is_global_layer(layers, index))
            .collect::<Vec<_>>();
        assert_eq!(global, reference);
        assert_eq!(
            global.iter().filter(|&&global| global).count(),
            layers.div_ceil(4)
        );
    }
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

/// Planning bounds clamp to `K` and saturate instead of wrapping.
#[test]
fn planning_bounds_follow_each_rule() {
    assert_eq!(
        planning_bound(KeyRule::AlignedWindow { window: 4 }, 16, 16, 3),
        6
    );
    assert_eq!(
        planning_bound(KeyRule::AlignedWindow { window: 40 }, 16, 16, 3),
        16
    );
    assert_eq!(
        planning_bound(KeyRule::AlignedWindow { window: 4 }, 16, 16, 0),
        4
    );
    assert_eq!(planning_bound(KeyRule::AlignedPrefix, 4, 9, 2), 9);
    assert_eq!(planning_bound(KeyRule::Limits, 4, 9, 2), 9);
    // Ring: (K - Q) + W - 1 + q = 5 + 2 + 3.
    assert_eq!(planning_bound(KeyRule::Ring { window: 3 }, 8, 13, 3), 10);
    assert_eq!(
        planning_bound(KeyRule::Ring { window: usize::MAX }, 1, 7, usize::MAX),
        7
    );
}

/// Prefix limits, plane products, and cache groups match their definitions.
#[test]
fn chunk_limits_planes_and_groups_are_exact() {
    assert_eq!(prefix_chunk_limit(&[2, 3, 5], 3, 5, 2), Some(3));
    assert_eq!(prefix_chunk_limit(&[2, 3, 5], 3, 4, 3), None);
    assert_eq!(prefix_chunk_limit(&[0, 3, 5], 3, 5, 1), None);
    assert_eq!(prefix_chunk_limit(&[2, 3], 3, 5, 2), None);
    assert_eq!(prefix_chunk_limit(&[2, 3], 2, 5, 3), None);
    assert_eq!(plane_elements(2, 4, 3, 5), Some(120));
    assert_eq!(plane_elements(usize::MAX, 2, 0, 1), None);
    assert_eq!(plane_elements(1, 1, usize::MAX, 2), None);

    let plan = cache_groups(&[false, true, false, true], Some(8)).unwrap();
    assert_eq!((plan.sliding_layers, plan.full_layers), (2, 2));
    assert_eq!(
        (plan.window, plan.sliding_group, plan.full_group),
        (Some(8), Some(0), Some(1))
    );
    assert_eq!(plan.groups, [0, 1, 0, 1]);
    assert_eq!(plan.slots, [0, 0, 1, 1]);
    let full_only = cache_groups(&[true, true], None).unwrap();
    assert_eq!(
        (
            full_only.window,
            full_only.sliding_group,
            full_only.full_group
        ),
        (None, None, Some(0))
    );
    assert_eq!(full_only.groups, [0, 0]);
    let local_only = cache_groups(&[false], Some(2)).unwrap();
    assert_eq!(
        (local_only.sliding_group, local_only.full_group),
        (Some(0), None)
    );
    assert_eq!(
        cache_groups(&[false, true], Some(0)),
        Err(GroupFault::ZeroWindow)
    );
    assert_eq!(
        cache_groups(&[true, false], None),
        Err(GroupFault::MissingWindow)
    );
    assert_eq!(cache_groups(&[], None).unwrap().groups, Vec::<usize>::new());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// No ring slice of an admitted chunk exceeds the ring planning bound.
    #[test]
    fn ring_slices_respect_the_planning_bound(
        history in 0usize..40,
        queries in 1usize..30,
        window in 1usize..45,
        maximum_query in 1usize..30,
        start_seed in 0usize..30,
        rows_seed in 0usize..30,
    ) {
        let keys = history + queries;
        let start = start_seed % queries;
        let rows = 1 + rows_seed % maximum_query.min(queries - start);
        let (lo, hi) = ring_chunk_keys(history, start, start + rows, window).unwrap();
        let bound = planning_bound(KeyRule::Ring { window }, queries, keys, maximum_query);
        prop_assert!(hi - lo <= bound);
    }

    /// Plane products equal the checked fold they replace.
    #[test]
    fn plane_elements_match_the_checked_fold(
        shape in proptest::array::uniform4(prop_oneof![0usize..9, Just(usize::MAX / 3)]),
    ) {
        let fold = shape.into_iter().try_fold(1usize, usize::checked_mul);
        prop_assert_eq!(plane_elements(shape[0], shape[1], shape[2], shape[3]), fold);
    }
}
