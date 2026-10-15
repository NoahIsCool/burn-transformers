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
