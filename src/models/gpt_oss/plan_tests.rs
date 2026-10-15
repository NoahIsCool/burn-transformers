//! Planning: the default schedule, cache-group numbering, and interleaved
//! expert offsets, with property tests over arbitrary inputs.

use proptest::prelude::*;

use super::*;

/// The default schedule matches the reference expression `bool((l + 1) % 2)`.
#[test]
fn default_layer_slides_exactly_on_even_layers() {
    for layer in 0..100 {
        assert_eq!(default_layer_slides(layer), (layer + 1) % 2 != 0, "{layer}");
    }
    assert!(!default_layer_slides(usize::MAX));
}

/// Alternating layers number each group densely in model order.
#[test]
fn cache_slots_number_alternating_layers() {
    let plan = cache_slots(&[true, false, true, false, true]);
    assert_eq!(plan.slots, [0, 0, 1, 1, 2]);
    assert_eq!((plan.full_layers, plan.ring_layers), (2, 3));
    let empty = cache_slots(&[]);
    assert!(empty.slots.is_empty());
    assert_eq!((empty.full_layers, empty.ring_layers), (0, 0));
}

/// Gate outputs are even columns and up outputs odd columns of their row.
#[test]
fn interleaved_offsets_select_even_gate_and_odd_up_columns() {
    // Two experts, three rows, width four: rows hold eight interleaved columns.
    for expert in 0..2 {
        for row in 0..3 {
            for column in 0..4 {
                let base = (expert * 3 + row) * 8;
                assert_eq!(
                    interleaved_offset(2, 3, 4, expert, row, column, GluHalf::Gate),
                    base + 2 * column
                );
                assert_eq!(
                    interleaved_offset(2, 3, 4, expert, row, column, GluHalf::Up),
                    base + 2 * column + 1
                );
            }
        }
    }
    assert_eq!(
        interleaved_offset(2, 3, 4, 1, 2, 3, GluHalf::Up),
        2 * 3 * 8 - 1
    );
}

proptest! {
    /// Slots count earlier layers of the same group, and the groups partition the layers.
    #[test]
    fn cache_slots_partition_any_schedule(ring in proptest::collection::vec(any::<bool>(), 0..40)) {
        let plan = cache_slots(&ring);
        prop_assert_eq!(plan.slots.len(), ring.len());
        prop_assert_eq!(plan.full_layers + plan.ring_layers, ring.len());
        for (layer, &in_ring) in ring.iter().enumerate() {
            let earlier = ring[..layer].iter().filter(|&&flag| flag == in_ring).count();
            prop_assert_eq!(plan.slots[layer], earlier);
            let size = if in_ring { plan.ring_layers } else { plan.full_layers };
            prop_assert!(plan.slots[layer] < size);
        }
    }

    /// Gate and up offsets of every output are distinct, in bounds, and cover each row.
    #[test]
    fn interleaved_offsets_tile_the_stacked_tensor(
        experts in 1_usize..4,
        rows in 1_usize..5,
        width in 1_usize..6,
    ) {
        let total = experts * rows * 2 * width;
        let mut seen = vec![false; total];
        for expert in 0..experts {
            for row in 0..rows {
                for column in 0..width {
                    for half in [GluHalf::Gate, GluHalf::Up] {
                        let offset = interleaved_offset(experts, rows, width, expert, row, column, half);
                        prop_assert!(offset < total);
                        prop_assert!(!seen[offset]);
                        seen[offset] = true;
                    }
                }
            }
        }
        prop_assert!(seen.into_iter().all(|covered| covered));
    }
}
