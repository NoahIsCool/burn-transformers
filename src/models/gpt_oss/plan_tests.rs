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

/// The default schedule alternates from a sliding layer and has `⌊L / 2⌋` full layers.
#[test]
fn default_schedules_alternate_from_a_sliding_layer() {
    assert_eq!(default_schedule(5), [true, false, true, false, true]);
    assert!(default_schedule(0).is_empty());
    for layers in 0..40 {
        let flags = default_schedule(layers);
        assert_eq!(flags.iter().filter(|&&slides| !slides).count(), layers / 2);
    }
}

/// Sliding layers join the ring only for a nonzero window below the capacity.
#[test]
fn ring_assignments_follow_the_window_and_capacity() {
    let sliding = [true, false, true];
    let ring = ring_assignment(&sliding, Some(4), 8);
    assert_eq!((ring.in_ring, ring.window), (vec![true, false, true], 4));
    for (window, capacity) in [(Some(8), 8), (Some(0), 8), (None, 8), (Some(4), 0)] {
        let full = ring_assignment(&sliding, window, capacity);
        assert_eq!((full.in_ring, full.window), (vec![false; 3], 0));
    }
}

/// Group indices, the clean-tail rule, and the geometry bound.
#[test]
fn group_indices_tails_and_expert_bounds() {
    let both = group_indices(1, 2);
    assert_eq!((both.full, both.ring), (Some(0), Some(1)));
    let ring_only = group_indices(0, 2);
    assert_eq!((ring_only.full, ring_only.ring), (None, Some(0)));
    assert!(truncation_leaves_bytes(&[4, 4], &[4, 3]));
    assert!(!truncation_leaves_bytes(&[4, 4], &[0, 4]));
    assert_eq!(fused_expert_elements(3, 5, 7), Some(210));
    assert_eq!(fused_expert_elements(usize::MAX / 2 + 1, 1, 1), None);
    assert_eq!(fused_expert_elements(1, usize::MAX, 1), None);
    assert_eq!(fused_expert_elements(1, 1, usize::MAX), None);
    assert_eq!(stacked_layout(2, 3, 4, 2), Some((24, 48)));
    assert_eq!(stacked_layout(2, usize::MAX, 2, 1), None);
    assert_eq!(stacked_layout(3, 1, 1, usize::MAX), None);
}

/// Expert slabs are contiguous; halves take even (gate) and odd (up) columns bytewise.
#[test]
fn expert_bytes_split_into_slabs_and_halves() {
    let bytes: Vec<u8> = (0..24).collect();
    assert_eq!(expert_slab(&bytes, 3, 8, 1), (8..16).collect::<Vec<u8>>());
    // Two experts, one row, width two, two-byte scalars: rows hold [g0 u0 g1 u1].
    let gate = deinterleave(&bytes[..16], 2, 1, 2, 2, 1, GluHalf::Gate);
    let up = deinterleave(&bytes[..16], 2, 1, 2, 2, 1, GluHalf::Up);
    assert_eq!(gate, [8, 9, 12, 13]);
    assert_eq!(up, [10, 11, 14, 15]);
    assert!(deinterleave(&[], 1, 3, 0, 4, 0, GluHalf::Up).is_empty());
}

proptest! {
    /// Each group's slots are an order-preserving bijection onto `0..size`.
    #[test]
    fn cache_slots_biject_each_group_in_order(ring in proptest::collection::vec(any::<bool>(), 0..24)) {
        let plan = cache_slots(&ring);
        for flag in [false, true] {
            let slots: Vec<usize> = ring
                .iter()
                .zip(&plan.slots)
                .filter(|&(&in_ring, _)| in_ring == flag)
                .map(|(_, &slot)| slot)
                .collect();
            let size = if flag { plan.ring_layers } else { plan.full_layers };
            prop_assert_eq!(slots, (0..size).collect::<Vec<_>>());
        }
    }

    /// De-interleaving equals a direct strided copy for any geometry and scalar width.
    #[test]
    fn deinterleave_matches_a_strided_copy(
        experts in 1_usize..4,
        rows in 1_usize..4,
        width in 1_usize..5,
        element in 1_usize..5,
    ) {
        let total = experts * rows * 2 * width * element;
        let bytes: Vec<u8> = (0..total).map(|index| (index * 7 % 251) as u8).collect();
        for expert in 0..experts {
            for half in [GluHalf::Gate, GluHalf::Up] {
                let out = deinterleave(&bytes, experts, rows, width, element, expert, half);
                let mut expected = Vec::new();
                for row in 0..rows {
                    for column in 0..width {
                        let offset = interleaved_offset(experts, rows, width, expert, row, column, half);
                        expected.extend_from_slice(&bytes[offset * element..(offset + 1) * element]);
                    }
                }
                prop_assert_eq!(out, expected);
            }
            let span = rows * 2 * width * element;
            prop_assert_eq!(
                expert_slab(&bytes, experts, span, expert),
                bytes[expert * span..(expert + 1) * span].to_vec()
            );
        }
    }
}
