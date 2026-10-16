//! Planning functions against direct reference formulations.
//!
//! The Verus harness proves these postconditions for all inputs; the tests
//! pin concrete values and cross-check each function against a naive
//! implementation of its specification.

use proptest::prelude::*;

use super::*;

/// Layers are numbered densely inside their own group.
#[test]
fn cache_slots_number_each_group_densely() {
    let plan = cache_slots(&[false, true, true, false, true]);
    assert_eq!(plan.slots, [0, 0, 1, 1, 2]);
    assert_eq!((plan.full_layers, plan.ring_layers), (2, 3));
    let empty = cache_slots(&[]);
    assert!(empty.slots.is_empty());
    assert_eq!((empty.full_layers, empty.ring_layers), (0, 0));
}

/// Slot positions continue the cache inside the run and fill elsewhere.
#[test]
fn slot_positions_continue_the_cache_and_fill_padding() {
    assert_eq!(slot_positions(5, 1, 2, 4), Some(vec![6, 5, 6, 6]));
    assert_eq!(slot_positions(5, 0, 0, 2), Some(vec![4, 4]));
    assert_eq!(slot_positions(0, 0, 0, 3), Some(vec![0, 0, 0]));
    assert_eq!(slot_positions(usize::MAX, 0, 1, 1), None);
}

/// Compaction gathers valid columns and repeats the first one as padding.
#[test]
fn compaction_repeats_the_first_valid_column() {
    assert_eq!(compaction_sources(&[1, 3], 4), [1, 3, 1, 1]);
    assert_eq!(compaction_sources(&[], 2), [0, 0]);
}

/// Scattering maps valid columns to their compact slot and padding to the zero slot.
#[test]
fn scatter_restores_valid_columns_and_zeroes_padding() {
    assert_eq!(scatter_sources(&[1, 3], 5, 2), [2, 0, 2, 1, 2]);
    assert_eq!(scatter_sources(&[], 3, 4), [4, 4, 4]);
}

/// Runs are recognized by their span; gaps and the empty row are classified exactly.
#[test]
fn runs_are_recognized_by_their_span() {
    assert!(is_one_run(&[]));
    assert!(is_one_run(&[4]));
    assert!(is_one_run(&[2, 3, 4]));
    assert!(!is_one_run(&[2, 4]));
    assert!(is_one_run(&[usize::MAX - 1, usize::MAX]));
}

/// Restoration offsets compact indices by the run's first slot.
#[test]
fn restore_slots_offset_the_compact_indices() {
    assert_eq!(restore_slots(&[1, 3], 5, 4, 2), [4, 2, 4, 3, 4]);
    assert_eq!(restore_slots(&[], 2, 0, 0), [0, 0]);
}

/// Strictly increasing columns below `bound`.
fn sorted_columns(bound: usize) -> impl Strategy<Value = Vec<usize>> {
    prop::collection::btree_set(0..bound, 0..=bound).prop_map(|set| set.into_iter().collect())
}

proptest! {
    /// Cache slots equal per-group prefix counts.
    #[test]
    fn cache_slots_equal_prefix_counts(ring in prop::collection::vec(any::<bool>(), 0..24)) {
        let plan = cache_slots(&ring);
        for (layer, &in_ring) in ring.iter().enumerate() {
            let earlier = ring[..layer].iter().filter(|&&other| other == in_ring).count();
            prop_assert_eq!(plan.slots[layer], earlier);
        }
        prop_assert_eq!(plan.ring_layers, ring.iter().filter(|&&flag| flag).count());
        prop_assert_eq!(plan.full_layers + plan.ring_layers, ring.len());
    }

    /// Slot positions equal the module's definition slot by slot.
    #[test]
    fn slot_positions_match_the_definition(
        (start, block, column, count) in (0usize..1000, 0usize..12).prop_flat_map(|(start, block)| {
            (Just(start), Just(block), 0..=block).prop_flat_map(|(start, block, column)| {
                (Just(start), Just(block), Just(column), 0..=block - column)
            })
        }),
    ) {
        let positions = slot_positions(start, column, count, block).unwrap();
        let fill = if count > 0 { start + count - 1 } else { start.saturating_sub(1) };
        for (slot, &position) in positions.iter().enumerate() {
            let expected = if (column..column + count).contains(&slot) {
                start + slot - column
            } else {
                fill
            };
            prop_assert_eq!(position, expected);
        }
    }

    /// The run test equals a direct check that consecutive columns differ by one.
    #[test]
    fn run_test_matches_consecutive_differences(columns in sorted_columns(12)) {
        let consecutive = columns.windows(2).all(|pair| pair[1] == pair[0] + 1);
        prop_assert_eq!(is_one_run(&columns), consecutive);
    }

    /// Restoration reads each valid column's own slot and the zero slot elsewhere.
    #[test]
    fn restore_slots_match_the_definition(columns in sorted_columns(10), offset in 0usize..3) {
        let block = columns.len() + offset + 1;
        let slots = restore_slots(&columns, 10, block, offset);
        for (column, &slot) in slots.iter().enumerate() {
            let expected = columns
                .iter()
                .position(|&valid| valid == column)
                .map_or(block, |index| offset + index);
            prop_assert_eq!(slot, expected);
        }
    }

    /// Compaction followed by scattering is the identity on valid columns.
    #[test]
    fn compaction_and_scatter_are_inverse_on_valid_columns(
        columns in sorted_columns(10),
        extra in 0usize..3,
    ) {
        let block = columns.len() + extra;
        let sources = compaction_sources(&columns, block);
        let back = scatter_sources(&columns, 10, block);
        for (column, &slot) in back.iter().enumerate() {
            match columns.iter().position(|&valid| valid == column) {
                Some(index) => {
                    prop_assert_eq!(slot, index);
                    prop_assert_eq!(sources[slot], column);
                }
                None => prop_assert_eq!(slot, block),
            }
        }
    }
}

/// The SmolLM3 and Qwen schedule rules match their reference formulas.
#[test]
fn schedule_rules_match_the_reference_formulas() {
    let nope: Vec<bool> = (0..8).map(|layer| rotary_by_interval(layer, 4)).collect();
    assert_eq!(nope, [true, true, true, false, true, true, true, false]);
    assert!(rotary_by_interval(0, 2) && !rotary_by_interval(1, 1));
    assert!(sliding_from_layer(28, 28) && !sliding_from_layer(27, 28));
}

/// Whole schedules: SmolLM3's NoPE interval, explicit lists, Qwen suffixes, and NoPE sliding.
#[test]
fn whole_schedules_follow_their_rules() {
    assert_eq!(
        interval_rotary_flags(8, 4),
        [true, true, true, false, true, true, true, false]
    );
    assert!(interval_rotary_flags(0, 3).is_empty());
    assert_eq!(
        listed_rotary_flags(&[1, 0, 1]),
        Some(vec![true, false, true])
    );
    assert_eq!(listed_rotary_flags(&[]), Some(Vec::new()));
    assert_eq!(listed_rotary_flags(&[1, 2, 0]), None);
    assert_eq!(suffix_sliding_flags(4, 2, true), [false, false, true, true]);
    assert_eq!(suffix_sliding_flags(3, 0, false), [false, false, false]);
    assert_eq!(
        nope_sliding_flags(&[true, false, true, false], true),
        [false, true, false, true]
    );
    assert_eq!(nope_sliding_flags(&[false], false), [false]);
}

/// The first set flag is found; an all-clear list has none.
#[test]
fn first_set_finds_the_earliest_flag() {
    assert_eq!(first_set(&[false, true, true]), Some(1));
    assert_eq!(first_set(&[false, false]), None);
    assert_eq!(first_set(&[]), None);
}

/// The derived head width divides, and is zero without heads.
#[test]
fn derived_head_dim_divides_or_is_zero() {
    assert_eq!(derived_head_dim(4096, 32), 128);
    assert_eq!(derived_head_dim(10, 3), 3);
    assert_eq!(derived_head_dim(64, 0), 0);
}

proptest! {
    /// NoPE schedules match the reference formula and count `⌊L / k⌋` NoPE layers.
    #[test]
    fn interval_schedules_count_their_nope_layers(layers in 0usize..200, interval in 1usize..40) {
        let flags = interval_rotary_flags(layers, interval);
        prop_assert_eq!(flags.len(), layers);
        for (layer, &rotates) in flags.iter().enumerate() {
            prop_assert_eq!(rotates, (layer + 1) % interval != 0);
        }
        prop_assert_eq!(flags.iter().filter(|&&rotates| !rotates).count(), layers / interval);
    }

    /// Explicit lists are accepted exactly when every entry is 0 or 1.
    #[test]
    fn listed_flags_accept_exactly_binary_lists(listed in prop::collection::vec(0usize..3, 0..12)) {
        let parsed = listed_rotary_flags(&listed);
        if listed.iter().all(|&flag| flag <= 1) {
            let expected: Vec<bool> = listed.iter().map(|&flag| flag == 1).collect();
            prop_assert_eq!(parsed, Some(expected));
        } else {
            prop_assert_eq!(parsed, None);
        }
    }
}

/// Ring planning: windows below the capacity share the ring; conflicts and chunks are rejected.
#[test]
fn ring_plans_assign_narrow_windows_and_reject_conflicts() {
    use LayerReach::{Chunked, Global, Sliding};
    let plan = plan_ring(&[Global, Sliding(4), Sliding(64), Sliding(4)], 16).unwrap();
    assert_eq!(plan.in_ring, [false, true, false, true]);
    assert_eq!(plan.window, 4);
    let none = plan_ring(&[Global, Sliding(0), Sliding(16)], 16).unwrap();
    assert_eq!(none.in_ring, [false, false, false]);
    assert_eq!(none.window, 0);
    assert_eq!(
        plan_ring(&[Sliding(4), Global, Sliding(8)], 16),
        Err(RingPlanError::Conflict {
            layer: 2,
            window: 8,
            earlier: 4
        })
    );
    assert_eq!(
        plan_ring(&[Global, Chunked, Sliding(8), Sliding(4)], 16),
        Err(RingPlanError::Chunked { layer: 1 })
    );
    assert_eq!(
        plan_ring(&[Sliding(4), Sliding(8), Chunked], 16),
        Err(RingPlanError::Conflict {
            layer: 1,
            window: 8,
            earlier: 4
        })
    );
}

/// Allocated groups are numbered densely, full storage first.
#[test]
fn group_indices_number_allocated_groups_densely() {
    let both = group_indices(2, 3);
    assert_eq!((both.full, both.ring), (Some(0), Some(1)));
    let ring_only = group_indices(0, 3);
    assert_eq!((ring_only.full, ring_only.ring), (None, Some(0)));
    let full_only = group_indices(2, 0);
    assert_eq!((full_only.full, full_only.ring), (Some(0), None));
    let none = group_indices(0, 0);
    assert_eq!((none.full, none.ring), (None, None));
}

/// Only a truncation to a shorter nonzero length leaves stale bytes.
#[test]
fn truncation_leaves_bytes_only_when_shortening_to_nonzero() {
    assert!(truncation_leaves_bytes(&[5, 3], &[5, 2]));
    assert!(!truncation_leaves_bytes(&[5, 3], &[0, 3]));
    assert!(!truncation_leaves_bytes(&[5, 3], &[5, 4]));
    assert!(!truncation_leaves_bytes(&[5], &[]));
    assert!(truncation_leaves_bytes(&[5, 3], &[4]));
}

proptest! {
    /// Ring plans equal a direct statement of the rule.
    #[test]
    fn ring_plans_match_the_rule(
        reach in prop::collection::vec(
            prop_oneof![
                Just(LayerReach::Global),
                (0usize..6).prop_map(LayerReach::Sliding),
                Just(LayerReach::Chunked),
            ],
            0..10,
        ),
        capacity in 0usize..8,
    ) {
        let member = |reach: &LayerReach| matches!(*reach, LayerReach::Sliding(w) if w > 0 && w < capacity);
        let mut windows = reach.iter().filter(|r| member(r)).map(|r| match r {
            LayerReach::Sliding(w) => *w,
            _ => unreachable!(),
        });
        let first = windows.next();
        let storable = !reach.contains(&LayerReach::Chunked)
            && windows.all(|w| Some(w) == first);
        match plan_ring(&reach, capacity) {
            Ok(plan) => {
                prop_assert!(storable);
                prop_assert_eq!(plan.in_ring, reach.iter().map(member).collect::<Vec<_>>());
                prop_assert_eq!(plan.window, first.unwrap_or(0));
            }
            Err(_) => prop_assert!(!storable),
        }
    }

    /// Each group's slots are an order-preserving bijection onto `0..size`.
    #[test]
    fn cache_slots_biject_each_group_in_order(ring in prop::collection::vec(any::<bool>(), 0..24)) {
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
}

/// Runs are sliced from their narrowest common window; interior padding is gathered.
#[test]
fn step_plans_slice_runs_and_gather_interior_padding() {
    let plan = plan_step(&[vec![2, 3], vec![1, 2]], &[5, 0], 4).unwrap();
    assert_eq!(plan.offset, Some(1));
    assert_eq!(plan.block, 3);
    assert_eq!(plan.columns, [1, 0]);
    assert_eq!(plan.rows[0].positions, [6, 5, 6]);
    assert_eq!(plan.rows[0].valid, [false, true, true]);
    assert_eq!(plan.rows[0].restore, [3, 3, 1, 2]);
    assert!(plan.rows[0].sources.is_empty());
    assert_eq!((plan.rows[0].last, plan.rows[1].last), (2, 1));

    let gathered = plan_step(&[vec![0, 2], vec![], vec![3]], &[1, 4, 0], 4).unwrap();
    assert_eq!(gathered.offset, None);
    assert_eq!(gathered.block, 2);
    assert_eq!(gathered.columns, [0, 0, 0]);
    assert_eq!(gathered.rows[0].sources, [0, 2]);
    assert_eq!(gathered.rows[1].sources, [0, 0]);
    assert_eq!(gathered.rows[1].positions, [3, 3]);
    assert_eq!(gathered.rows[1].valid, [false, false]);
    assert_eq!(gathered.rows[2].sources, [3, 3]);
    assert_eq!(gathered.rows[2].restore, [2, 2, 2, 0]);
}

/// Steps without active rows and positions beyond `usize` or `i64` are rejected.
#[test]
fn step_plans_reject_idle_batches_and_overflowing_positions() {
    assert_eq!(
        plan_step(&[vec![], vec![]], &[3, 0], 2),
        Err(StepPlanError::NoActiveRow)
    );
    assert_eq!(
        plan_step(&[vec![0], vec![0]], &[0, usize::MAX], 1),
        Err(StepPlanError::PositionOverflow)
    );
    let beyond = usize::try_from(i64::MAX).unwrap();
    assert_eq!(
        plan_step(&[vec![0, 1]], &[beyond], 2),
        Err(StepPlanError::PositionBeyondI64)
    );
    assert!(plan_step(&[vec![0]], &[beyond], 1).is_ok());
}

proptest! {
    /// Step plans reproduce every row's tokens, positions, and restoration.
    #[test]
    fn step_plans_match_a_direct_layout(
        rows in prop::collection::vec((sorted_columns(8), 0usize..20), 1..5),
    ) {
        let columns: Vec<Vec<usize>> = rows.iter().map(|(columns, _)| columns.clone()).collect();
        let starts: Vec<usize> = rows.iter().map(|&(_, start)| start).collect();
        let plan = plan_step(&columns, &starts, 8);
        if columns.iter().all(Vec::is_empty) {
            prop_assert_eq!(plan, Err(StepPlanError::NoActiveRow));
            return Ok(());
        }
        let plan = plan.unwrap();
        let widest = columns.iter().map(Vec::len).max().unwrap();
        for (row, valid) in columns.iter().enumerate() {
            let (column, count, step) = (plan.columns[row], valid.len(), &plan.rows[row]);
            prop_assert!(column + count <= plan.block);
            for slot in 0..plan.block {
                // The block's slot holds input column `source` of this row.
                let source = match plan.offset {
                    Some(offset) => offset + slot,
                    None => step.sources[slot],
                };
                let inside = (column..column + count).contains(&slot);
                prop_assert_eq!(step.valid[slot], inside);
                if inside {
                    prop_assert_eq!(source, valid[slot - column]);
                    prop_assert_eq!(step.positions[slot] as usize, starts[row] + slot - column);
                }
            }
            for (input, &slot) in step.restore.iter().enumerate() {
                let expected = valid.iter().position(|&c| c == input).map_or(plan.block, |k| column + k);
                prop_assert_eq!(slot, expected);
            }
            prop_assert_eq!(step.last, (column + count).saturating_sub(1));
        }
        match plan.offset {
            Some(offset) => {
                let lo = columns.iter().filter_map(|c| c.first()).min().unwrap();
                let hi = columns.iter().filter_map(|c| c.last()).max().unwrap() + 1;
                prop_assert_eq!((offset, plan.block), (*lo, hi - lo));
            }
            None => {
                prop_assert_eq!(plan.block, widest);
                prop_assert!(!columns.iter().all(|c| is_one_run(c)));
            }
        }
    }
}

/// A set is used exactly when some layer names it; NoPE and out-of-range layers mark nothing.
#[test]
fn used_rotary_sets_mark_exactly_the_named_sets() {
    assert_eq!(
        used_rotary_sets(3, &[Some(0), None, Some(2), Some(0)]),
        [true, false, true]
    );
    assert_eq!(used_rotary_sets(2, &[None, Some(5)]), [false, false]);
    assert!(used_rotary_sets(0, &[Some(0)]).is_empty());
}

proptest! {
    /// Used sets equal a direct membership test.
    #[test]
    fn used_rotary_sets_match_membership(
        count in 0usize..5,
        layers in prop::collection::vec(prop::option::of(0usize..6), 0..10),
    ) {
        let used = used_rotary_sets(count, &layers);
        prop_assert_eq!(used.len(), count);
        for (set, &flag) in used.iter().enumerate() {
            prop_assert_eq!(flag, layers.contains(&Some(set)));
        }
    }
}
