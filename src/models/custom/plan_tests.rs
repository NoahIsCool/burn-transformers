//! Displacement-table geometry and history-append planning: examples, edges, and properties.

use super::*;

/// Naive reference for one append: per-row counts, new lengths, and slots, or the first error.
///
/// Mirrors the documented policy order without sharing code with the planner.
fn reference_append(
    validity: &[bool],
    step: usize,
    lengths: &[usize],
    capacity: usize,
    require_every_row: bool,
) -> Result<HistoryAppend, HistoryAppendError> {
    let rows = validity.chunks_exact(step).collect::<Vec<_>>();
    let counts = rows
        .iter()
        .map(|row| row.iter().filter(|&&valid| valid).count())
        .collect::<Vec<_>>();
    if require_every_row && let Some(row) = counts.iter().position(|&count| count == 0) {
        return Err(HistoryAppendError::EmptyRow { row });
    }
    if counts.iter().all(|&count| count == 0) {
        return Err(HistoryAppendError::EmptyStep);
    }
    let mut new_lengths = Vec::new();
    for (row, (&length, &count)) in lengths.iter().zip(&counts).enumerate() {
        match length.checked_add(count) {
            Some(requested) if requested <= capacity => new_lengths.push(requested),
            requested => return Err(HistoryAppendError::Capacity { row, requested }),
        }
    }
    if let Some(row) = new_lengths.iter().position(|&length| length == 0) {
        return Err(HistoryAppendError::NeverStarted { row });
    }
    let mut slots = Vec::new();
    for (row, cells) in rows.iter().enumerate() {
        let mut next = lengths[row];
        for &valid in *cells {
            if valid {
                slots.push(next);
                next += 1;
            } else {
                slots.push(0);
            }
        }
    }
    Ok(HistoryAppend {
        counts,
        lengths: new_lengths,
        slots,
    })
}

/// The table radius is `min(max_distance, C - 1)` and its length `2R + 1`.
#[test]
fn displacement_table_clamps_to_the_shorter_of_distance_and_context() {
    assert_eq!(displacement_table(128, 0), None);
    assert_eq!(
        displacement_table(128, 1),
        Some(DisplacementTable {
            radius: 0,
            length: 1
        })
    );
    assert_eq!(
        displacement_table(128, 10),
        Some(DisplacementTable {
            radius: 9,
            length: 19
        })
    );
    assert_eq!(
        displacement_table(8, 1000),
        Some(DisplacementTable {
            radius: 8,
            length: 17
        })
    );
    // `2R + 1` must fit `usize`: the largest representable radius is `usize::MAX / 2`.
    assert_eq!(displacement_table(usize::MAX, usize::MAX), None);
    assert_eq!(displacement_table(usize::MAX / 2 + 1, usize::MAX), None);
    assert_eq!(
        displacement_table(usize::MAX / 2, usize::MAX),
        Some(DisplacementTable {
            radius: usize::MAX / 2,
            length: usize::MAX
        })
    );
}

/// Portable tables keep the verified table only while its length fits `i32::MAX`.
#[test]
fn portable_displacement_tables_stop_at_the_32_bit_index_limit() {
    let limit = i32::MAX as usize;
    assert_eq!(portable_displacement_table(128, 0), None);
    assert_eq!(
        portable_displacement_table(128, 10),
        displacement_table(128, 10)
    );
    // `R = i32::MAX / 2` gives `2R + 1 = i32::MAX`, the largest portable table.
    let largest = portable_displacement_table(limit / 2, usize::MAX).unwrap();
    assert_eq!((largest.radius, largest.length), (limit / 2, limit));
    assert_eq!(portable_displacement_table(limit / 2 + 1, usize::MAX), None);
    assert!(displacement_table(limit / 2 + 1, usize::MAX).is_some());
    assert_eq!(portable_displacement_table(usize::MAX, usize::MAX), None);
}

/// Slots clamp out-of-range displacements and invert the stored displacement exactly.
#[test]
fn displacement_slots_clamp_and_invert() {
    for radius in [0usize, 1, 3, 17] {
        for slot in 0..=2 * radius {
            let displacement = slot_displacement(slot, radius);
            assert_eq!(displacement, slot as i64 - radius as i64);
            assert_eq!(displacement_slot(displacement, radius), slot);
        }
        assert_eq!(displacement_slot(i64::MIN, radius), 0);
        assert_eq!(displacement_slot(-(radius as i64) - 5, radius), 0);
        assert_eq!(displacement_slot(radius as i64 + 5, radius), 2 * radius);
        assert_eq!(displacement_slot(i64::MAX, radius), 2 * radius);
    }
}

/// Valid columns fill the row's next slots in order; masked columns read slot zero.
#[test]
fn row_appends_assign_consecutive_slots_to_valid_columns() {
    let plan = plan_row_append(&[false, true, false, true, true], 3, 8).unwrap();
    assert_eq!(plan.count, 3);
    assert_eq!(plan.length, 6);
    assert_eq!(plan.slots, vec![0, 3, 0, 4, 5]);
    let empty = plan_row_append(&[false, false], 2, 2).unwrap();
    assert_eq!((empty.count, empty.length, empty.slots), (0, 2, vec![0, 0]));
    assert!(plan_row_append(&[true, true, true], 6, 8).is_none());
    assert!(plan_row_append(&[], 9, 8).is_none());
    assert!(plan_row_append(&[true], usize::MAX, usize::MAX).is_none());
}

/// Errors follow the documented order and name the first offending row.
#[test]
fn history_appends_report_policy_violations_in_order() {
    // An empty prompt row wins over a capacity overflow in an earlier row.
    assert_eq!(
        plan_history_append(&[true, true, false, false], 2, &[3, 0], 3, true),
        Err(HistoryAppendError::EmptyRow { row: 1 })
    );
    assert_eq!(
        plan_history_append(&[false, false, false, false], 2, &[1, 1], 4, false),
        Err(HistoryAppendError::EmptyStep)
    );
    assert_eq!(
        plan_history_append(&[true, false, true, true], 2, &[0, 3], 4, false),
        Err(HistoryAppendError::Capacity {
            row: 1,
            requested: Some(5)
        })
    );
    assert_eq!(
        plan_history_append(
            &[true, false, true],
            1,
            &[usize::MAX, 0, 0],
            usize::MAX,
            false
        ),
        Err(HistoryAppendError::Capacity {
            row: 0,
            requested: None
        })
    );
    assert_eq!(
        plan_history_append(&[true, false], 1, &[0, 0], 4, false),
        Err(HistoryAppendError::NeverStarted { row: 1 })
    );
    let plan =
        plan_history_append(&[true, false, false, true, true, true], 3, &[2, 0], 4, true).unwrap();
    assert_eq!(plan.counts, vec![1, 3]);
    assert_eq!(plan.lengths, vec![3, 3]);
    assert_eq!(plan.slots, vec![2, 0, 0, 0, 1, 2]);
}

/// The scalar helpers count, locate zeros, and split rows exactly.
#[test]
fn history_helpers_count_and_search_rows() {
    assert_eq!(count_valid_cells(&[]), 0);
    assert_eq!(count_valid_cells(&[true, false, true, true]), 3);
    assert_eq!(
        grid_row_counts(&[true, false, false, false, true, true], 2, 3),
        vec![1, 0, 2]
    );
    assert_eq!(grid_row_counts(&[], 0, 4), vec![0, 0, 0, 0]);
    assert_eq!(first_zero(&[3, 1, 0, 0]), Some(2));
    assert_eq!(first_zero(&[3, 1]), None);
    assert_eq!(first_nonzero(&[0, 0, 4, 1]), Some(2));
    assert_eq!(first_nonzero(&[0, 0]), None);
    assert_eq!(first_nonzero(&[]), None);
}

/// Degenerate grids follow the policy order instead of panicking.
#[test]
fn history_appends_without_cells_report_the_first_policy() {
    assert_eq!(
        plan_history_append(&[], 3, &[], 4, true),
        Err(HistoryAppendError::EmptyStep)
    );
    assert_eq!(
        plan_history_append(&[], 0, &[1, 2], 4, true),
        Err(HistoryAppendError::EmptyRow { row: 0 })
    );
    assert_eq!(
        plan_history_append(&[], 0, &[1, 2], 4, false),
        Err(HistoryAppendError::EmptyStep)
    );
}

/// The window spans the appended slots of every advancing row and nothing else.
#[test]
fn history_writes_cover_exactly_the_appended_slots() {
    let validity = [false, true, true, true, false, false, true, false, true];
    let plan = plan_history_append(&validity, 3, &[4, 2, 1], 8, false).unwrap();
    let writes = plan_history_writes(&plan, &validity, 3);
    assert_eq!((writes.start, writes.end), (1, 6));
    // Row 0 writes slots 4..6 from columns 1, 2; row 1 slot 2 from column 0;
    // row 2 slots 1..3 from columns 0, 2.
    assert_eq!(
        writes.written,
        vec![
            false, false, false, true, true, //
            false, true, false, false, false, //
            true, true, false, false, false,
        ]
    );
    assert_eq!(
        writes.sources,
        vec![
            0, 0, 0, 1, 2, //
            0, 0, 0, 0, 0, //
            0, 2, 0, 0, 0,
        ]
    );
}

/// Full ranges are attained bounds; an empty list has none.
#[test]
fn full_range_reports_attained_bounds() {
    assert_eq!(full_range(&[]), None);
    assert_eq!(full_range(&[4]), Some((4, 4)));
    assert_eq!(full_range(&[3, -2, 9, 0]), Some((-2, 9)));
}

/// Position violations ignore masked cells and report the first failing row's valid range.
#[test]
fn position_violations_report_the_first_failing_row() {
    let validity = [true, false, true, true];
    assert_eq!(
        first_position_violation(&[1, 99, 0, 3], &validity, 2, 2, 4),
        None
    );
    assert_eq!(
        first_position_violation(&[1, 0, -1, 7], &validity, 2, 2, 4),
        Some((-1, 7))
    );
    assert_eq!(
        first_position_violation(&[4, 0, -1, 7], &validity, 2, 2, 4),
        Some((4, 4))
    );
    assert_eq!(first_position_violation(&[], &[], 0, 3, 4), None);
}

/// The history prefix spans the longest row and marks each row's stored length.
#[test]
fn prefix_validity_marks_each_rows_length() {
    assert_eq!(prefix_validity(&[]), None);
    assert_eq!(
        prefix_validity(&[2, 0, 3]),
        Some((
            3,
            vec![true, true, false, false, false, false, true, true, true]
        ))
    );
    assert_eq!(prefix_validity(&[0, 0]), Some((0, Vec::new())));
}

proptest::proptest! {
    /// Position violations agree with a naive per-row scan of valid cells.
    #[test]
    fn position_violations_match_the_reference(
        rows in 0usize..4,
        step in 1usize..5,
        values in proptest::collection::vec(-2i64..8, 16),
        flags in proptest::collection::vec(proptest::bool::ANY, 16),
        limit in 0usize..8,
    ) {
        let positions = &values[..rows * step];
        let validity = &flags[..rows * step];
        let expected = positions
            .chunks_exact(step)
            .zip(validity.chunks_exact(step))
            .find_map(|(values, valid)| {
                let listed = values.iter().zip(valid).filter(|(_, valid)| **valid).map(|(v, _)| *v);
                let range = listed.clone().min().zip(listed.max())?;
                let within = range.0 >= 0 && range.1 >= 0 && (range.1 as u64) < limit as u64;
                (!within).then_some(range)
            });
        proptest::prop_assert_eq!(
            first_position_violation(positions, validity, rows, step, limit),
            expected
        );
    }

    /// The prefix validity agrees with a naive comparison grid.
    #[test]
    fn prefix_validity_matches_the_reference(
        lengths in proptest::collection::vec(0usize..6, 1..5),
    ) {
        let sequence = *lengths.iter().max().unwrap();
        let expected = lengths
            .iter()
            .flat_map(|&length| (0..sequence).map(move |column| column < length))
            .collect::<Vec<_>>();
        proptest::prop_assert_eq!(prefix_validity(&lengths), Some((sequence, expected)));
    }

    /// The planner agrees with the naive reference, and its window writes exactly the new slots.
    #[test]
    fn history_planning_matches_the_reference(
        batch in 1usize..5,
        step in 1usize..6,
        cells in proptest::collection::vec(proptest::bool::ANY, 30),
        lengths in proptest::collection::vec(0usize..6, 5),
        capacity in 0usize..12,
        require_every_row in proptest::bool::ANY,
    ) {
        let validity = &cells[..batch * step];
        let lengths = &lengths[..batch];
        let planned = plan_history_append(validity, step, lengths, capacity, require_every_row);
        proptest::prop_assert_eq!(
            &planned,
            &reference_append(validity, step, lengths, capacity, require_every_row)
        );
        let combined = plan_history(validity, step, lengths, capacity, require_every_row);
        match (&planned, &combined) {
            (Ok(plan), Ok((combined_plan, combined_writes))) => {
                proptest::prop_assert_eq!(plan, combined_plan);
                proptest::prop_assert_eq!(
                    &plan_history_writes(plan, validity, step),
                    combined_writes
                );
            }
            (Err(error), Err(combined_error)) => proptest::prop_assert_eq!(error, combined_error),
            _ => proptest::prop_assert!(false, "plan_history disagrees with its parts"),
        }
        if let Ok(plan) = planned {
            let writes = plan_history_writes(&plan, validity, step);
            let width = writes.end - writes.start;
            for row in 0..batch {
                let appended = lengths[row]..plan.lengths[row];
                let columns = (0..step)
                    .filter(|&column| validity[row * step + column])
                    .collect::<Vec<_>>();
                for offset in 0..width {
                    let slot = writes.start + offset;
                    let cell = row * width + offset;
                    proptest::prop_assert_eq!(writes.written[cell], appended.contains(&slot));
                    if writes.written[cell] {
                        proptest::prop_assert_eq!(
                            writes.sources[cell],
                            columns[slot - lengths[row]]
                        );
                    }
                }
                // Every appended slot lies inside the window.
                proptest::prop_assert!(
                    appended.is_empty()
                        || (writes.start <= appended.start && appended.end <= writes.end)
                );
            }
        }
    }
}
