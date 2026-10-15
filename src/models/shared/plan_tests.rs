//! Unit and property tests for the pure padded-row index plans.
//!
//! Every planner is compared with a deliberately naive reference over random
//! masks, column lists, cursors, and capacities, complementing the Verus
//! proofs with executable evidence on concrete values.

use proptest::prelude::*;

use super::*;

/// Applies a scatter layout to compact host values, using `0` at padding.
fn apply_layout(layout: &ScatterLayout, compact: &[u32], sequence: usize) -> Vec<u32> {
    match layout {
        ScatterLayout::Identity => compact.to_vec(),
        ScatterLayout::Contiguous { leading, trailing } => {
            let mut output = vec![0; *leading];
            output.extend_from_slice(compact);
            output.extend(std::iter::repeat_n(0, *trailing));
            output
        }
        ScatterLayout::Gather { sources } => {
            assert_eq!(sources.len(), sequence);
            sources
                .iter()
                .map(|&source| compact.get(source).copied().unwrap_or(0))
                .collect()
        }
    }
}

/// Reference scatter: writes compact entry `k` to column `indices[k]`.
fn reference_scatter(indices: &[usize], compact: &[u32], sequence: usize) -> Vec<u32> {
    let mut output = vec![0; sequence];
    for (&column, &value) in indices.iter().zip(compact) {
        output[column] = value;
    }
    output
}

/// Dense, left-padded, right-padded, and interior-padded rows pick the cheapest layout.
#[test]
fn scatter_layout_matches_padding_shape() {
    assert_eq!(plan_scatter(&[0, 1, 2], 3), ScatterLayout::Identity);
    assert_eq!(
        plan_scatter(&[2, 3], 4),
        ScatterLayout::Contiguous {
            leading: 2,
            trailing: 0
        }
    );
    assert_eq!(
        plan_scatter(&[0, 1], 4),
        ScatterLayout::Contiguous {
            leading: 0,
            trailing: 2
        }
    );
    assert_eq!(
        plan_scatter(&[1], 3),
        ScatterLayout::Contiguous {
            leading: 1,
            trailing: 1
        }
    );
    assert_eq!(
        plan_scatter(&[0, 2], 4),
        ScatterLayout::Gather {
            sources: vec![0, 2, 1, 2]
        }
    );
}

/// An empty row scatters to zeros and a single-column row is the identity.
#[test]
fn scatter_layout_handles_empty_and_singleton_rows() {
    assert_eq!(
        plan_scatter(&[], 3),
        ScatterLayout::Contiguous {
            leading: 3,
            trailing: 0
        }
    );
    assert_eq!(apply_layout(&plan_scatter(&[], 3), &[], 3), vec![0, 0, 0]);
    assert_eq!(plan_scatter(&[0], 1), ScatterLayout::Identity);
}

/// Compaction lists exactly the valid columns, in order.
#[test]
fn valid_columns_lists_true_entries() {
    assert_eq!(valid_columns(&[]), Vec::<usize>::new());
    assert_eq!(valid_columns(&[false, false]), Vec::<usize>::new());
    assert_eq!(valid_columns(&[true, false, true, true]), vec![0, 2, 3]);
}

/// The sortedness check rejects duplicates, inversions, and out-of-range entries.
#[test]
fn sorted_unique_check_rejects_each_violation() {
    assert!(is_sorted_unique_below(&[], 0));
    assert!(is_sorted_unique_below(&[0, 2, 5], 6));
    assert!(!is_sorted_unique_below(&[0, 2, 6], 6));
    assert!(!is_sorted_unique_below(&[1, 1], 6));
    assert!(!is_sorted_unique_below(&[2, 1], 6));
}

/// Explicit logits selections must be non-empty and in range.
#[test]
fn positions_within_requires_nonempty_in_range_lists() {
    assert!(!positions_within(&[], 4));
    assert!(positions_within(&[3, 0, 3], 4));
    assert!(!positions_within(&[0, 4], 4));
}

/// The last valid column ignores trailing padding and reports empty rows.
#[test]
fn last_valid_column_skips_trailing_padding() {
    assert_eq!(last_valid_column(&[]), None);
    assert_eq!(last_valid_column(&[false, false]), None);
    assert_eq!(last_valid_column(&[true, true, false]), Some(1));
    assert_eq!(last_valid_column(&[false, true]), Some(1));
}

/// Signed conversion preserves values and rejects indices beyond `i64::MAX`.
#[test]
fn signed_indices_preserve_values_and_reject_overflow() {
    assert_eq!(signed_indices(&[0, 7, 3]), Some(vec![0, 7, 3]));
    assert_eq!(signed_indices(&[]), Some(Vec::new()));
    let beyond = usize::try_from(i64::MAX).expect("64-bit usize") + 1;
    assert_eq!(signed_indices(&[1, beyond]), None);
}

/// Implicit positions continue from the committed cursor and reject overflow.
#[test]
fn implicit_positions_continue_from_baseline() {
    assert_eq!(implicit_positions(5, 3), Some(vec![5, 6, 7]));
    assert_eq!(implicit_positions(5, 0), Some(Vec::new()));
    assert_eq!(implicit_positions(usize::MAX, 2), None);
    let last_signed = usize::try_from(i64::MAX).expect("64-bit usize");
    assert_eq!(implicit_positions(last_signed, 1), Some(vec![i64::MAX]));
    assert_eq!(implicit_positions(last_signed, 2), None);
}

/// The context extent is `max + 1` for non-negative maxima only.
#[test]
fn context_extent_is_one_past_the_maximum() {
    assert_eq!(context_from_maximum(0), Some(1));
    assert_eq!(context_from_maximum(41), Some(42));
    assert_eq!(context_from_maximum(-1), None);
    assert_eq!(
        context_from_maximum(i64::MAX),
        usize::try_from(i64::MAX).ok().map(|maximum| maximum + 1)
    );
}

/// Range checks treat both bounds as part of the half-open interval `0..limit`.
#[test]
fn range_within_uses_half_open_bounds() {
    assert!(range_within(0, 3, 4));
    assert!(!range_within(0, 4, 4));
    assert!(!range_within(-1, 2, 4));
    assert!(!range_within(0, 0, 0));
    assert!(!range_within(-3, -2, 4));
}

/// Value ranges cover only the listed columns.
#[test]
fn value_range_ignores_unlisted_columns() {
    let values = [-1, 7, 3, 99];
    assert_eq!(value_range(&values, &[]), None);
    assert_eq!(value_range(&values, &[1, 2]), Some((3, 7)));
    assert_eq!(value_range(&values, &[2]), Some((3, 3)));
}

/// Appends report the first overflowing or over-capacity row.
#[test]
fn append_plans_report_first_failure() {
    assert_eq!(plan_appends(&[1, 2], &[2, 0], 3), Ok(vec![3, 2]));
    assert_eq!(plan_appends(&[], &[], 0), Ok(Vec::new()));
    assert_eq!(
        plan_appends(&[1, 2, 9], &[3, 9, 0], 3),
        Err(AppendError::Capacity {
            row: 0,
            requested: 4
        })
    );
    assert_eq!(
        plan_appends(&[0, usize::MAX], &[1, 1], usize::MAX),
        Err(AppendError::Overflow {
            row: 1,
            length: usize::MAX,
            additional: 1
        })
    );
}

/// Strategy for a random padded mask row of up to 24 columns.
fn mask_row() -> impl Strategy<Value = Vec<bool>> {
    prop::collection::vec(any::<bool>(), 0..24)
}

proptest! {
    /// Compaction and every scatter layout reproduce the naive per-column scatter.
    #[test]
    fn scatter_layouts_match_reference(row in mask_row()) {
        let sequence = row.len();
        let indices = valid_columns(&row);
        prop_assert!(is_sorted_unique_below(&indices, sequence));
        let compact = (1..=u32::try_from(indices.len()).expect("small row")).collect::<Vec<_>>();
        let layout = plan_scatter(&indices, sequence);
        prop_assert_eq!(
            apply_layout(&layout, &compact, sequence),
            reference_scatter(&indices, &compact, sequence)
        );
        for (column, &valid) in row.iter().enumerate() {
            prop_assert_eq!(valid, indices.contains(&column));
        }
    }

    /// The gather map is an exact inverse of compaction on valid columns.
    #[test]
    fn gather_sources_invert_compaction(row in mask_row()) {
        let indices = valid_columns(&row);
        let sources = gather_sources(&indices, row.len());
        prop_assert_eq!(sources.len(), row.len());
        for (k, &column) in indices.iter().enumerate() {
            prop_assert_eq!(sources[column], k);
        }
        for (column, &source) in sources.iter().enumerate() {
            prop_assert_eq!(source < indices.len(), row[column]);
        }
    }

    /// The last valid column agrees with a reverse search of the mask.
    #[test]
    fn last_column_matches_reverse_search(row in mask_row()) {
        prop_assert_eq!(last_valid_column(&row), row.iter().rposition(|&valid| valid));
    }

    /// Value ranges agree with iterator minima and maxima over the listed columns.
    #[test]
    fn value_range_matches_iterator_bounds(
        values in prop::collection::vec(any::<i64>(), 1..16),
        picks in prop::collection::vec(any::<prop::sample::Index>(), 0..16),
    ) {
        let columns = picks.iter().map(|pick| pick.index(values.len())).collect::<Vec<_>>();
        let listed = columns.iter().map(|&column| values[column]);
        let expected = listed.clone().min().zip(listed.max());
        prop_assert_eq!(value_range(&values, &columns), expected);
    }

    /// Append plans agree with checked per-row arithmetic.
    #[test]
    fn append_plans_match_checked_arithmetic(
        rows in prop::collection::vec((0usize..12, 0usize..12), 0..8),
        capacity in 0usize..20,
    ) {
        let (lengths, counts): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
        let expected = lengths
            .iter()
            .zip(&counts)
            .enumerate()
            .map(|(row, (&length, &count))| {
                let requested = length + count;
                if requested > capacity {
                    Err(AppendError::Capacity { row, requested })
                } else {
                    Ok(requested)
                }
            })
            .collect::<Result<Vec<_>, _>>();
        prop_assert_eq!(plan_appends(&lengths, &counts, capacity), expected);
    }
}
