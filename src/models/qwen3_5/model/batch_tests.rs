//! Compact batch layout tests: compaction, restoration, last-slot selection, and validation.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use proptest::prelude::*;

use super::CompactBatch;
use crate::LogitsSelection;

/// CPU Flex backend with F32 values and I32 indices.
type TestBackend = Flex<f32, i32>;

/// Lists the valid columns of every row of a row-major mask.
fn columns_of(mask: &[Vec<bool>]) -> Vec<Vec<usize>> {
    mask.iter()
        .map(|row| (0..row.len()).filter(|&column| row[column]).collect())
        .collect()
}

/// Builds `[B, S]` IDs `100 * b + c` so every caller column is recognizable.
fn ids(batch: usize, sequence: usize) -> Tensor<TestBackend, 2, Int> {
    let values = (0..batch)
        .flat_map(|row| (0..sequence).map(move |column| (100 * row + column) as i64))
        .collect::<Vec<_>>();
    Tensor::from_data(
        TensorData::new(values, [batch, sequence]),
        &Default::default(),
    )
}

/// Embeds compact IDs as `[B, N, 2]` values `(id, id + 0.5)`.
fn embed(ids: Tensor<TestBackend, 2, Int>) -> Tensor<TestBackend, 3> {
    let values = ids.float().unsqueeze_dim::<3>(2);
    Tensor::cat(vec![values.clone(), values + 0.5], 2)
}

/// Reads a float tensor as a flat vector.
fn flat<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

/// Expected restored `[B, S, 2]` values: the embedding at valid columns, zero elsewhere.
fn expected_restored(mask: &[Vec<bool>]) -> Vec<f32> {
    let mut values = Vec::new();
    for (row, flags) in mask.iter().enumerate() {
        for (column, &valid) in flags.iter().enumerate() {
            let id = (100 * row + column) as f32;
            values.extend(if valid { [id, id + 0.5] } else { [0.0, 0.0] });
        }
    }
    values
}

/// Expected `[B, 1, 2]` last-slot values: each row's last valid column, zero for inactive rows.
fn expected_last(mask: &[Vec<bool>]) -> Vec<f32> {
    mask.iter()
        .enumerate()
        .flat_map(
            |(row, flags)| match flags.iter().rposition(|&valid| valid) {
                Some(column) => {
                    let id = (100 * row + column) as f32;
                    [id, id + 0.5]
                }
                None => [0.0, 0.0],
            },
        )
        .collect()
}

/// Without padding the compact block is the input and no mask exists.
#[test]
fn unpadded_batches_are_the_identity() {
    let device = Default::default();
    let layout =
        CompactBatch::<TestBackend>::new(&[vec![0, 1, 2], vec![0, 1, 2]], 3, &[4, 4], &device)
            .unwrap();
    assert!(layout.is_uniform() && layout.token_mask().is_none());
    let compact = layout.compact(&ids(2, 3), 0);
    assert_eq!(compact.to_data(), ids(2, 3).to_data());
    assert_eq!(
        layout.cache_positions().to_data().to_vec::<i32>().unwrap(),
        vec![4, 5, 6, 4, 5, 6]
    );
    let hidden = embed(compact);
    assert_eq!(flat(layout.restore(hidden.clone())), flat(hidden.clone()));
    assert_eq!(
        flat(layout.last(hidden.clone())),
        flat(hidden.narrow(1, 2, 1))
    );
}

/// Distinct working lengths keep the layout padding-free but not uniform.
#[test]
fn ragged_cursors_are_not_uniform() {
    let layout =
        CompactBatch::<TestBackend>::new(&[vec![0], vec![0]], 1, &[3, 7], &Default::default())
            .unwrap();
    assert!(!layout.is_uniform());
    assert_eq!(
        layout.cache_positions().to_data().to_vec::<i32>().unwrap(),
        vec![3, 7]
    );
}

/// Left, interior, right, and fully masked rows restore exactly with zero padding.
#[test]
fn every_padding_layout_round_trips() {
    let device = Default::default();
    let masks = [
        // Right padding: a prefix layout, compacted by a narrow.
        vec![
            vec![true, true, false, false],
            vec![true, true, true, false],
        ],
        // Left padding and an inactive row.
        vec![
            vec![false, false, true, true],
            vec![false, false, false, false],
        ],
        // Interior padding.
        vec![
            vec![true, false, true, false],
            vec![false, true, false, true],
        ],
    ];
    for mask in masks {
        let columns = columns_of(&mask);
        let starts = [2, 5];
        let layout = CompactBatch::<TestBackend>::new(&columns, 4, &starts, &device).unwrap();
        let width = columns.iter().map(Vec::len).max().unwrap();
        assert!(!layout.is_uniform());
        assert_eq!(
            layout.token_mask().is_some(),
            columns.iter().any(|row| row.len() < width)
        );
        let compact = layout.compact(&ids(2, 4), -1);
        assert_eq!(compact.dims(), [2, width]);
        // Valid slots hold the row's valid columns in order.
        let compact_values = compact.clone().to_data().to_vec::<i32>().unwrap();
        for (row, row_columns) in columns.iter().enumerate() {
            for (slot, &column) in row_columns.iter().enumerate() {
                assert_eq!(
                    compact_values[row * width + slot],
                    (100 * row + column) as i32
                );
            }
            // Padding slots carry the requested inert value, never a sentinel.
            for slot in row_columns.len()..width {
                assert_eq!(compact_values[row * width + slot], -1);
            }
        }
        assert_eq!(
            flat(layout.restore(embed(compact.clone()))),
            expected_restored(&mask)
        );
        assert_eq!(
            flat(layout.last(embed(compact.clone()))),
            expected_last(&mask)
        );
        let positions = layout.cache_positions().to_data().to_vec::<i32>().unwrap();
        for (row, &start) in starts.iter().enumerate() {
            for slot in 0..width {
                assert_eq!(positions[row * width + slot], (start + slot) as i32);
            }
        }
    }
}

/// Selection reads `Last` from the compact block and other selections from caller columns.
#[test]
fn selections_match_caller_columns() {
    let device = Default::default();
    let mask = vec![vec![false, true, true], vec![true, false, false]];
    let layout = CompactBatch::<TestBackend>::new(&columns_of(&mask), 3, &[0, 0], &device).unwrap();
    let hidden = embed(layout.compact(&ids(2, 3), 0));
    let last = layout
        .select(hidden.clone(), &LogitsSelection::Last, &[2, 0])
        .unwrap();
    assert_eq!(flat(last), expected_last(&mask));
    let all = layout
        .select(hidden.clone(), &LogitsSelection::All, &[2, 0])
        .unwrap();
    assert_eq!(flat(all), expected_restored(&mask));
    let positions = layout
        .select(hidden, &LogitsSelection::Positions(vec![2, 0]), &[2, 0])
        .unwrap();
    let restored = expected_restored(&mask);
    let expected = [2, 0, 5, 3]
        .into_iter()
        .flat_map(|column| [restored[2 * column], restored[2 * column + 1]])
        .collect::<Vec<_>>();
    assert_eq!(flat(positions), expected);
}

/// Zeroing replaces non-finite padding values instead of multiplying them.
#[test]
fn padding_is_replaced_even_when_non_finite() {
    let device = Default::default();
    let layout =
        CompactBatch::<TestBackend>::new(&[vec![0], vec![0, 1]], 2, &[0, 0], &device).unwrap();
    let poisoned = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0, f32::NAN, 2.0, 3.0], [2, 2, 1]),
        &device,
    );
    assert_eq!(
        flat(layout.zero_padding(poisoned)),
        vec![1.0, 0.0, 2.0, 3.0]
    );
}

/// Malformed plans are rejected before any tensor is built.
#[test]
fn invalid_plans_are_rejected() {
    let device = Default::default();
    assert!(CompactBatch::<TestBackend>::new(&[vec![], vec![]], 2, &[0, 0], &device).is_err());
    assert!(CompactBatch::<TestBackend>::new(&[vec![0, 3]], 3, &[0], &device).is_err());
    assert!(CompactBatch::<TestBackend>::new(&[vec![0]], 1, &[0, 1], &device).is_err());
    assert!(CompactBatch::<TestBackend>::new(&[], 1, &[], &device).is_err());
    assert!(CompactBatch::<TestBackend>::new(&[vec![0]], 1, &[usize::MAX], &device).is_err());
    // Columns must be strictly increasing, as a mask produces them.
    assert!(CompactBatch::<TestBackend>::new(&[vec![2, 1]], 3, &[0], &device).is_err());
}

proptest! {
    /// Compaction followed by restoration is exact for every mask with an active row.
    #[test]
    fn restore_inverts_compaction(
        mask in proptest::collection::vec(proptest::collection::vec(any::<bool>(), 1..7), 1..4)
            .prop_filter("rectangular with an active row", |rows| {
                rows.iter().all(|row| row.len() == rows[0].len())
                    && rows.iter().flatten().any(|&valid| valid)
            }),
        start in 0usize..9,
    ) {
        let device = Default::default();
        let (batch, sequence) = (mask.len(), mask[0].len());
        let starts = (0..batch).map(|row| start + row).collect::<Vec<_>>();
        let layout = CompactBatch::<TestBackend>::new(&columns_of(&mask), sequence, &starts, &device)
            .unwrap();
        let hidden = embed(layout.compact(&ids(batch, sequence), 0));
        prop_assert_eq!(flat(layout.restore(hidden.clone())), expected_restored(&mask));
        prop_assert_eq!(flat(layout.last(hidden)), expected_last(&mask));
    }
}
