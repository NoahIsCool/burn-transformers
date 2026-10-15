//! Step layouts: slices for runs, gathers for interior padding, fill positions,
//! exact restoration, and a property test over random masks and lengths.

use burn::tensor::{Int, Tensor, TensorData};
use proptest::prelude::*;

use super::super::test_support::{B, host, input, ints, mask};
use super::*;
use crate::models::shared::{EmptyRows, InputNames, StatefulContract, prepare_stateful_rows};

/// Names used by the planning contract.
const NAMES: InputNames<'static> = InputNames {
    token_ids: "tokens",
    token_mask: "mask",
    position_ids: "positions",
};

/// Validates `[rows, columns]` validity against committed `lengths`.
fn rows_of(valid: &[Vec<bool>], lengths: &[usize]) -> StatefulRows {
    let batch = valid.len();
    let columns = valid[0].len();
    let flags: Vec<bool> = valid.concat();
    let device = Default::default();
    let tokens = Tensor::<B, 2, Int>::zeros([batch, columns], &device);
    let token_mask = crate::TokenMask::new(crate::architecture::bool_tensor_from_values(
        flags,
        [batch, columns],
        &device,
    ));
    prepare_stateful_rows(
        &input(tokens, Some(token_mask), None),
        &device,
        lengths,
        64,
        &StatefulContract {
            names: NAMES,
            owner: "test",
            vocabulary: 4,
            position_limit: None,
            empty_rows: EmptyRows::Permit,
        },
    )
    .unwrap()
}

/// Builds `[rows, columns, 1]` states whose value is `1 + 10·row + column`.
fn labelled(batch: usize, columns: usize) -> Tensor<B, 3> {
    let values: Vec<f32> = (0..batch * columns)
        .map(|index| 1.0 + (10 * (index / columns) + index % columns) as f32)
        .collect();
    Tensor::from_data(
        TensorData::new(values, [batch, columns, 1]),
        &Default::default(),
    )
}

/// Left and right padding share one slice; positions continue each row's cache.
#[test]
fn runs_are_cut_with_one_slice() {
    let valid = vec![
        vec![false, false, true, true],
        vec![true, true, true, false],
    ];
    let layout = StepLayout::plan(&rows_of(&valid, &[5, 0]), &[5, 0]).unwrap();
    assert_eq!(layout.source, BlockSource::Slice { offset: 0 });
    assert_eq!(layout.block, 4);
    assert_eq!(layout.columns(), [2, 0]);
    assert_eq!(layout.counts(), [2, 3]);
    // Row 0 fills with its last new position 6; row 1 continues from 0.
    assert_eq!(layout.positions, [6, 6, 5, 6, 0, 1, 2, 2]);
    let restored = layout.restore(labelled(2, 4)).unwrap();
    assert_eq!(host(restored), [0.0, 0.0, 3.0, 4.0, 11.0, 12.0, 13.0, 0.0]);
}

/// A common window narrower than the input is sliced out.
#[test]
fn shared_padding_columns_are_trimmed_from_the_block() {
    let valid = vec![
        vec![false, true, true, false],
        vec![false, false, true, false],
    ];
    let layout = StepLayout::plan(&rows_of(&valid, &[0, 0]), &[0, 0]).unwrap();
    assert_eq!(layout.source, BlockSource::Slice { offset: 1 });
    assert_eq!(layout.block, 2);
    assert_eq!(layout.columns(), [0, 1]);
    let tokens = ints([[9, 1, 2, 9], [9, 9, 3, 9]]);
    assert_eq!(
        layout
            .cut(&tokens)
            .unwrap()
            .into_data()
            .to_vec::<i32>()
            .unwrap(),
        [1, 2, 9, 3]
    );
    let restored = layout.restore(labelled(2, 2)).unwrap();
    assert_eq!(host(restored), [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 12.0, 0.0]);
}

/// Interior padding is compacted by a gather into a right-padded block.
#[test]
fn interior_padding_is_compacted_with_one_gather() {
    let valid = vec![
        vec![true, false, true, true],
        vec![false, true, false, false],
    ];
    let layout = StepLayout::plan(&rows_of(&valid, &[1, 2]), &[1, 2]).unwrap();
    assert!(matches!(layout.source, BlockSource::Gather { .. }));
    assert_eq!(layout.block, 3);
    assert_eq!(layout.columns(), [0, 0]);
    let tokens = ints([[5, 0, 6, 7], [0, 8, 0, 0]]);
    assert_eq!(
        layout
            .cut(&tokens)
            .unwrap()
            .into_data()
            .to_vec::<i32>()
            .unwrap(),
        [5, 6, 7, 8, 8, 8]
    );
    assert_eq!(layout.positions, [1, 2, 3, 2, 2, 2]);
    let restored = layout.restore(labelled(2, 3)).unwrap();
    assert_eq!(host(restored), [1.0, 0.0, 2.0, 3.0, 0.0, 11.0, 0.0, 0.0]);
}

/// Inactive rows are all padding at their last cached position, with zero last states.
#[test]
fn inactive_rows_fill_with_their_last_cached_position() {
    let valid = vec![vec![true, true], vec![false, false], vec![false, false]];
    let layout = StepLayout::plan(&rows_of(&valid, &[3, 4, 0]), &[3, 4, 0]).unwrap();
    assert_eq!(layout.counts(), [2, 0, 0]);
    assert_eq!(layout.positions, [3, 4, 3, 3, 0, 0]);
    assert_eq!(layout.valid, [true, true, false, false, false, false]);
    let last = layout.last_states(labelled(3, 2)).unwrap();
    assert_eq!(host(last), [2.0, 0.0, 0.0]);
    let none = vec![vec![false]];
    assert!(StepLayout::plan(&rows_of(&none, &[1]), &[1]).is_err());
}

/// Explicit positions replace slot positions only on new tokens.
#[test]
fn padded_explicit_positions_are_replaced_by_fill_positions() {
    let valid = vec![vec![false, true, true]];
    let layout = StepLayout::plan(&rows_of(&valid, &[4]), &[4]).unwrap();
    let device = Default::default();
    let slots = layout.slot_positions::<B>(&device).unwrap();
    let explicit = ints([[777, 10, 11]]);
    let rotary = layout.rotary_positions(Some(&explicit), &slots).unwrap();
    assert_eq!(rotary.into_data().to_vec::<i32>().unwrap(), [10, 11]);
    let implicit = layout.rotary_positions(None, &slots).unwrap();
    assert_eq!(implicit.into_data().to_vec::<i32>().unwrap(), [4, 5]);
    let padded = vec![vec![true, false, true]];
    let layout = StepLayout::plan(&rows_of(&padded, &[0]), &[0]).unwrap();
    let slots = layout.slot_positions::<B>(&device).unwrap();
    let rotary = layout
        .rotary_positions(Some(&ints([[3, -9, 8]])), &slots)
        .unwrap();
    assert_eq!(rotary.into_data().to_vec::<i32>().unwrap(), [3, 8]);
    let mismatched = layout.restore(labelled(1, 5));
    assert!(mismatched.is_err());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Cut, slot positions, and restoration agree with a direct per-row model.
    #[test]
    fn layouts_restore_every_valid_column(
        flags in prop::collection::vec(prop::collection::vec(any::<bool>(), 5), 1..4),
        lengths in prop::collection::vec(0usize..6, 4),
    ) {
        let lengths = &lengths[..flags.len()];
        prop_assume!(flags.iter().flatten().any(|&flag| flag));
        let layout = StepLayout::plan(&rows_of(&flags, lengths), lengths).unwrap();
        let batch = flags.len();
        let restored = host(layout.restore(labelled(batch, layout.block)).unwrap());
        for (row, row_flags) in flags.iter().enumerate() {
            let mut ordinal = 0;
            for (column, &valid) in row_flags.iter().enumerate() {
                let value = restored[row * 5 + column];
                if valid {
                    let slot = layout.columns[row] + ordinal;
                    prop_assert_eq!(value, 1.0 + (10 * row + slot) as f32);
                    prop_assert_eq!(
                        layout.positions[row * layout.block + slot],
                        (lengths[row] + ordinal) as i64
                    );
                    ordinal += 1;
                } else {
                    prop_assert_eq!(value, 0.0);
                }
            }
            prop_assert_eq!(layout.counts[row], ordinal);
            let valid_slots = layout.valid[row * layout.block..(row + 1) * layout.block]
                .iter()
                .filter(|&&valid| valid)
                .count();
            prop_assert_eq!(valid_slots, ordinal);
        }
    }
}

/// The masks used by these tests are well formed.
#[test]
fn helper_masks_round_trip() {
    let token_mask = mask([[true, false]]);
    assert_eq!(token_mask.as_tensor().dims(), [1, 2]);
}
