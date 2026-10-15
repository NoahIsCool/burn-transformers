//! Packing, inert padding, and scatter tests for ragged `[B, T]` blocks.
//!
//! The host maps are verified by Verus and property-tested in `batch_tests`;
//! these tests pin the device side: what [`BlockLayout::compact`] and
//! [`BlockLayout::compact_inert`] gather, which entries count as padding, and
//! that [`BlockLayout::scatter`] restores exact zeros at masked columns.

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, TensorData},
};

use super::*;
use crate::{
    CausalLmInput, TokenMask,
    models::{
        deepseek_v2::model::INPUT_NAMES,
        shared::{EmptyRows, StatefulContract, prepare_stateful_rows},
    },
};

/// Deterministic CPU backend with I32 token indices.
type TestBackend = Flex<f32, i32>;

/// Vocabulary bound of the fixtures; every valid token is below it.
const VOCABULARY: usize = 32;

/// Plans `tokens` under `mask` appended at `baselines`, as a decode step would.
///
/// Masked columns are never validated, so they may hold any sentinel.
fn plan(
    tokens: Vec<i32>,
    mask: Vec<bool>,
    shape: [usize; 2],
    baselines: &[usize],
) -> (Option<BlockLayout<TestBackend>>, CausalLmInput<TestBackend>) {
    let device = Default::default();
    let input = CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(tokens, shape),
            &device,
        ),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_data(
            TensorData::new(mask, shape),
            &device,
        ))),
        position_ids: None,
    };
    let rows = prepare_stateful_rows(
        &input,
        &device,
        baselines,
        64,
        &StatefulContract {
            names: INPUT_NAMES,
            owner: "block fixture",
            vocabulary: VOCABULARY,
            position_limit: None,
            empty_rows: EmptyRows::Permit,
        },
    )
    .expect("valid columns pass validation");
    (BlockLayout::new(&rows, baselines, &device).unwrap(), input)
}

/// Reads an integer `[B, T]` tensor back as rows.
fn rows_of(tensor: Tensor<TestBackend, 2, Int>) -> Vec<Vec<i64>> {
    let [_, width] = tensor.dims();
    tensor
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .unwrap()
        .chunks(width)
        .map(<[i64]>::to_vec)
        .collect()
}

/// A finished row's sentinels never leave `compact_inert`; valid tokens pack left.
///
/// Row 0 is left-padded (`[S, 7, 8]`), row 1 is finished (every column a
/// sentinel). Plain `compact` gathers row 1's unvalidated column 0, which is
/// exactly what reached the embedding gather before the inert fill existed.
#[test]
fn compact_inert_replaces_every_padding_entry() {
    let sentinel = i32::MAX;
    let (layout, input) = plan(
        vec![-1, 7, 8, sentinel, sentinel, 32],
        vec![false, true, true, false, false, false],
        [2, 3],
        &[4, 9],
    );
    let layout = layout.expect("one active row");
    assert_eq!(layout.block(), 2);
    let raw = rows_of(layout.compact(input.token_ids.clone()));
    assert_eq!(raw[0], [7, 8]);
    assert_eq!(
        raw[1],
        [i64::from(sentinel); 2],
        "plain compaction leaks sentinels"
    );
    let inert = rows_of(layout.compact_inert(input.token_ids, 0));
    assert_eq!(inert, [vec![7, 8], vec![0, 0]]);
    assert!(
        inert
            .iter()
            .flatten()
            .all(|&token| (0..VOCABULARY as i64).contains(&token)),
        "every packed token is a valid embedding index"
    );
}

/// Right-padded rows overwrite only their padding tail; valid entries keep values.
#[test]
fn compact_inert_keeps_valid_entries_of_ragged_rows() {
    let (layout, input) = plan(
        vec![3, 4, 5, 6, -7, -7],
        vec![true, true, true, true, false, false],
        [2, 3],
        &[0, 2],
    );
    let layout = layout.expect("active rows");
    assert_eq!(
        rows_of(layout.compact_inert(input.token_ids, 0)),
        [[3, 4, 5], [6, 0, 0]]
    );
    assert_eq!(rows_of(layout.implicit_positions()), [[0, 1, 2], [2, 0, 0]]);
}

/// Fully valid calls are the identity: no gather, no fill, and no padding entries.
#[test]
fn fully_valid_calls_pack_without_any_rewrite() {
    let (layout, input) = plan(vec![1, 2, 3, 4], vec![true; 4], [2, 2], &[5, 0]);
    let layout = layout.expect("active rows");
    assert_eq!(
        rows_of(layout.compact_inert(input.token_ids, 9)),
        [[1, 2], [3, 4]]
    );
    assert_eq!(rows_of(layout.implicit_positions()), [[5, 6], [0, 1]]);
}

/// Equal-width rows at different columns gather without any padding fill.
#[test]
fn equal_width_rows_need_no_fill() {
    let (layout, input) = plan(
        vec![-5, 1, 2, -5],
        vec![false, true, true, false],
        [2, 2],
        &[0, 0],
    );
    let layout = layout.expect("active rows");
    assert_eq!(
        rows_of(layout.compact_inert(input.token_ids, 0)),
        [[1], [2]]
    );
}

/// A step in which every row is finished plans no block at all.
#[test]
fn all_finished_rows_plan_no_block() {
    let (layout, _) = plan(vec![-1, 99], vec![false, false], [2, 1], &[3, 4]);
    assert!(layout.is_none());
}

/// Scatter restores valid columns and writes exact zeros at every masked column.
#[test]
fn scatter_restores_columns_with_exact_zero_padding() {
    let (layout, _) = plan(
        vec![0, 5, 6, 7, 0, 0],
        vec![false, true, true, true, false, false],
        [2, 3],
        &[0, 0],
    );
    let layout = layout.expect("active rows");
    let device = Default::default();
    // Block entry k of row b carries 10 * b + k + 1 in both channels.
    let block = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            vec![1.0_f32, 1.0, 2.0, 2.0, 11.0, 11.0, 12.0, 12.0],
            [2, 2, 2],
        ),
        &device,
    );
    let restored = layout.scatter(block).into_data().to_vec::<f32>().unwrap();
    assert_eq!(
        restored,
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 11.0, 11.0, 0.0, 0.0, 0.0, 0.0]
    );
}
