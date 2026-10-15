//! Tests for hidden-state selection before the vocabulary projection.

use burn::tensor::Tensor;
use proptest::prelude::*;

use super::*;
use crate::models::shared::test_support::{
    TestBackend, assert_same_error, floats, mask, mask_from_rows,
};

/// A `[batch, sequence, 2]` tensor whose entries encode `(row, column, channel)`.
fn hidden(batch: usize, sequence: usize) -> Tensor<TestBackend, 3> {
    let values = (0..batch * sequence * 2)
        .map(|index| index as f32)
        .collect::<Vec<_>>();
    Tensor::<TestBackend, 1>::from_floats(values.as_slice(), &Default::default())
        .reshape([batch, sequence, 2])
}

/// Naive reference: narrows each row at the requested column and concatenates.
fn reference_rows(hidden: &Tensor<TestBackend, 3>, columns: &[usize]) -> Tensor<TestBackend, 3> {
    let rows = columns
        .iter()
        .enumerate()
        .map(|(row, &column)| hidden.clone().narrow(0, row, 1).narrow(1, column, 1))
        .collect();
    Tensor::cat(rows, 0)
}

/// `All` returns the states unchanged and `Last` without a mask takes the final column.
#[test]
fn unmasked_selection_uses_physical_columns() {
    let states = hidden(2, 3);
    assert_eq!(
        floats(select_hidden(states.clone(), None, &LogitsSelection::All).unwrap()),
        floats(states.clone())
    );
    assert_eq!(
        floats(select_hidden(states.clone(), None, &LogitsSelection::Last).unwrap()),
        floats(reference_rows(&states, &[2, 2]))
    );
}

/// `Last` with a mask picks each row's greatest valid column, including interior gaps.
#[test]
fn masked_last_selection_skips_padding() {
    let states = hidden(3, 4);
    let padding = mask([
        [true, true, false, false],
        [false, true, false, true],
        [true, false, false, false],
    ]);
    let selected = select_hidden(states.clone(), Some(&padding), &LogitsSelection::Last).unwrap();
    assert_eq!(selected.dims(), [3, 1, 2]);
    assert_eq!(
        floats(selected),
        floats(reference_rows(&states, &[1, 3, 0]))
    );
}

/// Explicit positions keep caller order and may repeat a column.
#[test]
fn explicit_positions_preserve_order() {
    let states = hidden(1, 4);
    let selected = select_hidden(
        states,
        Some(&mask([[false, true, true, true]])),
        &LogitsSelection::Positions(vec![3, 0, 3]),
    )
    .unwrap();
    assert_eq!(floats(selected), vec![6.0, 7.0, 0.0, 1.0, 6.0, 7.0]);
}

/// Empty or out-of-range explicit selections are shape errors before any work.
#[test]
fn invalid_explicit_positions_are_rejected() {
    for positions in [Vec::new(), vec![0, 4]] {
        assert_same_error(
            select_hidden(
                hidden(1, 4),
                None,
                &LogitsSelection::Positions(positions.clone()),
            )
            .unwrap_err(),
            Error::shape(
                "forward_options.logits.positions",
                "a non-empty list of indices below 4",
                format!("{positions:?}"),
            ),
        );
    }
    assert!(validate_logits_selection(&LogitsSelection::All, 0).is_ok());
    assert!(validate_logits_selection(&LogitsSelection::Positions(vec![1]), 2).is_ok());
}

/// Host-planned `Last` selection slices when every row ends at the final column.
#[test]
fn host_planned_last_selection_slices_dense_rows() {
    let states = hidden(2, 3);
    let selected =
        select_hidden_with_last_columns(states.clone(), &[2, 2], &LogitsSelection::Last).unwrap();
    assert_eq!(floats(selected), floats(reference_rows(&states, &[2, 2])));
    assert_eq!(
        floats(
            select_hidden_with_last_columns(states.clone(), &[0, 1], &LogitsSelection::All)
                .unwrap()
        ),
        floats(states.clone())
    );
    assert_eq!(
        floats(
            select_hidden_with_last_columns(states, &[0, 1], &LogitsSelection::Positions(vec![1]))
                .unwrap()
        ),
        vec![2.0, 3.0, 8.0, 9.0]
    );
}

/// Host-planned selection rejects a wrong row count or an out-of-range column.
#[test]
fn host_planned_last_selection_validates_columns() {
    for columns in [vec![1], vec![1, 3]] {
        assert!(matches!(
            select_hidden_with_last_columns(hidden(2, 3), &columns, &LogitsSelection::Last),
            Err(Error::Shape { tensor, .. }) if tensor == "logits.last_columns"
        ));
    }
}

proptest! {
    /// Device argmax selection and host-planned selection agree with the naive reference.
    #[test]
    fn last_selection_paths_agree(
        rows in (1usize..6).prop_flat_map(|width| {
            prop::collection::vec(prop::collection::vec(any::<bool>(), width), 1..4)
        })
    ) {
        prop_assume!(rows.iter().all(|row| row.contains(&true)));
        let states = hidden(rows.len(), rows[0].len());
        let last = rows
            .iter()
            .map(|row| super::plan::last_valid_column(row).expect("non-empty row"))
            .collect::<Vec<_>>();
        let expected = floats(reference_rows(&states, &last));
        let padding = mask_from_rows(&rows);
        prop_assert_eq!(
            floats(select_hidden(states.clone(), Some(&padding), &LogitsSelection::Last).unwrap()),
            expected.clone()
        );
        prop_assert_eq!(
            floats(select_hidden_with_last_columns(states, &last, &LogitsSelection::Last).unwrap()),
            expected
        );
    }
}

/// Gradients flow through masked gathers and explicit selections to exactly the chosen states.
#[cfg(feature = "autodiff")]
#[test]
fn selection_is_differentiable() {
    use burn::backend::Autodiff;

    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let states = Tensor::<AdBackend, 3>::ones([2, 3, 2], &device).require_grad();
    let padding = TokenMask::new(Tensor::<AdBackend, 2, burn::tensor::Bool>::from_data(
        [[true, true, false], [true, false, false]],
        &device,
    ));
    let last = select_hidden(states.clone(), Some(&padding), &LogitsSelection::Last).unwrap();
    let explicit =
        select_hidden(states.clone(), None, &LogitsSelection::Positions(vec![2])).unwrap();
    let gradients = (last.sum() + explicit.sum()).backward();
    let gradient = states
        .grad(&gradients)
        .expect("selected states receive gradients")
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(
        gradient,
        vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    );
}
