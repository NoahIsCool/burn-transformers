//! Tests for stateful validation, compaction, scatter parity, and row assembly.

use std::time::Instant;

use burn::tensor::{DType, Int, Tensor};
use proptest::prelude::*;

use super::*;
use crate::architecture::TokenMask;
use crate::models::shared::test_support::{
    TestBackend, assert_same_error, floats, int_values, ints, mask, mask_from_rows,
};

/// Error names used by every contract in these tests.
const NAMES: InputNames<'static> = InputNames {
    token_ids: "s.input_ids",
    token_mask: "s.token_mask",
    position_ids: "s.position_ids",
};

/// A decode contract over 10 tokens and 16 positions.
fn decode_contract() -> StatefulContract<'static> {
    StatefulContract {
        names: NAMES,
        owner: "test state",
        vocabulary: 10,
        position_limit: Some(16),
        empty_rows: EmptyRows::Permit,
    }
}

/// The same contract with prefill's empty-row rejection.
fn prefill_contract() -> StatefulContract<'static> {
    StatefulContract {
        empty_rows: EmptyRows::Reject,
        ..decode_contract()
    }
}

/// Builds an input from its three optional parts.
fn input(
    token_ids: Tensor<TestBackend, 2, Int>,
    token_mask: Option<TokenMask<TestBackend>>,
    position_ids: Option<Tensor<TestBackend, 2, Int>>,
) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids,
        token_mask,
        position_ids,
    }
}

/// Prepares `input` against cursors `lengths` and capacity 8.
fn prepare(
    input: &CausalLmInput<TestBackend>,
    lengths: &[usize],
    contract: &StatefulContract<'_>,
) -> Result<StatefulRows> {
    prepare_stateful_rows(input, &Default::default(), lengths, 8, contract)
}

/// Former per-token scatter used as the parity oracle for every layout.
fn reference_scatter<K>(
    compact: Tensor<TestBackend, 3, K>,
    sequence: usize,
    indices: &[usize],
) -> Tensor<TestBackend, 3, K>
where
    K: TensorKind<TestBackend> + BasicOps<TestBackend>,
{
    let [_, _, width] = compact.dims();
    let options = (&compact.device(), compact.dtype());
    let mut segments = Vec::new();
    let mut cursor = 0;
    for (compact_index, &output_index) in indices.iter().enumerate() {
        if output_index > cursor {
            segments.push(Tensor::zeros([1, output_index - cursor, width], options));
        }
        segments.push(compact.clone().narrow(1, compact_index, 1));
        cursor = output_index + 1;
    }
    if cursor < sequence {
        segments.push(Tensor::zeros([1, sequence - cursor, width], options));
    }
    Tensor::cat(segments, 1)
}

/// Distinct compact values `[1, n, width]` starting at one.
fn compact_values(count: usize, width: usize) -> Tensor<TestBackend, 3> {
    let values = (1..=count * width)
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    Tensor::<TestBackend, 1>::from_floats(values.as_slice(), &Default::default())
        .reshape([1, count, width])
}

/// Left padding, an inactive decode row, and implicit contexts are planned exactly.
#[test]
fn decode_rows_plan_valid_columns_and_contexts() {
    let rows = prepare(
        &input(
            ints([[0, 3, 4], [5, 6, 7], [1, 1, 1]]),
            Some(mask([
                [false, true, true],
                [true, true, true],
                [false, false, false],
            ])),
            None,
        ),
        &[2, 0, 5],
        &decode_contract(),
    )
    .expect("valid decode");
    assert_eq!((rows.batch, rows.sequence), (3, 3));
    assert_eq!(rows.valid_indices, vec![vec![1, 2], vec![0, 1, 2], vec![]]);
    assert_eq!(rows.token_counts, vec![2, 3, 0]);
    assert_eq!(rows.evaluated_contexts, vec![4, 3, 5]);
    assert_eq!(rows.last_columns(), &[2, 2, 2]);
    assert!(
        rows.compact_row::<TestBackend>(2, &Default::default())
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        rows.compact_row::<TestBackend>(3, &Default::default()),
        Err(Error::Shape { .. })
    ));
}

/// An unmasked call treats every column as valid.
#[test]
fn unmasked_rows_are_dense() {
    let rows = prepare(
        &input(ints([[1, 2]]), None, None),
        &[3],
        &prefill_contract(),
    )
    .unwrap();
    assert_eq!(rows.valid_indices, vec![vec![0, 1]]);
    assert_eq!(rows.evaluated_contexts, vec![5]);
    assert_eq!(rows.last_columns(), &[1]);
}

/// Padding sentinels are ignored, while valid out-of-range values are reported exactly.
#[test]
fn values_are_validated_over_valid_columns_only() {
    let padded = || Some(mask([[false, true, true]]));
    let rows = prepare(
        &input(ints([[-1, 2, 3]]), padded(), Some(ints([[-7, 4, 9]]))),
        &[0],
        &decode_contract(),
    )
    .expect("sentinels at padding are inert");
    assert_eq!(rows.evaluated_contexts, vec![10]);

    assert_same_error(
        prepare(
            &input(ints([[-1, 2, 12]]), padded(), None),
            &[0],
            &decode_contract(),
        )
        .unwrap_err(),
        Error::shape(
            "s.input_ids",
            "token values in 0..10",
            "observed range 2..=12",
        ),
    );
    assert_same_error(
        prepare(
            &input(ints([[0, 2, 3]]), padded(), Some(ints([[0, 4, 16]]))),
            &[0],
            &decode_contract(),
        )
        .unwrap_err(),
        Error::shape(
            "s.position_ids",
            "position values in 0..16",
            "observed range 4..=16",
        ),
    );
}

/// Without a position bound, negative valid positions still have no context.
#[test]
fn unbounded_negative_positions_are_rejected() {
    let contract = StatefulContract {
        position_limit: None,
        ..decode_contract()
    };
    assert!(matches!(
        prepare(
            &input(ints([[1]]), None, Some(ints([[-2]]))),
            &[0],
            &contract
        ),
        Err(Error::Configuration { .. })
    ));
}

/// Prefill rejects fully masked rows; decode treats them as no-ops.
#[test]
fn empty_row_policy_depends_on_phase() {
    let call = input(
        ints([[1, 2], [3, 4]]),
        Some(mask([[true, true], [false, false]])),
        None,
    );
    assert_same_error(
        prepare(&call, &[0, 0], &prefill_contract()).unwrap_err(),
        Error::shape(
            "s.token_mask",
            "at least one valid token in every prefill row",
            "a fully masked prompt row",
        ),
    );
    assert_eq!(
        prepare(&call, &[0, 0], &decode_contract())
            .unwrap()
            .token_counts,
        vec![2, 0]
    );
}

/// Batch, capacity, and cursor-overflow violations are typed cache errors.
#[test]
fn cache_transitions_report_typed_errors() {
    let call = input(ints([[1, 2]]), None, None);
    assert!(matches!(
        prepare(&call, &[0, 0], &decode_contract()),
        Err(Error::TensorCache(TensorCacheError::BatchSizeMismatch {
            configured: 2,
            requested: 1
        }))
    ));
    assert!(matches!(
        prepare(&call, &[7], &decode_contract()),
        Err(Error::TensorCache(TensorCacheError::Metadata(
            DecodeStateError::CapacityExceeded {
                sequence: 0,
                requested: 9,
                capacity: 8
            }
        )))
    ));
    assert!(matches!(
        prepare_stateful_rows(
            &call,
            &Default::default(),
            &[usize::MAX],
            usize::MAX,
            &decode_contract()
        ),
        Err(Error::TensorCache(TensorCacheError::Metadata(
            DecodeStateError::LengthOverflow { sequence: 0, .. }
        )))
    ));
}

/// Empty grids and mismatched auxiliary extents are shape errors.
#[test]
fn geometric_violations_are_shape_errors() {
    let empty = Tensor::<TestBackend, 2, Int>::from_data([[0_i64; 0]; 1], &Default::default());
    assert_same_error(
        prepare(&input(empty, None, None), &[0], &decode_contract()).unwrap_err(),
        Error::shape("s.input_ids", "non-empty [batch, sequence]", "[1, 0]"),
    );
    assert!(matches!(
        prepare(&input(ints([[1, 2]]), Some(mask([[true]])), None), &[0], &decode_contract()),
        Err(Error::Shape { tensor, .. }) if tensor == "s.token_mask"
    ));
    assert!(matches!(
        prepare(&input(ints([[1, 2]]), None, Some(ints([[1]]))), &[0], &decode_contract()),
        Err(Error::Shape { tensor, .. }) if tensor == "s.position_ids"
    ));
}

/// Compact rows reject column lists that violate the planning precondition.
#[test]
fn compact_rows_require_sorted_unique_columns() {
    let device = Default::default();
    for columns in [vec![], vec![1, 1], vec![2, 1], vec![0, 3]] {
        assert!(matches!(
            CompactRow::<TestBackend>::new(0, &columns, 3, &device),
            Err(Error::Shape { .. })
        ));
    }
    assert!(matches!(
        CompactRow::<TestBackend>::new(0, &[0], 2, &device)
            .unwrap()
            .scatter(compact_values(2, 1)),
        Err(Error::Shape { .. })
    ));
}

/// Compaction selects valid tokens of the addressed row for every layout.
#[test]
fn compaction_selects_row_columns() {
    let device = Default::default();
    let tokens = ints([[10, 11, 12, 13], [20, 21, 22, 23]]);
    for (columns, expected) in [
        (vec![0, 1, 2, 3], vec![20, 21, 22, 23]),
        (vec![2, 3], vec![22, 23]),
        (vec![0, 2], vec![20, 22]),
    ] {
        let row = CompactRow::<TestBackend>::new(1, &columns, 4, &device).unwrap();
        assert_eq!(row.count(), columns.len());
        assert_eq!(int_values(row.compact(&tokens)), expected);
    }
}

/// Scatter preserves reduced-precision dtypes and integer kinds exactly.
#[test]
fn scatter_preserves_dtype_and_kind() {
    let device = Default::default();
    let row = CompactRow::<TestBackend>::new(0, &[1, 3], 4, &device).unwrap();
    let half = row.scatter(compact_values(2, 2).cast(DType::F16)).unwrap();
    assert_eq!(half.dtype(), DType::F16);
    assert_eq!(
        floats(half.cast(DType::F32)),
        vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0]
    );
    let experts = Tensor::<TestBackend, 3, Int>::from_data([[[5, 6], [7, 8]]], &device);
    assert_eq!(
        int_values(row.scatter(experts).unwrap()),
        vec![0, 0, 5, 6, 0, 0, 7, 8]
    );
}

/// Row assembly rejects mixed dtypes instead of reaching a panicking backend `cat`.
#[test]
fn concat_rows_rejects_mixed_dtypes() {
    let full = compact_values(1, 2);
    let half = full.clone().cast(DType::F16);
    assert!(matches!(
        concat_rows(vec![full.clone(), half], "router_logits"),
        Err(Error::BackendCapability(message)) if message.contains("router_logits")
    ));
    assert!(matches!(
        concat_rows::<TestBackend, burn::tensor::Float>(Vec::new(), "rows"),
        Err(Error::Shape { .. })
    ));
    assert_eq!(
        concat_rows(vec![full.clone(), full], "rows")
            .unwrap()
            .dims(),
        [2, 1, 2]
    );
}

/// Implicit cached positions continue from the cursor and reject overflow.
#[test]
fn implicit_position_ids_continue_from_cursor() {
    let device = Default::default();
    assert_eq!(
        int_values(implicit_position_ids::<TestBackend>(4, 3, &device, "p").unwrap()),
        vec![4, 5, 6]
    );
    assert!(matches!(
        implicit_position_ids::<TestBackend>(usize::MAX, 2, &device, "p"),
        Err(Error::Configuration { .. })
    ));
}

/// Active contexts skip inactive rows and keep row indices.
#[test]
fn active_contexts_skip_inactive_rows() {
    let rows = prepare(
        &input(
            ints([[1, 2], [3, 4], [5, 6]]),
            Some(mask([[true, false], [false, false], [true, true]])),
            None,
        ),
        &[1, 4, 0],
        &decode_contract(),
    )
    .unwrap();
    assert_eq!(
        rows.active_contexts().collect::<Vec<_>>(),
        vec![(0, 2), (2, 2)]
    );
}

/// Cached dynamic-NTK decoding is limited to the original window; static scalings are not.
#[test]
fn dynamic_ntk_contexts_are_bounded_by_the_original_window() {
    use crate::position::{DynamicNtkScaling, RopeScaling};

    let dynamic = RopeScaling::DynamicNtk(DynamicNtkScaling::new(2.0, 16).unwrap());
    assert!(ensure_static_rotary_context(dynamic, 16, "Test").is_ok());
    assert_same_error(
        ensure_static_rotary_context(dynamic, 17, "Test").unwrap_err(),
        Error::BackendCapability(
            "Test dynamic-NTK decoding beyond the original context requires re-rotating cached keys"
                .into(),
        ),
    );
    assert!(ensure_static_rotary_context(RopeScaling::None, usize::MAX, "Test").is_ok());
}

proptest! {
    /// Every scatter layout reproduces the former per-token concatenation exactly.
    #[test]
    fn scatter_matches_per_token_reference(
        rows in (1usize..10).prop_flat_map(|width| {
            prop::collection::vec(prop::collection::vec(any::<bool>(), width), 1..4)
        }),
        width in 1usize..4,
    ) {
        let device = Default::default();
        let sequence = rows[0].len();
        let tokens = Tensor::<TestBackend, 1, Int>::arange(0..(rows.len() * sequence) as i64, &device)
            .reshape([rows.len(), sequence]);
        let prepared = prepare_stateful_rows(
            &input(tokens.clone(), Some(mask_from_rows(&rows)), None),
            &device,
            &vec![0; rows.len()],
            sequence,
            &StatefulContract { vocabulary: rows.len() * sequence, ..decode_contract() },
        )
        .unwrap();
        for (row, columns) in prepared.valid_indices.iter().enumerate() {
            let Some(plan) = prepared.compact_row::<TestBackend>(row, &device).unwrap() else {
                prop_assert!(columns.is_empty());
                continue;
            };
            let expected_tokens = columns
                .iter()
                .map(|&column| (row * sequence + column) as i64)
                .collect::<Vec<_>>();
            prop_assert_eq!(int_values(plan.compact(&tokens)), expected_tokens);
            let compact = compact_values(columns.len(), width);
            prop_assert_eq!(
                floats(plan.scatter(compact.clone()).unwrap()),
                floats(reference_scatter(compact, sequence, columns))
            );
        }
    }
}

/// Scatter has an exact adjoint: every compact entry receives its output gradient.
#[cfg(feature = "autodiff")]
#[test]
fn scatter_is_differentiable() {
    use burn::backend::Autodiff;

    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    for columns in [vec![1, 2], vec![0, 2, 3]] {
        let row = CompactRow::<AdBackend>::new(0, &columns, 4, &device).unwrap();
        let compact = Tensor::<AdBackend, 3>::ones([1, columns.len(), 2], &device).require_grad();
        let weights =
            Tensor::<AdBackend, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device).reshape([1, 4, 1]);
        let gradients = (row.scatter(compact.clone()).unwrap() * weights)
            .sum()
            .backward();
        let gradient = compact
            .grad(&gradients)
            .expect("compact entries receive gradients")
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let expected = columns
            .iter()
            .flat_map(|&column| [(column + 1) as f32; 2])
            .collect::<Vec<_>>();
        assert_eq!(gradient, expected);
    }
}

/// Measures planned scatter against the former per-token concatenation.
///
/// Run with `cargo test --release scatter_benchmark -- --ignored --nocapture`.
/// Both paths produce identical values; the measurement reports wall time on
/// the Flex CPU backend, where kernel-launch savings are smallest.
#[test]
#[ignore = "timing benchmark; run explicitly"]
fn scatter_benchmark() {
    let device = Default::default();
    let sequence = 4096;
    let width = 64;
    let repetitions = 20;
    for (label, columns) in [
        ("dense", (0..sequence).collect::<Vec<_>>()),
        ("left-padded", (sequence / 4..sequence).collect::<Vec<_>>()),
        (
            "interior",
            (0..sequence).filter(|column| column % 7 != 3).collect(),
        ),
    ] {
        let plan = CompactRow::<TestBackend>::new(0, &columns, sequence, &device).unwrap();
        let compact = compact_values(columns.len(), width);
        let start = Instant::now();
        for _ in 0..repetitions {
            let _ = floats(reference_scatter(compact.clone(), sequence, &columns));
        }
        let reference = start.elapsed() / repetitions;
        let start = Instant::now();
        for _ in 0..repetitions {
            let _ = floats(plan.scatter(compact.clone()).unwrap());
        }
        let planned = start.elapsed() / repetitions;
        println!("{label}: per-token {reference:?}, planned {planned:?}");
    }
}
