//! Tests for fused validation readback, check precedence, and device/extent errors.

use burn::tensor::{DType, Int, Tensor};
use proptest::prelude::*;

use super::*;
use crate::models::shared::test_support::{
    TestBackend, assert_same_error, ints, mask, mask_from_rows,
};

/// A satisfied check list returns exact ranges, counts, and the context extent.
#[test]
fn passing_checks_report_ranges_counts_and_context() {
    let tokens = ints([[3, 1], [0, 2]]);
    let positions = ints([[0, 1], [5, 2]]);
    let mut checks = DeferredChecks::new();
    let token_range = checks.values_below(&tokens, 4, "t.input_ids", "token");
    let counts = checks.row_counts(&mask([[true, false], [true, true]]), "t.mask");
    let position_range = checks.values_below(&positions, 8, "t.position_ids", "position");
    let readout = checks.run("test").expect("valid values");
    assert_eq!(readout.range(token_range), (0, 3));
    assert_eq!(readout.min_row_count(counts), 1);
    assert_eq!(readout.range(position_range), (0, 5));
    assert_eq!(
        readout.context(position_range, "t.position_ids").unwrap(),
        6
    );
}

/// The first registered failing check determines the error.
#[test]
fn failing_checks_report_in_registration_order() {
    let tokens = ints([[9, 1]]);
    let positions = ints([[-1, 0]]);
    let empty = mask([[false, false]]);

    let mut checks = DeferredChecks::new();
    checks.values_below(&tokens, 4, "t.input_ids", "token");
    checks.row_counts(&empty, "t.mask");
    checks.values_below(&positions, 8, "t.position_ids", "position");
    assert_same_error(
        checks.run("test").unwrap_err(),
        Error::shape(
            "t.input_ids",
            "token values in 0..4",
            "observed range 1..=9",
        ),
    );

    let mut checks = DeferredChecks::new();
    checks.row_counts(&empty, "t.mask");
    checks.values_below(&tokens, 4, "t.input_ids", "token");
    assert_same_error(
        checks.run("test").unwrap_err(),
        Error::shape(
            "t.mask",
            "at least one valid token in every row",
            "a fully masked row",
        ),
    );

    let mut checks = DeferredChecks::new();
    checks.values_below(&positions, 8, "t.position_ids", "position");
    assert_same_error(
        checks.run("test").unwrap_err(),
        Error::shape(
            "t.position_ids",
            "position values in 0..8",
            "observed range -1..=0",
        ),
    );
}

/// A negative or unrepresentable maximum has no context extent.
#[test]
fn context_rejects_maxima_without_an_extent() {
    let readout = Readout {
        values: vec![-4, -2, 0, i64::MAX],
    };
    assert!(matches!(
        readout.context(RangeId(0), "t.position_ids"),
        Err(Error::Configuration { .. })
    ));
    assert_eq!(
        readout.context(RangeId(2), "t.position_ids").ok(),
        usize::try_from(i64::MAX).ok().map(|maximum| maximum + 1)
    );
}

/// An empty check list performs no backend work and returns an empty readout.
#[test]
fn empty_check_list_reads_nothing() {
    let readout = DeferredChecks::<TestBackend>::new().run("test").unwrap();
    assert_eq!(readout, Readout { values: Vec::new() });
    assert!(
        read_int_tensors::<TestBackend>(Vec::new(), "test")
            .unwrap()
            .is_empty()
    );
}

/// Tensors of different integer dtypes are widened losslessly before concatenation.
#[test]
fn mixed_integer_dtypes_are_read_together() {
    let device = Default::default();
    let narrow = Tensor::<TestBackend, 1, Int>::from_data([7, -3], &device);
    let wide = Tensor::<TestBackend, 1, Int>::from_data([1_i64 << 40], (&device, DType::I64));
    assert_ne!(narrow.dtype(), wide.dtype());
    assert_eq!(wide.dtype(), DType::I64);
    assert_eq!(
        read_int_tensors(vec![narrow, wide], "test").unwrap(),
        vec![7, -3, 1_i64 << 40]
    );
}

/// Device, emptiness, and grid helpers produce the shared error convention.
#[test]
fn layout_helpers_report_contextual_errors() {
    assert!(ensure_same_device("x", &1, &1, "owner").is_ok());
    assert_same_error(
        ensure_same_device("x", &1, &2, "owner").unwrap_err(),
        Error::BackendCapability("x and owner are on different devices".into()),
    );
    assert!(ensure_nonempty(1, 1, "x").is_ok());
    assert_same_error(
        ensure_nonempty(0, 3, "x").unwrap_err(),
        Error::shape("x", "non-empty [batch, sequence]", "[0, 3]"),
    );
    assert!(ensure_nonempty(2, 0, "x").is_err());
    assert!(ensure_grid([2, 3], 2, 3, "m").is_ok());
    assert_same_error(
        ensure_grid([3, 2], 2, 3, "m").unwrap_err(),
        Error::shape("m", "[2, 3]", "[3, 2]"),
    );
}

/// A whole mask shares the readback, and later reductions keep their own slots.
#[test]
fn mask_columns_share_the_readback_with_reductions() {
    let tokens = ints([[3, 1, 2], [0, 2, 1]]);
    let positions = ints([[0, 4, 1], [2, 5, 3]]);
    let padded = mask([[false, true, true], [true, false, true]]);
    let mut checks = DeferredChecks::new();
    let token_range = checks.values_below(&tokens, 4, "t.input_ids", "token");
    let columns = checks.mask_columns(&padded);
    let counts = checks.row_counts(&padded, "t.mask");
    let position_range = checks.values_below(&positions, 8, "t.position_ids", "position");
    let readout = checks.run("test").expect("valid values");
    assert_eq!(readout.range(token_range), (0, 3));
    assert_eq!(readout.valid_columns(columns), vec![vec![1, 2], vec![0, 2]]);
    assert_eq!(readout.min_row_count(counts), 2);
    assert_eq!(readout.range(position_range), (0, 5));
}

/// Registering a mask adds no predicate: fully masked rows need `row_counts`.
#[test]
fn mask_columns_alone_accept_empty_rows() {
    let mut checks = DeferredChecks::<TestBackend>::new();
    let columns = checks.mask_columns(&mask([[false, false], [true, false]]));
    let readout = checks.run("test").expect("no predicate registered");
    assert_eq!(readout.valid_columns(columns), vec![vec![], vec![0]]);
}

/// A zero-width mask handle yields one empty column list per row.
#[test]
fn zero_width_mask_has_empty_rows() {
    let readout = Readout { values: Vec::new() };
    let id = MaskId {
        slot: 0,
        batch: 2,
        sequence: 0,
    };
    assert_eq!(readout.valid_columns(id), vec![Vec::<usize>::new(); 2]);
}

proptest! {
    /// Columns read with the checks equal a host filter, and reductions
    /// registered around the mask keep their exact values.
    #[test]
    fn mask_columns_match_host_reference(
        rows in (1usize..5, 1usize..9).prop_flat_map(|(batch, sequence)| {
            prop::collection::vec(prop::collection::vec(any::<bool>(), sequence), batch)
        }),
        low in 0i64..100,
        span in 0i64..100,
    ) {
        let values = Tensor::<TestBackend, 1, Int>::from_data([low, low + span], &Default::default());
        let mut checks = DeferredChecks::new();
        let before = checks.values_below(&values, usize::MAX, "t.before", "value");
        let columns = checks.mask_columns(&mask_from_rows(&rows));
        let after = checks.values_below(&values, usize::MAX, "t.after", "value");
        let readout = checks.run("test").expect("values are in range");
        let expected = rows
            .iter()
            .map(|row| (0..row.len()).filter(|&column| row[column]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        prop_assert_eq!(readout.valid_columns(columns), expected);
        prop_assert_eq!(readout.range(before), (low, low + span));
        prop_assert_eq!(readout.range(after), (low, low + span));
    }
}
