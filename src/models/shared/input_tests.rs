//! Tests for stateless input validation, error precedence, and position numbering.

use proptest::prelude::*;

use super::*;
use crate::models::shared::test_support::{
    TestBackend, assert_same_error, int_values, ints, mask, mask_from_rows,
};

/// Error names used by every contract in these tests.
const NAMES: InputNames<'static> = InputNames {
    token_ids: "t.input_ids",
    token_mask: "t.token_mask",
    position_ids: "t.position_ids",
};

/// A contract over a vocabulary of 10 tokens and 8 positions.
fn contract() -> InputContract<'static> {
    InputContract::new(NAMES, "test parameters", 10).with_position_limit(8)
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

/// Validates `input` against `contract` on the default device.
fn validate(
    input: &CausalLmInput<TestBackend>,
    contract: &InputContract<'_>,
) -> Result<ValidatedInput<TestBackend>> {
    validate_input(input, &Default::default(), contract)
}

/// Unmasked inputs number physical columns and use the full width as context.
#[test]
fn dense_input_uses_physical_columns() {
    let validated = validate(
        &input(ints([[1, 2, 3], [4, 5, 6]]), None, None),
        &contract(),
    )
    .expect("valid input");
    assert_eq!(
        int_values(validated.require_positions("p").unwrap()),
        vec![0, 1, 2, 0, 1, 2]
    );
    assert_eq!(validated.evaluated_context, 3);
}

/// Left padding does not shift valid-token ordinals, and padding maps to zero.
#[test]
fn left_padding_uses_valid_token_ordinals() {
    let padded = mask([[false, false, true, true], [true, true, true, true]]);
    let validated = validate(
        &input(ints([[0, 0, 7, 8], [1, 2, 3, 4]]), Some(padded), None),
        &contract(),
    )
    .expect("valid input");
    assert_eq!(
        int_values(validated.positions.unwrap()),
        vec![0, 0, 0, 1, 0, 1, 2, 3]
    );
    assert_eq!(validated.evaluated_context, 4);
}

/// Explicit positions are validated as given, including padded columns.
#[test]
fn explicit_positions_are_kept_and_validated() {
    let padded = || Some(mask([[false, true]]));
    let validated = validate(
        &input(ints([[4, 5]]), padded(), Some(ints([[3, 7]]))),
        &contract(),
    )
    .unwrap();
    assert_eq!(int_values(validated.positions.unwrap()), vec![3, 7]);
    assert_eq!(validated.evaluated_context, 8);
    assert_same_error(
        validate(
            &input(ints([[4, 5]]), padded(), Some(ints([[-1, 7]]))),
            &contract(),
        )
        .unwrap_err(),
        Error::shape(
            "t.position_ids",
            "position values in 0..8",
            "observed range -1..=7",
        ),
    );
}

/// The composable mask check validates the extent eagerly and the rows lazily.
#[test]
fn composable_mask_check_defers_row_counts() {
    let mut checks = DeferredChecks::<TestBackend>::new();
    assert!(check_nonempty_mask(&mut checks, None, 1, 2, "x.mask").is_ok());
    assert!(check_nonempty_mask(&mut checks, Some(&mask([[true, false]])), 1, 2, "x.mask").is_ok());
    assert!(checks.run("test").is_ok());
    let mut checks = DeferredChecks::<TestBackend>::new();
    assert_same_error(
        check_nonempty_mask(&mut checks, Some(&mask([[true, false]])), 2, 1, "x.mask").unwrap_err(),
        Error::shape("x.mask", "[2, 1]", "[1, 2]"),
    );
    check_nonempty_mask(&mut checks, Some(&mask([[false, false]])), 1, 2, "x.mask").unwrap();
    assert!(checks.run("test").is_err());
}

/// Contracts without a position bound still derive the context extent.
#[test]
fn unbounded_positions_still_report_context() {
    let contract = InputContract::new(NAMES, "test parameters", 10);
    let validated = validate(
        &input(ints([[1, 2]]), None, Some(ints([[40, 41]]))),
        &contract,
    )
    .unwrap();
    assert_eq!(validated.evaluated_context, 42);
}

/// Every geometric violation reports the offending tensor.
#[test]
fn geometric_violations_are_shape_errors() {
    let empty = Tensor::<TestBackend, 2, Int>::from_data([[0_i64; 0]; 1], &Default::default());
    assert_same_error(
        validate(&input(empty, None, None), &contract()).unwrap_err(),
        Error::shape("t.input_ids", "non-empty [batch, sequence]", "[1, 0]"),
    );
    assert_same_error(
        validate(
            &input(ints([[1, 2]]), Some(mask([[true]])), None),
            &contract(),
        )
        .unwrap_err(),
        Error::shape("t.token_mask", "[1, 2]", "[1, 1]"),
    );
    assert_same_error(
        validate(
            &input(ints([[1, 2]]), None, Some(ints([[0], [1]]))),
            &contract(),
        )
        .unwrap_err(),
        Error::shape("t.position_ids", "[1, 2]", "[2, 1]"),
    );
}

/// Value checks run tokens, then mask rows, then positions; extents come first.
#[test]
fn value_errors_follow_documented_precedence() {
    let everything_wrong = input(
        ints([[11, 2]]),
        Some(mask([[false, false]])),
        Some(ints([[9, 9]])),
    );
    assert_same_error(
        validate(&everything_wrong, &contract()).unwrap_err(),
        Error::shape(
            "t.input_ids",
            "token values in 0..10",
            "observed range 2..=11",
        ),
    );
    let mask_and_positions = input(
        ints([[1, 2]]),
        Some(mask([[false, false]])),
        Some(ints([[9, 9]])),
    );
    assert_same_error(
        validate(&mask_and_positions, &contract()).unwrap_err(),
        Error::shape(
            "t.token_mask",
            "at least one valid token in every row",
            "a fully masked row",
        ),
    );
    let bad_extent_and_tokens = input(ints([[-1, 2]]), Some(mask([[true]])), None);
    assert!(matches!(
        validate(&bad_extent_and_tokens, &contract()),
        Err(Error::Shape { tensor, .. }) if tensor == "t.token_mask"
    ));
}

proptest! {
    /// Device ordinals equal a host cumulative count of valid tokens, zero at padding.
    #[test]
    fn valid_token_ordinals_match_host_reference(
        rows in (1usize..6).prop_flat_map(|width| {
            prop::collection::vec(prop::collection::vec(any::<bool>(), width), 1..4)
        })
    ) {
        let expected = rows
            .iter()
            .flat_map(|row| {
                let mut seen = 0_i64;
                row.iter().map(move |&valid| {
                    seen += i64::from(valid);
                    if valid { seen - 1 } else { 0 }
                })
            })
            .collect::<Vec<_>>();
        prop_assert_eq!(int_values(valid_token_ordinals(&mask_from_rows(&rows))), expected);
    }
}
