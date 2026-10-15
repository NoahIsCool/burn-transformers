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
    assert!(validated.fully_valid);
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
    assert!(!validated.fully_valid);
}

/// A mask without padding still reports a fully valid batch.
#[test]
fn all_true_mask_is_fully_valid() {
    let validated = validate(
        &input(ints([[1, 2]]), Some(mask([[true, true]])), None),
        &contract(),
    )
    .unwrap();
    assert!(validated.fully_valid);
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

/// Kept padded positions are validated; zero-filled padded positions are inert.
#[test]
fn padded_explicit_positions_follow_policy() {
    let sentinel = || {
        input(
            ints([[0, 4, 5]]),
            Some(mask([[false, true, true]])),
            Some(ints([[-1, 0, 1]])),
        )
    };
    assert_same_error(
        validate(&sentinel(), &contract()).unwrap_err(),
        Error::shape(
            "t.position_ids",
            "position values in 0..8",
            "observed range -1..=1",
        ),
    );
    let zero_filled = validate(
        &sentinel(),
        &contract().with_padded_positions(PaddedPositions::ZeroFill),
    )
    .unwrap();
    assert_eq!(int_values(zero_filled.positions.unwrap()), vec![0, 0, 1]);
    assert_eq!(zero_filled.evaluated_context, 2);
}

/// Staged position validation returns the context extent from one readback.
#[test]
fn staged_position_validation_returns_context() {
    assert_eq!(
        validate_position_values(&ints([[2, 6]]), 7, "x.pos").unwrap(),
        7
    );
    assert_same_error(
        validate_position_values(&ints([[2, 7]]), 7, "x.pos").unwrap_err(),
        Error::shape("x.pos", "position values in 0..7", "observed range 2..=7"),
    );
}

/// Contracts without a position bound pass positions through uninspected.
#[test]
fn unbounded_positions_are_not_inspected() {
    let contract = InputContract::new(NAMES, "test parameters", 10);
    let validated = validate(
        &input(ints([[1, 2]]), None, Some(ints([[-40, 41]]))),
        &contract,
    )
    .unwrap();
    assert_eq!(int_values(validated.positions.unwrap()), vec![-40, 41]);
    assert_eq!(validated.evaluated_context, 0);
}

/// Physical numbering ignores the mask, as GPT-2 and GPT-NeoX require.
#[test]
fn physical_numbering_keeps_padded_columns() {
    let contract = contract().with_implicit_positions(ImplicitPositions::PhysicalColumns);
    let validated = validate(
        &input(ints([[0, 5, 6]]), Some(mask([[false, true, true]])), None),
        &contract,
    )
    .unwrap();
    assert_eq!(int_values(validated.positions.unwrap()), vec![0, 1, 2]);
    assert_eq!(validated.evaluated_context, 3);
    assert!(!validated.fully_valid);
}

/// Position-free contracts materialize nothing unless positions are explicit.
#[test]
fn unused_numbering_resolves_only_explicit_positions() {
    let contract = contract().with_implicit_positions(ImplicitPositions::Unused);
    let implicit = validate(&input(ints([[1, 2]]), None, None), &contract).unwrap();
    assert!(implicit.positions.is_none());
    assert_eq!(implicit.evaluated_context, 0);
    assert!(matches!(
        implicit.require_positions("p"),
        Err(Error::Configuration { .. })
    ));

    let explicit = validate(
        &input(ints([[1, 2]]), None, Some(ints([[3, 4]]))),
        &contract,
    )
    .unwrap();
    assert_eq!(int_values(explicit.positions.unwrap()), vec![3, 4]);
    assert_eq!(explicit.evaluated_context, 5);
    assert_same_error(
        validate(
            &input(ints([[1, 2]]), None, Some(ints([[3, 9]]))),
            &contract,
        )
        .unwrap_err(),
        Error::shape(
            "t.position_ids",
            "position values in 0..8",
            "observed range 3..=9",
        ),
    );
}

/// Standalone helpers validate tokens and masks with one readback each.
#[test]
fn standalone_helpers_share_error_convention() {
    assert!(validate_token_values(&ints([[0, 9]]), 10, "x.ids").is_ok());
    assert_same_error(
        validate_token_values(&ints([[0, 10]]), 10, "x.ids").unwrap_err(),
        Error::shape("x.ids", "token values in 0..10", "observed range 0..=10"),
    );
    assert!(validate_nonempty_mask::<TestBackend>(None, 1, 1, "x.mask").is_ok());
    assert!(validate_nonempty_mask(Some(&mask([[true, false]])), 1, 2, "x.mask").is_ok());
    assert_same_error(
        validate_nonempty_mask(Some(&mask([[true, false]])), 2, 1, "x.mask").unwrap_err(),
        Error::shape("x.mask", "[2, 1]", "[1, 2]"),
    );
    assert!(validate_nonempty_mask(Some(&mask([[false, false]])), 1, 2, "x.mask").is_err());
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
