//! Tests for truncation budgets and rectangular padding arithmetic.

use proptest::prelude::*;

use super::*;

/// The budget reserves exactly the template width and rejects narrower limits.
#[test]
fn content_budget_reserves_template_tokens() {
    assert_eq!(content_token_budget(512, 1), Some(511));
    assert_eq!(content_token_budget(8, 2), Some(6));
    assert_eq!(content_token_budget(2, 2), Some(0));
    assert_eq!(content_token_budget(5, 0), Some(5));
    assert_eq!(content_token_budget(1, 2), None);
    assert_eq!(content_token_budget(0, 1), None);
    assert_eq!(content_token_budget(usize::MAX, usize::MAX), Some(0));
}

/// Padding goes entirely to the configured side, including the zero-width case.
#[test]
fn padding_split_uses_only_the_configured_side() {
    assert_eq!(
        padding_split(2, 5, true),
        PaddingSplit {
            before: 3,
            after: 0
        }
    );
    assert_eq!(
        padding_split(2, 5, false),
        PaddingSplit {
            before: 0,
            after: 3
        }
    );
    for pad_left in [true, false] {
        assert_eq!(
            padding_split(4, 4, pad_left),
            PaddingSplit {
                before: 0,
                after: 0
            }
        );
    }
}

/// Rows and masks agree for left, right, and no padding, and for empty content.
#[test]
fn padded_rows_and_masks_match_reference_layouts() {
    let left = padding_split(2, 4, true);
    assert_eq!(pad_token_row(&[7, 8], 0, left), [0, 0, 7, 8]);
    assert_eq!(validity_mask(2, left), [false, false, true, true]);

    let right = padding_split(2, 4, false);
    assert_eq!(pad_token_row(&[7, 8], 0, right), [7, 8, 0, 0]);
    assert_eq!(validity_mask(2, right), [true, true, false, false]);

    let none = padding_split(3, 3, true);
    assert_eq!(pad_token_row(&[1, 2, 3], 9, none), [1, 2, 3]);
    assert_eq!(validity_mask(3, none), [true; 3]);

    let only_padding = padding_split(0, 2, false);
    assert_eq!(pad_token_row(&[], 5, only_padding), [5, 5]);
    assert_eq!(validity_mask(0, only_padding), [false, false]);
    assert!(pad_token_row(&[], 5, padding_split(0, 0, true)).is_empty());
}

proptest! {
    /// For every admissible row, padding preserves content and the mask's true-count.
    #[test]
    fn padding_preserves_tokens_and_marks_exactly_the_real_block(
        tokens in proptest::collection::vec(any::<u32>(), 0..32),
        extra in 0_usize..32,
        pad_left in any::<bool>(),
        pad_id in any::<u32>(),
    ) {
        let target = tokens.len() + extra;
        let split = padding_split(tokens.len(), target, pad_left);
        let row = pad_token_row(&tokens, pad_id, split);
        let mask = validity_mask(tokens.len(), split);

        prop_assert_eq!(row.len(), target);
        prop_assert_eq!(mask.len(), target);
        prop_assert_eq!(mask.iter().filter(|valid| **valid).count(), tokens.len());
        let real = row
            .iter()
            .zip(&mask)
            .filter_map(|(token, valid)| valid.then_some(*token))
            .collect::<Vec<_>>();
        prop_assert_eq!(&real, &tokens);
        prop_assert!(row.iter().zip(&mask).all(|(token, valid)| *valid || *token == pad_id));
        // Left padding puts every invalid slot first; right padding puts them last.
        let first_valid = mask.iter().position(|valid| *valid).unwrap_or(target);
        if pad_left {
            prop_assert!(mask[..first_valid].iter().all(|valid| !valid));
            prop_assert!(mask[first_valid..].iter().all(|valid| *valid));
        } else {
            prop_assert!(mask[..tokens.len()].iter().all(|valid| *valid));
            prop_assert!(mask[tokens.len()..].iter().all(|valid| !valid));
        }
    }

    /// The budget is the checked difference, so it never wraps.
    #[test]
    fn content_budget_is_the_checked_difference(max_length in any::<usize>(), reserved in any::<usize>()) {
        prop_assert_eq!(content_token_budget(max_length, reserved), max_length.checked_sub(reserved));
    }
}
