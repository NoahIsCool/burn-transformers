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

/// Limit validation reports the first violated rule, in order.
#[test]
fn encoding_limits_report_the_first_violation() {
    assert_eq!(encoding_limit_violation(None, None), None);
    assert_eq!(encoding_limit_violation(Some(4), Some(4)), None);
    assert_eq!(encoding_limit_violation(Some(3), None), None);
    assert_eq!(encoding_limit_violation(None, Some(3)), None);
    assert_eq!(
        encoding_limit_violation(Some(0), Some(0)),
        Some(LimitViolation::ZeroMaxLength)
    );
    assert_eq!(
        encoding_limit_violation(Some(1), Some(0)),
        Some(LimitViolation::ZeroPaddingWidth)
    );
    assert_eq!(
        encoding_limit_violation(Some(5), Some(4)),
        Some(LimitViolation::MaxLengthExceedsWidth {
            max_length: 5,
            width: 4
        })
    );
}

/// Batches pad every row on one side or name the first row that does not fit.
#[test]
fn batches_pad_every_row_or_name_the_first_overlong_row() {
    let rows = vec![vec![1, 2, 3], vec![4], vec![]];
    assert_eq!(longest_row(&rows), Some(3));
    assert_eq!(longest_row(&[]), None);
    let left = pad_batch(&rows, 3, 0, true).unwrap();
    assert_eq!(
        left.token_ids,
        [vec![1, 2, 3], vec![0, 0, 4], vec![0, 0, 0]]
    );
    assert_eq!(
        left.valid_tokens,
        [
            vec![true, true, true],
            vec![false, false, true],
            vec![false, false, false]
        ]
    );
    let right = pad_batch(&rows, 4, 9, false).unwrap();
    assert_eq!(
        right.token_ids,
        [vec![1, 2, 3, 9], vec![4, 9, 9, 9], vec![9, 9, 9, 9]]
    );
    assert_eq!(pad_batch(&rows, 2, 0, false), Err(0));
    assert_eq!(
        pad_batch(&[vec![1], vec![1, 2, 3], vec![1, 2, 3]], 2, 0, true),
        Err(1)
    );
    assert_eq!(
        pad_batch(&[], 2, 0, true),
        Ok(PaddedBatch {
            token_ids: Vec::new(),
            valid_tokens: Vec::new()
        })
    );
    assert_eq!(
        unpadded_masks(&rows),
        [vec![true, true, true], vec![true], vec![]]
    );
}

/// Rectangularity checks find the first ragged pair; flattening is row-major.
#[test]
fn rectangular_batches_flatten_row_major() {
    let ids = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let masks = vec![vec![true, false], vec![true, true], vec![false, true]];
    assert_eq!(first_ragged_row(&ids, &masks, 2), None);
    assert_eq!(first_ragged_row(&ids, &masks, 3), Some(0));
    assert_eq!(
        first_ragged_row(&ids, &[vec![true, true], vec![true]], 2),
        Some(1)
    );
    assert_eq!(
        first_ragged_row(&[vec![1, 2], vec![3]], &masks[..1], 2),
        None
    );
    assert_eq!(flatten_token_rows(&ids, 2), [1, 2, 3, 4, 5, 6]);
    assert_eq!(
        flatten_mask_rows(&masks, 2),
        [true, false, true, true, false, true]
    );
    assert!(flatten_token_rows(&[], 7).is_empty());
}

proptest! {
    /// Batch padding agrees with padding each row on its own, and flattening with `Iterator::flatten`.
    #[test]
    fn batch_layouts_match_per_row_references(
        rows in proptest::collection::vec(proptest::collection::vec(any::<u32>(), 0..8), 0..6),
        extra in 0_usize..4,
        pad_left in any::<bool>(),
        pad_id in any::<u32>(),
    ) {
        let target = rows.iter().map(Vec::len).max().unwrap_or(0) + extra;
        prop_assert_eq!(longest_row(&rows), rows.iter().map(Vec::len).max());
        let batch = pad_batch(&rows, target, pad_id, pad_left).unwrap();
        for (index, row) in rows.iter().enumerate() {
            let split = padding_split(row.len(), target, pad_left);
            prop_assert_eq!(&batch.token_ids[index], &pad_token_row(row, pad_id, split));
            prop_assert_eq!(&batch.valid_tokens[index], &validity_mask(row.len(), split));
        }
        prop_assert_eq!(first_ragged_row(&batch.token_ids, &batch.valid_tokens, target), None);
        let flat = flatten_token_rows(&batch.token_ids, target);
        let expected = batch.token_ids.iter().flatten().map(|id| i64::from(*id)).collect::<Vec<_>>();
        prop_assert_eq!(flat, expected);
        let flat_mask = flatten_mask_rows(&batch.valid_tokens, target);
        let expected_mask = batch.valid_tokens.iter().flatten().copied().collect::<Vec<_>>();
        prop_assert_eq!(flat_mask, expected_mask);
        if target > 0 && !rows.is_empty() {
            let shorter = target - 1;
            let first = rows.iter().position(|row| row.len() > shorter);
            prop_assert_eq!(pad_batch(&rows, shorter, pad_id, pad_left).err(), first);
        }
    }

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
