//! Dependency-free index arithmetic for tokenizer truncation and batch padding.
//!
//! Tokenization produces one row of IDs per input string. Pure integer logic
//! turns those rows into a model batch:
//!
//! - **Limits.** A present `max_length` must be positive, a fixed padding
//!   width must be positive, and `max_length` may not exceed that width
//!   ([`encoding_limit_violation`]).
//! - **Truncation budget.** Hugging Face tokenizers truncate the *content*
//!   before the post-processor inserts template special tokens (`<s>`, `</s>`,
//!   `[CLS]`, ...). With `r` reserved special tokens and a total limit `m`, the
//!   content keeps at most `m - r` IDs, so the template tokens always survive.
//!   A limit smaller than `r` cannot hold the template at all and is rejected.
//! - **Rectangular padding.** A row of `n` real tokens widened to `t >= n`
//!   positions receives `t - n` pad IDs on exactly one side. Its validity mask
//!   uses the crate-wide convention `true == real token`, so the mask holds
//!   exactly `n` `true` entries, contiguous and in the original order. A batch
//!   is padded to its longest row or a fixed width ([`pad_batch`]); without
//!   padding, every mask is all `true` ([`unpadded_masks`]).
//! - **Tensor layout.** A rectangular batch of `B` rows of width `S` becomes
//!   one row-major buffer of `B * S` entries ([`flatten_token_rows`],
//!   [`flatten_mask_rows`]) once every row is checked to have width `S`
//!   ([`first_ragged_row`]).
//!
//! # Verified behavior
//!
//! Every function here is checked in place by Verus through
//! `verus/tokenization.rs` (see `verus/README.md`). For all admissible inputs,
//! the proofs establish that the arithmetic never overflows or underflows,
//! that limits are rejected by exactly the first violated rule, that budgets
//! reserve exactly the template width, that padded rows and masks have the
//! target width, keep every real token at its shifted index, and mark exactly
//! the real positions as valid, that a padded batch is exactly the per-row
//! padding of every input (or names the first overlong row), and that the
//! flattened buffers put row `r`, column `c` at index `r * S + c`. Tokenizer
//! calls, error messages, and tensor conversion remain covered by the Rust
//! test suite.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Number of padding positions placed on each side of one encoded row.
///
/// At most one side is non-zero: rows are padded on the checkpoint's configured
/// side only, never split around the real tokens.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PaddingSplit {
    /// Pad IDs inserted before the first real token (left padding).
    pub(crate) before: usize,
    /// Pad IDs appended after the last real token (right padding).
    pub(crate) after: usize,
}

/// Returns how many content tokens fit beside `reserved` template tokens.
///
/// `max_length` bounds the complete post-processed row. The post-processor of
/// a checkpoint tokenizer adds a fixed number of special tokens to every
/// single-sequence encoding, so the content budget is `max_length - reserved`.
/// `None` means even an empty content sequence would exceed the limit.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(budget =>
    ensures
        reserved <= max_length ==> budget == Some((max_length - reserved) as usize),
        reserved > max_length ==> budget == None::<usize>,
))]
pub(crate) fn content_token_budget(max_length: usize, reserved: usize) -> Option<usize> {
    if reserved <= max_length {
        Some(max_length - reserved)
    } else {
        None
    }
}

/// Why a combination of encoding limits is rejected.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LimitViolation {
    /// `max_length` is zero, so no row could hold any token.
    ZeroMaxLength,
    /// A fixed padding width is zero.
    ZeroPaddingWidth,
    /// `max_length` exceeds the fixed padding width, so a truncated row might not fit it.
    MaxLengthExceedsWidth {
        /// The requested `max_length`.
        max_length: usize,
        /// The fixed padding width.
        width: usize,
    },
}

/// Checks `max_length` and a fixed padding width, reporting the first violated rule.
///
/// The rules, in order: a present `max_length` is positive, a fixed width is
/// positive, and `max_length` does not exceed a fixed width. Verified: the
/// result is exactly the first rule that fails, so accepted limits satisfy
/// `0 < max_length <= width` whenever both are present.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(violation =>
    ensures
        max_length == Some(0usize) ==> violation == Some(LimitViolation::ZeroMaxLength),
        max_length != Some(0usize) && fixed_width == Some(0usize)
            ==> violation == Some(LimitViolation::ZeroPaddingWidth),
        max_length != Some(0usize) && fixed_width != Some(0usize) ==> match (max_length, fixed_width) {
            (Some(maximum), Some(width)) => if maximum > width {
                violation == Some(LimitViolation::MaxLengthExceedsWidth { max_length: maximum, width })
            } else {
                violation.is_none()
            },
            _ => violation.is_none(),
        },
))]
pub(crate) fn encoding_limit_violation(
    max_length: Option<usize>,
    fixed_width: Option<usize>,
) -> Option<LimitViolation> {
    if let Some(0) = max_length {
        return Some(LimitViolation::ZeroMaxLength);
    }
    if let Some(0) = fixed_width {
        return Some(LimitViolation::ZeroPaddingWidth);
    }
    match (max_length, fixed_width) {
        (Some(maximum), Some(width)) if maximum > width => {
            Some(LimitViolation::MaxLengthExceedsWidth {
                max_length: maximum,
                width,
            })
        }
        _ => None,
    }
}

/// Splits the padding needed to widen a row of `length` real tokens to `target`.
///
/// The caller rejects rows longer than `target` before calling, because
/// padding never truncates silently. `pad_left` selects decoder-style left
/// padding; otherwise the padding follows the real tokens.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(split =>
    requires
        length <= target,
    ensures
        split.before + length + split.after == target,
        pad_left ==> split.after == 0,
        !pad_left ==> split.before == 0,
))]
pub(crate) fn padding_split(length: usize, target: usize, pad_left: bool) -> PaddingSplit {
    let amount = target - length;
    if pad_left {
        PaddingSplit {
            before: amount,
            after: 0,
        }
    } else {
        PaddingSplit {
            before: 0,
            after: amount,
        }
    }
}

/// Builds one padded token row: `split.before` pads, the real IDs, then `split.after` pads.
///
/// The row is written once into an exact-capacity allocation, so padding a
/// batch costs `O(batch * target)` with no intermediate reallocation.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(padded =>
    requires
        split.before + tokens.len() + split.after <= usize::MAX,
    ensures
        padded.len() == split.before + tokens.len() + split.after,
        forall|i: int| 0 <= i < split.before ==> padded[i] == pad_id,
        forall|i: int| 0 <= i < tokens.len() ==> padded[split.before + i] == tokens[i],
        forall|i: int| split.before + tokens.len() <= i < padded.len() ==> padded[i] == pad_id,
))]
pub(crate) fn pad_token_row(tokens: &[u32], pad_id: u32, split: PaddingSplit) -> Vec<u32> {
    let width = split.before + tokens.len() + split.after;
    let mut padded = Vec::with_capacity(width);
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= split.before,
            padded.len() == index,
            forall|i: int| 0 <= i < index ==> padded[i] == pad_id,
        decreases split.before - index,
    ))]
    while index < split.before {
        padded.push(pad_id);
        index += 1;
    }
    let mut token = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            token <= tokens.len(),
            padded.len() == split.before + token,
            forall|i: int| 0 <= i < split.before ==> padded[i] == pad_id,
            forall|i: int| 0 <= i < token ==> padded[split.before + i] == tokens[i],
        decreases tokens.len() - token,
    ))]
    while token < tokens.len() {
        padded.push(tokens[token]);
        token += 1;
    }
    let mut trailing = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            trailing <= split.after,
            split.before + tokens.len() + split.after <= usize::MAX,
            padded.len() == split.before + tokens.len() + trailing,
            forall|i: int| 0 <= i < split.before ==> padded[i] == pad_id,
            forall|i: int| 0 <= i < tokens.len() ==> padded[split.before + i] == tokens[i],
            forall|i: int| split.before + tokens.len() <= i < padded.len() ==> padded[i] == pad_id,
        decreases split.after - trailing,
    ))]
    while trailing < split.after {
        padded.push(pad_id);
        trailing += 1;
    }
    padded
}

/// Builds the `true == real token` mask matching [`pad_token_row`].
///
/// Exactly `length` entries are `true`, and they occupy the contiguous block
/// `[split.before, split.before + length)`, so the true-count of the mask is
/// the unpadded row length.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(mask =>
    requires
        split.before + length + split.after <= usize::MAX,
    ensures
        mask.len() == split.before + length + split.after,
        forall|i: int| 0 <= i < mask.len() ==>
            (mask[i] == (split.before <= i && i < split.before + length)),
))]
pub(crate) fn validity_mask(length: usize, split: PaddingSplit) -> Vec<bool> {
    let width = split.before + length + split.after;
    let real_end = split.before + length;
    let mut mask = Vec::with_capacity(width);
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= width,
            width == split.before + length + split.after,
            real_end == split.before + length,
            mask.len() == index,
            forall|i: int| 0 <= i < index ==>
                (mask[i] == (split.before <= i && i < real_end)),
        decreases width - index,
    ))]
    while index < width {
        mask.push(split.before <= index && index < real_end);
        index += 1;
    }
    mask
}

// Batch layout: rectangular padding, unpadded masks, and tensor flattening.

#[cfg(verus_keep_ghost)]
verus! {

/// `ids` and `mask` are `row` widened to `target` positions with `pad_id`.
///
/// All padding goes before the row when `pad_left` and after it otherwise;
/// every real token keeps its order at its shifted index, and the mask is
/// `true` exactly on the real block.
pub(crate) open spec fn is_padded_row(
    row: Seq<u32>,
    ids: Seq<u32>,
    mask: Seq<bool>,
    target: int,
    pad_id: u32,
    pad_left: bool,
) -> bool {
    let before = if pad_left { target - row.len() } else { 0 };
    &&& ids.len() == target
    &&& mask.len() == target
    &&& forall|i: int| 0 <= i < target ==> #[trigger] ids[i] == if before <= i < before + row.len() {
        row[i - before]
    } else {
        pad_id
    }
    &&& forall|i: int| 0 <= i < target ==> #[trigger] mask[i] == (before <= i && i < before + row.len())
}

/// The first `count` rows all have exactly `width` entries.
pub(crate) open spec fn rows_have_width<T>(rows: Seq<Vec<T>>, count: int, width: int) -> bool {
    forall|r: int| 0 <= r < count ==> #[trigger] rows[r].len() == width
}

/// The first `count` rows have at most `width` entries.
pub(crate) open spec fn rows_fit_width(rows: Seq<Vec<u32>>, count: int, width: int) -> bool {
    forall|r: int| 0 <= r < count ==> #[trigger] rows[r].len() <= width
}

/// Row `index / width`, column `index % width` of a row-major `rows.len() x width` layout.
pub(crate) open spec fn row_major_entry<T>(rows: Seq<Vec<T>>, width: int, index: int) -> T {
    rows[index / width]@[index % width]
}

/// `row * width + column` addresses row `row`, column `column` of a row-major layout.
pub(crate) proof fn lemma_row_major_index(row: int, column: int, width: int)
    requires
        0 <= row,
        0 <= column < width,
    ensures
        (row * width + column) / width == row,
        (row * width + column) % width == column,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
        row * width + column,
        width,
        row,
        column,
    );
}

} // verus!

/// Token rows and validity masks of a rectangular batch.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PaddedBatch {
    /// One row of token IDs per input, each widened to the target.
    pub(crate) token_ids: Vec<Vec<u32>>,
    /// One `true == real token` mask per input, aligned with `token_ids`.
    pub(crate) valid_tokens: Vec<Vec<bool>>,
}

/// Length of the longest row, or `None` for an empty batch.
///
/// Verified: the result bounds every row length and is attained by some row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(longest =>
    ensures
        longest.is_none() == (rows.len() == 0),
        longest.is_some() ==> rows_fit_width(rows@, rows.len() as int, longest.unwrap() as int),
        longest.is_some() ==> exists|r: int| 0 <= r < rows.len()
            && #[trigger] rows@[r].len() == longest.unwrap(),
))]
pub(crate) fn longest_row(rows: &[Vec<u32>]) -> Option<usize> {
    if rows.is_empty() {
        return None;
    }
    let mut longest = rows[0].len();
    let mut index = 1;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            1 <= index <= rows.len(),
            rows_fit_width(rows@, index as int, longest as int),
            exists|r: int| 0 <= r < index && #[trigger] rows@[r].len() == longest,
        decreases rows.len() - index,
    ))]
    while index < rows.len() {
        if rows[index].len() > longest {
            longest = rows[index].len();
        }
        index += 1;
    }
    Some(longest)
}

/// Pads every row to `target` and builds its mask, or names the first row longer than `target`.
///
/// Padding never truncates: `Err(index)` reports the first row that does not
/// fit, so the caller can name it. Verified: on success there is one padded
/// row and one mask per input, each exactly [`pad_token_row`] and
/// [`validity_mask`] of its row (the spec `is_padded_row`), so the batch is
/// rectangular with width `target`; an error index is the first row longer
/// than `target`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(padded =>
    ensures
        match padded {
            Ok(batch) => {
                &&& batch.token_ids.len() == rows.len()
                &&& batch.valid_tokens.len() == rows.len()
                &&& rows_fit_width(rows@, rows.len() as int, target as int)
                &&& forall|r: int| 0 <= r < rows.len() ==> #[trigger] is_padded_row(
                    rows@[r]@,
                    batch.token_ids@[r]@,
                    batch.valid_tokens@[r]@,
                    target as int,
                    pad_id,
                    pad_left,
                )
            },
            Err(index) => {
                &&& index < rows.len()
                &&& rows@[index as int].len() > target
                &&& rows_fit_width(rows@, index as int, target as int)
            },
        },
))]
pub(crate) fn pad_batch(
    rows: &[Vec<u32>],
    target: usize,
    pad_id: u32,
    pad_left: bool,
) -> Result<PaddedBatch, usize> {
    let mut token_ids: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
    let mut valid_tokens: Vec<Vec<bool>> = Vec::with_capacity(rows.len());
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= rows.len(),
            token_ids.len() == index,
            valid_tokens.len() == index,
            rows_fit_width(rows@, index as int, target as int),
            forall|r: int| 0 <= r < index ==> #[trigger] is_padded_row(
                rows@[r]@,
                token_ids@[r]@,
                valid_tokens@[r]@,
                target as int,
                pad_id,
                pad_left,
            ),
        decreases rows.len() - index,
    ))]
    while index < rows.len() {
        let row = &rows[index];
        if row.len() > target {
            return Err(index);
        }
        let split = padding_split(row.len(), target, pad_left);
        let ids = pad_token_row(row, pad_id, split);
        let mask = validity_mask(row.len(), split);
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(is_padded_row(row@, ids@, mask@, target as int, pad_id, pad_left));
        }
        token_ids.push(ids);
        valid_tokens.push(mask);
        index += 1;
    }
    Ok(PaddedBatch {
        token_ids,
        valid_tokens,
    })
}

/// All-`true` masks for rows kept at their own lengths (no padding requested).
///
/// Verified: one mask per row, as long as its row, with every entry `true`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(masks =>
    ensures
        masks.len() == rows.len(),
        forall|r: int| 0 <= r < rows.len() ==> #[trigger] masks@[r].len() == rows@[r].len(),
        forall|r: int, i: int| 0 <= r < rows.len() && 0 <= i < rows@[r].len()
            ==> #[trigger] masks@[r]@[i],
))]
pub(crate) fn unpadded_masks(rows: &[Vec<u32>]) -> Vec<Vec<bool>> {
    let mut masks: Vec<Vec<bool>> = Vec::with_capacity(rows.len());
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= rows.len(),
            masks.len() == index,
            forall|r: int| 0 <= r < index ==> #[trigger] masks@[r].len() == rows@[r].len(),
            forall|r: int, i: int| 0 <= r < index && 0 <= i < rows@[r].len()
                ==> #[trigger] masks@[r]@[i],
        decreases rows.len() - index,
    ))]
    while index < rows.len() {
        let mask = validity_mask(
            rows[index].len(),
            PaddingSplit {
                before: 0,
                after: 0,
            },
        );
        masks.push(mask);
        index += 1;
    }
    masks
}

/// Index of the first row whose token row or mask is not exactly `width` long.
///
/// Rows are paired like `zip`: only the first `min(token_ids.len(),
/// valid_tokens.len())` pairs are examined, so `BatchEncoding::to_causal_input`
/// first checks that both hold the same number of rows. Verified: `None`
/// means every examined pair has width `width`; `Some(r)` is the first pair
/// that does not.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(ragged =>
    ensures
        ({
            let pairs = if token_ids.len() <= valid_tokens.len() {
                token_ids.len() as int
            } else {
                valid_tokens.len() as int
            };
            match ragged {
                None => {
                    &&& rows_have_width(token_ids@, pairs, width as int)
                    &&& rows_have_width(valid_tokens@, pairs, width as int)
                },
                Some(row) => {
                    &&& row < pairs
                    &&& (token_ids@[row as int].len() != width
                        || valid_tokens@[row as int].len() != width)
                    &&& rows_have_width(token_ids@, row as int, width as int)
                    &&& rows_have_width(valid_tokens@, row as int, width as int)
                },
            }
        }),
))]
pub(crate) fn first_ragged_row(
    token_ids: &[Vec<u32>],
    valid_tokens: &[Vec<bool>],
    width: usize,
) -> Option<usize> {
    let pairs = token_ids.len().min(valid_tokens.len());
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= pairs,
            pairs <= token_ids.len(),
            pairs <= valid_tokens.len(),
            pairs == token_ids.len() || pairs == valid_tokens.len(),
            rows_have_width(token_ids@, row as int, width as int),
            rows_have_width(valid_tokens@, row as int, width as int),
        decreases pairs - row,
    ))]
    while row < pairs {
        if token_ids[row].len() != width || valid_tokens[row].len() != width {
            return Some(row);
        }
        row += 1;
    }
    None
}

/// Concatenates token rows in row-major order, widening each ID to `i64`.
///
/// Every row is copied whole, so the result is the plain concatenation for
/// any input; `width` sizes the allocation. Verified: for rows of equal
/// width `width`, entry `r * width + c` is row `r`, column `c`, and the
/// length is `rows.len() * width`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(flat =>
    requires
        rows_have_width(rows@, rows.len() as int, width as int),
        rows.len() * width <= usize::MAX,
    ensures
        flat.len() == rows.len() * width,
        forall|index: int| 0 <= index < flat.len()
            ==> #[trigger] flat@[index] == row_major_entry(rows@, width as int, index) as i64,
))]
pub(crate) fn flatten_token_rows(rows: &[Vec<u32>], width: usize) -> Vec<i64> {
    let mut flat: Vec<i64> = Vec::with_capacity(rows.len() * width);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows.len(),
            rows_have_width(rows@, rows.len() as int, width as int),
            rows.len() * width <= usize::MAX,
            flat.len() == row * width,
            forall|index: int| 0 <= index < flat.len()
                ==> #[trigger] flat@[index] == row_major_entry(rows@, width as int, index) as i64,
        decreases rows.len() - row,
    ))]
    while row < rows.len() {
        let tokens = &rows[row];
        let mut column = 0;
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(rows@[row as int].len() == width);
            assert((row + 1) * width <= rows.len() * width) by (nonlinear_arith)
                requires row + 1 <= rows.len();
            assert((row + 1) * width == row * width + width) by (nonlinear_arith);
        }
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                column <= tokens.len(),
                tokens.len() == width,
                tokens@ == rows@[row as int]@,
                row < rows.len(),
                (row + 1) * width <= usize::MAX,
                (row + 1) * width == row * width + width,
                flat.len() == row * width + column,
                forall|index: int| 0 <= index < flat.len()
                    ==> #[trigger] flat@[index] == row_major_entry(rows@, width as int, index) as i64,
            decreases tokens.len() - column,
        ))]
        while column < tokens.len() {
            #[cfg(verus_keep_ghost)]
            proof! {
                lemma_row_major_index(row as int, column as int, width as int);
            }
            flat.push(tokens[column] as i64);
            column += 1;
        }
        row += 1;
    }
    flat
}

/// Concatenates mask rows in row-major order.
///
/// Every row is copied whole, so the result is the plain concatenation for
/// any input; `width` sizes the allocation. Verified: for rows of equal width `width`, entry
/// `r * width + c` is row `r`, column `c`, and the length is
/// `rows.len() * width`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(flat =>
    requires
        rows_have_width(rows@, rows.len() as int, width as int),
        rows.len() * width <= usize::MAX,
    ensures
        flat.len() == rows.len() * width,
        forall|index: int| 0 <= index < flat.len()
            ==> #[trigger] flat@[index] == row_major_entry(rows@, width as int, index),
))]
pub(crate) fn flatten_mask_rows(rows: &[Vec<bool>], width: usize) -> Vec<bool> {
    let mut flat: Vec<bool> = Vec::with_capacity(rows.len() * width);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows.len(),
            rows_have_width(rows@, rows.len() as int, width as int),
            rows.len() * width <= usize::MAX,
            flat.len() == row * width,
            forall|index: int| 0 <= index < flat.len()
                ==> #[trigger] flat@[index] == row_major_entry(rows@, width as int, index),
        decreases rows.len() - row,
    ))]
    while row < rows.len() {
        let mask = &rows[row];
        let mut column = 0;
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(rows@[row as int].len() == width);
            assert((row + 1) * width <= rows.len() * width) by (nonlinear_arith)
                requires row + 1 <= rows.len();
            assert((row + 1) * width == row * width + width) by (nonlinear_arith);
        }
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                column <= mask.len(),
                mask.len() == width,
                mask@ == rows@[row as int]@,
                row < rows.len(),
                (row + 1) * width <= usize::MAX,
                (row + 1) * width == row * width + width,
                flat.len() == row * width + column,
                forall|index: int| 0 <= index < flat.len()
                    ==> #[trigger] flat@[index] == row_major_entry(rows@, width as int, index),
            decreases mask.len() - column,
        ))]
        while column < mask.len() {
            #[cfg(verus_keep_ghost)]
            proof! {
                lemma_row_major_index(row as int, column as int, width as int);
            }
            flat.push(mask[column]);
            column += 1;
        }
        row += 1;
    }
    flat
}

#[cfg(test)]
#[path = "layout_tests.rs"]
mod tests;
