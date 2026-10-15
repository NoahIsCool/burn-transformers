//! Dependency-free index arithmetic for tokenizer truncation and batch padding.
//!
//! Tokenization produces one row of IDs per input string. Two pieces of pure
//! integer logic turn those rows into a model batch:
//!
//! - **Truncation budget.** Hugging Face tokenizers truncate the *content*
//!   before the post-processor inserts template special tokens (`<s>`, `</s>`,
//!   `[CLS]`, ...). With `r` reserved special tokens and a total limit `m`, the
//!   content keeps at most `m - r` IDs, so the template tokens always survive.
//!   A limit smaller than `r` cannot hold the template at all and is rejected.
//! - **Rectangular padding.** A row of `n` real tokens widened to `t >= n`
//!   positions receives `t - n` pad IDs on exactly one side. Its validity mask
//!   uses the crate-wide convention `true == real token`, so the mask holds
//!   exactly `n` `true` entries, contiguous and in the original order.
//!
//! # Verified behavior
//!
//! Every function here is checked in place by Verus through
//! `verus/tokenization.rs` (see `verus/README.md`). For all admissible inputs,
//! the proofs establish that the arithmetic never overflows or underflows,
//! that budgets reserve exactly the template width, and that padded rows and
//! masks have the target width, keep every real token at its shifted index,
//! and mark exactly the real positions as valid. Tokenizer calls, error
//! messages, and tensor conversion remain covered by the Rust test suite.

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

#[cfg(test)]
#[path = "layout_tests.rs"]
mod tests;
