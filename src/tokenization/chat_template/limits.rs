//! Dependency-free size arithmetic of the chat-template sandbox.
//!
//! The sandbox (see the `sandbox` module) checks every size *before* it
//! allocates, so the arithmetic that computes those sizes must itself never
//! overflow: a wrapped product would let `'ab' * 2^63` pass as a tiny string.
//! Every function here either saturates at `usize::MAX` (a bound that then
//! fails any limit check) or reports overflow as `None`.
//!
//! # Verified behavior
//!
//! Every function is checked in place by Verus through `verus/robustness.rs`
//! (see `verus/README.md`). For all inputs the proofs establish that no
//! operation overflows; that the limits are at least their floors and exact
//! below saturation; that a budget charge succeeds exactly when the new total
//! stays within the budget and then adds exactly the charge; that repetition
//! sizes are exact products or `None` on overflow; and that the replacement
//! and indentation bounds are at least the input length and exact below
//! saturation. Whether these bounds dominate the text MiniJinja's `replace`
//! and `indent` produce is a property of string semantics, covered by the
//! sandbox's Rust tests.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Smallest rendered-prompt limit, whatever the context size: 1 MiB.
///
/// Encoding is the costliest consumer of a rendered prompt: the Hugging Face
/// tokenizer keeps per-token strings, offsets, and masks, so encoding costs
/// about 120 bytes of memory per prompt byte for six-byte words (a 16 MiB
/// prompt of `hello ` took 1.9 GB) and about 220 for one-character words (a
/// 4 MiB prompt of `a ` took 911 MiB). One MiB is still about a thousand
/// times the text any published template adds to its messages; longer
/// prompts come from the context, which the second term scales with.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const OUTPUT_FLOOR_BYTES: usize = 1024 * 1024;
/// Rendered prompt bytes allowed per byte of serialized template context.
///
/// Templates print each message once or twice (content plus a tool-call or
/// reasoning echo); four times the context leaves room for markup.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const OUTPUT_BYTES_PER_CONTEXT_BYTE: usize = 4;
/// Cumulative production allowed per byte of the rendered-prompt limit.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const TOTAL_BYTES_PER_OUTPUT_BYTE: usize = 16;

/// The rendered-prompt limit for a context of `context_bytes`: `1 MiB + 4 * context_bytes`.
///
/// Saturates at `usize::MAX` instead of overflowing.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(limit =>
    ensures
        limit >= OUTPUT_FLOOR_BYTES,
        OUTPUT_FLOOR_BYTES + 4 * context_bytes <= usize::MAX
            ==> limit == OUTPUT_FLOOR_BYTES + 4 * context_bytes,
        OUTPUT_FLOOR_BYTES + 4 * context_bytes > usize::MAX ==> limit == usize::MAX,
))]
pub(crate) fn output_limit(context_bytes: usize) -> usize {
    if context_bytes > (usize::MAX - OUTPUT_FLOOR_BYTES) / OUTPUT_BYTES_PER_CONTEXT_BYTE {
        usize::MAX
    } else {
        OUTPUT_FLOOR_BYTES + context_bytes * OUTPUT_BYTES_PER_CONTEXT_BYTE
    }
}

/// The cumulative production budget for an output limit: `16 * output_bytes`, saturating.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(budget =>
    ensures
        budget >= output_bytes,
        16 * output_bytes <= usize::MAX ==> budget == 16 * output_bytes,
        16 * output_bytes > usize::MAX ==> budget == usize::MAX,
))]
pub(crate) fn total_budget(output_bytes: usize) -> usize {
    if output_bytes > usize::MAX / TOTAL_BYTES_PER_OUTPUT_BYTE {
        usize::MAX
    } else {
        output_bytes * TOTAL_BYTES_PER_OUTPUT_BYTE
    }
}

/// The produced total after charging `bytes`, or `None` when it would pass `total`.
///
/// This is the update rule of the render budget's counter: it keeps the
/// invariant `produced <= total` and never wraps.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(charged =>
    ensures
        produced + bytes <= total ==> charged == Some((produced + bytes) as usize),
        produced + bytes > total ==> charged == None::<usize>,
))]
pub(crate) fn charged_total(produced: usize, bytes: usize, total: usize) -> Option<usize> {
    if bytes <= total && produced <= total - bytes {
        Some(produced + bytes)
    } else {
        None
    }
}

/// Bytes of `count` repetitions of a `length`-byte string, or `None` on overflow.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bytes =>
    ensures
        length * count <= usize::MAX ==> bytes == Some((length * count) as usize),
        length * count > usize::MAX ==> bytes == None::<usize>,
))]
pub(crate) fn repeated_size(length: usize, count: usize) -> Option<usize> {
    length.checked_mul(count)
}

/// Upper bound in bytes of replacing up to `count` matches of a `from`-byte pattern by `to` bytes.
///
/// The text has `length` bytes and `characters` characters. An empty pattern
/// matches before every character and at the end (`characters + 1` matches);
/// otherwise at most `length / from` matches fit. Each match grows the text by
/// `to - from` bytes when `to` is longer. Saturates at `usize::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bound =>
    ensures
        bound >= length,
))]
pub(crate) fn replacement_bound(
    length: usize,
    characters: usize,
    from: usize,
    to: usize,
    count: Option<usize>,
) -> usize {
    let mut matches = if from == 0 {
        if characters == usize::MAX {
            usize::MAX
        } else {
            characters + 1
        }
    } else {
        length / from
    };
    // A `match` rather than a let-chain: Verus checks this body with an older edition.
    matches = match count {
        Some(count) if count < matches => count,
        _ => matches,
    };
    let growth = to.saturating_sub(from);
    match repeated_size(matches, growth) {
        Some(added) if added <= usize::MAX - length => length + added,
        _ => usize::MAX,
    }
}

/// Upper bound in bytes of indenting `lines` lines of `length`-byte text by `width` spaces.
///
/// Every line gains at most `width` spaces and a newline. Saturates at `usize::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bound =>
    ensures
        bound >= length,
        width < usize::MAX && lines * (width + 1) + length <= usize::MAX
            ==> bound == lines * (width + 1) + length,
))]
pub(crate) fn indentation_bound(length: usize, lines: usize, width: usize) -> usize {
    if width == usize::MAX {
        return usize::MAX;
    }
    match repeated_size(lines, width + 1) {
        Some(padding) if padding <= usize::MAX - length => padding + length,
        _ => usize::MAX,
    }
}

#[cfg(test)]
#[path = "limits_tests.rs"]
mod tests;
