//! Dependency-free size arithmetic of the chat-template sandbox.
//!
//! The sandbox (see the `sandbox` module) checks every size *before* it
//! allocates, so the arithmetic that computes those sizes must itself never
//! overflow: a wrapped product would let `'ab' * 2^63` pass as a tiny string.
//! Every function here either saturates at `usize::MAX` (a bound that then
//! fails any limit check) or reports overflow as `None`. The module also holds
//! the integer cores of the render budget's bookkeeping: the per-render
//! [`RenderLimits`], the text allowance, the byte charge of copied items, the
//! write check of the bounded output buffers, and the fuel of the namespace
//! cleanup.
//!
//! # Verified behavior
//!
//! Every function is checked in place by Verus (see `verus/README.md`). For
//! all inputs the proofs establish that no operation overflows and that each
//! result is exact:
//!
//! - the limits are their formulas below saturation and `usize::MAX` above
//!   it; [`render_limits`] derives every per-render limit from them, so the
//!   cumulative budget always admits one maximal value;
//! - a budget charge succeeds exactly when the new total stays within the
//!   budget and then adds exactly the charge;
//! - [`text_allowance`] is exactly the largest text a reservation accepts:
//!   while the counter is within its budget, `bytes <= allowance` holds
//!   exactly when `bytes` fits the per-value limit and the charge succeeds;
//! - repetition sizes are exact products or `None` on overflow, and copy
//!   charges are the saturated product `items * values * item_bytes`;
//! - the replacement and indentation bounds are their saturated formulas;
//! - a bounded buffer accepts a write exactly when the grown length stays
//!   within its limit (below `usize::MAX`), so its length never passes it;
//! - the cleanup fuel is the saturated `32 * (assignments + groups + 1)`.
//!
//! Whether the replacement and indentation formulas dominate the text
//! MiniJinja's `replace` and `indent` produce is a property of string
//! semantics, covered by the sandbox's Rust tests.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// `value` clamped to `usize::MAX`: the result of a saturating computation.
pub(crate) open spec fn saturated(value: int) -> int {
    if value > usize::MAX {
        usize::MAX as int
    } else {
        value
    }
}

/// Whether the sandbox's text reservation of `bytes` succeeds after `produced` bytes.
///
/// `RenderBudget::reserve_text` first checks the per-value limit and then
/// charges the budget with [`charged_total`], which accepts exactly when
/// `produced + bytes <= total_bytes`.
pub(crate) open spec fn text_reservable(limits: RenderLimits, produced: usize, bytes: usize) -> bool {
    bytes <= limits.value_bytes && produced + bytes <= limits.total_bytes
}

/// Matches a replacement may make: every possible match, capped by `count`.
///
/// An empty pattern matches before every character and at the end
/// (`characters + 1`, saturated); otherwise at most `length / from` matches fit.
pub(crate) open spec fn replacement_matches(
    length: usize,
    characters: usize,
    from: usize,
    count: Option<usize>,
) -> int {
    let possible = if from == 0 {
        saturated(characters + 1)
    } else {
        length as int / from as int
    };
    match count {
        Some(count) => if count < possible {
            count as int
        } else {
            possible
        },
        None => possible,
    }
}

} // verus!

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
/// Fewest items one sequence may always hold, `2^20` (24 MiB of `Value`s).
///
/// A sequence may hold as many items as one value may hold bytes, so a list
/// of the characters (or the split parts) of any string a render may build
/// fits; for an empty context that is exactly this floor, the size of the
/// smallest output limit. A chat has at most a few thousand messages.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const MIN_SEQUENCE_ITEMS: usize = 1 << 20;

/// Size limits of one render, derived from the size of its template context.
///
/// With `c` the serialized context bytes, the rendered prompt may hold
/// `1 MiB + 4c` bytes ([`Self::output_bytes`]), and so may any single string
/// ([`Self::value_bytes`]); a sequence may hold as many items as a string may
/// hold bytes (at least `2^20`); and every bounded operation together may
/// produce `16` times the prompt limit ([`Self::total_bytes`]), counting
/// sequence items as 24 bytes each. See [`render_limits`].
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RenderLimits {
    /// Largest rendered prompt, in bytes.
    pub(crate) output_bytes: usize,
    /// Largest string one operation may produce, in bytes.
    pub(crate) value_bytes: usize,
    /// Longest sequence one operation may produce, in items.
    pub(crate) sequence_items: usize,
    /// Cumulative bytes all bounded operations may produce in one render.
    pub(crate) total_bytes: usize,
}

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

/// The limits of a render whose template context serializes to `context_bytes`.
///
/// Every product saturates, so even an absurd context size yields valid
/// (merely huge) limits rather than an overflow. Verified: each limit is its
/// formula (saturated), and the cumulative budget is at least the per-value
/// limit, so a fresh render can always produce one maximal value.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(limits =>
    ensures
        limits.output_bytes == saturated(OUTPUT_FLOOR_BYTES + 4 * context_bytes),
        limits.value_bytes == limits.output_bytes,
        limits.sequence_items == if limits.output_bytes >= MIN_SEQUENCE_ITEMS {
            limits.output_bytes
        } else {
            MIN_SEQUENCE_ITEMS
        },
        limits.total_bytes == saturated(16 * limits.output_bytes),
        limits.total_bytes >= limits.value_bytes,
        limits.sequence_items >= MIN_SEQUENCE_ITEMS,
))]
pub(crate) fn render_limits(context_bytes: usize) -> RenderLimits {
    let output_bytes = output_limit(context_bytes);
    RenderLimits {
        output_bytes,
        value_bytes: output_bytes,
        sequence_items: output_bytes.max(MIN_SEQUENCE_ITEMS),
        total_bytes: total_budget(output_bytes),
    }
}

/// Largest text an operation may still produce: the per-value limit or the remaining budget.
///
/// This is `min(value_bytes, total_bytes - produced)`, the remainder
/// saturating at zero. Verified: while the counter is within its budget,
/// `bytes <= allowance` holds exactly when a text reservation of `bytes`
/// succeeds (`text_reservable`: `bytes` fits the per-value limit and
/// [`charged_total`] accepts it); so printing into an allowance-sized buffer never builds text
/// that the following reservation then refuses, and vice versa.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(allowance =>
    ensures
        allowance <= limits.value_bytes,
        produced <= limits.total_bytes ==> allowance <= limits.total_bytes - produced,
        produced <= limits.total_bytes ==> (allowance == limits.value_bytes
            || allowance == limits.total_bytes - produced),
        produced > limits.total_bytes ==> allowance == 0,
        produced <= limits.total_bytes ==> forall|bytes: usize|
            (bytes <= allowance) == #[trigger] text_reservable(limits, produced, bytes),
))]
pub(crate) fn text_allowance(limits: RenderLimits, produced: usize) -> usize {
    limits
        .value_bytes
        .min(limits.total_bytes.saturating_sub(produced))
}

/// Bytes charged for `items` new items of `values_per_item` values of `item_bytes` each.
///
/// Verified: exactly the product `items * values_per_item * item_bytes`,
/// saturated at `usize::MAX`, so a charge never undercounts what it copies.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bytes =>
    ensures
        bytes == saturated(items * values_per_item * item_bytes),
))]
pub(crate) fn copy_charge(items: usize, values_per_item: usize, item_bytes: usize) -> usize {
    #[cfg(verus_keep_ghost)]
    proof! {
        let pairs = items * values_per_item;
        assert(items * values_per_item * item_bytes == pairs * item_bytes);
        if item_bytes == 0 {
            assert(pairs * item_bytes == 0) by (nonlinear_arith)
                requires item_bytes == 0;
        } else if pairs > usize::MAX {
            assert(pairs * item_bytes >= pairs) by (nonlinear_arith)
                requires item_bytes > 0, pairs > usize::MAX;
            assert(usize::MAX * item_bytes >= usize::MAX) by (nonlinear_arith)
                requires item_bytes > 0;
        }
    }
    items
        .saturating_mul(values_per_item)
        .saturating_mul(item_bytes)
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
///
/// Verified: the bound is exactly `length + matches * max(to - from, 0)`,
/// saturated, where `matches` is the possible match count (`characters + 1`,
/// saturated, for an empty pattern; `length / from` otherwise) lowered to
/// `count` when that is smaller; so it is never below `length`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bound =>
    ensures
        bound == saturated(length + replacement_matches(length, characters, from, count) * (
            if to > from { to - from } else { 0 })),
        bound >= length,
))]
pub(crate) fn replacement_bound(
    length: usize,
    characters: usize,
    from: usize,
    to: usize,
    count: Option<usize>,
) -> usize {
    // An empty pattern matches between every pair of characters and at both
    // ends; any other pattern matches at most `length / from` times.
    let mut matches = match length.checked_div(from) {
        Some(quotient) => quotient,
        None if characters == usize::MAX => usize::MAX,
        None => characters + 1,
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
///
/// Verified: the bound is exactly `lines * (width + 1) + length`, saturated
/// (and `usize::MAX` for a width of `usize::MAX`), so never below `length`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bound =>
    ensures
        bound >= length,
        width < usize::MAX ==> bound == saturated(lines * (width + 1) + length),
        width == usize::MAX ==> bound == usize::MAX,
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

/// Whether a buffer of `current` bytes may grow by `added` bytes within `limit`.
///
/// This is the write check of the sandbox's bounded buffers (the rendered
/// prompt and printed values): the sum saturates, so it never wraps.
/// Verified: below `usize::MAX`, a write is accepted exactly when the grown
/// length stays within the limit, so a buffer that starts within its limit
/// never passes it.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(accepted =>
    ensures
        accepted == (saturated(current + added) <= limit),
        limit < usize::MAX ==> accepted == (current + added <= limit),
))]
pub(crate) fn appended_within(current: usize, added: usize, limit: usize) -> bool {
    current.saturating_add(added) <= limit
}

/// Fuel granted to the namespace cleanup: `fuel_per_step` per assignment, per group, and once more.
///
/// Verified: exactly `fuel_per_step * (assignments + groups + 1)`, with the
/// step count saturated at `usize::MAX` and the product at `u64::MAX`, so the
/// cleanup always receives at least its accounted work (until saturation).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fuel =>
    ensures
        fuel == if saturated(assignments + groups + 1) * fuel_per_step > u64::MAX {
            u64::MAX as int
        } else {
            saturated(assignments + groups + 1) * fuel_per_step
        },
))]
pub(crate) fn cleanup_fuel(assignments: usize, groups: usize, fuel_per_step: u64) -> u64 {
    let steps = assignments.saturating_add(groups).saturating_add(1);
    (steps as u64).saturating_mul(fuel_per_step)
}

#[cfg(test)]
#[path = "limits_tests.rs"]
mod tests;
