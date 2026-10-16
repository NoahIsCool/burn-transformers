//! Dependency-free integer cores of model-configuration validation.
//!
//! Configuration loading mixes serde, heap-allocated diagnostics, and
//! floating-point checks with a small amount of integer logic that decides
//! whether a checkpoint's shapes are consistent. This module isolates that
//! logic so Verus can check it in place (`verus/config.rs`, see
//! `verus/README.md`); the callers in the parent module and in [`super::rope`]
//! translate each verdict into the same contextual error as before.
//!
//! * [`mrope_section_fault`]: multimodal RoPE sections are non-empty, positive,
//!   and their complex pairs exactly cover the rotary width,
//!   `2 * sum(sections) == rotary_width`, with overflow treated as a mismatch.
//! * [`divisibility_fault`]: grouped heads need non-zero counts and an exact
//!   quotient `numerator = groups * denominator`.
//! * [`token_id_fault`]: a token-ID field is non-empty and indexes the
//!   embedding table.
//! * [`original_context_fits`] and [`resolve_original_context`]: the trained
//!   context `L` of extrapolating RoPE variants is non-zero, at most
//!   `max_position_embeddings`, and resolved from nested, top-level, and
//!   fallback sources with the documented precedence.
//! * [`infinity_token_starts`]: the lexer behind the legacy `Infinity`
//!   normalization finds exactly the `Infinity` tokens outside JSON strings.
//!
//! # Verified properties
//!
//! Every function here is verified against an exact specification: each
//! verdict holds exactly when its condition does, and faults follow the order
//! of the original checks. The lexer is specified by `lexer_state`, the
//! string/escape state after a byte prefix, and returns every token start
//! outside a string, in increasing order and without overlap.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Sum of `values[0..end]`.
pub open spec fn prefix_sum(values: Seq<usize>, end: int) -> int
    decreases end,
{
    if end <= 0 {
        0
    } else {
        prefix_sum(values, end - 1) + values[end - 1]
    }
}

/// Prefix sums of unsigned values never decrease.
proof fn lemma_prefix_sum_monotone(values: Seq<usize>, start: int, end: int)
    requires
        0 <= start <= end <= values.len(),
    ensures
        prefix_sum(values, start) <= prefix_sum(values, end),
    decreases end - start,
{
    if start < end {
        lemma_prefix_sum_monotone(values, start, end - 1);
    }
}

/// `Infinity` occupies `bytes[start..start + 8]`.
pub open spec fn spec_infinity_at(bytes: Seq<u8>, start: int) -> bool {
    &&& 0 <= start
    &&& start + 8 <= bytes.len()
    &&& bytes[start] == 73u8
    &&& bytes[start + 1] == 110u8
    &&& bytes[start + 2] == 102u8
    &&& bytes[start + 3] == 105u8
    &&& bytes[start + 4] == 110u8
    &&& bytes[start + 5] == 105u8
    &&& bytes[start + 6] == 116u8
    &&& bytes[start + 7] == 121u8
}

/// JSON string state `(in_string, escaped)` after reading `bytes[0..end]`.
///
/// A `"` outside a string opens one; inside, a backslash escapes the next byte
/// and an unescaped `"` closes the string. Multi-byte UTF-8 sequences contain
/// only bytes of at least `0x80`, so reading bytes instead of characters gives
/// the same states at every character boundary.
pub open spec fn lexer_state(bytes: Seq<u8>, end: int) -> (bool, bool)
    decreases end,
{
    if end <= 0 {
        (false, false)
    } else {
        let previous = lexer_state(bytes, end - 1);
        let byte = bytes[end - 1];
        if previous.0 {
            if previous.1 {
                (true, false)
            } else if byte == 92u8 {
                (true, true)
            } else if byte == 34u8 {
                (false, false)
            } else {
                (true, false)
            }
        } else if byte == 34u8 {
            (true, false)
        } else {
            (false, false)
        }
    }
}

/// Reading a token outside a string stays outside a string after each of its bytes.
proof fn lemma_token_stays_outside_strings(bytes: Seq<u8>, start: int)
    requires
        spec_infinity_at(bytes, start),
        !lexer_state(bytes, start).0,
    ensures
        forall|offset: int|
            1 <= offset <= 8 ==> #[trigger] lexer_state(bytes, start + offset) == (false, false),
{
    assert(lexer_state(bytes, start + 1) == (false, false));
    assert(lexer_state(bytes, start + 2) == (false, false));
    assert(lexer_state(bytes, start + 3) == (false, false));
    assert(lexer_state(bytes, start + 4) == (false, false));
    assert(lexer_state(bytes, start + 5) == (false, false));
    assert(lexer_state(bytes, start + 6) == (false, false));
    assert(lexer_state(bytes, start + 7) == (false, false));
    assert(lexer_state(bytes, start + 8) == (false, false));
}

/// The trained context resolved before the zero check, following the precedence table.
///
/// A per-layer-type mapping reads only its nested value and falls back to
/// `max_position_embeddings`. A flat mapping prefers the top-level value on
/// a conflict (Transformers 5.2), then the nested value, then the fallback.
pub open spec fn spec_resolved_original(
    nested: Option<usize>,
    top_level: Option<usize>,
    max_position_embeddings: usize,
    per_layer_type: bool,
) -> int {
    if per_layer_type {
        match nested {
            Some(value) => value as int,
            None => max_position_embeddings as int,
        }
    } else {
        match (nested, top_level) {
            (_, Some(top)) => top as int,
            (Some(value), None) => value as int,
            (None, None) => max_position_embeddings as int,
        }
    }
}

/// Whether two configured sources disagree in a way the precedence table reports.
pub open spec fn original_sources_conflict(
    nested: Option<usize>,
    top_level: Option<usize>,
    max_position_embeddings: usize,
    per_layer_type: bool,
) -> bool {
    match top_level {
        Some(top) => if per_layer_type {
            top as int != spec_resolved_original(
                nested,
                top_level,
                max_position_embeddings,
                per_layer_type,
            )
        } else {
            match nested {
                Some(value) => value != top,
                None => false,
            }
        },
        None => false,
    }
}

} // verus!

/// Why multimodal RoPE sections were rejected, in check order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MropeSectionFault {
    /// The section list is empty or contains a zero width.
    EmptyOrZero,
    /// Twice the section sum differs from the rotary width or overflows.
    WidthMismatch,
}

/// Checks that multimodal RoPE sections tile the rotary width with complex pairs.
///
/// Each section counts complex pairs (two channels each) of one position
/// axis, so the sections must be non-empty, positive, and satisfy
/// `2 * sum(sections) == rotary_width`. A sum or doubling that overflows
/// `usize` cannot equal a `usize` width and is a mismatch.
///
/// Verified: `None` exactly in that case; otherwise empty or zero sections are
/// reported before a width mismatch.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fault =>
    ensures
        match fault {
            None => {
                &&& sections.len() > 0
                &&& forall|index: int| 0 <= index < sections.len() ==> #[trigger] sections[index] > 0
                &&& 2 * prefix_sum(sections@, sections.len() as int) == rotary_width
            },
            Some(MropeSectionFault::EmptyOrZero) => {
                sections.len() == 0
                    || exists|index: int| 0 <= index < sections.len() && #[trigger] sections[index] == 0
            },
            Some(MropeSectionFault::WidthMismatch) => {
                &&& sections.len() > 0
                &&& forall|index: int| 0 <= index < sections.len() ==> #[trigger] sections[index] > 0
                &&& 2 * prefix_sum(sections@, sections.len() as int) != rotary_width
            },
        }
))]
pub(crate) fn mrope_section_fault(
    sections: &[usize],
    rotary_width: usize,
) -> Option<MropeSectionFault> {
    if sections.is_empty() {
        return Some(MropeSectionFault::EmptyOrZero);
    }
    let mut index = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= sections.len(),
            forall|earlier: int| 0 <= earlier < index ==> #[trigger] sections[earlier] > 0,
        decreases sections.len() - index,
    ))]
    while index < sections.len() {
        if sections[index] == 0 {
            return Some(MropeSectionFault::EmptyOrZero);
        }
        index += 1;
    }
    let mut total = 0usize;
    let mut summed = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            summed <= sections.len(),
            sections.len() > 0,
            forall|earlier: int| 0 <= earlier < sections.len() ==> #[trigger] sections[earlier] > 0,
            total == prefix_sum(sections@, summed as int),
        decreases sections.len() - summed,
    ))]
    while summed < sections.len() {
        let Some(next) = total.checked_add(sections[summed]) else {
            #[cfg(verus_keep_ghost)]
            proof! {
                // The full sum is at least this overflowing prefix.
                lemma_prefix_sum_monotone(sections@, summed + 1, sections.len() as int);
            }
            return Some(MropeSectionFault::WidthMismatch);
        };
        total = next;
        summed += 1;
    }
    match total.checked_mul(2) {
        Some(width) if width == rotary_width => None,
        _ => Some(MropeSectionFault::WidthMismatch),
    }
}

/// Why a grouped-head relationship was rejected, in check order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DivisibilityFault {
    /// The numerator (for example query heads) is zero.
    ZeroNumerator,
    /// The denominator (for example key/value heads) is zero.
    ZeroDenominator,
    /// The denominator does not divide the numerator.
    Indivisible,
}

/// Checks that `denominator` groups `numerator` exactly, as grouped-query attention requires.
///
/// Verified: `None` exactly when both counts are non-zero and the denominator
/// divides the numerator, and then `numerator = (numerator / denominator) *
/// denominator` with at least one member per group; otherwise the first
/// failed check in the order zero numerator, zero denominator, remainder.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fault =>
    ensures
        match fault {
            None => {
                &&& numerator > 0 && denominator > 0
                &&& numerator % denominator == 0
                &&& (numerator / denominator) * denominator == numerator
                &&& numerator / denominator >= 1
            },
            Some(DivisibilityFault::ZeroNumerator) => numerator == 0,
            Some(DivisibilityFault::ZeroDenominator) => numerator > 0 && denominator == 0,
            Some(DivisibilityFault::Indivisible) => {
                numerator > 0 && denominator > 0 && numerator % denominator != 0
            },
        }
))]
pub(crate) fn divisibility_fault(
    numerator: usize,
    denominator: usize,
) -> Option<DivisibilityFault> {
    if numerator == 0 {
        return Some(DivisibilityFault::ZeroNumerator);
    }
    if denominator == 0 {
        return Some(DivisibilityFault::ZeroDenominator);
    }
    if !numerator.is_multiple_of(denominator) {
        return Some(DivisibilityFault::Indivisible);
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        // `n = d * (n / d) + n % d` with a zero remainder, and a positive
        // multiple of `d` has a positive quotient.
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
            numerator as int,
            denominator as int,
        );
        assert((numerator / denominator) * denominator == numerator
            && numerator / denominator >= 1) by (nonlinear_arith)
            requires
                numerator == denominator * (numerator / denominator) + numerator % denominator,
                numerator % denominator == 0,
                numerator > 0,
                denominator > 0,
        ;
    }
    None
}

/// Why a token-ID field was rejected, in check order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenIdFault {
    /// The field lists no token ID.
    Empty,
    /// Some ID is not a row of the embedding table.
    OutOfVocabulary,
}

/// Checks that a token-ID field is non-empty and indexes rows `0..vocab_size`.
///
/// Verified: `None` exactly when `ids` is non-empty and every ID is below
/// `vocab_size`; otherwise emptiness is reported before an out-of-range ID.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fault =>
    ensures
        match fault {
            None => {
                &&& ids.len() > 0
                &&& forall|index: int| 0 <= index < ids.len() ==> #[trigger] ids[index] < vocab_size
            },
            Some(TokenIdFault::Empty) => ids.len() == 0,
            Some(TokenIdFault::OutOfVocabulary) => {
                &&& ids.len() > 0
                &&& exists|index: int| 0 <= index < ids.len() && #[trigger] ids[index] >= vocab_size
            },
        }
))]
pub(crate) fn token_id_fault(ids: &[usize], vocab_size: usize) -> Option<TokenIdFault> {
    if ids.is_empty() {
        return Some(TokenIdFault::Empty);
    }
    let mut index = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= ids.len(),
            forall|earlier: int| 0 <= earlier < index ==> #[trigger] ids[earlier] < vocab_size,
        decreases ids.len() - index,
    ))]
    while index < ids.len() {
        if ids[index] >= vocab_size {
            return Some(TokenIdFault::OutOfVocabulary);
        }
        index += 1;
    }
    None
}

/// Whether a trained context `original` is usable with `max_position_embeddings`.
///
/// Verified: exactly `0 < original <= max_position_embeddings`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fits =>
    ensures
        fits == (0 < original && original <= max_position_embeddings),
))]
pub(crate) fn original_context_fits(original: usize, max_position_embeddings: usize) -> bool {
    original != 0 && original <= max_position_embeddings
}

/// Why the trained context of a RoPE translation could not be resolved, in check order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OriginalContextFault {
    /// The nested `original_max_position_embeddings` is zero.
    NestedZero,
    /// A per-layer-type mapping ignores a top-level value that differs from its layer's.
    LayerConflict {
        /// The unread top-level value.
        top_level: usize,
        /// The layer's nested value or `max_position_embeddings`.
        resolved: usize,
    },
    /// A flat mapping's nested and top-level values disagree.
    FlatConflict {
        /// The nested value.
        nested: usize,
        /// The top-level value.
        top_level: usize,
    },
    /// The resolved context is zero.
    Zero,
}

/// Resolves the trained context `L` from nested, top-level, and fallback sources.
///
/// See `spec_resolved_original` for the precedence. With
/// `reject_conflicts`, any disagreement the precedence table reports is a
/// fault; without it, the documented source wins silently, as in
/// Transformers 5.2.
///
/// Verified: the result is the resolved value exactly when it is positive and
/// no reported conflict applies; otherwise the faults follow the order nested
/// zero, conflict, resolved zero, each carrying the conflicting values.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Ok(original) => {
                &&& nested != Some(0usize)
                &&& !(reject_conflicts && original_sources_conflict(
                    nested,
                    top_level,
                    max_position_embeddings,
                    per_layer_type,
                ))
                &&& original == spec_resolved_original(
                    nested,
                    top_level,
                    max_position_embeddings,
                    per_layer_type,
                )
                &&& original > 0
            },
            Err(OriginalContextFault::NestedZero) => nested == Some(0usize),
            Err(OriginalContextFault::LayerConflict { top_level: top, resolved }) => {
                &&& reject_conflicts && per_layer_type
                &&& nested != Some(0usize)
                &&& top_level == Some(top)
                &&& resolved == spec_resolved_original(
                    nested,
                    top_level,
                    max_position_embeddings,
                    per_layer_type,
                )
                &&& top != resolved
            },
            Err(OriginalContextFault::FlatConflict { nested: value, top_level: top }) => {
                &&& reject_conflicts && !per_layer_type
                &&& nested == Some(value) && value != 0
                &&& top_level == Some(top)
                &&& value != top
            },
            Err(OriginalContextFault::Zero) => {
                &&& nested != Some(0usize)
                &&& !(reject_conflicts && original_sources_conflict(
                    nested,
                    top_level,
                    max_position_embeddings,
                    per_layer_type,
                ))
                &&& spec_resolved_original(
                    nested,
                    top_level,
                    max_position_embeddings,
                    per_layer_type,
                ) == 0
            },
        }
))]
pub(crate) fn resolve_original_context(
    nested: Option<usize>,
    top_level: Option<usize>,
    max_position_embeddings: usize,
    per_layer_type: bool,
    reject_conflicts: bool,
) -> Result<usize, OriginalContextFault> {
    if nested == Some(0) {
        return Err(OriginalContextFault::NestedZero);
    }
    let resolved = if per_layer_type {
        // `standardize_rope_params` (per-layer branch) only `setdefault`s
        // `max_position_embeddings`; the top-level value is never read.
        let resolved = match nested {
            Some(value) => value,
            None => max_position_embeddings,
        };
        if let Some(top) = top_level {
            if reject_conflicts && top != resolved {
                return Err(OriginalContextFault::LayerConflict {
                    top_level: top,
                    resolved,
                });
            }
        }
        resolved
    } else {
        // A disagreement resolves to the top-level value, as Transformers 5.2 does.
        match (nested, top_level) {
            (Some(value), Some(top)) => {
                if reject_conflicts && value != top {
                    return Err(OriginalContextFault::FlatConflict {
                        nested: value,
                        top_level: top,
                    });
                }
                top
            }
            (Some(value), None) => value,
            (None, Some(top)) => top,
            (None, None) => max_position_embeddings,
        }
    };
    if resolved == 0 {
        return Err(OriginalContextFault::Zero);
    }
    Ok(resolved)
}

/// Byte length of the legacy `Infinity` literal.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const INFINITY_TOKEN_LENGTH: usize = 8;

/// Whether `bytes[start..start + 8]` is the ASCII literal `Infinity`.
///
/// Verified: exactly `spec_infinity_at(bytes, start)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
        found == spec_infinity_at(bytes@, start as int),
))]
fn infinity_at(bytes: &[u8], start: usize) -> bool {
    start <= bytes.len()
        && bytes.len() - start >= INFINITY_TOKEN_LENGTH
        && bytes[start] == b'I'
        && bytes[start + 1] == b'n'
        && bytes[start + 2] == b'f'
        && bytes[start + 3] == b'i'
        && bytes[start + 4] == b'n'
        && bytes[start + 5] == b'i'
        && bytes[start + 6] == b't'
        && bytes[start + 7] == b'y'
}

/// Lists the byte offsets of every `Infinity` token outside JSON strings.
///
/// The scan tracks whether it is inside a string literal and whether the
/// previous string byte was an escaping backslash; outside strings it records
/// a token and skips its eight bytes. Tokens are ASCII, so every reported
/// offset and offset `+ 8` is a UTF-8 character boundary of a `str`.
///
/// Verified: the offsets are strictly increasing and at least eight apart;
/// each starts an `Infinity` token outside a string (`lexer_state`); and
/// every such token start in the input is listed.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(starts =>
    ensures
        forall|token: int| 0 <= token < starts.len() ==> {
            &&& spec_infinity_at(bytes@, #[trigger] starts[token] as int)
            &&& !lexer_state(bytes@, starts[token] as int).0
        },
        forall|token: int| 0 < token < starts.len()
            ==> starts[token - 1] + INFINITY_TOKEN_LENGTH <= #[trigger] starts[token],
        forall|position: int|
            0 <= position < bytes.len() && #[trigger] spec_infinity_at(bytes@, position)
                && !lexer_state(bytes@, position).0 ==> starts@.contains(position as usize),
))]
pub(crate) fn infinity_token_starts(bytes: &[u8]) -> Vec<usize> {
    let mut starts: Vec<usize> = Vec::new();
    let mut index = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= bytes.len(),
            in_string == lexer_state(bytes@, index as int).0,
            escaped == lexer_state(bytes@, index as int).1,
            forall|token: int| 0 <= token < starts.len() ==> {
                &&& spec_infinity_at(bytes@, #[trigger] starts[token] as int)
                &&& !lexer_state(bytes@, starts[token] as int).0
                &&& starts[token] + INFINITY_TOKEN_LENGTH <= index
            },
            forall|token: int| 0 < token < starts.len()
                ==> starts[token - 1] + INFINITY_TOKEN_LENGTH <= #[trigger] starts[token],
            forall|position: int|
                0 <= position < index && #[trigger] spec_infinity_at(bytes@, position)
                    && !lexer_state(bytes@, position).0 ==> starts@.contains(position as usize),
        decreases bytes.len() - index,
    ))]
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            // Inside a string no token is recorded; only the escape state moves.
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
        } else if byte == b'"' {
            // A quote is not the token's first byte, and it opens a string.
            in_string = true;
            index += 1;
        } else if infinity_at(bytes, index) {
            #[cfg(verus_keep_ghost)]
            proof_decl! {
                let ghost before = starts@;
                let ghost start = index as int;
            }
            #[cfg(verus_keep_ghost)]
            proof! {
                lemma_token_stays_outside_strings(bytes@, start);
            }
            starts.push(index);
            index += INFINITY_TOKEN_LENGTH;
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(lexer_state(bytes@, index as int) == (false, false));
                assert(starts@[starts.len() - 1] == start);
                // Earlier tokens keep their witnesses, the new one is last, and
                // no token starts inside this one: only its first byte is `I`.
                assert forall|position: int|
                    0 <= position < index && #[trigger] spec_infinity_at(bytes@, position)
                        && !lexer_state(bytes@, position).0 implies starts@.contains(
                        position as usize,
                    ) by {
                    if position < start {
                        let witness = choose|witness: int|
                            0 <= witness < before.len() && before[witness] == position as usize;
                        assert(starts@[witness] == position as usize);
                    } else if position == start {
                        assert(starts@[starts.len() - 1] == position as usize);
                    } else {
                        assert(bytes@[position] != 73u8);
                    }
                }
            }
        } else {
            index += 1;
        }
    }
    starts
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
