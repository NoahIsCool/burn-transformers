//! Integer decision rules for automatic generation defaults.
//!
//! A local `generation_config.json` may declare logits-processor settings the
//! selected consumer cannot honor. Causal generation applies
//! `min_new_tokens` (clamped to `max_new_tokens`) and `no_repeat_ngram_size`;
//! encoder-decoder beam search applies neither. A setting that changes
//! Hugging Face's behavior but is not applied, or is applied only after
//! clamping, is reported on the loaded model instead of being dropped
//! silently. Zero disables both settings in Hugging Face, so a zero is never
//! reported. A tokenizer padding declaration must name at most one token.
//!
//! # Verified properties
//!
//! Verus checks every function here in place through `verus/loaders.rs` (see
//! `verus/README.md`):
//!
//! - [`reported_min_new_tokens`] reports a declared minimum `m` exactly when
//!   `m > 0` and either the consumer ignores it or the applied minimum is
//!   smaller; with the applied minimum `min(m, max_new_tokens)` that the
//!   loader installs, `lemma_clamped_minimum_reported` shows a causal load
//!   reports `m` exactly when `m > max_new_tokens`;
//! - [`reported_no_repeat_ngram`] reports a positive n-gram size exactly when
//!   the consumer does not apply it, so a positive size is always either
//!   applied or reported, never both;
//! - [`single_token`] accepts an empty or one-element list exactly and
//!   rejects two or more tokens.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// The declared `min_new_tokens` reported as unapplied, if any.
pub open spec fn min_new_tokens_report(
    declared: Option<usize>,
    applied: usize,
    constraints: bool,
) -> Option<usize> {
    match declared {
        Some(count) => if count > 0 && (!constraints || count > applied) {
            Some(count)
        } else {
            None
        },
        None => None,
    }
}

/// A causal load applies `min(m, max_new_tokens)` and therefore reports a
/// declared minimum `m` exactly when it exceeds the length bound.
pub proof fn lemma_clamped_minimum_reported(declared: usize, max_new_tokens: usize)
    ensures
        ({
            let applied = if declared <= max_new_tokens { declared } else { max_new_tokens };
            min_new_tokens_report(Some(declared), applied, true) is Some <==> declared
                > max_new_tokens
        }),
{
}

} // verus!

/// Declared `min_new_tokens` to report as not (fully) applied.
///
/// `applied` is the minimum the loaded configuration enforces and
/// `constraints` whether the consumer applies processor constraints at all.
/// A positive declared minimum is reported when the consumer ignores it or
/// enforces a smaller value; zero (disabled) and an absent value never are.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(reported =>
    ensures
        reported == min_new_tokens_report(declared, applied, constraints),
))]
pub(super) fn reported_min_new_tokens(
    declared: Option<usize>,
    applied: usize,
    constraints: bool,
) -> Option<usize> {
    match declared {
        Some(count) if count > 0 && (!constraints || count > applied) => Some(count),
        _ => None,
    }
}

/// Declared `no_repeat_ngram_size` to report as unapplied.
///
/// A positive size is reported exactly when the consumer does not apply
/// processor constraints, which is exactly when the loader leaves it unset.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(reported =>
    ensures
        reported == match declared {
            Some(size) => if size > 0 && !constraints { Some(size) } else { None },
            None => None,
        },
))]
pub(super) fn reported_no_repeat_ngram(
    declared: Option<usize>,
    constraints: bool,
) -> Option<usize> {
    match declared {
        Some(size) if size > 0 && !constraints => Some(size),
        _ => None,
    }
}

/// A padding declaration lists more than one token ID.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MultipleTokens;

/// The only token of a normalized ID list: none for an empty list, the token
/// for one element, and [`MultipleTokens`] for two or more.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(token =>
    ensures
        match token {
            Ok(None) => ids@.len() == 0,
            Ok(Some(token)) => ids@.len() == 1 && token == ids@[0],
            Err(_) => ids@.len() >= 2,
        },
))]
pub(super) fn single_token(ids: &[u32]) -> Result<Option<u32>, MultipleTokens> {
    if ids.is_empty() {
        Ok(None)
    } else if ids.len() == 1 {
        Ok(Some(ids[0]))
    } else {
        Err(MultipleTokens)
    }
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
