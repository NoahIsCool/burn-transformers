//! Dependency-free vocabulary-ID checks for special tokens and model embeddings.
//!
//! Tokenizer metadata may declare special-token IDs (`pad_token_id`, ...) next
//! to their text. Those IDs cannot create vocabulary rows: the tokenizer's own
//! text-to-ID mapping is authoritative, so a declared ID must fit `u32` and
//! must equal the ID the vocabulary gives the declared text. Every ID the
//! tokenizer can emit must then index a row of the model's embedding table.
//!
//! # Verified behavior
//!
//! `verus/tokenization.rs` checks these functions in place:
//!
//! - [`declared_special_id`] accepts exactly the JSON integers that fit
//!   `u32`, unchanged;
//! - [`reconcile_special_id`] returns the vocabulary's ID, and only when no
//!   ID was declared or the declared ID equals it, so every accepted special
//!   token names the vocabulary's own row;
//! - [`id_fits_vocabulary`] holds exactly when the ID is below the model's
//!   vocabulary size, so checking the tokenizer's largest ID bounds every ID
//!   it emits.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// A declared special-token ID from JSON, if it fits `u32`.
///
/// Verified: `Some(value)` exactly when `value <= u32::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(id =>
    ensures
        value <= u32::MAX ==> id == Some(value as u32),
        value > u32::MAX ==> id.is_none(),
))]
pub(crate) fn declared_special_id(value: u64) -> Option<u32> {
    if value <= u64::from(u32::MAX) {
        Some(value as u32)
    } else {
        None
    }
}

/// The vocabulary's ID for a special token, or the conflicting declared ID.
///
/// Verified: `Ok(vocabulary)` exactly when nothing was declared or the
/// declaration equals the vocabulary's ID; otherwise `Err(declared)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(id =>
    ensures
        match declared {
            None => id == Ok::<u32, u32>(vocabulary),
            Some(declared) => if declared == vocabulary {
                id == Ok::<u32, u32>(vocabulary)
            } else {
                id == Err::<u32, u32>(declared)
            },
        },
))]
pub(crate) fn reconcile_special_id(declared: Option<u32>, vocabulary: u32) -> Result<u32, u32> {
    match declared {
        Some(declared) if declared != vocabulary => Err(declared),
        _ => Ok(vocabulary),
    }
}

/// Whether `id` indexes a row of an embedding table with `vocabulary_size` rows.
///
/// Verified: exactly `id < vocabulary_size`; since every smaller ID then
/// fits too, the tokenizer's largest ID decides the whole vocabulary.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fits =>
    ensures
        fits == (id < vocabulary_size),
))]
pub(crate) fn id_fits_vocabulary(id: u32, vocabulary_size: usize) -> bool {
    (id as usize) < vocabulary_size
}

#[cfg(test)]
#[path = "vocabulary_tests.rs"]
mod tests;
