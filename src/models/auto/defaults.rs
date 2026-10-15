//! Pure Hugging Face generation-default arithmetic for automatic loading.
//!
//! `generation_config.json` values are translated with the pinned Hugging Face
//! `GenerationConfig` rules (transformers `cfa309cf`). The integer part of that
//! translation lives here, free of Burn, serde, and allocation, so Verus checks
//! these exact bodies through `verus/integration.rs` (see `verus/README.md`).
//!
//! # Verified properties
//!
//! - [`sampling_top_k`] disables top-k exactly for an explicit zero, otherwise
//!   resolves the requested (or default 50) cutoff clamped to the vocabulary,
//!   and for any nonempty vocabulary returns a cutoff in `1..=vocabulary`, so
//!   the resolved value always passes `GenerationConfig::validate`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Hugging Face's `top_k` when sampling without an explicit value (`_get_default_generation_params`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const HUGGING_FACE_DEFAULT_TOP_K: usize = 50;

/// Resolves the sampling `top_k` exactly as Hugging Face's generation does.
///
/// An unset value takes Hugging Face's default of 50, zero disables top-k
/// filtering, and every value is clamped to the vocabulary, as its top-k
/// warper computes `min(top_k, vocabulary_size)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(resolved =>
    ensures
        resolved.is_none() <==> top_k == Some(0usize),
        match resolved {
            Some(k) => {
                &&& k <= vocabulary_size
                &&& vocabulary_size > 0 ==> 1 <= k
                &&& match top_k {
                    Some(requested) => k == if requested < vocabulary_size { requested } else { vocabulary_size },
                    None => k == if HUGGING_FACE_DEFAULT_TOP_K < vocabulary_size { HUGGING_FACE_DEFAULT_TOP_K } else { vocabulary_size },
                }
            },
            None => true,
        },
))]
pub(crate) fn sampling_top_k(top_k: Option<usize>, vocabulary_size: usize) -> Option<usize> {
    let requested = match top_k {
        Some(0) => return None,
        Some(requested) => requested,
        None => HUGGING_FACE_DEFAULT_TOP_K,
    };
    if requested < vocabulary_size {
        Some(requested)
    } else {
        Some(vocabulary_size)
    }
}

#[cfg(test)]
#[path = "defaults_tests.rs"]
mod tests;
