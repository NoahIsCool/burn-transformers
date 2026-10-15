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
//! - [`clamped_min_new_tokens`] is `min(declared, max_new_tokens)` and lowers
//!   the declared value exactly when it exceeds `max_new_tokens`, which is
//!   when the loader reports the declared value as unapplied.
//! - [`canonical_token_ids`] accepts a token list exactly when every ID lies
//!   in the vocabulary and fits `u32`, reporting the first offending ID
//!   otherwise, and returns the strictly increasing list of the distinct IDs:
//!   the same set, sorted and deduplicated.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Every entry is strictly smaller than the next, so the list is sorted and duplicate-free.
pub open spec fn strictly_increasing(tokens: Seq<u32>) -> bool {
    forall|a: int, b: int| 0 <= a < b < tokens.len() ==> tokens[a] < tokens[b]
}

/// The first `count` IDs are in the vocabulary and fit `u32`.
pub open spec fn ids_accepted(ids: Seq<usize>, count: int, vocabulary_size: int) -> bool {
    forall|i: int| 0 <= i < count ==> #[trigger] ids[i] < vocabulary_size && ids[i] <= u32::MAX
}

/// `tokens` holds exactly the distinct values among the first `count` IDs.
pub open spec fn same_token_set(tokens: Seq<u32>, ids: Seq<usize>, count: int) -> bool {
    &&& forall|i: int| 0 <= i < count ==> tokens.contains(#[trigger] ids[i] as u32)
    &&& forall|k: int| #![trigger tokens[k]] 0 <= k < tokens.len()
        ==> exists|i: int| 0 <= i < count && #[trigger] ids[i] == tokens[k] as usize
}

} // verus!

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

/// Resolves Hugging Face's `min_new_tokens` against the length bound.
///
/// Hugging Face only warns when the minimum exceeds `max_new_tokens` and then
/// suppresses EOS for the whole run, which the clamped value does too. The
/// result is lower than `declared` exactly when `declared > max_new_tokens`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(resolved =>
    ensures
        resolved == if declared <= max_new_tokens { declared } else { max_new_tokens },
        resolved <= max_new_tokens,
        (resolved < declared) == (declared > max_new_tokens),
))]
pub(crate) fn clamped_min_new_tokens(declared: usize, max_new_tokens: usize) -> usize {
    if declared <= max_new_tokens {
        declared
    } else {
        max_new_tokens
    }
}

/// Why a token ID list cannot be normalized.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenIdError {
    /// The first ID that is not below the vocabulary size.
    OutsideVocabulary {
        /// The offending token ID.
        id: usize,
    },
    /// The first in-vocabulary ID that does not fit the `u32` runtime.
    ExceedsU32 {
        /// The offending token ID.
        id: usize,
    },
}

/// Inserts `token` into a strictly increasing list unless it is already present.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(
    requires
        strictly_increasing(old(tokens)@),
    ensures
        strictly_increasing(final(tokens)@),
        forall|value: u32| #[trigger] final(tokens)@.contains(value)
            <==> (old(tokens)@.contains(value) || value == token),
))]
fn insert_sorted_unique(tokens: &mut Vec<u32>, token: u32) {
    let mut position: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            position <= tokens.len(),
            tokens@ == old(tokens)@,
            strictly_increasing(tokens@),
            forall|k: int| 0 <= k < position ==> #[trigger] tokens@[k] < token,
        decreases tokens.len() - position,
    ))]
    while position < tokens.len() && tokens[position] < token {
        position += 1;
    }
    if position < tokens.len() && tokens[position] == token {
        return;
    }
    #[cfg(verus_keep_ghost)]
    proof_decl! {
        let ghost before = tokens@;
    }
    tokens.insert(position, token);
    #[cfg(verus_keep_ghost)]
    proof! {
        // Entries before `position` are smaller and entries after it larger than `token`.
        assert forall|a: int, b: int| 0 <= a < b < tokens@.len()
            implies tokens@[a] < tokens@[b] by {
            if b < position {
                assert(before[a] < before[b]);
            } else if a < position && b == position {
                assert(before[a] < token);
            } else if a < position {
                assert(before[a] < before[b - 1]);
            } else if a == position {
                assert(token <= before[b - 1]);
                if b - 1 > position {
                    assert(before[position as int] < before[b - 1]);
                }
            } else {
                assert(before[a - 1] < before[b - 1]);
            }
        }
        assert forall|value: u32| #[trigger] tokens@.contains(value)
            <==> (before.contains(value) || value == token) by {
            if tokens@.contains(value) {
                let k = choose|k: int| 0 <= k < tokens@.len() && tokens@[k] == value;
                if k < position {
                    assert(before[k] == value);
                } else if k > position {
                    assert(before[k - 1] == value);
                }
            }
            if before.contains(value) {
                let k = choose|k: int| 0 <= k < before.len() && before[k] == value;
                if k < position {
                    assert(tokens@[k] == value);
                } else {
                    assert(tokens@[k + 1] == value);
                }
            }
            if value == token {
                assert(tokens@[position as int] == value);
            }
        }
    }
}

/// Converts a token ID list into its canonical sorted, deduplicated `u32` form.
///
/// IDs are checked in order: the first ID outside the vocabulary, or the
/// first in-vocabulary ID beyond `u32`, is reported. Otherwise the result is
/// the strictly increasing list of the distinct IDs, exactly the set the
/// input names.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(normalized =>
    ensures
        match normalized {
            Ok(tokens) => {
                &&& ids_accepted(ids@, ids.len() as int, vocabulary_size as int)
                &&& strictly_increasing(tokens@)
                &&& same_token_set(tokens@, ids@, ids.len() as int)
            },
            Err(TokenIdError::OutsideVocabulary { id }) => exists|i: int| {
                &&& 0 <= i < ids.len()
                &&& ids[i] == id
                &&& id >= vocabulary_size
                &&& ids_accepted(ids@, i, vocabulary_size as int)
            },
            Err(TokenIdError::ExceedsU32 { id }) => exists|i: int| {
                &&& 0 <= i < ids.len()
                &&& ids[i] == id
                &&& id < vocabulary_size
                &&& id > u32::MAX
                &&& ids_accepted(ids@, i, vocabulary_size as int)
            },
        },
))]
pub(crate) fn canonical_token_ids(
    ids: &[usize],
    vocabulary_size: usize,
) -> Result<Vec<u32>, TokenIdError> {
    let mut tokens: Vec<u32> = Vec::new();
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= ids.len(),
            ids_accepted(ids@, index as int, vocabulary_size as int),
            strictly_increasing(tokens@),
            same_token_set(tokens@, ids@, index as int),
        decreases ids.len() - index,
    ))]
    while index < ids.len() {
        let id = ids[index];
        if id >= vocabulary_size {
            return Err(TokenIdError::OutsideVocabulary { id });
        }
        if id > u32::MAX as usize {
            return Err(TokenIdError::ExceedsU32 { id });
        }
        #[cfg(verus_keep_ghost)]
        proof_decl! {
            let ghost before = tokens@;
        }
        insert_sorted_unique(&mut tokens, id as u32);
        // The new set adds exactly `ids[index]`; earlier witnesses survive.
        #[cfg(verus_keep_ghost)]
        proof! {
            assert forall|i: int| 0 <= i < index + 1
                implies tokens@.contains(#[trigger] ids@[i] as u32) by {
                if i < index {
                    assert(before.contains(ids@[i] as u32));
                }
            }
            assert forall|k: int| #![trigger tokens@[k]] 0 <= k < tokens@.len()
                implies exists|i: int|
                    0 <= i < index + 1 && #[trigger] ids@[i] == tokens@[k] as usize by {
                assert(tokens@.contains(tokens@[k]));
                if tokens@[k] == id as u32 {
                    assert(ids@[index as int] == tokens@[k] as usize);
                } else {
                    assert(before.contains(tokens@[k]));
                    let j = choose|j: int| 0 <= j < before.len() && before[j] == tokens@[k];
                    let i = choose|i: int|
                        0 <= i < index && #[trigger] ids@[i] == before[j] as usize;
                    assert(ids@[i] == tokens@[k] as usize);
                }
            }
        }
        index += 1;
    }
    Ok(tokens)
}

#[cfg(test)]
#[path = "defaults_tests.rs"]
mod tests;
