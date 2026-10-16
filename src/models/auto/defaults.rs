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
//!   the same set, sorted and deduplicated by a merge sort whose result is
//!   verified. Verus also proves the sort's recursion depth is at most
//!   `⌈log₂ n⌉` (a balanced split) and each merge linear in its inputs, from
//!   which its `O(n log n)` cost follows by the standard per-level argument,
//!   so adversarially long configuration lists stay cheap; the total itself is
//!   exercised by a 400k-ID test rather than proved.

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

/// `token` is one of the first `count` IDs.
pub open spec fn id_listed(ids: Seq<usize>, count: int, token: u32) -> bool {
    exists|i: int| 0 <= i < count && ids[i] == token as usize
}

/// `tokens` holds exactly the distinct values among the first `count` IDs.
pub open spec fn same_token_set(tokens: Seq<u32>, ids: Seq<usize>, count: int) -> bool {
    &&& forall|i: int| 0 <= i < count ==> tokens.contains(#[trigger] ids[i] as u32)
    &&& forall|k: int| 0 <= k < tokens.len() ==> id_listed(ids, count, #[trigger] tokens[k])
}

/// Every value of `values` occurs in `out`.
pub open spec fn covers(out: Seq<u32>, values: Seq<u32>) -> bool {
    forall|a: int| 0 <= a < values.len() ==> out.contains(#[trigger] values[a])
}

/// Every value of `out` occurs in `left` or in `right`.
pub open spec fn drawn_from(out: Seq<u32>, left: Seq<u32>, right: Seq<u32>) -> bool {
    forall|k: int| 0 <= k < out.len() ==> left.contains(#[trigger] out[k]) || right.contains(out[k])
}

/// `value` occurs in `values[start..end)`.
pub open spec fn occurs_in(values: Seq<u32>, start: int, end: int, value: u32) -> bool {
    exists|i: int| start <= i < end && values[i] == value
}

/// `out` holds exactly the distinct values of `values[start..end)`, strictly increasing.
pub open spec fn sorted_set_of(out: Seq<u32>, values: Seq<u32>, start: int, end: int) -> bool {
    &&& strictly_increasing(out)
    &&& forall|i: int| start <= i < end ==> out.contains(#[trigger] values[i])
    &&& forall|k: int| 0 <= k < out.len() ==> occurs_in(values, start, end, #[trigger] out[k])
}

/// `⌈log₂ n⌉` for `n ≥ 1`, and `0` for `n = 0`.
///
/// It counts the rounded-up halvings `n ↦ ⌈n / 2⌉` that reduce `n` to at most
/// one, so it bounds the depth of a recursion that splits a range in half.
pub open spec fn ceil_log2(n: nat) -> nat
    decreases n,
{
    if n <= 1 {
        0
    } else {
        1 + ceil_log2(((n + 1) / 2) as nat)
    }
}

/// `ceil_log2` is monotonic: `a <= b` implies `⌈log₂ a⌉ <= ⌈log₂ b⌉`.
pub proof fn lemma_ceil_log2_monotonic(a: nat, b: nat)
    requires
        a <= b,
    ensures
        ceil_log2(a) <= ceil_log2(b),
    decreases b,
{
    if a > 1 {
        lemma_ceil_log2_monotonic(((a + 1) / 2) as nat, ((b + 1) / 2) as nat);
    }
}

/// Splitting `n ≥ 2` at `n / 2` leaves halves of strictly smaller `⌈log₂⌉`.
///
/// The halves have lengths `⌊n / 2⌋` and `⌈n / 2⌉`, and `⌈log₂ ⌈n / 2⌉⌉` is
/// exactly `⌈log₂ n⌉ - 1`.
pub proof fn lemma_halves_shrink(n: nat)
    requires
        n >= 2,
    ensures
        ceil_log2(n / 2) < ceil_log2(n),
        ceil_log2((n - n / 2) as nat) < ceil_log2(n),
{
    assert(n - n / 2 == (n + 1) / 2);
    lemma_ceil_log2_monotonic(n / 2, ((n + 1) / 2) as nat);
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

/// Merges two strictly increasing lists into the strictly increasing list of their union.
///
/// A value present in both lists appears once. The proof also bounds the
/// cost: every iteration consumes the head of at least one list (the loop
/// measure is the number of unconsumed entries), so the merge runs at most
/// `|left| + |right|` iterations and its result is no longer than that.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(merged =>
    requires
        strictly_increasing(left@),
        strictly_increasing(right@),
    ensures
        strictly_increasing(merged@),
        covers(merged@, left@),
        covers(merged@, right@),
        drawn_from(merged@, left@, right@),
        merged.len() <= left.len() + right.len(),
))]
fn merge_unique(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut merged: Vec<u32> = Vec::new();
    let mut i: usize = 0;
    let mut j: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            i <= left.len(),
            j <= right.len(),
            strictly_increasing(left@),
            strictly_increasing(right@),
            strictly_increasing(merged@),
            covers(merged@, left@.subrange(0, i as int)),
            covers(merged@, right@.subrange(0, j as int)),
            drawn_from(merged@, left@, right@),
            merged.len() <= i + j,
            forall|k: int| 0 <= k < merged.len() ==> {
                &&& (i < left.len() ==> #[trigger] merged@[k] < left@[i as int])
                &&& (j < right.len() ==> merged@[k] < right@[j as int])
            },
        decreases left.len() - i + right.len() - j,
    ))]
    while i < left.len() || j < right.len() {
        #[cfg(verus_keep_ghost)]
        proof_decl! {
            let ghost before = merged@;
            let ghost old_i = i as int;
            let ghost old_j = j as int;
        }
        let take_left = j == right.len() || (i < left.len() && left[i] <= right[j]);
        let take_right = i == left.len() || (j < right.len() && right[j] <= left[i]);
        let value = if take_left { left[i] } else { right[j] };
        merged.push(value);
        if take_left {
            i += 1;
        }
        if take_right {
            j += 1;
        }
        // The pushed value exceeds every earlier one and stays below the next candidates.
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(merged@ == before.push(value));
            assert forall|a: int, b: int| 0 <= a < b < merged@.len()
                implies merged@[a] < merged@[b] by {
                if b < before.len() {
                    assert(before[a] < before[b]);
                }
            }
            assert forall|a: int| 0 <= a < i
                implies merged@.contains(#[trigger] left@.subrange(0, i as int)[a]) by {
                if a < old_i {
                    assert(left@.subrange(0, old_i)[a] == left@[a]);
                    let k = choose|k: int| 0 <= k < before.len() && before[k] == left@[a];
                    assert(merged@[k] == left@[a]);
                } else {
                    assert(merged@[before.len() as int] == left@[a]);
                }
            }
            assert forall|b: int| 0 <= b < j
                implies merged@.contains(#[trigger] right@.subrange(0, j as int)[b]) by {
                if b < old_j {
                    assert(right@.subrange(0, old_j)[b] == right@[b]);
                    let k = choose|k: int| 0 <= k < before.len() && before[k] == right@[b];
                    assert(merged@[k] == right@[b]);
                } else {
                    assert(merged@[before.len() as int] == right@[b]);
                }
            }
            assert forall|k: int| 0 <= k < merged@.len()
                implies left@.contains(#[trigger] merged@[k]) || right@.contains(merged@[k]) by {
                if k < before.len() {
                    assert(merged@[k] == before[k]);
                } else if take_left {
                    assert(left@[old_i] == value);
                } else {
                    assert(right@[old_j] == value);
                }
            }
            assert forall|k: int| 0 <= k < merged@.len() implies {
                &&& (i < left.len() ==> #[trigger] merged@[k] < left@[i as int])
                &&& (j < right.len() ==> merged@[k] < right@[j as int])
            } by {
                if i < left.len() && i > old_i {
                    assert(left@[old_i] < left@[i as int]);
                }
                if j < right.len() && j > old_j {
                    assert(right@[old_j] < right@[j as int]);
                }
            }
        }
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(left@.subrange(0, left.len() as int) =~= left@);
        assert(right@.subrange(0, right.len() as int) =~= right@);
    }
    merged
}

/// Sorts and deduplicates `values[start..end)` by recursive merging.
///
/// The result is the strictly increasing list of exactly the distinct values
/// in the range, no longer than the range. The proof also shows the split is
/// balanced: its termination measure is `⌈log₂(end - start)⌉`, which strictly
/// decreases at every recursive call, so the recursion depth is at most
/// `⌈log₂ n⌉`. Together with each merge's proved bound of `|left| + |right|`
/// iterations over disjoint subranges, this gives the `O(n log n)` running
/// time by the standard per-level argument (that final sum is not itself a
/// Verus theorem; `long_token_lists_canonicalize_quickly` exercises it).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sorted =>
    requires
        start <= end <= values.len(),
    ensures
        sorted_set_of(sorted@, values@, start as int, end as int),
        sorted.len() <= end - start,
    decreases ceil_log2((end - start) as nat),
))]
fn sorted_unique(values: &[u32], start: usize, end: usize) -> Vec<u32> {
    if end - start == 0 {
        return Vec::new();
    }
    if end - start == 1 {
        let single: Vec<u32> = vec![values[start]; 1];
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(single@[0] == values@[start as int]);
        }
        return single;
    }
    let middle = start + (end - start) / 2;
    // Both halves have a strictly smaller depth measure `⌈log₂⌉`.
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_halves_shrink((end - start) as nat);
    }
    let left = sorted_unique(values, start, middle);
    let right = sorted_unique(values, middle, end);
    let merged = merge_unique(left.as_slice(), right.as_slice());
    // Both halves' values reach the merge, and the merge adds nothing else.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(sorted_set_of(left@, values@, start as int, middle as int));
        assert(sorted_set_of(right@, values@, middle as int, end as int));
        assert forall|i: int| start <= i < end implies merged@.contains(#[trigger] values@[i]) by {
            if i < middle {
                assert(left@.contains(values@[i]));
                let a = choose|a: int| 0 <= a < left@.len() && left@[a] == values@[i];
                assert(merged@.contains(left@[a]));
            } else {
                assert(right@.contains(values@[i]));
                let b = choose|b: int| 0 <= b < right@.len() && right@[b] == values@[i];
                assert(merged@.contains(right@[b]));
            }
        }
        assert forall|k: int| 0 <= k < merged@.len()
            implies occurs_in(values@, start as int, end as int, #[trigger] merged@[k]) by {
            if left@.contains(merged@[k]) {
                let a = choose|a: int| 0 <= a < left@.len() && left@[a] == merged@[k];
                assert(occurs_in(values@, start as int, middle as int, left@[a]));
                let i = choose|i: int| start <= i < middle && values@[i] == left@[a];
                assert(values@[i] == merged@[k]);
            } else {
                assert(right@.contains(merged@[k]));
                let b = choose|b: int| 0 <= b < right@.len() && right@[b] == merged@[k];
                assert(occurs_in(values@, middle as int, end as int, right@[b]));
                let i = choose|i: int| middle <= i < end && values@[i] == right@[b];
                assert(values@[i] == merged@[k]);
            }
        }
        assert(sorted_set_of(merged@, values@, start as int, end as int));
    }
    merged
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
    let mut tokens: Vec<u32> = Vec::with_capacity(ids.len());
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= ids.len(),
            ids_accepted(ids@, index as int, vocabulary_size as int),
            tokens.len() == index,
            forall|i: int| 0 <= i < index ==> #[trigger] tokens@[i] as usize == ids@[i],
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
        tokens.push(id as u32);
        index += 1;
    }
    let canonical = sorted_unique(tokens.as_slice(), 0, tokens.len());
    // Converted IDs equal the inputs, so the sorted set is the input set.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(sorted_set_of(canonical@, tokens@, 0, tokens.len() as int));
        assert forall|i: int| 0 <= i < ids.len()
            implies canonical@.contains(#[trigger] ids@[i] as u32) by {
            assert(canonical@.contains(tokens@[i]));
            assert(tokens@[i] == ids@[i] as u32);
        }
        assert forall|k: int| 0 <= k < canonical@.len()
            implies id_listed(ids@, ids.len() as int, #[trigger] canonical@[k]) by {
            assert(occurs_in(tokens@, 0, tokens.len() as int, canonical@[k]));
            let i = choose|i: int| 0 <= i < tokens.len() && tokens@[i] == canonical@[k];
            assert(ids@[i] == canonical@[k] as usize);
        }
    }
    Ok(canonical)
}

#[cfg(test)]
#[path = "defaults_tests.rs"]
mod tests;
