//! Verified index arithmetic of the no-repeat n-gram constraint.
//!
//! `no_repeat_ngram_size = n` forbids a token that would complete an n-gram
//! already present in the row (Hugging Face `NoRepeatNGramLogitsProcessor`).
//! The host index maps every `(n - 1)`-token prefix to the sorted, deduplicated
//! tokens that followed it. Two ranges must agree for the ban to be exact: the
//! prefix under which [`completed_ngram_prefix`] records a newly appended
//! token, and the key [`ngram_lookup_start`] reads before the next token. The
//! follower lists stay strictly increasing through [`insert_follower`], so a
//! token is stored once however often its n-gram repeats.
//!
//! # Verified properties
//!
//! Every function below is checked in place by Verus through
//! `verus/generation.rs` (see `verus/README.md`). For all admissible inputs
//! the proofs establish that arithmetic cannot overflow and:
//!
//! - after `length` tokens, the completed n-gram's prefix is exactly
//!   `[length - n, length - 1)`, present exactly when `length >= n`;
//! - the lookup key before the next token starts at `length + 1 - n`, the
//!   prefix start the next recorded token will use, present exactly when
//!   `length + 1 >= n`;
//! - inserting a follower keeps the list strictly increasing and its set of
//!   elements becomes exactly the old set plus the token.
//!
//! The hash map and the scatter that bans the tokens stay unverified; the
//! Rust test suite covers them.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Every element of `values` is smaller than every later element.
pub open spec fn strictly_increasing(values: Seq<u32>) -> bool {
    forall|i: int, j: int| 0 <= i < j < values.len() ==> values[i] < values[j]
}

} // verus!

/// Prefix range of the n-gram completed by the newest of `length` tokens.
///
/// Returns `[start, end)` with `start = length - size` and `end = length - 1`:
/// the `size - 1` tokens immediately before the newest one, which is the
/// n-gram's last token. `None` when fewer than `size` tokens exist.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(prefix =>
    requires
        size > 0,
    ensures
        match prefix {
            Some((start, end)) => {
                &&& size <= length
                &&& start == length - size
                &&& end == length - 1
                &&& end - start == size - 1
            },
            None => length < size,
        },
))]
pub(crate) fn completed_ngram_prefix(length: usize, size: usize) -> Option<(usize, usize)> {
    if length >= size {
        Some((length - size, length - 1))
    } else {
        None
    }
}

/// Start of the key whose recorded followers the next token may not repeat.
///
/// After `length` tokens, the next token would complete the n-gram whose
/// prefix is the last `size - 1` tokens, `[length + 1 - size, length)`. That
/// is exactly the prefix [`completed_ngram_prefix`] records once the next
/// token arrives (at length `length + 1`), so the lookup and the index agree.
/// `None` while no n-gram can be completed yet (`length + 1 < size`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(start =>
    requires
        size > 0,
    ensures
        match start {
            Some(start) => {
                &&& size <= length + 1
                &&& start == length + 1 - size
                &&& length - start == size - 1
            },
            None => length + 1 < size,
        },
))]
pub(crate) fn ngram_lookup_start(length: usize, size: usize) -> Option<usize> {
    // `length + 1 < size` and `length + 1 - size`, without forming `length + 1`.
    if length < size - 1 {
        None
    } else {
        Some(length - (size - 1))
    }
}

/// Inserts `token` into the strictly increasing `followers` unless present.
///
/// Binary search finds the first element not below `token`; the token is
/// inserted there unless that element equals it. The proof shows the list
/// stays strictly increasing and holds exactly its old elements plus `token`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(
    requires
        strictly_increasing(old(followers)@),
    ensures
        strictly_increasing(final(followers)@),
        forall|value: u32| #[trigger] final(followers)@.contains(value)
            <==> (old(followers)@.contains(value) || value == token),
))]
pub(crate) fn insert_follower(followers: &mut Vec<u32>, token: u32) {
    let mut low = 0;
    let mut high = followers.len();
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            low <= high <= followers.len(),
            followers@ == old(followers)@,
            strictly_increasing(followers@),
            forall|i: int| 0 <= i < low ==> #[trigger] followers@[i] < token,
            forall|i: int| high <= i < followers.len() ==> #[trigger] followers@[i] >= token,
        decreases high - low,
    ))]
    while low < high {
        let middle = low + (high - low) / 2;
        if followers[middle] < token {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if low < followers.len() && followers[low] == token {
        return;
    }
    #[cfg(verus_keep_ghost)]
    proof_decl! { let ghost previous = followers@; }
    followers.insert(low, token);
    #[cfg(verus_keep_ghost)]
    proof! {
        // Elements before `low` are below `token`, elements after it above.
        assert forall|i: int, j: int| 0 <= i < j < followers@.len() implies followers@[i]
            < followers@[j] by {
            if j < low {
                assert(followers@[i] == previous[i] && followers@[j] == previous[j]);
            } else if j == low {
                assert(followers@[i] == previous[i]);
            } else if i < low {
                assert(followers@[i] == previous[i] && followers@[j] == previous[j - 1]);
            } else if i == low {
                assert(followers@[j] == previous[j - 1]);
            } else {
                assert(followers@[i] == previous[i - 1] && followers@[j] == previous[j - 1]);
            }
        }
        assert forall|value: u32| #[trigger] followers@.contains(value) <==> (previous.contains(
            value,
        ) || value == token) by {
            if followers@.contains(value) {
                let k = choose|k: int| 0 <= k < followers@.len() && followers@[k] == value;
                if k < low {
                    assert(previous[k] == value);
                } else if k > low {
                    assert(previous[k - 1] == value);
                }
            }
            if previous.contains(value) {
                let k = choose|k: int| 0 <= k < previous.len() && previous[k] == value;
                if k < low {
                    assert(followers@[k] == value);
                } else {
                    assert(followers@[k + 1] == value);
                }
            }
            if value == token {
                assert(followers@[low as int] == value);
            }
        }
    }
}

#[cfg(test)]
#[path = "ngram_tests.rs"]
mod tests;
