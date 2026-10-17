//! Verified "last match wins" precedence for ordered overrides.
//!
//! Several experiment surfaces resolve ordered overrides by letting the most
//! recent matching entry win: per-layer attention policies in
//! [`super::ExecutionContext`] and path overrides in
//! [`crate::initialization::InitPolicy`]. Callers compute which entries match
//! (a layer address or a path prefix) and this core picks the winner. It uses
//! no Burn types or allocation, so Verus checks this exact body through
//! `verus/experiment.rs` (see `verus/README.md`).
//!
//! # Verified properties
//!
//! - [`last_match`] returns exactly the greatest index whose flag is set, or
//!   `None` when no flag is set: no later entry matches, so the returned
//!   override is the one registered last.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Returns the index of the last `true` flag, so the most recent matching override wins.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
        match found {
            None => forall|i: int| 0 <= i < matches.len() ==> !matches@[i],
            Some(index) => {
                &&& index < matches.len()
                &&& matches@[index as int]
                &&& forall|later: int| index < later < matches.len() ==> !matches@[later]
            },
        },
))]
pub(crate) fn last_match(matches: &[bool]) -> Option<usize> {
    let mut remaining = matches.len();
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            remaining <= matches.len(),
            forall|later: int| remaining <= later < matches.len() ==> !matches@[later],
        decreases remaining,
    ))]
    while remaining > 0 {
        remaining -= 1;
        if matches[remaining] {
            return Some(remaining);
        }
    }
    None
}

#[cfg(test)]
#[path = "precedence_tests.rs"]
mod tests;
