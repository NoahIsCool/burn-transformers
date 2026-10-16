//! Verus harness for the generation coordinators' integer bookkeeping.
//!
//! Verifies, in place, the specifications attached to each included source:
//!
//! - `src/generation/bookkeeping.rs`: longest-suffix stop matching and finish
//!   precedence; prefill chunk partitioning and rectangular row spans; beam-row
//!   group preservation and initial expansion; the beam candidate frontier
//!   capacity and its partition into finalized and live ranks (both strictly
//!   increasing, at most `beams` live); probability-row spans with each error
//!   under its own condition; the explicit-position preflight and the per-step
//!   advance of exactly the active positions; the cache-capacity preflight
//!   (exact sum, `None` exactly on overflow); the accepted-prefix arithmetic of
//!   one speculative-decoding cycle; and prompt-lookup drafting for one n-gram
//!   size and for the largest matching size of a range.
//! - `src/generation/layout.rs`: exact valid-entry counts, last valid indices,
//!   and any-valid scans of row-major masks, with the first empty row
//!   reported; prefill-chunk row activity and the skip, prefill, or decode
//!   choice; the hard-constraint and history-count scatter layouts (each row
//!   in its own `width`-wide segment, every entry kept, padded with no-op
//!   entries); the logit-bias layout (one row repeated exactly, no padding);
//!   the decode-step scatter (no-op entries for inactive rows); the per-step
//!   generated-length advance of active rows; the split of a packed selection
//!   readback into IDs and scores; the index of the first token (or first
//!   active row's token) outside the vocabulary; and the conversion of a
//!   history value into a vocabulary token ID.
//! - `src/generation/ngram.rs`: the prefix range under which a completed
//!   n-gram is recorded; the lookup key before the next token (the same
//!   prefix, so the no-repeat ban is exact); and sorted, deduplicated follower
//!   insertion (the list stays strictly increasing and gains exactly the new
//!   token).
//! - `src/generation/random.rs`: `splitmix64` computes exactly
//!   `mix(seed + gamma * (index + 1) mod 2^64)`; that draw is injective in the
//!   counter for every seed (the finalizer is inverted round by round and
//!   `gamma` by its modular inverse), so a stream's `2^64` raw draws are
//!   pairwise distinct; advancing the counter adds `gamma` to the Weyl state,
//!   as the classic sequential generator does; `SplitMix64::new` starts at
//!   position zero, `SplitMix64::position` reports the counter, and
//!   `SplitMix64::next_u64` returns the draw at its position and advances it
//!   by one, so the `k`-th draw of a new stream is `splitmix64(seed, k)`.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/generation/bookkeeping.rs"]
pub mod bookkeeping;

#[path = "../src/generation/layout.rs"]
pub mod layout;

#[path = "../src/generation/ngram.rs"]
pub mod ngram;

#[path = "../src/generation/random.rs"]
pub mod random;
