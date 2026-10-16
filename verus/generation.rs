//! Verus harness for the generation coordinators' integer bookkeeping.
//!
//! Verifies, in place, the specifications attached to each included source:
//!
//! - `src/generation/bookkeeping.rs`: longest-suffix stop matching
//!   (`ends_with_stop`, `longest_stop_suffix`, smallest index on ties) and
//!   finish precedence (`stop_decision`: EOS, then the longest stop sequence,
//!   then the length budget); prefill chunk partitioning (`prefill_chunk_end`)
//!   and rectangular row spans (`row_span`, `matrix_row_span`, each error
//!   under its own condition); beam-row group preservation (`beam_row`, whose
//!   row lies in group `source`) and initial expansion
//!   (`fill_beam_expansion`); the beam candidate frontier capacity
//!   (`frontier_capacity`) and its partition into finalized and live ranks
//!   (`partition_frontier`: both strictly increasing, at most `beams` live);
//!   the explicit-position preflight (`next_position_after`) and the per-step
//!   advance of exactly the active positions (`advance_active_positions`); the
//!   cache-capacity preflight (`required_cache_capacity`: the exact sum
//!   `longest_prompt + max_new_tokens - 1`, `None` exactly on overflow); the
//!   accepted-prefix arithmetic of one speculative-decoding cycle
//!   (`speculative_commit`); and prompt-lookup drafting for one n-gram size
//!   (`prompt_lookup_span`, the earliest occurrence with a nonempty
//!   continuation, compared by `windows_equal`) and for the largest matching
//!   size of a range (`prompt_lookup_draft_span`).
//! - `src/generation/layout.rs`: exact valid-entry counts (`count_valid`),
//!   last valid indices (`last_valid`), and any-valid scans (`any_valid`) of
//!   row-major masks, with the first empty row reported (`fill_row_lengths`);
//!   prefill-chunk row activity (`chunk_activity`) and the skip, prefill, or
//!   decode choice (`chunk_call`); the hard-constraint and history-count
//!   scatter layouts (`push_constraint_row`, `push_history_row`: each row in
//!   its own `width`-wide segment, every entry kept, padded with no-op
//!   entries); the logit-bias layout (`repeat_row`: one row repeated exactly,
//!   no padding); the decode-step scatter (`record_scatter`: no-op entries for
//!   inactive rows); the per-step generated-length advance of active rows
//!   (`advance_generated_lengths`); the split of a packed selection readback
//!   into IDs and scores (`deinterleave`); the index of the first token
//!   (`first_outside_vocabulary`) or first active row's token
//!   (`first_active_outside_vocabulary`) outside the vocabulary; and the
//!   conversion of a history value into a vocabulary token ID
//!   (`history_token_id`).
//! - `src/generation/ngram.rs`: the prefix range under which a completed
//!   n-gram is recorded (`completed_ngram_prefix`); the lookup key before the
//!   next token (`ngram_lookup_start`, the same prefix, so the no-repeat ban is
//!   exact); and sorted, deduplicated follower insertion (`insert_follower`:
//!   the list stays strictly increasing and gains exactly the new token).
//! - `src/generation/random.rs`: `splitmix64` computes exactly
//!   `mix(seed + gamma * (index + 1) mod 2^64)`; that draw is injective in the
//!   counter for every seed (`lemma_draws_are_distinct`: the finalizer is
//!   inverted round by round and `gamma` by its modular inverse), so a
//!   stream's `2^64` raw draws are pairwise distinct; advancing the counter
//!   adds `gamma` to the Weyl state (`lemma_counter_step_adds_gamma`), as the
//!   classic sequential generator does; `SplitMix64::new` (through
//!   `stream_start`) starts at position zero, `SplitMix64::position` reports
//!   the counter, and `SplitMix64::next_u64` returns the draw at its position
//!   and advances it by one, so the `k`-th draw of a new stream is
//!   `splitmix64(seed, k)`.
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
