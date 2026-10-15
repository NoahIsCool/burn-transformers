//! Verus harness for DeepSeek-V2 and Muse-Glimmer family logic.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/deepseek_v2/model/batch.rs`: the host planning of batched
//! stateful execution. For all admissible inputs the proofs establish
//! freedom from integer overflow and that
//!
//! * `row_maps` gathers exactly the valid columns into the first `n` block
//!   entries and scatters each valid column back to its entry, every other
//!   column to the zero row `T`;
//! * `batch_extents` returns the widest row `T = max_b n_b` (attained) and a
//!   key extent covering every row's cache end `L_b + n_b`;
//! * `key_limits` returns nondecreasing limits in `1..=E` that cover every
//!   active row's causal key prefix `L_b + i + 1`, the bound that lets
//!   bounded attention skip keys a query chunk cannot see;
//! * `query_slots` places entry `k < n_b` of row `b` at cache slot `L_b + k`
//!   and keeps padding placeholders within the key extent;
//!
//! and the specifications attached to `src/models/muse_glimmer/schedule.rs`:
//!
//! * `is_global_layer` is the backward-counted `(n - 1 - i) mod 4 = 0` rule
//!   without underflow, and the last layer is always global;
//! * `group_slots` gives every layer the number of earlier layers of its kind,
//!   so each cache group's slots are dense and strictly increasing in layer
//!   order (an injective schedule);
//! * `ring_chunk_keys` returns, for cached sliding attention over
//!   `[ring history | fresh block]`, a key slice that contains every fresh key
//!   visible to the chunk's rows, the whole ring while any row can still see
//!   history, no history otherwise, and at most `H + W - 1 + rows` keys.
//!
//! See `verus/README.md` for the method, toolchain, and command. Burn tensor
//! programs driven by these plans (gathers, scatters, cache writes, and
//! attention) remain covered by the Rust tests, in particular
//! `batched_rows_match_independent_rows`.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/deepseek_v2/model/batch.rs"]
pub mod batch;

#[path = "../src/models/muse_glimmer/schedule.rs"]
pub mod schedule;
