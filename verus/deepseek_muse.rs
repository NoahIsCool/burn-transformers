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
//!   and keeps padding placeholders within the key extent.
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
