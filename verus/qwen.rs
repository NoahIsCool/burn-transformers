//! Verus harness for the Qwen3.5 and Qwen3-MoE family planning.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/qwen3_5/plan.rs` and `src/models/qwen3_moe/plan.rs`:
//!
//! * the derived hybrid schedule rule (`default_full_attention`: full
//!   attention exactly when `i mod interval = interval - 1`, every layer for
//!   intervals zero and one) and the whole derived schedule
//!   (`derived_full_attention`), which holds exactly `floor(L / interval)`
//!   full-attention layers;
//! * dense per-family cache slots (`cache_slots`): the slot of layer `i` is
//!   exactly the number of earlier layers of its family, the family sizes are
//!   exactly the two counts, and slots strictly increase within a family;
//! * the compact right-padded batch layout used by batched cached execution:
//!   column validation (`columns_below`, `strictly_increasing`: all below `S`,
//!   strictly increasing) is decided exactly; `n_b` (`row_counts`),
//!   `N = max_b n_b` (`widest`, attained), and prefix detection
//!   (`prefix_rows`) are exact; `compact_row_plan` (the shared `row_maps`
//!   under Qwen's field names) makes slot `k` gather valid column `c_k` and
//!   padding slots `c_0` (or `0`), and restoration map `c_k` back to `k` and
//!   every other column to the zero slot `N`; slot
//!   validity (`slot_validity`) is `i < n_b`; slot positions
//!   (`slot_positions`) are `W_b + i`, with `None` exactly when one exceeds
//!   `usize` or `i64`; the uniform-cursor test (`all_equal`) is exact; and
//!   index conversion (`signed_index`) saturates at `i64::MAX`;
//! * MTP and speculative bookkeeping: the truncated MTP cursor
//!   (`shifted_mtp_length`, `shifted_mtp_lengths`) is exactly
//!   `min(mtp, target - 1)` per row (restoring `target = MTP + 1`); the
//!   staging preflight (`first_staging_overflow`) reports the first row whose
//!   saturating request `(W_b - C_b) + n_b` exceeds the capacity; a cycle
//!   needs exactly `K + 1` staging slots (`staging_requirement`); an MTP
//!   append ends exactly at `baseline + n` within the capacity
//!   (`append_end`); and a cycle starts only from batch one with
//!   `target = MTP + 1`, a matching history, and a suffix that fits
//!   (`speculative_cycle_plan`), each failure naming its first violated
//!   condition;
//! * the call rotary context (`call_rotary_context`: the maximum active
//!   context, or one), the context of the first active row beyond a
//!   cached-context limit (`first_context_above`, `None` exactly when every
//!   active row stays within it), the first out-of-vocabulary host token
//!   (`first_token_at_or_above`), and the gated query projection width
//!   (`gated_query_width`: exactly `heads * head_dim * (2 if gated)`, `None`
//!   exactly when that exceeds `usize::MAX`);
//! * the Qwen3-MoE sparse-layer predicate (`sparse_layer`), `mlp_only_layers`
//!   membership (`listed`), and schedule existence (`any_sparse_layer`: some
//!   block `i < L` is sparse).
//!
//! Both planning files are dependency-free, so they are included as plain
//! modules; the shared ragged-block plan (`src/models/shared/block_plan.rs`,
//! whose `row_maps` and `increasing_below` the Qwen3.5 plan reuses) is
//! included at its crate path and re-verified here, with `verus/models.rs`
//! as its home harness. Burn tensor programs driven by these plans (gathers, cache
//! writes, attention masks, expert dispatch) remain covered by the Rust
//! tests, including the batched-versus-per-row and stateless parity suites.
//! The Gated DeltaNet block planner and recurrence heuristic in
//! `src/delta_net/plan.rs` are verified by `verus/mixers.rs`. See
//! `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/shared/block_plan.rs"]
pub mod shared_block_plan;

/// Mirrors the crate path `crate::models::shared::block_plan` used by the Qwen3.5 plan.
pub mod models {
    /// Model-independent family glue.
    pub mod shared {
        pub use crate::shared_block_plan as block_plan;
    }
}

#[path = "../src/models/qwen3_5/plan.rs"]
pub mod qwen3_5_plan;

#[path = "../src/models/qwen3_moe/plan.rs"]
pub mod qwen3_moe_plan;
