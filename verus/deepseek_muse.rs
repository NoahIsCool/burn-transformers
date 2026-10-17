//! Verus harness for DeepSeek-V2 and Muse-Glimmer family logic.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/deepseek_v2/plan.rs`:
//!
//! * `sparse_layer` is exactly the `moe_layer_freq` / `first_k_dense_replace`
//!   rule and `any_sparse_layer` decides whether some layer qualifies;
//! * `eligible_experts` is exactly `(experts / groups) * top_groups` or
//!   reports overflow;
//! * `latent_cache_bytes` accepts a latent cache exactly when every
//!   intermediate product (`batch * capacity`, then `* width`, `* layers`,
//!   and `* scalar_bytes`) is representable and one layer's
//!   `batch * capacity * width` elements fit `i32::MAX`, each error naming the
//!   first failing check; it returns the exact total bytes and the per-token
//!   bytes `width * layers * scalar_bytes` (saturated at `usize::MAX`, and
//!   exact with `total = batch * capacity * per_token` whenever the cache has
//!   a row and a slot), which the decode state stores as its byte accessors;
//!
//! and the specifications attached to `src/models/muse_glimmer/schedule.rs`:
//!
//! * `is_global_layer` is the backward-counted `(n - 1 - i) mod 4 = 0` rule
//!   without underflow, and the last layer is always global;
//!   `derived_global_layers` builds the whole schedule, which holds exactly
//!   `ceil(n / 4)` global layers;
//! * `group_slots` gives every layer the number of earlier layers of its kind,
//!   so each cache group's slots are dense and strictly increasing in layer
//!   order (an injective schedule);
//! * `ring_chunk_keys` returns, for cached sliding attention over
//!   `[ring history | fresh block]`, a key slice that contains every fresh key
//!   visible to the chunk's rows, the whole ring while any row can still see
//!   history, no history otherwise, and at most `H + W - 1 + rows` keys;
//!   every fresh key it covers is visible to some row, so without history
//!   it is exactly the rows' window union;
//! * `planning_bound` is exactly the saturating per-rule formula clamped to
//!   `K`, and no ring slice (or aligned window union of `rows + W - 1` keys)
//!   of a chunk the policy admits exceeds it;
//! * `prefix_chunk_limit` returns the chunk's last row limit exactly when
//!   the limits cover every query row within `1..=K`; `plane_elements` is the
//!   exact checked product of a score plane's extents;
//! * `cache_groups` counts each layer kind exactly, allocates the ring
//!   (group `0`) exactly with local layers and a non-zero window and the full
//!   group after it exactly with global layers, and maps every layer to its
//!   kind's existing group and dense slot.
//!
//! The batched-row host plan both families drive their cached steps with,
//! `src/models/shared/block_plan.rs`, is family-independent and verified by
//! `verus/models.rs`. See `verus/README.md` for the method, toolchain, and
//! command. Burn tensor programs driven by these plans (gathers, scatters,
//! cache writes, and attention) remain covered by the Rust tests, in
//! particular `batched_rows_match_independent_rows`.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/deepseek_v2/plan.rs"]
pub mod deepseek_plan;

#[path = "../src/models/muse_glimmer/schedule.rs"]
pub mod schedule;
