//! Verus harness for DeepSeek-V2 and Muse-Glimmer family logic.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/deepseek_v2/model/batch.rs`: the host planning of batched
//! stateful execution shared by DeepSeek-V2 and Muse-Glimmer. For all
//! admissible inputs the proofs establish freedom from integer overflow and
//! that
//!
//! * `row_maps` gathers exactly the valid columns into the first `n` block
//!   entries (padding entries gather `c_0`, or `0` for an empty row) and
//!   scatters each valid column back to its entry, every other column to the
//!   zero row `T`;
//! * `batch_extents` returns exactly the widest row `T = max_b n_b` and the
//!   furthest cache end `E = max_b (L_b + n_b)` (both attained, zero for no
//!   rows), failing only on a length mismatch or an overflowing end;
//! * `key_limits` returns nondecreasing limits in `1..=E` that are exactly
//!   `max(1, max_{j <= i, n_b > j} (L_b + j + 1))`: they cover every active
//!   row's causal key prefix and are attained by one, so bounded attention
//!   reads no key a query prefix cannot see;
//! * `query_slots` places entry `k < n_b` of row `b` at cache slot `L_b + k`
//!   and padding entries at `L_b`, within the key extent;
//! * `identity_layout`, `query_validity`, and `implicit_positions` decide the
//!   identity packing, the validity `k < n_b`, and the implicit positions
//!   `L_b + k` (zero at padding) exactly; `signed_indices` converts to `i64`
//!   exactly or reports an unrepresentable index; `clamp_limits` keeps
//!   limits nondecreasing and in `1..=K`;
//! * `rotary_contexts` returns the first active row's context (at least one),
//!   whether every active context equals it, and each row's own context;
//!
//! the specifications attached to `src/models/deepseek_v2/plan.rs`:
//!
//! * `sparse_layer` is exactly the `moe_layer_freq` / `first_k_dense_replace`
//!   rule and `any_sparse_layer` decides whether some layer qualifies;
//! * `eligible_experts` is exactly `(experts / groups) * top_groups` or
//!   reports overflow;
//! * `latent_cache_bytes` accepts a latent cache exactly when one layer's
//!   `batch * capacity * width` elements fit `i32::MAX` and the total bytes
//!   fit `usize`, returning the exact total, which also bounds the per-token
//!   bytes;
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

#[path = "../src/models/deepseek_v2/plan.rs"]
pub mod deepseek_plan;

#[path = "../src/models/muse_glimmer/schedule.rs"]
pub mod schedule;
