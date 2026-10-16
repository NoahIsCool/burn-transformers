//! Verus harness for the Llama-family schedule, cache layout, and batched-step planning.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/llama/plan.rs`:
//!
//! - schedules: the SmolLM3 NoPE period and Qwen sliding-layer rules, and
//!   the whole-model schedules built from them (the NoPE schedule equals
//!   `(l + 1) mod k != 0` per layer and has exactly `⌊L / k⌋` NoPE layers; an
//!   explicit `no_rope_layers` list is accepted exactly when every entry is
//!   0 or 1; Qwen's default sliding layers are the suffix
//!   `l >= max_window_layers`; SmolLM3's are its NoPE layers when enabled;
//!   the reported sliding layer without a window is the first one), and the
//!   derived `head_dim = hidden / heads`; the checkpoint contract's
//!   extents (`checkpoint_extent` is `Some(v)` exactly when `v` fits `u64`,
//!   and `projection_extent` is exactly `heads · head_dim` or names the
//!   `usize` or `u64` overflow);
//! - cache layout: `plan_ring` puts exactly the layers sliding over
//!   `0 < W < capacity` keys in the ring, succeeds exactly when no layer is
//!   chunked and all ring windows agree (reporting the first offending
//!   layer otherwise), and returns the shared window (zero exactly when the
//!   ring is empty); `cache_slots` numbers each group's layers by an
//!   order-preserving bijection onto `0..size`, and the two sizes partition
//!   the layers; `group_indices` numbers the allocated groups densely, full
//!   storage first; `truncation_leaves_bytes` is exactly "some row is
//!   shortened to a nonzero length", the condition that clears clean tails;
//! - rotary sets: `used_rotary_sets` marks set `s` exactly when some layer
//!   rotates with it, so each call evaluates exactly the tables it needs;
//!   `tightest_limit` returns the first used set with the smallest
//!   cached-context limit (`Iterator::min_by_key`'s choice), `None` exactly
//!   when no used set has one; `widest_context` is the largest active
//!   row's context, 1 when no row is active;
//! - batched steps: `plan_step` plans the whole `[batch, Q]` block. It fails
//!   with `NoActiveRow` exactly when no row has a token; when every row is a
//!   run it slices the minimal input window `[lo, hi)` (some row starts at
//!   `lo`, some ends at `hi`) and places row `b`'s `k`-th valid column at
//!   slot `c_b + k` (input column `lo + c_b + k`); otherwise it gathers into a
//!   block as wide as the longest row with every `c_b = 0`; every row fits
//!   (`c_b + n_b <= Q`), which discharges the contracts of the per-row
//!   plans it composes, and a position error names a row whose end overflows.
//!   `plan_row_step` gives each row exactly its slot positions (as `i64`),
//!   slot flags `c_b <= t < c_b + n_b`, restoring gather, compaction gather,
//!   and in-range last-token slot; `signed_indices` converts gather indices
//!   to `i64` exactly or rejects the batch. The per-row plans are the contiguous-run
//!   test, the cache positions of a block's slots (exact
//!   `start + (t - column)` inside the run, the fill position elsewhere, and
//!   no overflow), the compaction gather of interior-padded rows, and the two
//!   restoring gathers (from a compact row, and from a run at any block
//!   offset) with their appended zero slot.
//!
//! See `verus/README.md` for the method, toolchain, and command.
//!
//! The planning file is dependency-free, so it is included as a single module.
//! The Burn tensor programs driven by these plans (block slices and gathers,
//! cache write-ahead and ring segments, masks, and the restoring gather)
//! remain covered by `src/models/llama/model/step_tests.rs` and the parity
//! tests in `src/models/llama/model/decode_tests.rs`; the mapping from
//! `AttentionConnectivity` to the plan's `LayerReach` and the cache
//! specifications built from the plan are covered by the decode tests.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/llama/plan.rs"]
pub mod plan;
