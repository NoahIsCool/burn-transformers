//! Verus harness for the Qwen3.5 and Qwen3-MoE family planning.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/qwen3_5/plan.rs` and `src/models/qwen3_moe/plan.rs`:
//!
//! * the derived hybrid schedule rule (full attention exactly when
//!   `i mod interval = interval - 1`, every layer for intervals zero and one)
//!   and the whole derived schedule, which holds exactly
//!   `floor(L / interval)` full-attention layers;
//! * dense per-family cache slots: the slot of layer `i` is exactly the number
//!   of earlier layers of its family, the family sizes are exactly the two
//!   counts, and slots strictly increase within a family;
//! * the compact right-padded batch layout used by batched cached execution:
//!   column validation (all below `S`, strictly increasing) is decided
//!   exactly, `n_b`, `N = max_b n_b` (attained), and prefix detection are
//!   exact, slot `k` gathers valid column `c_k` and padding slots `c_0` (or
//!   `0`), restoration maps `c_k` back to `k` and every other column to the
//!   zero slot `N`, slot validity is `i < n_b`, slot positions are `W_b + i`
//!   with `None` exactly when one exceeds `usize`/`i64`, the uniform-cursor
//!   test is exact, and index conversion saturates at `i64::MAX`;
//! * MTP and speculative bookkeeping: the truncated MTP cursor is exactly
//!   `min(mtp, target - 1)` per row (restoring `target = MTP + 1`), the
//!   staging preflight reports the first row whose saturating request
//!   `(W_b - C_b) + n_b` exceeds the capacity, a cycle needs exactly `K + 1`
//!   staging slots, an MTP append ends exactly at `baseline + n` within the
//!   capacity, and a cycle starts only from batch one with
//!   `target = MTP + 1`, a matching history, and a suffix that fits, each
//!   failure naming its first violated condition;
//! * the call rotary context (the maximum active context, or one), the first
//!   out-of-vocabulary host token, and the gated query projection width;
//! * the Qwen3-MoE sparse-layer predicate, `mlp_only_layers` membership, and
//!   schedule existence (some block `i < L` is sparse).
//!
//! Both planning files are dependency-free, so they are included as plain
//! modules. Burn tensor programs driven by these plans (gathers, cache
//! writes, attention masks, expert dispatch) remain covered by the Rust
//! tests, including the batched-versus-per-row and stateless parity suites.
//! The DeltaNet block planner and recurrence heuristic live in
//! `src/delta_net.rs`, which depends on Burn, and are covered by proptests.
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/qwen3_5/plan.rs"]
pub mod qwen3_5_plan;

#[path = "../src/models/qwen3_moe/plan.rs"]
pub mod qwen3_moe_plan;
