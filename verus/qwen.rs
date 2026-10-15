//! Verus harness for the Qwen3.5 and Qwen3-MoE family planning.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/qwen3_5/plan.rs` and `src/models/qwen3_moe/plan.rs`:
//!
//! * the derived hybrid schedule rule (full attention exactly when
//!   `i mod interval = interval - 1`, every layer for intervals zero and one);
//! * dense per-family cache slots: slot counts sum to the layer count, every
//!   slot lies below its family's size, and slots strictly increase within a
//!   family in schedule order (so each family's slots are exactly `0..size`);
//! * the compact right-padded batch row plan used by batched cached
//!   execution: slot `k` gathers valid column `c_k`, every gather index is in
//!   range, restoration maps `c_k` back to `k` and every other column to the
//!   zero slot `width`;
//! * the MTP cursor after a target truncation, which never grows and restores
//!   `target = MTP + 1` whenever the caches were aligned;
//! * the Qwen3-MoE sparse-layer predicate and sliding-window boundary.
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
