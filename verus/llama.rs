//! Verus harness for the Llama-family schedule and batched-step planning.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/llama/plan.rs`: the SmolLM3 NoPE period and Qwen sliding-layer
//! rules and the whole-model schedules built from them (the NoPE schedule
//! equals `(l + 1) mod k != 0` per layer and has exactly `⌊L / k⌋` NoPE
//! layers; an explicit `no_rope_layers` list is accepted exactly when every
//! entry is 0 or 1; Qwen's default sliding layers are the suffix
//! `l >= max_window_layers`; SmolLM3's are its NoPE layers when enabled; the
//! first sliding layer without a window is the earliest one), the derived
//! `head_dim = hidden / heads`, the numbering of layers inside their KV cache groups (each slot is
//! the count of earlier layers of its group, so slots are dense and in
//! range), the contiguous-run test of a row's valid columns (overflow-free),
//! the cache positions of a batched step's block slots (exact
//! `start + (t - column)` inside the run, a bounded fill elsewhere, and no
//! overflow), the compaction gather of interior-padded rows, and the two
//! restoring gathers (from a compact row, and from a run at any block
//! offset) with their appended zero slot. See `verus/README.md` for the
//! method, toolchain, and command.
//!
//! The planning file is dependency-free, so it is included as a single module.
//! The Burn tensor programs driven by these plans (block slices and gathers,
//! cache write-ahead and ring segments, masks, and the restoring gather)
//! remain covered by `src/models/llama/model/step_tests.rs` and the parity
//! tests in `src/models/llama/model/decode_tests.rs`.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/llama/plan.rs"]
pub mod plan;
