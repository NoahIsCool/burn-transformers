//! Verus harness for model-family glue shared by every architecture.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/shared/plan.rs`: valid-column compaction of padded mask rows,
//! scatter plans that restore the rectangular layout (identity, one contiguous
//! run, or a gather map with an appended zero row), explicit logits-selection
//! checks and last-valid-column selection, signed index conversion, implicit
//! cached positions `baseline + k`, the rotary context extent `max + 1`, exact
//! value ranges over valid columns, and per-row cache-append planning against
//! a capacity. See `verus/README.md` for the method, toolchain, and command.
//!
//! The planning file is dependency-free, so it is included as a single module.
//! Burn tensor programs driven by these plans (selection, concatenation,
//! gathers, and the fused validation readback) remain covered by the Rust
//! tests in `src/models/shared/*_tests.rs` and every family's parity suite.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/shared/plan.rs"]
pub mod plan;
