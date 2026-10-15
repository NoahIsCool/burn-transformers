//! Verus harness for the Gemma layer schedule and per-layer rotary sets.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/gemma/plan.rs`: the overflow-free default local/global
//! schedule equals the reference's `(l + 1) mod p != 0` for every layer and
//! every positive period (Gemma 2's alternation and Gemma 3's
//! `sliding_window_pattern`), and every layer's rotary set index names its
//! layer type's set and lies below the number of sets (one when both layer
//! types share a set, two otherwise), so no layer can index past the call's
//! rotary tables. See `verus/README.md` for the method, toolchain, and
//! command.
//!
//! The planning file is dependency-free, so it is included as a single
//! module. The tensor code driven by these plans (the per-layer-type rotary
//! tables and sliding masks) remains covered by
//! `src/models/gemma/{architecture,model}_tests.rs`.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/gemma/plan.rs"]
pub mod plan;
