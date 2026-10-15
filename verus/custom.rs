//! Verus harness for the modular (custom) model family.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/custom/plan.rs`:
//!
//! - `displacement_table`: the clamped T5 displacement table has radius
//!   `R = min(max_distance, C - 1)` and length `2R + 1` without overflow, and
//!   an empty context has no table;
//! - `displacement_slot`: the device index `clamp(d, -R, R) + R` lies in the
//!   table and equals `d + R` for every in-range displacement;
//! - `slot_displacement`: slot `k` stores displacement `k - R`, the inverse of
//!   `displacement_slot` on the table;
//! - `plan_row_append`: one row's append counts its valid columns, keeps the
//!   new length within capacity (or reports the overflow), and gives valid
//!   column `c` the slot `L + rank(c)`, so valid columns fill `L..L + n` in
//!   strictly increasing order while masked columns read slot zero.
//!
//! See `verus/README.md` for the method, toolchain, and command.
//!
//! Not covered here and left to the Rust tests in
//! `src/models/custom/*_tests.rs`: the batch-level append policy and write
//! window (`plan_history_append`, `plan_history_writes`, checked against a
//! naive reference by a property test), the host bucket evaluation that fills
//! the displacement table (its exactness relies on the floating-point
//! saturation argument in the module documentation and is checked
//! exhaustively against the pairwise reference), and every Burn tensor
//! program driven by these plans.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/custom/plan.rs"]
pub mod plan;
