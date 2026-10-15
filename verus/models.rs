//! Verus harness for model-family glue shared by every architecture.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/shared/plan.rs`:
//!
//! - `valid_columns`: a mask row compacts to exactly the recursive list of
//!   its valid columns (`spec_valid_columns`), which is strictly increasing,
//!   in range, and contains every valid column;
//! - `plan_scatter` and helpers: scatter plans restore the rectangular layout
//!   (identity, one contiguous run, or a gather map with an appended zero
//!   row), and the identity or a single run is chosen whenever it applies;
//! - `row_valid_columns` and `row_last_columns`: every row of an optional
//!   `[batch, sequence]` mask lists exactly its valid columns, and reports its
//!   last valid column or `sequence - 1` when empty (dense when no mask);
//! - `row_contexts` and `plan_host_rows`: the host side of every stateful
//!   prefill and decode call returns exactly the specified outcome, with
//!   errors in precedence order (rejected empty row, first impossible append,
//!   first row with an out-of-range valid token, out-of-range valid position,
//!   or unrepresentable extent) and otherwise the per-row columns, counts,
//!   last columns, and rotary extents `1 + max p` or `L + n`;
//! - `split_readback`: the fused readback splits into its mask, token, and
//!   position segments exactly;
//! - `plan_last_selection`, `positions_within`, and `last_valid_column`:
//!   logits selections are classified and validated exactly;
//! - `count_fills_width`, `signed_indices`, `implicit_positions`,
//!   `context_from_maximum`, `range_within`, `value_range`, and
//!   `plan_appends`: padding detection, signed index conversion, implicit
//!   cached positions `baseline + k`, the rotary extent `max + 1`, attained
//!   value ranges over valid columns, and per-row cache appends against a
//!   capacity are exact and overflow-free.
//!
//! See `verus/README.md` for the method, toolchain, and command.
//!
//! The planning file is dependency-free, so it is included as a single module.
//! Burn tensor programs driven by these plans (selection, concatenation,
//! gathers, and the fused validation readback itself), the readback slot
//! bookkeeping of `DeferredChecks`, and the shape preconditions established by
//! the unverified callers (`[batch, sequence]` host buffers and one cursor per
//! row, checked there at runtime) remain covered by the Rust tests in
//! `src/models/shared/*_tests.rs` and every family's parity suite.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/shared/plan.rs"]
pub mod plan;
