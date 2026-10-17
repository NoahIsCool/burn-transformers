//! Verus harness for the modular (custom) model family.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/custom/plan.rs`:
//!
//! - `displacement_table`: the clamped T5 displacement table has radius
//!   exactly `R = min(max_distance, C - 1)` and length `2R + 1`, and is absent
//!   exactly for an empty context or a length beyond `usize`;
//!   `portable_displacement_table` keeps that table exactly when
//!   `2R + 1 <= i32::MAX`, so its radius converts to `i64` losslessly;
//! - `displacement_slot`: the device index `clamp(d, -R, R) + R` lies in the
//!   table, equals `d + R` for every in-range displacement, and is `0` or
//!   `2R` beyond either end; it is the scalar specification of the device
//!   expression, compiled only for tests and proofs
//!   (`#[cfg(any(test, verus_keep_ghost))]`);
//! - `slot_displacement`: slot `k` stores displacement `k - R`, the inverse of
//!   `displacement_slot` on the table;
//! - `plan_row_append`: one row's append counts its valid columns, keeps the
//!   new length within capacity (or reports the overflow), and gives valid
//!   column `c` the slot `L + rank(c)`, so valid columns fill `L..L + n` in
//!   strictly increasing order while masked columns read slot zero;
//! - `plan_history_append`: the batch append returns exactly the first policy
//!   error in order (empty prompt row, entirely masked step, capacity with the
//!   requested length when representable, never-started row), and otherwise
//!   per-row counts `n_r`, lengths `L_r + n_r <= capacity`, and the rank slots
//!   of every row; its helpers `count_valid_cells`, `grid_row_counts`,
//!   `first_zero`, and `first_nonzero` return exact counts and the first
//!   matching index;
//! - `plan_history_writes`: the write window `[start, end)` is the tightest
//!   one over advancing rows (`[0, 0)` when none advances), cell `(r, j)` is
//!   written exactly when `L_r <= start + j < L_r + n_r`, and a written cell
//!   gathers the unique valid column whose rank is `start + j - L_r`;
//! - `plan_history`: reports a validity grid that does not hold exactly
//!   `rows * step` cells (computed without overflow) as a layout error, and
//!   otherwise composes both, discharging the window's precondition from the
//!   append's postcondition, so its decode-state caller assumes nothing;
//! - `full_range`, `valid_range`, and `first_position_violation`: decode
//!   validation reads attained token bounds and reports the first row whose
//!   valid explicit positions leave `0..limit`, ignoring masked cells;
//! - `prefix_validity`: the recomputed history prefix has width `max L_r`
//!   and marks cell `(r, c)` valid exactly when `c < L_r`.
//!
//! See `verus/README.md` for the method, toolchain, and command.
//!
//! Not covered here and left to the Rust tests in
//! `src/models/custom/*_tests.rs`: the host bucket evaluation that fills the
//! displacement table (its exactness relies on the floating-point saturation
//! argument in the module documentation and is checked exhaustively against
//! the pairwise reference) and every Burn tensor program driven by these
//! plans.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/custom/plan.rs"]
pub mod plan;
