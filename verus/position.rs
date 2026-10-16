//! Verus harness for position-index planning.
//!
//! Verifies, in place, the specifications attached to:
//!
//! * `src/position/index_plan.rs`: multimodal-RoPE axis selection and the T5
//!   displacement-table length and lookup index (in bounds, overflow-free, and
//!   displacement-preserving);
//! * `src/position/t5.rs`: `checked_bucket_geometry` (behind
//!   `T5RelativePositionBuckets::new`) accepts exactly the valid geometries and
//!   otherwise returns the first failed check with its values;
//!   `signed_direction_distance` and `position_direction_distance` return the
//!   direction `key > query` and the exact distance `|key - query|` (including
//!   `i64::MIN` and the whole `usize` range); `combine_bucket` equals the
//!   reference formula `spec_bucket` for any distance and proposed logarithmic
//!   offset, so it is a valid table index on the correct side of a
//!   bidirectional table, exact for short distances, `min(E + offset, B - 1)`
//!   in the logarithmic region, and zero for causal future keys;
//! * `src/position/ids.rs`: batched and single-row position-ID derivation (no
//!   overflow or out-of-bounds access, `batch * sequence` outputs, each valid
//!   token at its row start plus the number of earlier valid tokens, padding at
//!   the sentinel, errors in a fixed precedence, and a position overflow
//!   reported at the first overflowing valid token in row-major order).
//!
//! Not proved here: the floating-point logarithm that proposes a T5 offset,
//! RoPE frequency arithmetic, and Burn tensor construction, which the unit
//! tests cover. Loop invariants are attributes on `while` statements, which
//! needs `proc_macro_hygiene` under the Verus compiler only.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
#![feature(proc_macro_hygiene)]

#[path = "../src/position/index_plan.rs"]
pub mod index_plan;

#[path = "../src/position/t5.rs"]
pub mod t5;

#[path = "../src/position/ids.rs"]
pub mod ids;
