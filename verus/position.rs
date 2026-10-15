//! Verus harness for position-index planning.
//!
//! Verifies, in place, the specifications attached to:
//!
//! * `src/position/index_plan.rs`: multimodal-RoPE axis selection and the T5
//!   displacement-table length and lookup index (in bounds, overflow-free, and
//!   displacement-preserving);
//! * `src/position/t5.rs`: T5 bucket assembly from direction, distance, and any
//!   proposed logarithmic offset (always a valid table index on the correct
//!   side of a bidirectional table, exact for short distances, and zero for
//!   causal future keys);
//! * `src/position/ids.rs`: batched position-ID derivation (no overflow or
//!   out-of-bounds access, `batch * sequence` outputs, each valid token at its
//!   row start plus the number of earlier valid tokens, padding at the
//!   sentinel, and each error returned only when its condition holds).
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
