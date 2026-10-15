//! Verus harness for attention's pure planning and shape arithmetic.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/attention/plan.rs`: query-chunk planning against score-element
//!   limits (exact peak size, both limits respected, the largest chunk they
//!   admit, `ceil(Q / c)` chunks with no overflow, and each error exactly when
//!   every earlier check passed), chunk ranges that partition the query rows,
//!   sliding-window key ranges that are exactly the union of their rows'
//!   visible keys, the row cap and window key bound of bounded execution
//!   (with a lemma that every planned chunk re-plans as one call), and chunk
//!   frames that preserve every position's absolute chunk index with small,
//!   overflow-free parameters;
//! - `src/attention/widths.rs`: grouped-query projection widths, whose
//!   success implies exact widths, head grouping, and representable element
//!   counts for every projection matrix.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]

#[path = "../src/attention/plan.rs"]
pub mod plan;

#[path = "../src/attention/widths.rs"]
pub mod widths;
