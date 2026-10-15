//! Verus harness for the cache's pure planning and lifecycle code.
//!
//! Verifies, in place, the specifications attached to:
//!
//! * `src/cache/ring.rs`: standard and mirrored ring write and read planning,
//!   staging plans, sliding-window starts, and staged-commit planning,
//!   including the slot invariant `slot(p) = p mod capacity` for every token
//!   written, mirrored, or read. Mirrored writes and staged commits cover
//!   exactly the retained suffix, and every planner fails exactly when its
//!   overflow or bound condition holds.
//! * `src/cache/lengths.rs`: the length ledger behind every decode state and
//!   speculative transaction. Each transition (advance, begin, commit,
//!   rollback, truncate, reorder, reset) preserves `committed <= working <=
//!   capacity`, computes its documented formula without overflow, and is
//!   rejected exactly when its rule is violated.
//! * `src/cache/retention.rs`: ring and recurrent rewind predicates, proved
//!   sound and complete against the ring eviction model (row scans name the
//!   lowest offending row), the working window
//!   that splits a speculative read into committed and staged parts, and the
//!   absolute position held by each ring slot in a batched slot-order read.
//! * `src/cache/rows.rs`: grouping of batch rows into runs that share one
//!   tensor assignment; runs are ordered, disjoint, carry their rows'
//!   parameters, and cover exactly the rows with new tokens.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/cache/lengths.rs"]
pub mod lengths;
#[path = "../src/cache/retention.rs"]
pub mod retention;
#[path = "../src/cache/ring.rs"]
pub mod ring;
#[path = "../src/cache/rows.rs"]
pub mod rows;
