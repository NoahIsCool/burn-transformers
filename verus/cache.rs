//! Verus harness for the cache's pure planning and lifecycle code.
//!
//! Verifies, in place, the specifications attached to:
//!
//! * `src/cache/ring.rs`: ring write and read planning, including the slot
//!   invariant `slot(p) = p mod capacity` for every token written or read.
//! * `src/cache/lengths.rs`: the length ledger behind every decode state and
//!   speculative transaction. Each transition (advance, begin, commit,
//!   rollback, truncate, reorder, reset) preserves `committed <= working <=
//!   capacity`, computes its documented formula without overflow, and is
//!   rejected exactly when its rule is violated.
//! * `src/cache/retention.rs`: ring and recurrent rewind predicates, proved
//!   sound and complete against the ring eviction model, and the working
//!   window that splits a speculative read into committed and staged parts.
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
