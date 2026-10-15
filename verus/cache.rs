//! Verus harness for the cache's pure planning code.
//!
//! Verifies, in place, the specifications attached to `src/cache/ring.rs`:
//! ring write and read planning, including the slot invariant
//! `slot(p) = p mod capacity` for every token written or read. See
//! `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]

#[path = "../src/cache/ring.rs"]
pub mod ring;
