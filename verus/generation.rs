//! Verus harness for the generation coordinators' integer bookkeeping.
//!
//! Verifies, in place, the specifications attached to
//! `src/generation/bookkeeping.rs`: longest-suffix stop matching and finish
//! precedence, prefill chunk partitioning and rectangular row spans, beam-row
//! group preservation and initial expansion, the beam candidate frontier, the
//! explicit-position preflight, the accepted-prefix arithmetic of one
//! speculative-decoding cycle, and prompt-lookup drafting. See
//! `verus/README.md` for the method, toolchain, and command.

#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/generation/bookkeeping.rs"]
pub mod bookkeeping;
