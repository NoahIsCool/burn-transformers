//! Verus harness for the generation coordinators' integer bookkeeping.
//!
//! Verifies, in place, the specifications attached to
//! `src/generation/bookkeeping.rs`: longest-suffix stop matching and finish
//! precedence, prefill chunk partitioning and rectangular row spans, beam-row
//! group preservation and initial expansion, the beam candidate frontier
//! capacity and its partition into finalized and live ranks, probability-row
//! spans, the explicit-position preflight and per-step position advance, the
//! cache-capacity preflight, the accepted-prefix arithmetic of one
//! speculative-decoding cycle, and prompt-lookup drafting for one n-gram size
//! and across a size range. See `verus/README.md` for the method, toolchain,
//! and command.

#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/generation/bookkeeping.rs"]
pub mod bookkeeping;
