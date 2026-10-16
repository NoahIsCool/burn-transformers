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
//! and across a size range.
//!
//! It also verifies `src/generation/layout.rs`: exact valid-entry counts,
//! last valid indices, and any-valid scans of row-major masks (with the first
//! empty row reported); prefill-chunk row activity and the skip, prefill, or
//! decode choice; the logit-bias, hard-constraint, history-count, and
//! decode-step scatter layouts (each row in its own segment, padded with
//! no-op entries); the split of a packed selection readback into IDs and
//! scores; and the conversion of a history value into a vocabulary token ID.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/generation/bookkeeping.rs"]
pub mod bookkeeping;

#[path = "../src/generation/layout.rs"]
pub mod layout;
