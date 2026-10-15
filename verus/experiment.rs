//! Verus harness for the experiment interface's pure index and bounds logic.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/intervention/selection.rs`: semantic selection ranges. Length-one
//!   selections never overflow; accepted selections name distinct, singly
//!   declared axes with nonempty ranges; and every expanded per-dimension range
//!   lies inside the emitted tensor's extents, so slicing cannot go out of bounds.
//! - `src/experiment/bounds.rs`: positional extents (`max + 1` without
//!   overflow), position/label range checks, broadcast compatibility of masks
//!   and attention planes, staged block cursors, and the pairwise reduction
//!   schedule, which strictly shrinks the width and therefore terminates.
//!
//! See `verus/README.md` for the method, toolchain, and command. Tensor slicing,
//! edits, device readbacks, the conversion from semantic axes to dimensions,
//! and floating-point reduction accuracy remain covered by the Rust test suite.

// Loop invariants are attributes on `while` expressions, which needs this
// feature under Verus only; ordinary builds never compile this harness.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/intervention/selection.rs"]
pub mod selection;

#[path = "../src/experiment/bounds.rs"]
pub mod bounds;
