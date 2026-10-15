//! Verus harness for the experiment interface's pure index and bounds logic.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/intervention/selection.rs`: semantic selection ranges. Length-one
//!   selections never overflow; accepted selections name distinct, singly
//!   declared axes with nonempty ranges; and each expanded per-dimension range
//!   is exactly the requested one (or the whole extent when unselected) and
//!   lies inside the emitted tensor's extents, so slicing cannot go out of
//!   bounds or silently select the wrong elements.
//! - `src/experiment/bounds.rs`: positional extents (`max + 1` without
//!   overflow), position/label range checks, broadcast compatibility of masks
//!   and attention planes, and staged block cursors, whose consecutively
//!   accepted ranges run every block below the cursor exactly once and in
//!   order. Each pairwise-reduction level partitions the row into added left
//!   and right blocks plus the carried odd middle column (every column is read
//!   exactly once), and the whole reduction takes exactly `ceil(log2 w)` levels.
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
