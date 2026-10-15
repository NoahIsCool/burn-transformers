//! Verus harness for the experiment and architecture interfaces' pure integer logic.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/intervention/selection.rs`: semantic selection ranges. A requested
//!   axis resolves to its first declaring dimension with an exact declaration
//!   count, so a singly declared axis resolves to its unique dimension.
//!   Length-one selections never overflow; accepted selections name distinct,
//!   singly declared axes with nonempty ranges; and each expanded
//!   per-dimension range is exactly the requested one (or the whole extent
//!   when unselected) and lies inside the emitted tensor's extents, so slicing
//!   cannot go out of bounds or silently select the wrong elements. In an
//!   accepted selection each requested range is the only one naming its
//!   dimension.
//! - `src/experiment/bounds.rs`: positional extents (`max + 1` without
//!   overflow), position/label range checks, broadcast compatibility of masks
//!   and attention planes, and staged block cursors, whose consecutively
//!   accepted ranges run every block below the cursor exactly once and in
//!   order. Each pairwise-reduction level partitions the row into added left
//!   and right blocks plus the carried odd middle column (every column is read
//!   exactly once), and the whole reduction takes exactly `ceil(log2 w)` levels.
//! - `src/architecture/constraints.rs`: the integer rules of specification
//!   validation. Head layouts are classified in precedence order (zero
//!   dimension, then indivisible groups), and a valid layout maps every head
//!   to an in-range group of exactly `heads / groups` members; Mamba2 group
//!   sizes, routed-expert selection counts, MLA query-key width overflow (and
//!   its saturating builder width), cross-attention projection sizes, and
//!   stack dimensions are decided exactly. Group-preserving decode-state
//!   reorders report the first row leaving its group, and an accepted reorder
//!   of a full batch reads only in-bounds rows.
//!
//! See `verus/README.md` for the method, toolchain, and command. Tensor slicing,
//! edits, device readbacks, the comparison of semantic axes (which produces the
//! match flags the resolution consumes), error formatting, and floating-point
//! reduction accuracy remain covered by the Rust test suite.

// Loop invariants are attributes on `while` expressions, which needs this
// feature under Verus only; ordinary builds never compile this harness.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/intervention/selection.rs"]
pub mod selection;

#[path = "../src/experiment/bounds.rs"]
pub mod bounds;

#[path = "../src/architecture/constraints.rs"]
pub mod constraints;
