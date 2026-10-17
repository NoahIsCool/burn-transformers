//! Verus harness for the experiment and architecture interfaces' pure integer
//! logic.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/intervention/selection.rs`: semantic selection ranges.
//!   `resolve_slice` resolves a requested axis to its first declaring
//!   dimension with an exact declaration count, so a singly declared axis
//!   resolves to its unique dimension. `unit_range` never overflows (at
//!   `usize::MAX` it yields an empty range that validation rejects);
//!   `first_ill_formed_slice` accepts exactly selections that name distinct,
//!   singly declared axes with nonempty ranges, and otherwise names the
//!   first ill-formed slice; and `selection_ranges` expands
//!   each per-dimension range to exactly the requested one (or the whole
//!   extent when unselected) inside the emitted tensor's extents, so slicing
//!   cannot go out of bounds or silently select the wrong elements. In an
//!   accepted selection each requested range is the only one naming its
//!   dimension (`lemma_well_formed_slice_is_selected`).
//! - `src/experiment/bounds.rs`: positional extents (`extent_after`,
//!   `max + 1` without overflow), position/label range checks
//!   (`range_within`), the exact classification of a position readback
//!   (`position_extrema`: value count, then target range, then source
//!   minimum), broadcast compatibility of masks and attention planes
//!   (`broadcasts_to`: every axis one or equal), and staged block cursors
//!   (`follows_cursor`), whose consecutively accepted ranges run every block
//!   below the cursor exactly once and in order (`lemma_accepted_run_tiles`).
//!   `reduction_rows` is the product of every other extent, so
//!   `rows * width` is exactly the element count and no partial product
//!   overflows. Each pairwise-reduction level (`fold_level`) sends output
//!   column `j` exactly input columns `j` and `partner_start + j`, carries an
//!   odd width's middle column, and sends nothing else (every column is read
//!   exactly once); the whole reduction takes exactly `ceil(log2 w)` levels
//!   (`lemma_fold_depth`).
//! - `src/experiment/precedence.rs`: `last_match` resolves ordered overrides
//!   (per-layer attention policies and initialization path overrides) to
//!   exactly the last matching entry.
//! - `src/architecture/constraints.rs`: the integer rules of specification
//!   validation. Head layouts (`attention_heads`, `delta_net_heads`) are
//!   classified in precedence order (zero dimension, then indivisible
//!   groups), and a valid layout leaves no head over (`size * groups ==
//!   heads`) and maps every head to an in-range group (`lemma_heads_grouped`);
//!   Mamba2 group sizes (`mamba2_dimensions`), routed-expert selection counts
//!   (`expert_counts`, `1 <= k <= experts`), MLA dimensions and query-key
//!   width overflow (`mla_dimensions`, with the saturating builder width
//!   `mla_qk_head_dim`), cross-attention projection sizes
//!   (`cross_projection_fits`), and stack dimensions (`stack_dimensions`, the
//!   first zero extent) are decided exactly. `group_reorder` reports the
//!   first row of a group-preserving decode-state reorder that leaves its
//!   group, and an accepted reorder of a full batch reads only in-bounds rows
//!   (`lemma_same_group_in_bounds`).
//!
//! See `verus/README.md` for the method, toolchain, and command. Tensor
//! slicing, edits, device readbacks, the comparison of semantic axes (which
//! produces the match flags the resolution consumes), error formatting, and
//! floating-point reduction accuracy remain covered by the Rust test suite.

// Loop invariants are attributes on `while` expressions, which needs this
// feature under Verus only; ordinary builds never compile this harness.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/intervention/selection.rs"]
pub mod selection;

#[path = "../src/experiment/bounds.rs"]
pub mod bounds;

#[path = "../src/experiment/precedence.rs"]
pub mod precedence;

#[path = "../src/architecture/constraints.rs"]
pub mod constraints;
