//! Verus harness for position-index planning.
//!
//! Verifies, in place, the specifications attached to:
//!
//! * `src/position/index_plan.rs`: multimodal-RoPE axis selection equals
//!   `spec_mrope_axis`, and `lemma_mrope_axis_counts` proves that the first
//!   `half` frequencies give height `min(h, (half + 1) / 3)`, width
//!   `min(w, half / 3)`, and temporal the rest; `mrope_section_total` is the
//!   exact section sum or `None` on overflow; `rotary_operand_fits` and
//!   `split_half_width_fits` are exactly the rotary shape relations; the T5
//!   displacement-table length and lookup index are in bounds, overflow-free,
//!   and displacement-preserving, and `lemma_relative_table_entries_are_read`
//!   shows every entry is read by some pair; `plan_relative_table` accepts
//!   exactly the representable `[query, key]` geometries with a query offset
//!   (faults in check order), returns the exact length and anchor, and proves
//!   that pair `(i, j)` reads the entry whose anchor displacement is
//!   `j - (offset + i)`; `relative_bias_shape_fault` reports the first failed
//!   learned-table shape check;
//! * `src/position/t5.rs`: `checked_bucket_geometry` (behind
//!   `T5RelativePositionBuckets::new`) accepts exactly the valid geometries and
//!   otherwise returns the first failed check with its values;
//!   `signed_direction_distance` and `position_direction_distance` return the
//!   direction `key > query` and the exact distance `|key - query|` (including
//!   `i64::MIN` and the whole `usize` range); `combine_bucket` equals the
//!   reference formula `spec_bucket` for any distance and proposed logarithmic
//!   offset, so it is a valid table index on the correct side of a
//!   bidirectional table, exact for short distances, `min(E + offset, B - 1)`
//!   in the logarithmic region, and zero for causal future keys;
//! * `src/position/ids.rs`: batched and single-row position-ID derivation (no
//!   overflow or out-of-bounds access, `batch * sequence` outputs, each valid
//!   token at its row start plus the number of earlier valid tokens, padding at
//!   the sentinel, errors in a fixed precedence, and a position overflow
//!   reported at the first overflowing valid token in row-major order);
//! * `src/position/rope.rs`: `validate_dimensions` accepts exactly a positive,
//!   even rotary width no wider than a non-empty head (first failure
//!   otherwise); `check_scaling_width` adds exactly each policy's width rule
//!   (dynamic NTK above two, one LongRoPE factor per pair, proportional over
//!   the whole head); `validate_original_context` accepts exactly a non-zero
//!   trained context;
//! * `src/position/rope_scaling.rs`: `long_rope_frequency_count` accepts
//!   exactly equal table lengths in `1..=MAX_LONG_ROPE_FREQUENCIES` (the bound
//!   behind the inline-table slices), and `clamp_rotated_pairs` is exactly
//!   `min(pairs, head / 2)`.
//!
//! Not proved here: the floating-point logarithm that proposes a T5 offset,
//! RoPE frequency arithmetic and validation of floating-point parameters
//! (theta, factors, amplitudes, YaRN correction ranges), and Burn tensor
//! construction, which the unit tests cover. Loop invariants are attributes on `while` statements, which
//! needs `proc_macro_hygiene` under the Verus compiler only.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
#![feature(proc_macro_hygiene)]

#[path = "../src/position/index_plan.rs"]
pub mod index_plan;

#[path = "../src/position/t5.rs"]
pub mod t5;

#[path = "../src/position/ids.rs"]
pub mod ids;

#[path = "../src/position/rope.rs"]
pub mod rope;

#[path = "../src/position/rope_scaling.rs"]
pub mod rope_scaling;
