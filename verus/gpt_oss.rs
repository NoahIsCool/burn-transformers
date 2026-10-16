//! Verus harness for gpt-oss: exact MXFP4 decoding arithmetic and the family's planning.
//!
//! Verifies, in place, the specifications attached to `src/checkpoint/mxfp4.rs`.
//! Throughout, `E`, `R`, and `G` are the expert, row, and block-group axes of
//! a `[E, R, G, 16]` blocks tensor, a row decodes to `32 G` columns, and
//! `describes(g)` is the invariant of a validated `Mxfp4Geometry` (nonzero
//! axes, `columns == 32 G`, and `E · R · 32 G` within both `usize` and `u64`).
//!
//! - `mxfp4_geometry` returns `Ok` if and only if `accepts` holds: blocks
//!   `[E, R, G, 16]`, scales `[E, R, G]`, nonzero axes, and a decoded value
//!   count `E · R · 32 G` that fits both `usize` and `u64` (soundness and
//!   completeness). Each error is exactly the first violated rule:
//!   `Blocks` exactly when the blocks are not rank four with a trailing 16;
//!   `Scales` exactly when the blocks are well formed but the scales are not
//!   the blocks' leading three axes; `Empty` exactly when both shapes are
//!   well formed but an axis is zero; `Overflow` exactly when every earlier
//!   rule holds but the value count does not fit. An accepted geometry
//!   satisfies `describes`, with extents equal to the shape's axes and
//!   `columns == 32 G`.
//! - `decoded_shape` has the same `Ok` condition and the same four error
//!   classes, and returns the transposed shape `[E, 32 G, R]`.
//! - `Mxfp4Geometry::values`, `Mxfp4Geometry::block_bytes`, and
//!   `Mxfp4Geometry::scale_count` return exactly `E · R · 32 G`,
//!   `16 · E · R · G`, and `E · R · G`; `values_of` computes the value count
//!   without overflow (`lemma_count_factors` relates the three counts).
//! - `fp4_codes` splits a byte into its low nibble (`byte mod 16`, the even
//!   decoded column) and high nibble (`byte / 16`, the odd column), both
//!   below 16, with `low + 16 · high == byte`.
//! - `exact_value_bits` returns the binary64 encoding of the exact decoded
//!   value `fp4(code) · 2^(scale - 127)` for every `scale < 255` and every
//!   `code < 16`: the code's sign bit; a signed zero for magnitude code
//!   `m = 0`; the biased exponent `scale + 895` with no fraction for `m = 1`;
//!   and the biased exponent `⌊m / 2⌋ + scale + 895` with fraction
//!   `(code mod 2) · 2^51` for `m >= 2`. Scale code 255 yields the canonical
//!   positive quiet NaN for `m = 0` and `±inf` otherwise, as Transformers'
//!   infinite factor does.
//! - `scale_offset` and `block_offset` return the row-major offsets
//!   `(e · R + r) · G + g` of a block's scale and 16 times that for its
//!   bytes; the scale offset is below `E · R · G` and the block's 16 bytes
//!   end within the `16 · E · R · G` block bytes. `transposed_offset`
//!   returns `(e · 32 G + c) · R + r`, below `E · R · 32 G`: the
//!   specification of the `[E, 32 G, R]` output layout, compiled only for
//!   tests and proofs (`#[cfg(any(test, verus_keep_ghost))]`).
//! - `table_index` returns `16 · scale + code mod 16`, below the 4096-entry
//!   decode table, and `table_coordinates` inverts it on `0..4096`
//!   (`code < 16` and `16 · scale + code == index`), so the two are a
//!   bijection between table entries and `(scale, code)` pairs and filling
//!   the table in index order visits every pair exactly once.
//! - `decode_partition` splits the `E · G` decode units (one per expert and
//!   block group) over threads for a `W`-byte target scalar: the thread
//!   count is exactly 1 below `PARALLEL_MIN_VALUES` (`2^20`) decoded values
//!   and `min(available, E · G)` otherwise, so it lies in
//!   `1..=min(available, E · G)`; `unit_bytes == 32 · R · W` and the `E · G`
//!   units tile the `E · R · 32 G · W` output bytes exactly;
//!   `units_per_thread` is exactly `k = ⌈E · G / threads⌉`, in `1..=E · G`,
//!   and `chunk_bytes == k · unit_bytes` never exceeds the output; the
//!   `n = ⌈E · G / k⌉` chunks number at most `threads` and cover every unit
//!   exactly once (`(n - 1) · k < E · G <= n · k`), with every product
//!   overflow-free (`lemma_unit_sizes`, `lemma_chunk_cover`). The ceilings in
//!   these statements are the shared `ceiling` of `src/division.rs`, which is
//!   included at its crate path and re-verified here.
//! - `unit_coordinates` maps unit `u` to `(e, g)` with `e < E`, `g < G`, and
//!   `u == e · G + g`, and proves that decoded column `32 g + j` of row `r`
//!   (`j < 32`, `r < R`) sits at output element `u · 32 R + j · R + r`, so
//!   unit `u` owns exactly the contiguous elements `[u · 32 R, (u + 1) · 32 R)`.
//! - `unit_value_bytes` returns the byte offsets `(2 b · R + r) · W` and
//!   `((2 b + 1) · R + r) · W` of the two values of block byte `b < 16` of
//!   row `r < R`; the even slot ends at or before the odd one starts and
//!   both end within the unit's `32 · R · W` bytes, and
//!   `lemma_value_slots_injective` proves that distinct `(column, row)` pairs
//!   occupy distinct elements `column · R + row`, so every unit element is
//!   written exactly once.
//!
//! It also verifies `src/models/gpt_oss/plan.rs`:
//!
//! - `default_layer_slides` equals the reference's default schedule
//!   `bool((l + 1) % 2)` without overflow, and `default_schedule` applies it
//!   to every layer: consecutive layers alternate from a sliding layer and
//!   exactly `⌊L / 2⌋` layers are full-attention layers;
//! - `ring_assignment` activates the ring exactly for a shared window
//!   `0 < W < capacity` and then puts exactly the sliding layers in it;
//!   `cache_slots` (shared, see below) numbers each KV group's layers by an
//!   order-preserving bijection onto `0..size`, and the two sizes partition
//!   the layers; `group_indices` numbers the allocated groups densely, full
//!   storage first; `truncation_leaves_bytes` is exactly "some row is
//!   shortened to a nonzero length", the condition that clears clean tails;
//! - `fused_expert_elements` returns `2f · d · E` exactly when every partial
//!   product (`2f`, `2f · d`, `2f · d · E`) fits `usize` (the configuration's
//!   overflow check); `checkpoint_extent` is `Some(v)` exactly when `v` fits
//!   `u64`, and `projection_extent` is exactly `heads · head_dim` or names
//!   the `usize` or `u64` overflow; `derived_head_dim` is `hidden / heads`,
//!   or zero without heads; `widest_context` is the largest active row's
//!   rotary context, 1 when no row is active. `cache_slots`,
//!   `group_indices`, `truncation_leaves_bytes`, the checkpoint extents,
//!   `derived_head_dim`, and `widest_context` are the shared decoder
//!   planning (`src/models/shared/decoder_plan.rs`) that the family plan
//!   re-exports; it is included at its crate path and re-verified here, with
//!   `verus/models.rs` as its home harness;
//! - `interleaved_offset` locates gate output `j` at column `2j` and up output
//!   `j` at column `2j + 1` of a row of the stacked `gate_up_proj`, provably
//!   inside the `[experts, rows, 2 · width]` tensor; the lemmas
//!   `lemma_interleaved_injective` and `lemma_interleaved_surjective` prove
//!   that this index is a bijection between `(expert, row, output, half)`
//!   coordinates and the tensor's elements, so the gate and up halves of all
//!   experts partition the stacked weights;
//! - `stacked_layout` computes one expert's and the whole tensor's byte
//!   counts exactly when the element count and both byte counts fit
//!   `usize`, and reports overflow otherwise; `expert_slab` copies exactly
//!   bytes `[e · span, (e + 1) · span)`; and `deinterleave` copies, for every
//!   output byte, exactly the matching byte of column `2j` (gate) or
//!   `2j + 1` (up) of the stacked row, with every read in bounds, so the
//!   loader's split equals the reference's `gate_up[e, :, ::2]` and
//!   `gate_up[e, :, 1::2]` byte for byte.
//!
//! Not proved here, and ordinary Rust: the byte copies of the table-lookup
//! kernel (`decode_unit` and `Mxfp4Table`), the scoped threads of
//! `decode_transposed` that run the verified partition, and the dtype
//! encoders that round each exact binary64 value. They are covered by
//! `src/checkpoint/mxfp4_tests.rs`, which checks every `(scale, code)` pair
//! and a seeded multi-expert tensor bit for bit against Transformers (and,
//! in an opt-in ignored test, the pinned checkpoint's expert tensors). The
//! loader's payload check that establishes `deinterleave`'s and
//! `expert_slab`'s length precondition, the mapping of
//! `AttentionConnectivity` to sliding flags, and the cache specifications
//! built from the plans are covered by the gpt-oss load and decode tests.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/division.rs"]
pub mod division;

#[path = "../src/checkpoint/mxfp4.rs"]
pub mod mxfp4;

#[path = "../src/models/shared/decoder_plan.rs"]
pub mod shared_decoder_plan;

/// Mirrors the crate paths `crate::models::shared::*` used by the family plan.
pub mod models {
    /// Model-independent family glue.
    pub mod shared {
        pub use crate::shared_decoder_plan as decoder_plan;
    }
}

#[path = "../src/models/gpt_oss/plan.rs"]
pub mod plan;
