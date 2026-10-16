//! Verus harness for gpt-oss: exact MXFP4 decoding arithmetic and the family's planning.
//!
//! Verifies, in place, the specifications attached to `src/checkpoint/mxfp4.rs`:
//!
//! - `mxfp4_geometry` returns `Ok` if and only if `accepts` holds: `[E, R, G,
//!   16]` blocks with `[E, R, G]` scales and nonzero axes whose decoded value
//!   count `E · R · 32 G` fits both `usize` and `u64` (soundness and
//!   completeness, so it neither admits a bad shape nor refuses a good one);
//!   the returned extents equal the shape's axes and `columns == 32 G`;
//! - `values_of` computes that count without overflow;
//! - `fp4_codes` splits a byte into its low nibble (`byte mod 16`, the even
//!   decoded column) and high nibble (`byte / 16`, the odd column);
//! - `exact_value_bits` returns the binary64 encoding of the exact decoded
//!   value `fp4(code) · 2^(scale - 127)`: the code's sign bit, the biased
//!   exponent `scale + 895` (`m = 1`) or `⌊m/2⌋ + scale + 895` (`m >= 2`), and
//!   the fraction `2^51` for odd magnitude codes; scale code 255 yields the
//!   canonical NaN or `±inf` as Transformers' infinite factor does;
//! - `scale_offset`, `block_offset`, and `transposed_offset` compute the
//!   row-major offsets of a block's scale, its 16 bytes, and a decoded value
//!   in the transposed `[E, 32 G, R]` output, each provably in bounds.
//!
//! It also verifies `src/models/gpt_oss/plan.rs`:
//!
//! - `default_layer_slides` equals the reference's default schedule
//!   `bool((l + 1) % 2)` without overflow, and `default_schedule` applies it
//!   to every layer: consecutive layers alternate from a sliding layer and
//!   exactly `⌊L / 2⌋` layers are full-attention layers;
//! - `ring_assignment` activates the ring exactly for a shared window
//!   `0 < W < capacity` and then puts exactly the sliding layers in it;
//!   `cache_slots` numbers each KV group's layers by an order-preserving
//!   bijection onto `0..size`, and the two sizes partition the layers;
//!   `group_indices` numbers the allocated groups densely, full storage
//!   first; `truncation_leaves_bytes` is exactly "some row is shortened to a
//!   nonzero length", the condition that clears clean tails;
//! - `fused_expert_elements` returns `2f · d · E` exactly when every partial
//!   product fits `usize` (the configuration's overflow check);
//!   `checkpoint_extent` and `projection_extent` form the contract's `u64`
//!   extents exactly or name the `usize` or `u64` overflow; and
//!   `derived_head_dim` is `hidden / heads`, or zero without heads;
//!   `widest_context` is the largest active row's rotary context, 1 when no
//!   row is active;
//! - `interleaved_offset` locates gate output `j` at column `2j` and up output
//!   `j` at column `2j + 1` of a row of the stacked `gate_up_proj`, provably
//!   inside the `[experts, rows, 2 · width]` tensor; the lemmas
//!   `lemma_interleaved_injective` and `lemma_interleaved_surjective` prove
//!   that this index is a bijection between `(expert, row, output, half)`
//!   coordinates and the tensor's elements, so the gate and up halves of all
//!   experts partition the stacked weights;
//! - `stacked_layout` computes one expert's and the whole tensor's byte
//!   counts exactly or reports overflow; `expert_slab` copies exactly bytes
//!   `[e · span, (e + 1) · span)`; and `deinterleave` copies, for every
//!   output byte, exactly the matching byte of column `2j` (gate) or
//!   `2j + 1` (up) of the stacked row, with every read in bounds, so the
//!   loader's split equals the reference's `gate_up[e, :, ::2]` and
//!   `gate_up[e, :, 1::2]` byte for byte.
//!
//! The table-lookup kernel that applies these offsets, its thread partition,
//! and the dtype encoders remain covered by `src/checkpoint/mxfp4_tests.rs`,
//! which checks every `(scale, code)` pair and the pinned checkpoint's
//! expert tensors bit for bit against Transformers. The loader's payload
//! check that establishes `deinterleave`'s and `expert_slab`'s length
//! precondition, the mapping of `AttentionConnectivity` to sliding flags,
//! and the cache specifications built from the plans are covered by the
//! gpt-oss load and decode tests.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/checkpoint/mxfp4.rs"]
pub mod mxfp4;

#[path = "../src/models/gpt_oss/plan.rs"]
pub mod plan;
