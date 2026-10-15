//! Verus harness for gpt-oss checkpoints: exact MXFP4 decoding arithmetic.
//!
//! Verifies, in place, the specifications attached to `src/checkpoint/mxfp4.rs`:
//!
//! - `mxfp4_geometry` accepts exactly `[E, R, G, 16]` blocks with `[E, R, G]`
//!   scales and nonzero axes, and only when the decoded value count
//!   `E · R · 32 G` fits both `usize` and `u64`; the returned extents equal
//!   the shape's axes and `columns == 32 G`;
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
//! The table-lookup kernel that applies these offsets, its thread partition,
//! and the dtype encoders remain covered by `src/checkpoint/mxfp4_tests.rs`,
//! which checks every `(scale, code)` pair and the pinned checkpoint's
//! expert tensors bit for bit against Transformers.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/checkpoint/mxfp4.rs"]
pub mod mxfp4;
