//! Verus harness for the checkpoint loader's pure layout and packing arithmetic.
//!
//! Verifies, in place, the specifications attached to `src/checkpoint/layout.rs`:
//!
//! - `element_count` returns the exact mathematical shape product, or `None`
//!   exactly when that product exceeds `u64::MAX` (a zero dimension yields zero);
//! - `storage_bytes` returns `n` with `8 * n == elements * bits`, and rejects
//!   overflow and trailing partial bytes precisely;
//! - `validate_contiguous_tiling` accepts sorted SafeTensors byte intervals only
//!   when they chain from zero, each beginning where the previous ended, and the
//!   last ends at the data-section size, so every interval lies inside the
//!   section, nonempty intervals are disjoint, and their lengths sum exactly to
//!   the data-section size;
//! - `tensor_file_offset` computes `8 + header + begin` or reports overflow;
//! - `slice_block` keeps every slice and concatenation block inside its tensor
//!   with overflow-free arithmetic, and consecutive target blocks partition the
//!   output in order;
//! - `section_starts` returns prefix-sum section starts only when the split
//!   lengths sum exactly to the axis, so every section lies inside the axis, and
//!   reports overflow or the exact mismatching total otherwise.
//!
//! It also verifies `src/checkpoint/quantization/packing.rs`: Q8/Q4 packed
//! lengths (`n` and `ceil(n / 2)`, never above `n`), the two's-complement nibble
//! and byte encodings and their inverses, nibble joining and splitting, and
//! in-bounds code reads.
//!
//! See `verus/README.md` for the method, toolchain, and command. Tensor copies
//! driven by these plans, file I/O, error mapping, and the floating-point
//! rounding encoder in `src/checkpoint/runtime/rounding.rs` (checked
//! exhaustively at every 16-bit midpoint and against independent oracles) remain
//! covered by tests rather than proofs.

// Loop invariants are attributes on `while` expressions, which rustc accepts
// only with this feature; it applies solely under Verus.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/checkpoint/layout.rs"]
pub mod layout;

#[path = "../src/checkpoint/quantization/packing.rs"]
pub mod packing;
