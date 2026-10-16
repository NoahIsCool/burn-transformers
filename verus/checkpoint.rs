//! Verus harness for the checkpoint loader's pure layout and packing arithmetic.
//!
//! Verifies, in place, the specifications attached to `src/checkpoint/layout.rs`:
//!
//! - `element_count` returns the exact mathematical shape product when every
//!   left-to-right prefix product fits in `u64`, and `None` exactly when some
//!   prefix overflows, matching the pinned SafeTensors validator;
//! - `storage_bytes` returns `n` with `8 * n == elements * bits`, and rejects
//!   overflow and trailing partial bytes precisely;
//! - `validate_contiguous_tiling` returns `Ok` exactly when the sorted
//!   SafeTensors byte intervals tile the data section (soundness and
//!   completeness): they chain from zero, each beginning where the previous
//!   ended, and the last ends at the data-section size, so every interval lies
//!   inside the section, nonempty intervals are disjoint, and their lengths sum
//!   exactly to the data-section size. Each error names the first violation,
//!   with every earlier interval proved valid;
//! - `tensor_file_offset` computes `8 + header + begin` or reports overflow;
//! - `slice_block` keeps every slice and concatenation block inside its tensor
//!   with overflow-free arithmetic, and consecutive target blocks partition the
//!   output in order;
//! - `section_starts` returns prefix-sum section starts only when the split
//!   lengths sum exactly to the axis, so every section lies inside the axis, and
//!   reports overflow or the exact mismatching total otherwise;
//! - `checked_range_product` is `element_count` of a subslice; `axis_view`
//!   splits a shape into exact `[outer, axis, inner]` products whose
//!   `outer * axis` and `outer * axis * inner` fit `usize` (exactly
//!   `slice_block`'s preconditions) and whose product is the full shape
//!   product, naming the first failed check otherwise; `axis_elements` and
//!   `range_within` decide their products and ranges exactly;
//! - `extend_concatenation` adds a source's axis length to the running target
//!   shape exactly when the source agrees on every other axis and the sum fits,
//!   and leaves the shape unchanged on either error;
//! - `checked_host_product` and `dense_byte_len` compute host element and byte
//!   counts or report the overflowing running product;
//! - `data_section` places the data section at `8 + header` and proves it ends
//!   exactly at the end of the file, rejecting an empty or oversized header.
//!
//! It also verifies `src/checkpoint/quantization/packing.rs`: Q8/Q4 packed
//! lengths (`n` and `ceil(n / 2)`, never above `n`), the two's-complement nibble
//! and byte encodings and their inverses, nibble joining and splitting, and code
//! reads that return exactly the stored code (Q8 byte `i`; Q4 low nibble of byte
//! `i / 2` for even `i`, high nibble for odd `i`) and `None` exactly past the end.
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
