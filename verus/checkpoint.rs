//! Verus harness for the checkpoint loader's pure layout, packing, and count arithmetic.
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
//! It verifies `src/checkpoint/lexical.rs`, the byte rules applied to untrusted
//! shard-index paths, tensor names, and byte totals:
//!
//! - `lexical_path_violation` names exactly the first broken rule (empty, NUL,
//!   backslash, Windows drive prefix) and returns `None` exactly when none is
//!   broken;
//! - `normal_components` holds exactly when the path is not rooted, does not
//!   begin with a `.` component, and has no `..` component anywhere (the Unix
//!   `Path::components` rule, cross-checked exhaustively by tests), so an
//!   accepted path cannot climb out of the directory it is joined to;
//! - `decimal_layer_between` holds exactly when a name is `prefix ++ layer ++
//!   suffix` with a canonical decimal layer (digits, no leading zero except
//!   `0`), with non-overlapping anchors;
//! - `integral_lexeme_digits` accepts exactly the lexemes whose part before the
//!   first `.` is nonempty digits and whose fraction, if any, is nonempty zeros,
//!   returning the integer digits' length;
//! - `canonical_components` accepts exactly the paths whose every component is
//!   nonempty and neither `.` nor `..`, so a provenance shard has one key, and
//!   `is_lowercase_sha256` accepts exactly 64 lowercase hexadecimal bytes.
//!
//! It verifies `src/checkpoint/runtime/permutation.rs`: `probe_dimension`
//! lengths strictly increase with the axis (so probe axes are pairwise
//! distinct), and for distinct input lengths `permutation_of` succeeds exactly
//! when the mapped shape is a rearrangement of the probe, recovering a
//! bijection `axes` with `output[k] == input[axes[k]]`.
//!
//! It also verifies `src/checkpoint/quantization/packing.rs`: Q8/Q4 packed
//! lengths (`n` and `ceil(n / 2)`, never above `n`), the two's-complement nibble
//! and byte encodings and their inverses, nibble joining and splitting, and code
//! reads that return exactly the stored code (Q8 byte `i`; Q4 low nibble of byte
//! `i / 2` for even `i`, high nibble for odd `i`) and `None` exactly past the end;
//! `push_code` preserves the packing invariant (a buffer holding codes `0..i`
//! becomes one holding `0..=i`, every earlier code unchanged, the new code
//! readable as pushed, and an unused Q4 high nibble kept zero), so packing then
//! reading is the identity; `first_code_violation` returns `None` exactly for
//! canonical buffers and otherwise names the first short read, reserved code, or
//! dirty Q4 padding.
//!
//! It verifies `src/checkpoint/quantization/staging.rs`, the Burnpack staging
//! session's admission arithmetic: `bounded_add` admits exactly the sums within
//! a limit (so a running total kept below its limit stays below it, with no
//! wraparound), and `admit_tensor` admits a tensor exactly when the new count,
//! its payload, and the new staged total are within their limits, naming the
//! first breached limit otherwise.
//!
//! And it verifies `src/checkpoint/quantization/counts.rs`: exact ceilings in
//! `u64` and `usize`; persisted group (`⌈n / G⌉`), packed-byte, and scale-byte
//! counts, with the scale length proved never to overflow for `G >= 32`; and the
//! Burn block count `rows · (columns / G)`, proved equal to the flat group count
//! `⌈rows · columns / G⌉` whenever `G` divides the width.
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

#[path = "../src/checkpoint/lexical.rs"]
pub mod lexical;

#[path = "../src/checkpoint/runtime/permutation.rs"]
pub mod permutation;

#[path = "../src/checkpoint/quantization/packing.rs"]
pub mod packing;

#[path = "../src/checkpoint/quantization/counts.rs"]
pub mod counts;

#[path = "../src/checkpoint/quantization/staging.rs"]
pub mod staging;
