//! Verus harness for the checkpoint loader's pure layout arithmetic.
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
//!   section, nonempty intervals are disjoint, and every byte is accounted for;
//! - `tensor_file_offset` computes `8 + header + begin` or reports overflow;
//! - `slice_block` keeps every slice and concatenation block inside its tensor
//!   with overflow-free arithmetic, and consecutive target blocks partition the
//!   output in order.
//!
//! See `verus/README.md` for the method, toolchain, and command. Tensor copies
//! driven by these plans, file I/O, and error mapping remain covered by tests.

// Loop invariants are attributes on `while` expressions, which rustc accepts
// only with this feature; it applies solely under Verus.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/checkpoint/layout.rs"]
pub mod layout;
