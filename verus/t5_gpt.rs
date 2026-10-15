//! Verus harness for the T5 and GPT model families' pure planning logic.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/t5/model/packing.rs`, the host plans behind batched T5
//! stateful decoding:
//!
//! - `plan_rows` (behind `BatchedRows::new`): the block width is the largest
//!   row count; the prefix layout is chosen exactly when every row's valid
//!   columns are `0..n_b`; otherwise slot `(b, i)` gathers rectangular column
//!   `c_{b,i}` of row `b` (or column 0 when padded), column `c_{b,k}` scatters
//!   back from slot `(b, k)`, and every other column reads slot `(b, 0)`; a
//!   plan exists only when every row is strictly increasing below `S`, and a
//!   refusal means a malformed row or an unrepresentable `B * S`.
//! - `compaction_sources` and `scatter_sources`: every returned entry equals
//!   its formula over the flat column list, and each refusal is
//!   characterized (a count above the block, too few columns, an unsorted or
//!   out-of-range row, or an unrepresentable extent).
//! - `padded_slots` and `has_padded_slot`: exactly the slots `i >= n_b` are
//!   padding.
//! - `row_positions`: every block slot's causal visibility and relative-bias
//!   query position equal their documented formulas (`L + i` for new tokens,
//!   the row's last new position for padding, `L - 1` for inactive rows), so
//!   no slot of an active row sees a key past `L + n - 1`, and no position
//!   overflows `i64`.
//! - `groups_preserved`: exactly characterizes reorders that keep every row
//!   inside its group of `group_size` consecutive rows, which is what lets
//!   beam search leave T5's encoder binding and cross-attention keys in place.
//!
//! See `verus/README.md` for the method, toolchain, and command.
//!
//! The planning file is dependency-free, so it is included as a single module.
//! The Burn tensor programs driven by these plans (token packing, state
//! unpacking, write-ahead cache appends, masks, and the relative bias) remain
//! covered by the Rust tests in `src/models/t5/model/*_tests.rs`, including
//! parity against the per-row reference algorithm and the stateless forward.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/t5/model/packing.rs"]
pub mod packing;
