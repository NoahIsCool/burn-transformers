//! Verus harness for the T5 and GPT model families' pure planning logic.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/t5/model/packing.rs`, the host plans behind batched T5
//! stateful decoding:
//!
//! - `plan_rows` (which builds every `BatchedRows`): the block width is the
//!   largest row count; the prefix layout is chosen exactly when every row's
//!   valid columns are `0..n_b`; otherwise slot `(b, i)` gathers rectangular
//!   column `c_{b,i}` of row `b` (or column 0 when padded), column `c_{b,k}`
//!   scatters back from slot `(b, k)`, and every other column reads slot
//!   `(b, 0)`. A plan is refused exactly for a row that is not strictly
//!   increasing below `S`, or for an unrepresentable `B * S` when some row is
//!   not a prefix.
//! - `compaction_sources` and `scatter_sources`: every returned entry equals
//!   its formula over the flat column list, and each refusal rule (a count
//!   above the block, too few columns, an unsorted or out-of-range row, or
//!   an unrepresentable extent) holds in both directions.
//! - `padded_slots` and `has_padded_slot`, and the `BatchedRows` methods over
//!   them: exactly the slots `i >= n_b` are padding.
//! - `row_positions`: every block slot's causal visibility and relative-bias
//!   query position equal their documented formulas (`L + i` for new tokens,
//!   the row's last new position for padding, `L - 1` for inactive rows), so
//!   no slot of an active row sees a key past `L + n - 1`, and no position
//!   overflows `i64`.
//! - `step_positions` (behind the decoder's `StepGeometry`): every slot of
//!   every row carries its `row_positions` entries; the shared committed
//!   length exists exactly when some row is active and every active row has
//!   that length; the largest bias position is attained; and a refusal
//!   names a row with `n_b > Q` or an unrepresentable position.
//!   `displacement_bound` bounds every key-to-query displacement.
//! - `groups_preserved`: exactly characterizes reorders that keep every row
//!   inside its group of `group_size` consecutive rows, which is what lets
//!   beam search leave T5's encoder binding and cross-attention keys in place;
//!   `first_out_of_range` returns the first reorder index outside the batch.
//!
//! It also verifies `src/models/t5/buckets.rs`: the configuration's
//! relative-bucket geometry check accepts exactly layouts with an even count
//! of at least four buckets whose exact-distance region (`B / 4` for the
//! encoder, `B / 2` for a causal decoder) lies below `max_distance`, naming
//! each rejection's cause, and the explicit-position bias table has radius
//! `min(bound, max_distance)`. A lemma states the index arithmetic of the
//! device's clamped lookup (every clamped index lies inside the table's
//! `2r + 1` entries); the device code itself is covered by Rust tests.
//! `bias_owners` gives every encoder layer the first layer with its head
//! count as relative-bias owner, so two layers share a learned table exactly
//! when their head counts agree, and `positions_within` accepts staged
//! positions exactly when they lie in the evaluated context.
//!
//! For the GPT families it verifies `src/models/gpt2/extents.rs` (the first
//! extent outside `1..=i32::MAX`, GPT-2's MLP width rule `n_inner` or `4 d`,
//! and the checked parameter-size product chains of both configurations)
//! and `src/models/gpt_neox/widths.rs` (an edited attention layer's
//! projection widths are exact and every invalid geometry is named).
//!
//! See `verus/README.md` for the method, toolchain, and command.
//!
//! The planning files are dependency-free, so each is included as a single module.
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

#[path = "../src/models/t5/buckets.rs"]
pub mod buckets;

#[path = "../src/models/gpt2/extents.rs"]
pub mod extents;

#[path = "../src/models/gpt_neox/widths.rs"]
pub mod widths;
