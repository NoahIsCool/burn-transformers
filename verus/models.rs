//! Verus harness for model-family glue shared by every architecture.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/shared/plan.rs`:
//!
//! - `valid_columns`: a mask row compacts to exactly the recursive list of
//!   its valid columns (`spec_valid_columns`), which is strictly increasing,
//!   in range, and contains every valid column;
//! - `plan_scatter` and its helpers `is_sorted_unique_below` (its exact
//!   precondition check), `is_single_run`, and `gather_sources`: scatter
//!   plans restore the rectangular layout (identity, one contiguous run, or a
//!   gather map with an appended zero row), and the identity or a single run
//!   is chosen whenever it applies;
//! - `row_valid_columns`, `plan_row_grid`, and `row_last_columns`: every row
//!   of an optional `[batch, sequence]` mask lists exactly its valid columns
//!   (with its count), and reports its last valid column or `sequence - 1`
//!   when empty (dense when no mask);
//! - `row_contexts` and `plan_host_rows` (with `list_lengths` and
//!   `contains_zero`): the host side of every stateful prefill and decode
//!   call returns exactly the specified outcome, with errors in precedence
//!   order (rejected empty row, first impossible append, first row with an
//!   out-of-range valid token, out-of-range valid position, or
//!   unrepresentable extent, where an explicit-position row succeeds only
//!   when `1 + max p` is a positive `usize`, with or without a position
//!   limit) and otherwise the per-row columns, counts, last columns, and
//!   rotary extents `1 + max p` or `L + n`;
//! - `split_readback`: the fused readback splits into its mask, token, and
//!   position segments exactly;
//! - `plan_last_selection`, `positions_within`, and `last_valid_column`:
//!   logits selections are classified and validated exactly;
//! - `count_fills_width`, `signed_indices`, `implicit_positions`,
//!   `context_from_maximum`, `range_within`, `value_range`, and
//!   `plan_appends`: padding detection, signed index conversion, implicit
//!   cached positions `baseline + k`, the rotary extent `max + 1`, attained
//!   value ranges over valid columns, and per-row cache appends against a
//!   capacity are exact and overflow-free;
//! - `active_row_contexts`: the active rows of a stateful call are listed
//!   exactly once each, in order, with their rotary extents.
//!
//! It also verifies `src/models/shared/load_audit.rs`: `payload_totals`
//! returns the exact sum (or reports `u64` overflow) and the attained maximum
//! of quantized payload byte counts, and `equivalence_reads` is exactly `2 n`
//! or reports overflow, as `LoadedModel` audits require.
//!
//! Finally, it verifies `src/models/shared/block_plan.rs`, the host planning
//! of batched stateful blocks shared by DeepSeek-V2 and Muse Glimmer:
//!
//! - `row_maps` gathers exactly the valid columns into the first `n` block
//!   entries (padding entries gather `c_0`, or `0` for an empty row) and
//!   scatters each valid column back to its entry, every other column to the
//!   zero row `T`;
//! - `batch_extents` returns exactly the widest row `T = max_b n_b` and the
//!   furthest cache end `E = max_b (L_b + n_b)` (both attained, zero for no
//!   rows), failing only on a length mismatch or an overflowing end;
//! - `key_limits` returns nondecreasing limits in `1..=E` that are exactly
//!   `max(1, max_{j <= i, n_b > j} (L_b + j + 1))`: they cover every active
//!   row's causal key prefix and are attained by one, so bounded attention
//!   reads no key a query prefix cannot see;
//! - `query_slots` places entry `k < n_b` of row `b` at cache slot `L_b + k`
//!   and padding entries at `L_b`, within the key extent;
//! - `identity_layout`, `query_validity`, and `implicit_positions` decide the
//!   identity packing, the validity `k < n_b`, and the implicit positions
//!   `L_b + k` (zero at padding) exactly; `clamp_limits` keeps limits
//!   nondecreasing and in `1..=K`;
//! - `rotary_contexts` returns the first active row's context (at least one),
//!   whether every active context equals it, and each row's own context;
//!   `first_uncacheable_context` names the first active row with cached keys
//!   whose context passes a rotary configuration's cached-context limit.
//!
//! Index conversion for the uploaded block maps reuses the verified
//! `plan::signed_indices` above.
//!
//! See `verus/README.md` for the method, toolchain, and command.
//!
//! The planning files are dependency-free, so each is included as a single
//! module. Burn tensor programs driven by these plans (selection,
//! concatenation, gathers and scatters, cache writes, attention, and the fused
//! validation readback itself), the readback slot
//! bookkeeping of `DeferredChecks`, and the shape preconditions established by
//! the unverified callers (`[batch, sequence]` host buffers and one cursor per
//! row, checked there at runtime) remain covered by the Rust tests in
//! `src/models/shared/*_tests.rs` and every family's parity suite.
//! `verus/nemotron.rs` includes the same planning file at its crate path and
//! re-verifies it alongside the Nemotron-H plans.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/shared/plan.rs"]
pub mod plan;

#[path = "../src/models/shared/load_audit.rs"]
pub mod load_audit;

#[path = "../src/models/shared/block_plan.rs"]
pub mod block_plan;
