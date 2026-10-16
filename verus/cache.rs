//! Verus harness for the cache's pure planning and lifecycle code.
//!
//! Verifies, in place, the specifications attached to:
//!
//! * `src/cache/ring.rs`: standard and mirrored ring write and read planning
//!   (`plan_ring_write_slots`, `mirrored_write_slots`, `ring_read_slots`,
//!   `mirrored_read_slots`, and their splitters `write_segments` and
//!   `read_segments`), staging plans (`staged_transaction_slots`),
//!   sliding-window starts (`sliding_window_start`, exactly
//!   `max(0, length - window)`), and staged-commit planning
//!   (`commit_prefix_slots`), including the slot invariant
//!   `slot(p) = p mod capacity` for every token written, mirrored, or read
//!   (`lemma_slot_after_offset` handles the single wrap). Writes keep exactly
//!   the newest `min(n, capacity)` tokens; mirrored writes and staged commits
//!   cover exactly the retained suffix; reads partition the retained window in
//!   chronological order; and every planner fails exactly when its overflow or
//!   bound condition holds.
//! * `src/cache/lengths.rs`: the length ledger behind every decode state and
//!   speculative transaction. Each transition (`advanced`, `begun`,
//!   `committed_prefix`, `rolled_back`, `truncated`, `reordered`, `reset`)
//!   preserves `committed <= working <= capacity`, computes its documented
//!   formula without overflow, and is rejected exactly when its rule is
//!   violated. Every rejection, from a free validator (`advanced_lengths`,
//!   `validate_rewind`, `committed_prefix_lengths`, `reordered_lengths`) or a
//!   ledger transition, names the first offending row with its exact operands
//!   or the blocking transaction state. Constructors (`empty_ledger`,
//!   `ledger_from_lengths`, `zero_lengths`, `copy_lengths`), proposal lengths
//!   (`proposal_lengths`, `proposal`, `proposals`), remaining capacities
//!   (`remaining`), per-row transitions (`length_transitions`), and appended
//!   ranges (`appended_range`, `[min(previous, current), current)`) are exact.
//! * `src/cache/retention.rs`: ring and recurrent rewind predicates
//!   (`ring_rewind_is_safe`, `recurrent_rewind_is_exact`), proved sound and
//!   complete against the ring eviction model, and their row scans
//!   (`first_unsafe_ring_rewind`, `first_inexact_recurrent_rewind`), which name
//!   the lowest offending row; the working window that splits a speculative
//!   read into committed and staged parts (`ring_working_window`); and the
//!   absolute position held by each ring slot in a batched slot-order read
//!   (`ring_slot_position`, the newest `p < length` with `p mod W == slot`).
//! * `src/cache/rows.rs`: `row_runs` (with `run_end`) groups batch rows into
//!   runs that share one tensor assignment; runs are ordered, disjoint, carry
//!   their rows' parameters, and cover exactly the rows with new tokens.
//! * `src/cache/tensor/plan.rs`: integer planning shared by the tensor
//!   caches. Allocation scalar counts (`tensor_elements`, `repeated_elements`)
//!   are exact running products checked against overflow, the portable `i32`
//!   element limit, and byte size; `checked_product` is the exact product or
//!   `None`. Append ends (`write_end`, and `checked_ends` per row) are exact
//!   and within capacity, and each write-end violation is exact. The
//!   staged-commit scan `check_staged_commit` accepts exactly fully staged,
//!   in-proposal prefixes and names the first failure.
//!   `rows_cleared_by_truncate` marks exactly the rows reset from history;
//!   beam sources, read lengths, and host index vectors are checked and
//!   converted exactly (`first_index_out_of_range`, `first_index_above`,
//!   `portable_indices`); `staging_end` keeps staging lengths within the
//!   staging capacity; and `next_version` always differs from its predecessor.
//! * `src/cache/tensor/recurrent/plan.rs`: recurrent snapshot indexing. The
//!   working and committed snapshot of `n` tokens is `n - 1`
//!   (`last_snapshot`); a uniform batch shares one snapshot exactly when every
//!   row staged the same positive count (`uniform_snapshot`); each flat gather
//!   index `b * P + max(s_b, 1) - 1` of `flat_snapshot_indices` decodes to row
//!   `b` and its own snapshot (`lemma_flat_decodes`), failing exactly beyond
//!   `i32::MAX`; and `first_nonuniform_offset` names the lowest row whose
//!   staging offset differs from row zero's.
//! * `src/cache/tensor/kv/plan.rs`: native KV planning. The effective ring
//!   width (`effective_window_of`) is the group's window, else the bank's; the
//!   committed time extent (`committed_time_slots`) is exactly the capacity,
//!   `W`, or `2W` for that same width, and every inconsistent storage/window
//!   configuration is rejected with its first rule. Committed reads
//!   (`visible_count`, `committed_read_rows`) keep `min(L, W)` positions;
//!   working reads (`working_read_rows`, `working_window`) check every end and
//!   split each window exactly as the verified working window does; mirrored
//!   runs (`mirrored_window_slots`) read the slot of each matching position
//!   below `2W`; committed reads (`committed_read_segments`) partition the
//!   newest `k` positions in order and read each from a slot that holds it; a
//!   working read's staged ranges (`staged_read_ranges`) align offset `j` with
//!   position `C + j`. Committed writes (direct appends
//!   `direct_write_segments`, `ring_direct_segments`, and
//!   `mirrored_direct_segments`; batched runs `plan_direct_run`; and
//!   staged-prefix commits `commit_write_segments`) write input column `c + j`
//!   to slot `s + j` under full storage and to slot `(s + j) mod W` (plus `W`
//!   for the mirror) in a ring, keep exactly the newest `min(n, W)` tokens,
//!   shift run source rows exactly, and report the first failed check.
//!   `write_ahead_allowed` admits write-ahead exactly for full storage, in
//!   place inside a transaction.
//! * `src/cache/tensor/kv/batch/layout.rs`: batched write layouts. Append
//!   layouts (`validate_append_layout`) are accepted exactly when every row
//!   fits its block and the position space; left padding
//!   (`left_padded_columns`) and ends (`append_ends`) are exact. A staging
//!   write (`plan_stage_updates`) is accepted exactly when every active row
//!   passes its cursor, staging, and capacity checks and then updates exactly
//!   the active rows; a rejection names the first failing check of the first
//!   failing row with exact operands. Proposal cursors (`proposal_cursor_of`,
//!   `proposal_starts`) are the exact saturated sums. Linear staging runs
//!   (`linear_stage_runs`) partition the active rows and address each row's
//!   own columns and proposal offsets; single-row stages use exactly one-hot
//!   counts (`one_hot_counts`) starting at each row's proposal cursor; and a
//!   step's token table (`token_segment_table`) marks exactly each row's
//!   columns with their positions.
//! * `src/cache/tensor/kv/batch/slots.rs`: batched read slot tables. Full
//!   storage (`full_slot_mask`) marks exactly each row's visible window; each
//!   written ring slot (`ring_slot_table`) reports the newest position with
//!   its residue and is valid exactly when that position is visible
//!   (unwritten slots are invalid); staged slots report `C_b + s`
//!   (`staged_slot_positions`) and are valid exactly between the skipped and
//!   staged proposals (`staged_slot_mask`); the all-valid shortcuts
//!   (`full_slots_all_valid`, `staged_slots_all_valid`), the per-layer staging
//!   comparison (`first_unequal_vector`), and `max_value` (an attained
//!   maximum, zero when empty) are exact, and `table_with_capacity` returns
//!   an empty table without an overflowing capacity product.
//! * `src/cache/tensor/state/step/coverage.rs`: step write-set coverage and
//!   lifecycle versions. `mark_covered` accepts a row mark exactly for an
//!   in-batch, active, unmarked row and sets only that flag, and a batched
//!   mark exactly when every active row is unmarked, then setting exactly the
//!   active rows; `first_uncovered_row` names the first active unmarked row.
//!   Hence an accepted, finished step writes every active row exactly once.
//!   `bump` changes a bank's version to a different value and leaves the
//!   other bank's untouched (`initial_versions` starts both at zero and
//!   `current` reads one); the row scans (`first_active_row`,
//!   `first_unstaged_proposal`, `all_set`) and set-position lists
//!   (`set_positions`) are exact.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/cache/lengths.rs"]
pub mod lengths;
#[path = "../src/cache/retention.rs"]
pub mod retention;
#[path = "../src/cache/ring.rs"]
pub mod ring;
#[path = "../src/cache/rows.rs"]
pub mod rows;
#[path = "../src/cache/tensor/plan.rs"]
pub mod tensor_plan;
#[path = "../src/cache/tensor/recurrent/plan.rs"]
pub mod recurrent_plan;
#[path = "../src/cache/tensor/kv/plan.rs"]
pub mod kv_plan;
#[path = "../src/cache/tensor/kv/batch/layout.rs"]
pub mod kv_batch_layout;
#[path = "../src/cache/tensor/kv/batch/slots.rs"]
pub mod kv_batch_slots;
#[path = "../src/cache/tensor/state/step/coverage.rs"]
pub mod step_coverage;

/// Mirrors the crate paths `crate::cache::*` used by the tensor-cache planners.
pub mod cache {
    pub use crate::{lengths, retention, ring, rows};

    /// Mirrors `crate::cache::tensor`.
    pub mod tensor {
        pub use crate::tensor_plan as plan;
    }
}
