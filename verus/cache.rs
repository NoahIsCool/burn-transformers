//! Verus harness for the cache's pure planning and lifecycle code.
//!
//! Verifies, in place, the specifications attached to:
//!
//! * `src/cache/ring.rs`: standard and mirrored ring write and read planning,
//!   staging plans, sliding-window starts, and staged-commit planning,
//!   including the slot invariant `slot(p) = p mod capacity` for every token
//!   written, mirrored, or read. Mirrored writes and staged commits cover
//!   exactly the retained suffix, and every planner fails exactly when its
//!   overflow or bound condition holds.
//! * `src/cache/lengths.rs`: the length ledger behind every decode state and
//!   speculative transaction. Each transition (advance, begin, commit,
//!   rollback, truncate, reorder, reset) preserves
//!   `committed <= working <= capacity`, computes its documented formula
//!   without overflow, and is rejected exactly when its rule is violated.
//!   Every rejection, from a free validator or a ledger transition, names the
//!   first offending row with its exact operands or the blocking transaction
//!   state. Proposal lengths, remaining capacities, per-row transitions, and
//!   appended ranges are exact.
//! * `src/cache/retention.rs`: ring and recurrent rewind predicates, proved
//!   sound and complete against the ring eviction model (row scans name the
//!   lowest offending row); the working window that splits a speculative read
//!   into committed and staged parts; and the absolute position held by each
//!   ring slot in a batched slot-order read.
//! * `src/cache/rows.rs`: grouping of batch rows into runs that share one
//!   tensor assignment; runs are ordered, disjoint, carry their rows'
//!   parameters, and cover exactly the rows with new tokens.
//! * `src/cache/tensor/plan.rs`: integer planning shared by the tensor
//!   caches. Allocation scalar counts are exact running products checked
//!   against overflow, the portable `i32` element limit, and byte size;
//!   plain checked products are exact. Append ends (and checked row ends) are
//!   exact and within capacity, and each write-end violation is exact. The
//!   staged-commit scan accepts exactly fully staged, in-proposal prefixes
//!   and names the first failure. Truncation clears exactly the rows reset
//!   from history; beam sources, read lengths, and host index vectors are
//!   checked and converted exactly; staging lengths stay within the staging
//!   capacity; version successors always differ from their predecessor.
//! * `src/cache/tensor/recurrent/plan.rs`: recurrent snapshot indexing. The
//!   working and committed snapshot of `n` tokens is `n - 1`, a uniform batch
//!   shares one snapshot exactly when every row staged the same positive
//!   count, and each flat gather index `b * P + max(s_b, 1) - 1` decodes to
//!   row `b` and its own snapshot, failing exactly beyond `i32::MAX`.
//! * `src/cache/tensor/kv/plan.rs`: native KV planning. The effective ring
//!   width is the group's window, else the bank's; the committed time extent
//!   is exactly the capacity, `W`, or `2W` for that same width, and every
//!   inconsistent storage/window configuration is rejected with its first
//!   rule. Committed reads keep `min(L, W)` positions; working reads check
//!   every end and split each window exactly as the verified working window
//!   does; mirrored runs read the slot of each matching position below `2W`;
//!   a working read's staged ranges align offset `j` with position `C + j`.
//!   Committed writes (direct appends, batched runs, and staged-prefix
//!   commits) write input column `c + j` to slot `s + j` under full storage
//!   and to slot `(s + j) mod W` (plus `W` for the mirror) in a ring, keep
//!   exactly the newest `min(n, W)` tokens, shift run source rows exactly,
//!   and report the first failed check. Write-ahead is admitted exactly for
//!   full storage, in place inside a transaction.
//! * `src/cache/tensor/kv/batch/layout.rs`: batched write layouts. Append
//!   layouts are accepted exactly when every row fits its block and the
//!   position space; left padding and ends are exact. A staging write is
//!   accepted exactly when every active row passes its cursor, staging, and
//!   capacity checks and then updates exactly the active rows; a rejection
//!   names the first failing check of the first failing row with exact
//!   operands. Linear staging runs partition the active rows and address
//!   each row's own columns and proposal offsets; single-row stages use
//!   exactly one-hot counts starting at each row's proposal cursor; a step's
//!   token table marks exactly each row's columns with their positions.
//! * `src/cache/tensor/kv/batch/slots.rs`: batched read slot tables. Full
//!   storage marks exactly each row's visible window; each written ring slot
//!   reports the newest position with its residue and is valid exactly when
//!   that position is visible (unwritten slots are invalid); staged slots
//!   report `C_b + s` and are valid exactly between the skipped and staged
//!   proposals; all-valid shortcuts and the per-layer staging comparison are
//!   exact.
//! * `src/cache/tensor/state/step/coverage.rs`: step write-set coverage and
//!   lifecycle versions. A row mark succeeds exactly for an in-batch, active,
//!   unmarked row and sets only that flag; a batched mark succeeds exactly
//!   when every active row is unmarked and sets exactly the active rows; the
//!   finish check names the first active unmarked row. Hence an accepted,
//!   finished step writes every active row exactly once. Bumping a bank's
//!   version changes it to a different value and leaves the other bank's
//!   untouched; row scans and set-position lists are exact.
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
