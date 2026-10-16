//! Verus harness for the integer planning of the recurrent mixers.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/delta_net/plan.rs`: Gated DeltaNet prefill planning. `chunk_plan`
//!   returns a chunk width `C = min(chunk_size, L)` and count
//!   `n = ceil(L / C)` that cover the tokens (`(n - 1) * C < L <= n * C`) with
//!   less than one chunk of padding and no overflow, using the exact ceiling
//!   `ceiling_division`; `block_chunk_ranges` returns balanced chunk blocks
//!   that are consecutive, start at zero, end at the chunk count, hold between
//!   one and the limit of chunks, and follow the exact balanced size formula;
//!   the doubling-inverse schedule (`inverse_block_sizes`, `inverse_levels`)
//!   starts at the least power of two `P` not below the chunk width and halves
//!   exactly down to two, and its levels hold `2^k` blocks that tile the
//!   `P x P` tile; `projection_widths` returns the key, value, and fused
//!   convolution widths exactly, and `None` exactly on overflow;
//!   `mixer_layout` returns query, key, and value channel slices that tile
//!   the convolution output; `head_repetitions` repeats key heads exactly
//!   onto the value heads (`repetitions * key_heads == value_heads`);
//!   `state_elements` is the exact recurrent state size;
//!   `check_delta_geometry` accepts exactly the admissible geometry (non-zero
//!   widths, key heads dividing value heads); and `prefers_recurrence` equals
//!   its closed-form cost comparison and is monotone
//!   (`lemma_recurrence_choice_monotone`).
//! - `src/mamba2/plan.rs`: Mamba-2 bookkeeping. `projection_width` and
//!   `convolution_width` are exact or name the first overflowing step in
//!   validation order, and `projection_slices` (`[z | xBC | dt]`) and
//!   `convolution_slices` (`[x | B | C]`) tile their layouts exactly;
//!   `group_width` (normalization groups) and `heads_per_group` (`B`/`C` head
//!   groups) split their axes exactly; `check_mamba2_geometry` and
//!   `row_counts_fit` accept exactly the admissible inputs, and `rows_full`
//!   decides full rows exactly; right-padding flags (`right_padded_flags`),
//!   row-end history columns (`row_end_columns`), per-token window columns
//!   (`window_columns`), and batch-row selectors (`signed_rows`) follow their
//!   closed forms or are rejected exactly when an index exceeds `i64`;
//!   `partition_modes` (with `same_mode` and `fallback_mode`) holds every row
//!   once, in ascending order, by effective mode (inactive rows take the first
//!   active row's mode), with a restore permutation that is a two-sided
//!   inverse of the grouped order; and the SSD scan blocks hold
//!   `clamp(budget / plane, 1, chunks)` chunks (`scan_block_chunks`) and tile
//!   the tokens in chunk-aligned blocks with only the last padded, by less
//!   than one chunk (`scan_blocks`).
//!
//! Not proved here: the Burn tensor algebra (chunk-major rearrangement, the
//! triangular inverse itself, causal convolutions, token compaction, the SSD
//! segment sums, and the recurrences) and floating-point numerics; the mixer
//! unit tests cover those. Loop invariants are attributes on `while`
//! statements, which needs `proc_macro_hygiene` under the Verus compiler only.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/delta_net/plan.rs"]
pub mod delta_net_plan;

#[path = "../src/mamba2/plan.rs"]
pub mod mamba2_plan;

/// Mirrors the crate path `crate::delta_net::plan` used by the Mamba-2 plan.
pub mod delta_net {
    pub use crate::delta_net_plan as plan;
}
