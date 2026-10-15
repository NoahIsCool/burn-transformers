//! Host planning for batched stateful execution of ragged, padded rows.
//!
//! A stateful call receives a rectangular `[B, S]` input whose row `b` holds
//! `n_b` valid tokens at columns `c_{b,0} < … < c_{b,n_b-1}` and appends them
//! to a cache whose row already holds `L_b` tokens. Batched execution packs
//! every row's valid tokens to the *left* of one `[B, T]` block with
//! `T = max_b n_b` (right padding), runs the model once on that block, and
//! restores the rectangular layout afterwards:
//!
//! ```text
//! block[b, k]  = input[b, c_{b,k}]            k < n_b   (gather)
//! output[b, c] = block_out[b, k]  if c = c_{b,k}         (scatter)
//!              = 0                otherwise   (zero row T)
//! ```
//!
//! Token `k < n_b` of row `b` lands at cache slot `L_b + k`. Attention keys
//! are read in slot order over `0..E` with `E = max_b (L_b + n_b)`; query `k`
//! of row `b` may see slots `0..=L_b + k`. The *key limits*
//! `λ_i = max_{i' <= i} max_{b : n_b > i'} (L_b + i' + 1)` bound the key prefix
//! that query rows `0..=i` of any row need, which lets bounded attention skip
//! keys a query chunk cannot see.
//!
//! When every row is fully valid (`n_b = S` for all `b`, the unmasked prefill
//! and the common one-token decode), both maps are the identity and no gather
//! or scatter kernel runs.
//!
//! This module is dependency-free and family-independent; its functions are
//! verified in place by Verus through `verus/deepseek_muse.rs` (no overflow,
//! map correctness, and the key-limit bound).

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// `s` is strictly increasing and every entry lies below `bound`.
pub open spec fn increasing_below(s: Seq<usize>, bound: int) -> bool {
    &&& forall|i: int, j: int| 0 <= i < j < s.len() ==> s[i] < s[j]
    &&& forall|i: int| 0 <= i < s.len() ==> s[i] < bound
}


/// Every row's cache end `L_b + n_b` fits within `extent` (and `usize`).
pub open spec fn ends_within(baselines: Seq<usize>, counts: Seq<usize>, extent: int) -> bool {
    &&& baselines.len() == counts.len()
    &&& forall|b: int| 0 <= b < counts.len() ==> baselines[b] + counts[b] <= extent
}

} // verus!

/// Gather and scatter maps of one row between rectangular and block layouts.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RowMaps {
    /// Source column of each of the `T` block entries (valid columns first).
    pub(crate) gather: Vec<usize>,
    /// Block entry of each of the `S` rectangular columns; `T` addresses a zero row.
    pub(crate) scatter: Vec<usize>,
}

/// Plans one row's gather and scatter maps.
///
/// `columns` are the row's valid columns `c_0 < … < c_{n-1} < sequence` and
/// `block >= n` is the packed width `T`. Block entry `k < n` gathers column
/// `c_k`; padding entries `k >= n` gather column `c_0`, or column `0` for an
/// empty row. That column is in range but its *value* is arbitrary: an empty
/// row's column `0` is masked and therefore never validated, so consumers
/// that index with gathered values (token IDs, positions) must overwrite
/// padding entries (`BlockLayout::compact_inert`), and every other consumer
/// masks them. Column `c_k` scatters entry `k`, and every other column
/// scatters the zero row `T`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(maps =>
    requires
        increasing_below(columns@, sequence as int),
        columns.len() <= block,
        sequence > 0,
    ensures
        maps.gather.len() == block,
        maps.scatter.len() == sequence,
        forall|k: int| 0 <= k < block ==> maps.gather@[k] < sequence,
        forall|k: int| 0 <= k < columns.len() ==> maps.gather@[k] == columns@[k],
        forall|k: int| 0 <= k < columns.len() ==> maps.scatter@[columns@[k] as int] == k,
        forall|c: int| 0 <= c < sequence ==> maps.scatter@[c] <= block,
        forall|c: int| 0 <= c < sequence && maps.scatter@[c] < block
            ==> maps.scatter@[c] < columns.len() && columns@[maps.scatter@[c] as int] == c,
))]
pub(crate) fn row_maps(columns: &[usize], sequence: usize, block: usize) -> RowMaps {
    let filler = if columns.is_empty() { 0 } else { columns[0] };
    let mut gather: Vec<usize> = Vec::with_capacity(block);
    let mut k: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            k <= block,
            gather.len() == k,
            filler < sequence,
            increasing_below(columns@, sequence as int),
            columns.len() <= block,
            forall|i: int| 0 <= i < k ==> gather@[i] < sequence,
            forall|i: int| 0 <= i < k && i < columns.len() ==> gather@[i] == columns@[i],
        decreases block - k,
    ))]
    while k < block {
        gather.push(if k < columns.len() {
            columns[k]
        } else {
            filler
        });
        k += 1;
    }
    let mut scatter: Vec<usize> = Vec::with_capacity(sequence);
    let mut column: usize = 0;
    let mut next: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            column <= sequence,
            next <= columns.len(),
            scatter.len() == column,
            increasing_below(columns@, sequence as int),
            columns.len() <= block,
            // Every valid column before `column` has been consumed, in order.
            forall|i: int| 0 <= i < next ==> columns@[i] < column,
            next < columns.len() ==> columns@[next as int] >= column,
            forall|i: int| 0 <= i < next ==> scatter@[columns@[i] as int] == i,
            forall|c: int| 0 <= c < column ==> scatter@[c] <= block,
            forall|c: int| 0 <= c < column && scatter@[c] < block
                ==> scatter@[c] < columns.len() && columns@[scatter@[c] as int] == c,
        decreases sequence - column,
    ))]
    while column < sequence {
        if next < columns.len() && columns[next] == column {
            scatter.push(next);
            next += 1;
        } else {
            #[cfg(verus_keep_ghost)]
            proof! {
                // `column` is no valid column: earlier ones are smaller and later ones larger.
                if next < columns.len() {
                    assert forall|i: int| next < i < columns.len() implies columns@[i] > column by {
                        assert(columns@[next as int] < columns@[i]);
                    }
                }
            }
            scatter.push(block);
        }
        column += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        // Every valid column lies below `sequence`, so all were consumed.
        if next < columns.len() {
            assert(columns@[next as int] < sequence);
        }
    }
    RowMaps { gather, scatter }
}

/// Returns `(T, E)`: the packed block width `max_b n_b` and key extent `max_b (L_b + n_b)`.
///
/// Returns `None` when the vectors differ in length or an end overflows `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Some((block, extent)) => {
                &&& baselines.len() == counts.len()
                &&& ends_within(baselines@, counts@, extent as int)
                &&& forall|b: int| 0 <= b < counts.len() ==> counts@[b] <= block
                &&& (counts.len() > 0 ==> exists|b: int| 0 <= b < counts.len() && counts@[b] == block)
                &&& (counts.len() == 0 ==> block == 0)
            },
            None => true,
        },
))]
pub(crate) fn batch_extents(baselines: &[usize], counts: &[usize]) -> Option<(usize, usize)> {
    if baselines.len() != counts.len() {
        return None;
    }
    let mut block: usize = 0;
    let mut extent: usize = 0;
    let mut row: usize = 0;
    #[cfg(verus_keep_ghost)]
    proof_decl! { let ghost mut witness: int = 0; }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            baselines.len() == counts.len(),
            forall|b: int| 0 <= b < row ==> baselines@[b] + counts@[b] <= extent,
            forall|b: int| 0 <= b < row ==> counts@[b] <= block,
            row > 0 ==> 0 <= witness < row && counts@[witness] == block,
            row == 0 ==> block == 0,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        let end = baselines[row].checked_add(counts[row])?;
        if end > extent {
            extent = end;
        }
        if row == 0 || counts[row] > block {
            block = counts[row];
            #[cfg(verus_keep_ghost)]
            proof! {
                witness = row as int;
            }
        }
        row += 1;
    }
    Some((block, extent))
}

/// Returns the key-prefix limits `λ_0 <= … <= λ_{T-1}` of the module documentation.
///
/// For every row with `n_b > i`, query `i` of that row needs keys
/// `0..=L_b + i`, so `λ_i >= L_b + i + 1`; limits are prefix maxima, never
/// exceed `extent`, and are at least one so every chunk reads a key.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(limits =>
    requires
        ends_within(baselines@, counts@, extent as int),
        extent >= 1,
    ensures
        limits.len() == block,
        forall|i: int| 0 <= i < block ==> 1 <= #[trigger] limits@[i] && limits@[i] <= extent,
        forall|i: int, j: int| 0 <= i <= j < block ==> limits@[i] <= limits@[j],
        forall|i: int, b: int| 0 <= i < block && 0 <= b < counts.len() && counts@[b] > i
            ==> baselines@[b] + i + 1 <= limits@[i],
))]
pub(crate) fn key_limits(
    baselines: &[usize],
    counts: &[usize],
    block: usize,
    extent: usize,
) -> Vec<usize> {
    let mut limits: Vec<usize> = Vec::with_capacity(block);
    let mut running: usize = 1;
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= block,
            limits.len() == index,
            1 <= running <= extent,
            ends_within(baselines@, counts@, extent as int),
            forall|i: int| 0 <= i < index ==> 1 <= #[trigger] limits@[i] && limits@[i] <= running,
            forall|i: int, j: int| 0 <= i <= j < index ==> limits@[i] <= limits@[j],
            forall|i: int, b: int| 0 <= i < index && 0 <= b < counts.len() && counts@[b] > i
                ==> baselines@[b] + i + 1 <= limits@[i],
        decreases block - index,
    ))]
    while index < block {
        let mut row: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                row <= counts.len(),
                index < block,
                1 <= running <= extent,
                ends_within(baselines@, counts@, extent as int),
                forall|i: int| 0 <= i < limits.len() ==> limits@[i] <= running,
                forall|b: int| 0 <= b < row && counts@[b] > index
                    ==> baselines@[b] + index + 1 <= running,
            decreases counts.len() - row,
        ))]
        while row < counts.len() {
            if counts[row] > index {
                // L_b + index < L_b + n_b <= extent, so the sum cannot overflow
                // and the `extent` guard never rejects a row's need.
                let need = baselines[row] + index + 1;
                if need > running && need <= extent {
                    running = need;
                }
            }
            row += 1;
        }
        limits.push(running);
        index += 1;
    }
    limits
}

/// Returns each row's `T` block-entry cache slots for queries.
///
/// Entry `k < n_b` of row `b` sits at slot `L_b + k`; padding entries report
/// `L_b`, an in-range placeholder for queries that every consumer masks.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(slots =>
    requires
        ends_within(baselines@, counts@, extent as int),
        forall|b: int| 0 <= b < counts.len() ==> counts@[b] <= block,
    ensures
        slots.len() == counts.len(),
        forall|b: int| 0 <= b < counts.len() ==> (#[trigger] slots@[b])@.len() == block,
        forall|b: int, k: int| 0 <= b < counts.len() && 0 <= k < counts@[b]
            ==> #[trigger] slots@[b]@[k] == baselines@[b] + k,
        forall|b: int, k: int| 0 <= b < counts.len() && 0 <= k < block
            ==> #[trigger] slots@[b]@[k] <= extent,
))]
pub(crate) fn query_slots(
    baselines: &[usize],
    counts: &[usize],
    block: usize,
    extent: usize,
) -> Vec<Vec<usize>> {
    let mut slots: Vec<Vec<usize>> = Vec::with_capacity(counts.len());
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            slots.len() == row,
            ends_within(baselines@, counts@, extent as int),
            forall|b: int| 0 <= b < counts.len() ==> counts@[b] <= block,
            forall|b: int| 0 <= b < row ==> (#[trigger] slots@[b])@.len() == block,
            forall|b: int, k: int| 0 <= b < row && 0 <= k < counts@[b]
                ==> #[trigger] slots@[b]@[k] == baselines@[b] + k,
            forall|b: int, k: int| 0 <= b < row && 0 <= k < block
                ==> #[trigger] slots@[b]@[k] <= extent,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        let mut entries: Vec<usize> = Vec::with_capacity(block);
        let mut k: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                k <= block,
                row < counts.len(),
                entries.len() == k,
                ends_within(baselines@, counts@, extent as int),
                counts@[row as int] <= block,
                forall|j: int| 0 <= j < k && j < counts@[row as int]
                    ==> entries@[j] == baselines@[row as int] + j,
                forall|j: int| 0 <= j < k ==> entries@[j] <= extent,
            decreases block - k,
        ))]
        while k < block {
            entries.push(if k < counts[row] {
                baselines[row] + k
            } else {
                // A placeholder slot within the read; `L_b <= extent` already.
                baselines[row].min(extent)
            });
            k += 1;
        }
        slots.push(entries);
        row += 1;
    }
    slots
}

#[cfg(test)]
#[path = "batch_tests.rs"]
/// Map, extent, key-limit, and slot planning properties against brute-force oracles.
mod tests;
