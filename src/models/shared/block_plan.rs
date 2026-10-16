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
//! This module is dependency-free and family-independent: DeepSeek-V2 (latent
//! attention) and Muse Glimmer (grouped-query attention with ring caches)
//! drive their batched cached steps through it, and [`super::block`] applies
//! the plans to tensors. Its functions are verified in place by Verus through
//! `verus/models.rs` (no overflow, map correctness, exact extents and key
//! limits, query validity, implicit positions, and per-row rotary contexts);
//! index conversion to `i64` reuses the verified [`super::plan::signed_indices`].

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// `s` is strictly increasing and every entry lies below `bound`.
///
/// Twin of `increasing_below` in `src/models/qwen3_5/plan.rs` (checked by
/// `verus/qwen.rs`); keep the two definitions identical.
pub open spec fn increasing_below(s: Seq<usize>, bound: int) -> bool {
    &&& forall|i: int, j: int| 0 <= i < j < s.len() ==> s[i] < s[j]
    &&& forall|i: int| 0 <= i < s.len() ==> s[i] < bound
}


/// Every row's cache end `L_b + n_b` fits within `extent` (and `usize`).
pub open spec fn ends_within(baselines: Seq<usize>, counts: Seq<usize>, extent: int) -> bool {
    &&& baselines.len() == counts.len()
    &&& forall|b: int| 0 <= b < counts.len() ==> baselines[b] + counts[b] <= extent
}

/// Keys `0..L_b + j + 1` that query `j` of row `b` needs (its causal prefix).
pub open spec fn key_need(baselines: Seq<usize>, b: int, j: int) -> int {
    baselines[b] + j + 1
}

/// Row `b` contributes tokens (`n_b > 0`).
pub open spec fn active(counts: Seq<usize>, b: int) -> bool {
    0 <= b < counts.len() && counts[b] > 0
}

/// Row `row` appends tokens after cached keys and its context passes rotary limit `rope`.
pub open spec fn exceeds_cached_limit(
    lengths: Seq<usize>,
    counts: Seq<usize>,
    contexts: Seq<usize>,
    limits: Seq<Option<usize>>,
    row: int,
    rope: int,
) -> bool {
    &&& 0 <= row < counts.len()
    &&& 0 <= rope < limits.len()
    &&& counts[row] > 0
    &&& lengths[row] > 0
    &&& match limits[rope] {
        Some(limit) => contexts[row] > limit,
        None => false,
    }
}

/// `max(value, 1)`.
pub open spec fn at_least_one(value: int) -> int {
    if value < 1 {
        1
    } else {
        value
    }
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
/// scatters the zero row `T`. Proved: exactly these gather and scatter
/// entries, including the padding gathers.
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
        forall|k: int| columns.len() <= k < block
            ==> maps.gather@[k] == (if columns.len() > 0 { columns@[0] } else { 0 }),
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
            filler == (if columns.len() > 0 { columns@[0] } else { 0 }),
            increasing_below(columns@, sequence as int),
            columns.len() <= block,
            forall|i: int| 0 <= i < k ==> gather@[i] < sequence,
            forall|i: int| 0 <= i < k && i < columns.len() ==> gather@[i] == columns@[i],
            forall|i: int| columns.len() <= i < k ==> gather@[i] == filler,
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
/// Proved: both maxima are exact (bounding every row and attained by one,
/// zero for no rows), and `None` happens only for those two reasons.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Some((block, extent)) => {
                &&& baselines.len() == counts.len()
                &&& ends_within(baselines@, counts@, extent as int)
                &&& forall|b: int| 0 <= b < counts.len() ==> counts@[b] <= block
                &&& (counts.len() > 0 ==> exists|b: int| 0 <= b < counts.len() && counts@[b] == block)
                &&& (counts.len() > 0 ==> exists|b: int| 0 <= b < counts.len()
                    && #[trigger] baselines@[b] + counts@[b] == extent)
                &&& (counts.len() == 0 ==> block == 0 && extent == 0)
            },
            None => baselines.len() != counts.len() || exists|b: int| 0 <= b < counts.len()
                && #[trigger] baselines@[b] + counts@[b] > usize::MAX,
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
    proof_decl! { let ghost mut witness: int = 0; let ghost mut furthest: int = 0; }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            baselines.len() == counts.len(),
            forall|b: int| 0 <= b < row ==> baselines@[b] + counts@[b] <= extent,
            forall|b: int| 0 <= b < row ==> counts@[b] <= block,
            row > 0 ==> 0 <= witness < row && counts@[witness] == block,
            row > 0 ==> 0 <= furthest < row && baselines@[furthest] + counts@[furthest] == extent,
            row == 0 ==> block == 0 && extent == 0 && furthest == 0,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        let end = baselines[row].checked_add(counts[row])?;
        if end > extent {
            extent = end;
            #[cfg(verus_keep_ghost)]
            proof! {
                furthest = row as int;
            }
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
/// Proved: each limit is exactly `max(1, max_{j <= i, n_b > j} (L_b + j + 1))`
/// — it covers every need and is either one or attained by a need, so
/// bounded attention never reads a key no query of the prefix can see.
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
        forall|i: int| 0 <= i < block ==> #[trigger] limits@[i] == 1
            || exists|j: int, b: int| 0 <= j <= i && 0 <= b < counts.len() && counts@[b] > j
                && limits@[i] == #[trigger] key_need(baselines@, b, j),
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
    #[cfg(verus_keep_ghost)]
    proof_decl! { let ghost mut need_query: int = 0; let ghost mut need_row: int = 0; }
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
            forall|i: int| 0 <= i < index ==> #[trigger] limits@[i] == 1
                || exists|j: int, b: int| 0 <= j <= i && 0 <= b < counts.len() && counts@[b] > j
                    && limits@[i] == #[trigger] key_need(baselines@, b, j),
            // The running maximum is one or the need of an earlier (query, row).
            running == 1 || (0 <= need_query < index && 0 <= need_row < counts.len()
                && counts@[need_row] > need_query
                && running == key_need(baselines@, need_row, need_query)),
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
                running == 1 || (0 <= need_query <= index && 0 <= need_row < counts.len()
                    && counts@[need_row] > need_query
                    && running == key_need(baselines@, need_row, need_query)),
            decreases counts.len() - row,
        ))]
        while row < counts.len() {
            if counts[row] > index {
                // L_b + index < L_b + n_b <= extent, so the sum cannot overflow
                // and the `extent` guard never rejects a row's need.
                let need = baselines[row] + index + 1;
                if need > running && need <= extent {
                    running = need;
                    #[cfg(verus_keep_ghost)]
                    proof! {
                        need_query = index as int;
                        need_row = row as int;
                    }
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
/// Proved: exactly these slots, all within the key extent.
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
        forall|b: int, k: int| 0 <= b < counts.len() && counts@[b] <= k < block
            ==> #[trigger] slots@[b]@[k] == baselines@[b],
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
            forall|b: int, k: int| 0 <= b < row && counts@[b] <= k < block
                ==> #[trigger] slots@[b]@[k] == baselines@[b],
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
                forall|j: int| counts@[row as int] <= j < k ==> entries@[j] == baselines@[row as int],
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

/// Returns whether packing is the identity: every row fills all `S` columns.
///
/// Proved: exactly when `T = S` and every `n_b = S`; then the gather and
/// scatter maps would both be the identity, so no kernel needs to run.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(identity =>
    ensures
        identity == (block == sequence && forall|b: int| 0 <= b < counts.len() ==> counts@[b] == sequence),
))]
pub(crate) fn identity_layout(counts: &[usize], block: usize, sequence: usize) -> bool {
    if block != sequence {
        return false;
    }
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            forall|b: int| 0 <= b < row ==> counts@[b] == sequence,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] != sequence {
            return false;
        }
        row += 1;
    }
    true
}

/// Returns each row's query validity `k < n_b`, or `None` when every entry is valid.
///
/// Proved: `None` exactly when every `n_b = T`; otherwise row `b` holds `T`
/// flags and flag `k` is `k < n_b`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(validity =>
    ensures
        match validity {
            None => forall|b: int| 0 <= b < counts.len() ==> counts@[b] == block,
            Some(rows) => {
                &&& exists|b: int| 0 <= b < counts.len() && counts@[b] != block
                &&& rows.len() == counts.len()
                &&& forall|b: int| 0 <= b < counts.len() ==> (#[trigger] rows@[b])@.len() == block
                &&& forall|b: int, k: int| 0 <= b < counts.len() && 0 <= k < block
                    ==> #[trigger] rows@[b]@[k] == (k < counts@[b])
            },
        },
))]
pub(crate) fn query_validity(counts: &[usize], block: usize) -> Option<Vec<Vec<bool>>> {
    let mut ragged = false;
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            ragged == (exists|b: int| 0 <= b < row && counts@[b] != block),
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] != block {
            ragged = true;
        }
        row += 1;
    }
    if !ragged {
        return None;
    }
    let mut rows: Vec<Vec<bool>> = Vec::with_capacity(counts.len());
    let mut b: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            b <= counts.len(),
            rows.len() == b,
            forall|r: int| 0 <= r < b ==> (#[trigger] rows@[r])@.len() == block,
            forall|r: int, k: int| 0 <= r < b && 0 <= k < block
                ==> #[trigger] rows@[r]@[k] == (k < counts@[r]),
        decreases counts.len() - b,
    ))]
    while b < counts.len() {
        let mut flags: Vec<bool> = Vec::with_capacity(block);
        let mut k: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                b < counts.len(),
                k <= block,
                flags.len() == k,
                forall|j: int| 0 <= j < k ==> flags@[j] == (j < counts@[b as int]),
            decreases block - k,
        ))]
        while k < block {
            flags.push(k < counts[b]);
            k += 1;
        }
        rows.push(flags);
        b += 1;
    }
    Some(rows)
}

/// Returns each row's `T` implicit positions: `L_b + k` for valid entries, zero for padding.
///
/// Proved: exact for every entry and overflow-free whenever every row's
/// cache end `L_b + n_b` is representable (as [`batch_extents`] checks).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(positions =>
    requires
        ends_within(baselines@, counts@, usize::MAX as int),
    ensures
        positions.len() == counts.len(),
        forall|b: int| 0 <= b < counts.len() ==> (#[trigger] positions@[b])@.len() == block,
        forall|b: int, k: int| 0 <= b < counts.len() && 0 <= k < block
            ==> #[trigger] positions@[b]@[k] == (if k < counts@[b] { baselines@[b] + k } else { 0 }),
))]
pub(crate) fn implicit_positions(
    baselines: &[usize],
    counts: &[usize],
    block: usize,
) -> Vec<Vec<usize>> {
    let mut positions: Vec<Vec<usize>> = Vec::with_capacity(counts.len());
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            positions.len() == row,
            ends_within(baselines@, counts@, usize::MAX as int),
            forall|b: int| 0 <= b < row ==> (#[trigger] positions@[b])@.len() == block,
            forall|b: int, k: int| 0 <= b < row && 0 <= k < block
                ==> #[trigger] positions@[b]@[k] == (if k < counts@[b] { baselines@[b] + k } else { 0 }),
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        let mut entries: Vec<usize> = Vec::with_capacity(block);
        let mut k: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                row < counts.len(),
                k <= block,
                entries.len() == k,
                ends_within(baselines@, counts@, usize::MAX as int),
                forall|j: int| 0 <= j < k ==> entries@[j]
                    == (if j < counts@[row as int] { baselines@[row as int] + j } else { 0 }),
            decreases block - k,
        ))]
        while k < block {
            // `k < n_b` keeps `L_b + k < L_b + n_b <= usize::MAX`.
            entries.push(if k < counts[row] {
                baselines[row] + k
            } else {
                0
            });
            k += 1;
        }
        positions.push(entries);
        row += 1;
    }
    positions
}

/// Clamps key-prefix limits to the `keys` a cache read returned.
///
/// Proved: each result is `min(λ_i, keys)`, the clamp keeps limits
/// nondecreasing, and limits in `1..` stay in `1..=keys` when `keys >= 1`,
/// as the chunk planner requires.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(clamped =>
    ensures
        clamped.len() == limits.len(),
        forall|i: int| 0 <= i < limits.len()
            ==> #[trigger] clamped@[i] == (if limits@[i] < keys { limits@[i] } else { keys }),
        (forall|i: int, j: int| 0 <= i <= j < limits.len() ==> limits@[i] <= limits@[j])
            ==> forall|i: int, j: int| 0 <= i <= j < limits.len() ==> clamped@[i] <= clamped@[j],
        keys >= 1 && (forall|i: int| 0 <= i < limits.len() ==> limits@[i] >= 1)
            ==> forall|i: int| 0 <= i < limits.len() ==> 1 <= #[trigger] clamped@[i] <= keys,
))]
pub(crate) fn clamp_limits(limits: &[usize], keys: usize) -> Vec<usize> {
    let mut clamped: Vec<usize> = Vec::with_capacity(limits.len());
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= limits.len(),
            clamped.len() == index,
            forall|i: int| 0 <= i < index
                ==> #[trigger] clamped@[i] == (if limits@[i] < keys { limits@[i] } else { keys }),
        decreases limits.len() - index,
    ))]
    while index < limits.len() {
        clamped.push(limits[index].min(keys));
        index += 1;
    }
    clamped
}

/// Rotary contexts of one batched call (see [`rotary_contexts`]).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RotaryContexts {
    /// The first active row's context (at least one), or one without active rows.
    pub(crate) first: usize,
    /// Whether every active row's context equals `first`.
    pub(crate) uniform: bool,
    /// Each row's own context: `first` for an inactive row, else `max(context, 1)`.
    pub(crate) rows: Vec<usize>,
}

/// Plans the rotary contexts of a batched cached call.
///
/// Static rotary scalings use one table at `first`; context-dependent ones
/// evaluate every row at its own context, as an independent batch-one call
/// would, unless the call is `uniform`. Proved: `first` is the first active
/// row's context raised to at least one (one without active rows),
/// `uniform` is exactly "every active context equals `first`", and row `b`
/// uses `first` when inactive and `max(context_b, 1)` otherwise.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    requires
        counts.len() == contexts.len(),
    ensures
        (exists|b: int| active(counts@, b)) ==> exists|b: int| #[trigger] active(counts@, b)
            && (forall|c: int| 0 <= c < b ==> !active(counts@, c))
            && plan.first == at_least_one(contexts@[b] as int),
        !(exists|b: int| active(counts@, b)) ==> plan.first == 1,
        plan.uniform == (forall|b: int| #[trigger] active(counts@, b) ==> contexts@[b] == plan.first),
        plan.rows.len() == counts.len(),
        forall|b: int| 0 <= b < counts.len() ==> #[trigger] plan.rows@[b]
            == (if counts@[b] == 0 { plan.first as int } else { at_least_one(contexts@[b] as int) }),
))]
pub(crate) fn rotary_contexts(counts: &[usize], contexts: &[usize]) -> RotaryContexts {
    let mut first: usize = 1;
    let mut found = false;
    let mut row: usize = 0;
    #[cfg(verus_keep_ghost)]
    proof_decl! { let ghost mut first_row: int = 0; }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            counts.len() == contexts.len(),
            found ==> 0 <= first_row < row && active(counts@, first_row)
                && (forall|c: int| 0 <= c < first_row ==> !active(counts@, c))
                && first == at_least_one(contexts@[first_row] as int),
            !found ==> first == 1 && forall|c: int| 0 <= c < row ==> !active(counts@, c),
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if !found && counts[row] > 0 {
            first = contexts[row].max(1);
            found = true;
            #[cfg(verus_keep_ghost)]
            proof! {
                first_row = row as int;
            }
        }
        row += 1;
    }
    let mut uniform = true;
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= counts.len(),
            counts.len() == contexts.len(),
            uniform == (forall|b: int| 0 <= b < index && #[trigger] active(counts@, b)
                ==> contexts@[b] == first),
        decreases counts.len() - index,
    ))]
    while index < counts.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(active(counts@, index as int) == (counts@[index as int] > 0));
        }
        if counts[index] > 0 && contexts[index] != first {
            uniform = false;
        }
        index += 1;
    }
    let mut rows: Vec<usize> = Vec::with_capacity(counts.len());
    let mut b: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            b <= counts.len(),
            counts.len() == contexts.len(),
            rows.len() == b,
            forall|r: int| 0 <= r < b ==> #[trigger] rows@[r]
                == (if counts@[r] == 0 { first as int } else { at_least_one(contexts@[r] as int) }),
        decreases counts.len() - b,
    ))]
    while b < counts.len() {
        rows.push(if counts[b] == 0 {
            first
        } else {
            contexts[b].max(1)
        });
        b += 1;
    }
    RotaryContexts {
        first,
        uniform,
        rows,
    }
}

/// Returns the first `(context, limit)` that would extend cached rotary keys past a limit.
///
/// Context-dependent rotary scalings change their frequencies beyond a
/// cached-context limit, so a row that already caches rotated keys
/// (`L_b > 0`) and appends tokens (`n_b > 0`) must keep its evaluated
/// context within every configuration's limit (`None` never invalidates
/// keys). Rows are visited in order and, within a row, configurations in
/// order. Proved: `Some` names the first violating `(row, configuration)`
/// pair in that order, and `None` means no pair violates.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        lengths.len() == counts.len(),
        contexts.len() == counts.len(),
    ensures
        match found {
            Some((context, limit)) => exists|row: int, rope: int|
                #[trigger] exceeds_cached_limit(lengths@, counts@, contexts@, limits@, row, rope)
                && context == contexts@[row] && limits@[rope] == Some(limit)
                && (forall|r: int, i: int| 0 <= r < row
                    ==> !#[trigger] exceeds_cached_limit(lengths@, counts@, contexts@, limits@, r, i))
                && (forall|i: int| 0 <= i < rope
                    ==> !#[trigger] exceeds_cached_limit(lengths@, counts@, contexts@, limits@, row, i)),
            None => forall|r: int, i: int|
                !#[trigger] exceeds_cached_limit(lengths@, counts@, contexts@, limits@, r, i),
        },
))]
#[allow(
    clippy::collapsible_if,
    reason = "Verus compiles verified bodies with an older edition that has no let-chains"
)]
pub(crate) fn first_uncacheable_context(
    lengths: &[usize],
    counts: &[usize],
    contexts: &[usize],
    limits: &[Option<usize>],
) -> Option<(usize, usize)> {
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            lengths.len() == counts.len(),
            contexts.len() == counts.len(),
            forall|r: int, i: int| 0 <= r < row
                ==> !#[trigger] exceeds_cached_limit(lengths@, counts@, contexts@, limits@, r, i),
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] > 0 && lengths[row] > 0 {
            let mut rope: usize = 0;
            #[cfg_attr(verus_keep_ghost, verus_spec(
                invariant
                    row < counts.len(),
                    rope <= limits.len(),
                    lengths.len() == counts.len(),
                    contexts.len() == counts.len(),
                    counts@[row as int] > 0 && lengths@[row as int] > 0,
                    forall|r: int, i: int| 0 <= r < row
                        ==> !#[trigger] exceeds_cached_limit(lengths@, counts@, contexts@, limits@, r, i),
                    forall|i: int| 0 <= i < rope
                        ==> !#[trigger] exceeds_cached_limit(lengths@, counts@, contexts@, limits@, row as int, i),
                decreases limits.len() - rope,
            ))]
            while rope < limits.len() {
                if let Some(limit) = limits[rope] {
                    if contexts[row] > limit {
                        #[cfg(verus_keep_ghost)]
                        proof! {
                            assert(exceeds_cached_limit(lengths@, counts@, contexts@, limits@,
                                row as int, rope as int));
                        }
                        return Some((contexts[row], limit));
                    }
                }
                rope += 1;
            }
        }
        row += 1;
    }
    None
}

#[cfg(test)]
#[path = "block_plan_tests.rs"]
/// Map, extent, key-limit, and slot planning properties against brute-force oracles.
mod tests;
