//! Verified snapshot indexing for speculative recurrent state.
//!
//! A recurrent layer stages the state *after* every proposal token: snapshot
//! `i` of row `b` is the state after proposal token `i`, so a row with `s_b`
//! staged snapshots reads snapshot `s_b - 1` as its working state and
//! accepting `a_b > 0` tokens commits snapshot `a_b - 1`. Rows with no
//! snapshot keep their committed state.
//!
//! Staging tensors are laid out `[batch, proposal, ...]` with `proposal = P`
//! slots per row. A ragged batched read views them as `[batch * P, ...]` and
//! gathers flat row `b * P + max(s_b, 1) - 1`; when `s_b <= P` that flat index
//! decodes back to exactly `(b, max(s_b, 1) - 1)`, so every row reads its own
//! snapshot and never another row's.
//!
//! The module is burn-free so Verus checks it in place through
//! `verus/cache.rs`; the tensor slicing driven by these indices remains
//! covered by the recurrent cache tests.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Flat index `row * capacity + max(staged, 1) - 1` of a row's working snapshot.
pub open spec fn flat_snapshot(row: int, staged: int, capacity: int) -> int {
    row * capacity + (if staged > 0 { staged - 1 } else { 0 })
}

/// Euclidean division recovers `(row, slot)` from `row * capacity + slot`.
proof fn lemma_flat_decodes(row: int, slot: int, capacity: int)
    requires
        capacity > 0,
        row >= 0,
        0 <= slot < capacity,
    ensures
        (row * capacity + slot) / capacity == row,
        (row * capacity + slot) % capacity == slot,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
        row * capacity + slot,
        capacity,
        row,
        slot,
    );
}

} // verus!

/// Returns the snapshot holding the state after the last of `count` tokens.
///
/// Proved: `None` exactly for `count == 0` (the committed state applies),
/// otherwise snapshot `count - 1`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(snapshot =>
    ensures
        snapshot is None <==> count == 0,
        snapshot matches Some(index) ==> index + 1 == count,
))]
pub(crate) fn last_snapshot(count: usize) -> Option<usize> {
    if count == 0 { None } else { Some(count - 1) }
}

/// Returns the shared snapshot index `s - 1` when every row staged the same `s > 0`.
///
/// Proved: `Some(s - 1)` exactly when the batch is non-empty and every row
/// staged the same positive count `s`; otherwise `None`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(snapshot =>
    ensures
        snapshot is Some <==> {
            &&& staged_lengths.len() > 0
            &&& staged_lengths[0] > 0
            &&& forall|i: int| 0 <= i < staged_lengths.len() ==>
                #[trigger] staged_lengths[i] == staged_lengths[0]
        },
        snapshot matches Some(index) ==> index + 1 == staged_lengths[0],
))]
pub(crate) fn uniform_snapshot(staged_lengths: &[usize]) -> Option<usize> {
    if staged_lengths.is_empty() {
        return None;
    }
    let first = staged_lengths[0];
    if first == 0 {
        return None;
    }
    let mut row = 1;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            1 <= row <= staged_lengths.len(),
            first == staged_lengths[0],
            forall|i: int| 0 <= i < row ==> #[trigger] staged_lengths[i] == first,
        decreases staged_lengths.len() - row,
    ))]
    while row < staged_lengths.len() {
        if staged_lengths[row] != first {
            return None;
        }
        row += 1;
    }
    Some(first - 1)
}

/// Computes each row's flat working-snapshot index `row * capacity + max(staged, 1) - 1`.
///
/// Rows without a proposal point at their first (unused) snapshot slot; the
/// caller replaces them with committed state. Indices are returned in Burn's
/// portable `i32` type.
///
/// Proved: on success every entry equals its flat index exactly and, for a
/// row with `staged <= capacity`, decodes to that row and to snapshot
/// `max(staged, 1) - 1` of it. The conversion fails exactly when some flat
/// index exceeds `i32::MAX`, reporting the first such index (saturated at
/// `usize::MAX` when it is not even representable).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result is Ok <==> forall|r: int| 0 <= r < staged_lengths.len() ==>
            flat_snapshot(r, #[trigger] staged_lengths[r] as int, capacity as int) <= i32::MAX,
        result matches Ok(flat) ==> {
            &&& flat.len() == staged_lengths.len()
            &&& forall|r: int| 0 <= r < staged_lengths.len() ==>
                #[trigger] flat[r] as int == flat_snapshot(r, staged_lengths[r] as int,
                    capacity as int)
            &&& forall|r: int| 0 <= r < staged_lengths.len() && 0 < capacity
                && staged_lengths[r] <= capacity ==> {
                &&& (#[trigger] flat[r] as int) / (capacity as int) == r
                &&& (flat[r] as int) % (capacity as int)
                    == if staged_lengths[r] > 0 { staged_lengths[r] - 1 } else { 0 }
            }
        },
        result matches Err(index) ==> exists|r: int| {
            &&& 0 <= r < staged_lengths.len()
            &&& #[trigger] flat_snapshot(r, staged_lengths[r] as int, capacity as int) > i32::MAX
            &&& index as int == if flat_snapshot(r, staged_lengths[r] as int, capacity as int)
                <= usize::MAX {
                flat_snapshot(r, staged_lengths[r] as int, capacity as int)
            } else {
                usize::MAX as int
            }
            &&& forall|q: int| 0 <= q < r ==>
                flat_snapshot(q, #[trigger] staged_lengths[q] as int, capacity as int)
                    <= i32::MAX
        },
))]
pub(crate) fn flat_snapshot_indices(
    staged_lengths: &[usize],
    capacity: usize,
) -> Result<Vec<i32>, usize> {
    let mut flat: Vec<i32> = Vec::with_capacity(staged_lengths.len());
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= staged_lengths.len(),
            flat.len() == row,
            forall|r: int| 0 <= r < row ==>
                flat_snapshot(r, #[trigger] staged_lengths[r] as int, capacity as int)
                    <= i32::MAX,
            forall|r: int| 0 <= r < row ==>
                #[trigger] flat[r] as int == flat_snapshot(r, staged_lengths[r] as int,
                    capacity as int),
        decreases staged_lengths.len() - row,
    ))]
    while row < staged_lengths.len() {
        let snapshot = if staged_lengths[row] > 0 {
            staged_lengths[row] - 1
        } else {
            0
        };
        let index = match row.checked_mul(capacity) {
            Some(base) => match base.checked_add(snapshot) {
                Some(index) => index,
                None => return Err(usize::MAX),
            },
            None => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(row as int * capacity as int + snapshot as int > usize::MAX)
                        by (nonlinear_arith)
                        requires
                            row as int * capacity as int > usize::MAX,
                            snapshot >= 0,
                    ;
                }
                return Err(usize::MAX);
            }
        };
        if index > i32::MAX as usize {
            return Err(index);
        }
        flat.push(index as i32);
        row += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|r: int| 0 <= r < staged_lengths.len() && 0 < capacity
            && staged_lengths[r] <= capacity implies {
            &&& (#[trigger] flat[r] as int) / (capacity as int) == r
            &&& (flat[r] as int) % (capacity as int)
                == if staged_lengths[r] > 0 { staged_lengths[r] - 1 } else { 0 }
        } by {
            let slot = if staged_lengths[r] > 0 { staged_lengths[r] - 1 } else { 0 };
            lemma_flat_decodes(r, slot, capacity as int);
        }
    }
    Ok(flat)
}

/// Returns the first row whose staging offset differs from row zero's.
///
/// Proved: `None` exactly when every row shares row zero's offset (including
/// an empty batch); otherwise the lowest differing row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(first =>
    ensures
        first is None <==> forall|i: int| 0 <= i < offsets.len() ==>
            #[trigger] offsets[i] == offsets[0],
        first matches Some(row) ==> {
            &&& 0 < row < offsets.len()
            &&& offsets[row as int] != offsets[0]
            &&& forall|i: int| 0 <= i < row ==> #[trigger] offsets[i] == offsets[0]
        },
))]
pub(crate) fn first_nonuniform_offset(offsets: &[usize]) -> Option<usize> {
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= offsets.len(),
            forall|i: int| 0 <= i < row ==> #[trigger] offsets[i] == offsets[0],
        decreases offsets.len() - row,
    ))]
    while row < offsets.len() {
        if offsets[row] != offsets[0] {
            return Some(row);
        }
        row += 1;
    }
    None
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
