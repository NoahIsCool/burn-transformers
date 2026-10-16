//! Verified slot tables of batched KV reads.
//!
//! A batched read slices the same slot range from every row and describes
//! each key by its absolute position and validity, row-major over
//! `[rows, slots]`:
//!
//! * **Full storage.** Slot `s` holds position `s`; the read spans
//!   `max_b L_b` slots and slot `s` of row `b` is valid exactly when it lies in
//!   the row's visible window `[L_b - keep_b, L_b)`.
//! * **Ring storage.** The read spans `min(max_b L_b, W)` physical slots. Slot
//!   `s < L_b` of row `b` holds the newest position `p < L_b` with
//!   `p mod W = s` (the verified `ring_slot_position`), and is valid exactly
//!   when `p` lies in the visible window, i.e. `p + keep_b >= L_b`; slots the
//!   row never wrote are invalid and report position zero.
//! * **Separate staging.** Proposal slot `s` of row `b` holds position
//!   `C_b + s` and is valid exactly when `staged_skip_b <= s < P_b`.
//!
//! Position tables are later converted to Burn's portable `i32` type by the
//! verified `portable_indices`; a position that is not even representable
//! saturates at `usize::MAX` so that conversion reports it.
//!
//! Every function is checked in place by Verus through `verus/cache.rs`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

use crate::cache::retention::{WorkingWindow, ring_slot_position};

#[cfg(verus_keep_ghost)]
verus! {

/// Full-storage entry `index` of a `[rows, slots]` table is valid exactly in its row's window.
pub open spec fn full_entry_correct(
    valid: Seq<bool>,
    lengths: Seq<usize>,
    keep: Seq<usize>,
    slots: int,
    index: int,
) -> bool {
    let row = index / slots;
    let slot = index % slots;
    valid[index] == (slot < lengths[row] && slot >= lengths[row] - keep[row])
}

/// Ring entry `index` holds the newest position with the slot's residue, valid in the window.
pub open spec fn ring_entry_correct(
    positions: Seq<usize>,
    valid: Seq<bool>,
    lengths: Seq<usize>,
    keep: Seq<usize>,
    width: int,
    slots: int,
    index: int,
) -> bool {
    let row = index / slots;
    let slot = index % slots;
    let length = lengths[row] as int;
    let position = positions[index] as int;
    if slot < length {
        &&& position < length
        &&& position % width == slot
        &&& position + width >= length
        &&& valid[index] == (position + keep[row] >= length)
    } else {
        position == 0 && !valid[index]
    }
}

/// Staged entry `index` holds position `committed + slot`, saturated at `usize::MAX`.
pub open spec fn staged_position_correct(
    positions: Seq<usize>,
    committed: Seq<usize>,
    slots: int,
    index: int,
) -> bool {
    let row = index / slots;
    let slot = index % slots;
    positions[index] as int == if committed[row] + slot <= usize::MAX {
        committed[row] + slot
    } else {
        usize::MAX as int
    }
}

/// Staged entry `index` is valid exactly between its row's skipped and staged proposals.
pub(crate) open spec fn staged_entry_valid(
    valid: Seq<bool>,
    staged: Seq<usize>,
    windows: Seq<WorkingWindow>,
    slots: int,
    index: int,
) -> bool {
    let row = index / slots;
    let slot = index % slots;
    valid[index] == (slot >= windows[row].staged_skip && slot < staged[row])
}

} // verus!

/// Returns the largest value, or zero for an empty slice.
///
/// Proved: the result bounds every value and, for a non-empty slice, equals
/// one of them.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(largest =>
    ensures
        forall|i: int| 0 <= i < values.len() ==> #[trigger] values[i] <= largest,
        values.len() == 0 ==> largest == 0,
        values.len() > 0 ==> exists|i: int| 0 <= i < values.len() && #[trigger] values[i]
            == largest,
))]
pub(crate) fn max_value(values: &[usize]) -> usize {
    let mut largest = 0;
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= values.len(),
            forall|i: int| 0 <= i < index ==> #[trigger] values[i] <= largest,
            index == 0 ==> largest == 0,
            index > 0 ==> exists|i: int| 0 <= i < index && #[trigger] values[i] == largest,
        decreases values.len() - index,
    ))]
    while index < values.len() {
        if values[index] > largest {
            largest = values[index];
        }
        index += 1;
    }
    largest
}

/// Returns an empty vector with room for a `[rows, slots]` table when that size fits.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(table =>
    ensures
        table.len() == 0,
))]
fn table_with_capacity<T>(rows: usize, slots: usize) -> Vec<T> {
    match rows.checked_mul(slots) {
        Some(entries) => Vec::with_capacity(entries),
        None => Vec::new(),
    }
}

/// Builds the full-storage validity table: slot `s` of row `b` is visible in `[L_b - keep_b, L_b)`.
///
/// The caller passes `keep_b <= L_b`, which the verified read planners
/// guarantee.
///
/// Proved: the table has `rows * slots` entries, each exactly as described.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(valid =>
    requires
        keep.len() == lengths.len(),
        forall|r: int| 0 <= r < lengths.len() ==> #[trigger] keep[r] <= lengths[r],
    ensures
        valid.len() == lengths.len() * slots,
        forall|index: int| 0 <= index < lengths.len() * slots ==>
            #[trigger] full_entry_correct(valid@, lengths@, keep@, slots as int, index),
))]
pub(crate) fn full_slot_mask(lengths: &[usize], keep: &[usize], slots: usize) -> Vec<bool> {
    let mut valid: Vec<bool> = table_with_capacity(lengths.len(), slots);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= lengths.len(),
            keep.len() == lengths.len(),
            forall|r: int| 0 <= r < lengths.len() ==> #[trigger] keep[r] <= lengths[r],
            valid.len() == row * slots,
            forall|index: int| 0 <= index < row * slots ==>
                #[trigger] full_entry_correct(valid@, lengths@, keep@, slots as int, index),
        decreases lengths.len() - row,
    ))]
    while row < lengths.len() {
        let length = lengths[row];
        let first = length - keep[row];
        let mut slot = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                row < lengths.len(),
                slot <= slots,
                keep.len() == lengths.len(),
                forall|r: int| 0 <= r < lengths.len() ==> #[trigger] keep[r] <= lengths[r],
                length == lengths[row as int],
                first == lengths[row as int] - keep[row as int],
                valid.len() == row * slots + slot,
                forall|index: int| 0 <= index < row * slots + slot ==>
                    #[trigger] full_entry_correct(valid@, lengths@, keep@, slots as int, index),
            decreases slots - slot,
        ))]
        while slot < slots {
            #[cfg(verus_keep_ghost)]
            proof! {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
                    row * slots + slot,
                    slots as int,
                    row as int,
                    slot as int,
                );
                let entry = slot < length && slot >= first;
                let old_valid = valid@;
                assert forall|index: int| 0 <= index < row * slots + slot + 1 implies
                    #[trigger] full_entry_correct(old_valid.push(entry), lengths@, keep@,
                        slots as int, index)
                by {
                    if index < row * slots + slot {
                        assert(full_entry_correct(old_valid, lengths@, keep@, slots as int,
                            index));
                    }
                }
            }
            valid.push(slot < length && slot >= first);
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row + 1) * slots == row * slots + slots) by (nonlinear_arith);
        }
        row += 1;
    }
    valid
}

/// Returns whether every full-storage slot of every row is visible.
///
/// Proved: exactly when every row has length `slots` and keeps all of it
/// (over the zipped rows).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(all_valid =>
    ensures
        all_valid == forall|r: int| 0 <= r < lengths.len() && r < keep.len() ==>
            #[trigger] lengths[r] == slots && keep[r] == lengths[r],
))]
pub(crate) fn full_slots_all_valid(lengths: &[usize], keep: &[usize], slots: usize) -> bool {
    let rows = if lengths.len() <= keep.len() {
        lengths.len()
    } else {
        keep.len()
    };
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows <= lengths.len(),
            rows <= keep.len(),
            rows == lengths.len() || rows == keep.len(),
            forall|r: int| 0 <= r < row ==> #[trigger] lengths[r] == slots && keep[r] == lengths[r],
        decreases rows - row,
    ))]
    while row < rows {
        if lengths[row] != slots || keep[row] != lengths[row] {
            return false;
        }
        row += 1;
    }
    true
}

/// Builds the ring positions and validity of the first `slots` physical slots of every row.
///
/// The caller passes `slots <= width`, the read span `min(max_b L_b, W)`.
///
/// Proved: both tables have `rows * slots` entries; for each written slot
/// the position is the newest one with that residue and the entry is valid
/// exactly when that position lies in the row's visible window; unwritten
/// slots are invalid with position zero.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(table =>
    requires
        width > 0,
        slots <= width,
        keep.len() == lengths.len(),
    ensures
        table.0.len() == lengths.len() * slots,
        table.1.len() == lengths.len() * slots,
        forall|index: int| 0 <= index < lengths.len() * slots ==>
            #[trigger] ring_entry_correct(table.0@, table.1@, lengths@, keep@, width as int,
                slots as int, index),
))]
pub(crate) fn ring_slot_table(
    lengths: &[usize],
    keep: &[usize],
    width: usize,
    slots: usize,
) -> (Vec<usize>, Vec<bool>) {
    let mut positions: Vec<usize> = table_with_capacity(lengths.len(), slots);
    let mut valid: Vec<bool> = table_with_capacity(lengths.len(), slots);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= lengths.len(),
            width > 0,
            slots <= width,
            keep.len() == lengths.len(),
            positions.len() == row * slots,
            valid.len() == row * slots,
            forall|index: int| 0 <= index < row * slots ==>
                #[trigger] ring_entry_correct(positions@, valid@, lengths@, keep@, width as int,
                    slots as int, index),
        decreases lengths.len() - row,
    ))]
    while row < lengths.len() {
        let length = lengths[row];
        let visible = keep[row];
        let mut slot = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                row < lengths.len(),
                slot <= slots,
                width > 0,
                slots <= width,
                keep.len() == lengths.len(),
                length == lengths[row as int],
                visible == keep[row as int],
                positions.len() == row * slots + slot,
                valid.len() == row * slots + slot,
                forall|index: int| 0 <= index < row * slots + slot ==>
                    #[trigger] ring_entry_correct(positions@, valid@, lengths@, keep@,
                        width as int, slots as int, index),
            decreases slots - slot,
        ))]
        while slot < slots {
            // Compare `length - position <= visible` rather than forming
            // `position + visible`, which could overflow for huge lengths.
            let (position, entry) = match ring_slot_position(length, slot, width) {
                Some(position) => (position, length - position <= visible),
                None => (0, false),
            };
            #[cfg(verus_keep_ghost)]
            proof! {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
                    row * slots + slot,
                    slots as int,
                    row as int,
                    slot as int,
                );
                let old_positions = positions@;
                let old_valid = valid@;
                assert forall|index: int| 0 <= index < row * slots + slot + 1 implies
                    #[trigger] ring_entry_correct(old_positions.push(position),
                        old_valid.push(entry), lengths@, keep@, width as int, slots as int,
                        index)
                by {
                    if index < row * slots + slot {
                        assert(ring_entry_correct(old_positions, old_valid, lengths@, keep@,
                            width as int, slots as int, index));
                    }
                }
            }
            positions.push(position);
            valid.push(entry);
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row + 1) * slots == row * slots + slots) by (nonlinear_arith);
        }
        row += 1;
    }
    (positions, valid)
}

/// Builds the separate-staging positions: slot `s` of row `b` holds `C_b + s`.
///
/// Proved: the table has `rows * slots` entries, each exactly `C_b + s`, or
/// `usize::MAX` when that sum is not representable.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(positions =>
    ensures
        positions.len() == committed.len() * slots,
        forall|index: int| 0 <= index < committed.len() * slots ==>
            #[trigger] staged_position_correct(positions@, committed@, slots as int, index),
))]
pub(crate) fn staged_slot_positions(committed: &[usize], slots: usize) -> Vec<usize> {
    let mut positions: Vec<usize> = table_with_capacity(committed.len(), slots);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= committed.len(),
            positions.len() == row * slots,
            forall|index: int| 0 <= index < row * slots ==>
                #[trigger] staged_position_correct(positions@, committed@, slots as int, index),
        decreases committed.len() - row,
    ))]
    while row < committed.len() {
        let length = committed[row];
        let mut slot = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                row < committed.len(),
                slot <= slots,
                length == committed[row as int],
                positions.len() == row * slots + slot,
                forall|index: int| 0 <= index < row * slots + slot ==>
                    #[trigger] staged_position_correct(positions@, committed@, slots as int,
                        index),
            decreases slots - slot,
        ))]
        while slot < slots {
            let position = length.saturating_add(slot);
            #[cfg(verus_keep_ghost)]
            proof! {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
                    row * slots + slot,
                    slots as int,
                    row as int,
                    slot as int,
                );
                let old_positions = positions@;
                assert forall|index: int| 0 <= index < row * slots + slot + 1 implies
                    #[trigger] staged_position_correct(old_positions.push(position), committed@,
                        slots as int, index)
                by {
                    if index < row * slots + slot {
                        assert(staged_position_correct(old_positions, committed@, slots as int,
                            index));
                    }
                }
            }
            positions.push(position);
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row + 1) * slots == row * slots + slots) by (nonlinear_arith);
        }
        row += 1;
    }
    positions
}

/// Builds the separate-staging validity: slot `s` of row `b` is valid in `[skip_b, P_b)`.
///
/// Proved: the table has `rows * slots` entries, each exactly as described.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(valid =>
    requires
        windows.len() == staged.len(),
    ensures
        valid.len() == staged.len() * slots,
        forall|index: int| 0 <= index < staged.len() * slots ==>
            #[trigger] staged_entry_valid(valid@, staged@, windows@, slots as int, index),
))]
pub(crate) fn staged_slot_mask(
    staged: &[usize],
    windows: &[WorkingWindow],
    slots: usize,
) -> Vec<bool> {
    let mut valid: Vec<bool> = table_with_capacity(staged.len(), slots);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= staged.len(),
            windows.len() == staged.len(),
            valid.len() == row * slots,
            forall|index: int| 0 <= index < row * slots ==>
                #[trigger] staged_entry_valid(valid@, staged@, windows@, slots as int, index),
        decreases staged.len() - row,
    ))]
    while row < staged.len() {
        let proposed = staged[row];
        let skip = windows[row].staged_skip;
        let mut slot = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                row < staged.len(),
                slot <= slots,
                windows.len() == staged.len(),
                proposed == staged[row as int],
                skip == windows[row as int].staged_skip,
                valid.len() == row * slots + slot,
                forall|index: int| 0 <= index < row * slots + slot ==>
                    #[trigger] staged_entry_valid(valid@, staged@, windows@, slots as int,
                        index),
            decreases slots - slot,
        ))]
        while slot < slots {
            #[cfg(verus_keep_ghost)]
            proof! {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
                    row * slots + slot,
                    slots as int,
                    row as int,
                    slot as int,
                );
                let entry = slot >= skip && slot < proposed;
                let old_valid = valid@;
                assert forall|index: int| 0 <= index < row * slots + slot + 1 implies
                    #[trigger] staged_entry_valid(old_valid.push(entry), staged@, windows@,
                        slots as int, index)
                by {
                    if index < row * slots + slot {
                        assert(staged_entry_valid(old_valid, staged@, windows@, slots as int,
                            index));
                    }
                }
            }
            valid.push(slot >= skip && slot < proposed);
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row + 1) * slots == row * slots + slots) by (nonlinear_arith);
        }
        row += 1;
    }
    valid
}

/// Returns whether every separate-staging slot of every row is visible.
///
/// Proved: exactly when every row staged `slots` proposals and skips none
/// (over the zipped rows).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(all_valid =>
    ensures
        all_valid == forall|r: int| 0 <= r < staged.len() && r < windows.len() ==>
            #[trigger] staged[r] == slots && windows[r].staged_skip == 0,
))]
pub(crate) fn staged_slots_all_valid(
    staged: &[usize],
    windows: &[WorkingWindow],
    slots: usize,
) -> bool {
    let rows = if staged.len() <= windows.len() {
        staged.len()
    } else {
        windows.len()
    };
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows <= staged.len(),
            rows <= windows.len(),
            rows == staged.len() || rows == windows.len(),
            forall|r: int| 0 <= r < row ==> #[trigger] staged[r] == slots,
            forall|r: int| 0 <= r < row ==> (#[trigger] windows[r]).staged_skip == 0,
        decreases rows - row,
    ))]
    while row < rows {
        if staged[row] != slots || windows[row].staged_skip != 0 {
            return false;
        }
        row += 1;
    }
    true
}

/// Returns the first row vector that differs from `reference`.
///
/// Proved: `None` exactly when every vector equals `reference` element for
/// element; otherwise the lowest differing vector.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(first =>
    ensures
        first is None <==> forall|l: int| 0 <= l < vectors.len() ==>
            #[trigger] vectors@[l]@ == reference@,
        first matches Some(layer) ==> {
            &&& layer < vectors.len()
            &&& vectors@[layer as int]@ != reference@
            &&& forall|l: int| 0 <= l < layer ==> #[trigger] vectors@[l]@ == reference@
        },
))]
pub(crate) fn first_unequal_vector(vectors: &[Vec<usize>], reference: &[usize]) -> Option<usize> {
    let mut layer = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= vectors.len(),
            forall|l: int| 0 <= l < layer ==> #[trigger] vectors@[l]@ == reference@,
        decreases vectors.len() - layer,
    ))]
    while layer < vectors.len() {
        let candidate = &vectors[layer];
        if candidate.len() != reference.len() {
            return Some(layer);
        }
        let mut index = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                layer < vectors.len(),
                index <= reference.len(),
                candidate@ == vectors@[layer as int]@,
                candidate.len() == reference.len(),
                forall|l: int| 0 <= l < layer ==> #[trigger] vectors@[l]@ == reference@,
                forall|i: int| 0 <= i < index ==> #[trigger] candidate@[i] == reference@[i],
            decreases reference.len() - index,
        ))]
        while index < reference.len() {
            if candidate[index] != reference[index] {
                return Some(layer);
            }
            index += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(candidate@ =~= reference@);
        }
        layer += 1;
    }
    None
}

#[cfg(test)]
#[path = "slots_tests.rs"]
mod tests;
