//! Verified write-set coverage and lifecycle versions of model steps.
//!
//! A step appends `n_b` tokens to every row `b` of one bank. Publication
//! requires each `(group, layer)` to write every *active* row (`n_b > 0`)
//! exactly once and never an inactive row. Coverage is tracked per
//! `(group, layer)` as one flag per row:
//!
//! * marking one row requires it to be in the batch, active, and unmarked;
//! * marking a batched write marks every active row, each of which must be
//!   unmarked;
//! * finishing requires every active row to be marked.
//!
//! Starting from no marks, any sequence of successful marks followed by a
//! successful finish therefore writes every active row exactly once: a second
//! write of a row fails its mark, and a missing row fails the finish.
//!
//! A step is also bound to the lifecycle version of its bank. Every
//! operation that could move the bank's cursors or slots bumps that version,
//! and a step whose recorded version differs is stale. Bumping one bank
//! changes its version and leaves the other unchanged.
//!
//! Every function is checked in place by Verus through `verus/cache.rs`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

use crate::cache::tensor::plan::next_version;

/// Bank whose groups a step writes.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepBank {
    /// The decoder's ordinary cache groups and committed cursor.
    Main,
    /// The auxiliary multi-token-prediction groups and their own cursor.
    Mtp,
}

/// Integer reason a coverage mark is rejected.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CoverageViolation {
    /// The written row lies outside the batch.
    RowOutOfBounds {
        /// Requested row.
        row: usize,
    },
    /// The written row has no new tokens in this step.
    InactiveRow {
        /// Requested row.
        row: usize,
    },
    /// A row was already written by an earlier entry of the step.
    Duplicate {
        /// Row written twice.
        row: usize,
    },
}

#[cfg(verus_keep_ghost)]
verus! {

/// The row named by a duplicate-write violation, or `-1` for any other violation.
pub(crate) open spec fn duplicate_row(violation: CoverageViolation) -> int {
    match violation {
        CoverageViolation::Duplicate { row } => row as int,
        _ => -1,
    }
}

/// Every active row of `counts` is unmarked in `covered`.
pub open spec fn active_rows_unmarked(covered: Seq<bool>, counts: Seq<usize>) -> bool {
    forall|i: int| 0 <= i < counts.len() && #[trigger] counts[i] > 0 ==> !covered[i]
}

} // verus!

/// Marks one written row, or every active row for a batched write (`row == None`).
///
/// Proved: a single row is accepted exactly when it is in the batch, active,
/// and unmarked, and then exactly that flag is set; a batched write is
/// accepted exactly when every active row is unmarked, and then exactly the
/// active rows are set. A duplicate names the first already-marked row (its
/// earlier active rows are then marked, as the scan left them); every other
/// rejection leaves the flags untouched.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        old(covered).len() == counts.len(),
    ensures
        final(covered).len() == counts.len(),
        row is Some && row->0 >= counts.len() ==>
            result == Err::<(), CoverageViolation>(CoverageViolation::RowOutOfBounds {
                row: row->0,
            }) && final(covered)@ == old(covered)@,
        row is Some && row->0 < counts.len() && counts[row->0 as int] == 0 ==>
            result == Err::<(), CoverageViolation>(CoverageViolation::InactiveRow {
                row: row->0,
            }) && final(covered)@ == old(covered)@,
        row is Some && row->0 < counts.len() && counts[row->0 as int] > 0 ==> {
            &&& old(covered)[row->0 as int] ==> result == Err::<(), CoverageViolation>(
                CoverageViolation::Duplicate { row: row->0 })
            &&& !old(covered)[row->0 as int] ==> result is Ok
            &&& final(covered)@ == old(covered)@.update(row->0 as int, true)
        },
        row is None ==> (result is Ok <==> active_rows_unmarked(old(covered)@, counts@)),
        row is None && result is Ok ==> forall|i: int| 0 <= i < counts.len() ==>
            #[trigger] final(covered)[i] == (old(covered)[i] || counts[i] > 0),
        row is None && result is Err ==> {
            let duplicate = duplicate_row(result->Err_0);
            &&& 0 <= duplicate < counts.len()
            &&& counts[duplicate] > 0
            &&& old(covered)[duplicate]
            &&& forall|i: int| 0 <= i < duplicate && #[trigger] counts[i] > 0
                ==> !old(covered)[i]
            &&& forall|i: int| 0 <= i < counts.len() ==>
                #[trigger] final(covered)[i] == (old(covered)[i]
                    || (counts[i] > 0 && i < duplicate))
        },
))]
pub(crate) fn mark_covered(
    covered: &mut [bool],
    row: Option<usize>,
    counts: &[usize],
) -> Result<(), CoverageViolation> {
    match row {
        Some(written) => {
            if written >= counts.len() {
                return Err(CoverageViolation::RowOutOfBounds { row: written });
            }
            if counts[written] == 0 {
                return Err(CoverageViolation::InactiveRow { row: written });
            }
            let previous = covered[written];
            covered[written] = true;
            if previous {
                return Err(CoverageViolation::Duplicate { row: written });
            }
            Ok(())
        }
        None => {
            let mut index = 0;
            #[cfg_attr(verus_keep_ghost, verus_spec(
                invariant
                    row is None,
                    index <= counts.len(),
                    covered.len() == counts.len(),
                    forall|i: int| 0 <= i < index && #[trigger] counts[i] > 0
                        ==> !old(covered)[i],
                    forall|i: int| 0 <= i < counts.len() ==>
                        #[trigger] covered[i] == (old(covered)[i]
                            || (counts[i] > 0 && i < index)),
                decreases counts.len() - index,
            ))]
            while index < counts.len() {
                if counts[index] > 0 {
                    let previous = covered[index];
                    covered[index] = true;
                    if previous {
                        #[cfg(verus_keep_ghost)]
                        proof! {
                            // Row `index` is an active, already-marked witness.
                            assert(counts[index as int] > 0 && old(covered)[index as int]);
                            assert(!active_rows_unmarked(old(covered)@, counts@));
                        }
                        return Err(CoverageViolation::Duplicate { row: index });
                    }
                }
                index += 1;
            }
            Ok(())
        }
    }
}

/// Returns the first active row that no entry of the step marked.
///
/// Proved: `None` exactly when every active row is marked; otherwise the
/// lowest active, unmarked row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(first =>
    requires
        covered.len() == counts.len(),
    ensures
        first is None <==> forall|i: int| 0 <= i < counts.len() && #[trigger] counts[i] > 0
            ==> covered[i],
        first matches Some(row) ==> {
            &&& row < counts.len()
            &&& counts[row as int] > 0
            &&& !covered[row as int]
            &&& forall|i: int| 0 <= i < row && #[trigger] counts[i] > 0 ==> covered[i]
        },
))]
pub(crate) fn first_uncovered_row(covered: &[bool], counts: &[usize]) -> Option<usize> {
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            covered.len() == counts.len(),
            forall|i: int| 0 <= i < row && #[trigger] counts[i] > 0 ==> covered[i],
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] > 0 && !covered[row] {
            return Some(row);
        }
        row += 1;
    }
    None
}

/// Returns the first row with new tokens, or row zero when every row is inactive.
///
/// Proved: an active row with only inactive rows before it, or zero exactly
/// when no row is active.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(first =>
    ensures
        (exists|i: int| 0 <= i < counts.len() && #[trigger] counts[i] > 0) ==> {
            &&& first < counts.len()
            &&& counts[first as int] > 0
            &&& forall|i: int| 0 <= i < first ==> #[trigger] counts[i] == 0
        },
        (forall|i: int| 0 <= i < counts.len() ==> #[trigger] counts[i] == 0) ==> first == 0,
))]
pub(crate) fn first_active_row(counts: &[usize]) -> usize {
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            forall|i: int| 0 <= i < row ==> #[trigger] counts[i] == 0,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] > 0 {
            return row;
        }
        row += 1;
    }
    0
}

/// Lists the positions of every set flag in increasing order.
///
/// Proved: the list is strictly increasing and contains exactly the set
/// positions.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rows =>
    ensures
        forall|k: int| 0 <= k < rows.len() ==>
            #[trigger] rows[k] < flags.len() && flags[rows[k] as int],
        forall|j: int, k: int| 0 <= j < k < rows.len() ==> #[trigger] rows[j] < #[trigger] rows[k],
        forall|i: int| 0 <= i < flags.len() && #[trigger] flags[i] ==>
            exists|k: int| 0 <= k < rows.len() && #[trigger] rows[k] == i,
))]
pub(crate) fn set_positions(flags: &[bool]) -> Vec<usize> {
    let mut rows: Vec<usize> = Vec::new();
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= flags.len(),
            forall|k: int| 0 <= k < rows.len() ==>
                #[trigger] rows[k] < index && flags[rows[k] as int],
            forall|j: int, k: int| 0 <= j < k < rows.len() ==>
                #[trigger] rows[j] < #[trigger] rows[k],
            forall|i: int| 0 <= i < index && #[trigger] flags[i] ==>
                exists|k: int| 0 <= k < rows.len() && #[trigger] rows[k] == i,
        decreases flags.len() - index,
    ))]
    while index < flags.len() {
        if flags[index] {
            #[cfg(verus_keep_ghost)]
            proof! {
                let old_rows = rows@;
                let new_rows = old_rows.push(index);
                assert(new_rows[old_rows.len() as int] == index);
                assert forall|i: int| 0 <= i < index + 1 && #[trigger] flags[i] implies
                    exists|k: int| 0 <= k < new_rows.len() && #[trigger] new_rows[k] == i
                by {
                    if i == index {
                        assert(new_rows[old_rows.len() as int] == i);
                    } else {
                        let k = choose|k: int| 0 <= k < old_rows.len() && #[trigger] old_rows[k]
                            == i;
                        assert(new_rows[k] == old_rows[k]);
                    }
                }
            }
            rows.push(index);
        }
        index += 1;
    }
    rows
}

/// Returns whether every flag is set.
///
/// Proved: exactly when no flag is clear (true for an empty slice).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(all =>
    ensures
        all == forall|i: int| 0 <= i < flags.len() ==> #[trigger] flags[i],
))]
pub(crate) fn all_set(flags: &[bool]) -> bool {
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= flags.len(),
            forall|i: int| 0 <= i < index ==> #[trigger] flags[i],
        decreases flags.len() - index,
    ))]
    while index < flags.len() {
        if !flags[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// Returns the position in `rows` of the first row whose staged count differs from its proposal.
///
/// The caller passes rows inside both vectors (coverage validated them).
///
/// Proved: `None` exactly when every listed row staged its proposal length;
/// otherwise the lowest position whose row did not.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(first =>
    requires
        forall|k: int| 0 <= k < rows.len() ==>
            #[trigger] rows[k] < staged.len() && rows[k] < proposals.len(),
    ensures
        first is None <==> forall|k: int| 0 <= k < rows.len() ==>
            staged[#[trigger] rows[k] as int] == proposals[rows[k] as int],
        first matches Some(position) ==> {
            &&& position < rows.len()
            &&& staged[rows[position as int] as int] != proposals[rows[position as int] as int]
            &&& forall|k: int| 0 <= k < position ==>
                staged[#[trigger] rows[k] as int] == proposals[rows[k] as int]
        },
))]
pub(crate) fn first_unstaged_proposal(
    rows: &[usize],
    staged: &[usize],
    proposals: &[usize],
) -> Option<usize> {
    let mut position = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            position <= rows.len(),
            forall|k: int| 0 <= k < rows.len() ==>
                #[trigger] rows[k] < staged.len() && rows[k] < proposals.len(),
            forall|k: int| 0 <= k < position ==>
                staged[#[trigger] rows[k] as int] == proposals[rows[k] as int],
        decreases rows.len() - position,
    ))]
    while position < rows.len() {
        let row = rows[position];
        if staged[row] != proposals[row] {
            return Some(position);
        }
        position += 1;
    }
    None
}

/// Per-bank lifecycle versions that bind step write sets to one bank state.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BankVersions {
    /// Lifecycle version of the main bank.
    main: u64,
    /// Lifecycle version of the MTP bank.
    mtp: u64,
}

#[cfg(verus_keep_ghost)]
verus! {

impl BankVersions {
    /// The version of `bank`.
    pub closed spec fn version(&self, bank: StepBank) -> u64 {
        match bank {
            StepBank::Main => self.main,
            StepBank::Mtp => self.mtp,
        }
    }
}

} // verus!

/// Returns both banks at version zero.
///
/// Verified as a free function because Verus cannot attach a specification
/// to an associated function without a receiver.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(versions =>
    ensures
        versions.version(StepBank::Main) == 0,
        versions.version(StepBank::Mtp) == 0,
))]
pub(crate) const fn initial_versions() -> BankVersions {
    BankVersions { main: 0, mtp: 0 }
}

impl BankVersions {
    /// Advances one bank's version, invalidating every step stamped with the old one.
    ///
    /// Proved: the bank's version becomes its wrapping successor, which
    /// differs from the old version, and the other bank's version is
    /// unchanged.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(
        ensures
            final(self).version(bank) != old(self).version(bank),
            final(self).version(bank) as int == (old(self).version(bank) as int + 1)
                % 0x1_0000_0000_0000_0000,
            forall|other: StepBank| other != bank ==>
                #[trigger] final(self).version(other) == old(self).version(other),
    ))]
    pub(crate) const fn bump(&mut self, bank: StepBank) {
        match bank {
            StepBank::Main => self.main = next_version(self.main),
            StepBank::Mtp => self.mtp = next_version(self.mtp),
        }
    }

    /// Returns a bank's current version.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(version =>
        ensures
            version == self.version(bank),
    ))]
    pub(crate) const fn current(&self, bank: StepBank) -> u64 {
        match bank {
            StepBank::Main => self.main,
            StepBank::Mtp => self.mtp,
        }
    }
}

#[cfg(test)]
#[path = "coverage_tests.rs"]
mod tests;
