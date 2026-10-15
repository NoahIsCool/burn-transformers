//! Verified integer core for per-row logical lengths and speculative cursors.
//!
//! Every decode state tracks, for each batch row `i`, a *committed* absolute
//! length `C_i`. A speculative transaction adds a *working* cursor `W_i` that
//! includes tentative proposal tokens. Writing `K` for the logical capacity,
//! the ledger maintains the invariant
//!
//! ```text
//! C_i <= K                      (always)
//! C_i <= W_i <= K               (while a transaction is active)
//! ```
//!
//! and implements every lifecycle transition as a pure function from one
//! well-formed ledger to another:
//!
//! | Transition | Effect |
//! | --- | --- |
//! | advance by `n_i` | the active cursor becomes `cursor_i + n_i` (checked, `<= K`) |
//! | begin | `W := C` |
//! | commit `a_i` | requires `a_i <= W_i - C_i`; `C_i := C_i + a_i`, transaction ends |
//! | rollback | the transaction ends, `C` unchanged |
//! | truncate to `t_i` | requires `t_i <= C_i` and no transaction |
//! | reorder by `s_i` | `C_i := C_(s_i)`, every `s_i` in bounds, no transaction |
//! | reset | `C_i := 0`, no transaction |
//!
//! Because each transition returns a new value instead of mutating in place,
//! callers compose them atomically with tensor validation: compute the next
//! ledger, validate the physical caches, then publish both. A rejected
//! transition never changes the caller's state.
//!
//! # Verified behavior
//!
//! `verus/cache.rs` checks these functions in place (see `verus/README.md`).
//! For every admissible input the proofs establish that no machine arithmetic
//! overflows or underflows, that every successful transition preserves the
//! invariant above, that each result equals the documented formula, and that
//! each function rejects *exactly* the inputs violating its rule (soundness
//! and completeness). Every rejection also carries the proved diagnostic: the
//! first offending row with its exact operands, or the transaction state that
//! forbids the transition.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Rule broken by a rejected per-row length transition.
///
/// Callers map these integer-level violations to their own public error
/// types, so the same verified core can serve metadata-only and tensor-backed
/// decode states with different diagnostics.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LengthViolation {
    /// A per-row vector did not have exactly one entry per tracked row.
    CountMismatch {
        /// Number of tracked rows.
        expected: usize,
        /// Number of supplied values.
        actual: usize,
    },
    /// `length + additional` does not fit in `usize`.
    Overflow {
        /// Affected row.
        sequence: usize,
        /// Cursor before the advance.
        length: usize,
        /// Requested increment.
        additional: usize,
    },
    /// A requested cursor exceeds the logical capacity.
    CapacityExceeded {
        /// Affected row.
        sequence: usize,
        /// Requested cursor.
        requested: usize,
        /// Maximum logical length.
        capacity: usize,
    },
    /// A truncation asked to lengthen a row.
    Extension {
        /// Affected row.
        sequence: usize,
        /// Committed length.
        current: usize,
        /// Requested longer length.
        requested: usize,
    },
    /// A commit accepted more tokens than were proposed for a row.
    PrefixExceedsProposal {
        /// Affected row.
        sequence: usize,
        /// Requested accepted prefix.
        accepted: usize,
        /// Tokens proposed since the transaction began.
        proposed: usize,
    },
    /// A reorder named a source row outside the batch.
    SourceOutOfBounds {
        /// Output row whose source index is invalid.
        sequence: usize,
        /// Invalid source index.
        source: usize,
        /// Number of tracked rows.
        sequence_count: usize,
    },
    /// `begin` was called while a transaction is already active, or a
    /// reset/reorder/truncate was requested during a transaction.
    TransactionActive,
    /// `commit` or `rollback` was called without an active transaction.
    TransactionInactive,
}

#[cfg(verus_keep_ghost)]
verus! {

/// Every entry of `lengths` is at most `capacity`.
pub open spec fn within_capacity(lengths: Seq<usize>, capacity: usize) -> bool {
    forall|i: int| 0 <= i < lengths.len() ==> #[trigger] lengths[i] <= capacity
}

/// `low` and `high` have equal length and `low[i] <= high[i] <= capacity`.
pub open spec fn bracketed(low: Seq<usize>, high: Seq<usize>, capacity: usize) -> bool {
    &&& low.len() == high.len()
    &&& forall|i: int| 0 <= i < low.len() ==> #[trigger] low[i] <= high[i]
    &&& within_capacity(high, capacity)
}

} // verus!

/// Copies a length vector element by element.
///
/// Verified code cannot rely on `Vec::clone` for element equality, so the
/// ledger copies through this loop; vectors here hold one entry per batch row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(copy =>
    ensures
        copy@ == lengths@,
))]
fn copy_lengths(lengths: &[usize]) -> Vec<usize> {
    let mut copy: Vec<usize> = Vec::with_capacity(lengths.len());
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= lengths.len(),
            copy@ == lengths@.subrange(0, index as int),
        decreases lengths.len() - index,
    ))]
    while index < lengths.len() {
        copy.push(lengths[index]);
        index += 1;
    }
    copy
}

/// Returns `count` zeros, the cursor of an empty session.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(zeros =>
    ensures
        zeros.len() == count,
        forall|i: int| 0 <= i < count ==> #[trigger] zeros[i] == 0,
))]
fn zero_lengths(count: usize) -> Vec<usize> {
    let mut zeros: Vec<usize> = Vec::with_capacity(count);
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= count,
            zeros.len() == index,
            forall|i: int| 0 <= i < index ==> #[trigger] zeros[i] == 0,
        decreases count - index,
    ))]
    while index < count {
        zeros.push(0);
        index += 1;
    }
    zeros
}

/// Computes `lengths[i] + counts[i]` for every row, checking each against `capacity`.
///
/// This is the single advance rule shared by committed prefill/decode and by a
/// transaction's working cursor. Validation completes for every row before the
/// new vector is returned, so a ragged batch can never be partially advanced.
/// A rejection names the first row whose sum overflows or exceeds `capacity`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result.is_ok() == (counts.len() == lengths.len()
            && forall|i: int| 0 <= i < lengths.len() ==>
                (#[trigger] lengths[i]) + counts[i] <= capacity),
        result matches Ok(next) ==> {
            &&& next.len() == lengths.len()
            &&& forall|i: int| 0 <= i < lengths.len() ==>
                #[trigger] next[i] == lengths[i] + counts[i]
            &&& within_capacity(next@, capacity)
        },
        result matches Err(violation) ==> match violation {
            LengthViolation::CountMismatch { expected, actual } =>
                counts.len() != lengths.len() && expected == lengths.len()
                    && actual == counts.len(),
            LengthViolation::Overflow { sequence, length, additional } => {
                &&& counts.len() == lengths.len()
                &&& sequence < lengths.len()
                &&& length == lengths[sequence as int]
                &&& additional == counts[sequence as int]
                &&& length + additional > usize::MAX
                &&& forall|i: int| 0 <= i < sequence ==>
                    (#[trigger] lengths[i]) + counts[i] <= capacity
            },
            LengthViolation::CapacityExceeded { sequence, requested, capacity: limit } => {
                &&& counts.len() == lengths.len()
                &&& sequence < lengths.len()
                &&& requested == lengths[sequence as int] + counts[sequence as int]
                &&& requested > capacity
                &&& limit == capacity
                &&& forall|i: int| 0 <= i < sequence ==>
                    (#[trigger] lengths[i]) + counts[i] <= capacity
            },
            _ => false,
        },
))]
pub(crate) fn advanced_lengths(
    lengths: &[usize],
    counts: &[usize],
    capacity: usize,
) -> Result<Vec<usize>, LengthViolation> {
    if counts.len() != lengths.len() {
        return Err(LengthViolation::CountMismatch {
            expected: lengths.len(),
            actual: counts.len(),
        });
    }
    let mut next: Vec<usize> = Vec::with_capacity(lengths.len());
    let mut sequence = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            sequence <= lengths.len(),
            counts.len() == lengths.len(),
            next.len() == sequence,
            forall|i: int| 0 <= i < sequence ==>
                #[trigger] next[i] == lengths[i] + counts[i] && next[i] <= capacity,
            forall|i: int| 0 <= i < sequence ==> (#[trigger] lengths[i]) + counts[i] <= capacity,
        decreases lengths.len() - sequence,
    ))]
    while sequence < lengths.len() {
        let length = lengths[sequence];
        let additional = counts[sequence];
        let requested = match length.checked_add(additional) {
            Some(requested) => requested,
            None => {
                return Err(LengthViolation::Overflow {
                    sequence,
                    length,
                    additional,
                });
            }
        };
        if requested > capacity {
            return Err(LengthViolation::CapacityExceeded {
                sequence,
                requested,
                capacity,
            });
        }
        next.push(requested);
        sequence += 1;
    }
    Ok(next)
}

/// Checks that every requested length is no longer than its current length.
///
/// A rewind may only shorten rows; physical caches separately decide whether
/// the shortened prefix is still reconstructible (see `retention.rs`). A
/// rejection names the first row that would grow.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result.is_ok() == (requested.len() == current.len()
            && forall|i: int| 0 <= i < current.len() ==>
                #[trigger] requested[i] <= current[i]),
        result matches Err(violation) ==> match violation {
            LengthViolation::CountMismatch { expected, actual } =>
                requested.len() != current.len() && expected == current.len()
                    && actual == requested.len(),
            LengthViolation::Extension { sequence, current: from, requested: to } => {
                &&& requested.len() == current.len()
                &&& sequence < current.len()
                &&& from == current[sequence as int]
                &&& to == requested[sequence as int]
                &&& to > from
                &&& forall|i: int| 0 <= i < sequence ==> #[trigger] requested[i] <= current[i]
            },
            _ => false,
        },
))]
pub(crate) fn validate_rewind(
    current: &[usize],
    requested: &[usize],
) -> Result<(), LengthViolation> {
    if requested.len() != current.len() {
        return Err(LengthViolation::CountMismatch {
            expected: current.len(),
            actual: requested.len(),
        });
    }
    let mut sequence = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            sequence <= current.len(),
            requested.len() == current.len(),
            forall|i: int| 0 <= i < sequence ==> #[trigger] requested[i] <= current[i],
        decreases current.len() - sequence,
    ))]
    while sequence < current.len() {
        if requested[sequence] > current[sequence] {
            return Err(LengthViolation::Extension {
                sequence,
                current: current[sequence],
                requested: requested[sequence],
            });
        }
        sequence += 1;
    }
    Ok(())
}

/// Returns each row's tentative proposal length `working[i] - baseline[i]`.
///
/// The caller guarantees the ledger invariant `baseline[i] <= working[i]`, so
/// the subtraction never underflows.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(proposals =>
    requires
        baseline.len() == working.len(),
        forall|i: int| 0 <= i < baseline.len() ==> #[trigger] baseline[i] <= working[i],
    ensures
        proposals.len() == baseline.len(),
        forall|i: int| 0 <= i < baseline.len() ==>
            #[trigger] proposals[i] == working[i] - baseline[i],
))]
pub(crate) fn proposal_lengths(baseline: &[usize], working: &[usize]) -> Vec<usize> {
    let mut proposals: Vec<usize> = Vec::with_capacity(baseline.len());
    let mut sequence = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            sequence <= baseline.len(),
            baseline.len() == working.len(),
            forall|i: int| 0 <= i < baseline.len() ==> #[trigger] baseline[i] <= working[i],
            proposals.len() == sequence,
            forall|i: int| 0 <= i < sequence ==>
                #[trigger] proposals[i] == working[i] - baseline[i],
        decreases baseline.len() - sequence,
    ))]
    while sequence < baseline.len() {
        proposals.push(working[sequence] - baseline[sequence]);
        sequence += 1;
    }
    proposals
}

/// Computes `baseline[i] + accepted[i]` after checking every accepted prefix.
///
/// Accepting `a_i` requires `a_i <= working[i] - baseline[i]`; the result then
/// lies in `[baseline[i], working[i]]`, so it inherits the capacity bound of
/// the working cursor and the addition cannot overflow. A rejection names the
/// first row that accepts more than it proposed.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        baseline.len() == working.len(),
        forall|i: int| 0 <= i < baseline.len() ==> #[trigger] baseline[i] <= working[i],
    ensures
        result.is_ok() == (accepted.len() == baseline.len()
            && forall|i: int| 0 <= i < baseline.len() ==>
                #[trigger] accepted[i] <= working[i] - baseline[i]),
        result matches Ok(committed) ==> {
            &&& committed.len() == baseline.len()
            &&& forall|i: int| 0 <= i < baseline.len() ==> {
                &&& #[trigger] committed[i] == baseline[i] + accepted[i]
                &&& baseline[i] <= committed[i] <= working[i]
            }
        },
        result matches Err(violation) ==> match violation {
            LengthViolation::CountMismatch { expected, actual } =>
                accepted.len() != baseline.len() && expected == baseline.len()
                    && actual == accepted.len(),
            LengthViolation::PrefixExceedsProposal { sequence, accepted: taken, proposed } => {
                &&& accepted.len() == baseline.len()
                &&& sequence < baseline.len()
                &&& taken == accepted[sequence as int]
                &&& proposed == working[sequence as int] - baseline[sequence as int]
                &&& taken > proposed
                &&& forall|i: int| 0 <= i < sequence ==>
                    #[trigger] accepted[i] <= working[i] - baseline[i]
            },
            _ => false,
        },
))]
pub(crate) fn committed_prefix_lengths(
    baseline: &[usize],
    working: &[usize],
    accepted: &[usize],
) -> Result<Vec<usize>, LengthViolation> {
    if accepted.len() != baseline.len() {
        return Err(LengthViolation::CountMismatch {
            expected: baseline.len(),
            actual: accepted.len(),
        });
    }
    let mut committed: Vec<usize> = Vec::with_capacity(baseline.len());
    let mut sequence = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            sequence <= baseline.len(),
            baseline.len() == working.len(),
            accepted.len() == baseline.len(),
            forall|i: int| 0 <= i < baseline.len() ==> #[trigger] baseline[i] <= working[i],
            committed.len() == sequence,
            forall|i: int| 0 <= i < sequence ==>
                #[trigger] committed[i] == baseline[i] + accepted[i],
            forall|i: int| 0 <= i < sequence ==>
                #[trigger] accepted[i] <= working[i] - baseline[i],
        decreases baseline.len() - sequence,
    ))]
    while sequence < baseline.len() {
        let proposed = working[sequence] - baseline[sequence];
        if accepted[sequence] > proposed {
            return Err(LengthViolation::PrefixExceedsProposal {
                sequence,
                accepted: accepted[sequence],
                proposed,
            });
        }
        committed.push(baseline[sequence] + accepted[sequence]);
        sequence += 1;
    }
    Ok(committed)
}

/// Gathers `lengths[indices[i]]` for every output row.
///
/// Duplicate indices are valid and implement beam expansion; the number of
/// rows never changes, so tensor shapes stay fixed. A rejection names the
/// first output row whose source lies outside the batch.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result.is_ok() == (indices.len() == lengths.len()
            && forall|i: int| 0 <= i < indices.len() ==>
                #[trigger] indices[i] < lengths.len()),
        result matches Ok(reordered) ==> {
            &&& reordered.len() == lengths.len()
            &&& forall|i: int| 0 <= i < indices.len() ==>
                #[trigger] reordered[i] == lengths[indices[i] as int]
        },
        result matches Err(violation) ==> match violation {
            LengthViolation::CountMismatch { expected, actual } =>
                indices.len() != lengths.len() && expected == lengths.len()
                    && actual == indices.len(),
            LengthViolation::SourceOutOfBounds { sequence, source, sequence_count } => {
                &&& indices.len() == lengths.len()
                &&& sequence < indices.len()
                &&& source == indices[sequence as int]
                &&& source >= lengths.len()
                &&& sequence_count == lengths.len()
                &&& forall|i: int| 0 <= i < sequence ==> #[trigger] indices[i] < lengths.len()
            },
            _ => false,
        },
))]
pub(crate) fn reordered_lengths(
    lengths: &[usize],
    indices: &[usize],
) -> Result<Vec<usize>, LengthViolation> {
    if indices.len() != lengths.len() {
        return Err(LengthViolation::CountMismatch {
            expected: lengths.len(),
            actual: indices.len(),
        });
    }
    let mut reordered: Vec<usize> = Vec::with_capacity(lengths.len());
    let mut sequence = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            sequence <= indices.len(),
            indices.len() == lengths.len(),
            reordered.len() == sequence,
            forall|i: int| 0 <= i < sequence ==> {
                &&& #[trigger] indices[i] < lengths.len()
                &&& reordered[i] == lengths[indices[i] as int]
            },
        decreases indices.len() - sequence,
    ))]
    while sequence < indices.len() {
        let source = indices[sequence];
        if source >= lengths.len() {
            return Err(LengthViolation::SourceOutOfBounds {
                sequence,
                source,
                sequence_count: lengths.len(),
            });
        }
        reordered.push(lengths[source]);
        sequence += 1;
    }
    Ok(reordered)
}

/// Committed lengths plus an optional speculative working cursor.
///
/// The fields are private and every transition below returns a new ledger, so
/// the invariant `C_i <= W_i <= capacity` documented in the module header can
/// only be established by verified code.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct LengthLedger {
    /// Maximum logical length of every row.
    capacity: usize,
    /// Committed absolute lengths, one per row.
    committed: Vec<usize>,
    /// Working cursor including proposals while a transaction is active.
    working: Option<Vec<usize>>,
}

#[cfg(verus_keep_ghost)]
verus! {

impl LengthLedger {
    /// The ledger invariant: bounded committed lengths and a bracketed cursor.
    pub closed spec fn well_formed(&self) -> bool {
        &&& within_capacity(self.committed@, self.capacity)
        &&& match self.working {
            Some(working) => bracketed(self.committed@, working@, self.capacity),
            None => true,
        }
    }

    /// Committed lengths as a mathematical sequence.
    pub closed spec fn committed_view(&self) -> Seq<usize> {
        self.committed@
    }

    /// The cursor the next advance moves: working during a transaction.
    pub closed spec fn cursor_view(&self) -> Seq<usize> {
        match self.working {
            Some(working) => working@,
            None => self.committed@,
        }
    }

    /// Whether a speculative transaction is active.
    pub closed spec fn active(&self) -> bool {
        self.working is Some
    }

    /// The configured logical capacity.
    pub closed spec fn capacity_view(&self) -> usize {
        self.capacity
    }
}

} // verus!

/// Creates an empty ledger of `sequence_count` zero-length rows.
///
/// Verified as a free function: Verus cannot attach a specification to an
/// associated function without a receiver, so [`LengthLedger::new`] wraps it.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(ledger =>
    ensures
        ledger.well_formed(),
        !ledger.active(),
        ledger.capacity_view() == capacity,
        ledger.committed_view().len() == sequence_count,
        forall|i: int| 0 <= i < sequence_count ==> #[trigger] ledger.committed_view()[i] == 0,
))]
fn empty_ledger(capacity: usize, sequence_count: usize) -> LengthLedger {
    LengthLedger {
        capacity,
        committed: zero_lengths(sequence_count),
        working: None,
    }
}

/// Creates a ledger with explicit committed lengths, each at most `capacity`.
///
/// [`LengthLedger::from_lengths`] wraps this verified free function.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result.is_ok() == within_capacity(lengths@, capacity),
        result matches Ok(ledger) ==> {
            &&& ledger.well_formed()
            &&& !ledger.active()
            &&& ledger.capacity_view() == capacity
            &&& ledger.committed_view() == lengths@
        },
        result matches Err(violation) ==> match violation {
            LengthViolation::CapacityExceeded { sequence, requested, capacity: limit } => {
                &&& sequence < lengths.len()
                &&& requested == lengths[sequence as int]
                &&& requested > capacity
                &&& limit == capacity
                &&& forall|i: int| 0 <= i < sequence ==> #[trigger] lengths[i] <= capacity
            },
            _ => false,
        },
))]
fn ledger_from_lengths(
    capacity: usize,
    lengths: &[usize],
) -> Result<LengthLedger, LengthViolation> {
    let mut sequence = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            sequence <= lengths.len(),
            forall|i: int| 0 <= i < sequence ==> #[trigger] lengths[i] <= capacity,
        decreases lengths.len() - sequence,
    ))]
    while sequence < lengths.len() {
        if lengths[sequence] > capacity {
            return Err(LengthViolation::CapacityExceeded {
                sequence,
                requested: lengths[sequence],
                capacity,
            });
        }
        sequence += 1;
    }
    Ok(LengthLedger {
        capacity,
        committed: copy_lengths(lengths),
        working: None,
    })
}

impl LengthLedger {
    /// Creates an empty ledger of `sequence_count` zero-length rows.
    pub(crate) fn new(capacity: usize, sequence_count: usize) -> Self {
        empty_ledger(capacity, sequence_count)
    }

    /// Creates a ledger with explicit committed lengths, each at most `capacity`.
    pub(crate) fn from_lengths(
        capacity: usize,
        lengths: &[usize],
    ) -> Result<Self, LengthViolation> {
        ledger_from_lengths(capacity, lengths)
    }

    /// Advances the active cursor: the working cursor during a transaction,
    /// otherwise the committed lengths.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(result =>
        requires
            self.well_formed(),
        ensures
            result.is_ok() == (counts.len() == self.cursor_view().len()
                && forall|i: int| 0 <= i < counts.len() ==>
                    #[trigger] self.cursor_view()[i] + counts[i] <= self.capacity_view()),
            result matches Ok(next) ==> {
                &&& next.well_formed()
                &&& next.active() == self.active()
                &&& next.capacity_view() == self.capacity_view()
                &&& next.cursor_view().len() == self.cursor_view().len()
                &&& forall|i: int| 0 <= i < counts.len() ==>
                    #[trigger] next.cursor_view()[i] == self.cursor_view()[i] + counts[i]
                &&& self.active() ==> next.committed_view() == self.committed_view()
            },
    ))]
    pub(crate) fn advanced(&self, counts: &[usize]) -> Result<Self, LengthViolation> {
        match &self.working {
            Some(working) => {
                let next = advanced_lengths(working, counts, self.capacity)?;
                Ok(Self {
                    capacity: self.capacity,
                    committed: copy_lengths(&self.committed),
                    working: Some(next),
                })
            }
            None => {
                let next = advanced_lengths(&self.committed, counts, self.capacity)?;
                Ok(Self {
                    capacity: self.capacity,
                    committed: next,
                    working: None,
                })
            }
        }
    }

    /// Begins a transaction whose working cursor starts at the committed lengths.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(result =>
        requires
            self.well_formed(),
        ensures
            result.is_ok() == !self.active(),
            self.active() ==> result == Err::<Self, LengthViolation>(
                LengthViolation::TransactionActive),
            result matches Ok(next) ==> {
                &&& next.well_formed()
                &&& next.active()
                &&& next.capacity_view() == self.capacity_view()
                &&& next.committed_view() == self.committed_view()
                &&& next.cursor_view() == self.committed_view()
            },
    ))]
    pub(crate) fn begun(&self) -> Result<Self, LengthViolation> {
        if self.working.is_some() {
            return Err(LengthViolation::TransactionActive);
        }
        Ok(Self {
            capacity: self.capacity,
            committed: copy_lengths(&self.committed),
            working: Some(copy_lengths(&self.committed)),
        })
    }

    /// Commits an accepted prefix per row and ends the transaction.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(result =>
        requires
            self.well_formed(),
        ensures
            result.is_ok() == (self.active()
                && accepted.len() == self.committed_view().len()
                && forall|i: int| 0 <= i < accepted.len() ==>
                    #[trigger] accepted[i] <= self.cursor_view()[i] - self.committed_view()[i]),
            !self.active() ==> result == Err::<Self, LengthViolation>(
                LengthViolation::TransactionInactive),
            result matches Ok(next) ==> {
                &&& next.well_formed()
                &&& !next.active()
                &&& next.capacity_view() == self.capacity_view()
                &&& next.committed_view().len() == self.committed_view().len()
                &&& forall|i: int| 0 <= i < accepted.len() ==> {
                    &&& #[trigger] next.committed_view()[i]
                        == self.committed_view()[i] + accepted[i]
                    &&& next.committed_view()[i] <= self.cursor_view()[i]
                }
            },
    ))]
    pub(crate) fn committed_prefix(&self, accepted: &[usize]) -> Result<Self, LengthViolation> {
        let working = match &self.working {
            Some(working) => working,
            None => return Err(LengthViolation::TransactionInactive),
        };
        let committed = committed_prefix_lengths(&self.committed, working, accepted)?;
        Ok(Self {
            capacity: self.capacity,
            committed,
            working: None,
        })
    }

    /// Ends the transaction and discards every proposal.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(result =>
        requires
            self.well_formed(),
        ensures
            result.is_ok() == self.active(),
            !self.active() ==> result == Err::<Self, LengthViolation>(
                LengthViolation::TransactionInactive),
            result matches Ok(next) ==> {
                &&& next.well_formed()
                &&& !next.active()
                &&& next.capacity_view() == self.capacity_view()
                &&& next.committed_view() == self.committed_view()
            },
    ))]
    pub(crate) fn rolled_back(&self) -> Result<Self, LengthViolation> {
        if self.working.is_none() {
            return Err(LengthViolation::TransactionInactive);
        }
        Ok(Self {
            capacity: self.capacity,
            committed: copy_lengths(&self.committed),
            working: None,
        })
    }

    /// Shortens committed rows outside a transaction.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(result =>
        requires
            self.well_formed(),
        ensures
            result.is_ok() == (!self.active()
                && requested.len() == self.committed_view().len()
                && forall|i: int| 0 <= i < requested.len() ==>
                    #[trigger] requested[i] <= self.committed_view()[i]),
            self.active() ==> result == Err::<Self, LengthViolation>(
                LengthViolation::TransactionActive),
            result matches Ok(next) ==> {
                &&& next.well_formed()
                &&& !next.active()
                &&& next.capacity_view() == self.capacity_view()
                &&& next.committed_view() == requested@
            },
    ))]
    pub(crate) fn truncated(&self, requested: &[usize]) -> Result<Self, LengthViolation> {
        if self.working.is_some() {
            return Err(LengthViolation::TransactionActive);
        }
        validate_rewind(&self.committed, requested)?;
        Ok(Self {
            capacity: self.capacity,
            committed: copy_lengths(requested),
            working: None,
        })
    }

    /// Gathers committed rows by source index outside a transaction.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(result =>
        requires
            self.well_formed(),
        ensures
            result.is_ok() == (!self.active()
                && indices.len() == self.committed_view().len()
                && forall|i: int| 0 <= i < indices.len() ==>
                    #[trigger] indices[i] < self.committed_view().len()),
            self.active() ==> result == Err::<Self, LengthViolation>(
                LengthViolation::TransactionActive),
            result matches Ok(next) ==> {
                &&& next.well_formed()
                &&& !next.active()
                &&& next.capacity_view() == self.capacity_view()
                &&& next.committed_view().len() == self.committed_view().len()
                &&& forall|i: int| 0 <= i < indices.len() ==>
                    #[trigger] next.committed_view()[i]
                        == self.committed_view()[indices[i] as int]
            },
    ))]
    pub(crate) fn reordered(&self, indices: &[usize]) -> Result<Self, LengthViolation> {
        if self.working.is_some() {
            return Err(LengthViolation::TransactionActive);
        }
        let committed = reordered_lengths(&self.committed, indices)?;
        Ok(Self {
            capacity: self.capacity,
            committed,
            working: None,
        })
    }

    /// Returns every committed row to length zero outside a transaction.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(result =>
        requires
            self.well_formed(),
        ensures
            result.is_ok() == !self.active(),
            self.active() ==> result == Err::<Self, LengthViolation>(
                LengthViolation::TransactionActive),
            result matches Ok(next) ==> {
                &&& next.well_formed()
                &&& !next.active()
                &&& next.capacity_view() == self.capacity_view()
                &&& next.committed_view().len() == self.committed_view().len()
                &&& forall|i: int| 0 <= i < next.committed_view().len() ==>
                    #[trigger] next.committed_view()[i] == 0
            },
    ))]
    pub(crate) fn reset(&self) -> Result<Self, LengthViolation> {
        if self.working.is_some() {
            return Err(LengthViolation::TransactionActive);
        }
        Ok(Self {
            capacity: self.capacity,
            committed: zero_lengths(self.committed.len()),
            working: None,
        })
    }

    /// Borrows committed lengths, which only `committed_prefix` changes mid-transaction.
    pub(crate) fn committed(&self) -> &[usize] {
        &self.committed
    }

    /// Borrows the working cursor, equal to the committed lengths outside a transaction.
    pub(crate) fn working(&self) -> &[usize] {
        self.working.as_deref().unwrap_or(&self.committed)
    }

    /// Returns whether a speculative transaction is active.
    pub(crate) const fn is_active(&self) -> bool {
        self.working.is_some()
    }

    /// Returns each row's proposal length `working - committed` (all zero outside a transaction).
    ///
    /// Verified: entry `i` is exactly `cursor_i - committed_i`, which cannot
    /// underflow because the ledger invariant brackets the working cursor.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(proposals =>
        requires
            self.well_formed(),
        ensures
            proposals.len() == self.committed_view().len(),
            forall|i: int| 0 <= i < proposals.len() ==>
                #[trigger] proposals[i] == self.cursor_view()[i] - self.committed_view()[i],
            !self.active() ==> forall|i: int| 0 <= i < proposals.len() ==>
                #[trigger] proposals[i] == 0,
    ))]
    pub(crate) fn proposals(&self) -> Vec<usize> {
        match &self.working {
            Some(working) => proposal_lengths(&self.committed, working),
            None => proposal_lengths(&self.committed, &self.committed),
        }
    }
}

#[cfg(test)]
#[path = "lengths_tests.rs"]
mod tests;
