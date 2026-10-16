//! Length-transition rules and the transaction ledger against a naive model.

use proptest::prelude::*;

use super::*;

/// Advances every row, reports the first failing row, and accepts the exact capacity.
#[test]
fn advanced_lengths_checks_count_overflow_and_capacity_per_row() {
    assert_eq!(advanced_lengths(&[1, 2], &[3, 0], 4), Ok(vec![4, 2]));
    assert_eq!(advanced_lengths(&[], &[], 0), Ok(Vec::new()));
    assert_eq!(
        advanced_lengths(&[1, 2], &[1], 8),
        Err(LengthViolation::CountMismatch {
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        advanced_lengths(&[1, 2, 3], &[0, 7, usize::MAX], 8),
        Err(LengthViolation::CapacityExceeded {
            sequence: 1,
            requested: 9,
            capacity: 8
        })
    );
    assert_eq!(
        advanced_lengths(&[1, 2], &[usize::MAX, 0], usize::MAX),
        Err(LengthViolation::Overflow {
            sequence: 0,
            length: 1,
            additional: usize::MAX
        })
    );
}

/// Accepts equal or shorter rows and names the first lengthened row.
#[test]
fn validate_rewind_rejects_extension_and_count_mismatch() {
    assert_eq!(validate_rewind(&[3, 4], &[3, 0]), Ok(()));
    assert_eq!(
        validate_rewind(&[3, 4], &[4, 5]),
        Err(LengthViolation::Extension {
            sequence: 0,
            current: 3,
            requested: 4
        })
    );
    assert_eq!(
        validate_rewind(&[3, 4], &[3]),
        Err(LengthViolation::CountMismatch {
            expected: 2,
            actual: 1
        })
    );
}

/// Commits ragged prefixes inside each row's proposal and rejects over-acceptance.
#[test]
fn committed_prefix_lengths_bounds_every_row_by_its_proposal() {
    assert_eq!(proposal_lengths(&[2, 5], &[4, 5]), vec![2, 0]);
    assert_eq!(
        committed_prefix_lengths(&[2, 5], &[4, 5], &[1, 0]),
        Ok(vec![3, 5])
    );
    assert_eq!(
        committed_prefix_lengths(&[2, 5], &[4, 5], &[2, 1]),
        Err(LengthViolation::PrefixExceedsProposal {
            sequence: 1,
            accepted: 1,
            proposed: 0
        })
    );
    assert_eq!(
        committed_prefix_lengths(&[2, 5], &[4, 5], &[0, 0, 0]),
        Err(LengthViolation::CountMismatch {
            expected: 2,
            actual: 3
        })
    );
}

/// Gathers duplicated source rows and rejects out-of-range sources.
#[test]
fn reordered_lengths_supports_duplicates_and_rejects_bad_sources() {
    assert_eq!(reordered_lengths(&[2, 5, 7], &[2, 2, 0]), Ok(vec![7, 7, 2]));
    assert_eq!(
        reordered_lengths(&[2, 5, 7], &[0, 3, 1]),
        Err(LengthViolation::SourceOutOfBounds {
            sequence: 1,
            source: 3,
            sequence_count: 3
        })
    );
    assert_eq!(
        reordered_lengths(&[2, 5, 7], &[0]),
        Err(LengthViolation::CountMismatch {
            expected: 3,
            actual: 1
        })
    );
}

/// Runs one transaction through advance, commit, and the blocked lifecycle calls.
#[test]
fn ledger_transaction_moves_only_the_working_cursor_until_commit() {
    let ledger = LengthLedger::from_lengths(8, &[2, 5]).expect("valid lengths");
    assert_eq!(
        LengthLedger::from_lengths(4, &[2, 5]),
        Err(LengthViolation::CapacityExceeded {
            sequence: 1,
            requested: 5,
            capacity: 4
        })
    );
    assert_eq!(
        ledger.committed_prefix(&[0, 0]),
        Err(LengthViolation::TransactionInactive)
    );
    assert_eq!(
        ledger.rolled_back(),
        Err(LengthViolation::TransactionInactive)
    );

    let active = ledger.begun().expect("begin");
    assert!(active.is_active());
    assert_eq!(active.begun(), Err(LengthViolation::TransactionActive));
    for blocked in [
        active.truncated(&[0, 0]),
        active.reordered(&[1, 0]),
        active.reset(),
    ] {
        assert_eq!(blocked, Err(LengthViolation::TransactionActive));
    }

    let drafted = active.advanced(&[3, 1]).expect("draft");
    assert_eq!(drafted.committed(), &[2, 5]);
    assert_eq!(drafted.working(), &[5, 6]);
    assert_eq!(drafted.proposals(), vec![3, 1]);
    assert_eq!(
        drafted.advanced(&[0, 3]),
        Err(LengthViolation::CapacityExceeded {
            sequence: 1,
            requested: 9,
            capacity: 8
        })
    );

    let committed = drafted.committed_prefix(&[2, 0]).expect("commit");
    assert!(!committed.is_active());
    assert_eq!(committed.committed(), &[4, 5]);
    assert_eq!(committed.working(), &[4, 5]);

    let rolled_back = drafted.rolled_back().expect("rollback");
    assert_eq!(rolled_back.committed(), &[2, 5]);
    assert_eq!(rolled_back.proposals(), vec![0, 0]);
}

/// Truncates, reorders, and resets outside a transaction.
#[test]
fn ledger_topology_transitions_outside_a_transaction() {
    let ledger = LengthLedger::new(8, 3)
        .advanced(&[2, 5, 7])
        .expect("prefill");
    assert_eq!(ledger.committed(), &[2, 5, 7]);
    assert_eq!(
        ledger.truncated(&[2, 6, 7]),
        Err(LengthViolation::Extension {
            sequence: 1,
            current: 5,
            requested: 6
        })
    );
    assert_eq!(
        ledger.truncated(&[0, 5, 1]).expect("rewind").committed(),
        &[0, 5, 1]
    );
    assert_eq!(
        ledger.reordered(&[2, 2, 0]).expect("beam").committed(),
        &[7, 7, 2]
    );
    assert_eq!(ledger.reset().expect("reset").committed(), &[0, 0, 0]);
}

/// One lifecycle operation in the model-based property test.
#[derive(Debug, Clone)]
enum Operation {
    /// Advance the active cursor by per-row counts.
    Advance(Vec<usize>),
    /// Begin a transaction.
    Begin,
    /// Commit per-row accepted prefixes.
    Commit(Vec<usize>),
    /// Discard the transaction.
    Rollback,
    /// Rewind committed rows.
    Truncate(Vec<usize>),
    /// Gather committed rows by source index.
    Reorder(Vec<usize>),
    /// Return every row to zero.
    Reset,
}

/// Straightforward reference: committed lengths plus an optional working copy.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Model {
    /// Maximum logical length.
    capacity: usize,
    /// Committed lengths.
    committed: Vec<usize>,
    /// Working cursor while a transaction is active.
    working: Option<Vec<usize>>,
}

impl Model {
    /// Applies one operation, returning `false` (and leaving the model unchanged) on rejection.
    fn apply(&mut self, operation: &Operation) -> bool {
        let rows = self.committed.len();
        match operation {
            Operation::Advance(counts) => {
                let cursor = self.working.as_ref().unwrap_or(&self.committed);
                if counts.len() != rows {
                    return false;
                }
                let next: Option<Vec<usize>> = cursor
                    .iter()
                    .zip(counts)
                    .map(|(&length, &count)| {
                        length.checked_add(count).filter(|&n| n <= self.capacity)
                    })
                    .collect();
                let Some(next) = next else { return false };
                match &mut self.working {
                    Some(working) => *working = next,
                    None => self.committed = next,
                }
                true
            }
            Operation::Begin => {
                if self.working.is_some() {
                    return false;
                }
                self.working = Some(self.committed.clone());
                true
            }
            Operation::Commit(accepted) => {
                let Some(working) = &self.working else {
                    return false;
                };
                let valid = accepted.len() == rows
                    && (0..rows).all(|i| accepted[i] <= working[i] - self.committed[i]);
                if valid {
                    self.committed = (0..rows).map(|i| self.committed[i] + accepted[i]).collect();
                    self.working = None;
                }
                valid
            }
            Operation::Rollback => self.working.take().is_some(),
            Operation::Truncate(requested) => {
                let valid = self.working.is_none()
                    && requested.len() == rows
                    && (0..rows).all(|i| requested[i] <= self.committed[i]);
                if valid {
                    self.committed.clone_from(requested);
                }
                valid
            }
            Operation::Reorder(indices) => {
                let valid = self.working.is_none()
                    && indices.len() == rows
                    && indices.iter().all(|&index| index < rows);
                if valid {
                    self.committed = indices.iter().map(|&i| self.committed[i]).collect();
                }
                valid
            }
            Operation::Reset => {
                let valid = self.working.is_none();
                if valid {
                    self.committed.fill(0);
                }
                valid
            }
        }
    }
}

/// Applies one operation to the verified ledger.
fn apply(ledger: &LengthLedger, operation: &Operation) -> Result<LengthLedger, LengthViolation> {
    match operation {
        Operation::Advance(counts) => ledger.advanced(counts),
        Operation::Begin => ledger.begun(),
        Operation::Commit(accepted) => ledger.committed_prefix(accepted),
        Operation::Rollback => ledger.rolled_back(),
        Operation::Truncate(requested) => ledger.truncated(requested),
        Operation::Reorder(indices) => ledger.reordered(indices),
        Operation::Reset => ledger.reset(),
    }
}

/// Generates a per-row vector that is occasionally the wrong length.
fn row_vector(rows: usize, maximum: usize) -> impl Strategy<Value = Vec<usize>> {
    prop_oneof![
        8 => prop::collection::vec(0..=maximum, rows),
        1 => prop::collection::vec(0..=maximum, 0..=rows + 1),
    ]
}

/// Generates operations whose values straddle every validity boundary.
fn operation(rows: usize) -> impl Strategy<Value = Operation> {
    prop_oneof![
        4 => row_vector(rows, 4).prop_map(Operation::Advance),
        2 => Just(Operation::Begin),
        2 => row_vector(rows, 3).prop_map(Operation::Commit),
        1 => Just(Operation::Rollback),
        1 => row_vector(rows, 8).prop_map(Operation::Truncate),
        1 => row_vector(rows, rows).prop_map(Operation::Reorder),
        1 => Just(Operation::Reset),
    ]
}

proptest! {
    /// Random lifecycles agree with the naive model and keep `C <= W <= capacity`.
    #[test]
    fn ledger_matches_naive_model_on_random_lifecycles(
        (rows, operations) in (1_usize..4).prop_flat_map(|rows| {
            (Just(rows), prop::collection::vec(operation(rows), 0..48))
        }),
        capacity in 1_usize..10,
    ) {
        let mut ledger = LengthLedger::new(capacity, rows);
        let mut model = Model { capacity, committed: vec![0; rows], working: None };
        for operation in &operations {
            let accepted = model.apply(operation);
            match apply(&ledger, operation) {
                Ok(next) => {
                    prop_assert!(accepted, "ledger accepted {operation:?} that the model rejected");
                    ledger = next;
                }
                Err(_) => prop_assert!(!accepted, "ledger rejected valid {operation:?}"),
            }
            prop_assert_eq!(ledger.committed(), model.committed.as_slice());
            prop_assert_eq!(ledger.is_active(), model.working.is_some());
            let working = model.working.as_ref().unwrap_or(&model.committed);
            prop_assert_eq!(ledger.working(), working.as_slice());
            for (&committed, &working) in ledger.committed().iter().zip(ledger.working()) {
                prop_assert!(committed <= working && working <= capacity);
            }
            for (row, (&committed, &working)) in model.committed.iter().zip(working).enumerate() {
                prop_assert_eq!(ledger.remaining(row), capacity - working);
                prop_assert_eq!(ledger.proposal(row), working - committed);
            }
        }
    }
}
