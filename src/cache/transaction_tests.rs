//! Single-use transaction marks, accepted prefixes, rollback, and atomic rejection.

use super::*;
use crate::cache::{CacheConfig, DecodeStateError};

/// Builds a two-row transaction fixture from explicit baseline lengths.
fn state(lengths: [usize; 2]) -> SequenceLengths {
    let config = CacheConfig::new(2, 32).expect("valid config");
    SequenceLengths::from_lengths(config, lengths.to_vec()).expect("valid state")
}

/// Restores every baseline length after speculative appends are rolled back.
#[test]
fn rollback_restores_exact_baseline() {
    let mut state = state([3, 7]);
    let mark = state.mark().expect("starts transaction");
    assert_eq!(mark.baseline_lengths(), &[3, 7]);
    state.advance_all(&[4, 2]).expect("speculative append");
    state.rollback(mark).expect("valid rollback");
    assert_eq!(state.lengths(), &[3, 7]);
    assert!(!state.has_active_transaction());
}

/// Keeps only the same accepted proposal prefix for each row.
#[test]
fn commits_uniform_prefix_and_drops_suffix() {
    for accepted in 0..=4 {
        let mut state = state([3, 7]);
        let mark = state.mark().expect("starts transaction");
        state.advance_all(&[4, 4]).expect("speculative append");
        state
            .commit_prefix(mark, accepted)
            .expect("prefix is within suffix");
        assert_eq!(state.lengths(), &[3 + accepted, 7 + accepted]);
        assert!(!state.has_active_transaction());
    }
}

/// Commits independent accepted-prefix lengths for different rows.
#[test]
fn commits_ragged_prefixes() {
    let mut state = state([3, 7]);
    let mark = state.mark().expect("starts transaction");
    state.advance_all(&[5, 2]).expect("ragged append");
    state
        .commit_prefix_per_sequence(mark, &[4, 0])
        .expect("valid ragged prefixes");
    assert_eq!(state.lengths(), &[7, 7]);
}

/// Rejects uniform acceptance when even one row proposed fewer tokens.
#[test]
fn uniform_commit_rejects_a_prefix_too_long_for_any_row() {
    let mut state = state([3, 7]);
    let mark = state.mark().expect("starts transaction");
    let rollback_mark = mark.clone();
    state.advance_all(&[2, 4]).expect("ragged append");
    assert_eq!(
        state.commit_prefix(mark, 3),
        Err(DecodeStateError::Transaction(
            TransactionError::PrefixExceedsAppended {
                sequence: 0,
                accepted: 3,
                appended: 2,
            }
        ))
    );
    state.rollback(rollback_mark).expect("still recoverable");
}

/// Leaves an invalid commit untouched so its original mark can still roll back.
#[test]
fn invalid_commit_is_atomic_and_can_be_rolled_back() {
    let mut state = state([3, 7]);
    let mark = state.mark().expect("starts transaction");
    let rollback_mark = mark.clone();
    state.advance_all(&[2, 4]).expect("speculative append");
    let speculative = state.working_lengths().to_vec();

    assert_eq!(
        state.commit_prefix_per_sequence(mark, &[1, 5]),
        Err(DecodeStateError::Transaction(
            TransactionError::PrefixExceedsAppended {
                sequence: 1,
                accepted: 5,
                appended: 4,
            }
        ))
    );
    assert_eq!(state.working_lengths(), speculative);
    assert_eq!(state.lengths(), &[3, 7]);
    assert!(state.has_active_transaction());
    state
        .rollback(rollback_mark)
        .expect("transaction remains recoverable");
    assert_eq!(state.lengths(), &[3, 7]);
}

/// Rejects nested transactions and marks from completed or unrelated transactions.
#[test]
fn rejects_nested_stale_and_foreign_marks() {
    let mut first = state([1, 2]);
    let mut second = state([1, 2]);
    let first_mark = first.mark().expect("starts transaction");
    assert_eq!(
        first.mark(),
        Err(DecodeStateError::Transaction(
            TransactionError::AlreadyActive
        ))
    );

    let second_mark = second.mark().expect("starts transaction");
    assert_eq!(
        first.rollback(second_mark),
        Err(DecodeStateError::Transaction(
            TransactionError::StaleOrForeignMark
        ))
    );
    first.rollback(first_mark.clone()).expect("correct mark");
    assert_eq!(
        first.rollback(first_mark),
        Err(DecodeStateError::Transaction(
            TransactionError::NoActiveTransaction
        ))
    );
}

/// Forbids reset, reorder, and truncate while a transaction is active.
#[test]
fn blocks_topology_changes_during_transaction() {
    let mut state = state([3, 7]);
    let mark = state.mark().expect("starts transaction");
    assert_eq!(
        state.reset(),
        Err(DecodeStateError::TransactionActive { operation: "reset" })
    );
    assert_eq!(
        state.truncate(0, 1),
        Err(DecodeStateError::TransactionActive {
            operation: "truncate"
        })
    );
    assert_eq!(
        state.reorder(&[1, 0]),
        Err(DecodeStateError::TransactionActive {
            operation: "reorder"
        })
    );
    state.rollback(mark).expect("valid rollback");
}

/// Rejects an accepted-prefix vector with the wrong batch size.
#[test]
fn rejects_wrong_ragged_prefix_count() {
    let mut state = state([0, 0]);
    let mark = state.mark().expect("starts transaction");
    assert_eq!(
        state.commit_prefix_per_sequence(mark, &[0]),
        Err(DecodeStateError::Transaction(
            TransactionError::SequenceCountMismatch {
                expected: 2,
                actual: 1,
            }
        ))
    );
    assert!(state.has_active_transaction());
}

/// Includes the failed lifecycle condition in transaction diagnostics.
#[test]
fn transaction_errors_have_actionable_messages() {
    let messages = [
        TransactionError::AlreadyActive.to_string(),
        TransactionError::NoActiveTransaction.to_string(),
        TransactionError::StaleOrForeignMark.to_string(),
        TransactionError::SequenceCountMismatch {
            expected: 2,
            actual: 1,
        }
        .to_string(),
        TransactionError::PrefixExceedsAppended {
            sequence: 1,
            accepted: 4,
            appended: 2,
        }
        .to_string(),
    ];
    assert_eq!(messages[0], "a cache transaction is already active");
    assert_eq!(messages[1], "no cache transaction is active");
    assert!(messages[2].contains("stale"));
    assert!(messages[3].contains("expected 2"));
    assert!(messages[4].contains("cannot accept 4"));

    let mut state = state([0, 0]);
    let mark = state.mark().expect("starts transaction");
    assert!(format!("{state:?}").contains("TransactionToken"));
    state.rollback(mark).expect("valid rollback");
}
