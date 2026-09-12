//! Atomic logical-length updates, invalid inputs, and sequence lifecycle boundaries.

use super::*;

/// Builds a three-row logical cache fixture with capacity eight.
fn config() -> CacheConfig {
    CacheConfig::new(3, 8).expect("valid config")
}

/// Constructs zero lengths and preserves explicitly initialized lengths.
#[test]
fn constructs_empty_and_initialized_states() {
    let empty = DecodeState::new(config());
    assert_eq!(empty.config(), config());
    assert_eq!(empty.lengths(), &[0, 0, 0]);
    assert_eq!(empty.remaining_capacity(1), Ok(8));

    let initialized =
        DecodeState::from_lengths(config(), [1, 4, 8].to_vec()).expect("valid initial lengths");
    assert_eq!(initialized.lengths(), &[1, 4, 8]);
    assert_eq!(initialized.remaining_capacity(2), Ok(0));
}

/// Rejects initial lengths with the wrong batch size or excess capacity.
#[test]
fn rejects_invalid_initial_lengths() {
    assert_eq!(
        DecodeState::from_lengths(config(), vec![1, 2]),
        Err(DecodeStateError::SequenceCountMismatch {
            expected: 3,
            actual: 2,
        })
    );
    assert_eq!(
        DecodeState::from_lengths(config(), vec![1, 9, 2]),
        Err(DecodeStateError::CapacityExceeded {
            sequence: 1,
            requested: 9,
            capacity: 8,
        })
    );
}

/// Reports absolute appended ranges while advancing independent sequences.
#[test]
fn advances_and_reports_absolute_ranges() {
    let mut state = DecodeState::from_lengths(config(), vec![2, 3, 4]).expect("valid state");
    let transition = state.advance(1, 3).expect("within capacity");
    assert_eq!(transition.sequence(), 1);
    assert_eq!(transition.previous(), 3);
    assert_eq!(transition.current(), 6);
    assert_eq!(transition.appended_range(), 3..6);
    assert_eq!(state.lengths(), &[2, 6, 4]);

    let no_op = state.advance(1, 0).expect("zero advance is valid");
    assert_eq!(no_op.appended_range(), 6..6);
}

/// Leaves every row unchanged when a batched append fails validation.
#[test]
fn advance_all_is_atomic_on_every_error() {
    let mut state = DecodeState::from_lengths(config(), vec![2, 7, 4]).expect("valid state");
    let original = state.lengths().to_vec();
    assert_eq!(
        state.advance_all(&[1, 2, 1]),
        Err(DecodeStateError::CapacityExceeded {
            sequence: 1,
            requested: 9,
            capacity: 8,
        })
    );
    assert_eq!(state.lengths(), original);

    assert_eq!(
        state.advance_all(&[1, 1]),
        Err(DecodeStateError::SequenceCountMismatch {
            expected: 3,
            actual: 2,
        })
    );
    assert_eq!(state.lengths(), original);
}

/// Distinguishes invalid row indices, capacity excess, and integer overflow.
#[test]
fn reports_index_and_arithmetic_failures() {
    let mut state = DecodeState::from_lengths(config(), vec![1, 2, 3]).expect("valid state");
    assert_eq!(
        state.length(3),
        Err(DecodeStateError::SequenceOutOfBounds {
            sequence: 3,
            sequence_count: 3,
        })
    );
    assert_eq!(
        state.advance(0, usize::MAX),
        Err(DecodeStateError::LengthOverflow {
            sequence: 0,
            length: 1,
            additional: usize::MAX,
        })
    );
    assert_eq!(state.lengths(), &[1, 2, 3]);
    assert_eq!(
        state.advance(2, 6),
        Err(DecodeStateError::CapacityExceeded {
            sequence: 2,
            requested: 9,
            capacity: 8,
        })
    );
    assert_eq!(state.lengths(), &[1, 2, 3]);
}

/// Truncates valid prefixes and rejects extensions before changing any row.
#[test]
fn truncates_atomically_and_never_extends() {
    let mut state = DecodeState::from_lengths(config(), vec![5, 6, 7]).expect("valid state");
    let transition = state.truncate(1, 2).expect("valid truncation");
    assert_eq!(transition.previous(), 6);
    assert_eq!(transition.current(), 2);
    assert_eq!(transition.appended_range(), 2..2);
    assert_eq!(
        state.truncate(1, 3),
        Err(DecodeStateError::CannotExtendByTruncation {
            sequence: 1,
            current: 2,
            requested: 3,
        })
    );

    let original = state.lengths().to_vec();
    assert_eq!(
        state.truncate_all(&[1, 3, 6]),
        Err(DecodeStateError::CannotExtendByTruncation {
            sequence: 1,
            current: 2,
            requested: 3,
        })
    );
    assert_eq!(state.lengths(), original);

    state
        .truncate_all(&[0, 1, 6])
        .expect("all lengths decrease");
    assert_eq!(state.lengths(), &[0, 1, 6]);
}

/// Resets all lengths and preserves repeated row selections during reordering.
#[test]
fn resets_and_reorders_with_duplicates() {
    let mut state = DecodeState::from_lengths(config(), vec![2, 5, 7]).expect("valid state");
    state.reorder(&[2, 2, 0]).expect("valid beam reorder");
    assert_eq!(state.lengths(), &[7, 7, 2]);

    let transitions = state.reset().expect("no transaction");
    assert_eq!(
        transitions
            .iter()
            .map(|transition| transition.previous())
            .collect::<Vec<_>>(),
        vec![7, 7, 2]
    );
    assert_eq!(state.lengths(), &[0, 0, 0]);
}

/// Rejects invalid reorder vectors without modifying the original lengths.
#[test]
fn reorder_validation_is_atomic() {
    let mut state = DecodeState::from_lengths(config(), vec![2, 5, 7]).expect("valid state");
    let original = state.lengths().to_vec();
    assert_eq!(
        state.reorder(&[0, 1]),
        Err(DecodeStateError::SequenceCountMismatch {
            expected: 3,
            actual: 2,
        })
    );
    assert_eq!(state.lengths(), original);

    assert_eq!(
        state.reorder(&[0, 3, 1]),
        Err(DecodeStateError::SequenceOutOfBounds {
            sequence: 3,
            sequence_count: 3,
        })
    );
    assert_eq!(state.lengths(), original);
}

/// Includes operation, index, and dimension context in logical-state diagnostics.
#[test]
fn display_messages_are_actionable() {
    let messages = [
        DecodeStateError::SequenceCountMismatch {
            expected: 2,
            actual: 1,
        }
        .to_string(),
        DecodeStateError::SequenceOutOfBounds {
            sequence: 4,
            sequence_count: 2,
        }
        .to_string(),
        DecodeStateError::LengthOverflow {
            sequence: 0,
            length: usize::MAX,
            additional: 1,
        }
        .to_string(),
        DecodeStateError::CannotExtendByTruncation {
            sequence: 1,
            current: 2,
            requested: 3,
        }
        .to_string(),
        DecodeStateError::TransactionActive { operation: "reset" }.to_string(),
    ];
    assert_eq!(messages[0], "expected 2 per-sequence values, received 1");
    assert!(messages[1].contains("sequence index 4"));
    assert!(messages[2].contains("overflows when advanced"));
    assert!(messages[3].contains("cannot truncate sequence 1"));
    assert_eq!(
        messages[4],
        "cannot reset while a cache transaction is active"
    );
    assert_eq!(
        DecodeStateError::CapacityExceeded {
            sequence: 2,
            requested: 9,
            capacity: 8,
        }
        .to_string(),
        "sequence 2 requested length 9 exceeds capacity 8"
    );

    let transaction = DecodeStateError::from(TransactionError::NoActiveTransaction);
    assert_eq!(transaction.to_string(), "no cache transaction is active");
    assert!(std::error::Error::source(&transaction).is_some());
    assert!(
        std::error::Error::source(&DecodeStateError::CapacityExceeded {
            sequence: 0,
            requested: 2,
            capacity: 1,
        })
        .is_none()
    );
}
