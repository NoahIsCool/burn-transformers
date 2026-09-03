use core::fmt;
use std::sync::Arc;

use super::{DecodeState, DecodeStateError};

/// A single-use logical checkpoint for speculative cache writes.
///
/// The mark belongs to the [`DecodeState`] transaction that created it. A
/// cloned or otherwise stale mark is rejected after the original transaction
/// commits or rolls back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheMark {
    token: TransactionToken,
    baseline_lengths: Vec<usize>,
}

impl CacheMark {
    /// Returns the sequence lengths captured before speculative appends.
    #[must_use]
    pub fn baseline_lengths(&self) -> &[usize] {
        &self.baseline_lengths
    }
}

#[derive(Clone)]
struct TransactionToken(Arc<()>);

impl TransactionToken {
    fn new() -> Self {
        Self(Arc::new(()))
    }
}

impl fmt::Debug for TransactionToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("TransactionToken(..)")
    }
}

impl PartialEq for TransactionToken {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for TransactionToken {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveTransaction {
    token: TransactionToken,
    pub(crate) baseline_lengths: Vec<usize>,
}

/// Transactional lifecycle operations required by speculative decoding.
///
/// A transaction consists only of appends after [`mark`](Self::mark). Calling
/// [`commit_prefix`](Self::commit_prefix) retains the first `accepted_tokens`
/// appended to every sequence and logically drops the remaining suffix.
/// [`rollback`](Self::rollback) drops every append since the mark.
pub trait TransactionalCache {
    /// Opaque mark used to finish a transaction.
    type Mark;
    /// Error returned for invalid transaction transitions.
    type Error;

    /// Begins a transaction at the current logical lengths.
    fn mark(&mut self) -> Result<Self::Mark, Self::Error>;

    /// Keeps a uniform prefix of tokens appended after `mark`.
    fn commit_prefix(
        &mut self,
        mark: Self::Mark,
        accepted_tokens: usize,
    ) -> Result<(), Self::Error>;

    /// Restores all logical lengths captured by `mark`.
    fn rollback(&mut self, mark: Self::Mark) -> Result<(), Self::Error>;
}

impl DecodeState {
    /// Commits a possibly ragged accepted prefix for each sequence.
    ///
    /// Each accepted count must be no greater than the number of tokens
    /// appended to that sequence since `mark`. Validation is atomic: a bad row
    /// leaves the transaction active and all current lengths unchanged, so the
    /// caller may still roll it back.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale mark, a count mismatch, or an accepted
    /// prefix longer than the corresponding speculative suffix.
    pub fn commit_prefix_per_sequence(
        &mut self,
        mark: CacheMark,
        accepted_tokens: &[usize],
    ) -> Result<(), DecodeStateError> {
        let active = self.validate_mark(&mark)?;
        if accepted_tokens.len() != self.lengths.len() {
            return Err(DecodeStateError::Transaction(
                TransactionError::SequenceCountMismatch {
                    expected: self.lengths.len(),
                    actual: accepted_tokens.len(),
                },
            ));
        }

        let mut committed = Vec::with_capacity(self.lengths.len());
        for (sequence, ((&baseline, &current), &accepted)) in active
            .baseline_lengths
            .iter()
            .zip(&self.lengths)
            .zip(accepted_tokens)
            .enumerate()
        {
            let appended = current - baseline;
            if accepted > appended {
                return Err(DecodeStateError::Transaction(
                    TransactionError::PrefixExceedsAppended {
                        sequence,
                        accepted,
                        appended,
                    },
                ));
            }
            committed.push(baseline + accepted);
        }

        self.lengths = committed;
        self.active_transaction = None;
        Ok(())
    }

    fn validate_mark(&self, mark: &CacheMark) -> Result<&ActiveTransaction, DecodeStateError> {
        let active = self
            .active_transaction
            .as_ref()
            .ok_or(DecodeStateError::Transaction(
                TransactionError::NoActiveTransaction,
            ))?;
        if active.token != mark.token || active.baseline_lengths != mark.baseline_lengths {
            return Err(DecodeStateError::Transaction(
                TransactionError::StaleOrForeignMark,
            ));
        }
        Ok(active)
    }
}

impl TransactionalCache for DecodeState {
    type Mark = CacheMark;
    type Error = DecodeStateError;

    fn mark(&mut self) -> Result<Self::Mark, Self::Error> {
        if self.active_transaction.is_some() {
            return Err(DecodeStateError::Transaction(
                TransactionError::AlreadyActive,
            ));
        }

        let token = TransactionToken::new();
        let baseline_lengths = self.lengths.clone();
        self.active_transaction = Some(ActiveTransaction {
            token: token.clone(),
            baseline_lengths: baseline_lengths.clone(),
        });

        Ok(CacheMark {
            token,
            baseline_lengths,
        })
    }

    fn commit_prefix(
        &mut self,
        mark: Self::Mark,
        accepted_tokens: usize,
    ) -> Result<(), Self::Error> {
        let counts = vec![accepted_tokens; self.lengths.len()];
        self.commit_prefix_per_sequence(mark, &counts)
    }

    fn rollback(&mut self, mark: Self::Mark) -> Result<(), Self::Error> {
        let active = self.validate_mark(&mark)?;
        let baseline_lengths = active.baseline_lengths.clone();
        self.lengths = baseline_lengths;
        self.active_transaction = None;
        Ok(())
    }
}

/// Invalid transition in a [`TransactionalCache`] lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionError {
    /// A second mark was requested before finishing the active transaction.
    AlreadyActive,
    /// Commit or rollback was requested without an active transaction.
    NoActiveTransaction,
    /// The supplied mark does not identify the active transaction.
    StaleOrForeignMark,
    /// Ragged commit counts did not match the tracked sequence count.
    SequenceCountMismatch {
        /// Expected number of accepted-prefix counts.
        expected: usize,
        /// Supplied number of counts.
        actual: usize,
    },
    /// A commit attempted to retain more tokens than were appended.
    PrefixExceedsAppended {
        /// Affected sequence index.
        sequence: usize,
        /// Requested accepted-prefix length.
        accepted: usize,
        /// Number of tokens appended since the mark.
        appended: usize,
    },
}

impl fmt::Display for TransactionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyActive => formatter.write_str("a cache transaction is already active"),
            Self::NoActiveTransaction => formatter.write_str("no cache transaction is active"),
            Self::StaleOrForeignMark => {
                formatter.write_str("cache mark is stale or belongs to another transaction")
            }
            Self::SequenceCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} accepted-prefix counts, received {actual}"
            ),
            Self::PrefixExceedsAppended {
                sequence,
                accepted,
                appended,
            } => write!(
                formatter,
                "sequence {sequence} cannot accept {accepted} tokens from an appended suffix of {appended}"
            ),
        }
    }
}

impl std::error::Error for TransactionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{CacheConfig, DecodeStateError};

    fn state(lengths: [usize; 2]) -> DecodeState {
        let config = CacheConfig::new(2, 32).expect("valid config");
        DecodeState::from_lengths(config, lengths.to_vec()).expect("valid state")
    }

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

    #[test]
    fn invalid_commit_is_atomic_and_can_be_rolled_back() {
        let mut state = state([3, 7]);
        let mark = state.mark().expect("starts transaction");
        let rollback_mark = mark.clone();
        state.advance_all(&[2, 4]).expect("speculative append");
        let speculative = state.lengths().to_vec();

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
        assert_eq!(state.lengths(), speculative);
        assert!(state.has_active_transaction());
        state
            .rollback(rollback_mark)
            .expect("transaction remains recoverable");
        assert_eq!(state.lengths(), &[3, 7]);
    }

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
}
