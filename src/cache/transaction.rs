//! Single-use transactions for speculative logical cache lengths.
//!
//! A mark captures baseline lengths `L_b`; after proposing `P_b` tokens, accepting
//! `A_b` requires `0 <= A_b <= P_b` and commits `L_b + A_b`. Validation finishes
//! for every row before metadata changes. Rollback restores every `L_b`.
//!
//! Allocation identity distinguishes active, stale, and foreign marks, including
//! cloned handles. These integer transactions do not journal tensor values: ring
//! and recurrent caches must stage their proposed data outside committed storage.

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
    /// Allocates a unique transaction identity whose cloned handles still identify the same mark.
    fn new() -> Self {
        Self(Arc::new(()))
    }
}

impl fmt::Debug for TransactionToken {
    /// Formats the token without exposing its process-specific allocation address.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("TransactionToken(..)")
    }
}

impl PartialEq for TransactionToken {
    /// Compares allocation identity so unrelated transactions cannot share a valid mark.
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
/// appended to every sequence and drops the remaining suffix.
/// [`rollback`](Self::rollback) restores the marked state.
///
/// Implementations owning physical storage must make those guarantees for both
/// metadata and tensor values. In particular, a sliding ring must stage or
/// journal proposals instead of overwriting committed slots. The pure
/// [`DecodeState`] implementation tracks metadata only; tensor-backed callers
/// can pair it with [`super::StagedRingTransactionPlan`].
pub trait TransactionalCache {
    /// Opaque mark used to finish a transaction.
    type Mark;
    /// Error returned for invalid transaction transitions.
    type Error;

    /// Begins a transaction at the current logical lengths.
    fn mark(&mut self) -> Result<Self::Mark, Self::Error>;

    /// Keeps a uniform prefix of tokens appended after `mark`.
    ///
    /// Physical-cache implementations must also discard or restore every value
    /// belonging to the rejected suffix.
    fn commit_prefix(
        &mut self,
        mark: Self::Mark,
        accepted_tokens: usize,
    ) -> Result<(), Self::Error>;

    /// Restores all metadata and physical values captured by `mark`.
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

    /// Checks that a mark names the active transaction and its original logical lengths.
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

    /// Begins one append-only transaction, rejecting nesting and retaining every baseline length.
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

    /// Commits the same accepted prefix count for each sequence through atomic ragged validation.
    fn commit_prefix(
        &mut self,
        mark: Self::Mark,
        accepted_tokens: usize,
    ) -> Result<(), Self::Error> {
        let counts = vec![accepted_tokens; self.lengths.len()];
        self.commit_prefix_per_sequence(mark, &counts)
    }

    /// Restores all baseline logical lengths and consumes the active transaction identity.
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
    /// Formats the rejected transaction transition with sequence/count context when available.
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
#[path = "transaction_tests.rs"]
mod tests;
