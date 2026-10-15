//! Single-use transactions for speculative logical cache lengths.
//!
//! A mark captures baseline lengths `L_b`; after proposing `P_b` tokens, accepting
//! `A_b` requires `0 <= A_b <= P_b` and commits `L_b + A_b`. Validation finishes
//! for every row before metadata changes. Rollback restores every `L_b`.
//!
//! Allocation identity distinguishes active, stale, and foreign marks, including
//! cloned handles. The arithmetic itself is the Verus-verified ledger in
//! `lengths.rs`, shared with the tensor-backed decode state. These integer transactions do not journal tensor values: ring
//! and recurrent caches must stage their proposed data outside committed storage.

use core::fmt;
use std::sync::Arc;

use super::lengths::{LengthLedger, LengthViolation};
use super::state::length_error;
use super::{DecodeStateError, SequenceLengths};

/// A single-use logical checkpoint for speculative cache writes.
///
/// The mark belongs to the [`SequenceLengths`] transaction that created it. A
/// cloned or otherwise stale mark is rejected after the original transaction
/// commits or rolls back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheMark {
    /// Allocation identity linking this mark to one transaction.
    token: TransactionToken,
    /// Absolute per-row lengths before speculative appends.
    baseline_lengths: Vec<usize>,
}

impl CacheMark {
    /// Returns the sequence lengths captured before speculative appends.
    #[must_use]
    pub fn baseline_lengths(&self) -> &[usize] {
        &self.baseline_lengths
    }
}

/// Shared allocation identity that remains distinct from every unrelated transaction.
///
/// Equality compares the allocation address, so a mark can only finish the
/// transaction that issued it, even when another state has identical lengths.
#[derive(Clone)]
pub(crate) struct TransactionToken(
    /// Address identity shared by cloned handles; the payload itself carries no state.
    Arc<()>,
);

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
/// [`SequenceLengths`] implementation tracks metadata only; tensor-backed callers
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

impl SequenceLengths {
    /// Commits a possibly ragged accepted prefix for each sequence.
    ///
    /// Each accepted count must be no greater than the number of tokens
    /// appended to that sequence since `mark`. Validation is atomic: a bad row
    /// leaves the transaction active and all lengths unchanged, so the caller
    /// may still roll it back.
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
        self.check_mark(&mark)?;
        let next = self.plan_commit(accepted_tokens).map_err(commit_error)?;
        self.finish_transaction(next);
        Ok(())
    }

    /// Checks that a mark names the active transaction and its original logical lengths.
    pub(crate) fn check_mark(&self, mark: &CacheMark) -> Result<(), TransactionError> {
        let token = self
            .transaction
            .as_ref()
            .ok_or(TransactionError::NoActiveTransaction)?;
        if *token != mark.token || self.ledger.committed() != mark.baseline_lengths {
            return Err(TransactionError::StaleOrForeignMark);
        }
        Ok(())
    }

    /// Plans a ragged commit through the verified ledger without publishing it.
    pub(crate) fn plan_commit(&self, accepted: &[usize]) -> Result<LengthLedger, LengthViolation> {
        self.ledger.committed_prefix(accepted)
    }

    /// Ends the active transaction by publishing a planned commit or rollback ledger.
    pub(crate) fn finish_transaction(&mut self, ledger: LengthLedger) {
        debug_assert!(!ledger.is_active());
        self.ledger = ledger;
        self.transaction = None;
    }

    /// Plans the rollback ledger: committed lengths with no working cursor.
    pub(crate) fn plan_rollback(&self) -> Result<LengthLedger, LengthViolation> {
        self.ledger.rolled_back()
    }
}

impl TransactionalCache for SequenceLengths {
    /// Single-use checkpoint carrying the transaction identity and baseline lengths.
    type Mark = CacheMark;
    /// Logical metadata and transaction validation errors.
    type Error = DecodeStateError;

    /// Begins one append-only transaction, rejecting nesting and retaining every baseline length.
    fn mark(&mut self) -> Result<Self::Mark, Self::Error> {
        if self.transaction.is_some() {
            return Err(TransactionError::AlreadyActive.into());
        }
        let next = self
            .ledger
            .begun()
            .map_err(|_| DecodeStateError::from(TransactionError::AlreadyActive))?;
        let token = TransactionToken::new();
        let baseline_lengths = self.ledger.committed().to_vec();
        self.ledger = next;
        self.transaction = Some(token.clone());
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
        let counts = vec![accepted_tokens; self.lengths().len()];
        self.commit_prefix_per_sequence(mark, &counts)
    }

    /// Restores all baseline logical lengths and consumes the active transaction identity.
    fn rollback(&mut self, mark: Self::Mark) -> Result<(), Self::Error> {
        self.check_mark(&mark)?;
        let next = self
            .plan_rollback()
            .map_err(|_| DecodeStateError::from(TransactionError::NoActiveTransaction))?;
        self.finish_transaction(next);
        Ok(())
    }
}

/// Maps a rejected commit plan to the transaction-specific public errors.
fn commit_error(violation: LengthViolation) -> DecodeStateError {
    match violation {
        LengthViolation::CountMismatch { expected, actual } => {
            TransactionError::SequenceCountMismatch { expected, actual }.into()
        }
        violation => length_error(violation, "commit"),
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
