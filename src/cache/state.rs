//! Checked logical lengths for independent decoder sequences.
//!
//! Each row retains an absolute processed-token count `0 <= length <= capacity`.
//! An append proposes `next = length + valid_tokens` using checked arithmetic;
//! all rows are validated before publishing a batched transition. Sliding eviction
//! does not reduce these counts because attention and RoPE need absolute offsets.
//!
//! This module owns integer metadata only. Reset/reorder/truncate are forbidden
//! during speculation, and tensor-owning callers must coordinate the corresponding
//! physical changes before treating a metadata transition as committed.

use core::fmt;

use super::lengths::{LengthLedger, LengthViolation};
use super::transaction::TransactionToken;
use super::{CacheConfig, TransactionError};

/// A checked before/after description of one sequence-length mutation.
///
/// Tensor-backed caches can use this metadata to derive the exact physical
/// write or invalidation range without duplicating overflow checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthTransition {
    /// Batch-row index affected by this transition.
    sequence: usize,
    /// Absolute token count before the transition.
    previous: usize,
    /// Absolute token count after the transition.
    current: usize,
}

impl LengthTransition {
    /// Returns the affected sequence index.
    #[must_use]
    pub const fn sequence(self) -> usize {
        self.sequence
    }

    /// Returns the logical length before the mutation.
    #[must_use]
    pub const fn previous(self) -> usize {
        self.previous
    }

    /// Returns the logical length after the mutation.
    #[must_use]
    pub const fn current(self) -> usize {
        self.current
    }

    /// Returns the appended absolute token range.
    ///
    /// This method is intended for transitions returned by [`SequenceLengths::advance`]
    /// or [`SequenceLengths::advance_all`]. A truncating transition yields an empty
    /// range at its new endpoint.
    #[must_use]
    pub fn appended_range(self) -> core::ops::Range<usize> {
        self.previous.min(self.current)..self.current
    }
}

/// Backend-independent lifecycle metadata for an autoregressive cache.
///
/// Lengths count all logically processed tokens, including tokens evicted from
/// a sliding-window layer. They therefore remain suitable as absolute RoPE
/// offsets. Physical cache tensors are intentionally not stored here.
///
/// # Committed and working lengths
///
/// [`lengths`](Self::lengths) always reports *committed* lengths. During a
/// speculative transaction ([`TransactionalCache::mark`]), appends move a
/// separate *working* cursor reported by
/// [`working_lengths`](Self::working_lengths); committing an accepted prefix
/// is the only way committed lengths change while a transaction is active.
/// Outside a transaction both accessors return the same lengths. Every
/// transition is computed by the Verus-verified ledger in `lengths.rs` and
/// published only after it succeeds, so a rejected call never changes state.
///
/// [`TransactionalCache::mark`]: super::TransactionalCache::mark
#[derive(Debug, PartialEq, Eq)]
pub struct SequenceLengths {
    /// Validated batch, capacity, and retention geometry.
    config: CacheConfig,
    /// Verified committed lengths and optional working cursor.
    pub(super) ledger: LengthLedger,
    /// Identity of the active transaction; present exactly when the ledger is active.
    pub(super) transaction: Option<TransactionToken>,
}

/// Former name of [`SequenceLengths`].
///
/// The crate root also exports an unrelated `DecodeState` lifecycle *trait*
/// for architecture states, so new code should prefer `SequenceLengths`.
pub type DecodeState = SequenceLengths;

impl SequenceLengths {
    /// Creates an empty decode state for the validated configuration.
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            ledger: LengthLedger::new(config.capacity(), config.sequence_count()),
            transaction: None,
        }
    }

    /// Creates a decode state with explicit initial logical lengths.
    ///
    /// # Errors
    ///
    /// Returns an error when the number of lengths differs from the configured
    /// sequence count or any length exceeds capacity.
    pub fn from_lengths(
        config: CacheConfig,
        lengths: impl Into<Vec<usize>>,
    ) -> Result<Self, DecodeStateError> {
        let lengths = lengths.into();
        validate_length_count(config.sequence_count(), lengths.len())?;
        let ledger = LengthLedger::from_lengths(config.capacity(), &lengths)
            .map_err(|violation| length_error(violation, "initialize"))?;
        Ok(Self {
            config,
            ledger,
            transaction: None,
        })
    }

    /// Returns the immutable cache configuration.
    #[must_use]
    pub const fn config(&self) -> CacheConfig {
        self.config
    }

    /// Returns all committed logical sequence lengths.
    ///
    /// During a transaction these stay at the transaction baseline; see
    /// [`working_lengths`](Self::working_lengths) for the tentative cursor.
    #[must_use]
    pub fn lengths(&self) -> &[usize] {
        self.ledger.committed()
    }

    /// Returns committed lengths; an explicit synonym of [`lengths`](Self::lengths).
    #[must_use]
    pub fn committed_lengths(&self) -> &[usize] {
        self.ledger.committed()
    }

    /// Returns the lengths the next append extends.
    ///
    /// Inside a transaction these include tentative proposal tokens; outside a
    /// transaction they equal [`lengths`](Self::lengths).
    #[must_use]
    pub fn working_lengths(&self) -> &[usize] {
        self.ledger.working()
    }

    /// Returns one committed logical sequence length.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStateError::SequenceOutOfBounds`] for an invalid index.
    pub fn length(&self, sequence: usize) -> Result<usize, DecodeStateError> {
        self.validate_sequence(sequence)?;
        Ok(self.ledger.committed()[sequence])
    }

    /// Returns how many more tokens the working cursor of one sequence accepts.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStateError::SequenceOutOfBounds`] for an invalid index.
    pub fn remaining_capacity(&self, sequence: usize) -> Result<usize, DecodeStateError> {
        self.validate_sequence(sequence)?;
        Ok(self.config.capacity() - self.ledger.working()[sequence])
    }

    /// Advances one sequence by a checked token count.
    ///
    /// An advance is the only lifecycle operation allowed inside an active
    /// transaction because speculative decoding may only append tentative
    /// state; it then moves the working cursor, and the returned transition
    /// describes that cursor.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid sequence, integer overflow, or exceeding
    /// the configured logical capacity.
    pub fn advance(
        &mut self,
        sequence: usize,
        token_count: usize,
    ) -> Result<LengthTransition, DecodeStateError> {
        self.validate_sequence(sequence)?;
        let mut counts = vec![0; self.ledger.committed().len()];
        counts[sequence] = token_count;
        let previous = self.ledger.working()[sequence];
        let next = self
            .ledger
            .advanced(&counts)
            .map_err(|violation| length_error(violation, "advance"))?;
        let current = next.working()[sequence];
        self.ledger = next;
        Ok(LengthTransition {
            sequence,
            previous,
            current,
        })
    }

    /// Atomically advances every sequence by its corresponding token count.
    ///
    /// Validation happens for all sequences before any length is changed.
    /// This matters for ragged batches: one overflowing row cannot leave other
    /// rows partially advanced. Inside a transaction the working cursor moves.
    ///
    /// # Errors
    ///
    /// Returns an error for a count mismatch, arithmetic overflow, or capacity
    /// overflow. On error the state is unchanged.
    pub fn advance_all(
        &mut self,
        token_counts: &[usize],
    ) -> Result<Vec<LengthTransition>, DecodeStateError> {
        let next = self
            .ledger
            .advanced(token_counts)
            .map_err(|violation| length_error(violation, "advance"))?;
        let transitions = transitions(self.ledger.working(), next.working());
        self.ledger = next;
        Ok(transitions)
    }

    /// Truncates one sequence to an earlier logical length.
    ///
    /// This operation updates metadata only. An append-only tensor cache may
    /// leave stale suffix values unreachable, but a wrapped sliding ring may
    /// already have evicted positions made visible by the rewind. A physical
    /// cache must restore those values from a checkpoint or journal, or reject
    /// such a truncation.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid sequence, an attempted extension, or an
    /// active speculative transaction.
    pub fn truncate(
        &mut self,
        sequence: usize,
        new_length: usize,
    ) -> Result<LengthTransition, DecodeStateError> {
        self.ensure_topology_mutation_allowed("truncate")?;
        let previous = self.length(sequence)?;
        let mut requested = self.ledger.committed().to_vec();
        requested[sequence] = new_length;
        self.ledger = self.plan_truncate(&requested)?;
        Ok(LengthTransition {
            sequence,
            previous,
            current: new_length,
        })
    }

    /// Atomically truncates all sequences to corresponding earlier lengths.
    ///
    /// Like [`truncate`](Self::truncate), this changes metadata only. Tensor
    /// caches backed by wrapped rings must separately restore any history that
    /// the new lengths make visible.
    ///
    /// # Errors
    ///
    /// Returns an error for a count mismatch, any attempted extension, or an
    /// active speculative transaction. On error the state is unchanged.
    pub fn truncate_all(
        &mut self,
        new_lengths: &[usize],
    ) -> Result<Vec<LengthTransition>, DecodeStateError> {
        let next = self.plan_truncate(new_lengths)?;
        let transitions = transitions(self.ledger.committed(), next.committed());
        self.ledger = next;
        Ok(transitions)
    }

    /// Resets every logical sequence length to zero.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStateError::TransactionActive`] while speculative state
    /// is active; callers must commit or roll back first.
    pub fn reset(&mut self) -> Result<Vec<LengthTransition>, DecodeStateError> {
        let next = self.plan_reset()?;
        let transitions = transitions(self.ledger.committed(), next.committed());
        self.ledger = next;
        Ok(transitions)
    }

    /// Reorders sequence metadata according to source indices.
    ///
    /// Duplicate indices are allowed, which supports beam expansion from an
    /// existing beam. The number of output sequences remains fixed so tensor
    /// storage shapes do not change.
    ///
    /// # Errors
    ///
    /// Returns an error for a length mismatch, an out-of-range source index, or
    /// an active speculative transaction. On error the state is unchanged.
    pub fn reorder(&mut self, source_indices: &[usize]) -> Result<(), DecodeStateError> {
        self.ledger = self.plan_reorder(source_indices)?;
        Ok(())
    }

    /// Returns whether a speculative transaction is active.
    #[must_use]
    pub const fn has_active_transaction(&self) -> bool {
        self.transaction.is_some()
    }

    /// Returns each row's tentative proposal length (all zero outside a transaction).
    #[must_use]
    pub fn proposal_lengths(&self) -> Vec<usize> {
        self.ledger.proposals()
    }

    /// Plans an atomic truncation without publishing it.
    pub(crate) fn plan_truncate(
        &self,
        requested: &[usize],
    ) -> Result<LengthLedger, DecodeStateError> {
        self.ensure_topology_mutation_allowed("truncate")?;
        self.ledger
            .truncated(requested)
            .map_err(|violation| length_error(violation, "truncate"))
    }

    /// Plans an atomic reorder without publishing it.
    pub(crate) fn plan_reorder(&self, indices: &[usize]) -> Result<LengthLedger, DecodeStateError> {
        self.ensure_topology_mutation_allowed("reorder")?;
        self.ledger
            .reordered(indices)
            .map_err(|violation| length_error(violation, "reorder"))
    }

    /// Plans a reset to zero lengths without publishing it.
    pub(crate) fn plan_reset(&self) -> Result<LengthLedger, DecodeStateError> {
        self.ensure_topology_mutation_allowed("reset")?;
        self.ledger
            .reset()
            .map_err(|violation| length_error(violation, "reset"))
    }

    /// Publishes a ledger planned from this state outside any transaction.
    ///
    /// Callers pass only ledgers returned by `plan_*` on this same state after
    /// all physical validation succeeded; ending a transaction goes through
    /// [`finish_transaction`](Self::finish_transaction) instead.
    pub(crate) fn publish(&mut self, ledger: LengthLedger) {
        debug_assert!(!ledger.is_active() && self.transaction.is_none());
        self.ledger = ledger;
    }

    /// Rejects an out-of-range row index with the batch size for context.
    fn validate_sequence(&self, sequence: usize) -> Result<(), DecodeStateError> {
        let sequence_count = self.ledger.committed().len();
        if sequence < sequence_count {
            Ok(())
        } else {
            Err(DecodeStateError::SequenceOutOfBounds {
                sequence,
                sequence_count,
            })
        }
    }

    /// Rejects reset/reorder/truncate operations while an append-only transaction is active.
    fn ensure_topology_mutation_allowed(
        &self,
        operation: &'static str,
    ) -> Result<(), DecodeStateError> {
        if self.has_active_transaction() {
            Err(DecodeStateError::TransactionActive { operation })
        } else {
            Ok(())
        }
    }
}

/// Describes the row-wise change from `previous` to `current` lengths.
fn transitions(previous: &[usize], current: &[usize]) -> Vec<LengthTransition> {
    previous
        .iter()
        .zip(current)
        .enumerate()
        .map(|(sequence, (&previous, &current))| LengthTransition {
            sequence,
            previous,
            current,
        })
        .collect()
}

/// Maps a verified-ledger violation to the public metadata error.
///
/// `operation` names the rejected lifecycle call when a transaction blocks it.
/// Transaction-specific violations (`commit`, `rollback`) are mapped by the
/// transaction module, which owns their richer diagnostics.
pub(crate) fn length_error(
    violation: LengthViolation,
    operation: &'static str,
) -> DecodeStateError {
    match violation {
        LengthViolation::CountMismatch { expected, actual } => {
            DecodeStateError::SequenceCountMismatch { expected, actual }
        }
        LengthViolation::Overflow {
            sequence,
            length,
            additional,
        } => DecodeStateError::LengthOverflow {
            sequence,
            length,
            additional,
        },
        LengthViolation::CapacityExceeded {
            sequence,
            requested,
            capacity,
        } => DecodeStateError::CapacityExceeded {
            sequence,
            requested,
            capacity,
        },
        LengthViolation::Extension {
            sequence,
            current,
            requested,
        } => DecodeStateError::CannotExtendByTruncation {
            sequence,
            current,
            requested,
        },
        LengthViolation::SourceOutOfBounds {
            source,
            sequence_count,
            ..
        } => DecodeStateError::SequenceOutOfBounds {
            sequence: source,
            sequence_count,
        },
        LengthViolation::PrefixExceedsProposal {
            sequence,
            accepted,
            proposed,
        } => TransactionError::PrefixExceedsAppended {
            sequence,
            accepted,
            appended: proposed,
        }
        .into(),
        LengthViolation::TransactionActive => DecodeStateError::TransactionActive { operation },
        LengthViolation::TransactionInactive => TransactionError::NoActiveTransaction.into(),
    }
}

/// Failure while updating [`SequenceLengths`] lifecycle metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeStateError {
    /// The caller supplied a different number of per-sequence values.
    SequenceCountMismatch {
        /// Expected number of values.
        expected: usize,
        /// Supplied number of values.
        actual: usize,
    },
    /// A sequence index was outside the tracked batch.
    SequenceOutOfBounds {
        /// Invalid sequence index.
        sequence: usize,
        /// Number of tracked sequences.
        sequence_count: usize,
    },
    /// Adding a token count overflowed `usize`.
    LengthOverflow {
        /// Affected sequence index.
        sequence: usize,
        /// Existing length.
        length: usize,
        /// Requested increment.
        additional: usize,
    },
    /// A logical length exceeded configured capacity.
    CapacityExceeded {
        /// Affected sequence index.
        sequence: usize,
        /// Requested logical length.
        requested: usize,
        /// Configured maximum logical length.
        capacity: usize,
    },
    /// Truncation was asked to increase a sequence length.
    CannotExtendByTruncation {
        /// Affected sequence index.
        sequence: usize,
        /// Existing logical length.
        current: usize,
        /// Invalid requested length.
        requested: usize,
    },
    /// A topology-changing operation was attempted during speculation.
    TransactionActive {
        /// Name of the rejected operation.
        operation: &'static str,
    },
    /// A transactional operation failed.
    Transaction(
        /// Underlying mark, lifecycle, or accepted-prefix validation failure.
        TransactionError,
    ),
}

impl fmt::Display for DecodeStateError {
    /// Formats lifecycle failures with sequence indices, lengths, and capacity where available.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SequenceCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} per-sequence values, received {actual}"
            ),
            Self::SequenceOutOfBounds {
                sequence,
                sequence_count,
            } => write!(
                formatter,
                "sequence index {sequence} is out of bounds for {sequence_count} sequences"
            ),
            Self::LengthOverflow {
                sequence,
                length,
                additional,
            } => write!(
                formatter,
                "sequence {sequence} length {length} overflows when advanced by {additional}"
            ),
            Self::CapacityExceeded {
                sequence,
                requested,
                capacity,
            } => write!(
                formatter,
                "sequence {sequence} requested length {requested} exceeds capacity {capacity}"
            ),
            Self::CannotExtendByTruncation {
                sequence,
                current,
                requested,
            } => write!(
                formatter,
                "cannot truncate sequence {sequence} from {current} to larger length {requested}"
            ),
            Self::TransactionActive { operation } => write!(
                formatter,
                "cannot {operation} while a cache transaction is active"
            ),
            Self::Transaction(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DecodeStateError {
    /// Exposes an underlying transaction error while leaving direct validation failures unchained.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transaction(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TransactionError> for DecodeStateError {
    /// Wraps a transaction failure without discarding its typed cause.
    fn from(value: TransactionError) -> Self {
        Self::Transaction(value)
    }
}

/// Requires exactly one logical length or increment per tracked sequence.
fn validate_length_count(expected: usize, actual: usize) -> Result<(), DecodeStateError> {
    if expected == actual {
        Ok(())
    } else {
        Err(DecodeStateError::SequenceCountMismatch { expected, actual })
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;
