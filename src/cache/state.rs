use core::fmt;

use super::{CacheConfig, TransactionError};

/// A checked before/after description of one sequence-length mutation.
///
/// Tensor-backed caches can use this metadata to derive the exact physical
/// write or invalidation range without duplicating overflow checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthTransition {
    sequence: usize,
    previous: usize,
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
    /// This method is intended for transitions returned by [`DecodeState::advance`]
    /// or [`DecodeState::advance_all`]. A truncating transition yields an empty
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
#[derive(Debug, PartialEq, Eq)]
pub struct DecodeState {
    pub(crate) config: CacheConfig,
    pub(crate) lengths: Vec<usize>,
    pub(crate) active_transaction: Option<super::transaction::ActiveTransaction>,
}

impl DecodeState {
    /// Creates an empty decode state for the validated configuration.
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            lengths: vec![0; config.sequence_count()],
            active_transaction: None,
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
        for (sequence, &length) in lengths.iter().enumerate() {
            validate_capacity(sequence, length, config.capacity())?;
        }

        Ok(Self {
            config,
            lengths,
            active_transaction: None,
        })
    }

    /// Returns the immutable cache configuration.
    #[must_use]
    pub const fn config(&self) -> CacheConfig {
        self.config
    }

    /// Returns all logical sequence lengths.
    #[must_use]
    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// Returns one logical sequence length.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStateError::SequenceOutOfBounds`] for an invalid index.
    pub fn length(&self, sequence: usize) -> Result<usize, DecodeStateError> {
        self.lengths
            .get(sequence)
            .copied()
            .ok_or(DecodeStateError::SequenceOutOfBounds {
                sequence,
                sequence_count: self.lengths.len(),
            })
    }

    /// Returns the unused logical capacity for one sequence.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStateError::SequenceOutOfBounds`] for an invalid index.
    pub fn remaining_capacity(&self, sequence: usize) -> Result<usize, DecodeStateError> {
        Ok(self.config.capacity() - self.length(sequence)?)
    }

    /// Advances one sequence by a checked token count.
    ///
    /// An advance is the only lifecycle operation allowed inside an active
    /// transaction because speculative decoding may only append tentative state.
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
        let sequence_count = self.lengths.len();
        let length =
            self.lengths
                .get_mut(sequence)
                .ok_or(DecodeStateError::SequenceOutOfBounds {
                    sequence,
                    sequence_count,
                })?;
        let previous = *length;
        let current =
            previous
                .checked_add(token_count)
                .ok_or(DecodeStateError::LengthOverflow {
                    sequence,
                    length: previous,
                    additional: token_count,
                })?;
        validate_capacity(sequence, current, self.config.capacity())?;
        *length = current;

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
    /// rows partially advanced.
    ///
    /// # Errors
    ///
    /// Returns an error for a count mismatch, arithmetic overflow, or capacity
    /// overflow. On error the state is unchanged.
    pub fn advance_all(
        &mut self,
        token_counts: &[usize],
    ) -> Result<Vec<LengthTransition>, DecodeStateError> {
        validate_length_count(self.lengths.len(), token_counts.len())?;

        let mut next = Vec::with_capacity(self.lengths.len());
        for (sequence, (&length, &additional)) in self.lengths.iter().zip(token_counts).enumerate()
        {
            let current =
                length
                    .checked_add(additional)
                    .ok_or(DecodeStateError::LengthOverflow {
                        sequence,
                        length,
                        additional,
                    })?;
            validate_capacity(sequence, current, self.config.capacity())?;
            next.push(current);
        }

        let transitions = self
            .lengths
            .iter()
            .copied()
            .zip(next.iter().copied())
            .enumerate()
            .map(|(sequence, (previous, current))| LengthTransition {
                sequence,
                previous,
                current,
            })
            .collect();
        self.lengths = next;
        Ok(transitions)
    }

    /// Truncates one sequence to an earlier logical length.
    ///
    /// Stale tensor values beyond the returned `current` length may remain in
    /// physical storage but are unreachable and may be overwritten.
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
        if new_length > previous {
            return Err(DecodeStateError::CannotExtendByTruncation {
                sequence,
                current: previous,
                requested: new_length,
            });
        }
        self.lengths[sequence] = new_length;
        Ok(LengthTransition {
            sequence,
            previous,
            current: new_length,
        })
    }

    /// Atomically truncates all sequences to corresponding earlier lengths.
    ///
    /// # Errors
    ///
    /// Returns an error for a count mismatch, any attempted extension, or an
    /// active speculative transaction. On error the state is unchanged.
    pub fn truncate_all(
        &mut self,
        new_lengths: &[usize],
    ) -> Result<Vec<LengthTransition>, DecodeStateError> {
        self.ensure_topology_mutation_allowed("truncate_all")?;
        validate_length_count(self.lengths.len(), new_lengths.len())?;
        for (sequence, (&current, &requested)) in self.lengths.iter().zip(new_lengths).enumerate() {
            if requested > current {
                return Err(DecodeStateError::CannotExtendByTruncation {
                    sequence,
                    current,
                    requested,
                });
            }
        }

        let transitions = self
            .lengths
            .iter()
            .copied()
            .zip(new_lengths.iter().copied())
            .enumerate()
            .map(|(sequence, (previous, current))| LengthTransition {
                sequence,
                previous,
                current,
            })
            .collect();
        self.lengths.copy_from_slice(new_lengths);
        Ok(transitions)
    }

    /// Resets every logical sequence length to zero.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStateError::TransactionActive`] while speculative state
    /// is active; callers must commit or roll back first.
    pub fn reset(&mut self) -> Result<Vec<LengthTransition>, DecodeStateError> {
        self.ensure_topology_mutation_allowed("reset")?;
        let transitions = self
            .lengths
            .iter()
            .copied()
            .enumerate()
            .map(|(sequence, previous)| LengthTransition {
                sequence,
                previous,
                current: 0,
            })
            .collect();
        self.lengths.fill(0);
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
        self.ensure_topology_mutation_allowed("reorder")?;
        validate_length_count(self.lengths.len(), source_indices.len())?;

        let mut reordered = Vec::with_capacity(self.lengths.len());
        for &source in source_indices {
            reordered.push(self.length(source)?);
        }
        self.lengths = reordered;
        Ok(())
    }

    /// Returns whether a speculative transaction is active.
    #[must_use]
    pub const fn has_active_transaction(&self) -> bool {
        self.active_transaction.is_some()
    }

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

/// Failure while updating [`DecodeState`] lifecycle metadata.
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
    Transaction(TransactionError),
}

impl fmt::Display for DecodeStateError {
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
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transaction(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TransactionError> for DecodeStateError {
    fn from(value: TransactionError) -> Self {
        Self::Transaction(value)
    }
}

fn validate_length_count(expected: usize, actual: usize) -> Result<(), DecodeStateError> {
    if expected == actual {
        Ok(())
    } else {
        Err(DecodeStateError::SequenceCountMismatch { expected, actual })
    }
}

fn validate_capacity(
    sequence: usize,
    requested: usize,
    capacity: usize,
) -> Result<(), DecodeStateError> {
    if requested <= capacity {
        Ok(())
    } else {
        Err(DecodeStateError::CapacityExceeded {
            sequence,
            requested,
            capacity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> CacheConfig {
        CacheConfig::new(3, 8).expect("valid config")
    }

    #[test]
    fn constructs_empty_and_initialized_states() {
        let empty = DecodeState::new(config());
        assert_eq!(empty.lengths(), &[0, 0, 0]);
        assert_eq!(empty.remaining_capacity(1), Ok(8));

        let initialized =
            DecodeState::from_lengths(config(), [1, 4, 8].to_vec()).expect("valid initial lengths");
        assert_eq!(initialized.lengths(), &[1, 4, 8]);
        assert_eq!(initialized.remaining_capacity(2), Ok(0));
    }

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
    }

    #[test]
    fn truncates_atomically_and_never_extends() {
        let mut state = DecodeState::from_lengths(config(), vec![5, 6, 7]).expect("valid state");
        let transition = state.truncate(1, 2).expect("valid truncation");
        assert_eq!(transition.previous(), 6);
        assert_eq!(transition.current(), 2);
        assert_eq!(transition.appended_range(), 2..2);

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

    #[test]
    fn display_messages_are_actionable() {
        assert_eq!(
            DecodeStateError::CapacityExceeded {
                sequence: 2,
                requested: 9,
                capacity: 8,
            }
            .to_string(),
            "sequence 2 requested length 9 exceeds capacity 8"
        );
    }
}
