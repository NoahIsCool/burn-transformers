//! Validated logical capacity and physical retention dimensions.
//!
//! A session tracks `batch > 0` sequences and permits absolute lengths in
//! `0..=capacity`. A sliding ring retains at most `window <= capacity` tokens;
//! eviction changes physical storage, never absolute position numbering. These
//! integer-only checks precede tensor allocation and are independent of dtype.
//!
//! Model families size every layer's retention from their own architecture
//! configuration (a Mistral or Gemma sliding window, for example), so the
//! optional window here never changes what a model remembers; it is the
//! default ring width of raw cache banks built directly from a
//! [`CacheConfig`]. Every family ignores it, so one configuration serves
//! every family.

use core::fmt;
use core::num::NonZeroUsize;

/// Validated dimensions and retention policy for a decode cache.
///
/// `capacity` is the maximum *logical* length of every sequence. The optional
/// sliding window is the ring width of raw [`TensorCacheBank`] and
/// [`NativeKvCache`] sliding-ring groups that carry no window of their own; it
/// never changes absolute token positions. Model decode states take each
/// layer's window from the architecture instead and ignore this field, so one
/// configuration serves every family.
///
/// [`TensorCacheBank`]: super::TensorCacheBank
/// [`NativeKvCache`]: super::NativeKvCache
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheConfig {
    /// Number of independently tracked sequences, always nonzero.
    sequence_count: NonZeroUsize,
    /// Maximum absolute token count of each sequence, always nonzero.
    capacity: NonZeroUsize,
    /// Optional physical retention width, no greater than logical capacity.
    sliding_window: Option<NonZeroUsize>,
}

impl CacheConfig {
    /// Creates a cache configuration without sliding-window eviction.
    ///
    /// # Errors
    ///
    /// Returns [`CacheConfigError::ZeroSequences`] or
    /// [`CacheConfigError::ZeroCapacity`] when the corresponding value is zero.
    pub fn new(sequence_count: usize, capacity: usize) -> Result<Self, CacheConfigError> {
        let sequence_count =
            NonZeroUsize::new(sequence_count).ok_or(CacheConfigError::ZeroSequences)?;
        let capacity = NonZeroUsize::new(capacity).ok_or(CacheConfigError::ZeroCapacity)?;

        Ok(Self {
            sequence_count,
            capacity,
            sliding_window: None,
        })
    }

    /// Sets the default ring width of raw sliding-ring cache groups.
    ///
    /// Groups with their own window keep it, full-storage groups ignore it,
    /// and model families size retention from their architecture, so this
    /// does not make a model forget older tokens.
    ///
    /// # Errors
    ///
    /// Returns [`CacheConfigError::ZeroSlidingWindow`] for zero, or
    /// [`CacheConfigError::SlidingWindowExceedsCapacity`] when the physical
    /// window would be larger than the logical context capacity.
    pub fn with_sliding_window(mut self, window: usize) -> Result<Self, CacheConfigError> {
        let window = NonZeroUsize::new(window).ok_or(CacheConfigError::ZeroSlidingWindow)?;
        if window > self.capacity {
            return Err(CacheConfigError::SlidingWindowExceedsCapacity {
                window: window.get(),
                capacity: self.capacity.get(),
            });
        }

        self.sliding_window = Some(window);
        Ok(self)
    }

    /// Returns the number of independently tracked sequences.
    #[must_use]
    pub const fn sequence_count(self) -> usize {
        self.sequence_count.get()
    }

    /// Returns the maximum logical token count for each sequence.
    #[must_use]
    pub const fn capacity(self) -> usize {
        self.capacity.get()
    }

    /// Returns the default ring width for raw sliding-ring groups, when configured.
    #[must_use]
    pub const fn sliding_window(self) -> Option<NonZeroUsize> {
        self.sliding_window
    }

    /// Returns the window width when one is configured, else the logical capacity.
    ///
    /// This is the slot count of a *standard sliding ring* under this config.
    /// It is not the allocation of every group: full storage allocates the
    /// whole capacity even when a window is configured, a mirrored ring
    /// allocates twice its window, and a group may carry its own window. Use
    /// [`NativeKvCacheSpec::committed_time_slots`] for the exact extent.
    ///
    /// [`NativeKvCacheSpec::committed_time_slots`]: super::NativeKvCacheSpec::committed_time_slots
    #[deprecated(
        note = "ambiguous across storage modes; use NativeKvCacheSpec::committed_time_slots"
    )]
    #[must_use]
    pub const fn physical_capacity(self) -> usize {
        match self.sliding_window {
            Some(window) => window.get(),
            None => self.capacity.get(),
        }
    }
}

/// Validation failure while constructing a [`CacheConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheConfigError {
    /// At least one sequence is required.
    ZeroSequences,
    /// Logical context capacity must be non-zero.
    ZeroCapacity,
    /// A present sliding window must be non-zero.
    ZeroSlidingWindow,
    /// The physical window cannot exceed the logical context capacity.
    SlidingWindowExceedsCapacity {
        /// Requested window width.
        window: usize,
        /// Configured logical capacity.
        capacity: usize,
    },
}

impl fmt::Display for CacheConfigError {
    /// Formats the invalid cache dimensions or retention window with their actual values.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSequences => formatter.write_str("cache must track at least one sequence"),
            Self::ZeroCapacity => formatter.write_str("cache capacity must be non-zero"),
            Self::ZeroSlidingWindow => formatter.write_str("cache sliding window must be non-zero"),
            Self::SlidingWindowExceedsCapacity { window, capacity } => write!(
                formatter,
                "sliding window {window} exceeds logical capacity {capacity}"
            ),
        }
    }
}

impl std::error::Error for CacheConfigError {}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
