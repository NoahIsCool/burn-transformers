use core::fmt;
use core::num::NonZeroUsize;

/// Validated dimensions and retention policy for a decode cache.
///
/// `capacity` is the maximum *logical* length of every sequence. A sliding
/// window, when present, controls how many recent tokens a local-attention
/// layer retains physically; it does not change absolute token positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheConfig {
    sequence_count: NonZeroUsize,
    capacity: NonZeroUsize,
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

    /// Enables a non-zero physical sliding window.
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

    /// Returns the local-attention retention window, when configured.
    #[must_use]
    pub const fn sliding_window(self) -> Option<NonZeroUsize> {
        self.sliding_window
    }

    /// Returns the number of physical time slots needed per cache tensor.
    ///
    /// Global layers allocate the full logical capacity. Local layers need
    /// only the sliding-window width when using ring-buffer storage.
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
mod tests {
    use super::*;

    #[test]
    fn validates_required_dimensions() {
        assert_eq!(CacheConfig::new(0, 1), Err(CacheConfigError::ZeroSequences));
        assert_eq!(CacheConfig::new(1, 0), Err(CacheConfigError::ZeroCapacity));
    }

    #[test]
    fn validates_sliding_window() {
        let config = CacheConfig::new(2, 8).expect("valid config");
        assert_eq!(
            config.with_sliding_window(0),
            Err(CacheConfigError::ZeroSlidingWindow)
        );
        assert_eq!(
            config.with_sliding_window(9),
            Err(CacheConfigError::SlidingWindowExceedsCapacity {
                window: 9,
                capacity: 8,
            })
        );
    }

    #[test]
    fn reports_logical_and_physical_capacity() {
        let global = CacheConfig::new(3, 128).expect("valid config");
        assert_eq!(global.sequence_count(), 3);
        assert_eq!(global.capacity(), 128);
        assert_eq!(global.sliding_window(), None);
        assert_eq!(global.physical_capacity(), 128);

        let local = global.with_sliding_window(32).expect("valid window");
        assert_eq!(local.capacity(), 128);
        assert_eq!(local.sliding_window().map(NonZeroUsize::get), Some(32));
        assert_eq!(local.physical_capacity(), 32);
    }

    #[test]
    fn error_messages_include_invalid_values() {
        assert_eq!(
            CacheConfigError::SlidingWindowExceedsCapacity {
                window: 17,
                capacity: 16,
            }
            .to_string(),
            "sliding window 17 exceeds logical capacity 16"
        );
    }
}
