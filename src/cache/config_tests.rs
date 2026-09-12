//! Configuration dimensions, retention bounds, and diagnostic contracts.

use super::*;

/// Rejects zero batch size and zero logical capacity.
#[test]
fn validates_required_dimensions() {
    assert_eq!(CacheConfig::new(0, 1), Err(CacheConfigError::ZeroSequences));
    assert_eq!(CacheConfig::new(1, 0), Err(CacheConfigError::ZeroCapacity));
}

/// Rejects zero or oversized windows and accepts a window equal to capacity.
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

/// Distinguishes absolute context capacity from retained physical slots.
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

/// Includes invalid dimensions in configuration diagnostics.
#[test]
fn error_messages_include_invalid_values() {
    assert_eq!(
        CacheConfigError::ZeroSequences.to_string(),
        "cache must track at least one sequence"
    );
    assert_eq!(
        CacheConfigError::ZeroCapacity.to_string(),
        "cache capacity must be non-zero"
    );
    assert_eq!(
        CacheConfigError::ZeroSlidingWindow.to_string(),
        "cache sliding window must be non-zero"
    );
    assert_eq!(
        CacheConfigError::SlidingWindowExceedsCapacity {
            window: 17,
            capacity: 16,
        }
        .to_string(),
        "sliding window 17 exceeds logical capacity 16"
    );
}
