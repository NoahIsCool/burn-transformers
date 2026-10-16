//! Exhaustive checks of the provenance hash worker count and buffer shares.

use super::*;

/// The loop matches the original `1 << ilog2` choice for every small input.
#[test]
fn worker_counts_match_the_largest_power_of_two() {
    for shards in 0..40 {
        for parallelism in 0..40 {
            let usable = parallelism.min(MAX_HASH_WORKERS).min(shards).max(1);
            assert_eq!(
                hash_worker_count(shards, parallelism),
                1 << usable.ilog2(),
                "{shards} shards, {parallelism} threads"
            );
        }
    }
    assert_eq!(hash_worker_count(usize::MAX, usize::MAX), 8);
}

/// Power-of-two worker shares sum to exactly the budget.
#[test]
fn worker_shares_split_the_budget_exactly() {
    for workers in [1, 2, 4, 8] {
        assert_eq!(
            worker_share(workers) * workers,
            PROVENANCE_HASH_BUFFER_BYTES
        );
    }
    assert_eq!(worker_share(0), PROVENANCE_HASH_BUFFER_BYTES);
    assert_eq!(worker_share(3), PROVENANCE_HASH_BUFFER_BYTES / 3);
}
