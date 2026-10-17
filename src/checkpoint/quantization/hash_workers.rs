//! Worker count and buffer shares of the concurrent provenance hash pass.
//!
//! Shards hash concurrently on `w` workers, each streaming through its own
//! `B / w`-byte share of the pass budget `B` ([`PROVENANCE_HASH_BUFFER_BYTES`]).
//! The worker count is the largest power of two not above
//! `usable = max(1, min(parallelism, 8, shards))` ([`hash_worker_count`]), so it
//! divides the power-of-two budget and the shares sum to exactly `B`
//! ([`worker_share`]).
//!
//! The module has no crate or Burn dependencies so that Verus can check it in
//! place through `verus/perf.rs`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Total streaming-buffer budget of one provenance hash pass.
///
/// Concurrent shard workers split this budget evenly, so a pass never owns more
/// hash-buffer memory than this, however many workers run.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const PROVENANCE_HASH_BUFFER_BYTES: usize = 1024 * 1024;

/// Upper bound on concurrent shard-hashing workers.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const MAX_HASH_WORKERS: usize = 8;

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: the workers a pass may use, `max(1, min(parallelism, 8, shards))`.
pub open spec fn usable_workers(shards: int, parallelism: int) -> int {
    let bounded = if parallelism < 8 { parallelism } else { 8 };
    let limited = if shards < bounded { shards } else { bounded };
    if limited < 1 { 1 } else { limited }
}

} // verus!

/// Chooses how many workers hash `shards` shards on `parallelism` hardware threads.
///
/// The count is the largest power of two not above `min(parallelism, 8, shards)`
/// (and at least one), so it divides [`PROVENANCE_HASH_BUFFER_BYTES`] exactly and
/// the workers' buffer shares sum to the budget with no remainder. Rounding down
/// gives up less than half of the usable workers, and only when the shard or
/// thread count is not a power of two (three shards hash on two workers).
///
/// Verified: the result is 1, 2, 4, or 8, at most the usable worker count and
/// more than half of it, and it divides the hash-buffer budget.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(workers =>
    ensures
        workers == 1 || workers == 2 || workers == 4 || workers == 8,
        workers <= usable_workers(shards as int, parallelism as int),
        usable_workers(shards as int, parallelism as int) < 2 * workers,
        PROVENANCE_HASH_BUFFER_BYTES % workers == 0,
))]
pub(crate) fn hash_worker_count(shards: usize, parallelism: usize) -> usize {
    let usable = parallelism.min(MAX_HASH_WORKERS).min(shards).max(1);
    // The largest power of two not above `usable`; `ilog2` has no specification.
    let mut workers = 1;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            1 <= usable <= 8,
            usable == usable_workers(shards as int, parallelism as int),
            workers == 1 || workers == 2 || workers == 4 || workers == 8,
            workers <= usable,
        decreases usable - workers,
    ))]
    while workers * 2 <= usable {
        workers *= 2;
    }
    workers
}

/// Returns one worker's share of the hash-buffer budget, `B / max(workers, 1)`.
///
/// Verified: the exact quotient, and for every power-of-two worker count up to
/// eight the shares sum to exactly the budget, so a pass that charges each
/// worker's share once reports the budget itself.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(share =>
    ensures
        share == PROVENANCE_HASH_BUFFER_BYTES / (if workers > 1 { workers } else { 1usize }),
        (workers == 1 || workers == 2 || workers == 4 || workers == 8)
            ==> share * workers == PROVENANCE_HASH_BUFFER_BYTES,
))]
pub(crate) fn worker_share(workers: usize) -> usize {
    let divisor = workers.max(1);
    let share = PROVENANCE_HASH_BUFFER_BYTES / divisor;
    #[cfg(verus_keep_ghost)]
    proof! {
        // The budget 2^20 is a multiple of every power of two up to eight.
        if workers == 1 || workers == 2 || workers == 4 || workers == 8 {
            assert(divisor == workers);
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
                PROVENANCE_HASH_BUFFER_BYTES as int,
                divisor as int,
            );
            assert(1048576int % 1 == 0 && 1048576int % 2 == 0) by (compute);
            assert(1048576int % 4 == 0 && 1048576int % 8 == 0) by (compute);
        }
    }
    share
}

#[cfg(test)]
#[path = "hash_workers_tests.rs"]
mod tests;
