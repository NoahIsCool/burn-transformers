//! Exhaustive and property checks of the transpose work partition.
//!
//! The Verus proofs in `verus/perf.rs` establish these invariants for every
//! admissible input; the tests keep them visible to ordinary `cargo test` runs
//! and check the concrete thread choices the proofs leave open.

use proptest::prelude::*;

use super::{PARALLEL_MIN_SCALARS, TILE_EDGE, TransposePlan, transpose_plan};

/// Checks every documented invariant of one plan.
fn assert_partition(rows: usize, columns: usize, available: usize, plan: TransposePlan) {
    assert_eq!(plan.block_columns, columns.min(TILE_EDGE));
    assert!((1..=columns).contains(&plan.band_columns));
    assert!(
        plan.band_columns == columns || plan.band_columns.is_multiple_of(plan.block_columns),
        "{plan:?}"
    );
    assert!(1 <= plan.bands && plan.bands <= plan.threads && plan.threads <= available);
    assert!((plan.bands - 1) * plan.band_columns < columns);
    assert!(columns <= plan.bands * plan.band_columns);
    assert!(rows * plan.band_columns <= rows * columns);
    if rows * columns < PARALLEL_MIN_SCALARS {
        assert_eq!(plan.threads, 1);
    }
}

/// Small matrices stay on the caller's thread in a single band.
#[test]
fn small_matrices_use_one_band() {
    let plan = transpose_plan(1023, 1024, 14);
    assert_eq!(plan.threads, 1);
    assert_eq!(plan.bands, 1);
    assert_eq!(plan.band_columns, 1024);
}

/// Large matrices spread whole tile blocks over the available threads.
#[test]
fn large_matrices_split_whole_blocks_across_threads() {
    let plan = transpose_plan(1024, 1037, 14);
    assert_eq!(plan.block_columns, TILE_EDGE);
    assert_eq!(plan.threads, 14);
    // 33 blocks over 14 threads: three blocks (96 columns) per band, 11 bands.
    assert_eq!(plan.band_columns, 96);
    assert_eq!(plan.bands, 11);
    assert_partition(1024, 1037, 14, plan);
}

/// Narrow matrices never request more threads than they have blocks.
#[test]
fn threads_never_exceed_blocks() {
    let plan = transpose_plan(1 << 20, 40, 64);
    assert_eq!(plan.threads, 2);
    assert_eq!(plan.bands, 2);
    assert_eq!(plan.band_columns, 32);
    assert_partition(1 << 20, 40, 64, plan);
}

/// Every geometry up to 70 x 70 with up to 16 threads satisfies the partition invariants.
#[test]
fn small_geometries_partition_exhaustively() {
    for rows in 1..=70 {
        for columns in 1..=70 {
            for available in 1..=16 {
                assert_partition(
                    rows,
                    columns,
                    available,
                    transpose_plan(rows, columns, available),
                );
            }
        }
    }
}

proptest! {
    /// Large and parallel geometries satisfy the same invariants.
    #[test]
    fn large_geometries_partition_the_columns(
        rows in 1_usize..1 << 16,
        columns in 1_usize..1 << 16,
        available in 1_usize..256,
    ) {
        assert_partition(rows, columns, available, transpose_plan(rows, columns, available));
    }
}
