//! Work partition of the loader's cache-tiled, multi-threaded matrix transpose.
//!
//! Transposing a row-major `[rows, columns]` source writes target row `c` (of
//! length `rows`) from source column `c`. Target rows are independent, so the
//! work is cut along the column axis at two granularities:
//!
//! - a *block* is `b = min(columns, TILE_EDGE)` consecutive columns, the unit
//!   of the cache tile walked by one inner loop;
//! - a *band* is `w` consecutive columns handled by one thread. Every band but
//!   the last holds whole blocks, so the threaded walk visits exactly the tiles
//!   of the serial walk.
//!
//! [`transpose_plan`] chooses `t` threads (one below `PARALLEL_MIN_SCALARS`
//! scalars, where start-up would dominate, otherwise at most the available
//! parallelism and at most one per block), then `w = ⌈⌈columns / b⌉ / t⌉ · b`,
//! capped at `columns`, and `n = ⌈columns / w⌉` bands.
//!
//! # Verified planning
//!
//! Verus checks [`transpose_plan`] in place through `verus/perf.rs` (see
//! `verus/README.md`). For every admissible geometry the proofs establish that
//! the plan equals the formulas above exactly (`t`, `w`, and `n` are pinned, so
//! a plan that silently stayed on one thread would fail verification); that
//! its arithmetic cannot overflow; that `1 <= w <= columns` and `w` is a whole
//! number of blocks unless it is the entire width; that the `n` bands cover the
//! columns exactly, `(n - 1) · w < columns <= n · w`, so every band is nonempty
//! and every column lies in exactly one band; that `n <= t <= available`, so
//! no more threads run than requested; and that one band's target slice,
//! `rows · w` scalars, never exceeds the `rows · columns` scalars of the whole
//! matrix. The tiled copy that consumes the plan is covered by tests, not
//! proofs: exact-permutation property tests on small shapes, and element-wise
//! checks of a ragged `[1031, 1037]` matrix with injected parallelism budgets
//! of 1, 2, 3, 7, 14 and 64 threads, for identity and dtype-converting copies.
//!
//! The module has no crate, Burn, or allocator dependencies so that Verus can
//! check it in isolation.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: the exact rounded-up quotient `⌈dividend / divisor⌉` for `divisor > 0`.
pub open spec fn ceiling(dividend: int, divisor: int) -> int {
    if dividend % divisor == 0 {
        dividend / divisor
    } else {
        dividend / divisor + 1
    }
}

} // verus!

/// Maximum tile edge in scalars, independent of byte width.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const TILE_EDGE: usize = 32;

/// Smallest matrix, in scalars, whose transpose is split across threads.
///
/// Below one mebi-scalar (4 MiB of F32) the copy takes well under a millisecond
/// and thread start-up would dominate; above it the work is memory-bound and
/// scales with the number of cores feeding the memory system.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const PARALLEL_MIN_SCALARS: usize = 1 << 20;

/// How one transpose is split into cache blocks and per-thread column bands.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TransposePlan {
    /// Columns per cache block, `min(columns, TILE_EDGE)`.
    pub(crate) block_columns: usize,
    /// Columns per thread band; a whole number of blocks unless it is every column.
    pub(crate) band_columns: usize,
    /// Number of bands, `⌈columns / band_columns⌉`.
    pub(crate) bands: usize,
    /// Threads to run the bands on; one means the caller's thread alone.
    pub(crate) threads: usize,
}

/// Returns `⌈dividend / divisor⌉` without forming the overflow-prone `dividend + divisor - 1`.
///
/// Verified: the result is exactly the mathematical ceiling, never exceeds
/// `dividend`, and is positive whenever `dividend` is.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(quotient =>
    requires
        divisor > 0,
    ensures
        quotient == ceiling(dividend as int, divisor as int),
        dividend <= quotient * divisor,
        quotient == 0 || (quotient - 1) * divisor < dividend,
        dividend > 0 ==> quotient > 0,
        quotient <= dividend,
))]
fn ceiling_division(dividend: usize, divisor: usize) -> usize {
    #[cfg(verus_keep_ghost)]
    proof! {
        ceiling_division_bounds(dividend, divisor);
    }
    let quotient = dividend / divisor;
    // A named remainder keeps this Verus-checkable (`is_multiple_of` has no spec).
    let remainder = dividend % divisor;
    if remainder == 0 {
        quotient
    } else {
        quotient + 1
    }
}

/// Plans the blocks, bands, and threads of one `[rows, columns]` transpose.
///
/// `available` is the parallelism the caller may use (for example
/// [`std::thread::available_parallelism`]). See the module documentation for
/// the formulas and the verified guarantees.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    requires
        rows >= 1,
        columns >= 1,
        available >= 1,
        rows * columns <= usize::MAX,
    ensures
        plan.block_columns == (if columns < TILE_EDGE { columns } else { TILE_EDGE }),
        plan.threads == (if rows * columns < PARALLEL_MIN_SCALARS {
            1
        } else if available < ceiling(columns as int, plan.block_columns as int) {
            available as int
        } else {
            ceiling(columns as int, plan.block_columns as int)
        }),
        plan.band_columns == (if ceiling(
            ceiling(columns as int, plan.block_columns as int),
            plan.threads as int,
        ) >= ceiling(columns as int, plan.block_columns as int) {
            columns as int
        } else {
            ceiling(ceiling(columns as int, plan.block_columns as int), plan.threads as int)
                * plan.block_columns
        }),
        plan.bands == ceiling(columns as int, plan.band_columns as int),
        1 <= plan.band_columns <= columns,
        plan.band_columns == columns || plan.band_columns % plan.block_columns == 0,
        1 <= plan.bands,
        plan.bands <= plan.threads,
        plan.threads <= available,
        rows * columns < PARALLEL_MIN_SCALARS ==> plan.threads == 1,
        (plan.bands - 1) * plan.band_columns < columns,
        columns <= plan.bands * plan.band_columns,
        rows * plan.band_columns <= rows * columns,
))]
pub(crate) fn transpose_plan(rows: usize, columns: usize, available: usize) -> TransposePlan {
    let block_columns = if columns < TILE_EDGE {
        columns
    } else {
        TILE_EDGE
    };
    let blocks = ceiling_division(columns, block_columns);
    let threads = if rows * columns < PARALLEL_MIN_SCALARS {
        1
    } else if available < blocks {
        available
    } else {
        blocks
    };
    let blocks_per_band = ceiling_division(blocks, threads);
    #[cfg(verus_keep_ghost)]
    proof! {
        band_width_bounds(columns, block_columns, blocks, threads, blocks_per_band);
    }
    let band_columns = if blocks_per_band >= blocks {
        columns
    } else {
        blocks_per_band * block_columns
    };
    let bands = ceiling_division(columns, band_columns);
    #[cfg(verus_keep_ghost)]
    proof! {
        band_count_bounds(rows, columns, band_columns, bands, threads, blocks_per_band >= blocks);
    }
    TransposePlan {
        block_columns,
        band_columns,
        bands,
        threads,
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// Euclidean division facts behind [`ceiling_division`], including the absence of overflow.
proof fn ceiling_division_bounds(dividend: usize, divisor: usize)
    requires
        divisor > 0,
    ensures
        dividend % divisor != 0 ==> dividend / divisor < usize::MAX,
        dividend % divisor == 0 ==> dividend <= (dividend / divisor) * divisor,
        dividend % divisor != 0 ==> dividend <= (dividend / divisor + 1) * divisor,
        dividend / divisor * divisor <= dividend,
        dividend % divisor != 0 ==> dividend / divisor * divisor < dividend,
        dividend % divisor != 0 ==> dividend / divisor + 1 <= dividend,
        dividend / divisor <= dividend,
        dividend % divisor == 0 ==> (dividend / divisor == 0 || (dividend / divisor - 1)
            * divisor < dividend),
        dividend % divisor == 0 && dividend > 0 ==> dividend / divisor > 0,
{
    let q = dividend as int / divisor as int;
    let r = dividend as int % divisor as int;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(dividend as int, divisor as int);
    assert(0 <= r < divisor);
    assert(dividend == q * divisor + r);
    assert(q * divisor >= q) by (nonlinear_arith)
        requires
            divisor >= 1,
            q >= 0,
    ;
    if q == 0 {
        assert(q * divisor == 0) by (nonlinear_arith)
            requires
                q == 0,
        ;
    }
    if r == 0 && q > 0 {
        assert((q - 1) * divisor == q * divisor - divisor) by (nonlinear_arith);
    }
    if r != 0 {
        // A nonzero remainder needs a divisor of at least two, so q <= dividend / 2.
        assert(divisor >= 2);
        assert(q * divisor >= 2 * q) by (nonlinear_arith)
            requires
                divisor >= 2,
                q >= 0,
        ;
        assert((q + 1) * divisor == q * divisor + divisor) by (nonlinear_arith);
    }
}

/// Bounds the band width: whole blocks strictly inside the width, without overflow.
proof fn band_width_bounds(
    columns: usize,
    block_columns: usize,
    blocks: usize,
    threads: usize,
    blocks_per_band: usize,
)
    requires
        block_columns >= 1,
        threads >= 1,
        columns <= blocks * block_columns,
        blocks == 0 || (blocks - 1) * block_columns < columns,
        blocks <= blocks_per_band * threads,
    ensures
        (blocks_per_band * block_columns) % (block_columns as int) == 0,
        blocks_per_band < blocks ==> blocks_per_band * block_columns < columns,
        blocks_per_band < blocks ==> blocks_per_band * block_columns >= 1,
        blocks_per_band < blocks ==> columns <= threads * (blocks_per_band * block_columns),
{
    vstd::arithmetic::div_mod::lemma_mod_multiples_basic(
        blocks_per_band as int,
        block_columns as int,
    );
    if blocks_per_band < blocks {
        assert(columns <= threads * (blocks_per_band * block_columns)) by (nonlinear_arith)
            requires
                columns <= blocks * block_columns,
                blocks <= blocks_per_band * threads,
                block_columns >= 1,
        ;
        assert(blocks_per_band * block_columns <= (blocks - 1) * block_columns)
            by (nonlinear_arith)
            requires
                blocks_per_band <= blocks - 1,
                block_columns >= 1,
        ;
        assert(blocks_per_band >= 1) by (nonlinear_arith)
            requires
                blocks <= blocks_per_band * threads,
                blocks_per_band < blocks,
        ;
        assert(blocks_per_band * block_columns >= 1) by (nonlinear_arith)
            requires
                blocks_per_band >= 1,
                block_columns >= 1,
        ;
    }
}

/// Bounds the band count by the thread count and one band's slice by the matrix.
proof fn band_count_bounds(
    rows: usize,
    columns: usize,
    band_columns: usize,
    bands: usize,
    threads: usize,
    single_band: bool,
)
    requires
        rows >= 1,
        1 <= band_columns <= columns,
        threads >= 1,
        columns <= bands * band_columns,
        bands == 0 || (bands - 1) * band_columns < columns,
        single_band ==> band_columns == columns,
        !single_band ==> columns <= threads * band_columns,
    ensures
        bands <= threads,
        rows * band_columns <= rows * columns,
{
    if single_band {
        assert(bands <= 1) by (nonlinear_arith)
            requires
                bands == 0 || (bands - 1) * band_columns < columns,
                band_columns == columns,
        ;
    } else {
        assert(bands <= threads) by (nonlinear_arith)
            requires
                bands == 0 || (bands - 1) * band_columns < columns,
                columns <= threads * band_columns,
                band_columns >= 1,
        ;
    }
    assert(rows * band_columns <= rows * columns) by (nonlinear_arith)
        requires
            band_columns <= columns,
    ;
}

} // verus!

#[cfg(test)]
#[path = "tiles_tests.rs"]
mod tests;
