//! Pure integer planning for bounded and local attention execution.
//!
//! A score plane holds one element per `[batch, head, query, key]` link, so
//! one query row costs `batch * heads * key_bound` elements. Query chunking
//! evaluates `c` complete rows at a time, bounding every score-like plane by
//! `batch * heads * key_bound * c` while each row still sees its full key
//! support (a storage bound, not an approximation).
//!
//! Chunk `i` covers query rows `[i * c, min((i + 1) * c, Q))`. With
//! `n = ceil(Q / c)` chunks these intervals partition `[0, Q)`: they are
//! non-empty, consecutive, start at zero, and end at `Q`.
//!
//! Local attention additionally narrows each chunk's keys. With bottom-right
//! alignment, query row `t` sits at key position `t + K - Q`; a sliding window
//! `W` lets it see keys `(t + K - Q - W, t + K - Q]`, so the rows `[s, e)`
//! jointly need only keys `[max(0, s + K - Q + 1 - W), e + K - Q)`.
//!
//! # Chunk frames
//!
//! Chunked-local attention compares absolute chunk indices
//! `floor((key_start + p) / C)` of local positions `p in [0, span)`, where
//! `key_start` is the absolute position of the first key (non-zero when a
//! cache dropped earlier tokens). Writing `key_start = q * C + r`, the index is
//! `q + floor((p + r) / C)`, so only the phase `r` matters. When `C >= span`,
//! at most one boundary (local position `C - r`) falls inside the span, and
//! the frame `(width = span, phase = span - (C - r))` reproduces it. Either way
//! `phase < width <= span`, so tensor arithmetic stays below `2 * span` even
//! for chunk widths far beyond a 32-bit backend integer.
//!
//! # Verified planning
//!
//! Every function here is checked in place by Verus through
//! `verus/attention.rs` (see `verus/README.md`). For every input, the proofs
//! establish that no machine arithmetic overflows; that a successful plan has
//! `1 <= c <= Q`, respects both configured limits, takes the largest chunk
//! they admit (`c = min(Q, max_query_tokens, max_score_elements / row)`),
//! reports the exact peak `batch * heads * key_bound * c`, and uses `n` chunks
//! with `(n - 1) * c < Q <= n * c`; that each error variant holds exactly when
//! every earlier check passed and its own condition failed
//! (`chunk_plan_contract`); that chunk ranges are consecutive and partition
//! `[0, Q)`; that a sliding-window key range is non-empty, inside `[0, K)`,
//! and exactly the union of its rows' visible keys; and that a chunk frame
//! preserves every position's absolute chunk index.
//!
//! Bounded execution composes these results. It plans with the key bound
//! [`windowed_key_bound`]`(min(max_query_tokens, Q), W, K)`, which no chunk's
//! window range exceeds, and re-plans every concrete chunk;
//! `lemma_planned_chunk_fits_one_call` proves that such a re-plan always
//! succeeds as one call of exactly the chunk's rows, so the bounded loop's
//! per-chunk check never rejects a planned chunk. Tensor slicing driven by
//! these plans remains covered by the Rust test suite.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// `count` chunks of `size` rows exactly cover `rows` rows: only the last may be short.
pub open spec fn chunks_cover(size: int, count: int, rows: int) -> bool {
    &&& size >= 1
    &&& count >= 1
    &&& (count - 1) * size < rows
    &&& rows <= count * size
}

/// The largest admissible chunk: no more than the rows, the token limit, or the rows the score limit admits.
pub open spec fn largest_chunk(rows: int, max_tokens: int, admitted_rows: int) -> int {
    let by_tokens = if rows <= max_tokens { rows } else { max_tokens };
    if by_tokens <= admitted_rows { by_tokens } else { admitted_rows }
}

/// Both configured limits are non-zero, or execution is unbounded.
pub open spec fn limits_admissible(limits: Option<ChunkLimits>) -> bool {
    match limits {
        Some(limits) => limits.max_query_tokens >= 1 && limits.max_score_elements >= 1,
        None => true,
    }
}

/// Every axis of a `[batch, heads, query, key]` score plane is non-empty.
pub open spec fn axes_nonempty(batch: int, heads: int, query_sequence: int, key_bound: int) -> bool {
    batch >= 1 && heads >= 1 && query_sequence >= 1 && key_bound >= 1
}

/// The complete contract of [`plan_query_chunks`], shared with the lemmas that compose plans.
///
/// A successful plan takes the largest admissible chunk, reports its exact
/// peak, and covers the queries with `ceil(Q / c)` chunks. Each error holds
/// exactly when every earlier check passed and its own condition failed, so
/// errors are reported in the documented order: limits, axes, row size, then
/// the score bound or the unbounded peak.
pub open spec fn chunk_plan_contract(
    batch: usize,
    heads: usize,
    query_sequence: usize,
    key_bound: usize,
    limits: Option<ChunkLimits>,
    result: Result<ChunkGeometry, ChunkPlanError>,
) -> bool {
    let row = batch * heads * key_bound;
    let shaped = axes_nonempty(batch as int, heads as int, query_sequence as int, key_bound as int);
    match result {
        Ok(plan) => {
            &&& limits_admissible(limits)
            &&& shaped
            &&& 1 <= plan.query_chunk_size <= query_sequence
            &&& plan.peak_score_elements == row * plan.query_chunk_size
            &&& chunks_cover(
                plan.query_chunk_size as int,
                plan.chunk_count as int,
                query_sequence as int,
            )
            &&& match limits {
                Some(limits) => {
                    &&& plan.query_chunk_size <= limits.max_query_tokens
                    &&& plan.peak_score_elements <= limits.max_score_elements
                    &&& plan.query_chunk_size == largest_chunk(
                        query_sequence as int,
                        limits.max_query_tokens as int,
                        limits.max_score_elements as int / row,
                    )
                },
                None => plan.query_chunk_size == query_sequence && plan.chunk_count == 1,
            }
        },
        Err(ChunkPlanError::ZeroMaxQueryTokens) =>
            limits is Some && limits->Some_0.max_query_tokens == 0,
        Err(ChunkPlanError::ZeroMaxScoreElements) => {
            &&& limits is Some
            &&& limits->Some_0.max_query_tokens >= 1
            &&& limits->Some_0.max_score_elements == 0
        },
        Err(ChunkPlanError::EmptyAxis) => limits_admissible(limits) && !shaped,
        Err(ChunkPlanError::RowOverflow) => limits_admissible(limits) && shaped && row > usize::MAX,
        Err(ChunkPlanError::RowExceedsLimit { row_elements, max_score_elements }) => {
            &&& limits is Some
            &&& limits_admissible(limits)
            &&& shaped
            &&& max_score_elements == limits->Some_0.max_score_elements
            &&& row_elements == row
            &&& row_elements > max_score_elements
        },
        Err(ChunkPlanError::PeakOverflow) => {
            &&& limits is None
            &&& shaped
            &&& row <= usize::MAX
            &&& row * query_sequence > usize::MAX
        },
    }
}

/// Products of positive factors are positive and dominate each factor.
proof fn lemma_positive_products(batch: int, heads: int, keys: int)
    requires
        batch >= 1,
        heads >= 1,
        keys >= 1,
    ensures
        batch * heads >= batch,
        batch * heads >= 1,
        batch * heads * keys >= batch * heads,
        batch * heads * keys >= 1,
{
    assert(batch * heads >= batch) by (nonlinear_arith)
        requires batch >= 1, heads >= 1;
    assert(batch * heads * keys >= batch * heads) by (nonlinear_arith)
        requires batch * heads >= 1, keys >= 1;
}

/// The ceiling-division chunk count covers the rows and fits beside them.
proof fn lemma_ceiling_cover(rows: int, size: int, quotient: int, remainder: int)
    requires
        rows >= 1,
        size >= 1,
        rows == size * quotient + remainder,
        0 <= remainder < size,
        quotient >= 0,
    ensures
        remainder == 0 ==> chunks_cover(size, quotient, rows),
        remainder > 0 ==> chunks_cover(size, quotient + 1, rows),
        remainder > 0 ==> quotient + 1 <= rows,
{
    if remainder == 0 {
        assert(quotient >= 1) by (nonlinear_arith)
            requires rows >= 1, rows == size * quotient, quotient >= 0;
        assert((quotient - 1) * size < rows) by (nonlinear_arith)
            requires rows == size * quotient, size >= 1;
        assert(rows <= quotient * size) by (nonlinear_arith)
            requires rows == size * quotient;
    } else {
        assert(quotient * size < rows) by (nonlinear_arith)
            requires rows == size * quotient + remainder, remainder > 0;
        assert(rows <= (quotient + 1) * size) by (nonlinear_arith)
            requires rows == size * quotient + remainder, remainder < size;
        assert(quotient <= size * quotient) by (nonlinear_arith)
            requires size >= 1, quotient >= 0;
    }
}

/// A chunk no larger than `max / row` keeps `row * chunk` within `max`, and at least one row fits.
proof fn lemma_bounded_chunk(max: int, row: int, quotient: int, chunk: int)
    requires
        row >= 1,
        row <= max,
        quotient == max / row,
        chunk <= quotient,
        chunk >= 0,
    ensures
        quotient >= 1,
        row * chunk <= max,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(max, row);
    vstd::arithmetic::div_mod::lemma_mod_pos_bound(max, row);
    assert(quotient >= 1) by (nonlinear_arith)
        requires
            max == row * quotient + max % row,
            max % row < row,
            row <= max;
    assert(row * chunk <= row * quotient) by (nonlinear_arith)
        requires chunk <= quotient, row >= 1;
}

/// Key `key` is inside the causal window of the query at aligned position `row + offset`.
pub open spec fn window_visible(row: int, key: int, offset: int, window: int) -> bool {
    &&& 0 <= key
    &&& key <= row + offset
    &&& row + offset < key + window
}

/// Some query row in `[start, end)` sees key `key` through the window.
pub open spec fn visible_to_rows(start: int, end: int, key: int, offset: int, window: int) -> bool {
    exists|row: int| start <= row < end && #[trigger] window_visible(row, key, offset, window)
}

/// Chunk `index < count` starts before the covered rows and, unless last, ends at the next start.
proof fn lemma_chunk_bounds(size: int, count: int, rows: int, index: int)
    requires
        chunks_cover(size, count, rows),
        0 <= index < count,
    ensures
        index * size < rows,
        index * size >= 0,
        (index + 1) * size == index * size + size,
        index + 1 < count ==> (index + 1) * size < rows,
        index + 1 == count ==> rows <= index * size + size,
{
    assert(index * size <= (count - 1) * size) by (nonlinear_arith)
        requires index <= count - 1, size >= 1;
    assert(index * size >= 0) by (nonlinear_arith)
        requires index >= 0, size >= 1;
    assert((index + 1) * size == index * size + size) by (nonlinear_arith);
    if index + 1 < count {
        assert((index + 1) * size <= (count - 1) * size) by (nonlinear_arith)
            requires index + 1 <= count - 1, size >= 1;
    }
    if index + 1 == count {
        assert(index * size + size == count * size) by (nonlinear_arith)
            requires index + 1 == count;
    }
}

/// Absolute chunk index of `position` for chunks of `chunk` positions.
pub open spec fn chunk_of(position: int, chunk: int) -> int {
    position / chunk
}

/// A local position below the first boundary keeps the start's chunk index.
proof fn lemma_same_chunk(start: int, chunk: int, local: int)
    requires
        chunk >= 1,
        start >= 0,
        local >= 0,
        start % chunk + local < chunk,
    ensures
        chunk_of(start + local, chunk) == start / chunk,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(start, chunk);
    vstd::arithmetic::div_mod::lemma_mod_pos_bound(start, chunk);
    vstd::arithmetic::div_mod::lemma_basic_div(start % chunk + local, chunk);
    vstd::arithmetic::div_mod::lemma_hoist_over_denominator(
        start % chunk + local,
        start / chunk,
        chunk as nat,
    );
    assert(start % chunk + local + (start / chunk) * chunk == start + local) by (nonlinear_arith)
        requires start == chunk * (start / chunk) + start % chunk;
}

/// Chunk indices of `start + local` follow the phase-shifted local index.
proof fn lemma_phase_shift(start: int, chunk: int, local: int)
    requires
        chunk >= 1,
        start >= 0,
        local >= 0,
    ensures
        chunk_of(start + local, chunk) == start / chunk + (local + start % chunk) / chunk,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(start, chunk);
    vstd::arithmetic::div_mod::lemma_hoist_over_denominator(
        local + start % chunk,
        start / chunk,
        chunk as nat,
    );
    assert(local + start % chunk + (start / chunk) * chunk == start + local) by (nonlinear_arith)
        requires start == chunk * (start / chunk) + start % chunk;
}

/// A value in `[0, 2 * divisor)` divides to one exactly when it reaches `divisor`.
proof fn lemma_binary_quotient(value: int, divisor: int)
    requires
        divisor >= 1,
        0 <= value < 2 * divisor,
    ensures
        value / divisor == if value < divisor { 0int } else { 1int },
{
    if value < divisor {
        vstd::arithmetic::div_mod::lemma_basic_div(value, divisor);
    } else {
        vstd::arithmetic::div_mod::lemma_basic_div(value - divisor, divisor);
        vstd::arithmetic::div_mod::lemma_hoist_over_denominator(value - divisor, 1, divisor as nat);
    }
}

/// A chunk of a successful plan always re-plans as one call: `validate_chunk` never rejects it.
///
/// Bounded execution plans with a key bound `Kb`, then re-plans every concrete
/// chunk of `rows <= c` query rows and `1 <= keys <= Kb` keys and requires a
/// single chunk. Fewer keys only shrink a row (`b * h * keys <= b * h * Kb`),
/// which admits at least as many rows, so the re-plan succeeds with exactly
/// `rows` rows in one chunk.
pub proof fn lemma_planned_chunk_fits_one_call(
    batch: usize,
    heads: usize,
    query_sequence: usize,
    key_bound: usize,
    limits: Option<ChunkLimits>,
    plan: ChunkGeometry,
    rows: usize,
    keys: usize,
    replan: Result<ChunkGeometry, ChunkPlanError>,
)
    requires
        chunk_plan_contract(batch, heads, query_sequence, key_bound, limits, Ok(plan)),
        1 <= rows <= plan.query_chunk_size,
        1 <= keys <= key_bound,
        chunk_plan_contract(batch, heads, rows, keys, limits, replan),
    ensures
        replan is Ok,
        replan->Ok_0.query_chunk_size == rows,
        replan->Ok_0.chunk_count == 1,
{
    let pair = batch * heads;
    let row = pair * key_bound;
    let chunk_row = pair * keys;
    let chunk = plan.query_chunk_size as int;
    lemma_positive_products(batch as int, heads as int, keys as int);
    assert(chunk_row <= row) by (nonlinear_arith)
        requires keys <= key_bound, pair >= 1, chunk_row == pair * keys, row == pair * key_bound;
    assert(row <= row * chunk) by (nonlinear_arith)
        requires chunk >= 1, row >= 1;
    assert(chunk_row * rows <= row * chunk) by (nonlinear_arith)
        requires chunk_row <= row, rows <= chunk, chunk_row >= 0, rows >= 0;
    match limits {
        None => {
            assert(chunk_row * rows <= row * query_sequence) by (nonlinear_arith)
                requires chunk_row <= row, rows <= chunk, chunk <= query_sequence, chunk_row >= 0,
                    rows >= 0;
        },
        Some(bounds) => {
            let max = bounds.max_score_elements as int;
            // Rows admitted by the plan's row size are admitted by the smaller chunk row.
            vstd::arithmetic::div_mod::lemma_div_is_ordered_by_denominator(max, chunk_row, row);
            assert(chunk <= max / row);
            assert(rows <= max / chunk_row);
            assert(largest_chunk(rows as int, bounds.max_query_tokens as int, max / chunk_row)
                == rows);
        },
    }
    if let Ok(replanned) = replan {
        let count = replanned.chunk_count as int;
        assert(count == 1) by (nonlinear_arith)
            requires
                count >= 1,
                (count - 1) * rows < rows,
                rows >= 1;
    }
}

} // verus!

/// Hard limits of a bounded (query-chunked) attention plan.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkLimits {
    /// Upper bound on query rows evaluated in one chunk.
    pub max_query_tokens: usize,
    /// Upper bound on elements in any score-like chunk plane.
    pub max_score_elements: usize,
}

/// Query-chunk geometry produced by [`plan_query_chunks`].
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkGeometry {
    /// Query rows per chunk; the final chunk may be shorter.
    pub query_chunk_size: usize,
    /// Number of chunks, `ceil(query_sequence / query_chunk_size)`.
    pub chunk_count: usize,
    /// Elements in the largest chunk plane: `batch * heads * key_bound * chunk`.
    pub peak_score_elements: usize,
}

/// Why a chunk plan could not be formed.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkPlanError {
    /// The configured query-row limit is zero.
    ZeroMaxQueryTokens,
    /// The configured score-element limit is zero.
    ZeroMaxScoreElements,
    /// Batch, heads, query sequence, or key bound is zero.
    EmptyAxis,
    /// `batch * heads * key_bound` does not fit in `usize`.
    RowOverflow,
    /// One query row alone exceeds the configured score-element limit.
    RowExceedsLimit {
        /// Elements in one query row, `batch * heads * key_bound`.
        row_elements: usize,
        /// Configured score-element limit.
        max_score_elements: usize,
    },
    /// Unbounded planning of the whole plane overflows `usize`.
    PeakOverflow,
}

/// Plans query chunks whose score planes respect `limits` (`None` = one chunk).
///
/// Bounded plans take the largest chunk allowed by both limits and the query
/// length. Limits are checked before shapes, matching configuration-first
/// error reporting.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        chunk_plan_contract(batch, heads, query_sequence, key_bound, limits, result),
))]
pub fn plan_query_chunks(
    batch: usize,
    heads: usize,
    query_sequence: usize,
    key_bound: usize,
    limits: Option<ChunkLimits>,
) -> Result<ChunkGeometry, ChunkPlanError> {
    if let Some(limits) = limits {
        if limits.max_query_tokens == 0 {
            return Err(ChunkPlanError::ZeroMaxQueryTokens);
        }
        if limits.max_score_elements == 0 {
            return Err(ChunkPlanError::ZeroMaxScoreElements);
        }
    }
    if batch == 0 || heads == 0 || query_sequence == 0 || key_bound == 0 {
        return Err(ChunkPlanError::EmptyAxis);
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_positive_products(batch as int, heads as int, key_bound as int);
    }
    let Some(plane_rows) = batch.checked_mul(heads) else {
        return Err(ChunkPlanError::RowOverflow);
    };
    let Some(row_elements) = plane_rows.checked_mul(key_bound) else {
        return Err(ChunkPlanError::RowOverflow);
    };
    let (query_chunk_size, peak_score_elements) = match limits {
        None => {
            let Some(peak) = row_elements.checked_mul(query_sequence) else {
                return Err(ChunkPlanError::PeakOverflow);
            };
            (query_sequence, peak)
        }
        Some(limits) => {
            if row_elements > limits.max_score_elements {
                return Err(ChunkPlanError::RowExceedsLimit {
                    row_elements,
                    max_score_elements: limits.max_score_elements,
                });
            }
            // Largest chunk permitted by both limits; at least one row fits.
            let rows_within_score_limit = limits.max_score_elements / row_elements;
            let chunk = query_sequence
                .min(limits.max_query_tokens)
                .min(rows_within_score_limit);
            #[cfg(verus_keep_ghost)]
            proof! {
                lemma_bounded_chunk(
                    limits.max_score_elements as int,
                    row_elements as int,
                    rows_within_score_limit as int,
                    chunk as int,
                );
            }
            // Cannot overflow: chunk * row <= (max / row) * row <= max.
            (chunk, row_elements * chunk)
        }
    };
    let whole_chunks = query_sequence / query_chunk_size;
    let remainder = query_sequence % query_chunk_size;
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
            query_sequence as int,
            query_chunk_size as int,
        );
        vstd::arithmetic::div_mod::lemma_mod_pos_bound(
            query_sequence as int,
            query_chunk_size as int,
        );
        lemma_ceiling_cover(
            query_sequence as int,
            query_chunk_size as int,
            whole_chunks as int,
            remainder as int,
        );
    }
    let chunk_count = if remainder == 0 {
        whole_chunks
    } else {
        whole_chunks + 1
    };
    Ok(ChunkGeometry {
        query_chunk_size,
        chunk_count,
        peak_score_elements,
    })
}

/// Returns the query rows `[start, end)` of chunk `index`, or `None` past the last chunk.
///
/// The geometry must cover the rows (as every successful [`plan_query_chunks`]
/// result does). Consecutive chunks share boundaries (`end` of chunk `i` is
/// `start` of chunk `i + 1`), the first starts at zero, and the last ends at
/// `query_sequence`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        chunks_cover(query_chunk_size as int, chunk_count as int, query_sequence as int),
    ensures
        match result {
            Some(range) => {
                &&& index < chunk_count
                &&& range.0 == index * query_chunk_size
                &&& range.0 < range.1
                &&& range.1 <= query_sequence
                &&& range.1 - range.0 <= query_chunk_size
                &&& index + 1 < chunk_count ==> range.1 == (index + 1) * query_chunk_size
                &&& index + 1 == chunk_count ==> range.1 == query_sequence
            },
            None => index >= chunk_count,
        }
))]
pub fn query_chunk_range(
    query_chunk_size: usize,
    chunk_count: usize,
    query_sequence: usize,
    index: usize,
) -> Option<(usize, usize)> {
    if index >= chunk_count {
        return None;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_chunk_bounds(
            query_chunk_size as int,
            chunk_count as int,
            query_sequence as int,
            index as int,
        );
    }
    let start = index * query_chunk_size;
    let end = start + query_chunk_size.min(query_sequence - start);
    Some((start, end))
}

/// Returns the keys `[start, end)` visible to aligned query rows `[query_start, query_end)`
/// under a causal sliding window of `window` positions.
///
/// Requires bottom-right alignment with `query_sequence <= key_sequence`;
/// returns `None` for empty or out-of-bounds query ranges and a zero window.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Some(range) => {
                let offset = key_sequence - query_sequence;
                &&& query_start < query_end <= query_sequence <= key_sequence
                &&& window >= 1
                &&& range.0 < range.1 <= key_sequence
                &&& range.1 == query_end + offset
                &&& range.1 - range.0 <= (query_end - query_start) + window - 1
                &&& range.0 == if query_start + offset + 1 >= window {
                    query_start + offset + 1 - window
                } else {
                    0
                }
                &&& forall|row: int, key: int|
                    query_start <= row < query_end
                        && #[trigger] window_visible(row, key, offset as int, window as int)
                        ==> range.0 <= key < range.1
                &&& forall|key: int| range.0 <= key < range.1 ==> #[trigger] visible_to_rows(
                    query_start as int,
                    query_end as int,
                    key,
                    offset as int,
                    window as int,
                )
            },
            None => !(query_start < query_end && query_end <= query_sequence
                && query_sequence <= key_sequence && window >= 1),
        }
))]
pub fn sliding_window_key_range(
    query_sequence: usize,
    key_sequence: usize,
    query_start: usize,
    query_end: usize,
    window: usize,
) -> Option<(usize, usize)> {
    if query_start >= query_end
        || query_end > query_sequence
        || query_sequence > key_sequence
        || window == 0
    {
        return None;
    }
    let offset = key_sequence - query_sequence;
    // First aligned position minus `window - 1`, clamped at key zero.
    let first_position = query_start + offset;
    let key_start = first_position - first_position.min(window - 1);
    let key_end = query_end + offset;
    #[cfg(verus_keep_ghost)]
    proof! {
        // Exactness: key `j` of the range is visible to row `max(query_start, j - offset)`.
        assert forall|key: int| key_start <= key < key_end implies #[trigger] visible_to_rows(
            query_start as int,
            query_end as int,
            key,
            offset as int,
            window as int,
        ) by {
            let row = if key - offset >= query_start { key - offset } else { query_start as int };
            assert(window_visible(row, key, offset as int, window as int));
        }
    }
    Some((key_start, key_end))
}

/// Query rows one chunk may hold under a row limit: `min(max_query_tokens, query_sequence)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rows =>
    ensures
        rows == if max_query_tokens < query_sequence { max_query_tokens } else { query_sequence },
))]
pub const fn capped_query_rows(max_query_tokens: usize, query_sequence: usize) -> usize {
    if max_query_tokens < query_sequence {
        max_query_tokens
    } else {
        query_sequence
    }
}

/// Key bound of causal sliding-window chunks of at most `query_cap` rows.
///
/// Returns `min(query_cap + window - 1, key_sequence)`, computed without
/// overflow. Every [`sliding_window_key_range`] of at most `query_cap` rows is
/// no wider (its width is at most `rows + window - 1` and it lies inside
/// `[0, key_sequence)`), so planning with this bound covers every chunk's keys.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bound =>
    requires
        window >= 1,
    ensures
        bound == if query_cap + window - 1 <= key_sequence {
            query_cap + window - 1
        } else {
            key_sequence as int
        },
))]
pub fn windowed_key_bound(query_cap: usize, window: usize, key_sequence: usize) -> usize {
    query_cap.saturating_add(window - 1).min(key_sequence)
}

/// Re-expresses chunk membership of positions `key_start + p`, `p in [0, span)`, locally.
///
/// Returns `None` when the whole span lies inside one chunk (the rule then
/// excludes nothing); otherwise `(width, phase)` with `phase < width <= span`
/// and `floor((key_start + p) / chunk) = floor(key_start / chunk) +
/// floor((p + phase) / width)` for every `p` in the span. Two local positions
/// therefore share an absolute chunk exactly when their
/// `floor((p + phase) / width)` agree. See the module documentation.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        chunk >= 1,
        span >= 1,
    ensures
        match result {
            Some(frame) => {
                &&& 1 <= frame.0 <= span
                &&& frame.1 < frame.0
                &&& forall|local: int| 0 <= local < span ==> #[trigger] chunk_of(
                    key_start + local,
                    chunk as int,
                ) == key_start / chunk + (local + frame.1) / (frame.0 as int)
            },
            None => forall|local: int| 0 <= local < span ==> #[trigger] chunk_of(
                key_start + local,
                chunk as int,
            ) == key_start / chunk,
        }
))]
pub fn chunk_frame(chunk: usize, key_start: usize, span: usize) -> Option<(usize, usize)> {
    let phase = key_start % chunk;
    if chunk < span {
        #[cfg(verus_keep_ghost)]
        proof! {
            assert forall|local: int| 0 <= local < span implies #[trigger] chunk_of(
                key_start + local,
                chunk as int,
            ) == key_start / chunk + (local + phase) / (chunk as int) by {
                lemma_phase_shift(key_start as int, chunk as int, local);
            }
        }
        return Some((chunk, phase));
    }
    // Wider chunks cross at most one boundary, at local position `chunk - phase`.
    let boundary = chunk - phase;
    if boundary >= span {
        #[cfg(verus_keep_ghost)]
        proof! {
            assert forall|local: int| 0 <= local < span implies #[trigger] chunk_of(
                key_start + local,
                chunk as int,
            ) == key_start / chunk by {
                lemma_same_chunk(key_start as int, chunk as int, local);
            }
        }
        return None;
    }
    let shifted_phase = span - boundary;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|local: int| 0 <= local < span implies #[trigger] chunk_of(
            key_start + local,
            chunk as int,
        ) == key_start / chunk + (local + shifted_phase) / (span as int) by {
            vstd::arithmetic::div_mod::lemma_mod_pos_bound(key_start as int, chunk as int);
            lemma_phase_shift(key_start as int, chunk as int, local);
            lemma_binary_quotient(local + phase, chunk as int);
            lemma_binary_quotient(local + shifted_phase, span as int);
        }
    }
    Some((span, shifted_phase))
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
