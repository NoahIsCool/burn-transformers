//! Dependency-free integer planning and row bookkeeping for the Mamba-2 mixer.
//!
//! The mixer's fused input projection emits `[z, x, B, C, dt]` with widths
//! `I = heads * head_dim`, `I`, `G * N`, `G * N`, and `heads`, so the depthwise
//! convolution sees `I + 2 * G * N` channels and the projection is
//! `2 * I + 2 * G * N + heads` wide. Batched rows are right-padded: row `b`
//! holds `counts[b]` valid tokens, its next convolution history is joined
//! columns `counts[b] .. counts[b] + K - 1`, and rows are grouped by execution
//! mode with a permutation that restores the batch order afterwards. The SSD
//! scan walks `ceil(L / T)` chunks of `T` tokens in blocks whose score planes
//! fit an element budget.
//!
//! # Verified planning
//!
//! Every function here is checked in place by Verus through
//! `verus/mixers.rs`. The proofs establish, for every admissible input, that
//! no arithmetic overflows and that each result is exact: widths equal their
//! formulas or name the first overflowing step, the `[z | xBC | dt]` and
//! `[x | B | C]` channel slices tile their layouts, and normalization groups
//! and head groups split their axes exactly; geometry and row-count checks
//! accept exactly the admissible inputs; validity flags and gather indices
//! follow their closed forms or are rejected exactly when an index exceeds
//! `i64`; the mode grouping partitions the rows by effective mode, in
//! ascending order, and its restore permutation is a two-sided inverse; and
//! the scan blocks tile the tokens in chunk-aligned blocks that respect the
//! plane budget, with only the last block padded, by less than one chunk.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

use crate::delta_net::plan::ceiling_division;

/// Selects the transparent recurrent loop or the chunked SSD scan.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mamba2Mode {
    /// One explicit state update per token (the reference decode step).
    Recurrent,
    /// Masked semiseparable scan inside bounded chunks (the reference prefill).
    ChunkedPrefill,
}

/// One right-padded batch row of [`super::Mamba2::forward_rows`].
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mamba2Row {
    /// Number of valid tokens at the start of the row; zero leaves its state unchanged.
    pub tokens: usize,
    /// Execution mode of the row's segment.
    pub mode: Mamba2Mode,
}

#[cfg(verus_keep_ghost)]
verus! {

/// Mode of the first active row at or after `start`, or chunked prefill when none is active.
pub open spec fn fallback_from(rows: Seq<Mamba2Row>, start: int) -> Mamba2Mode
    decreases rows.len() - start,
{
    if start < 0 || start >= rows.len() {
        Mamba2Mode::ChunkedPrefill
    } else if rows[start].tokens > 0 {
        rows[start].mode
    } else {
        fallback_from(rows, start + 1)
    }
}

/// Mode a row runs under: its own when active, otherwise the first active row's.
///
/// Inactive rows are identity transitions under either mode, so they join an
/// existing group instead of forcing another pass.
pub open spec fn effective_mode(rows: Seq<Mamba2Row>, row: int) -> Mamba2Mode {
    if rows[row].tokens > 0 { rows[row].mode } else { fallback_from(rows, 0) }
}

/// Batch row at position `p` of the concatenation `first ++ second`.
pub open spec fn grouped_row(first: Seq<usize>, second: Seq<usize>, p: int) -> int {
    if p < first.len() { first[p] as int } else { second[p - first.len()] as int }
}

/// Joined history column `counts[row] + offset` read for a row's next convolution state.
pub open spec fn row_column(counts: Seq<usize>, row: int, offset: int) -> int {
    counts[row] + offset
}

/// `a * b` saturated to the largest `usize`.
pub open spec fn saturating_product(a: int, b: int) -> int {
    if a * b > usize::MAX { usize::MAX as int } else { a * b }
}

/// `clamp(admitted, 1, chunks)`: at least one chunk and at most every chunk.
pub open spec fn clamped_chunks(admitted: int, chunks: int) -> int {
    if admitted < 1 { 1 } else if admitted > chunks { chunks } else { admitted }
}

/// Elements of one chunk's `[batch, heads, T, T]` score plane, saturated and at least one.
pub open spec fn scan_plane(batch: int, heads: int, chunk_size: int) -> int {
    let plane = saturating_product(
        saturating_product(saturating_product(batch, heads), chunk_size),
        chunk_size,
    );
    if plane < 1 { 1 } else { plane }
}

} // verus!

/// First overflowing step of a projection-width computation.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WidthOverflow {
    /// `heads * head_dim` overflows.
    Intermediate,
    /// `groups * state_size * 2` overflows.
    StateChannels,
    /// The convolution width `I + 2 * G * N` overflows.
    Convolution,
    /// The projection width `I + convolution + heads` overflows.
    Projection,
}

/// Returns the convolution width `heads * head_dim + 2 * groups * state_size`.
///
/// Steps run in the order of `Mamba2Config::convolution_channels`: the B/C
/// width, the intermediate width, then their sum. The proof shows the result
/// is the exact width, and that an error names the first step whose exact
/// value exceeds `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    ensures
        match width {
            Ok(channels) => channels == heads * head_dim + groups * state_size * 2,
            Err(WidthOverflow::StateChannels) => groups * state_size * 2 > usize::MAX,
            Err(WidthOverflow::Intermediate) => {
                &&& groups * state_size * 2 <= usize::MAX
                &&& heads * head_dim > usize::MAX
            },
            Err(WidthOverflow::Convolution) => {
                &&& groups * state_size * 2 <= usize::MAX
                &&& heads * head_dim <= usize::MAX
                &&& heads * head_dim + groups * state_size * 2 > usize::MAX
            },
            Err(WidthOverflow::Projection) => false,
        },
))]
pub(super) fn convolution_width(
    heads: usize,
    head_dim: usize,
    groups: usize,
    state_size: usize,
) -> Result<usize, WidthOverflow> {
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_products_nonnegative(heads as int, head_dim as int, groups as int, state_size as int);
    }
    let state_channels = match groups.checked_mul(state_size) {
        Some(width) => match width.checked_mul(2) {
            Some(channels) => channels,
            None => return Err(WidthOverflow::StateChannels),
        },
        None => return Err(WidthOverflow::StateChannels),
    };
    let Some(intermediate) = heads.checked_mul(head_dim) else {
        return Err(WidthOverflow::Intermediate);
    };
    match intermediate.checked_add(state_channels) {
        Some(channels) => Ok(channels),
        None => Err(WidthOverflow::Convolution),
    }
}

/// Returns the fused projection width `2 * heads * head_dim + 2 * groups * state_size + heads`.
///
/// Steps run in the order of `Mamba2Config::projection_size`: the
/// intermediate width, the convolution width (with its own order), then the
/// two sums. The proof shows the result is the exact width, and that an
/// error names the first step whose exact value exceeds `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    ensures
        match width {
            Ok(projection) => projection
                == 2 * (heads * head_dim) + groups * state_size * 2 + heads,
            Err(WidthOverflow::Intermediate) => heads * head_dim > usize::MAX,
            Err(WidthOverflow::StateChannels) => {
                &&& heads * head_dim <= usize::MAX
                &&& groups * state_size * 2 > usize::MAX
            },
            Err(WidthOverflow::Convolution) => {
                &&& heads * head_dim <= usize::MAX
                &&& groups * state_size * 2 <= usize::MAX
                &&& heads * head_dim + groups * state_size * 2 > usize::MAX
            },
            Err(WidthOverflow::Projection) => {
                &&& heads * head_dim + groups * state_size * 2 <= usize::MAX
                &&& 2 * (heads * head_dim) + groups * state_size * 2 + heads > usize::MAX
            },
        },
))]
pub(super) fn projection_width(
    heads: usize,
    head_dim: usize,
    groups: usize,
    state_size: usize,
) -> Result<usize, WidthOverflow> {
    let Some(intermediate) = heads.checked_mul(head_dim) else {
        return Err(WidthOverflow::Intermediate);
    };
    let channels = convolution_width(heads, head_dim, groups, state_size)?;
    match intermediate.checked_add(channels) {
        Some(width) => match width.checked_add(heads) {
            Some(projection) => Ok(projection),
            None => Err(WidthOverflow::Projection),
        },
        None => Err(WidthOverflow::Projection),
    }
}

/// Channel offsets of the fused projection `[z | xBC | dt]`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ProjectionSlices {
    /// Gate channels `0..I` and, after them, the convolution input starting at `I = heads * head_dim`.
    pub(super) intermediate: usize,
    /// First time-step channel, `I + convolution_channels`; `dt` spans `heads` channels.
    pub(super) dt_start: usize,
}

/// Splits a mixer's fused projection into its gate, convolution-input, and time-step slices.
///
/// The proof shows the slices `0..I`, `I..I + channels`, and
/// `I + channels..I + channels + heads` (with `I = heads * head_dim`) are
/// consecutive and end exactly at the projection width, so they tile the
/// projection with no gap or overlap. With the convolution width of
/// [`convolution_width`] that end is the width [`projection_width`] accepted,
/// so a mixer built from a validated configuration meets the precondition;
/// the arithmetic is the mixer's own, so behavior is unchanged.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(slices =>
    requires
        heads * head_dim + convolution_channels + heads <= usize::MAX,
    ensures
        slices.intermediate == heads * head_dim,
        slices.dt_start == slices.intermediate + convolution_channels,
        slices.dt_start + heads == heads * head_dim + convolution_channels + heads,
))]
pub(super) fn projection_slices(
    heads: usize,
    head_dim: usize,
    convolution_channels: usize,
) -> ProjectionSlices {
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(heads * head_dim >= 0) by (nonlinear_arith);
    }
    let intermediate = heads * head_dim;
    ProjectionSlices {
        intermediate,
        dt_start: intermediate + convolution_channels,
    }
}

/// Channel offsets of the convolved `[x | B | C]` activations.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ConvolutionSlices {
    /// Width `G * N` of each of the group-shared `B` and `C` slices.
    pub(super) group_state_width: usize,
    /// First `C` channel, `I + G * N`; `B` spans `I..I + G * N`.
    pub(super) c_start: usize,
}

/// Splits a mixer's convolved activations into its `x`, `B`, and `C` slices.
///
/// The proof shows the slices `0..I`, `I..I + G * N`, and
/// `I + G * N..I + 2 * G * N` are consecutive and end exactly at the
/// convolution width `I + 2 * G * N` of [`convolution_width`], so they tile
/// the convolution output. A mixer built from a validated configuration meets
/// the precondition; the arithmetic is the mixer's own, so behavior is
/// unchanged.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(slices =>
    requires
        intermediate + groups * state_size * 2 <= usize::MAX,
    ensures
        slices.group_state_width == groups * state_size,
        slices.c_start == intermediate + slices.group_state_width,
        slices.c_start + slices.group_state_width == intermediate + groups * state_size * 2,
))]
pub(super) fn convolution_slices(
    intermediate: usize,
    groups: usize,
    state_size: usize,
) -> ConvolutionSlices {
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(groups * state_size >= 0) by (nonlinear_arith);
    }
    let group_state_width = groups * state_size;
    ConvolutionSlices {
        group_state_width,
        c_start: intermediate + group_state_width,
    }
}

/// Returns the width of each of `groups` equal contiguous groups of `total`, if they exist.
///
/// The proof shows the result is `Some(w)` with `w * groups == total`
/// exactly when `groups > 0` divides `total`, and `None` otherwise, so a
/// `[groups, w]` reshape of the grouped axis is exact.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    ensures
        match width {
            Some(width) => groups > 0 && width * groups == total,
            None => groups == 0 || total % groups != 0,
        },
))]
pub(super) fn group_width(total: usize, groups: usize) -> Option<usize> {
    if groups == 0 || !total.is_multiple_of(groups) {
        return None;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_exact_quotient(total as int, groups as int);
    }
    Some(total / groups)
}

/// Returns the heads per group-shared `B`/`C` group.
///
/// The proof shows `per_group * groups == heads`, so broadcasting each group
/// over `per_group` contiguous heads covers every head exactly once. The
/// precondition is the divisibility that [`check_mamba2_geometry`] accepts
/// and the chunked scan checks again.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(per_group =>
    requires
        groups > 0,
        heads % groups == 0,
    ensures
        per_group * groups == heads,
))]
pub(super) fn heads_per_group(heads: usize, groups: usize) -> usize {
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_exact_quotient(heads as int, groups as int);
    }
    heads / groups
}

/// First violated integer constraint of a Mamba-2 configuration.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mamba2GeometryViolation {
    /// A width or the chunk size is zero, or the kernel is below two.
    ZeroWidth,
    /// The groups do not divide the heads.
    Groups,
}

/// Checks non-zero widths, a kernel of at least two, and groups dividing the heads.
///
/// The proof shows acceptance exactly when every width and the chunk size
/// are non-zero, the kernel is at least two, and `groups` divides `heads`,
/// with width violations reported first.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(checked =>
    ensures
        match checked {
            Ok(()) => {
                &&& hidden > 0 && heads > 0 && head_dim > 0 && state_size > 0 && groups > 0
                &&& kernel >= 2 && chunk_size > 0
                &&& heads % groups == 0
            },
            Err(Mamba2GeometryViolation::ZeroWidth) => {
                hidden == 0 || heads == 0 || head_dim == 0 || state_size == 0 || groups == 0
                    || kernel < 2 || chunk_size == 0
            },
            Err(Mamba2GeometryViolation::Groups) => {
                &&& hidden > 0 && heads > 0 && head_dim > 0 && state_size > 0 && groups > 0
                &&& kernel >= 2 && chunk_size > 0
                &&& heads % groups != 0
            },
        },
))]
pub(super) fn check_mamba2_geometry(
    hidden: usize,
    heads: usize,
    head_dim: usize,
    state_size: usize,
    groups: usize,
    kernel: usize,
    chunk_size: usize,
) -> Result<(), Mamba2GeometryViolation> {
    if hidden == 0
        || heads == 0
        || head_dim == 0
        || state_size == 0
        || groups == 0
        || kernel < 2
        || chunk_size == 0
    {
        return Err(Mamba2GeometryViolation::ZeroWidth);
    }
    if !heads.is_multiple_of(groups) {
        return Err(Mamba2GeometryViolation::Groups);
    }
    Ok(())
}

/// Returns whether `counts` has one entry per batch row, none above the padded width.
///
/// The proof shows the result is exactly `counts.len() == batch` and
/// `counts[b] <= sequence` for every row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fits =>
    ensures
        fits == (counts.len() == batch
            && forall|b: int| 0 <= b < counts.len() ==> #[trigger] counts[b] <= sequence),
))]
pub(super) fn row_counts_fit(counts: &[usize], batch: usize, sequence: usize) -> bool {
    if counts.len() != batch {
        return false;
    }
    let mut row = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            forall|b: int| 0 <= b < row ==> #[trigger] counts[b] <= sequence,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] > sequence {
            return false;
        }
        row += 1;
    }
    true
}

/// Returns whether every row holds exactly `sequence` valid tokens.
///
/// The proof shows the result is exactly `counts[b] == sequence` for all rows.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(full =>
    ensures
        full == forall|b: int| 0 <= b < counts.len() ==> #[trigger] counts[b] == sequence,
))]
pub(super) fn rows_full(counts: &[usize], sequence: usize) -> bool {
    let mut row = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            forall|b: int| 0 <= b < row ==> #[trigger] counts[b] == sequence,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] != sequence {
            return false;
        }
        row += 1;
    }
    true
}

/// Returns row-major `[rows, sequence]` validity of right-padded rows.
///
/// The proof shows entry `k` is `true` exactly when its column
/// `k % sequence` is below its row's count `counts[k / sequence]`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(flags =>
    requires
        counts.len() * sequence <= usize::MAX,
    ensures
        flags.len() == counts.len() * sequence,
        forall|k: int| 0 <= k < flags.len() ==>
            #[trigger] flags[k] == ((k % (sequence as int)) < counts[k / (sequence as int)]),
))]
pub(super) fn right_padded_flags(counts: &[usize], sequence: usize) -> Vec<bool> {
    let mut flags: Vec<bool> = Vec::with_capacity(counts.len() * sequence);
    let mut row = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            counts.len() * sequence <= usize::MAX,
            flags.len() == row * sequence,
            forall|k: int| 0 <= k < flags.len() ==>
                #[trigger] flags[k] == ((k % (sequence as int)) < counts[k / (sequence as int)]),
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        let mut column = 0usize;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                row < counts.len(),
                column <= sequence,
                flags.len() == row * sequence + column,
                forall|k: int| 0 <= k < flags.len() ==>
                    #[trigger] flags[k] == ((k % (sequence as int)) < counts[k / (sequence as int)]),
            decreases sequence - column,
        ))]
        while column < sequence {
            #[cfg(verus_keep_ghost)]
            proof! {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
                    (row * sequence + column) as int,
                    sequence as int,
                    row as int,
                    column as int,
                );
            }
            flags.push(column < counts[row]);
            column += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row + 1) * sequence == row * sequence + sequence) by (nonlinear_arith);
        }
        row += 1;
    }
    flags
}

/// Returns the joined history columns `counts[b] .. counts[b] + width` of every row as `i64` indices.
///
/// The proof shows entry `k` is `counts[k / width] + k % width` (row-major,
/// `width` columns per row), or that `None` is returned exactly because some
/// such column exceeds `i64::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(starts =>
    requires
        counts.len() * width <= usize::MAX,
        forall|b: int| 0 <= b < counts.len() ==> #[trigger] counts[b] + width <= usize::MAX,
    ensures
        match starts {
            Some(starts) => {
                &&& starts.len() == counts.len() * width
                &&& forall|k: int| 0 <= k < starts.len() ==>
                    #[trigger] starts[k] == counts[k / width as int] + k % width as int
            },
            None => exists|b: int, offset: int|
                0 <= b < counts.len() && 0 <= offset < width
                    && #[trigger] row_column(counts@, b, offset) > i64::MAX,
        },
))]
pub(super) fn row_end_columns(counts: &[usize], width: usize) -> Option<Vec<i64>> {
    let mut starts: Vec<i64> = Vec::with_capacity(counts.len() * width);
    let mut row = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            counts.len() * width <= usize::MAX,
            forall|b: int| 0 <= b < counts.len() ==> #[trigger] counts[b] + width <= usize::MAX,
            starts.len() == row * width,
            forall|k: int| 0 <= k < starts.len() ==>
                #[trigger] starts[k] == counts[k / width as int] + k % width as int,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        let count = counts[row];
        let mut offset = 0usize;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                row < counts.len(),
                count == counts[row as int],
                count + width <= usize::MAX,
                offset <= width,
                starts.len() == row * width + offset,
                forall|k: int| 0 <= k < starts.len() ==>
                    #[trigger] starts[k] == counts[k / width as int] + k % width as int,
            decreases width - offset,
        ))]
        while offset < width {
            let column = count + offset;
            if column as u64 > i64::MAX as u64 {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(row_column(counts@, row as int, offset as int) > i64::MAX);
                }
                return None;
            }
            #[cfg(verus_keep_ghost)]
            proof! {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
                    (row * width + offset) as int,
                    width as int,
                    row as int,
                    offset as int,
                );
            }
            starts.push(column as i64);
            offset += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row + 1) * width == row * width + width) by (nonlinear_arith);
        }
        row += 1;
    }
    Some(starts)
}

/// Returns the joined columns `t + 1 .. t + 1 + width` read after every token `t < sequence`.
///
/// The proof shows entry `k` is `k / width + 1 + k % width` (token-major), or
/// that `None` is returned exactly because the largest such column,
/// `sequence + width - 1`, exceeds `i64::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(columns =>
    requires
        sequence + width <= usize::MAX,
        sequence * width <= usize::MAX,
    ensures
        match columns {
            Some(columns) => {
                &&& columns.len() == sequence * width
                &&& forall|k: int| 0 <= k < columns.len() ==>
                    #[trigger] columns[k] == k / width as int + 1 + k % width as int
            },
            None => sequence > 0 && width > 0 && sequence + width - 1 > i64::MAX,
        },
))]
pub(super) fn window_columns(sequence: usize, width: usize) -> Option<Vec<i64>> {
    let mut columns: Vec<i64> = Vec::with_capacity(sequence * width);
    let mut token = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            token <= sequence,
            sequence + width <= usize::MAX,
            sequence * width <= usize::MAX,
            columns.len() == token * width,
            forall|k: int| 0 <= k < columns.len() ==>
                #[trigger] columns[k] == k / width as int + 1 + k % width as int,
        decreases sequence - token,
    ))]
    while token < sequence {
        let mut offset = 0usize;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                token < sequence,
                sequence + width <= usize::MAX,
                offset <= width,
                columns.len() == token * width + offset,
                forall|k: int| 0 <= k < columns.len() ==>
                    #[trigger] columns[k] == k / width as int + 1 + k % width as int,
            decreases width - offset,
        ))]
        while offset < width {
            let column = token + 1 + offset;
            if column as u64 > i64::MAX as u64 {
                return None;
            }
            #[cfg(verus_keep_ghost)]
            proof! {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
                    (token * width + offset) as int,
                    width as int,
                    token as int,
                    offset as int,
                );
            }
            columns.push(column as i64);
            offset += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((token + 1) * width == token * width + width) by (nonlinear_arith);
        }
        token += 1;
    }
    Some(columns)
}

/// Converts batch-row indices to the `i64` values Burn index tensors use.
///
/// The proof shows every value is preserved exactly, or that `None` is
/// returned exactly because some index exceeds `i64::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(signed =>
    ensures
        match signed {
            Some(values) => {
                &&& values.len() == rows.len()
                &&& forall|k: int| 0 <= k < rows.len() ==> #[trigger] values[k] == rows[k]
            },
            None => exists|k: int| 0 <= k < rows.len() && #[trigger] rows[k] > i64::MAX,
        },
))]
pub(super) fn signed_rows(rows: &[usize]) -> Option<Vec<i64>> {
    let mut values: Vec<i64> = Vec::with_capacity(rows.len());
    let mut index = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= rows.len(),
            values.len() == index,
            forall|k: int| 0 <= k < index ==> #[trigger] values[k] == rows[k],
        decreases rows.len() - index,
    ))]
    while index < rows.len() {
        let row = rows[index];
        if row as u64 > i64::MAX as u64 {
            return None;
        }
        values.push(row as i64);
        index += 1;
    }
    Some(values)
}

/// Rows grouped by effective execution mode, with the permutation that restores batch order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(super) struct ModePartition {
    /// Mode of the first group: the effective mode of row zero.
    pub(super) first_mode: Mamba2Mode,
    /// Ascending rows of the first group.
    pub(super) first: Vec<usize>,
    /// The other mode.
    pub(super) second_mode: Mamba2Mode,
    /// Ascending rows of the second group; empty when every row shares the first mode.
    pub(super) second: Vec<usize>,
    /// `restore[b]` is row `b`'s position in the concatenation `first ++ second`.
    pub(super) restore: Vec<usize>,
}

/// Returns whether two modes are equal.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(same =>
    ensures
        same == (left == right),
))]
fn same_mode(left: Mamba2Mode, right: Mamba2Mode) -> bool {
    matches!(
        (left, right),
        (Mamba2Mode::Recurrent, Mamba2Mode::Recurrent)
            | (Mamba2Mode::ChunkedPrefill, Mamba2Mode::ChunkedPrefill)
    )
}

/// Returns the mode of the first active row, or chunked prefill when every row is inactive.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(mode =>
    ensures
        mode == fallback_from(rows@, 0),
))]
fn fallback_mode(rows: &[Mamba2Row]) -> Mamba2Mode {
    let mut row = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows.len(),
            fallback_from(rows@, 0) == fallback_from(rows@, row as int),
        decreases rows.len() - row,
    ))]
    while row < rows.len() && rows[row].tokens == 0 {
        row += 1;
    }
    if row < rows.len() {
        rows[row].mode
    } else {
        Mamba2Mode::ChunkedPrefill
    }
}

/// Partitions rows by effective mode, in first-appearance order of the modes.
///
/// An inactive row (zero tokens) is an identity transition under either
/// mode, so it takes the mode of the first active row and joins that group
/// instead of forcing a second pass. The proof shows that the two groups
/// together hold every row exactly once, each in ascending order and only
/// with rows of its mode; that the first group's mode is row zero's
/// effective mode and the second's is the other mode; and that `restore` is
/// a two-sided inverse of the concatenation `first ++ second`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(partition =>
    ensures
        partition.first.len() + partition.second.len() == rows.len(),
        partition.first_mode != partition.second_mode,
        rows.len() > 0 ==> partition.first.len() > 0,
        rows.len() > 0 ==> partition.first_mode == effective_mode(rows@, 0),
        forall|j: int| 0 <= j < partition.first.len() ==> {
            &&& #[trigger] partition.first[j] < rows.len()
            &&& effective_mode(rows@, partition.first[j] as int) == partition.first_mode
        },
        forall|j: int| 0 <= j < partition.second.len() ==> {
            &&& #[trigger] partition.second[j] < rows.len()
            &&& effective_mode(rows@, partition.second[j] as int) == partition.second_mode
        },
        forall|j: int| 0 <= j && j + 1 < partition.first.len() ==>
            #[trigger] partition.first[j] < partition.first[j + 1],
        forall|j: int| 0 <= j && j + 1 < partition.second.len() ==>
            #[trigger] partition.second[j] < partition.second[j + 1],
        partition.restore.len() == rows.len(),
        forall|b: int| 0 <= b < rows.len() ==> {
            &&& #[trigger] partition.restore[b] < rows.len()
            &&& grouped_row(partition.first@, partition.second@, partition.restore[b] as int) == b
        },
        forall|p: int| 0 <= p < rows.len() ==>
            partition.restore[#[trigger] grouped_row(partition.first@, partition.second@, p)] == p,
))]
pub(super) fn partition_modes(rows: &[Mamba2Row]) -> ModePartition {
    let fallback = fallback_mode(rows);
    let first_mode = if rows.is_empty() || rows[0].tokens == 0 {
        fallback
    } else {
        rows[0].mode
    };
    let second_mode = match first_mode {
        Mamba2Mode::Recurrent => Mamba2Mode::ChunkedPrefill,
        Mamba2Mode::ChunkedPrefill => Mamba2Mode::Recurrent,
    };
    let mut first: Vec<usize> = Vec::new();
    let mut second: Vec<usize> = Vec::new();
    let mut in_first: Vec<bool> = Vec::with_capacity(rows.len());
    let mut ranks: Vec<usize> = Vec::with_capacity(rows.len());
    let mut row = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows.len(),
            fallback == fallback_from(rows@, 0),
            rows.len() > 0 ==> first_mode == effective_mode(rows@, 0),
            first_mode != second_mode,
            in_first.len() == row,
            ranks.len() == row,
            first.len() + second.len() == row,
            row > 0 ==> first.len() > 0,
            forall|j: int| 0 <= j < first.len() ==> {
                &&& #[trigger] first[j] < row
                &&& effective_mode(rows@, first[j] as int) == first_mode
                &&& in_first[first[j] as int]
                &&& ranks[first[j] as int] == j
            },
            forall|j: int| 0 <= j < second.len() ==> {
                &&& #[trigger] second[j] < row
                &&& effective_mode(rows@, second[j] as int) == second_mode
                &&& !in_first[second[j] as int]
                &&& ranks[second[j] as int] == j
            },
            forall|j: int| 0 <= j && j + 1 < first.len() ==> #[trigger] first[j] < first[j + 1],
            forall|j: int| 0 <= j && j + 1 < second.len() ==> #[trigger] second[j] < second[j + 1],
            forall|b: int| 0 <= b < row ==> {
                &&& #[trigger] in_first[b] ==> ranks[b] < first.len() && first[ranks[b] as int] == b
                &&& !in_first[b] ==> ranks[b] < second.len() && second[ranks[b] as int] == b
            },
        decreases rows.len() - row,
    ))]
    while row < rows.len() {
        let mode = if rows[row].tokens > 0 {
            rows[row].mode
        } else {
            fallback
        };
        if same_mode(mode, first_mode) {
            in_first.push(true);
            ranks.push(first.len());
            first.push(row);
        } else {
            #[cfg(verus_keep_ghost)]
            proof! {
                // Two modes exist, so a mode other than the first is the second.
                assert(mode == second_mode);
            }
            in_first.push(false);
            ranks.push(second.len());
            second.push(row);
        }
        row += 1;
    }
    let mut restore: Vec<usize> = Vec::with_capacity(rows.len());
    let mut row = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows.len(),
            in_first.len() == rows.len(),
            ranks.len() == rows.len(),
            first.len() + second.len() == rows.len(),
            restore.len() == row,
            forall|b: int| 0 <= b < rows.len() ==> {
                &&& #[trigger] in_first[b] ==> ranks[b] < first.len() && first[ranks[b] as int] == b
                &&& !in_first[b] ==> ranks[b] < second.len() && second[ranks[b] as int] == b
            },
            forall|b: int| 0 <= b < row ==> {
                &&& #[trigger] restore[b] < rows.len()
                &&& grouped_row(first@, second@, restore[b] as int) == b
                &&& restore[b] == if in_first[b] { ranks[b] as int } else { first.len() + ranks[b] }
            },
        decreases rows.len() - row,
    ))]
    while row < rows.len() {
        let position = if in_first[row] {
            ranks[row]
        } else {
            first.len() + ranks[row]
        };
        restore.push(position);
        row += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|p: int| 0 <= p < rows.len() implies
            restore[#[trigger] grouped_row(first@, second@, p)] == p
        by {
            if p < first.len() {
                assert(in_first[first[p] as int]);
                assert(ranks[first[p] as int] == p);
            } else {
                let j = p - first.len();
                assert(!in_first[second[j] as int]);
                assert(ranks[second[j] as int] == j);
            }
        }
    }
    ModePartition {
        first_mode,
        first,
        second_mode,
        second,
        restore,
    }
}

/// Returns how many chunks one SSD scan block holds.
///
/// One chunk's `[batch, heads, T, T]` F32 score plane has `plane =
/// max(1, batch * heads * T * T)` elements (saturated). The proof shows the
/// result is exactly `clamp(budget / plane, 1, chunks)`, so a block of
/// `count` chunks keeps `count * plane <= budget` unless a single chunk alone
/// exceeds the budget.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(count =>
    requires
        chunks >= 1,
    ensures
        1 <= count <= chunks,
        count == clamped_chunks(
            budget as int / scan_plane(batch as int, heads as int, chunk_size as int),
            chunks as int,
        ),
        count == 1 || count * scan_plane(batch as int, heads as int, chunk_size as int) <= budget,
))]
pub(super) fn scan_block_chunks(
    batch: usize,
    heads: usize,
    chunk_size: usize,
    budget: usize,
    chunks: usize,
) -> usize {
    let plane = batch
        .saturating_mul(heads)
        .saturating_mul(chunk_size)
        .saturating_mul(chunk_size)
        .max(1);
    let admitted = budget / plane;
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(budget as int, plane as int);
        vstd::arithmetic::div_mod::lemma_mod_pos_bound(budget as int, plane as int);
        assert(admitted * plane <= budget) by (nonlinear_arith)
            requires
                budget == plane * admitted + budget % plane,
                budget % plane >= 0;
        if admitted > chunks {
            assert(chunks * plane <= admitted * plane) by (nonlinear_arith)
                requires chunks <= admitted, plane >= 1;
        }
    }
    // `clamp(1, chunks)`: at least one chunk, at most every chunk.
    if admitted < 1 {
        1
    } else if admitted > chunks {
        chunks
    } else {
        admitted
    }
}

/// One block of whole chunks of an SSD scan.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ScanBlock {
    /// First token of the block.
    pub(super) start: usize,
    /// Valid tokens in the block.
    pub(super) width: usize,
    /// Whole chunks in the block.
    pub(super) chunks: usize,
    /// Zero steps appended to complete the last chunk.
    pub(super) padding: usize,
}

/// Splits `sequence` tokens into scan blocks of `block_chunks` chunks of `chunk_size` tokens.
///
/// Chunk boundaries sit at multiples of `chunk_size` from the segment start,
/// as in the reference. The proof shows the blocks tile `0..sequence` in
/// order from token zero; every block holds between one and `block_chunks`
/// chunks, and all but the last hold exactly `block_chunks` unpadded chunks;
/// and each block's valid width plus padding is its chunks times
/// `chunk_size`, with less than one chunk of padding. The bound on
/// `sequence` holds for every materialized token axis.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(blocks =>
    requires
        0 < sequence <= usize::MAX / 2,
        chunk_size > 0,
        block_chunks > 0,
    ensures
        blocks.len() >= 1,
        blocks[0].start == 0,
        blocks[blocks.len() - 1].start + blocks[blocks.len() - 1].width == sequence,
        forall|i: int| 0 <= i < blocks.len() ==> {
            &&& 1 <= #[trigger] blocks[i].chunks <= block_chunks
            &&& 1 <= blocks[i].width
            &&& blocks[i].width + blocks[i].padding == blocks[i].chunks * chunk_size
            &&& blocks[i].padding < chunk_size
            &&& blocks[i].start + blocks[i].width <= sequence
        },
        forall|i: int| 0 <= i && i + 1 < blocks.len() ==> {
            &&& #[trigger] blocks[i].chunks == block_chunks
            &&& blocks[i].padding == 0
            &&& blocks[i + 1].start == blocks[i].start + blocks[i].width
        },
))]
pub(super) fn scan_blocks(
    sequence: usize,
    chunk_size: usize,
    block_chunks: usize,
) -> Vec<ScanBlock> {
    let chunks = ceiling_division(sequence, chunk_size);
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_padded_fits(sequence as int, chunk_size as int, chunks as int);
    }
    let mut blocks: Vec<ScanBlock> = Vec::new();
    let mut first_chunk = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            0 < sequence <= usize::MAX / 2,
            chunk_size > 0,
            block_chunks > 0,
            chunks >= 1,
            (chunks - 1) * chunk_size < sequence,
            sequence <= chunks * chunk_size,
            chunks * chunk_size <= usize::MAX,
            first_chunk <= chunks,
            (blocks.len() == 0) == (first_chunk == 0),
            forall|i: int| 0 <= i < blocks.len() ==> {
                &&& 1 <= #[trigger] blocks[i].chunks <= block_chunks
                &&& 1 <= blocks[i].width
                &&& blocks[i].width + blocks[i].padding == blocks[i].chunks * chunk_size
                &&& blocks[i].padding < chunk_size
                &&& blocks[i].start + blocks[i].width <= sequence
            },
            forall|i: int| 0 <= i && i + 1 < blocks.len() ==> {
                &&& #[trigger] blocks[i].chunks == block_chunks
                &&& blocks[i].padding == 0
                &&& blocks[i + 1].start == blocks[i].start + blocks[i].width
            },
            blocks.len() > 0 && first_chunk < chunks ==> {
                &&& blocks[blocks.len() - 1].chunks == block_chunks
                &&& blocks[blocks.len() - 1].padding == 0
                &&& blocks[blocks.len() - 1].start + blocks[blocks.len() - 1].width
                    == first_chunk * chunk_size
            },
            blocks.len() > 0 && first_chunk == chunks ==>
                blocks[blocks.len() - 1].start + blocks[blocks.len() - 1].width == sequence,
            blocks.len() > 0 ==> blocks[0].start == 0,
        decreases chunks - first_chunk,
    ))]
    while first_chunk < chunks {
        let count = block_chunks.min(chunks - first_chunk);
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_block_extent(
                sequence as int,
                chunk_size as int,
                chunks as int,
                first_chunk as int,
                count as int,
            );
        }
        let start = first_chunk * chunk_size;
        let width = (count * chunk_size).min(sequence - start);
        let padding = count * chunk_size - width;
        blocks.push(ScanBlock {
            start,
            width,
            chunks: count,
            padding,
        });
        first_chunk += count;
    }
    blocks
}

#[cfg(verus_keep_ghost)]
verus! {

/// A divisor of `total` splits it into `total / groups` equal parts exactly.
proof fn lemma_exact_quotient(total: int, groups: int)
    requires
        total >= 0,
        groups > 0,
        total % groups == 0,
    ensures
        (total / groups) * groups == total,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(total, groups);
    assert((total / groups) * groups == total) by (nonlinear_arith)
        requires
            total == groups * (total / groups) + total % groups,
            total % groups == 0;
}

/// Products of nonnegative widths are nonnegative.
proof fn lemma_products_nonnegative(heads: int, head_dim: int, groups: int, state_size: int)
    requires
        heads >= 0,
        head_dim >= 0,
        groups >= 0,
        state_size >= 0,
    ensures
        heads * head_dim >= 0,
        groups * state_size >= 0,
        groups * state_size * 2 == (groups * state_size) * 2,
{
    assert(heads * head_dim >= 0) by (nonlinear_arith)
        requires heads >= 0, head_dim >= 0;
    assert(groups * state_size >= 0) by (nonlinear_arith)
        requires groups >= 0, state_size >= 0;
}

/// The padded length `ceil(L / T) * T` covers `L` by less than one chunk and fits in `usize`.
proof fn lemma_padded_fits(sequence: int, chunk_size: int, chunks: int)
    requires
        0 < sequence <= usize::MAX / 2,
        0 < chunk_size <= usize::MAX,
        chunks == crate::delta_net::plan::ceiling(sequence, chunk_size),
        chunks * chunk_size >= sequence,
        chunks * chunk_size < sequence + chunk_size,
        chunks >= 1,
    ensures
        (chunks - 1) * chunk_size < sequence,
        chunks * chunk_size <= usize::MAX,
{
    assert((chunks - 1) * chunk_size == chunks * chunk_size - chunk_size) by (nonlinear_arith);
    if chunk_size >= sequence {
        // One chunk covers the whole sequence.
        assert(chunks == 1) by (nonlinear_arith)
            requires
                chunks * chunk_size < sequence + chunk_size,
                chunk_size >= sequence,
                chunks >= 1,
                sequence > 0;
        assert(chunks * chunk_size == chunk_size) by (nonlinear_arith)
            requires chunks == 1;
    } else {
        // Less than one chunk of padding on a sequence of at most half the range.
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(usize::MAX as int, 2);
        assert(chunks * chunk_size < 2 * sequence);
        assert(2 * sequence <= usize::MAX);
    }
}

/// Block extent arithmetic: the block starts inside the sequence, and its chunks fit the padded length.
proof fn lemma_block_extent(
    sequence: int,
    chunk_size: int,
    chunks: int,
    first_chunk: int,
    count: int,
)
    requires
        chunk_size > 0,
        0 <= first_chunk < chunks,
        1 <= count <= chunks - first_chunk,
        (chunks - 1) * chunk_size < sequence,
        sequence <= chunks * chunk_size,
        chunks * chunk_size <= usize::MAX,
    ensures
        first_chunk * chunk_size < sequence,
        (first_chunk + count) * chunk_size == first_chunk * chunk_size + count * chunk_size,
        count * chunk_size <= usize::MAX,
        count * chunk_size >= chunk_size,
        first_chunk + count < chunks ==> (first_chunk + count) * chunk_size <= sequence - 1,
        first_chunk + count == chunks ==> {
            &&& first_chunk * chunk_size + count * chunk_size >= sequence
            &&& first_chunk * chunk_size + count * chunk_size < sequence + chunk_size
        },
{
    assert(first_chunk * chunk_size <= (chunks - 1) * chunk_size) by (nonlinear_arith)
        requires first_chunk <= chunks - 1, chunk_size > 0;
    assert((first_chunk + count) * chunk_size == first_chunk * chunk_size + count * chunk_size)
        by (nonlinear_arith);
    assert(count * chunk_size <= chunks * chunk_size) by (nonlinear_arith)
        requires count <= chunks, chunk_size > 0;
    assert(count * chunk_size >= chunk_size) by (nonlinear_arith)
        requires count >= 1, chunk_size > 0;
    assert(first_chunk * chunk_size >= 0) by (nonlinear_arith)
        requires first_chunk >= 0, chunk_size > 0;
    if first_chunk + count < chunks {
        assert((first_chunk + count) * chunk_size <= (chunks - 1) * chunk_size) by (nonlinear_arith)
            requires first_chunk + count <= chunks - 1, chunk_size > 0;
    } else {
        assert((chunks - 1) * chunk_size == chunks * chunk_size - chunk_size) by (nonlinear_arith);
    }
}

} // verus!

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
