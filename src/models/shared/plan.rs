//! Pure index plans for padded rows: compaction, scatter, selection, and appends.
//!
//! A rectangular batch row of `S` columns carries a validity mask
//! `m ∈ {0, 1}^S` (`true` marks a real token). Stateful execution removes the
//! padding, runs the model on the `n` valid columns `c_0 < c_1 < … < c_{n-1}`,
//! and finally restores the rectangular layout with exact zeros at padding:
//!
//! ```text
//! compact[k] = rect[c_k]                     (compaction, k < n)
//! rect'[j]   = compact[k]  if j = c_k        (scatter)
//!            = 0           otherwise
//! ```
//!
//! [`plan_scatter`] chooses the cheapest tensor program for the scatter:
//! the identity when every column is valid, one zero-padded concatenation when
//! the valid columns form a single run (left or right padding), and otherwise a
//! single gather whose source index `n` addresses an appended zero row. This
//! replaces the former one-`narrow`-per-token concatenation, whose kernel count
//! grew linearly with the number of valid tokens.
//!
//! The remaining helpers convert indices to the signed form Burn selectors
//! expect, derive implicit cached positions `baseline + k`, turn a maximal
//! position into the rotary context extent `max + 1`, plan per-row cache
//! appends against a capacity, and compute exact value ranges over the valid
//! columns of a row.
//!
//! [`plan_host_rows`] composes them into the whole host side of a stateful
//! call after its single readback ([`split_readback`]): every row's valid and
//! last columns, the empty-row policy, the cache appends, and validation of
//! valid token and position values with each row's rotary extent.
//!
//! # Verified planning
//!
//! Every function in this file is dependency-free and checked in place by
//! Verus through `verus/models.rs` (see `verus/README.md`). For all admissible
//! inputs the proofs establish freedom from integer overflow and the stated
//! postconditions: compaction lists exactly the valid columns in increasing
//! order; scatter plans place compact entry `k` at column `c_k` and zeros
//! elsewhere; append plans either return `length + count <= capacity` for every
//! row or report the first failing row; value ranges are attained bounds; and
//! host row planning returns exactly the specified plan or the first error in
//! precedence order. Tensor programs driven by these plans remain covered by
//! the Rust tests.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// `s` is strictly increasing and every entry lies below `bound`.
///
/// For mask-derived column lists this is exactly "sorted, unique, in range".
pub open spec fn sorted_unique_below(s: Seq<usize>, bound: int) -> bool {
    &&& forall|i: int, j: int| 0 <= i < j < s.len() ==> s[i] < s[j]
    &&& forall|i: int| 0 <= i < s.len() ==> s[i] < bound
}

/// `sources` gathers compact entries into `sequence` rectangular columns.
///
/// Column `j` reads compact entry `sources[j]` when it is below `n`, and the
/// appended zero entry `n` otherwise; each valid column `c_k` reads entry `k`.
pub open spec fn gathers_into_columns(
    indices: Seq<usize>,
    sequence: int,
    sources: Seq<usize>,
) -> bool {
    &&& sources.len() == sequence
    &&& forall|k: int| 0 <= k < indices.len() ==> sources[indices[k] as int] == k
    &&& forall|j: int| 0 <= j < sequence ==> sources[j] <= indices.len()
    &&& forall|j: int| 0 <= j < sequence && sources[j] < indices.len()
        ==> indices[sources[j] as int] == j
}

/// `indices` is the single run `start, start + 1, …`.
pub open spec fn is_run_from(indices: Seq<usize>, start: int) -> bool {
    forall|k: int| 0 <= k < indices.len() ==> indices[k] == start + k
}

/// The valid (`true`) columns of `row` in increasing order, defined by recursion on the width.
pub open spec fn spec_valid_columns(row: Seq<bool>) -> Seq<usize>
    decreases row.len(),
{
    if row.len() == 0 {
        Seq::empty()
    } else {
        let earlier = spec_valid_columns(row.drop_last());
        if row.last() {
            earlier.push((row.len() - 1) as usize)
        } else {
            earlier
        }
    }
}

/// `columns` lists exactly the valid columns of `row`, strictly increasing.
pub open spec fn lists_valid_columns(row: Seq<bool>, columns: Seq<usize>) -> bool {
    &&& sorted_unique_below(columns, row.len() as int)
    &&& forall|i: int| 0 <= i < columns.len() ==> row[#[trigger] columns[i] as int]
    &&& forall|c: int| 0 <= c < row.len() && #[trigger] row[c] ==> columns.contains(c as usize)
}

/// Row `row` of a row-major grid with `width` columns: `cells[row·width .. row·width + width)`.
pub open spec fn grid_row<T>(cells: Seq<T>, width: int, row: int) -> Seq<T> {
    cells.subrange(row * width, row * width + width)
}

/// Validity of row `row` of an optional row-major `[batch, sequence]` mask.
///
/// An absent mask marks every column valid.
pub open spec fn mask_row(mask: Option<&[bool]>, sequence: int, row: int) -> Seq<bool> {
    match mask {
        Some(flags) => grid_row(flags@, sequence, row),
        None => Seq::new(sequence as nat, |column: int| true),
    }
}

/// The valid columns of row `row` of an optional mask.
pub open spec fn row_columns(mask: Option<&[bool]>, sequence: int, row: int) -> Seq<usize> {
    spec_valid_columns(mask_row(mask, sequence, row))
}

/// `column` is the last valid column of `row`, or the final column when none is valid.
pub open spec fn last_valid_or_final(row: Seq<bool>, column: int) -> bool {
    if exists|c: int| 0 <= c < row.len() && #[trigger] row[c] {
        &&& 0 <= column < row.len()
        &&& row[column]
        &&& forall|c: int| column < c < row.len() ==> !#[trigger] row[c]
    } else {
        column == row.len() - 1
    }
}

/// Every listed value lies in `0..limit`.
pub open spec fn values_within(values: Seq<i64>, columns: Seq<usize>, limit: int) -> bool {
    forall|k: int| 0 <= k < columns.len() ==> 0 <= #[trigger] values[columns[k] as int] < limit
}

/// `minimum` is attained by a listed value and bounds every listed value from below.
pub open spec fn is_minimum_at(values: Seq<i64>, columns: Seq<usize>, minimum: int) -> bool {
    &&& forall|k: int| 0 <= k < columns.len() ==> minimum <= #[trigger] values[columns[k] as int]
    &&& exists|k: int| 0 <= k < columns.len() && values[columns[k] as int] == minimum
}

/// `maximum` is attained by a listed value and bounds every listed value from above.
pub open spec fn is_maximum_at(values: Seq<i64>, columns: Seq<usize>, maximum: int) -> bool {
    &&& forall|k: int| 0 <= k < columns.len() ==> #[trigger] values[columns[k] as int] <= maximum
    &&& exists|k: int| 0 <= k < columns.len() && values[columns[k] as int] == maximum
}

/// The listed positions of a row whose explicit positions exist, or `None`.
///
/// Rows without explicit positions, or without valid columns, use their
/// implicit cursor instead.
pub open spec fn explicit_row(
    positions: Option<&[i64]>,
    columns: Seq<usize>,
    sequence: int,
    row: int,
) -> Option<Seq<i64>> {
    match positions {
        Some(values) => if columns.len() > 0 {
            Some(grid_row(values@, sequence, row))
        } else {
            None
        },
        None => None,
    }
}

/// Row `row` passes value validation and has rotary extent `context`.
///
/// Valid tokens lie in `0..vocabulary`; explicit valid positions lie in
/// `0..limit` when a limit is given and the extent is `1 + max p`; without
/// explicit positions the extent is the row's implicit cursor `end`.
pub open spec fn row_context_ok(
    tokens: Seq<i64>,
    positions: Option<&[i64]>,
    columns: Seq<usize>,
    sequence: int,
    row: int,
    vocabulary: int,
    position_limit: Option<usize>,
    end: int,
    context: int,
) -> bool {
    &&& values_within(grid_row(tokens, sequence, row), columns, vocabulary)
    &&& match explicit_row(positions, columns, sequence, row) {
        Some(values) => {
            &&& (position_limit matches Some(limit)
                ==> values_within(values, columns, limit as int))
            &&& is_maximum_at(values, columns, context - 1)
        },
        None => context == end,
    }
}

/// Row `row` passes value validation with some rotary extent.
pub open spec fn row_succeeds(
    tokens: Seq<i64>,
    positions: Option<&[i64]>,
    columns: Seq<usize>,
    sequence: int,
    row: int,
    vocabulary: int,
    position_limit: Option<usize>,
    end: int,
) -> bool {
    exists|context: int| #[trigger] row_context_ok(
        tokens,
        positions,
        columns,
        sequence,
        row,
        vocabulary,
        position_limit,
        end,
        context,
    )
}

/// Row `row` fails value validation with `error`, checked tokens first.
pub(crate) open spec fn row_value_error(
    tokens: Seq<i64>,
    positions: Option<&[i64]>,
    columns: Seq<usize>,
    sequence: int,
    row: int,
    vocabulary: int,
    position_limit: Option<usize>,
    error: RowValueError,
) -> bool {
    let token_row = grid_row(tokens, sequence, row);
    match error {
        RowValueError::Tokens { minimum, maximum } => {
            &&& !values_within(token_row, columns, vocabulary)
            &&& is_minimum_at(token_row, columns, minimum as int)
            &&& is_maximum_at(token_row, columns, maximum as int)
        },
        RowValueError::Positions { minimum, maximum, limit } => {
            &&& values_within(token_row, columns, vocabulary)
            &&& position_limit == Some(limit)
            &&& match explicit_row(positions, columns, sequence, row) {
                Some(values) => {
                    &&& !values_within(values, columns, limit as int)
                    &&& is_minimum_at(values, columns, minimum as int)
                    &&& is_maximum_at(values, columns, maximum as int)
                },
                None => false,
            }
        },
        RowValueError::Unrepresentable => {
            &&& values_within(token_row, columns, vocabulary)
            &&& match explicit_row(positions, columns, sequence, row) {
                Some(values) => {
                    &&& (position_limit matches Some(limit)
                        ==> values_within(values, columns, limit as int))
                    &&& exists|maximum: int| #[trigger] is_maximum_at(values, columns, maximum)
                        && (maximum < 0 || maximum + 1 > usize::MAX)
                },
                None => false,
            }
        },
    }
}

/// Exact outcome of planning the host side of one stateful call.
///
/// Errors follow the documented precedence: a rejected empty row, then the
/// first impossible cache append, then the first row whose valid token or
/// position values are invalid.
pub(crate) open spec fn host_rows_outcome(
    mask: Option<&[bool]>,
    tokens: Seq<i64>,
    positions: Option<&[i64]>,
    batch: int,
    sequence: int,
    lengths: Seq<usize>,
    capacity: int,
    reject_empty_rows: bool,
    vocabulary: int,
    position_limit: Option<usize>,
    planned: Result<HostRows, HostRowsError>,
) -> bool {
    let nonempty = forall|r: int| 0 <= r < batch
        ==> #[trigger] row_columns(mask, sequence, r).len() > 0;
    let fits = forall|r: int| 0 <= r < batch
        ==> lengths[r] + #[trigger] row_columns(mask, sequence, r).len() <= capacity;
    match planned {
        Err(HostRowsError::EmptyRow) => reject_empty_rows && !nonempty,
        Err(HostRowsError::Append(AppendError::Overflow { row, length, additional })) => {
            &&& (reject_empty_rows ==> nonempty)
            &&& row < batch
            &&& length == lengths[row as int]
            &&& additional == row_columns(mask, sequence, row as int).len()
            &&& length + additional > usize::MAX
            &&& forall|r: int| 0 <= r < row
                ==> lengths[r] + #[trigger] row_columns(mask, sequence, r).len() <= capacity
        },
        Err(HostRowsError::Append(AppendError::Capacity { row, requested })) => {
            &&& (reject_empty_rows ==> nonempty)
            &&& row < batch
            &&& requested == lengths[row as int] + row_columns(mask, sequence, row as int).len()
            &&& requested > capacity
            &&& forall|r: int| 0 <= r < row
                ==> lengths[r] + #[trigger] row_columns(mask, sequence, r).len() <= capacity
        },
        Err(HostRowsError::Values(error)) => {
            &&& (reject_empty_rows ==> nonempty)
            &&& fits
            &&& exists|row: int| 0 <= row < batch && {
                &&& forall|r: int| 0 <= r < row ==> #[trigger] row_succeeds(
                    tokens,
                    positions,
                    row_columns(mask, sequence, r),
                    sequence,
                    r,
                    vocabulary,
                    position_limit,
                    lengths[r] + row_columns(mask, sequence, r).len(),
                )
                &&& row_value_error(
                    tokens,
                    positions,
                    row_columns(mask, sequence, row),
                    sequence,
                    row,
                    vocabulary,
                    position_limit,
                    error,
                )
            }
        },
        Ok(rows) => {
            &&& (reject_empty_rows ==> nonempty)
            &&& fits
            &&& rows.valid_indices.len() == batch
            &&& rows.token_counts.len() == batch
            &&& rows.evaluated_contexts.len() == batch
            &&& rows.last_columns.len() == batch
            &&& forall|r: int| 0 <= r < batch ==> {
                &&& #[trigger] rows.valid_indices@[r]@ == row_columns(mask, sequence, r)
                &&& rows.token_counts@[r] == row_columns(mask, sequence, r).len()
                &&& row_context_ok(
                    tokens,
                    positions,
                    row_columns(mask, sequence, r),
                    sequence,
                    r,
                    vocabulary,
                    position_limit,
                    lengths[r] + rows.token_counts@[r],
                    rows.evaluated_contexts@[r] as int,
                )
                &&& rows.last_columns@[r] < sequence
                &&& last_valid_or_final(mask_row(mask, sequence, r), rows.last_columns@[r] as int)
            }
        },
    }
}

/// Row `row` of a `[rows, width]` grid lies inside it, so its offsets are representable.
pub proof fn lemma_grid_row_in_bounds(row: int, rows: int, width: int)
    requires
        0 <= row < rows,
        0 <= width,
    ensures
        0 <= row * width,
        row * width + width <= rows * width,
{
    assert(0 <= row * width) by (nonlinear_arith)
        requires
            0 <= row,
            0 <= width,
    ;
    assert(row * width + width <= rows * width) by (nonlinear_arith)
        requires
            row < rows,
            0 <= width,
    ;
}

} // verus!

/// Rectangular reconstruction strategy for one compact row.
///
/// Every variant describes the same mathematical scatter; they differ only in
/// the number and kind of tensor kernels needed to realize it.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ScatterLayout {
    /// Every column is valid, so the compact row already is the rectangle.
    Identity,
    /// Valid columns are one run preceded by `leading` and followed by
    /// `trailing` padding columns (left or right padding).
    Contiguous {
        /// Number of padding columns before the first valid column.
        leading: usize,
        /// Number of padding columns after the last valid column.
        trailing: usize,
    },
    /// Interior padding: output column `j` gathers compact entry `sources[j]`,
    /// where entry `n` (the valid-token count) is an appended zero row.
    Gather {
        /// One compact source index per rectangular column.
        sources: Vec<usize>,
    },
}

/// Lists the valid (`true`) columns of one mask row in increasing order.
///
/// The result `c` satisfies `row[j] ⇔ j ∈ c`, is strictly increasing, and
/// never exceeds the row width; it is the compaction selector for the row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(columns =>
    ensures
        columns@ == spec_valid_columns(row@),
        lists_valid_columns(row@, columns@),
        sorted_unique_below(columns@, row.len() as int),
        forall|i: int| 0 <= i < columns.len() ==> row[#[trigger] columns[i] as int],
        forall|c: int| 0 <= c < row.len() && #[trigger] row[c] ==> columns@.contains(c as usize),
))]
pub(crate) fn valid_columns(row: &[bool]) -> Vec<usize> {
    let mut columns: Vec<usize> = Vec::new();
    let mut column: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            column <= row.len(),
            columns@ == spec_valid_columns(row@.subrange(0, column as int)),
            sorted_unique_below(columns@, column as int),
            forall|i: int| 0 <= i < columns.len() ==> row[#[trigger] columns[i] as int],
            forall|c: int| 0 <= c < column && #[trigger] row[c] ==> columns@.contains(c as usize),
        decreases row.len() - column,
    ))]
    while column < row.len() {
        // The recursive specification peels the last column of the longer prefix.
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(row@.subrange(0, column as int + 1).drop_last()
                =~= row@.subrange(0, column as int));
        }
        if row[column] {
            #[cfg(verus_keep_ghost)]
            proof_decl! { let ghost before = columns@; }
            columns.push(column);
            // Earlier valid columns keep their witnesses; `column` gains the new last entry.
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(columns@[columns.len() - 1] == column);
                assert forall|c: int| 0 <= c < column && #[trigger] row[c]
                    implies columns@.contains(c as usize) by {
                    let i = choose|i: int| 0 <= i < before.len() && before[i] == c as usize;
                    assert(columns@[i] == c as usize);
                }
            }
        }
        column += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(row@.subrange(0, row.len() as int) =~= row@);
    }
    columns
}

/// Returns whether `indices` is strictly increasing with every entry below `bound`.
///
/// This is the precondition of [`plan_scatter`]: valid columns produced by
/// [`valid_columns`] always satisfy it, while caller-supplied plans are checked.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(valid =>
    ensures
        valid == sorted_unique_below(indices@, bound as int),
))]
pub(crate) fn is_sorted_unique_below(indices: &[usize], bound: usize) -> bool {
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= indices.len(),
            sorted_unique_below(indices@.subrange(0, index as int), bound as int),
        decreases indices.len() - index,
    ))]
    while index < indices.len() {
        if indices[index] >= bound || (index > 0 && indices[index - 1] >= indices[index]) {
            // A violation at `index` falsifies the predicate for the full sequence.
            #[cfg(verus_keep_ghost)]
            proof! {
                if indices[index as int] >= bound {
                    assert(!sorted_unique_below(indices@, bound as int));
                } else {
                    assert(indices@[index - 1] >= indices@[index as int]);
                    assert(!sorted_unique_below(indices@, bound as int));
                }
            }
            return false;
        }
        // Extending a strictly increasing prefix by a larger in-range entry preserves it.
        #[cfg(verus_keep_ghost)]
        proof! {
            let prefix = indices@.subrange(0, index as int);
            let next = indices@.subrange(0, index as int + 1);
            assert forall|i: int, j: int| 0 <= i < j < next.len() implies next[i] < next[j] by {
                if j < index {
                    assert(prefix[i] < prefix[j]);
                } else if i < index as int - 1 {
                    assert(prefix[i] < prefix[index as int - 1]);
                }
            }
            assert forall|i: int| 0 <= i < next.len() implies next[i] < bound by {
                if i < index {
                    assert(prefix[i] < bound);
                }
            }
        }
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(indices@.subrange(0, indices.len() as int) == indices@);
    }
    true
}

/// Plans the rectangular reconstruction of a compact row.
///
/// `indices` are the row's valid columns `c_0 < … < c_{n-1} < sequence`.
/// The plan is [`ScatterLayout::Identity`] exactly when `c_k = k` and
/// `n = sequence`, [`ScatterLayout::Contiguous`] when the columns form one run,
/// and a [`ScatterLayout::Gather`] map otherwise. An empty row is the
/// contiguous plan with `sequence` leading padding columns, i.e. all zeros.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(layout =>
    requires
        sorted_unique_below(indices@, sequence as int),
    ensures
        match layout {
            ScatterLayout::Identity => {
                &&& indices.len() == sequence
                &&& is_run_from(indices@, 0)
            },
            ScatterLayout::Contiguous { leading, trailing } => {
                &&& leading + indices.len() + trailing == sequence
                &&& is_run_from(indices@, leading as int)
                // The identity is preferred whenever it applies.
                &&& (indices.len() == 0 || leading > 0 || trailing > 0)
            },
            ScatterLayout::Gather { sources } => {
                &&& gathers_into_columns(indices@, sequence as int, sources@)
                // A gather is used only when no single run exists.
                &&& indices.len() > 0
                &&& !is_run_from(indices@, indices[0] as int)
            },
        },
))]
pub(crate) fn plan_scatter(indices: &[usize], sequence: usize) -> ScatterLayout {
    if indices.is_empty() {
        return ScatterLayout::Contiguous {
            leading: sequence,
            trailing: 0,
        };
    }
    let count = indices.len();
    let leading = indices[0];
    if !is_single_run(indices) {
        return ScatterLayout::Gather {
            sources: gather_sources(indices, sequence),
        };
    }
    // The run ends at `leading + count - 1 = indices[count - 1] < sequence`.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(indices@[count - 1] == leading + count - 1);
        assert(indices@[count - 1] < sequence);
    }
    let trailing = sequence - (leading + count);
    if leading == 0 && trailing == 0 {
        ScatterLayout::Identity
    } else {
        ScatterLayout::Contiguous { leading, trailing }
    }
}

/// Returns whether non-empty, strictly increasing `indices` advance by one.
///
/// Checking adjacent pairs keeps the arithmetic overflow-free: every entry is
/// below `sequence <= usize::MAX`, so `indices[k - 1] + 1` is representable.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(run =>
    requires
        indices.len() > 0,
        forall|i: int| 0 <= i < indices.len() ==> indices[i] < usize::MAX,
    ensures
        run == is_run_from(indices@, indices[0] as int),
))]
fn is_single_run(indices: &[usize]) -> bool {
    let mut index: usize = 1;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            1 <= index <= indices.len(),
            forall|i: int| 0 <= i < indices.len() ==> indices[i] < usize::MAX,
            is_run_from(indices@.subrange(0, index as int), indices[0] as int),
        decreases indices.len() - index,
    ))]
    while index < indices.len() {
        if indices[index] != indices[index - 1] + 1 {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(indices@.subrange(0, index as int)[index - 1] == indices[index - 1]);
            }
            return false;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            let prefix = indices@.subrange(0, index as int);
            let next = indices@.subrange(0, index as int + 1);
            assert forall|k: int| 0 <= k < next.len()
                implies next[k] == indices[0] as int + k by {
                if k < index {
                    assert(prefix[k] == next[k]);
                } else {
                    assert(prefix[index - 1] == indices[index - 1]);
                }
            }
        }
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(indices@.subrange(0, indices.len() as int) == indices@);
    }
    true
}

/// Builds the gather map that restores interior-padded columns.
///
/// Column `j` receives compact entry `k` when `j = c_k`, and the zero entry
/// `n = indices.len()` otherwise. A single merge pass over the sorted columns
/// suffices because both sequences increase.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sources =>
    requires
        sorted_unique_below(indices@, sequence as int),
    ensures
        gathers_into_columns(indices@, sequence as int, sources@),
))]
fn gather_sources(indices: &[usize], sequence: usize) -> Vec<usize> {
    let count = indices.len();
    let mut sources: Vec<usize> = Vec::with_capacity(sequence);
    let mut next: usize = 0;
    let mut column: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            column <= sequence,
            next <= count,
            count == indices.len(),
            sorted_unique_below(indices@, sequence as int),
            sources.len() == column,
            next < count ==> indices[next as int] >= column,
            forall|k: int| 0 <= k < next ==> indices[k] < column,
            forall|k: int| 0 <= k < next ==> sources[indices[k] as int] == k,
            forall|j: int| 0 <= j < column ==> sources[j] <= count,
            forall|j: int| 0 <= j < column && sources[j] < count
                ==> indices[sources[j] as int] == j,
        decreases sequence - column,
    ))]
    while column < sequence {
        if next < count && indices[next] == column {
            sources.push(next);
            // Strict increase moves the next unmatched column beyond `column`.
            #[cfg(verus_keep_ghost)]
            proof! {
                if next + 1 < count {
                    assert(indices[next as int] < indices[next + 1]);
                }
            }
            next += 1;
        } else {
            sources.push(count);
        }
        column += 1;
    }
    // Every column is below `sequence`, so the merge consumed all of them.
    #[cfg(verus_keep_ghost)]
    proof! {
        if next < count {
            assert(indices[next as int] < sequence);
        }
    }
    sources
}

/// Returns whether an explicit logits selection is non-empty and in range.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(valid =>
    ensures
        valid == (positions.len() > 0
            && forall|i: int| 0 <= i < positions.len() ==> #[trigger] positions[i] < sequence),
))]
pub(crate) fn positions_within(positions: &[usize], sequence: usize) -> bool {
    if positions.is_empty() {
        return false;
    }
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= positions.len(),
            forall|i: int| 0 <= i < index ==> #[trigger] positions[i] < sequence,
        decreases positions.len() - index,
    ))]
    while index < positions.len() {
        if positions[index] >= sequence {
            return false;
        }
        index += 1;
    }
    true
}

/// Returns the last valid column of a mask row, or `None` for an empty row.
///
/// This is the per-row index that [`LogitsSelection::Last`] projects.
///
/// [`LogitsSelection::Last`]: crate::LogitsSelection::Last
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(last =>
    ensures
        match last {
            Some(column) => {
                &&& column < row.len()
                &&& row[column as int]
                &&& forall|c: int| column < c < row.len() ==> !#[trigger] row[c]
            },
            None => forall|c: int| 0 <= c < row.len() ==> !#[trigger] row[c],
        },
))]
pub(crate) fn last_valid_column(row: &[bool]) -> Option<usize> {
    let mut remaining: usize = row.len();
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            remaining <= row.len(),
            forall|c: int| remaining <= c < row.len() ==> !#[trigger] row[c],
        decreases remaining,
    ))]
    while remaining > 0 {
        if row[remaining - 1] {
            return Some(remaining - 1);
        }
        remaining -= 1;
    }
    None
}

/// Converts column indices to the signed `i64` values Burn index tensors use.
///
/// Returns `None` when an index exceeds `i64::MAX`; otherwise every value is
/// preserved exactly.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(signed =>
    ensures
        match signed {
            Some(values) => {
                &&& values.len() == indices.len()
                &&& forall|k: int| 0 <= k < indices.len() ==> values[k] == indices[k]
            },
            None => exists|k: int| 0 <= k < indices.len() && indices[k] > i64::MAX,
        },
))]
pub(crate) fn signed_indices(indices: &[usize]) -> Option<Vec<i64>> {
    let mut values: Vec<i64> = Vec::with_capacity(indices.len());
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= indices.len(),
            values.len() == index,
            forall|k: int| 0 <= k < index ==> values[k] == indices[k],
        decreases indices.len() - index,
    ))]
    while index < indices.len() {
        let value = indices[index];
        if value as u64 > i64::MAX as u64 {
            return None;
        }
        values.push(value as i64);
        index += 1;
    }
    Some(values)
}

/// Derives implicit cached positions `baseline, baseline + 1, …` for `count` tokens.
///
/// Returns `None` when the last position `baseline + count - 1` would exceed
/// `usize` or `i64`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(positions =>
    ensures
        match positions {
            Some(values) => {
                &&& values.len() == count
                &&& forall|k: int| 0 <= k < count ==> values[k] == baseline + k
            },
            None => {
                &&& count > 0
                &&& (baseline + count - 1 > i64::MAX || baseline + count - 1 > usize::MAX)
            },
        },
))]
pub(crate) fn implicit_positions(baseline: usize, count: usize) -> Option<Vec<i64>> {
    let mut values: Vec<i64> = Vec::with_capacity(count);
    let mut offset: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            offset <= count,
            values.len() == offset,
            forall|k: int| 0 <= k < offset ==> values[k] == baseline + k,
        decreases count - offset,
    ))]
    while offset < count {
        let position = match baseline.checked_add(offset) {
            Some(position) if position as u64 <= i64::MAX as u64 => position,
            _ => return None,
        };
        values.push(position as i64);
        offset += 1;
    }
    Some(values)
}

/// Converts the greatest position into the rotary table extent `maximum + 1`.
///
/// Returns `None` for a negative maximum or when the extent exceeds `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(context =>
    ensures
        match context {
            Some(extent) => maximum >= 0 && extent == maximum + 1,
            None => maximum < 0 || maximum + 1 > usize::MAX,
        },
))]
pub(crate) fn context_from_maximum(maximum: i64) -> Option<usize> {
    if maximum < 0 || maximum as u64 >= usize::MAX as u64 {
        return None;
    }
    Some(maximum as usize + 1)
}

/// Returns whether an observed `[minimum, maximum]` range lies in `0..limit`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(within =>
    ensures
        within == (0 <= minimum && 0 <= maximum && maximum < limit),
))]
pub(crate) fn range_within(minimum: i64, maximum: i64, limit: usize) -> bool {
    minimum >= 0 && maximum >= 0 && (maximum as u64) < (limit as u64)
}

/// Exact bounds of `values[c]` over the listed columns `c`.
///
/// Returns `None` for an empty column list. Otherwise both bounds are
/// attained by some listed column and enclose every listed value; stateful
/// validation uses this to ignore padding sentinels.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(range =>
    requires
        forall|k: int| 0 <= k < columns.len() ==> #[trigger] columns[k] < values.len(),
    ensures
        match range {
            Some((minimum, maximum)) => {
                &&& columns.len() > 0
                &&& is_minimum_at(values@, columns@, minimum as int)
                &&& is_maximum_at(values@, columns@, maximum as int)
            },
            None => columns.len() == 0,
        },
))]
pub(crate) fn value_range(values: &[i64], columns: &[usize]) -> Option<(i64, i64)> {
    if columns.is_empty() {
        return None;
    }
    let mut minimum = values[columns[0]];
    let mut maximum = minimum;
    let mut index: usize = 1;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            1 <= index <= columns.len(),
            forall|k: int| 0 <= k < columns.len() ==> #[trigger] columns[k] < values.len(),
            forall|k: int| 0 <= k < index
                ==> minimum <= #[trigger] values[columns[k] as int] <= maximum,
            exists|k: int| 0 <= k < index && values[columns[k] as int] == minimum,
            exists|k: int| 0 <= k < index && values[columns[k] as int] == maximum,
        decreases columns.len() - index,
    ))]
    while index < columns.len() {
        let value = values[columns[index]];
        if value < minimum {
            minimum = value;
        }
        if value > maximum {
            maximum = value;
        }
        index += 1;
    }
    Some((minimum, maximum))
}

/// First row whose cache append is impossible.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppendError {
    /// `length + additional` is not representable as `usize`.
    Overflow {
        /// Batch row whose cursor overflowed.
        row: usize,
        /// Committed tokens already in the row.
        length: usize,
        /// Valid tokens the call tried to append.
        additional: usize,
    },
    /// The appended length would exceed the cache capacity.
    Capacity {
        /// Batch row whose cache would overflow.
        row: usize,
        /// Requested logical length after the append.
        requested: usize,
    },
}

/// Plans per-row cache appends `length[r] + count[r]` against `capacity`.
///
/// On success the result holds every row's post-append length, which is also
/// that row's implicit rotary context. Otherwise the first failing row is
/// reported, so earlier rows are known to fit.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(planned =>
    requires
        lengths.len() == counts.len(),
    ensures
        match planned {
            Ok(ends) => {
                &&& ends.len() == lengths.len()
                &&& forall|r: int| 0 <= r < lengths.len() ==> {
                    &&& #[trigger] ends[r] == lengths[r] + counts[r]
                    &&& ends[r] <= capacity
                }
            },
            Err(AppendError::Overflow { row, length, additional }) => {
                &&& row < lengths.len()
                &&& length == lengths[row as int]
                &&& additional == counts[row as int]
                &&& length + additional > usize::MAX
                &&& forall|r: int| 0 <= r < row ==> #[trigger] lengths[r] + counts[r] <= capacity
            },
            Err(AppendError::Capacity { row, requested }) => {
                &&& row < lengths.len()
                &&& requested == lengths[row as int] + counts[row as int]
                &&& requested > capacity
                &&& forall|r: int| 0 <= r < row ==> #[trigger] lengths[r] + counts[r] <= capacity
            },
        },
))]
pub(crate) fn plan_appends(
    lengths: &[usize],
    counts: &[usize],
    capacity: usize,
) -> Result<Vec<usize>, AppendError> {
    let mut ends: Vec<usize> = Vec::with_capacity(lengths.len());
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= lengths.len(),
            lengths.len() == counts.len(),
            ends.len() == row,
            forall|r: int| 0 <= r < row ==> {
                &&& #[trigger] ends[r] == lengths[r] + counts[r]
                &&& ends[r] <= capacity
            },
            forall|r: int| 0 <= r < row ==> #[trigger] lengths[r] + counts[r] <= capacity,
        decreases lengths.len() - row,
    ))]
    while row < lengths.len() {
        let length = lengths[row];
        let additional = counts[row];
        let Some(requested) = length.checked_add(additional) else {
            return Err(AppendError::Overflow {
                row,
                length,
                additional,
            });
        };
        if requested > capacity {
            return Err(AppendError::Capacity { row, requested });
        }
        ends.push(requested);
        row += 1;
    }
    Ok(ends)
}

/// How [`LogitsSelection::Last`] gathers each row's last valid column.
///
/// [`LogitsSelection::Last`]: crate::LogitsSelection::Last
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LastSelection {
    /// The column list does not have one in-range entry per row.
    Invalid,
    /// Every row's last column is `sequence - 1`, so one slice suffices.
    FinalColumn,
    /// Rows end at different columns, so one gather is needed.
    Gather,
}

/// Classifies host-known last columns for [`LogitsSelection::Last`].
///
/// The list is [`LastSelection::Invalid`] exactly when its length differs
/// from `batch` or some entry is not below `sequence`; otherwise it is
/// [`LastSelection::FinalColumn`] exactly when every entry is `sequence - 1`.
///
/// [`LogitsSelection::Last`]: crate::LogitsSelection::Last
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(selection =>
    ensures
        (selection is Invalid) == (last_columns.len() != batch
            || exists|b: int|
                0 <= b < last_columns.len() && #[trigger] last_columns[b] >= sequence),
        selection is FinalColumn ==> forall|b: int| 0 <= b < last_columns.len()
            ==> #[trigger] last_columns[b] + 1 == sequence,
        selection is Gather ==> exists|b: int| 0 <= b < last_columns.len()
            && #[trigger] last_columns[b] + 1 != sequence,
))]
pub(crate) fn plan_last_selection(
    last_columns: &[usize],
    batch: usize,
    sequence: usize,
) -> LastSelection {
    if last_columns.len() != batch {
        return LastSelection::Invalid;
    }
    let mut final_only = true;
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= last_columns.len(),
            forall|b: int| 0 <= b < index ==> #[trigger] last_columns[b] < sequence,
            final_only == forall|b: int|
                0 <= b < index ==> #[trigger] last_columns[b] + 1 == sequence,
        decreases last_columns.len() - index,
    ))]
    while index < last_columns.len() {
        if last_columns[index] >= sequence {
            return LastSelection::Invalid;
        }
        if last_columns[index] + 1 != sequence {
            final_only = false;
        }
        index += 1;
    }
    if final_only {
        LastSelection::FinalColumn
    } else {
        LastSelection::Gather
    }
}

/// Returns whether the smallest per-row valid count fills the width, i.e. no row is padded.
///
/// The count comes from a signed device reduction; the result is `true`
/// exactly when it is non-negative and equals `sequence`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fills =>
    ensures
        fills == (minimum_count >= 0 && minimum_count == sequence),
))]
pub(crate) fn count_fills_width(minimum_count: i64, sequence: usize) -> bool {
    minimum_count >= 0 && minimum_count as u64 == sequence as u64
}

/// Splits one fused readback into its mask flags, token IDs, and explicit positions.
///
/// The readback concatenates `[mask?, tokens, positions?]`, each of
/// `elements` values. Tokens are the segment after the optional mask,
/// positions the final segment when present, and mask flags are the nonzero
/// tests of the leading segment.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(split =>
    requires
        values.len() == elements * ((if has_mask { 1int } else { 0int }) + 1
            + (if has_positions { 1int } else { 0int })),
    ensures
        ({
            let (mask, tokens, positions) = split;
            let offset = if has_mask { elements as int } else { 0int };
            &&& tokens@ == values@.subrange(offset, offset + elements)
            &&& match positions {
                Some(positions) => {
                    &&& has_positions
                    &&& positions@ == values@.subrange(offset + elements, offset + 2 * elements)
                },
                None => !has_positions,
            }
            &&& match mask {
                Some(mask) => {
                    &&& has_mask
                    &&& mask.len() == elements
                    &&& forall|i: int|
                        0 <= i < elements ==> #[trigger] mask@[i] == (values@[i] != 0)
                },
                None => !has_mask,
            }
        }),
))]
pub(crate) fn split_readback(
    values: Vec<i64>,
    elements: usize,
    has_mask: bool,
    has_positions: bool,
) -> (Option<Vec<bool>>, Vec<i64>, Option<Vec<i64>>) {
    let mut values = values;
    #[cfg(verus_keep_ghost)]
    proof_decl! {
        let ghost all = values@;
        let ghost offset: int = if has_mask { elements as int } else { 0int };
    }
    // The readback holds one `elements`-long segment per registered tensor.
    #[cfg(verus_keep_ghost)]
    proof! {
        let masks: int = if has_mask { 1int } else { 0int };
        let tails: int = if has_positions { 1int } else { 0int };
        assert(elements * (masks + 1 + tails) == elements * masks + elements + elements * tails)
            by (nonlinear_arith);
        assert(elements * masks == offset) by (nonlinear_arith)
            requires
                masks == 0 || masks == 1,
                offset == if masks == 1 { elements as int } else { 0int },
        ;
        assert(elements * tails == if has_positions { elements as int } else { 0int })
            by (nonlinear_arith)
            requires
                tails == 0 || tails == 1,
                tails == 1 <==> has_positions,
        ;
    }
    // Split in reverse registration order: positions, tokens, then mask.
    let positions = if has_positions {
        Some(values.split_off(values.len() - elements))
    } else {
        None
    };
    let tokens = values.split_off(values.len() - elements);
    let mask = if has_mask {
        let mut flags: Vec<bool> = Vec::with_capacity(values.len());
        let mut index: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                index <= values.len(),
                values.len() == elements,
                all.len() >= elements,
                values@ == all.subrange(0, elements as int),
                flags.len() == index,
                forall|i: int| 0 <= i < index ==> #[trigger] flags@[i] == (all[i] != 0),
            decreases values.len() - index,
        ))]
        while index < values.len() {
            flags.push(values[index] != 0);
            index += 1;
        }
        Some(flags)
    } else {
        None
    };
    (mask, tokens, positions)
}

// ---------------------------------------------------------------------------
// Host planning of stateful rows
// ---------------------------------------------------------------------------

/// Lists every row's valid columns of a row-major `[batch, sequence]` mask.
///
/// An absent mask marks every column valid. Row `r` of the result is exactly
/// the strictly increasing list of valid columns of mask row `r`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rows =>
    requires
        mask matches Some(flags) ==> flags.len() == batch * sequence,
    ensures
        rows.len() == batch,
        forall|r: int| 0 <= r < batch ==> {
            &&& #[trigger] rows@[r]@ == row_columns(mask, sequence as int, r)
            &&& lists_valid_columns(mask_row(mask, sequence as int, r), rows@[r]@)
        },
))]
pub(crate) fn row_valid_columns(
    mask: Option<&[bool]>,
    batch: usize,
    sequence: usize,
) -> Vec<Vec<usize>> {
    let mut rows: Vec<Vec<usize>> = Vec::with_capacity(batch);
    // Every dense row lists `0..sequence`; plan it once and copy it per row.
    let dense = match mask {
        Some(_) => Vec::new(),
        None => {
            let flags = vec![true; sequence];
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(flags@ =~= mask_row(mask, sequence as int, 0));
            }
            valid_columns(&flags)
        }
    };
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= batch,
            mask matches Some(flags) ==> flags.len() == batch * sequence,
            mask is None ==> dense@ == row_columns(mask, sequence as int, 0),
            mask is None ==> lists_valid_columns(mask_row(mask, sequence as int, 0), dense@),
            rows.len() == row,
            forall|r: int| 0 <= r < row ==> {
                &&& #[trigger] rows@[r]@ == row_columns(mask, sequence as int, r)
                &&& lists_valid_columns(mask_row(mask, sequence as int, r), rows@[r]@)
            },
        decreases batch - row,
    ))]
    while row < batch {
        let columns = match mask {
            Some(flags) => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    lemma_grid_row_in_bounds(row as int, batch as int, sequence as int);
                }
                let offset = row * sequence;
                valid_columns(&flags[offset..offset + sequence])
            }
            None => dense.clone(),
        };
        #[cfg(verus_keep_ghost)]
        proof! {
            if mask is None {
                assert(columns@ =~= dense@);
                assert(mask_row(mask, sequence as int, row as int)
                    =~= mask_row(mask, sequence as int, 0));
            }
        }
        rows.push(columns);
        row += 1;
    }
    rows
}

/// Returns every row's last valid column, or the final column `sequence - 1` for an empty row.
///
/// An absent mask marks every column valid, so every row reports
/// `sequence - 1`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(columns =>
    requires
        sequence > 0,
        mask matches Some(flags) ==> flags.len() == batch * sequence,
    ensures
        columns.len() == batch,
        forall|r: int| 0 <= r < batch ==> {
            &&& #[trigger] columns@[r] < sequence
            &&& last_valid_or_final(mask_row(mask, sequence as int, r), columns@[r] as int)
        },
))]
pub(crate) fn row_last_columns(mask: Option<&[bool]>, batch: usize, sequence: usize) -> Vec<usize> {
    let mut columns: Vec<usize> = Vec::with_capacity(batch);
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= batch,
            sequence > 0,
            mask matches Some(flags) ==> flags.len() == batch * sequence,
            columns.len() == row,
            forall|r: int| 0 <= r < row ==> {
                &&& #[trigger] columns@[r] < sequence
                &&& last_valid_or_final(mask_row(mask, sequence as int, r), columns@[r] as int)
            },
        decreases batch - row,
    ))]
    while row < batch {
        let column = match mask {
            Some(flags) => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    lemma_grid_row_in_bounds(row as int, batch as int, sequence as int);
                }
                let offset = row * sequence;
                match last_valid_column(&flags[offset..offset + sequence]) {
                    Some(column) => column,
                    None => sequence - 1,
                }
            }
            None => sequence - 1,
        };
        #[cfg(verus_keep_ghost)]
        proof! {
            let flags = mask_row(mask, sequence as int, row as int);
            if mask is None {
                assert(flags[sequence - 1]);
            }
        }
        columns.push(column);
        row += 1;
    }
    columns
}

/// Lists `(row, context)` for every row that appends at least one token, in row order.
///
/// The result contains exactly the rows with `counts[row] > 0`, each once and
/// in increasing order, paired with `contexts[row]`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(active =>
    requires
        contexts.len() == counts.len(),
    ensures
        forall|k: int| 0 <= k < active.len() ==> {
            &&& #[trigger] active@[k].0 < counts.len()
            &&& counts[active@[k].0 as int] > 0
            &&& active@[k].1 == contexts[active@[k].0 as int]
        },
        forall|a: int, b: int| 0 <= a < b < active.len()
            ==> #[trigger] active@[a].0 < #[trigger] active@[b].0,
        forall|r: int| 0 <= r < counts.len() && #[trigger] counts[r] > 0
            ==> exists|k: int| 0 <= k < active.len() && active@[k].0 == r,
))]
pub(crate) fn active_row_contexts(contexts: &[usize], counts: &[usize]) -> Vec<(usize, usize)> {
    let mut active: Vec<(usize, usize)> = Vec::new();
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            contexts.len() == counts.len(),
            forall|k: int| 0 <= k < active.len() ==> {
                &&& #[trigger] active@[k].0 < row
                &&& counts[active@[k].0 as int] > 0
                &&& active@[k].1 == contexts[active@[k].0 as int]
            },
            forall|a: int, b: int| 0 <= a < b < active.len()
                ==> #[trigger] active@[a].0 < #[trigger] active@[b].0,
            forall|r: int| 0 <= r < row && #[trigger] counts[r] > 0
                ==> exists|k: int| 0 <= k < active.len() && active@[k].0 == r,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] > 0 {
            #[cfg(verus_keep_ghost)]
            proof_decl! {
                let ghost before = active@;
            }
            active.push((row, contexts[row]));
            #[cfg(verus_keep_ghost)]
            proof! {
                assert forall|r: int| 0 <= r < row + 1 && #[trigger] counts[r] > 0
                    implies exists|k: int| 0 <= k < active.len() && active@[k].0 == r by {
                    if r < row {
                        let k = choose|k: int| 0 <= k < before.len() && before[k].0 == r;
                        assert(active@[k] == before[k]);
                    } else {
                        assert(active@[active.len() - 1].0 == r);
                    }
                }
            }
        }
        row += 1;
    }
    active
}

/// Why a row's valid token or position values are rejected.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowValueError {
    /// A valid token lies outside `0..vocabulary`; the observed range follows.
    Tokens {
        /// Smallest valid token ID of the row.
        minimum: i64,
        /// Largest valid token ID of the row.
        maximum: i64,
    },
    /// A valid explicit position lies outside `0..limit`.
    Positions {
        /// Smallest valid position of the row.
        minimum: i64,
        /// Largest valid position of the row.
        maximum: i64,
        /// Exclusive position bound.
        limit: usize,
    },
    /// The greatest valid position has no context extent `max + 1` in `usize`.
    Unrepresentable,
}

/// Validates every row's valid values and resolves its rotary extent.
///
/// Row `r` reads `tokens[r·S .. r·S + S)` and, when supplied, the matching
/// positions, at its valid columns only, so padding sentinels are inert.
/// Tokens must lie in `0..vocabulary`; explicit positions in `0..limit` when
/// a limit is given. The extent is `1 + max p` for explicit positions and the
/// implicit cursor `ends[r]` otherwise. The first failing row's error is
/// returned, tokens before positions.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(contexts =>
    requires
        tokens.len() == valid_indices.len() * sequence,
        positions matches Some(values) ==> values.len() == tokens.len(),
        ends.len() == valid_indices.len(),
        forall|r: int, k: int| 0 <= r < valid_indices.len() && 0 <= k < valid_indices@[r].len()
            ==> #[trigger] valid_indices@[r]@[k] < sequence,
    ensures
        match contexts {
            Ok(contexts) => {
                &&& contexts.len() == valid_indices.len()
                &&& forall|r: int| 0 <= r < valid_indices.len() ==> #[trigger] row_context_ok(
                    tokens@,
                    positions,
                    valid_indices@[r]@,
                    sequence as int,
                    r,
                    vocabulary as int,
                    position_limit,
                    ends[r] as int,
                    contexts@[r] as int,
                )
            },
            Err(error) => exists|row: int| 0 <= row < valid_indices.len() && {
                &&& forall|r: int| 0 <= r < row ==> #[trigger] row_succeeds(
                    tokens@,
                    positions,
                    valid_indices@[r]@,
                    sequence as int,
                    r,
                    vocabulary as int,
                    position_limit,
                    ends[r] as int,
                )
                &&& row_value_error(
                    tokens@,
                    positions,
                    valid_indices@[row]@,
                    sequence as int,
                    row,
                    vocabulary as int,
                    position_limit,
                    error,
                )
            },
        },
))]
// A separate solver keeps this proof independent of the other queries in the module.
#[cfg_attr(verus_keep_ghost, verifier::spinoff_prover)]
// Verus compiles verified bodies as Rust 2021, which has no `let` chains, and
// proof blocks separate the nested conditions in Verus builds.
#[allow(clippy::collapsible_if)]
pub(crate) fn row_contexts(
    tokens: &[i64],
    positions: Option<&[i64]>,
    valid_indices: &[Vec<usize>],
    ends: &[usize],
    sequence: usize,
    vocabulary: usize,
    position_limit: Option<usize>,
) -> Result<Vec<usize>, RowValueError> {
    let rows = valid_indices.len();
    let mut contexts: Vec<usize> = Vec::with_capacity(rows);
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows == valid_indices.len(),
            tokens.len() == rows * sequence,
            positions matches Some(values) ==> values.len() == tokens.len(),
            ends.len() == rows,
            forall|r: int, k: int| 0 <= r < rows && 0 <= k < valid_indices@[r].len()
                ==> #[trigger] valid_indices@[r]@[k] < sequence,
            contexts.len() == row,
            forall|r: int| 0 <= r < row ==> #[trigger] row_context_ok(
                tokens@,
                positions,
                valid_indices@[r]@,
                sequence as int,
                r,
                vocabulary as int,
                position_limit,
                ends[r] as int,
                contexts@[r] as int,
            ),
        decreases rows - row,
    ))]
    while row < rows {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_grid_row_in_bounds(row as int, rows as int, sequence as int);
        }
        let offset = row * sequence;
        let columns = valid_indices[row].as_slice();
        let token_row = &tokens[offset..offset + sequence];
        if let Some((minimum, maximum)) = value_range(token_row, columns) {
            if !range_within(minimum, maximum, vocabulary) {
                #[cfg(verus_keep_ghost)]
                proof! {
                    lemma_range_within(token_row@, columns@, minimum, maximum, vocabulary as int);
                    lemma_rows_before_succeed(
                        tokens@, positions, valid_indices@, ends@, sequence as int, row as int,
                        vocabulary as int, position_limit, contexts@,
                    );
                }
                return Err(RowValueError::Tokens { minimum, maximum });
            }
            #[cfg(verus_keep_ghost)]
            proof! {
                lemma_range_within(token_row@, columns@, minimum, maximum, vocabulary as int);
            }
        }
        let explicit = match positions {
            Some(values) => value_range(&values[offset..offset + sequence], columns),
            None => None,
        };
        let context = match explicit {
            Some((minimum, maximum)) => {
                if let Some(limit) = position_limit {
                    if !range_within(minimum, maximum, limit) {
                        #[cfg(verus_keep_ghost)]
                        proof! {
                            let values = grid_row(positions.unwrap()@, sequence as int, row as int);
                            lemma_range_within(values, columns@, minimum, maximum, limit as int);
                            lemma_rows_before_succeed(
                                tokens@, positions, valid_indices@, ends@, sequence as int,
                                row as int, vocabulary as int, position_limit, contexts@,
                            );
                        }
                        return Err(RowValueError::Positions {
                            minimum,
                            maximum,
                            limit,
                        });
                    }
                    #[cfg(verus_keep_ghost)]
                    proof! {
                        let values = grid_row(positions.unwrap()@, sequence as int, row as int);
                        lemma_range_within(values, columns@, minimum, maximum, limit as int);
                    }
                }
                match context_from_maximum(maximum) {
                    Some(context) => context,
                    None => {
                        #[cfg(verus_keep_ghost)]
                        proof! {
                            let values = grid_row(positions.unwrap()@, sequence as int, row as int);
                            assert(is_maximum_at(values, columns@, maximum as int));
                            lemma_rows_before_succeed(
                                tokens@, positions, valid_indices@, ends@, sequence as int,
                                row as int, vocabulary as int, position_limit, contexts@,
                            );
                        }
                        return Err(RowValueError::Unrepresentable);
                    }
                }
            }
            None => ends[row],
        };
        contexts.push(context);
        row += 1;
    }
    Ok(contexts)
}

#[cfg(verus_keep_ghost)]
verus! {

/// For a nonempty list with attained bounds, `range_within` is the per-value range check.
pub proof fn lemma_range_within(
    values: Seq<i64>,
    columns: Seq<usize>,
    minimum: i64,
    maximum: i64,
    limit: int,
)
    requires
        columns.len() > 0,
        is_minimum_at(values, columns, minimum as int),
        is_maximum_at(values, columns, maximum as int),
    ensures
        values_within(values, columns, limit)
            == (0 <= minimum && 0 <= maximum && maximum < limit),
{
    if values_within(values, columns, limit) {
        let low = choose|k: int| 0 <= k < columns.len() && values[columns[k] as int] == minimum;
        let high = choose|k: int| 0 <= k < columns.len() && values[columns[k] as int] == maximum;
        assert(0 <= values[columns[low] as int] < limit);
        assert(0 <= values[columns[high] as int] < limit);
    }
}

/// Rows already planned keep a witness context, as error postconditions require.
pub proof fn lemma_rows_before_succeed(
    tokens: Seq<i64>,
    positions: Option<&[i64]>,
    valid_indices: Seq<Vec<usize>>,
    ends: Seq<usize>,
    sequence: int,
    row: int,
    vocabulary: int,
    position_limit: Option<usize>,
    contexts: Seq<usize>,
)
    requires
        contexts.len() == row,
        forall|r: int| 0 <= r < row ==> #[trigger] row_context_ok(
            tokens,
            positions,
            valid_indices[r]@,
            sequence,
            r,
            vocabulary,
            position_limit,
            ends[r] as int,
            contexts[r] as int,
        ),
    ensures
        forall|r: int| 0 <= r < row ==> #[trigger] row_succeeds(
            tokens,
            positions,
            valid_indices[r]@,
            sequence,
            r,
            vocabulary,
            position_limit,
            ends[r] as int,
        ),
{
    assert forall|r: int| 0 <= r < row implies #[trigger] row_succeeds(
        tokens,
        positions,
        valid_indices[r]@,
        sequence,
        r,
        vocabulary,
        position_limit,
        ends[r] as int,
    ) by {
        assert(row_context_ok(
            tokens,
            positions,
            valid_indices[r]@,
            sequence,
            r,
            vocabulary,
            position_limit,
            ends[r] as int,
            contexts[r] as int,
        ));
    }
}

} // verus!

/// Host plan of one validated rectangular stateful call.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct HostRows {
    /// Valid source columns of each row in increasing order.
    pub(crate) valid_indices: Vec<Vec<usize>>,
    /// Number of tokens appended to each row; zero marks an inactive row.
    pub(crate) token_counts: Vec<usize>,
    /// Per-row rotary extent: `1 + max p_b` (explicit) or `L_b + n_b` (implicit).
    pub(crate) evaluated_contexts: Vec<usize>,
    /// Each row's last valid column, or `S - 1` for an inactive row.
    pub(crate) last_columns: Vec<usize>,
}

/// Why the host side of a stateful call is rejected, in precedence order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostRowsError {
    /// A row has no valid token although empty rows are rejected.
    EmptyRow,
    /// A row's cache append overflows `usize` or the capacity.
    Append(
        /// The first impossible append.
        AppendError,
    ),
    /// A row's valid token or position values are invalid.
    Values(
        /// The first failing row's value error.
        RowValueError,
    ),
}

/// Returns each list's length.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(counts =>
    ensures
        counts.len() == rows.len(),
        forall|r: int| 0 <= r < rows.len() ==> #[trigger] counts@[r] == rows@[r].len(),
))]
fn list_lengths(rows: &[Vec<usize>]) -> Vec<usize> {
    let mut counts: Vec<usize> = Vec::with_capacity(rows.len());
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows.len(),
            counts.len() == row,
            forall|r: int| 0 <= r < row ==> #[trigger] counts@[r] == rows@[r].len(),
        decreases rows.len() - row,
    ))]
    while row < rows.len() {
        counts.push(rows[row].len());
        row += 1;
    }
    counts
}

/// Returns whether any entry of `values` is zero.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
        found == exists|j: int| 0 <= j < values.len() && #[trigger] values[j] == 0,
))]
fn contains_zero(values: &[usize]) -> bool {
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= values.len(),
            forall|j: int| 0 <= j < index ==> #[trigger] values[j] != 0,
        decreases values.len() - index,
    ))]
    while index < values.len() {
        if values[index] == 0 {
            return true;
        }
        index += 1;
    }
    false
}

/// Lists every row's valid columns of an optional mask together with their counts.
///
/// Row `r` lists exactly `row_columns(mask, sequence, r)`, every listed
/// column lies below `sequence`, and the count is the list's length.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(grid =>
    requires
        mask matches Some(flags) ==> flags.len() == batch * sequence,
    ensures
        grid.0.len() == batch,
        grid.1.len() == batch,
        forall|r: int|
            #![trigger grid.0@[r]]
            #![trigger grid.1@[r]]
            #![trigger row_columns(mask, sequence as int, r)]
            0 <= r < batch ==> {
                &&& grid.0@[r]@ == row_columns(mask, sequence as int, r)
                &&& grid.1@[r] == row_columns(mask, sequence as int, r).len()
            },
        forall|r: int, k: int| 0 <= r < batch && 0 <= k < grid.0@[r].len()
            ==> #[trigger] grid.0@[r]@[k] < sequence,
))]
fn plan_row_grid(
    mask: Option<&[bool]>,
    batch: usize,
    sequence: usize,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let valid_indices = row_valid_columns(mask, batch, sequence);
    let token_counts = list_lengths(valid_indices.as_slice());
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|r: int, k: int| 0 <= r < batch && 0 <= k < valid_indices@[r].len()
            implies #[trigger] valid_indices@[r]@[k] < sequence by {
            lemma_grid_row_in_bounds(r, batch as int, sequence as int);
            assert(lists_valid_columns(mask_row(mask, sequence as int, r), valid_indices@[r]@));
            assert(mask_row(mask, sequence as int, r).len() == sequence);
        }
    }
    (valid_indices, token_counts)
}

#[cfg(verus_keep_ghost)]
verus! {

/// Restates a row-context failure over planned lists as one over the mask's columns.
pub(crate) proof fn lemma_host_value_error(
    mask: Option<&[bool]>,
    tokens: Seq<i64>,
    positions: Option<&[i64]>,
    batch: int,
    sequence: int,
    lengths: Seq<usize>,
    vocabulary: int,
    position_limit: Option<usize>,
    valid_indices: Seq<Vec<usize>>,
    ends: Seq<usize>,
    error: RowValueError,
)
    requires
        forall|r: int| #![trigger valid_indices[r]] 0 <= r < batch ==> {
            &&& valid_indices[r]@ == row_columns(mask, sequence, r)
            &&& ends[r] == lengths[r] + row_columns(mask, sequence, r).len()
        },
        exists|row: int| 0 <= row < batch && {
            &&& forall|r: int| 0 <= r < row ==> #[trigger] row_succeeds(
                tokens,
                positions,
                valid_indices[r]@,
                sequence,
                r,
                vocabulary,
                position_limit,
                ends[r] as int,
            )
            &&& row_value_error(
                tokens,
                positions,
                valid_indices[row]@,
                sequence,
                row,
                vocabulary,
                position_limit,
                error,
            )
        },
    ensures
        exists|row: int| 0 <= row < batch && {
            &&& forall|r: int| 0 <= r < row ==> #[trigger] row_succeeds(
                tokens,
                positions,
                row_columns(mask, sequence, r),
                sequence,
                r,
                vocabulary,
                position_limit,
                lengths[r] + row_columns(mask, sequence, r).len(),
            )
            &&& row_value_error(
                tokens,
                positions,
                row_columns(mask, sequence, row),
                sequence,
                row,
                vocabulary,
                position_limit,
                error,
            )
        },
{
    let row = choose|row: int| 0 <= row < batch && {
        &&& forall|r: int| 0 <= r < row ==> #[trigger] row_succeeds(
            tokens,
            positions,
            valid_indices[r]@,
            sequence,
            r,
            vocabulary,
            position_limit,
            ends[r] as int,
        )
        &&& row_value_error(
            tokens,
            positions,
            valid_indices[row]@,
            sequence,
            row,
            vocabulary,
            position_limit,
            error,
        )
    };
    assert forall|r: int| 0 <= r < row implies #[trigger] row_succeeds(
        tokens,
        positions,
        row_columns(mask, sequence, r),
        sequence,
        r,
        vocabulary,
        position_limit,
        lengths[r] + row_columns(mask, sequence, r).len(),
    ) by {
        assert(valid_indices[r]@ == row_columns(mask, sequence, r));
        assert(row_succeeds(
            tokens,
            positions,
            valid_indices[r]@,
            sequence,
            r,
            vocabulary,
            position_limit,
            ends[r] as int,
        ));
    }
    assert(valid_indices[row]@ == row_columns(mask, sequence, row));
}

/// Restates successful row contexts over planned lists as ones over the mask's columns.
pub(crate) proof fn lemma_host_contexts(
    mask: Option<&[bool]>,
    tokens: Seq<i64>,
    positions: Option<&[i64]>,
    batch: int,
    sequence: int,
    lengths: Seq<usize>,
    vocabulary: int,
    position_limit: Option<usize>,
    valid_indices: Seq<Vec<usize>>,
    token_counts: Seq<usize>,
    ends: Seq<usize>,
    contexts: Seq<usize>,
)
    requires
        forall|r: int| #![trigger valid_indices[r]] 0 <= r < batch ==> {
            &&& valid_indices[r]@ == row_columns(mask, sequence, r)
            &&& ends[r] == lengths[r] + token_counts[r]
        },
        forall|r: int| 0 <= r < batch ==> #[trigger] row_context_ok(
            tokens,
            positions,
            valid_indices[r]@,
            sequence,
            r,
            vocabulary,
            position_limit,
            ends[r] as int,
            contexts[r] as int,
        ),
    ensures
        forall|r: int| 0 <= r < batch ==> #[trigger] row_context_ok(
            tokens,
            positions,
            row_columns(mask, sequence, r),
            sequence,
            r,
            vocabulary,
            position_limit,
            lengths[r] + token_counts[r],
            contexts[r] as int,
        ),
{
    assert forall|r: int| 0 <= r < batch implies #[trigger] row_context_ok(
        tokens,
        positions,
        row_columns(mask, sequence, r),
        sequence,
        r,
        vocabulary,
        position_limit,
        lengths[r] + token_counts[r],
        contexts[r] as int,
    ) by {
        assert(valid_indices[r]@ == row_columns(mask, sequence, r));
        assert(row_context_ok(
            tokens,
            positions,
            valid_indices[r]@,
            sequence,
            r,
            vocabulary,
            position_limit,
            ends[r] as int,
            contexts[r] as int,
        ));
    }
}

} // verus!

/// Plans the host side of one stateful call after its single readback.
///
/// The mask, token IDs, and optional explicit positions are row-major
/// `[batch, sequence]`; `lengths` are the per-row cursors the call appends
/// to. The plan lists every row's valid columns and last valid column,
/// rejects empty rows when `reject_empty_rows`, plans every append against
/// `capacity`, and validates valid values only, so padding is inert. The
/// proof shows the result is exactly `host_rows_outcome`, including the
/// precedence of its errors.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(planned =>
    requires
        sequence > 0,
        mask matches Some(flags) ==> flags.len() == batch * sequence,
        tokens.len() == batch * sequence,
        positions matches Some(values) ==> values.len() == batch * sequence,
        lengths.len() == batch,
    ensures
        host_rows_outcome(
            mask,
            tokens@,
            positions,
            batch as int,
            sequence as int,
            lengths@,
            capacity as int,
            reject_empty_rows,
            vocabulary as int,
            position_limit,
            planned,
        ),
))]
// A separate solver keeps this composition's proof independent of the
// order and content of the other queries in the module.
#[cfg_attr(verus_keep_ghost, verifier::spinoff_prover)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_host_rows(
    mask: Option<&[bool]>,
    tokens: &[i64],
    positions: Option<&[i64]>,
    batch: usize,
    sequence: usize,
    lengths: &[usize],
    capacity: usize,
    reject_empty_rows: bool,
    vocabulary: usize,
    position_limit: Option<usize>,
) -> Result<HostRows, HostRowsError> {
    let (valid_indices, token_counts) = plan_row_grid(mask, batch, sequence);
    let last_columns = row_last_columns(mask, batch, sequence);
    // Verus compiles verified bodies as Rust 2021, so this avoids a `let` chain.
    let empty_row = if reject_empty_rows {
        contains_zero(token_counts.as_slice())
    } else {
        false
    };
    if empty_row {
        return Err(HostRowsError::EmptyRow);
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        if reject_empty_rows {
            assert forall|r: int| 0 <= r < batch
                implies #[trigger] row_columns(mask, sequence as int, r).len() > 0 by {
                assert(token_counts@[r] != 0);
            }
        }
    }
    let ends = match plan_appends(lengths, token_counts.as_slice(), capacity) {
        Ok(ends) => ends,
        Err(error) => return Err(HostRowsError::Append(error)),
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|r: int| 0 <= r < batch implies #[trigger] ends@[r]
            == lengths[r] + row_columns(mask, sequence as int, r).len() by {
            assert(token_counts@[r] == row_columns(mask, sequence as int, r).len());
        }
        assert forall|r: int| 0 <= r < batch
            implies lengths[r] + #[trigger] row_columns(mask, sequence as int, r).len()
            <= capacity by {
            assert(ends@[r] <= capacity);
        }
    }
    let evaluated_contexts = match row_contexts(
        tokens,
        positions,
        valid_indices.as_slice(),
        ends.as_slice(),
        sequence,
        vocabulary,
        position_limit,
    ) {
        Ok(contexts) => contexts,
        Err(error) => {
            #[cfg(verus_keep_ghost)]
            proof! {
                lemma_host_value_error(
                    mask,
                    tokens@,
                    positions,
                    batch as int,
                    sequence as int,
                    lengths@,
                    vocabulary as int,
                    position_limit,
                    valid_indices@,
                    ends@,
                    error,
                );
            }
            return Err(HostRowsError::Values(error));
        }
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_host_contexts(
            mask,
            tokens@,
            positions,
            batch as int,
            sequence as int,
            lengths@,
            vocabulary as int,
            position_limit,
            valid_indices@,
            token_counts@,
            ends@,
            evaluated_contexts@,
        );
    }
    Ok(HostRows {
        valid_indices,
        token_counts,
        evaluated_contexts,
        last_columns,
    })
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
