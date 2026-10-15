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
//! # Verified planning
//!
//! Every function in this file is dependency-free and checked in place by
//! Verus through `verus/models.rs` (see `verus/README.md`). For all admissible
//! inputs the proofs establish freedom from integer overflow and the stated
//! postconditions: compaction lists exactly the valid columns in increasing
//! order; scatter plans place compact entry `k` at column `c_k` and zeros
//! elsewhere; append plans either return `length + count <= capacity` for every
//! row or report the first failing row; and value ranges are attained bounds.
//! Tensor programs driven by these plans remain covered by the Rust tests.

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
            sorted_unique_below(columns@, column as int),
            forall|i: int| 0 <= i < columns.len() ==> row[#[trigger] columns[i] as int],
            forall|c: int| 0 <= c < column && #[trigger] row[c] ==> columns@.contains(c as usize),
        decreases row.len() - column,
    ))]
    while column < row.len() {
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
            },
            ScatterLayout::Gather { sources } =>
                gathers_into_columns(indices@, sequence as int, sources@),
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
                &&& forall|k: int| 0 <= k < columns.len()
                    ==> minimum <= #[trigger] values[columns[k] as int] <= maximum
                &&& exists|k: int| 0 <= k < columns.len() && values[columns[k] as int] == minimum
                &&& exists|k: int| 0 <= k < columns.len() && values[columns[k] as int] == maximum
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

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
