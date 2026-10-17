//! Pure integer bounds shared by execution contexts, hooks, and training losses.
//!
//! These functions hold the index and range arithmetic behind tensor checks:
//! positional extents, value ranges read back from the device, broadcast
//! compatibility, staged block cursors, and the pairwise reduction schedule.
//! They use no Burn types, floating point, or allocation, so Verus checks these
//! exact bodies through `verus/experiment.rs` (see `verus/README.md`).
//!
//! # Verified properties
//!
//! - [`extent_after`] returns `high + 1` exactly when that fits in `usize`.
//! - [`range_within`] accepts exactly the ranges `0 <= low <= high < extent`.
//! - [`position_extrema`] classifies a position readback exactly: a value
//!   count other than the request, then a target range outside the extent,
//!   then a negative source minimum, so callers never index the readback.
//! - [`broadcasts_to`] accepts exactly the shapes whose every axis is one or equal.
//! - [`follows_cursor`] accepts exactly the ranges that resume at the cursor and
//!   stay within the stack; `lemma_accepted_run_tiles` proves that ranges
//!   accepted consecutively from cursor zero run every block below the final
//!   cursor exactly once and never run a block at or past it.
//! - [`reduction_rows`] multiplies every extent except the reduced axis, so
//!   rows times the reduced width is exactly the element count.
//! - [`fold_level`] returns the exact plan of one pairwise-reduction level:
//!   output column `j < pairs` receives exactly input columns `j` and
//!   `partner_start + j`, an odd width's middle column is carried into output
//!   column `pairs`, and no output column receives anything else, so every
//!   input column is read exactly once. The width shrinks to `ceil(w/2)`, one
//!   level fewer remains, and `lemma_fold_depth` proves the reduction takes
//!   exactly `ceil(log2(w))` levels.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Returns the positional extent `high + 1` for a nonnegative maximum position.
///
/// `None` means `high` is negative or `high + 1` does not fit in `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(extent =>
    ensures
        match extent {
            Some(extent) => 0 <= high && extent == high + 1,
            None => high < 0 || high + 1 > usize::MAX,
        },
))]
pub(crate) fn extent_after(high: i64) -> Option<usize> {
    if high < 0 {
        return None;
    }
    let high = high as u64;
    if high >= usize::MAX as u64 {
        return None;
    }
    Some(high as usize + 1)
}

/// Whether every value in `low..=high` lies in `0..extent`.
///
/// For device-read extrema (`low = min`, `high = max`) this is exactly the
/// check that every position or label indexes an `extent`-sized table.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(within =>
    ensures
        within == (0 <= low && low <= high && high < extent),
))]
pub(crate) fn range_within(low: i64, high: i64, extent: usize) -> bool {
    low >= 0 && low <= high && (high as u64) < (extent as u64)
}

/// Outcome of checking position extrema read back from the device.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PositionExtrema {
    /// The readback returned a different number of values than were requested.
    Incomplete,
    /// The target positions `low..=high` do not lie in `0..extent`.
    OutOfRange {
        /// Smallest target position.
        low: i64,
        /// Largest target position.
        high: i64,
    },
    /// The minimum source (encoder) position is negative.
    NegativeSource,
    /// Target positions lie in `0..extent` and any source minimum is nonnegative.
    Valid,
}

/// Classifies `[min, max]`, or `[min, max, source_min]` with an encoder, against `extent`.
///
/// The value count must match the request exactly, then the target range is
/// checked before the source minimum. The result is exact: each outcome holds
/// precisely under its stated condition, and the reported range is the one
/// read back, so the caller indexes nothing itself.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(check =>
    ensures
        (check == PositionExtrema::Incomplete)
            == (values.len() != if with_source { 3int } else { 2int }),
        match check {
            PositionExtrema::Incomplete => true,
            PositionExtrema::OutOfRange { low, high } => {
                &&& low == values@[0]
                &&& high == values@[1]
                &&& !(0 <= low && low <= high && high < extent)
            },
            PositionExtrema::NegativeSource => {
                &&& with_source
                &&& 0 <= values@[0] && values@[0] <= values@[1] && values@[1] < extent
                &&& values@[2] < 0
            },
            PositionExtrema::Valid => {
                &&& 0 <= values@[0] && values@[0] <= values@[1] && values@[1] < extent
                &&& with_source ==> values@[2] >= 0
            },
        },
))]
pub(crate) fn position_extrema(
    values: &[i64],
    extent: usize,
    with_source: bool,
) -> PositionExtrema {
    let expected = if with_source { 3 } else { 2 };
    if values.len() != expected {
        return PositionExtrema::Incomplete;
    }
    let (low, high) = (values[0], values[1]);
    if !range_within(low, high, extent) {
        return PositionExtrema::OutOfRange { low, high };
    }
    if with_source && values[2] < 0 {
        return PositionExtrema::NegativeSource;
    }
    PositionExtrema::Valid
}

/// Whether `actual` broadcasts to `expected`: every axis is one or equal.
///
/// Shapes of different rank never broadcast here; callers compare equal ranks.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(broadcasts =>
    ensures
        broadcasts == (actual.len() == expected.len() && forall|axis: int| 0 <= axis < actual.len() ==>
            actual@[axis] == 1 || actual@[axis] == expected@[axis]),
))]
pub(crate) fn broadcasts_to(actual: &[usize], expected: &[usize]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    let mut axis = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            0 <= axis <= actual.len(),
            actual.len() == expected.len(),
            forall|earlier: int| 0 <= earlier < axis ==>
                actual@[earlier] == 1 || actual@[earlier] == expected@[earlier],
        decreases actual.len() - axis,
    ))]
    while axis < actual.len() {
        if actual[axis] != 1 && actual[axis] != expected[axis] {
            return false;
        }
        axis += 1;
    }
    true
}

/// Whether blocks `start..end` resume a staged stack at `cursor` and stay within `total`.
///
/// Accepting a range and then setting `cursor = end` makes successive accepted
/// ranges contiguous and nondecreasing, so every block of `0..total` runs at
/// most once and in order. Empty ranges at the cursor are accepted.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(follows =>
    ensures
        follows == (start == cursor && start <= end && end <= total),
))]
pub(crate) fn follows_cursor(cursor: usize, start: usize, end: usize, total: usize) -> bool {
    start == cursor && start <= end && end <= total
}

/// Number of independent rows when axis `dim` of `shape` is reduced.
///
/// A reduction moves `dim` last and flattens the other axes into rows, so the
/// result is the product of every extent except `shape[dim]`. Multiplying it
/// by `shape[dim]` gives exactly the element count, so the `[rows, width]`
/// reshape is valid. Callers pass a validated nonempty tensor shape, whose
/// element count fits in `usize`; under that precondition no partial product
/// overflows.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rows =>
    requires
        dim < shape.len(),
        forall|axis: int| 0 <= axis < shape.len() ==> shape@[axis] > 0,
        shape_product(shape@) <= usize::MAX,
    ensures
        rows == product_except(shape@, dim as int),
        rows * shape@[dim as int] == shape_product(shape@),
        rows >= 1,
))]
pub(crate) fn reduction_rows(shape: &[usize], dim: usize) -> usize {
    let mut rows: usize = 1;
    let mut axis = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            dim < shape.len(),
            0 <= axis <= shape.len(),
            forall|other: int| 0 <= other < shape.len() ==> shape@[other] > 0,
            shape_product(shape@) <= usize::MAX,
            rows == product_except(shape@.subrange(0, axis as int), dim as int),
            rows >= 1,
        decreases shape.len() - axis,
    ))]
    while axis < shape.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            let next = shape@.subrange(0, axis as int + 1);
            assert(next.drop_last() == shape@.subrange(0, axis as int));
            lemma_product_except_at_most_product(next, dim as int);
            lemma_prefix_product_at_most(shape@, axis as int + 1);
        }
        if axis != dim {
            rows *= shape[axis];
        }
        axis += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(shape@.subrange(0, shape.len() as int) == shape@);
        lemma_product_except_times_extent(shape@, dim as int);
    }
    rows
}

/// One level of a pairwise row reduction: which columns are added and the next width.
///
/// The level adds column `j` to its partner `partner_start + j` for every
/// `j < pairs`, and an odd width carries its middle column `pairs` unchanged.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FoldLevel {
    /// Number of column pairs added at this level, `floor(width / 2)`.
    pub(crate) pairs: usize,
    /// First column of the right half, `width - pairs`; pair `j` adds columns `j` and `partner_start + j`.
    pub(crate) partner_start: usize,
    /// Width after this level, `ceil(width / 2)`: the pair sums plus any carried middle column.
    pub(crate) next: usize,
}

/// Plans one folding level of a pairwise row reduction of `width >= 2` columns.
///
/// Output column `j < pairs` is the sum of input columns `j` and
/// `partner_start + j`; for an odd width, output column `pairs` carries the
/// middle input column `pairs` unchanged. The proved postcondition says every
/// input column lands in one output column below `next`, each output column
/// receives exactly those inputs and no others, and the middle block
/// `pairs..partner_start` holds `width % 2` columns, so the level reads every
/// column exactly once and preserves the row sum. The next width is
/// `ceil(width / 2)`, at least one and strictly smaller than `width`, and the
/// remaining depth drops by exactly one, so repeated folding terminates after
/// exactly `ceil(log2(width))` levels (`lemma_fold_depth`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(level =>
    requires
        width >= 2,
    ensures
        level.pairs == width / 2,
        level.partner_start == width - level.pairs,
        level.next == level.partner_start,
        level.partner_start - level.pairs == width % 2,
        1 <= level.pairs,
        1 <= level.next < width,
        fold_depth(width as int) == fold_depth(level.next as int) + 1,
        forall|column: int| 0 <= column < width
            ==> 0 <= #[trigger] folded_column(level, column) < level.next,
        forall|pair: int| 0 <= pair < level.pairs ==> {
            &&& level.partner_start + pair < width
            &&& #[trigger] folded_column(level, pair) == pair
            &&& folded_column(level, level.partner_start + pair) == pair
        },
        level.pairs < level.next ==> folded_column(level, level.pairs as int) == level.pairs,
        forall|first: int, second: int|
            #![trigger folded_column(level, first), folded_column(level, second)]
            0 <= first < second < width && folded_column(level, first) == folded_column(level, second)
            ==> first < level.pairs && second == level.partner_start + first,
))]
pub(crate) fn fold_level(width: usize) -> FoldLevel {
    let pairs = width / 2;
    let partner_start = width - pairs;
    FoldLevel {
        pairs,
        partner_start,
        next: partner_start,
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// Product of every extent in `shape`.
pub(crate) open spec fn shape_product(shape: Seq<usize>) -> int
    decreases shape.len(),
{
    if shape.len() == 0 {
        1
    } else {
        shape_product(shape.drop_last()) * shape.last()
    }
}

/// Product of every extent in `shape` except the one at axis `dim`.
pub(crate) open spec fn product_except(shape: Seq<usize>, dim: int) -> int
    decreases shape.len(),
{
    if shape.len() == 0 {
        1
    } else {
        product_except(shape.drop_last(), dim) * if shape.len() - 1 == dim {
            1int
        } else {
            shape.last() as int
        }
    }
}

/// The product of the other extents times the reduced extent is the element count.
pub(crate) proof fn lemma_product_except_times_extent(shape: Seq<usize>, dim: int)
    requires
        0 <= dim < shape.len(),
    ensures
        product_except(shape, dim) * shape[dim] == shape_product(shape),
    decreases shape.len(),
{
    let prefix = shape.drop_last();
    if shape.len() - 1 == dim {
        lemma_product_except_skips_absent_axis(prefix, dim);
    } else {
        lemma_product_except_times_extent(prefix, dim);
        assert(prefix[dim] == shape[dim]);
        let others = product_except(prefix, dim);
        assert(others * shape.last() * shape[dim] == others * shape[dim] * shape.last())
            by (nonlinear_arith);
    }
}

/// Excluding an axis past the end of `shape` excludes nothing.
pub(crate) proof fn lemma_product_except_skips_absent_axis(shape: Seq<usize>, dim: int)
    requires
        dim >= shape.len(),
    ensures
        product_except(shape, dim) == shape_product(shape),
    decreases shape.len(),
{
    if shape.len() > 0 {
        lemma_product_except_skips_absent_axis(shape.drop_last(), dim);
    }
}

/// With positive extents, leaving an axis out never increases the product.
pub(crate) proof fn lemma_product_except_at_most_product(shape: Seq<usize>, dim: int)
    requires
        forall|axis: int| 0 <= axis < shape.len() ==> shape[axis] > 0,
    ensures
        1 <= product_except(shape, dim) <= shape_product(shape),
    decreases shape.len(),
{
    if shape.len() > 0 {
        let prefix = shape.drop_last();
        assert forall|axis: int| 0 <= axis < prefix.len() implies prefix[axis] > 0 by {
            assert(prefix[axis] == shape[axis]);
        }
        lemma_product_except_at_most_product(prefix, dim);
        let others = product_except(prefix, dim);
        let all = shape_product(prefix);
        let last = shape.last() as int;
        assert(last >= 1);
        if shape.len() - 1 == dim {
            assert(others * 1 <= all * last) by (nonlinear_arith)
                requires 1 <= others <= all, last >= 1;
        } else {
            assert(others * last <= all * last) by (nonlinear_arith)
                requires 1 <= others <= all, last >= 1;
            assert(others * last >= 1) by (nonlinear_arith)
                requires others >= 1, last >= 1;
        }
    }
}

/// With positive extents, the product of a prefix never exceeds the whole product.
pub(crate) proof fn lemma_prefix_product_at_most(shape: Seq<usize>, length: int)
    requires
        0 <= length <= shape.len(),
        forall|axis: int| 0 <= axis < shape.len() ==> shape[axis] > 0,
    ensures
        1 <= shape_product(shape.subrange(0, length)) <= shape_product(shape),
    decreases shape.len() - length,
{
    if length == shape.len() {
        assert(shape.subrange(0, length) == shape);
        lemma_product_except_at_most_product(shape, shape.len() as int);
        lemma_product_except_skips_absent_axis(shape, shape.len() as int);
    } else {
        lemma_prefix_product_at_most(shape, length + 1);
        let longer = shape.subrange(0, length + 1);
        assert(longer.drop_last() == shape.subrange(0, length));
        let partial = shape_product(shape.subrange(0, length));
        let extent = shape[length] as int;
        assert(longer.last() == shape[length]);
        assert(partial >= 1) by {
            lemma_product_except_at_most_product(shape.subrange(0, length), length);
            lemma_product_except_skips_absent_axis(shape.subrange(0, length), length);
        }
        assert(partial <= partial * extent) by (nonlinear_arith)
            requires partial >= 1, extent >= 1;
    }
}

/// Output column of a folding level that receives input `column`.
///
/// The left block and the carried middle column keep their index; the right
/// block `partner_start..` shifts onto its partners `0..pairs`. This mirrors
/// the runtime layout: the pair sums are concatenated first, then the middle
/// column.
pub(crate) open spec fn folded_column(level: FoldLevel, column: int) -> int {
    if column < level.partner_start {
        column
    } else {
        column - level.partner_start
    }
}

/// Number of folding levels that reduce `width` columns to one.
pub(crate) open spec fn fold_depth(width: int) -> nat
    decreases width,
{
    if width <= 1 {
        0
    } else {
        1 + fold_depth(width - width / 2)
    }
}

/// `2^exponent`.
pub(crate) open spec fn power_of_two(exponent: nat) -> int
    decreases exponent,
{
    if exponent == 0 {
        1
    } else {
        2 * power_of_two((exponent - 1) as nat)
    }
}

/// Folding `width` columns takes exactly `ceil(log2(width))` levels.
///
/// `2^(depth - 1) < width <= 2^depth`, so each input passes through at most
/// `ceil(log2(width))` two-term additions.
pub(crate) proof fn lemma_fold_depth(width: int)
    requires
        width >= 1,
    ensures
        width <= power_of_two(fold_depth(width)),
        fold_depth(width) > 0 ==> power_of_two((fold_depth(width) - 1) as nat) < width,
    decreases width,
{
    if width > 1 {
        let next = width - width / 2;
        lemma_fold_depth(next);
        let depth = fold_depth(next);
        assert(fold_depth(width) == depth + 1);
        assert(power_of_two(depth + 1) == 2 * power_of_two(depth));
        if depth > 0 {
            assert(power_of_two(depth) == 2 * power_of_two((depth - 1) as nat));
        }
    }
}

/// Cursor value before staged range `index`: zero, then the previous range's end.
pub(crate) open spec fn cursor_before(ranges: Seq<core::ops::Range<usize>>, index: int) -> int {
    if index <= 0 {
        0
    } else {
        ranges[index - 1].end as int
    }
}

/// Whether every range was accepted by [`follows_cursor`] with the cursor advanced to its end.
pub(crate) open spec fn accepted_run(ranges: Seq<core::ops::Range<usize>>, total: int) -> bool {
    forall|index: int| #![trigger ranges[index]] 0 <= index < ranges.len() ==> {
        &&& ranges[index].start as int == cursor_before(ranges, index)
        &&& ranges[index].start <= ranges[index].end
        &&& ranges[index].end as int <= total
    }
}

/// Accepted ranges never move backwards: an earlier range ends before a later one starts.
pub(crate) proof fn lemma_accepted_run_monotone(
    ranges: Seq<core::ops::Range<usize>>,
    total: int,
    earlier: int,
    later: int,
)
    requires
        accepted_run(ranges, total),
        0 <= earlier < later < ranges.len(),
    ensures
        ranges[earlier].end <= ranges[later].start,
    decreases later - earlier,
{
    if later > earlier + 1 {
        lemma_accepted_run_monotone(ranges, total, earlier, later - 1);
        assert(ranges[later - 1].start <= ranges[later - 1].end);
    }
    assert(ranges[later].start as int == cursor_before(ranges, later));
}

/// Consecutively accepted staged ranges run every block of `0..cursor` exactly once, in order.
///
/// After accepting `ranges` from cursor zero (setting the cursor to each end),
/// the final cursor is at most `total`, each block below it lies in exactly
/// one range, and no block at or above it was run.
pub(crate) proof fn lemma_accepted_run_tiles(
    ranges: Seq<core::ops::Range<usize>>,
    total: int,
    block: int,
)
    requires
        accepted_run(ranges, total),
        ranges.len() > 0,
    ensures
        0 <= ranges[ranges.len() - 1].end <= total,
        (0 <= block < ranges[ranges.len() - 1].end) ==> exists|index: int| #![trigger ranges[index]] {
            &&& 0 <= index < ranges.len()
            &&& ranges[index].start <= block < ranges[index].end
        },
        forall|first: int, second: int| #![trigger ranges[first], ranges[second]] {
            &&& 0 <= first < ranges.len()
            &&& 0 <= second < ranges.len()
            &&& ranges[first].start <= block < ranges[first].end
            &&& ranges[second].start <= block < ranges[second].end
        } ==> first == second,
        block >= ranges[ranges.len() - 1].end ==> forall|index: int| #![trigger ranges[index]] 0 <= index < ranges.len()
            ==> !(ranges[index].start <= block < ranges[index].end),
    decreases ranges.len(),
{
    let last = ranges.len() - 1;
    assert(ranges[last].end as int <= total);
    assert forall|first: int, second: int| #![trigger ranges[first], ranges[second]] {
        &&& 0 <= first < ranges.len()
        &&& 0 <= second < ranges.len()
        &&& ranges[first].start <= block < ranges[first].end
        &&& ranges[second].start <= block < ranges[second].end
    } implies first == second by {
        if first < second {
            lemma_accepted_run_monotone(ranges, total, first, second);
        } else if second < first {
            lemma_accepted_run_monotone(ranges, total, second, first);
        }
    }
    assert forall|index: int| #![trigger ranges[index]] 0 <= index < ranges.len() && block >= ranges[last].end
        implies !(ranges[index].start <= block < ranges[index].end) by {
        if index < last {
            lemma_accepted_run_monotone(ranges, total, index, last);
        }
    }
    if 0 <= block < ranges[last].end {
        if block >= ranges[last].start {
            assert(ranges[last].start <= block < ranges[last].end);
        } else {
            assert(last > 0) by {
                assert(ranges[last].start as int == cursor_before(ranges, last));
            }
            let prefix = ranges.subrange(0, last);
            assert(accepted_run(prefix, total)) by {
                assert forall|index: int| #![trigger prefix[index]] 0 <= index < prefix.len() implies {
                    &&& prefix[index].start as int == cursor_before(prefix, index)
                    &&& prefix[index].start <= prefix[index].end
                    &&& prefix[index].end as int <= total
                } by {
                    assert(prefix[index] == ranges[index]);
                    assert(ranges[index].start as int == cursor_before(ranges, index));
                    if index > 0 {
                        assert(prefix[index - 1] == ranges[index - 1]);
                    }
                }
            }
            assert(ranges[last].start as int == ranges[last - 1].end);
            lemma_accepted_run_tiles(prefix, total, block);
            let index = choose|index: int| #![trigger prefix[index]] {
                &&& 0 <= index < prefix.len()
                &&& prefix[index].start <= block < prefix[index].end
            };
            assert(prefix[index] == ranges[index]);
        }
    }
}

} // verus!

#[cfg(test)]
#[path = "bounds_tests.rs"]
mod tests;
