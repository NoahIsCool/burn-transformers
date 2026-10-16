//! Dependency-free planning for batched Nemotron-H execution.
//!
//! # Compact batches
//!
//! A stateful call receives a rectangular `[B, S]` batch in which row `b`
//! holds `n_b` valid tokens at columns `c_{b,0} < … < c_{b,n_b-1}`. The model
//! evaluates every row in one pass over a *right-padded compact* batch
//! `[B, N]` with `N = max_b n_b`: slot `(b, j)` holds column `c_{b,j}` for
//! `j < n_b` and is padding otherwise. Row-major flat indices make each
//! rearrangement a single index-select:
//!
//! ```text
//! gather[b N + j]  = b S + c_{b,j}            compact slot <- rectangular column
//! scatter[b S + s] = b N + j   if s = c_{b,j} rectangular column <- compact slot
//!                  = B N       otherwise      (an appended zero row)
//! packed[o_b + j]  = b N + j                  packed token <- compact slot
//! unpack[b N + j]  = o_b + j   if j < n_b     compact slot <- packed token
//!                  = T         otherwise      (an appended zero row)
//! last[b]          = b N + n_b - 1, or B N for an inactive row
//! ```
//!
//! with row offsets `o_b = sum_{r<b} n_r` and `T = sum_b n_b` valid tokens.
//! Sparse MoE blocks route only the `T` packed tokens; padded compact slots
//! never reach a router or an expert. Padded `gather` entries point at the
//! row's first rectangular column, which is always in range; callers replace
//! those tokens before embedding them.
//!
//! Attention queries take absolute positions from [`query_positions`]: slot
//! `(b, j)` sits at `W_b + j` after the row's `W_b` cached tokens, and a
//! padded slot repeats the row's last readable position, so causality and a
//! sliding window never exclude every key of a padded query in a row with
//! history.
//!
//! # Execution policies
//!
//! [`mamba_step`] chooses the Mamba-2 path of one row exactly as the pinned
//! reference does: the single-step update only for one token appended to an
//! existing state, the chunk scan otherwise. [`transactional_mamba`] decides
//! how a speculative step records recurrent state: one terminal state when
//! every row appends at most one token, per-token snapshots otherwise (which
//! requires existing state in every active row). [`mtp_length_after_truncate`]
//! keeps the shifted MTP cursor at most one position behind the target, and
//! [`verification_tokens`] sizes the one-pass target verification of a draft.
//!
//! # Hybrid caches and call bookkeeping
//!
//! Attention and Mamba-2 blocks occupy separate compact caches although the
//! schedule interleaves them: [`plan_cache_slots`] numbers each kind's blocks
//! `0, 1, …` in schedule order, and the same plan sizes both caches.
//! [`longest_end`] and [`active_window`] decide whether a sliding window can
//! exclude any key of a call, [`all_equal`] detects rows of equal history
//! (which need no position mask), [`exceeds_limit`] bounds every appended
//! length by the context, [`mtp_lengths_after_truncate`] maps a truncation to
//! every row's MTP cursor, and [`convolution_channels`] sizes the recurrent
//! convolution state.
//!
//! # Verified planning
//!
//! Every function and method here is dependency-free integer logic, checked
//! in place by Verus through `verus/nemotron.rs` (see `verus/README.md`).
//! For all admissible inputs the proofs establish freedom from overflow and
//! the following properties.
//!
//! - **Maps.** `N` is the largest row count. Each `gather`, `unpack`,
//!   `packed`, and `last` entry equals its equation above. `scatter` sends
//!   valid column `c_{b,k}` to slot `(b, k)` and every other column to the
//!   zero row, and `unpack` addresses the zero row exactly at padded slots.
//! - **Shortcuts.** A dense plan (`T = B N`, [`CompactPlan::is_dense`]) has
//!   full rows, identity `packed` and `unpack` maps, and
//!   `last[b] = b N + N - 1`. An identity plan (also `N = S`,
//!   [`CompactPlan::is_identity`]) has identity `gather` and `scatter` maps.
//! - **Refusals.** A plan is refused exactly for malformed columns or
//!   unrepresentable extents.
//! - **Positions.** Every slot's query position is exact, so a query never
//!   lies beyond its row's last readable key, and the refusal rule of
//!   [`query_positions`] holds in both directions.
//! - **Policies.** The Mamba-2 path, transactional staging, MTP cursor, and
//!   verification length follow their exact decision rules.
//! - **Bookkeeping.** Each cache kind's slots are exactly `0..count` in
//!   schedule order, so every block addresses a distinct allocated layer and
//!   every layer is used. The call bookkeeping above computes exact results.
//!
//! Tensor programs driven by the plans remain covered by Rust tests.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

use crate::models::shared::plan::is_sorted_unique_below;
#[cfg(verus_keep_ghost)]
use crate::models::shared::plan::sorted_unique_below;

#[cfg(verus_keep_ghost)]
verus! {

/// Total number of valid tokens in the first `rows` rows.
pub open spec fn total_tokens(columns: Seq<Vec<usize>>, rows: int) -> int
    decreases rows,
{
    if rows <= 0 {
        0
    } else {
        total_tokens(columns, rows - 1) + columns[rows - 1]@.len()
    }
}

/// Every entry of `values` is strictly below `bound`.
pub open spec fn all_below(values: Seq<usize>, bound: int) -> bool {
    forall|i: int| 0 <= i < values.len() ==> values[i] < bound
}

/// Every entry of `values` is at most `bound`.
pub open spec fn all_at_most(values: Seq<usize>, bound: int) -> bool {
    forall|i: int| 0 <= i < values.len() ==> values[i] <= bound
}

/// Row `r`'s slot range `[r * width, (r + 1) * width)` lies inside `batch * width`.
proof fn lemma_row_block(row: int, batch: int, width: int)
    requires
        0 <= row < batch,
        width >= 0,
    ensures
        row * width + width <= batch * width,
        row * width >= 0,
{
    assert(row * width + width <= batch * width) by (nonlinear_arith)
        requires 0 <= row < batch, width >= 0;
    assert(row * width >= 0) by (nonlinear_arith)
        requires row >= 0, width >= 0;
}

/// Adding one row's tokens keeps the running total within one more row block.
proof fn lemma_total_step(columns: Seq<Vec<usize>>, row: int)
    requires
        0 <= row < columns.len(),
    ensures
        total_tokens(columns, row + 1) == total_tokens(columns, row) + columns[row]@.len(),
{
}

/// Running totals never decrease: `total(rows) <= total(more)` for `rows <= more`.
proof fn lemma_total_monotone(columns: Seq<Vec<usize>>, rows: int, more: int)
    requires
        0 <= rows <= more <= columns.len(),
    ensures
        total_tokens(columns, rows) <= total_tokens(columns, more),
    decreases more - rows,
{
    if rows < more {
        lemma_total_monotone(columns, rows, more - 1);
    }
}

/// Running totals are never negative.
proof fn lemma_total_nonnegative(columns: Seq<Vec<usize>>, rows: int)
    ensures
        total_tokens(columns, rows) >= 0,
    decreases rows,
{
    if rows > 0 {
        lemma_total_nonnegative(columns, rows - 1);
    }
}

/// Row-major flat index of entry `(row, column)` in a matrix `width` columns wide.
pub open spec fn flat(row: int, column: int, width: int) -> int {
    row * width + column
}

/// An entry of an earlier row lies before the first entry of row `rows`.
proof fn lemma_flat_below(row: int, column: int, rows: int, width: int)
    requires
        0 <= row < rows,
        0 <= column < width,
    ensures
        0 <= flat(row, column, width) < rows * width,
{
    assert(row * width + width <= rows * width) by (nonlinear_arith)
        requires 0 <= row < rows, width >= 0;
    assert(row * width >= 0) by (nonlinear_arith)
        requires row >= 0, width >= 0;
}

/// Valid token `j` of row `row` precedes every token of later rows in packed order.
proof fn lemma_packed_below(columns: Seq<Vec<usize>>, row: int, rows: int, j: int)
    requires
        0 <= row < rows <= columns.len(),
        0 <= j < columns[row]@.len(),
    ensures
        0 <= total_tokens(columns, row) + j < total_tokens(columns, rows),
{
    lemma_total_step(columns, row);
    lemma_total_monotone(columns, row + 1, rows);
    lemma_total_nonnegative(columns, row);
}

/// Rows of at most `width` tokens total at most `rows * width`, with equality only for full rows.
proof fn lemma_total_at_most(columns: Seq<Vec<usize>>, rows: int, width: int)
    requires
        0 <= rows <= columns.len(),
        width >= 0,
        forall|b: int| 0 <= b < rows ==> #[trigger] columns[b]@.len() <= width,
    ensures
        total_tokens(columns, rows) <= rows * width,
        total_tokens(columns, rows) == rows * width
            ==> forall|b: int| 0 <= b < rows ==> #[trigger] columns[b]@.len() == width,
    decreases rows,
{
    if rows > 0 {
        lemma_total_at_most(columns, rows - 1, width);
        assert(rows * width == (rows - 1) * width + width) by (nonlinear_arith);
        assert(columns[rows - 1]@.len() <= width);
    }
}

/// Full rows of `width` tokens start at packed offset `rows * width`.
proof fn lemma_total_full(columns: Seq<Vec<usize>>, rows: int, width: int)
    requires
        0 <= rows <= columns.len(),
        forall|b: int| 0 <= b < rows ==> #[trigger] columns[b]@.len() == width,
    ensures
        total_tokens(columns, rows) == rows * width,
    decreases rows,
{
    if rows > 0 {
        lemma_total_full(columns, rows - 1, width);
        assert(rows * width == (rows - 1) * width + width) by (nonlinear_arith);
        assert(columns[rows - 1]@.len() == width);
    } else {
        assert(rows * width == 0) by (nonlinear_arith)
            requires rows == 0;
    }
}

/// Every flat index below `rows * width` is entry `(i / width, i % width)`.
proof fn lemma_flat_cover(i: int, rows: int, width: int)
    requires
        rows >= 0,
        0 <= i < rows * width,
    ensures
        width > 0,
        0 <= i / width < rows,
        0 <= i % width < width,
        i == flat(i / width, i % width, width),
{
    if width <= 0 {
        assert(rows * width <= 0) by (nonlinear_arith)
            requires width <= 0, rows >= 0;
    }
    assert(width > 0);
    assert(i == width * (i / width) + i % width) by (nonlinear_arith)
        requires width > 0;
    assert(0 <= i % width < width) by (nonlinear_arith)
        requires width > 0, i >= 0;
    assert(i / width >= 0) by (nonlinear_arith)
        requires width > 0, i >= 0;
    assert(i / width < rows) by (nonlinear_arith)
        requires width > 0, 0 <= i < rows * width, i == width * (i / width) + i % width,
            0 <= i % width;
    assert(width * (i / width) == (i / width) * width) by (nonlinear_arith);
}

/// Entry `j` of a strictly increasing sequence of naturals is at least `j`.
proof fn lemma_sorted_at_least(s: Seq<usize>, bound: int, j: int)
    requires
        sorted_unique_below(s, bound),
        0 <= j < s.len(),
    ensures
        s[j] >= j,
    decreases j,
{
    if j > 0 {
        lemma_sorted_at_least(s, bound, j - 1);
        assert(s[j - 1] < s[j]);
    }
}

/// Entry `j` of a strictly increasing sequence below `bound` leaves room for the rest.
proof fn lemma_sorted_at_most(s: Seq<usize>, bound: int, j: int)
    requires
        sorted_unique_below(s, bound),
        0 <= j < s.len(),
    ensures
        s[j] <= bound - (s.len() - j),
    decreases s.len() - j,
{
    if j + 1 < s.len() {
        lemma_sorted_at_most(s, bound, j + 1);
        assert(s[j] < s[j + 1]);
    } else {
        assert(s[j] < bound);
    }
}

/// Some row among the first `rows` has exactly `width` valid columns.
pub open spec fn attains(columns: Seq<Vec<usize>>, rows: int, width: int) -> bool {
    exists|b: int| 0 <= b < rows && #[trigger] columns[b]@.len() == width
}

/// The position [`slot_position`] assigns to slot `slot` of a row (see its documentation).
pub open spec fn slot_position_spec(baseline: int, count: int, slot: int) -> int {
    if slot < count {
        baseline + slot
    } else if baseline + count > 0 {
        baseline + count - 1
    } else {
        0
    }
}

/// Blocks of `kind` among the first `n` blocks of a schedule.
pub(crate) open spec fn kind_count(kinds: Seq<CacheKind>, n: int, kind: CacheKind) -> int
    decreases n,
{
    if n <= 0 {
        0
    } else {
        kind_count(kinds, n - 1, kind) + if kinds[n - 1] == kind { 1int } else { 0int }
    }
}

/// Counting one more block adds one exactly when it has the counted kind.
proof fn lemma_kind_count_step(kinds: Seq<CacheKind>, n: int, kind: CacheKind)
    requires
        0 <= n,
    ensures
        kind_count(kinds, n + 1, kind)
            == kind_count(kinds, n, kind) + if kinds[n] == kind { 1int } else { 0int },
{
}

/// Kind counts never decrease along the schedule.
proof fn lemma_kind_count_monotone(kinds: Seq<CacheKind>, n: int, more: int, kind: CacheKind)
    requires
        0 <= n <= more,
    ensures
        kind_count(kinds, n, kind) <= kind_count(kinds, more, kind),
    decreases more - n,
{
    if n < more {
        lemma_kind_count_monotone(kinds, n, more - 1, kind);
    }
}

/// Every count below `kind_count(n)` is the count before some block of `kind`.
proof fn lemma_kind_count_covers(kinds: Seq<CacheKind>, n: int, kind: CacheKind, s: int)
    requires
        0 <= s < kind_count(kinds, n, kind),
    ensures
        exists|i: int| 0 <= i < n && kinds[i] == kind
            && #[trigger] kind_count(kinds, i, kind) == s,
    decreases n,
{
    if n > 0 {
        if s < kind_count(kinds, n - 1, kind) {
            lemma_kind_count_covers(kinds, n - 1, kind, s);
            let i = choose|i: int| 0 <= i < n - 1 && kinds[i] == kind
                && #[trigger] kind_count(kinds, i, kind) == s;
            assert(kind_count(kinds, i, kind) == s);
        } else {
            assert(kind_count(kinds, n - 1, kind) == s);
        }
    }
}

/// Some block of `kind` has slot `s`.
pub(crate) open spec fn slot_used(
    kinds: Seq<CacheKind>,
    slots: Seq<usize>,
    kind: CacheKind,
    s: int,
) -> bool {
    exists|i: int| 0 <= i < kinds.len() && kinds[i] == kind && #[trigger] slots[i] == s
}

/// `b + c`, saturated at `usize::MAX` as `usize::saturating_add` does.
pub open spec fn saturated_end(baseline: int, count: int) -> int {
    if baseline + count > usize::MAX {
        usize::MAX as int
    } else {
        baseline + count
    }
}

/// The shifted MTP cursor [`mtp_length_after_truncate`] keeps: `min(mtp, max(target, 1) - 1)`.
pub open spec fn mtp_after_truncate(target: int, mtp: int) -> int {
    let limit = if target > 0 { target - 1 } else { 0 };
    if mtp < limit {
        mtp
    } else {
        limit
    }
}

/// Packed (row-major valid-token) index `o_b + j` of valid token `j` of row `b`.
pub open spec fn packed_index(columns: Seq<Vec<usize>>, row: int, j: int) -> int {
    total_tokens(columns, row) + j
}

/// Every flat entry of rows `0..rows` lies before the first entry of row `rows`.
pub open spec fn rows_below(rows: int, width: int) -> bool {
    forall|b: int, j: int| 0 <= b < rows && 0 <= j < width
        ==> 0 <= #[trigger] flat(b, j, width) < rows * width
}

/// Every valid token of rows `0..rows` precedes the packed tokens of row `rows`.
pub open spec fn packed_rows_below(columns: Seq<Vec<usize>>, rows: int) -> bool {
    forall|b: int, j: int| 0 <= b < rows && 0 <= j < columns[b]@.len()
        ==> 0 <= #[trigger] packed_index(columns, b, j) < total_tokens(columns, rows)
}

/// `gather[b N + j] = b S + c_{b,j}` for valid slots and `b S` for padded slots of rows `0..rows`.
pub open spec fn gathers_rows(
    gather: Seq<usize>,
    columns: Seq<Vec<usize>>,
    rows: int,
    width: int,
    sequence: int,
) -> bool {
    forall|b: int, j: int| 0 <= b < rows && 0 <= j < width
        ==> #[trigger] gather[flat(b, j, width)]
            == flat(b, if j < columns[b]@.len() { columns[b]@[j] as int } else { 0 }, sequence)
}

/// `unpack[b N + j] = o_b + j` for valid slots and `T` for padded slots of rows `0..rows`.
pub open spec fn unpacks_rows(
    unpack: Seq<usize>,
    columns: Seq<Vec<usize>>,
    rows: int,
    width: int,
    tokens: int,
) -> bool {
    forall|b: int, j: int| 0 <= b < rows && 0 <= j < width
        ==> #[trigger] unpack[flat(b, j, width)]
            == if j < columns[b]@.len() { packed_index(columns, b, j) } else { tokens }
}

/// `packed[o_b + j] = b N + j` for every valid token of rows `0..rows`.
pub open spec fn packs_rows(
    packed: Seq<usize>,
    columns: Seq<Vec<usize>>,
    rows: int,
    width: int,
) -> bool {
    forall|b: int, j: int| 0 <= b < rows && 0 <= j < columns[b]@.len()
        ==> #[trigger] packed[packed_index(columns, b, j)] == flat(b, j, width)
}

/// `scatter[b S + c_{b,k}] = b N + k`, and every other column of rows `0..rows` reads `padding`.
pub open spec fn scatters_rows(
    scatter: Seq<usize>,
    columns: Seq<Vec<usize>>,
    rows: int,
    width: int,
    sequence: int,
    padding: int,
) -> bool {
    &&& forall|b: int, k: int| 0 <= b < rows && 0 <= k < columns[b]@.len()
        ==> #[trigger] scatter[flat(b, columns[b]@[k] as int, sequence)] == flat(b, k, width)
    &&& forall|b: int, s: int| 0 <= b < rows && 0 <= s < sequence
        && !columns[b]@.contains(s as usize)
        ==> #[trigger] scatter[flat(b, s, sequence)] == padding
}

/// `last[b] = b N + n_b - 1`, or `padding` for an inactive row, for rows `0..rows`.
pub open spec fn lasts_rows(
    last: Seq<usize>,
    columns: Seq<Vec<usize>>,
    rows: int,
    width: int,
    padding: int,
) -> bool {
    forall|b: int| 0 <= b < rows
        ==> #[trigger] last[b] == if columns[b]@.len() == 0 {
            padding
        } else {
            flat(b, columns[b]@.len() - 1, width)
        }
}

} // verus!

/// Right-padded compact layout of one ragged batch (see the module documentation).
///
/// Per row, the valid `gather` entries (`gather[b N + k] = b S + c_{b,k}`)
/// and the valid `scatter` entries (`scatter[b S + c_{b,k}] = b N + k`) equal
/// the shared row plan ([`crate::models::shared::block_plan::row_maps`],
/// which Qwen3.5 and the DeepSeek/Muse block layout use) in flat
/// `[B * N]` / `[B * S]` form. The padding entries differ: a padded `gather`
/// entry reads the row's column 0 (`b S`) rather than `row_maps`' first valid
/// column `c_{b,0}`, and callers overwrite those tokens before embedding them;
/// a padded `scatter` entry reads the batch's single zero row `B N` rather
/// than a per-row zero slot. This plan stays separate because Nemotron-H also
/// needs what the per-row plan lacks: one flat select per tensor instead of a
/// per-row gather, the `packed`/`unpack` maps that feed only the `T` valid
/// tokens to the Mamba-2 scan and restore them, and each row's flat
/// last-token slot.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verifier::allow(autoderive_clone_without_spec))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompactPlan {
    /// Rectangular width `S`.
    pub(crate) sequence: usize,
    /// Compact width `N = max_b n_b`; zero when every row is inactive.
    pub(crate) width: usize,
    /// Valid tokens `n_b` of every row.
    pub(crate) counts: Vec<usize>,
    /// Flat rectangular source of every compact slot, `[B * N]`.
    pub(crate) gather: Vec<usize>,
    /// Flat compact source of every rectangular slot, `[B * S]`, `B * N` for padding.
    pub(crate) scatter: Vec<usize>,
    /// Flat compact slot of every valid token in row-major order, `[T]`.
    pub(crate) packed: Vec<usize>,
    /// Packed index of every compact slot, `[B * N]`, `T` for padding.
    pub(crate) unpack: Vec<usize>,
    /// Flat compact slot of each row's last token, `[B]`, `B * N` for inactive rows.
    pub(crate) last: Vec<usize>,
}

// The accessors restate in executable form the predicates under which
// `plan_compaction` proves its shortcuts, so the tensor wrappers' branches on
// `is_dense` and `is_identity` select exactly the proved identity maps.
impl CompactPlan {
    /// Returns the number of rows `B`.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(rows =>
        ensures
            rows == self.counts.len(),
    ))]
    pub(crate) fn batch(&self) -> usize {
        self.counts.len()
    }

    /// Returns the number of valid tokens `T`.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(tokens =>
        ensures
            tokens == self.packed.len(),
    ))]
    pub(crate) fn tokens(&self) -> usize {
        self.packed.len()
    }

    /// Returns whether every compact slot holds a valid token (`T = B * N`).
    ///
    /// This is exactly the dense-plan premise of [`plan_compaction`]'s proof,
    /// so a dense plan has full rows and identity `packed` and `unpack` maps.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(dense =>
        ensures
            dense == (self.packed.len() == self.unpack.len()),
    ))]
    pub(crate) fn is_dense(&self) -> bool {
        self.packed.len() == self.unpack.len()
    }

    /// Returns whether the compact batch equals the rectangular one (`N = S`, all valid).
    ///
    /// This is exactly the identity-plan premise of [`plan_compaction`]'s
    /// proof, so an identity plan also gathers and scatters in place.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(identity =>
        ensures
            identity == (self.width == self.sequence && self.packed.len() == self.unpack.len()),
    ))]
    pub(crate) fn is_identity(&self) -> bool {
        self.width == self.sequence && self.is_dense()
    }
}

/// Plans the compact layout of rows whose valid columns are `columns[b]`.
///
/// Returns `None` exactly when a row's columns are not strictly increasing
/// below `sequence`, or when a flat extent (`T`, `B * N`, or `B * S`) would
/// overflow `usize`. On success the proof establishes every equation of the
/// module documentation: `N` is the largest row count, `gather`, `unpack`,
/// `packed`, and `last` hold exactly their documented entries, `scatter`
/// sends valid column `c_{b,k}` to compact slot `(b, k)` and every other
/// column to the zero row `B * N`, and an `unpack` entry equals `T` exactly
/// at padded slots. Each map therefore stays inside its source.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(planned =>
    ensures
        match planned {
            Some(plan) => {
                let batch = columns.len() as int;
                let width = plan.width as int;
                let tokens = total_tokens(columns@, batch);
                &&& plan.sequence == sequence
                &&& plan.counts.len() == batch
                &&& forall|b: int| 0 <= b < batch ==> {
                    &&& #[trigger] plan.counts@[b] == columns@[b]@.len()
                    &&& plan.counts@[b] <= plan.width
                    &&& sorted_unique_below(columns@[b]@, sequence as int)
                }
                &&& plan.width > 0 ==> attains(columns@, batch, width)
                &&& plan.gather.len() == batch * width
                &&& plan.scatter.len() == batch * sequence
                &&& plan.unpack.len() == batch * width
                &&& plan.packed.len() == tokens
                &&& plan.last.len() == batch
                &&& gathers_rows(plan.gather@, columns@, batch, width, sequence as int)
                &&& unpacks_rows(plan.unpack@, columns@, batch, width, tokens)
                &&& packs_rows(plan.packed@, columns@, batch, width)
                &&& scatters_rows(
                    plan.scatter@,
                    columns@,
                    batch,
                    width,
                    sequence as int,
                    batch * width,
                )
                &&& lasts_rows(plan.last@, columns@, batch, width, batch * width)
                &&& forall|b: int, j: int| 0 <= b < batch && 0 <= j < width
                    ==> (#[trigger] plan.unpack@[flat(b, j, width)] == tokens)
                        == (j >= columns@[b]@.len())
                &&& all_below(plan.gather@, batch * sequence)
                &&& all_at_most(plan.scatter@, batch * width)
                &&& all_below(plan.packed@, batch * width)
                &&& all_at_most(plan.unpack@, tokens)
                &&& all_at_most(plan.last@, batch * width)
                // A dense plan (`T = B N`) has full rows and identity packing maps.
                &&& plan.packed.len() == plan.unpack.len() ==> {
                    &&& forall|b: int| 0 <= b < batch ==> #[trigger] plan.counts@[b] == plan.width
                    &&& forall|i: int| 0 <= i < plan.packed.len()
                        ==> #[trigger] plan.packed@[i] == i
                    &&& forall|i: int| 0 <= i < plan.unpack.len()
                        ==> #[trigger] plan.unpack@[i] == i
                    &&& plan.width > 0 ==> forall|b: int| 0 <= b < batch
                        ==> #[trigger] plan.last@[b] == flat(b, width - 1, width)
                }
                // An identity plan (dense with `N = S`) gathers and scatters in place.
                &&& plan.width == sequence && plan.packed.len() == plan.unpack.len() ==> {
                    &&& forall|i: int| 0 <= i < plan.gather.len()
                        ==> #[trigger] plan.gather@[i] == i
                    &&& forall|i: int| 0 <= i < plan.scatter.len()
                        ==> #[trigger] plan.scatter@[i] == i
                }
            },
            None => {
                ||| exists|b: int| 0 <= b < columns.len()
                    && !sorted_unique_below(#[trigger] columns@[b]@, sequence as int)
                ||| total_tokens(columns@, columns.len() as int) > usize::MAX
                ||| exists|b: int| 0 <= b < columns.len()
                    && columns.len() * (#[trigger] columns@[b]@.len()) > usize::MAX
                ||| columns.len() * sequence > usize::MAX
            },
        },
))]
pub(crate) fn plan_compaction(columns: &[Vec<usize>], sequence: usize) -> Option<CompactPlan> {
    let batch = columns.len();
    let mut width: usize = 0;
    let mut tokens: usize = 0;
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            batch == columns.len(),
            row <= batch,
            tokens == total_tokens(columns@, row as int),
            width > 0 ==> sequence > 0,
            width > 0 ==> attains(columns@, row as int, width as int),
            forall|b: int| 0 <= b < row ==> {
                &&& #[trigger] columns@[b]@.len() <= width
                &&& sorted_unique_below(columns@[b]@, sequence as int)
            },
        decreases batch - row,
    ))]
    while row < batch {
        if !is_sorted_unique_below(columns[row].as_slice(), sequence) {
            return None;
        }
        let count = columns[row].len();
        // A non-empty row has a column below `sequence`, so `sequence > 0`.
        #[cfg(verus_keep_ghost)]
        proof! {
            if count > 0 {
                assert(columns@[row as int]@[0] < sequence);
            }
            lemma_total_step(columns@, row as int);
            lemma_total_monotone(columns@, row as int + 1, batch as int);
        }
        if count > width {
            width = count;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            if width > 0 && width == count {
                assert(columns@[row as int]@.len() == width);
            }
        }
        tokens = tokens.checked_add(count)?;
        row += 1;
    }
    // An overflowing `B * N` is witnessed by a row whose count is `N`.
    #[cfg(verus_keep_ghost)]
    proof! {
        if batch * width > usize::MAX {
            if width == 0 {
                assert(batch * width == 0) by (nonlinear_arith)
                    requires width == 0;
            }
            let b = choose|b: int| 0 <= b < batch && #[trigger] columns@[b]@.len() == width;
            assert(columns@[b]@.len() == width);
        }
    }
    let compact_slots = batch.checked_mul(width)?;
    let rectangular_slots = batch.checked_mul(sequence)?;
    let mut counts: Vec<usize> = Vec::with_capacity(batch);
    let mut gather: Vec<usize> = Vec::with_capacity(compact_slots);
    let mut scatter: Vec<usize> = Vec::with_capacity(rectangular_slots);
    let mut packed: Vec<usize> = Vec::with_capacity(tokens);
    let mut unpack: Vec<usize> = Vec::with_capacity(compact_slots);
    let mut last: Vec<usize> = Vec::with_capacity(batch);
    // `compact_start = row * width` and `rectangular_start = row * sequence`
    // advance by addition, so no product is formed inside the loop.
    let mut compact_start: usize = 0;
    let mut rectangular_start: usize = 0;
    row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            batch == columns.len(),
            row <= batch,
            compact_slots == batch * width,
            rectangular_slots == batch * sequence,
            compact_start == row * width,
            rectangular_start == row * sequence,
            tokens == total_tokens(columns@, batch as int),
            width > 0 ==> sequence > 0,
            forall|b: int| 0 <= b < batch ==> {
                &&& #[trigger] columns@[b]@.len() <= width
                &&& sorted_unique_below(columns@[b]@, sequence as int)
            },
            counts.len() == row,
            forall|b: int| 0 <= b < row ==> #[trigger] counts@[b] == columns@[b]@.len(),
            gather.len() == row * width,
            unpack.len() == row * width,
            scatter.len() == row * sequence,
            last.len() == row,
            packed.len() == total_tokens(columns@, row as int),
            gathers_rows(gather@, columns@, row as int, width as int, sequence as int),
            unpacks_rows(unpack@, columns@, row as int, width as int, tokens as int),
            packs_rows(packed@, columns@, row as int, width as int),
            scatters_rows(
                scatter@,
                columns@,
                row as int,
                width as int,
                sequence as int,
                compact_slots as int,
            ),
            lasts_rows(last@, columns@, row as int, width as int, compact_slots as int),
            all_below(gather@, batch * sequence),
            all_at_most(scatter@, batch * width),
            all_below(packed@, batch * width),
            all_at_most(unpack@, tokens as int),
            all_at_most(last@, batch * width),
        decreases batch - row,
    ))]
    while row < batch {
        let valid = &columns[row];
        let count = valid.len();
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_row_block(row as int, batch as int, width as int);
            lemma_row_block(row as int, batch as int, sequence as int);
            lemma_total_step(columns@, row as int);
            assert(total_tokens(columns@, row as int + 1) <= total_tokens(columns@, batch as int))
                by {
                    lemma_total_monotone(columns@, row as int + 1, batch as int);
                };
            lemma_total_nonnegative(columns@, row as int);
            // Entries of earlier rows lie before everything this row appends.
            assert forall|b: int, j: int| 0 <= b < row && 0 <= j < width
                implies 0 <= #[trigger] flat(b, j, width as int) < row * width by {
                lemma_flat_below(b, j, row as int, width as int);
            }
            assert forall|b: int, j: int| 0 <= b < row && 0 <= j < sequence
                implies 0 <= #[trigger] flat(b, j, sequence as int) < row * sequence by {
                lemma_flat_below(b, j, row as int, sequence as int);
            }
            assert forall|b: int, j: int| 0 <= b < row && 0 <= j < columns@[b]@.len()
                implies 0 <= #[trigger] packed_index(columns@, b, j)
                    < total_tokens(columns@, row as int) by {
                lemma_packed_below(columns@, b, row as int, j);
            }
        }
        counts.push(count);
        let mut slot: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                batch == columns.len(),
                row < batch,
                valid@ == columns@[row as int]@,
                count == columns@[row as int]@.len(),
                count <= width,
                sorted_unique_below(columns@[row as int]@, sequence as int),
                rectangular_slots == batch * sequence,
                width > 0 ==> sequence > 0,
                compact_start == row * width,
                rectangular_start == row * sequence,
                compact_start + width <= batch * width,
                rectangular_start + sequence <= batch * sequence,
                compact_slots == batch * width,
                tokens == total_tokens(columns@, batch as int),
                total_tokens(columns@, row as int) >= 0,
                total_tokens(columns@, row as int + 1) <= tokens,
                total_tokens(columns@, row as int + 1)
                    == total_tokens(columns@, row as int) + count,
                slot <= width,
                gather.len() == row * width + slot,
                unpack.len() == row * width + slot,
                packed.len() == total_tokens(columns@, row as int)
                    + if slot < count { slot as int } else { count as int },
                rows_below(row as int, width as int),
                packed_rows_below(columns@, row as int),
                gathers_rows(gather@, columns@, row as int, width as int, sequence as int),
                unpacks_rows(unpack@, columns@, row as int, width as int, tokens as int),
                packs_rows(packed@, columns@, row as int, width as int),
                forall|j: int| 0 <= j < slot
                    ==> #[trigger] gather@[flat(row as int, j, width as int)] == flat(
                        row as int,
                        if j < count { valid@[j] as int } else { 0 },
                        sequence as int,
                    ),
                forall|j: int| 0 <= j < slot
                    ==> #[trigger] unpack@[flat(row as int, j, width as int)] == if j < count {
                        packed_index(columns@, row as int, j)
                    } else {
                        tokens as int
                    },
                forall|j: int| 0 <= j < slot && j < count
                    ==> #[trigger] packed@[packed_index(columns@, row as int, j)]
                        == flat(row as int, j, width as int),
                all_below(gather@, batch * sequence),
                all_below(packed@, batch * width),
                all_at_most(unpack@, tokens as int),
            decreases width - slot,
        ))]
        while slot < width {
            if slot < count {
                gather.push(rectangular_start + valid[slot]);
                unpack.push(packed.len());
                packed.push(compact_start + slot);
            } else {
                // Padding addresses the zero row appended after the packed tokens.
                gather.push(rectangular_start);
                unpack.push(tokens);
            }
            slot += 1;
        }
        // Valid columns increase, so one merge pass places compact slot `k`
        // at column `c_k` and the zero row everywhere else.
        let mut next: usize = 0;
        let mut column: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                batch == columns.len(),
                row < batch,
                valid@ == columns@[row as int]@,
                count == columns@[row as int]@.len(),
                count <= width,
                sorted_unique_below(columns@[row as int]@, sequence as int),
                forall|b: int| 0 <= b < batch ==> {
                    &&& #[trigger] columns@[b]@.len() <= width
                    &&& sorted_unique_below(columns@[b]@, sequence as int)
                },
                compact_start == row * width,
                rectangular_start == row * sequence,
                compact_start + width <= batch * width,
                compact_slots == batch * width,
                column <= sequence,
                next <= count,
                scatter.len() == row * sequence + column,
                rows_below(row as int, sequence as int),
                scatters_rows(
                    scatter@,
                    columns@,
                    row as int,
                    width as int,
                    sequence as int,
                    compact_slots as int,
                ),
                // Matched columns lie before the cursor; unmatched ones at or after it.
                forall|k: int| 0 <= k < next ==> #[trigger] valid@[k] < column,
                forall|k: int| next <= k < count ==> #[trigger] valid@[k] >= column,
                forall|k: int| 0 <= k < next
                    ==> #[trigger] scatter@[flat(row as int, valid@[k] as int, sequence as int)]
                        == flat(row as int, k, width as int),
                forall|s: int| 0 <= s < column && !valid@.contains(s as usize)
                    ==> #[trigger] scatter@[flat(row as int, s, sequence as int)] == compact_slots,
                all_at_most(scatter@, batch * width),
            decreases sequence - column,
        ))]
        while column < sequence {
            if next < count && valid[next] == column {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(valid@.contains(column));
                }
                scatter.push(compact_start + next);
                next += 1;
            } else {
                scatter.push(compact_slots);
            }
            column += 1;
        }
        // Every valid column lies below `sequence`, so the merge matched all of them.
        #[cfg(verus_keep_ghost)]
        proof! {
            if next < count {
                assert(valid@[next as int] >= column);
                assert(valid@[next as int] < sequence);
            }
        }
        last.push(if count == 0 {
            compact_slots
        } else {
            compact_start + count - 1
        });
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row as int + 1) * width as int == row as int * width as int + width as int)
                by (nonlinear_arith);
            assert((row as int + 1) * sequence as int
                == row as int * sequence as int + sequence as int)
                by (nonlinear_arith);
            let next_row = row as int + 1;
            assert forall|b: int, j: int| 0 <= b < next_row && 0 <= j < width
                implies #[trigger] gather@[flat(b, j, width as int)]
                    == flat(
                        b,
                        if j < columns@[b]@.len() { columns@[b]@[j] as int } else { 0 },
                        sequence as int,
                    )
            by {}
            assert forall|b: int, j: int| 0 <= b < next_row && 0 <= j < width
                implies #[trigger] unpack@[flat(b, j, width as int)]
                    == if j < columns@[b]@.len() {
                        packed_index(columns@, b, j)
                    } else {
                        tokens as int
                    }
            by {}
            assert forall|b: int, j: int| 0 <= b < next_row && 0 <= j < columns@[b]@.len()
                implies #[trigger] packed@[packed_index(columns@, b, j)] == flat(b, j, width as int)
            by {}
            assert forall|b: int, k: int| 0 <= b < next_row && 0 <= k < columns@[b]@.len()
                implies #[trigger] scatter@[flat(b, columns@[b]@[k] as int, sequence as int)]
                    == flat(b, k, width as int)
            by {}
            assert forall|b: int, s: int|
                0 <= b < next_row && 0 <= s < sequence && !columns@[b]@.contains(s as usize)
                implies #[trigger] scatter@[flat(b, s, sequence as int)] == compact_slots
            by {}
            assert forall|b: int| 0 <= b < next_row
                implies #[trigger] last@[b] == if columns@[b]@.len() == 0 {
                    compact_slots as int
                } else {
                    flat(b, columns@[b]@.len() - 1, width as int)
                }
            by {}
        }
        compact_start += width;
        rectangular_start += sequence;
        row += 1;
    }
    // An `unpack` entry equals `T` exactly at padded slots.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|b: int, j: int| 0 <= b < batch && 0 <= j < width
            implies (#[trigger] unpack@[flat(b, j, width as int)] == tokens)
                == (j >= columns@[b]@.len()) by {
            if j < columns@[b]@.len() {
                lemma_packed_below(columns@, b, batch as int, j);
            }
        }
        // `T = B N` forces full rows, so every packing map is the identity.
        if packed.len() == unpack.len() {
            let n = width as int;
            lemma_total_at_most(columns@, batch as int, n);
            assert forall|i: int| 0 <= i < packed.len() implies #[trigger] packed@[i] == i by {
                lemma_flat_cover(i, batch as int, n);
                let b = i / n;
                let j = i % n;
                lemma_total_full(columns@, b, n);
                assert(columns@[b]@.len() == n);
                assert(packed@[packed_index(columns@, b, j)] == flat(b, j, n));
            }
            assert forall|i: int| 0 <= i < unpack.len() implies #[trigger] unpack@[i] == i by {
                lemma_flat_cover(i, batch as int, n);
                let b = i / n;
                let j = i % n;
                lemma_total_full(columns@, b, n);
                assert(columns@[b]@.len() == n);
                assert(unpack@[flat(b, j, n)] == packed_index(columns@, b, j));
            }
            assert forall|b: int| 0 <= b < batch implies #[trigger] counts@[b] == width by {
                assert(columns@[b]@.len() == n);
            }
            if width > 0 {
                assert forall|b: int| 0 <= b < batch
                    implies #[trigger] last@[b] == flat(b, n - 1, n) by {
                    assert(columns@[b]@.len() == n);
                }
            }
            // Full rows of `S` strictly increasing columns below `S` are `0..S`.
            if width == sequence {
                assert forall|i: int| 0 <= i < gather.len() implies #[trigger] gather@[i] == i by {
                    lemma_flat_cover(i, batch as int, n);
                    let b = i / n;
                    let j = i % n;
                    assert(columns@[b]@.len() == n);
                    lemma_sorted_at_least(columns@[b]@, sequence as int, j);
                    lemma_sorted_at_most(columns@[b]@, sequence as int, j);
                    assert(columns@[b]@[j] == j);
                    assert(gather@[flat(b, j, n)]
                        == flat(b, columns@[b]@[j] as int, sequence as int));
                }
                assert forall|i: int| 0 <= i < scatter.len()
                    implies #[trigger] scatter@[i] == i by {
                    lemma_flat_cover(i, batch as int, n);
                    let b = i / n;
                    let j = i % n;
                    assert(columns@[b]@.len() == n);
                    lemma_sorted_at_least(columns@[b]@, sequence as int, j);
                    lemma_sorted_at_most(columns@[b]@, sequence as int, j);
                    assert(columns@[b]@[j] == j);
                    assert(scatter@[flat(b, columns@[b]@[j] as int, sequence as int)]
                        == flat(b, j, n));
                }
            }
        }
    }
    Some(CompactPlan {
        sequence,
        width,
        counts,
        gather,
        scatter,
        packed,
        unpack,
        last,
    })
}

/// Absolute query position of compact slot `slot` of a row with `count` new tokens.
///
/// The row's `count = n` valid tokens follow its `W = baseline` cached
/// ones. A valid slot (`slot < n`) sits at `W + slot`. A padded slot repeats
/// the row's last readable position `max(W + n, 1) - 1`, the position of its
/// last valid (or, for an inactive row, last cached) key.
///
/// Every key at or before that position is readable, and the key at that
/// position lies inside any sliding window. A padded query of a row with
/// history is therefore never fully excluded: its output is discarded, but it
/// stays a finite, ordinary softmax on every backend. Numbering padded slots
/// `W + slot` instead could push a padded query beyond a sliding window of
/// every valid key. Returns `None` exactly when `W + n` overflows.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(position =>
    ensures
        match position {
            Some(position) => {
                &&& baseline + count <= usize::MAX
                &&& position == slot_position_spec(baseline as int, count as int, slot as int)
                &&& slot < count ==> position == baseline + slot
                &&& slot >= count && baseline + count > 0 ==> position + 1 == baseline + count
                &&& slot >= count && baseline + count == 0 ==> position == 0
                &&& position <= baseline + slot
                &&& baseline + count > 0 ==> position < baseline + count
            },
            None => baseline + count > usize::MAX,
        },
))]
pub(crate) const fn slot_position(baseline: usize, count: usize, slot: usize) -> Option<usize> {
    match baseline.checked_add(count) {
        None => None,
        Some(_) if slot < count => Some(baseline + slot),
        Some(end) if end > 0 => Some(end - 1),
        Some(_) => Some(0),
    }
}

/// Absolute query positions of every compact slot `(b, j)`, `[B * N]`.
///
/// Slot `(b, j)` holds [`slot_position`]`(W_b, n_b, j)`: valid slots continue
/// the row's numbering and padded slots repeat its last readable position, so
/// no query of a row with history is ever fully excluded by causality or a
/// sliding window. Returns `None` exactly when `baselines` and `counts`
/// differ in length, `B * N` overflows, or some slot's position exceeds
/// `usize` or `i64`. The proof establishes both directions of that rule
/// (with `N > 0`, every returned row has a representable `W_b + n_b`) and
/// the exact value of every slot.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(positions =>
    ensures
        match positions {
            Some(positions) => {
                &&& counts.len() == baselines.len()
                &&& positions.len() == baselines.len() * width
                &&& forall|b: int, j: int| 0 <= b < baselines.len() && 0 <= j < width
                    ==> #[trigger] positions@[flat(b, j, width as int)]
                        == slot_position_spec(baselines@[b] as int, counts@[b] as int, j)
                &&& forall|i: int| 0 <= i < positions.len() ==> #[trigger] positions@[i] >= 0
                // Every slot's `W_b + n_b` was representable.
                &&& width > 0 ==> forall|b: int| 0 <= b < baselines.len()
                    ==> #[trigger] baselines@[b] + counts@[b] <= usize::MAX
            },
            None => {
                ||| counts.len() != baselines.len()
                ||| baselines.len() * width > usize::MAX
                ||| exists|b: int, j: int| 0 <= b < baselines.len() && 0 <= j < width && (
                    baselines@[b] + counts@[b] > usize::MAX
                    || #[trigger] slot_position_spec(baselines@[b] as int, counts@[b] as int, j)
                        > i64::MAX)
            },
        },
))]
pub(crate) fn query_positions(
    baselines: &[usize],
    counts: &[usize],
    width: usize,
) -> Option<Vec<i64>> {
    if counts.len() != baselines.len() {
        return None;
    }
    let total = baselines.len().checked_mul(width)?;
    let mut positions: Vec<i64> = Vec::with_capacity(total);
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            counts.len() == baselines.len(),
            row <= baselines.len(),
            positions.len() == row * width,
            forall|b: int, j: int| 0 <= b < row && 0 <= j < width
                ==> #[trigger] positions@[flat(b, j, width as int)]
                    == slot_position_spec(baselines@[b] as int, counts@[b] as int, j),
            forall|i: int| 0 <= i < positions.len() ==> #[trigger] positions@[i] >= 0,
            width > 0 ==> forall|b: int| 0 <= b < row
                ==> #[trigger] baselines@[b] + counts@[b] <= usize::MAX,
        decreases baselines.len() - row,
    ))]
    while row < baselines.len() {
        // Entries of earlier rows lie before everything this row appends.
        #[cfg(verus_keep_ghost)]
        proof! {
            assert forall|b: int, j: int| 0 <= b < row && 0 <= j < width
                implies 0 <= #[trigger] flat(b, j, width as int) < row * width by {
                lemma_flat_below(b, j, row as int, width as int);
            }
        }
        let mut slot: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                counts.len() == baselines.len(),
                row < baselines.len(),
                slot <= width,
                positions.len() == row * width + slot,
                rows_below(row as int, width as int),
                forall|b: int, j: int| 0 <= b < row && 0 <= j < width
                    ==> #[trigger] positions@[flat(b, j, width as int)]
                        == slot_position_spec(baselines@[b] as int, counts@[b] as int, j),
                forall|j: int| 0 <= j < slot
                    ==> #[trigger] positions@[flat(row as int, j, width as int)]
                        == slot_position_spec(
                            baselines@[row as int] as int,
                            counts@[row as int] as int,
                            j,
                        ),
                forall|i: int| 0 <= i < positions.len() ==> #[trigger] positions@[i] >= 0,
                width > 0 ==> forall|b: int| 0 <= b < row
                    ==> #[trigger] baselines@[b] + counts@[b] <= usize::MAX,
                slot > 0 ==> baselines@[row as int] + counts@[row as int] <= usize::MAX,
            decreases width - slot,
        ))]
        while slot < width {
            // Slot `(row, slot)` witnesses either failure below.
            #[cfg(verus_keep_ghost)]
            proof! {
                let ghost spec_position = slot_position_spec(
                    baselines@[row as int] as int,
                    counts@[row as int] as int,
                    slot as int,
                );
                assert(baselines@[row as int] + counts@[row as int] > usize::MAX
                    || spec_position > i64::MAX
                    ==> (0 <= row < baselines.len() && 0 <= slot < width && (
                        baselines@[row as int] + counts@[row as int] > usize::MAX
                        || slot_position_spec(
                            baselines@[row as int] as int,
                            counts@[row as int] as int,
                            slot as int,
                        ) > i64::MAX)));
            }
            let position = slot_position(baselines[row], counts[row], slot)?;
            // `usize` has at most 64 bits, so widening to `u64` is lossless.
            if position as u64 > i64::MAX as u64 {
                return None;
            }
            positions.push(position as i64);
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row as int + 1) * width as int == row as int * width as int + width as int)
                by (nonlinear_arith);
            assert forall|b: int, j: int| 0 <= b < row + 1 && 0 <= j < width
                implies #[trigger] positions@[flat(b, j, width as int)]
                    == slot_position_spec(baselines@[b] as int, counts@[b] as int, j) by {}
        }
        row += 1;
    }
    Some(positions)
}

/// Mamba-2 execution path of one stateful row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MambaStep {
    /// The reference single-step selective-state update.
    Recurrent,
    /// The reference chunked SSD scan (with its prefill dt clamp).
    Chunked,
}

/// Chooses the reference Mamba-2 path for `tokens` new tokens after `baseline` cached ones.
///
/// The pinned reference takes its single-step update only when
/// `use_precomputed_states and seq_len == 1`; a first prompt token, and every
/// multi-token segment, runs the chunk scan.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(step =>
    ensures
        (step == MambaStep::Recurrent) == (tokens == 1 && baseline > 0),
))]
pub(crate) const fn mamba_step(tokens: usize, baseline: usize) -> MambaStep {
    if tokens == 1 && baseline > 0 {
        MambaStep::Recurrent
    } else {
        MambaStep::Chunked
    }
}

/// How a transactional step records Mamba-2 state.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransactionalMamba {
    /// Every active row appends one token: its terminal state is its only snapshot.
    Terminal,
    /// Some row appends several tokens: every active row runs the recurrent
    /// update and records its state after each token.
    Snapshots,
}

/// Plans recurrent staging for a step inside a speculative transaction.
///
/// A rejected suffix cannot be removed from a recurrence, so a multi-token
/// proposal must keep the state after every token. Only the recurrent update
/// produces per-token states, and the reference uses it only for rows that
/// already have state; a row that would start its prompt inside a
/// transaction is therefore reported as `Err(row)`, naming the first such row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(policy =>
    ensures
        match policy {
            Ok(TransactionalMamba::Terminal) =>
                forall|r: int| 0 <= r < counts.len() ==> #[trigger] counts@[r] <= 1,
            Ok(TransactionalMamba::Snapshots) => {
                &&& exists|r: int| 0 <= r < counts.len() && #[trigger] counts@[r] > 1
                &&& forall|r: int| 0 <= r < counts.len() && #[trigger] counts@[r] > 0
                    ==> r < baselines.len() && baselines@[r] > 0
            },
            Err(row) => {
                &&& exists|r: int| 0 <= r < counts.len() && #[trigger] counts@[r] > 1
                &&& row < counts.len()
                &&& counts@[row as int] > 0
                &&& (row >= baselines.len() || baselines@[row as int] == 0)
                &&& forall|r: int| 0 <= r < row && #[trigger] counts@[r] > 0
                    ==> r < baselines.len() && baselines@[r] > 0
            },
        },
))]
pub(crate) fn transactional_mamba(
    counts: &[usize],
    baselines: &[usize],
) -> Result<TransactionalMamba, usize> {
    let mut row: usize = 0;
    let mut multi_token = false;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            multi_token == exists|r: int| 0 <= r < row && #[trigger] counts@[r] > 1,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] > 1 {
            multi_token = true;
        }
        row += 1;
    }
    if !multi_token {
        return Ok(TransactionalMamba::Terminal);
    }
    row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            exists|r: int| 0 <= r < counts.len() && #[trigger] counts@[r] > 1,
            forall|r: int| 0 <= r < row && #[trigger] counts@[r] > 0
                ==> r < baselines.len() && baselines@[r] > 0,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] > 0 && (row >= baselines.len() || baselines[row] == 0) {
            return Err(row);
        }
        row += 1;
    }
    Ok(TransactionalMamba::Snapshots)
}

/// Which compact cache a block reads and writes.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheKind {
    /// An attention block: one layer of the native KV cache.
    Attention,
    /// A Mamba-2 block: one layer of the recurrent and convolution cache.
    Recurrent,
    /// A dense or MoE block, which keeps no state.
    Stateless,
}

/// Compact cache slot of every block of a hybrid schedule.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verifier::allow(autoderive_clone_without_spec))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheSlots {
    /// Layer of block `i` inside its cache (zero for stateless blocks).
    pub(crate) slots: Vec<usize>,
    /// Attention blocks, the layer count of the KV cache.
    pub(crate) attention: usize,
    /// Mamba-2 blocks, the layer count of the recurrent cache.
    pub(crate) recurrent: usize,
}

/// Numbers the attention and Mamba-2 blocks of an interleaved schedule separately.
///
/// Attention and Mamba-2 blocks occupy separate compact cache index spaces:
/// block `i` of a stateful kind gets the number of earlier blocks of the
/// same kind. The proof establishes that each kind's slots are exactly
/// `0..count` in schedule order, so every block addresses a distinct
/// allocated layer of its cache and every allocated layer is used.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    ensures
        plan.slots.len() == kinds.len(),
        plan.attention == kind_count(kinds@, kinds.len() as int, CacheKind::Attention),
        plan.recurrent == kind_count(kinds@, kinds.len() as int, CacheKind::Recurrent),
        forall|i: int| 0 <= i < kinds.len() ==> #[trigger] plan.slots@[i] == (
            if kinds@[i] == CacheKind::Stateless {
                0
            } else {
                kind_count(kinds@, i, kinds@[i])
            }),
        forall|i: int| 0 <= i < kinds.len() && kinds@[i] == CacheKind::Attention
            ==> #[trigger] plan.slots@[i] < plan.attention,
        forall|i: int| 0 <= i < kinds.len() && kinds@[i] == CacheKind::Recurrent
            ==> #[trigger] plan.slots@[i] < plan.recurrent,
        forall|i: int, j: int| 0 <= i < j < kinds.len() && kinds@[i] == kinds@[j]
            && kinds@[i] != CacheKind::Stateless
            ==> #[trigger] plan.slots@[i] < #[trigger] plan.slots@[j],
        forall|s: int| 0 <= s < plan.attention
            ==> #[trigger] slot_used(kinds@, plan.slots@, CacheKind::Attention, s),
        forall|s: int| 0 <= s < plan.recurrent
            ==> #[trigger] slot_used(kinds@, plan.slots@, CacheKind::Recurrent, s),
))]
pub(crate) fn plan_cache_slots(kinds: &[CacheKind]) -> CacheSlots {
    let mut slots: Vec<usize> = Vec::with_capacity(kinds.len());
    let mut attention: usize = 0;
    let mut recurrent: usize = 0;
    let mut block: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            block <= kinds.len(),
            slots.len() == block,
            attention == kind_count(kinds@, block as int, CacheKind::Attention),
            recurrent == kind_count(kinds@, block as int, CacheKind::Recurrent),
            attention + recurrent <= block,
            forall|i: int| 0 <= i < block ==> #[trigger] slots@[i] == (
                if kinds@[i] == CacheKind::Stateless {
                    0
                } else {
                    kind_count(kinds@, i, kinds@[i])
                }),
        decreases kinds.len() - block,
    ))]
    while block < kinds.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_kind_count_step(kinds@, block as int, CacheKind::Attention);
            lemma_kind_count_step(kinds@, block as int, CacheKind::Recurrent);
        }
        match kinds[block] {
            CacheKind::Attention => {
                slots.push(attention);
                attention += 1;
            }
            CacheKind::Recurrent => {
                slots.push(recurrent);
                recurrent += 1;
            }
            CacheKind::Stateless => slots.push(0),
        }
        block += 1;
    }
    // Same-kind slots strictly increase, stay below the kind's count, and cover it.
    #[cfg(verus_keep_ghost)]
    proof! {
        let n = kinds.len() as int;
        assert forall|i: int| 0 <= i < n && kinds@[i] != CacheKind::Stateless
            implies #[trigger] slots@[i] < kind_count(kinds@, n, kinds@[i]) by {
            lemma_kind_count_step(kinds@, i, kinds@[i]);
            lemma_kind_count_monotone(kinds@, i + 1, n, kinds@[i]);
        }
        assert forall|i: int, j: int| 0 <= i < j < n && kinds@[i] == kinds@[j]
            && kinds@[i] != CacheKind::Stateless
            implies #[trigger] slots@[i] < #[trigger] slots@[j] by {
            lemma_kind_count_step(kinds@, i, kinds@[i]);
            lemma_kind_count_monotone(kinds@, i + 1, j, kinds@[i]);
        }
        assert forall|s: int| 0 <= s < attention
            implies #[trigger] slot_used(kinds@, slots@, CacheKind::Attention, s) by {
            lemma_kind_count_covers(kinds@, n, CacheKind::Attention, s);
            let i = choose|i: int| 0 <= i < n && kinds@[i] == CacheKind::Attention
                && #[trigger] kind_count(kinds@, i, CacheKind::Attention) == s;
            assert(slots@[i] == s);
        }
        assert forall|s: int| 0 <= s < recurrent
            implies #[trigger] slot_used(kinds@, slots@, CacheKind::Recurrent, s) by {
            lemma_kind_count_covers(kinds@, n, CacheKind::Recurrent, s);
            let i = choose|i: int| 0 <= i < n && kinds@[i] == CacheKind::Recurrent
                && #[trigger] kind_count(kinds@, i, CacheKind::Recurrent) == s;
            assert(slots@[i] == s);
        }
    }
    CacheSlots {
        slots,
        attention,
        recurrent,
    }
}

/// Largest row end `W_b + n_b` of a call, or zero without rows.
///
/// Rows are paired as `baselines.iter().zip(counts)` pairs them. Callers have
/// already proved every end representable; an unrepresentable end saturates
/// at `usize::MAX`, which the proof covers but no caller reaches.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(longest =>
    ensures
        ({
            let rows = if baselines.len() < counts.len() {
                baselines.len() as int
            } else {
                counts.len() as int
            };
            &&& forall|b: int| 0 <= b < rows
                ==> #[trigger] saturated_end(baselines@[b] as int, counts@[b] as int) <= longest
            &&& rows > 0 ==> exists|b: int| 0 <= b < rows
                && #[trigger] saturated_end(baselines@[b] as int, counts@[b] as int) == longest
            &&& rows == 0 ==> longest == 0
        }),
))]
pub(crate) fn longest_end(baselines: &[usize], counts: &[usize]) -> usize {
    let rows = if baselines.len() < counts.len() {
        baselines.len()
    } else {
        counts.len()
    };
    let mut longest: usize = 0;
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            rows <= baselines.len(),
            rows <= counts.len(),
            row <= rows,
            forall|b: int| 0 <= b < row
                ==> #[trigger] saturated_end(baselines@[b] as int, counts@[b] as int) <= longest,
            row > 0 ==> exists|b: int| 0 <= b < row
                && #[trigger] saturated_end(baselines@[b] as int, counts@[b] as int) == longest,
            row == 0 ==> longest == 0,
        decreases rows - row,
    ))]
    while row < rows {
        let end = baselines[row].saturating_add(counts[row]);
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(end == saturated_end(baselines@[row as int] as int, counts@[row as int] as int));
        }
        if row == 0 || end > longest {
            longest = end;
        }
        row += 1;
    }
    longest
}

/// The sliding window that can exclude a key when every history ends by `end`.
///
/// A window `w` excludes some key only when `w < end`; a wider window, a
/// zero window, and no window exclude nothing, so the result is `None`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(active =>
    ensures
        match active {
            Some(width) => window == Some(width) && 0 < width < end,
            None => match window {
                Some(width) => width == 0 || width >= end,
                None => true,
            },
        },
))]
pub(crate) const fn active_window(window: Option<usize>, end: usize) -> Option<usize> {
    match window {
        Some(width) if width > 0 && width < end => Some(width),
        _ => None,
    }
}

/// Whether every entry of `values` is equal.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(equal =>
    ensures
        equal == forall|i: int, j: int| 0 <= i < values.len() && 0 <= j < values.len()
            ==> #[trigger] values@[i] == #[trigger] values@[j],
))]
pub(crate) fn all_equal(values: &[usize]) -> bool {
    let mut index: usize = 1;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            1 <= index,
            values.len() > 0 ==> index <= values.len(),
            values.len() > 0 ==> forall|i: int| 0 <= i < index
                ==> #[trigger] values@[i] == values@[0],
        decreases values.len() - index,
    ))]
    while index < values.len() {
        if values[index] != values[index - 1] {
            return false;
        }
        index += 1;
    }
    true
}

/// Whether some row's appended end `L_b + n_b` exceeds `limit`.
///
/// Rows are paired as `lengths.iter().zip(counts)` pairs them. The sum is
/// compared without forming it, so the answer is exact even when it would
/// not fit `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(exceeds =>
    ensures
        ({
            let rows = if lengths.len() < counts.len() {
                lengths.len() as int
            } else {
                counts.len() as int
            };
            exceeds == exists|b: int| 0 <= b < rows && #[trigger] lengths@[b] + counts@[b] > limit
        }),
))]
pub(crate) fn exceeds_limit(lengths: &[usize], counts: &[usize], limit: usize) -> bool {
    let rows = if lengths.len() < counts.len() {
        lengths.len()
    } else {
        counts.len()
    };
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            rows <= lengths.len(),
            rows <= counts.len(),
            row <= rows,
            forall|b: int| 0 <= b < row ==> #[trigger] lengths@[b] + counts@[b] <= limit,
        decreases rows - row,
    ))]
    while row < rows {
        if lengths[row] > limit || counts[row] > limit - lengths[row] {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(lengths@[row as int] + counts@[row as int] > limit);
            }
            return true;
        }
        row += 1;
    }
    false
}

/// Shifted MTP cursors after truncating every target row to `targets[b]`.
///
/// Rows are paired as `targets.iter().zip(current)` pairs them; row `b`
/// becomes [`mtp_length_after_truncate`]`(targets[b], current[b])`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(lengths =>
    ensures
        ({
            let rows = if targets.len() < current.len() {
                targets.len() as int
            } else {
                current.len() as int
            };
            &&& lengths.len() == rows
            &&& forall|b: int| 0 <= b < rows ==> #[trigger] lengths@[b]
                == mtp_after_truncate(targets@[b] as int, current@[b] as int)
        }),
))]
pub(crate) fn mtp_lengths_after_truncate(targets: &[usize], current: &[usize]) -> Vec<usize> {
    let rows = if targets.len() < current.len() {
        targets.len()
    } else {
        current.len()
    };
    let mut lengths: Vec<usize> = Vec::with_capacity(rows);
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            rows <= targets.len(),
            rows <= current.len(),
            row <= rows,
            lengths.len() == row,
            forall|b: int| 0 <= b < row ==> #[trigger] lengths@[b]
                == mtp_after_truncate(targets@[b] as int, current@[b] as int),
        decreases rows - row,
    ))]
    while row < rows {
        lengths.push(mtp_length_after_truncate(targets[row], current[row]));
        row += 1;
    }
    lengths
}

/// Convolution channels `H d + 2 G N` of a Mamba-2 block, or `None` when unrepresentable.
///
/// The causal convolution runs over the concatenated `x` (`H d` channels for
/// `H` heads of width `d`) and the `B` and `C` projections (`G N` channels
/// each for `G` groups of state size `N`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(channels =>
    ensures
        match channels {
            Some(channels) => channels == heads * head_dim + 2 * (groups * state),
            None => heads * head_dim + 2 * (groups * state) > usize::MAX,
        },
))]
pub(crate) fn convolution_channels(
    heads: usize,
    head_dim: usize,
    groups: usize,
    state: usize,
) -> Option<usize> {
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(heads * head_dim >= 0) by (nonlinear_arith);
        assert(groups * state >= 0) by (nonlinear_arith);
    }
    let intermediate = heads.checked_mul(head_dim)?;
    let grouped = groups.checked_mul(state)?;
    let projections = grouped.checked_mul(2)?;
    intermediate.checked_add(projections)
}

/// Maps a target truncation to the shifted MTP cursor.
///
/// MTP position `i` consumes `(x_{i+1}, h_i)`, so it can never be ahead of
/// `target - 1`. The result only rewinds (`<= mtp`) and keeps
/// `result + 1 <= max(target, 1)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(length =>
    ensures
        length == mtp_after_truncate(target as int, mtp as int),
        length <= mtp,
        length + 1 <= if target > 0 { target as int } else { 1 },
        length == mtp || length + 1 == target || (target == 0 && length == 0),
))]
pub(crate) const fn mtp_length_after_truncate(target: usize, mtp: usize) -> usize {
    let limit = target.saturating_sub(1);
    if mtp < limit { mtp } else { limit }
}

/// Number of target tokens one verification pass must evaluate.
///
/// The pass consumes the lead token and the drafts `y_1..y_d`. When the last
/// draft is end-of-sequence its successor distribution is never needed: an
/// accepted EOS ends the cycle and a rejection stops earlier, so `y_d` itself
/// is not evaluated. Returns `None` on overflow.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(tokens =>
    ensures
        match tokens {
            Some(tokens) => {
                &&& last_draft_is_eos && drafted > 0 ==> tokens == drafted
                &&& !(last_draft_is_eos && drafted > 0) ==> tokens == drafted + 1
                &&& tokens >= 1
            },
            None => !(last_draft_is_eos && drafted > 0) && drafted == usize::MAX,
        },
))]
pub(crate) const fn verification_tokens(drafted: usize, last_draft_is_eos: bool) -> Option<usize> {
    if last_draft_is_eos && drafted > 0 {
        Some(drafted)
    } else {
        drafted.checked_add(1)
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
