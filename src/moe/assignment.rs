//! Dependency-free grouping of routed `(token, slot)` assignments by expert.
//!
//! A router emits one expert ID per flattened assignment `a = token * K + slot`.
//! Dispatch needs those assignments grouped by expert so each expert runs once
//! on a contiguous batch. [`plan_expert_assignments`] performs a stable
//! counting sort in `O(assignments + experts)`:
//!
//! 1. count assignments per expert;
//! 2. take exclusive prefix sums as `offsets` (`offsets[e]..offsets[e + 1]` is
//!    expert `e`'s range);
//! 3. scatter each assignment index to the next free slot of its expert.
//!
//! The resulting `order` is expert-major and, within an expert, increasing in
//! assignment index (token-major, slot-minor). Accumulating expert outputs in
//! this order reproduces the per-expert loop's floating-point summation order.
//!
//! # Verified planning
//!
//! The planner is checked in place by Verus through `verus/moe.rs`. For every
//! input, the proof establishes that no arithmetic overflows and no index is
//! out of bounds; that the first ID outside `0..experts` is reported with its
//! position; and that a successful plan has `experts + 1` nondecreasing
//! offsets from `0` to the assignment count, where every range holds exactly
//! as many entries as that expert has assignments, each entry names an
//! assignment routed to that expert, and entries increase strictly within a
//! range. The proof also gives the inverse explicitly: assignment `a` routed
//! to expert `e` sits at `offsets[e]` plus the number of earlier assignments
//! routed to `e`. So `order` is a bijection onto the assignments and each
//! range is exactly its expert's assignment set in ascending order.
//!
//! [`next_dispatch_block`] then groups consecutive experts into blocks whose
//! assignment count stays within a row budget, so dispatch can bound the
//! routed rows it materializes at once; its proof shows each block is a
//! non-empty run of whole experts within the budget (or one oversized
//! expert) and is maximal. [`dispatch_blocks`] iterates it and is proved to
//! partition all experts into such blocks, [`range_lengths`] yields the exact
//! per-expert counts, [`block_row_range`] locates each block's slice of the
//! order, and [`block_expert_rows`] locates each expert's rows inside its
//! block without passing the block's end. The planner's postcondition
//! includes the pairwise offset order these functions require, and
//! [`signed_order`] and [`signed_index`] convert the order and the slot count
//! to Burn's `i64` indices exactly.
//!
//! Before planning, [`routed_assignment_count`] bounds `tokens * K` by the
//! `i64` index range and [`routed_ids`] flattens the host copy of the
//! `[tokens, K]` IDs (plus an optional non-finite flag column); both are
//! proved exact, and the flattening reports a wrong length or the first
//! flagged token.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Number of assignments among the first `end` whose expert ID equals `expert`.
pub open spec fn expert_count(ids: Seq<i64>, expert: int, end: int) -> int
    decreases end,
{
    if end <= 0 {
        0
    } else {
        expert_count(ids, expert, end - 1) + if ids[end - 1] == expert { 1int } else { 0int }
    }
}

/// Position of assignment `a` in a plan: its expert's offset plus that expert's earlier assignments.
///
/// The planner proves `order[assignment_slot(offsets, ids, a)] == a` for every
/// assignment, so this is the inverse of the expert-major permutation.
pub open spec fn assignment_slot(offsets: Seq<usize>, ids: Seq<i64>, a: int) -> int {
    offsets[ids[a] as int] + expert_count(ids, ids[a] as int, a)
}

/// Sum of `expert_count(ids, e, end)` over experts `e < experts`.
pub open spec fn count_sum(ids: Seq<i64>, experts: int, end: int) -> int
    decreases experts,
{
    if experts <= 0 {
        0
    } else {
        count_sum(ids, experts - 1, end) + expert_count(ids, experts - 1, end)
    }
}

/// Counts are nonnegative, bounded by the prefix length, and monotone in the prefix.
proof fn lemma_expert_count_bounds(ids: Seq<i64>, expert: int, end: int)
    requires
        0 <= end,
    ensures
        0 <= expert_count(ids, expert, end) <= end,
    decreases end,
{
    if end > 0 {
        lemma_expert_count_bounds(ids, expert, end - 1);
    }
}

/// Extending a prefix never decreases an expert's count.
proof fn lemma_expert_count_monotone(ids: Seq<i64>, expert: int, shorter: int, longer: int)
    requires
        0 <= shorter <= longer,
    ensures
        expert_count(ids, expert, shorter) <= expert_count(ids, expert, longer),
    decreases longer - shorter,
{
    if shorter < longer {
        lemma_expert_count_monotone(ids, expert, shorter, longer - 1);
    }
}

/// Appending one in-range assignment adds exactly one to the total over all experts.
proof fn lemma_count_sum_step(ids: Seq<i64>, experts: int, end: int)
    requires
        0 <= end,
        0 <= experts,
    ensures
        count_sum(ids, experts, end + 1) == count_sum(ids, experts, end)
            + if 0 <= ids[end] < experts { 1int } else { 0int },
    decreases experts,
{
    if experts > 0 {
        lemma_count_sum_step(ids, experts - 1, end);
    }
}

/// When every ID is in range, the per-expert counts partition the assignments.
proof fn lemma_count_sum_total(ids: Seq<i64>, experts: int, end: int)
    requires
        0 <= end <= ids.len(),
        0 <= experts,
        forall|i: int| 0 <= i < end ==> 0 <= #[trigger] ids[i] < experts,
    ensures
        count_sum(ids, experts, end) == end,
    decreases end,
{
    if end > 0 {
        lemma_count_sum_total(ids, experts, end - 1);
        lemma_count_sum_step(ids, experts, end - 1);
    } else {
        lemma_count_sum_zero(ids, experts);
    }
}

/// An empty prefix contributes nothing to any expert.
proof fn lemma_count_sum_zero(ids: Seq<i64>, experts: int)
    requires
        0 <= experts,
    ensures
        count_sum(ids, experts, 0) == 0,
    decreases experts,
{
    if experts > 0 {
        lemma_count_sum_zero(ids, experts - 1);
    }
}

} // verus!

/// Expert-major order of routed assignments with per-expert ranges.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ExpertAssignmentPlan {
    /// Flattened assignment indices, grouped by expert and ascending within each group.
    pub(crate) order: Vec<usize>,
    /// `experts + 1` nondecreasing offsets; expert `e` owns `order[offsets[e]..offsets[e + 1]]`.
    pub(crate) offsets: Vec<usize>,
}

/// An expert ID outside `0..experts` at a flattened assignment position.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InvalidExpertId {
    /// Flattened `(token, slot)` assignment index.
    pub(crate) assignment: usize,
    /// Offending expert ID.
    pub(crate) expert: i64,
}

/// Groups flattened assignments by expert with a stable counting sort.
///
/// Besides the plan properties in the module documentation, the proof states
/// that any two offsets are ordered by index, which is exactly the
/// precondition of [`next_dispatch_block`], [`dispatch_blocks`],
/// [`block_row_range`], and [`block_expert_rows`].
///
/// # Errors
///
/// Returns the first assignment whose ID lies outside `0..experts`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Err(invalid) => {
                &&& invalid.assignment < ids.len()
                &&& invalid.expert == ids[invalid.assignment as int]
                &&& (invalid.expert < 0 || invalid.expert >= experts)
                &&& forall|i: int| 0 <= i < invalid.assignment ==> 0 <= #[trigger] ids[i] < experts
            },
            Ok(plan) => {
                &&& forall|i: int| 0 <= i < ids.len() ==> 0 <= #[trigger] ids[i] < experts
                &&& plan.offsets.len() == experts + 1
                &&& plan.offsets[0] == 0
                &&& plan.offsets[experts as int] == ids.len()
                &&& plan.order.len() == ids.len()
                &&& forall|low: int, high: int| 0 <= low <= high < plan.offsets.len() ==>
                    #[trigger] plan.offsets[low] <= #[trigger] plan.offsets[high]
                &&& forall|e: int| 0 <= e < experts ==> {
                    &&& plan.offsets[e] <= #[trigger] plan.offsets[e + 1]
                    &&& plan.offsets[e + 1] - plan.offsets[e]
                        == expert_count(ids@, e, ids.len() as int)
                }
                &&& forall|e: int, p: int|
                    #![trigger plan.offsets[e], plan.order[p]]
                    0 <= e < experts && plan.offsets[e] <= p < plan.offsets[e + 1] ==> {
                        &&& plan.order[p] < ids.len()
                        &&& ids[plan.order[p] as int] == e
                    }
                &&& forall|e: int, p: int|
                    #![trigger plan.offsets[e], plan.order[p]]
                    0 <= e < experts && plan.offsets[e] <= p && p + 1 < plan.offsets[e + 1] ==>
                        plan.order[p] < plan.order[p + 1]
                &&& forall|a: int| 0 <= a < ids.len() ==> {
                    &&& plan.offsets[ids[a] as int]
                        <= #[trigger] assignment_slot(plan.offsets@, ids@, a)
                    &&& assignment_slot(plan.offsets@, ids@, a) < plan.offsets[ids[a] as int + 1]
                    &&& plan.order[assignment_slot(plan.offsets@, ids@, a)] == a
                }
            },
        }
))]
pub(crate) fn plan_expert_assignments(
    ids: &[i64],
    experts: usize,
) -> Result<ExpertAssignmentPlan, InvalidExpertId> {
    // Pass 1: validate and count.
    let mut counts: Vec<usize> = Vec::with_capacity(experts);
    let mut expert = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            expert <= experts,
            counts.len() == expert,
            forall|e: int| 0 <= e < expert ==> #[trigger] counts[e] == 0,
        decreases experts - expert,
    ))]
    while expert < experts {
        counts.push(0);
        expert += 1;
    }
    let mut assignment = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            assignment <= ids.len(),
            counts.len() == experts,
            forall|i: int| 0 <= i < assignment ==> 0 <= #[trigger] ids[i] < experts,
            forall|e: int| 0 <= e < experts ==>
                #[trigger] counts[e] == expert_count(ids@, e, assignment as int),
        decreases ids.len() - assignment,
    ))]
    while assignment < ids.len() {
        let id = ids[assignment];
        if id < 0 || id as u64 >= experts as u64 {
            return Err(InvalidExpertId {
                assignment,
                expert: id,
            });
        }
        let id = id as usize;
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_expert_count_bounds(ids@, id as int, assignment as int);
        }
        let incremented = counts[id] + 1;
        set_entry(&mut counts, id, incremented);
        assignment += 1;
    }

    // Pass 2: exclusive prefix sums.
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_count_sum_total(ids@, experts as int, ids.len() as int);
    }
    let mut offsets: Vec<usize> = Vec::with_capacity(experts.saturating_add(1));
    offsets.push(0);
    let mut expert = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            expert <= experts,
            counts.len() == experts,
            offsets.len() == expert + 1,
            offsets[0] == 0,
            count_sum(ids@, experts as int, ids.len() as int) == ids.len(),
            offsets[expert as int] == count_sum(ids@, expert as int, ids.len() as int),
            forall|e: int| 0 <= e < experts ==>
                #[trigger] counts[e] == expert_count(ids@, e, ids.len() as int),
            forall|e: int| 0 <= e < expert ==>
                #[trigger] offsets[e + 1] == offsets[e] + counts[e],
            forall|e: int| 0 <= e <= expert ==>
                #[trigger] offsets[e] == count_sum(ids@, e, ids.len() as int),
            forall|e: int| 0 <= e <= expert ==> #[trigger] offsets[e] <= ids.len(),
        decreases experts - expert,
    ))]
    while expert < experts {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_sum_prefix_bounded(ids@, expert as int + 1, experts as int, ids.len() as int);
        }
        let next = offsets[expert] + counts[expert];
        offsets.push(next);
        expert += 1;
    }

    // Pass 3: stable scatter into each expert's next free slot.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|low: int, high: int| 0 <= low <= high <= experts implies
            #[trigger] offsets[low] <= #[trigger] offsets[high]
        by {
            lemma_count_sum_prefix_bounded(ids@, low, high, ids.len() as int);
        }
        assert forall|e: int| 0 <= e < experts implies
            #[trigger] offsets[e + 1] - offsets[e] == expert_count(ids@, e, ids.len() as int)
        by {
            assert(offsets[e + 1] == offsets[e] + counts[e]);
        }
        assert(plan_offsets(ids@, offsets@, experts as int));
    }
    let order = scatter_by_expert(ids, &offsets, experts);

    Ok(ExpertAssignmentPlan { order, offsets })
}

/// Scatters every assignment to the next free slot of its expert's range.
///
/// `offsets` are the exclusive prefix sums of the per-expert counts. The proof
/// shows that the result is the expert-major, ascending permutation described
/// by those offsets, and that assignment `a` lands exactly at
/// `assignment_slot(offsets, ids, a)`: its expert's offset plus the number of
/// earlier assignments routed to the same expert.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(order =>
    requires
        plan_offsets(ids@, offsets@, experts as int),
    ensures
        order.len() == ids.len(),
        forall|e: int, p: int|
            #![trigger offsets[e], order[p]]
            0 <= e < experts && offsets[e] <= p < offsets[e + 1] ==> {
                &&& order[p] < ids.len()
                &&& ids[order[p] as int] == e
            },
        forall|e: int, p: int|
            #![trigger offsets[e], order[p]]
            0 <= e < experts && offsets[e] <= p && p + 1 < offsets[e + 1] ==>
                order[p] < order[p + 1],
        forall|a: int| 0 <= a < ids.len() ==> {
            &&& offsets[ids[a] as int] <= #[trigger] assignment_slot(offsets@, ids@, a)
            &&& assignment_slot(offsets@, ids@, a) < offsets[ids[a] as int + 1]
            &&& order[assignment_slot(offsets@, ids@, a)] == a
        },
))]
fn scatter_by_expert(ids: &[i64], offsets: &[usize], experts: usize) -> Vec<usize> {
    // One cursor per offset: the next free slot of every expert, then the end.
    let mut cursors: Vec<usize> = Vec::with_capacity(experts + 1);
    let mut copied = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            copied <= offsets.len(),
            cursors.len() == copied,
            forall|e: int| 0 <= e < copied ==> #[trigger] cursors[e] == offsets[e],
        decreases offsets.len() - copied,
    ))]
    while copied < offsets.len() {
        cursors.push(offsets[copied]);
        copied += 1;
    }
    let mut order: Vec<usize> = Vec::with_capacity(ids.len());
    let mut fill = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            fill <= ids.len(),
            order.len() == fill,
        decreases ids.len() - fill,
    ))]
    while fill < ids.len() {
        order.push(0);
        fill += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_scatter_start(ids@, offsets@, experts as int, order@, cursors@);
    }
    let mut assignment = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            assignment <= ids.len(),
            order.len() == ids.len(),
            cursors.len() == experts + 1,
            plan_offsets(ids@, offsets@, experts as int),
            scatter_state(ids@, offsets@, experts as int, assignment as int, order@, cursors@),
        decreases ids.len() - assignment,
    ))]
    while assignment < ids.len() {
        let expert = ids[assignment] as usize;
        #[cfg(verus_keep_ghost)]
        proof! {
            // This assignment is the next one of `expert`, whose range still has room.
            lemma_scatter_room(ids@, offsets@, experts as int, assignment as int, order@, cursors@);
        }
        let slot = cursors[expert];
        #[cfg(verus_keep_ghost)]
        proof_decl! {
            let ghost previous_order = order@;
            let ghost previous_cursors = cursors@;
        }
        set_entry(&mut order, slot, assignment);
        set_entry(&mut cursors, expert, slot + 1);
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_scatter_step(
                ids@,
                offsets@,
                experts as int,
                assignment,
                previous_order,
                previous_cursors,
                order@,
                cursors@,
            );
        }
        assignment += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_scatter_done(ids@, offsets@, experts as int, order@, cursors@);
    }
    order
}

/// Returns the exclusive end of the next dispatch block of whole experts starting at `first`.
///
/// `offsets` are a plan's `experts + 1` nondecreasing range offsets. Experts
/// `first..end` own `offsets[end] - offsets[first]` assignments, the number of
/// routed rows that dispatch gathers and evaluates at once. That count is at
/// most `budget` unless the block is a single expert that alone exceeds it;
/// experts are never split, so each still runs on one contiguous batch. The
/// block is maximal: either no expert remains or adding expert `end` would
/// exceed the budget. Successive calls starting at the previous `end`
/// therefore partition `0..experts` into contiguous, non-empty blocks.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(end =>
    requires
        first + 1 < offsets.len(),
        forall|low: int, high: int| 0 <= low <= high < offsets.len() ==>
            #[trigger] offsets[low] <= #[trigger] offsets[high],
    ensures
        first < end,
        end + 1 <= offsets.len(),
        end == first + 1 || offsets[end as int] - offsets[first as int] <= budget,
        end + 1 == offsets.len() || offsets[end + 1] - offsets[first as int] > budget,
))]
pub(crate) fn next_dispatch_block(offsets: &[usize], first: usize, budget: usize) -> usize {
    let last = offsets.len() - 1;
    let mut end = first + 1;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            first < end <= last,
            last + 1 == offsets.len(),
            forall|low: int, high: int| 0 <= low <= high < offsets.len() ==>
                #[trigger] offsets[low] <= #[trigger] offsets[high],
            end == first + 1 || offsets[end as int] - offsets[first as int] <= budget,
        decreases last - end,
    ))]
    // Offsets are nondecreasing, so `offsets[end + 1] - offsets[first]` cannot underflow.
    while end < last && offsets[end + 1] - offsets[first] <= budget {
        end += 1;
    }
    end
}

/// Splits a plan's experts into consecutive dispatch blocks and returns their boundaries.
///
/// Block `i` holds experts `bounds[i]..bounds[i + 1]`. The proof shows that the
/// boundaries start at zero, end at the expert count, and strictly increase, so
/// the blocks partition `0..experts` into non-empty runs of whole experts; and
/// that every block is exactly the one [`next_dispatch_block`] chooses: within
/// the row budget unless it is a single expert, and maximal. With no experts
/// there are no blocks.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bounds =>
    requires
        offsets.len() >= 1,
        forall|low: int, high: int| 0 <= low <= high < offsets.len() ==>
            #[trigger] offsets[low] <= #[trigger] offsets[high],
    ensures
        bounds.len() >= 1,
        bounds[0] == 0,
        bounds[bounds.len() - 1] + 1 == offsets.len(),
        forall|i: int| 0 <= i < bounds.len() - 1 ==> {
            &&& #[trigger] bounds[i] < bounds[i + 1]
            &&& bounds[i + 1] == bounds[i] + 1
                || offsets[bounds[i + 1] as int] - offsets[bounds[i] as int] <= budget
            &&& bounds[i + 1] + 1 == offsets.len()
                || offsets[bounds[i + 1] + 1] - offsets[bounds[i] as int] > budget
        },
))]
pub(crate) fn dispatch_blocks(offsets: &[usize], budget: usize) -> Vec<usize> {
    let experts = offsets.len() - 1;
    let mut bounds: Vec<usize> = Vec::new();
    bounds.push(0);
    let mut first = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            first <= experts,
            experts + 1 == offsets.len(),
            forall|low: int, high: int| 0 <= low <= high < offsets.len() ==>
                #[trigger] offsets[low] <= #[trigger] offsets[high],
            bounds.len() >= 1,
            bounds[0] == 0,
            bounds[bounds.len() - 1] == first,
            forall|i: int| 0 <= i < bounds.len() - 1 ==> {
                &&& #[trigger] bounds[i] < bounds[i + 1]
                &&& bounds[i + 1] == bounds[i] + 1
                    || offsets[bounds[i + 1] as int] - offsets[bounds[i] as int] <= budget
                &&& bounds[i + 1] + 1 == offsets.len()
                    || offsets[bounds[i + 1] + 1] - offsets[bounds[i] as int] > budget
            },
        decreases experts - first,
    ))]
    while first < experts {
        let end = next_dispatch_block(offsets, first, budget);
        bounds.push(end);
        first = end;
    }
    bounds
}

/// Returns the length `offsets[e + 1] - offsets[e]` of every consecutive range.
///
/// For a plan these are the per-expert assignment counts; the proof shows each
/// length is exact and that there is one per range.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(lengths =>
    requires
        forall|e: int| 0 <= e && e + 1 < offsets.len() ==>
            #[trigger] offsets[e] <= offsets[e + 1],
    ensures
        offsets.len() == 0 ==> lengths.len() == 0,
        offsets.len() > 0 ==> lengths.len() + 1 == offsets.len(),
        forall|e: int| 0 <= e < lengths.len() ==>
            #[trigger] lengths[e] == offsets[e + 1] - offsets[e],
))]
pub(crate) fn range_lengths(offsets: &[usize]) -> Vec<usize> {
    let mut lengths: Vec<usize> = Vec::new();
    let mut range = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            offsets.len() == 0 ==> range == 0,
            offsets.len() > 0 ==> range < offsets.len(),
            lengths.len() == range,
            forall|e: int| 0 <= e && e + 1 < offsets.len() ==>
                #[trigger] offsets[e] <= offsets[e + 1],
            forall|e: int| 0 <= e < range ==>
                #[trigger] lengths[e] == offsets[e + 1] - offsets[e],
        decreases offsets.len() - range,
    ))]
    while range + 1 < offsets.len() {
        lengths.push(offsets[range + 1] - offsets[range]);
        range += 1;
    }
    lengths
}

/// Locates the routed rows of the dispatch block of experts `first..end` in the expert-major order.
///
/// Returns `(start, rows)`: the block's assignments are
/// `order[start..start + rows]`. The proof shows `start` is expert `first`'s
/// offset, `rows` is the block's exact assignment count, and the slice ends
/// at expert `end`'s offset, never past the last offset (the assignment
/// count), so narrowing the uploaded order to it stays in bounds.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(range =>
    requires
        first <= end,
        end < offsets.len(),
        forall|low: int, high: int| 0 <= low <= high < offsets.len() ==>
            #[trigger] offsets[low] <= #[trigger] offsets[high],
    ensures
        range.0 == offsets[first as int],
        range.1 == offsets[end as int] - offsets[first as int],
        range.0 + range.1 == offsets[end as int],
        range.0 + range.1 <= offsets[offsets.len() - 1],
))]
pub(crate) fn block_row_range(offsets: &[usize], first: usize, end: usize) -> (usize, usize) {
    (offsets[first], offsets[end] - offsets[first])
}

/// Converts the expert-major order to the `i64` values Burn index tensors use.
///
/// The proof shows every index is preserved exactly, or that `None` is
/// returned exactly because some index exceeds `i64::MAX`. A plan built from
/// [`routed_assignment_count`] assignments never takes the `None` branch,
/// since every entry is an assignment index below that count.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(signed =>
    ensures
        match signed {
            Some(values) => {
                &&& values.len() == order.len()
                &&& forall|k: int| 0 <= k < order.len() ==> #[trigger] values[k] == order[k]
            },
            None => exists|k: int| 0 <= k < order.len() && #[trigger] order[k] > i64::MAX,
        },
))]
pub(crate) fn signed_order(order: &[usize]) -> Option<Vec<i64>> {
    let mut values: Vec<i64> = Vec::with_capacity(order.len());
    let mut index = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= order.len(),
            values.len() == index,
            forall|k: int| 0 <= k < index ==> #[trigger] values[k] == order[k],
        decreases order.len() - index,
    ))]
    while index < order.len() {
        let entry = order[index];
        if entry as u64 > i64::MAX as u64 {
            return None;
        }
        values.push(entry as i64);
        index += 1;
    }
    Some(values)
}

/// Converts one index to `i64`, or `None` exactly when it exceeds `i64::MAX`.
///
/// Dispatch divides the uploaded order by the slot count `K` on the device;
/// `K <= tokens * K <= i64::MAX` for every accepted plan, so this is exact.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(signed =>
    ensures
        match signed {
            Some(value) => value == index,
            None => index > i64::MAX,
        },
))]
pub(crate) fn signed_index(index: usize) -> Option<i64> {
    if index as u64 > i64::MAX as u64 {
        None
    } else {
        Some(index as i64)
    }
}

/// Locates expert `expert`'s rows inside a dispatch block that starts at expert `first`.
///
/// Returns `(local, count)`: the expert's rows are block rows
/// `local..local + count`. The proof shows `local` is the expert's offset
/// relative to the block start, `count` is its assignment count, and the
/// slice ends within every block that contains the expert, so narrowing a
/// gathered block to it never reads past the block.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rows =>
    requires
        first <= expert,
        expert + 1 < offsets.len(),
        forall|low: int, high: int| 0 <= low <= high < offsets.len() ==>
            #[trigger] offsets[low] <= #[trigger] offsets[high],
    ensures
        rows.0 == offsets[expert as int] - offsets[first as int],
        rows.1 == offsets[expert + 1] - offsets[expert as int],
        forall|end: int| expert < end < offsets.len() ==>
            rows.0 + rows.1 <= #[trigger] offsets[end] - offsets[first as int],
))]
pub(crate) fn block_expert_rows(offsets: &[usize], first: usize, expert: usize) -> (usize, usize) {
    (
        offsets[expert] - offsets[first],
        offsets[expert + 1] - offsets[expert],
    )
}

/// Number of routed `(token, slot)` assignments, if Burn's `i64` indices can address them.
///
/// The proof shows the result is exactly `tokens * slots` whenever that
/// product is at most `i64::MAX`, and `None` otherwise.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(count =>
    ensures
        match count {
            Some(assignments) => {
                &&& assignments == tokens * slots
                &&& assignments <= i64::MAX
            },
            None => tokens * slots > i64::MAX || tokens * slots > usize::MAX,
        },
))]
pub(crate) fn routed_assignment_count(tokens: usize, slots: usize) -> Option<usize> {
    match tokens.checked_mul(slots) {
        Some(count) if count as u64 <= i64::MAX as u64 => Some(count),
        _ => None,
    }
}

/// Why a host copy of routed expert IDs cannot be flattened.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RoutedIdsError {
    /// The copy holds `found` values instead of `tokens * columns`.
    Length {
        /// Number of values read from the device.
        found: usize,
    },
    /// The non-finite flag column marks this flattened token.
    NonFinite {
        /// First flagged token.
        token: usize,
    },
}

/// Flattens a row-major `[tokens, slots (+ flag)]` host copy into `tokens * slots` expert IDs.
///
/// With `flagged`, every row carries one extra trailing column whose non-zero
/// value marks a token with non-finite router scores. The proof shows that a
/// copy of the wrong length is rejected with its length; that otherwise the
/// first flagged token is reported, all earlier ones being unflagged; and
/// that a success holds exactly the ID columns in token-major, slot-minor
/// order, `ids[k] == values[(k / slots) * columns + k % slots]`, with every
/// flag zero.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(ids =>
    requires
        0 < slots < usize::MAX,
    ensures
        match ids {
            Ok(ids) => {
                &&& values.len() == tokens * routed_columns(slots, flagged)
                &&& ids.len() == tokens * slots
                &&& forall|k: int| 0 <= k < ids.len() ==> #[trigger] ids[k]
                    == routed_entry(values@, routed_columns(slots, flagged), slots as int, k)
                &&& flagged ==> forall|t: int| 0 <= t < tokens ==>
                    #[trigger] routed_flag(values@, slots as int + 1, slots as int, t) == 0
            },
            Err(RoutedIdsError::Length { found }) => {
                &&& found == values.len()
                &&& found != tokens * routed_columns(slots, flagged)
            },
            Err(RoutedIdsError::NonFinite { token }) => {
                &&& flagged
                &&& values.len() == tokens * (slots + 1)
                &&& token < tokens
                &&& routed_flag(values@, slots as int + 1, slots as int, token as int) != 0
                &&& forall|t: int| 0 <= t < token ==>
                    #[trigger] routed_flag(values@, slots as int + 1, slots as int, t) == 0
            },
        },
))]
pub(crate) fn routed_ids(
    values: &[i64],
    tokens: usize,
    slots: usize,
    flagged: bool,
) -> Result<Vec<i64>, RoutedIdsError> {
    let columns = if flagged { slots + 1 } else { slots };
    // Equivalent to `values.len() != tokens * columns` without forming a product that could overflow.
    if !values.len().is_multiple_of(columns) || values.len() / columns != tokens {
        #[cfg(verus_keep_ghost)]
        proof! {
            if values.len() == tokens * columns {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
                    values.len() as int,
                    columns as int,
                    tokens as int,
                    0,
                );
            }
        }
        return Err(RoutedIdsError::Length {
            found: values.len(),
        });
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(values.len() as int, columns as int);
        assert(values.len() == tokens * columns) by (nonlinear_arith)
            requires
                values.len() == columns * (values.len() / columns) + values.len() % columns,
                values.len() % columns == 0,
                values.len() / columns == tokens;
        assert(tokens * slots <= tokens * columns) by (nonlinear_arith)
            requires slots <= columns;
    }
    let mut ids: Vec<i64> = Vec::with_capacity(tokens * slots);
    let mut token = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            token <= tokens,
            0 < slots <= columns,
            columns == routed_columns(slots, flagged),
            values.len() == tokens * columns,
            tokens * slots <= values.len(),
            ids.len() == token * slots,
            forall|k: int| 0 <= k < ids.len() ==>
                #[trigger] ids[k] == routed_entry(values@, columns as int, slots as int, k),
            flagged ==> forall|t: int| 0 <= t < token ==>
                #[trigger] routed_flag(values@, slots as int + 1, slots as int, t) == 0,
        decreases tokens - token,
    ))]
    while token < tokens {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_row_in_bounds(token as int, tokens as int, columns as int, slots as int);
        }
        let row = token * columns;
        if flagged && values[row + slots] != 0 {
            return Err(RoutedIdsError::NonFinite { token });
        }
        let mut slot = 0usize;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                token < tokens,
                slot <= slots,
                0 < slots <= columns,
                columns == routed_columns(slots, flagged),
                values.len() == tokens * columns,
                row == token * columns,
                row + columns <= values.len(),
                ids.len() == token * slots + slot,
                forall|k: int| 0 <= k < ids.len() ==>
                    #[trigger] ids[k] == routed_entry(values@, columns as int, slots as int, k),
            decreases slots - slot,
        ))]
        while slot < slots {
            #[cfg(verus_keep_ghost)]
            proof! {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
                    (token * slots + slot) as int,
                    slots as int,
                    token as int,
                    slot as int,
                );
            }
            ids.push(values[row + slot]);
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((token + 1) * slots == token * slots + slots) by (nonlinear_arith);
        }
        token += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(ids.len() == tokens * slots);
    }
    Ok(ids)
}

/// Overwrites one in-bounds entry; isolating the write keeps the planner's proof small.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(
    requires
        index < old(values).len(),
    ensures
        final(values)@ == old(values)@.update(index as int, value),
))]
fn set_entry(values: &mut [usize], index: usize, value: usize) {
    values[index] = value;
}

#[cfg(verus_keep_ghost)]
verus! {

/// Facts about a plan's offsets that the scatter pass relies on.
///
/// Every ID names an expert, and `offsets` are nondecreasing exclusive prefix
/// sums of the per-expert counts, from `0` to the number of assignments.
pub open spec fn plan_offsets(ids: Seq<i64>, offsets: Seq<usize>, experts: int) -> bool {
    &&& 0 <= experts
    &&& offsets.len() == experts + 1
    &&& offsets[0] == 0
    &&& offsets[experts] == ids.len()
    &&& forall|i: int| 0 <= i < ids.len() ==> 0 <= #[trigger] ids[i] < experts
    &&& forall|low: int, high: int| 0 <= low <= high <= experts ==>
        #[trigger] offsets[low] <= #[trigger] offsets[high]
    &&& forall|e: int| 0 <= e < experts ==>
        #[trigger] offsets[e + 1] - offsets[e] == expert_count(ids, e, ids.len() as int)
}

/// Scatter state after the first `assignment` assignments.
///
/// Each cursor sits after the expert's scattered assignments; every filled
/// slot holds an earlier assignment of its range's expert, in ascending order;
/// and every scattered assignment sits at its `assignment_slot`. Opaque, so
/// the scatter loop reasons about it only through the lemmas below.
#[verifier::opaque]
pub open spec fn scatter_state(
    ids: Seq<i64>,
    offsets: Seq<usize>,
    experts: int,
    assignment: int,
    order: Seq<usize>,
    cursors: Seq<usize>,
) -> bool {
    &&& 0 <= assignment <= ids.len()
    &&& order.len() == ids.len()
    &&& cursors.len() == experts + 1
    &&& forall|e: int| 0 <= e < experts ==>
        #[trigger] cursors[e] == offsets[e] + expert_count(ids, e, assignment)
    &&& forall|e: int, p: int|
        #![trigger offsets[e], order[p]]
        0 <= e < experts && offsets[e] <= p < cursors[e] ==> {
            &&& order[p] < assignment
            &&& ids[order[p] as int] == e
        }
    &&& forall|e: int, p: int|
        #![trigger offsets[e], order[p]]
        0 <= e < experts && offsets[e] <= p && p + 1 < cursors[e] ==>
            order[p] < order[p + 1]
    &&& forall|a: int| 0 <= a < assignment ==> {
        &&& offsets[ids[a] as int] <= #[trigger] assignment_slot(offsets, ids, a)
        &&& assignment_slot(offsets, ids, a) < cursors[ids[a] as int]
        &&& order[assignment_slot(offsets, ids, a)] == a
    }
}

/// A cursor never passes the end of its expert's range.
proof fn lemma_cursor_within_range(ids: Seq<i64>, offsets: Seq<usize>, experts: int, assignment: int, e: int)
    requires
        plan_offsets(ids, offsets, experts),
        0 <= assignment <= ids.len(),
        0 <= e < experts,
    ensures
        offsets[e] <= offsets[e] + expert_count(ids, e, assignment) <= offsets[e + 1],
{
    lemma_expert_count_bounds(ids, e, assignment);
    lemma_expert_count_monotone(ids, e, assignment, ids.len() as int);
    assert(offsets[e + 1] - offsets[e] == expert_count(ids, e, ids.len() as int));
}

/// Before any scatter, cursors are the offsets and no slot is filled.
proof fn lemma_scatter_start(
    ids: Seq<i64>,
    offsets: Seq<usize>,
    experts: int,
    order: Seq<usize>,
    cursors: Seq<usize>,
)
    requires
        plan_offsets(ids, offsets, experts),
        order.len() == ids.len(),
        cursors.len() == offsets.len(),
        forall|e: int| 0 <= e < cursors.len() ==> #[trigger] cursors[e] == offsets[e],
    ensures
        scatter_state(ids, offsets, experts, 0, order, cursors),
{
    reveal(scatter_state);
}

/// The next assignment's expert still has a free slot at its cursor.
proof fn lemma_scatter_room(
    ids: Seq<i64>,
    offsets: Seq<usize>,
    experts: int,
    assignment: int,
    order: Seq<usize>,
    cursors: Seq<usize>,
)
    requires
        plan_offsets(ids, offsets, experts),
        scatter_state(ids, offsets, experts, assignment, order, cursors),
        0 <= assignment < ids.len(),
    ensures
        0 <= ids[assignment] < experts,
        cursors[ids[assignment] as int] == assignment_slot(offsets, ids, assignment),
        offsets[ids[assignment] as int] <= cursors[ids[assignment] as int],
        cursors[ids[assignment] as int] < offsets[ids[assignment] as int + 1],
        offsets[ids[assignment] as int + 1] <= ids.len(),
{
    reveal(scatter_state);
    let e = ids[assignment] as int;
    lemma_expert_count_bounds(ids, e, assignment);
    lemma_expert_count_monotone(ids, e, assignment + 1, ids.len() as int);
    assert(offsets[e + 1] - offsets[e] == expert_count(ids, e, ids.len() as int));
    assert(offsets[e + 1] <= offsets[experts]);
}

/// Scattering assignment `assignment` to its cursor and advancing that cursor preserves the state.
///
/// Expert ranges are disjoint and earlier slots of the same expert lie below
/// the cursor, so the write overwrites no earlier slot, lands in the expert's
/// own range above all of that expert's earlier assignments, and sits at the
/// new assignment's `assignment_slot`.
proof fn lemma_scatter_step(
    ids: Seq<i64>,
    offsets: Seq<usize>,
    experts: int,
    assignment: usize,
    previous_order: Seq<usize>,
    previous_cursors: Seq<usize>,
    order: Seq<usize>,
    cursors: Seq<usize>,
)
    requires
        plan_offsets(ids, offsets, experts),
        scatter_state(ids, offsets, experts, assignment as int, previous_order, previous_cursors),
        assignment < ids.len(),
        previous_cursors[ids[assignment as int] as int] < offsets[ids[assignment as int] as int + 1],
        order == previous_order.update(
            previous_cursors[ids[assignment as int] as int] as int,
            assignment,
        ),
        cursors.len() == previous_cursors.len(),
        cursors[ids[assignment as int] as int] == previous_cursors[ids[assignment as int] as int] + 1,
        forall|e: int| 0 <= e < cursors.len() && e != ids[assignment as int] as int ==>
            #[trigger] cursors[e] == previous_cursors[e],
    ensures
        scatter_state(ids, offsets, experts, assignment + 1, order, cursors),
{
    reveal(scatter_state);
    let current = assignment as int;
    let expert = ids[current] as int;
    let slot = previous_cursors[expert] as int;
    lemma_expert_count_bounds(ids, expert, current);
    assert(offsets[expert + 1] <= offsets[experts]);
    assert(slot == assignment_slot(offsets, ids, current));
    assert forall|e: int| 0 <= e < experts implies
        #[trigger] cursors[e] == offsets[e] + expert_count(ids, e, current + 1)
    by {
        assert(expert_count(ids, e, current + 1)
            == expert_count(ids, e, current) + if ids[current] == e { 1int } else { 0int });
    }
    // Ranges are disjoint, so the new entry lands only in `expert`'s range.
    assert forall|e: int, p: int|
        #![trigger offsets[e], order[p]]
        0 <= e < experts && offsets[e] <= p < cursors[e] implies
            order[p] < current + 1 && ids[order[p] as int] == e
    by {
        lemma_cursor_within_range(ids, offsets, experts, current, e);
        if p != slot {
            assert(p < previous_cursors[e]);
            assert(order[p] == previous_order[p]);
        } else if e < expert {
            assert(offsets[e + 1] <= offsets[expert]);
        } else if e > expert {
            assert(offsets[expert + 1] <= offsets[e]);
        }
    }
    assert forall|e: int, p: int|
        #![trigger offsets[e], order[p]]
        0 <= e < experts && offsets[e] <= p && p + 1 < cursors[e] implies
            order[p] < order[p + 1]
    by {
        lemma_cursor_within_range(ids, offsets, experts, current, e);
        if p + 1 == slot && e == expert {
            assert(order[p] == previous_order[p]);
        } else if p + 1 < slot || p > slot || e != expert {
            if e < expert {
                assert(offsets[e + 1] <= offsets[expert]);
            } else if e > expert {
                assert(offsets[expert + 1] <= offsets[e]);
            }
            assert(order[p] == previous_order[p]);
            assert(order[p + 1] == previous_order[p + 1]);
        }
    }
    // Each earlier assignment keeps its slot, and this one lands at its own.
    assert forall|a: int| 0 <= a < current + 1 implies {
        &&& offsets[ids[a] as int] <= #[trigger] assignment_slot(offsets, ids, a)
        &&& assignment_slot(offsets, ids, a) < cursors[ids[a] as int]
        &&& order[assignment_slot(offsets, ids, a)] == a
    } by {
        if a < current {
            let owner = ids[a] as int;
            let position = assignment_slot(offsets, ids, a);
            lemma_cursor_within_range(ids, offsets, experts, current, owner);
            assert(position < previous_cursors[owner]);
            if owner < expert {
                assert(offsets[owner + 1] <= offsets[expert]);
            } else if owner > expert {
                assert(offsets[expert + 1] <= offsets[owner]);
            }
            assert(position != slot);
            assert(offsets[owner + 1] <= offsets[experts]);
        }
    }
}

/// After every assignment is scattered, each cursor is at its range's end.
proof fn lemma_scatter_done(
    ids: Seq<i64>,
    offsets: Seq<usize>,
    experts: int,
    order: Seq<usize>,
    cursors: Seq<usize>,
)
    requires
        plan_offsets(ids, offsets, experts),
        scatter_state(ids, offsets, experts, ids.len() as int, order, cursors),
    ensures
        order.len() == ids.len(),
        forall|e: int, p: int|
            #![trigger offsets[e], order[p]]
            0 <= e < experts && offsets[e] <= p < offsets[e + 1] ==> {
                &&& order[p] < ids.len()
                &&& ids[order[p] as int] == e
            },
        forall|e: int, p: int|
            #![trigger offsets[e], order[p]]
            0 <= e < experts && offsets[e] <= p && p + 1 < offsets[e + 1] ==>
                order[p] < order[p + 1],
        forall|a: int| 0 <= a < ids.len() ==> {
            &&& offsets[ids[a] as int] <= #[trigger] assignment_slot(offsets, ids, a)
            &&& assignment_slot(offsets, ids, a) < offsets[ids[a] as int + 1]
            &&& order[assignment_slot(offsets, ids, a)] == a
        },
{
    reveal(scatter_state);
    assert forall|e: int| 0 <= e < experts implies #[trigger] cursors[e] == offsets[e + 1] by {
        assert(offsets[e + 1] - offsets[e] == expert_count(ids, e, ids.len() as int));
    }
}

/// Number of host columns per token: the expert IDs plus an optional flag column.
pub open spec fn routed_columns(slots: usize, flagged: bool) -> int {
    if flagged { slots + 1 } else { slots as int }
}

/// Non-finite flag of `token` in a row-major copy with `columns` columns.
pub open spec fn routed_flag(values: Seq<i64>, columns: int, slots: int, token: int) -> i64 {
    values[token * columns + slots]
}

/// Expert ID that flattened assignment `k` reads from a row-major copy.
pub open spec fn routed_entry(values: Seq<i64>, columns: int, slots: int, k: int) -> i64 {
    values[(k / slots) * columns + k % slots]
}

/// Row `token < tokens` of a `[tokens, columns]` copy lies inside it and leaves room for its flag.
proof fn lemma_row_in_bounds(token: int, tokens: int, columns: int, slots: int)
    requires
        0 <= token < tokens,
        0 < slots <= columns,
    ensures
        token * columns + columns <= tokens * columns,
        token * columns >= 0,
{
    assert(token * columns + columns <= tokens * columns) by (nonlinear_arith)
        requires token < tokens, columns > 0;
    assert(token * columns >= 0) by (nonlinear_arith)
        requires token >= 0, columns > 0;
}

/// Partial sums of per-expert counts never exceed the full sum.
proof fn lemma_count_sum_prefix_bounded(ids: Seq<i64>, prefix: int, experts: int, end: int)
    requires
        0 <= prefix <= experts,
        0 <= end,
    ensures
        count_sum(ids, prefix, end) <= count_sum(ids, experts, end),
    decreases experts - prefix,
{
    if prefix < experts {
        lemma_count_sum_prefix_bounded(ids, prefix + 1, experts, end);
        lemma_expert_count_bounds(ids, prefix, end);
    }
}

} // verus!

#[cfg(test)]
#[path = "assignment_tests.rs"]
mod tests;
