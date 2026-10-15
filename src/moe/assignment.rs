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
//! out of bounds; that an ID outside `0..experts` is reported with its
//! position; and that a successful plan has `experts + 1` nondecreasing
//! offsets from `0` to the assignment count, where every range holds exactly
//! as many entries as that expert has assignments, each entry names an
//! assignment routed to that expert, and entries increase strictly within a
//! range. Together these imply that each range is exactly the expert's
//! assignment set in ascending order.

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
            },
            Ok(plan) => {
                &&& forall|i: int| 0 <= i < ids.len() ==> 0 <= #[trigger] ids[i] < experts
                &&& plan.offsets.len() == experts + 1
                &&& plan.offsets[0] == 0
                &&& plan.offsets[experts as int] == ids.len()
                &&& plan.order.len() == ids.len()
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
    }
    let mut cursors = offsets.clone();
    let mut order: Vec<usize> = Vec::with_capacity(ids.len());
    let mut fill = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            fill <= ids.len(),
            order.len() == fill,
            forall|p: int| 0 <= p < fill ==> #[trigger] order[p] == 0,
        decreases ids.len() - fill,
    ))]
    while fill < ids.len() {
        order.push(0);
        fill += 1;
    }
    let mut assignment = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            assignment <= ids.len(),
            order.len() == ids.len(),
            offsets.len() == experts + 1,
            cursors.len() == experts + 1,
            offsets[0] == 0,
            offsets[experts as int] == ids.len(),
            forall|e: int| 0 <= e <= experts ==> #[trigger] offsets[e] <= ids.len(),
            forall|low: int, high: int| 0 <= low <= high <= experts ==>
                #[trigger] offsets[low] <= #[trigger] offsets[high],
            forall|e: int| 0 <= e < experts ==> #[trigger] cursors[e] <= offsets[e + 1],
            forall|i: int| 0 <= i < ids.len() ==> 0 <= #[trigger] ids[i] < experts,
            forall|e: int| 0 <= e < experts ==> {
                &&& #[trigger] offsets[e + 1] - offsets[e]
                    == expert_count(ids@, e, ids.len() as int)
                &&& cursors[e] == offsets[e] + expert_count(ids@, e, assignment as int)
            },
            forall|e: int, p: int|
                #![trigger offsets[e], order[p]]
                0 <= e < experts && offsets[e] <= p < cursors[e] ==> {
                    &&& order[p] < assignment
                    &&& ids[order[p] as int] == e
                },
            forall|e: int, p: int|
                #![trigger offsets[e], order[p]]
                0 <= e < experts && offsets[e] <= p && p + 1 < cursors[e] ==>
                    order[p] < order[p + 1],
        decreases ids.len() - assignment,
    ))]
    while assignment < ids.len() {
        let expert = ids[assignment] as usize;
        #[cfg(verus_keep_ghost)]
        proof! {
            // This assignment is the next one of `expert`, whose range still has room.
            assert(expert as int == ids[assignment as int]);
            lemma_expert_count_monotone(
                ids@,
                expert as int,
                assignment as int + 1,
                ids.len() as int,
            );
            assert(expert_count(ids@, expert as int, assignment as int + 1)
                == expert_count(ids@, expert as int, assignment as int) + 1);
            assert(offsets[expert as int + 1] - offsets[expert as int]
                == expert_count(ids@, expert as int, ids.len() as int));
            assert(offsets[expert as int + 1] <= ids.len());
            lemma_expert_count_bounds(ids@, expert as int, assignment as int);
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
            // Ranges are disjoint, so the new entry lands only in `expert`'s range.
            assert forall|e: int, p: int|
                #![trigger offsets[e], order[p]]
                0 <= e < experts && offsets[e] <= p < cursors[e] implies
                    order[p] < assignment + 1 && ids[order[p] as int] == e
            by {
                if p != slot as int {
                    assert(p < previous_cursors[e]);
                    assert(order[p] == previous_order[p]);
                } else if e < expert as int {
                    // slot >= offsets[expert] >= offsets[e + 1] >= cursors[e] > p.
                    assert(previous_cursors[e] <= offsets[e + 1]);
                    assert(offsets[e + 1] <= offsets[expert as int]);
                    assert(slot as int >= offsets[expert as int]);
                } else if e > expert as int {
                    // slot < offsets[expert + 1] <= offsets[e] <= p.
                    assert(offsets[expert as int + 1] <= offsets[e]);
                    assert(slot as int + 1 <= offsets[expert as int + 1]);
                }
            }
            assert forall|e: int, p: int|
                #![trigger offsets[e], order[p]]
                0 <= e < experts && offsets[e] <= p && p + 1 < cursors[e] implies
                    order[p] < order[p + 1]
            by {
                if p + 1 == slot as int && e == expert as int {
                    assert(order[p] == previous_order[p]);
                } else if p + 1 < slot as int || p > slot as int || e != expert as int {
                    if e < expert as int {
                        assert(offsets[e + 1] <= offsets[expert as int]);
                    } else if e > expert as int {
                        assert(offsets[expert as int + 1] <= offsets[e]);
                    }
                    assert(order[p] == previous_order[p]);
                    assert(order[p + 1] == previous_order[p + 1]);
                }
            }
        }
        assignment += 1;
    }

    Ok(ExpertAssignmentPlan { order, offsets })
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
