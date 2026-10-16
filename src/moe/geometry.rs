//! Dependency-free integer geometry checks for sparse expert banks and routers.
//!
//! A sparse block has `E` routed experts, each choosing `K` per token, and an
//! optional group-limited router that splits the experts into `G` equal,
//! contiguous groups of `E / G` and keeps the best `T` of them, leaving
//! `T * (E / G)` experts eligible for the final top-`K`. These checks decide
//! which integer configurations are admissible; the callers turn a rejection
//! into the contextual [`crate::Error`] that names the offending field.
//!
//! # Verified checks
//!
//! Every function here is checked in place by Verus through `verus/moe.rs`.
//! The proofs establish, for every input, that no arithmetic overflows (in
//! particular `T * (E / G) <= E` once `T <= G`), that each check accepts
//! exactly the documented configurations, and that each rejection names the
//! first violated condition in the order the configuration is validated. For
//! accepted groups, [`group_widths`] tiles the experts exactly and never
//! ranks more scores than a group holds.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// First violated integer constraint of a sparse expert bank.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpertBankViolation {
    /// The hidden width, intermediate width, or routed expert count is zero.
    ZeroWidth,
    /// `hidden * intermediate` (one expert matrix) overflows `usize`.
    ExpertMatrixOverflow,
    /// `hidden * experts` (the router matrix) overflows `usize`.
    RouterMatrixOverflow,
    /// The experts selected per token are not in `1..=experts`.
    ExpertsPerToken,
}

/// Checks the integer geometry of a sparse expert bank in validation order.
///
/// The proof shows acceptance exactly when every width is non-zero, both
/// `hidden * intermediate` and `hidden * experts` fit in `usize`, and
/// `1 <= experts_per_token <= experts`; a rejection names the first of those
/// conditions that fails.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(checked =>
    ensures
        match checked {
            Ok(()) => {
                &&& hidden > 0 && intermediate > 0 && experts > 0
                &&& hidden * intermediate <= usize::MAX
                &&& hidden * experts <= usize::MAX
                &&& 1 <= experts_per_token <= experts
            },
            Err(ExpertBankViolation::ZeroWidth) => hidden == 0 || intermediate == 0 || experts == 0,
            Err(ExpertBankViolation::ExpertMatrixOverflow) => {
                &&& hidden > 0 && intermediate > 0 && experts > 0
                &&& hidden * intermediate > usize::MAX
            },
            Err(ExpertBankViolation::RouterMatrixOverflow) => {
                &&& hidden > 0 && intermediate > 0 && experts > 0
                &&& hidden * intermediate <= usize::MAX
                &&& hidden * experts > usize::MAX
            },
            Err(ExpertBankViolation::ExpertsPerToken) => {
                &&& hidden > 0 && intermediate > 0 && experts > 0
                &&& hidden * intermediate <= usize::MAX
                &&& hidden * experts <= usize::MAX
                &&& (experts_per_token == 0 || experts_per_token > experts)
            },
        },
))]
pub(crate) fn check_expert_bank(
    hidden: usize,
    intermediate: usize,
    experts: usize,
    experts_per_token: usize,
) -> Result<(), ExpertBankViolation> {
    if hidden == 0 || intermediate == 0 || experts == 0 {
        return Err(ExpertBankViolation::ZeroWidth);
    }
    // Burn parameters are lazy, so initialization alone cannot reject a
    // matrix whose logical element count will overflow at first use.
    if hidden.checked_mul(intermediate).is_none() {
        return Err(ExpertBankViolation::ExpertMatrixOverflow);
    }
    if hidden.checked_mul(experts).is_none() {
        return Err(ExpertBankViolation::RouterMatrixOverflow);
    }
    if experts_per_token == 0 || experts_per_token > experts {
        return Err(ExpertBankViolation::ExpertsPerToken);
    }
    Ok(())
}

/// First violated cardinality constraint of a router.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RouterCardinalityViolation {
    /// The router has no experts.
    NoExperts,
    /// The experts selected per token are not in `1..=experts`.
    ExpertsPerToken,
}

/// Checks that a router has experts and selects between one and all of them.
///
/// The proof shows acceptance exactly when `experts > 0` and
/// `1 <= experts_per_token <= experts`, with an empty router reported first.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(checked =>
    ensures
        match checked {
            Ok(()) => experts > 0 && 1 <= experts_per_token <= experts,
            Err(RouterCardinalityViolation::NoExperts) => experts == 0,
            Err(RouterCardinalityViolation::ExpertsPerToken) => {
                &&& experts > 0
                &&& (experts_per_token == 0 || experts_per_token > experts)
            },
        },
))]
pub(crate) fn check_router_cardinality(
    experts: usize,
    experts_per_token: usize,
) -> Result<(), RouterCardinalityViolation> {
    if experts == 0 {
        return Err(RouterCardinalityViolation::NoExperts);
    }
    if experts_per_token == 0 || experts_per_token > experts {
        return Err(RouterCardinalityViolation::ExpertsPerToken);
    }
    Ok(())
}

/// First violated constraint of group-limited routing.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupGeometryViolation {
    /// The group count is zero or does not divide the experts.
    Groups,
    /// The retained group count is not in `1..=groups`.
    TopGroups,
    /// The retained groups hold fewer experts than are selected per token.
    TooFewEligible {
        /// Experts in the retained groups, `top_groups * (experts / groups)`.
        eligible: usize,
    },
}

/// Checks group-limited routing geometry and returns the experts per group.
///
/// The proof shows acceptance exactly when `groups` is a non-zero divisor of
/// `experts`, `1 <= top_groups <= groups`, and the retained groups hold
/// `top_groups * (experts / groups) >= experts_per_token` experts. On success
/// the result `w` satisfies `w * groups == experts`, so `[groups, w]` exactly
/// reshapes the expert axis, and `top_groups * w <= experts`, so the eligible
/// count never overflows. A rejection names the first failing condition and,
/// for too few eligible experts, their exact number.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(checked =>
    ensures
        match checked {
            Ok(width) => {
                &&& groups > 0
                &&& experts % groups == 0
                &&& width * groups == experts
                &&& 1 <= top_groups <= groups
                &&& top_groups * width <= experts
                &&& top_groups * width >= experts_per_token
            },
            Err(GroupGeometryViolation::Groups) => groups == 0 || experts % groups != 0,
            Err(GroupGeometryViolation::TopGroups) => {
                &&& groups > 0 && experts % groups == 0
                &&& (top_groups == 0 || top_groups > groups)
            },
            Err(GroupGeometryViolation::TooFewEligible { eligible }) => {
                &&& groups > 0 && experts % groups == 0
                &&& 1 <= top_groups <= groups
                &&& eligible == top_groups * (experts / groups)
                &&& eligible < experts_per_token
            },
        },
))]
pub(crate) fn check_group_geometry(
    experts: usize,
    experts_per_token: usize,
    groups: usize,
    top_groups: usize,
) -> Result<usize, GroupGeometryViolation> {
    if groups == 0 || !experts.is_multiple_of(groups) {
        return Err(GroupGeometryViolation::Groups);
    }
    if top_groups == 0 || top_groups > groups {
        return Err(GroupGeometryViolation::TopGroups);
    }
    let width = experts / groups;
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_groups_tile(experts as int, groups as int, top_groups as int);
    }
    let eligible = top_groups * width;
    if eligible < experts_per_token {
        return Err(GroupGeometryViolation::TooFewEligible { eligible });
    }
    Ok(width)
}

/// Group-limited routing widths: experts per group and the scores ranked inside each group.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GroupWidths {
    /// Experts in each contiguous group, `experts / groups`.
    pub(crate) per_group: usize,
    /// Scores a top-two group score sums, `min(2, per_group)`.
    pub(crate) ranked: usize,
}

/// Returns the per-group expert count and the top-two group-score width.
///
/// The proof shows `per_group * groups == experts`, so `[groups, per_group]`
/// reshapes the expert axis exactly into contiguous groups, and that the
/// ranked width is `min(2, per_group)`, never more than a group holds, which
/// is the `k <= columns` precondition of the deterministic top-k. The
/// precondition is what [`check_group_geometry`] accepts.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(widths =>
    requires
        groups > 0,
        experts % groups == 0,
    ensures
        widths.per_group * groups == experts,
        widths.ranked == if widths.per_group < 2 { widths.per_group } else { 2 },
        widths.ranked <= widths.per_group,
))]
pub(crate) fn group_widths(experts: usize, groups: usize) -> GroupWidths {
    let per_group = experts / groups;
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_groups_tile(experts as int, groups as int, 0);
    }
    GroupWidths {
        per_group,
        ranked: per_group.min(2),
    }
}

/// Returns whether an observed expert-ID range `[minimum, maximum]` lies in `0..experts`.
///
/// The proof shows the result is exactly `0 <= minimum` and
/// `0 <= maximum < experts`; when `minimum <= maximum`, as for the bounds of
/// a reduction, every ID between them is then a valid expert.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(within =>
    ensures
        within == (0 <= minimum && 0 <= maximum && maximum < experts),
))]
pub(crate) fn expert_ids_within(minimum: i64, maximum: i64, experts: usize) -> bool {
    minimum >= 0 && maximum >= 0 && (maximum as u64) < (experts as u64)
}

#[cfg(verus_keep_ghost)]
verus! {

/// Equal groups tile the experts, and fewer retained groups hold no more than all experts.
proof fn lemma_groups_tile(experts: int, groups: int, top_groups: int)
    requires
        groups > 0,
        experts >= 0,
        experts % groups == 0,
        0 <= top_groups <= groups,
    ensures
        (experts / groups) * groups == experts,
        top_groups * (experts / groups) <= experts,
        experts / groups >= 0,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(experts, groups);
    vstd::arithmetic::div_mod::lemma_div_pos_is_pos(experts, groups);
    assert((experts / groups) * groups == experts) by (nonlinear_arith)
        requires
            experts == groups * (experts / groups) + experts % groups,
            experts % groups == 0;
    assert(top_groups * (experts / groups) <= groups * (experts / groups)) by (nonlinear_arith)
        requires
            top_groups <= groups,
            experts / groups >= 0;
}

} // verus!

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod tests;
