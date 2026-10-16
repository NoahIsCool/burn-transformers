//! Dependency-free feasibility check of Nemotron-H group-limited routing.
//!
//! The router splits `E` routed experts into `G` equal groups of `E / G`,
//! scores each group by its top two experts, keeps the best `g` groups, and
//! selects the top `k` experts among the `(E / G) g` eligible ones. A
//! configuration is feasible when every group holds at least two experts
//! (top-2 group scoring), at most `G` groups are kept, and `k` fits the
//! eligible experts.
//!
//! # Verified check
//!
//! [`routing_groups`] is checked in place by Verus through
//! `verus/nemotron.rs` (see `verus/README.md`): it never overflows or
//! divides by zero, it accepts exactly the feasible configurations and
//! reports the eligible count, every rejection names its exact cause, and
//! for evenly divided experts an accepted configuration never asks for more
//! experts per token than exist.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Why a group-limited routing configuration is infeasible.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoutingError {
    /// There are no groups, or a group holds fewer than two experts.
    SmallGroups,
    /// The eligible expert count `(E / G) g` overflows `usize`.
    EligibleOverflow,
    /// More groups are kept than exist, or `k` exceeds the eligible experts.
    Insufficient,
}

/// Checks group-limited routing and returns the eligible expert count `(E / G) g`.
///
/// `experts` is `E`, `groups` is `G`, `top_groups` is `g`, and `per_token`
/// is `k` (see the module documentation).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(eligible =>
    ensures
        match eligible {
            Ok(eligible) => {
                &&& groups > 0
                &&& experts / groups >= 2
                &&& eligible == (experts / groups) * top_groups
                &&& top_groups <= groups
                &&& per_token <= eligible
                &&& experts % groups == 0 ==> eligible <= experts && per_token <= experts
            },
            Err(RoutingError::SmallGroups) => groups == 0 || experts / groups < 2,
            Err(RoutingError::EligibleOverflow) => {
                &&& groups > 0
                &&& experts / groups >= 2
                &&& (experts / groups) * top_groups > usize::MAX
            },
            Err(RoutingError::Insufficient) => {
                &&& groups > 0
                &&& experts / groups >= 2
                &&& (experts / groups) * top_groups <= usize::MAX
                &&& (top_groups > groups || per_token > (experts / groups) * top_groups)
            },
        },
))]
pub(crate) fn routing_groups(
    experts: usize,
    groups: usize,
    top_groups: usize,
    per_token: usize,
) -> Result<usize, RoutingError> {
    if groups == 0 || experts / groups < 2 {
        return Err(RoutingError::SmallGroups);
    }
    let per_group = experts / groups;
    let eligible = match per_group.checked_mul(top_groups) {
        Some(eligible) => eligible,
        None => return Err(RoutingError::EligibleOverflow),
    };
    if top_groups > groups || per_token > eligible {
        return Err(RoutingError::Insufficient);
    }
    // Keeping at most every group never exceeds the evenly divided experts.
    #[cfg(verus_keep_ghost)]
    proof! {
        if experts % groups == 0 {
            assert(per_group * top_groups <= per_group * groups) by (nonlinear_arith)
                requires top_groups <= groups, per_group >= 0;
            assert(per_group * groups == experts) by (nonlinear_arith)
                requires experts % groups == 0, per_group == experts / groups, groups > 0;
        }
    }
    Ok(eligible)
}

#[cfg(test)]
#[path = "routing_tests.rs"]
mod tests;
