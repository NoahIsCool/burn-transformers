//! Test-only routing oracles built on the production assignment plan.
//!
//! Production dispatch goes through [`super::dispatch_routed_experts`], which
//! walks one expert-major permutation and never materializes per-expert
//! vectors. Tests still want a readable host-side view of that plan, and a
//! uniform-precision shorthand for the validated dispatcher. Both live here,
//! compiled only under `cfg(test)`, so they neither ship in the library nor
//! count as covered production code. Each one is a thin view over a live
//! production function ([`super::read_assignment_plan`] and
//! `SparseMoe::dispatch_selected_experts_with_precision`), so tests through
//! them still exercise the shipped validation.

use burn::tensor::{DType, Int, Tensor, backend::Backend};

use super::{RoutedExpertDispatch, RoutedExpertPrecision, SparseMoe, read_assignment_plan};
use crate::Result;

/// Host metadata for one expert, ordered by flattened `(token, slot)` index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExpertAssignmentGroup {
    /// Token rows selected from the residual stream.
    pub(crate) token_indices: Vec<i64>,
    /// Flattened `(token, top-k slot)` indices selected from router weights.
    pub(crate) weight_indices: Vec<i64>,
}

/// Reads discrete routing IDs once and partitions the assignments by expert.
///
/// Returns one group per expert in `0..routed_experts`, empty for an expert
/// with no assignment. Within each expert, assignments keep token-major,
/// slot-minor order: exactly the segment of the production permutation that
/// [`read_assignment_plan`] assigns to that expert, so this is a host-side
/// reference for the order in which [`super::dispatch_routed_experts`] sums
/// contributions.
///
/// # Errors
///
/// Returns the validation errors of [`read_assignment_plan`]: empty routing
/// axes, an index space beyond `i64`, and out-of-range expert IDs.
pub(crate) fn group_selected_experts<B: Backend>(
    selected_experts: Tensor<B, 2, Int>,
    routed_experts: usize,
) -> Result<Vec<ExpertAssignmentGroup>> {
    let slots = selected_experts.dims()[1];
    let plan = read_assignment_plan(selected_experts, None, routed_experts)?;
    let mut groups = Vec::with_capacity(routed_experts);
    for expert in 0..routed_experts {
        let assignments = &plan.order[plan.offsets[expert]..plan.offsets[expert + 1]];
        let mut group = ExpertAssignmentGroup::default();
        for &assignment in assignments {
            // `read_assignment_plan` checked that every index fits in i64.
            group.token_indices.push((assignment / slots) as i64);
            group.weight_indices.push(assignment as i64);
        }
        groups.push(group);
    }
    Ok(groups)
}

impl<B: Backend> SparseMoe<B> {
    /// Dispatches with one dtype for both the weight multiplication and the accumulation.
    ///
    /// A shorthand for `dispatch_selected_experts_with_precision` without
    /// non-finite flags and with [`RoutedExpertPrecision::uniform`]; family
    /// tests use it to drive the validated production dispatcher directly.
    ///
    /// # Errors
    ///
    /// Returns every error of `dispatch_selected_experts_with_precision`.
    pub(crate) fn dispatch_selected_experts(
        &self,
        flat_input: Tensor<B, 2>,
        flat_weights: Tensor<B, 2>,
        selected_experts: Tensor<B, 2, Int>,
        accumulation_dtype: DType,
    ) -> Result<RoutedExpertDispatch<B>> {
        self.dispatch_selected_experts_with_precision(
            flat_input,
            flat_weights,
            selected_experts,
            None,
            RoutedExpertPrecision::uniform(accumulation_dtype),
        )
    }
}
