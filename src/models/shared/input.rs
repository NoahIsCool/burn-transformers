//! Stateless causal-input validation and position resolution.
//!
//! Every family accepts the same public [`CausalLmInput`]: token IDs
//! `x ∈ ℤ^{B×S}`, an optional validity mask `m ∈ {0, 1}^{B×S}` (`true` marks a
//! real token), and optional explicit positions `p ∈ ℤ^{B×S}`. Before any
//! embedding lookup or rotary table construction, [`validate_input`] checks
//!
//! * a non-empty grid and that every tensor lives on the model's device,
//! * that the mask and positions share the `[B, S]` grid,
//! * token values `0 <= x < V`, at least one valid token per row (unless the
//!   contract permits empty rows), and positions `0 <= p < P_max`,
//!
//! with all value checks sharing one device-to-host readback (see
//! [`DeferredChecks`]). Size and device errors are reported before value
//! errors; among value errors tokens precede the mask, which precedes
//! positions.
//!
//! # Implicit positions
//!
//! Families disagree only in how omitted positions are numbered, captured by
//! [`ImplicitPositions`]:
//!
//! * valid-token ordinals `p[b, t] = Σ_{u <= t} m[b, u] - 1`, zero at padding,
//!   so left padding does not shift RoPE phases (Qwen, DeepSeek, Muse);
//! * physical columns `p[b, t] = t`, including padding, as the reference GPT-2
//!   and GPT-NeoX implementations number learned or rotary positions;
//! * no positions at all for relative or position-free mixers (T5, Nemotron-H),
//!   which still validate explicit positions when they accept them.
//!
//! Positions are inspected only under a bound `P_max`; the rotary context
//! extent `1 + max p` then comes from the same readback.

use burn::tensor::{Int, Tensor, backend::Backend};

use crate::{CausalLmInput, Error, Result, architecture::TokenMask};

use super::checks::{DeferredChecks, ensure_grid, ensure_nonempty, ensure_same_device};

/// Logical tensor names that appear in a family's input errors.
///
/// Names follow the `family.tensor` convention, e.g. `qwen3_moe.input_ids`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InputNames<'a> {
    /// Name of the `[batch, sequence]` token-ID tensor.
    pub(crate) token_ids: &'a str,
    /// Name of the `[batch, sequence]` validity mask.
    pub(crate) token_mask: &'a str,
    /// Name of the `[batch, sequence]` explicit position tensor.
    pub(crate) position_ids: &'a str,
}

/// Numbering used when the caller omits `position_ids`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImplicitPositions {
    /// `p[b, t] = Σ_{u <= t} m[b, u] - 1` at valid tokens and `0` at padding.
    ValidTokenOrdinals,
    /// `p[b, t] = t` for every column, including padding.
    PhysicalColumns,
    /// No positions are materialized; explicit positions are still validated.
    Unused,
}

/// Policy for rows that contain no valid token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmptyRows {
    /// Reject fully masked rows, whose attention softmax would have no key.
    Reject,
    /// Accept fully masked rows (finished sequences during decoding).
    Permit,
}

/// Complete stateless input contract of one family entry point.
///
/// Construct with [`InputContract::new`] and refine the defaults (no position
/// bound, valid-token ordinals, rejected empty rows) with the builder methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InputContract<'a> {
    /// Tensor names reported in errors.
    pub(crate) names: InputNames<'a>,
    /// Owner named in device errors, e.g. `Qwen3 MoE model parameters`.
    pub(crate) owner: &'a str,
    /// Exclusive bound `V` on token IDs.
    pub(crate) vocabulary: usize,
    /// Exclusive bound on positions; `None` leaves positions uninspected.
    pub(crate) position_limit: Option<usize>,
    /// Numbering of omitted positions.
    pub(crate) implicit_positions: ImplicitPositions,
    /// Policy for fully masked rows.
    pub(crate) empty_rows: EmptyRows,
}

impl<'a> InputContract<'a> {
    /// Creates a contract with the most common family defaults.
    pub(crate) const fn new(names: InputNames<'a>, owner: &'a str, vocabulary: usize) -> Self {
        Self {
            names,
            owner,
            vocabulary,
            position_limit: None,
            implicit_positions: ImplicitPositions::ValidTokenOrdinals,
            empty_rows: EmptyRows::Reject,
        }
    }

    /// Requires every resolved position to lie in `0..limit`.
    pub(crate) const fn with_position_limit(mut self, limit: usize) -> Self {
        self.position_limit = Some(limit);
        self
    }

    /// Selects how omitted positions are numbered.
    pub(crate) const fn with_implicit_positions(mut self, implicit: ImplicitPositions) -> Self {
        self.implicit_positions = implicit;
        self
    }
}

/// A stateless input that satisfied its [`InputContract`].
#[derive(Debug, Clone)]
pub(crate) struct ValidatedInput<B: Backend> {
    /// Resolved `[B, S]` positions; `None` only for [`ImplicitPositions::Unused`]
    /// without explicit positions.
    pub(crate) positions: Option<Tensor<B, 2, Int>>,
    /// Rotary extent `1 + max p` for bounded positions, otherwise `0`.
    pub(crate) evaluated_context: usize,
    /// Whether every token is valid (no mask, or a mask without padding).
    pub(crate) fully_valid: bool,
}

impl<B: Backend> ValidatedInput<B> {
    /// Returns the resolved positions, which position-dependent families require.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] when the contract resolved no positions.
    pub(crate) fn require_positions(&self, name: &str) -> Result<Tensor<B, 2, Int>> {
        self.positions
            .clone()
            .ok_or_else(|| Error::configuration(name, "this contract resolves no positions"))
    }
}

/// Validates a stateless input and resolves its positions.
///
/// `device` is the device every input tensor must share, normally the model
/// parameters' device. Exactly one blocking device-to-host read is performed.
///
/// # Errors
///
/// Returns [`Error::Shape`] for empty or mismatched extents, out-of-range
/// tokens or positions, and rejected empty rows; [`Error::BackendCapability`]
/// for device mismatches or failed readback.
pub(crate) fn validate_input<B: Backend>(
    input: &CausalLmInput<B>,
    device: &B::Device,
    contract: &InputContract<'_>,
) -> Result<ValidatedInput<B>> {
    let names = contract.names;
    let [batch, sequence] = input.token_ids.dims();
    ensure_nonempty(batch, sequence, names.token_ids)?;
    validate_layout(input, batch, sequence, device, names, contract.owner)?;
    let positions = resolve_positions(input, batch, sequence, contract.implicit_positions)?;

    let mut checks = DeferredChecks::new();
    checks.values_below(
        &input.token_ids,
        contract.vocabulary,
        names.token_ids,
        "token",
    );
    let row_counts = input.token_mask.as_ref().map(|mask| {
        checks.row_counts(
            mask,
            names.token_mask,
            contract.empty_rows == EmptyRows::Reject,
        )
    });
    let position_range =
        positions
            .as_ref()
            .zip(contract.position_limit)
            .map(|(positions, limit)| {
                checks.values_below(positions, limit, names.position_ids, "position")
            });
    let readout = checks.run(names.token_ids)?;

    let evaluated_context = position_range
        .map(|range| readout.context(range, names.position_ids))
        .transpose()?
        .unwrap_or(0);
    // A row is padding-free exactly when its valid count equals the width S.
    let fully_valid = row_counts.is_none_or(|counts| {
        usize::try_from(readout.min_row_count(counts)).is_ok_and(|count| count == sequence)
    });
    Ok(ValidatedInput {
        positions,
        evaluated_context,
        fully_valid,
    })
}

/// Checks devices and `[batch, sequence]` extents of the mask and positions.
fn validate_layout<B: Backend>(
    input: &CausalLmInput<B>,
    batch: usize,
    sequence: usize,
    device: &B::Device,
    names: InputNames<'_>,
    owner: &str,
) -> Result<()> {
    ensure_same_device(names.token_ids, &input.token_ids.device(), device, owner)?;
    if let Some(mask) = &input.token_mask {
        ensure_same_device(names.token_mask, &mask.as_tensor().device(), device, owner)?;
        ensure_grid(mask.as_tensor().dims(), batch, sequence, names.token_mask)?;
    }
    if let Some(positions) = &input.position_ids {
        ensure_same_device(names.position_ids, &positions.device(), device, owner)?;
        ensure_grid(positions.dims(), batch, sequence, names.position_ids)?;
    }
    Ok(())
}

/// Resolves explicit positions or derives the family's implicit numbering.
///
/// Explicit positions are returned unchanged. The caller has already checked
/// that the mask and positions share the token grid and device.
///
/// # Errors
///
/// Returns [`Error::Configuration`] when `sequence` exceeds `i64::MAX`.
pub(crate) fn resolve_positions<B: Backend>(
    input: &CausalLmInput<B>,
    batch: usize,
    sequence: usize,
    implicit: ImplicitPositions,
) -> Result<Option<Tensor<B, 2, Int>>> {
    if let Some(explicit) = &input.position_ids {
        return Ok(Some(explicit.clone()));
    }
    Ok(match (implicit, &input.token_mask) {
        (ImplicitPositions::ValidTokenOrdinals, Some(mask)) => Some(valid_token_ordinals(mask)),
        (ImplicitPositions::ValidTokenOrdinals, None) | (ImplicitPositions::PhysicalColumns, _) => {
            Some(physical_columns(
                batch,
                sequence,
                &input.token_ids.device(),
            )?)
        }
        (ImplicitPositions::Unused, _) => None,
    })
}

/// Computes `p[b, t] = Σ_{u <= t} m[b, u] - 1` at valid tokens and `0` at padding.
///
/// The cumulative sum runs in the backend's integer dtype, so the result is
/// exact for any representable sequence length.
pub(crate) fn valid_token_ordinals<B: Backend>(mask: &TokenMask<B>) -> Tensor<B, 2, Int> {
    let valid = mask.as_tensor().clone();
    (valid.clone().int().cumsum(1) - 1).mask_fill(valid.bool_not(), 0)
}

/// Returns `p[b, t] = t` broadcast over `batch` rows without copying.
///
/// # Errors
///
/// Returns [`Error::Configuration`] when `sequence` exceeds `i64::MAX`.
pub(crate) fn physical_columns<B: Backend>(
    batch: usize,
    sequence: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
    let end = i64::try_from(sequence).map_err(|_| {
        Error::configuration("sequence", "sequence length exceeds the signed index range")
    })?;
    Ok(Tensor::<B, 1, Int>::arange(0..end, device)
        .reshape([1, sequence])
        .expand([batch, sequence]))
}

/// Validates token values against `0..vocabulary` with one device-to-host read.
///
/// # Errors
///
/// Returns [`Error::Shape`] for out-of-vocabulary IDs and
/// [`Error::BackendCapability`] when the backend cannot read the values.
pub(crate) fn validate_token_values<B: Backend>(
    token_ids: &Tensor<B, 2, Int>,
    vocabulary: usize,
    name: &str,
) -> Result<()> {
    let mut checks = DeferredChecks::new();
    checks.values_below(token_ids, vocabulary, name, "token");
    checks.run(name).map(|_| ())
}

/// Checks a mask's `[batch, sequence]` extent and defers its empty-row check.
///
/// Use this to fold mask validation into a caller's [`DeferredChecks`], so the
/// row counts share the caller's single readback. An absent mask is accepted.
///
/// # Errors
///
/// Returns [`Error::Shape`] for a mismatched extent.
pub(crate) fn check_nonempty_mask<'a, B: Backend>(
    checks: &mut DeferredChecks<'a, B>,
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
    name: &'a str,
) -> Result<()> {
    if let Some(mask) = mask {
        ensure_grid(mask.as_tensor().dims(), batch, sequence, name)?;
        checks.row_counts(mask, name, true);
    }
    Ok(())
}

/// Validates a mask's extent and rejects rows without a valid token.
///
/// # Errors
///
/// Returns [`Error::Shape`] for a mismatched extent or a fully masked row and
/// [`Error::BackendCapability`] when the backend cannot read the counts.
pub(crate) fn validate_nonempty_mask<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
    name: &str,
) -> Result<()> {
    let mut checks = DeferredChecks::new();
    check_nonempty_mask(&mut checks, mask, batch, sequence, name)?;
    checks.run(name).map(|_| ())
}

#[cfg(test)]
#[path = "input_tests.rs"]
mod tests;
