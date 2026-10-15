//! Stateless causal-input validation and position resolution.
//!
//! Every family accepts the same public [`CausalLmInput`]: token IDs
//! `x ∈ ℤ^{B×S}`, an optional validity mask `m ∈ {0, 1}^{B×S}` (`true` marks a
//! real token), and optional explicit positions `p ∈ ℤ^{B×S}`. Before any
//! embedding lookup or rotary table construction, [`validate_input`] checks
//!
//! * a non-empty grid and that every tensor lives on the model's device,
//! * that the mask and positions share the `[B, S]` grid,
//! * token values `0 <= x < V`, at least one valid token per row (a fully
//!   masked row would leave its attention softmax without a key), and
//!   positions `0 <= p < P_max`,
//!
//! with all value checks sharing one device-to-host readback (see
//! [`DeferredChecks`]). Size and device errors are reported before value
//! errors; among value errors tokens precede the mask, which precedes
//! positions. [`defer_input`] registers the same checks on a caller's list
//! instead, so an encoder-decoder pass can validate its source, its target,
//! and a supplied encoder mask with one readback.
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
//! Explicit positions at padded columns are kept by default; the custom
//! family replaces them by zero ([`PaddedPositions::ZeroFill`]) so padding
//! sentinels are inert. Positions are inspected only under a bound `P_max`;
//! the rotary context extent `1 + max p` then comes from the same readback.

use burn::tensor::{Int, Tensor, backend::Backend};

use crate::{CausalLmInput, Error, Result, architecture::TokenMask};

use super::checks::{
    DeferredChecks, RangeId, Readout, RowCountId, ensure_grid, ensure_nonempty, ensure_same_device,
};

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

/// Treatment of caller-supplied positions at padded columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PaddedPositions {
    /// Keep the caller's values; they are validated like any other position.
    Keep,
    /// Replace padded values by `0` before validation, so sentinels are inert.
    ZeroFill,
}

/// Complete stateless input contract of one family entry point.
///
/// Construct with [`InputContract::new`] and refine the defaults (no position
/// bound, valid-token ordinals, kept padded positions) with the builder
/// methods. Fully masked rows are always rejected.
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
    /// Treatment of explicit positions at padded columns.
    pub(crate) padded_positions: PaddedPositions,
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
            padded_positions: PaddedPositions::Keep,
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

    /// Selects how explicit positions at padded columns are treated.
    pub(crate) const fn with_padded_positions(mut self, padded: PaddedPositions) -> Self {
        self.padded_positions = padded;
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
/// Use [`defer_input`] instead to share that read with further checks.
///
/// # Errors
///
/// Returns [`Error::Shape`] for empty or mismatched extents, out-of-range
/// tokens or positions, and fully masked rows; [`Error::BackendCapability`]
/// for device mismatches or failed readback.
pub(crate) fn validate_input<B: Backend>(
    input: &CausalLmInput<B>,
    device: &B::Device,
    contract: &InputContract<'_>,
) -> Result<ValidatedInput<B>> {
    let mut checks = DeferredChecks::new();
    let pending = defer_input(input, device, contract, &mut checks)?;
    let readout = checks.run(contract.names.token_ids)?;
    pending.finish(&readout)
}

/// A stateless input whose layout is valid and whose value checks are registered.
///
/// Produced by [`defer_input`]; [`Self::finish`] completes validation once the
/// shared [`DeferredChecks`] have been read back.
#[derive(Debug)]
pub(crate) struct DeferredInput<'a, B: Backend> {
    /// Resolved positions, as [`ValidatedInput::positions`].
    positions: Option<Tensor<B, 2, Int>>,
    /// Width `S` of the token grid.
    sequence: usize,
    /// Smallest per-row valid count, registered when a mask is present.
    row_counts: Option<RowCountId>,
    /// Observed position range, registered when positions are bounded.
    position_range: Option<RangeId>,
    /// Name reported for an unrepresentable rotary context.
    position_name: &'a str,
}

impl<B: Backend> DeferredInput<'_, B> {
    /// Derives the rotary context and padding summary from the shared readback.
    ///
    /// `readout` must come from the [`DeferredChecks`] passed to [`defer_input`];
    /// its predicates (token range, non-empty rows, position range) have
    /// already been enforced by [`DeferredChecks::run`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] for a maximum position without a
    /// representable context extent.
    pub(crate) fn finish(self, readout: &Readout) -> Result<ValidatedInput<B>> {
        let evaluated_context = self
            .position_range
            .map(|range| readout.context(range, self.position_name))
            .transpose()?
            .unwrap_or(0);
        // A row is padding-free exactly when its valid count equals the width S.
        let fully_valid = self.row_counts.is_none_or(|counts| {
            super::plan::count_fills_width(readout.min_row_count(counts), self.sequence)
        });
        Ok(ValidatedInput {
            positions: self.positions,
            evaluated_context,
            fully_valid,
        })
    }
}

/// Checks an input's layout and registers its value checks on a shared list.
///
/// Size and device errors are returned immediately. Token range, non-empty
/// rows, and bounded positions are registered on `checks` in that order, so a
/// caller can append further checks (for example an encoder mask) and still
/// pay a single readback. Call [`DeferredInput::finish`] with the readout.
///
/// # Errors
///
/// Returns [`Error::Shape`] for empty or mismatched extents and
/// [`Error::BackendCapability`] for device mismatches.
pub(crate) fn defer_input<'a, B: Backend>(
    input: &CausalLmInput<B>,
    device: &B::Device,
    contract: &InputContract<'a>,
    checks: &mut DeferredChecks<'a, B>,
) -> Result<DeferredInput<'a, B>> {
    let names = contract.names;
    let [batch, sequence] = input.token_ids.dims();
    ensure_nonempty(batch, sequence, names.token_ids)?;
    validate_layout(input, batch, sequence, device, names, contract.owner)?;
    let positions = resolve_positions(
        input,
        batch,
        sequence,
        contract.implicit_positions,
        contract.padded_positions,
    )?;

    checks.values_below(
        &input.token_ids,
        contract.vocabulary,
        names.token_ids,
        "token",
    );
    let row_counts = input
        .token_mask
        .as_ref()
        .map(|mask| checks.row_counts(mask, names.token_mask));
    let position_range =
        positions
            .as_ref()
            .zip(contract.position_limit)
            .map(|(positions, limit)| {
                checks.values_below(positions, limit, names.position_ids, "position")
            });
    Ok(DeferredInput {
        positions,
        sequence,
        row_counts,
        position_range,
        position_name: names.position_ids,
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
/// Explicit positions are returned unchanged unless [`PaddedPositions::ZeroFill`]
/// replaces padded entries by zero. The caller has already checked that the
/// mask and positions share the token grid and device.
///
/// # Errors
///
/// Returns [`Error::Configuration`] when `sequence` exceeds `i64::MAX`.
pub(crate) fn resolve_positions<B: Backend>(
    input: &CausalLmInput<B>,
    batch: usize,
    sequence: usize,
    implicit: ImplicitPositions,
    padded: PaddedPositions,
) -> Result<Option<Tensor<B, 2, Int>>> {
    if let Some(explicit) = &input.position_ids {
        return Ok(Some(match (padded, &input.token_mask) {
            (PaddedPositions::ZeroFill, Some(mask)) => explicit
                .clone()
                .mask_fill(mask.as_tensor().clone().bool_not(), 0),
            _ => explicit.clone(),
        }));
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

/// Validates position values against `0..limit` and returns the context `1 + max p`.
///
/// Used by staged entry points that receive already resolved positions; it
/// performs exactly one device-to-host read.
///
/// # Errors
///
/// Returns [`Error::Shape`] for out-of-range positions and
/// [`Error::BackendCapability`] when the backend cannot read the values.
pub(crate) fn validate_position_values<B: Backend>(
    positions: &Tensor<B, 2, Int>,
    limit: usize,
    name: &str,
) -> Result<usize> {
    let mut checks = DeferredChecks::new();
    let range = checks.values_below(positions, limit, name, "position");
    checks.run(name)?.context(range, name)
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
        checks.row_counts(mask, name);
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
