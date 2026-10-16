//! A cheap check, before any contract exists, that a checkpoint can hold a configuration.
//!
//! A family's tensor contract lists one entry per parameter, so building it
//! costs time and memory proportional to the declared depth `L` (up to
//! [`MAX_LAYERS`](crate::models::config::MAX_LAYERS)) and, for mixture-of-experts
//! families, to `L · E` experts. Configurations are untrusted input: a hostile
//! `config.json` within those bounds, paired with a one-tensor checkpoint, used
//! to cost seconds and gigabytes before strict planning reported the mismatch.
//!
//! Every family can state, in `O(L)` time and without allocating, a lower bound
//! `r` on the distinct source tensors any loadable checkpoint must contain:
//! every layer owns at least one tensor, and one-to-one contracts know their
//! exact inventory. A discovered checkpoint offers at most
//! `a = |tensors| + |index aliases|` sources, so `r > a` proves that planning
//! must fail, and the loader can say so before it builds the contract.
//!
//! A missing-tensor allowlist ([`LoadOptions::allow_missing`]) can excuse any
//! number of tensors, so the bound proves nothing when one is configured and
//! the check is skipped.

use super::{CheckpointError, CheckpointManifest, LoadOptions, Result};

/// Rejects a checkpoint that cannot provide `required` distinct source tensors.
///
/// `required` must be a lower bound on the sources the caller's contract
/// consumes; the check then never rejects a checkpoint that strict planning
/// would accept.
///
/// # Errors
///
/// [`CheckpointError::TooFewTensors`] when `required` exceeds the manifest's
/// tensors plus resolved index aliases and `options` has no missing-tensor
/// allowlist.
pub(crate) fn require_source_tensors(
    manifest: &CheckpointManifest,
    options: &LoadOptions,
    required: usize,
) -> Result<()> {
    if !options.allow_missing.is_empty() {
        return Ok(());
    }
    let available = manifest
        .tensors
        .len()
        .saturating_add(manifest.index_aliases.len());
    tensor_budget(required, available)
}

/// Accepts exactly when `required <= available`.
fn tensor_budget(required: usize, available: usize) -> Result<()> {
    if required <= available {
        Ok(())
    } else {
        Err(CheckpointError::TooFewTensors {
            required,
            available,
        })
    }
}

#[cfg(test)]
#[path = "budget_tests.rs"]
mod tests;
