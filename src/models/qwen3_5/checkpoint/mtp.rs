//! Which Qwen3.5 multi-token-prediction tensors a checkpoint contract covers.
//!
//! A configuration declaring `L > 0` MTP layers ships `mtp.*` tensors. The
//! contract either loads them (when the caller asks for the MTP graph) or
//! exempts the whole `mtp.` namespace as an intentionally ignored optional
//! subsystem, so a declared MTP stage is never reported as unknown keys and
//! never silently half-loaded. Without MTP layers there is nothing to cover.
//! Dedicated MTP embeddings have no released tensor contract, so requesting
//! the MTP graph for such a configuration is rejected.
//!
//! # Verified properties
//!
//! Verus checks [`mtp_inventory`] in place through `verus/loaders.rs` (see
//! `verus/README.md`): it rejects exactly the dedicated-embedding request,
//! and otherwise loads exactly when the graph is requested and `L > 0`,
//! exempts exactly when it is not requested and `L > 0`, and covers nothing
//! exactly when `L = 0`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// How the contract covers a configuration's `mtp.*` tensors.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MtpInventory {
    /// The configuration declares no MTP layers.
    Absent,
    /// Every MTP tensor is expected and loaded into the MTP graph.
    Load,
    /// The `mtp.` namespace is an allowlisted, intentionally ignored subsystem.
    Exempt,
}

/// The MTP graph was requested for a dedicated-embedding configuration.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DedicatedMtpEmbeddings;

/// Decides how the contract covers the configuration's MTP tensors.
///
/// `include_mtp` requests the MTP graph, `dedicated_embeddings` is the
/// configuration's `mtp_use_dedicated_embeddings`, and `layers` its
/// `mtp_num_hidden_layers`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(inventory =>
    ensures
        (inventory is Err) == (include_mtp && dedicated_embeddings),
        inventory == Ok::<MtpInventory, DedicatedMtpEmbeddings>(MtpInventory::Load)
            <==> include_mtp && !dedicated_embeddings && layers > 0,
        inventory == Ok::<MtpInventory, DedicatedMtpEmbeddings>(MtpInventory::Exempt)
            <==> !include_mtp && layers > 0,
        inventory == Ok::<MtpInventory, DedicatedMtpEmbeddings>(MtpInventory::Absent)
            <==> !(include_mtp && dedicated_embeddings) && layers == 0,
))]
pub(super) fn mtp_inventory(
    include_mtp: bool,
    dedicated_embeddings: bool,
    layers: usize,
) -> Result<MtpInventory, DedicatedMtpEmbeddings> {
    if include_mtp && dedicated_embeddings {
        return Err(DedicatedMtpEmbeddings);
    }
    if layers == 0 {
        Ok(MtpInventory::Absent)
    } else if include_mtp {
        Ok(MtpInventory::Load)
    } else {
        Ok(MtpInventory::Exempt)
    }
}

#[cfg(test)]
#[path = "mtp_tests.rs"]
mod tests;
