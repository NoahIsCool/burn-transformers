//! Loader policy and resource bounds.
//!
//! Policy is explicit data shared by discovery and load planning. Resource bounds
//! constrain untrusted metadata before weight allocation; missing/unexpected key
//! exceptions and index aliases must be named rather than inferred. Dtype policy
//! selects float conversion while concrete architecture dtype requirements remain
//! authoritative for numerically sensitive parameters. The range-hazard policy
//! decides whether a dtype that a family declares its checkpoints overflow
//! refuses the load (the default) or runs the faithful graph (see
//! [`super::RangeHazard`]).

use serde::{Deserialize, Serialize};

use super::{RangeHazardPolicy, TensorDType};

/// One checkpoint tensor name required to be equivalent to a storage owner.
///
/// When an index retains a framework alias that the shard writer deduplicated,
/// discovery accepts the absent entry only when both keys share a shard and the
/// owner is physically present. When both entries have physical payloads, pass
/// two requires identical shape, dtype, and raw bytes before any model mutation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IndexTensorAlias {
    /// Index key that may be absent from its assigned shard header.
    pub alias: String,
    /// Indexed key whose validated tensor payload owns the shared storage.
    pub storage_key: String,
}

impl IndexTensorAlias {
    /// Creates one exact alias-to-storage declaration.
    #[must_use]
    pub fn new(alias: impl Into<String>, storage_key: impl Into<String>) -> Self {
        Self {
            alias: alias.into(),
            storage_key: storage_key.into(),
        }
    }
}

/// Defines how `metadata.total_size` accounts for index-only aliases.
///
/// SafeTensors exporters have used both conventions. The selected policy is
/// explicit so discovery never guesses which byte total a checkpoint intended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum IndexAliasSizeAccounting {
    /// Count each logical index entry, including an alias owner's byte length.
    #[default]
    Logical,
    /// Count each physical payload once, excluding index-only aliases.
    Physical,
}

/// Hard resource bounds applied while parsing untrusted artifact metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactLimits {
    /// Maximum bytes in one SafeTensors JSON header.
    pub max_header_bytes: u64,
    /// Maximum bytes in a sharded checkpoint index.
    pub max_index_bytes: u64,
    /// Maximum number of shards referenced by an index.
    pub max_shards: usize,
    /// Maximum total tensor count across all shards.
    pub max_tensors: usize,
    /// Maximum rank of one tensor.
    pub max_rank: usize,
    /// Maximum entries in a SafeTensors `__metadata__` object.
    pub max_metadata_entries: usize,
}

impl Default for ArtifactLimits {
    /// Sets finite metadata bounds without allocating buffers or inspecting artifacts.
    fn default() -> Self {
        Self {
            max_header_bytes: 100_000_000,
            max_index_bytes: 64 * 1024 * 1024,
            max_shards: 16_384,
            max_tensors: 10_000_000,
            max_rank: 32,
            max_metadata_entries: 16_384,
        }
    }
}

/// Selects which local SafeTensors artifact discovery should inspect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum ArtifactSelection {
    /// Use whichever of `model.safetensors.index.json` and `model.safetensors`
    /// exists, and only then fall back to an unambiguous nonstandard filename in
    /// the root. When both standard artifacts exist, discovery loads through
    /// the index only if it parses and its `weight_map` is non-empty and
    /// assigns every tensor to `model.safetensors` (so both provably describe
    /// the same weights); otherwise it reports
    /// [`super::CheckpointError::AmbiguousArtifacts`] instead of guessing, and
    /// the caller selects one with [`Self::Index`] or [`Self::Single`].
    #[default]
    Auto,
    /// Load a specific checkpoint index, relative to the repository root.
    ///
    /// Shard names in its `weight_map` resolve relative to the index's own
    /// directory, so a nested index finds its co-located shards.
    Index(
        /// Root-relative index path checked for containment during discovery.
        String,
    ),
    /// Load a specific unsharded SafeTensors file, relative to the repository root.
    Single(
        /// Root-relative SafeTensors path checked for containment during discovery.
        String,
    ),
}

/// Determines the dtype produced by the second-pass tensor loader.
///
/// The policy never consults the backend: Burn parameters keep whatever dtype
/// the loader delivers, even when it differs from the backend's default float
/// element. To execute a BF16 or F16 checkpoint in F32 on `Flex<f32, _>`, for
/// example, select [`Self::Cast`]`(`[`TensorDType::F32`]`)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum DTypePolicy {
    /// Keep the checkpoint's float dtype unless the architecture contract pins one.
    ///
    /// The expected dtype is the contract's per-tensor
    /// [`super::ExpectedTensor`] dtype, not the backend's float element. A float
    /// tensor whose contract names a float dtype (such as an F32 router or
    /// recurrent buffer) is converted to it; every other tensor keeps its
    /// stored dtype. Ordinary dtype-generic weights therefore stay in the
    /// checkpoint's precision: a BF16 checkpoint loaded into `Flex<f32, _>`
    /// holds BF16 parameters and runs its forward pass in BF16. Use
    /// [`Self::Cast`]`(`[`TensorDType::F32`]`)` for F32 execution.
    #[default]
    Auto,
    /// Never convert checkpoint dtypes.
    ///
    /// Planning fails with [`super::CheckpointError::DTypeMismatch`] when the
    /// contract pins a dtype the checkpoint does not store.
    Preserve,
    /// Convert dtype-generic real floating-point tensors to the selected dtype.
    ///
    /// An architecture's concrete [`super::ExpectedTensor`] dtype remains
    /// authoritative for numerically sensitive parameters and buffers.
    Cast(
        /// Requested real floating-point destination dtype.
        TensorDType,
    ),
}

/// Options shared by artifact discovery and strict load-plan construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct LoadOptions {
    /// Artifact selection policy.
    pub artifact: ArtifactSelection,
    /// Dtype conversion policy.
    pub dtype_policy: DTypePolicy,
    /// Defensive metadata allocation bounds.
    pub limits: ArtifactLimits,
    /// Explicit exceptions for expected model keys absent from a checkpoint.
    pub allow_missing: Vec<super::KeyMatcher>,
    /// Explicit exceptions for checkpoint keys not consumed by the text model.
    pub allow_unexpected: Vec<super::KeyMatcher>,
    /// Exact alias/owner equivalences authorized by the selected architecture.
    ///
    /// This is deliberately not a pattern allowlist. An accepted alias must
    /// either resolve to a same-shard index-only owner or pass an exact physical
    /// payload comparison, preventing a distinct tensor from being ignored as
    /// shared storage.
    #[serde(default)]
    pub index_aliases: Vec<IndexTensorAlias>,
    /// Convention used to reconcile index-only aliases with `total_size`.
    #[serde(default)]
    pub index_alias_size_accounting: IndexAliasSizeAccounting,
    /// Whether a load may execute a dtype the family's checkpoints are measured to overflow.
    ///
    /// Families declare such measurements as [`super::RangeHazard`]s on their
    /// [`super::MappingPlan`] (Gemma 3 declares F16), and the shared planner
    /// [`super::build_load_plan`] applies this policy to them, so every
    /// caller planning with the family's mapping plan is covered. The
    /// default, [`RangeHazardPolicy::Refuse`], fails strict planning with
    /// [`super::CheckpointError::RangeHazard`] when a planned parameter takes
    /// a declared dtype, before any payload is read.
    /// [`RangeHazardPolicy::Accept`] is the explicit opt-in that loads and
    /// evaluates the faithful graph, whose logits are then expected to be
    /// non-finite; see its documentation. Absent from serialized options, it
    /// takes the default.
    #[serde(default)]
    pub range_hazards: RangeHazardPolicy,
}
