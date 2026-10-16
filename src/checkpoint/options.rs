//! Loader policy and resource bounds.
//!
//! Policy is explicit data shared by discovery and load planning. Resource bounds
//! constrain untrusted metadata before weight allocation; missing/unexpected key
//! exceptions and index aliases must be named rather than inferred. Dtype policy
//! selects float conversion while concrete architecture dtype requirements remain
//! authoritative for numerically sensitive parameters.

use serde::{Deserialize, Serialize};

use super::TensorDType;

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
    /// the index only if it parses and its `weight_map` names nothing but
    /// `model.safetensors` (so both provably describe the same weights);
    /// otherwise it reports [`super::CheckpointError::AmbiguousArtifacts`]
    /// instead of guessing, and the caller selects one with [`Self::Index`] or
    /// [`Self::Single`].
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum DTypePolicy {
    /// Use the expected model dtype for safe float-to-float conversions and
    /// otherwise preserve the source dtype.
    #[default]
    Auto,
    /// Never convert checkpoint dtypes.
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
}
