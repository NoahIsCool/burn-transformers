//! Loader policy and resource bounds.

use serde::{Deserialize, Serialize};

use super::TensorDType;

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
    /// Prefer `model.safetensors.index.json`, then `model.safetensors`, and only
    /// then fall back to an unambiguous nonstandard filename in the root.
    #[default]
    Auto,
    /// Load a specific checkpoint index, relative to the repository root.
    Index(String),
    /// Load a specific unsharded SafeTensors file, relative to the repository root.
    Single(String),
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
    Cast(TensorDType),
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
}
