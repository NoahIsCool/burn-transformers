//! Safe, bounded discovery and planning for local SafeTensors checkpoints.
//!
//! Model repositories are input data, not trusted code. In particular, a Hugging
//! Face shard index controls filenames and tensor names. This module therefore
//! validates paths before opening them, bounds all metadata allocations, verifies
//! every byte range in each SafeTensors header, and compares the complete observed
//! tensor set with an explicit architecture-owned mapping plan.
//!
//! Discovery is deliberately separate from parameter application. Pass one reads
//! only the index and shard headers and produces a [`CheckpointManifest`]. Pass two
//! is described by a [`LoadPlan`] and executed through [`execute_load_plan`]. The
//! executor owns at most one source payload and one transformation output at a
//! time, then transfers the resulting [`burn::tensor::TensorData`] into an
//! architecture-owned [`TensorSink`]. That separation makes malformed artifacts
//! fail before expensive model allocations occur and avoids staging the complete
//! checkpoint in host memory.

pub(crate) mod accounting;
mod artifact;
mod dtype;
mod error;
mod layout;
mod mapping;
mod options;
pub mod quantization;
mod report;
mod runtime;

pub(crate) use artifact::discover_checkpoint_with_index_total_size_policy;
pub use artifact::{
    ArtifactSource, CheckpointManifest, ResolvedIndexTensorAlias, ShardInventory, TensorInventory,
    discover_checkpoint,
};
pub use dtype::TensorDType;
pub use error::{CheckpointError, Result};
pub(crate) use mapping::build_load_plan_with_index_total_size_policy;
pub use mapping::{
    ExpectedTensor, KeyMatcher, LoadPlan, MappingPlan, PlannedSource, PlannedTensor,
    PlannedTensorEquivalence, PlannedTransform, TensorOperation, TensorTransform, build_load_plan,
};
pub use options::{
    ArtifactLimits, ArtifactSelection, DTypePolicy, IndexAliasSizeAccounting, IndexTensorAlias,
    LoadOptions,
};
pub use report::{
    AliasRecord, CheckpointEquivalenceRecord, DTypeConversion, LoadReport, MemoryEstimate,
    QuantizationRecord, TransformRecord,
};
pub use runtime::{
    ExecutionFailure, ExecutionFailureCause, ExecutionOutcome, ExecutionResult, ExecutionStats,
    LocalShardSource, PackedInstall, PackedLayoutPolicy, ParameterInstallError, SourceStats,
    SourceTensor, TensorSink, TensorSource, execute_load_plan, execute_local_load_plan,
    install_float_parameter, install_packed_parameter, packed_layout_is_exact,
};

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
