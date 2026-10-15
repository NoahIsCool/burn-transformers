//! Streaming, weight-only symmetric quantization for Burnpack artifacts.
//!
//! This module defines an offline storage format rather than a backend tensor
//! format. Source tensors are traversed in flattened row-major order and divided
//! into contiguous groups of `G` values. For a group `x`, symmetric quantization
//! stores
//!
//! `s = max(abs(x)) / q_max` and `q_i = clamp(round(x_i / s), -q_max, q_max)`,
//!
//! where `q_max` is 127 for Q8 and 7 for Q4. An all-zero group uses a scale of
//! one, making its quantized representation canonical. Q8 values are stored as
//! two's-complement bytes. Q4 values use their low four two's-complement bits;
//! the even flattened element occupies the low nibble and the odd element the
//! high nibble. A missing high nibble in an odd-length tensor is zero.
//!
//! Conversion holds one source tensor and at most one group of decoded `f32`
//! values at a time. It therefore composes with the checkpoint loader's
//! one-tensor-at-a-time traversal without staging the whole model. Matrices whose
//! input width ends exactly on every group boundary map losslessly onto Burn's
//! block-QFloat representation. Such artifacts are marked for packed matmul
//! execution; incompatible layouts remain valid reference-only storage.

mod burnpack;
mod checkpoint;
mod config;
mod error;
mod load;
mod manifest;
mod packing;
mod policy;
mod provenance;
mod reference;

pub use burnpack::{
    BURNPACK_MANIFEST_TENSOR, BurnpackAppendOutcome, BurnpackArtifactSummary,
    BurnpackSessionLimits, BurnpackStagingStats, BurnpackValidationReport,
    QuantizedBurnpackArtifact, QuantizedBurnpackSession, StagedTensorSummary,
};
pub use checkpoint::{
    CheckpointQuantizationFailure, CheckpointQuantizationFailureCause,
    CheckpointQuantizationOutcome, CheckpointQuantizationResult, QuantizationExclusionRecord,
    QuantizedCheckpointOptions, quantize_local_checkpoint_to_burnpack,
};
pub use config::{
    PackedValueLayout, QuantizationGroupSize, QuantizedGroupLayout, QuantizedPayloadLayout,
    ScaleLayout, SymmetricQuantization, WeightQuantizationConfig,
};
pub use error::{QuantizationError, Result};
pub use load::{
    AuditedQuantizedExecutionFailure, AuditedQuantizedExecutionOutcome,
    AuditedQuantizedExecutionResult, PackedLoadStats, PackedPreflightStats,
    QuantizedExecutionFailure, QuantizedExecutionOutcome, QuantizedExecutionResult,
    QuantizedSinkError, burn_qfloat_execution_support, execute_local_quantized_load_plan,
    execute_local_quantized_load_plan_audited, execute_quantized_load_plan,
    validate_sidecar_contract,
};
// Re-exported for source compatibility; the item itself carries the deprecation.
#[allow(deprecated)]
pub use manifest::BURNPACK_QUANTIZATION_FORMAT_VERSION;
pub use manifest::{
    BURNPACK_QUANTIZATION_FORMAT_VERSION_CURRENT, BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
    BURNPACK_QUANTIZATION_FORMAT_VERSION_V2, BurnpackByteOrder, BurnpackQuantizationManifest,
    PackedExecutionSupport, QuantizedTensorMetadata, SourceFloatDType,
};
pub use policy::{
    QuantizationEligibility, QuantizationExclusion, classify_weight_for_quantization,
};
pub use provenance::{
    BurnpackProvenance, BurnpackShardProvenance, PROVENANCE_HASH_BUFFER_BYTES,
    ProvenanceExecutionStats, ProvenanceHashStats, hash_local_checkpoint_provenance,
    load_plan_sha256, verify_load_plan_provenance, verify_local_checkpoint_provenance,
};
pub use reference::{
    QuantizedWeight, ReferenceQuantizationStats, WeightQuantizationOutcome, quantize_tensor_data,
    quantize_weight,
};

#[cfg(test)]
mod burnpack_tests;
#[cfg(test)]
#[path = "quantization_tests.rs"]
mod tests;
