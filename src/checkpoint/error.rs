//! Errors emitted while inspecting or planning a checkpoint load.

use std::{io, path::PathBuf};

use thiserror::Error;

use super::{LoadReport, TensorDType};

/// Result type used by the checkpoint subsystem.
pub type Result<T> = std::result::Result<T, CheckpointError>;

/// A contextual failure produced before or during checkpoint load planning.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CheckpointError {
    /// A filesystem operation failed.
    #[error("failed to {operation} `{path}`: {source}")]
    Io {
        /// Human-readable operation, such as `read shard header`.
        operation: &'static str,
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Operating-system error.
        #[source]
        source: io::Error,
    },
    /// The requested repository root is not a directory.
    #[error("checkpoint root `{path}` is not a directory")]
    RootNotDirectory {
        /// Invalid root path.
        path: PathBuf,
    },
    /// No supported SafeTensors artifact was found.
    #[error("no SafeTensors checkpoint found below `{root}`")]
    ArtifactNotFound {
        /// Canonical repository root.
        root: PathBuf,
    },
    /// Auto-discovery found multiple equally plausible artifacts.
    #[error("ambiguous SafeTensors artifacts: {candidates:?}")]
    AmbiguousArtifacts {
        /// Candidate filenames, sorted for deterministic diagnostics.
        candidates: Vec<String>,
    },
    /// A user- or index-supplied relative path was unsafe.
    #[error("unsafe checkpoint-relative path `{path}`: {reason}")]
    UnsafeRelativePath {
        /// Untrusted path as supplied.
        path: String,
        /// Security rule that rejected it.
        reason: &'static str,
    },
    /// A lexically safe path resolved outside the canonical repository root.
    #[error("checkpoint path `{path}` resolves outside repository root `{root}`")]
    PathEscapesRoot {
        /// Canonical repository root.
        root: PathBuf,
        /// Canonical candidate path.
        path: PathBuf,
    },
    /// A configured metadata limit was exceeded.
    #[error("{kind} count/size {actual} exceeds configured limit {limit}")]
    LimitExceeded {
        /// Kind of bounded data.
        kind: &'static str,
        /// Observed count or byte size.
        actual: u64,
        /// Configured maximum.
        limit: u64,
    },
    /// A SafeTensors file was shorter than its mandatory prefix.
    #[error("SafeTensors file `{path}` is too short: {size} bytes")]
    TruncatedFile {
        /// Tensor file path.
        path: PathBuf,
        /// Actual file size.
        size: u64,
    },
    /// The SafeTensors header length was invalid for the file.
    #[error("invalid SafeTensors header length {header_bytes} for `{path}` ({file_bytes} bytes)")]
    InvalidHeaderLength {
        /// Tensor file path.
        path: PathBuf,
        /// Declared JSON header length.
        header_bytes: u64,
        /// Complete file length.
        file_bytes: u64,
    },
    /// JSON metadata could not be decoded.
    #[error("invalid {kind} JSON in `{path}`: {message}")]
    InvalidJson {
        /// Either `SafeTensors header` or `checkpoint index`.
        kind: &'static str,
        /// Input path.
        path: PathBuf,
        /// Parser diagnostic.
        message: String,
    },
    /// A JSON object repeated a key.
    #[error("duplicate key `{key}` in {kind} `{path}`")]
    DuplicateJsonKey {
        /// Either header or index-map context.
        kind: &'static str,
        /// Input path.
        path: PathBuf,
        /// Repeated key.
        key: String,
    },
    /// A tensor used a dtype outside the SafeTensors vocabulary understood here.
    #[error("unsupported SafeTensors dtype `{dtype}`")]
    UnsupportedDType {
        /// Unrecognized wire spelling.
        dtype: String,
    },
    /// A sub-byte tensor did not end at a whole-byte boundary.
    #[error("{dtype} tensor with {elements} elements does not occupy a whole number of bytes")]
    MisalignedTensorStorage {
        /// Tensor dtype.
        dtype: TensorDType,
        /// Tensor element count.
        elements: u64,
    },
    /// Checked arithmetic overflowed while validating untrusted metadata.
    #[error("integer overflow while {context}")]
    ArithmeticOverflow {
        /// Operation being performed.
        context: &'static str,
    },
    /// A tensor's described data interval was structurally invalid.
    #[error("invalid data offsets for tensor `{tensor}` in `{shard}`: {reason}")]
    InvalidDataOffsets {
        /// Tensor name.
        tensor: String,
        /// Shard filename.
        shard: String,
        /// Violated invariant.
        reason: String,
    },
    /// A tensor shape and dtype disagreed with its described byte interval.
    #[error("tensor `{tensor}` describes {actual} bytes but shape/dtype require {expected}")]
    TensorByteLengthMismatch {
        /// Tensor name.
        tensor: String,
        /// Required byte length.
        expected: u64,
        /// Described byte length.
        actual: u64,
    },
    /// A tensor key occurred in more than one shard.
    #[error("tensor `{tensor}` appears in both `{first_shard}` and `{second_shard}`")]
    DuplicateTensor {
        /// Repeated tensor name.
        tensor: String,
        /// First declaring shard.
        first_shard: String,
        /// Later declaring shard.
        second_shard: String,
    },
    /// The index and shard headers disagreed.
    #[error("checkpoint index mismatch for tensor `{tensor}`: {reason}")]
    IndexMismatch {
        /// Tensor name.
        tensor: String,
        /// Mismatch description.
        reason: String,
    },
    /// The index total-size field disagreed with tensor metadata.
    #[error("checkpoint index declares {declared} tensor bytes, headers describe {observed}")]
    TotalSizeMismatch {
        /// Index metadata value.
        declared: u64,
        /// Sum derived from validated tensors.
        observed: u64,
    },
    /// An explicit mapping operation was malformed.
    #[error("invalid tensor mapping: {message}")]
    InvalidMapping {
        /// Deterministic mapping diagnostic.
        message: String,
    },
    /// A mapping referenced an observed tensor that does not exist.
    #[error("mapping source tensor `{tensor}` does not exist")]
    MissingMappingSource {
        /// Missing observed key.
        tensor: String,
    },
    /// A mapping attempted to construct a target unknown to the model.
    #[error("mapping target tensor `{tensor}` is not in the expected model inventory")]
    UnknownMappingTarget {
        /// Unknown target key.
        tensor: String,
    },
    /// Two operations attempted to provide one model parameter.
    #[error("multiple mapping operations produce target tensor `{tensor}`")]
    MappingTargetCollision {
        /// Colliding target key.
        tensor: String,
    },
    /// Two consuming operations attempted to reuse a checkpoint tensor.
    #[error("checkpoint tensor `{tensor}` is consumed by multiple mapping operations")]
    MappingSourceCollision {
        /// Reused source key.
        tensor: String,
    },
    /// A mapped tensor shape differed from the model parameter shape.
    #[error("shape mismatch for `{tensor}`: mapped {observed:?}, expected {expected:?}")]
    ShapeMismatch {
        /// Model parameter key.
        tensor: String,
        /// Expected shape.
        expected: Vec<u64>,
        /// Shape produced by the mapping.
        observed: Vec<u64>,
    },
    /// A mapped tensor dtype was incompatible with the configured policy.
    #[error("dtype mismatch for `{tensor}`: observed {observed}, expected {expected}")]
    DTypeMismatch {
        /// Model parameter key.
        tensor: String,
        /// Source or post-policy dtype.
        observed: TensorDType,
        /// Required model dtype.
        expected: TensorDType,
    },
    /// Strict validation found unexplained missing or unexpected keys.
    #[error(
        "strict checkpoint validation failed: {missing} missing, {unexpected} unexpected tensors"
    )]
    StrictValidation {
        /// Number of unexplained expected tensors.
        missing: usize,
        /// Number of unexplained observed tensors.
        unexpected: usize,
        /// Detailed report suitable for user-facing diagnostics.
        report: Box<LoadReport>,
    },
    /// A shard changed after its header was inventoried during pass one.
    #[error("checkpoint artifact `{path}` changed between discovery and loading: {reason}")]
    ArtifactChanged {
        /// Canonical shard path recorded by discovery.
        path: PathBuf,
        /// Metadata invariant that no longer holds.
        reason: String,
    },
    /// A pass-two source descriptor disagreed with the discovered manifest.
    #[error("load plan source `{tensor}` does not match the checkpoint manifest: {reason}")]
    PlanManifestMismatch {
        /// Source tensor named by the plan.
        tensor: String,
        /// Descriptor mismatch.
        reason: String,
    },
    /// Tensor payload bytes were inconsistent with their declared metadata.
    #[error("invalid runtime payload for tensor `{tensor}`: {reason}")]
    InvalidTensorPayload {
        /// Tensor being materialized.
        tensor: String,
        /// Violated payload or shape invariant.
        reason: String,
    },
    /// A valid SafeTensors dtype has no unquantized Burn tensor representation.
    #[error("SafeTensors dtype {dtype} cannot be materialized by the Burn runtime")]
    UnsupportedRuntimeDType {
        /// Source dtype without a matching Burn [`burn::tensor::DType`].
        dtype: TensorDType,
    },
    /// The runtime cannot perform the requested scalar conversion.
    #[error("unsupported runtime dtype conversion from {source_dtype} to {target_dtype}")]
    UnsupportedDTypeConversion {
        /// Stored checkpoint dtype.
        source_dtype: TensorDType,
        /// Requested target dtype.
        target_dtype: TensorDType,
    },
    /// A checked host allocation could not be reserved.
    #[error("failed to reserve {bytes} host bytes while {context}")]
    AllocationFailed {
        /// Requested allocation size.
        bytes: u64,
        /// Operation requiring the allocation.
        context: &'static str,
    },
    /// Runtime staging would exceed the conservative pass-one bound.
    #[error("runtime staging for `{tensor}` requires {required} bytes, plan permits {planned}")]
    MemoryEstimateExceeded {
        /// Model parameter being materialized.
        tensor: String,
        /// Runtime-computed source-plus-output bound.
        required: u64,
        /// Bound recorded in [`super::MemoryEstimate`].
        planned: u64,
    },
}

impl CheckpointError {
    /// Wraps an I/O error with its operation and path.
    pub(crate) fn io(operation: &'static str, path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            operation,
            path: path.into(),
            source,
        }
    }
}
