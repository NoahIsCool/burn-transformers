//! Errors for validated quantization and Burnpack metadata handling.

use std::path::PathBuf;

use burn::tensor::DType;
use thiserror::Error;

/// Result type used by the native checkpoint quantizer.
pub type Result<T> = std::result::Result<T, QuantizationError>;

/// A failure while validating, quantizing, or decoding a packed weight.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum QuantizationError {
    /// A group size outside the fixed format vocabulary was requested.
    #[error("invalid quantization group size {value}; expected 32, 64, or 128")]
    InvalidGroupSize {
        /// Rejected number of values per group.
        value: usize,
    },
    /// The input is not an unpacked floating-point tensor.
    #[error(
        "unsupported quantization input dtype {dtype:?}; expected f16, bf16, f32, flex32, or f64"
    )]
    UnsupportedInputDType {
        /// Rejected Burn dtype.
        dtype: DType,
    },
    /// Multiplying the dimensions of a source shape overflowed `usize`.
    #[error("tensor element count overflows the host address space")]
    ShapeElementOverflow,
    /// A shape dimension cannot be represented by the manifest's `u64` field.
    #[error("shape dimension {dimension} cannot be represented in Burnpack metadata")]
    ShapeDimensionOverflow {
        /// Source dimension that failed conversion.
        dimension: usize,
    },
    /// A manifest dimension cannot be represented on this host.
    #[error("manifest shape dimension {dimension} cannot be represented on this host")]
    PlatformDimensionOverflow {
        /// Manifest dimension that failed conversion.
        dimension: u64,
    },
    /// Computing a byte count overflowed.
    #[error("integer overflow while computing {context}")]
    ByteLengthOverflow {
        /// Size calculation that overflowed.
        context: &'static str,
    },
    /// TensorData contained too few or too many source bytes.
    #[error("tensor payload has {actual} bytes but shape and dtype require {expected}")]
    SourceByteLengthMismatch {
        /// Required source payload length.
        expected: usize,
        /// Observed source payload length.
        actual: usize,
    },
    /// A source value was NaN or infinite.
    #[error("source tensor contains a non-finite value at flattened index {index}")]
    NonFiniteValue {
        /// Flattened row-major element index.
        index: usize,
    },
    /// A fallible output allocation could not be reserved.
    #[error("unable to reserve {requested} bytes/elements for {context}")]
    AllocationFailed {
        /// Buffer being allocated.
        context: &'static str,
        /// Requested capacity in buffer units.
        requested: usize,
    },
    /// Serialized metadata violated a derived layout invariant.
    #[error("invalid quantized tensor metadata: {reason}")]
    InvalidMetadata {
        /// Stable explanation of the violated invariant.
        reason: String,
    },
    /// Packed bytes disagreed with metadata.
    #[error("packed value payload has {actual} bytes but metadata requires {expected}")]
    PackedLengthMismatch {
        /// Required packed bytes.
        expected: usize,
        /// Observed packed bytes.
        actual: usize,
    },
    /// Scale count disagreed with metadata.
    #[error("scale payload has {actual} values but metadata requires {expected}")]
    ScaleCountMismatch {
        /// Required number of scales.
        expected: usize,
        /// Observed number of scales.
        actual: usize,
    },
    /// A scale was zero, negative, NaN, or infinite.
    #[error("group {group} has invalid scale {scale}")]
    InvalidScale {
        /// Flattened group index.
        group: usize,
        /// Invalid scale value.
        scale: f32,
    },
    /// A packed field used the deliberately reserved negative code.
    #[error("packed element {index} contains reserved quantized value {value}")]
    InvalidQuantizedValue {
        /// Flattened element index.
        index: usize,
        /// Rejected signed quantized code.
        value: i8,
    },
    /// The flat persisted grouping cannot be represented exactly by Burn QFloat blocks.
    #[error("quantized weight is incompatible with Burn's packed QFloat layout: {reason}")]
    BurnQFloatIncompatible {
        /// Exact rank, alignment, or grouping mismatch.
        reason: String,
    },
    /// The manifest format version is not implemented by this crate.
    #[error("unsupported Burnpack quantization format version {version}")]
    UnsupportedFormatVersion {
        /// Rejected version number.
        version: u32,
    },
    /// A manifest tensor key was empty.
    #[error("Burnpack tensor names must not be empty")]
    EmptyTensorName,
    /// A tensor was inserted twice into one manifest.
    #[error("Burnpack manifest already contains tensor `{name}`")]
    DuplicateManifestTensor {
        /// Repeated tensor name.
        name: String,
    },
    /// A session resource limit was zero or exceeded Burnpack's public bounds.
    #[error("invalid Burnpack session limit `{field}`={value}: {reason}")]
    InvalidSessionLimit {
        /// Configuration field that was rejected.
        field: &'static str,
        /// Rejected value.
        value: u64,
        /// Stable explanation of the supported range.
        reason: &'static str,
    },
    /// The output path cannot identify an exact `.bpk` file.
    #[error("invalid Burnpack output path `{path}`: {reason}")]
    InvalidArtifactPath {
        /// Rejected output path.
        path: PathBuf,
        /// Path invariant that was violated.
        reason: &'static str,
    },
    /// A filesystem operation failed while staging or publishing an artifact.
    #[error("failed to {operation} `{path}`: {source}")]
    ArtifactIo {
        /// Human-readable operation being attempted.
        operation: &'static str,
        /// Path involved in the operation.
        path: PathBuf,
        /// Operating-system failure.
        #[source]
        source: std::io::Error,
    },
    /// The official Burn Store implementation rejected an artifact operation.
    #[error("Burn Store failed to {operation}: {message}")]
    BurnStore {
        /// Burnpack operation being attempted.
        operation: &'static str,
        /// Upstream diagnostic.
        message: String,
    },
    /// A logical model key collides with the manifest's reserved tensor name.
    #[error("tensor name `{name}` is reserved for the Burnpack quantization manifest")]
    ReservedTensorCollision {
        /// Colliding model tensor name.
        name: String,
    },
    /// A mutation was attempted after successful artifact publication.
    #[error("Burnpack quantization session has already been finalized")]
    SessionFinalized,
    /// The staging file length changed outside the session's append protocol.
    #[error("quantized staging file has {actual} bytes but session expected {expected}")]
    StagingCorrupted {
        /// Length derived from retained offsets.
        expected: u64,
        /// Observed file length.
        actual: u64,
    },
    /// A bounded staging or metadata resource would exceed its configured limit.
    #[error("Burnpack {kind} size/count {attempted} exceeds configured limit {limit}")]
    SessionLimitExceeded {
        /// Resource being bounded.
        kind: &'static str,
        /// Size or count after the proposed operation.
        attempted: u64,
        /// Configured maximum.
        limit: u64,
    },
    /// Atomic publication refused to replace an existing path.
    #[error("Burnpack output already exists: `{path}`")]
    OutputAlreadyExists {
        /// Existing output path.
        path: PathBuf,
    },
    /// The official artifact did not contain the reserved manifest tensor.
    #[error("Burnpack artifact is missing reserved manifest tensor `{name}`")]
    MissingManifestTensor {
        /// Reserved tensor name expected by this format.
        name: &'static str,
    },
    /// A requested model tensor is not present in the quantization manifest.
    #[error("quantized Burnpack tensor `{name}` was not found")]
    QuantizedTensorNotFound {
        /// Missing architecture-owned name.
        name: String,
    },
    /// Artifact tensor names or physical U8 layouts disagree with the manifest.
    #[error("invalid quantized Burnpack tensor `{name}`: {reason}")]
    InvalidArtifactTensor {
        /// Tensor whose physical representation was invalid.
        name: String,
        /// Violated physical or set invariant.
        reason: String,
    },
    /// UTF-8 decoding failed for the reserved JSON manifest tensor.
    #[error("Burnpack quantization manifest tensor is not UTF-8: {0}")]
    ManifestUtf8(#[from] std::str::Utf8Error),
    /// JSON encoding or decoding failed.
    #[error("invalid Burnpack quantization manifest JSON: {0}")]
    ManifestJson(#[from] serde_json::Error),
    /// Writing a packed payload failed.
    #[error("failed to write Burnpack quantized payload: {0}")]
    PayloadIo(#[from] std::io::Error),
    /// No backend has a native packed operator for this format yet.
    #[error("native packed quantized execution is unavailable; dequantize to f32 first")]
    PackedExecutionUnavailable,
}
