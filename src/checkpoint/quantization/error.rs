//! Errors for validated quantization and Burnpack metadata handling.

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
