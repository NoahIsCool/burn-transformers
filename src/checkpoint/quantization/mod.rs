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
//! one-tensor-at-a-time traversal without staging the whole model. The resulting
//! packed representation deliberately has no Burn or Metal execution path yet:
//! callers must dequantize through the reference implementation or persist it
//! for a future real packed kernel.

mod config;
mod error;
mod manifest;
mod policy;
mod reference;

pub use config::{
    PackedValueLayout, QuantizationGroupSize, QuantizedGroupLayout, QuantizedPayloadLayout,
    ScaleLayout, SymmetricQuantization, WeightQuantizationConfig,
};
pub use error::{QuantizationError, Result};
pub use manifest::{
    BURNPACK_QUANTIZATION_FORMAT_VERSION, BurnpackByteOrder, BurnpackQuantizationManifest,
    PackedExecutionSupport, QuantizedTensorMetadata, SourceFloatDType,
};
pub use policy::{
    QuantizationEligibility, QuantizationExclusion, classify_weight_for_quantization,
};
pub use reference::{
    QuantizedWeight, ReferenceQuantizationStats, WeightQuantizationOutcome, quantize_tensor_data,
    quantize_weight,
};

#[cfg(test)]
mod tests;
