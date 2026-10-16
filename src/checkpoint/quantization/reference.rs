//! Deterministic `f32` reference conversion for packed Burnpack weights.
//!
//! For each contiguous row-major group, compute `s = max(abs(x)) / qmax` and
//! `q = clamp(round(x / s), -qmax, qmax)`, then reconstruct `x_hat = s * q`.
//! Arithmetic uses F32, including narrowed F64 inputs; half formats widen before
//! the maximum reduction. Ties round away from zero. Zero groups use `s = 1`,
//! and a subnormal fallback prevents nonzero groups from acquiring a zero scale.
//!
//! Each group is decoded once into a stack buffer of at most 128 `f32` values,
//! which avoids a whole-tensor decoded copy while reading every source byte only
//! once. Q4 uses low-nibble-first signed codes, zeroes an unused final nibble,
//! and excludes the asymmetric minimum code (see the `packing` module). Persisted
//! scales must be positive and finite. Native packed execution additionally
//! requires row/group alignment; representable storage alone does not establish
//! backend or autodiff support.

use std::io::Write;

use burn::tensor::{
    DType, TensorData, bf16, f16,
    quantization::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue},
};

use super::{
    QuantizationEligibility, QuantizationError, QuantizationExclusion, QuantizationGroupSize,
    QuantizedTensorMetadata, Result, SourceFloatDType, SymmetricQuantization,
    WeightQuantizationConfig, classify_weight_for_quantization,
    counts::{block_scale_count, ceil_div_usize},
    packing::{CodeViolation, CodeWidth, first_code_violation, push_code, read_code},
    provenance::quantized_payload_sha256,
};

/// Bytes of little-endian scales encoded per staging write or digest update.
///
/// Encoding scales through one reusable 64 KiB buffer keeps memory bounded while
/// turning one 4-byte `write(2)` per scale into one write per 16,384 scales.
pub(super) const SCALE_CHUNK_BYTES: usize = 64 * 1024;

// Validated packed ownership and storage/backend conversion boundaries.

/// Bounded-work statistics produced by the reference converter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceQuantizationStats {
    /// Number of source values visited.
    pub source_elements: usize,
    /// Number of independent symmetric scales produced.
    pub group_count: usize,
    /// Largest group visited by the converter, never greater than the configured size.
    pub peak_group_elements: usize,
}

/// Validated packed values and per-group scales for one weight tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedWeight {
    /// Validated logical shape, scheme, source dtype, and derived storage counts.
    metadata: QuantizedTensorMetadata,
    /// Signed quantized codes packed in the format's defined byte/nibble order.
    packed_values: Vec<u8>,
    /// Positive F32 scale for each flattened quantization group.
    scales: Vec<f32>,
    /// Converter work and peak group size associated with this weight.
    stats: ReferenceQuantizationStats,
}

impl QuantizedWeight {
    /// Builds a packed weight from persisted regions after exhaustive validation.
    pub fn from_parts(
        metadata: QuantizedTensorMetadata,
        packed_values: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self> {
        let weight = Self::from_payload_digest_verified_parts(metadata, packed_values, scales)?;
        if let Some(expected) = &weight.metadata.payload_sha256 {
            let actual = quantized_payload_sha256(&weight.packed_values, &weight.scales);
            if actual != *expected {
                return Err(QuantizationError::PackedPayloadDigestMismatch {
                    name: "<in-memory packed weight>".into(),
                    expected: expected.clone(),
                    actual,
                });
            }
        }
        Ok(weight)
    }

    /// Builds a packed weight after its exact owned wire bytes passed their digest check.
    ///
    /// This is the crate's integrity trust boundary: the caller must first copy
    /// both packed values and little-endian scale bytes into owned memory and
    /// compare the SHA-256 of that exact canonical payload with the manifest.
    /// This constructor deliberately repeats every structural and numerical
    /// validation while omitting only that already-completed digest computation.
    pub(super) fn from_payload_digest_verified_parts(
        metadata: QuantizedTensorMetadata,
        packed_values: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self> {
        let (elements, groups, expected_packed) = metadata.host_counts()?;
        if packed_values.len() != expected_packed {
            return Err(QuantizationError::PackedLengthMismatch {
                expected: expected_packed,
                actual: packed_values.len(),
            });
        }
        if scales.len() != groups {
            return Err(QuantizationError::ScaleCountMismatch {
                expected: groups,
                actual: scales.len(),
            });
        }
        validate_scales(&scales)?;
        validate_codes(metadata.scheme, elements, &packed_values)?;
        let peak_group_elements = elements.min(metadata.group_size.get());

        Ok(Self {
            metadata,
            packed_values,
            scales,
            stats: ReferenceQuantizationStats {
                source_elements: elements,
                group_count: groups,
                peak_group_elements,
            },
        })
    }

    /// Returns the complete wire-layout metadata.
    #[must_use]
    pub const fn metadata(&self) -> &QuantizedTensorMetadata {
        &self.metadata
    }

    /// Returns packed Q8 bytes or low-nibble-first Q4 bytes.
    #[must_use]
    pub fn packed_values(&self) -> &[u8] {
        &self.packed_values
    }

    /// Returns one scale per contiguous flattened group.
    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Returns bounded-work accounting for the reference converter.
    #[must_use]
    pub const fn stats(&self) -> ReferenceQuantizationStats {
        self.stats
    }

    /// Splits this value into manifest metadata and its two payload regions.
    #[must_use]
    pub fn into_parts(self) -> (QuantizedTensorMetadata, Vec<u8>, Vec<f32>) {
        (self.metadata, self.packed_values, self.scales)
    }

    /// Writes the canonical tensor payload: packed bytes, then `f32` LE scales.
    ///
    /// Packed codes go out in one write. Scales are encoded through one bounded
    /// 64 KiB buffer, one write per chunk, so the payload is never
    /// assembled as a second combined copy and an unbuffered file sees a handful
    /// of large writes rather than one syscall per scale. The caller owns framing
    /// and tensor offsets in the outer Burnpack.
    pub fn write_burnpack_payload<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.packed_values)?;
        for_each_scale_chunk(&self.scales, |chunk| writer.write_all(chunk))?;
        Ok(())
    }

    /// Dequantizes into a row-major Burn `TensorData` with dtype `f32`.
    ///
    /// Element `i` is reconstructed as `q_i * scales[i / group_size]`.
    /// This reference path intentionally expands the complete tensor and is not
    /// presented as a packed backend kernel. It applies to both v1 payloads and
    /// provenance-bound v2 payloads after their integrity checks succeed.
    pub fn dequantize_f32(&self) -> Result<TensorData> {
        let (elements, _, _) = self.metadata.host_counts()?;
        validate_scales(&self.scales)?;
        validate_codes(self.metadata.scheme, elements, &self.packed_values)?;

        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| QuantizationError::AllocationFailed {
                context: "dequantized f32 tensor",
                requested: elements,
            })?;
        let width = code_width(self.metadata.scheme);
        for index in 0..elements {
            let code = read_code(width, &self.packed_values, index).ok_or(
                QuantizationError::PackedLengthMismatch {
                    expected: elements,
                    actual: self.packed_values.len(),
                },
            )?;
            let scale = self.scales[index / self.metadata.group_size.get()];
            values.push(f32::from(code) * scale);
        }
        let shape = self.metadata.host_shape()?;
        Ok(TensorData::new(values, shape))
    }

    /// Converts a compatible matrix into Burn's native packed `QFloat` data.
    ///
    /// The persisted packed format groups a completely flattened row-major stream,
    /// while Burn describes independent rectangular blocks. Those definitions
    /// are identical for a rank-two `[output, input]` matrix only when `input`
    /// is a multiple of the configured group size. This method enforces that
    /// equivalence rather than silently changing scales at row boundaries.
    ///
    /// Values remain packed and scales remain F32. On CubeCL backends, using
    /// the resulting tensor as a matmul operand allows the packed matmul kernel
    /// to consume it directly. Burn's higher-level `Linear::forward` currently
    /// dequantizes weights, so callers must use a quantization-preserving matmul
    /// dispatch rather than treating this conversion alone as execution proof.
    pub fn into_burn_qfloat_data(self) -> Result<TensorData> {
        self.metadata.validate_burn_qfloat_layout()?;
        let (metadata, mut packed_values, scales) = self.into_parts();
        let shape = metadata.host_shape()?;
        let [rows, columns] = shape.as_slice() else {
            return Err(QuantizationError::BurnQFloatIncompatible {
                reason: format!("expected a rank-two matrix, found shape {shape:?}"),
            });
        };
        let group_size = metadata.group_size.get();
        let (value, values_per_u32) = match metadata.scheme {
            SymmetricQuantization::Q8 => (QuantValue::Q8S, 4),
            SymmetricQuantization::Q4 => (QuantValue::Q4S, 8),
        };
        debug_assert!(columns.is_multiple_of(values_per_u32));
        let expected_scales = block_scale_count(*rows, *columns, group_size)
            .ok_or(QuantizationError::ShapeElementOverflow)?;
        if scales.len() != expected_scales {
            return Err(QuantizationError::ScaleCountMismatch {
                expected: expected_scales,
                actual: scales.len(),
            });
        }

        // Our wire format numbers the first code in the least-significant bits
        // of each u32. Byte reversal preserves that bit numbering on big-endian
        // hosts before CubeCL receives native u32 storage.
        #[cfg(target_endian = "big")]
        for word in packed_values.chunks_exact_mut(core::mem::size_of::<u32>()) {
            word.reverse();
        }
        let scale_bytes = scales
            .len()
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "Burn QFloat scale payload",
            })?;
        packed_values.try_reserve_exact(scale_bytes).map_err(|_| {
            QuantizationError::AllocationFailed {
                context: "Burn QFloat payload",
                requested: scale_bytes,
            }
        })?;
        for scale in scales {
            packed_values.extend_from_slice(&scale.to_ne_bytes());
        }

        let group_size =
            u8::try_from(group_size).map_err(|_| QuantizationError::BurnQFloatIncompatible {
                reason: format!("group size {group_size} does not fit Burn's u8 block extent"),
            })?;
        let scheme = QuantScheme::default()
            .with_value(value)
            .with_param(QuantParam::F32)
            .with_store(QuantStore::PackedU32(0))
            .with_level(QuantLevel::block([1, group_size]))
            .with_mode(QuantMode::Symmetric);
        Ok(TensorData::from_bytes_vec(
            packed_values,
            [*rows, *columns],
            DType::QFloat(scheme),
        ))
    }
}

/// Result of applying the mandatory eligibility policy to one parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum WeightQuantizationOutcome {
    /// Eligible matrix weight converted to the requested packed format.
    Quantized(
        /// Validated codes, scales, metadata, and conversion counters.
        QuantizedWeight,
    ),
    /// Parameter retained in source precision for the stated policy reason.
    Excluded(
        /// Policy reason for retaining source precision.
        QuantizationExclusion,
    ),
}

/// Applies eligibility policy and quantizes one architecture-owned parameter.
///
/// Exclusions are returned as data rather than errors so streaming checkpoint
/// conversion can account for every parameter without exceptional control flow.
pub fn quantize_weight(
    name: &str,
    source: &TensorData,
    config: WeightQuantizationConfig,
) -> Result<WeightQuantizationOutcome> {
    match classify_weight_for_quantization(name, source.shape.as_slice()) {
        QuantizationEligibility::Eligible => Ok(WeightQuantizationOutcome::Quantized(
            quantize_tensor_data(source, config)?,
        )),
        QuantizationEligibility::Excluded(reason) => {
            Ok(WeightQuantizationOutcome::Excluded(reason))
        }
    }
}

/// Quantizes a single unpacked floating-point tensor with bounded temporary work.
///
/// The source is never copied into a whole-tensor `f32` staging buffer. Each
/// group of at most [`QuantizationGroupSize::MAX`] values is decoded once into a
/// stack buffer, which then supplies both the maximum reduction and the codes.
/// Arithmetic, including `f64` input conversion, is intentionally performed in
/// `f32` so results are stable across source dtypes and backends.
pub fn quantize_tensor_data(
    source: &TensorData,
    config: WeightQuantizationConfig,
) -> Result<QuantizedWeight> {
    let source_dtype = SourceFloatDType::try_from(source.dtype)?;
    let elements = checked_num_elements(source.shape.as_slice())?;
    validate_source_bytes(source, source_dtype, elements)?;

    let packed_len = config.scheme.packed_bytes(elements)?;
    let group_size = config.group_size.get();
    let groups = ceil_div_usize(elements, group_size);
    let metadata = QuantizedTensorMetadata::new(source.shape.as_slice(), source_dtype, config)?;

    let mut packed_values = Vec::new();
    packed_values.try_reserve_exact(packed_len).map_err(|_| {
        QuantizationError::AllocationFailed {
            context: "packed quantized values",
            requested: packed_len,
        }
    })?;
    let mut scales = Vec::new();
    scales
        .try_reserve_exact(groups)
        .map_err(|_| QuantizationError::AllocationFailed {
            context: "quantization scales",
            requested: groups,
        })?;

    let width = code_width(config.scheme);
    let mut buffer = [0.0_f32; QuantizationGroupSize::MAX];
    for (group, encoded) in source
        .as_bytes()
        .chunks(group_size * source_dtype.byte_width())
        .enumerate()
    {
        let start = group * group_size;
        let values = &mut buffer[..encoded.len() / source_dtype.byte_width()];
        decode_group(encoded, source_dtype, values);
        let mut max_abs = 0.0_f32;
        for (offset, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(QuantizationError::NonFiniteValue {
                    index: start + offset,
                });
            }
            max_abs = max_abs.max(value.abs());
        }
        let scale = canonical_scale(max_abs, config.scheme);
        scales.push(scale);
        for (offset, &value) in values.iter().enumerate() {
            let code = quantize_value(value, scale, config.scheme);
            push_code(width, &mut packed_values, start + offset, code);
        }
    }

    // Codes are clamped to `[-q_max, q_max]` and scales are positive and finite
    // by construction, so the persisted-payload validation of `from_parts` would
    // only re-scan what this loop just produced.
    debug_assert_eq!(packed_values.len(), packed_len);
    debug_assert_eq!(scales.len(), groups);
    debug_assert!(validate_scales(&scales).is_ok());
    debug_assert!(validate_codes(config.scheme, elements, &packed_values).is_ok());
    Ok(QuantizedWeight {
        metadata,
        packed_values,
        scales,
        stats: ReferenceQuantizationStats {
            source_elements: elements,
            group_count: groups,
            peak_group_elements: elements.min(group_size),
        },
    })
}

/// Decodes one group of native-endian source scalars into `f32`, once.
///
/// `encoded` holds exactly `values.len()` scalars: the caller splits the
/// length-validated source into whole groups. F16/BF16 widen exactly; F64
/// rounds to the nearest F32, and the caller rejects non-finite results.
fn decode_group(encoded: &[u8], dtype: SourceFloatDType, values: &mut [f32]) {
    match dtype {
        SourceFloatDType::F16 => {
            for (value, bytes) in values.iter_mut().zip(encoded.as_chunks::<2>().0) {
                *value = f16::from_bits(u16::from_ne_bytes(*bytes)).to_f32();
            }
        }
        SourceFloatDType::Bf16 => {
            for (value, bytes) in values.iter_mut().zip(encoded.as_chunks::<2>().0) {
                *value = bf16::from_bits(u16::from_ne_bytes(*bytes)).to_f32();
            }
        }
        SourceFloatDType::F32 | SourceFloatDType::Flex32 => {
            for (value, bytes) in values.iter_mut().zip(encoded.as_chunks::<4>().0) {
                *value = f32::from_ne_bytes(*bytes);
            }
        }
        SourceFloatDType::F64 => {
            for (value, bytes) in values.iter_mut().zip(encoded.as_chunks::<8>().0) {
                *value = f64::from_ne_bytes(*bytes) as f32;
            }
        }
    }
}

/// Calls `sink` with consecutive little-endian encodings of at most
/// [`SCALE_CHUNK_BYTES`] bytes of scales, reusing one bounded buffer.
pub(super) fn for_each_scale_chunk<E>(
    scales: &[f32],
    mut sink: impl FnMut(&[u8]) -> std::result::Result<(), E>,
) -> std::result::Result<(), E> {
    /// Scales encoded per chunk: 16,384 four-byte scales fill 64 KiB.
    const SCALES_PER_CHUNK: usize = SCALE_CHUNK_BYTES / core::mem::size_of::<f32>();
    let mut buffer = Vec::with_capacity(scales.len().min(SCALES_PER_CHUNK) * 4);
    for chunk in scales.chunks(SCALES_PER_CHUNK) {
        buffer.clear();
        buffer.extend(chunk.iter().flat_map(|scale| scale.to_le_bytes()));
        sink(&buffer)?;
    }
    Ok(())
}

// Scalar reference arithmetic and canonical wire-code validation.

/// Computes the flattened element count without overflowing the host index type.
fn checked_num_elements(shape: &[usize]) -> Result<usize> {
    crate::checkpoint::layout::checked_host_product(shape)
        .ok_or(QuantizationError::ShapeElementOverflow)
}

/// Requires exact native-endian source byte length before indexed scalar decoding.
fn validate_source_bytes(
    source: &TensorData,
    dtype: SourceFloatDType,
    elements: usize,
) -> Result<()> {
    let expected =
        elements
            .checked_mul(dtype.byte_width())
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "source tensor byte length",
            })?;
    let actual = source.as_bytes().len();
    if actual != expected {
        return Err(QuantizationError::SourceByteLengthMismatch { expected, actual });
    }
    Ok(())
}

/// Computes `max_abs / qmax`, using one for zero groups and a subnormal fallback.
///
/// If division underflows to zero, retaining `max_abs` keeps the scale positive
/// and representable, at the cost of using fewer integer codes for that group.
fn canonical_scale(max_abs: f32, scheme: SymmetricQuantization) -> f32 {
    if max_abs == 0.0 {
        return 1.0;
    }
    let divided = max_abs / f32::from(scheme.qmax());
    if divided == 0.0 { max_abs } else { divided }
}

/// Rounds `value / scale` with ties away from zero and clamps to the symmetric code grid.
fn quantize_value(value: f32, scale: f32, scheme: SymmetricQuantization) -> i8 {
    let qmax = f32::from(scheme.qmax());
    (value / scale).round().clamp(-qmax, qmax) as i8
}

/// Maps a quantization precision to its packed code width.
const fn code_width(scheme: SymmetricQuantization) -> CodeWidth {
    match scheme {
        SymmetricQuantization::Q8 => CodeWidth::Byte,
        SymmetricQuantization::Q4 => CodeWidth::Nibble,
    }
}

/// Requires each persisted F32 scale to be finite and strictly positive.
fn validate_scales(scales: &[f32]) -> Result<()> {
    for (group, &scale) in scales.iter().enumerate() {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(QuantizationError::InvalidScale { group, scale });
        }
    }
    Ok(())
}

/// Rejects asymmetric reserved codes and nonzero unused high-nibble padding.
///
/// The verified [`first_code_violation`] scans the codes in index order and
/// reports the first defect, which maps to the matching contextual error.
fn validate_codes(scheme: SymmetricQuantization, elements: usize, packed: &[u8]) -> Result<()> {
    match first_code_violation(code_width(scheme), elements, packed) {
        None => Ok(()),
        Some(CodeViolation::Short) => Err(QuantizationError::PackedLengthMismatch {
            expected: elements,
            actual: packed.len(),
        }),
        Some(CodeViolation::Reserved { index, value }) => {
            Err(QuantizationError::InvalidQuantizedValue { index, value })
        }
        Some(CodeViolation::Padding) => Err(QuantizationError::InvalidMetadata {
            reason: "unused final Q4 high nibble must be zero".into(),
        }),
    }
}

#[cfg(test)]
#[path = "reference_tests.rs"]
mod tests;
