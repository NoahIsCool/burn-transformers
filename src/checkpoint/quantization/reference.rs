//! Deterministic `f32` reference conversion for packed Burnpack weights.

use std::io::Write;

use burn::tensor::{
    DType, TensorData, bf16, f16,
    quantization::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue},
};

use super::{
    QuantizationEligibility, QuantizationError, QuantizationExclusion, QuantizedTensorMetadata,
    Result, SourceFloatDType, SymmetricQuantization, WeightQuantizationConfig,
    classify_weight_for_quantization,
};

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
    metadata: QuantizedTensorMetadata,
    packed_values: Vec<u8>,
    scales: Vec<f32>,
    stats: ReferenceQuantizationStats,
}

impl QuantizedWeight {
    /// Builds a packed weight from persisted regions after exhaustive validation.
    pub fn from_parts(
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
    /// Scales are encoded individually to avoid allocating a second combined
    /// payload. The caller owns framing and tensor offsets in the outer Burnpack.
    pub fn write_burnpack_payload<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.packed_values)?;
        for scale in &self.scales {
            writer.write_all(&scale.to_le_bytes())?;
        }
        Ok(())
    }

    /// Dequantizes into a row-major Burn `TensorData` with dtype `f32`.
    ///
    /// Element `i` is reconstructed as `q_i * scales[i / group_size]`.
    /// This is the only v1 execution path; it intentionally expands the complete
    /// tensor and is not presented as a packed backend kernel.
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
        for index in 0..elements {
            let code = decode_code(self.metadata.scheme, &self.packed_values, index);
            let scale = self.scales[index / self.metadata.group_size.get()];
            values.push(f32::from(code) * scale);
        }
        let shape = self.metadata.host_shape()?;
        Ok(TensorData::new(values, shape))
    }

    /// Converts a compatible matrix into Burn's native packed `QFloat` data.
    ///
    /// The persisted v1 format groups a completely flattened row-major stream,
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
        let (metadata, mut packed_values, scales) = self.into_parts();
        let shape = metadata.host_shape()?;
        let [rows, columns] = shape.as_slice() else {
            return Err(QuantizationError::BurnQFloatIncompatible {
                reason: format!("expected a rank-two matrix, found shape {shape:?}"),
            });
        };
        let group_size = metadata.group_size.get();
        if !columns.is_multiple_of(group_size) {
            return Err(QuantizationError::BurnQFloatIncompatible {
                reason: format!(
                    "innermost dimension {columns} is not divisible by group size {group_size}"
                ),
            });
        }
        let (value, values_per_u32) = match metadata.scheme {
            SymmetricQuantization::Q8 => (QuantValue::Q8S, 4),
            SymmetricQuantization::Q4 => (QuantValue::Q4S, 8),
        };
        if !columns.is_multiple_of(values_per_u32) {
            return Err(QuantizationError::BurnQFloatIncompatible {
                reason: format!(
                    "innermost dimension {columns} is not divisible by packed-u32 width {values_per_u32}"
                ),
            });
        }
        let expected_scales = rows
            .checked_mul(columns / group_size)
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
    Quantized(QuantizedWeight),
    /// Parameter retained in source precision for the stated policy reason.
    Excluded(QuantizationExclusion),
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
/// The source is never copied into a whole-tensor `f32` staging buffer. Instead,
/// each group is scanned once for its maximum absolute value and once to emit
/// integer codes. Arithmetic, including `f64` input conversion, is intentionally
/// performed in `f32` so results are stable across source dtypes and backends.
pub fn quantize_tensor_data(
    source: &TensorData,
    config: WeightQuantizationConfig,
) -> Result<QuantizedWeight> {
    let source_dtype = SourceFloatDType::try_from(source.dtype)?;
    let elements = checked_num_elements(source.shape.as_slice())?;
    validate_source_bytes(source, source_dtype, elements)?;

    let packed_len = config.scheme.packed_bytes(elements)?;
    let group_size = config.group_size.get();
    let groups = ceil_div(elements, group_size);
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

    for group in 0..groups {
        let start = group
            .checked_mul(group_size)
            .ok_or(QuantizationError::ShapeElementOverflow)?;
        let end = start.saturating_add(group_size).min(elements);
        let mut max_abs = 0.0_f32;
        for index in start..end {
            let value = read_f32(source, source_dtype, index)?;
            if !value.is_finite() {
                return Err(QuantizationError::NonFiniteValue { index });
            }
            max_abs = max_abs.max(value.abs());
        }

        let scale = canonical_scale(max_abs, config.scheme);
        scales.push(scale);
        for index in start..end {
            let value = read_f32(source, source_dtype, index)?;
            let code = quantize_value(value, scale, config.scheme);
            push_code(config.scheme, &mut packed_values, index, code);
        }
    }

    debug_assert_eq!(packed_values.len(), packed_len);
    QuantizedWeight::from_parts(metadata, packed_values, scales).map(|mut weight| {
        weight.stats = ReferenceQuantizationStats {
            source_elements: elements,
            group_count: groups,
            peak_group_elements: elements.min(group_size),
        };
        weight
    })
}

fn checked_num_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |elements, &dimension| {
        elements
            .checked_mul(dimension)
            .ok_or(QuantizationError::ShapeElementOverflow)
    })
}

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

fn read_f32(source: &TensorData, dtype: SourceFloatDType, index: usize) -> Result<f32> {
    let width = dtype.byte_width();
    let offset = index
        .checked_mul(width)
        .ok_or(QuantizationError::ByteLengthOverflow {
            context: "source element offset",
        })?;
    let bytes = source.as_bytes();
    let value = match dtype {
        SourceFloatDType::F16 => {
            let bits = u16::from_ne_bytes([bytes[offset], bytes[offset + 1]]);
            f16::from_bits(bits).to_f32()
        }
        SourceFloatDType::Bf16 => {
            let bits = u16::from_ne_bytes([bytes[offset], bytes[offset + 1]]);
            bf16::from_bits(bits).to_f32()
        }
        SourceFloatDType::F32 | SourceFloatDType::Flex32 => f32::from_ne_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]),
        SourceFloatDType::F64 => f64::from_ne_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as f32,
    };
    Ok(value)
}

fn canonical_scale(max_abs: f32, scheme: SymmetricQuantization) -> f32 {
    if max_abs == 0.0 {
        return 1.0;
    }
    let divided = max_abs / f32::from(scheme.qmax());
    if divided == 0.0 { max_abs } else { divided }
}

fn quantize_value(value: f32, scale: f32, scheme: SymmetricQuantization) -> i8 {
    let qmax = f32::from(scheme.qmax());
    (value / scale).round().clamp(-qmax, qmax) as i8
}

fn push_code(scheme: SymmetricQuantization, output: &mut Vec<u8>, index: usize, code: i8) {
    match scheme {
        SymmetricQuantization::Q8 => output.push(code as u8),
        SymmetricQuantization::Q4 => {
            let nibble = (code as u8) & 0x0f;
            if index.is_multiple_of(2) {
                output.push(nibble);
            } else if let Some(byte) = output.last_mut() {
                *byte |= nibble << 4;
            }
        }
    }
}

fn decode_code(scheme: SymmetricQuantization, packed: &[u8], index: usize) -> i8 {
    match scheme {
        SymmetricQuantization::Q8 => packed[index] as i8,
        SymmetricQuantization::Q4 => {
            let byte = packed[index / 2];
            let nibble = if index.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            };
            ((nibble as i8) << 4) >> 4
        }
    }
}

fn validate_scales(scales: &[f32]) -> Result<()> {
    for (group, &scale) in scales.iter().enumerate() {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(QuantizationError::InvalidScale { group, scale });
        }
    }
    Ok(())
}

fn validate_codes(scheme: SymmetricQuantization, elements: usize, packed: &[u8]) -> Result<()> {
    for index in 0..elements {
        let value = decode_code(scheme, packed, index);
        let reserved = match scheme {
            SymmetricQuantization::Q8 => -128,
            SymmetricQuantization::Q4 => -8,
        };
        if value == reserved {
            return Err(QuantizationError::InvalidQuantizedValue { index, value });
        }
    }
    if scheme == SymmetricQuantization::Q4
        && elements % 2 == 1
        && packed.last().is_some_and(|byte| byte & 0xf0 != 0)
    {
        return Err(QuantizationError::InvalidMetadata {
            reason: "unused final Q4 high nibble must be zero".into(),
        });
    }
    Ok(())
}

const fn ceil_div(value: usize, divisor: usize) -> usize {
    value / divisor + if value.is_multiple_of(divisor) { 0 } else { 1 }
}
