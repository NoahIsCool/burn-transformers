//! Serializable metadata for versioned Burnpack quantization formats.

use std::collections::BTreeMap;

use burn::tensor::DType;
use serde::{Deserialize, Serialize};

use super::{
    BurnpackProvenance, PackedValueLayout, QuantizationError, QuantizationGroupSize,
    QuantizedGroupLayout, QuantizedPayloadLayout, Result, ScaleLayout, SymmetricQuantization,
    WeightQuantizationConfig, provenance::validate_sha256,
};

/// Legacy Burnpack quantization format without checkpoint content provenance.
pub const BURNPACK_QUANTIZATION_FORMAT_VERSION_V1: u32 = 1;

/// Burnpack format carrying exact SHA-256 checkpoint and load-plan provenance.
pub const BURNPACK_QUANTIZATION_FORMAT_VERSION_V2: u32 = 2;

/// Current architecture-aware Burnpack quantization format written after binding.
pub const BURNPACK_QUANTIZATION_FORMAT_VERSION_CURRENT: u32 =
    BURNPACK_QUANTIZATION_FORMAT_VERSION_V2;

/// Compatibility name for the original version-one wire format.
///
/// New code must select [`BURNPACK_QUANTIZATION_FORMAT_VERSION_V1`] when it
/// deliberately handles an unbound legacy artifact, or
/// [`BURNPACK_QUANTIZATION_FORMAT_VERSION_CURRENT`] when it needs the newest
/// architecture-aware format. Keeping this historical name equal to one avoids
/// silently changing the meaning of existing downstream version checks.
pub const BURNPACK_QUANTIZATION_FORMAT_VERSION: u32 = BURNPACK_QUANTIZATION_FORMAT_VERSION_V1;

/// Byte order used by explicitly byte-encoded scalar fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BurnpackByteOrder {
    /// Least-significant byte is written first.
    LittleEndian,
}

/// Availability of direct backend execution for every tensor in an artifact.
///
/// The discriminator is persisted only after the writer has checked all tensor
/// layouts. Readers must still verify their selected backend/device before
/// constructing parameters: this field describes format compatibility, not the
/// hardware available to the current process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedExecutionSupport {
    /// No native packed operator is implemented on any backend.
    Unavailable,
    /// The payload maps exactly to Burn block-QFloat and packed matmul dispatch.
    BurnQFloatMatmul,
}

impl PackedExecutionSupport {
    /// Returns whether a backend can execute this packed representation directly.
    #[must_use]
    pub const fn is_available(self) -> bool {
        matches!(self, Self::BurnQFloatMatmul)
    }

    /// Requires a format that can enter Burn's packed matmul dispatch.
    pub fn require_available(self) -> Result<()> {
        if self.is_available() {
            Ok(())
        } else {
            Err(QuantizationError::PackedExecutionUnavailable)
        }
    }
}

/// Floating-point representation presented to the quantizer.
///
/// Architecture-aware conversion records the post-transform value. For example,
/// a BF16 checkpoint loaded through `Cast(F16)` is recorded as F16. Legacy
/// low-level writers record the dtype of the `TensorData` they receive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFloatDType {
    /// IEEE-754 half precision.
    F16,
    /// Brain floating point half precision.
    Bf16,
    /// IEEE-754 single precision.
    F32,
    /// Burn's single-precision storage with flexible compute precision.
    Flex32,
    /// IEEE-754 double precision, rounded to `f32` before quantization.
    F64,
}

impl SourceFloatDType {
    /// Returns the number of bytes occupied by one unpacked source value.
    #[must_use]
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 | Self::Flex32 => 4,
            Self::F64 => 8,
        }
    }

    /// Returns the corresponding Burn tensor dtype.
    #[must_use]
    pub const fn burn_dtype(self) -> DType {
        match self {
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
            Self::F32 => DType::F32,
            Self::Flex32 => DType::Flex32,
            Self::F64 => DType::F64,
        }
    }
}

impl TryFrom<DType> for SourceFloatDType {
    type Error = QuantizationError;

    fn try_from(dtype: DType) -> Result<Self> {
        match dtype {
            DType::F16 => Ok(Self::F16),
            DType::BF16 => Ok(Self::Bf16),
            DType::F32 => Ok(Self::F32),
            DType::Flex32 => Ok(Self::Flex32),
            DType::F64 => Ok(Self::F64),
            _ => Err(QuantizationError::UnsupportedInputDType { dtype }),
        }
    }
}

/// Complete, self-validating layout description for one packed tensor.
///
/// Group `g` owns flattened elements `[g * G, min((g + 1) * G, N))` and scale
/// `g`. `packed_value_bytes` is `N` for Q8 and `ceil(N / 2)` for Q4;
/// `scale_bytes` is exactly `4 * ceil(N / G)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizedTensorMetadata {
    /// Logical row-major tensor shape.
    pub shape: Vec<u64>,
    /// Floating-point dtype of the post-transform quantizer input.
    pub source_dtype: SourceFloatDType,
    /// Symmetric quantization precision.
    pub scheme: SymmetricQuantization,
    /// Number of flattened values sharing each scale.
    pub group_size: QuantizationGroupSize,
    /// Rule used to map logical dimensions into the group stream.
    pub group_layout: QuantizedGroupLayout,
    /// Bit-level value packing rule.
    pub packed_value_layout: PackedValueLayout,
    /// Scale encoding and ordering rule.
    pub scale_layout: ScaleLayout,
    /// Ordering of packed-value and scale regions in a tensor payload.
    pub payload_layout: QuantizedPayloadLayout,
    /// Product of all logical dimensions.
    pub num_elements: u64,
    /// Number of scales and contiguous flattened groups.
    pub group_count: u64,
    /// Exact byte length of the packed-value region.
    pub packed_value_bytes: u64,
    /// Exact byte length of the scale region.
    pub scale_bytes: u64,
    /// SHA-256 over the exact packed-values-then-little-endian-scales payload.
    ///
    /// Version-one artifacts omit this field. Version two requires it for every
    /// tensor so the complete sidecar can be checked before model mutation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_sha256: Option<String>,
}

impl QuantizedTensorMetadata {
    /// Derives canonical metadata from a source shape, dtype, and configuration.
    pub fn new(
        shape: &[usize],
        source_dtype: SourceFloatDType,
        config: WeightQuantizationConfig,
    ) -> Result<Self> {
        let mut shape_wire = Vec::new();
        shape_wire.try_reserve_exact(shape.len()).map_err(|_| {
            QuantizationError::AllocationFailed {
                context: "quantized tensor shape",
                requested: shape.len(),
            }
        })?;
        for &dimension in shape {
            shape_wire.push(
                u64::try_from(dimension)
                    .map_err(|_| QuantizationError::ShapeDimensionOverflow { dimension })?,
            );
        }

        let num_elements = checked_num_elements_u64(&shape_wire)?;
        let group_size = u64::try_from(config.group_size.get()).map_err(|_| {
            QuantizationError::ByteLengthOverflow {
                context: "group size conversion",
            }
        })?;
        let group_count = ceil_div_u64(num_elements, group_size);
        let packed_value_bytes = match config.scheme {
            SymmetricQuantization::Q8 => num_elements,
            SymmetricQuantization::Q4 => ceil_div_u64(num_elements, 2),
        };
        let scale_bytes =
            group_count
                .checked_mul(4)
                .ok_or(QuantizationError::ByteLengthOverflow {
                    context: "scale payload length",
                })?;

        Ok(Self {
            shape: shape_wire,
            source_dtype,
            scheme: config.scheme,
            group_size: config.group_size,
            group_layout: QuantizedGroupLayout::FlatRowMajorContiguous,
            packed_value_layout: packed_layout(config.scheme),
            scale_layout: ScaleLayout::F32LittleEndianPerGroup,
            payload_layout: QuantizedPayloadLayout::PackedValuesThenScales,
            num_elements,
            group_count,
            packed_value_bytes,
            scale_bytes,
            payload_sha256: None,
        })
    }

    /// Validates all redundant counts and fixed layout discriminants.
    pub fn validate(&self) -> Result<()> {
        let expected_elements = checked_num_elements_u64(&self.shape)?;
        if self.num_elements != expected_elements {
            return invalid_metadata(format!(
                "num_elements {} does not match shape product {expected_elements}",
                self.num_elements
            ));
        }

        let group_size = u64::try_from(self.group_size.get()).map_err(|_| {
            QuantizationError::ByteLengthOverflow {
                context: "group size conversion",
            }
        })?;
        let expected_groups = ceil_div_u64(expected_elements, group_size);
        if self.group_count != expected_groups {
            return invalid_metadata(format!(
                "group_count {} does not match expected {expected_groups}",
                self.group_count
            ));
        }

        let expected_packed = match self.scheme {
            SymmetricQuantization::Q8 => expected_elements,
            SymmetricQuantization::Q4 => ceil_div_u64(expected_elements, 2),
        };
        if self.packed_value_bytes != expected_packed {
            return invalid_metadata(format!(
                "packed_value_bytes {} does not match expected {expected_packed}",
                self.packed_value_bytes
            ));
        }

        let expected_scale_bytes =
            expected_groups
                .checked_mul(4)
                .ok_or(QuantizationError::ByteLengthOverflow {
                    context: "scale payload length",
                })?;
        if self.scale_bytes != expected_scale_bytes {
            return invalid_metadata(format!(
                "scale_bytes {} does not match expected {expected_scale_bytes}",
                self.scale_bytes
            ));
        }

        if self.group_layout != QuantizedGroupLayout::FlatRowMajorContiguous {
            return invalid_metadata("unsupported group layout");
        }
        if self.packed_value_layout != packed_layout(self.scheme) {
            return invalid_metadata("packed-value layout does not match scheme");
        }
        if self.scale_layout != ScaleLayout::F32LittleEndianPerGroup {
            return invalid_metadata("unsupported scale layout");
        }
        if self.payload_layout != QuantizedPayloadLayout::PackedValuesThenScales {
            return invalid_metadata("unsupported payload layout");
        }
        if let Some(digest) = &self.payload_sha256 {
            validate_sha256(digest, "packed tensor payload")?;
        }
        Ok(())
    }

    /// Converts the logical shape into host dimensions after validation.
    pub fn host_shape(&self) -> Result<Vec<usize>> {
        self.validate()?;
        let mut shape = Vec::new();
        shape.try_reserve_exact(self.shape.len()).map_err(|_| {
            QuantizationError::AllocationFailed {
                context: "host tensor shape",
                requested: self.shape.len(),
            }
        })?;
        for &dimension in &self.shape {
            shape.push(
                usize::try_from(dimension)
                    .map_err(|_| QuantizationError::PlatformDimensionOverflow { dimension })?,
            );
        }
        Ok(shape)
    }

    /// Verifies that flattened groups are also exact Burn rectangular blocks.
    ///
    /// Both supported artifact versions group the row-major stream contiguously. For a
    /// matrix `[output, input]`, that stream is representable as Burn blocks
    /// `[1, group_size]` exactly when every input row ends on a group boundary.
    /// Packed-U32 storage additionally requires the input width to contain a
    /// whole number of four Q8 or eight Q4 values.
    pub fn validate_burn_qfloat_layout(&self) -> Result<()> {
        let shape = self.host_shape()?;
        let [rows, columns] = shape.as_slice() else {
            return Err(QuantizationError::BurnQFloatIncompatible {
                reason: format!("expected a rank-two matrix, found shape {shape:?}"),
            });
        };
        let group_size = self.group_size.get();
        if !columns.is_multiple_of(group_size) {
            return Err(QuantizationError::BurnQFloatIncompatible {
                reason: format!(
                    "innermost dimension {columns} is not divisible by group size {group_size}"
                ),
            });
        }
        let values_per_u32 = match self.scheme {
            SymmetricQuantization::Q8 => 4,
            SymmetricQuantization::Q4 => 8,
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
        let actual_scales = usize::try_from(self.group_count).map_err(|_| {
            QuantizationError::ByteLengthOverflow {
                context: "Burn QFloat scale count conversion",
            }
        })?;
        if actual_scales != expected_scales {
            return Err(QuantizationError::ScaleCountMismatch {
                expected: expected_scales,
                actual: actual_scales,
            });
        }
        Ok(())
    }

    pub(crate) fn host_counts(&self) -> Result<(usize, usize, usize)> {
        self.validate()?;
        let elements = usize::try_from(self.num_elements).map_err(|_| {
            QuantizationError::ByteLengthOverflow {
                context: "host element count conversion",
            }
        })?;
        let groups = usize::try_from(self.group_count).map_err(|_| {
            QuantizationError::ByteLengthOverflow {
                context: "host group count conversion",
            }
        })?;
        let packed = usize::try_from(self.packed_value_bytes).map_err(|_| {
            QuantizationError::ByteLengthOverflow {
                context: "host packed length conversion",
            }
        })?;
        Ok((elements, groups, packed))
    }
}

/// Deterministic manifest for all quantized tensors in a Burnpack artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BurnpackQuantizationManifest {
    /// Schema and wire-layout version.
    pub format_version: u32,
    /// Byte order used for scale payloads.
    pub byte_order: BurnpackByteOrder,
    /// Direct backend execution availability.
    pub packed_execution: PackedExecutionSupport,
    /// Exact checkpoint and architecture content binding, required by version two.
    ///
    /// Version-one artifacts omit this field and remain readable, but cannot
    /// content-bind source-payload bypass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<BurnpackProvenance>,
    /// Tensor metadata keyed by architecture-owned parameter name.
    pub tensors: BTreeMap<String, QuantizedTensorMetadata>,
}

impl BurnpackQuantizationManifest {
    /// Creates the legacy, provenance-free version-one manifest used by the
    /// original public API.
    ///
    /// Architecture-aware writers should use [`QuantizedBurnpackSession`](super::QuantizedBurnpackSession),
    /// whose checkpoint conversion path binds provenance and publishes v2.
    #[must_use]
    pub fn new() -> Self {
        Self::new_legacy_v1()
    }

    /// Creates an explicitly legacy, provenance-free version-one manifest.
    ///
    /// This constructor is intentionally named: the current writer format is v2,
    /// which cannot be valid until checkpoint provenance and every payload digest
    /// have been bound. Architecture-aware callers should use
    /// [`QuantizedBurnpackSession`](super::QuantizedBurnpackSession), which stages
    /// this legacy form privately and promotes it atomically during binding.
    #[must_use]
    pub fn new_legacy_v1() -> Self {
        Self {
            format_version: BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
            byte_order: BurnpackByteOrder::LittleEndian,
            packed_execution: PackedExecutionSupport::Unavailable,
            provenance: None,
            tensors: BTreeMap::new(),
        }
    }

    /// Promotes a legacy in-progress manifest to v2 with validated provenance.
    pub fn bind_provenance(&mut self, provenance: BurnpackProvenance) -> Result<()> {
        provenance.validate()?;
        let mut candidate = self.clone();
        candidate.format_version = BURNPACK_QUANTIZATION_FORMAT_VERSION_CURRENT;
        candidate.provenance = Some(provenance);
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }

    /// Inserts one tensor, rejecting empty and duplicate architecture keys.
    pub fn insert(
        &mut self,
        name: impl Into<String>,
        metadata: QuantizedTensorMetadata,
    ) -> Result<()> {
        let name = name.into();
        if name.is_empty() {
            return Err(QuantizationError::EmptyTensorName);
        }
        metadata.validate()?;
        if self.tensors.contains_key(&name) {
            return Err(QuantizationError::DuplicateManifestTensor { name });
        }
        self.tensors.insert(name, metadata);
        Ok(())
    }

    /// Validates the schema version and every tensor's redundant layout fields.
    pub fn validate(&self) -> Result<()> {
        match (self.format_version, &self.provenance) {
            (BURNPACK_QUANTIZATION_FORMAT_VERSION_V1, None) => {}
            (BURNPACK_QUANTIZATION_FORMAT_VERSION_V1, Some(_)) => {
                return Err(QuantizationError::InvalidProvenance {
                    reason: "format version one must not claim v2 provenance".into(),
                });
            }
            (BURNPACK_QUANTIZATION_FORMAT_VERSION_V2, Some(provenance)) => {
                provenance.validate()?;
            }
            (BURNPACK_QUANTIZATION_FORMAT_VERSION_V2, None) => {
                return Err(QuantizationError::InvalidProvenance {
                    reason: "format version two requires checkpoint and load-plan provenance"
                        .into(),
                });
            }
            (version, _) => {
                return Err(QuantizationError::UnsupportedFormatVersion { version });
            }
        }
        for (name, metadata) in &self.tensors {
            if name.is_empty() {
                return Err(QuantizationError::EmptyTensorName);
            }
            metadata.validate()?;
            match self.format_version {
                BURNPACK_QUANTIZATION_FORMAT_VERSION_V1 if metadata.payload_sha256.is_some() => {
                    return Err(QuantizationError::InvalidProvenance {
                        reason: format!(
                            "format version one tensor `{name}` must not claim a v2 payload digest"
                        ),
                    });
                }
                BURNPACK_QUANTIZATION_FORMAT_VERSION_V2 if metadata.payload_sha256.is_none() => {
                    return Err(QuantizationError::InvalidProvenance {
                        reason: format!(
                            "format version two tensor `{name}` is missing its payload SHA-256"
                        ),
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Serializes validated metadata as deterministic pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String> {
        self.validate()?;
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Parses JSON and validates every derived count before returning it.
    pub fn from_json(json: &str) -> Result<Self> {
        let manifest: Self = serde_json::from_str(json)?;
        manifest.validate()?;
        Ok(manifest)
    }
}

/// Backward-compatible default for callers compiled against the version-one API.
///
/// The default is intentionally equivalent to [`Self::new_legacy_v1`]. New
/// architecture-aware artifacts do not use this implementation: they become v2
/// only after a [`BurnpackProvenance`] binding validates their checkpoint, plan,
/// and payload digests.
impl Default for BurnpackQuantizationManifest {
    fn default() -> Self {
        Self::new_legacy_v1()
    }
}

fn packed_layout(scheme: SymmetricQuantization) -> PackedValueLayout {
    match scheme {
        SymmetricQuantization::Q8 => PackedValueLayout::SignedI8TwosComplement,
        SymmetricQuantization::Q4 => PackedValueLayout::SignedI4TwosComplementLowNibbleFirst,
    }
}

fn checked_num_elements_u64(shape: &[u64]) -> Result<u64> {
    shape.iter().try_fold(1_u64, |elements, &dimension| {
        elements
            .checked_mul(dimension)
            .ok_or(QuantizationError::ShapeElementOverflow)
    })
}

const fn ceil_div_u64(value: u64, divisor: u64) -> u64 {
    value / divisor + if value.is_multiple_of(divisor) { 0 } else { 1 }
}

fn invalid_metadata<T>(reason: impl Into<String>) -> Result<T> {
    Err(QuantizationError::InvalidMetadata {
        reason: reason.into(),
    })
}
