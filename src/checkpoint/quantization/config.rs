//! Quantizer configuration and exact storage-layout vocabulary.
//!
//! Contiguous row-major groups share one F32 scale, with symmetric integer codes
//! `[-127, 127]` for Q8 or `[-7, 7]` for Q4. The unused most-negative code is
//! reserved for corruption detection. Groups of 32, 64, or 128 values trade scale
//! overhead against quantization error; their precise packing and scale byte order
//! are persisted so loading cannot guess a backend-dependent layout.

use serde::{Deserialize, Deserializer, Serialize, de};

use super::{QuantizationError, Result};

/// Symmetric integer precision used for a weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymmetricQuantization {
    /// One signed byte per value, with usable codes `[-127, 127]`.
    Q8,
    /// One signed four-bit value per element, with usable codes `[-7, 7]`.
    Q4,
}

impl SymmetricQuantization {
    /// Returns the largest-magnitude integer code used by the symmetric grid.
    #[must_use]
    pub const fn qmax(self) -> i8 {
        match self {
            Self::Q8 => 127,
            Self::Q4 => 7,
        }
    }

    /// Computes the packed bytes required for `elements` values.
    ///
    /// This is `n` for Q8 and `ceil(n / 2)` for Q4; the Verus-checked
    /// `packing::packed_len` proves the formula never overflows.
    pub(crate) fn packed_bytes(self, elements: usize) -> Result<usize> {
        let width = match self {
            Self::Q8 => super::packing::CodeWidth::Byte,
            Self::Q4 => super::packing::CodeWidth::Nibble,
        };
        Ok(super::packing::packed_len(width, elements))
    }
}

/// Number of flattened values sharing one symmetric scale.
///
/// Restricting this parameter to a small vocabulary makes persisted artifacts
/// predictable for future kernels while retaining useful accuracy/size tradeoffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct QuantizationGroupSize(
    /// Validated group size: exactly 32, 64, or 128 scalar values.
    usize,
);

impl QuantizationGroupSize {
    /// Largest supported group size; bounds per-group stack buffers.
    pub const MAX: usize = 128;

    /// Creates a supported group size.
    pub fn new(value: usize) -> Result<Self> {
        match value {
            32 | 64 | 128 => Ok(Self(value)),
            _ => Err(QuantizationError::InvalidGroupSize { value }),
        }
    }

    /// Returns the number of contiguous flattened values in a full group.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl Default for QuantizationGroupSize {
    /// Selects 64 contiguous values per scale as the default storage group.
    fn default() -> Self {
        Self(64)
    }
}

impl<'de> Deserialize<'de> for QuantizationGroupSize {
    /// Deserializes through the constructor so unsupported group sizes cannot bypass validation.
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = usize::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// Ordering of source values before groups are formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizedGroupLayout {
    /// Flatten the logical tensor in row-major order, then take contiguous groups.
    FlatRowMajorContiguous,
}

/// Bit-level ordering of signed quantized values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedValueLayout {
    /// One two's-complement `i8` code in each byte.
    SignedI8TwosComplement,
    /// Two four-bit two's-complement codes, even index in the low nibble.
    SignedI4TwosComplementLowNibbleFirst,
}

/// Encoding and ordering of per-group scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleLayout {
    /// One IEEE-754 `f32` little-endian scale for every contiguous group.
    F32LittleEndianPerGroup,
}

/// Ordering of the two byte regions in a serialized tensor payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizedPayloadLayout {
    /// Packed values first, immediately followed by little-endian group scales.
    PackedValuesThenScales,
}

/// Parameters for deterministic reference weight quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WeightQuantizationConfig {
    /// Quantized precision and integer grid.
    pub scheme: SymmetricQuantization,
    /// Number of contiguous source values sharing a scale.
    pub group_size: QuantizationGroupSize,
}

impl WeightQuantizationConfig {
    /// Stable report label of the persisted storage format.
    ///
    /// Conversion records `symmetric-{q8|q4}-g{G}-flat-row-major`: contiguous
    /// row-major groups of `G` values in the sidecar. Loading records the
    /// runtime form of the same payload,
    /// [`QuantizedTensorMetadata::runtime_label`](super::QuantizedTensorMetadata::runtime_label),
    /// because the installed tensor is Burn block-QFloat; for aligned matrices
    /// the two describe identical groups.
    #[must_use]
    pub fn storage_label(self) -> String {
        format!(
            "symmetric-{}-g{}-flat-row-major",
            self.scheme.label(),
            self.group_size.get()
        )
    }
}

impl SymmetricQuantization {
    /// Lowercase precision spelling used in report labels: `q8` or `q4`.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }
}

impl Default for WeightQuantizationConfig {
    /// Selects symmetric Q8 with 64-value groups for the default accuracy/storage tradeoff.
    fn default() -> Self {
        Self {
            scheme: SymmetricQuantization::Q8,
            group_size: QuantizationGroupSize::default(),
        }
    }
}
