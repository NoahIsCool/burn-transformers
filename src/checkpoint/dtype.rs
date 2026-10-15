//! SafeTensors dtype metadata and checked storage-size arithmetic.
//!
//! Storage validation is independent of backend execution support. A logical
//! scalar count `N` requires exactly `N * bits / 8` bytes, with overflow and
//! partial-byte tails rejected. This layer preserves the source dtype spelling;
//! numerical conversion and backend capability checks happen during execution.

use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

use super::{CheckpointError, Result, layout::StorageSizeError};

/// A scalar dtype defined by the SafeTensors format.
///
/// Sub-byte formats are represented because discovery must be able to inventory
/// them even when the selected Burn backend cannot execute them. Execution support
/// is a later capability check; silently interpreting an unknown dtype here would
/// make byte-range validation unsound.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TensorDType {
    /// Boolean, stored in eight bits by SafeTensors.
    BOOL,
    /// Four-bit microscaling floating point.
    F4,
    /// Six-bit E2M3 microscaling floating point.
    F6_E2M3,
    /// Six-bit E3M2 microscaling floating point.
    F6_E3M2,
    /// Unsigned eight-bit integer.
    U8,
    /// Signed eight-bit integer.
    I8,
    /// Eight-bit E5M2 floating point.
    F8_E5M2,
    /// Eight-bit E4M3 floating point.
    F8_E4M3,
    /// Eight-bit E8M0 scale value.
    F8_E8M0,
    /// Signed sixteen-bit integer.
    I16,
    /// Unsigned sixteen-bit integer.
    U16,
    /// IEEE binary16.
    F16,
    /// Brain floating point 16.
    BF16,
    /// Signed thirty-two-bit integer.
    I32,
    /// Unsigned thirty-two-bit integer.
    U32,
    /// IEEE binary32.
    F32,
    /// Complex value made from two IEEE binary32 components.
    C64,
    /// IEEE binary64.
    F64,
    /// Signed sixty-four-bit integer.
    I64,
    /// Unsigned sixty-four-bit integer.
    U64,
}

impl TensorDType {
    /// Returns the number of stored bits per scalar.
    #[must_use]
    pub const fn bits(self) -> u64 {
        match self {
            Self::F4 => 4,
            Self::F6_E2M3 | Self::F6_E3M2 => 6,
            Self::BOOL | Self::U8 | Self::I8 | Self::F8_E5M2 | Self::F8_E4M3 | Self::F8_E8M0 => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::C64 | Self::F64 | Self::I64 | Self::U64 => 64,
        }
    }

    /// Returns whether this is a real-valued floating-point representation.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F4
                | Self::F6_E2M3
                | Self::F6_E3M2
                | Self::F8_E5M2
                | Self::F8_E4M3
                | Self::F8_E8M0
                | Self::F16
                | Self::BF16
                | Self::F32
                | Self::F64
        )
    }

    /// Returns whether the loader can numerically cast this dtype to another.
    ///
    /// Casts are implemented, and correctly rounded, only among the conventional
    /// IEEE-style formats F16, BF16, F32, and F64. Microscaling and FP8 formats are
    /// floating point ([`Self::is_float`]) but can only be loaded unchanged. The
    /// planner and the executor share this predicate, so a plan never records a
    /// conversion that execution would reject.
    #[must_use]
    pub const fn is_castable_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    /// Returns whether loading `self` as `target` can overwrite the source buffer.
    ///
    /// Equal dtypes need at most an endianness swap, and BF16 <-> F16 casts keep
    /// the 16-bit scalar width, so each slot can be converted where it is read.
    /// The loader then stages only the source payload for such an operation.
    #[must_use]
    pub const fn converts_in_place_to(self, target: Self) -> bool {
        // Discriminant comparison: derived `==` is not callable in a `const fn`.
        self as u8 == target as u8
            || (self.bits() == target.bits()
                && self.is_castable_float()
                && target.is_castable_float())
    }

    /// Computes the exact byte length of a tensor with `elements` scalars.
    ///
    /// SafeTensors does not permit a trailing partial byte. For example, a tensor
    /// containing an odd number of four-bit values is malformed rather than being
    /// rounded up to the next byte.
    pub fn storage_bytes(self, elements: u64) -> Result<u64> {
        super::layout::storage_bytes(elements, self.bits()).map_err(|error| match error {
            StorageSizeError::Overflow => CheckpointError::ArithmeticOverflow {
                context: "computing tensor storage bits",
            },
            StorageSizeError::PartialByte => CheckpointError::MisalignedTensorStorage {
                dtype: self,
                elements,
            },
        })
    }

    /// Computes the exact byte length of a tensor with the given dimensions.
    ///
    /// This composes the checked shape product with [`Self::storage_bytes`]; a
    /// rank-zero shape is one scalar and any zero dimension gives zero bytes.
    pub fn tensor_storage_bytes(self, shape: &[u64]) -> Result<u64> {
        let elements =
            super::layout::element_count(shape).ok_or(CheckpointError::ArithmeticOverflow {
                context: "computing tensor element count",
            })?;
        self.storage_bytes(elements)
    }

    /// Returns the canonical SafeTensors spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BOOL => "BOOL",
            Self::F4 => "F4",
            Self::F6_E2M3 => "F6_E2M3",
            Self::F6_E3M2 => "F6_E3M2",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::F8_E5M2 => "F8_E5M2",
            Self::F8_E4M3 => "F8_E4M3",
            Self::F8_E8M0 => "F8_E8M0",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::F32 => "F32",
            Self::C64 => "C64",
            Self::F64 => "F64",
            Self::I64 => "I64",
            Self::U64 => "U64",
        }
    }
}

impl fmt::Display for TensorDType {
    /// Formats the canonical SafeTensors dtype name without backend-specific aliases.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for TensorDType {
    /// Diagnostic for a wire spelling outside the supported dtype vocabulary.
    type Err = CheckpointError;

    /// Parses only recognized SafeTensors scalar spellings; unknown types fail closed.
    fn from_str(value: &str) -> Result<Self> {
        let dtype = match value {
            "BOOL" => Self::BOOL,
            "F4" => Self::F4,
            "F6_E2M3" => Self::F6_E2M3,
            "F6_E3M2" => Self::F6_E3M2,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "F8_E5M2" => Self::F8_E5M2,
            "F8_E4M3" => Self::F8_E4M3,
            "F8_E8M0" => Self::F8_E8M0,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "F32" => Self::F32,
            "C64" => Self::C64,
            "F64" => Self::F64,
            "I64" => Self::I64,
            "U64" => Self::U64,
            other => {
                return Err(CheckpointError::UnsupportedDType {
                    dtype: other.to_owned(),
                });
            }
        };
        Ok(dtype)
    }
}

#[cfg(test)]
#[path = "dtype_tests.rs"]
mod tests;
