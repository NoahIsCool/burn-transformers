//! SafeTensors dtype metadata and checked storage-size arithmetic.
//!
//! Storage validation is independent of backend execution support. A logical
//! scalar count `N` requires exactly `N * bits / 8` bytes, with overflow and
//! partial-byte tails rejected. This layer preserves the source dtype spelling;
//! numerical conversion and backend capability checks happen during execution.

use std::{fmt, str::FromStr};

use super::{CheckpointError, Result, layout::StorageSizeError};

mod scalar;

pub use scalar::TensorDType;

impl TensorDType {
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
    /// This composes the checked left-to-right shape product with
    /// [`Self::storage_bytes`]; a rank-zero shape is one scalar, and a shape
    /// whose running product overflows fails even if a later dimension is zero,
    /// as in the pinned SafeTensors validator.
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
