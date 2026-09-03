//! SafeTensors dtype metadata and checked storage-size arithmetic.

use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

use super::{CheckpointError, Result};

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

    /// Computes the exact byte length of a tensor with `elements` scalars.
    ///
    /// SafeTensors does not permit a trailing partial byte. For example, a tensor
    /// containing an odd number of four-bit values is malformed rather than being
    /// rounded up to the next byte.
    pub fn storage_bytes(self, elements: u64) -> Result<u64> {
        let bits =
            elements
                .checked_mul(self.bits())
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "computing tensor storage bits",
                })?;
        if bits % 8 != 0 {
            return Err(CheckpointError::MisalignedTensorStorage {
                dtype: self,
                elements,
            });
        }
        Ok(bits / 8)
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
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for TensorDType {
    type Err = CheckpointError;

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
mod tests {
    use super::*;

    #[test]
    fn every_dtype_round_trips_through_its_wire_name() {
        let all = [
            TensorDType::BOOL,
            TensorDType::F4,
            TensorDType::F6_E2M3,
            TensorDType::F6_E3M2,
            TensorDType::U8,
            TensorDType::I8,
            TensorDType::F8_E5M2,
            TensorDType::F8_E4M3,
            TensorDType::F8_E8M0,
            TensorDType::I16,
            TensorDType::U16,
            TensorDType::F16,
            TensorDType::BF16,
            TensorDType::I32,
            TensorDType::U32,
            TensorDType::F32,
            TensorDType::C64,
            TensorDType::F64,
            TensorDType::I64,
            TensorDType::U64,
        ];
        for dtype in all {
            assert_eq!(dtype.as_str().parse::<TensorDType>().unwrap(), dtype);
        }
    }

    #[test]
    fn sub_byte_tensors_must_end_on_a_byte_boundary() {
        assert_eq!(TensorDType::F4.storage_bytes(2).unwrap(), 1);
        assert!(matches!(
            TensorDType::F4.storage_bytes(1),
            Err(CheckpointError::MisalignedTensorStorage { .. })
        ));
        assert_eq!(TensorDType::F6_E2M3.storage_bytes(4).unwrap(), 3);
    }

    #[test]
    fn storage_size_overflow_is_reported() {
        assert!(matches!(
            TensorDType::F64.storage_bytes(u64::MAX),
            Err(CheckpointError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn unknown_dtype_is_not_guessed() {
        assert!(matches!(
            "FP16".parse::<TensorDType>(),
            Err(CheckpointError::UnsupportedDType { .. })
        ));
    }
}
