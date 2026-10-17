//! SafeTensors scalar dtypes, their stored bit widths, and the cast tables.
//!
//! Storage sizing multiplies an element count by [`TensorDType::bits`], and the
//! loader stages a conversion in place only when
//! [`TensorDType::converts_in_place_to`] holds, which must imply equal scalar
//! widths so each converted value overwrites exactly the bytes it was read
//! from. Verus checks these tables in place through `verus/checkpoint.rs`:
//! the widths are exact, the castable formats are exactly the conventional
//! IEEE-style floats, and the only in-place conversions besides identity are
//! the two same-width BF16/F16 casts.
//!
//! Serde derives are compiled only outside Verus, which checks this file
//! without crate dependencies; the wire spellings, storage arithmetic, and
//! parsing live in the parent module.

#[cfg(not(verus_keep_ghost))]
use serde::{Deserialize, Serialize};
#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// A scalar dtype defined by the SafeTensors format.
///
/// Sub-byte formats are represented because discovery must be able to inventory
/// them even when the selected Burn backend cannot execute them. Execution support
/// is a later capability check; silently interpreting an unknown dtype here would
/// make byte-range validation unsound.
#[allow(non_camel_case_types)]
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(not(verus_keep_ghost), derive(Serialize, Deserialize))]
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

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: the SafeTensors stored width of one `dtype` scalar, in bits.
pub open spec fn width_bits(dtype: TensorDType) -> u64 {
    if dtype is F4 {
        4
    } else if dtype is F6_E2M3 || dtype is F6_E3M2 {
        6
    } else if dtype is BOOL || dtype is U8 || dtype is I8 || dtype is F8_E5M2
        || dtype is F8_E4M3 || dtype is F8_E8M0 {
        8
    } else if dtype is I16 || dtype is U16 || dtype is F16 || dtype is BF16 {
        16
    } else if dtype is I32 || dtype is U32 || dtype is F32 {
        32
    } else {
        64
    }
}

/// Specification: `dtype` is a real-valued floating-point representation.
pub open spec fn floating(dtype: TensorDType) -> bool {
    dtype is F4 || dtype is F6_E2M3 || dtype is F6_E3M2 || dtype is F8_E5M2
        || dtype is F8_E4M3 || dtype is F8_E8M0 || dtype is F16 || dtype is BF16
        || dtype is F32 || dtype is F64
}

/// Specification: `dtype` is one of the conventional IEEE-style floats the loader casts.
pub open spec fn castable(dtype: TensorDType) -> bool {
    dtype is F16 || dtype is BF16 || dtype is F32 || dtype is F64
}

/// Specification: `dtype` is one of the two 16-bit castable floats.
pub open spec fn half_width_float(dtype: TensorDType) -> bool {
    dtype is F16 || dtype is BF16
}

} // verus!

impl TensorDType {
    /// Returns the number of stored bits per scalar.
    ///
    /// Verified: exactly the SafeTensors width of each dtype; only F4 and the
    /// two F6 formats are narrower than a byte, every other width is a whole
    /// number of bytes, and C64 is two F32 components.
    #[must_use]
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(bits =>
        ensures
            bits == width_bits(self),
            4 <= bits <= 64,
            bits < 8 <==> (self is F4 || self is F6_E2M3 || self is F6_E3M2),
            bits >= 8 ==> bits % 8 == 0,
            self is C64 ==> bits == 2 * width_bits(TensorDType::F32),
    ))]
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
    ///
    /// Verified: true exactly for the ten floating-point formats.
    #[must_use]
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(float =>
        ensures
            float == floating(self),
    ))]
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
    ///
    /// Verified: true exactly for F16, BF16, F32, and F64, each a floating-point
    /// format.
    #[must_use]
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(castable_float =>
        ensures
            castable_float == castable(self),
            castable_float ==> floating(self),
    ))]
    pub const fn is_castable_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    /// Returns whether loading `self` as `target` can overwrite the source buffer.
    ///
    /// Equal dtypes need at most an endianness swap, and BF16 <-> F16 casts keep
    /// the 16-bit scalar width, so each slot can be converted where it is read.
    /// The loader then stages only the source payload for such an operation.
    ///
    /// Verified: true exactly for identity and the BF16 <-> F16 casts, so an
    /// in-place conversion always keeps the scalar width (each converted value
    /// overwrites exactly the bytes it was read from) and is symmetric.
    #[must_use]
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(in_place =>
        ensures
            in_place == (self == target
                || (half_width_float(self) && half_width_float(target))),
            in_place ==> width_bits(self) == width_bits(target),
            in_place == (self == target || (castable(self) && castable(target)
                && width_bits(self) == width_bits(target))),
    ))]
    pub const fn converts_in_place_to(self, target: Self) -> bool {
        // Discriminant comparison: derived `==` is not callable in a `const fn`.
        self as u8 == target as u8
            || (self.bits() == target.bits()
                && self.is_castable_float()
                && target.is_castable_float())
    }
}
