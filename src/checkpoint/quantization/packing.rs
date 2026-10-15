//! Pure index and value arithmetic for packed Q8/Q4 codes.
//!
//! Q8 stores code `i` as one two's-complement byte at byte `i`. Q4 stores two
//! codes per byte: code `i` occupies the low nibble of byte `i / 2` when `i` is
//! even and the high nibble when it is odd, as four-bit two's complement. A
//! tensor of `n` codes therefore occupies `n` Q8 bytes or `ceil(n / 2)` Q4 bytes,
//! and the unused high nibble of an odd-length Q4 tensor is zero.
//!
//! On a little-endian `u32` word view, Q4 code `i` sits in word `i / 8` at bits
//! `4 * (i % 8)` and Q8 code `i` in word `i / 4` at bits `8 * (i % 4)`, which is
//! the order Burn's packed QFloat storage unpacks.
//!
//! Symmetric quantization uses only codes in `[-q_max, q_max]`; the most negative
//! code (`-128` for Q8, `-8` for Q4) is reserved so corruption is detectable.
//! Nibble encoding is written arithmetically (`c + 16` for negative `c`) rather
//! than with casts and masks, which states the two's-complement map directly and
//! lets Verus check it in place through `verus/checkpoint.rs` without bit-vector
//! reasoning. The module has no crate or Burn dependencies for that reason.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: the Q8 code stored as two's-complement byte `byte` in `[0, 255]`.
pub open spec fn byte_code(byte: int) -> int {
    if byte < 128 { byte } else { byte - 256 }
}

/// Specification: the Q4 code stored as two's-complement nibble `nibble` in `[0, 15]`.
pub open spec fn nibble_code(nibble: int) -> int {
    if nibble < 8 { nibble } else { nibble - 16 }
}

/// Specification: Q4 code `index` of `packed`, the low nibble of byte `index / 2`
/// for even `index` and its high nibble for odd `index`.
pub open spec fn packed_nibble_code(packed: Seq<u8>, index: int) -> int {
    let byte = packed[index / 2] as int;
    nibble_code(if index % 2 == 0 { byte % 16 } else { byte / 16 })
}

} // verus!

/// Packed storage width of one quantized code.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodeWidth {
    /// One code per byte, usable range `[-127, 127]`.
    Byte,
    /// Two codes per byte, usable range `[-7, 7]`.
    Nibble,
}

/// Returns the bytes needed for `elements` codes: `n` (Q8) or `ceil(n / 2)` (Q4).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bytes =>
    ensures
        width == CodeWidth::Byte ==> bytes == elements,
        width == CodeWidth::Nibble ==> 2 * bytes == elements + elements % 2,
        bytes <= elements,
))]
pub(crate) fn packed_len(width: CodeWidth, elements: usize) -> usize {
    match width {
        CodeWidth::Byte => elements,
        CodeWidth::Nibble => elements / 2 + elements % 2,
    }
}

/// Encodes a code in `[-8, 7]` as its four-bit two's-complement nibble `[0, 15]`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(nibble =>
    requires
        -8 <= code <= 7,
    ensures
        nibble <= 15,
        code >= 0 ==> nibble == code,
        code < 0 ==> nibble == code + 16,
))]
pub(crate) fn encode_nibble(code: i8) -> u8 {
    if code < 0 {
        (code + 16) as u8
    } else {
        code as u8
    }
}

/// Decodes a four-bit two's-complement nibble `[0, 15]` to a code in `[-8, 7]`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(code =>
    requires
        nibble <= 15,
    ensures
        -8 <= code <= 7,
        nibble < 8 ==> code == nibble,
        nibble >= 8 ==> code == nibble - 16,
))]
pub(crate) fn decode_nibble(nibble: u8) -> i8 {
    if nibble >= 8 {
        nibble as i8 - 16
    } else {
        nibble as i8
    }
}

/// Encodes a Q8 code as its two's-complement byte.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(byte =>
    ensures
        code >= 0 ==> byte == code,
        code < 0 ==> byte == code + 256,
))]
pub(crate) fn encode_byte(code: i8) -> u8 {
    if code < 0 {
        (code as i16 + 256) as u8
    } else {
        code as u8
    }
}

/// Decodes a two's-complement byte to its Q8 code.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(code =>
    ensures
        byte < 128 ==> code == byte,
        byte >= 128 ==> code == byte - 256,
))]
pub(crate) fn decode_byte(byte: u8) -> i8 {
    if byte >= 128 {
        (byte as i16 - 256) as i8
    } else {
        byte as i8
    }
}

/// Combines a low and a high Q4 nibble into one packed byte.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(byte =>
    requires
        low <= 15,
        high <= 15,
    ensures
        byte == low + 16 * high,
        byte % 16 == low,
        byte / 16 == high,
))]
pub(crate) fn join_nibbles(low: u8, high: u8) -> u8 {
    low + 16 * high
}

/// Splits a packed byte into its `(low, high)` Q4 nibbles.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(nibbles =>
    ensures
        nibbles.0 == byte % 16,
        nibbles.1 == byte / 16,
        nibbles.0 <= 15,
        nibbles.1 <= 15,
        byte == nibbles.0 + 16 * nibbles.1,
))]
pub(crate) fn split_nibbles(byte: u8) -> (u8, u8) {
    (byte % 16, byte / 16)
}

/// Reads code `index` from a packed buffer, or `None` past its end.
///
/// The postcondition pins the decoded value, not only its range: Q8 code `i` is
/// byte `i`, and Q4 code `i` is the low nibble of byte `i / 2` for even `i` and
/// the high nibble for odd `i`, so swapping the nibble order fails verification.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(code =>
    ensures
        match code {
            Some(value) => match width {
                CodeWidth::Byte => index < packed@.len()
                    && value as int == byte_code(packed@[index as int] as int),
                CodeWidth::Nibble => index / 2 < packed@.len()
                    && -8 <= value <= 7
                    && value as int == packed_nibble_code(packed@, index as int),
            },
            None => match width {
                CodeWidth::Byte => index >= packed@.len(),
                CodeWidth::Nibble => index / 2 >= packed@.len(),
            },
        }
))]
pub(crate) fn read_code(width: CodeWidth, packed: &[u8], index: usize) -> Option<i8> {
    match width {
        CodeWidth::Byte => {
            if index < packed.len() {
                Some(decode_byte(packed[index]))
            } else {
                None
            }
        }
        CodeWidth::Nibble => {
            let byte = index / 2;
            if byte < packed.len() {
                let (low, high) = split_nibbles(packed[byte]);
                Some(decode_nibble(if index % 2 == 1 { high } else { low }))
            } else {
                None
            }
        }
    }
}

/// Returns the reserved most-negative code for a width: `-128` (Q8) or `-8` (Q4).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(code =>
    ensures
        width == CodeWidth::Byte ==> code == -128,
        width == CodeWidth::Nibble ==> code == -8,
))]
pub(crate) fn reserved_code(width: CodeWidth) -> i8 {
    match width {
        CodeWidth::Byte => -128,
        CodeWidth::Nibble => -8,
    }
}

#[cfg(test)]
#[path = "packing_tests.rs"]
mod tests;
