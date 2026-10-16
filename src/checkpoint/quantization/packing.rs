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
//! [`push_code`] appends codes in index order and [`first_code_violation`]
//! rejects a persisted buffer that is short, holds a reserved code, or leaves a
//! nonzero unused Q4 high nibble.
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

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: code `index` of `packed` decoded at `width`.
pub(crate) open spec fn stored_code(width: CodeWidth, packed: Seq<u8>, index: int) -> int {
    match width {
        CodeWidth::Byte => byte_code(packed[index] as int),
        CodeWidth::Nibble => packed_nibble_code(packed, index),
    }
}

/// Specification: code `index` lies inside `packed` at `width`.
pub(crate) open spec fn readable(width: CodeWidth, packed: Seq<u8>, index: int) -> bool {
    match width {
        CodeWidth::Byte => index < packed.len(),
        CodeWidth::Nibble => index / 2 < packed.len(),
    }
}

/// Specification: the packed bytes holding `count` codes, `n` (Q8) or `⌈n / 2⌉` (Q4).
pub(crate) open spec fn packed_bytes(width: CodeWidth, count: int) -> int {
    match width {
        CodeWidth::Byte => count,
        CodeWidth::Nibble => count / 2 + count % 2,
    }
}

/// Specification: codes `0..count` are readable and none is the reserved code.
pub(crate) open spec fn codes_valid(width: CodeWidth, packed: Seq<u8>, count: int) -> bool {
    forall|i: int| 0 <= i < count ==> {
        &&& #[trigger] readable(width, packed, i)
        &&& stored_code(width, packed, i) != reserved_value(width)
    }
}

/// Specification: the reserved most-negative code of `width`.
pub(crate) open spec fn reserved_value(width: CodeWidth) -> int {
    match width {
        CodeWidth::Byte => -128,
        CodeWidth::Nibble => -8,
    }
}

/// Specification: an odd-length Q4 buffer whose unused final high nibble is nonzero.
pub(crate) open spec fn dirty_padding(width: CodeWidth, packed: Seq<u8>, count: int) -> bool {
    &&& width == CodeWidth::Nibble
    &&& count % 2 == 1
    &&& packed.len() > 0
    &&& packed[packed.len() - 1] / 16 != 0
}

} // verus!

/// Appends code `index` to a packed buffer that already holds codes `0..index`.
///
/// Q8 pushes one byte. Q4 starts a new byte (high nibble zero) for an even
/// index and fills the high nibble of the last byte for an odd index; the
/// caller visits contiguous increasing indices, so every odd index completes
/// the byte created by its preceding even index.
///
/// Verified: from a buffer holding exactly codes `0..index` (with a zero
/// unused high nibble after an even count), the result holds exactly codes
/// `0..=index`: every earlier code reads back unchanged, code `index` reads
/// back as `code`, the length is the packed length of `index + 1` codes, and
/// an odd count again leaves the unused high nibble zero. Packing a sequence
/// one code at a time therefore reads back as that sequence.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(
    requires
        index < usize::MAX,
        old(output)@.len() == packed_bytes(width, index as int),
        width == CodeWidth::Nibble ==> -8 <= code <= 7,
        width == CodeWidth::Nibble && index % 2 == 1
            ==> old(output)@[old(output)@.len() - 1] < 16,
    ensures
        final(output)@.len() == packed_bytes(width, index as int + 1),
        forall|j: int| 0 <= j < index ==> {
            &&& #[trigger] readable(width, final(output)@, j)
            &&& stored_code(width, final(output)@, j) == stored_code(width, old(output)@, j)
        },
        readable(width, final(output)@, index as int),
        stored_code(width, final(output)@, index as int) == code,
        width == CodeWidth::Nibble && index % 2 == 0
            ==> final(output)@[final(output)@.len() - 1] < 16,
))]
pub(crate) fn push_code(width: CodeWidth, output: &mut Vec<u8>, index: usize, code: i8) {
    match width {
        CodeWidth::Byte => output.push(encode_byte(code)),
        CodeWidth::Nibble => {
            let nibble = encode_nibble(code);
            // A named parity keeps this Verus-checkable (`is_multiple_of` has no spec).
            let parity = index % 2;
            if parity == 0 {
                output.push(join_nibbles(nibble, 0));
            } else {
                let length = output.len();
                if length > 0 {
                    let (low, _) = split_nibbles(output[length - 1]);
                    output[length - 1] = join_nibbles(low, nibble);
                }
            }
        }
    }
}

/// First defect of a packed code buffer, as found by [`first_code_violation`].
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodeViolation {
    /// The buffer ends before code `elements - 1`.
    Short,
    /// Code `index` is the reserved most-negative code `value`.
    Reserved {
        /// Position of the reserved code.
        index: usize,
        /// The reserved code itself.
        value: i8,
    },
    /// An odd-length Q4 buffer's unused final high nibble is nonzero.
    Padding,
}

/// Scans `elements` packed codes for the first defect, or returns `None` when canonical.
///
/// Codes are checked in index order: an unreadable code (`Short`) or a
/// reserved one stops the scan; only a fully valid code stream is then
/// checked for a nonzero unused Q4 high nibble.
///
/// Verified: `None` exactly when every code `0..elements` is readable and not
/// reserved and the padding is clean; `Short` and `Reserved` name the first
/// failing code with every earlier code valid; `Padding` means every code is
/// valid but the final high nibble is dirty.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(violation =>
    ensures
        violation is None <==> codes_valid(width, packed@, elements as int)
            && !dirty_padding(width, packed@, elements as int),
        match violation {
            None => true,
            Some(CodeViolation::Short) => exists|first: int| 0 <= first < elements
                && codes_valid(width, packed@, first)
                && !readable(width, packed@, first),
            Some(CodeViolation::Reserved { index, value }) => {
                &&& index < elements
                &&& codes_valid(width, packed@, index as int)
                &&& readable(width, packed@, index as int)
                &&& stored_code(width, packed@, index as int) == value
                &&& value == reserved_value(width)
            },
            Some(CodeViolation::Padding) => codes_valid(width, packed@, elements as int)
                && dirty_padding(width, packed@, elements as int),
        }
))]
pub(crate) fn first_code_violation(
    width: CodeWidth,
    elements: usize,
    packed: &[u8],
) -> Option<CodeViolation> {
    let reserved = reserved_code(width);
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= elements,
            reserved == reserved_value(width),
            codes_valid(width, packed@, index as int),
        decreases elements - index,
    ))]
    while index < elements {
        let Some(value) = read_code(width, packed, index) else {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(!readable(width, packed@, index as int));
            }
            return Some(CodeViolation::Short);
        };
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(readable(width, packed@, index as int));
            assert(stored_code(width, packed@, index as int) == value as int);
        }
        if value == reserved {
            return Some(CodeViolation::Reserved { index, value });
        }
        index += 1;
    }
    // Only an odd Q4 stream leaves an unused final high nibble.
    let has_padding = match width {
        CodeWidth::Byte => false,
        CodeWidth::Nibble => elements % 2 == 1,
    };
    let length = packed.len();
    if has_padding && length > 0 {
        let (_, high) = split_nibbles(packed[length - 1]);
        if high != 0 {
            return Some(CodeViolation::Padding);
        }
    }
    None
}

#[cfg(test)]
#[path = "packing_tests.rs"]
mod tests;
