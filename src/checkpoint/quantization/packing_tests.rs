//! Exhaustive and property checks for packed Q8/Q4 code arithmetic.

use proptest::prelude::*;

use super::*;

/// Every Q4 code round-trips and matches the two's-complement mask encoding.
#[test]
fn nibble_encoding_is_four_bit_twos_complement() {
    for code in -8_i8..=7 {
        let nibble = encode_nibble(code);
        assert_eq!(nibble, (code as u8) & 0x0f, "code {code}");
        assert_eq!(decode_nibble(nibble), code);
    }
    for nibble in 0_u8..=15 {
        // The previous shift-based sign extension, as an independent oracle.
        assert_eq!(decode_nibble(nibble), ((nibble as i8) << 4) >> 4);
    }
}

/// Every byte decodes like an `i8` reinterpretation and every code round-trips.
#[test]
fn byte_encoding_is_eight_bit_twos_complement() {
    for code in i8::MIN..=i8::MAX {
        assert_eq!(encode_byte(code), code as u8);
        assert_eq!(decode_byte(encode_byte(code)), code);
    }
    for byte in u8::MIN..=u8::MAX {
        assert_eq!(decode_byte(byte), byte as i8);
    }
}

/// Packed bytes split into the nibbles they were joined from.
#[test]
fn nibble_pairs_round_trip_through_bytes() {
    for low in 0_u8..=15 {
        for high in 0_u8..=15 {
            let byte = join_nibbles(low, high);
            assert_eq!(byte, low | (high << 4));
            assert_eq!(split_nibbles(byte), (low, high));
        }
    }
}

/// Packed lengths and reserved codes match the format definition.
#[test]
fn packed_lengths_and_reserved_codes_match_the_format() {
    for elements in [0, 1, 2, 3, 63, 64, usize::MAX] {
        assert_eq!(packed_len(CodeWidth::Byte, elements), elements);
        assert_eq!(
            packed_len(CodeWidth::Nibble, elements),
            elements.div_ceil(2)
        );
    }
    assert_eq!(reserved_code(CodeWidth::Byte), -128);
    assert_eq!(reserved_code(CodeWidth::Nibble), -8);
    assert_eq!(read_code(CodeWidth::Byte, &[1, 2], 2), None);
    assert_eq!(read_code(CodeWidth::Nibble, &[0x21], 2), None);
}

proptest! {
    /// Codes packed in order read back exactly, with a zero unused Q4 high nibble.
    #[test]
    fn packed_codes_read_back_in_order(codes in proptest::collection::vec(-7_i8..=7, 0..100)) {
        let mut q4 = Vec::new();
        for (index, &code) in codes.iter().enumerate() {
            let nibble = encode_nibble(code);
            if index % 2 == 0 {
                q4.push(join_nibbles(nibble, 0));
            } else {
                let last = q4.last_mut().unwrap();
                *last = join_nibbles(split_nibbles(*last).0, nibble);
            }
        }
        prop_assert_eq!(q4.len(), packed_len(CodeWidth::Nibble, codes.len()));
        let q8 = codes.iter().map(|&code| encode_byte(code)).collect::<Vec<_>>();
        for (index, &code) in codes.iter().enumerate() {
            prop_assert_eq!(read_code(CodeWidth::Nibble, &q4, index), Some(code));
            prop_assert_eq!(read_code(CodeWidth::Byte, &q8, index), Some(code));
            // Little-endian u32 word view used by Burn's packed QFloat storage.
            let word = |bytes: &[u8], word: usize| {
                let mut array = [0_u8; 4];
                for (slot, byte) in array.iter_mut().zip(bytes.iter().skip(4 * word)) {
                    *slot = *byte;
                }
                u32::from_le_bytes(array)
            };
            let q4_bits = (word(&q4, index / 8) >> (4 * (index % 8))) & 0x0f;
            prop_assert_eq!(decode_nibble(q4_bits as u8), code);
            let q8_bits = (word(&q8, index / 4) >> (8 * (index % 4))) & 0xff;
            prop_assert_eq!(decode_byte(q8_bits as u8), code);
        }
        if codes.len() % 2 == 1 {
            prop_assert_eq!(split_nibbles(*q4.last().unwrap()).1, 0);
        }
    }
}
