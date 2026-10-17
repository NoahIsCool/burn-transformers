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

/// Code validation names the first defect in index order.
#[test]
fn code_violations_name_the_first_defect() {
    assert_eq!(first_code_violation(CodeWidth::Byte, 2, &[1, 0xff]), None);
    assert_eq!(
        first_code_violation(CodeWidth::Byte, 3, &[1, 0x80, 2]),
        Some(CodeViolation::Reserved {
            index: 1,
            value: -128
        })
    );
    assert_eq!(
        first_code_violation(CodeWidth::Byte, 3, &[1, 2]),
        Some(CodeViolation::Short)
    );
    // A reserved code before the end of the buffer is reported before shortness.
    assert_eq!(
        first_code_violation(CodeWidth::Byte, 3, &[0x80]),
        Some(CodeViolation::Reserved {
            index: 0,
            value: -128
        })
    );
    assert_eq!(
        first_code_violation(CodeWidth::Nibble, 3, &[0x21, 0x03]),
        None
    );
    assert_eq!(
        first_code_violation(CodeWidth::Nibble, 3, &[0x21, 0x13]),
        Some(CodeViolation::Padding)
    );
    assert_eq!(
        first_code_violation(CodeWidth::Nibble, 2, &[0x81]),
        Some(CodeViolation::Reserved {
            index: 1,
            value: -8
        })
    );
    assert_eq!(
        first_code_violation(CodeWidth::Nibble, 3, &[0x21]),
        Some(CodeViolation::Short)
    );
    // Even Q4 streams and Q8 streams have no padding nibble to check.
    assert_eq!(
        first_code_violation(CodeWidth::Nibble, 2, &[0x21, 0xf0]),
        None
    );
    assert_eq!(first_code_violation(CodeWidth::Nibble, 0, &[]), None);
}

proptest! {
    /// Pushing codes one at a time reads back as the pushed sequence.
    #[test]
    fn pushed_codes_read_back_in_order(codes in proptest::collection::vec(-7_i8..=7, 0..100)) {
        for width in [CodeWidth::Byte, CodeWidth::Nibble] {
            let mut packed = Vec::new();
            for (index, &code) in codes.iter().enumerate() {
                push_code(width, &mut packed, index, code);
                prop_assert_eq!(packed.len(), packed_len(width, index + 1));
            }
            for (index, &code) in codes.iter().enumerate() {
                prop_assert_eq!(read_code(width, &packed, index), Some(code));
            }
            prop_assert_eq!(first_code_violation(width, codes.len(), &packed), None);
        }
    }

    /// Code validation agrees with a direct scan of the original rules.
    #[test]
    fn code_violations_match_a_direct_scan(
        packed in proptest::collection::vec(any::<u8>(), 0..12),
        elements in 0_usize..26,
        nibble in any::<bool>(),
    ) {
        let width = if nibble { CodeWidth::Nibble } else { CodeWidth::Byte };
        let mut expected = None;
        for index in 0..elements {
            match read_code(width, &packed, index) {
                None => {
                    expected = Some(CodeViolation::Short);
                    break;
                }
                Some(value) if value == reserved_code(width) => {
                    expected = Some(CodeViolation::Reserved { index, value });
                    break;
                }
                Some(_) => {}
            }
        }
        if expected.is_none()
            && width == CodeWidth::Nibble
            && elements % 2 == 1
            && packed.last().is_some_and(|&byte| split_nibbles(byte).1 != 0)
        {
            expected = Some(CodeViolation::Padding);
        }
        prop_assert_eq!(first_code_violation(width, elements, &packed), expected);
    }

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
