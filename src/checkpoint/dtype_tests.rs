//! Tests exact SafeTensors dtype names and checked wire-storage sizes.

use super::*;

#[test]
/// Every supported dtype round-trips through its exact wire spelling.
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
/// Sub-byte storage rejects element counts that leave a partial byte.
fn sub_byte_tensors_must_end_on_a_byte_boundary() {
    assert_eq!(TensorDType::F4.storage_bytes(2).unwrap(), 1);
    assert!(matches!(
        TensorDType::F4.storage_bytes(1),
        Err(CheckpointError::MisalignedTensorStorage { .. })
    ));
    assert_eq!(TensorDType::F6_E2M3.storage_bytes(4).unwrap(), 3);
}

#[test]
/// Overflowing dtype storage calculations return arithmetic errors.
fn storage_size_overflow_is_reported() {
    assert!(matches!(
        TensorDType::F64.storage_bytes(u64::MAX),
        Err(CheckpointError::ArithmeticOverflow { .. })
    ));
}

#[test]
/// Unknown dtype spellings fail instead of being guessed.
fn unknown_dtype_is_not_guessed() {
    assert!(matches!(
        "FP16".parse::<TensorDType>(),
        Err(CheckpointError::UnsupportedDType { .. })
    ));
}

/// Every dtype in the SafeTensors vocabulary, for exhaustive predicate tests.
const ALL: [TensorDType; 20] = [
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

/// Only the four IEEE-style formats are castable, and all of them are floats.
#[test]
fn castable_floats_are_exactly_the_ieee_style_formats() {
    let castable = ALL
        .into_iter()
        .filter(|dtype| dtype.is_castable_float())
        .collect::<Vec<_>>();
    assert_eq!(
        castable,
        [
            TensorDType::F16,
            TensorDType::BF16,
            TensorDType::F32,
            TensorDType::F64
        ]
    );
    assert!(castable.iter().all(|dtype| dtype.is_float()));
}

/// In-place loading covers equal dtypes and exactly the BF16 <-> F16 casts.
#[test]
fn in_place_conversion_requires_equal_dtypes_or_a_same_width_float_cast() {
    for source in ALL {
        for target in ALL {
            let expected = source == target
                || matches!(
                    (source, target),
                    (TensorDType::BF16, TensorDType::F16) | (TensorDType::F16, TensorDType::BF16)
                );
            assert_eq!(
                source.converts_in_place_to(target),
                expected,
                "{source}->{target}"
            );
        }
    }
}

/// Tensor storage composes the checked shape product with the per-dtype size.
#[test]
fn tensor_storage_bytes_composes_shape_and_dtype_checks() {
    assert_eq!(TensorDType::BF16.tensor_storage_bytes(&[]).unwrap(), 2);
    assert_eq!(TensorDType::F32.tensor_storage_bytes(&[2, 3]).unwrap(), 24);
    assert_eq!(
        TensorDType::F64
            .tensor_storage_bytes(&[0, u64::MAX])
            .unwrap(),
        0
    );
    assert!(matches!(
        TensorDType::U8.tensor_storage_bytes(&[u64::MAX, 2]),
        Err(CheckpointError::ArithmeticOverflow {
            context: "computing tensor element count"
        })
    ));
    assert!(matches!(
        TensorDType::F4.tensor_storage_bytes(&[3]),
        Err(CheckpointError::MisalignedTensorStorage { elements: 3, .. })
    ));
}
