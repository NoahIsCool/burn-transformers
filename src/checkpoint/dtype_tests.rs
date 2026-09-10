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
