//! Arithmetic boundaries for tensor geometry shared by cache implementations.

use super::*;

/// Integer histories and repeated tensors report their exact validated scalar counts.
#[test]
fn validated_counts_cover_integer_histories_scalars_and_empty_allocations() {
    let elements = validate_tensor_shape("token history", &[2, 17], DType::I64)
        .expect("small integer history");
    assert_eq!(elements, 34);
    assert_eq!(
        validate_repeated_allocation("token and position histories", elements, 2, DType::I64)
            .expect("two histories"),
        68
    );
    assert_eq!(
        validate_tensor_shape("scalar", &[], DType::F32).expect("scalar shape"),
        1
    );
    assert_eq!(
        validate_tensor_shape("empty", &[3, 0], DType::F32).expect("empty shape"),
        0
    );
    assert_eq!(
        validate_repeated_allocation("disabled", 34, 0, DType::I64).expect("no allocations"),
        0
    );
}

/// Element-product overflow and portable backend limits fail before any allocation occurs.
#[test]
fn tensor_geometry_reports_product_overflow_and_portable_limit() {
    assert!(matches!(
        validate_tensor_shape("overflow", &[usize::MAX, 2], DType::F32),
        Err(TensorCacheError::TensorShapeOverflow { tensor: "overflow" })
    ));
    assert!(matches!(
        validate_tensor_shape(
            "too large",
            &[PORTABLE_TENSOR_ELEMENT_LIMIT + 1],
            DType::I32
        ),
        Err(TensorCacheError::TensorElementLimitExceeded {
            tensor: "too large",
            ..
        })
    ));
}

/// Repeated counts distinguish scalar-count overflow from byte-count overflow.
#[test]
fn repeated_allocations_report_element_and_byte_overflow_separately() {
    assert!(matches!(
        validate_repeated_allocation("count overflow", usize::MAX, 2, DType::F32),
        Err(TensorCacheError::TensorShapeOverflow {
            tensor: "count overflow"
        })
    ));
    assert!(matches!(
        validate_repeated_allocation("byte overflow", usize::MAX, 1, DType::I64),
        Err(TensorCacheError::AllocationSizeOverflow {
            tensor: "byte overflow",
            elements: usize::MAX,
            element_size: 8,
        })
    ));
}
