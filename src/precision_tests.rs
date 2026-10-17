//! Evaluation-dtype rule tests covering every Burn 0.21 dtype family.

use burn::{
    backend::Flex,
    tensor::{BoolStore, DType, Tensor, TensorData, quantization::QuantScheme},
};

use super::{at_least_f32, promote};
use crate::precision::test_support::{assert_bitwise_eq, widened_values};

/// Reduced-precision floating activations promote to F32 and F32 stays F32.
#[test]
fn reduced_precision_and_single_precision_evaluate_in_f32() {
    for dtype in [DType::F16, DType::BF16, DType::Flex32, DType::F32] {
        assert_eq!(at_least_f32(dtype), DType::F32, "{dtype:?}");
    }
}

/// Double-precision activations are never narrowed.
#[test]
fn double_precision_is_preserved() {
    assert_eq!(at_least_f32(DType::F64), DType::F64);
}

/// Non-floating tags keep the historical unconditional F32 promotion.
#[test]
fn non_floating_tags_keep_historical_f32_promotion() {
    for dtype in [
        DType::I64,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
        DType::Bool(BoolStore::Native),
        DType::Bool(BoolStore::U8),
        DType::Bool(BoolStore::U32),
        DType::QFloat(QuantScheme::default()),
    ] {
        assert_eq!(at_least_f32(dtype), DType::F32, "{dtype:?}");
    }
}

/// The rule is usable in constant contexts, so policies can be fixed at compile time.
#[test]
fn rule_is_const_evaluable() {
    const PROMOTED: DType = at_least_f32(DType::F16);
    const PRESERVED: DType = at_least_f32(DType::F64);
    assert_eq!(PROMOTED, DType::F32);
    assert_eq!(PRESERVED, DType::F64);
}

/// Promotion widens reduced precision exactly and leaves F32 and F64 storage untouched.
#[test]
fn promotion_widens_reduced_precision_and_preserves_full_precision() {
    let values = vec![0.1, -2.5, 3.0e-3, 1.0 / 3.0];
    let tensor = |dtype: DType| {
        Tensor::<Flex, 1>::from_data(
            TensorData::new(values.clone(), [4]),
            (&Default::default(), DType::F64),
        )
        .cast(dtype)
    };
    for dtype in [DType::F16, DType::BF16] {
        let original = tensor(dtype).into_data();
        let promoted = promote(tensor(dtype)).into_data();
        assert_eq!(promoted.dtype, DType::F32, "{dtype:?}");
        assert_eq!(
            widened_values(&promoted),
            widened_values(&original),
            "{dtype:?}"
        );
    }
    for dtype in [DType::F32, DType::F64] {
        assert_bitwise_eq(
            &promote(tensor(dtype)).into_data(),
            &tensor(dtype).into_data(),
            &format!("{dtype:?}"),
        );
    }
}
