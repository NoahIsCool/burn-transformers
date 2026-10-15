//! Deterministic fixtures shared by the family-glue unit tests.

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor},
};

use crate::{Error, architecture::TokenMask};

/// CPU Flex backend with F32 activations and portable I32 indices.
pub(super) type TestBackend = Flex<f32, i32>;

/// Asserts that two errors have the same variant and the same rendered message.
pub(super) fn assert_same_error(actual: Error, expected: Error) {
    assert_eq!(
        std::mem::discriminant(&actual),
        std::mem::discriminant(&expected),
        "{actual} != {expected}"
    );
    assert_eq!(actual.to_string(), expected.to_string());
}

/// Builds a `[rows, columns]` integer tensor on the default device.
pub(super) fn ints<const R: usize, const C: usize>(
    values: [[i64; C]; R],
) -> Tensor<TestBackend, 2, Int> {
    Tensor::from_data(values, &Default::default())
}

/// Builds a token mask (`true == valid`) from host booleans.
pub(super) fn mask<const R: usize, const C: usize>(
    values: [[bool; C]; R],
) -> TokenMask<TestBackend> {
    TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_data(
        values,
        &Default::default(),
    ))
}

/// Builds a token mask from a row-major host vector.
pub(super) fn mask_from_rows(rows: &[Vec<bool>]) -> TokenMask<TestBackend> {
    let batch = rows.len();
    let sequence = rows.first().map_or(0, Vec::len);
    let values = rows.iter().flatten().copied().collect::<Vec<_>>();
    TokenMask::new(crate::architecture::bool_tensor_from_values(
        values,
        [batch, sequence],
        &Default::default(),
    ))
}

/// Reads a float tensor into row-major host values.
pub(super) fn floats<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("f32 tensor data")
}

/// Reads an integer tensor into row-major host values.
pub(super) fn int_values<const D: usize>(tensor: Tensor<TestBackend, D, Int>) -> Vec<i64> {
    tensor.into_data().iter::<i64>().collect()
}
