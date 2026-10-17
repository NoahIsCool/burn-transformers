//! Exact-bit and double-precision comparison helpers for precision-rule tests.
//!
//! Bit comparisons establish that an F16/BF16/F32 operation graph is unchanged:
//! two executions agree only when the shape, dtype tag, and every stored IEEE
//! pattern are identical. Double-precision comparisons use a relative error
//! bound against host `f64` references. That bound separates genuine F64
//! evaluation (errors near F64 unit roundoff, about `1.1e-16`) from F32
//! evaluation cast back to F64 (errors near F32 unit roundoff, about `6e-8`).

use burn::tensor::{DType, TensorData, bf16, f16};

/// Relative tolerance for agreement between F64 tensor execution and a host F64 reference.
///
/// It leaves several orders of magnitude for reassociation in small reductions
/// while remaining four orders below the error of any F32 intermediate.
pub(crate) const F64_RELATIVE_TOLERANCE: f64 = 1e-12;

/// Minimum discrepancy that proves an F32 path was not mistaken for F64 evaluation.
///
/// F32 rounding of non-dyadic inputs produces relative errors near `6e-8`; a
/// constructed input whose F32 path misses the F64 reference by more than this
/// bound cannot pass [`F64_RELATIVE_TOLERANCE`] if any intermediate is narrowed.
pub(crate) const F32_DISCREPANCY_FLOOR: f64 = 1e-9;

/// Returns the dtype tag and raw IEEE bit patterns of floating tensor data.
///
/// # Panics
///
/// Panics for non-floating data or storage whose bytes disagree with its tag.
pub(crate) fn exact_bits(data: &TensorData) -> (DType, Vec<u64>) {
    let bits = match data.dtype {
        DType::F64 => data
            .as_slice::<f64>()
            .expect("F64 tensor storage")
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        DType::F32 => data
            .as_slice::<f32>()
            .expect("F32 tensor storage")
            .iter()
            .map(|value| u64::from(value.to_bits()))
            .collect(),
        DType::F16 => data
            .as_slice::<f16>()
            .expect("F16 tensor storage")
            .iter()
            .map(|value| u64::from(value.to_bits()))
            .collect(),
        DType::BF16 => data
            .as_slice::<bf16>()
            .expect("BF16 tensor storage")
            .iter()
            .map(|value| u64::from(value.to_bits()))
            .collect(),
        other => panic!("exact bit comparison requires floating data, found {other:?}"),
    };
    (data.dtype, bits)
}

/// Asserts that two tensors have identical shape, dtype tag, and IEEE bits.
pub(crate) fn assert_bitwise_eq(actual: &TensorData, expected: &TensorData, context: &str) {
    assert_eq!(actual.shape, expected.shape, "{context}: shape");
    assert_eq!(
        exact_bits(actual),
        exact_bits(expected),
        "{context}: dtype or IEEE bits differ"
    );
}

/// Reads F64 tensor storage as host doubles, rejecting every other dtype.
pub(crate) fn f64_values(data: &TensorData) -> Vec<f64> {
    assert_eq!(data.dtype, DType::F64, "expected F64 tensor storage");
    data.as_slice::<f64>().expect("F64 tensor storage").to_vec()
}

/// Widens floating tensor storage of any supported dtype exactly to host doubles.
pub(crate) fn widened_values(data: &TensorData) -> Vec<f64> {
    assert!(
        matches!(
            data.dtype,
            DType::F64 | DType::F32 | DType::F16 | DType::BF16
        ),
        "expected floating tensor storage, found {:?}",
        data.dtype
    );
    data.iter::<f64>().collect()
}

/// Returns `max_i |actual_i - expected_i| / max(|expected_i|, floor)`.
///
/// `floor` bounds the denominator where a reference is near zero, so callers
/// pass the natural output scale. Non-finite values fail instead of being
/// ignored by the maximum.
pub(crate) fn max_relative_error(actual: &[f64], expected: &[f64], floor: f64) -> f64 {
    assert_eq!(actual.len(), expected.len(), "compared value counts");
    assert!(floor > 0.0, "relative-error floor must be positive");
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            assert!(
                actual.is_finite() && expected.is_finite(),
                "non-finite comparison: actual {actual}, expected {expected}"
            );
            (actual - expected).abs() / expected.abs().max(floor)
        })
        .fold(0.0, f64::max)
}

/// Asserts F64 storage that matches a host reference within [`F64_RELATIVE_TOLERANCE`].
pub(crate) fn assert_matches_f64_reference(
    actual: &TensorData,
    expected: &[f64],
    floor: f64,
    context: &str,
) {
    let error = max_relative_error(&f64_values(actual), expected, floor);
    assert!(
        error <= F64_RELATIVE_TOLERANCE,
        "{context}: F64 relative error {error:e} exceeds {F64_RELATIVE_TOLERANCE:e}"
    );
}

/// Asserts that an F32 execution path visibly misses the F64 reference.
///
/// Used with [`assert_matches_f64_reference`], this proves that a constructed
/// input distinguishes double-precision evaluation from any F32 intermediate.
pub(crate) fn assert_f32_path_is_distinguishable(
    f32_path: &TensorData,
    expected: &[f64],
    floor: f64,
    context: &str,
) {
    let error = max_relative_error(&widened_values(f32_path), expected, floor);
    assert!(
        error > F32_DISCREPANCY_FLOOR,
        "{context}: F32 path error {error:e} does not exceed {F32_DISCREPANCY_FLOOR:e}"
    );
}

/// Returns `count` values near 1000 whose offsets are below F32 resolution there.
///
/// F32 spacing near 1000 is `2^-14`, about `6.1e-5`, while neighboring values
/// differ by roughly `1e-5`. As softmax logits, an F32 intermediate rounds the
/// offsets that F64 preserves and changes probabilities by about `1e-5` relative.
pub(crate) fn near_tie_logits(count: usize) -> Vec<f64> {
    (0..count)
        .map(|index| 1000.0 + 1.3e-5 * ((index * 7 % 11) as f64) + 1e-7 * index as f64)
        .collect()
}

/// Host F64 softmax over consecutive rows of `keys` logits.
///
/// Excluded entries and entries equal to negative infinity carry no mass. Rows
/// without any remaining finite logit are all zero; every other entry is
/// `exp(s - max) / sum exp(s - max)` over the remaining logits of its row.
pub(crate) fn host_masked_softmax(
    logits: &[f64],
    excluded: Option<&[bool]>,
    keys: usize,
) -> Vec<f64> {
    assert!(
        keys > 0 && logits.len().is_multiple_of(keys),
        "complete softmax rows"
    );
    logits
        .chunks_exact(keys)
        .enumerate()
        .flat_map(|(row, values)| {
            let supported = |key: usize| {
                values[key].is_finite()
                    && !excluded.is_some_and(|excluded| excluded[row * keys + key])
            };
            let maximum = (0..keys)
                .filter(|&key| supported(key))
                .map(|key| values[key])
                .fold(f64::NEG_INFINITY, f64::max);
            if maximum == f64::NEG_INFINITY {
                return vec![0.0; keys];
            }
            let weights = (0..keys)
                .map(|key| {
                    if supported(key) {
                        (values[key] - maximum).exp()
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f64>();
            weights.into_iter().map(|weight| weight / total).collect()
        })
        .collect()
}
