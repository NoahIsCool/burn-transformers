//! Tests for shared host metadata: allocation failures, rectangles, positions, and selection readback.

use burn::{
    backend::Flex,
    tensor::{DType, Int, Tensor, TensorData},
};

use super::{
    can_pack_token_ids, checked_token_id, empty_rows, nonempty_row_lengths, read_token_selection,
    rectangle_size, required_cache_capacity, vec_with_capacity,
};
use crate::generation::TokenSelection;

/// Deterministic F32 CPU backend.
type TestBackend = Flex<f32, i32>;

/// Impossible allocations and rectangles fail with context instead of aborting.
#[test]
fn allocation_and_rectangle_failures_are_recoverable() {
    assert!(vec_with_capacity::<u64>(usize::MAX, "fixture").is_err());
    assert!(empty_rows::<u8>(usize::MAX, "fixture").is_err());
    let error = rectangle_size(usize::MAX, 2, "fixture").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("fixture element count overflows")
    );
    assert_eq!(rectangle_size(3, 4, "fixture").unwrap(), 12);
    assert!(nonempty_row_lengths(&[], 0, 0, "empty").unwrap().is_empty());
}

/// The cache holds the prompt plus every fed-back token, never the final one.
#[test]
fn cache_capacity_excludes_the_final_token() {
    assert_eq!(required_cache_capacity(5, 1), Some(5));
    assert_eq!(required_cache_capacity(5, 3), Some(7));
    assert_eq!(required_cache_capacity(5, 0), Some(5));
    assert_eq!(required_cache_capacity(usize::MAX, 2), None);
}

/// Packing requires every token ID to be exact in both float dtypes.
#[test]
fn token_ids_pack_only_when_exactly_representable() {
    assert!(can_pack_token_ids(DType::F32, DType::F32, 1 << 24));
    assert!(!can_pack_token_ids(DType::F32, DType::F32, (1 << 24) + 1));
    assert!(can_pack_token_ids(DType::F32, DType::F64, 1 << 24));
    assert!(can_pack_token_ids(DType::F64, DType::F64, 1 << 30));
    assert!(!can_pack_token_ids(DType::F16, DType::F32, 8));
    assert!(!can_pack_token_ids(DType::F32, DType::Flex32, 8));
    assert_eq!(checked_token_id(7.0, 8).unwrap(), 7);
    for invalid in [8.0, -1.0, 0.5, f64::NAN] {
        assert!(checked_token_id(invalid, 8).is_err(), "{invalid}");
    }
}

/// Packed single-readback selection returns the same values as separate reads.
#[test]
fn selection_readback_is_exact_in_both_paths() {
    let device = Default::default();
    let selection = |dtype: DType| TokenSelection::<TestBackend> {
        token_ids: Tensor::<TestBackend, 1, Int>::from_ints([3, 0, 7], &device),
        log_probabilities: Tensor::<TestBackend, 1>::from_floats([-0.25, -1.5, -3.0e-7], &device)
            .cast(dtype),
    };
    for dtype in [DType::F32, DType::F64] {
        let (ids, scores) = read_token_selection(selection(dtype), 8).expect("packed read");
        assert_eq!(ids, [3, 0, 7]);
        assert_eq!(scores, [-0.25, -1.5, f64::from(-3.0e-7_f32)]);
    }
    // A vocabulary too wide for exact F32 IDs takes the two-read path.
    let (ids, scores) =
        read_token_selection(selection(DType::F32), (1 << 24) + 1).expect("separate reads");
    assert_eq!(ids, [3, 0, 7]);
    assert_eq!(scores[0], -0.25);
    // Out-of-vocabulary selections and mismatched shapes are rejected.
    assert!(read_token_selection(selection(DType::F32), 7).is_err());
    let ragged = TokenSelection::<TestBackend> {
        token_ids: Tensor::from_ints([1, 2], &device),
        log_probabilities: Tensor::from_data(TensorData::new(vec![0.0_f32], [1]), &device),
    };
    assert!(read_token_selection(ragged, 8).is_err());
}
