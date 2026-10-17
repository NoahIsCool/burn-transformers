//! Tests for shared host metadata: allocation failures, rectangles, positions, and selection readback.

use burn::{
    backend::Flex,
    tensor::{Bool, DType, Int, Tensor, TensorData},
};

use super::{
    HostGrid, can_pack_token_ids, checked_token_id, empty_rows, next_positions,
    nonempty_row_lengths, read_grid, read_token_selection, rectangle_size, tensor_i64_values,
    vec_with_capacity,
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

/// A mask and its aligned values come back from one fused read, split exactly.
///
/// Every combination is covered: neither tensor (no read, all valid), a mask
/// alone, values alone, and both concatenated in one transfer.
#[test]
fn grids_split_fused_mask_and_value_reads_exactly() {
    let device = Default::default();
    let mask = || {
        Tensor::<TestBackend, 2, Bool>::from_data(
            TensorData::from([[false, true, true], [true, true, false]]),
            &device,
        )
    };
    let values = || Tensor::<TestBackend, 2, Int>::from_ints([[7, 0, 1], [2, 3, -4]], &device);
    let flags = vec![false, true, true, true, true, false];
    let numbers = vec![7, 0, 1, 2, 3, -4];
    for (mask, values, expected) in [
        (
            None,
            None,
            HostGrid {
                valid: vec![true; 6],
                values: None,
            },
        ),
        (
            Some(mask()),
            None,
            HostGrid {
                valid: flags.clone(),
                values: None,
            },
        ),
        (
            None,
            Some(values()),
            HostGrid {
                valid: vec![true; 6],
                values: Some(numbers.clone()),
            },
        ),
        (
            Some(mask()),
            Some(values()),
            HostGrid {
                valid: flags,
                values: Some(numbers),
            },
        ),
    ] {
        assert_eq!(read_grid(mask, values, 2, 3, "grid").unwrap(), expected);
    }
}

/// Misshaped grids and impossible rectangles fail before any device read.
#[test]
fn grids_reject_misshaped_tensors_and_overflowing_rectangles() {
    let device = Default::default();
    let values = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2], [3, 4]], &device);
    let error = read_grid(None, Some(values), 2, 3, "prompt").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("prompt values has shape [2, 2], expected [2, 3]"),
        "{error}"
    );
    let mask = Tensor::<TestBackend, 2, Bool>::from_data(TensorData::from([[true]]), &device);
    let error = read_grid(Some(mask), None, 1, 2, "prompt").unwrap_err();
    assert!(error.to_string().contains("prompt mask has shape [1, 1]"));
    assert!(read_grid::<TestBackend>(None, None, usize::MAX, 2, "prompt").is_err());
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

/// Token values beyond `u32` are rejected even when the vocabulary is larger.
#[test]
fn selected_tokens_must_fit_the_u32_id_domain() {
    let error = checked_token_id(2_f64.powi(32), usize::MAX).unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Generation(ref reason)
            if reason == "backend selected token 4294967296 outside the u32 token-ID domain"
    ));
    assert_eq!(
        checked_token_id(f64::from(u32::MAX), usize::MAX).unwrap(),
        u32::MAX
    );
}

/// Positions and validity must each cover the whole prompt rectangle.
#[test]
fn next_positions_require_a_full_validity_rectangle() {
    let error = next_positions(&[0, 1, 2, 3], &[true; 3], 2, 2, 4, "prompt").unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Generation(ref reason)
            if reason == "prompt metadata is not rectangular: 4 positions and 3 mask entries for 4 slots"
    ));
    assert_eq!(
        next_positions(&[0, 1, 5, 6], &[true, true, true, false], 2, 2, 4, "prompt").unwrap(),
        [2, 6]
    );
}

/// Integer readback accepts every integer storage type, not only the backend default.
///
/// Burn keeps an explicit `cast` to I64 even on an I32-default backend, so a
/// caller's 64-bit token IDs or positions must read back unchanged.
#[test]
fn integer_readback_accepts_non_default_integer_storage() {
    let device = Default::default();
    let ids = Tensor::<TestBackend, 2, Int>::from_ints([[3, -4], [5, 6]], &device);
    for dtype in [burn::tensor::IntDType::I32, burn::tensor::IntDType::I64] {
        let cast = ids.clone().cast(dtype);
        assert_eq!(
            tensor_i64_values(cast, "token IDs").unwrap(),
            [3, -4, 5, 6],
            "{dtype:?}"
        );
    }
    // The unpacked selection path (here forced by a vocabulary too wide for
    // exact F32 IDs) reads 64-bit token IDs through the same readback.
    let selection = TokenSelection::<TestBackend> {
        token_ids: Tensor::<TestBackend, 1, Int>::from_ints([3, 0], &device)
            .cast(burn::tensor::IntDType::I64),
        log_probabilities: Tensor::<TestBackend, 1>::from_floats([-0.5, -2.0], &device),
    };
    let (ids, scores) = read_token_selection(selection, (1 << 24) + 1).unwrap();
    assert_eq!(ids, [3, 0]);
    assert_eq!(scores, [-0.5, -2.0]);
}

/// Marks the re-executed test binary that runs the 64-bit default-integer child.
const I64_DEFAULT_CHILD: &str = "TRANSFORMERS_METADATA_I64_DEFAULT_CHILD";

/// Metadata readback accepts the integer dtype a device is configured to produce.
///
/// `Tensor::int` (mask readback) and argmax (greedy token selection) produce
/// the device's default integer dtype, which `set_default_dtypes` can make I64
/// on this I32 backend. Readers that required the backend's `IntElem` then
/// failed with "expected I64, got I32". Default dtypes are per device,
/// process-global, and settable only once, so the check runs in a fresh child
/// process of this test binary.
#[test]
fn metadata_readback_follows_a_changed_default_integer_dtype() {
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "generation::metadata::tests::i64_default_integer_child",
            "--ignored",
            "--test-threads=1",
        ])
        .env(I64_DEFAULT_CHILD, "1")
        .output()
        .expect("spawn child test process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "child failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Child half of [`metadata_readback_follows_a_changed_default_integer_dtype`].
#[test]
#[ignore = "runs in a fresh process spawned by metadata_readback_follows_a_changed_default_integer_dtype"]
fn i64_default_integer_child() {
    if std::env::var_os(I64_DEFAULT_CHILD).is_none() {
        return;
    }
    let device = Default::default();
    burn::tensor::set_default_dtypes::<TestBackend>(
        &device,
        DType::F32,
        burn::tensor::IntDType::I64,
    )
    .expect("fresh process has no tensors yet");
    let mask = Tensor::<TestBackend, 2, Bool>::from_data(
        TensorData::from([[true, false, true], [false, true, true]]),
        &device,
    );
    assert_eq!(mask.clone().int().dtype(), DType::I64);
    let flags = vec![true, false, true, false, true, true];
    // A mask alone is read through the Boolean reader.
    let grid = read_grid(Some(mask.clone()), None, 2, 3, "mask").expect("mask alone");
    assert_eq!(grid.valid, flags);
    assert_eq!(grid.values, None);
    // A mask with values is read in one fused integer transfer.
    let values = Tensor::<TestBackend, 2, Int>::from_ints([[7, 0, 1], [2, 3, -4]], &device);
    let grid = read_grid(Some(mask), Some(values), 2, 3, "mask").expect("fused read");
    assert_eq!(grid.valid, flags);
    assert_eq!(grid.values, Some(vec![7, 0, 1, 2, 3, -4]));
    // Greedy selection reads argmax output in both the packed and separate paths.
    let logits = Tensor::<TestBackend, 2>::from_floats([[0.0, 2.0, 1.0], [3.0, 0.0, 0.5]], &device);
    let token_ids = logits.argmax(1).reshape([2]);
    assert_eq!(token_ids.dtype(), DType::I64);
    for vocabulary_size in [3, (1 << 24) + 1] {
        let selection = TokenSelection::<TestBackend> {
            token_ids: token_ids.clone(),
            log_probabilities: Tensor::<TestBackend, 1>::from_floats([-0.5, -2.0], &device),
        };
        let (ids, scores) =
            read_token_selection(selection, vocabulary_size).expect("argmax selection");
        assert_eq!(ids, [1, 0], "vocabulary {vocabulary_size}");
        assert_eq!(scores, [-0.5, -2.0], "vocabulary {vocabulary_size}");
    }
}
