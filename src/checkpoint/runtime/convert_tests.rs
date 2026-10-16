//! Parity, layout, and failure tests for the bulk scalar conversion kernels.
//!
//! The reference converts one element at a time: it decodes each little-endian
//! scalar exactly into `f64`, then rounds once with the independently tested
//! integer encoder (`round_f64_bits`) or Rust's correctly rounded `as f32`. For
//! F16, BF16, and F32 sources this is a different implementation from the
//! kernels' `half`-based `f32` path, so agreement on every 16-bit pattern and on
//! random 32/64-bit patterns checks the optimized path bit for bit.

use burn::tensor::{bf16, f16};
use proptest::prelude::*;

use super::super::rounding::{BFLOAT16, BINARY16, round_f64_bits};
use super::*;

/// The four dtypes between which numerical casts are supported.
const FLOATS: [TensorDType; 4] = [
    TensorDType::F16,
    TensorDType::BF16,
    TensorDType::F32,
    TensorDType::F64,
];

/// Returns a dtype's scalar width in bytes.
fn width(dtype: TensorDType) -> usize {
    (dtype.bits() / 8) as usize
}

/// Decodes one little-endian scalar of a castable float dtype exactly into `f64`.
fn decode_exact(dtype: TensorDType, bytes: &[u8]) -> f64 {
    match dtype {
        TensorDType::F16 => {
            f64::from(f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
        }
        TensorDType::BF16 => {
            f64::from(bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
        }
        TensorDType::F32 => f64::from(f32::from_le_bytes(bytes.try_into().unwrap())),
        TensorDType::F64 => f64::from_le_bytes(bytes.try_into().unwrap()),
        _ => unreachable!("not a castable float"),
    }
}

/// Encodes an exact `f64` into native-endian target bytes with one rounding.
fn encode_once(dtype: TensorDType, value: f64) -> Vec<u8> {
    match dtype {
        TensorDType::F16 => round_f64_bits(value.to_bits(), BINARY16)
            .to_ne_bytes()
            .to_vec(),
        TensorDType::BF16 => round_f64_bits(value.to_bits(), BFLOAT16)
            .to_ne_bytes()
            .to_vec(),
        TensorDType::F32 => (value as f32).to_ne_bytes().to_vec(),
        TensorDType::F64 => value.to_ne_bytes().to_vec(),
        _ => unreachable!("not a castable float"),
    }
}

/// Element-at-a-time reference conversion of a whole little-endian buffer.
fn reference(source: &[u8], source_dtype: TensorDType, target_dtype: TensorDType) -> Vec<u8> {
    source
        .chunks_exact(width(source_dtype))
        .flat_map(|scalar| {
            if source_dtype == target_dtype {
                let mut native = scalar.to_vec();
                if cfg!(target_endian = "big") {
                    native.reverse();
                }
                native
            } else {
                encode_once(target_dtype, decode_exact(source_dtype, scalar))
            }
        })
        .collect()
}

/// Runs the whole-buffer gather kernel for one dtype pair.
fn gather_all(source: &[u8], source_dtype: TensorDType, target_dtype: TensorDType) -> Vec<u8> {
    let elements = source.len() / width(source_dtype);
    with_scalar_conversion(
        source_dtype,
        target_dtype,
        GatherIntervals {
            tensor: "fixture",
            source,
            intervals: core::iter::once(0..elements),
            elements,
        },
    )
    .unwrap()
    .unwrap()
}

/// Compares converted buffers scalar by scalar, matching NaNs by classification.
fn assert_same_scalars(actual: &[u8], expected: &[u8], dtype: TensorDType, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    let step = width(dtype);
    for (index, (left, right)) in actual
        .chunks_exact(step)
        .zip(expected.chunks_exact(step))
        .enumerate()
    {
        if left == right {
            continue;
        }
        let as_le = |bytes: &[u8]| {
            let mut little = bytes.to_vec();
            if cfg!(target_endian = "big") {
                little.reverse();
            }
            decode_exact(dtype, &little)
        };
        assert!(
            as_le(left).is_nan() && as_le(right).is_nan(),
            "{context}: element {index} is {left:02x?}, expected {right:02x?}"
        );
    }
}

/// Every F16 and BF16 bit pattern converts to every float dtype like the reference.
#[test]
fn every_sixteen_bit_pattern_matches_the_reference_for_all_targets() {
    let source = (0..=u16::MAX)
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    for source_dtype in [TensorDType::F16, TensorDType::BF16] {
        for target_dtype in FLOATS {
            let actual = gather_all(&source, source_dtype, target_dtype);
            let expected = reference(&source, source_dtype, target_dtype);
            assert_same_scalars(
                &actual,
                &expected,
                target_dtype,
                &format!("{source_dtype}->{target_dtype}"),
            );
        }
    }
}

/// F32 values one unit above a BF16 tie round up; the previous path truncated them.
#[test]
fn f32_sticky_bits_below_the_bf16_tie_round_up() {
    let source = [0x3F80_8001_u32, 0x3F80_8000, 0x3F81_8000, 0xBF80_8001]
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect::<Vec<_>>();
    let actual = gather_all(&source, TensorDType::F32, TensorDType::BF16);
    let bits = actual
        .as_chunks::<2>()
        .0
        .iter()
        .map(|scalar| u16::from_ne_bytes(*scalar))
        .collect::<Vec<_>>();
    assert_eq!(bits, [0x3F81, 0x3F80, 0x3F82, 0xBF81]);
}

/// F64 inputs round once with a full sticky bit for both 16-bit targets.
#[test]
fn f64_sources_round_once_to_sixteen_bit_targets() {
    let values = [
        1.0 + 2f64.powi(-8) + 2f64.powi(-40),
        1.0 + 2f64.powi(-11) + 2f64.powi(-40),
    ];
    let source = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let bf16_bits = gather_all(&source, TensorDType::F64, TensorDType::BF16);
    assert_eq!(u16::from_ne_bytes([bf16_bits[0], bf16_bits[1]]), 0x3F81);
    let f16_bits = gather_all(&source, TensorDType::F64, TensorDType::F16);
    assert_eq!(u16::from_ne_bytes([f16_bits[2], f16_bits[3]]), 0x3C01);
}

/// Equal dtypes of every whole-byte width preserve each bit, including NaN payloads.
#[test]
fn equal_dtypes_preserve_every_bit_at_each_width() {
    let payload = (0..64_u8)
        .map(|byte| byte.wrapping_mul(37) ^ 0xA5)
        .collect::<Vec<_>>();
    for dtype in [
        TensorDType::U8,
        TensorDType::BOOL,
        TensorDType::I16,
        TensorDType::BF16,
        TensorDType::F32,
        TensorDType::U32,
        TensorDType::F64,
        TensorDType::I64,
    ] {
        let actual = gather_all(&payload, dtype, dtype);
        let expected = reference(&payload, dtype, dtype);
        assert_eq!(actual, expected, "{dtype}");
    }
    let nan_payload = 0x7FF0_0000_0000_0001_u64.to_le_bytes();
    let preserved = gather_all(&nan_payload, TensorDType::F64, TensorDType::F64);
    assert_eq!(
        f64::from_ne_bytes(preserved.try_into().unwrap()).to_bits(),
        0x7FF0_0000_0000_0001
    );
}

/// Only conventional float pairs convert, and sub-byte or complex storage cannot run.
#[test]
fn unsupported_pairs_and_widths_fail_before_the_kernel_runs() {
    let kernel = |source: &'static [u8]| GatherIntervals {
        tensor: "fixture",
        source,
        intervals: core::iter::once(0..0),
        elements: 0,
    };
    for (source_dtype, target_dtype) in [
        (TensorDType::I32, TensorDType::F32),
        (TensorDType::F32, TensorDType::U32),
        (TensorDType::F8_E4M3, TensorDType::F16),
        (TensorDType::BF16, TensorDType::F8_E5M2),
    ] {
        assert!(matches!(
            with_scalar_conversion(source_dtype, target_dtype, kernel(&[])),
            Err(CheckpointError::UnsupportedDTypeConversion { source_dtype: source, target_dtype: target })
            if source == source_dtype && target == target_dtype
        ));
    }
    for dtype in [TensorDType::F4, TensorDType::F6_E2M3] {
        assert!(matches!(
            with_scalar_conversion(dtype, dtype, kernel(&[])),
            Err(CheckpointError::UnsupportedRuntimeDType { dtype: actual }) if actual == dtype
        ));
    }
}

/// Partial scalars, out-of-range intervals, and miscounted outputs are contextual errors.
#[test]
fn malformed_kernel_inputs_are_rejected_without_panicking() {
    let source = [0_u8; 6];
    let partial = GatherIntervals {
        tensor: "partial",
        source: &source[..5],
        intervals: core::iter::once(0..1),
        elements: 1,
    };
    assert!(matches!(
        with_scalar_conversion(TensorDType::F16, TensorDType::F32, partial),
        Ok(Err(CheckpointError::InvalidTensorPayload { tensor, .. })) if tensor == "partial"
    ));
    let beyond = GatherIntervals {
        tensor: "beyond",
        source: &source,
        intervals: core::iter::once(2..4),
        elements: 2,
    };
    assert!(matches!(
        with_scalar_conversion(TensorDType::F16, TensorDType::F32, beyond),
        Ok(Err(CheckpointError::InvalidTensorPayload { tensor, .. })) if tensor == "beyond"
    ));
    let miscounted = GatherIntervals {
        tensor: "count",
        source: &source,
        intervals: core::iter::once(0..2),
        elements: 3,
    };
    assert!(matches!(
        with_scalar_conversion(TensorDType::F16, TensorDType::F16, miscounted),
        Ok(Err(CheckpointError::InvalidTensorPayload { tensor, .. })) if tensor == "count"
    ));

    let mut target = [0xA5_u8; 8];
    let mismatched = ScatterIntervals {
        tensor: "scatter",
        source: &source,
        target: &mut target,
        blocks: core::iter::once((0..2, 0..1)),
    };
    assert!(matches!(
        with_scalar_conversion(TensorDType::F16, TensorDType::F32, mismatched),
        Ok(Err(CheckpointError::InvalidTensorPayload { tensor, .. })) if tensor == "scatter"
    ));
    assert_eq!(target, [0xA5; 8], "a rejected block writes nothing");
    let outside = ScatterIntervals {
        tensor: "scatter",
        source: &source,
        target: &mut target,
        blocks: core::iter::once((0..1, 2..3)),
    };
    assert!(matches!(
        with_scalar_conversion(TensorDType::F16, TensorDType::F32, outside),
        Ok(Err(CheckpointError::InvalidTensorPayload { .. }))
    ));
    assert_eq!(target, [0xA5; 8]);
}

/// Empty tensors produce empty outputs on every path.
#[test]
fn empty_inputs_produce_empty_outputs() {
    assert!(gather_all(&[], TensorDType::BF16, TensorDType::F32).is_empty());
    let mut target = [];
    let transpose = TransposeInto {
        tensor: "empty",
        source: &[],
        target: &mut target,
        rows: 0,
        columns: 7,
    };
    with_scalar_conversion(TensorDType::F32, TensorDType::F16, transpose)
        .unwrap()
        .unwrap();
}

/// Builds `count` little-endian scalars of `dtype` from arbitrary seed bits.
fn scalars(dtype: TensorDType, count: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut bytes = Vec::with_capacity(count * width(dtype));
    for _ in 0..count {
        // xorshift64: deterministic, cheap, and covers NaNs and infinities.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        bytes.extend_from_slice(&state.to_le_bytes()[..width(dtype)]);
    }
    bytes
}

// ---------------------------------------------------------------------------
// Multi-threaded transpose
//
// `transpose_tiles` splits matrices of at least `PARALLEL_MIN_SCALARS` scalars
// into column bands, one scoped thread each, sized from the host's available
// parallelism. These tests drive `transpose_tiles_with_parallelism` with chosen
// budgets so every band layout runs on any host, including a single-core one.
// ---------------------------------------------------------------------------

/// Source rows of the threaded-transpose fixtures.
const THREADED_ROWS: usize = 1031;

/// Source columns of the threaded-transpose fixtures: 33 tile blocks, the last
/// only 13 columns wide, so every multi-band plan ends in a ragged band.
const THREADED_COLUMNS: usize = 1037;

/// Parallelism budgets covering one band, two, three, seven, the 14 cores of
/// the measurement host (fewer bands than threads), and more threads than the
/// 33 tile blocks (one block per band).
const THREADED_BUDGETS: [usize; 6] = [1, 2, 3, 7, 14, 64];

/// Runs the production conversion selected for a dtype pair through the
/// multi-band transpose with an explicit parallelism budget.
///
/// This mirrors [`TransposeInto`] except that the budget is injected, so the
/// conversion closure is exactly the one [`with_scalar_conversion`] selects.
struct TransposeWithParallelism<'a> {
    /// Little-endian `[rows, columns]` source scalars.
    source: &'a [u8],
    /// Native-endian `[columns, rows]` destination scalars.
    target: &'a mut [u8],
    /// Source row count.
    rows: usize,
    /// Source column count.
    columns: usize,
    /// Most threads the transpose may use.
    available: usize,
}

impl ScalarKernel for TransposeWithParallelism<'_> {
    /// The transpose cannot fail once both buffers are whole scalars.
    type Output = ();

    /// Views both buffers as whole scalars and transposes with the injected budget.
    fn run<const SOURCE: usize, const TARGET: usize>(
        self,
        convert: impl Fn([u8; SOURCE]) -> [u8; TARGET] + Copy + Send + Sync,
    ) {
        let source = whole_scalars::<SOURCE>("threaded", self.source).unwrap();
        let target = whole_scalars_mut::<TARGET>("threaded", self.target).unwrap();
        assert_eq!(source.len(), self.rows * self.columns);
        assert_eq!(target.len(), self.rows * self.columns);
        transpose_tiles_with_parallelism(
            source,
            target,
            self.rows,
            self.columns,
            self.available,
            convert,
        );
    }
}

/// Reorders a row-major `[rows, columns]` buffer of `step`-byte scalars into
/// `[columns, rows]` one scalar at a time, the deliberately naive oracle.
fn naive_transpose(bytes: &[u8], rows: usize, columns: usize, step: usize) -> Vec<u8> {
    let mut output = vec![0_u8; bytes.len()];
    for row in 0..rows {
        for column in 0..columns {
            let from = (row * columns + column) * step;
            let to = (column * rows + row) * step;
            output[to..to + step].copy_from_slice(&bytes[from..from + step]);
        }
    }
    output
}

/// The fixtures are large enough to take the threaded path and every budget
/// above one really splits the columns into several bands, the last ragged.
#[test]
fn threaded_fixtures_split_into_several_bands() {
    const { assert!(THREADED_ROWS * THREADED_COLUMNS >= PARALLEL_MIN_SCALARS) };
    let blocks = THREADED_COLUMNS.div_ceil(TILE_EDGE);
    for available in THREADED_BUDGETS {
        let plan = transpose_plan(THREADED_ROWS, THREADED_COLUMNS, available);
        assert_eq!(plan.threads, available.min(blocks), "budget {available}");
        assert_eq!(plan.bands > 1, available > 1, "budget {available}");
        let last_band = THREADED_COLUMNS - (plan.bands - 1) * plan.band_columns;
        if available > 1 {
            assert!(
                last_band < plan.band_columns,
                "ragged final band, budget {available}"
            );
        }
    }
}

/// Every element of a threaded index transpose lands at `c · rows + r`, for
/// every band layout, so the output is independent of the thread count.
#[test]
fn threaded_index_transpose_places_every_element_for_every_budget() {
    let (rows, columns) = (THREADED_ROWS, THREADED_COLUMNS);
    let source = (0..rows * columns)
        .map(|index| u32::try_from(index).unwrap().to_ne_bytes())
        .collect::<Vec<_>>();
    for available in THREADED_BUDGETS {
        let mut target = vec![[0xA5_u8; 4]; rows * columns];
        transpose_tiles_with_parallelism(&source, &mut target, rows, columns, available, |s| s);
        for (index, scalar) in target.iter().enumerate() {
            let (column, row) = (index / rows, index % rows);
            assert_eq!(
                u32::from_ne_bytes(*scalar) as usize,
                row * columns + column,
                "budget {available}, source ({row}, {column})"
            );
        }
    }
}

/// Threaded transposes that change dtype (BF16 to F32 and F32 to F64, so the
/// target scalar is wider than the source) equal the in-order bulk conversion
/// followed by a naive transpose, byte for byte, for every band layout.
#[test]
fn threaded_converting_transpose_matches_bulk_conversion_for_every_budget() {
    let (rows, columns) = (THREADED_ROWS, THREADED_COLUMNS);
    for (seed, (source_dtype, target_dtype)) in [
        (0x7a9e_u64, (TensorDType::BF16, TensorDType::F32)),
        (0x7a9f_u64, (TensorDType::F32, TensorDType::F64)),
    ] {
        let source = scalars(source_dtype, rows * columns, seed);
        let expected = naive_transpose(
            &gather_all(&source, source_dtype, target_dtype),
            rows,
            columns,
            width(target_dtype),
        );
        for available in THREADED_BUDGETS {
            let mut target = vec![0xA5_u8; expected.len()];
            with_scalar_conversion(
                source_dtype,
                target_dtype,
                TransposeWithParallelism {
                    source: &source,
                    target: &mut target,
                    rows,
                    columns,
                    available,
                },
            )
            .unwrap();
            assert!(
                target == expected,
                "{source_dtype}->{target_dtype} transpose differs with budget {available}"
            );
        }
    }
}

/// The production [`TransposeInto`] kernel, with the host's own parallelism,
/// converts and transposes a matrix above the threading threshold exactly.
#[test]
fn transpose_into_converts_large_matrices_exactly() {
    let (rows, columns) = (THREADED_ROWS, THREADED_COLUMNS);
    let source = scalars(TensorDType::BF16, rows * columns, 0x0b16);
    let expected = naive_transpose(
        &gather_all(&source, TensorDType::BF16, TensorDType::F32),
        rows,
        columns,
        width(TensorDType::F32),
    );
    let mut target = vec![0xA5_u8; expected.len()];
    with_scalar_conversion(
        TensorDType::BF16,
        TensorDType::F32,
        TransposeInto {
            tensor: "large",
            source: &source,
            target: &mut target,
            rows,
            columns,
        },
    )
    .unwrap()
    .unwrap();
    assert!(target == expected, "BF16->F32 TransposeInto differs");
}

/// A zero budget is treated as one thread rather than rejected.
#[test]
fn zero_parallelism_budget_runs_on_the_caller_thread() {
    let source = (0..6_u32).map(u32::to_ne_bytes).collect::<Vec<_>>();
    let mut target = vec![[0_u8; 4]; 6];
    transpose_tiles_with_parallelism(&source, &mut target, 2, 3, 0, |s| s);
    let values = target
        .iter()
        .map(|scalar| u32::from_ne_bytes(*scalar))
        .collect::<Vec<_>>();
    assert_eq!(values, [0, 3, 1, 4, 2, 5]);
}

proptest! {
    /// Random F32 and F64 payloads agree with the per-element reference.
    #[test]
    fn wide_sources_match_the_reference(seed in any::<u64>(), count in 0_usize..256) {
        for source_dtype in [TensorDType::F32, TensorDType::F64] {
            let source = scalars(source_dtype, count, seed);
            for target_dtype in FLOATS {
                let actual = gather_all(&source, source_dtype, target_dtype);
                let expected = reference(&source, source_dtype, target_dtype);
                assert_same_scalars(&actual, &expected, target_dtype, "wide source");
            }
        }
    }

    /// Gathering arbitrary in-bounds intervals equals converting their concatenation.
    #[test]
    fn gathered_intervals_match_naive_concatenation(
        seed in any::<u64>(),
        cuts in proptest::collection::vec((0_usize..40, 0_usize..40), 0..6),
        pair in 0_usize..16,
    ) {
        let (source_dtype, target_dtype) = (FLOATS[pair / 4], FLOATS[pair % 4]);
        let source = scalars(source_dtype, 40, seed);
        let intervals = cuts
            .iter()
            .map(|&(left, right)| left.min(right)..left.max(right))
            .collect::<Vec<_>>();
        let elements = intervals.iter().map(ExactSizeIterator::len).sum();
        let actual = with_scalar_conversion(
            source_dtype,
            target_dtype,
            GatherIntervals {
                tensor: "gather",
                source: &source,
                intervals: intervals.clone().into_iter(),
                elements,
            },
        )
        .unwrap()
        .unwrap();
        let step = width(source_dtype);
        let selected = intervals
            .iter()
            .flat_map(|interval| source[interval.start * step..interval.end * step].to_vec())
            .collect::<Vec<_>>();
        assert_same_scalars(
            &actual,
            &reference(&selected, source_dtype, target_dtype),
            target_dtype,
            "gather",
        );
    }

    /// The tiled transpose matches a naive index transpose for edge-crossing shapes.
    #[test]
    fn tiled_transpose_matches_naive_index_transpose(
        seed in any::<u64>(),
        rows in 0_usize..70,
        columns in 0_usize..70,
        pair in 0_usize..16,
    ) {
        let (source_dtype, target_dtype) = (FLOATS[pair / 4], FLOATS[pair % 4]);
        let source = scalars(source_dtype, rows * columns, seed);
        let mut target = vec![0_u8; rows * columns * width(target_dtype)];
        with_scalar_conversion(
            source_dtype,
            target_dtype,
            TransposeInto {
                tensor: "transpose",
                source: &source,
                target: &mut target,
                rows,
                columns,
            },
        )
        .unwrap()
        .unwrap();
        let step = width(source_dtype);
        let mut transposed = Vec::with_capacity(source.len());
        for column in 0..columns {
            for row in 0..rows {
                let index = row * columns + column;
                transposed.extend_from_slice(&source[index * step..(index + 1) * step]);
            }
        }
        assert_same_scalars(
            &target,
            &reference(&transposed, source_dtype, target_dtype),
            target_dtype,
            "transpose",
        );
    }

    /// Scattered blocks equal converting each block and writing it at its target offset.
    #[test]
    fn scattered_blocks_match_naive_placement(
        seed in any::<u64>(),
        rows in 1_usize..8,
        block in 0_usize..9,
        pair in 0_usize..16,
    ) {
        let (source_dtype, target_dtype) = (FLOATS[pair / 4], FLOATS[pair % 4]);
        let source = scalars(source_dtype, rows * block, seed);
        // Interleave each source row with an untouched gap of equal width.
        let mut target = vec![0x5A_u8; 2 * rows * block * width(target_dtype)];
        let blocks = (0..rows).map(|row| (row * block..(row + 1) * block, (2 * row + 1) * block..(2 * row + 2) * block));
        with_scalar_conversion(
            source_dtype,
            target_dtype,
            ScatterIntervals { tensor: "scatter", source: &source, target: &mut target, blocks },
        )
        .unwrap()
        .unwrap();
        let converted = reference(&source, source_dtype, target_dtype);
        let row_bytes = block * width(target_dtype);
        for row in 0..rows {
            let gap = &target[2 * row * row_bytes..(2 * row + 1) * row_bytes];
            prop_assert!(gap.iter().all(|&byte| byte == 0x5A));
            assert_same_scalars(
                &target[(2 * row + 1) * row_bytes..(2 * row + 2) * row_bytes],
                &converted[row * row_bytes..(row + 1) * row_bytes],
                target_dtype,
                "scatter",
            );
        }
    }
}

/// The element loop of the previous executor, kept as a timing baseline.
///
/// It recomputed checked byte ranges, re-dispatched on both dtypes, and decoded
/// through `f64` with `half::from_f64` for every element.
fn previous_scalar_path(
    source: &[u8],
    source_dtype: TensorDType,
    target: &mut [u8],
    target_dtype: TensorDType,
) {
    let source_width = width(source_dtype) as u64;
    let target_width = width(target_dtype) as u64;
    let elements = source.len() as u64 / source_width;
    for index in 0..elements {
        let source_start = usize::try_from(index.checked_mul(source_width).unwrap()).unwrap();
        let source_end = usize::try_from(
            index
                .checked_mul(source_width)
                .and_then(|start| start.checked_add(source_width))
                .unwrap(),
        )
        .unwrap();
        let target_start = usize::try_from(index.checked_mul(target_width).unwrap()).unwrap();
        let target_end = usize::try_from(
            index
                .checked_mul(target_width)
                .and_then(|start| start.checked_add(target_width))
                .unwrap(),
        )
        .unwrap();
        let bytes = source.get(source_start..source_end).unwrap();
        let slot = target.get_mut(target_start..target_end).unwrap();
        let value = match source_dtype {
            TensorDType::F16 => {
                f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f64()
            }
            TensorDType::BF16 => {
                bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f64()
            }
            TensorDType::F32 => f64::from(f32::from_le_bytes(bytes.try_into().unwrap())),
            _ => f64::from_le_bytes(bytes.try_into().unwrap()),
        };
        match target_dtype {
            TensorDType::F16 => slot.copy_from_slice(&f16::from_f64(value).to_bits().to_ne_bytes()),
            TensorDType::BF16 => {
                slot.copy_from_slice(&bf16::from_f64(value).to_bits().to_ne_bytes())
            }
            TensorDType::F32 => slot.copy_from_slice(&(value as f32).to_ne_bytes()),
            _ => slot.copy_from_slice(&value.to_ne_bytes()),
        }
    }
}

/// The previous executor's equal-dtype tiled byte transpose, kept as a timing baseline.
fn previous_tiled_transpose<const WIDTH: usize>(
    source: &[u8],
    target: &mut [u8],
    rows: usize,
    columns: usize,
) {
    /// Tile edge of the previous implementation, in scalars.
    const TILE_EDGE: usize = 32;
    let source_row_bytes = columns * WIDTH;
    let target_row_bytes = rows * WIDTH;
    let columns_per_tile = columns.min(TILE_EDGE);
    let target_tile_bytes = target_row_bytes * columns_per_tile;
    for (column_block, target_block) in target.chunks_mut(target_tile_bytes).enumerate() {
        let column_start = column_block * columns_per_tile;
        for row_start in (0..rows).step_by(TILE_EDGE) {
            let row_end = (row_start + TILE_EDGE).min(rows);
            let source_rows = &source[row_start * source_row_bytes..row_end * source_row_bytes];
            for (local_column, target_row) in
                target_block.chunks_exact_mut(target_row_bytes).enumerate()
            {
                let source_column = (column_start + local_column) * WIDTH;
                let target_scalars = &mut target_row[row_start * WIDTH..row_end * WIDTH];
                for (source_row, target_scalar) in source_rows
                    .chunks_exact(source_row_bytes)
                    .zip(target_scalars.as_chunks_mut::<WIDTH>().0)
                {
                    target_scalar
                        .copy_from_slice(&source_row[source_column..source_column + WIDTH]);
                }
            }
        }
    }
}

/// Returns the fastest of several timed runs, in nanoseconds per element.
fn best_ns_per_element(elements: usize, mut run: impl FnMut()) -> f64 {
    (0..5)
        .map(|_| {
            let started = std::time::Instant::now();
            run();
            started.elapsed().as_secs_f64() * 1e9 / elements as f64
        })
        .fold(f64::INFINITY, f64::min)
}

/// Times equal-dtype transposes against the previous specialized byte path.
///
/// Destination buffers are written before timing so page faults are excluded,
/// and each path reports its best of five runs.
#[test]
#[ignore = "timing benchmark; run explicitly in release mode"]
fn equal_dtype_transpose_throughput_against_previous_tiled_path() {
    let side = 4096;
    for dtype in [TensorDType::BF16, TensorDType::F32] {
        let source = scalars(dtype, side * side, 0x5eed);
        let mut current = vec![1_u8; source.len()];
        let current_time = best_ns_per_element(side * side, || {
            with_scalar_conversion(
                dtype,
                dtype,
                TransposeInto {
                    tensor: "bench",
                    source: &source,
                    target: &mut current,
                    rows: side,
                    columns: side,
                },
            )
            .unwrap()
            .unwrap();
        });
        let mut previous = vec![1_u8; source.len()];
        let previous_time = best_ns_per_element(side * side, || {
            if width(dtype) == 2 {
                previous_tiled_transpose::<2>(&source, &mut previous, side, side);
            } else {
                previous_tiled_transpose::<4>(&source, &mut previous, side, side);
            }
        });
        assert_eq!(current, previous);
        println!(
            "{dtype} transpose: current {current_time:.3} ns/element, previous {previous_time:.3} ns/element"
        );
    }
}

/// Measures bulk conversion against the previous per-element path.
///
/// Run with `cargo test --release --lib conversion_throughput -- --ignored --nocapture`.
/// It prints nanoseconds per element for 16M-element casts; numbers depend on
/// the host and are reported, not asserted.
#[test]
#[ignore = "timing benchmark; run explicitly in release mode"]
fn conversion_throughput_against_previous_scalar_path() {
    use std::time::Instant;

    /// Scalars converted per benchmark case (16M).
    const ELEMENTS: usize = 1 << 24;
    for (source_dtype, target_dtype) in [
        (TensorDType::BF16, TensorDType::F32),
        (TensorDType::F32, TensorDType::BF16),
        (TensorDType::F32, TensorDType::F16),
        (TensorDType::F16, TensorDType::F32),
        (TensorDType::F64, TensorDType::BF16),
    ] {
        // Finite values only, so both paths take their ordinary branches.
        let source = (0..ELEMENTS)
            .flat_map(|index| {
                let value = (index as f64 * 0.618_033_988_75).sin() * 4.0;
                match source_dtype {
                    TensorDType::BF16 => bf16::from_f64(value).to_bits().to_le_bytes().to_vec(),
                    TensorDType::F16 => f16::from_f64(value).to_bits().to_le_bytes().to_vec(),
                    TensorDType::F32 => (value as f32).to_le_bytes().to_vec(),
                    _ => value.to_le_bytes().to_vec(),
                }
            })
            .collect::<Vec<_>>();
        let started = Instant::now();
        let bulk = gather_all(&source, source_dtype, target_dtype);
        let bulk_time = started.elapsed();
        let mut previous = vec![0_u8; bulk.len()];
        let started = Instant::now();
        previous_scalar_path(&source, source_dtype, &mut previous, target_dtype);
        let previous_time = started.elapsed();
        std::hint::black_box((&bulk, &previous));
        println!(
            "{source_dtype}->{target_dtype}: bulk {:.3} ns/element, previous {:.3} ns/element ({:.1}x)",
            bulk_time.as_secs_f64() * 1e9 / ELEMENTS as f64,
            previous_time.as_secs_f64() * 1e9 / ELEMENTS as f64,
            previous_time.as_secs_f64() / bulk_time.as_secs_f64(),
        );

        // A 4096 x 4096 transpose with the same cast; the previous executor
        // called its scalar path once per element at the transposed offset.
        let side = 4096;
        let matrix = &source[..side * side * width(source_dtype)];
        let mut tiled = vec![0_u8; side * side * width(target_dtype)];
        let started = Instant::now();
        with_scalar_conversion(
            source_dtype,
            target_dtype,
            TransposeInto {
                tensor: "bench",
                source: matrix,
                target: &mut tiled,
                rows: side,
                columns: side,
            },
        )
        .unwrap()
        .unwrap();
        let tiled_time = started.elapsed();
        let mut per_element = vec![0_u8; tiled.len()];
        let (source_step, target_step) = (width(source_dtype), width(target_dtype));
        let started = Instant::now();
        for row in 0..side {
            for column in 0..side {
                let from = (row * side + column) * source_step;
                let to = (column * side + row) * target_step;
                previous_scalar_path(
                    &matrix[from..from + source_step],
                    source_dtype,
                    &mut per_element[to..to + target_step],
                    target_dtype,
                );
            }
        }
        let per_element_time = started.elapsed();
        std::hint::black_box((&tiled, &per_element));
        println!(
            "{source_dtype}->{target_dtype} transpose: tiled {:.3} ns/element, previous {:.3} ns/element ({:.1}x)",
            tiled_time.as_secs_f64() * 1e9 / (side * side) as f64,
            per_element_time.as_secs_f64() * 1e9 / (side * side) as f64,
            per_element_time.as_secs_f64() / tiled_time.as_secs_f64(),
        );
    }
}
