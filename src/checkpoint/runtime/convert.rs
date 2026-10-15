//! Monomorphic bulk scalar conversion from checkpoint bytes to Burn storage.
//!
//! SafeTensors payloads are little-endian; Burn [`burn::tensor::TensorData`]
//! bytes are native-endian. Every transform therefore applies one element-wise
//! map `f: [u8; S] -> [u8; T]` from an `S`-byte source scalar to a `T`-byte
//! target scalar:
//!
//! - equal dtypes: `f` reorders bytes to native endianness, the identity on
//!   little-endian hosts, and preserves every bit (NaN payloads, signed zero);
//! - F16/BF16 sources widen exactly to `f32`, and F32 sources are already `f32`;
//!   a narrower target is then produced by one round-to-nearest-even step
//!   (`half`'s `from_f32`, correctly rounded on every code path) and a wider one
//!   exactly;
//! - F64 sources narrow to F32 with Rust's correctly rounded `as f32`, and to
//!   F16/BF16 with [`super::rounding::round_f64_bits`], which keeps every
//!   discarded bit in the sticky remainder.
//!
//! Each value is thus rounded at most once from its exact decoded value. NaNs stay
//! NaN, are quieted when a format changes, and keep their most significant
//! payload bits.
//!
//! The dtype pair is matched once per tensor, not once per element. The selected
//! map is a statically known function, so each `(S, T)` instantiation of a
//! [`ScalarKernel`] compiles to a tight loop over fixed-size arrays without
//! per-element dispatch, range arithmetic, or bounds checks. Kernels write into
//! a reserved `Vec` through `extend` where the output is produced in order, which
//! avoids an initial zero-fill pass over the destination.

use burn::tensor::{bf16, f16};

use super::rounding::{BFLOAT16, BINARY16, round_f64_bits};
use crate::checkpoint::{CheckpointError, Result, TensorDType};

/// An element-wise operation instantiated for one statically selected conversion.
///
/// [`with_scalar_conversion`] matches the runtime dtype pair once and invokes
/// [`Self::run`] with a concrete conversion function, so the kernel body is
/// monomorphized for that pair's source and target scalar widths.
pub(super) trait ScalarKernel {
    /// Value produced by the kernel, typically a `Result` with output bytes.
    type Output;

    /// Runs the kernel with `convert` mapping one little-endian `SOURCE`-byte
    /// scalar to one native-endian `TARGET`-byte scalar.
    ///
    /// `convert` is a pure function of its scalar (every conversion selected by
    /// [`with_scalar_conversion`] is a capture-free closure), so it is `Send +
    /// Sync` and a kernel may apply it from several threads.
    fn run<const SOURCE: usize, const TARGET: usize>(
        self,
        convert: impl Fn([u8; SOURCE]) -> [u8; TARGET] + Copy + Send + Sync,
    ) -> Self::Output;
}

/// Selects the conversion for `(source, target)` and runs `kernel` with it.
///
/// Equal dtypes of any whole-byte width are byte-preserving; distinct dtypes are
/// supported only among F16, BF16, F32, and F64. Every other pair, including
/// integer reinterpretation, fails before the kernel runs.
pub(super) fn with_scalar_conversion<K: ScalarKernel>(
    source: TensorDType,
    target: TensorDType,
    kernel: K,
) -> Result<K::Output> {
    use TensorDType::{BF16, F16, F32, F64};

    if source == target {
        return match source.bits() {
            8 => Ok(kernel.run(little_endian_to_native::<1>)),
            16 => Ok(kernel.run(little_endian_to_native::<2>)),
            32 => Ok(kernel.run(little_endian_to_native::<4>)),
            64 => Ok(kernel.run(little_endian_to_native::<8>)),
            _ => Err(CheckpointError::UnsupportedRuntimeDType { dtype: source }),
        };
    }
    let output = match (source, target) {
        (F16, BF16) => kernel.run(|bytes| encode_bf16(decode_f16(bytes))),
        (F16, F32) => kernel.run(|bytes| encode_f32(decode_f16(bytes))),
        (F16, F64) => kernel.run(|bytes| encode_f64(f64::from(decode_f16(bytes)))),
        (BF16, F16) => kernel.run(|bytes| encode_f16(decode_bf16(bytes))),
        (BF16, F32) => kernel.run(|bytes| encode_f32(decode_bf16(bytes))),
        (BF16, F64) => kernel.run(|bytes| encode_f64(f64::from(decode_bf16(bytes)))),
        (F32, F16) => kernel.run(|bytes| encode_f16(f32::from_le_bytes(bytes))),
        (F32, BF16) => kernel.run(|bytes| encode_bf16(f32::from_le_bytes(bytes))),
        (F32, F64) => kernel.run(|bytes| encode_f64(f64::from(f32::from_le_bytes(bytes)))),
        (F64, F16) => kernel.run(|bytes| round_f64_to_native(bytes, BINARY16)),
        (F64, BF16) => kernel.run(|bytes| round_f64_to_native(bytes, BFLOAT16)),
        (F64, F32) => kernel.run(|bytes| encode_f32(f64::from_le_bytes(bytes) as f32)),
        _ => {
            return Err(CheckpointError::UnsupportedDTypeConversion {
                source_dtype: source,
                target_dtype: target,
            });
        }
    };
    Ok(output)
}

/// Rejects reinterpretation and permits casts only among conventional real floats.
///
/// This is exactly the domain accepted by [`with_scalar_conversion`] for
/// distinct dtypes; equal dtypes are validated by their Burn storage mapping.
pub(super) fn validate_cast(source: TensorDType, target: TensorDType) -> Result<()> {
    if source == target || (source.is_castable_float() && target.is_castable_float()) {
        Ok(())
    } else {
        Err(CheckpointError::UnsupportedDTypeConversion {
            source_dtype: source,
            target_dtype: target,
        })
    }
}

// Scalar codecs: little-endian decode, then one rounding step to native bytes.

/// Reorders one little-endian scalar to native byte order; a no-op on little-endian hosts.
#[inline]
fn little_endian_to_native<const WIDTH: usize>(mut bytes: [u8; WIDTH]) -> [u8; WIDTH] {
    if cfg!(target_endian = "big") {
        bytes.reverse();
    }
    bytes
}

/// Decodes a little-endian binary16 scalar exactly into `f32`.
#[inline]
fn decode_f16(bytes: [u8; 2]) -> f32 {
    f16::from_bits(u16::from_le_bytes(bytes)).to_f32()
}

/// Decodes a little-endian bfloat16 scalar exactly into `f32`.
#[inline]
fn decode_bf16(bytes: [u8; 2]) -> f32 {
    bf16::from_bits(u16::from_le_bytes(bytes)).to_f32()
}

/// Rounds an `f32` to binary16 with ties to even and returns native bytes.
#[inline]
fn encode_f16(value: f32) -> [u8; 2] {
    f16::from_f32(value).to_bits().to_ne_bytes()
}

/// Rounds an `f32` to bfloat16 with ties to even and returns native bytes.
#[inline]
fn encode_bf16(value: f32) -> [u8; 2] {
    bf16::from_f32(value).to_bits().to_ne_bytes()
}

/// Returns the native bytes of an `f32`.
#[inline]
fn encode_f32(value: f32) -> [u8; 4] {
    value.to_ne_bytes()
}

/// Returns the native bytes of an `f64`.
#[inline]
fn encode_f64(value: f64) -> [u8; 8] {
    value.to_ne_bytes()
}

/// Rounds a little-endian binary64 scalar once to a 16-bit format's native bytes.
#[inline]
fn round_f64_to_native(bytes: [u8; 8], format: super::rounding::HalfWidthFormat) -> [u8; 2] {
    round_f64_bits(u64::from_le_bytes(bytes), format).to_ne_bytes()
}

// Kernels: whole-tensor gathers, in-place same-width casts, and tiled transposes.

/// Converts selected source scalar intervals, in order, into a new output buffer.
///
/// Identity transfers use one interval covering the tensor; slices use one
/// interval per outer index. The output is reserved once with `elements`
/// scalars and filled by `extend`, so it is never zero-initialized.
pub(super) struct GatherIntervals<'a, I> {
    /// Diagnostic tensor name used in bounds and allocation errors.
    pub(super) tensor: &'a str,
    /// Complete little-endian source payload.
    pub(super) source: &'a [u8],
    /// Half-open source element intervals copied in this order.
    pub(super) intervals: I,
    /// Total element count across `intervals`, reserved up front.
    pub(super) elements: usize,
}

impl<I: Iterator<Item = core::ops::Range<usize>>> ScalarKernel for GatherIntervals<'_, I> {
    /// Native-endian output bytes, or a bounds/allocation failure.
    type Output = Result<Vec<u8>>;

    /// Reserves the output once, then appends each converted interval.
    fn run<const SOURCE: usize, const TARGET: usize>(
        self,
        convert: impl Fn([u8; SOURCE]) -> [u8; TARGET] + Copy + Send + Sync,
    ) -> Self::Output {
        let scalars = whole_scalars::<SOURCE>(self.tensor, self.source)?;
        let mut output: Vec<[u8; TARGET]> = Vec::new();
        output
            .try_reserve_exact(self.elements)
            .map_err(|_| allocation_failed(self.elements, TARGET))?;
        for interval in self.intervals {
            let block = scalars
                .get(interval)
                .ok_or_else(|| payload_bounds(self.tensor))?;
            output.extend(block.iter().map(|&scalar| convert(scalar)));
        }
        if output.len() != self.elements {
            return Err(payload_bounds(self.tensor));
        }
        Ok(output.into_flattened())
    }
}

/// Converts source element intervals into equally long, preallocated target intervals.
///
/// Concatenation reads one source at a time and writes each of its outer rows to
/// a strided block of the final output, so it targets an already allocated
/// buffer. Every pair is bounds-checked as a whole before its elements convert.
pub(super) struct ScatterIntervals<'a, I> {
    /// Diagnostic tensor name used in bounds errors.
    pub(super) tensor: &'a str,
    /// Little-endian source scalars.
    pub(super) source: &'a [u8],
    /// Native-endian destination scalars.
    pub(super) target: &'a mut [u8],
    /// `(source interval, target interval)` element pairs of equal length.
    pub(super) blocks: I,
}

impl<I> ScalarKernel for ScatterIntervals<'_, I>
where
    I: Iterator<Item = (core::ops::Range<usize>, core::ops::Range<usize>)>,
{
    /// Success, or a bounds error for the first invalid pair.
    type Output = Result<()>;

    /// Converts each source interval into its target interval.
    fn run<const SOURCE: usize, const TARGET: usize>(
        self,
        convert: impl Fn([u8; SOURCE]) -> [u8; TARGET] + Copy + Send + Sync,
    ) -> Self::Output {
        let source = whole_scalars::<SOURCE>(self.tensor, self.source)?;
        let target = whole_scalars_mut::<TARGET>(self.tensor, self.target)?;
        for (source_interval, target_interval) in self.blocks {
            let input = source
                .get(source_interval)
                .ok_or_else(|| payload_bounds(self.tensor))?;
            let output = target
                .get_mut(target_interval)
                .ok_or_else(|| payload_bounds(self.tensor))?;
            if input.len() != output.len() {
                return Err(payload_bounds(self.tensor));
            }
            for (slot, &scalar) in output.iter_mut().zip(input) {
                *slot = convert(scalar);
            }
        }
        Ok(())
    }
}

/// Converts a buffer in place when source and target scalars have equal width.
///
/// Each slot is read before it is overwritten, so no second buffer is needed.
/// This covers same-dtype endianness normalization and BF16 <-> F16 casts.
pub(super) struct ConvertInPlace<'a> {
    /// Diagnostic tensor name used in errors.
    pub(super) tensor: &'a str,
    /// Little-endian scalars replaced by native-endian converted scalars.
    pub(super) bytes: &'a mut [u8],
}

impl ScalarKernel for ConvertInPlace<'_> {
    /// Success, or an error when the conversion changes scalar width.
    type Output = Result<()>;

    /// Rewrites every scalar slot through `convert`.
    fn run<const SOURCE: usize, const TARGET: usize>(
        self,
        convert: impl Fn([u8; SOURCE]) -> [u8; TARGET] + Copy + Send + Sync,
    ) -> Self::Output {
        if SOURCE != TARGET {
            return Err(CheckpointError::InvalidTensorPayload {
                tensor: self.tensor.to_owned(),
                reason: format!(
                    "an in-place conversion cannot change scalar width from {SOURCE} to {TARGET} bytes"
                ),
            });
        }
        let slots = whole_scalars_mut::<SOURCE>(self.tensor, self.bytes)?;
        for slot in slots {
            let converted = convert(*slot);
            // `SOURCE == TARGET` was checked above, so this copies a full slot.
            slot.copy_from_slice(&converted);
        }
        Ok(())
    }
}

/// Transposes a row-major `[rows, columns]` matrix while converting each scalar.
///
/// Source scalar `(r, c)` at element `r * columns + c` lands at target element
/// `c * rows + r`. Both axes are blocked into `TILE_EDGE`-element tiles, so the
/// active source rows and destination row segments stay in cache. The target
/// must be preallocated: transposed writes are not produced in output order.
pub(super) struct TransposeInto<'a> {
    /// Diagnostic tensor name used in bounds errors.
    pub(super) tensor: &'a str,
    /// Little-endian `[rows, columns]` source scalars.
    pub(super) source: &'a [u8],
    /// Native-endian `[columns, rows]` destination scalars.
    pub(super) target: &'a mut [u8],
    /// Source row count.
    pub(super) rows: usize,
    /// Source column count.
    pub(super) columns: usize,
}

impl ScalarKernel for TransposeInto<'_> {
    /// Success, or a layout error detected before any target byte changes.
    type Output = Result<()>;

    /// Validates both buffers hold exactly `rows * columns` scalars, then transposes.
    fn run<const SOURCE: usize, const TARGET: usize>(
        self,
        convert: impl Fn([u8; SOURCE]) -> [u8; TARGET] + Copy + Send + Sync,
    ) -> Self::Output {
        let source = whole_scalars::<SOURCE>(self.tensor, self.source)?;
        let target = whole_scalars_mut::<TARGET>(self.tensor, self.target)?;
        let elements =
            self.rows
                .checked_mul(self.columns)
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "computing tiled transpose element count",
                })?;
        if source.len() != elements || target.len() != elements {
            return Err(payload_bounds(self.tensor));
        }
        transpose_tiles(source, target, self.rows, self.columns, convert);
        Ok(())
    }
}

/// Maximum tile edge in scalars, independent of byte width.
const TILE_EDGE: usize = 32;

/// Smallest matrix, in scalars, whose transpose is split across threads.
///
/// Below one mebi-scalar (4 MiB of F32) the copy takes well under a millisecond
/// and thread start-up would dominate; above it the work is memory-bound and
/// scales with the number of cores feeding the memory system.
const PARALLEL_MIN_SCALARS: usize = 1 << 20;

/// Applies a cache-tiled transpose to buffers already proven to hold `rows * columns` scalars.
///
/// Target row `c` (of length `rows`) holds source column `c`, i.e. source scalar
/// `(r, c)` lands at target element `c * rows + r` after `convert`. Target rows
/// are independent, so the target is cut into contiguous bands of whole
/// `TILE_EDGE`-column blocks; bands of a matrix with at least
/// `PARALLEL_MIN_SCALARS` scalars run on up to
/// [`std::thread::available_parallelism`] scoped threads, each owning its band.
/// Every target scalar is written exactly once by exactly one band, so the
/// result is independent of the thread count.
pub(super) fn transpose_tiles<const SOURCE: usize, const TARGET: usize>(
    source: &[[u8; SOURCE]],
    target: &mut [[u8; TARGET]],
    rows: usize,
    columns: usize,
    convert: impl Fn([u8; SOURCE]) -> [u8; TARGET] + Copy + Send + Sync,
) {
    if rows == 0 || columns == 0 {
        return;
    }
    let blocks = columns.div_ceil(TILE_EDGE);
    let threads = if rows.saturating_mul(columns) < PARALLEL_MIN_SCALARS {
        1
    } else {
        std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(blocks)
    };
    // Whole blocks per band keep every band's tiles identical to the serial walk.
    let band_columns = blocks.div_ceil(threads) * columns.min(TILE_EDGE);
    std::thread::scope(|scope| {
        for (band, target_band) in target.chunks_mut(rows * band_columns).enumerate() {
            let first_column = band * band_columns;
            let mut work = move || {
                transpose_band(source, target_band, rows, columns, first_column, convert);
            };
            if threads == 1 {
                work();
            } else {
                scope.spawn(work);
            }
        }
    });
}

/// Fills the target rows of source columns `first_column..` one tile block at a time.
///
/// `target_band` holds whole target rows, starting with the row of source column
/// `first_column`. It is processed in blocks of `TILE_EDGE` target rows (source
/// columns). Within a block, each band of at most `TILE_EDGE` source rows is
/// borrowed once and walked with `chunks_exact`, filling the matching segment of
/// every target row in the block with a sequential write stream. Slice bounds
/// hold because `row_end <= rows` and every visited column is `< columns`.
fn transpose_band<const SOURCE: usize, const TARGET: usize>(
    source: &[[u8; SOURCE]],
    target_band: &mut [[u8; TARGET]],
    rows: usize,
    columns: usize,
    first_column: usize,
    convert: impl Fn([u8; SOURCE]) -> [u8; TARGET] + Copy,
) {
    let columns_per_block = columns.min(TILE_EDGE);
    for (block, target_block) in target_band.chunks_mut(rows * columns_per_block).enumerate() {
        let column_start = first_column + block * columns_per_block;
        for row_start in (0..rows).step_by(TILE_EDGE) {
            let row_end = (row_start + TILE_EDGE).min(rows);
            let source_rows = &source[row_start * columns..row_end * columns];
            for (local_column, target_row) in target_block.chunks_exact_mut(rows).enumerate() {
                // Target row `column` holds source column `column`.
                let column = column_start + local_column;
                let segment = &mut target_row[row_start..row_end];
                for (output, source_row) in
                    segment.iter_mut().zip(source_rows.chunks_exact(columns))
                {
                    *output = convert(source_row[column]);
                }
            }
        }
    }
}

/// Views a byte buffer as whole fixed-width scalars, rejecting a partial tail.
fn whole_scalars<'a, const WIDTH: usize>(
    tensor: &str,
    bytes: &'a [u8],
) -> Result<&'a [[u8; WIDTH]]> {
    let (scalars, tail) = bytes.as_chunks::<WIDTH>();
    if tail.is_empty() {
        Ok(scalars)
    } else {
        Err(payload_bounds(tensor))
    }
}

/// Mutable form of [`whole_scalars`].
fn whole_scalars_mut<'a, const WIDTH: usize>(
    tensor: &str,
    bytes: &'a mut [u8],
) -> Result<&'a mut [[u8; WIDTH]]> {
    let (scalars, tail) = bytes.as_chunks_mut::<WIDTH>();
    if tail.is_empty() {
        Ok(scalars)
    } else {
        Err(payload_bounds(tensor))
    }
}

/// Creates a contextual out-of-bounds payload error.
fn payload_bounds(tensor: &str) -> CheckpointError {
    CheckpointError::InvalidTensorPayload {
        tensor: tensor.to_owned(),
        reason: "computed scalar range is outside the payload".into(),
    }
}

/// Reports a failed output reservation of `elements` scalars of `width` bytes.
fn allocation_failed(elements: usize, width: usize) -> CheckpointError {
    CheckpointError::AllocationFailed {
        bytes: (elements as u64).saturating_mul(width as u64),
        context: "allocating a transformed target tensor",
    }
}

#[cfg(test)]
#[path = "convert_tests.rs"]
mod tests;
