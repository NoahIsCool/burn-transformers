//! Physical storage order of checkpoint matrices installed through a transposing mapper.
//!
//! A checkpoint stores a projection weight as `C` with shape `[output, input]`,
//! while [`crate::operations::linear`] computes `y = x · W` with the logical
//! weight `W = Cᵀ` of shape `[input, output]`. Burn's `LinearLayout::Col`, which
//! the shared attention, feed-forward, and expert modules use, produces `W` with a
//! load mapper that transposes the incoming tensor. A transpose is a metadata
//! operation on every Burn backend, so by default the installed weight is a
//! *strided view* of the checkpoint buffer: `W[i, o]` lives at element
//! `o · input + i` (strides `[1, input]`) instead of `i · output + o` (strides
//! `[output, 1]`).
//!
//! Whether a matrix product reads that view as fast as contiguous `[input,
//! output]` storage is a property of the backend's GEMM, measured with the
//! `linear_layout` micro-benchmarks in `tests/small_model_benchmarks.rs` and
//! recorded in `BENCHMARKS.md`:
//!
//! - Burn Flex (CPU) runs the `gemm` crate, whose single-row (decode) products
//!   with a strided right-hand side took about twice as long as with contiguous
//!   storage for every Qwen3.5-0.8B projection shape (for example 608 µs versus
//!   297 µs for `1 × 1024 · 1024 × 4096` in F32); multi-row prefill products were
//!   within run-to-run noise of each other.
//! - CubeCL on Metal reads the strided view natively: single-row F32 products were
//!   20-27% *faster* strided, and F16 single-row products were equal or faster.
//!
//! [`stored_for_mapper`] therefore materializes contiguous `[input, output]`
//! storage only on backends listed by [`prefers_contiguous_matrices`] (Flex
//! without autodiff) and keeps the zero-copy view everywhere else.
//!
//! # Values are never changed
//!
//! On the contiguous path the checkpoint bytes are transposed on the host into
//! `Cᵀ`, stored contiguously as a `[input, output]` device tensor `S`, and the
//! mapper receives the view `Sᵀ`. That view has exactly the checkpoint's shape
//! and element values (`Sᵀ[o, i] = C[o, i]`, a pure permutation of bytes with no
//! arithmetic), so for **any** load mapper `f`, `f(Sᵀ)` and `f(C)` are the same
//! logical tensor; only the physical layout of the result differs. For the
//! transposing mapper `f(Sᵀ) = (Sᵀ)ᵀ = S`, which is contiguous. The decision
//! procedure below can therefore only affect speed, never the installed values,
//! the saved record (the save mapper still emits `[output, input]`), gradients, or
//! dtype. Matrix products may round differently with a different operand layout
//! only through floating-point reassociation inside the backend's GEMM.
//!
//! # Detecting a transposing mapper
//!
//! Burn exposes a parameter's mapper only as an opaque closure. The probe in
//! [`mapper_transposes`] loads a `[2, 3]` zero tensor into a *clone* of the
//! parameter (a lazy clone never runs the initializer) and checks whether the
//! mapped value has shape `[3, 2]`. Distinct dimensions make that shape test
//! unambiguous for square checkpoint matrices too, and no value is read back.
//! A mapper that accepts only one shape (for example `reshape([4, 4])`) panics on
//! the probe; the panic is caught, the default panic hook still reports it on
//! standard error, and the install falls back to checkpoint order.
//!
//! # Cost
//!
//! The probe costs one six-element device tensor per eligible install. The host
//! transpose is a cache-tiled copy split across available cores for large
//! matrices, and it holds one additional host copy of the tensor being installed
//! until the device tensor is built.

use std::{
    any::TypeId,
    panic::{AssertUnwindSafe, catch_unwind},
};

use burn::{
    module::{Module, Param, ParamId},
    tensor::{Tensor, TensorData, backend::Backend},
};

use super::convert::transpose_tiles;

/// Shape of the transposition probe; distinct axes make a transpose visible.
const PROBE_SHAPE: [usize; 2] = [2, 3];

/// Builds the device tensor that `parameter`'s load mapper should receive for `data`.
///
/// The result always has `data`'s shape, dtype, and element values. When the
/// backend prefers contiguous matrices, `data` is a dense rank-2 float matrix
/// with both dimensions at least two, and the parameter's mapper transposes,
/// the result is the transposed view of a contiguous host-transposed copy, so
/// the mapper produces contiguous storage (see the module documentation).
/// Otherwise `data` is uploaded unchanged. Packed `QFloat` data is never
/// reordered: its block scales are tied to the checkpoint's row-major layout.
pub(super) fn stored_for_mapper<B: Backend, const D: usize>(
    parameter: &Param<Tensor<B, D>>,
    data: TensorData,
    device: &B::Device,
) -> Tensor<B, D> {
    let dtype = data.dtype;
    match host_transposed_matrix(parameter, &data, device) {
        Some(transposed) => Tensor::<B, D>::from_data(transposed, (device, dtype)).transpose(),
        None => Tensor::<B, D>::from_data(data, (device, dtype)),
    }
}

/// Returns `Cᵀ` as `[columns, rows]` data when contiguous storage should be built.
///
/// Returns `None` for every case that keeps checkpoint order: a backend without
/// the preference, a rank other than two, a dimension below two (the two layouts
/// then coincide), a non-float or packed dtype, an element width without a
/// transpose kernel, or a mapper that does not transpose.
fn host_transposed_matrix<B: Backend, const D: usize>(
    parameter: &Param<Tensor<B, D>>,
    data: &TensorData,
    device: &B::Device,
) -> Option<TensorData> {
    if D != 2 || !prefers_contiguous_matrices::<B>(device) || !data.dtype.is_float() {
        return None;
    }
    let [rows, columns] = data.shape.as_slice() else {
        return None;
    };
    let (rows, columns) = (*rows, *columns);
    if rows < 2 || columns < 2 || !mapper_transposes(parameter, device) {
        return None;
    }
    let bytes = transpose_bytes(data.as_bytes(), rows, columns, data.dtype.size())?;
    Some(TensorData::from_bytes_vec(
        bytes,
        [columns, rows],
        data.dtype,
    ))
}

/// Whether `B` multiplies by contiguous matrices markedly faster than by transposed views.
///
/// True exactly for the inference Burn Flex backend, identified by its float
/// tensor primitive so every `Flex<F, I>` instantiation qualifies while
/// `Autodiff<Flex>` (whose primitive differs) keeps checkpoint order and thus
/// its unchanged training-time layout. Other backends keep the zero-copy view;
/// see the module documentation for the measurements behind both choices.
pub(super) fn prefers_contiguous_matrices<B: Backend>(device: &B::Device) -> bool {
    if B::ad_enabled(device) {
        return false;
    }
    is_flex_primitive::<B::FloatTensorPrimitive>()
}

/// Whether `P` is the float tensor primitive of the Burn Flex CPU backend.
#[cfg(feature = "flex")]
fn is_flex_primitive<P: 'static>() -> bool {
    TypeId::of::<P>() == TypeId::of::<burn::backend::flex::FlexTensor>()
}

/// Without this crate's `flex` feature, no backend is known to prefer contiguous matrices.
#[cfg(not(feature = "flex"))]
fn is_flex_primitive<P: 'static>() -> bool {
    let _ = TypeId::of::<P>();
    false
}

/// Whether the parameter's load mapper turns a `[2, 3]` probe into a `[3, 2]` tensor.
///
/// The probe loads into a clone, so the parameter itself is untouched and a lazy
/// parameter stays uninitialized. A mapper that panics on the probe's shape
/// counts as not transposing; see the module documentation.
fn mapper_transposes<B: Backend, const D: usize>(
    parameter: &Param<Tensor<B, D>>,
    device: &B::Device,
) -> bool {
    if D != 2 {
        return false;
    }
    let dims: [usize; D] = core::array::from_fn(|axis| PROBE_SHAPE[axis]);
    let probe = catch_unwind(AssertUnwindSafe(|| {
        let tensor = Tensor::<B, D>::zeros(dims, device);
        parameter
            .clone()
            .load_record(Param::initialized(ParamId::new(), tensor))
            .val()
            .dims()
    }));
    matches!(probe, Ok(mapped) if mapped[..] == [PROBE_SHAPE[1], PROBE_SHAPE[0]])
}

/// Transposes a native-endian row-major `[rows, columns]` matrix of `width`-byte scalars.
///
/// Returns `None` for a width without a kernel or a buffer that is not exactly
/// `rows · columns` scalars; callers then keep checkpoint order.
fn transpose_bytes(source: &[u8], rows: usize, columns: usize, width: usize) -> Option<Vec<u8>> {
    match width {
        2 => transpose_scalars::<2>(source, rows, columns),
        4 => transpose_scalars::<4>(source, rows, columns),
        8 => transpose_scalars::<8>(source, rows, columns),
        _ => None,
    }
}

/// Monomorphic body of [`transpose_bytes`] for `WIDTH`-byte scalars.
fn transpose_scalars<const WIDTH: usize>(
    source: &[u8],
    rows: usize,
    columns: usize,
) -> Option<Vec<u8>> {
    let (scalars, tail) = source.as_chunks::<WIDTH>();
    if !tail.is_empty() || Some(scalars.len()) != rows.checked_mul(columns) {
        return None;
    }
    let mut target = vec![[0_u8; WIDTH]; scalars.len()];
    transpose_tiles(scalars, &mut target, rows, columns, |scalar| scalar);
    Some(target.into_flattened())
}

#[cfg(test)]
#[path = "storage_order_tests.rs"]
mod tests;
