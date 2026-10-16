//! Quantization-preserving transformer operations.
//!
//! Burn's high-level linear module currently extracts a full-precision
//! primitive from a quantized tensor before dispatch. In contrast, tensor
//! matrix multiplication retains [`burn::tensor::TensorPrimitive::QFloat`] and
//! lets CubeCL consume packed values and scales inside its matmul kernel. This
//! module supplies the narrow projection boundary needed by transformer layers
//! while continuing to store parameters in Burn's ordinary [`burn::nn::Linear`]
//! modules.
//!
//! For an input with shape `[..., I]`, weight `[I, O]`, and optional bias `[O]`,
//! [`linear`] evaluates `reshape(x, [-1, I]) @ W + b` and restores
//! `[..., O]`. Flattening only the activation's leading axes leaves packed
//! weight block boundaries and packing dimensions unchanged.
//!
//! [`matmul_transposed`] evaluates the same rank-2 product against the transpose
//! of a row-major `[output, input]` matrix, the readout `h · Eᵀ` of a tied
//! `[vocabulary, hidden]` embedding table. It never reshapes the matrix: Burn's
//! `module::linear` prepends unit axes to its weight, which Burn Flex 0.21 cannot
//! express as a view of a transposed matrix and therefore answers by copying the
//! complete table on every call. Avoiding the reshape removes that copy for the
//! dtypes whose Flex GEMM reads strided operands directly (F32, F16 and F64);
//! Burn Flex 0.21 still converts BF16 operands to contiguous F32 on every
//! product, so a BF16 readout on Flex keeps a per-call conversion copy.
//!
//! Dense floating projections remain differentiable through both inputs and
//! parameters. Packed quantized projections retain their inference representation,
//! but Burn 0.21 cannot differentiate those operations, including a frozen packed
//! suffix. The autodiff path rejects packed values contextually; callers needing
//! input gradients must explicitly dequantize before constructing the float graph.
//!
//! # Storage order of initialized and cached tensors
//!
//! Transposes, axis swaps and head splits are metadata-only views on every Burn
//! backend. CubeCL kernels read such strided views natively, but Burn Flex 0.21
//! does not: its GEMM multiplies a single row by a strided right-hand matrix
//! about twice as slowly as by contiguous storage, and its attention kernels copy
//! every non-contiguous query, key and value operand on **every** call (see
//! `BENCHMARKS.md`). The crate-internal `contiguous_where_preferred` therefore
//! materializes contiguous row-major storage exactly on the backends that
//! `prefers_contiguous_storage` names (inference Flex) and returns the view
//! unchanged everywhere else. It is a pure permutation of elements: values,
//! shape and dtype are identical, and only the physical layout changes. Model
//! constructors apply it to randomly initialized column-layout projection
//! weights, and cached decoders apply it to tensors that every later decode
//! step reads (for example T5's projected cross-attention keys and values).
//! Checkpoint loads choose their storage order separately in the installer;
//! a record restored through a Burn recorder keeps Burn's strided view.
//!
//! # In-place region writes
//!
//! Decode caches write a few new tokens into large preallocated planes
//! (`[batch, heads, capacity, head_dim]` keys and values, recurrent states).
//! Burn's `slice_assign` on a uniquely owned tensor is free to reuse its
//! buffer, but Burn Flex 0.21 never does: its slice writer calls
//! `to_contiguous(&tensor)`, which clones the buffer's `Arc` while the
//! original binding is still alive, so the following `Arc::make_mut` always
//! copies the *whole* destination. Every one-token write therefore cost
//! `O(batch · heads · capacity · head_dim)` (about 1.3 ms for a 64 MiB plane,
//! linear in capacity) and allocated a fresh plane. The crate-internal
//! `assign_region` and `zero_region` avoid that on inference Flex: they take
//! the destination's host bytes by value (zero-copy for a uniquely owned,
//! contiguous, whole-buffer tensor), copy only the region's contiguous runs,
//! and rewrap the same buffer, so a write costs `O(region)` and the plane's
//! allocation is stable. A destination that is shared (for example by a live
//! read view) or strided is first copied exactly as `slice_assign` would, so
//! no other tensor can observe the write. Every other backend, autodiff
//! included, and every packed or mismatched operand, uses `slice_assign`.
//!
//! # One-column depthwise convolution
//!
//! Recurrent mixers (Mamba-2, Gated DeltaNet) run a `K`-tap depthwise
//! convolution over `[history | new tokens]`; a one-token decode step has
//! exactly one output column, `C · K` multiply-adds. Burn Flex 0.21's
//! depthwise kernel nevertheless fans every `(batch, channel)` plane out to
//! rayon, and under CPU contention the decode thread then waits in rayon's
//! latch. The crate-internal `depthwise_conv1d` evaluates that single column
//! on inference Flex in F32 as one elementwise product and one sequential
//! middle-axis sum, which reproduces the Flex kernel's accumulation order bit
//! for bit; every other case is Burn's `conv1d`.

use core::ops::Range;

use burn::{
    module::Param,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{DType, Tensor, backend::Backend, module::conv1d, ops::ConvOptions},
};

use crate::{Error, Result};

/// Initializes a lazy projection whose stored weight is an autodiff leaf.
///
/// Burn 0.21 applies the column-layout transpose after marking its random
/// initializer as trainable. That makes the stored `[in, out]` weight an
/// intermediate node, whose gradient is consumed during backward rather than
/// retained for the optimizer. Initialization defines a parameter, not part of
/// the learned function: detaching after that layout transform establishes the
/// correct leaf without detaching any subsequent forward computation. The
/// mapper remains lazy and retains the column layout's checkpoint transpose.
/// The outer initializer receives Burn's current gradient policy so freezing
/// the parameter before its first use is respected as well.
/// Loading likewise defines new parameter leaves: Burn deserializes trainable
/// tensors before applying the checkpoint-layout transpose, so the load mapper
/// detaches that layout transformation before Burn restores the target's
/// gradient policy. This does not detach any model forward computation.
///
/// Inference backends retain exactly Burn's original initialization values.
/// A column-layout weight is drawn as `[out, in]` and exposed as its transposed
/// view, so on backends that [`prefer contiguous storage`](prefers_contiguous_storage)
/// the lazy initializer additionally copies that view once into contiguous
/// `[in, out]` storage ([`contiguous_where_preferred`]). The copy runs at the
/// weight's first read (a checkpoint load that replaces the weight first never
/// runs it), consumes no random numbers, keeps every value bit for bit, and
/// transiently holds two copies of that one matrix. Row-layout weights are
/// already contiguous and are left untouched.
///
/// Only random initialization is re-laid out here. The load mapper stays
/// Burn's plain transpose, because the checkpoint installer probes and applies
/// it to choose its own storage order (`checkpoint::runtime::storage_order`
/// already stores loaded weights contiguously on the same backends). A record
/// loaded through a Burn recorder therefore keeps the strided `[in, out]` view
/// of its row-major `[out, in]` payload, exactly as a plain Burn module would.
pub(crate) fn init_linear<B: Backend>(config: &LinearConfig, device: &B::Device) -> Linear<B> {
    let mut projection = config.init::<B>(device);
    if !matches!(config.layout, LinearLayout::Col) {
        return projection;
    }
    if prefers_contiguous_storage::<B>(device) {
        projection.weight = projection
            .weight
            .init_mapper(contiguous_where_preferred::<B, 2>);
    } else if B::ad_enabled(device) {
        let initialized_weight = projection.weight.set_require_grad(false);
        let shape = initialized_weight.lazy_shape();
        projection.weight = Param::uninitialized(
            initialized_weight.id,
            move |_, require_grad| {
                initialized_weight
                    .val()
                    .detach()
                    .set_require_grad(require_grad)
            },
            device.clone(),
            true,
            shape,
        )
        .save_mapper(|tensor| tensor.transpose())
        .load_mapper(|tensor| tensor.transpose().detach());
    }
    projection
}

/// Whether `B` runs matrix products and attention markedly faster on contiguous storage.
///
/// True exactly for the inference Burn Flex backend, identified by its float
/// tensor primitive so every `Flex<F, I>` instantiation qualifies. Autodiff
/// backends (including `Autodiff<Flex>`) answer `false`: a copy made outside the
/// autodiff graph would detach the copied tensor from its gradient. CubeCL
/// backends answer `false` because they read strided views natively (see the
/// module documentation and `BENCHMARKS.md`).
///
/// The checkpoint installer's `prefers_contiguous_matrices` applies the same
/// rule to loaded weights.
pub(crate) fn prefers_contiguous_storage<B: Backend>(device: &B::Device) -> bool {
    is_inference_flex::<B>(device)
}

/// Whether `B` is the inference (non-autodiff) Burn Flex CPU backend on `device`.
///
/// Flex-specific fast paths key on this: an autodiff backend wraps Flex in
/// another primitive and must keep Burn's differentiable operations, and
/// CubeCL backends have their own kernels.
pub(crate) fn is_inference_flex<B: Backend>(device: &B::Device) -> bool {
    !B::ad_enabled(device) && is_flex_primitive::<B::FloatTensorPrimitive>()
}

/// Whether `P` is the float tensor primitive of the Burn Flex CPU backend.
#[cfg(feature = "flex")]
fn is_flex_primitive<P: 'static>() -> bool {
    core::any::TypeId::of::<P>() == core::any::TypeId::of::<burn::backend::flex::FlexTensor>()
}

/// Without this crate's `flex` feature, no backend is known to prefer contiguous storage.
#[cfg(not(feature = "flex"))]
fn is_flex_primitive<P: 'static>() -> bool {
    let _ = core::any::TypeId::of::<P>();
    false
}

/// Returns the Burn Flex primitive behind a floating `tensor` on inference Flex.
///
/// The primitive is a cheap clone: it shares the tensor's host buffer (an
/// `Arc` increment) and exposes its storage, dtype, and strided layout, so
/// Flex-specific kernels can read a strided view without first copying it.
/// Returns `None` on every other backend (autodiff included) and for packed
/// tensors. The downcast is a checked `Any` conversion, not a cast.
#[cfg(feature = "flex")]
pub(crate) fn inference_flex_primitive<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
) -> Option<burn::backend::flex::FlexTensor> {
    if !is_inference_flex::<B>(&tensor.device()) {
        return None;
    }
    match tensor.clone().into_primitive() {
        burn::tensor::TensorPrimitive::Float(primitive) => {
            let primitive: Box<dyn core::any::Any> = Box::new(primitive);
            primitive
                .downcast::<burn::backend::flex::FlexTensor>()
                .ok()
                .map(|primitive| *primitive)
        }
        burn::tensor::TensorPrimitive::QFloat(_) => None,
    }
}

/// Returns `tensor` in contiguous row-major storage on backends that prefer it.
///
/// On a backend for which [`prefers_contiguous_storage`] holds, a floating
/// tensor is copied once into row-major order: Flex keeps tensors in host
/// memory, so reading the data materializes a strided view in logical order
/// and wrapping that buffer again is free (a contiguous, uniquely owned
/// tensor is moved through without copying). Every other backend, and every
/// non-floating or packed tensor, is returned unchanged. The result always has
/// the input's shape, dtype and element values; only its physical layout can
/// differ, so downstream products round differently at most through F32
/// reassociation inside the backend's kernels.
///
/// Use it for tensors that are read many times after being computed once,
/// such as cached cross-attention keys or a weight, never in a per-step hot
/// path: the copy costs one pass over the tensor.
pub(crate) fn contiguous_where_preferred<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
) -> Tensor<B, D> {
    let device = tensor.device();
    let dtype = tensor.dtype();
    if !dtype.is_float() || !prefers_contiguous_storage::<B>(&device) {
        return tensor;
    }
    Tensor::from_data(tensor.into_data(), (&device, dtype))
}

/// Writes `values` into the `ranges` region of `destination`, in place on inference Flex.
///
/// Semantically `*destination = destination.slice_assign(ranges, values)`:
/// the region `ranges[0] × … × ranges[D-1]` takes the values of `values`
/// (whose shape is the ranges' lengths) and every other element is kept. On
/// inference Flex the write costs `O(region)` instead of Flex's
/// `O(destination)` copy, and a uniquely owned, contiguous destination keeps
/// its buffer (see the module documentation). Elsewhere, and whenever the
/// operands do not qualify (packed or mismatched dtypes, a region outside the
/// destination, or an empty region), it is exactly `slice_assign`, including
/// Burn's panics for invalid ranges.
pub(crate) fn assign_region<B: Backend, const D: usize>(
    destination: &mut Tensor<B, D>,
    ranges: [Range<usize>; D],
    values: Tensor<B, D>,
) {
    destination.inplace(|tensor| {
        if region_writable_in_place(&tensor, &ranges, Some(&values)) {
            write_region(tensor, &ranges, Some(values))
        } else {
            tensor.slice_assign(ranges, values)
        }
    });
}

/// Sets the `ranges` region of `destination` to zero, in place on inference Flex.
///
/// Equivalent to [`assign_region`] with a zero tensor of the region's shape
/// and the destination's dtype, without allocating that tensor on Flex. The
/// written zeros are `+0.0`, whose bit pattern is all zero bytes in every
/// floating dtype.
pub(crate) fn zero_region<B: Backend, const D: usize>(
    destination: &mut Tensor<B, D>,
    ranges: [Range<usize>; D],
) {
    destination.inplace(|tensor| {
        if region_writable_in_place(&tensor, &ranges, None) {
            write_region(tensor, &ranges, None)
        } else {
            let shape = ranges.clone().map(|range| range.len());
            let zeros = Tensor::zeros(shape, (&tensor.device(), tensor.dtype()));
            tensor.slice_assign(ranges, zeros)
        }
    });
}

/// Whether [`write_region`] may replace `slice_assign` for this destination and region.
///
/// Requires a positive rank, inference Flex, an unpacked floating destination,
/// a non-empty region inside the destination, and (for value writes) values
/// of the destination's dtype whose shape is the region's.
fn region_writable_in_place<B: Backend, const D: usize>(
    destination: &Tensor<B, D>,
    ranges: &[Range<usize>; D],
    values: Option<&Tensor<B, D>>,
) -> bool {
    let dtype = destination.dtype();
    let shape = destination.dims();
    D > 0
        && is_inference_flex::<B>(&destination.device())
        && dtype.is_float()
        && !matches!(dtype, DType::QFloat(_))
        && ranges
            .iter()
            .zip(shape)
            .all(|(range, extent)| range.start < range.end && range.end <= extent)
        && values.is_none_or(|values| {
            values.dtype() == dtype
                && values
                    .dims()
                    .iter()
                    .zip(ranges)
                    .all(|(&length, range)| length == range.len())
        })
}

/// Writes `values` (or zeros) into a validated region through the destination's host bytes.
///
/// `destination.into_data()` hands over Flex's buffer without copying when the
/// tensor is uniquely owned, contiguous, and spans its whole buffer, and
/// copies it otherwise (which is what keeps shared views unchanged). The
/// region is copied run by run (see [`region_runs`]), and
/// `Tensor::from_data` rewraps the same buffer without converting it, because
/// the dtype is unchanged. Should the host buffers ever disagree with the
/// shapes, the write falls back to `slice_assign` on the rebuilt tensor.
fn write_region<B: Backend, const D: usize>(
    destination: Tensor<B, D>,
    ranges: &[Range<usize>; D],
    values: Option<Tensor<B, D>>,
) -> Tensor<B, D> {
    let device = destination.device();
    let dtype = destination.dtype();
    let shape = destination.dims();
    let element = dtype.size();
    let mut data = destination.into_data();
    let source = values.map(Tensor::into_data);
    let region_elements: usize = ranges.iter().map(Range::len).product();
    let consistent = data.bytes.len() == shape.iter().product::<usize>() * element
        && source
            .as_ref()
            .is_none_or(|source| source.bytes.len() == region_elements * element);
    if !consistent {
        let rebuilt = Tensor::<B, D>::from_data(data, (&device, dtype));
        let values = match source {
            Some(source) => Tensor::from_data(source, (&device, dtype)),
            None => Tensor::zeros(ranges.clone().map(|range| range.len()), (&device, dtype)),
        };
        return rebuilt.slice_assign(ranges.clone(), values);
    }
    let (run, offsets) = region_runs(&shape, ranges);
    let run_bytes = run * element;
    let destination_bytes: &mut [u8] = &mut data.bytes;
    for (index, offset) in offsets.into_iter().enumerate() {
        let target = &mut destination_bytes[offset * element..offset * element + run_bytes];
        match &source {
            Some(source) => {
                target.copy_from_slice(&source.bytes[index * run_bytes..(index + 1) * run_bytes]);
            }
            None => target.fill(0),
        }
    }
    Tensor::from_data(data, (&device, dtype))
}

/// Splits a region of a row-major tensor into equal contiguous runs.
///
/// Returns `(run, offsets)`: every run holds `run` consecutive elements, and
/// `offsets[j]` is the element offset in the destination of the run that holds
/// region elements `[j · run, (j + 1) · run)` in row-major region order, so the
/// `j`-th run of a contiguous source tensor of the region's shape is its
/// elements `[j · run, (j + 1) · run)`.
///
/// Let `k` be the first axis after which every axis is fully covered
/// (`ranges[i] = 0..shape[i]` for all `i > k`). One run then spans
/// `len(ranges[k]) · Π_{i>k} shape[i]` elements, and there is one run per
/// index tuple of the axes before `k` inside their ranges. A one-token write
/// `[rows, 0..H, t..t+1, 0..d]` into `[B, H, C, d]` is `rows · H` runs of `d`
/// elements; a whole-batch state write is a single run. Requires non-empty
/// ranges inside `shape` ([`region_writable_in_place`]).
fn region_runs<const D: usize>(
    shape: &[usize; D],
    ranges: &[Range<usize>; D],
) -> (usize, Vec<usize>) {
    let mut strides = [1_usize; D];
    for axis in (0..D.saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    // The innermost axis that is not fully covered; axis 0 when all are.
    let split = (0..D)
        .rev()
        .find(|&axis| ranges[axis].start != 0 || ranges[axis].end != shape[axis])
        .unwrap_or(0);
    let run = ranges[split].len() * strides[split];
    let base = ranges[split].start * strides[split];
    let mut offsets = vec![base];
    for axis in (0..split).rev() {
        offsets = ranges[axis]
            .clone()
            .flat_map(|index| {
                offsets
                    .iter()
                    .map(move |offset| index * strides[axis] + offset)
            })
            .collect();
    }
    (run, offsets)
}

/// Depthwise one-dimensional convolution with no padding, unit stride, and unit dilation.
///
/// `input` is `[batch, channels, width]`, `weight` the depthwise
/// `[channels, 1, kernel]` (with `groups == channels`), and `bias` an optional
/// `[channels]`; the result is `[batch, channels, width - kernel + 1]` with
/// `y[b, c, t] = bias[c] + Σ_k weight[c, 0, k] · input[b, c, t + k]`. It equals
/// Burn's `conv1d` with `ConvOptions::new([1], [0], [1], groups)`, which it
/// calls in every case but one.
///
/// The exception is a single output column (`width == kernel`, the one-token
/// decode step) on inference Flex with F32 operands and `groups == channels`.
/// Flex's depthwise kernel would spawn one rayon task per `(batch, channel)`
/// for `kernel` multiply-adds each. Instead, the products `x · w` are one
/// elementwise multiplication of `[batch, channels · kernel]` by a shared
/// `[1, channels · kernel]` weight row (Flex's SIMD shared-row path), and the
/// taps are summed as the middle axis of `[batch · channels, kernel, 1]`,
/// which Flex accumulates sequentially from zero. Flex's kernel computes
/// `((0 + w₀x₀) + w₁x₁) + …` and then adds the bias, so the result is bit
/// for bit the same (tested); this is not a reassociated sum. F16 and BF16
/// keep `conv1d`, whose per-tap rounding the F32-accumulating reduction would
/// not reproduce.
pub(crate) fn depthwise_conv1d<B: Backend>(
    input: Tensor<B, 3>,
    weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 1>>,
    groups: usize,
) -> Tensor<B, 3> {
    let [batch, channels, width] = input.dims();
    let [output_channels, channels_per_group, kernel] = weight.dims();
    let single_column = width == kernel
        && groups == channels
        && output_channels == channels
        && channels_per_group == 1
        && input.dtype() == DType::F32
        && weight.dtype() == DType::F32
        && bias
            .as_ref()
            .is_none_or(|bias| bias.dtype() == DType::F32 && bias.dims() == [channels])
        && is_inference_flex::<B>(&input.device());
    if !single_column {
        return conv1d(input, weight, bias, ConvOptions::new([1], [0], [1], groups));
    }
    let taps = channels * kernel;
    let products = input.reshape([batch, taps]) * weight.reshape([1, taps]);
    let summed = products
        .reshape([batch * channels, kernel, 1])
        .sum_dim(1)
        .reshape([batch, channels]);
    let output = match bias {
        Some(bias) => summed + bias.reshape([1, channels]),
        None => summed,
    };
    output.reshape([batch, channels, 1])
}

/// Applies a Burn linear module without dequantizing a packed weight up front.
///
/// Full-precision weights take the ordinary backend matmul path. A `QFloat`
/// weight reaches the backend's quantized matmul dispatch, which CubeCL can
/// execute from packed Q8/Q4 storage. Activations deliberately remain floating
/// point in v1; quantized activations are rejected because reshaping sub-byte
/// tensors is not generally implemented by Burn 0.21. Floating weights and bias
/// must match the activation dtype, and all operands must share a device.
///
/// Packed weights are inference-only in Burn 0.21: its autodiff backend cannot
/// differentiate even the activation input of a frozen packed projection. This
/// boundary rejects such execution before backend dispatch. To train an adapter
/// before a quantized checkpoint weight, explicitly dequantize that weight on
/// the inner backend, lift the floating tensor into autodiff, and freeze it.
/// That workflow uses floating storage and computation during training.
pub fn linear<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    projection: &Linear<B>,
) -> Result<Tensor<B, D>> {
    if D == 0 {
        return Err(Error::shape(
            "linear.input",
            "a tensor with at least one dimension",
            "rank zero",
        ));
    }
    let mut output_shape = input.dims();
    if output_shape.contains(&0) {
        return Err(Error::shape(
            "linear.input",
            "non-zero dimensions ending in the projection input width",
            format!("{output_shape:?}"),
        ));
    }
    if matches!(input.dtype(), DType::QFloat(_)) {
        return Err(Error::BackendCapability(
            "v1 packed linear supports floating activations with a quantized weight, not quantized activations"
                .into(),
        ));
    }

    let weight = projection.weight.val();
    if matches!(weight.dtype(), DType::QFloat(_)) && B::ad_enabled(&input.device()) {
        return Err(Error::BackendCapability(
            "Burn 0.21 autodiff does not support packed Q4/Q8 linear weights, including input gradients through frozen weights; explicitly dequantize on the inner backend before constructing the autodiff parameter"
                .into(),
        ));
    }
    if weight.device() != input.device() {
        return Err(Error::configuration(
            "linear.weight",
            "projection weight and activations must share a device",
        ));
    }
    if !matches!(weight.dtype(), DType::QFloat(_)) && weight.dtype() != input.dtype() {
        return Err(Error::BackendCapability(format!(
            "linear.weight dtype {:?} must match activation dtype {:?}; cast parameters or activations explicitly",
            weight.dtype(),
            input.dtype()
        )));
    }
    #[cfg(all(
        feature = "metal",
        any(feature = "rocm", feature = "attention-kernel-tracing")
    ))]
    if packed_metal_autotune_is_unavailable(weight.dtype(), &B::name(&weight.device())) {
        return Err(Error::BackendCapability(
            "packed Q4/Q8 Metal matmul is unavailable when CubeCL autotuning is enabled (the `rocm` or `attention-kernel-tracing` crate feature): Burn 0.21 feature unification enables a CubeCL autotuner with quantized-unsafe candidates; build the Metal runtime with `--no-default-features --features std,metal` (plus optional tokenizer features)"
                .into(),
        ));
    }
    let [weight_input, weight_output] = weight.dims();
    let input_width = output_shape[D - 1];
    if weight_input != input_width || weight_output == 0 {
        return Err(Error::shape(
            "linear.weight",
            format!("[{input_width}, nonzero_output]"),
            format!("[{weight_input}, {weight_output}]"),
        ));
    }
    if let Some(bias) = &projection.bias {
        let [bias_width] = bias.shape().dims();
        if bias_width != weight_output {
            return Err(Error::shape(
                "linear.bias",
                format!("[{weight_output}]"),
                format!("[{bias_width}]"),
            ));
        }
        if bias.lazy_device() != input.device() {
            return Err(Error::configuration(
                "linear.bias",
                "projection bias and activations must share a device",
            ));
        }
        if bias.val().dtype() != input.dtype() {
            return Err(Error::BackendCapability(format!(
                "linear.bias dtype {:?} must match activation dtype {:?}",
                bias.val().dtype(),
                input.dtype()
            )));
        }
    }
    let rows = leading_rows("linear.input", &output_shape)?;
    output_shape[D - 1] = weight_output;

    let mut output = input.reshape([rows, input_width]).matmul(weight);
    if let Some(bias) = &projection.bias {
        output = output + bias.val().unsqueeze();
    }
    Ok(output.reshape(output_shape))
}

/// Multiplies the last axis of `input` by the transpose of a `[output, input]` matrix.
///
/// For `x` of shape `[..., I]` and a row-major matrix `M` of shape `[O, I]`, this
/// returns `y = reshape(x, [-1, I]) · Mᵀ` restored to `[..., O]`, that is
/// `y[..., o] = Σ_i x[..., i] · M[o, i]`. With `M` the `[vocabulary, hidden]`
/// embedding table this is the tied language-model readout.
///
/// The product is one rank-2 matmul whose right operand is the transposed *view*
/// of `M`, and the matrix is never reshaped. Burn's `module::linear(x, Mᵀ)`
/// instead reshapes the weight to `[1, ..., I, O]`, which Burn Flex 0.21 can only
/// perform by materializing `Mᵀ` contiguously on every call (113 MB per decode
/// step for SmolLM-135M's table, about two thirds of that model's Flex decode
/// time). With the reshape gone, Flex's F32, F16 and F64 GEMM reads the strided
/// view without copying it. Burn Flex 0.21's BF16 matmul still makes both
/// operands contiguous and converts them to F32 on every product, so a BF16
/// matrix is copied per call on Flex regardless of this formulation; CubeCL
/// backends read the transposed view through their own stride handling.
/// Flattening the activation's leading axes is free for the contiguous hidden
/// states models produce. Values equal `module::linear(x, Mᵀ, None)` up to F32
/// reassociation inside the backend's GEMM.
///
/// The matrix must be floating point with the activation's dtype and device;
/// gradients flow to both operands.
///
/// # Errors
///
/// Returns [`Error::Shape`] for a rank-zero or empty input, or a matrix whose
/// second axis differs from the input width or whose first axis is empty;
/// [`Error::BackendCapability`] for quantized operands or mismatched dtypes; and
/// [`Error::Configuration`] when the operands live on different devices.
pub fn matmul_transposed<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    matrix: Tensor<B, 2>,
) -> Result<Tensor<B, D>> {
    if D == 0 {
        return Err(Error::shape(
            "matmul_transposed.input",
            "a tensor with at least one dimension",
            "rank zero",
        ));
    }
    let mut output_shape = input.dims();
    if output_shape.contains(&0) {
        return Err(Error::shape(
            "matmul_transposed.input",
            "non-zero dimensions ending in the matrix input width",
            format!("{output_shape:?}"),
        ));
    }
    if matches!(input.dtype(), DType::QFloat(_)) || matches!(matrix.dtype(), DType::QFloat(_)) {
        return Err(Error::BackendCapability(
            "matmul_transposed multiplies floating operands; dequantize packed tensors first"
                .into(),
        ));
    }
    if matrix.dtype() != input.dtype() {
        return Err(Error::BackendCapability(format!(
            "matmul_transposed.matrix dtype {:?} must match activation dtype {:?}",
            matrix.dtype(),
            input.dtype()
        )));
    }
    if matrix.device() != input.device() {
        return Err(Error::configuration(
            "matmul_transposed.matrix",
            "matrix and activations must share a device",
        ));
    }
    let [matrix_output, matrix_input] = matrix.dims();
    let input_width = output_shape[D - 1];
    if matrix_input != input_width || matrix_output == 0 {
        return Err(Error::shape(
            "matmul_transposed.matrix",
            format!("[nonzero_output, {input_width}]"),
            format!("[{matrix_output}, {matrix_input}]"),
        ));
    }
    let rows = leading_rows("matmul_transposed.input", &output_shape)?;
    output_shape[D - 1] = matrix_output;
    Ok(input
        .reshape([rows, input_width])
        .matmul(matrix.transpose())
        .reshape(output_shape))
}

/// Returns the product of every axis but the last, the row count of a flattened input.
///
/// # Errors
///
/// Returns a shape error naming `tensor` when the product overflows `usize`.
pub(crate) fn leading_rows(tensor: &str, shape: &[usize]) -> Result<usize> {
    shape[..shape.len().saturating_sub(1)]
        .iter()
        .try_fold(1_usize, |product, dimension| {
            product.checked_mul(*dimension)
        })
        .ok_or_else(|| Error::shape(tensor, "representable leading axes", "overflow"))
}

/// Detects Burn 0.21's known packed Metal/autotune feature-unification hazard.
///
/// Two crate features activate `burn-cubecl/autotune` for the shared CubeCL
/// dependency: `rocm` (through Burn's ROCm backend) and the test-only
/// `attention-kernel-tracing` (through `burn/autotune`, which Burn forwards to
/// `burn-wgpu/autotune` and from there to `burn-cubecl/autotune`). Cargo then
/// applies that strategy to Metal as well, where Burn 0.21 can benchmark
/// quantized-incompatible candidate kernels. The backend name check limits the
/// guard to Metal; non-Metal execution keeps its independently validated path.
/// Both Q4 and Q8 use packed U32 storage and can enter candidates whose logical
/// dequantized vector width differs from the floating activation width.
#[cfg(all(
    feature = "metal",
    any(feature = "rocm", feature = "attention-kernel-tracing")
))]
fn packed_metal_autotune_is_unavailable(dtype: DType, backend_name: &str) -> bool {
    matches!(dtype, DType::QFloat(_)) && backend_name.contains("wgpu<msl>")
}

#[cfg(test)]
#[path = "operations_tests.rs"]
mod tests;
