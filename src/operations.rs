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
//! Dense floating projections remain differentiable through both inputs and
//! parameters. Packed quantized projections retain their inference representation,
//! but Burn 0.21 cannot differentiate those operations, including a frozen packed
//! suffix. The autodiff path rejects packed values contextually; callers needing
//! input gradients must explicitly dequantize before constructing the float graph.

use burn::{
    module::Param,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{DType, Tensor, backend::Backend},
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
/// Ordinary inference backends retain exactly Burn's original initialization.
pub(crate) fn init_linear<B: Backend>(config: &LinearConfig, device: &B::Device) -> Linear<B> {
    let mut projection = config.init::<B>(device);
    if B::ad_enabled(device) && matches!(config.layout, LinearLayout::Col) {
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
    #[cfg(all(feature = "metal", feature = "rocm"))]
    if packed_metal_autotune_is_unavailable(weight.dtype(), &B::name(&weight.device())) {
        return Err(Error::BackendCapability(
            "packed Q4/Q8 Metal matmul is unavailable when the `rocm` crate feature is also enabled: Burn 0.21 feature unification enables a CubeCL autotuner with quantized-unsafe candidates; build the Metal runtime with `--no-default-features --features std,metal` (plus optional tokenizer features)"
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
    let rows = output_shape[..D - 1]
        .iter()
        .try_fold(1_usize, |product, dimension| {
            product.checked_mul(*dimension)
        })
        .ok_or_else(|| Error::shape("linear.input", "representable leading axes", "overflow"))?;
    output_shape[D - 1] = weight_output;

    let mut output = input.reshape([rows, input_width]).matmul(weight);
    if let Some(bias) = &projection.bias {
        output = output + bias.val().unsqueeze();
    }
    Ok(output.reshape(output_shape))
}

/// Detects Burn 0.21's known packed Metal/autotune feature-unification hazard.
///
/// Enabling this crate's ROCm feature activates `burn-cubecl/autotune` for the
/// shared CubeCL dependency. Cargo then applies that strategy to Metal as well,
/// where Burn 0.21 can benchmark quantized-incompatible candidate kernels. The
/// backend name check limits the guard to Metal; non-Metal execution keeps its
/// independently validated path. Both Q4 and Q8 use packed U32 storage and can
/// enter candidates whose logical dequantized vector width differs from the
/// floating activation width.
#[cfg(all(feature = "metal", feature = "rocm"))]
fn packed_metal_autotune_is_unavailable(dtype: DType, backend_name: &str) -> bool {
    matches!(dtype, DType::QFloat(_)) && backend_name.contains("wgpu<msl>")
}

#[cfg(test)]
#[path = "operations_tests.rs"]
mod tests;
