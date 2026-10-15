//! Exact installation of packed block-QFloat weights into mapped Burn parameters.
//!
//! A packed sidecar stores `[output, input]` matrices whose `[1, G]` blocks each
//! carry one scale, so the artifact defines `W_hat[o, i] = q[o, i] * s[o, i / G]`.
//! Column-layout parameters (`LinearLayout::Col`) install that matrix through a
//! load mapper that transposes it. Whether a transpose preserves `W_hat` is a
//! backend property: CubeCL backends keep a strided view of the same blocks,
//! while Burn Flex 0.21 dequantizes, transposes, and *requantizes* with fresh
//! scales over contiguous `G`-element runs of the transposed buffer. The second
//! rounding roughly doubles quantization error, and when `output % G != 0` its
//! blocks also straddle rows, so the installed weight is no longer the artifact.
//!
//! [`install_packed_parameter`] detects both conditions generically, with no
//! backend-name branches:
//!
//! - [`packed_layout_is_exact`] runs one tiny probe per backend type: it builds a
//!   `[2, 64]` Q8 tensor whose rows use scales 100x apart, and compares
//!   `dequantize(transpose(x))` with `transpose(dequantize(x))` on the backend
//!   itself, so the result is independent of the backend's float dtype;
//! - a mapper probe applies the parameter's own load mapper to a tiny float
//!   tensor with pairwise-distinct dimensions `2, 3, 5, ...` and distinct signed
//!   values, then classifies it as the identity, a pure axis permutation, or
//!   anything else (value arithmetic, casts, reshapes).
//!
//! Only a pure permutation on a requantizing backend, or a mapper of the third
//! kind on any backend, is inexact; then the [`PackedLayoutPolicy`] decides.
//! Dequantizing first installs exactly `mapper(W_hat)` in float storage (one
//! `q * s` product per element, then the mapper's float arithmetic), at the cost
//! of the packed memory savings for that parameter.
//!
//! Every ordinary install precondition is checked before either probe runs, so a
//! rejected tensor never reaches a backend constructor. That ordering matters on
//! autodiff backends: Burn 0.21 panics there when asked to build QFloat data,
//! while this module returns [`ParameterInstallError::QuantizedAutodiff`].

use std::{
    any::TypeId,
    collections::BTreeMap,
    sync::{Mutex, PoisonError},
};

use burn::{
    module::{Module, Param, ParamId},
    tensor::{
        DType, Tensor, TensorData,
        backend::{Backend, DTypeUsage},
    },
};

use super::{
    ParameterInstallError, install_float_parameter, install_validated_parameter,
    load_parameter_tensor, validate_parameter_install,
};
use crate::checkpoint::quantization::{
    QuantizationGroupSize, SymmetricQuantization, WeightQuantizationConfig, quantize_tensor_data,
};

/// How to install packed data when its exact blocks would not survive the mapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PackedLayoutPolicy {
    /// Install the packed tensor and let the mapper run, as
    /// [`install_float_parameter`] does; the result may approximate the artifact.
    #[default]
    BackendDefault,
    /// Install exactly the mapped dequantized values in float storage.
    DequantizeWhenInexact,
    /// Return [`ParameterInstallError::InexactPackedLayout`] instead.
    RequireExact,
}

/// How a packed tensor ended up installed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PackedInstall {
    /// Installed as packed QFloat with the artifact's exact blocks.
    Packed,
    /// Installed as packed QFloat through an axis-permuting mapper that this
    /// backend requantizes ([`packed_layout_is_exact`] is `false`).
    ///
    /// Returned only under [`PackedLayoutPolicy::BackendDefault`].
    Requantized,
    /// Installed as packed QFloat through a load mapper that is neither the
    /// identity nor a pure axis permutation (for example value scaling or a
    /// cast), so the stored blocks are whatever the backend's packed arithmetic
    /// produces.
    ///
    /// Returned only under [`PackedLayoutPolicy::BackendDefault`].
    Transformed,
    /// Installed as exact dequantized float values.
    Dequantized,
    /// The data was not packed; it took the ordinary float path.
    Float,
}

/// What a parameter's load mapper does, as observed on a small probe tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MapperEffect {
    /// The mapper returns its input unchanged (including "no mapper").
    Identity,
    /// The mapper permutes axes without changing values or dtype, as the
    /// transpose of `LinearLayout::Col` does.
    AxisPermutation,
    /// Anything else: value arithmetic, dtype casts, or non-permuting reshapes.
    Other,
}

/// Installs checkpoint data, keeping packed weights exactly equal to the artifact.
///
/// Float data takes [`install_float_parameter`] unchanged. Packed QFloat data is
/// first checked against every [`install_float_parameter`] precondition (rank,
/// autodiff, initialized QFloat, shape, storage), returning the same errors before
/// anything is probed or allocated. It is then installed as packed storage
/// whenever that is exact: the parameter has no load mapper (or an identity one),
/// or the mapper is a pure axis permutation and the backend's quantized layout
/// operations are exact ([`packed_layout_is_exact`]). Otherwise `policy` decides.
/// Every failure leaves the parameter untouched.
///
/// The mapper probe reads two tiny tensors back to the host once per call; it is
/// a load-time cost independent of the parameter's size.
///
/// # Panics
///
/// The parameter's load mapper runs on a rank-`D` probe with pairwise-distinct
/// dimensions `2, 3, 5, 7, ...`, not on the parameter's own shape. A mapper that
/// accepts only one shape, such as `reshape([2, 2])`, panics on that probe;
/// install such parameters with [`install_float_parameter`] instead. Element-wise
/// and axis-permuting mappers (`transpose`, `swap_dims`, `permute`, `detach`)
/// accept every shape.
pub fn install_packed_parameter<B: Backend, const D: usize>(
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
    policy: PackedLayoutPolicy,
) -> Result<PackedInstall, ParameterInstallError> {
    if !matches!(data.dtype, DType::QFloat(_)) {
        install_float_parameter(parameter, data)?;
        return Ok(PackedInstall::Float);
    }
    // Reject before probing: autodiff backends panic when building QFloat data.
    validate_parameter_install(parameter, &data)?;
    let device = parameter.lazy_device();
    let effect = load_mapper_effect(parameter);
    let exact = match effect {
        MapperEffect::Identity => true,
        MapperEffect::AxisPermutation => packed_layout_is_exact::<B>(&device),
        MapperEffect::Other => false,
    };
    if exact {
        install_validated_parameter(parameter, data);
        return Ok(PackedInstall::Packed);
    }
    match policy {
        PackedLayoutPolicy::BackendDefault => {
            install_validated_parameter(parameter, data);
            Ok(match effect {
                MapperEffect::AxisPermutation => PackedInstall::Requantized,
                MapperEffect::Identity | MapperEffect::Other => PackedInstall::Transformed,
            })
        }
        PackedLayoutPolicy::RequireExact => Err(ParameterInstallError::InexactPackedLayout),
        PackedLayoutPolicy::DequantizeWhenInexact => {
            // `q * s` per element in the checkpoint layout, on the device; the
            // mapper then runs float arithmetic, which a transpose keeps exact.
            // The shape was validated above and the dequantized tensor uses the
            // backend's own float dtype, so no host round trip is needed.
            let dequantized = Tensor::<B, D>::from_data(data, &device).dequantize();
            load_parameter_tensor(parameter, dequantized);
            Ok(PackedInstall::Dequantized)
        }
    }
}

/// Returns whether this backend's quantized transpose preserves block values.
///
/// The probe runs once per backend type and is cached. It builds a `[2, 64]` Q8
/// weight `W` with group size 32 whose rows are scaled 100x apart, then performs
/// the operation inference actually executes with a transposed packed weight,
/// `x · Wᵀ`, and compares it with `x · dequantize(W)ᵀ`. A requantizing transpose
/// (Burn 0.21 Flex) mixes the two rows' magnitudes into shared scales, which
/// moves the small row's output by tens of percent; a stride-preserving view
/// (CubeCL backends such as Metal) matches to float rounding. The comparison
/// allows a relative error of `1e-4`, far above F32 accumulation differences
/// between quantized and float matmul kernels and far below requantization error.
///
/// The reference dequantizes `W` in its original layout. Burn 0.21's CubeCL
/// `dequantize` ignores the strides of a transposed packed view, so
/// `dequantize(Wᵀ)` is not a valid reference there even though quantized
/// matmul through the same view is exact.
///
/// Returns `false` without probing on autodiff backends, which cannot construct
/// packed tensors in Burn 0.21, and on devices that cannot store the probe's
/// QFloat dtype. The cache is keyed by backend type only, so on a backend whose
/// devices differ in QFloat support the first device's answer is reused; that
/// answer errs toward `false`, which only selects the exact fallback.
pub fn packed_layout_is_exact<B: Backend>(device: &B::Device) -> bool {
    /// Probe outcomes keyed by backend type.
    static CACHE: Mutex<BTreeMap<TypeId, bool>> = Mutex::new(BTreeMap::new());

    if B::ad_enabled(device) {
        return false;
    }
    let key = TypeId::of::<B>();
    if let Some(&cached) = CACHE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return cached;
    }
    let exact = probe_quantized_transpose::<B>(device);
    CACHE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, exact);
    exact
}

/// Runs the transpose probe described in [`packed_layout_is_exact`].
fn probe_quantized_transpose<B: Backend>(device: &B::Device) -> bool {
    let values = (0..128)
        .map(|index| {
            // Row 0 lies in [-1, 1] and row 1 in [-100, 100], with irregular
            // fractions so neither row sits on the other's quantization grid.
            let unit = ((index % 64) as f32 * 0.731 - 23.0).sin();
            if index < 64 { unit } else { 100.0 * unit }
        })
        .collect::<Vec<_>>();
    let config = WeightQuantizationConfig {
        scheme: SymmetricQuantization::Q8,
        group_size: QuantizationGroupSize::new(32).unwrap_or_default(),
    };
    let Ok(packed) = quantize_tensor_data(&TensorData::new(values, [2, 64]), config)
        .and_then(|weight| weight.into_burn_qfloat_data())
    else {
        return false;
    };
    if !B::dtype_usage(device, packed.dtype).contains(DTypeUsage::Storage) {
        return false;
    }
    let weight = Tensor::<B, 2>::from_data(packed, device);
    let inputs = (0..64)
        .map(|index| (index as f32 * 0.37).cos())
        .collect::<Vec<_>>();
    let input = Tensor::<B, 2>::from_data(TensorData::new(inputs, [1, 64]), device);
    let packed_product = input.clone().matmul(weight.clone().transpose());
    let float_product = input.matmul(weight.dequantize().transpose());
    let (Ok(packed_product), Ok(float_product)) = (
        packed_product.into_data().to_vec::<f32>(),
        float_product.into_data().to_vec::<f32>(),
    ) else {
        return false;
    };
    packed_product.len() == float_product.len()
        && packed_product
            .iter()
            .zip(&float_product)
            .all(|(packed, float)| (packed - float).abs() <= 1e-4 * (float.abs() + 1.0))
}

/// Classifies the parameter's load mapper by running it on a float probe.
///
/// A clone of the parameter (lazy clones never initialize) loads a probe with
/// pairwise-distinct dimensions `2, 3, 5, ...` and values `+1, -2, +3, ...`. The
/// distinct dimensions determine the only axis permutation that could explain
/// the output shape; the mapper is a pure permutation exactly when its output
/// equals the probe permuted that way, bit for bit and at the same dtype. Signed,
/// distinct integers (exact in F16 and BF16 for every rank up to 4) expose value
/// arithmetic such as scaling or `abs`.
fn load_mapper_effect<B: Backend, const D: usize>(parameter: &Param<Tensor<B, D>>) -> MapperEffect {
    let device = parameter.lazy_device();
    let dims = probe_dims::<D>();
    let count = dims.iter().product::<usize>();
    let values = (1..=count)
        .map(|value| {
            if value % 2 == 1 {
                value as f32
            } else {
                -(value as f32)
            }
        })
        .collect::<Vec<_>>();
    let probe = Tensor::<B, D>::from_data(TensorData::new(values, dims), &device);
    let mapped = parameter
        .clone()
        .load_record(Param::initialized(ParamId::new(), probe.clone()))
        .val();
    let Some(axes) = axis_permutation(&dims, &mapped.dims()) else {
        return MapperEffect::Other;
    };
    if mapped.into_data() != probe.permute(axes).into_data() {
        return MapperEffect::Other;
    }
    if axes
        .iter()
        .enumerate()
        .all(|(axis, &source)| axis == source)
    {
        MapperEffect::Identity
    } else {
        MapperEffect::AxisPermutation
    }
}

/// Returns pairwise-distinct probe dimensions for a rank-`D` mapper probe.
///
/// The first eight axes use the primes `2..=19`; any further axis `k` uses
/// `k + 23`, which exceeds every prime used and every earlier fallback.
fn probe_dims<const D: usize>() -> [usize; D] {
    /// Pairwise-distinct small dimensions for the common ranks.
    const DISTINCT: [usize; 8] = [2, 3, 5, 7, 11, 13, 17, 19];
    core::array::from_fn(|axis| DISTINCT.get(axis).copied().unwrap_or(axis + 23))
}

/// Recovers the axis permutation that maps `input` dimensions to `output`.
///
/// Requires pairwise-distinct `input` dimensions. Returns `axes` with
/// `output[k] == input[axes[k]]` for every `k`, the convention of Burn's
/// `Tensor::permute`, or `None` when `output` is not a rearrangement of `input`
/// (a different rank, an unknown dimension, or a repeated one).
fn axis_permutation<const D: usize>(input: &[usize; D], output: &[usize]) -> Option<[usize; D]> {
    if output.len() != D {
        return None;
    }
    let mut axes = [0; D];
    let mut used = [false; D];
    for (slot, &dimension) in axes.iter_mut().zip(output) {
        let source = input.iter().position(|&candidate| candidate == dimension)?;
        if core::mem::replace(&mut used[source], true) {
            return None;
        }
        *slot = source;
    }
    Some(axes)
}

#[cfg(test)]
#[path = "packed_install_tests.rs"]
mod tests;
