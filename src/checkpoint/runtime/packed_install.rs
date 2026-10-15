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
//! - a shape probe applies the parameter's own load mapper to a tiny float
//!   tensor with distinct dimensions and reports whether it permutes axes.
//!
//! Only when the mapper changes layout *and* the backend requantizes does the
//! [`PackedLayoutPolicy`] matter. Dequantizing first installs exactly `W_hat`
//! (one `q * s` product per element, then an exact float transpose), at the
//! cost of the packed memory savings for that parameter.

use std::{
    any::TypeId,
    collections::BTreeMap,
    sync::{Mutex, PoisonError},
};

use burn::{
    module::{Module, Param, ParamId},
    tensor::{DType, Tensor, TensorData, backend::Backend},
};

use super::{ParameterInstallError, install_float_parameter};
use crate::checkpoint::quantization::{
    QuantizationGroupSize, SymmetricQuantization, WeightQuantizationConfig, quantize_tensor_data,
};

/// How to install packed data when the backend would requantize it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PackedLayoutPolicy {
    /// Install the packed tensor and let the mapper run, as
    /// [`install_float_parameter`] does; on a requantizing backend the result
    /// approximates the artifact.
    #[default]
    BackendDefault,
    /// Install exactly the artifact's dequantized values in float storage.
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
    /// Installed as packed QFloat through a mapper the backend requantizes.
    ///
    /// Returned only under [`PackedLayoutPolicy::BackendDefault`].
    Requantized,
    /// Installed as exact dequantized float values.
    Dequantized,
    /// The data was not packed; it took the ordinary float path.
    Float,
}

/// Installs checkpoint data, keeping packed weights exactly equal to the artifact.
///
/// Float data takes [`install_float_parameter`] unchanged. Packed QFloat data is
/// installed as packed storage whenever that is exact: the parameter's load
/// mapper preserves layout, or the backend's quantized layout operations are
/// exact ([`packed_layout_is_exact`]). Otherwise `policy` decides. Validation
/// failures leave the parameter untouched.
pub fn install_packed_parameter<B: Backend, const D: usize>(
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
    policy: PackedLayoutPolicy,
) -> Result<PackedInstall, ParameterInstallError> {
    if !matches!(data.dtype, DType::QFloat(_)) {
        install_float_parameter(parameter, data)?;
        return Ok(PackedInstall::Float);
    }
    let device = parameter.lazy_device();
    let exact = !load_mapper_changes_layout(parameter) || packed_layout_is_exact::<B>(&device);
    if exact {
        install_float_parameter(parameter, data)?;
        return Ok(PackedInstall::Packed);
    }
    match policy {
        PackedLayoutPolicy::BackendDefault => {
            install_float_parameter(parameter, data)?;
            Ok(PackedInstall::Requantized)
        }
        PackedLayoutPolicy::RequireExact => Err(ParameterInstallError::InexactPackedLayout),
        PackedLayoutPolicy::DequantizeWhenInexact => {
            if data.shape.len() != D {
                return Err(ParameterInstallError::Rank {
                    expected: D,
                    actual: data.shape.len(),
                });
            }
            // `q * s` per element in the checkpoint layout; the mapper then
            // transposes float values, which is exact on every backend.
            let dequantized = Tensor::<B, D>::from_data(data, &device).dequantize();
            install_float_parameter(parameter, dequantized.into_data())?;
            Ok(PackedInstall::Dequantized)
        }
    }
}

/// Returns whether this backend's quantized transpose preserves block values.
///
/// The probe runs once per backend type and is cached: a `[2, 64]` Q8 tensor
/// with group size 32 and rows scaled 100x apart is transposed and dequantized,
/// and the result is compared exactly with the transpose of its dequantization,
/// both computed on `device`. A requantizing transpose mixes the two rows'
/// magnitudes into shared scales and changes the small row's values.
pub fn packed_layout_is_exact<B: Backend>(device: &B::Device) -> bool {
    /// Probe outcomes keyed by backend type.
    static CACHE: Mutex<BTreeMap<TypeId, bool>> = Mutex::new(BTreeMap::new());

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
    let tensor = Tensor::<B, 2>::from_data(packed, device);
    let transposed_first = tensor.clone().transpose().dequantize().into_data();
    let dequantized_first = tensor.dequantize().transpose().into_data();
    transposed_first == dequantized_first
}

/// Returns whether the parameter's load mapper permutes tensor axes.
///
/// A clone of the parameter (lazy clones never initialize) loads a tiny float
/// tensor with distinct dimensions `2, 3, 5, ...`; any axis permutation changes
/// its shape. No host readback occurs.
fn load_mapper_changes_layout<B: Backend, const D: usize>(parameter: &Param<Tensor<B, D>>) -> bool {
    const DISTINCT: [usize; 8] = [2, 3, 5, 7, 11, 13, 17, 19];
    let device = parameter.lazy_device();
    let dims: [usize; D] =
        core::array::from_fn(|axis| DISTINCT.get(axis).copied().unwrap_or(axis + 23));
    let probe = Tensor::<B, D>::zeros(dims, &device);
    let mapped = parameter
        .clone()
        .load_record(Param::initialized(ParamId::new(), probe));
    mapped.val().dims() != dims
}

#[cfg(test)]
#[path = "packed_install_tests.rs"]
mod tests;
