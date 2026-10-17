//! Burn tensor construction and application for validated rotary positions.
//!
//! A RoPE pair treats two feature subvectors as the real and imaginary parts of
//! complex coordinates. At absolute position `p`, frequency `omega_i` rotates
//! the pair by `p * omega_i`. Hugging Face's text-model convention stores all
//! real coordinates in the first half and all imaginary coordinates in the
//! second, so `rotate_half([x_r, x_i]) = [-x_i, x_r]`.
//!
//! Frequency vectors are tiny compared with activations and are calculated on
//! the host from validated scalar configuration. Only the resulting vector is
//! copied to the backend. This avoids backend-specific scalar control flow for
//! dynamic NTK and YaRN while leaving the large angle table and rotations on
//! device.
//!
//! # Precision
//!
//! [`rotary_angles_for_dtype`] and [`mrope_angles_for_dtype`] build tables in
//! [`at_least_f32`] of the activation dtype. For F16, BF16, and F32 activations
//! they reproduce the historical graph bit for bit: inverse frequencies are
//! evaluated in F64 on the host and rounded to F32, positions convert to F32,
//! and angles and their cosine/sine are formed in F32. For F64 activations the
//! same formulas run in F64 throughout: frequencies keep their unrounded host
//! values ([`inverse_frequencies_f64`]), integer positions convert directly to
//! F64, and angle products and cosine/sine are F64. Rounding the frequency and
//! the product leaves an F32 angle `p * omega` off by up to roughly
//! `p * omega * 2^-23` radians, about `1e-4` once `p * omega` reaches 1000;
//! F64 reduces that bound to `p * omega * 2^-52`. Positions above `2^24` are
//! not even representable in F32. The legacy [`rotary_angles`] and
//! [`mrope_angles`] always build F32 tables. [`apply_rotary`] casts tables to
//! the query/key dtype, so F64 activations need F64 tables to rotate without
//! F32 phase error.

use core::f64::consts::TAU;

use burn::tensor::{DType, FloatDType, Int, Tensor, TensorData, backend::Backend};

use crate::{Error, Result, precision::at_least_f32};

use super::{RopeConfig, RopeScaling};

/// Cosine and sine tables broadcastable across attention heads.
#[derive(Clone, Debug)]
pub struct RotaryAngles<B: Backend> {
    /// Cosines shaped `[batch, 1, sequence, rotary_dimension]`.
    pub cosine: Tensor<B, 4>,
    /// Sines shaped `[batch, 1, sequence, rotary_dimension]`.
    pub sine: Tensor<B, 4>,
    /// Number of leading query/key channels rotated by these tables.
    pub rotary_dimension: usize,
}

/// Computes inverse angular frequencies for a concrete evaluated context.
///
/// `evaluated_context` is the largest logical position plus one, not merely the
/// current tensor width when decoding from a cache. It affects dynamic NTK only.
/// Each value is the [`inverse_frequencies_f64`] value rounded to the nearest
/// F32, which is the frequency row of F32 rotary tables.
pub fn inverse_frequencies(config: RopeConfig, evaluated_context: usize) -> Result<Vec<f32>> {
    Ok(inverse_frequencies_f64(config, evaluated_context)?
        .into_iter()
        .map(|frequency| frequency as f32)
        .collect())
}

/// Computes unrounded F64 inverse angular frequencies for a concrete evaluated context.
///
/// Frequencies `omega_i` for `i in 0..rotary_dimension / 2` use the same F64
/// host formulas as [`inverse_frequencies`]: `theta^(-2i/d)` divided by a
/// linear factor, a dynamic-NTK base `theta * stretch^(d / (d - 2))` beyond the
/// original context, or YaRN's ramp blend of interpolated and extrapolated
/// frequencies. No F32 rounding is applied; these values feed F64 rotary tables.
/// `evaluated_context` must be non-zero.
pub fn inverse_frequencies_f64(config: RopeConfig, evaluated_context: usize) -> Result<Vec<f64>> {
    if evaluated_context == 0 {
        return Err(Error::configuration(
            "rope.evaluated_context",
            "must be non-zero",
        ));
    }
    let dimension = config.rotary_dimension();
    let half = dimension / 2;
    let (base, divisor) = match config.scaling() {
        RopeScaling::None => (config.theta(), 1.0),
        RopeScaling::Linear(linear) => (config.theta(), linear.factor()),
        RopeScaling::DynamicNtk(dynamic) => {
            let original = dynamic.original_max_position_embeddings();
            if evaluated_context <= original {
                (config.theta(), 1.0)
            } else {
                let context_ratio = evaluated_context as f64 / original as f64;
                let stretch = dynamic.factor() * context_ratio - (dynamic.factor() - 1.0);
                let exponent = dimension as f64 / (dimension as f64 - 2.0);
                (config.theta() * stretch.powf(exponent), 1.0)
            }
        }
        RopeScaling::Yarn(yarn) => {
            let base = config.theta();
            let low = yarn_correction_dimension(
                yarn.beta_fast(),
                dimension,
                base,
                yarn.original_max_position_embeddings(),
            )
            .floor()
            .clamp(0.0, (half.saturating_sub(1)) as f64);
            let high = yarn_correction_dimension(
                yarn.beta_slow(),
                dimension,
                base,
                yarn.original_max_position_embeddings(),
            )
            .ceil()
            .clamp(0.0, (half.saturating_sub(1)) as f64);
            let width = (high - low).max(1e-3);
            return Ok((0..half)
                .map(|index| {
                    let exponent = 2.0 * index as f64 / dimension as f64;
                    let extrapolated = base.powf(-exponent);
                    let interpolated = extrapolated / yarn.factor();
                    let ramp = ((index as f64 - low) / width).clamp(0.0, 1.0);
                    let extrapolation_weight = 1.0 - ramp;
                    interpolated * (1.0 - extrapolation_weight)
                        + extrapolated * extrapolation_weight
                })
                .collect());
        }
    };

    Ok((0..half)
        .map(|index| {
            let exponent = 2.0 * index as f64 / dimension as f64;
            base.powf(-exponent) / divisor
        })
        .collect())
}

/// Builds F32 RoPE tables from `[batch, sequence]` absolute positions.
///
/// This is [`rotary_angles_for_dtype`] for F32 activations. Its tables are F32
/// for every activation dtype and therefore carry F32 phase error even when
/// [`apply_rotary`] casts them to F64; pass the activation dtype to
/// [`rotary_angles_for_dtype`] to keep F64 activations in F64.
pub fn rotary_angles<B: Backend>(
    config: RopeConfig,
    position_ids: Tensor<B, 2, Int>,
    evaluated_context: usize,
) -> Result<RotaryAngles<B>> {
    rotary_angles_for_dtype(config, position_ids, evaluated_context, DType::F32)
}

/// Builds ordinary RoPE tables for query/key activations of `activation_dtype`.
///
/// `position_ids` holds `[batch, sequence]` absolute positions. Tables have dtype
/// [`at_least_f32`]`(activation_dtype)`: F32 for F16/BF16/F32 activations,
/// bit-identical to [`rotary_angles`], and F64 for F64 activations, with
/// frequencies, positions, angles, and cosine/sine all evaluated in F64. See
/// the [module documentation](self) for the error this removes.
pub fn rotary_angles_for_dtype<B: Backend>(
    config: RopeConfig,
    position_ids: Tensor<B, 2, Int>,
    evaluated_context: usize,
    activation_dtype: DType,
) -> Result<RotaryAngles<B>> {
    let [batch, sequence] = position_ids.dims();
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "rope.position_ids",
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    let table_dtype = at_least_f32(activation_dtype);
    let device = position_ids.device();
    let frequencies = frequency_row::<B>(config, evaluated_context, table_dtype, &device)?;
    let angle = float_positions(position_ids, table_dtype).unsqueeze_dim::<3>(2) * frequencies;
    finish_angles(config, angle)
}

/// Builds F32 Qwen multimodal RoPE tables from three position axes.
///
/// This is [`mrope_angles_for_dtype`] for F32 activations; its tables are F32
/// for every activation dtype, as described for [`rotary_angles`].
pub fn mrope_angles<B: Backend>(
    config: RopeConfig,
    position_ids: Tensor<B, 3, Int>,
    sections: [usize; 3],
    evaluated_context: usize,
) -> Result<RotaryAngles<B>> {
    mrope_angles_for_dtype(
        config,
        position_ids,
        sections,
        evaluated_context,
        DType::F32,
    )
}

/// Builds Qwen multimodal RoPE tables for activations of `activation_dtype`.
///
/// `position_ids` is `[3, batch, sequence]` for temporal, height, and width.
/// `sections` counts complex frequencies assigned to those axes and must sum to
/// half the rotary dimension. Qwen interleaves H at `1 mod 3`, W at `2 mod 3`,
/// and uses T for the remaining frequencies. Text supplies equal values on all
/// three axes, reducing exactly to ordinary RoPE. Tables have dtype
/// [`at_least_f32`]`(activation_dtype)`, exactly as in
/// [`rotary_angles_for_dtype`].
pub fn mrope_angles_for_dtype<B: Backend>(
    config: RopeConfig,
    position_ids: Tensor<B, 3, Int>,
    sections: [usize; 3],
    evaluated_context: usize,
    activation_dtype: DType,
) -> Result<RotaryAngles<B>> {
    let [axes, batch, sequence] = position_ids.dims();
    if axes != 3 || batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "mrope.position_ids",
            "[3, batch>0, sequence>0]",
            format!("[{axes}, {batch}, {sequence}]"),
        ));
    }
    let half = config.rotary_dimension() / 2;
    let section_total = sections
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| Error::configuration("mrope.sections", "sum overflows usize"))?;
    if section_total != half {
        return Err(Error::configuration(
            "mrope.sections",
            format!("must sum to {half}, found {section_total}"),
        ));
    }

    let table_dtype = at_least_f32(activation_dtype);
    let device = position_ids.device();
    let frequency = frequency_row::<B>(config, evaluated_context, table_dtype, &device)?;
    let axis_angle = |axis| {
        float_positions(
            position_ids.clone().narrow(0, axis, 1).squeeze_dim::<2>(0),
            table_dtype,
        )
        .unsqueeze_dim::<3>(2)
            * frequency.clone()
    };
    let temporal = axis_angle(0);
    let height = axis_angle(1);
    let width = axis_angle(2);
    let mut temporal_mask = vec![1.0_f32; half];
    let mut height_mask = vec![0.0_f32; half];
    let mut width_mask = vec![0.0_f32; half];
    for index in (1..sections[1].saturating_mul(3).min(half)).step_by(3) {
        temporal_mask[index] = 0.0;
        height_mask[index] = 1.0;
    }
    for index in (2..sections[2].saturating_mul(3).min(half)).step_by(3) {
        temporal_mask[index] = 0.0;
        width_mask[index] = 1.0;
    }
    let dense_mask =
        |values: Vec<f32>| table_row::<B>(TensorData::new(values, [half]), table_dtype, &device);
    let angle = temporal * dense_mask(temporal_mask)
        + height * dense_mask(height_mask)
        + width * dense_mask(width_mask);
    finish_angles(config, angle)
}

/// Applies prepared rotary tables to Q and K, preserving an unrotated suffix.
///
/// Tables are cast to the query/key dtype before rotation. Build them with
/// [`rotary_angles_for_dtype`] or [`mrope_angles_for_dtype`] for the same
/// activation dtype so F64 activations rotate with F64 phases.
pub fn apply_rotary<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    angles: &RotaryAngles<B>,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
    let [query_batch, _, query_sequence, query_width] = query.dims();
    let [key_batch, _, key_sequence, key_width] = key.dims();
    let [angle_batch, one, angle_sequence, angle_width] = angles.cosine.dims();
    if query_batch != key_batch
        || query_sequence != key_sequence
        || query_width != key_width
        || angle_batch != query_batch
        || one != 1
        || angle_sequence != query_sequence
        || angle_width != angles.rotary_dimension
        || angles.sine.dims() != angles.cosine.dims()
        || angles.rotary_dimension > query_width
    {
        return Err(Error::shape(
            "rope.query_key_angles",
            "Q/K [batch, heads, sequence, equal_width] and angles [batch, 1, sequence, rotary<=width]",
            format!(
                "query={:?}, key={:?}, cosine={:?}, sine={:?}",
                query.dims(),
                key.dims(),
                angles.cosine.dims(),
                angles.sine.dims()
            ),
        ));
    }
    let rotate = |input: Tensor<B, 4>| {
        let dtype = input.dtype();
        let prefix = input.clone().narrow(3, 0, angles.rotary_dimension);
        let rotated = prefix.clone() * angles.cosine.clone().cast(dtype)
            + rotate_half(prefix) * angles.sine.clone().cast(dtype);
        if angles.rotary_dimension == query_width {
            rotated
        } else {
            Tensor::cat(
                vec![
                    rotated,
                    input.narrow(
                        3,
                        angles.rotary_dimension,
                        query_width - angles.rotary_dimension,
                    ),
                ],
                3,
            )
        }
    };
    Ok((rotate(query), rotate(key)))
}

/// Uploads host inverse frequencies as a broadcastable `[1, 1, half]` row.
///
/// F64 tables upload the unrounded [`inverse_frequencies_f64`] values with
/// explicit F64 storage. Every other table dtype keeps the historical graph:
/// [`inverse_frequencies`] values uploaded with default float creation.
fn frequency_row<B: Backend>(
    config: RopeConfig,
    evaluated_context: usize,
    table_dtype: DType,
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    Ok(if table_dtype == DType::F64 {
        let frequencies = inverse_frequencies_f64(config, evaluated_context)?;
        let half = frequencies.len();
        table_row::<B>(TensorData::new(frequencies, [half]), table_dtype, device)
    } else {
        let frequencies = inverse_frequencies(config, evaluated_context)?;
        let half = frequencies.len();
        table_row::<B>(TensorData::new(frequencies, [half]), table_dtype, device)
    })
}

/// Uploads a host vector `[half]` as a broadcastable `[1, 1, half]` table row.
///
/// F64 tables request F64 storage explicitly, converting host values exactly.
/// Other tables keep the historical default-float creation unchanged.
fn table_row<B: Backend>(
    values: TensorData,
    table_dtype: DType,
    device: &B::Device,
) -> Tensor<B, 3> {
    let half = values.shape.num_elements();
    let row = if table_dtype == DType::F64 {
        Tensor::<B, 1>::from_data(values, (device, DType::F64))
    } else {
        Tensor::<B, 1>::from_data(values, device)
    };
    row.reshape([1, 1, half])
}

/// Converts integer positions to the floating table dtype.
///
/// F64 tables convert integers directly to F64, so every `i32`/`i64` position
/// below `2^53` is exact. Other tables keep the historical conversion through
/// the default float dtype followed by an F32 cast, which rounds positions
/// above `2^24`.
fn float_positions<B: Backend, const D: usize>(
    positions: Tensor<B, D, Int>,
    table_dtype: DType,
) -> Tensor<B, D> {
    if table_dtype == DType::F64 {
        positions.cast(FloatDType::F64)
    } else {
        positions.float().cast(DType::F32)
    }
}

/// Converts half-width angles to duplicated cosine/sine feature tables.
fn finish_angles<B: Backend>(config: RopeConfig, angle: Tensor<B, 3>) -> Result<RotaryAngles<B>> {
    let factor = match config.scaling() {
        RopeScaling::Yarn(yarn) => yarn.attention_factor(),
        _ => 1.0,
    };
    let duplicated = Tensor::cat(vec![angle.clone(), angle], 2);
    Ok(RotaryAngles {
        cosine: duplicated.clone().cos().mul_scalar(factor).unsqueeze_dim(1),
        sine: duplicated.sin().mul_scalar(factor).unsqueeze_dim(1),
        rotary_dimension: config.rotary_dimension(),
    })
}

/// Applies the split-half complex quarter turn `[-imaginary, real]`.
fn rotate_half<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let half = input.dims()[3] / 2;
    let real = input.clone().narrow(3, 0, half);
    let imaginary = input.narrow(3, half, half);
    Tensor::cat(vec![-imaginary, real], 3)
}

/// Finds the YaRN frequency index associated with a rotation-count boundary.
fn yarn_correction_dimension(
    rotations: f64,
    dimension: usize,
    theta: f64,
    original_context: usize,
) -> f64 {
    dimension as f64 * (original_context as f64 / (rotations * TAU)).ln() / (2.0 * theta.ln())
}

#[cfg(test)]
#[path = "tensor_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tensor_precision_tests.rs"]
mod precision_tests;
