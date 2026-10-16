//! Burn tensor construction and application for validated rotary positions.
//!
//! A RoPE pair treats two feature subvectors as the real and imaginary parts of
//! complex coordinates. At absolute position `p`, frequency `omega_i` rotates
//! the pair by `p * omega_i`. Hugging Face's text-model convention stores all
//! real coordinates in the first half and all imaginary coordinates in the
//! second, so `rotate_half([x_r, x_i]) = [-x_i, x_r]` and
//! `rotate(x) = x * cos + rotate_half(x) * sin`.
//!
//! Frequency vectors are tiny compared with activations and are calculated on
//! the host ([`super::RotaryFrequencies`]); only the resulting `d/2` values are
//! copied to the backend. Angle tables `[batch, 1, sequence, d]` are formed on
//! the device in F32 regardless of the device's default float dtype, because
//! positions above 256 (BF16) or 2048 (F16) are not representable exactly in
//! half precision. [`RotaryAngles::to_dtype`] converts tables once when every
//! layer runs in a reduced activation dtype.
//!
//! Interleaved checkpoints (`[re_0, im_0, re_1, im_1, ...]`) convert to the
//! split-half layout with [`interleaved_to_split_half`] before rotation.

use burn::tensor::{DType, FloatDType, Int, Tensor, TensorData, backend::Backend};

use crate::{Error, Result};

use super::index_plan::{mrope_section_total, rotary_operand_fits, split_half_width_fits};
use super::{RopeConfig, RotaryFrequencies, mrope_frequency_axis};

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

impl<B: Backend> RotaryAngles<B> {
    /// Converts both tables to `dtype`, for example once per forward in F16 models.
    ///
    /// [`apply_rotary_to`] casts tables lazily when their dtype differs from the
    /// rotated tensor; converting once up front avoids repeating those two cast
    /// kernels in every layer. Tables are created in F32, so converting to a
    /// reduced dtype rounds them exactly once.
    #[must_use]
    pub fn to_dtype(self, dtype: DType) -> Self {
        if self.cosine.dtype() == dtype && self.sine.dtype() == dtype {
            return self;
        }
        Self {
            cosine: self.cosine.cast(dtype),
            sine: self.sine.cast(dtype),
            rotary_dimension: self.rotary_dimension,
        }
    }
}

/// Builds ordinary RoPE tables from `[batch, sequence]` absolute positions.
///
/// This evaluates the configuration's frequencies on the host each call; use
/// [`RotaryFrequencies`] with [`rotary_angles_from_frequencies`] to share them
/// across layers.
///
/// # Errors
///
/// Returns a shape error for an empty position tensor and a configuration
/// error for a zero `evaluated_context`.
pub fn rotary_angles<B: Backend>(
    config: RopeConfig,
    position_ids: Tensor<B, 2, Int>,
    evaluated_context: usize,
) -> Result<RotaryAngles<B>> {
    validate_position_shape(&position_ids)?;
    let frequencies = RotaryFrequencies::new(config, evaluated_context)?;
    rotary_angles_from_frequencies(&frequencies, position_ids)
}

/// Builds RoPE tables from precomputed host frequencies and `[batch, sequence]` positions.
///
/// The angle `p * omega_i` is formed in F32: positions convert directly from
/// integers to F32 and frequencies upload as F32, independent of the device's
/// default float dtype.
///
/// # Errors
///
/// Returns a shape error for an empty position tensor.
pub fn rotary_angles_from_frequencies<B: Backend>(
    frequencies: &RotaryFrequencies,
    position_ids: Tensor<B, 2, Int>,
) -> Result<RotaryAngles<B>> {
    validate_position_shape(&position_ids)?;
    let device = position_ids.device();
    let frequency = frequency_row::<B>(frequencies, &device);
    let angle = position_ids.cast(FloatDType::F32).unsqueeze_dim::<3>(2) * frequency;
    Ok(finish_angles(frequencies, angle))
}

/// Builds Qwen multimodal RoPE tables from three position axes.
///
/// `position_ids` is `[3, batch, sequence]` for temporal, height, and width.
/// `sections = [t, h, w]` must sum to `half = rotary_dimension / 2`. Qwen
/// interleaves H at `1 mod 3` within the first `h` groups of three, W at
/// `2 mod 3` within the first `w` groups, and uses T for the remaining
/// frequencies (see [`mrope_frequency_axis`]). Height therefore receives
/// `min(h, (half + 1) / 3)` frequencies, width `min(w, half / 3)`, and
/// temporal the rest (Verus lemma `lemma_mrope_axis_counts` in
/// `src/position/index_plan.rs`). Sections within those limits, such as
/// Qwen3-VL's `[24, 20, 20]` over 64 frequencies, receive exactly their
/// counts. Larger height or width sections are not rejected: they are
/// clamped exactly as the Transformers interleaved slices `[1 : 3 * h : 3]`
/// and `[2 : 3 * w : 3]` stop at `half`. Text supplies equal values on all
/// three axes, reducing exactly to ordinary RoPE.
///
/// # Errors
///
/// Returns a shape error unless positions are `[3, batch>0, sequence>0]`, a
/// configuration error when sections do not sum to `rotary_dimension / 2`, and
/// the errors of [`RotaryFrequencies::new`].
pub fn mrope_angles<B: Backend>(
    config: RopeConfig,
    position_ids: Tensor<B, 3, Int>,
    sections: [usize; 3],
    evaluated_context: usize,
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
    let section_total = mrope_section_total(sections)
        .ok_or_else(|| Error::configuration("mrope.sections", "sum overflows usize"))?;
    if section_total != half {
        return Err(Error::configuration(
            "mrope.sections",
            format!("must sum to {half}, found {section_total}"),
        ));
    }

    let frequencies = RotaryFrequencies::new(config, evaluated_context)?;
    let device = position_ids.device();
    let frequency = frequency_row::<B>(&frequencies, &device);
    let axis_angle = |axis| {
        position_ids
            .clone()
            .narrow(0, axis, 1)
            .squeeze_dim::<2>(0)
            .cast(FloatDType::F32)
            .unsqueeze_dim::<3>(2)
            * frequency.clone()
    };
    // One-hot F32 masks select, for every complex frequency, the axis whose
    // position drives it; the three products then sum to the selected angle.
    let mut masks = [
        vec![0.0_f32; half],
        vec![0.0_f32; half],
        vec![0.0_f32; half],
    ];
    for index in 0..half {
        masks[mrope_frequency_axis(index, sections[1], sections[2])][index] = 1.0;
    }
    let [temporal_mask, height_mask, width_mask] =
        masks.map(|mask| f32_row::<B>(mask, &device).reshape([1, 1, half]));
    let angle =
        axis_angle(0) * temporal_mask + axis_angle(1) * height_mask + axis_angle(2) * width_mask;
    Ok(finish_angles(&frequencies, angle))
}

/// Rotates the leading `angles.rotary_dimension` channels of one tensor.
///
/// `input` is `[batch, heads, sequence, width]` with `width >= rotary_dimension`;
/// channels beyond the rotary prefix are returned unchanged. Tables are cast to
/// the input dtype only when they differ. This single-tensor primitive serves
/// queries and keys of different lengths (cross attention, MLA) without
/// rotating a placeholder partner.
///
/// # Errors
///
/// Returns a shape error unless the tables are `[batch, 1, sequence, rotary]`
/// and match the input's batch, sequence, and minimum width.
pub fn apply_rotary_to<B: Backend>(
    input: Tensor<B, 4>,
    angles: &RotaryAngles<B>,
) -> Result<Tensor<B, 4>> {
    validate_rotary_operand("rope.input_angles", input.dims(), angles)?;
    let (cosine, sine) = tables_in(angles, input.dtype());
    Ok(rotate_prefix(
        input,
        &cosine,
        &sine,
        angles.rotary_dimension,
    ))
}

/// Applies prepared rotary tables to Q and K, preserving an unrotated suffix.
///
/// Q and K must share batch, sequence, and width; use [`apply_rotary_to`] for
/// independently shaped operands. The tables are converted to the operand
/// dtype at most once for both tensors.
///
/// # Errors
///
/// Returns a shape error when Q and K disagree or do not match the tables.
pub fn apply_rotary<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    angles: &RotaryAngles<B>,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
    let [query_batch, _, query_sequence, query_width] = query.dims();
    let [key_batch, _, key_sequence, key_width] = key.dims();
    if query_batch != key_batch
        || query_sequence != key_sequence
        || query_width != key_width
        || validate_rotary_operand("rope.query_key_angles", query.dims(), angles).is_err()
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
    let rotary = angles.rotary_dimension;
    let (cosine, sine) = tables_in(angles, query.dtype());
    let query = rotate_prefix(query, &cosine, &sine, rotary);
    let key = if key.dtype() == cosine.dtype() {
        rotate_prefix(key, &cosine, &sine, rotary)
    } else {
        let (cosine, sine) = tables_in(angles, key.dtype());
        rotate_prefix(key, &cosine, &sine, rotary)
    };
    Ok((query, key))
}

/// Converts interleaved complex pairs `[re_0, im_0, re_1, im_1, ...]` to split-half layout.
///
/// Output channel `j < d/2` is input channel `2j` (real part) and channel
/// `d/2 + j` is input channel `2j + 1` (imaginary part), so split-half RoPE
/// then rotates exactly the checkpoint's pairs. Applying the same permutation
/// to queries and keys leaves their dot products unchanged.
///
/// # Errors
///
/// Returns a shape error unless the final dimension is non-zero and even.
pub fn interleaved_to_split_half<B: Backend>(input: Tensor<B, 4>) -> Result<Tensor<B, 4>> {
    let [batch, heads, sequence, width] = input.dims();
    validate_pair_width("rope.interleaved_input", input.dims())?;
    Ok(input
        .reshape([batch, heads, sequence, width / 2, 2])
        .swap_dims(3, 4)
        .reshape([batch, heads, sequence, width]))
}

/// Converts split-half complex pairs back to interleaved `[re_0, im_0, ...]` layout.
///
/// This is the exact inverse of [`interleaved_to_split_half`].
///
/// # Errors
///
/// Returns a shape error unless the final dimension is non-zero and even.
pub fn split_half_to_interleaved<B: Backend>(input: Tensor<B, 4>) -> Result<Tensor<B, 4>> {
    let [batch, heads, sequence, width] = input.dims();
    validate_pair_width("rope.split_half_input", input.dims())?;
    Ok(input
        .reshape([batch, heads, sequence, 2, width / 2])
        .swap_dims(3, 4)
        .reshape([batch, heads, sequence, width]))
}

/// Requires a non-empty `[batch, sequence]` position tensor.
fn validate_position_shape<B: Backend>(position_ids: &Tensor<B, 2, Int>) -> Result<()> {
    let [batch, sequence] = position_ids.dims();
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "rope.position_ids",
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    Ok(())
}

/// Requires a non-zero even channel count for complex-pair permutations.
///
/// The check is the Verus-verified [`split_half_width_fits`].
fn validate_pair_width(name: &'static str, dims: [usize; 4]) -> Result<()> {
    if !split_half_width_fits(dims[3]) {
        return Err(Error::shape(
            name,
            "a non-zero even final dimension",
            format!("{dims:?}"),
        ));
    }
    Ok(())
}

/// Checks that tables `[batch, 1, sequence, rotary]` fit one `[batch, heads, sequence, width]` operand.
///
/// The check is the Verus-verified [`rotary_operand_fits`], which also
/// guarantees the `width - rotary` suffix that [`rotate_prefix`] keeps.
fn validate_rotary_operand<B: Backend>(
    name: &'static str,
    dims: [usize; 4],
    angles: &RotaryAngles<B>,
) -> Result<()> {
    if !rotary_operand_fits(
        dims,
        angles.cosine.dims(),
        angles.sine.dims(),
        angles.rotary_dimension,
    ) {
        return Err(Error::shape(
            name,
            "operand [batch, heads, sequence, width] and angles [batch, 1, sequence, rotary<=width]",
            format!(
                "operand={dims:?}, cosine={:?}, sine={:?}, rotary={}",
                angles.cosine.dims(),
                angles.sine.dims(),
                angles.rotary_dimension
            ),
        ));
    }
    Ok(())
}

/// Returns the cosine and sine tables in `dtype`, casting only when they differ.
fn tables_in<B: Backend>(angles: &RotaryAngles<B>, dtype: DType) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let convert = |table: &Tensor<B, 4>| {
        if table.dtype() == dtype {
            table.clone()
        } else {
            table.clone().cast(dtype)
        }
    };
    (convert(&angles.cosine), convert(&angles.sine))
}

/// Applies `x * cos + rotate_half(x) * sin` to the rotary prefix and keeps the suffix.
fn rotate_prefix<B: Backend>(
    input: Tensor<B, 4>,
    cosine: &Tensor<B, 4>,
    sine: &Tensor<B, 4>,
    rotary_dimension: usize,
) -> Tensor<B, 4> {
    let width = input.dims()[3];
    let prefix = input.clone().narrow(3, 0, rotary_dimension);
    let rotated = prefix.clone() * cosine.clone() + rotate_half(prefix) * sine.clone();
    if rotary_dimension == width {
        rotated
    } else {
        let suffix = input.narrow(3, rotary_dimension, width - rotary_dimension);
        Tensor::cat(vec![rotated, suffix], 3)
    }
}

/// Uploads host frequencies as an explicit-F32 `[1, 1, d/2]` broadcast row.
fn frequency_row<B: Backend>(frequencies: &RotaryFrequencies, device: &B::Device) -> Tensor<B, 3> {
    let half = frequencies.inverse().len();
    f32_row::<B>(frequencies.inverse().to_vec(), device).reshape([1, 1, half])
}

/// Uploads host F32 values without consulting the device's default float dtype.
fn f32_row<B: Backend>(values: Vec<f32>, device: &B::Device) -> Tensor<B, 1> {
    let length = values.len();
    Tensor::from_data(TensorData::new(values, [length]), (device, DType::F32))
}

/// Converts half-width F32 angles `[batch, sequence, d/2]` to duplicated cosine/sine tables.
///
/// Split-half RoPE uses the same phase for channels `i` and `i + d/2`, so the
/// transcendental functions run on the half-width angles and the results are
/// duplicated. A unit amplitude skips the multiplication entirely.
fn finish_angles<B: Backend>(
    frequencies: &RotaryFrequencies,
    angle: Tensor<B, 3>,
) -> RotaryAngles<B> {
    let amplitude = frequencies.amplitude();
    let duplicate = |half: Tensor<B, 3>| {
        let half = if amplitude == 1.0 {
            half
        } else {
            half.mul_scalar(amplitude)
        };
        Tensor::cat(vec![half.clone(), half], 2).unsqueeze_dim(1)
    };
    RotaryAngles {
        cosine: duplicate(angle.clone().cos()),
        sine: duplicate(angle.sin()),
        rotary_dimension: frequencies.rotary_dimension(),
    }
}

/// Applies the split-half complex quarter turn `[-imaginary, real]`.
fn rotate_half<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let half = input.dims()[3] / 2;
    let real = input.clone().narrow(3, 0, half);
    let imaginary = input.narrow(3, half, half);
    Tensor::cat(vec![-imaginary, real], 3)
}

#[cfg(test)]
#[path = "tensor_tests.rs"]
mod tests;
