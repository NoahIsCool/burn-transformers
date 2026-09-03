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

use core::f64::consts::TAU;

use burn::tensor::{DType, Int, Tensor, TensorData, backend::Backend};

use crate::{Error, Result};

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
pub fn inverse_frequencies(config: RopeConfig, evaluated_context: usize) -> Result<Vec<f32>> {
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
                    (interpolated * (1.0 - extrapolation_weight)
                        + extrapolated * extrapolation_weight) as f32
                })
                .collect());
        }
    };

    Ok((0..half)
        .map(|index| {
            let exponent = 2.0 * index as f64 / dimension as f64;
            (base.powf(-exponent) / divisor) as f32
        })
        .collect())
}

/// Builds ordinary RoPE tables from `[batch, sequence]` absolute positions.
pub fn rotary_angles<B: Backend>(
    config: RopeConfig,
    position_ids: Tensor<B, 2, Int>,
    evaluated_context: usize,
) -> Result<RotaryAngles<B>> {
    let [batch, sequence] = position_ids.dims();
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "rope.position_ids",
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    let device = position_ids.device();
    let frequencies = inverse_frequencies(config, evaluated_context)?;
    let half = frequencies.len();
    let frequencies = Tensor::<B, 1>::from_data(TensorData::new(frequencies, [half]), &device)
        .reshape([1, 1, half]);
    let angle = position_ids.float().cast(DType::F32).unsqueeze_dim::<3>(2) * frequencies;
    finish_angles(config, angle)
}

/// Builds Qwen multimodal RoPE tables from three position axes.
///
/// `position_ids` is `[3, batch, sequence]` for temporal, height, and width.
/// `sections` counts complex frequencies assigned to those axes and must sum to
/// half the rotary dimension. Qwen interleaves H at `1 mod 3`, W at `2 mod 3`,
/// and uses T for the remaining frequencies. Text supplies equal values on all
/// three axes, reducing exactly to ordinary RoPE.
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

    let device = position_ids.device();
    let frequencies = inverse_frequencies(config, evaluated_context)?;
    let frequency = Tensor::<B, 1>::from_data(TensorData::new(frequencies, [half]), &device)
        .reshape([1, 1, half]);
    let axis_angle = |axis| {
        position_ids
            .clone()
            .narrow(0, axis, 1)
            .squeeze_dim::<2>(0)
            .float()
            .cast(DType::F32)
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
    let dense_mask = |values: Vec<f32>| {
        Tensor::<B, 1>::from_data(TensorData::new(values, [half]), &device).reshape([1, 1, half])
    };
    let angle = temporal * dense_mask(temporal_mask)
        + height * dense_mask(height_mask)
        + width * dense_mask(width_mask);
    finish_angles(config, angle)
}

/// Applies prepared rotary tables to Q and K, preserving an unrotated suffix.
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
mod tests {
    use burn::{
        backend::Flex,
        tensor::{Int, Tensor, TensorData, Tolerance},
    };

    use super::{apply_rotary, inverse_frequencies, mrope_angles, rotary_angles};
    use crate::position::{
        DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling, YarnConfig,
    };

    type TestBackend = Flex<f32, i32>;

    /// Position zero is the identity and partial RoPE leaves suffix channels intact.
    #[test]
    fn zero_position_and_partial_suffix_are_identity() {
        let device = Default::default();
        let config = RopeConfig::new(6, 4, 10_000.0, RopeScaling::None).expect("valid rope");
        let positions = Tensor::<TestBackend, 2, Int>::from_ints([[0]], &device);
        let angles = rotary_angles(config, positions, 1).expect("valid positions");
        let query = Tensor::from_floats([[[[1.0, 2.0, 3.0, 4.0, 8.0, 9.0]]]], &device);
        let (rotated, _) =
            apply_rotary(query.clone(), query.clone(), &angles).expect("valid shapes");
        rotated
            .into_data()
            .assert_approx_eq::<f32>(&query.into_data(), Tolerance::absolute(1e-6));
    }

    /// Split-half rotation matches the scalar complex-number definition.
    #[test]
    fn split_half_rotation_matches_reference_values() {
        let device = Default::default();
        let config = RopeConfig::new(4, 4, 10_000.0, RopeScaling::None).expect("valid rope");
        let angles = rotary_angles(
            config,
            Tensor::<TestBackend, 2, Int>::from_ints([[1]], &device),
            2,
        )
        .expect("valid positions");
        let input = Tensor::from_floats([[[[1.0, 2.0, 3.0, 4.0]]]], &device);
        let (actual, _) = apply_rotary(input.clone(), input, &angles).expect("valid shapes");
        let expected = [[[[
            1.0_f32 * 1.0_f32.cos() - 3.0 * 1.0_f32.sin(),
            2.0 * 0.01_f32.cos() - 4.0 * 0.01_f32.sin(),
            3.0 * 1.0_f32.cos() + 1.0 * 1.0_f32.sin(),
            4.0 * 0.01_f32.cos() + 2.0 * 0.01_f32.sin(),
        ]]]];
        actual
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from(expected), Tolerance::absolute(1e-5));
    }

    /// Equal multimodal axes collapse exactly to ordinary text RoPE.
    #[test]
    fn equal_mrope_axes_match_standard_angles() {
        let device = Default::default();
        let config = RopeConfig::new(8, 8, 10_000.0, RopeScaling::None).expect("valid rope");
        let text = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 2]], &device);
        let standard = rotary_angles(config, text.clone(), 3).expect("valid text positions");
        let multimodal = text.unsqueeze_dim::<3>(0).repeat_dim(0, 3);
        let multimodal =
            mrope_angles(config, multimodal, [2, 1, 1], 3).expect("valid multimodal positions");
        assert_eq!(standard.cosine.into_data(), multimodal.cosine.into_data());
        assert_eq!(standard.sine.into_data(), multimodal.sine.into_data());
    }

    /// Linear and dynamic NTK scaling use the declared context semantics.
    #[test]
    fn inverse_frequency_scaling_matches_closed_forms() {
        let linear = RopeConfig::new(
            4,
            4,
            10_000.0,
            LinearRopeScaling::new(2.0).expect("valid linear").into(),
        )
        .expect("valid rope");
        let frequency = inverse_frequencies(linear, 8).expect("valid context");
        assert!((frequency[0] - 0.5).abs() < 1e-6);
        assert!((frequency[1] - 0.005).abs() < 1e-6);

        let dynamic_scaling = DynamicNtkScaling::new(2.0, 4).expect("valid dynamic");
        let dynamic = RopeConfig::new(4, 4, 100.0, dynamic_scaling.into()).expect("valid rope");
        assert_eq!(
            inverse_frequencies(dynamic, 4).expect("base context"),
            vec![1.0, 0.1]
        );
        let extended = inverse_frequencies(dynamic, 8).expect("extended context");
        assert_eq!(extended[0], 1.0);
        assert!(extended[1] < 0.1);
    }

    /// YaRN interpolates low frequencies and applies its table amplitude.
    #[test]
    fn yarn_interpolates_and_scales_tables() {
        let device = Default::default();
        let yarn = YarnConfig::new(4.0, 128)
            .expect("valid yarn")
            .with_attention_factor(1.25)
            .expect("valid amplitude");
        let config = RopeConfig::new(8, 8, 10_000.0, yarn.into()).expect("valid rope");
        let base = inverse_frequencies(
            RopeConfig::new(8, 8, 10_000.0, RopeScaling::None).expect("base rope"),
            256,
        )
        .expect("base frequencies");
        let scaled = inverse_frequencies(config, 256).expect("yarn frequencies");
        for (scaled, base) in scaled.iter().zip(base) {
            assert!(*scaled <= base + f32::EPSILON);
            assert!(*scaled >= base / 4.0 - f32::EPSILON);
        }
        let angles = rotary_angles(
            config,
            Tensor::<TestBackend, 2, Int>::from_ints([[0]], &device),
            256,
        )
        .expect("valid angles");
        angles.cosine.into_data().assert_approx_eq::<f32>(
            &TensorData::from([[[[1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25]]]]),
            Tolerance::absolute(1e-6),
        );
    }

    /// Shape and section mismatches return contextual errors.
    #[test]
    fn rejects_invalid_positions_sections_and_application_shapes() {
        let device = Default::default();
        let config = RopeConfig::new(4, 4, 10_000.0, RopeScaling::None).expect("valid rope");
        assert!(inverse_frequencies(config, 0).is_err());
        assert!(
            mrope_angles(
                config,
                Tensor::<TestBackend, 3, Int>::zeros([2, 1, 1], &device),
                [1, 1, 0],
                1,
            )
            .is_err()
        );
        let angles = rotary_angles(
            config,
            Tensor::<TestBackend, 2, Int>::zeros([1, 2], &device),
            2,
        )
        .expect("valid angles");
        assert!(
            apply_rotary(
                Tensor::zeros([1, 2, 1, 4], &device),
                Tensor::zeros([1, 2, 2, 4], &device),
                &angles,
            )
            .is_err()
        );
    }
}
