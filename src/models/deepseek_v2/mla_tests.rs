use burn::{
    backend::Flex,
    module::Param,
    tensor::{Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::position::{RopeScaling, YarnConfig};

type TestBackend = Flex<f32, i32>;

/// Creates deterministic non-zero MLA parameters for path-equivalence tests.
fn deterministic_mla(
    q_lora_rank: Option<usize>,
) -> (
    MultiLatentAttention<TestBackend>,
    <TestBackend as BackendTypes>::Device,
) {
    let device = Default::default();
    let mut config = MultiLatentAttentionConfig::new(8, 2, 3, 2, 4, 5, q_lora_rank);
    config.rms_norm_epsilon = 1e-6;
    let mut mla = config.init::<TestBackend>(&device).unwrap();
    let mut counter = 0usize;
    let mut fill = |linear: &mut Linear<TestBackend>| {
        let [input, output] = linear.weight.val().dims();
        let values = (0..input * output)
            .map(|index| ((index + counter * 3) % 19) as f32 * 0.01 - 0.09)
            .collect::<Vec<_>>();
        counter += 1;
        linear.weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(values, [input, output]),
            &device,
        ));
        if let Some(bias) = &mut linear.bias {
            *bias = Param::from_tensor(Tensor::zeros([output], &device));
        }
    };
    if let Some(linear) = &mut mla.q_proj {
        fill(linear);
    }
    if let Some(linear) = &mut mla.q_a_proj {
        fill(linear);
    }
    if let Some(linear) = &mut mla.q_b_proj {
        fill(linear);
    }
    fill(&mut mla.kv_a_proj_with_mqa);
    fill(&mut mla.kv_b_proj);
    fill(&mut mla.o_proj);
    (mla, device)
}

/// Direct and low-rank query topologies allocate mutually exclusive fields.
#[test]
fn query_topology_matches_optional_lora_rank() {
    let (direct, _) = deterministic_mla(None);
    assert!(direct.q_proj.is_some());
    assert!(direct.q_a_proj.is_none());
    let (low_rank, _) = deterministic_mla(Some(4));
    assert!(low_rank.q_proj.is_none());
    assert!(low_rank.q_a_proj.is_some());
    assert!(low_rank.q_a_layernorm.is_some());
    assert!(low_rank.q_b_proj.is_some());
}

/// DeepSeek reorders projection-interleaved real/imaginary channels before RoPE.
#[test]
fn interleaved_rotary_channels_are_permuted_to_split_halves() {
    let device = Default::default();
    let input = Tensor::<TestBackend, 4>::from_floats(
        [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]],
        &device,
    );
    let actual = interleaved_to_split(input, "test").unwrap().into_data();
    actual.assert_approx_eq::<f32>(
        &TensorData::from([[[[1.0, 3.0, 2.0, 4.0], [5.0, 7.0, 6.0, 8.0]]]]),
        Tolerance::absolute(0.0),
    );
}

/// Absorbing K/V expansion preserves reference attention numerically.
#[test]
fn absorbed_attention_matches_reference_unabsorbed_path() {
    let (mla, device) = deterministic_mla(None);
    let rope = RopeConfig::new(2, 2, 10_000.0, RopeScaling::None).unwrap();
    let projection = mla
        .project(
            Tensor::from_floats(
                [[
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.2, -0.1, 0.4, -0.3, 0.6, -0.5, 0.8, -0.7],
                ]],
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 2]], &device),
            rope,
            3,
        )
        .unwrap();
    let reference = mla
        .mix_unabsorbed(
            &projection,
            projection.normalized_latent.clone(),
            projection.rotated_key.clone(),
            None,
            true,
        )
        .unwrap();
    let absorbed = mla
        .mix_absorbed(
            &projection,
            projection.normalized_latent.clone(),
            projection.rotated_key.clone(),
            None,
            true,
        )
        .unwrap();
    absorbed.hidden_state.into_data().assert_approx_eq::<f32>(
        &reference.hidden_state.into_data(),
        Tolerance::absolute(1e-5),
    );
    absorbed
        .probabilities
        .unwrap()
        .into_data()
        .assert_approx_eq::<f32>(
            &reference.probabilities.unwrap().into_data(),
            Tolerance::absolute(1e-5),
        );
}

/// YaRN tables remain compatible with DeepSeek's interleaved input channels.
#[test]
fn projection_accepts_yarn_and_preserves_cache_geometry() {
    let (mla, device) = deterministic_mla(Some(4));
    let yarn = YarnConfig::new(4.0, 8)
        .unwrap()
        .with_attention_factor(1.0)
        .unwrap();
    let rope = RopeConfig::new(2, 2, 10_000.0, yarn.into()).unwrap();
    let output = mla
        .project(
            Tensor::ones([2, 3, 8], &device),
            Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 2], [3, 4, 5]], &device),
            rope,
            6,
        )
        .unwrap();
    assert_eq!(output.normalized_latent.dims(), [2, 3, 5]);
    assert_eq!(output.rotated_key.dims(), [2, 1, 3, 2]);
}

/// Invalid ranks, rotary widths, scales, and runtime history shapes fail safely.
#[test]
fn invalid_config_and_history_shapes_return_contextual_errors() {
    let mut config = MultiLatentAttentionConfig::new(8, 2, 3, 3, 4, 5, None);
    assert!(config.validate().is_err());
    config.qk_rope_head_dim = 2;
    config.softmax_scale = f64::NAN;
    assert!(config.validate().is_err());

    let (mla, device) = deterministic_mla(None);
    let rope = RopeConfig::new(2, 2, 10_000.0, RopeScaling::None).unwrap();
    let projection = mla
        .project(
            Tensor::ones([1, 1, 8], &device),
            Tensor::<TestBackend, 2, Int>::from_ints([[0]], &device),
            rope,
            1,
        )
        .unwrap();
    assert!(
        mla.mix_unabsorbed(
            &projection,
            Tensor::zeros([1, 1, 4], &device),
            Tensor::zeros([1, 1, 1, 2], &device),
            None,
            false,
        )
        .is_err()
    );
}
