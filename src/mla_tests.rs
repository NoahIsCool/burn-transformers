use burn::{
    backend::Flex,
    module::Param,
    tensor::{Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::intervention::{Hooks, Layer};
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
        .project_rotary(
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
        .project_rotary(
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
        .project_rotary(
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

#[test]
fn public_projection_preserves_legacy_rotary_values() {
    for rank in [None, Some(4)] {
        let (mla, device) = deterministic_mla(rank);
        let hidden = Tensor::from_floats(
            [[
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                [0.8, -0.1, 0.6, -0.3, 0.4, -0.5, 0.2, -0.7],
            ]],
            &device,
        );
        let positions = Tensor::<TestBackend, 2, Int>::from_ints([[1, 4]], &device);
        let rope = RopeConfig::new(2, 2, 10_000.0, RopeScaling::None).unwrap();
        let legacy = mla
            .project_rotary(hidden.clone(), positions.clone(), rope, 5)
            .unwrap();
        let public = mla
            .project_self(hidden)
            .unwrap()
            .with_rotary(positions.clone(), positions, rope, 5)
            .unwrap();
        let legacy_output = mla
            .mix_unabsorbed(
                &legacy,
                legacy.normalized_latent.clone(),
                legacy.rotated_key.clone(),
                None,
                false,
            )
            .unwrap()
            .hidden_state;
        let public_output = mla.attend(public.clone(), None, None, false).unwrap();
        assert_eq!(public_output.into_data(), legacy_output.into_data());
        assert_eq!(
            public.query_content.into_data(),
            legacy.query_content.into_data()
        );
        assert_eq!(
            public.query_rotary.into_data(),
            legacy.query_rotary.into_data()
        );
        assert_eq!(
            public.normalized_latent.into_data(),
            legacy.normalized_latent.into_data()
        );
        assert_eq!(
            public.rotated_key.into_data(),
            legacy.rotated_key.into_data()
        );
    }
}

#[test]
fn cross_attention_supports_unequal_lengths_and_independent_rotary_positions() {
    let (mla, device) = deterministic_mla(Some(4));
    let query = Tensor::ones([1, 2, 8], &device);
    let source = Tensor::ones([1, 3, 8], &device) * 0.3;
    let projection = mla.project(query.clone(), source.clone()).unwrap();
    let projection = projection
        .with_rotary(
            Tensor::<TestBackend, 2, Int>::from_ints([[0, 1]], &device),
            Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device),
            RopeConfig::new(2, 2, 10_000.0, RopeScaling::None).unwrap(),
            4,
        )
        .unwrap();
    let ordinary = mla.attend(projection.clone(), None, None, false).unwrap();
    let points = Layer::decoder(0).cross_attention();
    let mut hooks = Hooks::new();
    let latent = hooks.capture(points.kv_latent());
    let key = hooks.capture(points.key());
    let pattern = hooks.capture(points.pattern());
    let output = hooks
        .run(mla.attention_points(points), |hooks| {
            mla.attend_with_hooks(projection, None, None, false, points, hooks)
        })
        .unwrap();
    assert_eq!(output.into_data(), ordinary.into_data());
    assert_eq!(hooks.get(&latent).unwrap().dims(), [1, 3, 5]);
    assert_eq!(hooks.get(&key).unwrap().dims(), [1, 2, 3, 5]);
    assert_eq!(hooks.get(&pattern).unwrap().dims(), [1, 2, 2, 3]);

    let mut hooks = Hooks::new();
    let query_latent = hooks.capture(points.query_latent());
    hooks
        .run(mla.activation_points(points), |hooks| {
            let projection = mla.project_with_hooks(query, source, points, hooks)?;
            mla.attend_with_hooks(projection, None, None, false, points, hooks)
        })
        .unwrap();
    assert_eq!(hooks.get(&query_latent).unwrap().dims(), [1, 2, 4]);
}

#[test]
fn latent_ablation_changes_values_and_restoration_recovers_attention() {
    let (mla, device) = deterministic_mla(None);
    let projection = mla.project_self(Tensor::ones([1, 2, 8], &device)).unwrap();
    let expected = mla.attend(projection.clone(), None, None, false).unwrap();
    let points = Layer::encoder(0).self_attention();
    let mut hooks = Hooks::new();
    hooks.edit(points.kv_latent(), |latent| latent.zeros_like());
    let context = hooks.capture(points.head_context());
    hooks
        .run(mla.attention_points(points), |hooks| {
            mla.attend_with_hooks(projection.clone(), None, None, false, points, hooks)
        })
        .unwrap();
    assert!(
        hooks
            .get(&context)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .all(|value| value == 0.0)
    );
    hooks.replace(points.kv_latent(), projection.normalized_latent.clone());
    let restored = hooks
        .run(mla.attention_points(points), |hooks| {
            mla.attend_with_hooks(projection, None, None, false, points, hooks)
        })
        .unwrap();
    assert_eq!(restored.into_data(), expected.into_data());
}

#[test]
fn mla_pattern_edits_reapply_causal_and_padding_masks() {
    let (mla, device) = deterministic_mla(None);
    let projection = mla
        .project(
            Tensor::ones([1, 2, 8], &device),
            Tensor::ones([1, 3, 8], &device),
        )
        .unwrap();
    let mask = Tensor::<TestBackend, 4, Bool>::from_bool(
        [[
            [[false, true, false], [true, true, true]],
            [[false, true, false], [true, true, true]],
        ]]
        .into(),
        &device,
    );
    let bias = Tensor::full([1, 1, 1, 3], 0.7, &device);
    let points = Layer::decoder(0).self_attention();
    let mut hooks = Hooks::new();
    hooks.edit(points.pattern().head(1), |pattern| pattern * 0.0 + 1.0);
    let pattern = hooks.capture(points.pattern());
    let context = hooks.capture(points.head_context());
    hooks
        .run(mla.attention_points(points), |hooks| {
            mla.attend_with_hooks(projection, Some(mask), Some(bias), true, points, hooks)
        })
        .unwrap();
    assert_eq!(
        hooks
            .get(&pattern)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        vec![1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
    );
    assert!(
        hooks
            .get(&context)
            .unwrap()
            .narrow(2, 1, 1)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .all(|value| value == 0.0)
    );
}

#[test]
fn mla_rejects_bad_source_projection_and_masks_before_backend_kernels() {
    let (mla, device) = deterministic_mla(None);
    assert!(
        mla.project(
            Tensor::ones([1, 2, 8], &device),
            Tensor::ones([2, 3, 8], &device)
        )
        .is_err()
    );
    let projection = mla.project_self(Tensor::ones([1, 2, 8], &device)).unwrap();
    let mut malformed = projection.clone();
    malformed.query_rotary = Tensor::zeros([1, 1, 2, 2], &device);
    assert!(
        mla.attend(malformed, None, None, false)
            .unwrap_err()
            .to_string()
            .contains("mla.projection")
    );
    let mask = Tensor::<TestBackend, 4, Bool>::zeros([1, 1, 2, 2], &device);
    assert!(
        mla.attend(projection.clone(), Some(mask), None, false)
            .is_err()
    );
    assert!(
        mla.attend(
            projection,
            None,
            Some(Tensor::zeros([1, 2, 2, 3], &device)),
            false
        )
        .is_err()
    );
    assert!(
        mla.projection_points(Layer::encoder(0).self_attention())
            .is_empty()
    );
}

#[cfg(feature = "autodiff")]
#[test]
fn mla_latent_adapter_receives_gradients_along_with_encoder_states() {
    use burn::backend::Autodiff;
    type Ad = Autodiff<TestBackend>;
    let device = Default::default();
    let mla = MultiLatentAttentionConfig::new(8, 2, 3, 2, 4, 5, Some(4))
        .init::<Ad>(&device)
        .unwrap();
    let adapter = LinearConfig::new(5, 5).init::<Ad>(&device);
    let query = Tensor::<Ad, 3>::ones([1, 2, 8], &device).require_grad();
    let source = Tensor::<Ad, 3>::ones([1, 3, 8], &device).require_grad();
    let points = Layer::decoder(0).cross_attention();
    let mut hooks = Hooks::new();
    hooks.edit(points.kv_latent(), |latent| {
        latent.clone() + adapter.forward(latent) * 0.1
    });
    let output = hooks
        .run(mla.activation_points(points), |hooks| {
            let projection =
                mla.project_with_hooks(query.clone(), source.clone(), points, hooks)?;
            mla.attend_with_hooks(projection, None, None, false, points, hooks)
        })
        .unwrap();
    let gradients = output.sum().backward();
    assert!(adapter.weight.grad(&gradients).is_some());
    assert!(query.grad(&gradients).is_some());
    assert!(source.grad(&gradients).is_some());
    assert!(mla.kv_b_proj.weight.grad(&gradients).is_some());
}
