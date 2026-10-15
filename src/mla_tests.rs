//! Multi-head latent attention tests for scalar parity, rotary positions, cache shapes, and gradients.

use burn::{
    backend::Flex,
    module::Param,
    tensor::{DType, Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::intervention::{Hooks, Layer};
use crate::position::{RopeScaling, YarnConfig};

/// Deterministic F32 CPU backend for latent-attention projections and scalar references.
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
    let actual = interleaved_to_split_half(input).unwrap().into_data();
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

/// The public projection path reproduces legacy rotary tensors and attention with either query topology.
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

/// Cross-attention preserves distinct query/source lengths and rotary positions through latent captures.
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

/// Zeroing the key/value latent removes head context, and restoring it recovers the original output.
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

/// Latent-attention pattern edits preserve future-key exclusion and zero fully padded query rows.
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

/// Malformed source, projected head geometry, masks, and bias return validation errors before attention.
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
/// A latent adapter remains differentiable together with query inputs, source states, and KV expansion.
#[test]
fn mla_latent_adapter_receives_gradients_along_with_encoder_states() {
    use burn::backend::Autodiff;
    /// Differentiable CPU backend for latent-adapter and source-state gradient checks.
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

/// Every required width must be positive, and epsilon and softmax scale must be finite and positive.
#[test]
fn configuration_rejects_each_zero_extent_and_nonfinite_numerical_constant() {
    for field in [
        "hidden_size",
        "num_heads",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "value_head_dim",
        "kv_lora_rank",
        "q_lora_rank",
    ] {
        let mut config = MultiLatentAttentionConfig::new(8, 2, 3, 2, 4, 5, None);
        match field {
            "hidden_size" => config.hidden_size = 0,
            "num_heads" => config.num_heads = 0,
            "qk_nope_head_dim" => config.qk_nope_head_dim = 0,
            "qk_rope_head_dim" => config.qk_rope_head_dim = 0,
            "value_head_dim" => config.value_head_dim = 0,
            "kv_lora_rank" => config.kv_lora_rank = 0,
            "q_lora_rank" => config.q_lora_rank = Some(0),
            _ => unreachable!(),
        }
        assert!(
            matches!(config.validate(), Err(Error::Configuration { field: actual, .. })
            if actual == format!("mla.{field}"))
        );
    }
    for field in ["rms_norm_epsilon", "softmax_scale"] {
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut config = MultiLatentAttentionConfig::new(8, 2, 3, 2, 4, 5, None);
            if field == "rms_norm_epsilon" {
                config.rms_norm_epsilon = value;
            } else {
                config.softmax_scale = value;
            }
            assert!(
                matches!(config.validate(), Err(Error::Configuration { field: actual, .. })
                if actual == format!("mla.{field}"))
            );
        }
    }
}

/// Overflowing head sums and projection widths fail configuration before allocating parameter tensors.
#[test]
fn projection_extent_overflow_is_rejected_before_parameter_allocation() {
    let base = MultiLatentAttentionConfig::new(8, 2, 3, 2, 4, 5, None);
    let mut config = base.clone();
    config.qk_nope_head_dim = usize::MAX;
    assert!(
        config
            .qk_head_dim()
            .unwrap_err()
            .to_string()
            .contains("mla.qk_head_dim")
    );
    config = base.clone();
    config.kv_lora_rank = usize::MAX;
    assert!(
        config
            .compressed_projection_width()
            .unwrap_err()
            .to_string()
            .contains("mla.compressed_projection_width")
    );
    config = base.clone();
    config.value_head_dim = usize::MAX;
    assert!(
        config
            .expanded_kv_width()
            .unwrap_err()
            .to_string()
            .contains("per-head sum")
    );
    config = base;
    config.num_heads = usize::MAX;
    assert!(
        config
            .expanded_kv_width()
            .unwrap_err()
            .to_string()
            .contains("head product")
    );

    // KV expansion fits, while the wider rotary query projection cannot fit.
    let config = MultiLatentAttentionConfig::new(1, usize::MAX / 2, 1, 4, 1, 1, None);
    assert!(
        matches!(config.validate(), Err(Error::Configuration { field, .. }) if field == "mla.query_width")
    );
    let error = config.init::<TestBackend>(&Default::default()).unwrap_err();
    assert!(matches!(error, Error::Configuration { field, .. } if field == "mla.query_width"));
}

/// Projection preflight identifies invalid query/source axes and mismatched floating dtypes.
#[test]
fn projection_checks_query_source_dtype_and_all_input_axes() {
    let (mla, device) = deterministic_mla(None);
    for (query, source, expected) in [
        ([0, 2, 8], [0, 2, 8], "mla.query_input"),
        ([1, 0, 8], [1, 2, 8], "mla.query_input"),
        ([1, 2, 7], [1, 2, 8], "mla.query_input"),
        ([1, 2, 8], [2, 2, 8], "mla.key_value_input"),
        ([1, 2, 8], [1, 0, 8], "mla.key_value_input"),
        ([1, 2, 8], [1, 2, 7], "mla.key_value_input"),
    ] {
        let error = mla
            .project(
                Tensor::zeros(query, &device),
                Tensor::zeros(source, &device),
            )
            .unwrap_err();
        assert!(matches!(error, Error::Shape { tensor, .. } if tensor == expected));
    }
    let error = mla
        .project(
            Tensor::ones([1, 2, 8], &device),
            Tensor::<TestBackend, 3>::ones([1, 3, 8], &device).cast(DType::F64),
        )
        .unwrap_err();
    assert!(matches!(error, Error::Configuration { field, .. } if field == "mla.inputs"));
}

/// Missing direct or low-rank query modules fail both ordinary and rotary projection APIs.
#[test]
fn malformed_query_module_topology_fails_both_projection_entry_points() {
    for rank in [None, Some(4)] {
        let (mut mla, device) = deterministic_mla(rank);
        if rank.is_none() {
            mla.q_proj = None;
        } else {
            mla.q_b_proj = None;
        }
        let input = Tensor::ones([1, 2, 8], &device);
        let error = mla.project_self(input.clone()).unwrap_err();
        assert!(
            matches!(error, Error::Checkpoint(reason) if reason.contains("query projection topology"))
        );
        let error = mla
            .project_rotary(
                input,
                Tensor::from_ints([[0, 1]], &device),
                RopeConfig::new(2, 2, 10_000.0, RopeScaling::None).unwrap(),
                2,
            )
            .unwrap_err();
        assert!(
            matches!(error, Error::Checkpoint(reason) if reason.contains("query projection topology"))
        );
    }
}

/// Rotary projection rejects invalid hidden-state extents and incompatible position tensors.
#[test]
fn rotary_projection_rejects_input_and_position_geometry_before_rotation() {
    let (mla, device) = deterministic_mla(None);
    let rope = RopeConfig::new(2, 2, 10_000.0, RopeScaling::None).unwrap();
    for shape in [[0, 2, 8], [1, 0, 8], [1, 2, 7]] {
        let error = mla
            .project_rotary(
                Tensor::zeros(shape, &device),
                Tensor::zeros([shape[0], shape[1]], &device),
                rope,
                2,
            )
            .unwrap_err();
        assert!(matches!(error, Error::Shape { tensor, .. } if tensor == "mla.input"));
    }
    for shape in [[2, 2], [1, 1]] {
        let error = mla
            .project_rotary(
                Tensor::ones([1, 2, 8], &device),
                Tensor::zeros(shape, &device),
                rope,
                2,
            )
            .unwrap_err();
        assert!(matches!(error, Error::Shape { tensor, .. } if tensor == "mla.position_ids"));
    }
}

/// Each projected tensor and additive bias must match the attention computation's floating dtype.
#[test]
fn attention_rejects_each_mixed_projection_dtype_and_bias_dtype() {
    let (mla, device) = deterministic_mla(None);
    let projection = mla.project_self(Tensor::ones([1, 2, 8], &device)).unwrap();
    for component in ["query_rotary", "normalized_latent", "rotated_key"] {
        let mut changed = projection.clone();
        match component {
            "query_rotary" => changed.query_rotary = changed.query_rotary.cast(DType::F64),
            "normalized_latent" => {
                changed.normalized_latent = changed.normalized_latent.cast(DType::F64)
            }
            "rotated_key" => changed.rotated_key = changed.rotated_key.cast(DType::F64),
            _ => unreachable!(),
        }
        let error = mla.attend(changed, None, None, false).unwrap_err();
        assert!(matches!(error, Error::Configuration { field, .. } if field == "mla.projection"));
    }
    let error = mla
        .attend(
            projection,
            None,
            Some(Tensor::<TestBackend, 4>::zeros([1, 1, 1, 2], &device).cast(DType::F64)),
            false,
        )
        .unwrap_err();
    assert!(matches!(error, Error::Configuration { field, .. } if field == "mla.attention_bias"));
}

/// Absorbed attention validates latent history, rotary history, query geometry, and expansion weights.
#[test]
fn absorbed_decode_rejects_history_and_expansion_weight_mismatches() {
    let (mut mla, device) = deterministic_mla(None);
    let projection = mla.project_self(Tensor::ones([1, 1, 8], &device)).unwrap();
    for (latent, rotary) in [
        ([2, 3, 5], [1, 1, 3, 2]),
        ([1, 3, 4], [1, 1, 3, 2]),
        ([1, 3, 5], [1, 2, 3, 2]),
        ([1, 3, 5], [1, 1, 2, 2]),
        ([1, 3, 5], [1, 1, 3, 4]),
    ] {
        let error = mla
            .mix_absorbed(
                &projection,
                Tensor::zeros(latent, &device),
                Tensor::zeros(rotary, &device),
                None,
                false,
            )
            .unwrap_err();
        assert!(matches!(error, Error::Shape { tensor, .. } if tensor == "mla.absorbed_history"));
    }
    for query_shape in [[1, 1, 1, 3], [1, 2, 1, 4]] {
        let mut changed = projection.clone();
        changed.query_content = Tensor::ones(query_shape, &device);
        let error = mla
            .mix_absorbed(
                &changed,
                Tensor::ones([1, 3, 5], &device),
                Tensor::ones([1, 1, 3, 2], &device),
                None,
                false,
            )
            .unwrap_err();
        assert!(matches!(error, Error::Shape { tensor, .. } if tensor == "mla.absorbed_history"));
    }
    mla.kv_b_proj.weight = Param::from_tensor(Tensor::zeros([5, 13], &device));
    let error = mla
        .mix_absorbed(
            &projection,
            Tensor::ones([1, 3, 5], &device),
            Tensor::ones([1, 1, 3, 2], &device),
            None,
            false,
        )
        .unwrap_err();
    assert!(matches!(error, Error::Shape { tensor, .. } if tensor == "mla.kv_b_proj.weight"));
}

/// Latent expansion and rotary conversion reject incompatible channels, and masks require exact geometry.
#[test]
fn latent_and_rotary_helpers_reject_invalid_channel_widths() {
    let (mla, device) = deterministic_mla(None);
    let error = mla
        .expand_latents(Tensor::zeros([1, 2, 4], &device))
        .unwrap_err();
    assert!(matches!(error, Error::Shape { tensor, .. } if tensor == "mla.latents"));
    for width in [0, 1, 3] {
        let error =
            interleaved_to_split_half(Tensor::<TestBackend, 4>::zeros([1, 1, 2, width], &device))
                .unwrap_err();
        assert!(matches!(error, Error::Shape { tensor, .. } if tensor == "rope.interleaved_input"));
    }
    let error = attention_probabilities(
        Tensor::<TestBackend, 4>::zeros([1, 2, 1, 3], &device),
        Some(Tensor::zeros([1, 1, 1, 3], &device)),
    )
    .unwrap_err();
    assert!(matches!(error, Error::Shape { tensor, .. } if tensor == "mla.attention_mask"));
}

/// Latent attention matches a scalar dot-product oracle while explicit pattern edits may alter normalization.
#[test]
fn public_attention_matches_scalar_oracle_and_allows_unmasked_pattern_interventions() {
    let device = Default::default();
    let mut config = MultiLatentAttentionConfig::new(1, 1, 1, 2, 1, 2, None);
    config.softmax_scale = 1.0;
    let mut mla = config.init::<TestBackend>(&device).unwrap();
    // c_0=[1,0], c_1=[0,1] expand to content keys [1,3] and values [2,4].
    mla.kv_b_proj.weight =
        Param::from_tensor(Tensor::from_floats([[1.0, 2.0], [3.0, 4.0]], &device));
    mla.o_proj.weight = Param::from_tensor(Tensor::from_floats([[1.0]], &device));
    let projection = MlaProjection {
        query_content: Tensor::from_floats([[[[2.0]]]], &device),
        query_rotary: Tensor::from_floats([[[[1.0, -1.0]]]], &device),
        normalized_latent: Tensor::from_floats([[[1.0, 0.0], [0.0, 1.0]]], &device),
        rotated_key: Tensor::from_floats([[[[0.0, 1.0], [2.0, 0.0]]]], &device),
    };
    // Content plus rotary scores are [2*1-1, 2*3+2] = [1,8].
    let first_probability = 1.0 / (1.0 + 7.0_f64.exp());
    let expected = 2.0 * first_probability + 4.0 * (1.0 - first_probability);
    for causal in [false, true] {
        // A suffix query aligns with the last key, so both keys are causal.
        let result = mla.attend(projection.clone(), None, None, causal).unwrap();
        assert!((f64::from(result.into_scalar()) - expected).abs() < 1e-6);
    }
    for (mask, expected) in [([[[[true, false]]]], 4.0), ([[[[true, true]]]], 0.0)] {
        let result = mla
            .attend(
                projection.clone(),
                Some(Tensor::from_bool(mask.into(), &device)),
                None,
                false,
            )
            .unwrap();
        assert_eq!(result.into_scalar(), expected);
    }
    let points = Layer::encoder(0).self_attention();
    let mut hooks = Hooks::new();
    hooks.edit(points.pattern(), |pattern| pattern * 0.0 + 2.0);
    let output = hooks
        .run(mla.attention_points(points), |hooks| {
            mla.attend_with_hooks(projection, None, None, false, points, hooks)
        })
        .unwrap();
    // Explicit interventions need not preserve probability normalization.
    assert_eq!(output.into_scalar(), 2.0 * (2.0 + 4.0));
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(24))]

    /// Random batched histories preserve absorbed/expanded parity, exclusion zeros, and row normalization.
    #[test]
    fn absorbed_and_expanded_attention_agree_for_batched_masked_histories(
        batch in 1usize..=2,
        queries in 1usize..=3,
        keys in 1usize..=4,
        mask_phase in 0usize..3,
        low_rank_query in proptest::bool::ANY,
    ) {
        let (mla, device) = deterministic_mla(low_rank_query.then_some(4));
        let query_values = (0..batch * queries * 8).map(|i| ((i % 13) as f32 - 6.0) * 0.13).collect::<Vec<_>>();
        let source_values = (0..batch * keys * 8).map(|i| ((i % 17) as f32 - 8.0) * 0.07).collect::<Vec<_>>();
        let projection = mla.project(
            Tensor::from_data(TensorData::new(query_values, [batch, queries, 8]), &device),
            Tensor::from_data(TensorData::new(source_values, [batch, keys, 8]), &device),
        ).unwrap();
        let exclusions = (0..batch * 2 * queries * keys).map(|i| i32::from((i + mask_phase).is_multiple_of(3))).collect::<Vec<_>>();
        let mask = Tensor::<TestBackend, 4, Int>::from_data(TensorData::new(exclusions, [batch, 2, queries, keys]), &device).bool();
        let reference = mla.mix_unabsorbed(&projection, projection.normalized_latent.clone(), projection.rotated_key.clone(), Some(mask.clone()), true).unwrap();
        let absorbed = mla.mix_absorbed(&projection, projection.normalized_latent.clone(), projection.rotated_key.clone(), Some(mask), false).unwrap();
        proptest::prop_assert!(absorbed.probabilities.is_none());
        absorbed.hidden_state.into_data().assert_approx_eq::<f32>(&reference.hidden_state.into_data(), Tolerance::absolute(1e-5));
        let probabilities = reference.probabilities.unwrap().into_data().to_vec::<f32>().unwrap();
        for (row, values) in probabilities.chunks_exact(keys).enumerate() {
            let valid_count = (0..keys).filter(|&key| !(row * keys + key + mask_phase).is_multiple_of(3)).count();
            let total = values.iter().sum::<f32>();
            let expected_total = if valid_count > 0 { 1.0 } else { 0.0 };
            proptest::prop_assert!((total - expected_total).abs() < 1e-6);
            for (key, &value) in values.iter().enumerate() {
                if (row * keys + key + mask_phase).is_multiple_of(3) {
                    proptest::prop_assert_eq!(value, 0.0);
                }
            }
        }
    }
}

/// Each registered projection rejects overflowing element counts without treating arithmetic checks as memory admission.
#[test]
fn configuration_rejects_impossible_elements_in_every_registered_projection() {
    let cases = [
        (
            "q_proj",
            MultiLatentAttentionConfig::new(usize::MAX, 1, 1, 2, 1, 1, None),
        ),
        (
            "q_a_proj",
            MultiLatentAttentionConfig::new(usize::MAX, 1, 1, 2, 1, 1, Some(2)),
        ),
        (
            "q_b_proj",
            MultiLatentAttentionConfig::new(1, 1, 1, 2, 1, 1, Some(usize::MAX / 2 + 1)),
        ),
        (
            "kv_a_proj_with_mqa",
            MultiLatentAttentionConfig::new(usize::MAX / 3, 1, 1, 2, 1, 4, None),
        ),
        (
            "kv_b_proj",
            MultiLatentAttentionConfig::new(1, 2, 1, 2, 1, usize::MAX / 4 + 1, None),
        ),
        (
            "o_proj",
            MultiLatentAttentionConfig::new(2, 1, 1, 2, usize::MAX / 2 + 1, 1, None),
        ),
    ];
    let device = Default::default();
    for (projection, config) in cases {
        for error in [
            config.validate().unwrap_err(),
            config.init::<TestBackend>(&device).unwrap_err(),
        ] {
            assert!(matches!(error, Error::Configuration { field, reason }
                if field == "mla.projection_shape" && reason.starts_with(projection) && reason.contains("element count overflows")));
        }
    }
    // Arithmetic validation accepts large representable matrices; it is not
    // device-memory admission control and these configurations are never allocated.
    for config in [
        MultiLatentAttentionConfig::new(1, 1, 1, 2, 1, usize::MAX / 2, None),
        MultiLatentAttentionConfig::new(1, 1, 1, 2, 1, 1, Some(usize::MAX / 3)),
    ] {
        config.validate().unwrap();
    }
}

/// Builds a deterministic multi-head MLA module with the given head count and latent rank.
fn deterministic_wide_mla(
    heads: usize,
    rank: usize,
) -> (
    MultiLatentAttention<TestBackend>,
    <TestBackend as BackendTypes>::Device,
) {
    let device = Default::default();
    let mut mla = MultiLatentAttentionConfig::new(8, heads, 3, 2, 4, rank, None)
        .init::<TestBackend>(&device)
        .unwrap();
    for (offset, linear) in [
        mla.q_proj.as_mut().unwrap(),
        &mut mla.kv_a_proj_with_mqa,
        &mut mla.kv_b_proj,
        &mut mla.o_proj,
    ]
    .into_iter()
    .enumerate()
    {
        let [input, output] = linear.weight.val().dims();
        let values = (0..input * output)
            .map(|index| ((index * 7 + offset * 5) % 17) as f32 * 0.02 - 0.16)
            .collect::<Vec<_>>();
        linear.weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(values, [input, output]),
            &device,
        ));
    }
    (mla, device)
}

/// The previous absorbed algorithm: latents expanded per head and one GEMV per head.
///
/// Retained in tests as the parity and timing reference for the folded path.
fn per_head_absorbed_reference(
    mla: &MultiLatentAttention<TestBackend>,
    projection: &MlaProjection<TestBackend>,
    latents: Tensor<TestBackend, 3>,
    rotary_keys: Tensor<TestBackend, 4>,
    mask: Option<Tensor<TestBackend, 4, Bool>>,
) -> Tensor<TestBackend, 3> {
    let config = &mla.config;
    let [batch, heads, queries, _] = projection.query_content.dims();
    let keys = latents.dims()[1];
    let per_head = config.qk_nope_head_dim + config.value_head_dim;
    let per_head_weight = mla
        .kv_b_proj
        .weight
        .val()
        .reshape([config.kv_lora_rank, heads, per_head])
        .swap_dims(0, 1);
    let key_weight = per_head_weight
        .clone()
        .narrow(2, 0, config.qk_nope_head_dim)
        .swap_dims(1, 2)
        .unsqueeze_dim::<4>(0)
        .expand([batch, heads, config.qk_nope_head_dim, config.kv_lora_rank]);
    let value_weight = per_head_weight
        .narrow(2, config.qk_nope_head_dim, config.value_head_dim)
        .unsqueeze_dim::<4>(0)
        .expand([batch, heads, config.kv_lora_rank, config.value_head_dim]);
    let latent_heads =
        latents
            .unsqueeze_dim::<4>(1)
            .expand([batch, heads, keys, config.kv_lora_rank]);
    let content = projection
        .query_content
        .clone()
        .matmul(key_weight)
        .matmul(latent_heads.clone().swap_dims(2, 3));
    let rotary = projection.query_rotary.clone().matmul(
        rotary_keys
            .expand([batch, heads, keys, config.qk_rope_head_dim])
            .swap_dims(2, 3),
    );
    let probabilities =
        attention_probabilities((content + rotary) * config.softmax_scale, mask).unwrap();
    let mixed = probabilities
        .matmul(latent_heads)
        .matmul(value_weight)
        .swap_dims(1, 2)
        .reshape([batch, queries, heads * config.value_head_dim]);
    operations::linear(mixed, &mla.o_proj).unwrap()
}

/// Folded multi-head decode matches the per-head reference and the expanded path,
/// including a fully padded history row.
#[test]
fn folded_absorbed_decode_matches_per_head_and_expanded_paths() {
    let (mla, device) = deterministic_wide_mla(4, 5);
    let hidden = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            (0..2 * 6 * 8)
                .map(|index| ((index % 11) as f32 - 5.0) * 0.09)
                .collect::<Vec<_>>(),
            [2, 6, 8],
        ),
        &device,
    );
    let history = mla.project_self(hidden.clone()).unwrap();
    let query = mla.project(hidden.clone().narrow(1, 5, 1), hidden).unwrap();
    // Batch 1's single query sees no key: every history link is padding.
    let mut exclusions = vec![false; 2 * 4 * 6];
    for (index, excluded) in exclusions.iter_mut().enumerate() {
        *excluded = index >= 4 * 6 || index % 6 == 0;
    }
    let mask = Tensor::<TestBackend, 4, Bool>::from_data(
        TensorData::new(exclusions, [2, 4, 1, 6]),
        &device,
    );
    let absorbed = mla
        .mix_absorbed(
            &query,
            history.normalized_latent.clone(),
            history.rotated_key.clone(),
            Some(mask.clone()),
            true,
        )
        .unwrap();
    let expanded = mla
        .mix_unabsorbed(
            &query,
            history.normalized_latent.clone(),
            history.rotated_key.clone(),
            Some(mask.clone()),
            false,
        )
        .unwrap();
    let reference = per_head_absorbed_reference(
        &mla,
        &query,
        history.normalized_latent,
        history.rotated_key,
        Some(mask),
    );
    absorbed
        .hidden_state
        .clone()
        .into_data()
        .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::absolute(1e-6));
    absorbed.hidden_state.into_data().assert_approx_eq::<f32>(
        &expanded.hidden_state.into_data(),
        Tolerance::absolute(1e-5),
    );
    let probabilities = absorbed
        .probabilities
        .unwrap()
        .narrow(0, 1, 1)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(probabilities.iter().all(|value| *value == 0.0));
}

/// Single-tensor rotation equals the former self-paired rotation exactly.
#[test]
fn with_rotary_rotates_each_tensor_once_with_identical_values() {
    let (mla, device) = deterministic_mla(Some(4));
    let projection = mla
        .project(
            Tensor::ones([1, 2, 8], &device) * 0.4,
            Tensor::ones([1, 3, 8], &device) * -0.2,
        )
        .unwrap();
    let rope = RopeConfig::new(2, 2, 10_000.0, RopeScaling::None).unwrap();
    let query_positions = Tensor::<TestBackend, 2, Int>::from_ints([[3, 4]], &device);
    let key_positions = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 2]], &device);
    let rotated = projection
        .clone()
        .with_rotary(query_positions.clone(), key_positions.clone(), rope, 5)
        .unwrap();
    let query_angles = rotary_angles(rope, query_positions, 5).unwrap();
    let key_angles = rotary_angles(rope, key_positions, 5).unwrap();
    let expected_query = apply_rotary(
        projection.query_rotary.clone(),
        projection.query_rotary,
        &query_angles,
    )
    .unwrap()
    .0;
    let expected_key = apply_rotary(
        projection.rotated_key.clone(),
        projection.rotated_key,
        &key_angles,
    )
    .unwrap()
    .0;
    assert_eq!(rotated.query_rotary.into_data(), expected_query.into_data());
    assert_eq!(rotated.rotated_key.into_data(), expected_key.into_data());
}

/// Measures one-token absorbed decode: folded heads versus per-head GEMVs.
///
/// Run with `cargo test --release --lib absorbed_decode_benchmark -- --ignored --nocapture`.
#[test]
#[ignore = "timing benchmark; run explicitly in release mode"]
fn absorbed_decode_benchmark() {
    use std::time::Instant;
    let device = Default::default();
    let (heads, rank, rotary, keys) = (16, 512, 64, 4096);
    let mut config = MultiLatentAttentionConfig::new(512, heads, 128, rotary, 128, rank, None);
    config.softmax_scale = 0.07;
    let mla = config.init::<TestBackend>(&device).unwrap();
    let projection = mla
        .project_self(Tensor::random(
            [1, 1, 512],
            burn::tensor::Distribution::Default,
            &device,
        ))
        .unwrap();
    let latents = Tensor::<TestBackend, 3>::random(
        [1, keys, rank],
        burn::tensor::Distribution::Default,
        &device,
    );
    let rotary_keys = Tensor::<TestBackend, 4>::random(
        [1, 1, keys, rotary],
        burn::tensor::Distribution::Default,
        &device,
    );
    let iterations = 10;
    let time = |folded: bool| {
        let started = Instant::now();
        for _ in 0..iterations {
            let output = if folded {
                mla.mix_absorbed(
                    &projection,
                    latents.clone(),
                    rotary_keys.clone(),
                    None,
                    false,
                )
                .unwrap()
                .hidden_state
            } else {
                per_head_absorbed_reference(
                    &mla,
                    &projection,
                    latents.clone(),
                    rotary_keys.clone(),
                    None,
                )
            };
            let _ = output.into_data();
        }
        started.elapsed().as_secs_f64() * 1e3 / f64::from(iterations)
    };
    let _ = (time(true), time(false));
    let (folded, per_head) = (time(true), time(false));
    println!(
        "absorbed MLA decode h={heads} r={rank} dr={rotary} k={keys}: folded {folded:.3} ms, per-head {per_head:.3} ms, speedup {:.2}x",
        per_head / folded
    );
}
