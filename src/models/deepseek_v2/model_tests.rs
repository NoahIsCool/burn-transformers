use burn::{
    backend::Flex,
    module::Param,
    tensor::{Bool, Int, Tensor, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::architecture::TokenMask;

type TestBackend = Flex<f32, i32>;

/// Builds deterministic tiny weights for forward tests.
fn tiny_model() -> (
    DeepseekV2ForCausalLm<TestBackend>,
    <TestBackend as BackendTypes>::Device,
) {
    let device = Default::default();
    let model = tests_support::initialized_model(tests_support::tiny_config(), &device);
    (model, device)
}

/// Stateless execution exposes hidden, attention, and router diagnostics.
#[test]
fn stateless_shapes_and_diagnostics_are_exact() {
    let (model, device) = tiny_model();
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                    [[1, 2, 3], [4, 5, 6]],
                    &device,
                ),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                attentions: true,
                router_logits: true,
                auxiliary_losses: true,
            },
        )
        .unwrap();
    assert_eq!(output.logits.dims(), [2, 3, 32]);
    assert_eq!(output.hidden_states.len(), 4);
    assert_eq!(output.attentions.len(), 2);
    assert_eq!(output.attentions[0].dims(), [2, 2, 3, 3]);
    assert_eq!(output.routers.len(), 1);
    assert_eq!(output.routers[0].logits.as_ref().unwrap().dims(), [2, 3, 4]);
    assert!(output.routers[0].load_balancing_loss.is_some());
}

/// Matches an independent PyTorch transcription of the pinned official equations.
///
/// The oracle uses reference revision
/// `604d5664dddd88a0433dbae533b7fe9472482de0`, the deterministic checkpoint
/// installed by `tests_support`, prompt `[1, 2, 3]`, and FP32 execution.
#[test]
fn stateless_logits_match_pinned_reference_oracle() {
    let (model, device) = tiny_model();
    let actual = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits
        .into_data();
    let expected = TensorData::from([[[
        -0.000_076_392_36_f32,
        -0.000_215_352_72,
        -0.000_141_240_54,
        0.000_092_633_3,
        -0.000_206_088_66,
        -0.000_131_976_5,
        0.000_261_666_37,
        -0.000_196_824_65,
        -0.000_122_712_49,
        0.000_404_032_35,
        -0.000_187_560_65,
        -0.000_113_448_46,
        0.000_493_179_6,
        -0.000_178_296_61,
        -0.000_104_184_44,
        0.000_502_443_6,
        -0.000_169_032_59,
        -0.000_094_920_42,
        0.000_405_196_83,
        -0.000_159_768_56,
        -0.000_085_656_39,
        0.000_174_808_55,
        -0.000_150_504_55,
        -0.000_076_392_36,
        -0.000_215_352_72,
        -0.000_141_240_54,
        0.000_092_633_3,
        -0.000_206_088_66,
        -0.000_131_976_5,
        0.000_261_666_37,
        -0.000_196_824_65,
        -0.000_122_712_49,
    ]]]);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-6));
}

/// Layer zero is dense and layer one owns selected plus widened shared experts.
#[test]
fn dense_prefix_and_sparse_shared_topology_are_distinct() {
    let (model, _) = tiny_model();
    assert!(model.model.layers[0].dense_mlp.is_some());
    assert!(model.model.layers[0].sparse_mlp.is_none());
    let sparse = model.model.layers[1].sparse_mlp.as_ref().unwrap();
    assert_eq!(sparse.routed.routed.len(), 4);
    assert_eq!(sparse.shared_experts.gate_proj.weight.val().dims(), [8, 12]);
}

/// DeepSeek routing evaluates only experts selected by the FP32 top-k projection.
#[test]
fn router_projection_is_fp32_and_dispatch_is_selected_only() {
    let (mut model, device) = tiny_model();
    let sparse = model.model.layers[1].sparse_mlp.as_mut().unwrap();
    let mut weights = Vec::with_capacity(8 * 4);
    for _ in 0..8 {
        weights.extend([1.0_f32, 0.5, -0.5, -1.0]);
    }
    sparse.routed.router.weight =
        Param::from_tensor(Tensor::from_data(TensorData::new(weights, [8, 4]), &device));
    let output = sparse.forward(Tensor::ones([1, 3, 8], &device)).unwrap();
    assert_eq!(output.routed.router_logits.dtype(), DType::F32);
    assert_eq!(output.routed.expert_token_counts, vec![3, 3, 0, 0]);
    assert_eq!(output.routed.evaluated_experts, 2);
}

/// Reduced-precision experts are weighted and accumulated before one final cast.
#[test]
fn routed_mixture_accumulates_in_fp32_before_residual_cast() {
    let device = Default::default();
    let config = tests_support::tiny_config();
    let mut moe = DeepseekV2Moe::<TestBackend>::init(&config, &device).unwrap();
    moe.routed.router.weight = Param::from_tensor(Tensor::zeros([8, 4], &device));

    let single_path = |shape: [usize; 2], value: f32| {
        let mut values = vec![0.0_f32; shape.iter().product()];
        values[0] = value;
        Tensor::<TestBackend, 2>::from_data(TensorData::new(values, shape), &device)
            .cast(DType::F16)
    };
    for (expert, sign) in moe.routed.routed[..2].iter_mut().zip([1.0, -1.0]) {
        expert.gate_proj.weight = Param::from_tensor(single_path([8, 6], 1.0));
        expert.up_proj.weight = Param::from_tensor(single_path([8, 6], 1.0));
        expert.down_proj.weight = Param::from_tensor(single_path([6, 8], sign));
    }
    let input = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1, 8]),
        &device,
    )
    .cast(DType::F16);
    let expert_zero = moe.routed.routed[0]
        .forward(input.clone())
        .unwrap()
        .cast(DType::F32);
    let expert_one = moe.routed.routed[1]
        .forward(input.clone())
        .unwrap()
        .cast(DType::F32);
    let dispatch = moe
        .routed
        .dispatch_selected_experts(
            input.clone(),
            Tensor::from_floats([[0.5001, 0.4999]], &device),
            Tensor::<TestBackend, 2, Int>::from_ints([[0, 1]], &device),
            DType::F32,
        )
        .unwrap();
    assert_eq!(dispatch.hidden_state.dtype(), DType::F32);

    let actual = dispatch.hidden_state.cast(DType::F16).cast(DType::F32);
    let expected = (expert_zero.clone() * 0.5001 + expert_one.clone() * 0.4999)
        .cast(DType::F16)
        .cast(DType::F32);
    let prematurely_rounded = ((expert_zero * 0.5001).cast(DType::F16)
        + (expert_one * 0.4999).cast(DType::F16))
    .cast(DType::F32);
    actual
        .clone()
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(0.0));
    let actual_first = actual.into_data().to_vec::<f32>().unwrap()[0];
    let premature_first = prematurely_rounded.into_data().to_vec::<f32>().unwrap()[0];
    assert!(
        (actual_first - premature_first).abs() > 1e-5,
        "the fixture must distinguish final-only rounding ({actual_first}) from per-expert rounding ({premature_first})"
    );

    for projection in [
        &mut moe.shared_experts.gate_proj,
        &mut moe.shared_experts.up_proj,
        &mut moe.shared_experts.down_proj,
    ] {
        projection.weight = Param::from_tensor(projection.weight.val().cast(DType::F16));
    }
    let output = moe.forward(input.reshape([1, 1, 8])).unwrap();
    assert_eq!(output.routed.selected_weights.dtype(), DType::F32);
    assert_eq!(output.routed.hidden_state.dtype(), DType::F16);
    assert_eq!(output.hidden_state.dtype(), DType::F16);
}

/// Top-one routing preserves confidence and the configured routed scale.
#[test]
fn top_one_routing_does_not_renormalize_away_confidence() {
    let device = Default::default();
    let mut config = tests_support::tiny_config();
    config.num_experts_per_tok = 1;
    config.norm_topk_prob = true;
    config.routed_scaling_factor = 2.5;
    let mut moe = DeepseekV2Moe::<TestBackend>::init(&config, &device).unwrap();
    moe.routed.router.weight = Param::from_tensor(Tensor::zeros([8, 4], &device));

    let output = moe.forward(Tensor::ones([1, 1, 8], &device)).unwrap();
    let weights = output
        .routed
        .selected_weights
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(weights, [0.625]);
}

/// Left padding is ignored and fully masked prompt rows fail before attention.
#[test]
fn semantic_padding_and_invalid_rows_are_validated() {
    let (model, device) = tiny_model();
    let input = CausalLmInput {
        token_ids: Tensor::from_ints([[0, 1, 2]], &device),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[false, true, true]].into(),
            &device,
        ))),
        position_ids: None,
    };
    let output = model
        .forward(
            input,
            &ForwardOptions {
                logits: LogitsSelection::Last,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(output.logits.dims(), [1, 1, 32]);

    let error = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("fully masked"));
}

/// Stateless device preflight names every independently supplied tensor.
#[test]
fn stateless_device_affinity_errors_are_contextual() {
    assert!(ensure_model_device("deepseek_v2.input_ids", &0_u8, &0_u8).is_ok());
    for tensor in [
        "deepseek_v2.input_ids",
        "deepseek_v2.token_mask",
        "deepseek_v2.position_ids",
    ] {
        let error = ensure_model_device(tensor, &1_u8, &0_u8)
            .expect_err("different device identities must fail preflight");
        assert!(error.to_string().contains(tensor));
        assert!(error.to_string().contains("model parameters"));
    }
}

/// DeepSeek's YaRN scale uses both table amplitude and all-dimension score scale.
#[test]
fn pinned_yarn_scaling_matches_closed_form() {
    let config = DeepseekV2Config::default();
    let rope = deepseek_rope_config(&config).unwrap();
    let RopeScaling::Yarn(yarn) = rope.scaling() else {
        panic!("pinned config must use YaRN")
    };
    assert!((yarn.attention_factor() - 1.0).abs() < 1e-12);
    let expected = (192.0_f64).sqrt().recip() * yarn_mscale(40.0, 0.707).powi(2);
    assert!((deepseek_attention_scale(&config).unwrap() - expected).abs() < 1e-12);
}

/// One-sided YaRN beta overrides retain the official default for the other bound.
#[test]
fn yarn_beta_overrides_are_resolved_independently() {
    let mut config = tests_support::tiny_config();
    config.rope_scaling = Some(crate::models::config::RopeParameters {
        rope_type: RopeType::Yarn,
        factor: Some(2.0),
        original_max_position_embeddings: Some(16),
        beta_fast: Some(24.0),
        ..Default::default()
    });
    let RopeScaling::Yarn(fast_only) = deepseek_rope_config(&config).unwrap().scaling() else {
        panic!("expected YaRN")
    };
    assert_eq!(fast_only.beta_fast(), 24.0);
    assert_eq!(fast_only.beta_slow(), 1.0);

    config.rope_scaling.as_mut().unwrap().beta_fast = None;
    config.rope_scaling.as_mut().unwrap().beta_slow = Some(2.0);
    let RopeScaling::Yarn(slow_only) = deepseek_rope_config(&config).unwrap().scaling() else {
        panic!("expected YaRN")
    };
    assert_eq!(slow_only.beta_fast(), 32.0);
    assert_eq!(slow_only.beta_slow(), 2.0);
}

/// DeepSeek returns vocabulary logits in FP32 even for reduced-precision modules.
#[test]
fn vocabulary_projection_always_returns_fp32() {
    let (mut model, device) = tiny_model();
    let head = model.lm_head.as_mut().unwrap();
    head.weight = Param::from_tensor(head.weight.val().cast(DType::F16));
    let hidden = Tensor::<TestBackend, 3>::zeros([1, 2, 8], (&device, DType::F16));

    let logits = model.project_logits(hidden).unwrap();
    assert_eq!(logits.dtype(), DType::F32);
}

/// Invalid token IDs and selective logits return errors rather than panics.
#[test]
fn invalid_runtime_values_are_rejected() {
    let (model, device) = tiny_model();
    assert!(
        model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[32]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert!(
        model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::Positions(vec![2]),
                    ..Default::default()
                },
            )
            .is_err()
    );
}

/// Unsupported distributed/router/dropout variants fail before large allocation.
#[test]
fn unsupported_runtime_variants_fail_during_initialization() {
    let device = Default::default();
    let mut config = tests_support::tiny_config();
    config.ep_size = 2;
    assert!(DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).is_err());

    let mut config = tests_support::tiny_config();
    config.topk_method = DeepseekTopKMethod::GroupLimitedGreedy;
    assert!(DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).is_err());

    let mut config = tests_support::tiny_config();
    config.attention_dropout = 0.1;
    assert!(DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).is_err());

    let mut config = tests_support::tiny_config();
    config.use_mla = false;
    assert!(DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).is_err());
}

/// Full-precision stateless execution retains an embedding gradient.
#[cfg(feature = "autodiff")]
#[test]
fn autodiff_stateless_smoke_reaches_embedding() {
    use burn::backend::Autodiff;

    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let model =
        DeepseekV2ForCausalLm::<AdBackend>::init(tests_support::tiny_config(), &device).unwrap();
    let loss = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits
        .sum();
    let gradients = loss.backward();
    assert!(model.model.embed_tokens.weight.grad(&gradients).is_some());
}
