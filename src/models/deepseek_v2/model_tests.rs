//! Deterministic DeepSeek-V2 forward, routing, numerical parity, and input-validation tests.

use burn::{
    backend::Flex,
    module::Param,
    tensor::{Bool, Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::{
    LogitsSelection,
    architecture::TokenMask,
    models::{
        config::{RopeType, yarn_mscale},
        shared::{ensure_same_device, test_support::deterministic_parameters},
    },
    position::RopeScaling,
};

/// CPU backend with F32 tensor values and I32 token indices for deterministic fixtures.
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
    assert!(ensure_same_device("deepseek_v2.input_ids", &0_u8, &0_u8, PARAMETERS).is_ok());
    for tensor in [
        "deepseek_v2.input_ids",
        "deepseek_v2.token_mask",
        "deepseek_v2.position_ids",
    ] {
        let error = ensure_same_device(tensor, &1_u8, &0_u8, PARAMETERS)
            .expect_err("different device identities must fail preflight");
        assert!(error.to_string().contains(tensor));
        assert!(error.to_string().contains("model parameters"));
    }
}

/// DeepSeek's YaRN scale uses both table amplitude and all-dimension score scale.
#[test]
fn pinned_yarn_scaling_matches_closed_form() {
    let config = DeepseekV2Config::default();
    let (rope, scale) = rope_and_softmax_scale(&config).unwrap();
    let RopeScaling::Yarn(yarn) = rope.scaling() else {
        panic!("pinned config must use YaRN")
    };
    assert!((yarn.attention_factor() - 1.0).abs() < 1e-12);
    let expected = (192.0_f64).sqrt().recip() * yarn_mscale(40.0, 0.707).powi(2);
    assert!((scale - expected).abs() < 1e-12);
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
    let yarn = |config: &DeepseekV2Config| {
        let RopeScaling::Yarn(yarn) = rope_and_softmax_scale(config).unwrap().0.scaling() else {
            panic!("expected YaRN")
        };
        yarn
    };
    let fast_only = yarn(&config);
    assert_eq!(fast_only.beta_fast(), 24.0);
    assert_eq!(fast_only.beta_slow(), 1.0);

    config.rope_scaling.as_mut().unwrap().beta_fast = None;
    config.rope_scaling.as_mut().unwrap().beta_slow = Some(2.0);
    let slow_only = yarn(&config);
    assert_eq!(slow_only.beta_fast(), 32.0);
    assert_eq!(slow_only.beta_slow(), 2.0);
}

/// The rotary sub-head always rotates completely (API-2): a partial factor is an error.
#[test]
fn partial_rotary_factors_are_rejected_instead_of_ignored() {
    let mut config = tests_support::tiny_config();
    // Four rotary channels keep the factor itself valid (two rotated channels).
    config.qk_rope_head_dim = 4;
    config.rope_scaling = Some(crate::models::config::RopeParameters {
        partial_rotary_factor: Some(0.5),
        ..Default::default()
    });
    let error = DeepseekV2ForCausalLm::<TestBackend>::init(config.clone(), &Default::default())
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("deepseek_v2.rope.partial_rotary_factor"),
        "{error}"
    );
    config.rope_scaling.as_mut().unwrap().partial_rotary_factor = Some(1.0);
    assert!(DeepseekV2ForCausalLm::<TestBackend>::init(config, &Default::default()).is_ok());
}

/// The softmax correction follows the reference for every scaled `rope_type`.
///
/// DeepSeek's reference multiplies the softmax scale by `m(f, mscale_all_dim)^2`
/// whenever `rope_scaling` is set, not only for YaRN; default RoPE has none.
#[test]
fn softmax_correction_applies_to_every_scaled_rope_type() {
    let mut config = tests_support::tiny_config();
    let base = (5.0_f64).sqrt().recip();
    assert!((rope_and_softmax_scale(&config).unwrap().1 - base).abs() < 1e-15);
    config.rope_scaling = Some(crate::models::config::RopeParameters {
        rope_type: RopeType::Linear,
        factor: Some(4.0),
        mscale_all_dim: Some(1.0),
        ..Default::default()
    });
    let scale = rope_and_softmax_scale(&config).unwrap().1;
    assert!((scale - base * yarn_mscale(4.0, 1.0).powi(2)).abs() < 1e-15);
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
    config.scoring_func = crate::models::deepseek_v2::DeepseekScoringFunction::Sigmoid;
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

    /// Reverse-mode backend used to verify gradients reach the embedding parameters.
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

/// Bounded chunking, one full plane, and the probability-retaining path agree.
///
/// One-token chunks exercise per-chunk key prefixes in every layer; the
/// padded row checks that padding visibility survives chunking.
#[test]
fn bounded_stateless_prefill_matches_unbounded_execution() {
    let (model, device) = tiny_model();
    let input = || CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
            [[1, 2, 3, 4, 5], [0, 0, 6, 7, 8]],
            &device,
        ),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true; 5], [false, false, true, true, true]].into(),
            &device,
        ))),
        position_ids: None,
    };
    let logits = |model: &DeepseekV2ForCausalLm<TestBackend>, attentions: bool| {
        model
            .forward(
                input(),
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    attentions,
                    ..Default::default()
                },
            )
            .unwrap()
            .logits
            .into_data()
    };
    let reference = logits(&model, true);
    for policy in [
        AttentionExecutionPolicy::Unbounded,
        AttentionExecutionPolicy::query_chunked(1, 1 << 20),
        AttentionExecutionPolicy::default(),
    ] {
        let model = model
            .clone()
            .with_attention_execution_policy(policy)
            .unwrap();
        assert_eq!(model.attention_execution_policy(), policy);
        logits(&model, false).assert_approx_eq::<f32>(&reference, Tolerance::absolute(1e-6));
    }
    let error = model
        .with_attention_execution_policy(AttentionExecutionPolicy::query_chunked(0, 1))
        .unwrap_err();
    assert!(error.to_string().contains("max_query_tokens"), "{error}");
}

/// Deterministic `[2, 3, width]` hidden states with irregular F32 values.
fn readout_hidden(
    width: usize,
    device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<TestBackend, 3> {
    Tensor::<TestBackend, 1, burn::tensor::Int>::arange(0..(6 * width) as i64, device)
        .float()
        .mul_scalar(0.731)
        .sin()
        .reshape([2, 3, width])
}

/// The tied readout equals Burn's reshaping `linear(h, Eᵀ)` up to F32 reassociation.
///
/// It multiplies by the transposed table view without reshaping it (no
/// per-call table copy on Flex) and still returns F32 logits.
///
/// Each logit is a dot product of `n = 8` terms, and any summation order
/// (any kernel, with or without fused multiply-adds) stays within
/// `γ_n Σ|h_i E_vi|` of the exact value, `γ_n = n u / (1 - n u)` with
/// `u = 2^-24`. Two orders therefore differ by at most twice that. The table
/// comes from seed 211 through [`deterministic_parameters`] rather than from
/// Burn's process-wide generator, whose draws depend on test scheduling.
/// The embedding is the first parameter, as in the Qwen3.5 tied-readout
/// test, so the `[32, 8]` table is the same, and twice the bound is at most
/// 0.51 of the tolerance at every logit: the comparison passes for any
/// summation order, not only for the orders measured on this host. (Each of
/// 1,000 seeds stayed below 0.76.)
#[test]
fn tied_readout_matches_reshaping_linear_oracle() {
    let device = Default::default();
    let config = DeepseekV2Config {
        tie_word_embeddings: true,
        ..tests_support::tiny_config()
    };
    let model = deterministic_parameters(
        DeepseekV2ForCausalLm::<TestBackend>::init(config.clone(), &device)
            .expect("tied tiny model"),
        211,
    );
    assert!(model.lm_head.is_none());
    let hidden = readout_hidden(config.hidden_size, &device);
    let oracle = burn::tensor::module::linear(
        hidden.clone(),
        model.model.embed_tokens.weight.val().transpose(),
        None,
    )
    .into_data();
    let logits = model.project_logits(hidden).expect("tied head");
    assert_eq!(logits.dtype(), DType::F32);
    logits
        .into_data()
        // A tied or reloaded weight is a transposed view; x86 GEMM may sum it in another order.
        .assert_approx_eq::<f32>(&oracle, Tolerance::rel_abs(1e-5, 1e-6));
}

/// A block owning both or neither feed-forward branch is rejected rather than guessed.
#[test]
fn layers_require_exactly_one_feed_forward_branch() {
    let (mut model, device) = tiny_model();
    let input = || CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
        token_mask: None,
        position_ids: None,
    };
    // Block 0 is dense; adding block 1's expert bank gives it both branches.
    model.model.layers[0].sparse_mlp = model.model.layers[1].sparse_mlp.clone();
    let both = model
        .forward(input(), &ForwardOptions::default())
        .expect_err("a dense and sparse block is ambiguous");
    model.model.layers[0].sparse_mlp = None;
    model.model.layers[0].dense_mlp = None;
    let neither = model
        .forward(input(), &ForwardOptions::default())
        .expect_err("a block without a feed-forward branch is incomplete");
    for error in [both, neither] {
        assert!(matches!(error, Error::Checkpoint(_)));
        assert!(error.to_string().contains("dense/MoE topology"), "{error}");
    }
}

/// Custom staged connectivity is evaluated only inside a hook scope.
///
/// The staged API always supplies a scope; calling a block with custom
/// connectivity and no scope is a private-contract violation that must fail
/// rather than silently fall back to causal attention.
#[test]
fn custom_connectivity_outside_a_hook_scope_is_rejected() {
    let (model, device) = tiny_model();
    let positions = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1]], &device);
    let angles = model.model.rotary_tables(positions, 2).unwrap();
    let visibility = PositionalVisibility::causal_columns(2, None, &device).unwrap();
    let inputs = AttentionInputs {
        angles: &angles,
        visibility: &visibility,
        policy: model.model.execution_policy,
        retain_attention: false,
    };
    let custom = CustomConnectivity {
        causal: false,
        exclusion: None,
        bias: None,
    };
    let hidden = model
        .model
        .embed_tokens
        .forward(Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device));
    let error = model.model.layers[0]
        .forward(hidden, &inputs, Some(&custom), Layer::decoder(0), None)
        .err()
        .expect("custom connectivity requires hooks");
    assert!(
        matches!(&error, Error::Configuration { field, .. } if field == "deepseek_v2.execution.attention"),
        "{error}"
    );
}
