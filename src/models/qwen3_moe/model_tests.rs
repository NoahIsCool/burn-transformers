use burn::{
    backend::Flex,
    module::Param,
    tensor::{Bool, DType, Int, Tensor, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::{
    CausalLanguageModel, DecodeInput, DecodeState, cache::CacheConfig, checkpoint::TensorSink,
    models::qwen3_moe::Qwen3MoeCheckpointOptions,
};

type TestBackend = Flex<f32, i32>;

/// Creates a small model that exercises routed and dense feed-forward paths.
fn tiny_config() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        rope_theta: Some(1_000_000.0),
        max_window_layers: 2,
        decoder_sparse_step: 1,
        moe_intermediate_size: 6,
        num_experts: 4,
        num_experts_per_tok: 2,
        norm_topk_prob: true,
        ..Default::default()
    }
}

/// Installs deterministic tensors without consuming Burn's process-wide test RNG.
fn initialized_model<B: Backend>(
    config: Qwen3MoeConfig,
    device: &B::Device,
) -> Qwen3MoeForCausalLm<B> {
    let contract = config
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .expect("tiny contract");
    let mut model = Qwen3MoeForCausalLm::init(config, device).expect("tiny model");
    for (tensor_index, expected) in contract.expected_tensors().iter().enumerate() {
        let shape = expected
            .shape
            .iter()
            .map(|&dimension| usize::try_from(dimension).expect("tiny dimension"))
            .collect::<Vec<_>>();
        let elements = shape.iter().product();
        let values = (0..elements)
            .map(|element| {
                let code = (element + tensor_index * 7) % 23;
                (code as f32 - 11.0) * 0.002
            })
            .collect::<Vec<_>>();
        TensorSink::write_tensor(&mut model, &expected.name, TensorData::new(values, shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    model
}

/// Builds deterministic tiny weights for parity tests.
fn tiny_model() -> (
    Qwen3MoeForCausalLm<TestBackend>,
    <TestBackend as BackendTypes>::Device,
) {
    let device = Default::default();
    let model = initialized_model(tiny_config(), &device);
    (model, device)
}

/// Every Qwen3-MoE norm must round before scale, including per-head Q/K norms.
#[test]
fn every_rms_norm_uses_the_upstream_precision_policy() {
    let device = Default::default();
    let model = Qwen3MoeForCausalLm::<TestBackend>::init(tiny_config(), &device).unwrap();

    assert!(model.model.norm.round_before_scale);
    for layer in &model.model.layers {
        assert!(layer.input_layernorm.round_before_scale);
        assert!(layer.post_attention_layernorm.round_before_scale);
        assert!(
            layer
                .self_attn
                .query_norm
                .as_ref()
                .expect("Qwen3-MoE query RMSNorm")
                .round_before_scale
        );
        assert!(
            layer
                .self_attn
                .key_norm
                .as_ref()
                .expect("Qwen3-MoE key RMSNorm")
                .round_before_scale
        );
    }
}

/// Every routed layer selects Qwen's FP32-weight/per-contribution rounding boundary.
#[test]
fn every_sparse_layer_uses_the_upstream_weighting_precision() {
    let device = Default::default();
    let model = Qwen3MoeForCausalLm::<TestBackend>::init(tiny_config(), &device).unwrap();

    for layer in &model.model.layers {
        assert_eq!(
            layer
                .sparse_mlp
                .as_ref()
                .expect("tiny topology is all-MoE")
                .weighting_precision,
            RoutedWeightingPrecision::Fp32ThenResidual
        );
    }
}

/// F16 exposes the upstream cast-before-scale order as a bitwise distinction.
#[test]
fn rms_norm_f16_matches_upstream_cast_before_scale_order() {
    let device = Default::default();
    let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(tiny_config(), &device).unwrap();
    let norm = &mut model.model.layers[0].input_layernorm;
    let weight = Tensor::<TestBackend, 1>::from_floats(
        [0.731, 1.337, -0.419, 0.613, -1.217, 1.091, 0.527, -0.883],
        &device,
    )
    .cast(DType::F16);
    norm.weight = Param::from_tensor(weight.clone());
    let input = Tensor::<TestBackend, 2>::from_floats(
        [[0.337, 1.719, -2.113, 0.491, -0.827, 1.239, 2.051, -1.477]],
        &device,
    )
    .cast(DType::F16);

    let input_f32 = input.clone().cast(DType::F32);
    let rms = (input_f32.clone().square().mean_dim(1) + norm.epsilon).sqrt();
    let unit_rms = input_f32 / rms;
    let upstream = unit_rms.clone().cast(DType::F16) * weight.clone().unsqueeze();
    let late_round = (unit_rms * weight.cast(DType::F32).unsqueeze()).cast(DType::F16);
    let actual = norm.forward(input);

    assert_eq!(actual.dtype(), DType::F16);
    assert_eq!(actual.to_data(), upstream.to_data());
    assert_ne!(actual.to_data(), late_round.to_data());
}

/// The v4.51 boundary is a full-attention prefix length, not a windowed prefix.
#[test]
fn sliding_window_schedule_matches_every_valid_boundary() {
    let mut config = tiny_config();
    config.num_hidden_layers = 8;
    config.use_sliding_window = true;
    config.sliding_window = Some(3);

    for boundary in 0..=config.num_hidden_layers {
        config.max_window_layers = boundary;
        for layer_index in 0..config.num_hidden_layers {
            let expected = (layer_index >= boundary).then_some(3);
            assert_eq!(
                config.layer_sliding_window(layer_index),
                expected,
                "boundary={boundary}, layer={layer_index}"
            );
        }
    }
}

/// Both upstream feature guards suppress the window independently.
#[test]
fn sliding_window_requires_enablement_and_a_configured_width() {
    let mut config = tiny_config();
    config.max_window_layers = 0;
    config.sliding_window = Some(3);
    for layer_index in 0..config.num_hidden_layers {
        assert_eq!(config.layer_sliding_window(layer_index), None);
    }

    config.use_sliding_window = true;
    config.sliding_window = None;
    for layer_index in 0..config.num_hidden_layers {
        assert_eq!(config.layer_sliding_window(layer_index), None);
    }
}

/// Stateless output exposes requested hidden and router diagnostics.
#[test]
fn stateless_shapes_and_router_options_are_exact() {
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
                router_logits: true,
                auxiliary_losses: true,
                ..Default::default()
            },
        )
        .expect("tiny stateless forward");
    assert_eq!(output.logits.dims(), [2, 3, 32]);
    assert_eq!(output.hidden_states.len(), 4);
    assert_eq!(
        output
            .load_balancing_loss
            .as_ref()
            .expect("requested stack-wide loss")
            .dims(),
        [1]
    );
    assert_eq!(output.routers.len(), 2);
    for router in output.routers {
        assert_eq!(router.logits.expect("requested").dims(), [2, 3, 4]);
        assert!(router.load_balancing_loss.is_none());
        assert!(router.z_loss.is_none());
    }
}

/// A pinned Transformers 5.2 fixture proves the layer-stack and double-softmax semantics.
#[test]
fn stack_wide_auxiliary_loss_matches_transformers_5_2_golden() {
    let device = Default::default();
    let raw_router_logits = vec![
        Tensor::<TestBackend, 3>::from_floats(
            [
                [[2.0, 0.0, -1.0], [0.5, 0.4, -0.2]],
                [[-1.0, 1.5, 0.25], [3.0, -2.0, 1.0]],
            ],
            &device,
        ),
        Tensor::<TestBackend, 3>::from_floats(
            [
                [[0.0, 1.0, 2.0], [-0.5, 0.25, 1.25]],
                [[1.0, 0.99, -1.0], [0.2, -0.3, 0.7]],
            ],
            &device,
        ),
    ];
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, true], [true, false]].into(),
        &device,
    ));

    let loss = qwen3_moe_load_balancing_loss(&raw_router_logits, 2, Some(&mask))
        .expect("valid pinned auxiliary-loss fixture");

    assert_eq!(loss.dtype(), DType::F32);
    loss.into_data().assert_approx_eq::<f32>(
        &TensorData::from([2.009_133_8_f32]),
        Tolerance::absolute(1.0e-6),
    );
}

/// Auxiliary loss is global and does not force raw per-layer diagnostics into the output.
#[test]
fn auxiliary_loss_only_retains_no_public_router_tensors() {
    let (model, device) = tiny_model();
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                auxiliary_losses: true,
                ..Default::default()
            },
        )
        .expect("tiny auxiliary-only forward");

    assert!(output.routers.is_empty());
    let loss = output
        .load_balancing_loss
        .expect("Qwen auxiliary flag requests one global loss")
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    assert!(
        loss > 1.0,
        "the raw objective must not apply the 0.001 coefficient"
    );
}

/// Stateful prefill returns the same global loss API and no fabricated per-layer losses.
#[test]
fn cached_auxiliary_loss_uses_the_stack_wide_path() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let output = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                router_logits: true,
                auxiliary_losses: true,
                ..Default::default()
            },
        )
        .expect("tiny stateful auxiliary forward");
    let raw_router_logits = output
        .routers
        .iter()
        .map(|router| {
            assert!(router.load_balancing_loss.is_none());
            router.logits.clone().expect("raw diagnostics requested")
        })
        .collect::<Vec<_>>();
    let expected =
        qwen3_moe_load_balancing_loss(&raw_router_logits, model.config().num_experts_per_tok, None)
            .unwrap();

    output
        .load_balancing_loss
        .expect("stateful path returns one global objective")
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(0.0));
}

/// The public loss helper rejects missing layers and invalid top-k before tensor kernels.
#[test]
fn auxiliary_loss_validates_family_inputs() {
    let device = Default::default();
    let no_layers = qwen3_moe_load_balancing_loss::<TestBackend>(&[], 1, None)
        .expect_err("an empty sparse stack has no balance objective");
    assert!(no_layers.to_string().contains("at least one sparse-layer"));

    let logits = [Tensor::<TestBackend, 3>::zeros([1, 2, 3], &device)];
    let invalid_top_k = qwen3_moe_load_balancing_loss(&logits, 4, None)
        .expect_err("top-k cannot exceed the expert count");
    assert!(invalid_top_k.to_string().contains("experts_per_token"));

    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true]].into(),
        &device,
    ));
    let invalid_mask = qwen3_moe_load_balancing_loss(&logits, 1, Some(&mask))
        .expect_err("mask axes must match every router layer");
    assert!(invalid_mask.to_string().contains("token_mask"));
}

/// Stateless device preflight names every independently supplied tensor.
#[test]
fn stateless_device_affinity_errors_are_contextual() {
    assert!(ensure_model_device("qwen3_moe.input_ids", &0_u8, &0_u8).is_ok());
    for tensor in [
        "qwen3_moe.input_ids",
        "qwen3_moe.token_mask",
        "qwen3_moe.position_ids",
        "qwen3_moe.lm_head",
    ] {
        let error = ensure_model_device(tensor, &1_u8, &0_u8)
            .expect_err("different device identities must fail preflight");
        assert!(error.to_string().contains(tensor));
        assert!(error.to_string().contains("model parameters"));
    }
}

/// Dense override layers instantiate ordinary SwiGLU instead of an expert bank.
#[test]
fn dense_override_uses_exactly_one_feed_forward_kind() {
    let device = Default::default();
    let mut config = tiny_config();
    config.mlp_only_layers = vec![1];
    let model = Qwen3MoeForCausalLm::<TestBackend>::init(config, &device).unwrap();
    assert!(model.model.layers[0].sparse_mlp.is_some());
    assert!(model.model.layers[0].dense_mlp.is_none());
    assert!(model.model.layers[1].sparse_mlp.is_none());
    assert!(model.model.layers[1].dense_mlp.is_some());
}

/// Serialized router-output policy is honored in addition to call-site options.
#[test]
fn config_can_enable_router_logits_by_default() {
    let device = Default::default();
    let mut config = tiny_config();
    config.output_router_logits = true;
    let model = initialized_model::<TestBackend>(config, &device);
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(output.routers.len(), 2);
    assert!(output.routers.iter().all(|router| router.logits.is_some()));
}

/// One-shot and incremental native-KV execution agree at the final token.
#[test]
fn stateless_prefill_decode_parity() {
    let (model, device) = tiny_model();
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect("full forward")
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("prefill");
    let cached = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("decode")
        .logits;
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
    assert_eq!(state.lengths(), &[4]);
}

/// Multi-token decode retains causal masking within the appended chunk.
#[test]
fn chunked_decode_matches_full_forward() {
    let (model, device) = tiny_model();
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap()
        .logits
        .slice([0..1, 2..4, 0..32]);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let chunk = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap()
        .logits;
    chunk
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
}

/// Finished rows remain inactive while active peers append ragged chunks.
#[test]
fn mixed_active_and_inactive_decode_rows_preserve_state() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2], [3, 0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true], [true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(state.lengths(), &[2, 1]);
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[4, 5], [0, 0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true], [false, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                router_logits: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(state.lengths(), &[4, 1]);
    let inactive = output.logits.slice([1..2, 0..2, 0..32]).into_data();
    assert!(
        inactive
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&x| x == 0.0)
    );
    for router in output.routers {
        let inactive = router.logits.unwrap().slice([1..2, 0..2, 0..4]).into_data();
        assert!(
            inactive
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|&x| x == 0.0)
        );
    }
}

/// An entirely inactive decode is an exact no-op for cache bytes and lengths.
#[test]
fn all_inactive_decode_is_exact_noop() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let before = state
        .tensor_state()
        .main()
        .native_kv(0)
        .unwrap()
        .keys(0)
        .unwrap()
        .clone()
        .into_data();
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[0, 0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(state.lengths(), &[2]);
    assert_eq!(
        before,
        state
            .tensor_state()
            .main()
            .native_kv(0)
            .unwrap()
            .keys(0)
            .unwrap()
            .clone()
            .into_data()
    );
    assert!(
        output
            .logits
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&x| x == 0.0)
    );
}

/// A later-layer recoverable error leaves earlier pending cache writes unpublished.
#[test]
fn later_layer_failure_is_cache_atomic() {
    let (mut model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let before = state
        .tensor_state()
        .main()
        .native_kv(0)
        .unwrap()
        .keys(0)
        .unwrap()
        .clone()
        .into_data();
    model.model.layers[1].sparse_mlp = None;
    let error = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("invalid later layer must fail");
    assert!(error.to_string().contains("feed-forward topology"));
    assert_eq!(state.lengths(), &[0]);
    assert_eq!(
        before,
        state
            .tensor_state()
            .main()
            .native_kv(0)
            .unwrap()
            .keys(0)
            .unwrap()
            .clone()
            .into_data()
    );
}

/// Reorder/duplicate, safe truncate, reset, and capacity errors use common lifecycle rules.
#[test]
fn decode_state_lifecycle_and_capacity_boundaries() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 3).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2], [3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    state.reorder(&[1, 1]).unwrap();
    assert_eq!(state.lengths(), &[2, 2]);
    state.truncate(&[1, 2]).unwrap();
    assert_eq!(state.lengths(), &[1, 2]);
    let error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[5, 6, 7], [5, 6, 7]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("capacity must be checked before model work");
    assert!(error.to_string().contains("capacity"));
    assert_eq!(state.lengths(), &[1, 2]);
    state.reset().unwrap();
    assert_eq!(state.lengths(), &[0, 0]);
}

/// Optional sliding masks preserve full-cache correctness across a chunk boundary.
#[test]
fn sliding_window_cached_and_stateless_parity() {
    let device = Default::default();
    let mut config = tiny_config();
    config.use_sliding_window = true;
    config.sliding_window = Some(2);
    config.max_window_layers = 0;
    let model = initialized_model::<TestBackend>(config, &device);
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cached = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
}

/// Stateless graph construction remains differentiable under Burn autodiff.
#[cfg(feature = "autodiff")]
#[test]
fn autodiff_smoke_reaches_embedding_weights() {
    use burn::backend::Autodiff;

    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    // Burn's TensorData installation path creates an autodiff non-leaf;
    // lazy initialization is required here so parameter gradients remain
    // tracked as leaf tensors.
    let model = Qwen3MoeForCausalLm::<AdBackend>::init(tiny_config(), &device).unwrap();
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
