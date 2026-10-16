//! Qwen3 MoE numerical, routing, decode-cache, and public-input contract tests.

use burn::{
    backend::Flex,
    module::Param,
    tensor::{Bool, DType, Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::{
    CausalLanguageModel, DecodeInput, DecodeState, LogitsSelection, cache::CacheConfig,
    checkpoint::TensorSink, models::qwen3_moe::Qwen3MoeCheckpointOptions,
};

/// CPU Flex backend with F32 activations and portable I32 indices.
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

/// An enabled window covers every block; `max_window_layers` is inert, as in Transformers 5.2.
///
/// Transformers 4.51 kept blocks `0..max_window_layers` on full attention;
/// the pinned 5.2 reference windows the whole stack (see `reference_tests`
/// for the logit-level golden).
#[test]
fn sliding_window_covers_every_block_whatever_max_window_layers() {
    let mut config = tiny_config();
    config.num_hidden_layers = 8;
    config.use_sliding_window = true;
    config.sliding_window = Some(3);

    for boundary in [0, 1, 4, 8, 9, usize::MAX] {
        config.max_window_layers = boundary;
        config
            .validate()
            .unwrap_or_else(|error| panic!("boundary={boundary}: {error}"));
        for layer_index in 0..config.num_hidden_layers {
            assert_eq!(
                config.layer_sliding_window(layer_index),
                Some(3),
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
    assert!(ensure_same_device("qwen3_moe.input_ids", &0_u8, &0_u8, PARAMETERS).is_ok());
    for tensor in [
        "qwen3_moe.input_ids",
        "qwen3_moe.token_mask",
        "qwen3_moe.position_ids",
        "qwen3_moe.lm_head",
    ] {
        let error = ensure_same_device(tensor, &1_u8, &0_u8, PARAMETERS)
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

/// A later-layer recoverable error publishes nothing, and a retry equals a fresh call.
///
/// Earlier layers may already have written their keys ahead into slots beyond
/// the committed lengths; those slots are unreadable, so the failed call is
/// logically atomic: lengths are unchanged and the retry sees no trace of it.
#[test]
fn later_layer_failure_is_cache_atomic() {
    let (mut model, device) = tiny_model();
    let prompt = || CausalLmInput {
        token_ids: Tensor::from_ints([[1, 2]], &device),
        token_mask: None,
        position_ids: None,
    };
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let sparse = model.model.layers[1].sparse_mlp.take();
    let error = model
        .prefill(prompt(), &mut state, &options)
        .expect_err("invalid later layer must fail");
    assert!(error.to_string().contains("feed-forward topology"));
    assert_eq!(state.lengths(), &[0]);
    model.model.layers[1].sparse_mlp = sparse;
    let retry = model.prefill(prompt(), &mut state, &options).unwrap();
    let mut fresh_state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let fresh = model.prefill(prompt(), &mut fresh_state, &options).unwrap();
    assert_eq!(retry.logits.to_data(), fresh.logits.to_data());
    let next = |state: &mut Qwen3MoeDecodeState<TestBackend>| {
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                state,
                &options,
            )
            .unwrap()
            .logits
            .to_data()
    };
    assert_eq!(next(&mut state), next(&mut fresh_state));
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

/// Builds malformed external inputs without bypassing Burn's tensor constructors.
fn sparse_contract_bad_inputs(
    device: &<TestBackend as BackendTypes>::Device,
) -> Vec<(CausalLmInput<TestBackend>, ForwardOptions, &'static str)> {
    let plain = CausalLmInput {
        token_ids: Tensor::from_ints([[3, 4]], device),
        token_mask: None,
        position_ids: None,
    };
    let mut cases = Vec::new();
    for shape in [[0, 2], [1, 0]] {
        let mut input = plain.clone();
        input.token_ids = Tensor::from_data(TensorData::new(Vec::<i32>::new(), shape), device);
        cases.push((input, ForwardOptions::default(), "qwen3_moe.input_ids"));
    }
    for shape in [[1, 1], [2, 2]] {
        let mut input = plain.clone();
        input.token_mask = Some(TokenMask::new(
            Tensor::<TestBackend, 2, Int>::ones(shape, device).bool(),
        ));
        cases.push((input, ForwardOptions::default(), "qwen3_moe.token_mask"));
        let mut input = plain.clone();
        input.position_ids = Some(Tensor::<TestBackend, 2, Int>::zeros(shape, device));
        cases.push((input, ForwardOptions::default(), "qwen3_moe.position_ids"));
    }
    for positions in [Vec::new(), vec![2]] {
        cases.push((
            plain.clone(),
            ForwardOptions {
                logits: LogitsSelection::Positions(positions),
                ..Default::default()
            },
            "forward_options.logits.positions",
        ));
    }
    cases
}

/// Requires the public shape-error taxonomy rather than an arbitrary failure.
fn assert_sparse_contract_shape(error: Error, context: &str) {
    let Error::Shape {
        tensor,
        expected,
        actual,
    } = error
    else {
        panic!("expected a shape error at {context}, got {error}");
    };
    assert_eq!(tensor, context);
    assert!(!expected.is_empty());
    assert!(!actual.is_empty());
}

/// Stateful validation cannot publish any layer's tensors before reporting an error.
fn sparse_contract_snapshot(state: &Qwen3MoeDecodeState<TestBackend>) -> Vec<TensorData> {
    let cache = state.tensor_state().main().native_kv(0).unwrap();
    (0..state.layer_count())
        .flat_map(|layer| {
            [
                cache.keys(layer).unwrap().to_data(),
                cache.values(layer).unwrap().to_data(),
            ]
        })
        .collect()
}

/// Stateless geometry and selective-logit errors retain the correct public context.
#[test]
fn sparse_public_contract_stateless_shape_errors_are_contextual() {
    let (model, device) = tiny_model();
    for (input, options, context) in sparse_contract_bad_inputs(&device) {
        assert_sparse_contract_shape(model.forward(input, &options).unwrap_err(), context);
    }
    let fully_masked = CausalLmInput {
        token_ids: Tensor::from_ints([[1, 2]], &device),
        token_mask: Some(TokenMask::new(
            Tensor::<TestBackend, 2, Int>::zeros([1, 2], &device).bool(),
        )),
        position_ids: None,
    };
    assert_sparse_contract_shape(
        model
            .forward(fully_masked, &ForwardOptions::default())
            .unwrap_err(),
        "qwen3_moe.token_mask",
    );
}

/// Populated cache bytes and metadata survive prefill/decode errors and a valid retry.
#[test]
fn sparse_public_contract_cached_preflight_errors_are_atomic() {
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
    let before = sparse_contract_snapshot(&state);
    for (input, options, context) in sparse_contract_bad_inputs(&device) {
        for prefill in [true, false] {
            let error = if prefill {
                model.prefill(input.clone(), &mut state, &options)
            } else {
                model.decode(
                    DecodeInput {
                        token_ids: input.token_ids.clone(),
                        token_mask: input.token_mask.clone(),
                        position_ids: input.position_ids.clone(),
                    },
                    &mut state,
                    &options,
                )
            }
            .unwrap_err();
            assert_sparse_contract_shape(error, context);
            assert_eq!(state.lengths(), &[2]);
            assert_eq!(sparse_contract_snapshot(&state), before);
        }
    }
    let batch_error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3], [4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert!(batch_error.to_string().contains("batch"));
    let masked_error = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[3]], &device),
                token_mask: Some(TokenMask::new(
                    Tensor::<TestBackend, 2, Int>::zeros([1, 1], &device).bool(),
                )),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert_sparse_contract_shape(masked_error, "qwen3_moe.token_mask");
    let attention_error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                attentions: true,
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(
        matches!(attention_error, Error::BackendCapability(reason) if reason.contains("attention"))
    );
    assert_eq!(state.lengths(), &[2]);
    assert_eq!(sparse_contract_snapshot(&state), before);

    let actual = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let expected = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    assert_eq!(actual.dims(), [1, 1, 32]);
    assert!(
        actual
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
    actual
        .to_data()
        .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(1e-6));
    assert_eq!(state.lengths(), &[3]);
}

/// Mask holes do not consume cache slots; explicit positions and repeated selections retain order.
#[test]
fn sparse_public_contract_gapped_positions_and_diagnostics_match_compact_tokens() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let options = ForwardOptions {
        logits: LogitsSelection::Positions(vec![3, 1, 3]),
        hidden_states: true,
        router_logits: true,
        ..Default::default()
    };
    for (step, tokens, positions) in [
        (0, [0, 1, 0, 2, 0], [0, 4, 0, 7, 0]),
        (1, [0, 3, 0, 4, 0], [0, 10, 0, 12, 0]),
    ] {
        let segment = CausalLmInput {
            token_ids: Tensor::from_ints([tokens], &device),
            token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                [[false, true, false, true, false]].into(),
                &device,
            ))),
            position_ids: Some(Tensor::from_ints([positions], &device)),
        };
        let actual = if step == 0 {
            model.prefill(segment, &mut state, &options)
        } else {
            model.decode(
                DecodeInput {
                    token_ids: segment.token_ids,
                    token_mask: segment.token_mask,
                    position_ids: segment.position_ids,
                },
                &mut state,
                &options,
            )
        }
        .unwrap();
        let (tokens, positions, selected) = if step == 0 {
            (vec![1_i32, 2], vec![4_i32, 7], vec![1_i32, 0, 1])
        } else {
            (
                vec![1_i32, 2, 3, 4],
                vec![4_i32, 7, 10, 12],
                vec![3_i32, 2, 3],
            )
        };
        let width = tokens.len();
        let expected = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_data(TensorData::new(tokens, [1, width]), &device),
                    token_mask: None,
                    position_ids: Some(Tensor::from_data(
                        TensorData::new(positions, [1, width]),
                        &device,
                    )),
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..Default::default()
                },
            )
            .unwrap()
            .logits
            .select(
                1,
                Tensor::<TestBackend, 1, Int>::from_data(TensorData::new(selected, [3]), &device),
            );
        assert_eq!(actual.logits.dims(), [1, 3, 32]);
        let oracle = expected.to_data().to_vec::<f32>().unwrap();
        assert!(oracle.iter().all(|value| value.is_finite()));
        assert_eq!(&oracle[0..32], &oracle[64..96]);
        assert!(
            oracle[0..32]
                .iter()
                .zip(&oracle[32..64])
                .any(|(left, right)| (left - right).abs() > 2e-6),
            "distinct selected tokens must be distinguishable at the parity tolerance"
        );
        assert!(
            actual
                .logits
                .to_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        actual
            .logits
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(1e-6));
        assert_eq!(state.lengths(), &[2 * (step + 1)]);
        assert_eq!(actual.hidden_states.len(), 4);
        assert_eq!(actual.routers.len(), 2);
        let mut diagnostics = actual.hidden_states;
        diagnostics.push(actual.final_hidden_state.unwrap());
        diagnostics.extend(
            actual
                .routers
                .into_iter()
                .map(|router| router.logits.unwrap()),
        );
        for tensor in diagnostics {
            let [batch, sequence, width] = tensor.dims();
            assert_eq!([batch, sequence], [1, 5]);
            assert!(width == 8 || width == 4);
            let values = tensor.to_data().to_vec::<f32>().unwrap();
            assert!(values.iter().all(|value| value.is_finite()));
            for column in [0, 2, 4] {
                assert!(
                    values[column * width..(column + 1) * width]
                        .iter()
                        .all(|&value| value == 0.0)
                );
            }
        }
    }
}

/// Stateless graph construction remains differentiable under Burn autodiff.
#[cfg(feature = "autodiff")]
#[test]
fn autodiff_smoke_reaches_embedding_weights() {
    use burn::backend::Autodiff;

    /// Autodiff wrapper used to verify gradients reach the embedding parameter.
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

/// Builds a three-layer windowed model: dense layer 0 and sparse layers 1 and 2, all windowed.
fn experiment_model() -> (
    Qwen3MoeForCausalLm<TestBackend>,
    <TestBackend as BackendTypes>::Device,
) {
    let device = Default::default();
    let mut config = tiny_config();
    config.num_hidden_layers = 3;
    config.mlp_only_layers = vec![0];
    config.use_sliding_window = true;
    config.sliding_window = Some(2);
    (initialized_model(config, &device), device)
}

/// A padded two-row input for the experimentation tests.
fn experiment_input(device: &<TestBackend as BackendTypes>::Device) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4], [5, 6, 7, 8]], device),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[false, true, true, true], [true, true, true, true]].into(),
            device,
        ))),
        position_ids: None,
    }
}

/// The catalog lists router points for sparse layers and gate points for dense ones, in order.
#[test]
fn activation_catalog_names_router_and_dense_points() {
    let (model, _) = experiment_model();
    let names = model
        .activation_points()
        .into_iter()
        .map(|point| point.name)
        .collect::<Vec<_>>();
    assert_eq!(names.first().unwrap(), "decoder.embedding");
    assert_eq!(names.last().unwrap(), "decoder.final_norm");
    assert!(names.contains(&"decoder.layers.0.mlp.pre_activation".to_string()));
    assert!(!names.contains(&"decoder.layers.0.mlp.router_logits".to_string()));
    for layer in [1, 2] {
        let router = format!("decoder.layers.{layer}.mlp.router_logits");
        let weights = format!("decoder.layers.{layer}.mlp.router_weights");
        let input = names
            .iter()
            .position(|name| *name == format!("decoder.layers.{layer}.mlp.input"))
            .unwrap();
        assert_eq!(names[input + 1], router);
        assert_eq!(names[input + 2], weights);
        assert_eq!(
            router_logits_point(Layer::decoder(layer)).info().name,
            router
        );
        assert_eq!(
            router_weights_point(Layer::decoder(layer)).info().name,
            weights
        );
    }
}

/// Capturing every point leaves logits, routers, and loss bit-identical to the plain forward.
#[test]
fn observing_every_point_preserves_the_forward_exactly() {
    let (model, device) = experiment_model();
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        router_logits: true,
        auxiliary_losses: true,
        ..Default::default()
    };
    let plain = model.forward(experiment_input(&device), &options).unwrap();
    let mut hooks = crate::intervention::Hooks::new();
    let captures = model
        .activation_points()
        .into_iter()
        .map(|point| hooks.capture_dynamic(point))
        .collect::<Vec<_>>();
    let hooked = model
        .forward_with_hooks(experiment_input(&device), &options, &mut hooks)
        .unwrap();
    assert_eq!(plain.logits.to_data(), hooked.logits.to_data());
    for (left, right) in plain.routers.iter().zip(&hooked.routers) {
        assert_eq!(
            left.logits.as_ref().unwrap().to_data(),
            right.logits.as_ref().unwrap().to_data()
        );
    }
    assert_eq!(
        plain.load_balancing_loss.unwrap().to_data(),
        hooked.load_balancing_loss.unwrap().to_data()
    );
    // Router logits captured at the point equal the reported diagnostics.
    let router = captures
        .iter()
        .find(|capture| capture.point().name == "decoder.layers.1.mlp.router_logits")
        .unwrap();
    let captured: Tensor<TestBackend, 3> =
        hooks.get_dynamic(router).unwrap().into_tensor().unwrap();
    assert_eq!(
        captured.to_data(),
        plain.routers[0].logits.as_ref().unwrap().to_data()
    );
    let weights = captures
        .iter()
        .find(|capture| capture.point().name == "decoder.layers.1.mlp.router_weights")
        .unwrap();
    let weights: Tensor<TestBackend, 3> =
        hooks.get_dynamic(weights).unwrap().into_tensor().unwrap();
    assert_eq!(weights.dims(), [2, 4, 2]);
    // Renormalized top-2 weights sum to one per token.
    weights.sum_dim(2).into_data().assert_approx_eq::<f32>(
        &Tensor::<TestBackend, 3>::ones([2, 4, 1], &device).into_data(),
        Tolerance::rel_abs(1e-6, 1e-6),
    );
}

/// Editing router logits changes which experts run, and zeroing weights removes the routed output.
#[test]
fn router_edits_change_the_routed_computation() {
    let (model, device) = experiment_model();
    let plain = model
        .forward(experiment_input(&device), &ForwardOptions::default())
        .unwrap()
        .logits;
    let mut forced = crate::intervention::Hooks::new();
    // Force experts 2 and 3 for every token of layer 1.
    forced.edit(
        router_logits_point(Layer::decoder(1)),
        |logits: Tensor<TestBackend, 3>| {
            let bonus =
                Tensor::<TestBackend, 1>::from_floats([0.0, 0.0, 50.0, 40.0], &logits.device())
                    .reshape([1, 1, 4]);
            logits + bonus
        },
    );
    let rerouted = model
        .forward_with_hooks(
            experiment_input(&device),
            &ForwardOptions::default(),
            &mut forced,
        )
        .unwrap()
        .logits;
    assert_ne!(plain.to_data(), rerouted.to_data());

    let mut silenced = crate::intervention::Hooks::new();
    silenced.edit(
        router_weights_point(Layer::decoder(1)),
        |weights: Tensor<TestBackend, 3>| weights.zeros_like(),
    );
    let mlp = silenced.capture(Layer::decoder(1).mlp_output());
    model
        .forward_with_hooks(
            experiment_input(&device),
            &ForwardOptions::default(),
            &mut silenced,
        )
        .unwrap();
    let output = silenced.get(&mlp).unwrap();
    assert!(
        output
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0),
        "zero mixture weights must cancel every routed contribution"
    );
}

/// Staged execution and the generic traits reproduce the fused backbone exactly.
#[test]
fn staged_execution_matches_the_fused_backbone() {
    use crate::architecture::{HookedModel, StagedModel};

    let (model, device) = experiment_model();
    let input = experiment_input(&device);
    let fused = model.model.forward(&input, true, true).unwrap();
    assert_eq!(fused.hidden_states.len(), 5);
    assert_eq!(fused.router_logits.len(), 2);
    // One block at a time: each range builds its own tables.
    let mut state = model.model.embed(&input).unwrap();
    for index in 0..3 {
        state = model.model.run_layers(state, index..index + 1).unwrap();
    }
    let staged = model.model.normalize(state).unwrap();
    assert_eq!(staged.hidden.to_data(), fused.hidden.to_data());
    for traced in [
        StagedModel::run_stack(&model, input.clone()).unwrap(),
        StagedModel::run_stack(&model.model, input.clone()).unwrap(),
    ] {
        assert_eq!(traced.hidden.to_data(), fused.hidden.to_data());
    }
    assert_eq!(StagedModel::num_layers(&model), 3);
    let points = HookedModel::activation_points(&model).unwrap();
    assert_eq!(points, model.activation_points());
    // A hooked range emits exactly its own blocks' points.
    let mut hooks = crate::intervention::Hooks::new();
    let router = hooks.capture(router_logits_point(Layer::decoder(2)));
    let state = model.model.embed(&input).unwrap();
    let state = model.model.run_layers(state, 0..2).unwrap();
    model
        .model
        .run_layers_with_hooks(state, 2..3, &mut hooks)
        .unwrap();
    assert_eq!(
        hooks.get(&router).unwrap().to_data(),
        fused.router_logits[1].to_data()
    );
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
/// per-call table copy on Flex); the logits must not change beyond rounding.
#[test]
fn tied_readout_matches_reshaping_linear_oracle() {
    let device = Default::default();
    let config = Qwen3MoeConfig {
        tie_word_embeddings: true,
        ..tiny_config()
    };
    let model =
        Qwen3MoeForCausalLm::<TestBackend>::init(config.clone(), &device).expect("tied tiny model");
    assert!(model.lm_head.is_none());
    let hidden = readout_hidden(config.hidden_size, &device);
    let oracle = burn::tensor::module::linear(
        hidden.clone(),
        model.model.embed_tokens.weight.val().transpose(),
        None,
    )
    .into_data();
    model
        .project_logits(hidden)
        .expect("tied head")
        .into_data()
        .assert_approx_eq::<f32>(&oracle, Tolerance::absolute(1e-6));
}

/// Message carried by the deliberately failing hook callbacks below.
const HOOK_FAILURE: &str = "deliberate hook failure";

/// Registers one edit at `point` whose callback fails with [`HOOK_FAILURE`].
///
/// The edit is addressed by a custom point equal to the catalog entry, so it
/// is accepted by the scope's catalog check and fires when the point is emitted.
fn failing_hooks(point: &PointInfo) -> Hooks<'static, TestBackend> {
    /// Builds the callback error returned at every point.
    fn failure() -> Error {
        Error::configuration("test.hook", HOOK_FAILURE)
    }
    let mut hooks = Hooks::new();
    match point.rank() {
        3 => hooks.try_edit(
            Point::<3>::custom(point.name.clone(), point.axes.clone().try_into().unwrap()),
            |_| Err(failure()),
        ),
        4 => hooks.try_edit(
            Point::<4>::custom(point.name.clone(), point.axes.clone().try_into().unwrap()),
            |_| Err(failure()),
        ),
        rank => panic!("{}: unexpected rank {rank}", point.name),
    }
    hooks
}

/// Asserts that a hooked call failed with the deliberate callback error.
fn assert_hook_failure<T: core::fmt::Debug>(result: Result<T>, point: &PointInfo) {
    let error = result.expect_err(&point.name);
    assert!(
        error.to_string().contains(HOOK_FAILURE),
        "{}: {error}",
        point.name
    );
}

/// A failing callback at any catalog point aborts the fused and staged passes with its error.
///
/// Every point of the dense and both routed blocks (including the eager
/// scores and pattern of layer 1) is emitted on both execution paths, and the
/// user's error is returned unchanged rather than swallowed or replaced.
#[test]
fn a_failing_edit_at_any_point_aborts_with_its_error() {
    let (mut model, device) = experiment_model();
    model.model.layers[1].self_attn.precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
    let input = experiment_input(&device);
    let points = model.activation_points();
    assert!(
        points
            .iter()
            .any(|point| point.name == "decoder.layers.1.attention.pattern")
    );
    for point in &points {
        let mut hooks = failing_hooks(point);
        assert_hook_failure(
            model.forward_with_hooks(input.clone(), &ForwardOptions::default(), &mut hooks),
            point,
        );
    }
    for point in &points {
        let mut hooks = failing_hooks(point);
        let staged = if *point == Stack::Decoder.embedding().info() {
            model.model.embed_with_hooks(&input, &mut hooks).map(|_| ())
        } else if *point == Stack::Decoder.final_norm().info() {
            let state = model.model.embed(&input).unwrap();
            let state = model.model.run_layers(state, 0..3).unwrap();
            model
                .model
                .normalize_with_hooks(state, &mut hooks)
                .map(|_| ())
        } else {
            let state = model.model.embed(&input).unwrap();
            model
                .model
                .run_layers_with_hooks(state, 0..3, &mut hooks)
                .map(|_| ())
        };
        assert_hook_failure(staged, point);
    }
}

/// Eager-softmax blocks add score and pattern points whose rows are probability distributions.
///
/// The captured `[batch, heads, query, key]` pattern normalizes over keys for
/// every query that sees at least one valid key: all four queries of the
/// unpadded row and queries 1..3 of the row whose first column is padding.
/// Those padded-row queries give the padded key exactly zero weight, and no
/// query of either row attends to a strictly later key.
#[test]
fn eager_softmax_blocks_expose_scores_and_patterns() {
    let (mut model, device) = experiment_model();
    let fused_points = model.activation_points().len();
    model.model.layers[2].self_attn.precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
    let names = model
        .activation_points()
        .into_iter()
        .map(|point| point.name)
        .collect::<Vec<_>>();
    assert_eq!(names.len(), fused_points + 2);
    let key = names
        .iter()
        .position(|name| name == "decoder.layers.2.attention.key")
        .unwrap();
    assert_eq!(
        names[key + 2..key + 5],
        [
            "decoder.layers.2.attention.scores".to_string(),
            "decoder.layers.2.attention.pattern".to_string(),
            "decoder.layers.2.attention.head_context".to_string(),
        ]
    );
    let mut hooks = Hooks::new();
    let pattern = hooks.capture(Layer::decoder(2).self_attention().pattern());
    model
        .forward_with_hooks(
            experiment_input(&device),
            &ForwardOptions::default(),
            &mut hooks,
        )
        .unwrap();
    let pattern = hooks.get(&pattern).unwrap();
    assert_eq!(pattern.dims(), [2, 2, 4, 4]);
    // Row 0 has a padded first column: its first query attends to nothing
    // valid, so it is the only query without a normalization requirement.
    pattern
        .clone()
        .slice([1..2, 0..2, 0..4, 0..4])
        .sum_dim(3)
        .into_data()
        .assert_approx_eq::<f32>(
            &Tensor::<TestBackend, 4>::ones([1, 2, 4, 1], &device).into_data(),
            Tolerance::absolute(1e-5),
        );
    pattern
        .clone()
        .slice([0..1, 0..2, 1..4, 0..4])
        .sum_dim(3)
        .into_data()
        .assert_approx_eq::<f32>(
            &Tensor::<TestBackend, 4>::ones([1, 2, 3, 1], &device).into_data(),
            Tolerance::absolute(1e-5),
        );
    let weights = pattern.into_data().to_vec::<f32>().unwrap();
    let weight = |row: usize, head: usize, query: usize, key: usize| {
        weights[((row * 2 + head) * 4 + query) * 4 + key]
    };
    for head in 0..2 {
        for query in 1..4 {
            assert_eq!(weight(0, head, query, 0), 0.0, "padded key, query {query}");
        }
        for row in 0..2 {
            for query in 0..4 {
                for key in query + 1..4 {
                    assert_eq!(
                        weight(row, head, query, key),
                        0.0,
                        "future key {key} of query {query} in row {row}"
                    );
                }
            }
        }
    }
}

/// Step-wise routing with no hook action equals the fused sparse module under both precisions.
#[test]
fn stepwise_routing_matches_the_fused_module_for_both_weighting_precisions() {
    let (mut model, device) = experiment_model();
    let input = readout_hidden(8, &device);
    for precision in [
        RoutedWeightingPrecision::Fp32ThenResidual,
        RoutedWeightingPrecision::Residual,
    ] {
        let moe = model.model.layers[1].sparse_mlp.as_mut().unwrap();
        moe.weighting_precision = precision;
        let fused = moe.forward(input.clone()).unwrap();
        let (hidden, logits) =
            routed_with_hooks(moe, input.clone(), Layer::decoder(1), &mut Hooks::new()).unwrap();
        assert_eq!(
            hidden.to_data(),
            fused.hidden_state.to_data(),
            "{precision:?}"
        );
        assert_eq!(
            logits.to_data(),
            fused.router_logits.to_data(),
            "{precision:?}"
        );
    }
}

/// Step-wise routing rejects shared experts it cannot expose and inputs of the wrong extent.
#[test]
fn stepwise_routing_rejects_shared_experts_and_malformed_inputs() {
    let (mut model, device) = experiment_model();
    let layer = Layer::decoder(1);
    let moe = model.model.layers[1].sparse_mlp.as_mut().unwrap();
    for shape in [[0, 2, 8], [1, 0, 8], [1, 2, 7]] {
        let error = routed_with_hooks(moe, Tensor::zeros(shape, &device), layer, &mut Hooks::new())
            .expect_err("malformed router input");
        assert!(error.to_string().contains("moe.input"), "{error}");
    }
    let shared = moe.routed[0].clone();
    moe.shared.push(shared);
    let error = routed_with_hooks(moe, readout_hidden(8, &device), layer, &mut Hooks::new())
        .expect_err("shared experts have no router point");
    assert!(matches!(error, Error::BackendCapability(_)));
    assert!(error.to_string().contains("shared experts"), "{error}");
}

/// The runtime implements only the released zero-dropout attention.
#[test]
fn initialization_rejects_attention_dropout() {
    let config = Qwen3MoeConfig {
        attention_dropout: 0.1,
        ..tiny_config()
    };
    config
        .validate()
        .expect("a dropout probability is a valid config");
    let error = Qwen3MoeForCausalLm::<TestBackend>::init(config, &Default::default())
        .expect_err("dropout is not implemented");
    assert!(matches!(error, Error::BackendCapability(_)));
    assert!(error.to_string().contains("zero-dropout"), "{error}");
}

/// The hooked headless backbone equals the plain backbone and records its traces.
#[test]
fn hooked_backbone_matches_the_plain_backbone() {
    let (model, device) = experiment_model();
    let input = experiment_input(&device);
    let plain = model.model.forward(&input, true, true).unwrap();
    let mut hooks = Hooks::new();
    let residual = hooks.capture(Layer::decoder(2).residual_post());
    let hooked = model
        .model
        .forward_with_hooks(&input, true, true, &mut hooks)
        .unwrap();
    assert_eq!(hooked.hidden.to_data(), plain.hidden.to_data());
    assert_eq!(hooked.hidden_states.len(), plain.hidden_states.len());
    assert_eq!(hooked.router_logits.len(), 2);
    for (left, right) in hooked.router_logits.iter().zip(&plain.router_logits) {
        assert_eq!(left.to_data(), right.to_data());
    }
    assert_eq!(
        hooks.get(&residual).unwrap().to_data(),
        plain.hidden_states[3].to_data()
    );
    let without = model.model.forward(&input, false, false).unwrap();
    assert!(without.hidden_states.is_empty() && without.router_logits.is_empty());
    assert_eq!(without.hidden.to_data(), plain.hidden.to_data());
}

/// Staged execution rejects states the blocks cannot evaluate before any block runs.
#[test]
fn staged_execution_rejects_invalid_states_and_policies() {
    let (model, device) = experiment_model();
    let input = experiment_input(&device);
    let fresh = || model.model.embed(&input).unwrap();
    let message = |result: Result<HiddenState<TestBackend>>| {
        result.expect_err("the staged call must fail").to_string()
    };

    // Final normalization happens exactly once, at a valid block boundary.
    let normalized = model.model.normalize(fresh()).unwrap();
    assert!(message(model.model.normalize(normalized)).contains("unnormalized"));
    let mut beyond = fresh();
    beyond.next_layer = 4;
    assert!(message(model.model.normalize(beyond)).contains("valid block boundary"));

    // Latents must keep the parameters' dtype.
    let mut half = fresh();
    half.hidden = half.hidden.cast(DType::F16);
    assert!(message(model.model.run_layers(half.clone(), 0..1)).contains("dtype must match"));
    assert!(message(model.model.normalize(half)).contains("dtype must match"));

    // Decoder-only blocks have no cross-attention source, default or per layer.
    let mut cross = fresh();
    cross.context.cross_attention = Some(AttentionPolicy::causal());
    assert!(message(model.model.run_layers(cross, 0..1)).contains("cross-attention"));
    let mut layer_cross = fresh();
    layer_cross
        .context
        .layer_cross_attention
        .push((Layer::decoder(0), AttentionPolicy::causal()));
    assert!(message(model.model.run_layers(layer_cross, 0..1)).contains("cross-attention"));
    let mut encoded = fresh();
    encoded.context.encoder = Some(crate::experiment::EncoderContext {
        hidden: Tensor::zeros([2, 3, 8], &device),
        positions: Tensor::<TestBackend, 1, Int>::arange(0..3, &device)
            .reshape([1, 3])
            .expand([2, 3]),
        token_mask: None,
    });
    assert!(message(model.model.run_layers(encoded, 0..1)).contains("cross-attention"));

    // Overrides must name an existing block.
    let mut stray = fresh();
    stray.context = stray
        .context
        .with_layer_attention(Layer::decoder(3), AttentionPolicy::bidirectional());
    assert!(message(model.model.run_layers(stray, 0..1)).contains("unavailable"));

    // A score bias must match the activation dtype.
    let mut biased = fresh();
    biased.context = biased.context.with_layer_attention(
        Layer::decoder(0),
        AttentionPolicy::causal()
            .with_bias(Tensor::<TestBackend, 4>::zeros([1, 1, 4, 4], &device).cast(DType::F16)),
    );
    assert!(message(model.model.run_layers(biased, 0..1)).contains("bias dtype"));

    // A matching bias of zeros leaves the block unchanged.
    let mut zero_bias = fresh();
    zero_bias.context = zero_bias.context.with_layer_attention(
        Layer::decoder(0),
        AttentionPolicy::causal().with_bias(Tensor::zeros([1, 1, 4, 4], &device)),
    );
    assert_eq!(
        model
            .model
            .run_layers(zero_bias, 0..1)
            .unwrap()
            .hidden
            .to_data(),
        model
            .model
            .run_layers(fresh(), 0..1)
            .unwrap()
            .hidden
            .to_data()
    );
}

/// Attention-probability capture is unavailable on the fused stateless paths.
#[test]
fn attention_probability_capture_is_rejected() {
    let (model, device) = experiment_model();
    let options = ForwardOptions {
        attentions: true,
        ..Default::default()
    };
    let plain = model
        .forward(experiment_input(&device), &options)
        .expect_err("fused attention returns no probabilities");
    let hooked = model
        .forward_with_hooks(experiment_input(&device), &options, &mut Hooks::new())
        .expect_err("hooked attention returns no probabilities");
    for error in [plain, hooked] {
        assert!(matches!(error, Error::BackendCapability(_)));
        assert!(
            error.to_string().contains("attention-probability"),
            "{error}"
        );
    }
}

/// The generic hooked and staged traits delegate to the family methods.
#[test]
fn generic_traits_delegate_to_the_family_methods() {
    use crate::architecture::{HookedModel, StagedModel};

    let (model, device) = experiment_model();
    let options = ForwardOptions {
        router_logits: true,
        ..Default::default()
    };
    let direct = model.forward(experiment_input(&device), &options).unwrap();
    let mut hooks = Hooks::new();
    let embedding = hooks.capture(Stack::Decoder.embedding());
    let generic =
        HookedModel::forward_with_hooks(&model, experiment_input(&device), &options, &mut hooks)
            .unwrap();
    assert_eq!(generic.logits.to_data(), direct.logits.to_data());
    assert_eq!(generic.routers.len(), 2);
    assert_eq!(hooks.get(&embedding).unwrap().dims(), [2, 4, 8]);
    assert_eq!(StagedModel::stack(&model), Stack::Decoder);
    assert_eq!(StagedModel::stack(&model.model), Stack::Decoder);
    assert_eq!(StagedModel::num_layers(&model.model), 3);
}

/// The stack-wide loss rejects empty router tensors and layers whose extents disagree.
#[test]
fn auxiliary_loss_rejects_empty_and_disagreeing_layers() {
    let device = Default::default();
    for shape in [[0, 2, 3], [1, 0, 3], [1, 2, 0]] {
        let error = qwen3_moe_load_balancing_loss(
            &[Tensor::<TestBackend, 3>::zeros(shape, &device)],
            1,
            None,
        )
        .expect_err("an empty router tensor has no objective");
        assert!(error.to_string().contains("router_logits[0]"), "{error}");
    }
    let error = qwen3_moe_load_balancing_loss(
        &[
            Tensor::<TestBackend, 3>::zeros([1, 2, 3], &device),
            Tensor::<TestBackend, 3>::zeros([1, 3, 3], &device),
        ],
        1,
        None,
    )
    .expect_err("every sparse layer must share the token grid");
    assert!(error.to_string().contains("router_logits[1]"), "{error}");
    let error = qwen3_moe_load_balancing_loss(
        &[Tensor::<TestBackend, 3>::zeros([1, 2, 3], &device)],
        0,
        None,
    )
    .expect_err("zero experts per token is invalid");
    assert!(error.to_string().contains("experts_per_token"), "{error}");
}

/// A NaN activation makes step-wise routing fail with a contextual error instead of indexing.
#[test]
fn stepwise_routing_rejects_non_finite_router_scores() {
    let (model, device) = experiment_model();
    let moe = model.model.layers[1].sparse_mlp.as_ref().unwrap();
    let mut values = vec![0.25_f32; 2 * 8];
    values[8] = f32::NAN;
    let input = Tensor::<TestBackend, 3>::from_data(TensorData::new(values, [1, 2, 8]), &device);
    let stepwise = routed_with_hooks(moe, input.clone(), Layer::decoder(1), &mut Hooks::new())
        .expect_err("a NaN token has no expert ranking");
    let fused = moe
        .forward(input)
        .expect_err("the fused module rejects it too");
    assert_eq!(stepwise.to_string(), fused.to_string());
    assert!(stepwise.to_string().contains("token 1"), "{stepwise}");
}
