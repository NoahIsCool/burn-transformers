use burn::{
    backend::Flex,
    tensor::{
        Bool, Int, Tensor, TensorData, Tolerance,
        backend::{Backend, BackendTypes},
    },
};

use super::*;
use crate::{
    DecodeState, LogitsSelection,
    architecture::TokenMask,
    cache::TensorCacheError,
    models::config::{RopeParameters, RopeType},
    models::qwen3_5::{Qwen3_5LayerType, Qwen3_5TextConfig},
};

type TestBackend = Flex<f32, i32>;

/// Builds a small interleaved topology with every cache family represented.
fn tiny_config() -> Qwen3_5TextConfig {
    Qwen3_5TextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 64,
        layer_types: vec![
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ],
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_conv_kernel_dim: 3,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        mtp_num_hidden_layers: 0,
        ..Default::default()
    }
}

/// Initializes deterministic tiny parameters for parity comparisons.
fn tiny_model(device: &<TestBackend as BackendTypes>::Device) -> Qwen3_5ForCausalLm<TestBackend> {
    TestBackend::seed(device, 211);
    Qwen3_5ForCausalLm::init(tiny_config(), device).expect("tiny Qwen model must initialize")
}

/// Requests every vocabulary position and the complete hidden-state trace.
fn diagnostic_options() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..Default::default()
    }
}

/// Compares two float tensors using tight recurrent/chunked numerical tolerance.
fn assert_close<const D: usize>(actual: Tensor<TestBackend, D>, expected: Tensor<TestBackend, D>) {
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(2e-4, 2e-5));
}

/// Verifies the exact zero-output contract for an inactive decode row.
fn assert_zero<const D: usize>(tensor: Tensor<TestBackend, D>) {
    assert!(
        tensor
            .into_data()
            .to_vec::<f32>()
            .expect("test tensors use f32")
            .iter()
            .all(|&value| value == 0.0)
    );
}

/// Exact physical cache contents for one batch row across every group.
#[derive(Debug, PartialEq)]
struct CacheRowSnapshot {
    tensors: Vec<TensorData>,
}

/// Materializes one row so failed/no-op calls can prove byte-exact isolation.
fn cache_row_snapshot(state: &Qwen3_5DecodeState<TestBackend>, row: usize) -> CacheRowSnapshot {
    let bank = state.tensor_state().main();
    let mut tensors = Vec::new();
    for group in 0..bank.native_kv_count() {
        let cache = bank.native_kv(group).expect("native group");
        for layer in 0..cache.spec().layers() {
            for tensor in [
                cache.keys(layer).expect("keys"),
                cache.values(layer).expect("values"),
            ] {
                let [_, heads, time, width] = tensor.dims();
                tensors.push(
                    tensor
                        .clone()
                        .slice([row..row + 1, 0..heads, 0..time, 0..width])
                        .into_data(),
                );
            }
        }
    }
    for group in 0..bank.recurrent_count() {
        let cache = bank.recurrent(group).expect("recurrent group");
        for layer in 0..cache.spec().layers() {
            let matrix = cache.matrix(layer).expect("matrix");
            let [_, heads, rows, columns] = matrix.dims();
            tensors.push(
                matrix
                    .clone()
                    .slice([row..row + 1, 0..heads, 0..rows, 0..columns])
                    .into_data(),
            );
            let convolution = cache.convolution(layer).expect("convolution");
            let [_, channels, width] = convolution.dims();
            tensors.push(
                convolution
                    .clone()
                    .slice([row..row + 1, 0..channels, 0..width])
                    .into_data(),
            );
        }
    }
    CacheRowSnapshot { tensors }
}

/// The hybrid schedule maps monotonically into exactly two homogeneous groups.
#[test]
fn decode_state_maps_hybrid_layers_deterministically() {
    let device = Default::default();
    let model = tiny_model(&device);
    let state = model
        .new_decode_state(CacheConfig::new(2, 12).expect("cache config"), 2, &device)
        .expect("state must allocate");

    assert_eq!(state.full_attention_layer_count(), 1);
    assert_eq!(state.delta_net_layer_count(), 3);
    assert_eq!(state.layer_cache(0), Some(Qwen3_5LayerCache::DeltaNet(0)));
    assert_eq!(state.layer_cache(1), Some(Qwen3_5LayerCache::DeltaNet(1)));
    assert_eq!(state.layer_cache(2), Some(Qwen3_5LayerCache::DeltaNet(2)));
    assert_eq!(
        state.layer_cache(3),
        Some(Qwen3_5LayerCache::FullAttention(0))
    );
    assert_eq!(state.layer_cache(4), None);
    assert_eq!(state.tensor_state().main().native_kv_count(), 1);
    assert_eq!(state.tensor_state().main().recurrent_count(), 1);
    let recurrent = state
        .tensor_state()
        .main()
        .recurrent(0)
        .expect("recurrent group");
    assert_eq!(recurrent.spec().matrix_dtype(), DType::F32);
    assert_eq!(
        recurrent.spec().convolution_dtype(),
        model.model.embed_tokens.weight.val().dtype()
    );
}

/// A cache config must describe the requested batch and supported model context.
#[test]
fn decode_state_rejects_batch_and_context_mismatch() {
    let device = Default::default();
    let model = tiny_model(&device);
    let batch_error = model
        .new_decode_state(CacheConfig::new(1, 8).expect("cache config"), 2, &device)
        .expect_err("batch mismatch must fail");
    assert!(matches!(
        batch_error,
        Error::TensorCache(TensorCacheError::BatchSizeMismatch { .. })
    ));

    let context_error = model
        .new_decode_state(CacheConfig::new(1, 65).expect("cache config"), 1, &device)
        .expect_err("model context overflow must fail");
    assert!(matches!(context_error, Error::Configuration { .. }));
}

/// Homogeneous schedules allocate no unused cache family and remain exact.
#[test]
fn homogeneous_schedules_decode_without_unused_cache_groups() {
    let device = Default::default();
    for (seed, layer_type) in [
        (301, Qwen3_5LayerType::LinearAttention),
        (302, Qwen3_5LayerType::FullAttention),
    ] {
        let mut config = tiny_config();
        config.num_hidden_layers = 2;
        config.layer_types = vec![layer_type; 2];
        TestBackend::seed(&device, seed);
        let model = Qwen3_5ForCausalLm::init(config, &device).expect("homogeneous model");
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 4).expect("cache config"), 1, &device)
            .expect("decode state");
        let expected = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..Default::default()
                },
            )
            .expect("full forward")
            .logits
            .narrow(1, 2, 2);
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect("prefill");
        let actual = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[3, 4]], &device),
                    position_ids: None,
                    token_mask: None,
                },
                &mut state,
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..Default::default()
                },
            )
            .expect("decode")
            .logits;

        assert_close(actual, expected);
        match layer_type {
            Qwen3_5LayerType::LinearAttention => {
                assert_eq!(state.full_attention_layer_count(), 0);
                assert_eq!(state.delta_net_layer_count(), 2);
                assert_eq!(state.tensor_state().main().native_kv_count(), 0);
                assert_eq!(state.tensor_state().main().recurrent_count(), 1);
            }
            Qwen3_5LayerType::FullAttention => {
                assert_eq!(state.full_attention_layer_count(), 2);
                assert_eq!(state.delta_net_layer_count(), 0);
                assert_eq!(state.tensor_state().main().native_kv_count(), 1);
                assert_eq!(state.tensor_state().main().recurrent_count(), 0);
            }
        }
    }
}

/// A one-wide convolution has no external history axis and is rejected safely.
#[test]
fn stateful_delta_net_rejects_unit_convolution_kernel() {
    let device = Default::default();
    let mut config = tiny_config();
    config.linear_conv_kernel_dim = 1;
    let model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).expect("stateless model");
    let error = model
        .new_decode_state(CacheConfig::new(1, 4).expect("cache config"), 1, &device)
        .expect_err("zero-width convolution history must fail");
    assert!(matches!(error, Error::Configuration { .. }));
}

/// Context-dependent dynamic NTK cannot silently reuse stale rotated keys.
#[test]
fn dynamic_ntk_extension_is_rejected_before_cache_mutation() {
    let device = Default::default();
    let mut config = tiny_config();
    config.rope_parameters = Some(RopeParameters {
        rope_type: RopeType::Dynamic,
        factor: Some(2.0),
        original_max_position_embeddings: Some(2),
        ..Default::default()
    });
    let model = Qwen3_5ForCausalLm::init(config, &device).expect("dynamic-NTK model");
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).expect("cache config"), 1, &device)
        .expect("decode state");
    let error = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("dynamic cache extension must fail");

    assert!(matches!(error, Error::BackendCapability(_)));
    assert_eq!(state.lengths(), &[0]);
}

/// Invalid explicit positions and logit selections fail before state changes.
#[test]
fn invalid_positions_and_logit_selection_are_atomic() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).expect("cache config"), 1, &device)
        .expect("decode state");
    let invalid_position = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1]], &device),
                token_mask: None,
                position_ids: Some(Tensor::<TestBackend, 2, Int>::from_ints([[-1]], &device)),
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("negative position must fail");
    assert!(matches!(invalid_position, Error::Shape { .. }));
    assert_eq!(state.lengths(), &[0]);

    let invalid_selection = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::Positions(Vec::new()),
                ..Default::default()
            },
        )
        .expect_err("empty selection must fail");
    assert!(matches!(invalid_selection, Error::Shape { .. }));
    assert_eq!(state.lengths(), &[0]);
}

/// A recoverable error in a later layer publishes none of the earlier work.
#[test]
fn later_layer_failure_preserves_every_cache_tensor() {
    let device = Default::default();
    let mut model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).expect("cache config"), 1, &device)
        .expect("decode state");
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("baseline prefill");
    let expected = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .expect("retry reference")
        .logits
        .narrow(1, 2, 2);
    let before = cache_row_snapshot(&state, 0);
    let attention = model.model.layers[3]
        .self_attn
        .take()
        .expect("final attention");

    let error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("missing final attention must fail");

    assert!(matches!(error, Error::Checkpoint(_)));
    assert_eq!(state.lengths(), &[2]);
    assert_eq!(cache_row_snapshot(&state, 0), before);

    model.model.layers[3].self_attn = Some(attention);
    let retry = model
        .decode(
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .expect("retry from unchanged baseline")
        .logits;
    assert_close(retry, expected);
    assert_eq!(state.lengths(), &[4]);
}

/// Finished rows remain exact no-ops while active peers reach capacity.
#[test]
fn decode_supports_mixed_active_and_inactive_rows() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 4).expect("cache config"), 2, &device)
        .expect("decode state");
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                    [[1, 2, 0, 0], [5, 6, 7, 8]],
                    &device,
                ),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true, false, false], [true, true, true, true]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("ragged prefill");
    assert_eq!(state.lengths(), &[2, 4]);
    let inactive_before = cache_row_snapshot(&state, 1);
    let expected = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .expect("active-row reference")
        .logits
        .narrow(1, 2, 2);
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[3, 4], [-1, -1]], &device),
                position_ids: Some(Tensor::<TestBackend, 2, Int>::from_ints(
                    [[2, 3], [-1, -1]],
                    &device,
                )),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true], [false, false]].into(),
                    &device,
                ))),
            },
            &mut state,
            &diagnostic_options(),
        )
        .expect("mixed decode");

    assert_close(output.logits.clone().slice([0..1, 0..2, 0..32]), expected);
    assert_zero(output.logits.slice([1..2, 0..2, 0..32]));
    let final_hidden = output.final_hidden_state.expect("requested hidden");
    assert_zero(final_hidden.slice([1..2, 0..2, 0..8]));
    for hidden in output.hidden_states {
        assert_zero(hidden.slice([1..2, 0..2, 0..8]));
    }
    assert_eq!(state.lengths(), &[4, 4]);
    assert_eq!(cache_row_snapshot(&state, 1), inactive_before);
}

/// An entirely inactive decode returns rectangular zeros and changes nothing.
#[test]
fn all_inactive_decode_is_an_exact_no_op() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 2).expect("cache config"), 2, &device)
        .expect("decode state");
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2], [3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("capacity-filling prefill");
    let before = [cache_row_snapshot(&state, 0), cache_row_snapshot(&state, 1)];
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[-1, -1], [-1, -1]], &device),
                position_ids: Some(Tensor::<TestBackend, 2, Int>::from_ints(
                    [[-1, -1], [-1, -1]],
                    &device,
                )),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false, false], [false, false]].into(),
                    &device,
                ))),
            },
            &mut state,
            &ForwardOptions {
                hidden_states: true,
                ..Default::default()
            },
        )
        .expect("inactive decode");

    assert_eq!(output.logits.dims(), [2, 1, 32]);
    assert_zero(output.logits);
    assert_zero(output.final_hidden_state.expect("requested hidden"));
    for hidden in output.hidden_states {
        assert_zero(hidden);
    }
    assert_eq!(state.lengths(), &[2, 2]);
    assert_eq!(
        [cache_row_snapshot(&state, 0), cache_row_snapshot(&state, 1)],
        before
    );
}

/// Full forward and cache-filling prefill produce the same valid graph outputs.
#[test]
fn prefill_matches_full_forward() {
    let device = Default::default();
    let model = tiny_model(&device);
    let input = CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3], [4, 5, 6]], &device),
        token_mask: None,
        position_ids: None,
    };
    let options = diagnostic_options();
    let expected = model
        .forward(input.clone(), &options)
        .expect("stateless forward");
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).expect("cache config"), 2, &device)
        .expect("decode state");
    let actual = model
        .prefill(input, &mut state, &options)
        .expect("cached prefill");

    assert_close(actual.logits, expected.logits);
    assert_close(
        actual.final_hidden_state.expect("requested hidden"),
        expected.final_hidden_state.expect("requested hidden"),
    );
    assert_eq!(actual.hidden_states.len(), expected.hidden_states.len());
    for (actual, expected) in actual.hidden_states.into_iter().zip(expected.hidden_states) {
        assert_close(actual, expected);
    }
    assert_eq!(state.lengths(), &[3, 3]);
}

/// A multi-token recurrent decode equals the corresponding suffix of full forward.
#[test]
fn multi_token_decode_matches_full_forward_suffix() {
    let device = Default::default();
    let model = tiny_model(&device);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let expected = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4, 5]], &device),
                token_mask: None,
                position_ids: None,
            },
            &options,
        )
        .expect("full forward")
        .logits
        .narrow(1, 2, 3);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 5).expect("cache config"), 1, &device)
        .expect("decode state");
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("prefill");
    let actual = model
        .decode(
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[3, 4, 5]], &device),
                position_ids: None,
                token_mask: None,
            },
            &mut state,
            &options,
        )
        .expect("multi-token decode")
        .logits;

    assert_close(actual, expected);
    assert_eq!(state.lengths(), &[5]);
}

/// Ragged masks neither consume positions nor shift DeltaNet convolution state.
#[test]
fn ragged_prefill_and_decode_match_compacted_full_forward() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).expect("cache config"), 2, &device)
        .expect("decode state");
    let last = ForwardOptions::default();

    let prefill = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                    [[1, 2, 3, 4], [5, 6, 0, 0]],
                    &device,
                ),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true, true, true], [true, true, false, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &last,
        )
        .expect("ragged prefill");
    let reference_prefill = Tensor::cat(
        vec![
            model
                .forward(
                    CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                            [[1, 2, 3, 4]],
                            &device,
                        ),
                        token_mask: None,
                        position_ids: None,
                    },
                    &last,
                )
                .expect("row zero reference")
                .logits,
            model
                .forward(
                    CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[5, 6]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    &last,
                )
                .expect("row one reference")
                .logits,
        ],
        0,
    );
    assert_close(prefill.logits, reference_prefill);
    assert_eq!(state.lengths(), &[4, 2]);

    let decoded = model
        .decode(
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[7, 8], [9, 0]], &device),
                position_ids: None,
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true], [true, false]].into(),
                    &device,
                ))),
            },
            &mut state,
            &last,
        )
        .expect("ragged decode");
    let reference_decode = Tensor::cat(
        vec![
            model
                .forward(
                    CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                            [[1, 2, 3, 4, 7, 8]],
                            &device,
                        ),
                        token_mask: None,
                        position_ids: None,
                    },
                    &last,
                )
                .expect("row zero continuation")
                .logits,
            model
                .forward(
                    CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[5, 6, 9]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    &last,
                )
                .expect("row one continuation")
                .logits,
        ],
        0,
    );
    assert_close(decoded.logits, reference_decode);
    assert_eq!(state.lengths(), &[6, 3]);
}

/// Explicit continuation positions equal the default per-sequence offsets.
#[test]
fn explicit_positions_match_implicit_decode_offsets() {
    let device = Default::default();
    let model = tiny_model(&device);
    let config = CacheConfig::new(2, 8).expect("cache config");
    let mut implicit = model
        .new_decode_state(config, 2, &device)
        .expect("implicit state");
    let mut explicit = model
        .new_decode_state(config, 2, &device)
        .expect("explicit state");
    let prompt = || CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3], [4, 5, 6]], &device),
        token_mask: None,
        position_ids: None,
    };
    model
        .prefill(prompt(), &mut implicit, &ForwardOptions::default())
        .expect("implicit prefill");
    model
        .prefill(prompt(), &mut explicit, &ForwardOptions::default())
        .expect("explicit prefill");
    let implicit_output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[7, 8], [9, 10]], &device),
                position_ids: None,
                token_mask: None,
            },
            &mut implicit,
            &diagnostic_options(),
        )
        .expect("implicit decode");
    let explicit_output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[7, 8], [9, 10]], &device),
                position_ids: Some(Tensor::<TestBackend, 2, Int>::from_ints(
                    [[3, 4], [3, 4]],
                    &device,
                )),
                token_mask: None,
            },
            &mut explicit,
            &diagnostic_options(),
        )
        .expect("explicit decode");
    assert_close(implicit_output.logits, explicit_output.logits);
    assert_close(
        implicit_output.final_hidden_state.expect("implicit hidden"),
        explicit_output.final_hidden_state.expect("explicit hidden"),
    );
}

/// Capacity failure is detected before any logical or tensor state changes.
#[test]
fn capacity_boundary_is_preflighted_atomically() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 3).expect("cache config"), 1, &device)
        .expect("decode state");
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("exact-capacity prefill");
    let keys_before = state
        .tensor_state()
        .main()
        .native_kv(0)
        .expect("KV group")
        .keys(0)
        .expect("KV layer")
        .clone()
        .into_data();
    let matrix_before = state
        .tensor_state()
        .main()
        .recurrent(0)
        .expect("recurrent group")
        .matrix(0)
        .expect("recurrent layer")
        .clone()
        .into_data();
    let error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[4]], &device),
                position_ids: None,
                token_mask: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("capacity overflow must fail");
    assert!(matches!(
        error,
        Error::TensorCache(TensorCacheError::Metadata(
            DecodeStateError::CapacityExceeded { .. }
        ))
    ));
    assert_eq!(state.lengths(), &[3]);
    assert_eq!(
        state
            .tensor_state()
            .main()
            .native_kv(0)
            .expect("KV group")
            .keys(0)
            .expect("KV layer")
            .clone()
            .into_data(),
        keys_before
    );
    assert_eq!(
        state
            .tensor_state()
            .main()
            .recurrent(0)
            .expect("recurrent group")
            .matrix(0)
            .expect("recurrent layer")
            .clone()
            .into_data(),
        matrix_before
    );
}

/// Fully masked prompts remain invalid even though decode permits inactive rows.
#[test]
fn fully_masked_prefill_is_rejected_without_advancing_state() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).expect("cache config"), 1, &device)
        .expect("decode state");
    let error = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0, 0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("fully masked row must fail");
    assert!(matches!(error, Error::Shape { .. }));
    assert_eq!(state.lengths(), &[0]);
}

/// The cached mask exposes every old key and only the causal current prefix.
#[test]
fn cached_causal_mask_uses_bottom_right_alignment() {
    let device = Default::default();
    let mask = cached_causal_mask::<TestBackend>(2, 3, 1, &device)
        .expect("causal geometry")
        .into_data();
    assert_eq!(
        mask.as_slice::<bool>().expect("bool mask"),
        &[
            false, false, false, true, true, // query at absolute position 2
            false, false, false, false, true, // query at absolute position 3
            false, false, false, false, false, // query at absolute position 4
        ]
    );
}

/// Scatter semantics preserve valid order and define padding outputs as zero.
#[test]
fn compact_scatter_restores_rectangular_positions() {
    let device = Default::default();
    let compact = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 2, 2]),
        &device,
    );
    let actual = scatter_compact(compact, 4, &[0, 2])
        .expect("valid scatter")
        .into_data();
    assert_eq!(
        actual.as_slice::<f32>().expect("f32 output"),
        &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
    );
}

/// Shifted MTP prefill and tokenwise cache updates match stateless attention.
#[test]
fn mtp_full_and_incremental_paths_are_equivalent() {
    let device = Default::default();
    let mut config = tiny_config();
    config.mtp_num_hidden_layers = 1;
    config.num_hidden_layers = 1;
    config.layer_types = vec![Qwen3_5LayerType::FullAttention];
    TestBackend::seed(&device, 991);
    let model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device);

    let mut full_state = model
        .new_speculative_decode_state(CacheConfig::new(1, 12).unwrap(), 1, 2, &device)
        .unwrap();
    let target = model
        .prefill(
            CausalLmInput {
                token_ids: prompt.clone(),
                token_mask: None,
                position_ids: None,
            },
            &mut full_state,
            &diagnostic_options(),
        )
        .unwrap()
        .final_hidden_state
        .unwrap();
    let shifted_tokens = prompt.clone().narrow(1, 1, 2);
    let shifted_hidden = target.clone().narrow(1, 0, 2);
    let stateless = model
        .forward_mtp(
            shifted_tokens.clone(),
            shifted_hidden.clone(),
            Tensor::<TestBackend, 2, Int>::from_ints([[0, 1]], &device),
            0,
        )
        .unwrap();
    let cached = model
        .forward_mtp_cached(
            shifted_tokens.clone(),
            shifted_hidden.clone(),
            None,
            0,
            &mut full_state,
        )
        .unwrap();
    assert_close(cached.hidden_state, stateless.hidden_state.clone());
    assert_eq!(
        full_state.tensor_state().mtp_lengths(),
        Some([2].as_slice())
    );

    let mut incremental_state = model
        .new_speculative_decode_state(CacheConfig::new(1, 12).unwrap(), 1, 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: prompt,
                token_mask: None,
                position_ids: None,
            },
            &mut incremental_state,
            &diagnostic_options(),
        )
        .unwrap();
    let first = model
        .forward_mtp_cached(
            shifted_tokens.clone().narrow(1, 0, 1),
            shifted_hidden.clone().narrow(1, 0, 1),
            None,
            0,
            &mut incremental_state,
        )
        .unwrap();
    let second = model
        .forward_mtp_cached(
            shifted_tokens.narrow(1, 1, 1),
            shifted_hidden.narrow(1, 1, 1),
            None,
            0,
            &mut incremental_state,
        )
        .unwrap();
    assert_close(second.hidden_state, stateless.hidden_state.narrow(1, 1, 1));
    assert_eq!(first.hidden_state.dims(), [1, 1, 8]);
    assert_eq!(
        incremental_state.tensor_state().mtp_lengths(),
        Some([2].as_slice())
    );
}

/// A rejected target/MTP suffix rolls both tensor banks back byte-exactly.
#[test]
fn speculative_transaction_restores_target_and_mtp_bytes() {
    let device = Default::default();
    let mut config = tiny_config();
    config.mtp_num_hidden_layers = 1;
    TestBackend::seed(&device, 992);
    let model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 12).unwrap(), 1, 2, &device)
        .unwrap();
    let target = model
        .prefill(
            CausalLmInput {
                token_ids: prompt.clone(),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &diagnostic_options(),
        )
        .unwrap()
        .final_hidden_state
        .unwrap();
    model
        .prefill_mtp_cache(prompt, target.clone(), &mut state)
        .unwrap();
    let main_before = cache_row_snapshot(&state, 0);
    let mtp_before = state
        .tensor_state()
        .mtp()
        .unwrap()
        .native_kv(0)
        .unwrap()
        .keys(0)
        .unwrap()
        .clone()
        .into_data();
    let mtp_values_before = state
        .tensor_state()
        .mtp()
        .unwrap()
        .native_kv(0)
        .unwrap()
        .values(0)
        .unwrap()
        .clone()
        .into_data();
    let mark = state.mark().unwrap();
    model
        .forward_mtp_cached(
            Tensor::<TestBackend, 2, Int>::from_ints([[3]], &device),
            target.narrow(1, 1, 1),
            None,
            0,
            &mut state,
        )
        .unwrap();
    model
        .decode(
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &diagnostic_options(),
        )
        .unwrap();
    assert_eq!(state.tensor_state().working_lengths(), &[3]);
    assert_eq!(
        state.tensor_state().mtp_working_lengths(),
        Some([2].as_slice())
    );
    state.rollback(mark).unwrap();
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));
    assert_eq!(cache_row_snapshot(&state, 0), main_before);
    assert_eq!(
        state
            .tensor_state()
            .mtp()
            .unwrap()
            .native_kv(0)
            .unwrap()
            .keys(0)
            .unwrap()
            .clone()
            .into_data(),
        mtp_before
    );
    assert_eq!(
        state
            .tensor_state()
            .mtp()
            .unwrap()
            .native_kv(0)
            .unwrap()
            .values(0)
            .unwrap()
            .clone()
            .into_data(),
        mtp_values_before
    );
}
