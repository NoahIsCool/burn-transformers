//! Qwen hybrid cache parity, masked-row, lifecycle, and failure-atomicity tests.

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

/// CPU Flex backend with F32 activations and portable I32 indices.
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
    /// Materialized main-cache keys, values, association matrices, and convolution history.
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

/// A context-dependent rotary policy cannot silently reuse stale rotated keys.
///
/// LongRoPE switches from its short to its long frequency table once the
/// evaluated context exceeds the original training context `L = 2`; keys
/// cached under the short table would then disagree with new queries, so a
/// three-token prefill must fail before any cache mutation.
#[test]
fn context_dependent_rope_extension_is_rejected_before_cache_mutation() {
    let device = Default::default();
    let mut config = tiny_config();
    config.rope_parameters = Some(RopeParameters {
        rope_type: RopeType::LongRope,
        factor: Some(2.0),
        original_max_position_embeddings: Some(2),
        short_factor: Some(vec![1.0, 1.0]),
        long_factor: Some(vec![1.5, 2.0]),
        ..Default::default()
    });
    let model = Qwen3_5ForCausalLm::init(config, &device).expect("LongRoPE model");
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
        .expect_err("a context beyond the original LongRoPE window must fail");

    assert!(matches!(error, Error::BackendCapability(_)));
    assert_eq!(state.lengths(), &[0]);
}

/// Dynamic NTK scales against `max_position_embeddings`, which bounds the cache.
///
/// Under the Transformers rule the trained context of dynamic NTK is
/// `max_position_embeddings`, and neither the cache capacity nor explicit
/// positions may exceed it, so cached decoding never leaves the unscaled
/// regime and must reproduce the stateless forward.
#[test]
fn dynamic_ntk_cached_decode_matches_stateless_within_trained_context() {
    let device = Default::default();
    let mut config = tiny_config();
    config.rope_parameters = Some(RopeParameters {
        rope_type: RopeType::Dynamic,
        factor: Some(2.0),
        ..Default::default()
    });
    TestBackend::seed(&device, 211);
    let model = Qwen3_5ForCausalLm::init(config, &device).expect("dynamic-NTK model");
    let tokens = [[1, 2, 3, 4]];
    let reference = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints(tokens, &device),
                token_mask: None,
                position_ids: None,
            },
            &diagnostic_options(),
        )
        .expect("stateless dynamic-NTK forward");
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).expect("cache config"), 1, &device)
        .expect("decode state");
    let prefill = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &diagnostic_options(),
        )
        .expect("cached dynamic-NTK prefill");
    assert_close(prefill.logits, reference.logits.clone().narrow(1, 0, 3));
    let decode = model
        .decode(
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &diagnostic_options(),
        )
        .expect("cached dynamic-NTK decode");
    assert_close(decode.logits, reference.logits.narrow(1, 3, 1));
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
    let actual =
        crate::models::shared::rows::CompactRow::<TestBackend>::new(0, &[0, 2], 4, &device)
            .and_then(|row| row.scatter(compact))
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

/// Captures committed bytes, proposal cursors, and both logical cache timelines.
#[derive(Debug, PartialEq)]
struct ContractSnapshot {
    /// Physical main-cache contents captured independently for every batch row.
    rows: Vec<CacheRowSnapshot>,
    /// Full native-KV buffers from the main bank and optional MTP bank.
    auxiliary: Vec<TensorData>,
    /// Committed target-cache cursor for each sequence.
    lengths: Vec<usize>,
    /// Target cursor including unpublished transactional appends.
    working: Vec<usize>,
    /// Committed shifted MTP cursors when the auxiliary bank is enabled.
    mtp_lengths: Option<Vec<usize>>,
    /// MTP cursors including unpublished transactional appends.
    mtp_working: Option<Vec<usize>>,
    /// Per-layer, per-row staging lengths across both cache banks.
    staged: Vec<usize>,
    /// Whether a transaction mark is active at capture time.
    transaction: bool,
}

/// Includes every layer's staging cursor so unpublished proposals cannot hide a write.
fn contract_snapshot(state: &Qwen3_5DecodeState<TestBackend>) -> ContractSnapshot {
    let tensors = state.tensor_state();
    let mut auxiliary = Vec::new();
    let mut staged = Vec::new();
    for bank in std::iter::once(tensors.main()).chain(tensors.mtp()) {
        for group in 0..bank.native_kv_count() {
            let cache = bank.native_kv(group).unwrap();
            for layer in 0..cache.spec().layers() {
                auxiliary.push(cache.keys(layer).unwrap().clone().into_data());
                auxiliary.push(cache.values(layer).unwrap().clone().into_data());
                for row in 0..state.batch_size() {
                    staged.push(cache.staged_length(layer, row).unwrap());
                }
            }
        }
        for group in 0..bank.recurrent_count() {
            let cache = bank.recurrent(group).unwrap();
            for layer in 0..cache.spec().layers() {
                for row in 0..state.batch_size() {
                    staged.push(cache.staged_length(layer, row).unwrap());
                }
            }
        }
    }
    ContractSnapshot {
        rows: (0..state.batch_size())
            .map(|row| cache_row_snapshot(state, row))
            .collect(),
        auxiliary,
        lengths: state.lengths().to_vec(),
        working: tensors.working_lengths().to_vec(),
        mtp_lengths: tensors.mtp_lengths().map(<[usize]>::to_vec),
        mtp_working: tensors.mtp_working_lengths().map(<[usize]>::to_vec),
        staged,
        transaction: tensors.has_active_transaction(),
    }
}

/// Produces a complete, nonzero synthetic write set with the real cache geometry.
fn contract_pending(
    state: &Qwen3_5DecodeState<TestBackend>,
    counts: &[usize],
) -> PendingQwenCacheWrites<TestBackend> {
    let mut pending = PendingQwenCacheWrites::new();
    for (row, &count) in counts.iter().enumerate().filter(|(_, count)| **count > 0) {
        if let Some(spec) = state.layout.native_kv {
            for layer in 0..spec.layers() {
                let cache = state.tensors.main().native_kv(0).unwrap();
                let keys = cache
                    .keys(layer)
                    .unwrap()
                    .clone()
                    .narrow(0, row, 1)
                    .narrow(2, 0, count)
                    + 1.0;
                pending.native_kv.push(PendingNativeKvWrite {
                    cache_layer: layer,
                    sequence: row,
                    start: state.tensors.working_lengths()[row],
                    values: keys.clone() + 1.0,
                    keys,
                });
            }
        }
        if let Some(spec) = state.layout.recurrent {
            let cache = state.tensors.main().recurrent(0).unwrap();
            for layer in 0..spec.layers() {
                pending.recurrent.push(PendingRecurrentWrite {
                    cache_layer: layer,
                    sequence: row,
                    token_count: count,
                    matrix: cache.matrix(layer).unwrap().clone().narrow(0, row, 1) + 1.0,
                    convolution: cache.convolution(layer).unwrap().clone().narrow(0, row, 1) + 1.0,
                });
            }
        }
    }
    pending
}

/// Builds an MTP-enabled model without changing the ordinary hybrid fixture.
fn contract_mtp_model(
    device: &<TestBackend as BackendTypes>::Device,
) -> Qwen3_5ForCausalLm<TestBackend> {
    let mut config = tiny_config();
    config.mtp_num_hidden_layers = 1;
    Qwen3_5ForCausalLm::init(config, device).unwrap()
}

/// Malformed rectangles and active out-of-context positions leave a populated cache intact.
#[test]
fn cache_contract_invalid_rectangles_are_atomic() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 6).unwrap(), 1, &device)
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
    let before = contract_snapshot(&state);
    for (case, context) in [
        (0, "qwen.input_ids"),
        (1, "batch"),
        (2, "qwen.token_mask"),
        (3, "qwen.position_ids"),
        (4, "qwen.position_ids"),
    ] {
        let mut input = DecodeInput {
            token_ids: Tensor::from_ints([[3]], &device),
            token_mask: None,
            position_ids: None,
        };
        match case {
            0 => input.token_ids = Tensor::zeros([1, 0], &device),
            1 => input.token_ids = Tensor::zeros([2, 1], &device),
            2 => {
                input.token_mask = Some(TokenMask::new(Tensor::from_bool(
                    [[true, false]].into(),
                    &device,
                )))
            }
            3 => input.position_ids = Some(Tensor::zeros([1, 2], &device)),
            4 => input.position_ids = Some(Tensor::from_ints([[64]], &device)),
            _ => unreachable!(),
        }
        let error = model
            .decode(input, &mut state, &ForwardOptions::default())
            .unwrap_err();
        assert!(error.to_string().contains(context), "case {case}: {error}");
        assert_eq!(contract_snapshot(&state), before, "case {case}");
    }
}

/// A schedule mismatch is rejected before either physical cache family is touched.
#[test]
fn cache_contract_model_schedule_mismatch_is_atomic() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    let mut config = tiny_config();
    config.layer_types.swap(0, 3);
    let other = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let before = contract_snapshot(&state);
    let error = other
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("incompatible model layout"));
    assert_eq!(contract_snapshot(&state), before);
}

/// Missing rows or layers cannot advance metadata despite otherwise well-shaped tensors.
#[test]
fn cache_contract_pending_cardinality_is_atomic() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    let before = contract_snapshot(&state);
    for case in 0..3 {
        let mut pending = contract_pending(&state, &[1]);
        let counts: &[usize] = if case == 0 { &[] } else { &[1] };
        if case == 1 {
            pending.native_kv.clear();
        }
        if case == 2 {
            pending.recurrent.pop();
        }
        let error = pending
            .commit(&mut state.tensors, &state.layout, counts)
            .unwrap_err();
        assert!(error.to_string().contains(if case == 0 {
            "describe 0 rows"
        } else {
            "coverage mismatch"
        }));
        assert_eq!(contract_snapshot(&state), before);
    }
}

/// Native appends must have unique active coordinates, the working cursor, and exact time axes.
#[test]
fn cache_contract_pending_native_coordinates_are_atomic() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 4).unwrap(), 2, &device)
        .unwrap();
    let before = contract_snapshot(&state);
    for case in 0..7 {
        let counts = if case == 6 { [1, 0] } else { [1, 1] };
        let mut pending = contract_pending(&state, &counts);
        let write = pending.native_kv.last_mut().unwrap();
        match case {
            0 => write.sequence = 2,
            1 => write.cache_layer = 1,
            2 => write.start = 1,
            3 => write.keys = Tensor::zeros([1, 1, 2, 4], &device),
            4 => write.values = Tensor::zeros([1, 1, 2, 4], &device),
            5 => write.sequence = 0,
            6 => write.sequence = 1,
            _ => unreachable!(),
        }
        let error = pending
            .commit(&mut state.tensors, &state.layout, &counts)
            .unwrap_err();
        assert!(
            error.to_string().contains(if case == 0 {
                "outside 2"
            } else {
                "invalid or duplicate Qwen native write"
            }),
            "case {case}: {error}"
        );
        assert_eq!(contract_snapshot(&state), before, "case {case}");
    }
}

/// DeltaNet replacements cover every active row and layer exactly once at the declared token count.
#[test]
fn cache_contract_pending_recurrent_coordinates_are_atomic() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 4).unwrap(), 2, &device)
        .unwrap();
    let before = contract_snapshot(&state);
    for case in 0..5 {
        let counts = if case == 4 { [1, 0] } else { [1, 1] };
        let mut pending = contract_pending(&state, &counts);
        let write = pending.recurrent.last_mut().unwrap();
        match case {
            0 => write.sequence = 2,
            1 => write.cache_layer = 3,
            2 => write.token_count = 2,
            3 => write.cache_layer = 0,
            4 => write.sequence = 1,
            _ => unreachable!(),
        }
        let error = pending
            .commit(&mut state.tensors, &state.layout, &counts)
            .unwrap_err();
        assert!(
            error.to_string().contains(if case == 0 {
                "outside 2"
            } else {
                "invalid or duplicate Qwen recurrent write"
            }),
            "case {case}: {error}"
        );
        assert_eq!(contract_snapshot(&state), before, "case {case}");
    }
}

/// Failure preparing a late recurrent tensor cannot publish earlier prepared KV appends.
#[test]
fn cache_contract_pending_tensor_geometry_is_atomic() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    let before = contract_snapshot(&state);
    let mut pending = contract_pending(&state, &[1]);
    pending.recurrent.last_mut().unwrap().matrix = Tensor::zeros([1, 2, 3, 4], &device);
    let error = pending
        .commit(&mut state.tensors, &state.layout, &[1])
        .unwrap_err();
    assert!(error.to_string().contains("shape"), "{error}");
    assert_eq!(contract_snapshot(&state), before);
    contract_pending(&state, &[1])
        .commit(&mut state.tensors, &state.layout, &[1])
        .unwrap();
    assert_eq!(state.lengths(), &[1]);
    assert_ne!(contract_snapshot(&state), before);
}

/// Hybrid transactions need one post-token recurrent snapshot for each accepted prefix.
#[test]
fn cache_contract_transactional_hybrid_chunk_rejection_is_atomic() {
    let device = Default::default();
    let model = contract_mtp_model(&device);
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 6).unwrap(), 1, 2, &device)
        .unwrap();
    let mark = state.mark().unwrap();
    let before = contract_snapshot(&state);
    let error = contract_pending(&state, &[2])
        .commit(&mut state.tensors, &state.layout, &[2])
        .unwrap_err();
    assert!(error.to_string().contains("at most one token per row"));
    assert_eq!(contract_snapshot(&state), before);
    contract_pending(&state, &[1])
        .commit(&mut state.tensors, &state.layout, &[1])
        .unwrap();
    assert_eq!(state.lengths(), &[0]);
    assert_eq!(state.tensors.working_lengths(), &[1]);
    state.rollback(mark).unwrap();
    assert_eq!(state.tensors.working_lengths(), &[0]);
    assert_eq!(cache_row_snapshot(&state, 0), before.rows[0]);
}

/// MTP prompt alignment validates both tensors and the already-committed target cursor.
#[test]
fn cache_contract_mtp_prefill_alignment_is_atomic() {
    let device = Default::default();
    let model = contract_mtp_model(&device);
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 6).unwrap(), 1, 2, &device)
        .unwrap();
    let prompt = Tensor::from_ints([[1, 2]], &device);
    let hidden = model
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
    let before = contract_snapshot(&state);
    for (tokens, hidden, context) in [
        (
            prompt.clone(),
            Tensor::zeros([1, 2, 7], &device),
            "qwen.mtp.prefill",
        ),
        (
            prompt.clone().narrow(1, 0, 1),
            hidden.clone().narrow(1, 0, 1),
            "target cache length",
        ),
    ] {
        let error = model
            .prefill_mtp_cache(tokens, hidden, &mut state)
            .unwrap_err();
        assert!(error.to_string().contains(context), "{error}");
        assert_eq!(contract_snapshot(&state), before);
    }
    model
        .prefill_mtp_cache(prompt.clone(), hidden.clone(), &mut state)
        .unwrap();
    let bound = contract_snapshot(&state);
    let error = model
        .prefill_mtp_cache(prompt, hidden, &mut state)
        .unwrap_err();
    assert!(error.to_string().contains("fresh auxiliary cache"));
    assert_eq!(contract_snapshot(&state), bound);
}

/// Invalid auxiliary rectangles and rotary coordinates do not advance either cache bank.
#[test]
fn cache_contract_mtp_cached_shapes_are_atomic() {
    let device = Default::default();
    let model = contract_mtp_model(&device);
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 6).unwrap(), 1, 2, &device)
        .unwrap();
    let before = contract_snapshot(&state);
    for case in 0..4 {
        let tokens = if case == 0 {
            Tensor::zeros([2, 1], &device)
        } else {
            Tensor::from_ints([[1]], &device)
        };
        let hidden = Tensor::zeros([1, 1, if case == 3 { 7 } else { 8 }], &device);
        let positions = match case {
            1 => Some(Tensor::zeros([1, 2], &device)),
            2 => Some(Tensor::from_ints([[-1]], &device)),
            _ => None,
        };
        let error = model
            .forward_mtp_cached(tokens, hidden, positions, 0, &mut state)
            .unwrap_err();
        assert!(matches!(error, Error::Shape { .. }), "case {case}: {error}");
        assert!(error.to_string().contains("qwen"), "case {case}: {error}");
        assert_eq!(contract_snapshot(&state), before);
    }
}

/// Auxiliary prefill and multi-token proposal calls are forbidden inside an open transaction.
#[test]
fn cache_contract_mtp_transaction_rejections_are_atomic() {
    let device = Default::default();
    let model = contract_mtp_model(&device);
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 6).unwrap(), 1, 2, &device)
        .unwrap();
    let mark = state.mark().unwrap();
    let before = contract_snapshot(&state);
    let error = model
        .prefill_mtp_cache(
            Tensor::from_ints([[1]], &device),
            Tensor::zeros([1, 1, 8], &device),
            &mut state,
        )
        .unwrap_err();
    assert!(error.to_string().contains("prefill Qwen MTP cache"));
    assert_eq!(contract_snapshot(&state), before);
    let error = model
        .forward_mtp_cached(
            Tensor::from_ints([[1, 2]], &device),
            Tensor::zeros([1, 2, 8], &device),
            None,
            0,
            &mut state,
        )
        .unwrap_err();
    assert!(error.to_string().contains("exactly one token per call"));
    assert_eq!(contract_snapshot(&state), before);
    state.rollback(mark).unwrap();
}

/// Reordering and duplicating a ragged beam moves both DeltaNet history and full-attention KV.
#[test]
fn cache_contract_hybrid_reorder_preserves_continuation_parity() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 6).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2], [3, 99]], &device),
                token_mask: Some(TokenMask::new(Tensor::from_bool(
                    [[true, true], [true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let first = cache_row_snapshot(&state, 0);
    let second = cache_row_snapshot(&state, 1);
    state.reorder(&[1, 0]).unwrap();
    assert_eq!(state.lengths(), &[1, 2]);
    assert_eq!(cache_row_snapshot(&state, 0), second);
    assert_eq!(cache_row_snapshot(&state, 1), first);
    state.reorder(&[0, 0]).unwrap();
    assert_eq!(state.lengths(), &[1, 1]);
    assert_eq!(cache_row_snapshot(&state, 0), second);
    assert_eq!(cache_row_snapshot(&state, 1), second);
    let actual = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[4], [5]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let expected = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[3, 4], [3, 5]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_close(actual.logits, expected.logits);
    assert_eq!(state.lengths(), &[2, 2]);
}

/// Full-KV rewind keeps the shifted auxiliary cursor one behind the target and permits replay.
#[test]
fn cache_contract_mtp_truncate_preserves_shifted_continuation_parity() {
    let device = Default::default();
    let mut config = tiny_config();
    config.num_hidden_layers = 1;
    config.layer_types = vec![Qwen3_5LayerType::FullAttention];
    config.mtp_num_hidden_layers = 1;
    let model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 6).unwrap(), 1, 2, &device)
        .unwrap();
    let prompt = Tensor::from_ints([[1, 2, 3]], &device);
    let hidden = model
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
        .prefill_mtp_cache(prompt, hidden.clone(), &mut state)
        .unwrap();
    let before = contract_snapshot(&state);
    let error = state.truncate(&[1, 1]).unwrap_err();
    assert!(error.to_string().contains("sequence"), "{error}");
    assert_eq!(contract_snapshot(&state), before);
    state.truncate(&[2]).unwrap();
    assert_eq!(state.lengths(), &[2]);
    assert_eq!(state.tensors.mtp_lengths(), Some([1].as_slice()));
    let mtp = model
        .forward_mtp_cached(
            Tensor::from_ints([[4]], &device),
            hidden.clone().narrow(1, 1, 1),
            None,
            0,
            &mut state,
        )
        .unwrap();
    let mtp_expected = model
        .forward_mtp(
            Tensor::from_ints([[2, 4]], &device),
            hidden.narrow(1, 0, 2),
            Tensor::from_ints([[0, 1]], &device),
            0,
        )
        .unwrap();
    assert_close(mtp.logits, mtp_expected.logits.narrow(1, 1, 1));
    let actual = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let expected = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_close(actual.logits, expected.logits);
    assert_eq!(state.lengths(), &[3]);
    assert_eq!(state.tensors.mtp_lengths(), Some([2].as_slice()));
}

/// Compares finite, equally shaped release-contract tensors at absolute F32 tolerance.
fn assert_release_close<const D: usize>(
    actual: Tensor<TestBackend, D>,
    expected: Tensor<TestBackend, D>,
) {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.into_data().to_vec::<f32>().unwrap();
    let expected = expected.into_data().to_vec::<f32>().unwrap();
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "element {index}"
        );
        assert!(
            (actual - expected).abs() <= 1e-4,
            "element {index}: {actual} != {expected}"
        );
    }
}

/// Explicit RoPE coordinates and valid-token cache age stay independent across ragged chunks.
#[test]
fn release_contract_ragged_explicit_positions_match_chunk_and_recurrent_decode() {
    let device = Default::default();
    let model = tiny_model(&device);
    let options = diagnostic_options();
    let config = CacheConfig::new(2, 4).unwrap();
    let mut chunked = model.new_decode_state(config, 2, &device).unwrap();
    let mut recurrent = model.new_decode_state(config, 2, &device).unwrap();
    let mut whole = model.new_decode_state(config, 2, &device).unwrap();
    let prompt = CausalLmInput {
        token_ids: Tensor::from_ints([[99, 1, 99, 2], [3, 99, 4, 99]], &device),
        token_mask: Some(TokenMask::new(Tensor::from_bool(
            [[false, true, false, true], [true, false, true, false]].into(),
            &device,
        ))),
        position_ids: Some(Tensor::from_ints([[-1, 5, -1, 8], [2, -1, 9, -1]], &device)),
    };
    model
        .prefill(prompt.clone(), &mut chunked, &options)
        .unwrap();
    model
        .prefill(prompt.clone(), &mut recurrent, &options)
        .unwrap();
    let tokens = Tensor::<TestBackend, 2, Int>::from_ints([[99, 5, 6], [7, 99, 8]], &device);
    let mask = Tensor::<TestBackend, 2, Bool>::from_bool(
        [[false, true, true], [true, false, true]].into(),
        &device,
    );
    let positions = Tensor::<TestBackend, 2, Int>::from_ints([[-1, 10, 12], [11, -1, 15]], &device);
    let expected = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::cat(vec![prompt.token_ids, tokens.clone()], 1),
                token_mask: Some(TokenMask::new(Tensor::cat(
                    vec![prompt.token_mask.unwrap().into_tensor(), mask.clone()],
                    1,
                ))),
                position_ids: Some(Tensor::cat(
                    vec![prompt.position_ids.unwrap(), positions.clone()],
                    1,
                )),
            },
            &mut whole,
            &options,
        )
        .unwrap();
    let actual = model
        .decode(
            DecodeInput {
                token_ids: tokens.clone(),
                token_mask: Some(TokenMask::new(mask.clone())),
                position_ids: Some(positions.clone()),
            },
            &mut chunked,
            &options,
        )
        .unwrap();
    let mut logits = Vec::new();
    let mut hidden = Vec::new();
    for column in 0..3 {
        let output = model
            .decode(
                DecodeInput {
                    token_ids: tokens.clone().narrow(1, column, 1),
                    token_mask: Some(TokenMask::new(mask.clone().narrow(1, column, 1))),
                    position_ids: Some(positions.clone().narrow(1, column, 1)),
                },
                &mut recurrent,
                &options,
            )
            .unwrap();
        logits.push(output.logits);
        hidden.push(output.final_hidden_state.unwrap());
    }
    assert_release_close(actual.logits.clone(), expected.logits.narrow(1, 4, 3));
    assert_release_close(actual.logits, Tensor::cat(logits, 1));
    let actual_hidden = actual.final_hidden_state.unwrap();
    assert_release_close(
        actual_hidden.clone(),
        expected.final_hidden_state.unwrap().narrow(1, 4, 3),
    );
    assert_release_close(actual_hidden, Tensor::cat(hidden, 1));
    assert_eq!(actual.hidden_states.len(), expected.hidden_states.len());
    for (actual, expected) in actual.hidden_states.into_iter().zip(expected.hidden_states) {
        assert_release_close(actual, expected.narrow(1, 4, 3));
    }
    for state in [&chunked, &recurrent, &whole] {
        assert_eq!(state.lengths(), &[4, 4]);
    }
}

/// Unsupported attention diagnostics return an error without consuming a valid continuation.
#[test]
fn release_contract_attention_capture_failure_preserves_hybrid_history() {
    let device = Default::default();
    let model = tiny_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
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
    let before = contract_snapshot(&state);
    let rejected = ForwardOptions {
        attentions: true,
        ..diagnostic_options()
    };
    for prefill in [true, false] {
        let token_ids = Tensor::from_ints([[3, 4]], &device);
        let result = if prefill {
            model.prefill(
                CausalLmInput {
                    token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &rejected,
            )
        } else {
            model.decode(
                DecodeInput {
                    token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &rejected,
            )
        };
        let error = result.unwrap_err();
        assert!(matches!(error, Error::BackendCapability(_)));
        assert!(error.to_string().contains("attention probability capture"));
        assert_eq!(contract_snapshot(&state), before);
    }
    let actual = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &diagnostic_options(),
        )
        .unwrap();
    let expected = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &diagnostic_options(),
        )
        .unwrap();
    assert_release_close(actual.logits, expected.logits.narrow(1, 2, 2));
    assert_eq!(state.lengths(), &[4]);
}

/// An advertised large model context cannot bypass the portable cache-index limit.
#[test]
fn release_contract_capacity_rejects_unrepresentable_indices_before_allocation() {
    let device = Default::default();
    let capacity = i32::MAX as usize + 1;
    let mut config = tiny_config();
    config.max_position_embeddings = capacity;
    let model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let error = model
        .new_decode_state(CacheConfig::new(1, capacity).unwrap(), 1, &device)
        .unwrap_err();
    assert!(
        matches!(error, Error::Configuration { ref field, .. } if field == "qwen3_5.decode_state.capacity")
    );
    assert!(
        error
            .to_string()
            .contains("portable signed position-index range")
    );
}
