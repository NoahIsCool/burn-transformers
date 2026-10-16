//! DeepSeek-V2 compressed-cache parity, lifecycle, ragged batches, and atomic failure tests.

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::{
    DecodeState, LogitsSelection, TokenMask,
    models::deepseek_v2::model::tests_support::{initialized_model, tiny_config},
};

/// CPU backend with F32 tensor values and I32 token indices for deterministic fixtures.
type TestBackend = Flex<f32, i32>;

/// Returns a deterministic tiny model and device.
fn tiny_model() -> (
    DeepseekV2ForCausalLm<TestBackend>,
    <TestBackend as BackendTypes>::Device,
) {
    let device = Default::default();
    (initialized_model(tiny_config(), &device), device)
}

/// Builds the standard input wrapper for one dense batch row.
fn input(tokens: Tensor<TestBackend, 2, Int>) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: tokens,
        token_mask: None,
        position_ids: None,
    }
}

/// Single-token decode uses absorbed MLA and matches full stateless logits.
#[test]
fn prefill_then_absorbed_decode_matches_full_forward() {
    let (model, device) = tiny_model();
    let full = model
        .forward(
            input(Tensor::from_ints([[1, 2, 3, 4]], &device)),
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            input(Tensor::from_ints([[1, 2, 3]], &device)),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(state.lengths(), &[3]);
    assert_eq!(
        state.last_execution_path(),
        Some(MlaExecutionPath::ReferenceUnabsorbed)
    );
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
        .unwrap()
        .logits;
    assert_eq!(state.lengths(), &[4]);
    assert_eq!(
        state.last_execution_path(),
        Some(MlaExecutionPath::AbsorbedDecode)
    );
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
}

/// Multi-token chunks stay on reference MLA and preserve causal parity.
#[test]
fn chunked_decode_matches_full_forward() {
    let (model, device) = tiny_model();
    let full = model
        .forward(
            input(Tensor::from_ints([[1, 2, 3, 4, 5]], &device)),
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap()
        .logits
        .slice([0..1, 2..5, 0..32]);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            input(Tensor::from_ints([[1, 2]], &device)),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cached = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3, 4, 5]], &device),
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
    assert_eq!(
        state.last_execution_path(),
        Some(MlaExecutionPath::ReferenceUnabsorbed)
    );
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
}

/// Physical storage contains only normalized latent plus one rotary key.
#[test]
fn compressed_cache_geometry_and_byte_count_are_exact() {
    let (model, device) = tiny_model();
    let state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    assert_eq!(state.scalars_per_layer_token(), 7);
    assert_eq!(state.layer_storage(0).unwrap().dims(), [2, 8, 7]);
    assert_eq!(state.bytes_per_token(), 7 * 2 * 4);
    assert_eq!(state.allocated_bytes(), 2 * 8 * 7 * 2 * 4);
    let conventional_kv_scalars = 2 * (5 + 4);
    assert!(state.scalars_per_layer_token() < conventional_kv_scalars);
}

/// Ragged prompts and inactive decode rows preserve independent lengths.
#[test]
fn ragged_batch_and_inactive_rows_are_supported() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3], [4, 5, 0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true, true], [true, true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(state.lengths(), &[3, 2]);
    let before_inactive = state.layer_storage(0).unwrap().into_data();
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[6], [0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true], [false]].into(),
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
    assert_eq!(state.lengths(), &[4, 2]);
    assert!(
        output
            .logits
            .slice([1..2, 0..1, 0..32])
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0)
    );
    let after = state.layer_storage(0).unwrap();
    let before = Tensor::<TestBackend, 3>::from_data(before_inactive, &device);
    assert_eq!(
        after.slice([1..2, 0..8, 0..7]).into_data(),
        before.slice([1..2, 0..8, 0..7]).into_data()
    );
}

/// Reorder duplicates rows, truncate rewinds logically, and reset zeros all bytes.
///
/// Full-history storage keeps a truncated row's later slots: they are
/// unreachable (reads mask slots at or beyond the length) and the next write
/// overwrites them. Truncating a row to zero clears it.
#[test]
fn lifecycle_operations_update_metadata_and_storage() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3], [4, 5, 0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true, true], [true, true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    state.reorder(&[1, 1]).unwrap();
    assert_eq!(state.lengths(), &[2, 2]);
    let storage = state.layer_storage(0).unwrap();
    assert_eq!(
        storage.clone().slice([0..1, 0..8, 0..7]).into_data(),
        storage.slice([1..2, 0..8, 0..7]).into_data()
    );
    let kept = state
        .layer_storage(0)
        .unwrap()
        .slice([0..1, 0..1, 0..7])
        .into_data();
    state.truncate(&[1, 0]).unwrap();
    assert_eq!(state.lengths(), &[1, 0]);
    let values = state.layer_storage(0).unwrap();
    assert_eq!(values.clone().slice([0..1, 0..1, 0..7]).into_data(), kept);
    assert!(
        values
            .slice([1..2, 0..8, 0..7])
            .into_data()
            .iter::<f32>()
            .all(|value| value == 0.0)
    );
    state.reset().unwrap();
    assert_eq!(state.lengths(), &[0, 0]);
    assert!(
        state
            .layer_storage(0)
            .unwrap()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0)
    );
}

/// Capacity, token, and diagnostic errors leave lengths and bytes unchanged.
#[test]
fn failures_are_byte_and_metadata_atomic() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 3).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            input(Tensor::from_ints([[1, 2]], &device)),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let before = state.layer_storage(0).unwrap().into_data();
    let before_path = state.last_execution_path();
    assert!(
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert!(
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[32]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert!(
        model
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
            .is_err()
    );
    assert_eq!(state.lengths(), &[2]);
    assert_eq!(state.last_execution_path(), before_path);
    assert_eq!(state.layer_storage(0).unwrap().into_data(), before);
}

/// Explicit positions are range-checked before any cache write.
#[test]
fn explicit_position_errors_are_atomic() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let error = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[32]], &device)),
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("position_ids"));
    assert_eq!(state.lengths(), &[0]);
}

/// Query-LoRA uses the same compressed cache and cached parity contract.
#[test]
fn query_lora_cached_path_matches_stateless() {
    let device = Default::default();
    let mut config = tiny_config();
    config.q_lora_rank = Some(4);
    let model = initialized_model(config, &device);
    let full = model
        .forward(
            input(Tensor::from_ints([[1, 2, 3]], &device)),
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            input(Tensor::from_ints([[1, 2]], &device)),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cached = model
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
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
}

/// Cache construction and lifecycle reject incompatible dimensions safely.
#[test]
fn cache_configuration_and_lifecycle_errors_are_contextual() {
    let (model, device) = tiny_model();
    assert!(
        model
            .new_decode_state(CacheConfig::new(2, 8).unwrap(), 1, &device)
            .is_err()
    );
    // A cache-level window is ignored like in every other family; MLA keeps
    // the full history regardless.
    assert!(
        model
            .new_decode_state(
                CacheConfig::new(1, 8)
                    .unwrap()
                    .with_sliding_window(4)
                    .unwrap(),
                1,
                &device,
            )
            .is_ok()
    );
    assert!(
        model
            .new_decode_state(CacheConfig::new(1, 33).unwrap(), 1, &device)
            .is_err()
    );
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    assert!(state.reorder(&[0]).is_err());
    assert!(state.reorder(&[0, 2]).is_err());
    assert!(state.truncate(&[0]).is_err());
    assert!(state.truncate(&[1, 0]).is_err());
    assert!(state.layer_storage(2).is_err());
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
        cases.push((input, ForwardOptions::default(), "deepseek_v2.input_ids"));
    }
    for shape in [[1, 1], [2, 2]] {
        let mut input = plain.clone();
        input.token_mask = Some(TokenMask::new(
            Tensor::<TestBackend, 2, Int>::ones(shape, device).bool(),
        ));
        cases.push((input, ForwardOptions::default(), "deepseek_v2.token_mask"));
        let mut input = plain.clone();
        input.position_ids = Some(Tensor::<TestBackend, 2, Int>::zeros(shape, device));
        cases.push((input, ForwardOptions::default(), "deepseek_v2.position_ids"));
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
fn sparse_contract_snapshot(state: &DeepseekV2DecodeState<TestBackend>) -> Vec<TensorData> {
    (0..tiny_config().num_hidden_layers)
        .map(|layer| state.layer_storage(layer).unwrap().to_data())
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
        "deepseek_v2.token_mask",
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
    let path_before = state.last_execution_path();
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
            assert_eq!(state.last_execution_path(), path_before);
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
    assert_sparse_contract_shape(masked_error, "deepseek_v2.token_mask");
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
    assert_eq!(state.last_execution_path(), path_before);
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
        assert_eq!(actual.routers.len(), 1);
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

/// A state created for different MLA geometry is rejected without mutation.
#[test]
fn decode_state_model_mismatch_is_atomic() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let mut other_config = tiny_config();
    other_config.kv_lora_rank = 6;
    let other = initialized_model(other_config, &device);
    let before = state.layer_storage(0).unwrap().into_data();
    assert!(
        other
            .prefill(
                input(Tensor::from_ints([[1]], &device)),
                &mut state,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert_eq!(state.lengths(), &[0]);
    assert_eq!(state.layer_storage(0).unwrap().into_data(), before);
}

/// Casts every floating parameter to F16, emulating a reduced-precision checkpoint.
struct HalfPrecision;

impl burn::module::ModuleMapper<TestBackend> for HalfPrecision {
    /// Rounds one floating parameter to F16 while preserving its shape and device.
    fn map_float<const D: usize>(
        &mut self,
        param: burn::module::Param<Tensor<TestBackend, D>>,
    ) -> burn::module::Param<Tensor<TestBackend, D>> {
        param.map(|tensor| tensor.cast(burn::tensor::DType::F16))
    }
}

/// An F16 model's inactive decode rows use the router's F32 dtype for placeholders.
///
/// Regression for DSV-1: the zero router logits of a finished row were created
/// in the embedding dtype, while active rows carry F32 router logits, so the
/// batch concatenation panicked for F16/BF16 models whenever router logits or
/// auxiliary losses were requested.
#[test]
fn reduced_precision_inactive_rows_keep_f32_router_logits() {
    use burn::module::Module;

    let (model, device) = tiny_model();
    let model = model.map(&mut HalfPrecision);
    for options in [
        ForwardOptions {
            router_logits: true,
            ..Default::default()
        },
        ForwardOptions {
            auxiliary_losses: true,
            ..Default::default()
        },
    ] {
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
            .unwrap();
        model
            .prefill(
                input(Tensor::from_ints([[1, 2], [3, 4]], &device)),
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let output = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[5], [0]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[true], [false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &options,
            )
            .expect("an inactive row must not break reduced-precision router assembly");
        assert_eq!(state.lengths(), &[3, 2]);
        assert!(!output.routers.is_empty());
        for router in &output.routers {
            if let Some(logits) = &router.logits {
                assert_eq!(logits.dtype(), burn::tensor::DType::F32);
                assert!(
                    logits
                        .clone()
                        .slice([1..2, 0..1, 0..logits.dims()[2]])
                        .into_data()
                        .iter::<f32>()
                        .all(|value| value == 0.0)
                );
            }
            assert_eq!(
                router.load_balancing_loss.is_some(),
                options.auxiliary_losses
            );
        }
    }
}

/// Context-dependent rotary scalings refuse to extend a non-empty cache past their limit.
#[test]
fn context_dependent_rotary_scaling_guards_cached_keys() {
    let dynamic = crate::position::RopeScaling::from(
        crate::position::DynamicNtkScaling::new(2.0, 16).unwrap(),
    );
    let limits = [dynamic.cached_context_limit()];
    assert_eq!(first_uncacheable_context(&[4], &[1], &[16], &limits), None);
    let (context, limit) = first_uncacheable_context(&[4], &[1], &[17], &limits).unwrap();
    let error = uncacheable_rotary_context(context, limit);
    assert!(
        matches!(&error, Error::BackendCapability(message) if message.contains("16-token limit")),
        "{error}"
    );
    // A fresh prefill and a finished row have no cached keys to invalidate.
    assert_eq!(first_uncacheable_context(&[0], &[1], &[17], &limits), None);
    assert_eq!(first_uncacheable_context(&[4], &[0], &[17], &limits), None);
    let static_limits = [crate::position::RopeScaling::None.cached_context_limit()];
    assert_eq!(
        first_uncacheable_context(&[4], &[1], &[usize::MAX], &static_limits),
        None
    );
}

/// One stateful call of a scenario: rectangular tokens and validity per row.
struct Step {
    /// Token IDs `[B][S]`.
    tokens: Vec<Vec<i64>>,
    /// Validity `[B][S]`, or `None` when every token is valid.
    mask: Option<Vec<Vec<bool>>>,
}

/// Builds a rectangular `[B, S]` input from host rows.
fn step_input(step: &Step) -> CausalLmInput<TestBackend> {
    let device = Default::default();
    let [batch, sequence] = [step.tokens.len(), step.tokens[0].len()];
    CausalLmInput {
        token_ids: Tensor::from_data(
            TensorData::new(step.tokens.concat(), [batch, sequence]),
            &device,
        ),
        token_mask: step.mask.as_ref().map(|mask| {
            TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_data(
                TensorData::new(mask.concat(), [batch, sequence]),
                &device,
            ))
        }),
        position_ids: None,
    }
}

/// Requests every logit, every hidden state, and router logits.
fn diagnostics() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        router_logits: true,
        ..Default::default()
    }
}

/// Runs a scenario in one batched state; returns every step's outputs.
fn run_batched(
    model: &DeepseekV2ForCausalLm<TestBackend>,
    steps: &[Step],
) -> (Vec<CausalLmOutput<TestBackend>>, Vec<usize>) {
    let batch = steps[0].tokens.len();
    let mut state = model
        .new_decode_state(
            CacheConfig::new(batch, 16).unwrap(),
            batch,
            &Default::default(),
        )
        .unwrap();
    let outputs = steps
        .iter()
        .enumerate()
        .map(|(index, step)| {
            let input = step_input(step);
            if index == 0 {
                model.prefill(input, &mut state, &diagnostics()).unwrap()
            } else {
                model
                    .decode(
                        DecodeInput {
                            token_ids: input.token_ids,
                            token_mask: input.token_mask,
                            position_ids: None,
                        },
                        &mut state,
                        &diagnostics(),
                    )
                    .unwrap()
            }
        })
        .collect();
    (outputs, state.lengths().to_vec())
}

/// One row's logits `[n_b * V]` and final hidden states `[n_b * H]`, or `None` when inactive.
type RowResult = Option<(Vec<f32>, Vec<f32>)>;

/// The previous algorithm's semantics: each row alone, compacted, in a batch-one state.
///
/// Returns, per step and row, the logits `[n_b, V]` and final hidden states
/// `[n_b, H]` of that row's valid tokens (empty when the row is inactive).
fn run_per_row(model: &DeepseekV2ForCausalLm<TestBackend>, steps: &[Step]) -> Vec<Vec<RowResult>> {
    let device = Default::default();
    let batch = steps[0].tokens.len();
    let mut results = vec![Vec::new(); steps.len()];
    for row in 0..batch {
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, &device)
            .unwrap();
        for (index, step) in steps.iter().enumerate() {
            let tokens = step.tokens[row]
                .iter()
                .enumerate()
                .filter(|&(column, _)| step.mask.as_ref().is_none_or(|mask| mask[row][column]))
                .map(|(_, &token)| token)
                .collect::<Vec<_>>();
            if tokens.is_empty() {
                results[index].push(None);
                continue;
            }
            let count = tokens.len();
            let input = input(Tensor::from_data(
                TensorData::new(tokens, [1, count]),
                &device,
            ));
            let output = if state.lengths()[0] == 0 {
                model.prefill(input, &mut state, &diagnostics())
            } else {
                model.decode(
                    DecodeInput {
                        token_ids: input.token_ids,
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &diagnostics(),
                )
            }
            .unwrap();
            results[index].push(Some((
                output.logits.into_data().to_vec::<f32>().unwrap(),
                output
                    .final_hidden_state
                    .unwrap()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            )));
        }
    }
    results
}

/// Asserts `|actual - expected| <= tolerance` elementwise with context.
fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (left - right).abs() <= tolerance,
            "{context}[{index}]: {left} vs {right}"
        );
    }
}

/// Ragged, padded, and inactive rows in one batched pass equal independent rows.
///
/// The scenario covers left padding, right padding, an interior hole, a
/// multi-token reference step with unequal per-row counts, absorbed one-token
/// decode with ragged cache lengths, and a finished row. Padding columns and
/// inactive rows must be exact zeros.
#[test]
fn batched_rows_match_independent_rows() {
    let model = initialized_reference();
    let steps = [
        Step {
            tokens: vec![vec![0, 3, 17, 5], vec![9, 30, 0, 0], vec![7, 0, 1, 22]],
            mask: Some(vec![
                vec![false, true, true, true],
                vec![true, true, false, false],
                vec![true, false, true, true],
            ]),
        },
        Step {
            tokens: vec![vec![4, 4, 2], vec![19, 0, 0], vec![0, 6, 0]],
            mask: Some(vec![
                vec![true, true, true],
                vec![true, false, false],
                vec![false, true, false],
            ]),
        },
        Step {
            tokens: vec![vec![11], vec![12], vec![13]],
            mask: None,
        },
        Step {
            tokens: vec![vec![14], vec![0], vec![15]],
            mask: Some(vec![vec![true], vec![false], vec![true]]),
        },
    ];
    let (batched, lengths) = run_batched(&model, &steps);
    assert_eq!(lengths, [8, 4, 6]);
    let oracle = run_per_row(&model, &steps);
    for (index, (step, output)) in steps.iter().zip(&batched).enumerate() {
        let [_, sequence, vocabulary] = output.logits.dims();
        let hidden_width = 8;
        let logits = output.logits.clone().into_data().to_vec::<f32>().unwrap();
        let hidden = output
            .final_hidden_state
            .clone()
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (row, expected) in oracle[index].iter().enumerate() {
            let valid = (0..sequence)
                .filter(|&column| step.mask.as_ref().is_none_or(|mask| mask[row][column]))
                .collect::<Vec<_>>();
            for column in 0..sequence {
                let logit = |values: &[f32], width: usize| {
                    values[(row * sequence + column) * width..(row * sequence + column + 1) * width]
                        .to_vec()
                };
                match (valid.iter().position(|&c| c == column), expected) {
                    (Some(k), Some((oracle_logits, oracle_hidden))) => {
                        let context = format!("step {index} row {row} column {column}");
                        assert_close(
                            &logit(&logits, vocabulary),
                            &oracle_logits[k * vocabulary..(k + 1) * vocabulary],
                            2e-5,
                            &context,
                        );
                        assert_close(
                            &logit(&hidden, hidden_width),
                            &oracle_hidden[k * hidden_width..(k + 1) * hidden_width],
                            2e-5,
                            &context,
                        );
                    }
                    _ => {
                        assert!(logit(&logits, vocabulary).iter().all(|&value| value == 0.0));
                        assert!(
                            logit(&hidden, hidden_width)
                                .iter()
                                .all(|&value| value == 0.0)
                        );
                    }
                }
            }
        }
        assert_eq!(output.hidden_states.len(), 4);
        assert_eq!(output.routers.len(), 1);
    }
}

/// Builds the HF reference V2 topology, whose O(1) logits make tolerances meaningful.
fn initialized_reference() -> DeepseekV2ForCausalLm<TestBackend> {
    crate::models::deepseek_v2::model::tests_support::reference_model(
        crate::models::deepseek_v2::model::tests_support::reference_v2_config(),
        &Default::default(),
    )
}

/// Batched cached logits equal the stateless forward of each complete row.
#[test]
fn batched_cached_logits_match_stateless_rows() {
    let model = initialized_reference();
    let device = Default::default();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 16).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[3, 17, 5], [0, 9, 30]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true, true], [false, true, true]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let decoded = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[2], [4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    assert_eq!(
        state.last_execution_path(),
        Some(MlaExecutionPath::AbsorbedDecode)
    );
    for (row, tokens) in [vec![3, 17, 5, 2], vec![9, 30, 4]].into_iter().enumerate() {
        let width = tokens.len();
        let expected = model
            .forward(
                input(Tensor::from_data(
                    TensorData::new(tokens, [1, width]),
                    &device,
                )),
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        decoded
            .clone()
            .slice([row..row + 1, 0..1, 0..32])
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(2e-5));
    }
}

/// Speculative proposals stage in place; commit keeps a prefix and rollback discards all.
#[test]
fn transactions_commit_prefixes_and_roll_back_proposals() {
    use crate::cache::TransactionalCache;

    let model = initialized_reference();
    let device = Default::default();
    let decode = |state: &mut DeepseekV2DecodeState<TestBackend>, tokens: Vec<i64>| {
        let width = tokens.len();
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_data(TensorData::new(tokens, [1, width]), &device),
                    token_mask: None,
                    position_ids: None,
                },
                state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
    };
    let fresh = || {
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill(
                input(Tensor::from_ints([[3, 17, 5]], &device)),
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        state
    };

    // Reference: commit tokens 9 then 30 without any transaction.
    let mut reference = fresh();
    decode(&mut reference, vec![9]);
    let expected = decode(&mut reference, vec![30]);

    // Speculate [9, 21, 22], accept one, then decode 30 normally.
    let mut state = fresh();
    let mark = state.mark().unwrap();
    decode(&mut state, vec![9, 21]);
    decode(&mut state, vec![22]);
    assert_eq!(state.lengths(), &[3]);
    assert_eq!(state.working_lengths(), &[6]);
    state.commit_prefix(mark, 1).unwrap();
    assert_eq!(state.lengths(), &[4]);
    // Token 9's cache entry came from a two-token step here and a one-token step
    // in the reference; F32 GEMMs of different row counts may sum in different
    // orders, so allow the batched-versus-stateless tolerance used above.
    decode(&mut state, vec![30])
        .into_data()
        .assert_approx_eq::<f32>(&expected.clone().into_data(), Tolerance::absolute(2e-5));

    // A rolled-back proposal leaves no trace.
    let mut state = fresh();
    let mark = state.mark().unwrap();
    decode(&mut state, vec![1, 2, 3]);
    state.rollback(mark).unwrap();
    assert_eq!(state.lengths(), &[3]);
    decode(&mut state, vec![9]);
    decode(&mut state, vec![30])
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));
}

/// A failure after some layers wrote ahead keeps lengths and readable history intact.
#[test]
fn mid_step_failures_are_logically_atomic() {
    use burn::module::Param;

    let mut model = initialized_reference();
    let device = Default::default();
    // A NaN embedding row passes layer 0, then makes layer 1's router fail.
    let mut table = model
        .model
        .embed_tokens
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    table[31 * 8..32 * 8].fill(f32::NAN);
    model.model.embed_tokens.weight =
        Param::from_tensor(Tensor::from_data(TensorData::new(table, [32, 8]), &device));
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            input(Tensor::from_ints([[3, 17, 5]], &device)),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let path = state.last_execution_path();
    let error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[31]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, .. } if tensor == "moe.router_scores"),
        "{error}"
    );
    assert_eq!(state.lengths(), &[3]);
    assert_eq!(state.last_execution_path(), path);
    let recovered = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[9]], &device),
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
            input(Tensor::from_ints([[3, 17, 5, 9]], &device)),
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    recovered
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(2e-5));
}

/// A step in which every row is finished returns zeros and changes nothing.
#[test]
fn all_inactive_decode_is_a_no_op() {
    let model = initialized_reference();
    let device = Default::default();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 16).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            input(Tensor::from_ints([[3, 17], [5, 9]], &device)),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let before = state.layer_storage(1).unwrap().into_data();
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[0], [0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false], [false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                hidden_states: true,
                router_logits: true,
                auxiliary_losses: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(state.lengths(), &[2, 2]);
    assert_eq!(state.layer_storage(1).unwrap().into_data(), before);
    assert!(
        output
            .logits
            .into_data()
            .iter::<f32>()
            .all(|value| value == 0.0)
    );
    assert_eq!(output.hidden_states.len(), 4);
    assert_eq!(output.routers.len(), 1);
    assert_eq!(
        output.routers[0].logits.as_ref().unwrap().dtype(),
        burn::tensor::DType::F32
    );
}

/// Masked sentinels of a finished or padded row never reach the embedding or RoPE.
///
/// Masked columns are not validated, so a finished row may carry an
/// out-of-vocabulary pad ID, `-1`, `i32::MIN`, or `i32::MAX`, with an equally
/// invalid explicit position. Regression (review DSM-R1): batched packing
/// gathered a finished row's column 0 into the block and Flex panicked in the
/// embedding gather (an out-of-bounds read on GPU backends). Every sentinel
/// must now produce bit-identical logits to padding with the valid token and
/// position 0, for one-token (absorbed) and two-token (reference) steps, with
/// implicit and explicit positions, and for finished and left-padded rows.
#[test]
fn masked_sentinel_tokens_and_positions_are_inert() {
    let model = initialized_reference();
    let device = Default::default();
    /// Builds row 1's tokens, mask, and positions around a sentinel token `pad` and position `at`.
    type MaskedRow = fn(i32, i32) -> (Vec<i32>, Vec<bool>, Vec<i32>);
    let cases: [(usize, MaskedRow); 3] = [
        (1, |pad, at| (vec![pad], vec![false], vec![at])),
        (2, |pad, at| {
            (vec![pad, pad], vec![false, false], vec![at, at])
        }),
        (2, |pad, at| (vec![pad, 8], vec![false, true], vec![at, 3])),
    ];
    for (width, row) in cases {
        for explicit in [false, true] {
            let run = |pad: i32, at: i32| {
                let (tokens, mask, positions) = row(pad, at);
                let active_tokens = [7, 9][..width].to_vec();
                let active_positions = [3, 4][..width].to_vec();
                let mut state = model
                    .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
                    .unwrap();
                model
                    .prefill(
                        input(Tensor::from_ints([[1, 2, 3], [4, 5, 6]], &device)),
                        &mut state,
                        &ForwardOptions::default(),
                    )
                    .unwrap();
                let output = model
                    .decode(
                        DecodeInput {
                            token_ids: Tensor::from_data(
                                TensorData::new([active_tokens, tokens].concat(), [2, width]),
                                &device,
                            ),
                            token_mask: Some(TokenMask::new(
                                Tensor::<TestBackend, 2, Bool>::from_data(
                                    TensorData::new([vec![true; width], mask].concat(), [2, width]),
                                    &device,
                                ),
                            )),
                            position_ids: explicit.then(|| {
                                Tensor::from_data(
                                    TensorData::new(
                                        [active_positions, positions].concat(),
                                        [2, width],
                                    ),
                                    &device,
                                )
                            }),
                        },
                        &mut state,
                        &ForwardOptions {
                            logits: LogitsSelection::All,
                            router_logits: true,
                            ..Default::default()
                        },
                    )
                    .unwrap_or_else(|error| panic!("pad {pad} at {at}: {error}"));
                (output.logits.into_data(), state.lengths().to_vec())
            };
            let baseline = run(0, 0);
            for (pad, at) in [
                (32, 32),
                (-1, -1),
                (i32::MAX, i32::MAX),
                (i32::MIN, 1 << 20),
            ] {
                assert_eq!(
                    run(pad, at),
                    baseline,
                    "width {width}, explicit {explicit}, pad {pad}, position {at}"
                );
            }
        }
    }
}

/// Measures batched prefill/decode against one batch-one call per row (the previous algorithm).
///
/// Run with `cargo test --release --lib batched_decode_benchmark -- --ignored --nocapture`.
/// Each variant runs five times in alternation; the minimum and median wall
/// times of the prefill and of 16 one-token decode steps are reported.
#[test]
#[ignore = "release-mode timing benchmark"]
fn batched_decode_benchmark() {
    use std::time::Instant;

    let device = Default::default();
    let mut config = crate::models::deepseek_v2::model::tests_support::reference_v2_config();
    config.hidden_size = 256;
    config.intermediate_size = 512;
    config.moe_intermediate_size = 128;
    config.num_attention_heads = 8;
    config.kv_lora_rank = 64;
    config.qk_nope_head_dim = 32;
    config.qk_rope_head_dim = 16;
    config.v_head_dim = 32;
    config.num_hidden_layers = 4;
    config.max_position_embeddings = 1024;
    let model = initialized_model(config, &device);
    let (batch, prompt, steps) = (8, 128, 16);
    let prompts = (0..batch * prompt)
        .map(|index| ((index * 7) % 31) as i64)
        .collect::<Vec<_>>();
    // Runs `rows` rows starting at `first` in one state; returns (prefill, decode) seconds.
    let run = |first: usize, rows: usize| {
        let mut state = model
            .new_decode_state(CacheConfig::new(rows, 512).unwrap(), rows, &device)
            .unwrap();
        let tokens = prompts[first * prompt..(first + rows) * prompt].to_vec();
        let started = Instant::now();
        let _ = model
            .prefill(
                input(Tensor::from_data(
                    TensorData::new(tokens, [rows, prompt]),
                    &device,
                )),
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
            .into_data();
        let prefill = started.elapsed().as_secs_f64();
        let started = Instant::now();
        for step in 0..steps {
            let _ = model
                .decode(
                    DecodeInput {
                        token_ids: Tensor::from_data(
                            TensorData::new(vec![(step % 31) as i64; rows], [rows, 1]),
                            &device,
                        ),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap()
                .logits
                .into_data();
        }
        (prefill, started.elapsed().as_secs_f64())
    };
    let per_row = || {
        (0..batch)
            .map(|row| run(row, 1))
            .fold((0.0, 0.0), |sum, time| (sum.0 + time.0, sum.1 + time.1))
    };
    let _ = run(0, batch);
    let (mut batched, mut separate) = (Vec::new(), Vec::new());
    for _ in 0..5 {
        batched.push(run(0, batch));
        separate.push(per_row());
    }
    let summary = |samples: &[(f64, f64)], pick: fn(&(f64, f64)) -> f64| {
        let mut values = samples.iter().map(pick).collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        (values[0], values[values.len() / 2])
    };
    for (label, pick) in [
        (
            "prefill",
            (|time: &(f64, f64)| time.0) as fn(&(f64, f64)) -> f64,
        ),
        ("decode x16", |time: &(f64, f64)| time.1),
    ] {
        let (batched_min, batched_median) = summary(&batched, pick);
        let (separate_min, separate_median) = summary(&separate, pick);
        println!(
            "{label}: per-row min {separate_min:.4} s median {separate_median:.4} s; \
             batched min {batched_min:.4} s median {batched_median:.4} s; \
             speedup min {:.2}x median {:.2}x",
            separate_min / batched_min,
            separate_median / batched_median
        );
    }
}

/// Allocation rejects unaddressable capacities, overflowing widths, and oversized layers.
///
/// Every check runs before any cache tensor is allocated, so even a
/// `2^24`-row request fails without reserving memory.
#[test]
fn decode_state_preflight_rejects_unaddressable_geometry() {
    let (model, device) = tiny_model();
    let state = model
        .new_decode_state(CacheConfig::new(2, 32).unwrap(), 2, &device)
        .expect("capacity equal to the position table");
    assert_eq!(state.batch_size(), 2);
    assert_eq!(state.capacity(), 32);
    assert_eq!(state.device(), &device);
    assert_eq!(state.tensor_state().batch_size(), 2);
    assert_eq!(state.working_lengths(), &[0, 0]);

    let rows = 1 << 24;
    let error = model
        .new_decode_state(CacheConfig::new(rows, 32).unwrap(), rows, &device)
        .expect_err("one layer exceeds the signed tensor-index range");
    assert!(
        error.to_string().contains("portable signed tensor-index"),
        "{error}"
    );

    let mut wide = initialized_model::<TestBackend>(tiny_config(), &device);
    wide.model.config.max_position_embeddings = i32::MAX as usize + 1;
    let error = wide
        .new_decode_state(
            CacheConfig::new(1, i32::MAX as usize + 1).unwrap(),
            1,
            &device,
        )
        .expect_err("capacity beyond the signed position range");
    assert!(
        error.to_string().contains("portable signed position-index"),
        "{error}"
    );

    let mut overflowing = initialized_model::<TestBackend>(tiny_config(), &device);
    overflowing.model.config.kv_lora_rank = usize::MAX;
    let error = overflowing
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .expect_err("latent plus rotary width overflows");
    assert!(
        error.to_string().contains("cache width overflows"),
        "{error}"
    );
}

/// A finished-only step without requested traces returns logits and nothing else.
#[test]
fn all_inactive_decode_without_diagnostics_returns_only_logits() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            input(Tensor::from_ints([[1, 2]], &device)),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(state.lengths(), &[2]);
    assert!(output.hidden_states.is_empty());
    assert!(output.routers.is_empty());
    assert!(output.final_hidden_state.is_none());
    assert_eq!(output.logits.dims(), [1, 1, 32]);
    assert!(
        output
            .logits
            .into_data()
            .iter::<f32>()
            .all(|value| value == 0.0)
    );
}

/// Tiny model whose LongRoPE switches to long factors past an original context of two.
fn long_rope_model(
    device: &<TestBackend as BackendTypes>::Device,
) -> DeepseekV2ForCausalLm<TestBackend> {
    let mut config = tiny_config();
    config.rope_scaling = Some(crate::models::config::RopeParameters {
        rope_type: crate::models::config::RopeType::LongRope,
        factor: Some(2.0),
        original_max_position_embeddings: Some(2),
        short_factor: Some(vec![1.0]),
        long_factor: Some(vec![3.0]),
        ..Default::default()
    });
    initialized_model(config, device)
}

/// Context-dependent rotary tables are built per row, matching batch-one prefills.
///
/// Under LongRoPE row 0 (three tokens) runs past the original two-token
/// context and rotates with the long factors, while row 1 (two tokens) keeps
/// the short factors. A fresh prefill has no cached keys to invalidate, so
/// the ragged batch is accepted and must reproduce each row's own prefill.
/// Extending row 1 past the limit afterwards would re-rotate its cached keys
/// and is rejected without changing any length.
#[test]
fn context_dependent_rotary_tables_are_built_per_row() {
    let device = Default::default();
    let model = long_rope_model(&device);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    let batched = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[3, 5, 7], [9, 11, 0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true, true], [true, true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap()
        .logits;
    assert_eq!(state.lengths(), &[3, 2]);
    for (row, tokens) in [vec![3, 5, 7], vec![9, 11]].into_iter().enumerate() {
        let count = tokens.len();
        let mut single = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        let expected = model
            .prefill(
                input(Tensor::from_data(
                    TensorData::new(tokens, [1, count]),
                    &device,
                )),
                &mut single,
                &options,
            )
            .unwrap()
            .logits;
        batched
            .clone()
            .slice([row..row + 1, 0..count])
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(1e-5, 1e-6));
    }

    let error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[0], [4]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false], [true]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("row 1 would re-rotate its cached keys");
    assert!(matches!(error, Error::BackendCapability(_)));
    assert!(error.to_string().contains("2-token limit"), "{error}");
    assert_eq!(state.lengths(), &[3, 2]);
}

/// A state allocated for another rotary width or activation dtype is rejected atomically.
#[test]
fn decode_state_rejects_foreign_rotary_width_and_dtype() {
    use burn::module::Module;

    let (model, device) = tiny_model();
    let mut wide_rope = tiny_config();
    wide_rope.qk_rope_head_dim = 4;
    let wide_rope = initialized_model::<TestBackend>(wide_rope, &device);
    let half = model.clone().map(&mut HalfPrecision);
    for other in [&wide_rope, &half] {
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        let error = other
            .prefill(
                input(Tensor::from_ints([[1]], &device)),
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("the state belongs to a different geometry or dtype");
        assert!(
            matches!(
                &error,
                Error::Cache(message)
                    if message == "DeepSeek decode state is incompatible with the model"
            ),
            "{error:?}"
        );
        assert_eq!(state.lengths(), &[0]);
    }
}

/// A finished-only step that requests only the auxiliary loss still returns router diagnostics.
///
/// The sparse layer reports no logits and a `[1]` load-balancing loss of
/// exactly zero: the fully masked row contributes no selections or
/// probability mass, and the clamped token and active-row denominators turn
/// `0 / 0` into `0 / 1`, so finished rows can neither add loss nor produce
/// NaN. The committed length is unchanged.
#[test]
fn all_inactive_decode_reports_auxiliary_losses_without_router_logits() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            input(Tensor::from_ints([[1, 2]], &device)),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                auxiliary_losses: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(output.routers.len(), 1);
    assert!(output.routers[0].logits.is_none());
    let loss = output.routers[0]
        .load_balancing_loss
        .clone()
        .expect("the auxiliary loss was requested");
    assert_eq!(loss.dims(), [1]);
    assert_eq!(loss.into_data().to_vec::<f32>().unwrap(), [0.0]);
    assert_eq!(state.lengths(), &[2]);
}
