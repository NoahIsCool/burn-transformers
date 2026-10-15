//! DeepSeek-V2 compressed-cache parity, lifecycle, ragged batches, and atomic failure tests.

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::{
    DecodeState, LogitsSelection,
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

/// Reorder permits duplication, truncate clears suffixes, and reset zeros all bytes.
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
    state.truncate(&[1, 0]).unwrap();
    assert_eq!(state.lengths(), &[1, 0]);
    let values = state.layer_storage(0).unwrap().into_data();
    let values = values.as_slice::<f32>().unwrap();
    assert!(values[7..].iter().all(|&value| value == 0.0));
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
            .is_err()
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
