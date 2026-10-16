//! Nemotron-H cached parity, ragged diagnostics, and atomic cache-lifecycle tests.

use burn::{
    backend::Flex,
    tensor::{Bool, Tensor, TensorData, Tolerance},
};

use crate::{
    DecodeState,
    architecture::TokenMask,
    models::nemotron_h::model::tests::{
        assert_relative_close, relative_difference, tiny_config, with_dt_bias,
    },
};

use super::*;

/// CPU Flex backend with F32 activations and I32 indices for cache comparisons.
type TestBackend = Flex<f32, i32>;

/// Loads deterministic nonzero parameters through the public checkpoint sink.
fn contract_model() -> NemotronHForCausalLm<TestBackend> {
    contract_model_with(false)
}

/// [`contract_model`], optionally with the MTP stage that speculative states require.
fn contract_model_with(mtp: bool) -> NemotronHForCausalLm<TestBackend> {
    use crate::{checkpoint::TensorSink, models::nemotron_h::NemotronHCheckpointOptions};
    let device = Default::default();
    let mut config = tiny_config(mtp);
    // One eligible group makes top-k independently reproducible from raw logits.
    config.n_group = 1;
    config.topk_group = 1;
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    let mut model = NemotronHForCausalLm::init(config, &device).unwrap();
    for (tensor_index, expected) in contract.expected_tensors().iter().enumerate() {
        let shape = expected
            .shape
            .iter()
            .map(|&n| n as usize)
            .collect::<Vec<_>>();
        let values = (0..shape.iter().product())
            .map(|element| {
                if expected.name.ends_with("correction_bias") {
                    0.0
                } else if expected.name.ends_with("norm.weight") {
                    1.0
                } else {
                    (((element + tensor_index * 13) as f32 + 0.5) * 0.37).sin() * 0.05
                }
            })
            .collect::<Vec<_>>();
        model
            .write_tensor(&expected.name, TensorData::new(values, shape))
            .unwrap();
    }
    model
}

/// Captures public KV, recurrent and convolution buffers without mutable access.
fn contract_cache_snapshot(state: &NemotronHDecodeState<TestBackend>) -> Vec<TensorData> {
    let bank = state.tensor_state().main();
    let mut data = Vec::new();
    for group in 0..bank.native_kv_count() {
        let cache = bank.native_kv(group).unwrap();
        for layer in 0..cache.spec().layers() {
            data.push(cache.keys(layer).unwrap().to_data());
            data.push(cache.values(layer).unwrap().to_data());
        }
    }
    for group in 0..bank.recurrent_count() {
        let cache = bank.recurrent(group).unwrap();
        for layer in 0..cache.spec().layers() {
            data.push(cache.matrix(layer).unwrap().to_data());
            data.push(cache.convolution(layer).unwrap().to_data());
        }
    }
    data
}

/// Requires finite, equally shaped F32 tensors before comparing absolute errors.
fn contract_close(actual: Tensor<TestBackend, 3>, expected: Tensor<TestBackend, 3>) {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.to_data().to_vec::<f32>().unwrap();
    let expected = expected.to_data().to_vec::<f32>().unwrap();
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
    }
}

/// Public validation and late output-selection errors cannot publish hybrid history.
#[test]
fn experimental_contract_public_rejections_preserve_nemotron_history() {
    let model = contract_model();
    let device = Default::default();
    let error = model
        .new_decode_state(CacheConfig::new(1, 33).unwrap(), 1, &device)
        .unwrap_err();
    assert!(matches!(error, Error::Configuration { field, reason }
        if field == "nemotron_h.decode_state.capacity"
        && reason == "33 exceeds max_position_embeddings 32"));
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
    let baseline = contract_cache_snapshot(&state);
    assert!(baseline.iter().all(|tensor| {
        tensor
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .any(|value| value != 0.0)
    }));
    for case in 0..7 {
        let mut input = CausalLmInput {
            token_ids: Tensor::from_ints([[3]], &device),
            token_mask: None,
            position_ids: None,
        };
        let mut options = ForwardOptions::default();
        let (tensor, expected, actual) = match case {
            0 => {
                input.token_mask = Some(TokenMask::new(Tensor::from_bool(
                    [[true, true]].into(),
                    &device,
                )));
                ("nemotron_h.token_mask", "[1, 1]", "[1, 2]")
            }
            1 => {
                input.position_ids = Some(Tensor::from_ints([[2, 3]], &device));
                ("nemotron_h.position_ids", "[1, 1]", "[1, 2]")
            }
            2 | 3 => {
                input.position_ids = Some(Tensor::from_ints(
                    [[if case == 2 { -1 } else { 32 }]],
                    &device,
                ));
                (
                    "nemotron_h.position_ids",
                    "position values in 0..32",
                    if case == 2 {
                        "observed range -1..=-1"
                    } else {
                        "observed range 32..=32"
                    },
                )
            }
            4 | 5 => {
                options.logits =
                    crate::LogitsSelection::Positions(if case == 4 { vec![] } else { vec![1] });
                (
                    "forward_options.logits.positions",
                    "a non-empty list of indices below 1",
                    if case == 4 { "[]" } else { "[1]" },
                )
            }
            _ => {
                input.token_mask =
                    Some(TokenMask::new(Tensor::from_bool([[false]].into(), &device)));
                (
                    "nemotron_h.token_mask",
                    "at least one valid token in every prefill row",
                    "a fully masked prompt row",
                )
            }
        };
        let error = if case == 6 {
            model.prefill(input, &mut state, &options)
        } else {
            model.decode(
                DecodeInput {
                    token_ids: input.token_ids,
                    token_mask: input.token_mask,
                    position_ids: input.position_ids,
                },
                &mut state,
                &options,
            )
        }
        .unwrap_err();
        assert!(
            matches!(error, Error::Shape { tensor: found, expected: expected_found, actual: actual_found }
            if found == tensor && expected_found == expected && actual_found == actual)
        );
        assert_eq!(state.lengths(), &[2]);
        assert_eq!(contract_cache_snapshot(&state), baseline);
    }
    let error = model
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
    assert!(matches!(error, Error::BackendCapability(reason)
        if reason == "Nemotron-H cached attention capture is unavailable"));
    assert_eq!(state.lengths(), &[2]);
    assert_eq!(contract_cache_snapshot(&state), baseline);
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
        .unwrap();
    let expected = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    contract_close(actual.logits, expected.logits);
    assert_eq!(state.lengths(), &[3]);
}

/// Host oracle of Megatron-LM's normalized-sigmoid top-two switch loss over four experts.
///
/// Scores are `sigmoid(z) / sum sigmoid(z)` per token; assignments are the
/// top two normalized scores (smallest ID on ties); the loss is
/// `E / (K T^2) * sum_e (sum_t s[t,e]) (sum_t m[t,e])`.
fn contract_sigmoid_load_loss(logits: &[f32]) -> f64 {
    assert!(!logits.is_empty() && logits.len().is_multiple_of(4));
    let tokens = (logits.len() / 4) as f64;
    let mut scores = [0.0_f64; 4];
    let mut assignments = [0.0_f64; 4];
    for row in logits.chunks_exact(4) {
        let sigmoid = row
            .iter()
            .map(|&logit| {
                assert!(logit.is_finite());
                1.0 / (1.0 + (-f64::from(logit)).exp())
            })
            .collect::<Vec<_>>();
        let total = sigmoid.iter().sum::<f64>() + 1e-20;
        let mut indices = [0, 1, 2, 3];
        indices.sort_by(|&left, &right| sigmoid[right].total_cmp(&sigmoid[left]));
        for &expert in &indices[..2] {
            assignments[expert] += 1.0;
        }
        for (expert, value) in sigmoid.into_iter().enumerate() {
            scores[expert] += value / total;
        }
    }
    scores
        .into_iter()
        .zip(assignments)
        .map(|(score, assignment)| score * assignment)
        .sum::<f64>()
        * 4.0
        / (2.0 * tokens * tokens)
}

/// Cached diagnostics describe new valid tokens, excluding history and inactive rows.
#[test]
fn experimental_contract_nemotron_ragged_router_and_hidden_diagnostics() {
    let model = contract_model();
    let device = Default::default();
    let reference_options = ForwardOptions {
        logits: crate::LogitsSelection::All,
        hidden_states: true,
        router_logits: true,
        ..Default::default()
    };
    let reference = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4, 5]], &device),
                token_mask: None,
                position_ids: None,
            },
            &reference_options,
        )
        .unwrap();
    let reference_router = reference.routers[0]
        .logits
        .as_ref()
        .unwrap()
        .clone()
        .narrow(1, 2, 3);
    let expected_loss =
        contract_sigmoid_load_loss(&reference_router.to_data().to_vec::<f32>().unwrap());
    let historical_loss = contract_sigmoid_load_loss(
        &reference.routers[0]
            .logits
            .as_ref()
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap(),
    );
    assert!(
        (expected_loss - historical_loss).abs() > 1e-4,
        "fixture must distinguish new-token loss {expected_loss} from full-history loss {historical_loss}"
    );
    for (router_logits, auxiliary_losses) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let options = ForwardOptions {
            router_logits,
            auxiliary_losses,
            ..reference_options.clone()
        };
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2], [5, 6]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let actual = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints(
                        [[31, 3, 31, 4, 5], [31, 31, 31, 31, 31]],
                        &device,
                    ),
                    token_mask: Some(TokenMask::new(Tensor::from_bool(
                        [
                            [false, true, false, true, true],
                            [false, false, false, false, false],
                        ]
                        .into(),
                        &device,
                    ))),
                    position_ids: Some(Tensor::from_ints(
                        [[-1, 9, 99, 12, 13], [-1, 99, 100, 101, 102]],
                        &device,
                    )),
                },
                &mut state,
                &options,
            )
            .unwrap();
        assert_eq!(state.lengths(), &[5, 2]);
        assert_eq!(actual.hidden_states.len(), 6);
        for (hidden, expected) in actual.hidden_states.iter().zip(&reference.hidden_states) {
            assert_eq!(hidden.dims(), [2, 5, 8]);
            for (column, reference_column) in [(1, 2), (3, 3), (4, 4)] {
                contract_close(
                    hidden.clone().slice([0..1, column..column + 1, 0..8]),
                    expected.clone().narrow(1, reference_column, 1),
                );
            }
            for (index, value) in hidden
                .to_data()
                .to_vec::<f32>()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                if !(8..16).contains(&index) && !(24..40).contains(&index) {
                    assert_eq!(value, 0.0);
                }
            }
        }
        for (column, reference_column) in [(1, 2), (3, 3), (4, 4)] {
            contract_close(
                actual
                    .logits
                    .clone()
                    .slice([0..1, column..column + 1, 0..32]),
                reference.logits.clone().narrow(1, reference_column, 1),
            );
        }
        assert_eq!(
            actual.routers.len(),
            usize::from(router_logits || auxiliary_losses)
        );
        if let Some(router) = actual.routers.first() {
            assert_eq!(router.logits.is_some(), router_logits);
            assert_eq!(router.load_balancing_loss.is_some(), auxiliary_losses);
            assert!(router.z_loss.is_none());
            if let Some(logits) = &router.logits {
                assert_eq!(logits.dims(), [2, 5, 4]);
                for (column, reference_column) in [(1, 0), (3, 1), (4, 2)] {
                    contract_close(
                        logits.clone().slice([0..1, column..column + 1, 0..4]),
                        reference_router.clone().narrow(1, reference_column, 1),
                    );
                }
                for (index, value) in logits
                    .to_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .enumerate()
                {
                    if !(4..8).contains(&index) && !(12..20).contains(&index) {
                        assert_eq!(value, 0.0);
                    }
                }
            }
            if let Some(loss) = &router.load_balancing_loss {
                let loss = f64::from(loss.clone().into_scalar());
                assert!(loss.is_finite());
                assert!(
                    (loss - expected_loss).abs() <= 1e-4,
                    "{loss} != {expected_loss}"
                );
            }
        }
        let baseline = contract_cache_snapshot(&state);
        let empty = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[31], [31]], &device),
                    token_mask: Some(TokenMask::new(Tensor::from_bool(
                        [[false], [false]].into(),
                        &device,
                    ))),
                    position_ids: Some(Tensor::from_ints([[-1], [99]], &device)),
                },
                &mut state,
                &options,
            )
            .unwrap();
        assert_eq!(state.lengths(), &[5, 2]);
        assert_eq!(contract_cache_snapshot(&state), baseline);
        assert!(
            empty
                .logits
                .to_data()
                .to_vec::<f32>()
                .unwrap()
                .into_iter()
                .all(|value| value == 0.0)
        );
        if let Some(loss) = empty
            .routers
            .first()
            .and_then(|router| router.load_balancing_loss.as_ref())
        {
            assert_eq!(loss.clone().into_scalar(), 0.0);
        }
    }
}

/// Cached chunk/decode execution matches the full stateless continuation.
#[test]
fn cached_prefill_decode_matches_stateless() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
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
    assert_eq!(state.attention_layer_count(), 1);
    assert_eq!(state.mamba_layer_count(), 1);
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
    assert_eq!(state.lengths(), &[4]);
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(3e-5));
}

/// Sliding attention keeps full storage while stateless and cached math agree.
#[test]
fn sliding_window_cached_path_matches_stateless() {
    let device = Default::default();
    let mut config = tiny_config(false);
    config.sliding_window = Some(2);
    let model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4, 5]], &device),
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
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
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
                token_ids: Tensor::from_ints([[4, 5]], &device),
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
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(3e-5));
}

/// Left padding does not advance recurrent or KV state for semantic tokens.
#[test]
fn left_padded_prefill_matches_stateless_last_token() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let mask = || {
        TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[false, false, true, true, true]].into(),
            &device,
        ))
    };
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 0, 1, 2, 3]], &device),
                token_mask: Some(mask()),
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let cached = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 0, 1, 2, 3]], &device),
                token_mask: Some(mask()),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    assert_eq!(state.lengths(), &[3]);
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(3e-5));
}

/// Readable keys of attention layer `layer`: every row's committed prefix.
///
/// Slots beyond a row's length are unreachable and may hold abandoned
/// write-ahead keys, so atomicity is stated over this prefix.
fn readable_keys(state: &NemotronHDecodeState<TestBackend>, layer: usize) -> Vec<TensorData> {
    let keys = state
        .tensors
        .main()
        .native_kv(KV_GROUP)
        .unwrap()
        .keys(layer)
        .unwrap()
        .clone();
    state
        .lengths()
        .iter()
        .enumerate()
        .map(|(row, &length)| {
            keys.clone()
                .narrow(0, row, 1)
                .narrow(2, 0, length)
                .into_data()
        })
        .collect()
}

/// A failed decode leaves logical, KV, recurrent, and convolution state intact.
#[test]
fn failed_decode_is_atomic_across_cache_kinds() {
    let device = Default::default();
    let mut model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
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
    let kv = readable_keys(&state, 0);
    let recurrent = state
        .tensors
        .main()
        .recurrent(RECURRENT_GROUP)
        .unwrap()
        .matrix(0)
        .unwrap()
        .clone()
        .into_data();
    let convolution = state
        .tensors
        .main()
        .recurrent(RECURRENT_GROUP)
        .unwrap()
        .convolution(0)
        .unwrap()
        .clone()
        .into_data();
    // Fail in the final MoE block, after both earlier Mamba and attention
    // blocks have produced operation-local candidate cache updates.
    model.model.layers[3].moe = None;
    assert!(
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert_eq!(state.lengths(), &[2]);
    assert_eq!(readable_keys(&state, 0), kv);
    assert_eq!(
        state
            .tensors
            .main()
            .recurrent(RECURRENT_GROUP)
            .unwrap()
            .matrix(0)
            .unwrap()
            .clone()
            .into_data(),
        recurrent
    );
    assert_eq!(
        state
            .tensors
            .main()
            .recurrent(RECURRENT_GROUP)
            .unwrap()
            .convolution(0)
            .unwrap()
            .clone()
            .into_data(),
        convolution
    );
}

/// Inactive decode rows are exact no-ops while active peers advance.
#[test]
fn ragged_decode_and_lifecycle_operations_are_safe() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
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
    model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[0], [5]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false], [true]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(state.lengths(), &[2, 3]);
    state.reorder(&[1, 0]).unwrap();
    assert_eq!(state.lengths(), &[3, 2]);
    assert!(state.truncate(&[2, 2]).is_err());
    state.truncate(&[0, 0]).unwrap();
    assert_eq!(state.lengths(), &[0, 0]);
    state.reset().unwrap();
}

/// Capacity failures happen before any terminal-state publication.
#[test]
fn capacity_error_preserves_lengths() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 2).unwrap(), 1, &device)
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
    assert!(
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert_eq!(state.lengths(), &[2]);
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

/// A non-trivial model whose Mamba steps fall below `time_step_min`.
///
/// `softplus(dt + dt_bias)` with `dt_bias = -12` is about `6e-6`, far below
/// the `1e-3` prefill clamp, so chunked prefill and the unclamped recurrent
/// update observably disagree on the same token (see [`with_dt_bias`]).
fn clamped_dt_model(half: bool) -> NemotronHForCausalLm<TestBackend> {
    use burn::module::Module;
    let model = with_dt_bias(contract_model(), -12.0);
    if half {
        model.map(&mut HalfPrecision)
    } else {
        model
    }
}

/// A one-token prompt uses the reference chunk scan, matching the stateless forward.
///
/// Regression for NEM-4: the compact row length alone selected the recurrent
/// update, so a `[BOS]`-only prompt skipped the prefill dt clamp and the F32
/// scan, and its logits and terminal state diverged from the stateless graph.
#[test]
fn one_token_prefill_uses_the_chunk_scan() {
    let device = Default::default();
    for half in [false, true] {
        let model = clamped_dt_model(half);
        let stateless = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[7]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
            .unwrap();
        let cached = model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[7]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        assert_eq!(
            cached.into_data(),
            stateless.into_data(),
            "half precision: {half}"
        );
    }
}

/// A left-padded batch whose first row holds one token matches per-row stateless runs.
#[test]
fn left_padded_single_token_row_matches_stateless() {
    let device = Default::default();
    let model = clamped_dt_model(false);
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 4).unwrap(), 2, &device)
        .unwrap();
    let cached = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 0, 5], [1, 2, 3]], &device),
                token_mask: Some(TokenMask::new(Tensor::from_bool(
                    [[false, false, true], [true, true, true]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    assert_eq!(state.lengths(), &[1, 3]);
    for (row, tokens) in [(0usize, vec![5]), (1, vec![1, 2, 3])] {
        let stateless = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_data(
                        TensorData::new(tokens.clone(), [1, tokens.len()]),
                        &device,
                    ),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        contract_close(cached.clone().narrow(0, row, 1), stateless);
    }
    // The tiny model's logits barely depend on one token's state, so compare
    // the committed state of layer 0 (the first block, fed by the normalized
    // embedding) with the reference chunk scan directly: the unclamped
    // recurrent update leaves it about 160 times smaller.
    let mixer = model.model.layers[0].mamba.as_ref().unwrap();
    let normalized = model.model.layers[0].norm.forward(
        model
            .model
            .embed_tokens
            .forward(Tensor::from_ints([[5]], &device)),
    );
    let zero = NemotronMamba2State::zeros(&mixer.config(), 1, normalized.dtype(), &device).unwrap();
    let (_, reference) = mixer
        .forward_with_state(normalized, None, &zero, NemotronMamba2Mode::ChunkedPrefill)
        .unwrap();
    let matrix = |state: &NemotronHDecodeState<TestBackend>, row: usize| {
        state
            .tensors
            .main()
            .recurrent(RECURRENT_GROUP)
            .unwrap()
            .matrix(0)
            .unwrap()
            .clone()
            .narrow(0, row, 1)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    };
    let expected = reference.recurrent.into_data().to_vec::<f32>().unwrap();
    let scale = expected
        .iter()
        .fold(0.0_f32, |max, value| max.max(value.abs()));
    assert!(scale > 0.0);
    for (actual, expected) in matrix(&state, 0).into_iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1e-4 * scale,
            "{actual} != {expected}"
        );
    }
}

/// Builds a rectangular input from per-row `(token, valid)` columns.
fn ragged_input(rows: &[Vec<(i64, bool)>]) -> CausalLmInput<TestBackend> {
    let device = Default::default();
    let width = rows[0].len();
    let tokens = rows
        .iter()
        .flat_map(|row| row.iter().map(|&(token, _)| token))
        .collect::<Vec<_>>();
    let mask = rows
        .iter()
        .flat_map(|row| row.iter().map(|&(_, valid)| valid))
        .collect::<Vec<_>>();
    CausalLmInput {
        token_ids: Tensor::from_data(TensorData::new(tokens, [rows.len(), width]), &device),
        token_mask: Some(TokenMask::new(Tensor::from_data(
            TensorData::new(mask, [rows.len(), width]),
            &device,
        ))),
        position_ids: None,
    }
}

/// Valid token IDs of one ragged row.
fn valid_tokens(row: &[(i64, bool)]) -> Vec<i64> {
    row.iter()
        .filter_map(|&(token, valid)| valid.then_some(token))
        .collect()
}

/// Columns of one ragged row that hold valid tokens.
fn valid_columns(row: &[(i64, bool)]) -> Vec<usize> {
    row.iter()
        .enumerate()
        .filter_map(|(column, &(_, valid))| valid.then_some(column))
        .collect()
}

/// Runs one row's valid tokens through its own batch-one state.
fn single_row_step(
    model: &NemotronHForCausalLm<TestBackend>,
    state: &mut NemotronHDecodeState<TestBackend>,
    tokens: &[i64],
    prefill: bool,
    options: &ForwardOptions,
) -> CausalLmOutput<TestBackend> {
    let token_ids = Tensor::from_data(
        TensorData::new(tokens.to_vec(), [1, tokens.len()]),
        &Default::default(),
    );
    if prefill {
        model
            .prefill(
                CausalLmInput {
                    token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                state,
                options,
            )
            .unwrap()
    } else {
        model
            .decode(
                DecodeInput {
                    token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                state,
                options,
            )
            .unwrap()
    }
}

/// Readable recurrent matrix and convolution of one row of Mamba layer zero.
fn row_recurrent(
    state: &NemotronHDecodeState<TestBackend>,
    row: usize,
) -> [Tensor<TestBackend, 3>; 2] {
    let cache = state.tensors.main().recurrent(RECURRENT_GROUP).unwrap();
    let matrix = cache.matrix(0).unwrap().clone().narrow(0, row, 1);
    let [_, heads, rows, columns] = matrix.dims();
    [
        matrix.reshape([1, heads * rows, columns]),
        cache.convolution(0).unwrap().clone().narrow(0, row, 1),
    ]
}

/// Readable keys `[1, kv_heads * length, head_dim]` of one row of attention layer zero.
fn row_keys(state: &NemotronHDecodeState<TestBackend>, row: usize) -> Tensor<TestBackend, 3> {
    let length = state.lengths()[row];
    let keys = state
        .tensors
        .main()
        .native_kv(KV_GROUP)
        .unwrap()
        .keys(0)
        .unwrap()
        .clone()
        .narrow(0, row, 1)
        .narrow(2, 0, length);
    let [_, heads, _, width] = keys.dims();
    keys.reshape([1, heads * length, width])
}

/// Compares every batched row with an independent batch-one run of the same steps.
///
/// `steps[s][b]` is row `b`'s rectangular columns in step `s` (step zero is
/// the prefill). Logits, every hidden state, router logits, committed KV,
/// and recurrent state must agree at every valid column; padded columns of
/// every output are exact zeros.
fn assert_batched_matches_rows(
    model: &NemotronHForCausalLm<TestBackend>,
    steps: &[Vec<Vec<(i64, bool)>>],
) {
    let device = Default::default();
    let rows = steps[0].len();
    let options = ForwardOptions {
        logits: crate::LogitsSelection::All,
        hidden_states: true,
        router_logits: true,
        ..Default::default()
    };
    let mut batched = model
        .new_decode_state(CacheConfig::new(rows, 16).unwrap(), rows, &device)
        .unwrap();
    let mut singles = (0..rows)
        .map(|_| {
            model
                .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, &device)
                .unwrap()
        })
        .collect::<Vec<_>>();
    for (index, step) in steps.iter().enumerate() {
        let prefill = index == 0;
        let input = ragged_input(step);
        let output = if prefill {
            model.prefill(input, &mut batched, &options).unwrap()
        } else {
            model
                .decode(
                    DecodeInput {
                        token_ids: input.token_ids,
                        token_mask: input.token_mask,
                        position_ids: input.position_ids,
                    },
                    &mut batched,
                    &options,
                )
                .unwrap()
        };
        let width = step[0].len();
        assert_eq!(output.logits.dims(), [rows, width, 32]);
        for (row, columns) in step.iter().enumerate() {
            let tokens = valid_tokens(columns);
            let valid = valid_columns(columns);
            let padding = (0..width).filter(|column| !valid.contains(column));
            for column in padding {
                let zero = |tensor: &Tensor<TestBackend, 3>| {
                    let channels = tensor.dims()[2];
                    tensor
                        .clone()
                        .slice([row..row + 1, column..column + 1, 0..channels])
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .all(|value| value == 0.0)
                };
                assert!(zero(&output.logits));
                assert!(output.hidden_states.iter().all(zero));
            }
            if tokens.is_empty() {
                continue;
            }
            let expected = single_row_step(model, &mut singles[row], &tokens, prefill, &options);
            let pick = |tensor: &Tensor<TestBackend, 3>, compact: usize| {
                let channels = tensor.dims()[2];
                tensor.clone().slice([
                    row..row + 1,
                    valid[compact]..valid[compact] + 1,
                    0..channels,
                ])
            };
            for compact in 0..tokens.len() {
                contract_close(
                    pick(&output.logits, compact),
                    expected.logits.clone().narrow(1, compact, 1),
                );
                for (actual, reference) in output.hidden_states.iter().zip(&expected.hidden_states)
                {
                    contract_close(
                        pick(actual, compact),
                        reference.clone().narrow(1, compact, 1),
                    );
                }
                let router = output.routers[0].logits.as_ref().unwrap();
                let reference = expected.routers[0].logits.as_ref().unwrap();
                contract_close(
                    pick(router, compact),
                    reference.clone().narrow(1, compact, 1),
                );
            }
        }
    }
    for (row, single) in singles.iter().enumerate() {
        assert_eq!(batched.lengths()[row], single.lengths()[0]);
        contract_close(row_keys(&batched, row), row_keys(single, 0));
        // Relative: with tiny steps the matrix state is far below the
        // logits' absolute tolerance, yet the two Mamba-2 paths differ by an
        // order one relative amount there.
        for (actual, expected) in row_recurrent(&batched, row)
            .into_iter()
            .zip(row_recurrent(single, 0))
        {
            assert_relative_close(
                actual.into_data(),
                expected.into_data(),
                1e-4,
                &format!("row {row} recurrent state"),
            );
        }
    }
}

/// One batched pass equals independent per-row passes for ragged prefill and decode.
///
/// The prefill mixes left padding, right padding, and an interior hole; the
/// decode steps include a finished row and ragged multi-token suffixes, so
/// both Mamba-2 modes, the attention exclusion mask, and MoE packing run with
/// padding. The clamped-`dt` model makes the two Mamba-2 paths disagree, so
/// a batched row that took another path than its batch-one run would fail
/// the relative state comparison.
#[test]
fn batched_ragged_steps_match_independent_rows() {
    for model in [contract_model(), clamped_dt_model(false)] {
        assert_ragged_steps_match_independent_rows(&model);
    }
}

/// The ragged prefill and decode schedule of [`batched_ragged_steps_match_independent_rows`].
fn assert_ragged_steps_match_independent_rows(model: &NemotronHForCausalLm<TestBackend>) {
    let pad = |token| (token, false);
    let valid = |token| (token, true);
    let steps = vec![
        vec![
            vec![pad(31), valid(1), valid(2), valid(3)],
            vec![valid(4), valid(5), pad(31), pad(31)],
            vec![valid(6), pad(31), valid(7), pad(31)],
        ],
        vec![vec![valid(10)], vec![pad(31)], vec![valid(12)]],
        vec![
            vec![valid(13), valid(14)],
            vec![valid(15), pad(31)],
            vec![pad(31), valid(16)],
        ],
        vec![vec![valid(17)], vec![valid(18)], vec![valid(19)]],
    ];
    assert_batched_matches_rows(model, &steps);
}

/// Sliding-window exclusions follow every row's own positions.
///
/// The left-padded second row has padded query slots beyond the window of
/// its valid keys under the former `W_b + j` numbering (review NEMR-6).
#[test]
fn batched_sliding_window_matches_independent_rows() {
    for mut model in [contract_model(), clamped_dt_model(false)] {
        model.model.config.sliding_window = Some(2);
        let pad = |token| (token, false);
        let valid = |token| (token, true);
        let steps = vec![
            vec![
                vec![valid(1), valid(2), valid(3), valid(4)],
                vec![pad(31), pad(31), valid(5), valid(6)],
            ],
            vec![vec![valid(7), valid(8)], vec![valid(9), pad(31)]],
            vec![vec![valid(10)], vec![valid(11)]],
        ];
        assert_batched_matches_rows(&model, &steps);
    }
}

/// Mamba-2 layer-zero state of `tokens` from a zero state, run by the mixer directly in `mode`.
///
/// Layer zero is a Mamba-2 block fed by the normalized embedding, so this is
/// the state the stateless forward reaches after `tokens`.
fn layer_zero_state(
    model: &NemotronHForCausalLm<TestBackend>,
    segments: &[(&[i64], NemotronMamba2Mode)],
) -> NemotronMamba2State<TestBackend> {
    let device = Default::default();
    let mixer = model.model.layers[0].mamba.as_ref().unwrap();
    let mut state = NemotronMamba2State::zeros(&mixer.config(), 1, DType::F32, &device).unwrap();
    for &(tokens, mode) in segments {
        let normalized = model.model.layers[0]
            .norm
            .forward(model.model.embed_tokens.forward(Tensor::from_data(
                TensorData::new(tokens.to_vec(), [1, tokens.len()]),
                &device,
            )));
        state = mixer
            .forward_with_state(normalized, None, &state, mode)
            .unwrap()
            .1;
    }
    state
}

/// Batched rows take the reference Mamba-2 path, anchored to the stateless chunk scan.
///
/// Review NEMR-1: batched-versus-batch-one comparisons cannot detect a mode
/// choice that is wrong on both sides. Here, in one ragged decode call, the
/// row appending one token must take the single-step update and the row
/// appending two must continue the chunk scan, whose state equals the scan of
/// the whole sequence from zero (the stateless graph). With the clamped `dt`
/// the alternative paths are far outside the tolerance, which the test
/// checks so it cannot pass vacuously.
#[test]
fn batched_mamba_modes_match_the_reference_scan() {
    use crate::mamba2::Mamba2Mode::{ChunkedPrefill, Recurrent};
    let model = clamped_dt_model(false);
    let device = Default::default();
    let options = ForwardOptions::default();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2], [3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap();
    let input = ragged_input(&[vec![(5, true), (31, false)], vec![(6, true), (7, true)]]);
    model
        .decode(
            DecodeInput {
                token_ids: input.token_ids,
                token_mask: input.token_mask,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap();
    assert_eq!(state.lengths(), &[3, 4]);
    // Each row: (index, reference state, the other path's state).
    let cases = [
        (
            0,
            layer_zero_state(&model, &[(&[1, 2], ChunkedPrefill), (&[5], Recurrent)]),
            layer_zero_state(&model, &[(&[1, 2, 5], ChunkedPrefill)]),
        ),
        (
            1,
            layer_zero_state(&model, &[(&[3, 4, 6, 7], ChunkedPrefill)]),
            layer_zero_state(&model, &[(&[3, 4], ChunkedPrefill), (&[6, 7], Recurrent)]),
        ),
    ];
    for (row, expected, alternative) in cases {
        let [matrix, convolution] = row_recurrent(&state, row);
        let [heads, rows, columns] = [
            expected.recurrent.dims()[1],
            expected.recurrent.dims()[2],
            expected.recurrent.dims()[3],
        ];
        let flatten = |tensor: Tensor<TestBackend, 4>| tensor.reshape([1, heads * rows, columns]);
        assert_relative_close(
            matrix.clone().into_data(),
            flatten(expected.recurrent).into_data(),
            1e-4,
            &format!("row {row} matrix state"),
        );
        assert_relative_close(
            convolution.into_data(),
            expected.convolution.into_data(),
            1e-5,
            &format!("row {row} convolution state"),
        );
        let gap = relative_difference(
            matrix.into_data(),
            flatten(alternative.recurrent).into_data(),
        );
        assert!(
            gap > 0.05,
            "row {row}: the other Mamba-2 path is indistinguishable ({gap})"
        );
    }
}

/// Batched rows survive reorder and truncation like independent sequences.
#[test]
fn batched_rows_follow_reorder_and_reset() {
    let model = contract_model();
    let device = Default::default();
    let options = ForwardOptions::default();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2], [3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap();
    state.reorder(&[1, 0]).unwrap();
    let swapped = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[5], [6]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap();
    for (row, prompt) in [(0, [3, 4, 5]), (1, [1, 2, 6])] {
        let expected = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([prompt], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &options,
            )
            .unwrap();
        contract_close(swapped.logits.clone().narrow(0, row, 1), expected.logits);
    }
    state.truncate(&[0, 0]).unwrap();
    let fresh = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[7], [8]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap();
    assert_eq!(state.lengths(), &[1, 1]);
    for (row, token) in [(0, 7), (1, 8)] {
        let expected = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[token]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &options,
            )
            .unwrap();
        contract_close(fresh.logits.clone().narrow(0, row, 1), expected.logits);
    }
}

/// A multi-token transactional step stages per-token states, so any prefix commits exactly.
///
/// The staged logits equal token-by-token decoding, and after committing the
/// first `k` of three proposals the state continues exactly like a reference
/// that decoded only those `k` tokens. The clamped-`dt` model makes a step
/// that ran the prefill chunk scan (with its `dt` clamp) instead of the
/// single-step update fail the relative state comparison.
#[test]
fn multi_token_transaction_commits_any_prefix() {
    let device = Default::default();
    let model = with_dt_bias(contract_model_with(true), -12.0);
    let options = ForwardOptions {
        logits: crate::LogitsSelection::All,
        ..Default::default()
    };
    let draft = [4, 5, 6];
    for accepted in 1..=draft.len() {
        let mut speculative = model
            .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 3, &device)
            .unwrap();
        single_row_step(&model, &mut speculative, &[1, 2, 3], true, &options);
        let mark = speculative.mark().unwrap();
        let staged = single_row_step(&model, &mut speculative, &draft, false, &options);
        assert_eq!(speculative.tensor_state().working_lengths(), &[6]);
        speculative.commit_prefixes(mark, accepted, 0).unwrap();
        assert_eq!(speculative.lengths(), &[3 + accepted]);

        let mut reference = model
            .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, &device)
            .unwrap();
        single_row_step(&model, &mut reference, &[1, 2, 3], true, &options);
        for (index, &token) in draft.iter().enumerate() {
            let expected = single_row_step(&model, &mut reference, &[token], false, &options);
            contract_close(staged.logits.clone().narrow(1, index, 1), expected.logits);
            if index + 1 == accepted {
                break;
            }
        }
        contract_close(row_keys(&speculative, 0), row_keys(&reference, 0));
        for (actual, expected) in row_recurrent(&speculative, 0)
            .into_iter()
            .zip(row_recurrent(&reference, 0))
        {
            assert_relative_close(
                actual.into_data(),
                expected.into_data(),
                1e-4,
                &format!("{accepted} accepted tokens"),
            );
        }
        let next = single_row_step(&model, &mut speculative, &[7], false, &options);
        let expected = single_row_step(&model, &mut reference, &[7], false, &options);
        contract_close(next.logits, expected.logits);
    }
}

/// A multi-token transactional step follows token-by-token decode semantics (review NEMR-3).
///
/// Inside a transaction the step runs the single-step update per token, which
/// skips the prefill `dt` clamp; the same multi-token decode outside a
/// transaction runs the chunk scan. With steps below `time_step_min` the two
/// committed states differ, and the transactional one equals one-token
/// decodes. This pins the documented contract of `NemotronHDecodeState::mark`.
#[test]
fn transactional_multi_token_steps_follow_token_by_token_decoding() {
    let device = Default::default();
    let model = with_dt_bias(contract_model_with(true), -12.0);
    let options = ForwardOptions::default();
    let prompt = [1, 2, 3];
    let suffix = [4, 5, 6];
    let mut transactional = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 3, &device)
        .unwrap();
    single_row_step(&model, &mut transactional, &prompt, true, &options);
    let mark = transactional.mark().unwrap();
    single_row_step(&model, &mut transactional, &suffix, false, &options);
    transactional
        .commit_prefixes(mark, suffix.len(), 0)
        .unwrap();

    let mut stepwise = model
        .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, &device)
        .unwrap();
    single_row_step(&model, &mut stepwise, &prompt, true, &options);
    for token in suffix {
        single_row_step(&model, &mut stepwise, &[token], false, &options);
    }
    let mut chunked = model
        .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, &device)
        .unwrap();
    single_row_step(&model, &mut chunked, &prompt, true, &options);
    single_row_step(&model, &mut chunked, &suffix, false, &options);

    let [matrix, _] = row_recurrent(&transactional, 0);
    let [stepwise_matrix, _] = row_recurrent(&stepwise, 0);
    let [chunked_matrix, _] = row_recurrent(&chunked, 0);
    assert_relative_close(
        matrix.clone().into_data(),
        stepwise_matrix.into_data(),
        1e-4,
        "transactional against one-token decodes",
    );
    let gap = relative_difference(matrix.into_data(), chunked_matrix.into_data());
    assert!(gap > 0.05, "chunked decode is indistinguishable ({gap})");
}

/// Multi-token proposals need prefilled rows; the failed step leaves the transaction usable.
#[test]
fn multi_token_transaction_requires_existing_state() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let options = ForwardOptions::default();
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 3, &device)
        .unwrap();
    let mark = state.mark().unwrap();
    let error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap_err();
    assert!(matches!(&error, Error::Cache(reason) if reason.contains("prefilled")));
    assert_eq!(state.tensor_state().working_lengths(), &[0]);
    // A single first token still runs the reference chunk scan inside a transaction.
    single_row_step(&model, &mut state, &[1], false, &options);
    assert_eq!(state.tensor_state().working_lengths(), &[1]);
    state.rollback(mark).unwrap();
    assert_eq!(state.lengths(), &[0]);
}

/// Release-mode timing of one batched pass against one pass per row.
///
/// The per-row baseline evaluates every row through its own batch-one state,
/// which is the cost model of the former row-by-row implementation. Run with
/// `cargo test --release --lib nemotron_h::model::decode::tests::batched_decode_timing
/// -- --ignored --nocapture`.
#[test]
#[ignore = "release-mode timing benchmark"]
fn batched_decode_timing() {
    use std::time::Instant;

    let device = Default::default();
    let mut config = tiny_config(false);
    config.vocab_size = 512;
    config.hidden_size = 256;
    config.layers_block_type = vec![
        NemotronHBlockType::LinearAttention,
        NemotronHBlockType::Moe,
        NemotronHBlockType::FullAttention,
        NemotronHBlockType::Moe,
        NemotronHBlockType::LinearAttention,
        NemotronHBlockType::Moe,
        NemotronHBlockType::LinearAttention,
        NemotronHBlockType::Mlp,
    ];
    config.num_hidden_layers = Some(8);
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
    config.head_dim = 64;
    config.max_position_embeddings = 512;
    config.intermediate_size = 512;
    config.ssm_state_size = 32;
    config.mamba_num_heads = 8;
    config.mamba_head_dim = 32;
    config.n_groups = 2;
    config.conv_kernel = 4;
    config.chunk_size = 32;
    config.n_routed_experts = 16;
    config.num_experts_per_tok = 4;
    config.n_group = 1;
    config.topk_group = 1;
    config.moe_intermediate_size = 128;
    config.moe_shared_expert_intermediate_size = 256;
    let model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let (rows, prompt, steps) = (8, 64, 16);
    let prompt_tokens = |row: usize| {
        (0..prompt)
            .map(|column| ((row * 131 + column * 17) % 500) as i64 + 1)
            .collect::<Vec<_>>()
    };
    let options = ForwardOptions::default();
    let timed = |label: &str, run: &mut dyn FnMut()| {
        let started = Instant::now();
        run();
        let seconds = started.elapsed().as_secs_f64();
        eprintln!("nemotron batched timing: {label}: {:.1} ms", seconds * 1e3);
        seconds
    };
    for repeat in 0..2 {
        let mut batched = model
            .new_decode_state(
                CacheConfig::new(rows, prompt + steps).unwrap(),
                rows,
                &device,
            )
            .unwrap();
        let mut singles = (0..rows)
            .map(|_| {
                model
                    .new_decode_state(CacheConfig::new(1, prompt + steps).unwrap(), 1, &device)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let flat = (0..rows).flat_map(prompt_tokens).collect::<Vec<_>>();
        let batched_prefill = timed("batched prefill", &mut || {
            model
                .prefill(
                    CausalLmInput {
                        token_ids: Tensor::from_data(
                            TensorData::new(flat.clone(), [rows, prompt]),
                            &device,
                        ),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut batched,
                    &options,
                )
                .unwrap()
                .logits
                .into_data();
        });
        let per_row_prefill = timed("per-row prefill", &mut || {
            for (row, state) in singles.iter_mut().enumerate() {
                single_row_step(&model, state, &prompt_tokens(row), true, &options)
                    .logits
                    .into_data();
            }
        });
        let batched_decode = timed("batched decode", &mut || {
            for step in 0..steps {
                let tokens = (0..rows)
                    .map(|row| ((row + step * 7) % 500) as i64 + 1)
                    .collect::<Vec<_>>();
                model
                    .decode(
                        DecodeInput {
                            token_ids: Tensor::from_data(
                                TensorData::new(tokens, [rows, 1]),
                                &device,
                            ),
                            token_mask: None,
                            position_ids: None,
                        },
                        &mut batched,
                        &options,
                    )
                    .unwrap()
                    .logits
                    .into_data();
            }
        });
        let per_row_decode = timed("per-row decode", &mut || {
            for step in 0..steps {
                for (row, state) in singles.iter_mut().enumerate() {
                    single_row_step(
                        &model,
                        state,
                        &[((row + step * 7) % 500) as i64 + 1],
                        false,
                        &options,
                    )
                    .logits
                    .into_data();
                }
            }
        });
        if repeat == 1 {
            eprintln!(
                "nemotron batched timing: prefill speedup {:.2}x, decode speedup {:.2}x",
                per_row_prefill / batched_prefill,
                per_row_decode / batched_decode
            );
        }
    }
}

/// The trait accessors report the allocated rows, capacity, and device.
#[test]
fn decode_state_trait_accessors_report_allocation_geometry() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let state = model
        .new_decode_state(CacheConfig::new(3, 8).unwrap(), 3, &device)
        .unwrap();
    assert_eq!(DecodeState::batch_size(&state), 3);
    assert_eq!(DecodeState::capacity(&state), 8);
    assert_eq!(DecodeState::device(&state), &device);
    assert_eq!(state.lengths(), &[0, 0, 0]);
    assert_eq!(state.attention_layer_count(), 1);
    assert_eq!(state.mamba_layer_count(), 1);
    assert_eq!(state.mtp_layer_count(), 0);
    assert_eq!(state.staging_capacity(), 0);
}

/// A state allocated for another schedule is rejected before any model work or mutation.
#[test]
fn foreign_decode_state_is_rejected_before_work() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let other = NemotronHForCausalLm::<TestBackend>::init(
        crate::models::nemotron_h::NemotronHConfig {
            layers_block_type: vec![
                NemotronHBlockType::FullAttention,
                NemotronHBlockType::FullAttention,
            ],
            num_hidden_layers: Some(2),
            ..tiny_config(false)
        },
        &device,
    )
    .unwrap();
    let mut foreign = other
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let incompatible = |error: Error| {
        assert!(
            matches!(&error, Error::Cache(reason)
                if reason == "Nemotron-H decode state is incompatible with the model"),
            "{error}"
        );
    };
    incompatible(
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut foreign,
                &ForwardOptions::default(),
            )
            .unwrap_err(),
    );
    incompatible(model.verify_tokens(&[1], &mut foreign).unwrap_err());
    assert_eq!(foreign.lengths(), &[0]);
}

/// The shifted MTP cursor requires one truncation length per row.
#[test]
fn speculative_truncate_rejects_a_mismatched_row_count() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 8).unwrap(), 1, 2, &device)
        .unwrap();
    let error = state.truncate(&[0, 0]).unwrap_err();
    assert!(
        matches!(
            error,
            Error::TensorCache(TensorCacheError::Metadata(
                DecodeStateError::SequenceCountMismatch {
                    expected: 1,
                    actual: 2
                }
            ))
        ),
        "{error}"
    );
}

/// The verifier pass accepts only a non-empty in-vocabulary batch-one row within the context.
#[test]
fn verify_tokens_rejects_invalid_rows_and_context_overflow() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let mut batched = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 32).unwrap(), 1, &device)
        .unwrap();
    for (tokens, batch, actual) in [
        (&[1_u32][..], 2, "1 tokens for batch 2"),
        (&[][..], 1, "0 tokens for batch 1"),
        (&[1, 32][..], 1, "2 tokens for batch 1"),
    ] {
        let target = if batch == 2 { &mut batched } else { &mut state };
        let error = model.verify_tokens(tokens, target).unwrap_err();
        assert!(
            matches!(&error, Error::Shape { tensor, expected, actual: found }
                if tensor == "nemotron_h.input_ids"
                && expected == "a non-empty batch-one row of token values in 0..32"
                && found == actual),
            "{error}"
        );
    }
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1; 30]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let error = model.verify_tokens(&[1, 2, 3], &mut state).unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, reason }
            if field == "nemotron_h.position"
            && reason == "cached length exceeds max_position_embeddings"),
        "{error}"
    );
    assert_eq!(state.lengths(), &[30]);
    // Two more tokens still fit and are published.
    let (logits, hidden) = model.verify_tokens(&[1, 2], &mut state).unwrap();
    assert_eq!(logits.dims(), [2, 32]);
    assert_eq!(hidden.dims(), [1, 2, 8]);
    assert_eq!(state.lengths(), &[32]);
}

/// A step in which every row is finished costs no model work and returns only exact zeros.
#[test]
fn fully_inactive_decode_returns_zero_logits_without_diagnostics() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
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
    let snapshot = contract_cache_snapshot(&state);
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[5], [6]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false], [false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(output.logits.dims(), [2, 1, 32]);
    assert!(
        output
            .logits
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .all(|value| value == 0.0)
    );
    assert!(output.final_hidden_state.is_none());
    assert!(output.hidden_states.is_empty());
    assert!(output.routers.is_empty());
    assert_eq!(state.lengths(), &[2, 2]);
    assert_eq!(contract_cache_snapshot(&state), snapshot);
}

/// Every serialized recurrent-cache dtype maps to its Burn dtype; others are rejected.
#[test]
fn mamba_state_dtype_maps_every_serialized_name() {
    assert_eq!(mamba_state_dtype("float32").unwrap(), DType::F32);
    assert_eq!(mamba_state_dtype("float16").unwrap(), DType::F16);
    assert_eq!(mamba_state_dtype("bfloat16").unwrap(), DType::BF16);
    let error = mamba_state_dtype("int8").unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, reason }
            if field == "nemotron_h.mamba_ssm_cache_dtype"
            && reason == "unsupported recurrent cache dtype \"int8\""),
        "{error}"
    );
}

/// Reduced-precision recurrent caches allocate in the declared dtype.
#[test]
fn reduced_precision_recurrent_cache_uses_the_declared_dtype() {
    let device = Default::default();
    for (name, dtype) in [("float16", DType::F16), ("bfloat16", DType::BF16)] {
        let model = NemotronHForCausalLm::<TestBackend>::init(
            crate::models::nemotron_h::NemotronHConfig {
                mamba_ssm_cache_dtype: name.into(),
                ..tiny_config(false)
            },
            &device,
        )
        .unwrap();
        let state = model
            .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
            .unwrap();
        let recurrent = state
            .tensor_state()
            .main()
            .recurrent(RECURRENT_GROUP)
            .unwrap();
        assert_eq!(recurrent.matrix(0).unwrap().dtype(), dtype, "{name}");
    }
}
