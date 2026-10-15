//! Nemotron-H cached parity, ragged diagnostics, and atomic cache-lifecycle tests.

use burn::{
    backend::Flex,
    tensor::{Bool, Tensor, TensorData, Tolerance},
};

use crate::{DecodeState, architecture::TokenMask, models::nemotron_h::model::tests::tiny_config};

use super::*;

/// CPU Flex backend with F32 activations and I32 indices for cache comparisons.
type TestBackend = Flex<f32, i32>;

/// Loads deterministic nonzero parameters through the public checkpoint sink.
fn contract_model() -> NemotronHForCausalLm<TestBackend> {
    use crate::{checkpoint::TensorSink, models::nemotron_h::NemotronHCheckpointOptions};
    let device = Default::default();
    let mut config = tiny_config(false);
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
    let kv = state
        .tensors
        .main()
        .native_kv(KV_GROUP)
        .unwrap()
        .keys(0)
        .unwrap()
        .clone()
        .into_data();
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
    assert_eq!(
        state
            .tensors
            .main()
            .native_kv(KV_GROUP)
            .unwrap()
            .keys(0)
            .unwrap()
            .clone()
            .into_data(),
        kv
    );
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

/// Cached mask construction rejects windows that cannot enter signed indices.
#[cfg(target_pointer_width = "64")]
#[test]
fn oversized_cached_attention_window_is_rejected() {
    let device = Default::default();
    assert!(cached_attention_mask::<TestBackend>(0, 1, 1, Some(usize::MAX), &device).is_err());
}
