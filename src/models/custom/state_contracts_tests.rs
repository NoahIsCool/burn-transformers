//! Public decode-state contracts, including sparse diagnostics and transactional failures.

use super::*;
use crate::{CausalLmOutput, Error};

/// Creates two distinct sparse layers with every expert initialized before any inference.
fn sparse_model(device: &<TestBackend as BackendTypes>::Device) -> ModularCausalLm<TestBackend> {
    let mut block = layer(attention(PositionEncodingSpec::None));
    block.feed_forward = FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
        routed_experts: 4,
        experts_per_token: 2,
        shared_experts: 1,
        expert_intermediate_size: 6,
        activation: Activation::Silu,
        renormalize_selected: true,
        routed_scale: 0.75,
    });
    ModularCausalLm::init(decoder(vec![block.clone(), block]), device)
        .unwrap()
        .map(&mut DeterministicParameters::default())
}

/// Checks structured shape context rather than accepting a generic error string.
fn assert_shape_error(error: Error, tensor: &str, expected: &str, actual: &str) {
    match error {
        Error::Shape {
            tensor: found_tensor,
            expected: found_expected,
            actual: found_actual,
        } => {
            assert_eq!(found_tensor, tensor);
            assert_eq!(found_expected, expected);
            assert_eq!(found_actual, actual);
        }
        other => panic!("expected contextual shape error, received {other:?}"),
    }
}

/// Evaluates `E * sum_e mean(top_k_assignments_e) * mean(softmax(logits)_e)` in F64.
///
/// The supplied rows contain only newly appended valid tokens. The assignment mean counts each
/// selected expert once per token, without dividing by top-k, matching the documented MoE loss.
fn host_load_balancing_loss(rows: &[f32], experts: usize, top_k: usize) -> f32 {
    assert!(!rows.is_empty() && rows.len().is_multiple_of(experts));
    let count = rows.len() / experts;
    let mut assignments = vec![0.0_f64; experts];
    let mut probabilities = vec![0.0_f64; experts];
    for row in rows.chunks_exact(experts) {
        assert!(row.iter().all(|value| value.is_finite()));
        let mut order = (0..experts).collect::<Vec<_>>();
        order.sort_by(|&left, &right| row[right].total_cmp(&row[left]).then(left.cmp(&right)));
        assert!((row[order[top_k - 1]] - row[order[top_k]]).abs() > 1e-5);
        for &expert in &order[..top_k] {
            assignments[expert] += 1.0;
        }
        let maximum = f64::from(row[order[0]]);
        let exponentials = row
            .iter()
            .map(|&value| (f64::from(value) - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = exponentials.iter().sum::<f64>();
        for (sum, probability) in probabilities.iter_mut().zip(exponentials) {
            *sum += probability / denominator;
        }
    }
    (experts as f64
        * assignments
            .iter()
            .zip(probabilities)
            .map(|(assigned, probability)| assigned * probability)
            .sum::<f64>()
        / (count * count) as f64) as f32
}

/// Reconstructs the new-step diagnostic layout from a compact two-row full-history output.
fn expected_new_step(tensor: Tensor<TestBackend, 3>) -> TensorData {
    let [batch, sequence, width] = tensor.dims();
    assert_eq!([batch, sequence], [2, 4]);
    let data = tensor.into_data();
    let source = data.as_slice::<f32>().unwrap();
    let mut values = vec![0.0; 2 * 3 * width];
    values[..width].copy_from_slice(&source[2 * width..3 * width]);
    values[2 * width..3 * width].copy_from_slice(&source[3 * width..4 * width]);
    TensorData::new(values, [2, 3, width])
}

/// Sparse cached output returns only the new valid tokens and independently gated diagnostics.
#[test]
fn cached_sparse_diagnostics_match_new_tokens_not_entire_history() {
    let device = Default::default();
    let model = sparse_model(&device);
    let full_options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        router_logits: true,
        auxiliary_losses: true,
        ..Default::default()
    };
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 6, 7], [3, 4, 5, 0]], &device),
                token_mask: Some(mask(
                    [[true, true, true, true], [true, true, true, false]],
                    &device,
                )),
                position_ids: None,
            },
            &full_options,
        )
        .unwrap();
    assert_eq!(full.routers.len(), 2);
    for (router_logits, auxiliary_losses) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 0], [3, 4, 5]], &device),
                    token_mask: Some(mask([[true, true, false], [true, true, true]], &device)),
                    position_ids: None,
                },
                &mut state,
                &full_options,
            )
            .unwrap();
        assert_eq!(state.lengths(), &[2, 3]);
        let cached = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[6, 0, 7], [0, 0, 0]], &device),
                    token_mask: Some(mask([[true, false, true], [false, false, false]], &device)),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions {
                    router_logits,
                    auxiliary_losses,
                    ..full_options.clone()
                },
            )
            .unwrap();
        assert_eq!(state.lengths(), &[4, 3]);
        assert_finite_attention_parity(
            &cached.logits.into_data(),
            &expected_new_step(full.logits.clone()),
        );
        assert_finite_attention_parity(
            &cached.final_hidden_state.unwrap().into_data(),
            &expected_new_step(full.final_hidden_state.clone().unwrap()),
        );
        assert_eq!(cached.hidden_states.len(), full.hidden_states.len());
        for (actual, reference) in cached.hidden_states.into_iter().zip(&full.hidden_states) {
            assert_finite_attention_parity(
                &actual.into_data(),
                &expected_new_step(reference.clone()),
            );
        }
        if !router_logits && !auxiliary_losses {
            assert!(cached.routers.is_empty());
            continue;
        }
        assert_eq!(cached.routers.len(), full.routers.len());
        for (actual, reference) in cached.routers.iter().zip(&full.routers) {
            assert!(actual.z_loss.is_none());
            assert_eq!(actual.logits.is_some(), router_logits);
            assert_eq!(actual.load_balancing_loss.is_some(), auxiliary_losses);
            let raw = reference.logits.as_ref().unwrap();
            if let Some(logits) = &actual.logits {
                assert_finite_attention_parity(
                    &logits.clone().into_data(),
                    &expected_new_step(raw.clone()),
                );
            }
            if let Some(loss) = &actual.load_balancing_loss {
                let raw = raw.clone().into_data();
                let rows = &raw.as_slice::<f32>().unwrap()[8..16];
                let expected = host_load_balancing_loss(rows, 4, 2);
                assert_finite_attention_parity(
                    &loss.clone().into_data(),
                    &TensorData::new(vec![expected], [1]),
                );
                let history_loss = reference
                    .load_balancing_loss
                    .as_ref()
                    .unwrap()
                    .clone()
                    .into_scalar();
                assert!(
                    (expected - history_loss).abs() > 1e-4,
                    "fixture must distinguish new-token loss from full-history loss"
                );
            }
        }
    }
}

/// Recomputes the two valid post-error continuations from their committed token/position prefixes.
fn assert_causal_continuation(
    model: &ModularCausalLm<TestBackend>,
    state: &mut super::super::ModularDecodeState<TestBackend>,
    device: &<TestBackend as BackendTypes>::Device,
) {
    let options = ForwardOptions::default();
    let actual = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[9], [10]], device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[12], [13]], device)),
            },
            state,
            &options,
        )
        .unwrap()
        .logits;
    assert_eq!(state.lengths(), &[3, 3]);
    for (row, ids, positions) in [(0, [1, 2, 9], [2, 5, 12]), (1, [3, 4, 10], [3, 6, 13])] {
        let expected = model
            .forward(parity_input(&ids, None, Some(&positions), device), &options)
            .unwrap()
            .logits;
        assert_token_logits_match(&actual, row, 0, &expected, 0);
    }
}

/// Invalid output selection fails after graph evaluation without committing prospective tokens.
#[test]
fn invalid_cached_output_selection_preserves_committed_history() {
    let device = Default::default();
    let model = deterministic_attention_model(
        Some(2),
        PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 4,
            scaling: RopeScaling::None,
        },
        &device,
    );
    for selection in [vec![], vec![2]] {
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2], [3, 4]], &device),
                    token_mask: None,
                    position_ids: Some(Tensor::from_ints([[2, 5], [3, 6]], &device)),
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let error = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[5, 6], [7, 8]], &device),
                    token_mask: None,
                    position_ids: Some(Tensor::from_ints([[8, 9], [10, 11]], &device)),
                },
                &mut state,
                &ForwardOptions {
                    logits: LogitsSelection::Positions(selection.clone()),
                    hidden_states: true,
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert_shape_error(
            error,
            "forward_options.logits.positions",
            "a non-empty list of indices below 2",
            &format!("{selection:?}"),
        );
        assert_eq!(state.lengths(), &[2, 2]);
        assert_causal_continuation(&model, &mut state, &device);
    }
}

/// A bad position in a later batch row cannot publish an earlier row's tentative writes.
#[test]
fn late_row_position_validation_preserves_committed_history() {
    let device = Default::default();
    let model = deterministic_attention_model(
        Some(2),
        PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 4,
            scaling: RopeScaling::None,
        },
        &device,
    );
    for invalid_position in [-1, 32] {
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2], [3, 4]], &device),
                    token_mask: None,
                    position_ids: Some(Tensor::from_ints([[2, 5], [3, 6]], &device)),
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let error = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[5], [6]], &device),
                    token_mask: None,
                    position_ids: Some(Tensor::from_ints([[8], [invalid_position]], &device)),
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap_err();
        assert_shape_error(
            error,
            "custom.decode.position_ids",
            "values in 0..32",
            &format!("observed range {invalid_position}..={invalid_position}"),
        );
        assert_eq!(state.lengths(), &[2, 2]);
        assert_causal_continuation(&model, &mut state, &device);
    }
}

/// Requires every returned tensor row for an inactive sequence to be finite zero.
fn assert_inactive_output_row(output: &CausalLmOutput<TestBackend>, row: usize) {
    let tensors = core::iter::once(&output.logits)
        .chain(output.final_hidden_state.iter())
        .chain(&output.hidden_states);
    for tensor in tensors {
        let [_, sequence, width] = tensor.dims();
        let data = tensor
            .clone()
            .slice([row..row + 1, 0..sequence, 0..width])
            .into_data();
        assert!(
            data.as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|&value| value == 0.0)
        );
    }
}

/// Empty prompt rows and entirely inactive steps fail atomically; started inactive rows are valid.
#[test]
fn inactive_rows_obey_prefill_and_decode_state_contracts() {
    let device = Default::default();
    let model = deterministic_attention_model(None, PositionEncodingSpec::None, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    assert_eq!(state.device(), &device);
    let error = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1], [0]], &device),
                token_mask: Some(mask([[true], [false]], &device)),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert_shape_error(
        error,
        "custom.prefill.token_mask",
        "at least one valid token in every prompt row",
        "a fully masked row",
    );
    assert_eq!(state.lengths(), &[0, 0]);
    let error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[1], [0]], &device),
                token_mask: Some(mask([[true], [false]], &device)),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert!(matches!(error, Error::BackendCapability(ref reason)
        if reason == "reference batched decoding cannot evaluate a never-started inactive row"));
    assert_eq!(state.lengths(), &[0, 0]);
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
    let error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[0, 0], [0, 0]], &device),
                token_mask: Some(mask([[false, false], [false, false]], &device)),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert_shape_error(
        error,
        "custom.decode.token_mask",
        "at least one active token across the batch",
        "an entirely inactive decode step",
    );
    assert_eq!(state.lengths(), &[2, 2]);
    let actual = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[0, 5], [0, 0]], &device),
                token_mask: Some(mask([[false, true], [false, false]], &device)),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(state.lengths(), &[3, 2]);
    assert_eq!(actual.logits.dims(), [2, 1, 32]);
    assert_inactive_output_row(&actual, 1);
    let expected = model
        .forward(
            parity_input(&[1, 2, 5], None, None, &device),
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_token_logits_match(&actual.logits, 0, 0, &expected.logits, 0);
    let resumed = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[0], [6]], &device),
                token_mask: Some(mask([[false], [true]], &device)),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(state.lengths(), &[3, 3]);
    assert_inactive_output_row(&resumed, 0);
    let expected = model
        .forward(
            parity_input(&[3, 4, 6], None, None, &device),
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_token_logits_match(&resumed.logits, 1, 0, &expected.logits, 0);
}

/// Equal decoder specs still reject causal/seq2seq state-role swaps without corrupting histories.
#[test]
fn equal_decoder_topologies_do_not_allow_cross_role_state_reuse() {
    let device = Default::default();
    let spec = seq2seq_spec();
    let causal = ModularCausalLm::<TestBackend>::init(spec.decoder.clone(), &device)
        .unwrap()
        .map(&mut DeterministicParameters::default());
    let seq2seq = ModularSeq2SeqLm::<TestBackend>::init(spec, &device)
        .unwrap()
        .map(&mut DeterministicParameters::default());
    assert_eq!(causal.spec(), &seq2seq.spec().decoder);
    let options = ForwardOptions::default();
    let encoder = seq2seq
        .encode(parity_input(&[4, 5], None, None, &device), &options)
        .unwrap();
    let mut causal_state = causal
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let mut seq2seq_state = seq2seq
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    causal
        .prefill(
            parity_input(&[1, 2], None, None, &device),
            &mut causal_state,
            &options,
        )
        .unwrap();
    seq2seq
        .prefill_decoder(
            &encoder,
            parity_input(&[1, 2], None, None, &device),
            &mut seq2seq_state,
            &options,
        )
        .unwrap();
    let input = || DecodeInput {
        token_ids: Tensor::from_ints([[3]], &device),
        token_mask: None,
        position_ids: None,
    };
    for error in [
        causal
            .decode(input(), &mut seq2seq_state, &options)
            .unwrap_err(),
        seq2seq
            .decode(&encoder, input(), &mut causal_state, &options)
            .unwrap_err(),
    ] {
        assert!(matches!(error, Error::Cache(ref reason)
            if reason == "custom decode state belongs to a different model topology or role"));
    }
    assert_eq!(causal_state.lengths(), &[2]);
    assert_eq!(seq2seq_state.lengths(), &[2]);
    let causal_actual = causal.decode(input(), &mut causal_state, &options).unwrap();
    let causal_expected = causal
        .forward(parity_input(&[1, 2, 3], None, None, &device), &options)
        .unwrap();
    assert_finite_attention_parity(
        &causal_actual.logits.into_data(),
        &causal_expected.logits.into_data(),
    );
    let seq2seq_actual = seq2seq
        .decode(&encoder, input(), &mut seq2seq_state, &options)
        .unwrap();
    let seq2seq_expected = seq2seq
        .forward(
            Seq2SeqInput {
                encoder: parity_input(&[4, 5], None, None, &device),
                decoder: parity_input(&[1, 2, 3], None, None, &device),
            },
            &options,
        )
        .unwrap();
    assert_finite_attention_parity(
        &seq2seq_actual.decoder.logits.into_data(),
        &seq2seq_expected.decoder.logits.into_data(),
    );
    assert_eq!(causal_state.lengths(), &[3]);
    assert_eq!(seq2seq_state.lengths(), &[3]);
}

/// Invalid seq2seq output selection leaves the target prefix available for a correct retry.
#[test]
fn seq2seq_output_validation_failure_does_not_commit_decoder_tokens() {
    let device = Default::default();
    let model = ModularSeq2SeqLm::<TestBackend>::init(seq2seq_spec(), &device)
        .unwrap()
        .map(&mut DeterministicParameters::default());
    let options = ForwardOptions::default();
    let encoder = model
        .encode(parity_input(&[4, 5], None, None, &device), &options)
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            parity_input(&[1, 2], None, None, &device),
            &mut state,
            &options,
        )
        .unwrap();
    let error = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: Tensor::from_ints([[6]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::Positions(vec![1]),
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_shape_error(
        error,
        "forward_options.logits.positions",
        "a non-empty list of indices below 1",
        "[1]",
    );
    assert_eq!(state.lengths(), &[2]);
    let actual = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: Tensor::from_ints([[3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap();
    let expected = model
        .forward(
            Seq2SeqInput {
                encoder: parity_input(&[4, 5], None, None, &device),
                decoder: parity_input(&[1, 2, 3], None, None, &device),
            },
            &options,
        )
        .unwrap();
    assert_finite_attention_parity(
        &actual.decoder.logits.into_data(),
        &expected.decoder.logits.into_data(),
    );
    assert_eq!(state.lengths(), &[3]);
}
