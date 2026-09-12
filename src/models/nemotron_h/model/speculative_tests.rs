//! Nemotron-H speculative acceptance, EOS alignment, sampling, and retry tests.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData, backend::BackendTypes},
};

use crate::{
    CausalLmInput,
    cache::CacheConfig,
    checkpoint::TensorSink,
    models::nemotron_h::{
        NemotronHBlockType, NemotronHCheckpointOptions, NemotronHConfig, model::tests::tiny_config,
    },
};

use super::*;

/// CPU Flex backend with F32 logits and I32 indices for reproducible draft fixtures.
type TestBackend = Flex<f32, i32>;

/// Uses a cheap base MLP while retaining Lightning's exact MTP stage.
fn speculative_config() -> NemotronHConfig {
    let mut config = tiny_config(true);
    config.layers_block_type = vec![NemotronHBlockType::Mlp];
    config.num_hidden_layers = Some(1);
    config.tie_word_embeddings = false;
    config
}

/// Installs deterministic parameters, optionally adding a two-token target cycle.
///
/// Each parameter is installed exactly once in checkpoint layout. Burn stores
/// linear weights transposed relative to SafeTensors, so overwriting an already
/// mapped parameter would make this fixture depend on the sink's runtime-layout
/// compatibility path rather than exercising the checkpoint mapping itself.
fn install_parameters(model: &mut NemotronHForCausalLm<TestBackend>, scripted_target: bool) {
    let contract = model
        .config()
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    for expected in contract.expected_tensors() {
        let shape = expected
            .shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap())
            .collect::<Vec<_>>();
        let elements = shape.iter().product();
        let mut values = vec![0.0_f32; elements];
        if expected.name.contains("norm.weight") {
            values.fill(1.0);
        }
        if scripted_target && expected.name == "model.embed_tokens.weight" {
            let hidden = model.config().hidden_size;
            values[0] = 1.0;
            values[hidden + 1] = 1.0;
        }
        if scripted_target && expected.name == "lm_head.weight" {
            let hidden = model.config().hidden_size;
            // SafeTensors layout is [vocabulary, hidden]: hidden coordinate
            // one predicts token zero and coordinate zero predicts token one.
            values[1] = 1.0;
            values[hidden] = 1.0;
        }
        model
            .write_tensor(&expected.name, TensorData::new(values, shape))
            .unwrap();
    }
}

/// Creates a graph whose target and auxiliary logits tie at token zero.
fn zero_model(device: &<TestBackend as BackendTypes>::Device) -> NemotronHForCausalLm<TestBackend> {
    let mut model = NemotronHForCausalLm::init(speculative_config(), device).unwrap();
    install_parameters(&mut model, false);
    model
}

/// Creates target transitions `1 -> 0` and `0 -> 1`, while MTP proposes zero.
fn scripted_rejection_model(
    device: &<TestBackend as BackendTypes>::Device,
) -> NemotronHForCausalLm<TestBackend> {
    let mut model = NemotronHForCausalLm::init(speculative_config(), device).unwrap();
    install_parameters(&mut model, true);
    model
}

/// Prefills target and shifted MTP streams at the stable one-token offset.
fn aligned_state(
    model: &NemotronHForCausalLm<TestBackend>,
    device: &<TestBackend as BackendTypes>::Device,
) -> (NemotronHDecodeState<TestBackend>, Tensor<TestBackend, 3>) {
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, device)
        .unwrap();
    let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[2, 3]], device);
    let hidden = model
        .prefill(
            CausalLmInput {
                token_ids: prompt.clone(),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap()
        .final_hidden_state
        .unwrap();
    model
        .prefill_mtp_cache(prompt, hidden.clone(), &mut state)
        .unwrap();
    (state, hidden.narrow(1, 1, 1))
}

/// Default capacity is two and zero-draft configurations fail closed.
#[test]
fn config_defaults_and_validation_are_strict() {
    assert_eq!(NemotronHSpeculativeConfig::default().max_draft_tokens, 2);
    assert!(
        NemotronHSpeculativeConfig {
            max_draft_tokens: 0,
            ..Default::default()
        }
        .validate(8)
        .is_err()
    );
    assert!(
        NemotronHSpeculativeConfig {
            max_draft_tokens: usize::MAX,
            ..Default::default()
        }
        .validate(8)
        .is_err()
    );
}

/// Probability conversion accepts negative infinity but rejects invalid mass.
#[test]
fn probability_validation_rejects_non_finite_distributions() {
    let device = Default::default();
    assert_eq!(
        normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 0.0, f32::NEG_INFINITY]],
            &device,
        ))
        .unwrap(),
        [0.5, 0.5, 0.0]
    );
    assert!(
        normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
            [[f32::NAN, 0.0]],
            &device,
        ))
        .is_err()
    );
    assert!(
        normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
            [[f32::INFINITY, 0.0]],
            &device,
        ))
        .is_err()
    );
    assert!(
        normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
            [[f32::NEG_INFINITY, f32::NEG_INFINITY]],
            &device,
        ))
        .is_err()
    );
}

/// Greedy verification covers rejected, partial, and fully accepted prefixes.
#[test]
fn greedy_acceptance_covers_every_prefix_length() {
    let device = Default::default();
    for accepted_prefix in 0..=2 {
        let mut verifier = NemotronHSpeculativeDecoder::<TestBackend>::new(
            NemotronHSpeculativeConfig::default(),
            3,
            &device,
        )
        .unwrap();
        for index in 0..2 {
            let proposal = 1;
            let p = if index < accepted_prefix {
                [0.1, 0.8, 0.1]
            } else {
                [0.8, 0.1, 0.1]
            };
            assert_eq!(
                verifier.accept(proposal, &p, &p).unwrap(),
                index < accepted_prefix
            );
            if index == accepted_prefix {
                break;
            }
        }
    }
}

/// Inverse-CDF sampling never leaves positive residual support.
#[test]
fn residual_sampling_respects_support_and_validates_weights() {
    for uniform in [0.0, 0.1, 0.499, 0.999_999] {
        let token = sample_weights(&[0.0, 0.4, 0.0, 0.6], uniform).unwrap();
        assert!(token == 1 || token == 3);
    }
    assert!(sample_weights(&[0.0, 0.0], 0.5).is_err());
    assert!(sample_weights(&[f64::NAN, 1.0], 0.5).is_err());
    assert!(sample_weights(&[1.0, -0.1], 0.0).is_err());
}

/// The verifier stream is reproducible and strictly inside the unit interval.
#[test]
fn verifier_rng_is_reproducible() {
    let mut left = SplitMix64::new(19);
    let mut right = SplitMix64::new(19);
    for _ in 0..128 {
        let value = left.next_unit();
        assert_eq!(value, right.next_unit());
        assert!((0.0..1.0).contains(&value));
    }
}

/// Fully accepted drafts commit catch-up and preserve target=MTP+1.
#[test]
fn all_accepted_cycle_commits_catch_up() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    let mut decoder = NemotronHSpeculativeDecoder::new(
        NemotronHSpeculativeConfig::default(),
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    let output = decoder
        .speculate(&model, 1, hidden, &[2, 3], &mut state)
        .unwrap();
    assert_eq!(output.token_ids, [1, 0, 0, 0]);
    assert_eq!(output.drafted_tokens, 2);
    assert_eq!(output.accepted_draft_tokens, 2);
    assert!(output.all_drafts_accepted);
    assert!(!output.terminated);
    assert!(output.continuation_hidden_state.is_some());
    assert_eq!(state.tensor_state().lengths(), &[5]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([4].as_slice()));
}

/// Real model cycles commit zero, one, and two accepted draft prefixes.
#[test]
fn complete_cycles_cover_every_acceptance_prefix() {
    let device = Default::default();
    let model = scripted_rejection_model(&device);
    for (lead, accepted, emitted) in [(0, 0, vec![0, 1]), (1, 1, vec![1, 0, 1])] {
        let (mut state, hidden) = aligned_state(&model, &device);
        let mut decoder = NemotronHSpeculativeDecoder::new(
            NemotronHSpeculativeConfig::default(),
            model.config().vocab_size,
            &device,
        )
        .unwrap();
        let output = decoder
            .speculate(&model, lead, hidden, &[2, 3], &mut state)
            .unwrap();
        assert_eq!(output.token_ids, emitted);
        assert_eq!(output.accepted_draft_tokens, accepted);
        assert!(!output.all_drafts_accepted);
        assert_eq!(state.tensor_state().lengths(), &[3 + accepted]);
        assert_eq!(
            state.tensor_state().mtp_lengths(),
            Some([2 + accepted].as_slice())
        );
    }
}

/// EOS at the lead or either draft boundary terminates without cursor drift.
#[test]
fn eos_placement_preserves_shifted_cache_alignment() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    let mut lead_decoder = NemotronHSpeculativeDecoder::new(
        NemotronHSpeculativeConfig {
            generation: GenerationConfig {
                eos_token_ids: vec![1],
                ..Default::default()
            },
            ..Default::default()
        },
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    let lead = lead_decoder
        .speculate(&model, 1, hidden.clone(), &[2, 3], &mut state)
        .unwrap();
    assert_eq!(lead.token_ids, [1]);
    assert!(lead.terminated);
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));

    let mut first_decoder = NemotronHSpeculativeDecoder::new(
        NemotronHSpeculativeConfig {
            generation: GenerationConfig {
                eos_token_ids: vec![0],
                ..Default::default()
            },
            ..Default::default()
        },
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    let first = first_decoder
        .speculate(&model, 1, hidden, &[2, 3], &mut state)
        .unwrap();
    assert_eq!(first.token_ids, [1, 0]);
    assert_eq!(first.accepted_draft_tokens, 1);
    assert!(first.terminated);
    assert_eq!(state.tensor_state().lengths(), &[3]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));

    let (mut state, hidden) = aligned_state(&model, &device);
    let mut second_decoder = NemotronHSpeculativeDecoder::new(
        NemotronHSpeculativeConfig {
            generation: GenerationConfig {
                presence_penalty: 1.0,
                eos_token_ids: vec![4],
                ..Default::default()
            },
            ..Default::default()
        },
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    let second = second_decoder
        .speculate(&model, 1, hidden, &[2, 3], &mut state)
        .unwrap();
    assert_eq!(second.token_ids, [1, 0, 4]);
    assert_eq!(second.accepted_draft_tokens, 2);
    assert!(second.terminated);
    assert_eq!(state.tensor_state().lengths(), &[4]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([3].as_slice()));
}

/// EOS cannot bypass the ownership check for a caller-open transaction.
#[test]
fn eos_lead_rejects_a_caller_owned_active_transaction() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    let mark = state.mark().unwrap();
    let mut decoder = NemotronHSpeculativeDecoder::new(
        NemotronHSpeculativeConfig {
            generation: GenerationConfig {
                eos_token_ids: vec![1],
                ..Default::default()
            },
            ..Default::default()
        },
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    assert!(matches!(
        decoder.speculate(&model, 1, hidden, &[2, 3], &mut state),
        Err(Error::TensorCache(
            crate::cache::TensorCacheError::TransactionAlreadyActive
        ))
    ));
    assert!(state.tensor_state().has_active_transaction());
    state.rollback(mark).unwrap();
}

/// Inactive rows and post-stage failures are exact retryable no-ops.
#[test]
fn inactive_and_recoverable_failure_restore_state_and_rng() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    let mut decoder = NemotronHSpeculativeDecoder::new(
        NemotronHSpeculativeConfig::default(),
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    let inactive = decoder
        .speculate_if_active(&model, 1, hidden.clone(), &[2, 3], false, &mut state)
        .unwrap();
    assert!(inactive.token_ids.is_empty());
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));

    decoder.draft = LogitsProcessor::new(
        decoder.config.generation.clone(),
        model.config().vocab_size - 1,
        &device,
    )
    .unwrap();
    let first = decoder
        .speculate(&model, 1, hidden.clone(), &[2, 3], &mut state)
        .unwrap_err();
    assert!(matches!(first, Error::Shape { .. }));
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));
    let second = decoder
        .speculate(&model, 1, hidden, &[2, 3], &mut state)
        .unwrap_err();
    assert_eq!(first.to_string(), second.to_string());
}

/// Cycle preflight rejects vocabulary, history, token, hidden, and capacity drift.
#[test]
fn cycle_preflight_rejects_all_alignment_mismatches() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    let mut wrong_vocabulary = NemotronHSpeculativeDecoder::new(
        NemotronHSpeculativeConfig::default(),
        model.config().vocab_size - 1,
        &device,
    )
    .unwrap();
    assert!(
        wrong_vocabulary
            .speculate(&model, 1, hidden.clone(), &[2, 3], &mut state)
            .is_err()
    );

    let mut decoder = NemotronHSpeculativeDecoder::new(
        NemotronHSpeculativeConfig::default(),
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    assert!(
        decoder
            .speculate(&model, 1, hidden.clone(), &[2], &mut state)
            .is_err()
    );
    assert!(
        decoder
            .speculate(
                &model,
                model.config().vocab_size as u32,
                hidden.clone(),
                &[2, 3],
                &mut state,
            )
            .is_err()
    );
    assert!(
        decoder
            .speculate(
                &model,
                1,
                Tensor::zeros([1, 2, model.config().hidden_size], &device),
                &[2, 3],
                &mut state,
            )
            .is_err()
    );
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));
}

/// A physically rewindable graph clips MTP to the surviving shifted prefix.
#[test]
fn trait_truncate_rewinds_target_and_mtp_to_independent_lengths() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, _) = aligned_state(&model, &device);
    crate::architecture::DecodeState::<TestBackend>::truncate(&mut state, &[1]).unwrap();
    assert_eq!(state.tensor_state().lengths(), &[1]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([0].as_slice()));
}

/// Monte Carlo rejection sampling recovers the target distribution.
#[test]
fn stochastic_rejection_recovers_target_marginal() {
    let p = [0.1, 0.2, 0.7];
    let q = [0.6, 0.3, 0.1];
    let mut proposal_rng = SplitMix64::new(71);
    let mut verifier_rng = SplitMix64::new(72);
    let mut counts = [0usize; 3];
    let trials = 100_000usize;
    for _ in 0..trials {
        let proposal = sample_weights(&q, proposal_rng.next_unit()).unwrap();
        let ratio = (p[proposal as usize] / q[proposal as usize]).min(1.0);
        let token = if verifier_rng.next_unit() < ratio {
            proposal
        } else {
            let residual = p
                .iter()
                .zip(q)
                .map(|(&target, draft)| (target - draft).max(0.0))
                .collect::<Vec<_>>();
            sample_weights(&residual, verifier_rng.next_unit()).unwrap()
        };
        counts[token as usize] += 1;
    }
    for (observed, expected) in counts.into_iter().zip(p) {
        assert!((observed as f64 / trials as f64 - expected).abs() < 0.01);
    }
}

/// Equal seeds produce equal sampled cycles and cache cursors.
#[test]
fn sampled_cycles_are_reproducible() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut left_state, left_hidden) = aligned_state(&model, &device);
    let (mut right_state, right_hidden) = aligned_state(&model, &device);
    let config = NemotronHSpeculativeConfig {
        generation: GenerationConfig {
            strategy: SamplingStrategy::Sample,
            seed: 0x5eed,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut left =
        NemotronHSpeculativeDecoder::new(config.clone(), model.config().vocab_size, &device)
            .unwrap();
    let mut right =
        NemotronHSpeculativeDecoder::new(config, model.config().vocab_size, &device).unwrap();
    let left = left
        .speculate(&model, 1, left_hidden, &[2, 3], &mut left_state)
        .unwrap();
    let right = right
        .speculate(&model, 1, right_hidden, &[2, 3], &mut right_state)
        .unwrap();
    assert_eq!(left.token_ids, right.token_ids);
    assert_eq!(left.accepted_draft_tokens, right.accepted_draft_tokens);
    assert_eq!(
        left_state.tensor_state().lengths(),
        right_state.tensor_state().lengths()
    );
    assert_eq!(
        left_state.tensor_state().mtp_lengths(),
        right_state.tensor_state().mtp_lengths()
    );
}

/// A failure after draft sampling restores the proposal stream for retry.
#[test]
fn sampled_failure_after_rng_draw_retries_bit_for_bit() {
    let device = Default::default();
    let model = zero_model(&device);
    let config = NemotronHSpeculativeConfig {
        generation: GenerationConfig {
            strategy: SamplingStrategy::Sample,
            seed: 0x5eed_cafe,
            ..Default::default()
        },
        ..Default::default()
    };

    let (mut expected_state, expected_hidden) = aligned_state(&model, &device);
    let mut expected_decoder =
        NemotronHSpeculativeDecoder::new(config.clone(), model.config().vocab_size, &device)
            .unwrap();
    let expected = expected_decoder
        .speculate(&model, 1, expected_hidden, &[2, 3], &mut expected_state)
        .unwrap();

    let (mut retry_state, retry_hidden) = aligned_state(&model, &device);
    let mut retry_decoder =
        NemotronHSpeculativeDecoder::new(config.clone(), model.config().vocab_size, &device)
            .unwrap();
    retry_decoder.target = LogitsProcessor::new(
        config.generation.clone(),
        model.config().vocab_size - 1,
        &device,
    )
    .unwrap();
    assert!(matches!(
        retry_decoder.speculate(&model, 1, retry_hidden.clone(), &[2, 3], &mut retry_state,),
        Err(Error::Shape { .. })
    ));
    assert_eq!(retry_state.tensor_state().lengths(), &[2]);
    assert_eq!(
        retry_state.tensor_state().mtp_lengths(),
        Some([1].as_slice())
    );

    retry_decoder.target =
        LogitsProcessor::new(config.generation, model.config().vocab_size, &device).unwrap();
    let retried = retry_decoder
        .speculate(&model, 1, retry_hidden, &[2, 3], &mut retry_state)
        .unwrap();
    assert_eq!(retried.token_ids, expected.token_ids);
    assert_eq!(
        retried.accepted_draft_tokens,
        expected.accepted_draft_tokens
    );
    assert_eq!(retried.all_drafts_accepted, expected.all_drafts_accepted);
    assert_eq!(
        retry_state.tensor_state().lengths(),
        expected_state.tensor_state().lengths()
    );
    assert_eq!(
        retry_state.tensor_state().mtp_lengths(),
        expected_state.tensor_state().mtp_lengths()
    );
}
