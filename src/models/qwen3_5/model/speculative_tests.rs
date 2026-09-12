//! Qwen MTP acceptance, sampling, cache alignment, and transaction-recovery tests.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData, backend::BackendTypes},
};

use super::*;
use crate::{
    CausalLmInput,
    cache::CacheConfig,
    checkpoint::TensorSink,
    models::qwen3_5::{
        Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5LayerType, Qwen3_5TextConfig,
    },
};

/// CPU Flex backend with F32 activations and portable I32 indices.
type TestBackend = Flex<f32, i32>;

/// Creates a zero-logit graph whose greedy target and draft both select ID zero.
fn zero_model(device: &<TestBackend as BackendTypes>::Device) -> Qwen3_5ForCausalLm<TestBackend> {
    let config = Qwen3_5TextConfig {
        vocab_size: 8,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        layer_types: vec![Qwen3_5LayerType::FullAttention],
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_conv_kernel_dim: 3,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        mtp_num_hidden_layers: 1,
        tie_word_embeddings: true,
        ..Default::default()
    };
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: true,
        })
        .unwrap();
    let mut model = Qwen3_5ForCausalLm::init(config, device).unwrap();
    for expected in contract.expected_tensors() {
        if expected.name == "lm_head.weight" {
            continue;
        }
        let shape = expected
            .shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap())
            .collect::<Vec<_>>();
        model
            .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
            .unwrap();
    }
    model
        .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap();
    model
}

/// Creates a graph whose MTP head proposes zero while target transitions
/// `1 -> 0` and `0 -> 1`, producing controlled acceptance prefixes.
fn scripted_rejection_model(
    device: &<TestBackend as BackendTypes>::Device,
) -> Qwen3_5ForCausalLm<TestBackend> {
    let config = Qwen3_5TextConfig {
        vocab_size: 8,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        layer_types: vec![Qwen3_5LayerType::FullAttention],
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_conv_kernel_dim: 3,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        mtp_num_hidden_layers: 1,
        tie_word_embeddings: false,
        ..Default::default()
    };
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: true,
        })
        .unwrap();
    let mut model = Qwen3_5ForCausalLm::init(config, device).unwrap();
    for expected in contract.expected_tensors() {
        let shape = expected
            .shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap())
            .collect::<Vec<_>>();
        model
            .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
            .unwrap();
    }

    let mut embeddings = vec![0.0_f32; 8 * 8];
    embeddings[0] = 1.0;
    embeddings[8 + 1] = 1.0;
    model
        .write_tensor(
            "model.embed_tokens.weight",
            TensorData::new(embeddings, [8, 8]),
        )
        .unwrap();
    let mut head = vec![0.0_f32; 8 * 8];
    head[1] = 1.0; // hidden for token 1 predicts token 0
    head[8] = 1.0; // hidden for token 0 predicts token 1
    model
        .write_tensor("lm_head.weight", TensorData::new(head, [8, 8]))
        .unwrap();
    model
}

/// Prefills target and shifted MTP caches and returns the last target hidden state.
fn aligned_state(
    model: &Qwen3_5ForCausalLm<TestBackend>,
    device: &<TestBackend as BackendTypes>::Device,
) -> (Qwen3_5DecodeState<TestBackend>, Tensor<TestBackend, 3>) {
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

/// The default proposes two tokens, while a zero-length speculative proposal is invalid.
#[test]
fn defaults_to_two_drafts_and_rejects_zero() {
    assert_eq!(Qwen3_5SpeculativeConfig::default().max_draft_tokens, 2);
    let config = Qwen3_5SpeculativeConfig {
        max_draft_tokens: 0,
        ..Default::default()
    };
    assert!(config.validate(8).is_err());
}

/// Softmax preserves zero mass at negative infinity and rejects positive infinity or no finite mass.
#[test]
fn probability_normalization_matches_backend_softmax_and_rejects_non_finite_logits() {
    let device = Default::default();
    let finite = normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
        [[0.0, 0.0, f32::NEG_INFINITY]],
        &device,
    ))
    .unwrap();
    assert_eq!(finite, vec![0.5, 0.5, 0.0]);
    let infinite = normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
        [[f32::INFINITY, 1.0, f32::INFINITY]],
        &device,
    ));
    assert!(infinite.is_err());
    let empty_mass = normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
        [[f32::NEG_INFINITY, f32::NEG_INFINITY]],
        &device,
    ));
    assert!(empty_mass.is_err());
}

/// Greedy verification accepts exactly the matching zero-, one-, or two-token proposal prefix.
#[test]
fn greedy_acceptance_covers_every_prefix_length() {
    for accepted_prefix in 0..=2 {
        let device = Default::default();
        let mut verifier = Qwen3_5SpeculativeDecoder::<TestBackend>::new(
            Qwen3_5SpeculativeConfig::default(),
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

/// Inverse-CDF samples stay on positive weights, and an all-zero residual is rejected.
#[test]
fn residual_sampling_never_returns_non_positive_support() {
    for uniform in [0.0, 0.1, 0.499, 0.999_999] {
        let token = sample_weights(&[0.0, 0.4, 0.0, 0.6], uniform).unwrap();
        assert!(token == 1 || token == 3);
    }
    assert!(sample_weights(&[0.0, 0.0], 0.5).is_err());
}

/// Equal SplitMix seeds yield equal draws in the half-open unit interval.
#[test]
fn splitmix_stream_is_reproducible_and_strictly_unit_interval() {
    let mut left = SplitMix64::new(19);
    let mut right = SplitMix64::new(19);
    for _ in 0..128 {
        let lhs = left.next_unit();
        assert_eq!(lhs, right.next_unit());
        assert!((0.0..1.0).contains(&lhs));
    }
}

/// Accepting both drafts emits a bonus and leaves target and MTP cursors one token apart.
#[test]
fn all_accepted_cycle_commits_catch_up_and_preserves_shift() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    let mut decoder = Qwen3_5SpeculativeDecoder::new(
        Qwen3_5SpeculativeConfig::default(),
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    let output = decoder
        .speculate(&model, 1, hidden, &[2, 3], &mut state)
        .unwrap();
    assert_eq!(output.token_ids, vec![1, 0, 0, 0]);
    assert_eq!(output.drafted_tokens, 2);
    assert_eq!(output.accepted_draft_tokens, 2);
    assert!(output.all_drafts_accepted);
    assert!(!output.terminated);
    assert!(output.continuation_hidden_state.is_some());
    assert_eq!(state.tensor_state().lengths(), &[5]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([4].as_slice()));
}

/// Controlled zero- and one-draft acceptance retains exactly the verified cache prefix.
#[test]
fn complete_cycles_commit_every_acceptance_prefix() {
    let device = Default::default();
    let model = scripted_rejection_model(&device);
    for (lead, accepted, emitted) in [(0, 0, vec![0, 1]), (1, 1, vec![1, 0, 1])] {
        let (mut state, hidden) = aligned_state(&model, &device);
        let mut decoder = Qwen3_5SpeculativeDecoder::new(
            Qwen3_5SpeculativeConfig::default(),
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

/// EOS at the lead or either draft position ends emission with the expected shifted cursors.
#[test]
fn eos_lead_and_eos_draft_stop_without_cache_drift() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    let config = Qwen3_5SpeculativeConfig {
        generation: GenerationConfig {
            eos_token_ids: vec![1],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut decoder =
        Qwen3_5SpeculativeDecoder::new(config, model.config().vocab_size, &device).unwrap();
    let lead = decoder
        .speculate(&model, 1, hidden.clone(), &[2, 3], &mut state)
        .unwrap();
    assert_eq!(lead.token_ids, vec![1]);
    assert!(lead.terminated);
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));

    let config = Qwen3_5SpeculativeConfig {
        generation: GenerationConfig {
            eos_token_ids: vec![0],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut decoder =
        Qwen3_5SpeculativeDecoder::new(config, model.config().vocab_size, &device).unwrap();
    let draft = decoder
        .speculate(&model, 1, hidden, &[2, 3], &mut state)
        .unwrap();
    assert_eq!(draft.token_ids, vec![1, 0]);
    assert_eq!(draft.accepted_draft_tokens, 1);
    assert!(draft.terminated);
    assert_eq!(state.tensor_state().lengths(), &[3]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));

    let (mut state, hidden) = aligned_state(&model, &device);
    let config = Qwen3_5SpeculativeConfig {
        generation: GenerationConfig {
            presence_penalty: 1.0,
            eos_token_ids: vec![4],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut decoder =
        Qwen3_5SpeculativeDecoder::new(config, model.config().vocab_size, &device).unwrap();
    let second_draft = decoder
        .speculate(&model, 1, hidden, &[2, 3], &mut state)
        .unwrap();
    assert_eq!(second_draft.token_ids, vec![1, 0, 4]);
    assert_eq!(second_draft.accepted_draft_tokens, 2);
    assert!(second_draft.terminated);
    assert_eq!(state.tensor_state().lengths(), &[4]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([3].as_slice()));
}

/// Even an EOS lead refuses an existing caller transaction without consuming its mark.
#[test]
fn eos_lead_rejects_a_caller_owned_active_transaction() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    let mark = state.mark().unwrap();
    let mut decoder = Qwen3_5SpeculativeDecoder::new(
        Qwen3_5SpeculativeConfig {
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

/// Inactive cycles and a staged-draft failure preserve both cursors and a repeatable retry error.
#[test]
fn inactive_and_recoverable_failure_leave_both_cursors_unchanged() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    let mut decoder = Qwen3_5SpeculativeDecoder::new(
        Qwen3_5SpeculativeConfig::default(),
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

    // Deliberately replace the internal draft processor with one for a
    // smaller vocabulary. Public construction cannot create this mismatch;
    // the test-only corruption forces an error after an MTP KV proposal was
    // staged and therefore exercises the recovery path itself.
    let mut wrong = Qwen3_5SpeculativeDecoder::new(
        Qwen3_5SpeculativeConfig::default(),
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    wrong.draft = LogitsProcessor::new(
        wrong.config.generation.clone(),
        model.config().vocab_size - 1,
        &device,
    )
    .unwrap();
    let first = wrong
        .speculate(&model, 1, hidden.clone(), &[2, 3], &mut state)
        .unwrap_err();
    assert!(matches!(first, Error::Shape { .. }));
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));
    let second = wrong
        .speculate(&model, 1, hidden, &[2, 3], &mut state)
        .unwrap_err();
    assert_eq!(first.to_string(), second.to_string());
}

/// State creation rejects multiple rows, insufficient proposal capacity, and missing MTP layers.
#[test]
fn speculative_state_rejects_batch_capacity_and_missing_mtp() {
    let device = Default::default();
    let model = zero_model(&device);
    assert!(
        model
            .new_speculative_decode_state(CacheConfig::new(2, 8).unwrap(), 2, 2, &device)
            .is_err()
    );
    assert!(
        model
            .new_speculative_decode_state(CacheConfig::new(1, 2).unwrap(), 1, 2, &device)
            .is_err()
    );
    let mut config = model.config().clone();
    config.mtp_num_hidden_layers = 0;
    let base = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    assert!(
        base.new_speculative_decode_state(CacheConfig::new(1, 8).unwrap(), 1, 2, &device)
            .is_err()
    );
}

/// Invalid vocabulary, history, lead ID, or hidden shape fails before either cache cursor changes.
#[test]
fn cycle_preflight_rejects_vocabulary_history_token_and_hidden_mismatches() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    let mut wrong_vocabulary = Qwen3_5SpeculativeDecoder::new(
        Qwen3_5SpeculativeConfig::default(),
        model.config().vocab_size - 1,
        &device,
    )
    .unwrap();
    assert!(
        wrong_vocabulary
            .speculate(&model, 1, hidden.clone(), &[2, 3], &mut state)
            .is_err()
    );

    let mut decoder = Qwen3_5SpeculativeDecoder::new(
        Qwen3_5SpeculativeConfig::default(),
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

/// Truncating both shifted caches to zero permits a fresh target and MTP prefill.
#[test]
fn speculative_state_zero_truncate_rewinds_both_shifted_caches() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, _) = aligned_state(&model, &device);

    crate::architecture::DecodeState::<TestBackend>::truncate(&mut state, &[2]).unwrap();
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));
    crate::architecture::DecodeState::<TestBackend>::truncate(&mut state, &[0]).unwrap();
    assert_eq!(state.tensor_state().lengths(), &[0]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([0].as_slice()));

    let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[2, 3]], &device);
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
    model.prefill_mtp_cache(prompt, hidden, &mut state).unwrap();
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));
}

/// A seeded 100,000-draw rejection experiment matches each target mass within 0.01.
#[test]
fn stochastic_rejection_rule_recovers_target_marginal() {
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
        let frequency = observed as f64 / trials as f64;
        assert!(
            (frequency - expected).abs() < 0.01,
            "{frequency} != {expected}"
        );
    }
}

/// Equal sampling seeds produce identical speculative tokens and acceptance diagnostics.
#[test]
fn stochastic_cycles_with_equal_seeds_are_reproducible() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut first_state, first_hidden) = aligned_state(&model, &device);
    let (mut second_state, second_hidden) = aligned_state(&model, &device);
    let config = Qwen3_5SpeculativeConfig {
        generation: GenerationConfig {
            strategy: SamplingStrategy::Sample,
            seed: 0x5eed,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut first =
        Qwen3_5SpeculativeDecoder::new(config.clone(), model.config().vocab_size, &device).unwrap();
    let mut second =
        Qwen3_5SpeculativeDecoder::new(config, model.config().vocab_size, &device).unwrap();
    let first = first
        .speculate(&model, 1, first_hidden, &[2, 3], &mut first_state)
        .unwrap();
    let second = second
        .speculate(&model, 1, second_hidden, &[2, 3], &mut second_state)
        .unwrap();
    assert_eq!(first.token_ids, second.token_ids);
    assert_eq!(first.accepted_draft_tokens, 2);
    assert_eq!(
        first_state.tensor_state().lengths(),
        second_state.tensor_state().lengths()
    );
    assert_eq!(
        first_state.tensor_state().mtp_lengths(),
        second_state.tensor_state().mtp_lengths()
    );
}
