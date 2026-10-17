//! Qwen MTP acceptance, sampling, cache alignment, and transaction-recovery tests.

use burn::{
    backend::Flex,
    tensor::{
        Int, Tensor, TensorData, Tolerance,
        backend::{Backend, BackendTypes},
    },
};

use super::*;
use crate::{
    CausalLanguageModel, CausalLmInput,
    cache::CacheConfig,
    checkpoint::TensorSink,
    models::qwen3_5::{
        Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5LayerType, Qwen3_5TextConfig,
    },
    models::shared::test_support::deterministic_parameters,
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

/// Inverse-CDF samples stay on positive weights, and an all-zero residual is rejected.
#[test]
fn residual_sampling_never_returns_non_positive_support() {
    for uniform in [0.0, 0.1, 0.499, 0.999_999] {
        let token = sample_weights(&[0.0, 0.4, 0.0, 0.6], uniform).unwrap();
        assert!(token == 1 || token == 3);
    }
    assert!(sample_weights(&[0.0, 0.0], 0.5).is_err());
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
        .speculate(&model, 1, hidden, &[2, 3], 2, &mut state)
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
            .speculate(&model, lead, hidden, &[2, 3], 2, &mut state)
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
        .speculate(&model, 1, hidden.clone(), &[2, 3], 2, &mut state)
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
        .speculate(&model, 1, hidden, &[2, 3], 2, &mut state)
        .unwrap();
    assert_eq!(draft.token_ids, vec![1, 0]);
    assert_eq!(draft.accepted_draft_tokens, 1);
    assert!(draft.terminated);
    assert_eq!(state.tensor_state().lengths(), &[3]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));

    // Presence penalties count generated tokens only: after the lead 1 and
    // draft 0 are penalized, the zero logits pick 2, not a prompt token's
    // successor. (Counting the prompt [2, 3] as well would pick 4.)
    let (mut state, hidden) = aligned_state(&model, &device);
    let config = Qwen3_5SpeculativeConfig {
        generation: GenerationConfig {
            presence_penalty: 1.0,
            eos_token_ids: vec![2],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut decoder =
        Qwen3_5SpeculativeDecoder::new(config, model.config().vocab_size, &device).unwrap();
    let second_draft = decoder
        .speculate(&model, 1, hidden, &[2, 3], 2, &mut state)
        .unwrap();
    assert_eq!(second_draft.token_ids, vec![1, 0, 2]);
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
        decoder.speculate(&model, 1, hidden, &[2, 3], 2, &mut state),
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
        .speculate_if_active(&model, 1, hidden.clone(), &[2, 3], 2, false, &mut state)
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
        .speculate(&model, 1, hidden.clone(), &[2, 3], 2, &mut state)
        .unwrap_err();
    assert!(matches!(first, Error::Shape { .. }));
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));
    let second = wrong
        .speculate(&model, 1, hidden, &[2, 3], 2, &mut state)
        .unwrap_err();
    assert_eq!(first.to_string(), second.to_string());
}

/// A commit that fails validation rolls the staged cycle back instead of stranding the transaction.
///
/// `commit_prefixes` consumes its mark and leaves the transaction open when
/// validation fails, so a cycle that called it directly could neither retry
/// nor reset afterwards. The cycle's commit helper keeps a recovery mark.
#[test]
fn failed_commit_rolls_back_the_staged_cycle() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    let stage = |state: &mut Qwen3_5DecodeState<TestBackend>| {
        let mark = state.mark().unwrap();
        model
            .forward_mtp_cached(
                Tensor::<TestBackend, 2, Int>::from_ints([[1]], &device),
                hidden.clone(),
                None,
                0,
                state,
            )
            .unwrap();
        model
            .decode(
                crate::DecodeInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                state,
                &ForwardOptions::default(),
            )
            .unwrap();
        assert_eq!(state.tensor_state().working_lengths(), &[4]);
        mark
    };

    // The raw commit rejects over-acceptance but leaves the transaction open.
    let mark = stage(&mut state);
    let recovery = mark.clone();
    assert!(state.commit_prefixes(mark, 3, 1).is_err());
    assert!(state.tensor_state().has_active_transaction());
    state.rollback(recovery).unwrap();

    // The cycle helper reports the same error with no open transaction.
    let mark = stage(&mut state);
    let error = commit_or_roll_back(&mut state, mark, 3, 1).unwrap_err();
    assert!(matches!(error, Error::TensorCache(_)), "{error}");
    assert!(!state.tensor_state().has_active_transaction());
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));

    // The state stays usable: a later valid commit succeeds.
    let mark = stage(&mut state);
    commit_or_roll_back(&mut state, mark, 2, 1).unwrap();
    assert!(!state.tensor_state().has_active_transaction());
    assert_eq!(state.tensor_state().lengths(), &[4]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));
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
            .speculate(&model, 1, hidden.clone(), &[2, 3], 2, &mut state)
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
            .speculate(&model, 1, hidden.clone(), &[2], 1, &mut state)
            .is_err()
    );
    assert!(
        decoder
            .speculate(
                &model,
                model.config().vocab_size as u32,
                hidden.clone(),
                &[2, 3],
                2,
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
                2,
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
        let proposal = sample_weights(&q, proposal_rng.next_unit_f64()).unwrap();
        let ratio = (p[proposal as usize] / q[proposal as usize]).min(1.0);
        let token = if verifier_rng.next_unit_f64() < ratio {
            proposal
        } else {
            let residual = p
                .iter()
                .zip(q)
                .map(|(&target, draft)| (target - draft).max(0.0))
                .collect::<Vec<_>>();
            sample_weights(&residual, verifier_rng.next_unit_f64()).unwrap()
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
        .speculate(&model, 1, first_hidden, &[2, 3], 2, &mut first_state)
        .unwrap();
    let second = second
        .speculate(&model, 1, second_hidden, &[2, 3], 2, &mut second_state)
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

/// Builds a randomly initialized hybrid model with MTP and a tiny vocabulary.
///
/// Six tokens make the random MTP head agree with the target often enough
/// that cycles exercise rejections, partial acceptance, and full acceptance.
/// The weights come from `seed` alone ([`deterministic_parameters`]), not
/// from Burn's process-wide generator, which tests running in parallel
/// share: a drafter that accepts no drafts is common among random models, so
/// weights that depended on test scheduling made assertions about acceptance
/// flaky.
fn random_hybrid_model(
    seed: u64,
    device: &<TestBackend as BackendTypes>::Device,
) -> Qwen3_5ForCausalLm<TestBackend> {
    let model = Qwen3_5ForCausalLm::init(
        Qwen3_5TextConfig {
            vocab_size: 6,
            hidden_size: 16,
            intermediate_size: 24,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            max_position_embeddings: 96,
            layer_types: vec![
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::FullAttention,
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::FullAttention,
            ],
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 1,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 3,
            partial_rotary_factor: Some(0.5),
            rope_theta: Some(10_000.0),
            mtp_num_hidden_layers: 1,
            ..Default::default()
        },
        device,
    )
    .unwrap();
    deterministic_parameters(model, seed)
}

/// Returns the first greatest logit of a `[1, 1, V]` tensor.
fn argmax_token(logits: Tensor<TestBackend, 3>) -> u32 {
    let values = logits.into_data().to_vec::<f32>().unwrap();
    let mut best = 0;
    for (index, &value) in values.iter().enumerate() {
        if value > values[best] {
            best = index;
        }
    }
    best as u32
}

/// Plain greedy decoding: prefill, then one cached decode call per token.
fn plain_greedy(model: &Qwen3_5ForCausalLm<TestBackend>, prompt: &[u32], count: usize) -> Vec<u32> {
    let device = Default::default();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 64).unwrap(), 1, &device)
        .unwrap();
    let mut logits = model
        .prefill(
            CausalLmInput {
                token_ids: token_tensor(prompt, &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let mut tokens = Vec::with_capacity(count);
    while tokens.len() < count {
        let token = argmax_token(logits);
        tokens.push(token);
        logits = model
            .decode(
                crate::DecodeInput {
                    token_ids: token_tensor(&[token], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
    }
    tokens
}

/// Result of a speculative generation run.
struct SpeculativeRun {
    /// Emitted tokens, each emitted once.
    tokens: Vec<u32>,
    /// Drafts accepted across all cycles.
    accepted: usize,
    /// Drafts proposed across all cycles.
    drafted: usize,
}

/// Prefills, then runs greedy speculative cycles until `count` tokens are emitted.
fn speculative_greedy(
    model: &Qwen3_5ForCausalLm<TestBackend>,
    prompt: &[u32],
    count: usize,
    config: Qwen3_5SpeculativeConfig,
    state: &mut Qwen3_5DecodeState<TestBackend>,
) -> SpeculativeRun {
    let device = Default::default();
    let prefill = model
        .prefill(
            CausalLmInput {
                token_ids: token_tensor(prompt, &device),
                token_mask: None,
                position_ids: None,
            },
            state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap();
    let hidden = prefill.final_hidden_state.unwrap();
    model
        .prefill_mtp_cache(token_tensor(prompt, &device), hidden.clone(), state)
        .unwrap();
    let mut decoder =
        Qwen3_5SpeculativeDecoder::new(config, model.config().vocab_size, &device).unwrap();
    let length = prompt.len();
    let mut lead = decoder
        .select_lead(
            prefill.logits.narrow(1, length - 1, 1).squeeze_dim::<2>(1),
            prompt,
            length,
        )
        .unwrap();
    let mut hidden = hidden.narrow(1, length - 1, 1);
    let mut history = prompt.to_vec();
    let mut run = SpeculativeRun {
        tokens: vec![lead],
        accepted: 0,
        drafted: 0,
    };
    while run.tokens.len() < count {
        let output = decoder
            .speculate(model, lead, hidden, &history, length, state)
            .unwrap();
        assert_eq!(output.token_ids[0], lead);
        assert!(!output.terminated);
        run.accepted += output.accepted_draft_tokens;
        run.drafted += output.drafted_tokens;
        // The committed prefix is everything but the next lead.
        history.extend(&output.token_ids[..output.token_ids.len() - 1]);
        run.tokens.extend(&output.token_ids[1..]);
        lead = *output.token_ids.last().unwrap();
        hidden = output.continuation_hidden_state.unwrap();
        let lengths = state.tensor_state();
        assert_eq!(lengths.lengths(), &[history.len()]);
        assert_eq!(lengths.mtp_lengths(), Some([history.len() - 1].as_slice()));
    }
    run
}

/// QW-11: greedy speculation emits exactly the plain greedy sequence on random hybrid models.
///
/// Across seeds and draft lengths the cycles see rejections and acceptances;
/// truncating both caches to zero and repeating reproduces the same tokens.
#[test]
fn greedy_speculation_reproduces_plain_greedy_decoding() {
    let device = Default::default();
    let prompt = [1, 4, 2, 5, 3];
    let count = 20;
    let (mut accepted, mut drafted) = (0, 0);
    // Each seed and draft length both accepts and rejects drafts (62 of 168
    // accepted in total), so the assertions below hold with a wide margin.
    for seed in [109, 36, 41, 83] {
        let model = random_hybrid_model(seed, &device);
        let reference = plain_greedy(&model, &prompt, count);
        for max_draft_tokens in [1, 3] {
            let config = Qwen3_5SpeculativeConfig {
                max_draft_tokens,
                ..Default::default()
            };
            let mut state = model
                .new_speculative_decode_state(
                    CacheConfig::new(1, 64).unwrap(),
                    1,
                    max_draft_tokens,
                    &device,
                )
                .unwrap();
            let run = speculative_greedy(&model, &prompt, count, config.clone(), &mut state);
            assert_eq!(&run.tokens[..count], reference.as_slice(), "seed {seed}");
            accepted += run.accepted;
            drafted += run.drafted;

            crate::architecture::DecodeState::<TestBackend>::truncate(&mut state, &[0]).unwrap();
            let repeated = speculative_greedy(&model, &prompt, count, config, &mut state);
            assert_eq!(repeated.tokens, run.tokens, "seed {seed} after truncation");
        }
    }
    assert!(accepted > 0, "no draft was ever accepted");
    assert!(accepted < drafted, "no draft was ever rejected");
}

/// QW-5: committed MTP history is built from target hidden states, as in the reference drafter.
///
/// After speculative cycles, every readable MTP cache entry equals the one a
/// fresh MTP prefill writes from the target's hidden states over the same
/// committed tokens.
#[test]
fn committed_mtp_history_comes_from_target_hidden_states() {
    let device = Default::default();
    let prompt = [2, 0, 3, 1];
    let config = Qwen3_5SpeculativeConfig {
        max_draft_tokens: 3,
        ..Default::default()
    };
    // Burn's backend seed is process-global and parameters initialize
    // lazily, so concurrent tests can change the weights; try models until
    // one accepts a draft, which the comparison needs.
    let (model, state, run) = (0..16)
        .find_map(|seed| {
            let model = random_hybrid_model(100 + seed, &device);
            let mut state = model
                .new_speculative_decode_state(CacheConfig::new(1, 64).unwrap(), 1, 3, &device)
                .unwrap();
            let run = speculative_greedy(&model, &prompt, 12, config.clone(), &mut state);
            (run.accepted > 0).then_some((model, state, run))
        })
        .expect("some random model accepts a draft");
    let committed = state.tensor_state().lengths()[0];
    let mut sequence = prompt.to_vec();
    sequence.extend(&run.tokens[..committed - prompt.len()]);

    let mut fresh = model
        .new_speculative_decode_state(CacheConfig::new(1, 64).unwrap(), 1, 3, &device)
        .unwrap();
    let hidden = model
        .prefill(
            CausalLmInput {
                token_ids: token_tensor(&sequence, &device),
                token_mask: None,
                position_ids: None,
            },
            &mut fresh,
            &ForwardOptions {
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap()
        .final_hidden_state
        .unwrap();
    model
        .prefill_mtp_cache(token_tensor(&sequence, &device), hidden, &mut fresh)
        .unwrap();
    let mtp_length = committed - 1;
    assert_eq!(
        fresh.tensor_state().mtp_lengths(),
        Some([mtp_length].as_slice())
    );
    let readable = |state: &Qwen3_5DecodeState<TestBackend>| {
        let cache = state.tensor_state().mtp().unwrap().native_kv(0).unwrap();
        [cache.keys(0).unwrap(), cache.values(0).unwrap()].map(|plane| {
            let [_, heads, _, width] = plane.dims();
            plane
                .clone()
                .slice([0..1, 0..heads, 0..mtp_length, 0..width])
                .into_data()
        })
    };
    for (speculative, reference) in readable(&state).into_iter().zip(readable(&fresh)) {
        speculative.assert_approx_eq::<f32>(&reference, Tolerance::rel_abs(2e-4, 2e-5));
    }
}

/// Seeded stochastic cycles on a random model are reproducible and keep both caches aligned.
#[test]
fn stochastic_cycles_on_a_random_model_are_reproducible_and_aligned() {
    let device = Default::default();
    let prompt = [3, 1, 4];
    let model = random_hybrid_model(41, &device);
    let config = Qwen3_5SpeculativeConfig {
        max_draft_tokens: 2,
        generation: GenerationConfig {
            strategy: SamplingStrategy::Sample,
            temperature: 0.8,
            top_k: Some(4),
            seed: 0x51de,
            ..Default::default()
        },
    };
    let mut runs = Vec::new();
    for _ in 0..2 {
        let mut state = model
            .new_speculative_decode_state(CacheConfig::new(1, 64).unwrap(), 1, 2, &device)
            .unwrap();
        runs.push(speculative_greedy(&model, &prompt, 16, config.clone(), &mut state).tokens);
    }
    assert_eq!(runs[0], runs[1]);
    assert!(runs[0].iter().all(|&token| token < 6));
}

/// Measures verifying `K + 1 = 4` tokens in one target pass against four sequential passes.
///
/// The former verifier decoded the lead and every accepted draft separately;
/// the batched verifier stages all of them in one call. Both run inside a
/// transaction that is rolled back, on a small hybrid model after a 32-token
/// prompt, and report the median of nine runs. Run with
/// `cargo test --release --lib qwen3_5::model::speculative::tests::verification_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing measurement; run explicitly in release mode"]
fn verification_timing() {
    let device = Default::default();
    TestBackend::seed(&device, 9);
    let model = Qwen3_5ForCausalLm::<TestBackend>::init(
        Qwen3_5TextConfig {
            vocab_size: 1024,
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 64,
            max_position_embeddings: 128,
            layer_types: vec![
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::FullAttention,
            ],
            linear_key_head_dim: 64,
            linear_value_head_dim: 64,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_conv_kernel_dim: 4,
            mtp_num_hidden_layers: 1,
            ..Default::default()
        },
        &device,
    )
    .unwrap();
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 64).unwrap(), 1, 3, &device)
        .unwrap();
    let prompt = (0..32).map(|token| token * 7 % 1024).collect::<Vec<u32>>();
    model
        .prefill(
            CausalLmInput {
                token_ids: token_tensor(&prompt, &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let tokens = [5, 9, 13, 17];
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let decode = |state: &mut Qwen3_5DecodeState<TestBackend>, ids: &[u32]| {
        model
            .decode(
                crate::DecodeInput {
                    token_ids: token_tensor(ids, &device),
                    token_mask: None,
                    position_ids: None,
                },
                state,
                &options,
            )
            .unwrap()
            .logits
            .into_data()
    };
    let (mut sequential, mut batched) = (Vec::new(), Vec::new());
    for _ in 0..9 {
        let mark = state.mark().unwrap();
        let start = std::time::Instant::now();
        for token in tokens {
            decode(&mut state, &[token]);
        }
        sequential.push(start.elapsed());
        state.rollback(mark).unwrap();
        let mark = state.mark().unwrap();
        let start = std::time::Instant::now();
        decode(&mut state, &tokens);
        batched.push(start.elapsed());
        state.rollback(mark).unwrap();
    }
    sequential.sort();
    batched.sort();
    println!(
        "verify 4 tokens: one pass {:?}, four passes {:?} ({:.2}x)",
        batched[4],
        sequential[4],
        sequential[4].as_secs_f64() / batched[4].as_secs_f64()
    );
}

/// A proposal count whose `K + 1` staging width overflows is rejected, and controls are exposed.
#[test]
fn unrepresentable_proposal_counts_are_rejected() {
    let config = Qwen3_5SpeculativeConfig {
        max_draft_tokens: usize::MAX,
        ..Default::default()
    };
    let error = config.validate(8).unwrap_err();
    assert!(
        error.to_string().contains("proposal capacity overflows"),
        "{error}"
    );
    let decoder = Qwen3_5SpeculativeDecoder::<TestBackend>::new(
        Qwen3_5SpeculativeConfig::default(),
        8,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(decoder.config(), &Qwen3_5SpeculativeConfig::default());
}

/// A cycle whose target verification fails restores both caches and every random stream.
///
/// Drafting succeeds (the MTP head has its own final norm), but a NaN target
/// final norm makes the stochastic verification distribution invalid. The
/// staged suffix must be rolled back, and once the norm is repaired the same
/// decoder must reproduce exactly what a fresh decoder with the same seed
/// produces, proving no random draw of the failed cycle leaked.
#[test]
fn verification_failure_restores_caches_and_random_streams() {
    let device = Default::default();
    let mut model = zero_model(&device);
    let config = Qwen3_5SpeculativeConfig {
        generation: GenerationConfig {
            strategy: SamplingStrategy::Sample,
            seed: 0xfa11,
            ..Default::default()
        },
        ..Default::default()
    };
    let (mut state, hidden) = aligned_state(&model, &device);
    let mut decoder =
        Qwen3_5SpeculativeDecoder::new(config.clone(), model.config().vocab_size, &device).unwrap();
    let norm = model.model.norm.weight.val();
    model.model.norm.weight = burn::module::Param::from_tensor(norm.clone() * f32::NAN);
    let error = decoder
        .speculate(&model, 1, hidden.clone(), &[2, 3], 2, &mut state)
        .expect_err("NaN verification logits cannot be sampled");
    assert!(error.to_string().contains("NaN"), "{error}");
    assert!(!state.tensor_state().has_active_transaction());
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));

    model.model.norm.weight = burn::module::Param::from_tensor(norm);
    let retried = decoder
        .speculate(&model, 1, hidden, &[2, 3], 2, &mut state)
        .unwrap();
    let (mut fresh_state, fresh_hidden) = aligned_state(&model, &device);
    let fresh = Qwen3_5SpeculativeDecoder::new(config, model.config().vocab_size, &device)
        .unwrap()
        .speculate(&model, 1, fresh_hidden, &[2, 3], 2, &mut fresh_state)
        .unwrap();
    assert_eq!(retried.token_ids, fresh.token_ids);
    assert_eq!(retried.accepted_draft_tokens, fresh.accepted_draft_tokens);
    assert_eq!(
        state.tensor_state().lengths(),
        fresh_state.tensor_state().lengths()
    );
}

/// Each cycle preflight fault is reported with its own diagnostic before any cache changes.
#[test]
fn cycle_preflight_reports_each_fault() {
    let device = Default::default();
    let model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    let mut decoder = Qwen3_5SpeculativeDecoder::new(
        Qwen3_5SpeculativeConfig::default(),
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    let check = |decoder: &mut Qwen3_5SpeculativeDecoder<TestBackend>,
                 state: &mut Qwen3_5DecodeState<TestBackend>,
                 history: &[u32],
                 prompt_length: usize,
                 expected: &str| {
        let lengths = state.tensor_state().lengths().to_vec();
        let error = decoder
            .speculate(&model, 1, hidden.clone(), history, prompt_length, state)
            .expect_err(expected);
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(state.tensor_state().lengths(), lengths);
    };
    check(
        &mut decoder,
        &mut state,
        &[2, 8],
        2,
        "qwen.speculative.history",
    );
    check(
        &mut decoder,
        &mut state,
        &[2, 3],
        3,
        "qwen.speculative.prompt_length",
    );

    // Test-only corruption: public construction validates the proposal count.
    let mut overflowing = Qwen3_5SpeculativeDecoder::new(
        Qwen3_5SpeculativeConfig::default(),
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    overflowing.config.max_draft_tokens = usize::MAX;
    check(
        &mut overflowing,
        &mut state,
        &[2, 3],
        2,
        "proposal capacity overflows",
    );
    let mut greedy_three = Qwen3_5SpeculativeDecoder::new(
        Qwen3_5SpeculativeConfig {
            max_draft_tokens: 3,
            ..Default::default()
        },
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    check(
        &mut greedy_three,
        &mut state,
        &[2, 3],
        2,
        "staging capacity 3 is below required 4",
    );

    // A target prompt without the shifted MTP prefill is misaligned.
    let mut misaligned = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut misaligned,
            &ForwardOptions::default(),
        )
        .unwrap();
    check(
        &mut decoder,
        &mut misaligned,
        &[2, 3],
        2,
        "requires target = MTP + 1",
    );

    // A two-token prompt plus three staged tokens does not fit four slots.
    let mut full = model
        .new_speculative_decode_state(CacheConfig::new(1, 4).unwrap(), 1, 2, &device)
        .unwrap();
    let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[2, 3]], &device);
    let prompt_hidden = model
        .prefill(
            CausalLmInput {
                token_ids: prompt.clone(),
                token_mask: None,
                position_ids: None,
            },
            &mut full,
            &ForwardOptions {
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap()
        .final_hidden_state
        .unwrap();
    model
        .prefill_mtp_cache(prompt, prompt_hidden, &mut full)
        .unwrap();
    check(
        &mut decoder,
        &mut full,
        &[2, 3],
        2,
        "speculative target suffix reaches 5, exceeding cache capacity 4",
    );
}

/// Selected-token readback rejects negative IDs and a row that did not yield exactly one token.
#[test]
fn token_readback_rejects_negative_ids_and_wrong_counts() {
    let device = Default::default();
    let error =
        read_tokens(Tensor::<TestBackend, 1, Int>::from_ints([3, -1], &device)).unwrap_err();
    assert!(error.to_string().contains("invalid ID -1"), "{error}");
    assert_eq!(
        read_tokens(Tensor::<TestBackend, 1, Int>::from_ints([3, 0], &device)).unwrap(),
        [3, 0]
    );
    let error =
        single_token(Tensor::<TestBackend, 1, Int>::from_ints([1, 2], &device)).unwrap_err();
    assert!(error.to_string().contains("returned 2 tokens"), "{error}");
    assert_eq!(
        single_token(Tensor::<TestBackend, 1, Int>::from_ints([5], &device)).unwrap(),
        5
    );
}

/// A target pass that fails during verification rolls the cycle back.
///
/// The MTP head drafts normally, but the target's only block lost its
/// attention module, so the verification forward fails. Both cursors must
/// be unchanged and no transaction may remain open.
#[test]
fn failing_target_verification_rolls_the_cycle_back() {
    let device = Default::default();
    let mut model = zero_model(&device);
    let (mut state, hidden) = aligned_state(&model, &device);
    model.model.layers[0].self_attn = None;
    let mut decoder = Qwen3_5SpeculativeDecoder::new(
        Qwen3_5SpeculativeConfig::default(),
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    let error = decoder
        .speculate(&model, 1, hidden, &[2, 3], 2, &mut state)
        .expect_err("the target cannot verify without its attention");
    assert!(matches!(error, Error::Checkpoint(_)), "{error}");
    assert!(!state.tensor_state().has_active_transaction());
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));
}
