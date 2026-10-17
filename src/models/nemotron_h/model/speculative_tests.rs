//! Nemotron-H speculative acceptance, EOS alignment, sampling, and retry tests.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData, backend::BackendTypes},
};

use crate::{
    CausalLanguageModel, CausalLmInput, ForwardOptions, LogitsSelection,
    cache::CacheConfig,
    checkpoint::TensorSink,
    models::nemotron_h::{
        NemotronHBlockType, NemotronHCheckpointOptions, NemotronHConfig,
        model::tests::{assert_relative_close, tiny_config, with_dt_bias},
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
        .speculate(&model, 1, hidden, &[2, 3], 2, &mut state)
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
        .speculate(&model, 1, hidden.clone(), &[2, 3], 2, &mut state)
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
        .speculate(&model, 1, hidden, &[2, 3], 2, &mut state)
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
                // Penalize the prompt too, so the second draft skips tokens 0..=3.
                additive_penalty_scope: crate::generation::PenaltyScope::PromptAndGenerated,
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
        .speculate(&model, 1, hidden, &[2, 3], 2, &mut state)
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
        decoder.speculate(&model, 1, hidden, &[2, 3], 2, &mut state),
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
        .speculate_if_active(&model, 1, hidden.clone(), &[2, 3], 2, false, &mut state)
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
        .speculate(&model, 1, hidden.clone(), &[2, 3], 2, &mut state)
        .unwrap_err();
    assert!(matches!(first, Error::Shape { .. }));
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));
    let second = decoder
        .speculate(&model, 1, hidden, &[2, 3], 2, &mut state)
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
            .speculate(&model, 1, hidden.clone(), &[2, 3], 2, &mut state)
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
    assert!(
        decoder
            .speculate(&model, 1, hidden.clone(), &[2, 3], 3, &mut state)
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
        .speculate(&model, 1, left_hidden, &[2, 3], 2, &mut left_state)
        .unwrap();
    let right = right
        .speculate(&model, 1, right_hidden, &[2, 3], 2, &mut right_state)
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
        .speculate(&model, 1, expected_hidden, &[2, 3], 2, &mut expected_state)
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
        retry_decoder.speculate(
            &model,
            1,
            retry_hidden.clone(),
            &[2, 3],
            2,
            &mut retry_state,
        ),
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
        .speculate(&model, 1, retry_hidden, &[2, 3], 2, &mut retry_state)
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

/// Installs deterministic, non-degenerate parameters (sinusoids, unit norms, zero corrections).
fn sinusoidal_model(
    device: &<TestBackend as BackendTypes>::Device,
) -> NemotronHForCausalLm<TestBackend> {
    let mut config = tiny_config(true);
    // One eligible group makes routing independent of group pruning ties.
    config.n_group = 1;
    config.topk_group = 1;
    let mut model = NemotronHForCausalLm::init(config, device).unwrap();
    let contract = model
        .config()
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    for (tensor_index, expected) in contract.expected_tensors().iter().enumerate() {
        let shape = expected
            .shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap())
            .collect::<Vec<_>>();
        let values = (0..shape.iter().product())
            .map(|element| {
                if expected.name.ends_with("correction_bias") {
                    0.0
                } else if expected.name.ends_with("norm.weight") {
                    1.0
                } else {
                    (((element + tensor_index * 13) as f32 + 0.5) * 0.37).sin() * 0.3
                }
            })
            .collect::<Vec<_>>();
        model
            .write_tensor(&expected.name, TensorData::new(values, shape))
            .unwrap();
    }
    model
}

/// Target transition of the hybrid transition fixture: `a -> 3a + 1 (mod width)`.
///
/// Three is coprime with every power of two, so this permutes `0..width`.
const fn transition(token: usize, width: usize) -> usize {
    (3 * token + 1) % width
}

/// A full hybrid model (Mamba-2, attention, MLP, MoE, MTP) with a scripted token transition.
///
/// With hidden width `H`, tokens `0..H` embed as `4 e_a` and the vocabulary
/// head maps hidden coordinate `c` to token `transition(c, H)`. Every other
/// parameter is a small sinusoid, so all mixers run and perturb the residual
/// stream without changing the argmax. The MTP stage reads only the token
/// half of its fusion and copies coordinate `a` except `5 -> 3`, so it
/// drafts the target's successor for every token but 5. Greedy speculation
/// therefore mixes full, partial, and zero acceptance.
fn transition_fixture(
    mut config: NemotronHConfig,
    device: &<TestBackend as BackendTypes>::Device,
) -> NemotronHForCausalLm<TestBackend> {
    config.n_group = 1;
    config.topk_group = 1;
    config.tie_word_embeddings = false;
    let mut model = NemotronHForCausalLm::init(config, device).unwrap();
    let hidden = model.config().hidden_size;
    let contract = model
        .config()
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    for (tensor_index, expected) in contract.expected_tensors().iter().enumerate() {
        let shape = expected
            .shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap())
            .collect::<Vec<_>>();
        let small =
            |element: usize| (((element + tensor_index * 13) as f32 + 0.5) * 0.37).sin() * 0.02;
        let mut values = (0..shape.iter().product()).map(small).collect::<Vec<_>>();
        match expected.name.as_str() {
            name if name.ends_with("correction_bias") => values.fill(0.0),
            name if name.ends_with("norm.weight") => values.fill(1.0),
            "model.embed_tokens.weight" => {
                for token in 0..hidden {
                    values[token * hidden..(token + 1) * hidden].fill(0.0);
                    values[token * hidden + token] = 4.0;
                }
            }
            "lm_head.weight" => {
                for coordinate in 0..hidden {
                    values[transition(coordinate, hidden) * hidden + coordinate] = 1.0;
                }
            }
            "next_token_head.eh_proj.weight" => {
                values.fill(0.0);
                for coordinate in 0..hidden {
                    let output = if coordinate == 5 { 3 } else { coordinate };
                    values[output * 2 * hidden + coordinate] = 1.0;
                }
            }
            _ => {}
        }
        model
            .write_tensor(&expected.name, TensorData::new(values, shape))
            .unwrap();
    }
    model
}

/// The tiny transition fixture: hidden width 8, vocabulary 32.
fn transition_model(
    device: &<TestBackend as BackendTypes>::Device,
) -> NemotronHForCausalLm<TestBackend> {
    transition_fixture(tiny_config(true), device)
}

/// Uploads host token IDs as a `[1, n]` batch-one row.
fn token_row(tokens: &[u32]) -> Tensor<TestBackend, 2, Int> {
    let values = tokens
        .iter()
        .map(|&token| i64::from(token))
        .collect::<Vec<_>>();
    Tensor::from_data(
        TensorData::new(values, [1, tokens.len()]),
        &Default::default(),
    )
}

/// Greedy decoding with one ordinary target pass per token.
fn plain_greedy(
    model: &NemotronHForCausalLm<TestBackend>,
    prompt: &[u32],
    new_tokens: usize,
) -> Vec<u32> {
    plain_greedy_with(model, prompt, new_tokens, 32)
}

/// [`plain_greedy`] with an explicit cache capacity.
fn plain_greedy_with(
    model: &NemotronHForCausalLm<TestBackend>,
    prompt: &[u32],
    new_tokens: usize,
    capacity: usize,
) -> Vec<u32> {
    plain_greedy_configured(
        model,
        prompt,
        new_tokens,
        capacity,
        &GenerationConfig::default(),
        prompt.len(),
    )
}

/// Greedy decoding exactly as `generate_causal` processes logits, one target pass per token.
///
/// Logits go through `generation`'s processors with the first
/// `prompt_length` history tokens treated as the prompt (tests pass a wrong
/// length to prove that the comparison is sensitive to it). Decoding stops
/// after `new_tokens` tokens or right after an EOS token.
fn plain_greedy_configured(
    model: &NemotronHForCausalLm<TestBackend>,
    prompt: &[u32],
    new_tokens: usize,
    capacity: usize,
    generation: &GenerationConfig,
    prompt_length: usize,
) -> Vec<u32> {
    let device = Default::default();
    let processor =
        LogitsProcessor::new(generation.clone(), model.config().vocab_size, &device).unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, capacity).unwrap(), 1, &device)
        .unwrap();
    let mut logits = model
        .prefill(
            CausalLmInput {
                token_ids: token_row(prompt),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits
        .squeeze_dim::<2>(1);
    let mut history = prompt.to_vec();
    let mut generated = Vec::with_capacity(new_tokens);
    while generated.len() < new_tokens {
        let processed = processor
            .process_with_prompt_lengths(logits, &[history.as_slice()], &[prompt_length])
            .unwrap();
        let token = first_token(processor.select(processed).unwrap()).unwrap();
        generated.push(token);
        history.push(token);
        if generation.eos_token_ids.contains(&token) || generated.len() == new_tokens {
            break;
        }
        logits = model
            .decode(
                crate::DecodeInput {
                    token_ids: token_row(&[token]),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
            .squeeze_dim::<2>(1);
    }
    generated
}

/// Statistics of one speculative greedy generation.
struct SpeculativeRun {
    /// Generated tokens, truncated to the requested count.
    generated: Vec<u32>,
    /// Committed target tokens (prompt plus every consumed token).
    history: Vec<u32>,
    /// Final decode state.
    state: NemotronHDecodeState<TestBackend>,
    /// Accepted drafts and drafted candidates of every cycle.
    cycles: Vec<(usize, usize)>,
}

/// Greedy decoding through MTP speculation, one target pass per cycle.
fn speculative_greedy(
    model: &NemotronHForCausalLm<TestBackend>,
    prompt: &[u32],
    new_tokens: usize,
    max_draft_tokens: usize,
) -> SpeculativeRun {
    speculative_greedy_with(model, prompt, new_tokens, max_draft_tokens, 32)
}

/// [`speculative_greedy`] with an explicit cache capacity.
fn speculative_greedy_with(
    model: &NemotronHForCausalLm<TestBackend>,
    prompt: &[u32],
    new_tokens: usize,
    max_draft_tokens: usize,
    capacity: usize,
) -> SpeculativeRun {
    speculative_greedy_configured(
        model,
        prompt,
        new_tokens,
        max_draft_tokens,
        capacity,
        GenerationConfig::default(),
    )
}

/// [`speculative_greedy`] with explicit logit processing; stops at an emitted EOS.
fn speculative_greedy_configured(
    model: &NemotronHForCausalLm<TestBackend>,
    prompt: &[u32],
    new_tokens: usize,
    max_draft_tokens: usize,
    capacity: usize,
    generation: GenerationConfig,
) -> SpeculativeRun {
    let device = Default::default();
    let mut decoder = NemotronHSpeculativeDecoder::new(
        NemotronHSpeculativeConfig {
            max_draft_tokens,
            generation,
        },
        model.config().vocab_size,
        &device,
    )
    .unwrap();
    let mut state = model
        .new_speculative_decode_state(
            CacheConfig::new(1, capacity).unwrap(),
            1,
            max_draft_tokens,
            &device,
        )
        .unwrap();
    let prefill = model
        .prefill_speculative(token_row(prompt), &mut state)
        .unwrap();
    let mut lead = decoder
        .select_lead(prefill.logits, prompt, prompt.len())
        .unwrap();
    let mut hidden = prefill.hidden_state;
    let mut history = prompt.to_vec();
    let mut generated = vec![lead];
    let mut cycles = Vec::new();
    let eos = decoder.config().generation.eos_token_ids.clone();
    let mut terminated = eos.contains(&lead);
    while generated.len() < new_tokens && !terminated {
        let passes = super::super::decode::CACHED_PASSES.with(std::cell::Cell::get);
        let output = decoder
            .speculate(model, lead, hidden, &history, prompt.len(), &mut state)
            .unwrap();
        let after = super::super::decode::CACHED_PASSES.with(std::cell::Cell::get);
        assert_eq!(after - passes, 1, "one target pass per cycle");
        cycles.push((output.accepted_draft_tokens, output.drafted_tokens));
        let emitted = &output.token_ids;
        history.extend_from_slice(&emitted[..emitted.len() - 1]);
        generated.extend_from_slice(&emitted[1..]);
        lead = *emitted.last().unwrap();
        terminated = output.terminated;
        assert_eq!(terminated, eos.contains(&lead));
        match output.continuation_hidden_state {
            Some(continuation) => hidden = continuation,
            None => {
                assert!(terminated, "only EOS ends the continuation");
                break;
            }
        }
        assert_eq!(state.tensor_state().lengths(), &[history.len()]);
        assert_eq!(
            state.tensor_state().mtp_lengths(),
            Some([history.len() - 1].as_slice())
        );
    }
    generated.truncate(new_tokens);
    SpeculativeRun {
        generated,
        history,
        state,
        cycles,
    }
}

/// Requires a speculative run's committed cache to equal token-by-token decoding of its history.
///
/// The reference prefills the prompt and decodes every later history token
/// in its own call, which is exactly the cache that plain greedy decoding
/// builds. Attention keys and values must agree within F32 rounding, and
/// every Mamba-2 state relatively (see `assert_relative_close`), so on a
/// clamped-`dt` model a verification pass that ran the chunk scan instead of
/// per-token single-step updates fails here even when greedy tokens agree.
fn assert_cache_matches_token_by_token(
    model: &NemotronHForCausalLm<TestBackend>,
    prompt: &[u32],
    run: &SpeculativeRun,
) {
    let device = Default::default();
    let mut reference = model
        .new_decode_state(CacheConfig::new(1, 32).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: token_row(prompt),
                token_mask: None,
                position_ids: None,
            },
            &mut reference,
            &ForwardOptions::default(),
        )
        .unwrap();
    for &token in &run.history[prompt.len()..] {
        model
            .decode(
                crate::DecodeInput {
                    token_ids: token_row(&[token]),
                    token_mask: None,
                    position_ids: None,
                },
                &mut reference,
                &ForwardOptions::default(),
            )
            .unwrap();
    }
    let length = run.history.len();
    assert_eq!(reference.tensor_state().lengths(), &[length]);
    let (actual, expected) = (
        run.state.tensor_state().main(),
        reference.tensor_state().main(),
    );
    for group in 0..expected.native_kv_count() {
        let (actual, expected) = (
            actual.native_kv(group).unwrap(),
            expected.native_kv(group).unwrap(),
        );
        for layer in 0..expected.spec().layers() {
            for (actual, expected) in [
                (actual.keys(layer).unwrap(), expected.keys(layer).unwrap()),
                (
                    actual.values(layer).unwrap(),
                    expected.values(layer).unwrap(),
                ),
            ] {
                actual
                    .clone()
                    .narrow(2, 0, length)
                    .into_data()
                    .assert_approx_eq::<f32>(
                        &expected.clone().narrow(2, 0, length).into_data(),
                        burn::tensor::Tolerance::absolute(1e-4),
                    );
            }
        }
    }
    for group in 0..expected.recurrent_count() {
        let (actual, expected) = (
            actual.recurrent(group).unwrap(),
            expected.recurrent(group).unwrap(),
        );
        for layer in 0..expected.spec().layers() {
            assert_relative_close(
                actual.matrix(layer).unwrap().clone().into_data(),
                expected.matrix(layer).unwrap().clone().into_data(),
                1e-4,
                &format!("Mamba-2 layer {layer} matrix state"),
            );
            assert_relative_close(
                actual.convolution(layer).unwrap().clone().into_data(),
                expected.convolution(layer).unwrap().clone().into_data(),
                1e-4,
                &format!("Mamba-2 layer {layer} convolution state"),
            );
        }
    }
}

/// Greedy speculation emits exactly the plain greedy sequence (NEM-1 losslessness).
///
/// The graphs cover the acceptance space: deterministic sinusoidal weights
/// (a non-degenerate hybrid whose drafts are rejected), the hybrid transition
/// fixture (full, partial, and zero acceptance through Mamba-2 snapshots and
/// attention), a tied-logit graph whose drafts are always accepted, and a
/// scripted graph that alternates. Every cycle costs exactly one target pass,
/// and the committed cache equals token-by-token decoding. The clamped-`dt`
/// variants (review NEMR-1) make the chunk scan and the single-step update
/// leave different Mamba-2 states, so the cache comparison would catch a
/// verification pass that ran the wrong path.
#[test]
fn greedy_speculation_matches_plain_greedy_decoding() {
    let device = Default::default();
    let (mut rejected, mut partial, mut fully_accepted) = (0, 0, 0);
    for (model, prompt, draft) in [
        (sinusoidal_model(&device), [5, 9, 2, 7], 2),
        (sinusoidal_model(&device), [5, 9, 2, 7], 3),
        (transition_model(&device), [6, 3, 1, 4], 2),
        (transition_model(&device), [6, 3, 1, 4], 4),
        (zero_model(&device), [5, 9, 2, 7], 2),
        (scripted_rejection_model(&device), [5, 9, 2, 7], 3),
        (
            with_dt_bias(sinusoidal_model(&device), -12.0),
            [5, 9, 2, 7],
            3,
        ),
        (
            with_dt_bias(transition_model(&device), -12.0),
            [6, 3, 1, 4],
            3,
        ),
    ] {
        let expected = plain_greedy(&model, &prompt, 14);
        let run = speculative_greedy(&model, &prompt, 14, draft);
        assert_eq!(run.generated, expected);
        assert_cache_matches_token_by_token(&model, &prompt, &run);
        for &(accepted, drafted) in &run.cycles {
            match accepted {
                0 => rejected += 1,
                accepted if accepted < drafted => partial += 1,
                _ => fully_accepted += 1,
            }
        }
    }
    assert!(
        rejected > 0 && partial > 0 && fully_accepted > 0,
        "rejected={rejected}, partial={partial}, full={fully_accepted}"
    );
}

/// Greedy speculation with logit processing matches plain greedy decoding (review NEMR-2).
///
/// The verifier processes verification row `i` with its own history
/// `sequence[..base + i + 1]` and the prompt length, and each draft with the
/// tokens drafted so far. Repetition, presence, and frequency penalties
/// (generated-only scope), an n-gram block, and EOS with `min_new_tokens`
/// all read that history, so an off-by-one or a wrong prompt length in the
/// target processing changes the emitted tokens. The test also proves it is
/// not vacuous: processing changes every plain greedy output, treating the
/// prompt as generated changes at least one output per configuration, and
/// the delayed EOS both suppresses an early EOS and terminates a run (once
/// through an accepted EOS draft, once through an EOS bonus).
///
/// Greedy speculation is lossless whatever the drafts are, so the draft
/// processor is checked through acceptance instead: the transition
/// fixture's MTP stage mirrors the target's transition, so with
/// generated-only presence penalties every cycle after the first (whose lead
/// is the one token MTP mispredicts) accepts all drafts only if drafts are
/// processed with the same history and prompt boundary as the target.
#[test]
fn processed_greedy_speculation_matches_plain_greedy_decoding() {
    let device = Default::default();
    let penalties = GenerationConfig {
        repetition_penalty: 1.5,
        presence_penalty: 0.6,
        frequency_penalty: 0.3,
        no_repeat_ngram_size: 2,
        ..Default::default()
    };
    let generated_only = GenerationConfig {
        presence_penalty: 3.0,
        ..Default::default()
    };
    let delayed_eos = GenerationConfig {
        eos_token_ids: vec![1, 7],
        min_new_tokens: 6,
        max_new_tokens: 64,
        ..Default::default()
    };
    let new_tokens = 16;
    let mut terminated = 0;
    // Final cycles of terminated runs: (accepted, drafted, K).
    let mut terminal_cycles = Vec::new();
    for generation in [penalties, generated_only.clone(), delayed_eos] {
        let mut prompt_sensitive = false;
        for (transition, model, prompt) in [
            (false, sinusoidal_model(&device), [5, 9, 2, 7]),
            (true, transition_model(&device), [6, 3, 1, 4]),
        ] {
            let expected =
                plain_greedy_configured(&model, &prompt, new_tokens, 32, &generation, prompt.len());
            let unprocessed = plain_greedy(&model, &prompt, new_tokens);
            assert_ne!(
                expected[..],
                unprocessed[..expected.len()],
                "{generation:?}"
            );
            prompt_sensitive |= expected
                != plain_greedy_configured(&model, &prompt, new_tokens, 32, &generation, 0);
            if expected.len() < new_tokens {
                terminated += 1;
                assert!(expected.len() > generation.min_new_tokens);
                assert!(generation.eos_token_ids.contains(expected.last().unwrap()));
            }
            for draft in [2, 3] {
                let run = speculative_greedy_configured(
                    &model,
                    &prompt,
                    new_tokens,
                    draft,
                    32,
                    generation.clone(),
                );
                assert_eq!(run.generated, expected, "{generation:?}, K = {draft}");
                assert_cache_matches_token_by_token(&model, &prompt, &run);
                if expected.len() < new_tokens {
                    let &(accepted, drafted) = run.cycles.last().unwrap();
                    terminal_cycles.push((accepted, drafted, draft));
                }
                if transition && generation == generated_only {
                    assert!(
                        run.cycles[1..]
                            .iter()
                            .all(|&(accepted, drafted)| accepted == drafted),
                        "draft processing diverged from target processing: {:?}",
                        run.cycles
                    );
                }
            }
        }
        assert!(
            prompt_sensitive,
            "{generation:?} ignores the prompt boundary"
        );
    }
    assert_eq!(terminated, 1);
    // One run ends on an accepted EOS draft (drafting stopped early), the
    // other on an EOS bonus after accepting all K drafts.
    assert!(
        terminal_cycles
            .iter()
            .any(|&(accepted, drafted, k)| accepted == drafted && drafted < k)
    );
    assert!(
        terminal_cycles
            .iter()
            .any(|&(accepted, drafted, k)| accepted == drafted && drafted == k)
    );
}

/// Committed MTP pairs come from target hidden states, as a fresh prefill would build them (NEM-6).
#[test]
fn cycle_mtp_cache_matches_a_fresh_prefill_of_the_verified_prefix() {
    let device = Default::default();
    for (model, prompt, draft) in [
        (sinusoidal_model(&device), [5, 9, 2, 7], 3),
        (transition_model(&device), [6, 3, 1, 4], 4),
        (zero_model(&device), [5, 9, 2, 7], 2),
    ] {
        let run = speculative_greedy(&model, &prompt, 10, draft);
        let mut fresh = model
            .new_speculative_decode_state(CacheConfig::new(1, 32).unwrap(), 1, draft, &device)
            .unwrap();
        let hidden = model
            .prefill(
                CausalLmInput {
                    token_ids: token_row(&run.history),
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
            .prefill_mtp_cache(token_row(&run.history), hidden, &mut fresh)
            .unwrap();
        let mtp = |state: &NemotronHDecodeState<TestBackend>| {
            let length = state.tensor_state().mtp_lengths().unwrap()[0];
            let cache = state
                .tensor_state()
                .mtp()
                .unwrap()
                .native_kv(super::super::decode::KV_GROUP)
                .unwrap();
            [cache.keys(0).unwrap(), cache.values(0).unwrap()]
                .map(|plane| plane.clone().narrow(2, 0, length).into_data())
        };
        for (actual, expected) in mtp(&run.state).into_iter().zip(mtp(&fresh)) {
            actual.assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::absolute(1e-5));
        }
    }
}

/// Lead selection honors the prompt boundary like `generate_causal`.
///
/// With a presence penalty scoped to generated tokens, a prompt token stays
/// unpenalized; treating it as generated (prompt length zero) penalizes it.
/// `min_new_tokens` suppresses EOS until enough tokens were generated.
#[test]
fn lead_selection_respects_the_prompt_boundary() {
    let device = Default::default();
    let tied = Tensor::<TestBackend, 2>::zeros([1, 32], &device);
    let penalized = NemotronHSpeculativeDecoder::<TestBackend>::new(
        NemotronHSpeculativeConfig {
            generation: GenerationConfig {
                presence_penalty: 1.0,
                ..Default::default()
            },
            ..Default::default()
        },
        32,
        &device,
    )
    .unwrap();
    assert_eq!(penalized.select_lead(tied.clone(), &[0, 3], 2).unwrap(), 0);
    assert_eq!(penalized.select_lead(tied.clone(), &[0, 3], 0).unwrap(), 1);
    assert!(penalized.select_lead(tied.clone(), &[0, 3], 3).is_err());
    let eager_eos = NemotronHSpeculativeDecoder::<TestBackend>::new(
        NemotronHSpeculativeConfig {
            generation: GenerationConfig {
                eos_token_ids: vec![0],
                min_new_tokens: 1,
                max_new_tokens: 4,
                ..Default::default()
            },
            ..Default::default()
        },
        32,
        &device,
    )
    .unwrap();
    assert_eq!(eager_eos.select_lead(tied.clone(), &[5, 6], 2).unwrap(), 1);
    assert_eq!(eager_eos.select_lead(tied, &[5, 6], 1).unwrap(), 0);
}

/// Release-mode timing of MTP speculation against plain greedy decoding.
///
/// Uses the transition fixture at hidden width 128 with eight hybrid blocks,
/// whose drafts are accepted except after token 5. Run with
/// `cargo test --release --lib nemotron_h::model::speculative::tests::speculative_decoding_timing
/// -- --ignored --nocapture`.
#[test]
#[ignore = "release-mode timing benchmark"]
fn speculative_decoding_timing() {
    use std::time::Instant;

    let device = Default::default();
    let mut config = tiny_config(true);
    config.vocab_size = 256;
    config.hidden_size = 128;
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
    config.head_dim = 32;
    config.max_position_embeddings = 256;
    config.intermediate_size = 256;
    config.ssm_state_size = 32;
    config.mamba_num_heads = 8;
    config.mamba_head_dim = 16;
    config.n_groups = 2;
    config.conv_kernel = 4;
    config.chunk_size = 32;
    config.n_routed_experts = 16;
    config.num_experts_per_tok = 4;
    config.moe_intermediate_size = 64;
    config.moe_shared_expert_intermediate_size = 128;
    let model = transition_fixture(config, &device);
    let prompt = [6, 19, 58, 47];
    let new_tokens = 48;
    let time = |label: &str, run: &dyn Fn() -> Vec<u32>| {
        let _ = run();
        let started = Instant::now();
        let tokens = run();
        let seconds = started.elapsed().as_secs_f64();
        eprintln!(
            "nemotron speculative timing: {label}: {:.2} ms/token",
            seconds * 1e3 / new_tokens as f64
        );
        (seconds, tokens)
    };
    let (plain, expected) = time("plain greedy", &|| {
        plain_greedy_with(&model, &prompt, new_tokens, 256)
    });
    for draft in [2, 3, 4] {
        let (speculative, tokens) = time(&format!("MTP speculation, K = {draft}"), &|| {
            speculative_greedy_with(&model, &prompt, new_tokens, draft, 256).generated
        });
        assert_eq!(tokens, expected);
        let run = speculative_greedy_with(&model, &prompt, new_tokens, draft, 256);
        let accepted = run.cycles.iter().map(|cycle| cycle.0).sum::<usize>();
        let drafted = run.cycles.iter().map(|cycle| cycle.1).sum::<usize>();
        eprintln!(
            "nemotron speculative timing: K = {draft}: {} cycles, acceptance {:.2}, speedup {:.2}x",
            run.cycles.len(),
            accepted as f64 / drafted as f64,
            plain / speculative
        );
    }
}

/// Preflight names out-of-vocabulary history, missing staging, misalignment, and capacity.
#[test]
fn cycle_preflight_reports_each_rejection_exactly() {
    let device = Default::default();
    let model = zero_model(&device);
    let decoder = |max_draft_tokens| {
        NemotronHSpeculativeDecoder::<TestBackend>::new(
            NemotronHSpeculativeConfig {
                max_draft_tokens,
                ..Default::default()
            },
            model.config().vocab_size,
            &device,
        )
        .unwrap()
    };

    let (mut state, hidden) = aligned_state(&model, &device);
    let error = decoder(2)
        .speculate(&model, 1, hidden.clone(), &[2, 32], 2, &mut state)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, expected, actual }
            if tensor == "nemotron_h.speculative.history"
            && expected == "token values in 0..32"
            && actual == "an out-of-vocabulary token"),
        "{error}"
    );
    let error = decoder(3)
        .speculate(&model, 1, hidden.clone(), &[2, 3], 2, &mut state)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Generation(reason)
            if reason == "Nemotron-H decode-state staging capacity 3 is below required 4"),
        "{error}"
    );
    let error = decoder(2)
        .select_lead(Tensor::zeros([1, 32], &device), &[2, 3], 3)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, expected, actual }
            if tensor == "nemotron_h.speculative.prompt_length"
            && expected == "at most the 2 history tokens"
            && actual == "3"),
        "{error}"
    );
    assert_eq!(state.tensor_state().lengths(), &[2]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));

    // A target prefilled without its shifted MTP pairs is not a cycle boundary.
    let mut unaligned = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut unaligned,
            &ForwardOptions::default(),
        )
        .unwrap();
    let error = decoder(2)
        .speculate(&model, 1, hidden.clone(), &[2, 3], 2, &mut unaligned)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Cache(reason)
            if reason == "Nemotron-H speculative alignment requires target = MTP + 1, found target=[2], MTP=[0]"),
        "{error}"
    );

    // A cycle whose `K + 1` tentative tokens would pass the cache capacity.
    let mut full = model
        .new_speculative_decode_state(CacheConfig::new(1, 4).unwrap(), 1, 2, &device)
        .unwrap();
    let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[2, 3]], &device);
    let prefill = model.prefill_speculative(prompt, &mut full).unwrap();
    let error = decoder(2)
        .speculate(&model, 1, prefill.hidden_state, &[2, 3], 2, &mut full)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Generation(reason)
            if reason == "Nemotron-H speculative suffix reaches 5, exceeding cache capacity 4"),
        "{error}"
    );
    assert_eq!(full.tensor_state().lengths(), &[2]);
    assert!(!full.tensor_state().has_active_transaction());
}

/// The decoder exposes its validated configuration unchanged.
#[test]
fn decoder_exposes_its_validated_configuration() {
    let device = Default::default();
    let config = NemotronHSpeculativeConfig {
        max_draft_tokens: 3,
        ..Default::default()
    };
    let decoder =
        NemotronHSpeculativeDecoder::<TestBackend>::new(config.clone(), 32, &device).unwrap();
    assert_eq!(decoder.config(), &config);
    let error = NemotronHSpeculativeDecoder::<TestBackend>::new(
        NemotronHSpeculativeConfig {
            max_draft_tokens: 0,
            ..Default::default()
        },
        32,
        &device,
    )
    .unwrap_err();
    assert!(
        matches!(&error, Error::Generation(reason)
            if reason == "Nemotron-H MTP max_draft_tokens must be greater than zero"),
        "{error}"
    );
    let error = NemotronHSpeculativeConfig {
        max_draft_tokens: usize::MAX,
        ..Default::default()
    }
    .validate(32)
    .unwrap_err();
    assert!(
        matches!(&error, Error::Generation(reason)
            if reason == "Nemotron-H MTP proposal capacity overflows usize"),
        "{error}"
    );
}

/// Selected-token readback converts backend integers and rejects an empty selection.
#[test]
fn selected_token_readback_converts_and_requires_a_row() {
    let device = Default::default();
    assert_eq!(
        tokens(Tensor::<TestBackend, 1, Int>::from_ints(
            [3, 0, 31],
            &device
        ))
        .unwrap(),
        [3, 0, 31]
    );
    assert_eq!(
        first_token(Tensor::<TestBackend, 1, Int>::from_ints([7, 2], &device)).unwrap(),
        7
    );
    let error = tokens(Tensor::<TestBackend, 1, Int>::from_ints([-1], &device)).unwrap_err();
    assert!(
        matches!(&error, Error::Generation(reason) if reason == "Nemotron-H selected invalid token -1"),
        "{error}"
    );
    let error = first_token(Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(Vec::<i64>::new(), [0]),
        &device,
    ))
    .unwrap_err();
    assert!(
        matches!(&error, Error::Generation(reason)
            if reason == "Nemotron-H token selector returned no row"),
        "{error}"
    );
}
