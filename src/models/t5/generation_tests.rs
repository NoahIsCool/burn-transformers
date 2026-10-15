//! Tests T5 beam-search configuration, execution, and pinned reference tokens.

use std::{fs, path::PathBuf};

use burn::{
    backend::Flex,
    tensor::{Int, Tensor},
};

use super::{T5BeamSearchOptions, generate_t5_beam_search, one_token};
use crate::{
    CausalLmInput, FinishReason,
    checkpoint::LoadOptions,
    models::{
        config::{ModelConfig, TokenIds},
        t5::{T5Config, T5FeedForwardProjection, T5ForConditionalGeneration},
    },
};

/// CPU backend used by T5 beam-search regressions.
type TestBackend = Flex<f32, i32>;

/// Builds a minimal real T5 graph for family-wrapper coverage.
fn tiny_config() -> T5Config {
    T5Config {
        vocab_size: 16,
        d_model: 8,
        d_kv: 4,
        d_ff: 16,
        num_layers: 1,
        num_decoder_layers: Some(1),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        dropout_rate: 0.0,
        feed_forward_proj: T5FeedForwardProjection::Relu,
        tie_word_embeddings: true,
        ..T5Config::default()
    }
}

/// Checkpoint start/EOS/PAD fields drive a complete cached beam request.
#[test]
fn family_helper_runs_a_tiny_real_t5_graph() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(tiny_config(), &device)
        .expect("tiny T5 model");
    let output = generate_t5_beam_search(
        &model,
        CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[3, 4]], &device),
            token_mask: None,
            position_ids: None,
        },
        T5BeamSearchOptions {
            max_new_tokens: 2,
            num_beams: 2,
            num_return_sequences: 2,
            ..T5BeamSearchOptions::default()
        },
    )
    .expect("tiny T5 beam search");
    assert_eq!(output.sequences.len(), 1);
    assert_eq!(output.sequences[0].len(), 2);
    assert!(output.sequences[0].iter().all(|sequence| {
        sequence.decoder_prompt_length == 1
            && (1..=2).contains(&sequence.token_ids.len())
            && sequence.log_probabilities.len() == sequence.token_ids.len()
            && matches!(
                sequence.finish_reason,
                FinishReason::Length | FinishReason::EndOfSequence { .. }
            )
    }));
}

/// An exactly sized cache (`max_new_tokens` slots) suffices for every length and stopping flag.
///
/// The decoder-start token takes one slot and the final selected token is
/// never decoded, so a run writes at most `1 + max_new_tokens - 1` slots;
/// the shared preflight rejects smaller caches and the decode state rejects
/// writes beyond its capacity, so success proves the size is sufficient.
#[test]
fn family_helper_sizes_the_cache_exactly_for_every_length() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(tiny_config(), &device)
        .expect("tiny T5 model");
    for max_new_tokens in 1..=4 {
        for early_stopping in [false, true] {
            let output = generate_t5_beam_search(
                &model,
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                        [[3, 4, 5], [6, 7, 8]],
                        &device,
                    ),
                    token_mask: None,
                    position_ids: None,
                },
                T5BeamSearchOptions {
                    max_new_tokens,
                    num_beams: 3,
                    num_return_sequences: 2,
                    early_stopping,
                    ..T5BeamSearchOptions::default()
                },
            )
            .expect("exactly sized T5 beam search");
            assert_eq!(output.sequences.len(), 2);
            assert!(
                output
                    .sequences
                    .iter()
                    .flatten()
                    .all(|sequence| { (1..=max_new_tokens).contains(&sequence.token_ids.len()) })
            );
        }
    }
}

/// PAD is intentionally singular because inactive beam rows need one ID.
#[test]
fn family_token_defaults_reject_missing_empty_and_ambiguous_pad() {
    assert!(one_token(None, "pad_token_id").is_err());
    assert!(one_token(Some(&TokenIds::Many(Vec::new())), "pad_token_id").is_err());
    assert!(one_token(Some(&TokenIds::Many(vec![0, 1])), "pad_token_id").is_err());
    assert_eq!(
        one_token(Some(&TokenIds::One(7)), "pad_token_id").unwrap(),
        7
    );
}

/// Pinned FLAN-Small beam output matches Transformers 5.2.0 token-for-token.
#[test]
#[ignore = "requires pinned google/flan-t5-small SafeTensors artifacts"]
fn pinned_flan_small_beam_search_matches_reference_tokens() {
    let root = std::env::var_os("TRANSFORMERS_FLAN_T5_SMALL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models/google/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab")
        });
    let config = T5Config::from_json_str(
        &fs::read_to_string(root.join("config.json")).expect("read pinned FLAN config"),
    )
    .expect("parse pinned FLAN config");
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(config, &device)
        .expect("initialize FLAN graph");
    model
        .load_checkpoint(&root, &LoadOptions::default())
        .expect("strictly load pinned FLAN weights");

    let output = generate_t5_beam_search(
        &model,
        CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                [[13_959, 1_566, 12, 2_968, 10, 37, 629, 19, 1_627, 5, 1]],
                &device,
            ),
            token_mask: None,
            position_ids: None,
        },
        T5BeamSearchOptions {
            max_new_tokens: 16,
            num_beams: 4,
            num_return_sequences: 1,
            length_penalty: 1.0,
            early_stopping: false,
        },
    )
    .expect("run cached FLAN beam search");
    assert_eq!(output.sequences.len(), 1);
    assert_eq!(output.sequences[0].len(), 1);
    assert_eq!(
        output.sequences[0][0].token_ids,
        [644, 4_598, 229, 9_685, 5, 1]
    );
    assert!(matches!(
        output.sequences[0][0].finish_reason,
        FinishReason::EndOfSequence { token_id: 1 }
    ));
}
