//! T5-family defaults for the generic encoder-decoder beam coordinator.
//!
//! T5 begins autoregressive decoding with `decoder_start_token_id` (normally
//! PAD), while EOS and PAD policy live in checkpoint configuration. This module
//! translates those family fields into the architecture-neutral beam request;
//! the actual cache reordering and probability search remain shared in
//! [`crate::generation`].

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::{
    CausalLmInput, Error, Result,
    cache::CacheConfig,
    generation::{
        BeamSearchConfig, Seq2SeqBeamSearchOutput, Seq2SeqBeamSearchRequest,
        generate_seq2seq_beam_search,
    },
    models::config::TokenIds,
};

use super::T5ForConditionalGeneration;

/// Family-level T5 beam controls; token IDs come from the loaded checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct T5BeamSearchOptions {
    /// Maximum number of tokens generated after T5's decoder-start token.
    pub max_new_tokens: usize,
    /// Number of decoder prefixes retained per source row.
    pub num_beams: usize,
    /// Number of ranked hypotheses returned per source row.
    pub num_return_sequences: usize,
    /// Exponent used to normalize cumulative log probability by output length.
    pub length_penalty: f64,
    /// Whether to stop once every source row has `num_beams` EOS hypotheses.
    pub early_stopping: bool,
}

impl Default for T5BeamSearchOptions {
    /// Returns common FLAN/T5 beam-search defaults.
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            num_beams: 4,
            num_return_sequences: 1,
            length_penalty: 1.0,
            early_stopping: false,
        }
    }
}

/// Generates ranked T5/FLAN/UL2 continuations using checkpoint token defaults.
///
/// The helper constructs a one-token decoder prompt and a cache whose fixed
/// batch is `source_batch * num_beams`. Call
/// [`generate_seq2seq_beam_search`] directly when prefix-constrained decoding
/// needs a longer caller-supplied decoder prompt.
pub fn generate_t5_beam_search<B: Backend>(
    model: &T5ForConditionalGeneration<B>,
    source: CausalLmInput<B>,
    options: T5BeamSearchOptions,
) -> Result<Seq2SeqBeamSearchOutput> {
    let config = model.config();
    let decoder_start = config
        .decoder_start_token_id
        .ok_or_else(|| Error::Generation("T5 config has no decoder_start_token_id".into()))?;
    let decoder_start = token_to_u32(decoder_start, "decoder_start_token_id")?;
    let pad_token_id = one_token(config.pad_token_id.as_ref(), "pad_token_id")?;
    let eos_token_ids = config
        .eos_token_id
        .as_ref()
        .ok_or_else(|| Error::Generation("T5 config has no eos_token_id".into()))?
        .iter()
        .map(|token| token_to_u32(token, "eos_token_id"))
        .collect::<Result<Vec<_>>>()?;
    if eos_token_ids.is_empty() {
        return Err(Error::Generation(
            "T5 config eos_token_id list must not be empty".into(),
        ));
    }
    let search = BeamSearchConfig {
        max_new_tokens: options.max_new_tokens,
        num_beams: options.num_beams,
        num_return_sequences: options.num_return_sequences,
        length_penalty: options.length_penalty,
        early_stopping: options.early_stopping,
        eos_token_ids,
        pad_token_id,
    };
    search.validate(config.vocab_size)?;
    let [batch, _] = source.token_ids.dims();
    let expanded_batch = batch
        .checked_mul(search.num_beams)
        .ok_or_else(|| Error::Generation("T5 beam batch overflows usize".into()))?;
    let capacity = search
        .max_new_tokens
        .checked_add(1)
        .ok_or_else(|| Error::Generation("T5 beam cache capacity overflows usize".into()))?;
    let cache = CacheConfig::new(expanded_batch, capacity)
        .map_err(|error| Error::Generation(format!("invalid T5 beam cache geometry: {error}")))?;
    let device = source.token_ids.device();
    let decoder_prompt = CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![i64::from(decoder_start); batch], [batch, 1]),
            &device,
        ),
        token_mask: None,
        position_ids: None,
    };
    generate_seq2seq_beam_search(
        model,
        Seq2SeqBeamSearchRequest {
            source,
            decoder_prompt,
            cache,
            search,
            vocabulary_size: config.vocab_size,
        },
    )
}

/// Extracts one unambiguous scalar token ID from a configuration field.
fn one_token(tokens: Option<&TokenIds>, field: &str) -> Result<u32> {
    let mut values = tokens
        .ok_or_else(|| Error::Generation(format!("T5 config has no {field}")))?
        .iter();
    let token = values
        .next()
        .ok_or_else(|| Error::Generation(format!("T5 config {field} list is empty")))?;
    if values.next().is_some() {
        return Err(Error::Generation(format!(
            "T5 config {field} must contain exactly one token"
        )));
    }
    token_to_u32(token, field)
}

/// Narrows a validated model token ID to the public generation domain.
fn token_to_u32(token: usize, field: &str) -> Result<u32> {
    u32::try_from(token).map_err(|_| {
        Error::Generation(format!(
            "T5 config {field} token {token} exceeds the u32 token-ID domain"
        ))
    })
}

#[cfg(test)]
mod tests {
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
}
