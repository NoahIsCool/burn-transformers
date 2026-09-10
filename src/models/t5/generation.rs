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
#[path = "generation_tests.rs"]
mod tests;
