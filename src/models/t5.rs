//! T5, FLAN-T5, T5 v1.1, and UL2 encoder-decoder topology.
//!
//! T5 attention projects the residual width `d_model` into an independent inner
//! width `num_heads * d_kv`; these values are intentionally not constrained to be
//! equal. Only the first block in each stack owns the learned relative-position
//! bias table, whose logarithmic bucket count and maximum distance determine all
//! later attention biases. FLAN changes training, while UL2 and T5 v1.1 primarily
//! select gated feed-forward variants, so one exact schema covers the family.
//!
//! The runtime uses a deterministic graph, including with an autodiff backend.
//! `dropout_rate` and `classifier_dropout` preserve checkpoint metadata; they do
//! not currently activate stochastic dropout. Experiments must compose their
//! training-time regularization explicitly.

use serde::{Deserialize, Serialize};

use super::config::{
    ConfigError, ModelConfig, TokenIds, Validate, model_type, nonzero, positive_finite,
    probability, token_ids,
};

mod checkpoint;
mod generation;
mod load;
mod model;

pub use checkpoint::{T5CheckpointContract, T5CheckpointPlan};
pub use generation::{T5BeamSearchOptions, generate_t5_beam_search};
pub use load::{T5CheckpointLoadError, T5LoadError};
pub use model::{
    T5Attention, T5Block, T5DecodeState, T5Encoder, T5EncoderLayerSpec, T5EncoderRecipe,
    T5ForConditionalGeneration, T5LayerCrossAttention, T5LayerFeedForward, T5LayerSelfAttention,
    T5Stack,
};

/// Feed-forward projection layout and activation stored by T5 checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum T5FeedForwardProjection {
    /// Original T5's single ReLU branch.
    #[serde(rename = "relu")]
    Relu,
    /// Single GELU branch supported by compatible T5 checkpoints.
    #[serde(rename = "gelu")]
    Gelu,
    /// Two input projections using T5's historical `gelu_new` approximation.
    #[serde(rename = "gated-gelu")]
    GatedGelu,
    /// Explicit gated-`gelu_new` spelling used by compatible custom configs.
    #[serde(rename = "gated-gelu_new")]
    GatedGeluNew,
    /// Two input projections multiplied after SiLU, as used by UL2.
    #[serde(rename = "gated-silu")]
    GatedSilu,
}

impl T5FeedForwardProjection {
    /// Reports whether the FFN owns two input matrices (`wi_0` and `wi_1`).
    pub fn is_gated(self) -> bool {
        matches!(self, Self::GatedGelu | Self::GatedGeluNew | Self::GatedSilu)
    }

    /// Returns the activation name used to validate historic derived fields.
    pub fn activation(self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::Gelu => "gelu",
            Self::GatedGelu | Self::GatedGeluNew => "gelu_new",
            Self::GatedSilu => "silu",
        }
    }
}

/// Serialized T5-family encoder-decoder architecture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct T5Config {
    /// Hugging Face discriminator; FLAN and UL2 both serialize `t5`.
    pub model_type: String,
    /// Shared encoder/decoder token-embedding vocabulary size.
    pub vocab_size: usize,
    /// Residual-stream and shared-embedding width.
    #[serde(alias = "hidden_size")]
    pub d_model: usize,
    /// Width of each query/key/value head.
    #[serde(alias = "head_dim")]
    pub d_kv: usize,
    /// Feed-forward hidden width.
    #[serde(alias = "intermediate_size")]
    pub d_ff: usize,
    /// Encoder block count.
    #[serde(alias = "num_hidden_layers")]
    pub num_layers: usize,
    /// Decoder block count; omitted configurations are symmetric with the encoder.
    pub num_decoder_layers: Option<usize>,
    /// Attention head count in both stacks.
    #[serde(alias = "num_attention_heads")]
    pub num_heads: usize,
    /// Number of learned relative-position buckets.
    pub relative_attention_num_buckets: usize,
    /// Distance after which relative positions share the furthest log bucket.
    pub relative_attention_max_distance: usize,
    /// Dropout used throughout attention, FFN, and residual branches.
    pub dropout_rate: f64,
    /// Epsilon in T5's scale-only RMS normalization.
    pub layer_norm_epsilon: f64,
    /// Global multiplier for T5's variance-aware weight initialization.
    pub initializer_factor: f64,
    /// Dense or gated feed-forward matrix layout and activation.
    pub feed_forward_proj: T5FeedForwardProjection,
    /// Historic derived activation field emitted by early UL2 configs.
    pub dense_act_fn: Option<String>,
    /// Historic derived gating flag emitted by early UL2 configs.
    pub is_gated_act: Option<bool>,
    /// Must remain true because decoder blocks contain cross-attention.
    pub is_encoder_decoder: bool,
    /// Whether autoregressive decoder attention returns/accepts a KV cache.
    pub use_cache: bool,
    /// Whether the vocabulary head shares the embedding and uses T5 output scaling.
    ///
    /// Original T5 sets this to true. T5 v1.1, FLAN, and UL2 checkpoints set it
    /// to false and store an independent `lm_head.weight`.
    pub tie_word_embeddings: bool,
    /// First decoder token, conventionally the padding token zero.
    pub decoder_start_token_id: Option<usize>,
    /// Legacy advisory input length; relative attention is not table-limited by it.
    pub n_positions: Option<usize>,
    /// Padding token ID.
    pub pad_token_id: Option<TokenIds>,
    /// End-of-sequence token ID or IDs.
    pub eos_token_id: Option<TokenIds>,
    /// Optional beginning-of-sequence token ID.
    pub bos_token_id: Option<TokenIds>,
    /// Dropout used by optional classification heads.
    pub classifier_dropout: Option<f64>,
}

impl Default for T5Config {
    /// Supplies the original T5-small architecture defaults.
    fn default() -> Self {
        Self {
            model_type: "t5".into(),
            vocab_size: 32_128,
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 6,
            num_decoder_layers: None,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: T5FeedForwardProjection::Relu,
            dense_act_fn: None,
            is_gated_act: None,
            is_encoder_decoder: true,
            use_cache: true,
            tie_word_embeddings: true,
            decoder_start_token_id: Some(0),
            n_positions: None,
            pad_token_id: Some(TokenIds::One(0)),
            eos_token_id: Some(TokenIds::One(1)),
            bos_token_id: None,
            classifier_dropout: Some(0.0),
        }
    }
}

impl T5Config {
    /// Returns decoder depth after applying T5's symmetric-stack default.
    pub fn decoder_layers(&self) -> usize {
        self.num_decoder_layers.unwrap_or(self.num_layers)
    }

    /// Returns the projected Q/K/V width `num_heads * d_kv`.
    pub fn attention_inner_dim(&self) -> usize {
        self.num_heads.saturating_mul(self.d_kv)
    }
}

impl Validate for T5Config {
    /// Checks both stack depths, independent attention width, buckets, FFN, and IDs.
    fn validate(&self) -> Result<(), ConfigError> {
        const FAMILY: &str = "t5";
        model_type(FAMILY, &self.model_type, &["t5"])?;
        for (field, value) in [
            ("vocab_size", self.vocab_size),
            ("d_model", self.d_model),
            ("d_kv", self.d_kv),
            ("d_ff", self.d_ff),
            ("num_layers", self.num_layers),
            ("num_decoder_layers", self.decoder_layers()),
            ("num_heads", self.num_heads),
            (
                "relative_attention_num_buckets",
                self.relative_attention_num_buckets,
            ),
            (
                "relative_attention_max_distance",
                self.relative_attention_max_distance,
            ),
        ] {
            nonzero(FAMILY, field, value)?;
        }
        if self.relative_attention_num_buckets < 4 {
            return Err(ConfigError::new(
                FAMILY,
                "relative_attention_num_buckets",
                "must contain at least four buckets for bidirectional encoder attention",
            ));
        }
        if !self.relative_attention_num_buckets.is_multiple_of(2) {
            return Err(ConfigError::new(
                FAMILY,
                "relative_attention_num_buckets",
                "must be even so bidirectional encoder buckets split by sign",
            ));
        }
        let encoder_exact_buckets = self.relative_attention_num_buckets / 4;
        if self.relative_attention_max_distance <= encoder_exact_buckets {
            return Err(ConfigError::new(
                FAMILY,
                "relative_attention_max_distance",
                "must exceed the encoder's exact-distance bucket region",
            ));
        }
        if !self.is_encoder_decoder {
            return Err(ConfigError::new(
                FAMILY,
                "is_encoder_decoder",
                "must be true because T5 decoder blocks contain cross-attention",
            ));
        }
        let attention_inner_dim = self.num_heads.checked_mul(self.d_kv);
        for (field, product) in [
            ("attention_inner_dim", attention_inner_dim),
            (
                "attention_projection_elements",
                attention_inner_dim.and_then(|inner| inner.checked_mul(self.d_model)),
            ),
            (
                "embedding_elements",
                self.vocab_size.checked_mul(self.d_model),
            ),
            ("feed_forward_elements", self.d_model.checked_mul(self.d_ff)),
            (
                "relative_attention_bias_elements",
                self.relative_attention_num_buckets
                    .checked_mul(self.num_heads),
            ),
        ] {
            if product.is_none() {
                return Err(ConfigError::new(
                    FAMILY,
                    field,
                    "dimension product overflows usize",
                ));
            }
        }
        positive_finite(FAMILY, "layer_norm_epsilon", self.layer_norm_epsilon)?;
        positive_finite(FAMILY, "initializer_factor", self.initializer_factor)?;
        probability(FAMILY, "dropout_rate", self.dropout_rate)?;
        if let Some(dropout) = self.classifier_dropout {
            probability(FAMILY, "classifier_dropout", dropout)?;
        }
        if let Some(position_count) = self.n_positions {
            nonzero(FAMILY, "n_positions", position_count)?;
        }
        if let Some(start) = self.decoder_start_token_id
            && start >= self.vocab_size
        {
            return Err(ConfigError::new(
                FAMILY,
                "decoder_start_token_id",
                "must index the shared vocabulary",
            ));
        }
        if let Some(activation) = &self.dense_act_fn
            && activation != self.feed_forward_proj.activation()
        {
            return Err(ConfigError::new(
                FAMILY,
                "dense_act_fn",
                "disagrees with feed_forward_proj",
            ));
        }
        if let Some(gated) = self.is_gated_act
            && gated != self.feed_forward_proj.is_gated()
        {
            return Err(ConfigError::new(
                FAMILY,
                "is_gated_act",
                "disagrees with feed_forward_proj",
            ));
        }
        token_ids(
            FAMILY,
            "pad_token_id",
            self.pad_token_id.as_ref(),
            self.vocab_size,
        )?;
        token_ids(
            FAMILY,
            "bos_token_id",
            self.bos_token_id.as_ref(),
            self.vocab_size,
        )?;
        token_ids(
            FAMILY,
            "eos_token_id",
            self.eos_token_id.as_ref(),
            self.vocab_size,
        )?;
        Ok(())
    }
}

impl ModelConfig for T5Config {
    const FAMILY: &'static str = "t5";
    const MODEL_TYPES: &'static [&'static str] = &["t5"];
}

/// FLAN changes instruction tuning, not T5's serialized transformer topology.
pub type FlanT5Config = T5Config;

/// UL2 checkpoints use the same schema with a gated-SiLU feed-forward projection.
pub type Ul2Config = T5Config;

#[cfg(test)]
#[path = "t5_tests.rs"]
mod tests;
