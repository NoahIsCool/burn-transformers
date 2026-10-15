//! Qwen3.5, Qwen3.6, and Qwen3.8 text-transformer topology.
//!
//! The open Qwen3.8 checkpoint deliberately serializes `model_type =
//! "qwen3_5_text"`; it is the same architectural family, not a Rust subtype. Its
//! decoder alternates gated-delta-network (linear-attention) blocks with ordinary
//! causal attention. Unlike classic multi-head attention, `hidden_size` need not
//! equal `num_attention_heads * head_dim`: projections map between the residual
//! stream and an independently sized attention space.
//!
//! The runtime applies no stochastic attention dropout, including with an
//! autodiff backend. `attention_dropout` is checkpoint metadata rather than an
//! execution-mode switch. Compose stochastic regularization explicitly when
//! optimizing floating parameters or inserted modules.

pub mod model;

pub use model::{
    Qwen3_5DecodeState, Qwen3_5DecoderLayer, Qwen3_5ForCausalLm, Qwen3_5LayerCache, Qwen3_5Mtp,
    Qwen3_5MtpOutput, Qwen3_5MtpPrediction, Qwen3_5SpeculativeConfig, Qwen3_5SpeculativeDecoder,
    Qwen3_5SpeculativeOutput, Qwen3_5TextModel,
};

use serde::{Deserialize, Serialize};

use super::config::{
    ConfigError, ModelConfig, MropeHandling, RopeContext, RopeParameters, TokenIds, Validate,
    divisible, model_type, nonzero, positive_finite, probability, token_ids,
};
use crate::position::RopeConfig;

/// Fraction of each full-attention head that Qwen3.5 rotates when no config states one.
///
/// Transformers' `Qwen3_5TextConfig.__post_init__` applies it with
/// `kwargs.setdefault("partial_rotary_factor", 0.25)`, so it is the weakest
/// source: a nested `rope_parameters.partial_rotary_factor` and then a
/// top-level `partial_rotary_factor` both take precedence.
pub const QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR: f64 = 0.25;

mod checkpoint;
mod load;
mod recipe;

pub use checkpoint::{
    Qwen3_5CheckpointContract, Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource,
};
pub use load::Qwen3_5LoadError;
pub use recipe::{
    LoadedQwen3_5Backbone, Qwen3_5BackboneRecipe, Qwen3_5LayerRecipe, Qwen3_5RecipeLoadOptions,
};

/// Per-layer mixer selected by the serialized Qwen schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Qwen3_5LayerType {
    /// Gated delta-rule linear attention with convolutional local state.
    #[serde(rename = "linear_attention")]
    LinearAttention,
    /// Grouped-query causal softmax attention with a KV cache.
    #[serde(rename = "full_attention")]
    FullAttention,
}

/// Exact text-backbone schema shared by released Qwen3.5–Qwen3.8 checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Qwen3_5TextConfig {
    /// Hugging Face architecture discriminator; Qwen3.8 still uses `qwen3_5_text`.
    pub model_type: String,
    /// Number of rows in the token embedding and output vocabulary.
    pub vocab_size: usize,
    /// Width `H` of the residual stream.
    pub hidden_size: usize,
    /// Hidden width of each dense SwiGLU feed-forward network.
    pub intermediate_size: usize,
    /// Number of decoder blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads in full-attention blocks.
    pub num_attention_heads: usize,
    /// Number of shared key/value heads in full-attention blocks.
    pub num_key_value_heads: usize,
    /// Width of each full-attention query, key, and value head.
    pub head_dim: usize,
    /// Feed-forward activation; released checkpoints use SiLU in SwiGLU.
    pub hidden_act: String,
    /// Maximum indexed context length.
    pub max_position_embeddings: usize,
    /// Standard deviation used only when initializing new weights.
    pub initializer_range: f64,
    /// Epsilon in pre-attention, pre-MLP, and final RMS normalization.
    pub rms_norm_eps: f64,
    /// Whether inference should maintain recurrent/KV state.
    pub use_cache: bool,
    /// Whether token embedding and language-model projection share parameters.
    pub tie_word_embeddings: bool,
    /// Canonical modern RoPE object; `rope_scaling` is accepted as a legacy alias.
    #[serde(alias = "rope_scaling")]
    pub rope_parameters: Option<RopeParameters>,
    /// Legacy top-level RoPE theta used when the nested object omits it.
    pub rope_theta: Option<f64>,
    /// Legacy top-level fraction of each head to rotate.
    ///
    /// `None` means "not serialized". Resolution follows Transformers: the
    /// nested `rope_parameters.partial_rotary_factor` wins, then this field,
    /// then [`QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR`]. A JSON `null` is
    /// indistinguishable from an absent key and therefore also resolves to the
    /// default (Transformers would rotate whole heads for an explicit `null`;
    /// no released checkpoint serializes one).
    pub partial_rotary_factor: Option<f64>,
    /// Whether attention projections contain biases.
    pub attention_bias: bool,
    /// Dropout probability applied to attention weights.
    pub attention_dropout: f64,
    /// Whether each query projection also emits a head-wise output gate.
    pub attn_output_gate: bool,
    /// DeltaNet output-gate metadata used by released hybrid checkpoints.
    ///
    /// The pinned reference implements SiLU unconditionally. `swish` is accepted
    /// as the equivalent spelling serialized by Qwen3.8; any other value fails
    /// validation instead of being silently ignored.
    pub output_gate_type: Option<String>,
    /// Interval used to derive the three-linear/one-full schedule when omitted.
    pub full_attention_interval: usize,
    /// Explicit mixer selected for each decoder layer.
    pub layer_types: Vec<Qwen3_5LayerType>,
    /// Causal depthwise-convolution kernel in gated delta layers.
    pub linear_conv_kernel_dim: usize,
    /// Width of each gated-delta key/query head.
    pub linear_key_head_dim: usize,
    /// Width of each gated-delta value head.
    pub linear_value_head_dim: usize,
    /// Number of key/query heads in gated-delta layers.
    pub linear_num_key_heads: usize,
    /// Number of value heads in gated-delta layers.
    pub linear_num_value_heads: usize,
    /// Accumulator/cache dtype requested for the recurrent state.
    ///
    /// The current runtime deliberately keeps the association matrix in FP32,
    /// matching the released Qwen3.5 and Qwen3.8 checkpoints.
    pub mamba_ssm_dtype: Option<String>,
    /// Number of appended multi-token-prediction decoder layers.
    pub mtp_num_hidden_layers: usize,
    /// Whether MTP owns an embedding rather than reusing the target embedding.
    pub mtp_use_dedicated_embeddings: bool,
    /// Legacy dense-only layer indices retained for fail-closed config parsing.
    ///
    /// Qwen3.5 removes this inherited Qwen3-Next option and all released values
    /// are empty. Non-empty lists are rejected because the dense runtime has no
    /// alternate mixer-free decoder block.
    pub mlp_only_layers: Vec<usize>,
    /// Padding token, if the checkpoint defines one.
    pub pad_token_id: Option<TokenIds>,
    /// Beginning-of-sequence token, if defined.
    pub bos_token_id: Option<TokenIds>,
    /// One or more end-of-sequence tokens.
    pub eos_token_id: Option<TokenIds>,
}

impl Default for Qwen3_5TextConfig {
    /// Supplies Hugging Face's architecture defaults, with optional MTP disabled.
    fn default() -> Self {
        Self {
            model_type: "qwen3_5_text".into(),
            vocab_size: 248_320,
            hidden_size: 4096,
            intermediate_size: 12_288,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            hidden_act: "silu".into(),
            max_position_embeddings: 32_768,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            tie_word_embeddings: false,
            rope_parameters: None,
            rope_theta: None,
            partial_rotary_factor: None,
            attention_bias: false,
            attention_dropout: 0.0,
            attn_output_gate: true,
            output_gate_type: None,
            full_attention_interval: 4,
            layer_types: Vec::new(),
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            mamba_ssm_dtype: None,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            mlp_only_layers: Vec::new(),
            pad_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
        }
    }
}

impl Qwen3_5TextConfig {
    /// Returns an explicit schedule, deriving it from `full_attention_interval`
    /// when an older config omitted `layer_types`.
    pub fn resolved_layer_types(&self) -> Vec<Qwen3_5LayerType> {
        if !self.layer_types.is_empty() {
            return self.layer_types.clone();
        }
        let interval = self.full_attention_interval.max(1);
        (0..self.num_hidden_layers)
            .map(|index| {
                if (index + 1).is_multiple_of(interval) {
                    Qwen3_5LayerType::FullAttention
                } else {
                    Qwen3_5LayerType::LinearAttention
                }
            })
            .collect()
    }

    /// Returns the top-level partial rotary factor the RoPE translator should see.
    ///
    /// The Qwen default fills only a gap left by *both* sources, mirroring
    /// Transformers' `setdefault`: a nested value with no top-level value is
    /// used as is, instead of being compared against a synthetic default.
    fn top_level_partial_rotary_factor(&self) -> Option<f64> {
        let nested = self
            .rope_parameters
            .as_ref()
            .and_then(|parameters| parameters.partial_rotary_factor);
        self.partial_rotary_factor.or_else(|| {
            nested
                .is_none()
                .then_some(QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR)
        })
    }

    /// Translates the serialized RoPE metadata into the executable rotary configuration.
    ///
    /// This is the one place Qwen3.5 resolves `rope_parameters`, the legacy
    /// top-level `rope_theta` and `partial_rotary_factor`, and the family
    /// default above, through the shared
    /// [`RopeParameters::to_rope_config`] translator. Multimodal section
    /// metadata is accepted because the text runtime feeds identical temporal,
    /// height, and width positions, for which sectioned MRoPE equals ordinary
    /// one-dimensional RoPE.
    ///
    /// # Errors
    ///
    /// Returns a configuration error at `qwen3_5.rope.<field>` (or
    /// `qwen3_5.head_dim`) for a missing, out-of-range, conflicting, or unused
    /// RoPE field.
    pub fn rope_config(&self) -> crate::Result<RopeConfig> {
        self.rope_parameters
            .clone()
            .unwrap_or_default()
            .to_rope_config(
                &RopeContext::new("qwen3_5", self.head_dim, self.max_position_embeddings)
                    .with_rope_theta(self.rope_theta)
                    .with_partial_rotary_factor(self.top_level_partial_rotary_factor())
                    .with_mrope(MropeHandling::TextPositions),
            )
    }

    /// Returns the number of channels produced by a full-attention Q projection.
    ///
    /// Output-gated Qwen packs an equally sized gate beside Q, hence the factor of
    /// two. For Qwen3.8-27B this is `24 * 256 * 2 = 12_288` channels.
    pub fn query_projection_width(&self) -> usize {
        let gate_factor = if self.attn_output_gate { 2 } else { 1 };
        self.num_attention_heads * self.head_dim * gate_factor
    }
}

impl Validate for Qwen3_5TextConfig {
    /// Checks full-attention, gated-delta, RoPE, schedule, MTP, and token topology.
    fn validate(&self) -> Result<(), ConfigError> {
        /// Family label attached to configuration validation errors.
        const FAMILY: &str = "qwen3_5";
        model_type(FAMILY, &self.model_type, &["qwen3_5_text"])?;
        for (field, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("head_dim", self.head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
            ("full_attention_interval", self.full_attention_interval),
            ("linear_conv_kernel_dim", self.linear_conv_kernel_dim),
            ("linear_key_head_dim", self.linear_key_head_dim),
            ("linear_value_head_dim", self.linear_value_head_dim),
        ] {
            nonzero(FAMILY, field, value)?;
        }
        divisible(
            FAMILY,
            "num_attention_heads",
            self.num_attention_heads,
            "num_key_value_heads",
            self.num_key_value_heads,
        )?;
        divisible(
            FAMILY,
            "linear_num_value_heads",
            self.linear_num_value_heads,
            "linear_num_key_heads",
            self.linear_num_key_heads,
        )?;
        positive_finite(FAMILY, "initializer_range", self.initializer_range)?;
        positive_finite(FAMILY, "rms_norm_eps", self.rms_norm_eps)?;
        probability(FAMILY, "attention_dropout", self.attention_dropout)?;
        if self.hidden_act != "silu" {
            return Err(ConfigError::new(
                FAMILY,
                "hidden_act",
                "only the released SiLU/SwiGLU topology is supported",
            ));
        }
        if let Some(gate) = &self.output_gate_type
            && gate != "swish"
            && gate != "silu"
        {
            return Err(ConfigError::new(
                FAMILY,
                "output_gate_type",
                "the pinned DeltaNet graph supports only equivalent `swish` or `silu` spellings",
            ));
        }
        if let Some(dtype) = &self.mamba_ssm_dtype
            && dtype != "float32"
        {
            return Err(ConfigError::new(
                FAMILY,
                "mamba_ssm_dtype",
                "the recurrent association state is implemented only in float32",
            ));
        }

        let schedule = self.resolved_layer_types();
        if schedule.len() != self.num_hidden_layers {
            return Err(ConfigError::new(
                FAMILY,
                "layer_types",
                "length must equal num_hidden_layers",
            ));
        }
        if !self.mlp_only_layers.is_empty() {
            return Err(ConfigError::new(
                FAMILY,
                "mlp_only_layers",
                "Qwen3.5 removes the inherited MLP-only schedule; released configs must use an empty list",
            ));
        }
        if self.mtp_use_dedicated_embeddings {
            return Err(ConfigError::new(
                FAMILY,
                "mtp_use_dedicated_embeddings",
                "no released Qwen3.5 tensor contract defines a dedicated MTP embedding",
            ));
        }

        let mut rope = self.rope_parameters.clone().unwrap_or_default();
        // Only a serialized top-level value can disagree with the nested one;
        // the family default never does (see `top_level_partial_rotary_factor`).
        if let Some(factor) = self.top_level_partial_rotary_factor() {
            if let Some(nested) = rope.partial_rotary_factor
                && (nested - factor).abs() > f64::EPSILON
            {
                return Err(ConfigError::new(
                    FAMILY,
                    "partial_rotary_factor",
                    "top-level and rope_parameters values disagree",
                ));
            }
            rope.partial_rotary_factor = Some(factor);
        }
        rope.validate_for(
            FAMILY,
            self.head_dim,
            self.max_position_embeddings,
            self.rope_theta,
        )?;
        if let Some(sections) = &rope.mrope_section
            && sections.len() != 3
        {
            return Err(ConfigError::new(
                FAMILY,
                "mrope_section",
                "Qwen multimodal RoPE requires temporal, height, and width sections",
            ));
        }
        if self.mtp_num_hidden_layers > self.num_hidden_layers {
            return Err(ConfigError::new(
                FAMILY,
                "mtp_num_hidden_layers",
                "cannot exceed the target decoder depth",
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

impl ModelConfig for Qwen3_5TextConfig {
    /// Diagnostic family name used by generic configuration parsing.
    const FAMILY: &'static str = "qwen3_5";
    /// Checkpoint discriminator accepted after extracting a wrapped text configuration.
    const MODEL_TYPES: &'static [&'static str] = &["qwen3_5_text"];
}

/// Source-compatible alias that makes Qwen3.8 intent explicit at call sites.
pub type Qwen38TextConfig = Qwen3_5TextConfig;

#[cfg(test)]
#[path = "qwen3_5_tests.rs"]
mod tests;
