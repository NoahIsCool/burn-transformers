//! Muse Glimmer's dense hybrid text decoder.
//!
//! The released decoder is dense, but its attention schedule is hybrid: three
//! sliding-window layers are followed by one full-attention layer, counted backward
//! so the final layer is full attention. Full-attention layers use NoPE, represented
//! by a per-layer theta of zero. Sliding layers retain RoPE. The default derives
//! that exact coupling, while explicit equal-length vectors remain available for
//! controlled architecture experiments.
//!
//! The executable family implementation deliberately excludes image and video
//! towers. Its strict checkpoint contract consumes the nested language model and
//! vocabulary head while reporting every visual tensor through an explicit
//! allowlist. Cache state remains external to the Burn parameter graph: local
//! layers use mirrored sliding rings and global NoPE layers retain full KV history.
//! Stateless and cached attention share a validated query-chunk policy that
//! bounds score-like workspace without approximating any softmax row. The
//! reference eager operation remains unchanged for short single-chunk calls.
//!
//! # Packed inference
//!
//! The automatic registry accepts provenance-bound Q8/Q4 Burnpack overlays for
//! `muse_glimmer_text`. Every attention projection (`q`, `k`, `v`, gate, and
//! output), all three SwiGLU projections, and an independent LM head execute
//! through the crate's packed-preserving linear boundary. Embeddings,
//! normalization scales, and optional attention biases remain at source
//! precision. Registration records packed-kernel execution evidence; it does
//! not advertise Muse as release-supported or imply quantized-quality parity.

mod checkpoint;
mod load;
pub mod model;
/// Dependency-free hybrid layer schedule and cache-slot planning (Verus-verified).
pub(crate) mod schedule;

pub use checkpoint::{
    MuseGlimmerCheckpointContract, MuseGlimmerCheckpointOptions, MuseGlimmerCheckpointSource,
};
pub use load::{MuseGlimmerCheckpointLoadError, MuseGlimmerLoadError};
pub use model::{
    MuseGlimmerAttention, MuseGlimmerDecodeState, MuseGlimmerDecoderLayer, MuseGlimmerForCausalLm,
    MuseGlimmerLayerCache, MuseGlimmerModel,
};

use serde::{Deserialize, Serialize};

use super::config::{
    ConfigError, ModelConfig, RopeParameters, TokenIds, Validate, divisible, model_type, nonzero,
    positive_finite, probability, token_ids,
};

/// Attention implementation selected for a Muse Glimmer decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MuseGlimmerLayerType {
    /// Causal attention restricted to `sliding_window` preceding tokens.
    #[serde(rename = "sliding_attention")]
    SlidingAttention,
    /// Full causal attention over the available context.
    #[serde(rename = "full_attention")]
    FullAttention,
}

/// Text-backbone schema extracted from a Muse Glimmer multimodal config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MuseGlimmerTextConfig {
    /// Hugging Face model discriminator for the nested language backbone.
    pub model_type: String,
    /// Vocabulary and embedding row count.
    pub vocab_size: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// SwiGLU hidden width.
    pub intermediate_size: usize,
    /// Number of decoder blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of shared key/value heads.
    pub num_key_value_heads: usize,
    /// Width of each query/key/value head.
    pub head_dim: usize,
    /// SwiGLU gate activation; `hidden_act` is accepted as an alias.
    #[serde(alias = "hidden_act")]
    pub hidden_activation: String,
    /// Maximum indexed context length.
    pub max_position_embeddings: usize,
    /// Upstream initialization standard deviation; validated as positive and
    /// finite, but never read.
    ///
    /// Fresh linear and embedding weights use Burn's default initializers,
    /// and every RMS norm starts at unit effective scale: zero offsets for the
    /// `1 + weight` sublayer norms and ones for the final norm.
    pub initializer_range: f64,
    /// Epsilon in pre-sublayer RMS normalization.
    pub rms_norm_eps: f64,
    /// Hugging Face generation hint; ignored here, where callers own [`MuseGlimmerDecodeState`].
    pub use_cache: bool,
    /// Whether token embedding and LM head share parameters.
    pub tie_word_embeddings: bool,
    /// Global RoPE parameters for layers whose per-layer theta is nonzero.
    #[serde(alias = "rope_scaling")]
    pub rope_parameters: Option<RopeParameters>,
    /// Whether attention projections include bias.
    pub attention_bias: bool,
    /// Attention probability dropout; inference rejects any nonzero value.
    pub attention_dropout: f64,
    /// Local receptive-field width of sliding layers.
    pub sliding_window: Option<usize>,
    /// Explicit full/sliding layer schedule.
    pub layer_types: Vec<MuseGlimmerLayerType>,
    /// Theta per layer; zero disables rotary position encoding (NoPE).
    pub layer_rope_theta: Vec<f64>,
    /// Tanh soft-cap applied to final vocabulary logits.
    pub final_logit_softcapping: f64,
    /// Extra query multiplier after QK normalization.
    pub qk_scale_factor: f64,
    /// Multiplier applied before the final tanh soft-cap.
    pub output_multiplier: f64,
    /// Epsilon in post-attention and post-FFN normalization.
    pub post_norm_eps: f64,
    /// Optional padding token ID.
    pub pad_token_id: Option<TokenIds>,
    /// Beginning-of-sequence token ID.
    pub bos_token_id: Option<TokenIds>,
    /// End-of-sequence token ID or IDs.
    pub eos_token_id: Option<TokenIds>,
}

impl Default for MuseGlimmerTextConfig {
    /// Supplies defaults published for Muse-Glimmer-30B.
    fn default() -> Self {
        Self {
            model_type: "muse_glimmer_text".into(),
            vocab_size: 202_048,
            hidden_size: 6656,
            intermediate_size: 19_968,
            num_hidden_layers: 52,
            num_attention_heads: 32,
            num_key_value_heads: 2,
            head_dim: 128,
            hidden_activation: "silu".into(),
            max_position_embeddings: 131_072,
            initializer_range: 0.02,
            rms_norm_eps: 1e-5,
            use_cache: true,
            tie_word_embeddings: false,
            rope_parameters: Some(RopeParameters {
                rope_theta: Some(500_000.0),
                ..RopeParameters::default()
            }),
            attention_bias: false,
            attention_dropout: 0.0,
            sliding_window: Some(2048),
            layer_types: Vec::new(),
            layer_rope_theta: Vec::new(),
            final_logit_softcapping: 20.0,
            qk_scale_factor: 3.87,
            output_multiplier: 0.196_116_135_138_184_04,
            post_norm_eps: 1e-8,
            pad_token_id: None,
            bos_token_id: Some(TokenIds::One(200_000)),
            eos_token_id: Some(TokenIds::One(200_001)),
        }
    }
}

impl MuseGlimmerTextConfig {
    /// Produces the documented backward-counted three-sliding/one-full schedule.
    ///
    /// The derived schedule (`schedule::derived_global_layers`, verified in
    /// `verus/deepseek_muse.rs`) makes layer `i` global exactly when
    /// `(n - 1 - i) mod 4 = 0`, `ceil(n / 4)` of the `n` layers.
    pub fn resolved_layer_types(&self) -> Vec<MuseGlimmerLayerType> {
        if !self.layer_types.is_empty() {
            return self.layer_types.clone();
        }
        schedule::derived_global_layers(self.num_hidden_layers)
            .into_iter()
            .map(|global| {
                if global {
                    MuseGlimmerLayerType::FullAttention
                } else {
                    MuseGlimmerLayerType::SlidingAttention
                }
            })
            .collect()
    }

    /// Returns the global RoPE base: `rope_parameters.rope_theta`, else Transformers' default.
    ///
    /// The reference (`MuseGlimmerTextConfig.__post_init__` and its rotary
    /// embedding) reads `rope_parameters["rope_theta"]`, which Transformers'
    /// `RotaryEmbeddingConfigMixin` fills with its `default_theta` of
    /// `10_000` when absent. Validation, the default per-layer schedule, and
    /// the rotary translator all use this one resolver.
    #[must_use]
    pub fn global_rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|rope| rope.rope_theta)
            .unwrap_or(super::config::DEFAULT_ROPE_THETA)
    }

    /// Produces per-layer theta, setting NoPE full-attention layers to zero.
    ///
    /// The reference uses each entry only as a RoPE/NoPE switch and rotates
    /// every RoPE layer with [`Self::global_rope_theta`]; the derived default
    /// schedule therefore repeats the global theta. Explicit nonzero entries
    /// that differ from it are honored as an experiment extension.
    pub fn resolved_layer_rope_theta(&self) -> Vec<f64> {
        if !self.layer_rope_theta.is_empty() {
            return self.layer_rope_theta.clone();
        }
        let global_theta = self.global_rope_theta();
        self.resolved_layer_types()
            .iter()
            .map(|kind| match kind {
                MuseGlimmerLayerType::FullAttention => 0.0,
                MuseGlimmerLayerType::SlidingAttention => global_theta,
            })
            .collect()
    }

    /// Returns the configured local window only when the resolved schedule uses it.
    ///
    /// Full-attention-only experiments do not allocate a ring cache and therefore
    /// must not acquire an otherwise meaningless sliding-window requirement.
    #[must_use]
    pub fn resolved_sliding_window(&self) -> Option<usize> {
        if self
            .resolved_layer_types()
            .contains(&MuseGlimmerLayerType::SlidingAttention)
        {
            self.sliding_window
        } else {
            None
        }
    }
}

impl Validate for MuseGlimmerTextConfig {
    /// Checks head grouping, backward schedule, NoPE coupling, scales, and tokens.
    fn validate(&self) -> Result<(), ConfigError> {
        /// Family label attached to configuration diagnostics.
        const FAMILY: &str = "muse_glimmer";
        model_type(FAMILY, &self.model_type, &["muse_glimmer_text"])?;
        for (field, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("head_dim", self.head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
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
        if !self.head_dim.is_multiple_of(2) {
            return Err(ConfigError::new(
                FAMILY,
                "head_dim",
                "RoPE rotates pairs, so head_dim must be even",
            ));
        }
        if self.hidden_activation != "silu" {
            return Err(ConfigError::new(
                FAMILY,
                "hidden_activation",
                "released Muse Glimmer text blocks require SiLU/SwiGLU",
            ));
        }
        positive_finite(FAMILY, "initializer_range", self.initializer_range)?;
        positive_finite(FAMILY, "rms_norm_eps", self.rms_norm_eps)?;
        positive_finite(FAMILY, "post_norm_eps", self.post_norm_eps)?;
        positive_finite(
            FAMILY,
            "final_logit_softcapping",
            self.final_logit_softcapping,
        )?;
        positive_finite(FAMILY, "qk_scale_factor", self.qk_scale_factor)?;
        positive_finite(FAMILY, "output_multiplier", self.output_multiplier)?;
        probability(FAMILY, "attention_dropout", self.attention_dropout)?;

        let schedule = self.resolved_layer_types();
        let theta = self.resolved_layer_rope_theta();
        if schedule.len() != self.num_hidden_layers || theta.len() != self.num_hidden_layers {
            return Err(ConfigError::new(
                FAMILY,
                "layer_types",
                "layer_types and layer_rope_theta must match num_hidden_layers",
            ));
        }
        if schedule.contains(&MuseGlimmerLayerType::SlidingAttention) {
            let window = self.sliding_window.ok_or_else(|| {
                ConfigError::new(
                    FAMILY,
                    "sliding_window",
                    "is required by sliding-attention layers",
                )
            })?;
            nonzero(FAMILY, "sliding_window", window)?;
            if window > self.max_position_embeddings {
                return Err(ConfigError::new(
                    FAMILY,
                    "sliding_window",
                    "cannot exceed max_position_embeddings",
                ));
            }
        }
        for value in theta {
            if !value.is_finite() || value < 0.0 {
                return Err(ConfigError::new(
                    FAMILY,
                    "layer_rope_theta",
                    "values must be finite and non-negative; zero denotes NoPE",
                ));
            }
        }
        self.rope_parameters
            .clone()
            .unwrap_or_default()
            .validate_for(
                FAMILY,
                self.head_dim,
                self.max_position_embeddings,
                Some(self.global_rope_theta()),
            )?;
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

impl ModelConfig for MuseGlimmerTextConfig {
    /// Canonical family name used by configuration dispatch.
    const FAMILY: &'static str = "muse_glimmer";
    /// Accepted discriminator for the text-only configuration schema.
    const MODEL_TYPES: &'static [&'static str] = &["muse_glimmer_text"];
}

#[cfg(test)]
#[path = "muse_glimmer_tests.rs"]
mod tests;
