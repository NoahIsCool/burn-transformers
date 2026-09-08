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

mod checkpoint;
mod load;
pub mod model;

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
    /// New-weight initialization standard deviation.
    pub initializer_range: f64,
    /// Epsilon in pre-sublayer RMS normalization.
    pub rms_norm_eps: f64,
    /// Whether generation maintains KV caches.
    pub use_cache: bool,
    /// Whether token embedding and LM head share parameters.
    pub tie_word_embeddings: bool,
    /// Global RoPE parameters for layers whose per-layer theta is nonzero.
    #[serde(alias = "rope_scaling")]
    pub rope_parameters: Option<RopeParameters>,
    /// Whether attention projections include bias.
    pub attention_bias: bool,
    /// Attention probability dropout.
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
    pub fn resolved_layer_types(&self) -> Vec<MuseGlimmerLayerType> {
        if !self.layer_types.is_empty() {
            return self.layer_types.clone();
        }
        (0..self.num_hidden_layers)
            .map(|index| {
                if (self.num_hidden_layers - 1 - index).is_multiple_of(4) {
                    MuseGlimmerLayerType::FullAttention
                } else {
                    MuseGlimmerLayerType::SlidingAttention
                }
            })
            .collect()
    }

    /// Produces per-layer theta, setting NoPE full-attention layers to zero.
    pub fn resolved_layer_rope_theta(&self) -> Vec<f64> {
        if !self.layer_rope_theta.is_empty() {
            return self.layer_rope_theta.clone();
        }
        let global_theta = self
            .rope_parameters
            .as_ref()
            .and_then(|rope| rope.rope_theta)
            .unwrap_or(10_000.0);
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
                Some(500_000.0),
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
    const FAMILY: &'static str = "muse_glimmer";
    const MODEL_TYPES: &'static [&'static str] = &["muse_glimmer_text"];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_schedule_is_three_rotary_local_layers_then_one_nope_global_layer() {
        let config = MuseGlimmerTextConfig::default();
        let schedule = config.resolved_layer_types();
        let theta = config.resolved_layer_rope_theta();
        assert_eq!(schedule.len(), 52);
        assert_eq!(theta.len(), 52);
        for (index, (layer_type, theta)) in schedule.into_iter().zip(theta).enumerate() {
            if index % 4 == 3 {
                assert_eq!(layer_type, MuseGlimmerLayerType::FullAttention);
                assert_eq!(theta, 0.0);
            } else {
                assert_eq!(layer_type, MuseGlimmerLayerType::SlidingAttention);
                assert_eq!(theta, 500_000.0);
            }
        }
    }

    /// Explicit head dimensions and full-only schedules remain independent of a window.
    #[test]
    fn explicit_schedule_and_head_width_allow_architecture_experiments() {
        let config = MuseGlimmerTextConfig {
            hidden_size: 10,
            num_hidden_layers: 4,
            num_attention_heads: 3,
            num_key_value_heads: 1,
            head_dim: 4,
            sliding_window: None,
            layer_types: vec![MuseGlimmerLayerType::FullAttention; 4],
            layer_rope_theta: vec![500_000.0; 4],
            ..MuseGlimmerTextConfig::default()
        };
        config.validate().unwrap();
        assert_eq!(config.resolved_sliding_window(), None);

        let unused_window = MuseGlimmerTextConfig {
            sliding_window: Some(0),
            ..config
        };
        unused_window.validate().unwrap();
        assert_eq!(unused_window.resolved_sliding_window(), None);
    }

    /// A local layer still requires a nonzero cache and mask window.
    #[test]
    fn sliding_schedule_requires_a_valid_window() {
        let config = MuseGlimmerTextConfig {
            num_hidden_layers: 1,
            sliding_window: None,
            layer_types: vec![MuseGlimmerLayerType::SlidingAttention],
            layer_rope_theta: vec![500_000.0],
            ..MuseGlimmerTextConfig::default()
        };
        assert!(config.validate().is_err());

        let config = MuseGlimmerTextConfig {
            sliding_window: Some(0),
            ..config
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_a_negative_per_layer_rope_theta() {
        let config = MuseGlimmerTextConfig {
            num_hidden_layers: 4,
            layer_rope_theta: vec![500_000.0, 500_000.0, 500_000.0, -1.0],
            ..MuseGlimmerTextConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
