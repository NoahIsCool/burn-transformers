//! GPT-NeoX/Pythia transformer graphs for activation experiments.
//!
//! Pythia uses affine LayerNorm, a dense GELU MLP, partial split-half RoPE,
//! and parallel attention/MLP residual branches. Parameters remain ordinary
//! Burn modules; this stateless implementation has no autoregressive cache.
//! The initial checkpoint target is `EleutherAI/pythia-160m-deduped-v0`.
//!
//! Execution is deterministic: source dropout settings are not represented by
//! this configuration or applied to the graph. Autodiff can optimize its
//! floating parameters, but does not enable the source model's training-time
//! regularization. Compose any stochastic training operations explicitly.
//!
//! With residual width `d`, each head has width `d / num_attention_heads`.
//! Only the configured even prefix of each Q/K head receives split-half RoPE;
//! unrotated channels remain unchanged. Parallel residual mode computes both
//! branches from separately normalized copies of the same residual, while sequential
//! mode normalizes the attention-updated residual before its MLP. Exact GELU is
//! required; this graph does not infer activations or rotary choices from weights.
//! The historical Pythia checkpoint and reference equations are pinned in
//! `examples/mechanistic/tuned_lens/README.md`.

use serde::{Deserialize, Serialize};

use crate::{
    Error, Result,
    position::{RopeConfig, RopeScaling},
};

mod model;
pub use model::{GptNeoXLayer, GptNeoXModel};
mod load;
pub use load::{GptNeoXCheckpointContract, GptNeoXCheckpointLoadError, GptNeoXLoadError};

// Source topology, rotary dimensions, and allocation preflight.

/// Serialized GPT-NeoX topology. Defaults match Pythia-160M.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GptNeoXConfig {
    /// Hugging Face architecture discriminator.
    pub model_type: String,
    /// Number of embedding and vocabulary-projection rows.
    pub vocab_size: usize,
    /// Width of the residual stream.
    pub hidden_size: usize,
    /// Number of decoder blocks.
    pub num_hidden_layers: usize,
    /// Equal numbers of query, key, and value heads.
    pub num_attention_heads: usize,
    /// Dense feed-forward intermediate width.
    pub intermediate_size: usize,
    /// Maximum stateless sequence length and position index plus one.
    pub max_position_embeddings: usize,
    /// Fraction of each head rotated; the resulting dimension must be even.
    pub rotary_pct: f64,
    /// Rotary inverse-frequency base.
    pub rotary_emb_base: f64,
    /// Positive LayerNorm variance stabilizer.
    pub layer_norm_eps: f64,
    /// Supported nonlinearity: exact `gelu`.
    pub hidden_act: String,
    /// Whether attention and MLP read separate normalizations of the same input.
    pub use_parallel_residual: bool,
    /// This implementation requires independent embedding and vocabulary weights.
    pub tie_word_embeddings: bool,
    /// Standard deviation for newly initialized weight matrices.
    pub initializer_range: f64,
}

impl Default for GptNeoXConfig {
    /// Describes Pythia-160M with parallel residual branches and one-quarter-head rotary encoding.
    fn default() -> Self {
        Self {
            model_type: "gpt_neox".into(),
            vocab_size: 50_304,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 2048,
            rotary_pct: 0.25,
            rotary_emb_base: 10_000.0,
            layer_norm_eps: 1e-5,
            hidden_act: "gelu".into(),
            use_parallel_residual: true,
            tie_word_embeddings: false,
            initializer_range: 0.02,
        }
    }
}

impl GptNeoXConfig {
    /// Checks supported mathematical choices and dimensions before allocation.
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "gpt_neox" {
            return Err(Error::configuration(
                "gpt_neox.model_type",
                "expected gpt_neox",
            ));
        }
        for (field, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("intermediate_size", self.intermediate_size),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if value == 0 || value > i32::MAX as usize {
                return Err(Error::configuration(
                    format!("gpt_neox.{field}"),
                    "must be nonzero and fit a signed 32-bit tensor index",
                ));
            }
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(Error::configuration(
                "gpt_neox.hidden_size",
                "must be divisible by num_attention_heads",
            ));
        }
        for (field, value) in [
            ("layer_norm_eps", self.layer_norm_eps),
            ("initializer_range", self.initializer_range),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::configuration(
                    format!("gpt_neox.{field}"),
                    "must be finite and positive",
                ));
            }
        }
        if self.hidden_act != "gelu" {
            return Err(Error::configuration(
                "gpt_neox.hidden_act",
                "this recipe requires exact gelu",
            ));
        }
        if self.tie_word_embeddings {
            return Err(Error::configuration(
                "gpt_neox.tie_word_embeddings",
                "tied GPT-NeoX embeddings are not implemented",
            ));
        }
        if self.hidden_size.checked_mul(self.vocab_size).is_none()
            || self
                .hidden_size
                .checked_mul(self.intermediate_size)
                .is_none()
            || self
                .hidden_size
                .checked_mul(self.hidden_size)
                .and_then(|n| n.checked_mul(3))
                .is_none()
        {
            return Err(Error::configuration(
                "gpt_neox",
                "parameter dimensions overflow usize",
            ));
        }
        self.rope()?;
        Ok(())
    }

    /// Returns the width of one attention head after validation.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads.max(1)
    }

    /// Builds the validated even partial rotary width from the configured head width and fraction.
    pub(super) fn rope(&self) -> Result<RopeConfig> {
        RopeConfig::from_partial_factor(
            self.head_dim(),
            self.rotary_pct,
            self.rotary_emb_base,
            RopeScaling::None,
        )
        .map_err(|error| Error::configuration("gpt_neox.rotary", error.to_string()))
    }
}

#[cfg(test)]
mod tests;
