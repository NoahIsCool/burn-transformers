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
//!
//! [`GptNeoXForCausalLm`] (formerly `GptNeoXModel`, still available under that
//! name) implements [`crate::architecture::HookedModel`] and
//! [`crate::architecture::StagedModel`], so generic activation caches and
//! residual patching run on it unchanged. Random initialization follows
//! Hugging Face ([`GptNeoXConfig::init_policy`]).

use serde::{Deserialize, Serialize};

use crate::{
    Error, Result,
    initialization::InitPolicy,
    models::{
        config::{ConfigError, ModelConfig, Validate, layer_count, model_type, positive_finite},
        gpt2::extents::{first_invalid_extent, product_fits},
    },
    position::{RopeConfig, RopeScaling},
};

mod model;
pub use model::{GptNeoXForCausalLm, GptNeoXLayer, GptNeoXModel};
mod load;
mod widths;
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
    ///
    /// The same checks as [`Validate::validate`], reported as a crate error at
    /// `gpt_neox.<field>`.
    pub fn validate(&self) -> Result<()> {
        Validate::validate(self).map_err(Error::from)
    }

    /// GPT-NeoX's random-initialization scheme as an explicit, auditable policy.
    ///
    /// Following Hugging Face `GPTNeoXPreTrainedModel._init_weights`: the
    /// input embedding and every linear weight (including the untied
    /// vocabulary projection) draw `N(0, initializer_range^2)` and projection
    /// biases are zero. GPT-NeoX applies no depth scaling.
    /// [`GptNeoXForCausalLm::init`] constructs parameters lazily with exactly
    /// these distributions, and builds every LayerNorm with scale one and
    /// shift zero as Hugging Face does.
    ///
    /// The policy itself covers embeddings, projections, and projection biases
    /// only: LayerNorm scales and shifts have the role
    /// [`ParameterRole::Other`](crate::initialization::ParameterRole::Other),
    /// which the policy keeps, because [`ParameterInit`](crate::initialization::ParameterInit)
    /// cannot express "ones". Applying it to a trained or edited model
    /// therefore re-samples the embeddings and projections and zeroes the
    /// biases but leaves the LayerNorm parameters as they are; reset those
    /// separately (for example by rebuilding each norm with
    /// `LayerNormConfig::new(hidden_size).with_epsilon(layer_norm_eps)`) for
    /// a complete Hugging Face-style re-initialization.
    #[must_use]
    pub fn init_policy(&self) -> InitPolicy {
        InitPolicy::normal(self.initializer_range)
    }

    /// Returns the width of one attention head after validation.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads.max(1)
    }
}

impl Validate for GptNeoXConfig {
    /// Checks dimensions, head divisibility, numerics, activation, tying, and rotary width.
    fn validate(&self) -> std::result::Result<(), ConfigError> {
        /// Family label attached to configuration validation errors.
        const FAMILY: &str = "gpt_neox";
        model_type(FAMILY, &self.model_type, &["gpt_neox"])?;
        let fields = [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("intermediate_size", self.intermediate_size),
            ("max_position_embeddings", self.max_position_embeddings),
        ];
        if let Some(position) = first_invalid_extent(&fields.map(|(_, value)| value)) {
            return Err(ConfigError::new(
                FAMILY,
                fields[position].0,
                "must be nonzero and fit a signed 32-bit tensor index",
            ));
        }
        layer_count(FAMILY, "num_hidden_layers", self.num_hidden_layers)?;
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(ConfigError::new(
                FAMILY,
                "hidden_size",
                "must be divisible by num_attention_heads",
            ));
        }
        positive_finite(FAMILY, "layer_norm_eps", self.layer_norm_eps)?;
        positive_finite(FAMILY, "initializer_range", self.initializer_range)?;
        if self.hidden_act != "gelu" {
            return Err(ConfigError::new(
                FAMILY,
                "hidden_act",
                "this recipe requires exact gelu",
            ));
        }
        if self.tie_word_embeddings {
            return Err(ConfigError::new(
                FAMILY,
                "tie_word_embeddings",
                "tied GPT-NeoX embeddings are not implemented",
            ));
        }
        if !product_fits(&[self.hidden_size, self.vocab_size])
            || !product_fits(&[self.hidden_size, self.intermediate_size])
            || !product_fits(&[self.hidden_size, self.hidden_size, 3])
        {
            return Err(ConfigError::new(
                FAMILY,
                "hidden_size",
                "parameter dimensions overflow usize",
            ));
        }
        RopeConfig::from_partial_factor(
            self.head_dim(),
            self.rotary_pct,
            self.rotary_emb_base,
            RopeScaling::None,
        )
        .map_err(|error| ConfigError::new(FAMILY, "rotary_pct", error.to_string()))?;
        Ok(())
    }
}

impl ModelConfig for GptNeoXConfig {
    /// Family label used by generic configuration dispatch and diagnostics.
    const FAMILY: &'static str = "gpt_neox";
    /// Serialized model discriminators supported by this GPT-NeoX schema.
    const MODEL_TYPES: &'static [&'static str] = &["gpt_neox"];
}

#[cfg(test)]
mod tests;
