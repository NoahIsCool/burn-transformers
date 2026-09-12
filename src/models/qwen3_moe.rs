//! Qwen3 sparse mixture-of-experts decoder configuration.
//!
//! Every decoder block contains grouped-query attention. Its feed-forward branch is
//! either a dense SwiGLU or a token-routed bank of SwiGLU experts. At an MoE layer,
//! the router chooses `num_experts_per_tok` of `num_experts` matrices for every
//! token; consequently routing counts are shape-bearing configuration, not sampling
//! policy.
//!
//! # Automatic registry
//!
//! The executable model and strict local loader are available directly through
//! [`Qwen3MoeForCausalLm`] and through the crate's automatic causal-model registry.
//! Automatic loading accepts full-precision SafeTensors and Q8/Q4 Burnpack
//! overlays. Version-two sidecars content-bind the checkpoint, load plan, and
//! packed payloads, while legacy version-one compatibility retains full source
//! validation. Registration reflects successful strict conversion, short Metal
//! smokes, and Q8/Q4 packed-kernel audits in which all 1,345 expected decode
//! matmuls used packed execution without whole-weight dequantization. It is not a
//! claim that the model has passed every release gate.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

mod checkpoint;
mod load;
pub mod model;

pub use checkpoint::{Qwen3MoeCheckpointContract, Qwen3MoeCheckpointOptions};
pub use load::{Qwen3MoeCheckpointLoadError, Qwen3MoeLoadError};
pub use model::{
    Qwen3MoeDecodeState, Qwen3MoeDecoderLayer, Qwen3MoeForCausalLm, Qwen3MoeModel,
    qwen3_moe_load_balancing_loss,
};

use super::config::{
    ConfigError, ModelConfig, RopeParameters, TokenIds, Validate, divisible, model_type, nonzero,
    positive_finite, probability, token_ids,
};

/// Serialized topology for Qwen3 MoE models such as Qwen3-30B-A3B.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Qwen3MoeConfig {
    /// Hugging Face architecture discriminator.
    pub model_type: String,
    /// Vocabulary/embedding row count.
    pub vocab_size: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Dense SwiGLU hidden width used on non-MoE layers.
    pub intermediate_size: usize,
    /// Number of causal decoder blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads shared by query groups.
    pub num_key_value_heads: usize,
    /// Explicit head width; Qwen3 does not require `H / num_heads`.
    pub head_dim: usize,
    /// Dense and expert gate activation.
    pub hidden_act: String,
    /// Maximum indexed sequence length.
    pub max_position_embeddings: usize,
    /// New-weight initialization standard deviation.
    pub initializer_range: f64,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f64,
    /// Whether generation maintains a KV cache.
    pub use_cache: bool,
    /// Whether embedding and output projection share parameters.
    pub tie_word_embeddings: bool,
    /// Modern RoPE object, also accepting the historic `rope_scaling` key.
    #[serde(alias = "rope_scaling")]
    pub rope_parameters: Option<RopeParameters>,
    /// Top-level theta retained by early Qwen3 artifacts.
    pub rope_theta: Option<f64>,
    /// Whether the attention projections contain bias vectors.
    pub attention_bias: bool,
    /// Attention-probability dropout.
    pub attention_dropout: f64,
    /// Enables a finite attention window instead of full causal history.
    pub use_sliding_window: bool,
    /// Number of preceding tokens visible under sliding attention.
    pub sliding_window: Option<usize>,
    /// Legacy limit on layers eligible for a sliding window.
    pub max_window_layers: usize,
    /// Period between MoE feed-forward blocks.
    pub decoder_sparse_step: usize,
    /// Hidden width of each routed expert.
    pub moe_intermediate_size: usize,
    /// Total number of independently parameterized routed experts.
    #[serde(alias = "num_local_experts")]
    pub num_experts: usize,
    /// Experts selected for each token.
    pub num_experts_per_tok: usize,
    /// Whether selected routing probabilities are renormalized to sum to one.
    pub norm_topk_prob: bool,
    /// Whether forward output includes router logits.
    pub output_router_logits: bool,
    /// Training-only load-balancing loss multiplier, retained for round trips.
    pub router_aux_loss_coef: f64,
    /// Layer indices forced to use the dense MLP.
    pub mlp_only_layers: Vec<usize>,
    /// Optional padding token ID.
    pub pad_token_id: Option<TokenIds>,
    /// Optional beginning-of-sequence token ID.
    pub bos_token_id: Option<TokenIds>,
    /// One or more end-of-sequence token IDs.
    pub eos_token_id: Option<TokenIds>,
}

impl Default for Qwen3MoeConfig {
    /// Supplies defaults matching the upstream Qwen3 MoE configuration class.
    fn default() -> Self {
        Self {
            model_type: "qwen3_moe".into(),
            vocab_size: 151_936,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 24,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            head_dim: 128,
            hidden_act: "silu".into(),
            max_position_embeddings: 32_768,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            tie_word_embeddings: false,
            rope_parameters: None,
            rope_theta: Some(10_000.0),
            attention_bias: false,
            attention_dropout: 0.0,
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: 24,
            decoder_sparse_step: 1,
            moe_intermediate_size: 768,
            num_experts: 128,
            num_experts_per_tok: 8,
            norm_topk_prob: false,
            output_router_logits: false,
            router_aux_loss_coef: 0.001,
            mlp_only_layers: Vec::new(),
            pad_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
        }
    }
}

impl Qwen3MoeConfig {
    /// Reports whether the indexed decoder block owns routed expert matrices.
    pub fn is_moe_layer(&self, layer_index: usize) -> bool {
        self.decoder_sparse_step != 0
            && !self.mlp_only_layers.contains(&layer_index)
            && self.num_experts > 0
            && (layer_index + 1).is_multiple_of(self.decoder_sparse_step)
    }
}

impl Validate for Qwen3MoeConfig {
    /// Checks attention grouping, expert routing, sparse cadence, RoPE, and tokens.
    fn validate(&self) -> Result<(), ConfigError> {
        /// Family label attached to configuration validation errors.
        const FAMILY: &str = "qwen3_moe";
        model_type(FAMILY, &self.model_type, &["qwen3_moe"])?;
        for (field, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("head_dim", self.head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
            ("decoder_sparse_step", self.decoder_sparse_step),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("num_experts", self.num_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
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
        if self.num_experts_per_tok > self.num_experts {
            return Err(ConfigError::new(
                FAMILY,
                "num_experts_per_tok",
                "cannot exceed num_experts",
            ));
        }
        if self.hidden_act != "silu" {
            return Err(ConfigError::new(
                FAMILY,
                "hidden_act",
                "released Qwen3 expert and dense blocks require SiLU",
            ));
        }
        positive_finite(FAMILY, "initializer_range", self.initializer_range)?;
        positive_finite(FAMILY, "rms_norm_eps", self.rms_norm_eps)?;
        probability(FAMILY, "attention_dropout", self.attention_dropout)?;
        if !self.router_aux_loss_coef.is_finite() || self.router_aux_loss_coef < 0.0 {
            return Err(ConfigError::new(
                FAMILY,
                "router_aux_loss_coef",
                "must be finite and non-negative",
            ));
        }
        if self.use_sliding_window {
            let window = self.sliding_window.ok_or_else(|| {
                ConfigError::new(
                    FAMILY,
                    "sliding_window",
                    "is required when use_sliding_window is true",
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
            if self.max_window_layers > self.num_hidden_layers {
                return Err(ConfigError::new(
                    FAMILY,
                    "max_window_layers",
                    "cannot exceed num_hidden_layers",
                ));
            }
        }

        let mut seen = HashSet::new();
        for index in &self.mlp_only_layers {
            if *index >= self.num_hidden_layers || !seen.insert(*index) {
                return Err(ConfigError::new(
                    FAMILY,
                    "mlp_only_layers",
                    "indices must be unique and less than num_hidden_layers",
                ));
            }
        }
        if !(0..self.num_hidden_layers).any(|index| self.is_moe_layer(index)) {
            return Err(ConfigError::new(
                FAMILY,
                "decoder_sparse_step",
                "schedule does not contain an MoE layer",
            ));
        }

        self.rope_parameters
            .clone()
            .unwrap_or_default()
            .validate_for(
                FAMILY,
                self.head_dim,
                self.max_position_embeddings,
                self.rope_theta,
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

impl ModelConfig for Qwen3MoeConfig {
    /// Diagnostic family name used by generic configuration parsing.
    const FAMILY: &'static str = "qwen3_moe";
    /// Checkpoint model discriminators accepted by this configuration.
    const MODEL_TYPES: &'static [&'static str] = &["qwen3_moe"];
}

#[cfg(test)]
#[path = "qwen3_moe_tests.rs"]
mod tests;
