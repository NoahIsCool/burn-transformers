//! DeepSeek-V2 multi-latent-attention and sparse-expert topology.
//!
//! Multi-latent attention (MLA) compresses the cached key/value representation to
//! `kv_lora_rank`, then expands it into a content subspace of width
//! `qk_nope_head_dim` and a rotary subspace of width `qk_rope_head_dim`. Queries may
//! use the analogous `q_lora_rank` bottleneck or a direct projection. These ranks
//! must remain explicit: inferring ordinary QKV shapes from `hidden_size` would load
//! valid DeepSeek weights into the wrong matrices.

use serde::{Deserialize, Serialize};

mod checkpoint;
mod load;
pub mod mla;
pub mod model;

pub use checkpoint::{DeepseekV2CheckpointContract, DeepseekV2CheckpointOptions};
pub use load::{DeepseekV2CheckpointLoadError, DeepseekV2LoadError};
pub use model::{
    DeepseekV2DecodeState, DeepseekV2DecoderLayer, DeepseekV2ForCausalLm, DeepseekV2Model,
    MlaExecutionPath,
};

use super::config::{
    ConfigError, ModelConfig, RopeParameters, TokenIds, Validate, divisible, model_type, nonzero,
    positive_finite, probability, token_ids,
};

/// Function used to convert router logits into expert weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeepseekScoringFunction {
    /// Exponential normalization across all experts.
    #[serde(rename = "softmax")]
    Softmax,
    /// Independent logistic scores, used by later compatible checkpoints.
    #[serde(rename = "sigmoid")]
    Sigmoid,
}

/// Algorithm used to restrict expert selection to routing groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeepseekTopKMethod {
    /// Global top-k expert selection; `gready` accepts the upstream historic typo.
    #[serde(rename = "greedy", alias = "gready")]
    Greedy,
    /// Select top groups, then select experts only from those groups.
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
    /// Bias-corrected group routing used by later DeepSeek-compatible models.
    #[serde(rename = "noaux_tc")]
    NoAuxTc,
}

/// Serialized DeepSeek-V2 text-decoder topology.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeepseekV2Config {
    /// Hugging Face model discriminator.
    pub model_type: String,
    /// Vocabulary and embedding row count.
    pub vocab_size: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Dense SwiGLU hidden width in the initial dense layers.
    pub intermediate_size: usize,
    /// Hidden width of every routed and shared expert.
    pub moe_intermediate_size: usize,
    /// Number of causal decoder blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads after MLA expansion.
    pub num_attention_heads: usize,
    /// Compatibility count for non-MLA grouped-query attention.
    pub num_key_value_heads: usize,
    /// Number of shared experts evaluated for every token.
    pub n_shared_experts: usize,
    /// Total bank of routed experts.
    pub n_routed_experts: usize,
    /// Expert-parallel partition count encoded by the source config.
    pub ep_size: usize,
    /// Multiplier applied after combining routed expert outputs.
    pub routed_scaling_factor: f64,
    /// Rank of the compressed key/value latent cached per token.
    pub kv_lora_rank: usize,
    /// Optional query bottleneck; `None` means a direct query projection.
    pub q_lora_rank: Option<usize>,
    /// Per-head query/key channels that do not receive RoPE.
    pub qk_nope_head_dim: usize,
    /// Per-head query/key channels that receive RoPE.
    pub qk_rope_head_dim: usize,
    /// Per-head value width, independent of total query/key width.
    pub v_head_dim: usize,
    /// Router's top-k/group-selection method.
    pub topk_method: DeepseekTopKMethod,
    /// Number of equally sized expert groups.
    pub n_group: usize,
    /// Number of groups eligible for a token under group-limited routing.
    pub topk_group: usize,
    /// Routed experts selected per token.
    pub num_experts_per_tok: usize,
    /// Frequency of MoE blocks after the initial dense prefix.
    pub moe_layer_freq: usize,
    /// Number of shallow blocks forced to use dense SwiGLU.
    pub first_k_dense_replace: usize,
    /// Whether selected route weights are normalized after top-k.
    pub norm_topk_prob: bool,
    /// Router scoring function.
    pub scoring_func: DeepseekScoringFunction,
    /// Training-only auxiliary routing loss multiplier.
    pub aux_loss_alpha: f64,
    /// Whether auxiliary balance is computed independently per sequence.
    pub seq_aux: bool,
    /// Dense/expert gate activation.
    pub hidden_act: String,
    /// Maximum indexed sequence length.
    pub max_position_embeddings: usize,
    /// New-weight initialization standard deviation.
    pub initializer_range: f64,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f64,
    /// Whether inference maintains attention caches.
    pub use_cache: bool,
    /// Legacy tensor-parallel conversion factor.
    pub pretraining_tp: usize,
    /// Whether input embedding and output projection are tied.
    pub tie_word_embeddings: bool,
    /// Base RoPE theta before optional YaRN scaling.
    pub rope_theta: f64,
    /// Scaling parameters, serialized by DeepSeek-V2 as `rope_scaling`.
    #[serde(alias = "rope_parameters")]
    pub rope_scaling: Option<RopeParameters>,
    /// Whether MLA projection matrices contain bias.
    pub attention_bias: bool,
    /// Attention probability dropout.
    pub attention_dropout: f64,
    /// Whether to construct MLA rather than compatibility MHA/GQA.
    pub use_mla: bool,
    /// Optional padding token ID.
    pub pad_token_id: Option<TokenIds>,
    /// Beginning-of-sequence token ID.
    pub bos_token_id: Option<TokenIds>,
    /// One or more end-of-sequence token IDs.
    pub eos_token_id: Option<TokenIds>,
}

impl Default for DeepseekV2Config {
    /// Supplies the pinned DeepSeek-V2-Lite topology used by the v1 acceptance target.
    fn default() -> Self {
        Self {
            model_type: "deepseek_v2".into(),
            vocab_size: 102_400,
            hidden_size: 2048,
            intermediate_size: 10_944,
            moe_intermediate_size: 1408,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            n_shared_experts: 2,
            n_routed_experts: 64,
            ep_size: 1,
            routed_scaling_factor: 1.0,
            kv_lora_rank: 512,
            q_lora_rank: None,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            topk_method: DeepseekTopKMethod::Greedy,
            n_group: 1,
            topk_group: 1,
            num_experts_per_tok: 6,
            moe_layer_freq: 1,
            first_k_dense_replace: 1,
            norm_topk_prob: false,
            scoring_func: DeepseekScoringFunction::Softmax,
            aux_loss_alpha: 0.001,
            seq_aux: true,
            hidden_act: "silu".into(),
            max_position_embeddings: 163_840,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pretraining_tp: 1,
            tie_word_embeddings: false,
            rope_theta: 10_000.0,
            rope_scaling: Some(RopeParameters {
                rope_type: super::config::RopeType::Yarn,
                factor: Some(40.0),
                original_max_position_embeddings: Some(4096),
                beta_fast: Some(32.0),
                beta_slow: Some(1.0),
                mscale: Some(0.707),
                mscale_all_dim: Some(0.707),
                ..RopeParameters::default()
            }),
            attention_bias: false,
            attention_dropout: 0.0,
            use_mla: true,
            pad_token_id: None,
            bos_token_id: Some(TokenIds::One(100_000)),
            eos_token_id: Some(TokenIds::One(100_001)),
        }
    }
}

impl DeepseekV2Config {
    /// Returns the concatenated content-plus-rotary query/key width per head.
    ///
    /// Invalid configurations whose dimensions overflow saturate here and are
    /// rejected by [`Validate::validate`]. This keeps the convenience accessor
    /// non-panicking even when it is used while reporting a malformed artifact.
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim.saturating_add(self.qk_rope_head_dim)
    }

    /// Reports whether a decoder index uses shared+routed experts.
    pub fn is_moe_layer(&self, layer_index: usize) -> bool {
        self.moe_layer_freq != 0
            && layer_index >= self.first_k_dense_replace
            && layer_index.is_multiple_of(self.moe_layer_freq)
    }
}

impl Validate for DeepseekV2Config {
    /// Checks MLA ranks, head spaces, expert grouping/cadence, YaRN, and tokens.
    fn validate(&self) -> Result<(), ConfigError> {
        const FAMILY: &str = "deepseek_v2";
        model_type(FAMILY, &self.model_type, &["deepseek_v2"])?;
        for (field, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("qk_nope_head_dim", self.qk_nope_head_dim),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("v_head_dim", self.v_head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
            ("pretraining_tp", self.pretraining_tp),
            ("moe_layer_freq", self.moe_layer_freq),
        ] {
            nonzero(FAMILY, field, value)?;
        }
        if !self.qk_rope_head_dim.is_multiple_of(2) {
            return Err(ConfigError::new(
                FAMILY,
                "qk_rope_head_dim",
                "RoPE rotates channel pairs, so this width must be even",
            ));
        }
        self.qk_nope_head_dim
            .checked_add(self.qk_rope_head_dim)
            .ok_or_else(|| {
                ConfigError::new(
                    FAMILY,
                    "qk_nope_head_dim",
                    "content-plus-rotary head width overflows usize",
                )
            })?;
        if self.use_mla {
            nonzero(FAMILY, "kv_lora_rank", self.kv_lora_rank)?;
            if let Some(rank) = self.q_lora_rank {
                nonzero(FAMILY, "q_lora_rank", rank)?;
            }
        } else {
            divisible(
                FAMILY,
                "num_attention_heads",
                self.num_attention_heads,
                "num_key_value_heads",
                self.num_key_value_heads,
            )?;
        }
        for (field, value) in [
            ("n_shared_experts", self.n_shared_experts),
            ("n_routed_experts", self.n_routed_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("n_group", self.n_group),
            ("topk_group", self.topk_group),
            ("ep_size", self.ep_size),
        ] {
            nonzero(FAMILY, field, value)?;
        }
        divisible(
            FAMILY,
            "n_routed_experts",
            self.n_routed_experts,
            "n_group",
            self.n_group,
        )?;
        if self.num_experts_per_tok > self.n_routed_experts {
            return Err(ConfigError::new(
                FAMILY,
                "num_experts_per_tok",
                "cannot exceed n_routed_experts",
            ));
        }
        if self.topk_group > self.n_group {
            return Err(ConfigError::new(
                FAMILY,
                "topk_group",
                "cannot exceed n_group",
            ));
        }
        let eligible = (self.n_routed_experts / self.n_group)
            .checked_mul(self.topk_group)
            .ok_or_else(|| {
                ConfigError::new(
                    FAMILY,
                    "topk_group",
                    "eligible expert count overflows usize",
                )
            })?;
        if self.num_experts_per_tok > eligible {
            return Err(ConfigError::new(
                FAMILY,
                "num_experts_per_tok",
                "exceeds the experts available in selected routing groups",
            ));
        }
        if self.first_k_dense_replace >= self.num_hidden_layers {
            return Err(ConfigError::new(
                FAMILY,
                "first_k_dense_replace",
                "must leave at least one decoder layer eligible for MoE",
            ));
        }
        if !(0..self.num_hidden_layers).any(|index| self.is_moe_layer(index)) {
            return Err(ConfigError::new(
                FAMILY,
                "moe_layer_freq",
                "schedule does not contain an MoE layer",
            ));
        }
        if self.hidden_act != "silu" {
            return Err(ConfigError::new(
                FAMILY,
                "hidden_act",
                "released DeepSeek-V2 blocks require SiLU/SwiGLU",
            ));
        }
        positive_finite(FAMILY, "routed_scaling_factor", self.routed_scaling_factor)?;
        positive_finite(FAMILY, "initializer_range", self.initializer_range)?;
        positive_finite(FAMILY, "rms_norm_eps", self.rms_norm_eps)?;
        positive_finite(FAMILY, "rope_theta", self.rope_theta)?;
        probability(FAMILY, "attention_dropout", self.attention_dropout)?;
        if !self.aux_loss_alpha.is_finite() || self.aux_loss_alpha < 0.0 {
            return Err(ConfigError::new(
                FAMILY,
                "aux_loss_alpha",
                "must be finite and non-negative",
            ));
        }
        self.rope_scaling.clone().unwrap_or_default().validate_for(
            FAMILY,
            self.qk_rope_head_dim,
            self.max_position_embeddings,
            Some(self.rope_theta),
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

impl ModelConfig for DeepseekV2Config {
    const FAMILY: &'static str = "deepseek_v2";
    const MODEL_TYPES: &'static [&'static str] = &["deepseek_v2"];
}

#[cfg(test)]
#[path = "deepseek_v2_tests.rs"]
mod tests;
