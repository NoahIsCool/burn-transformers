//! NVIDIA Nemotron-H hybrid Mamba/attention/MLP/MoE topology.
//!
//! Nemotron-H serializes a heterogeneous block sequence. Legacy checkpoints encode
//! one character per block (`M` Mamba, `*` attention, `-` dense MLP, `E` MoE), while
//! current configs use descriptive arrays. Both representations normalize to
//! [`NemotronHBlockType`]. The sequence itself, rather than a nominal depth, is the
//! authoritative topology because each block owns fundamentally different state.

use serde::de::Deserializer;
use serde::{Deserialize, Serialize};

use super::config::{
    ConfigError, ModelConfig, TokenIds, Validate, divisible, model_type, nonzero, positive_finite,
    probability, token_ids,
};

/// Mixer/FFN operation owned by one Nemotron-H residual block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NemotronHBlockType {
    /// Mamba-2 selective state-space mixer with convolutional state.
    #[serde(rename = "linear_attention", alias = "mamba")]
    LinearAttention,
    /// Ordinary causal grouped-query attention.
    #[serde(rename = "full_attention", alias = "attention")]
    FullAttention,
    /// Dense feed-forward residual block.
    #[serde(rename = "mlp")]
    Mlp,
    /// Shared-plus-routed sparse expert residual block.
    #[serde(rename = "moe")]
    Moe,
}

impl NemotronHBlockType {
    /// Converts one legacy pattern character to its normalized block type.
    fn from_pattern(character: char) -> Option<Self> {
        match character {
            'M' => Some(Self::LinearAttention),
            '*' => Some(Self::FullAttention),
            '-' => Some(Self::Mlp),
            'E' => Some(Self::Moe),
            _ => None,
        }
    }

    /// Converts a normalized block type back to NVIDIA's compact pattern syntax.
    fn pattern_character(self) -> char {
        match self {
            Self::LinearAttention => 'M',
            Self::FullAttention => '*',
            Self::Mlp => '-',
            Self::Moe => 'E',
        }
    }
}

/// Wire representation accepted for both old pattern strings and modern arrays.
#[derive(Deserialize)]
#[serde(untagged)]
enum BlockScheduleWire {
    /// Compact legacy string such as `M-M-M*`.
    Pattern(String),
    /// Modern list of descriptive layer names.
    List(Vec<NemotronHBlockType>),
}

/// Deserializes and normalizes a Nemotron block schedule.
fn deserialize_block_schedule<'de, D>(deserializer: D) -> Result<Vec<NemotronHBlockType>, D::Error>
where
    D: Deserializer<'de>,
{
    let wire = BlockScheduleWire::deserialize(deserializer)?;
    match wire {
        BlockScheduleWire::List(schedule) => Ok(schedule),
        BlockScheduleWire::Pattern(pattern) => pattern
            .chars()
            .enumerate()
            .map(|(index, character)| {
                NemotronHBlockType::from_pattern(character).ok_or_else(|| {
                    serde::de::Error::custom(format!(
                        "invalid Nemotron-H pattern character {character:?} at index {index}"
                    ))
                })
            })
            .collect(),
    }
}

/// Normalized configuration spanning legacy Nemotron-H and modern Nemotron Nano.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NemotronHConfig {
    /// Hugging Face discriminator.
    pub model_type: String,
    /// Vocabulary/embedding row count.
    pub vocab_size: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Authoritative sequence of heterogeneous residual blocks.
    #[serde(
        alias = "hybrid_override_pattern",
        alias = "layer_types",
        deserialize_with = "deserialize_block_schedule"
    )]
    pub layers_block_type: Vec<NemotronHBlockType>,
    /// Deprecated declared depth retained to detect contradictory legacy configs.
    pub num_hidden_layers: Option<usize>,
    /// Whether token embedding and output projection share parameters.
    pub tie_word_embeddings: bool,
    /// Whether inference maintains Mamba and attention caches.
    pub use_cache: bool,
    /// Number of final prompt positions projected to vocabulary logits.
    pub num_logits_to_keep: usize,
    /// Optional padding token ID.
    pub pad_token_id: Option<TokenIds>,
    /// Optional beginning-of-sequence token ID.
    pub bos_token_id: Option<TokenIds>,
    /// End-of-sequence token ID or IDs.
    pub eos_token_id: Option<TokenIds>,
    /// Query-head count in full-attention blocks.
    pub num_attention_heads: usize,
    /// Shared key/value head count in full-attention blocks.
    pub num_key_value_heads: usize,
    /// Full-attention head width; legacy configs use `attention_head_dim`.
    #[serde(alias = "attention_head_dim")]
    pub head_dim: usize,
    /// Maximum indexed context length for attention blocks.
    pub max_position_embeddings: usize,
    /// Whether attention projections include bias.
    pub attention_bias: bool,
    /// Attention-probability dropout.
    pub attention_dropout: f64,
    /// Optional causal sliding window for attention blocks.
    pub sliding_window: Option<usize>,
    /// Hidden width of dense MLP blocks.
    pub intermediate_size: usize,
    /// Dense MLP activation, normally squared ReLU (`relu2`).
    pub mlp_hidden_act: String,
    /// Whether dense MLP projections contain bias.
    pub mlp_bias: bool,
    /// Runtime hint retained from official configs; it does not change topology.
    pub use_mamba_kernels: bool,
    /// State dimension `N` in each selective state-space head.
    pub ssm_state_size: usize,
    /// Number of Mamba-2 heads.
    pub mamba_num_heads: usize,
    /// Per-head Mamba expansion width.
    pub mamba_head_dim: usize,
    /// Activation after Mamba's causal convolution.
    pub mamba_hidden_act: String,
    /// Number of groups sharing Mamba B/C state parameters.
    #[serde(alias = "mamba_n_groups")]
    pub n_groups: usize,
    /// Width of the causal convolution kernel.
    #[serde(alias = "mamba_d_conv")]
    pub conv_kernel: usize,
    /// Legacy expansion hint retained for checkpoint compatibility.
    #[serde(alias = "mamba_expand")]
    pub expand: usize,
    /// Minimum learned discretization step.
    #[serde(alias = "mamba_dt_min")]
    pub time_step_min: f64,
    /// Maximum learned discretization step.
    #[serde(alias = "mamba_dt_max")]
    pub time_step_max: f64,
    /// Runtime clamp `[lower, upper]`; `None` denotes positive infinity.
    #[serde(alias = "mamba_dt_limit")]
    pub time_step_limit: Vec<Option<f64>>,
    /// Small positive floor used while initializing discretization steps.
    #[serde(alias = "mamba_dt_init_floor")]
    pub time_step_floor: f64,
    /// Optional legacy low-rank discretization projection width.
    pub time_step_rank: Option<usize>,
    /// Whether the causal convolution has a bias.
    #[serde(alias = "mamba_conv_bias")]
    pub use_conv_bias: bool,
    /// Scan chunk width used by the Mamba-2 recurrence.
    #[serde(alias = "mamba_chunk_size")]
    pub chunk_size: usize,
    /// Whether Mamba input/output projections contain bias.
    pub mamba_proj_bias: bool,
    /// Accumulator/cache dtype for recurrent SSM state.
    pub mamba_ssm_cache_dtype: String,
    /// Number of routed experts available in MoE blocks.
    #[serde(alias = "num_local_experts")]
    pub n_routed_experts: usize,
    /// Number of experts evaluated for every token.
    pub n_shared_experts: usize,
    /// Hidden width of routed expert projections.
    pub moe_intermediate_size: usize,
    /// Hidden width of shared expert projections.
    pub moe_shared_expert_intermediate_size: usize,
    /// Optional low-rank bottleneck shared by expert projections.
    pub moe_latent_size: Option<usize>,
    /// Whether shared-expert execution may overlap routed experts.
    pub moe_shared_expert_overlap: bool,
    /// Routed experts selected per token.
    pub num_experts_per_tok: usize,
    /// Multiplier applied to combined routed expert output.
    pub routed_scaling_factor: f64,
    /// Number of equally sized router groups.
    pub n_group: usize,
    /// Number of router groups eligible for a token.
    pub topk_group: usize,
    /// Whether top-k route weights are normalized.
    pub norm_topk_prob: bool,
    /// Number of additional next-token-prediction stages.
    pub num_nextn_predict_layers: usize,
    /// Block sequence used inside each MTP stage.
    #[serde(
        alias = "mtp_hybrid_override_pattern",
        deserialize_with = "deserialize_block_schedule"
    )]
    pub mtp_layers_block_type: Vec<NemotronHBlockType>,
    /// Global projection-bias compatibility switch.
    pub use_bias: bool,
    /// New-weight initialization standard deviation.
    pub initializer_range: f64,
    /// Epsilon applied by residual-block normalization.
    pub layer_norm_epsilon: f64,
    /// Redundant epsilon emitted by original Nemotron-H checkpoints.
    pub rms_norm_eps: Option<f64>,
    /// Redundant epsilon emitted by newer Nemotron Nano checkpoints.
    pub norm_eps: Option<f64>,
    /// Whether residual accumulation uses FP32.
    pub residual_in_fp32: bool,
    /// Residual-stream dropout.
    pub hidden_dropout: f64,
    /// Whether initialization rescales pre-normalized residual branches by depth.
    pub rescale_prenorm_residual: bool,
    /// Base RoPE theta for full-attention blocks.
    pub rope_theta: Option<f64>,
    /// Fraction of attention-head channels receiving RoPE.
    pub partial_rotary_factor: Option<f64>,
}

impl Default for NemotronHConfig {
    /// Supplies current upstream defaults with an explicit heterogeneous schedule.
    fn default() -> Self {
        Self {
            model_type: "nemotron_h".into(),
            vocab_size: 131_072,
            hidden_size: 4096,
            layers_block_type: vec![
                NemotronHBlockType::LinearAttention,
                NemotronHBlockType::Moe,
                NemotronHBlockType::FullAttention,
                NemotronHBlockType::Mlp,
            ],
            num_hidden_layers: None,
            tie_word_embeddings: false,
            use_cache: true,
            num_logits_to_keep: 1,
            pad_token_id: Some(TokenIds::One(0)),
            bos_token_id: Some(TokenIds::One(1)),
            eos_token_id: Some(TokenIds::One(2)),
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 4096,
            attention_bias: false,
            attention_dropout: 0.0,
            sliding_window: None,
            intermediate_size: 21_504,
            mlp_hidden_act: "relu2".into(),
            mlp_bias: false,
            use_mamba_kernels: true,
            ssm_state_size: 128,
            mamba_num_heads: 128,
            mamba_head_dim: 64,
            mamba_hidden_act: "silu".into(),
            n_groups: 8,
            conv_kernel: 4,
            expand: 2,
            time_step_min: 0.001,
            time_step_max: 0.1,
            time_step_limit: vec![Some(0.0), None],
            time_step_floor: 1e-4,
            time_step_rank: None,
            use_conv_bias: true,
            chunk_size: 128,
            mamba_proj_bias: false,
            mamba_ssm_cache_dtype: "float32".into(),
            n_routed_experts: 8,
            n_shared_experts: 1,
            moe_intermediate_size: 7688,
            moe_shared_expert_intermediate_size: 7688,
            moe_latent_size: None,
            moe_shared_expert_overlap: true,
            num_experts_per_tok: 2,
            routed_scaling_factor: 1.0,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            num_nextn_predict_layers: 0,
            mtp_layers_block_type: vec![NemotronHBlockType::FullAttention, NemotronHBlockType::Moe],
            use_bias: false,
            initializer_range: 0.02,
            layer_norm_epsilon: 1e-5,
            rms_norm_eps: None,
            norm_eps: None,
            residual_in_fp32: false,
            hidden_dropout: 0.0,
            rescale_prenorm_residual: true,
            rope_theta: None,
            partial_rotary_factor: None,
        }
    }
}

impl NemotronHConfig {
    /// Returns the authoritative block count derived from the normalized schedule.
    pub fn block_count(&self) -> usize {
        self.layers_block_type.len()
    }

    /// Serializes the normalized schedule to NVIDIA's legacy compact syntax.
    pub fn hybrid_override_pattern(&self) -> String {
        self.layers_block_type
            .iter()
            .map(|kind| kind.pattern_character())
            .collect()
    }
}

impl Validate for NemotronHConfig {
    /// Checks heterogeneous schedule, per-mixer dimensions, routing, time steps, and IDs.
    fn validate(&self) -> Result<(), ConfigError> {
        const FAMILY: &str = "nemotron_h";
        model_type(FAMILY, &self.model_type, &["nemotron_h"])?;
        for (field, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("layers_block_type", self.block_count()),
            ("max_position_embeddings", self.max_position_embeddings),
            ("num_logits_to_keep", self.num_logits_to_keep),
        ] {
            nonzero(FAMILY, field, value)?;
        }
        if let Some(declared) = self.num_hidden_layers
            && declared != self.block_count()
        {
            return Err(ConfigError::new(
                FAMILY,
                "num_hidden_layers",
                "must match the authoritative block schedule length",
            ));
        }
        let has_attention = self
            .layers_block_type
            .contains(&NemotronHBlockType::FullAttention);
        let has_mamba = self
            .layers_block_type
            .contains(&NemotronHBlockType::LinearAttention);
        let has_mlp = self.layers_block_type.contains(&NemotronHBlockType::Mlp);
        let has_moe = self.layers_block_type.contains(&NemotronHBlockType::Moe);

        if has_attention {
            nonzero(FAMILY, "head_dim", self.head_dim)?;
            divisible(
                FAMILY,
                "num_attention_heads",
                self.num_attention_heads,
                "num_key_value_heads",
                self.num_key_value_heads,
            )?;
            let rotary_factor = self.partial_rotary_factor.unwrap_or(1.0);
            if !(rotary_factor.is_finite() && 0.0 < rotary_factor && rotary_factor <= 1.0) {
                return Err(ConfigError::new(
                    FAMILY,
                    "partial_rotary_factor",
                    "must be finite and in (0, 1]",
                ));
            }
            let rotary_width = self.head_dim as f64 * rotary_factor;
            let rounded_rotary_width = rotary_width.round();
            if (rotary_width - rounded_rotary_width).abs() > 1e-9
                || !(rounded_rotary_width as usize).is_multiple_of(2)
            {
                return Err(ConfigError::new(
                    FAMILY,
                    "partial_rotary_factor",
                    "must produce an even integral rotary width",
                ));
            }
            positive_finite(FAMILY, "rope_theta", self.rope_theta.unwrap_or(10_000.0))?;
            if let Some(window) = self.sliding_window {
                nonzero(FAMILY, "sliding_window", window)?;
                if window > self.max_position_embeddings {
                    return Err(ConfigError::new(
                        FAMILY,
                        "sliding_window",
                        "cannot exceed max_position_embeddings",
                    ));
                }
            }
        }
        if has_mlp {
            nonzero(FAMILY, "intermediate_size", self.intermediate_size)?;
            if self.mlp_hidden_act != "relu2" && self.mlp_hidden_act != "silu" {
                return Err(ConfigError::new(
                    FAMILY,
                    "mlp_hidden_act",
                    "must be `relu2` or `silu`",
                ));
            }
        }
        if has_mamba {
            for (field, value) in [
                ("ssm_state_size", self.ssm_state_size),
                ("mamba_num_heads", self.mamba_num_heads),
                ("mamba_head_dim", self.mamba_head_dim),
                ("n_groups", self.n_groups),
                ("conv_kernel", self.conv_kernel),
                ("expand", self.expand),
                ("chunk_size", self.chunk_size),
            ] {
                nonzero(FAMILY, field, value)?;
            }
            if !self.mamba_num_heads.is_multiple_of(self.n_groups) {
                return Err(ConfigError::new(
                    FAMILY,
                    "mamba_num_heads",
                    "must be divisible by n_groups",
                ));
            }
            if self.mamba_hidden_act != "silu" {
                return Err(ConfigError::new(
                    FAMILY,
                    "mamba_hidden_act",
                    "released Nemotron-H Mamba blocks require SiLU",
                ));
            }
            if let Some(rank) = self.time_step_rank {
                nonzero(FAMILY, "time_step_rank", rank)?;
            }
            positive_finite(FAMILY, "time_step_min", self.time_step_min)?;
            positive_finite(FAMILY, "time_step_max", self.time_step_max)?;
            positive_finite(FAMILY, "time_step_floor", self.time_step_floor)?;
            if self.time_step_min > self.time_step_max || self.time_step_floor > self.time_step_min
            {
                return Err(ConfigError::new(
                    FAMILY,
                    "time_step_min",
                    "requires time_step_floor <= time_step_min <= time_step_max",
                ));
            }
            if self.time_step_limit.len() != 2 {
                return Err(ConfigError::new(
                    FAMILY,
                    "time_step_limit",
                    "must contain exactly [lower, upper]",
                ));
            }
            let lower = self.time_step_limit[0].ok_or_else(|| {
                ConfigError::new(FAMILY, "time_step_limit", "lower bound must be finite")
            })?;
            if !lower.is_finite() || lower < 0.0 {
                return Err(ConfigError::new(
                    FAMILY,
                    "time_step_limit",
                    "lower bound must be finite and non-negative",
                ));
            }
            if let Some(upper) = self.time_step_limit[1]
                && (!upper.is_finite() || upper < lower)
            {
                return Err(ConfigError::new(
                    FAMILY,
                    "time_step_limit",
                    "finite upper bound must be at least the lower bound",
                ));
            }
            if !["float32", "float16", "bfloat16"].contains(&self.mamba_ssm_cache_dtype.as_str()) {
                return Err(ConfigError::new(
                    FAMILY,
                    "mamba_ssm_cache_dtype",
                    "must be float32, float16, or bfloat16",
                ));
            }
        }
        if has_moe {
            for (field, value) in [
                ("n_routed_experts", self.n_routed_experts),
                ("n_shared_experts", self.n_shared_experts),
                ("moe_intermediate_size", self.moe_intermediate_size),
                (
                    "moe_shared_expert_intermediate_size",
                    self.moe_shared_expert_intermediate_size,
                ),
                ("num_experts_per_tok", self.num_experts_per_tok),
                ("n_group", self.n_group),
                ("topk_group", self.topk_group),
            ] {
                nonzero(FAMILY, field, value)?;
            }
            if let Some(latent) = self.moe_latent_size {
                nonzero(FAMILY, "moe_latent_size", latent)?;
            }
            divisible(
                FAMILY,
                "n_routed_experts",
                self.n_routed_experts,
                "n_group",
                self.n_group,
            )?;
            let eligible_experts = (self.n_routed_experts / self.n_group)
                .checked_mul(self.topk_group)
                .ok_or_else(|| {
                    ConfigError::new(
                        FAMILY,
                        "topk_group",
                        "eligible expert count overflows usize",
                    )
                })?;
            if self.topk_group > self.n_group || self.num_experts_per_tok > eligible_experts {
                return Err(ConfigError::new(
                    FAMILY,
                    "num_experts_per_tok",
                    "routing groups do not contain enough eligible experts",
                ));
            }
            positive_finite(FAMILY, "routed_scaling_factor", self.routed_scaling_factor)?;
        }
        if self.num_nextn_predict_layers > 0 && self.mtp_layers_block_type.is_empty() {
            return Err(ConfigError::new(
                FAMILY,
                "mtp_layers_block_type",
                "must be non-empty when MTP is enabled",
            ));
        }
        positive_finite(FAMILY, "initializer_range", self.initializer_range)?;
        positive_finite(FAMILY, "layer_norm_epsilon", self.layer_norm_epsilon)?;
        probability(FAMILY, "attention_dropout", self.attention_dropout)?;
        probability(FAMILY, "hidden_dropout", self.hidden_dropout)?;
        for (field, epsilon) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("norm_eps", self.norm_eps),
        ] {
            if let Some(epsilon) = epsilon {
                positive_finite(FAMILY, field, epsilon)?;
                if (epsilon - self.layer_norm_epsilon).abs() > f64::EPSILON {
                    return Err(ConfigError::new(
                        FAMILY,
                        field,
                        "must agree with layer_norm_epsilon",
                    ));
                }
            }
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

impl ModelConfig for NemotronHConfig {
    const FAMILY: &'static str = "nemotron_h";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_pattern_round_trips_to_normalized_blocks() {
        let json = r#"{
            "model_type":"nemotron_h",
            "hybrid_override_pattern":"M-*E",
            "num_hidden_layers":4
        }"#;
        let config: NemotronHConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hybrid_override_pattern(), "M-*E");
        config.validate().unwrap();
    }

    #[test]
    fn rejects_declared_depth_that_conflicts_with_pattern() {
        let config = NemotronHConfig {
            num_hidden_layers: Some(99),
            ..NemotronHConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
