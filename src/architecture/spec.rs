//! Serializable specifications for assembling experimental transformer graphs.
//!
//! These types describe topology rather than checkpoint families. A layer may
//! select a sequence mixer, feed-forward operator, normalization placement, and
//! residual topology independently. Exact family models may bypass this surface
//! when a checkpoint quirk cannot be represented faithfully.

use crate::{Error, Result};

/// Nonlinearity applied by a feed-forward expert or gated projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    /// Rectified linear unit `max(0, x)`, used by original T5 feed-forwards.
    Relu,
    /// The SiLU function `x * sigmoid(x)` used by SwiGLU.
    Silu,
    /// The exact Gaussian error linear unit.
    Gelu,
    /// PyTorch's tanh approximation to GELU.
    GeluTanh,
    /// Squared ReLU, `max(0, x)^2`, used by Nemotron-H experts.
    ReluSquared,
}

/// Layer-normalization operation and its numerical convention.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NormalizationSpec {
    /// Root-mean-square normalization without mean subtraction.
    Rms {
        /// Stabilizer added to the mean square before the reciprocal root.
        epsilon: f64,
        /// Whether the learned scale is interpreted as `1 + weight`.
        zero_centered: bool,
    },
    /// Mean-subtracting layer normalization.
    Layer {
        /// Stabilizer added to the variance before the reciprocal root.
        epsilon: f64,
        /// Whether a learned additive bias is present.
        bias: bool,
    },
}

impl NormalizationSpec {
    /// Validates that the variance stabilizer is finite and strictly positive.
    pub fn validate(&self, field: &str) -> Result<()> {
        let epsilon = match self {
            Self::Rms { epsilon, .. } | Self::Layer { epsilon, .. } => *epsilon,
        };
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::configuration(
                format!("{field}.epsilon"),
                "must be finite and greater than zero",
            ));
        }
        Ok(())
    }
}

/// Long-context scaling applied to rotary frequencies.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RopeScaling {
    /// Frequencies are used without extrapolation scaling.
    None,
    /// Static YaRN interpolation and extrapolation scaling.
    Yarn {
        /// Requested context expansion factor.
        factor: f64,
        /// Maximum sequence length used during pretraining.
        original_max_positions: usize,
        /// Optional attention-temperature multiplier.
        attention_factor: Option<f64>,
        /// Low-frequency transition parameter.
        beta_slow: f64,
        /// High-frequency transition parameter.
        beta_fast: f64,
    },
}

/// Position information supplied to one mixer layer.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PositionEncodingSpec {
    /// No explicit positional transform is applied.
    None,
    /// Rotary position encoding over an even prefix of every head.
    Rope {
        /// Base period controlling the geometric frequency sequence.
        theta: f64,
        /// Number of head channels rotated in pairs.
        rotary_dim: usize,
        /// Optional long-context scaling policy.
        scaling: RopeScaling,
    },
    /// Learned T5 relative-position buckets added to attention logits.
    T5RelativeBias {
        /// Number of signed/logarithmic distance buckets.
        buckets: usize,
        /// Maximum distance distinguished by logarithmic buckets.
        max_distance: usize,
        /// Whether distances in both directions receive independent buckets.
        bidirectional: bool,
    },
}

impl PositionEncodingSpec {
    /// Validates pairwise rotary dimensions or T5 bucket boundaries.
    pub fn validate(&self, head_dim: usize, field: &str) -> Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Rope {
                theta,
                rotary_dim,
                scaling,
            } => {
                if !theta.is_finite() || *theta <= 0.0 {
                    return Err(Error::configuration(
                        format!("{field}.theta"),
                        "must be finite and greater than zero",
                    ));
                }
                if *rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || *rotary_dim > head_dim {
                    return Err(Error::configuration(
                        format!("{field}.rotary_dim"),
                        "must be a non-zero even number no larger than head_dim",
                    ));
                }
                if let RopeScaling::Yarn {
                    factor,
                    original_max_positions,
                    attention_factor,
                    beta_slow,
                    beta_fast,
                } = scaling
                {
                    if !factor.is_finite() || *factor <= 1.0 {
                        return Err(Error::configuration(
                            format!("{field}.scaling.factor"),
                            "must be finite and greater than one",
                        ));
                    }
                    if *original_max_positions == 0 {
                        return Err(Error::configuration(
                            format!("{field}.scaling.original_max_positions"),
                            "must be non-zero",
                        ));
                    }
                    if attention_factor.is_some_and(|value| !value.is_finite() || value <= 0.0) {
                        return Err(Error::configuration(
                            format!("{field}.scaling.attention_factor"),
                            "must be finite and greater than zero when present",
                        ));
                    }
                    if !beta_slow.is_finite()
                        || !beta_fast.is_finite()
                        || *beta_slow <= 0.0
                        || *beta_fast <= *beta_slow
                    {
                        return Err(Error::configuration(
                            format!("{field}.scaling.beta"),
                            "requires finite 0 < beta_slow < beta_fast",
                        ));
                    }
                }
                Ok(())
            }
            Self::T5RelativeBias {
                buckets,
                max_distance,
                ..
            } => {
                if *buckets < 2 {
                    return Err(Error::configuration(
                        format!("{field}.buckets"),
                        "must contain at least two buckets",
                    ));
                }
                let exact_buckets = buckets / 2;
                if *max_distance <= exact_buckets {
                    return Err(Error::configuration(
                        format!("{field}.max_distance"),
                        "must exceed the exact-distance bucket region",
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Standard or grouped-query attention topology.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AttentionSpec {
    /// Number of independently projected query heads.
    pub query_heads: usize,
    /// Number of key/value heads retained in the cache.
    pub key_value_heads: usize,
    /// Width of one query/key/value head.
    pub head_dim: usize,
    /// Whether Q and K are normalized independently before their dot product.
    pub query_key_norm: bool,
    /// Whether the query projection also emits a sigmoid/SiLU output gate.
    pub output_gate: bool,
    /// Position transformation used by this attention layer.
    pub position: PositionEncodingSpec,
}

impl AttentionSpec {
    /// Validates grouped-head divisibility and positional dimensions.
    pub fn validate(&self, field: &str) -> Result<()> {
        if self.query_heads == 0 || self.key_value_heads == 0 || self.head_dim == 0 {
            return Err(Error::configuration(
                field,
                "query_heads, key_value_heads, and head_dim must be non-zero",
            ));
        }
        if !self.query_heads.is_multiple_of(self.key_value_heads) {
            return Err(Error::configuration(
                format!("{field}.key_value_heads"),
                "must divide query_heads exactly",
            ));
        }
        self.position
            .validate(self.head_dim, &format!("{field}.position"))
    }
}

/// Bounded-context attention that discards keys outside a trailing window.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SlidingWindowSpec {
    /// Attention topology evaluated inside the window.
    pub attention: AttentionSpec,
    /// Maximum number of key positions visible to a query.
    pub window: usize,
}

/// Multi-head latent attention projection and cache ranks.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MlaSpec {
    /// Number of query heads.
    pub query_heads: usize,
    /// Non-rotary query/key width per head.
    pub content_head_dim: usize,
    /// Rotary query/key width per head.
    pub rotary_head_dim: usize,
    /// Value width per head.
    pub value_head_dim: usize,
    /// Rank of the compressed key/value latent cached during decode.
    pub key_value_lora_rank: usize,
    /// Optional rank of a compressed query projection.
    pub query_lora_rank: Option<usize>,
    /// Position transformation applied only to rotary channels.
    pub position: PositionEncodingSpec,
}

/// Mamba2 state-space mixer dimensions.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Mamba2Spec {
    /// Number of independent state-space heads.
    pub heads: usize,
    /// Per-head input/output width.
    pub head_dim: usize,
    /// Per-head recurrent state dimension.
    pub state_dim: usize,
    /// Width of the causal depthwise convolution history.
    pub convolution_kernel: usize,
    /// Number of heads sharing one state-space group.
    pub group_size: usize,
}

/// Sequence-mixing operation used by one decoder layer.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MixerSpec {
    /// Unbounded causal attention.
    Attention(AttentionSpec),
    /// Causal attention restricted to a trailing window.
    SlidingAttention(SlidingWindowSpec),
    /// DeepSeek-style multi-head latent attention.
    Mla(MlaSpec),
    /// Gated DeltaNet linear attention.
    DeltaNet {
        /// Number of normalized query/key heads.
        key_heads: usize,
        /// Number of recurrent value heads.
        value_heads: usize,
        /// Query/key channel width per head.
        key_head_dim: usize,
        /// Value channel width per head.
        value_head_dim: usize,
        /// Width of the causal depthwise convolution.
        convolution_kernel: usize,
    },
    /// Mamba2 structured state-space mixing.
    Mamba2(Mamba2Spec),
}

impl MixerSpec {
    /// Validates head grouping and non-zero recurrent-state dimensions.
    pub fn validate(&self, field: &str) -> Result<()> {
        match self {
            Self::Attention(spec) => spec.validate(field),
            Self::SlidingAttention(spec) => {
                if spec.window == 0 {
                    return Err(Error::configuration(
                        format!("{field}.window"),
                        "must be non-zero",
                    ));
                }
                spec.attention.validate(&format!("{field}.attention"))
            }
            Self::Mla(spec) => {
                if spec.query_heads == 0
                    || spec.content_head_dim == 0
                    || spec.rotary_head_dim == 0
                    || spec.value_head_dim == 0
                    || spec.key_value_lora_rank == 0
                    || spec.query_lora_rank == Some(0)
                {
                    return Err(Error::configuration(
                        field,
                        "MLA dimensions must be non-zero",
                    ));
                }
                let head_dim = spec
                    .content_head_dim
                    .checked_add(spec.rotary_head_dim)
                    .ok_or_else(|| {
                        Error::configuration(
                            format!("{field}.head_dim"),
                            "MLA content and rotary dimensions overflow usize",
                        )
                    })?;
                spec.position
                    .validate(head_dim, &format!("{field}.position"))
            }
            Self::DeltaNet {
                key_heads,
                value_heads,
                key_head_dim,
                value_head_dim,
                convolution_kernel,
            } => {
                if *key_heads == 0
                    || *value_heads == 0
                    || *key_head_dim == 0
                    || *value_head_dim == 0
                    || *convolution_kernel == 0
                {
                    return Err(Error::configuration(
                        field,
                        "DeltaNet dimensions must be non-zero",
                    ));
                }
                if !value_heads.is_multiple_of(*key_heads) {
                    return Err(Error::configuration(
                        format!("{field}.value_heads"),
                        "must be an integer multiple of key_heads",
                    ));
                }
                Ok(())
            }
            Self::Mamba2(spec) => {
                if spec.heads == 0
                    || spec.head_dim == 0
                    || spec.state_dim == 0
                    || spec.convolution_kernel == 0
                    || spec.group_size == 0
                    || !spec.heads.is_multiple_of(spec.group_size)
                {
                    return Err(Error::configuration(
                        field,
                        "Mamba2 dimensions must be non-zero and group_size must divide heads",
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Dense gated feed-forward topology `down(activation(gate(x)) * up(x))`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DenseFeedForwardSpec {
    /// Intermediate channel count.
    pub intermediate_size: usize,
    /// Gate activation.
    pub activation: Activation,
    /// Whether linear projections contain additive biases.
    pub bias: bool,
}

/// Sparse routed mixture-of-experts topology.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MixtureOfExpertsSpec {
    /// Number of separately parameterized routed experts.
    pub routed_experts: usize,
    /// Number of selected routed experts per token.
    pub experts_per_token: usize,
    /// Number of always-evaluated shared experts.
    pub shared_experts: usize,
    /// Width of each expert's intermediate representation.
    pub expert_intermediate_size: usize,
    /// Expert gate activation.
    pub activation: Activation,
    /// Whether selected router probabilities are renormalized to sum to one.
    pub renormalize_selected: bool,
    /// Multiplicative scale applied to routed expert outputs.
    pub routed_scale: f64,
}

/// Feed-forward operation attached to a transformer block.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FeedForwardSpec {
    /// One dense gated feed-forward network.
    Dense(DenseFeedForwardSpec),
    /// A sparse router plus independent expert networks.
    MixtureOfExperts(MixtureOfExpertsSpec),
}

impl FeedForwardSpec {
    /// Validates intermediate widths and sparse-routing cardinalities.
    pub fn validate(&self, field: &str) -> Result<()> {
        match self {
            Self::Dense(spec) if spec.intermediate_size == 0 => Err(Error::configuration(
                format!("{field}.intermediate_size"),
                "must be non-zero",
            )),
            Self::Dense(_) => Ok(()),
            Self::MixtureOfExperts(spec) => {
                if spec.routed_experts == 0 || spec.expert_intermediate_size == 0 {
                    return Err(Error::configuration(
                        field,
                        "routed_experts and expert_intermediate_size must be non-zero",
                    ));
                }
                if spec.experts_per_token == 0 || spec.experts_per_token > spec.routed_experts {
                    return Err(Error::configuration(
                        format!("{field}.experts_per_token"),
                        "must be in 1..=routed_experts",
                    ));
                }
                if !spec.routed_scale.is_finite() || spec.routed_scale <= 0.0 {
                    return Err(Error::configuration(
                        format!("{field}.routed_scale"),
                        "must be finite and greater than zero",
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Placement of block normalization relative to its sublayer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormPlacement {
    /// Normalize before the sublayer only.
    Pre,
    /// Normalize after the sublayer only.
    Post,
    /// Normalize both before and after, as in Muse sandwich normalization.
    Sandwich,
}

/// Residual-update topology used around a block sublayer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualTopology {
    /// Add the sublayer output directly to its input.
    Add,
    /// Apply a learned multiplicative gate before the residual addition.
    Gated,
}

/// Complete specification for one heterogeneous decoder block.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DecoderLayerSpec {
    /// Sequence-mixing operation.
    pub mixer: MixerSpec,
    /// Dense or sparse feed-forward operation.
    pub feed_forward: FeedForwardSpec,
    /// Attention/mixer-side normalization.
    pub mixer_norm: NormalizationSpec,
    /// Feed-forward-side normalization.
    pub feed_forward_norm: NormalizationSpec,
    /// Relative normalization placement.
    pub norm_placement: NormPlacement,
    /// Residual-update topology.
    pub residual: ResidualTopology,
}

impl DecoderLayerSpec {
    /// Validates every independently configurable subcomponent.
    pub fn validate(&self, index: usize) -> Result<()> {
        let field = format!("layers[{index}]");
        self.mixer.validate(&format!("{field}.mixer"))?;
        self.feed_forward
            .validate(&format!("{field}.feed_forward"))?;
        self.mixer_norm.validate(&format!("{field}.mixer_norm"))?;
        self.feed_forward_norm
            .validate(&format!("{field}.feed_forward_norm"))
    }
}

/// Validated topology for a modular decoder-only language model.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DecoderSpec {
    /// Token vocabulary cardinality.
    pub vocabulary_size: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Ordered heterogeneous decoder blocks.
    pub layers: Vec<DecoderLayerSpec>,
    /// Final normalization before vocabulary projection.
    pub final_norm: NormalizationSpec,
    /// Whether the embedding matrix is also the vocabulary projection matrix.
    pub tie_word_embeddings: bool,
    /// Maximum supported logical position.
    pub max_position_embeddings: usize,
}

impl DecoderSpec {
    /// Validates global dimensions and every layer specification.
    pub fn validate(&self) -> Result<()> {
        if self.vocabulary_size == 0 {
            return Err(Error::configuration("vocabulary_size", "must be non-zero"));
        }
        if self.hidden_size == 0 {
            return Err(Error::configuration("hidden_size", "must be non-zero"));
        }
        if self.layers.is_empty() {
            return Err(Error::configuration("layers", "must not be empty"));
        }
        if self.max_position_embeddings == 0 {
            return Err(Error::configuration(
                "max_position_embeddings",
                "must be non-zero",
            ));
        }
        self.final_norm.validate("final_norm")?;
        for (index, layer) in self.layers.iter().enumerate() {
            layer.validate(index)?;
        }
        Ok(())
    }
}

/// Modular encoder-decoder topology with independent source and target stacks.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Seq2SeqSpec {
    /// Bidirectional encoder stack, represented using decoder layer building blocks.
    pub encoder: DecoderSpec,
    /// Causal decoder stack.
    pub decoder: DecoderSpec,
    /// Number of decoder cross-attention heads.
    pub cross_attention_heads: usize,
    /// Per-head cross-attention width.
    pub cross_attention_head_dim: usize,
}

impl Seq2SeqSpec {
    /// Validates stack compatibility and cross-attention projection width.
    pub fn validate(&self) -> Result<()> {
        self.encoder.validate()?;
        self.decoder.validate()?;
        if self.encoder.hidden_size != self.decoder.hidden_size {
            return Err(Error::configuration(
                "decoder.hidden_size",
                "must equal encoder.hidden_size",
            ));
        }
        if self.cross_attention_heads == 0
            || self.cross_attention_head_dim == 0
            || self
                .cross_attention_heads
                .checked_mul(self.cross_attention_head_dim)
                != Some(self.decoder.hidden_size)
        {
            return Err(Error::configuration(
                "cross_attention",
                "heads and head_dim must be non-zero and multiply to hidden_size",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "spec_tests.rs"]
mod tests;
