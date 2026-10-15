//! Serializable specifications for assembling experimental transformer graphs.
//!
//! These types describe topology rather than checkpoint families. A layer may
//! select a sequence mixer, feed-forward operator, normalization placement, and
//! residual topology independently. Exact family models may bypass this surface
//! when a checkpoint quirk cannot be represented faithfully.
//!
//! # One source of truth for validation
//!
//! Position encodings are validated by constructing the shared runtime objects
//! they describe ([`RopeConfig`], [`YarnConfig`], [`T5RelativePositionBuckets`])
//! and mapping their errors to spec field paths; mixers are validated by their
//! components' own configuration checks at the stack's width. Validation thus
//! applies exactly the components' rules, and a property test checks it against
//! direct construction of each component in both directions.
//! [`PositionEncodingSpec::rope_config`] and [`PositionEncodingSpec::t5_buckets`]
//! expose those objects to consumers. (The modular models still build RoPE
//! through their own copy of this mapping until they adopt these accessors.)
//!
//! `validate` checks that a spec is well formed. A few well-formed variants
//! ([`ResidualTopology::Gated`], [`MixerSpec::Mamba2`]) are describable but not
//! yet executable by the modular models, whose `init` rejects them with a
//! capability error before allocating parameters.

use crate::{
    Error, Result,
    attention::{AttentionOutputGate, GroupedQueryAttentionConfig},
    delta_net::GatedDeltaNetConfig,
    mla::MultiLatentAttentionConfig,
    position::{
        RopeConfig, RopeConfigError, RopeScaling as RuntimeRopeScaling, T5BucketError,
        T5RelativePositionBuckets, YarnConfig,
    },
};

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
    ///
    /// Delegates to the shared constructors used at initialization:
    /// [`Self::rope_config`] for RoPE over `head_dim` channels and
    /// [`Self::t5_buckets`] for T5 relative buckets. Errors name the spec field
    /// below `field`, such as `layers[0].mixer.position.rotary_dim`.
    pub fn validate(&self, head_dim: usize, field: &str) -> Result<()> {
        self.rope_config_at(head_dim, field)?;
        self.t5_buckets_at(field)?;
        Ok(())
    }

    /// Builds the shared RoPE policy for heads of `head_dim` channels.
    ///
    /// Returns `None` for non-rotary encodings. YaRN parameters pass through
    /// [`YarnConfig`] with the same defaults and bounds as checkpoint configs
    /// (factor at least one, `0 < beta_slow < beta_fast`, `theta > 1`).
    pub fn rope_config(&self, head_dim: usize) -> Result<Option<RopeConfig>> {
        self.rope_config_at(head_dim, "position")
    }

    /// Builds the shared T5 bucket mapping, or `None` for non-T5 encodings.
    ///
    /// Bidirectional tables need an even bucket count of at least four, and
    /// `max_distance` must exceed the exact region `buckets_per_direction / 2`.
    pub fn t5_buckets(&self) -> Result<Option<T5RelativePositionBuckets>> {
        self.t5_buckets_at("position")
    }

    /// [`Self::rope_config`] with errors reported below an explicit field path.
    fn rope_config_at(&self, head_dim: usize, field: &str) -> Result<Option<RopeConfig>> {
        let Self::Rope {
            theta,
            rotary_dim,
            scaling,
        } = self
        else {
            return Ok(None);
        };
        let at = |error: RopeConfigError| {
            Error::configuration(
                format!("{field}.{}", rope_error_field(&error)),
                error.to_string(),
            )
        };
        let scaling = match *scaling {
            RopeScaling::None => RuntimeRopeScaling::None,
            RopeScaling::Yarn {
                factor,
                original_max_positions,
                attention_factor,
                beta_slow,
                beta_fast,
            } => {
                let mut yarn = YarnConfig::new(factor, original_max_positions)
                    .and_then(|yarn| yarn.with_betas(beta_fast, beta_slow))
                    .map_err(at)?;
                if let Some(attention_factor) = attention_factor {
                    yarn = yarn.with_attention_factor(attention_factor).map_err(at)?;
                }
                RuntimeRopeScaling::Yarn(yarn)
            }
        };
        RopeConfig::new(head_dim, *rotary_dim, *theta, scaling)
            .map(Some)
            .map_err(at)
    }

    /// [`Self::t5_buckets`] with errors reported below an explicit field path.
    fn t5_buckets_at(&self, field: &str) -> Result<Option<T5RelativePositionBuckets>> {
        let Self::T5RelativeBias {
            buckets,
            max_distance,
            bidirectional,
        } = self
        else {
            return Ok(None);
        };
        T5RelativePositionBuckets::new(*buckets, *max_distance, *bidirectional)
            .map(Some)
            .map_err(|error| {
                let name = match error {
                    T5BucketError::TooFewBuckets { .. }
                    | T5BucketError::OddBidirectionalBucketCount { .. } => "buckets",
                    T5BucketError::MaxDistanceNotBeyondExact { .. } => "max_distance",
                };
                Error::configuration(format!("{field}.{name}"), error.to_string())
            })
    }
}

/// Maps a shared RoPE construction error to the spec field a caller can repair.
fn rope_error_field(error: &RopeConfigError) -> &'static str {
    match error {
        RopeConfigError::ZeroHeadDimension => "head_dim",
        RopeConfigError::ZeroRotaryDimension
        | RopeConfigError::RotaryDimensionExceedsHead { .. }
        | RopeConfigError::OddRotaryDimension { .. }
        | RopeConfigError::InvalidPartialRotaryFactor { .. }
        | RopeConfigError::NonIntegralPartialRotaryDimension { .. }
        | RopeConfigError::DynamicNtkDimensionTooSmall { .. } => "rotary_dim",
        RopeConfigError::InvalidTheta { .. } | RopeConfigError::YarnThetaTooSmall { .. } => "theta",
        RopeConfigError::InvalidScalingFactor { .. } => "scaling.factor",
        RopeConfigError::ZeroOriginalContext { .. } => "scaling.original_max_positions",
        RopeConfigError::InvalidYarnAttentionFactor { .. } => "scaling.attention_factor",
        RopeConfigError::InvalidYarnBeta { .. } | RopeConfigError::InvalidYarnBetaOrder { .. } => {
            "scaling.beta"
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
    /// Whether the query projection also emits a per-head sigmoid output gate.
    ///
    /// The modular models map `true` to `AttentionOutputGate::Sigmoid`; a SiLU
    /// gate is available on `GroupedQueryAttentionConfig` but not through this spec.
    pub output_gate: bool,
    /// Position transformation used by this attention layer.
    pub position: PositionEncodingSpec,
}

impl AttentionSpec {
    /// Validates grouped-head divisibility and positional dimensions.
    ///
    /// Width-independent form of the check [`DecoderSpec::validate`] performs at
    /// the stack's residual width.
    pub fn validate(&self, field: &str) -> Result<()> {
        self.validate_for_width(field, 1)
    }

    /// Validates spec invariants, the shared projection config, and the position encoding.
    fn validate_for_width(&self, field: &str, hidden_size: usize) -> Result<()> {
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
        self.projection_config(hidden_size)
            .validate()
            .map_err(|error| nested(field, error))?;
        self.position
            .validate(self.head_dim, &format!("{field}.position"))
    }

    /// The shared projection configuration this spec describes at `hidden_size`.
    ///
    /// Mirrors the modular models: `output_gate` selects a sigmoid gate and T5
    /// relative bias uses unit score scaling. Q/K-norm epsilon comes from the
    /// block's normalization spec, which is validated separately.
    fn projection_config(&self, hidden_size: usize) -> GroupedQueryAttentionConfig {
        let mut config = GroupedQueryAttentionConfig::new(
            hidden_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
        );
        if self.output_gate {
            config.output_gate = AttentionOutputGate::Sigmoid;
        }
        if matches!(self.position, PositionEncodingSpec::T5RelativeBias { .. }) {
            config.scale = Some(1.0);
        }
        config
    }
}

/// Prefixes a shared component's configuration field with the spec path that produced it.
///
/// For example, an MLA error at `mla.qk_rope_head_dim` raised while validating
/// `layers[0].mixer` becomes `layers[0].mixer.mla.qk_rope_head_dim`.
fn nested(field: &str, error: Error) -> Error {
    match error {
        Error::Configuration {
            field: component,
            reason,
        } => Error::configuration(format!("{field}.{component}"), reason),
        other => other,
    }
}

/// Bounded-context attention that discards keys outside a trailing window.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SlidingWindowSpec {
    /// Attention topology evaluated inside the window.
    pub attention: AttentionSpec,
    /// Maximum number of valid keys visible to a query, including the query's own key.
    ///
    /// Masked padding columns do not consume the window. Its age is measured in valid-token
    /// order, independently of offsets or gaps in explicit positional-encoding IDs.
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
    Attention(
        /// Head grouping, projection dimensions, and positional encoding.
        AttentionSpec,
    ),
    /// Causal attention restricted to a trailing window.
    SlidingAttention(
        /// Attention geometry and the permitted trailing key window.
        SlidingWindowSpec,
    ),
    /// DeepSeek-style multi-head latent attention.
    Mla(
        /// Latent compression ranks and query/key/value head dimensions.
        MlaSpec,
    ),
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
    ///
    /// Describable and validated, but not yet executable: the modular models
    /// reject it at `init` with a capability error.
    Mamba2(
        /// Recurrent state, head grouping, and convolution dimensions.
        Mamba2Spec,
    ),
}

impl MixerSpec {
    /// Validates head grouping and non-zero recurrent-state dimensions.
    ///
    /// Each executable mixer is also checked by its shared component
    /// configuration (grouped-query attention, MLA, or Gated DeltaNet), so a
    /// mixer that validates can be constructed. This width-independent form
    /// uses residual width one; [`DecoderSpec::validate`] repeats the component
    /// checks at the stack's real width, which also bounds projection sizes.
    pub fn validate(&self, field: &str) -> Result<()> {
        self.validate_for_width(field, 1)
    }

    /// Validates this mixer as it would be built inside a stack of `hidden_size` channels.
    fn validate_for_width(&self, field: &str, hidden_size: usize) -> Result<()> {
        match self {
            Self::Attention(spec) => spec.validate_for_width(field, hidden_size),
            Self::SlidingAttention(spec) => {
                if spec.window == 0 {
                    return Err(Error::configuration(
                        format!("{field}.window"),
                        "must be non-zero",
                    ));
                }
                spec.attention
                    .validate_for_width(&format!("{field}.attention"), hidden_size)
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
                if spec
                    .content_head_dim
                    .checked_add(spec.rotary_head_dim)
                    .is_none()
                {
                    return Err(Error::configuration(
                        format!("{field}.head_dim"),
                        "MLA content and rotary dimensions overflow usize",
                    ));
                }
                MultiLatentAttentionConfig::new(
                    hidden_size,
                    spec.query_heads,
                    spec.content_head_dim,
                    spec.rotary_head_dim,
                    spec.value_head_dim,
                    spec.key_value_lora_rank,
                    spec.query_lora_rank,
                )
                .validate()
                .map_err(|error| nested(field, error))?;
                // MLA rotates only its decoupled rotary sub-head; the content
                // channels are never position-encoded.
                spec.position
                    .validate(spec.rotary_head_dim, &format!("{field}.position"))
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
                let mut config = GatedDeltaNetConfig::new(
                    hidden_size,
                    *key_heads,
                    *value_heads,
                    *key_head_dim,
                    *value_head_dim,
                );
                config.convolution_kernel = *convolution_kernel;
                config.validate().map_err(|error| nested(field, error))
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
    Dense(
        /// Intermediate width, activation, and projection-bias policy.
        DenseFeedForwardSpec,
    ),
    /// A sparse router plus independent expert networks.
    MixtureOfExperts(
        /// Router selection and per-expert network configuration.
        MixtureOfExpertsSpec,
    ),
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
    ///
    /// Reserved: the gate's activation, shape, and initialization are not yet
    /// specified, so the modular models reject it at `init`.
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
    ///
    /// Width-independent form of the per-layer check in [`DecoderSpec::validate`].
    pub fn validate(&self, index: usize) -> Result<()> {
        self.validate_for_width(index, 1)
    }

    /// Validates this layer as it would be built inside a stack of `hidden_size` channels.
    fn validate_for_width(&self, index: usize, hidden_size: usize) -> Result<()> {
        let field = format!("layers[{index}]");
        self.mixer
            .validate_for_width(&format!("{field}.mixer"), hidden_size)?;
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
            layer.validate_for_width(index, self.hidden_size)?;
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
    ///
    /// Cross-attention projects `hidden_size -> heads * head_dim -> hidden_size`,
    /// so the inner width may differ from the residual width (as in T5-3B/11B);
    /// it must only be nonzero and its projection matrices must not overflow.
    pub fn validate(&self) -> Result<()> {
        self.encoder.validate()?;
        self.decoder.validate()?;
        if self.encoder.hidden_size != self.decoder.hidden_size {
            return Err(Error::configuration(
                "decoder.hidden_size",
                "must equal encoder.hidden_size",
            ));
        }
        let projection_elements = self
            .cross_attention_heads
            .checked_mul(self.cross_attention_head_dim)
            .and_then(|inner| inner.checked_mul(self.decoder.hidden_size));
        if self.cross_attention_heads == 0
            || self.cross_attention_head_dim == 0
            || projection_elements.is_none()
        {
            return Err(Error::configuration(
                "cross_attention",
                "heads and head_dim must be non-zero, and hidden_size * heads * head_dim must fit in usize",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "spec_tests.rs"]
mod tests;
