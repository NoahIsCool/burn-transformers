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
mod tests {
    use super::*;

    /// Builds the smallest valid dense attention layer used by validation tests.
    fn dense_layer() -> DecoderLayerSpec {
        DecoderLayerSpec {
            mixer: MixerSpec::Attention(AttentionSpec {
                query_heads: 4,
                key_value_heads: 2,
                head_dim: 8,
                query_key_norm: true,
                output_gate: false,
                position: PositionEncodingSpec::Rope {
                    theta: 10_000.0,
                    rotary_dim: 8,
                    scaling: RopeScaling::None,
                },
            }),
            feed_forward: FeedForwardSpec::Dense(DenseFeedForwardSpec {
                intermediate_size: 64,
                activation: Activation::Silu,
                bias: false,
            }),
            mixer_norm: NormalizationSpec::Rms {
                epsilon: 1e-6,
                zero_centered: false,
            },
            feed_forward_norm: NormalizationSpec::Rms {
                epsilon: 1e-6,
                zero_centered: false,
            },
            norm_placement: NormPlacement::Pre,
            residual: ResidualTopology::Add,
        }
    }

    /// Builds the smallest valid modular decoder.
    fn decoder() -> DecoderSpec {
        DecoderSpec {
            vocabulary_size: 128,
            hidden_size: 32,
            layers: vec![dense_layer()],
            final_norm: NormalizationSpec::Rms {
                epsilon: 1e-6,
                zero_centered: false,
            },
            tie_word_embeddings: false,
            max_position_embeddings: 256,
        }
    }

    /// Accepts a valid grouped-query decoder whose query width equals the residual stream.
    #[test]
    fn decoder_validation_accepts_valid_spec() {
        decoder().validate().expect("valid decoder must pass");
    }

    /// Configuration errors retain the exact nested field a caller can repair.
    fn assert_configuration_field(result: Result<()>, expected: &str) {
        match result.expect_err("invalid configuration must return an error") {
            Error::Configuration { field, reason } => {
                assert_eq!(field, expected);
                assert!(!reason.is_empty());
            }
            error => panic!("expected configuration error for {expected}, got {error}"),
        }
    }

    /// Both normalization variants reject non-positive and non-finite stabilizers.
    #[test]
    fn normalization_validates_each_epsilon_boundary() {
        for epsilon in [0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            for norm in [
                NormalizationSpec::Rms {
                    epsilon,
                    zero_centered: true,
                },
                NormalizationSpec::Layer {
                    epsilon,
                    bias: true,
                },
            ] {
                assert_configuration_field(norm.validate("norm"), "norm.epsilon");
            }
        }
        NormalizationSpec::Layer {
            epsilon: f64::MIN_POSITIVE,
            bias: false,
        }
        .validate("norm")
        .unwrap();
    }

    /// Rotary pairs and their geometric base are validated independently of head grouping.
    #[test]
    fn position_encoding_rejects_invalid_base_width_and_bucket_distance() {
        for theta in [0.0, -10.0, f64::NAN, f64::INFINITY] {
            assert_configuration_field(
                PositionEncodingSpec::Rope {
                    theta,
                    rotary_dim: 2,
                    scaling: RopeScaling::None,
                }
                .validate(4, "position"),
                "position.theta",
            );
        }
        for rotary_dim in [0, 3, 6] {
            assert_configuration_field(
                PositionEncodingSpec::Rope {
                    theta: 10_000.0,
                    rotary_dim,
                    scaling: RopeScaling::None,
                }
                .validate(4, "position"),
                "position.rotary_dim",
            );
        }
        for buckets in [0, 1] {
            assert_configuration_field(
                PositionEncodingSpec::T5RelativeBias {
                    buckets,
                    max_distance: 16,
                    bidirectional: false,
                }
                .validate(4, "position"),
                "position.buckets",
            );
        }
        for max_distance in [0, 15, 16] {
            assert_configuration_field(
                PositionEncodingSpec::T5RelativeBias {
                    buckets: 32,
                    max_distance,
                    bidirectional: false,
                }
                .validate(4, "position"),
                "position.max_distance",
            );
        }
        PositionEncodingSpec::None.validate(0, "position").unwrap();
        PositionEncodingSpec::T5RelativeBias {
            buckets: 32,
            max_distance: 17,
            bidirectional: false,
        }
        .validate(4, "position")
        .unwrap();
    }

    /// YaRN rejects each singular interpolation/temperature boundary with its own field.
    #[test]
    fn yarn_validation_reports_expansion_temperature_and_transition_errors() {
        let validate = |factor, original_max_positions, attention_factor, beta_slow, beta_fast| {
            PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: 4,
                scaling: RopeScaling::Yarn {
                    factor,
                    original_max_positions,
                    attention_factor,
                    beta_slow,
                    beta_fast,
                },
            }
            .validate(4, "position")
        };
        validate(2.0, 128, None, 1.0, 32.0).unwrap();
        validate(2.0, 128, Some(1.0), 1.0, 32.0).unwrap();
        for factor in [1.0, 0.0, f64::NAN, f64::INFINITY] {
            assert_configuration_field(
                validate(factor, 128, None, 1.0, 32.0),
                "position.scaling.factor",
            );
        }
        assert_configuration_field(
            validate(2.0, 0, None, 1.0, 32.0),
            "position.scaling.original_max_positions",
        );
        for factor in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_configuration_field(
                validate(2.0, 128, Some(factor), 1.0, 32.0),
                "position.scaling.attention_factor",
            );
        }
        for (slow, fast) in [
            (0.0, 32.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (f64::NAN, 32.0),
            (1.0, f64::INFINITY),
        ] {
            assert_configuration_field(
                validate(2.0, 128, None, slow, fast),
                "position.scaling.beta",
            );
        }
    }

    /// Every attention and recurrent-state axis must be nonzero before shape arithmetic.
    #[test]
    fn mixers_validate_zero_dimensions_and_grouping() {
        for index in 0..3 {
            let mut axes = [2, 1, 4];
            axes[index] = 0;
            assert_configuration_field(
                AttentionSpec {
                    query_heads: axes[0],
                    key_value_heads: axes[1],
                    head_dim: axes[2],
                    query_key_norm: false,
                    output_gate: false,
                    position: PositionEncodingSpec::None,
                }
                .validate("attention"),
                "attention",
            );
        }
        let attention = AttentionSpec {
            query_heads: 2,
            key_value_heads: 1,
            head_dim: 4,
            query_key_norm: false,
            output_gate: false,
            position: PositionEncodingSpec::None,
        };
        assert_configuration_field(
            MixerSpec::SlidingAttention(SlidingWindowSpec {
                attention: attention.clone(),
                window: 0,
            })
            .validate("mixer"),
            "mixer.window",
        );
        MixerSpec::SlidingAttention(SlidingWindowSpec {
            attention,
            window: 1,
        })
        .validate("mixer")
        .unwrap();
        for index in 0..5 {
            let mut axes = [1, 2, 4, 4, 3];
            axes[index] = 0;
            assert_configuration_field(
                MixerSpec::DeltaNet {
                    key_heads: axes[0],
                    value_heads: axes[1],
                    key_head_dim: axes[2],
                    value_head_dim: axes[3],
                    convolution_kernel: axes[4],
                }
                .validate("delta"),
                "delta",
            );
        }
        assert_configuration_field(
            MixerSpec::DeltaNet {
                key_heads: 2,
                value_heads: 3,
                key_head_dim: 4,
                value_head_dim: 4,
                convolution_kernel: 3,
            }
            .validate("delta"),
            "delta.value_heads",
        );
        for index in 0..5 {
            let mut axes = [4, 4, 8, 3, 2];
            axes[index] = 0;
            assert_configuration_field(
                MixerSpec::Mamba2(Mamba2Spec {
                    heads: axes[0],
                    head_dim: axes[1],
                    state_dim: axes[2],
                    convolution_kernel: axes[3],
                    group_size: axes[4],
                })
                .validate("mamba"),
                "mamba",
            );
        }
        for group_size in [1, 2, 3, 4, 5] {
            let result = MixerSpec::Mamba2(Mamba2Spec {
                heads: 4,
                head_dim: 4,
                state_dim: 8,
                convolution_kernel: 3,
                group_size,
            })
            .validate("mamba");
            if [1, 2, 4].contains(&group_size) {
                result.unwrap();
            } else {
                assert_configuration_field(result, "mamba");
            }
        }
    }

    /// Serialized MLA dimensions cannot overflow even before a runtime rejects the mixer.
    #[test]
    fn mla_validation_rejects_zero_ranks_and_overflowing_head_width() {
        let valid = MlaSpec {
            query_heads: 2,
            content_head_dim: 4,
            rotary_head_dim: 2,
            value_head_dim: 4,
            key_value_lora_rank: 2,
            query_lora_rank: None,
            position: PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: 2,
                scaling: RopeScaling::None,
            },
        };
        MixerSpec::Mla(valid.clone()).validate("mla").unwrap();
        for index in 0..6 {
            let mut invalid = valid.clone();
            match index {
                0 => invalid.query_heads = 0,
                1 => invalid.content_head_dim = 0,
                2 => invalid.rotary_head_dim = 0,
                3 => invalid.value_head_dim = 0,
                4 => invalid.key_value_lora_rank = 0,
                _ => invalid.query_lora_rank = Some(0),
            }
            assert_configuration_field(MixerSpec::Mla(invalid).validate("mla"), "mla");
        }
        let mut invalid = valid;
        invalid.content_head_dim = usize::MAX;
        assert_configuration_field(MixerSpec::Mla(invalid).validate("mla"), "mla.head_dim");
    }

    /// Dense widths, routed cardinalities and scaling each have recoverable diagnostics.
    #[test]
    fn feed_forward_validation_covers_empty_experts_and_invalid_scales() {
        assert_configuration_field(
            FeedForwardSpec::Dense(DenseFeedForwardSpec {
                intermediate_size: 0,
                activation: Activation::Silu,
                bias: false,
            })
            .validate("ffn"),
            "ffn.intermediate_size",
        );
        let valid = MixtureOfExpertsSpec {
            routed_experts: 4,
            experts_per_token: 2,
            shared_experts: 1,
            expert_intermediate_size: 8,
            activation: Activation::ReluSquared,
            renormalize_selected: true,
            routed_scale: 1.0,
        };
        FeedForwardSpec::MixtureOfExperts(valid.clone())
            .validate("ffn")
            .unwrap();
        for index in 0..3 {
            let mut invalid = valid.clone();
            match index {
                0 => invalid.routed_experts = 0,
                1 => invalid.expert_intermediate_size = 0,
                _ => invalid.experts_per_token = 0,
            }
            assert_configuration_field(
                FeedForwardSpec::MixtureOfExperts(invalid).validate("ffn"),
                if index == 2 {
                    "ffn.experts_per_token"
                } else {
                    "ffn"
                },
            );
        }
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut invalid = valid.clone();
            invalid.routed_scale = scale;
            assert_configuration_field(
                FeedForwardSpec::MixtureOfExperts(invalid).validate("ffn"),
                "ffn.routed_scale",
            );
        }
    }

    /// Global and later-layer validation errors retain precise field paths.
    #[test]
    fn decoder_reports_global_and_nested_normalization_fields() {
        for (index, expected) in [
            "vocabulary_size",
            "hidden_size",
            "layers",
            "max_position_embeddings",
            "final_norm.epsilon",
        ]
        .into_iter()
        .enumerate()
        {
            let mut invalid = decoder();
            match index {
                0 => invalid.vocabulary_size = 0,
                1 => invalid.hidden_size = 0,
                2 => invalid.layers.clear(),
                3 => invalid.max_position_embeddings = 0,
                _ => {
                    invalid.final_norm = NormalizationSpec::Layer {
                        epsilon: 0.0,
                        bias: false,
                    }
                }
            }
            assert_configuration_field(invalid.validate(), expected);
        }
        for mixer_side in [true, false] {
            let mut invalid = decoder();
            invalid.layers.push(dense_layer());
            let invalid_norm = NormalizationSpec::Rms {
                epsilon: 0.0,
                zero_centered: false,
            };
            if mixer_side {
                invalid.layers[1].mixer_norm = invalid_norm;
            } else {
                invalid.layers[1].feed_forward_norm = invalid_norm;
            }
            assert_configuration_field(
                invalid.validate(),
                if mixer_side {
                    "layers[1].mixer_norm.epsilon"
                } else {
                    "layers[1].feed_forward_norm.epsilon"
                },
            );
        }
    }

    /// Rejects KV head counts that cannot be repeated evenly across query heads.
    #[test]
    fn attention_rejects_non_divisible_grouping() {
        let mut spec = decoder();
        let MixerSpec::Attention(attention) = &mut spec.layers[0].mixer else {
            unreachable!("fixture uses attention")
        };
        attention.key_value_heads = 3;
        let error = spec.validate().expect_err("invalid GQA ratio must fail");
        assert!(error.to_string().contains("must divide query_heads"));
    }

    /// Rejects odd rotary dimensions because channel pairs would be incomplete.
    #[test]
    fn rope_rejects_unpaired_channel() {
        let mut spec = decoder();
        let MixerSpec::Attention(attention) = &mut spec.layers[0].mixer else {
            unreachable!("fixture uses attention")
        };
        let PositionEncodingSpec::Rope { rotary_dim, .. } = &mut attention.position else {
            unreachable!("fixture uses RoPE")
        };
        *rotary_dim = 7;
        let error = spec.validate().expect_err("odd rotary width must fail");
        assert!(error.to_string().contains("non-zero even number"));
    }

    /// Rejects routers that select more experts than the checkpoint owns.
    #[test]
    fn moe_rejects_impossible_top_k() {
        let spec = FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
            routed_experts: 4,
            experts_per_token: 5,
            shared_experts: 0,
            expert_intermediate_size: 16,
            activation: Activation::Silu,
            renormalize_selected: true,
            routed_scale: 1.0,
        });
        let error = spec
            .validate("moe")
            .expect_err("top-k exceeds expert count");
        assert!(error.to_string().contains("1..=routed_experts"));
    }

    /// Validates matching encoder and decoder widths for cross attention.
    #[test]
    fn seq2seq_requires_matching_residual_widths() {
        let encoder = decoder();
        let mut decoder = decoder();
        decoder.hidden_size = 64;
        let MixerSpec::Attention(attention) = &mut decoder.layers[0].mixer else {
            unreachable!("fixture uses attention")
        };
        attention.head_dim = 16;
        let spec = Seq2SeqSpec {
            encoder,
            decoder,
            cross_attention_heads: 4,
            cross_attention_head_dim: 8,
        };
        let error = spec.validate().expect_err("mismatched stacks must fail");
        assert!(error.to_string().contains("encoder.hidden_size"));
    }

    /// Rejects overflow before it can panic or wrap into a valid residual width.
    #[test]
    fn seq2seq_rejects_overflowing_cross_attention_width() {
        let spec = Seq2SeqSpec {
            encoder: decoder(),
            decoder: decoder(),
            // On every supported usize width this product wraps to 32, matching
            // the fixture's hidden size and bypassing unchecked release builds.
            cross_attention_heads: usize::MAX / 2 + 17,
            cross_attention_head_dim: 2,
        };
        let error = spec.validate().expect_err("overflow must be rejected");
        assert!(error.to_string().contains("cross_attention"));
    }

    /// Accepts exact dimensions but rejects zero and mismatched products.
    #[test]
    fn seq2seq_checks_cross_attention_width() {
        for (heads, head_dim, valid) in [(4, 8, true), (0, 8, false), (4, 0, false), (3, 8, false)]
        {
            let spec = Seq2SeqSpec {
                encoder: decoder(),
                decoder: decoder(),
                cross_attention_heads: heads,
                cross_attention_head_dim: head_dim,
            };
            assert_eq!(spec.validate().is_ok(), valid);
        }
    }

    proptest::proptest! {
        /// Checks every valid GQA divisor accepted by the arithmetic invariant.
        #[test]
        fn gqa_accepts_exact_divisors(query_heads in 1usize..64, divisor in 1usize..64) {
            let key_value_heads = divisor.min(query_heads);
            let spec = AttentionSpec {
                query_heads,
                key_value_heads,
                head_dim: 2,
                query_key_norm: false,
                output_gate: false,
                position: PositionEncodingSpec::Rope {
                    theta: 10_000.0,
                    rotary_dim: 2,
                    scaling: RopeScaling::None,
                },
            };
            proptest::prop_assert_eq!(
                spec.validate("attention").is_ok(),
                query_heads.is_multiple_of(key_value_heads)
            );
        }
    }
}
