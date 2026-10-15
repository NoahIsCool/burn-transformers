//! Translation of a gpt-oss configuration into an executable topology.
//!
//! Every gpt-oss block is
//!
//! ```text
//! x' = x + Attn(RMSNorm₁(x))         grouped-query attention with learned sinks
//! y  = x' + MoE(RMSNorm₂(x'))        biased top-k-then-softmax router, clamped SwiGLU experts
//! ```
//!
//! with RMSNorm `N(x) = cast(w · x_f32 · rsqrt(mean(x²) + ε))` (the scale is
//! applied in F32 before the single rounding, as `GptOssRMSNorm` does), and
//! attention whose `sliding_attention` layers see only the `W` newest keys.
//! [`GptOssArchitecture`] records these choices; the model reads nothing
//! else, so the shared attention, routing, and expert code never branches on
//! a model name.
//!
//! # Translation table
//!
//! | executable choice | source | rule |
//! | --- | --- | --- |
//! | q/k/v/o bias | `attention_bias` | all four or none ([`ProjectionBias::uniform`]) |
//! | attention sinks | always | one learned logit per query head, dropped after softmax |
//! | score scale | `head_dim` | `1/sqrt(head_dim)` (a power of two for 64, so sinks are exact on fused kernels) |
//! | layer `l` connectivity | `layer_types[l]` | `sliding_attention` → sliding window `sliding_window`, `full_attention` → global |
//! | RoPE | `rope_parameters`/`rope_scaling`, `rope_theta`, default 150000 | shared translator, whole heads; YaRN honors `truncate: false` |
//! | rotary precision | — | cosine/sine rounded to the activation dtype before rotating (`cos.to(x.dtype)`) |
//! | router | `num_local_experts`, `num_experts_per_tok` | `z = x Wᵀ + b`; top-`K` of `z`; `softmax` over the selected logits; no renormalization or scale |
//! | experts | `intermediate_size` | `down((clamp(u, ±7) + 1) · g · σ(1.702 g))`, `g = min(gate, 7)`, biased projections |
//! | weight format | `quantization_config` | absent: dense experts; `quant_method: mxfp4`: MXFP4 expert blocks |
//!
//! * **Router precision.** Transformers projects the router in the
//!   activation dtype and runs top-k and softmax there; this crate projects
//!   in the activation dtype and ranks and normalizes in F32. In F32 the two
//!   agree up to matmul reduction order. In BF16/F16 the projections
//!   accumulate differently, so router logits can differ by a rounding; the
//!   selected experts then agree except where the `K`-th and `(K + 1)`-th
//!   logits are equal or nearly so (exact ties go to the lowest expert index
//!   here, while `torch.topk` leaves their order unspecified), and the
//!   weights of agreeing selections differ by about one rounding of the
//!   activation dtype.
//! * **Rejected.** A `model_type` other than `gpt_oss`, non-zero
//!   `attention_dropout`, an activation other than `silu`, a `swiglu_limit`
//!   other than the reference's hard-coded `7.0`, a malformed `layer_types`,
//!   a `sliding_attention` layer without a window, a zero window, a
//!   `quantization_config` whose `quant_method` is not `mxfp4`, and any
//!   geometry whose matrices or routing cannot be formed.

use serde::{Deserialize, Serialize};

use super::config::{FAMILY, GptOssConfig, MODEL_TYPE, SWIGLU_LIMIT};
use crate::{
    attention::{AttentionConnectivity, GroupedQueryAttentionConfig, ProjectionBias},
    feed_forward::ClampedSwiGlu,
    models::config::{
        ConfigError, PartialRotary, RopeContext, divisible, nonzero, positive_finite, token_ids,
    },
    moe::{RouterConfig, RouterProjectionPrecision, RouterScoring, RoutingPolicy, ScoringOrder},
    normalization::{RmsNormConfig, RmsNormRootMode},
    position::RopeConfig,
};

/// Sigmoid sharpness `alpha` of the reference's clamped SwiGLU.
pub(super) const SWIGLU_ALPHA: f64 = 1.702;

/// Constant the reference adds to the clamped up projection.
pub(super) const SWIGLU_UP_OFFSET: f64 = 1.0;

/// How routed-expert projections are stored in the checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GptOssWeightFormat {
    /// Dense `gate_up_proj` `[E, d, 2f]` and `down_proj` `[E, f, d]` tensors.
    Dense,
    /// MXFP4 `*_blocks`/`*_scales` pairs, decoded exactly at load time.
    Mxfp4,
}

/// The validated, executable topology of one gpt-oss checkpoint.
///
/// Produced by [`GptOssConfig::architecture`]; every field is a direct input
/// of the model graph, the checkpoint contract, or the decode cache.
#[derive(Debug, Clone, PartialEq)]
pub struct GptOssArchitecture {
    /// Vocabulary size `V`.
    pub vocab_size: usize,
    /// Residual width `d`.
    pub hidden_size: usize,
    /// Expert intermediate width `f`.
    pub intermediate_size: usize,
    /// Positions must lie in `0..max_position_embeddings`.
    pub max_position_embeddings: usize,
    /// Query heads `H`.
    pub query_heads: usize,
    /// Key/value heads `K`, dividing `H`.
    pub key_value_heads: usize,
    /// Per-head width `h`.
    pub head_dim: usize,
    /// Whether the four attention projections carry biases.
    pub attention_bias: bool,
    /// RMSNorm epsilon.
    pub norm_epsilon: f64,
    /// Shared rotary configuration of every layer.
    pub rope: RopeConfig,
    /// Key visibility of each block, one entry per layer.
    pub layers: Vec<AttentionConnectivity>,
    /// Routed experts per block `E`.
    pub experts: usize,
    /// Experts selected per token `K`.
    pub experts_per_token: usize,
    /// The experts' clamped gated unit.
    pub glu: ClampedSwiGlu,
    /// Whether the LM head reuses the embedding table.
    pub tie_word_embeddings: bool,
    /// Storage format of the routed-expert projections.
    pub weight_format: GptOssWeightFormat,
}

impl GptOssArchitecture {
    /// Translates and validates `config`; see the module translation table.
    pub(super) fn from_config(config: &GptOssConfig) -> Result<Self, ConfigError> {
        if config.model_type != MODEL_TYPE {
            return Err(ConfigError::new(
                FAMILY,
                "model_type",
                format!("expected `{MODEL_TYPE}`, found `{}`", config.model_type),
            ));
        }
        validate_geometry(config)?;
        reject_unsupported(config)?;
        let architecture = Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            max_position_embeddings: config.max_position_embeddings,
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            attention_bias: config.attention_bias,
            norm_epsilon: config.rms_norm_eps,
            rope: rope(config)?,
            layers: layer_schedule(config)?,
            experts: config.num_local_experts,
            experts_per_token: config.num_experts_per_tok,
            glu: ClampedSwiGlu::new(SWIGLU_LIMIT)
                .with_alpha(SWIGLU_ALPHA)
                .with_up_offset(SWIGLU_UP_OFFSET),
            tie_word_embeddings: config.tie_word_embeddings,
            weight_format: weight_format(config)?,
        };
        for (field, ids) in [
            ("pad_token_id", &config.pad_token_id),
            ("bos_token_id", &config.bos_token_id),
            ("eos_token_id", &config.eos_token_id),
        ] {
            token_ids(FAMILY, field, ids.as_ref(), config.vocab_size)?;
        }
        architecture
            .attention_config()
            .validate()
            .map_err(|error| ConfigError::new(FAMILY, "attention", error.to_string()))?;
        architecture
            .router_config()
            .validate()
            .map_err(|error| ConfigError::new(FAMILY, "num_experts_per_tok", error.to_string()))?;
        Ok(architecture)
    }

    /// Returns the number of decoder blocks.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the shared grouped-query attention configuration of every block.
    ///
    /// Learned sinks are always on; biases follow `attention_bias`.
    #[must_use]
    pub fn attention_config(&self) -> GroupedQueryAttentionConfig {
        GroupedQueryAttentionConfig::new(
            self.hidden_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
        )
        .with_projection_bias(ProjectionBias::uniform(self.attention_bias))
        .with_attention_sinks(true)
    }

    /// Returns the routing policy of every block.
    ///
    /// The reference ranks raw logits (`torch.topk(router_logits, K)`) and
    /// applies softmax to the selected logits only, so the weights already
    /// sum to one: no renormalization and unit scale. The router has a bias
    /// and is projected in the activation dtype.
    #[must_use]
    pub fn router_config(&self) -> RouterConfig {
        RouterConfig {
            experts: self.experts,
            experts_per_token: self.experts_per_token,
            policy: RoutingPolicy {
                scoring: RouterScoring::Softmax,
                order: ScoringOrder::TopKThenScore,
                router_bias: true,
                projection: RouterProjectionPrecision::ModelDtype,
                ..RoutingPolicy::default()
            },
            renormalize_selected: false,
            routed_scale: 1.0,
        }
    }

    /// Returns the RMSNorm configuration over `size` channels.
    ///
    /// `GptOssRMSNorm` multiplies the F32 normalized states by its scale in
    /// F32 and rounds once (`(w · x).to(input_dtype)`), evaluating
    /// `x · rsqrt(mean(x²) + ε)`.
    #[must_use]
    pub fn rms_norm(&self, size: usize) -> RmsNormConfig {
        RmsNormConfig::new(size)
            .with_epsilon(self.norm_epsilon)
            .with_round_before_scale(false)
            .with_root_mode(RmsNormRootMode::MultiplyByReciprocalSqrt)
    }

    /// Returns the sliding window shared by every sliding layer, if any layer slides.
    #[must_use]
    pub fn sliding_window(&self) -> Option<usize> {
        self.layers.iter().find_map(|layer| match layer {
            AttentionConnectivity::SlidingWindow { window } => Some(*window),
            AttentionConnectivity::Global | AttentionConnectivity::ChunkedLocal { .. } => None,
        })
    }
}

/// Checks positive extents, grouped-query divisibility, and the epsilon.
fn validate_geometry(config: &GptOssConfig) -> Result<(), ConfigError> {
    for (field, value) in [
        ("vocab_size", config.vocab_size),
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("head_dim", config.head_dim),
        ("max_position_embeddings", config.max_position_embeddings),
        ("num_local_experts", config.num_local_experts),
        ("num_experts_per_tok", config.num_experts_per_tok),
    ] {
        nonzero(FAMILY, field, value)?;
    }
    divisible(
        FAMILY,
        "num_attention_heads",
        config.num_attention_heads,
        "num_key_value_heads",
        config.num_key_value_heads,
    )?;
    if config
        .intermediate_size
        .checked_mul(2)
        .and_then(|width| width.checked_mul(config.hidden_size))
        .and_then(|matrix| matrix.checked_mul(config.num_local_experts))
        .is_none()
    {
        return Err(ConfigError::new(
            FAMILY,
            "intermediate_size",
            "the stacked expert projections overflow usize",
        ));
    }
    positive_finite(FAMILY, "rms_norm_eps", config.rms_norm_eps)
}

/// Rejects settings the reference would ignore or this graph cannot honor.
fn reject_unsupported(config: &GptOssConfig) -> Result<(), ConfigError> {
    if config.attention_dropout != 0.0 {
        return Err(ConfigError::new(
            FAMILY,
            "attention_dropout",
            "attention dropout is not implemented; inference and training use 0.0",
        ));
    }
    if config.hidden_act != "silu" {
        return Err(ConfigError::new(
            FAMILY,
            "hidden_act",
            format!(
                "`{}` is ignored by the reference, whose experts hard-code the clamped SwiGLU; only `silu` is accepted",
                config.hidden_act
            ),
        ));
    }
    if config
        .swiglu_limit
        .is_some_and(|limit| limit != SWIGLU_LIMIT)
    {
        return Err(ConfigError::new(
            FAMILY,
            "swiglu_limit",
            format!("the reference hard-codes {SWIGLU_LIMIT} and ignores this field"),
        ));
    }
    Ok(())
}

/// Derives the expert storage format from `quantization_config`.
fn weight_format(config: &GptOssConfig) -> Result<GptOssWeightFormat, ConfigError> {
    let Some(quantization) = &config.quantization_config else {
        return Ok(GptOssWeightFormat::Dense);
    };
    match quantization
        .get("quant_method")
        .and_then(|method| method.as_str())
    {
        Some("mxfp4") => Ok(GptOssWeightFormat::Mxfp4),
        other => Err(ConfigError::new(
            FAMILY,
            "quantization_config",
            format!(
                "quant_method {other:?} is not supported; gpt-oss loads dense or MXFP4 expert weights"
            ),
        )),
    }
}

/// Translates the serialized RoPE fields through the shared translator.
fn rope(config: &GptOssConfig) -> Result<RopeConfig, ConfigError> {
    let context = RopeContext::new(FAMILY, config.head_dim, config.max_position_embeddings)
        .with_rope_theta(config.rope_theta)
        .with_default_rope_theta(GptOssConfig::default_rope_theta())
        .with_partial_rotary(PartialRotary::FullHeadOnly);
    config
        .rope_parameters
        .to_rope_config(&context)
        .map_err(|error| ConfigError::new(FAMILY, "rope_parameters", error.to_string()))
}

/// Parses `layer_types` into per-layer connectivity.
fn layer_schedule(config: &GptOssConfig) -> Result<Vec<AttentionConnectivity>, ConfigError> {
    let layers = config.num_hidden_layers;
    if config.layer_types.len() != layers {
        return Err(ConfigError::new(
            FAMILY,
            "layer_types",
            format!(
                "must list {layers} entries, found {}",
                config.layer_types.len()
            ),
        ));
    }
    if config.sliding_window == Some(0) {
        return Err(ConfigError::new(
            FAMILY,
            "sliding_window",
            "must be nonzero",
        ));
    }
    config
        .layer_types
        .iter()
        .enumerate()
        .map(
            |(layer, kind)| match (kind.as_str(), config.sliding_window) {
                ("full_attention", _) => Ok(AttentionConnectivity::Global),
                ("sliding_attention", Some(window)) => {
                    Ok(AttentionConnectivity::SlidingWindow { window })
                }
                ("sliding_attention", None) => Err(ConfigError::new(
                    FAMILY,
                    "sliding_window",
                    format!("layer {layer} is sliding_attention but no window is configured"),
                )),
                (other, _) => Err(ConfigError::new(
                    FAMILY,
                    "layer_types",
                    format!("`{other}` is not full_attention or sliding_attention"),
                )),
            },
        )
        .collect()
}

#[cfg(test)]
#[path = "architecture_tests.rs"]
mod tests;
