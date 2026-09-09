//! Executable Qwen3 sparse mixture-of-experts decoder.
//!
//! Every block is pre-normalized and computes
//! `x' = x + Attention(RMSNorm(x))`, followed by
//! `y = x' + MoE(RMSNorm(x'))`. Qwen3 uses grouped-query attention with
//! per-head Q/K RMSNorm and full-head RoPE. The official 30B-A3B topology puts
//! a bias-free 128-expert SwiGLU router in all 48 blocks, chooses eight experts
//! per token, renormalizes their probabilities, and has no shared expert.
//! Qwen3-MoE RMSNorm evaluates the mean square in FP32, rounds the resulting
//! unit-RMS activation back to the residual dtype, and only then applies the
//! learned scale: `u = x_f32 / sqrt(mean(x_f32^2) + epsilon)` and
//! `y = weight * cast_input_dtype(u)`. That last cast precedes multiplication,
//! which is observable for F16 activations.
//! Expert route weights follow a different rounding boundary: normalized FP32
//! probabilities multiply the reduced-precision expert result in FP32, then
//! each weighted contribution is rounded to the residual dtype immediately
//! before index-add. Router diagnostics expose the raw linear projections, but
//! the optional family loss reproduces Transformers 5.2's unusual stack-wide
//! `softmax(softmax(raw))` objective through
//! [`qwen3_moe_load_balancing_loss`].
//!
//! This module deliberately uses the shared attention, normalization, position,
//! feed-forward, and sparse-dispatch implementations. It owns only the exact
//! family wiring and output-policy decisions. Cache tensors live in
//! [`Qwen3MoeDecodeState`] rather than this Burn [`Module`].

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{
        Bool, DType, ElementConversion, Int, Tensor, TensorData, activation::softmax,
        backend::Backend,
    },
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, LogitsSelection,
    Result,
    architecture::{RouterDiagnostics, TokenMask},
    attention::{GroupedQueryAttention, GroupedQueryAttentionConfig, padding_attention_mask},
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    moe::{
        RoutedWeightingPrecision, SparseMoe, SparseMoeConfig, SparseMoeOutput, deterministic_top_k,
        load_balancing_loss,
    },
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{
        DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling, YarnConfig, apply_rotary,
        rotary_angles,
    },
};

use super::Qwen3MoeConfig;
use crate::models::config::RopeType;

mod decode;

pub use decode::Qwen3MoeDecodeState;

/// One decoder block with GQA and either a routed or dense SwiGLU branch.
#[derive(Module, Debug)]
pub struct Qwen3MoeDecoderLayer<B: Backend> {
    /// Grouped-query self-attention with per-head Q/K normalization.
    pub self_attn: GroupedQueryAttention<B>,
    /// Routed expert bank for sparse layers.
    pub sparse_mlp: Option<SparseMoe<B>>,
    /// Ordinary SwiGLU network for explicitly dense layers.
    pub dense_mlp: Option<GatedFeedForward<B>>,
    /// RMSNorm applied before self-attention.
    pub input_layernorm: RmsNorm<B>,
    /// RMSNorm applied before the feed-forward branch.
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> Qwen3MoeDecoderLayer<B> {
    /// Allocates one layer whose feed-forward kind follows the validated schedule.
    fn init(config: &Qwen3MoeConfig, layer_index: usize, device: &B::Device) -> Result<Self> {
        let mut attention = GroupedQueryAttentionConfig::new(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        );
        attention.bias = config.attention_bias;
        attention.query_key_norm_epsilon = Some(config.rms_norm_eps);
        let norm = || {
            RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_round_before_scale(true)
                .init(device)
        };
        let (sparse_mlp, dense_mlp) = if config.is_moe_layer(layer_index) {
            let mut sparse = SparseMoeConfig::new(
                config.hidden_size,
                config.moe_intermediate_size,
                config.num_experts,
                config.num_experts_per_tok,
            );
            sparse.renormalize_selected = config.norm_topk_prob;
            sparse.weighting_precision = RoutedWeightingPrecision::Fp32ThenResidual;
            (Some(sparse.init(device)?), None)
        } else {
            (
                None,
                Some(
                    GatedFeedForwardConfig::new(config.hidden_size, config.intermediate_size)
                        .init(device)?,
                ),
            )
        };
        let mut self_attn = attention.init(device)?;
        // Q/K norms use the same upstream RMSNorm class as the residual-stream
        // norms. The shared attention builder intentionally defaults to late
        // scaling, so select Qwen3-MoE's reduced-precision ordering here.
        for attention_norm in [&mut self_attn.query_norm, &mut self_attn.key_norm] {
            if let Some(attention_norm) = attention_norm.as_mut() {
                attention_norm.round_before_scale = true;
            }
        }
        Ok(Self {
            self_attn,
            sparse_mlp,
            dense_mlp,
            input_layernorm: norm()?,
            post_attention_layernorm: norm()?,
        })
    }

    /// Evaluates a stateless block and returns routing state for an MoE layer.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        rope: RopeConfig,
        evaluated_context: usize,
        sliding_window: Option<usize>,
    ) -> Result<(Tensor<B, 3>, Option<SparseMoeOutput<B>>)> {
        let residual = hidden.clone();
        let normalized = self.input_layernorm.forward(hidden);
        let mut projection = self.self_attn.project_self(normalized)?;
        let angles = rotary_angles(rope, positions, evaluated_context)?;
        (projection.query, projection.key) =
            apply_rotary(projection.query, projection.key, &angles)?;
        let [batch, _, sequence, _] = projection.query.dims();
        let mask = stateless_attention_mask(
            token_mask,
            batch,
            self.self_attn.query_heads,
            sequence,
            sliding_window,
            &projection.query.device(),
        )?;
        let hidden = residual + self.self_attn.attend(projection, mask, None, true)?;
        let residual = hidden.clone();
        let normalized = self.post_attention_layernorm.forward(hidden);
        match (&self.sparse_mlp, &self.dense_mlp) {
            (Some(moe), None) => {
                let output = moe.forward(normalized)?;
                Ok((residual + output.hidden_state.clone(), Some(output)))
            }
            (None, Some(mlp)) => Ok((residual + mlp.forward(normalized)?, None)),
            _ => Err(Error::Checkpoint(
                "Qwen3 MoE layer has an invalid feed-forward topology".into(),
            )),
        }
    }
}

/// Qwen3 token embedding, decoder stack, and final normalization.
#[derive(Module, Debug)]
pub struct Qwen3MoeModel<B: Backend> {
    /// Token embedding table.
    pub embed_tokens: Embedding<B>,
    /// Ordered decoder layers.
    pub layers: Vec<Qwen3MoeDecoderLayer<B>>,
    /// Final RMSNorm.
    pub norm: RmsNorm<B>,
    /// Validated family configuration excluded from parameter records.
    #[module(skip)]
    pub config: Qwen3MoeConfig,
    /// Validated rotary configuration.
    #[module(skip)]
    pub rope: RopeConfig,
}

impl<B: Backend> Qwen3MoeModel<B> {
    /// Initializes the exact configured dense/MoE schedule.
    pub fn init(config: Qwen3MoeConfig, device: &B::Device) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        if config.attention_dropout != 0.0 {
            return Err(Error::BackendCapability(
                "Qwen3 MoE runtime currently supports the released zero-dropout topology only"
                    .into(),
            ));
        }
        let rope = qwen_rope_config(&config)?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| Qwen3MoeDecoderLayer::init(&config, index, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers,
            norm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_round_before_scale(true)
                .init(device)?,
            config,
            rope,
        })
    }

    /// Evaluates the backbone and collects only requested diagnostics.
    fn forward(
        &self,
        input: &CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<Qwen3MoeBackboneOutput<B>> {
        let [batch, sequence] = input.token_ids.dims();
        validate_nonempty_input(batch, sequence)?;
        validate_input_devices(input, &self.embed_tokens.weight.lazy_device())?;
        validate_token_ids(input.token_ids.clone(), self.config.vocab_size)?;
        validate_mask(input.token_mask.as_ref(), batch, sequence, false)?;
        let positions = resolve_positions(input, batch, sequence)?;
        validate_position_range(positions.clone(), self.config.max_position_embeddings)?;
        let evaluated_context = evaluated_context(positions.clone())?;
        let mut hidden = self.embed_tokens.forward(input.token_ids.clone());
        let mut hidden_states = Vec::with_capacity(if options.hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        let retain_router_logits = options.router_logits || self.config.output_router_logits;
        let sparse_layers = self
            .layers
            .iter()
            .filter(|layer| layer.sparse_mlp.is_some())
            .count();
        let mut routers = Vec::with_capacity(if retain_router_logits {
            sparse_layers
        } else {
            0
        });
        let mut auxiliary_router_logits = Vec::with_capacity(if options.auxiliary_losses {
            sparse_layers
        } else {
            0
        });
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let window = self.config.layer_sliding_window(index);
            let (next, routed) = layer.forward(
                hidden,
                positions.clone(),
                input.token_mask.as_ref(),
                self.rope,
                evaluated_context,
                window,
            )?;
            hidden = next;
            if options.hidden_states {
                hidden_states.push(hidden.clone());
            }
            if let Some(routed) = routed {
                if options.auxiliary_losses {
                    auxiliary_router_logits.push(routed.router_logits.clone());
                }
                if retain_router_logits {
                    routers.push(router_diagnostics(&routed));
                }
            }
        }
        hidden = self.norm.forward(hidden);
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(Qwen3MoeBackboneOutput {
            hidden,
            hidden_states,
            routers,
            auxiliary_router_logits,
        })
    }
}

/// Internal backbone result whose diagnostic fields evolve independently.
struct Qwen3MoeBackboneOutput<B: Backend> {
    hidden: Tensor<B, 3>,
    hidden_states: Vec<Tensor<B, 3>>,
    routers: Vec<RouterDiagnostics<B>>,
    auxiliary_router_logits: Vec<Tensor<B, 3>>,
}

/// Decoder-only Qwen3 MoE language model.
#[derive(Module, Debug)]
pub struct Qwen3MoeForCausalLm<B: Backend> {
    /// Qwen3 transformer backbone.
    pub model: Qwen3MoeModel<B>,
    /// Independent vocabulary projection, omitted for tied checkpoints.
    pub lm_head: Option<Linear<B>>,
}

impl<B: Backend> Qwen3MoeForCausalLm<B> {
    /// Initializes a validated full-precision graph.
    pub fn init(config: Qwen3MoeConfig, device: &B::Device) -> Result<Self> {
        let hidden = config.hidden_size;
        let vocabulary = config.vocab_size;
        let tied = config.tie_word_embeddings;
        Ok(Self {
            model: Qwen3MoeModel::init(config, device)?,
            lm_head: (!tied).then(|| {
                operations::init_linear(
                    &LinearConfig::new(hidden, vocabulary)
                        .with_bias(false)
                        .with_layout(LinearLayout::Col),
                    device,
                )
            }),
        })
    }

    /// Evaluates differentiable stateless inference or training math.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Qwen3 MoE attention-probability capture is not yet exposed by Burn's fused path"
                    .into(),
            ));
        }
        let backbone = self.model.forward(&input, options)?;
        let load_balancing_loss = options
            .auxiliary_losses
            .then(|| {
                qwen3_moe_load_balancing_loss(
                    &backbone.auxiliary_router_logits,
                    self.config().num_experts_per_tok,
                    input.token_mask.as_ref(),
                )
            })
            .transpose()?;
        let hidden = backbone.hidden;
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
        let logits = self.project_logits(selected)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states: backbone.hidden_states,
            attentions: Vec::new(),
            routers: backbone.routers,
            load_balancing_loss,
        })
    }

    /// Returns the validated checkpoint configuration.
    #[must_use]
    pub const fn config(&self) -> &Qwen3MoeConfig {
        &self.model.config
    }

    /// Applies the independent or tied vocabulary projection.
    fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let hidden_device = hidden.device();
        match &self.lm_head {
            Some(head) => {
                ensure_model_device(
                    "qwen3_moe.lm_head",
                    &head.weight.lazy_device(),
                    &hidden_device,
                )?;
                operations::linear(hidden, head)
            }
            None => {
                ensure_model_device(
                    "qwen3_moe.model.embed_tokens",
                    &self.model.embed_tokens.weight.lazy_device(),
                    &hidden_device,
                )?;
                Ok(burn::tensor::module::linear(
                    hidden,
                    self.model.embed_tokens.weight.val().transpose(),
                    None,
                ))
            }
        }
    }
}

impl Qwen3MoeConfig {
    /// Returns the local-attention window for a layer, when enabled.
    ///
    /// Transformers 4.51's Qwen3-MoE implementation leaves the prefix
    /// `0..max_window_layers` on full attention and applies the finite window
    /// starting exactly at `max_window_layers`. Released Qwen3-30B-A3B disables
    /// the feature, so every layer in that checkpoint returns `None`.
    #[must_use]
    pub fn layer_sliding_window(&self, layer_index: usize) -> Option<usize> {
        (self.use_sliding_window && layer_index >= self.max_window_layers)
            .then_some(self.sliding_window)
            .flatten()
    }
}

/// Converts serialized RoPE metadata into the shared validated representation.
fn qwen_rope_config(config: &Qwen3MoeConfig) -> Result<RopeConfig> {
    let parameters = config.rope_parameters.clone().unwrap_or_default();
    let theta = parameters
        .rope_theta
        .or(config.rope_theta)
        .unwrap_or(10_000.0);
    let scaling = match parameters.rope_type {
        RopeType::Default => RopeScaling::None,
        RopeType::Linear => LinearRopeScaling::new(parameters.factor.unwrap_or(1.0))
            .map(RopeScaling::from)
            .map_err(|error| Error::configuration("qwen3_moe.rope", error.to_string()))?,
        RopeType::Dynamic => DynamicNtkScaling::new(
            parameters.factor.unwrap_or(1.0),
            parameters
                .original_max_position_embeddings
                .unwrap_or(config.max_position_embeddings),
        )
        .map(RopeScaling::from)
        .map_err(|error| Error::configuration("qwen3_moe.rope", error.to_string()))?,
        RopeType::Yarn => {
            let mut yarn = YarnConfig::new(
                parameters.factor.unwrap_or(1.0),
                parameters
                    .original_max_position_embeddings
                    .unwrap_or(config.max_position_embeddings),
            )
            .map_err(|error| Error::configuration("qwen3_moe.rope", error.to_string()))?;
            if let Some(factor) = parameters.attention_factor {
                yarn = yarn
                    .with_attention_factor(factor)
                    .map_err(|error| Error::configuration("qwen3_moe.rope", error.to_string()))?;
            }
            yarn = yarn
                .with_beta_overrides(parameters.beta_fast, parameters.beta_slow)
                .map_err(|error| Error::configuration("qwen3_moe.rope", error.to_string()))?;
            RopeScaling::Yarn(yarn)
        }
        unsupported => {
            return Err(Error::configuration(
                "qwen3_moe.rope_type",
                format!("{unsupported:?} is not implemented for Qwen3 MoE"),
            ));
        }
    };
    RopeConfig::new(config.head_dim, config.head_dim, theta, scaling)
        .map_err(|error| Error::configuration("qwen3_moe.rope", error.to_string()))
}

/// Computes the pinned Transformers 5.2 Qwen3-MoE stack-wide balance objective.
///
/// The upstream output recorder captures each router *after* its FP32 softmax,
/// despite naming the captured tensor `router_logits`. Its causal-LM wrapper
/// concatenates every layer/token row, applies softmax a second time, recomputes
/// top-k assignments, then evaluates the Switch Transformer balance objective
///
/// `E * sum_(j,e) mean_(l,b,t)(a[l,b,t,j,e]) * mean_(l,b,t)(q[l,b,t,e])`,
///
/// where `p = softmax(raw_logits)`, `q = softmax(p)`, and `a` is the one-hot
/// top-k assignment from `q`. This API accepts the raw logits exposed by this
/// crate and performs both softmax operations explicitly. Padding is replicated
/// across layers exactly as in the pinned reference. The returned scalar is not
/// multiplied by [`Qwen3MoeConfig::router_aux_loss_coef`].
///
/// # Errors
///
/// Returns a contextual error when there are no sparse-layer tensors, tensor
/// shapes/devices differ, `experts_per_token` is invalid, or the optional mask
/// does not match each layer's batch and sequence axes.
pub fn qwen3_moe_load_balancing_loss<B: Backend>(
    raw_router_logits: &[Tensor<B, 3>],
    experts_per_token: usize,
    token_mask: Option<&TokenMask<B>>,
) -> Result<Tensor<B, 1>> {
    let Some(first) = raw_router_logits.first() else {
        return Err(Error::configuration(
            "qwen3_moe.auxiliary_loss.router_logits",
            "requires at least one sparse-layer router tensor",
        ));
    };
    let [batch, sequence, experts] = first.dims();
    if batch == 0 || sequence == 0 || experts == 0 {
        return Err(Error::shape(
            "qwen3_moe.auxiliary_loss.router_logits[0]",
            "non-empty [batch, sequence, experts]",
            format!("{:?}", first.dims()),
        ));
    }
    if experts_per_token == 0 || experts_per_token > experts {
        return Err(Error::configuration(
            "qwen3_moe.auxiliary_loss.experts_per_token",
            format!("must be in 1..={experts}"),
        ));
    }
    if !first.dtype().is_float() {
        return Err(Error::configuration(
            "qwen3_moe.auxiliary_loss.router_logits[0]",
            format!("must have a floating dtype, found {:?}", first.dtype()),
        ));
    }
    let device = first.device();
    if let Some(mask) = token_mask {
        if mask.as_tensor().dims() != [batch, sequence] {
            return Err(Error::shape(
                "qwen3_moe.auxiliary_loss.token_mask",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", mask.as_tensor().dims()),
            ));
        }
        if mask.as_tensor().device() != device {
            return Err(Error::BackendCapability(
                "Qwen3 MoE auxiliary-loss mask and router logits must reside on the same device"
                    .into(),
            ));
        }
    }
    batch.checked_mul(raw_router_logits.len()).ok_or_else(|| {
        Error::shape(
            "qwen3_moe.auxiliary_loss.router_logits",
            "a layer * batch product representable by usize",
            format!("{} * {batch}", raw_router_logits.len()),
        )
    })?;

    let mut recorded_router_outputs = Vec::with_capacity(raw_router_logits.len());
    for (layer, logits) in raw_router_logits.iter().enumerate() {
        if logits.dims() != [batch, sequence, experts] {
            return Err(Error::shape(
                format!("qwen3_moe.auxiliary_loss.router_logits[{layer}]"),
                format!("[{batch}, {sequence}, {experts}]"),
                format!("{:?}", logits.dims()),
            ));
        }
        if logits.device() != device {
            return Err(Error::BackendCapability(format!(
                "Qwen3 MoE auxiliary-loss router tensor {layer} is on a different device"
            )));
        }
        if !logits.dtype().is_float() {
            return Err(Error::configuration(
                format!("qwen3_moe.auxiliary_loss.router_logits[{layer}]"),
                format!("must have a floating dtype, found {:?}", logits.dtype()),
            ));
        }
        recorded_router_outputs.push(softmax(logits.clone().cast(DType::F32), 2));
    }

    let recorded_router_outputs = Tensor::cat(recorded_router_outputs, 0);
    let second_pass = softmax(recorded_router_outputs.clone(), 2);
    let (_, selected_experts) = deterministic_top_k(second_pass, experts_per_token);
    let stacked_mask = token_mask.map(|mask| {
        TokenMask::new(Tensor::cat(
            (0..raw_router_logits.len())
                .map(|_| mask.as_tensor().clone())
                .collect(),
            0,
        ))
    });
    load_balancing_loss(
        recorded_router_outputs,
        selected_experts,
        stacked_mask.as_ref(),
    )
}

/// Preserves Qwen's raw pre-softmax router projection for diagnostics.
fn router_diagnostics<B: Backend>(output: &SparseMoeOutput<B>) -> RouterDiagnostics<B> {
    RouterDiagnostics {
        logits: Some(output.router_logits.clone()),
        // The pinned objective spans every sparse layer and is returned on the
        // enclosing CausalLmOutput instead of being duplicated per layer.
        load_balancing_loss: None,
        z_loss: None,
    }
}

/// Combines padding with an optional left-bounded causal window.
fn stateless_attention_mask<B: Backend>(
    token_mask: Option<&TokenMask<B>>,
    batch: usize,
    heads: usize,
    sequence: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Result<Option<Tensor<B, 4, Bool>>> {
    let padding = padding_attention_mask(token_mask, token_mask, batch, heads, sequence, sequence)?;
    let Some(window) = window else {
        return Ok(padding);
    };
    let query = Tensor::<B, 1, Int>::arange(0..sequence as i64, device)
        .reshape([1, 1, sequence, 1])
        .expand([batch, heads, sequence, sequence]);
    let key = Tensor::<B, 1, Int>::arange(0..sequence as i64, device)
        .reshape([1, 1, 1, sequence])
        .expand([batch, heads, sequence, sequence]);
    let too_old = (key + window as i64).lower_equal(query);
    Ok(Some(match padding {
        Some(padding) => padding.bool_or(too_old),
        None => too_old,
    }))
}

/// Rejects empty tensor extents before embedding or reduction kernels run.
fn validate_nonempty_input(batch: usize, sequence: usize) -> Result<()> {
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "qwen3_moe.input_ids",
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    Ok(())
}

/// Rejects stateless input tensors that do not share the model parameter device.
///
/// Checking device identity before reductions, mask scans, position math, and
/// embedding lookup keeps backend mixed-device failures inside this crate's
/// recoverable error taxonomy.
fn validate_input_devices<B: Backend>(
    input: &CausalLmInput<B>,
    model_device: &B::Device,
) -> Result<()> {
    ensure_model_device(
        "qwen3_moe.input_ids",
        &input.token_ids.device(),
        model_device,
    )?;
    if let Some(mask) = &input.token_mask {
        ensure_model_device(
            "qwen3_moe.token_mask",
            &mask.as_tensor().device(),
            model_device,
        )?;
    }
    if let Some(positions) = &input.position_ids {
        ensure_model_device("qwen3_moe.position_ids", &positions.device(), model_device)?;
    }
    Ok(())
}

/// Reports one stateless tensor/parameter device mismatch contextually.
fn ensure_model_device<D: PartialEq>(tensor: &str, actual: &D, expected: &D) -> Result<()> {
    if actual != expected {
        return Err(Error::BackendCapability(format!(
            "{tensor} and Qwen3 MoE model parameters are on different devices"
        )));
    }
    Ok(())
}

/// Validates token values before the embedding backend receives indices.
fn validate_token_ids<B: Backend>(token_ids: Tensor<B, 2, Int>, vocabulary: usize) -> Result<()> {
    let minimum = token_ids
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate token IDs: {error}"))
        })?
        .elem::<i64>();
    let maximum = token_ids
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate token IDs: {error}"))
        })?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= vocabulary {
        return Err(Error::shape(
            "qwen3_moe.input_ids",
            format!("token values in 0..{vocabulary}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Validates semantic-mask shape and optionally permits fully masked rows.
fn validate_mask<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
    permit_empty_rows: bool,
) -> Result<()> {
    let Some(mask) = mask else {
        return Ok(());
    };
    if mask.as_tensor().dims() != [batch, sequence] {
        return Err(Error::shape(
            "qwen3_moe.token_mask",
            format!("[{batch}, {sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    if !permit_empty_rows {
        let minimum = mask
            .as_tensor()
            .clone()
            .int()
            .sum_dim(1)
            .min()
            .try_into_scalar()
            .map_err(|error| Error::BackendCapability(format!("failed to validate mask: {error}")))?
            .elem::<i64>();
        if minimum == 0 {
            return Err(Error::shape(
                "qwen3_moe.token_mask",
                "at least one valid token in every prompt row",
                "a fully masked row",
            ));
        }
    }
    Ok(())
}

/// Resolves explicit position IDs or derives padding-independent ordinals.
fn resolve_positions<B: Backend>(
    input: &CausalLmInput<B>,
    batch: usize,
    sequence: usize,
) -> Result<Tensor<B, 2, Int>> {
    if let Some(positions) = &input.position_ids {
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "qwen3_moe.position_ids",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
        return Ok(positions.clone());
    }
    let device = input.token_ids.device();
    match &input.token_mask {
        Some(mask) => {
            let valid = mask.as_tensor().clone();
            Ok((valid.clone().int().cumsum(1) - 1).mask_fill(valid.bool_not(), 0))
        }
        None => Ok(Tensor::<B, 1, Int>::arange(0..sequence as i64, &device)
            .reshape([1, sequence])
            .expand([batch, sequence])),
    }
}

/// Validates position IDs before RoPE constructs angle tables.
fn validate_position_range<B: Backend>(
    positions: Tensor<B, 2, Int>,
    capacity: usize,
) -> Result<()> {
    let minimum = positions
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate positions: {error}"))
        })?
        .elem::<i64>();
    let maximum = positions
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate positions: {error}"))
        })?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= capacity {
        return Err(Error::shape(
            "qwen3_moe.position_ids",
            format!("values in 0..{capacity}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Returns one plus the greatest position used by a stateless batch.
fn evaluated_context<B: Backend>(positions: Tensor<B, 2, Int>) -> Result<usize> {
    let maximum = positions
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to inspect Qwen3 MoE positions: {error}"))
        })?
        .elem::<i64>();
    usize::try_from(maximum)
        .map_err(|_| Error::configuration("qwen3_moe.position_ids", "must be non-negative"))?
        .checked_add(1)
        .ok_or_else(|| Error::configuration("qwen3_moe.position_ids", "context overflows usize"))
}

/// Selects hidden positions before the expensive vocabulary projection.
fn select_hidden<B: Backend>(
    hidden: Tensor<B, 3>,
    mask: Option<&TokenMask<B>>,
    selection: &LogitsSelection,
) -> Result<Tensor<B, 3>> {
    let [batch, sequence, width] = hidden.dims();
    validate_logits_selection(selection, sequence)?;
    match selection {
        LogitsSelection::All => Ok(hidden),
        LogitsSelection::Last => match mask {
            None => Ok(hidden.narrow(1, sequence - 1, 1)),
            Some(mask) => {
                let device = hidden.device();
                let positions = Tensor::<B, 1, Int>::arange(0..sequence as i64, &device)
                    .reshape([1, sequence])
                    .expand([batch, sequence]);
                let last = positions
                    .mask_fill(mask.as_tensor().clone().bool_not(), -1)
                    .argmax(1)
                    .unsqueeze_dim::<3>(2)
                    .expand([batch, 1, width]);
                Ok(hidden.gather(1, last))
            }
        },
        LogitsSelection::Positions(positions) => {
            let indices = positions
                .iter()
                .map(|&value| value as i64)
                .collect::<Vec<_>>();
            let count = indices.len();
            let device = hidden.device();
            Ok(hidden.select(
                1,
                Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [count]), &device),
            ))
        }
    }
}

/// Validates explicit selective-logit indices.
fn validate_logits_selection(selection: &LogitsSelection, sequence: usize) -> Result<()> {
    if let LogitsSelection::Positions(positions) = selection
        && (positions.is_empty() || positions.iter().any(|&position| position >= sequence))
    {
        return Err(Error::shape(
            "forward_options.logits.positions",
            format!("a non-empty list of indices below {sequence}"),
            format!("{positions:?}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        module::Param,
        tensor::{Bool, DType, Int, Tensor, Tolerance, backend::BackendTypes},
    };

    use super::*;
    use crate::{
        CausalLanguageModel, DecodeInput, DecodeState, cache::CacheConfig, checkpoint::TensorSink,
        models::qwen3_moe::Qwen3MoeCheckpointOptions,
    };

    type TestBackend = Flex<f32, i32>;

    /// Creates a small model that exercises routed and dense feed-forward paths.
    fn tiny_config() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 32,
            rope_theta: Some(1_000_000.0),
            max_window_layers: 2,
            decoder_sparse_step: 1,
            moe_intermediate_size: 6,
            num_experts: 4,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            ..Default::default()
        }
    }

    /// Installs deterministic tensors without consuming Burn's process-wide test RNG.
    fn initialized_model<B: Backend>(
        config: Qwen3MoeConfig,
        device: &B::Device,
    ) -> Qwen3MoeForCausalLm<B> {
        let contract = config
            .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
            .expect("tiny contract");
        let mut model = Qwen3MoeForCausalLm::init(config, device).expect("tiny model");
        for (tensor_index, expected) in contract.expected_tensors().iter().enumerate() {
            let shape = expected
                .shape
                .iter()
                .map(|&dimension| usize::try_from(dimension).expect("tiny dimension"))
                .collect::<Vec<_>>();
            let elements = shape.iter().product();
            let values = (0..elements)
                .map(|element| {
                    let code = (element + tensor_index * 7) % 23;
                    (code as f32 - 11.0) * 0.002
                })
                .collect::<Vec<_>>();
            TensorSink::write_tensor(&mut model, &expected.name, TensorData::new(values, shape))
                .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
        }
        model
    }

    /// Builds deterministic tiny weights for parity tests.
    fn tiny_model() -> (
        Qwen3MoeForCausalLm<TestBackend>,
        <TestBackend as BackendTypes>::Device,
    ) {
        let device = Default::default();
        let model = initialized_model(tiny_config(), &device);
        (model, device)
    }

    /// Every Qwen3-MoE norm must round before scale, including per-head Q/K norms.
    #[test]
    fn every_rms_norm_uses_the_upstream_precision_policy() {
        let device = Default::default();
        let model = Qwen3MoeForCausalLm::<TestBackend>::init(tiny_config(), &device).unwrap();

        assert!(model.model.norm.round_before_scale);
        for layer in &model.model.layers {
            assert!(layer.input_layernorm.round_before_scale);
            assert!(layer.post_attention_layernorm.round_before_scale);
            assert!(
                layer
                    .self_attn
                    .query_norm
                    .as_ref()
                    .expect("Qwen3-MoE query RMSNorm")
                    .round_before_scale
            );
            assert!(
                layer
                    .self_attn
                    .key_norm
                    .as_ref()
                    .expect("Qwen3-MoE key RMSNorm")
                    .round_before_scale
            );
        }
    }

    /// Every routed layer selects Qwen's FP32-weight/per-contribution rounding boundary.
    #[test]
    fn every_sparse_layer_uses_the_upstream_weighting_precision() {
        let device = Default::default();
        let model = Qwen3MoeForCausalLm::<TestBackend>::init(tiny_config(), &device).unwrap();

        for layer in &model.model.layers {
            assert_eq!(
                layer
                    .sparse_mlp
                    .as_ref()
                    .expect("tiny topology is all-MoE")
                    .weighting_precision,
                RoutedWeightingPrecision::Fp32ThenResidual
            );
        }
    }

    /// F16 exposes the upstream cast-before-scale order as a bitwise distinction.
    #[test]
    fn rms_norm_f16_matches_upstream_cast_before_scale_order() {
        let device = Default::default();
        let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(tiny_config(), &device).unwrap();
        let norm = &mut model.model.layers[0].input_layernorm;
        let weight = Tensor::<TestBackend, 1>::from_floats(
            [0.731, 1.337, -0.419, 0.613, -1.217, 1.091, 0.527, -0.883],
            &device,
        )
        .cast(DType::F16);
        norm.weight = Param::from_tensor(weight.clone());
        let input = Tensor::<TestBackend, 2>::from_floats(
            [[0.337, 1.719, -2.113, 0.491, -0.827, 1.239, 2.051, -1.477]],
            &device,
        )
        .cast(DType::F16);

        let input_f32 = input.clone().cast(DType::F32);
        let rms = (input_f32.clone().square().mean_dim(1) + norm.epsilon).sqrt();
        let unit_rms = input_f32 / rms;
        let upstream = unit_rms.clone().cast(DType::F16) * weight.clone().unsqueeze();
        let late_round = (unit_rms * weight.cast(DType::F32).unsqueeze()).cast(DType::F16);
        let actual = norm.forward(input);

        assert_eq!(actual.dtype(), DType::F16);
        assert_eq!(actual.to_data(), upstream.to_data());
        assert_ne!(actual.to_data(), late_round.to_data());
    }

    /// The v4.51 boundary is a full-attention prefix length, not a windowed prefix.
    #[test]
    fn sliding_window_schedule_matches_every_valid_boundary() {
        let mut config = tiny_config();
        config.num_hidden_layers = 8;
        config.use_sliding_window = true;
        config.sliding_window = Some(3);

        for boundary in 0..=config.num_hidden_layers {
            config.max_window_layers = boundary;
            for layer_index in 0..config.num_hidden_layers {
                let expected = (layer_index >= boundary).then_some(3);
                assert_eq!(
                    config.layer_sliding_window(layer_index),
                    expected,
                    "boundary={boundary}, layer={layer_index}"
                );
            }
        }
    }

    /// Both upstream feature guards suppress the window independently.
    #[test]
    fn sliding_window_requires_enablement_and_a_configured_width() {
        let mut config = tiny_config();
        config.max_window_layers = 0;
        config.sliding_window = Some(3);
        for layer_index in 0..config.num_hidden_layers {
            assert_eq!(config.layer_sliding_window(layer_index), None);
        }

        config.use_sliding_window = true;
        config.sliding_window = None;
        for layer_index in 0..config.num_hidden_layers {
            assert_eq!(config.layer_sliding_window(layer_index), None);
        }
    }

    /// Stateless output exposes requested hidden and router diagnostics.
    #[test]
    fn stateless_shapes_and_router_options_are_exact() {
        let (model, device) = tiny_model();
        let output = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                        [[1, 2, 3], [4, 5, 6]],
                        &device,
                    ),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    hidden_states: true,
                    router_logits: true,
                    auxiliary_losses: true,
                    ..Default::default()
                },
            )
            .expect("tiny stateless forward");
        assert_eq!(output.logits.dims(), [2, 3, 32]);
        assert_eq!(output.hidden_states.len(), 4);
        assert_eq!(
            output
                .load_balancing_loss
                .as_ref()
                .expect("requested stack-wide loss")
                .dims(),
            [1]
        );
        assert_eq!(output.routers.len(), 2);
        for router in output.routers {
            assert_eq!(router.logits.expect("requested").dims(), [2, 3, 4]);
            assert!(router.load_balancing_loss.is_none());
            assert!(router.z_loss.is_none());
        }
    }

    /// A pinned Transformers 5.2 fixture proves the layer-stack and double-softmax semantics.
    #[test]
    fn stack_wide_auxiliary_loss_matches_transformers_5_2_golden() {
        let device = Default::default();
        let raw_router_logits = vec![
            Tensor::<TestBackend, 3>::from_floats(
                [
                    [[2.0, 0.0, -1.0], [0.5, 0.4, -0.2]],
                    [[-1.0, 1.5, 0.25], [3.0, -2.0, 1.0]],
                ],
                &device,
            ),
            Tensor::<TestBackend, 3>::from_floats(
                [
                    [[0.0, 1.0, 2.0], [-0.5, 0.25, 1.25]],
                    [[1.0, 0.99, -1.0], [0.2, -0.3, 0.7]],
                ],
                &device,
            ),
        ];
        let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, true], [true, false]].into(),
            &device,
        ));

        let loss = qwen3_moe_load_balancing_loss(&raw_router_logits, 2, Some(&mask))
            .expect("valid pinned auxiliary-loss fixture");

        assert_eq!(loss.dtype(), DType::F32);
        loss.into_data().assert_approx_eq::<f32>(
            &TensorData::from([2.009_133_8_f32]),
            Tolerance::absolute(1.0e-6),
        );
    }

    /// Auxiliary loss is global and does not force raw per-layer diagnostics into the output.
    #[test]
    fn auxiliary_loss_only_retains_no_public_router_tensors() {
        let (model, device) = tiny_model();
        let output = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    auxiliary_losses: true,
                    ..Default::default()
                },
            )
            .expect("tiny auxiliary-only forward");

        assert!(output.routers.is_empty());
        let loss = output
            .load_balancing_loss
            .expect("Qwen auxiliary flag requests one global loss")
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            loss > 1.0,
            "the raw objective must not apply the 0.001 coefficient"
        );
    }

    /// Stateful prefill returns the same global loss API and no fabricated per-layer losses.
    #[test]
    fn cached_auxiliary_loss_uses_the_stack_wide_path() {
        let (model, device) = tiny_model();
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        let output = model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions {
                    router_logits: true,
                    auxiliary_losses: true,
                    ..Default::default()
                },
            )
            .expect("tiny stateful auxiliary forward");
        let raw_router_logits = output
            .routers
            .iter()
            .map(|router| {
                assert!(router.load_balancing_loss.is_none());
                router.logits.clone().expect("raw diagnostics requested")
            })
            .collect::<Vec<_>>();
        let expected = qwen3_moe_load_balancing_loss(
            &raw_router_logits,
            model.config().num_experts_per_tok,
            None,
        )
        .unwrap();

        output
            .load_balancing_loss
            .expect("stateful path returns one global objective")
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(0.0));
    }

    /// The public loss helper rejects missing layers and invalid top-k before tensor kernels.
    #[test]
    fn auxiliary_loss_validates_family_inputs() {
        let device = Default::default();
        let no_layers = qwen3_moe_load_balancing_loss::<TestBackend>(&[], 1, None)
            .expect_err("an empty sparse stack has no balance objective");
        assert!(no_layers.to_string().contains("at least one sparse-layer"));

        let logits = [Tensor::<TestBackend, 3>::zeros([1, 2, 3], &device)];
        let invalid_top_k = qwen3_moe_load_balancing_loss(&logits, 4, None)
            .expect_err("top-k cannot exceed the expert count");
        assert!(invalid_top_k.to_string().contains("experts_per_token"));

        let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true]].into(),
            &device,
        ));
        let invalid_mask = qwen3_moe_load_balancing_loss(&logits, 1, Some(&mask))
            .expect_err("mask axes must match every router layer");
        assert!(invalid_mask.to_string().contains("token_mask"));
    }

    /// Stateless device preflight names every independently supplied tensor.
    #[test]
    fn stateless_device_affinity_errors_are_contextual() {
        assert!(ensure_model_device("qwen3_moe.input_ids", &0_u8, &0_u8).is_ok());
        for tensor in [
            "qwen3_moe.input_ids",
            "qwen3_moe.token_mask",
            "qwen3_moe.position_ids",
            "qwen3_moe.lm_head",
        ] {
            let error = ensure_model_device(tensor, &1_u8, &0_u8)
                .expect_err("different device identities must fail preflight");
            assert!(error.to_string().contains(tensor));
            assert!(error.to_string().contains("model parameters"));
        }
    }

    /// Dense override layers instantiate ordinary SwiGLU instead of an expert bank.
    #[test]
    fn dense_override_uses_exactly_one_feed_forward_kind() {
        let device = Default::default();
        let mut config = tiny_config();
        config.mlp_only_layers = vec![1];
        let model = Qwen3MoeForCausalLm::<TestBackend>::init(config, &device).unwrap();
        assert!(model.model.layers[0].sparse_mlp.is_some());
        assert!(model.model.layers[0].dense_mlp.is_none());
        assert!(model.model.layers[1].sparse_mlp.is_none());
        assert!(model.model.layers[1].dense_mlp.is_some());
    }

    /// Serialized router-output policy is honored in addition to call-site options.
    #[test]
    fn config_can_enable_router_logits_by_default() {
        let device = Default::default();
        let mut config = tiny_config();
        config.output_router_logits = true;
        let model = initialized_model::<TestBackend>(config, &device);
        let output = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap();
        assert_eq!(output.routers.len(), 2);
        assert!(output.routers.iter().all(|router| router.logits.is_some()));
    }

    /// One-shot and incremental native-KV execution agree at the final token.
    #[test]
    fn stateless_prefill_decode_parity() {
        let (model, device) = tiny_model();
        let full = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .expect("full forward")
            .logits;
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect("prefill");
        let cached = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect("decode")
            .logits;
        cached
            .into_data()
            .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
        assert_eq!(state.lengths(), &[4]);
    }

    /// Multi-token decode retains causal masking within the appended chunk.
    #[test]
    fn chunked_decode_matches_full_forward() {
        let (model, device) = tiny_model();
        let full = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..Default::default()
                },
            )
            .unwrap()
            .logits
            .slice([0..1, 2..4, 0..32]);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let chunk = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..Default::default()
                },
            )
            .unwrap()
            .logits;
        chunk
            .into_data()
            .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
    }

    /// Finished rows remain inactive while active peers append ragged chunks.
    #[test]
    fn mixed_active_and_inactive_decode_rows_preserve_state() {
        let (model, device) = tiny_model();
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2], [3, 0]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[true, true], [true, false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        assert_eq!(state.lengths(), &[2, 1]);
        let output = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[4, 5], [0, 0]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[true, true], [false, false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    router_logits: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(state.lengths(), &[4, 1]);
        let inactive = output.logits.slice([1..2, 0..2, 0..32]).into_data();
        assert!(
            inactive
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|&x| x == 0.0)
        );
        for router in output.routers {
            let inactive = router.logits.unwrap().slice([1..2, 0..2, 0..4]).into_data();
            assert!(
                inactive
                    .as_slice::<f32>()
                    .unwrap()
                    .iter()
                    .all(|&x| x == 0.0)
            );
        }
    }

    /// An entirely inactive decode is an exact no-op for cache bytes and lengths.
    #[test]
    fn all_inactive_decode_is_exact_noop() {
        let (model, device) = tiny_model();
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let before = state
            .tensor_state()
            .main()
            .native_kv(0)
            .unwrap()
            .keys(0)
            .unwrap()
            .clone()
            .into_data();
        let output = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[0, 0]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[false, false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(state.lengths(), &[2]);
        assert_eq!(
            before,
            state
                .tensor_state()
                .main()
                .native_kv(0)
                .unwrap()
                .keys(0)
                .unwrap()
                .clone()
                .into_data()
        );
        assert!(
            output
                .logits
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|&x| x == 0.0)
        );
    }

    /// A later-layer recoverable error leaves earlier pending cache writes unpublished.
    #[test]
    fn later_layer_failure_is_cache_atomic() {
        let (mut model, device) = tiny_model();
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        let before = state
            .tensor_state()
            .main()
            .native_kv(0)
            .unwrap()
            .keys(0)
            .unwrap()
            .clone()
            .into_data();
        model.model.layers[1].sparse_mlp = None;
        let error = model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("invalid later layer must fail");
        assert!(error.to_string().contains("feed-forward topology"));
        assert_eq!(state.lengths(), &[0]);
        assert_eq!(
            before,
            state
                .tensor_state()
                .main()
                .native_kv(0)
                .unwrap()
                .keys(0)
                .unwrap()
                .clone()
                .into_data()
        );
    }

    /// Reorder/duplicate, safe truncate, reset, and capacity errors use common lifecycle rules.
    #[test]
    fn decode_state_lifecycle_and_capacity_boundaries() {
        let (model, device) = tiny_model();
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 3).unwrap(), 2, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2], [3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        state.reorder(&[1, 1]).unwrap();
        assert_eq!(state.lengths(), &[2, 2]);
        state.truncate(&[1, 2]).unwrap();
        assert_eq!(state.lengths(), &[1, 2]);
        let error = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[5, 6, 7], [5, 6, 7]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("capacity must be checked before model work");
        assert!(error.to_string().contains("capacity"));
        assert_eq!(state.lengths(), &[1, 2]);
        state.reset().unwrap();
        assert_eq!(state.lengths(), &[0, 0]);
    }

    /// Optional sliding masks preserve full-cache correctness across a chunk boundary.
    #[test]
    fn sliding_window_cached_and_stateless_parity() {
        let device = Default::default();
        let mut config = tiny_config();
        config.use_sliding_window = true;
        config.sliding_window = Some(2);
        config.max_window_layers = 0;
        let model = initialized_model::<TestBackend>(config, &device);
        let full = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let cached = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        cached
            .into_data()
            .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
    }

    /// Stateless graph construction remains differentiable under Burn autodiff.
    #[cfg(feature = "autodiff")]
    #[test]
    fn autodiff_smoke_reaches_embedding_weights() {
        use burn::backend::Autodiff;

        type AdBackend = Autodiff<TestBackend>;
        let device = Default::default();
        // Burn's TensorData installation path creates an autodiff non-leaf;
        // lazy initialization is required here so parameter gradients remain
        // tracked as leaf tensors.
        let model = Qwen3MoeForCausalLm::<AdBackend>::init(tiny_config(), &device).unwrap();
        let loss = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
            .sum();
        let gradients = loss.backward();
        assert!(model.model.embed_tokens.weight.grad(&gradients).is_some());
    }
}
