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
    tensor::{Bool, DType, Int, Tensor, activation::softmax, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::{RouterDiagnostics, TokenMask},
    attention::{GroupedQueryAttention, GroupedQueryAttentionConfig, padding_attention_mask},
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    moe::{
        RoutedWeightingPrecision, SparseMoe, SparseMoeConfig, SparseMoeOutput, deterministic_top_k,
        load_balancing_loss,
    },
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{RopeConfig, apply_rotary, rotary_angles},
};

use super::Qwen3MoeConfig;
use crate::models::shared::{
    InputContract, InputNames, ensure_same_device, select_hidden, validate_input,
};

mod decode;

pub use decode::Qwen3MoeDecodeState;

/// Tensor names reported by Qwen3 MoE input errors.
const INPUT_NAMES: InputNames<'static> = InputNames {
    token_ids: "qwen3_moe.input_ids",
    token_mask: "qwen3_moe.token_mask",
    position_ids: "qwen3_moe.position_ids",
};

/// Owner named when a stateless tensor is on another device than the weights.
const PARAMETERS: &str = "Qwen3 MoE model parameters";

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
        let rope = config.rope_config()?;
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
        let validated = validate_input(
            input,
            &self.embed_tokens.weight.lazy_device(),
            &InputContract::new(INPUT_NAMES, PARAMETERS, self.config.vocab_size)
                .with_position_limit(self.config.max_position_embeddings),
        )?;
        let positions = validated.require_positions(INPUT_NAMES.position_ids)?;
        let evaluated_context = validated.evaluated_context;
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
    /// Final-normalized decoder states shaped `[batch, sequence, hidden]`.
    hidden: Tensor<B, 3>,
    /// Optional embedding, post-block, and final-normalized snapshots.
    hidden_states: Vec<Tensor<B, 3>>,
    /// Public sparse-layer routing diagnostics retained when requested.
    routers: Vec<RouterDiagnostics<B>>,
    /// Sparse-layer logits retained internally for stack-wide balancing loss.
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
                ensure_same_device(
                    "qwen3_moe.lm_head",
                    &head.weight.lazy_device(),
                    &hidden_device,
                    PARAMETERS,
                )?;
                operations::linear(hidden, head)
            }
            None => {
                ensure_same_device(
                    "qwen3_moe.model.embed_tokens",
                    &self.model.embed_tokens.weight.lazy_device(),
                    &hidden_device,
                    PARAMETERS,
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

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
