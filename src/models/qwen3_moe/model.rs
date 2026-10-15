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

use std::ops::Range;

use burn::{
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{Bool, DType, Int, Tensor, activation::softmax, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::{HookedModel, RouterDiagnostics, StagedModel, TokenMask},
    attention::{
        AttentionPrecisionMode, GroupedQueryAttention, GroupedQueryAttentionConfig,
        padding_attention_mask,
    },
    experiment::{AttentionPolicy, ExecutionContext, HiddenState},
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    intervention::{Axis, Hooks, Layer, Point, PointInfo, Stack},
    moe::{
        RoutedExpertPrecision, RoutedWeightingPrecision, SparseMoe, SparseMoeConfig,
        deterministic_top_k, dispatch_routed_experts, load_balancing_loss, router_logits,
    },
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{
        RopeConfig, RotaryAngles, RotaryFrequencies, apply_rotary, rotary_angles_from_frequencies,
    },
};

use super::Qwen3MoeConfig;
use crate::models::shared::{
    DeferredChecks, InputContract, InputNames, ValidatedInput, check_nonempty_mask,
    ensure_same_device, select_hidden, validate_input,
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

/// Pre-softmax router scores of a sparse layer, `[batch, token, expert]`.
///
/// Emitted by [`Qwen3MoeForCausalLm::forward_with_hooks`] for every sparse
/// layer, after the router projection and before routing. Edits change which
/// experts are selected and their mixture weights.
#[must_use]
pub fn router_logits_point(layer: Layer) -> Point<3> {
    Point::custom(
        mlp_point_name(layer, "router_logits"),
        [Axis::Batch, Axis::Token, Axis::Other("expert")],
    )
}

/// FP32 mixture weights of the selected experts, `[batch, token, expert_slot]`.
///
/// Slot `j` weights the `j`-th selected expert (ordered by decreasing router
/// score) after Qwen's top-k renormalization. Edits rescale each expert's
/// contribution without changing which experts run.
#[must_use]
pub fn router_weights_point(layer: Layer) -> Point<3> {
    Point::custom(
        mlp_point_name(layer, "router_weights"),
        [Axis::Batch, Axis::Token, Axis::Other("expert_slot")],
    )
}

/// Names a point beside the layer's built-in MLP points, e.g. `decoder.layers.3.mlp.<suffix>`.
fn mlp_point_name(layer: Layer, suffix: &str) -> String {
    let input = layer.mlp_input().info().name;
    format!("{}.{suffix}", input.trim_end_matches(".input"))
}

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

    /// Evaluates one stateless block, returning the new residual and a sparse layer's router logits.
    ///
    /// `inputs` carries the call's shared rotary tables and exclusion plane
    /// (see [`PositionTables`]). With `hooks` every point of
    /// [`Qwen3MoeModel::activation_points`] for block `index` is emitted;
    /// `policy` overrides the default causal connectivity.
    fn forward_inner(
        &self,
        hidden: Tensor<B, 3>,
        inputs: &BlockInputs<'_, B>,
        index: usize,
        mut hooks: Option<&mut Hooks<'_, B>>,
        policy: Option<&AttentionPolicy<B>>,
    ) -> Result<(Tensor<B, 3>, Option<Tensor<B, 3>>)> {
        let layer = Layer::decoder(index);
        let hidden = match hooks.as_deref_mut() {
            Some(hooks) => hooks.apply(layer.residual_pre(), hidden)?,
            None => hidden,
        };
        let residual = hidden.clone();
        let normalized = self.input_layernorm.forward(hidden);
        let normalized = match hooks.as_deref_mut() {
            Some(hooks) => hooks.apply(layer.attention_input(), normalized)?,
            None => normalized,
        };
        let mut projection = self.self_attn.project_self(normalized)?;
        (projection.query, projection.key) =
            apply_rotary(projection.query, projection.key, inputs.rotary)?;
        let [batch, heads, sequence, _] = projection.query.dims();
        let shape = [batch, heads, sequence, sequence];
        let mut mask = inputs.mask.map(|plane| plane.clone().expand(shape));
        let mut bias = None;
        let mut causal = true;
        if let Some(policy) = policy {
            let device = projection.query.device();
            mask = policy.exclusion_mask(mask, shape, &device)?;
            bias = policy.expanded_bias(shape, &device)?;
            causal = policy.causal;
        }
        let mixed = match hooks.as_deref_mut() {
            Some(hooks) => self.self_attn.attend_with_hooks(
                projection,
                mask,
                bias,
                causal,
                layer.self_attention(),
                hooks,
            )?,
            None => self.self_attn.attend(projection, mask, bias, causal)?,
        };
        let mixed = match hooks.as_deref_mut() {
            Some(hooks) => hooks.apply(layer.attention_output(), mixed)?,
            None => mixed,
        };
        let hidden = residual + mixed;
        let residual = hidden.clone();
        let normalized = self.post_attention_layernorm.forward(hidden);
        let normalized = match hooks.as_deref_mut() {
            Some(hooks) => hooks.apply(layer.mlp_input(), normalized)?,
            None => normalized,
        };
        let (mixed, router_logits) = self.feed_forward(normalized, layer, hooks.as_deref_mut())?;
        let mixed = match hooks.as_deref_mut() {
            Some(hooks) => hooks.apply(layer.mlp_output(), mixed)?,
            None => mixed,
        };
        let hidden = residual + mixed;
        let hidden = match hooks {
            Some(hooks) => hooks.apply(layer.residual_post(), hidden)?,
            None => hidden,
        };
        Ok((hidden, router_logits))
    }

    /// Runs the routed or dense feed-forward branch and returns sparse router logits.
    ///
    /// Without hooks a sparse layer is exactly [`SparseMoe::forward`]. With
    /// hooks the same routing and dispatch run as separate steps so the router
    /// logits and selected weights can be observed and edited in place.
    pub(super) fn feed_forward(
        &self,
        normalized: Tensor<B, 3>,
        layer: Layer,
        hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<(Tensor<B, 3>, Option<Tensor<B, 3>>)> {
        match (&self.sparse_mlp, &self.dense_mlp, hooks) {
            (Some(moe), None, None) => {
                let output = moe.forward(normalized)?;
                Ok((output.hidden_state, Some(output.router_logits)))
            }
            (Some(moe), None, Some(hooks)) => {
                let (output, logits) = routed_with_hooks(moe, normalized, layer, hooks)?;
                Ok((output, Some(logits)))
            }
            (None, Some(mlp), None) => Ok((mlp.forward(normalized)?, None)),
            (None, Some(mlp), Some(hooks)) => {
                Ok((mlp.forward_with_hooks(normalized, layer, hooks)?, None))
            }
            _ => Err(Error::Checkpoint(
                "Qwen3 MoE layer has an invalid feed-forward topology".into(),
            )),
        }
    }

    /// Lists this block's activation points in emission order.
    fn activation_points(&self, layer: Layer) -> Vec<PointInfo> {
        let attention = layer.self_attention();
        let mut points = vec![
            layer.residual_pre().info(),
            layer.attention_input().info(),
            attention.query().info(),
            attention.key().info(),
            attention.value().info(),
        ];
        if self.self_attn.precision_mode == AttentionPrecisionMode::EagerFp32Softmax {
            points.extend([attention.scores().info(), attention.pattern().info()]);
        }
        points.extend([
            attention.head_context().info(),
            layer.attention_output().info(),
            layer.mlp_input().info(),
        ]);
        if self.sparse_mlp.is_some() {
            points.extend([
                router_logits_point(layer).info(),
                router_weights_point(layer).info(),
            ]);
        } else {
            points.extend([
                layer.mlp_pre_activation().info(),
                layer.mlp_activation().info(),
            ]);
        }
        points.extend([layer.mlp_output().info(), layer.residual_post().info()]);
        points
    }
}

/// Routes and dispatches one sparse block in steps, emitting its router points.
///
/// The steps are those of [`SparseMoe::forward`]: router projection, FP32
/// routing, then expert dispatch under the module's weighting precision. With
/// no action at either point the result is bit-identical to it. Qwen3-MoE has
/// no shared experts; a manually assembled module with some is rejected
/// rather than silently changed.
fn routed_with_hooks<B: Backend>(
    moe: &SparseMoe<B>,
    input: Tensor<B, 3>,
    layer: Layer,
    hooks: &mut Hooks<'_, B>,
) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
    if !moe.shared.is_empty() || moe.shared_expert_gate.is_some() {
        return Err(Error::BackendCapability(
            "Qwen3 MoE router hooks support routed experts only; this module has shared experts"
                .into(),
        ));
    }
    let [batch, sequence, hidden] = input.dims();
    if batch == 0 || sequence == 0 || hidden != moe.hidden_size {
        return Err(Error::shape(
            "moe.input",
            format!("[batch>0, sequence>0, {}]", moe.hidden_size),
            format!("[{batch}, {sequence}, {hidden}]"),
        ));
    }
    let tokens = batch * sequence;
    let experts = moe.routed.len();
    let slots = moe.experts_per_token;
    let residual_dtype = input.dtype();
    let flat_input = input.clone().reshape([tokens, hidden]);
    let logits = hooks.apply(
        router_logits_point(layer),
        router_logits(input, &moe.router, moe.routing.projection)?,
    )?;
    let decision = moe.router_config().route(
        logits.clone().reshape([tokens, experts]),
        moe.correction_bias.as_ref().map(Param::val),
    )?;
    let weights = hooks
        .apply(
            router_weights_point(layer),
            decision.weights.reshape([batch, sequence, slots]),
        )?
        .reshape([tokens, slots]);
    let (weights, precision) = match moe.weighting_precision {
        RoutedWeightingPrecision::Residual => (
            weights.cast(residual_dtype),
            RoutedExpertPrecision::uniform(residual_dtype),
        ),
        RoutedWeightingPrecision::Fp32ThenResidual => (
            weights,
            RoutedExpertPrecision::fp32_weights_round_each(residual_dtype),
        ),
    };
    let dispatch = dispatch_routed_experts(
        flat_input,
        weights,
        decision.experts,
        Some(decision.non_finite_tokens),
        experts,
        precision,
        |expert, rows| moe.routed[expert].forward(rows),
    )?;
    Ok((
        dispatch.hidden_state.reshape([batch, sequence, hidden]),
        logits,
    ))
}

/// Exclusion plane of one sliding window: `(window, plane)`, `None` meaning unrestricted.
type WindowPlane<B> = (Option<usize>, Option<Tensor<B, 4, Bool>>);

/// Rotary tables and exclusion planes of one stateless call.
///
/// Every block shares the positions, evaluated context, and token mask, so
/// the rotary table is built once (converted once to the activation dtype)
/// and one `[batch, 1, sequence, sequence]` exclusion plane per distinct
/// sliding window (padding united with the window rule) is built once; each
/// block expands its plane to its head count without a copy.
struct PositionTables<B: Backend> {
    /// Rotary tables `[batch, 1, sequence, rotary]`.
    rotary: RotaryAngles<B>,
    /// Exclusion plane (`true` = excluded) per window; `None` means unrestricted.
    masks: Vec<WindowPlane<B>>,
}

impl<B: Backend> PositionTables<B> {
    /// Builds tables for the windows in `windows` (one entry per block is fine).
    fn new(
        rope: RopeConfig,
        positions: &Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        evaluated_context: usize,
        dtype: DType,
        windows: impl IntoIterator<Item = Option<usize>>,
    ) -> Result<Self> {
        let frequencies = RotaryFrequencies::new(rope, evaluated_context)?;
        let rotary =
            rotary_angles_from_frequencies(&frequencies, positions.clone())?.to_dtype(dtype);
        let [batch, sequence] = positions.dims();
        let device = positions.device();
        let padding = padding_attention_mask(token_mask, token_mask, batch, 1, sequence, sequence)?;
        let mut masks: Vec<WindowPlane<B>> = Vec::new();
        for window in windows {
            if masks.iter().any(|(known, _)| *known == window) {
                continue;
            }
            let mask = match window {
                None => padding.clone(),
                Some(window) => {
                    let too_old = window_plane::<B>(sequence, window, &device)?
                        .expand([batch, 1, sequence, sequence]);
                    Some(match &padding {
                        Some(padding) => padding.clone().bool_or(too_old),
                        None => too_old,
                    })
                }
            };
            masks.push((window, mask));
        }
        Ok(Self { rotary, masks })
    }

    /// Returns the inputs of a block with sliding window `window`.
    fn block(&self, window: Option<usize>) -> BlockInputs<'_, B> {
        BlockInputs {
            rotary: &self.rotary,
            mask: self
                .masks
                .iter()
                .find(|(known, _)| *known == window)
                .and_then(|(_, mask)| mask.as_ref()),
        }
    }
}

/// Position-dependent inputs of one block, borrowed from [`PositionTables`].
struct BlockInputs<'a, B: Backend> {
    /// Rotary tables `[batch, 1, sequence, rotary]`.
    rotary: &'a RotaryAngles<B>,
    /// Exclusion plane `[batch, 1, sequence, sequence]` (padding and window), if any.
    mask: Option<&'a Tensor<B, 4, Bool>>,
}

/// Builds the left-bounded window plane `excluded(q, k) = k + window <= q` as `[1, 1, S, S]`.
///
/// Positions are physical columns, as in the reference stateless graph;
/// causality is applied separately by the attention kernel.
fn window_plane<B: Backend>(
    sequence: usize,
    window: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4, Bool>> {
    let length = i64::try_from(sequence)
        .map_err(|_| Error::configuration("qwen3_moe.sliding_window", "sequence exceeds i64"))?;
    let width = i64::try_from(window)
        .map_err(|_| Error::configuration("qwen3_moe.sliding_window", "window exceeds i64"))?;
    let query = Tensor::<B, 1, Int>::arange(0..length, device).reshape([1, 1, sequence, 1]);
    let key = Tensor::<B, 1, Int>::arange(0..length, device).reshape([1, 1, 1, sequence]);
    Ok((key + width)
        .expand([1, 1, sequence, sequence])
        .lower_equal(query.expand([1, 1, sequence, sequence])))
}

/// Qwen3 token embedding, decoder stack, and final normalization.
///
/// Besides the language model's forward pass, the backbone can run headless
/// ([`Self::forward`] returns final-normalized hidden states for a custom task
/// head) and in explicit stages over a [`HiddenState`]
/// ([`Self::embed`], [`Self::run_layers`], [`Self::normalize`]), with or
/// without activation hooks.
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

/// Headless backbone result.
#[derive(Debug, Clone)]
pub struct Qwen3MoeBackboneOutput<B: Backend> {
    /// Final-normalized decoder states `[batch, sequence, hidden]`.
    pub hidden: Tensor<B, 3>,
    /// Embedding, every block output, and the normalized output, when requested.
    pub hidden_states: Vec<Tensor<B, 3>>,
    /// Raw pre-softmax router scores `[batch, sequence, experts]` of every
    /// sparse layer, in layer order, when requested.
    pub router_logits: Vec<Tensor<B, 3>>,
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

    /// Validates a public input and resolves positions with one readback.
    fn validate_public_input(&self, input: &CausalLmInput<B>) -> Result<ValidatedInput<B>> {
        validate_input(
            input,
            &self.embed_tokens.weight.lazy_device(),
            &InputContract::new(INPUT_NAMES, PARAMETERS, self.config.vocab_size)
                .with_position_limit(self.config.max_position_embeddings),
        )
    }

    /// Evaluates the backbone without projecting into the vocabulary.
    ///
    /// Returns the final-normalized `[batch, sequence, hidden]` states, plus
    /// the hidden-state trace and every sparse layer's raw router logits when
    /// requested.
    pub fn forward(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
        retain_router_logits: bool,
    ) -> Result<Qwen3MoeBackboneOutput<B>> {
        self.forward_inner(input, retain_hidden_states, retain_router_logits, None)
    }

    /// Evaluates the backbone while capturing or replacing selected activations.
    ///
    /// Hooks are validated against [`Self::activation_points`] before the
    /// model runs; replacements affect every later block and stay in the
    /// autodiff graph. Router-logit edits change expert selection.
    pub fn forward_with_hooks(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
        retain_router_logits: bool,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Qwen3MoeBackboneOutput<B>> {
        hooks.run(self.activation_points(), |hooks| {
            self.forward_inner(
                input,
                retain_hidden_states,
                retain_router_logits,
                Some(hooks),
            )
        })
    }

    /// Lists every supported activation point in execution order.
    ///
    /// Each block exposes its residual, attention input/output, Q/K after
    /// rotary embedding, native-head V, and head context (scores and pattern
    /// only under [`AttentionPrecisionMode::EagerFp32Softmax`]), the MLP input
    /// and output, and residual after the block. Sparse blocks add
    /// [`router_logits_point`] and [`router_weights_point`]; dense blocks the
    /// gate pre-activation and activation.
    pub fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Decoder.embedding().info()];
        for (index, layer) in self.layers.iter().enumerate() {
            points.extend(layer.activation_points(Layer::decoder(index)));
        }
        points.push(Stack::Decoder.final_norm().info());
        points
    }

    /// Embeds tokens and resolves positions and padding once, before block zero.
    pub fn embed(&self, input: &CausalLmInput<B>) -> Result<HiddenState<B>> {
        let positions = self
            .validate_public_input(input)?
            .require_positions(INPUT_NAMES.position_ids)?;
        let context = ExecutionContext::new(positions, input.token_mask.clone())?;
        HiddenState::new(self.embed_tokens.forward(input.token_ids.clone()), context)
    }

    /// Embeds tokens while applying only the embedding point's scoped actions.
    pub fn embed_with_hooks(
        &self,
        input: &CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.embedding().info()], |hooks| {
            let mut state = self.embed(input)?;
            state.hidden = hooks.apply(Stack::Decoder.embedding(), state.hidden)?;
            Ok(state)
        })
    }

    /// Runs exactly `range`, retaining positions and padding for a later resume.
    ///
    /// Attention accepts the context's connectivity policies; sliding-window
    /// layers combine them with their window. Router diagnostics are not
    /// returned by staged execution.
    pub fn run_layers(&self, state: HiddenState<B>, range: Range<usize>) -> Result<HiddenState<B>> {
        self.run_layers_inner(state, range, None)
    }

    /// Runs a block range with actions addressed by their original layer indices.
    pub fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        let points = range
            .clone()
            .filter(|&index| index < self.layers.len())
            .flat_map(|index| self.layers[index].activation_points(Layer::decoder(index)))
            .collect();
        hooks.run(points, |hooks| {
            self.run_layers_inner(state, range, Some(hooks))
        })
    }

    /// Applies the final norm exactly once at an explicit boundary.
    pub fn normalize(&self, mut state: HiddenState<B>) -> Result<HiddenState<B>> {
        state.validate(self.config.hidden_size)?;
        if state.normalized || state.next_layer > self.layers.len() {
            return Err(Error::configuration(
                "qwen3_moe.execution.normalize",
                "expected an unnormalized state at a valid block boundary",
            ));
        }
        self.ensure_activation_contract("qwen3_moe.execution.hidden", &state.hidden)?;
        state.hidden = self.norm.forward(state.hidden);
        state.normalized = true;
        Ok(state)
    }

    /// Applies the final norm with a scoped final-norm capture or edit.
    pub fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.final_norm().info()], |hooks| {
            let mut state = self.normalize(state)?;
            state.hidden = hooks.apply(Stack::Decoder.final_norm(), state.hidden)?;
            Ok(state)
        })
    }

    /// Requires a latent on the parameters' device in the embedding dtype.
    fn ensure_activation_contract(&self, name: &str, hidden: &Tensor<B, 3>) -> Result<()> {
        ensure_same_device(
            name,
            &hidden.device(),
            &self.embed_tokens.weight.lazy_device(),
            PARAMETERS,
        )?;
        let expected = self.embed_tokens.weight.val().dtype();
        if hidden.dtype() != expected {
            return Err(Error::BackendCapability(format!(
                "{name} dtype must match Qwen3 MoE parameters: expected {expected:?}, found {:?}",
                hidden.dtype()
            )));
        }
        Ok(())
    }

    /// Validates and executes a contiguous block range on a staged state.
    fn run_layers_inner(
        &self,
        mut state: HiddenState<B>,
        range: Range<usize>,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<HiddenState<B>> {
        state.validate(self.config.hidden_size)?;
        state.validate_range(range.clone(), self.layers.len())?;
        self.ensure_activation_contract("qwen3_moe.execution.hidden", &state.hidden)?;
        if state.context.encoder.is_some()
            || state.context.cross_attention.is_some()
            || !state.context.layer_cross_attention.is_empty()
        {
            return Err(Error::configuration(
                "qwen3_moe.execution.encoder",
                "Qwen3 MoE decoder blocks do not define cross-attention",
            ));
        }
        let [batch, sequence, _] = state.hidden.dims();
        // One readback covers the mask rows and both position bounds.
        let mut checks = DeferredChecks::new();
        check_nonempty_mask(
            &mut checks,
            state.context.token_mask.as_ref(),
            batch,
            sequence,
            INPUT_NAMES.token_mask,
        )?;
        let positions = &state.context.positions;
        checks.values_below(
            positions,
            self.config.max_position_embeddings,
            INPUT_NAMES.position_ids,
            "position",
        );
        checks.values_below(
            positions,
            state.context.evaluated_context,
            INPUT_NAMES.position_ids,
            "position",
        );
        checks.run("qwen3_moe.execution")?;
        for (address, _) in &state.context.layer_attention {
            if !(0..self.layers.len()).any(|index| *address == Layer::decoder(index)) {
                return Err(Error::configuration(
                    "qwen3_moe.execution.layer_attention",
                    "override addresses an unavailable Qwen3 MoE decoder layer",
                ));
            }
        }
        let mut policies = Vec::with_capacity(range.len());
        for index in range.clone() {
            let policy = state.context.attention_for(Layer::decoder(index), true);
            let heads = self.layers[index].self_attn.query_heads;
            policy.validate([batch, heads, sequence, sequence], &state.hidden.device())?;
            if let Some(bias) = &policy.bias
                && bias.dtype() != state.hidden.dtype()
            {
                return Err(Error::BackendCapability(format!(
                    "qwen3_moe.execution.attention.bias dtype must match the activations: expected {:?}, found {:?}",
                    state.hidden.dtype(),
                    bias.dtype()
                )));
            }
            policies.push(policy);
        }
        let tables = PositionTables::new(
            self.rope,
            &state.context.positions,
            state.context.token_mask.as_ref(),
            state.context.evaluated_context,
            state.hidden.dtype(),
            range
                .clone()
                .map(|index| self.config.layer_sliding_window(index)),
        )?;
        for (index, policy) in range.clone().zip(&policies) {
            let inputs = tables.block(self.config.layer_sliding_window(index));
            (state.hidden, _) = self.layers[index].forward_inner(
                state.hidden,
                &inputs,
                index,
                hooks.as_deref_mut(),
                Some(policy),
            )?;
        }
        state.next_layer = range.end;
        Ok(state)
    }

    /// Shared stateless execution with optional hooks and requested diagnostics.
    fn forward_inner(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
        retain_router_logits: bool,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<Qwen3MoeBackboneOutput<B>> {
        let validated = self.validate_public_input(input)?;
        let positions = validated.require_positions(INPUT_NAMES.position_ids)?;
        let mut hidden = self.embed_tokens.forward(input.token_ids.clone());
        if let Some(hooks) = hooks.as_deref_mut() {
            hidden = hooks.apply(Stack::Decoder.embedding(), hidden)?;
        }
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        // One rotary table and one exclusion plane per distinct window serve every block.
        let tables = PositionTables::new(
            self.rope,
            &positions,
            input.token_mask.as_ref(),
            validated.evaluated_context,
            hidden.dtype(),
            (0..self.layers.len()).map(|index| self.config.layer_sliding_window(index)),
        )?;
        let mut router_logits = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let inputs = tables.block(self.config.layer_sliding_window(index));
            let (next, logits) =
                layer.forward_inner(hidden, &inputs, index, hooks.as_deref_mut(), None)?;
            hidden = next;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
            if retain_router_logits && let Some(logits) = logits {
                router_logits.push(logits);
            }
        }
        hidden = self.norm.forward(hidden);
        if let Some(hooks) = hooks {
            hidden = hooks.apply(Stack::Decoder.final_norm(), hidden)?;
        }
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(Qwen3MoeBackboneOutput {
            hidden,
            hidden_states,
            router_logits,
        })
    }
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
        reject_attention_capture(options)?;
        let backbone = self.model.forward(
            &input,
            options.hidden_states,
            self.retains_router_logits(options),
        )?;
        self.finish_forward(input, options, backbone)
    }

    /// Lists the stateless activation points available for experimentation.
    ///
    /// See [`Qwen3MoeModel::activation_points`].
    pub fn activation_points(&self) -> Vec<PointInfo> {
        self.model.activation_points()
    }

    /// Captures or transforms selected tensors during a stateless forward pass.
    ///
    /// Replacements affect the returned logits, diagnostics, and auxiliary
    /// loss. Cached decoding does not invoke hooks. Unknown points fail
    /// before model evaluation.
    pub fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        hooks.run(self.activation_points(), |hooks| {
            reject_attention_capture(options)?;
            let backbone = self.model.forward_inner(
                &input,
                options.hidden_states,
                self.retains_router_logits(options),
                Some(hooks),
            )?;
            self.finish_forward(input, options, backbone)
        })
    }

    /// Whether a call needs raw router logits, for diagnostics or the auxiliary loss.
    fn retains_router_logits(&self, options: &ForwardOptions) -> bool {
        options.router_logits || options.auxiliary_losses || self.config().output_router_logits
    }

    /// Projects the requested positions and assembles diagnostics and the auxiliary loss.
    fn finish_forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        backbone: Qwen3MoeBackboneOutput<B>,
    ) -> Result<CausalLmOutput<B>> {
        let load_balancing_loss = options
            .auxiliary_losses
            .then(|| {
                qwen3_moe_load_balancing_loss(
                    &backbone.router_logits,
                    self.config().num_experts_per_tok,
                    input.token_mask.as_ref(),
                )
            })
            .transpose()?;
        let routers = if options.router_logits || self.config().output_router_logits {
            backbone
                .router_logits
                .into_iter()
                .map(router_diagnostics)
                .collect()
        } else {
            Vec::new()
        };
        let hidden = backbone.hidden;
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
        let logits = self.project_logits(selected)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states: backbone.hidden_states,
            attentions: Vec::new(),
            routers,
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

/// Rejects attention-probability capture, which the fused attention path does not expose.
fn reject_attention_capture(options: &ForwardOptions) -> Result<()> {
    if options.attentions {
        return Err(Error::BackendCapability(
            "Qwen3 MoE attention-probability capture is not yet exposed by Burn's fused path"
                .into(),
        ));
    }
    Ok(())
}

impl<B: Backend> HookedModel<B> for Qwen3MoeForCausalLm<B> {
    /// Complete causal-LM input.
    type Input = CausalLmInput<B>;
    /// Logits and requested diagnostics.
    type Output = CausalLmOutput<B>;

    /// Lists the stateless activation points in emission order.
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        Ok(Qwen3MoeForCausalLm::activation_points(self))
    }

    /// Runs the stateless forward with hooks; see [`Qwen3MoeForCausalLm::forward_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        Qwen3MoeForCausalLm::forward_with_hooks(self, input, options, hooks)
    }
}

impl<B: Backend> StagedModel<B> for Qwen3MoeModel<B> {
    /// Number of decoder blocks.
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Qwen3 MoE blocks emit decoder-stack points.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Embeds tokens; see [`Qwen3MoeModel::embed_with_hooks`].
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        Qwen3MoeModel::embed_with_hooks(self, &input, hooks)
    }

    /// Runs a block range; see [`Qwen3MoeModel::run_layers_with_hooks`].
    fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        Qwen3MoeModel::run_layers_with_hooks(self, state, range, hooks)
    }

    /// Applies the final norm; see [`Qwen3MoeModel::normalize_with_hooks`].
    fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        Qwen3MoeModel::normalize_with_hooks(self, state, hooks)
    }
}

impl<B: Backend> StagedModel<B> for Qwen3MoeForCausalLm<B> {
    /// Number of decoder blocks of the backbone.
    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    /// Qwen3 MoE blocks emit decoder-stack points.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Embeds tokens through the backbone.
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model.embed_with_hooks(&input, hooks)
    }

    /// Runs a backbone block range.
    fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model.run_layers_with_hooks(state, range, hooks)
    }

    /// Applies the backbone's final norm.
    fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model.normalize_with_hooks(state, hooks)
    }
}

impl Qwen3MoeConfig {
    /// Returns the local-attention window of block `layer_index`, when enabled.
    ///
    /// The pinned Transformers 5.2 reference stores
    /// `sliding_window if use_sliding_window else None`, passes that window to
    /// every `Qwen3MoeAttention`, and builds one sliding-window causal mask for
    /// the whole stack, so the window is the same for every block and
    /// `max_window_layers` has no effect. (Transformers 4.51 instead kept
    /// blocks `0..max_window_layers` on full attention; 5.2 dropped that rule
    /// for this family.) Released Qwen3-30B-A3B disables the feature, so every
    /// block of that checkpoint returns `None`.
    #[must_use]
    pub fn layer_sliding_window(&self, _layer_index: usize) -> Option<usize> {
        self.sliding_window.filter(|_| self.use_sliding_window)
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
fn router_diagnostics<B: Backend>(logits: Tensor<B, 3>) -> RouterDiagnostics<B> {
    RouterDiagnostics {
        logits: Some(logits),
        // The pinned objective spans every sparse layer and is returned on the
        // enclosing CausalLmOutput instead of being duplicated per layer.
        load_balancing_loss: None,
        z_loss: None,
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "reference_tests.rs"]
mod reference_tests;
