//! The executable gpt-oss decoder graph.
//!
//! # One block for every path
//!
//! A block (see the [`architecture`](super::GptOssArchitecture) translation
//! table) is
//!
//! ```text
//! a  = Attn(N₁(x))      rotated queries/keys, learned sinks, global or sliding keys
//! x' = x + a
//! y  = x' + MoE(N₂(x'))
//! MoE(u) = Σ_{j ∈ top-K(z)} softmax(z_S)_j · E_j(u),   z = u Wᵣᵀ + bᵣ
//! E_j(u) = down_j((clamp(up_j u, ±7) + 1) · g · σ(1.702 g)),   g = min(gate_j u, 7)
//! ```
//!
//! [`GptOssDecoderLayer::forward`] is the only implementation of that
//! equation. The stateless pass, the hooked pass, staged execution, and the
//! batched cached step differ only in where attention finds its keys, which
//! the private `LayerKeys` trait abstracts: the `sequence` submodule attends
//! over the input's own tokens with causal/sliding/padding masks, and the
//! `decode` submodule attends over full-storage and sliding-ring KV caches.
//!
//! # Shared components
//!
//! Attention is the crate's [`GroupedQueryAttention`] with learned sinks
//! (native KV heads, fused kernels with the sink folded in as an extra key,
//! or the eager F32-softmax graph); masks come from
//! [`AttentionConnectivity`]; RoPE from the shared translator and
//! [`RotaryFrequencies`]; routing from [`RouterConfig`]
//! (top-k-then-softmax); dispatch from the crate's memory-bounded
//! `dispatch_routed_experts`; and each expert is a
//! [`GatedFeedForward`] evaluated with [`ClampedSwiGlu`].
//!
//! # Tensor axes and precision
//!
//! Tokens and positions are `[batch, token]`, residuals `[batch, token, d]`,
//! queries `[batch, H, token, h]`, keys and values keep their native
//! `[batch, K, token, h]` heads, router logits are `[batch, token, E]`, and
//! logits `[batch, selected, V]`. RMSNorm reduces and scales in F32 and
//! rounds once. Rotary tables are built once per call in F32 and rounded to
//! the activation dtype, as the reference rounds `cos`/`sin` to `x.dtype`.
//! Router logits are projected in the activation dtype and ranked and
//! normalized in F32; the selected weights are rounded to the activation
//! dtype before they scale expert outputs, as the reference's weights are.
//!
//! # Padding
//!
//! Padding never consumes a position: implicit positions are valid-token
//! ordinals, and a padded key is excluded from every query. A padded query
//! still attends to the valid keys up to its ordinal, and the sinks give
//! every row probability mass, so padded outputs stay finite; they are
//! unspecified, and hooked passes zero them at every activation point.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{DType, Int, Tensor, activation::softmax, backend::Backend},
};

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result, TokenMask,
    architecture::{Activation, RouterDiagnostics},
    attention::{
        AttentionConnectivity, AttentionPrecisionMode, AttentionProjection, GroupedQueryAttention,
    },
    feed_forward::{ClampedSwiGlu, GatedFeedForward},
    intervention::{AttentionPoints, Hooks, Layer, PointInfo, Stack},
    models::qwen3_moe::{router_logits_point, router_weights_point},
    moe::{
        RoutedExpertPrecision, RouterConfig, deterministic_top_k, dispatch_routed_experts,
        load_balancing_loss, router_logits,
    },
    normalization::RmsNorm,
    operations,
    position::{RotaryAngles, RotaryFrequencies, apply_rotary, rotary_angles_from_frequencies},
};

use super::architecture::GptOssArchitecture;
use super::config::GptOssConfig;
use crate::models::shared::{
    InputContract, InputNames, ensure_same_device, select_hidden, validate_input,
};

mod decode;
mod experiment;
mod sequence;
#[cfg(test)]
pub(super) mod test_support;

pub use decode::GptOssDecodeState;
use sequence::SequenceKeys;

/// Tensor names reported by input errors.
pub(super) const INPUT_NAMES: InputNames<'static> = InputNames {
    token_ids: "gpt_oss.input_ids",
    token_mask: "gpt_oss.token_mask",
    position_ids: "gpt_oss.position_ids",
};

/// Owner named when an input lives on another device than the parameters.
const PARAMETERS: &str = "gpt-oss model parameters";

// The shared block.

/// Where one attention call finds its keys and values.
///
/// Implementations own the masks and rotary tables of one call; the block
/// projects and rotates, then hands the projection over.
pub(super) trait LayerKeys<B: Backend> {
    /// Rotary tables of this call, in the activation dtype.
    fn angles(&self) -> &RotaryAngles<B>;

    /// Attends the rotated `projection` for block `layer`.
    ///
    /// Returns the output-projected `[batch, token, d]` attention output.
    fn attend(
        &mut self,
        layer: usize,
        attention: &GroupedQueryAttention<B>,
        projection: AttentionProjection<B>,
        points: AttentionPoints,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>>;
}

/// The routed feed-forward network of one block: a biased router and `E` experts.
///
/// Expert `e` keeps the reference's input-major weights in Burn's row layout:
/// `gate_proj` and `up_proj` are `[d, f]` (the even and odd columns of the
/// checkpoint's `gate_up_proj[e]`) and `down_proj` is `[f, d]`, each with a
/// bias. The experts' `activation` field is unused: [`ClampedSwiGlu`]
/// defines their complete nonlinearity.
#[derive(Module, Debug)]
pub struct GptOssSparseMlp<B: Backend> {
    /// Router projection `[E, d]` (column layout) with bias `[E]`.
    pub router: Linear<B>,
    /// Routed experts in expert-ID order.
    pub experts: Vec<GatedFeedForward<B>>,
    /// Top-k-then-softmax routing policy.
    #[module(skip)]
    pub routing: RouterConfig,
    /// The experts' clamped gated unit.
    #[module(skip)]
    pub glu: ClampedSwiGlu,
}

impl<B: Backend> GptOssSparseMlp<B> {
    /// Allocates the router and experts of a validated architecture.
    fn init(architecture: &GptOssArchitecture, device: &B::Device) -> Result<Self> {
        let (hidden, width) = (architecture.hidden_size, architecture.intermediate_size);
        let expert_projection = |input, output| {
            operations::init_linear(
                &LinearConfig::new(input, output)
                    .with_bias(true)
                    .with_layout(LinearLayout::Row),
                device,
            )
        };
        let experts = (0..architecture.experts)
            .map(|_| GatedFeedForward {
                gate_proj: expert_projection(hidden, width),
                up_proj: expert_projection(hidden, width),
                down_proj: expert_projection(width, hidden),
                activation: Activation::Silu,
            })
            .collect();
        Ok(Self {
            router: operations::init_linear(
                &LinearConfig::new(hidden, architecture.experts)
                    .with_bias(true)
                    .with_layout(LinearLayout::Col),
                device,
            ),
            experts,
            routing: architecture.router_config(),
            glu: architecture.glu,
        })
    }

    /// Routes and evaluates `[batch, token, d]` states, emitting the router points of `layer`.
    ///
    /// Returns the `[batch, token, d]` mixture and the raw `[batch, token, E]`
    /// router logits (after any edit at `mlp.router_logits`). Expert outputs
    /// are weighted and accumulated in the activation dtype, in ascending
    /// expert order, as the reference's `index_add_` loop does. With empty
    /// hooks every emission is a no-op.
    ///
    /// # Errors
    ///
    /// Returns routing errors (including a non-finite router score, reported
    /// with its token) and expert projection errors.
    pub fn forward_with_hooks(
        &self,
        input: Tensor<B, 3>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
        let [batch, tokens, hidden] = input.dims();
        let rows = batch * tokens;
        let dtype = input.dtype();
        let flat = input.clone().reshape([rows, hidden]);
        let logits = hooks.apply(
            router_logits_point(layer),
            router_logits(input, &self.router, self.routing.policy.projection)?,
        )?;
        let experts = self.routing.experts;
        let decision = self
            .routing
            .route(logits.clone().reshape([rows, experts]), None)?;
        let slots = self.routing.experts_per_token;
        let weights = hooks
            .apply(
                router_weights_point(layer),
                decision.weights.reshape([batch, tokens, slots]),
            )?
            .reshape([rows, slots])
            .cast(dtype);
        let dispatch = dispatch_routed_experts(
            flat,
            weights,
            decision.experts,
            Some(decision.non_finite_tokens),
            experts,
            RoutedExpertPrecision::uniform(dtype),
            |expert, rows| self.experts[expert].forward_clamped(rows, &self.glu),
        )?;
        Ok((
            dispatch.hidden_state.reshape([batch, tokens, hidden]),
            logits,
        ))
    }
}

/// One gpt-oss decoder block.
#[derive(Module, Debug)]
pub struct GptOssDecoderLayer<B: Backend> {
    /// Grouped-query self-attention with biases and learned sinks.
    pub self_attn: GroupedQueryAttention<B>,
    /// Routed clamped-SwiGLU experts.
    pub mlp: GptOssSparseMlp<B>,
    /// RMSNorm before attention.
    pub input_layernorm: RmsNorm<B>,
    /// RMSNorm before the routed experts.
    pub post_attention_layernorm: RmsNorm<B>,
    /// Which earlier keys a query may see, on top of causality and padding.
    #[module(skip)]
    pub connectivity: AttentionConnectivity,
}

impl<B: Backend> GptOssDecoderLayer<B> {
    /// Allocates block `index` of a validated architecture.
    fn init(architecture: &GptOssArchitecture, index: usize, device: &B::Device) -> Result<Self> {
        Ok(Self {
            self_attn: architecture.attention_config().init(device)?,
            mlp: GptOssSparseMlp::init(architecture, device)?,
            input_layernorm: architecture
                .rms_norm(architecture.hidden_size)
                .init(device)?,
            post_attention_layernorm: architecture
                .rms_norm(architecture.hidden_size)
                .init(device)?,
            connectivity: architecture.layers[index],
        })
    }

    /// Runs block `index` on `[batch, token, d]` states, emitting its hook points.
    ///
    /// Returns the block output and the block's raw router logits. Points
    /// are emitted in [`Self::activation_points`] order; with empty hooks
    /// every emission is a no-op.
    pub(super) fn forward(
        &self,
        hidden: Tensor<B, 3>,
        index: usize,
        keys: &mut impl LayerKeys<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
        let layer = Layer::decoder(index);
        let residual = hooks.apply(layer.residual_pre(), hidden)?;
        let input = hooks.apply(
            layer.attention_input(),
            self.input_layernorm.forward(residual.clone()),
        )?;
        let mut projection = self.self_attn.project_self(input)?;
        (projection.query, projection.key) =
            apply_rotary(projection.query, projection.key, keys.angles())?;
        let attention = keys.attend(
            index,
            &self.self_attn,
            projection,
            layer.self_attention(),
            hooks,
        )?;
        let attention = hooks.apply(layer.attention_output(), attention)?;
        let hidden = residual + attention;
        let input = hooks.apply(
            layer.mlp_input(),
            self.post_attention_layernorm.forward(hidden.clone()),
        )?;
        let (mlp, router_logits) = self.mlp.forward_with_hooks(input, layer, hooks)?;
        let mlp = hooks.apply(layer.mlp_output(), mlp)?;
        Ok((
            hooks.apply(layer.residual_post(), hidden + mlp)?,
            router_logits,
        ))
    }

    /// Lists the points this block emits as block `index`, in emission order.
    ///
    /// Score and pattern points exist only under
    /// [`AttentionPrecisionMode::EagerFp32Softmax`]; with sinks the pattern
    /// rows sum to `1 - p_sink`.
    #[must_use]
    pub fn activation_points(&self, index: usize) -> Vec<PointInfo> {
        let layer = Layer::decoder(index);
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
            router_logits_point(layer).info(),
            router_weights_point(layer).info(),
            layer.mlp_output().info(),
            layer.residual_post().info(),
        ]);
        points
    }
}

// The backbone.

/// Token embedding, decoder blocks, and final RMSNorm.
#[derive(Module, Debug)]
pub struct GptOssModel<B: Backend> {
    /// Token embedding table `[V, d]`.
    pub embed_tokens: Embedding<B>,
    /// Decoder blocks in order.
    pub layers: Vec<GptOssDecoderLayer<B>>,
    /// Final RMSNorm.
    pub norm: RmsNorm<B>,
    /// Checkpoint configuration, excluded from parameter records.
    #[module(skip)]
    pub config: GptOssConfig,
    /// Validated executable topology derived from `config`.
    #[module(skip)]
    pub architecture: GptOssArchitecture,
}

/// Final states and requested diagnostics of one backbone pass.
pub(super) struct Backbone<B: Backend> {
    /// Final-normalized states `[batch, token, d]`.
    pub(super) hidden: Tensor<B, 3>,
    /// Embedding, every block output, and the final states, when requested.
    pub(super) hidden_states: Vec<Tensor<B, 3>>,
    /// Raw `[batch, token, E]` router logits of every block, when requested.
    pub(super) router_logits: Vec<Tensor<B, 3>>,
}

impl<B: Backend> GptOssModel<B> {
    /// Allocates a randomly initialized backbone for a validated configuration.
    fn init(config: GptOssConfig, device: &B::Device) -> Result<Self> {
        let architecture = config.architecture()?;
        let layers = (0..architecture.num_layers())
            .map(|index| GptOssDecoderLayer::init(&architecture, index, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(architecture.vocab_size, architecture.hidden_size)
                .init(device),
            layers,
            norm: architecture
                .rms_norm(architecture.hidden_size)
                .init(device)?,
            config,
            architecture,
        })
    }

    /// Returns the stateless input contract: valid-token ordinal positions below the trained context.
    fn input_contract(&self) -> InputContract<'static> {
        InputContract::new(INPUT_NAMES, PARAMETERS, self.architecture.vocab_size)
            .with_position_limit(self.architecture.max_position_embeddings)
    }

    /// Returns the activation dtype, which is the embedding table's dtype.
    pub(super) fn activation_dtype(&self) -> DType {
        self.embed_tokens.weight.val().dtype()
    }

    /// Builds the rotary tables of one call in the activation dtype.
    ///
    /// Frequencies are evaluated once on the host for `evaluated_context`,
    /// angles once on the device in F32, and the tables rounded once to the
    /// activation dtype; every block then shares them.
    pub(super) fn rotary_tables(
        &self,
        positions: Tensor<B, 2, Int>,
        evaluated_context: usize,
    ) -> Result<RotaryAngles<B>> {
        let frequencies = RotaryFrequencies::new(self.architecture.rope, evaluated_context)?;
        Ok(rotary_angles_from_frequencies(&frequencies, positions)?
            .to_dtype(self.activation_dtype()))
    }

    /// Runs embedding, every block, and the final norm inside the caller's hook scope.
    fn forward(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
        retain_router_logits: bool,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Backbone<B>> {
        let device = self.embed_tokens.weight.lazy_device();
        let validated = validate_input(input, &device, &self.input_contract())?;
        let positions = validated.require_positions(INPUT_NAMES.position_ids)?;
        let angles = self.rotary_tables(positions, validated.evaluated_context)?;
        // A mask proven all-valid is dropped: global layers then keep the
        // fused causal path and build no plane.
        let padding = input.token_mask.as_ref().filter(|_| !validated.fully_valid);
        let mut keys = SequenceKeys::new(self, angles, padding, input.token_ids.dims())?;
        let mut hidden = hooks.apply(
            Stack::Decoder.embedding(),
            self.embed_tokens.forward(input.token_ids.clone()),
        )?;
        let mut hidden_states = Vec::new();
        let mut router_logits = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
            let (output, logits) = layer.forward(hidden, index, &mut keys, hooks)?;
            hidden = output;
            if retain_router_logits {
                router_logits.push(logits);
            }
        }
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        let hidden = hooks.apply(Stack::Decoder.final_norm(), self.norm.forward(hidden))?;
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(Backbone {
            hidden,
            hidden_states,
            router_logits,
        })
    }

    /// Lists every point of a complete pass, in emission order.
    fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Decoder.embedding().info()];
        for (index, layer) in self.layers.iter().enumerate() {
            points.extend(layer.activation_points(index));
        }
        points.push(Stack::Decoder.final_norm().info());
        points
    }
}

// The causal language model.

/// An OpenAI gpt-oss decoder-only mixture-of-experts language model.
#[derive(Module, Debug)]
pub struct GptOssForCausalLm<B: Backend> {
    /// Embedding, blocks, and final norm.
    pub model: GptOssModel<B>,
    /// Independent vocabulary projection `[V, d]`; `None` when tied to the embedding.
    pub lm_head: Option<Linear<B>>,
}

impl<B: Backend> GptOssForCausalLm<B> {
    /// Validates `config` and allocates a randomly initialized model.
    ///
    /// Load weights with [`Self::load_checkpoint`] or the automatic registry.
    ///
    /// # Errors
    ///
    /// Returns the configuration's validation error or a parameter
    /// allocation error.
    pub fn init(config: GptOssConfig, device: &B::Device) -> Result<Self> {
        let model = GptOssModel::init(config, device)?;
        let architecture = &model.architecture;
        let lm_head = (!architecture.tie_word_embeddings).then(|| {
            operations::init_linear(
                &LinearConfig::new(architecture.hidden_size, architecture.vocab_size)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col),
                device,
            )
        });
        Ok(Self { model, lm_head })
    }

    /// Returns the checkpoint configuration.
    #[must_use]
    pub const fn config(&self) -> &GptOssConfig {
        &self.model.config
    }

    /// Returns the validated executable topology.
    #[must_use]
    pub const fn architecture(&self) -> &GptOssArchitecture {
        &self.model.architecture
    }

    /// Selects the attention operation graph of every layer.
    ///
    /// [`AttentionPrecisionMode::EagerFp32Softmax`] materializes scores, runs
    /// softmax (with the sink column) in F32 as the eager reference does, and
    /// exposes the score and pattern hook points and
    /// [`ForwardOptions::attentions`]; the default backend graph uses fused
    /// kernels with the sink folded in as an extra key.
    #[must_use]
    pub fn with_attention_precision(mut self, mode: AttentionPrecisionMode) -> Self {
        for layer in &mut self.model.layers {
            layer.self_attn.precision_mode = mode;
        }
        self
    }

    /// Evaluates the stateless, differentiable forward pass.
    ///
    /// Implicit positions are valid-token ordinals, so left padding does not
    /// shift rotary phases. `options.router_logits` (or the configuration's
    /// `output_router_logits`) returns every block's raw router logits;
    /// `options.auxiliary_losses` returns the reference's stack-wide
    /// load-balancing objective, without its training coefficient.
    /// `options.attentions` requires
    /// [`AttentionPrecisionMode::EagerFp32Softmax`].
    ///
    /// # Errors
    ///
    /// Returns input validation errors (see `models::shared`), routing
    /// errors, or a capability error for attention capture under fused
    /// kernels.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return self.forward_capturing_attentions(input, options);
        }
        let backbone = self.model.forward(
            &input,
            options.hidden_states,
            self.retains_router_logits(options),
            &mut Hooks::new(),
        )?;
        self.finish(input, options, backbone)
    }

    /// Returns whether a call needs every block's router logits.
    fn retains_router_logits(&self, options: &ForwardOptions) -> bool {
        options.router_logits || options.auxiliary_losses || self.config().output_router_logits
    }

    /// Projects the requested positions and assembles the output.
    fn finish(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        backbone: Backbone<B>,
    ) -> Result<CausalLmOutput<B>> {
        let selected = select_hidden(
            backbone.hidden.clone(),
            input.token_mask.as_ref(),
            &options.logits,
        )?;
        let logits = self.project_logits(selected)?;
        let final_hidden_state = options.hidden_states.then_some(backbone.hidden);
        self.assemble(
            logits,
            final_hidden_state,
            backbone.hidden_states,
            backbone.router_logits,
            input.token_mask.as_ref(),
            options,
        )
    }

    /// Builds the output with router diagnostics and the auxiliary loss as requested.
    ///
    /// `router_logits` are rectangular `[batch, token, E]` tensors aligned
    /// with `token_mask`, which excludes padding from the loss.
    pub(super) fn assemble(
        &self,
        logits: Tensor<B, 3>,
        final_hidden_state: Option<Tensor<B, 3>>,
        hidden_states: Vec<Tensor<B, 3>>,
        router_logits: Vec<Tensor<B, 3>>,
        token_mask: Option<&TokenMask<B>>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        let load_balancing_loss = options
            .auxiliary_losses
            .then(|| {
                gpt_oss_load_balancing_loss(
                    &router_logits,
                    self.architecture().experts_per_token,
                    token_mask,
                )
            })
            .transpose()?;
        let routers = if options.router_logits || self.config().output_router_logits {
            router_logits
                .into_iter()
                .map(|logits| RouterDiagnostics {
                    logits: Some(logits),
                    // The reference objective spans every block; it is returned once below.
                    load_balancing_loss: None,
                    z_loss: None,
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(CausalLmOutput {
            logits,
            final_hidden_state,
            hidden_states,
            attentions: Vec::new(),
            routers,
            load_balancing_loss,
        })
    }

    /// Applies the independent or tied vocabulary projection to `[batch, token, d]`.
    pub(super) fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let device = hidden.device();
        match &self.lm_head {
            Some(head) => {
                ensure_same_device(
                    "gpt_oss.lm_head",
                    &head.weight.lazy_device(),
                    &device,
                    PARAMETERS,
                )?;
                operations::linear(hidden, head)
            }
            None => {
                let table = &self.model.embed_tokens.weight;
                ensure_same_device(
                    "gpt_oss.model.embed_tokens",
                    &table.lazy_device(),
                    &device,
                    PARAMETERS,
                )?;
                Ok(burn::tensor::module::linear(
                    hidden,
                    table.val().transpose(),
                    None,
                ))
            }
        }
    }
}

/// Computes the reference's load-balancing objective over every block.
///
/// Transformers' `load_balancing_loss_func` concatenates the raw router
/// logits of all blocks, applies one softmax `p = softmax(z)`, re-selects the
/// top `K` experts of `p`, and returns
/// `E · Σ_{j,e} mean_t(a[t,j,e]) · mean_t(p[t,e])` with the token mask
/// excluding padding from both means (the shared
/// [`load_balancing_loss`]). The training coefficient
/// `router_aux_loss_coef` is left to the caller.
///
/// # Errors
///
/// Returns a configuration error without router logits, and the shape or
/// device errors of [`load_balancing_loss`].
pub fn gpt_oss_load_balancing_loss<B: Backend>(
    router_logits: &[Tensor<B, 3>],
    experts_per_token: usize,
    token_mask: Option<&TokenMask<B>>,
) -> Result<Tensor<B, 1>> {
    if router_logits.is_empty() {
        return Err(Error::configuration(
            "gpt_oss.auxiliary_loss.router_logits",
            "requires at least one block's router logits",
        ));
    }
    let [_, _, experts] = router_logits[0].dims();
    if experts_per_token == 0 || experts_per_token > experts {
        return Err(Error::configuration(
            "gpt_oss.auxiliary_loss.experts_per_token",
            format!("must be in 1..={experts}"),
        ));
    }
    let stacked = Tensor::cat(
        router_logits
            .iter()
            .map(|logits| logits.clone().cast(DType::F32))
            .collect(),
        0,
    );
    let (_, selected) = deterministic_top_k(softmax(stacked.clone(), 2), experts_per_token);
    let stacked_mask = token_mask.map(|mask| {
        TokenMask::new(Tensor::cat(
            vec![mask.as_tensor().clone(); router_logits.len()],
            0,
        ))
    });
    load_balancing_loss(stacked, selected, stacked_mask.as_ref())
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
