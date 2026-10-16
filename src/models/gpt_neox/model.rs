//! Stateless GPT-NeoX execution using shared transformer mathematics.
//!
//! Residuals use `[batch, token, hidden]`; attention projections use
//! `[batch, head, token, head_channel]`. Each block uses two affine LayerNorms,
//! scaled dot-product attention, and a dense exact-GELU MLP. In parallel mode
//! `x_next = MLP(LN_m(x)) + Attention(LN_a(x)) + x`, preserving source addition
//! order; sequential mode feeds the attention-updated residual into `LN_m`.
//!
//! RoPE rotates an even prefix of each actual head width using physical positions.
//! The cosine and sine tables are built once per forward for every distinct
//! head width (one table for an unedited model) and shared by its layers.
//! Padding removes keys but retains query computations; empty-valid-token rows are
//! rejected. Causal and bidirectional entry points share this padding contract.
//! Parameter and residual dtype/device must agree. Source dropout and decode
//! caches are absent from this explicit graph. The untied vocabulary projection
//! follows final normalization and can be called independently for logit-lens or
//! trainable tuned-lens experiments.
//!
//! # Attention precision
//!
//! Attention defaults to [`AttentionPrecisionMode::EagerFp32Softmax`]: the
//! score plane is materialized, softmax runs in F32, and `scores`/`pattern`
//! hook points exist. [`GptNeoXForCausalLm::set_attention_precision`] selects
//! [`AttentionPrecisionMode::BackendDefault`], the backend's fused kernel,
//! which is faster but has no score or pattern points.
//!
//! # Hidden states
//!
//! [`ForwardOptions::hidden_states`] returns the crate-wide `L + 2` tensors:
//! the embedding, every block's output, and the final normalized state.
//! Hugging Face returns `L + 1` (the embedding, the outputs of blocks
//! `0..L-1`, and the final normalized state), so crate index `i` equals HF
//! index `i` for `i < L`, crate index `L` is block `L - 1`'s raw output (the
//! residual a final-layer lens needs), and crate index `L + 1` equals HF
//! index `L`.
//!
//! # Diagnostics
//!
//! Requests in [`ForwardOptions`] behave as in every other dense family, so a
//! family-generic experiment runs unchanged on Pythia:
//!
//! - `attentions` is rejected before any work with [`Error::BackendCapability`],
//!   exactly as GPT-2 rejects it: probabilities are not returned in
//!   [`CausalLmOutput::attentions`] but are captured through the
//!   `self_attention().pattern()` hook points of the default eager precision
//!   mode;
//! - `router_logits` and `auxiliary_losses` succeed with empty
//!   [`CausalLmOutput::routers`] and no load-balancing loss, because every
//!   block is dense and there is nothing to route.

use burn::{
    module::{Initializer, Module},
    nn::{
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, LinearLayout,
    },
    tensor::{DType, Int, Tensor, backend::Backend},
};

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::{Activation, TokenMask},
    attention::{
        AttentionOutputGate, AttentionPrecisionMode, GroupedQueryAttention,
        GroupedQueryAttentionConfig, padding_attention_mask,
    },
    feed_forward::{DenseFeedForward, DenseFeedForwardConfig},
    intervention::{Hooks, Layer, PointInfo, Stack},
    operations,
    position::{RopeConfig, RopeScaling, RotaryAngles, apply_rotary, rotary_angles},
};

mod experiment;

use super::GptNeoXConfig;
use super::widths::{LayerWidthError, layer_widths};
use crate::models::shared::{
    ImplicitPositions, InputContract, InputNames, ValidatedInput, select_hidden, validate_input,
    validate_nonempty_mask,
};

/// Tensor names reported by GPT-NeoX input errors.
const INPUT_NAMES: InputNames<'static> = InputNames {
    token_ids: "gpt_neox.token_ids",
    token_mask: "gpt_neox.token_mask",
    position_ids: "gpt_neox.position_ids",
};

/// Owner named when an input tensor is on another device than the weights.
const PARAMETERS: &str = "GPT-NeoX model parameters";

// Editable transformer blocks and model composition.

/// One public GPT-NeoX block. Attention and MLP parameters can be edited separately.
#[derive(Module, Debug)]
pub struct GptNeoXLayer<B: Backend> {
    /// Affine normalization feeding attention.
    pub input_layernorm: LayerNorm<B>,
    /// Affine normalization feeding the dense MLP.
    pub post_attention_layernorm: LayerNorm<B>,
    /// Split Q/K/V/O projections; the checkpoint loader unpacks head-major QKV.
    pub attention: GroupedQueryAttention<B>,
    /// Ordinary biased, two-projection GELU network.
    pub mlp: DenseFeedForward<B>,
}

impl<B: Backend> GptNeoXLayer<B> {
    /// Builds one independently editable block from validated dimensions.
    pub fn init(config: &GptNeoXConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        let mut attention_config = GroupedQueryAttentionConfig::new(
            config.hidden_size,
            config.num_attention_heads,
            config.num_attention_heads,
            config.head_dim(),
        )
        .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax);
        attention_config.bias = true;
        let mut attention = attention_config.init(device)?;
        for projection in [
            &mut attention.q_proj,
            &mut attention.k_proj,
            &mut attention.v_proj,
            &mut attention.o_proj,
        ] {
            *projection = normal_linear(
                config.hidden_size,
                config.hidden_size,
                true,
                config.initializer_range,
                device,
            );
        }
        let mut mlp = DenseFeedForwardConfig::new(config.hidden_size, config.intermediate_size)
            .with_activation(Activation::Gelu)
            .with_bias(true)
            .init(device)?;
        mlp.up_proj = normal_linear(
            config.hidden_size,
            config.intermediate_size,
            true,
            config.initializer_range,
            device,
        );
        mlp.down_proj = normal_linear(
            config.intermediate_size,
            config.hidden_size,
            true,
            config.initializer_range,
            device,
        );
        let norm = || {
            LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device)
        };
        Ok(Self {
            input_layernorm: norm(),
            post_attention_layernorm: norm(),
            attention,
            mlp,
        })
    }

    /// Lists the points layer `index` emits, in emission order.
    ///
    /// Query and key points follow RoPE. Scores and patterns appear only in
    /// the eager attention precision mode.
    fn activation_points(&self, index: usize) -> Vec<PointInfo> {
        let layer = Layer::decoder(index);
        let attention = layer.self_attention();
        let mut points = vec![
            layer.residual_pre().info(),
            layer.attention_input().info(),
            attention.query().info(),
            attention.key().info(),
            attention.value().info(),
        ];
        if self.attention.precision_mode == AttentionPrecisionMode::EagerFp32Softmax {
            points.extend([attention.scores().info(), attention.pattern().info()]);
        }
        points.extend([
            attention.head_context().info(),
            layer.attention_output().info(),
            layer.mlp_input().info(),
            layer.mlp_pre_activation().info(),
            layer.mlp_activation().info(),
            layer.mlp_output().info(),
            layer.residual_post().info(),
        ]);
        points
    }

    #[allow(clippy::too_many_arguments)]
    /// Runs one block with post-RoPE Q/K hooks and explicit causal or bidirectional visibility.
    ///
    /// Parallel mode computes `MLP(LN_m(x)) + Attention(LN_a(x)) + x` in upstream order;
    /// sequential mode feeds `x + Attention(LN_a(x))` into the MLP normalization.
    fn forward_scoped(
        &self,
        hidden: Tensor<B, 3>,
        angles: &RotaryAngles<B>,
        token_mask: Option<&TokenMask<B>>,
        parallel: bool,
        causal: bool,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let hidden = hooks.apply(layer.residual_pre(), hidden)?;
        let attention_input = hooks.apply(
            layer.attention_input(),
            self.input_layernorm.forward(hidden.clone()),
        )?;
        let mut projection = self.attention.project_self(attention_input)?;
        (projection.query, projection.key) =
            apply_rotary(projection.query, projection.key, angles)?;
        let [batch, _, sequence, _] = projection.query.dims();
        let mask = padding_attention_mask(
            None,
            token_mask,
            batch,
            self.attention.query_heads,
            sequence,
            sequence,
        )?;
        let attention = self.attention.attend_with_hooks(
            projection,
            mask,
            None,
            causal,
            layer.self_attention(),
            hooks,
        )?;
        let attention = hooks.apply(layer.attention_output(), attention)?;
        let mlp_residual = if parallel {
            hidden.clone()
        } else {
            hidden.clone() + attention.clone()
        };
        let mlp_input = hooks.apply(
            layer.mlp_input(),
            self.post_attention_layernorm.forward(mlp_residual),
        )?;
        let mlp = self.mlp.forward_with_hooks(mlp_input, layer, hooks)?;
        let mlp = hooks.apply(layer.mlp_output(), mlp)?;
        // Preserve the upstream addition order in the parallel case.
        let output = if parallel {
            mlp + attention + hidden
        } else {
            hidden + attention + mlp
        };
        hooks.apply(layer.residual_post(), output)
    }
}

/// A GPT-NeoX backbone and vocabulary head with stateless, differentiable execution.
///
/// This is Hugging Face's `GPTNeoXForCausalLM`: [`Self::forward`] returns
/// vocabulary logits, like [`crate::models::gpt2::Gpt2ForCausalLm`].
/// Use [`Self::forward_hidden_with_hooks`] for task heads, and [`Self::project_hidden`]
/// to decode a raw intermediate residual state through the final normalization.
/// Parameter initialization is random ([`GptNeoXConfig::init_policy`]);
/// [`Self::load_checkpoint`] initializes from explicitly supplied local
/// SafeTensors artifacts.
#[derive(Module, Debug)]
pub struct GptNeoXForCausalLm<B: Backend> {
    /// Input token embeddings, independent of the output projection.
    pub embed_in: Embedding<B>,
    /// Ordered decoder blocks.
    pub layers: Vec<GptNeoXLayer<B>>,
    /// Affine normalization before the vocabulary head.
    pub final_layer_norm: LayerNorm<B>,
    /// Bias-free vocabulary projection.
    pub embed_out: Linear<B>,
    /// Original validated topology and shared numerical conventions for edited modules.
    #[module(skip)]
    pub(super) config: GptNeoXConfig,
}

impl<B: Backend> GptNeoXForCausalLm<B> {
    /// Validates the recipe before allocating randomly initialized modules.
    pub fn init(config: GptNeoXConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        let embed_in = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: config.initializer_range,
            })
            .init(device);
        let layers = (0..config.num_hidden_layers)
            .map(|_| GptNeoXLayer::init(&config, device))
            .collect::<Result<_>>()?;
        let final_layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let embed_out = normal_linear(
            config.hidden_size,
            config.vocab_size,
            false,
            config.initializer_range,
            device,
        );
        Ok(Self {
            embed_in,
            layers,
            final_layer_norm,
            embed_out,
            config,
        })
    }

    /// Returns the validated architectural choices used by this parameter graph.
    pub fn config(&self) -> &GptNeoXConfig {
        &self.config
    }

    /// Sets every layer's attention precision mode.
    ///
    /// [`AttentionPrecisionMode::EagerFp32Softmax`] (the default) exposes
    /// score and pattern hook points; [`AttentionPrecisionMode::BackendDefault`]
    /// runs the backend's fused attention kernel without them. The catalog of
    /// [`Self::activation_points`] follows each layer's mode.
    pub fn set_attention_precision(&mut self, mode: AttentionPrecisionMode) {
        for layer in &mut self.layers {
            layer.attention.precision_mode = mode;
        }
    }

    /// Returns the model with every layer's attention precision mode set to `mode`.
    #[must_use]
    pub fn with_attention_precision(mut self, mode: AttentionPrecisionMode) -> Self {
        self.set_attention_precision(mode);
        self
    }

    /// Lists semantic activation points with their axes, in emission order.
    ///
    /// Q/K points follow RoPE; MLP activation is exact GELU before the output
    /// projection.
    pub fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Decoder.embedding().info()];
        for (index, layer) in self.layers.iter().enumerate() {
            points.extend(layer.activation_points(index));
        }
        points.push(Stack::Decoder.final_norm().info());
        points
    }

    /// Runs ordinary causal attention without requesting intermediate tensors.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_with_hooks(input, options, &mut Hooks::new())
    }

    /// Runs a causal sequence with selected activation capture and modifications.
    ///
    /// Hidden-state diagnostics are the embedding, every block's output, and
    /// the final normalized state (`L + 2` tensors; see the module docs for the
    /// mapping to Hugging Face's `L + 1`).
    pub fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_direction(input, options, true, hooks)
    }

    /// Runs full-sequence bidirectional attention without requesting activations.
    pub fn forward_bidirectional(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_bidirectional_with_hooks(input, options, &mut Hooks::new())
    }

    /// Allows future tokens while retaining padding exclusions and positional encoding.
    pub fn forward_bidirectional_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_direction(input, options, false, hooks)
    }

    /// Validates diagnostic requests and scopes hooks around backbone execution and selected readout.
    ///
    /// See the module's "Diagnostics" section: `options.attentions` fails with
    /// [`Error::BackendCapability`] before any work, while `router_logits` and
    /// `auxiliary_losses` succeed with empty dense diagnostics.
    fn forward_direction(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        causal: bool,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        hooks.run(self.activation_points(), |hooks| {
            if options.attentions {
                return Err(Error::BackendCapability(
                    "GPT-NeoX attention probabilities are available through \
                     self_attention().pattern() hooks"
                        .into(),
                ));
            }
            let (hidden, hidden_states) =
                self.backbone(&input, options.hidden_states, causal, hooks)?;
            let selected =
                select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
            let mut output = CausalLmOutput::from_logits(self.readout(selected)?);
            output.final_hidden_state = options.hidden_states.then_some(hidden);
            output.hidden_states = hidden_states;
            Ok(output)
        })
    }

    /// Returns final normalized `[batch, token, hidden]` states without vocabulary logits.
    pub fn forward_hidden(&self, input: CausalLmInput<B>) -> Result<Tensor<B, 3>> {
        self.forward_hidden_with_hooks(input, &mut Hooks::new())
    }

    /// Returns final normalized states after causal activation interventions.
    pub fn forward_hidden_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        hooks.run(self.activation_points(), |hooks| {
            Ok(self.backbone(&input, false, true, hooks)?.0)
        })
    }

    /// Applies only the final LayerNorm to a raw residual state.
    pub fn normalize_hidden(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        self.validate_hidden(&hidden)?;
        Ok(self.final_layer_norm.forward(hidden))
    }

    /// Applies only the vocabulary projection to an already normalized hidden state.
    pub fn readout(&self, normalized: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        self.validate_hidden(&normalized)?;
        operations::linear(normalized, &self.embed_out)
    }

    /// Decodes a raw intermediate residual with final LayerNorm and vocabulary weights.
    /// A tuned lens supplies `h + translator.forward(h)` here; a raw logit lens supplies `h`.
    pub fn project_hidden(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        self.readout(self.normalize_hidden(hidden)?)
    }

    /// Checks nonempty `[batch, token, hidden]` states and the final norm's dtype/device contract.
    fn validate_hidden(&self, hidden: &Tensor<B, 3>) -> Result<()> {
        let [batch, sequence, width] = hidden.dims();
        if batch == 0 || sequence == 0 || width != self.config.hidden_size {
            return Err(Error::shape(
                "gpt_neox.hidden",
                format!("nonempty [batch, token, {}]", self.config.hidden_size),
                format!("{:?}", hidden.dims()),
            ));
        }
        let parameter = self.final_layer_norm.gamma.val();
        Expected::<B> {
            device: &parameter.device(),
            dtype: parameter.dtype(),
        }
        .norm(
            &self.final_layer_norm,
            self.config.hidden_size,
            Label::Model("final_layer_norm"),
        )?;
        if hidden.device() != parameter.device() || hidden.dtype() != parameter.dtype() {
            return Err(Error::configuration(
                "gpt_neox.hidden",
                "dtype and device must match the model parameters",
            ));
        }
        Ok(())
    }

    /// Embeds validated tokens, runs editable layers with their actual rotary widths, and normalizes.
    ///
    /// Retained states are the embedding, every block's output, and the final
    /// normalized state (`L + 2` tensors).
    fn backbone(
        &self,
        input: &CausalLmInput<B>,
        retain: bool,
        causal: bool,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>)> {
        self.validate_parameters()?;
        let positions =
            validate_public_input(input, &self.config, &self.embed_in.weight.lazy_device())?
                .require_positions(INPUT_NAMES.position_ids)?;
        let mut hidden = hooks.apply(
            Stack::Decoder.embedding(),
            self.embed_in.forward(input.token_ids.clone()),
        )?;
        let mut states = Vec::with_capacity(if retain { self.layers.len() + 2 } else { 0 });
        if retain {
            states.push(hidden.clone());
        }
        hidden = self.run_blocks(
            hidden,
            &positions,
            input.token_mask.as_ref(),
            0..self.layers.len(),
            causal,
            hooks,
            retain.then_some(&mut states),
        )?;
        hidden = hooks.apply(Stack::Decoder.final_norm(), self.normalize_hidden(hidden)?)?;
        if retain {
            states.push(hidden.clone());
        }
        Ok((hidden, states))
    }

    /// Runs layers `range` on validated positions, sharing RoPE tables across layers.
    ///
    /// Publicly replacing a layer's attention can change its head width, and
    /// the configured rotary fraction applies to each layer's actual width, so
    /// tables are keyed by head width: an unedited model builds one table pair
    /// per call instead of one per layer. Tables are cast once to the residual
    /// dtype. Each block output is appended to `retained` when given.
    #[allow(clippy::too_many_arguments)]
    fn run_blocks(
        &self,
        mut hidden: Tensor<B, 3>,
        positions: &Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        range: std::ops::Range<usize>,
        causal: bool,
        hooks: &mut Hooks<'_, B>,
        mut retained: Option<&mut Vec<Tensor<B, 3>>>,
    ) -> Result<Tensor<B, 3>> {
        let mut tables: Vec<(usize, RotaryAngles<B>)> = Vec::with_capacity(1);
        for index in range {
            let layer = &self.layers[index];
            let width = layer.attention.head_dim;
            let slot = match tables.iter().position(|(cached, _)| *cached == width) {
                Some(slot) => slot,
                None => {
                    let rope = RopeConfig::from_partial_factor(
                        width,
                        self.config.rotary_pct,
                        self.config.rotary_emb_base,
                        RopeScaling::None,
                    )
                    .map_err(|error| {
                        Error::configuration(
                            format!("gpt_neox.layers.{index}.rotary"),
                            error.to_string(),
                        )
                    })?;
                    let angles = rotary_angles(
                        rope,
                        positions.clone(),
                        self.config.max_position_embeddings,
                    )?
                    .to_dtype(hidden.dtype());
                    tables.push((width, angles));
                    tables.len() - 1
                }
            };
            hidden = layer.forward_scoped(
                hidden,
                &tables[slot].1,
                token_mask,
                self.config.use_parallel_residual,
                causal,
                Layer::decoder(index),
                hooks,
            )?;
            if let Some(retained) = retained.as_deref_mut() {
                retained.push(hidden.clone());
            }
        }
        Ok(hidden)
    }

    /// Checks edited public modules before unchecked backend reshapes or residual additions.
    fn validate_parameters(&self) -> Result<()> {
        let embedding = self.embed_in.weight.val();
        let device = embedding.device();
        let dtype = embedding.dtype();
        let width = self.config.hidden_size;
        let expect = Expected {
            device: &device,
            dtype,
        };
        expect.tensor(
            &embedding,
            [self.config.vocab_size, width],
            Label::Model("embed_in.weight"),
        )?;
        expect.norm(
            &self.final_layer_norm,
            width,
            Label::Model("final_layer_norm"),
        )?;
        expect.linear(
            &self.embed_out,
            width,
            self.config.vocab_size,
            Label::Model("embed_out"),
        )?;
        for (index, layer) in self.layers.iter().enumerate() {
            let label = |component| Label::Layer(index, component);
            expect.norm(&layer.input_layernorm, width, label("input_layernorm"))?;
            expect.norm(
                &layer.post_attention_layernorm,
                width,
                label("post_attention_layernorm"),
            )?;
            let attention = &layer.attention;
            let widths = match layer_widths(
                attention.hidden_size,
                width,
                attention.query_heads,
                attention.key_value_heads,
                attention.head_dim,
                attention.output_gate != AttentionOutputGate::None,
            ) {
                Ok(widths) => widths,
                Err(LayerWidthError::Geometry) => {
                    return Err(Error::configuration(
                        label("attention").render(),
                        "head dimensions must be nonzero, compatible, and preserve residual width",
                    ));
                }
                Err(LayerWidthError::Overflow) => {
                    return Err(Error::configuration(
                        label("attention").render(),
                        "attention projection widths overflow usize",
                    ));
                }
                Err(LayerWidthError::GatedOverflow) => {
                    return Err(Error::configuration(
                        Label::Layer(index, "").render(),
                        "gated query width overflows usize",
                    ));
                }
            };
            let (query_width, key_width, q_projection_width) =
                (widths.query, widths.key_value, widths.query_projection);
            expect.linear(
                &attention.q_proj,
                width,
                q_projection_width,
                label("attention.q_proj"),
            )?;
            expect.linear(
                &attention.k_proj,
                width,
                key_width,
                label("attention.k_proj"),
            )?;
            expect.linear(
                &attention.v_proj,
                width,
                key_width,
                label("attention.v_proj"),
            )?;
            expect.linear(
                &attention.o_proj,
                query_width,
                width,
                label("attention.o_proj"),
            )?;
            for norm in [&attention.query_norm, &attention.key_norm]
                .into_iter()
                .flatten()
            {
                expect.tensor(
                    &norm.weight.val(),
                    [attention.head_dim],
                    label("attention.query_key_norm"),
                )?;
            }
            let intermediate = layer.mlp.up_proj.weight.val().dims()[1];
            if intermediate == 0 {
                return Err(Error::configuration(
                    label("mlp").render(),
                    "intermediate width must be nonzero",
                ));
            }
            expect.linear(
                &layer.mlp.up_proj,
                width,
                intermediate,
                label("mlp.up_proj"),
            )?;
            expect.linear(
                &layer.mlp.down_proj,
                intermediate,
                width,
                label("mlp.down_proj"),
            )?;
        }
        Ok(())
    }
}

// Public-edit safety checks, initialization, and input/output boundaries.

/// Names a parameter in validation errors; the name is formatted only on failure.
///
/// Validation runs on every forward, so labels stay `(index, &'static str)`
/// pairs instead of allocating formatted strings for parameters that pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Label {
    /// A model-level parameter, `gpt_neox.<name>`.
    Model(
        /// Parameter path below `gpt_neox.`, such as `"embed_in.weight"`.
        &'static str,
    ),
    /// A layer parameter, `gpt_neox.layers.<index>.<name>` (or the layer itself for `""`).
    Layer(
        /// Zero-based decoder layer index.
        usize,
        /// Parameter path below the layer, or `""` for the layer itself.
        &'static str,
    ),
}

impl Label {
    /// Formats the dotted parameter path reported in errors.
    fn render(self) -> String {
        match self {
            Self::Model(name) => format!("gpt_neox.{name}"),
            Self::Layer(index, "") => format!("gpt_neox.layers.{index}"),
            Self::Layer(index, name) => format!("gpt_neox.layers.{index}.{name}"),
        }
    }
}

/// The device and dtype every parameter must share with the token embedding.
#[derive(Debug, Clone, Copy)]
struct Expected<'a, B: Backend> {
    /// Device of the token embedding.
    device: &'a B::Device,
    /// Floating dtype of the token embedding.
    dtype: DType,
}

impl<B: Backend> Expected<'_, B> {
    /// Checks an exact parameter shape and the common dtype/device before backend arithmetic.
    fn tensor<const D: usize>(
        &self,
        tensor: &Tensor<B, D>,
        shape: [usize; D],
        label: Label,
    ) -> Result<()> {
        if tensor.dims() != shape {
            return Err(Error::shape(
                label.render(),
                format!("{shape:?}"),
                format!("{:?}", tensor.dims()),
            ));
        }
        if tensor.device() != *self.device || tensor.dtype() != self.dtype {
            return Err(Error::configuration(
                label.render(),
                "parameters must share device and dtype",
            ));
        }
        Ok(())
    }

    /// Checks channelwise affine LayerNorm parameters against the residual width and storage.
    fn norm(&self, norm: &LayerNorm<B>, width: usize, label: Label) -> Result<()> {
        self.tensor(&norm.gamma.val(), [width], label)?;
        if let Some(beta) = &norm.beta {
            self.tensor(&beta.val(), [width], label)?;
        }
        Ok(())
    }

    /// Checks runtime `[input, output]` weights and optional `[output]` bias storage.
    fn linear(&self, linear: &Linear<B>, input: usize, output: usize, label: Label) -> Result<()> {
        self.tensor(&linear.weight.val(), [input, output], label)?;
        if let Some(bias) = &linear.bias {
            self.tensor(&bias.val(), [output], label)?;
        }
        Ok(())
    }
}

/// Initializes Gaussian weights and zero bias with checkpoint-facing Col record layout.
///
/// Shared initialization keeps runtime `[input, output]` tensors autodiff-safe while
/// preserving the source `[output, input]` save/load convention.
fn normal_linear<B: Backend>(
    input: usize,
    output: usize,
    bias: bool,
    std: f64,
    device: &B::Device,
) -> Linear<B> {
    let mut layer = operations::init_linear(
        &LinearConfig::new(input, output)
            .with_bias(bias)
            .with_layout(LinearLayout::Col)
            .with_initializer(Initializer::Normal { mean: 0.0, std }),
        device,
    );
    if bias {
        layer.bias = Some(Initializer::Zeros.init([output], device));
    }
    layer
}

/// Validates token and position ranges, padding rows, sequence capacity, and device agreement.
///
/// Absent position IDs use physical `0..sequence` positions, including padded
/// slots, as the reference GPT-NeoX forward does. Token, mask, and position
/// values share one device-to-host readback; implicit positions are built on
/// the device and need no read.
fn validate_public_input<B: Backend>(
    input: &CausalLmInput<B>,
    config: &GptNeoXConfig,
    device: &B::Device,
) -> Result<ValidatedInput<B>> {
    let [batch, sequence] = input.token_ids.dims();
    if batch == 0 || sequence == 0 || sequence > config.max_position_embeddings {
        return Err(Error::shape(
            INPUT_NAMES.token_ids,
            format!(
                "nonempty [batch, sequence<={}]",
                config.max_position_embeddings
            ),
            format!("[{batch}, {sequence}]"),
        ));
    }
    validate_input(
        input,
        device,
        &InputContract::new(INPUT_NAMES, PARAMETERS, config.vocab_size)
            .with_position_limit(config.max_position_embeddings)
            .with_implicit_positions(ImplicitPositions::PhysicalColumns),
    )
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
