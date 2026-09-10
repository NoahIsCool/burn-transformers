//! Burn implementation of the Qwen3.5/Qwen3.8 text decoder.
//!
//! The architecture is a pre-normalized residual decoder. Three Gated DeltaNet
//! layers normally alternate with one gated full-attention layer, while every
//! layer shares the same dense SwiGLU feed-forward topology. Qwen3.8 stores
//! zero-centered RMSNorm parameters (`scale = 1 + weight`) throughout the text
//! decoder and rotates only a prefix of each full-attention head.
//!
//! This module owns exact family wiring. Projection, rotary, attention,
//! DeltaNet, feed-forward, and normalization mathematics remain in shared
//! modules so custom experiments can recombine them without model-name checks.

use std::ops::Range;

use burn::{
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{DType, ElementConversion, Int, Tensor, TensorData, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, LogitsSelection,
    Result,
    architecture::TokenMask,
    attention::{
        AttentionOutputGate, AttentionPrecisionMode, GroupedQueryAttention,
        GroupedQueryAttentionConfig, padding_attention_mask,
    },
    delta_net::{GatedDeltaNet, GatedDeltaNetConfig},
    experiment::{AttentionPolicy, ExecutionContext, HiddenState},
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    intervention::{Hooks, Layer, PointInfo, Stack},
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{
        DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling, YarnConfig, apply_rotary,
        rotary_angles,
    },
};

use super::{Qwen3_5LayerType, Qwen3_5TextConfig};
use crate::models::config::RopeType;

mod decode;
mod mtp;
mod speculative;

pub use decode::{Qwen3_5DecodeState, Qwen3_5LayerCache};
pub use mtp::{Qwen3_5Mtp, Qwen3_5MtpOutput, Qwen3_5MtpPrediction};
pub use speculative::{
    Qwen3_5SpeculativeConfig, Qwen3_5SpeculativeDecoder, Qwen3_5SpeculativeOutput,
};

/// One Qwen decoder block with exactly one active sequence mixer.
#[derive(Module, Debug)]
pub struct Qwen3_5DecoderLayer<B: Backend> {
    /// Gated DeltaNet module for a linear-attention layer.
    pub linear_attn: Option<GatedDeltaNet<B>>,
    /// Gated GQA module for a full-attention layer.
    pub self_attn: Option<GroupedQueryAttention<B>>,
    /// Dense SwiGLU feed-forward network.
    pub mlp: GatedFeedForward<B>,
    /// Zero-centered RMSNorm before the sequence mixer.
    pub input_layernorm: RmsNorm<B>,
    /// Zero-centered RMSNorm before the feed-forward network.
    pub post_attention_layernorm: RmsNorm<B>,
    /// Serialized mixer schedule entry; it owns no parameters.
    #[module(skip)]
    pub layer_type: Qwen3_5LayerType,
}

impl<B: Backend> Qwen3_5DecoderLayer<B> {
    /// Initializes the block selected by one schedule entry.
    pub(super) fn init(
        config: &Qwen3_5TextConfig,
        layer_type: Qwen3_5LayerType,
        device: &B::Device,
    ) -> Result<Self> {
        let norm = || {
            RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_zero_centered(true)
                .init(device)
        };
        let (linear_attn, self_attn) = match layer_type {
            Qwen3_5LayerType::LinearAttention => {
                let mut delta = GatedDeltaNetConfig::new(
                    config.hidden_size,
                    config.linear_num_key_heads,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                );
                delta.convolution_kernel = config.linear_conv_kernel_dim;
                delta.normalization_epsilon = config.rms_norm_eps;
                (Some(delta.init(device)?), None)
            }
            Qwen3_5LayerType::FullAttention => {
                let mut attention = GroupedQueryAttentionConfig::new(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                );
                attention.bias = config.attention_bias;
                attention.query_key_norm_epsilon = Some(config.rms_norm_eps);
                attention.query_key_norm_zero_centered = true;
                attention.output_gate = if config.attn_output_gate {
                    AttentionOutputGate::Sigmoid
                } else {
                    AttentionOutputGate::None
                };
                (None, Some(attention.init(device)?))
            }
        };
        Ok(Self {
            linear_attn,
            self_attn,
            mlp: GatedFeedForwardConfig::new(config.hidden_size, config.intermediate_size)
                .init(device)?,
            input_layernorm: norm()?,
            post_attention_layernorm: norm()?,
            layer_type,
        })
    }

    /// Evaluates one stateless residual block from an existing hidden tensor.
    ///
    /// This can be used to run a prefix or splice custom processing between
    /// blocks. `hidden` is `[batch, sequence, hidden_size]`; `positions` is
    /// `[batch, sequence]`. Positions, padding, and rotary conventions must be
    /// consistent with the surrounding blocks. `evaluated_context` must exceed
    /// every position ID and remain consistent across blocks for rotary scaling.
    /// DeltaNet and full attention both preserve the family's causal behavior.
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        rope: RopeConfig,
        evaluated_context: usize,
    ) -> Result<Tensor<B, 3>> {
        let [batch, sequence, width] = hidden.dims();
        let expected_width = self.input_layernorm.weight.lazy_shape()[0];
        if batch == 0 || sequence == 0 || width != expected_width {
            return Err(Error::shape(
                "qwen.block.hidden",
                format!("non-empty [batch, sequence, {expected_width}]"),
                format!("{:?}", hidden.dims()),
            ));
        }
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "qwen.block.positions",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
        let device = self.input_layernorm.weight.lazy_device();
        ensure_model_device("qwen.block.hidden", &hidden.device(), &device)?;
        let projection_dtype = self.mlp.gate_proj.weight.val().dtype();
        if !matches!(projection_dtype, DType::QFloat(_)) {
            ensure_model_dtype("qwen.block.hidden", hidden.dtype(), projection_dtype)?;
        }
        ensure_model_device("qwen.block.positions", &positions.device(), &device)?;
        if let Some(mask) = token_mask {
            ensure_model_device("qwen.block.token_mask", &mask.as_tensor().device(), &device)?;
        }
        validate_mask(token_mask, batch, sequence)?;
        decode::validate_position_range(positions.clone(), evaluated_context)?;
        self.forward_inner(
            hidden,
            positions,
            token_mask,
            rope,
            evaluated_context,
            0,
            None,
            None,
        )
    }

    /// Evaluates a block, applying optional interventions to its actual graph.
    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        rope: RopeConfig,
        evaluated_context: usize,
        index: usize,
        mut hooks: Option<&mut Hooks<'_, B>>,
        policy: Option<&AttentionPolicy<B>>,
    ) -> Result<Tensor<B, 3>> {
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
        let mixed = match self.layer_type {
            Qwen3_5LayerType::LinearAttention => self
                .linear_attn
                .as_ref()
                .ok_or_else(|| {
                    Error::Checkpoint("Qwen linear layer has no DeltaNet module".into())
                })?
                .forward(normalized, token_mask)?,
            Qwen3_5LayerType::FullAttention => {
                let attention = self.self_attn.as_ref().ok_or_else(|| {
                    Error::Checkpoint("Qwen full-attention layer has no attention module".into())
                })?;
                let mut projection = attention.project_self(normalized)?;
                let angles = rotary_angles(rope, positions, evaluated_context)?;
                (projection.query, projection.key) =
                    apply_rotary(projection.query, projection.key, &angles)?;
                let [batch, _, query_sequence, _] = projection.query.dims();
                let key_sequence = projection.key.dims()[2];
                let mut mask = padding_attention_mask(
                    token_mask,
                    token_mask,
                    batch,
                    attention.query_heads,
                    query_sequence,
                    key_sequence,
                )?;
                let mut causal = true;
                let mut bias = None;
                if let Some(policy) = policy {
                    let shape = [batch, attention.query_heads, query_sequence, key_sequence];
                    mask = policy.exclusion_mask(mask, shape, &projection.query.device())?;
                    bias = policy.expanded_bias(shape, &projection.query.device())?;
                    causal = policy.causal;
                }
                match hooks.as_deref_mut() {
                    Some(hooks) => attention.attend_with_hooks(
                        projection,
                        mask,
                        bias,
                        causal,
                        layer.self_attention(),
                        hooks,
                    )?,
                    None => attention.attend(projection, mask, bias, causal)?,
                }
            }
        };
        let mixed = match hooks.as_deref_mut() {
            Some(hooks) => hooks.apply(layer.attention_output(), mixed)?,
            None => mixed,
        };
        let hidden = residual + mixed;
        let residual = hidden.clone();
        let normalized = self.post_attention_layernorm.forward(hidden);
        let mixed = match hooks.as_deref_mut() {
            Some(hooks) => {
                let normalized = hooks.apply(layer.mlp_input(), normalized)?;
                let gate = hooks.apply(
                    layer.mlp_pre_activation(),
                    operations::linear(normalized.clone(), &self.mlp.gate_proj)?,
                )?;
                let gate = hooks.apply(
                    layer.mlp_activation(),
                    crate::feed_forward::apply_activation(gate, self.mlp.activation),
                )?;
                let content = operations::linear(normalized, &self.mlp.up_proj)?;
                hooks.apply(
                    layer.mlp_output(),
                    operations::linear(gate * content, &self.mlp.down_proj)?,
                )?
            }
            None => self.mlp.forward(normalized)?,
        };
        let hidden = residual + mixed;
        match hooks {
            Some(hooks) => hooks.apply(layer.residual_post(), hidden),
            None => Ok(hidden),
        }
    }
}

/// Parameterized Qwen text backbone without the vocabulary projection.
#[derive(Module, Debug)]
pub struct Qwen3_5TextModel<B: Backend> {
    /// Token embedding table.
    pub embed_tokens: Embedding<B>,
    /// Ordered hybrid decoder blocks.
    pub layers: Vec<Qwen3_5DecoderLayer<B>>,
    /// Final zero-centered RMSNorm.
    pub norm: RmsNorm<B>,
    /// Validated family configuration excluded from parameter records.
    #[module(skip)]
    pub config: Qwen3_5TextConfig,
    /// Validated partial rotary configuration.
    #[module(skip)]
    pub rope: RopeConfig,
    /// Rotary configuration for each block, allowing edited attention head widths.
    #[module(skip)]
    pub layer_rope: Vec<RopeConfig>,
    /// Whether this backbone was built for the homogeneous family cache layout.
    #[module(skip)]
    pub(crate) supports_family_cache: bool,
}

impl<B: Backend> Qwen3_5TextModel<B> {
    /// Initializes lazy parameters for the exact configured hybrid schedule.
    pub fn init(config: Qwen3_5TextConfig, device: &B::Device) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        let rope = qwen_rope_config(&config)?;
        let layers = config
            .resolved_layer_types()
            .into_iter()
            .map(|layer_type| Qwen3_5DecoderLayer::init(&config, layer_type, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers,
            norm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_zero_centered(true)
                .init(device)?,
            layer_rope: vec![rope; config.num_hidden_layers],
            supports_family_cache: true,
            config,
            rope,
        })
    }

    /// Embeds tokens and resolves positions and padding once, before block zero.
    ///
    /// Edit `state.context` to select attention connectivity, then pass the state
    /// through explicit block ranges. This method never applies final normalization.
    pub fn embed(&self, input: &CausalLmInput<B>) -> Result<HiddenState<B>> {
        let [batch, sequence] = input.token_ids.dims();
        if batch == 0 || sequence == 0 {
            return Err(Error::shape(
                "qwen.input_ids",
                "non-empty [batch, sequence]",
                format!("[{batch}, {sequence}]"),
            ));
        }
        validate_input_devices(input, &self.embed_tokens.weight.lazy_device())?;
        validate_token_ids(input.token_ids.clone(), self.config.vocab_size)?;
        validate_mask(input.token_mask.as_ref(), batch, sequence)?;
        let positions = resolve_positions(input, batch, sequence)?;
        decode::validate_position_range(positions.clone(), self.config.max_position_embeddings)?;
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
    /// For example, run `0..3`, transform `state.hidden` with a registered module,
    /// then resume `3..model.layers.len()`. Neither embeddings nor final norm are
    /// reapplied. Full attention accepts context policies; DeltaNet requires its
    /// original causal recurrence and rejects arbitrary attention masks/biases.
    pub fn run_layers(&self, state: HiddenState<B>, range: Range<usize>) -> Result<HiddenState<B>> {
        self.run_layers_inner(state, range, None)
    }

    /// Runs a block range with actions addressed by their original layer indices.
    /// Requests outside this range fail before any callback or block executes.
    pub fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        let prefixes = (0..self.layers.len())
            .filter(|index| range.contains(index))
            .map(|index| format!("decoder.layers.{index}."))
            .collect::<Vec<_>>();
        let points = self
            .activation_points()
            .into_iter()
            .filter(|point| prefixes.iter().any(|prefix| point.name.starts_with(prefix)))
            .collect();
        hooks.run(points, |hooks| {
            self.run_layers_inner(state, range, Some(hooks))
        })
    }

    /// Validates and executes a contiguous Qwen block range on `[batch, token, hidden]`.
    ///
    /// All policies and model dtype/device/position contracts are checked before
    /// execution. Full attention accepts custom connectivity; DeltaNet requires
    /// its causal recurrence. The returned cursor names the first unexecuted block.
    fn run_layers_inner(
        &self,
        mut state: HiddenState<B>,
        range: Range<usize>,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<HiddenState<B>> {
        state.validate(self.config.hidden_size)?;
        state.validate_range(range.clone(), self.layers.len())?;
        ensure_model_device(
            "qwen.execution.hidden",
            &state.hidden.device(),
            &self.embed_tokens.weight.lazy_device(),
        )?;
        ensure_model_dtype(
            "qwen.execution.hidden",
            state.hidden.dtype(),
            self.embed_tokens.weight.val().dtype(),
        )?;
        if state.context.encoder.is_some()
            || state.context.cross_attention.is_some()
            || !state.context.layer_cross_attention.is_empty()
        {
            return Err(Error::configuration(
                "qwen.execution.encoder",
                "Qwen decoder blocks do not define cross-attention",
            ));
        }
        let [batch, sequence, _] = state.hidden.dims();
        validate_mask(state.context.token_mask.as_ref(), batch, sequence)?;
        decode::validate_position_range(
            state.context.positions.clone(),
            self.config.max_position_embeddings,
        )?;
        decode::validate_position_range(
            state.context.positions.clone(),
            state.context.evaluated_context,
        )?;
        if self.layer_rope.len() != self.layers.len() {
            return Err(Error::configuration(
                "qwen.layer_rope",
                "one rotary configuration is required per block",
            ));
        }
        for (address, _) in &state.context.layer_attention {
            if !(0..self.layers.len()).any(|index| *address == Layer::decoder(index)) {
                return Err(Error::configuration(
                    "qwen.execution.layer_attention",
                    "override addresses an unavailable Qwen decoder layer",
                ));
            }
        }
        for index in range.clone() {
            let block = &self.layers[index];
            let policy = state.context.attention_for(Layer::decoder(index), true);
            match block.layer_type {
                Qwen3_5LayerType::LinearAttention
                    if !policy.causal || policy.allowed.is_some() || policy.bias.is_some() =>
                {
                    return Err(Error::BackendCapability(format!(
                        "Qwen DeltaNet layer {index} requires its causal recurrence and does not support custom softmax connectivity or score bias"
                    )));
                }
                Qwen3_5LayerType::FullAttention => {
                    let attention = block.self_attn.as_ref().ok_or_else(|| {
                        Error::Checkpoint("full-attention block lacks its mixer".into())
                    })?;
                    policy.validate(
                        [batch, attention.query_heads, sequence, sequence],
                        &state.hidden.device(),
                    )?;
                    if let Some(bias) = &policy.bias {
                        ensure_model_dtype(
                            "qwen.execution.attention.bias",
                            bias.dtype(),
                            state.hidden.dtype(),
                        )?;
                    }
                }
                _ => {}
            }
        }
        for index in range.clone() {
            let policy = state.context.attention_for(Layer::decoder(index), true);
            state.hidden = self.layers[index].forward_inner(
                state.hidden,
                state.context.positions.clone(),
                state.context.token_mask.as_ref(),
                self.layer_rope[index],
                state.context.evaluated_context,
                index,
                hooks.as_deref_mut(),
                Some(&policy),
            )?;
        }
        state.next_layer = range.end;
        Ok(state)
    }

    /// Applies the backbone's final norm exactly once at an explicit boundary.
    /// This may terminate a prefix; resuming blocks requires an unnormalized state.
    pub fn normalize(&self, mut state: HiddenState<B>) -> Result<HiddenState<B>> {
        state.validate(self.config.hidden_size)?;
        if state.normalized || state.next_layer > self.layers.len() {
            return Err(Error::configuration(
                "qwen.execution.normalize",
                "expected an unnormalized state at a valid block boundary",
            ));
        }
        ensure_model_device(
            "qwen.execution.hidden",
            &state.hidden.device(),
            &self.norm.weight.lazy_device(),
        )?;
        ensure_model_dtype(
            "qwen.execution.hidden",
            state.hidden.dtype(),
            self.embed_tokens.weight.val().dtype(),
        )?;
        state.hidden = self.norm.forward(state.hidden);
        state.normalized = true;
        Ok(state)
    }

    /// Applies final normalization with a scoped final-norm capture or edit.
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

    /// Projects hidden states through the live embedding table without a second parameter.
    /// This reproduces a tied Qwen vocabulary head; for an originally untied checkpoint,
    /// choosing this method deliberately changes its readout.
    pub fn tied_vocabulary_projection(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [batch, sequence, width] = hidden.dims();
        if batch == 0 || sequence == 0 || width != self.config.hidden_size {
            return Err(Error::shape(
                "qwen.tied_vocabulary_projection",
                format!("non-empty [batch, token, {}]", self.config.hidden_size),
                format!("{:?}", hidden.dims()),
            ));
        }
        ensure_model_device(
            "qwen.tied_vocabulary_projection",
            &hidden.device(),
            &self.embed_tokens.weight.lazy_device(),
        )?;
        ensure_model_dtype(
            "qwen.tied_vocabulary_projection",
            hidden.dtype(),
            self.embed_tokens.weight.val().dtype(),
        )?;
        Ok(burn::tensor::module::linear(
            hidden,
            self.embed_tokens.weight.val().transpose(),
            None,
        ))
    }

    /// Makes an independent readout initialized from the embedding table.
    /// The new parameter has a distinct identity and autodiff leaf, so future
    /// updates to this head and the embedding are independent. Own it in your
    /// experiment's `Module` tree to include it in saving and optimization.
    pub fn independent_vocabulary_head(&self) -> Linear<B> {
        Linear {
            weight: Param::from_tensor(self.embed_tokens.weight.val().transpose().detach()),
            bias: None,
        }
    }

    /// Evaluates the text backbone without projecting into the vocabulary.
    ///
    /// The first returned tensor is the normalized `[batch, sequence, hidden]`
    /// representation, suitable for a classifier or custom task head. When
    /// requested, the second return value contains embedding, block outputs,
    /// and final normalization output, in that order.
    pub fn forward(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
    ) -> Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>)> {
        self.forward_inner(input, retain_hidden_states, None)
    }

    /// Lists supported activation points in execution order.
    ///
    /// Attention input/output are the normalized input and projected output of
    /// the configured sequence mixer, whether full attention or DeltaNet. MLP
    /// pre-activation and activation refer to the gate branch before and after
    /// SiLU, respectively, before multiplication by the content branch.
    /// Full-attention layers additionally expose Q/K after rotary transforms,
    /// native-head V, and head context before gating. Scores and patterns are
    /// available only after explicitly configuring that attention module with
    /// [`AttentionPrecisionMode::EagerFp32Softmax`].
    /// DeltaNet recurrent internals and cached decoding are not hookable here.
    pub fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Decoder.embedding().info()];
        for (index, block) in self.layers.iter().enumerate() {
            let layer = Layer::decoder(index);
            points.extend([layer.residual_pre().info(), layer.attention_input().info()]);
            if block.layer_type == Qwen3_5LayerType::FullAttention {
                let attention = layer.self_attention();
                points.extend([
                    attention.query().info(),
                    attention.key().info(),
                    attention.value().info(),
                ]);
                if block.self_attn.as_ref().is_some_and(|module| {
                    module.precision_mode == AttentionPrecisionMode::EagerFp32Softmax
                }) {
                    points.extend([attention.scores().info(), attention.pattern().info()]);
                }
                points.push(attention.head_context().info());
            }
            points.extend([
                layer.attention_output().info(),
                layer.mlp_input().info(),
                layer.mlp_pre_activation().info(),
                layer.mlp_activation().info(),
                layer.mlp_output().info(),
                layer.residual_post().info(),
            ]);
        }
        points.push(Stack::Decoder.final_norm().info());
        points
    }

    /// Evaluates the backbone while capturing or replacing selected activations.
    ///
    /// Replacements participate in subsequent blocks and remain in the
    /// autodiff graph. Hooks are validated against [`Self::activation_points`]
    /// before evaluating the model. This stateless method does not read or
    /// update decode caches.
    pub fn forward_with_hooks(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>)> {
        hooks.run(self.activation_points(), |hooks| {
            self.forward_inner(input, retain_hidden_states, Some(hooks))
        })
    }

    /// Shared stateless execution, retaining the original path without hooks.
    fn forward_inner(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>)> {
        let [batch, sequence] = input.token_ids.dims();
        if batch == 0 || sequence == 0 {
            return Err(Error::shape(
                "qwen.input_ids",
                "non-empty [batch, sequence]",
                format!("[{batch}, {sequence}]"),
            ));
        }
        validate_input_devices(input, &self.embed_tokens.weight.lazy_device())?;
        validate_token_ids(input.token_ids.clone(), self.config.vocab_size)?;
        validate_mask(input.token_mask.as_ref(), batch, sequence)?;
        let positions = resolve_positions(input, batch, sequence)?;
        decode::validate_position_range(positions.clone(), self.config.max_position_embeddings)?;
        let evaluated_context = decode::evaluated_context(positions.clone())?;
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
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_inner(
                hidden,
                positions.clone(),
                input.token_mask.as_ref(),
                self.layer_rope.get(index).copied().ok_or_else(|| {
                    Error::configuration(
                        "qwen.layer_rope",
                        "missing rotary configuration for a block",
                    )
                })?,
                evaluated_context,
                index,
                hooks.as_deref_mut(),
                None,
            )?;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.norm.forward(hidden);
        if let Some(hooks) = hooks {
            hidden = hooks.apply(Stack::Decoder.final_norm(), hidden)?;
        }
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok((hidden, hidden_states))
    }
}

/// Decoder-only Qwen language model with a tied or independent output head.
#[derive(Module, Debug)]
pub struct Qwen3_5ForCausalLm<B: Backend> {
    /// Hybrid text backbone.
    pub model: Qwen3_5TextModel<B>,
    /// Independent vocabulary projection, absent when the embedding is tied.
    ///
    /// Tied checkpoints reuse `model.embed_tokens.weight` directly during the
    /// forward pass. Keeping that case parameter-free guarantees one physical
    /// allocation instead of relying on two differently mapped Burn parameters.
    pub lm_head: Option<Linear<B>>,
    /// Optional official multi-token-prediction auxiliary graph.
    pub mtp: Option<Qwen3_5Mtp<B>>,
}

impl<B: Backend> Qwen3_5ForCausalLm<B> {
    /// Initializes a validated inference/training graph with lazy parameters.
    pub fn init(config: Qwen3_5TextConfig, device: &B::Device) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        let hidden_size = config.hidden_size;
        let vocabulary_size = config.vocab_size;
        let tie_word_embeddings = config.tie_word_embeddings;
        Ok(Self {
            mtp: (config.mtp_num_hidden_layers > 0)
                .then(|| Qwen3_5Mtp::init(&config, device))
                .transpose()?,
            model: Qwen3_5TextModel::init(config, device)?,
            lm_head: (!tie_word_embeddings).then(|| {
                operations::init_linear(
                    &LinearConfig::new(hidden_size, vocabulary_size)
                        .with_bias(false)
                        .with_layout(LinearLayout::Col),
                    device,
                )
            }),
        })
    }

    /// Evaluates stateless Qwen inference with selective vocabulary projection.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Qwen attention probability capture requires the reference attention path".into(),
            ));
        }
        let (hidden, hidden_states) = self.model.forward(&input, options.hidden_states)?;
        self.finish_forward(input, options, hidden, hidden_states)
    }

    /// Lists the stateless backbone activations available for experimentation.
    ///
    /// See [`Qwen3_5TextModel::activation_points`] for the meaning of the hybrid
    /// attention and gated MLP boundaries.
    pub fn activation_points(&self) -> Vec<PointInfo> {
        self.model.activation_points()
    }

    /// Captures or transforms selected tensors during a stateless forward pass.
    ///
    /// Hook replacements affect the returned logits and hidden states. The
    /// optional MTP graph and cached decoding are separate execution paths and
    /// do not invoke these hooks. Unknown points fail before model evaluation.
    pub fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        hooks.run(self.activation_points(), |hooks| {
            if options.attentions {
                return Err(Error::BackendCapability(
                    "Qwen attention probability capture requires the reference attention path"
                        .into(),
                ));
            }
            let (hidden, hidden_states) =
                self.model
                    .forward_inner(&input, options.hidden_states, Some(hooks))?;
            self.finish_forward(input, options, hidden, hidden_states)
        })
    }

    /// Projects the requested positions after all backbone interventions.
    fn finish_forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hidden: Tensor<B, 3>,
        hidden_states: Vec<Tensor<B, 3>>,
    ) -> Result<CausalLmOutput<B>> {
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
        let logits = self.project_hidden(selected)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
        })
    }

    /// Returns the exact validated checkpoint configuration.
    pub fn config(&self) -> &Qwen3_5TextConfig {
        &self.model.config
    }

    /// Runs the shifted MTP graph and the base model's shared LM head.
    ///
    /// `next_token_ids[b,t]` is paired with the target final hidden state at the
    /// preceding autoregressive position: `x_(t+1)` combines with `h_t` to
    /// predict `x_(t+2)`. Incremental callers must preserve this one-token shift.
    pub fn forward_mtp(
        &self,
        next_token_ids: Tensor<B, 2, Int>,
        target_hidden_state: Tensor<B, 3>,
        position_ids: Tensor<B, 2, Int>,
        step: usize,
    ) -> Result<Qwen3_5MtpPrediction<B>> {
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            Error::configuration(
                "qwen3_5.mtp_num_hidden_layers",
                "model was initialized without the MTP graph",
            )
        })?;
        let [batch, sequence] = next_token_ids.dims();
        if batch == 0 || sequence == 0 {
            return Err(Error::shape(
                "qwen.mtp.next_token_ids",
                "non-empty [batch, sequence]",
                format!("[{batch}, {sequence}]"),
            ));
        }
        let model_device = self.model.embed_tokens.weight.lazy_device();
        ensure_model_device(
            "qwen.mtp.next_token_ids",
            &next_token_ids.device(),
            &model_device,
        )?;
        ensure_model_device(
            "qwen.mtp.target_hidden_state",
            &target_hidden_state.device(),
            &model_device,
        )?;
        ensure_model_device(
            "qwen.mtp.position_ids",
            &position_ids.device(),
            &model_device,
        )?;
        validate_token_ids(next_token_ids.clone(), self.config().vocab_size)?;
        decode::validate_position_range(
            position_ids.clone(),
            self.config().max_position_embeddings,
        )?;
        let context = decode::evaluated_context(position_ids.clone())?;
        let embedding = self.model.embed_tokens.forward(next_token_ids);
        let output = mtp.forward(
            embedding,
            target_hidden_state,
            position_ids,
            step,
            self.model.rope,
            context,
        )?;
        let logits = self.project_hidden(output.hidden_state.clone())?;
        Ok(Qwen3_5MtpPrediction {
            logits,
            hidden_state: output.hidden_state,
        })
    }

    /// Projects normalized hidden states with the shared base-model LM head.
    pub(crate) fn project_hidden(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let hidden_device = hidden.device();
        match &self.lm_head {
            Some(head) => {
                ensure_model_device("qwen.lm_head", &head.weight.lazy_device(), &hidden_device)?;
                operations::linear(hidden, head)
            }
            None => {
                ensure_model_device(
                    "qwen.model.embed_tokens",
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

/// Translates Hugging Face RoPE metadata to the shared validated implementation.
pub(super) fn qwen_rope_config(config: &Qwen3_5TextConfig) -> Result<RopeConfig> {
    let parameters = config.rope_parameters.clone().unwrap_or_default();
    let theta = parameters
        .rope_theta
        .or(config.rope_theta)
        .unwrap_or(10_000.0);
    let partial = parameters
        .partial_rotary_factor
        .or(config.partial_rotary_factor)
        .unwrap_or(1.0);
    let scaling = match parameters.rope_type {
        RopeType::Default => RopeScaling::None,
        RopeType::Linear => LinearRopeScaling::new(parameters.factor.unwrap_or(1.0))
            .map(RopeScaling::from)
            .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?,
        RopeType::Dynamic => DynamicNtkScaling::new(
            parameters.factor.unwrap_or(1.0),
            parameters
                .original_max_position_embeddings
                .unwrap_or(config.max_position_embeddings),
        )
        .map(RopeScaling::from)
        .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?,
        RopeType::Yarn => {
            let mut yarn = YarnConfig::new(
                parameters.factor.unwrap_or(1.0),
                parameters
                    .original_max_position_embeddings
                    .unwrap_or(config.max_position_embeddings),
            )
            .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?;
            if let Some(factor) = parameters.attention_factor {
                yarn = yarn
                    .with_attention_factor(factor)
                    .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?;
            }
            yarn = yarn
                .with_beta_overrides(parameters.beta_fast, parameters.beta_slow)
                .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?;
            RopeScaling::Yarn(yarn)
        }
        unsupported => {
            return Err(Error::configuration(
                "qwen3_5.rope_type",
                format!("{unsupported:?} is not implemented for the Qwen text runtime"),
            ));
        }
    };
    RopeConfig::from_partial_factor(config.head_dim, partial, theta, scaling)
        .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))
}

/// Rejects stateless input tensors that do not share the model parameter device.
///
/// This preflight precedes reductions, positional arithmetic, and embedding
/// lookup so mixed-device user input remains a recoverable library error.
fn validate_input_devices<B: Backend>(
    input: &CausalLmInput<B>,
    model_device: &B::Device,
) -> Result<()> {
    ensure_model_device("qwen.input_ids", &input.token_ids.device(), model_device)?;
    if let Some(mask) = &input.token_mask {
        ensure_model_device("qwen.token_mask", &mask.as_tensor().device(), model_device)?;
    }
    if let Some(positions) = &input.position_ids {
        ensure_model_device("qwen.position_ids", &positions.device(), model_device)?;
    }
    Ok(())
}

/// Reports a stateless tensor/parameter device mismatch through the error taxonomy.
fn ensure_model_device<D: PartialEq>(tensor: &str, actual: &D, expected: &D) -> Result<()> {
    if actual != expected {
        return Err(Error::BackendCapability(format!(
            "{tensor} and Qwen3.5 model parameters are on different devices"
        )));
    }
    Ok(())
}

/// Check public latent boundaries before mixed-dtype operations reach the backend.
fn ensure_model_dtype(tensor: &str, actual: DType, expected: DType) -> Result<()> {
    if actual != expected {
        return Err(Error::BackendCapability(format!(
            "{tensor} dtype must match Qwen3.5 parameters: expected {expected:?}, found {actual:?}"
        )));
    }
    Ok(())
}

/// Validates token values before the embedding backend sees an index.
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
            "qwen.input_ids",
            format!("token values in 0..{vocabulary}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Validates a semantic mask and rejects rows with no usable token.
fn validate_mask<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
) -> Result<()> {
    let Some(mask) = mask else {
        return Ok(());
    };
    if mask.as_tensor().dims() != [batch, sequence] {
        return Err(Error::shape(
            "qwen.token_mask",
            format!("[{batch}, {sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    let minimum_valid = mask
        .as_tensor()
        .clone()
        .int()
        .sum_dim(1)
        .min()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to validate mask: {error}")))?
        .elem::<i64>();
    if minimum_valid == 0 {
        return Err(Error::shape(
            "qwen.token_mask",
            "at least one valid token in every batch row",
            "a fully masked row",
        ));
    }
    Ok(())
}

/// Resolves explicit positions or derives padding-independent text positions.
fn resolve_positions<B: Backend>(
    input: &CausalLmInput<B>,
    batch: usize,
    sequence: usize,
) -> Result<Tensor<B, 2, Int>> {
    if let Some(positions) = &input.position_ids {
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "qwen.position_ids",
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

/// Selects requested sequence positions before the expensive vocabulary map.
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
            let device = hidden.device();
            let indices = positions
                .iter()
                .map(|position| *position as i64)
                .collect::<Vec<_>>();
            let count = indices.len();
            Ok(hidden.select(
                1,
                Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [count]), &device),
            ))
        }
    }
}

/// Validates selective-logit indices before potentially stateful model work.
fn validate_logits_selection(selection: &LogitsSelection, sequence: usize) -> Result<()> {
    if let LogitsSelection::Positions(positions) = selection
        && (positions.is_empty() || positions.iter().any(|position| *position >= sequence))
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
#[path = "model_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "intervention_tests.rs"]
mod intervention_tests;

#[cfg(test)]
#[path = "execution_tests.rs"]
mod execution_tests;
