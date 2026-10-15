//! Stateless GPT-NeoX execution using shared transformer mathematics.
//!
//! Residuals use `[batch, token, hidden]`; attention projections use
//! `[batch, head, token, head_channel]`. Each block uses two affine LayerNorms,
//! scaled dot-product attention, and a dense exact-GELU MLP. In parallel mode
//! `x_next = MLP(LN_m(x)) + Attention(LN_a(x)) + x`, preserving source addition
//! order; sequential mode feeds the attention-updated residual into `LN_m`.
//!
//! RoPE rotates an even prefix of each actual head width using physical positions.
//! Padding removes keys but retains query computations; empty-valid-token rows are
//! rejected. Causal and bidirectional entry points share this padding contract.
//! Eager attention requests F32 softmax; parameter and residual dtype/device must
//! agree. Source dropout and decode caches are absent from this explicit graph.
//! The untied vocabulary projection follows final normalization and can be called
//! independently for logit-lens or trainable tuned-lens experiments.

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
    feed_forward::{DenseFeedForward, DenseFeedForwardConfig, apply_activation},
    intervention::{Hooks, Layer, PointInfo, Stack},
    operations,
    position::{RopeConfig, RopeScaling, RotaryAngles, apply_rotary, rotary_angles},
};

use super::GptNeoXConfig;
use crate::models::shared::{
    ImplicitPositions, InputContract, InputNames, select_hidden, validate_input,
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
        let mlp_pre = hooks.apply(
            layer.mlp_pre_activation(),
            operations::linear(mlp_input, &self.mlp.up_proj)?,
        )?;
        let mlp_activation = hooks.apply(
            layer.mlp_activation(),
            apply_activation(mlp_pre, self.mlp.activation),
        )?;
        let mlp = hooks.apply(
            layer.mlp_output(),
            operations::linear(mlp_activation, &self.mlp.down_proj)?,
        )?;
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
/// Use [`Self::forward_hidden_with_hooks`] for task heads, and [`Self::project_hidden`]
/// to decode a raw intermediate residual state through the final normalization.
/// Parameter initialization is random; [`Self::load_checkpoint`] initializes from
/// explicitly supplied local SafeTensors artifacts.
#[derive(Module, Debug)]
pub struct GptNeoXModel<B: Backend> {
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

impl<B: Backend> GptNeoXModel<B> {
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

    /// Lists semantic activation points with their axes. Q/K points follow RoPE;
    /// MLP activation is exact GELU before the output projection.
    pub fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Decoder.embedding().info()];
        for index in 0..self.layers.len() {
            let layer = Layer::decoder(index);
            points.extend([
                layer.residual_pre().info(),
                layer.attention_input().info(),
                layer.attention_output().info(),
                layer.mlp_input().info(),
                layer.mlp_pre_activation().info(),
                layer.mlp_activation().info(),
                layer.mlp_output().info(),
                layer.residual_post().info(),
            ]);
            let attention = layer.self_attention();
            points.extend([
                attention.query().info(),
                attention.key().info(),
                attention.value().info(),
                attention.head_context().info(),
            ]);
            if self.layers[index].attention.precision_mode
                == AttentionPrecisionMode::EagerFp32Softmax
            {
                points.extend([attention.scores().info(), attention.pattern().info()]);
            }
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
    /// Hidden-state diagnostics follow Hugging Face: embedding, outputs of all
    /// blocks except the last, and the final normalized state.
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
    fn forward_direction(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        causal: bool,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        hooks.run(self.activation_points(), |hooks| {
            if options.attentions || options.router_logits || options.auxiliary_losses {
                return Err(Error::configuration("gpt_neox.forward_options", "use attention pattern hooks for probabilities; this dense model has no router diagnostics"));
            }
            let (hidden, hidden_states) = self.backbone(&input, options.hidden_states, causal, hooks)?;
            let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
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
        validate_norm(
            &self.final_layer_norm,
            self.config.hidden_size,
            &parameter.device(),
            parameter.dtype(),
            "gpt_neox.final_layer_norm",
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
    /// Each layer derives RoPE from its current head width; retained states follow the source
    /// convention of pre-block residuals followed by the final normalized output.
    fn backbone(
        &self,
        input: &CausalLmInput<B>,
        retain: bool,
        causal: bool,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>)> {
        self.validate_parameters()?;
        let positions =
            validate_public_input(input, &self.config, &self.embed_in.weight.lazy_device())?;
        let mut hidden = hooks.apply(
            Stack::Decoder.embedding(),
            self.embed_in.forward(input.token_ids.clone()),
        )?;
        let mut states = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            if retain {
                states.push(hidden.clone());
            }
            // Publicly replacing a layer's attention can change its head width.
            // Keep the configured rotary fraction relative to that actual width.
            let rope = RopeConfig::from_partial_factor(
                layer.attention.head_dim,
                self.config.rotary_pct,
                self.config.rotary_emb_base,
                RopeScaling::None,
            )
            .map_err(|error| {
                Error::configuration(format!("gpt_neox.layers.{index}.rotary"), error.to_string())
            })?;
            let angles =
                rotary_angles(rope, positions.clone(), self.config.max_position_embeddings)?;
            hidden = layer.forward_scoped(
                hidden,
                &angles,
                input.token_mask.as_ref(),
                self.config.use_parallel_residual,
                causal,
                Layer::decoder(index),
                hooks,
            )?;
        }
        hidden = hooks.apply(Stack::Decoder.final_norm(), self.normalize_hidden(hidden)?)?;
        if retain {
            states.push(hidden.clone());
        }
        Ok((hidden, states))
    }

    /// Checks edited public modules before unchecked backend reshapes or residual additions.
    fn validate_parameters(&self) -> Result<()> {
        let embedding = self.embed_in.weight.val();
        let device = embedding.device();
        let dtype = embedding.dtype();
        let width = self.config.hidden_size;
        validate_tensor(
            &embedding,
            [self.config.vocab_size, width],
            &device,
            dtype,
            "gpt_neox.embed_in.weight",
        )?;
        validate_norm(
            &self.final_layer_norm,
            width,
            &device,
            dtype,
            "gpt_neox.final_layer_norm",
        )?;
        validate_linear(
            &self.embed_out,
            width,
            self.config.vocab_size,
            &device,
            dtype,
            "gpt_neox.embed_out",
        )?;
        for (index, layer) in self.layers.iter().enumerate() {
            let label = format!("gpt_neox.layers.{index}");
            validate_norm(
                &layer.input_layernorm,
                width,
                &device,
                dtype,
                &format!("{label}.input_layernorm"),
            )?;
            validate_norm(
                &layer.post_attention_layernorm,
                width,
                &device,
                dtype,
                &format!("{label}.post_attention_layernorm"),
            )?;
            let attention = &layer.attention;
            if attention.hidden_size != width
                || attention.query_heads == 0
                || attention.key_value_heads == 0
                || attention.head_dim == 0
                || !attention
                    .query_heads
                    .is_multiple_of(attention.key_value_heads)
            {
                return Err(Error::configuration(
                    format!("{label}.attention"),
                    "head dimensions must be nonzero, compatible, and preserve residual width",
                ));
            }
            let (query_width, key_width) = attention
                .query_heads
                .checked_mul(attention.head_dim)
                .zip(attention.key_value_heads.checked_mul(attention.head_dim))
                .ok_or_else(|| {
                    Error::configuration(
                        format!("{label}.attention"),
                        "attention projection widths overflow usize",
                    )
                })?;
            let q_projection_width = query_width
                .checked_mul(if attention.output_gate == AttentionOutputGate::None {
                    1
                } else {
                    2
                })
                .ok_or_else(|| Error::configuration(&label, "gated query width overflows usize"))?;
            validate_linear(
                &attention.q_proj,
                width,
                q_projection_width,
                &device,
                dtype,
                &format!("{label}.attention.q_proj"),
            )?;
            validate_linear(
                &attention.k_proj,
                width,
                key_width,
                &device,
                dtype,
                &format!("{label}.attention.k_proj"),
            )?;
            validate_linear(
                &attention.v_proj,
                width,
                key_width,
                &device,
                dtype,
                &format!("{label}.attention.v_proj"),
            )?;
            validate_linear(
                &attention.o_proj,
                query_width,
                width,
                &device,
                dtype,
                &format!("{label}.attention.o_proj"),
            )?;
            for norm in [&attention.query_norm, &attention.key_norm]
                .into_iter()
                .flatten()
            {
                validate_tensor(
                    &norm.weight.val(),
                    [attention.head_dim],
                    &device,
                    dtype,
                    &format!("{label}.attention.query_key_norm"),
                )?;
            }
            let intermediate = layer.mlp.up_proj.weight.val().dims()[1];
            if intermediate == 0 {
                return Err(Error::configuration(
                    format!("{label}.mlp"),
                    "intermediate width must be nonzero",
                ));
            }
            validate_linear(
                &layer.mlp.up_proj,
                width,
                intermediate,
                &device,
                dtype,
                &format!("{label}.mlp.up_proj"),
            )?;
            validate_linear(
                &layer.mlp.down_proj,
                intermediate,
                width,
                &device,
                dtype,
                &format!("{label}.mlp.down_proj"),
            )?;
        }
        Ok(())
    }
}

// Public-edit safety checks, initialization, and input/output boundaries.

/// Checks an exact parameter shape and common dtype/device before backend arithmetic.
fn validate_tensor<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    shape: [usize; D],
    device: &B::Device,
    dtype: DType,
    label: &str,
) -> Result<()> {
    if tensor.dims() != shape {
        return Err(Error::shape(
            label,
            format!("{shape:?}"),
            format!("{:?}", tensor.dims()),
        ));
    }
    if tensor.device() != *device || tensor.dtype() != dtype {
        return Err(Error::configuration(
            label,
            "parameters must share device and dtype",
        ));
    }
    Ok(())
}

/// Checks channelwise affine LayerNorm parameters against the residual width and storage.
fn validate_norm<B: Backend>(
    norm: &LayerNorm<B>,
    width: usize,
    device: &B::Device,
    dtype: DType,
    label: &str,
) -> Result<()> {
    validate_tensor(&norm.gamma.val(), [width], device, dtype, label)?;
    if let Some(beta) = &norm.beta {
        validate_tensor(&beta.val(), [width], device, dtype, label)?;
    }
    Ok(())
}

/// Checks runtime `[input, output]` weights and optional `[output]` bias storage.
fn validate_linear<B: Backend>(
    linear: &Linear<B>,
    input: usize,
    output: usize,
    device: &B::Device,
    dtype: DType,
    label: &str,
) -> Result<()> {
    validate_tensor(&linear.weight.val(), [input, output], device, dtype, label)?;
    if let Some(bias) = &linear.bias {
        validate_tensor(&bias.val(), [output], device, dtype, label)?;
    }
    Ok(())
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
/// values share one device-to-host readback.
fn validate_public_input<B: Backend>(
    input: &CausalLmInput<B>,
    config: &GptNeoXConfig,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
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
    )?
    .require_positions(INPUT_NAMES.position_ids)
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
