//! GPT-2 stateless blocks, explicit activation boundaries, and tied vocabulary readout.
//!
//! # Equations and axes
//!
//! Embeddings add learned token and absolute-position vectors in `[batch, token, d]`.
//! Each block computes `y = x + Attention(LN_a(x))`, then
//! `z = y + GELU(LN_m(y) W_up + b_up) W_down + b_down`.
//! Affine LayerNorm uses the channel mean and variance with configured epsilon.
//! Attention exposes `[batch, head, query, key]` scores and
//! `[batch, head, token, head_channel]` projections. The configured score scale is
//! optionally `1 / sqrt(head_channel)` and optionally also `1 / (layer + 1)`.
//!
//! # Precision and visibility
//!
//! Shared eager attention requests F32 softmax; weights and residuals retain their
//! backend dtype. Configuration chooses exact or tanh GELU explicitly. Causality
//! and padding are independent: padding excludes keys, while padded queries still
//! execute. Every batch row needs a valid token; physical positions include padding.
//! There is no decode cache or implicit dropout. The final readout is `LN(z) E^T`,
//! reusing the single registered token embedding `E` without a second weight owner.

use burn::{
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, LinearLayout,
    },
    tensor::{Tensor, backend::Backend},
};

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::TokenMask,
    attention::{
        AttentionPrecisionMode, GroupedQueryAttention, GroupedQueryAttentionConfig,
        padding_attention_mask,
    },
    feed_forward::{DenseFeedForward, apply_activation},
    intervention::{Hooks, Layer, PointInfo, Stack},
    operations,
};

use super::Gpt2Config;
use crate::models::shared::{
    ImplicitPositions, InputContract, InputNames, ensure_same_device, select_hidden,
    validate_input, validate_nonempty_mask,
};

/// Tensor names reported by GPT-2 input errors.
const INPUT_NAMES: InputNames<'static> = InputNames {
    token_ids: "gpt2.token_ids",
    token_mask: "gpt2.token_mask",
    position_ids: "gpt2.position_ids",
};

/// Owner named when a tensor is on another device than the weights.
const PARAMETERS: &str = "GPT-2 parameters";

// Editable block equations and intermediate activation boundaries.

/// One GPT-2 block: `x + attention(LN(x))`, then `x + MLP(LN(x))`.
#[derive(Module, Debug)]
pub struct Gpt2Block<B: Backend> {
    /// Affine LayerNorm before attention.
    pub attention_norm: LayerNorm<B>,
    /// Shared attention mechanics with independent, biased Q/K/V projections.
    pub attention: GroupedQueryAttention<B>,
    /// Affine LayerNorm before the MLP.
    pub mlp_norm: LayerNorm<B>,
    /// Dense nonlinear intermediate processing, including both projection biases.
    pub mlp: DenseFeedForward<B>,
    /// Whether this block excludes future keys; may be changed for experiments.
    pub causal: bool,
}

/// Creates checkpoint-native `[input, output]` GPT-2 Conv1D projections.
fn linear<B: Backend>(input: usize, output: usize, device: &B::Device) -> Linear<B> {
    operations::init_linear(
        &LinearConfig::new(input, output)
            .with_bias(true)
            .with_layout(LinearLayout::Row),
        device,
    )
}

impl<B: Backend> Gpt2Block<B> {
    /// Builds a randomly initialized block with its zero-based attention scaling index.
    pub fn init(config: &Gpt2Config, index: usize, device: &B::Device) -> Result<Self> {
        config.validate()?;
        let mut attention_config = GroupedQueryAttentionConfig::new(
            config.n_embd,
            config.n_head,
            config.n_head,
            config.n_embd / config.n_head,
        );
        attention_config.bias = true;
        let scale = if config.scale_attn_weights {
            ((config.n_embd / config.n_head) as f64).sqrt().recip()
        } else {
            1.0
        };
        attention_config.scale = Some(if config.scale_attn_by_inverse_layer_idx {
            let denominator = index
                .checked_add(1)
                .ok_or_else(|| Error::configuration("gpt2.layer", "index overflows"))?;
            scale / denominator as f64
        } else {
            scale
        });
        attention_config.precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
        let mut attention = attention_config.init(device)?;
        // HF Conv1D is already [input, output]. Row layout preserves that
        // parameter/record convention; importing must not transpose it again.
        attention.q_proj = linear(config.n_embd, config.n_embd, device);
        attention.k_proj = linear(config.n_embd, config.n_embd, device);
        attention.v_proj = linear(config.n_embd, config.n_embd, device);
        attention.o_proj = linear(config.n_embd, config.n_embd, device);
        let norm = || {
            LayerNormConfig::new(config.n_embd)
                .with_epsilon(config.layer_norm_epsilon)
                .with_bias(true)
                .init(device)
        };
        Ok(Self {
            attention_norm: norm(),
            attention,
            mlp_norm: norm(),
            mlp: DenseFeedForward {
                up_proj: linear(config.n_embd, config.intermediate_size()?, device),
                down_proj: linear(config.intermediate_size()?, config.n_embd, device),
                activation: config.activation_function.activation(),
            },
            causal: config.causal,
        })
    }

    /// Runs this block on `[batch, token, channel]` states, without embedding or readout.
    /// Padding excludes keys; padded query positions are still evaluated as in GPT-2.
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        mask: Option<&TokenMask<B>>,
    ) -> Result<Tensor<B, 3>> {
        let [batch, sequence, _] = hidden.dims();
        if let Some(mask) = mask {
            ensure_same_device(
                INPUT_NAMES.token_mask,
                &mask.as_tensor().device(),
                &hidden.device(),
                "the GPT-2 hidden states",
            )?;
        }
        validate_nonempty_mask(mask, batch, sequence, INPUT_NAMES.token_mask)?;
        self.forward_inner(hidden, mask, 0, None)
    }

    /// Runs validated pre-norm attention and MLP residual updates with optional ordered hooks.
    ///
    /// Attention padding masks keys only; the causal flag independently limits future keys.
    fn forward_inner(
        &self,
        hidden: Tensor<B, 3>,
        mask: Option<&TokenMask<B>>,
        index: usize,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<Tensor<B, 3>> {
        validate_hidden(
            &hidden,
            self.attention.hidden_size,
            &self.attention_norm.gamma.lazy_device(),
        )?;
        validate_norm(&hidden, &self.attention_norm, "gpt2.attention_norm")?;
        validate_norm(&hidden, &self.mlp_norm, "gpt2.mlp_norm")?;
        let layer = Layer::decoder(index);
        let hidden = emit(&mut hooks, layer.residual_pre(), hidden)?;
        let residual = hidden.clone();
        let normalized = emit(
            &mut hooks,
            layer.attention_input(),
            self.attention_norm.forward(hidden),
        )?;
        let projection = self.attention.project_self(normalized)?;
        let [batch, _, sequence, _] = projection.query.dims();
        let mask = padding_attention_mask(
            None,
            mask,
            batch,
            self.attention.query_heads,
            sequence,
            sequence,
        )?;
        let mixed = match hooks.as_deref_mut() {
            Some(hooks) => self.attention.attend_with_hooks(
                projection,
                mask,
                None,
                self.causal,
                layer.self_attention(),
                hooks,
            )?,
            None => self.attention.attend(projection, mask, None, self.causal)?,
        };
        let hidden = residual + emit(&mut hooks, layer.attention_output(), mixed)?;
        let residual = hidden.clone();
        let normalized = emit(&mut hooks, layer.mlp_input(), self.mlp_norm.forward(hidden))?;
        let mixed = if hooks.is_some() {
            let projected = emit(
                &mut hooks,
                layer.mlp_pre_activation(),
                operations::linear(normalized, &self.mlp.up_proj)?,
            )?;
            let activated = emit(
                &mut hooks,
                layer.mlp_activation(),
                apply_activation(projected, self.mlp.activation),
            )?;
            operations::linear(activated, &self.mlp.down_proj)?
        } else {
            self.mlp.forward(normalized)?
        };
        let hidden = residual + emit(&mut hooks, layer.mlp_output(), mixed)?;
        emit(&mut hooks, layer.residual_post(), hidden)
    }
}

// Embedding/backbone composition and tied vocabulary readout.

/// Hidden representations returned independently of any task head.
#[derive(Clone, Debug)]
pub struct Gpt2ModelOutput<B: Backend> {
    /// Final normalized `[batch, token, channel]` states.
    pub last_hidden_state: Tensor<B, 3>,
    /// When requested: embedding output, each block output, then final normalized states.
    pub hidden_states: Vec<Tensor<B, 3>>,
}

/// GPT-2 token/position embeddings and independently accessible transformer blocks.
#[derive(Module, Debug)]
pub struct Gpt2Model<B: Backend> {
    /// Shared token embedding; the language-model readout reuses this storage.
    pub token_embedding: Embedding<B>,
    /// Learned absolute-position embeddings, added before the first block.
    pub position_embedding: Embedding<B>,
    /// Public blocks permit prefix execution and ordinary module composition.
    pub layers: Vec<Gpt2Block<B>>,
    /// Final affine LayerNorm, applied after the last block.
    pub final_norm: LayerNorm<B>,
    /// Original graph configuration; runtime fields are public for experimentation.
    #[module(skip)]
    pub config: Gpt2Config,
}

impl<B: Backend> Gpt2Model<B> {
    /// Constructs random parameters; loading pretrained values is a separate operation.
    pub fn init(config: Gpt2Config, device: &B::Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            token_embedding: EmbeddingConfig::new(config.vocab_size, config.n_embd).init(device),
            position_embedding: EmbeddingConfig::new(config.n_positions, config.n_embd)
                .init(device),
            layers: (0..config.n_layer)
                .map(|index| Gpt2Block::init(&config, index, device))
                .collect::<Result<_>>()?,
            final_norm: LayerNormConfig::new(config.n_embd)
                .with_epsilon(config.layer_norm_epsilon)
                .with_bias(true)
                .init(device),
            config,
        })
    }

    /// Available points in execution order; the embedding point includes absolute positions.
    /// Attention points expose Q/K/V before head repetition and contexts before projection.
    /// Scores/patterns require the block's explicit eager attention precision mode.
    pub fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Decoder.embedding().info()];
        for (index, block) in self.layers.iter().enumerate() {
            let layer = Layer::decoder(index);
            points.extend([layer.residual_pre().info(), layer.attention_input().info()]);
            let attention = layer.self_attention();
            points.extend([
                attention.query().info(),
                attention.key().info(),
                attention.value().info(),
            ]);
            if block.attention.precision_mode == AttentionPrecisionMode::EagerFp32Softmax {
                points.extend([attention.scores().info(), attention.pattern().info()]);
            }
            points.push(attention.head_context().info());
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

    /// Evaluates embeddings and blocks, without projecting into the vocabulary.
    /// Default positions are physical `0..sequence`, including padding, as in HF GPT-2.
    /// Supply explicit position IDs to choose a different convention.
    pub fn forward(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
    ) -> Result<Gpt2ModelOutput<B>> {
        self.forward_inner(input, retain_hidden_states, None)
    }

    /// Captures or modifies selected activations in one validated stateless run.
    pub fn forward_with_hooks(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Gpt2ModelOutput<B>> {
        hooks.run(self.activation_points(), |hooks| {
            self.forward_inner(input, retain_hidden_states, Some(hooks))
        })
    }

    /// Validates IDs and masks, adds token and absolute-position embeddings, then runs all blocks.
    ///
    /// Retained states include the embedding, every block output, and final affine normalization.
    fn forward_inner(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<Gpt2ModelOutput<B>> {
        let [batch, sequence] = input.token_ids.dims();
        let device = self.token_embedding.weight.lazy_device();
        if batch == 0 || sequence == 0 || sequence > self.config.n_positions {
            return Err(Error::shape(
                "gpt2.token_ids",
                "nonempty batch and sequence within position capacity",
                format!("{:?}", input.token_ids.dims()),
            ));
        }
        // Learned absolute positions number physical columns, including padding.
        let positions = validate_input(
            input,
            &device,
            &InputContract::new(INPUT_NAMES, PARAMETERS, self.config.vocab_size)
                .with_position_limit(self.config.n_positions)
                .with_implicit_positions(ImplicitPositions::PhysicalColumns),
        )?
        .require_positions(INPUT_NAMES.position_ids)?;
        let mut hidden = self.token_embedding.forward(input.token_ids.clone())
            + self.position_embedding.forward(positions);
        hidden = emit(&mut hooks, Stack::Decoder.embedding(), hidden)?;
        let mut states = Vec::new();
        if retain_hidden_states {
            states.push(hidden.clone());
        }
        for (index, block) in self.layers.iter().enumerate() {
            hidden = block.forward_inner(
                hidden,
                input.token_mask.as_ref(),
                index,
                hooks.as_deref_mut(),
            )?;
            if retain_hidden_states {
                states.push(hidden.clone());
            }
        }
        hidden = emit(
            &mut hooks,
            Stack::Decoder.final_norm(),
            self.normalize(hidden)?,
        )?;
        if retain_hidden_states {
            states.push(hidden.clone());
        }
        Ok(Gpt2ModelOutput {
            last_hidden_state: hidden,
            hidden_states: states,
        })
    }

    /// Applies the final affine LayerNorm to an existing residual tensor.
    pub fn normalize(&self, residual: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        validate_hidden(
            &residual,
            self.config.n_embd,
            &self.final_norm.gamma.lazy_device(),
        )?;
        validate_norm(&residual, &self.final_norm, "gpt2.final_norm")?;
        Ok(self.final_norm.forward(residual))
    }
}

/// A stateless GPT-2 backbone with its parameter-free tied vocabulary readout.
#[derive(Module, Debug)]
pub struct Gpt2ForCausalLm<B: Backend> {
    /// Replaceable public backbone; no independent LM-head tensor is allocated.
    pub model: Gpt2Model<B>,
}

impl<B: Backend> Gpt2ForCausalLm<B> {
    /// Initializes a random graph; use [`Self::load_checkpoint`] for local weights.
    pub fn init(config: Gpt2Config, device: &B::Device) -> Result<Self> {
        Ok(Self {
            model: Gpt2Model::init(config, device)?,
        })
    }

    /// Lists the same activation points as the public backbone.
    pub fn activation_points(&self) -> Vec<PointInfo> {
        self.model.activation_points()
    }

    /// Projects already normalized states with the tied token embedding.
    pub fn project(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        validate_hidden(
            &hidden,
            self.model.config.n_embd,
            &self.model.token_embedding.weight.lazy_device(),
        )?;
        let [batch, sequence, width] = hidden.dims();
        let embedding = self.model.token_embedding.weight.val();
        if hidden.dtype() != embedding.dtype() {
            return Err(Error::BackendCapability(
                "GPT-2 readout activations and tied embedding must share a floating dtype".into(),
            ));
        }
        Ok(hidden
            .reshape([batch * sequence, width])
            .matmul(embedding.transpose())
            .reshape([batch, sequence, self.model.config.vocab_size]))
    }

    /// Normalizes a final residual then projects it. Useful for direct-effect interventions.
    /// The input is before final LayerNorm; do not pass an already normalized capture.
    pub fn readout(&self, residual: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        self.project(self.model.normalize(residual)?)
    }

    /// Evaluates the normal stateless graph with selected vocabulary positions.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        validate_options(options)?;
        let output = self.model.forward(&input, options.hidden_states)?;
        self.finish(&input, options, output)
    }

    /// Evaluates the actual graph with ordered capture/edit callbacks.
    /// Failed input, callback, or readout operations discard this run's captures.
    pub fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        hooks.run(self.activation_points(), |hooks| {
            validate_options(options)?;
            let output = self
                .model
                .forward_inner(&input, options.hidden_states, Some(hooks))?;
            self.finish(&input, options, output)
        })
    }

    /// Selects requested normalized token positions before the tied vocabulary projection.
    fn finish(
        &self,
        input: &CausalLmInput<B>,
        options: &ForwardOptions,
        output: Gpt2ModelOutput<B>,
    ) -> Result<CausalLmOutput<B>> {
        let selected = select_hidden(
            output.last_hidden_state.clone(),
            input.token_mask.as_ref(),
            &options.logits,
        )?;
        Ok(CausalLmOutput {
            logits: self.project(selected)?,
            final_hidden_state: options.hidden_states.then_some(output.last_hidden_state),
            hidden_states: output.hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
        })
    }
}

// Boundary validation and rank-preserving selection.

/// Applies a rank-preserving activation intervention, or returns the connected tensor unchanged.
fn emit<B: Backend, const D: usize>(
    hooks: &mut Option<&mut Hooks<'_, B>>,
    point: crate::intervention::Point<D>,
    tensor: Tensor<B, D>,
) -> Result<Tensor<B, D>> {
    match hooks {
        Some(hooks) => hooks.apply(point, tensor),
        None => Ok(tensor),
    }
}

/// Rejects attention diagnostics that must instead be requested through pattern hooks.
fn validate_options(options: &ForwardOptions) -> Result<()> {
    if options.attentions {
        return Err(Error::BackendCapability(
            "GPT-2 attention probabilities are available through self_attention().pattern() hooks"
                .into(),
        ));
    }
    Ok(())
}

/// Checks a nonempty `[batch, token, width]` residual and its parameter device.
fn validate_hidden<B: Backend>(
    hidden: &Tensor<B, 3>,
    width: usize,
    device: &B::Device,
) -> Result<()> {
    let [batch, sequence, actual_width] = hidden.dims();
    if batch == 0 || sequence == 0 || actual_width != width {
        return Err(Error::shape(
            "gpt2.hidden",
            format!("[batch>0, token>0, {width}]"),
            format!("{:?}", hidden.dims()),
        ));
    }
    ensure_same_device("gpt2.hidden", &hidden.device(), device, PARAMETERS)
}

/// Checks affine LayerNorm vector widths, floating dtypes, and devices against the residual.
fn validate_norm<B: Backend>(hidden: &Tensor<B, 3>, norm: &LayerNorm<B>, name: &str) -> Result<()> {
    let width = hidden.dims()[2];
    let gamma = norm.gamma.val();
    let beta = norm.beta.as_ref().map(|parameter| parameter.val());
    for parameter in std::iter::once(&gamma).chain(beta.as_ref()) {
        if parameter.dims() != [width] {
            return Err(Error::shape(
                name,
                format!("[{width}]"),
                format!("{:?}", parameter.dims()),
            ));
        }
        ensure_same_device(name, &parameter.device(), &hidden.device(), PARAMETERS)?;
        if parameter.dtype() != hidden.dtype() {
            return Err(Error::configuration(
                name,
                "LayerNorm parameters and activations must share a floating dtype",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
