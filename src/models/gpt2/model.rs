use burn::{
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, LinearLayout,
    },
    tensor::{ElementConversion, Int, Tensor, backend::Backend},
};

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, LogitsSelection, Result,
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
        validate_mask(mask, hidden.dims()[0], hidden.dims()[1], &hidden.device())?;
        self.forward_inner(hidden, mask, 0, None)
    }

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
        ensure_device("gpt2.token_ids", &input.token_ids.device(), &device)?;
        validate_indices(
            "gpt2.token_ids",
            input.token_ids.clone(),
            self.config.vocab_size,
        )?;
        validate_mask(input.token_mask.as_ref(), batch, sequence, &device)?;
        let positions = match &input.position_ids {
            Some(positions) => {
                if positions.dims() != [batch, sequence] {
                    return Err(Error::shape(
                        "gpt2.position_ids",
                        format!("[{batch}, {sequence}]"),
                        format!("{:?}", positions.dims()),
                    ));
                }
                ensure_device("gpt2.position_ids", &positions.device(), &device)?;
                validate_indices(
                    "gpt2.position_ids",
                    positions.clone(),
                    self.config.n_positions,
                )?;
                positions.clone()
            }
            None => Tensor::<B, 1, Int>::arange(0..sequence as i64, &device)
                .unsqueeze_dim(0)
                .repeat_dim(0, batch),
        };
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

fn validate_options(options: &ForwardOptions) -> Result<()> {
    if options.attentions {
        return Err(Error::BackendCapability(
            "GPT-2 attention probabilities are available through self_attention().pattern() hooks"
                .into(),
        ));
    }
    Ok(())
}

fn ensure_device<D: PartialEq>(name: &str, actual: &D, expected: &D) -> Result<()> {
    if actual != expected {
        return Err(Error::configuration(
            name,
            "tensor and parameters must share a device",
        ));
    }
    Ok(())
}

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
    ensure_device("gpt2.hidden", &hidden.device(), device)
}

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
        ensure_device(name, &parameter.device(), &hidden.device())?;
        if parameter.dtype() != hidden.dtype() {
            return Err(Error::configuration(
                name,
                "LayerNorm parameters and activations must share a floating dtype",
            ));
        }
    }
    Ok(())
}

fn validate_indices<B: Backend>(name: &str, ids: Tensor<B, 2, Int>, limit: usize) -> Result<()> {
    let minimum = ids
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("{name} range read: {error}")))?
        .elem::<i64>();
    let maximum = ids
        .max()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("{name} range read: {error}")))?
        .elem::<i64>();
    if minimum < 0 || maximum >= limit as i64 {
        return Err(Error::configuration(
            name,
            format!("expected indices in 0..{limit}, observed {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

fn validate_mask<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
    device: &B::Device,
) -> Result<()> {
    if let Some(mask) = mask {
        if mask.as_tensor().dims() != [batch, sequence] {
            return Err(Error::shape(
                "gpt2.token_mask",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", mask.as_tensor().dims()),
            ));
        }
        ensure_device("gpt2.token_mask", &mask.as_tensor().device(), device)?;
        let least = mask
            .as_tensor()
            .clone()
            .int()
            .sum_dim(1)
            .min()
            .try_into_scalar()
            .map_err(|error| Error::BackendCapability(format!("gpt2 mask read: {error}")))?
            .elem::<i64>();
        if least == 0 {
            return Err(Error::configuration(
                "gpt2.token_mask",
                "each batch row needs a valid token",
            ));
        }
    }
    Ok(())
}

fn select_hidden<B: Backend>(
    hidden: Tensor<B, 3>,
    mask: Option<&TokenMask<B>>,
    selection: &LogitsSelection,
) -> Result<Tensor<B, 3>> {
    let [batch, sequence, _] = hidden.dims();
    match selection {
        LogitsSelection::All => Ok(hidden),
        LogitsSelection::Positions(indices) => {
            if indices.is_empty() || indices.iter().any(|index| *index >= sequence) {
                return Err(Error::configuration(
                    "gpt2.logits",
                    "positions must be nonempty and within the sequence",
                ));
            }
            Ok(Tensor::cat(
                indices
                    .iter()
                    .map(|index| hidden.clone().narrow(1, *index, 1))
                    .collect(),
                1,
            ))
        }
        LogitsSelection::Last => match mask {
            None => Ok(hidden.narrow(1, sequence - 1, 1)),
            Some(mask) => {
                let mask = crate::architecture::tensor_bool_values(
                    mask.as_tensor().clone(),
                    "gpt2 token mask",
                )?;
                let mut rows = Vec::with_capacity(batch);
                for (row, valid) in mask.chunks(sequence).enumerate() {
                    let last = valid
                        .iter()
                        .rposition(|value| *value)
                        .ok_or_else(|| Error::configuration("gpt2.token_mask", "empty row"))?;
                    rows.push(hidden.clone().narrow(0, row, 1).narrow(1, last, 1));
                }
                Ok(Tensor::cat(rows, 0))
            }
        },
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
