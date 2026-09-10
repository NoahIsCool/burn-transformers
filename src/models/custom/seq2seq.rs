//! Modular encoder-decoder transformer assembly.
//!
//! Source blocks use bidirectional self-attention. Target blocks use causal self-attention,
//! then an independently parameterized cross-attention branch, then their configured dense or
//! routed feed-forward branch. Every cross-attention block uses the decoder layer's mixer-side
//! normalization convention because [`Seq2SeqSpec`] intentionally has no separate cross-norm
//! field.

use std::ops::Range;

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{ElementConversion, Tensor, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLmInput, DecodeInput, EncoderOutput, Error, ForwardOptions, Result,
    Seq2SeqInput, Seq2SeqLanguageModel, Seq2SeqOutput,
    architecture::Seq2SeqSpec,
    attention::AttentionPrecisionMode,
    cache::CacheConfig,
    experiment::{EncoderContext, ExecutionContext, HiddenState, HiddenTransform},
    intervention::{Hooks, Layer, PointInfo, Stack},
};

use super::{
    common::{
        AttentionDirection, BackboneOutput, ModularLayer, ModularNorm, causal_output,
        evaluated_context, has_moe, router_diagnostics, select_hidden, tied_projection,
        validate_and_resolve_input, validate_decoder_execution, validate_forward_options,
        validate_positions,
    },
    state::{ModularDecodeState, StatePurpose},
};

/// A modular bidirectional encoder and causal cross-attending decoder.
#[derive(Module, Debug)]
pub struct ModularSeq2SeqLm<B: Backend> {
    /// Source token embedding table.
    pub encoder_embed_tokens: Embedding<B>,
    /// Target token embedding table.
    pub decoder_embed_tokens: Embedding<B>,
    /// Bidirectional source layers.
    encoder_layers: Vec<ModularLayer<B>>,
    /// Causal target layers with cross-attention branches.
    decoder_layers: Vec<ModularLayer<B>>,
    /// Final source normalization.
    encoder_final_norm: ModularNorm<B>,
    /// Final target normalization.
    decoder_final_norm: ModularNorm<B>,
    /// Independent target vocabulary head, absent when tied to target embeddings.
    pub lm_head: Option<Linear<B>>,
    /// Validated topology excluded from parameter records.
    #[module(skip)]
    spec: Seq2SeqSpec,
}

impl<B: Backend> ModularSeq2SeqLm<B> {
    /// Validates both stacks and allocates the complete encoder-decoder parameter graph.
    pub fn init(spec: Seq2SeqSpec, device: &B::Device) -> Result<Self> {
        validate_seq2seq_execution(&spec)?;
        let encoder_layers = spec
            .encoder
            .layers
            .iter()
            .map(|layer| ModularLayer::init(layer, spec.encoder.hidden_size, None, device))
            .collect::<Result<Vec<_>>>()?;
        let decoder_layers = spec
            .decoder
            .layers
            .iter()
            .map(|layer| {
                ModularLayer::init(
                    layer,
                    spec.decoder.hidden_size,
                    Some((spec.cross_attention_heads, spec.cross_attention_head_dim)),
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            encoder_embed_tokens: EmbeddingConfig::new(
                spec.encoder.vocabulary_size,
                spec.encoder.hidden_size,
            )
            .init(device),
            decoder_embed_tokens: EmbeddingConfig::new(
                spec.decoder.vocabulary_size,
                spec.decoder.hidden_size,
            )
            .init(device),
            encoder_layers,
            decoder_layers,
            encoder_final_norm: ModularNorm::init(
                spec.encoder.final_norm,
                spec.encoder.hidden_size,
                device,
            )?,
            decoder_final_norm: ModularNorm::init(
                spec.decoder.final_norm,
                spec.decoder.hidden_size,
                device,
            )?,
            lm_head: (!spec.decoder.tie_word_embeddings).then(|| {
                crate::operations::init_linear(
                    &LinearConfig::new(spec.decoder.hidden_size, spec.decoder.vocabulary_size)
                        .with_bias(false)
                        .with_layout(LinearLayout::Col),
                    device,
                )
            }),
            spec,
        })
    }

    /// Selects numerical execution for conventional encoder, decoder, and cross-attention branches.
    ///
    /// [`AttentionPrecisionMode::EagerFp32Softmax`] enables score and probability
    /// capture/edit points, listed by [`Self::activation_points`]. This reference path
    /// materializes attention matrices with quadratic sequence memory cost; select
    /// it explicitly for experiments that need those points. Initialization keeps
    /// [`AttentionPrecisionMode::BackendDefault`]. Any decoder DeltaNet mixers
    /// retain their recurrent computation; their cross-attention uses the chosen mode.
    /// MLA self-attention retains its explicit eager graph.
    #[must_use]
    pub fn with_attention_precision(mut self, mode: AttentionPrecisionMode) -> Self {
        for layer in self
            .encoder_layers
            .iter_mut()
            .chain(&mut self.decoder_layers)
        {
            layer.set_attention_precision(mode);
        }
        self
    }

    /// Returns the exact serializable topology used to construct both stacks.
    #[must_use]
    pub const fn spec(&self) -> &Seq2SeqSpec {
        &self.spec
    }

    /// Lists executable activation points from both independently identified stacks.
    ///
    /// Dense MLP pre/activation points expose the gate before/after its nonlinearity,
    /// before multiplying by the content projection. MoE exposes only MLP input/output.
    /// Cross-attention points belong to decoder layers and surround the source-reading
    /// branch. All residual points use `[batch, token, hidden]` axes.
    pub fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = self.encoder_activation_points();
        points.push(Stack::Decoder.embedding().info());
        for (index, layer) in self.decoder_layers.iter().enumerate() {
            points.extend(layer.activation_points(Layer::decoder(index)));
        }
        points.push(Stack::Decoder.final_norm().info());
        points
    }

    /// Lists only points visited by [`Self::encode_with_hooks`].
    pub fn encoder_activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Encoder.embedding().info()];
        for (index, layer) in self.encoder_layers.iter().enumerate() {
            points.extend(layer.activation_points(Layer::encoder(index)));
        }
        points.push(Stack::Encoder.final_norm().info());
        points
    }

    /// Borrows independently executable source blocks.
    pub fn encoder_layers(&self) -> &[ModularLayer<B>] {
        &self.encoder_layers
    }

    /// Borrows independently executable target blocks.
    pub fn decoder_layers(&self) -> &[ModularLayer<B>] {
        &self.decoder_layers
    }

    /// Embeds source tokens before encoder block zero, retaining positions and padding.
    pub fn embed_encoder(&self, input: CausalLmInput<B>) -> Result<HiddenState<B>> {
        self.embed_encoder_with_hooks(input, &mut Hooks::new())
    }

    /// Embeds source tokens in a scope containing the encoder embedding point.
    pub fn embed_encoder_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.embed_stack(input, Stack::Encoder, None, hooks)
    }

    /// Embeds target tokens with an explicit, reusable source representation.
    pub fn embed_decoder(
        &self,
        input: CausalLmInput<B>,
        encoder: EncoderContext<B>,
    ) -> Result<HiddenState<B>> {
        self.embed_decoder_with_hooks(input, encoder, &mut Hooks::new())
    }

    /// Embeds target tokens in a scope containing the decoder embedding point.
    pub fn embed_decoder_with_hooks(
        &self,
        input: CausalLmInput<B>,
        encoder: EncoderContext<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.embed_stack(input, Stack::Decoder, Some(encoder), hooks)
    }

    fn embed_stack(
        &self,
        input: CausalLmInput<B>,
        stack: Stack,
        encoder: Option<EncoderContext<B>>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![stack.embedding().info()], |hooks| {
            let (spec, embeddings) = match stack {
                Stack::Encoder => (&self.spec.encoder, &self.encoder_embed_tokens),
                Stack::Decoder => (&self.spec.decoder, &self.decoder_embed_tokens),
            };
            let positions = validate_and_resolve_input(
                &input,
                spec.vocabulary_size,
                spec.max_position_embeddings,
                "custom.seq2seq.embed",
            )?;
            let mut context = ExecutionContext::new(positions, input.token_mask)?;
            context.encoder = encoder;
            let hidden = hooks.apply(stack.embedding(), embeddings.forward(input.token_ids))?;
            HiddenState::new(hidden, context)
        })
    }

    /// Runs only the requested source blocks, preserving the raw residual boundary.
    pub fn run_encoder_layers(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
    ) -> Result<HiddenState<B>> {
        self.run_encoder_layers_with_hooks(state, range, &mut Hooks::new())
    }

    /// Runs source blocks with only their points in the current hook scope.
    pub fn run_encoder_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.run_stack_layers(state, range, Stack::Encoder, hooks)
    }

    /// Runs only the requested target blocks against the stored encoder context.
    pub fn run_decoder_layers(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
    ) -> Result<HiddenState<B>> {
        self.run_decoder_layers_with_hooks(state, range, &mut Hooks::new())
    }

    /// Runs target blocks with self/cross policies and only their points in scope.
    pub fn run_decoder_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.run_stack_layers(state, range, Stack::Decoder, hooks)
    }

    fn run_stack_layers(
        &self,
        mut state: HiddenState<B>,
        range: Range<usize>,
        stack: Stack,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        let (spec, layers) = match stack {
            Stack::Encoder => (&self.spec.encoder, &self.encoder_layers),
            Stack::Decoder => (&self.spec.decoder, &self.decoder_layers),
        };
        let address = |index| match stack {
            Stack::Encoder => Layer::encoder(index),
            Stack::Decoder => Layer::decoder(index),
        };
        let points = layers
            .iter()
            .enumerate()
            .filter(|(index, _)| range.contains(index))
            .flat_map(|(index, layer)| layer.activation_points(address(index)))
            .collect();
        hooks.run(points, |hooks| {
            self.validate_execution_state(&state, stack)?;
            state.validate_range(range.clone(), layers.len())?;
            validate_positions(
                state.context.positions.clone(),
                spec.max_position_embeddings,
                "custom.seq2seq.run_layers.positions",
            )?;
            for index in range.clone() {
                layers[index].validate_context(&state.hidden, &state.context, address(index))?;
            }
            for index in range.clone() {
                state.hidden = HiddenTransform::forward(
                    &layers[index],
                    state.hidden,
                    &state.context,
                    address(index),
                    hooks,
                )?;
            }
            state.next_layer = range.end;
            Ok(state)
        })
    }

    fn validate_execution_state(&self, state: &HiddenState<B>, stack: Stack) -> Result<()> {
        let (spec, embeddings, total) = match stack {
            Stack::Encoder => (
                &self.spec.encoder,
                &self.encoder_embed_tokens,
                self.encoder_layers.len(),
            ),
            Stack::Decoder => (
                &self.spec.decoder,
                &self.decoder_embed_tokens,
                self.decoder_layers.len(),
            ),
        };
        state.validate(spec.hidden_size)?;
        let embedding = embeddings.weight.val();
        if state.hidden.dtype() != embedding.dtype() || state.hidden.device() != embedding.device()
        {
            return Err(Error::configuration(
                "custom.seq2seq.execution.hidden",
                "hidden states must match this stack's embedding dtype and device",
            ));
        }
        if state.next_layer > total {
            return Err(Error::configuration(
                "custom.seq2seq.execution.next_layer",
                "next_layer exceeds this stack's block count",
            ));
        }
        for (layer, _) in &state.context.layer_attention {
            let count = match layer.stack() {
                Stack::Encoder => self.encoder_layers.len(),
                Stack::Decoder => self.decoder_layers.len(),
            };
            if layer.index() >= count {
                return Err(Error::configuration(
                    "custom.seq2seq.execution.layer_attention",
                    "self-attention override names a nonexistent layer",
                ));
            }
        }
        if state
            .context
            .layer_cross_attention
            .iter()
            .any(|(layer, _)| {
                layer.stack() != Stack::Decoder || layer.index() >= self.decoder_layers.len()
            })
        {
            return Err(Error::configuration(
                "custom.seq2seq.execution.layer_cross_attention",
                "cross-attention overrides must address existing decoder layers",
            ));
        }
        Ok(())
    }

    /// Normalizes an encoder state exactly once, including an intentionally retained prefix.
    pub fn normalize_encoder(&self, state: HiddenState<B>) -> Result<HiddenState<B>> {
        self.normalize_encoder_with_hooks(state, &mut Hooks::new())
    }

    /// Normalizes source states with a scope containing only the final encoder point.
    pub fn normalize_encoder_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.normalize_stack(state, Stack::Encoder, hooks)
    }

    /// Normalizes a decoder state exactly once, before any external task head.
    pub fn normalize_decoder(&self, state: HiddenState<B>) -> Result<HiddenState<B>> {
        self.normalize_decoder_with_hooks(state, &mut Hooks::new())
    }

    /// Normalizes target states with a scope containing only the final decoder point.
    pub fn normalize_decoder_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.normalize_stack(state, Stack::Decoder, hooks)
    }

    fn normalize_stack(
        &self,
        mut state: HiddenState<B>,
        stack: Stack,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![stack.final_norm().info()], |hooks| {
            let norm = match stack {
                Stack::Encoder => &self.encoder_final_norm,
                Stack::Decoder => &self.decoder_final_norm,
            };
            self.validate_execution_state(&state, stack)?;
            if state.normalized {
                return Err(Error::configuration(
                    "custom.seq2seq.normalize",
                    "final normalization has already been applied",
                ));
            }
            state.hidden = hooks.apply(stack.final_norm(), norm.forward(state.hidden))?;
            state.normalized = true;
            Ok(state)
        })
    }

    /// Evaluates the bidirectional source stack.
    pub fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        self.encode_with_hooks(source, options, &mut Hooks::new())
    }

    /// Returns bidirectional encoder states with selective capture or replacement.
    ///
    /// The final normalized states are suitable for pooling and a custom task head.
    /// Decoder point registrations are rejected because this call visits only the encoder.
    pub fn encode_with_hooks(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<EncoderOutput<B>> {
        hooks.run(self.encoder_activation_points(), |hooks| {
            self.encode_inner(source, options, hooks)
        })
    }

    /// Executes the source stack inside either an encoder-only or complete-model scope.
    fn encode_inner(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<EncoderOutput<B>> {
        validate_forward_options(options)?;
        if has_moe(&self.spec.encoder) && (options.router_logits || options.auxiliary_losses) {
            return Err(Error::BackendCapability(
                "EncoderOutput has no router-diagnostic field; encoder MoE diagnostics cannot be returned faithfully"
                    .into(),
            ));
        }
        let positions = validate_and_resolve_input(
            &source,
            self.spec.encoder.vocabulary_size,
            self.spec.encoder.max_position_embeddings,
            "custom.seq2seq.encoder",
        )?;
        let context = evaluated_context(positions.clone())?;
        let hidden = self.encoder_embed_tokens.forward(source.token_ids.clone());
        let mut hidden = hooks.apply(Stack::Encoder.embedding(), hidden)?;
        let mut hidden_states = Vec::with_capacity(if options.hidden_states {
            self.encoder_layers.len() + 2
        } else {
            0
        });
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        for (index, layer) in self.encoder_layers.iter().enumerate() {
            let (next, _router) = layer.forward(
                hidden,
                positions.clone(),
                source.token_mask.as_ref(),
                AttentionDirection::Bidirectional,
                context,
                None,
                Layer::encoder(index),
                hooks,
                None,
                None,
            )?;
            hidden = next;
            if options.hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = hooks.apply(
            Stack::Encoder.final_norm(),
            self.encoder_final_norm.forward(hidden),
        )?;
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(EncoderOutput {
            hidden_state: hidden,
            token_mask: source.token_mask,
            hidden_states,
            attentions: Vec::new(),
        })
    }

    /// Evaluates the causal target stack against an already encoded source.
    pub(super) fn forward_decoder_backbone(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: &CausalLmInput<B>,
        retain_hidden_states: bool,
    ) -> Result<BackboneOutput<B>> {
        self.forward_decoder_backbone_with_hooks(
            encoder,
            decoder,
            retain_hidden_states,
            &mut Hooks::new(),
        )
    }

    /// Executes the target stack without resetting the complete-model intervention scope.
    fn forward_decoder_backbone_with_hooks(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: &CausalLmInput<B>,
        retain_hidden_states: bool,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<BackboneOutput<B>> {
        validate_encoder_output(
            encoder,
            decoder.token_ids.dims()[0],
            self.spec.encoder.hidden_size,
            &decoder.token_ids.device(),
        )?;
        let positions = validate_and_resolve_input(
            decoder,
            self.spec.decoder.vocabulary_size,
            self.spec.decoder.max_position_embeddings,
            "custom.seq2seq.decoder",
        )?;
        let context = evaluated_context(positions.clone())?;
        let hidden = self.decoder_embed_tokens.forward(decoder.token_ids.clone());
        let mut hidden = hooks.apply(Stack::Decoder.embedding(), hidden)?;
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.decoder_layers.len() + 2
        } else {
            0
        });
        let mut routers = Vec::new();
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for (index, layer) in self.decoder_layers.iter().enumerate() {
            let (next, router) = layer.forward(
                hidden,
                positions.clone(),
                decoder.token_mask.as_ref(),
                AttentionDirection::Causal,
                context,
                Some((&encoder.hidden_state, encoder.token_mask.as_ref())),
                Layer::decoder(index),
                hooks,
                None,
                None,
            )?;
            hidden = next;
            if let Some(router) = router {
                routers.push(router);
            }
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = hooks.apply(
            Stack::Decoder.final_norm(),
            self.decoder_final_norm.forward(hidden),
        )?;
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(BackboneOutput {
            hidden,
            hidden_states,
            routers,
        })
    }

    /// Applies an independent or target-embedding-tied vocabulary projection.
    pub(super) fn project_decoder(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        match &self.lm_head {
            Some(head) => crate::operations::linear(hidden, head),
            None => Ok(tied_projection(
                hidden,
                self.decoder_embed_tokens.weight.val(),
            )),
        }
    }

    /// Evaluates a complete stateless encoder-decoder graph.
    pub fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        self.forward_with_hooks(input, options, &mut Hooks::new())
    }

    /// Evaluates both stacks with one shared, explicitly scoped intervention registry.
    ///
    /// Encoder edits propagate through decoder cross-attention. Captures from both
    /// stacks remain available after this call; encoding never resets decoder hooks.
    pub fn forward_with_hooks(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Seq2SeqOutput<B>> {
        hooks.run(self.activation_points(), |hooks| {
            validate_forward_options(options)?;
            let encoder = self.encode_inner(input.encoder, options, hooks)?;
            let backbone = self.forward_decoder_backbone_with_hooks(
                &encoder,
                &input.decoder,
                options.hidden_states,
                hooks,
            )?;
            self.finish_forward(encoder, backbone, &input.decoder, options)
        })
    }

    /// Produces the decoder outputs after both modified stacks have executed.
    fn finish_forward(
        &self,
        encoder: EncoderOutput<B>,
        backbone: BackboneOutput<B>,
        input: &CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        let selected = select_hidden(
            backbone.hidden.clone(),
            input.token_mask.as_ref(),
            &options.logits,
        )?;
        let routers = router_diagnostics(&backbone.routers, input.token_mask.as_ref(), options)?;
        let decoder = causal_output(
            self.project_decoder(selected)?,
            backbone.hidden,
            backbone.hidden_states,
            routers,
            options,
        );
        Ok(Seq2SeqOutput {
            decoder,
            encoder,
            cross_attentions: Vec::new(),
        })
    }
}

impl ArchitectureConfig for Seq2SeqSpec {
    const MODEL_TYPES: &'static [&'static str] = &["modular_seq2seq"];

    fn validate(&self) -> Result<()> {
        validate_seq2seq_execution(self)
    }
}

impl<B: Backend> Seq2SeqLanguageModel<B> for ModularSeq2SeqLm<B> {
    type Config = Seq2SeqSpec;
    type State = ModularDecodeState<B>;

    fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        ModularSeq2SeqLm::encode(self, source, options)
    }

    fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        ModularSeq2SeqLm::forward(self, input, options)
    }

    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        if config.capacity() > self.spec.decoder.max_position_embeddings {
            return Err(Error::configuration(
                "custom.seq2seq.decode_state.capacity",
                format!(
                    "{} exceeds decoder max_position_embeddings {}",
                    config.capacity(),
                    self.spec.decoder.max_position_embeddings
                ),
            ));
        }
        ModularDecodeState::new(
            config,
            batch_size,
            self.spec.decoder.clone(),
            StatePurpose::Seq2Seq,
            device,
        )
    }

    fn prefill_decoder(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        state.run_seq2seq(self, encoder, decoder, options, true)
    }

    fn decode(
        &self,
        encoder: &EncoderOutput<B>,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        state.run_seq2seq(
            self,
            encoder,
            CausalLmInput {
                token_ids: input.token_ids,
                token_mask: input.token_mask,
                position_ids: input.position_ids,
            },
            options,
            false,
        )
    }
}

/// Validates cross-stack topology and executable encoder/decoder capabilities.
fn validate_seq2seq_execution(spec: &Seq2SeqSpec) -> Result<()> {
    spec.validate()?;
    validate_decoder_execution(
        &spec.encoder,
        AttentionDirection::Bidirectional,
        "custom.seq2seq.encoder",
    )?;
    validate_decoder_execution(
        &spec.decoder,
        AttentionDirection::Causal,
        "custom.seq2seq.decoder",
    )
}

/// Checks externally supplied encoder states before cross-attention projection.
fn validate_encoder_output<B: Backend>(
    encoder: &EncoderOutput<B>,
    decoder_batch: usize,
    hidden_size: usize,
    decoder_device: &B::Device,
) -> Result<()> {
    let [batch, source_sequence, width] = encoder.hidden_state.dims();
    if batch != decoder_batch || source_sequence == 0 || width != hidden_size {
        return Err(Error::shape(
            "custom.seq2seq.encoder_output",
            format!("[{decoder_batch}, source>0, {hidden_size}]"),
            format!("[{batch}, {source_sequence}, {width}]"),
        ));
    }
    if encoder.hidden_state.device() != *decoder_device {
        return Err(Error::BackendCapability(
            "custom encoder output and decoder input must reside on the same device".into(),
        ));
    }
    if let Some(mask) = &encoder.token_mask
        && mask.as_tensor().dims() != [batch, source_sequence]
    {
        return Err(Error::shape(
            "custom.seq2seq.encoder_mask",
            format!("[{batch}, {source_sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    if let Some(mask) = &encoder.token_mask {
        if mask.as_tensor().device() != *decoder_device {
            return Err(Error::BackendCapability(
                "custom encoder mask and decoder input must reside on the same device".into(),
            ));
        }
        let minimum = mask
            .as_tensor()
            .clone()
            .int()
            .sum_dim(1)
            .min()
            .try_into_scalar()
            .map_err(|error| {
                Error::BackendCapability(format!("failed to validate custom encoder mask: {error}"))
            })?
            .elem::<i64>();
        if minimum == 0 {
            return Err(Error::shape(
                "custom.seq2seq.encoder_mask",
                "at least one valid source token in every row",
                "a fully masked row",
            ));
        }
    }
    Ok(())
}
