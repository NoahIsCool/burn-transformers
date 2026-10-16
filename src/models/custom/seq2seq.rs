//! Modular encoder-decoder transformer assembly.
//!
//! Source blocks use bidirectional self-attention. Target blocks use causal self-attention,
//! then an independently parameterized cross-attention branch ([`CrossAttentionSpec`]), then
//! their configured feed-forward branch. Every cross-attention block uses the decoder layer's
//! mixer-side normalization convention because [`Seq2SeqSpec`] has no separate cross-norm
//! field. Both stacks run their blocks through the shared [`ModularLayer`] path over one
//! [`StackPass`] per stack execution.
//!
//! Staged experiments address each stack through [`EncoderStages`] and [`DecoderStages`],
//! which implement [`StagedModel`]; the model itself implements [`HookedModel`].
//!
//! [`CrossAttentionSpec`]: crate::architecture::CrossAttentionSpec

use std::ops::Range;

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{Tensor, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLmInput, DecodeInput, EncoderOutput, Error, ForwardOptions, Result,
    Seq2SeqInput, Seq2SeqLanguageModel, Seq2SeqOutput,
    architecture::{HookedModel, Seq2SeqSpec, StagedModel},
    attention::AttentionPrecisionMode,
    cache::CacheConfig,
    experiment::{EncoderContext, HiddenState},
    initialization::InitPolicy,
    intervention::{Hooks, Layer, PointInfo, Stack},
};

use crate::models::shared::{check_nonempty_mask, ensure_grid, ensure_same_device};

use super::{
    common::{
        BackboneOutput, causal_output, embedded_state, has_moe, router_diagnostics, select_hidden,
        tied_projection, validate_and_resolve_input, validate_and_resolve_input_with,
        validate_forward_options, validate_stack_execution, validate_staged_positions,
    },
    init::scaled_init_policy,
    layer::ModularLayer,
    norm::ModularNorm,
    pass::StackPass,
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
    ///
    /// Parameters use Burn's default initializers; see [`Self::init_with`] and
    /// [`Self::scaled_init_policy`] for training from scratch.
    pub fn init(spec: Seq2SeqSpec, device: &B::Device) -> Result<Self> {
        validate_seq2seq_execution(&spec)?;
        let cross_attention = spec.cross_attention();
        let encoder_layers = spec
            .encoder
            .layers
            .iter()
            .map(|layer| ModularLayer::build(layer, spec.encoder.hidden_size, None, device))
            .collect::<Result<Vec<_>>>()?;
        let decoder_layers = spec
            .decoder
            .layers
            .iter()
            .map(|layer| {
                ModularLayer::build(
                    layer,
                    spec.decoder.hidden_size,
                    Some(&cross_attention),
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

    /// Builds both stacks and initializes their parameters with an explicit policy.
    pub fn init_with(spec: Seq2SeqSpec, policy: &InitPolicy, device: &B::Device) -> Result<Self> {
        Ok(policy.apply(Self::init(spec, device)?)?.0)
    }

    /// The recommended random-initialization policy for this topology.
    ///
    /// As [`super::ModularCausalLm::scaled_init_policy`], with cross-attention output
    /// projections included and depth `L` the larger of the two stacks' block counts.
    #[must_use]
    pub fn scaled_init_policy(spec: &Seq2SeqSpec, std: f64) -> InitPolicy {
        scaled_init_policy(
            [&spec.encoder, &spec.decoder],
            spec.encoder.layers.len().max(spec.decoder.layers.len()),
            true,
            std,
        )
    }

    /// Selects numerical execution for conventional encoder, decoder, and cross-attention branches.
    ///
    /// [`AttentionPrecisionMode::EagerFp32Softmax`] enables score and probability
    /// capture/edit points, listed by [`Self::activation_points`]. This reference path
    /// materializes attention matrices with quadratic sequence memory cost; select
    /// it explicitly for experiments that need those points. Initialization keeps
    /// [`AttentionPrecisionMode::BackendDefault`]. Any decoder DeltaNet or Mamba-2
    /// mixers retain their recurrent computation; their cross-attention uses the chosen mode.
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
        points.extend(self.decoder_activation_points());
        points
    }

    /// Lists only points visited by [`Self::encode_with_hooks`].
    pub fn encoder_activation_points(&self) -> Vec<PointInfo> {
        stack_points(&self.encoder_layers, Stack::Encoder)
    }

    /// Lists only points visited by the decoder half of [`Self::forward_with_hooks`].
    fn decoder_activation_points(&self) -> Vec<PointInfo> {
        stack_points(&self.decoder_layers, Stack::Decoder)
    }

    /// Borrows independently executable source blocks.
    pub fn encoder_layers(&self) -> &[ModularLayer<B>] {
        &self.encoder_layers
    }

    /// Borrows independently executable target blocks.
    pub fn decoder_layers(&self) -> &[ModularLayer<B>] {
        &self.decoder_layers
    }

    /// A staged view of the encoder stack implementing [`StagedModel`].
    #[must_use]
    pub fn encoder_stages(&self) -> EncoderStages<'_, B> {
        EncoderStages { model: self }
    }

    /// A staged view of the decoder stack reading `encoder` through cross-attention.
    ///
    /// The encoder context typically comes from
    /// `encoder_stages().run_stack(source)?.into_encoder_context()`.
    #[must_use]
    pub fn decoder_stages(&self, encoder: EncoderContext<B>) -> DecoderStages<'_, B> {
        DecoderStages {
            model: self,
            encoder,
        }
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

    /// Validates tokens and creates `[batch, token, hidden]` state for one selected stack.
    ///
    /// Token, mask, and position values cost one device read; an attached encoder's
    /// layout is checked without reading values (its positions are checked when blocks
    /// run). Decoder state retains the supplied encoder representation separately from
    /// its own positions and padding; hooks see only that embedding point.
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
            let (positions, evaluated_context) = validate_and_resolve_input(
                &input,
                spec.vocabulary_size,
                spec.max_position_embeddings,
                "custom.seq2seq.embed",
            )?;
            let hidden = hooks.apply(stack.embedding(), embeddings.forward(input.token_ids))?;
            embedded_state(
                hidden,
                positions,
                input.token_mask,
                evaluated_context,
                encoder,
            )
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

    /// Runs a contiguous unnormalized block range and advances the continuation cursor.
    ///
    /// All selected contexts are validated before the first block or callback,
    /// position values are read once, and only points belonging to the selected
    /// stack/range enter the hook scope.
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
        let points = layers
            .iter()
            .enumerate()
            .filter(|(index, _)| range.contains(index))
            .flat_map(|(index, layer)| layer.activation_points(stack.layer(index)))
            .collect();
        hooks.run(points, |hooks| {
            self.validate_execution_state(&state, stack)?;
            state.validate_range(range.clone(), layers.len())?;
            validate_staged_positions(
                &state.context,
                spec.max_position_embeddings,
                "custom.seq2seq.run_layers.positions",
            )?;
            for index in range.clone() {
                layers[index].validate_prevalidated(
                    &state.hidden,
                    &state.context,
                    stack.layer(index),
                )?;
            }
            let pass = StackPass::from_context(&state.context, stack, state.hidden.dtype());
            for index in range.clone() {
                let layer = stack.layer(index);
                // Hooks may replace hidden states between blocks; re-check them sync-free.
                layers[index].validate_prevalidated(&state.hidden, &state.context, layer)?;
                state.hidden = layers[index].forward(state.hidden, &pass, layer, hooks)?.0;
            }
            state.next_layer = range.end;
            Ok(state)
        })
    }

    /// Checks stack width, dtype/device, layout, cursor, and self/cross-attention addresses.
    ///
    /// Encoder and decoder states have distinct topology contracts even when
    /// their residual widths happen to match. No device value is read.
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
        let [batch, tokens, width] = state.hidden.dims();
        if width != spec.hidden_size || batch == 0 || tokens == 0 {
            return Err(Error::shape(
                "execution.hidden",
                format!("nonempty [batch, token, {}]", spec.hidden_size),
                format!("{:?}", state.hidden.dims()),
            ));
        }
        state
            .context
            .validate_layout([batch, tokens], &state.hidden.device())?;
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

    /// Applies one stack-specific final norm exactly once without running omitted blocks.
    ///
    /// Normalization preserves `[batch, token, hidden]`, dtype, and the graph
    /// needed by an external trainable task head.
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
        let (positions, evaluated_context) = validate_and_resolve_input(
            &source,
            self.spec.encoder.vocabulary_size,
            self.spec.encoder.max_position_embeddings,
            "custom.seq2seq.encoder",
        )?;
        let hidden = self.encoder_embed_tokens.forward(source.token_ids.clone());
        let pass = StackPass::new(
            positions,
            source.token_mask.as_ref(),
            evaluated_context,
            Stack::Encoder,
            hidden.dtype(),
        )
        .with_row_invariant_positions(source.position_ids.is_none() && source.token_mask.is_none());
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
            hidden = layer
                .forward(hidden, &pass, Layer::encoder(index), hooks)?
                .0;
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
        validate_encoder_layout(
            encoder,
            decoder.token_ids.dims()[0],
            self.spec.encoder.hidden_size,
            &decoder.token_ids.device(),
        )?;
        // The source mask's row counts share the decoder input's readback.
        let (positions, evaluated_context) = validate_and_resolve_input_with(
            decoder,
            self.spec.decoder.vocabulary_size,
            self.spec.decoder.max_position_embeddings,
            "custom.seq2seq.decoder",
            |checks| {
                let [batch, source_sequence, _] = encoder.hidden_state.dims();
                check_nonempty_mask(
                    checks,
                    encoder.token_mask.as_ref(),
                    batch,
                    source_sequence,
                    ENCODER_MASK,
                )
            },
        )?;
        let hidden = self.decoder_embed_tokens.forward(decoder.token_ids.clone());
        let pass = StackPass::new(
            positions,
            decoder.token_mask.as_ref(),
            evaluated_context,
            Stack::Decoder,
            hidden.dtype(),
        )
        .with_row_invariant_positions(
            decoder.position_ids.is_none() && decoder.token_mask.is_none(),
        )
        .with_source(&encoder.hidden_state, encoder.token_mask.as_ref());
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
            let (next, router) = layer.forward(hidden, &pass, Layer::decoder(index), hooks)?;
            hidden = next;
            routers.extend(router);
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
            None => tied_projection(hidden, self.decoder_embed_tokens.weight.val()),
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

/// Lists one stack's embedding, block, and final-norm points in emission order.
fn stack_points<B: Backend>(layers: &[ModularLayer<B>], stack: Stack) -> Vec<PointInfo> {
    let mut points = vec![stack.embedding().info()];
    for (index, layer) in layers.iter().enumerate() {
        points.extend(layer.activation_points(stack.layer(index)));
    }
    points.push(stack.final_norm().info());
    points
}

/// Staged execution of a [`ModularSeq2SeqLm`] encoder stack.
///
/// `run_stack(source)?.into_encoder_context()` produces the source for
/// [`ModularSeq2SeqLm::decoder_stages`].
#[derive(Clone, Copy, Debug)]
pub struct EncoderStages<'a, B: Backend> {
    /// The model whose encoder stack runs.
    model: &'a ModularSeq2SeqLm<B>,
}

impl<B: Backend> StagedModel<B> for EncoderStages<'_, B> {
    /// Number of encoder blocks.
    fn num_layers(&self) -> usize {
        self.model.encoder_layers.len()
    }

    /// The encoder stack.
    fn stack(&self) -> Stack {
        Stack::Encoder
    }

    /// Runs [`ModularSeq2SeqLm::embed_encoder_with_hooks`].
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model.embed_encoder_with_hooks(input, hooks)
    }

    /// Runs [`ModularSeq2SeqLm::run_encoder_layers_with_hooks`].
    fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model
            .run_encoder_layers_with_hooks(state, range, hooks)
    }

    /// Runs [`ModularSeq2SeqLm::normalize_encoder_with_hooks`].
    fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model.normalize_encoder_with_hooks(state, hooks)
    }
}

/// Staged execution of a [`ModularSeq2SeqLm`] decoder stack against one encoder context.
#[derive(Clone, Debug)]
pub struct DecoderStages<'a, B: Backend> {
    /// The model whose decoder stack runs.
    model: &'a ModularSeq2SeqLm<B>,
    /// Normalized source states read by every cross-attention branch.
    encoder: EncoderContext<B>,
}

impl<B: Backend> StagedModel<B> for DecoderStages<'_, B> {
    /// Number of decoder blocks.
    fn num_layers(&self) -> usize {
        self.model.decoder_layers.len()
    }

    /// The decoder stack.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Runs [`ModularSeq2SeqLm::embed_decoder_with_hooks`] with this view's encoder context.
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model
            .embed_decoder_with_hooks(input, self.encoder.clone(), hooks)
    }

    /// Runs [`ModularSeq2SeqLm::run_decoder_layers_with_hooks`].
    fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model
            .run_decoder_layers_with_hooks(state, range, hooks)
    }

    /// Runs [`ModularSeq2SeqLm::normalize_decoder_with_hooks`].
    fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model.normalize_decoder_with_hooks(state, hooks)
    }
}

impl<B: Backend> HookedModel<B> for ModularSeq2SeqLm<B> {
    /// Source and target inputs.
    type Input = Seq2SeqInput<B>;
    /// Encoder and decoder outputs.
    type Output = Seq2SeqOutput<B>;

    /// Every point of both stacks, encoder first, in emission order.
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        Ok(ModularSeq2SeqLm::activation_points(self))
    }

    /// Runs [`ModularSeq2SeqLm::forward_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Seq2SeqOutput<B>> {
        ModularSeq2SeqLm::forward_with_hooks(self, input, options, hooks)
    }
}

impl ArchitectureConfig for Seq2SeqSpec {
    /// Architecture discriminator for configurable encoder-decoder graphs.
    const MODEL_TYPES: &'static [&'static str] = &["modular_seq2seq"];

    /// Validates cross-stack topology and the supported encoder/decoder mixer roles.
    fn validate(&self) -> Result<()> {
        validate_seq2seq_execution(self)
    }
}

impl<B: Backend> Seq2SeqLanguageModel<B> for ModularSeq2SeqLm<B> {
    /// Serializable encoder, decoder, and cross-attention specification.
    type Config = Seq2SeqSpec;
    /// External decoder-token history paired with the encoder-decoder model role.
    type State = ModularDecodeState<B>;

    /// Runs the stateless bidirectional encoder through the common seq2seq interface.
    fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        ModularSeq2SeqLm::encode(self, source, options)
    }

    /// Runs the encoder and conditioned decoder through the common seq2seq interface.
    fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        ModularSeq2SeqLm::forward(self, input, options)
    }

    /// Allocates external decoder token history within the target position capacity.
    ///
    /// Encoder representations are supplied by the caller; model parameters
    /// never own mutable generation history.
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

    /// Evaluates and commits an initial decoder prompt conditioned on encoder states.
    fn prefill_decoder(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        state.run_seq2seq(self, encoder, decoder, options, true)
    }

    /// Appends target tokens against encoder states and commits the evaluated history.
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
    validate_stack_execution(&spec.encoder, Stack::Encoder, "custom.seq2seq.encoder")?;
    validate_stack_execution(&spec.decoder, Stack::Decoder, "custom.seq2seq.decoder")
}

/// Name of a supplied encoder result's source mask in errors.
const ENCODER_MASK: &str = "custom.seq2seq.encoder_mask";

/// Checks a supplied encoder result's extents and devices without reading values.
///
/// The caller defers the source mask's non-empty-row check into the decoder
/// input's readback with [`check_nonempty_mask`].
fn validate_encoder_layout<B: Backend>(
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
    ensure_same_device(
        "custom.seq2seq.encoder_output",
        &encoder.hidden_state.device(),
        decoder_device,
        "the custom decoder input",
    )?;
    if let Some(mask) = &encoder.token_mask {
        ensure_same_device(
            ENCODER_MASK,
            &mask.as_tensor().device(),
            decoder_device,
            "the custom decoder input",
        )?;
        ensure_grid(
            mask.as_tensor().dims(),
            batch,
            source_sequence,
            ENCODER_MASK,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "seq2seq_tests.rs"]
/// Hook propagation through both stacks, staged-state checks, and encoder-output contracts.
mod tests;
