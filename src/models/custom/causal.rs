//! Decoder-only modular language model assembly.
//!
//! The model owns only trainable parameters and immutable topology. Autoregressive session
//! history remains in [`ModularDecodeState`], preserving a clean
//! stateless graph for autodiff and model surgery.

use std::ops::Range;

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error,
    ForwardOptions, Result,
    architecture::DecoderSpec,
    attention::AttentionPrecisionMode,
    cache::CacheConfig,
    experiment::{AttentionPolicy, ExecutionContext, HiddenState, HiddenTransform},
    intervention::{Hooks, Layer, PointInfo, Stack},
};

use super::{
    common::{
        AttentionDirection, BackboneOutput, ModularLayer, ModularNorm, causal_output,
        evaluated_context, router_diagnostics, select_hidden, tied_projection,
        validate_and_resolve_input, validate_decoder_execution, validate_forward_options,
        validate_positions,
    },
    state::{ModularDecodeState, StatePurpose},
};

/// Immutable, host-validated tokens for unpadded causal sequence training.
///
/// Construct with [`ModularCausalLm::prepare_host_tokens`]. Tokens have contiguous
/// positions starting at zero in every row. Private fields prevent replacing the
/// checked payload or its bounds; forwarding checks model compatibility using
/// metadata only, without GPU-to-host token or position reductions. This is an
/// input capability, not model state, and can be reused across optimizer updates.
#[derive(Clone, Debug)]
pub struct PreparedCausalInput<B: Backend> {
    /// Checked token payload; no padding or caller-specified positions.
    input: CausalLmInput<B>,
    /// Generated contiguous positions, shared by all blocks and repeated calls.
    positions: Tensor<B, 2, Int>,
    /// Vocabulary bound against which every host token was validated.
    vocabulary_size: usize,
    /// Position-capacity contract of the model that prepared the input.
    max_position_embeddings: usize,
}

impl<B: Backend> PreparedCausalInput<B> {
    /// Returns the checked `[batch, sequence]` shape without reading the device.
    pub fn shape(&self) -> [usize; 2] {
        self.input.token_ids.dims()
    }
}

/// A heterogeneous decoder-only transformer assembled from [`DecoderSpec`].
#[derive(Module, Debug)]
pub struct ModularCausalLm<B: Backend> {
    /// Token lookup table.
    pub embed_tokens: Embedding<B>,
    /// Ordered heterogeneous residual blocks.
    layers: Vec<ModularLayer<B>>,
    /// Final normalization before vocabulary projection.
    final_norm: ModularNorm<B>,
    /// Independent vocabulary projection, absent when embeddings are tied.
    pub lm_head: Option<Linear<B>>,
    /// Validated topology excluded from parameter records.
    #[module(skip)]
    spec: DecoderSpec,
}

impl<B: Backend> ModularCausalLm<B> {
    /// Validates a modular topology and allocates its Burn parameter graph.
    ///
    /// Unsupported but well-formed features return a capability error before the embedding or
    /// any layer parameter is initialized. This makes failed experiment construction cheap and
    /// prevents an unavailable mixer from being approximated silently.
    pub fn init(spec: DecoderSpec, device: &B::Device) -> Result<Self> {
        validate_decoder_execution(&spec, AttentionDirection::Causal, "custom.causal")?;
        let layers = spec
            .layers
            .iter()
            .map(|layer| ModularLayer::init(layer, spec.hidden_size, None, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(spec.vocabulary_size, spec.hidden_size).init(device),
            layers,
            final_norm: ModularNorm::init(spec.final_norm, spec.hidden_size, device)?,
            lm_head: (!spec.tie_word_embeddings).then(|| {
                crate::operations::init_linear(
                    &LinearConfig::new(spec.hidden_size, spec.vocabulary_size)
                        .with_bias(false)
                        .with_layout(LinearLayout::Col),
                    device,
                )
            }),
            spec,
        })
    }

    /// Selects the numerical execution path for conventional projected self-attention layers.
    ///
    /// [`AttentionPrecisionMode::EagerFp32Softmax`] enables score and probability
    /// capture/edit points, listed by [`Self::activation_points`]. This reference path
    /// materializes attention matrices with quadratic sequence memory cost; select
    /// it explicitly for experiments that need those points. Initialization uses
    /// [`AttentionPrecisionMode::BackendDefault`] unless this method is called.
    /// DeltaNet keeps its recurrent computation; MLA retains its explicit eager graph.
    #[must_use]
    pub fn with_attention_precision(mut self, mode: AttentionPrecisionMode) -> Self {
        for layer in &mut self.layers {
            layer.set_attention_precision(mode);
        }
        self
    }

    /// Returns the exact serializable topology used to construct the graph.
    #[must_use]
    pub const fn spec(&self) -> &DecoderSpec {
        &self.spec
    }

    /// Checks complete host token rows once, then uploads an opaque causal input.
    ///
    /// Every row has `shape[1]` valid tokens at positions `0..shape[1]`, without
    /// BOS insertion or target shifting. Rejects zero/overflowing dimensions,
    /// mismatched storage, tokens outside this vocabulary, and excessive context
    /// before allocating device tensors. The input uses this model's device and
    /// may be reused by compatible models with the same vocabulary and context
    /// capacity. Use ordinary [`Self::forward`] for padding or explicit positions.
    pub fn prepare_host_tokens(
        &self,
        tokens: Vec<i32>,
        shape: [usize; 2],
    ) -> Result<PreparedCausalInput<B>> {
        let [batch, sequence] = shape;
        let count = batch
            .checked_mul(sequence)
            .filter(|_| batch > 0 && sequence > 0);
        if count != Some(tokens.len()) {
            return Err(Error::shape(
                "custom.prepared.token_ids",
                "nonempty [batch, sequence] matching host token storage",
                format!("shape {shape:?}, {} tokens", tokens.len()),
            ));
        }
        if sequence > self.spec.max_position_embeddings || sequence > i32::MAX as usize {
            return Err(Error::shape(
                "custom.prepared.positions",
                format!(
                    "sequence at most {} and within I32 positions",
                    self.spec.max_position_embeddings
                ),
                sequence.to_string(),
            ));
        }
        if let Some((index, token)) = tokens
            .iter()
            .enumerate()
            .find(|(_, token)| **token < 0 || **token as usize >= self.spec.vocabulary_size)
        {
            return Err(Error::configuration(
                "custom.prepared.token_ids",
                format!(
                    "token {token} at flat index {index} is outside 0..{}",
                    self.spec.vocabulary_size
                ),
            ));
        }
        let device = self.embed_tokens.weight.lazy_device();
        let token_ids = Tensor::from_data(TensorData::new(tokens, shape), &device);
        let positions = Tensor::<B, 1, Int>::arange(0..sequence as i64, &device)
            .reshape([1, sequence])
            .expand(shape);
        Ok(PreparedCausalInput {
            input: CausalLmInput {
                token_ids,
                token_mask: None,
                position_ids: None,
            },
            positions,
            vocabulary_size: self.spec.vocabulary_size,
            max_position_embeddings: self.spec.max_position_embeddings,
        })
    }

    /// Executes host-validated causal inputs without redundant device scalar reads.
    ///
    /// Computes the same differentiable graph and output selection as
    /// [`Self::forward`]. Preparation changes input validation, not transformer
    /// arithmetic, model parameters, attention visibility, or target timing.
    pub fn forward_prepared(
        &self,
        input: &PreparedCausalInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_prepared_with_hooks(input, options, &mut Hooks::new())
    }

    /// Executes prepared causal inputs with ordinary activation captures and edits.
    ///
    /// Rejects different vocabulary/context contracts or model devices before
    /// invoking hooks. Tensor storage remains private and cannot be substituted
    /// after host validation. No device values are read to establish compatibility.
    pub fn forward_prepared_with_hooks(
        &self,
        input: &PreparedCausalInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        hooks.run(self.activation_points(), |hooks| {
            validate_forward_options(options)?;
            if input.vocabulary_size != self.spec.vocabulary_size
                || input.max_position_embeddings != self.spec.max_position_embeddings
                || input.input.token_ids.device() != self.embed_tokens.weight.lazy_device()
            {
                return Err(Error::configuration(
                    "custom.prepared.compatibility",
                    "prepared input must match the model vocabulary, context capacity, and device",
                ));
            }
            let backbone = self.forward_resolved_backbone_with_hooks(
                &input.input,
                input.positions.clone(),
                input.shape()[1],
                options.hidden_states,
                AttentionDirection::Causal,
                hooks,
            )?;
            self.finish_forward(backbone, &input.input, options)
        })
    }

    /// Lists the exact activations that this model can observe or replace.
    ///
    /// Residual points have shape `[batch, token, hidden]`. Dense feed-forward
    /// pre/activation points expose the gate projection before/after its nonlinearity,
    /// before multiplication by the content projection. MoE layers expose only MLP
    /// input/output. Attention input/output surround the configured mixer, including
    /// DeltaNet when that layer uses a recurrent mixer.
    pub fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Decoder.embedding().info()];
        for (index, layer) in self.layers.iter().enumerate() {
            points.extend(layer.activation_points(Layer::decoder(index)));
        }
        points.push(Stack::Decoder.final_norm().info());
        points
    }

    /// Borrows the independently executable blocks, in architecture order.
    pub fn layers(&self) -> &[ModularLayer<B>] {
        &self.layers
    }

    /// Embeds tokens into a raw state carrying positions and semantic padding.
    ///
    /// Change `state.context` to choose self-attention behavior before calling
    /// [`Self::run_layers`]. Explicit policies replace causal/window defaults;
    /// padding remains excluded and extra bias adds to learned positional bias.
    pub fn embed(&self, input: CausalLmInput<B>) -> Result<HiddenState<B>> {
        self.embed_with_hooks(input, &mut Hooks::new())
    }

    /// Embeds tokens with a scope containing only the embedding point.
    pub fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.embedding().info()], |hooks| {
            let positions = validate_and_resolve_input(
                &input,
                self.spec.vocabulary_size,
                self.spec.max_position_embeddings,
                "custom.embed",
            )?;
            let context = ExecutionContext::new(positions, input.token_mask)?;
            let hidden = hooks.apply(
                Stack::Decoder.embedding(),
                self.embed_tokens.forward(input.token_ids),
            )?;
            HiddenState::new(hidden, context)
        })
    }

    /// Runs exactly the next block range without embeddings, final normalization, or logits.
    pub fn run_layers(&self, state: HiddenState<B>, range: Range<usize>) -> Result<HiddenState<B>> {
        self.run_layers_with_hooks(state, range, &mut Hooks::new())
    }

    /// Runs a range using its stored context and a scope containing only those blocks.
    ///
    /// Ranges must begin at `state.next_layer`. Every policy in the requested range
    /// is validated before callbacks run, including unsupported recurrent policies.
    /// Use separate hook collections when retaining captures across separate stages.
    pub fn run_layers_with_hooks(
        &self,
        mut state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        let points = self
            .layers
            .iter()
            .enumerate()
            .filter(|(index, _)| range.contains(index))
            .flat_map(|(index, layer)| layer.activation_points(Layer::decoder(index)))
            .collect();
        hooks.run(points, |hooks| {
            self.validate_execution_state(&state)?;
            state.validate_range(range.clone(), self.layers.len())?;
            validate_positions(
                state.context.positions.clone(),
                self.spec.max_position_embeddings,
                "custom.run_layers.positions",
            )?;
            for index in range.clone() {
                self.layers[index].validate_context(
                    &state.hidden,
                    &state.context,
                    Layer::decoder(index),
                )?;
            }
            for index in range.clone() {
                state.hidden = HiddenTransform::forward(
                    &self.layers[index],
                    state.hidden,
                    &state.context,
                    Layer::decoder(index),
                    hooks,
                )?;
            }
            state.next_layer = range.end;
            Ok(state)
        })
    }

    /// Checks `[batch, token, hidden]` state compatibility before staged execution.
    ///
    /// Dtype/device must match embeddings, the cursor must fit the block list,
    /// and attention overrides must address this decoder without cross-attention.
    fn validate_execution_state(&self, state: &HiddenState<B>) -> Result<()> {
        state.validate(self.spec.hidden_size)?;
        let embedding = self.embed_tokens.weight.val();
        if state.hidden.dtype() != embedding.dtype() || state.hidden.device() != embedding.device()
        {
            return Err(Error::configuration(
                "custom.execution.hidden",
                "hidden states must match the model embedding dtype and device",
            ));
        }
        if state.next_layer > self.layers.len() {
            return Err(Error::configuration(
                "custom.execution.next_layer",
                "next_layer exceeds the model's block count",
            ));
        }
        if state
            .context
            .layer_attention
            .iter()
            .any(|(layer, _)| layer.stack() != Stack::Decoder || layer.index() >= self.layers.len())
        {
            return Err(Error::configuration(
                "custom.execution.layer_attention",
                "self-attention overrides must address existing decoder layers",
            ));
        }
        if state.context.cross_attention.is_some()
            || !state.context.layer_cross_attention.is_empty()
            || state.context.encoder.is_some()
        {
            return Err(Error::configuration(
                "custom.execution.cross_attention",
                "this decoder-only model has no cross-attention branches",
            ));
        }
        Ok(())
    }

    /// Applies final normalization exactly once, including to an intentionally retained prefix.
    ///
    /// The normalized state can feed any task head. Calling this does not execute omitted blocks.
    pub fn normalize(&self, state: HiddenState<B>) -> Result<HiddenState<B>> {
        self.normalize_with_hooks(state, &mut Hooks::new())
    }

    /// Applies final normalization with its own one-point hook scope.
    pub fn normalize_with_hooks(
        &self,
        mut state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.final_norm().info()], |hooks| {
            self.validate_execution_state(&state)?;
            if state.normalized {
                return Err(Error::configuration(
                    "custom.normalize",
                    "final normalization has already been applied",
                ));
            }
            state.hidden = hooks.apply(
                Stack::Decoder.final_norm(),
                self.final_norm.forward(state.hidden),
            )?;
            state.normalized = true;
            Ok(state)
        })
    }

    /// Evaluates embeddings, heterogeneous blocks, and final normalization.
    pub(super) fn forward_backbone(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
    ) -> Result<BackboneOutput<B>> {
        self.forward_backbone_with_hooks(
            input,
            retain_hidden_states,
            AttentionDirection::Causal,
            &mut Hooks::new(),
        )
    }

    /// Executes the chosen full-sequence graph inside its caller's intervention scope.
    fn forward_backbone_with_hooks(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
        direction: AttentionDirection,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<BackboneOutput<B>> {
        let positions = validate_and_resolve_input(
            input,
            self.spec.vocabulary_size,
            self.spec.max_position_embeddings,
            "custom.causal",
        )?;
        let context = evaluated_context(positions.clone())?;
        self.forward_resolved_backbone_with_hooks(
            input,
            positions,
            context,
            retain_hidden_states,
            direction,
            hooks,
        )
    }

    /// Executes a backbone after either host or device input validation.
    ///
    /// Callers supply checked positions and their exclusive maximum. Keeping one
    /// arithmetic path preserves ordinary/prepared output, hook, and gradient parity.
    fn forward_resolved_backbone_with_hooks(
        &self,
        input: &CausalLmInput<B>,
        positions: Tensor<B, 2, Int>,
        context: usize,
        retain_hidden_states: bool,
        direction: AttentionDirection,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<BackboneOutput<B>> {
        let hidden = self.embed_tokens.forward(input.token_ids.clone());
        let mut hidden = hooks.apply(Stack::Decoder.embedding(), hidden)?;
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        let mut routers = Vec::new();
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        let policy =
            (direction == AttentionDirection::Bidirectional).then(AttentionPolicy::bidirectional);
        for (index, layer) in self.layers.iter().enumerate() {
            let (next, router) = layer.forward(
                hidden,
                positions.clone(),
                input.token_mask.as_ref(),
                direction,
                context,
                None,
                Layer::decoder(index),
                hooks,
                policy.as_ref(),
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
        hidden = hooks.apply(Stack::Decoder.final_norm(), self.final_norm.forward(hidden))?;
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(BackboneOutput {
            hidden,
            hidden_states,
            routers,
        })
    }

    /// Applies an independent or exactly tied vocabulary projection.
    pub(super) fn project(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        match &self.lm_head {
            Some(head) => crate::operations::linear(hidden, head),
            None => Ok(tied_projection(hidden, self.embed_tokens.weight.val())),
        }
    }

    /// Evaluates the stateless, differentiable decoder graph.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_with_hooks(input, options, &mut Hooks::new())
    }

    /// Evaluates a causal sequence with scoped activation capture and intervention.
    ///
    /// Replacements participate in every downstream operation. These hooks apply only
    /// to this stateless call; cached generation continues to use its unchanged graph.
    pub fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_direction_with_hooks(input, options, AttentionDirection::Causal, hooks)
    }

    /// Evaluates full-sequence attention with future tokens visible.
    ///
    /// Padding is still excluded, and a configured trailing window is replaced by
    /// full bidirectional visibility. DeltaNet returns a capability error. Positional
    /// encoding parameters are retained, including any directional relative bias.
    pub fn forward_bidirectional(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_bidirectional_with_hooks(input, options, &mut Hooks::new())
    }

    /// Evaluates a bidirectional sequence with scoped activation interventions.
    ///
    /// This full-sequence experiment does not create or reuse an autoregressive cache.
    pub fn forward_bidirectional_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_direction_with_hooks(input, options, AttentionDirection::Bidirectional, hooks)
    }

    /// Returns final normalized `[batch, token, hidden]` states without vocabulary logits.
    ///
    /// Pool these states or pass them directly to an ordinary Burn task head.
    pub fn forward_hidden(&self, input: CausalLmInput<B>) -> Result<Tensor<B, 3>> {
        self.forward_hidden_with_hooks(input, &mut Hooks::new())
    }

    /// Returns final normalized hidden states after scoped activation interventions.
    pub fn forward_hidden_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        self.forward_hidden_direction_with_hooks(input, AttentionDirection::Causal, hooks)
    }

    /// Returns bidirectional final normalized states without evaluating a vocabulary head.
    ///
    /// Padding remains excluded and trailing windows are replaced. Intrinsically
    /// causal DeltaNet is rejected, as in [`Self::forward_bidirectional`].
    pub fn forward_hidden_bidirectional(&self, input: CausalLmInput<B>) -> Result<Tensor<B, 3>> {
        self.forward_hidden_bidirectional_with_hooks(input, &mut Hooks::new())
    }

    /// Returns bidirectional hidden states after scoped activation interventions.
    pub fn forward_hidden_bidirectional_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        self.forward_hidden_direction_with_hooks(input, AttentionDirection::Bidirectional, hooks)
    }

    /// Runs the selected connectivity policy and returns normalized token representations.
    ///
    /// The hook scope exposes this backbone only; causal recurrent mixers reject
    /// bidirectional execution before processing tokens.
    fn forward_hidden_direction_with_hooks(
        &self,
        input: CausalLmInput<B>,
        direction: AttentionDirection,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        hooks.run(self.activation_points(), |hooks| {
            self.validate_direction(direction)?;
            Ok(self
                .forward_backbone_with_hooks(&input, false, direction, hooks)?
                .hidden)
        })
    }

    /// Rejects bidirectional execution when any block contains causal DeltaNet state.
    fn validate_direction(&self, direction: AttentionDirection) -> Result<()> {
        if direction == AttentionDirection::Bidirectional
            && self
                .spec
                .layers
                .iter()
                .any(|layer| matches!(layer.mixer, crate::architecture::MixerSpec::DeltaNet { .. }))
        {
            return Err(Error::BackendCapability(
                "DeltaNet is a causal recurrent mixer and cannot execute a bidirectional policy"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Applies the selected attention direction and projects the requested token states.
    fn forward_direction_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        direction: AttentionDirection,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        hooks.run(self.activation_points(), |hooks| {
            validate_forward_options(options)?;
            self.validate_direction(direction)?;
            let backbone =
                self.forward_backbone_with_hooks(&input, options.hidden_states, direction, hooks)?;
            self.finish_forward(backbone, &input, options)
        })
    }

    /// Produces optional diagnostics and logits after a completed backbone execution.
    fn finish_forward(
        &self,
        backbone: BackboneOutput<B>,
        input: &CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        let selected = select_hidden(
            backbone.hidden.clone(),
            input.token_mask.as_ref(),
            &options.logits,
        )?;
        let routers = router_diagnostics(&backbone.routers, input.token_mask.as_ref(), options)?;
        Ok(causal_output(
            self.project(selected)?,
            backbone.hidden,
            backbone.hidden_states,
            routers,
            options,
        ))
    }
}

impl ArchitectureConfig for DecoderSpec {
    /// Architecture discriminator for configurable decoder-only graphs.
    const MODEL_TYPES: &'static [&'static str] = &["modular_causal"];

    /// Validates the decoder topology and its executable causal mixer capabilities.
    fn validate(&self) -> Result<()> {
        validate_decoder_execution(self, AttentionDirection::Causal, "custom.causal")
    }
}

impl<B: Backend> CausalLanguageModel<B> for ModularCausalLm<B> {
    /// Serializable decoder specification associated with the causal-model interface.
    type Config = DecoderSpec;
    /// External token-history state used to recompute exact reference predictions.
    type State = ModularDecodeState<B>;

    /// Evaluates the ordinary stateless causal graph through the common model interface.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        ModularCausalLm::forward(self, input, options)
    }

    /// Allocates external token-history state bounded by the configured position capacity.
    ///
    /// The generic modular path recomputes the retained prefix, allowing mixed
    /// architectures without embedding architecture-specific KV caches in parameters.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        if config.capacity() > self.spec.max_position_embeddings {
            return Err(Error::configuration(
                "custom.causal.decode_state.capacity",
                format!(
                    "{} exceeds max_position_embeddings {}",
                    config.capacity(),
                    self.spec.max_position_embeddings
                ),
            ));
        }
        ModularDecodeState::new(
            config,
            batch_size,
            self.spec.clone(),
            StatePurpose::Causal,
            device,
        )
    }

    /// Evaluates and commits a prompt requiring at least one valid token in each row.
    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        state.run_causal(self, input, options, true)
    }

    /// Appends valid tokens, recomputes their retained prefix, and commits only on success.
    fn decode(
        &self,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        state.run_causal(
            self,
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

#[cfg(test)]
#[path = "intervention_tests.rs"]
/// Public activation-hook, staged execution, and consumer composition regressions.
mod intervention_tests;
