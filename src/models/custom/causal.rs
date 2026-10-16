//! Decoder-only modular language model assembly.
//!
//! The model owns only trainable parameters and immutable topology. Autoregressive session
//! history remains in [`ModularDecodeState`], preserving a clean stateless graph for autodiff
//! and model surgery. Every execution mode (full forward, bidirectional forward, staged
//! ranges, cached decode) runs the blocks through the same [`ModularLayer`] path over one
//! [`StackPass`], so positions, masks, and position tables are built once per pass.

use std::ops::Range;

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{Int, Tensor, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error,
    ForwardOptions, Result,
    architecture::{DecoderSpec, HookedModel, StagedModel},
    attention::AttentionPrecisionMode,
    cache::CacheConfig,
    experiment::{AttentionPolicy, HiddenState},
    initialization::InitPolicy,
    intervention::{Hooks, Layer, PointInfo, Stack},
};

use super::{
    common::{
        BackboneOutput, causal_output, embed_tokens, embedded_state, router_diagnostics,
        select_hidden, soft_cap_logits, tied_projection, validate_and_resolve_input,
        validate_forward_options, validate_stack_execution, validate_staged_positions,
    },
    init::scaled_init_policy,
    layer::ModularLayer,
    norm::ModularNorm,
    pass::StackPass,
    state::{ModularDecodeState, StatePurpose},
};

/// Whether a full-sequence causal-model pass keeps its causal structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    /// Structural defaults: causal attention and each block's connectivity.
    Causal,
    /// Every attention block sees all valid tokens; recurrent mixers are rejected.
    Bidirectional,
}

/// A heterogeneous decoder-only transformer assembled from [`DecoderSpec`].
#[derive(Module, Debug)]
pub struct ModularCausalLm<B: Backend> {
    /// Token lookup table `E`; lookups are multiplied by the spec's optional `embedding_scale`.
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
    ///
    /// Parameters use Burn's default initializers: `N(0, 1)` embeddings and Kaiming-uniform
    /// projections. With tied embeddings that makes initial logits scale like
    /// `sqrt(hidden_size)`; train from [`Self::init_with`] and [`Self::scaled_init_policy`]
    /// instead (see [`crate::initialization`]).
    pub fn init(spec: DecoderSpec, device: &B::Device) -> Result<Self> {
        validate_stack_execution(&spec, Stack::Decoder, "custom.causal")?;
        let layers = spec
            .layers
            .iter()
            .map(|layer| ModularLayer::build(layer, spec.hidden_size, None, device))
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

    /// Builds the graph and initializes its parameters with an explicit policy.
    ///
    /// The policy is checked against the model's parameter names before any value changes;
    /// call `B::seed` first for reproducible draws.
    pub fn init_with(spec: DecoderSpec, policy: &InitPolicy, device: &B::Device) -> Result<Self> {
        Ok(policy.apply(Self::init(spec, device)?)?.0)
    }

    /// The recommended random-initialization policy for this topology.
    ///
    /// Embeddings and projections draw from `N(0, std^2)` with zero biases (Hugging Face's
    /// `initializer_range`, typically `0.02`), and projections that write into the residual
    /// stream (`o_proj`, `down_proj`, DeltaNet and Mamba-2 `out_proj`, present in this spec)
    /// are scaled by `1/sqrt(2 * layers)` as in GPT-2. Normalization scales, T5 biases,
    /// DeltaNet and Mamba-2 decay and step parameters, convolutions, and other non-projection
    /// parameters keep their constructed values.
    #[must_use]
    pub fn scaled_init_policy(spec: &DecoderSpec, std: f64) -> InitPolicy {
        scaled_init_policy([spec], spec.layers.len(), false, std)
    }

    /// Selects the numerical execution path for conventional projected self-attention layers.
    ///
    /// [`AttentionPrecisionMode::EagerFp32Softmax`] enables score and probability
    /// capture/edit points, listed by [`Self::activation_points`]. This reference path
    /// materializes attention matrices with quadratic sequence memory cost; select
    /// it explicitly for experiments that need those points. Initialization uses
    /// [`AttentionPrecisionMode::BackendDefault`] unless this method is called.
    /// DeltaNet and Mamba-2 keep their recurrent computation; MLA retains its explicit eager
    /// graph.
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

    /// Lists the exact activations that this model can observe or replace.
    ///
    /// Residual points have shape `[batch, token, hidden]`. Dense feed-forward
    /// pre/activation points expose the gate (or up) projection before/after its nonlinearity.
    /// MoE layers expose only MLP input/output, and attention-only blocks no MLP point.
    /// Attention input/output surround the configured mixer, including DeltaNet or Mamba-2
    /// when that layer uses a recurrent mixer.
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
    ///
    /// Token, mask, and position values are validated with one device read.
    pub fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.embedding().info()], |hooks| {
            let (positions, evaluated_context) = validate_and_resolve_input(
                &input,
                self.spec.vocabulary_size,
                self.spec.max_position_embeddings,
                "custom.embed",
            )?;
            let hidden = hooks.apply(Stack::Decoder.embedding(), self.lookup(input.token_ids))?;
            embedded_state(hidden, positions, input.token_mask, evaluated_context, None)
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
    /// Position values are read once per call. Use separate hook collections when
    /// retaining captures across separate stages.
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
            validate_staged_positions(
                &state.context,
                self.spec.max_position_embeddings,
                "custom.run_layers.positions",
            )?;
            for index in range.clone() {
                self.layers[index].validate_prevalidated(
                    &state.hidden,
                    &state.context,
                    Layer::decoder(index),
                )?;
            }
            let pass =
                StackPass::from_context(&state.context, Stack::Decoder, state.hidden.dtype());
            for index in range.clone() {
                let layer = Layer::decoder(index);
                // Hooks may replace hidden states between blocks; re-check them sync-free.
                self.layers[index].validate_prevalidated(&state.hidden, &state.context, layer)?;
                state.hidden = self.layers[index]
                    .forward(state.hidden, &pass, layer, hooks)?
                    .0;
            }
            state.next_layer = range.end;
            Ok(state)
        })
    }

    /// Checks `[batch, token, hidden]` state compatibility before staged execution.
    ///
    /// Width, dtype, and device must match embeddings, the context layout must match the
    /// hidden grid, the cursor must fit the block list, and attention overrides must address
    /// this decoder without cross-attention. No device value is read.
    fn validate_execution_state(&self, state: &HiddenState<B>) -> Result<()> {
        let [batch, tokens, width] = state.hidden.dims();
        if width != self.spec.hidden_size || batch == 0 || tokens == 0 {
            return Err(Error::shape(
                "execution.hidden",
                format!("nonempty [batch, token, {}]", self.spec.hidden_size),
                format!("{:?}", state.hidden.dims()),
            ));
        }
        state
            .context
            .validate_layout([batch, tokens], &state.hidden.device())?;
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
            Direction::Causal,
            &mut Hooks::new(),
        )
    }

    /// Executes the chosen full-sequence graph inside its caller's intervention scope.
    fn forward_backbone_with_hooks(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
        direction: Direction,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<BackboneOutput<B>> {
        let (positions, evaluated_context) = validate_and_resolve_input(
            input,
            self.spec.vocabulary_size,
            self.spec.max_position_embeddings,
            "custom.causal",
        )?;
        let hidden = self.lookup(input.token_ids.clone());
        let mut pass = StackPass::new(
            positions,
            input.token_mask.as_ref(),
            evaluated_context,
            Stack::Decoder,
            hidden.dtype(),
        )
        .with_row_invariant_positions(input.position_ids.is_none() && input.token_mask.is_none());
        if direction == Direction::Bidirectional {
            pass = pass.with_self_attention_policy(AttentionPolicy::bidirectional());
        }
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
        for (index, layer) in self.layers.iter().enumerate() {
            let (next, router) = layer.forward(hidden, &pass, Layer::decoder(index), hooks)?;
            hidden = next;
            routers.extend(router);
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

    /// Looks up `[batch, token]` IDs and applies the spec's optional embedding scale.
    ///
    /// Every execution mode embeds through this method, so the stateless,
    /// staged, and cached paths see the same layer-0 input.
    fn lookup(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        embed_tokens(&self.embed_tokens, token_ids, self.spec.embedding_scale)
    }

    /// Applies an independent or exactly tied vocabulary projection, then the optional soft cap.
    ///
    /// A tied projection reads the unscaled embedding table; the spec's
    /// `final_logit_softcap` `c` maps each logit `z` to `c · tanh(z / c)`.
    pub(super) fn project(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let logits = match &self.lm_head {
            Some(head) => crate::operations::linear(hidden, head),
            None => tied_projection(hidden, self.embed_tokens.weight.val()),
        }?;
        Ok(soft_cap_logits(logits, self.spec.final_logit_softcap))
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
        self.forward_direction_with_hooks(input, options, Direction::Causal, hooks)
    }

    /// Evaluates full-sequence attention with future tokens visible.
    ///
    /// Padding is still excluded, and every block's structural connectivity (trailing
    /// windows, local chunks) is replaced by full bidirectional visibility. A causal
    /// recurrent mixer (DeltaNet, Mamba-2) returns a capability error. Positional encoding parameters are retained,
    /// including any directional relative bias.
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
        self.forward_direction_with_hooks(input, options, Direction::Bidirectional, hooks)
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
        self.forward_hidden_direction_with_hooks(input, Direction::Causal, hooks)
    }

    /// Returns bidirectional final normalized states without evaluating a vocabulary head.
    ///
    /// Padding remains excluded and structural connectivity is replaced. Intrinsically
    /// causal DeltaNet and Mamba-2 are rejected, as in [`Self::forward_bidirectional`].
    pub fn forward_hidden_bidirectional(&self, input: CausalLmInput<B>) -> Result<Tensor<B, 3>> {
        self.forward_hidden_bidirectional_with_hooks(input, &mut Hooks::new())
    }

    /// Returns bidirectional hidden states after scoped activation interventions.
    pub fn forward_hidden_bidirectional_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        self.forward_hidden_direction_with_hooks(input, Direction::Bidirectional, hooks)
    }

    /// Runs the selected connectivity policy and returns normalized token representations.
    ///
    /// The hook scope exposes this backbone only; causal recurrent mixers reject
    /// bidirectional execution before processing tokens.
    fn forward_hidden_direction_with_hooks(
        &self,
        input: CausalLmInput<B>,
        direction: Direction,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        hooks.run(self.activation_points(), |hooks| {
            self.validate_direction(direction)?;
            Ok(self
                .forward_backbone_with_hooks(&input, false, direction, hooks)?
                .hidden)
        })
    }

    /// Rejects bidirectional execution when any block contains a causal recurrent mixer.
    fn validate_direction(&self, direction: Direction) -> Result<()> {
        if direction == Direction::Bidirectional
            && self
                .spec
                .layers
                .iter()
                .any(|layer| layer.mixer.is_recurrent())
        {
            return Err(Error::BackendCapability(
                "DeltaNet and Mamba2 are causal recurrent mixers and cannot execute a bidirectional policy"
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
        direction: Direction,
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
        validate_stack_execution(self, Stack::Decoder, "custom.causal")
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

impl<B: Backend> HookedModel<B> for ModularCausalLm<B> {
    /// Complete causal-model input.
    type Input = CausalLmInput<B>;
    /// Complete causal-model output.
    type Output = CausalLmOutput<B>;

    /// Every point of the causal forward pass, in emission order.
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        Ok(ModularCausalLm::activation_points(self))
    }

    /// Runs [`ModularCausalLm::forward_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        ModularCausalLm::forward_with_hooks(self, input, options, hooks)
    }
}

impl<B: Backend> StagedModel<B> for ModularCausalLm<B> {
    /// Number of decoder blocks.
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// The decoder stack.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Runs [`ModularCausalLm::embed_with_hooks`].
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        ModularCausalLm::embed_with_hooks(self, input, hooks)
    }

    /// Runs [`ModularCausalLm::run_layers_with_hooks`].
    fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        ModularCausalLm::run_layers_with_hooks(self, state, range, hooks)
    }

    /// Runs [`ModularCausalLm::normalize_with_hooks`].
    fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        ModularCausalLm::normalize_with_hooks(self, state, hooks)
    }
}

#[cfg(test)]
#[path = "intervention_tests.rs"]
/// Public activation-hook, staged execution, and consumer composition regressions.
mod intervention_tests;

#[cfg(test)]
#[path = "causal_tests.rs"]
/// Hook propagation across every mixer kind, staged-state checks, and trait entry points.
mod tests;
