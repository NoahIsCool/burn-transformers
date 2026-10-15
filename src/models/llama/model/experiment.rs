//! Activation hooks and staged execution of the stateless graph.
//!
//! Every Llama-family model implements [`HookedModel`] and [`StagedModel`].
//! Both run the same block code as [`LlamaForCausalLm::forward`]; hooks
//! observe or edit the points listed by [`LlamaForCausalLm::activation_points`]:
//!
//! ```text
//! decoder.embedding
//! per layer l: residual.pre, attention.input, attention.{query, key, value},
//!              [attention.{scores, pattern} under eager attention],
//!              attention.head_context, attention.output, mlp.input,
//!              mlp.pre_activation, mlp.activation, mlp.output, residual.post
//! decoder.final_norm
//! ```
//!
//! Queries and keys are observed after Q/K normalization and rotation; keys
//! and values keep their native KV heads. With padding, every point carries
//! a validity mask so padded entries read as zero and edits cannot revive
//! them (and a padded value can never reach a valid query).
//!
//! Staged execution exchanges [`HiddenState`] values: `embed` validates the
//! input once and records positions, the evaluated rotary context, and the
//! mask; `run_layers` runs any contiguous block range; `normalize` applies the
//! final norm once. An [`AttentionPolicy`](crate::experiment::AttentionPolicy)
//! in the state's context replaces a layer's causal/sliding rule (padding
//! stays excluded and the policy's bias is added); cross-attention contexts
//! are rejected because the family is decoder-only.
//!
//! # Shared implementation
//!
//! The functions of this module are generic over the crate-private
//! `DecoderGraph` interface, so every dense decoder built on it (the Llama
//! and Gemma families) gets the same catalog, validity masks, and staged
//! semantics; the inherent methods and trait implementations of each family
//! only delegate.

use core::ops::Range;

use burn::tensor::{Bool, Tensor, backend::Backend};

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::{HookedModel, StagedModel},
    attention::AttentionPrecisionMode,
    experiment::{ExecutionContext, HiddenState},
    intervention::{AttentionPoints, Hooks, Layer, PointInfo, PointValidity, Stack},
    models::shared::validate_input,
};

use super::{
    LlamaForCausalLm,
    graph::{DecoderBlock, DecoderGraph, HIDDEN_POINTS},
    sequence::{SequenceKeys, backbone, finish},
};

impl<B: Backend> LlamaForCausalLm<B> {
    /// Lists every activation point of a complete stateless pass, in emission order.
    #[must_use]
    pub fn activation_points(&self) -> Vec<PointInfo> {
        activation_points(self)
    }

    /// Runs the stateless forward pass inside one hook scope.
    ///
    /// Captures and edits act on the points of [`Self::activation_points`];
    /// edits change every later computation and the returned logits. With a
    /// token mask, padded entries of every point are zero and stay zero.
    ///
    /// # Errors
    ///
    /// Returns hook registration errors before computation, input errors, and
    /// [`Error::Intervention`] for failing callbacks.
    pub fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        forward_with_hooks(self, input, options, hooks)
    }

    /// Embeds tokens into a raw state before layer zero, in a scope of the embedding point.
    ///
    /// # Errors
    ///
    /// Returns input validation errors.
    pub fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        embed_with_hooks(self, input, hooks)
    }

    /// Runs blocks `range`, which must start at `state.next_layer`, in one hook scope.
    ///
    /// # Errors
    ///
    /// Returns state, range, or position errors, and hook errors.
    pub fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        run_layers_with_hooks(self, state, range, hooks)
    }

    /// Applies the final norm once, in a scope of the final-norm point.
    ///
    /// # Errors
    ///
    /// Returns state errors and hook errors.
    pub fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        normalize_with_hooks(self, state, hooks)
    }
}

// The shared stateless entry points.

/// Evaluates the stateless forward pass, capturing attention probabilities on request.
///
/// `options.attentions` requires eager attention in every block; it then
/// runs one hooked pass that captures every `attention.pattern` point.
///
/// # Errors
///
/// Returns the errors of the backbone and of [`forward_capturing_attentions`].
pub(crate) fn forward<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    input: CausalLmInput<B>,
    options: &ForwardOptions,
) -> Result<CausalLmOutput<B>> {
    if options.attentions {
        return forward_capturing_attentions(graph, input, options);
    }
    let backbone = backbone(graph, &input, options.hidden_states, &mut Hooks::new())?;
    finish(graph, input, options, backbone)
}

/// Lists every point of a complete pass, in emission order.
pub(crate) fn activation_points<B: Backend, G: DecoderGraph<B>>(graph: &G) -> Vec<PointInfo> {
    let mut points = vec![Stack::Decoder.embedding().info()];
    for (index, block) in graph.blocks().iter().enumerate() {
        points.extend(block.block_points(index));
    }
    points.push(Stack::Decoder.final_norm().info());
    points
}

/// Runs the stateless forward pass inside one hook scope, with padding validity.
///
/// # Errors
///
/// Returns hook registration errors before computation, input errors, a
/// capability error for `options.attentions` (patterns are captured through
/// the `attention.pattern` points instead), and callback errors.
pub(crate) fn forward_with_hooks<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    input: CausalLmInput<B>,
    options: &ForwardOptions,
    hooks: &mut Hooks<'_, B>,
) -> Result<CausalLmOutput<B>> {
    let validity = padding_validity(
        graph,
        input.token_mask.as_ref().map(|mask| mask.as_tensor()),
    )?;
    hooks.run_with_validity(activation_points(graph), validity, |hooks| {
        if options.attentions {
            return Err(Error::BackendCapability(
                "request attention probabilities through the attention.pattern hook point inside a hooked pass".into(),
            ));
        }
        let backbone = backbone(graph, &input, options.hidden_states, hooks)?;
        finish(graph, input, options, backbone)
    })
}

/// Runs the stateless pass and returns every layer's attention probabilities.
///
/// Probabilities are `[batch, H, token, token]` patterns captured at the
/// `attention.pattern` points, which exist only under eager attention.
fn forward_capturing_attentions<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    input: CausalLmInput<B>,
    options: &ForwardOptions,
) -> Result<CausalLmOutput<B>> {
    let eager = graph
        .blocks()
        .iter()
        .all(|block| block.attention_precision() == AttentionPrecisionMode::EagerFp32Softmax);
    if !eager {
        return Err(Error::BackendCapability(format!(
            "{} attention probabilities require with_attention_precision(EagerFp32Softmax); fused kernels do not materialize them",
            G::LABELS.family
        )));
    }
    let mut hooks = Hooks::new();
    let captures = (0..graph.blocks().len())
        .map(|index| hooks.capture(Layer::decoder(index).self_attention().pattern()))
        .collect::<Vec<_>>();
    let options = ForwardOptions {
        attentions: false,
        ..options.clone()
    };
    let mut output = forward_with_hooks(graph, input, &options, &mut hooks)?;
    output.attentions = captures
        .iter()
        .map(|capture| hooks.get(capture))
        .collect::<Result<Vec<_>>>()?;
    Ok(output)
}

/// Builds zeroing validity masks for every point of a padded batch.
///
/// Hidden points use `[batch, token, 1]`; queries and head contexts
/// `[batch, 1, token, 1]`; keys and values likewise over key tokens; and
/// eager score/pattern planes `[batch, 1, query, key]`. A malformed mask
/// yields no validity so input validation reports it inside the scope.
fn padding_validity<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    mask: Option<&Tensor<B, 2, Bool>>,
) -> Result<Vec<PointValidity<B>>> {
    let Some(mask) = mask else {
        return Ok(Vec::new());
    };
    let [batch, tokens] = mask.dims();
    if batch == 0 || tokens == 0 || mask.device() != graph.parameter_device() {
        return Ok(Vec::new());
    }
    let valid = mask.clone().int().bool();
    let hidden = valid.clone().unsqueeze_dim::<3>(2);
    let heads = valid.clone().reshape([batch, 1, tokens, 1]);
    let plane = heads.clone().bool_and(valid.reshape([batch, 1, 1, tokens]));
    let mut validity = vec![
        PointValidity::new(Stack::Decoder.embedding(), hidden.clone())?,
        PointValidity::new(Stack::Decoder.final_norm(), hidden.clone())?,
    ];
    for (index, block) in graph.blocks().iter().enumerate() {
        let point = Layer::decoder(index);
        for hidden_point in HIDDEN_POINTS {
            validity.push(PointValidity::new(hidden_point(point), hidden.clone())?);
        }
        let eager = block.attention_precision() == AttentionPrecisionMode::EagerFp32Softmax;
        attention_validity(&mut validity, point.self_attention(), &heads, &plane, eager)?;
    }
    Ok(validity)
}

// The shared staged entry points.

/// Embeds tokens into a raw state before layer zero, in a scope of the embedding point.
///
/// # Errors
///
/// Returns input validation errors.
pub(crate) fn embed_with_hooks<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    input: CausalLmInput<B>,
    hooks: &mut Hooks<'_, B>,
) -> Result<HiddenState<B>> {
    hooks.run(vec![Stack::Decoder.embedding().info()], |hooks| {
        let device = graph.parameter_device();
        let validated = validate_input(&input, &device, &graph.input_contract())?;
        let positions = validated.require_positions(G::LABELS.inputs.position_ids)?;
        let hidden = hooks.apply(Stack::Decoder.embedding(), graph.embed(input.token_ids))?;
        let context = ExecutionContext {
            positions,
            token_mask: input.token_mask,
            evaluated_context: validated.evaluated_context,
            attention: None,
            layer_attention: Vec::new(),
            cross_attention: None,
            layer_cross_attention: Vec::new(),
            encoder: None,
        };
        Ok(HiddenState {
            hidden,
            context,
            next_layer: 0,
            normalized: false,
        })
    })
}

/// Runs blocks `range`, which must start at `state.next_layer`, in one hook scope.
///
/// # Errors
///
/// Returns state, range, or position errors, and hook errors.
pub(crate) fn run_layers_with_hooks<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    mut state: HiddenState<B>,
    range: Range<usize>,
    hooks: &mut Hooks<'_, B>,
) -> Result<HiddenState<B>> {
    let blocks = graph.blocks();
    let points = range
        .clone()
        .filter_map(|index| blocks.get(index).map(|block| (index, block)))
        .flat_map(|(index, block)| block.block_points(index))
        .collect();
    hooks.run(points, |hooks| {
        validate_staged_state(graph, &state)?;
        state.validate_range(range.clone(), blocks.len())?;
        let context = &state.context;
        let angles = graph.rotary_tables(context.positions.clone(), context.evaluated_context)?;
        let policies = (0..blocks.len())
            .map(|index| {
                let layer = Layer::decoder(index);
                let overridden = context.attention.is_some()
                    || context
                        .layer_attention
                        .iter()
                        .any(|(target, _)| *target == layer);
                overridden.then(|| context.attention_for(layer, true))
            })
            .collect::<Vec<_>>();
        let mut keys = SequenceKeys::with_policies(
            graph,
            angles,
            context.token_mask.as_ref(),
            &policies,
            context.positions.dims(),
        )?;
        for index in range.clone() {
            state.hidden = blocks[index].forward(state.hidden, index, &mut keys, hooks)?;
        }
        state.next_layer = range.end;
        Ok(state)
    })
}

/// Applies the final norm once, in a scope of the final-norm point.
///
/// # Errors
///
/// Returns state errors and hook errors.
pub(crate) fn normalize_with_hooks<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    mut state: HiddenState<B>,
    hooks: &mut Hooks<'_, B>,
) -> Result<HiddenState<B>> {
    hooks.run(vec![Stack::Decoder.final_norm().info()], |hooks| {
        validate_staged_state(graph, &state)?;
        if state.normalized {
            return Err(Error::configuration(
                G::LABELS.field("normalize"),
                "final normalization has already been applied",
            ));
        }
        state.hidden = hooks.apply(Stack::Decoder.final_norm(), graph.final_norm(state.hidden))?;
        state.normalized = true;
        Ok(state)
    })
}

/// Checks a staged state's width, dtype, device, positions, and decoder-only context.
///
/// [`HiddenState::validate`] reads positions back once and checks
/// `0 <= p < evaluated_context`, because callers may have edited them;
/// the host check `evaluated_context <= max_position_embeddings` then
/// bounds both the positions and the context that context-dependent RoPE
/// is evaluated at, exactly as the stateless pass's input contract does.
fn validate_staged_state<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    state: &HiddenState<B>,
) -> Result<()> {
    let labels = G::LABELS;
    let geometry = graph.geometry();
    state.validate(geometry.hidden_size)?;
    let embedding = graph.embedding().weight.val();
    if state.hidden.dtype() != embedding.dtype() || state.hidden.device() != embedding.device() {
        return Err(Error::configuration(
            labels.field("execution.hidden"),
            "hidden states must match the embedding dtype and device",
        ));
    }
    if state.context.encoder.is_some()
        || state.context.cross_attention.is_some()
        || !state.context.layer_cross_attention.is_empty()
    {
        return Err(Error::configuration(
            labels.field("execution.cross_attention"),
            format!(
                "the {} model is decoder-only and has no cross-attention",
                labels.family
            ),
        ));
    }
    let layers = graph.blocks().len();
    if state
        .context
        .layer_attention
        .iter()
        .any(|(layer, _)| layer.stack() != Stack::Decoder || layer.index() >= layers)
    {
        return Err(Error::configuration(
            labels.field("execution.layer_attention"),
            "attention overrides must address existing decoder layers",
        ));
    }
    if state.context.evaluated_context > geometry.max_position_embeddings {
        return Err(Error::configuration(
            labels.field("execution.evaluated_context"),
            format!(
                "{} exceeds max_position_embeddings {}",
                state.context.evaluated_context, geometry.max_position_embeddings
            ),
        ));
    }
    Ok(())
}

/// Adds validity masks for one layer's attention points.
fn attention_validity<B: Backend>(
    validity: &mut Vec<PointValidity<B>>,
    points: AttentionPoints,
    heads: &Tensor<B, 4, Bool>,
    plane: &Tensor<B, 4, Bool>,
    eager: bool,
) -> Result<()> {
    validity.extend([
        PointValidity::new(points.query(), heads.clone())?,
        PointValidity::new(points.key(), heads.clone())?,
        PointValidity::new(points.value(), heads.clone())?,
        PointValidity::new(points.head_context(), heads.clone())?,
    ]);
    if eager {
        validity.extend([
            PointValidity::new(points.scores(), plane.clone())?,
            PointValidity::new(points.pattern(), plane.clone())?,
        ]);
    }
    Ok(())
}

impl<B: Backend> HookedModel<B> for LlamaForCausalLm<B> {
    /// Token IDs, optional mask, and optional positions.
    type Input = CausalLmInput<B>;
    /// Logits and requested diagnostics.
    type Output = CausalLmOutput<B>;

    /// Delegates to [`LlamaForCausalLm::activation_points`].
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        Ok(LlamaForCausalLm::activation_points(self))
    }

    /// Delegates to [`LlamaForCausalLm::forward_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        LlamaForCausalLm::forward_with_hooks(self, input, options, hooks)
    }
}

impl<B: Backend> StagedModel<B> for LlamaForCausalLm<B> {
    /// Returns the number of decoder blocks.
    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    /// The family is decoder-only.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Delegates to [`LlamaForCausalLm::embed_with_hooks`].
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        LlamaForCausalLm::embed_with_hooks(self, input, hooks)
    }

    /// Delegates to [`LlamaForCausalLm::run_layers_with_hooks`].
    fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        LlamaForCausalLm::run_layers_with_hooks(self, state, range, hooks)
    }

    /// Delegates to [`LlamaForCausalLm::normalize_with_hooks`].
    fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        LlamaForCausalLm::normalize_with_hooks(self, state, hooks)
    }
}
