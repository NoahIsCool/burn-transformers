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

use super::{INPUT_NAMES, LlamaForCausalLm, sequence::SequenceKeys};

impl<B: Backend> LlamaForCausalLm<B> {
    /// Lists every activation point of a complete stateless pass, in emission order.
    #[must_use]
    pub fn activation_points(&self) -> Vec<PointInfo> {
        self.model.activation_points()
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
        let validity =
            self.padding_validity(input.token_mask.as_ref().map(|mask| mask.as_tensor()))?;
        hooks.run_with_validity(self.activation_points(), validity, |hooks| {
            if options.attentions {
                return Err(Error::BackendCapability(
                    "request attention probabilities through the attention.pattern hook point inside a hooked pass".into(),
                ));
            }
            let backbone = self.model.forward(&input, options.hidden_states, hooks)?;
            self.finish(input, options, backbone)
        })
    }

    /// Runs the stateless pass and returns every layer's attention probabilities.
    ///
    /// Probabilities are `[batch, H, token, token]` patterns captured at the
    /// `attention.pattern` points, which exist only under eager attention.
    pub(super) fn forward_capturing_attentions(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        let eager = self.model.layers.iter().all(|layer| {
            layer.self_attn.precision_mode == AttentionPrecisionMode::EagerFp32Softmax
        });
        if !eager {
            return Err(Error::BackendCapability(
                "Llama-family attention probabilities require with_attention_precision(EagerFp32Softmax); fused kernels do not materialize them".into(),
            ));
        }
        let mut hooks = Hooks::new();
        let captures = (0..self.model.layers.len())
            .map(|index| hooks.capture(Layer::decoder(index).self_attention().pattern()))
            .collect::<Vec<_>>();
        let options = ForwardOptions {
            attentions: false,
            ..options.clone()
        };
        let mut output = self.forward_with_hooks(input, &options, &mut hooks)?;
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
    fn padding_validity(&self, mask: Option<&Tensor<B, 2, Bool>>) -> Result<Vec<PointValidity<B>>> {
        let Some(mask) = mask else {
            return Ok(Vec::new());
        };
        let [batch, tokens] = mask.dims();
        if batch == 0
            || tokens == 0
            || mask.device() != self.model.embed_tokens.weight.lazy_device()
        {
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
        for (index, layer) in self.model.layers.iter().enumerate() {
            let point = Layer::decoder(index);
            for hidden_point in [
                point.residual_pre(),
                point.attention_input(),
                point.attention_output(),
                point.mlp_input(),
                point.mlp_pre_activation(),
                point.mlp_activation(),
                point.mlp_output(),
                point.residual_post(),
            ] {
                validity.push(PointValidity::new(hidden_point, hidden.clone())?);
            }
            let eager = layer.self_attn.precision_mode == AttentionPrecisionMode::EagerFp32Softmax;
            attention_validity(&mut validity, point.self_attention(), &heads, &plane, eager)?;
        }
        Ok(validity)
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
        hooks.run(vec![Stack::Decoder.embedding().info()], |hooks| {
            let device = self.model.embed_tokens.weight.lazy_device();
            let validated = validate_input(&input, &device, &self.model.input_contract())?;
            let positions = validated.require_positions(INPUT_NAMES.position_ids)?;
            let hidden = hooks.apply(
                Stack::Decoder.embedding(),
                self.model.embed_tokens.forward(input.token_ids),
            )?;
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
    pub fn run_layers_with_hooks(
        &self,
        mut state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        let points = range
            .clone()
            .filter_map(|index| self.model.layers.get(index).map(|layer| (index, layer)))
            .flat_map(|(index, layer)| layer.activation_points(index))
            .collect();
        hooks.run(points, |hooks| {
            self.validate_staged_state(&state)?;
            state.validate_range(range.clone(), self.model.layers.len())?;
            let context = &state.context;
            let angles = self
                .model
                .rotary_tables(context.positions.clone(), context.evaluated_context)?;
            let policies = (0..self.model.layers.len())
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
                &self.model,
                angles,
                context.token_mask.as_ref(),
                &policies,
                context.positions.dims(),
            )?;
            for index in range.clone() {
                state.hidden =
                    self.model.layers[index].forward(state.hidden, index, &mut keys, hooks)?;
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
    pub fn normalize_with_hooks(
        &self,
        mut state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.final_norm().info()], |hooks| {
            self.validate_staged_state(&state)?;
            if state.normalized {
                return Err(Error::configuration(
                    "llama.normalize",
                    "final normalization has already been applied",
                ));
            }
            state.hidden = hooks.apply(
                Stack::Decoder.final_norm(),
                self.model.norm.forward(state.hidden),
            )?;
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
    fn validate_staged_state(&self, state: &HiddenState<B>) -> Result<()> {
        let architecture = self.architecture();
        state.validate(architecture.hidden_size)?;
        let embedding = self.model.embed_tokens.weight.val();
        if state.hidden.dtype() != embedding.dtype() || state.hidden.device() != embedding.device()
        {
            return Err(Error::configuration(
                "llama.execution.hidden",
                "hidden states must match the embedding dtype and device",
            ));
        }
        if state.context.encoder.is_some()
            || state.context.cross_attention.is_some()
            || !state.context.layer_cross_attention.is_empty()
        {
            return Err(Error::configuration(
                "llama.execution.cross_attention",
                "the Llama family is decoder-only and has no cross-attention",
            ));
        }
        if state.context.layer_attention.iter().any(|(layer, _)| {
            layer.stack() != Stack::Decoder || layer.index() >= self.model.layers.len()
        }) {
            return Err(Error::configuration(
                "llama.execution.layer_attention",
                "attention overrides must address existing decoder layers",
            ));
        }
        if state.context.evaluated_context > architecture.max_position_embeddings {
            return Err(Error::configuration(
                "llama.execution.evaluated_context",
                format!(
                    "{} exceeds max_position_embeddings {}",
                    state.context.evaluated_context, architecture.max_position_embeddings
                ),
            ));
        }
        Ok(())
    }
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
