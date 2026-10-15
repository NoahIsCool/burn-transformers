//! Activation hooks and staged execution for Muse-Glimmer.
//!
//! # Activation points
//!
//! [`MuseGlimmerForCausalLm::activation_points`] lists, in emission order:
//!
//! 1. `decoder.embedding` — the scale-free-normalized token embeddings;
//! 2. per block `i`: `residual.pre`; `attention.input` (after the input
//!    sandwich norm); `attention.query` `[batch, query_head, token, channel]`
//!    and the native-head `attention.key`/`attention.value`
//!    `[batch, kv_head, token, channel]`, after Q/K normalization and RoPE;
//!    `attention.head_context` before gating; `attention.gate_logits` and the
//!    activated `attention.gate` `[batch, token, query_heads * head_dim]`;
//!    `attention.output` (output projection); `post_attention_norm`;
//!    `mlp.input`; `mlp.pre_activation` and `mlp.activation` (the SwiGLU gate
//!    branch before and after SiLU); `mlp.output`; `post_feedforward_norm`;
//!    and `residual.post`;
//! 3. `decoder.final_norm`.
//!
//! Attention keeps its bounded chunked execution under hooks: edits act on
//! the projected tensors before attention, and the head context is joined
//! from its chunks before its point is emitted. Scores and patterns are not
//! exposed, because the F32 path dispatches backend attention kernels.
//!
//! # Staged execution
//!
//! [`StagedModel`] runs the stack in resumable stages. A layer without an
//! attention override uses its own connectivity (causal, plus the window on
//! sliding layers). An override from [`ExecutionContext::attention_for`]
//! replaces both: `causal` selects bottom-right causality, `allowed`
//! restricts links, and `bias` adds to the scaled scores; padding always
//! applies, and the layer evaluates one full score plane. Cached decoding is
//! not hookable.

use core::ops::Range;

use burn::tensor::backend::Backend;

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::{HookedModel, StagedModel},
    attention::{grouped_query_attention, padding_attention_mask, union_exclusions},
    experiment::{ExecutionContext, HiddenState},
    intervention::{Hooks, Layer, PointInfo, Stack},
    models::shared::{DeferredChecks, check_nonempty_mask, ensure_same_device},
};

use super::{
    INPUT_NAMES, LayerInputs, MuseGlimmerDecoderLayer, MuseGlimmerForCausalLm, MuseGlimmerModel,
    PARAMETERS, muse_attention_options, muse_attention_precision,
};

impl<B: Backend> MuseGlimmerForCausalLm<B> {
    /// Lists every activation point in emission order (module documentation).
    #[must_use]
    pub fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Decoder.embedding().info()];
        for (index, layer) in self.model.layers.iter().enumerate() {
            points.extend(layer.activation_points(Layer::decoder(index)));
        }
        points.push(Stack::Decoder.final_norm().info());
        points
    }

    /// Evaluates the stateless model while capturing or editing activations.
    ///
    /// Edits participate in every later computation and stay in the autodiff
    /// graph; hooks are validated against [`Self::activation_points`] first.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::forward`], hook-scope validation errors,
    /// and the errors of registered actions.
    pub fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Muse-Glimmer attention-probability capture is not exposed by its bounded execution path"
                    .into(),
            ));
        }
        hooks.run(self.activation_points(), |hooks| {
            let (hidden, hidden_states) =
                self.model
                    .forward(&input, options.hidden_states, Some(hooks))?;
            self.finish_forward(&input, options, hidden, hidden_states)
        })
    }
}

impl<B: Backend> HookedModel<B> for MuseGlimmerForCausalLm<B> {
    /// Complete causal-language-model input.
    type Input = CausalLmInput<B>;
    /// Complete causal-language-model output.
    type Output = CausalLmOutput<B>;

    /// Delegates to [`MuseGlimmerForCausalLm::activation_points`].
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        Ok(Self::activation_points(self))
    }

    /// Delegates to [`MuseGlimmerForCausalLm::forward_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: Self::Input,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Self::Output> {
        Self::forward_with_hooks(self, input, options, hooks)
    }
}

impl<B: Backend> StagedModel<B> for MuseGlimmerForCausalLm<B> {
    /// Number of decoder blocks.
    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    /// Muse blocks live in the decoder stack.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Validates the input, embeds and normalizes its tokens, and emits the embedding point.
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.embedding().info()], |hooks| {
            let validated = self.model.validate_public_input(&input)?;
            let positions = validated.require_positions(INPUT_NAMES.position_ids)?;
            let context = ExecutionContext::new(positions, input.token_mask.clone())?;
            let hidden = hooks.apply(
                Stack::Decoder.embedding(),
                self.model.embed(input.token_ids)?,
            )?;
            HiddenState::new(hidden, context)
        })
    }

    /// Runs blocks `range` with the points of exactly those blocks in scope.
    fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        state.validate_range(range.clone(), self.model.layers.len())?;
        let points = range
            .clone()
            .flat_map(|index| self.model.layers[index].activation_points(Layer::decoder(index)))
            .collect();
        hooks.run(points, |hooks| self.model.run_layers(state, range, hooks))
    }

    /// Applies the final learned RMS norm once and emits the final-norm point.
    fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.final_norm().info()], |hooks| {
            let mut state = state;
            self.model.check_staged_hidden(&state)?;
            if state.normalized {
                return Err(Error::configuration(
                    "muse_glimmer.execution.normalize",
                    "the state is already normalized",
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
}

impl<B: Backend> MuseGlimmerModel<B> {
    /// Requires a staged state on the model's device and activation dtype.
    fn check_staged_hidden(&self, state: &HiddenState<B>) -> Result<()> {
        state.validate(self.config.hidden_size)?;
        ensure_same_device(
            "muse_glimmer.execution.hidden",
            &state.hidden.device(),
            &self.embed_tokens.weight.lazy_device(),
            PARAMETERS,
        )?;
        let expected = self.embed_tokens.weight.val().dtype();
        if state.hidden.dtype() != expected {
            return Err(Error::configuration(
                "muse_glimmer.execution.hidden",
                format!(
                    "hidden states must use the model dtype {expected:?}, found {:?}",
                    state.hidden.dtype()
                ),
            ));
        }
        Ok(())
    }

    /// Validates a staged state and its policies, then executes blocks `range`.
    ///
    /// Every check completes before any block runs; the returned cursor names
    /// the first unexecuted block.
    fn run_layers(
        &self,
        mut state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.check_staged_hidden(&state)?;
        state.validate_range(range.clone(), self.layers.len())?;
        let context = &state.context;
        if context.encoder.is_some()
            || context.cross_attention.is_some()
            || !context.layer_cross_attention.is_empty()
        {
            return Err(Error::configuration(
                "muse_glimmer.execution.encoder",
                "Muse decoder blocks do not define cross-attention",
            ));
        }
        for (address, _) in &context.layer_attention {
            if !(0..self.layers.len()).any(|index| *address == Layer::decoder(index)) {
                return Err(Error::configuration(
                    "muse_glimmer.execution.layer_attention",
                    "override addresses an unavailable Muse decoder layer",
                ));
            }
        }
        let [batch, sequence, _] = state.hidden.dims();
        let device = state.hidden.device();
        let heads = self.config.num_attention_heads;
        let mut checks = DeferredChecks::new();
        check_nonempty_mask(
            &mut checks,
            context.token_mask.as_ref(),
            batch,
            sequence,
            INPUT_NAMES.token_mask,
        )?;
        checks.values_below(
            &context.positions,
            self.config.max_position_embeddings,
            INPUT_NAMES.position_ids,
            "position",
        );
        checks.run("muse_glimmer.execution")?;
        let overrides = range
            .clone()
            .map(|index| {
                let address = Layer::decoder(index);
                let overridden = context.attention.is_some()
                    || context
                        .layer_attention
                        .iter()
                        .any(|(candidate, _)| *candidate == address);
                let policy = context.attention_for(address, true);
                policy.validate([batch, heads, sequence, sequence], &device)?;
                Ok(overridden.then_some(policy))
            })
            .collect::<Result<Vec<_>>>()?;
        let tables = self.rotary_tables(context.positions.clone(), context.evaluated_context)?;
        let token_mask = context.token_mask.clone();
        let inputs = LayerInputs {
            tables: &tables,
            token_mask: token_mask.as_ref(),
            sliding_window: self.config.resolved_sliding_window(),
        };
        for (index, policy) in range.clone().zip(overrides) {
            let layer = &self.layers[index];
            let address = Layer::decoder(index);
            state.hidden = match policy {
                None => layer.forward(state.hidden, &inputs, Some((address, &mut *hooks)))?,
                Some(policy) => {
                    layer.forward_with_policy(state.hidden, &inputs, policy, address, hooks)?
                }
            };
        }
        state.next_layer = range.end;
        Ok(state)
    }
}

impl<B: Backend> MuseGlimmerDecoderLayer<B> {
    /// Evaluates one layer whose attention connectivity an experiment overrides.
    ///
    /// The override replaces causality and the local window; padding still
    /// excludes invalid queries and keys. One full score plane is evaluated.
    fn forward_with_policy(
        &self,
        hidden: burn::tensor::Tensor<B, 3>,
        inputs: &LayerInputs<'_, B>,
        policy: crate::experiment::AttentionPolicy<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<burn::tensor::Tensor<B, 3>> {
        let angles = self
            .rope
            .as_ref()
            .map(|rope| inputs.tables.get(rope))
            .transpose()?;
        let mut scope = Some((layer, &mut *hooks));
        let (residual, projection) = self.attention_projection(hidden, angles, &mut scope)?;
        let [batch, heads, sequence, _] = projection.query.dims();
        let padding = padding_attention_mask(
            inputs.token_mask,
            inputs.token_mask,
            batch,
            heads,
            sequence,
            sequence,
        )?;
        let disallowed = policy.allowed.map(|allowed| {
            allowed
                .bool_not()
                .expand([batch, heads, sequence, sequence])
        });
        let bias = policy.bias.map(|bias| {
            bias.cast(projection.query.dtype())
                .expand([batch, heads, sequence, sequence])
        });
        let context = grouped_query_attention(
            projection.query.clone(),
            projection.key.clone(),
            projection.value.clone(),
            union_exclusions(padding, disallowed),
            bias,
            muse_attention_options(policy.causal),
            muse_attention_precision(projection.query.dtype()),
        )?;
        let attended =
            self.self_attn
                .gated_output(context, projection.gate, Some((layer, &mut *hooks)))?;
        self.finish_block(residual, attended, Some((layer, hooks)))
    }
}

#[cfg(test)]
#[path = "hooks_tests.rs"]
/// Identity-hook parity, gate and norm interventions, staging, and policies.
mod tests;
