//! Activation hooks and staged execution for DeepSeek-V2.
//!
//! # Activation points
//!
//! [`DeepseekV2ForCausalLm::activation_points`] lists, in emission order:
//!
//! 1. `decoder.embedding` — token embeddings `[batch, token, channel]`;
//! 2. per block `i`: `residual.pre`, `attention.input` (after the input RMS
//!    norm), MLA's `query_latent` (low-rank-query checkpoints only) and
//!    `kv_latent` `[batch, token, rank]`, the expanded attention tensors
//!    `query`, `key`, `value` `[batch, head, token, channel]`, `scores` and
//!    `pattern` `[batch, head, query, key]`, and `head_context`, then
//!    `attention.output`, `mlp.input`; sparse blocks emit
//!    `mlp.router_logits` `[batch, token, expert]` (F32, edits change which
//!    experts run), `mlp.routed_output`, and `mlp.shared_output`, dense blocks
//!    `mlp.pre_activation` and `mlp.activation` (the SwiGLU gate branch);
//!    finally `mlp.output` and `residual.post`;
//! 3. `decoder.final_norm`.
//!
//! Requesting any MLA point makes that layer evaluate the reference graph on
//! one full `[batch, head, query, key]` plane (the points are that plane's
//! tensors); every other point keeps the bounded chunked attention.
//!
//! # Staged execution
//!
//! [`StagedModel`] runs the stack in resumable stages: `embed` validates the
//! input and builds an [`ExecutionContext`] with its positions, padding, and
//! rotary extent; `run_layers` advances any contiguous block range; and
//! `normalize` applies the final norm once. A context's attention overrides
//! ([`ExecutionContext::attention_for`]) replace causal visibility for a
//! layer: `causal` adds bottom-right causality, `allowed` restricts links,
//! and `bias` adds to the scaled scores, always together with padding. Such
//! layers evaluate one full score plane. Cached decoding is not hookable.

use core::ops::Range;

use burn::tensor::backend::Backend;

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::{HookedModel, StagedModel},
    attention::{padding_attention_mask, union_exclusions},
    experiment::{AttentionPolicy, ExecutionContext, HiddenState},
    intervention::{Hooks, Layer, PointInfo, Stack},
    models::shared::{DeferredChecks, check_nonempty_mask, ensure_same_device},
};

use super::{
    AttentionInputs, CustomConnectivity, DeepseekV2ForCausalLm, DeepseekV2Model, INPUT_NAMES,
    PARAMETERS, PositionalVisibility,
};

impl<B: Backend> DeepseekV2ForCausalLm<B> {
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
    /// graph. Hooks are validated against [`Self::activation_points`] before
    /// any computation. Probabilities (`options.attentions`) cannot be
    /// combined with attention-internal points; capture `pattern` instead.
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
        hooks.run(self.activation_points(), |hooks| {
            let backbone = self.model.forward(&input, options, Some(hooks))?;
            self.finish_forward(&input, options, backbone)
        })
    }
}

impl<B: Backend> HookedModel<B> for DeepseekV2ForCausalLm<B> {
    /// Complete causal-language-model input.
    type Input = CausalLmInput<B>;
    /// Complete causal-language-model output.
    type Output = CausalLmOutput<B>;

    /// Delegates to [`DeepseekV2ForCausalLm::activation_points`].
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        Ok(Self::activation_points(self))
    }

    /// Delegates to [`DeepseekV2ForCausalLm::forward_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: Self::Input,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Self::Output> {
        Self::forward_with_hooks(self, input, options, hooks)
    }
}

impl<B: Backend> StagedModel<B> for DeepseekV2ForCausalLm<B> {
    /// Number of decoder blocks.
    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    /// DeepSeek blocks live in the decoder stack.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Validates the input, embeds its tokens, and emits the embedding point.
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.embedding().info()], |hooks| {
            let (positions, _) = self.model.validate_public_input(&input)?;
            let context = ExecutionContext::new(positions, input.token_mask.clone())?;
            let hidden = hooks.apply(
                Stack::Decoder.embedding(),
                self.model.embed_tokens.forward(input.token_ids),
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

    /// Applies the final RMS norm once and emits the final-norm point.
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
                    "deepseek_v2.execution.normalize",
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

impl<B: Backend> DeepseekV2Model<B> {
    /// Requires a staged state on the model's device and activation dtype.
    fn check_staged_hidden(&self, state: &HiddenState<B>) -> Result<()> {
        state.validate(self.config.hidden_size)?;
        ensure_same_device(
            "deepseek_v2.execution.hidden",
            &state.hidden.device(),
            &self.embed_tokens.weight.lazy_device(),
            PARAMETERS,
        )?;
        let expected = self.embed_tokens.weight.val().dtype();
        if state.hidden.dtype() != expected {
            return Err(Error::configuration(
                "deepseek_v2.execution.hidden",
                format!(
                    "hidden states must use the model dtype {expected:?}, found {:?}",
                    state.hidden.dtype()
                ),
            ));
        }
        Ok(())
    }

    /// Validates a staged state and policies, then executes blocks `range`.
    ///
    /// Every check (layout, devices, dtype, padding rows, the position limit,
    /// attention-policy extents, and encoder absence) completes before any
    /// block runs. The returned cursor names the first unexecuted block.
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
                "deepseek_v2.execution.encoder",
                "DeepSeek decoder blocks do not define cross-attention",
            ));
        }
        for (address, _) in &context.layer_attention {
            if !(0..self.layers.len()).any(|index| *address == Layer::decoder(index)) {
                return Err(Error::configuration(
                    "deepseek_v2.execution.layer_attention",
                    "override addresses an unavailable DeepSeek decoder layer",
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
        checks.run("deepseek_v2.execution")?;
        let policies = range
            .clone()
            .map(|index| {
                let policy = context.attention_for(Layer::decoder(index), true);
                policy.validate([batch, heads, sequence, sequence], &device)?;
                custom_connectivity(policy, context, batch, heads, sequence)
            })
            .collect::<Result<Vec<_>>>()?;
        let angles = self.rotary_tables(context.positions.clone(), context.evaluated_context)?;
        let visibility = PositionalVisibility::causal_columns(
            sequence,
            context
                .token_mask
                .as_ref()
                .map(|mask| mask.as_tensor().clone()),
            &device,
        )?;
        let inputs = AttentionInputs {
            angles: &angles,
            visibility: &visibility,
            policy: self.execution_policy,
            retain_attention: false,
        };
        for (index, custom) in range.clone().zip(policies) {
            state.hidden = self.layers[index]
                .forward(
                    state.hidden,
                    &inputs,
                    custom.as_ref(),
                    Layer::decoder(index),
                    Some(hooks),
                )?
                .hidden;
        }
        state.next_layer = range.end;
        Ok(state)
    }
}

/// Translates a layer's attention policy into custom connectivity.
///
/// The default causal policy returns `None` (bounded causal attention).
/// Otherwise padding exclusions `[batch, head, query, key]` are combined with
/// the complement of `allowed`, and `bias` is passed through for the scores.
fn custom_connectivity<B: Backend>(
    policy: AttentionPolicy<B>,
    context: &ExecutionContext<B>,
    batch: usize,
    heads: usize,
    sequence: usize,
) -> Result<Option<CustomConnectivity<B>>> {
    if policy.causal && policy.allowed.is_none() && policy.bias.is_none() {
        return Ok(None);
    }
    let shape = [batch, heads, sequence, sequence];
    let padding = padding_attention_mask(
        context.token_mask.as_ref(),
        context.token_mask.as_ref(),
        batch,
        heads,
        sequence,
        sequence,
    )?;
    // `allowed` broadcasts over singleton axes; the expansion is a view.
    let disallowed = policy
        .allowed
        .map(|allowed| allowed.bool_not().expand(shape));
    Ok(Some(CustomConnectivity {
        causal: policy.causal,
        exclusion: union_exclusions(padding, disallowed),
        bias: policy.bias,
    }))
}

#[cfg(test)]
#[path = "hooks_tests.rs"]
/// Identity-hook parity, interventions, staged execution, and policy validation.
mod tests;
