//! Activation hooks and staged execution of the stateless Nemotron-H graph.
//!
//! # Activation points
//!
//! Every Nemotron-H block computes `x + Mixer(RMSNorm(x))` with exactly one
//! mixer, so its points depend on the block kind. With `L = Layer::decoder(i)`
//! (all hidden points are `[batch, token, channel]`):
//!
//! | block | points between `L.residual_pre()` and `L.residual_post()` |
//! | --- | --- |
//! | Mamba-2 | `L.mixer_input()`, `L.mixer_output()` |
//! | attention | `L.mixer_input()`, query, key, value, (scores, pattern), head context, `L.mixer_output()` |
//! | dense MLP | `L.mlp_input()`, `L.mlp_pre_activation()`, `L.mlp_activation()`, `L.mlp_output()` |
//! | sparse MoE | `L.mlp_input()`, `L.mlp_output()` |
//!
//! `mixer_*` are the crate's `attention.input`/`attention.output` points,
//! used for every sequence mixer. Attention query/key/value and head-context
//! points come from the shared attention module; scores and patterns appear
//! only when the block's attention is configured with
//! [`AttentionPrecisionMode::EagerFp32Softmax`], because backend kernels never
//! materialize them. A MoE block exposes its combined routed-plus-shared
//! output; individual experts run on gathered token subsets and have no
//! token-aligned activations. The stack adds `Stack::Decoder.embedding()`
//! before block zero and `Stack::Decoder.final_norm()` after the last block.
//!
//! Hooks run on the stateless graph only. Cached prefill/decode and the MTP
//! stage are separate execution paths and never invoke them.
//!
//! # Staged execution
//!
//! [`NemotronHModel::embed`], [`NemotronHModel::run_layers`], and
//! [`NemotronHModel::normalize`] split the stateless stack at block
//! boundaries, exchanging a [`HiddenState`]. Attention blocks honor the
//! context's attention policies (custom connectivity and additive bias,
//! intersected with padding and the sliding window). Mamba-2 blocks are
//! causal recurrences and reject non-default policies; dense and MoE blocks
//! mix no tokens and ignore them. Token positions are carried for the shared
//! context contract but do not enter Nemotron-H's position-free mathematics.

use core::ops::Range;

use burn::tensor::backend::Backend;

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::{HookedModel, StagedModel},
    attention::AttentionPrecisionMode,
    experiment::{ExecutionContext, HiddenState},
    intervention::{Hooks, Layer, PointInfo, Stack},
    models::shared::{
        DeferredChecks, ImplicitPositions, InputContract, check_nonempty_mask, ensure_same_device,
        validate_input,
    },
};

use super::{
    BlockContext, BlockMixer, INPUT_NAMES, NemotronHForCausalLm, NemotronHModel, PARAMETERS,
};

impl<B: Backend> NemotronHModel<B> {
    /// Lists every activation point of the stateless stack in emission order.
    ///
    /// See the module documentation for the points of each block kind.
    #[must_use]
    pub fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Decoder.embedding().info()];
        for (index, block) in self.layers.iter().enumerate() {
            let layer = Layer::decoder(index);
            points.push(layer.residual_pre().info());
            match block.mixer() {
                Ok(BlockMixer::Mamba2(_)) => {
                    points.extend([layer.mixer_input().info(), layer.mixer_output().info()]);
                }
                Ok(BlockMixer::Attention(attention)) => {
                    let projected = layer.self_attention();
                    points.extend([
                        layer.mixer_input().info(),
                        projected.query().info(),
                        projected.key().info(),
                        projected.value().info(),
                    ]);
                    if attention.precision_mode == AttentionPrecisionMode::EagerFp32Softmax {
                        points.extend([projected.scores().info(), projected.pattern().info()]);
                    }
                    points.extend([projected.head_context().info(), layer.mixer_output().info()]);
                }
                Ok(BlockMixer::Mlp(_)) => points.extend([
                    layer.mlp_input().info(),
                    layer.mlp_pre_activation().info(),
                    layer.mlp_activation().info(),
                    layer.mlp_output().info(),
                ]),
                Ok(BlockMixer::Moe(_)) => {
                    points.extend([layer.mlp_input().info(), layer.mlp_output().info()]);
                }
                // An invalid topology fails when the block executes.
                Err(_) => {}
            }
            points.push(layer.residual_post().info());
        }
        points.push(Stack::Decoder.final_norm().info());
        points
    }

    /// Embeds tokens and resolves the execution context before block zero.
    ///
    /// Positions default to each row's valid-token ordinals. They accompany
    /// the state for generic experiments; Nemotron-H's mixers do not read
    /// them. Final normalization is never applied here.
    ///
    /// # Errors
    ///
    /// Returns the input-contract errors of the stateless forward pass.
    pub fn embed(&self, input: &CausalLmInput<B>) -> Result<HiddenState<B>> {
        let positions = validate_input(
            input,
            &self.embed_tokens.weight.lazy_device(),
            &InputContract::new(INPUT_NAMES, PARAMETERS, self.config.vocab_size)
                .with_position_limit(self.config.max_position_embeddings)
                .with_implicit_positions(ImplicitPositions::ValidTokenOrdinals),
        )?
        .require_positions(INPUT_NAMES.position_ids)?;
        let context = ExecutionContext::new(positions, input.token_mask.clone())?;
        HiddenState::new(self.embed_tokens.forward(input.token_ids.clone()), context)
    }

    /// Embeds tokens while applying only the embedding point's scoped actions.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::embed`] and of the hook scope.
    pub fn embed_with_hooks(
        &self,
        input: &CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.embedding().info()], |hooks| {
            let mut state = self.embed(input)?;
            state.hidden = hooks.apply(Stack::Decoder.embedding(), state.hidden)?;
            Ok(state)
        })
    }

    /// Runs exactly the blocks `range`, which must start at `state.next_layer`.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for a normalized state, a range that
    /// skips or repeats blocks, an encoder context, a policy that a Mamba-2
    /// block cannot honor, or an override addressing a missing block; shape
    /// and device errors for an incompatible state; and execution errors.
    pub fn run_layers(&self, state: HiddenState<B>, range: Range<usize>) -> Result<HiddenState<B>> {
        self.run_layers_inner(state, range, None)
    }

    /// Runs a block range with actions addressed by the blocks' original indices.
    ///
    /// Requests for points outside `range` fail before any block executes.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::run_layers`] and of the hook scope.
    pub fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        let prefixes = range
            .clone()
            .map(|index| format!("decoder.layers.{index}."))
            .collect::<Vec<_>>();
        let points = self
            .activation_points()
            .into_iter()
            .filter(|point| prefixes.iter().any(|prefix| point.name.starts_with(prefix)))
            .collect();
        hooks.run(points, |hooks| {
            self.run_layers_inner(state, range, Some(hooks))
        })
    }

    /// Validates a staged state and policies, then executes `range`.
    fn run_layers_inner(
        &self,
        mut state: HiddenState<B>,
        range: Range<usize>,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<HiddenState<B>> {
        self.validate_state(&state)?;
        state.validate_range(range.clone(), self.layers.len())?;
        let [batch, sequence, _] = state.hidden.dims();
        let mut checks = DeferredChecks::new();
        check_nonempty_mask(
            &mut checks,
            state.context.token_mask.as_ref(),
            batch,
            sequence,
            INPUT_NAMES.token_mask,
        )?;
        checks.run("nemotron_h.execution")?;
        for (address, _) in &state.context.layer_attention {
            if !(0..self.layers.len()).any(|index| *address == Layer::decoder(index)) {
                return Err(Error::configuration(
                    "nemotron_h.execution.layer_attention",
                    "override addresses an unavailable Nemotron-H decoder layer",
                ));
            }
        }
        let device = state.hidden.device();
        for index in range.clone() {
            let policy = state.context.attention_for(Layer::decoder(index), true);
            match self.layers[index].mixer()? {
                BlockMixer::Mamba2(_)
                    if !policy.causal || policy.allowed.is_some() || policy.bias.is_some() =>
                {
                    return Err(Error::BackendCapability(format!(
                        "Nemotron-H Mamba-2 block {index} is a causal recurrence and supports no custom attention connectivity or score bias"
                    )));
                }
                BlockMixer::Attention(attention) => {
                    policy.validate([batch, attention.query_heads, sequence, sequence], &device)?;
                    if policy
                        .bias
                        .as_ref()
                        .is_some_and(|bias| bias.dtype() != state.hidden.dtype())
                    {
                        return Err(Error::BackendCapability(
                            "Nemotron-H attention bias must use the activation dtype".into(),
                        ));
                    }
                }
                _ => {}
            }
        }
        for index in range.clone() {
            let policy = state.context.attention_for(Layer::decoder(index), true);
            let context = BlockContext {
                index,
                token_mask: state.context.token_mask.as_ref(),
                sliding_window: self.config.sliding_window,
                policy: Some(&policy),
            };
            let (hidden, _) =
                self.layers[index].forward(state.hidden, &context, hooks.as_deref_mut())?;
            state.hidden = hidden;
        }
        state.next_layer = range.end;
        Ok(state)
    }

    /// Applies the final normalization once, at an explicit block boundary.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for an already normalized state or a
    /// cursor beyond the stack, and the state-compatibility errors of
    /// [`Self::run_layers`].
    pub fn normalize(&self, mut state: HiddenState<B>) -> Result<HiddenState<B>> {
        if state.normalized || state.next_layer > self.layers.len() {
            return Err(Error::configuration(
                "nemotron_h.execution.normalize",
                "expected an unnormalized state at a valid block boundary",
            ));
        }
        self.validate_state(&state)?;
        state.hidden = self.norm.forward(state.hidden);
        state.normalized = true;
        Ok(state)
    }

    /// Applies the final normalization with a scoped final-norm capture or edit.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::normalize`] and of the hook scope.
    pub fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.final_norm().info()], |hooks| {
            let mut state = self.normalize(state)?;
            state.hidden = hooks.apply(Stack::Decoder.final_norm(), state.hidden)?;
            Ok(state)
        })
    }

    /// Checks a staged state's width, context, device, dtype, and position range.
    fn validate_state(&self, state: &HiddenState<B>) -> Result<()> {
        state.validate(self.config.hidden_size)?;
        ensure_same_device(
            "nemotron_h.execution.hidden",
            &state.hidden.device(),
            &self.embed_tokens.weight.lazy_device(),
            PARAMETERS,
        )?;
        let dtype = self.embed_tokens.weight.val().dtype();
        if state.hidden.dtype() != dtype {
            return Err(Error::BackendCapability(format!(
                "Nemotron-H staged states must use the model dtype {dtype:?}, found {:?}",
                state.hidden.dtype()
            )));
        }
        if state.context.encoder.is_some()
            || state.context.cross_attention.is_some()
            || !state.context.layer_cross_attention.is_empty()
        {
            return Err(Error::configuration(
                "nemotron_h.execution.encoder",
                "Nemotron-H decoder blocks do not define cross-attention",
            ));
        }
        if state.context.evaluated_context > self.config.max_position_embeddings {
            return Err(Error::configuration(
                "nemotron_h.execution.positions",
                format!(
                    "positions must lie below max_position_embeddings {}",
                    self.config.max_position_embeddings
                ),
            ));
        }
        Ok(())
    }
}

impl<B: Backend> NemotronHForCausalLm<B> {
    /// Lists the stateless backbone's activation points in emission order.
    ///
    /// See the hooks module documentation for the points of each block kind.
    #[must_use]
    pub fn activation_points(&self) -> Vec<PointInfo> {
        self.model.activation_points()
    }

    /// Runs the stateless forward pass while capturing or editing activations.
    ///
    /// Edits change every later activation, the returned hidden states, and
    /// the logits, and remain in the autodiff graph. Unknown points fail
    /// before the model runs.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::forward`] and of the hook scope.
    pub fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        hooks.run(self.activation_points(), |hooks| {
            if options.attentions {
                return Err(Error::BackendCapability(
                    "Nemotron-H attention probability capture is not yet exposed".into(),
                ));
            }
            let backbone = self.model.forward(&input, options, Some(hooks))?;
            self.finish_forward(&input, options, backbone)
        })
    }
}

impl<B: Backend> HookedModel<B> for NemotronHForCausalLm<B> {
    /// Token IDs with optional validity and positions.
    type Input = CausalLmInput<B>;
    /// Logits and requested diagnostics.
    type Output = CausalLmOutput<B>;

    /// Delegates to [`NemotronHForCausalLm::activation_points`].
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        Ok(NemotronHForCausalLm::activation_points(self))
    }

    /// Delegates to [`NemotronHForCausalLm::forward_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: Self::Input,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Self::Output> {
        NemotronHForCausalLm::forward_with_hooks(self, input, options, hooks)
    }
}

impl<B: Backend> StagedModel<B> for NemotronHForCausalLm<B> {
    /// Number of hybrid blocks.
    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    /// Decoder-only stack.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Delegates to [`NemotronHModel::embed_with_hooks`].
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model.embed_with_hooks(&input, hooks)
    }

    /// Delegates to [`NemotronHModel::run_layers_with_hooks`].
    fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model.run_layers_with_hooks(state, range, hooks)
    }

    /// Delegates to [`NemotronHModel::normalize_with_hooks`].
    fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model.normalize_with_hooks(state, hooks)
    }
}

#[cfg(test)]
#[path = "hooks_tests.rs"]
mod tests;
