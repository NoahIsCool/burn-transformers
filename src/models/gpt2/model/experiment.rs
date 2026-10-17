//! Family-generic experiment interfaces for GPT-2.
//!
//! [`Gpt2ForCausalLm`] implements [`HookedModel`] with its complete stateless
//! pass, and both [`Gpt2Model`] and [`Gpt2ForCausalLm`] implement
//! [`StagedModel`] over the decoder stack, so family-generic activation
//! caches and residual-patching sweeps run on GPT-2 unchanged:
//!
//! - `embed` validates the input exactly like a complete pass (one readback)
//!   and returns `E_token[x] + E_position[p]` with a context holding the
//!   positions `p` (default: physical columns, including padding) and the
//!   padding mask;
//! - `run_layers` applies blocks `range` with the context's padding mask and
//!   each block's own `causal` flag, emitting exactly those blocks' points;
//! - `normalize` applies the final LayerNorm once.
//!
//! The stages compose to the complete pass bit for bit. GPT-2 blocks own
//! their attention visibility (`causal`, padding), so a context carrying
//! experiment attention policies, cross-attention, or an encoder is rejected
//! with a capability error instead of being silently ignored.

use std::ops::Range;

use super::*;
use crate::{
    architecture::{HookedModel, StagedModel},
    experiment::{ExecutionContext, HiddenState},
};

impl<B: Backend> HookedModel<B> for Gpt2ForCausalLm<B> {
    /// Token IDs, optional padding mask, and optional absolute positions.
    type Input = CausalLmInput<B>;
    /// Selected logits and requested hidden states.
    type Output = CausalLmOutput<B>;

    /// Lists every point of the stateless pass in emission order.
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        Ok(Gpt2ForCausalLm::activation_points(self))
    }

    /// Runs [`Gpt2ForCausalLm::forward_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        Gpt2ForCausalLm::forward_with_hooks(self, input, options, hooks)
    }
}

impl<B: Backend> StagedModel<B> for Gpt2Model<B> {
    /// Number of transformer blocks.
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// GPT-2 is decoder-only: blocks emit [`Layer::decoder`] addresses.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Validates and embeds tokens, emitting only the embedding point.
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.embedding().info()], |hooks| {
            let (hidden, validated) = self.embed_tokens(&input)?;
            let hidden = hooks.apply(Stack::Decoder.embedding(), hidden)?;
            let positions = validated.require_positions(INPUT_NAMES.position_ids)?;
            let [batch, tokens] = positions.dims();
            // Values were validated in the embedding readback; implicit
            // physical columns span exactly `0..tokens`.
            let evaluated_context = if input.position_ids.is_some() {
                validated.evaluated_context
            } else {
                tokens
            };
            let mut context = ExecutionContext::contiguous(batch, tokens, &positions.device())?;
            context.positions = positions;
            context.token_mask = input.token_mask;
            context.evaluated_context = evaluated_context;
            Ok(HiddenState {
                hidden,
                context,
                next_layer: 0,
                normalized: false,
            })
        })
    }

    /// Runs blocks `range` with the context's padding, in one hook scope.
    fn run_layers_with_hooks(
        &self,
        mut state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        // A bounded catalog that never indexes an unvalidated range; the range
        // itself is validated inside the scope so failures clear old captures.
        let points = self
            .layers
            .iter()
            .enumerate()
            .filter(|(index, _)| range.contains(index))
            .flat_map(|(index, block)| block.activation_points(index))
            .collect();
        hooks.run(points, |hooks| {
            self.validate_stage(&state)?;
            state.validate_range(range.clone(), self.layers.len())?;
            let [batch, tokens, _] = state.hidden.dims();
            validate_nonempty_mask(
                state.context.token_mask.as_ref(),
                batch,
                tokens,
                INPUT_NAMES.token_mask,
            )?;
            let mut hidden = state.hidden.clone();
            for index in range.clone() {
                hidden = self.layers[index].forward_inner(
                    hidden,
                    state.context.token_mask.as_ref(),
                    index,
                    Some(&mut *hooks),
                )?;
            }
            state.hidden = hidden;
            state.next_layer = range.end;
            Ok(state)
        })
    }

    /// Applies the final LayerNorm once all blocks have run.
    fn normalize_with_hooks(
        &self,
        mut state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.final_norm().info()], |hooks| {
            self.validate_stage(&state)?;
            let depth = self.layers.len();
            state.validate_range(depth..depth, depth)?;
            state.hidden = hooks.apply(
                Stack::Decoder.final_norm(),
                self.normalize(state.hidden.clone())?,
            )?;
            state.normalized = true;
            Ok(state)
        })
    }
}

impl<B: Backend> Gpt2Model<B> {
    /// Checks a staged state's sync-free layout and rejects policies GPT-2 cannot apply.
    fn validate_stage(&self, state: &HiddenState<B>) -> Result<()> {
        let [batch, tokens, _] = state.hidden.dims();
        validate_hidden(
            &state.hidden,
            self.config.n_embd,
            &self.token_embedding.weight.lazy_device(),
        )?;
        state
            .context
            .validate_layout([batch, tokens], &state.hidden.device())?;
        let context = &state.context;
        if context.attention.is_some()
            || !context.layer_attention.is_empty()
            || context.cross_attention.is_some()
            || !context.layer_cross_attention.is_empty()
            || context.encoder.is_some()
        {
            return Err(Error::BackendCapability(
                "GPT-2 staged blocks apply their own causal and padding attention; \
                 experiment attention policies, cross-attention, and encoders are not supported"
                    .into(),
            ));
        }
        Ok(())
    }
}

impl<B: Backend> StagedModel<B> for Gpt2ForCausalLm<B> {
    /// Number of transformer blocks.
    fn num_layers(&self) -> usize {
        StagedModel::num_layers(&self.model)
    }

    /// GPT-2 is decoder-only: blocks emit [`Layer::decoder`] addresses.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Delegates to [`Gpt2Model`]'s staged embedding.
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model.embed_with_hooks(input, hooks)
    }

    /// Delegates to [`Gpt2Model`]'s staged block range.
    fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model.run_layers_with_hooks(state, range, hooks)
    }

    /// Delegates to [`Gpt2Model`]'s final normalization.
    fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        self.model.normalize_with_hooks(state, hooks)
    }
}

#[cfg(test)]
#[path = "experiment_tests.rs"]
mod tests;
