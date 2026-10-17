//! Family-generic experiment interfaces for GPT-NeoX.
//!
//! [`GptNeoXForCausalLm`] implements [`HookedModel`] with its complete causal
//! pass and [`StagedModel`] over the decoder stack, so family-generic
//! activation caches, tuned-lens probes, and residual-patching sweeps run on
//! Pythia checkpoints unchanged:
//!
//! - `embed` validates parameters and input exactly like a complete pass (one
//!   readback) and returns `E[x]` with a context holding the RoPE positions
//!   (default: physical columns, including padding) and the padding mask;
//! - `run_layers` applies blocks `range` causally with partial RoPE at the
//!   context's positions, sharing one table per head width across the range;
//! - `normalize` applies the final LayerNorm once.
//!
//! The stages compose to the complete causal pass bit for bit. Because
//! positions drive RoPE in every block, `run_layers` re-validates them (and
//! the padding rows) against `max_position_embeddings`, so edited contexts
//! fail contextually. Blocks own their attention visibility, so a context
//! carrying experiment attention policies, cross-attention, or an encoder is
//! rejected rather than silently ignored.

use std::ops::Range;

use super::*;
use crate::{
    architecture::{HookedModel, StagedModel},
    experiment::{ExecutionContext, HiddenState},
};

impl<B: Backend> HookedModel<B> for GptNeoXForCausalLm<B> {
    /// Token IDs, optional padding mask, and optional absolute positions.
    type Input = CausalLmInput<B>;
    /// Selected logits and requested hidden states.
    type Output = CausalLmOutput<B>;

    /// Lists every point of the causal pass in emission order.
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        Ok(GptNeoXForCausalLm::activation_points(self))
    }

    /// Runs [`GptNeoXForCausalLm::forward_with_hooks`] (causal attention).
    fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        GptNeoXForCausalLm::forward_with_hooks(self, input, options, hooks)
    }
}

impl<B: Backend> StagedModel<B> for GptNeoXForCausalLm<B> {
    /// Number of transformer blocks.
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// GPT-NeoX is decoder-only: blocks emit [`Layer::decoder`] addresses.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Validates parameters and input, then embeds tokens, emitting only the embedding point.
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        hooks.run(vec![Stack::Decoder.embedding().info()], |hooks| {
            self.validate_parameters()?;
            let validated =
                validate_public_input(&input, &self.config, &self.embed_in.weight.lazy_device())?;
            let positions = validated.require_positions(INPUT_NAMES.position_ids)?;
            let hidden = hooks.apply(
                Stack::Decoder.embedding(),
                self.embed_in.forward(input.token_ids),
            )?;
            let [batch, tokens] = positions.dims();
            // The embedding readback bounded every position (implicit columns
            // included), so `evaluated_context = max + 1` is exact.
            let mut context = ExecutionContext::contiguous(batch, tokens, &positions.device())?;
            context.positions = positions;
            context.token_mask = input.token_mask;
            context.evaluated_context = validated.evaluated_context;
            Ok(HiddenState {
                hidden,
                context,
                next_layer: 0,
                normalized: false,
            })
        })
    }

    /// Runs blocks `range` causally at the context's positions, in one hook scope.
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
            .flat_map(|(index, layer)| layer.activation_points(index))
            .collect();
        hooks.run(points, |hooks| {
            self.validate_parameters()?;
            self.validate_stage(&state)?;
            state.validate_range(range.clone(), self.layers.len())?;
            // Positions drive RoPE in every block: re-check edited contexts.
            let [batch, tokens, width] = state.hidden.dims();
            state.validate(width)?;
            if state.context.evaluated_context > self.config.max_position_embeddings {
                return Err(Error::shape(
                    INPUT_NAMES.position_ids,
                    format!(
                        "position values in 0..{}",
                        self.config.max_position_embeddings
                    ),
                    format!(
                        "an evaluated context of {}",
                        state.context.evaluated_context
                    ),
                ));
            }
            validate_nonempty_mask(
                state.context.token_mask.as_ref(),
                batch,
                tokens,
                INPUT_NAMES.token_mask,
            )?;
            state.hidden = self.run_blocks(
                state.hidden.clone(),
                &state.context.positions,
                state.context.token_mask.as_ref(),
                range.clone(),
                true,
                hooks,
                None,
            )?;
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
                self.normalize_hidden(state.hidden.clone())?,
            )?;
            state.normalized = true;
            Ok(state)
        })
    }
}

impl<B: Backend> GptNeoXForCausalLm<B> {
    /// Checks a staged state's sync-free layout and rejects policies GPT-NeoX cannot apply.
    fn validate_stage(&self, state: &HiddenState<B>) -> Result<()> {
        self.validate_hidden(&state.hidden)?;
        let [batch, tokens, _] = state.hidden.dims();
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
                "GPT-NeoX staged blocks apply their own causal and padding attention; \
                 experiment attention policies, cross-attention, and encoders are not supported"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "experiment_tests.rs"]
mod tests;
