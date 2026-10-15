//! Activation hooks and staged execution of the stateless Gemma graph.
//!
//! Every Gemma model implements [`HookedModel`] and [`StagedModel`] through
//! the dense-decoder engine, so the catalog, validity masks, and staged
//! semantics are exactly those of the Llama family:
//!
//! ```text
//! decoder.embedding                    the sqrt(d)-scaled embedding (layer 0's input)
//! per layer l: residual.pre, attention.input (after input_layernorm),
//!              attention.{query, key, value} (after Q/K norms and rotation),
//!              [attention.{scores, pattern} under eager attention; scores soft-capped],
//!              attention.head_context,
//!              attention.output (after post_attention_layernorm),
//!              mlp.input (after pre_feedforward_layernorm),
//!              mlp.pre_activation, mlp.activation,
//!              mlp.output (after post_feedforward_layernorm), residual.post
//! decoder.final_norm
//! ```
//!
//! Keys and values keep their native KV heads. With padding, every point
//! carries a validity mask so padded entries read as zero and edits cannot
//! revive them. Staged execution exchanges [`HiddenState`] values; an
//! [`AttentionPolicy`](crate::experiment::AttentionPolicy) replaces a
//! layer's causal/sliding rule while padding stays excluded.

use core::ops::Range;

use burn::tensor::backend::Backend;

use crate::{
    CausalLmInput, CausalLmOutput, ForwardOptions, Result,
    architecture::{HookedModel, StagedModel},
    experiment::HiddenState,
    intervention::{Hooks, PointInfo, Stack},
    models::llama::model::engine,
};

use super::GemmaForCausalLm;

impl<B: Backend> GemmaForCausalLm<B> {
    /// Lists every activation point of a complete stateless pass, in emission order.
    #[must_use]
    pub fn activation_points(&self) -> Vec<PointInfo> {
        engine::activation_points(self)
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
    /// [`crate::Error::Intervention`] for failing callbacks.
    pub fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        engine::forward_with_hooks(self, input, options, hooks)
    }

    /// Embeds and scales tokens into a raw state before layer zero, in a scope of the embedding point.
    ///
    /// # Errors
    ///
    /// Returns input validation errors.
    pub fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        engine::embed_with_hooks(self, input, hooks)
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
        engine::run_layers_with_hooks(self, state, range, hooks)
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
        engine::normalize_with_hooks(self, state, hooks)
    }
}

impl<B: Backend> HookedModel<B> for GemmaForCausalLm<B> {
    /// Token IDs, optional mask, and optional positions.
    type Input = CausalLmInput<B>;
    /// Logits and requested diagnostics.
    type Output = CausalLmOutput<B>;

    /// Delegates to [`GemmaForCausalLm::activation_points`].
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        Ok(GemmaForCausalLm::activation_points(self))
    }

    /// Delegates to [`GemmaForCausalLm::forward_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        GemmaForCausalLm::forward_with_hooks(self, input, options, hooks)
    }
}

impl<B: Backend> StagedModel<B> for GemmaForCausalLm<B> {
    /// Returns the number of decoder blocks.
    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    /// The family is decoder-only.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Delegates to [`GemmaForCausalLm::embed_with_hooks`].
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        GemmaForCausalLm::embed_with_hooks(self, input, hooks)
    }

    /// Delegates to [`GemmaForCausalLm::run_layers_with_hooks`].
    fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        GemmaForCausalLm::run_layers_with_hooks(self, state, range, hooks)
    }

    /// Delegates to [`GemmaForCausalLm::normalize_with_hooks`].
    fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        GemmaForCausalLm::normalize_with_hooks(self, state, hooks)
    }
}
