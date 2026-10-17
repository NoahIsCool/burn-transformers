//! Family-generic experiment interfaces for T5 models.
//!
//! [`HookedModel`] lets generic experiment code (activation caches, patching
//! sweeps, probes) run any family's complete stateless pass inside one hook
//! scope without naming the family. [`StagedModel`] lets it stop and resume a
//! residual stack between blocks.
//!
//! - [`T5ForConditionalGeneration`] is hooked as a whole encoder-decoder pass:
//!   input [`Seq2SeqInput`], output [`Seq2SeqOutput`], and every encoder point
//!   followed by every decoder point.
//! - [`T5Encoder`] is hooked as an encoder pass (input [`CausalLmInput`],
//!   output [`EncoderOutput`]) and staged over [`Stack::Encoder`]: `embed`,
//!   `run_layers`, and `normalize` keep the relative positions and padding in
//!   the state's execution context, exactly as the inherent methods do.
//!
//! Every method delegates to the inherent method of the same name, so trait
//! and inherent calls are the same computation (the tests check them bit for
//! bit).

use crate::{
    architecture::{HookedModel, StagedModel},
    experiment::HiddenState,
};

use super::*;
use crate::intervention::Stack;

impl<B: Backend> HookedModel<B> for T5ForConditionalGeneration<B> {
    /// Source and target tokens of one complete pass.
    type Input = Seq2SeqInput<B>;
    /// Encoder states and decoder logits.
    type Output = Seq2SeqOutput<B>;

    /// Lists every encoder point, then every decoder point, in emission order.
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        Ok(T5ForConditionalGeneration::activation_points(self))
    }

    /// Runs [`T5ForConditionalGeneration::forward_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Seq2SeqOutput<B>> {
        T5ForConditionalGeneration::forward_with_hooks(self, input, options, hooks)
    }
}

impl<B: Backend> HookedModel<B> for T5Encoder<B> {
    /// Source tokens, optional padding mask, and optional relative positions.
    type Input = CausalLmInput<B>;
    /// Final normalized encoder states and requested diagnostics.
    type Output = EncoderOutput<B>;

    /// Lists the embedding, every block's points, and the final normalization.
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        Ok(T5Encoder::activation_points(self))
    }

    /// Runs [`T5Encoder::encode_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<EncoderOutput<B>> {
        self.encode_with_hooks(input, options, hooks)
    }
}

impl<B: Backend> StagedModel<B> for T5Encoder<B> {
    /// Number of encoder blocks.
    fn num_layers(&self) -> usize {
        self.encoder.block.len()
    }

    /// The encoder stack: staged blocks emit [`Layer::encoder`] addresses.
    fn stack(&self) -> Stack {
        Stack::Encoder
    }

    /// Runs [`T5Encoder::embed_with_hooks`].
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        T5Encoder::embed_with_hooks(self, input, hooks)
    }

    /// Runs [`T5Encoder::run_layers_with_hooks`].
    fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: std::ops::Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        T5Encoder::run_layers_with_hooks(self, state, range, hooks)
    }

    /// Runs [`T5Encoder::normalize_with_hooks`].
    fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        T5Encoder::normalize_with_hooks(self, state, hooks)
    }
}

#[cfg(test)]
#[path = "experiment_tests.rs"]
mod tests;
