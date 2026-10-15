//! Batched cached prefill and decode of Gemma models.
//!
//! Gemma runs the dense-decoder engine's cache step unchanged (see the Llama
//! family's `model::decode` documentation for the step itself): every call
//! validates its rows with one device-to-host read, runs one model pass over
//! the whole `[batch, Q]` block of new tokens, and publishes the step
//! atomically. Global layers live in one full-storage group written ahead
//! and read through a per-step plan; sliding layers whose window is below
//! the cache capacity share one ring group of `sliding_window` slots, so a
//! Gemma 3 4B cache of capacity `C > 1024` stores 29 of its 34 layers in
//! rings of 1024 positions instead of `C`. Masks always use cache positions;
//! rotary tables use the explicit `position_ids` when given, per layer type.
//!
//! [`GemmaDecodeState`] wraps the engine's state type so a Llama-family
//! state can never be passed to a Gemma model (or vice versa) by type, and a
//! state allocated for another Gemma geometry is rejected at run time.

use burn::tensor::backend::Backend;

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, ForwardOptions, Result,
    architecture::DecodeState,
    cache::{CacheConfig, TensorDecodeState},
    models::llama::{
        LlamaDecodeState,
        model::engine::{self, Phase},
    },
};

use super::{GemmaConfig, GemmaForCausalLm};

/// External KV state of a Gemma model: full storage for global layers, a ring for sliding ones.
#[derive(Debug)]
pub struct GemmaDecodeState<B: Backend> {
    /// The dense-decoder engine's cache groups, lengths, and clean-tail flag.
    inner: LlamaDecodeState<B>,
}

impl<B: Backend> GemmaDecodeState<B> {
    /// Borrows the generic cache state for diagnostics and lifecycle integration.
    #[must_use]
    pub const fn tensor_state(&self) -> &TensorDecodeState<B> {
        self.inner.tensor_state()
    }

    /// Returns how many layers live in a sliding ring rather than full storage.
    #[must_use]
    pub fn ring_layers(&self) -> usize {
        self.inner.ring_layers()
    }
}

impl<B: Backend> DecodeState<B> for GemmaDecodeState<B> {
    /// Returns the number of rows.
    fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    /// Returns the logical capacity per row.
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns each row's committed token count.
    fn lengths(&self) -> &[usize] {
        self.inner.lengths()
    }

    /// Clears every row and zeroes every cache group.
    fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }

    /// Reorders or duplicates rows across every group atomically.
    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        self.inner.reorder(indices)
    }

    /// Rolls rows back to shorter lengths (ring groups only within their window).
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        self.inner.truncate(lengths)
    }

    /// Returns the device holding the cache.
    fn device(&self) -> &B::Device {
        self.inner.device()
    }
}

impl<B: Backend> CausalLanguageModel<B> for GemmaForCausalLm<B> {
    /// The serialized checkpoint configuration.
    type Config = GemmaConfig;
    /// External KV state.
    type State = GemmaDecodeState<B>;

    /// Delegates to the stateless forward pass.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        GemmaForCausalLm::forward(self, input, options)
    }

    /// Allocates full and sliding-ring KV groups for `batch_size` rows.
    ///
    /// The capacity may not exceed `max_position_embeddings`.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        Ok(GemmaDecodeState {
            inner: LlamaDecodeState::for_graph(self, config, batch_size, device)?,
        })
    }

    /// Appends prompt rows (each with at least one valid token) in one batched step.
    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        engine::forward_cached(self, input, &mut state.inner, options, Phase::Prefill)
    }

    /// Appends continuation tokens in one batched step; fully masked rows are no-ops.
    fn decode(
        &self,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        let input = CausalLmInput {
            token_ids: input.token_ids,
            token_mask: input.token_mask,
            position_ids: input.position_ids,
        };
        engine::forward_cached(self, input, &mut state.inner, options, Phase::Decode)
    }
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
