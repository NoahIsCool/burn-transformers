//! Traits implemented by concrete causal and encoder-decoder architectures.
//!
//! [`CausalLanguageModel`] and [`Seq2SeqLanguageModel`] cover inference:
//! stateless forward passes and external decode state. Two further traits
//! cover experiments, so generic code can inspect and edit any family:
//!
//! - [`HookedModel`]: discover activation points and run a complete forward
//!   pass inside one [`Hooks`] scope.
//! - [`StagedModel`]: run a stack in explicit stages, embedding, block ranges,
//!   and final normalization, exchanging [`HiddenState`] values between them.
//!
//! Families implement both by delegating to their inherent methods of the same
//! names, which keep working unchanged. Inherent methods take precedence in
//! method-call syntax, so `model.activation_points()` on a concrete type still
//! returns that family's own type, while generic code bounded by these traits
//! sees the uniform signatures below.

use core::ops::Range;

use burn::{module::Module, tensor::backend::Backend};

use crate::{
    Result,
    architecture::types::*,
    cache::CacheConfig,
    experiment::HiddenState,
    intervention::{Hooks, Layer, PointInfo, Stack},
};

/// Configuration behavior shared by every supported architecture family.
pub trait ArchitectureConfig: Clone + core::fmt::Debug + Send + Sync {
    /// Checkpoint `model_type` values accepted by this configuration.
    const MODEL_TYPES: &'static [&'static str];

    /// Validates all cross-field tensor and topology invariants.
    fn validate(&self) -> Result<()>;
}

/// Lifecycle operations required of an external autoregressive decode state.
///
/// Concrete payloads differ: attention keeps keys and values, MLA keeps latent
/// vectors, and recurrent mixers keep matrix-valued state. The common contract
/// deliberately exposes lifecycle rather than pretending those payloads are
/// interchangeable.
pub trait DecodeState<B: Backend>: Send + core::fmt::Debug {
    /// Returns the number of independent sequences represented by the state.
    fn batch_size(&self) -> usize;

    /// Returns the maximum logical sequence length the state can represent.
    fn capacity(&self) -> usize;

    /// Returns each sequence's number of committed tokens.
    fn lengths(&self) -> &[usize];

    /// Clears every cached tensor and logical sequence length for reuse.
    fn reset(&mut self) -> Result<()>;

    /// Reorders or duplicates batch rows according to source indices.
    fn reorder(&mut self, indices: &[usize]) -> Result<()>;

    /// Rolls each sequence back to a previously computed logical length.
    fn truncate(&mut self, lengths: &[usize]) -> Result<()>;

    /// Returns the device on which state tensors reside.
    fn device(&self) -> &B::Device;
}

/// Backend-generic interface for decoder-only language models.
pub trait CausalLanguageModel<B: Backend>: Module<B> {
    /// Validated architecture configuration used to construct this model type.
    type Config: ArchitectureConfig;
    /// Architecture-specific external cache state.
    type State: DecodeState<B>;

    /// Evaluates a stateless, differentiable causal language-model forward pass.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>>;

    /// Allocates external decode state for a known batch and capacity policy.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State>;

    /// Processes a prompt and initializes or extends its decode state.
    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>>;

    /// Processes one or more continuation tokens against existing state.
    fn decode(
        &self,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>>;
}

/// Backend-generic interface for encoder-decoder language models such as T5.
pub trait Seq2SeqLanguageModel<B: Backend>: Module<B> {
    /// Validated architecture configuration used to construct this model type.
    type Config: ArchitectureConfig;
    /// Decoder self-attention and projected cross-attention cache state.
    type State: DecodeState<B>;

    /// Encodes source tokens using bidirectional attention.
    fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>>;

    /// Evaluates the complete stateless encoder-decoder graph.
    fn forward(&self, input: Seq2SeqInput<B>, options: &ForwardOptions)
    -> Result<Seq2SeqOutput<B>>;

    /// Allocates decoder state for a known batch and target capacity.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State>;

    /// Projects encoder output and processes an initial decoder prompt.
    fn prefill_decoder(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>>;

    /// Extends the decoder sequence using self- and cross-attention state.
    fn decode(
        &self,
        encoder: &EncoderOutput<B>,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>>;
}

/// A model whose complete stateless forward pass can run inside a hook scope.
///
/// Generic experiments use it to capture or edit activations without knowing
/// the family. For example, a TransformerLens-style activation cache:
///
/// ```
/// use burn::tensor::backend::Backend;
/// use transformers::{
///     ForwardOptions,
///     architecture::HookedModel,
///     intervention::{DynamicActivation, Hooks, PointInfo},
/// };
///
/// fn run_with_cache<B: Backend, M: HookedModel<B>>(
///     model: &M,
///     input: M::Input,
/// ) -> transformers::Result<(M::Output, Vec<(PointInfo, DynamicActivation<B>)>)> {
///     let mut hooks = Hooks::new();
///     let handles = model
///         .activation_points()?
///         .into_iter()
///         .map(|point| hooks.capture_dynamic(point))
///         .collect::<Vec<_>>();
///     let output = model.forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)?;
///     let cache = handles
///         .iter()
///         .map(|handle| Ok((handle.point().clone(), hooks.get_dynamic(handle)?)))
///         .collect::<transformers::Result<_>>()?;
///     Ok((output, cache))
/// }
/// ```
pub trait HookedModel<B: Backend> {
    /// Complete model input, such as [`CausalLmInput`] or [`Seq2SeqInput`].
    type Input;
    /// Complete model output, such as [`CausalLmOutput`] or [`Seq2SeqOutput`].
    type Output;

    /// Every activation point the forward pass emits, in emission order.
    ///
    /// Returns a capability error when the model, or the variant behind an
    /// automatic loader, does not support hooks.
    fn activation_points(&self) -> Result<Vec<PointInfo>>;

    /// Runs the complete stateless forward pass in its own hook scope.
    ///
    /// Registered actions must name points from [`Self::activation_points`];
    /// edits change the returned output.
    fn forward_with_hooks(
        &self,
        input: Self::Input,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Self::Output>;
}

/// A transformer stack that can be executed in explicit, resumable stages.
///
/// `embed` produces a raw [`HiddenState`] before layer zero, `run_layers`
/// advances it through a contiguous block range starting at
/// `state.next_layer`, and `normalize` applies the final normalization once.
/// Each `*_with_hooks` call is its own hook scope containing only that stage's
/// points; use one [`Hooks`] collection per stage. The provided methods run a
/// stage without hooks or the whole stack:
///
/// ```
/// use burn::tensor::backend::Backend;
/// use transformers::{CausalLmInput, architecture::StagedModel, intervention::Hooks};
///
/// /// Replaces the residual stream entering `layer` with a donor state.
/// fn patch_residual<B: Backend, M: StagedModel<B>>(
///     model: &M,
///     input: CausalLmInput<B>,
///     donor: burn::tensor::Tensor<B, 3>,
///     layer: usize,
/// ) -> transformers::Result<burn::tensor::Tensor<B, 3>> {
///     let state = model.embed(input)?;
///     let mut state = model.run_layers(state, 0..layer)?;
///     state.hidden = donor;
///     let state = model.run_layers(state, layer..model.num_layers())?;
///     Ok(model.normalize(state)?.hidden)
/// }
/// ```
pub trait StagedModel<B: Backend> {
    /// Number of blocks in the staged stack.
    fn num_layers(&self) -> usize;

    /// Stack whose [`Layer`] addresses the staged blocks emit.
    fn stack(&self) -> Stack;

    /// Embeds tokens into a raw state before layer zero, emitting the embedding point.
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>>;

    /// Runs blocks `range`, which must start at `state.next_layer`, in one hook scope.
    fn run_layers_with_hooks(
        &self,
        state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>>;

    /// Applies the final normalization once, emitting the final-norm point.
    fn normalize_with_hooks(
        &self,
        state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>>;

    /// Address of block `index` in this stack.
    fn layer(&self, index: usize) -> Layer {
        self.stack().layer(index)
    }

    /// Embeds tokens without hooks.
    fn embed(&self, input: CausalLmInput<B>) -> Result<HiddenState<B>> {
        self.embed_with_hooks(input, &mut Hooks::new())
    }

    /// Runs a block range without hooks.
    fn run_layers(&self, state: HiddenState<B>, range: Range<usize>) -> Result<HiddenState<B>> {
        self.run_layers_with_hooks(state, range, &mut Hooks::new())
    }

    /// Applies the final normalization without hooks.
    fn normalize(&self, state: HiddenState<B>) -> Result<HiddenState<B>> {
        self.normalize_with_hooks(state, &mut Hooks::new())
    }

    /// Runs the whole stack: embedding, every block, and the final normalization.
    fn run_stack(&self, input: CausalLmInput<B>) -> Result<HiddenState<B>> {
        let state = self.embed(input)?;
        let state = self.run_layers(state, 0..self.num_layers())?;
        self.normalize(state)
    }
}

#[cfg(all(test, feature = "flex"))]
#[path = "traits_tests.rs"]
mod tests;
