//! Traits implemented by concrete causal and encoder-decoder architectures.

use burn::{module::Module, tensor::backend::Backend};

use crate::{Result, architecture::types::*};

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
    /// Architecture configuration associated with this model.
    type Config: ArchitectureConfig;
    /// Architecture-specific external cache state.
    type State: DecodeState<B>;

    /// Returns the validated architecture configuration.
    fn config(&self) -> &Self::Config;

    /// Evaluates a stateless, differentiable causal language-model forward pass.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>>;

    /// Allocates external decode state for a known batch and capacity policy.
    fn new_decode_state(
        &self,
        capacity: usize,
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
    /// Architecture configuration associated with this model.
    type Config: ArchitectureConfig;
    /// Decoder self-attention and projected cross-attention cache state.
    type State: DecodeState<B>;

    /// Returns the validated architecture configuration.
    fn config(&self) -> &Self::Config;

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
        capacity: usize,
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
