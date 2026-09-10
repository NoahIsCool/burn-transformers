//! Typed model inputs, outputs, masks, and diagnostic requests.
//!
//! Inputs carry integer token/position IDs with `[batch, token]` axes; hidden
//! states use `[batch, token, channel]`, and logits append a vocabulary axis.
//! Token masks always mean `true == valid`, including padded decode no-ops.
//! Boolean adapters reconstruct the backend's logical storage through integer
//! indicators instead of assuming a specific byte encoding. These wrappers define
//! shape and semantic contracts; execution validates concrete geometry and devices.

use burn::tensor::{Bool, ElementConversion, Int, Tensor, TensorData, backend::Backend};

use crate::{Error, Result};

/// Creates a backend Boolean tensor from dynamically shaped host validity metadata.
///
/// A logical mask is the indicator function `m(i) in {0, 1}`. Constructing the indicator as an
/// integer tensor and comparing it with zero delegates the physical Boolean representation to the
/// backend. This matters for backends such as Flex, where directly importing a dynamically built
/// `Vec<bool>` can retain a plain `U8` dtype tag that later Boolean operations reject.
pub(crate) fn bool_tensor_from_values<B: Backend, const D: usize>(
    values: Vec<bool>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D, Bool> {
    let indicators = values
        .into_iter()
        .map(|value| if value { 1_i64 } else { 0_i64 })
        .collect::<Vec<_>>();
    Tensor::<B, D, Int>::from_data(TensorData::new(indicators, shape), device).bool()
}

/// Selects Boolean rows while preserving a backend-supported logical dtype.
///
/// Selection is mathematically a gather `y[j] = x[indices[j]]`. Burn 0.21's Flex Boolean gather
/// currently returns bytes tagged as numeric `U8`, so a subsequent logical operation can panic.
/// The reversible Boolean-to-integer conversion performs the same gather and then reconstructs a
/// backend-native Boolean result by comparing every selected indicator with zero.
pub(crate) fn select_bool_tensor<B: Backend, const D: usize>(
    tensor: Tensor<B, D, Bool>,
    dim: usize,
    indices: Tensor<B, 1, Int>,
) -> Tensor<B, D, Bool> {
    tensor.int().select(dim, indices).bool()
}

/// Copies a small Boolean metadata tensor to the host without assuming its storage encoding.
///
/// Burn backends may expose logical booleans as native bits, `u8`, or another backend-selected
/// representation. Converting through the backend's integer element type avoids asking
/// [`TensorData`](burn::tensor::TensorData) to reinterpret one Boolean representation as another;
/// every nonzero integer retains the public `true == valid token` convention.
pub(crate) fn tensor_bool_values<B: Backend, const D: usize>(
    tensor: Tensor<B, D, Bool>,
    label: &str,
) -> Result<Vec<bool>> {
    tensor
        .int()
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to read {label} from backend: {error}"))
        })?
        .to_vec::<B::IntElem>()
        .map(|values| {
            values
                .into_iter()
                .map(|value| value.elem::<i64>() != 0)
                .collect()
        })
        .map_err(|error| {
            Error::BackendCapability(format!("failed to read {label} from backend: {error}"))
        })
}

/// A token-validity mask with the unambiguous convention `true == real token`.
#[derive(Clone, Debug)]
pub struct TokenMask<B: Backend> {
    tensor: Tensor<B, 2, Bool>,
}

impl<B: Backend> TokenMask<B> {
    /// Wraps a `[batch, sequence]` tensor using the valid-token convention.
    pub fn new(tensor: Tensor<B, 2, Bool>) -> Self {
        Self { tensor }
    }

    /// Borrows the underlying Burn tensor without changing mask semantics.
    pub fn as_tensor(&self) -> &Tensor<B, 2, Bool> {
        &self.tensor
    }

    /// Consumes the semantic wrapper and returns its Burn tensor.
    pub fn into_tensor(self) -> Tensor<B, 2, Bool> {
        self.tensor
    }
}

/// Token and position inputs for a causal forward or prefill operation.
#[derive(Clone, Debug)]
pub struct CausalLmInput<B: Backend> {
    /// Integer token IDs shaped `[batch, sequence]`.
    pub token_ids: Tensor<B, 2, Int>,
    /// Optional token-validity mask; absence means every token is valid.
    pub token_mask: Option<TokenMask<B>>,
    /// Optional explicit positions shaped `[batch, sequence]`.
    pub position_ids: Option<Tensor<B, 2, Int>>,
}

/// Continuation tokens supplied to an existing decode state.
#[derive(Clone, Debug)]
pub struct DecodeInput<B: Backend> {
    /// One or more new token IDs shaped `[batch, step]`.
    pub token_ids: Tensor<B, 2, Int>,
    /// Optional explicit positions shaped `[batch, step]`.
    pub position_ids: Option<Tensor<B, 2, Int>>,
    /// Optional validity mask for padded batched decode steps.
    pub token_mask: Option<TokenMask<B>>,
}

/// Selects which sequence positions are projected to vocabulary logits.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogitsSelection {
    /// Project every hidden-state position.
    All,
    /// Project only the last valid position in each sequence.
    #[default]
    Last,
    /// Project an explicit ordered set of zero-based sequence positions.
    Positions(Vec<usize>),
}

/// Controls optional, potentially expensive model diagnostics.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ForwardOptions {
    /// Positions for which vocabulary logits are requested.
    pub logits: LogitsSelection,
    /// Whether to return the embedding output and every layer hidden state.
    pub hidden_states: bool,
    /// Whether to return materialized attention probability tensors.
    pub attentions: bool,
    /// Whether to return pre-top-k router logits for MoE layers.
    pub router_logits: bool,
    /// Whether to return differentiable router auxiliary losses.
    pub auxiliary_losses: bool,
}

/// Diagnostics produced by one mixture-of-experts layer.
#[derive(Clone, Debug)]
pub struct RouterDiagnostics<B: Backend> {
    /// Router logits shaped `[batch, sequence, experts]` when requested.
    pub logits: Option<Tensor<B, 3>>,
    /// Scalar load-balancing loss when requested by the caller.
    pub load_balancing_loss: Option<Tensor<B, 1>>,
    /// Scalar router z-loss when defined by the architecture.
    pub z_loss: Option<Tensor<B, 1>>,
}

/// Output of a causal language-model operation.
#[derive(Clone, Debug)]
pub struct CausalLmOutput<B: Backend> {
    /// Vocabulary logits shaped `[batch, selected_sequence, vocabulary]`.
    pub logits: Tensor<B, 3>,
    /// Final normalized hidden states, retained when requested or needed by MTP.
    pub final_hidden_state: Option<Tensor<B, 3>>,
    /// Embedding and per-layer hidden states when requested.
    pub hidden_states: Vec<Tensor<B, 3>>,
    /// Per-attention-layer probabilities when requested.
    pub attentions: Vec<Tensor<B, 4>>,
    /// Per-MoE-layer router data when requested.
    pub routers: Vec<RouterDiagnostics<B>>,
    /// One architecture-defined stack-wide load-balancing objective, when requested.
    ///
    /// Some reference implementations, notably Qwen3-MoE, define one loss over
    /// the concatenated router outputs from every sparse layer. Such a scalar
    /// cannot be represented truthfully as a per-layer [`RouterDiagnostics`]
    /// value. The objective is returned without the model configuration's
    /// training coefficient; applying that coefficient belongs to the caller's
    /// loss composition policy.
    pub load_balancing_loss: Option<Tensor<B, 1>>,
}

impl<B: Backend> CausalLmOutput<B> {
    /// Creates an output containing vocabulary logits and no optional diagnostics.
    ///
    /// Architecture implementations that do not expose hidden states, attention,
    /// router data, or a stack-wide balancing objective can use this constructor
    /// to make the absence of every diagnostic explicit and future-proof.
    pub fn from_logits(logits: Tensor<B, 3>) -> Self {
        Self {
            logits,
            final_hidden_state: None,
            hidden_states: Vec::new(),
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
        }
    }
}

/// Source-side output retained for encoder-decoder inference.
#[derive(Clone, Debug)]
pub struct EncoderOutput<B: Backend> {
    /// Final encoder states shaped `[batch, source_sequence, hidden]`.
    pub hidden_state: Tensor<B, 3>,
    /// Token-validity mask associated with the source sequence.
    pub token_mask: Option<TokenMask<B>>,
    /// Optional encoder hidden-state diagnostics.
    pub hidden_states: Vec<Tensor<B, 3>>,
    /// Optional encoder attention diagnostics.
    pub attentions: Vec<Tensor<B, 4>>,
}

/// Stateless input to an encoder-decoder model.
#[derive(Clone, Debug)]
pub struct Seq2SeqInput<B: Backend> {
    /// Source tokens processed bidirectionally by the encoder.
    pub encoder: CausalLmInput<B>,
    /// Shifted target tokens processed causally by the decoder.
    pub decoder: CausalLmInput<B>,
}

/// Output of an encoder-decoder model operation.
#[derive(Clone, Debug)]
pub struct Seq2SeqOutput<B: Backend> {
    /// Vocabulary logits and decoder-side diagnostics.
    pub decoder: CausalLmOutput<B>,
    /// Encoder output for reuse across autoregressive decoder steps.
    pub encoder: EncoderOutput<B>,
    /// Optional decoder cross-attention probability tensors.
    pub cross_attentions: Vec<Tensor<B, 4>>,
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
