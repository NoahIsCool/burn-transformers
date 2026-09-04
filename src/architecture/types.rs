//! Typed model inputs, outputs, masks, and diagnostic requests.

use burn::tensor::{Bool, ElementConversion, Int, Tensor, backend::Backend};

use crate::{Error, Result};

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
mod tests {
    use super::{ForwardOptions, LogitsSelection};

    #[cfg(feature = "flex")]
    use burn::backend::Flex;
    #[cfg(feature = "metal")]
    use burn::backend::Metal;
    #[cfg(any(feature = "flex", feature = "metal"))]
    use burn::tensor::{Bool, Tensor, TensorData};

    #[cfg(feature = "flex")]
    type TestBackend = Flex<f32, i32>;

    /// Boolean host extraction preserves logical values independently of physical storage.
    #[cfg(feature = "flex")]
    #[test]
    fn boolean_metadata_uses_backend_portable_integer_conversion() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2, Bool>::from_data(
            TensorData::new(vec![true, false, false, true], [2, 2]),
            &device,
        );
        assert_eq!(
            super::tensor_bool_values(tensor, "test mask").unwrap(),
            [true, false, false, true]
        );
    }

    /// Metal's native Boolean readback remains portable through the integer conversion path.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a host-visible Metal adapter"]
    fn metal_boolean_metadata_does_not_assume_u8_tensor_data() {
        let device = Default::default();
        let tensor = Tensor::<Metal, 2, Bool>::from_data(
            TensorData::new(vec![true, false, false, true], [2, 2]),
            &device,
        );
        assert_eq!(
            super::tensor_bool_values(tensor, "Metal test mask").unwrap(),
            [true, false, false, true]
        );
    }

    /// Ensures inference avoids a full vocabulary projection unless explicitly requested.
    #[test]
    fn forward_options_default_to_last_logits_only() {
        let options = ForwardOptions::default();
        assert_eq!(options.logits, LogitsSelection::Last);
        assert!(!options.hidden_states);
        assert!(!options.attentions);
        assert!(!options.router_logits);
        assert!(!options.auxiliary_losses);
    }

    /// Verifies an explicit position list survives serde round trips without ambiguity.
    #[test]
    fn logits_selection_round_trips_through_json() {
        let selection = LogitsSelection::Positions(vec![0, 3, 8]);
        let encoded = serde_json::to_string(&selection).expect("selection should serialize");
        let decoded: LogitsSelection =
            serde_json::from_str(&encoded).expect("selection should deserialize");
        assert_eq!(decoded, selection);
    }
}
