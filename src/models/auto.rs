//! Closed, backend-generic dispatch over executable model families.
//!
//! Checkpoint family selection belongs at the runtime boundary, not inside
//! shared attention or feed-forward implementations. [`AutoCausalLm`] and
//! [`AutoSeq2SeqLm`] therefore contain concrete modules and dispatch each public
//! operation by enum variant. Their matching state enums prevent a cache built
//! for one architecture from being passed to another architecture by accident.
//!
//! The registry advertises only executable graphs. Configuration-only schemas
//! for families still under construction remain available from their family
//! modules, but are intentionally rejected here until their forward and
//! checkpoint paths meet the same contract.

use burn::{module::Module, tensor::backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, DecodeState, EncoderOutput,
    Error, ForwardOptions, Result, Seq2SeqInput, Seq2SeqLanguageModel, Seq2SeqOutput,
    cache::{CacheConfig, TensorDecodeState},
};

use super::{
    qwen3_5::{Qwen3_5DecodeState, Qwen3_5ForCausalLm, Qwen3_5TextConfig},
    t5::{T5Config, T5ForConditionalGeneration},
};
use crate::models::config::ModelConfig;

/// Immutable architecture configuration borrowed from an automatic causal model.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum AutoCausalConfigRef<'a> {
    /// Qwen3.5/Qwen3.8 hybrid text configuration.
    Qwen3_5(&'a Qwen3_5TextConfig),
}

impl AutoCausalConfigRef<'_> {
    /// Returns the exact checkpoint `model_type` discriminator.
    pub fn model_type(self) -> &'static str {
        match self {
            Self::Qwen3_5(_) => "qwen3_5_text",
        }
    }

    /// Returns the model vocabulary size.
    pub fn vocabulary_size(self) -> usize {
        match self {
            Self::Qwen3_5(config) => config.vocab_size,
        }
    }
}

/// Immutable architecture configuration borrowed from an automatic seq2seq model.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum AutoSeq2SeqConfigRef<'a> {
    /// T5, FLAN-T5, T5 v1.1, or UL2 configuration.
    T5(&'a T5Config),
}

impl AutoSeq2SeqConfigRef<'_> {
    /// Returns the exact checkpoint `model_type` discriminator.
    pub fn model_type(self) -> &'static str {
        match self {
            Self::T5(_) => "t5",
        }
    }

    /// Returns the model vocabulary size.
    pub fn vocabulary_size(self) -> usize {
        match self {
            Self::T5(config) => config.vocab_size,
        }
    }
}

/// External decode state paired with an [`AutoCausalLm`] variant.
#[derive(Debug)]
#[non_exhaustive]
pub enum AutoCausalDecodeState<B: Backend> {
    /// Qwen native-KV and DeltaNet recurrent state.
    Qwen3_5(Qwen3_5DecodeState<B>),
}

impl<B: Backend> DecodeState<B> for AutoCausalDecodeState<B> {
    fn batch_size(&self) -> usize {
        match self {
            Self::Qwen3_5(state) => state.batch_size(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Qwen3_5(state) => state.capacity(),
        }
    }

    fn lengths(&self) -> &[usize] {
        match self {
            Self::Qwen3_5(state) => state.lengths(),
        }
    }

    fn reset(&mut self) -> Result<()> {
        match self {
            Self::Qwen3_5(state) => state.reset(),
        }
    }

    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        match self {
            Self::Qwen3_5(state) => state.reorder(indices),
        }
    }

    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        match self {
            Self::Qwen3_5(state) => state.truncate(lengths),
        }
    }

    fn device(&self) -> &B::Device {
        match self {
            Self::Qwen3_5(state) => state.device(),
        }
    }
}

/// External decoder/cross-attention state paired with [`AutoSeq2SeqLm`].
#[derive(Debug)]
#[non_exhaustive]
pub enum AutoSeq2SeqDecodeState<B: Backend> {
    /// T5 decoder state. Construction remains unavailable until cached T5 lands.
    T5(TensorDecodeState<B>),
}

impl<B: Backend> DecodeState<B> for AutoSeq2SeqDecodeState<B> {
    fn batch_size(&self) -> usize {
        match self {
            Self::T5(state) => state.batch_size(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::T5(state) => state.capacity(),
        }
    }

    fn lengths(&self) -> &[usize] {
        match self {
            Self::T5(state) => state.lengths(),
        }
    }

    fn reset(&mut self) -> Result<()> {
        match self {
            Self::T5(state) => DecodeState::reset(state),
        }
    }

    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        match self {
            Self::T5(state) => DecodeState::reorder(state, indices),
        }
    }

    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        match self {
            Self::T5(state) => DecodeState::truncate(state, lengths),
        }
    }

    fn device(&self) -> &B::Device {
        match self {
            Self::T5(state) => state.device(),
        }
    }
}

/// Automatic dispatch over all causal architectures with executable graphs.
#[derive(Module, Debug)]
#[non_exhaustive]
pub enum AutoCausalLm<B: Backend> {
    /// Qwen3.5/Qwen3.8 gated-DeltaNet and full-attention hybrid decoder.
    Qwen3_5(Qwen3_5ForCausalLm<B>),
}

impl<B: Backend> AutoCausalLm<B> {
    /// Checkpoint model types currently accepted by this executable registry.
    pub const MODEL_TYPES: &'static [&'static str] = &["qwen3_5_text"];

    /// Parses a direct or wrapped configuration and initializes its lazy graph.
    pub fn from_config_json(json: &str, device: &B::Device) -> Result<Self> {
        match detect_model_type(json)?.as_str() {
            "qwen3_5_text" => {
                let config = Qwen3_5TextConfig::from_json_str(json).map_err(|error| {
                    Error::configuration("auto_causal.config", error.to_string())
                })?;
                Qwen3_5ForCausalLm::init(config, device).map(Self::Qwen3_5)
            }
            model_type => Err(Error::UnsupportedArchitecture(format!(
                "causal model_type `{model_type}` is not executable; supported values: {}",
                Self::MODEL_TYPES.join(", ")
            ))),
        }
    }

    /// Returns the concrete architecture configuration without cloning it.
    pub fn config(&self) -> AutoCausalConfigRef<'_> {
        match self {
            Self::Qwen3_5(model) => AutoCausalConfigRef::Qwen3_5(model.config()),
        }
    }

    /// Evaluates a stateless causal forward pass through the selected variant.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        match self {
            Self::Qwen3_5(model) => CausalLanguageModel::forward(model, input, options),
        }
    }

    /// Allocates architecture-matched external decode state.
    pub fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<AutoCausalDecodeState<B>> {
        match self {
            Self::Qwen3_5(model) => {
                CausalLanguageModel::new_decode_state(model, config, batch_size, device)
                    .map(AutoCausalDecodeState::Qwen3_5)
            }
        }
    }

    /// Prefills architecture-matched state from a prompt.
    pub fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut AutoCausalDecodeState<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        match (self, state) {
            (Self::Qwen3_5(model), AutoCausalDecodeState::Qwen3_5(state)) => {
                CausalLanguageModel::prefill(model, input, state, options)
            }
        }
    }

    /// Decodes one or more continuation tokens against matched state.
    pub fn decode(
        &self,
        input: DecodeInput<B>,
        state: &mut AutoCausalDecodeState<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        match (self, state) {
            (Self::Qwen3_5(model), AutoCausalDecodeState::Qwen3_5(state)) => {
                CausalLanguageModel::decode(model, input, state, options)
            }
        }
    }
}

impl<B: Backend> From<Qwen3_5ForCausalLm<B>> for AutoCausalLm<B> {
    fn from(model: Qwen3_5ForCausalLm<B>) -> Self {
        Self::Qwen3_5(model)
    }
}

/// Automatic dispatch over all encoder-decoder architectures with executable graphs.
#[derive(Module, Debug)]
#[non_exhaustive]
pub enum AutoSeq2SeqLm<B: Backend> {
    /// T5, FLAN-T5, T5 v1.1, and UL2 encoder-decoder graph.
    T5(T5ForConditionalGeneration<B>),
}

impl<B: Backend> AutoSeq2SeqLm<B> {
    /// Checkpoint model types currently accepted by this executable registry.
    pub const MODEL_TYPES: &'static [&'static str] = &["t5"];

    /// Parses a direct configuration and initializes its lazy graph.
    pub fn from_config_json(json: &str, device: &B::Device) -> Result<Self> {
        match detect_model_type(json)?.as_str() {
            "t5" => {
                let config = T5Config::from_json_str(json).map_err(|error| {
                    Error::configuration("auto_seq2seq.config", error.to_string())
                })?;
                T5ForConditionalGeneration::init(config, device).map(Self::T5)
            }
            model_type => Err(Error::UnsupportedArchitecture(format!(
                "seq2seq model_type `{model_type}` is not executable; supported values: {}",
                Self::MODEL_TYPES.join(", ")
            ))),
        }
    }

    /// Returns the concrete architecture configuration without cloning it.
    pub fn config(&self) -> AutoSeq2SeqConfigRef<'_> {
        match self {
            Self::T5(model) => AutoSeq2SeqConfigRef::T5(model.config()),
        }
    }

    /// Encodes source tokens bidirectionally through the selected variant.
    pub fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        match self {
            Self::T5(model) => Seq2SeqLanguageModel::encode(model, source, options),
        }
    }

    /// Evaluates the complete stateless encoder-decoder graph.
    pub fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        match self {
            Self::T5(model) => Seq2SeqLanguageModel::forward(model, input, options),
        }
    }

    /// Allocates architecture-matched decoder state when that path is available.
    pub fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<AutoSeq2SeqDecodeState<B>> {
        match self {
            Self::T5(model) => {
                Seq2SeqLanguageModel::new_decode_state(model, config, batch_size, device)
                    .map(AutoSeq2SeqDecodeState::T5)
            }
        }
    }

    /// Prefills an architecture-matched decoder from reusable encoder output.
    pub fn prefill_decoder(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        state: &mut AutoSeq2SeqDecodeState<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        match (self, state) {
            (Self::T5(model), AutoSeq2SeqDecodeState::T5(state)) => {
                Seq2SeqLanguageModel::prefill_decoder(model, encoder, decoder, state, options)
            }
        }
    }

    /// Decodes continuation tokens using matched encoder and decoder state.
    pub fn decode(
        &self,
        encoder: &EncoderOutput<B>,
        input: DecodeInput<B>,
        state: &mut AutoSeq2SeqDecodeState<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        match (self, state) {
            (Self::T5(model), AutoSeq2SeqDecodeState::T5(state)) => {
                Seq2SeqLanguageModel::decode(model, encoder, input, state, options)
            }
        }
    }
}

impl<B: Backend> From<T5ForConditionalGeneration<B>> for AutoSeq2SeqLm<B> {
    fn from(model: T5ForConditionalGeneration<B>) -> Self {
        Self::T5(model)
    }
}

/// Extracts the architecture discriminator from direct or text-wrapped JSON.
fn detect_model_type(json: &str) -> Result<String> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|error| {
        Error::configuration("auto.config_json", format!("invalid JSON: {error}"))
    })?;
    let candidate = ["text_config", "language_config", "llm_config"]
        .iter()
        .find_map(|key| value.get(*key).filter(|nested| !nested.is_null()))
        .unwrap_or(&value);
    candidate
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            Error::configuration(
                "auto.model_type",
                "configuration must contain a string model_type",
            )
        })
}

#[cfg(test)]
mod tests {
    use burn::backend::Flex;

    use super::*;
    use crate::models::{qwen3_5::Qwen3_5LayerType, t5::T5FeedForwardProjection};

    type TestBackend = Flex<f32, i32>;

    /// Constructs a tiny validated hybrid configuration for registry tests.
    fn tiny_qwen() -> Qwen3_5TextConfig {
        Qwen3_5TextConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 64,
            partial_rotary_factor: Some(0.5),
            layer_types: vec![
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::FullAttention,
            ],
            linear_conv_kernel_dim: 2,
            linear_key_head_dim: 2,
            linear_value_head_dim: 2,
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            mtp_num_hidden_layers: 0,
            ..Qwen3_5TextConfig::default()
        }
    }

    /// Constructs a tiny validated T5 configuration for registry tests.
    fn tiny_t5() -> T5Config {
        T5Config {
            vocab_size: 32,
            d_model: 8,
            d_kv: 4,
            d_ff: 16,
            num_layers: 1,
            num_decoder_layers: Some(1),
            num_heads: 2,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 16,
            dropout_rate: 0.0,
            feed_forward_proj: T5FeedForwardProjection::Relu,
            ..T5Config::default()
        }
    }

    /// Wrapped Qwen config detection prefers the language tower and allocates state.
    #[test]
    fn causal_registry_parses_wrapped_config_and_dispatches_state() {
        let device = Default::default();
        let wrapped = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": tiny_qwen(),
        });
        let model = AutoCausalLm::<TestBackend>::from_config_json(&wrapped.to_string(), &device)
            .expect("registered Qwen config");
        assert_eq!(model.config().model_type(), "qwen3_5_text");
        assert_eq!(model.config().vocabulary_size(), 32);
        let state = model
            .new_decode_state(CacheConfig::new(1, 8).expect("valid cache"), 1, &device)
            .expect("registered state");
        assert_eq!(state.batch_size(), 1);
        assert_eq!(state.capacity(), 8);
        assert_eq!(state.lengths(), [0]);
    }

    /// T5 registry initializes the stateless graph and preserves its config view.
    #[test]
    fn seq2seq_registry_parses_t5_and_reports_cached_boundary() {
        let device = Default::default();
        let json = serde_json::to_string(&tiny_t5()).expect("serializable config");
        let model = AutoSeq2SeqLm::<TestBackend>::from_config_json(&json, &device)
            .expect("registered T5 config");
        assert_eq!(model.config().model_type(), "t5");
        assert_eq!(model.config().vocabulary_size(), 32);
        assert!(
            model
                .new_decode_state(CacheConfig::new(1, 8).expect("valid cache"), 1, &device)
                .is_err(),
            "auto dispatch must preserve T5's explicit cached-decoder boundary"
        );
    }

    /// Malformed, missing, and non-executable model types fail contextually.
    #[test]
    fn registry_rejects_unknown_or_malformed_configurations() {
        let device = Default::default();
        for json in ["{", "{}", r#"{"model_type": 7}"#] {
            assert!(AutoCausalLm::<TestBackend>::from_config_json(json, &device).is_err());
        }
        let error = AutoCausalLm::<TestBackend>::from_config_json(
            r#"{"model_type":"deepseek_v2"}"#,
            &device,
        )
        .expect_err("config-only family must not be advertised as executable");
        assert!(matches!(error, Error::UnsupportedArchitecture(_)));
    }
}
