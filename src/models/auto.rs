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
    ArchitectureConfig, CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput,
    DecodeState, EncoderOutput, Error, ForwardOptions, Result, Seq2SeqInput, Seq2SeqLanguageModel,
    Seq2SeqOutput, cache::CacheConfig,
};

use super::{
    deepseek_v2::{DeepseekV2Config, DeepseekV2DecodeState, DeepseekV2ForCausalLm},
    muse_glimmer::{MuseGlimmerDecodeState, MuseGlimmerForCausalLm, MuseGlimmerTextConfig},
    nemotron_h::{NemotronHConfig, NemotronHDecodeState, NemotronHForCausalLm},
    qwen3_5::{Qwen3_5DecodeState, Qwen3_5ForCausalLm, Qwen3_5TextConfig},
    qwen3_moe::{Qwen3MoeConfig, Qwen3MoeDecodeState, Qwen3MoeForCausalLm},
    t5::{T5Config, T5DecodeState, T5ForConditionalGeneration},
};
use crate::models::config::ModelConfig;

mod load;

pub use load::AutoLoadOptions;

/// Registry-level configuration associated with [`CausalLanguageModel`].
///
/// The detailed checkpoint configuration remains available through
/// [`AutoCausalLm::config`]. This lightweight discriminator lets generic code
/// use the architecture trait without duplicating a model's complete config
/// inside the dispatch enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AutoCausalConfig {
    /// DeepSeek-V2 MLA and sparse-expert runtime.
    DeepseekV2,
    /// Muse Glimmer hybrid-attention runtime.
    MuseGlimmer,
    /// NVIDIA Nemotron-H heterogeneous runtime.
    NemotronH,
    /// Qwen3.5/Qwen3.8 hybrid runtime.
    Qwen3_5,
    /// Qwen3 sparse-MoE runtime.
    Qwen3Moe,
}

impl ArchitectureConfig for AutoCausalConfig {
    const MODEL_TYPES: &'static [&'static str] = &[
        "deepseek_v2",
        "muse_glimmer_text",
        "nemotron_h",
        "qwen3_5_text",
        "qwen3_moe",
    ];

    /// Registry discriminators are constructed only for executable families.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Registry-level configuration associated with [`Seq2SeqLanguageModel`].
///
/// The complete checkpoint configuration remains available through
/// [`AutoSeq2SeqLm::config`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AutoSeq2SeqConfig {
    /// T5, FLAN-T5, T5 v1.1, and UL2 runtime.
    T5,
}

impl ArchitectureConfig for AutoSeq2SeqConfig {
    const MODEL_TYPES: &'static [&'static str] = &["t5"];

    /// Registry discriminators are constructed only for executable families.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Immutable architecture configuration borrowed from an automatic causal model.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum AutoCausalConfigRef<'a> {
    /// DeepSeek-V2 MLA and sparse-expert configuration.
    DeepseekV2(&'a DeepseekV2Config),
    /// Muse Glimmer dense hybrid text configuration.
    MuseGlimmer(&'a MuseGlimmerTextConfig),
    /// NVIDIA Nemotron-H heterogeneous block configuration.
    NemotronH(&'a NemotronHConfig),
    /// Qwen3.5/Qwen3.8 hybrid text configuration.
    Qwen3_5(&'a Qwen3_5TextConfig),
    /// Qwen3 sparse mixture-of-experts decoder configuration.
    Qwen3Moe(&'a Qwen3MoeConfig),
}

impl AutoCausalConfigRef<'_> {
    /// Returns the exact checkpoint `model_type` discriminator.
    pub fn model_type(self) -> &'static str {
        match self {
            Self::DeepseekV2(_) => "deepseek_v2",
            Self::MuseGlimmer(_) => "muse_glimmer_text",
            Self::NemotronH(_) => "nemotron_h",
            Self::Qwen3_5(_) => "qwen3_5_text",
            Self::Qwen3Moe(_) => "qwen3_moe",
        }
    }

    /// Returns the model vocabulary size.
    pub fn vocabulary_size(self) -> usize {
        match self {
            Self::DeepseekV2(config) => config.vocab_size,
            Self::MuseGlimmer(config) => config.vocab_size,
            Self::NemotronH(config) => config.vocab_size,
            Self::Qwen3_5(config) => config.vocab_size,
            Self::Qwen3Moe(config) => config.vocab_size,
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
    /// DeepSeek-V2 compressed MLA and recurrent length state.
    DeepseekV2(Box<DeepseekV2DecodeState<B>>),
    /// Muse sliding-ring and full-attention KV state.
    MuseGlimmer(Box<MuseGlimmerDecodeState<B>>),
    /// Nemotron-H Mamba2 and grouped-query attention state.
    NemotronH(Box<NemotronHDecodeState<B>>),
    /// Qwen native-KV and DeltaNet recurrent state.
    Qwen3_5(Box<Qwen3_5DecodeState<B>>),
    /// Qwen3 MoE native grouped-query KV state.
    Qwen3Moe(Box<Qwen3MoeDecodeState<B>>),
}

impl<B: Backend> DecodeState<B> for AutoCausalDecodeState<B> {
    fn batch_size(&self) -> usize {
        match self {
            Self::DeepseekV2(state) => state.batch_size(),
            Self::MuseGlimmer(state) => state.batch_size(),
            Self::NemotronH(state) => state.batch_size(),
            Self::Qwen3_5(state) => state.batch_size(),
            Self::Qwen3Moe(state) => state.batch_size(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::DeepseekV2(state) => state.capacity(),
            Self::MuseGlimmer(state) => state.capacity(),
            Self::NemotronH(state) => state.capacity(),
            Self::Qwen3_5(state) => state.capacity(),
            Self::Qwen3Moe(state) => state.capacity(),
        }
    }

    fn lengths(&self) -> &[usize] {
        match self {
            Self::DeepseekV2(state) => state.lengths(),
            Self::MuseGlimmer(state) => state.lengths(),
            Self::NemotronH(state) => state.lengths(),
            Self::Qwen3_5(state) => state.lengths(),
            Self::Qwen3Moe(state) => state.lengths(),
        }
    }

    fn reset(&mut self) -> Result<()> {
        match self {
            Self::DeepseekV2(state) => state.reset(),
            Self::MuseGlimmer(state) => state.reset(),
            Self::NemotronH(state) => state.reset(),
            Self::Qwen3_5(state) => state.reset(),
            Self::Qwen3Moe(state) => state.reset(),
        }
    }

    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        match self {
            Self::DeepseekV2(state) => state.reorder(indices),
            Self::MuseGlimmer(state) => state.reorder(indices),
            Self::NemotronH(state) => state.reorder(indices),
            Self::Qwen3_5(state) => state.reorder(indices),
            Self::Qwen3Moe(state) => state.reorder(indices),
        }
    }

    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        match self {
            Self::DeepseekV2(state) => state.truncate(lengths),
            Self::MuseGlimmer(state) => state.truncate(lengths),
            Self::NemotronH(state) => state.truncate(lengths),
            Self::Qwen3_5(state) => state.truncate(lengths),
            Self::Qwen3Moe(state) => state.truncate(lengths),
        }
    }

    fn device(&self) -> &B::Device {
        match self {
            Self::DeepseekV2(state) => state.device(),
            Self::MuseGlimmer(state) => state.device(),
            Self::NemotronH(state) => state.device(),
            Self::Qwen3_5(state) => state.device(),
            Self::Qwen3Moe(state) => state.device(),
        }
    }
}

/// External decoder/cross-attention state paired with [`AutoSeq2SeqLm`].
#[derive(Debug)]
#[non_exhaustive]
pub enum AutoSeq2SeqDecodeState<B: Backend> {
    /// T5 decoder self-attention and immutable encoder cross-attention state.
    T5(T5DecodeState<B>),
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
// Burn 0.21 does not implement `Module` for `Box<M>`, so boxing the parameter
// graphs would prevent derive-generated device/dtype/parameter traversal.
#[allow(clippy::large_enum_variant)]
pub enum AutoCausalLm<B: Backend> {
    /// DeepSeek-V2 decoder with multi-head latent attention and sparse experts.
    DeepseekV2(DeepseekV2ForCausalLm<B>),
    /// Muse Glimmer dense decoder with sliding/full hybrid attention.
    MuseGlimmer(MuseGlimmerForCausalLm<B>),
    /// Nemotron-H decoder with interleaved Mamba2, attention, dense, and MoE blocks.
    NemotronH(NemotronHForCausalLm<B>),
    /// Qwen3.5/Qwen3.8 gated-DeltaNet and full-attention hybrid decoder.
    Qwen3_5(Qwen3_5ForCausalLm<B>),
    /// Qwen3 decoder with token-routed sparse SwiGLU experts.
    Qwen3Moe(Qwen3MoeForCausalLm<B>),
}

impl<B: Backend> AutoCausalLm<B> {
    /// Checkpoint model types currently accepted by this executable registry.
    pub const MODEL_TYPES: &'static [&'static str] = &[
        "deepseek_v2",
        "muse_glimmer_text",
        "nemotron_h",
        "qwen3_5_text",
        "qwen3_moe",
    ];

    /// Parses a direct or wrapped configuration and initializes its lazy graph.
    pub fn from_config_json(json: &str, device: &B::Device) -> Result<Self> {
        match detect_model_type(json)?.as_str() {
            "deepseek_v2" => {
                let config = DeepseekV2Config::from_json_str(json).map_err(|error| {
                    Error::configuration("auto_causal.config", error.to_string())
                })?;
                DeepseekV2ForCausalLm::init(config, device).map(Self::DeepseekV2)
            }
            "muse_glimmer_text" => {
                let config = MuseGlimmerTextConfig::from_json_str(json).map_err(|error| {
                    Error::configuration("auto_causal.config", error.to_string())
                })?;
                MuseGlimmerForCausalLm::init(config, device).map(Self::MuseGlimmer)
            }
            "nemotron_h" => {
                let config = NemotronHConfig::from_json_str(json).map_err(|error| {
                    Error::configuration("auto_causal.config", error.to_string())
                })?;
                NemotronHForCausalLm::init(config, device).map(Self::NemotronH)
            }
            "qwen3_5_text" => {
                let config = Qwen3_5TextConfig::from_json_str(json).map_err(|error| {
                    Error::configuration("auto_causal.config", error.to_string())
                })?;
                Qwen3_5ForCausalLm::init(config, device).map(Self::Qwen3_5)
            }
            "qwen3_moe" => {
                let config = Qwen3MoeConfig::from_json_str(json).map_err(|error| {
                    Error::configuration("auto_causal.config", error.to_string())
                })?;
                Qwen3MoeForCausalLm::init(config, device).map(Self::Qwen3Moe)
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
            Self::DeepseekV2(model) => AutoCausalConfigRef::DeepseekV2(model.config()),
            Self::MuseGlimmer(model) => AutoCausalConfigRef::MuseGlimmer(model.config()),
            Self::NemotronH(model) => AutoCausalConfigRef::NemotronH(model.config()),
            Self::Qwen3_5(model) => AutoCausalConfigRef::Qwen3_5(model.config()),
            Self::Qwen3Moe(model) => AutoCausalConfigRef::Qwen3Moe(model.config()),
        }
    }

    /// Evaluates a stateless causal forward pass through the selected variant.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        match self {
            Self::DeepseekV2(model) => CausalLanguageModel::forward(model, input, options),
            Self::MuseGlimmer(model) => CausalLanguageModel::forward(model, input, options),
            Self::NemotronH(model) => CausalLanguageModel::forward(model, input, options),
            Self::Qwen3_5(model) => CausalLanguageModel::forward(model, input, options),
            Self::Qwen3Moe(model) => CausalLanguageModel::forward(model, input, options),
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
            Self::DeepseekV2(model) => {
                CausalLanguageModel::new_decode_state(model, config, batch_size, device)
                    .map(Box::new)
                    .map(AutoCausalDecodeState::DeepseekV2)
            }
            Self::MuseGlimmer(model) => {
                CausalLanguageModel::new_decode_state(model, config, batch_size, device)
                    .map(Box::new)
                    .map(AutoCausalDecodeState::MuseGlimmer)
            }
            Self::NemotronH(model) => {
                CausalLanguageModel::new_decode_state(model, config, batch_size, device)
                    .map(Box::new)
                    .map(AutoCausalDecodeState::NemotronH)
            }
            Self::Qwen3_5(model) => {
                CausalLanguageModel::new_decode_state(model, config, batch_size, device)
                    .map(Box::new)
                    .map(AutoCausalDecodeState::Qwen3_5)
            }
            Self::Qwen3Moe(model) => {
                CausalLanguageModel::new_decode_state(model, config, batch_size, device)
                    .map(Box::new)
                    .map(AutoCausalDecodeState::Qwen3Moe)
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
            (Self::DeepseekV2(model), AutoCausalDecodeState::DeepseekV2(state)) => {
                CausalLanguageModel::prefill(model, input, state.as_mut(), options)
            }
            (Self::MuseGlimmer(model), AutoCausalDecodeState::MuseGlimmer(state)) => {
                CausalLanguageModel::prefill(model, input, state.as_mut(), options)
            }
            (Self::NemotronH(model), AutoCausalDecodeState::NemotronH(state)) => {
                CausalLanguageModel::prefill(model, input, state.as_mut(), options)
            }
            (Self::Qwen3_5(model), AutoCausalDecodeState::Qwen3_5(state)) => {
                CausalLanguageModel::prefill(model, input, state.as_mut(), options)
            }
            (Self::Qwen3Moe(model), AutoCausalDecodeState::Qwen3Moe(state)) => {
                CausalLanguageModel::prefill(model, input, state.as_mut(), options)
            }
            _ => Err(Error::Cache(
                "automatic causal decode state belongs to a different architecture".into(),
            )),
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
            (Self::DeepseekV2(model), AutoCausalDecodeState::DeepseekV2(state)) => {
                CausalLanguageModel::decode(model, input, state.as_mut(), options)
            }
            (Self::MuseGlimmer(model), AutoCausalDecodeState::MuseGlimmer(state)) => {
                CausalLanguageModel::decode(model, input, state.as_mut(), options)
            }
            (Self::NemotronH(model), AutoCausalDecodeState::NemotronH(state)) => {
                CausalLanguageModel::decode(model, input, state.as_mut(), options)
            }
            (Self::Qwen3_5(model), AutoCausalDecodeState::Qwen3_5(state)) => {
                CausalLanguageModel::decode(model, input, state.as_mut(), options)
            }
            (Self::Qwen3Moe(model), AutoCausalDecodeState::Qwen3Moe(state)) => {
                CausalLanguageModel::decode(model, input, state.as_mut(), options)
            }
            _ => Err(Error::Cache(
                "automatic causal decode state belongs to a different architecture".into(),
            )),
        }
    }
}

impl<B: Backend> CausalLanguageModel<B> for AutoCausalLm<B> {
    type Config = AutoCausalConfig;
    type State = AutoCausalDecodeState<B>;

    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        AutoCausalLm::forward(self, input, options)
    }

    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        AutoCausalLm::new_decode_state(self, config, batch_size, device)
    }

    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        AutoCausalLm::prefill(self, input, state, options)
    }

    fn decode(
        &self,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        AutoCausalLm::decode(self, input, state, options)
    }
}

impl<B: Backend> From<DeepseekV2ForCausalLm<B>> for AutoCausalLm<B> {
    fn from(model: DeepseekV2ForCausalLm<B>) -> Self {
        Self::DeepseekV2(model)
    }
}

impl<B: Backend> From<MuseGlimmerForCausalLm<B>> for AutoCausalLm<B> {
    fn from(model: MuseGlimmerForCausalLm<B>) -> Self {
        Self::MuseGlimmer(model)
    }
}

impl<B: Backend> From<NemotronHForCausalLm<B>> for AutoCausalLm<B> {
    fn from(model: NemotronHForCausalLm<B>) -> Self {
        Self::NemotronH(model)
    }
}

impl<B: Backend> From<Qwen3_5ForCausalLm<B>> for AutoCausalLm<B> {
    fn from(model: Qwen3_5ForCausalLm<B>) -> Self {
        Self::Qwen3_5(model)
    }
}

impl<B: Backend> From<Qwen3MoeForCausalLm<B>> for AutoCausalLm<B> {
    fn from(model: Qwen3MoeForCausalLm<B>) -> Self {
        Self::Qwen3Moe(model)
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

    /// Allocates architecture-matched decoder state for the selected model.
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

impl<B: Backend> Seq2SeqLanguageModel<B> for AutoSeq2SeqLm<B> {
    type Config = AutoSeq2SeqConfig;
    type State = AutoSeq2SeqDecodeState<B>;

    fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        AutoSeq2SeqLm::encode(self, source, options)
    }

    fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        AutoSeq2SeqLm::forward(self, input, options)
    }

    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        AutoSeq2SeqLm::new_decode_state(self, config, batch_size, device)
    }

    fn prefill_decoder(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        AutoSeq2SeqLm::prefill_decoder(self, encoder, decoder, state, options)
    }

    fn decode(
        &self,
        encoder: &EncoderOutput<B>,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        AutoSeq2SeqLm::decode(self, encoder, input, state, options)
    }
}

impl<B: Backend> From<T5ForConditionalGeneration<B>> for AutoSeq2SeqLm<B> {
    fn from(model: T5ForConditionalGeneration<B>) -> Self {
        Self::T5(model)
    }
}

/// Extracts the architecture discriminator from direct or text-wrapped JSON.
pub(super) fn detect_model_type(json: &str) -> Result<String> {
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
    use burn::{
        backend::Flex,
        tensor::{Int, Tensor},
    };

    use super::*;
    use crate::models::{
        muse_glimmer::MuseGlimmerLayerType, nemotron_h::NemotronHBlockType,
        qwen3_5::Qwen3_5LayerType, t5::T5FeedForwardProjection,
    };
    use crate::{CausalGenerationRequest, GenerationConfig, generate_causal};

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

    /// Constructs a tiny validated sparse Qwen configuration for registry tests.
    fn tiny_qwen_moe() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 64,
            max_window_layers: 1,
            decoder_sparse_step: 1,
            moe_intermediate_size: 12,
            num_experts: 4,
            num_experts_per_tok: 2,
            ..Qwen3MoeConfig::default()
        }
    }

    /// Constructs a tiny Muse decoder retaining local and global attention.
    fn tiny_muse() -> MuseGlimmerTextConfig {
        MuseGlimmerTextConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 32,
            sliding_window: Some(3),
            bos_token_id: Some(crate::models::config::TokenIds::One(1)),
            eos_token_id: Some(crate::models::config::TokenIds::One(2)),
            layer_types: vec![
                MuseGlimmerLayerType::SlidingAttention,
                MuseGlimmerLayerType::SlidingAttention,
                MuseGlimmerLayerType::SlidingAttention,
                MuseGlimmerLayerType::FullAttention,
            ],
            ..MuseGlimmerTextConfig::default()
        }
    }

    /// Constructs a tiny Nemotron schedule retaining every executable block kind.
    fn tiny_nemotron() -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 32,
            hidden_size: 8,
            layers_block_type: vec![
                NemotronHBlockType::LinearAttention,
                NemotronHBlockType::FullAttention,
                NemotronHBlockType::Mlp,
                NemotronHBlockType::Moe,
            ],
            num_hidden_layers: Some(4),
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 32,
            intermediate_size: 12,
            ssm_state_size: 3,
            mamba_num_heads: 4,
            mamba_head_dim: 2,
            n_groups: 2,
            conv_kernel: 3,
            chunk_size: 2,
            n_routed_experts: 4,
            n_shared_experts: 1,
            moe_intermediate_size: 6,
            moe_shared_expert_intermediate_size: 8,
            num_experts_per_tok: 2,
            n_group: 2,
            topk_group: 1,
            num_nextn_predict_layers: 0,
            ..NemotronHConfig::default()
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

    /// Sparse Qwen config selection reaches its executable graph and cache variant.
    #[test]
    fn causal_registry_dispatches_qwen3_moe() {
        let device = Default::default();
        let json = serde_json::to_string(&tiny_qwen_moe()).expect("serializable config");
        let model = AutoCausalLm::<TestBackend>::from_config_json(&json, &device)
            .expect("registered Qwen3 MoE config");
        assert_eq!(model.config().model_type(), "qwen3_moe");
        assert_eq!(model.config().vocabulary_size(), 32);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).expect("valid cache"), 1, &device)
            .expect("registered state");
        assert!(matches!(state, AutoCausalDecodeState::Qwen3Moe(_)));
        assert_eq!(state.lengths(), [0]);
        state.reset().expect("empty state reset");
    }

    /// Remaining v1 causal families select their concrete graphs and state variants.
    #[test]
    fn causal_registry_dispatches_deepseek_muse_and_nemotron() {
        let device = Default::default();

        let deepseek_config = crate::models::deepseek_v2::model::tests_support::tiny_config();
        let deepseek = AutoCausalLm::<TestBackend>::from_config_json(
            &serde_json::to_string(&deepseek_config).expect("serializable DeepSeek config"),
            &device,
        )
        .expect("registered DeepSeek config");
        assert_eq!(deepseek.config().model_type(), "deepseek_v2");
        assert!(matches!(
            deepseek
                .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
                .expect("DeepSeek state"),
            AutoCausalDecodeState::DeepseekV2(_)
        ));

        let muse = AutoCausalLm::<TestBackend>::from_config_json(
            &serde_json::to_string(&tiny_muse()).expect("serializable Muse config"),
            &device,
        )
        .expect("registered Muse config");
        assert_eq!(muse.config().model_type(), "muse_glimmer_text");
        assert!(matches!(
            muse.new_decode_state(
                CacheConfig::new(1, 8)
                    .unwrap()
                    .with_sliding_window(3)
                    .unwrap(),
                1,
                &device,
            )
            .expect("Muse state"),
            AutoCausalDecodeState::MuseGlimmer(_)
        ));

        let nemotron = AutoCausalLm::<TestBackend>::from_config_json(
            &serde_json::to_string(&tiny_nemotron()).expect("serializable Nemotron config"),
            &device,
        )
        .expect("registered Nemotron config");
        assert_eq!(nemotron.config().model_type(), "nemotron_h");
        assert!(matches!(
            nemotron
                .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
                .expect("Nemotron state"),
            AutoCausalDecodeState::NemotronH(_)
        ));
    }

    /// The automatic causal enum can be passed directly to generic generation.
    #[test]
    fn causal_registry_implements_generation_contract_end_to_end() {
        let device = Default::default();
        let concrete = crate::models::deepseek_v2::model::tests_support::initialized_model(
            crate::models::deepseek_v2::model::tests_support::tiny_config(),
            &device,
        );
        let model = AutoCausalLm::from(concrete);
        let output = generate_causal(
            &model,
            CausalGenerationRequest {
                prompt: CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                cache: CacheConfig::new(1, 8).unwrap(),
                generation: GenerationConfig {
                    max_new_tokens: 2,
                    ..Default::default()
                },
                prefill_chunk_size: None,
                vocabulary_size: 32,
            },
        )
        .expect("auto-dispatched cached generation");
        assert_eq!(output.sequences[0].token_ids.len(), 2);
    }

    /// Cross-family state mismatches fail without advancing the foreign cache.
    #[test]
    fn causal_registry_rejects_mismatched_state_atomically() {
        let device = Default::default();
        let deepseek = AutoCausalLm::from(
            crate::models::deepseek_v2::model::tests_support::initialized_model(
                crate::models::deepseek_v2::model::tests_support::tiny_config(),
                &device,
            ),
        );
        let qwen = AutoCausalLm::from(
            Qwen3_5ForCausalLm::init(tiny_qwen(), &device).expect("tiny Qwen model"),
        );
        let mut foreign_state = qwen
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        let error = deepseek
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut foreign_state,
                &ForwardOptions::default(),
            )
            .expect_err("foreign state must be rejected");
        assert!(matches!(error, Error::Cache(_)));
        assert_eq!(foreign_state.lengths(), [0]);
    }

    /// T5 registry initializes the graph and allocates architecture-matched state.
    #[test]
    fn seq2seq_registry_parses_t5_and_allocates_cached_state() {
        let device = Default::default();
        let json = serde_json::to_string(&tiny_t5()).expect("serializable config");
        let model = AutoSeq2SeqLm::<TestBackend>::from_config_json(&json, &device)
            .expect("registered T5 config");
        assert_eq!(model.config().model_type(), "t5");
        assert_eq!(model.config().vocabulary_size(), 32);
        let state = model
            .new_decode_state(CacheConfig::new(1, 8).expect("valid cache"), 1, &device)
            .expect("registered T5 state");
        assert_eq!(state.batch_size(), 1);
        assert_eq!(state.capacity(), 8);
    }

    /// Malformed, missing, and non-executable model types fail contextually.
    #[test]
    fn registry_rejects_unknown_or_malformed_configurations() {
        let device = Default::default();
        for json in ["{", "{}", r#"{"model_type": 7}"#] {
            assert!(AutoCausalLm::<TestBackend>::from_config_json(json, &device).is_err());
        }
        let error =
            AutoCausalLm::<TestBackend>::from_config_json(r#"{"model_type":"bert"}"#, &device)
                .expect_err("unregistered family must not be advertised as executable");
        assert!(matches!(error, Error::UnsupportedArchitecture(_)));
        assert!(
            AutoCausalLm::<TestBackend>::from_config_json(
                r#"{"model_type":"deepseek_v2","hidden_size":0}"#,
                &device,
            )
            .is_err()
        );
    }
}
