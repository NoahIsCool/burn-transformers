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
    Seq2SeqOutput,
    cache::CacheConfig,
    intervention::{Hooks, PointInfo},
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

    /// Lists the stateless activation points supported by the selected model.
    ///
    /// Qwen3.5/Qwen3.8 currently supports interventions. Other registered causal
    /// families return a capability error rather than an empty, misleading catalog.
    /// Loaded checkpoints expose this method through `loaded.model()`.
    pub fn activation_points(&self) -> Result<Vec<PointInfo>> {
        match self {
            Self::Qwen3_5(model) => Ok(model.activation_points()),
            _ => Err(self.unsupported_hooks_error()),
        }
    }

    /// Observes or changes activations during the selected model's stateless forward.
    ///
    /// Qwen3.5/Qwen3.8 delegates to its concrete intervention implementation.
    /// Other families fail before callbacks or model computation, including when
    /// the collection is empty. Use [`Self::forward`] for ordinary execution.
    /// Failed calls invalidate captures left from earlier runs of these hooks.
    pub fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        match self {
            Self::Qwen3_5(model) => model.forward_with_hooks(input, options, hooks),
            _ => hooks
                .run(Vec::new(), |_| Err(self.unsupported_hooks_error()))
                // Add family context even when preflight rejects a named point.
                .map_err(|_| self.unsupported_hooks_error()),
        }
    }

    fn unsupported_hooks_error(&self) -> Error {
        Error::BackendCapability(format!(
            "activation interventions are not implemented for model_type `{}`; the automatic causal registry currently supports Qwen3.5/Qwen3.8",
            self.config().model_type()
        ))
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

    /// Lists the selected model's encoder and decoder activation points.
    /// Loaded checkpoints expose this method through `loaded.model()`.
    pub fn activation_points(&self) -> Result<Vec<PointInfo>> {
        match self {
            Self::T5(model) => Ok(model.activation_points()),
        }
    }

    /// Observes or changes tensors during the complete stateless encoder-decoder graph.
    ///
    /// Delegates to T5's public intervention path, including its documented mask
    /// and cache restrictions; see [`T5ForConditionalGeneration::forward_with_hooks`].
    pub fn forward_with_hooks(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Seq2SeqOutput<B>> {
        match self {
            Self::T5(model) => model.forward_with_hooks(input, options, hooks),
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
#[path = "auto_tests.rs"]
mod tests;
