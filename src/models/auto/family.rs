//! Family-specific steps of strict automatic causal loading.
//!
//! [`super::AutoCausalLm::load_local`] runs one pipeline for every family:
//! parse the configuration, resolve generation defaults, record provenance,
//! load the tokenizer, plan the checkpoint, initialize the graph, and execute
//! the plan. That pipeline lives once in `load::load_causal_family`. A family
//! contributes only what genuinely differs, through [`CausalFamily`]: its
//! configuration type, how its strict tensor contract and load plan are built
//! (namespace probing, provenance policies, optional MTP tensors), and any
//! load-time adjustment of the executable configuration.
//!
//! # Adding a causal family
//!
//! 1. Add one entry to the `causal_registry!` invocation in `auto.rs`; it
//!    generates the dispatch enums, `MODEL_TYPES`, and every `match`.
//! 2. Implement [`CausalFamily`] for the family's model type below.
//!
//! The registry tests then check that the new `model_type` parses, loads, and
//! dispatches like every other family.

use std::path::Path;

use burn::tensor::backend::Backend;

use crate::{
    Error, Result,
    checkpoint::{CheckpointManifest, LoadPlan, TensorSink},
    models::{
        config::{ConfigError, ModelConfig, TokenIds},
        deepseek_v2::{DeepseekV2CheckpointOptions, DeepseekV2Config, DeepseekV2ForCausalLm},
        muse_glimmer::{
            MuseGlimmerCheckpointOptions, MuseGlimmerCheckpointSource, MuseGlimmerForCausalLm,
            MuseGlimmerTextConfig,
        },
        nemotron_h::{
            NemotronHCheckpointOptions, NemotronHCheckpointProvenance, NemotronHConfig,
            NemotronHForCausalLm,
        },
        qwen3_5::{
            Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5ForCausalLm,
            Qwen3_5TextConfig,
        },
        qwen3_moe::{Qwen3MoeCheckpointOptions, Qwen3MoeConfig, Qwen3MoeForCausalLm},
    },
};

use super::AutoLoadOptions;

/// Generation-relevant token metadata read from a family configuration.
#[derive(Debug, Clone, Copy)]
pub(super) struct TokenMetadata<'a> {
    /// Rows of the model's embedding and output projection.
    pub(super) vocabulary_size: usize,
    /// End-of-sequence IDs declared by the model configuration.
    pub(super) eos: Option<&'a TokenIds>,
    /// Padding ID declared by the model configuration.
    pub(super) pad: Option<&'a TokenIds>,
}

impl<'a> TokenMetadata<'a> {
    /// Borrows the three fields every registered configuration declares.
    fn of(vocabulary_size: usize, eos: &'a Option<TokenIds>, pad: &'a Option<TokenIds>) -> Self {
        Self {
            vocabulary_size,
            eos: eos.as_ref(),
            pad: pad.as_ref(),
        }
    }
}

/// A discovered checkpoint and its strict, validated installation plan.
pub(super) struct PlannedCheckpoint {
    /// Complete SafeTensors header inventory of the local checkpoint.
    pub(super) manifest: CheckpointManifest,
    /// Exact tensor-by-tensor installation plan derived from the family contract.
    pub(super) plan: LoadPlan,
}

/// Family-specific steps of the shared automatic causal loading pipeline.
pub(super) trait CausalFamily<B: Backend>: Sized + TensorSink {
    /// Parsed and validated checkpoint configuration.
    type Config: ModelConfig;

    /// Returns the vocabulary size and generation token IDs of a configuration.
    fn token_metadata(config: &Self::Config) -> TokenMetadata<'_>;

    /// Discovers the local checkpoint and plans its strict installation.
    fn plan_checkpoint(
        config: &Self::Config,
        root: &Path,
        options: &AutoLoadOptions,
    ) -> Result<PlannedCheckpoint>;

    /// Builds the executable graph that will receive the planned tensors.
    fn init_for_load(
        config: Self::Config,
        options: &AutoLoadOptions,
        device: &B::Device,
    ) -> Result<Self>;
}

// Shared planning helpers.

/// Maps a family contract error to its `architecture.field` configuration context.
fn contract_error(error: ConfigError) -> Error {
    Error::configuration(
        format!("{}.{}", error.architecture, error.field),
        error.reason,
    )
}

/// Maps a checkpoint discovery or planning failure to a checkpoint error.
fn checkpoint_error(error: impl core::fmt::Display) -> Error {
    Error::Checkpoint(error.to_string())
}

/// Discovers headers with the crate's generic strict artifact policy.
fn discover(root: &Path, options: &AutoLoadOptions) -> Result<CheckpointManifest> {
    CheckpointManifest::discover(root, &options.checkpoint).map_err(checkpoint_error)
}

/// Parameter namespace holding a checkpoint's language model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LanguageNamespace {
    /// Released multimodal layout: `model.language_model.*`.
    Wrapped,
    /// Standalone text layout: `model.embed_tokens.weight` and siblings.
    TextOnly,
}

/// Classifies a validated tensor-name set into exactly one language namespace.
///
/// The classification is total and exclusive: a name set with only wrapped
/// names is [`LanguageNamespace::Wrapped`], one with only the text-only
/// embedding is [`LanguageNamespace::TextOnly`], and having neither or both is
/// an error naming `family`. No filename heuristics are involved.
pub(super) fn classify_language_namespace<'a>(
    names: impl IntoIterator<Item = &'a str>,
    family: &str,
) -> Result<LanguageNamespace> {
    let mut wrapped = false;
    let mut text_only = false;
    for name in names {
        wrapped |= name.starts_with("model.language_model.");
        text_only |= name == "model.embed_tokens.weight";
    }
    match (wrapped, text_only) {
        (true, false) => Ok(LanguageNamespace::Wrapped),
        (false, true) => Ok(LanguageNamespace::TextOnly),
        (false, false) => Err(Error::Checkpoint(format!(
            "{family} checkpoint contains neither official language parameter namespace"
        ))),
        (true, true) => Err(Error::Checkpoint(format!(
            "{family} checkpoint ambiguously contains wrapped and text-only language namespaces"
        ))),
    }
}

/// Classifies a discovered manifest's language namespace.
fn manifest_namespace(manifest: &CheckpointManifest, family: &str) -> Result<LanguageNamespace> {
    classify_language_namespace(manifest.tensors.keys().map(String::as_str), family)
}

// Registered families.

impl<B: Backend> CausalFamily<B> for DeepseekV2ForCausalLm<B> {
    type Config = DeepseekV2Config;

    /// Reads DeepSeek-V2's vocabulary and token fields.
    fn token_metadata(config: &DeepseekV2Config) -> TokenMetadata<'_> {
        TokenMetadata::of(
            config.vocab_size,
            &config.eos_token_id,
            &config.pad_token_id,
        )
    }

    /// Plans the single published DeepSeek-V2 namespace.
    fn plan_checkpoint(
        config: &DeepseekV2Config,
        root: &Path,
        options: &AutoLoadOptions,
    ) -> Result<PlannedCheckpoint> {
        let manifest = discover(root, options)?;
        let contract = config
            .checkpoint_contract(DeepseekV2CheckpointOptions::default())
            .map_err(contract_error)?;
        let plan = contract
            .build_load_plan(&manifest, &options.checkpoint)
            .map_err(checkpoint_error)?;
        Ok(PlannedCheckpoint { manifest, plan })
    }

    /// Initializes the graph from the unchanged configuration.
    fn init_for_load(
        config: DeepseekV2Config,
        _options: &AutoLoadOptions,
        device: &B::Device,
    ) -> Result<Self> {
        Self::init(config, device)
    }
}

impl<B: Backend> CausalFamily<B> for MuseGlimmerForCausalLm<B> {
    type Config = MuseGlimmerTextConfig;

    /// Reads Muse Glimmer's vocabulary and token fields.
    fn token_metadata(config: &MuseGlimmerTextConfig) -> TokenMetadata<'_> {
        TokenMetadata::of(
            config.vocab_size,
            &config.eos_token_id,
            &config.pad_token_id,
        )
    }

    /// Plans the multimodal or text-only namespace present in the checkpoint.
    fn plan_checkpoint(
        config: &MuseGlimmerTextConfig,
        root: &Path,
        options: &AutoLoadOptions,
    ) -> Result<PlannedCheckpoint> {
        let manifest = discover(root, options)?;
        let source = match manifest_namespace(&manifest, "Muse")? {
            LanguageNamespace::Wrapped => MuseGlimmerCheckpointSource::Multimodal,
            LanguageNamespace::TextOnly => MuseGlimmerCheckpointSource::TextOnly,
        };
        let contract = config
            .checkpoint_contract(MuseGlimmerCheckpointOptions { source })
            .map_err(contract_error)?;
        let plan = contract
            .build_load_plan(&manifest, &options.checkpoint)
            .map_err(checkpoint_error)?;
        Ok(PlannedCheckpoint { manifest, plan })
    }

    /// Initializes the graph from the unchanged configuration.
    fn init_for_load(
        config: MuseGlimmerTextConfig,
        _options: &AutoLoadOptions,
        device: &B::Device,
    ) -> Result<Self> {
        Self::init(config, device)
    }
}

impl<B: Backend> CausalFamily<B> for NemotronHForCausalLm<B> {
    type Config = NemotronHConfig;

    /// Reads Nemotron-H's vocabulary and token fields.
    fn token_metadata(config: &NemotronHConfig) -> TokenMetadata<'_> {
        TokenMetadata::of(
            config.vocab_size,
            &config.eos_token_id,
            &config.pad_token_id,
        )
    }

    /// Plans with Nemotron's own discovery, which knows pinned NVIDIA provenance.
    ///
    /// When both a repository and a revision are recorded, the contract may
    /// apply the accounting exception reviewed for that exact NVIDIA artifact;
    /// otherwise discovery uses the generic strict policy.
    fn plan_checkpoint(
        config: &NemotronHConfig,
        root: &Path,
        options: &AutoLoadOptions,
    ) -> Result<PlannedCheckpoint> {
        let architecture_options = match (&options.repository, &options.revision) {
            (Some(repository), Some(revision)) => NemotronHCheckpointOptions::default()
                .with_provenance(NemotronHCheckpointProvenance::new(repository, revision)),
            _ => NemotronHCheckpointOptions::default(),
        };
        let contract = config
            .checkpoint_contract(architecture_options)
            .map_err(contract_error)?;
        let manifest = contract
            .discover_checkpoint(root, &options.checkpoint)
            .map_err(checkpoint_error)?;
        let plan = contract
            .build_load_plan(&manifest, &options.checkpoint)
            .map_err(checkpoint_error)?;
        Ok(PlannedCheckpoint { manifest, plan })
    }

    /// Initializes the graph from the unchanged configuration.
    fn init_for_load(
        config: NemotronHConfig,
        _options: &AutoLoadOptions,
        device: &B::Device,
    ) -> Result<Self> {
        Self::init(config, device)
    }
}

impl<B: Backend> CausalFamily<B> for Qwen3_5ForCausalLm<B> {
    type Config = Qwen3_5TextConfig;

    /// Reads Qwen3.5's vocabulary and token fields.
    fn token_metadata(config: &Qwen3_5TextConfig) -> TokenMetadata<'_> {
        TokenMetadata::of(
            config.vocab_size,
            &config.eos_token_id,
            &config.pad_token_id,
        )
    }

    /// Plans the namespace present in the checkpoint, with MTP tensors on request.
    fn plan_checkpoint(
        config: &Qwen3_5TextConfig,
        root: &Path,
        options: &AutoLoadOptions,
    ) -> Result<PlannedCheckpoint> {
        let manifest = discover(root, options)?;
        let source = match manifest_namespace(&manifest, "Qwen")? {
            LanguageNamespace::Wrapped => Qwen3_5CheckpointSource::Multimodal,
            LanguageNamespace::TextOnly => Qwen3_5CheckpointSource::TextOnly,
        };
        let contract = config
            .checkpoint_contract(Qwen3_5CheckpointOptions {
                source,
                include_mtp: options.qwen_mtp,
            })
            .map_err(contract_error)?;
        let plan = contract
            .build_load_plan(&manifest, &options.checkpoint)
            .map_err(checkpoint_error)?;
        Ok(PlannedCheckpoint { manifest, plan })
    }

    /// Drops the MTP graph unless its tensors were planned.
    ///
    /// Base-only loading must not leave a randomly initialized MTP graph
    /// reachable through the public speculative API.
    fn init_for_load(
        mut config: Qwen3_5TextConfig,
        options: &AutoLoadOptions,
        device: &B::Device,
    ) -> Result<Self> {
        if !options.qwen_mtp {
            config.mtp_num_hidden_layers = 0;
        }
        Self::init(config, device)
    }
}

impl<B: Backend> CausalFamily<B> for Qwen3MoeForCausalLm<B> {
    type Config = Qwen3MoeConfig;

    /// Reads Qwen3-MoE's vocabulary and token fields.
    fn token_metadata(config: &Qwen3MoeConfig) -> TokenMetadata<'_> {
        TokenMetadata::of(
            config.vocab_size,
            &config.eos_token_id,
            &config.pad_token_id,
        )
    }

    /// Plans the single published Qwen3-MoE namespace.
    fn plan_checkpoint(
        config: &Qwen3MoeConfig,
        root: &Path,
        options: &AutoLoadOptions,
    ) -> Result<PlannedCheckpoint> {
        let manifest = discover(root, options)?;
        let contract = config
            .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
            .map_err(contract_error)?;
        let plan = contract
            .build_load_plan(&manifest, &options.checkpoint)
            .map_err(checkpoint_error)?;
        Ok(PlannedCheckpoint { manifest, plan })
    }

    /// Initializes the graph from the unchanged configuration.
    fn init_for_load(
        config: Qwen3MoeConfig,
        _options: &AutoLoadOptions,
        device: &B::Device,
    ) -> Result<Self> {
        Self::init(config, device)
    }
}

#[cfg(test)]
#[path = "family_tests.rs"]
mod tests;
