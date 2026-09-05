//! Local-only automatic checkpoint loading and generation-default normalization.
//!
//! The automatic registry performs family selection once, then hands strict
//! tensor planning and installation back to the architecture-owned checkpoint
//! contract. No filename heuristics select individual parameters. Qwen's two
//! published namespace layouts are distinguished only after complete SafeTensors
//! header discovery by checking which exact language prefix is present.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::{
    Error, GenerationConfig, LoadedModel, ModelSourceMetadata, ModelTokenizer, Result,
    SamplingStrategy,
    checkpoint::{
        CheckpointManifest, ExecutionOutcome, LoadOptions, execute_local_load_plan,
        quantization::{
            QuantizedBurnpackArtifact, QuantizedExecutionOutcome, execute_local_quantized_load_plan,
        },
    },
};

use super::{AutoCausalLm, AutoSeq2SeqLm, detect_model_type};
use crate::models::{
    config::{ModelConfig, TokenIds},
    deepseek_v2::{DeepseekV2CheckpointOptions, DeepseekV2Config, DeepseekV2ForCausalLm},
    muse_glimmer::{
        MuseGlimmerCheckpointOptions, MuseGlimmerCheckpointSource, MuseGlimmerForCausalLm,
        MuseGlimmerTextConfig,
    },
    nemotron_h::{NemotronHCheckpointOptions, NemotronHConfig, NemotronHForCausalLm},
    qwen3_5::{
        Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5ForCausalLm, Qwen3_5TextConfig,
    },
    qwen3_moe::{Qwen3MoeCheckpointOptions, Qwen3MoeConfig, Qwen3MoeForCausalLm},
    t5::{T5Config, T5ForConditionalGeneration},
};
use burn::tensor::backend::Backend;

const MAX_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_GENERATION_CONFIG_BYTES: u64 = 1024 * 1024;

/// Host-side policy for automatic local model loading.
#[derive(Debug, Clone)]
pub struct AutoLoadOptions {
    /// Strict SafeTensors discovery, dtype, and exception policy.
    pub checkpoint: LoadOptions,
    /// Explicit normalized generation defaults, bypassing local defaults when set.
    pub generation: Option<GenerationConfig>,
    /// Load `tokenizer.json` when the crate feature and artifact are both present.
    pub load_tokenizer: bool,
    /// Load Qwen's optional 15-tensor MTP graph instead of the base LM alone.
    pub qwen_mtp: bool,
    /// Optional packed weight sidecar; relative paths resolve below the model root.
    pub quantized_burnpack: Option<PathBuf>,
    /// Optional upstream repository recorded in [`ModelSourceMetadata`].
    pub repository: Option<String>,
    /// Optional immutable upstream revision recorded in source metadata.
    pub revision: Option<String>,
}

impl Default for AutoLoadOptions {
    fn default() -> Self {
        Self {
            checkpoint: LoadOptions::default(),
            generation: None,
            load_tokenizer: true,
            qwen_mtp: false,
            quantized_burnpack: None,
            repository: None,
            revision: None,
        }
    }
}

impl<B: Backend> AutoCausalLm<B> {
    /// Strictly loads one supported causal model from an existing local directory.
    ///
    /// The method never contacts a model hub. Configuration, tokenizer, and all
    /// SafeTensors files must already exist below `root`. It returns a
    /// [`LoadedModel`] only after the load report has no unexplained mismatch.
    pub fn load_local(
        root: impl AsRef<Path>,
        options: &AutoLoadOptions,
        device: &B::Device,
    ) -> Result<LoadedModel<Self>> {
        let (root, json) = read_required_config(root.as_ref())?;
        let model_type = detect_model_type(&json)?;
        if options.qwen_mtp && model_type != "qwen3_5_text" {
            return Err(Error::configuration(
                "auto_causal.qwen_mtp",
                "Qwen MTP is defined only for qwen3_5_text checkpoints",
            ));
        }
        if options.quantized_burnpack.is_some() && model_type != "qwen3_5_text" {
            return Err(Error::BackendCapability(format!(
                "packed Burnpack loading for `{model_type}` is not yet certified; v1 currently enables automatic packed execution only for qwen3_5_text"
            )));
        }
        match model_type.as_str() {
            "deepseek_v2" => {
                let config = DeepseekV2Config::from_json_str(&json).map_err(|error| {
                    Error::configuration("auto_causal.config", error.to_string())
                })?;
                let vocabulary_size = config.vocab_size;
                let generation = normalize_generation_defaults(
                    &root,
                    config.eos_token_id.as_ref(),
                    config.pad_token_id.as_ref(),
                    vocabulary_size,
                    options.generation.as_ref(),
                )?;
                let source = source_metadata(&root, &model_type, options)?;
                let tokenizer = load_optional_tokenizer(&root, options.load_tokenizer)?;
                let manifest = CheckpointManifest::discover(&root, &options.checkpoint)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?;
                let contract = config
                    .checkpoint_contract(DeepseekV2CheckpointOptions::default())
                    .map_err(|error| {
                        Error::configuration(
                            format!("{}.{}", error.architecture, error.field),
                            error.reason,
                        )
                    })?;
                let plan = contract
                    .build_load_plan(&manifest, &options.checkpoint)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?;
                let mut model = DeepseekV2ForCausalLm::init(config, device)?;
                let outcome = execute_causal_plan(&root, &manifest, &plan, options, &mut model)?;
                LoadedModel::new_with_execution_stats(
                    Self::DeepseekV2(model),
                    tokenizer,
                    generation,
                    source,
                    outcome.report,
                    outcome.stats,
                    vocabulary_size,
                )
            }
            "muse_glimmer_text" => {
                let config = MuseGlimmerTextConfig::from_json_str(&json).map_err(|error| {
                    Error::configuration("auto_causal.config", error.to_string())
                })?;
                let vocabulary_size = config.vocab_size;
                let generation = normalize_generation_defaults(
                    &root,
                    config.eos_token_id.as_ref(),
                    config.pad_token_id.as_ref(),
                    vocabulary_size,
                    options.generation.as_ref(),
                )?;
                let source = source_metadata(&root, &model_type, options)?;
                let tokenizer = load_optional_tokenizer(&root, options.load_tokenizer)?;
                let manifest = CheckpointManifest::discover(&root, &options.checkpoint)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?;
                let source_layout = muse_source_layout(&manifest)?;
                let contract = config
                    .checkpoint_contract(MuseGlimmerCheckpointOptions {
                        source: source_layout,
                    })
                    .map_err(|error| {
                        Error::configuration(
                            format!("{}.{}", error.architecture, error.field),
                            error.reason,
                        )
                    })?;
                let plan = contract
                    .build_load_plan(&manifest, &options.checkpoint)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?;
                let mut model = MuseGlimmerForCausalLm::init(config, device)?;
                let outcome = execute_causal_plan(&root, &manifest, &plan, options, &mut model)?;
                LoadedModel::new_with_execution_stats(
                    Self::MuseGlimmer(model),
                    tokenizer,
                    generation,
                    source,
                    outcome.report,
                    outcome.stats,
                    vocabulary_size,
                )
            }
            "nemotron_h" => {
                let config = NemotronHConfig::from_json_str(&json).map_err(|error| {
                    Error::configuration("auto_causal.config", error.to_string())
                })?;
                let vocabulary_size = config.vocab_size;
                let generation = normalize_generation_defaults(
                    &root,
                    config.eos_token_id.as_ref(),
                    config.pad_token_id.as_ref(),
                    vocabulary_size,
                    options.generation.as_ref(),
                )?;
                let source = source_metadata(&root, &model_type, options)?;
                let tokenizer = load_optional_tokenizer(&root, options.load_tokenizer)?;
                let manifest = CheckpointManifest::discover(&root, &options.checkpoint)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?;
                let contract = config
                    .checkpoint_contract(NemotronHCheckpointOptions::default())
                    .map_err(|error| {
                        Error::configuration(
                            format!("{}.{}", error.architecture, error.field),
                            error.reason,
                        )
                    })?;
                let plan = contract
                    .build_load_plan(&manifest, &options.checkpoint)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?;
                let mut model = NemotronHForCausalLm::init(config, device)?;
                let outcome = execute_causal_plan(&root, &manifest, &plan, options, &mut model)?;
                LoadedModel::new_with_execution_stats(
                    Self::NemotronH(model),
                    tokenizer,
                    generation,
                    source,
                    outcome.report,
                    outcome.stats,
                    vocabulary_size,
                )
            }
            "qwen3_5_text" => {
                let config = Qwen3_5TextConfig::from_json_str(&json).map_err(|error| {
                    Error::configuration("auto_causal.config", error.to_string())
                })?;
                let vocabulary_size = config.vocab_size;
                let generation = normalize_generation_defaults(
                    &root,
                    config.eos_token_id.as_ref(),
                    config.pad_token_id.as_ref(),
                    vocabulary_size,
                    options.generation.as_ref(),
                )?;
                let source = source_metadata(&root, &model_type, options)?;
                let tokenizer = load_optional_tokenizer(&root, options.load_tokenizer)?;
                let manifest = CheckpointManifest::discover(&root, &options.checkpoint)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?;
                let source_layout = qwen_source_layout(&manifest)?;
                let contract = config
                    .checkpoint_contract(Qwen3_5CheckpointOptions {
                        source: source_layout,
                        include_mtp: options.qwen_mtp,
                    })
                    .map_err(|error| {
                        Error::configuration(
                            format!("{}.{}", error.architecture, error.field),
                            error.reason,
                        )
                    })?;
                let plan = contract
                    .build_load_plan(&manifest, &options.checkpoint)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?;
                let mut executable_config = config;
                if !options.qwen_mtp {
                    // Base-only loading must not leave a randomly initialized MTP graph
                    // reachable through the public speculative API.
                    executable_config.mtp_num_hidden_layers = 0;
                }
                let mut model = Qwen3_5ForCausalLm::init(executable_config, device)?;
                let outcome = execute_causal_plan(&root, &manifest, &plan, options, &mut model)?;
                LoadedModel::new_with_execution_stats(
                    Self::Qwen3_5(model),
                    tokenizer,
                    generation,
                    source,
                    outcome.report,
                    outcome.stats,
                    vocabulary_size,
                )
            }
            "qwen3_moe" => {
                let config = Qwen3MoeConfig::from_json_str(&json).map_err(|error| {
                    Error::configuration("auto_causal.config", error.to_string())
                })?;
                let vocabulary_size = config.vocab_size;
                let generation = normalize_generation_defaults(
                    &root,
                    config.eos_token_id.as_ref(),
                    config.pad_token_id.as_ref(),
                    vocabulary_size,
                    options.generation.as_ref(),
                )?;
                let source = source_metadata(&root, &model_type, options)?;
                let tokenizer = load_optional_tokenizer(&root, options.load_tokenizer)?;
                let manifest = CheckpointManifest::discover(&root, &options.checkpoint)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?;
                let contract = config
                    .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
                    .map_err(|error| {
                        Error::configuration(
                            format!("{}.{}", error.architecture, error.field),
                            error.reason,
                        )
                    })?;
                let plan = contract
                    .build_load_plan(&manifest, &options.checkpoint)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?;
                let mut model = Qwen3MoeForCausalLm::init(config, device)?;
                let outcome = execute_causal_plan(&root, &manifest, &plan, options, &mut model)?;
                LoadedModel::new_with_execution_stats(
                    Self::Qwen3Moe(model),
                    tokenizer,
                    generation,
                    source,
                    outcome.report,
                    outcome.stats,
                    vocabulary_size,
                )
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "causal model_type `{model_type}` is not executable; supported values: {}",
                Self::MODEL_TYPES.join(", ")
            ))),
        }
    }
}

/// Executes a validated causal-family plan with an optional packed sidecar.
fn execute_causal_plan<M: crate::checkpoint::TensorSink>(
    root: &Path,
    manifest: &CheckpointManifest,
    plan: &crate::checkpoint::LoadPlan,
    options: &AutoLoadOptions,
    model: &mut M,
) -> Result<ExecutionOutcome>
where
    M::Error: core::fmt::Display,
{
    if let Some(path) = &options.quantized_burnpack {
        let path = if path.is_absolute() {
            path.clone()
        } else {
            root.join(path)
        };
        let mut artifact =
            QuantizedBurnpackArtifact::open(&path).map_err(|error| Error::Artifact {
                path: path.clone(),
                reason: error.to_string(),
            })?;
        execute_local_quantized_load_plan(manifest, plan, &mut artifact, model)
            .map_err(|error| Error::Checkpoint(error.to_string()))
            .map(checkpoint_execution_from_quantized)
    } else {
        execute_local_load_plan(manifest, plan, model)
            .map_err(|error| Error::Checkpoint(error.to_string()))
    }
}

/// Retains generic source execution counters from one packed-overlay load.
///
/// Burnpack reads have a separate [`crate::checkpoint::quantization::PackedLoadStats`]
/// audit at the lower-level API. [`LoadedModel`] carries the checkpoint executor's
/// [`crate::checkpoint::ExecutionStats`], whose source and host-staging counters
/// have identical meaning for full-precision and packed-overlay execution.
fn checkpoint_execution_from_quantized(outcome: QuantizedExecutionOutcome) -> ExecutionOutcome {
    ExecutionOutcome {
        report: outcome.report,
        stats: outcome.execution,
    }
}

impl<B: Backend> AutoSeq2SeqLm<B> {
    /// Strictly loads one supported encoder-decoder model from a local directory.
    pub fn load_local(
        root: impl AsRef<Path>,
        options: &AutoLoadOptions,
        device: &B::Device,
    ) -> Result<LoadedModel<Self>> {
        if options.quantized_burnpack.is_some() {
            return Err(Error::BackendCapability(
                "packed Burnpack loading is currently certified only for Qwen causal models".into(),
            ));
        }
        let (root, json) = read_required_config(root.as_ref())?;
        let model_type = detect_model_type(&json)?;
        if model_type != "t5" {
            return Err(Error::UnsupportedArchitecture(format!(
                "seq2seq model_type `{model_type}` is not executable; supported values: {}",
                Self::MODEL_TYPES.join(", ")
            )));
        }

        let config = T5Config::from_json_str(&json)
            .map_err(|error| Error::configuration("auto_seq2seq.config", error.to_string()))?;
        let vocabulary_size = config.vocab_size;
        let generation = normalize_generation_defaults(
            &root,
            config.eos_token_id.as_ref(),
            config.pad_token_id.as_ref(),
            vocabulary_size,
            options.generation.as_ref(),
        )?;
        let source = source_metadata(&root, &model_type, options)?;
        let tokenizer = load_optional_tokenizer(&root, options.load_tokenizer)?;
        let mut model = T5ForConditionalGeneration::init(config, device)?;
        let outcome = model
            .load_checkpoint(&root, &options.checkpoint)
            .map_err(|error| Error::Checkpoint(error.to_string()))?;
        LoadedModel::new_with_execution_stats(
            Self::T5(model),
            tokenizer,
            generation,
            source,
            outcome.report,
            outcome.stats,
            vocabulary_size,
        )
    }
}

/// Distinguishes official wrapped and text-only Qwen parameter namespaces.
fn qwen_source_layout(manifest: &CheckpointManifest) -> Result<Qwen3_5CheckpointSource> {
    classify_qwen_names(manifest.tensors.keys().map(String::as_str))
}

/// Distinguishes released multimodal and standalone Muse text namespaces.
fn muse_source_layout(manifest: &CheckpointManifest) -> Result<MuseGlimmerCheckpointSource> {
    classify_muse_names(manifest.tensors.keys().map(String::as_str))
}

/// Classifies a previously validated set of checkpoint tensor names.
fn classify_qwen_names<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Result<Qwen3_5CheckpointSource> {
    let mut wrapped = false;
    let mut text_only = false;
    for name in names {
        wrapped |= name.starts_with("model.language_model.");
        text_only |= name == "model.embed_tokens.weight";
    }
    match (wrapped, text_only) {
        (true, false) => Ok(Qwen3_5CheckpointSource::Multimodal),
        (false, true) => Ok(Qwen3_5CheckpointSource::TextOnly),
        (false, false) => Err(Error::Checkpoint(
            "Qwen checkpoint contains neither official language parameter namespace".into(),
        )),
        (true, true) => Err(Error::Checkpoint(
            "Qwen checkpoint ambiguously contains wrapped and text-only language namespaces".into(),
        )),
    }
}

/// Classifies a validated Muse checkpoint key set without filename heuristics.
fn classify_muse_names<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Result<MuseGlimmerCheckpointSource> {
    let mut wrapped = false;
    let mut text_only = false;
    for name in names {
        wrapped |= name.starts_with("model.language_model.");
        text_only |= name == "model.embed_tokens.weight";
    }
    match (wrapped, text_only) {
        (true, false) => Ok(MuseGlimmerCheckpointSource::Multimodal),
        (false, true) => Ok(MuseGlimmerCheckpointSource::TextOnly),
        (false, false) => Err(Error::Checkpoint(
            "Muse checkpoint contains neither official language parameter namespace".into(),
        )),
        (true, true) => Err(Error::Checkpoint(
            "Muse checkpoint ambiguously contains wrapped and text-only language namespaces".into(),
        )),
    }
}

/// Builds validated source metadata before expensive tensor materialization.
fn source_metadata(
    root: &Path,
    model_type: &str,
    options: &AutoLoadOptions,
) -> Result<ModelSourceMetadata> {
    let mut source = ModelSourceMetadata::local(root, model_type)?;
    if let Some(repository) = &options.repository {
        source = source.with_repository(repository.clone())?;
    }
    if let Some(revision) = &options.revision {
        source = source.with_revision(revision.clone())?;
    }
    Ok(source)
}

/// Reads a required bounded `config.json` and returns its canonical root.
fn read_required_config(root: &Path) -> Result<(PathBuf, String)> {
    let root = root
        .canonicalize()
        .map_err(|source| Error::io(root, source))?;
    if !root.is_dir() {
        return Err(Error::Artifact {
            path: root,
            reason: "model source root must be a directory".into(),
        });
    }
    let path =
        contained_regular_file(&root, "config.json", MAX_CONFIG_BYTES, true)?.ok_or_else(|| {
            Error::Artifact {
                path: root.join("config.json"),
                reason: "required config.json was not resolved".into(),
            }
        })?;
    let json = fs::read_to_string(&path).map_err(|source| Error::io(&path, source))?;
    if json.len() as u64 > MAX_CONFIG_BYTES {
        return Err(Error::Artifact {
            path,
            reason: format!("config exceeds {MAX_CONFIG_BYTES} bytes while being read"),
        });
    }
    Ok((root, json))
}

/// Loads a tokenizer only when both its feature and primary artifact are present.
#[cfg(feature = "tokenizers")]
fn load_optional_tokenizer(root: &Path, enabled: bool) -> Result<Option<ModelTokenizer>> {
    if !enabled || contained_regular_file(root, "tokenizer.json", u64::MAX, false)?.is_none() {
        return Ok(None);
    }
    crate::LocalTokenizer::from_directory(root).map(Some)
}

/// Feature-disabled builds preserve the bundle API with an always-empty slot.
#[cfg(not(feature = "tokenizers"))]
fn load_optional_tokenizer(_root: &Path, _enabled: bool) -> Result<Option<ModelTokenizer>> {
    Ok(None)
}

/// Normalizes model tokens and optional Hugging Face generation metadata.
fn normalize_generation_defaults(
    root: &Path,
    model_eos: Option<&TokenIds>,
    model_pad: Option<&TokenIds>,
    vocabulary_size: usize,
    explicit: Option<&GenerationConfig>,
) -> Result<GenerationConfig> {
    if let Some(explicit) = explicit {
        explicit.validate(vocabulary_size)?;
        return Ok(explicit.clone());
    }

    let mut config = GenerationConfig {
        eos_token_ids: normalize_token_list(model_eos, vocabulary_size, "eos_token_id")?,
        pad_token_id: normalize_pad_token(model_pad, vocabulary_size)?,
        ..GenerationConfig::default()
    };
    if let Some(local) = read_generation_config(root)? {
        if let Some(max_new_tokens) = local.max_new_tokens {
            config.max_new_tokens = max_new_tokens;
        }
        if local.do_sample.unwrap_or(false) {
            config.strategy = SamplingStrategy::Sample;
            if let Some(temperature) = local.temperature {
                config.temperature = temperature;
            }
            config.top_k = local.top_k.filter(|top_k| *top_k != 0);
            config.top_p = local.top_p;
        }
        if let Some(penalty) = local.repetition_penalty {
            config.repetition_penalty = penalty;
        }
        if let Some(penalty) = local.presence_penalty {
            config.presence_penalty = penalty;
        }
        if let Some(seed) = local.seed {
            config.seed = seed;
        }
        if local.eos_token_id.is_some() {
            config.eos_token_ids = normalize_token_list(
                local.eos_token_id.as_ref(),
                vocabulary_size,
                "generation_config.eos_token_id",
            )?;
        }
        if local.pad_token_id.is_some() {
            config.pad_token_id = normalize_pad_token_named(
                local.pad_token_id.as_ref(),
                vocabulary_size,
                "generation_config.pad_token_id",
            )?;
        }
    }
    config.validate(vocabulary_size)?;
    Ok(config)
}

/// Supported, architecture-independent subset of `generation_config.json`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct LocalGenerationConfig {
    max_new_tokens: Option<usize>,
    do_sample: Option<bool>,
    temperature: Option<f64>,
    top_k: Option<usize>,
    top_p: Option<f64>,
    repetition_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    seed: Option<u64>,
    eos_token_id: Option<TokenIds>,
    pad_token_id: Option<TokenIds>,
}

/// Reads optional bounded generation metadata without accepting escaping symlinks.
fn read_generation_config(root: &Path) -> Result<Option<LocalGenerationConfig>> {
    let Some(path) = contained_regular_file(
        root,
        "generation_config.json",
        MAX_GENERATION_CONFIG_BYTES,
        false,
    )?
    else {
        return Ok(None);
    };
    let bytes = fs::read(&path).map_err(|source| Error::io(&path, source))?;
    if bytes.len() as u64 > MAX_GENERATION_CONFIG_BYTES {
        return Err(Error::Artifact {
            path,
            reason: format!(
                "generation config exceeds {MAX_GENERATION_CONFIG_BYTES} bytes while being read"
            ),
        });
    }
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|source| Error::json(path, source))
}

/// Resolves one fixed child path and enforces containment, type, and byte limits.
fn contained_regular_file(
    root: &Path,
    name: &str,
    max_bytes: u64,
    required: bool,
) -> Result<Option<PathBuf>> {
    let canonical_root = root
        .canonicalize()
        .map_err(|source| Error::io(root, source))?;
    let supplied = canonical_root.join(name);
    let path = match supplied.canonicalize() {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !required => return Ok(None),
        Err(error) => return Err(Error::io(supplied, error)),
    };
    if !path.starts_with(&canonical_root) {
        return Err(Error::Artifact {
            path,
            reason: format!("{name} resolves outside the model source root"),
        });
    }
    let metadata = fs::metadata(&path).map_err(|source| Error::io(&path, source))?;
    if !metadata.is_file() {
        return Err(Error::Artifact {
            path,
            reason: format!("{name} must be a regular file"),
        });
    }
    if metadata.len() > max_bytes {
        return Err(Error::Artifact {
            path,
            reason: format!("{name} exceeds the {max_bytes}-byte limit"),
        });
    }
    Ok(Some(path))
}

/// Converts and canonicalizes a scalar-or-vector token field.
fn normalize_token_list(
    ids: Option<&TokenIds>,
    vocabulary_size: usize,
    field: &str,
) -> Result<Vec<u32>> {
    let mut normalized = Vec::new();
    if let Some(ids) = ids {
        for id in ids.iter() {
            if id >= vocabulary_size {
                return Err(Error::configuration(
                    field,
                    format!("token {id} is outside vocabulary size {vocabulary_size}"),
                ));
            }
            normalized.push(u32::try_from(id).map_err(|_| {
                Error::configuration(field, format!("token {id} does not fit the u32 runtime"))
            })?);
        }
    }
    normalized.sort_unstable();
    normalized.dedup();
    Ok(normalized)
}

/// Normalizes the model-level padding token.
fn normalize_pad_token(ids: Option<&TokenIds>, vocabulary_size: usize) -> Result<Option<u32>> {
    normalize_pad_token_named(ids, vocabulary_size, "pad_token_id")
}

/// Requires a padding field to identify at most one token.
fn normalize_pad_token_named(
    ids: Option<&TokenIds>,
    vocabulary_size: usize,
    field: &str,
) -> Result<Option<u32>> {
    let normalized = normalize_token_list(ids, vocabulary_size, field)?;
    match normalized.as_slice() {
        [] => Ok(None),
        [token] => Ok(Some(*token)),
        _ => Err(Error::configuration(
            field,
            "padding requires exactly one token ID",
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Write, path::Path};

    use burn::{
        backend::Flex,
        tensor::{Int, Tensor},
    };
    use serde_json::{Map, Value, json};

    use super::*;
    use crate::{
        CausalLmInput, ForwardOptions, Seq2SeqInput,
        checkpoint::{
            ExecutionStats, ExpectedTensor, LoadReport, MappingPlan, SourceStats, TensorOperation,
            TensorTransform,
            quantization::{PackedLoadStats, QuantizedExecutionOutcome},
        },
        models::{
            deepseek_v2::model::tests_support::tiny_config as tiny_deepseek_config,
            muse_glimmer::MuseGlimmerLayerType, nemotron_h::NemotronHBlockType,
            t5::T5FeedForwardProjection,
        },
    };

    type TestBackend = Flex<f32, i32>;

    /// Writes one complete zero-valued F32 checkpoint from a family contract.
    ///
    /// The contract remains the source of truth for both external names and
    /// shapes. Alias operations have no independent checkpoint payload; the
    /// automatic-loader fixtures use untied embeddings so every logical target
    /// is represented by one physical `Map` operation.
    fn write_contract_checkpoint(root: &Path, expected: &[ExpectedTensor], mappings: &MappingPlan) {
        let mut tensors = Vec::new();
        for operation in mappings.operations() {
            match operation {
                TensorOperation::Map {
                    source,
                    target,
                    transform,
                } => {
                    let target_shape = expected
                        .iter()
                        .find(|tensor| tensor.name == *target)
                        .unwrap_or_else(|| panic!("mapping target `{target}` must be expected"))
                        .shape
                        .clone();
                    let source_shape = match transform {
                        TensorTransform::Identity => target_shape,
                        TensorTransform::Transpose2d => {
                            assert_eq!(target_shape.len(), 2, "transpose fixture must be rank two");
                            vec![target_shape[1], target_shape[0]]
                        }
                    };
                    tensors.push((source.clone(), source_shape));
                }
                TensorOperation::Alias { .. } => {}
                TensorOperation::Split { .. } | TensorOperation::Concatenate { .. } => {
                    panic!("automatic family fixture supports map and alias contracts only")
                }
            }
        }

        let mut offset = 0_u64;
        let mut header = Map::new();
        for (name, shape) in tensors {
            let elements = shape.iter().copied().product::<u64>();
            let end = offset
                .checked_add(elements.checked_mul(4).expect("fixture byte size"))
                .expect("fixture data offset");
            assert!(
                header
                    .insert(
                        name.clone(),
                        json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}),
                    )
                    .is_none(),
                "fixture source `{name}` must be unique"
            );
            offset = end;
        }
        let mut header = serde_json::to_vec(&Value::Object(header)).expect("fixture header");
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = File::create(root.join("model.safetensors")).expect("fixture checkpoint");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("fixture header length");
        file.write_all(&header).expect("fixture header bytes");
        file.write_all(&vec![0; offset as usize])
            .expect("fixture tensor bytes");
    }

    /// Persists one direct architecture config plus deterministic generation metadata.
    fn write_auto_metadata<T: serde::Serialize>(root: &Path, config: &T) {
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(config).expect("serializable family config"),
        )
        .expect("config fixture");
        std::fs::write(
            root.join("generation_config.json"),
            br#"{"max_new_tokens":5,"do_sample":false}"#,
        )
        .expect("generation fixture");
    }

    /// Returns provenance options that also avoid requiring a tokenizer fixture.
    fn fixture_options(repository: &str, revision: &str) -> AutoLoadOptions {
        AutoLoadOptions {
            load_tokenizer: false,
            repository: Some(repository.into()),
            revision: Some(revision.into()),
            ..AutoLoadOptions::default()
        }
    }

    /// Verifies exhaustive strict accounting for one untied identity/rename contract.
    fn assert_strict_report(
        report: &LoadReport,
        expected: &[ExpectedTensor],
        mappings: &MappingPlan,
    ) {
        let mut expected_names = expected
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        expected_names.sort();
        assert_eq!(report.applied, expected_names);
        assert!(report.is_strictly_valid());
        assert!(report.missing.is_empty());
        assert!(report.unexpected.is_empty());
        assert!(report.allowlisted_missing.is_empty());
        assert!(report.allowlisted_ignored.is_empty());
        assert!(report.aliased.is_empty());
        assert!(report.dtype_converted.is_empty());
        assert!(report.quantized.is_empty());

        let rename_count = mappings
            .operations()
            .iter()
            .filter(|operation| {
                matches!(
                    operation,
                    TensorOperation::Map { source, target, .. } if source != target
                )
            })
            .count();
        assert_eq!(report.transformed.len(), rename_count);
        assert!(
            report
                .transformed
                .iter()
                .all(|record| record.operation == "rename")
        );
        let memory = report.memory.expect("strict plan must estimate memory");
        assert!(memory.checkpoint_tensor_bytes > 0);
        assert!(memory.target_parameter_bytes > 0);
    }

    /// Verifies source identity and normalized host-side generation defaults.
    fn assert_bundle_metadata(
        loaded: &LoadedModel<AutoCausalLm<TestBackend>>,
        root: &Path,
        model_type: &str,
        repository: &str,
        revision: &str,
        eos_token_ids: &[u32],
        pad_token_id: u32,
    ) {
        assert_eq!(
            loaded.source().root(),
            root.canonicalize().expect("canonical fixture root")
        );
        assert_eq!(loaded.source().model_type(), model_type);
        assert_eq!(loaded.source().repository(), Some(repository));
        assert_eq!(loaded.source().revision(), Some(revision));
        assert_eq!(loaded.generation_defaults().max_new_tokens, 5);
        assert_eq!(
            loaded.generation_defaults().strategy,
            SamplingStrategy::Greedy
        );
        assert_eq!(loaded.generation_defaults().eos_token_ids, eos_token_ids);
        assert_eq!(
            loaded.generation_defaults().pad_token_id,
            Some(pad_token_id)
        );
        assert!(loaded.tokenizer().is_none());
        assert_execution_audit(loaded);
    }

    /// Requires internally consistent, bounded local checkpoint execution counters.
    fn assert_execution_audit<M>(loaded: &LoadedModel<M>) {
        let stats = loaded
            .execution_stats()
            .expect("automatic loads must retain execution counters");
        assert_eq!(
            stats.tensors_applied + stats.aliases_applied,
            loaded.load_report().applied.len()
        );
        assert_eq!(stats.source_reads, stats.source.tensor_reads);
        assert_eq!(stats.source_payload_bytes, stats.source.payload_bytes_read);
        assert!(stats.source_reads > 0);
        assert!(stats.source_payload_bytes > 0);
        assert_eq!(stats.source.shard_opens, 1);
        assert_eq!(stats.source.max_open_shards, 1);
    }

    /// Runs one automatic causal variant and requires finite vocabulary logits.
    fn assert_finite_forward(
        loaded: &LoadedModel<AutoCausalLm<TestBackend>>,
        vocabulary_size: usize,
    ) {
        let device = Default::default();
        let output = loaded
            .model()
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .expect("loaded automatic model must execute");
        let [batch, selected, vocabulary] = output.logits.dims();
        assert_eq!([batch, selected, vocabulary], [1, 1, vocabulary_size]);
        assert!(
            output
                .logits
                .into_data()
                .as_slice::<f32>()
                .expect("F32 reference backend")
                .iter()
                .all(|value| value.is_finite())
        );
    }

    /// Automatic loading is base-only unless MTP consumption is explicit.
    #[test]
    fn automatic_options_keep_mtp_opt_in() {
        assert!(!AutoLoadOptions::default().qwen_mtp);
    }

    /// Packed automatic loading keeps the generic checkpoint counters verbatim.
    #[test]
    fn packed_outcome_retains_exact_checkpoint_execution_stats() {
        let stats = ExecutionStats {
            tensors_applied: 2,
            source_reads: 2,
            source_payload_bytes: 96,
            delivered_tensor_bytes: 96,
            largest_source_payload_bytes: 64,
            largest_output_tensor_bytes: 64,
            peak_host_staging_bytes: 128,
            source: SourceStats {
                shard_opens: 1,
                tensor_reads: 2,
                payload_bytes_read: 96,
                max_open_shards: 1,
                shard_open_order: vec!["model.safetensors".into()],
            },
            ..ExecutionStats::default()
        };
        let mut report = LoadReport::new();
        report.applied = vec!["model.weight".into(), "lm_head.weight".into()];
        let converted = checkpoint_execution_from_quantized(QuantizedExecutionOutcome {
            report: report.clone(),
            execution: stats.clone(),
            packed: PackedLoadStats {
                tensor_reads: 1,
                payload_bytes: 48,
                largest_payload_bytes: 48,
                tensors_applied: 1,
            },
        });

        assert_eq!(converted.report, report);
        assert_eq!(converted.stats, stats);
    }

    /// Automatic packed loading rejects families without a completed backend gate.
    #[test]
    fn automatic_packed_loading_is_limited_to_certified_qwen_hybrid_models() {
        let root = tempfile::tempdir().expect("temporary model root");
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&Qwen3MoeConfig::default()).expect("serializable config"),
        )
        .expect("config fixture");
        let error = AutoCausalLm::<TestBackend>::load_local(
            root.path(),
            &AutoLoadOptions {
                load_tokenizer: false,
                quantized_burnpack: Some(PathBuf::from("weights.bpk")),
                ..AutoLoadOptions::default()
            },
            &Default::default(),
        )
        .expect_err("uncertified family must fail before checkpoint discovery");
        assert!(matches!(error, Error::BackendCapability(_)));
    }

    /// Model tokens provide defaults and local sampling metadata overrides them.
    #[test]
    fn generation_defaults_are_normalized_from_local_metadata() {
        let root = tempfile::tempdir().expect("temporary model root");
        std::fs::write(
            root.path().join("generation_config.json"),
            br#"{
                "max_new_tokens": 17,
                "do_sample": true,
                "temperature": 0.7,
                "top_k": 0,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "presence_penalty": 0.2,
                "seed": 42,
                "eos_token_id": [9, 8, 9],
                "pad_token_id": 7,
                "unknown_future_field": true
            }"#,
        )
        .expect("generation fixture");
        let config = normalize_generation_defaults(
            root.path(),
            Some(&TokenIds::Many(vec![3, 2, 3])),
            Some(&TokenIds::One(1)),
            16,
            None,
        )
        .expect("valid normalized defaults");
        assert_eq!(config.max_new_tokens, 17);
        assert_eq!(config.strategy, SamplingStrategy::Sample);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, None);
        assert_eq!(config.top_p, Some(0.9));
        assert_eq!(config.repetition_penalty, 1.1);
        assert_eq!(config.presence_penalty, 0.2);
        assert_eq!(config.seed, 42);
        assert_eq!(config.eos_token_ids, [8, 9]);
        assert_eq!(config.pad_token_id, Some(7));
    }

    /// Greedy local defaults discard inactive sampling warpers and explicit config wins.
    #[test]
    fn explicit_defaults_take_precedence_and_greedy_ignores_warpers() {
        let root = tempfile::tempdir().expect("temporary model root");
        std::fs::write(
            root.path().join("generation_config.json"),
            br#"{"do_sample":false,"temperature":0.0,"top_k":50,"top_p":0.8}"#,
        )
        .expect("generation fixture");
        let greedy = normalize_generation_defaults(root.path(), None, None, 64, None)
            .expect("inactive invalid temperature is normalized away");
        assert_eq!(greedy.strategy, SamplingStrategy::Greedy);
        assert_eq!(greedy.temperature, 1.0);
        assert_eq!(greedy.top_k, None);
        assert_eq!(greedy.top_p, None);

        let explicit = GenerationConfig {
            max_new_tokens: 3,
            eos_token_ids: vec![5],
            ..GenerationConfig::default()
        };
        let normalized = normalize_generation_defaults(
            root.path(),
            Some(&TokenIds::One(7)),
            None,
            64,
            Some(&explicit),
        )
        .expect("explicit defaults");
        assert_eq!(normalized, explicit);
    }

    /// Invalid local metadata, token bounds, and ambiguous padding are rejected.
    #[test]
    fn generation_defaults_reject_invalid_artifacts_and_tokens() {
        let root = tempfile::tempdir().expect("temporary model root");
        std::fs::write(root.path().join("generation_config.json"), b"{")
            .expect("invalid generation fixture");
        assert!(normalize_generation_defaults(root.path(), None, None, 8, None).is_err());
        std::fs::remove_file(root.path().join("generation_config.json"))
            .expect("remove invalid fixture");
        assert!(
            normalize_generation_defaults(root.path(), Some(&TokenIds::One(8)), None, 8, None)
                .is_err()
        );
        assert!(
            normalize_generation_defaults(
                root.path(),
                None,
                Some(&TokenIds::Many(vec![1, 2])),
                8,
                None
            )
            .is_err()
        );
    }

    /// Qwen namespace detection accepts exactly one official layout.
    #[test]
    fn qwen_namespace_detection_is_unambiguous() {
        assert_eq!(
            classify_qwen_names(["model.language_model.embed_tokens.weight"])
                .expect("wrapped namespace"),
            Qwen3_5CheckpointSource::Multimodal
        );
        assert_eq!(
            classify_qwen_names(["model.embed_tokens.weight"]).expect("text namespace"),
            Qwen3_5CheckpointSource::TextOnly
        );
        assert!(
            classify_qwen_names([
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight"
            ])
            .is_err()
        );
        assert!(classify_qwen_names(["foreign.weight"]).is_err());
    }

    /// Muse namespace detection accepts exactly one language-tower layout.
    #[test]
    fn muse_namespace_detection_is_unambiguous() {
        assert_eq!(
            classify_muse_names(["model.language_model.embed_tokens.weight"])
                .expect("wrapped namespace"),
            MuseGlimmerCheckpointSource::Multimodal
        );
        assert_eq!(
            classify_muse_names(["model.embed_tokens.weight"]).expect("text namespace"),
            MuseGlimmerCheckpointSource::TextOnly
        );
        assert!(
            classify_muse_names([
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight"
            ])
            .is_err()
        );
        assert!(classify_muse_names(["foreign.weight"]).is_err());
    }

    /// Required config reading rejects missing files and non-directory roots.
    #[test]
    fn required_config_is_bounded_and_regular() {
        let root = tempfile::tempdir().expect("temporary model root");
        assert!(read_required_config(root.path()).is_err());
        let file = root.path().join("not-a-directory");
        std::fs::write(&file, b"{}").expect("fixture file");
        assert!(read_required_config(&file).is_err());
        std::fs::create_dir(root.path().join("config.json")).expect("fixture directory");
        assert!(read_required_config(root.path()).is_err());
    }

    /// The DeepSeek branch performs strict discovery, installation, and execution.
    #[test]
    fn automatic_deepseek_loads_complete_strict_fixture() {
        let root = tempfile::tempdir().expect("temporary model root");
        let mut config = tiny_deepseek_config();
        config.eos_token_id = Some(TokenIds::Many(vec![3, 2, 3]));
        config.pad_token_id = Some(TokenIds::One(0));
        let contract = config
            .checkpoint_contract(DeepseekV2CheckpointOptions::default())
            .expect("valid DeepSeek contract");
        write_auto_metadata(root.path(), &config);
        write_contract_checkpoint(
            root.path(),
            contract.expected_tensors(),
            contract.mappings(),
        );

        let repository = "deepseek-ai/tiny-deepseek-v2-fixture";
        let revision = "deepseek-fixture-revision";
        let loaded = AutoCausalLm::<TestBackend>::load_local(
            root.path(),
            &fixture_options(repository, revision),
            &Default::default(),
        )
        .expect("automatic DeepSeek load");

        assert!(matches!(loaded.model(), AutoCausalLm::DeepseekV2(_)));
        assert_strict_report(
            loaded.load_report(),
            contract.expected_tensors(),
            contract.mappings(),
        );
        assert_bundle_metadata(
            &loaded,
            root.path(),
            "deepseek_v2",
            repository,
            revision,
            &[2, 3],
            0,
        );
        assert_finite_forward(&loaded, config.vocab_size);
    }

    /// The Muse text-only branch classifies its namespace and streams every tensor.
    #[test]
    fn automatic_muse_glimmer_loads_complete_strict_fixture() {
        let root = tempfile::tempdir().expect("temporary model root");
        let config = MuseGlimmerTextConfig {
            vocab_size: 16,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 16,
            sliding_window: Some(4),
            layer_types: vec![
                MuseGlimmerLayerType::SlidingAttention,
                MuseGlimmerLayerType::FullAttention,
            ],
            layer_rope_theta: vec![10_000.0, 0.0],
            bos_token_id: Some(TokenIds::One(1)),
            eos_token_id: Some(TokenIds::Many(vec![5, 4, 5])),
            pad_token_id: Some(TokenIds::One(0)),
            tie_word_embeddings: false,
            ..MuseGlimmerTextConfig::default()
        };
        let family_options = MuseGlimmerCheckpointOptions {
            source: MuseGlimmerCheckpointSource::TextOnly,
        };
        let contract = config
            .checkpoint_contract(family_options)
            .expect("valid Muse contract");
        write_auto_metadata(root.path(), &config);
        write_contract_checkpoint(
            root.path(),
            contract.expected_tensors(),
            contract.mappings(),
        );

        let repository = "meta-models/tiny-muse-glimmer-fixture";
        let revision = "muse-fixture-revision";
        let loaded = AutoCausalLm::<TestBackend>::load_local(
            root.path(),
            &fixture_options(repository, revision),
            &Default::default(),
        )
        .expect("automatic Muse load");

        assert!(matches!(loaded.model(), AutoCausalLm::MuseGlimmer(_)));
        assert_strict_report(
            loaded.load_report(),
            contract.expected_tensors(),
            contract.mappings(),
        );
        assert_bundle_metadata(
            &loaded,
            root.path(),
            "muse_glimmer_text",
            repository,
            revision,
            &[4, 5],
            0,
        );
        assert_finite_forward(&loaded, config.vocab_size);
    }

    /// The Nemotron branch loads a complete heterogeneous hybrid schedule.
    #[test]
    fn automatic_nemotron_h_loads_complete_strict_fixture() {
        let root = tempfile::tempdir().expect("temporary model root");
        let config = NemotronHConfig {
            vocab_size: 16,
            hidden_size: 8,
            layers_block_type: vec![
                NemotronHBlockType::LinearAttention,
                NemotronHBlockType::FullAttention,
                NemotronHBlockType::Mlp,
                NemotronHBlockType::Moe,
            ],
            num_hidden_layers: Some(4),
            tie_word_embeddings: false,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 16,
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
            bos_token_id: Some(TokenIds::One(1)),
            eos_token_id: Some(TokenIds::Many(vec![7, 6, 7])),
            pad_token_id: Some(TokenIds::One(0)),
            ..NemotronHConfig::default()
        };
        let contract = config
            .checkpoint_contract(NemotronHCheckpointOptions::default())
            .expect("valid Nemotron-H contract");
        write_auto_metadata(root.path(), &config);
        write_contract_checkpoint(
            root.path(),
            contract.expected_tensors(),
            contract.mappings(),
        );

        let repository = "nvidia/tiny-nemotron-h-fixture";
        let revision = "nemotron-fixture-revision";
        let loaded = AutoCausalLm::<TestBackend>::load_local(
            root.path(),
            &fixture_options(repository, revision),
            &Default::default(),
        )
        .expect("automatic Nemotron-H load");

        assert!(matches!(loaded.model(), AutoCausalLm::NemotronH(_)));
        assert_strict_report(
            loaded.load_report(),
            contract.expected_tensors(),
            contract.mappings(),
        );
        assert_bundle_metadata(
            &loaded,
            root.path(),
            "nemotron_h",
            repository,
            revision,
            &[6, 7],
            0,
        );
        assert_finite_forward(&loaded, config.vocab_size);
    }

    /// The encoder-decoder registry retains the same pass-two audit as causal loads.
    #[test]
    fn automatic_t5_load_retains_execution_stats() {
        let root = tempfile::tempdir().expect("temporary model root");
        let config = T5Config {
            vocab_size: 13,
            d_model: 4,
            d_kv: 2,
            d_ff: 6,
            num_layers: 2,
            num_decoder_layers: Some(1),
            num_heads: 2,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 16,
            feed_forward_proj: T5FeedForwardProjection::GatedGelu,
            dense_act_fn: None,
            is_gated_act: None,
            tie_word_embeddings: false,
            ..T5Config::default()
        };
        let contract = config
            .checkpoint_contract()
            .expect("valid tiny T5 contract");
        write_auto_metadata(root.path(), &config);
        write_contract_checkpoint(
            root.path(),
            contract.expected_tensors(),
            contract.mappings(),
        );

        let loaded = AutoSeq2SeqLm::<TestBackend>::load_local(
            root.path(),
            &fixture_options("google/tiny-flan-fixture", "flan-fixture-revision"),
            &Default::default(),
        )
        .expect("automatic T5 load");

        assert!(matches!(loaded.model(), AutoSeq2SeqLm::T5(_)));
        assert_strict_report(
            loaded.load_report(),
            contract.expected_tensors(),
            contract.mappings(),
        );
        assert_execution_audit(&loaded);
        assert_eq!(loaded.source().model_type(), "t5");
        assert_eq!(loaded.generation_defaults().eos_token_ids, [1]);
        assert_eq!(loaded.generation_defaults().pad_token_id, Some(0));
    }

    /// Loads the pinned real FLAN checkpoint through the automatic public API.
    #[test]
    #[ignore = "requires pinned google/flan-t5-small SafeTensors artifacts"]
    fn automatic_flan_load_bundles_and_executes_real_weights() {
        let root = std::env::var_os("TRANSFORMERS_FLAN_T5_SMALL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("models/google/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab")
            });
        let options = AutoLoadOptions {
            repository: Some("google/flan-t5-small".into()),
            revision: Some("0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab".into()),
            ..AutoLoadOptions::default()
        };
        let device = Default::default();
        let loaded = AutoSeq2SeqLm::<TestBackend>::load_local(&root, &options, &device)
            .expect("automatic FLAN load must be strict");
        assert_eq!(loaded.source().repository(), Some("google/flan-t5-small"));
        assert_eq!(
            loaded.source().revision(),
            Some("0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab")
        );
        assert_eq!(loaded.load_report().applied.len(), 190);
        assert_eq!(loaded.generation_defaults().eos_token_ids, [1]);
        assert_eq!(loaded.generation_defaults().pad_token_id, Some(0));
        #[cfg(feature = "tokenizers")]
        assert!(loaded.tokenizer().is_some());
        #[cfg(not(feature = "tokenizers"))]
        assert!(loaded.tokenizer().is_none());

        let output = loaded
            .model()
            .forward(
                Seq2SeqInput {
                    encoder: CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[71, 8, 57]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    decoder: CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                },
                &ForwardOptions::default(),
            )
            .expect("automatic FLAN model must execute");
        assert_eq!(output.decoder.logits.dims(), [1, 1, 32_128]);
        assert!(
            output
                .decoder
                .logits
                .to_data()
                .as_slice::<f32>()
                .expect("F32 reference backend")
                .iter()
                .all(|value| value.is_finite())
        );
    }
}
