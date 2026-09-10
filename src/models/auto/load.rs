//! Local-only automatic checkpoint loading and generation-default normalization.
//!
//! The automatic registry performs family selection once, then hands strict
//! tensor planning and installation back to the architecture-owned checkpoint
//! contract. No filename heuristics select individual parameters. Qwen's two
//! published namespace layouts are distinguished only after complete SafeTensors
//! header discovery by checking which exact language prefix is present.
//!
//! Loading is a host-side boundary: canonical contained paths and bounded JSON are
//! validated before tensor installation. Explicit generation defaults take precedence
//! over local generation metadata, which overrides model token IDs. EOS IDs are
//! sorted and deduplicated; padding identifies at most one in-vocabulary token.
//! Floating casts, packed layouts, aliases, and tensor shapes remain the responsibility
//! of the selected architecture contract and shared checkpoint planner.

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
            AuditedQuantizedExecutionOutcome, QuantizedBurnpackArtifact,
            execute_local_quantized_load_plan_audited,
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
    nemotron_h::{
        NemotronHCheckpointOptions, NemotronHCheckpointProvenance, NemotronHConfig,
        NemotronHForCausalLm,
    },
    qwen3_5::{
        Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5ForCausalLm, Qwen3_5TextConfig,
    },
    qwen3_moe::{Qwen3MoeCheckpointOptions, Qwen3MoeConfig, Qwen3MoeForCausalLm},
    t5::{T5Config, T5ForConditionalGeneration},
};
use burn::tensor::backend::Backend;

const MAX_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_GENERATION_CONFIG_BYTES: u64 = 1024 * 1024;

/// Causal families whose packed installation path has an acceptance contract.
const PACKED_CAUSAL_MODEL_TYPES: &[&str] = &["muse_glimmer_text", "qwen3_5_text", "qwen3_moe"];

// User policy and family-specific strict loading.

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
    /// Selects strict local loading, optional tokenizer discovery, and no packed overlay or MTP graph.
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
        if options.quantized_burnpack.is_some()
            && !PACKED_CAUSAL_MODEL_TYPES.contains(&model_type.as_str())
        {
            return Err(Error::BackendCapability(format!(
                "packed Burnpack loading for `{model_type}` is not yet certified; automatic packed execution is limited to {}",
                PACKED_CAUSAL_MODEL_TYPES.join(", ")
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
                outcome.into_loaded(
                    Self::DeepseekV2(model),
                    tokenizer,
                    generation,
                    source,
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
                outcome.into_loaded(
                    Self::MuseGlimmer(model),
                    tokenizer,
                    generation,
                    source,
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
                let architecture_options = match (&options.repository, &options.revision) {
                    (Some(repository), Some(revision)) => NemotronHCheckpointOptions::default()
                        .with_provenance(NemotronHCheckpointProvenance::new(repository, revision)),
                    _ => NemotronHCheckpointOptions::default(),
                };
                let contract =
                    config
                        .checkpoint_contract(architecture_options)
                        .map_err(|error| {
                            Error::configuration(
                                format!("{}.{}", error.architecture, error.field),
                                error.reason,
                            )
                        })?;
                let manifest = contract
                    .discover_checkpoint(&root, &options.checkpoint)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?;
                let plan = contract
                    .build_load_plan(&manifest, &options.checkpoint)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?;
                let mut model = NemotronHForCausalLm::init(config, device)?;
                let outcome = execute_causal_plan(&root, &manifest, &plan, options, &mut model)?;
                outcome.into_loaded(
                    Self::NemotronH(model),
                    tokenizer,
                    generation,
                    source,
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
                outcome.into_loaded(
                    Self::Qwen3_5(model),
                    tokenizer,
                    generation,
                    source,
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
                outcome.into_loaded(
                    Self::Qwen3Moe(model),
                    tokenizer,
                    generation,
                    source,
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

/// Successful causal checkpoint execution, preserving sidecar-only counters.
enum CausalExecutionOutcome {
    /// Ordinary SafeTensors execution without a packed overlay.
    FullPrecision(ExecutionOutcome),
    /// Mixed source-precision and Burnpack execution with provenance counters.
    Quantized(AuditedQuantizedExecutionOutcome),
}

impl CausalExecutionOutcome {
    /// Attaches every available execution counter to one loaded model bundle.
    fn into_loaded<M>(
        self,
        model: M,
        tokenizer: Option<ModelTokenizer>,
        generation_defaults: GenerationConfig,
        source: ModelSourceMetadata,
        vocabulary_size: usize,
    ) -> Result<LoadedModel<M>> {
        match self {
            Self::FullPrecision(outcome) => LoadedModel::new_with_execution_stats(
                model,
                tokenizer,
                generation_defaults,
                source,
                outcome.report,
                outcome.stats,
                vocabulary_size,
            ),
            Self::Quantized(outcome) => LoadedModel::new_with_quantized_execution(
                model,
                tokenizer,
                generation_defaults,
                source,
                outcome,
                vocabulary_size,
            ),
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
) -> Result<CausalExecutionOutcome>
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
        execute_local_quantized_load_plan_audited(manifest, plan, &mut artifact, model)
            .map_err(|error| Error::Checkpoint(error.to_string()))
            .map(CausalExecutionOutcome::Quantized)
    } else {
        execute_local_load_plan(manifest, plan, model)
            .map_err(|error| Error::Checkpoint(error.to_string()))
            .map(CausalExecutionOutcome::FullPrecision)
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
            return Err(Error::BackendCapability(format!(
                "packed Burnpack loading is currently certified only for causal model types: {}",
                PACKED_CAUSAL_MODEL_TYPES.join(", ")
            )));
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

// Contained artifact discovery and generation-default normalization.

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
#[path = "load_tests.rs"]
mod tests;
