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
    checkpoint::{CheckpointManifest, LoadOptions, execute_local_load_plan},
};

use super::{AutoCausalLm, AutoSeq2SeqLm, detect_model_type};
use crate::models::{
    config::{ModelConfig, TokenIds},
    qwen3_5::{
        Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5ForCausalLm, Qwen3_5TextConfig,
    },
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
        if model_type != "qwen3_5_text" {
            return Err(Error::UnsupportedArchitecture(format!(
                "causal model_type `{model_type}` is not executable; supported values: {}",
                Self::MODEL_TYPES.join(", ")
            )));
        }

        let config = Qwen3_5TextConfig::from_json_str(&json)
            .map_err(|error| Error::configuration("auto_causal.config", error.to_string()))?;
        let vocabulary_size = config.vocab_size;
        let generation = normalize_generation_defaults(
            &root,
            config.eos_token_id.as_ref(),
            config.pad_token_id.as_ref(),
            config.vocab_size,
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
        let outcome = execute_local_load_plan(&manifest, &plan, &mut model)
            .map_err(|error| Error::Checkpoint(error.to_string()))?;
        LoadedModel::new(
            Self::Qwen3_5(model),
            tokenizer,
            generation,
            source,
            outcome.report,
            vocabulary_size,
        )
    }
}

impl<B: Backend> AutoSeq2SeqLm<B> {
    /// Strictly loads one supported encoder-decoder model from a local directory.
    pub fn load_local(
        root: impl AsRef<Path>,
        options: &AutoLoadOptions,
        device: &B::Device,
    ) -> Result<LoadedModel<Self>> {
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
        LoadedModel::new(
            Self::T5(model),
            tokenizer,
            generation,
            source,
            outcome.report,
            vocabulary_size,
        )
    }
}

/// Distinguishes official wrapped and text-only Qwen parameter namespaces.
fn qwen_source_layout(manifest: &CheckpointManifest) -> Result<Qwen3_5CheckpointSource> {
    classify_qwen_names(manifest.tensors.keys().map(String::as_str))
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
    let path = contained_regular_file(&root, "config.json", MAX_CONFIG_BYTES, true)?
        .expect("required contained file returns Some");
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
    use burn::{
        backend::Flex,
        tensor::{Int, Tensor},
    };

    use super::*;
    use crate::{CausalLmInput, ForwardOptions, Seq2SeqInput};

    type TestBackend = Flex<f32, i32>;

    /// Automatic loading is base-only unless MTP consumption is explicit.
    #[test]
    fn automatic_options_keep_mtp_opt_in() {
        assert!(!AutoLoadOptions::default().qwen_mtp);
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
