//! Local-only automatic checkpoint loading and generation-default normalization.
//!
//! The automatic registry performs family selection once, then runs one shared
//! pipeline ([`load_causal_family`]) that hands strict tensor planning and
//! installation back to the architecture-owned checkpoint contract through
//! [`CausalFamily`]. No filename heuristics select individual parameters.
//! Qwen's and Muse's two published namespace layouts are distinguished only
//! after complete SafeTensors header discovery by checking which exact
//! language prefix is present.
//!
//! Loading is a host-side boundary: canonical contained paths and bounded JSON are
//! validated before tensor installation. Explicit generation defaults take precedence
//! over local generation metadata, which overrides model token IDs. EOS IDs are
//! sorted and deduplicated; padding identifies at most one in-vocabulary token.
//!
//! `generation_config.json` follows the pinned Hugging Face `GenerationConfig`
//! semantics (transformers `cfa309cf`): with `do_sample`, an unset `top_k`
//! means Hugging Face's default of 50 (clamped to the vocabulary, as its top-k
//! warper does), `top_k: 0` disables top-k, `top_p >= 1` disables nucleus
//! filtering, and `min_p` adds the min-p warper (Hugging Face applies every
//! warper only when sampling). `no_repeat_ngram_size` and `min_new_tokens`
//! apply under every strategy, with zero meaning disabled. Hugging Face only
//! warns when `min_new_tokens` exceeds the length bound and then suppresses
//! EOS for the whole run, so such a value is clamped to `max_new_tokens`
//! (the same tokens) and its declared value is reported, because raising
//! `max_new_tokens` later would not restore it.
//!
//! Behavior-changing settings without an equivalent here (beam search,
//! `typical_p`, `max_length`, ...) are recorded on the [`LoadedModel`] rather
//! than silently dropped. What counts as applied depends on the
//! [`GenerationConsumer`]: causal generation applies every
//! [`GenerationConfig`] logits processor and warper, whereas encoder-decoder
//! beam search reads only its own `BeamSearchConfig`. Seq2seq loads therefore
//! leave every processor Hugging Face beam search would apply neutral on the
//! defaults and report it instead: the `min_p`, `no_repeat_ngram_size`, and
//! `min_new_tokens` constraints, the repetition penalty, and sampling
//! (`do_sample`, which selects Hugging Face's beam *sampling*, with its
//! temperature, top-k, and top-p warpers). `presence_penalty`, which Hugging
//! Face's `GenerationConfig` does not define, is likewise left neutral and,
//! matching Hugging Face, not reported.
//! Floating casts, packed layouts, aliases, and tensor shapes remain the responsibility
//! of the selected architecture contract and shared checkpoint planner.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use serde_json::Value;

use crate::{
    Error, GenerationConfig, LoadedModel, ModelSourceMetadata, ModelTokenizer, Result,
    SamplingStrategy,
    checkpoint::{
        CheckpointManifest, ExecutionOutcome, LoadOptions, LoadPlan, TensorSink,
        execute_local_load_plan,
        quantization::{
            AuditedQuantizedExecutionOutcome, QuantizedBurnpackArtifact,
            execute_local_quantized_load_plan_audited,
        },
    },
};

use super::{
    AutoCausalConfig, AutoCausalLm, AutoSeq2SeqLm,
    defaults::{TokenIdError, canonical_token_ids, clamped_min_new_tokens, sampling_top_k},
    detect_model_type,
    family::{CausalFamily, PlannedCheckpoint},
};
use crate::models::{
    config::{ModelConfig, TokenIds},
    t5::{T5CheckpointLoadError, T5Config, T5ForConditionalGeneration},
};
use burn::tensor::backend::Backend;

mod report; // Verified reporting rules for unapplied generation defaults.

use report::{reported_min_new_tokens, reported_no_repeat_ngram, single_token};

/// Maximum accepted architecture JSON size, in bytes, before and after reading.
const MAX_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
/// Maximum accepted generation-default JSON size, in bytes, before and after reading.
const MAX_GENERATION_CONFIG_BYTES: u64 = 1024 * 1024;

/// Causal families whose packed installation path has an acceptance contract.
const PACKED_CAUSAL_MODEL_TYPES: &[&str] = &["muse_glimmer_text", "qwen3_5_text", "qwen3_moe"];

// User policy and family-specific strict loading.

/// Host-side policy for automatic local model loading.
#[derive(Debug, Clone)]
pub struct AutoLoadOptions {
    /// Strict SafeTensors discovery, dtype, and exception policy.
    ///
    /// Its default [`DTypePolicy::Auto`](crate::checkpoint::DTypePolicy::Auto)
    /// keeps the checkpoint's float dtype (for example BF16) whatever the
    /// backend's float element; set `dtype_policy:
    /// DTypePolicy::Cast(TensorDType::F32)` to execute in F32.
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
        validate_causal_options(&model_type, options)?;
        Self::load_registered_family(&model_type, &root, &json, options, device)
    }
}

/// Rejects load options that do not apply to the detected causal family.
fn validate_causal_options(model_type: &str, options: &AutoLoadOptions) -> Result<()> {
    let qwen = AutoCausalConfig::Qwen3_5.model_type();
    if options.qwen_mtp && model_type != qwen {
        return Err(Error::configuration(
            "auto_causal.qwen_mtp",
            format!("Qwen MTP is defined only for {qwen} checkpoints"),
        ));
    }
    if options.quantized_burnpack.is_some() && !PACKED_CAUSAL_MODEL_TYPES.contains(&model_type) {
        return Err(Error::BackendCapability(format!(
            "packed Burnpack loading for `{model_type}` is not yet certified; automatic packed execution is limited to {}",
            PACKED_CAUSAL_MODEL_TYPES.join(", ")
        )));
    }
    Ok(())
}

/// Strictly loads one causal family through the shared automatic pipeline.
///
/// The steps run in a fixed order so cheap validation precedes expensive
/// work: parse the family configuration, resolve generation defaults, record
/// provenance, load the optional tokenizer, discover and plan the checkpoint
/// (family-specific), initialize the graph, then execute the plan with an
/// optional packed sidecar. The bundle is returned only if the final load
/// report is strictly valid.
pub(super) fn load_causal_family<B, F>(
    root: &Path,
    json: &str,
    model_type: &str,
    options: &AutoLoadOptions,
    device: &B::Device,
) -> Result<LoadedModel<AutoCausalLm<B>>>
where
    B: Backend,
    F: CausalFamily<B> + Into<AutoCausalLm<B>>,
    <F as TensorSink>::Error: core::fmt::Display,
{
    let config = F::Config::from_json_str(json)?;
    let tokens = F::token_metadata(&config);
    let vocabulary_size = tokens.vocabulary_size;
    let generation = resolve_generation_defaults(
        GenerationConsumer::Causal,
        root,
        tokens.eos,
        tokens.pad,
        vocabulary_size,
        options.generation.as_ref(),
    )?;
    let source = source_metadata(root, model_type, options)?;
    let tokenizer = load_optional_tokenizer(root, options.load_tokenizer)?;
    let PlannedCheckpoint { manifest, plan } = F::plan_checkpoint(&config, root, options)?;
    let mut model = F::init_for_load(config, options, device)?;
    let outcome = execute_causal_plan(root, &manifest, &plan, options, &mut model)?;
    outcome.into_loaded(model.into(), tokenizer, generation, source, vocabulary_size)
}

/// Successful causal checkpoint execution, preserving sidecar-only counters.
enum CausalExecutionOutcome {
    /// Ordinary SafeTensors execution without a packed overlay.
    FullPrecision(
        /// Strict tensor load report and ordinary execution counters.
        ExecutionOutcome,
    ),
    /// Mixed source-precision and Burnpack execution with provenance counters.
    Quantized(
        /// Audited mixed-precision report, including packed-sidecar execution counters.
        AuditedQuantizedExecutionOutcome,
    ),
}

impl CausalExecutionOutcome {
    /// Attaches every available execution counter to one loaded model bundle.
    fn into_loaded<M>(
        self,
        model: M,
        tokenizer: Option<ModelTokenizer>,
        generation: ResolvedGenerationDefaults,
        source: ModelSourceMetadata,
        vocabulary_size: usize,
    ) -> Result<LoadedModel<M>> {
        let loaded = match self {
            Self::FullPrecision(outcome) => LoadedModel::new_with_execution_stats(
                model,
                tokenizer,
                generation.config,
                source,
                outcome.report,
                outcome.stats,
                vocabulary_size,
            ),
            Self::Quantized(outcome) => LoadedModel::new_with_quantized_execution(
                model,
                tokenizer,
                generation.config,
                source,
                outcome,
                vocabulary_size,
            ),
        }?;
        Ok(loaded.with_unsupported_generation_settings(generation.unsupported))
    }
}

/// Executes a validated causal-family plan with an optional packed sidecar.
fn execute_causal_plan<M: TensorSink>(
    root: &Path,
    manifest: &CheckpointManifest,
    plan: &LoadPlan,
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
            .map_err(Error::quantized_checkpoint_execution)
            .map(CausalExecutionOutcome::Quantized)
    } else {
        execute_local_load_plan(manifest, plan, model)
            .map_err(Error::checkpoint_execution)
            .map(CausalExecutionOutcome::FullPrecision)
    }
}

impl<B: Backend> AutoSeq2SeqLm<B> {
    /// Strictly loads one supported encoder-decoder model from a local directory.
    ///
    /// Like the causal pipeline, it parses the configuration, resolves
    /// generation defaults, records provenance, loads the optional tokenizer,
    /// and discovers and strictly plans the checkpoint before it allocates the
    /// graph, so a checkpoint that cannot satisfy the configuration costs no
    /// per-block module allocation.
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

        let config = T5Config::from_json_str(&json)?;
        let vocabulary_size = config.vocab_size;
        let generation = resolve_generation_defaults(
            GenerationConsumer::Seq2SeqBeamSearch,
            &root,
            config.eos_token_id.as_ref(),
            config.pad_token_id.as_ref(),
            vocabulary_size,
            options.generation.as_ref(),
        )?;
        let source = source_metadata(&root, &model_type, options)?;
        let tokenizer = load_optional_tokenizer(&root, options.load_tokenizer)?;
        // Plan before building the graph, as the causal pipeline does: a
        // configuration that the discovered checkpoint cannot satisfy fails
        // here, before one module per encoder and decoder block is allocated.
        let planned = config
            .plan_checkpoint(&root, &options.checkpoint)
            .map_err(|error| match error {
                T5CheckpointLoadError::Config(error) => Error::from(error),
                T5CheckpointLoadError::Checkpoint(error) => Error::CheckpointPlan(error),
                T5CheckpointLoadError::Execution(failure) => Error::checkpoint_execution(failure),
            })?;
        let mut model = T5ForConditionalGeneration::init(config, device)?;
        let outcome = execute_local_load_plan(planned.manifest(), planned.load_plan(), &mut model)
            .map_err(Error::checkpoint_execution)?;
        LoadedModel::new_with_execution_stats(
            Self::T5(model),
            tokenizer,
            generation.config,
            source,
            outcome.report,
            outcome.stats,
            vocabulary_size,
        )
        .map(|loaded| loaded.with_unsupported_generation_settings(generation.unsupported))
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

/// Generation loop that consumes the defaults resolved for one load.
///
/// The loop decides which `generation_config.json` settings are applied and
/// which are reported as unsupported: a setting stored in [`GenerationConfig`]
/// that the consuming loop never reads would otherwise be dropped silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationConsumer {
    /// [`crate::generate_causal`], which applies every [`GenerationConfig`] logits processor.
    Causal,
    /// Encoder-decoder beam search, which reads only its own [`crate::generation::BeamSearchConfig`].
    ///
    /// Hugging Face beam search applies the `min_p`, `no_repeat_ngram_size`,
    /// and `min_new_tokens` constraints and the repetition penalty, and
    /// `do_sample` switches it to beam sampling with the temperature, top-k,
    /// and top-p warpers. None of them reaches this beam search, so they are
    /// left neutral on the defaults (greedy, unit penalties) and reported
    /// instead.
    Seq2SeqBeamSearch,
}

impl GenerationConsumer {
    /// Whether the loop applies the logits processors and sampling warpers of [`GenerationConfig`].
    const fn applies_processor_constraints(self) -> bool {
        matches!(self, Self::Causal)
    }
}

/// Generation defaults resolved for one load, plus checkpoint settings left unapplied.
#[derive(Debug, Clone, PartialEq)]
struct ResolvedGenerationDefaults {
    /// Validated defaults used to initialize generation requests.
    config: GenerationConfig,
    /// Behavior-changing `generation_config.json` settings with no equivalent here.
    unsupported: BTreeMap<String, Value>,
}

/// Resolves causal generation defaults, discarding the unsupported-settings report (tests only).
#[cfg(test)]
fn normalize_generation_defaults(
    root: &Path,
    model_eos: Option<&TokenIds>,
    model_pad: Option<&TokenIds>,
    vocabulary_size: usize,
    explicit: Option<&GenerationConfig>,
) -> Result<GenerationConfig> {
    resolve_generation_defaults(
        GenerationConsumer::Causal,
        root,
        model_eos,
        model_pad,
        vocabulary_size,
        explicit,
    )
    .map(|resolved| resolved.config)
}

/// Resolves generation defaults for `consumer` and reports the checkpoint settings left unapplied.
///
/// Precedence: an explicit configuration wins outright (and reads no file);
/// otherwise `generation_config.json` overrides the model config's token IDs.
fn resolve_generation_defaults(
    consumer: GenerationConsumer,
    root: &Path,
    model_eos: Option<&TokenIds>,
    model_pad: Option<&TokenIds>,
    vocabulary_size: usize,
    explicit: Option<&GenerationConfig>,
) -> Result<ResolvedGenerationDefaults> {
    if let Some(explicit) = explicit {
        explicit.validate(vocabulary_size)?;
        return Ok(ResolvedGenerationDefaults {
            config: explicit.clone(),
            unsupported: BTreeMap::new(),
        });
    }

    let mut config = GenerationConfig {
        eos_token_ids: normalize_token_list(model_eos, vocabulary_size, "eos_token_id")?,
        pad_token_id: normalize_pad_token(model_pad, vocabulary_size)?,
        ..GenerationConfig::default()
    };
    let mut unsupported = BTreeMap::new();
    if let Some(local) = read_generation_config(root)? {
        apply_local_generation_config(&mut config, &local, vocabulary_size, consumer)?;
        unsupported = local.unsupported_settings(consumer, &config);
    }
    config.validate(vocabulary_size)?;
    Ok(ResolvedGenerationDefaults {
        config,
        unsupported,
    })
}

/// Applies one parsed `generation_config.json` with Hugging Face semantics.
///
/// The `min_p`, n-gram, and minimum-length constraints, the repetition and
/// presence penalties, and sampling with its warpers are applied only when
/// `consumer` reads them (see [`GenerationConsumer`]); otherwise the defaults
/// stay neutral and `unsupported_settings` reports what Hugging Face would apply.
fn apply_local_generation_config(
    config: &mut GenerationConfig,
    local: &LocalGenerationConfig,
    vocabulary_size: usize,
    consumer: GenerationConsumer,
) -> Result<()> {
    let constraints = consumer.applies_processor_constraints();
    if let Some(max_new_tokens) = local.max_new_tokens {
        config.max_new_tokens = max_new_tokens;
    }
    if let Some(min_new_tokens) = local.min_new_tokens.filter(|_| constraints) {
        // Hugging Face only warns about an unreachable minimum and suppresses
        // EOS for the whole run, which the clamped value does too.
        // `unsupported_settings` reports the declared value when this clamps.
        config.min_new_tokens = clamped_min_new_tokens(min_new_tokens, config.max_new_tokens);
    }
    if let Some(size) = local.no_repeat_ngram_size.filter(|_| constraints) {
        config.no_repeat_ngram_size = size;
    }
    if constraints && local.do_sample.unwrap_or(false) {
        config.strategy = SamplingStrategy::Sample;
        if let Some(temperature) = local.temperature {
            config.temperature = temperature;
        }
        config.top_k = sampling_top_k(local.top_k, vocabulary_size);
        // Hugging Face adds its nucleus warper only for `top_p < 1`. NaN is kept
        // so validation rejects it instead of silently disabling the filter.
        config.top_p = local.top_p.filter(|top_p| top_p.is_nan() || *top_p < 1.0);
        // Hugging Face adds `MinPLogitsWarper` whenever `min_p` is set; validation
        // rejects values outside `[0, 1]`, as its warper does.
        config.min_p = local.min_p;
    }
    if let Some(penalty) = local.repetition_penalty.filter(|_| constraints) {
        config.repetition_penalty = penalty;
    }
    if let Some(penalty) = local.presence_penalty.filter(|_| constraints) {
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
    Ok(())
}

/// Supported, architecture-independent subset of `generation_config.json`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct LocalGenerationConfig {
    /// Optional upper bound on newly generated tokens, excluding the prompt.
    max_new_tokens: Option<usize>,
    /// Optional choice between stochastic sampling and greedy selection.
    do_sample: Option<bool>,
    /// Optional positive temperature used to scale sampling logits.
    temperature: Option<f64>,
    /// Optional cap on the number of candidate tokens retained for sampling.
    top_k: Option<usize>,
    /// Optional cumulative probability threshold for nucleus sampling.
    top_p: Option<f64>,
    /// Optional min-p cutoff relative to the most likely token, applied when sampling.
    min_p: Option<f64>,
    /// Optional n-gram size that may not repeat; zero disables the constraint.
    no_repeat_ngram_size: Option<usize>,
    /// Optional count of new tokens before EOS may be selected; zero disables it.
    min_new_tokens: Option<usize>,
    /// Optional multiplicative penalty applied to previously seen token logits.
    repetition_penalty: Option<f64>,
    /// Optional additive penalty for tokens already present in the sequence.
    presence_penalty: Option<f64>,
    /// Optional deterministic sampling seed overriding the runtime default.
    seed: Option<u64>,
    /// Optional scalar or list of stopping tokens, normalized against the vocabulary.
    eos_token_id: Option<TokenIds>,
    /// Optional scalar or singleton list identifying the vocabulary padding token.
    pad_token_id: Option<TokenIds>,
    /// Every other key, kept so behavior-changing settings can be reported.
    #[serde(flatten)]
    other: BTreeMap<String, Value>,
}

/// Value of a Hugging Face generation setting that leaves generation unchanged.
#[derive(Debug, Clone, Copy)]
enum NeutralSetting {
    /// `null`, or an empty list (no banned, suppressed, or biased tokens).
    Absent,
    /// This number disables the feature, such as `num_beams: 1`.
    Number(
        /// The neutral value; any other present value is reported as an unsupported setting.
        f64,
    ),
    /// This boolean disables the feature.
    Flag(
        /// The neutral value, such as `false` for `renormalize_logits`; any
        /// other present value is reported as an unsupported setting.
        bool,
    ),
    /// Any present value differs from this crate's semantics (`max_length`).
    Never,
}

/// Behavior-changing Hugging Face settings that [`GenerationConfig`] cannot express.
///
/// Each entry is `(key, neutral value, applies only when sampling)`; neutral
/// values come from Hugging Face's `_get_default_generation_params` and
/// `GenerationConfig` defaults. `max_length` caps prompt plus continuation in
/// Hugging Face, whereas this crate bounds new tokens only.
const UNSUPPORTED_GENERATION_SETTINGS: &[(&str, NeutralSetting, bool)] = &[
    ("bad_words_ids", NeutralSetting::Absent, false),
    ("begin_suppress_tokens", NeutralSetting::Absent, false),
    ("diversity_penalty", NeutralSetting::Number(0.0), false),
    ("dola_layers", NeutralSetting::Absent, false),
    (
        "encoder_no_repeat_ngram_size",
        NeutralSetting::Number(0.0),
        false,
    ),
    (
        "encoder_repetition_penalty",
        NeutralSetting::Number(1.0),
        false,
    ),
    ("epsilon_cutoff", NeutralSetting::Number(0.0), true),
    ("eta_cutoff", NeutralSetting::Number(0.0), true),
    (
        "exponential_decay_length_penalty",
        NeutralSetting::Absent,
        false,
    ),
    ("forced_bos_token_id", NeutralSetting::Absent, false),
    ("forced_eos_token_id", NeutralSetting::Absent, false),
    ("guidance_scale", NeutralSetting::Absent, false),
    ("length_penalty", NeutralSetting::Number(1.0), false),
    ("max_length", NeutralSetting::Never, false),
    ("max_time", NeutralSetting::Absent, false),
    ("min_length", NeutralSetting::Number(0.0), false),
    ("num_beam_groups", NeutralSetting::Number(1.0), false),
    ("num_beams", NeutralSetting::Number(1.0), false),
    ("num_return_sequences", NeutralSetting::Number(1.0), false),
    ("penalty_alpha", NeutralSetting::Absent, false),
    ("renormalize_logits", NeutralSetting::Flag(false), false),
    ("sequence_bias", NeutralSetting::Absent, false),
    ("stop_strings", NeutralSetting::Absent, false),
    ("suppress_tokens", NeutralSetting::Absent, false),
    ("top_h", NeutralSetting::Absent, true),
    ("typical_p", NeutralSetting::Number(1.0), true),
    ("watermarking_config", NeutralSetting::Absent, false),
];

impl NeutralSetting {
    /// Returns whether `value` is `null` or this setting's neutral value.
    ///
    /// A `false` result means the setting is reported as unsupported. That is
    /// conservative: some non-neutral values (a `min_length` shorter than the
    /// prompt, say) happen to leave Hugging Face output unchanged.
    fn matches(self, value: &Value) -> bool {
        match self {
            Self::Absent => value.is_null() || value.as_array().is_some_and(Vec::is_empty),
            Self::Number(neutral) => value.is_null() || value.as_f64() == Some(neutral),
            Self::Flag(neutral) => value.is_null() || value.as_bool() == Some(neutral),
            Self::Never => value.is_null(),
        }
    }
}

impl LocalGenerationConfig {
    /// Lists declared settings that would change Hugging Face generation but are not applied.
    ///
    /// Sampling-only warpers are reported only when `do_sample` is set,
    /// because Hugging Face ignores them under greedy decoding too. `applied`
    /// is the resolved configuration: for a consumer that does not read the
    /// logits processors, every active `min_p`, `no_repeat_ngram_size`,
    /// `min_new_tokens`, and `repetition_penalty` is reported, as are
    /// `do_sample: true` and its declared non-neutral `temperature`, `top_k`,
    /// and `top_p`; otherwise a `min_new_tokens` that was clamped to
    /// `max_new_tokens` is reported with its declared value.
    fn unsupported_settings(
        &self,
        consumer: GenerationConsumer,
        applied: &GenerationConfig,
    ) -> BTreeMap<String, Value> {
        let sampling = self.do_sample.unwrap_or(false);
        let mut report = UNSUPPORTED_GENERATION_SETTINGS
            .iter()
            .filter(|(_, _, sampling_only)| sampling || !sampling_only)
            .filter_map(|(key, neutral, _)| {
                let value = self.other.get(*key)?;
                (!neutral.matches(value)).then(|| ((*key).to_owned(), value.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        // Zero (and, for `min_p`, greedy decoding) leaves Hugging Face unchanged.
        let constraints = consumer.applies_processor_constraints();
        if let Some(count) =
            reported_min_new_tokens(self.min_new_tokens, applied.min_new_tokens, constraints)
        {
            report.insert("min_new_tokens".to_owned(), Value::from(count));
        }
        if let Some(size) = reported_no_repeat_ngram(self.no_repeat_ngram_size, constraints) {
            report.insert("no_repeat_ngram_size".to_owned(), Value::from(size));
        }
        if constraints {
            return report;
        }
        if let Some(min_p) = self.min_p.filter(|&min_p| sampling && min_p != 0.0) {
            report.insert("min_p".to_owned(), Value::from(min_p));
        }
        // Hugging Face applies its repetition penalty in every mode, beam search included.
        if let Some(penalty) = self.repetition_penalty.filter(|&penalty| penalty != 1.0) {
            report.insert("repetition_penalty".to_owned(), Value::from(penalty));
        }
        if sampling {
            // `do_sample` selects beam sampling; its warpers act only then.
            report.insert("do_sample".to_owned(), Value::Bool(true));
            if let Some(temperature) = self.temperature.filter(|&temperature| temperature != 1.0) {
                report.insert("temperature".to_owned(), Value::from(temperature));
            }
            if let Some(top_k) = self.top_k.filter(|&top_k| top_k != 0) {
                report.insert("top_k".to_owned(), Value::from(top_k));
            }
            if let Some(top_p) = self.top_p.filter(|&top_p| top_p.is_nan() || top_p < 1.0) {
                report.insert("top_p".to_owned(), Value::from(top_p));
            }
        }
        report
    }
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
///
/// The verified [`canonical_token_ids`] checks IDs in order and returns the
/// sorted, deduplicated `u32` set.
fn normalize_token_list(
    ids: Option<&TokenIds>,
    vocabulary_size: usize,
    field: &str,
) -> Result<Vec<u32>> {
    let ids = ids.map_or_else(Vec::new, |ids| ids.iter().collect::<Vec<_>>());
    canonical_token_ids(&ids, vocabulary_size).map_err(|error| match error {
        TokenIdError::OutsideVocabulary { id } => Error::configuration(
            field,
            format!("token {id} is outside vocabulary size {vocabulary_size}"),
        ),
        TokenIdError::ExceedsU32 { id } => {
            Error::configuration(field, format!("token {id} does not fit the u32 runtime"))
        }
    })
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
    single_token(&normalized)
        .map_err(|_| Error::configuration(field, "padding requires exactly one token ID"))
}

#[cfg(test)]
#[path = "load_tests.rs"]
/// Automatic loading, artifact containment, and normalized generation-policy tests.
mod tests;
