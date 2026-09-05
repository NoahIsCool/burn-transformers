//! Opt-in tests for large or externally downloaded model artifacts.
//!
//! Ordinary CI never assumes a model directory exists. Maintainers can point
//! `TRANSFORMERS_QWEN35_DIR` at an immutable local checkout and explicitly run
//! ignored tests when certifying a checkpoint revision.

#![cfg(feature = "tokenizers")]

#[cfg(feature = "metal")]
use sha2::{Digest, Sha256};
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(feature = "metal")]
use std::time::Instant;
#[cfg(feature = "metal")]
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io::Read,
    path::{Component, Path},
};
use std::{fs, path::PathBuf};

#[cfg(feature = "chat-templates")]
use transformers::{ChatMessage, ChatTemplateOptions};
use transformers::{
    EncodeOptions, LocalTokenizer,
    checkpoint::{
        ArtifactSource, CheckpointManifest, DTypePolicy, LoadOptions, LoadPlan, TensorDType,
        quantization::{
            QuantizationEligibility, QuantizationExclusion, QuantizedBurnpackArtifact,
            QuantizedCheckpointOptions, SourceFloatDType, SymmetricQuantization,
            WeightQuantizationConfig, classify_weight_for_quantization,
            quantize_local_checkpoint_to_burnpack,
        },
    },
    models::{
        config::ModelConfig,
        qwen3_5::{Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5TextConfig},
    },
};

#[cfg(feature = "metal")]
use burn::{
    backend::Metal,
    tensor::{Bool, DType, Int, Tensor, TensorData},
};
#[cfg(all(feature = "metal", not(feature = "rocm")))]
use burn::{module::Param, nn::Linear};
#[cfg(feature = "metal")]
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLanguageModel, DecodeInput, DecodeState, ForwardOptions,
    LogitsSelection, TokenMask, architecture::CausalLmInput, cache::CacheConfig,
    checkpoint::execute_local_load_plan, models::qwen3_5::Qwen3_5ForCausalLm,
};
#[cfg(all(feature = "metal", not(feature = "rocm")))]
use transformers::{
    delta_net::{DeltaNetMode, GatedDeltaNetState},
    operations,
};

/// Immutable identity recorded alongside the committed Qwen3.8 header inventory.
#[derive(Debug, serde::Deserialize)]
struct PinnedQwen38Inventory {
    checkpoint: String,
    revision: String,
    tensors: Vec<PinnedTensorInventory>,
}

/// One expected physical tensor from the pinned Qwen3.8 SafeTensors headers.
#[derive(Debug, serde::Deserialize)]
struct PinnedTensorInventory {
    name: String,
    shape: Vec<u64>,
    dtype: String,
}

/// Strict schema emitted by the pinned official Qwen3.8 reference generator.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38OfficialReference {
    schema_version: u32,
    repository: String,
    revision: String,
    reference_class: String,
    prompt: String,
    input_ids: Vec<i64>,
    execution: Qwen38ReferenceExecution,
    environment: Qwen38ReferenceEnvironment,
    provenance: Qwen38ReferenceProvenance,
    last_logits: Qwen38ReferenceLogits,
    generation: Qwen38ReferenceGeneration,
    teacher_forced: Qwen38ReferenceTeacherForced,
    timings_seconds: Qwen38ReferenceTimings,
}

/// Deterministic graph and accelerator choices used by the official runtime.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceExecution {
    dtype: String,
    device: String,
    attention_implementation: String,
    deterministic_algorithms: bool,
    seed: u64,
    use_cache: bool,
    float32_matmul_precision: String,
    mps_per_process_memory_fraction: f64,
    explicit_all_one_attention_masks: bool,
    allocator_warmup_override: Qwen38AllocatorWarmupOverride,
    kernel_status: BTreeMap<String, bool>,
}

/// Auditable scope and observations for the performance-only MPS loader workaround.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38AllocatorWarmupOverride {
    target: String,
    policy: String,
    scope: String,
    reason: String,
    upstream_source_module: String,
    call_count: usize,
    expanded_device_map_entries: usize,
    expanded_device_types: Vec<String>,
    hf_quantizer_present: bool,
    restored: bool,
}

/// Fully recorded Python process, wheel set, device, and offline controls.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceEnvironment {
    generated_at_utc: String,
    command: Vec<String>,
    python: Qwen38ReferencePython,
    packages: BTreeMap<String, String>,
    distribution_provenance: BTreeMap<String, Qwen38ReferenceDistribution>,
    distribution_manifest_sha256: String,
    device: serde_json::Value,
    offline_environment: BTreeMap<String, String>,
}

/// Installed wheel identity derived from its immutable dist-info records.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceDistribution {
    version: String,
    metadata: Qwen38ReferenceDistributionFile,
    record: Qwen38ReferenceDistributionFile,
}

/// One dist-info file whose bytes authenticate an installed Python wheel.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceDistributionFile {
    relative_path: String,
    bytes: usize,
    sha256: String,
}

/// Python interpreter details retained to make numerical provenance auditable.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferencePython {
    version: String,
    executable: String,
    implementation: String,
    platform: String,
}

/// Generator, upstream source, and checkpoint identities for one reference run.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceProvenance {
    generator: Qwen38ReferenceGenerator,
    reference_sources: Vec<Qwen38ReferenceSource>,
    artifacts: Vec<Qwen38ReferenceArtifact>,
    artifact_manifest_sha256: String,
    checkpoint_tensor_bytes: u64,
    loaded_parameter_count: u64,
    parameter_dtype_elements: BTreeMap<String, u64>,
    loading_info: Qwen38ReferenceLoadingInfo,
}

/// Exact generator content and Git state used to create the external result.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceGenerator {
    relative_path: String,
    sha256: String,
    git_head: String,
    git_dirty: bool,
}

/// One installed official Transformers source file and its byte identity.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceSource {
    module: String,
    relative_path: String,
    path: String,
    bytes: u64,
    sha256: String,
}

/// One immutable checkpoint control file or SafeTensors shard.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceArtifact {
    relative_path: String,
    bytes: u64,
    sha256: String,
}

/// Empty strict-load classifications produced by the official text-only class.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceLoadingInfo {
    missing_keys: Qwen38ReferenceLoadingClass,
    unexpected_keys: Qwen38ReferenceLoadingClass,
    mismatched_keys: Qwen38ReferenceLoadingClass,
    error_msgs: Qwen38ReferenceLoadingClass,
}

/// Count and stable digest for one official checkpoint-load classification.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceLoadingClass {
    count: usize,
    sha256: String,
}

/// Integrity metadata and diagnostic samples for the raw prompt logit vector.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceLogits {
    relative_path: String,
    encoding: String,
    elements: usize,
    bytes: usize,
    sha256: String,
    finite: bool,
    top1_token_id: i64,
    top10: Vec<Qwen38ReferenceLogit>,
    probes: Vec<Qwen38ReferenceLogit>,
    l2_norm: f64,
    sum: f64,
}

/// One token-indexed logit sampled from the complete raw reference vector.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceLogit {
    token_id: usize,
    logit: f64,
}

/// Exactly 32 manually selected cached greedy tokens and their top-two margins.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceGeneration {
    max_new_tokens: usize,
    token_ids: Vec<i64>,
    margins: Vec<f64>,
    stability_policy: Qwen38Top1StabilityPolicy,
    decision_stability: Vec<Qwen38DecisionStability>,
    decoded_text: String,
    eos_token_ids: Vec<i64>,
    first_eos_position: Option<usize>,
    completed_32_decisions: bool,
    minimum_observed_margin: f64,
    minimum_margin_to_epsilon_ratio: f64,
}

/// Scale-aware proof policy used to classify each F16 greedy decision.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38Top1StabilityPolicy {
    name: String,
    elementwise_perturbation_ulps: f64,
    safety_factor: f64,
    rationale: String,
}

/// Complete top-two and binary16-spacing evidence for one cached decision.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38DecisionStability {
    step: usize,
    top1_token_id: i64,
    top2_token_id: i64,
    top1_logit: f64,
    top2_logit: f64,
    margin: f64,
    top1_f16_ulp: f64,
    top2_f16_ulp: f64,
    elementwise_epsilon: f64,
    margin_to_epsilon_ratio: f64,
    stable_under_budget: bool,
}

/// Official stateless all-logit predictions and cached-prefix parity diagnostics.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceTeacherForced {
    input_ids: Vec<i64>,
    prediction_positions: Vec<usize>,
    all_position_top1_token_ids: Vec<i64>,
    continuation_top1_token_ids: Vec<i64>,
    cached_top1_token_ids: Vec<i64>,
    top1_equal: bool,
    per_prefix_cosine: Vec<f64>,
    minimum_cosine: f64,
    per_prefix_max_abs_error: Vec<f64>,
    maximum_abs_error: f64,
    prompt_cached_stateless_cosine: f64,
}

/// Wall-clock measurements retained as diagnostics rather than performance gates.
#[cfg(feature = "metal")]
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen38ReferenceTimings {
    load: f64,
    stateless_prompt: f64,
    cached_generation_32: f64,
    teacher_forced_all_logits: f64,
}

/// Resolves the opt-in Qwen3.8 artifact root without silently choosing mutable data.
fn pinned_qwen38_root() -> PathBuf {
    PathBuf::from(std::env::var_os("TRANSFORMERS_QWEN38_DIR").expect(
        "set TRANSFORMERS_QWEN38_DIR to Qwen/Qwen3.8-27B revision \
             1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ))
}

/// Parses the checked-in tensor-level inventory for the immutable Qwen3.8 revision.
fn pinned_qwen38_inventory() -> PinnedQwen38Inventory {
    serde_json::from_str(include_str!(
        "fixtures/qwen3_5_checkpoint/qwen3_8_27b_inventory.json"
    ))
    .expect("the committed Qwen3.8 tensor inventory must remain valid JSON")
}

/// Partitions the mapped Qwen3.8 parameters using the public weight-only policy.
fn qwen38_quantization_partition(
    plan: &LoadPlan,
) -> (Vec<String>, Vec<(String, QuantizationExclusion)>) {
    let mut quantized = Vec::new();
    let mut excluded = Vec::new();
    for tensor in &plan.tensors {
        let shape = tensor
            .shape
            .iter()
            .map(|&dimension| {
                usize::try_from(dimension)
                    .expect("Qwen3.8 checkpoint dimensions must fit the host architecture")
            })
            .collect::<Vec<_>>();
        match classify_weight_for_quantization(&tensor.target, &shape) {
            QuantizationEligibility::Eligible => quantized.push(tensor.target.clone()),
            QuantizationEligibility::Excluded(reason) => {
                excluded.push((tensor.target.clone(), reason));
            }
        }
    }
    (quantized, excluded)
}

/// Asserts the immutable Qwen3.8 mixed-precision policy inventory.
fn assert_qwen38_quantization_partition(
    quantized: &[String],
    excluded: &[(String, QuantizationExclusion)],
) {
    assert_eq!(quantized.len(), 497);
    assert_eq!(excluded.len(), 369);
    for (reason, expected) in [
        (QuantizationExclusion::Mtp, 15),
        (QuantizationExclusion::Embedding, 1),
        (QuantizationExclusion::Normalization, 209),
        (QuantizationExclusion::RecurrentScalar, 96),
        (QuantizationExclusion::RecurrentConvolution, 48),
    ] {
        assert_eq!(
            excluded
                .iter()
                .filter(|(_, observed)| *observed == reason)
                .count(),
            expected,
            "unexpected Qwen3.8 exclusion count for {reason:?}"
        );
    }
}

/// Validates every pinned Qwen3.8 configuration field and physical tensor header.
#[test]
#[ignore = "requires Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"]
fn validates_pinned_qwen38_config_and_header_inventory() {
    let root = pinned_qwen38_root();
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let observed_config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("the pinned nested Qwen3.8 text configuration must validate");
    let mut expected_config =
        Qwen3_5TextConfig::from_json_str(include_str!("fixtures/qwen3_8_27b.json"))
            .expect("the committed Qwen3.8 configuration must validate");
    expected_config.layer_types = expected_config.resolved_layer_types();
    assert_eq!(observed_config, expected_config);
    assert_eq!(observed_config.model_type, "qwen3_5_text");
    assert_eq!(observed_config.num_hidden_layers, 64);
    assert_eq!(observed_config.mtp_num_hidden_layers, 1);

    let expected = pinned_qwen38_inventory();
    assert_eq!(expected.checkpoint, "Qwen/Qwen3.8-27B");
    assert_eq!(
        expected.revision,
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    );
    let manifest = CheckpointManifest::discover(&root, &LoadOptions::default())
        .expect("all 18 pinned SafeTensors shard headers must pass strict discovery");
    assert_eq!(manifest.shards.len(), 18);
    assert_eq!(manifest.tensors.len(), expected.tensors.len());
    assert_eq!(manifest.tensors.len(), 1_199);
    match &manifest.source {
        ArtifactSource::Index {
            relative_path,
            declared_total_size,
        } => {
            assert_eq!(relative_path, "model.safetensors.index.json");
            assert_eq!(*declared_total_size, Some(manifest.tensor_bytes));
        }
        ArtifactSource::Single { relative_path } => {
            panic!("expected a sharded checkpoint index, found {relative_path}")
        }
        _ => panic!("expected the standard sharded checkpoint index"),
    }

    for tensor in &expected.tensors {
        let observed = manifest
            .tensor(&tensor.name)
            .unwrap_or_else(|| panic!("pinned header is missing tensor `{}`", tensor.name));
        assert_eq!(
            observed.shape, tensor.shape,
            "shape changed for `{}`",
            tensor.name
        );
        assert_eq!(
            observed.dtype.as_str(),
            tensor.dtype,
            "dtype changed for `{}`",
            tensor.name
        );
    }

    let language_tensors = manifest
        .tensors
        .keys()
        .filter(|name| name.starts_with("model.language_model."))
        .count();
    let visual_tensors = manifest
        .tensors
        .keys()
        .filter(|name| name.starts_with("model.visual."))
        .count();
    let mtp_tensors = manifest
        .tensors
        .keys()
        .filter(|name| name.starts_with("mtp."))
        .count();
    assert_eq!(language_tensors, 850);
    assert_eq!(usize::from(manifest.tensor("lm_head.weight").is_some()), 1);
    assert_eq!(mtp_tensors, 15);
    assert_eq!(visual_tensors, 333);
    assert_eq!(language_tensors + 1 + mtp_tensors + visual_tensors, 1_199);
}

/// Proves the full 866-tensor text-plus-MTP graph consumes every non-visual key.
#[test]
#[ignore = "requires Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"]
fn strictly_maps_pinned_qwen38_text_and_mtp() {
    let root = pinned_qwen38_root();
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("the pinned nested Qwen3.8 text configuration must validate");
    let manifest = CheckpointManifest::discover(&root, &LoadOptions::default())
        .expect("the pinned SafeTensors inventory must pass strict discovery");
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::Multimodal,
            include_mtp: true,
        })
        .expect("the pinned Qwen3.8 topology must produce a text-plus-MTP contract");
    let plan = contract
        .build_load_plan(&manifest, &LoadOptions::default())
        .expect("every text and MTP tensor must map while visual tensors are allowlisted");

    assert_eq!(contract.expected_tensors().len(), 866);
    assert_eq!(plan.tensors.len(), 866);
    assert!(plan.aliases.is_empty());
    assert!(plan.report.is_strictly_valid());
    assert_eq!(plan.report.allowlisted_ignored.len(), 333);
    assert!(
        plan.report
            .allowlisted_ignored
            .iter()
            .all(|name| name.starts_with("model.visual."))
    );
    assert_eq!(
        plan.tensors
            .iter()
            .filter(|tensor| tensor.target.starts_with("mtp."))
            .count(),
        15
    );

    let planned_sources = plan
        .tensors
        .iter()
        .flat_map(|tensor| tensor.sources.iter().map(|source| source.name.as_str()))
        .collect::<std::collections::BTreeSet<_>>();
    let expected_sources = manifest
        .tensors
        .keys()
        .filter(|name| !name.starts_with("model.visual."))
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(planned_sources, expected_sources);
    let (quantized, excluded) = qwen38_quantization_partition(&plan);
    assert_qwen38_quantization_partition(&quantized, &excluded);
    eprintln!("Qwen3.8 pre-allocation memory estimate: {:?}", plan.memory);
}

/// Exercises the exact local Qwen3.8 tokenizer vocabulary without network access.
#[test]
#[ignore = "requires Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"]
fn validates_pinned_qwen38_tokenizer() {
    let tokenizer = LocalTokenizer::from_directory(pinned_qwen38_root())
        .expect("the pinned Qwen3.8 tokenizer artifacts must load locally");
    let prompt = "Explain why cached decoding is faster in one sentence.";
    let encoding = tokenizer
        .encode(
            prompt,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("the pinned tokenizer must encode ordinary text");
    let token_ids = &encoding.token_ids()[0];
    assert!(!token_ids.is_empty());
    assert_eq!(encoding.valid_tokens()[0], vec![true; token_ids.len()]);
    assert_eq!(
        tokenizer
            .decode(token_ids, true)
            .expect("the pinned tokenizer must decode its own ordinary-text IDs"),
        prompt
    );
}

/// Validates the pinned Qwen3.8 no-thinking assistant-prefix chat rendering.
#[cfg(feature = "chat-templates")]
#[test]
#[ignore = "requires Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"]
fn validates_pinned_qwen38_chat_template() {
    let tokenizer = LocalTokenizer::from_directory(pinned_qwen38_root())
        .expect("the pinned Qwen3.8 tokenizer artifacts must load locally");
    let prompt = "Explain why cached decoding is faster in one sentence.";
    let rendered = tokenizer
        .render_chat(
            &[ChatMessage::text("user", prompt)],
            &ChatTemplateOptions {
                add_generation_prompt: true,
                enable_thinking: Some(false),
                ..ChatTemplateOptions::default()
            },
        )
        .expect("the pinned Qwen3.8 chat template must render a text-only request");
    assert_eq!(
        rendered,
        format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    );
}

/// Loads Qwen3.8 in Metal F16, executes its base and MTP graphs, and checks cached parity.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires pinned Qwen3.8 weights, about 80 GiB of free unified memory, and Metal"]
fn loads_pinned_qwen38_text_and_mtp_and_runs_metal() {
    let root = pinned_qwen38_root();
    let device = Default::default();
    let load_started = Instant::now();
    let loaded = AutoCausalLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            load_tokenizer: true,
            qwen_mtp: true,
            repository: Some("Qwen/Qwen3.8-27B".into()),
            revision: Some("1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0".into()),
            ..AutoLoadOptions::default()
        },
        &device,
    )
    .expect("the public auto loader must stream the complete Qwen3.8 text-plus-MTP graph");
    let load_seconds = load_started.elapsed().as_secs_f64();
    assert!(loaded.load_report().is_strictly_valid());
    assert_eq!(loaded.load_report().applied.len(), 866);
    assert_eq!(loaded.load_report().allowlisted_ignored.len(), 333);
    assert_eq!(loaded.source().repository(), Some("Qwen/Qwen3.8-27B"));
    assert_eq!(
        loaded.source().revision(),
        Some("1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    );

    let encoding = loaded
        .tokenizer()
        .expect("the public loaded-model bundle must retain the local tokenizer")
        .encode(
            "Hello world, cached decoding.",
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("the smoke prompt must tokenize");
    let sequence = encoding.token_ids()[0].len();
    assert_eq!(
        encoding.token_ids()[0],
        [9_419, 1_814, 11, 19_954, 45_543, 13],
        "the pinned six-token smoke fixture changed"
    );
    let input = encoding
        .to_causal_input::<Metal>(&device)
        .expect("the tokenizer batch must convert to a public causal input");
    let unmasked_input = CausalLmInput {
        token_ids: input.token_ids.clone(),
        token_mask: None,
        position_ids: input.position_ids.clone(),
    };
    let diagnostic_options = ForwardOptions {
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let stateless_started = Instant::now();
    let stateless = loaded
        .model()
        .forward(input.clone(), &diagnostic_options)
        .expect("the public auto model must execute stateless Qwen3.8 inference");
    assert_eq!(stateless.logits.dims(), [1, 1, 248_320]);
    let stateless_logits = metal_tensor_values("Qwen3.8 stateless logits", &stateless.logits);
    let stateless_seconds = stateless_started.elapsed().as_secs_f64();
    assert!(stateless_logits.iter().all(|value| value.is_finite()));
    let unmasked = loaded
        .model()
        .forward(unmasked_input, &diagnostic_options)
        .expect("an all-valid mask must be equivalent to an omitted mask");
    let unmasked_logits = metal_tensor_values("Qwen3.8 unmasked logits", &unmasked.logits);
    let mask_cosine = cosine_similarity(&stateless_logits, &unmasked_logits);

    let mut state = loaded
        .model()
        .new_decode_state(
            CacheConfig::new(1, sequence + 1).expect("the smoke cache geometry must validate"),
            1,
            &device,
        )
        .expect("the public auto model must allocate Qwen3.8 hybrid decode state");
    let prefill_started = Instant::now();
    let cached = loaded
        .model()
        .prefill(input, &mut state, &diagnostic_options)
        .expect("the public auto model must execute cached Qwen3.8 prefill");
    let cached_logits = metal_tensor_values("Qwen3.8 cached logits", &cached.logits);
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    assert!(cached_logits.iter().all(|value| value.is_finite()));
    let cached_cosine = cosine_similarity(&stateless_logits, &cached_logits);
    let stateless_trace = metal_tensor_values(
        "Qwen3.8 stateless hidden trace",
        &Tensor::cat(stateless.hidden_states.clone(), 0),
    );
    let unmasked_trace = metal_tensor_values(
        "Qwen3.8 unmasked hidden trace",
        &Tensor::cat(unmasked.hidden_states, 0),
    );
    let cached_trace = metal_tensor_values(
        "Qwen3.8 cached hidden trace",
        &Tensor::cat(cached.hidden_states, 0),
    );
    let trace_width = sequence
        * stateless
            .hidden_states
            .first()
            .expect("requested stateless hidden trace")
            .dims()[2];
    let mask_layer_cosines = stateless_trace
        .chunks_exact(trace_width)
        .zip(unmasked_trace.chunks_exact(trace_width))
        .map(|(masked, unmasked)| cosine_similarity(masked, unmasked))
        .collect::<Vec<_>>();
    let cached_layer_cosines = stateless_trace
        .chunks_exact(trace_width)
        .zip(cached_trace.chunks_exact(trace_width))
        .map(|(stateless, cached)| cosine_similarity(stateless, cached))
        .collect::<Vec<_>>();
    for (layer, cosine) in mask_layer_cosines.iter().copied().enumerate() {
        assert!(
            cosine >= 0.999,
            "Qwen3.8 all-valid/omitted-mask hidden cosine at trace {layer} was {cosine}"
        );
    }
    for (layer, cosine) in cached_layer_cosines.iter().copied().enumerate() {
        assert!(
            cosine >= 0.999,
            "Qwen3.8 stateless/cached hidden cosine at trace {layer} was {cosine}"
        );
    }
    let minimum_mask_layer_cosine = mask_layer_cosines
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let minimum_cached_layer_cosine = cached_layer_cosines
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        mask_cosine >= 0.999,
        "Qwen3.8 all-valid/omitted-mask cosine {mask_cosine} is below the gate"
    );
    assert!(
        cached_cosine >= 0.999,
        "Qwen3.8 cached/stateless cosine {cached_cosine} is below the acceptance gate"
    );
    let stateless_top = stateless
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("Metal top-1 token IDs must use i32 storage");
    let cached_top = cached
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("cached Metal top-1 token IDs must use i32 storage");
    assert_eq!(cached_top, stateless_top);

    let next_token = i64::from(stateless_top[0]);
    let stateless_logits_hash = fnv1a64_f32(&stateless_logits);
    let target_hidden = stateless
        .final_hidden_state
        .expect("MTP diagnostics requested the base model's final hidden state")
        .narrow(1, sequence - 1, 1);
    let mtp_started = Instant::now();
    let mtp_prediction = match loaded.model() {
        AutoCausalLm::Qwen3_5(model) => model
            .forward_mtp(
                Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(vec![next_token], [1, 1]),
                    &device,
                ),
                target_hidden,
                Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(vec![i64::try_from(sequence - 1).unwrap()], [1, 1]),
                    &device,
                ),
                0,
            )
            .expect("the loaded Qwen3.8 MTP graph must execute"),
        model => panic!("expected Qwen3.5-family auto dispatch, found {model:?}"),
    };
    assert_eq!(mtp_prediction.logits.dims(), [1, 1, 248_320]);
    let mtp_logits = metal_tensor_values("Qwen3.8 MTP logits", &mtp_prediction.logits);
    let mtp_seconds = mtp_started.elapsed().as_secs_f64();
    assert!(mtp_logits.iter().all(|value| value.is_finite()));

    let decode_started = Instant::now();
    let decoded = loaded
        .model()
        .decode(
            DecodeInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(vec![next_token], [1, 1]),
                    &device,
                ),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("the public auto model must execute one recurrent decode step");
    assert_eq!(decoded.logits.dims(), [1, 1, 248_320]);
    let decoded_logits = metal_tensor_values("Qwen3.8 decoded logits", &decoded.logits);
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    assert!(decoded_logits.iter().all(|value| value.is_finite()));
    assert_eq!(state.lengths(), &[sequence + 1]);
    eprintln!(
        "Qwen3.8 F16: prompt_tokens={sequence}, first_top1={next_token}, \
         mask_cosine={mask_cosine:.9}, cached_cosine={cached_cosine:.9}, \
         min_mask_layer_cosine={minimum_mask_layer_cosine:.9}, \
         min_cached_layer_cosine={minimum_cached_layer_cosine:.9}, \
         stateless_logits_fnv1a64={stateless_logits_hash:016x}, \
         load_seconds={load_seconds:.3}, \
         stateless_seconds={stateless_seconds:.3}, prefill_seconds={prefill_seconds:.3}, \
         mtp_seconds={mtp_seconds:.3}, decode_seconds={decode_seconds:.3}"
    );
}

/// Loads the external official-reference directory selected for Qwen3.8 acceptance.
#[cfg(feature = "metal")]
fn pinned_qwen38_reference_root() -> PathBuf {
    const BUNDLE_DIRECTORY: &str = "qwen3.8-27b-1d4bf0f2-reference.bundle";

    PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN38_REFERENCE_DIR").expect(
            "set TRANSFORMERS_QWEN38_REFERENCE_DIR to the external parent directory passed to \
             scripts/generate_qwen38_reference.py",
        ),
    )
    .join(BUNDLE_DIRECTORY)
}

/// Trusted hashes from the independently reviewed official reference run.
///
/// The JSON digest authenticates the complete run record; the raw digest
/// independently authenticates the full-vocabulary numerical payload used for
/// cosine parity. Both outputs were generated by commit `b7aab93` under the
/// strict offline MPS workflow and reviewed before being pinned here.
#[cfg(feature = "metal")]
const TRUSTED_QWEN38_REFERENCE_JSON_SHA256: Option<&str> =
    Some("1d4f7d3c02a996c72a1d767d672f933ce83792dea116c4964b794003ac345ce2");

/// Trusted digest for the generated full-vocabulary F32 prompt-logit vector.
#[cfg(feature = "metal")]
const TRUSTED_QWEN38_REFERENCE_RAW_SHA256: Option<&str> =
    Some("0deca0c71c576a529ec3ceec9297d10df9d3d024302b3418e09f74bec2eca998");

/// Returns whether a string is one canonical lowercase SHA-256 hexadecimal digest.
#[cfg(feature = "metal")]
fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Computes SHA-256 through the audited implementation used by the test harness.
#[cfg(feature = "metal")]
fn sha256_hex(payload: &[u8]) -> String {
    format!("{:x}", Sha256::digest(payload))
}

/// Opens one macOS file descriptor without following its final path component.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn open_regular_file_no_follow(path: &Path, expected_bytes: u64, maximum_bytes: u64) -> File {
    // Darwin's O_NOFOLLOW value is stable ABI. `OpenOptionsExt` is used instead
    // of a path metadata check so a symlink swap between inspection and open
    // cannot redirect the descriptor.
    const DARWIN_O_NOFOLLOW: i32 = 0x0000_0100;
    const DARWIN_O_NONBLOCK: i32 = 0x0000_0004;

    assert!(
        expected_bytes <= maximum_bytes,
        "expected size {expected_bytes} exceeds the {maximum_bytes}-byte safety cap for {}",
        path.display()
    );
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(DARWIN_O_NOFOLLOW | DARWIN_O_NONBLOCK)
        .open(path)
        .unwrap_or_else(|error| {
            panic!(
                "open regular file {} without following links: {error}",
                path.display()
            )
        });
    let metadata = file
        .metadata()
        .unwrap_or_else(|error| panic!("inspect opened file {}: {error}", path.display()));
    assert!(
        metadata.file_type().is_file(),
        "{} must resolve to a regular file descriptor",
        path.display()
    );
    assert_eq!(
        metadata.len(),
        expected_bytes,
        "unexpected byte length for {}",
        path.display()
    );
    file
}

/// Inspects a small path without following its final component before opening it.
#[cfg(feature = "metal")]
fn bounded_regular_file_length(path: &Path, maximum_bytes: u64) -> u64 {
    let metadata = fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("inspect bounded file {}: {error}", path.display()));
    assert!(
        metadata.file_type().is_file(),
        "{} must be a non-symlink regular file",
        path.display()
    );
    assert!(
        metadata.len() <= maximum_bytes,
        "{} is {} bytes, exceeding the {maximum_bytes}-byte safety cap",
        path.display(),
        metadata.len()
    );
    metadata.len()
}

/// Makes non-macOS compilation explicit while keeping the Metal gate macOS-only.
#[cfg(all(feature = "metal", not(target_os = "macos")))]
fn open_regular_file_no_follow(_path: &Path, _expected_bytes: u64, _maximum_bytes: u64) -> File {
    panic!("the Qwen3.8 Metal reference gate requires macOS O_NOFOLLOW semantics")
}

/// Reads one small regular file with a fixed allocation and a hard byte ceiling.
#[cfg(feature = "metal")]
fn read_regular_file_exact(path: &Path, expected_bytes: u64, maximum_bytes: u64) -> Vec<u8> {
    let file = open_regular_file_no_follow(path, expected_bytes, maximum_bytes);
    let capacity = usize::try_from(expected_bytes)
        .expect("bounded reference file length must fit the host address space");
    let mut payload = Vec::with_capacity(capacity);
    file.take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut payload)
        .unwrap_or_else(|error| panic!("read bounded regular file {}: {error}", path.display()));
    assert_eq!(
        payload.len(),
        capacity,
        "opened file {} changed length while being read",
        path.display()
    );
    payload
}

/// Streams one regular file into SHA-256 while enforcing its recorded length.
#[cfg(feature = "metal")]
fn sha256_regular_file(path: &Path, expected_bytes: u64) -> String {
    const HASH_BUFFER_BYTES: usize = 8 * 1024 * 1024;

    let mut file = open_regular_file_no_follow(path, expected_bytes, expected_bytes);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    let mut observed_bytes = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .unwrap_or_else(|error| panic!("hash regular file {}: {error}", path.display()));
        if read == 0 {
            break;
        }
        observed_bytes = observed_bytes
            .checked_add(u64::try_from(read).expect("read length fits u64"))
            .expect("hashed file length must not overflow u64");
        assert!(
            observed_bytes <= expected_bytes,
            "{} grew beyond its recorded {expected_bytes} bytes while being hashed",
            path.display()
        );
        digest.update(&buffer[..read]);
    }
    assert_eq!(
        observed_bytes,
        expected_bytes,
        "{} changed length while being hashed",
        path.display()
    );
    format!("{:x}", digest.finalize())
}

/// Fails closed until the generated result has an independently pinned digest.
#[cfg(feature = "metal")]
fn assert_trusted_reference_digest(label: &str, observed: &str, trusted: Option<&str>) {
    let trusted = trusted.unwrap_or_else(|| {
        panic!(
            "trusted {label} SHA-256 has not been pinned; set the narrowly scoped constant in \
             tests/local_artifacts.rs after reviewing the generated bundle (observed {observed})"
        )
    });
    assert!(is_sha256(trusted), "trusted {label} SHA-256 is malformed");
    assert_eq!(observed, trusted, "generated {label} identity changed");
}

/// Returns true only for a single ordinary UTF-8 filename component.
#[cfg(feature = "metal")]
fn is_safe_artifact_filename(relative_path: &str) -> bool {
    let mut components = Path::new(relative_path).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

/// Returns true for a nonempty relative path containing only ordinary components.
#[cfg(feature = "metal")]
fn is_safe_relative_path(relative_path: &str) -> bool {
    let mut saw_component = false;
    for component in Path::new(relative_path).components() {
        if !matches!(component, Component::Normal(_)) {
            return false;
        }
        saw_component = true;
    }
    saw_component
}

/// Authenticates every checkpoint input named by the trusted reference record.
#[cfg(feature = "metal")]
fn verify_qwen38_reference_artifacts(artifacts: &[Qwen38ReferenceArtifact]) {
    let checkpoint_root = pinned_qwen38_root();
    let checkpoint_root = fs::canonicalize(&checkpoint_root).unwrap_or_else(|error| {
        panic!(
            "canonicalize Qwen3.8 checkpoint root {}: {error}",
            checkpoint_root.display()
        )
    });
    assert!(
        checkpoint_root.is_dir(),
        "Qwen3.8 checkpoint root must be a directory"
    );

    for artifact in artifacts {
        assert!(
            is_safe_artifact_filename(&artifact.relative_path),
            "unsafe checkpoint artifact path `{}`",
            artifact.relative_path
        );
        let path = checkpoint_root.join(&artifact.relative_path);
        let digest = sha256_regular_file(&path, artifact.bytes);
        assert_eq!(
            digest, artifact.sha256,
            "checkpoint artifact content changed for `{}`",
            artifact.relative_path
        );
    }
}

/// Finds deterministic top-two token IDs and their F32 subtraction margin.
#[cfg(feature = "metal")]
fn top_two_with_margin(values: &[f32]) -> (usize, usize, f64) {
    assert!(values.len() >= 2, "top-two selection requires two logits");
    let mut ids = (0..values.len()).collect::<Vec<_>>();
    ids.select_nth_unstable_by(1, |left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    let first = ids[0];
    let second = ids[1];
    (first, second, f64::from(values[first] - values[second]))
}

/// Computes the wider adjacent IEEE-754 binary16 spacing at an exact F16 value.
#[cfg(feature = "metal")]
fn binary16_ulp(value: f64) -> f64 {
    assert!(value.is_finite(), "binary16 value must be finite");
    let magnitude = value.abs();
    assert!(
        magnitude <= 65_504.0,
        "value {value} lies outside finite binary16"
    );
    let ulp = if magnitude < 2.0_f64.powi(-14) {
        2.0_f64.powi(-24)
    } else {
        let exponent = magnitude.log2().floor() as i32;
        2.0_f64.powi(exponent - 10)
    };
    let ulp_units = value / ulp;
    assert_eq!(
        ulp_units,
        ulp_units.round(),
        "value {value} is not exactly representable as binary16"
    );
    ulp
}

/// Creates the explicit all-valid mask promised by the reference schema.
#[cfg(feature = "metal")]
fn all_valid_metal_mask(
    sequence: usize,
    device: &<Metal as burn::tensor::backend::BackendTypes>::Device,
) -> TokenMask<Metal> {
    TokenMask::new(Tensor::<Metal, 2, Bool>::from_data(
        TensorData::new(vec![true; sequence], [1, sequence]),
        device,
    ))
}

/// Reproduces the generator's canonical artifact-manifest digest construction.
#[cfg(feature = "metal")]
fn qwen38_reference_artifact_digest(artifacts: &[Qwen38ReferenceArtifact]) -> String {
    let mut artifacts = artifacts.iter().collect::<Vec<_>>();
    artifacts.sort_unstable_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let mut canonical = String::new();
    for artifact in artifacts {
        canonical.push_str(&artifact.relative_path);
        canonical.push('\0');
        canonical.push_str(&artifact.bytes.to_string());
        canonical.push('\0');
        canonical.push_str(&artifact.sha256);
        canonical.push('\n');
    }
    sha256_hex(canonical.as_bytes())
}

/// Reproduces the generator's canonical installed-distribution identity digest.
#[cfg(feature = "metal")]
fn qwen38_reference_distribution_digest(
    distributions: &BTreeMap<String, Qwen38ReferenceDistribution>,
) -> String {
    let mut canonical = String::new();
    for (name, distribution) in distributions {
        canonical.push_str(name);
        canonical.push('\0');
        canonical.push_str(&distribution.version);
        canonical.push('\n');
        for (logical_name, file) in [
            ("metadata", &distribution.metadata),
            ("record", &distribution.record),
        ] {
            canonical.push_str(logical_name);
            canonical.push('\0');
            canonical.push_str(&file.relative_path);
            canonical.push('\0');
            canonical.push_str(&file.bytes.to_string());
            canonical.push('\0');
            canonical.push_str(&file.sha256);
            canonical.push('\n');
        }
    }
    sha256_hex(canonical.as_bytes())
}

/// Checks one recomputed floating-point diagnostic with a scale-aware tolerance.
#[cfg(feature = "metal")]
fn assert_reference_close(label: &str, actual: f64, expected: f64) {
    let tolerance = expected.abs().max(1.0) * 1.0e-8;
    assert!(
        actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= tolerance,
        "{label} mismatch: expected {expected}, found {actual}, tolerance {tolerance}"
    );
}

/// Reads and exhaustively validates the official JSON and raw F32 prompt vector.
#[cfg(feature = "metal")]
fn read_pinned_qwen38_reference() -> (Qwen38OfficialReference, Vec<f32>) {
    const JSON_FILENAME: &str = "qwen3.8-27b-1d4bf0f2-reference.json";
    const RAW_FILENAME: &str = "qwen3.8-27b-1d4bf0f2-six-token-f16-last-logits.f32le";
    const MAXIMUM_JSON_BYTES: u64 = 1024 * 1024;
    const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const ARTIFACT_MANIFEST_SHA256: &str =
        "188de1fbf58b9e6897b9dc218d5afc4d9b9082657121db3ddb238b5296d66833";

    let root = pinned_qwen38_reference_root();
    let root_metadata = fs::symlink_metadata(&root)
        .unwrap_or_else(|error| panic!("inspect reference bundle {}: {error}", root.display()));
    assert!(
        root_metadata.file_type().is_dir(),
        "reference bundle must be a non-symlink directory"
    );
    let root = fs::canonicalize(&root)
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", root.display()));
    assert!(
        !root.starts_with(env!("CARGO_MANIFEST_DIR")),
        "large generated references must remain outside the crate"
    );
    let json_path = root.join(JSON_FILENAME);
    let json_bytes = read_regular_file_exact(
        &json_path,
        bounded_regular_file_length(&json_path, MAXIMUM_JSON_BYTES),
        MAXIMUM_JSON_BYTES,
    );
    let json_digest = sha256_hex(&json_bytes);
    assert_trusted_reference_digest(
        "Qwen3.8 reference JSON",
        &json_digest,
        TRUSTED_QWEN38_REFERENCE_JSON_SHA256,
    );
    let reference: Qwen38OfficialReference = serde_json::from_slice(&json_bytes)
        .unwrap_or_else(|error| panic!("parse strict reference {}: {error}", json_path.display()));

    assert_eq!(reference.schema_version, 1);
    assert_eq!(reference.repository, "Qwen/Qwen3.8-27B");
    assert_eq!(
        reference.revision,
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    );
    assert_eq!(reference.reference_class, "transformers.Qwen3_5ForCausalLM");
    // The repeated three-word cycle keeps all 32 reference choices comfortably
    // outside the binary16 perturbation boundary while still exercising varied
    // token identities, recurrent states, and whitespace-sensitive tokenization.
    assert_eq!(reference.prompt, "one two three one two three");
    assert_eq!(reference.input_ids, [588, 1_330, 2_250, 799, 1_330, 2_250]);
    assert_eq!(reference.execution.dtype, "float16");
    assert_eq!(reference.execution.device, "mps");
    assert_eq!(reference.execution.attention_implementation, "eager");
    assert!(reference.execution.deterministic_algorithms);
    assert_eq!(reference.execution.seed, 0);
    assert!(reference.execution.use_cache);
    assert_eq!(reference.execution.float32_matmul_precision, "highest");
    assert_eq!(reference.execution.mps_per_process_memory_fraction, 1.0);
    assert!(reference.execution.explicit_all_one_attention_masks);
    let allocator_override = &reference.execution.allocator_warmup_override;
    assert_eq!(
        allocator_override.target,
        "transformers.modeling_utils.caching_allocator_warmup"
    );
    assert_eq!(
        allocator_override.policy,
        "mps-exclusive-noop-during-from-pretrained"
    );
    assert_eq!(
        allocator_override.scope,
        "Qwen3_5ForCausalLM.from_pretrained"
    );
    assert_eq!(
        allocator_override.reason,
        "Transformers 5.2 caching allocator warmup attempts an MPS allocation with a \
         one-dimensional tensor exceeding INT_MAX; skipping this performance-only allocation \
         does not alter checkpoint tensors or model math."
    );
    assert_eq!(
        allocator_override.upstream_source_module,
        "transformers.modeling_utils"
    );
    assert_eq!(allocator_override.call_count, 1);
    assert!(allocator_override.expanded_device_map_entries > 0);
    assert_eq!(allocator_override.expanded_device_types, ["mps"]);
    assert!(!allocator_override.hf_quantizer_present);
    assert!(allocator_override.restored);
    assert_eq!(
        reference.execution.kernel_status,
        BTreeMap::from([
            ("causal_conv1d_kernel_available".into(), false),
            ("chunk_gated_delta_kernel_available".into(), false),
            ("fast_path_available".into(), false),
            ("recurrent_gated_delta_kernel_available".into(), false),
        ])
    );

    assert!(reference.environment.generated_at_utc.ends_with('Z'));
    assert!(
        reference
            .environment
            .command
            .iter()
            .any(|part| part.ends_with("generate_qwen38_reference.py"))
    );
    assert!(!reference.environment.python.version.is_empty());
    assert!(!reference.environment.python.executable.is_empty());
    assert_eq!(reference.environment.python.implementation, "CPython");
    assert!(!reference.environment.python.platform.is_empty());
    let expected_packages = BTreeMap::from([
        ("accelerate".into(), "1.12.0".into()),
        ("numpy".into(), "2.4.3".into()),
        ("safetensors".into(), "0.7.0".into()),
        ("tokenizers".into(), "0.22.2".into()),
        ("torch".into(), "2.10.0".into()),
        ("transformers".into(), "5.2.0".into()),
    ]);
    assert_eq!(reference.environment.packages, expected_packages);
    assert_eq!(
        reference
            .environment
            .distribution_provenance
            .keys()
            .collect::<Vec<_>>(),
        reference.environment.packages.keys().collect::<Vec<_>>()
    );
    for (distribution, identity) in &reference.environment.distribution_provenance {
        assert_eq!(
            reference.environment.packages.get(distribution),
            Some(&identity.version),
            "distribution provenance version changed for {distribution}"
        );
        for (label, file, expected_filename) in [
            ("METADATA", &identity.metadata, "METADATA"),
            ("RECORD", &identity.record, "RECORD"),
        ] {
            assert!(
                is_safe_relative_path(&file.relative_path),
                "unsafe {distribution} {label} path"
            );
            assert_eq!(
                Path::new(&file.relative_path)
                    .file_name()
                    .and_then(|value| value.to_str()),
                Some(expected_filename)
            );
            assert!(file.bytes > 0, "empty {distribution} {label} provenance");
            assert!(
                is_sha256(&file.sha256),
                "malformed {distribution} {label} SHA-256"
            );
        }
    }
    let distribution_digest =
        qwen38_reference_distribution_digest(&reference.environment.distribution_provenance);
    assert_eq!(
        distribution_digest,
        "b814bb3254d51a28c6d74e80c04c3532c9836a0aadc4fab0eed410d6612d59dd"
    );
    assert_eq!(
        reference.environment.distribution_manifest_sha256,
        distribution_digest
    );
    assert_eq!(reference.environment.device["kind"], "mps");
    assert_eq!(reference.environment.device["built"], true);
    assert_eq!(reference.environment.device["available"], true);
    assert_eq!(
        reference.environment.offline_environment,
        BTreeMap::from([
            ("HF_DATASETS_OFFLINE".into(), "1".into()),
            ("HF_DEACTIVATE_ASYNC_LOAD".into(), "1".into()),
            ("HF_HUB_OFFLINE".into(), "1".into()),
            ("PYTORCH_ENABLE_MPS_FALLBACK".into(), "0".into()),
            ("TOKENIZERS_PARALLELISM".into(), "false".into()),
            ("TRANSFORMERS_OFFLINE".into(), "1".into()),
        ])
    );

    assert_eq!(
        reference.provenance.generator.relative_path,
        "scripts/generate_qwen38_reference.py"
    );
    assert_eq!(
        reference.provenance.generator.sha256,
        sha256_hex(include_bytes!("../scripts/generate_qwen38_reference.py")),
        "reference must be regenerated after its generator changes"
    );
    assert_eq!(reference.provenance.generator.git_head.len(), 40);
    assert!(
        reference
            .provenance
            .generator
            .git_head
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    );
    assert!(
        !reference.provenance.generator.git_dirty,
        "advertised reference evidence must come from a clean generator"
    );
    assert_eq!(reference.provenance.reference_sources.len(), 8);
    let expected_sources = BTreeMap::from([
        (
            "transformers.cache_utils",
            (
                "cache_utils.py",
                60_147_u64,
                "e4fe20ca1eeb17ea0ce7e62d1dbd0d6eab32e7072044f293612403b8e0c477fd",
            ),
        ),
        (
            "transformers.configuration_utils",
            (
                "configuration_utils.py",
                57_894,
                "52100272e3bf59551c072a064053e90bc30f1fd0aa9e48fac7ad4b675c63ebcc",
            ),
        ),
        (
            "transformers.conversion_mapping",
            (
                "conversion_mapping.py",
                19_302,
                "c8efcc6ce2224939299804d495ba4f361ab596e59f33e3db28e9b4629fc481e0",
            ),
        ),
        (
            "transformers.masking_utils",
            (
                "masking_utils.py",
                74_373,
                "c3c82f7b7b6e03d3f04ba6c6c58a3dd6910623636452ec67ef70e3eb522f9fe7",
            ),
        ),
        (
            "transformers.modeling_attn_mask_utils",
            (
                "modeling_attn_mask_utils.py",
                21_930,
                "fb3ec46639a19fda1eb99339f4e7770332a0f4f75b9de4abb34a0b1ee379838b",
            ),
        ),
        (
            "transformers.modeling_utils",
            (
                "modeling_utils.py",
                247_340,
                "2f45c2139ce633e6e3eae9290675dd5815643b37c27f2cc4dd83586fb648f7bb",
            ),
        ),
        (
            "transformers.models.qwen3_5.configuration_qwen3_5",
            (
                "models/qwen3_5/configuration_qwen3_5.py",
                14_576,
                "2280c6e6bd9d66d7281155243f67cf5fed2756a828af41316566afe611ff16c0",
            ),
        ),
        (
            "transformers.models.qwen3_5.modeling_qwen3_5",
            (
                "models/qwen3_5/modeling_qwen3_5.py",
                97_339,
                "d1ae3856f53763591ec65054129af46e003a1715efcf7eef2b752a31e85526b8",
            ),
        ),
    ]);
    let mut observed_source_modules = BTreeSet::new();
    for source in &reference.provenance.reference_sources {
        assert!(observed_source_modules.insert(source.module.as_str()));
        let &(expected_path, expected_bytes, expected_sha256) = expected_sources
            .get(source.module.as_str())
            .unwrap_or_else(|| panic!("unexpected reference source module `{}`", source.module));
        assert_eq!(source.relative_path, expected_path);
        assert_eq!(source.bytes, expected_bytes);
        assert_eq!(source.sha256, expected_sha256);
        assert!(Path::new(&source.path).is_absolute());
        assert!(is_safe_relative_path(&source.relative_path));
        assert!(Path::new(&source.path).ends_with(&source.relative_path));
        assert!(is_sha256(&source.sha256));
    }
    assert_eq!(
        observed_source_modules,
        expected_sources.keys().copied().collect::<BTreeSet<_>>()
    );
    assert!(
        observed_source_modules.contains(
            reference
                .execution
                .allocator_warmup_override
                .upstream_source_module
                .as_str()
        )
    );
    assert_eq!(reference.provenance.artifacts.len(), 24);
    let mut artifact_names = BTreeSet::new();
    for artifact in &reference.provenance.artifacts {
        assert!(is_safe_artifact_filename(&artifact.relative_path));
        assert!(artifact_names.insert(artifact.relative_path.as_str()));
        assert!(artifact.bytes > 0);
        assert!(is_sha256(&artifact.sha256));
    }
    let expected_control_names = BTreeSet::from([
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]);
    let observed_control_names = artifact_names
        .iter()
        .copied()
        .filter(|name| !name.ends_with(".safetensors"))
        .collect::<BTreeSet<_>>();
    assert_eq!(observed_control_names, expected_control_names);
    assert_eq!(
        artifact_names
            .iter()
            .filter(|name| name.ends_with(".safetensors"))
            .count(),
        18
    );
    let artifact_digest = qwen38_reference_artifact_digest(&reference.provenance.artifacts);
    assert_eq!(artifact_digest, ARTIFACT_MANIFEST_SHA256);
    assert_eq!(
        reference.provenance.artifact_manifest_sha256,
        artifact_digest
    );
    assert_eq!(reference.provenance.checkpoint_tensor_bytes, 55_562_855_904);
    assert_eq!(reference.provenance.loaded_parameter_count, 26_895_998_464);
    assert_eq!(reference.provenance.parameter_dtype_elements.len(), 1);
    assert_eq!(
        reference.provenance.parameter_dtype_elements.get("float16"),
        Some(&reference.provenance.loaded_parameter_count)
    );
    for class in [
        &reference.provenance.loading_info.missing_keys,
        &reference.provenance.loading_info.unexpected_keys,
        &reference.provenance.loading_info.mismatched_keys,
        &reference.provenance.loading_info.error_msgs,
    ] {
        assert_eq!(class.count, 0);
        assert_eq!(class.sha256, EMPTY_SHA256);
    }
    // The trusted JSON digest binds this inventory. Re-hashing the actual local
    // files now prevents a same-revision directory with altered bytes from
    // reaching Burn's loader below.
    verify_qwen38_reference_artifacts(&reference.provenance.artifacts);

    assert_eq!(reference.last_logits.relative_path, RAW_FILENAME);
    assert_eq!(reference.last_logits.encoding, "ieee754-f32-little-endian");
    assert_eq!(reference.last_logits.elements, 248_320);
    assert_eq!(reference.last_logits.bytes, 248_320 * size_of::<f32>());
    assert!(reference.last_logits.finite);
    assert!(is_sha256(&reference.last_logits.sha256));
    let raw_path = root.join(&reference.last_logits.relative_path);
    let raw = read_regular_file_exact(
        &raw_path,
        u64::try_from(reference.last_logits.bytes).expect("raw byte count fits u64"),
        u64::try_from(reference.last_logits.bytes).expect("raw byte count fits u64"),
    );
    assert_eq!(raw.len(), reference.last_logits.bytes);
    let raw_digest = sha256_hex(&raw);
    assert_trusted_reference_digest(
        "Qwen3.8 reference raw logits",
        &raw_digest,
        TRUSTED_QWEN38_REFERENCE_RAW_SHA256,
    );
    assert_eq!(raw_digest, reference.last_logits.sha256);
    let values = raw
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte F32 logit")))
        .collect::<Vec<_>>();
    assert_eq!(values.len(), reference.last_logits.elements);
    assert!(values.iter().all(|value| value.is_finite()));
    let top1 = values
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .expect("reference vocabulary must be non-empty");
    assert_eq!(
        i64::try_from(top1).expect("token ID fits i64"),
        reference.last_logits.top1_token_id
    );
    let (first_id, second_id, observed_first_margin) = top_two_with_margin(&values);
    assert_eq!(first_id, top1);
    assert_ne!(first_id, second_id);
    let mut sorted_ids = (0..values.len()).collect::<Vec<_>>();
    sorted_ids.sort_unstable_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    assert_eq!(reference.last_logits.top10.len(), 10);
    for (record, expected_id) in reference
        .last_logits
        .top10
        .iter()
        .zip(sorted_ids.iter().copied())
    {
        assert_eq!(record.token_id, expected_id);
        assert_eq!(record.logit as f32, values[expected_id]);
    }
    let mut probe_ids = BTreeSet::new();
    for probe in &reference.last_logits.probes {
        assert!(probe_ids.insert(probe.token_id), "duplicate logit probe");
        assert_eq!(probe.logit as f32, values[probe.token_id]);
    }
    let observed_sum = values.iter().map(|value| f64::from(*value)).sum::<f64>();
    let observed_norm = values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    assert_reference_close(
        "reference logit sum",
        observed_sum,
        reference.last_logits.sum,
    );
    assert_reference_close(
        "reference logit L2 norm",
        observed_norm,
        reference.last_logits.l2_norm,
    );

    assert_eq!(reference.generation.max_new_tokens, 32);
    assert_eq!(reference.generation.token_ids.len(), 32);
    assert_eq!(reference.generation.margins.len(), 32);
    assert_eq!(reference.generation.decision_stability.len(), 32);
    let stability_policy = &reference.generation.stability_policy;
    assert_eq!(stability_policy.name, "binary16-elementwise-ulp-bound");
    assert_eq!(stability_policy.elementwise_perturbation_ulps, 1.0);
    assert_eq!(stability_policy.safety_factor, 2.0);
    assert_eq!(
        stability_policy.rationale,
        "If every score changes by at most epsilon, the top score can fall by epsilon while its \
         strongest competitor rises by epsilon; margin > 2*epsilon therefore preserves top-1. \
         Epsilon is one local binary16 ULP of the wider-spaced top-two logit, budgeting one \
         representable step per score. This fixture screen complements, and does not replace, \
         exact Burn/reference token parity."
    );
    assert_eq!(reference.generation.eos_token_ids, [248_046, 248_044]);
    let first_eos_position = reference
        .generation
        .token_ids
        .iter()
        .position(|token| reference.generation.eos_token_ids.contains(token));
    assert_eq!(reference.generation.first_eos_position, first_eos_position);
    assert_eq!(first_eos_position, None);
    assert_eq!(
        reference.generation.completed_32_decisions,
        reference.generation.token_ids.len() == reference.generation.max_new_tokens
            && first_eos_position.is_none()
    );
    assert!(reference.generation.completed_32_decisions);
    assert!(
        reference
            .generation
            .token_ids
            .iter()
            .all(|token| (0..248_320).contains(token))
    );
    let mut observed_minimum_margin = f64::INFINITY;
    let mut observed_minimum_ratio = f64::INFINITY;
    for (step, decision) in reference.generation.decision_stability.iter().enumerate() {
        assert_eq!(decision.step, step);
        assert_eq!(decision.top1_token_id, reference.generation.token_ids[step]);
        assert_ne!(decision.top1_token_id, decision.top2_token_id);
        assert!((0..248_320).contains(&decision.top1_token_id));
        assert!((0..248_320).contains(&decision.top2_token_id));
        assert!(decision.top1_logit.is_finite());
        assert!(decision.top2_logit.is_finite());
        assert!(decision.top1_logit > decision.top2_logit);
        assert_reference_close(
            "decision top-two subtraction margin",
            decision.top1_logit - decision.top2_logit,
            decision.margin,
        );
        assert_reference_close(
            "decision duplicated generation margin",
            reference.generation.margins[step],
            decision.margin,
        );
        let top1_ulp = binary16_ulp(decision.top1_logit);
        let top2_ulp = binary16_ulp(decision.top2_logit);
        assert_reference_close(
            "decision top-1 binary16 ULP",
            top1_ulp,
            decision.top1_f16_ulp,
        );
        assert_reference_close(
            "decision top-2 binary16 ULP",
            top2_ulp,
            decision.top2_f16_ulp,
        );
        let epsilon = top1_ulp.max(top2_ulp) * stability_policy.elementwise_perturbation_ulps;
        assert_reference_close(
            "decision elementwise perturbation epsilon",
            epsilon,
            decision.elementwise_epsilon,
        );
        assert_reference_close(
            "decision margin-to-epsilon ratio",
            decision.margin / epsilon,
            decision.margin_to_epsilon_ratio,
        );
        assert_eq!(
            decision.stable_under_budget,
            decision.margin_to_epsilon_ratio > stability_policy.safety_factor
        );
        assert!(decision.stable_under_budget);
        observed_minimum_margin = observed_minimum_margin.min(decision.margin);
        observed_minimum_ratio = observed_minimum_ratio.min(decision.margin_to_epsilon_ratio);
    }
    assert_reference_close(
        "first greedy top-two margin from raw logits",
        observed_first_margin,
        reference.generation.margins[0],
    );
    assert_eq!(
        i64::try_from(first_id).expect("top token ID fits i64"),
        reference.generation.token_ids[0]
    );
    assert_reference_close(
        "minimum generated greedy margin",
        observed_minimum_margin,
        reference.generation.minimum_observed_margin,
    );
    assert_reference_close(
        "minimum generated margin-to-epsilon ratio",
        observed_minimum_ratio,
        reference.generation.minimum_margin_to_epsilon_ratio,
    );
    assert!(reference.generation.minimum_observed_margin > 0.0);
    assert!(reference.generation.minimum_margin_to_epsilon_ratio > stability_policy.safety_factor);
    assert!(!reference.generation.decoded_text.contains('\0'));
    assert!(
        reference
            .generation
            .token_ids
            .iter()
            .all(|token| !reference.generation.eos_token_ids.contains(token)),
        "the accepted 32-decision continuation must not terminate early"
    );

    let mut expected_teacher_input = reference.input_ids.clone();
    expected_teacher_input.extend(reference.generation.token_ids.iter().take(31).copied());
    assert_eq!(reference.teacher_forced.input_ids, expected_teacher_input);
    assert_eq!(reference.teacher_forced.input_ids.len(), 37);
    assert_eq!(
        reference.teacher_forced.prediction_positions,
        (5..37).collect::<Vec<_>>()
    );
    assert_eq!(
        reference.teacher_forced.all_position_top1_token_ids.len(),
        37
    );
    assert_eq!(
        reference.teacher_forced.continuation_top1_token_ids,
        reference.generation.token_ids
    );
    assert_eq!(
        reference.teacher_forced.cached_top1_token_ids,
        reference.generation.token_ids
    );
    assert!(reference.teacher_forced.top1_equal);
    assert_eq!(reference.teacher_forced.per_prefix_cosine.len(), 32);
    assert_eq!(reference.teacher_forced.per_prefix_max_abs_error.len(), 32);
    assert!(
        reference
            .teacher_forced
            .per_prefix_cosine
            .iter()
            .all(|cosine| cosine.is_finite() && *cosine >= 0.999 && *cosine <= 1.000_001)
    );
    assert_reference_close(
        "minimum official cached/stateless cosine",
        reference
            .teacher_forced
            .per_prefix_cosine
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        reference.teacher_forced.minimum_cosine,
    );
    assert!(reference.teacher_forced.minimum_cosine >= 0.999);
    assert!(
        reference
            .teacher_forced
            .per_prefix_max_abs_error
            .iter()
            .all(|error| error.is_finite() && *error >= 0.0)
    );
    assert_reference_close(
        "maximum official cached/stateless absolute error",
        reference
            .teacher_forced
            .per_prefix_max_abs_error
            .iter()
            .copied()
            .fold(0.0, f64::max),
        reference.teacher_forced.maximum_abs_error,
    );
    assert!(reference.teacher_forced.prompt_cached_stateless_cosine >= 0.999);
    for timing in [
        reference.timings_seconds.load,
        reference.timings_seconds.stateless_prompt,
        reference.timings_seconds.cached_generation_32,
        reference.timings_seconds.teacher_forced_all_logits,
    ] {
        assert!(timing.is_finite() && timing >= 0.0);
    }
    (reference, values)
}

/// Compares Burn Metal against the complete official Qwen3.8 F16 reference run.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires pinned Qwen3.8 weights, generated official F16 references, about 80 GiB of unified memory, and Metal"]
fn matches_pinned_qwen38_official_reference_on_metal() {
    let (reference, reference_logits) = read_pinned_qwen38_reference();
    let root = pinned_qwen38_root();
    let device = Default::default();
    let loaded = AutoCausalLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            load_tokenizer: false,
            qwen_mtp: false,
            repository: Some(reference.repository.clone()),
            revision: Some(reference.revision.clone()),
            ..AutoLoadOptions::default()
        },
        &device,
    )
    .expect("strictly stream the pinned Qwen3.8 text graph into Metal F16");
    assert!(loaded.load_report().is_strictly_valid());
    assert_eq!(loaded.load_report().applied.len(), 851);
    assert_eq!(loaded.load_report().allowlisted_ignored.len(), 348);
    assert_eq!(
        loaded.source().repository(),
        Some(reference.repository.as_str())
    );
    assert_eq!(
        loaded.source().revision(),
        Some(reference.revision.as_str())
    );
    assert!(matches!(loaded.model(), AutoCausalLm::Qwen3_5(_)));

    let prompt_length = reference.input_ids.len();
    let prompt_ids = Tensor::<Metal, 2, Int>::from_data(
        TensorData::new(reference.input_ids.clone(), [1, prompt_length]),
        &device,
    );
    let prompt_mask = all_valid_metal_mask(prompt_length, &device);
    let stateless = loaded
        .model()
        .forward(
            CausalLmInput {
                token_ids: prompt_ids.clone(),
                token_mask: Some(prompt_mask.clone()),
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect("run stateless Burn Qwen3.8 prompt inference");
    assert_eq!(stateless.logits.dims(), [1, 1, 248_320]);
    let actual_logits = metal_tensor_values("Qwen3.8 official-reference logits", &stateless.logits);
    assert!(actual_logits.iter().all(|value| value.is_finite()));
    let cosine = cosine_similarity(&actual_logits, &reference_logits);
    assert!(
        cosine >= 0.999,
        "official/Burn full-vocabulary cosine {cosine} is below the 0.999 gate"
    );
    let stateless_top1 = stateless
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("Metal prompt top-1 token must use i32 storage");
    assert_eq!(
        i64::from(stateless_top1[0]),
        reference.last_logits.top1_token_id
    );

    let mut state = loaded
        .model()
        .new_decode_state(
            CacheConfig::new(1, prompt_length + reference.generation.max_new_tokens)
                .expect("official reference cache geometry must validate"),
            1,
            &device,
        )
        .expect("allocate the official-reference Qwen3.8 decode state");
    let mut cached = loaded
        .model()
        .prefill(
            CausalLmInput {
                token_ids: prompt_ids,
                token_mask: Some(prompt_mask),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("prefill the official-reference Qwen3.8 prompt");
    let mut cached_tokens = Vec::with_capacity(reference.generation.max_new_tokens);
    for (step, expected) in reference.generation.token_ids.iter().copied().enumerate() {
        let token = i64::from(
            *cached
                .logits
                .clone()
                .argmax(2)
                .to_data()
                .to_vec::<i32>()
                .expect("cached Metal top-1 token must use i32 storage")
                .first()
                .expect("one cached row must select one token"),
        );
        assert_eq!(
            token, expected,
            "Burn cached greedy token differs from the official reference at step {step}"
        );
        cached_tokens.push(token);
        if step + 1 < reference.generation.max_new_tokens {
            cached = loaded
                .model()
                .decode(
                    DecodeInput {
                        token_ids: Tensor::<Metal, 2, Int>::from_data(
                            TensorData::new(vec![token], [1, 1]),
                            &device,
                        ),
                        token_mask: Some(all_valid_metal_mask(1, &device)),
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap_or_else(|error| panic!("Qwen3.8 cached decode step {step}: {error}"));
        }
    }
    assert_eq!(cached_tokens, reference.generation.token_ids);
    assert_eq!(
        state.lengths(),
        &[prompt_length + reference.generation.max_new_tokens - 1]
    );

    let teacher_length = reference.teacher_forced.input_ids.len();
    let teacher = loaded
        .model()
        .forward(
            CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(
                        reference.teacher_forced.input_ids.clone(),
                        [1, teacher_length],
                    ),
                    &device,
                ),
                token_mask: Some(all_valid_metal_mask(teacher_length, &device)),
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .expect("run Qwen3.8 stateless teacher-forced all-logit inference");
    assert_eq!(teacher.logits.dims(), [1, teacher_length, 248_320]);
    let all_top1 = teacher
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("teacher-forced Metal top-1 tokens must use i32 storage")
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();
    assert_eq!(
        all_top1, reference.teacher_forced.all_position_top1_token_ids,
        "Burn teacher-forced predictions differ at one or more sequence positions"
    );
    let continuation_top1 = reference
        .teacher_forced
        .prediction_positions
        .iter()
        .map(|&position| all_top1[position])
        .collect::<Vec<_>>();
    assert_eq!(
        continuation_top1, cached_tokens,
        "Burn cached/stateless top-1 predictions must agree at every generated prefix"
    );
    assert_eq!(
        continuation_top1,
        reference.teacher_forced.continuation_top1_token_ids
    );
    eprintln!(
        "Qwen3.8 official F16 parity: prompt_cosine={cosine:.9}, top1={}, greedy_tokens={}",
        reference.last_logits.top1_token_id,
        cached_tokens.len()
    );
}

/// The test harness SHA-256 wrapper matches standard empty and `abc` vectors.
#[cfg(feature = "metal")]
#[test]
fn qwen38_reference_sha256_is_stable() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

/// Reference path and top-two helpers enforce deterministic boundary behavior.
#[cfg(feature = "metal")]
#[test]
fn qwen38_reference_validation_helpers_cover_edges() {
    assert!(is_safe_artifact_filename("config.json"));
    for unsafe_name in ["", ".", "..", "model/config.json", "/config.json"] {
        assert!(
            !is_safe_artifact_filename(unsafe_name),
            "unexpectedly accepted `{unsafe_name}`"
        );
    }

    assert_eq!(top_two_with_margin(&[1.0, 4.0, 2.0]), (1, 2, 2.0));
    assert_eq!(
        top_two_with_margin(&[3.0, 3.0, 2.0]),
        (0, 1, 0.0),
        "equal logits must use the lower token ID first"
    );
    assert_eq!(binary16_ulp(0.0), 2.0_f64.powi(-24));
    assert_eq!(binary16_ulp(7.996_093_75), 2.0_f64.powi(-8));
    assert_eq!(binary16_ulp(8.0), 2.0_f64.powi(-7));
    assert!(
        std::panic::catch_unwind(|| binary16_ulp(0.1)).is_err(),
        "non-binary16 values must fail closed"
    );

    let mut artifacts = vec![
        Qwen38ReferenceArtifact {
            relative_path: "b".into(),
            bytes: 2,
            sha256: "b".repeat(64),
        },
        Qwen38ReferenceArtifact {
            relative_path: "a".into(),
            bytes: 1,
            sha256: "a".repeat(64),
        },
    ];
    let first = qwen38_reference_artifact_digest(&artifacts);
    artifacts.reverse();
    assert_eq!(qwen38_reference_artifact_digest(&artifacts), first);
}

/// Bounded reads reject final-component symlinks and cap allocations.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn qwen38_reference_reader_is_bounded_and_no_follow() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().expect("create reference-reader fixture directory");
    let regular = directory.path().join("regular");
    fs::write(&regular, b"reference").expect("write regular reference-reader fixture");
    assert_eq!(read_regular_file_exact(&regular, 9, 9), b"reference");
    assert!(
        std::panic::catch_unwind(|| read_regular_file_exact(&regular, 9, 8)).is_err(),
        "the byte ceiling must be enforced before allocation"
    );

    let link = directory.path().join("link");
    symlink(&regular, &link).expect("create reference-reader symlink fixture");
    assert!(
        std::panic::catch_unwind(|| read_regular_file_exact(&link, 9, 9)).is_err(),
        "the final path component must never be followed"
    );
}

/// Computes a stable FNV-1a digest over little-endian F32 logit bit patterns.
fn fnv1a64_f32(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// The acceptance digest fixes byte order and the FNV constants explicitly.
#[test]
fn f32_acceptance_digest_is_stable() {
    assert_eq!(fnv1a64_f32(&[]), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64_f32(&[0.0]), 0x4d25_767f_9dce_13f5);
    assert_ne!(fnv1a64_f32(&[0.0]), fnv1a64_f32(&[-0.0]));
}

/// Converts the pinned Qwen3.8 text-plus-MTP graph to a persistent Q8 Burnpack.
#[test]
#[ignore = "requires pinned Qwen3.8 weights and TRANSFORMERS_QWEN38_Q8_OUTPUT_DIR on external storage"]
fn quantizes_pinned_qwen38_weights_to_q8_burnpack() {
    quantize_pinned_qwen38_weights_to_burnpack(
        SymmetricQuantization::Q8,
        "TRANSFORMERS_QWEN38_Q8_OUTPUT_DIR",
    );
}

/// Converts the pinned Qwen3.8 text-plus-MTP graph to a persistent Q4 Burnpack.
#[test]
#[ignore = "requires pinned Qwen3.8 weights and TRANSFORMERS_QWEN38_Q4_OUTPUT_DIR on external storage"]
fn quantizes_pinned_qwen38_weights_to_q4_burnpack() {
    quantize_pinned_qwen38_weights_to_burnpack(
        SymmetricQuantization::Q4,
        "TRANSFORMERS_QWEN38_Q4_OUTPUT_DIR",
    );
}

/// Streams one precision of the immutable Qwen3.8 checkpoint into Burnpack.
///
/// The checkpoint contract deliberately includes the auxiliary MTP graph even
/// though the v2 quantization policy retains every `mtp.*` parameter in F16.
/// Consequently, this gate proves both complete source consumption and the exact
/// mixed-precision boundary rather than merely checking that a sidecar was
/// written. The destination is explicitly supplied because finalization briefly
/// needs both staging and published copies of these large artifacts.
fn quantize_pinned_qwen38_weights_to_burnpack(
    scheme: SymmetricQuantization,
    output_directory_env: &str,
) {
    let root = pinned_qwen38_root();
    let output_directory =
        PathBuf::from(std::env::var_os(output_directory_env).unwrap_or_else(|| {
            panic!("set {output_directory_env} to a dedicated external output directory")
        }));
    fs::create_dir_all(&output_directory).unwrap_or_else(|error| {
        panic!(
            "create Qwen3.8 quantized output directory {}: {error}",
            output_directory.display()
        )
    });
    assert!(
        output_directory.is_dir(),
        "{output_directory_env} must name a directory, got {}",
        output_directory.display()
    );

    let label = match scheme {
        SymmetricQuantization::Q8 => "q8",
        SymmetricQuantization::Q4 => "q4",
    };
    let output = output_directory.join(format!("qwen3.8-27b-1d4bf0f2-{label}-g64.bpk"));
    assert!(
        !output.exists(),
        "Qwen3.8 conversion is create-only; choose an empty output directory or retain the existing artifact at {}",
        output.display()
    );

    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("the pinned nested Qwen3.8 text configuration must validate");
    // Burnpack v2 binds the complete executable plan, including its target
    // dtype. Conversion must therefore use the same streaming BF16-to-F16 plan
    // that the certified M3 Metal load consumes below.
    let load_options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        ..LoadOptions::default()
    };
    let manifest = CheckpointManifest::discover(&root, &load_options)
        .expect("all pinned Qwen3.8 SafeTensors headers must pass strict discovery");
    assert_eq!(manifest.shards.len(), 18);
    assert_eq!(manifest.tensors.len(), 1_199);

    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::Multimodal,
            include_mtp: true,
        })
        .expect("the Qwen3.8 topology must produce its text-plus-MTP contract");
    let plan = contract
        .build_load_plan(&manifest, &load_options)
        .expect("Qwen3.8 text and MTP tensors must map while vision remains allowlisted");
    assert_eq!(plan.tensors.len(), 866);
    assert!(plan.aliases.is_empty());
    assert!(plan.report.is_strictly_valid());
    assert_eq!(plan.report.allowlisted_ignored.len(), 333);
    assert!(
        plan.report
            .allowlisted_ignored
            .iter()
            .all(|name| name.starts_with("model.visual."))
    );

    let (expected_quantized, expected_excluded) = qwen38_quantization_partition(&plan);
    assert_qwen38_quantization_partition(&expected_quantized, &expected_excluded);

    let outcome = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions {
            weights: WeightQuantizationConfig {
                scheme,
                ..WeightQuantizationConfig::default()
            },
            ..QuantizedCheckpointOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("Qwen3.8 {label} conversion failed: {error}"));

    assert!(outcome.report.is_strictly_valid());
    assert_eq!(outcome.report.applied.len(), 866);
    assert_eq!(outcome.report.allowlisted_ignored.len(), 333);
    assert_eq!(outcome.report.dtype_converted.len(), 866);
    assert_eq!(outcome.report.quantized.len(), 497);
    assert_eq!(outcome.artifact.tensor_count, 497);
    assert_eq!(outcome.staging.tensor_count, 497);
    assert_eq!(outcome.execution.tensors_applied, 866);
    assert_eq!(outcome.execution.source.tensor_reads, 866);
    assert_eq!(outcome.execution.source.max_open_shards, 1);
    assert_eq!(
        outcome.execution.delivered_tensor_bytes,
        plan.memory.target_parameter_bytes
    );
    assert_eq!(outcome.artifact.path, output);
    assert!(outcome.artifact.quantized_payload_bytes > 0);
    assert!(
        outcome.artifact.quantized_payload_bytes < plan.memory.target_parameter_bytes,
        "{label} payload must be smaller than the mapped BF16 text-plus-MTP parameters"
    );

    let observed_quantized = outcome
        .report
        .quantized
        .iter()
        .map(|record| record.target.clone())
        .collect::<Vec<_>>();
    assert_eq!(observed_quantized, expected_quantized);
    assert!(outcome.report.quantized.iter().all(|record| {
        record.scheme == format!("symmetric-{label}-g64-flat-row-major") && record.bytes > 0
    }));
    let observed_excluded = outcome
        .excluded
        .iter()
        .map(|record| (record.target.clone(), record.reason))
        .collect::<Vec<_>>();
    assert_eq!(observed_excluded, expected_excluded);

    let mut artifact = QuantizedBurnpackArtifact::open(&output)
        .unwrap_or_else(|error| panic!("open Qwen3.8 {label} Burnpack: {error}"));
    let validation = artifact
        .validate_all()
        .unwrap_or_else(|error| panic!("validate Qwen3.8 {label} Burnpack: {error}"));
    assert_eq!(validation.tensor_count, 497);
    assert_eq!(
        validation.payload_bytes,
        outcome.artifact.quantized_payload_bytes
    );
    assert_eq!(artifact.manifest().tensors.len(), 497);
    assert!(artifact.manifest().packed_execution.is_available());
    assert!(artifact.manifest().tensors.iter().all(|(name, tensor)| {
        !name.starts_with("mtp.")
            && tensor.scheme == scheme
            && tensor.group_size.get() == 64
            && tensor.source_dtype == SourceFloatDType::F16
    }));
    assert_eq!(
        artifact
            .manifest()
            .tensors
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        expected_quantized
    );
    drop(artifact);

    #[cfg(all(feature = "metal", not(feature = "rocm")))]
    runs_pinned_qwen38_quantized_text_and_mtp_on_metal(&root, &output, scheme, 497);

    eprintln!(
        "Qwen3.8 {label}: tensors={}, excluded={}, payload_bytes={}, artifact_bytes={}, largest_tensor_bytes={}",
        outcome.artifact.tensor_count,
        outcome.excluded.len(),
        outcome.artifact.quantized_payload_bytes,
        outcome.artifact.artifact_bytes,
        outcome.artifact.largest_materialized_tensor_bytes,
    );
}

/// Loads one retained Qwen3.8 sidecar and exercises packed base and F16 MTP paths.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn runs_pinned_qwen38_quantized_text_and_mtp_on_metal(
    root: &std::path::Path,
    burnpack: &std::path::Path,
    scheme: SymmetricQuantization,
    quantized_count: usize,
) {
    let label = match scheme {
        SymmetricQuantization::Q8 => "q8",
        SymmetricQuantization::Q4 => "q4",
    };
    let device = Default::default();
    let loaded = AutoCausalLm::<Metal>::load_local(
        root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            load_tokenizer: false,
            qwen_mtp: true,
            quantized_burnpack: Some(burnpack.to_path_buf()),
            repository: Some("Qwen/Qwen3.8-27B".into()),
            revision: Some("1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0".into()),
            ..AutoLoadOptions::default()
        },
        &device,
    )
    .unwrap_or_else(|error| panic!("load mixed F16/{label} Qwen3.8 model: {error}"));
    assert!(loaded.load_report().is_strictly_valid());
    assert_eq!(loaded.load_report().applied.len(), 866);
    assert_eq!(loaded.load_report().quantized.len(), quantized_count);
    assert_eq!(loaded.load_report().allowlisted_ignored.len(), 333);
    assert!(
        loaded.load_report().quantized.iter().all(|record| {
            record.scheme == format!("burn-qfloat-symmetric-{label}-g64-block-1xg")
        })
    );

    let AutoCausalLm::Qwen3_5(model) = loaded.model() else {
        panic!("Qwen3.8 config must select the Qwen3.5-family hybrid decoder");
    };
    assert!(matches!(
        model.model.layers[0]
            .linear_attn
            .as_ref()
            .expect("Qwen3.8 layer zero must be DeltaNet")
            .in_proj_qkv
            .weight
            .val()
            .dtype(),
        DType::QFloat(_)
    ));
    assert_eq!(
        model
            .mtp
            .as_ref()
            .expect("the acceptance load must instantiate MTP")
            .fc
            .weight
            .val()
            .dtype(),
        DType::F16,
        "MTP parameters must remain outside the packed v2 sidecar"
    );

    let token_ids = vec![9_419_i64, 1_814, 11, 19_954, 45_543, 13];
    let sequence = token_ids.len();
    let input = CausalLmInput {
        token_ids: Tensor::<Metal, 2, Int>::from_data(
            TensorData::new(token_ids, [1, sequence]),
            &device,
        ),
        token_mask: None,
        position_ids: None,
    };
    let stateless = loaded
        .model()
        .forward(
            input.clone(),
            &ForwardOptions {
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("run stateless Qwen3.8 {label} inference: {error}"));
    assert_eq!(stateless.logits.dims(), [1, 1, 248_320]);
    let stateless_logits =
        metal_tensor_values("quantized Qwen3.8 stateless logits", &stateless.logits);
    assert!(stateless_logits.iter().all(|value| value.is_finite()));
    let stateless_top = stateless
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("quantized Qwen3.8 top-1 IDs must use i32 storage");

    let mut state = loaded
        .model()
        .new_decode_state(
            CacheConfig::new(1, sequence + 1).expect("quantized Qwen3.8 smoke cache"),
            1,
            &device,
        )
        .expect("quantized Qwen3.8 decode state must initialize");
    let cached = loaded
        .model()
        .prefill(input, &mut state, &ForwardOptions::default())
        .unwrap_or_else(|error| panic!("run cached Qwen3.8 {label} prefill: {error}"));
    let cached_logits = metal_tensor_values("quantized Qwen3.8 cached logits", &cached.logits);
    assert!(cached_logits.iter().all(|value| value.is_finite()));
    let cached_cosine = cosine_similarity(&stateless_logits, &cached_logits);
    assert!(
        cached_cosine >= 0.999,
        "Qwen3.8 {label} cached/stateless cosine {cached_cosine} is below the gate"
    );
    assert_eq!(
        cached
            .logits
            .argmax(2)
            .to_data()
            .to_vec::<i32>()
            .expect("cached quantized Qwen3.8 top-1 IDs must use i32 storage"),
        stateless_top
    );

    let next_token = i64::from(stateless_top[0]);
    let target_hidden = stateless
        .final_hidden_state
        .expect("quantized MTP smoke requires the base final hidden state")
        .narrow(1, sequence - 1, 1);
    let mtp = match loaded.model() {
        AutoCausalLm::Qwen3_5(model) => model
            .forward_mtp(
                Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(vec![next_token], [1, 1]),
                    &device,
                ),
                target_hidden,
                Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(vec![i64::try_from(sequence - 1).unwrap()], [1, 1]),
                    &device,
                ),
                0,
            )
            .unwrap_or_else(|error| panic!("run Qwen3.8 {label} MTP inference: {error}")),
        model => panic!("expected Qwen3.5-family auto dispatch, found {model:?}"),
    };
    assert_eq!(mtp.logits.dims(), [1, 1, 248_320]);
    assert!(
        metal_tensor_values("quantized Qwen3.8 MTP logits", &mtp.logits)
            .iter()
            .all(|value| value.is_finite())
    );

    let decoded = loaded
        .model()
        .decode(
            DecodeInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(vec![next_token], [1, 1]),
                    &device,
                ),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_or_else(|error| panic!("run Qwen3.8 {label} recurrent decode: {error}"));
    assert_eq!(decoded.logits.dims(), [1, 1, 248_320]);
    assert!(
        metal_tensor_values("quantized Qwen3.8 decoded logits", &decoded.logits)
            .iter()
            .all(|value| value.is_finite())
    );
    assert_eq!(state.lengths(), &[sequence + 1]);
    eprintln!(
        "Qwen3.8 {label} Metal: tensors={quantized_count}, top1={next_token}, cached_cosine={cached_cosine:.9}"
    );
}

/// Parses the official nested multimodal config and exercises text-only tokenization.
#[test]
#[ignore = "requires a pinned external Qwen3.5 artifact directory"]
fn validates_pinned_qwen35_local_artifacts() {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR")
            .expect("set TRANSFORMERS_QWEN35_DIR to the pinned local checkout"),
    );
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("official nested text configuration must validate");
    assert_eq!(config.model_type, "qwen3_5_text");
    assert_eq!(config.vocab_size, 248_320);

    let manifest = CheckpointManifest::discover(&root, &LoadOptions::default())
        .expect("official SafeTensors index and header must pass strict discovery");
    assert_eq!(manifest.tensors.len(), 488);
    assert_eq!(manifest.tensor_bytes, 1_746_882_752);
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.starts_with("model.language_model."))
            .count(),
        320
    );
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.starts_with("model.visual."))
            .count(),
        153
    );
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.starts_with("mtp."))
            .count(),
        15
    );

    let tokenizer = LocalTokenizer::from_directory(&root)
        .expect("official tokenizer artifacts must load locally");
    let prompt = "Explain why cached decoding is faster in one sentence.";
    let encoding = tokenizer
        .encode(
            prompt,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("official tokenizer must encode text");
    assert!(!encoding.token_ids()[0].is_empty());

    #[cfg(feature = "chat-templates")]
    {
        let rendered = tokenizer
            .render_chat(
                &[ChatMessage::text("user", prompt)],
                &ChatTemplateOptions {
                    add_generation_prompt: true,
                    ..ChatTemplateOptions::default()
                },
            )
            .expect("official text-only chat template must render");
        assert!(rendered.contains("<|im_start|>user"));
        assert!(rendered.ends_with("<think>\n\n</think>\n\n"));
    }
}

/// Converts every eligible mapped Qwen tensor to Q8 and Q4 official Burnpacks.
#[test]
#[ignore = "requires a pinned external Qwen3.5 artifact directory and temporary artifact space"]
fn quantizes_pinned_qwen35_weights_to_burnpack() {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR")
            .expect("set TRANSFORMERS_QWEN35_DIR to the pinned local checkout"),
    );
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("official nested text configuration must validate");
    // Persist the exact F16 execution plan used by every Metal replay. A v2
    // sidecar intentionally cannot be reused with a different BF16 target plan.
    let load_options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        ..LoadOptions::default()
    };
    let manifest = CheckpointManifest::discover(&root, &load_options)
        .expect("official SafeTensors headers must pass strict discovery");
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions::default())
        .expect("official topology must produce a checkpoint contract");
    let plan = contract
        .build_load_plan(&manifest, &load_options)
        .expect("every language tensor must map strictly");
    let requested_output_root =
        std::env::var_os("TRANSFORMERS_QUANTIZED_OUTPUT_DIR").map(PathBuf::from);
    let temporary_outputs = requested_output_root.is_none().then(|| {
        tempfile::tempdir_in(
            root.parent()
                .expect("pinned checkpoint must have a parent directory"),
        )
        .expect("create temporary quantized artifacts beside model storage")
    });
    let output_root = match &requested_output_root {
        Some(path) => {
            fs::create_dir_all(path).unwrap_or_else(|error| {
                panic!(
                    "create persistent quantized output {}: {error}",
                    path.display()
                )
            });
            path.clone()
        }
        None => temporary_outputs
            .as_ref()
            .expect("temporary output exists without an explicit path")
            .path()
            .to_path_buf(),
    };

    let schemes = match std::env::var("TRANSFORMERS_QUANT_SCHEME").as_deref() {
        Ok("q8") => vec![SymmetricQuantization::Q8],
        Ok("q4") => vec![SymmetricQuantization::Q4],
        Ok(other) => panic!("unsupported TRANSFORMERS_QUANT_SCHEME `{other}`; use q8 or q4"),
        Err(std::env::VarError::NotPresent) => {
            vec![SymmetricQuantization::Q8, SymmetricQuantization::Q4]
        }
        Err(error) => panic!("invalid TRANSFORMERS_QUANT_SCHEME: {error}"),
    };

    let mut summaries = Vec::new();
    for scheme in schemes {
        let label = match scheme {
            SymmetricQuantization::Q8 => "q8",
            SymmetricQuantization::Q4 => "q4",
        };
        let output = output_root.join(format!("qwen3.5-0.8b-{label}.bpk"));
        let outcome = quantize_local_checkpoint_to_burnpack(
            &manifest,
            &plan,
            &output,
            QuantizedCheckpointOptions {
                weights: WeightQuantizationConfig {
                    scheme,
                    ..WeightQuantizationConfig::default()
                },
                ..QuantizedCheckpointOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("{label} checkpoint conversion failed: {error}"));
        assert!(outcome.report.is_strictly_valid());
        assert!(!outcome.report.quantized.is_empty());
        assert_eq!(outcome.report.dtype_converted.len(), 320);
        assert_eq!(
            outcome.report.quantized.len(),
            outcome.artifact.tensor_count
        );
        assert_eq!(outcome.execution.source.tensor_reads, 320);
        assert_eq!(outcome.execution.source.max_open_shards, 1);
        assert_eq!(outcome.staging.tensor_count, outcome.artifact.tensor_count);
        assert!(
            outcome
                .report
                .quantized
                .windows(2)
                .all(|pair| pair[0].target < pair[1].target),
            "quantization report must be deterministic and deduplicated"
        );

        let mut artifact = QuantizedBurnpackArtifact::open(&output)
            .unwrap_or_else(|error| panic!("open {label} Burnpack: {error}"));
        let validation = artifact
            .validate_all()
            .unwrap_or_else(|error| panic!("validate {label} Burnpack: {error}"));
        assert_eq!(validation.tensor_count, outcome.artifact.tensor_count);
        assert_eq!(
            validation.payload_bytes,
            outcome.artifact.quantized_payload_bytes
        );
        assert_eq!(artifact.manifest().tensors.len(), validation.tensor_count);
        assert!(
            artifact.manifest().packed_execution.is_available(),
            "{label} writer must certify the native Burn QFloat layout"
        );
        assert!(
            artifact
                .manifest()
                .tensors
                .values()
                .all(|tensor| tensor.source_dtype == SourceFloatDType::F16)
        );

        #[cfg(all(feature = "metal", not(feature = "rocm")))]
        {
            let device = Default::default();
            let loaded = AutoCausalLm::<Metal>::load_local(
                &root,
                &AutoLoadOptions {
                    checkpoint: LoadOptions {
                        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                        ..LoadOptions::default()
                    },
                    load_tokenizer: false,
                    quantized_burnpack: Some(output.clone()),
                    repository: Some("Qwen/Qwen3.5-0.8B".into()),
                    revision: Some("2fc06364715b967f1860aea9cf38778875588b17".into()),
                    ..AutoLoadOptions::default()
                },
                &device,
            )
            .unwrap_or_else(|error| panic!("load mixed-precision {label} model: {error}"));
            assert_eq!(
                loaded.load_report().quantized.len(),
                outcome.artifact.tensor_count
            );
            let AutoCausalLm::Qwen3_5(model) = loaded.model() else {
                panic!("Qwen3.5 config must select the hybrid decoder variant");
            };
            let first_packed_dtype = model.model.layers[0]
                .linear_attn
                .as_ref()
                .expect("official first layer is DeltaNet")
                .in_proj_qkv
                .weight
                .val()
                .dtype();
            assert!(
                matches!(first_packed_dtype, DType::QFloat(_)),
                "{label} parameter must remain QFloat after installation"
            );

            let input = CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(vec![9_419_i64], [1, 1]),
                    &device,
                ),
                token_mask: None,
                position_ids: None,
            };
            let stateless = loaded
                .model()
                .forward(
                    input.clone(),
                    &ForwardOptions {
                        hidden_states: true,
                        ..ForwardOptions::default()
                    },
                )
                .unwrap_or_else(|error| panic!("run stateless {label} inference: {error}"));
            assert_eq!(stateless.logits.dims(), [1, 1, 248_320]);
            for (index, hidden) in stateless.hidden_states.iter().enumerate() {
                let values = hidden
                    .clone()
                    .cast(DType::F32)
                    .to_data()
                    .to_vec::<f32>()
                    .expect("quantized hidden state must materialize as f32");
                assert!(
                    values.iter().all(|value| value.is_finite()),
                    "{label} hidden-state checkpoint {index} is the first observed non-finite boundary"
                );
            }
            let stateless_logits = stateless
                .logits
                .clone()
                .cast(DType::F32)
                .to_data()
                .to_vec::<f32>()
                .expect("quantized Metal logits must materialize as f32");
            assert!(stateless_logits.iter().all(|value| value.is_finite()));
            let stateless_top = stateless
                .logits
                .argmax(2)
                .to_data()
                .to_vec::<i32>()
                .expect("quantized top-1 IDs must use i32 storage");

            let mut state = loaded
                .model()
                .new_decode_state(
                    CacheConfig::new(1, 2).expect("quantized smoke cache"),
                    1,
                    &device,
                )
                .expect("quantized model cache must initialize");
            let cached = loaded
                .model()
                .prefill(input, &mut state, &ForwardOptions::default())
                .unwrap_or_else(|error| panic!("run cached {label} prefill: {error}"));
            let cached_logits = cached
                .logits
                .clone()
                .cast(DType::F32)
                .to_data()
                .to_vec::<f32>()
                .expect("cached quantized Metal logits must materialize as f32");
            let cached_top = cached
                .logits
                .argmax(2)
                .to_data()
                .to_vec::<i32>()
                .expect("cached quantized top-1 IDs must use i32 storage");
            assert_eq!(cached_top, stateless_top);
            let cached_cosine = cosine_similarity(&stateless_logits, &cached_logits);
            assert!(
                cached_cosine >= 0.999,
                "{label} cached/stateless cosine {cached_cosine} is below the gate"
            );
            eprintln!(
                "{label}: packed_dtype={first_packed_dtype:?}, top1={}, cached_cosine={cached_cosine:.9}",
                stateless_top[0]
            );
        }
        eprintln!(
            "{label}: tensors={}, excluded={}, payload_bytes={}, artifact_bytes={}, largest_tensor_bytes={}",
            outcome.artifact.tensor_count,
            outcome.excluded.len(),
            outcome.artifact.quantized_payload_bytes,
            outcome.artifact.artifact_bytes,
            outcome.artifact.largest_materialized_tensor_bytes,
        );
        summaries.push((
            outcome.artifact.quantized_payload_bytes,
            outcome.artifact.artifact_bytes,
            outcome.report.quantized,
            outcome.excluded,
        ));
    }

    if let [q8, q4] = summaries.as_slice() {
        assert!(q4.0 < q8.0);
        assert!(q4.1 < q8.1);
        assert_eq!(
            q4.2.iter()
                .map(|record| record.target.as_str())
                .collect::<Vec<_>>(),
            q8.2.iter()
                .map(|record| record.target.as_str())
                .collect::<Vec<_>>()
        );
        assert_eq!(q4.3, q8.3);
    } else {
        assert_eq!(summaries.len(), 1, "a filtered run must execute one scheme");
    }
}

/// Replays the first real Qwen DeltaNet layer one packed projection at a time.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires pinned Qwen3.5 weights, a retained Burnpack, and Metal"]
fn diagnoses_pinned_qwen35_packed_delta_layer_on_metal() {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR")
            .expect("set TRANSFORMERS_QWEN35_DIR to the pinned local checkout"),
    );
    let burnpack = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_BURNPACK")
            .expect("set TRANSFORMERS_QWEN35_BURNPACK to a retained Q4/Q8 sidecar"),
    );
    let mut artifact =
        QuantizedBurnpackArtifact::open(&burnpack).expect("retained diagnostic Burnpack must open");
    let device = Default::default();
    let mut loaded = AutoCausalLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            load_tokenizer: false,
            quantized_burnpack: Some(burnpack.clone()),
            ..AutoLoadOptions::default()
        },
        &device,
    )
    .expect("retained packed model must load");
    let AutoCausalLm::Qwen3_5(model) = loaded.model_mut() else {
        panic!("Qwen config must select the hybrid decoder");
    };
    let token_ids =
        Tensor::<Metal, 2, Int>::from_data(TensorData::new(vec![9_419_i64], [1, 1]), &device);
    let embedding = model.model.embed_tokens.forward(token_ids);
    assert_metal_tensor_finite("embedding", &embedding);
    let layer = &mut model.model.layers[0];
    let normalized = layer.input_layernorm.forward(embedding.clone());
    assert_metal_tensor_finite("input norm", &normalized);
    let delta = layer
        .linear_attn
        .as_mut()
        .expect("official first layer is DeltaNet");
    for (name, target, projection) in [
        (
            "in_proj_qkv",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            &delta.in_proj_qkv,
        ),
        (
            "in_proj_z",
            "model.layers.0.linear_attn.in_proj_z.weight",
            &delta.in_proj_z,
        ),
        (
            "in_proj_b",
            "model.layers.0.linear_attn.in_proj_b.weight",
            &delta.in_proj_b,
        ),
        (
            "in_proj_a",
            "model.layers.0.linear_attn.in_proj_a.weight",
            &delta.in_proj_a,
        ),
    ] {
        let output = operations::linear(normalized.clone(), projection)
            .unwrap_or_else(|error| panic!("{name} execution failed: {error}"));
        assert_metal_tensor_finite(name, &output);
        let reference_weight = Tensor::<Metal, 2>::from_data(
            artifact
                .read_weight(target)
                .unwrap_or_else(|error| panic!("read {target}: {error}"))
                .dequantize_f32()
                .unwrap_or_else(|error| panic!("dequantize {target}: {error}")),
            &device,
        )
        .transpose()
        .cast(normalized.dtype());
        let reference_projection = Linear {
            weight: Param::from_tensor(reference_weight),
            bias: None,
        };
        let reference = operations::linear(normalized.clone(), &reference_projection)
            .unwrap_or_else(|error| panic!("{name} dequantized execution failed: {error}"));
        let actual_values = metal_tensor_values(name, &output);
        let reference_values = metal_tensor_values("dequantized projection", &reference);
        let cosine = cosine_similarity(&actual_values, &reference_values);
        let maximum = actual_values
            .iter()
            .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
        eprintln!("{name}: packed/dequantized cosine={cosine:.9}, max_abs={maximum}");
        assert!(
            cosine >= 0.999,
            "{name} packed/dequantized cosine {cosine} is below the dispatch gate"
        );
    }
    let out_probe = operations::linear(
        Tensor::<Metal, 3>::ones([1, 1, delta.value_heads * delta.value_head_dim], &device)
            .cast(DType::F16),
        &delta.out_proj,
    )
    .expect("standalone packed output projection must execute");
    assert_metal_tensor_finite("standalone out_proj", &out_probe);
    let packed_out = delta.out_proj.weight.val();
    assert!(matches!(packed_out.dtype(), DType::QFloat(_)));
    let delta_config = delta.config();
    let mut recurrent_state =
        GatedDeltaNetState::zeros(&delta_config, 1, &device).expect("recurrent diagnostic state");
    let recurrent_mixed = delta
        .forward_with_state(
            normalized.clone(),
            None,
            &mut recurrent_state,
            DeltaNetMode::Recurrent,
        )
        .expect("packed recurrent DeltaNet must execute");
    assert_metal_tensor_finite("recurrent DeltaNet", &recurrent_mixed);
    let mut chunked_state =
        GatedDeltaNetState::zeros(&delta_config, 1, &device).expect("chunked diagnostic state");
    let mixed = delta
        .forward_with_state(
            normalized,
            None,
            &mut chunked_state,
            DeltaNetMode::ChunkedPrefill,
        )
        .expect("packed chunked DeltaNet must execute");
    assert_metal_tensor_finite("chunked DeltaNet", &mixed);
    let residual = embedding + mixed;
    assert_metal_tensor_finite("post-DeltaNet residual", &residual);
    let mlp = layer
        .mlp
        .forward(layer.post_attention_layernorm.forward(residual.clone()))
        .expect("packed layer MLP must execute");
    assert_metal_tensor_finite("layer MLP", &mlp);
    assert_metal_tensor_finite("complete layer", &(residual + mlp));
}

/// Loads a retained Q4 sidecar and verifies complete stateless/cached Metal inference.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires pinned Qwen3.5 weights, a retained Q4 Burnpack, and Metal"]
fn loads_pinned_qwen35_q4_burnpack_and_runs_metal() {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR")
            .expect("set TRANSFORMERS_QWEN35_DIR to the pinned local checkout"),
    );
    let burnpack = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_BURNPACK")
            .expect("set TRANSFORMERS_QWEN35_BURNPACK to the retained Q4 sidecar"),
    );
    let artifact =
        QuantizedBurnpackArtifact::open(&burnpack).expect("retained Q4 Burnpack must open");
    let quantized_count = artifact.manifest().tensors.len();
    assert!(
        artifact
            .manifest()
            .tensors
            .values()
            .all(|tensor| tensor.scheme == SymmetricQuantization::Q4),
        "the Q4 acceptance path must not silently consume a different precision"
    );
    drop(artifact);

    let device = Default::default();
    let loaded = AutoCausalLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            load_tokenizer: false,
            quantized_burnpack: Some(burnpack),
            repository: Some("Qwen/Qwen3.5-0.8B".into()),
            revision: Some("2fc06364715b967f1860aea9cf38778875588b17".into()),
            ..AutoLoadOptions::default()
        },
        &device,
    )
    .expect("mixed F16/Q4 model must load transactionally");
    assert_eq!(loaded.load_report().quantized.len(), quantized_count);

    let input = CausalLmInput {
        token_ids: Tensor::<Metal, 2, Int>::from_data(
            TensorData::new(vec![9_419_i64], [1, 1]),
            &device,
        ),
        token_mask: None,
        position_ids: None,
    };
    let stateless = loaded
        .model()
        .forward(input.clone(), &ForwardOptions::default())
        .expect("Q4 stateless inference must execute");
    let stateless_values = stateless
        .logits
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("Q4 logits must materialize as f32");
    assert!(stateless_values.iter().all(|value| value.is_finite()));
    let stateless_top = stateless
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("Q4 top-1 IDs must use i32 storage");

    let mut state = loaded
        .model()
        .new_decode_state(CacheConfig::new(1, 2).expect("Q4 smoke cache"), 1, &device)
        .expect("Q4 decode state must initialize");
    let cached = loaded
        .model()
        .prefill(input, &mut state, &ForwardOptions::default())
        .expect("Q4 cached prefill must execute");
    let cached_values = cached
        .logits
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("cached Q4 logits must materialize as f32");
    let cached_top = cached
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("cached Q4 top-1 IDs must use i32 storage");
    let cached_cosine = cosine_similarity(&stateless_values, &cached_values);
    assert_eq!(cached_top, stateless_top);
    assert!(cached_cosine >= 0.999);
    eprintln!(
        "q4: tensors={quantized_count}, top1={}, cached_cosine={cached_cosine:.9}",
        stateless_top[0]
    );
}

/// Synchronizes one Metal tensor and rejects the first non-finite component result.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn assert_metal_tensor_finite<const D: usize>(label: &str, tensor: &Tensor<Metal, D>) {
    let values = metal_tensor_values(label, tensor);
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} produced non-finite values"
    );
}

/// Returns synchronized F32 values for quantitative Metal diagnostics.
#[cfg(feature = "metal")]
fn metal_tensor_values<const D: usize>(label: &str, tensor: &Tensor<Metal, D>) -> Vec<f32> {
    tensor
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .unwrap_or_else(|error| panic!("materialize {label}: {error}"))
}

/// Streams the pinned language weights into Metal F16 and executes one prompt.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a pinned external Qwen3.5 artifact directory and Metal GPU"]
fn loads_pinned_qwen35_weights_and_runs_metal_inference() {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR")
            .expect("set TRANSFORMERS_QWEN35_DIR to the pinned local checkout"),
    );
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("official nested text configuration must validate");
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        ..LoadOptions::default()
    };
    let manifest = CheckpointManifest::discover(&root, &options)
        .expect("official SafeTensors headers must pass strict discovery");
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions::default())
        .expect("official topology must produce a checkpoint contract");
    let plan = contract
        .build_load_plan(&manifest, &options)
        .expect("every language tensor must map strictly");
    assert_eq!(plan.tensors.len(), 320);
    assert_eq!(plan.aliases.len(), 1);
    assert_eq!(plan.report.allowlisted_ignored.len(), 168);

    let device = Default::default();
    let mut model = Qwen3_5ForCausalLm::<Metal>::init(config, &device)
        .expect("official Qwen graph must initialize lazily");
    let outcome = execute_local_load_plan(&manifest, &plan, &mut model)
        .unwrap_or_else(|error| panic!("streamed model load failed: {error}"));
    assert!(outcome.report.is_strictly_valid());
    assert_eq!(outcome.report.applied.len(), 321);
    assert_eq!(outcome.stats.tensors_applied, 320);
    assert_eq!(outcome.stats.aliases_applied, 1);

    let tokenizer = LocalTokenizer::from_directory(&root)
        .expect("official tokenizer artifacts must load locally");
    let encoding = tokenizer
        .encode(
            "Hello",
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("prompt must tokenize");
    let tokens = encoding.token_ids()[0]
        .iter()
        .map(|token| i64::from(*token))
        .collect::<Vec<_>>();
    let sequence = tokens.len();
    let input_ids =
        Tensor::<Metal, 2, Int>::from_data(TensorData::new(tokens.clone(), [1, sequence]), &device);
    let output = model
        .forward(
            CausalLmInput {
                token_ids: input_ids.clone(),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect("loaded model must execute inference");
    assert_eq!(output.logits.dims(), [1, 1, 248_320]);
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/qwen3_5_checkpoint/qwen3_5_0_8b_reference.json"
    ))
    .expect("committed reference logits must be valid JSON");
    let expected_top_ids = reference["top_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let (_, top_ids) = output.logits.clone().topk_with_indices(10, 2);
    let top_ids = top_ids
        .to_data()
        .to_vec::<i32>()
        .expect("Metal top-k must return i32 token IDs");
    assert_eq!(top_ids, expected_top_ids);

    let sample_indices = reference["sample_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap())
        .collect::<Vec<_>>();
    let sample_count = sample_indices.len();
    let sample_indices = Tensor::<Metal, 1, Int>::from_data(
        TensorData::new(sample_indices, [sample_count]),
        &device,
    );
    let sample_logits = output
        .logits
        .clone()
        .select(2, sample_indices)
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("selected Metal logits must convert to f32");
    let expected_sample_logits = reference["sample_logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    for (actual, expected) in sample_logits.iter().zip(expected_sample_logits) {
        assert!(
            (actual - expected).abs() <= 0.25,
            "reference logit mismatch: expected {expected}, found {actual}"
        );
    }

    if let Some(reference_path) = std::env::var_os("TRANSFORMERS_QWEN35_REFERENCE_LOGITS") {
        let reference_path = PathBuf::from(reference_path);
        let bytes = fs::read(&reference_path).unwrap_or_else(|error| {
            panic!(
                "failed to read raw little-endian f32 reference logits from {}: {error}",
                reference_path.display()
            )
        });
        let expected_bytes = 248_320_usize
            .checked_mul(size_of::<f32>())
            .expect("the fixed reference vocabulary byte length must fit usize");
        assert_eq!(
            bytes.len(),
            expected_bytes,
            "reference logit file must contain exactly one little-endian f32 vocabulary vector"
        );
        let reference_logits = bytes
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let actual_logits = output
            .logits
            .clone()
            .cast(DType::F32)
            .to_data()
            .to_vec::<f32>()
            .expect("Metal logits must convert to an f32 host vector");
        assert_eq!(actual_logits.len(), reference_logits.len());
        let cosine = cosine_similarity(&actual_logits, &reference_logits);
        eprintln!("full-vocabulary reference-logit cosine similarity: {cosine:.9}");
        assert!(
            cosine >= 0.999,
            "full-vocabulary cosine similarity {cosine} is below the 0.999 acceptance gate"
        );
    }

    let top_token = top_ids
        .first()
        .copied()
        .expect("top-k returns one result for a non-empty vocabulary");
    assert!((0..248_320).contains(&(top_token as usize)));

    let expected_greedy = reference["greedy_token_ids"]
        .as_array()
        .expect("reference must contain a greedy continuation")
        .iter()
        .map(|value| value.as_i64().expect("greedy token IDs must be integers") as i32)
        .collect::<Vec<_>>();
    assert_eq!(
        expected_greedy.len(),
        reference["greedy_max_new_tokens"]
            .as_u64()
            .expect("reference generation length must be an integer") as usize
    );
    let cache_config = CacheConfig::new(1, sequence + expected_greedy.len())
        .expect("reference generation cache geometry must be valid");
    let mut state = model
        .new_decode_state(cache_config, 1, &device)
        .expect("official Qwen cache geometry must initialize");
    let mut cached = model
        .prefill(
            CausalLmInput {
                token_ids: input_ids.clone(),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("cached prompt prefill must execute");
    let prompt_stateless = output
        .logits
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("stateless prompt logits must convert to f32");
    let prompt_cached = cached
        .logits
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("cached prompt logits must convert to f32");
    let prompt_cosine = cosine_similarity(&prompt_stateless, &prompt_cached);
    assert!(
        prompt_cosine >= 0.999,
        "cached prefill cosine similarity {prompt_cosine} is below the 0.999 gate"
    );

    let mut generated = Vec::with_capacity(expected_greedy.len());
    for (step, expected) in expected_greedy.iter().copied().enumerate() {
        let selected = cached
            .logits
            .clone()
            .argmax(2)
            .to_data()
            .to_vec::<i32>()
            .expect("cached greedy IDs must use i32 storage");
        let token = *selected
            .first()
            .expect("one cached logit row must select one token");
        assert_eq!(
            token, expected,
            "cached continuation differs from the official reference at step {step}"
        );
        generated.push(token);
        if step + 1 < expected_greedy.len() {
            cached = model
                .decode(
                    DecodeInput {
                        token_ids: Tensor::<Metal, 2, Int>::from_data(
                            TensorData::new(vec![i64::from(token)], [1, 1]),
                            &device,
                        ),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap_or_else(|error| panic!("cached decode step {step} failed: {error}"));
        }
    }
    assert_eq!(generated, expected_greedy);
    assert_eq!(state.lengths(), &[sequence + expected_greedy.len() - 1]);

    let mut teacher_tokens = tokens;
    teacher_tokens.extend(
        expected_greedy
            .iter()
            .take(expected_greedy.len() - 1)
            .map(|token| i64::from(*token)),
    );
    let teacher_sequence = teacher_tokens.len();
    let teacher_output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(teacher_tokens, [1, teacher_sequence]),
                    &device,
                ),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .expect("teacher-forced stateless forward must execute");
    let stateless_greedy = teacher_output
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("stateless greedy IDs must use i32 storage");
    assert_eq!(
        stateless_greedy, expected_greedy,
        "cached and stateless top-1 continuations must match at every step"
    );
}

/// Computes cosine similarity with f64 accumulation for acceptance diagnostics.
#[cfg(feature = "metal")]
fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len());
    let (dot, left_square, right_square) =
        left.iter()
            .zip(right)
            .fold((0.0_f64, 0.0_f64, 0.0_f64), |acc, (left, right)| {
                let left = f64::from(*left);
                let right = f64::from(*right);
                (
                    acc.0 + left * right,
                    acc.1 + left * left,
                    acc.2 + right * right,
                )
            });
    dot / (left_square.sqrt() * right_square.sqrt())
}
