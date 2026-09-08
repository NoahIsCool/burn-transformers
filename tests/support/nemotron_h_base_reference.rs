//! Strict reader for the pinned official Nemotron-H base-model oracle.
//!
//! This reader deliberately covers only the Hugging Face Transformers base
//! graph. The checkpoint's 270 physical `mtp.*` tensors are authenticated and
//! proven to be the exact namespace filtered by the official base loader, but
//! no MTP output is accepted here. A separately source-pinned vLLM oracle owns
//! that future evidence.

#![allow(dead_code)]

use super::{
    CHECKPOINT_ARTIFACT_MANIFEST_SHA256, CONTROL_FILES, HEADER_INVENTORY_SHA256,
    INDEX_REPORTED_BASE_PARAMETERS, INDEX_REPORTED_TOTAL_SIZE, INDEX_TOTAL_SIZE_DELTA,
    MTP_TENSOR_COUNT, PHYSICAL_TENSOR_BYTES, PHYSICAL_TENSOR_COUNT, REPOSITORY, REVISION, SHARDS,
    SMOKE_PROMPT, SMOKE_TOKEN_IDS,
};
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, Read},
    path::{Component, Path, PathBuf},
};

/// Environment variable selecting the external immutable bundle.
const REFERENCE_DIR_ENV: &str = "TRANSFORMERS_NEMOTRON_H_REFERENCE_DIR";

/// Exact create-only directory emitted by the reviewed generator.
const BUNDLE_DIRNAME: &str = "nemotron-3.5-lightning-30b-a3b-a9904d24-reference.bundle";

/// Exact official source revision used for every base-model output.
const TRANSFORMERS_REVISION: &str = "cfa309cfa34b5497a32fd50af49aa97aac94c054";

/// Exact reviewed generator identity.
const GENERATOR_GIT_HEAD: &str = "0f222f6213ebf320a5fd91221021d5499622987d";

/// Exact reviewed generator tree identity.
const GENERATOR_GIT_TREE: &str = "b77f32d3e6a621829dc78feecb173265a8e0a42f";

/// Exact reviewed generator source identity.
const GENERATOR_SHA256: &str = "10ab5c7a6aea744ba28120a2b5415ff91c2260e9f01da69c583a1cc4f060b94a";

/// Canonical empty-list diagnostic digest.
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Exact physical MTP-name inventory, authenticated but never executed here.
const MTP_NAME_SHA256: &str = "9dceb828e3bb078ebb0dfe24b1e7257ba6bb2f9c49c0be14539caad770078e48";

/// Canonical primary-distribution payload-manifest digest.
const DISTRIBUTION_MANIFEST_SHA256: &str =
    "6eca7252cebb0750fa344a7d05419a4511f3ecf985e6f893dadf3e0e94c2b2a6";

/// Exact binary16-stable official continuation.
const GENERATED_TOKEN_IDS: [u32; 32] = [
    1_256, 1_010, 1_256, 1_010, 1_256, 1_010, 1_256, 1_010, 1_256, 1_010, 1_256, 1_010, 1_256,
    1_010, 1_256, 1_010, 1_256, 1_010, 1_256, 1_010, 1_256, 1_010, 1_256, 1_010, 1_256, 1_010,
    1_256, 1_010, 1_256, 1_010, 1_256, 1_010,
];

/// One independently reviewed file anchor.
#[derive(Debug, Clone, Copy)]
struct FileAnchor {
    /// Bundle-relative filename.
    name: &'static str,
    /// Exact byte length.
    bytes: u64,
    /// Independently computed SHA-256.
    sha256: &'static str,
}

/// All seven reviewed output identities, independent of claims inside JSON.
const FILES: [FileAnchor; 7] = [
    FileAnchor {
        name: "nemotron-3.5-lightning-30b-a3b-a9904d24-reference.json",
        bytes: 98_220,
        sha256: "bd3abb417f2fee82d1bef8bb654ca846fa4816351b1d9d7d0d057105f92de196",
    },
    FileAnchor {
        name: "nemotron-3.5-lightning-30b-a3b-a9904d24-prompt-logits.f32le",
        bytes: 524_288,
        sha256: "f3c014bb043e9ffa3a114a4296ca0e0c2165c4abd63b942df6d910a4383a1dff",
    },
    FileAnchor {
        name: "nemotron-3.5-lightning-30b-a3b-a9904d24-hidden-last.f32le",
        bytes: 580_608,
        sha256: "56033b284b0c926a40fd50e51eeb232412d3d77ae97282782dc39b9645ff0be2",
    },
    FileAnchor {
        name: "nemotron-3.5-lightning-30b-a3b-a9904d24-cached-prefix-logits.f32le",
        bytes: 16_777_216,
        sha256: "b3a1ae948e55e6db409e0f9973c41c8767c874c6d64b1d351cfff7bbafa72480",
    },
    FileAnchor {
        name: "nemotron-3.5-lightning-30b-a3b-a9904d24-teacher-prefix-logits.f32le",
        bytes: 16_777_216,
        sha256: "c1878c5e1396ddf01f3dfa85729ea58aa879ece28507399bdaf7addf4a65601b",
    },
    FileAnchor {
        name: "nemotron-3.5-lightning-30b-a3b-a9904d24-cached-prefill-hidden-last.f32le",
        bytes: 580_608,
        sha256: "56033b284b0c926a40fd50e51eeb232412d3d77ae97282782dc39b9645ff0be2",
    },
    FileAnchor {
        name: "nemotron-3.5-lightning-30b-a3b-a9904d24-cached-decode-1-hidden-last.f32le",
        bytes: 580_608,
        sha256: "46c65b5cc91b53c3cdb2c3b6df9acaf4fbb216e91ae09f007c8d1b5571d93b79",
    },
];

/// Public summary returned only after every independent gate succeeds.
pub(super) struct ValidatedBaseReference {
    /// Number of authenticated external outputs, including JSON.
    pub(super) output_count: usize,
    /// Number of literal prompt tokens.
    pub(super) prompt_tokens: usize,
    /// Number of cached and teacher-forced decisions.
    pub(super) generation_tokens: usize,
    /// Lowest observed official cached-recurrent/teacher-chunked cosine.
    pub(super) minimum_official_path_cosine: f64,
    /// Full-vocabulary prompt row for later Burn parity.
    pub(super) prompt_logits: Vec<f32>,
    /// Stateless prompt residual-boundary trace.
    pub(super) stateless_hidden: Vec<f32>,
    /// All official cached decision rows.
    pub(super) cached_logits: Vec<f32>,
    /// All official teacher-forced decision rows.
    pub(super) teacher_logits: Vec<f32>,
    /// Cached-prefill residual-boundary trace.
    pub(super) cached_prefill_hidden: Vec<f32>,
    /// First recurrent-decode residual-boundary trace.
    pub(super) cached_decode_hidden: Vec<f32>,
}

/// Strict top-level schema; nested objects receive explicit closed-key checks.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaseReferenceRecord {
    /// Record schema revision.
    schema_version: u32,
    /// Immutable checkpoint repository.
    repository: String,
    /// Immutable checkpoint revision.
    revision: String,
    /// Pinned source implementation.
    reference: Value,
    /// Literal prompt.
    prompt: String,
    /// Exact prompt token IDs.
    input_ids: Vec<u32>,
    /// Tokenizer semantics.
    tokenizer: Value,
    /// Numerical execution contract.
    execution: Value,
    /// Authenticated runtime provenance.
    environment: Value,
    /// Generator, artifact, loader, and MTP-filter provenance.
    provenance: Value,
    /// External binary descriptors.
    outputs: OutputDescriptors,
    /// Greedy decisions and path-specific parity policy.
    generation: Value,
    /// Non-normative observed timings.
    timings_seconds: Value,
}

/// Exact six numerical descriptors.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputDescriptors {
    /// Stateless prompt vocabulary row.
    prompt_logits: PromptDescriptor,
    /// Stateless prompt hidden trace.
    hidden_last_trace: HiddenDescriptor,
    /// Cached decision rows.
    cached_prefix_logits: MatrixDescriptor,
    /// Teacher-forced decision rows.
    teacher_prefix_logits: MatrixDescriptor,
    /// Cached-prefill hidden trace.
    cached_prefill_hidden_trace: HiddenDescriptor,
    /// First recurrent-decode hidden trace.
    cached_decode_hidden_trace: HiddenDescriptor,
}

/// Full-vocabulary prompt descriptor and deterministic summary.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptDescriptor {
    relative_path: String,
    encoding: String,
    source_dtype: String,
    storage_dtype: String,
    shape: Vec<usize>,
    elements: usize,
    bytes: u64,
    sha256: String,
    finite: bool,
    sum: f64,
    l2_norm: f64,
    top10: Vec<RankedLogit>,
    top1_token_id: u32,
}

/// One ranked vocabulary entry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RankedLogit {
    token_id: u32,
    logit: f32,
}

/// One embedding/block/final-norm trace descriptor.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HiddenDescriptor {
    relative_path: String,
    encoding: String,
    source_dtype: String,
    storage_dtype: String,
    shape: Vec<usize>,
    elements: usize,
    bytes: u64,
    sha256: String,
    finite: bool,
    labels: Vec<String>,
    phase: String,
    token_position: usize,
}

/// One 32-row full-vocabulary matrix descriptor.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MatrixDescriptor {
    relative_path: String,
    encoding: String,
    source_dtype: String,
    storage_dtype: String,
    shape: Vec<usize>,
    elements: usize,
    bytes: u64,
    sha256: String,
    finite: bool,
    row_position_ids: Vec<usize>,
    row_top1_token_ids: Vec<u32>,
}

/// Returns the complete independently reviewed anchor map.
pub(super) fn trusted_reference_hashes() -> BTreeMap<&'static str, &'static str> {
    FILES
        .iter()
        .map(|entry| (entry.name, entry.sha256))
        .collect()
}

/// Reads and validates the complete external base-only oracle bundle.
pub(super) fn load_and_validate_reference_bundle() -> ValidatedBaseReference {
    let supplied = PathBuf::from(std::env::var_os(REFERENCE_DIR_ENV).unwrap_or_else(|| {
        panic!("set {REFERENCE_DIR_ENV} to the reviewed {BUNDLE_DIRNAME} directory")
    }));
    let supplied_metadata = fs::symlink_metadata(&supplied)
        .unwrap_or_else(|error| panic!("inspect {}: {error}", supplied.display()));
    assert!(
        supplied_metadata.is_dir() && !supplied_metadata.file_type().is_symlink(),
        "Nemotron base-reference root must be a non-symlink directory"
    );
    let root = supplied
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied.display()));
    assert_eq!(
        root.file_name().and_then(|name| name.to_str()),
        Some(BUNDLE_DIRNAME)
    );
    assert!(
        !root.starts_with(env!("CARGO_MANIFEST_DIR")),
        "generated numerical references must remain outside the crate"
    );
    let expected_names = FILES
        .iter()
        .map(|entry| entry.name.to_owned())
        .collect::<BTreeSet<_>>();
    let observed_names = fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("enumerate {}: {error}", root.display()))
        .map(|entry| {
            let entry = entry.expect("read Nemotron base-reference entry");
            let kind = entry
                .file_type()
                .expect("inspect base-reference entry type");
            assert!(
                kind.is_file() && !kind.is_symlink(),
                "bundle member {} must be a regular non-symlink file",
                entry.path().display()
            );
            entry
                .file_name()
                .into_string()
                .expect("bundle filenames must be UTF-8")
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(observed_names, expected_names, "bundle entry set changed");

    for anchor in FILES {
        let path = root.join(anchor.name);
        assert_regular_file(&path);
        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            anchor.bytes,
            "{}",
            anchor.name
        );
        assert_eq!(sha256_file_hex(&path), anchor.sha256, "{}", anchor.name);
    }

    let json_anchor = FILES[0];
    let record_bytes = fs::read(root.join(json_anchor.name)).expect("read authenticated JSON");
    let record: BaseReferenceRecord = serde_json::from_slice(&record_bytes)
        .expect("Nemotron base-reference JSON must satisfy its strict top-level schema");
    validate_record(&record);

    let prompt_logits = read_descriptor(
        &root,
        &record.outputs.prompt_logits.relative_path,
        record.outputs.prompt_logits.bytes,
        record.outputs.prompt_logits.elements,
        &record.outputs.prompt_logits.sha256,
    );
    let stateless_hidden = read_descriptor(
        &root,
        &record.outputs.hidden_last_trace.relative_path,
        record.outputs.hidden_last_trace.bytes,
        record.outputs.hidden_last_trace.elements,
        &record.outputs.hidden_last_trace.sha256,
    );
    let cached_logits = read_descriptor(
        &root,
        &record.outputs.cached_prefix_logits.relative_path,
        record.outputs.cached_prefix_logits.bytes,
        record.outputs.cached_prefix_logits.elements,
        &record.outputs.cached_prefix_logits.sha256,
    );
    let teacher_logits = read_descriptor(
        &root,
        &record.outputs.teacher_prefix_logits.relative_path,
        record.outputs.teacher_prefix_logits.bytes,
        record.outputs.teacher_prefix_logits.elements,
        &record.outputs.teacher_prefix_logits.sha256,
    );
    let cached_prefill_hidden = read_descriptor(
        &root,
        &record.outputs.cached_prefill_hidden_trace.relative_path,
        record.outputs.cached_prefill_hidden_trace.bytes,
        record.outputs.cached_prefill_hidden_trace.elements,
        &record.outputs.cached_prefill_hidden_trace.sha256,
    );
    let cached_decode_hidden = read_descriptor(
        &root,
        &record.outputs.cached_decode_hidden_trace.relative_path,
        record.outputs.cached_decode_hidden_trace.bytes,
        record.outputs.cached_decode_hidden_trace.elements,
        &record.outputs.cached_decode_hidden_trace.sha256,
    );
    validate_payload_values(
        &record,
        &prompt_logits,
        &stateless_hidden,
        &cached_logits,
        &teacher_logits,
        &cached_prefill_hidden,
    );

    ValidatedBaseReference {
        output_count: FILES.len(),
        prompt_tokens: record.input_ids.len(),
        generation_tokens: GENERATED_TOKEN_IDS.len(),
        minimum_official_path_cosine: number(
            field(
                object(&record.generation, "generation"),
                "minimum_cached_stateless_prefix_cosine",
                "generation",
            ),
            "generation.minimum_cached_stateless_prefix_cosine",
        ),
        prompt_logits,
        stateless_hidden,
        cached_logits,
        teacher_logits,
        cached_prefill_hidden,
        cached_decode_hidden,
    }
}

/// Validates all fixed metadata without trusting the record's own hashes.
fn validate_record(record: &BaseReferenceRecord) {
    assert_eq!(record.schema_version, 3);
    assert_eq!(record.repository, REPOSITORY);
    assert_eq!(record.revision, REVISION);
    assert_eq!(record.prompt, SMOKE_PROMPT);
    assert_eq!(record.input_ids, SMOKE_TOKEN_IDS);
    validate_reference(&record.reference);
    validate_tokenizer(&record.tokenizer);
    validate_execution(&record.execution);
    validate_environment(&record.environment);
    validate_provenance(&record.provenance);
    validate_outputs(&record.outputs);
    validate_generation(&record.generation);
    let timings = exact_object(
        &record.timings_seconds,
        &[
            "cached_generation",
            "load",
            "stateless_prompt",
            "teacher_forced_stateless",
        ],
        "timings_seconds",
    );
    assert!(timings.values().all(|value| number(value, "timing") >= 0.0));
}

/// Validates exact official source identities and proves MTP was not executed.
fn validate_reference(value: &Value) {
    let record = exact_object(
        value,
        &[
            "checkout_clean",
            "checkpoint_config_transformers_version",
            "class",
            "executes_mtp",
            "repository",
            "revision",
            "sole_unexpected_key_filter",
            "source_version",
            "sources",
        ],
        "reference",
    );
    assert_eq!(
        string(field(record, "repository", "reference"), "repository"),
        "huggingface/transformers"
    );
    assert_eq!(
        string(field(record, "revision", "reference"), "revision"),
        TRANSFORMERS_REVISION
    );
    assert_eq!(
        string(field(record, "class", "reference"), "class"),
        "transformers.NemotronHForCausalLM"
    );
    assert_eq!(
        string(
            field(record, "source_version", "reference"),
            "source_version"
        ),
        "5.17.0.dev0"
    );
    assert_eq!(
        string(
            field(
                record,
                "checkpoint_config_transformers_version",
                "reference"
            ),
            "config version"
        ),
        "4.57.6"
    );
    assert_eq!(
        string(
            field(record, "sole_unexpected_key_filter", "reference"),
            "filter"
        ),
        "mtp.*"
    );
    assert!(boolean(
        field(record, "checkout_clean", "reference"),
        "checkout"
    ));
    assert!(!boolean(
        field(record, "executes_mtp", "reference"),
        "executes_mtp"
    ));
    let sources = array(field(record, "sources", "reference"), "reference.sources");
    assert_eq!(sources.len(), 2);
    let expected = BTreeMap::from([
        (
            "src/transformers/models/nemotron_h/configuration_nemotron_h.py",
            (
                13_031,
                "00a4c60bd302d796b25c6ae25c608a5ee7e27f9a0adea92911b0c45470232052",
            ),
        ),
        (
            "src/transformers/models/nemotron_h/modeling_nemotron_h.py",
            (
                50_797,
                "d42067429e3268d1163579f5d257fe0baa9a2b02edba5fcb175fb8d42c9b6af4",
            ),
        ),
    ]);
    for source in sources {
        let source = exact_object(source, &["bytes", "relative_path", "sha256"], "source");
        let path = string(field(source, "relative_path", "source"), "source path");
        let (bytes, sha256) = expected.get(path).expect("unexpected reference source");
        assert_eq!(
            integer(field(source, "bytes", "source"), "source bytes"),
            *bytes
        );
        assert_eq!(
            string(field(source, "sha256", "source"), "source hash"),
            *sha256
        );
    }
}

/// Validates tokenizer/config distinctions and the retained whitespace loop.
fn validate_tokenizer(value: &Value) {
    let tokenizer = exact_object(
        value,
        &[
            "add_special_tokens",
            "class",
            "decoded_continuation",
            "generation_eos_token_ids",
            "model_config_eos_token_id",
            "model_config_pad_token_id",
            "prompt_token_ids",
            "tokenizer_eos_token_id",
            "tokenizer_pad_token_id",
            "whitespace_cycle_token_ids",
            "whitespace_cycle_token_pieces",
        ],
        "tokenizer",
    );
    assert!(!boolean(
        field(tokenizer, "add_special_tokens", "tokenizer"),
        "specials"
    ));
    assert_eq!(
        string(field(tokenizer, "class", "tokenizer"), "class"),
        "transformers.TokenizersBackend"
    );
    assert_eq!(
        u32_array(
            field(tokenizer, "prompt_token_ids", "tokenizer"),
            "prompt IDs"
        ),
        SMOKE_TOKEN_IDS
    );
    assert_eq!(
        integer(
            field(tokenizer, "model_config_eos_token_id", "tokenizer"),
            "model EOS"
        ),
        2
    );
    assert_eq!(
        integer(
            field(tokenizer, "model_config_pad_token_id", "tokenizer"),
            "model PAD"
        ),
        0
    );
    assert_eq!(
        integer(
            field(tokenizer, "tokenizer_eos_token_id", "tokenizer"),
            "tokenizer EOS"
        ),
        11
    );
    assert_eq!(
        integer(
            field(tokenizer, "tokenizer_pad_token_id", "tokenizer"),
            "tokenizer PAD"
        ),
        11
    );
    assert_eq!(
        u32_array(
            field(tokenizer, "generation_eos_token_ids", "tokenizer"),
            "generation EOS"
        ),
        [2, 11]
    );
    assert_eq!(
        u32_array(
            field(tokenizer, "whitespace_cycle_token_ids", "tokenizer"),
            "cycle IDs"
        ),
        [1_256, 1_010]
    );
    assert_eq!(
        string_array(
            field(tokenizer, "whitespace_cycle_token_pieces", "tokenizer"),
            "cycle pieces"
        ),
        ["ĠĠ", "Ċ"]
    );
    assert_eq!(
        string(
            field(tokenizer, "decoded_continuation", "tokenizer"),
            "decoded continuation"
        ),
        "  \n".repeat(16)
    );
}

/// Validates deterministic eager MPS execution and exact external cache geometry.
fn validate_execution(value: &Value) {
    let execution = exact_object(
        value,
        &[
            "allocator_warmup_override",
            "attention_implementation",
            "attention_precision",
            "authenticated_import_bytecode_policy",
            "authenticated_import_resolutions",
            "cache",
            "deterministic_algorithms",
            "device",
            "dtype",
            "experts_implementation",
            "explicit_all_one_attention_masks",
            "float32_matmul_precision",
            "hub_kernels_enabled",
            "imported_module_bindings",
            "integrity_threat_model",
            "mamba_implementation",
            "mps_per_process_memory_fraction",
            "seed",
            "use_cache",
        ],
        "execution",
    );
    assert_eq!(
        string(field(execution, "dtype", "execution"), "dtype"),
        "float16"
    );
    assert_eq!(
        string(field(execution, "device", "execution"), "device"),
        "mps"
    );
    assert_eq!(
        string(
            field(execution, "attention_implementation", "execution"),
            "attention"
        ),
        "eager"
    );
    assert_eq!(
        string(
            field(execution, "experts_implementation", "execution"),
            "experts"
        ),
        "eager_selected_expert_dispatch"
    );
    assert!(boolean(
        field(execution, "deterministic_algorithms", "execution"),
        "deterministic"
    ));
    assert!(boolean(
        field(execution, "use_cache", "execution"),
        "use_cache"
    ));
    assert!(!boolean(
        field(execution, "hub_kernels_enabled", "execution"),
        "hub kernels"
    ));
    assert_eq!(integer(field(execution, "seed", "execution"), "seed"), 0);
    validate_cache(field(execution, "cache", "execution"));
}

/// Validates physical cache axes separately from the API's `-1` sentinel.
fn validate_cache(value: &Value) {
    let cache = object(value, "execution.cache");
    assert_eq!(
        string(field(cache, "class", "cache"), "cache class"),
        "transformers.cache_utils.DynamicCache"
    );
    assert_eq!(
        integer(field(cache, "batch_size", "cache"), "physical batch"),
        1
    );
    assert_eq!(
        signed_integer(
            field(cache, "api_reported_batch_size", "cache"),
            "API batch"
        ),
        -1
    );
    assert_eq!(integer(field(cache, "layer_count", "cache"), "layers"), 52);
    assert_eq!(
        integer(field(cache, "mamba_layers", "cache"), "Mamba layers"),
        23
    );
    assert_eq!(
        integer(
            field(cache, "moe_placeholders", "cache"),
            "MoE placeholders"
        ),
        23
    );
    assert_eq!(
        integer(
            field(cache, "full_attention_layers", "cache"),
            "attention layers"
        ),
        6
    );
    assert_eq!(
        integer(field(cache, "key_value_heads", "cache"), "KV heads"),
        2
    );
    assert_eq!(
        integer(field(cache, "head_dimension", "cache"), "head dim"),
        128
    );
    assert_eq!(
        string(
            field(cache, "mamba_convolution_dtype", "cache"),
            "conv dtype"
        ),
        "float16"
    );
    assert_eq!(
        string(
            field(cache, "mamba_recurrent_dtype", "cache"),
            "state dtype"
        ),
        "float32"
    );
    assert_eq!(
        string(field(cache, "attention_tensor_dtype", "cache"), "KV dtype"),
        "float16"
    );
    assert_eq!(
        string(field(cache, "tensor_device", "cache"), "cache device"),
        "mps"
    );
    assert_eq!(
        usize_array(
            field(cache, "decision_logical_lengths", "cache"),
            "logical lengths"
        ),
        (6..=37).collect::<Vec<_>>()
    );
    assert_eq!(
        usize_array(
            field(cache, "decision_physical_lengths", "cache"),
            "physical lengths"
        ),
        (6..=37).collect::<Vec<_>>()
    );
    assert_eq!(
        usize_array(field(cache, "decision_position_ids", "cache"), "positions"),
        (5..=36).collect::<Vec<_>>()
    );
    validate_cache_geometry(field(cache, "first_decision_geometry", "cache"), 6);
    validate_cache_geometry(field(cache, "final_decision_geometry", "cache"), 37);
}

/// Validates one complete first/final cache snapshot.
fn validate_cache_geometry(value: &Value, length: usize) {
    let geometry = exact_object(
        value,
        &[
            "attention_dtype",
            "attention_key_value_shape",
            "attention_layer_indices",
            "class",
            "logical_length",
            "mamba_conv_dtype",
            "mamba_conv_shape",
            "mamba_layer_indices",
            "mamba_recurrent_dtype",
            "mamba_recurrent_shape",
            "moe_placeholder_indices",
            "reported_batch_size",
        ],
        "cache geometry",
    );
    assert_eq!(
        string(field(geometry, "class", "geometry"), "class"),
        "DynamicCache"
    );
    assert_eq!(
        signed_integer(field(geometry, "reported_batch_size", "geometry"), "batch"),
        -1
    );
    assert_eq!(
        integer(field(geometry, "logical_length", "geometry"), "length"),
        length as u64
    );
    assert_eq!(
        usize_array(
            field(geometry, "attention_key_value_shape", "geometry"),
            "KV shape"
        ),
        [1, 2, length, 128]
    );
    assert_eq!(
        usize_array(
            field(geometry, "mamba_conv_shape", "geometry"),
            "conv shape"
        ),
        [1, 6_144, 4]
    );
    assert_eq!(
        usize_array(
            field(geometry, "mamba_recurrent_shape", "geometry"),
            "state shape"
        ),
        [1, 64, 64, 128]
    );
    assert_eq!(
        usize_array(
            field(geometry, "attention_layer_indices", "geometry"),
            "attention layers"
        ),
        [5, 12, 19, 26, 33, 42]
    );
    assert_eq!(
        array(
            field(geometry, "mamba_layer_indices", "geometry"),
            "Mamba indices"
        )
        .len(),
        23
    );
    assert_eq!(
        array(
            field(geometry, "moe_placeholder_indices", "geometry"),
            "MoE indices"
        )
        .len(),
        23
    );
}

/// Validates environment identity, offline controls, and disk-safety evidence.
fn validate_environment(value: &Value) {
    let environment = exact_object(
        value,
        &[
            "absent_optional_distributions",
            "device",
            "disk_space",
            "distribution_manifest_sha256",
            "distribution_provenance",
            "generated_at_utc",
            "offline_environment",
            "packages",
            "python",
            "transitive_distributions",
            "trust_boundary",
        ],
        "environment",
    );
    assert_eq!(
        string(
            field(environment, "distribution_manifest_sha256", "environment"),
            "distribution manifest"
        ),
        DISTRIBUTION_MANIFEST_SHA256
    );
    assert_eq!(
        string_array(
            field(environment, "absent_optional_distributions", "environment"),
            "absent packages"
        ),
        ["flash-attn", "kernels", "torchao"]
    );
    let packages = object(field(environment, "packages", "environment"), "packages");
    assert_eq!(
        packages.get("torch").and_then(Value::as_str),
        Some("2.10.0")
    );
    assert_eq!(
        packages.get("safetensors").and_then(Value::as_str),
        Some("0.8.0")
    );
    assert_eq!(
        packages.get("tokenizers").and_then(Value::as_str),
        Some("0.23.1")
    );
    let offline = object(
        field(environment, "offline_environment", "environment"),
        "offline environment",
    );
    for name in [
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ] {
        assert_eq!(offline.get(name).and_then(Value::as_str), Some("1"));
    }
    assert_eq!(
        offline
            .get("PYTORCH_ENABLE_MPS_FALLBACK")
            .and_then(Value::as_str),
        Some("0")
    );
    assert_eq!(
        offline.get("USE_HUB_KERNELS").and_then(Value::as_str),
        Some("0")
    );
    let disk = object(
        field(environment, "disk_space", "environment"),
        "disk space",
    );
    assert_eq!(
        integer(
            field(disk, "system_preflight_minimum_free_bytes", "disk"),
            "preflight reserve"
        ),
        40 * 1024_u64.pow(3)
    );
    assert_eq!(
        integer(
            field(disk, "system_runtime_minimum_free_bytes", "disk"),
            "runtime reserve"
        ),
        10 * 1024_u64.pow(3)
    );
    assert_eq!(
        integer(
            field(disk, "output_minimum_free_bytes", "disk"),
            "output reserve"
        ),
        1024_u64.pow(3)
    );
    for sample in array(field(disk, "system_samples", "disk"), "disk samples") {
        validate_disk_sample(sample);
    }
    validate_disk_sample(field(disk, "output_preflight", "disk"));
    validate_disk_sample(field(disk, "output_postflight", "disk"));
}

/// Validates one free-space sample without trusting its pass claim.
fn validate_disk_sample(value: &Value) {
    let sample = exact_object(
        value,
        &[
            "free_bytes",
            "minimum_free_bytes",
            "path",
            "purpose",
            "total_bytes",
            "used_bytes",
        ],
        "disk sample",
    );
    let free = integer(field(sample, "free_bytes", "disk sample"), "free bytes");
    let minimum = integer(
        field(sample, "minimum_free_bytes", "disk sample"),
        "minimum bytes",
    );
    assert!(free >= minimum);
    assert!(integer(field(sample, "total_bytes", "disk sample"), "total bytes") > 0);
    assert!(!string(field(sample, "path", "disk sample"), "path").is_empty());
    assert!(!string(field(sample, "purpose", "disk sample"), "purpose").is_empty());
}

/// Validates checkpoint inventory, strict load diagnostics, and MTP filtering.
fn validate_provenance(value: &Value) {
    let provenance = object(value, "provenance");
    assert_eq!(
        string(
            field(provenance, "artifact_manifest_sha256", "provenance"),
            "artifact manifest"
        ),
        CHECKPOINT_ARTIFACT_MANIFEST_SHA256
    );
    assert_eq!(
        integer(
            field(provenance, "physical_tensor_count", "provenance"),
            "physical tensors"
        ),
        PHYSICAL_TENSOR_COUNT as u64
    );
    assert_eq!(
        integer(
            field(provenance, "physical_checkpoint_tensor_bytes", "provenance"),
            "physical bytes"
        ),
        PHYSICAL_TENSOR_BYTES
    );
    assert_eq!(
        integer(
            field(provenance, "reported_checkpoint_tensor_bytes", "provenance"),
            "reported bytes"
        ),
        INDEX_REPORTED_TOTAL_SIZE
    );
    assert_eq!(
        integer(
            field(provenance, "loaded_parameter_count", "provenance"),
            "parameters"
        ),
        INDEX_REPORTED_BASE_PARAMETERS
    );
    assert_eq!(
        integer(
            field(provenance, "loaded_state_tensor_count", "provenance"),
            "state tensors"
        ),
        401
    );
    assert!(boolean(
        field(provenance, "integrity_postflight_passed", "provenance"),
        "postflight"
    ));

    let generator = exact_object(
        field(provenance, "generator", "provenance"),
        &[
            "bytes",
            "git_dirty",
            "git_head",
            "git_tree",
            "relative_path",
            "sha256",
        ],
        "generator",
    );
    assert_eq!(
        integer(field(generator, "bytes", "generator"), "generator bytes"),
        139_608
    );
    assert_eq!(
        string(field(generator, "sha256", "generator"), "generator hash"),
        GENERATOR_SHA256
    );
    assert_eq!(
        string(field(generator, "git_head", "generator"), "generator head"),
        GENERATOR_GIT_HEAD
    );
    assert_eq!(
        string(field(generator, "git_tree", "generator"), "generator tree"),
        GENERATOR_GIT_TREE
    );
    assert!(!boolean(
        field(generator, "git_dirty", "generator"),
        "generator dirty"
    ));

    let artifacts = array(field(provenance, "artifacts", "provenance"), "artifacts");
    assert_eq!(artifacts.len(), CONTROL_FILES.len() + SHARDS.len());
    let observed = artifacts
        .iter()
        .map(|entry| {
            let entry = exact_object(entry, &["bytes", "relative_path", "sha256"], "artifact");
            (
                string(field(entry, "relative_path", "artifact"), "artifact path").to_owned(),
                (
                    integer(field(entry, "bytes", "artifact"), "artifact bytes"),
                    string(field(entry, "sha256", "artifact"), "artifact hash").to_owned(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for (name, bytes, sha256) in CONTROL_FILES {
        assert_eq!(observed.get(name), Some(&(bytes, sha256.to_owned())));
    }
    for shard in SHARDS {
        assert_eq!(
            observed.get(shard.name),
            Some(&(shard.file_bytes, shard.payload_sha256.to_owned()))
        );
    }

    let inventory = object(
        field(provenance, "checkpoint_inventory", "provenance"),
        "checkpoint inventory",
    );
    assert_eq!(
        integer(
            field(inventory, "physical_tensors", "inventory"),
            "physical tensors"
        ),
        PHYSICAL_TENSOR_COUNT as u64
    );
    assert_eq!(
        integer(
            field(inventory, "base_physical_tensors", "inventory"),
            "base tensors"
        ),
        6_243
    );
    assert_eq!(
        integer(
            field(inventory, "mtp_physical_tensors", "inventory"),
            "MTP tensors"
        ),
        MTP_TENSOR_COUNT as u64
    );
    assert_eq!(
        integer(
            field(inventory, "physical_tensor_bytes", "inventory"),
            "physical bytes"
        ),
        PHYSICAL_TENSOR_BYTES
    );
    assert_eq!(
        integer(
            field(inventory, "reported_tensor_bytes", "inventory"),
            "reported bytes"
        ),
        INDEX_REPORTED_TOTAL_SIZE
    );
    assert_eq!(
        integer(
            field(inventory, "reported_size_delta", "inventory"),
            "size delta"
        ),
        INDEX_TOTAL_SIZE_DELTA
    );
    assert_eq!(
        string(
            field(inventory, "header_inventory_sha256", "inventory"),
            "header digest"
        ),
        HEADER_INVENTORY_SHA256
    );
    assert_eq!(
        string(
            field(inventory, "mtp_name_sha256", "inventory"),
            "MTP digest"
        ),
        MTP_NAME_SHA256
    );

    let loading = object(
        field(provenance, "loading_info", "provenance"),
        "loading info",
    );
    for name in [
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    ] {
        validate_empty_diagnostic(field(loading, name, "loading info"));
    }
    let mtp = exact_object(
        field(provenance, "mtp_filtering", "provenance"),
        &[
            "mtp_tensors_executed",
            "official_filter",
            "post_filter_unexpected_keys",
            "raw_mtp_name_sha256",
            "raw_physical_mtp_tensors",
            "remaining_unexplained_anomalies",
        ],
        "MTP filtering",
    );
    assert!(!boolean(
        field(mtp, "mtp_tensors_executed", "MTP"),
        "MTP executed"
    ));
    assert_eq!(
        string(field(mtp, "official_filter", "MTP"), "MTP filter"),
        "mtp.*"
    );
    assert_eq!(
        integer(field(mtp, "raw_physical_mtp_tensors", "MTP"), "MTP tensors"),
        MTP_TENSOR_COUNT as u64
    );
    assert_eq!(
        string(field(mtp, "raw_mtp_name_sha256", "MTP"), "MTP digest"),
        MTP_NAME_SHA256
    );
    assert_eq!(
        integer(
            field(mtp, "remaining_unexplained_anomalies", "MTP"),
            "MTP anomalies"
        ),
        0
    );
    validate_empty_diagnostic(field(mtp, "post_filter_unexpected_keys", "MTP"));
}

/// Validates one strict empty loader-diagnostic class.
fn validate_empty_diagnostic(value: &Value) {
    let diagnostic = exact_object(value, &["count", "sha256"], "load diagnostic");
    assert_eq!(
        integer(field(diagnostic, "count", "diagnostic"), "count"),
        0
    );
    assert_eq!(
        string(field(diagnostic, "sha256", "diagnostic"), "digest"),
        EMPTY_SHA256
    );
}

/// Validates all descriptor shapes, names, internal hashes, and trace labels.
fn validate_outputs(outputs: &OutputDescriptors) {
    validate_common_descriptor(
        &outputs.prompt_logits.relative_path,
        &outputs.prompt_logits.encoding,
        &outputs.prompt_logits.storage_dtype,
        &outputs.prompt_logits.shape,
        outputs.prompt_logits.elements,
        outputs.prompt_logits.bytes,
        &outputs.prompt_logits.sha256,
        outputs.prompt_logits.finite,
        FILES[1],
        &[131_072],
    );
    assert_eq!(outputs.prompt_logits.source_dtype, "float32");
    assert_eq!(outputs.prompt_logits.top1_token_id, 1_256);
    assert_eq!(outputs.prompt_logits.top10.len(), 10);
    validate_hidden(&outputs.hidden_last_trace, FILES[2], "stateless_prompt", 5);
    validate_matrix(&outputs.cached_prefix_logits, FILES[3]);
    validate_matrix(&outputs.teacher_prefix_logits, FILES[4]);
    validate_hidden(
        &outputs.cached_prefill_hidden_trace,
        FILES[5],
        "cached_prefill",
        5,
    );
    validate_hidden(
        &outputs.cached_decode_hidden_trace,
        FILES[6],
        "cached_decode",
        6,
    );
}

/// Validates one hidden trace descriptor.
fn validate_hidden(
    descriptor: &HiddenDescriptor,
    anchor: FileAnchor,
    phase: &str,
    position: usize,
) {
    validate_common_descriptor(
        &descriptor.relative_path,
        &descriptor.encoding,
        &descriptor.storage_dtype,
        &descriptor.shape,
        descriptor.elements,
        descriptor.bytes,
        &descriptor.sha256,
        descriptor.finite,
        anchor,
        &[54, 2_688],
    );
    assert_eq!(descriptor.source_dtype, "float16");
    assert_eq!(descriptor.phase, phase);
    assert_eq!(descriptor.token_position, position);
    assert_eq!(descriptor.labels.len(), 54);
    assert_eq!(
        descriptor.labels.first().map(String::as_str),
        Some("embeddings")
    );
    assert_eq!(descriptor.labels.last().map(String::as_str), Some("norm_f"));
    for index in 0..52 {
        assert_eq!(descriptor.labels[index + 1], format!("layers.{index}"));
    }
}

/// Validates one 32-row vocabulary descriptor.
fn validate_matrix(descriptor: &MatrixDescriptor, anchor: FileAnchor) {
    validate_common_descriptor(
        &descriptor.relative_path,
        &descriptor.encoding,
        &descriptor.storage_dtype,
        &descriptor.shape,
        descriptor.elements,
        descriptor.bytes,
        &descriptor.sha256,
        descriptor.finite,
        anchor,
        &[32, 131_072],
    );
    assert_eq!(descriptor.source_dtype, "float32");
    assert_eq!(descriptor.row_position_ids, (5..=36).collect::<Vec<_>>());
    assert_eq!(descriptor.row_top1_token_ids, GENERATED_TOKEN_IDS);
}

/// Validates fields shared by every numerical payload.
#[allow(clippy::too_many_arguments)]
fn validate_common_descriptor(
    path: &str,
    encoding: &str,
    storage_dtype: &str,
    shape: &[usize],
    elements: usize,
    bytes: u64,
    sha256: &str,
    finite: bool,
    anchor: FileAnchor,
    expected_shape: &[usize],
) {
    assert_eq!(safe_filename(path), anchor.name);
    assert_eq!(encoding, "ieee754-f32-little-endian");
    assert_eq!(storage_dtype, "float32");
    assert_eq!(shape, expected_shape);
    assert_eq!(elements, expected_shape.iter().product::<usize>());
    assert_eq!(bytes, elements as u64 * 4);
    assert_eq!(bytes, anchor.bytes);
    assert_eq!(sha256, anchor.sha256);
    assert!(finite);
}

/// Validates generation decisions and the non-crossed parity contract.
fn validate_generation(value: &Value) {
    let generation = exact_object(
        value,
        &[
            "cached_prefix_top1",
            "decisions",
            "decoded_text",
            "eos_token_ids",
            "max_new_tokens",
            "minimum_cached_prefill_stateless_hidden_cosine",
            "minimum_cached_prefill_stateless_hidden_cosine_row",
            "minimum_cached_stateless_prefix_cosine",
            "minimum_cached_stateless_prefix_cosine_row",
            "minimum_margin",
            "minimum_margin_to_epsilon_ratio",
            "official_path_divergence",
            "parity_contract",
            "prompt_cached_cosine",
            "stateless_prefix_top1",
            "token_ids",
        ],
        "generation",
    );
    assert_eq!(
        integer(
            field(generation, "max_new_tokens", "generation"),
            "new tokens"
        ),
        32
    );
    assert_eq!(
        u32_array(field(generation, "eos_token_ids", "generation"), "EOS"),
        [2, 11]
    );
    for name in ["token_ids", "cached_prefix_top1", "stateless_prefix_top1"] {
        assert_eq!(
            u32_array(field(generation, name, "generation"), name),
            GENERATED_TOKEN_IDS
        );
    }
    assert_eq!(
        string(
            field(generation, "decoded_text", "generation"),
            "decoded text"
        ),
        "  \n".repeat(16)
    );
    assert!(
        number(
            field(generation, "prompt_cached_cosine", "generation"),
            "prompt cosine"
        ) >= 0.999
    );
    assert!(
        number(
            field(
                generation,
                "minimum_cached_prefill_stateless_hidden_cosine",
                "generation"
            ),
            "hidden cosine"
        ) >= 0.999
    );
    assert_eq!(
        integer(
            field(
                generation,
                "minimum_cached_stateless_prefix_cosine_row",
                "generation"
            ),
            "minimum row"
        ),
        22
    );
    assert_close(
        number(
            field(
                generation,
                "minimum_cached_stateless_prefix_cosine",
                "generation",
            ),
            "minimum cosine",
        ),
        0.921_504_794_757_117_6,
        1e-15,
    );
    let decisions = array(field(generation, "decisions", "generation"), "decisions");
    assert_eq!(decisions.len(), 32);
    for (step, decision) in decisions.iter().enumerate() {
        let decision = object(decision, "decision");
        assert_eq!(
            integer(field(decision, "step", "decision"), "step"),
            step as u64
        );
        assert_eq!(
            integer(field(decision, "top1_token_id", "decision"), "top1"),
            GENERATED_TOKEN_IDS[step] as u64
        );
        assert!(boolean(
            field(
                decision,
                "stable_under_two_sided_one_ulp_budget",
                "decision"
            ),
            "stability"
        ));
        assert!(
            number(
                field(decision, "margin_to_epsilon_ratio", "decision"),
                "margin ratio"
            ) > 2.0
        );
    }
    let contract = exact_object(
        field(generation, "parity_contract", "generation"),
        &["cross_runtime", "source_self_consistency"],
        "parity contract",
    );
    let cross_runtime = exact_object(
        field(contract, "cross_runtime", "parity contract"),
        &[
            "cached_comparison",
            "crossing_official_paths_is_not_a_cross_runtime_gate",
            "minimum_corresponding_path_cosine",
            "stateless_comparison",
        ],
        "cross runtime",
    );
    assert_close(
        number(
            field(
                cross_runtime,
                "minimum_corresponding_path_cosine",
                "cross runtime",
            ),
            "cross-runtime cosine",
        ),
        0.999,
        0.0,
    );
    assert!(boolean(
        field(
            cross_runtime,
            "crossing_official_paths_is_not_a_cross_runtime_gate",
            "cross runtime"
        ),
        "path crossing"
    ));
    assert_eq!(
        string(
            field(cross_runtime, "cached_comparison", "cross runtime"),
            "cached comparison",
        ),
        "Burn cached rows vs official cached rows"
    );
    assert_eq!(
        string(
            field(cross_runtime, "stateless_comparison", "cross runtime"),
            "stateless comparison",
        ),
        "Burn stateless rows vs official teacher-forced rows"
    );
    let self_consistency = exact_object(
        field(contract, "source_self_consistency", "parity contract"),
        &[
            "cached_prefill_teacher_observed_cosine",
            "cached_teacher_top1_exact_passed",
            "cached_teacher_top1_exact_required",
            "recurrent_cached_vs_chunked_reason",
            "recurrent_cached_vs_chunked_teacher_cosine_gate",
            "same_path_prefill_minimum_cosine",
            "stateless_cached_prefill_hidden_minimum_cosine",
            "stateless_cached_prefill_observed_cosine",
        ],
        "source self consistency",
    );
    assert!(boolean(
        field(
            self_consistency,
            "cached_teacher_top1_exact_passed",
            "source self consistency",
        ),
        "cached/teacher top-1 pass",
    ));
    assert!(boolean(
        field(
            self_consistency,
            "cached_teacher_top1_exact_required",
            "source self consistency",
        ),
        "cached/teacher top-1 requirement",
    ));
    assert!(
        field(
            self_consistency,
            "recurrent_cached_vs_chunked_teacher_cosine_gate",
            "source self consistency",
        )
        .is_null()
    );
    assert_close(
        number(
            field(
                self_consistency,
                "stateless_cached_prefill_observed_cosine",
                "source self consistency",
            ),
            "same-path observed cosine",
        ),
        1.000_000_000_000_000_9,
        1e-15,
    );
    let divergence = exact_object(
        field(generation, "official_path_divergence", "generation"),
        &[
            "hidden_localization",
            "minimum_logits_cosine",
            "minimum_logits_cosine_decision",
            "possible_amplifier_not_proven",
            "status",
            "worst_logits_comparison",
        ],
        "path divergence",
    );
    assert_eq!(
        string(field(divergence, "status", "divergence"), "status"),
        "observed_and_preserved"
    );
    assert_eq!(
        integer(
            field(divergence, "minimum_logits_cosine_decision", "divergence"),
            "minimum decision"
        ),
        22
    );
    assert_close(
        number(
            field(divergence, "minimum_logits_cosine", "divergence"),
            "minimum cosine",
        ),
        0.921_504_794_757_117_6,
        1e-15,
    );
    assert_eq!(
        string(
            field(divergence, "possible_amplifier_not_proven", "divergence",),
            "possible amplifier",
        ),
        "small recurrent/chunked boundary differences can cross discontinuous top-6 MoE routing boundaries"
    );
    validate_worst_logits_comparison(field(divergence, "worst_logits_comparison", "divergence"));
    validate_hidden_localization(field(divergence, "hidden_localization", "divergence"));
}

/// Validates the exact row-level diagnostic at the worst official path mismatch.
fn validate_worst_logits_comparison(value: &Value) {
    let comparison = exact_object(
        value,
        &[
            "cached_binary16_stability",
            "comparison",
            "decision",
            "dot",
            "elements",
            "gate_cosine",
            "left_at_max_absolute_error",
            "left_at_max_relative_error",
            "left_l2_norm",
            "left_name",
            "left_top2",
            "max_absolute_error",
            "max_absolute_error_index",
            "max_relative_error",
            "max_relative_error_index",
            "recomputed_float64_cosine",
            "relative_error_denominator",
            "right_at_max_absolute_error",
            "right_at_max_relative_error",
            "right_l2_norm",
            "right_name",
            "right_top2",
        ],
        "worst logits comparison",
    );
    assert_eq!(
        string(field(comparison, "comparison", "comparison"), "comparison"),
        "cached_recurrent_vs_teacher_forced_stateless"
    );
    assert_eq!(
        integer(field(comparison, "decision", "comparison"), "decision"),
        22
    );
    assert_eq!(
        integer(field(comparison, "elements", "comparison"), "elements"),
        131_072
    );
    for (field_name, expected) in [
        ("gate_cosine", 0.921_504_794_757_117_6),
        ("recomputed_float64_cosine", 0.921_504_794_757_116_9),
        ("dot", 401_545.881_047_064_2),
        ("left_l2_norm", 637.485_684_687_833_3),
        ("right_l2_norm", 683.545_044_854_600_4),
        ("max_absolute_error", 3.314_453_125),
        ("max_relative_error", 17_301.861_087_144_09),
        ("left_at_max_absolute_error", 5.617_187_5),
        ("right_at_max_absolute_error", 2.302_734_375),
        ("left_at_max_relative_error", 1.195_312_5),
        ("right_at_max_relative_error", 0.000_069_081_783_294_677_73),
    ] {
        assert_close(
            number(field(comparison, field_name, "comparison"), field_name),
            expected,
            1e-12,
        );
    }
    assert_eq!(
        integer(
            field(comparison, "max_absolute_error_index", "comparison"),
            "maximum absolute error index",
        ),
        6_507
    );
    assert_eq!(
        integer(
            field(comparison, "max_relative_error_index", "comparison"),
            "maximum relative error index",
        ),
        67_673
    );
    assert_eq!(
        string(field(comparison, "left_name", "comparison"), "left name"),
        "cached_prefix_logits"
    );
    assert_eq!(
        string(field(comparison, "right_name", "comparison"), "right name"),
        "teacher_forced_prefix_logits"
    );
    assert_eq!(
        string(
            field(comparison, "relative_error_denominator", "comparison"),
            "relative error denominator",
        ),
        "max(abs(right_reference), 1e-12)"
    );
    validate_top2(
        field(comparison, "left_top2", "comparison"),
        1_256,
        17.437_5,
        8_129,
        11.648_437_5,
        5.789_062_5,
    );
    validate_top2(
        field(comparison, "right_top2", "comparison"),
        1_256,
        17.031_25,
        1_032,
        12.039_062_5,
        4.992_187_5,
    );

    let stability = exact_object(
        field(comparison, "cached_binary16_stability", "comparison"),
        &[
            "elementwise_f16_ulp",
            "margin",
            "margin_to_epsilon_ratio",
            "stable_under_two_sided_one_ulp_budget",
            "step",
            "top1_logit",
            "top1_token_id",
            "top2_logit",
            "top2_token_id",
        ],
        "cached binary16 stability",
    );
    assert_eq!(integer(field(stability, "step", "stability"), "step"), 22);
    assert!(boolean(
        field(
            stability,
            "stable_under_two_sided_one_ulp_budget",
            "stability",
        ),
        "stable margin",
    ));
    assert_close(
        number(
            field(stability, "margin_to_epsilon_ratio", "stability"),
            "margin ratio",
        ),
        370.5,
        0.0,
    );
}

/// Validates one stable top-two row summary.
fn validate_top2(
    value: &Value,
    top1_id: u64,
    top1_logit: f64,
    top2_id: u64,
    top2_logit: f64,
    margin: f64,
) {
    let top = exact_object(
        value,
        &[
            "margin",
            "top1_logit",
            "top1_token_id",
            "top2_logit",
            "top2_token_id",
        ],
        "top-two summary",
    );
    assert_eq!(
        integer(field(top, "top1_token_id", "top two"), "top1 ID"),
        top1_id
    );
    assert_eq!(
        integer(field(top, "top2_token_id", "top two"), "top2 ID"),
        top2_id
    );
    assert_close(
        number(field(top, "top1_logit", "top two"), "top1 logit"),
        top1_logit,
        0.0,
    );
    assert_close(
        number(field(top, "top2_logit", "top two"), "top2 logit"),
        top2_logit,
        0.0,
    );
    assert_close(
        number(field(top, "margin", "top two"), "margin"),
        margin,
        0.0,
    );
}

/// Validates the all-layer localization that distinguishes the two official paths.
fn validate_hidden_localization(value: &Value) {
    let localization = exact_object(
        value,
        &[
            "comparison",
            "corresponding_path_cross_runtime_minimum_cosine",
            "cross_runtime_acceptance_gate",
            "first_decision_below_0_999_diagnostic_threshold",
            "per_decision",
            "worst_decision",
            "worst_decision_largest_boundary_drop",
            "worst_decision_layer_cosines",
        ],
        "hidden localization",
    );
    assert_eq!(
        string(
            field(localization, "comparison", "localization"),
            "comparison"
        ),
        "cached_recurrent_or_prefill_vs_teacher_forced_chunked"
    );
    assert!(!boolean(
        field(
            localization,
            "cross_runtime_acceptance_gate",
            "localization"
        ),
        "cross-runtime gate"
    ));
    assert_close(
        number(
            field(
                localization,
                "corresponding_path_cross_runtime_minimum_cosine",
                "localization",
            ),
            "corresponding cosine",
        ),
        0.999,
        0.0,
    );
    let first = object(
        field(
            localization,
            "first_decision_below_0_999_diagnostic_threshold",
            "localization",
        ),
        "first divergence",
    );
    assert_eq!(
        integer(field(first, "decision", "first divergence"), "decision"),
        1
    );
    assert_eq!(
        integer(field(first, "layer_index", "first divergence"), "layer"),
        27
    );
    assert_close(
        number(field(first, "minimum_cosine", "first divergence"), "cosine"),
        0.991_921_231_340_888_7,
        1e-15,
    );
    let worst = object(
        field(localization, "worst_decision", "localization"),
        "worst divergence",
    );
    assert_eq!(
        integer(field(worst, "decision", "worst divergence"), "decision"),
        14
    );
    assert_eq!(
        integer(field(worst, "layer_index", "worst divergence"), "layer"),
        1
    );
    assert_close(
        number(field(worst, "minimum_cosine", "worst divergence"), "cosine"),
        0.441_890_677_981_716_6,
        1e-15,
    );
    let drop = object(
        field(
            localization,
            "worst_decision_largest_boundary_drop",
            "localization",
        ),
        "largest drop",
    );
    assert_eq!(
        integer(field(drop, "layer_index", "largest drop"), "layer"),
        1
    );
    assert_close(
        number(
            field(drop, "change_from_previous_boundary", "largest drop"),
            "drop",
        ),
        -0.450_143_175_176_246_3,
        1e-15,
    );
    assert_eq!(
        array(
            field(localization, "per_decision", "localization"),
            "per-decision localization"
        )
        .len(),
        32
    );
    assert_eq!(
        array(
            field(localization, "worst_decision_layer_cosines", "localization"),
            "layer localization"
        )
        .len(),
        54
    );
}

/// Independently rechecks descriptor summaries and every vocabulary-row argmax.
fn validate_payload_values(
    record: &BaseReferenceRecord,
    prompt: &[f32],
    stateless_hidden: &[f32],
    cached: &[f32],
    teacher: &[f32],
    cached_prefill_hidden: &[f32],
) {
    assert_eq!(prompt.len(), 131_072);
    assert_eq!(stateless_hidden.len(), 54 * 2_688);
    assert_eq!(cached.len(), 32 * 131_072);
    assert_eq!(teacher.len(), 32 * 131_072);
    assert_eq!(
        stateless_hidden, cached_prefill_hidden,
        "same-path hidden payloads changed"
    );
    let top10 = ranked_indices(prompt, 10);
    for (expected, observed) in record
        .outputs
        .prompt_logits
        .top10
        .iter()
        .zip(top10.iter().copied())
    {
        assert_eq!(expected.token_id as usize, observed);
        assert_eq!(expected.logit, prompt[observed]);
    }
    assert_eq!(
        record.outputs.prompt_logits.top1_token_id as usize,
        top10[0]
    );
    let sum = prompt.iter().map(|value| f64::from(*value)).sum::<f64>();
    let norm = prompt
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    assert_close(sum, record.outputs.prompt_logits.sum, 1e-9);
    assert_close(norm, record.outputs.prompt_logits.l2_norm, 1e-9);
    let mut minimum = f64::INFINITY;
    let mut minimum_step = 0;
    for step in 0..32 {
        let cached_row = &cached[step * 131_072..(step + 1) * 131_072];
        let teacher_row = &teacher[step * 131_072..(step + 1) * 131_072];
        assert_eq!(top1(cached_row), GENERATED_TOKEN_IDS[step] as usize);
        assert_eq!(top1(teacher_row), GENERATED_TOKEN_IDS[step] as usize);
        let cosine = cosine(cached_row, teacher_row);
        if cosine < minimum {
            minimum = cosine;
            minimum_step = step;
        }
    }
    assert_eq!(minimum_step, 22);
    assert_close(minimum, 0.921_504_794_757_117_6, 2e-12);
}

/// Reads one independently anchored F32LE payload and rejects invalid scalars.
fn read_descriptor(
    root: &Path,
    relative: &str,
    bytes: u64,
    elements: usize,
    internal_sha256: &str,
) -> Vec<f32> {
    let name = safe_filename(relative);
    let anchor = FILES
        .iter()
        .find(|entry| entry.name == name)
        .expect("descriptor references an unanchored file");
    assert_eq!(bytes, anchor.bytes);
    assert_eq!(internal_sha256, anchor.sha256);
    let path = root.join(name);
    let payload =
        fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    assert_eq!(payload.len(), elements * 4);
    payload
        .chunks_exact(4)
        .enumerate()
        .map(|(index, bytes)| {
            let value = f32::from_le_bytes(bytes.try_into().unwrap());
            assert!(
                value.is_finite(),
                "{} scalar {index} is non-finite",
                path.display()
            );
            value
        })
        .collect()
}

/// Returns stable descending top-k indices with lower-index tie breaking.
fn ranked_indices(values: &[f32], count: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    indices.truncate(count);
    indices
}

/// Returns the stable argmax index.
fn top1(values: &[f32]) -> usize {
    ranked_indices(values, 1)[0]
}

/// Computes an F64 cosine over one pair of finite vectors.
fn cosine(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len());
    let mut dot = 0.0;
    let mut left_sq = 0.0;
    let mut right_sq = 0.0;
    for (left, right) in left.iter().zip(right) {
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        left_sq += left * left;
        right_sq += right * right;
    }
    dot / (left_sq.sqrt() * right_sq.sqrt())
}

/// Requires two reviewed floating-point facts to match within an explicit bound.
fn assert_close(observed: f64, expected: f64, tolerance: f64) {
    assert!(observed.is_finite() && expected.is_finite());
    assert!(
        (observed - expected).abs() <= tolerance,
        "{observed} != {expected}"
    );
}

/// Requires one object to expose exactly the reviewed keys.
fn exact_object<'a>(value: &'a Value, expected: &[&str], path: &str) -> &'a Map<String, Value> {
    let object = object(value, path);
    let observed = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(observed, expected, "{path} key set changed");
    object
}

/// Requires a JSON object.
fn object<'a>(value: &'a Value, path: &str) -> &'a Map<String, Value> {
    value
        .as_object()
        .unwrap_or_else(|| panic!("{path} must be an object"))
}

/// Requires one named object field.
fn field<'a>(object: &'a Map<String, Value>, name: &str, path: &str) -> &'a Value {
    object
        .get(name)
        .unwrap_or_else(|| panic!("{path}.{name} is missing"))
}

/// Requires a JSON array.
fn array<'a>(value: &'a Value, path: &str) -> &'a [Value] {
    value
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_else(|| panic!("{path} must be an array"))
}

/// Requires a JSON string.
fn string<'a>(value: &'a Value, path: &str) -> &'a str {
    value
        .as_str()
        .unwrap_or_else(|| panic!("{path} must be a string"))
}

/// Requires a JSON boolean.
fn boolean(value: &Value, path: &str) -> bool {
    value
        .as_bool()
        .unwrap_or_else(|| panic!("{path} must be a boolean"))
}

/// Requires an exact nonnegative integer.
fn integer(value: &Value, path: &str) -> u64 {
    value
        .as_u64()
        .unwrap_or_else(|| panic!("{path} must be a nonnegative integer"))
}

/// Requires an exact signed integer.
fn signed_integer(value: &Value, path: &str) -> i64 {
    value
        .as_i64()
        .unwrap_or_else(|| panic!("{path} must be an integer"))
}

/// Requires a finite JSON number.
fn number(value: &Value, path: &str) -> f64 {
    let value = value
        .as_f64()
        .unwrap_or_else(|| panic!("{path} must be a number"));
    assert!(value.is_finite(), "{path} must be finite");
    value
}

/// Decodes an exact U32 JSON array.
fn u32_array(value: &Value, path: &str) -> Vec<u32> {
    array(value, path)
        .iter()
        .map(|value| u32::try_from(integer(value, path)).expect("U32 array value overflow"))
        .collect()
}

/// Decodes an exact usize JSON array.
fn usize_array(value: &Value, path: &str) -> Vec<usize> {
    array(value, path)
        .iter()
        .map(|value| usize::try_from(integer(value, path)).expect("usize array value overflow"))
        .collect()
}

/// Decodes a JSON string array.
fn string_array<'a>(value: &'a Value, path: &str) -> Vec<&'a str> {
    array(value, path)
        .iter()
        .map(|value| string(value, path))
        .collect()
}

/// Accepts only one ordinary relative filename.
fn safe_filename(path: &str) -> &str {
    let path_value = Path::new(path);
    assert!(
        !path_value.is_absolute(),
        "bundle filename must be relative"
    );
    let components = path_value.components().collect::<Vec<_>>();
    assert_eq!(components.len(), 1, "bundle filename must not be nested");
    assert!(matches!(components[0], Component::Normal(_)));
    path
}

/// Rejects symlinks and non-regular files before hashing.
fn assert_regular_file(path: &Path) {
    let metadata = fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()));
    assert!(metadata.is_file() && !metadata.file_type().is_symlink());
}

/// Streams one regular file into SHA-256.
fn sha256_file_hex(path: &Path) -> String {
    let mut reader = BufReader::new(File::open(path).expect("open anchored bundle member"));
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .expect("hash anchored bundle member");
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every base-only anchor is populated and distinct by filename.
    #[test]
    fn reviewed_anchor_set_is_complete() {
        let anchors = trusted_reference_hashes();
        assert_eq!(anchors.len(), 7);
        assert!(anchors.values().all(|digest| digest.len() == 64));
    }

    /// Top-level additions cannot silently expand the reviewed schema.
    #[test]
    fn top_level_schema_rejects_unknown_fields() {
        let value = serde_json::json!({
            "schema_version": 3,
            "repository": REPOSITORY,
            "revision": REVISION,
            "reference": {},
            "prompt": SMOKE_PROMPT,
            "input_ids": SMOKE_TOKEN_IDS,
            "tokenizer": {},
            "execution": {},
            "environment": {},
            "provenance": {},
            "outputs": {},
            "generation": {},
            "timings_seconds": {},
            "trust_me": true
        });
        assert!(serde_json::from_value::<BaseReferenceRecord>(value).is_err());
    }

    /// Traversal, absolute, and nested payload references remain impossible.
    #[test]
    fn filename_policy_is_fail_closed() {
        for invalid in ["../oracle", "/tmp/oracle", "nested/oracle", "."] {
            assert!(std::panic::catch_unwind(|| safe_filename(invalid)).is_err());
        }
        assert_eq!(safe_filename("oracle.f32le"), "oracle.f32le");
    }

    /// Stable ranking uses lower token ID for exact ties.
    #[test]
    fn ranking_is_descending_and_tie_stable() {
        assert_eq!(ranked_indices(&[1.0, 3.0, 3.0, -1.0], 3), [1, 2, 0]);
    }
}
