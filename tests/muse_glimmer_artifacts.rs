//! Local-only provenance and pre-acceptance gates for Muse-Glimmer-30B.
//!
//! These tests never contact a model hub. Point `TRANSFORMERS_MUSE_GLIMMER_DIR`
//! at a complete local copy of `meta-models/Muse-Glimmer-30B` revision
//! `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, then inspect the bounded
//! metadata and SafeTensors headers with:
//!
//! ```text
//! cargo test --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,tokenizers,chat-templates \
//!   --test muse_glimmer_artifacts \
//!   validates_pinned_muse_glimmer_config_headers_plan_memory_and_tokenizer \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! Stream all 59.6 GB through the independent payload identity gate with:
//!
//! ```text
//! cargo test --release --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std \
//!   --test muse_glimmer_artifacts \
//!   validates_pinned_muse_glimmer_full_shard_payload_hashes \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! The optional Metal gate audits strict streaming and normalized generation
//! metadata, but deliberately does not claim reference-logit acceptance. To
//! create the separately authenticated official numerical oracle, first check
//! out Transformers at `cfa309cfa34b5497a32fd50af49aa97aac94c054`, prepare
//! the exact Python environment enforced by the generator, and run:
//!
//! ```text
//! python crates/transformers/scripts/generate_muse_glimmer_reference.py \
//!   --model-dir "$TRANSFORMERS_MUSE_GLIMMER_DIR" \
//!   --transformers-dir /absolute/path/to/pinned/transformers \
//!   --output-dir /absolute/external/reference-parent
//! ```
//!
//! The tool is deliberately local-only and MPS/F16/eager-only. It atomically
//! publishes a create-only three-file bundle after recording full-vocabulary
//! prompt logits, 54 hidden-trace rows, and 32 cached/stateless greedy choices.
//! An independent reviewer must pin all three printed output hashes below; the
//! ignored Rust acceptance gate fails closed until that has happened.
//!
//! Once those hashes are reviewed and pinned, run the full comparison with:
//!
//! ```text
//! TRANSFORMERS_MUSE_GLIMMER_REFERENCE_DIR=/absolute/external/reference-parent \
//! cargo test --release --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,metal,tokenizers \
//!   --test muse_glimmer_artifacts \
//!   matches_pinned_muse_glimmer_official_f16_reference_on_metal \
//!   -- --ignored --exact --nocapture
//! ```

use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    mem::size_of,
    path::{Component, Path, PathBuf},
    time::Instant,
};

#[cfg(all(feature = "metal", feature = "tokenizers", target_os = "macos"))]
use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

use transformers::{
    checkpoint::{TensorDType, TensorOperation, TensorTransform},
    models::{
        config::ModelConfig,
        muse_glimmer::{MuseGlimmerCheckpointOptions, MuseGlimmerLayerType, MuseGlimmerTextConfig},
    },
};

#[cfg(feature = "tokenizers")]
use transformers::checkpoint::{
    ArtifactSource, CheckpointManifest, DTypePolicy, LoadOptions, LoadPlan, PlannedTransform,
};

#[cfg(feature = "tokenizers")]
use transformers::{EncodeOptions, LocalTokenizer};

#[cfg(feature = "chat-templates")]
use transformers::{ChatMessage, ChatTemplateOptions};

#[cfg(all(feature = "metal", feature = "tokenizers"))]
use burn::{
    backend::Metal,
    tensor::{Bool, DType, Int, Tensor, TensorData, backend::Backend},
};
#[cfg(all(feature = "metal", feature = "tokenizers"))]
use transformers::{
    AutoCausalLm, AutoLoadOptions, DecodeInput, DecodeState, ForwardOptions, LogitsSelection,
    SamplingStrategy, TokenMask, architecture::CausalLmInput, cache::CacheConfig,
};

/// Immutable upstream repository certified by every local gate.
const REPOSITORY: &str = "meta-models/Muse-Glimmer-30B";

/// Immutable upstream Git revision certified by every local gate.
const REVISION: &str = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68";

/// Official implementation repository used as the semantic reference.
const REFERENCE_REPOSITORY: &str = "huggingface/transformers";

/// Immutable Transformers commit containing the referenced Muse implementation.
const REFERENCE_REVISION: &str = "cfa309cfa34b5497a32fd50af49aa97aac94c054";

/// Repository-relative official implementation used for equation parity.
const REFERENCE_SOURCE: &str = "src/transformers/models/muse_glimmer/modeling_muse_glimmer.py";

/// Exact byte length of the official implementation at `REFERENCE_REVISION`.
const REFERENCE_SOURCE_BYTES: u64 = 53_259;

/// SHA-256 of the official implementation at `REFERENCE_REVISION`.
const REFERENCE_SOURCE_SHA256: &str =
    "459498a758446f580fb4a9203c64ab4be68488b40a04acd379dd62658d5bad71";

/// Exact pinned config identity, also enforced by the committed fixture.
const CONFIG_SHA256: &str = "5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14";

/// Exact pinned generation-config identity, also enforced by its fixture.
const GENERATION_CONFIG_SHA256: &str =
    "1fa51889b1f8d3659802dedaa27e005b81e5c58483f13ecf13f2d97306bc6e35";

/// Exact lengths and SHA-256 identities of all pinned control artifacts.
const CONTROL_FILES: [(&str, u64, &str); 7] = [
    (
        "chat_template.jinja",
        9_992,
        "cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678",
    ),
    ("config.json", 5_109, CONFIG_SHA256),
    ("generation_config.json", 202, GENERATION_CONFIG_SHA256),
    (
        "model.safetensors.index.json",
        132_674,
        "7d817b4dccb1b123fc6c1939356c65cee3a0ad462a5b821ac88280990a27d1ba",
    ),
    (
        "processor_config.json",
        1_084,
        "97e2a486dd9866b81f40cf4b8bc0c9ced9a7cd8a5bc65aa4cc2f4de0712dae77",
    ),
    (
        "tokenizer.json",
        28_129_897,
        "c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d",
    ),
    (
        "tokenizer_config.json",
        79_936,
        "781e6c74f571642c71202167b67d9255b28cc439bdda1582ff31346182f5a9c5",
    ),
];

/// Exact physical weight files, lengths, and LFS SHA-256 identities.
const SHARDS: [(&str, u64, &str); 2] = [
    (
        "model-00001-of-00002.safetensors",
        49_950_112_952,
        "8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694",
    ),
    (
        "model-00002-of-00002.safetensors",
        9_603_322_320,
        "b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381",
    ),
];

/// SHA-256 of every physical tensor name, shard, dtype, shape, and data range.
#[cfg(feature = "tokenizers")]
const HEADER_INVENTORY_SHA256: &str =
    "7bf662cfb4142c1f9994926574082e9adf52f9c335f226e46b998a7d9c4547cb";

/// SHA-256 of every F16 plan target, source, dtype, and target shape.
const F16_PLAN_SHA256: &str = "fde94638b261fbaee47e53dbf035660e5bd1732a6d92e45c45cddb720ed95d6d";

/// Number of physical BF16 tensors in the complete multimodal checkpoint.
const PHYSICAL_TENSOR_COUNT: usize = 1_436;

/// Number of language-tower and vocabulary-head tensors consumed by Burn.
const CONSUMED_TENSOR_COUNT: usize = 627;

/// Number of consumed tensors renamed out of the multimodal namespace.
const RENAMED_TENSOR_COUNT: usize = 626;

/// Number of narrowly allowlisted visual tensors left uninstantiated.
#[cfg(feature = "tokenizers")]
const VISUAL_TENSOR_COUNT: usize = 809;

/// BF16 bytes across the complete multimodal tensor inventory.
const CHECKPOINT_TENSOR_BYTES: u64 = 59_553_253_376;

/// Physical bytes across the two complete SafeTensors files.
const CHECKPOINT_ARTIFACT_BYTES: u64 = 59_553_435_272;

/// BF16 source bytes and F16 target bytes for the consumed text graph.
#[cfg(feature = "tokenizers")]
const LANGUAGE_PARAMETER_BYTES: u64 = 55_709_561_856;

/// Largest target payload: either independent `[202048, 6656]` matrix.
#[cfg(feature = "tokenizers")]
const LARGEST_TARGET_PAYLOAD_BYTES: u64 = 2_689_662_976;

/// Conservative peak host staging: largest shard plus largest target payload.
#[cfg(feature = "tokenizers")]
const PEAK_HOST_STAGING_BYTES: u64 = 52_639_775_928;

/// Short prompt with a stable six-token representation in the pinned tokenizer.
const SMOKE_PROMPT: &str = "Hello world, cached decoding.";

/// Exact token IDs without tokenizer-injected special tokens.
const SMOKE_TOKEN_IDS: [u32; 6] = [19_873, 3_817, 24, 41_346, 63_623, 26];

/// Fixed create-only directory emitted by the official-reference generator.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const REFERENCE_BUNDLE_DIRECTORY: &str = "muse-glimmer-30b-a4e59da-reference.bundle";

/// Strict JSON record within the reference bundle.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const REFERENCE_JSON_FILENAME: &str = "muse-glimmer-30b-a4e59da-reference.json";

/// Full-vocabulary prompt-logit payload within the reference bundle.
const REFERENCE_LOGITS_FILENAME: &str = "muse-glimmer-30b-a4e59da-prompt-logits.f32le";

/// Last-token embedding/layer/final-norm trace within the reference bundle.
const REFERENCE_HIDDEN_FILENAME: &str = "muse-glimmer-30b-a4e59da-hidden-last.f32le";

/// Canonical digest of the nine independently pinned checkpoint artifacts.
const REFERENCE_ARTIFACT_MANIFEST_SHA256: &str =
    "5b8265f3a0fe6b07668caa8e5b949667b8f26f664550dfcc86d35f93fb7971ab";

/// JSON output identity to pin only after independent reference-run review.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const TRUSTED_MUSE_REFERENCE_JSON_SHA256: Option<&str> = None;

/// Numerical prompt-logit identity to pin independently of the JSON claim.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const TRUSTED_MUSE_REFERENCE_LOGITS_SHA256: Option<&str> = None;

/// Numerical hidden-trace identity to pin independently of the JSON claim.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const TRUSTED_MUSE_REFERENCE_HIDDEN_SHA256: Option<&str> = None;

/// Resolves the explicitly selected immutable local artifact directory.
fn pinned_root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("TRANSFORMERS_MUSE_GLIMMER_DIR").unwrap_or_else(|| {
            panic!("set TRANSFORMERS_MUSE_GLIMMER_DIR to {REPOSITORY} revision {REVISION}")
        }),
    )
}

/// Computes SHA-256 through the audited implementation used by the test harness.
fn sha256_hex(payload: &[u8]) -> String {
    format!("{:x}", Sha256::digest(payload))
}

/// Streams one regular file into SHA-256 without allocating a shard-sized buffer.
fn sha256_file_hex(path: &Path) -> String {
    let link_metadata = fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()));
    assert!(
        !link_metadata.file_type().is_symlink(),
        "{} must not be a symlink",
        path.display()
    );
    assert!(
        link_metadata.is_file(),
        "{} must be a regular file",
        path.display()
    );
    let mut file =
        File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let opened_metadata = file
        .metadata()
        .unwrap_or_else(|error| panic!("inspect open {}: {error}", path.display()));
    assert!(
        opened_metadata.is_file(),
        "{} changed away from a regular file while opening",
        path.display()
    );
    #[cfg(unix)]
    assert_eq!(
        (link_metadata.dev(), link_metadata.ino()),
        (opened_metadata.dev(), opened_metadata.ino()),
        "{} changed between no-follow inspection and open",
        path.display()
    );
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .unwrap_or_else(|error| panic!("hash {}: {error}", path.display()));
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    format!("{:x}", digest.finalize())
}

/// Verifies every pinned control file's regular-file status, length, and digest.
#[cfg(feature = "tokenizers")]
fn assert_control_files(root: &Path) {
    for (relative_path, expected_bytes, expected_hash) in CONTROL_FILES {
        let path = root.join(relative_path);
        let metadata = fs::symlink_metadata(&path)
            .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()));
        assert!(!metadata.file_type().is_symlink());
        assert!(metadata.is_file());
        assert_eq!(
            metadata.len(),
            expected_bytes,
            "changed control file: {relative_path}"
        );
        assert_eq!(
            sha256_file_hex(&path),
            expected_hash,
            "pinned control file changed: {relative_path}"
        );
    }
}

/// Verifies both complete multi-gigabyte payloads before real-model conclusions.
fn assert_shard_file_hashes(root: &Path) {
    for (relative_path, expected_bytes, expected_hash) in SHARDS {
        let path = root.join(relative_path);
        let metadata = fs::symlink_metadata(&path)
            .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()));
        assert!(!metadata.file_type().is_symlink());
        assert!(metadata.is_file());
        assert_eq!(
            metadata.len(),
            expected_bytes,
            "pinned shard length changed: {relative_path}"
        );
        assert_eq!(
            sha256_file_hex(&path),
            expected_hash,
            "pinned weight shard changed: {relative_path}"
        );
    }
}

/// Hashes every physical header fact in canonical tensor-name order.
#[cfg(feature = "tokenizers")]
fn header_inventory_digest(manifest: &CheckpointManifest) -> String {
    let mut canonical = Vec::new();
    for (name, tensor) in &manifest.tensors {
        canonical.extend_from_slice(name.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(tensor.shard.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(tensor.dtype.as_str().as_bytes());
        canonical.push(0);
        for dimension in &tensor.shape {
            canonical.extend_from_slice(dimension.to_string().as_bytes());
            canonical.push(b',');
        }
        canonical.push(0);
        canonical.extend_from_slice(tensor.data_offsets[0].to_string().as_bytes());
        canonical.push(b':');
        canonical.extend_from_slice(tensor.data_offsets[1].to_string().as_bytes());
        canonical.push(b'\n');
    }
    sha256_hex(&canonical)
}

/// Hashes strict target/source mapping, target dtype, and shape in plan order.
#[cfg(feature = "tokenizers")]
fn f16_plan_digest(plan: &LoadPlan) -> String {
    let mut canonical = Vec::new();
    for tensor in &plan.tensors {
        canonical.extend_from_slice(tensor.target.as_bytes());
        canonical.push(0);
        for source in &tensor.sources {
            canonical.extend_from_slice(source.name.as_bytes());
            canonical.push(0);
        }
        canonical.extend_from_slice(tensor.dtype.as_str().as_bytes());
        canonical.push(0);
        for dimension in &tensor.shape {
            canonical.extend_from_slice(dimension.to_string().as_bytes());
            canonical.push(b',');
        }
        canonical.push(b'\n');
    }
    sha256_hex(&canonical)
}

/// Returns the complete physical source-shape histogram for the pinned revision.
#[cfg(feature = "tokenizers")]
fn expected_physical_shape_counts() -> BTreeMap<Vec<u64>, usize> {
    BTreeMap::from([
        (vec![256, 6_656], 104),
        (vec![1_024, 1_536], 1),
        (vec![1_536], 454),
        (vec![1_536, 1_176], 1),
        (vec![1_536, 1_536], 200),
        (vec![1_536, 8_960], 50),
        (vec![4_096, 4_096], 1),
        (vec![4_096, 6_144], 1),
        (vec![4_096, 6_656], 104),
        (vec![6_656], 209),
        (vec![6_656, 4_096], 53),
        (vec![6_656, 19_968], 52),
        (vec![8_960], 50),
        (vec![8_960, 1_536], 50),
        (vec![19_968, 6_656], 104),
        (vec![202_048, 6_656], 2),
    ])
}

/// Returns the target-shape histogram for the executable text-only graph.
fn expected_language_shape_counts() -> BTreeMap<Vec<u64>, usize> {
    BTreeMap::from([
        (vec![256, 6_656], 104),
        (vec![4_096, 6_656], 104),
        (vec![6_656], 209),
        (vec![6_656, 4_096], 52),
        (vec![6_656, 19_968], 52),
        (vec![19_968, 6_656], 104),
        (vec![202_048, 6_656], 2),
    ])
}

/// Strict schema emitted by `generate_muse_glimmer_reference.py`.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseOfficialReference {
    schema_version: u32,
    repository: String,
    revision: String,
    reference: MuseReferenceImplementation,
    prompt: String,
    input_ids: Vec<u32>,
    execution: MuseReferenceExecution,
    environment: MuseReferenceEnvironment,
    provenance: MuseReferenceProvenance,
    outputs: MuseReferenceOutputs,
    generation: MuseReferenceGeneration,
    timings_seconds: MuseReferenceTimings,
}

/// Exact official source identity used to compute the numerical oracle.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceImplementation {
    repository: String,
    revision: String,
    class: String,
    source: MuseReferenceSource,
    checkout_clean: bool,
}

/// One pinned source file from the official Transformers checkout.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceSource {
    relative_path: String,
    bytes: u64,
    sha256: String,
}

/// Deterministic graph, dtype, cache, and accelerator choices.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceExecution {
    dtype: String,
    device: String,
    attention_implementation: String,
    attention_precision: MuseReferenceAttentionPrecision,
    hub_kernels_enabled: bool,
    deterministic_algorithms: bool,
    seed: u64,
    use_cache: bool,
    explicit_all_one_attention_masks: bool,
    float32_matmul_precision: String,
    mps_per_process_memory_fraction: f64,
    allocator_warmup_override: MuseReferenceWarmupOverride,
}

/// Official eager-attention score equation and mixed-precision boundary.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceAttentionPrecision {
    score_equation: String,
    softmax_accumulator_dtype: String,
    probability_output_dtype: String,
}

/// Audited scope of the performance-only MPS allocator warmup override.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceWarmupOverride {
    target: String,
    policy: String,
    call_count: usize,
    expanded_device_map_entries: usize,
    expanded_device_types: Vec<String>,
    hf_quantizer_present: bool,
    restored: bool,
}

/// Python process, dependency, and offline-control provenance.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceEnvironment {
    generated_at_utc: String,
    python_version: String,
    python_implementation: String,
    platform: String,
    packages: BTreeMap<String, String>,
    device: MuseReferenceDevice,
    offline_environment: BTreeMap<String, String>,
}

/// MPS availability and conservative full-model load-memory preflight.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceDevice {
    backend_available: bool,
    recommended_max_memory_bytes: u64,
    required_f16_load_bytes: u64,
    minimum_load_headroom_bytes: u64,
}

/// Generator and checkpoint identities recorded before model execution.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceProvenance {
    generator: MuseReferenceGenerator,
    artifacts: Vec<MuseReferenceArtifact>,
    artifact_manifest_sha256: String,
    physical_tensor_count: usize,
    checkpoint_tensor_bytes: u64,
    loaded_parameter_count: u64,
    parameter_dtype_elements: BTreeMap<String, u64>,
    loading_info: MuseReferenceLoadingInfo,
}

/// Exact committed generator source that produced an external bundle.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceGenerator {
    relative_path: String,
    bytes: usize,
    sha256: String,
    git_head: String,
    git_dirty: bool,
}

/// One authenticated checkpoint input.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceArtifact {
    relative_path: String,
    bytes: u64,
    sha256: String,
}

/// Strict empty-class summaries returned by official loading.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceLoadingInfo {
    missing_keys: MuseReferenceDiagnostic,
    unexpected_keys: MuseReferenceDiagnostic,
    mismatched_keys: MuseReferenceDiagnostic,
    error_msgs: MuseReferenceDiagnostic,
}

/// Count and canonical digest for one diagnostic class.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceDiagnostic {
    count: usize,
    sha256: String,
}

/// External numerical payloads used by the Rust acceptance test.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceOutputs {
    prompt_logits: MuseReferencePromptLogits,
    hidden_last_trace: MuseReferenceHiddenTrace,
}

/// Full-vocabulary F32LE prompt-logit descriptor and integrity summary.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferencePromptLogits {
    relative_path: String,
    encoding: String,
    shape: Vec<usize>,
    elements: usize,
    bytes: usize,
    sha256: String,
    finite: bool,
    sum: f64,
    l2_norm: f64,
    top10: Vec<MuseReferenceRankedLogit>,
    top1_token_id: u32,
}

/// One deterministic top-logit record with token-ID tie breaking.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceRankedLogit {
    token_id: u32,
    logit: f64,
}

/// F32LE last-token trace spanning embedding, every layer, and final norm.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceHiddenTrace {
    relative_path: String,
    encoding: String,
    shape: Vec<usize>,
    elements: usize,
    bytes: usize,
    sha256: String,
    labels: Vec<String>,
}

/// Cached and stateless top-1 evidence for 32 autoregressive prefixes.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceGeneration {
    max_new_tokens: usize,
    eos_token_ids: Vec<u32>,
    token_ids: Vec<u32>,
    cached_prefix_top1: Vec<u32>,
    stateless_prefix_top1: Vec<u32>,
    decisions: Vec<MuseReferenceDecision>,
    minimum_margin: f64,
    minimum_margin_to_epsilon_ratio: f64,
    decoded_text: String,
    prompt_cached_cosine: f64,
    minimum_cached_stateless_prefix_cosine: f64,
}

/// Binary16 perturbation evidence for one reference greedy decision.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceDecision {
    step: usize,
    top1_token_id: u32,
    top2_token_id: u32,
    top1_logit: f64,
    top2_logit: f64,
    margin: f64,
    elementwise_f16_ulp: f64,
    margin_to_epsilon_ratio: f64,
    stable_under_two_sided_one_ulp_budget: bool,
}

/// Synchronized wall-clock phases retained as non-acceptance diagnostics.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceTimings {
    load: f64,
    stateless_prompt: f64,
    cached_generation: f64,
    teacher_forced_stateless: f64,
}

/// Returns all checkpoint identities as an independently compiled Rust inventory.
fn expected_reference_artifacts() -> Vec<MuseReferenceArtifact> {
    CONTROL_FILES
        .iter()
        .chain(SHARDS.iter())
        .map(|(relative_path, bytes, sha256)| MuseReferenceArtifact {
            relative_path: (*relative_path).to_owned(),
            bytes: *bytes,
            sha256: (*sha256).to_owned(),
        })
        .collect()
}

/// Reproduces the generator's canonical checkpoint-artifact digest.
fn reference_artifact_digest(artifacts: &[MuseReferenceArtifact]) -> String {
    let mut artifacts = artifacts.iter().collect::<Vec<_>>();
    artifacts.sort_unstable_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let mut canonical = Vec::new();
    for artifact in artifacts {
        canonical.extend_from_slice(artifact.relative_path.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(artifact.bytes.to_string().as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(artifact.sha256.as_bytes());
        canonical.push(b'\n');
    }
    sha256_hex(&canonical)
}

/// Returns the exact 54 semantic trace labels in execution order.
fn expected_hidden_trace_labels() -> Vec<String> {
    std::iter::once("embed_norm".to_owned())
        .chain((0..52).map(|index| format!("layers.{index}")))
        .chain(std::iter::once("norm".to_owned()))
        .collect()
}

/// Returns whether a value is one canonical lowercase SHA-256 digest.
fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Returns whether a reference payload name is one safe ordinary component.
fn is_safe_filename(value: &str) -> bool {
    let mut components = Path::new(value).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

/// Validates the strict record without trusting any output's self-declared hash.
fn validate_reference_schema(reference: &MuseOfficialReference) -> Result<(), String> {
    let ensure = |condition: bool, message: &'static str| {
        condition.then_some(()).ok_or_else(|| message.to_owned())
    };
    ensure(
        reference.schema_version == 1,
        "reference schema version changed",
    )?;
    ensure(
        reference.repository == REPOSITORY,
        "reference repository changed",
    )?;
    ensure(reference.revision == REVISION, "reference revision changed")?;
    ensure(
        reference.reference.repository == REFERENCE_REPOSITORY,
        "official implementation repository changed",
    )?;
    ensure(
        reference.reference.revision == REFERENCE_REVISION,
        "official implementation revision changed",
    )?;
    ensure(
        reference.reference.class == "transformers.MuseGlimmerForConditionalGeneration",
        "official reference class changed",
    )?;
    ensure(
        reference.reference.checkout_clean,
        "official checkout was dirty",
    )?;
    ensure(
        reference.reference.source.relative_path == REFERENCE_SOURCE
            && reference.reference.source.bytes == REFERENCE_SOURCE_BYTES
            && reference.reference.source.sha256 == REFERENCE_SOURCE_SHA256,
        "official Muse source identity changed",
    )?;
    ensure(reference.prompt == SMOKE_PROMPT, "reference prompt changed")?;
    ensure(
        reference.input_ids == SMOKE_TOKEN_IDS,
        "reference tokenization changed",
    )?;

    let execution = &reference.execution;
    ensure(execution.dtype == "float16", "reference dtype changed")?;
    ensure(execution.device == "mps", "reference device changed")?;
    ensure(
        execution.attention_implementation == "eager",
        "reference attention implementation changed",
    )?;
    ensure(
        execution.attention_precision.score_equation
            == "(rms(q) * qk_scale_factor) @ rms(k)^T / sqrt(head_dim)"
            && execution.attention_precision.softmax_accumulator_dtype == "float32"
            && execution.attention_precision.probability_output_dtype == "float16",
        "reference attention precision contract changed",
    )?;
    ensure(
        !execution.hub_kernels_enabled,
        "reference execution enabled mutable Hub kernels",
    )?;
    ensure(
        execution.deterministic_algorithms && execution.seed == 0 && execution.use_cache,
        "deterministic execution controls changed",
    )?;
    ensure(
        execution.explicit_all_one_attention_masks
            && execution.float32_matmul_precision == "highest"
            && execution.mps_per_process_memory_fraction == 1.0,
        "reference precision or mask controls changed",
    )?;
    let warmup = &execution.allocator_warmup_override;
    ensure(
        warmup.target == "transformers.modeling_utils.caching_allocator_warmup"
            && warmup.policy == "mps-exclusive-noop-during-from-pretrained"
            && warmup.call_count == 1
            && warmup.expanded_device_map_entries > 0
            && warmup.expanded_device_types == ["mps"]
            && !warmup.hf_quantizer_present
            && warmup.restored,
        "allocator warmup override changed",
    )?;

    let environment = &reference.environment;
    ensure(
        environment.generated_at_utc.ends_with('Z')
            && !environment.python_version.is_empty()
            && environment.python_implementation == "CPython"
            && !environment.platform.is_empty(),
        "Python environment metadata changed",
    )?;
    let expected_packages = BTreeMap::from([
        ("accelerate".to_owned(), "1.12.0".to_owned()),
        ("numpy".to_owned(), "2.4.3".to_owned()),
        ("safetensors".to_owned(), "0.7.0".to_owned()),
        ("tokenizers".to_owned(), "0.22.2".to_owned()),
        ("torch".to_owned(), "2.10.0".to_owned()),
        ("transformers".to_owned(), "5.15.0.dev0".to_owned()),
    ]);
    ensure(
        environment.packages == expected_packages,
        "reference dependency versions changed",
    )?;
    ensure(
        environment.device.backend_available
            && environment.device.minimum_load_headroom_bytes == 2_000_000_000
            && environment.device.required_f16_load_bytes
                == CHECKPOINT_TENSOR_BYTES + environment.device.minimum_load_headroom_bytes
            && environment.device.recommended_max_memory_bytes
                >= environment.device.required_f16_load_bytes,
        "reference MPS memory preflight changed",
    )?;
    ensure(
        environment.offline_environment
            == BTreeMap::from([
                ("HF_DATASETS_OFFLINE".to_owned(), "1".to_owned()),
                ("HF_DEACTIVATE_ASYNC_LOAD".to_owned(), "1".to_owned()),
                ("HF_HUB_OFFLINE".to_owned(), "1".to_owned()),
                ("PYTORCH_ENABLE_MPS_FALLBACK".to_owned(), "0".to_owned()),
                ("TOKENIZERS_PARALLELISM".to_owned(), "false".to_owned()),
                ("TRANSFORMERS_OFFLINE".to_owned(), "1".to_owned()),
                ("USE_HUB_KERNELS".to_owned(), "0".to_owned()),
            ]),
        "offline environment controls changed",
    )?;

    let provenance = &reference.provenance;
    ensure(
        provenance.generator.relative_path == "scripts/generate_muse_glimmer_reference.py",
        "generator path changed",
    )?;
    let generator_bytes = include_bytes!("../scripts/generate_muse_glimmer_reference.py");
    ensure(
        provenance.generator.bytes == generator_bytes.len()
            && provenance.generator.sha256 == sha256_hex(generator_bytes)
            && provenance.generator.git_head.len() == 40
            && provenance
                .generator
                .git_head
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            && !provenance.generator.git_dirty,
        "generator provenance changed",
    )?;
    let expected_artifacts = expected_reference_artifacts()
        .into_iter()
        .map(|artifact| (artifact.relative_path.clone(), artifact))
        .collect::<BTreeMap<_, _>>();
    let observed_artifacts = provenance
        .artifacts
        .iter()
        .cloned()
        .map(|artifact| (artifact.relative_path.clone(), artifact))
        .collect::<BTreeMap<_, _>>();
    ensure(
        observed_artifacts.len() == provenance.artifacts.len()
            && observed_artifacts == expected_artifacts,
        "reference checkpoint artifact inventory changed",
    )?;
    ensure(
        reference_artifact_digest(&provenance.artifacts) == REFERENCE_ARTIFACT_MANIFEST_SHA256
            && provenance.artifact_manifest_sha256 == REFERENCE_ARTIFACT_MANIFEST_SHA256,
        "checkpoint artifact manifest digest changed",
    )?;
    ensure(
        provenance.physical_tensor_count == PHYSICAL_TENSOR_COUNT
            && provenance.checkpoint_tensor_bytes == CHECKPOINT_TENSOR_BYTES
            && provenance.loaded_parameter_count == 29_776_626_688
            && provenance.parameter_dtype_elements
                == BTreeMap::from([("float16".to_owned(), 29_776_626_688)]),
        "loaded model size or dtype inventory changed",
    )?;
    let empty_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    for diagnostic in [
        &provenance.loading_info.missing_keys,
        &provenance.loading_info.unexpected_keys,
        &provenance.loading_info.mismatched_keys,
        &provenance.loading_info.error_msgs,
    ] {
        ensure(
            diagnostic.count == 0 && diagnostic.sha256 == empty_sha,
            "official loading diagnostics were nonempty",
        )?;
    }

    let logits = &reference.outputs.prompt_logits;
    ensure(
        logits.relative_path == REFERENCE_LOGITS_FILENAME
            && is_safe_filename(&logits.relative_path)
            && logits.encoding == "ieee754-f32-little-endian"
            && logits.shape == [202_048]
            && logits.elements == 202_048
            && logits.bytes == 202_048 * size_of::<f32>()
            && is_sha256(&logits.sha256)
            && logits.finite
            && logits.sum.is_finite()
            && logits.l2_norm.is_finite()
            && logits.l2_norm > 0.0
            && logits.top10.len() == 10
            && logits
                .top10
                .iter()
                .all(|entry| entry.token_id < 202_048 && entry.logit.is_finite()),
        "prompt-logit descriptor changed",
    )?;
    let hidden = &reference.outputs.hidden_last_trace;
    ensure(
        hidden.relative_path == REFERENCE_HIDDEN_FILENAME
            && is_safe_filename(&hidden.relative_path)
            && hidden.encoding == "ieee754-f32-little-endian"
            && hidden.shape == [54, 6_656]
            && hidden.elements == 54 * 6_656
            && hidden.bytes == 54 * 6_656 * size_of::<f32>()
            && is_sha256(&hidden.sha256)
            && hidden.labels == expected_hidden_trace_labels(),
        "hidden-trace descriptor changed",
    )?;

    let generation = &reference.generation;
    ensure(
        generation.max_new_tokens == 32
            && generation.eos_token_ids == [200_001, 200_008]
            && generation.token_ids.len() == 32
            && generation.cached_prefix_top1 == generation.token_ids
            && generation.stateless_prefix_top1 == generation.token_ids
            && generation.decisions.len() == 32
            && !generation.decoded_text.is_empty()
            && generation.prompt_cached_cosine >= 0.999
            && generation.minimum_cached_stateless_prefix_cosine >= 0.999,
        "generation topology or cached/stateless parity changed",
    )?;
    ensure(
        generation
            .token_ids
            .iter()
            .all(|token| *token < 202_048 && !generation.eos_token_ids.contains(token)),
        "reference generation emitted an invalid or EOS token",
    )?;
    for (step, decision) in generation.decisions.iter().enumerate() {
        ensure(
            decision.step == step
                && decision.top1_token_id == generation.token_ids[step]
                && decision.top1_token_id != decision.top2_token_id
                && decision.top1_logit.is_finite()
                && decision.top2_logit.is_finite()
                && decision.margin.is_finite()
                && decision.margin > 0.0
                && decision.margin == decision.top1_logit - decision.top2_logit
                && decision.elementwise_f16_ulp.is_finite()
                && decision.elementwise_f16_ulp > 0.0
                && decision.margin_to_epsilon_ratio
                    == decision.margin / decision.elementwise_f16_ulp
                && decision.margin_to_epsilon_ratio > 2.0
                && decision.stable_under_two_sided_one_ulp_budget,
            "greedy stability evidence changed",
        )?;
    }
    let minimum_margin = generation
        .decisions
        .iter()
        .map(|decision| decision.margin)
        .reduce(f64::min)
        .ok_or_else(|| "missing greedy margins".to_owned())?;
    let minimum_ratio = generation
        .decisions
        .iter()
        .map(|decision| decision.margin_to_epsilon_ratio)
        .reduce(f64::min)
        .ok_or_else(|| "missing greedy ratios".to_owned())?;
    ensure(
        generation.minimum_margin == minimum_margin
            && generation.minimum_margin_to_epsilon_ratio == minimum_ratio
            && logits.top1_token_id == generation.token_ids[0]
            && logits.top10[0].token_id == logits.top1_token_id,
        "summary values do not match detailed evidence",
    )?;
    for timing in [
        reference.timings_seconds.load,
        reference.timings_seconds.stateless_prompt,
        reference.timings_seconds.cached_generation,
        reference.timings_seconds.teacher_forced_stateless,
    ] {
        ensure(timing.is_finite() && timing >= 0.0, "invalid timing")?;
    }
    Ok(())
}

/// Builds a compact, internally consistent record for weight-free schema tests.
fn synthetic_reference_json() -> serde_json::Value {
    let generator_bytes = include_bytes!("../scripts/generate_muse_glimmer_reference.py");
    let artifacts = expected_reference_artifacts()
        .into_iter()
        .map(|artifact| {
            serde_json::json!({
                "relative_path": artifact.relative_path,
                "bytes": artifact.bytes,
                "sha256": artifact.sha256,
            })
        })
        .collect::<Vec<_>>();
    let token_ids = vec![7_u32; 32];
    let decisions = (0..32)
        .map(|step| {
            serde_json::json!({
                "step": step,
                "top1_token_id": 7,
                "top2_token_id": 8,
                "top1_logit": 2.0,
                "top2_logit": 1.0,
                "margin": 1.0,
                "elementwise_f16_ulp": 0.0009765625,
                "margin_to_epsilon_ratio": 1024.0,
                "stable_under_two_sided_one_ulp_budget": true,
            })
        })
        .collect::<Vec<_>>();
    let empty_diagnostic = serde_json::json!({
        "count": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    });
    let top10 = (0..10)
        .map(|rank| {
            serde_json::json!({
                "token_id": 7 + rank,
                "logit": 2.0 - f64::from(rank) * 0.125,
            })
        })
        .collect::<Vec<_>>();

    serde_json::json!({
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "reference": {
            "repository": REFERENCE_REPOSITORY,
            "revision": REFERENCE_REVISION,
            "class": "transformers.MuseGlimmerForConditionalGeneration",
            "source": {
                "relative_path": REFERENCE_SOURCE,
                "bytes": REFERENCE_SOURCE_BYTES,
                "sha256": REFERENCE_SOURCE_SHA256,
            },
            "checkout_clean": true,
        },
        "prompt": SMOKE_PROMPT,
        "input_ids": SMOKE_TOKEN_IDS,
        "execution": {
            "dtype": "float16",
            "device": "mps",
            "attention_implementation": "eager",
            "attention_precision": {
                "score_equation": "(rms(q) * qk_scale_factor) @ rms(k)^T / sqrt(head_dim)",
                "softmax_accumulator_dtype": "float32",
                "probability_output_dtype": "float16",
            },
            "hub_kernels_enabled": false,
            "deterministic_algorithms": true,
            "seed": 0,
            "use_cache": true,
            "explicit_all_one_attention_masks": true,
            "float32_matmul_precision": "highest",
            "mps_per_process_memory_fraction": 1.0,
            "allocator_warmup_override": {
                "target": "transformers.modeling_utils.caching_allocator_warmup",
                "policy": "mps-exclusive-noop-during-from-pretrained",
                "call_count": 1,
                "expanded_device_map_entries": 627,
                "expanded_device_types": ["mps"],
                "hf_quantizer_present": false,
                "restored": true,
            },
        },
        "environment": {
            "generated_at_utc": "2026-09-05T00:00:00Z",
            "python_version": "3.14.0 fixture",
            "python_implementation": "CPython",
            "platform": "fixture",
            "packages": {
                "accelerate": "1.12.0",
                "numpy": "2.4.3",
                "safetensors": "0.7.0",
                "tokenizers": "0.22.2",
                "torch": "2.10.0",
                "transformers": "5.15.0.dev0",
            },
            "device": {
                "backend_available": true,
                "recommended_max_memory_bytes": 70_000_000_000_u64,
                "required_f16_load_bytes": CHECKPOINT_TENSOR_BYTES + 2_000_000_000,
                "minimum_load_headroom_bytes": 2_000_000_000_u64,
            },
            "offline_environment": {
                "HF_DATASETS_OFFLINE": "1",
                "HF_DEACTIVATE_ASYNC_LOAD": "1",
                "HF_HUB_OFFLINE": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
                "USE_HUB_KERNELS": "0",
            },
        },
        "provenance": {
            "generator": {
                "relative_path": "scripts/generate_muse_glimmer_reference.py",
                "bytes": generator_bytes.len(),
                "sha256": sha256_hex(generator_bytes),
                "git_head": "0000000000000000000000000000000000000000",
                "git_dirty": false,
            },
            "artifacts": artifacts,
            "artifact_manifest_sha256": REFERENCE_ARTIFACT_MANIFEST_SHA256,
            "physical_tensor_count": PHYSICAL_TENSOR_COUNT,
            "checkpoint_tensor_bytes": CHECKPOINT_TENSOR_BYTES,
            "loaded_parameter_count": 29_776_626_688_u64,
            "parameter_dtype_elements": {"float16": 29_776_626_688_u64},
            "loading_info": {
                "missing_keys": empty_diagnostic.clone(),
                "unexpected_keys": empty_diagnostic.clone(),
                "mismatched_keys": empty_diagnostic.clone(),
                "error_msgs": empty_diagnostic,
            },
        },
        "outputs": {
            "prompt_logits": {
                "relative_path": REFERENCE_LOGITS_FILENAME,
                "encoding": "ieee754-f32-little-endian",
                "shape": [202_048],
                "elements": 202_048,
                "bytes": 202_048 * size_of::<f32>(),
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "finite": true,
                "sum": 0.0,
                "l2_norm": 1.0,
                "top10": top10,
                "top1_token_id": 7,
            },
            "hidden_last_trace": {
                "relative_path": REFERENCE_HIDDEN_FILENAME,
                "encoding": "ieee754-f32-little-endian",
                "shape": [54, 6_656],
                "elements": 54 * 6_656,
                "bytes": 54 * 6_656 * size_of::<f32>(),
                "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
                "labels": expected_hidden_trace_labels(),
            },
        },
        "generation": {
            "max_new_tokens": 32,
            "eos_token_ids": [200_001, 200_008],
            "token_ids": token_ids.clone(),
            "cached_prefix_top1": token_ids.clone(),
            "stateless_prefix_top1": token_ids,
            "decisions": decisions,
            "minimum_margin": 1.0,
            "minimum_margin_to_epsilon_ratio": 1024.0,
            "decoded_text": "fixture",
            "prompt_cached_cosine": 1.0,
            "minimum_cached_stateless_prefix_cosine": 1.0,
        },
        "timings_seconds": {
            "load": 0.0,
            "stateless_prompt": 0.0,
            "cached_generation": 0.0,
            "teacher_forced_stateless": 0.0,
        },
    })
}

/// Proves schema, digest, strictness, and tamper detection without model files.
#[test]
fn muse_official_reference_schema_is_weight_independent_and_strict() {
    let expected_artifacts = expected_reference_artifacts();
    assert_eq!(expected_artifacts.len(), CONTROL_FILES.len() + SHARDS.len());
    assert_eq!(
        reference_artifact_digest(&expected_artifacts),
        REFERENCE_ARTIFACT_MANIFEST_SHA256
    );
    assert_eq!(expected_hidden_trace_labels().len(), 54);

    let schema = synthetic_reference_json();
    let reference: MuseOfficialReference =
        serde_json::from_value(schema.clone()).expect("synthetic strict schema must deserialize");
    validate_reference_schema(&reference).expect("synthetic strict schema must validate");

    let mut unknown = schema.clone();
    unknown
        .as_object_mut()
        .expect("root object")
        .insert("unreviewed".to_owned(), serde_json::json!(true));
    assert!(
        serde_json::from_value::<MuseOfficialReference>(unknown).is_err(),
        "unknown fields must fail closed"
    );

    let mut tampered = schema;
    tampered["generation"]["cached_prefix_top1"] = serde_json::json!([7]);
    let reference: MuseOfficialReference =
        serde_json::from_value(tampered).expect("structurally valid tamper must deserialize");
    assert!(
        validate_reference_schema(&reference)
            .expect_err("semantic tamper must fail")
            .contains("generation topology"),
        "semantic tampering must produce the expected failure class"
    );
}

/// Resolves the opt-in external parent containing the fixed reference bundle.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn pinned_reference_root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("TRANSFORMERS_MUSE_GLIMMER_REFERENCE_DIR").expect(
            "set TRANSFORMERS_MUSE_GLIMMER_REFERENCE_DIR to the external parent passed to \
             scripts/generate_muse_glimmer_reference.py",
        ),
    )
    .join(REFERENCE_BUNDLE_DIRECTORY)
}

/// Opens an exact-size reference file without following its final component.
#[cfg(all(feature = "metal", feature = "tokenizers", target_os = "macos"))]
fn open_reference_file_no_follow(path: &Path, expected_bytes: u64, maximum_bytes: u64) -> File {
    const DARWIN_O_NOFOLLOW: i32 = 0x0000_0100;
    const DARWIN_O_NONBLOCK: i32 = 0x0000_0004;

    assert!(
        expected_bytes <= maximum_bytes,
        "expected {expected_bytes} bytes exceeds the {maximum_bytes}-byte cap for {}",
        path.display()
    );
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(DARWIN_O_NOFOLLOW | DARWIN_O_NONBLOCK)
        .open(path)
        .unwrap_or_else(|error| {
            panic!(
                "open reference file {} without following links: {error}",
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

/// Keeps unsupported host compilation explicit for the macOS-only Metal gate.
#[cfg(all(feature = "metal", feature = "tokenizers", not(target_os = "macos")))]
fn open_reference_file_no_follow(_path: &Path, _expected_bytes: u64, _maximum_bytes: u64) -> File {
    panic!("the Muse-Glimmer Metal reference gate requires macOS O_NOFOLLOW semantics")
}

/// Inspects one small record while retaining a hard allocation ceiling.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn bounded_reference_file_length(path: &Path, maximum_bytes: u64) -> u64 {
    let metadata = fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("inspect bounded file {}: {error}", path.display()));
    assert!(
        metadata.file_type().is_file(),
        "{} must be a non-symlink regular file",
        path.display()
    );
    assert!(
        metadata.len() <= maximum_bytes,
        "{} is {} bytes, exceeding the {maximum_bytes}-byte cap",
        path.display(),
        metadata.len()
    );
    metadata.len()
}

/// Reads one bounded regular file and rejects concurrent length changes.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn read_reference_file_exact(path: &Path, expected_bytes: u64, maximum_bytes: u64) -> Vec<u8> {
    let file = open_reference_file_no_follow(path, expected_bytes, maximum_bytes);
    let capacity = usize::try_from(expected_bytes)
        .expect("bounded Muse reference length must fit the host address space");
    let mut payload = Vec::with_capacity(capacity);
    file.take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut payload)
        .unwrap_or_else(|error| panic!("read reference file {}: {error}", path.display()));
    assert_eq!(
        payload.len(),
        capacity,
        "{} changed length while being read",
        path.display()
    );
    payload
}

/// Fails closed until an independent reviewer pins one generated output hash.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn assert_trusted_reference_digest(label: &str, observed: &str, trusted: Option<&str>) {
    let trusted = trusted.unwrap_or_else(|| {
        panic!(
            "trusted {label} SHA-256 has not been pinned; independently review the generated \
             bundle, then set its narrowly scoped constant in tests/muse_glimmer_artifacts.rs \
             (observed {observed})"
        )
    });
    assert!(is_sha256(trusted), "trusted {label} SHA-256 is malformed");
    assert_eq!(observed, trusted, "generated {label} identity changed");
}

/// Decodes a complete little-endian binary32 payload without native casts.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn decode_f32le(label: &str, payload: &[u8]) -> Vec<f32> {
    assert_eq!(
        payload.len() % size_of::<f32>(),
        0,
        "{label} payload is not aligned to F32 elements"
    );
    payload
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte F32 element")))
        .collect()
}

/// Computes deterministic descending token ranks with low-ID tie breaking.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn ranked_token_ids(values: &[f32], count: usize) -> Vec<usize> {
    assert!(count > 0 && count <= values.len());
    let mut ids = (0..values.len()).collect::<Vec<_>>();
    ids.sort_unstable_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    ids.truncate(count);
    ids
}

/// Checks a recomputed floating summary with a conservative summation tolerance.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn assert_reference_close(label: &str, actual: f64, expected: f64) {
    let tolerance = expected.abs().max(1.0) * 1.0e-6;
    assert!(
        actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= tolerance,
        "{label} mismatch: expected {expected}, found {actual}, tolerance {tolerance}"
    );
}

/// Reads and independently authenticates the complete external oracle bundle.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn read_pinned_muse_reference() -> (MuseOfficialReference, Vec<f32>, Vec<f32>) {
    const MAXIMUM_JSON_BYTES: u64 = 1024 * 1024;

    let supplied_root = pinned_reference_root();
    let metadata = fs::symlink_metadata(&supplied_root).unwrap_or_else(|error| {
        panic!(
            "inspect Muse reference bundle {}: {error}",
            supplied_root.display()
        )
    });
    assert!(
        metadata.file_type().is_dir(),
        "Muse reference bundle must be a non-symlink directory"
    );
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert!(
        !root.starts_with(env!("CARGO_MANIFEST_DIR")),
        "generated numerical references must remain outside the crate"
    );

    let json_path = root.join(REFERENCE_JSON_FILENAME);
    let json_length = bounded_reference_file_length(&json_path, MAXIMUM_JSON_BYTES);
    let json = read_reference_file_exact(&json_path, json_length, MAXIMUM_JSON_BYTES);
    let json_digest = sha256_hex(&json);
    assert_trusted_reference_digest(
        "Muse reference JSON",
        &json_digest,
        TRUSTED_MUSE_REFERENCE_JSON_SHA256,
    );
    let reference: MuseOfficialReference = serde_json::from_slice(&json)
        .unwrap_or_else(|error| panic!("parse strict reference {}: {error}", json_path.display()));
    validate_reference_schema(&reference)
        .unwrap_or_else(|error| panic!("validate strict Muse reference: {error}"));

    let logits_path = root.join(&reference.outputs.prompt_logits.relative_path);
    let logits_bytes = u64::try_from(reference.outputs.prompt_logits.bytes)
        .expect("prompt-logit byte count fits u64");
    let raw_logits = read_reference_file_exact(&logits_path, logits_bytes, logits_bytes);
    let logits_digest = sha256_hex(&raw_logits);
    assert_trusted_reference_digest(
        "Muse reference prompt logits",
        &logits_digest,
        TRUSTED_MUSE_REFERENCE_LOGITS_SHA256,
    );
    assert_eq!(
        logits_digest, reference.outputs.prompt_logits.sha256,
        "trusted raw logit identity differs from the JSON descriptor"
    );
    let logits = decode_f32le("Muse prompt logits", &raw_logits);
    assert_eq!(logits.len(), reference.outputs.prompt_logits.elements);
    assert!(logits.iter().all(|value| value.is_finite()));
    let ranked = ranked_token_ids(&logits, 10);
    for (record, token_id) in reference
        .outputs
        .prompt_logits
        .top10
        .iter()
        .zip(ranked.iter().copied())
    {
        assert_eq!(usize::try_from(record.token_id).unwrap(), token_id);
        assert_eq!(record.logit as f32, logits[token_id]);
    }
    assert_eq!(
        usize::try_from(reference.outputs.prompt_logits.top1_token_id).unwrap(),
        ranked[0]
    );
    let logits_sum = logits.iter().map(|value| f64::from(*value)).sum::<f64>();
    let logits_l2 = logits
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    assert_reference_close(
        "Muse prompt-logit sum",
        logits_sum,
        reference.outputs.prompt_logits.sum,
    );
    assert_reference_close(
        "Muse prompt-logit L2 norm",
        logits_l2,
        reference.outputs.prompt_logits.l2_norm,
    );

    let hidden_path = root.join(&reference.outputs.hidden_last_trace.relative_path);
    let hidden_bytes = u64::try_from(reference.outputs.hidden_last_trace.bytes)
        .expect("hidden-trace byte count fits u64");
    let raw_hidden = read_reference_file_exact(&hidden_path, hidden_bytes, hidden_bytes);
    let hidden_digest = sha256_hex(&raw_hidden);
    assert_trusted_reference_digest(
        "Muse reference hidden trace",
        &hidden_digest,
        TRUSTED_MUSE_REFERENCE_HIDDEN_SHA256,
    );
    assert_eq!(
        hidden_digest, reference.outputs.hidden_last_trace.sha256,
        "trusted hidden identity differs from the JSON descriptor"
    );
    let hidden = decode_f32le("Muse hidden trace", &raw_hidden);
    assert_eq!(hidden.len(), reference.outputs.hidden_last_trace.elements);
    assert!(hidden.iter().all(|value| value.is_finite()));
    (reference, logits, hidden)
}

/// Materializes one Metal tensor as synchronized F32 acceptance values.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn metal_values<const D: usize>(label: &str, tensor: &Tensor<Metal, D>) -> Vec<f32> {
    tensor
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .unwrap_or_else(|error| panic!("materialize {label}: {error}"))
}

/// Computes finite cosine similarity with F64 accumulation.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len());
    let (dot, left_square, right_square) = left.iter().zip(right).fold(
        (0.0_f64, 0.0_f64, 0.0_f64),
        |(dot, left_square, right_square), (left, right)| {
            let left = f64::from(*left);
            let right = f64::from(*right);
            (
                dot + left * right,
                left_square + left * left,
                right_square + right * right,
            )
        },
    );
    assert!(left_square > 0.0 && right_square > 0.0);
    let cosine = dot / (left_square.sqrt() * right_square.sqrt());
    assert!(cosine.is_finite());
    cosine
}

/// Creates the explicit all-valid token mask promised by the oracle schema.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn all_valid_metal_mask(
    sequence: usize,
    device: &<Metal as burn::tensor::backend::BackendTypes>::Device,
) -> TokenMask<Metal> {
    TokenMask::new(Tensor::<Metal, 2, Bool>::from_data(
        TensorData::new(vec![true; sequence], [1, sequence]),
        device,
    ))
}

/// Proves the exact committed config, layer schedule, generation policy, and mapping.
#[test]
fn committed_muse_glimmer_fixture_has_exact_schedule_and_contract() {
    let config_bytes = include_bytes!("fixtures/muse_glimmer_30b.json");
    let generation_bytes = include_bytes!("fixtures/muse_glimmer_30b_generation.json");
    assert_eq!(config_bytes.len(), 5_109);
    assert_eq!(generation_bytes.len(), 202);
    assert_eq!(sha256_hex(config_bytes), CONFIG_SHA256);
    assert_eq!(sha256_hex(generation_bytes), GENERATION_CONFIG_SHA256);

    let config = MuseGlimmerTextConfig::from_json_str(
        std::str::from_utf8(config_bytes).expect("config fixture must be UTF-8"),
    )
    .expect("the pinned Muse-Glimmer config fixture must validate");
    assert_eq!(config.model_type, "muse_glimmer_text");
    assert_eq!(config.vocab_size, 202_048);
    assert_eq!(config.hidden_size, 6_656);
    assert_eq!(config.intermediate_size, 19_968);
    assert_eq!(config.num_hidden_layers, 52);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 2);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.max_position_embeddings, 131_072);
    assert_eq!(config.sliding_window, Some(2_048));
    assert_eq!(config.layer_types.len(), 52);
    assert_eq!(config.layer_rope_theta.len(), 52);
    for (layer_index, (layer_type, theta)) in config
        .layer_types
        .iter()
        .zip(&config.layer_rope_theta)
        .enumerate()
    {
        if layer_index % 4 == 3 {
            assert_eq!(*layer_type, MuseGlimmerLayerType::FullAttention);
            assert_eq!(*theta, 0.0);
        } else {
            assert_eq!(*layer_type, MuseGlimmerLayerType::SlidingAttention);
            assert_eq!(*theta, 500_000.0);
        }
    }
    assert_eq!(config.resolved_layer_types(), config.layer_types);
    assert_eq!(config.resolved_layer_rope_theta(), config.layer_rope_theta);

    let raw: serde_json::Value =
        serde_json::from_slice(config_bytes).expect("config fixture must be valid JSON");
    assert_eq!(
        raw["architectures"],
        serde_json::json!(["MuseGlimmerForConditionalGeneration"])
    );
    assert_eq!(raw["dtype"], "bfloat16");
    assert_eq!(raw["transformers_version"], "5.15.0.dev0");
    assert_eq!(
        raw["vision_config"]["layer_types"]
            .as_array()
            .unwrap()
            .len(),
        50
    );

    let generation: serde_json::Value =
        serde_json::from_slice(generation_bytes).expect("generation fixture must be valid JSON");
    assert_eq!(generation["bos_token_id"], 200_000);
    assert_eq!(
        generation["eos_token_id"],
        serde_json::json!([200_001, 200_008])
    );
    assert_eq!(generation["pad_token_id"], 200_018);
    assert_eq!(generation["max_length"], 131_072);
    assert_eq!(generation["do_sample"], true);
    assert_eq!(generation["temperature"], 1.0);
    assert_eq!(generation["top_p"], 0.95);
    assert_eq!(generation["top_k"], 64);
    assert!(generation.get("max_new_tokens").is_none());

    let contract = config
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .expect("the pinned topology must produce a checkpoint contract");
    assert_eq!(contract.expected_tensors().len(), CONSUMED_TENSOR_COUNT);
    let expected = contract
        .expected_tensors()
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.shape.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let mut mappings = BTreeMap::new();
    let mut sources = BTreeSet::new();
    let mut renames = 0_usize;
    for operation in contract.mappings().operations() {
        match operation {
            TensorOperation::Map {
                source,
                target,
                transform,
            } => {
                assert_eq!(*transform, TensorTransform::Identity);
                assert!(sources.insert(source.as_str()), "duplicate source {source}");
                renames += usize::from(source != target);
                assert!(
                    mappings
                        .insert(
                            target.as_str(),
                            (source.as_str(), expected[target.as_str()])
                        )
                        .is_none(),
                    "duplicate target {target}"
                );
            }
            operation => panic!("Muse must use one-to-one mappings, found {operation:?}"),
        }
    }
    assert_eq!(mappings.len(), CONSUMED_TENSOR_COUNT);
    assert_eq!(sources.len(), CONSUMED_TENSOR_COUNT);
    assert_eq!(renames, RENAMED_TENSOR_COUNT);
    let mut canonical = Vec::new();
    for (target, (source, shape)) in mappings {
        canonical.extend_from_slice(target.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(source.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(TensorDType::F16.as_str().as_bytes());
        canonical.push(0);
        for dimension in shape {
            canonical.extend_from_slice(dimension.to_string().as_bytes());
            canonical.push(b',');
        }
        canonical.push(b'\n');
    }
    assert_eq!(sha256_hex(&canonical), F16_PLAN_SHA256);
    let target_shapes = contract.expected_tensors().iter().fold(
        BTreeMap::<Vec<u64>, usize>::new(),
        |mut counts, tensor| {
            *counts.entry(tensor.shape.clone()).or_default() += 1;
            counts
        },
    );
    assert_eq!(target_shapes, expected_language_shape_counts());
    assert_eq!(REFERENCE_REPOSITORY, "huggingface/transformers");
    assert_eq!(REFERENCE_REVISION.len(), 40);
    assert_eq!(
        REFERENCE_SOURCE,
        "src/transformers/models/muse_glimmer/modeling_muse_glimmer.py"
    );
    assert_eq!(REFERENCE_SOURCE_BYTES, 53_259);
    assert_eq!(REFERENCE_SOURCE_SHA256.len(), 64);
}

/// Certifies every official shard payload independently of header metadata.
#[test]
#[ignore = "streams and hashes all 59.6 GB of pinned Muse-Glimmer-30B weights"]
fn validates_pinned_muse_glimmer_full_shard_payload_hashes() {
    let started = Instant::now();
    let supplied_root = pinned_root();
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert_shard_file_hashes(&root);
    eprintln!(
        "Muse-Glimmer-30B payload gate: repository={REPOSITORY}, revision={REVISION}, \
         shard_bytes={CHECKPOINT_ARTIFACT_BYTES}, elapsed_seconds={:.3}",
        started.elapsed().as_secs_f64()
    );
}

/// Certifies exact config, bounded headers, plan, memory, and tokenizer metadata.
#[cfg(feature = "tokenizers")]
#[test]
#[ignore = "requires meta-models/Muse-Glimmer-30B revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68"]
fn validates_pinned_muse_glimmer_config_headers_plan_memory_and_tokenizer() {
    let started = Instant::now();
    let supplied_root = pinned_root();
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert!(root.is_dir(), "pinned artifact root must be a directory");
    assert_control_files(&root);
    assert_eq!(
        fs::read(root.join("config.json")).expect("read pinned config"),
        include_bytes!("fixtures/muse_glimmer_30b.json")
    );
    assert_eq!(
        fs::read(root.join("generation_config.json")).expect("read pinned generation config"),
        include_bytes!("fixtures/muse_glimmer_30b_generation.json")
    );

    let config = MuseGlimmerTextConfig::from_json_str(
        std::str::from_utf8(include_bytes!("fixtures/muse_glimmer_30b.json"))
            .expect("config fixture UTF-8"),
    )
    .expect("the pinned Muse config must validate");
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        ..LoadOptions::default()
    };
    let manifest = CheckpointManifest::discover(&root, &options)
        .expect("all pinned SafeTensors headers must pass strict discovery");
    assert_eq!(manifest.root, root);
    match &manifest.source {
        ArtifactSource::Index {
            relative_path,
            declared_total_size,
        } => {
            assert_eq!(relative_path, "model.safetensors.index.json");
            assert_eq!(*declared_total_size, Some(CHECKPOINT_TENSOR_BYTES));
        }
        ArtifactSource::Single { relative_path } => {
            panic!("expected the pinned shard index, found {relative_path}")
        }
        _ => panic!("expected the standard indexed SafeTensors artifact"),
    }
    assert_eq!(manifest.shards.len(), 2);
    assert_eq!(manifest.tensors.len(), PHYSICAL_TENSOR_COUNT);
    assert_eq!(manifest.tensor_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(manifest.artifact_bytes, CHECKPOINT_ARTIFACT_BYTES);
    assert_eq!(header_inventory_digest(&manifest), HEADER_INVENTORY_SHA256);
    assert_eq!(
        manifest
            .shards
            .iter()
            .map(|shard| (shard.relative_path.as_str(), shard.file_bytes))
            .collect::<Vec<_>>(),
        SHARDS
            .iter()
            .map(|(name, bytes, _)| (*name, *bytes))
            .collect::<Vec<_>>()
    );
    assert!(manifest.shards.iter().all(|shard| {
        shard.metadata == BTreeMap::from([("format".to_owned(), "pt".to_owned())])
    }));
    assert!(
        manifest
            .tensors
            .values()
            .all(|tensor| tensor.dtype == TensorDType::BF16)
    );
    let physical_shapes =
        manifest
            .tensors
            .values()
            .fold(BTreeMap::<Vec<u64>, usize>::new(), |mut counts, tensor| {
                *counts.entry(tensor.shape.clone()).or_default() += 1;
                counts
            });
    assert_eq!(physical_shapes, expected_physical_shape_counts());

    let contract = config
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .expect("the pinned topology must produce its exact parameter contract");
    let plan = contract
        .build_load_plan(&manifest, &options)
        .expect("every language tensor must map under strict F16 planning");
    assert_eq!(plan.parameter_count(), CONSUMED_TENSOR_COUNT);
    assert_eq!(plan.tensors.len(), CONSUMED_TENSOR_COUNT);
    assert!(plan.aliases.is_empty());
    assert!(plan.equivalences.is_empty());
    assert!(plan.report.is_strictly_valid());
    assert!(
        plan.report.applied.is_empty(),
        "pass one plans tensors but cannot report pass-two application"
    );
    assert_eq!(plan.report.transformed.len(), RENAMED_TENSOR_COUNT);
    assert!(
        plan.report
            .transformed
            .iter()
            .all(|record| record.operation == "rename")
    );
    assert_eq!(plan.report.dtype_converted.len(), CONSUMED_TENSOR_COUNT);
    assert!(plan.report.dtype_converted.iter().all(|conversion| {
        conversion.source == TensorDType::BF16 && conversion.target_dtype == TensorDType::F16
    }));
    assert_eq!(plan.report.allowlisted_ignored.len(), VISUAL_TENSOR_COUNT);
    assert!(plan.report.allowlisted_missing.is_empty());
    assert!(plan.report.missing.is_empty());
    assert!(plan.report.unexpected.is_empty());
    assert!(plan.tensors.iter().all(|tensor| {
        tensor.sources.len() == 1
            && tensor.dtype == TensorDType::F16
            && matches!(tensor.transform, PlannedTransform::Identity)
    }));
    assert_eq!(f16_plan_digest(&plan), F16_PLAN_SHA256);

    let planned_sources = plan
        .tensors
        .iter()
        .map(|tensor| tensor.sources[0].name.as_str())
        .collect::<BTreeSet<_>>();
    let allowlisted = plan
        .report
        .allowlisted_ignored
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let physical_sources = manifest
        .tensors
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert!(planned_sources.is_disjoint(&allowlisted));
    assert_eq!(planned_sources.len(), CONSUMED_TENSOR_COUNT);
    assert_eq!(allowlisted.len(), VISUAL_TENSOR_COUNT);
    assert_eq!(
        planned_sources
            .union(&allowlisted)
            .copied()
            .collect::<BTreeSet<_>>(),
        physical_sources
    );
    assert!(allowlisted.iter().all(|name| {
        name.starts_with("model.vision_tower.")
            || name.starts_with("model.vision_adapter.")
            || name.starts_with("model.vision_projection.")
    }));

    assert_eq!(plan.memory.artifact_bytes, CHECKPOINT_ARTIFACT_BYTES);
    assert_eq!(plan.memory.checkpoint_tensor_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(plan.memory.target_parameter_bytes, LANGUAGE_PARAMETER_BYTES);
    assert_eq!(plan.memory.logical_model_bytes, LANGUAGE_PARAMETER_BYTES);
    assert_eq!(plan.memory.largest_shard_bytes, SHARDS[0].1);
    assert_eq!(
        plan.memory.largest_source_tensor_bytes,
        LARGEST_TARGET_PAYLOAD_BYTES
    );
    assert_eq!(
        plan.memory.largest_target_tensor_bytes,
        LARGEST_TARGET_PAYLOAD_BYTES
    );
    assert_eq!(plan.memory.peak_host_staging_bytes, PEAK_HOST_STAGING_BYTES);
    assert_eq!(plan.report.memory, Some(plan.memory));

    let tokenizer = LocalTokenizer::from_directory(&root)
        .expect("the pinned tokenizer artifacts must load without network access");
    assert_eq!(tokenizer.root(), root);
    assert_eq!(
        tokenizer.metadata().tokenizer_class.as_deref(),
        Some("TokenizersBackend")
    );
    assert_eq!(tokenizer.metadata().model_max_length, Some(131_072));
    assert_eq!(tokenizer.metadata().vocabulary_size, 202_048);
    assert_eq!(
        tokenizer
            .metadata()
            .chat_templates
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["default"]
    );
    assert_eq!(
        tokenizer
            .metadata()
            .special_tokens
            .bos
            .as_ref()
            .map(|token| token.id),
        Some(200_000)
    );
    assert_eq!(
        tokenizer
            .metadata()
            .special_tokens
            .eos
            .as_ref()
            .map(|token| token.id),
        Some(200_001)
    );
    assert_eq!(
        tokenizer
            .metadata()
            .special_tokens
            .pad
            .as_ref()
            .map(|token| token.id),
        Some(200_018)
    );
    let encoding = tokenizer
        .encode(
            SMOKE_PROMPT,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("the committed smoke prompt must encode");
    assert_eq!(encoding.token_ids()[0], SMOKE_TOKEN_IDS);
    assert_eq!(encoding.valid_tokens()[0], [true; SMOKE_TOKEN_IDS.len()]);
    assert_eq!(
        tokenizer
            .decode(&SMOKE_TOKEN_IDS, true)
            .expect("the pinned prompt IDs must decode"),
        SMOKE_PROMPT
    );

    #[cfg(feature = "chat-templates")]
    {
        let expected = concat!(
            "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\n",
            "Knowledge cutoff: 2026-01-04.\n",
            "Current date: 2026-09-05.\n\n",
            "Reasoning strength: low.\n\n",
            "# Valid recipients: \"self\", \"user\".<|eot|>",
            "<|start|>user<|message|>Hello world, cached decoding.<|eot|>",
            "<|start|>assistant"
        );
        let result = tokenizer.render_chat(
            &[ChatMessage::text("user", SMOKE_PROMPT)],
            &ChatTemplateOptions {
                add_generation_prompt: true,
                extra_context: BTreeMap::from([
                    ("current_date".into(), serde_json::json!("2026-09-05")),
                    ("reasoning_strength".into(), serde_json::json!("low")),
                ]),
                ..ChatTemplateOptions::default()
            },
        );
        match result {
            Ok(rendered) => assert_eq!(rendered, expected),
            Err(error) => {
                let message = error.to_string();
                assert!(
                    message.contains("syntax error: unexpected identifier")
                        && message.contains("<string>:164"),
                    "the exact template must render or fail only at its known unsupported Jinja conditional expression: {message}"
                );
            }
        }
    }

    eprintln!(
        "Muse-Glimmer-30B header gate: repository={REPOSITORY}, revision={REVISION}, \
         reference={REFERENCE_REPOSITORY}@{REFERENCE_REVISION}:{REFERENCE_SOURCE}, \
         reference_bytes={REFERENCE_SOURCE_BYTES}, reference_sha256={REFERENCE_SOURCE_SHA256}, \
         shards=2, tensors={PHYSICAL_TENSOR_COUNT}, consumed={CONSUMED_TENSOR_COUNT}, \
         allowlisted_visual={VISUAL_TENSOR_COUNT}, header_inventory_sha256={HEADER_INVENTORY_SHA256}, \
         f16_plan_sha256={F16_PLAN_SHA256}, f16_target_bytes={}, largest_target_bytes={}, \
         peak_host_staging_bytes={}, prompt_ids={:?}, elapsed_seconds={:.3}",
        plan.memory.target_parameter_bytes,
        plan.memory.largest_target_tensor_bytes,
        plan.memory.peak_host_staging_bytes,
        encoding.token_ids()[0],
        started.elapsed().as_secs_f64(),
    );
}

/// Strictly streams the F16 text graph and audits source order and generation defaults.
///
/// This intentionally stops before numerical reference-logit acceptance. Its
/// purpose is to prove pass-two accounting against complete, hash-verified local
/// artifacts and to keep generation's upstream `max_length` distinct from this
/// crate's conservative `max_new_tokens` default.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
#[test]
#[ignore = "requires pinned Muse-Glimmer-30B weights, Metal, and about 70 GiB free unified memory"]
fn loads_pinned_muse_glimmer_f16_and_audits_streaming() {
    let supplied_root = pinned_root();
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert_control_files(&root);
    assert_shard_file_hashes(&root);
    let device = Default::default();
    let loaded = AutoCausalLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            load_tokenizer: true,
            repository: Some(REPOSITORY.to_owned()),
            revision: Some(REVISION.to_owned()),
            ..AutoLoadOptions::default()
        },
        &device,
    )
    .expect("the public auto loader must strictly stream the complete Muse text graph");
    <Metal<f32, i32> as Backend>::sync(&device)
        .expect("all Metal parameter transfers must complete");
    assert!(matches!(loaded.model(), AutoCausalLm::MuseGlimmer(_)));
    assert_eq!(loaded.source().model_type(), "muse_glimmer_text");
    assert_eq!(loaded.source().repository(), Some(REPOSITORY));
    assert_eq!(loaded.source().revision(), Some(REVISION));
    assert_eq!(loaded.source().root(), root);

    let report = loaded.load_report();
    assert!(report.is_strictly_valid());
    assert_eq!(report.applied.len(), CONSUMED_TENSOR_COUNT);
    assert_eq!(report.transformed.len(), RENAMED_TENSOR_COUNT);
    assert_eq!(report.dtype_converted.len(), CONSUMED_TENSOR_COUNT);
    assert_eq!(report.allowlisted_ignored.len(), VISUAL_TENSOR_COUNT);
    assert!(report.aliased.is_empty());
    assert!(report.allowlisted_missing.is_empty());
    let execution = loaded
        .execution_stats()
        .expect("the automatic loader must retain pass-two counters");
    assert_eq!(execution.tensors_applied, CONSUMED_TENSOR_COUNT);
    assert_eq!(execution.aliases_applied, 0);
    assert_eq!(execution.source_reads, CONSUMED_TENSOR_COUNT as u64);
    assert_eq!(execution.source_payload_bytes, LANGUAGE_PARAMETER_BYTES);
    assert_eq!(execution.delivered_tensor_bytes, LANGUAGE_PARAMETER_BYTES);
    assert_eq!(
        execution.largest_source_payload_bytes,
        LARGEST_TARGET_PAYLOAD_BYTES
    );
    assert_eq!(
        execution.largest_output_tensor_bytes,
        LARGEST_TARGET_PAYLOAD_BYTES
    );
    assert_eq!(
        execution.peak_host_staging_bytes,
        LARGEST_TARGET_PAYLOAD_BYTES
    );
    assert_eq!(execution.source.shard_opens, 2);
    assert_eq!(execution.source.tensor_reads, CONSUMED_TENSOR_COUNT as u64);
    assert_eq!(
        execution.source.payload_bytes_read,
        LANGUAGE_PARAMETER_BYTES
    );
    assert_eq!(execution.source.max_open_shards, 1);
    assert_eq!(
        execution.source.shard_open_order,
        SHARDS
            .iter()
            .map(|(name, _, _)| (*name).to_owned())
            .collect::<Vec<_>>()
    );

    let generation = loaded.generation_defaults();
    assert_eq!(generation.max_new_tokens, 256);
    assert_eq!(generation.strategy, SamplingStrategy::Sample);
    assert_eq!(generation.temperature, 1.0);
    assert_eq!(generation.top_k, Some(64));
    assert_eq!(generation.top_p, Some(0.95));
    assert_eq!(generation.eos_token_ids, [200_001, 200_008]);
    assert_eq!(generation.pad_token_id, Some(200_018));
    let encoding = loaded
        .tokenizer()
        .expect("the loaded bundle must retain the local tokenizer")
        .encode(
            SMOKE_PROMPT,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("the pinned smoke prompt must encode");
    assert_eq!(encoding.token_ids()[0], SMOKE_TOKEN_IDS);
}

/// Compares Burn Metal with the authenticated official eager/F16 MPS oracle.
///
/// In addition to full-vocabulary prompt logits, this checks every semantic
/// last-token hidden row, all 32 recurrent greedy choices, and the same 32
/// choices from one stateless teacher-forced graph. Checkpoint controls and both
/// 59.6 GB shard payloads are authenticated before Burn sees the artifacts.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
#[test]
#[ignore = "requires reviewed Muse oracle hashes, pinned weights, Metal, and about 70 GiB free unified memory"]
fn matches_pinned_muse_glimmer_official_f16_reference_on_metal() {
    let (reference, reference_logits, reference_hidden) = read_pinned_muse_reference();
    let supplied_root = pinned_root();
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert_control_files(&root);
    assert_shard_file_hashes(&root);

    let device = Default::default();
    let loaded = AutoCausalLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            load_tokenizer: false,
            repository: Some(reference.repository.clone()),
            revision: Some(reference.revision.clone()),
            ..AutoLoadOptions::default()
        },
        &device,
    )
    .expect("strictly stream the pinned Muse text graph into Metal F16");
    <Metal<f32, i32> as Backend>::sync(&device)
        .expect("all Metal parameter transfers must complete");
    assert!(matches!(loaded.model(), AutoCausalLm::MuseGlimmer(_)));
    assert!(loaded.load_report().is_strictly_valid());
    assert_eq!(loaded.load_report().applied.len(), CONSUMED_TENSOR_COUNT);
    assert_eq!(loaded.load_report().transformed.len(), RENAMED_TENSOR_COUNT);
    assert_eq!(
        loaded.load_report().dtype_converted.len(),
        CONSUMED_TENSOR_COUNT
    );
    assert_eq!(loaded.load_report().allowlisted_ignored.len(), 809);

    let prompt_length = reference.input_ids.len();
    let prompt_ids = reference
        .input_ids
        .iter()
        .copied()
        .map(i64::from)
        .collect::<Vec<_>>();
    let prompt_tensor = Tensor::<Metal, 2, Int>::from_data(
        TensorData::new(prompt_ids.clone(), [1, prompt_length]),
        &device,
    );
    let prompt_mask = all_valid_metal_mask(prompt_length, &device);
    let diagnostics = ForwardOptions {
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let stateless = loaded
        .model()
        .forward(
            CausalLmInput {
                token_ids: prompt_tensor.clone(),
                token_mask: Some(prompt_mask.clone()),
                position_ids: None,
            },
            &diagnostics,
        )
        .expect("run stateless Burn Muse prompt inference");
    assert_eq!(stateless.logits.dims(), [1, 1, 202_048]);
    assert_eq!(stateless.hidden_states.len(), 54);
    let actual_logits = metal_values("Muse official-reference logits", &stateless.logits);
    assert!(actual_logits.iter().all(|value| value.is_finite()));
    let prompt_cosine = cosine_similarity(&actual_logits, &reference_logits);
    assert!(
        prompt_cosine >= 0.999,
        "official/Burn Muse prompt-logit cosine {prompt_cosine} is below 0.999"
    );
    let actual_top1 = stateless
        .logits
        .clone()
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("Metal prompt top-1 token must use i32 storage");
    assert_eq!(
        u32::try_from(actual_top1[0]).expect("nonnegative Muse token ID"),
        reference.outputs.prompt_logits.top1_token_id
    );

    let hidden_tensor = Tensor::cat(
        stateless
            .hidden_states
            .iter()
            .map(|hidden| {
                hidden
                    .clone()
                    .slice([0..1, prompt_length - 1..prompt_length, 0..6_656])
            })
            .collect(),
        0,
    );
    let actual_hidden = metal_values("Muse official-reference hidden trace", &hidden_tensor);
    assert_eq!(actual_hidden.len(), reference_hidden.len());
    let hidden_cosines = actual_hidden
        .chunks_exact(6_656)
        .zip(reference_hidden.chunks_exact(6_656))
        .map(|(actual, expected)| cosine_similarity(actual, expected))
        .collect::<Vec<_>>();
    assert_eq!(hidden_cosines.len(), 54);
    for ((label, cosine), row) in reference
        .outputs
        .hidden_last_trace
        .labels
        .iter()
        .zip(hidden_cosines.iter().copied())
        .zip(0..)
    {
        assert!(
            cosine >= 0.999,
            "official/Burn Muse hidden cosine for row {row} ({label}) was {cosine}"
        );
    }

    let mut state = loaded
        .model()
        .new_decode_state(
            CacheConfig::new(1, prompt_length + reference.generation.max_new_tokens)
                .expect("official reference cache geometry must validate"),
            1,
            &device,
        )
        .expect("allocate Muse reference decode state");
    let mut cached = loaded
        .model()
        .prefill(
            CausalLmInput {
                token_ids: prompt_tensor,
                token_mask: Some(prompt_mask),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("prefill the official-reference Muse prompt");
    assert_eq!(state.lengths(), &[prompt_length]);
    let cached_prompt_logits = metal_values("Muse cached prompt logits", &cached.logits);
    let cached_prompt_cosine = cosine_similarity(&cached_prompt_logits, &reference_logits);
    assert!(
        cached_prompt_cosine >= 0.999,
        "official/Burn Muse cached prompt cosine {cached_prompt_cosine} is below 0.999"
    );

    let mut generated = Vec::with_capacity(reference.generation.max_new_tokens);
    for (step, expected_token) in reference
        .generation
        .cached_prefix_top1
        .iter()
        .copied()
        .enumerate()
    {
        let selected = cached
            .logits
            .clone()
            .argmax(2)
            .to_data()
            .to_vec::<i32>()
            .expect("cached Metal top-1 token must use i32 storage");
        let selected = u32::try_from(selected[0]).expect("nonnegative Muse token ID");
        assert_eq!(
            selected, expected_token,
            "Burn Muse cached greedy token changed at step {step}"
        );
        generated.push(selected);
        if step + 1 < reference.generation.max_new_tokens {
            cached = loaded
                .model()
                .decode(
                    DecodeInput {
                        token_ids: Tensor::<Metal, 2, Int>::from_data(
                            TensorData::new(vec![i64::from(selected)], [1, 1]),
                            &device,
                        ),
                        token_mask: Some(all_valid_metal_mask(1, &device)),
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap_or_else(|error| panic!("Muse cached decode step {step}: {error}"));
        }
    }
    assert_eq!(generated, reference.generation.token_ids);
    assert_eq!(
        state.lengths(),
        &[prompt_length + reference.generation.max_new_tokens - 1]
    );

    let mut teacher_ids = prompt_ids;
    teacher_ids.extend(
        generated[..generated.len() - 1]
            .iter()
            .copied()
            .map(i64::from),
    );
    let teacher_length = teacher_ids.len();
    let prediction_positions = (prompt_length - 1
        ..prompt_length - 1 + reference.generation.max_new_tokens)
        .collect::<Vec<_>>();
    let teacher = loaded
        .model()
        .forward(
            CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(teacher_ids, [1, teacher_length]),
                    &device,
                ),
                token_mask: Some(all_valid_metal_mask(teacher_length, &device)),
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::Positions(prediction_positions),
                ..ForwardOptions::default()
            },
        )
        .expect("run Muse stateless teacher-forced prefix inference");
    assert_eq!(
        teacher.logits.dims(),
        [1, reference.generation.max_new_tokens, 202_048]
    );
    let stateless_top1 = teacher
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("teacher-forced Metal top-1 tokens must use i32 storage")
        .into_iter()
        .map(|token| u32::try_from(token).expect("nonnegative Muse token ID"))
        .collect::<Vec<_>>();
    assert_eq!(
        stateless_top1, reference.generation.stateless_prefix_top1,
        "Burn Muse stateless predictions differ from the official reference"
    );
    assert_eq!(
        stateless_top1, generated,
        "Burn Muse cached and stateless predictions differ"
    );

    let minimum_hidden_cosine = hidden_cosines.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!(
        "Muse-Glimmer-30B official F16 parity: prompt_cosine={prompt_cosine:.9}, \
         cached_prompt_cosine={cached_prompt_cosine:.9}, \
         minimum_hidden_cosine={minimum_hidden_cosine:.9}, greedy_tokens={}",
        generated.len()
    );
}
