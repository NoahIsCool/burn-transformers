//! Strict external official-reference bundle reader for Qwen3.5-0.8B.
//!
//! Large numerical payloads stay outside Git. The four independently reviewed
//! SHA-256 anchors below fail closed until a maintainer generates the bundle,
//! reviews its provenance, and pins its exact identity. Opening uses macOS
//! `O_NOFOLLOW`, bounded reads, exact tensor byte counts, and finite F32 checks.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::Read,
    os::unix::fs::OpenOptionsExt,
    path::Path,
};

use serde::Deserialize;
use sha2::{Digest, Sha256};

const BUNDLE_DIRECTORY: &str = "qwen3.5-0.8b-2fc06364-reference.bundle";
const JSON_FILENAME: &str = "qwen3.5-0.8b-2fc06364-reference.json";
const PROMPT_FILENAME: &str = "qwen3.5-0.8b-2fc06364-prompt-logits.f32le";
const CACHED_FILENAME: &str = "qwen3.5-0.8b-2fc06364-cached-logits-32x248320.f32le";
const TEACHER_FILENAME: &str = "qwen3.5-0.8b-2fc06364-teacher-logits-32x248320.f32le";
const REPOSITORY: &str = "Qwen/Qwen3.5-0.8B";
const REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
const VOCABULARY: usize = 248_320;
const ROWS: usize = 32;
const PROMPT_BYTES: u64 = (VOCABULARY * size_of::<f32>()) as u64;
const MATRIX_BYTES: u64 = (ROWS * VOCABULARY * size_of::<f32>()) as u64;
const MAX_JSON_BYTES: u64 = 2 * 1024 * 1024;
const DISTRIBUTION_MANIFEST_SHA256: &str =
    "b814bb3254d51a28c6d74e80c04c3532c9836a0aadc4fab0eed410d6612d59dd";
const EXPECTED_GREEDY_TOKEN_IDS: [i64; ROWS] = [
    11, 271, 40, 1_044, 3_133, 440, 264, 12_654, 5_148, 421, 5_533, 279, 1_510, 2_311, 22_233,
    71_962, 63, 4_536, 310, 11_290, 264, 2_397, 59_229, 22_868, 13, 561, 5_148, 369, 5_995, 310,
    1_423, 279,
];

// These anchors intentionally remain absent in the producer commit. A reviewed
// create-only run must pin all four in one subsequent commit before this ignored
// release gate can execute.
const TRUSTED_JSON_SHA256: Option<&str> = None;
const TRUSTED_PROMPT_SHA256: Option<&str> = None;
const TRUSTED_CACHED_SHA256: Option<&str> = None;
const TRUSTED_TEACHER_SHA256: Option<&str> = None;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceRecord {
    schema_version: u32,
    repository: String,
    revision: String,
    reference_class: String,
    prompt: String,
    input_ids: Vec<i64>,
    execution: Execution,
    environment: Environment,
    provenance: Provenance,
    payloads: Payloads,
    generation: Generation,
    self_parity: SelfParity,
    timings_seconds: Timings,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Execution {
    dtype: String,
    device: String,
    attention_implementation: String,
    deterministic_algorithms: bool,
    seed: u64,
    use_cache: bool,
    float32_matmul_precision: String,
    mps_per_process_memory_fraction: f64,
    explicit_all_one_attention_masks: bool,
    hub_kernels_enabled: bool,
    kernel_status: KernelStatus,
    integrity_threat_model: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct KernelStatus {
    fast_path_available: bool,
    causal_conv1d_kernel_available: bool,
    chunk_gated_delta_kernel_available: bool,
    recurrent_gated_delta_kernel_available: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Environment {
    generated_at_utc: String,
    command: Vec<String>,
    python: Python,
    packages: BTreeMap<String, String>,
    distribution_provenance: BTreeMap<String, Distribution>,
    distribution_manifest_sha256: String,
    device: Device,
    offline_environment: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Distribution {
    version: String,
    metadata: FileIdentity,
    record: FileIdentity,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileIdentity {
    relative_path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Python {
    version: String,
    executable: String,
    implementation: String,
    platform: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Device {
    kind: String,
    built: bool,
    available: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    generator: Generator,
    reference_sources: Vec<ReferenceSource>,
    artifacts: Vec<Artifact>,
    artifact_manifest_sha256: String,
    checkpoint_tensor_bytes: u64,
    loaded_parameter_count: u64,
    parameter_dtype_elements: BTreeMap<String, u64>,
    loading_info: BTreeMap<String, LoadingClass>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceSource {
    module: String,
    path: String,
    relative_path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Generator {
    git_head: String,
    git_tree: String,
    files: Vec<GeneratorFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratorFile {
    relative_path: String,
    bytes: usize,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    relative_path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LoadingClass {
    count: usize,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payloads {
    prompt: Payload,
    cached: Payload,
    teacher: Payload,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    relative_path: String,
    encoding: String,
    shape: Vec<usize>,
    elements: usize,
    bytes: usize,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Generation {
    max_new_tokens: usize,
    token_ids: Vec<i64>,
    eos_token_ids: Vec<i64>,
    completed_without_eos: bool,
    decision_stability: Vec<DecisionStability>,
    minimum_observed_margin: f64,
    minimum_margin_to_epsilon_ratio: f64,
    decoded_text: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionStability {
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SelfParity {
    gate: f64,
    prompt_cached_cosine: f64,
    cached_teacher_cosines: Vec<f64>,
    minimum_cached_teacher_cosine: f64,
    cached_top1_token_ids: Vec<i64>,
    teacher_top1_token_ids: Vec<i64>,
    top1_equal: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Timings {
    load: f64,
    stateless_prompt: f64,
    cached_generation_32: f64,
    teacher_forced_all_logits: f64,
}

/// Fully authenticated numerical evidence retained in host memory for one gate.
pub(super) struct ReferenceBundle {
    record: ReferenceRecord,
    prompt: Vec<f32>,
    cached: Vec<f32>,
    teacher: Vec<f32>,
}

impl ReferenceBundle {
    /// Returns the exact tokenized prompt consumed by the official graph.
    pub(super) fn input_ids(&self) -> &[i64] {
        &self.record.input_ids
    }

    /// Returns the exact official 32-token greedy continuation.
    pub(super) fn greedy_token_ids(&self) -> &[i64] {
        &self.record.generation.token_ids
    }

    /// Returns the official full-vocabulary prompt row.
    pub(super) fn prompt_logits(&self) -> &[f32] {
        &self.prompt
    }

    /// Returns one official cached decision row.
    pub(super) fn cached_row(&self, row: usize) -> &[f32] {
        matrix_row(&self.cached, row)
    }

    /// Returns one corresponding official stateless teacher-forced row.
    pub(super) fn teacher_row(&self, row: usize) -> &[f32] {
        matrix_row(&self.teacher, row)
    }
}

/// Loads the fixed bundle below an explicitly selected external parent.
pub(super) fn load(parent: &Path) -> ReferenceBundle {
    for (label, anchor) in [
        ("JSON", TRUSTED_JSON_SHA256),
        ("prompt logits", TRUSTED_PROMPT_SHA256),
        ("cached logits", TRUSTED_CACHED_SHA256),
        ("teacher logits", TRUSTED_TEACHER_SHA256),
    ] {
        require_trusted_anchor(label, anchor);
    }
    let root = parent.join(BUNDLE_DIRECTORY);
    let metadata = fs::symlink_metadata(&root)
        .unwrap_or_else(|error| panic!("inspect reference bundle {}: {error}", root.display()));
    assert!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "reference bundle must be a non-symlink directory"
    );
    let observed_names = fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("enumerate reference bundle {}: {error}", root.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|error| panic!("read reference bundle entry: {error}"))
                .file_name()
                .into_string()
                .expect("reference filenames must be UTF-8")
        })
        .collect::<BTreeSet<_>>();
    let expected_names = [
        JSON_FILENAME,
        PROMPT_FILENAME,
        CACHED_FILENAME,
        TEACHER_FILENAME,
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(
        observed_names, expected_names,
        "reference bundle file set changed"
    );

    let json_path = root.join(JSON_FILENAME);
    let json_bytes = read_bounded(&json_path, None, MAX_JSON_BYTES);
    assert_digest("JSON", &json_bytes, TRUSTED_JSON_SHA256);
    let record: ReferenceRecord = serde_json::from_slice(&json_bytes)
        .unwrap_or_else(|error| panic!("parse strict Qwen reference JSON: {error}"));
    validate_record(&record);

    let prompt_bytes = read_bounded(
        &root.join(PROMPT_FILENAME),
        Some(PROMPT_BYTES),
        PROMPT_BYTES,
    );
    assert_digest("prompt logits", &prompt_bytes, TRUSTED_PROMPT_SHA256);
    validate_payload(
        &record.payloads.prompt,
        PROMPT_FILENAME,
        &[VOCABULARY],
        &prompt_bytes,
    );
    let cached_bytes = read_bounded(
        &root.join(CACHED_FILENAME),
        Some(MATRIX_BYTES),
        MATRIX_BYTES,
    );
    assert_digest("cached logits", &cached_bytes, TRUSTED_CACHED_SHA256);
    validate_payload(
        &record.payloads.cached,
        CACHED_FILENAME,
        &[ROWS, VOCABULARY],
        &cached_bytes,
    );
    let teacher_bytes = read_bounded(
        &root.join(TEACHER_FILENAME),
        Some(MATRIX_BYTES),
        MATRIX_BYTES,
    );
    assert_digest("teacher logits", &teacher_bytes, TRUSTED_TEACHER_SHA256);
    validate_payload(
        &record.payloads.teacher,
        TEACHER_FILENAME,
        &[ROWS, VOCABULARY],
        &teacher_bytes,
    );

    ReferenceBundle {
        record,
        prompt: decode_f32(&prompt_bytes, "prompt logits"),
        cached: decode_f32(&cached_bytes, "cached logits"),
        teacher: decode_f32(&teacher_bytes, "teacher logits"),
    }
}

/// Enforces every invariant independent of the large raw payload bytes.
fn validate_record(record: &ReferenceRecord) {
    assert_eq!(record.schema_version, 1);
    assert_eq!(record.repository, REPOSITORY);
    assert_eq!(record.revision, REVISION);
    assert_eq!(record.reference_class, "transformers.Qwen3_5ForCausalLM");
    assert_eq!(record.prompt, "Hello");
    assert_eq!(record.input_ids, [9_419]);
    let execution = &record.execution;
    assert_eq!(execution.dtype, "float16");
    assert_eq!(execution.device, "mps");
    assert_eq!(execution.attention_implementation, "eager");
    assert!(execution.deterministic_algorithms && execution.use_cache);
    assert_eq!(execution.seed, 0);
    assert_eq!(execution.float32_matmul_precision, "highest");
    assert_eq!(execution.mps_per_process_memory_fraction, 1.0);
    assert!(execution.explicit_all_one_attention_masks);
    assert!(!execution.hub_kernels_enabled);
    assert!(!execution.integrity_threat_model.is_empty());
    assert!(!execution.kernel_status.fast_path_available);
    assert!(!execution.kernel_status.causal_conv1d_kernel_available);
    assert!(!execution.kernel_status.chunk_gated_delta_kernel_available);
    assert!(
        !execution
            .kernel_status
            .recurrent_gated_delta_kernel_available
    );

    let environment = &record.environment;
    assert!(!environment.generated_at_utc.is_empty());
    assert!(
        environment
            .command
            .iter()
            .any(|part| part.ends_with("generate_qwen35_08b_reference.py"))
    );
    assert!(!environment.python.version.is_empty());
    assert!(!environment.python.executable.is_empty());
    assert!(!environment.python.implementation.is_empty());
    assert!(!environment.python.platform.is_empty());
    let expected_packages = BTreeMap::from([
        ("accelerate", "1.12.0"),
        ("numpy", "2.4.3"),
        ("safetensors", "0.7.0"),
        ("tokenizers", "0.22.2"),
        ("torch", "2.10.0"),
        ("transformers", "5.2.0"),
    ]);
    assert_eq!(environment.packages.len(), expected_packages.len());
    for (name, version) in &expected_packages {
        assert_eq!(
            environment.packages.get(*name).map(String::as_str),
            Some(*version)
        );
    }
    assert_eq!(
        environment
            .distribution_provenance
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        expected_packages.keys().copied().collect()
    );
    for (name, distribution) in &environment.distribution_provenance {
        assert_eq!(
            distribution.version,
            *expected_packages
                .get(name.as_str())
                .expect("distribution package was checked above")
        );
        validate_file_identity(&distribution.metadata);
        validate_file_identity(&distribution.record);
        assert!(
            distribution
                .metadata
                .relative_path
                .ends_with(".dist-info/METADATA")
        );
        assert!(
            distribution
                .record
                .relative_path
                .ends_with(".dist-info/RECORD")
        );
    }
    assert_eq!(
        environment.distribution_manifest_sha256,
        DISTRIBUTION_MANIFEST_SHA256
    );
    assert_eq!(environment.device.kind, "mps");
    assert!(environment.device.built && environment.device.available);
    let expected_offline_environment = BTreeMap::from([
        ("HF_HUB_OFFLINE", "1"),
        ("TRANSFORMERS_OFFLINE", "1"),
        ("HF_DATASETS_OFFLINE", "1"),
        ("TOKENIZERS_PARALLELISM", "false"),
        ("HF_DEACTIVATE_ASYNC_LOAD", "1"),
        ("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        ("USE_HUB_KERNELS", "0"),
    ]);
    assert_eq!(
        environment.offline_environment.len(),
        expected_offline_environment.len()
    );
    for (name, expected) in expected_offline_environment {
        assert_eq!(
            environment
                .offline_environment
                .get(name)
                .map(String::as_str),
            Some(expected)
        );
    }

    validate_provenance(&record.provenance);
    let generation = &record.generation;
    assert_eq!(generation.max_new_tokens, ROWS);
    assert_eq!(generation.token_ids, EXPECTED_GREEDY_TOKEN_IDS);
    assert_eq!(generation.eos_token_ids, [248_044]);
    assert!(generation.completed_without_eos);
    assert_eq!(generation.decision_stability.len(), ROWS);
    for (step, decision) in generation.decision_stability.iter().enumerate() {
        assert_eq!(decision.step, step);
        assert_eq!(decision.top1_token_id, generation.token_ids[step]);
        assert_ne!(decision.top1_token_id, decision.top2_token_id);
        assert!((0..VOCABULARY as i64).contains(&decision.top1_token_id));
        assert!((0..VOCABULARY as i64).contains(&decision.top2_token_id));
        assert!(decision.top1_logit.is_finite() && decision.top2_logit.is_finite());
        assert!(decision.top1_logit > decision.top2_logit);
        assert_eq!(decision.margin, decision.top1_logit - decision.top2_logit);
        assert!(decision.top1_f16_ulp.is_finite() && decision.top1_f16_ulp > 0.0);
        assert!(decision.top2_f16_ulp.is_finite() && decision.top2_f16_ulp > 0.0);
        assert!(decision.elementwise_epsilon.is_finite() && decision.elementwise_epsilon > 0.0);
        assert_eq!(
            decision.elementwise_epsilon,
            decision.top1_f16_ulp.max(decision.top2_f16_ulp)
        );
        assert!(decision.margin_to_epsilon_ratio.is_finite());
        assert_eq!(
            decision.margin_to_epsilon_ratio,
            decision.margin / decision.elementwise_epsilon
        );
        assert!(decision.stable_under_budget && decision.margin_to_epsilon_ratio > 2.0);
    }
    assert!(
        generation.minimum_observed_margin.is_finite() && generation.minimum_observed_margin > 0.0
    );
    assert!(
        generation.minimum_margin_to_epsilon_ratio.is_finite()
            && generation.minimum_margin_to_epsilon_ratio > 2.0
    );
    assert!(!generation.decoded_text.is_empty());

    let parity = &record.self_parity;
    assert_eq!(parity.gate, 0.999);
    assert!(parity.prompt_cached_cosine >= parity.gate);
    assert_eq!(parity.cached_teacher_cosines.len(), ROWS);
    assert!(
        parity
            .cached_teacher_cosines
            .iter()
            .all(|value| value.is_finite() && *value >= parity.gate)
    );
    assert!(parity.minimum_cached_teacher_cosine >= parity.gate);
    assert_eq!(parity.cached_top1_token_ids, generation.token_ids);
    assert_eq!(parity.teacher_top1_token_ids, generation.token_ids);
    assert!(parity.top1_equal);
    for value in [
        record.timings_seconds.load,
        record.timings_seconds.stateless_prompt,
        record.timings_seconds.cached_generation_32,
        record.timings_seconds.teacher_forced_all_logits,
    ] {
        assert!(value.is_finite() && value >= 0.0);
    }
}

/// Validates immutable checkpoint and generator identities embedded in JSON.
fn validate_provenance(provenance: &Provenance) {
    assert_eq!(provenance.checkpoint_tensor_bytes, 1_746_882_752);
    assert_eq!(provenance.loaded_parameter_count, 752_393_024);
    assert_eq!(
        provenance.parameter_dtype_elements.get("float16"),
        Some(&752_393_024)
    );
    assert_eq!(provenance.parameter_dtype_elements.len(), 1);
    assert_sha256(&provenance.artifact_manifest_sha256);
    assert_git_oid(&provenance.generator.git_head);
    assert_git_oid(&provenance.generator.git_tree);
    assert_eq!(provenance.generator.files.len(), 2);
    for (relative_path, bytes) in [
        (
            "scripts/generate_qwen35_08b_reference.py",
            include_bytes!("../../scripts/generate_qwen35_08b_reference.py").as_slice(),
        ),
        (
            "scripts/generate_qwen38_reference.py",
            include_bytes!("../../scripts/generate_qwen38_reference.py").as_slice(),
        ),
    ] {
        let file = provenance
            .generator
            .files
            .iter()
            .find(|file| file.relative_path == relative_path)
            .unwrap_or_else(|| panic!("generator provenance omitted {relative_path}"));
        assert_eq!(file.bytes, bytes.len());
        assert_eq!(file.sha256, sha256(bytes));
    }
    assert_eq!(
        provenance
            .generator
            .files
            .iter()
            .map(|file| file.relative_path.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "scripts/generate_qwen35_08b_reference.py",
            "scripts/generate_qwen38_reference.py",
        ])
    );
    validate_reference_sources(&provenance.reference_sources);

    let expected_artifacts = BTreeMap::from([
        (
            "chat_template.jinja",
            (
                7_755,
                "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
            ),
        ),
        (
            "config.json",
            (
                2_907,
                "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
            ),
        ),
        (
            "model.safetensors-00001-of-00001.safetensors",
            (
                1_746_942_600,
                "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
            ),
        ),
        (
            "model.safetensors.index.json",
            (
                50_900,
                "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
            ),
        ),
        (
            "tokenizer.json",
            (
                12_807_982,
                "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            ),
        ),
        (
            "tokenizer_config.json",
            (
                16_709,
                "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
            ),
        ),
    ]);
    assert_eq!(provenance.artifacts.len(), expected_artifacts.len());
    assert_eq!(
        provenance
            .artifacts
            .iter()
            .map(|artifact| artifact.relative_path.as_str())
            .collect::<BTreeSet<_>>(),
        expected_artifacts.keys().copied().collect()
    );
    for artifact in &provenance.artifacts {
        let &(bytes, digest) = expected_artifacts
            .get(artifact.relative_path.as_str())
            .unwrap_or_else(|| panic!("unexpected checkpoint artifact {}", artifact.relative_path));
        assert_eq!(artifact.bytes, bytes);
        assert_eq!(artifact.sha256, digest);
    }
    assert_eq!(
        provenance
            .loading_info
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "error_msgs",
            "mismatched_keys",
            "missing_keys",
            "unexpected_keys",
        ])
    );
    for class in provenance.loading_info.values() {
        assert_eq!(class.count, 0);
        assert_sha256(&class.sha256);
    }
}

/// Validates one installed-package metadata or record identity.
fn validate_file_identity(identity: &FileIdentity) {
    assert!(!identity.relative_path.is_empty());
    assert!(identity.bytes > 0);
    assert_sha256(&identity.sha256);
}

/// Validates the complete module set used to execute the official graph.
fn validate_reference_sources(sources: &[ReferenceSource]) {
    let expected_modules = BTreeSet::from([
        "transformers.cache_utils",
        "transformers.configuration_utils",
        "transformers.conversion_mapping",
        "transformers.masking_utils",
        "transformers.modeling_attn_mask_utils",
        "transformers.modeling_utils",
        "transformers.models.qwen3_5.configuration_qwen3_5",
        "transformers.models.qwen3_5.modeling_qwen3_5",
    ]);
    assert_eq!(sources.len(), expected_modules.len());
    assert_eq!(
        sources
            .iter()
            .map(|source| source.module.as_str())
            .collect::<BTreeSet<_>>(),
        expected_modules
    );
    assert_eq!(
        sources
            .iter()
            .map(|source| source.relative_path.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        sources.len()
    );
    for source in sources {
        let relative = Path::new(&source.relative_path);
        assert!(!relative.is_absolute());
        assert!(!source.relative_path.is_empty());
        assert!(
            !relative
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        );
        assert!(Path::new(&source.path).is_absolute());
        assert!(Path::new(&source.path).ends_with(relative));
        assert!(source.bytes > 0 && source.bytes <= MAX_JSON_BYTES);
        assert_sha256(&source.sha256);
    }
}

/// Cross-checks JSON metadata against one exact raw tensor file.
fn validate_payload(payload: &Payload, filename: &str, shape: &[usize], bytes: &[u8]) {
    assert_eq!(payload.relative_path, filename);
    assert_eq!(payload.encoding, "ieee754-f32-little-endian");
    assert_eq!(payload.shape, shape);
    assert_eq!(payload.elements, shape.iter().product::<usize>());
    assert_eq!(payload.bytes, bytes.len());
    assert_eq!(payload.sha256, sha256(bytes));
}

/// Opens a bounded regular file without following its final path component.
fn read_bounded(path: &Path, expected: Option<u64>, maximum: u64) -> Vec<u8> {
    const DARWIN_O_NOFOLLOW: i32 = 0x0000_0100;
    const DARWIN_O_NONBLOCK: i32 = 0x0000_0004;

    let file = OpenOptions::new()
        .read(true)
        .custom_flags(DARWIN_O_NOFOLLOW | DARWIN_O_NONBLOCK)
        .open(path)
        .unwrap_or_else(|error| panic!("open reference file {}: {error}", path.display()));
    let metadata = file
        .metadata()
        .unwrap_or_else(|error| panic!("inspect reference file {}: {error}", path.display()));
    assert!(
        metadata.file_type().is_file(),
        "reference payload must be a regular file"
    );
    if let Some(expected) = expected {
        assert_eq!(metadata.len(), expected, "reference payload length changed");
    }
    assert!(
        metadata.len() <= maximum,
        "reference payload exceeds its safety bound"
    );
    let capacity = usize::try_from(metadata.len()).expect("bounded payload length fits usize");
    let mut bytes = Vec::with_capacity(capacity);
    file.take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)
        .unwrap_or_else(|error| panic!("read reference file {}: {error}", path.display()));
    assert_eq!(
        bytes.len(),
        capacity,
        "reference payload changed while reading"
    );
    bytes
}

/// Decodes exact little-endian IEEE F32 storage and rejects non-finite values.
fn decode_f32(bytes: &[u8], label: &str) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(size_of::<f32>()));
    bytes
        .chunks_exact(size_of::<f32>())
        .enumerate()
        .map(|(index, bytes)| {
            let value = f32::from_le_bytes(bytes.try_into().expect("four-byte F32 chunk"));
            assert!(
                value.is_finite(),
                "{label} contains non-finite value at {index}"
            );
            value
        })
        .collect()
}

/// Returns one checked row from flattened `[32, vocabulary]` storage.
fn matrix_row(matrix: &[f32], row: usize) -> &[f32] {
    assert!(row < ROWS, "reference row {row} is out of bounds");
    let start = row * VOCABULARY;
    &matrix[start..start + VOCABULARY]
}

/// Fails closed until a generated payload has an independently reviewed anchor.
fn require_trusted_anchor<'a>(label: &str, anchor: Option<&'a str>) -> &'a str {
    let anchor = anchor.unwrap_or_else(|| {
        panic!(
            "trusted Qwen3.5-0.8B {label} SHA-256 is not pinned; review the create-only bundle and update tests/support/qwen35_release_reference.rs"
        )
    });
    assert_sha256(anchor);
    anchor
}

/// Compares one payload against its independently reviewed identity.
fn assert_digest(label: &str, bytes: &[u8], anchor: Option<&str>) {
    let expected = require_trusted_anchor(label, anchor);
    assert_eq!(sha256(bytes), expected, "reviewed {label} identity changed");
}

/// Computes canonical lowercase SHA-256.
fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Requires one canonical lowercase SHA-256 string.
fn assert_sha256(value: &str) {
    assert!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "malformed SHA-256 {value:?}"
    );
}

/// Accepts the SHA-1 or SHA-256 object spelling used by the checked-out Git repository.
fn assert_git_oid(value: &str) {
    assert!(
        matches!(value.len(), 40 | 64)
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "malformed Git object ID {value:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::{
        MATRIX_BYTES, PROMPT_BYTES, assert_sha256, decode_f32, matrix_row, require_trusted_anchor,
    };

    #[test]
    fn fixed_payload_geometry_is_exact() {
        assert_eq!(PROMPT_BYTES, 993_280);
        assert_eq!(MATRIX_BYTES, 31_784_960);
    }

    #[test]
    fn f32_decoder_and_matrix_rows_are_little_endian() {
        let bytes = [1.0_f32, -2.0, 3.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(decode_f32(&bytes, "fixture"), [1.0, -2.0, 3.5]);

        let matrix = (0..2 * 248_320)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert_eq!(matrix_row(&matrix, 1)[0], 248_320.0);
    }

    #[test]
    #[should_panic(expected = "trusted Qwen3.5-0.8B fixture SHA-256 is not pinned")]
    fn missing_trust_anchor_fails_closed() {
        require_trusted_anchor("fixture", None);
    }

    #[test]
    #[should_panic(expected = "malformed SHA-256")]
    fn malformed_trust_anchor_is_rejected() {
        assert_sha256("ABC");
    }

    #[test]
    #[should_panic(expected = "contains non-finite value")]
    fn non_finite_reference_value_is_rejected() {
        decode_f32(&f32::NAN.to_le_bytes(), "fixture");
    }
}
