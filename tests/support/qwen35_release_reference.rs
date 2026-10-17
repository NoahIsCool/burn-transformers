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

/// Directory containing the authenticated metadata and three logit payloads.
const BUNDLE_DIRECTORY: &str = "qwen3.5-0.8b-2fc06364-reference.bundle";
/// Strict JSON record describing the pinned reference run and its payloads.
const JSON_FILENAME: &str = "qwen3.5-0.8b-2fc06364-reference.json";
/// Little-endian F32 vocabulary logits for the final prompt position.
const PROMPT_FILENAME: &str = "qwen3.5-0.8b-2fc06364-prompt-logits.f32le";
/// Little-endian F32 logits for all 32 cached generation decisions.
const CACHED_FILENAME: &str = "qwen3.5-0.8b-2fc06364-cached-logits-32x248320.f32le";
/// Little-endian F32 logits for the same decisions under stateless teacher forcing.
const TEACHER_FILENAME: &str = "qwen3.5-0.8b-2fc06364-teacher-logits-32x248320.f32le";
/// Official checkpoint repository whose local artifact bytes are authenticated.
const REPOSITORY: &str = "Qwen/Qwen3.5-0.8B";
/// Immutable official checkpoint revision required by the reference record.
const REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
/// Number of vocabulary logits stored for each prediction position.
const VOCABULARY: usize = 248_320;
/// Number of consecutive generation decisions retained in each matrix.
const ROWS: usize = 32;
/// Exact byte count of one full-vocabulary F32 prompt row.
const PROMPT_BYTES: u64 = (VOCABULARY * size_of::<f32>()) as u64;
/// Exact byte count of a row-major `[ROWS, VOCABULARY]` F32 matrix.
const MATRIX_BYTES: u64 = (ROWS * VOCABULARY * size_of::<f32>()) as u64;
/// Maximum accepted metadata size, also bounding each recorded reference source.
const MAX_JSON_BYTES: u64 = 2 * 1024 * 1024;
/// Independently pinned digest of the reference environment's package inventory.
const DISTRIBUTION_MANIFEST_SHA256: &str =
    "b814bb3254d51a28c6d74e80c04c3532c9836a0aadc4fab0eed410d6612d59dd";
/// Exact rendered non-thinking chat prompt consumed by the official graph.
const EXPECTED_PROMPT: &str = "<|im_start|>user\nExplain why the seasons change in four complete sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
/// Official tokenizer output for the fixed rendered prompt.
const EXPECTED_INPUT_IDS: [i64; 23] = [
    248045, 846, 198, 814, 20139, 3069, 279, 15127, 2222, 303, 2943, 4434, 22157, 13, 248046, 198,
    248045, 74455, 198, 248068, 271, 248069, 271,
];
/// Official greedy continuation in prediction order, with no EOS among its 32 tokens.
const EXPECTED_GREEDY_TOKEN_IDS: [i64; ROWS] = [
    760, 8964, 579, 7772, 369, 82951, 506, 449, 8937, 314, 12805, 220, 17, 18, 13, 20, 11995, 8410,
    310, 1141, 62409, 10709, 11, 13969, 279, 11247, 310, 381, 11784, 310, 279, 7806,
];

// Independently rehashed after create-only generation at commit 06f39e7, with
// exact package/source inventories, checkpoint bytes, and all 32 margins reviewed.
/// Reviewed JSON digest; an absent anchor disables bundle loading.
const TRUSTED_JSON_SHA256: Option<&str> =
    Some("1a1fb75e193f82c48ef0494f3d64066d32133f7654e5c485cb193adc54acd20a");
/// Reviewed prompt-logit payload digest; an absent anchor disables bundle loading.
const TRUSTED_PROMPT_SHA256: Option<&str> =
    Some("96c1291aebb1c36c15b42ad684a0a5bed5917c40808ad8cbbe82a03deb646df1");
/// Reviewed cached-logit payload digest; an absent anchor disables bundle loading.
const TRUSTED_CACHED_SHA256: Option<&str> =
    Some("a100338b3b1c6f755b3ef3c1925c090fe9cf5b2a38b8ca2df42d57bc8b9ae4c9");
/// Reviewed teacher-logit payload digest; an absent anchor disables bundle loading.
const TRUSTED_TEACHER_SHA256: Option<&str> =
    Some("ec7786b3f837c97b7ee13c16a9fc742333075a84bf3da126b81191ec4a190f03");

/// Strict metadata schema linking numerical evidence to its source and execution.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceRecord {
    /// Format version, currently required to be one.
    schema_version: u32,
    /// Official checkpoint repository name.
    repository: String,
    /// Immutable checkpoint revision used for generation.
    revision: String,
    /// Fully qualified official causal language model class.
    reference_class: String,
    /// Rendered chat prompt supplied to the official tokenizer.
    prompt: String,
    /// Prompt token IDs consumed by the official model.
    input_ids: Vec<i64>,
    /// Numerical, device, determinism, and kernel settings for inference.
    execution: Execution,
    /// Interpreter, installed packages, and offline environment identity.
    environment: Environment,
    /// Generator, reference code, checkpoint, and strict-loading identities.
    provenance: Provenance,
    /// Storage metadata for the three full-vocabulary logit payloads.
    payloads: Payloads,
    /// Greedy continuation and its numerical decision margins.
    generation: Generation,
    /// Agreement between cached and stateless execution of the official model.
    self_parity: SelfParity,
    /// Nonnegative wall-clock durations in seconds for the reference stages.
    timings_seconds: Timings,
}

/// Recorded inference settings required for the pinned official F16 MPS run.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Execution {
    /// Model parameter and execution dtype, required to be `float16`.
    dtype: String,
    /// Execution device identifier, required to be `mps`.
    device: String,
    /// Official attention implementation, required to be `eager`.
    attention_implementation: String,
    /// Whether PyTorch deterministic algorithms were enabled.
    deterministic_algorithms: bool,
    /// Random seed fixed for the official reference run.
    seed: u64,
    /// Whether the official model enabled its generation cache.
    use_cache: bool,
    /// PyTorch F32 matrix multiplication precision setting.
    float32_matmul_precision: String,
    /// Allowed fraction of recommended MPS process memory.
    mps_per_process_memory_fraction: f64,
    /// Whether every reference call explicitly supplied unpadded attention masks.
    explicit_all_one_attention_masks: bool,
    /// Whether optional Hub kernels were enabled; this reference requires false.
    hub_kernels_enabled: bool,
    /// Availability flags for optional accelerated reference kernels.
    kernel_status: KernelStatus,
    /// Generator's description of the provenance integrity assumptions.
    integrity_threat_model: String,
}

/// Optional official acceleration paths, all required to be unavailable here.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct KernelStatus {
    /// Whether the reference exposes its combined accelerated execution path.
    fast_path_available: bool,
    /// Whether an accelerated causal one-dimensional convolution kernel is present.
    causal_conv1d_kernel_available: bool,
    /// Whether an accelerated chunked gated-delta kernel is present.
    chunk_gated_delta_kernel_available: bool,
    /// Whether an accelerated recurrent gated-delta kernel is present.
    recurrent_gated_delta_kernel_available: bool,
}

/// Host and package provenance captured when the external bundle was generated.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Environment {
    /// Recorded generation timestamp in UTC.
    generated_at_utc: String,
    /// Generator command and arguments as separate strings.
    command: Vec<String>,
    /// Python interpreter and host-platform details.
    python: Python,
    /// Exact installed versions keyed by the six required package names.
    packages: BTreeMap<String, String>,
    /// Installed distribution metadata identities keyed by package name.
    distribution_provenance: BTreeMap<String, Distribution>,
    /// Digest of the complete installed-distribution manifest.
    distribution_manifest_sha256: String,
    /// MPS build and runtime availability recorded by the generator.
    device: Device,
    /// Exact environment settings enforcing offline inference and kernel policy.
    offline_environment: BTreeMap<String, String>,
}

/// Installed package version and identities of its distribution metadata files.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Distribution {
    /// Package version matching the pinned environment inventory.
    version: String,
    /// Identity of the installed `.dist-info/METADATA` file.
    metadata: FileIdentity,
    /// Identity of the installed `.dist-info/RECORD` file.
    record: FileIdentity,
}

/// Recorded relative path, length, and digest of an installed metadata file.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileIdentity {
    /// Recorded relative location of the file within the installed environment.
    relative_path: String,
    /// Nonzero file length in bytes.
    bytes: u64,
    /// Canonical lowercase SHA-256 digest of the file bytes.
    sha256: String,
}

/// Python interpreter identity and platform recorded for the reference run.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Python {
    /// Interpreter version string.
    version: String,
    /// Executable path reported by the generator.
    executable: String,
    /// Python implementation name.
    implementation: String,
    /// Host-platform description reported by Python.
    platform: String,
}

/// Availability evidence for the required official MPS execution device.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Device {
    /// Backend identifier, required to be `mps`.
    kind: String,
    /// Whether PyTorch was built with MPS support.
    built: bool,
    /// Whether MPS was available to the running generator.
    available: bool,
}

/// Source and checkpoint identities supporting the official numerical evidence.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    /// Git and source-file identities for the bundle generation tooling.
    generator: Generator,
    /// Complete inventory of official Python modules checked for the graph.
    reference_sources: Vec<ReferenceSource>,
    /// Exact checkpoint and tokenizer artifacts authenticated for loading.
    artifacts: Vec<Artifact>,
    /// Canonical digest of the checkpoint artifact manifest.
    artifact_manifest_sha256: String,
    /// Total bytes occupied by checkpoint tensor storage.
    checkpoint_tensor_bytes: u64,
    /// Number of parameter elements loaded into the official model.
    loaded_parameter_count: u64,
    /// Loaded parameter element counts grouped by dtype.
    parameter_dtype_elements: BTreeMap<String, u64>,
    /// Strict-loading diagnostic counts and digests grouped by diagnostic class.
    loading_info: BTreeMap<String, LoadingClass>,
}

/// Source-file identity for one official Python module used by the reference.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceSource {
    /// Fully qualified Python import name.
    module: String,
    /// Absolute path to the installed source file during generation.
    path: String,
    /// Relative source path that must suffix the recorded absolute path.
    relative_path: String,
    /// Source-file length, bounded by the metadata safety limit.
    bytes: u64,
    /// Canonical lowercase SHA-256 digest of the source bytes.
    sha256: String,
}

/// Git snapshot and individual script identities for the reference generator.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Generator {
    /// Recorded Git commit object ID in SHA-1 or SHA-256 spelling.
    git_head: String,
    /// Recorded Git tree object ID in SHA-1 or SHA-256 spelling.
    git_tree: String,
    /// Two generator scripts whose bytes must match the current checkout.
    files: Vec<GeneratorFile>,
}

/// Generator script identity checked against bytes embedded from this checkout.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratorFile {
    /// Repository-relative generator script path.
    relative_path: String,
    /// Exact script length in bytes.
    bytes: usize,
    /// Canonical lowercase SHA-256 digest of the script bytes.
    sha256: String,
}

/// Immutable official checkpoint or tokenizer artifact authenticated before use.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    /// Artifact filename relative to the selected checkpoint root.
    relative_path: String,
    /// Exact artifact length in bytes.
    bytes: u64,
    /// Canonical lowercase SHA-256 digest of the complete artifact.
    sha256: String,
}

/// Identity and size of one strict-loading diagnostic collection.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LoadingClass {
    /// Number of diagnostics, required to be zero for each class.
    count: usize,
    /// Canonical SHA-256 digest recorded for the diagnostic collection.
    sha256: String,
}

/// Storage descriptions for prompt, cached, and teacher-forced logit evidence.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payloads {
    /// Full-vocabulary logits for the final prompt position.
    prompt: Payload,
    /// Logit matrix for the cached greedy generation decisions.
    cached: Payload,
    /// Logit matrix for the same decisions under stateless teacher forcing.
    teacher: Payload,
}

/// Raw tensor metadata cross-checked against authenticated payload bytes.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    /// Exact payload filename relative to the bundle directory.
    relative_path: String,
    /// Encoding identifier, required to specify little-endian IEEE-754 F32.
    encoding: String,
    /// Tensor dimensions, either `[vocabulary]` or `[rows, vocabulary]`.
    shape: Vec<usize>,
    /// Number of scalar elements, equal to the product of the dimensions.
    elements: usize,
    /// Exact encoded payload length in bytes.
    bytes: usize,
    /// Canonical lowercase SHA-256 digest of the encoded payload.
    sha256: String,
}

/// Fixed greedy continuation with per-decision numerical stability evidence.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Generation {
    /// Requested continuation length, fixed to the number of stored matrix rows.
    max_new_tokens: usize,
    /// Selected token IDs in generation order.
    token_ids: Vec<i64>,
    /// EOS token IDs used by the official generation policy.
    eos_token_ids: Vec<i64>,
    /// Whether all requested decisions completed without producing EOS.
    completed_without_eos: bool,
    /// Winner and runner-up evidence for each generated token.
    decision_stability: Vec<DecisionStability>,
    /// Recorded minimum winner-to-runner-up logit margin across the continuation.
    minimum_observed_margin: f64,
    /// Recorded minimum margin divided by its per-decision perturbation budget.
    minimum_margin_to_epsilon_ratio: f64,
    /// Nonempty official decoding of the generated continuation.
    decoded_text: String,
}

/// Greedy winner margin compared with an elementwise F16 rounding budget.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionStability {
    /// Zero-based generation decision index.
    step: usize,
    /// Token ID with the greatest reference logit.
    top1_token_id: i64,
    /// Distinct token ID with the second-greatest reference logit.
    top2_token_id: i64,
    /// Finite logit of the greedy winner.
    top1_logit: f64,
    /// Finite logit of the runner-up.
    top2_logit: f64,
    /// Positive difference between the winning and runner-up logits.
    margin: f64,
    /// Positive F16 unit in the last place recorded at the winning logit.
    top1_f16_ulp: f64,
    /// Positive F16 unit in the last place recorded at the runner-up logit.
    top2_f16_ulp: f64,
    /// Elementwise perturbation budget, the larger of the two recorded F16 ULPs.
    elementwise_epsilon: f64,
    /// Winning margin divided by the elementwise perturbation budget.
    margin_to_epsilon_ratio: f64,
    /// Whether the margin exceeds twice the elementwise perturbation budget.
    stable_under_budget: bool,
}

/// Official-model agreement between cached generation and stateless evaluation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SelfParity {
    /// Required cosine similarity threshold, fixed to `0.999`.
    gate: f64,
    /// Cosine similarity between stateless prompt logits and the first cached row.
    prompt_cached_cosine: f64,
    /// Cosine similarity for each corresponding cached and teacher-forced row.
    cached_teacher_cosines: Vec<f64>,
    /// Recorded minimum cosine similarity across the cached and teacher rows.
    minimum_cached_teacher_cosine: f64,
    /// Greedy token IDs independently read from the cached logit rows.
    cached_top1_token_ids: Vec<i64>,
    /// Greedy token IDs independently read from the teacher-forced logit rows.
    teacher_top1_token_ids: Vec<i64>,
    /// Whether the cached and teacher-forced greedy token sequences agree.
    top1_equal: bool,
}

/// Finite, nonnegative durations in seconds recorded for the reference stages.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Timings {
    /// Duration of official checkpoint loading.
    load: f64,
    /// Duration of stateless prompt inference.
    stateless_prompt: f64,
    /// Duration of the complete 32-decision cached continuation.
    cached_generation_32: f64,
    /// Duration of stateless teacher forcing with all-position logits.
    teacher_forced_all_logits: f64,
}

/// Fully authenticated numerical evidence retained in host memory for one gate.
pub(super) struct ReferenceBundle {
    /// Authenticated metadata after strict schema and invariant validation.
    record: ReferenceRecord,
    /// Decoded full-vocabulary logits for the final prompt position.
    prompt: Vec<f32>,
    /// Decoded cached logits flattened in `[ROWS, VOCABULARY]` row-major order.
    cached: Vec<f32>,
    /// Corresponding stateless teacher logits in the same flattened row order.
    teacher: Vec<f32>,
}

impl ReferenceBundle {
    /// Authenticates the target files Burn will consume against the official source.
    ///
    /// Shape-compatible files are not necessarily the pinned checkpoint. Hash
    /// each complete artifact through a bounded streaming reader before loading
    /// and after inference so the acceptance claim covers the target bytes too.
    pub(super) fn validate_checkpoint(&self, root: &Path) {
        validate_checkpoint_artifacts(root, &self.record.provenance.artifacts);
    }

    /// Returns the exact tokenized prompt consumed by the official graph.
    pub(super) fn input_ids(&self) -> &[i64] {
        &self.record.input_ids
    }

    /// Returns the exact rendered non-thinking chat prompt, without retokenizing it.
    pub(super) fn prompt(&self) -> &str {
        &self.record.prompt
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

/// Streams the authenticated source inventory from a target checkpoint root.
fn validate_checkpoint_artifacts(root: &Path, artifacts: &[Artifact]) {
    for artifact in artifacts {
        let path = root.join(&artifact.relative_path);
        let observed = super::sha256_regular_file(&path, artifact.bytes);
        assert_eq!(
            observed,
            artifact.sha256,
            "target checkpoint artifact {} differs from the authenticated official source",
            path.display()
        );
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
    assert_eq!(record.prompt, EXPECTED_PROMPT);
    assert_eq!(record.input_ids, EXPECTED_INPUT_IDS);
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
    /// Darwin flag preventing a symlink at the final opened path component.
    const DARWIN_O_NOFOLLOW: i32 = 0x0000_0100;
    /// Darwin nonblocking flag preventing a substituted FIFO from blocking open.
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
        .as_chunks::<{ size_of::<f32>() }>()
        .0
        .iter()
        .enumerate()
        .map(|(index, bytes)| {
            let value = f32::from_le_bytes(*bytes);
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
    //! Checks fixed payload geometry, decoding, and fail-closed authentication.

    use super::{
        Artifact, MATRIX_BYTES, PROMPT_BYTES, assert_sha256, decode_f32, matrix_row,
        require_trusted_anchor, sha256, validate_checkpoint_artifacts,
    };

    /// The pinned vocabulary and decision count produce exact F32 payload sizes.
    #[test]
    fn fixed_payload_geometry_is_exact() {
        assert_eq!(PROMPT_BYTES, 993_280);
        assert_eq!(MATRIX_BYTES, 31_784_960);
    }

    /// Little-endian bytes decode correctly and rows retain vocabulary offsets.
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

    /// Loading without an independently pinned trust anchor must panic.
    #[test]
    #[should_panic(expected = "trusted Qwen3.5-0.8B fixture SHA-256 is not pinned")]
    fn missing_trust_anchor_fails_closed() {
        require_trusted_anchor("fixture", None);
    }

    /// A short uppercase digest is rejected as a malformed SHA-256 anchor.
    #[test]
    #[should_panic(expected = "malformed SHA-256")]
    fn malformed_trust_anchor_is_rejected() {
        assert_sha256("ABC");
    }

    /// Decoding a NaN payload fails instead of admitting non-finite evidence.
    #[test]
    #[should_panic(expected = "contains non-finite value")]
    fn non_finite_reference_value_is_rejected() {
        decode_f32(&f32::NAN.to_le_bytes(), "fixture");
    }

    /// Identically shaped checkpoint bytes must still match the reviewed digest.
    #[cfg(target_os = "macos")]
    #[test]
    fn target_checkpoint_authentication_rejects_same_size_mutation() {
        let temporary = tempfile::tempdir().expect("temporary checkpoint");
        let path = temporary.path().join("weights.bin");
        std::fs::write(&path, b"original").expect("fixture bytes");
        let artifacts = [Artifact {
            relative_path: "weights.bin".into(),
            bytes: 8,
            sha256: sha256(b"original"),
        }];
        validate_checkpoint_artifacts(temporary.path(), &artifacts);
        std::fs::write(&path, b"modified").expect("mutated bytes");
        assert!(
            std::panic::catch_unwind(|| {
                validate_checkpoint_artifacts(temporary.path(), &artifacts);
            })
            .is_err()
        );
    }
}
