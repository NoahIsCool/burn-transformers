//! Strict reader for the external Qwen3-30B-A3B official-reference bundle.
//!
//! The Python workflow stores F32 little-endian vocabulary rows so the Rust
//! Metal gate can compare complete distributions rather than a few sampled
//! logits.  This reader authenticates the producing scripts, pinned checkpoint
//! inventory, Transformers source revision, environment policy, and every raw
//! payload before returning any values to a numerical gate.

#![allow(dead_code)]

use super::{
    CHECKPOINT_ARTIFACT_BYTES, CHECKPOINT_TENSOR_BYTES, CONTROL_FILE_SHA256, REPOSITORY, REVISION,
    SHARD_FILE_SHA256, SIX_TOKEN_IDS, SIX_TOKEN_PROMPT,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

#[cfg(target_os = "macos")]
use std::fs::OpenOptions;
#[cfg(target_os = "macos")]
use std::os::unix::fs::OpenOptionsExt;

/// Environment variable selecting the immutable, create-only bundle.
const REFERENCE_DIR_ENV: &str = "TRANSFORMERS_QWEN3_MOE_REFERENCE_DIR";

/// Fixed directory name published by the official generator.
const BUNDLE_DIRNAME: &str = "qwen3-30b-a3b-ad44e777-reference.bundle";

/// Strict JSON manifest inside the reference bundle.
const JSON_FILENAME: &str = "qwen3-30b-a3b-ad44e777-reference.json";

/// Full vocabulary after the six-token stateless prompt.
const PROMPT_LOGITS_FILENAME: &str = "qwen3-30b-a3b-ad44e777-six-token-f16-last-logits.f32le";

/// Full vocabularies emitted by the cached prompt plus 31 decode calls.
const CACHED_LOGITS_FILENAME: &str = "qwen3-30b-a3b-ad44e777-cached-logits-32x151936.f32le";

/// Full vocabularies at all 32 teacher-forced continuation prefixes.
const TEACHER_LOGITS_FILENAME: &str =
    "qwen3-30b-a3b-ad44e777-teacher-prefix-logits-32x151936.f32le";

/// Official Transformers release whose wheel and source bytes are pinned.
const REFERENCE_SOURCE_RELEASE: &str = "transformers==5.2.0";

/// Annotated official Git tag used to resolve the immutable source commit.
const REFERENCE_SOURCE_TAG: &str = "v5.2.0";

/// Annotated Git tag object returned by the official repository.
const REFERENCE_SOURCE_TAG_OBJECT: &str = "0df260f32d71eb44327bed3c67de033dac004c8b";

/// Commit reached by peeling the official `v5.2.0` annotated tag.
const REFERENCE_SOURCE_COMMIT: &str = "7d9754a05193eb79b1d86aa744b622b8068008cd";

/// SHA-256 over package versions and exact dist-info METADATA/RECORD bytes.
const DISTRIBUTION_MANIFEST_SHA256: &str =
    "b814bb3254d51a28c6d74e80c04c3532c9836a0aadc4fab0eed410d6612d59dd";

/// Reviewed digest of the canonical JSON run record.
///
/// This deliberately remains `None` until the first real official run is
/// reviewed. Ignored acceptance tests therefore fail closed rather than
/// accepting a manifest that merely authenticates itself.
const TRUSTED_REFERENCE_JSON_SHA256: Option<&str> = None;

/// Reviewed digest of the full-vocabulary stateless prompt payload.
const TRUSTED_PROMPT_LOGITS_SHA256: Option<&str> = None;

/// Reviewed digest of all 32 official cached generation rows.
const TRUSTED_CACHED_LOGITS_SHA256: Option<&str> = None;

/// Reviewed digest of all 32 official teacher-forced rows.
const TRUSTED_TEACHER_LOGITS_SHA256: Option<&str> = None;

/// Number of elements in one vocabulary row.
const VOCAB_SIZE: usize = 151_936;

/// Number of required non-EOS greedy decisions.
const GREEDY_TOKENS: usize = 32;

/// Exact F32 bytes in the stateless prompt vector.
const PROMPT_LOGITS_BYTES: u64 = (VOCAB_SIZE * 4) as u64;

/// Exact F32 bytes in the 32-row teacher-forced matrix.
const TEACHER_LOGITS_BYTES: u64 = (GREEDY_TOKENS * VOCAB_SIZE * 4) as u64;

/// Loaded F16 elements in the official class after expert packing.
const EXPECTED_PARAMETER_ELEMENTS: u64 = 30_532_122_624;

/// Exact official Python package versions in the reference environment.
const EXPECTED_PACKAGES: [(&str, &str); 6] = [
    ("accelerate", "1.12.0"),
    ("numpy", "2.4.3"),
    ("safetensors", "0.7.0"),
    ("tokenizers", "0.22.2"),
    ("torch", "2.10.0"),
    ("transformers", "5.2.0"),
];

/// Exact dist-info identities bound into the package-manifest digest.
const EXPECTED_DISTRIBUTION_FILES: [(&str, &str, &str, u64, &str); 12] = [
    (
        "accelerate",
        "metadata",
        "accelerate-1.12.0.dist-info/METADATA",
        19_601,
        "31d880e2aa1eaffb50dfecb62bc81d53f0262ed19ff374e7ff7a76b41b87c6b8",
    ),
    (
        "accelerate",
        "record",
        "accelerate-1.12.0.dist-info/RECORD",
        14_253,
        "ebc6c296fe7d6bf449f6b776a4839cba2c8e8d8b649346ccfd2d505feea18f62",
    ),
    (
        "numpy",
        "metadata",
        "numpy-2.4.3.dist-info/METADATA",
        6_608,
        "621bf875d57eee8673f39344ea98ddc370e944e9c8c36b38b430e290f2b682cf",
    ),
    (
        "numpy",
        "record",
        "numpy-2.4.3.dist-info/RECORD",
        126_949,
        "b4b4d441229ce2edbbd2417b0b993d7b72d0ac801770fe2d32b26d8e1ab29829",
    ),
    (
        "safetensors",
        "metadata",
        "safetensors-0.7.0.dist-info/METADATA",
        4_125,
        "6809326fbf17444dd520ab19f0a55b7a18476dbe52a430466bbf5389967c2aaa",
    ),
    (
        "safetensors",
        "record",
        "safetensors-0.7.0.dist-info/RECORD",
        1_573,
        "f4f493f241d26041cf94fe7214f88c8f255df145eb064abac8550bc533d38f36",
    ),
    (
        "tokenizers",
        "metadata",
        "tokenizers-0.22.2.dist-info/METADATA",
        7_254,
        "15a5ddaf489f592b77e0a934c0eeb46b51130a94064ca129232afdb0a3868efc",
    ),
    (
        "tokenizers",
        "record",
        "tokenizers-0.22.2.dist-info/RECORD",
        3_778,
        "cd5bb15216c5d083bbe39ccc172f58e4114a8fe5b786f4f8f8239e53053ad9c3",
    ),
    (
        "torch",
        "metadata",
        "torch-2.10.0.dist-info/METADATA",
        31_092,
        "4b0f1217de69037355b4e582905cdb303a56a488ac6fd172f148d88fed634f54",
    ),
    (
        "torch",
        "record",
        "torch-2.10.0.dist-info/RECORD",
        1_477_169,
        "22546c990d209800cd92db2b052a8ea478cee3a6e9253485f282221e0a6225ae",
    ),
    (
        "transformers",
        "metadata",
        "transformers-5.2.0.dist-info/METADATA",
        32_319,
        "4340d0f550bf77d6595e4efafe066cd88306307577701df8e558ad620551c65f",
    ),
    (
        "transformers",
        "record",
        "transformers-5.2.0.dist-info/RECORD",
        394_090,
        "98d165cb699677711c15cc42eeeb25f7889fe164ecd90135f34e4b88109f931e",
    ),
];

/// Aggregate identity of every authenticated non-bytecode wheel payload.
const EXPECTED_DISTRIBUTION_PAYLOADS: [(&str, u64, u64, u64, &str); 6] = [
    (
        "accelerate",
        95,
        1_483_290,
        84,
        "1a5c1e846d1805592fe0090db994c3afc37f552ca6f4aa8b1d56f4f08bbfbfae",
    ),
    (
        "numpy",
        1_040,
        20_254_356,
        487,
        "bc49cc4a1cecd0b0e258d25f684b2250330c9672fcb64023b70eaf1f5d463f32",
    ),
    (
        "safetensors",
        14,
        1_070_455,
        7,
        "d0d712a1786e52b270faa8b558587b1cf41fd8255499c5eb25b967c2c9911afa",
    ),
    (
        "tokenizers",
        29,
        8_506_298,
        16,
        "b87e4f97df12e3603e619104d1b824792f6704cd405b340208980325844ff257",
    ),
    (
        "torch",
        12_340,
        338_442_177,
        2_199,
        "e7fc648575316519c0c996f4dea905684caff4aa6edc979e3173a4ee50209570",
    ),
    (
        "transformers",
        2_128,
        43_591_674,
        2_120,
        "e074d191c22d4b2dc0cb53d1a3afbd2f2cf81a8a2a8e8d8b685a8233c5ce09bc",
    ),
];

/// Exact official sources which participate in loading or forward execution.
const REFERENCE_SOURCE_HASHES: [(&str, &str, u64, &str); 21] = [
    (
        "transformers.activations",
        "activations.py",
        13_176,
        "10a5cb482f854c5d3afbe5bb1deab5f14ff21c800d08a38b1028538b9e1f3867",
    ),
    (
        "transformers.cache_utils",
        "cache_utils.py",
        60_147,
        "e4fe20ca1eeb17ea0ce7e62d1dbd0d6eab32e7072044f293612403b8e0c477fd",
    ),
    (
        "transformers.configuration_utils",
        "configuration_utils.py",
        57_894,
        "52100272e3bf59551c072a064053e90bc30f1fd0aa9e48fac7ad4b675c63ebcc",
    ),
    (
        "transformers.conversion_mapping",
        "conversion_mapping.py",
        19_302,
        "c8efcc6ce2224939299804d495ba4f361ab596e59f33e3db28e9b4629fc481e0",
    ),
    (
        "transformers.core_model_loading",
        "core_model_loading.py",
        57_649,
        "034f860989056b7c84ee6b6fbec197ac753d16bec120a588eef053d2d674182c",
    ),
    (
        "transformers.initialization",
        "initialization.py",
        10_979,
        "589934e5f43e62ea2d8923f5c38a0b917d8c8c1377f416c5a1f15b8977f0ecf1",
    ),
    (
        "transformers.integrations.__init__",
        "integrations/__init__.py",
        9_612,
        "f254dc19097673833d373a49acb071ab7d60e6df49c1376a70d4a8038ba47d18",
    ),
    (
        "transformers.integrations.hub_kernels",
        "integrations/hub_kernels.py",
        16_705,
        "d219a69a033322faae53950208748881e5073d2b8ef279b0fc3337a865042e04",
    ),
    (
        "transformers.integrations.moe",
        "integrations/moe.py",
        16_997,
        "1fe2594df6a3e870b87ccdf22c9b40d9b020f99ab3b90fca6ad43eca195ffa13",
    ),
    (
        "transformers.masking_utils",
        "masking_utils.py",
        74_373,
        "c3c82f7b7b6e03d3f04ba6c6c58a3dd6910623636452ec67ef70e3eb522f9fe7",
    ),
    (
        "transformers.modeling_attn_mask_utils",
        "modeling_attn_mask_utils.py",
        21_930,
        "fb3ec46639a19fda1eb99339f4e7770332a0f4f75b9de4abb34a0b1ee379838b",
    ),
    (
        "transformers.modeling_layers",
        "modeling_layers.py",
        11_476,
        "9e4726c07cfc895d95f9a75b0e7d23b8561575080811695e70fa879b0a18dae4",
    ),
    (
        "transformers.modeling_outputs",
        "modeling_outputs.py",
        108_678,
        "bb216b6b9c4b3d69f277367b3711c5f34c8e42d7f98e0a3930a64d107465f7f0",
    ),
    (
        "transformers.modeling_rope_utils",
        "modeling_rope_utils.py",
        52_962,
        "fa61288f66ce00bf060d3a16cc8e58aef2f94fe368c961b684bbb4bd19f90b2d",
    ),
    (
        "transformers.modeling_utils",
        "modeling_utils.py",
        247_340,
        "2f45c2139ce633e6e3eae9290675dd5815643b37c27f2cc4dd83586fb648f7bb",
    ),
    (
        "transformers.models.qwen3_moe.configuration_qwen3_moe",
        "models/qwen3_moe/configuration_qwen3_moe.py",
        10_579,
        "5277762b80c2a5e6158adae7c806989f61d8049af7278217ff2ec5a80833fc6d",
    ),
    (
        "transformers.models.qwen3_moe.modeling_qwen3_moe",
        "models/qwen3_moe/modeling_qwen3_moe.py",
        32_741,
        "566091429f4362f81b5bc52cbb3b3a06603b4ee8b35446ff0087b2b02be02ed0",
    ),
    (
        "transformers.processing_utils",
        "processing_utils.py",
        96_329,
        "ea1ed16955c829323ea95566ca8f7f7e01179d99ea4ab290000c980b0b837a77",
    ),
    (
        "transformers.utils.generic",
        "utils/generic.py",
        33_857,
        "4f4b1d947507d946f8c755ded8e160cb15862c122c0f977ed04a1b0be52cd9fe",
    ),
    (
        "transformers.utils.loading_report",
        "utils/loading_report.py",
        10_082,
        "43bc59a23a1a3aa4a9e099f97a3bf73f3aca96c89cbc126e679e4b31fb526449",
    ),
    (
        "transformers.utils.output_capturing",
        "utils/output_capturing.py",
        12_602,
        "af0454a352e7a14a4d52d3d3def16442fb74863bd924f1c6c8fc64e28eae67de",
    ),
];

/// Values returned only after every bundle invariant has passed.
pub(super) struct ReferenceBundle {
    /// Full F32 stateless prompt vocabulary.
    pub(super) prompt_logits: Vec<f32>,
    /// Thirty-two row-major F32 cached generation vocabularies.
    pub(super) cached_logits: Vec<f32>,
    /// Thirty-two row-major F32 teacher-forced vocabularies.
    pub(super) teacher_prefix_logits: Vec<f32>,
    /// Official cached greedy token IDs used by later Metal comparison.
    pub(super) greedy_token_ids: Vec<u32>,
}

/// Exact schema produced by `generate_qwen3_moe_reference.py`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceManifest {
    /// Version of the closed reference-manifest schema.
    schema_version: u32,
    /// Hugging Face repository that owns the checkpoint.
    repository: String,
    /// Immutable checkpoint revision used by the reference generator.
    revision: String,
    /// Official model class whose implementation produced the logits.
    reference_class: String,
    /// Human-readable release identifier for the official reference source.
    reference_source_release: String,
    /// Pinned source release tag name.
    reference_source_tag: String,
    /// Git object identity of the pinned source tag.
    reference_source_tag_object: String,
    /// Commit to which the official source tag resolves.
    reference_source_commit: String,
    /// Literal text tokenized to construct the fixed prompt.
    prompt: String,
    /// Exact token IDs passed for the fixed prompt.
    input_ids: Vec<u32>,
    /// Numerical, kernel, and deterministic execution settings.
    execution: ReferenceExecution,
    /// Interpreter, package, device, and offline-environment evidence.
    environment: ReferenceEnvironment,
    /// Authenticated generator, official source, checkpoint, and loader identities.
    provenance: ReferenceProvenance,
    /// Payload descriptor for the stateless prompt final-position logits.
    prompt_last_logits: PromptLogits,
    /// Payload descriptor for the cached continuation decision rows.
    cached_generation_logits: TeacherLogits,
    /// Payload descriptor for the stateless teacher-forced decision rows.
    teacher_prefix_logits: TeacherLogits,
    /// Cached greedy continuation and decision-stability diagnostics.
    generation: ReferenceGeneration,
    /// Teacher-forced token alignment and cached-versus-stateless diagnostics.
    teacher_forced: TeacherForced,
    /// Observed wall-clock durations in seconds; these are diagnostic only.
    timings_seconds: ReferenceTimings,
}

/// Deterministic graph/backend choices made by the reference workflow.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceExecution {
    /// Floating-point dtype used for official model execution.
    dtype: String,
    /// Device selected for model execution.
    device: String,
    /// Official attention implementation selected by the loader.
    attention_implementation: String,
    /// Official mixture-of-experts implementation selected by the loader.
    experts_implementation: String,
    /// Whether deterministic PyTorch algorithms were required.
    deterministic_algorithms: bool,
    /// Random seed fixed before official model execution.
    seed: u64,
    /// Whether cached generation was enabled.
    use_cache: bool,
    /// PyTorch policy controlling F32 matrix multiplication precision.
    float32_matmul_precision: String,
    /// Configured fraction of recommended MPS memory available to this process.
    mps_per_process_memory_fraction: f64,
    /// Whether every forward received an explicit all-valid attention mask.
    explicit_all_one_attention_masks: bool,
    /// Whether remotely supplied Hub kernels were enabled.
    hub_kernels_enabled: bool,
    /// Declared integrity assumptions under which reference files were authenticated.
    integrity_threat_model: String,
    /// Resolved module paths checked against authenticated source roots.
    authenticated_import_resolutions: BTreeMap<String, String>,
    /// Imported module bindings recorded after authentication.
    imported_module_bindings: BTreeMap<String, String>,
    /// Evidence that imports ignored and did not write unauthenticated bytecode.
    authenticated_import_bytecode_policy: BytecodePolicy,
    /// Recorded scope and restoration of the performance-only allocator warmup override.
    allocator_warmup_override: AllocatorWarmupOverride,
}

/// Proof that imports could not consume or leave unauthenticated bytecode.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BytecodePolicy {
    /// Whether source-adjacent compiled Python bytecode was bypassed during imports.
    adjacent_bytecode_ignored: bool,
    /// Whether Python bytecode writes were disabled during authenticated imports.
    writes_disabled: bool,
    /// Whether the private bytecode cache stayed empty throughout the import scope.
    private_cache_remained_empty: bool,
    /// Whether the previous bytecode policy was restored after the import scope.
    restored: bool,
}

/// Auditable scope for the one performance-only loader interception.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AllocatorWarmupOverride {
    /// Qualified loader function temporarily intercepted by the warmup override.
    target: String,
    /// Named policy applied to allocator warmup calls.
    policy: String,
    /// Loader operation during which the temporary override was active.
    scope: String,
    /// Recorded rationale for bypassing the allocator warmup.
    reason: String,
    /// Number of intercepted allocator warmup calls.
    call_count: usize,
    /// Number of entries in the expanded loader device map.
    expanded_device_map_entries: usize,
    /// Distinct device types found in the expanded device map.
    expanded_device_types: Vec<String>,
    /// Whether a Hugging Face quantizer was present during the intercepted call.
    hf_quantizer_present: bool,
    /// Whether the original allocator warmup function was restored.
    restored: bool,
}

/// Package, interpreter, device, and offline-policy provenance.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceEnvironment {
    /// UTC timestamp at which the reference bundle was generated.
    generated_at_utc: String,
    /// Argument vector used to invoke the reference generator.
    command: Vec<String>,
    /// Python interpreter version, executable, implementation, and platform metadata.
    python: serde_json::Value,
    /// Installed versions of the packages required by the reference run.
    packages: BTreeMap<String, String>,
    /// Authenticated wheel metadata and payload evidence keyed by package name.
    distribution_provenance: BTreeMap<String, DistributionProvenance>,
    /// Digest of the canonical installed-distribution metadata manifest.
    distribution_manifest_sha256: String,
    /// Optional packages whose absence was required and recorded.
    absent_optional_distributions: Vec<String>,
    /// Reported MPS device build and availability metadata.
    device: serde_json::Value,
    /// Environment-variable values enforcing offline artifact access.
    offline_environment: BTreeMap<String, String>,
}

/// Exact version and dist-info files for one installed wheel.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DistributionProvenance {
    /// Installed distribution version recorded for the reference environment.
    version: String,
    /// Authenticated identity of the wheel METADATA file.
    metadata: DistributionFile,
    /// Authenticated identity of the wheel RECORD file.
    record: DistributionFile,
    /// Aggregate verification of the wheel files listed by RECORD.
    payload: DistributionPayload,
}

/// Identity of one wheel METADATA or RECORD file.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DistributionFile {
    /// File path relative to its authenticated artifact root.
    relative_path: String,
    /// Exact file length in bytes.
    bytes: u64,
    /// Lowercase SHA-256 digest of the complete file contents.
    sha256: String,
}

/// Aggregate verification record for every hashed wheel RECORD entry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DistributionPayload {
    /// Number of wheel payload files authenticated against their recorded digests.
    verified_files: u64,
    /// Total byte count of the authenticated wheel payload files.
    verified_bytes: u64,
    /// Number of compiled Python bytecode files excluded from payload verification.
    ignored_bytecode_files: u64,
    /// Digest of the canonical verified wheel-payload manifest.
    manifest_sha256: String,
}

/// Generator, official source, checkpoint, and load identities.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceProvenance {
    /// Git and source-file identities of the reference generator and its helpers.
    generator: GeneratorProvenance,
    /// Authenticated official Transformers source files used during execution.
    reference_sources: Vec<ReferenceSource>,
    /// Authenticated checkpoint control files and tensor shards.
    artifacts: Vec<ReferenceArtifact>,
    /// Digest of the canonical checkpoint-artifact inventory.
    artifact_manifest_sha256: String,
    /// Total tensor-storage bytes declared by the checkpoint inventory.
    checkpoint_tensor_bytes: u64,
    /// Number of scalar parameter elements loaded into the official model.
    loaded_parameter_count: u64,
    /// Loaded scalar parameter counts grouped by dtype.
    parameter_dtype_elements: BTreeMap<String, u64>,
    /// Counts and digests for missing, unexpected, mismatched, and error loader reports.
    loading_info: BTreeMap<String, LoadingClass>,
}

/// Git state and exact local sources used by the generator.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratorProvenance {
    /// Git commit recorded for the generator checkout.
    git_head: String,
    /// Git tree object recorded for the generator checkout.
    git_tree: String,
    /// Content identities and clean-state flags for generator and helper source files.
    files: Vec<GeneratorFile>,
}

/// One generator/helper source file authenticated by content.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratorFile {
    /// Generator or helper source path relative to the repository root.
    relative_path: String,
    /// Exact file length in bytes.
    bytes: u64,
    /// Lowercase SHA-256 digest of the complete file contents.
    sha256: String,
    /// Whether the source file differed from its recorded Git checkout.
    git_dirty: bool,
}

/// One official Transformers source file authenticated by content.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceSource {
    /// Qualified Python module name of the authenticated official source.
    module: String,
    /// Source path relative to the pinned official package root.
    relative_path: String,
    /// Resolved source path used by the generator process.
    path: String,
    /// Exact official source-file length in bytes.
    bytes: u64,
    /// Lowercase SHA-256 digest of the official source contents.
    sha256: String,
}

/// One immutable checkpoint file authenticated before model loading.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceArtifact {
    /// File path relative to its authenticated artifact root.
    relative_path: String,
    /// Exact file length in bytes.
    bytes: u64,
    /// Lowercase SHA-256 digest of the complete file contents.
    sha256: String,
}

/// Stable classification for one official load-report category.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LoadingClass {
    /// Number of entries in this official loader diagnostic category.
    count: usize,
    /// Digest of the canonical diagnostic entries; empty categories hash empty bytes.
    sha256: String,
}

/// Integrity and diagnostic statistics for the stateless prompt vocabulary.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptLogits {
    /// Payload filename relative to the authenticated bundle root.
    relative_path: String,
    /// Binary scalar encoding, required to be little-endian F32.
    encoding: String,
    /// Ordered tensor-axis lengths used to validate payload geometry.
    shape: Vec<usize>,
    /// Total number of scalar elements in the payload.
    elements: usize,
    /// Exact binary payload length in bytes.
    bytes: u64,
    /// Lowercase SHA-256 digest authenticating the complete payload.
    sha256: String,
    /// Whether every exported prompt logit was reported finite.
    finite: bool,
    /// Stable highest-scoring vocabulary token for the prompt.
    top1_token_id: u32,
    /// Ten highest-scoring prompt entries retained in rank order.
    top10: Vec<IndexedLogit>,
    /// Prompt scores at predetermined vocabulary indices used as readable checks.
    probes: Vec<IndexedLogit>,
    /// Euclidean norm of the complete prompt vocabulary vector.
    l2_norm: f64,
    /// F64 diagnostic sum of all prompt vocabulary logits.
    sum: f64,
}

/// One token-indexed vocabulary score retained as a readable diagnostic.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexedLogit {
    /// Vocabulary index at which the diagnostic score was recorded.
    token_id: usize,
    /// Exported vocabulary score represented as F64 in JSON.
    logit: f64,
}

/// Integrity metadata for all teacher-forced comparison rows.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TeacherLogits {
    /// Payload filename relative to the authenticated bundle root.
    relative_path: String,
    /// Binary scalar encoding, required to be little-endian F32.
    encoding: String,
    /// Ordered tensor-axis lengths used to validate payload geometry.
    shape: Vec<usize>,
    /// Total number of scalar elements in the payload.
    elements: usize,
    /// Exact binary payload length in bytes.
    bytes: u64,
    /// Lowercase SHA-256 digest authenticating the complete payload.
    sha256: String,
    /// Stable highest-scoring token ID for each full-vocabulary decision row.
    row_top1_token_ids: Vec<u32>,
}

/// Cached greedy tokens, top-two scores, and stability proof.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceGeneration {
    /// Requested number of cached greedy generation decisions.
    max_new_tokens: usize,
    /// Generated continuation token IDs in decision order.
    token_ids: Vec<u32>,
    /// Winner-minus-runner-up logit margin at each cached decision.
    margins: Vec<f64>,
    /// Perturbation budget and safety rule used to assess greedy stability.
    stability_policy: StabilityPolicy,
    /// Top-two logits and binary16 perturbation evidence for each decision.
    decision_stability: Vec<DecisionStability>,
    /// Smallest observed winner-minus-runner-up margin across cached decisions.
    minimum_observed_margin: f64,
    /// Smallest per-decision ratio of logit margin to elementwise perturbation budget.
    minimum_margin_to_epsilon_ratio: f64,
    /// Decoded generated continuation text retained for human inspection.
    decoded_text: String,
    /// Token IDs designated as end-of-sequence by generation controls.
    eos_token_ids: Vec<u32>,
    /// Zero-based position of the first generated EOS token, if one occurred.
    first_eos_position: Option<usize>,
    /// Whether cached generation produced all 32 requested decisions.
    completed_32_decisions: bool,
}

/// Mathematical perturbation budget applied to all greedy decisions.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StabilityPolicy {
    /// Identifier of the recorded greedy-decision stability policy.
    name: String,
    /// Allowed per-element perturbation measured in binary16 units in the last place.
    elementwise_perturbation_ulps: f64,
    /// Additional safety multiplier required when comparing margin with perturbation.
    safety_factor: f64,
    /// Mathematical justification recorded for the chosen perturbation budget.
    rationale: String,
}

/// Complete evidence for one winner/runner-up comparison.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionStability {
    /// Zero-based cached generation decision index.
    step: usize,
    /// Vocabulary token with the largest logit.
    top1_token_id: u32,
    /// Vocabulary token with the second-largest logit.
    top2_token_id: u32,
    /// Winning logit represented as F64 in the JSON diagnostic.
    top1_logit: f64,
    /// Runner-up logit represented as F64 in the JSON diagnostic.
    top2_logit: f64,
    /// Winning logit minus runner-up logit.
    margin: f64,
    /// Binary16 spacing at the winning logit.
    top1_f16_ulp: f64,
    /// Binary16 spacing at the runner-up logit.
    top2_f16_ulp: f64,
    /// Absolute per-logit perturbation budget derived from binary16 spacing.
    elementwise_epsilon: f64,
    /// Observed top-two margin divided by the per-logit perturbation budget.
    margin_to_epsilon_ratio: f64,
    /// Whether the observed margin satisfies the declared perturbation safety rule.
    stable_under_budget: bool,
}

/// Stateless teacher-forced topology and cached self-parity evidence.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TeacherForced {
    /// Prompt followed by the first 31 generated tokens, predicting 32 continuation tokens.
    input_ids: Vec<u32>,
    /// Absolute input-token positions whose logits predict the recorded continuation.
    prediction_positions: Vec<usize>,
    /// Stateless argmax token IDs at every input-sequence position.
    all_position_top1_token_ids: Vec<u32>,
    /// Stateless argmax token IDs restricted to continuation prediction positions.
    continuation_top1_token_ids: Vec<u32>,
    /// Cached-generation argmax token IDs aligned with the continuation positions.
    cached_top1_token_ids: Vec<u32>,
    /// Whether corresponding stateless and cached argmax token IDs all agree.
    top1_equal: bool,
    /// Full-vocabulary cosines between corresponding cached and stateless decision rows.
    per_prefix_cosine: Vec<f64>,
    /// Smallest corresponding-row cosine across the continuation.
    minimum_cosine: f64,
    /// Largest absolute logit difference for each corresponding decision row.
    per_prefix_max_abs_error: Vec<f64>,
    /// Largest absolute logit difference over all corresponding decision rows.
    maximum_abs_error: f64,
    /// Full-vocabulary cosine between stateless prompt and cached-prefill logits.
    prompt_cached_stateless_cosine: f64,
}

/// Diagnostic wall-clock measurements, never used as performance gates.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceTimings {
    /// Observed model-loading wall-clock duration in seconds.
    load: f64,
    /// Observed stateless prompt-forward wall-clock duration in seconds.
    stateless_prompt: f64,
    /// Observed 32-decision cached-generation wall-clock duration in seconds.
    cached_generation_32: f64,
    /// Observed all-position teacher-forced forward wall-clock duration in seconds.
    teacher_forced_all_logits: f64,
}

/// Resolve the explicitly selected non-symlink create-only bundle root.
fn reference_root() -> PathBuf {
    let supplied =
        PathBuf::from(std::env::var_os(REFERENCE_DIR_ENV).unwrap_or_else(|| {
            panic!("set {REFERENCE_DIR_ENV} to the generated {BUNDLE_DIRNAME}")
        }));
    let metadata = fs::symlink_metadata(&supplied)
        .unwrap_or_else(|error| panic!("inspect {}: {error}", supplied.display()));
    assert!(metadata.is_dir(), "reference root must be a directory");
    assert!(
        !metadata.file_type().is_symlink(),
        "reference root must not be a symlink"
    );
    let root = supplied
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied.display()));
    assert_eq!(
        root.file_name().and_then(|name| name.to_str()),
        Some(BUNDLE_DIRNAME),
        "reference directory name changed"
    );
    root
}

/// Return lowercase SHA-256 for committed source bytes.
fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Return whether a string is one canonical lowercase SHA-256 digest.
fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Require one independently reviewed generated-artifact trust anchor.
fn trusted_digest(label: &str, digest: Option<&'static str>) -> &'static str {
    let digest = digest.unwrap_or_else(|| {
        panic!(
            "trusted {label} SHA-256 is not pinned; review the first official bundle, then set \
             its narrowly scoped constant in tests/support/qwen3_moe_reference.rs"
        )
    });
    assert!(is_sha256(digest), "trusted {label} SHA-256 is malformed");
    digest
}

/// Open one macOS descriptor without following a raced final symlink or FIFO.
#[cfg(target_os = "macos")]
fn open_regular_file_no_follow(path: &Path, maximum_bytes: u64) -> (File, u64) {
    /// Darwin open flag that rejects a symlink in the final path component.
    const DARWIN_O_NOFOLLOW: i32 = 0x0000_0100;
    /// Darwin open flag preventing a raced FIFO from blocking before descriptor validation.
    const DARWIN_O_NONBLOCK: i32 = 0x0000_0004;

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
    assert!(
        metadata.len() <= maximum_bytes,
        "{} exceeds its {maximum_bytes}-byte safety cap",
        path.display()
    );
    (file, metadata.len())
}

/// Make the macOS-only acceptance reader fail explicitly on other platforms.
#[cfg(not(target_os = "macos"))]
fn open_regular_file_no_follow(_path: &Path, _maximum_bytes: u64) -> (File, u64) {
    panic!("the Qwen3-MoE MPS reference reader requires macOS O_NOFOLLOW semantics")
}

/// Read one descriptor snapshot with a fixed allocation and a hard size cap.
fn read_regular_file_bounded(
    path: &Path,
    expected_bytes: Option<u64>,
    maximum_bytes: u64,
) -> Vec<u8> {
    let (file, opened_bytes) = open_regular_file_no_follow(path, maximum_bytes);
    if let Some(expected_bytes) = expected_bytes {
        assert_eq!(
            opened_bytes,
            expected_bytes,
            "{} byte length",
            path.display()
        );
    }
    let capacity = usize::try_from(opened_bytes).expect("bounded file length must fit usize");
    let mut payload = Vec::with_capacity(capacity);
    file.take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut payload)
        .unwrap_or_else(|error| panic!("read bounded file {}: {error}", path.display()));
    assert_eq!(
        payload.len(),
        capacity,
        "opened file {} changed length while being read",
        path.display()
    );
    payload
}

/// Decode and authenticate one canonical finite little-endian F32 payload.
fn read_f32le(path: &Path, expected_bytes: u64, expected_sha256: &str) -> Vec<f32> {
    let bytes = read_regular_file_bounded(path, Some(expected_bytes), expected_bytes);
    assert_eq!(
        sha256_bytes(&bytes),
        expected_sha256,
        "{} content changed",
        path.display()
    );
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte F32 chunk")))
        .collect::<Vec<_>>();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{} contains non-finite F32 values",
        path.display()
    );
    values
}

/// Return the lowest token index attaining the maximum finite score.
fn row_top1(row: &[f32]) -> u32 {
    assert_eq!(row.len(), VOCAB_SIZE, "vocabulary row width changed");
    let (index, _) = row
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
        .expect("vocabulary row must not be empty");
    u32::try_from(index).expect("vocabulary index must fit u32")
}

/// Return the deterministic top-two token IDs and their exact F64 margin.
fn row_top_two(row: &[f32]) -> (usize, usize, f64) {
    assert_eq!(row.len(), VOCAB_SIZE, "vocabulary row width changed");
    let ranks_before = |left: usize, right: usize| {
        row[left].total_cmp(&row[right]).is_gt()
            || (row[left].total_cmp(&row[right]).is_eq() && left < right)
    };
    let mut first = 0;
    let mut second = 1;
    if ranks_before(second, first) {
        std::mem::swap(&mut first, &mut second);
    }
    for candidate in 2..row.len() {
        if ranks_before(candidate, first) {
            second = first;
            first = candidate;
        } else if ranks_before(candidate, second) {
            second = candidate;
        }
    }
    (
        first,
        second,
        f64::from(row[first]) - f64::from(row[second]),
    )
}

/// Compute adjacent IEEE-754 binary16 spacing at an exactly represented value.
fn binary16_ulp(value: f64) -> f64 {
    assert!(value.is_finite(), "binary16 value must be finite");
    let magnitude = value.abs();
    assert!(magnitude <= 65_504.0, "value lies outside finite binary16");
    let ulp = if magnitude < 2.0_f64.powi(-14) {
        2.0_f64.powi(-24)
    } else {
        2.0_f64.powi(magnitude.log2().floor() as i32 - 10)
    };
    let units = value / ulp;
    assert_eq!(units, units.round(), "value is not exactly binary16");
    ulp
}

/// Compare one generated F64 diagnostic with scale-aware roundoff tolerance.
fn assert_reference_close(label: &str, actual: f64, expected: f64) {
    let tolerance = expected.abs().max(1.0) * 1.0e-8;
    assert!(
        actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= tolerance,
        "{label}: expected {expected}, found {actual}, tolerance {tolerance}"
    );
}

/// Compute full-vector cosine with F64 accumulation like the generator.
fn vector_cosine(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len());
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (&left, &right) in left.iter().zip(right) {
        let left = f64::from(left);
        let right = f64::from(right);
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    assert!(denominator.is_finite() && denominator > 0.0);
    dot / denominator
}

/// Compute maximum absolute difference between two vocabulary rows.
fn maximum_absolute_error(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&left, &right)| (f64::from(left) - f64::from(right)).abs())
        .fold(0.0, f64::max)
}

/// Validate exact generator file hashes against the committed test binary.
fn assert_generator_provenance(generator: &GeneratorProvenance) {
    assert_hex_sha1(&generator.git_head, "generator git_head");
    assert_hex_sha1(&generator.git_tree, "generator git_tree");
    let expected = BTreeMap::from([
        (
            "scripts/generate_qwen3_moe_reference.py",
            include_bytes!("../../scripts/generate_qwen3_moe_reference.py").as_slice(),
        ),
        (
            "scripts/generate_qwen38_reference.py",
            include_bytes!("../../scripts/generate_qwen38_reference.py").as_slice(),
        ),
    ]);
    assert_eq!(generator.files.len(), expected.len());
    let mut observed = BTreeSet::new();
    for file in &generator.files {
        assert!(
            observed.insert(file.relative_path.as_str()),
            "duplicate generator source {}",
            file.relative_path
        );
        let bytes = expected
            .get(file.relative_path.as_str())
            .unwrap_or_else(|| panic!("unexpected generator source {}", file.relative_path));
        assert!(!file.git_dirty, "{} was dirty", file.relative_path);
        assert_eq!(
            file.bytes,
            bytes.len() as u64,
            "{} bytes",
            file.relative_path
        );
        assert_eq!(
            file.sha256,
            sha256_bytes(bytes),
            "{} SHA-256",
            file.relative_path
        );
    }
}

/// Require one 40-character lowercase hexadecimal Git object identity.
fn assert_hex_sha1(value: &str, label: &str) {
    assert_eq!(value.len(), 40, "{label} length");
    assert!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} must be lowercase hexadecimal"
    );
}

/// Recompute the Python generator's canonical checkpoint-artifact digest.
fn artifact_manifest_sha256(artifacts: &[ReferenceArtifact]) -> String {
    let mut sorted = artifacts.iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let mut canonical = Vec::new();
    for item in sorted {
        canonical.extend_from_slice(item.relative_path.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(item.bytes.to_string().as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(item.sha256.as_bytes());
        canonical.push(b'\n');
    }
    sha256_bytes(&canonical)
}

/// Recompute the generator's canonical installed-distribution digest.
fn distribution_manifest_sha256(
    distributions: &BTreeMap<String, DistributionProvenance>,
) -> String {
    let mut canonical = Vec::new();
    for (name, distribution) in distributions {
        canonical.extend_from_slice(name.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(distribution.version.as_bytes());
        canonical.push(b'\n');
        for (logical_name, file) in [
            ("metadata", &distribution.metadata),
            ("record", &distribution.record),
        ] {
            canonical.extend_from_slice(logical_name.as_bytes());
            canonical.push(0);
            canonical.extend_from_slice(file.relative_path.as_bytes());
            canonical.push(0);
            canonical.extend_from_slice(file.bytes.to_string().as_bytes());
            canonical.push(0);
            canonical.extend_from_slice(file.sha256.as_bytes());
            canonical.push(b'\n');
        }
    }
    sha256_bytes(&canonical)
}

/// Validate the complete pinned checkpoint identity carried by the manifest.
fn assert_checkpoint_provenance(provenance: &ReferenceProvenance) {
    let expected = CONTROL_FILE_SHA256
        .into_iter()
        .chain(SHARD_FILE_SHA256)
        .collect::<BTreeMap<_, _>>();
    assert_eq!(provenance.artifacts.len(), expected.len());
    let mut observed_names = BTreeSet::new();
    let mut shard_bytes = 0_u64;
    for artifact in &provenance.artifacts {
        assert!(
            observed_names.insert(artifact.relative_path.as_str()),
            "duplicate checkpoint artifact {}",
            artifact.relative_path
        );
        assert_eq!(
            expected.get(artifact.relative_path.as_str()).copied(),
            Some(artifact.sha256.as_str()),
            "checkpoint artifact {} identity",
            artifact.relative_path
        );
        assert!(artifact.bytes > 0, "checkpoint artifact must be nonempty");
        if artifact.relative_path.ends_with(".safetensors") {
            shard_bytes = shard_bytes
                .checked_add(artifact.bytes)
                .expect("checkpoint shard-byte sum must fit u64");
        }
    }
    assert_eq!(shard_bytes, CHECKPOINT_ARTIFACT_BYTES);
    assert_eq!(provenance.checkpoint_tensor_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(
        provenance.loaded_parameter_count,
        EXPECTED_PARAMETER_ELEMENTS
    );
    assert_eq!(
        provenance.parameter_dtype_elements,
        BTreeMap::from([("float16".to_owned(), EXPECTED_PARAMETER_ELEMENTS)])
    );
    assert_eq!(
        provenance.artifact_manifest_sha256,
        artifact_manifest_sha256(&provenance.artifacts)
    );

    let empty_sha256 = sha256_bytes(b"");
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
            "unexpected_keys"
        ])
    );
    for class in provenance.loading_info.values() {
        assert_eq!(class.count, 0);
        assert_eq!(class.sha256, empty_sha256);
    }
}

/// Validate every official source identity against the reviewed source release.
fn assert_reference_sources(sources: &[ReferenceSource]) {
    let expected = REFERENCE_SOURCE_HASHES
        .into_iter()
        .map(|(module, relative_path, bytes, sha256)| (module, (relative_path, bytes, sha256)))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        sources.len(),
        expected.len(),
        "official source inventory changed"
    );
    let mut observed = BTreeMap::new();
    let mut relative_paths = BTreeSet::new();
    for source in sources {
        assert!(
            observed.insert(source.module.as_str(), source).is_none(),
            "duplicate source module {}",
            source.module
        );
        assert!(
            relative_paths.insert(source.relative_path.as_str()),
            "duplicate source path {}",
            source.relative_path
        );
        assert!(Path::new(&source.path).ends_with(&source.relative_path));
    }
    assert_eq!(
        observed.keys().copied().collect::<BTreeSet<_>>(),
        expected.keys().copied().collect()
    );
    for (module, (relative_path, bytes, sha256)) in expected {
        let source = observed[module];
        assert_eq!(source.relative_path, relative_path, "{module} path");
        assert_eq!(source.bytes, bytes, "{module} bytes");
        assert_eq!(source.sha256, sha256, "{module} SHA-256");
    }
}

/// Validate exact package versions and process-wide offline controls.
fn assert_environment(environment: &ReferenceEnvironment) {
    assert!(environment.generated_at_utc.ends_with('Z'));
    assert!(
        environment
            .command
            .iter()
            .any(|part| part.ends_with("generate_qwen3_moe_reference.py"))
    );
    assert_eq!(
        environment.packages,
        EXPECTED_PACKAGES
            .into_iter()
            .map(|(name, version)| (name.to_owned(), version.to_owned()))
            .collect()
    );
    assert_eq!(
        environment
            .distribution_provenance
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        EXPECTED_PACKAGES
            .into_iter()
            .map(|(name, _)| name)
            .collect::<BTreeSet<_>>()
    );
    let expected_files = EXPECTED_DISTRIBUTION_FILES
        .into_iter()
        .map(|(name, logical, path, bytes, sha256)| ((name, logical), (path, bytes, sha256)))
        .collect::<BTreeMap<_, _>>();
    let expected_payloads = EXPECTED_DISTRIBUTION_PAYLOADS
        .into_iter()
        .map(|(name, files, bytes, bytecode, sha256)| (name, (files, bytes, bytecode, sha256)))
        .collect::<BTreeMap<_, _>>();
    for (name, distribution) in &environment.distribution_provenance {
        assert_eq!(
            environment.packages.get(name),
            Some(&distribution.version),
            "{name} version provenance"
        );
        for (logical, file) in [
            ("metadata", &distribution.metadata),
            ("record", &distribution.record),
        ] {
            let (path, bytes, sha256) = expected_files[&(name.as_str(), logical)];
            assert_eq!(file.relative_path, path, "{name} {logical} path");
            assert_eq!(file.bytes, bytes, "{name} {logical} bytes");
            assert_eq!(file.sha256, sha256, "{name} {logical} SHA-256");
        }
        let (files, bytes, bytecode, sha256) = expected_payloads[name.as_str()];
        assert_eq!(distribution.payload.verified_files, files);
        assert_eq!(distribution.payload.verified_bytes, bytes);
        assert_eq!(distribution.payload.ignored_bytecode_files, bytecode);
        assert_eq!(distribution.payload.manifest_sha256, sha256);
    }
    let observed_distribution_digest =
        distribution_manifest_sha256(&environment.distribution_provenance);
    assert_eq!(
        environment.distribution_manifest_sha256,
        observed_distribution_digest
    );
    assert_eq!(observed_distribution_digest, DISTRIBUTION_MANIFEST_SHA256);
    assert_eq!(
        environment.absent_optional_distributions,
        ["flash-attn", "kernels", "torchao"]
    );
    assert!(environment.python.is_object());
    assert_eq!(environment.device["kind"], "mps");
    assert_eq!(environment.device["built"], true);
    assert_eq!(environment.device["available"], true);
    let recommended_memory = environment.device["recommended_max_memory_bytes"]
        .as_u64()
        .expect("recommended MPS memory must be an integer");
    let required_memory = environment.device["required_f16_load_bytes"]
        .as_u64()
        .expect("required MPS memory must be an integer");
    assert_eq!(
        environment.device["minimum_load_headroom_bytes"],
        2_000_000_000_u64
    );
    assert_eq!(required_memory, CHECKPOINT_TENSOR_BYTES + 2_000_000_000);
    assert!(recommended_memory >= required_memory);
    assert_eq!(
        environment.offline_environment,
        BTreeMap::from([
            ("HF_DATASETS_OFFLINE".to_owned(), "1".to_owned()),
            ("HF_DEACTIVATE_ASYNC_LOAD".to_owned(), "1".to_owned()),
            ("HF_HUB_OFFLINE".to_owned(), "1".to_owned()),
            ("PYTORCH_ENABLE_MPS_FALLBACK".to_owned(), "0".to_owned()),
            ("TOKENIZERS_PARALLELISM".to_owned(), "false".to_owned()),
            ("TRANSFORMERS_OFFLINE".to_owned(), "1".to_owned()),
            ("USE_HUB_KERNELS".to_owned(), "0".to_owned()),
        ])
    );
}

/// Validate deterministic execution choices and the scoped loader workaround.
fn assert_execution(execution: &ReferenceExecution) {
    assert_eq!(execution.dtype, "float16");
    assert_eq!(execution.device, "mps");
    assert_eq!(execution.attention_implementation, "eager");
    assert_eq!(execution.experts_implementation, "eager");
    assert!(execution.deterministic_algorithms);
    assert_eq!(execution.seed, 0);
    assert!(execution.use_cache);
    assert_eq!(execution.float32_matmul_precision, "highest");
    assert_eq!(execution.mps_per_process_memory_fraction, 1.0);
    assert!(execution.explicit_all_one_attention_masks);
    assert!(!execution.hub_kernels_enabled);
    assert_eq!(
        execution.integrity_threat_model,
        "Pre/post descriptor identities and SHA-256 detect persistent mutation and ordinary \
         races. An actively malicious same-UID process that swaps a pathname only while \
         Transformers or SafeTensors reopens it, then restores the original before postflight, \
         is outside scope because it can also tamper with this process."
    );
    assert_eq!(
        execution.authenticated_import_resolutions,
        execution.imported_module_bindings
    );
    assert_eq!(
        execution
            .authenticated_import_resolutions
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        EXPECTED_PACKAGES
            .into_iter()
            .map(|(name, _)| name)
            .collect::<BTreeSet<_>>()
    );
    assert!(
        execution
            .authenticated_import_resolutions
            .values()
            .all(|path| Path::new(path).is_absolute())
    );
    let bytecode = &execution.authenticated_import_bytecode_policy;
    assert!(bytecode.adjacent_bytecode_ignored);
    assert!(bytecode.writes_disabled);
    assert!(bytecode.private_cache_remained_empty);
    assert!(bytecode.restored);
    let override_ = &execution.allocator_warmup_override;
    assert_eq!(
        override_.target,
        "transformers.modeling_utils.caching_allocator_warmup"
    );
    assert_eq!(
        override_.policy,
        "mps-exclusive-noop-during-from-pretrained"
    );
    assert_eq!(override_.scope, "Qwen3MoeForCausalLM.from_pretrained");
    assert!(!override_.reason.is_empty());
    assert_eq!(override_.call_count, 1);
    assert_eq!(override_.expanded_device_map_entries, 531);
    assert_eq!(override_.expanded_device_types, ["mps"]);
    assert!(!override_.hf_quantizer_present);
    assert!(override_.restored);
}

/// Validate mathematical stability and cached/stateless teacher topology.
fn assert_generation(
    manifest: &ReferenceManifest,
    prompt_row: &[f32],
    cached_rows: &[f32],
    teacher_rows: &[f32],
) {
    let generation = &manifest.generation;
    assert_eq!(generation.max_new_tokens, GREEDY_TOKENS);
    assert_eq!(generation.token_ids.len(), GREEDY_TOKENS);
    assert_eq!(generation.margins.len(), GREEDY_TOKENS);
    assert_eq!(generation.decision_stability.len(), GREEDY_TOKENS);
    assert_eq!(generation.eos_token_ids, [151_645, 151_643]);
    assert_eq!(generation.first_eos_position, None);
    assert!(generation.completed_32_decisions);
    assert!(!generation.decoded_text.is_empty());
    assert_eq!(
        generation.stability_policy.name,
        "binary16-elementwise-ulp-bound"
    );
    assert_eq!(
        generation.stability_policy.elementwise_perturbation_ulps,
        1.0
    );
    assert_eq!(generation.stability_policy.safety_factor, 2.0);
    assert!(!generation.stability_policy.rationale.is_empty());

    let mut minimum_margin = f64::INFINITY;
    let mut minimum_ratio = f64::INFINITY;
    for (step, decision) in generation.decision_stability.iter().enumerate() {
        let row = &cached_rows[step * VOCAB_SIZE..(step + 1) * VOCAB_SIZE];
        let (top1, top2, raw_margin) = row_top_two(row);
        assert_eq!(decision.step, step);
        assert_eq!(decision.top1_token_id, generation.token_ids[step]);
        assert_eq!(decision.top1_token_id as usize, top1);
        assert_ne!(decision.top1_token_id, decision.top2_token_id);
        assert!((decision.top2_token_id as usize) < VOCAB_SIZE);
        assert!(decision.top1_logit.is_finite() && decision.top2_logit.is_finite());
        assert_eq!(decision.top1_logit as f32, row[top1]);
        // Torch deliberately leaves tied `topk` indices unspecified. Require
        // the recorded runner-up to attain the independently recomputed second
        // score without imposing a different tie-breaking contract.
        assert_eq!(decision.top2_logit as f32, row[top2]);
        assert_eq!(
            row[decision.top2_token_id as usize], row[top2],
            "recorded runner-up must attain the second-highest score"
        );
        assert_reference_close("raw top-two margin", raw_margin, decision.margin);
        assert_reference_close(
            "top-two subtraction margin",
            decision.top1_logit - decision.top2_logit,
            decision.margin,
        );
        assert_reference_close(
            "duplicated generation margin",
            decision.margin,
            generation.margins[step],
        );
        let top1_ulp = binary16_ulp(decision.top1_logit);
        let top2_ulp = binary16_ulp(decision.top2_logit);
        assert_reference_close("top-1 binary16 ULP", top1_ulp, decision.top1_f16_ulp);
        assert_reference_close("top-2 binary16 ULP", top2_ulp, decision.top2_f16_ulp);
        let epsilon = top1_ulp.max(top2_ulp);
        assert_reference_close("elementwise epsilon", epsilon, decision.elementwise_epsilon);
        assert_reference_close(
            "margin-to-epsilon ratio",
            decision.margin / epsilon,
            decision.margin_to_epsilon_ratio,
        );
        assert!(decision.margin_to_epsilon_ratio > 2.0);
        assert!(decision.stable_under_budget);
        minimum_margin = minimum_margin.min(decision.margin);
        minimum_ratio = minimum_ratio.min(decision.margin_to_epsilon_ratio);
    }
    assert_eq!(generation.minimum_observed_margin, minimum_margin);
    assert_eq!(generation.minimum_margin_to_epsilon_ratio, minimum_ratio);

    let cached_top1 = cached_rows
        .chunks_exact(VOCAB_SIZE)
        .map(row_top1)
        .collect::<Vec<_>>();
    assert_eq!(cached_top1, generation.token_ids);
    assert!(
        cached_top1
            .iter()
            .all(|token| !generation.eos_token_ids.contains(token)),
        "accepted continuation contains EOS"
    );
    assert_eq!(
        cached_top1,
        manifest.cached_generation_logits.row_top1_token_ids
    );
    let teacher_top1 = teacher_rows
        .chunks_exact(VOCAB_SIZE)
        .map(row_top1)
        .collect::<Vec<_>>();
    assert_eq!(teacher_top1, generation.token_ids);
    assert_eq!(
        teacher_top1,
        manifest.teacher_prefix_logits.row_top1_token_ids
    );

    let teacher = &manifest.teacher_forced;
    let mut expected_input = SIX_TOKEN_IDS.to_vec();
    expected_input.extend(generation.token_ids.iter().take(31));
    assert_eq!(teacher.input_ids, expected_input);
    assert_eq!(teacher.prediction_positions, (5..37).collect::<Vec<_>>());
    assert_eq!(teacher.all_position_top1_token_ids.len(), 37);
    assert_eq!(teacher.continuation_top1_token_ids, generation.token_ids);
    assert_eq!(teacher.cached_top1_token_ids, generation.token_ids);
    assert!(teacher.top1_equal);
    assert_eq!(teacher.per_prefix_cosine.len(), GREEDY_TOKENS);
    let observed_cosines = cached_rows
        .chunks_exact(VOCAB_SIZE)
        .zip(teacher_rows.chunks_exact(VOCAB_SIZE))
        .map(|(cached, teacher)| vector_cosine(cached, teacher))
        .collect::<Vec<_>>();
    for (observed, recorded) in observed_cosines.iter().zip(&teacher.per_prefix_cosine) {
        assert_reference_close("cached/teacher cosine", *observed, *recorded);
        assert!(*observed >= 0.999 && *observed <= 1.000_001);
    }
    assert_reference_close(
        "minimum cached/teacher cosine",
        observed_cosines
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        teacher.minimum_cosine,
    );
    assert_eq!(teacher.per_prefix_max_abs_error.len(), GREEDY_TOKENS);
    let observed_errors = cached_rows
        .chunks_exact(VOCAB_SIZE)
        .zip(teacher_rows.chunks_exact(VOCAB_SIZE))
        .map(|(cached, teacher)| maximum_absolute_error(cached, teacher))
        .collect::<Vec<_>>();
    for (observed, recorded) in observed_errors
        .iter()
        .zip(&teacher.per_prefix_max_abs_error)
    {
        assert_reference_close("cached/teacher maximum error", *observed, *recorded);
    }
    assert_reference_close(
        "overall cached/teacher maximum error",
        observed_errors.iter().copied().fold(0.0, f64::max),
        teacher.maximum_abs_error,
    );
    let prompt_cached_cosine = vector_cosine(prompt_row, &cached_rows[..VOCAB_SIZE]);
    assert_reference_close(
        "prompt/cached cosine",
        prompt_cached_cosine,
        teacher.prompt_cached_stateless_cosine,
    );
    assert!(prompt_cached_cosine >= 0.999);
}

/// Validate raw tensor metadata before trusting its file path or digest.
fn assert_tensor_metadata(manifest: &ReferenceManifest) {
    let prompt = &manifest.prompt_last_logits;
    assert_eq!(prompt.relative_path, PROMPT_LOGITS_FILENAME);
    assert_eq!(prompt.encoding, "ieee754-f32-little-endian");
    assert_eq!(prompt.shape, [VOCAB_SIZE]);
    assert_eq!(prompt.elements, VOCAB_SIZE);
    assert_eq!(prompt.bytes, PROMPT_LOGITS_BYTES);
    assert!(prompt.finite);
    assert!(prompt.l2_norm.is_finite() && prompt.l2_norm > 0.0);
    assert!(prompt.sum.is_finite());
    assert_eq!(prompt.top10.len(), 10);
    assert!(!prompt.probes.is_empty());

    for (tensor, filename) in [
        (&manifest.cached_generation_logits, CACHED_LOGITS_FILENAME),
        (&manifest.teacher_prefix_logits, TEACHER_LOGITS_FILENAME),
    ] {
        assert_eq!(tensor.relative_path, filename);
        assert_eq!(tensor.encoding, "ieee754-f32-little-endian");
        assert_eq!(tensor.shape, [GREEDY_TOKENS, VOCAB_SIZE]);
        assert_eq!(tensor.elements, GREEDY_TOKENS * VOCAB_SIZE);
        assert_eq!(tensor.bytes, TEACHER_LOGITS_BYTES);
        assert_eq!(tensor.row_top1_token_ids.len(), GREEDY_TOKENS);
    }
}

/// Recompute every human-readable prompt diagnostic from authenticated logits.
fn assert_prompt_diagnostics(prompt: &PromptLogits, values: &[f32]) {
    assert_eq!(values.len(), VOCAB_SIZE);
    let mut ranked = (0..values.len()).collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    assert_eq!(prompt.top10.len(), 10);
    for (record, expected_id) in prompt.top10.iter().zip(ranked.into_iter().take(10)) {
        assert_eq!(record.token_id, expected_id);
        assert_eq!(record.logit as f32, values[expected_id]);
    }
    let mut probes = BTreeSet::new();
    for probe in &prompt.probes {
        assert!(
            probe.token_id < values.len(),
            "prompt probe is out of bounds"
        );
        assert!(probes.insert(probe.token_id), "duplicate prompt probe");
        assert_eq!(probe.logit as f32, values[probe.token_id]);
    }
    let sum = values.iter().map(|value| f64::from(*value)).sum::<f64>();
    let norm = values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    assert_reference_close("prompt logit sum", sum, prompt.sum);
    assert_reference_close("prompt logit L2 norm", norm, prompt.l2_norm);
}

/// Require every diagnostic duration to be finite and nonnegative.
fn assert_timings(timings: &ReferenceTimings) {
    for value in [
        timings.load,
        timings.stateless_prompt,
        timings.cached_generation_32,
        timings.teacher_forced_all_logits,
    ] {
        assert!(value.is_finite() && value >= 0.0);
    }
}

/// Authenticate and decode the complete official bundle.
pub(super) fn load_and_validate_reference_bundle() -> ReferenceBundle {
    let trusted_json = trusted_digest("Qwen3-MoE reference JSON", TRUSTED_REFERENCE_JSON_SHA256);
    let trusted_prompt = trusted_digest("Qwen3-MoE prompt logits", TRUSTED_PROMPT_LOGITS_SHA256);
    let trusted_cached = trusted_digest("Qwen3-MoE cached logits", TRUSTED_CACHED_LOGITS_SHA256);
    let trusted_teacher = trusted_digest("Qwen3-MoE teacher logits", TRUSTED_TEACHER_LOGITS_SHA256);
    let root = reference_root();
    let expected_names = BTreeSet::from([
        JSON_FILENAME.to_owned(),
        PROMPT_LOGITS_FILENAME.to_owned(),
        CACHED_LOGITS_FILENAME.to_owned(),
        TEACHER_LOGITS_FILENAME.to_owned(),
    ]);
    let observed_names = fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("read {}: {error}", root.display()))
        .map(|entry| {
            let entry = entry.expect("read reference-bundle entry");
            let kind = entry.file_type().expect("inspect reference-bundle entry");
            assert!(kind.is_file(), "bundle entries must be regular files");
            assert!(!kind.is_symlink(), "bundle entries must not be symlinks");
            entry
                .file_name()
                .into_string()
                .expect("bundle filenames must be UTF-8")
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(observed_names, expected_names, "bundle entry set changed");

    let manifest_path = root.join(JSON_FILENAME);
    let manifest_bytes = read_regular_file_bounded(&manifest_path, None, 1024 * 1024);
    assert_eq!(
        sha256_bytes(&manifest_bytes),
        trusted_json,
        "reference manifest identity changed"
    );
    let manifest: ReferenceManifest = serde_json::from_slice(&manifest_bytes)
        .expect("reference manifest must match its strict schema");

    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.repository, REPOSITORY);
    assert_eq!(manifest.revision, REVISION);
    assert_eq!(manifest.reference_class, "transformers.Qwen3MoeForCausalLM");
    assert_eq!(manifest.reference_source_release, REFERENCE_SOURCE_RELEASE);
    assert_eq!(manifest.reference_source_tag, REFERENCE_SOURCE_TAG);
    assert_eq!(
        manifest.reference_source_tag_object,
        REFERENCE_SOURCE_TAG_OBJECT
    );
    assert_eq!(manifest.reference_source_commit, REFERENCE_SOURCE_COMMIT);
    assert_eq!(manifest.prompt, SIX_TOKEN_PROMPT);
    assert_eq!(manifest.input_ids, SIX_TOKEN_IDS);
    assert_execution(&manifest.execution);
    assert_environment(&manifest.environment);
    assert_generator_provenance(&manifest.provenance.generator);
    assert_reference_sources(&manifest.provenance.reference_sources);
    assert_checkpoint_provenance(&manifest.provenance);
    assert_tensor_metadata(&manifest);
    assert_timings(&manifest.timings_seconds);

    assert_eq!(manifest.prompt_last_logits.sha256, trusted_prompt);
    assert_eq!(manifest.cached_generation_logits.sha256, trusted_cached);
    assert_eq!(manifest.teacher_prefix_logits.sha256, trusted_teacher);

    let prompt_logits = read_f32le(
        &root.join(PROMPT_LOGITS_FILENAME),
        PROMPT_LOGITS_BYTES,
        trusted_prompt,
    );
    let cached_logits = read_f32le(
        &root.join(CACHED_LOGITS_FILENAME),
        TEACHER_LOGITS_BYTES,
        trusted_cached,
    );
    let teacher_prefix_logits = read_f32le(
        &root.join(TEACHER_LOGITS_FILENAME),
        TEACHER_LOGITS_BYTES,
        trusted_teacher,
    );
    assert_eq!(prompt_logits.len(), VOCAB_SIZE);
    assert_eq!(cached_logits.len(), GREEDY_TOKENS * VOCAB_SIZE);
    assert_eq!(teacher_prefix_logits.len(), GREEDY_TOKENS * VOCAB_SIZE);
    assert_eq!(
        row_top1(&prompt_logits),
        manifest.prompt_last_logits.top1_token_id
    );
    assert_eq!(
        manifest.prompt_last_logits.top1_token_id,
        manifest.generation.token_ids[0]
    );
    assert_prompt_diagnostics(&manifest.prompt_last_logits, &prompt_logits);
    assert_generation(
        &manifest,
        &prompt_logits,
        &cached_logits,
        &teacher_prefix_logits,
    );

    let final_names = fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("reread {}: {error}", root.display()))
        .map(|entry| {
            entry
                .expect("reread reference-bundle entry")
                .file_name()
                .into_string()
                .expect("bundle filenames must be UTF-8")
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        final_names, expected_names,
        "bundle changed while being read"
    );

    ReferenceBundle {
        prompt_logits,
        cached_logits,
        teacher_prefix_logits,
        greedy_token_ids: manifest.generation.token_ids,
    }
}

#[cfg(test)]
mod tests {
    //! Tests for bounded oracle decoding, trust anchors, ranking, and binary16 stability diagnostics.

    use super::*;

    /// Equal scores use the smaller token ID, matching Torch's stable top-k contract.
    #[test]
    fn row_top1_breaks_ties_by_lowest_token_id() {
        let mut row = vec![-1.0_f32; VOCAB_SIZE];
        row[7] = 3.0;
        row[2] = 3.0;
        assert_eq!(row_top1(&row), 2);
    }

    /// Top-two selection is descending and deterministically resolves score ties.
    #[test]
    fn row_top_two_recomputes_margin_and_ties() {
        let mut row = vec![-1.0_f32; VOCAB_SIZE];
        row[9] = 3.0;
        row[7] = 2.5;
        row[2] = 2.5;
        assert_eq!(row_top_two(&row), (9, 2, 0.5));
    }

    /// Binary16 spacing handles subnormal, bin-boundary, and signed values.
    #[test]
    fn binary16_spacing_matches_ieee_boundaries() {
        assert_eq!(binary16_ulp(0.0), 2.0_f64.powi(-24));
        assert_eq!(binary16_ulp(-0.0), 2.0_f64.powi(-24));
        assert_eq!(binary16_ulp(7.996_093_75), 2.0_f64.powi(-8));
        assert_eq!(binary16_ulp(8.0), 2.0_f64.powi(-7));
    }

    /// A generated bundle cannot authenticate itself before maintainer review.
    #[test]
    fn missing_generated_digest_fails_closed() {
        let failure = std::panic::catch_unwind(|| trusted_digest("fixture", None));
        assert!(failure.is_err());
    }

    /// The canonical artifact digest does not depend on input order.
    #[test]
    fn artifact_digest_is_order_independent() {
        let first = ReferenceArtifact {
            relative_path: "a".to_owned(),
            bytes: 1,
            sha256: "0".repeat(64),
        };
        let second = ReferenceArtifact {
            relative_path: "b".to_owned(),
            bytes: 2,
            sha256: "f".repeat(64),
        };
        assert_eq!(
            artifact_manifest_sha256(&[first, second]),
            artifact_manifest_sha256(&[
                ReferenceArtifact {
                    relative_path: "b".to_owned(),
                    bytes: 2,
                    sha256: "f".repeat(64),
                },
                ReferenceArtifact {
                    relative_path: "a".to_owned(),
                    bytes: 1,
                    sha256: "0".repeat(64),
                },
            ])
        );
    }
}
