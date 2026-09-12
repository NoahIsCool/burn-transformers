//! Strict, fail-closed reader for Nemotron base, MTP, and component oracles.
//!
//! Hugging Face Transformers commit `cfa309c...` executes the target backbone
//! but explicitly ignores `mtp.*`. The checkpoint card pins vLLM 0.27.1 for
//! serving, whose immutable tag commit `6e448d0...` contains a dedicated
//! Nemotron-H MTP implementation. The card's SGLang MTP recipe instead names a
//! mutable `dev` image. Consequently a source-bound real bundle must identify
//! two executions and may not represent the Transformers base class as an MTP
//! oracle.

#![allow(dead_code)]

use super::{
    BF16_TENSOR_COUNT, CHECKPOINT_ARTIFACT_MANIFEST_SHA256, F32_TENSOR_COUNT,
    HEADER_INVENTORY_SHA256, INDEX_INVENTORY_SHA256, INDEX_REPORTED_TOTAL_SIZE,
    INDEX_TOTAL_SIZE_DELTA, MTP_TENSOR_COUNT, PHYSICAL_ARTIFACT_BYTES, PHYSICAL_TENSOR_BYTES,
    PHYSICAL_TENSOR_COUNT, REPOSITORY, REVISION, SMOKE_PROMPT, SMOKE_TOKEN_IDS,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, Read},
    path::{Component, Path, PathBuf},
};

/// Exact committed component-oracle fixture.
const SYNTHETIC_ORACLE_BYTES: &[u8] =
    include_bytes!("../fixtures/nemotron_h_checkpoint/synthetic_oracles_v1.json");

/// Independently reviewed SHA-256 of [`SYNTHETIC_ORACLE_BYTES`].
const SYNTHETIC_ORACLE_SHA256: &str =
    "3b1549cede5872148a452eb8428952e6ed1f4cf56c2dbbd264b56d4f536d08bc";

/// Official Transformers repository used for target-backbone equations.
const TRANSFORMERS_REPOSITORY: &str = "huggingface/transformers";

/// Exact Transformers source revision used for target-backbone equations.
const TRANSFORMERS_REVISION: &str = "cfa309cfa34b5497a32fd50af49aa97aac94c054";

/// Exact base modeling source identity at [`TRANSFORMERS_REVISION`].
const TRANSFORMERS_MODEL_SOURCE: SourceIdentity = SourceIdentity {
    repository: TRANSFORMERS_REPOSITORY,
    revision: TRANSFORMERS_REVISION,
    source: "src/transformers/models/nemotron_h/modeling_nemotron_h.py",
    bytes: 50_797,
    sha256: "d42067429e3268d1163579f5d257fe0baa9a2b02edba5fcb175fb8d42c9b6af4",
};

/// Exact Transformers configuration source identity at the same revision.
const TRANSFORMERS_CONFIG_SOURCE: SourceIdentity = SourceIdentity {
    repository: TRANSFORMERS_REPOSITORY,
    revision: TRANSFORMERS_REVISION,
    source: "src/transformers/models/nemotron_h/configuration_nemotron_h.py",
    bytes: 13_031,
    sha256: "00a4c60bd302d796b25c6ae25c608a5ee7e27f9a0adea92911b0c45470232052",
};

/// vLLM repository used for the executable Nemotron MTP graph.
const VLLM_REPOSITORY: &str = "vllm-project/vllm";

/// Commit behind the checkpoint-card-specified vLLM 0.27.1 tag.
const VLLM_REVISION: &str = "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac";

/// Exact vLLM MTP implementation identity.
const VLLM_MTP_SOURCE: SourceIdentity = SourceIdentity {
    repository: VLLM_REPOSITORY,
    revision: VLLM_REVISION,
    source: "vllm/model_executor/models/nemotron_h_mtp.py",
    bytes: 18_008,
    sha256: "0a799030f5bda1b564b749aaefe6cc3f479c90ff28bd449070b4bb7a21a10176",
};

/// Exact vLLM base block implementation imported by its MTP graph.
const VLLM_BASE_SOURCE: SourceIdentity = SourceIdentity {
    repository: VLLM_REPOSITORY,
    revision: VLLM_REVISION,
    source: "vllm/model_executor/models/nemotron_h.py",
    bytes: 32_582,
    sha256: "4a0d76f7b7b8c9902adb65b0b2700e2924786cf2ba337f3ff0506fe75ff1ae67",
};

/// Exact vLLM Nemotron configuration adapter imported by the MTP graph.
const VLLM_CONFIG_SOURCE: SourceIdentity = SourceIdentity {
    repository: VLLM_REPOSITORY,
    revision: VLLM_REVISION,
    source: "vllm/transformers_utils/configs/nemotron_h.py",
    bytes: 13_414,
    sha256: "7cbf03b9d0c1b8734074101d9178cfd5785b1a9191148396f7ca971f7cc709d7",
};

/// Environment variable selecting a reviewed create-only reference bundle.
const REFERENCE_DIR_ENV: &str = "TRANSFORMERS_NEMOTRON_H_REFERENCE_DIR";

/// Fixed directory name for a reference bundle.
const BUNDLE_DIRNAME: &str = "nemotron-3.5-lightning-30b-a3b-a9904d24-reference.bundle";

/// Strict JSON record inside the reference bundle.
const JSON_FILENAME: &str = "nemotron-3.5-lightning-30b-a3b-a9904d24-reference.json";

/// Independently reviewed real-reference JSON hash; intentionally unset.
const TRUSTED_REFERENCE_JSON_SHA256: Option<&str> = None;

/// Independently reviewed real-reference payload hashes; intentionally unset.
const TRUSTED_REFERENCE_PAYLOAD_SHA256: [(&str, Option<&str>); 6] = [
    ("base-prompt-logits.f32le", None),
    ("base-hidden-last-trace.f32le", None),
    ("base-cached-logits.f32le", None),
    ("base-teacher-logits.f32le", None),
    ("mtp-draft-logits.f32le", None),
    ("mtp-hidden-last-trace.f32le", None),
];

/// Exact immutable source-file identity.
#[derive(Debug, Clone, Copy)]
struct SourceIdentity {
    /// Source repository.
    repository: &'static str,
    /// Immutable Git commit.
    revision: &'static str,
    /// Repository-relative source path.
    source: &'static str,
    /// Exact byte length.
    bytes: u64,
    /// Exact SHA-256.
    sha256: &'static str,
}

/// Public summary returned only after every real-bundle gate succeeds.
pub(super) struct ValidatedReference {
    /// Number of authenticated binary outputs.
    pub(super) output_count: usize,
    /// Number of prompt tokens.
    pub(super) prompt_tokens: usize,
    /// Number of named adversarial component cases bound into the run.
    pub(super) synthetic_cases: usize,
}

/// Strict source identity serialized by both synthetic and real records.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceRecord {
    /// Repository owner/name.
    repository: String,
    /// Forty-character Git revision.
    revision: String,
    /// Repository-relative source path.
    source: String,
    /// Source byte length.
    source_bytes: u64,
    /// Source SHA-256.
    source_sha256: String,
}

/// Source pair bound into the committed component fixture.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SyntheticReferences {
    /// Base-model equation source.
    base: SourceRecord,
    /// Executable MTP equation source.
    mtp: SourceRecord,
}

/// Python process that produced adversarial scalar cases.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SyntheticProducer {
    /// Interpreter implementation/version.
    python: String,
    /// PyTorch version.
    torch: String,
    /// Execution device.
    device: String,
    /// Fixed random seed.
    seed: u64,
}

/// Adversarial ordinary RMSNorm precision boundary.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RmsNormOracle {
    /// Human-readable operation order.
    equation: String,
    /// Variance epsilon.
    epsilon: f64,
    /// Exact F16 input bit patterns.
    input_f16_bits: Vec<String>,
    /// Exact F16 scale bit patterns.
    weight_f16_bits: Vec<String>,
    /// Official output bit patterns.
    official_output_f16_bits: Vec<String>,
    /// Deliberately wrong division/late-rounding result.
    division_late_round_output_f16_bits: Vec<String>,
}

/// Adversarial Mamba gated group-RMSNorm precision boundary.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupedNormOracle {
    /// Human-readable operation order.
    equation: String,
    /// Group count.
    groups: usize,
    /// Variance epsilon.
    epsilon: f64,
    /// Exact F16 activation bits.
    input_f16_bits: Vec<String>,
    /// Exact F16 gate bits.
    gate_f16_bits: Vec<String>,
    /// Exact F16 scale bits.
    weight_f16_bits: Vec<String>,
    /// Official output bits.
    official_output_f16_bits: Vec<String>,
    /// Deliberately wrong division result.
    division_output_f16_bits: Vec<String>,
}

/// Adversarial correction-biased grouped router case.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouterOracle {
    /// Human-readable selection/weight equation.
    equation: String,
    /// Router group count.
    groups: usize,
    /// Eligible group count.
    top_groups: usize,
    /// Selected expert count.
    top_k: usize,
    /// Post-normalization route multiplier.
    routed_scaling_factor: f32,
    /// Little-endian IEEE F32 logits.
    logits_f32le_hex: Vec<String>,
    /// Little-endian IEEE F32 correction bias.
    correction_f32le_hex: Vec<String>,
    /// Official selected group IDs.
    selected_group_ids: Vec<usize>,
    /// Official selected expert IDs.
    selected_expert_ids: Vec<usize>,
    /// Official unbiased, normalized, scaled route weights.
    normalized_scaled_weights_f32le_hex: Vec<String>,
}

/// Exact single-stage MTP topology facts.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MtpContractOracle {
    /// Concatenation operands in mathematical order.
    embedding_hidden_concatenation_order: Vec<String>,
    /// Residual-block schedule.
    schedule: Vec<String>,
    /// Physical MTP source count.
    checkpoint_tensor_count: usize,
    /// Shard containing all MTP weights.
    checkpoint_shard: String,
    /// Whether token embedding storage comes from the target.
    shares_target_embedding: bool,
    /// Whether vocabulary projection storage comes from the target.
    shares_target_lm_head: bool,
    /// Whether the first residual begins after concatenation projection.
    attention_residual_starts_after_fusion: bool,
    /// Whether final normalization follows the MoE residual sum.
    final_norm_after_moe_residual: bool,
}

/// Complete strict committed component-oracle record.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SyntheticOracle {
    /// Fixture schema revision.
    schema_version: u32,
    /// Immutable source references.
    references: SyntheticReferences,
    /// Producing environment.
    producer: SyntheticProducer,
    /// Ordinary norm boundary.
    rms_norm_f16: RmsNormOracle,
    /// Mamba grouped norm boundary.
    grouped_gated_rms_norm_f16: GroupedNormOracle,
    /// Router boundary.
    correction_biased_router_f32: RouterOracle,
    /// MTP topology boundary.
    mtp_contract: MtpContractOracle,
}

/// Strict implementation record for one half of a real combined oracle.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RealImplementation {
    /// Source repository.
    repository: String,
    /// Immutable source revision.
    revision: String,
    /// Executed class.
    class: String,
    /// Every directly pinned implementation file.
    sources: Vec<SourceRecord>,
    /// Whether this implementation consumes `mtp.*` weights.
    executes_mtp: bool,
}

/// Target-backbone and MTP implementations used by a real run.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RealReferences {
    /// Official Hugging Face target graph.
    target: RealImplementation,
    /// Model-card-recommended serving MTP graph.
    mtp: RealImplementation,
}

/// Immutable checkpoint facts repeated by a real run.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RealInventory {
    /// Canonical artifact manifest hash.
    artifact_manifest_sha256: String,
    /// Canonical index name/shard hash.
    index_inventory_sha256: String,
    /// Canonical header inventory hash.
    header_inventory_sha256: String,
    /// Physical source tensor count.
    physical_tensors: usize,
    /// BF16 source count.
    bf16_tensors: usize,
    /// F32 source count.
    f32_tensors: usize,
    /// Physical tensor payload bytes.
    physical_tensor_bytes: u64,
    /// Complete shard bytes.
    artifact_bytes: u64,
    /// Incorrect producer-reported size.
    index_reported_total_size: u64,
    /// Proven producer overstatement.
    index_total_size_delta: u64,
    /// Physical MTP tensor count.
    mtp_tensors: usize,
}

/// Exact prompt fed to both target and MTP reference workflows.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RealPrompt {
    /// Unformatted prompt text.
    text: String,
    /// Token IDs without injected specials.
    input_ids: Vec<u32>,
}

/// Deterministic execution boundary for one reference runtime.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RealExecutionPart {
    /// Device/backend name.
    device: String,
    /// Parameter/activation dtype.
    dtype: String,
    /// Attention implementation.
    attention_implementation: String,
    /// Whether caching was exercised.
    use_cache: bool,
    /// Fixed seed.
    seed: u64,
}

/// Two-runtime execution policy.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RealExecution {
    /// Target-backbone execution.
    target: RealExecutionPart,
    /// MTP execution.
    mtp: RealExecutionPart,
    /// Raw MTP keys independently counted before the target ignore filter.
    target_checkpoint_mtp_keys: usize,
    /// MTP keys still reported unexpected after Transformers' ignore filter.
    target_reported_unexpected_mtp_keys: usize,
}

/// One external little-endian F32 tensor descriptor.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BlobDescriptor {
    /// Bundle-relative filename.
    relative_path: String,
    /// Required `f32le` encoding.
    encoding: String,
    /// Logical tensor shape.
    shape: Vec<usize>,
    /// Product of all dimensions.
    elements: usize,
    /// Exact payload bytes.
    bytes: u64,
    /// Payload SHA-256 repeated inside the record.
    sha256: String,
    /// Whether every recorded scalar was finite.
    finite: bool,
}

/// Strict real reference record.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RealReference {
    /// Bundle schema revision.
    schema_version: u32,
    /// Immutable checkpoint repository.
    repository: String,
    /// Immutable checkpoint revision.
    revision: String,
    /// Dual reference implementations.
    references: RealReferences,
    /// Checkpoint inventory facts.
    inventory: RealInventory,
    /// Fixed prompt.
    prompt: RealPrompt,
    /// Dual execution policy.
    execution: RealExecution,
    /// Exact committed adversarial fixture hash.
    synthetic_oracle_sha256: String,
    /// Required numerical payload descriptors.
    outputs: BTreeMap<String, BlobDescriptor>,
}

/// Returns all independently pinned real-reference identities or fails closed.
pub(super) fn trusted_reference_hashes() -> Result<BTreeMap<&'static str, &'static str>, String> {
    let json = TRUSTED_REFERENCE_JSON_SHA256.ok_or_else(|| {
        "Nemotron real-reference JSON SHA-256 has not been independently reviewed".to_owned()
    })?;
    let mut hashes = BTreeMap::from([(JSON_FILENAME, json)]);
    for (name, digest) in TRUSTED_REFERENCE_PAYLOAD_SHA256 {
        hashes.insert(
            name,
            digest
                .ok_or_else(|| format!("Nemotron real-reference payload {name} is unreviewed"))?,
        );
    }
    Ok(hashes)
}

/// Validates the committed adversarial fixture and its distinguishing values.
pub(super) fn validate_committed_synthetic_oracle() {
    assert_eq!(sha256_hex(SYNTHETIC_ORACLE_BYTES), SYNTHETIC_ORACLE_SHA256);
    let oracle: SyntheticOracle = serde_json::from_slice(SYNTHETIC_ORACLE_BYTES)
        .expect("committed Nemotron component oracle must satisfy its strict schema");
    assert_eq!(oracle.schema_version, 1);
    assert_source(&oracle.references.base, TRANSFORMERS_MODEL_SOURCE);
    assert_source(&oracle.references.mtp, VLLM_MTP_SOURCE);
    assert_eq!(oracle.producer.python, "CPython 3.11.9");
    assert_eq!(oracle.producer.torch, "2.10.0");
    assert_eq!(oracle.producer.device, "cpu");
    assert_eq!(oracle.producer.seed, 0);

    assert!(oracle.rms_norm_f16.equation.contains("rsqrt"));
    assert_eq!(oracle.rms_norm_f16.epsilon, 1e-5);
    assert_eq!(oracle.rms_norm_f16.input_f16_bits, ["4775", "b73c"]);
    assert_eq!(oracle.rms_norm_f16.weight_f16_bits, ["349f", "c2e0"]);
    assert_eq!(
        oracle.rms_norm_f16.official_output_f16_bits,
        ["3685", "34b5"]
    );
    assert_ne!(
        oracle.rms_norm_f16.official_output_f16_bits,
        oracle.rms_norm_f16.division_late_round_output_f16_bits
    );

    assert!(oracle.grouped_gated_rms_norm_f16.equation.contains("silu"));
    assert_eq!(oracle.grouped_gated_rms_norm_f16.groups, 2);
    assert_eq!(oracle.grouped_gated_rms_norm_f16.epsilon, 1e-6);
    assert_eq!(oracle.grouped_gated_rms_norm_f16.input_f16_bits.len(), 4);
    assert_eq!(oracle.grouped_gated_rms_norm_f16.gate_f16_bits.len(), 4);
    assert_eq!(oracle.grouped_gated_rms_norm_f16.weight_f16_bits.len(), 4);
    assert_eq!(
        oracle.grouped_gated_rms_norm_f16.official_output_f16_bits,
        ["a34c", "c47c", "16af", "c464"]
    );
    assert_ne!(
        oracle.grouped_gated_rms_norm_f16.official_output_f16_bits,
        oracle.grouped_gated_rms_norm_f16.division_output_f16_bits
    );

    let router = oracle.correction_biased_router_f32;
    assert!(router.equation.contains("unbiased"));
    assert_eq!((router.groups, router.top_groups, router.top_k), (4, 2, 3));
    assert_eq!(router.selected_group_ids, [0, 2]);
    assert_eq!(router.selected_expert_ids, [0, 4, 5]);
    let logits = router
        .logits_f32le_hex
        .iter()
        .map(|hex| decode_f32le(hex))
        .collect::<Vec<_>>();
    let correction = router
        .correction_f32le_hex
        .iter()
        .map(|hex| decode_f32le(hex))
        .collect::<Vec<_>>();
    let weights = router
        .normalized_scaled_weights_f32le_hex
        .iter()
        .map(|hex| decode_f32le(hex))
        .collect::<Vec<_>>();
    assert_eq!(logits, [-2.0, -1.0, 0.0, 1.0, 2.0, 1.5, 0.5, -0.5]);
    assert_eq!(correction, [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert!((weights.iter().sum::<f32>() - router.routed_scaling_factor).abs() < 1e-6);
    assert!(
        logits[0] < logits[3],
        "correction bias must change selection, not merely preserve rank"
    );

    let mtp = oracle.mtp_contract;
    assert_eq!(
        mtp.embedding_hidden_concatenation_order,
        [
            "normalized_shifted_embedding",
            "normalized_target_hidden_state"
        ]
    );
    assert_eq!(mtp.schedule, ["full_attention", "moe"]);
    assert_eq!(mtp.checkpoint_tensor_count, MTP_TENSOR_COUNT);
    assert_eq!(mtp.checkpoint_shard, "model-00014-of-00014.safetensors");
    assert!(mtp.shares_target_embedding && mtp.shares_target_lm_head);
    assert!(mtp.attention_residual_starts_after_fusion && mtp.final_norm_after_moe_residual);
}

/// Loads and validates a reviewed real reference without trusting self-reported hashes.
pub(super) fn load_and_validate_reference_bundle() -> ValidatedReference {
    let trusted = trusted_reference_hashes().unwrap_or_else(|error| panic!("{error}"));
    let supplied = PathBuf::from(std::env::var_os(REFERENCE_DIR_ENV).unwrap_or_else(|| {
        panic!("set {REFERENCE_DIR_ENV} to the reviewed {BUNDLE_DIRNAME} directory")
    }));
    let root = supplied
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied.display()));
    assert_eq!(
        root.file_name().and_then(|name| name.to_str()),
        Some(BUNDLE_DIRNAME)
    );
    assert!(root.is_dir());
    let expected_files = trusted.keys().copied().collect::<BTreeSet<_>>();
    let observed_files = fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("list {}: {error}", root.display()))
        .map(|entry| {
            entry
                .expect("read bundle entry")
                .file_name()
                .into_string()
                .expect("bundle filenames must be UTF-8")
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        observed_files,
        expected_files.into_iter().map(str::to_owned).collect()
    );

    let json_path = root.join(JSON_FILENAME);
    assert_regular_file(&json_path);
    assert_eq!(sha256_file_hex(&json_path), trusted[JSON_FILENAME]);
    let record_bytes = fs::read(&json_path).expect("read authenticated reference JSON");
    let record: RealReference = serde_json::from_slice(&record_bytes)
        .expect("Nemotron real-reference JSON must satisfy its strict schema");
    validate_real_record(&record);

    let expected_outputs = expected_output_shapes();
    assert_eq!(
        record.outputs.keys().cloned().collect::<BTreeSet<_>>(),
        expected_outputs.keys().cloned().collect()
    );
    for (logical_name, descriptor) in &record.outputs {
        let (expected_filename, expected_shape) = &expected_outputs[logical_name];
        assert_eq!(&descriptor.shape, expected_shape, "{logical_name}");
        assert_eq!(descriptor.encoding, "f32le", "{logical_name}");
        assert!(descriptor.finite, "{logical_name}");
        let elements = descriptor
            .shape
            .iter()
            .try_fold(1_usize, |product, dimension| {
                product.checked_mul(*dimension)
            })
            .expect("reference shape product overflow");
        assert_eq!(descriptor.elements, elements, "{logical_name}");
        assert_eq!(descriptor.bytes, (elements as u64) * 4, "{logical_name}");
        let filename = safe_bundle_filename(&descriptor.relative_path);
        assert_eq!(&filename, expected_filename, "{logical_name}");
        let expected_hash = trusted
            .get(filename.as_str())
            .unwrap_or_else(|| panic!("untrusted payload name {filename}"));
        assert_eq!(&descriptor.sha256, expected_hash, "{logical_name}");
        let path = root.join(&filename);
        assert_regular_file(&path);
        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            descriptor.bytes,
            "{logical_name}"
        );
        assert_eq!(sha256_file_hex(&path), *expected_hash, "{logical_name}");
        assert_f32le_payload_is_finite(&path, elements);
    }
    ValidatedReference {
        output_count: record.outputs.len(),
        prompt_tokens: record.prompt.input_ids.len(),
        synthetic_cases: 4,
    }
}

/// Validates all fixed metadata in a real bundle record.
fn validate_real_record(record: &RealReference) {
    assert_eq!(record.schema_version, 1);
    assert_eq!(record.repository, REPOSITORY);
    assert_eq!(record.revision, REVISION);
    assert_eq!(record.prompt.text, SMOKE_PROMPT);
    assert_eq!(record.prompt.input_ids, SMOKE_TOKEN_IDS);
    assert_eq!(record.synthetic_oracle_sha256, SYNTHETIC_ORACLE_SHA256);

    let target = &record.references.target;
    assert_eq!(target.repository, TRANSFORMERS_REPOSITORY);
    assert_eq!(target.revision, TRANSFORMERS_REVISION);
    assert_eq!(target.class, "transformers.NemotronHForCausalLM");
    assert!(
        !target.executes_mtp,
        "Transformers cfa309 explicitly ignores mtp.*"
    );
    assert_source_set(
        &target.sources,
        &[TRANSFORMERS_MODEL_SOURCE, TRANSFORMERS_CONFIG_SOURCE],
    );

    let mtp = &record.references.mtp;
    assert_eq!(mtp.repository, VLLM_REPOSITORY);
    assert_eq!(mtp.revision, VLLM_REVISION);
    assert_eq!(
        mtp.class,
        "vllm.model_executor.models.nemotron_h_mtp.NemotronHMTP"
    );
    assert!(mtp.executes_mtp);
    assert_source_set(
        &mtp.sources,
        &[VLLM_MTP_SOURCE, VLLM_BASE_SOURCE, VLLM_CONFIG_SOURCE],
    );

    let inventory = &record.inventory;
    assert_eq!(
        inventory.artifact_manifest_sha256,
        CHECKPOINT_ARTIFACT_MANIFEST_SHA256
    );
    assert_eq!(inventory.index_inventory_sha256, INDEX_INVENTORY_SHA256);
    assert_eq!(inventory.header_inventory_sha256, HEADER_INVENTORY_SHA256);
    assert_eq!(inventory.physical_tensors, PHYSICAL_TENSOR_COUNT);
    assert_eq!(inventory.bf16_tensors, BF16_TENSOR_COUNT);
    assert_eq!(inventory.f32_tensors, F32_TENSOR_COUNT);
    assert_eq!(inventory.physical_tensor_bytes, PHYSICAL_TENSOR_BYTES);
    assert_eq!(inventory.artifact_bytes, PHYSICAL_ARTIFACT_BYTES);
    assert_eq!(
        inventory.index_reported_total_size,
        INDEX_REPORTED_TOTAL_SIZE
    );
    assert_eq!(inventory.index_total_size_delta, INDEX_TOTAL_SIZE_DELTA);
    assert_eq!(inventory.mtp_tensors, MTP_TENSOR_COUNT);

    assert_eq!(record.execution.target.dtype, "float16");
    assert_eq!(record.execution.target.device, "mps");
    assert_eq!(record.execution.target.attention_implementation, "eager");
    assert!(record.execution.target.use_cache);
    assert_eq!(record.execution.target.seed, 0);
    assert_eq!(record.execution.mtp.dtype, "float16");
    assert_eq!(record.execution.mtp.device, "cuda");
    assert_eq!(
        record.execution.mtp.attention_implementation,
        "vllm-attention"
    );
    assert!(record.execution.mtp.use_cache);
    assert_eq!(record.execution.mtp.seed, 0);
    assert_eq!(
        record.execution.target_checkpoint_mtp_keys,
        MTP_TENSOR_COUNT
    );
    assert_eq!(record.execution.target_reported_unexpected_mtp_keys, 0);
}

/// Returns the exact filename and shape assigned to every external output.
fn expected_output_shapes() -> BTreeMap<String, (String, Vec<usize>)> {
    BTreeMap::from([
        (
            "base_prompt_logits".into(),
            ("base-prompt-logits.f32le".into(), vec![1, 1, 131_072]),
        ),
        (
            "base_hidden_last_trace".into(),
            ("base-hidden-last-trace.f32le".into(), vec![54, 2_688]),
        ),
        (
            "base_cached_logits".into(),
            ("base-cached-logits.f32le".into(), vec![32, 131_072]),
        ),
        (
            "base_teacher_logits".into(),
            ("base-teacher-logits.f32le".into(), vec![32, 131_072]),
        ),
        (
            "mtp_draft_logits".into(),
            ("mtp-draft-logits.f32le".into(), vec![1, 1, 131_072]),
        ),
        (
            "mtp_hidden_last_trace".into(),
            ("mtp-hidden-last-trace.f32le".into(), vec![3, 2_688]),
        ),
    ])
}

/// Requires an exact source set with no omitted or additional files.
fn assert_source_set(observed: &[SourceRecord], expected: &[SourceIdentity]) {
    assert_eq!(observed.len(), expected.len());
    for identity in expected {
        let record = observed
            .iter()
            .find(|record| record.source == identity.source)
            .unwrap_or_else(|| panic!("missing source {}", identity.source));
        assert_source(record, *identity);
    }
}

/// Requires one serialized source identity to match an independently pinned one.
fn assert_source(record: &SourceRecord, expected: SourceIdentity) {
    assert_eq!(record.repository, expected.repository);
    assert_eq!(record.revision, expected.revision);
    assert_eq!(record.source, expected.source);
    assert_eq!(record.source_bytes, expected.bytes);
    assert_eq!(record.source_sha256, expected.sha256);
}

/// Converts one eight-hex-character little-endian IEEE F32 fixture.
fn decode_f32le(hex: &str) -> f32 {
    assert_eq!(hex.len(), 8);
    let mut bytes = [0_u8; 4];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16)
            .expect("F32 bit fixture must contain hexadecimal bytes");
    }
    f32::from_le_bytes(bytes)
}

/// Accepts only one plain filename, never traversal or nested output paths.
fn safe_bundle_filename(relative: &str) -> String {
    let path = Path::new(relative);
    assert!(
        !path.is_absolute(),
        "reference payload path must be relative"
    );
    let components = path.components().collect::<Vec<_>>();
    assert_eq!(
        components.len(),
        1,
        "reference payload path must be one filename"
    );
    assert!(matches!(components[0], Component::Normal(_)));
    relative.to_owned()
}

/// Rejects symlinks and non-regular bundle members.
fn assert_regular_file(path: &Path) {
    let metadata = fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()));
    assert!(
        !metadata.file_type().is_symlink(),
        "{} is a symlink",
        path.display()
    );
    assert!(
        metadata.is_file(),
        "{} is not a regular file",
        path.display()
    );
}

/// Parses every authenticated scalar and rejects NaN, infinity, or trailing bytes.
fn assert_f32le_payload_is_finite(path: &Path, elements: usize) {
    let file = File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let mut reader = BufReader::new(file);
    let mut bytes = [0_u8; 4];
    for index in 0..elements {
        reader
            .read_exact(&mut bytes)
            .unwrap_or_else(|error| panic!("read {} scalar {index}: {error}", path.display()));
        assert!(
            f32::from_le_bytes(bytes).is_finite(),
            "{} scalar {index} is not finite",
            path.display()
        );
    }
    assert_eq!(
        reader
            .read(&mut bytes)
            .expect("check F32 payload trailing bytes"),
        0,
        "{} contains trailing bytes",
        path.display()
    );
}

/// Hashes one regular file without allocating its complete contents.
fn sha256_file_hex(path: &Path) -> String {
    let mut file =
        File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
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

/// Hashes in-memory bytes as lower-case SHA-256.
fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    //! Tests for closed oracle schemas, bundle containment, and reference trust anchors.

    use super::*;
    use std::io::Write;

    /// Strict schemas reject additional self-authorized fields.
    #[test]
    fn synthetic_schema_rejects_unknown_fields() {
        let mut value: serde_json::Value = serde_json::from_slice(SYNTHETIC_ORACLE_BYTES).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("trust_me".into(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<SyntheticOracle>(value).is_err());
    }

    /// Path validation rejects traversal, absolute, and nested payload names.
    #[test]
    fn bundle_filename_policy_is_fail_closed() {
        for invalid in ["../oracle", "/tmp/oracle", "nested/oracle", "."] {
            assert!(std::panic::catch_unwind(|| safe_bundle_filename(invalid)).is_err());
        }
        assert_eq!(safe_bundle_filename("oracle.f32le"), "oracle.f32le");
    }

    /// Every real bundle anchor starts unreviewed rather than trusting its JSON.
    #[test]
    fn unreviewed_real_anchors_are_all_absent() {
        assert!(TRUSTED_REFERENCE_JSON_SHA256.is_none());
        assert!(
            TRUSTED_REFERENCE_PAYLOAD_SHA256
                .iter()
                .all(|(_, hash)| hash.is_none())
        );
        assert!(trusted_reference_hashes().is_err());
    }

    /// Little-endian scalar fixtures retain their exact IEEE interpretation.
    #[test]
    fn f32_hex_decoder_is_little_endian() {
        assert_eq!(decode_f32le("000000c0"), -2.0);
        assert_eq!(decode_f32le("00008040"), 4.0);
    }

    /// Binary output validation reads scalars rather than trusting JSON's finite flag.
    #[test]
    fn f32_payload_validation_rejects_non_finite_and_trailing_data() {
        let mut finite = tempfile::NamedTempFile::new().unwrap();
        finite.write_all(&1.0_f32.to_le_bytes()).unwrap();
        finite.write_all(&(-2.0_f32).to_le_bytes()).unwrap();
        assert_f32le_payload_is_finite(finite.path(), 2);

        let mut non_finite = tempfile::NamedTempFile::new().unwrap();
        non_finite.write_all(&f32::NAN.to_le_bytes()).unwrap();
        assert!(
            std::panic::catch_unwind(|| assert_f32le_payload_is_finite(non_finite.path(), 1))
                .is_err()
        );

        assert!(
            std::panic::catch_unwind(|| assert_f32le_payload_is_finite(finite.path(), 1)).is_err()
        );
    }
}
