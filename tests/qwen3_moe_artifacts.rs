//! Opt-in acceptance gates for the immutable Qwen3-30B-A3B checkpoint.
//!
//! These tests never contact a model hub. Point `TRANSFORMERS_QWEN3_MOE_DIR`
//! at a complete local copy of `Qwen/Qwen3-30B-A3B` revision
//! `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, then run the metadata gate with:
//!
//! ```text
//! cargo test --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,tokenizers \
//!   --test qwen3_moe_artifacts \
//!   validates_pinned_qwen3_moe_config_headers_plan_memory_and_tokenizer \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! Compile the Metal gate without executing the 30B load with:
//!
//! ```text
//! cargo test --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,metal,tokenizers,chat-templates \
//!   --test qwen3_moe_artifacts --no-run
//! ```
//!
//! Generate and authenticate the official full-vocabulary reference entirely
//! from the pinned local checkpoint with:
//!
//! ```text
//! python3 crates/transformers/scripts/generate_qwen3_moe_reference.py \
//!   --model-dir /path/to/Qwen3-30B-A3B-ad44e777 \
//!   --output-dir /external/reference-parent
//!
//! TRANSFORMERS_QWEN3_MOE_REFERENCE_DIR=/external/reference-parent/qwen3-30b-a3b-ad44e777-reference.bundle \
//! cargo test --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,tokenizers \
//!   --test qwen3_moe_artifacts \
//!   validates_pinned_qwen3_moe_official_reference_bundle \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! Add `metal` and select
//! `compares_pinned_qwen3_moe_f16_with_official_reference` while also setting
//! `TRANSFORMERS_QWEN3_MOE_DIR` to execute the expensive numerical gate.
//!
//! Q8 and Q4 conversion use explicit, create-only output paths on the same
//! external `/Volumes/<name>` tree as the checkpoint. For example:
//!
//! ```text
//! TRANSFORMERS_QWEN3_MOE_Q8_BURNPACK=/Volumes/<name>/quantized/qwen3-30b-a3b-q8-g64.bpk \
//! cargo test --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,metal,tokenizers \
//!   --test qwen3_moe_artifacts \
//!   quantizes_pinned_qwen3_moe_weights_to_q8_burnpack \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! Conversion and inference are separate gates. This is intentional: a
//! successful create-only conversion remains reusable when a later Metal smoke
//! uncovers a kernel or model issue.
//!
//! Add `kernel-tracing` to the exact single-backend feature set and select
//! `audits_pinned_qwen3_moe_q{4,8}_decode_packed_metal_kernels` to prove one
//! recurrent decode reaches all 1,345 packed projection kernels without a
//! whole-weight dequantization launch. This audit reuses the corresponding
//! `TRANSFORMERS_QWEN3_MOE_Q{4,8}_BURNPACK` path.

#![cfg(feature = "tokenizers")]

#[cfg(all(feature = "kernel-tracing", feature = "metal", not(feature = "rocm")))]
#[path = "support/kernel_trace.rs"]
mod kernel_trace;

#[path = "support/qwen3_moe_reference.rs"]
mod qwen3_moe_reference;

use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

use transformers::{
    EncodeOptions, LocalTokenizer,
    checkpoint::{
        ArtifactSource, CheckpointManifest, DTypePolicy, LoadOptions, LoadPlan, PlannedTransform,
        TensorDType,
        quantization::{
            QuantizationEligibility, QuantizationExclusion, SymmetricQuantization,
            WeightQuantizationConfig, classify_weight_for_quantization,
        },
    },
    models::{
        config::ModelConfig,
        qwen3_moe::{Qwen3MoeCheckpointOptions, Qwen3MoeConfig},
    },
};

#[cfg(feature = "metal")]
use burn::{
    backend::Metal,
    nn::Linear,
    tensor::{DType, Int, Tensor, TensorData, backend::Backend},
};
#[cfg(all(feature = "kernel-tracing", feature = "metal", not(feature = "rocm")))]
use kernel_trace::{
    calibrated_dequantization_kernel_names, capture_kernel_launches, compact_kernel_id,
    packed_matmul_launches, whole_weight_dequantization_launches,
};
#[cfg(feature = "metal")]
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, DecodeInput, DecodeState, ForwardOptions,
    LogitsSelection, cache::CacheConfig, models::qwen3_moe::Qwen3MoeForCausalLm,
};
#[cfg(all(feature = "metal", not(feature = "rocm")))]
use transformers::{
    CausalLanguageModel,
    checkpoint::ExecutionStats,
    checkpoint::quantization::{
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2, PROVENANCE_HASH_BUFFER_BYTES,
        ProvenanceExecutionStats, QuantizedBurnpackArtifact, QuantizedCheckpointOptions,
        SourceFloatDType, execute_local_quantized_load_plan, load_plan_sha256,
        quantize_local_checkpoint_to_burnpack, validate_sidecar_contract,
    },
};

/// Immutable upstream repository certified by both gates.
const REPOSITORY: &str = "Qwen/Qwen3-30B-A3B";

/// Immutable upstream Git revision certified by both gates.
const REVISION: &str = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39";

/// Exact output path used by the create-only Q8 conversion and later Metal load.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
const Q8_BURNPACK_ENV: &str = "TRANSFORMERS_QWEN3_MOE_Q8_BURNPACK";

/// Exact output path used by the create-only Q4 conversion and later Metal load.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
const Q4_BURNPACK_ENV: &str = "TRANSFORMERS_QWEN3_MOE_Q4_BURNPACK";

/// Number of architecture parameters accepted by the strict checkpoint plan.
const PARAMETER_COUNT: usize = 18_867;

/// Parameters eligible for group-64 symmetric weight-only quantization.
const QUANTIZED_PARAMETER_COUNT: usize = 18_625;

/// Embedding, normalization, and router parameters retained in F16.
const RETAINED_F16_PARAMETER_COUNT: usize = 242;

/// Exact F16 bytes delivered for the 242 parameters absent from either sidecar.
const RETAINED_SOURCE_TENSOR_BYTES: u64 = 647_917_568;

/// Exact group-64 Q8 packed-value and scale bytes for all eligible matrices.
const Q8_PAYLOAD_BYTES: u64 = 32_096_174_080;

/// Exact group-64 Q4 packed-value and scale bytes for all eligible matrices.
const Q4_PAYLOAD_BYTES: u64 = 16_992_092_160;

/// Largest Q8 payload: the independent `[151936, 2048]` LM-head matrix.
const Q8_LARGEST_PAYLOAD_BYTES: u64 = 330_612_736;

/// Largest Q4 payload: the independent `[151936, 2048]` LM-head matrix.
const Q4_LARGEST_PAYLOAD_BYTES: u64 = 175_030_272;

/// Packed projections executed by one batch-one, one-token recurrent decode.
///
/// Each of 48 layers launches four attention projections and three projections
/// for each of its eight selected experts. The independent LM head adds one:
/// `48 * (4 + 8 * 3) + 1 = 1_345`. Router matrices intentionally remain F16.
#[cfg(all(feature = "kernel-tracing", feature = "metal", not(feature = "rocm")))]
const PACKED_DECODE_MATMUL_LAUNCHES: usize = 1_345;

/// Exact BF16 bytes consumed by full-precision loading and sidecar conversion.
const CHECKPOINT_TENSOR_BYTES: u64 = 61_064_245_248;

/// Exact bytes hashed across the 16 complete SafeTensors shard files.
const CHECKPOINT_ARTIFACT_BYTES: u64 = 61_066_575_648;

/// Largest single checkpoint tensor and runtime staging allocation.
const LARGEST_CHECKPOINT_TENSOR_BYTES: u64 = 622_329_856;

/// Short prompt whose pinned tokenizer representation contains exactly six tokens.
const SIX_TOKEN_PROMPT: &str = "Hello world, cached decoding.";

/// Exact token IDs produced by the pinned local tokenizer without special tokens.
const SIX_TOKEN_IDS: [u32; 6] = [9_707, 1_879, 11, 20_579, 47_116, 13];

/// SHA-256 identities for every control file consumed by discovery or tokenization.
const CONTROL_FILE_SHA256: [(&str, &str); 7] = [
    (
        "config.json",
        "2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297",
    ),
    (
        "generation_config.json",
        "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    ),
    (
        "model.safetensors.index.json",
        "df0d481ec595c55a0ba58426d517390c6214a566ec4ff1c8fc4bbce9f57b3c24",
    ),
    (
        "tokenizer.json",
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    (
        "tokenizer_config.json",
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    ),
    (
        "merges.txt",
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    (
        "vocab.json",
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
];

/// SHA-256 identities for all weight shards at the immutable revision.
const SHARD_FILE_SHA256: [(&str, &str); 16] = [
    (
        "model-00001-of-00016.safetensors",
        "454e77b346a61bfb201d54df60e15158838cf930617ee135113556204f2802b5",
    ),
    (
        "model-00002-of-00016.safetensors",
        "47f015d6e5bb1782a834d75c06113c5e8c77ddca1cb7daf89686a4ec0deda19e",
    ),
    (
        "model-00003-of-00016.safetensors",
        "ac0bf5990f2da995c1e8b77a3149dee71900b4fe1b8a614231dea9647193d96b",
    ),
    (
        "model-00004-of-00016.safetensors",
        "89b01fd34a683c70fdb7f22c2e7090538d65063335ba0db46f3654642b17d1e5",
    ),
    (
        "model-00005-of-00016.safetensors",
        "9849eb3584d928e35bf5956ccfd91b64207467e1fad8304aae68dc4d92d6d6d9",
    ),
    (
        "model-00006-of-00016.safetensors",
        "f7a0f1525557d740158fb692986ad43506e1b5901e7406ae65808c6457bc53ad",
    ),
    (
        "model-00007-of-00016.safetensors",
        "920702a50f27a009e5f676da3b02978dbd794f6ad7854f301d709136154c1a84",
    ),
    (
        "model-00008-of-00016.safetensors",
        "a85bf0cc8a8047c116172d4c08cf4792eb59d4375032708ba3e1a7ffca0f708a",
    ),
    (
        "model-00009-of-00016.safetensors",
        "25cd8aaed86b8e17685efb152912f92448ea3fffb1bdc247f257f04d05907604",
    ),
    (
        "model-00010-of-00016.safetensors",
        "7c3307ee214797c4bc61c0994f81d690e768150df5f14d18edf84e53ba4810dc",
    ),
    (
        "model-00011-of-00016.safetensors",
        "c658cad2842d36fa4c7c7f726f8515dc5670a3d80c94e13257d4e322ffe61988",
    ),
    (
        "model-00012-of-00016.safetensors",
        "8cb898bc5e78492600053d1105c9ff61d7a719f5cc2802df63e41535a652cc26",
    ),
    (
        "model-00013-of-00016.safetensors",
        "599594421f314cb8e8ab4474db0eb490cc6310585067e82d73821254b93319ce",
    ),
    (
        "model-00014-of-00016.safetensors",
        "66d3294e9976b5f01d117a0a0bed768f128a8898eb6937e224d051de2961d363",
    ),
    (
        "model-00015-of-00016.safetensors",
        "2fe000da4fc7399ff6303bb0daec8c415a5d850f5ea53863392758c77fc37664",
    ),
    (
        "model-00016-of-00016.safetensors",
        "0c979e314faf06e94a5e1841abd285d4e26e763c79c1abe7710d543192d9da48",
    ),
];

/// SHA-256 of every physical tensor name, shard, dtype, shape, and data range.
const HEADER_INVENTORY_SHA256: &str =
    "c4bc720cd21ec7a0fde1358a12a06f6ee09f6c6c39dcda613abeda5d16a29861";

/// SHA-256 of every F16 plan target, source, dtype, and target shape.
const F16_PLAN_SHA256: &str = "b90fa8fbfa08b6500dc3be8dd6e0458501b62580de704acac3bd300150d5b49d";

/// Resolves the explicitly selected immutable local artifact directory.
fn pinned_root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN3_MOE_DIR").unwrap_or_else(|| {
            panic!("set TRANSFORMERS_QWEN3_MOE_DIR to {REPOSITORY} revision {REVISION}")
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

/// Verifies the immutable identities of the small files read by these gates.
fn assert_control_file_hashes(root: &Path) {
    for (relative_path, expected) in CONTROL_FILE_SHA256 {
        let path = root.join(relative_path);
        assert_eq!(
            sha256_file_hex(&path),
            expected,
            "pinned control file changed: {relative_path}"
        );
    }
}

/// Verifies every multi-gigabyte payload before accepting real-model results.
#[cfg(feature = "metal")]
fn assert_shard_file_hashes(root: &Path) {
    for (relative_path, expected) in SHARD_FILE_SHA256 {
        let path = root.join(relative_path);
        assert_eq!(
            sha256_file_hex(&path),
            expected,
            "pinned weight shard changed: {relative_path}"
        );
    }
}

/// Hashes all physical header facts in canonical tensor-name order.
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

/// Hashes the strict target/source mapping and target dtype in canonical plan order.
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

/// Certifies exact config, headers, mapping, memory bounds, and tokenizer state.
///
/// Discovery reads the index and bounded SafeTensors headers only; no weight
/// payload is materialized. The inventory digest covers every tensor rather than
/// trusting a few representative shapes, while the shape histogram makes the
/// official 48-layer, 128-expert topology legible in failure output.
#[test]
#[ignore = "requires Qwen/Qwen3-30B-A3B revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"]
fn validates_pinned_qwen3_moe_config_headers_plan_memory_and_tokenizer() {
    let started = Instant::now();
    let supplied_root = pinned_root();
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert!(root.is_dir(), "pinned artifact root must be a directory");
    assert_control_file_hashes(&root);

    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let observed = Qwen3MoeConfig::from_json_str(&config_json)
        .expect("the pinned Qwen3-MoE configuration must validate");
    let expected = Qwen3MoeConfig::from_json_str(include_str!("fixtures/qwen3_30b_a3b.json"))
        .expect("the committed Qwen3-MoE configuration fixture must validate");
    assert_eq!(observed, expected);
    let raw: serde_json::Value =
        serde_json::from_str(&config_json).expect("the pinned config must be valid JSON");
    assert_eq!(
        raw["architectures"],
        serde_json::json!(["Qwen3MoeForCausalLM"])
    );
    assert_eq!(raw["torch_dtype"], "bfloat16");
    assert_eq!(raw["transformers_version"], "4.51.0");
    assert_eq!(observed.model_type, "qwen3_moe");
    assert_eq!(observed.num_hidden_layers, 48);
    assert_eq!(observed.num_experts, 128);
    assert_eq!(observed.num_experts_per_tok, 8);
    assert!(!observed.use_sliding_window);
    assert_eq!(observed.max_window_layers, 48);

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
    assert_eq!(manifest.shards.len(), 16);
    assert_eq!(manifest.tensors.len(), PARAMETER_COUNT);
    assert_eq!(manifest.tensor_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(manifest.artifact_bytes, CHECKPOINT_ARTIFACT_BYTES);
    assert_eq!(header_inventory_digest(&manifest), HEADER_INVENTORY_SHA256);
    let expected_shards = (1..=16)
        .map(|index| format!("model-{index:05}-of-00016.safetensors"))
        .collect::<Vec<_>>();
    assert_eq!(
        manifest
            .shards
            .iter()
            .map(|shard| shard.relative_path.clone())
            .collect::<Vec<_>>(),
        expected_shards
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
    let mut shape_counts = BTreeMap::<Vec<u64>, usize>::new();
    for tensor in manifest.tensors.values() {
        *shape_counts.entry(tensor.shape.clone()).or_default() += 1;
    }
    assert_eq!(
        shape_counts,
        BTreeMap::from([
            (vec![128], 96),
            (vec![2_048], 97),
            (vec![128, 2_048], 48),
            (vec![512, 2_048], 96),
            (vec![768, 2_048], 12_288),
            (vec![2_048, 768], 6_144),
            (vec![2_048, 4_096], 48),
            (vec![4_096, 2_048], 48),
            (vec![151_936, 2_048], 2),
        ])
    );

    let contract = observed
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .expect("the pinned topology must produce its exact parameter contract");
    assert_eq!(contract.expected_tensors().len(), PARAMETER_COUNT);
    let plan = contract
        .build_load_plan(&manifest, &options)
        .expect("every pinned tensor must map under strict F16 planning");
    assert_eq!(plan.parameter_count(), PARAMETER_COUNT);
    assert_eq!(plan.tensors.len(), PARAMETER_COUNT);
    assert!(plan.aliases.is_empty());
    assert!(plan.report.is_strictly_valid());
    assert!(plan.report.allowlisted_ignored.is_empty());
    assert!(plan.report.allowlisted_missing.is_empty());
    assert_eq!(plan.report.transformed.len(), 18_480);
    assert!(
        plan.report
            .transformed
            .iter()
            .all(|record| record.operation == "rename")
    );
    assert_eq!(plan.report.dtype_converted.len(), PARAMETER_COUNT);
    assert!(plan.report.dtype_converted.iter().all(|conversion| {
        conversion.source == TensorDType::BF16 && conversion.target_dtype == TensorDType::F16
    }));
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
    let physical_sources = manifest.tensors.keys().map(String::as_str).collect();
    assert_eq!(planned_sources, physical_sources);
    assert_eq!(plan.memory.artifact_bytes, CHECKPOINT_ARTIFACT_BYTES);
    assert_eq!(plan.memory.checkpoint_tensor_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(plan.memory.target_parameter_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(plan.memory.logical_model_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(plan.memory.largest_shard_bytes, 3_999_975_472);
    assert_eq!(
        plan.memory.largest_source_tensor_bytes,
        LARGEST_CHECKPOINT_TENSOR_BYTES
    );
    assert_eq!(
        plan.memory.largest_target_tensor_bytes,
        LARGEST_CHECKPOINT_TENSOR_BYTES
    );
    assert_eq!(plan.memory.peak_host_staging_bytes, 4_622_305_328);
    assert_eq!(plan.report.memory, Some(plan.memory));

    let (expected_quantized, expected_excluded) = qwen3_moe_quantization_partition(&plan);
    assert_qwen3_moe_quantization_partition(&expected_quantized, &expected_excluded);
    assert_eq!(
        retained_source_tensor_bytes(&plan, &expected_excluded),
        RETAINED_SOURCE_TENSOR_BYTES
    );
    assert_eq!(
        expected_quantized_payload(&plan, SymmetricQuantization::Q8),
        pinned_quantized_payload(SymmetricQuantization::Q8)
    );
    assert_eq!(
        expected_quantized_payload(&plan, SymmetricQuantization::Q4),
        pinned_quantized_payload(SymmetricQuantization::Q4)
    );

    let tokenizer = LocalTokenizer::from_directory(&root)
        .expect("the pinned tokenizer artifacts must load without network access");
    assert_eq!(tokenizer.root(), root);
    assert_eq!(
        tokenizer.metadata().tokenizer_class.as_deref(),
        Some("Qwen2Tokenizer")
    );
    assert_eq!(tokenizer.metadata().model_max_length, Some(131_072));
    // The model reserves 267 embedding rows beyond the tokenizer's concrete
    // vocabulary; both counts are pinned independently.
    assert_eq!(tokenizer.metadata().vocabulary_size, 151_669);
    assert_eq!(
        tokenizer
            .metadata()
            .special_tokens
            .eos
            .as_ref()
            .map(|token| token.id),
        Some(151_645)
    );
    assert_eq!(
        tokenizer
            .metadata()
            .special_tokens
            .pad
            .as_ref()
            .map(|token| token.id),
        Some(151_643)
    );
    let encoding = tokenizer
        .encode(
            SIX_TOKEN_PROMPT,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("the pinned six-token fixture must encode");
    assert_eq!(encoding.token_ids()[0], SIX_TOKEN_IDS);
    assert_eq!(encoding.valid_tokens()[0], [true; 6]);
    assert_eq!(
        tokenizer
            .decode(&SIX_TOKEN_IDS, true)
            .expect("the pinned token IDs must decode"),
        SIX_TOKEN_PROMPT
    );

    eprintln!(
        "Qwen3-MoE header gate: repository={REPOSITORY}, revision={REVISION}, \
         shards=16, tensors=18867, tensor_bytes=61064245248, \
         header_inventory_sha256={HEADER_INVENTORY_SHA256}, \
         f16_plan_sha256={F16_PLAN_SHA256}, \
         f16_target_bytes={}, peak_host_staging_bytes={}, elapsed_seconds={:.3}",
        plan.memory.target_parameter_bytes,
        plan.memory.peak_host_staging_bytes,
        started.elapsed().as_secs_f64(),
    );
}

/// Authenticates the official Python bundle without allocating model parameters.
#[test]
#[ignore = "requires the externally generated pinned Qwen3-MoE reference bundle"]
fn validates_pinned_qwen3_moe_official_reference_bundle() {
    let reference = qwen3_moe_reference::load_and_validate_reference_bundle();
    eprintln!(
        "Qwen3-MoE official reference: revision={REVISION}, prompt_logits={}, \
         cached_logits={}, teacher_prefix_logits={}, greedy_tokens={}",
        reference.prompt_logits.len(),
        reference.cached_logits.len(),
        reference.teacher_prefix_logits.len(),
        reference.greedy_token_ids.len(),
    );
}

/// Compares full-vocabulary Burn F16 outputs with the pinned official graph.
///
/// This is deliberately separate from bundle authentication so maintainers can
/// validate a newly generated 40 MB reference before committing to a 61 GB
/// checkpoint load. It checks the stateless prompt, all 32 recurrent decisions,
/// and all 32 corresponding stateless teacher-forced rows at cosine `>= 0.999`.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires the pinned 30B checkpoint, official reference bundle, Metal, and about 70 GiB free unified memory"]
fn compares_pinned_qwen3_moe_f16_with_official_reference() {
    let reference = qwen3_moe_reference::load_and_validate_reference_bundle();
    let root = pinned_root();
    assert_control_file_hashes(&root);
    assert_shard_file_hashes(&root);
    let device = Default::default();
    assert_metal_backend_available(&device);

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
    .expect("strictly load Qwen3-MoE for official-reference comparison");
    <Metal<f32, i32> as Backend>::sync(&device).expect("finish reference-gate model load");
    let tokenizer = loaded
        .tokenizer()
        .expect("official-reference comparison requires the pinned tokenizer");
    let encoding = tokenizer
        .encode(
            SIX_TOKEN_PROMPT,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("encode official-reference prompt");
    assert_eq!(encoding.token_ids()[0], SIX_TOKEN_IDS);
    let prompt = encoding
        .to_causal_input::<Metal>(&device)
        .expect("construct official-reference prompt input");

    let stateless = loaded
        .model()
        .forward(prompt.clone(), &ForwardOptions::default())
        .expect("run stateless official-reference prompt");
    let stateless_values = metal_values("Qwen3-MoE reference stateless logits", &stateless.logits);
    let stateless_cosine = cosine_similarity(&stateless_values, &reference.prompt_logits);
    assert!(
        stateless_cosine >= 0.999,
        "stateless official-reference cosine {stateless_cosine} is below 0.999"
    );
    assert_eq!(
        stateless
            .logits
            .clone()
            .argmax(2)
            .to_data()
            .to_vec::<i32>()
            .expect("stateless top-1 storage"),
        [i32::try_from(reference.greedy_token_ids[0]).expect("token ID fits i32")]
    );

    let mut state = loaded
        .model()
        .new_decode_state(
            CacheConfig::new(1, SIX_TOKEN_IDS.len() + 32).expect("reference cache geometry"),
            1,
            &device,
        )
        .expect("allocate Qwen3-MoE official-reference cache");
    let mut cached = loaded
        .model()
        .prefill(prompt, &mut state, &ForwardOptions::default())
        .expect("run official-reference cached prefill");
    let mut minimum_cached_cosine = f64::INFINITY;
    for step in 0..32 {
        let actual = metal_values("Qwen3-MoE reference cached logits", &cached.logits);
        let expected = &reference.cached_logits[step * 151_936..(step + 1) * 151_936];
        let cosine = cosine_similarity(&actual, expected);
        assert!(
            cosine >= 0.999,
            "cached official-reference cosine at step {step} was {cosine}"
        );
        minimum_cached_cosine = minimum_cached_cosine.min(cosine);
        let actual_top1 = cached
            .logits
            .clone()
            .argmax(2)
            .to_data()
            .to_vec::<i32>()
            .expect("cached top-1 storage");
        assert_eq!(
            actual_top1,
            [i32::try_from(reference.greedy_token_ids[step]).expect("token ID fits i32")],
            "cached top-1 differs at step {step}"
        );
        if step + 1 < 32 {
            cached = loaded
                .model()
                .decode(
                    DecodeInput {
                        token_ids: Tensor::<Metal, 2, Int>::from_data(
                            TensorData::new(
                                vec![i64::from(reference.greedy_token_ids[step])],
                                [1, 1],
                            ),
                            &device,
                        ),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap_or_else(|error| panic!("decode reference step {step}: {error}"));
        }
    }
    assert_eq!(state.lengths(), &[SIX_TOKEN_IDS.len() + 31]);

    let mut teacher_ids = SIX_TOKEN_IDS.into_iter().map(i64::from).collect::<Vec<_>>();
    teacher_ids.extend(
        reference
            .greedy_token_ids
            .iter()
            .take(31)
            .copied()
            .map(i64::from),
    );
    let teacher_sequence = teacher_ids.len();
    let teacher = loaded
        .model()
        .forward(
            CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(teacher_ids, [1, teacher_sequence]),
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
        .expect("run all-logit Qwen3-MoE teacher-forced reference graph");
    assert_eq!(teacher.logits.dims(), [1, 37, 151_936]);
    let teacher_values = metal_values("Qwen3-MoE Burn teacher logits", &teacher.logits);
    let mut minimum_teacher_cosine = f64::INFINITY;
    for step in 0..32 {
        let position = SIX_TOKEN_IDS.len() - 1 + step;
        let actual = &teacher_values[position * 151_936..(position + 1) * 151_936];
        let expected = &reference.teacher_prefix_logits[step * 151_936..(step + 1) * 151_936];
        let cosine = cosine_similarity(actual, expected);
        assert!(
            cosine >= 0.999,
            "teacher-forced official-reference cosine at step {step} was {cosine}"
        );
        minimum_teacher_cosine = minimum_teacher_cosine.min(cosine);
    }
    eprintln!(
        "Qwen3-MoE official parity: stateless_cosine={stateless_cosine:.12}, \
         minimum_cached_cosine={minimum_cached_cosine:.12}, \
         minimum_teacher_cosine={minimum_teacher_cosine:.12}, greedy_tokens=32"
    );
}

/// Returns synchronized F32 values for one Metal acceptance diagnostic.
#[cfg(feature = "metal")]
fn metal_values<const D: usize>(label: &str, tensor: &Tensor<Metal, D>) -> Vec<f32> {
    tensor
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .unwrap_or_else(|error| panic!("materialize {label}: {error}"))
}

/// Computes cosine similarity with F64 accumulation and rejects zero-length vectors.
#[cfg(feature = "metal")]
fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len());
    assert!(!left.is_empty(), "cosine vectors must not be empty");
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
    assert!(left_square > 0.0 && right_square > 0.0);
    dot / (left_square.sqrt() * right_square.sqrt())
}

/// Computes a stable FNV-1a digest over little-endian F32 bit patterns.
#[cfg(feature = "metal")]
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

/// Verifies one installed Metal linear and returns its physical parameter count.
#[cfg(feature = "metal")]
fn assert_linear_f16(label: &str, linear: &Linear<Metal>) -> usize {
    assert_eq!(
        linear.weight.val().dtype(),
        DType::F16,
        "{label}.weight was not installed as F16"
    );
    match &linear.bias {
        Some(bias) => {
            assert_eq!(
                bias.val().dtype(),
                DType::F16,
                "{label}.bias was not installed as F16"
            );
            2
        }
        None => 1,
    }
}

/// Traverses all 18,867 installed Qwen3-MoE parameters and requires F16 storage.
#[cfg(feature = "metal")]
fn assert_all_parameters_f16(model: &Qwen3MoeForCausalLm<Metal>) {
    let mut parameters = 0_usize;
    assert_eq!(model.model.embed_tokens.weight.val().dtype(), DType::F16);
    parameters += 1;
    assert_eq!(model.model.norm.weight.val().dtype(), DType::F16);
    parameters += 1;
    parameters += assert_linear_f16(
        "lm_head",
        model
            .lm_head
            .as_ref()
            .expect("the untied pinned checkpoint must own an LM head"),
    );

    assert_eq!(model.model.layers.len(), 48);
    for (layer_index, layer) in model.model.layers.iter().enumerate() {
        for (name, norm) in [
            ("input_layernorm", &layer.input_layernorm),
            ("post_attention_layernorm", &layer.post_attention_layernorm),
        ] {
            assert_eq!(
                norm.weight.val().dtype(),
                DType::F16,
                "layer {layer_index} {name} was not installed as F16"
            );
            parameters += 1;
        }
        for (name, norm) in [
            (
                "q_norm",
                layer
                    .self_attn
                    .query_norm
                    .as_ref()
                    .expect("Qwen3-MoE query norm"),
            ),
            (
                "k_norm",
                layer
                    .self_attn
                    .key_norm
                    .as_ref()
                    .expect("Qwen3-MoE key norm"),
            ),
        ] {
            assert_eq!(
                norm.weight.val().dtype(),
                DType::F16,
                "layer {layer_index} {name} was not installed as F16"
            );
            parameters += 1;
        }
        for (name, projection) in [
            ("q_proj", &layer.self_attn.q_proj),
            ("k_proj", &layer.self_attn.k_proj),
            ("v_proj", &layer.self_attn.v_proj),
            ("o_proj", &layer.self_attn.o_proj),
        ] {
            parameters += assert_linear_f16(
                &format!("model.layers.{layer_index}.self_attn.{name}"),
                projection,
            );
        }

        assert!(
            layer.dense_mlp.is_none(),
            "the pinned layer {layer_index} must be sparse"
        );
        let sparse = layer
            .sparse_mlp
            .as_ref()
            .expect("every pinned Qwen3-MoE layer must own routed experts");
        assert_eq!(sparse.routed.len(), 128);
        assert!(sparse.shared.is_empty());
        parameters += assert_linear_f16(
            &format!("model.layers.{layer_index}.sparse_mlp.router"),
            &sparse.router,
        );
        for (expert_index, expert) in sparse.routed.iter().enumerate() {
            for (name, projection) in [
                ("gate_proj", &expert.gate_proj),
                ("up_proj", &expert.up_proj),
                ("down_proj", &expert.down_proj),
            ] {
                parameters += assert_linear_f16(
                    &format!("model.layers.{layer_index}.sparse_mlp.routed.{expert_index}.{name}"),
                    projection,
                );
            }
        }
    }
    assert_eq!(parameters, PARAMETER_COUNT);
}

/// Forces adapter discovery and one synchronized storage round trip.
#[cfg(feature = "metal")]
fn assert_metal_backend_available(device: &<Metal as burn::tensor::backend::BackendTypes>::Device) {
    let probe = Tensor::<Metal, 1>::zeros([1], device)
        .to_data()
        .to_vec::<f32>()
        .expect("the Metal acceptance backend must expose F32 storage");
    assert_eq!(probe, [0.0]);
}

/// Fails quickly when macOS does not expose a usable Metal adapter.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires an available Metal adapter"]
fn initializes_metal_acceptance_backend() {
    assert_metal_backend_available(&Default::default());
}

/// Strictly loads F16 weights on Metal and exercises stateless, cached, and decode paths.
///
/// This gate intentionally requires explicit execution because the checkpoint
/// contains 61,064,245,248 parameter bytes after BF16-to-F16 conversion. It
/// requests every hidden state so a failed parity gate reports the earliest
/// divergent trace as well as whole-vector cosine and deterministic digests.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires the pinned 30B checkpoint, Metal, and about 70 GiB free unified memory"]
fn loads_pinned_qwen3_moe_f16_and_runs_metal_smoke() {
    let root = pinned_root();
    assert_control_file_hashes(&root);
    let device = Default::default();
    assert_metal_backend_available(&device);
    assert_shard_file_hashes(&root);
    let load_started = Instant::now();
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
    .expect("the public auto loader must strictly stream the complete Qwen3-MoE graph");
    <Metal<f32, i32> as Backend>::sync(&device)
        .expect("all Metal parameter transfers must complete");
    let load_seconds = load_started.elapsed().as_secs_f64();
    let concrete = match loaded.model() {
        AutoCausalLm::Qwen3Moe(model) => model,
        model => panic!("expected Qwen3-MoE auto dispatch, found {model:?}"),
    };
    assert_all_parameters_f16(concrete);
    assert_eq!(loaded.source().model_type(), "qwen3_moe");
    assert_eq!(loaded.source().repository(), Some(REPOSITORY));
    assert_eq!(loaded.source().revision(), Some(REVISION));
    assert_eq!(
        loaded.source().root(),
        root.canonicalize().expect("canonical pinned artifact root")
    );
    let report = loaded.load_report();
    assert!(report.is_strictly_valid());
    assert_eq!(report.applied.len(), PARAMETER_COUNT);
    assert_eq!(report.transformed.len(), 18_480);
    assert_eq!(report.dtype_converted.len(), PARAMETER_COUNT);
    assert!(report.aliased.is_empty());
    assert!(report.allowlisted_ignored.is_empty());
    assert!(report.allowlisted_missing.is_empty());
    let execution = loaded
        .execution_stats()
        .expect("the automatic loader must retain pass-two execution counters");
    assert_eq!(execution.tensors_applied, PARAMETER_COUNT);
    assert_eq!(execution.aliases_applied, 0);
    assert_eq!(execution.source_reads, PARAMETER_COUNT as u64);
    assert_eq!(execution.source_payload_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(execution.delivered_tensor_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(
        execution.largest_source_payload_bytes,
        LARGEST_CHECKPOINT_TENSOR_BYTES
    );
    assert_eq!(
        execution.largest_output_tensor_bytes,
        LARGEST_CHECKPOINT_TENSOR_BYTES
    );
    assert_eq!(
        execution.peak_host_staging_bytes,
        LARGEST_CHECKPOINT_TENSOR_BYTES
    );
    assert_eq!(execution.source.shard_opens, 16);
    assert_eq!(execution.source.tensor_reads, PARAMETER_COUNT as u64);
    assert_eq!(execution.source.payload_bytes_read, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(execution.source.max_open_shards, 1);
    assert_eq!(
        execution.source.shard_open_order,
        (1..=16)
            .map(|index| format!("model-{index:05}-of-00016.safetensors"))
            .collect::<Vec<_>>()
    );

    let encoding = loaded
        .tokenizer()
        .expect("the loaded bundle must retain the local tokenizer")
        .encode(
            SIX_TOKEN_PROMPT,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("the six-token Metal fixture must encode");
    assert_eq!(encoding.token_ids()[0], SIX_TOKEN_IDS);
    let sequence = SIX_TOKEN_IDS.len();
    let input = encoding
        .to_causal_input::<Metal>(&device)
        .expect("the six-token encoding must become a causal input");
    let diagnostics = ForwardOptions {
        hidden_states: true,
        ..ForwardOptions::default()
    };

    let stateless_started = Instant::now();
    let stateless = loaded
        .model()
        .forward(input.clone(), &diagnostics)
        .expect("the loaded model must execute stateless inference");
    assert_eq!(stateless.logits.dims(), [1, 1, 151_936]);
    assert_eq!(stateless.hidden_states.len(), 50);
    let stateless_logits = metal_values("Qwen3-MoE stateless logits", &stateless.logits);
    let stateless_trace = metal_values(
        "Qwen3-MoE stateless hidden trace",
        &Tensor::cat(stateless.hidden_states.clone(), 0),
    );
    let stateless_seconds = stateless_started.elapsed().as_secs_f64();
    assert!(stateless_logits.iter().all(|value| value.is_finite()));
    assert!(stateless_trace.iter().all(|value| value.is_finite()));

    let mut state = loaded
        .model()
        .new_decode_state(
            CacheConfig::new(1, sequence + 1).expect("the smoke cache geometry must validate"),
            1,
            &device,
        )
        .expect("the loaded model must allocate native Qwen3-MoE KV state");
    let prefill_started = Instant::now();
    let cached = loaded
        .model()
        .prefill(input, &mut state, &diagnostics)
        .expect("the loaded model must execute cached prefill");
    assert_eq!(cached.logits.dims(), [1, 1, 151_936]);
    assert_eq!(cached.hidden_states.len(), 50);
    let cached_logits = metal_values("Qwen3-MoE cached logits", &cached.logits);
    let cached_trace = metal_values(
        "Qwen3-MoE cached hidden trace",
        &Tensor::cat(cached.hidden_states.clone(), 0),
    );
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    assert!(cached_logits.iter().all(|value| value.is_finite()));
    assert!(cached_trace.iter().all(|value| value.is_finite()));
    assert_eq!(state.lengths(), &[sequence]);

    let cached_cosine = cosine_similarity(&stateless_logits, &cached_logits);
    assert!(
        cached_cosine >= 0.999,
        "Qwen3-MoE cached/stateless logit cosine {cached_cosine} is below the gate"
    );
    let trace_width = sequence * 2_048;
    let hidden_cosines = stateless_trace
        .chunks_exact(trace_width)
        .zip(cached_trace.chunks_exact(trace_width))
        .map(|(stateless, cached)| cosine_similarity(stateless, cached))
        .collect::<Vec<_>>();
    assert_eq!(hidden_cosines.len(), 50);
    for (trace, cosine) in hidden_cosines.iter().copied().enumerate() {
        assert!(
            cosine >= 0.999,
            "Qwen3-MoE cached/stateless hidden cosine at trace {trace} was {cosine}"
        );
    }
    let minimum_hidden_cosine = hidden_cosines.iter().copied().fold(f64::INFINITY, f64::min);
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
        .expect("the loaded model must execute one cached decode step");
    assert_eq!(decoded.logits.dims(), [1, 1, 151_936]);
    let decoded_logits = metal_values("Qwen3-MoE decoded logits", &decoded.logits);
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    assert!(decoded_logits.iter().all(|value| value.is_finite()));
    assert_eq!(state.lengths(), &[sequence + 1]);

    eprintln!(
        "Qwen3-MoE F16 Metal: prompt_tokens={sequence}, first_top1={next_token}, \
         cached_cosine={cached_cosine:.9}, min_hidden_cosine={minimum_hidden_cosine:.9}, \
         stateless_logits_fnv1a64={:016x}, cached_logits_fnv1a64={:016x}, \
         stateless_trace_fnv1a64={:016x}, cached_trace_fnv1a64={:016x}, \
         decoded_logits_fnv1a64={:016x}, load_seconds={load_seconds:.3}, \
         shard_opens={}, max_open_shards={}, source_reads={}, \
         stateless_seconds={stateless_seconds:.3}, prefill_seconds={prefill_seconds:.3}, \
         decode_seconds={decode_seconds:.3}",
        fnv1a64_f32(&stateless_logits),
        fnv1a64_f32(&cached_logits),
        fnv1a64_f32(&stateless_trace),
        fnv1a64_f32(&cached_trace),
        fnv1a64_f32(&decoded_logits),
        execution.source.shard_opens,
        execution.source.max_open_shards,
        execution.source_reads,
    );
}

/// Converts every eligible Qwen3-MoE matrix into a persistent group-64 Q8 sidecar.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires pinned Qwen3-30B-A3B weights and an absent external Q8 Burnpack path"]
fn quantizes_pinned_qwen3_moe_weights_to_q8_burnpack() {
    quantize_pinned_qwen3_moe_weights_to_burnpack(SymmetricQuantization::Q8, Q8_BURNPACK_ENV);
}

/// Converts every eligible Qwen3-MoE matrix into a persistent group-64 Q4 sidecar.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires pinned Qwen3-30B-A3B weights and an absent external Q4 Burnpack path"]
fn quantizes_pinned_qwen3_moe_weights_to_q4_burnpack() {
    quantize_pinned_qwen3_moe_weights_to_burnpack(SymmetricQuantization::Q4, Q4_BURNPACK_ENV);
}

/// Loads the persistent Q8 sidecar and exercises packed Metal inference.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires pinned Qwen3-30B-A3B weights, a complete Q8 Burnpack, and Metal"]
fn loads_pinned_qwen3_moe_q8_and_runs_metal_smoke() {
    load_pinned_qwen3_moe_burnpack_and_run_metal(SymmetricQuantization::Q8, Q8_BURNPACK_ENV);
}

/// Loads the persistent Q4 sidecar and exercises packed Metal inference.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires pinned Qwen3-30B-A3B weights, a complete Q4 Burnpack, and Metal"]
fn loads_pinned_qwen3_moe_q4_and_runs_metal_smoke() {
    load_pinned_qwen3_moe_burnpack_and_run_metal(SymmetricQuantization::Q4, Q4_BURNPACK_ENV);
}

/// Audits every packed Q8 projection submitted by one recurrent Metal decode.
#[cfg(all(feature = "kernel-tracing", feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires pinned Qwen3-30B-A3B weights, a complete Q8 Burnpack, Metal, and kernel tracing"]
fn audits_pinned_qwen3_moe_q8_decode_packed_metal_kernels() {
    load_pinned_qwen3_moe_burnpack_and_run_metal(SymmetricQuantization::Q8, Q8_BURNPACK_ENV);
}

/// Audits every packed Q4 projection submitted by one recurrent Metal decode.
#[cfg(all(feature = "kernel-tracing", feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires pinned Qwen3-30B-A3B weights, a complete Q4 Burnpack, Metal, and kernel tracing"]
fn audits_pinned_qwen3_moe_q4_decode_packed_metal_kernels() {
    load_pinned_qwen3_moe_burnpack_and_run_metal(SymmetricQuantization::Q4, Q4_BURNPACK_ENV);
}

/// Streams one exact checkpoint revision into a create-only Burnpack sidecar.
///
/// Full source hashes are verified before conversion. The writer consumes the
/// strict architecture plan rather than raw Hugging Face names, and the result
/// is reopened and fully validated before its content digest is reported. The
/// caller must provide an exact path because finalization temporarily requires
/// both the private staging bytes and the published Burnpack on that filesystem.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn quantize_pinned_qwen3_moe_weights_to_burnpack(
    scheme: SymmetricQuantization,
    output_environment: &str,
) {
    let supplied_root = pinned_root();
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert_control_file_hashes(&root);
    assert_shard_file_hashes(&root);
    let output = explicit_external_burnpack_path(&root, output_environment, false);
    let (_, manifest, plan) = pinned_qwen3_moe_plan(&root, metal_f16_load_options());
    let (expected_quantized, expected_excluded) = qwen3_moe_quantization_partition(&plan);
    assert_qwen3_moe_quantization_partition(&expected_quantized, &expected_excluded);
    assert_eq!(
        retained_source_tensor_bytes(&plan, &expected_excluded),
        RETAINED_SOURCE_TENSOR_BYTES
    );
    let (expected_payload_bytes, expected_largest_payload) =
        expected_quantized_payload(&plan, scheme);
    assert_eq!(
        (expected_payload_bytes, expected_largest_payload),
        pinned_quantized_payload(scheme)
    );
    let label = quantization_label(scheme);

    let conversion_started = Instant::now();
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
    .unwrap_or_else(|error| panic!("Qwen3-MoE {label} conversion failed: {error}"));
    let conversion_seconds = conversion_started.elapsed().as_secs_f64();

    assert!(outcome.report.is_strictly_valid());
    assert_eq!(outcome.report.applied.len(), PARAMETER_COUNT);
    assert_eq!(outcome.report.transformed.len(), 18_480);
    assert!(outcome.report.aliased.is_empty());
    assert_eq!(outcome.report.dtype_converted.len(), PARAMETER_COUNT);
    assert!(outcome.report.dtype_converted.iter().all(|conversion| {
        conversion.source == TensorDType::BF16 && conversion.target_dtype == TensorDType::F16
    }));
    assert!(outcome.report.allowlisted_ignored.is_empty());
    assert!(outcome.report.allowlisted_missing.is_empty());
    assert_eq!(
        outcome
            .report
            .quantized
            .iter()
            .map(|record| record.target.clone())
            .collect::<Vec<_>>(),
        expected_quantized
    );
    assert!(outcome.report.quantized.iter().all(|record| {
        record.scheme == format!("symmetric-{label}-g64-flat-row-major") && record.bytes > 0
    }));
    assert_eq!(
        outcome
            .report
            .quantized
            .iter()
            .map(|record| record.bytes)
            .sum::<u64>(),
        expected_payload_bytes
    );
    assert_eq!(
        outcome
            .excluded
            .iter()
            .map(|record| (record.target.clone(), record.reason))
            .collect::<Vec<_>>(),
        expected_excluded
    );
    assert_complete_qwen3_moe_source_execution(&outcome.execution);
    assert_complete_qwen3_moe_provenance(outcome.provenance);

    assert_eq!(outcome.staging.tensor_count, QUANTIZED_PARAMETER_COUNT);
    assert_eq!(outcome.staging.staged_payload_bytes, expected_payload_bytes);
    assert_eq!(
        outcome.staging.largest_payload_bytes,
        expected_largest_payload
    );
    assert!(outcome.staging.estimated_manifest_bytes > 0);
    assert_eq!(outcome.artifact.path, output);
    assert_eq!(outcome.artifact.tensor_count, QUANTIZED_PARAMETER_COUNT);
    assert_eq!(
        outcome.artifact.quantized_payload_bytes,
        expected_payload_bytes
    );
    assert_eq!(
        outcome.artifact.largest_materialized_tensor_bytes,
        expected_largest_payload
    );
    assert!(outcome.artifact.manifest_bytes > 0);
    assert!(outcome.artifact.artifact_bytes > expected_payload_bytes);
    assert_eq!(
        fs::metadata(&output)
            .unwrap_or_else(|error| panic!("inspect {}: {error}", output.display()))
            .len(),
        outcome.artifact.artifact_bytes
    );

    let mut artifact = QuantizedBurnpackArtifact::open(&output)
        .unwrap_or_else(|error| panic!("open Qwen3-MoE {label} Burnpack: {error}"));
    assert_exact_qwen3_moe_burnpack(
        &mut artifact,
        &plan,
        scheme,
        &expected_quantized,
        expected_payload_bytes,
        expected_largest_payload,
    );
    drop(artifact);
    let artifact_sha256 = sha256_file_hex(&output);
    eprintln!(
        "Qwen3-MoE {label} conversion: path={}, sha256={artifact_sha256}, \
         tensors={}, retained_f16={}, payload_bytes={expected_payload_bytes}, \
         artifact_bytes={}, largest_payload_bytes={expected_largest_payload}, \
         source_reads={}, max_open_shards={}, conversion_seconds={conversion_seconds:.3}",
        output.display(),
        QUANTIZED_PARAMETER_COUNT,
        RETAINED_F16_PARAMETER_COUNT,
        outcome.artifact.artifact_bytes,
        outcome.execution.source_reads,
        outcome.execution.source.max_open_shards,
    );
}

/// Loads a complete sidecar through the public quantized-plan executor and runs it.
///
/// The automatic registry uses the audited form of this executor and retains its
/// format, preflight, packed-execution, and provenance evidence in
/// [`transformers::LoadedModel`]. This acceptance gate calls the compatibility
/// API directly to exercise the concrete architecture sink while independently
/// certifying its packed-payload and provenance-hash counters.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn load_pinned_qwen3_moe_burnpack_and_run_metal(
    scheme: SymmetricQuantization,
    output_environment: &str,
) {
    let supplied_root = pinned_root();
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert_control_file_hashes(&root);
    assert_shard_file_hashes(&root);
    let burnpack = explicit_external_burnpack_path(&root, output_environment, true);
    let artifact_sha256 = sha256_file_hex(&burnpack);
    let load_options = metal_f16_load_options();
    let (config, manifest, plan) = pinned_qwen3_moe_plan(&root, load_options);
    assert_eq!(f16_plan_digest(&plan), F16_PLAN_SHA256);
    let (expected_quantized, expected_excluded) = qwen3_moe_quantization_partition(&plan);
    assert_qwen3_moe_quantization_partition(&expected_quantized, &expected_excluded);
    assert_eq!(
        retained_source_tensor_bytes(&plan, &expected_excluded),
        RETAINED_SOURCE_TENSOR_BYTES
    );
    let (expected_payload_bytes, expected_largest_payload) =
        expected_quantized_payload(&plan, scheme);
    assert_eq!(
        (expected_payload_bytes, expected_largest_payload),
        pinned_quantized_payload(scheme)
    );
    let label = quantization_label(scheme);

    let mut artifact = QuantizedBurnpackArtifact::open(&burnpack)
        .unwrap_or_else(|error| panic!("open Qwen3-MoE {label} Burnpack: {error}"));
    assert_exact_qwen3_moe_burnpack(
        &mut artifact,
        &plan,
        scheme,
        &expected_quantized,
        expected_payload_bytes,
        expected_largest_payload,
    );
    let device = Default::default();
    assert_metal_backend_available(&device);
    let mut model = Qwen3MoeForCausalLm::<Metal>::init(config, &device)
        .expect("the pinned Qwen3-MoE topology must initialize lazily on Metal");
    let load_started = Instant::now();
    let outcome = execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut model)
        .unwrap_or_else(|error| panic!("load mixed F16/{label} Qwen3-MoE model: {error}"));
    <Metal<f32, i32> as Backend>::sync(&device)
        .expect("all packed and F16 Metal parameter transfers must complete");
    let load_seconds = load_started.elapsed().as_secs_f64();

    assert!(outcome.report.is_strictly_valid());
    assert_eq!(outcome.report.applied.len(), PARAMETER_COUNT);
    assert_eq!(outcome.report.transformed.len(), 18_480);
    assert_eq!(outcome.report.dtype_converted.len(), PARAMETER_COUNT);
    assert!(outcome.report.aliased.is_empty());
    assert!(outcome.report.allowlisted_ignored.is_empty());
    assert!(outcome.report.allowlisted_missing.is_empty());
    assert_eq!(outcome.report.quantized.len(), QUANTIZED_PARAMETER_COUNT);
    assert!(
        outcome.report.quantized.iter().all(|record| {
            record.scheme == format!("burn-qfloat-symmetric-{label}-g64-block-1xg")
        })
    );
    assert_eq!(
        outcome
            .report
            .quantized
            .iter()
            .map(|record| record.target.clone())
            .collect::<Vec<_>>(),
        expected_quantized
    );
    assert_qwen3_moe_packed_overlay_execution(&outcome.execution, expected_payload_bytes);
    assert_complete_qwen3_moe_provenance(outcome.provenance);
    assert_eq!(
        outcome.packed.tensor_reads,
        QUANTIZED_PARAMETER_COUNT as u64
    );
    assert_eq!(outcome.packed.tensors_applied, QUANTIZED_PARAMETER_COUNT);
    assert_eq!(outcome.packed.payload_bytes, expected_payload_bytes);
    assert_eq!(
        outcome.packed.largest_payload_bytes,
        expected_largest_payload
    );
    assert_qwen3_moe_mixed_parameter_dtypes(&model);

    run_qwen3_moe_quantized_metal_smoke(
        &model,
        &root,
        &burnpack,
        &artifact_sha256,
        scheme,
        load_seconds,
        &outcome.execution,
        expected_payload_bytes,
    );
}

/// Returns the one target-dtype policy signed by conversion and used on M3 Metal.
///
/// Burn's Metal backend cannot store the official BF16 parameters on the
/// acceptance device. Conversion must therefore quantize the exact F16 plan that
/// later installs the sidecar; Burnpack v2 deliberately rejects even a
/// shape-compatible plan whose target dtype differs.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn metal_f16_load_options() -> LoadOptions {
    LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        ..LoadOptions::default()
    }
}

/// Reconstructs the exact immutable family plan for one target dtype policy.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn pinned_qwen3_moe_plan(
    root: &Path,
    options: LoadOptions,
) -> (Qwen3MoeConfig, CheckpointManifest, LoadPlan) {
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", config_path.display()));
    let config = Qwen3MoeConfig::from_json_str(&config_json)
        .expect("the pinned Qwen3-MoE configuration must validate");
    let expected = Qwen3MoeConfig::from_json_str(include_str!("fixtures/qwen3_30b_a3b.json"))
        .expect("the committed Qwen3-MoE fixture must validate");
    assert_eq!(config, expected);

    let manifest = CheckpointManifest::discover(root, &options)
        .expect("the pinned Qwen3-MoE SafeTensors inventory must validate");
    assert_eq!(manifest.shards.len(), 16);
    assert_eq!(manifest.tensors.len(), PARAMETER_COUNT);
    assert_eq!(manifest.tensor_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(header_inventory_digest(&manifest), HEADER_INVENTORY_SHA256);
    assert!(
        manifest
            .tensors
            .values()
            .all(|tensor| tensor.dtype == TensorDType::BF16)
    );
    let contract = config
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .expect("the pinned Qwen3-MoE topology must produce a checkpoint contract");
    let plan = contract
        .build_load_plan(&manifest, &options)
        .expect("every pinned Qwen3-MoE tensor must map strictly");
    let canonical_f16_plan = contract
        .build_load_plan(
            &manifest,
            &LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
        )
        .expect("the pinned Qwen3-MoE mapping must remain valid under canonical F16 planning");
    assert_eq!(f16_plan_digest(&canonical_f16_plan), F16_PLAN_SHA256);
    assert_eq!(plan.parameter_count(), PARAMETER_COUNT);
    assert_eq!(plan.tensors.len(), PARAMETER_COUNT);
    assert!(plan.aliases.is_empty());
    assert!(plan.report.is_strictly_valid());
    assert!(plan.report.allowlisted_ignored.is_empty());
    assert!(plan.report.allowlisted_missing.is_empty());
    assert!(plan.tensors.iter().all(|tensor| {
        tensor.sources.len() == 1 && matches!(tensor.transform, PlannedTransform::Identity)
    }));
    (config, manifest, plan)
}

/// Partitions the strict plan with the public conservative weight-only policy.
fn qwen3_moe_quantization_partition(
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
                    .expect("Qwen3-MoE dimensions must fit the host architecture")
            })
            .collect::<Vec<_>>();
        match classify_weight_for_quantization(&tensor.target, &shape) {
            QuantizationEligibility::Eligible => quantized.push(tensor.target.clone()),
            QuantizationEligibility::Excluded(reason) => {
                excluded.push((tensor.target.clone(), reason));
            }
        }
    }
    quantized.sort();
    excluded.sort_by(|left, right| left.0.cmp(&right.0));
    (quantized, excluded)
}

/// Sums the exact source intervals retained by one quantization partition.
fn retained_source_tensor_bytes(
    plan: &LoadPlan,
    excluded: &[(String, QuantizationExclusion)],
) -> u64 {
    let excluded = excluded
        .iter()
        .map(|(target, _)| target.as_str())
        .collect::<BTreeSet<_>>();
    let retained = plan
        .tensors
        .iter()
        .filter(|tensor| excluded.contains(tensor.target.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(retained.len(), RETAINED_F16_PARAMETER_COUNT);
    retained.into_iter().fold(0_u64, |total, tensor| {
        let [source] = tensor.sources.as_slice() else {
            panic!("retained target {} must have one source", tensor.target);
        };
        let bytes = source.data_offsets[1]
            .checked_sub(source.data_offsets[0])
            .expect("retained source offsets must be ordered");
        total
            .checked_add(bytes)
            .expect("retained source-byte sum must fit u64")
    })
}

/// Requires the immutable 18,625-packed/242-retained mixed-precision boundary.
fn assert_qwen3_moe_quantization_partition(
    quantized: &[String],
    excluded: &[(String, QuantizationExclusion)],
) {
    assert_eq!(quantized.len(), QUANTIZED_PARAMETER_COUNT);
    assert_eq!(excluded.len(), RETAINED_F16_PARAMETER_COUNT);
    for (reason, expected) in [
        (QuantizationExclusion::Embedding, 1),
        (QuantizationExclusion::Normalization, 193),
        (QuantizationExclusion::Router, 48),
    ] {
        assert_eq!(
            excluded
                .iter()
                .filter(|(_, observed)| *observed == reason)
                .count(),
            expected,
            "unexpected Qwen3-MoE exclusion count for {reason:?}"
        );
    }
    assert!(excluded.iter().all(|(_, reason)| matches!(
        reason,
        QuantizationExclusion::Embedding
            | QuantizationExclusion::Normalization
            | QuantizationExclusion::Router
    )));
}

/// Calculates exact packed values plus one F32 scale per contiguous group.
fn packed_tensor_payload_bytes(
    elements: u64,
    scheme: SymmetricQuantization,
    group_size: u64,
) -> u64 {
    assert!(group_size > 0, "quantization group size must be nonzero");
    let packed = match scheme {
        SymmetricQuantization::Q8 => elements,
        SymmetricQuantization::Q4 => elements / 2 + elements % 2,
    };
    let groups = elements / group_size + u64::from(!elements.is_multiple_of(group_size));
    packed
        .checked_add(groups.checked_mul(4).expect("scale bytes must fit u64"))
        .expect("packed tensor payload must fit u64")
}

/// Computes exact total and largest sidecar payloads independently of the writer.
fn expected_quantized_payload(plan: &LoadPlan, scheme: SymmetricQuantization) -> (u64, u64) {
    let group_size = u64::try_from(WeightQuantizationConfig::default().group_size.get())
        .expect("group size must fit u64");
    assert_eq!(group_size, 64);
    let mut total = 0_u64;
    let mut largest = 0_u64;
    let mut count = 0_usize;
    for tensor in &plan.tensors {
        let host_shape = tensor
            .shape
            .iter()
            .map(|&dimension| usize::try_from(dimension).expect("dimension must fit usize"))
            .collect::<Vec<_>>();
        if !matches!(
            classify_weight_for_quantization(&tensor.target, &host_shape),
            QuantizationEligibility::Eligible
        ) {
            continue;
        }
        let elements = tensor
            .shape
            .iter()
            .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension));
        let elements = elements.expect("Qwen3-MoE tensor element count must fit u64");
        let bytes = packed_tensor_payload_bytes(elements, scheme, group_size);
        total = total
            .checked_add(bytes)
            .expect("Qwen3-MoE packed payload sum must fit u64");
        largest = largest.max(bytes);
        count += 1;
    }
    assert_eq!(count, QUANTIZED_PARAMETER_COUNT);
    (total, largest)
}

/// Returns the immutable group-64 payload sizes derived from the pinned inventory.
const fn pinned_quantized_payload(scheme: SymmetricQuantization) -> (u64, u64) {
    match scheme {
        SymmetricQuantization::Q8 => (Q8_PAYLOAD_BYTES, Q8_LARGEST_PAYLOAD_BYTES),
        SymmetricQuantization::Q4 => (Q4_PAYLOAD_BYTES, Q4_LARGEST_PAYLOAD_BYTES),
    }
}

/// Checks complete one-shard-at-a-time consumption of the 16-shard checkpoint.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn assert_complete_qwen3_moe_source_execution(execution: &ExecutionStats) {
    assert_eq!(execution.tensors_applied, PARAMETER_COUNT);
    assert_eq!(execution.aliases_applied, 0);
    assert_eq!(execution.source_reads, PARAMETER_COUNT as u64);
    assert_eq!(execution.source_payload_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(execution.delivered_tensor_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(
        execution.largest_source_payload_bytes,
        LARGEST_CHECKPOINT_TENSOR_BYTES
    );
    assert_eq!(
        execution.largest_output_tensor_bytes,
        LARGEST_CHECKPOINT_TENSOR_BYTES
    );
    assert_eq!(
        execution.peak_host_staging_bytes,
        LARGEST_CHECKPOINT_TENSOR_BYTES
    );
    assert_eq!(execution.source.shard_opens, 16);
    assert_eq!(execution.source.tensor_reads, PARAMETER_COUNT as u64);
    assert_eq!(execution.source.payload_bytes_read, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(execution.source.max_open_shards, 1);
    assert_eq!(
        execution.source.shard_open_order,
        (1..=16)
            .map(|index| format!("model-{index:05}-of-00016.safetensors"))
            .collect::<Vec<_>>()
    );
}

/// Checks both bounded full-shard hash passes around a successful v2 operation.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn assert_complete_qwen3_moe_provenance(provenance: ProvenanceExecutionStats) {
    for (phase, stats) in [
        ("pre-execution", provenance.pre_execution),
        ("post-execution", provenance.post_execution),
    ] {
        let stats = stats.unwrap_or_else(|| panic!("v2 omitted its {phase} provenance pass"));
        assert_eq!(stats.passes_completed, 1, "{phase} pass count");
        assert_eq!(stats.shards_hashed, 16, "{phase} shard count");
        assert_eq!(
            stats.bytes_hashed, CHECKPOINT_ARTIFACT_BYTES,
            "{phase} hashed byte count"
        );
        assert_eq!(
            stats.peak_buffer_bytes, PROVENANCE_HASH_BUFFER_BYTES,
            "{phase} hash-buffer bound"
        );
    }
    assert_eq!(
        provenance.bytes_hashed(),
        u128::from(CHECKPOINT_ARTIFACT_BYTES) * 2
    );
    assert_eq!(provenance.peak_buffer_bytes(), PROVENANCE_HASH_BUFFER_BYTES);
}

/// Checks content-verified v2 source bypass for the mixed packed/F16 load.
///
/// The executor hashes every complete shard before the first model write. Once
/// that checkpoint and the exact F16 plan match the v2 manifest, the 18,625
/// packed matrices need no redundant BF16 materialization; only embeddings,
/// norms, and routers are read from SafeTensors. Hash I/O is intentionally
/// separate from tensor-read counters, while the persisted official shard hashes
/// are checked by [`assert_exact_qwen3_moe_burnpack`].
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn assert_qwen3_moe_packed_overlay_execution(
    execution: &ExecutionStats,
    packed_payload_bytes: u64,
) {
    assert_eq!(execution.tensors_applied, PARAMETER_COUNT);
    assert_eq!(execution.aliases_applied, 0);
    assert_eq!(execution.source_reads, RETAINED_F16_PARAMETER_COUNT as u64);
    assert_eq!(execution.source_payload_bytes, RETAINED_SOURCE_TENSOR_BYTES);
    assert_eq!(
        execution.delivered_tensor_bytes,
        RETAINED_SOURCE_TENSOR_BYTES
            .checked_add(packed_payload_bytes)
            .expect("mixed Qwen3-MoE delivered bytes must fit u64")
    );
    assert_eq!(
        execution.largest_source_payload_bytes,
        LARGEST_CHECKPOINT_TENSOR_BYTES
    );
    assert_eq!(
        execution.largest_output_tensor_bytes,
        LARGEST_CHECKPOINT_TENSOR_BYTES
    );
    assert_eq!(
        execution.peak_host_staging_bytes,
        LARGEST_CHECKPOINT_TENSOR_BYTES
    );
    assert_eq!(execution.source.shard_opens, 16);
    assert_eq!(
        execution.source.tensor_reads,
        RETAINED_F16_PARAMETER_COUNT as u64
    );
    assert_eq!(
        execution.source.payload_bytes_read,
        RETAINED_SOURCE_TENSOR_BYTES
    );
    assert_eq!(execution.source.max_open_shards, 1);
    assert_eq!(
        execution.source.shard_open_order,
        (1..=16)
            .map(|index| format!("model-{index:05}-of-00016.safetensors"))
            .collect::<Vec<_>>()
    );
}

/// Reopens and validates the exact scheme, target set, and payload geometry.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn assert_exact_qwen3_moe_burnpack(
    artifact: &mut QuantizedBurnpackArtifact,
    plan: &LoadPlan,
    scheme: SymmetricQuantization,
    expected_quantized: &[String],
    expected_payload_bytes: u64,
    expected_largest_payload: u64,
) {
    validate_sidecar_contract(plan, artifact)
        .unwrap_or_else(|error| panic!("validate Burnpack contract: {error}"));
    let validation = artifact.validate_all().unwrap_or_else(|error| {
        panic!("validate {} Burnpack: {error}", quantization_label(scheme))
    });
    assert_eq!(validation.tensor_count, QUANTIZED_PARAMETER_COUNT);
    assert_eq!(validation.payload_bytes, expected_payload_bytes);
    let manifest = artifact.manifest();
    assert_eq!(
        manifest.format_version,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    let provenance = manifest
        .provenance
        .as_ref()
        .expect("architecture-aware conversion must publish v2 provenance");
    assert_eq!(provenance.load_plan_sha256, load_plan_sha256(plan));
    assert_eq!(provenance.checkpoint_shards.len(), SHARD_FILE_SHA256.len());
    let mut provenance_bytes = 0_u64;
    for (relative_path, expected_sha256) in SHARD_FILE_SHA256 {
        let shard = provenance
            .checkpoint_shards
            .get(relative_path)
            .unwrap_or_else(|| panic!("v2 provenance omitted {relative_path}"));
        assert_eq!(shard.sha256, expected_sha256);
        provenance_bytes = provenance_bytes
            .checked_add(shard.file_bytes)
            .expect("provenance byte total must fit u64");
    }
    assert_eq!(provenance_bytes, plan.memory.artifact_bytes);
    assert!(manifest.packed_execution.is_available());
    assert_eq!(manifest.tensors.len(), QUANTIZED_PARAMETER_COUNT);
    assert_eq!(
        manifest.tensors.keys().cloned().collect::<Vec<_>>(),
        expected_quantized
    );
    assert!(manifest.tensors.values().all(|tensor| {
        tensor.scheme == scheme
            && tensor.group_size.get() == 64
            && tensor.source_dtype == SourceFloatDType::F16
            && tensor.payload_sha256.is_some()
    }));
    assert_eq!(
        manifest
            .tensors
            .values()
            .map(|tensor| tensor.packed_value_bytes + tensor.scale_bytes)
            .sum::<u64>(),
        expected_payload_bytes
    );
    assert_eq!(
        manifest
            .tensors
            .values()
            .map(|tensor| tensor.packed_value_bytes + tensor.scale_bytes)
            .max(),
        Some(expected_largest_payload)
    );
}

/// Requires QFloat storage for eligible matrices and F16 for every exclusion.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn assert_qwen3_moe_mixed_parameter_dtypes(model: &Qwen3MoeForCausalLm<Metal>) {
    let mut quantized = 0_usize;
    let mut retained = 0_usize;
    assert_eq!(model.model.embed_tokens.weight.val().dtype(), DType::F16);
    retained += 1;
    assert_eq!(model.model.norm.weight.val().dtype(), DType::F16);
    retained += 1;
    assert_linear_qfloat(
        "lm_head",
        model
            .lm_head
            .as_ref()
            .expect("the pinned Qwen3-MoE checkpoint has an independent LM head"),
    );
    quantized += 1;

    assert_eq!(model.model.layers.len(), 48);
    for (layer_index, layer) in model.model.layers.iter().enumerate() {
        for (name, norm) in [
            ("input_layernorm", &layer.input_layernorm),
            ("post_attention_layernorm", &layer.post_attention_layernorm),
        ] {
            assert_eq!(
                norm.weight.val().dtype(),
                DType::F16,
                "layer {layer_index} {name} must remain F16"
            );
            retained += 1;
        }
        for (name, norm) in [
            (
                "q_norm",
                layer
                    .self_attn
                    .query_norm
                    .as_ref()
                    .expect("Qwen3-MoE query norm"),
            ),
            (
                "k_norm",
                layer
                    .self_attn
                    .key_norm
                    .as_ref()
                    .expect("Qwen3-MoE key norm"),
            ),
        ] {
            assert_eq!(
                norm.weight.val().dtype(),
                DType::F16,
                "layer {layer_index} {name} must remain F16"
            );
            retained += 1;
        }
        for (name, projection) in [
            ("q_proj", &layer.self_attn.q_proj),
            ("k_proj", &layer.self_attn.k_proj),
            ("v_proj", &layer.self_attn.v_proj),
            ("o_proj", &layer.self_attn.o_proj),
        ] {
            assert_linear_qfloat(
                &format!("model.layers.{layer_index}.self_attn.{name}"),
                projection,
            );
            quantized += 1;
        }

        assert!(layer.dense_mlp.is_none());
        let sparse = layer
            .sparse_mlp
            .as_ref()
            .expect("each pinned Qwen3-MoE layer must own routed experts");
        assert_eq!(sparse.routed.len(), 128);
        assert!(sparse.shared.is_empty());
        retained += assert_linear_f16(
            &format!("model.layers.{layer_index}.sparse_mlp.router"),
            &sparse.router,
        );
        for (expert_index, expert) in sparse.routed.iter().enumerate() {
            for (name, projection) in [
                ("gate_proj", &expert.gate_proj),
                ("up_proj", &expert.up_proj),
                ("down_proj", &expert.down_proj),
            ] {
                assert_linear_qfloat(
                    &format!("model.layers.{layer_index}.sparse_mlp.routed.{expert_index}.{name}"),
                    projection,
                );
                quantized += 1;
            }
        }
    }
    assert_eq!(quantized, QUANTIZED_PARAMETER_COUNT);
    assert_eq!(retained, RETAINED_F16_PARAMETER_COUNT);
    assert_eq!(quantized + retained, PARAMETER_COUNT);
}

/// Requires one bias-free linear weight to retain Burn's packed QFloat storage.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn assert_linear_qfloat(label: &str, linear: &Linear<Metal>) {
    assert!(
        matches!(linear.weight.val().dtype(), DType::QFloat(_)),
        "{label}.weight must retain QFloat storage"
    );
    assert!(
        linear.bias.is_none(),
        "{label} unexpectedly contains an unaccounted bias"
    );
}

/// Executes six-token stateless and cached prefill plus one recurrent decode.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[allow(clippy::too_many_arguments)]
fn run_qwen3_moe_quantized_metal_smoke(
    model: &Qwen3MoeForCausalLm<Metal>,
    root: &Path,
    burnpack: &Path,
    artifact_sha256: &str,
    scheme: SymmetricQuantization,
    load_seconds: f64,
    execution: &ExecutionStats,
    packed_payload_bytes: u64,
) {
    let label = quantization_label(scheme);
    let device = Default::default();
    let tokenizer = LocalTokenizer::from_directory(root)
        .expect("the exact pinned tokenizer must load locally for packed inference");
    let encoding = tokenizer
        .encode(
            SIX_TOKEN_PROMPT,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("the packed Metal prompt must encode");
    assert_eq!(encoding.token_ids()[0], SIX_TOKEN_IDS);
    let sequence = SIX_TOKEN_IDS.len();
    let input = encoding
        .to_causal_input::<Metal>(&device)
        .expect("the packed Metal prompt must become a causal input");
    let diagnostics = ForwardOptions {
        hidden_states: true,
        ..ForwardOptions::default()
    };

    let stateless_started = Instant::now();
    let stateless = model
        .forward(input.clone(), &diagnostics)
        .unwrap_or_else(|error| panic!("run stateless Qwen3-MoE {label}: {error}"));
    assert_eq!(stateless.logits.dims(), [1, 1, 151_936]);
    assert_eq!(stateless.hidden_states.len(), 50);
    let stateless_logits = metal_values("quantized Qwen3-MoE stateless logits", &stateless.logits);
    let stateless_trace = metal_values(
        "quantized Qwen3-MoE stateless trace",
        &Tensor::cat(stateless.hidden_states.clone(), 0),
    );
    let stateless_seconds = stateless_started.elapsed().as_secs_f64();
    assert!(stateless_logits.iter().all(|value| value.is_finite()));
    assert!(stateless_trace.iter().all(|value| value.is_finite()));

    let mut state = model
        .new_decode_state(
            CacheConfig::new(1, sequence + 1).expect("packed smoke cache geometry"),
            1,
            &device,
        )
        .expect("packed Qwen3-MoE KV state must initialize");
    let prefill_started = Instant::now();
    let cached = model
        .prefill(input, &mut state, &diagnostics)
        .unwrap_or_else(|error| panic!("run cached Qwen3-MoE {label} prefill: {error}"));
    assert_eq!(cached.logits.dims(), [1, 1, 151_936]);
    assert_eq!(cached.hidden_states.len(), 50);
    let cached_logits = metal_values("quantized Qwen3-MoE cached logits", &cached.logits);
    let cached_trace = metal_values(
        "quantized Qwen3-MoE cached trace",
        &Tensor::cat(cached.hidden_states.clone(), 0),
    );
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    assert!(cached_logits.iter().all(|value| value.is_finite()));
    assert!(cached_trace.iter().all(|value| value.is_finite()));
    assert_eq!(state.lengths(), &[sequence]);

    let cached_cosine = cosine_similarity(&stateless_logits, &cached_logits);
    assert!(
        cached_cosine >= 0.999,
        "Qwen3-MoE {label} cached/stateless cosine {cached_cosine} is below the gate"
    );
    let trace_width = sequence * 2_048;
    let hidden_cosines = stateless_trace
        .chunks_exact(trace_width)
        .zip(cached_trace.chunks_exact(trace_width))
        .map(|(stateless, cached)| cosine_similarity(stateless, cached))
        .collect::<Vec<_>>();
    assert_eq!(hidden_cosines.len(), 50);
    let minimum_hidden_cosine = hidden_cosines.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        minimum_hidden_cosine >= 0.999,
        "Qwen3-MoE {label} minimum hidden cached/stateless cosine was {minimum_hidden_cosine}"
    );
    let stateless_top = stateless
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("quantized Metal top-1 IDs must use i32 storage");
    let cached_top = cached
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("cached quantized Metal top-1 IDs must use i32 storage");
    assert_eq!(cached_top, stateless_top);
    let next_token = i64::from(stateless_top[0]);

    let decode_input = DecodeInput {
        token_ids: Tensor::<Metal, 2, Int>::from_data(
            TensorData::new(vec![next_token], [1, 1]),
            &device,
        ),
        token_mask: None,
        position_ids: None,
    };
    let decode_started = Instant::now();
    #[cfg(feature = "kernel-tracing")]
    let (decoded, decoded_logits) = {
        // Calibrate the exact installed scheme using one small expert matrix.
        // This materializes only that matrix, never the 30B parameter set.
        let calibration_weight = model.model.layers[0]
            .sparse_mlp
            .as_ref()
            .expect("the pinned first layer must be sparse")
            .routed[0]
            .gate_proj
            .weight
            .val();
        let (_, calibration_trace) =
            capture_kernel_launches(|| calibration_weight.dequantize().into_data());
        let dequantization_names = calibrated_dequantization_kernel_names(&calibration_trace);
        assert!(
            !dequantization_names.is_empty(),
            "installed {label} weight must calibrate an explicit whole-tensor dequantization kernel"
        );

        let ((decoded, decoded_logits), decode_trace) = capture_kernel_launches(|| {
            let decoded = model
                .decode(decode_input, &mut state, &ForwardOptions::default())
                .unwrap_or_else(|error| panic!("run Qwen3-MoE {label} cached decode: {error}"));
            assert_eq!(decoded.logits.dims(), [1, 1, 151_936]);
            let decoded_logits =
                metal_values("quantized Qwen3-MoE decoded logits", &decoded.logits);
            (decoded, decoded_logits)
        });
        let packed = packed_matmul_launches(&decode_trace);
        assert_eq!(
            packed.len(),
            PACKED_DECODE_MATMUL_LAUNCHES,
            "batch-one Qwen3-MoE decode must execute exactly the architecture-declared packed projections"
        );
        let marker = match scheme {
            SymmetricQuantization::Q8 => "Q8S",
            SymmetricQuantization::Q4 => "Q4S",
        };
        assert!(packed.iter().all(|launch| {
            launch.name.contains("matmul")
                && launch.id.contains(marker)
                && compact_kernel_id(launch).contains("PackedU32(1")
        }));
        let dequantizations =
            whole_weight_dequantization_launches(&decode_trace, &dequantization_names);
        assert!(
            dequantizations.is_empty(),
            "Qwen3-MoE {label} decode launched whole-weight dequantization: {dequantizations:#?}"
        );
        eprintln!(
            "Qwen3-MoE {label} packed kernel audit: matmul_launches={}, expected={}, whole_weight_dequantizations=0",
            packed.len(),
            PACKED_DECODE_MATMUL_LAUNCHES
        );
        (decoded, decoded_logits)
    };
    #[cfg(not(feature = "kernel-tracing"))]
    let (decoded, decoded_logits) = {
        let decoded = model
            .decode(decode_input, &mut state, &ForwardOptions::default())
            .unwrap_or_else(|error| panic!("run Qwen3-MoE {label} cached decode: {error}"));
        assert_eq!(decoded.logits.dims(), [1, 1, 151_936]);
        let decoded_logits = metal_values("quantized Qwen3-MoE decoded logits", &decoded.logits);
        (decoded, decoded_logits)
    };
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    assert!(decoded_logits.iter().all(|value| value.is_finite()));
    assert_eq!(state.lengths(), &[sequence + 1]);
    let decoded_top = decoded
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("decoded quantized Metal top-1 IDs must use i32 storage")[0];

    eprintln!(
        "Qwen3-MoE {label} Metal: burnpack={}, artifact_sha256={artifact_sha256}, \
         packed_payload_bytes={packed_payload_bytes}, prompt_tokens={sequence}, \
         first_top1={next_token}, decoded_top1={decoded_top}, \
         cached_cosine={cached_cosine:.9}, min_hidden_cosine={minimum_hidden_cosine:.9}, \
         stateless_logits_fnv1a64={:016x}, cached_logits_fnv1a64={:016x}, \
         stateless_trace_fnv1a64={:016x}, cached_trace_fnv1a64={:016x}, \
         decoded_logits_fnv1a64={:016x}, load_seconds={load_seconds:.3}, \
         source_reads={}, max_open_shards={}, stateless_seconds={stateless_seconds:.3}, \
         prefill_seconds={prefill_seconds:.3}, decode_seconds={decode_seconds:.3}",
        burnpack.display(),
        fnv1a64_f32(&stateless_logits),
        fnv1a64_f32(&cached_logits),
        fnv1a64_f32(&stateless_trace),
        fnv1a64_f32(&cached_trace),
        fnv1a64_f32(&decoded_logits),
        execution.source_reads,
        execution.source.max_open_shards,
    );
}

/// Resolves a canonical, non-symlink Burnpack path on the checkpoint's volume.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn explicit_external_burnpack_path(
    checkpoint_root: &Path,
    environment: &str,
    must_exist: bool,
) -> PathBuf {
    let supplied = PathBuf::from(std::env::var_os(environment).unwrap_or_else(|| {
        panic!("set {environment} to an exact .bpk path on the checkpoint's external volume")
    }));
    validate_external_burnpack_path(
        checkpoint_root,
        &supplied,
        must_exist,
        Path::new("/Volumes"),
    )
}

/// Applies the external-volume path policy with an injectable mount root for tests.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn validate_external_burnpack_path(
    checkpoint_root: &Path,
    supplied: &Path,
    must_exist: bool,
    volumes_root: &Path,
) -> PathBuf {
    assert!(supplied.is_absolute(), "Burnpack path must be absolute");
    assert_eq!(
        supplied
            .extension()
            .and_then(|extension| extension.to_str()),
        Some("bpk"),
        "Burnpack path must end in .bpk"
    );
    let file_name = supplied
        .file_name()
        .expect("the explicit Burnpack path must contain a file name");
    let supplied_parent = supplied
        .parent()
        .expect("the explicit Burnpack path must have a parent directory");
    let parent = supplied_parent.canonicalize().unwrap_or_else(|error| {
        panic!(
            "canonicalize pre-created Burnpack parent {}: {error}",
            supplied_parent.display()
        )
    });
    let normalized = parent.join(file_name);
    assert_eq!(
        supplied, normalized,
        "Burnpack path must use its canonical parent without traversal or symlink components"
    );

    let checkpoint_root = checkpoint_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize checkpoint root: {error}"));
    let volumes_root = volumes_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize volume root: {error}"));
    let relative = checkpoint_root
        .strip_prefix(&volumes_root)
        .unwrap_or_else(|_| {
            panic!(
                "checkpoint root {} is not below volume root {}",
                checkpoint_root.display(),
                volumes_root.display()
            )
        });
    let volume_name = relative
        .components()
        .next()
        .expect("external checkpoint path must identify a volume");
    let volume_root = volumes_root.join(volume_name.as_os_str());
    assert!(
        normalized.starts_with(&volume_root),
        "Burnpack path must remain below the checkpoint volume {}",
        volume_root.display()
    );

    match fs::symlink_metadata(&normalized) {
        Ok(metadata) if must_exist => {
            assert!(
                !metadata.file_type().is_symlink(),
                "Burnpack must not be a symlink"
            );
            assert!(metadata.is_file(), "Burnpack must be a regular file");
        }
        Ok(_) => panic!(
            "conversion is create-only and refuses existing output {}",
            normalized.display()
        ),
        Err(error) if must_exist => {
            panic!(
                "required Burnpack {} is unavailable: {error}",
                normalized.display()
            )
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!(
            "inspect Burnpack destination {}: {error}",
            normalized.display()
        ),
    }
    normalized
}

/// Constructs an isolated two-volume tree for external-path policy tests.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn external_path_fixture() -> (tempfile::TempDir, PathBuf, PathBuf, PathBuf) {
    let directory = tempfile::tempdir().expect("create path-policy fixture");
    let root = directory
        .path()
        .canonicalize()
        .expect("canonicalize path-policy fixture");
    let volumes = root.join("Volumes");
    let first = volumes.join("weights");
    let second = volumes.join("other");
    let checkpoint = first.join("models/qwen");
    let output_parent = first.join("quantized");
    fs::create_dir_all(&checkpoint).expect("create checkpoint directory");
    fs::create_dir_all(&output_parent).expect("create Burnpack directory");
    fs::create_dir_all(&second).expect("create second volume");
    (directory, volumes, checkpoint, output_parent)
}

/// Accepts canonical create-only and existing paths on the checkpoint volume.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
fn external_burnpack_path_accepts_only_the_selected_mode() {
    let (_directory, volumes, checkpoint, output_parent) = external_path_fixture();
    let output = output_parent.join("model-q8.bpk");
    assert_eq!(
        validate_external_burnpack_path(&checkpoint, &output, false, &volumes),
        output
    );
    File::create(&output).expect("create existing Burnpack fixture");
    assert_eq!(
        validate_external_burnpack_path(&checkpoint, &output, true, &volumes),
        output
    );
    assert!(
        std::panic::catch_unwind(|| {
            validate_external_burnpack_path(&checkpoint, &output, false, &volumes)
        })
        .is_err()
    );
    let directory_target = output_parent.join("directory.bpk");
    fs::create_dir(&directory_target).expect("create directory target fixture");
    assert!(
        std::panic::catch_unwind(|| {
            validate_external_burnpack_path(&checkpoint, &directory_target, true, &volumes)
        })
        .is_err()
    );
}

/// Rejects lexical escapes, another volume, wrong extensions, and relative paths.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
fn external_burnpack_path_rejects_unsafe_locations() {
    let (_directory, volumes, checkpoint, output_parent) = external_path_fixture();
    let candidates = [
        output_parent.join("../quantized/model.bpk"),
        volumes.join("other/model.bpk"),
        output_parent.join("model.bin"),
        PathBuf::from("model.bpk"),
    ];
    for candidate in candidates {
        assert!(
            std::panic::catch_unwind(|| {
                validate_external_burnpack_path(&checkpoint, &candidate, false, &volumes)
            })
            .is_err(),
            "unsafe path unexpectedly passed: {}",
            candidate.display()
        );
    }
}

/// Rejects a symlinked parent and a symlink masquerading as an existing Burnpack.
#[cfg(all(feature = "metal", not(feature = "rocm"), unix))]
#[test]
fn external_burnpack_path_rejects_symlink_components_and_targets() {
    use std::os::unix::fs::symlink;

    let (_directory, volumes, checkpoint, output_parent) = external_path_fixture();
    let real = output_parent.join("real.bpk");
    File::create(&real).expect("create real Burnpack fixture");
    let target_link = output_parent.join("target-link.bpk");
    symlink(&real, &target_link).expect("create target symlink");
    assert!(
        std::panic::catch_unwind(|| {
            validate_external_burnpack_path(&checkpoint, &target_link, true, &volumes)
        })
        .is_err()
    );

    let linked_parent = volumes.join("weights/linked-quantized");
    symlink(&output_parent, &linked_parent).expect("create parent symlink");
    let through_parent = linked_parent.join("new.bpk");
    assert!(
        std::panic::catch_unwind(|| {
            validate_external_burnpack_path(&checkpoint, &through_parent, false, &volumes)
        })
        .is_err()
    );
}

/// Returns the stable precision label used by artifacts and reports.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
const fn quantization_label(scheme: SymmetricQuantization) -> &'static str {
    match scheme {
        SymmetricQuantization::Q8 => "q8",
        SymmetricQuantization::Q4 => "q4",
    }
}

/// Locks the independent payload-accounting arithmetic used by the large gates.
#[test]
fn quantized_payload_accounting_is_stable() {
    assert_eq!(
        packed_tensor_payload_bytes(64, SymmetricQuantization::Q8, 64),
        68
    );
    assert_eq!(
        packed_tensor_payload_bytes(64, SymmetricQuantization::Q4, 64),
        36
    );
    assert_eq!(
        packed_tensor_payload_bytes(65, SymmetricQuantization::Q4, 64),
        41
    );
}

/// Conversion and loading sign the same M3-compatible F16 target policy.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
fn quantized_acceptance_uses_one_f16_load_policy() {
    assert_eq!(
        metal_f16_load_options().dtype_policy,
        DTypePolicy::Cast(TensorDType::F16)
    );
}
