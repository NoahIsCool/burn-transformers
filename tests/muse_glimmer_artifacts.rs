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
//! cargo test --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std \
//!   --test muse_glimmer_artifacts \
//!   validates_pinned_muse_glimmer_full_shard_payload_hashes \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! The optional Metal gate audits strict streaming and normalized generation
//! metadata, but deliberately does not claim reference-logit acceptance.

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
use burn::{backend::Metal, tensor::backend::Backend};
#[cfg(all(feature = "metal", feature = "tokenizers"))]
use transformers::{AutoCausalLm, AutoLoadOptions, SamplingStrategy};

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
#[cfg(feature = "tokenizers")]
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
#[cfg(feature = "tokenizers")]
const PHYSICAL_TENSOR_COUNT: usize = 1_436;

/// Number of language-tower and vocabulary-head tensors consumed by Burn.
const CONSUMED_TENSOR_COUNT: usize = 627;

/// Number of consumed tensors renamed out of the multimodal namespace.
const RENAMED_TENSOR_COUNT: usize = 626;

/// Number of narrowly allowlisted visual tensors left uninstantiated.
#[cfg(feature = "tokenizers")]
const VISUAL_TENSOR_COUNT: usize = 809;

/// BF16 bytes across the complete multimodal tensor inventory.
#[cfg(feature = "tokenizers")]
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
#[cfg(feature = "tokenizers")]
const SMOKE_PROMPT: &str = "Hello world, cached decoding.";

/// Exact token IDs without tokenizer-injected special tokens.
#[cfg(feature = "tokenizers")]
const SMOKE_TOKEN_IDS: [u32; 6] = [19_873, 3_817, 24, 41_346, 63_623, 26];

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
