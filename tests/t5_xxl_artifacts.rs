//! Opt-in acceptance gates for the immutable official FLAN-T5-XXL checkpoint.
//!
//! These tests never contact a model hub. Point `TRANSFORMERS_FLAN_T5_XXL_DIR`
//! at a complete local copy of `google/flan-t5-xxl` revision
//! `ae7c9136adc7555eeccc78cdd960dfd60fb346ce`, then run the metadata gate with:
//!
//! ```text
//! cargo test --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,tokenizers \
//!   --test t5_xxl_artifacts \
//!   validates_pinned_flan_t5_xxl_config_headers_plan_memory_and_tokenizer \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! Run `validates_pinned_flan_t5_xxl_shard_payload_hashes` separately when the
//! complete 45 GB payload identity, rather than fast header identity, is needed.
//!
//! The official index contains an important storage-alias edge case. It maps
//! `encoder.embed_tokens.weight` to shard one, but that key is absent from the
//! physical header because its bytes are deduplicated under `shared.weight`.
//! The loader accepts this only through T5's exact same-shard alias declaration.

#![cfg(feature = "tokenizers")]

use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use transformers::{
    EncodeOptions, LocalTokenizer,
    checkpoint::{
        ArtifactSource, CheckpointEquivalenceRecord, CheckpointManifest, DTypePolicy,
        IndexAliasSizeAccounting, IndexTensorAlias, LoadOptions, LoadPlan, PlannedTransform,
        ResolvedIndexTensorAlias, TensorDType, TensorOperation, TensorTransform,
    },
    models::{
        config::ModelConfig,
        t5::{T5Config, T5FeedForwardProjection},
    },
};

/// Immutable upstream repository certified by this gate.
const REPOSITORY: &str = "google/flan-t5-xxl";

/// Immutable upstream Git revision certified by this gate.
const REVISION: &str = "ae7c9136adc7555eeccc78cdd960dfd60fb346ce";

/// Stable instruction prompt used to pin local tokenization.
const SMOKE_PROMPT: &str = "translate English to German: The house is wonderful.";

/// Exact SentencePiece IDs, including T5's terminal EOS token.
const SMOKE_TOKEN_IDS: [u32; 11] = [13_959, 1_566, 12, 2_968, 10, 37, 629, 19, 1_627, 5, 1];

/// Exact byte lengths and SHA-256 identities of all consumed control artifacts.
const CONTROL_FILES: [(&str, u64, &str); 6] = [
    (
        "config.json",
        674,
        "dd923bd687781f336e7845b7e011b7bbeca5fdde5c27f8b06071540340707dbd",
    ),
    (
        "generation_config.json",
        147,
        "f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a",
    ),
    (
        "model.safetensors.index.json",
        53_032,
        "bab7fd6107b34412f8575908370f29f44f433617a93d557e57b299d6ee688efe",
    ),
    (
        "tokenizer.json",
        2_424_064,
        "fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a",
    ),
    (
        "tokenizer_config.json",
        2_539,
        "ef30e0628a2d067c5d9d02d98cbf9a2040f4a51883e5d16e1809eea548db0eff",
    ),
    (
        "special_tokens_map.json",
        2_201,
        "5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f",
    ),
];

/// Exact physical weight filenames, byte lengths, and LFS SHA-256 identities.
const SHARDS: [(&str, u64, &str); 5] = [
    (
        "model-00001-of-00005.safetensors",
        9_452_262_278,
        "2399615e60e983d1e647248b15bfad3c0ccd9473a0e387dbe03284fcaacf21c2",
    ),
    (
        "model-00002-of-00005.safetensors",
        9_597_007_440,
        "4d2459d60251e60af6314ae4785fb0bea32febb1989434c816de84c0fdae2d88",
    ),
    (
        "model-00003-of-00005.safetensors",
        9_955_647_022,
        "60a2c0b6f284a55df23ee68ce3065631368d7a6a35e5443b5cd98151f145fd67",
    ),
    (
        "model-00004-of-00005.safetensors",
        9_999_712_776,
        "5f0cbe2b9a51ab64c93d16993e79bf81cc39f8692f89122cc68fa3b824c2b3cf",
    ),
    (
        "model-00005-of-00005.safetensors",
        6_063_154_240,
        "b0001f5f48ce5bf0e7e997c57ce5170ce0c0b0d667ee2a72d01d9aa26f40b7b5",
    ),
];

/// SHA-256 of every physical tensor name, shard, dtype, shape, and data range.
const HEADER_INVENTORY_SHA256: &str =
    "794a9c5bb717021bf491ee6942cda6dab2b223a7bb5f7c2861be01248f590a0d";

/// SHA-256 of every F32 plan target, source, transform, dtype, and target shape.
const F32_PLAN_SHA256: &str = "711cd9f6d278f519d511c47614be1ec1353e2b1657d7d04cbe7c34cb7881365b";

/// Resolves the explicitly selected immutable local artifact directory.
fn pinned_root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("TRANSFORMERS_FLAN_T5_XXL_DIR").unwrap_or_else(|| {
            panic!("set TRANSFORMERS_FLAN_T5_XXL_DIR to {REPOSITORY} revision {REVISION}")
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
    assert!(
        file.metadata()
            .unwrap_or_else(|error| panic!("inspect open {}: {error}", path.display()))
            .is_file(),
        "{} changed away from a regular file while opening",
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

/// Verifies every small artifact's length and content identity.
fn assert_control_files(root: &Path) {
    for (relative_path, expected_bytes, expected_hash) in CONTROL_FILES {
        let path = root.join(relative_path);
        let metadata = fs::symlink_metadata(&path)
            .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()));
        assert!(!metadata.file_type().is_symlink());
        assert!(metadata.is_file());
        assert_eq!(metadata.len(), expected_bytes, "changed {relative_path}");
        assert_eq!(
            sha256_file_hex(&path),
            expected_hash,
            "pinned control file changed: {relative_path}"
        );
    }
}

/// Streams every multi-gigabyte payload to certify the pinned revision's bytes.
fn assert_shard_file_hashes(root: &Path) {
    for (relative_path, expected_bytes, expected_hash) in SHARDS {
        let path = root.join(relative_path);
        assert_eq!(
            fs::metadata(&path)
                .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()))
                .len(),
            expected_bytes,
            "pinned shard length changed: {relative_path}"
        );
        assert_eq!(
            sha256_file_hex(&path),
            expected_hash,
            "pinned shard payload changed: {relative_path}"
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

/// Returns a complete stable spelling for one planned layout operation.
fn planned_transform_name(transform: &PlannedTransform) -> String {
    match transform {
        PlannedTransform::Identity => "identity".into(),
        PlannedTransform::Transpose2d => "transpose_2d".into(),
        PlannedTransform::Slice {
            axis,
            start,
            length,
        } => format!("slice(axis={axis},start={start},length={length})"),
        PlannedTransform::Concatenate { axis } => format!("concatenate(axis={axis})"),
        PlannedTransform::Alias { source_target } => format!("alias({source_target})"),
        transform => panic!("unsupported plan transform in digest: {transform:?}"),
    }
}

/// Hashes the strict mapping, transform, dtype, and shape in canonical plan order.
fn f32_plan_digest(plan: &LoadPlan) -> String {
    let mut canonical = Vec::new();
    for tensor in &plan.tensors {
        canonical.extend_from_slice(tensor.target.as_bytes());
        canonical.push(0);
        for source in &tensor.sources {
            canonical.extend_from_slice(source.name.as_bytes());
            canonical.push(0);
        }
        canonical.extend_from_slice(planned_transform_name(&tensor.transform).as_bytes());
        canonical.push(0);
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

/// Returns the literal physical header-shape histogram for the pinned artifact.
fn physical_shape_counts() -> BTreeMap<Vec<u64>, usize> {
    BTreeMap::from([
        (vec![4_096], 122),
        (vec![32, 64], 2),
        (vec![4_096, 4_096], 288),
        (vec![4_096, 10_240], 48),
        (vec![10_240, 4_096], 96),
        (vec![32_128, 4_096], 3),
    ])
}

/// Returns the literal Burn runtime-shape histogram for the strict load plan.
fn runtime_shape_counts() -> BTreeMap<Vec<u64>, usize> {
    BTreeMap::from([
        (vec![4_096], 122),
        (vec![32, 64], 2),
        (vec![4_096, 4_096], 288),
        (vec![4_096, 10_240], 96),
        (vec![10_240, 4_096], 48),
        (vec![32_128, 4_096], 1),
        (vec![4_096, 32_128], 1),
    ])
}

/// Proves the committed XXL fixture's topology without external artifacts.
#[test]
fn committed_flan_t5_xxl_fixture_has_exact_contract_topology() {
    let config = T5Config::from_json_str(include_str!("fixtures/flan_t5_xxl.json"))
        .expect("the committed XXL fixture must validate");
    assert_eq!(config.d_model, 4_096);
    assert_eq!(config.d_ff, 10_240);
    assert_eq!(config.num_layers, 24);
    assert_eq!(config.decoder_layers(), 24);
    assert_eq!(config.num_heads, 64);
    assert_eq!(config.feed_forward_proj, T5FeedForwardProjection::GatedGelu);
    assert!(!config.tie_word_embeddings);

    let contract = config
        .checkpoint_contract()
        .expect("the XXL fixture must produce a checkpoint contract");
    assert_eq!(contract.expected_tensors().len(), 558);
    assert_eq!(
        contract.index_aliases(),
        [
            IndexTensorAlias::new("encoder.embed_tokens.weight", "shared.weight"),
            IndexTensorAlias::new("decoder.embed_tokens.weight", "shared.weight"),
        ]
    );
    let expected = contract
        .expected_tensors()
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.shape.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let mut sources = BTreeSet::new();
    let mut mappings = Vec::new();
    for operation in contract.mappings().operations() {
        let TensorOperation::Map {
            source,
            target,
            transform,
        } = operation
        else {
            panic!("untied FLAN-T5-XXL must use direct mappings, found {operation:?}")
        };
        assert_eq!(source, target);
        assert!(sources.insert(source.as_str()), "duplicate source {source}");
        let shape = expected
            .get(target.as_str())
            .unwrap_or_else(|| panic!("mapping produced unknown target {target}"));
        mappings.push((target.as_str(), source.as_str(), *transform, *shape));
    }
    assert_eq!(sources.len(), 558);
    mappings.sort_unstable_by_key(|(target, _, _, _)| *target);
    let mut canonical = Vec::new();
    let mut transpose_count = 0_usize;
    for (target, source, transform, shape) in mappings {
        canonical.extend_from_slice(target.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(source.as_bytes());
        canonical.push(0);
        let transform = match transform {
            TensorTransform::Identity => "identity",
            TensorTransform::Transpose2d => {
                transpose_count += 1;
                "transpose_2d"
            }
            transform => panic!("unsupported contract transform: {transform:?}"),
        };
        canonical.extend_from_slice(transform.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(TensorDType::F32.as_str().as_bytes());
        canonical.push(0);
        for dimension in shape {
            canonical.extend_from_slice(dimension.to_string().as_bytes());
            canonical.push(b',');
        }
        canonical.push(b'\n');
    }
    assert_eq!(transpose_count, 433);
    let target_shape_counts = contract.expected_tensors().iter().fold(
        BTreeMap::<Vec<u64>, usize>::new(),
        |mut counts, tensor| {
            *counts.entry(tensor.shape.clone()).or_default() += 1;
            counts
        },
    );
    assert_eq!(target_shape_counts, runtime_shape_counts());
    let digest = sha256_hex(&canonical);
    eprintln!("FLAN-T5-XXL fixture plan digest: {digest}");
    assert_eq!(digest, F32_PLAN_SHA256);
}

/// Certifies every official shard payload independently of header metadata.
#[test]
#[ignore = "streams and hashes all 45 GB of pinned FLAN-T5-XXL weights"]
fn validates_pinned_flan_t5_xxl_shard_payload_hashes() {
    let started = Instant::now();
    let supplied_root = pinned_root();
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert_shard_file_hashes(&root);
    eprintln!(
        "FLAN-T5-XXL payload gate: repository={REPOSITORY}, revision={REVISION}, \
         shard_bytes=45067783756, elapsed_seconds={:.3}",
        started.elapsed().as_secs_f64()
    );
}

/// Certifies exact XXL config, headers, mapping, memory bounds, and tokenizer state.
#[test]
#[ignore = "requires google/flan-t5-xxl revision ae7c9136adc7555eeccc78cdd960dfd60fb346ce"]
fn validates_pinned_flan_t5_xxl_config_headers_plan_memory_and_tokenizer() {
    let started = Instant::now();
    let supplied_root = pinned_root();
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert!(root.is_dir(), "pinned artifact root must be a directory");
    assert_control_files(&root);

    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let observed = T5Config::from_json_str(&config_json)
        .expect("the pinned FLAN-T5-XXL configuration must validate");
    let expected = T5Config::from_json_str(include_str!("fixtures/flan_t5_xxl.json"))
        .expect("the committed FLAN-T5-XXL fixture must validate");
    assert_eq!(observed, expected);
    let raw: serde_json::Value =
        serde_json::from_str(&config_json).expect("the pinned config must be valid JSON");
    assert_eq!(
        raw["architectures"],
        serde_json::json!(["T5ForConditionalGeneration"])
    );
    assert_eq!(raw["torch_dtype"], "float32");
    assert_eq!(raw["transformers_version"], "4.24.0.dev0");

    let planned = observed
        .plan_checkpoint(
            &root,
            &LoadOptions {
                dtype_policy: DTypePolicy::Preserve,
                ..LoadOptions::default()
            },
        )
        .expect("the official deduplicated embedding alias must plan strictly");
    let manifest = planned.manifest();
    match &manifest.source {
        ArtifactSource::Index {
            relative_path,
            declared_total_size,
        } => {
            assert_eq!(relative_path, "model.safetensors.index.json");
            assert_eq!(*declared_total_size, Some(45_594_099_712));
        }
        source => panic!("expected the official shard index, found {source:?}"),
    }
    assert_eq!(manifest.root, root);
    assert_eq!(manifest.shards.len(), 5);
    assert_eq!(manifest.tensors.len(), 559);
    assert_eq!(manifest.tensor_bytes, 45_067_714_560);
    assert_eq!(manifest.indexed_tensor_bytes, 45_594_099_712);
    assert_eq!(
        manifest.index_alias_size_accounting,
        Some(IndexAliasSizeAccounting::Logical)
    );
    assert_eq!(manifest.artifact_bytes, 45_067_783_756);
    assert_eq!(
        manifest.index_aliases,
        [ResolvedIndexTensorAlias::new(
            "encoder.embed_tokens.weight",
            "shared.weight",
            "model-00001-of-00005.safetensors"
        )]
    );
    assert_eq!(header_inventory_digest(manifest), HEADER_INVENTORY_SHA256);
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
            .all(|tensor| tensor.dtype == TensorDType::F32)
    );
    let shape_counts =
        manifest
            .tensors
            .values()
            .fold(BTreeMap::<Vec<u64>, usize>::new(), |mut counts, tensor| {
                *counts.entry(tensor.shape.clone()).or_default() += 1;
                counts
            });
    assert_eq!(shape_counts, physical_shape_counts());

    let plan = planned.load_plan();
    assert_eq!(plan.parameter_count(), 558);
    assert_eq!(plan.tensors.len(), 558);
    assert!(plan.aliases.is_empty());
    assert!(plan.report.is_strictly_valid());
    assert!(plan.report.allowlisted_ignored.is_empty());
    assert_eq!(plan.report.checkpoint_aliases, manifest.index_aliases);
    assert_eq!(
        plan.report.equivalence_planned,
        [CheckpointEquivalenceRecord::new(
            "decoder.embed_tokens.weight",
            "shared.weight"
        )]
    );
    assert!(plan.report.equivalence_verified.is_empty());
    assert!(plan.report.allowlisted_missing.is_empty());
    assert_eq!(plan.report.transformed.len(), 433);
    assert!(plan.report.transformed.iter().all(|record| {
        record.operation == "transpose_2d" && record.sources.as_slice() == [record.target.as_str()]
    }));
    assert!(plan.report.dtype_converted.is_empty());
    assert!(plan.tensors.iter().all(|tensor| {
        tensor.sources.len() == 1
            && tensor.sources[0].name == tensor.target
            && tensor.dtype == TensorDType::F32
    }));
    assert_eq!(
        plan.tensors
            .iter()
            .filter(|tensor| matches!(tensor.transform, PlannedTransform::Transpose2d))
            .count(),
        433
    );
    assert_eq!(
        plan.tensors
            .iter()
            .filter(|tensor| matches!(tensor.transform, PlannedTransform::Identity))
            .count(),
        125
    );
    let target_shape_counts =
        plan.tensors
            .iter()
            .fold(BTreeMap::<Vec<u64>, usize>::new(), |mut counts, tensor| {
                *counts.entry(tensor.shape.clone()).or_default() += 1;
                counts
            });
    assert_eq!(target_shape_counts, runtime_shape_counts());
    assert_eq!(f32_plan_digest(plan), F32_PLAN_SHA256);
    assert_eq!(plan.memory.artifact_bytes, 45_067_783_756);
    assert_eq!(plan.memory.checkpoint_tensor_bytes, 45_067_714_560);
    assert_eq!(plan.memory.target_parameter_bytes, 44_541_329_408);
    assert_eq!(plan.memory.logical_model_bytes, 44_541_329_408);
    assert_eq!(plan.memory.largest_shard_bytes, 9_999_712_776);
    assert_eq!(plan.memory.largest_source_tensor_bytes, 526_385_152);
    assert_eq!(plan.memory.largest_target_tensor_bytes, 526_385_152);
    assert_eq!(plan.memory.largest_equivalence_pair_bytes, 1_052_770_304);
    assert_eq!(plan.memory.peak_host_staging_bytes, 10_526_097_928);
    assert_eq!(plan.report.memory, Some(plan.memory));

    let tokenizer = LocalTokenizer::from_directory(&root)
        .expect("the pinned tokenizer artifacts must load without network access");
    assert_eq!(tokenizer.root(), root);
    assert_eq!(
        tokenizer.metadata().tokenizer_class.as_deref(),
        Some("T5Tokenizer")
    );
    assert_eq!(tokenizer.metadata().model_max_length, Some(512));
    assert_eq!(tokenizer.metadata().vocabulary_size, 32_100);
    assert_eq!(
        tokenizer
            .metadata()
            .special_tokens
            .eos
            .as_ref()
            .map(|token| token.id),
        Some(1)
    );
    assert_eq!(
        tokenizer
            .metadata()
            .special_tokens
            .pad
            .as_ref()
            .map(|token| token.id),
        Some(0)
    );
    assert_eq!(
        tokenizer
            .metadata()
            .special_tokens
            .unk
            .as_ref()
            .map(|token| token.id),
        Some(2)
    );
    let encoding = tokenizer
        .encode(SMOKE_PROMPT, &EncodeOptions::default())
        .expect("the committed instruction prompt must encode");
    let prompt_ids = &encoding.token_ids()[0];
    assert_eq!(prompt_ids.as_slice(), SMOKE_TOKEN_IDS);
    assert!(encoding.valid_tokens()[0].iter().all(|valid| *valid));
    assert_eq!(
        tokenizer
            .decode(prompt_ids, true)
            .expect("the pinned prompt token IDs must decode"),
        SMOKE_PROMPT
    );

    eprintln!(
        "FLAN-T5-XXL header gate: repository={REPOSITORY}, revision={REVISION}, \
         shards=5, physical_tensors=559, parameters=558, \
         header_inventory_sha256={HEADER_INVENTORY_SHA256}, \
         f32_plan_sha256={F32_PLAN_SHA256}, target_bytes={}, \
         peak_host_staging_bytes={}, prompt_ids={prompt_ids:?}, elapsed_seconds={:.3}",
        plan.memory.target_parameter_bytes,
        plan.memory.peak_host_staging_bytes,
        started.elapsed().as_secs_f64(),
    );
}
