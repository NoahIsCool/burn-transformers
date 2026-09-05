//! Opt-in acceptance gates for the immutable DeepSeek-V2-Lite checkpoint.
//!
//! These tests never contact a model hub. Point `TRANSFORMERS_DEEPSEEK_V2_DIR`
//! at a complete local copy of `deepseek-ai/DeepSeek-V2-Lite` revision
//! `604d5664dddd88a0433dbae533b7fe9472482de0`, then run the metadata gate with:
//!
//! ```text
//! cargo test --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,tokenizers \
//!   --test deepseek_v2_artifacts \
//!   validates_pinned_deepseek_v2_lite_config_headers_plan_memory_and_tokenizer \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! Compile the Metal gate without executing the 15.7B-parameter load with:
//!
//! ```text
//! cargo test --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,metal,tokenizers,chat-templates \
//!   --test deepseek_v2_artifacts --no-run
//! ```

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
        ArtifactSource, CheckpointManifest, DTypePolicy, LoadOptions, LoadPlan, PlannedTransform,
        TensorDType, TensorOperation, TensorTransform,
    },
    models::{
        config::ModelConfig,
        deepseek_v2::{DeepseekV2CheckpointOptions, DeepseekV2Config},
    },
};

#[cfg(feature = "metal")]
use burn::{
    backend::Metal,
    nn::Linear,
    tensor::{DType, Int, Tensor, TensorData, backend::Backend},
};
#[cfg(feature = "metal")]
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, DecodeInput, DecodeState, ForwardOptions,
    cache::CacheConfig,
    models::{
        AutoCausalDecodeState,
        deepseek_v2::{DeepseekV2DecodeState, DeepseekV2ForCausalLm, MlaExecutionPath},
    },
};

/// Immutable upstream repository certified by both gates.
const REPOSITORY: &str = "deepseek-ai/DeepSeek-V2-Lite";

/// Immutable upstream Git revision certified by both gates.
const REVISION: &str = "604d5664dddd88a0433dbae533b7fe9472482de0";

/// Short deterministic prompt shared by stateless, prefill, and decode checks.
const SMOKE_PROMPT: &str = "Hello world, cached decoding.";

/// Exact token IDs produced without artificial special tokens.
const SMOKE_TOKEN_IDS: [u32; 6] = [17_464, 1_843, 11, 25_890, 39_671, 13];

/// Exact byte lengths and SHA-256 identities of all consumed control artifacts.
const CONTROL_FILES: [(&str, u64, &str); 5] = [
    (
        "config.json",
        1_522,
        "f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d",
    ),
    (
        "generation_config.json",
        181,
        "63ec07f6cb36bd8359b44f671123518dd6fbf55007e0c74aa96c296a75def8b5",
    ),
    (
        "model.safetensors.index.json",
        479_924,
        "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf",
    ),
    (
        "tokenizer.json",
        4_607_451,
        "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669",
    ),
    (
        "tokenizer_config.json",
        1_279,
        "31181eaf79394ea26728d95ecb54fe7c8413e6f56085dbabc8b0818134380ec8",
    ),
];

/// Exact physical weight files, lengths, and LFS SHA-256 identities.
const SHARDS: [(&str, u64, &str); 4] = [
    (
        "model-00001-of-000004.safetensors",
        8_594_887_408,
        "0d7e9f39bde40111a4c0f390b87497dce4565cf578d916395e6b2c7851f1e8da",
    ),
    (
        "model-00002-of-000004.safetensors",
        8_591_757_448,
        "e0656832b0d594b4a64cad40ff8465231de6ed44c854f74f6b217797660aa4bb",
    ),
    (
        "model-00003-of-000004.safetensors",
        8_590_718_520,
        "843ec689624f3a520526e040f0326c4dc9865e8172942ca98a084fe136fdb21a",
    ),
    (
        "model-00004-of-000004.safetensors",
        5_636_263_200,
        "cfb51658f67cedfbbc4d62ad14187830ceec8ee82c788c5f718feea98905ef31",
    ),
];

/// SHA-256 of every F16 plan target, source, dtype, and target shape.
const F16_PLAN_SHA256: &str = "af3bb522e5a1d498cf1979978ab553d3ce53db088e74447997b33a914466ab2b";

/// SHA-256 of every tensor's physical name, shard, dtype, shape, and data range.
const HEADER_INVENTORY_SHA256: &str =
    "08f26cb4a2c6c5899e65ba3976a45a537b8811e312319f4945aa3ac95fe33e60";

/// Resolves the explicitly selected immutable local artifact directory.
fn pinned_root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("TRANSFORMERS_DEEPSEEK_V2_DIR").unwrap_or_else(|| {
            panic!("set TRANSFORMERS_DEEPSEEK_V2_DIR to {REPOSITORY} revision {REVISION}")
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

/// Verifies every multi-gigabyte payload before accepting real-model results.
#[cfg(feature = "metal")]
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

/// Returns the literal source-shape histogram for the pinned Lite topology.
fn expected_shape_counts() -> BTreeMap<Vec<u64>, usize> {
    BTreeMap::from([
        (vec![64, 2_048], 26),
        (vec![512], 27),
        (vec![576, 2_048], 27),
        (vec![1_408, 2_048], 3_328),
        (vec![2_048], 55),
        (vec![2_048, 1_408], 1_664),
        (vec![2_048, 2_048], 27),
        (vec![2_048, 2_816], 26),
        (vec![2_048, 10_944], 1),
        (vec![2_816, 2_048], 52),
        (vec![3_072, 2_048], 27),
        (vec![4_096, 512], 27),
        (vec![10_944, 2_048], 2),
        (vec![102_400, 2_048], 2),
    ])
}

/// Proves the committed Lite fixture independently of any external artifact.
#[test]
fn committed_lite_fixture_has_exact_contract_topology() {
    let config = DeepseekV2Config::from_json_str(include_str!("fixtures/deepseek_v2_lite.json"))
        .expect("the committed Lite fixture must validate");
    assert_eq!(config, DeepseekV2Config::default());
    let contract = config
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .expect("the Lite fixture must produce a checkpoint contract");
    assert_eq!(contract.expected_tensors().len(), 5_291);
    let expected = contract
        .expected_tensors()
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.shape.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let mut mappings = Vec::new();
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
                let shape = expected
                    .get(target.as_str())
                    .unwrap_or_else(|| panic!("mapping produced unknown target {target}"));
                renames += usize::from(source != target);
                mappings.push((target.as_str(), source.as_str(), *shape));
            }
            operation => panic!("Lite must use only one-to-one mappings, found {operation:?}"),
        }
    }
    assert_eq!(mappings.len(), 5_291);
    assert_eq!(sources.len(), 5_291);
    assert_eq!(renames, 5_099);
    mappings.sort_unstable_by_key(|(target, _, _)| *target);
    let mut canonical = Vec::new();
    for (target, source, shape) in mappings {
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
    let shape_counts = contract.expected_tensors().iter().fold(
        BTreeMap::<Vec<u64>, usize>::new(),
        |mut counts, tensor| {
            *counts.entry(tensor.shape.clone()).or_default() += 1;
            counts
        },
    );
    assert_eq!(shape_counts, expected_shape_counts());
}

/// Certifies exact Lite config, headers, mapping, memory bounds, and tokenizer state.
#[test]
#[ignore = "requires deepseek-ai/DeepSeek-V2-Lite revision 604d5664dddd88a0433dbae533b7fe9472482de0"]
fn validates_pinned_deepseek_v2_lite_config_headers_plan_memory_and_tokenizer() {
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
    let observed = DeepseekV2Config::from_json_str(&config_json)
        .expect("the pinned DeepSeek-V2-Lite configuration must validate");
    let expected = DeepseekV2Config::from_json_str(include_str!("fixtures/deepseek_v2_lite.json"))
        .expect("the committed DeepSeek-V2-Lite fixture must validate");
    assert_eq!(observed, expected);
    assert_eq!(observed, DeepseekV2Config::default());
    let raw: serde_json::Value =
        serde_json::from_str(&config_json).expect("the pinned config must be valid JSON");
    assert_eq!(
        raw["architectures"],
        serde_json::json!(["DeepseekV2ForCausalLM"])
    );
    assert_eq!(raw["torch_dtype"], "bfloat16");
    assert_eq!(raw["transformers_version"], "4.33.1");
    assert_eq!(observed.model_type, "deepseek_v2");
    assert_eq!(observed.num_hidden_layers, 27);
    assert_eq!(observed.num_attention_heads, 16);
    assert_eq!(observed.kv_lora_rank, 512);
    assert_eq!(observed.q_lora_rank, None);
    assert_eq!(observed.n_routed_experts, 64);
    assert_eq!(observed.num_experts_per_tok, 6);
    assert_eq!(observed.n_shared_experts, 2);
    assert_eq!(observed.first_k_dense_replace, 1);
    assert!(!observed.norm_topk_prob);
    assert!(observed.use_mla);

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
            assert_eq!(*declared_total_size, Some(31_412_968_448));
        }
        ArtifactSource::Single { relative_path } => {
            panic!("expected the pinned shard index, found {relative_path}")
        }
        _ => panic!("expected the standard indexed SafeTensors artifact"),
    }
    assert_eq!(manifest.shards.len(), 4);
    assert_eq!(manifest.tensors.len(), 5_291);
    assert_eq!(manifest.tensor_bytes, 31_412_968_448);
    assert_eq!(manifest.artifact_bytes, 31_413_626_576);
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
    let mut shape_counts = BTreeMap::<Vec<u64>, usize>::new();
    for tensor in manifest.tensors.values() {
        *shape_counts.entry(tensor.shape.clone()).or_default() += 1;
    }
    assert_eq!(shape_counts, expected_shape_counts());

    let contract = observed
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .expect("the pinned topology must produce its exact parameter contract");
    assert_eq!(contract.expected_tensors().len(), 5_291);
    let expected_contract_shapes = contract.expected_tensors().iter().fold(
        BTreeMap::<Vec<u64>, usize>::new(),
        |mut counts, tensor| {
            *counts.entry(tensor.shape.clone()).or_default() += 1;
            counts
        },
    );
    assert_eq!(expected_contract_shapes, expected_shape_counts());
    let plan = contract
        .build_load_plan(&manifest, &options)
        .expect("every pinned tensor must map under strict F16 planning");
    assert_eq!(plan.parameter_count(), 5_291);
    assert_eq!(plan.tensors.len(), 5_291);
    assert!(plan.aliases.is_empty());
    assert!(plan.report.is_strictly_valid());
    assert!(plan.report.allowlisted_ignored.is_empty());
    assert!(plan.report.allowlisted_missing.is_empty());
    assert_eq!(plan.report.transformed.len(), 5_099);
    assert!(
        plan.report
            .transformed
            .iter()
            .all(|record| record.operation == "rename")
    );
    assert_eq!(plan.report.dtype_converted.len(), 5_291);
    assert!(plan.report.dtype_converted.iter().all(|conversion| {
        conversion.source == TensorDType::BF16 && conversion.target_dtype == TensorDType::F16
    }));
    assert!(plan.tensors.iter().all(|tensor| {
        tensor.sources.len() == 1
            && tensor.dtype == TensorDType::F16
            && matches!(tensor.transform, PlannedTransform::Identity)
    }));
    let expected_sources = contract
        .mappings()
        .operations()
        .iter()
        .map(|operation| match operation {
            TensorOperation::Map { source, target, .. } => (target.as_str(), source.as_str()),
            operation => panic!("Lite must use only direct mappings, found {operation:?}"),
        })
        .collect::<BTreeMap<_, _>>();
    for tensor in &plan.tensors {
        assert_eq!(
            tensor.sources[0].name,
            expected_sources[tensor.target.as_str()],
            "planned source diverged for {}",
            tensor.target
        );
        assert_eq!(
            tensor.shape,
            contract
                .expected_tensors()
                .iter()
                .find(|expected| expected.name == tensor.target)
                .expect("every planned target must exist in the contract")
                .shape,
            "planned shape diverged for {}",
            tensor.target
        );
    }
    assert_eq!(f16_plan_digest(&plan), F16_PLAN_SHA256);
    let planned_sources = plan
        .tensors
        .iter()
        .map(|tensor| tensor.sources[0].name.as_str())
        .collect::<BTreeSet<_>>();
    let physical_sources = manifest.tensors.keys().map(String::as_str).collect();
    assert_eq!(planned_sources, physical_sources);
    assert_eq!(plan.memory.artifact_bytes, 31_413_626_576);
    assert_eq!(plan.memory.checkpoint_tensor_bytes, 31_412_968_448);
    assert_eq!(plan.memory.target_parameter_bytes, 31_412_968_448);
    assert_eq!(plan.memory.logical_model_bytes, 31_412_968_448);
    assert_eq!(plan.memory.largest_shard_bytes, 8_594_887_408);
    assert_eq!(plan.memory.largest_source_tensor_bytes, 419_430_400);
    assert_eq!(plan.memory.largest_target_tensor_bytes, 419_430_400);
    assert_eq!(plan.memory.peak_host_staging_bytes, 9_014_317_808);
    assert_eq!(plan.report.memory, Some(plan.memory));

    let tokenizer = LocalTokenizer::from_directory(&root)
        .expect("the pinned tokenizer artifacts must load without network access");
    assert_eq!(tokenizer.root(), root);
    assert_eq!(
        tokenizer.metadata().tokenizer_class.as_deref(),
        Some("LlamaTokenizerFast")
    );
    assert_eq!(tokenizer.metadata().model_max_length, Some(16_384));
    assert!(tokenizer.metadata().vocabulary_size <= observed.vocab_size);
    assert!(tokenizer.metadata().vocabulary_size > 100_001);
    assert_eq!(
        tokenizer
            .metadata()
            .special_tokens
            .bos
            .as_ref()
            .map(|token| token.id),
        Some(100_000)
    );
    assert_eq!(
        tokenizer
            .metadata()
            .special_tokens
            .eos
            .as_ref()
            .map(|token| token.id),
        Some(100_001)
    );
    assert_eq!(
        tokenizer
            .metadata()
            .special_tokens
            .pad
            .as_ref()
            .map(|token| token.id),
        Some(100_001)
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
    let prompt_ids = &encoding.token_ids()[0];
    assert_eq!(prompt_ids.as_slice(), SMOKE_TOKEN_IDS);
    assert!(!prompt_ids.is_empty());
    assert!(
        prompt_ids
            .iter()
            .all(|id| (*id as usize) < observed.vocab_size)
    );
    assert!(encoding.valid_tokens()[0].iter().all(|valid| *valid));
    assert_eq!(
        tokenizer
            .decode(prompt_ids, true)
            .expect("the pinned prompt token IDs must decode"),
        SMOKE_PROMPT
    );

    eprintln!(
        "DeepSeek-V2-Lite header gate: repository={REPOSITORY}, revision={REVISION}, \
         shards=4, tensors=5291, tensor_bytes=31412968448, \
         header_inventory_sha256={HEADER_INVENTORY_SHA256}, f16_plan_sha256={F16_PLAN_SHA256}, \
         f16_target_bytes={}, peak_host_staging_bytes={}, prompt_ids={prompt_ids:?}, \
         elapsed_seconds={:.3}",
        plan.memory.target_parameter_bytes,
        plan.memory.peak_host_staging_bytes,
        started.elapsed().as_secs_f64(),
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

/// Computes cosine similarity with F64 accumulation and rejects zero vectors.
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

/// Traverses all 5,291 installed Lite parameters and requires F16 storage.
#[cfg(feature = "metal")]
fn assert_all_parameters_f16(model: &DeepseekV2ForCausalLm<Metal>) {
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

    assert_eq!(model.model.layers.len(), 27);
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
        let attention = &layer.self_attn;
        parameters += assert_linear_f16(
            &format!("model.layers.{layer_index}.self_attn.q_proj"),
            attention
                .q_proj
                .as_ref()
                .expect("Lite uses a direct query projection"),
        );
        assert!(attention.q_a_proj.is_none());
        assert!(attention.q_a_layernorm.is_none());
        assert!(attention.q_b_proj.is_none());
        parameters += assert_linear_f16(
            &format!("model.layers.{layer_index}.self_attn.kv_a_proj_with_mqa"),
            &attention.kv_a_proj_with_mqa,
        );
        assert_eq!(attention.kv_a_layernorm.weight.val().dtype(), DType::F16);
        parameters += 1;
        parameters += assert_linear_f16(
            &format!("model.layers.{layer_index}.self_attn.kv_b_proj"),
            &attention.kv_b_proj,
        );
        parameters += assert_linear_f16(
            &format!("model.layers.{layer_index}.self_attn.o_proj"),
            &attention.o_proj,
        );

        match (&layer.dense_mlp, &layer.sparse_mlp) {
            (Some(dense), None) => {
                assert_eq!(layer_index, 0, "only the first Lite layer is dense");
                for (name, projection) in [
                    ("gate_proj", &dense.gate_proj),
                    ("up_proj", &dense.up_proj),
                    ("down_proj", &dense.down_proj),
                ] {
                    parameters += assert_linear_f16(
                        &format!("model.layers.{layer_index}.dense_mlp.{name}"),
                        projection,
                    );
                }
            }
            (None, Some(sparse)) => {
                assert!(layer_index > 0, "the first Lite layer must remain dense");
                assert_eq!(sparse.routed.routed.len(), 64);
                parameters += assert_linear_f16(
                    &format!("model.layers.{layer_index}.sparse_mlp.routed.router"),
                    &sparse.routed.router,
                );
                for (expert_index, expert) in sparse.routed.routed.iter().enumerate() {
                    for (name, projection) in [
                        ("gate_proj", &expert.gate_proj),
                        ("up_proj", &expert.up_proj),
                        ("down_proj", &expert.down_proj),
                    ] {
                        parameters += assert_linear_f16(
                            &format!(
                                "model.layers.{layer_index}.sparse_mlp.routed.routed.{expert_index}.{name}"
                            ),
                            projection,
                        );
                    }
                }
                for (name, projection) in [
                    ("gate_proj", &sparse.shared_experts.gate_proj),
                    ("up_proj", &sparse.shared_experts.up_proj),
                    ("down_proj", &sparse.shared_experts.down_proj),
                ] {
                    parameters += assert_linear_f16(
                        &format!("model.layers.{layer_index}.sparse_mlp.shared_experts.{name}"),
                        projection,
                    );
                }
            }
            _ => panic!("layer {layer_index} must contain exactly one feed-forward topology"),
        }
    }
    assert_eq!(parameters, 5_291);
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

/// Borrows the concrete compressed MLA state from automatic dispatch.
#[cfg(feature = "metal")]
fn deepseek_state(state: &AutoCausalDecodeState<Metal>) -> &DeepseekV2DecodeState<Metal> {
    match state {
        AutoCausalDecodeState::DeepseekV2(state) => state,
        _ => panic!("automatic decode state must be DeepSeek-V2"),
    }
}

/// Fails quickly when macOS does not expose a usable Metal adapter.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires an available Metal adapter"]
fn initializes_metal_acceptance_backend() {
    assert_metal_backend_available(&Default::default());
}

/// Strictly loads F16 weights and proves reference/absorbed MLA execution parity.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires the pinned 15.7B checkpoint, Metal, and about 40 GiB free unified memory"]
fn loads_pinned_deepseek_v2_lite_f16_and_runs_metal_smoke() {
    let root = pinned_root();
    assert_control_files(&root);
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
    .expect("the public auto loader must strictly stream the complete DeepSeek-V2-Lite graph");
    <Metal<f32, i32> as Backend>::sync(&device)
        .expect("all Metal parameter transfers must complete");
    let load_seconds = load_started.elapsed().as_secs_f64();
    let concrete = match loaded.model() {
        AutoCausalLm::DeepseekV2(model) => model,
        model => panic!("expected DeepSeek-V2 auto dispatch, found {model:?}"),
    };
    assert_all_parameters_f16(concrete);
    assert_eq!(loaded.source().model_type(), "deepseek_v2");
    assert_eq!(loaded.source().repository(), Some(REPOSITORY));
    assert_eq!(loaded.source().revision(), Some(REVISION));
    assert_eq!(
        loaded.source().root(),
        root.canonicalize().expect("canonical pinned artifact root")
    );
    let report = loaded.load_report();
    assert!(report.is_strictly_valid());
    assert_eq!(report.applied.len(), 5_291);
    assert_eq!(report.transformed.len(), 5_099);
    assert_eq!(report.dtype_converted.len(), 5_291);
    assert!(report.aliased.is_empty());
    assert!(report.allowlisted_ignored.is_empty());
    assert!(report.allowlisted_missing.is_empty());
    let execution = loaded
        .execution_stats()
        .expect("the automatic loader must retain pass-two execution counters");
    assert_eq!(execution.tensors_applied, 5_291);
    assert_eq!(execution.aliases_applied, 0);
    assert_eq!(execution.source_reads, 5_291);
    assert_eq!(execution.source_payload_bytes, 31_412_968_448);
    assert_eq!(execution.delivered_tensor_bytes, 31_412_968_448);
    assert_eq!(execution.largest_source_payload_bytes, 419_430_400);
    assert_eq!(execution.largest_output_tensor_bytes, 419_430_400);
    assert_eq!(execution.peak_host_staging_bytes, 419_430_400);
    assert_eq!(execution.source.shard_opens, 4);
    assert_eq!(execution.source.tensor_reads, 5_291);
    assert_eq!(execution.source.payload_bytes_read, 31_412_968_448);
    assert_eq!(execution.source.max_open_shards, 1);
    assert_eq!(
        execution.source.shard_open_order,
        SHARDS
            .iter()
            .map(|(name, _, _)| (*name).to_owned())
            .collect::<Vec<_>>()
    );

    let tokenizer = loaded
        .tokenizer()
        .expect("the loaded bundle must retain the local tokenizer");
    let encoding = tokenizer
        .encode(
            SMOKE_PROMPT,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("the committed Metal prompt must encode");
    let prompt_ids = encoding.token_ids()[0].clone();
    assert_eq!(prompt_ids.as_slice(), SMOKE_TOKEN_IDS);
    let sequence = prompt_ids.len();
    let input = encoding
        .to_causal_input::<Metal>(&device)
        .expect("the prompt encoding must become a causal input");
    let diagnostics = ForwardOptions {
        hidden_states: true,
        ..ForwardOptions::default()
    };

    let stateless_started = Instant::now();
    let stateless = loaded
        .model()
        .forward(input.clone(), &diagnostics)
        .expect("the loaded model must execute stateless inference");
    assert_eq!(stateless.logits.dims(), [1, 1, 102_400]);
    assert_eq!(stateless.hidden_states.len(), 29);
    let stateless_logits = metal_values("DeepSeek stateless logits", &stateless.logits);
    let stateless_trace = metal_values(
        "DeepSeek stateless hidden trace",
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
        .expect("the loaded model must allocate compressed MLA state");
    assert_eq!(deepseek_state(&state).scalars_per_layer_token(), 576);
    assert_eq!(deepseek_state(&state).bytes_per_token(), 576 * 27 * 2);
    assert_eq!(
        deepseek_state(&state).allocated_bytes(),
        (sequence + 1) * 576 * 27 * 2
    );
    assert_eq!(
        deepseek_state(&state)
            .layer_storage(0)
            .expect("first MLA layer storage")
            .dims(),
        [1, sequence + 1, 576]
    );
    assert_eq!(deepseek_state(&state).last_execution_path(), None);

    let prefill_started = Instant::now();
    let cached = loaded
        .model()
        .prefill(input, &mut state, &diagnostics)
        .expect("the loaded model must execute reference MLA prefill");
    assert_eq!(cached.logits.dims(), [1, 1, 102_400]);
    assert_eq!(cached.hidden_states.len(), 29);
    let cached_logits = metal_values("DeepSeek cached logits", &cached.logits);
    let cached_trace = metal_values(
        "DeepSeek cached hidden trace",
        &Tensor::cat(cached.hidden_states.clone(), 0),
    );
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    assert!(cached_logits.iter().all(|value| value.is_finite()));
    assert!(cached_trace.iter().all(|value| value.is_finite()));
    assert_eq!(state.lengths(), &[sequence]);
    assert_eq!(
        deepseek_state(&state).last_execution_path(),
        Some(MlaExecutionPath::ReferenceUnabsorbed)
    );

    let cached_cosine = cosine_similarity(&stateless_logits, &cached_logits);
    assert!(
        cached_cosine >= 0.999,
        "DeepSeek cached/stateless logit cosine {cached_cosine} is below the gate"
    );
    let trace_width = sequence * 2_048;
    let hidden_cosines = stateless_trace
        .chunks_exact(trace_width)
        .zip(cached_trace.chunks_exact(trace_width))
        .map(|(stateless, cached)| cosine_similarity(stateless, cached))
        .collect::<Vec<_>>();
    assert_eq!(hidden_cosines.len(), 29);
    for (trace, cosine) in hidden_cosines.iter().copied().enumerate() {
        assert!(
            cosine >= 0.999,
            "DeepSeek cached/stateless hidden cosine at trace {trace} was {cosine}"
        );
    }
    let minimum_prefill_hidden_cosine =
        hidden_cosines.iter().copied().fold(f64::INFINITY, f64::min);
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
            &diagnostics,
        )
        .expect("the loaded model must execute one absorbed MLA decode step");
    assert_eq!(decoded.logits.dims(), [1, 1, 102_400]);
    assert_eq!(decoded.hidden_states.len(), 29);
    let decoded_logits = metal_values("DeepSeek absorbed decoded logits", &decoded.logits);
    let decoded_trace = metal_values(
        "DeepSeek absorbed decoded hidden trace",
        &Tensor::cat(decoded.hidden_states.clone(), 0),
    );
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    assert!(decoded_logits.iter().all(|value| value.is_finite()));
    assert!(decoded_trace.iter().all(|value| value.is_finite()));
    assert_eq!(state.lengths(), &[sequence + 1]);
    assert_eq!(
        deepseek_state(&state).last_execution_path(),
        Some(MlaExecutionPath::AbsorbedDecode)
    );

    let mut full_ids = prompt_ids
        .iter()
        .copied()
        .map(i64::from)
        .collect::<Vec<_>>();
    full_ids.push(next_token);
    let extended_started = Instant::now();
    let extended = loaded
        .model()
        .forward(
            CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(full_ids, [1, sequence + 1]),
                    &device,
                ),
                token_mask: None,
                position_ids: None,
            },
            &diagnostics,
        )
        .expect("the full stateless graph must provide the absorbed-decode oracle");
    let extended_seconds = extended_started.elapsed().as_secs_f64();
    let extended_logits = metal_values("DeepSeek extended stateless logits", &extended.logits);
    let extended_last_hidden = Tensor::cat(
        extended
            .hidden_states
            .iter()
            .map(|hidden| {
                hidden
                    .clone()
                    .slice([0..1, sequence..sequence + 1, 0..2_048])
            })
            .collect(),
        0,
    );
    let extended_trace = metal_values("DeepSeek extended last hidden trace", &extended_last_hidden);
    let decoded_cosine = cosine_similarity(&extended_logits, &decoded_logits);
    assert!(
        decoded_cosine >= 0.999,
        "DeepSeek absorbed/stateless decode logit cosine {decoded_cosine} is below the gate"
    );
    let decode_hidden_cosines = extended_trace
        .chunks_exact(2_048)
        .zip(decoded_trace.chunks_exact(2_048))
        .map(|(stateless, cached)| cosine_similarity(stateless, cached))
        .collect::<Vec<_>>();
    assert_eq!(decode_hidden_cosines.len(), 29);
    for (trace, cosine) in decode_hidden_cosines.iter().copied().enumerate() {
        assert!(
            cosine >= 0.999,
            "DeepSeek absorbed/stateless hidden cosine at trace {trace} was {cosine}"
        );
    }
    let minimum_decode_hidden_cosine = decode_hidden_cosines
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert_eq!(
        decoded.logits.argmax(2).to_data().to_vec::<i32>().unwrap(),
        extended.logits.argmax(2).to_data().to_vec::<i32>().unwrap()
    );
    let next_token_u32 = u32::try_from(next_token).expect("top-1 token must fit u32");
    let decoded_token = tokenizer
        .decode(&[next_token_u32], true)
        .expect("the first generated token must decode");

    eprintln!(
        "DeepSeek-V2-Lite F16 Metal: prompt_tokens={sequence}, prompt_ids={prompt_ids:?}, \
         first_top1={next_token}, first_text={decoded_token:?}, cached_cosine={cached_cosine:.9}, \
         decoded_cosine={decoded_cosine:.9}, min_prefill_hidden_cosine={minimum_prefill_hidden_cosine:.9}, \
         min_decode_hidden_cosine={minimum_decode_hidden_cosine:.9}, cache_scalars_per_layer_token=576, \
         cache_bytes_per_token={}, stateless_logits_fnv1a64={:016x}, cached_logits_fnv1a64={:016x}, \
         decoded_logits_fnv1a64={:016x}, extended_logits_fnv1a64={:016x}, \
         stateless_trace_fnv1a64={:016x}, cached_trace_fnv1a64={:016x}, \
         decoded_trace_fnv1a64={:016x}, extended_trace_fnv1a64={:016x}, \
         load_seconds={load_seconds:.3}, shard_opens={}, max_open_shards={}, source_reads={}, \
         stateless_seconds={stateless_seconds:.3}, prefill_seconds={prefill_seconds:.3}, \
         decode_seconds={decode_seconds:.3}, extended_seconds={extended_seconds:.3}",
        deepseek_state(&state).bytes_per_token(),
        fnv1a64_f32(&stateless_logits),
        fnv1a64_f32(&cached_logits),
        fnv1a64_f32(&decoded_logits),
        fnv1a64_f32(&extended_logits),
        fnv1a64_f32(&stateless_trace),
        fnv1a64_f32(&cached_trace),
        fnv1a64_f32(&decoded_trace),
        fnv1a64_f32(&extended_trace),
        execution.source.shard_opens,
        execution.source.max_open_shards,
        execution.source_reads,
    );
}
