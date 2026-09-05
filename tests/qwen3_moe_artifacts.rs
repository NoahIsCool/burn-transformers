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
        TensorDType,
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
#[cfg(feature = "metal")]
use transformers::{
    AutoCausalLm, AutoLoadOptions, DecodeInput, DecodeState, ForwardOptions, cache::CacheConfig,
    models::qwen3_moe::Qwen3MoeForCausalLm,
};

/// Immutable upstream repository certified by both gates.
const REPOSITORY: &str = "Qwen/Qwen3-30B-A3B";

/// Immutable upstream Git revision certified by both gates.
const REVISION: &str = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39";

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
#[cfg(feature = "metal")]
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
            assert_eq!(*declared_total_size, Some(61_064_245_248));
        }
        ArtifactSource::Single { relative_path } => {
            panic!("expected the pinned shard index, found {relative_path}")
        }
        _ => panic!("expected the standard indexed SafeTensors artifact"),
    }
    assert_eq!(manifest.shards.len(), 16);
    assert_eq!(manifest.tensors.len(), 18_867);
    assert_eq!(manifest.tensor_bytes, 61_064_245_248);
    assert_eq!(manifest.artifact_bytes, 61_066_575_648);
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
    assert_eq!(contract.expected_tensors().len(), 18_867);
    let plan = contract
        .build_load_plan(&manifest, &options)
        .expect("every pinned tensor must map under strict F16 planning");
    assert_eq!(plan.parameter_count(), 18_867);
    assert_eq!(plan.tensors.len(), 18_867);
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
    assert_eq!(plan.report.dtype_converted.len(), 18_867);
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
    assert_eq!(plan.memory.artifact_bytes, 61_066_575_648);
    assert_eq!(plan.memory.checkpoint_tensor_bytes, 61_064_245_248);
    assert_eq!(plan.memory.target_parameter_bytes, 61_064_245_248);
    assert_eq!(plan.memory.logical_model_bytes, 61_064_245_248);
    assert_eq!(plan.memory.largest_shard_bytes, 3_999_975_472);
    assert_eq!(plan.memory.largest_source_tensor_bytes, 622_329_856);
    assert_eq!(plan.memory.largest_target_tensor_bytes, 622_329_856);
    assert_eq!(plan.memory.peak_host_staging_bytes, 4_622_305_328);
    assert_eq!(plan.report.memory, Some(plan.memory));

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
    assert_eq!(parameters, 18_867);
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
    assert_shard_file_hashes(&root);
    let device = Default::default();
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
    assert_eq!(report.applied.len(), 18_867);
    assert_eq!(report.transformed.len(), 18_480);
    assert_eq!(report.dtype_converted.len(), 18_867);
    assert!(report.aliased.is_empty());
    assert!(report.allowlisted_ignored.is_empty());
    assert!(report.allowlisted_missing.is_empty());
    let execution = loaded
        .execution_stats()
        .expect("the automatic loader must retain pass-two execution counters");
    assert_eq!(execution.tensors_applied, 18_867);
    assert_eq!(execution.aliases_applied, 0);
    assert_eq!(execution.source_reads, 18_867);
    assert_eq!(execution.source_payload_bytes, 61_064_245_248);
    assert_eq!(execution.delivered_tensor_bytes, 61_064_245_248);
    assert_eq!(execution.largest_source_payload_bytes, 622_329_856);
    assert_eq!(execution.largest_output_tensor_bytes, 622_329_856);
    assert_eq!(execution.peak_host_staging_bytes, 622_329_856);
    assert_eq!(execution.source.shard_opens, 16);
    assert_eq!(execution.source.tensor_reads, 18_867);
    assert_eq!(execution.source.payload_bytes_read, 61_064_245_248);
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
