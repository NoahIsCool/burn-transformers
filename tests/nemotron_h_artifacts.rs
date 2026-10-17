//! Immutable artifact and oracle gates for Nemotron 3.5 Lightning 30B-A3B.
//!
//! Ordinary CI authenticates the exact upstream config and shard index, parses
//! committed copies of all fourteen SafeTensors headers, and independently
//! checks every one of the 6,513 source names, shapes, dtypes, byte ranges, and
//! shard assignments. No tensor payload is committed or read by those tests.
//!
//! The upstream index contains a reproducible accounting defect: its reported
//! `metadata.total_size` exceeds the physical tensor payload by exactly the
//! combined size of the 23 base-model router matrices. The matrices occur only
//! once in both the index and the physical headers; the MTP router is not part
//! of the discrepancy. Generic checkpoint discovery continues to reject this
//! artifact; only the exact provenance-bound Nemotron family policy accepts the
//! reviewed overstatement while preserving the physical tensor inventory.
//!
//! Once a complete local checkpoint is available, authenticate every payload
//! without loading the model using:
//!
//! ```text
//! TRANSFORMERS_NEMOTRON_H_DIR=/absolute/pinned/checkpoint \
//! cargo test --release --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,tokenizers \
//!   --test nemotron_h_artifacts \
//!   validates_pinned_nemotron_control_and_weight_payloads \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! To load every tensor and exercise the base graph on Metal, use:
//!
//! ```text
//! TRANSFORMERS_NEMOTRON_H_DIR=/absolute/pinned/checkpoint \
//! cargo test --release --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,metal,tokenizers \
//!   --test nemotron_h_artifacts \
//!   loads_pinned_nemotron_f16_and_smokes_base_inference \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```
//!
//! To compare both Burn paths with their corresponding authenticated official
//! rows, additionally select the external oracle bundle:
//!
//! ```text
//! TRANSFORMERS_NEMOTRON_H_DIR=/absolute/pinned/checkpoint \
//! TRANSFORMERS_NEMOTRON_H_REFERENCE_DIR=/absolute/pinned/reference.bundle \
//! cargo test --release --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,metal,tokenizers \
//!   --test nemotron_h_artifacts \
//!   compares_pinned_nemotron_f16_with_official_base_reference \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```
//!
//! The separately produced numerical oracle is deliberately fail-closed. Its
//! JSON and every binary payload require independent SHA-256 anchors before the
//! ignored reader will accept the bundle.

#[path = "support/nemotron_h_reference.rs"]
mod nemotron_h_reference;

#[path = "support/nemotron_h_base_reference.rs"]
mod nemotron_h_base_reference;

#[path = "support/nemotron_h_parity.rs"]
mod nemotron_h_parity;

#[path = "support/nemotron_h_precision_reference.rs"]
mod nemotron_h_precision_reference;

use serde::{
    Deserialize, Deserializer,
    de::{Error as _, MapAccess, Visitor},
};
use sha2::{Digest, Sha256};
#[cfg(all(feature = "metal", feature = "tokenizers"))]
use std::time::Instant;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    str::FromStr,
};
use transformers::{
    checkpoint::{
        ArtifactSource, CheckpointError, CheckpointManifest, DTypePolicy, IndexAliasSizeAccounting,
        LoadOptions, PlannedTransform, ShardInventory, TensorDType, TensorInventory,
        TensorOperation,
    },
    models::{
        config::ModelConfig,
        nemotron_h::{NemotronHBlockType, NemotronHCheckpointOptions, NemotronHConfig},
    },
};

#[cfg(all(feature = "metal", feature = "tokenizers"))]
use burn::{
    backend::Metal,
    tensor::{Bool, DType, Int, Tensor, TensorData, backend::Backend},
};
#[cfg(all(feature = "metal", feature = "tokenizers"))]
use transformers::{
    AutoCausalLm, AutoLoadOptions, DecodeInput, DecodeState, ForwardOptions, LogitsSelection,
    TokenMask,
    architecture::CausalLmInput,
    cache::{CacheConfig, KvCacheStorage},
    models::AutoCausalDecodeState,
};

/// Immutable upstream repository certified by every gate.
const REPOSITORY: &str = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16";

/// Immutable upstream repository revision certified by every gate.
const REVISION: &str = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9";

/// Exact official config bytes committed as a compact CI fixture.
const CONFIG_BYTES: &[u8] =
    include_bytes!("fixtures/nemotron_h_checkpoint/nemotron_3_5_lightning_30b_a3b_config.json");

/// Exact official shard-index bytes committed as an independent key inventory.
const INDEX_BYTES: &[u8] =
    include_bytes!("fixtures/nemotron_h_checkpoint/nemotron_3_5_lightning_30b_a3b_index.json");

/// Machine-readable release state coupled to the executable fixture evidence.
const ACCEPTANCE_BYTES: &[u8] = include_bytes!(
    "../acceptance/nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json"
);

/// Exact config identity at [`REVISION`].
const CONFIG_SHA256: &str = "a3827a0f5e311547b40943dc081e3ff2f8a277466e8c1a3df2291e8db8a7617c";

/// Exact standard shard-index identity at [`REVISION`].
const INDEX_SHA256: &str = "67f21da80ce245a3e24967f54eef3fa10e67a63eba16ef19a2d0569f06103f50";

/// Canonical SHA-256 over sorted `name NUL shard LF` index assignments.
const INDEX_INVENTORY_SHA256: &str =
    "0e3b013820bc7ef347c3daf15f3886a2f3fa8e6e3c38e1859662b624601eda09";

/// Canonical SHA-256 over every physical name, shard, dtype, shape, and range.
const HEADER_INVENTORY_SHA256: &str =
    "e255cec4fe6d43dfddaed52b3a5ef719c111d6c9821ce7ff53219080b50762c4";

/// Canonical SHA-256 over the strict F16 target plan after correcting metadata.
const F16_LOAD_PLAN_SHA256: &str =
    "a47056d8d8e5ad814f79f4b147c3d7b340362013c31f126b7578e25efb51c906";

/// Canonical digest over the seven controls and fourteen complete shard identities.
const CHECKPOINT_ARTIFACT_MANIFEST_SHA256: &str =
    "cb72063ee5b215548d9adbc67052de5f54871f1e28b80a0e0db9782afd4ad759";

/// Exact physical source tensor count, including the 270-tensor MTP body.
const PHYSICAL_TENSOR_COUNT: usize = 6_513;

/// Base backbone tensors below `backbone.*`.
const BACKBONE_TENSOR_COUNT: usize = 6_242;

/// Complete single-stage MTP body stored below `mtp.*`.
const MTP_TENSOR_COUNT: usize = 270;

/// BF16 tensor count recovered from all physical headers.
const BF16_TENSOR_COUNT: usize = 6_489;

/// FP32 router-correction buffers recovered from all physical headers.
const F32_TENSOR_COUNT: usize = 24;

/// Exact sum of tensor data sections across the fourteen shards.
const PHYSICAL_TENSOR_BYTES: u64 = 65_826_538_624;

/// Exact sum of complete shard file lengths, including headers.
const PHYSICAL_ARTIFACT_BYTES: u64 = 65_827_374_264;

/// Upstream `metadata.total_size`, retained verbatim even though it is wrong.
const INDEX_REPORTED_TOTAL_SIZE: u64 = 65_842_365_568;

/// Exact upstream overstatement reconciled to the 23 base router matrices.
const INDEX_TOTAL_SIZE_DELTA: u64 = 15_826_944;

/// Exact base parameter count reported by the upstream index.
const INDEX_REPORTED_BASE_PARAMETERS: u64 = 31_577_937_344;

/// Physical scalar elements across base, head, MTP, and correction buffers.
const PHYSICAL_TENSOR_ELEMENTS: u64 = 32_913_266_240;

/// Physical base/head elements before the separate MTP shard.
const BASE_PHYSICAL_ELEMENTS: u64 = 31_577_940_288;

/// The 23 base routers each carry one 128-element non-trainable correction buffer.
const BASE_ROUTER_CORRECTION_ELEMENTS: u64 = 2_944;

/// Physical elements in the MTP body, including its 128-element FP32 buffer.
const MTP_PHYSICAL_ELEMENTS: u64 = 1_335_325_952;

/// Independent LM-head or embedding payload, the largest source tensor.
const LARGEST_TENSOR_BYTES: u64 = 704_643_072;

/// Largest complete physical shard.
const LARGEST_SHARD_BYTES: u64 = 4_995_692_816;

/// Conservative pass-two host staging upper bound.
const PEAK_HOST_STAGING_BYTES: u64 = 5_700_335_888;

/// Fixed smoke prompt shared with the other family gates.
const SMOKE_PROMPT: &str = "Hello world, cached decoding.";

/// Exact tokenization without injected special tokens.
const SMOKE_TOKEN_IDS: [u32; 6] = [22_177, 4_304, 1_044, 57_424, 59_417, 1_046];

/// Small immutable control artifacts required by loading or generation.
const CONTROL_FILES: [(&str, u64, &str); 7] = [
    (
        "chat_template.jinja",
        9_867,
        "58933db77d3099b4f78c55a38347a72e1ea05b97d6bd8f38775303dc0194e0a9",
    ),
    ("config.json", 2_352, CONFIG_SHA256),
    (
        "generation_config.json",
        210,
        "0ed94823ae737b507aef6af01e837041b7aa2fb0306793a310cd201d89a31ff4",
    ),
    ("model.safetensors.index.json", 612_207, INDEX_SHA256),
    (
        "special_tokens_map.json",
        563,
        "e9435fefd6d838fd9fcbbc44b97a8e3ff322be7f6dfb7e4fd2468586574bb52b",
    ),
    (
        "tokenizer.json",
        17_077_484,
        "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
    ),
    (
        "tokenizer_config.json",
        177_209,
        "10f93eabcb9b1602fbb991d6308e787ce1df28ee9cd7a1c6d1e8c3f338b957bc",
    ),
];

/// Exact metadata and full-payload identities for one physical shard.
#[derive(Debug, Clone, Copy)]
struct ShardIdentity {
    /// Repository-relative shard filename.
    name: &'static str,
    /// Complete LFS object length.
    file_bytes: u64,
    /// SafeTensors JSON header bytes, excluding its prefix.
    header_bytes: u64,
    /// Number of tensors in this header.
    tensors: usize,
    /// Tensor data-section bytes.
    data_bytes: u64,
    /// SHA-256 of the eight-byte prefix plus exact JSON header bytes.
    header_sha256: &'static str,
    /// SHA-256 of the complete LFS object.
    payload_sha256: &'static str,
}

/// All fourteen exact shard/header identities in filename order.
const SHARDS: [ShardIdentity; 14] = [
    ShardIdentity {
        name: "model-00001-of-00014.safetensors",
        file_bytes: 4_991_210_024,
        header_bytes: 55_584,
        tensors: 435,
        data_bytes: 4_991_154_432,
        header_sha256: "fb38f4b142305784dd21d85d73206656a3a0e928a050336dbc5a565f77af00b5",
        payload_sha256: "7a3e74d24969eac6657cb9b6422091fd062b7d30fd4bc90ecee6c3d7f692626a",
    },
    ShardIdentity {
        name: "model-00002-of-00014.safetensors",
        file_bytes: 4_992_601_016,
        header_bytes: 64_432,
        tensors: 505,
        data_bytes: 4_992_536_576,
        header_sha256: "08ec30982e2c7e084869288cb600c6c9153159db2a94a1696d4656d5048c5f6d",
        payload_sha256: "a80c3846e959f6dda34aa031cbb668b766643691ac7e5c9f43f62e6fb1221059",
    },
    ShardIdentity {
        name: "model-00003-of-00014.safetensors",
        file_bytes: 4_992_601_368,
        header_bytes: 64_784,
        tensors: 505,
        data_bytes: 4_992_536_576,
        header_sha256: "0ca2ca3a63693de596a3757212cb13b634b24bc6d1a955085585017fa3fd4958",
        payload_sha256: "996bf3a46668e14075cb82259390267c2b1f71446caeb3a5f1f9e0bd7a079ae7",
    },
    ShardIdentity {
        name: "model-00004-of-00014.safetensors",
        file_bytes: 4_995_692_800,
        header_bytes: 65_016,
        tensors: 505,
        data_bytes: 4_995_627_776,
        header_sha256: "891f6df23c1628935d655f93775cd8a0d447a574ffb5e0c5be5db50d703c2a11",
        payload_sha256: "cce1d79e7ff2e2b746b9d97166156aa1dcec796116675cc7fd1aa70a8d73ccb2",
    },
    ShardIdentity {
        name: "model-00005-of-00014.safetensors",
        file_bytes: 4_980_553_832,
        header_bytes: 64_864,
        tensors: 504,
        data_bytes: 4_980_488_960,
        header_sha256: "89aba226b7c9f8e2220b804f609b7104e3e73578b2c8b85499f4dd035f250079",
        payload_sha256: "9338f59d3fd5f074288c71a9559452d5b5c581da00e9ccb9fe7aa2ce9b04d2ad",
    },
    ShardIdentity {
        name: "model-00006-of-00014.safetensors",
        file_bytes: 4_989_423_536,
        header_bytes: 64_808,
        tensors: 502,
        data_bytes: 4_989_358_720,
        header_sha256: "4829d1fdfeb55765e21995b9d17c888ef3c247f356941c0a256da0aa841d9196",
        payload_sha256: "e947fa4097c68263ba3d097fceb32a300a22cebd9d909fb5198af7d18a4022fe",
    },
    ShardIdentity {
        name: "model-00007-of-00014.safetensors",
        file_bytes: 4_992_601_488,
        header_bytes: 64_904,
        tensors: 505,
        data_bytes: 4_992_536_576,
        header_sha256: "75123a4888d68002af022241a649406cc0f95183c23d9f9b94e218a12a942a85",
        payload_sha256: "c7362630bf4256275ca58354ed8f31506cec67ffab30573e753703fd5ae48795",
    },
    ShardIdentity {
        name: "model-00008-of-00014.safetensors",
        file_bytes: 4_992_601_520,
        header_bytes: 64_936,
        tensors: 505,
        data_bytes: 4_992_536_576,
        header_sha256: "42e44897a6deb75c8744b0f768400a70f38898f35e8a2c3696fb3088ae0db397",
        payload_sha256: "1b33a5a6b5505b25b57c90ddda19e9cd1598176468917df0c56116fa6f6072cb",
    },
    ShardIdentity {
        name: "model-00009-of-00014.safetensors",
        file_bytes: 4_995_692_800,
        header_bytes: 65_016,
        tensors: 505,
        data_bytes: 4_995_627_776,
        header_sha256: "88a8b4f83aeeb4339b15fb3325a652432a1e4ddcc1e174646d1f0b44ca39aa99",
        payload_sha256: "fcc9b48caff427beaf1f87b78cf4ab42bc2a3c43b3c487bccb013ba22d7468b2",
    },
    ShardIdentity {
        name: "model-00010-of-00014.safetensors",
        file_bytes: 4_992_601_520,
        header_bytes: 64_936,
        tensors: 505,
        data_bytes: 4_992_536_576,
        header_sha256: "e866c97f325cd3ac26c0fe2390e5c70af4a6d06216658af439ea217aa12be429",
        payload_sha256: "d1924fd54d1b77a15aff8df591a317e8a82a58e46d6f5742fa21e90e0be3cfa6",
    },
    ShardIdentity {
        name: "model-00011-of-00014.safetensors",
        file_bytes: 4_995_692_800,
        header_bytes: 65_016,
        tensors: 505,
        data_bytes: 4_995_627_776,
        header_sha256: "9899e46c25fd005368c78209f2301b9bf4168ca09b7dc5516b5c2a32b80fc24c",
        payload_sha256: "e9d0b83bcf04bd4d5758b94e6495a3f1595d121d82faffaa27ce6a884e2c8757",
    },
    ShardIdentity {
        name: "model-00012-of-00014.safetensors",
        file_bytes: 4_995_692_816,
        header_bytes: 65_032,
        tensors: 505,
        data_bytes: 4_995_627_776,
        header_sha256: "ca949857488ac8cf521b4570e25c6816104db6d84171190c12e7144157a9279e",
        payload_sha256: "fce184f112cb0b97024cadc60f993f6f660e365d04e21c240fb3c75747bd3a01",
    },
    ShardIdentity {
        name: "model-00013-of-00014.safetensors",
        file_bytes: 3_249_723_504,
        header_bytes: 33_128,
        tensors: 257,
        data_bytes: 3_249_690_368,
        header_sha256: "d4f22cfeb2b0e72ae08ce5ec215b6fa5ea4850ec934cda2a0f77d86d3e3a039e",
        payload_sha256: "f427d037d2a3e4d44323325aa38571e3551e8ad702f469c3dacc4e59ea2e50d1",
    },
    ShardIdentity {
        name: "model-00014-of-00014.safetensors",
        file_bytes: 2_670_685_240,
        header_bytes: 33_072,
        tensors: 270,
        data_bytes: 2_670_652_160,
        header_sha256: "6c2cabebac913a55562ba8c7730f07818664a6d729bee4b16a010924bb75804d",
        payload_sha256: "64577b275ca4e7e5266eae0903674f7f46ec2a8cbf4f4f1a3207f80d503cd1d0",
    },
];

/// Exact prefix-plus-header fixtures corresponding to [`SHARDS`].
const HEADER_FIXTURES: [&[u8]; 14] = [
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00001-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00002-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00003-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00004-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00005-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00006-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00007-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00008-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00009-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00010-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00011-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00012-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00013-of-00014.safetensors.header"
    ),
    include_bytes!(
        "fixtures/nemotron_h_checkpoint/headers/model-00014-of-00014.safetensors.header"
    ),
];

/// Strict upstream index metadata.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexMetadata {
    /// Base-model parameters reported by the producer.
    total_parameters: u64,
    /// Producer-reported payload size, known to be overstated.
    total_size: u64,
}

/// Strict upstream shard index.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PinnedIndex {
    /// Producer metadata.
    metadata: IndexMetadata,
    /// Duplicate-rejecting physical tensor assignments.
    weight_map: UniqueWeightMap,
}

/// A map whose deserializer rejects duplicate tensor names.
#[derive(Debug)]
struct UniqueWeightMap(
    /// Unique tensor names mapped to their physical checkpoint shard.
    BTreeMap<String, String>,
);

impl<'de> Deserialize<'de> for UniqueWeightMap {
    /// Parses an index map without serde_json's usual last-value-wins behavior.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        /// Serde visitor that detects repeated tensor names before constructing the index map.
        struct UniqueWeightMapVisitor;

        impl<'de> Visitor<'de> for UniqueWeightMapVisitor {
            /// Validated tensor-to-shard mapping returned by this visitor.
            type Value = UniqueWeightMap;

            /// Describes the exact JSON construct accepted here.
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a tensor-name to shard-name object without duplicate keys")
            }

            /// Rejects a repeated tensor key before constructing the map.
            fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((name, shard)) = access.next_entry::<String, String>()? {
                    if values.insert(name.clone(), shard).is_some() {
                        return Err(A::Error::custom(format!(
                            "duplicate tensor name in weight_map: {name}"
                        )));
                    }
                }
                Ok(UniqueWeightMap(values))
            }
        }

        deserializer.deserialize_map(UniqueWeightMapVisitor)
    }
}

/// One raw SafeTensors header entry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHeaderTensor {
    /// SafeTensors scalar spelling.
    dtype: String,
    /// Logical row-major shape.
    shape: Vec<u64>,
    /// Half-open offsets relative to the data section.
    data_offsets: [u64; 2],
}

/// Hashes an in-memory fixture as lower-case SHA-256.
fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Hashes one potentially multi-gigabyte regular file with bounded memory.
fn sha256_file_hex(path: &Path) -> String {
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
    let mut file =
        File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
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

/// Parses the exact config fixture through the production family schema.
fn pinned_config() -> NemotronHConfig {
    NemotronHConfig::from_json_str(
        std::str::from_utf8(CONFIG_BYTES).expect("official config fixture must be UTF-8"),
    )
    .expect("official Nemotron 3.5 config must validate")
}

/// Parses the exact index while rejecting repeated source tensor names.
fn pinned_index() -> PinnedIndex {
    serde_json::from_slice(INDEX_BYTES).expect("pinned shard index must satisfy its strict schema")
}

/// Parses every committed header and assembles the complete physical manifest.
fn pinned_header_manifest(index: &PinnedIndex) -> CheckpointManifest {
    let mut global = BTreeMap::new();
    let mut shards = Vec::with_capacity(SHARDS.len());
    let mut tensor_bytes = 0_u64;
    let mut artifact_bytes = 0_u64;

    for (identity, bytes) in SHARDS.iter().zip(HEADER_FIXTURES) {
        assert_eq!(
            sha256_hex(bytes),
            identity.header_sha256,
            "{}",
            identity.name
        );
        let prefix: [u8; 8] = bytes[..8]
            .try_into()
            .expect("header prefix must contain eight bytes");
        let header_bytes = u64::from_le_bytes(prefix);
        assert_eq!(header_bytes, identity.header_bytes, "{}", identity.name);
        assert_eq!(bytes.len() as u64, header_bytes + 8, "{}", identity.name);

        let mut raw: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..]).expect("pinned SafeTensors header must be JSON");
        let metadata = raw
            .remove("__metadata__")
            .map(|value| serde_json::from_value(value).expect("header metadata must be strings"))
            .unwrap_or_default();
        assert_eq!(raw.len(), identity.tensors, "{}", identity.name);

        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::with_capacity(raw.len());
        for (name, value) in raw {
            let header: RawHeaderTensor =
                serde_json::from_value(value).expect("header tensor metadata must be strict");
            assert_eq!(
                index.weight_map.0.get(&name).map(String::as_str),
                Some(identity.name)
            );
            let dtype =
                TensorDType::from_str(&header.dtype).expect("pinned dtype must be supported");
            let elements = header
                .shape
                .iter()
                .try_fold(1_u64, |product, dimension| {
                    product
                        .checked_mul(*dimension)
                        .ok_or("shape product overflow")
                })
                .expect("pinned shape product must fit u64");
            let byte_len = dtype
                .storage_bytes(elements)
                .expect("pinned tensor byte length must be valid");
            assert_eq!(
                header.data_offsets[1] - header.data_offsets[0],
                byte_len,
                "{name}"
            );
            ranges.push((header.data_offsets[0], header.data_offsets[1], name.clone()));
            let tensor = TensorInventory {
                name: name.clone(),
                shard: identity.name.to_owned(),
                shape: header.shape,
                dtype,
                data_offsets: header.data_offsets,
                byte_len,
            };
            assert!(tensors.insert(name.clone(), tensor.clone()).is_none());
            assert!(
                global.insert(name, tensor).is_none(),
                "duplicate tensor across headers"
            );
        }
        ranges.sort();
        assert_eq!(
            ranges.first().map(|range| range.0),
            Some(0),
            "{}",
            identity.name
        );
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].1, pair[1].0, "gap or overlap before {}", pair[1].2);
        }
        let data_bytes = ranges.last().map_or(0, |range| range.1);
        assert_eq!(data_bytes, identity.data_bytes, "{}", identity.name);
        assert_eq!(
            8 + header_bytes + data_bytes,
            identity.file_bytes,
            "{}",
            identity.name
        );
        tensor_bytes += data_bytes;
        artifact_bytes += identity.file_bytes;
        shards.push(ShardInventory {
            relative_path: identity.name.to_owned(),
            path: PathBuf::from(identity.name),
            file_bytes: identity.file_bytes,
            header_bytes,
            data_bytes,
            metadata,
            tensors,
        });
    }
    assert_eq!(global.len(), index.weight_map.0.len());
    assert_eq!(
        global.keys().collect::<Vec<_>>(),
        index.weight_map.0.keys().collect::<Vec<_>>()
    );
    CheckpointManifest {
        root: PathBuf::from("pinned-nemotron-header-fixtures"),
        source: ArtifactSource::Index {
            relative_path: "model.safetensors.index.json".into(),
            declared_total_size: Some(INDEX_REPORTED_TOTAL_SIZE),
        },
        shards,
        tensors: global,
        index_aliases: Vec::new(),
        index_alias_size_accounting: Some(IndexAliasSizeAccounting::Physical),
        indexed_tensor_bytes: tensor_bytes,
        tensor_bytes,
        artifact_bytes,
    }
}

/// Hashes canonical index name/shard assignments in sorted-name order.
fn index_inventory_digest(index: &PinnedIndex) -> String {
    let mut canonical = Vec::new();
    for (name, shard) in &index.weight_map.0 {
        canonical.extend_from_slice(name.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(shard.as_bytes());
        canonical.push(b'\n');
    }
    sha256_hex(&canonical)
}

/// Hashes every exact physical header fact in sorted-name order.
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

/// Hashes sorted `path NUL bytes NUL payload_sha256 LF` artifact identities.
fn checkpoint_artifact_manifest_digest() -> String {
    let mut artifacts = CONTROL_FILES
        .iter()
        .map(|(path, bytes, digest)| (*path, *bytes, *digest))
        .chain(
            SHARDS
                .iter()
                .map(|shard| (shard.name, shard.file_bytes, shard.payload_sha256)),
        )
        .collect::<Vec<_>>();
    artifacts.sort_by_key(|(path, _, _)| *path);
    let mut canonical = Vec::new();
    for (path, bytes, digest) in artifacts {
        canonical.extend_from_slice(path.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(bytes.to_string().as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(digest.as_bytes());
        canonical.push(b'\n');
    }
    sha256_hex(&canonical)
}

/// Hashes the executable F16 plan independently of vector/order formatting.
fn f16_plan_digest(plan: &transformers::checkpoint::LoadPlan) -> String {
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

/// Returns the exact physical dtype/shape histogram.
fn shape_histogram(manifest: &CheckpointManifest) -> BTreeMap<(TensorDType, Vec<u64>), usize> {
    let mut counts = BTreeMap::new();
    for tensor in manifest.tensors.values() {
        *counts
            .entry((tensor.dtype, tensor.shape.clone()))
            .or_default() += 1;
    }
    counts
}

/// Resolves the user-selected complete immutable artifact directory.
fn pinned_root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("TRANSFORMERS_NEMOTRON_H_DIR").unwrap_or_else(|| {
            panic!("set TRANSFORMERS_NEMOTRON_H_DIR to {REPOSITORY} revision {REVISION}")
        }),
    )
}

/// Authenticates every small local control artifact.
fn assert_control_files(root: &Path) {
    for (name, bytes, digest) in CONTROL_FILES {
        let path = root.join(name);
        let metadata = fs::symlink_metadata(&path)
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
        assert_eq!(metadata.len(), bytes, "changed control length: {name}");
        assert_eq!(
            sha256_file_hex(&path),
            digest,
            "changed control payload: {name}"
        );
    }
}

/// Authenticates all 65.8 GB of physical weight payloads.
fn assert_weight_files(root: &Path) {
    for shard in SHARDS {
        let path = root.join(shard.name);
        let metadata = fs::symlink_metadata(&path)
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
        assert_eq!(
            metadata.len(),
            shard.file_bytes,
            "changed shard length: {}",
            shard.name
        );
        assert_eq!(
            sha256_file_hex(&path),
            shard.payload_sha256,
            "changed shard payload: {}",
            shard.name
        );
    }
}

/// Creates an explicit true-valid Metal mask for the one-row smoke fixture.
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

/// Forces Metal adapter discovery before any multi-gigabyte checkpoint work.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn assert_metal_backend_available(device: &<Metal as burn::tensor::backend::BackendTypes>::Device) {
    let values = Tensor::<Metal, 1>::zeros([1], device)
        .to_data()
        .to_vec::<f32>()
        .expect("the Metal acceptance backend must expose F32 storage");
    assert_eq!(values, [0.0]);
}

/// Reads one tensor through explicit F32 storage and rejects non-finite output.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn finite_metal_values<const D: usize>(label: &str, tensor: &Tensor<Metal, D>) -> Vec<f32> {
    let values = tensor
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .unwrap_or_else(|error| panic!("read {label} as F32: {error}"));
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} contains a non-finite value"
    );
    values
}

/// Materializes the semantic last-token row from every requested hidden state.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn hidden_trace(label: &str, states: &[Tensor<Metal, 3>], sequence: usize) -> Vec<f32> {
    assert_eq!(states.len(), 54, "{label} hidden-state count");
    let mut trace = Vec::with_capacity(states.len() * 2_688);
    for (index, state) in states.iter().enumerate() {
        assert_eq!(state.dims(), [1, sequence, 2_688], "{label} row {index}");
        assert_eq!(state.dtype(), DType::F16, "{label} row {index} dtype");
        trace.extend(finite_metal_values(
            &format!("{label} row {index}"),
            &state.clone().narrow(1, sequence - 1, 1).reshape([2_688]),
        ));
    }
    trace
}

/// Hashes an F32 diagnostic in portable little-endian scalar order.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn f32_trace_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// Computes cosine similarity with finite, nonzero norm preconditions.
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

/// Borrows the concrete Nemotron state selected by automatic dispatch.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn nemotron_state(
    state: &AutoCausalDecodeState<Metal>,
) -> &transformers::models::nemotron_h::NemotronHDecodeState<Metal> {
    match state {
        AutoCausalDecodeState::NemotronH(state) => state,
        _ => panic!("automatic decode state must be Nemotron-H"),
    }
}

/// Checks exact Lightning cache topology, dtypes, and logical cursor state.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn assert_nemotron_cache_geometry(
    state: &AutoCausalDecodeState<Metal>,
    expected_length: usize,
    expected_capacity: usize,
) {
    assert_eq!(state.capacity(), expected_capacity);
    assert_eq!(state.lengths(), &[expected_length]);
    let state = nemotron_state(state);
    assert_eq!(state.attention_layer_count(), 6);
    assert_eq!(state.mamba_layer_count(), 23);
    let tensors = state.tensor_state();
    assert_eq!(tensors.main().native_kv_count(), 1);
    assert_eq!(tensors.main().recurrent_count(), 1);
    let kv = tensors
        .main()
        .native_kv(0)
        .expect("Nemotron attention KV cache");
    assert_eq!(kv.spec().layers(), 6);
    assert_eq!(kv.spec().key_value_heads(), 2);
    assert_eq!(kv.spec().head_dimension(), 128);
    assert_eq!(kv.spec().storage(), KvCacheStorage::Full);
    assert_eq!(kv.spec().dtype(), DType::F16);
    let recurrent = tensors
        .main()
        .recurrent(0)
        .expect("Nemotron Mamba recurrent cache");
    assert_eq!(recurrent.spec().layers(), 23);
    assert_eq!(recurrent.spec().state_heads(), 64);
    assert_eq!(recurrent.spec().state_rows(), 64);
    assert_eq!(recurrent.spec().state_columns(), 128);
    assert_eq!(recurrent.spec().convolution_channels(), 6_144);
    assert_eq!(recurrent.spec().convolution_width(), 3);
    assert_eq!(recurrent.spec().matrix_dtype(), DType::F32);
    assert_eq!(recurrent.spec().convolution_dtype(), DType::F16);
}

/// Strictly loads and audits the exact full Nemotron F16 graph on Metal.
///
/// Both the bounded implementation smoke and the official-reference parity
/// gate use this entry point, so artifact identity, dtype conversion, tensor
/// accounting, shard order, and host-staging requirements cannot drift between
/// the two expensive acceptance paths.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn load_pinned_nemotron_f16(
    root: &Path,
    device: &<Metal as burn::tensor::backend::BackendTypes>::Device,
) -> transformers::LoadedModel<AutoCausalLm<Metal>> {
    let loaded = AutoCausalLm::<Metal>::load_local(
        root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                index_alias_size_accounting: IndexAliasSizeAccounting::Physical,
                ..LoadOptions::default()
            },
            load_tokenizer: true,
            repository: Some(REPOSITORY.to_owned()),
            revision: Some(REVISION.to_owned()),
            ..AutoLoadOptions::default()
        },
        device,
    )
    .expect("strictly stream the exact Nemotron checkpoint into the Metal F16 graph");
    <Metal<f32, i32> as Backend>::sync(device)
        .expect("all Nemotron parameter transfers must complete");

    assert!(matches!(loaded.model(), AutoCausalLm::NemotronH(_)));
    assert_eq!(loaded.source().model_type(), "nemotron_h");
    assert_eq!(loaded.source().repository(), Some(REPOSITORY));
    assert_eq!(loaded.source().revision(), Some(REVISION));
    assert_eq!(loaded.source().root(), root);
    let report = loaded.load_report();
    assert!(report.is_strictly_valid());
    assert_eq!(report.applied.len(), PHYSICAL_TENSOR_COUNT);
    assert_eq!(report.transformed.len(), PHYSICAL_TENSOR_COUNT - 1);
    assert_eq!(report.dtype_converted.len(), BF16_TENSOR_COUNT);
    assert!(report.aliased.is_empty());
    assert!(report.allowlisted_missing.is_empty());
    assert!(report.allowlisted_ignored.is_empty());
    let execution = loaded
        .execution_stats()
        .expect("automatic loading must retain pass-two counters");
    assert_eq!(execution.tensors_applied, PHYSICAL_TENSOR_COUNT);
    assert_eq!(execution.aliases_applied, 0);
    assert_eq!(execution.source_reads, PHYSICAL_TENSOR_COUNT as u64);
    assert_eq!(execution.source_payload_bytes, PHYSICAL_TENSOR_BYTES);
    assert_eq!(execution.delivered_tensor_bytes, PHYSICAL_TENSOR_BYTES);
    assert_eq!(execution.largest_source_payload_bytes, LARGEST_TENSOR_BYTES);
    assert_eq!(execution.largest_output_tensor_bytes, LARGEST_TENSOR_BYTES);
    assert_eq!(execution.peak_host_staging_bytes, LARGEST_TENSOR_BYTES);
    assert_eq!(execution.source.shard_opens, SHARDS.len() as u64);
    assert_eq!(execution.source.tensor_reads, PHYSICAL_TENSOR_COUNT as u64);
    assert_eq!(execution.source.payload_bytes_read, PHYSICAL_TENSOR_BYTES);
    assert_eq!(execution.source.max_open_shards, 1);
    assert_eq!(
        execution.source.shard_open_order,
        SHARDS
            .iter()
            .map(|shard| shard.name.to_owned())
            .collect::<Vec<_>>()
    );
    loaded
}

/// The exact config, index, and all physical header ranges remain immutable.
#[test]
fn pinned_config_index_and_headers_have_exact_identity() {
    assert_eq!(CONFIG_BYTES.len(), 2_352);
    assert_eq!(sha256_hex(CONFIG_BYTES), CONFIG_SHA256);
    assert_eq!(INDEX_BYTES.len(), 612_207);
    assert_eq!(sha256_hex(INDEX_BYTES), INDEX_SHA256);
    assert_eq!(
        checkpoint_artifact_manifest_digest(),
        CHECKPOINT_ARTIFACT_MANIFEST_SHA256
    );

    let config = pinned_config();
    assert_eq!(config.model_type, "nemotron_h");
    assert_eq!(config.hidden_size, 2_688);
    assert_eq!(config.vocab_size, 131_072);
    assert_eq!(config.layers_block_type.len(), 52);
    assert_eq!(
        config
            .layers_block_type
            .iter()
            .filter(|kind| **kind == NemotronHBlockType::LinearAttention)
            .count(),
        23
    );
    assert_eq!(
        config
            .layers_block_type
            .iter()
            .filter(|kind| **kind == NemotronHBlockType::Moe)
            .count(),
        23
    );
    assert_eq!(
        config
            .layers_block_type
            .iter()
            .filter(|kind| **kind == NemotronHBlockType::FullAttention)
            .count(),
        6
    );
    assert_eq!(
        config.mtp_layers_block_type,
        [NemotronHBlockType::FullAttention, NemotronHBlockType::Moe]
    );
    assert_eq!(config.num_nextn_predict_layers, 1);

    let index = pinned_index();
    assert_eq!(
        index.metadata.total_parameters,
        INDEX_REPORTED_BASE_PARAMETERS
    );
    assert_eq!(index.metadata.total_size, INDEX_REPORTED_TOTAL_SIZE);
    assert_eq!(index.weight_map.0.len(), PHYSICAL_TENSOR_COUNT);
    assert_eq!(index_inventory_digest(&index), INDEX_INVENTORY_SHA256);

    let manifest = pinned_header_manifest(&index);
    assert_eq!(manifest.tensors.len(), PHYSICAL_TENSOR_COUNT);
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.starts_with("backbone."))
            .count(),
        BACKBONE_TENSOR_COUNT
    );
    assert_eq!(manifest.tensor_bytes, PHYSICAL_TENSOR_BYTES);
    assert_eq!(manifest.artifact_bytes, PHYSICAL_ARTIFACT_BYTES);
    let physical_elements = manifest
        .tensors
        .values()
        .map(|tensor| tensor.elements().unwrap())
        .sum::<u64>();
    assert_eq!(physical_elements, PHYSICAL_TENSOR_ELEMENTS);
    assert_eq!(
        physical_elements - MTP_PHYSICAL_ELEMENTS,
        BASE_PHYSICAL_ELEMENTS
    );
    assert_eq!(
        INDEX_REPORTED_BASE_PARAMETERS + BASE_ROUTER_CORRECTION_ELEMENTS,
        BASE_PHYSICAL_ELEMENTS
    );
    assert_eq!(BASE_ROUTER_CORRECTION_ELEMENTS, 23 * 128);
    assert_eq!(header_inventory_digest(&manifest), HEADER_INVENTORY_SHA256);
    assert_eq!(
        manifest
            .shards
            .iter()
            .map(|shard| shard.header_bytes + 8)
            .sum::<u64>(),
        PHYSICAL_ARTIFACT_BYTES - PHYSICAL_TENSOR_BYTES
    );
    assert!(
        manifest
            .shards
            .iter()
            .all(|shard| shard.metadata.is_empty())
    );

    let dtype_counts = manifest
        .tensors
        .values()
        .fold(BTreeMap::new(), |mut counts, tensor| {
            *counts.entry(tensor.dtype).or_insert(0_usize) += 1;
            counts
        });
    assert_eq!(
        dtype_counts,
        BTreeMap::from([
            (TensorDType::BF16, BF16_TENSOR_COUNT),
            (TensorDType::F32, F32_TENSOR_COUNT)
        ])
    );
}

/// The pre-release record repeats every immutable shard identity without promotion.
#[test]
fn acceptance_record_matches_pinned_header_evidence() {
    let record: serde_json::Value =
        serde_json::from_slice(ACCEPTANCE_BYTES).expect("acceptance record must be JSON");
    assert_eq!(record["repository"], REPOSITORY);
    assert_eq!(record["revision"], REVISION);
    assert_eq!(record["advertised_supported"], false);
    assert_eq!(
        record["artifact_layout"]["physical_tensors"],
        PHYSICAL_TENSOR_COUNT
    );
    assert_eq!(
        record["artifact_layout"]["physical_tensor_bytes"],
        PHYSICAL_TENSOR_BYTES
    );
    assert_eq!(
        record["inventory_digests"]["expected_controls_and_full_shards_manifest_sha256"],
        CHECKPOINT_ARTIFACT_MANIFEST_SHA256
    );

    let headers = record["shard_header_inventory"]
        .as_array()
        .expect("acceptance record must contain shard header identities");
    assert_eq!(headers.len(), SHARDS.len());
    for (header, shard) in headers.iter().zip(SHARDS) {
        assert_eq!(header["path"], shard.name);
        assert_eq!(header["complete_file_bytes"], shard.file_bytes);
        assert_eq!(header["header_bytes_excluding_prefix"], shard.header_bytes);
        assert_eq!(header["physical_tensors"], shard.tensors);
        assert_eq!(header["tensor_payload_bytes"], shard.data_bytes);
        assert_eq!(header["header_plus_prefix_sha256"], shard.header_sha256);
        assert_eq!(
            record["expected_weight_payload_sha256"][shard.name],
            shard.payload_sha256
        );
    }
    assert_eq!(record["checks"]["all_weight_shards_sha256_validated"], true);
    assert_eq!(
        record["checks"]["all_control_files_sha256_validated_against_local_checkpoint"],
        true
    );
    assert_eq!(record["artifact_authentication"]["status"], "passed");
    assert_eq!(
        record["artifact_authentication"]["result"]["artifact_bytes_hashed"],
        PHYSICAL_ARTIFACT_BYTES
    );
    assert_eq!(
        record["checks"]["provenance_bound_index_exception_implemented"],
        true
    );
    assert_eq!(record["checks"]["strict_real_checkpoint_load"], true);
    assert_eq!(record["checks"]["all_weights_applied"], true);
    assert_eq!(record["checks"]["metal_f16_stateless_inference"], true);
    assert_eq!(record["checks"]["metal_f16_cached_prefill"], true);
    assert_eq!(record["checks"]["metal_f16_recurrent_decode"], true);
    assert_eq!(record["checks"]["mtp_inference"], false);

    let smoke = &record["burn_metal_smoke"];
    assert_eq!(smoke["status"], "passed");
    assert_eq!(
        smoke["classification"],
        "burn_internal_consistency_smoke_not_official_reference_parity"
    );
    assert_eq!(
        smoke["source"]["crate_commit"],
        "7c1f816f520bb05c33c22774452cae434e305784"
    );
    assert_eq!(smoke["source"]["git_dirty"], false);
    assert_eq!(smoke["load_report"]["strictly_valid"], true);
    assert_eq!(
        smoke["load_report"]["applied_tensors"],
        PHYSICAL_TENSOR_COUNT
    );
    assert_eq!(
        smoke["load_report"]["source_payload_bytes"],
        PHYSICAL_TENSOR_BYTES
    );
    assert_eq!(smoke["load_report"]["shard_opens"], SHARDS.len());
    assert_eq!(
        smoke["load_report"]["maximum_simultaneously_open_shards"],
        1
    );
    assert_eq!(
        smoke["load_report"]["shard_open_order"],
        serde_json::Value::Array(
            SHARDS
                .iter()
                .map(|shard| serde_json::Value::String(shard.name.to_owned()))
                .collect::<Vec<_>>()
        )
    );
    let base = &smoke["base_inference"];
    assert_eq!(base["token_ids"], serde_json::json!(SMOKE_TOKEN_IDS));
    assert_eq!(base["hidden_trace_rows"], 54);
    assert_eq!(base["hidden_width"], 2_688);
    assert_eq!(base["first_greedy_token_id"], 1_256);
    assert_eq!(
        base["stateless_hidden_trace_f32le_sha256"],
        "80249a14decaa09f775b5e6bd469d17a5515e6c3cfc54863ef9bb93801cd2f5c"
    );
    assert_eq!(
        base["cached_prefill_hidden_trace_f32le_sha256"],
        base["stateless_hidden_trace_f32le_sha256"]
    );
    assert_eq!(
        base["recurrent_decode_hidden_trace_f32le_sha256"],
        "eac085e4be9796ba5f828ccd27e8caeda7294228333911beaf6bf33b74e4fc3c"
    );
    assert_eq!(base["recurrent_decode_teacher_forced_top1_equal"], true);
    assert_eq!(smoke["limitations"]["mtp_weights_loaded"], true);
    assert_eq!(smoke["limitations"]["mtp_inference_executed"], false);
    assert_eq!(
        smoke["limitations"]["official_transformers_base_parity"],
        false
    );
}

/// Every source key has the independently observed dtype and shape histogram.
#[test]
fn physical_header_shape_and_dtype_histogram_is_exact() {
    let manifest = pinned_header_manifest(&pinned_index());
    assert_eq!(
        shape_histogram(&manifest),
        BTreeMap::from([
            ((TensorDType::BF16, vec![64]), 69),
            ((TensorDType::F32, vec![128]), 24),
            ((TensorDType::BF16, vec![128, 2_688]), 24),
            ((TensorDType::BF16, vec![256, 2_688]), 14),
            ((TensorDType::BF16, vec![1_856, 2_688]), 3_072),
            ((TensorDType::BF16, vec![2_688]), 58),
            ((TensorDType::BF16, vec![2_688, 1_856]), 3_072),
            ((TensorDType::BF16, vec![2_688, 3_712]), 24),
            ((TensorDType::BF16, vec![2_688, 4_096]), 30),
            ((TensorDType::BF16, vec![2_688, 5_376]), 1),
            ((TensorDType::BF16, vec![3_712, 2_688]), 24),
            ((TensorDType::BF16, vec![4_096]), 23),
            ((TensorDType::BF16, vec![4_096, 2_688]), 7),
            ((TensorDType::BF16, vec![6_144]), 23),
            ((TensorDType::BF16, vec![6_144, 1, 4]), 23),
            ((TensorDType::BF16, vec![10_304, 2_688]), 23),
            ((TensorDType::BF16, vec![131_072, 2_688]), 2),
        ])
    );
}

/// The complete 270-key MTP body is isolated in shard 14 and matches its schedule.
#[test]
fn mtp_inventory_is_complete_isolated_and_matches_runtime_contract() {
    let index = pinned_index();
    let mtp = index
        .weight_map
        .0
        .iter()
        .filter(|(name, _)| name.starts_with("mtp."))
        .collect::<Vec<_>>();
    assert_eq!(mtp.len(), MTP_TENSOR_COUNT);
    assert!(
        mtp.iter()
            .all(|(_, shard)| shard.as_str() == SHARDS[13].name)
    );
    assert_eq!(
        index
            .weight_map
            .0
            .values()
            .filter(|shard| shard.as_str() == SHARDS[13].name)
            .count(),
        MTP_TENSOR_COUNT
    );

    let manifest = pinned_header_manifest(&index);
    let mtp_elements = manifest
        .tensors
        .iter()
        .filter(|(name, _)| name.starts_with("mtp."))
        .map(|(_, tensor)| tensor.elements().unwrap())
        .sum::<u64>();
    assert_eq!(mtp_elements, MTP_PHYSICAL_ELEMENTS);
    assert_eq!(manifest.shards[13].tensors.len(), MTP_TENSOR_COUNT);
    assert_eq!(manifest.shards[13].data_bytes, 2_670_652_160);
    let non_expert_mtp = manifest.shards[13]
        .tensors
        .iter()
        .filter(|(name, _)| !name.contains(".experts."))
        .map(|(name, tensor)| (name.as_str(), (tensor.dtype, tensor.shape.as_slice())))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        non_expert_mtp,
        BTreeMap::from([
            (
                "mtp.layers.0.eh_proj.weight",
                (TensorDType::BF16, &[2_688, 5_376][..]),
            ),
            (
                "mtp.layers.0.enorm.weight",
                (TensorDType::BF16, &[2_688][..]),
            ),
            (
                "mtp.layers.0.hnorm.weight",
                (TensorDType::BF16, &[2_688][..]),
            ),
            (
                "mtp.layers.0.mixer.k_proj.weight",
                (TensorDType::BF16, &[256, 2_688][..]),
            ),
            (
                "mtp.layers.0.mixer.o_proj.weight",
                (TensorDType::BF16, &[2_688, 4_096][..]),
            ),
            (
                "mtp.layers.0.mixer.q_proj.weight",
                (TensorDType::BF16, &[4_096, 2_688][..]),
            ),
            (
                "mtp.layers.0.mixer.v_proj.weight",
                (TensorDType::BF16, &[256, 2_688][..]),
            ),
            (
                "mtp.layers.0.norm.weight",
                (TensorDType::BF16, &[2_688][..]),
            ),
            (
                "mtp.layers.1.final_layernorm.weight",
                (TensorDType::BF16, &[2_688][..]),
            ),
            (
                "mtp.layers.1.mixer.gate.e_score_correction_bias",
                (TensorDType::F32, &[128][..]),
            ),
            (
                "mtp.layers.1.mixer.gate.weight",
                (TensorDType::BF16, &[128, 2_688][..]),
            ),
            (
                "mtp.layers.1.mixer.shared_experts.down_proj.weight",
                (TensorDType::BF16, &[2_688, 3_712][..]),
            ),
            (
                "mtp.layers.1.mixer.shared_experts.up_proj.weight",
                (TensorDType::BF16, &[3_712, 2_688][..]),
            ),
            (
                "mtp.layers.1.norm.weight",
                (TensorDType::BF16, &[2_688][..]),
            ),
        ])
    );
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.starts_with("mtp.layers.0."))
            .count(),
        8
    );
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.starts_with("mtp.layers.1."))
            .count(),
        262
    );
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.contains("mtp.layers.1.mixer.experts.")
                && name.ends_with(".up_proj.weight"))
            .count(),
        128
    );
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.contains("mtp.layers.1.mixer.experts.")
                && name.ends_with(".down_proj.weight"))
            .count(),
        128
    );

    let config = pinned_config();
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    let mapped_sources = contract
        .mappings()
        .operations()
        .iter()
        .filter_map(|operation| match operation {
            TensorOperation::Map { source, .. } => Some(source.as_str()),
            TensorOperation::Alias { .. } => None,
            _ => panic!("Nemotron's pinned mapping may contain only direct maps and aliases"),
        })
        .collect::<BTreeSet<_>>();
    let physical_sources = manifest
        .tensors
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(mapped_sources, physical_sources);
    assert_eq!(contract.expected_tensors().len(), PHYSICAL_TENSOR_COUNT);
}

/// The upstream total-size delta is exactly the 23 unique base router matrices.
#[test]
fn index_total_size_overcounts_only_base_router_matrices() {
    let index = pinned_index();
    let manifest = pinned_header_manifest(&index);
    let bytes_assigned_once_by_index = index
        .weight_map
        .0
        .keys()
        .map(|name| manifest.tensors[name].byte_len)
        .sum::<u64>();
    assert_eq!(bytes_assigned_once_by_index, PHYSICAL_TENSOR_BYTES);
    let base_router_names = manifest
        .tensors
        .keys()
        .filter(|name| name.starts_with("backbone.layers.") && name.ends_with(".mixer.gate.weight"))
        .collect::<Vec<_>>();
    assert_eq!(base_router_names.len(), 23);
    assert_eq!(base_router_names.iter().collect::<BTreeSet<_>>().len(), 23);
    assert!(base_router_names.iter().all(|name| {
        let tensor = &manifest.tensors[*name];
        tensor.dtype == TensorDType::BF16
            && tensor.shape == [128, 2_688]
            && tensor.byte_len == 688_128
    }));
    let base_router_bytes = base_router_names
        .iter()
        .map(|name| manifest.tensors[*name].byte_len)
        .sum::<u64>();
    assert_eq!(base_router_bytes, INDEX_TOTAL_SIZE_DELTA);
    assert_eq!(
        INDEX_REPORTED_TOTAL_SIZE - PHYSICAL_TENSOR_BYTES,
        base_router_bytes
    );
    assert_eq!(
        INDEX_REPORTED_TOTAL_SIZE,
        bytes_assigned_once_by_index + base_router_bytes
    );

    let mtp_router = manifest
        .tensors
        .get("mtp.layers.1.mixer.gate.weight")
        .expect("MTP router must be physical");
    assert_eq!(mtp_router.byte_len, 688_128);
    assert_ne!(
        INDEX_REPORTED_TOTAL_SIZE - PHYSICAL_TENSOR_BYTES,
        base_router_bytes + mtp_router.byte_len
    );
    assert_eq!(
        index
            .weight_map
            .0
            .keys()
            .filter(|name| name.ends_with(".mixer.gate.weight"))
            .count(),
        24
    );

    let config = pinned_config();
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        index_alias_size_accounting: IndexAliasSizeAccounting::Physical,
        ..LoadOptions::default()
    };
    assert!(matches!(
        contract.build_load_plan(&manifest, &options),
        Err(CheckpointError::TotalSizeMismatch {
            declared: INDEX_REPORTED_TOTAL_SIZE,
            observed: PHYSICAL_TENSOR_BYTES
        })
    ));
}

/// Index parsing rejects duplicate assignments and undeclared metadata fields.
#[test]
fn pinned_index_schema_is_fail_closed() {
    let duplicate = br#"{
        "metadata":{"total_parameters":1,"total_size":2},
        "weight_map":{"tensor":"first.safetensors","tensor":"second.safetensors"}
    }"#;
    assert!(serde_json::from_slice::<PinnedIndex>(duplicate).is_err());

    let unknown = br#"{
        "metadata":{"total_parameters":1,"total_size":2,"trusted":true},
        "weight_map":{}
    }"#;
    assert!(serde_json::from_slice::<PinnedIndex>(unknown).is_err());
}

/// Corrected metadata yields an otherwise strict, complete F16 execution plan.
#[test]
fn corrected_metadata_produces_complete_f16_plan() {
    let config = pinned_config();
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    let mut manifest = pinned_header_manifest(&pinned_index());
    manifest.source = ArtifactSource::Index {
        relative_path: "model.safetensors.index.json".into(),
        declared_total_size: Some(PHYSICAL_TENSOR_BYTES),
    };
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        index_alias_size_accounting: IndexAliasSizeAccounting::Physical,
        ..LoadOptions::default()
    };
    let plan = contract
        .build_load_plan(&manifest, &options)
        .expect("only upstream total-size accounting blocks the exact plan");
    assert_eq!(plan.parameter_count(), PHYSICAL_TENSOR_COUNT);
    assert!(plan.aliases.is_empty());
    assert!(plan.report.is_strictly_valid());
    assert_eq!(plan.report.dtype_converted.len(), BF16_TENSOR_COUNT);
    assert_eq!(plan.report.transformed.len(), PHYSICAL_TENSOR_COUNT - 1);
    assert!(
        plan.tensors
            .iter()
            .all(|tensor| matches!(tensor.transform, PlannedTransform::Identity))
    );
    assert_eq!(
        plan.tensors
            .iter()
            .filter(|tensor| tensor.dtype == TensorDType::F16)
            .count(),
        BF16_TENSOR_COUNT
    );
    assert_eq!(
        plan.tensors
            .iter()
            .filter(|tensor| tensor.dtype == TensorDType::F32)
            .count(),
        F32_TENSOR_COUNT
    );
    assert_eq!(plan.memory.artifact_bytes, PHYSICAL_ARTIFACT_BYTES);
    assert_eq!(plan.memory.checkpoint_tensor_bytes, PHYSICAL_TENSOR_BYTES);
    assert_eq!(plan.memory.target_parameter_bytes, PHYSICAL_TENSOR_BYTES);
    assert_eq!(plan.memory.logical_model_bytes, PHYSICAL_TENSOR_BYTES);
    assert_eq!(plan.memory.largest_shard_bytes, LARGEST_SHARD_BYTES);
    assert_eq!(
        plan.memory.largest_source_tensor_bytes,
        LARGEST_TENSOR_BYTES
    );
    assert_eq!(
        plan.memory.largest_target_tensor_bytes,
        LARGEST_TENSOR_BYTES
    );
    assert_eq!(plan.memory.peak_host_staging_bytes, PEAK_HOST_STAGING_BYTES);
    assert_eq!(f16_plan_digest(&plan), F16_LOAD_PLAN_SHA256);
}

/// The committed adversarial component oracle is source-bound and exact.
#[test]
fn adversarial_component_oracle_is_strict_and_source_bound() {
    nemotron_h_reference::validate_committed_synthetic_oracle();
}

/// The independently reviewed Transformers base-oracle anchors are complete.
#[test]
fn official_base_reference_anchors_are_complete() {
    assert_eq!(
        nemotron_h_base_reference::trusted_reference_hashes().len(),
        7
    );
}

/// MTP evidence remains impossible to accept before a separate vLLM review.
#[test]
fn official_mtp_reference_bundle_is_fail_closed_before_anchor_review() {
    assert!(nemotron_h_reference::trusted_reference_hashes().is_err());
}

/// Authenticates all local control and weight payloads without allocating tensors.
#[test]
#[ignore = "requires all 65.8 GB of nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9"]
fn validates_pinned_nemotron_control_and_weight_payloads() {
    let supplied = pinned_root();
    let root = supplied
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied.display()));
    assert!(root.is_dir());
    assert_control_files(&root);
    assert_weight_files(&root);
    eprintln!(
        "Nemotron 3.5 artifacts authenticated: repository={REPOSITORY}, revision={REVISION}, shards=14, bytes={PHYSICAL_ARTIFACT_BYTES}"
    );
}

/// Streams every base and MTP parameter in physical source order, then exercises
/// the base LM's stateless, cached-prefill, and one-token recurrent paths.
///
/// This is an implementation smoke, not official-reference parity. In
/// particular, the loaded 270-tensor MTP body is not invoked here because its
/// dedicated cache and speculative verification protocol remain release gates.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
#[test]
#[ignore = "requires authenticated Nemotron 3.5 weights, Metal, and about 80 GiB free unified memory"]
fn loads_pinned_nemotron_f16_and_smokes_base_inference() {
    let started = Instant::now();
    let supplied = pinned_root();
    let root = supplied
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied.display()));
    let device = Default::default();
    assert_metal_backend_available(&device);
    assert_control_files(&root);
    assert_weight_files(&root);

    let load_started = Instant::now();
    let loaded = load_pinned_nemotron_f16(&root, &device);
    let load_seconds = load_started.elapsed().as_secs_f64();
    let execution = loaded
        .execution_stats()
        .expect("automatic loading must retain pass-two counters");

    let tokenizer = loaded
        .tokenizer()
        .expect("the exact local tokenizer must remain attached");
    assert_eq!(tokenizer.root(), root);
    let encoding = tokenizer
        .encode(
            SMOKE_PROMPT,
            &transformers::EncodeOptions {
                add_special_tokens: false,
                ..transformers::EncodeOptions::default()
            },
        )
        .expect("tokenize the pinned Nemotron smoke prompt");
    assert_eq!(encoding.token_ids()[0], SMOKE_TOKEN_IDS);
    let prompt_ids = SMOKE_TOKEN_IDS
        .iter()
        .copied()
        .map(i64::from)
        .collect::<Vec<_>>();
    let prompt_length = prompt_ids.len();
    let prompt = Tensor::<Metal, 2, Int>::from_data(
        TensorData::new(prompt_ids.clone(), [1, prompt_length]),
        &device,
    );
    let prompt_mask = all_valid_metal_mask(prompt_length, &device);
    let diagnostics = ForwardOptions {
        hidden_states: true,
        ..ForwardOptions::default()
    };

    let stateless_started = Instant::now();
    let stateless = loaded
        .model()
        .forward(
            CausalLmInput {
                token_ids: prompt.clone(),
                token_mask: Some(prompt_mask.clone()),
                position_ids: None,
            },
            &diagnostics,
        )
        .expect("run stateless Nemotron base inference");
    <Metal<f32, i32> as Backend>::sync(&device).expect("complete stateless inference");
    let stateless_seconds = stateless_started.elapsed().as_secs_f64();
    assert_eq!(stateless.logits.dims(), [1, 1, 131_072]);
    assert_eq!(stateless.logits.dtype(), DType::F32);
    let stateless_logits = finite_metal_values("Nemotron stateless logits", &stateless.logits);
    let stateless_hidden = hidden_trace(
        "Nemotron stateless prompt hidden trace",
        &stateless.hidden_states,
        prompt_length,
    );

    let capacity = prompt_length + 1;
    let mut state = loaded
        .model()
        .new_decode_state(
            CacheConfig::new(1, capacity).expect("valid one-row smoke cache"),
            1,
            &device,
        )
        .expect("allocate exact Nemotron hybrid cache");
    assert_nemotron_cache_geometry(&state, 0, capacity);
    let prefill_started = Instant::now();
    let prefill = loaded
        .model()
        .prefill(
            CausalLmInput {
                token_ids: prompt,
                token_mask: Some(prompt_mask),
                position_ids: None,
            },
            &mut state,
            &diagnostics,
        )
        .expect("run cached Nemotron base prefill");
    <Metal<f32, i32> as Backend>::sync(&device).expect("complete cached prefill");
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    assert_nemotron_cache_geometry(&state, prompt_length, capacity);
    assert_eq!(prefill.logits.dims(), [1, 1, 131_072]);
    assert_eq!(prefill.logits.dtype(), DType::F32);
    let prefill_logits = finite_metal_values("Nemotron cached-prefill logits", &prefill.logits);
    let prefill_hidden = hidden_trace(
        "Nemotron cached-prefill hidden trace",
        &prefill.hidden_states,
        prompt_length,
    );
    let prompt_logit_cosine = cosine_similarity(&stateless_logits, &prefill_logits);
    let prompt_hidden_cosine = cosine_similarity(&stateless_hidden, &prefill_hidden);
    assert!(
        prompt_logit_cosine >= 0.999,
        "stateless/prefill logit cosine {prompt_logit_cosine} is below 0.999"
    );
    assert!(
        prompt_hidden_cosine >= 0.999,
        "stateless/prefill hidden cosine {prompt_hidden_cosine} is below 0.999"
    );
    let next_token = prefill
        .logits
        .clone()
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("Nemotron top-1 must use i32 storage")[0];
    assert!(next_token >= 0);

    let decode_started = Instant::now();
    let decoded = loaded
        .model()
        .decode(
            DecodeInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(vec![i64::from(next_token)], [1, 1]),
                    &device,
                ),
                token_mask: Some(all_valid_metal_mask(1, &device)),
                position_ids: None,
            },
            &mut state,
            &diagnostics,
        )
        .expect("run one recurrent Nemotron decode step");
    <Metal<f32, i32> as Backend>::sync(&device).expect("complete recurrent decode");
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    assert_nemotron_cache_geometry(&state, capacity, capacity);
    assert_eq!(decoded.logits.dims(), [1, 1, 131_072]);
    assert_eq!(decoded.logits.dtype(), DType::F32);
    let decode_logits = finite_metal_values("Nemotron recurrent decode logits", &decoded.logits);
    let decode_hidden = hidden_trace(
        "Nemotron recurrent decode hidden trace",
        &decoded.hidden_states,
        1,
    );

    let mut teacher_ids = prompt_ids;
    teacher_ids.push(i64::from(next_token));
    let teacher = loaded
        .model()
        .forward(
            CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(teacher_ids, [1, capacity]),
                    &device,
                ),
                token_mask: Some(all_valid_metal_mask(capacity, &device)),
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect("run teacher-forced Nemotron prefix");
    let teacher_logits = finite_metal_values("Nemotron teacher logits", &teacher.logits);
    let decode_teacher_cosine = cosine_similarity(&decode_logits, &teacher_logits);
    assert!(
        decode_teacher_cosine >= 0.999,
        "recurrent/teacher logit cosine {decode_teacher_cosine} is below 0.999"
    );
    let decoded_top1 = decoded
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("decoded top-1 must use i32 storage")[0];
    let teacher_top1 = teacher
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("teacher top-1 must use i32 storage")[0];
    assert_eq!(decoded_top1, teacher_top1);

    eprintln!(
        "Nemotron 3.5 Burn Metal smoke: tensors_applied={}, source_bytes={}, \
         shard_order={:?}, load_seconds={load_seconds:.3}, \
         stateless_seconds={stateless_seconds:.3}, prefill_seconds={prefill_seconds:.3}, \
         decode_seconds={decode_seconds:.3}, first_token={next_token}, \
         stateless_hidden_sha256={}, prefill_hidden_sha256={}, decode_hidden_sha256={}, \
         stateless_prefill_logit_cosine={prompt_logit_cosine:.9}, \
         stateless_prefill_hidden_cosine={prompt_hidden_cosine:.9}, \
         decode_teacher_logit_cosine={decode_teacher_cosine:.9}, total_seconds={:.3}",
        execution.tensors_applied,
        execution.source_payload_bytes,
        execution.source.shard_open_order,
        f32_trace_sha256(&stateless_hidden),
        f32_trace_sha256(&prefill_hidden),
        f32_trace_sha256(&decode_hidden),
        started.elapsed().as_secs_f64(),
    );
}

/// Compares every pinned base-model row with Burn on the same Mamba path.
///
/// The stateless half evaluates the literal prompt and one 37-token
/// teacher-forced chunk, selecting the 32 causal decision positions retained by
/// the official stateless oracle. The cached half evaluates the prompt prefill
/// followed by 31 one-token recurrent updates. Typed rows make a
/// cached-versus-teacher comparison fail before cosine is computed.
///
/// Logit rows are judged against the official class's F32 rows of the same
/// path from the precision bundle (`TRANSFORMERS_NEMOTRON_H_PRECISION_DIR`).
/// Each row needs exact top-1 and a cosine at least the path's official F16
/// floor: the worst cosine that the MPS/F16 oracle or the official CPU/F16
/// evaluation reaches against F32 on that path, computed here from the
/// authenticated rows (prompt `0.999194`, teacher `0.994466`, cached `0.998635`).
/// A Burn F16 row therefore passes exactly when it is at least as faithful to
/// F32 as the official F16 rows are; a `0.999` floor against the F16 oracle was
/// unreachable even for the oracle, which is `0.994466` from F32 at teacher
/// decision 18. Cosines against the F16 oracle are still reported. The three
/// available hidden traces, recorded only at F16, keep the `0.999` threshold at
/// their exact stateless/prefill/decode semantic points.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
#[test]
#[ignore = "requires the pinned Nemotron checkpoint and base oracle, Metal, and about 80 GiB free unified memory"]
fn compares_pinned_nemotron_f16_with_official_base_reference() {
    use nemotron_h_base_reference::{
        OfficialHiddenPoint, REFERENCE_DECISIONS, REFERENCE_HIDDEN_ROWS, REFERENCE_HIDDEN_SIZE,
        REFERENCE_VOCABULARY_SIZE,
    };
    use nemotron_h_parity::{
        TruthPathSummary, assert_corresponding_hidden_trace, assert_row_against_truth,
        half_precision_floor,
    };
    use nemotron_h_precision_reference::{Precision, RowKind};

    let started = Instant::now();
    let reference = nemotron_h_base_reference::load_and_validate_reference_bundle();
    assert_eq!(reference.prompt_token_ids(), SMOKE_TOKEN_IDS);
    assert_eq!(reference.generation_token_ids().len(), REFERENCE_DECISIONS);
    let generated = reference
        .generation_token_ids()
        .iter()
        .map(|&token| token as usize)
        .collect::<Vec<_>>();

    // The F32 truth and each path's official F16 floor, from authenticated rows only.
    let precision = nemotron_h_precision_reference::load_and_validate_precision_bundle(
        reference.generation_token_ids(),
    );
    let oracle_prompt = reference.stateless_prompt_row().values().to_vec();
    let oracle_matrix = |row: &dyn Fn(usize) -> Vec<f32>| {
        (0..REFERENCE_DECISIONS).flat_map(row).collect::<Vec<f32>>()
    };
    let oracle_teacher = oracle_matrix(&|decision| {
        reference
            .teacher_row(decision)
            .expect("official teacher row")
            .values()
            .to_vec()
    });
    let oracle_cached = oracle_matrix(&|decision| {
        reference
            .cached_row(decision)
            .expect("official cached row")
            .values()
            .to_vec()
    });
    let floor = |kind: RowKind, oracle: &[f32]| {
        half_precision_floor(
            "Nemotron official F16 floor",
            precision.matrix(Precision::Float32, kind),
            &[oracle, precision.matrix(Precision::Float16, kind)],
            REFERENCE_VOCABULARY_SIZE,
        )
    };
    let prompt_floor = floor(RowKind::Prompt, &oracle_prompt);
    let teacher_floor = floor(RowKind::Teacher, &oracle_teacher);
    let cached_floor = floor(RowKind::Cached, &oracle_cached);
    let oracle_row = |matrix: &[f32], decision: usize| {
        matrix[decision * REFERENCE_VOCABULARY_SIZE..(decision + 1) * REFERENCE_VOCABULARY_SIZE]
            .to_vec()
    };

    let supplied = pinned_root();
    let root = supplied
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied.display()));
    let device = Default::default();
    assert_metal_backend_available(&device);
    assert_control_files(&root);
    assert_weight_files(&root);

    let load_started = Instant::now();
    let loaded = load_pinned_nemotron_f16(&root, &device);
    let load_seconds = load_started.elapsed().as_secs_f64();
    let tokenizer = loaded
        .tokenizer()
        .expect("official parity requires the exact local tokenizer");
    let encoding = tokenizer
        .encode(
            SMOKE_PROMPT,
            &transformers::EncodeOptions {
                add_special_tokens: false,
                ..transformers::EncodeOptions::default()
            },
        )
        .expect("tokenize the pinned official-reference prompt");
    assert_eq!(encoding.token_ids()[0], reference.prompt_token_ids());

    let prompt_ids = reference
        .prompt_token_ids()
        .iter()
        .copied()
        .map(i64::from)
        .collect::<Vec<_>>();
    let prompt_length = prompt_ids.len();
    let prompt = Tensor::<Metal, 2, Int>::from_data(
        TensorData::new(prompt_ids.clone(), [1, prompt_length]),
        &device,
    );
    let prompt_mask = all_valid_metal_mask(prompt_length, &device);
    let hidden_options = ForwardOptions {
        hidden_states: true,
        ..ForwardOptions::default()
    };

    let stateless_started = Instant::now();
    let stateless = loaded
        .model()
        .forward(
            CausalLmInput {
                token_ids: prompt.clone(),
                token_mask: Some(prompt_mask.clone()),
                position_ids: None,
            },
            &hidden_options,
        )
        .expect("run Burn stateless prompt for official parity");
    assert_eq!(stateless.logits.dims(), [1, 1, REFERENCE_VOCABULARY_SIZE]);
    assert_eq!(stateless.logits.dtype(), DType::F32);
    let stateless_logits = finite_metal_values(
        "Nemotron official-parity stateless prompt logits",
        &stateless.logits,
    );
    assert_eq!(stateless_logits.len(), REFERENCE_VOCABULARY_SIZE);
    let stateless_prompt = assert_row_against_truth(
        "Nemotron stateless prompt",
        RowKind::Prompt,
        &stateless_logits,
        precision.truth(RowKind::Prompt, 0),
        generated[0],
        prompt_floor,
    );
    let stateless_prompt_oracle = cosine_similarity(&stateless_logits, &oracle_prompt);
    let stateless_hidden = hidden_trace(
        "Nemotron official-parity stateless hidden trace",
        &stateless.hidden_states,
        prompt_length,
    );
    assert_eq!(
        stateless_hidden.len(),
        REFERENCE_HIDDEN_ROWS * REFERENCE_HIDDEN_SIZE
    );
    let stateless_hidden_summary = assert_corresponding_hidden_trace(
        "Nemotron stateless prompt hidden trace",
        OfficialHiddenPoint::StatelessPrompt,
        &stateless_hidden,
        reference.stateless_hidden_trace(),
    );
    let stateless_prompt_seconds = stateless_started.elapsed().as_secs_f64();
    drop(stateless);
    drop(stateless_logits);

    // A single causal chunk produces every stateless teacher-forced decision.
    // Position `prompt_length - 1 + d` predicts generated token `d`.
    let mut teacher_ids = prompt_ids;
    teacher_ids.extend(
        reference
            .generation_token_ids()
            .iter()
            .take(REFERENCE_DECISIONS - 1)
            .copied()
            .map(i64::from),
    );
    let teacher_length = teacher_ids.len();
    assert_eq!(teacher_length, prompt_length + REFERENCE_DECISIONS - 1);
    let teacher_started = Instant::now();
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
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .expect("run Burn teacher-forced chunk for official stateless parity");
    assert_eq!(
        teacher.logits.dims(),
        [1, teacher_length, REFERENCE_VOCABULARY_SIZE]
    );
    assert_eq!(teacher.logits.dtype(), DType::F32);
    let teacher_values = finite_metal_values(
        "Nemotron official-parity teacher-forced logits",
        &teacher.logits,
    );
    let mut stateless_summary = TruthPathSummary::new(RowKind::Teacher, teacher_floor);
    let mut teacher_oracle_minimum = f64::INFINITY;
    for (decision, &expected) in generated.iter().enumerate() {
        let position = prompt_length - 1 + decision;
        let start = position * REFERENCE_VOCABULARY_SIZE;
        let end = start + REFERENCE_VOCABULARY_SIZE;
        stateless_summary.observe(
            "Nemotron teacher-forced row",
            &teacher_values[start..end],
            precision.truth(RowKind::Teacher, decision),
            expected,
        );
        teacher_oracle_minimum = teacher_oracle_minimum.min(cosine_similarity(
            &teacher_values[start..end],
            &oracle_row(&oracle_teacher, decision),
        ));
    }
    let stateless_summary = stateless_summary.finish(REFERENCE_DECISIONS);
    let teacher_seconds = teacher_started.elapsed().as_secs_f64();
    drop(teacher);
    drop(teacher_values);

    let mut cached_summary = TruthPathSummary::new(RowKind::Cached, cached_floor);
    let mut cached_oracle_minimum = f64::INFINITY;
    let mut state = loaded
        .model()
        .new_decode_state(
            CacheConfig::new(1, teacher_length).expect("valid official parity cache"),
            1,
            &device,
        )
        .expect("allocate the exact Nemotron hybrid parity cache");
    assert_nemotron_cache_geometry(&state, 0, teacher_length);

    let cached_started = Instant::now();
    let prefill = loaded
        .model()
        .prefill(
            CausalLmInput {
                token_ids: prompt,
                token_mask: Some(prompt_mask),
                position_ids: None,
            },
            &mut state,
            &hidden_options,
        )
        .expect("run Burn cached prefill for official parity");
    assert_nemotron_cache_geometry(&state, prompt_length, teacher_length);
    assert_eq!(prefill.logits.dims(), [1, 1, REFERENCE_VOCABULARY_SIZE]);
    assert_eq!(prefill.logits.dtype(), DType::F32);
    let prefill_logits = finite_metal_values(
        "Nemotron official-parity cached-prefill logits",
        &prefill.logits,
    );
    cached_summary.observe(
        "Nemotron cached-prefill row",
        &prefill_logits,
        precision.truth(RowKind::Cached, 0),
        generated[0],
    );
    cached_oracle_minimum = cached_oracle_minimum.min(cosine_similarity(
        &prefill_logits,
        &oracle_row(&oracle_cached, 0),
    ));
    let cached_prefill_hidden = hidden_trace(
        "Nemotron official-parity cached-prefill hidden trace",
        &prefill.hidden_states,
        prompt_length,
    );
    let cached_prefill_hidden_summary = assert_corresponding_hidden_trace(
        "Nemotron cached-prefill hidden trace",
        OfficialHiddenPoint::CachedPrefill,
        &cached_prefill_hidden,
        reference.cached_prefill_hidden_trace(),
    );
    drop(prefill);
    drop(prefill_logits);

    let mut cached_decode_hidden = None;
    let mut cached_decode_hidden_summary = None;
    for (decision, &expected) in generated.iter().enumerate().skip(1) {
        let capture_hidden = decision == 1;
        let decoded = loaded
            .model()
            .decode(
                DecodeInput {
                    token_ids: Tensor::<Metal, 2, Int>::from_data(
                        TensorData::new(
                            vec![i64::from(reference.generation_token_ids()[decision - 1])],
                            [1, 1],
                        ),
                        &device,
                    ),
                    token_mask: Some(all_valid_metal_mask(1, &device)),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions {
                    hidden_states: capture_hidden,
                    ..ForwardOptions::default()
                },
            )
            .unwrap_or_else(|error| panic!("run official cached decision {decision}: {error}"));
        assert_nemotron_cache_geometry(&state, prompt_length + decision, teacher_length);
        assert_eq!(decoded.logits.dims(), [1, 1, REFERENCE_VOCABULARY_SIZE]);
        assert_eq!(decoded.logits.dtype(), DType::F32);
        let decoded_logits =
            finite_metal_values("Nemotron official-parity recurrent logits", &decoded.logits);
        cached_summary.observe(
            "Nemotron recurrent cached row",
            &decoded_logits,
            precision.truth(RowKind::Cached, decision),
            expected,
        );
        cached_oracle_minimum = cached_oracle_minimum.min(cosine_similarity(
            &decoded_logits,
            &oracle_row(&oracle_cached, decision),
        ));
        if capture_hidden {
            let trace = hidden_trace(
                "Nemotron official-parity first recurrent hidden trace",
                &decoded.hidden_states,
                1,
            );
            cached_decode_hidden_summary = Some(assert_corresponding_hidden_trace(
                "Nemotron first recurrent-decode hidden trace",
                OfficialHiddenPoint::CachedDecodeOne,
                &trace,
                reference.cached_decode_hidden_trace(),
            ));
            cached_decode_hidden = Some(trace);
        } else {
            assert!(
                decoded.hidden_states.is_empty(),
                "later cached decisions must not retain hidden-state diagnostics"
            );
        }
    }
    let cached_summary = cached_summary.finish(REFERENCE_DECISIONS);
    let cached_seconds = cached_started.elapsed().as_secs_f64();
    let cached_decode_hidden =
        cached_decode_hidden.expect("the first recurrent hidden trace must be captured");
    let cached_decode_hidden_summary =
        cached_decode_hidden_summary.expect("the first recurrent hidden summary must be captured");
    assert_eq!(state.lengths(), &[teacher_length]);

    eprintln!(
        "Nemotron 3.5 official corresponding-path parity: load_seconds={load_seconds:.3}, \
         stateless_prompt_seconds={stateless_prompt_seconds:.3}, teacher_seconds={teacher_seconds:.3}, \
         cached_seconds={cached_seconds:.3}, f16_floors[prompt,teacher,cached]=[{prompt_floor:.9}, {teacher_floor:.9}, {cached_floor:.9}], \
         f32_truth_prompt_cosine={:.12}, f32_truth_teacher_minimum={:.12}@{}, f32_truth_cached_minimum={:.12}@{}, \
         f16_oracle_prompt_cosine={stateless_prompt_oracle:.12}, f16_oracle_teacher_minimum={teacher_oracle_minimum:.12}, \
         f16_oracle_cached_minimum={cached_oracle_minimum:.12}, \
         stateless_hidden_minimum_cosine={:.12}@{}, \
         cached_prefill_hidden_minimum_cosine={:.12}@{}, \
         cached_decode_hidden_minimum_cosine={:.12}@{}, \
         stateless_hidden_sha256={}, cached_prefill_hidden_sha256={}, \
         cached_decode_hidden_sha256={}, official_crossed_path_minimum_diagnostic={:.12}, \
         decisions_per_path={}, total_seconds={:.3}",
        stateless_prompt.cosine,
        stateless_summary.minimum_cosine,
        stateless_summary.minimum_decision,
        cached_summary.minimum_cosine,
        cached_summary.minimum_decision,
        stateless_hidden_summary.minimum_cosine,
        stateless_hidden_summary.minimum_boundary,
        cached_prefill_hidden_summary.minimum_cosine,
        cached_prefill_hidden_summary.minimum_boundary,
        cached_decode_hidden_summary.minimum_cosine,
        cached_decode_hidden_summary.minimum_boundary,
        f32_trace_sha256(&stateless_hidden),
        f32_trace_sha256(&cached_prefill_hidden),
        f32_trace_sha256(&cached_decode_hidden),
        reference.minimum_official_path_cosine,
        REFERENCE_DECISIONS,
        started.elapsed().as_secs_f64(),
    );
}

/// Decision whose official hidden trace localizes the recurrent/chunked split.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const DIAGNOSED_DECISION: usize = 14;

/// Number of semantic hidden boundaries: embedding, 52 blocks, final norm.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const HIDDEN_BOUNDARIES: usize = 54;

/// Lightning residual width.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const HIDDEN_WIDTH: usize = 2_688;

/// One Burn cached run: every decision's logits and one decision's hidden trace.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
struct CachedDiagnosticRun {
    /// Full-vocabulary rows for every reference decision.
    logits: Vec<Vec<f32>>,
    /// All [`HIDDEN_BOUNDARIES`] rows at [`DIAGNOSED_DECISION`], boundary-major.
    hidden: Vec<f32>,
}

/// One cached run measured against the Burn teacher-forced chunk.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
struct CrossedPathSummary {
    /// Minimum full-vocabulary logit cosine over all decisions.
    minimum: f64,
    /// Decision attaining [`Self::minimum`] (first on ties).
    minimum_decision: usize,
    /// Logit cosine at [`DIAGNOSED_DECISION`].
    diagnosed: f64,
    /// Decisions whose top-1 tokens agree.
    top1_agreements: usize,
    /// Hidden cosine at every boundary of [`DIAGNOSED_DECISION`].
    hidden: Vec<f64>,
}

#[cfg(all(feature = "metal", feature = "tokenizers"))]
impl CrossedPathSummary {
    /// Returns the boundary with the lowest hidden cosine and that cosine.
    fn worst_boundary(&self) -> (usize, f64) {
        first_minimum(self.hidden.iter().copied())
    }
}

/// Returns the first index attaining the minimum value, and that value.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn first_minimum(values: impl IntoIterator<Item = f64>) -> (usize, f64) {
    values
        .into_iter()
        .enumerate()
        .fold((0, f64::INFINITY), |best, (index, value)| {
            if value < best.1 { (index, value) } else { best }
        })
}

/// Runs chunked prefill and 31 one-token updates, keeping every decision row.
///
/// The token stream is the official greedy continuation, so cached decision
/// `d` consumes generated token `d - 1` exactly as the oracle's cached path
/// did. Only [`DIAGNOSED_DECISION`] retains its hidden trace.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn run_cached_diagnostic(
    model: &AutoCausalLm<Metal>,
    prompt_ids: &[i64],
    generation_ids: &[u32],
    device: &<Metal as burn::tensor::backend::BackendTypes>::Device,
) -> CachedDiagnosticRun {
    use nemotron_h_base_reference::{REFERENCE_DECISIONS, REFERENCE_VOCABULARY_SIZE};

    let prompt_length = prompt_ids.len();
    let mut state = model
        .new_decode_state(
            CacheConfig::new(1, prompt_length + REFERENCE_DECISIONS - 1)
                .expect("valid diagnostic cache"),
            1,
            device,
        )
        .expect("allocate the Nemotron diagnostic cache");
    let prefill = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(prompt_ids.to_vec(), [1, prompt_length]),
                    device,
                ),
                token_mask: Some(all_valid_metal_mask(prompt_length, device)),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("run diagnostic cached prefill");
    assert_eq!(prefill.logits.dims(), [1, 1, REFERENCE_VOCABULARY_SIZE]);
    let mut logits = vec![finite_metal_values(
        "Nemotron diagnostic prefill logits",
        &prefill.logits,
    )];
    let mut hidden = None;
    for decision in 1..REFERENCE_DECISIONS {
        let capture_hidden = decision == DIAGNOSED_DECISION;
        let decoded = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::<Metal, 2, Int>::from_data(
                        TensorData::new(vec![i64::from(generation_ids[decision - 1])], [1, 1]),
                        device,
                    ),
                    token_mask: Some(all_valid_metal_mask(1, device)),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions {
                    hidden_states: capture_hidden,
                    ..ForwardOptions::default()
                },
            )
            .unwrap_or_else(|error| panic!("run diagnostic decision {decision}: {error}"));
        assert_eq!(decoded.logits.dims(), [1, 1, REFERENCE_VOCABULARY_SIZE]);
        logits.push(finite_metal_values(
            "Nemotron diagnostic recurrent logits",
            &decoded.logits,
        ));
        if capture_hidden {
            hidden = Some(hidden_trace(
                "Nemotron diagnostic recurrent hidden trace",
                &decoded.hidden_states,
                1,
            ));
        }
    }
    CachedDiagnosticRun {
        logits,
        hidden: hidden.expect("the diagnosed decision must retain its hidden trace"),
    }
}

/// Returns the top-1 index of one full-vocabulary row (first maximum on ties).
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn top1(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 { (index, value) } else { best }
        })
        .0
}

/// Measures one cached run against the Burn teacher-forced rows and hidden trace.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn crossed_path_summary(
    cached: &CachedDiagnosticRun,
    teacher_rows: &[Vec<f32>],
    teacher_hidden: &[f32],
) -> CrossedPathSummary {
    let cosines = cached
        .logits
        .iter()
        .zip(teacher_rows)
        .map(|(cached, teacher)| cosine_similarity(cached, teacher))
        .collect::<Vec<_>>();
    let (minimum_decision, minimum) = first_minimum(cosines.iter().copied());
    CrossedPathSummary {
        minimum,
        minimum_decision,
        diagnosed: cosines[DIAGNOSED_DECISION],
        top1_agreements: cached
            .logits
            .iter()
            .zip(teacher_rows)
            .filter(|(cached, teacher)| top1(cached) == top1(teacher))
            .count(),
        hidden: cached
            .hidden
            .as_chunks::<HIDDEN_WIDTH>()
            .0
            .iter()
            .zip(teacher_hidden.as_chunks::<HIDDEN_WIDTH>().0)
            .map(|(cached, teacher)| cosine_similarity(cached, teacher))
            .collect(),
    }
}

/// Separates the two candidate causes of the official decision-14 divergence.
///
/// In the authenticated oracle the recurrent cached path and the chunked
/// teacher path keep identical top-1 tokens, but their hidden states at
/// decision 14 first diverge after Mamba layer 0 (cosine `0.892`) although the
/// embedding is identical. The pinned reference treats the two paths
/// differently in two ways: its chunk scan clamps the discretization step
/// `Δ = softplus(dt + dt_bias)` to `[time_step_min, ∞)` while its one-token
/// update does not, and its F16 one-token update rounds `(ΔB)x`, the `C`
/// contraction, and the `D` skip in the activation dtype. This diagnostic
/// measures both and asserts neither outcome:
///
/// 1. **Δ census.** Mamba layer 0 reads `RMSNorm(embedding)`, a per-token
///    function shared exactly by both paths, so its `Δ` does not depend on the
///    path. The stateless hook captures that input; the layer's own fused
///    projection, `dt_bias`, the reference softplus (`x > 20 ? x :
///    log1p(exp(x))` in F32, rounded once to F16), and the clamp then give
///    every token's 64 head steps. An entry counts when the clamp changes it,
///    which is exactly where the two official paths compute different `Δ`.
/// 2. **Clamp counterfactual.** The crate reproduces each reference path,
///    including the F16 one-token rounding, and [`TimeStepClamp`] states the
///    clamp as data. Re-running the cached path with the one-token update
///    clamped too changes only the clamp, so the cached-versus-teacher distance
///    that disappears is the clamp's and the distance that remains is the
///    precision policy's.
///
/// 3. **Official profiles.** Every Burn decision row (teacher, pinned cached,
///    and clamped cached) is compared with both official streams, so a
///    corresponding-path failure is seen in full rather than at its first row.
/// 4. **Router margins.** The hooked teacher pass also captures every MoE
///    block's input; re-running its router gives, per position, the gap
///    between the sixth and seventh selection scores. A gap within F16 noise
///    means an expert choice can flip between two faithful implementations.
///
/// Both cached runs share the same chunked prefill, so their decision-0 rows
/// must be bitwise identical; only one-token updates may differ.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
#[test]
#[ignore = "requires the pinned Nemotron checkpoint and base oracle, Metal, and about 80 GiB free unified memory"]
fn diagnoses_pinned_nemotron_decision_14_time_step_clamp() {
    use nemotron_h_base_reference::{REFERENCE_DECISIONS, REFERENCE_VOCABULARY_SIZE};
    use transformers::{
        intervention::{Hooks, Layer},
        mamba2::{TimeStepClamp, TimeStepLimits},
        operations,
    };

    let started = Instant::now();
    let reference = nemotron_h_base_reference::load_and_validate_reference_bundle();
    assert_eq!(reference.prompt_token_ids(), SMOKE_TOKEN_IDS);
    let generation_ids = reference.generation_token_ids().to_vec();
    assert_eq!(generation_ids.len(), REFERENCE_DECISIONS);

    let supplied = pinned_root();
    let root = supplied
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied.display()));
    let device = Default::default();
    assert_metal_backend_available(&device);
    assert_control_files(&root);
    assert_weight_files(&root);
    let mut loaded = load_pinned_nemotron_f16(&root, &device);

    let prompt_ids = SMOKE_TOKEN_IDS
        .iter()
        .copied()
        .map(i64::from)
        .collect::<Vec<_>>();
    let prompt_length = prompt_ids.len();
    let mut teacher_ids = prompt_ids.clone();
    teacher_ids.extend(
        generation_ids
            .iter()
            .take(REFERENCE_DECISIONS - 1)
            .copied()
            .map(i64::from),
    );
    let teacher_length = teacher_ids.len();
    // Teacher position `prompt_length - 1 + d` predicts generated token `d`.
    let diagnosed_position = prompt_length - 1 + DIAGNOSED_DECISION;

    // ---- Δ census and the Burn teacher path from one hooked stateless pass.
    let AutoCausalLm::NemotronH(model) = loaded.model() else {
        panic!("automatic dispatch must select Nemotron-H");
    };
    let time_step_min = model.model.config.time_step_min;
    let mamba = model.model.layers[0]
        .mamba
        .as_ref()
        .expect("Lightning block 0 is a Mamba-2 mixer");
    let pinned_clamp = TimeStepClamp::chunked_only(TimeStepLimits::at_least(time_step_min));
    assert_eq!(mamba.time_step_clamp, pinned_clamp);
    let heads = mamba.num_heads;

    let mut hooks = Hooks::new();
    let mixer_input = hooks.capture(Layer::decoder(0).mixer_input());
    let moe_inputs = model
        .model
        .layers
        .iter()
        .enumerate()
        .filter(|(_, block)| block.moe.is_some())
        .map(|(index, _)| (index, hooks.capture(Layer::decoder(index).mlp_input())))
        .collect::<Vec<_>>();
    assert_eq!(moe_inputs.len(), 23);
    let teacher = model
        .forward_with_hooks(
            CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(teacher_ids, [1, teacher_length]),
                    &device,
                ),
                token_mask: Some(all_valid_metal_mask(teacher_length, &device)),
                position_ids: None,
            },
            &ForwardOptions {
                hidden_states: true,
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
            &mut hooks,
        )
        .expect("run the hooked Burn teacher-forced chunk");
    let input = hooks
        .get(&mixer_input)
        .expect("Mamba layer 0 input must be captured");
    assert_eq!(input.dims(), [1, teacher_length, HIDDEN_WIDTH]);
    assert_eq!(input.dtype(), DType::F16);

    // The mixer's own tiling `[z | xBC | dt]`: `dt` is the last `heads` channels.
    let projected = operations::linear(input, &mamba.in_proj).expect("project Mamba layer 0 input");
    let dt_start = heads * mamba.head_dim + mamba.convolution_channels;
    assert_eq!(projected.dims()[2], dt_start + heads);
    let dt = projected.narrow(2, dt_start, heads);
    let dtype = dt.dtype();
    let biased = (dt + mamba.dt_bias.val().cast(dtype).reshape([1, 1, heads])).cast(DType::F32);
    let steps = biased
        .clone()
        .exp()
        .log1p()
        .mask_where(biased.clone().greater_elem(20.0), biased)
        .cast(dtype);
    let clamped = steps.clone().clamp_min(time_step_min);
    let steps = finite_metal_values("Nemotron layer-0 discretization steps", &steps);
    let clamped = finite_metal_values("Nemotron clamped layer-0 steps", &clamped);
    assert_eq!(steps.len(), teacher_length * heads);
    let changed_per_token = steps
        .chunks_exact(heads)
        .zip(clamped.chunks_exact(heads))
        .map(|(steps, clamped)| {
            steps
                .iter()
                .zip(clamped)
                .filter(|(step, clamped)| step != clamped)
                .count()
        })
        .collect::<Vec<_>>();
    // Prompt tokens run the chunked scan on both official paths; generated
    // tokens through the diagnosed decision run one-token updates when cached.
    let recurrent_positions = prompt_length..=diagnosed_position;
    let recurrent_changed = changed_per_token[recurrent_positions.clone()]
        .iter()
        .sum::<usize>();
    let recurrent_entries = recurrent_positions.clone().count() * heads;
    let all_changed = changed_per_token.iter().sum::<usize>();
    let minimum_step = steps.iter().copied().fold(f32::INFINITY, f32::min);
    let recurrent_minimum_step = steps[prompt_length * heads..(diagnosed_position + 1) * heads]
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let changed_per_recurrent_token = recurrent_positions
        .clone()
        .map(|position| {
            (
                generation_ids[position - prompt_length],
                changed_per_token[position],
            )
        })
        .collect::<BTreeMap<_, _>>();

    // ---- Router selection margins of the Burn teacher pass at every MoE block.
    // Lightning routes one group, so selection is a plain top-k of
    // `sigmoid(z) + correction`; the gap between the k-th and (k+1)-th score
    // is how far F16 noise must move one logit to change an expert.
    let mut narrowest_selections = Vec::new();
    let mut minimum_margin_per_position = vec![f32::INFINITY; teacher_length];
    for (index, capture) in &moe_inputs {
        let moe = model.model.layers[*index]
            .moe
            .as_ref()
            .expect("captured block is MoE");
        assert_eq!(
            moe.groups, moe.top_groups,
            "Lightning routes without group limits"
        );
        let routed = moe
            .forward(hooks.get(capture).expect("MoE input must be captured"))
            .expect("re-run the captured MoE routing");
        let logits = finite_metal_values("Nemotron router logits", &routed.router_logits);
        let correction =
            finite_metal_values("Nemotron router correction", &moe.correction_bias.val());
        let k = moe.experts_per_token;
        for (position, row) in logits.chunks_exact(correction.len()).enumerate() {
            let mut scores = row
                .iter()
                .zip(&correction)
                .map(|(&logit, &correction)| 1.0 / (1.0 + (-logit).exp()) + correction)
                .collect::<Vec<f32>>();
            scores.sort_by(|left, right| right.total_cmp(left));
            let margin = scores[k - 1] - scores[k];
            minimum_margin_per_position[position] =
                minimum_margin_per_position[position].min(margin);
            narrowest_selections.push((margin, *index, position));
        }
    }
    narrowest_selections.sort_by(|left, right| left.0.total_cmp(&right.0));
    narrowest_selections.truncate(8);

    assert_eq!(
        teacher.logits.dims(),
        [1, teacher_length, REFERENCE_VOCABULARY_SIZE]
    );
    let teacher_values = finite_metal_values("Nemotron diagnostic teacher logits", &teacher.logits);
    let teacher_rows = (0..REFERENCE_DECISIONS)
        .map(|decision| {
            let start = (prompt_length - 1 + decision) * REFERENCE_VOCABULARY_SIZE;
            teacher_values[start..start + REFERENCE_VOCABULARY_SIZE].to_vec()
        })
        .collect::<Vec<_>>();
    assert_eq!(teacher.hidden_states.len(), HIDDEN_BOUNDARIES);
    let mut teacher_hidden = Vec::with_capacity(HIDDEN_BOUNDARIES * HIDDEN_WIDTH);
    for (index, state) in teacher.hidden_states.iter().enumerate() {
        assert_eq!(state.dims(), [1, teacher_length, HIDDEN_WIDTH]);
        teacher_hidden.extend(finite_metal_values(
            &format!("Nemotron diagnostic teacher hidden {index}"),
            &state
                .clone()
                .narrow(1, diagnosed_position, 1)
                .reshape([HIDDEN_WIDTH]),
        ));
    }
    drop(teacher);
    drop(teacher_values);

    // ---- The cached path as pinned: unclamped one-token updates.
    let pinned = run_cached_diagnostic(loaded.model(), &prompt_ids, &generation_ids, &device);

    // ---- Counterfactual: clamp the one-token updates too; nothing else changes.
    let AutoCausalLm::NemotronH(model) = loaded.model_mut() else {
        panic!("automatic dispatch must select Nemotron-H");
    };
    let mut mamba_layers = 0;
    for block in &mut model.model.layers {
        if let Some(mamba) = block.mamba.as_mut() {
            assert_eq!(mamba.time_step_clamp, pinned_clamp);
            mamba.time_step_clamp = TimeStepClamp::both(TimeStepLimits::at_least(time_step_min));
            mamba_layers += 1;
        }
    }
    assert_eq!(mamba_layers, 23);
    let clamped_run = run_cached_diagnostic(loaded.model(), &prompt_ids, &generation_ids, &device);
    assert_eq!(
        pinned.logits[0], clamped_run.logits[0],
        "chunked prefill must not depend on the one-token clamp"
    );

    let pinned_summary = crossed_path_summary(&pinned, &teacher_rows, &teacher_hidden);
    let clamped_summary = crossed_path_summary(&clamped_run, &teacher_rows, &teacher_hidden);

    // ---- Every decision against both official streams, without stopping at a failure.
    let official_teacher = (0..REFERENCE_DECISIONS)
        .map(|decision| {
            reference
                .teacher_row(decision)
                .expect("official teacher row")
                .values()
                .to_vec()
        })
        .collect::<Vec<_>>();
    let official_cached = (0..REFERENCE_DECISIONS)
        .map(|decision| {
            reference
                .cached_row(decision)
                .expect("official cached row")
                .values()
                .to_vec()
        })
        .collect::<Vec<_>>();
    let profiles = [
        (
            "Burn teacher vs official teacher (corresponding)",
            &teacher_rows,
            &official_teacher,
        ),
        (
            "Burn pinned cached vs official cached (corresponding)",
            &pinned.logits,
            &official_cached,
        ),
        (
            "Burn teacher vs official cached (crossed)",
            &teacher_rows,
            &official_cached,
        ),
        (
            "Burn pinned cached vs official teacher (crossed)",
            &pinned.logits,
            &official_teacher,
        ),
        (
            "Burn clamped cached vs official teacher",
            &clamped_run.logits,
            &official_teacher,
        ),
        (
            "Burn clamped cached vs official cached",
            &clamped_run.logits,
            &official_cached,
        ),
        (
            "official teacher vs official cached",
            &official_teacher,
            &official_cached,
        ),
    ];
    for (label, burn, official) in profiles {
        let cosines = burn
            .iter()
            .zip(official.iter())
            .map(|(burn, official)| cosine_similarity(burn, official))
            .collect::<Vec<_>>();
        let (minimum_decision, minimum) = first_minimum(cosines.iter().copied());
        eprintln!(
            "Nemotron official profile ({label}): minimum_cosine={minimum:.9}@{minimum_decision}, \
             below_0.999={}, top1_agreements={}/{REFERENCE_DECISIONS}, cosines={cosines:.6?}",
            cosines.iter().filter(|&&cosine| cosine < 0.999).count(),
            burn.iter()
                .zip(official.iter())
                .filter(|(burn, official)| top1(burn) == top1(official))
                .count(),
        );
    }
    eprintln!(
        "Nemotron Burn teacher router selection: minimum_margin_per_position={:?}, \
         narrowest(margin,layer,position)={:?}",
        minimum_margin_per_position
            .iter()
            .map(|margin| format!("{margin:.2e}"))
            .collect::<Vec<_>>(),
        narrowest_selections
            .iter()
            .map(|(margin, layer, position)| format!("({margin:.2e},{layer},{position})"))
            .collect::<Vec<_>>(),
    );
    eprintln!(
        "Nemotron decision-{DIAGNOSED_DECISION} Δ census at Mamba layer 0: \
         time_step_min={time_step_min}, heads={heads}, tokens={teacher_length}, \
         clamp_changed_all={all_changed}/{}, \
         clamp_changed_recurrent={recurrent_changed}/{recurrent_entries} (positions {prompt_length}..={diagnosed_position}), \
         changed_per_token={changed_per_token:?}, \
         changed_per_recurrent_token={changed_per_recurrent_token:?}, \
         minimum_step={minimum_step:e}, recurrent_minimum_step={recurrent_minimum_step:e}",
        teacher_length * heads,
    );
    for (label, summary) in [
        ("pinned unclamped one-token updates", &pinned_summary),
        ("one-token updates clamped too", &clamped_summary),
    ] {
        let (worst_boundary, worst_hidden) = summary.worst_boundary();
        eprintln!(
            "Nemotron Burn cached-versus-teacher ({label}): \
             minimum_logit_cosine={:.12}@{}, decision_{DIAGNOSED_DECISION}_logit_cosine={:.12}, \
             top1_agreements={}/{REFERENCE_DECISIONS}, \
             decision_{DIAGNOSED_DECISION}_hidden_cosine[embedding,block0,block1,block2]={:.12?}, \
             worst_hidden_cosine={worst_hidden:.12}@{worst_boundary}",
            summary.minimum,
            summary.minimum_decision,
            summary.diagnosed,
            summary.top1_agreements,
            &summary.hidden[..4],
        );
    }
    eprintln!(
        "Nemotron official crossed-path minimum={:.12}, total_seconds={:.3}",
        reference.minimum_official_path_cosine,
        started.elapsed().as_secs_f64(),
    );
}

/// Authenticates the independently reviewed Transformers base-model oracle.
#[test]
#[ignore = "requires the independently reviewed external Nemotron base-reference bundle"]
fn validates_pinned_nemotron_official_base_reference_bundle() {
    let reference = nemotron_h_base_reference::load_and_validate_reference_bundle();
    eprintln!(
        "Nemotron base reference authenticated: outputs={}, prompt_tokens={}, generation_tokens={}, minimum_path_cosine={}",
        reference.output_count,
        reference.prompt_tokens,
        reference.generation_tokens,
        reference.minimum_official_path_cosine,
    );
}

/// A future combined target-plus-MTP bundle remains gated separately.
#[test]
#[ignore = "requires separately reviewed pinned vLLM Nemotron MTP evidence"]
fn validates_pinned_nemotron_official_mtp_reference_bundle() {
    let reference = nemotron_h_reference::load_and_validate_reference_bundle();
    eprintln!(
        "Nemotron combined reference authenticated: outputs={}, prompt_tokens={}, synthetic_cases={}",
        reference.output_count, reference.prompt_tokens, reference.synthetic_cases,
    );
}
