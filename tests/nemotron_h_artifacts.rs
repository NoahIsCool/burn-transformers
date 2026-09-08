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
//! of the discrepancy. Generic checkpoint discovery must therefore continue to
//! reject this artifact until a provenance-bound family exception is supplied.
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
//! The separately produced numerical oracle is deliberately fail-closed. Its
//! JSON and every binary payload require independent SHA-256 anchors before the
//! ignored reader will accept the bundle.

#[path = "support/nemotron_h_reference.rs"]
mod nemotron_h_reference;

use serde::{
    Deserialize, Deserializer,
    de::{Error as _, MapAccess, Visitor},
};
use sha2::{Digest, Sha256};
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
struct UniqueWeightMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for UniqueWeightMap {
    /// Parses an index map without serde_json's usual last-value-wins behavior.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueWeightMapVisitor;

        impl<'de> Visitor<'de> for UniqueWeightMapVisitor {
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
    assert_eq!(
        record["checks"]["all_weight_shards_sha256_validated"],
        false
    );
    assert_eq!(record["checks"]["strict_real_checkpoint_load"], false);
    assert_eq!(record["checks"]["mtp_inference"], false);
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

/// Real numerical evidence remains impossible to accept before review pins hashes.
#[test]
fn official_reference_bundle_is_fail_closed_before_anchor_review() {
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

/// Authenticates a separately generated base-plus-MTP numerical reference bundle.
#[test]
#[ignore = "requires an independently reviewed external Nemotron official-reference bundle"]
fn validates_pinned_nemotron_official_reference_bundle() {
    let reference = nemotron_h_reference::load_and_validate_reference_bundle();
    eprintln!(
        "Nemotron reference authenticated: outputs={}, prompt_tokens={}, synthetic_cases={}",
        reference.output_count, reference.prompt_tokens, reference.synthetic_cases,
    );
}
