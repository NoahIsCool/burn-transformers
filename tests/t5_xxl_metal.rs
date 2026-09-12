//! Tensor-level Metal acceptance for the immutable official FLAN-T5-XXL graph.
//!
//! The reference bundle is generated offline by the pinned Python workflow and
//! contains only about 9 MiB of F32 outputs. The inexpensive provenance gate
//! validates that bundle without loading model weights:
//!
//! ```text
//! TRANSFORMERS_FLAN_T5_XXL_REFERENCE_DIR=/path/to/reference.bundle \
//! cargo test --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,tokenizers \
//!   --test t5_xxl_metal validates_pinned_flan_t5_xxl_reference_bundle \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! The full gate strictly streams the 45 GB F32 checkpoint into Burn Metal,
//! evaluates encoder, stateless decoder, cached decoder, greedy, and beam paths,
//! and compares them with Transformers 5.2.0 F32 outputs:
//!
//! ```text
//! TRANSFORMERS_FLAN_T5_XXL_DIR=/path/to/google/flan-t5-xxl \
//! TRANSFORMERS_FLAN_T5_XXL_REFERENCE_DIR=/path/to/reference.bundle \
//! cargo test --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,metal,tokenizers \
//!   --test t5_xxl_metal loads_flan_t5_xxl_f32_and_matches_official_reference \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```
//!
//! The full test needs a macOS Metal adapter and approximately 70 GiB of free
//! unified memory for 45 GB of parameters, transient activations, a 62 MiB
//! greedy cache, and the four-beam verification pass. It performs no network I/O.

#![cfg(feature = "tokenizers")]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[cfg(all(feature = "metal", target_os = "macos"))]
use std::time::Instant;

#[cfg(all(feature = "metal", target_os = "macos"))]
use burn::{
    backend::Metal,
    tensor::{Bool, DType, Int, Tensor, TensorData, backend::Backend},
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use transformers::{
    AutoLoadOptions, AutoSeq2SeqLm, CausalLmInput, DecodeInput, DecodeState, EncodeOptions,
    FinishReason, ForwardOptions, LogitsSelection, Seq2SeqInput, Seq2SeqLanguageModel, TokenMask,
    cache::CacheConfig,
    checkpoint::{
        CheckpointEquivalenceRecord, DTypePolicy, ExecutionStats, LoadOptions, LoadReport,
        ResolvedIndexTensorAlias,
    },
    models::t5::{T5BeamSearchOptions, generate_t5_beam_search},
};

/// Immutable model repository certified by the reference and Metal gates.
const REPOSITORY: &str = "google/flan-t5-xxl";

/// Exact Git revision used for checkpoint and reference generation.
const REVISION: &str = "ae7c9136adc7555eeccc78cdd960dfd60fb346ce";

/// Stable long prompt used for tensor-level encoder and decoder parity.
const LONG_PROMPT: &str = "translate English to German: In the early morning, the scientist walked slowly through the quiet garden, carefully recording how sunlight touched every leaf, how birds moved between the branches, and how the cool air changed as the city began to wake around her.";

/// Exact SentencePiece IDs for [`LONG_PROMPT`], including terminal EOS.
const LONG_INPUT_IDS: [u32; 51] = [
    13_959, 1_566, 12, 2_968, 10, 86, 8, 778, 1_379, 6, 8, 17_901, 3, 6_412, 5_665, 190, 8, 4_354,
    2_004, 6, 4_321, 5_592, 149, 16_285, 15_880, 334, 8_384, 6, 149, 6_331, 2_301, 344, 8, 9_678,
    6, 11, 149, 8, 1_633, 799, 2_130, 38, 8, 690, 1_553, 12, 7_178, 300, 160, 5, 1,
];

/// Stable short prompt used by the official four-beam generation oracle.
const TRANSLATION_PROMPT: &str = "translate English to German: The house is wonderful.";

/// Exact SentencePiece IDs for [`TRANSLATION_PROMPT`], including terminal EOS.
const TRANSLATION_INPUT_IDS: [u32; 11] = [13_959, 1_566, 12, 2_968, 10, 37, 629, 19, 1_627, 5, 1];

/// Exact 32-token non-EOS greedy continuation from the pinned F32 reference.
const GREEDY_TOKEN_IDS: [u32; 32] = [
    86, 74, 13_558, 35, 20_147, 31_743, 3, 7, 235, 32, 17, 74, 17_589, 1_171, 20_194, 732, 177, 3,
    31_622, 10_469, 6, 3, 18_152, 15, 3, 49, 3, 11_059, 26, 1_411, 6, 587,
];

/// First greedy token, exposed separately for const-compatible slice storage.
const FIRST_GREEDY_TOKEN_ID: [u32; 1] = [GREEDY_TOKEN_IDS[0]];

/// Exact EOS-terminated Transformers four-beam continuation.
const BEAM_TOKEN_IDS: [u32; 6] = [644, 4_598, 229, 19_250, 5, 1];

/// SHA-256 of [`GREEDY_TOKEN_IDS`] encoded as canonical little-endian `u32` values.
#[cfg(all(feature = "metal", target_os = "macos"))]
const GREEDY_TOKEN_IDS_SHA256: &str =
    "39e7bb643f90ca0127d2f4401912774f3648fcce3ea96c5df3ff1ee0fe543008";

/// SHA-256 of [`BEAM_TOKEN_IDS`] encoded as canonical little-endian `u32` values.
#[cfg(all(feature = "metal", target_os = "macos"))]
const BEAM_TOKEN_IDS_SHA256: &str =
    "b818fc36304dd2cc12d37ba110ef2afa8500525fb5a2917f6776dc16a3303352";

/// Physical opens from equivalence preflight followed by source-ordered loading.
#[cfg(all(feature = "metal", target_os = "macos"))]
const EXPECTED_SHARD_OPEN_ORDER: [&str; 7] = [
    "model-00001-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
];

/// Immutable reference-manifest filename within the create-only bundle.
const REFERENCE_JSON: &str = "flan-t5-xxl-ae7c9136-reference.json";

/// Exact byte identity of the complete reference manifest.
const REFERENCE_JSON_SHA256: &str =
    "430c2b4f4891e518ee717e6bcd1a8b53d1200d41a596105b0ced96d8ba16582f";

/// Exact reference-manifest byte length.
const REFERENCE_JSON_BYTES: u64 = 54_555;

/// One bit-exact scalar probe in a raw F32 tensor.
#[derive(Debug, Clone, Copy)]
struct Probe {
    /// Flattened row-major scalar index.
    index: usize,
    /// Exact F32 value serialized at [`Self::index`].
    value: f32,
}

/// Immutable shape, content identity, and semantic checks for one raw tensor.
#[derive(Debug)]
struct ReferenceTensorSpec {
    /// JSON key below `reference_tensors`.
    key: &'static str,
    /// Create-only filename relative to the bundle root.
    filename: &'static str,
    /// Exact raw byte count.
    bytes: u64,
    /// SHA-256 over raw little-endian F32 bytes.
    sha256: &'static str,
    /// Row-major tensor shape.
    shape: &'static [usize],
    /// Sparse scalars that make endianness and indexing mistakes obvious.
    probes: &'static [Probe],
    /// Exact top-one token for every vocabulary row, when applicable.
    row_top1: Option<&'static [u32]>,
}

/// Every raw output emitted by the immutable reference workflow.
const REFERENCE_TENSORS: [ReferenceTensorSpec; 5] = [
    ReferenceTensorSpec {
        key: "encoder_final_hidden_state",
        filename: "flan-t5-xxl-ae7c9136-encoder-final.f32le",
        bytes: 835_584,
        sha256: "b1fbaec8d4a4d6d880d45f84df4b1481eb6b6b3bda202ad6d398956b40e22c78",
        shape: &[1, 51, 4_096],
        probes: &[
            Probe {
                index: 0,
                value: 0.003_795_907,
            },
            Probe {
                index: 1,
                value: 0.015_822_137,
            },
            Probe {
                index: 4_095,
                value: 0.005_481_692_5,
            },
            Probe {
                index: 208_895,
                value: 0.002_231_959_3,
            },
        ],
        row_top1: None,
    },
    ReferenceTensorSpec {
        key: "decoder_first_final_hidden_state",
        filename: "flan-t5-xxl-ae7c9136-decoder-first-final.f32le",
        bytes: 16_384,
        sha256: "560a1a8bccd175ae64b919fafdbfd92b34c2d3ec41f2e891a8fd6f238a1f9269",
        shape: &[1, 1, 4_096],
        probes: &[
            Probe {
                index: 0,
                value: -0.102_755_345,
            },
            Probe {
                index: 1,
                value: -0.015_245_788,
            },
            Probe {
                index: 100,
                value: -0.056_452_613,
            },
            Probe {
                index: 4_095,
                value: 0.117_333_38,
            },
        ],
        row_top1: None,
    },
    ReferenceTensorSpec {
        key: "first_logits",
        filename: "flan-t5-xxl-ae7c9136-first-logits.f32le",
        bytes: 128_512,
        sha256: "6f9e1b3f69d856a73eee8dda9e1c4fc54c2dc603b2671339222202a47a7650cc",
        shape: &[32_128],
        probes: &[
            Probe {
                index: 0,
                value: -34.331_436,
            },
            Probe {
                index: 1,
                value: 0.961_295_25,
            },
            Probe {
                index: 2,
                value: -3.006_852_9,
            },
            Probe {
                index: 100,
                value: -0.319_698_8,
            },
            Probe {
                index: 1_000,
                value: -3.715_967,
            },
            Probe {
                index: 10_000,
                value: -6.986_790_7,
            },
            Probe {
                index: 32_127,
                value: -34.430_786,
            },
            Probe {
                index: 86,
                value: 10.037_275,
            },
        ],
        row_top1: Some(&FIRST_GREEDY_TOKEN_ID),
    },
    ReferenceTensorSpec {
        key: "cached_greedy_logits",
        filename: "flan-t5-xxl-ae7c9136-cached-logits-32x32128.f32le",
        bytes: 4_112_384,
        sha256: "7884d5b65ea3ccc596858812cca0f9e67cb7061debabeb1d2b99fcc8ab484756",
        shape: &[32, 32_128],
        probes: &[
            Probe {
                index: 0,
                value: -34.331_436,
            },
            Probe {
                index: 32_127,
                value: -34.430_786,
            },
            Probe {
                index: 32_128,
                value: -31.808_966,
            },
            Probe {
                index: 514_048,
                value: -37.988_35,
            },
            Probe {
                index: 1_028_095,
                value: -36.938_37,
            },
        ],
        row_top1: Some(&GREEDY_TOKEN_IDS),
    },
    ReferenceTensorSpec {
        key: "teacher_forced_logits",
        filename: "flan-t5-xxl-ae7c9136-teacher-logits-32x32128.f32le",
        bytes: 4_112_384,
        sha256: "8d0b0b554d6dd61e4ec1b03f484acf15b12f87cfcad0d6fc28aec5d2018ae8eb",
        shape: &[32, 32_128],
        probes: &[
            Probe {
                index: 0,
                value: -34.331_474,
            },
            Probe {
                index: 32_127,
                value: -34.430_813,
            },
            Probe {
                index: 32_128,
                value: -31.808_985,
            },
            Probe {
                index: 514_048,
                value: -37.988_37,
            },
            Probe {
                index: 1_028_095,
                value: -36.938_362,
            },
        ],
        row_top1: Some(&GREEDY_TOKEN_IDS),
    },
];

/// Fully authenticated host-side reference values used by the Metal gate.
struct ReferenceBundle {
    /// Final encoder hidden states for the 51-token long prompt.
    encoder_final: Vec<f32>,
    /// Final decoder hidden state after its start token.
    decoder_first_final: Vec<f32>,
    /// Vocabulary logits after the decoder start token.
    first_logits: Vec<f32>,
    /// Thirty-two cached greedy vocabulary rows.
    cached_logits: Vec<f32>,
    /// Thirty-two stateless teacher-forced vocabulary rows.
    teacher_logits: Vec<f32>,
}

/// Resolves and validates the explicitly selected create-only reference root.
fn reference_root() -> PathBuf {
    let supplied = PathBuf::from(
        std::env::var_os("TRANSFORMERS_FLAN_T5_XXL_REFERENCE_DIR").expect(
            "set TRANSFORMERS_FLAN_T5_XXL_REFERENCE_DIR to the pinned FLAN-T5-XXL reference.bundle",
        ),
    );
    let metadata = fs::symlink_metadata(&supplied)
        .unwrap_or_else(|error| panic!("inspect {}: {error}", supplied.display()));
    assert!(metadata.is_dir(), "reference root must be a directory");
    assert!(
        !metadata.file_type().is_symlink(),
        "reference root must not be a symlink"
    );
    supplied
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied.display()))
}

/// Resolves the explicitly selected local checkpoint root for the Metal gate.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn checkpoint_root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("TRANSFORMERS_FLAN_T5_XXL_DIR").unwrap_or_else(|| {
            panic!("set TRANSFORMERS_FLAN_T5_XXL_DIR to {REPOSITORY} revision {REVISION}")
        }),
    )
}

/// Streams one regular file into SHA-256 without a shard-sized allocation.
fn sha256_file_hex(path: &Path, expected_bytes: u64) -> String {
    let metadata = fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()));
    assert!(
        !metadata.file_type().is_symlink(),
        "{} must not be a symlink",
        path.display()
    );
    assert!(metadata.is_file(), "{} must be a file", path.display());
    assert_eq!(
        metadata.len(),
        expected_bytes,
        "{} has a changed byte count",
        path.display()
    );
    let mut file =
        File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .unwrap_or_else(|error| panic!("hash {}: {error}", path.display()));
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    format!("{:x}", digest.finalize())
}

/// Hashes host F32 values in canonical row-major little-endian representation.
///
/// Burn materializes the tensors in logical row-major order. Encoding each
/// scalar explicitly makes the recorded digest independent of host endianness
/// while preserving the exact IEEE-754 bit pattern produced by Metal.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn f32le_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// Hashes token IDs in canonical row-major little-endian `u32` representation.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn u32le_sha256(values: &[u32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// Checked product used by analytical cache-memory accounting.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn checked_u64_product(label: &str, factors: &[usize]) -> u64 {
    factors.iter().fold(1_u64, |product, &factor| {
        let factor = u64::try_from(factor)
            .unwrap_or_else(|_| panic!("{label} factor {factor} does not fit u64"));
        product
            .checked_mul(factor)
            .unwrap_or_else(|| panic!("{label} byte count overflows u64"))
    })
}

/// Analytical payload sizes for T5 decoder self- and cross-attention caches.
///
/// Every decoder layer owns a key and value tensor. Both use
/// `num_heads * d_kv` scalars per sequence position, so the self-cache grows by
/// the same number of bytes per target token that the immutable cross-cache
/// consumes per source token.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AnalyticalT5CacheBytes {
    /// Preallocated self-KV payload for one target token across every layer.
    self_per_target_token: u64,
    /// Complete self-KV payload at the requested target capacity.
    self_total: u64,
    /// Immutable encoder cross-KV payload across every decoder layer.
    cross_total: u64,
    /// Sum of self- and cross-KV tensor payloads.
    total: u64,
}

/// Derives T5 cache payload bytes from runtime model and fixture geometry.
///
/// This deliberately reports tensor payload only. Backend allocation metadata,
/// retained encoder hidden states, masks, logical lengths, and temporary
/// attention workspaces are not cache payload and are not included.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn analytical_t5_cache_bytes(
    decoder_layers: usize,
    batch_size: usize,
    attention_heads: usize,
    head_dimension: usize,
    source_sequence: usize,
    target_capacity: usize,
    element_bytes: usize,
) -> AnalyticalT5CacheBytes {
    let self_per_target_token = checked_u64_product(
        "T5 self-KV bytes per target token",
        &[
            2,
            decoder_layers,
            batch_size,
            attention_heads,
            head_dimension,
            element_bytes,
        ],
    );
    let self_total = self_per_target_token
        .checked_mul(u64::try_from(target_capacity).expect("target capacity must fit u64"))
        .expect("T5 self-KV byte count must fit u64");
    let cross_total = self_per_target_token
        .checked_mul(u64::try_from(source_sequence).expect("source sequence must fit u64"))
        .expect("T5 cross-KV byte count must fit u64");
    let total = self_total
        .checked_add(cross_total)
        .expect("T5 total cache payload must fit u64");
    AnalyticalT5CacheBytes {
        self_per_target_token,
        self_total,
        cross_total,
        total,
    }
}

/// Certifies every load-report and pass-two counter for the pinned F32 plan.
///
/// The expected byte totals follow from the committed plan: 558 F32 target
/// tensors deliver 44,541,329,408 bytes, while the one redundant decoder/shared
/// embedding comparison reads two additional 526,385,152-byte payloads before
/// the first model write. All target operations are unary identity or transpose
/// transforms, so each applied tensor contributes exactly one ordinary source
/// read and preserves its byte length.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn assert_pinned_f32_load_accounting(report: &LoadReport, execution: &ExecutionStats) {
    /// Number of destination tensors in the pinned strict load plan.
    const TARGET_TENSORS: usize = 558;
    /// Number of destination tensors requiring a transpose.
    const TRANSPOSED_TENSORS: usize = 433;
    /// Physical tensor payload bytes across the checkpoint shards.
    const CHECKPOINT_TENSOR_BYTES: u64 = 45_067_714_560;
    /// F32 destination parameter bytes after resolving shared embeddings.
    const TARGET_PARAMETER_BYTES: u64 = 44_541_329_408;
    /// Bytes in one complete F32 embedding matrix.
    const EMBEDDING_BYTES: u64 = 526_385_152;
    /// Bytes read when comparing the redundant and canonical embedding matrices.
    const EQUIVALENCE_PAIR_BYTES: u64 = 2 * EMBEDDING_BYTES;
    /// Total source payload bytes read for equivalence preflight and parameter loading.
    const SOURCE_PAYLOAD_BYTES: u64 = TARGET_PARAMETER_BYTES + EQUIVALENCE_PAIR_BYTES;
    /// Total payload reads, including the two embedding-equivalence reads.
    const SOURCE_READS: u64 = TARGET_TENSORS as u64 + 2;

    assert!(report.is_strictly_valid());
    assert_eq!(report.applied.len(), TARGET_TENSORS);
    assert_eq!(report.transformed.len(), TRANSPOSED_TENSORS);
    assert!(report.aliased.is_empty());
    assert_eq!(
        report.checkpoint_aliases,
        [ResolvedIndexTensorAlias::new(
            "encoder.embed_tokens.weight",
            "shared.weight",
            "model-00001-of-00005.safetensors",
        )]
    );
    let equivalence =
        CheckpointEquivalenceRecord::new("decoder.embed_tokens.weight", "shared.weight");
    assert_eq!(
        report.equivalence_planned,
        std::slice::from_ref(&equivalence)
    );
    assert_eq!(report.equivalence_verified, [equivalence]);
    assert!(report.dtype_converted.is_empty());
    assert!(report.quantized.is_empty());
    assert!(report.allowlisted_missing.is_empty());
    assert!(report.allowlisted_ignored.is_empty());
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());

    let memory = report
        .memory
        .expect("strict FLAN-T5-XXL loading must retain its pass-one memory estimate");
    assert_eq!(memory.artifact_bytes, 45_067_783_756);
    assert_eq!(memory.checkpoint_tensor_bytes, CHECKPOINT_TENSOR_BYTES);
    assert_eq!(memory.target_parameter_bytes, TARGET_PARAMETER_BYTES);
    assert_eq!(memory.logical_model_bytes, TARGET_PARAMETER_BYTES);
    assert_eq!(memory.largest_shard_bytes, 9_999_712_776);
    assert_eq!(memory.largest_source_tensor_bytes, EMBEDDING_BYTES);
    assert_eq!(memory.largest_target_tensor_bytes, EMBEDDING_BYTES);
    assert_eq!(
        memory.largest_equivalence_pair_bytes,
        EQUIVALENCE_PAIR_BYTES
    );
    assert_eq!(memory.peak_host_staging_bytes, 10_526_097_928);

    assert_eq!(execution.tensors_applied, TARGET_TENSORS);
    assert_eq!(execution.aliases_applied, 0);
    assert_eq!(execution.equivalences_verified, 1);
    assert_eq!(execution.equivalence_source_reads, 2);
    assert_eq!(
        execution.equivalence_source_payload_bytes,
        EQUIVALENCE_PAIR_BYTES
    );
    assert_eq!(execution.source_reads, SOURCE_READS);
    assert_eq!(execution.source_payload_bytes, SOURCE_PAYLOAD_BYTES);
    assert_eq!(execution.delivered_tensor_bytes, TARGET_PARAMETER_BYTES);
    assert_eq!(execution.largest_source_payload_bytes, EMBEDDING_BYTES);
    assert_eq!(execution.largest_output_tensor_bytes, EMBEDDING_BYTES);
    assert_eq!(
        execution.largest_equivalence_pair_bytes,
        EQUIVALENCE_PAIR_BYTES
    );
    assert_eq!(execution.peak_host_staging_bytes, EQUIVALENCE_PAIR_BYTES);
    assert_eq!(execution.source.shard_opens, 7);
    assert_eq!(execution.source.tensor_reads, SOURCE_READS);
    assert_eq!(execution.source.payload_bytes_read, SOURCE_PAYLOAD_BYTES);
    assert_eq!(execution.source.max_open_shards, 1);
    assert_eq!(
        execution.source.shard_open_order,
        EXPECTED_SHARD_OPEN_ORDER
            .iter()
            .map(|shard| (*shard).to_owned())
            .collect::<Vec<_>>()
    );
}

/// Pins canonical hash encoding and the XXL cache-size arithmetic without weights.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn canonical_output_hashes_and_cache_geometry_are_stable() {
    assert_eq!(
        f32le_sha256(&[0.0, -0.0, 1.5]),
        "39093b17a422b3958b0b3f6789da2d88e11300cacde197bd811453b4d5daa852"
    );
    assert_eq!(
        u32le_sha256(&[0, 1, u32::MAX]),
        "de25d19943926b201c1693709bc5eca70ecf04229c1668e2f276249f9bebe043"
    );
    assert_eq!(u32le_sha256(&GREEDY_TOKEN_IDS), GREEDY_TOKEN_IDS_SHA256);
    assert_eq!(u32le_sha256(&BEAM_TOKEN_IDS), BEAM_TOKEN_IDS_SHA256);
    assert_eq!(
        analytical_t5_cache_bytes(24, 1, 64, 64, 51, 32, 4),
        AnalyticalT5CacheBytes {
            self_per_target_token: 786_432,
            self_total: 25_165_824,
            cross_total: 40_108_032,
            total: 65_273_856,
        }
    );
}

/// Requires an exact JSON value at one RFC 6901 pointer.
fn assert_json(manifest: &Value, pointer: &str, expected: Value) {
    let actual = manifest
        .pointer(pointer)
        .unwrap_or_else(|| panic!("reference manifest is missing `{pointer}`"));
    assert_eq!(actual, &expected, "changed reference field `{pointer}`");
}

/// Finds one immutable reference tensor contract by its JSON key.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn tensor_spec(key: &str) -> &'static ReferenceTensorSpec {
    REFERENCE_TENSORS
        .iter()
        .find(|spec| spec.key == key)
        .unwrap_or_else(|| panic!("unknown reference tensor `{key}`"))
}

/// Decodes a byte-authenticated raw tensor and checks its sparse scalar probes.
fn read_f32le(path: &Path, spec: &ReferenceTensorSpec) -> Vec<f32> {
    assert_eq!(
        sha256_file_hex(path, spec.bytes),
        spec.sha256,
        "changed reference tensor {}",
        spec.key
    );
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let mut chunks = bytes.chunks_exact(4);
    let values = chunks
        .by_ref()
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte F32 chunk")))
        .collect::<Vec<_>>();
    assert!(chunks.remainder().is_empty());
    assert_eq!(
        values.len(),
        spec.shape.iter().product::<usize>(),
        "{} shape does not match its payload",
        spec.key
    );
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{} contains a non-finite scalar",
        spec.key
    );
    for probe in spec.probes {
        assert_eq!(
            values[probe.index].to_bits(),
            probe.value.to_bits(),
            "{} probe {} changed",
            spec.key,
            probe.index
        );
    }
    if let Some(expected) = spec.row_top1 {
        assert_eq!(
            row_top1(&values, 32_128),
            expected,
            "{} top-one decisions changed",
            spec.key
        );
    }
    values
}

/// Returns deterministic lowest-index argmax decisions for fixed-width rows.
fn row_top1(values: &[f32], width: usize) -> Vec<u32> {
    assert!(width > 0 && values.len().is_multiple_of(width));
    values
        .chunks_exact(width)
        .map(|row| {
            let mut winner = 0_usize;
            for index in 1..row.len() {
                if row[index] > row[winner] {
                    winner = index;
                }
            }
            u32::try_from(winner).expect("vocabulary index must fit u32")
        })
        .collect()
}

/// Checks the exact schema and provenance facts needed to interpret raw tensors.
fn assert_reference_schema(manifest: &Value) {
    let top_level = manifest
        .as_object()
        .expect("reference manifest must be a JSON object")
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        top_level,
        BTreeSet::from([
            "beam_generation",
            "decoder_start_token_id",
            "environment",
            "execution",
            "greedy_generation",
            "long_form_input_ids",
            "long_form_prompt",
            "provenance",
            "reference_class",
            "reference_tensors",
            "repository",
            "revision",
            "schema_version",
            "teacher_forced",
            "timings_seconds",
            "translation_input_ids",
            "translation_prompt",
        ])
    );
    assert_json(manifest, "/schema_version", json!(2));
    assert_json(manifest, "/repository", json!(REPOSITORY));
    assert_json(manifest, "/revision", json!(REVISION));
    assert_json(
        manifest,
        "/reference_class",
        json!("transformers.T5ForConditionalGeneration"),
    );
    assert_json(manifest, "/decoder_start_token_id", json!(0));
    assert_json(manifest, "/long_form_prompt", json!(LONG_PROMPT));
    assert_json(
        manifest,
        "/long_form_input_ids",
        json!(LONG_INPUT_IDS.as_slice()),
    );
    assert_json(manifest, "/translation_prompt", json!(TRANSLATION_PROMPT));
    assert_json(
        manifest,
        "/translation_input_ids",
        json!(TRANSLATION_INPUT_IDS),
    );

    assert_json(
        manifest,
        "/provenance/artifact_manifest_sha256",
        json!("bbe4dfea65c912159a98ed8aa0584960e6c89e5fcb607f32eb437fa46e0f2d6e"),
    );
    assert_json(
        manifest,
        "/provenance/header_inventory_sha256",
        json!("794a9c5bb717021bf491ee6942cda6dab2b223a7bb5f7c2861be01248f590a0d"),
    );
    assert_json(
        manifest,
        "/provenance/reference_source_manifest_sha256",
        json!("06554bbcfc8fc87ce5d4f44cb6ef7da412e6870fe67d75b247d63420d7a58408"),
    );
    assert_json(
        manifest,
        "/provenance/checkpoint_tensor_bytes",
        json!(45_067_714_560_u64),
    );
    assert_json(
        manifest,
        "/provenance/index_declared_total_bytes",
        json!(45_594_099_712_u64),
    );
    assert_json(manifest, "/provenance/physical_tensor_count", json!(559));
    assert_json(manifest, "/provenance/logical_index_key_count", json!(560));
    assert_json(manifest, "/provenance/logical_state_keys", json!(560));
    assert_json(
        manifest,
        "/provenance/loaded_unique_parameter_tensors",
        json!(559),
    );
    assert_json(
        manifest,
        "/provenance/loaded_parameter_count",
        json!(11_266_928_640_u64),
    );
    assert_json(
        manifest,
        "/provenance/resolved_index_alias",
        json!({
            "alias": "encoder.embed_tokens.weight",
            "storage_key": "shared.weight"
        }),
    );
    assert_json(
        manifest,
        "/provenance/generator",
        json!({
            "git_dirty": false,
            "git_head": "163c9495b4a2f378edc1cd436cb3717aece81463",
            "relative_path": "crates/transformers/scripts/generate_flan_t5_xxl_reference.py",
            "sha256": "c2c7dae7aae08bdffb25a37740ee67d016693610bce57b3740e0ae398c4687cb"
        }),
    );
    assert_json(
        manifest,
        "/environment/distribution_manifest_sha256",
        json!("b814bb3254d51a28c6d74e80c04c3532c9836a0aadc4fab0eed410d6612d59dd"),
    );
    for (package, version) in [
        ("accelerate", "1.12.0"),
        ("numpy", "2.4.3"),
        ("safetensors", "0.7.0"),
        ("tokenizers", "0.22.2"),
        ("torch", "2.10.0"),
        ("transformers", "5.2.0"),
    ] {
        assert_json(
            manifest,
            &format!("/environment/packages/{package}"),
            json!(version),
        );
    }
    for variable in [
        "HF_DATASETS_OFFLINE",
        "HF_DEACTIVATE_ASYNC_LOAD",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ] {
        assert_json(
            manifest,
            &format!("/environment/offline_environment/{variable}"),
            json!("1"),
        );
    }
    assert_json(manifest, "/execution/device", json!("mps"));
    assert_json(manifest, "/execution/dtype", json!("float32"));
    assert_json(
        manifest,
        "/execution/attention_implementation",
        json!("eager"),
    );
    assert_json(manifest, "/execution/deterministic_algorithms", json!(true));
    assert_json(
        manifest,
        "/execution/explicit_all_one_encoder_and_decoder_masks",
        json!(true),
    );
    assert_json(
        manifest,
        "/execution/runtime_scale_decoder_outputs",
        json!(false),
    );
    assert_json(
        manifest,
        "/execution/runtime_tie_word_embeddings",
        json!(true),
    );
    assert_json(
        manifest,
        "/execution/serialized_tie_word_embeddings",
        json!(false),
    );
    assert_json(
        manifest,
        "/execution/required_non_eos_greedy_tokens",
        json!(32),
    );
    assert_json(
        manifest,
        "/execution/allocator_warmup_override/restored",
        json!(true),
    );
    assert_json(
        manifest,
        "/execution/authenticated_import_bytecode_policy/private_cache_remained_empty",
        json!(true),
    );
    for category in [
        "error_msgs",
        "mismatched_keys",
        "missing_keys",
        "unexpected_keys",
    ] {
        assert_json(
            manifest,
            &format!("/provenance/loading_info/{category}/count"),
            json!(0),
        );
    }
    assert_eq!(
        manifest["provenance"]["artifacts"]
            .as_array()
            .expect("provenance artifacts must be an array")
            .len(),
        12
    );
    assert_eq!(
        manifest["provenance"]["reference_sources"]
            .as_array()
            .expect("reference sources must be an array")
            .len(),
        13
    );
    assert_eq!(
        manifest["provenance"]["root_inventory"]
            .as_array()
            .expect("root inventory must be an array")
            .len(),
        33
    );

    assert_json(
        manifest,
        "/greedy_generation/token_ids",
        json!(GREEDY_TOKEN_IDS),
    );
    assert_json(manifest, "/greedy_generation/contains_eos", json!(false));
    assert_json(
        manifest,
        "/greedy_generation/finish_reason",
        json!("length:32"),
    );
    let mut teacher_input_ids = vec![0_u32];
    teacher_input_ids.extend_from_slice(&GREEDY_TOKEN_IDS[..31]);
    assert_json(
        manifest,
        "/teacher_forced/decoder_input_ids",
        json!(teacher_input_ids),
    );
    assert_json(
        manifest,
        "/teacher_forced/cached_top1_token_ids",
        json!(GREEDY_TOKEN_IDS),
    );
    assert_json(
        manifest,
        "/teacher_forced/stateless_top1_token_ids",
        json!(GREEDY_TOKEN_IDS),
    );
    assert_json(manifest, "/teacher_forced/top1_equal", json!(true));
    assert_json(manifest, "/beam_generation/num_beams", json!(4));
    assert_json(manifest, "/beam_generation/max_new_tokens", json!(16));
    assert_json(manifest, "/beam_generation/early_stopping", json!(true));
    assert_json(
        manifest,
        "/beam_generation/generated_token_ids",
        json!(BEAM_TOKEN_IDS),
    );
    assert_json(manifest, "/beam_generation/finish_reason", json!("eos:1"));

    let tensor_keys = manifest["reference_tensors"]
        .as_object()
        .expect("reference_tensors must be an object")
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        tensor_keys,
        REFERENCE_TENSORS
            .iter()
            .map(|spec| spec.key)
            .collect::<BTreeSet<_>>()
    );
    for spec in &REFERENCE_TENSORS {
        let base = format!("/reference_tensors/{}", spec.key);
        assert_json(
            manifest,
            &format!("{base}/relative_path"),
            json!(spec.filename),
        );
        assert_json(manifest, &format!("{base}/bytes"), json!(spec.bytes));
        assert_json(manifest, &format!("{base}/elements"), json!(spec.bytes / 4));
        assert_json(manifest, &format!("{base}/sha256"), json!(spec.sha256));
        assert_json(manifest, &format!("{base}/shape"), json!(spec.shape));
        assert_json(
            manifest,
            &format!("{base}/encoding"),
            json!("ieee754-f32-little-endian"),
        );
        assert_json(manifest, &format!("{base}/finite"), json!(true));
        let probes = manifest["reference_tensors"][spec.key]["probes"]
            .as_array()
            .expect("tensor probes must be an array");
        assert_eq!(probes.len(), spec.probes.len());
        for (actual, expected) in probes.iter().zip(spec.probes) {
            assert_eq!(actual["flat_index"], json!(expected.index));
            let value = actual["value"]
                .as_f64()
                .expect("probe value must be numeric") as f32;
            assert_eq!(value.to_bits(), expected.value.to_bits());
        }
        if let Some(top1) = spec.row_top1 {
            let pointer = if spec.key == "first_logits" {
                format!("{base}/top1_token_id")
            } else {
                format!("{base}/row_top1_token_ids")
            };
            if spec.key == "first_logits" {
                assert_json(manifest, &pointer, json!(top1[0]));
            } else {
                assert_json(manifest, &pointer, json!(top1));
            }
        }
    }
}

/// Authenticates every create-only bundle entry and returns decoded F32 values.
fn load_reference_bundle() -> ReferenceBundle {
    let root = reference_root();
    let expected_names = std::iter::once(REFERENCE_JSON)
        .chain(REFERENCE_TENSORS.iter().map(|spec| spec.filename))
        .collect::<BTreeSet<_>>();
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
    assert_eq!(
        observed_names,
        expected_names
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<BTreeSet<_>>(),
        "the create-only reference bundle contains changed or extra entries"
    );

    let manifest_path = root.join(REFERENCE_JSON);
    assert_eq!(
        sha256_file_hex(&manifest_path, REFERENCE_JSON_BYTES),
        REFERENCE_JSON_SHA256,
        "reference manifest changed"
    );
    let manifest: Value = serde_json::from_slice(
        &fs::read(&manifest_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", manifest_path.display())),
    )
    .expect("reference manifest must be valid JSON");
    assert_reference_schema(&manifest);

    let mut tensors = BTreeMap::new();
    for spec in &REFERENCE_TENSORS {
        tensors.insert(spec.key, read_f32le(&root.join(spec.filename), spec));
    }
    ReferenceBundle {
        encoder_final: tensors
            .remove("encoder_final_hidden_state")
            .expect("encoder reference"),
        decoder_first_final: tensors
            .remove("decoder_first_final_hidden_state")
            .expect("decoder reference"),
        first_logits: tensors
            .remove("first_logits")
            .expect("first logits reference"),
        cached_logits: tensors
            .remove("cached_greedy_logits")
            .expect("cached reference"),
        teacher_logits: tensors
            .remove("teacher_forced_logits")
            .expect("teacher reference"),
    }
}

/// Validates the immutable reference bundle without allocating model parameters.
#[test]
#[ignore = "requires the pinned FLAN-T5-XXL external reference bundle"]
fn validates_pinned_flan_t5_xxl_reference_bundle() {
    let reference = load_reference_bundle();
    assert_eq!(reference.encoder_final.len(), 208_896);
    assert_eq!(reference.decoder_first_final.len(), 4_096);
    assert_eq!(reference.first_logits.len(), 32_128);
    assert_eq!(reference.cached_logits.len(), 1_028_096);
    assert_eq!(reference.teacher_logits.len(), 1_028_096);
    eprintln!(
        "FLAN-T5-XXL reference bundle: revision={REVISION}, manifest_sha256={REFERENCE_JSON_SHA256}, raw_f32_bytes=9205248"
    );
}

/// Synchronizes all pending Metal work so timings include device execution.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn sync_metal(device: &<Metal as burn::tensor::backend::BackendTypes>::Device) {
    <Metal<f32, i32> as Backend>::sync(device).expect("synchronize Metal acceptance backend");
}

/// Materializes one Metal tensor as host F32 values with a contextual error.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_values<const D: usize>(label: &str, tensor: &Tensor<Metal, D>) -> Vec<f32> {
    tensor
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .unwrap_or_else(|error| panic!("materialize {label}: {error}"))
}

/// Computes cosine similarity using F64 accumulation and rejects zero vectors.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn cosine_similarity(actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    let (dot, actual_square, expected_square) = actual.iter().zip(expected).fold(
        (0.0_f64, 0.0_f64, 0.0_f64),
        |(dot, actual_square, expected_square), (&actual, &expected)| {
            let actual = f64::from(actual);
            let expected = f64::from(expected);
            (
                dot + actual * expected,
                actual_square + actual * actual,
                expected_square + expected * expected,
            )
        },
    );
    assert!(actual_square > 0.0 && expected_square > 0.0);
    dot / (actual_square.sqrt() * expected_square.sqrt())
}

/// Checks finiteness, sparse probes, global cosine, per-row cosine, and top one.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn assert_metal_reference(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    spec: &ReferenceTensorSpec,
) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{label} element count");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label} contains non-finite values"
    );
    let global_cosine = cosine_similarity(actual, expected);
    assert!(
        global_cosine >= 0.999,
        "{label} global cosine {global_cosine:.12} is below 0.999"
    );
    for probe in spec.probes {
        let reference = expected[probe.index];
        let tolerance = 1.0e-4_f32 + 1.0e-4_f32 * reference.abs();
        let error = (actual[probe.index] - reference).abs();
        assert!(
            error <= tolerance,
            "{label} probe {} differs by {error}, tolerance {tolerance}: actual={}, reference={reference}",
            probe.index,
            actual[probe.index]
        );
    }

    let mut minimum_cosine = global_cosine;
    if let Some(expected_top1) = spec.row_top1 {
        let width = 32_128;
        let actual_top1 = row_top1(actual, width);
        assert_eq!(actual_top1, expected_top1, "{label} top-one mismatch");
        for (row, (actual, expected)) in actual
            .chunks_exact(width)
            .zip(expected.chunks_exact(width))
            .enumerate()
        {
            let cosine = cosine_similarity(actual, expected);
            assert!(
                cosine >= 0.999,
                "{label} row {row} cosine {cosine:.12} is below 0.999"
            );
            minimum_cosine = minimum_cosine.min(cosine);
        }
    }
    minimum_cosine
}

/// Builds an explicitly all-valid one-row Metal token input.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_input(
    ids: &[u32],
    device: &<Metal as burn::tensor::backend::BackendTypes>::Device,
) -> CausalLmInput<Metal> {
    CausalLmInput {
        token_ids: Tensor::<Metal, 2, Int>::from_data(
            TensorData::new(
                ids.iter().copied().map(i64::from).collect::<Vec<_>>(),
                [1, ids.len()],
            ),
            device,
        ),
        token_mask: Some(TokenMask::new(Tensor::<Metal, 2, Bool>::from_data(
            TensorData::new(vec![true; ids.len()], [1, ids.len()]),
            device,
        ))),
        position_ids: None,
    }
}

/// Strictly loads F32 FLAN-T5-XXL and proves official stateless/cached parity.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires the pinned 45 GB F32 checkpoint, reference bundle, Metal, and about 70 GiB free unified memory"]
fn loads_flan_t5_xxl_f32_and_matches_official_reference() {
    let reference = load_reference_bundle();
    let root = checkpoint_root();
    let device = Default::default();
    let probe = Tensor::<Metal, 1>::zeros([1], &device);
    assert_eq!(metal_values("Metal availability probe", &probe), [0.0]);

    let load_started = Instant::now();
    let loaded = AutoSeq2SeqLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Preserve,
                ..LoadOptions::default()
            },
            load_tokenizer: true,
            repository: Some(REPOSITORY.to_owned()),
            revision: Some(REVISION.to_owned()),
            ..AutoLoadOptions::default()
        },
        &device,
    )
    .expect("strictly stream the complete official FLAN-T5-XXL graph");
    sync_metal(&device);
    let load_seconds = load_started.elapsed().as_secs_f64();

    assert_eq!(loaded.source().model_type(), "t5");
    assert_eq!(loaded.source().repository(), Some(REPOSITORY));
    assert_eq!(loaded.source().revision(), Some(REVISION));
    assert_eq!(
        loaded.source().root(),
        root.canonicalize().expect("canonical checkpoint root")
    );
    let report = loaded.load_report();
    assert!(report.is_strictly_valid());
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());
    assert!(report.allowlisted_missing.is_empty());
    assert!(report.allowlisted_ignored.is_empty());
    assert!(report.dtype_converted.is_empty());
    assert_eq!(report.applied.len(), 558);
    assert_eq!(report.transformed.len(), 433);

    let model = match loaded.model() {
        AutoSeq2SeqLm::T5(model) => model,
        model => panic!("expected T5 automatic dispatch, found {model:?}"),
    };
    assert_eq!(model.config.vocab_size, 32_128);
    assert_eq!(model.config.d_model, 4_096);
    assert_eq!(model.config.num_layers, 24);
    assert_eq!(model.config.decoder_layers(), 24);
    assert!(!model.config.tie_word_embeddings);
    assert_eq!(model.shared.weight.val().dtype(), DType::F32);
    assert_eq!(
        model
            .lm_head
            .as_ref()
            .expect("FLAN-T5-XXL has an independent LM head")
            .weight
            .val()
            .dtype(),
        DType::F32
    );

    let tokenizer = loaded
        .tokenizer()
        .expect("the strict load must attach tokenizer.json");
    for (prompt, expected) in [
        (LONG_PROMPT, LONG_INPUT_IDS.as_slice()),
        (TRANSLATION_PROMPT, TRANSLATION_INPUT_IDS.as_slice()),
    ] {
        let encoding = tokenizer
            .encode(
                prompt,
                &EncodeOptions {
                    // The authenticated Python oracle used
                    // `add_special_tokens=True`; T5's tokenizer post-processor
                    // appends the terminal EOS represented in both fixtures.
                    add_special_tokens: true,
                    ..EncodeOptions::default()
                },
            )
            .expect("encode pinned FLAN prompt");
        assert_eq!(encoding.token_ids(), &[expected.to_vec()]);
    }

    let source = metal_input(&LONG_INPUT_IDS, &device);
    let encoder_started = Instant::now();
    let encoder = model
        .encode(source.clone(), &ForwardOptions::default())
        .expect("execute the full FLAN-T5-XXL encoder");
    sync_metal(&device);
    let encoder_seconds = encoder_started.elapsed().as_secs_f64();
    assert_eq!(encoder.hidden_state.dims(), [1, 51, 4_096]);
    let encoder_values = metal_values("FLAN encoder final hidden", &encoder.hidden_state);
    let encoder_cosine = assert_metal_reference(
        "FLAN encoder final hidden",
        &encoder_values,
        &reference.encoder_final,
        tensor_spec("encoder_final_hidden_state"),
    );

    let first_options = ForwardOptions {
        logits: LogitsSelection::Last,
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let first_started = Instant::now();
    let first = model
        .forward(
            Seq2SeqInput {
                encoder: source.clone(),
                decoder: metal_input(&[0], &device),
            },
            &first_options,
        )
        .expect("execute stateless first-token FLAN decoder");
    sync_metal(&device);
    let stateless_first_seconds = first_started.elapsed().as_secs_f64();
    assert_eq!(first.decoder.logits.dims(), [1, 1, 32_128]);
    let first_hidden = first
        .decoder
        .final_hidden_state
        .as_ref()
        .expect("hidden-state diagnostics must retain the final decoder state");
    assert_eq!(first_hidden.dims(), [1, 1, 4_096]);
    let first_hidden_values = metal_values("FLAN first decoder final hidden", first_hidden);
    let decoder_hidden_cosine = assert_metal_reference(
        "FLAN first decoder final hidden",
        &first_hidden_values,
        &reference.decoder_first_final,
        tensor_spec("decoder_first_final_hidden_state"),
    );
    let first_logits = metal_values("FLAN first stateless logits", &first.decoder.logits);
    let first_logits_cosine = assert_metal_reference(
        "FLAN first stateless logits",
        &first_logits,
        &reference.first_logits,
        tensor_spec("first_logits"),
    );

    let cached_options = ForwardOptions {
        logits: LogitsSelection::Last,
        ..ForwardOptions::default()
    };
    let mut state = model
        .new_decode_state(
            CacheConfig::new(1, 32).expect("32-token greedy cache"),
            1,
            &device,
        )
        .expect("allocate FLAN decoder and cross-attention cache");
    assert_eq!(state.capacity(), 32);
    assert!(!state.has_encoder_cache());
    assert_eq!(state.encoder_sequence_length(), None);
    let cached_started = Instant::now();
    let prefill = model
        .prefill_decoder(
            &encoder,
            metal_input(&[0], &device),
            &mut state,
            &cached_options,
        )
        .expect("prefill FLAN decoder start token");
    let cached_encoder = prefill.encoder;
    let first_decoder = prefill.decoder;
    assert_eq!(state.lengths(), &[1]);
    assert!(state.has_encoder_cache());
    assert_eq!(state.encoder_sequence_length(), Some(LONG_INPUT_IDS.len()));
    assert_eq!(first_decoder.logits.dims(), [1, 1, 32_128]);
    assert!(first_decoder.final_hidden_state.is_none());
    let mut cached_logits = metal_values("FLAN first cached logits", &first_decoder.logits);
    let mut generated = vec![row_top1(&cached_logits, 32_128)[0]];
    assert_eq!(generated[0], GREEDY_TOKEN_IDS[0]);

    for step in 1..GREEDY_TOKEN_IDS.len() {
        let decoded = model
            .decode(
                &cached_encoder,
                DecodeInput {
                    token_ids: metal_input(&[generated[step - 1]], &device).token_ids,
                    token_mask: Some(TokenMask::new(Tensor::<Metal, 2, Bool>::from_data(
                        TensorData::new(vec![true], [1, 1]),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &cached_options,
            )
            .unwrap_or_else(|error| panic!("cached greedy decode step {step}: {error}"));
        assert_eq!(decoded.decoder.logits.dims(), [1, 1, 32_128]);
        let row = metal_values("FLAN cached logits", &decoded.decoder.logits);
        let token = row_top1(&row, 32_128)[0];
        assert_eq!(
            token, GREEDY_TOKEN_IDS[step],
            "cached greedy token mismatch at step {step}"
        );
        generated.push(token);
        cached_logits.extend(row);
    }
    sync_metal(&device);
    let cached_seconds = cached_started.elapsed().as_secs_f64();
    assert_eq!(state.lengths(), &[32]);
    assert_eq!(generated, GREEDY_TOKEN_IDS);
    let analytical_cache = analytical_t5_cache_bytes(
        model.config.decoder_layers(),
        1,
        model.config.num_heads,
        model.config.d_kv,
        state
            .encoder_sequence_length()
            .expect("successful prefill must bind the source sequence"),
        state.capacity(),
        model.shared.weight.val().dtype().size(),
    );
    assert_eq!(analytical_cache.self_per_target_token, 786_432);
    assert_eq!(analytical_cache.self_total, 25_165_824);
    assert_eq!(analytical_cache.cross_total, 40_108_032);
    assert_eq!(analytical_cache.total, 65_273_856);
    let cached_minimum_cosine = assert_metal_reference(
        "FLAN cached greedy logits",
        &cached_logits,
        &reference.cached_logits,
        tensor_spec("cached_greedy_logits"),
    );

    let mut teacher_ids = vec![0_u32];
    teacher_ids.extend_from_slice(&GREEDY_TOKEN_IDS[..31]);
    let teacher_started = Instant::now();
    let teacher = model
        .forward(
            Seq2SeqInput {
                encoder: source,
                decoder: metal_input(&teacher_ids, &device),
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .expect("execute stateless teacher-forced FLAN decoder");
    sync_metal(&device);
    let teacher_seconds = teacher_started.elapsed().as_secs_f64();
    assert_eq!(teacher.decoder.logits.dims(), [1, 32, 32_128]);
    let teacher_logits = metal_values("FLAN teacher-forced logits", &teacher.decoder.logits);
    let teacher_minimum_cosine = assert_metal_reference(
        "FLAN teacher-forced logits",
        &teacher_logits,
        &reference.teacher_logits,
        tensor_spec("teacher_forced_logits"),
    );

    let beam_started = Instant::now();
    let beam = generate_t5_beam_search(
        model,
        metal_input(&TRANSLATION_INPUT_IDS, &device),
        T5BeamSearchOptions {
            max_new_tokens: 16,
            num_beams: 4,
            num_return_sequences: 1,
            length_penalty: 1.0,
            // Both implementations use the conventional eager rule: finish
            // once the beam-width EOS pool is complete for the source row.
            early_stopping: true,
        },
    )
    .expect("run exact four-beam FLAN generation");
    sync_metal(&device);
    let beam_seconds = beam_started.elapsed().as_secs_f64();
    assert_eq!(beam.sequences.len(), 1);
    assert_eq!(beam.sequences[0].len(), 1);
    assert_eq!(beam.sequences[0][0].token_ids, BEAM_TOKEN_IDS);
    assert!(matches!(
        beam.sequences[0][0].finish_reason,
        FinishReason::EndOfSequence { token_id: 1 }
    ));

    let execution = loaded
        .execution_stats()
        .expect("automatic strict load must retain execution accounting");
    assert_pinned_f32_load_accounting(report, execution);
    let memory = report
        .memory
        .expect("strict FLAN-T5-XXL loading must retain its memory estimate");

    let encoder_f32le_sha256 = f32le_sha256(&encoder_values);
    let decoder_first_f32le_sha256 = f32le_sha256(&first_hidden_values);
    let first_logits_f32le_sha256 = f32le_sha256(&first_logits);
    let cached_logits_f32le_sha256 = f32le_sha256(&cached_logits);
    let teacher_logits_f32le_sha256 = f32le_sha256(&teacher_logits);
    let greedy_u32le_sha256 = u32le_sha256(&generated);
    let beam_u32le_sha256 = u32le_sha256(&beam.sequences[0][0].token_ids);
    assert_eq!(greedy_u32le_sha256, GREEDY_TOKEN_IDS_SHA256);
    assert_eq!(beam_u32le_sha256, BEAM_TOKEN_IDS_SHA256);

    eprintln!(
        "FLAN-T5-XXL F32 Metal: revision={REVISION}, load_seconds={load_seconds:.3}, \
         encoder_seconds={encoder_seconds:.3}, stateless_first_seconds={stateless_first_seconds:.3}, \
         cached_32_seconds={cached_seconds:.3}, \
         teacher_seconds={teacher_seconds:.3}, beam_seconds={beam_seconds:.3}, \
         encoder_cosine={encoder_cosine:.12}, decoder_first_hidden_cosine={decoder_hidden_cosine:.12}, \
         first_logits_cosine={first_logits_cosine:.12}, cached_minimum_row_cosine={cached_minimum_cosine:.12}, \
         teacher_minimum_row_cosine={teacher_minimum_cosine:.12}, \
         analytical_self_cache_bytes_per_target_token={}, analytical_self_cache_bytes={}, \
         analytical_cross_cache_bytes={}, analytical_total_cache_payload_bytes={}, \
         greedy_ids={generated:?}, beam_ids={:?}",
        analytical_cache.self_per_target_token,
        analytical_cache.self_total,
        analytical_cache.cross_total,
        analytical_cache.total,
        beam.sequences[0][0].token_ids,
    );
    eprintln!(
        "FLAN-T5-XXL load report: applied={}, transformed={}, aliased={}, \
         checkpoint_aliases={}, equivalence_planned={}, equivalence_verified={}, \
         dtype_converted={}, quantized={}, allowlisted_missing={}, allowlisted_ignored={}, \
         missing={}, unexpected={}, artifact_bytes={}, checkpoint_tensor_bytes={}, \
         target_parameter_bytes={}, logical_model_bytes={}, largest_shard_bytes={}, \
         largest_source_tensor_bytes={}, largest_target_tensor_bytes={}, \
         largest_planned_equivalence_pair_bytes={}, planned_peak_host_staging_bytes={}",
        report.applied.len(),
        report.transformed.len(),
        report.aliased.len(),
        report.checkpoint_aliases.len(),
        report.equivalence_planned.len(),
        report.equivalence_verified.len(),
        report.dtype_converted.len(),
        report.quantized.len(),
        report.allowlisted_missing.len(),
        report.allowlisted_ignored.len(),
        report.missing.len(),
        report.unexpected.len(),
        memory.artifact_bytes,
        memory.checkpoint_tensor_bytes,
        memory.target_parameter_bytes,
        memory.logical_model_bytes,
        memory.largest_shard_bytes,
        memory.largest_source_tensor_bytes,
        memory.largest_target_tensor_bytes,
        memory.largest_equivalence_pair_bytes,
        memory.peak_host_staging_bytes,
    );
    eprintln!(
        "FLAN-T5-XXL load execution: tensors_applied={}, aliases_applied={}, \
         equivalences_verified={}, equivalence_source_reads={}, \
         equivalence_source_payload_bytes={}, source_reads={}, source_payload_bytes={}, \
         delivered_tensor_bytes={}, largest_source_payload_bytes={}, \
         largest_output_tensor_bytes={}, largest_equivalence_pair_bytes={}, \
         peak_host_staging_bytes={}, shard_opens={}, source_tensor_reads={}, \
         source_payload_bytes_read={}, max_open_shards={}, shard_open_order={:?}",
        execution.tensors_applied,
        execution.aliases_applied,
        execution.equivalences_verified,
        execution.equivalence_source_reads,
        execution.equivalence_source_payload_bytes,
        execution.source_reads,
        execution.source_payload_bytes,
        execution.delivered_tensor_bytes,
        execution.largest_source_payload_bytes,
        execution.largest_output_tensor_bytes,
        execution.largest_equivalence_pair_bytes,
        execution.peak_host_staging_bytes,
        execution.source.shard_opens,
        execution.source.tensor_reads,
        execution.source.payload_bytes_read,
        execution.source.max_open_shards,
        execution.source.shard_open_order,
    );
    eprintln!(
        "FLAN-T5-XXL canonical output SHA-256: encoder_final_f32le={encoder_f32le_sha256}, \
         decoder_first_final_f32le={decoder_first_f32le_sha256}, \
         first_logits_f32le={first_logits_f32le_sha256}, \
         cached_logits_32x32128_f32le={cached_logits_f32le_sha256}, \
         teacher_logits_32x32128_f32le={teacher_logits_f32le_sha256}, \
         greedy_token_ids_u32le={greedy_u32le_sha256}, beam_token_ids_u32le={beam_u32le_sha256}"
    );
}
