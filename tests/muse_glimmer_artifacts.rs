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
//! Q8 and Q4 conversion use explicit, create-only paths on the checkpoint's
//! external volume. Conversion and Metal execution are separate gates so a
//! reviewed sidecar can be reused without rewriting tens of gigabytes:
//!
//! ```text
//! TRANSFORMERS_MUSE_GLIMMER_Q8_BURNPACK=/Volumes/<name>/quantized/muse-q8-g64.bpk \
//! cargo test --release --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,tokenizers \
//!   --test muse_glimmer_artifacts \
//!   quantizes_pinned_muse_glimmer_weights_to_q8_burnpack \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! Add `metal,kernel-tracing` and select
//! `smokes_and_audits_pinned_muse_glimmer_q{4,8}_burnpack_on_metal` to prove
//! that one recurrent decode executes all 417 packed projections without a
//! whole-weight dequantization launch. These are short correctness diagnostics,
//! not quantized-quality, 32K, or advertised-support gates.
//!
//! Quality is measured in separate Q8 and Q4 processes so each 30B graph is
//! released before the other precision loads. Both commands also require
//! `TRANSFORMERS_MUSE_GLIMMER_DIR` and
//! `TRANSFORMERS_MUSE_GLIMMER_REFERENCE_DIR`:
//!
//! ```text
//! TRANSFORMERS_MUSE_GLIMMER_Q8_BURNPACK=/absolute/path/to/reviewed-q8.bpk \
//! cargo test --release --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,metal,tokenizers \
//!   --test muse_glimmer_artifacts \
//!   measures_pinned_muse_glimmer_q8_official_cached_quality_on_metal \
//!   -- --ignored --exact --nocapture
//!
//! TRANSFORMERS_MUSE_GLIMMER_Q4_BURNPACK=/absolute/path/to/reviewed-q4.bpk \
//! cargo test --release --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,metal,tokenizers \
//!   --test muse_glimmer_artifacts \
//!   measures_pinned_muse_glimmer_q4_official_cached_quality_on_metal \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! Stream all 59.6 GB through the independent payload identity gate with:
//!
//! ```text
//! cargo test --release --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std \
//!   --test muse_glimmer_artifacts \
//!   validates_pinned_muse_glimmer_full_shard_payload_hashes \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! The optional Metal gate audits strict streaming and normalized generation
//! metadata, but deliberately does not claim reference-logit acceptance. To
//! create the separately authenticated official numerical oracle, first check
//! out Transformers at `cfa309cfa34b5497a32fd50af49aa97aac94c054`, prepare
//! the exact Python environment enforced by the generator, and run:
//!
//! ```text
//! python crates/transformers/scripts/generate_muse_glimmer_reference.py \
//!   --model-dir "$TRANSFORMERS_MUSE_GLIMMER_DIR" \
//!   --transformers-dir /absolute/path/to/pinned/transformers \
//!   --output-dir /absolute/external/reference-parent
//! ```
//!
//! The tool is deliberately local-only and MPS/F16/eager-only. It atomically
//! publishes a create-only seven-file bundle after recording full-vocabulary
//! prompt logits, all 32 cached and teacher-forced vocabulary rows, and three
//! 54-row hidden traces. An independent reviewer must pin every printed output
//! hash below; the
//! ignored Rust acceptance gate fails closed until that has happened.
//!
//! Once those hashes are reviewed and pinned, run the full comparison with:
//!
//! ```text
//! TRANSFORMERS_MUSE_GLIMMER_REFERENCE_DIR=/absolute/external/reference-parent \
//! cargo test --release --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,metal,tokenizers \
//!   --test muse_glimmer_artifacts \
//!   matches_pinned_muse_glimmer_official_f16_reference_on_metal \
//!   -- --ignored --exact --nocapture
//! ```

#![recursion_limit = "256"]

#[cfg(all(
    feature = "kernel-tracing",
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
#[path = "support/kernel_trace.rs"]
mod kernel_trace;

#[path = "support/quantized_quality.rs"]
mod quantized_quality;

use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    mem::size_of,
    path::{Component, Path, PathBuf},
    time::Instant,
};

use std::process::Command;

#[cfg(all(feature = "metal", feature = "tokenizers", target_os = "macos"))]
use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt};

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
    quantization::{
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2, PROVENANCE_HASH_BUFFER_BYTES,
        PackedExecutionSupport, QuantizationEligibility, QuantizationExclusion,
        QuantizationGroupSize, QuantizedBurnpackArtifact, QuantizedCheckpointOptions,
        SourceFloatDType, SymmetricQuantization, WeightQuantizationConfig,
        classify_weight_for_quantization, load_plan_sha256, quantize_local_checkpoint_to_burnpack,
        validate_sidecar_contract,
    },
};

#[cfg(feature = "tokenizers")]
use transformers::{EncodeOptions, LocalTokenizer};

#[cfg(feature = "chat-templates")]
use transformers::{ChatMessage, ChatTemplateOptions};

#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
use burn::nn::Linear;
#[cfg(all(feature = "metal", feature = "tokenizers"))]
use burn::{
    backend::Metal,
    tensor::{Bool, DType, Int, Tensor, TensorData, backend::Backend},
};
#[cfg(all(feature = "metal", feature = "tokenizers"))]
use transformers::{
    AutoCausalLm, AutoLoadOptions, DecodeInput, DecodeState, ForwardOptions, LogitsSelection,
    SamplingStrategy, TokenMask,
    architecture::CausalLmInput,
    cache::{CacheConfig, KvCacheStorage},
    models::{
        AutoCausalDecodeState,
        muse_glimmer::{MuseGlimmerDecodeState, MuseGlimmerLayerCache},
    },
};
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
use transformers::{LoadedModel, models::muse_glimmer::MuseGlimmerForCausalLm};

#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
use quantized_quality::{FixedSequenceQuality, enforce_regression_limit};

#[cfg(all(
    feature = "kernel-tracing",
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
use kernel_trace::{
    calibrated_dequantization_kernel_names, capture_kernel_launches, compact_kernel_id,
    packed_matmul_launches, whole_weight_dequantization_launches,
};

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

/// Package version declared by the exact pinned Transformers source checkout.
const REFERENCE_SOURCE_VERSION: &str = "5.17.0.dev0";

/// Older producer metadata embedded in the immutable checkpoint configuration.
const CHECKPOINT_CONFIG_TRANSFORMERS_VERSION: &str = "5.15.0.dev0";

/// Canonical identity of all authenticated importable wheel payload manifests.
const REFERENCE_DISTRIBUTION_MANIFEST_SHA256: &str =
    "6eca7252cebb0750fa344a7d05419a4511f3ecf985e6f893dadf3e0e94c2b2a6";

/// Explicit boundary of the descriptor/hash preflight and postflight checks.
const REFERENCE_INTEGRITY_THREAT_MODEL: &str = "Pre/post descriptor identities and SHA-256 detect persistent mutation and ordinary races. An actively malicious same-UID process that swaps a pathname only while Transformers or SafeTensors reopens it, then restores the original before postflight, is outside scope because it can also tamper with this process.";

/// Explicitly non-authenticated interpreter and operating-system trust boundary.
const REFERENCE_INTERPRETER_TRUST_BOUNDARY: &str = "CPython implementation and 3.11.9 final version are enforced and the executable paths are recorded, but interpreter, standard-library, system-library, Git executable, and OS payload bytes are trusted environment inputs rather than content-authenticated.";

/// Explicitly non-authenticated transitive-package trust boundary.
const REFERENCE_TRANSITIVE_TRUST_BOUNDARY: &str = "The 11 listed primary wheels have exact versions and RECORD payload manifests. Every other imported third-party module must belong to a recorded installed distribution, but those transitive distribution payloads are trusted from the selected offline virtual environment rather than content-authenticated.";

/// Exact reason the pinned official MPS warmup branch is wrapped and observed.
const REFERENCE_WARMUP_REASON: &str = "Pinned Transformers 5.17 explicitly skips allocator warmup on MPS. The scoped wrapper validates the complete expanded parameter map, then delegates to that official no-op branch without changing model tensors or numerical execution.";

/// Exact pinned config identity, also enforced by the committed fixture.
const CONFIG_SHA256: &str = "5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14";

/// Exact pinned generation-config identity, also enforced by its fixture.
const GENERATION_CONFIG_SHA256: &str =
    "1fa51889b1f8d3659802dedaa27e005b81e5c58483f13ecf13f2d97306bc6e35";

/// Exact lengths and SHA-256 identities of all pinned control artifacts.
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
const PHYSICAL_TENSOR_COUNT: usize = 1_436;

/// Number of language-tower and vocabulary-head tensors consumed by Burn.
const CONSUMED_TENSOR_COUNT: usize = 627;

/// Attention, SwiGLU, and independent-head matrices eligible for Burnpack.
#[cfg(feature = "tokenizers")]
const QUANTIZED_TENSOR_COUNT: usize = 417;

/// Embedding plus 209 learned normalization vectors retained as F16.
#[cfg(feature = "tokenizers")]
const RETAINED_F16_TENSOR_COUNT: usize = 210;

/// F16 bytes read from SafeTensors when a v2 sidecar supplies all matrices.
#[cfg(feature = "tokenizers")]
const RETAINED_F16_TENSOR_BYTES: u64 = 2_692_445_184;

/// Packed-value and scale bytes for all 417 Q8/G64 matrices.
#[cfg(feature = "tokenizers")]
const Q8_PACKED_PAYLOAD_BYTES: u64 = 28_165_343_232;

/// Packed-value and scale bytes for all 417 Q4/G64 matrices.
#[cfg(feature = "tokenizers")]
const Q4_PACKED_PAYLOAD_BYTES: u64 = 14_911_064_064;

/// Largest Q8 payload: the independent `[202048, 6656]` vocabulary head.
#[cfg(feature = "tokenizers")]
const Q8_LARGEST_PACKED_TENSOR_BYTES: u64 = 1_428_883_456;

/// Largest Q4 payload: the independent `[202048, 6656]` vocabulary head.
#[cfg(feature = "tokenizers")]
const Q4_LARGEST_PACKED_TENSOR_BYTES: u64 = 756_467_712;

/// Every matrix invoked by one batch-one recurrent decode.
#[cfg(all(
    feature = "kernel-tracing",
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
const PACKED_DECODE_MATMUL_LAUNCHES: usize = 417;

/// Environment variable selecting the create-only/reused Q8 artifact.
#[cfg(feature = "tokenizers")]
const Q8_BURNPACK_ENV: &str = "TRANSFORMERS_MUSE_GLIMMER_Q8_BURNPACK";

/// Environment variable selecting the create-only/reused Q4 artifact.
#[cfg(feature = "tokenizers")]
const Q4_BURNPACK_ENV: &str = "TRANSFORMERS_MUSE_GLIMMER_Q4_BURNPACK";

/// Immutable filename of the reviewed complete Q8/G64 v2 sidecar.
#[cfg(feature = "tokenizers")]
const Q8_BURNPACK_FILE_NAME: &str = "muse-glimmer-30b-a4e59da5-q8-g64-v2.bpk";

/// Immutable filename of the reviewed complete Q4/G64 v2 sidecar.
#[cfg(feature = "tokenizers")]
const Q4_BURNPACK_FILE_NAME: &str = "muse-glimmer-30b-a4e59da5-q4-g64-v2.bpk";

/// Absolute external path observed for the conversion acceptance run.
#[cfg(feature = "tokenizers")]
const Q8_BURNPACK_OBSERVED_PATH: &str = "/Volumes/bag of holding/models/transformers-acceptance/muse-glimmer-30b-q8-a4e59da5/muse-glimmer-30b-a4e59da5-q8-g64-v2.bpk";

/// Absolute external path observed for the Q4 conversion acceptance run.
#[cfg(feature = "tokenizers")]
const Q4_BURNPACK_OBSERVED_PATH: &str = "/Volumes/bag of holding/models/transformers-acceptance/muse-glimmer-30b-q4-a4e59da5/muse-glimmer-30b-a4e59da5-q4-g64-v2.bpk";

/// Reviewed complete Q8 Burnpack identity after conversion and full reopen.
#[cfg(feature = "tokenizers")]
const Q8_BURNPACK_SHA256: Option<&str> =
    Some("91cfd35bfe0a389cbc2eed2189c4e33312bdce1a0ff50d8452acf06d00f0e8eb");

/// Reviewed complete Q4 Burnpack identity after conversion and full reopen.
#[cfg(feature = "tokenizers")]
const Q4_BURNPACK_SHA256: Option<&str> =
    Some("05b69f67799b4dc0228b4298e39d466d753fcc5d6484f61c8920c551017daed3");

/// Reviewed Q8 artifact length after conversion and full reopen.
#[cfg(feature = "tokenizers")]
const Q8_BURNPACK_ARTIFACT_BYTES: Option<u64> = Some(28_165_645_337);

/// Reviewed Q4 artifact length after conversion and full reopen.
#[cfg(feature = "tokenizers")]
const Q4_BURNPACK_ARTIFACT_BYTES: Option<u64> = Some(14_911_372_485);

/// Inclusive Q8 mean-NLL regression budget from the v1 release policy.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
const Q8_MAXIMUM_RELATIVE_NLL_REGRESSION: f64 = 0.01;

/// Inclusive Q4 mean-NLL regression budget from the v1 release policy.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
const Q4_MAXIMUM_RELATIVE_NLL_REGRESSION: f64 = 0.05;

/// Number of consumed tensors renamed out of the multimodal namespace.
const RENAMED_TENSOR_COUNT: usize = 626;

/// Number of narrowly allowlisted visual tensors left uninstantiated.
#[cfg(feature = "tokenizers")]
const VISUAL_TENSOR_COUNT: usize = 809;

/// BF16 bytes across the complete multimodal tensor inventory.
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

/// Literal rendered chat prompt selected for 32 numerically stable decisions.
///
/// The date is intentionally part of the immutable string rather than being
/// expanded at test time by the checkpoint's dynamic chat template.
const REFERENCE_PROMPT: &str = concat!(
    "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\n",
    "Knowledge cutoff: 2026-01-04.\n",
    "Current date: 2026-09-08.\n\n",
    "Reasoning strength: high.\n\n",
    "# Valid recipients: \"self\", \"user\".<|eot|>",
    "<|start|>user<|message|>Explain why the sky appears blue in one concise paragraph.",
    "<|eot|><|start|>assistant",
);

/// Exact tokenization of [`REFERENCE_PROMPT`] without injected special tokens.
const REFERENCE_TOKEN_IDS: [u32; 67] = [
    200_000, 200_022, 15_651, 200_023, 4_662, 583, 262, 19_933, 12_133, 43_910, 335, 111_198,
    67_059, 38, 220, 837, 34, 25, 1_761, 25, 2_468, 335, 8_048, 4_282, 38, 220, 837, 34, 25, 2_961,
    25, 2_834, 1_574, 34_956, 300, 9_762, 38, 2_244, 1_574, 15, 14_757, 73_965, 38, 392, 2_540,
    706, 392, 1_556, 4_205, 200_008, 200_022, 1_556, 200_023, 51_015, 5_959, 290, 21_752, 14_774,
    10_950, 310, 1_085, 81_080, 24_429, 26, 200_008, 200_022, 140_680,
];

/// Reviewed greedy continuation produced by the pinned official F16 graph.
const REFERENCE_GENERATED_TOKEN_IDS: [u32; 32] = [
    328, 19_669, 200_023, 51_015, 5_959, 290, 21_752, 14_774, 10_950, 310, 1_085, 81_080, 24_429,
    1_574, 51_015, 5_959, 21_752, 14_774, 10_950, 310, 1_085, 81_080, 24_429, 1_574, 2_991, 1_657,
    81_080, 24_429, 26, 97_147, 10_758, 98_400,
];

/// Fixed create-only directory emitted by the official-reference generator.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const REFERENCE_BUNDLE_DIRECTORY: &str = "muse-glimmer-30b-a4e59da-reference.bundle";

/// Strict JSON record within the reference bundle.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const REFERENCE_JSON_FILENAME: &str = "muse-glimmer-30b-a4e59da-reference.json";

/// Full-vocabulary prompt-logit payload within the reference bundle.
const REFERENCE_LOGITS_FILENAME: &str = "muse-glimmer-30b-a4e59da-prompt-logits.f32le";

/// Last-token embedding/layer/final-norm trace within the reference bundle.
const REFERENCE_HIDDEN_FILENAME: &str = "muse-glimmer-30b-a4e59da-hidden-last.f32le";

/// All 32 recurrent cached full-vocabulary decision rows.
const REFERENCE_CACHED_LOGITS_FILENAME: &str =
    "muse-glimmer-30b-a4e59da-cached-prefix-logits.f32le";

/// All 32 stateless teacher-forced full-vocabulary decision rows.
const REFERENCE_TEACHER_LOGITS_FILENAME: &str =
    "muse-glimmer-30b-a4e59da-teacher-prefix-logits.f32le";

/// Cached-prefill last-token embedding/layer/final-norm trace.
const REFERENCE_CACHED_PREFILL_HIDDEN_FILENAME: &str =
    "muse-glimmer-30b-a4e59da-cached-prefill-hidden-last.f32le";

/// First true recurrent-decode embedding/layer/final-norm trace.
const REFERENCE_CACHED_DECODE_HIDDEN_FILENAME: &str =
    "muse-glimmer-30b-a4e59da-cached-decode-1-hidden-last.f32le";

/// Canonical digest of the nine independently pinned checkpoint artifacts.
const REFERENCE_ARTIFACT_MANIFEST_SHA256: &str =
    "5b8265f3a0fe6b07668caa8e5b949667b8f26f664550dfcc86d35f93fb7971ab";

/// Independently reviewed JSON output identity.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const TRUSTED_MUSE_REFERENCE_JSON_SHA256: Option<&str> =
    Some("cab22352cf833802b4e530b1ad6b6a223481f3a52378ce2c3bfb980e15543f10");

/// Numerical prompt-logit identity to pin independently of the JSON claim.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const TRUSTED_MUSE_REFERENCE_LOGITS_SHA256: Option<&str> =
    Some("c384665147d10a255110e9e034ddec557a2b49909e7105d98a5b541ff4c5d189");

/// Numerical hidden-trace identity to pin independently of the JSON claim.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const TRUSTED_MUSE_REFERENCE_HIDDEN_SHA256: Option<&str> =
    Some("6ce52668e97a612b0bf8c5adbb306eceda0d002aec8ed1726385b5dd11ab9c3f");

/// Independently reviewed cached-prefix payload identity.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const TRUSTED_MUSE_REFERENCE_CACHED_LOGITS_SHA256: Option<&str> =
    Some("864e67fef08cd4dcdd35b4e7181d8b4d9eb6a924d9fec04563ac91c3af719a0d");

/// Independently reviewed teacher-prefix payload identity.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const TRUSTED_MUSE_REFERENCE_TEACHER_LOGITS_SHA256: Option<&str> =
    Some("ff7eb300361f7ba7d036dacbf56cb8e4f8680ce0c01e427775623f03b09cc89d");

/// Independently reviewed cached-prefill hidden identity.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const TRUSTED_MUSE_REFERENCE_CACHED_PREFILL_HIDDEN_SHA256: Option<&str> =
    Some("6ce52668e97a612b0bf8c5adbb306eceda0d002aec8ed1726385b5dd11ab9c3f");

/// Independently reviewed recurrent-decode hidden identity.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
const TRUSTED_MUSE_REFERENCE_CACHED_DECODE_HIDDEN_SHA256: Option<&str> =
    Some("76a4b1e3a6a6b0fbb9dcc6696f90131f112b35e2ee9f117c2cca8d71549a80a4");

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

/// Builds the exact F16 text plan shared by conversion and packed execution.
#[cfg(feature = "tokenizers")]
fn pinned_muse_f16_plan(root: &Path) -> (CheckpointManifest, LoadPlan) {
    assert_eq!(
        fs::read(root.join("config.json")).expect("read pinned Muse config"),
        include_bytes!("fixtures/muse_glimmer_30b.json")
    );
    let config = MuseGlimmerTextConfig::from_json_str(
        std::str::from_utf8(include_bytes!("fixtures/muse_glimmer_30b.json"))
            .expect("pinned Muse config UTF-8"),
    )
    .expect("pinned Muse config must validate");
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        ..LoadOptions::default()
    };
    let manifest = CheckpointManifest::discover(root, &options)
        .expect("discover exact pinned Muse checkpoint");
    let contract = config
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .expect("derive exact pinned Muse contract");
    let plan = contract
        .build_load_plan(&manifest, &options)
        .expect("build exact pinned Muse F16 plan");
    assert_eq!(f16_plan_digest(&plan), F16_PLAN_SHA256);
    (manifest, plan)
}

/// Returns the exact architecture-declared matrix set that must execute packed.
#[cfg(feature = "tokenizers")]
fn expected_muse_quantized_targets() -> Vec<String> {
    let mut targets = (0..52)
        .flat_map(|layer| {
            [
                format!("model.layers.{layer}.self_attn.q_proj.weight"),
                format!("model.layers.{layer}.self_attn.k_proj.weight"),
                format!("model.layers.{layer}.self_attn.v_proj.weight"),
                format!("model.layers.{layer}.self_attn.gate_proj.weight"),
                format!("model.layers.{layer}.self_attn.o_proj.weight"),
                format!("model.layers.{layer}.mlp.gate_proj.weight"),
                format!("model.layers.{layer}.mlp.up_proj.weight"),
                format!("model.layers.{layer}.mlp.down_proj.weight"),
            ]
        })
        .chain(["lm_head.weight".to_owned()])
        .collect::<Vec<_>>();
    targets.sort();
    targets
}

/// Classifies every exact plan target without relying on raw checkpoint names.
#[cfg(feature = "tokenizers")]
fn muse_quantization_partition(
    plan: &LoadPlan,
) -> (Vec<String>, Vec<(String, QuantizationExclusion)>) {
    let mut quantized = Vec::new();
    let mut retained = Vec::new();
    for tensor in &plan.tensors {
        let shape = tensor
            .shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).expect("Muse shape must fit usize"))
            .collect::<Vec<_>>();
        match classify_weight_for_quantization(&tensor.target, &shape) {
            QuantizationEligibility::Eligible => quantized.push(tensor.target.clone()),
            QuantizationEligibility::Excluded(reason) => {
                retained.push((tensor.target.clone(), reason));
            }
        }
    }
    quantized.sort();
    retained.sort_by(|left, right| left.0.cmp(&right.0));
    (quantized, retained)
}

/// Proves the exact 417/210 Muse quantization boundary and G64 row alignment.
#[cfg(feature = "tokenizers")]
fn assert_muse_quantization_partition(plan: &LoadPlan) -> (Vec<String>, u64) {
    let (quantized, retained) = muse_quantization_partition(plan);
    assert_eq!(quantized, expected_muse_quantized_targets());
    assert_eq!(quantized.len(), QUANTIZED_TENSOR_COUNT);
    assert_eq!(retained.len(), RETAINED_F16_TENSOR_COUNT);
    assert_eq!(
        retained
            .iter()
            .filter(|(_, reason)| *reason == QuantizationExclusion::Embedding)
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["model.embed_tokens.weight"]
    );
    assert_eq!(
        retained
            .iter()
            .filter(|(_, reason)| *reason == QuantizationExclusion::Normalization)
            .count(),
        209
    );
    assert!(retained.iter().all(|(_, reason)| matches!(
        reason,
        QuantizationExclusion::Embedding | QuantizationExclusion::Normalization
    )));

    let quantized = quantized.into_iter().collect::<BTreeSet<_>>();
    let mut elements = 0_u64;
    for tensor in &plan.tensors {
        if !quantized.contains(&tensor.target) {
            continue;
        }
        assert_eq!(tensor.shape.len(), 2, "{} must be a matrix", tensor.target);
        assert_eq!(
            tensor.shape[1] % 64,
            0,
            "{} input width must align to G64",
            tensor.target
        );
        elements = elements
            .checked_add(tensor.shape.iter().copied().product::<u64>())
            .expect("Muse eligible element count must fit u64");
    }
    assert_eq!(elements, 26_508_558_336);
    assert_eq!(
        plan.memory.target_parameter_bytes - elements * 2,
        RETAINED_F16_TENSOR_BYTES
    );
    (quantized.into_iter().collect(), elements)
}

/// Returns immutable payload geometry for one accepted symmetric precision.
#[cfg(feature = "tokenizers")]
fn muse_packed_geometry(scheme: SymmetricQuantization) -> (u64, u64) {
    match scheme {
        SymmetricQuantization::Q8 => (Q8_PACKED_PAYLOAD_BYTES, Q8_LARGEST_PACKED_TENSOR_BYTES),
        SymmetricQuantization::Q4 => (Q4_PACKED_PAYLOAD_BYTES, Q4_LARGEST_PACKED_TENSOR_BYTES),
    }
}

/// Resolves an explicit sidecar path and constrains it to the checkpoint volume.
#[cfg(feature = "tokenizers")]
fn muse_burnpack_path(root: &Path, environment: &str, must_exist: bool) -> PathBuf {
    let supplied = PathBuf::from(
        std::env::var_os(environment)
            .unwrap_or_else(|| panic!("set {environment} to an absolute .bpk path")),
    );
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
        .expect("Burnpack path must contain a file name");
    let parent = supplied
        .parent()
        .expect("Burnpack path must have a parent")
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize Burnpack parent: {error}"));
    let resolved = parent.join(file_name);
    assert_eq!(
        resolved, supplied,
        "Burnpack parent must already be canonical"
    );
    #[cfg(unix)]
    assert_eq!(
        fs::metadata(root).expect("checkpoint root metadata").dev(),
        fs::metadata(&parent)
            .expect("Burnpack parent metadata")
            .dev(),
        "Burnpack and source checkpoint must reside on the same volume"
    );
    match fs::symlink_metadata(&resolved) {
        Ok(metadata) => {
            assert!(must_exist, "create-only Burnpack path already exists");
            assert!(
                !metadata.file_type().is_symlink(),
                "Burnpack must not be a symlink"
            );
            assert!(metadata.is_file(), "Burnpack must be a regular file");
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            assert!(!must_exist, "required Burnpack is unavailable");
        }
        Err(error) => panic!("inspect Burnpack path: {error}"),
    }
    resolved
}

/// Reopens one sidecar and proves exact target, payload, and source provenance.
#[cfg(feature = "tokenizers")]
fn assert_exact_muse_burnpack(
    artifact: &mut QuantizedBurnpackArtifact,
    plan: &LoadPlan,
    scheme: SymmetricQuantization,
    expected_quantized: &[String],
) {
    let (payload_bytes, largest_payload_bytes) = muse_packed_geometry(scheme);
    validate_sidecar_contract(plan, artifact).expect("validate Muse sidecar contract");
    let validation = artifact
        .validate_all()
        .expect("decode and validate every Muse sidecar payload");
    assert_eq!(validation.tensor_count, QUANTIZED_TENSOR_COUNT);
    assert_eq!(validation.payload_bytes, payload_bytes);
    let manifest = artifact.manifest();
    assert_eq!(
        manifest.format_version,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    assert_eq!(
        manifest.packed_execution,
        PackedExecutionSupport::BurnQFloatMatmul
    );
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
        payload_bytes
    );
    assert_eq!(
        manifest
            .tensors
            .values()
            .map(|tensor| tensor.packed_value_bytes + tensor.scale_bytes)
            .max(),
        Some(largest_payload_bytes)
    );
    let provenance = manifest
        .provenance
        .as_ref()
        .expect("Muse architecture conversion must publish v2 provenance");
    assert_eq!(provenance.load_plan_sha256, load_plan_sha256(plan));
    assert_eq!(provenance.checkpoint_shards.len(), SHARDS.len());
    for (relative_path, bytes, sha256) in SHARDS {
        let shard = provenance
            .checkpoint_shards
            .get(relative_path)
            .unwrap_or_else(|| panic!("Muse provenance omitted {relative_path}"));
        assert_eq!(shard.file_bytes, bytes);
        assert_eq!(shard.sha256, sha256);
    }
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

/// Returns the exact direct dependency versions authenticated by the oracle.
fn expected_reference_packages() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("PyYAML".to_owned(), "6.0.2".to_owned()),
        ("accelerate".to_owned(), "1.12.0".to_owned()),
        ("filelock".to_owned(), "3.17.0".to_owned()),
        ("huggingface-hub".to_owned(), "1.5.0".to_owned()),
        ("numpy".to_owned(), "2.4.3".to_owned()),
        ("packaging".to_owned(), "25.0".to_owned()),
        ("regex".to_owned(), "2026.1.15".to_owned()),
        ("safetensors".to_owned(), "0.8.0".to_owned()),
        ("tokenizers".to_owned(), "0.23.1".to_owned()),
        ("torch".to_owned(), "2.10.0".to_owned()),
        ("tqdm".to_owned(), "4.67.1".to_owned()),
    ])
}

/// Builds the independently compiled wheel-payload identities for schema validation.
fn expected_reference_distributions() -> BTreeMap<String, MuseReferenceDistribution> {
    let item = |version: &str,
                verified_files,
                verified_bytes,
                ignored_bytecode_files,
                excluded_external_scripts: &[&str],
                payload_manifest_sha256: &str| MuseReferenceDistribution {
        version: version.to_owned(),
        verified_files,
        verified_bytes,
        ignored_bytecode_files,
        excluded_external_scripts: excluded_external_scripts
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        payload_manifest_sha256: payload_manifest_sha256.to_owned(),
    };
    BTreeMap::from([
        (
            "PyYAML".to_owned(),
            item(
                "6.0.2",
                24,
                581_466,
                18,
                &[],
                "a2413d950416eb3e271ce2e79d0a0d9af772b2f509e4f6368ac21a21059951c8",
            ),
        ),
        (
            "accelerate".to_owned(),
            item(
                "1.12.0",
                90,
                1_482_046,
                84,
                &[
                    "../../../bin/accelerate",
                    "../../../bin/accelerate-config",
                    "../../../bin/accelerate-estimate-memory",
                    "../../../bin/accelerate-launch",
                    "../../../bin/accelerate-merge-weights",
                ],
                "a4777b667bf3cf092af2ac15403be4b1217f4eada9dea1c93729d0ba1fce374c",
            ),
        ),
        (
            "filelock".to_owned(),
            item(
                "3.17.0",
                14,
                42_043,
                9,
                &[],
                "fa010d76505458640b82b4245cc7e88a0279adaa5d35939a3f62632f86f41d9f",
            ),
        ),
        (
            "huggingface-hub".to_owned(),
            item(
                "1.5.0",
                174,
                2_379_101,
                164,
                &["../../../bin/hf", "../../../bin/tiny-agents"],
                "e31e33b4619feb0de20b31a105b3dd4a48a0b5c7082a879c39730cfab8950a38",
            ),
        ),
        (
            "numpy".to_owned(),
            item(
                "2.4.3",
                1_038,
                20_253_880,
                487,
                &["../../../bin/f2py", "../../../bin/numpy-config"],
                "ea91cc1904675f4a171c29d31fda752911857c256eef4b463f79b5f2647aa17c",
            ),
        ),
        (
            "packaging".to_owned(),
            item(
                "25.0",
                23,
                236_488,
                16,
                &[],
                "28a5d31e3899005c3c4765b86bbdf2617439c46a113eba215187948e1914cebb",
            ),
        ),
        (
            "regex".to_owned(),
            item(
                "2026.1.15",
                10,
                1_248_895,
                4,
                &[],
                "39c375db510bf8720c449985de413ec2b506bc22b5cecf51d0a0209817537a4f",
            ),
        ),
        (
            "safetensors".to_owned(),
            item(
                "0.8.0",
                16,
                1_211_818,
                7,
                &[],
                "cfa1be33517b651209e190e761fcdac179935a7df4fdbc8215bb20573cacb0b0",
            ),
        ),
        (
            "tokenizers".to_owned(),
            item(
                "0.23.1",
                33,
                8_879_675,
                16,
                &[],
                "384ef072334a048d972eecdb2d86f4b7e359fc3ac7b1054b3d00adb926f2c0c7",
            ),
        ),
        (
            "torch".to_owned(),
            item(
                "2.10.0",
                12_338,
                338_441_672,
                2_199,
                &["../../../bin/torchfrtrace", "../../../bin/torchrun"],
                "ac5e7ce059b386aeecec89677baacc9875bad01557c4947ed98e05d14634b3c6",
            ),
        ),
        (
            "tqdm".to_owned(),
            item(
                "4.67.1",
                40,
                222_872,
                32,
                &["../../../bin/tqdm"],
                "e821f85778c550267aa17a5c38b7db7e388892fbaf18cf996718c7d960be9704",
            ),
        ),
    ])
}

/// Reproduces the generator's canonical primary-wheel aggregate identity.
fn reference_distribution_digest(
    distributions: &BTreeMap<String, MuseReferenceDistribution>,
) -> String {
    let mut canonical = Vec::new();
    for (name, distribution) in distributions {
        canonical.extend_from_slice(name.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(distribution.version.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(distribution.verified_files.to_string().as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(distribution.verified_bytes.to_string().as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(distribution.ignored_bytecode_files.to_string().as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(distribution.payload_manifest_sha256.as_bytes());
        canonical.push(b'\n');
        for path in &distribution.excluded_external_scripts {
            canonical.extend_from_slice(b"external\0");
            canonical.extend_from_slice(path.as_bytes());
            canonical.push(b'\n');
        }
    }
    sha256_hex(&canonical)
}

/// Strict schema emitted by `generate_muse_glimmer_reference.py`.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseOfficialReference {
    /// Version of the strict official-reference JSON schema.
    schema_version: u32,
    /// Official model repository authenticated by the reference run.
    repository: String,
    /// Immutable checkpoint revision loaded by the reference generator.
    revision: String,
    /// Pinned official implementation and its source identity.
    reference: MuseReferenceImplementation,
    /// Exact rendered source prompt used for all reference comparisons.
    prompt: String,
    /// Tokenized prompt without additional special-token insertion.
    input_ids: Vec<u32>,
    /// Graph precision, cache geometry, import integrity, and execution controls.
    execution: MuseReferenceExecution,
    /// Interpreter, dependency, memory, and offline-environment provenance.
    environment: MuseReferenceEnvironment,
    /// Authenticated generator and checkpoint identities with strict load accounting.
    provenance: MuseReferenceProvenance,
    /// Descriptors of all numerical payloads in the external reference bundle.
    outputs: MuseReferenceOutputs,
    /// Greedy continuation, stability margins, and cached/stateless parity summaries.
    generation: MuseReferenceGeneration,
    /// Synchronized phase durations in seconds, retained only as diagnostics.
    timings_seconds: MuseReferenceTimings,
}

/// Exact official source identity used to compute the numerical oracle.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceImplementation {
    /// Repository containing the official reference implementation.
    repository: String,
    /// Immutable commit of the official implementation checkout.
    revision: String,
    /// Fully qualified official model class used for reference inference.
    class: String,
    /// Package version declared by the pinned official source checkout.
    source_version: String,
    /// Producer Transformers version embedded in the checkpoint configuration.
    checkpoint_config_transformers_version: String,
    /// Byte identity of the official Muse model implementation file.
    source: MuseReferenceSource,
    /// Whether the official implementation checkout was clean during generation.
    checkout_clean: bool,
}

/// One pinned source file from the official Transformers checkout.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceSource {
    /// Model implementation path relative to the official checkout.
    relative_path: String,
    /// Exact implementation-file length in bytes.
    bytes: u64,
    /// SHA-256 of the pinned official implementation bytes.
    sha256: String,
}

/// Deterministic graph, dtype, cache, and accelerator choices.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceExecution {
    /// Floating-point dtype requested from the official model loader.
    dtype: String,
    /// Device used for official tensor execution.
    device: String,
    /// Selected official attention implementation.
    attention_implementation: String,
    /// Score equation and the softmax precision boundary.
    attention_precision: MuseReferenceAttentionPrecision,
    /// Whether dynamically supplied model-hub kernels were enabled.
    hub_kernels_enabled: bool,
    /// Whether PyTorch deterministic algorithms were enabled.
    deterministic_algorithms: bool,
    /// Random seed recorded for official execution.
    seed: u64,
    /// Whether official generation used a decode cache.
    use_cache: bool,
    /// Whether each model call used an explicit all-valid attention mask.
    explicit_all_one_attention_masks: bool,
    /// PyTorch policy controlling F32 matrix multiplication precision.
    float32_matmul_precision: String,
    /// Fraction of device memory permitted to the MPS process.
    mps_per_process_memory_fraction: f64,
    /// Explicit limits of the reference preflight and postflight integrity checks.
    integrity_threat_model: String,
    /// Authenticated import-origin paths keyed by primary package or official source.
    authenticated_import_resolutions: BTreeMap<String, String>,
    /// Actual imported module paths keyed by primary package.
    imported_module_bindings: BTreeMap<String, String>,
    /// Controls forcing imports through authenticated source instead of stale bytecode.
    authenticated_import_bytecode_policy: MuseReferenceBytecodePolicy,
    /// Observed wrapper around the official MPS allocator-warmup no-op.
    allocator_warmup_override: MuseReferenceWarmupOverride,
    /// Official cache topology, lengths, and absolute positions for the short run.
    cache: MuseReferenceCacheContract,
}

/// Exact official DynamicCache topology and every short-run length/position.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceCacheContract {
    /// Fully qualified official cache implementation class.
    class: String,
    /// Batch size established from physical key/value tensor axis zero.
    batch_size: usize,
    /// Batch size reported by the cache API, including its unknown-value sentinel.
    api_reported_batch_size: i64,
    /// Recorded authority used to resolve the cache API batch-size sentinel.
    batch_size_authority: String,
    /// Number of cache-bearing transformer layers.
    layer_count: usize,
    /// Ordered sliding-attention and full-attention layer schedule.
    layer_types: Vec<String>,
    /// Number of layers retaining sliding-window key/value history.
    sliding_layers: usize,
    /// Number of layers retaining full key/value history.
    full_layers: usize,
    /// Maximum retained token window for sliding-attention layers.
    sliding_window: usize,
    /// Number of key/value heads in each cache tensor.
    key_value_heads: usize,
    /// Channels per key/value head.
    head_dimension: usize,
    /// Storage dtype of the official key/value cache tensors.
    tensor_dtype: String,
    /// Device holding the official key/value cache tensors.
    tensor_device: String,
    /// Consumed token count when each cached decision is produced.
    decision_logical_lengths: Vec<usize>,
    /// Physical key/value token extent observed at each short-run decision.
    decision_physical_lengths: Vec<usize>,
    /// Absolute position of the input whose output produces each decision.
    decision_position_ids: Vec<usize>,
    /// Absolute positions of the recurrent single-token inputs after prefill.
    decode_input_position_ids: Vec<usize>,
    /// Absolute input positions in the teacher-forced stateless comparison.
    teacher_position_ids: Vec<usize>,
    /// Whether implicitly inferred positions were checked against explicit positions.
    implicit_position_ids_verified: bool,
    /// Whether every layer cache was inspected at each decision boundary.
    all_layers_validated_at_every_decision: bool,
}

/// Controls that force imports through authenticated source rather than stale bytecode.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceBytecodePolicy {
    /// Whether adjacent cached Python bytecode was bypassed during authenticated imports.
    adjacent_bytecode_ignored: bool,
    /// Whether Python bytecode-cache writes were disabled.
    writes_disabled: bool,
    /// Whether the private import-cache directory remained empty.
    private_cache_remained_empty: bool,
    /// Whether the original interpreter bytecode settings were restored.
    restored: bool,
}

/// Official eager-attention score equation and mixed-precision boundary.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceAttentionPrecision {
    /// Recorded normalized-query/key scaling equation before softmax.
    score_equation: String,
    /// Accumulation dtype used to compute attention softmax.
    softmax_accumulator_dtype: String,
    /// Dtype of attention probabilities passed to the value contraction.
    probability_output_dtype: String,
}

/// Audited scope of the performance-only MPS allocator warmup override.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceWarmupOverride {
    /// Fully qualified allocator-warmup function observed by the wrapper.
    target: String,
    /// Named policy verifying the official MPS no-op branch.
    policy: String,
    /// Official loader call during which the wrapper was active.
    scope: String,
    /// Rationale for observing the warmup branch without changing tensor computation.
    reason: String,
    /// Number of allocator-warmup calls observed by the wrapper.
    call_count: usize,
    /// Number of entries inspected in the expanded parameter device map.
    expanded_device_map_entries: usize,
    /// Distinct device types present in the expanded map.
    expanded_device_types: Vec<String>,
    /// Whether a Hugging Face quantizer was passed to the wrapped call.
    hf_quantizer_present: bool,
    /// Whether the original allocator-warmup function was restored.
    restored: bool,
}

/// Python process, dependency, and offline-control provenance.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceEnvironment {
    /// UTC timestamp at which the reference record was generated.
    generated_at_utc: String,
    /// Exact interpreter version and virtual-environment identity.
    python: MuseReferencePython,
    /// Primary package versions keyed by distribution name.
    packages: BTreeMap<String, String>,
    /// Authenticated primary-wheel payload inventories keyed by distribution name.
    distribution_provenance: BTreeMap<String, MuseReferenceDistribution>,
    /// SHA-256 of the canonical primary-distribution provenance manifest.
    distribution_manifest_sha256: String,
    /// Optional packages required to be absent from the reference environment.
    absent_optional_distributions: Vec<String>,
    /// Versions of imported transitive distributions whose payload bytes remain trusted.
    transitive_distributions: BTreeMap<String, String>,
    /// Boundary between content-authenticated inputs and trusted environment state.
    trust_boundary: MuseReferenceTrustBoundary,
    /// MPS availability and conservative model-load memory preflight.
    device: MuseReferenceDevice,
    /// Environment variables enforcing offline execution and backend controls.
    offline_environment: BTreeMap<String, String>,
}

/// Exact CPython interpreter contract used by the official reference run.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferencePython {
    /// Complete interpreter version string.
    version: String,
    /// Structured interpreter version components.
    version_info: MuseReferencePythonVersion,
    /// Interpreter executable path reported by Python.
    executable: String,
    /// Resolved interpreter executable path after following environment links.
    resolved_executable: String,
    /// Active Python virtual-environment prefix.
    prefix: String,
    /// Base interpreter installation prefix outside the active virtual environment.
    base_prefix: String,
    /// Interpreter implementation name, required to be CPython.
    implementation: String,
    /// Platform description reported by the interpreter.
    platform: String,
}

/// Separates authenticated inputs from deliberately trusted environment state.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceTrustBoundary {
    /// Statement of the trusted interpreter, standard-library, and operating-system boundary.
    interpreter_and_stdlib: String,
    /// Whether interpreter and standard-library payload bytes were authenticated.
    interpreter_and_stdlib_content_authenticated: bool,
    /// Primary distributions whose installed payloads were checked against RECORD.
    record_authenticated_primary_distributions: Vec<String>,
    /// Statement of the trusted imported-transitive-distribution boundary.
    transitive_distributions: String,
    /// Whether imported transitive-distribution payload bytes were authenticated.
    transitive_distribution_content_authenticated: bool,
    /// Whether every imported module origin was attributed to a recorded source.
    all_module_origins_attributed: bool,
}

/// Structured Python version components that cannot be hidden in a free-form string.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferencePythonVersion {
    /// Major interpreter version component.
    major: u8,
    /// Minor interpreter version component.
    minor: u8,
    /// Patch interpreter version component.
    micro: u8,
    /// Interpreter release stage, required to be final for the pinned environment.
    releaselevel: String,
    /// Interpreter release-stage serial number.
    serial: u8,
}

/// Authenticated in-environment payload inventory for one exact wheel.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceDistribution {
    /// Installed version of the authenticated primary distribution.
    version: String,
    /// Number of wheel payload files verified against RECORD.
    verified_files: usize,
    /// Total bytes in the verified wheel payload files.
    verified_bytes: u64,
    /// Number of bytecode files ignored by the authenticated source-import policy.
    ignored_bytecode_files: usize,
    /// RECORD-listed external scripts excluded from the in-environment payload manifest.
    excluded_external_scripts: Vec<String>,
    /// SHA-256 of the canonical verified payload manifest for this wheel.
    payload_manifest_sha256: String,
}

/// MPS availability and conservative full-model load-memory preflight.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceDevice {
    /// Whether the MPS backend was available for reference execution.
    backend_available: bool,
    /// Device-recommended maximum working-set bytes observed before loading.
    recommended_max_memory_bytes: u64,
    /// Conservative full F16 model-load requirement, including reserved headroom, in bytes.
    required_f16_load_bytes: u64,
    /// Additional bytes reserved beyond checkpoint tensor payload for loading headroom.
    minimum_load_headroom_bytes: u64,
}

/// Generator and checkpoint identities recorded before model execution.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceProvenance {
    /// Exact committed generator source and repository state.
    generator: MuseReferenceGenerator,
    /// Checkpoint control files and weight shards authenticated for the run.
    artifacts: Vec<MuseReferenceArtifact>,
    /// SHA-256 of the canonical checkpoint artifact manifest.
    artifact_manifest_sha256: String,
    /// Number of physical tensors in the complete multimodal checkpoint.
    physical_tensor_count: usize,
    /// Total physical checkpoint tensor payload bytes.
    checkpoint_tensor_bytes: u64,
    /// Number of parameter elements loaded by the official model.
    loaded_parameter_count: u64,
    /// Loaded parameter element counts grouped by dtype.
    parameter_dtype_elements: BTreeMap<String, u64>,
    /// Official loader classifications, each required to be empty.
    loading_info: MuseReferenceLoadingInfo,
    /// Whether all post-execution artifact and source integrity checks passed.
    integrity_postflight_passed: bool,
}

/// Exact committed generator source that produced an external bundle.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceGenerator {
    /// Generator path relative to its repository root.
    relative_path: String,
    /// Exact generator-source length in bytes.
    bytes: usize,
    /// SHA-256 of the generator-source bytes.
    sha256: String,
    /// Generator repository HEAD recorded at reference creation.
    git_head: String,
    /// Git root-tree identity recorded for the generator repository.
    git_tree: String,
    /// Whether the generator repository had uncommitted changes.
    git_dirty: bool,
}

/// One authenticated checkpoint input.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceArtifact {
    /// Artifact path relative to the pinned checkpoint root.
    relative_path: String,
    /// Exact checkpoint artifact length in bytes.
    bytes: u64,
    /// SHA-256 of the exact checkpoint artifact bytes.
    sha256: String,
}

/// Strict empty-class summaries returned by official loading.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceLoadingInfo {
    /// Expected checkpoint keys absent during official loading.
    missing_keys: MuseReferenceDiagnostic,
    /// Checkpoint keys unused by the official model.
    unexpected_keys: MuseReferenceDiagnostic,
    /// Checkpoint keys whose shapes disagreed with the official model.
    mismatched_keys: MuseReferenceDiagnostic,
    /// Other errors reported by the official loader.
    error_msgs: MuseReferenceDiagnostic,
}

/// Count and canonical digest for one diagnostic class.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceDiagnostic {
    /// Number of entries in this loader diagnostic class.
    count: usize,
    /// SHA-256 of the canonical diagnostic-class contents.
    sha256: String,
}

/// External numerical payloads used by the Rust acceptance test.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceOutputs {
    /// Complete vocabulary logits from the final prompt position.
    prompt_logits: MuseReferencePromptLogits,
    /// Stateless prompt-final embedding, per-layer, and final-normalized hidden trace.
    hidden_last_trace: MuseReferenceHiddenTrace,
    /// Complete vocabulary logits for all 32 cached decisions.
    cached_prefix_logits: MuseReferenceLogitMatrix,
    /// Complete vocabulary logits at the corresponding teacher-forced positions.
    teacher_prefix_logits: MuseReferenceLogitMatrix,
    /// Prompt-final hidden trace produced by cached prefill.
    cached_prefill_hidden_trace: MuseReferenceHiddenTrace,
    /// Hidden trace produced by the first recurrent single-token decode.
    cached_decode_hidden_trace: MuseReferenceHiddenTrace,
}

/// Full-vocabulary F32LE prompt-logit descriptor and integrity summary.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferencePromptLogits {
    /// Prompt-logit payload path relative to the reference bundle.
    relative_path: String,
    /// Raw payload element encoding, required to be little-endian F32.
    encoding: String,
    /// Official computation dtype before exporting the prompt logits.
    source_dtype: String,
    /// Floating-point dtype stored in the exported payload.
    storage_dtype: String,
    /// Payload dimensions, consisting of the complete vocabulary width.
    shape: Vec<usize>,
    /// Number of vocabulary logits in the payload.
    elements: usize,
    /// Exact exported payload length in bytes.
    bytes: usize,
    /// SHA-256 of the raw prompt-logit payload.
    sha256: String,
    /// Whether every prompt logit is finite.
    finite: bool,
    /// Sum of all prompt-final vocabulary logits.
    sum: f64,
    /// Euclidean norm of the prompt-final logit vector.
    l2_norm: f64,
    /// Ten highest logits ordered with deterministic token-ID tie breaking.
    top10: Vec<MuseReferenceRankedLogit>,
    /// Vocabulary index of the largest prompt-final logit.
    top1_token_id: u32,
}

/// One deterministic top-logit record with token-ID tie breaking.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceRankedLogit {
    /// Vocabulary index of this ranked logit.
    token_id: u32,
    /// Reference logit at the associated vocabulary index.
    logit: f64,
}

/// F32LE last-token trace spanning embedding, every layer, and final norm.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceHiddenTrace {
    /// Hidden-trace payload path relative to the reference bundle.
    relative_path: String,
    /// Raw payload element encoding, required to be little-endian F32.
    encoding: String,
    /// Official hidden-state computation dtype before export.
    source_dtype: String,
    /// Floating-point dtype stored in the exported hidden trace.
    storage_dtype: String,
    /// Trace dimensions ordered as named activation rows and hidden channels.
    shape: Vec<usize>,
    /// Total scalar values across all trace rows.
    elements: usize,
    /// Exact hidden-trace payload length in bytes.
    bytes: usize,
    /// SHA-256 of the raw hidden-trace payload.
    sha256: String,
    /// Whether every hidden-trace value is finite.
    finite: bool,
    /// Ordered activation names for the embedding, block outputs, and final norm.
    labels: Vec<String>,
    /// Execution phase producing this trace: stateless, cached prefill, or cached decode.
    phase: String,
    /// Zero-based absolute input-token position represented by each trace row.
    token_position: usize,
}

/// F32LE matrix retaining every full-vocabulary autoregressive decision row.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceLogitMatrix {
    /// Logit-matrix payload path relative to the reference bundle.
    relative_path: String,
    /// Raw payload element encoding, required to be little-endian F32.
    encoding: String,
    /// Official logit computation dtype before export.
    source_dtype: String,
    /// Floating-point dtype stored in the exported logit matrix.
    storage_dtype: String,
    /// Matrix dimensions ordered as decision rows and vocabulary channels.
    shape: Vec<usize>,
    /// Total scalar logits across all decision rows.
    elements: usize,
    /// Exact logit-matrix payload length in bytes.
    bytes: usize,
    /// SHA-256 of the raw logit-matrix payload.
    sha256: String,
    /// Whether every exported matrix value is finite.
    finite: bool,
    /// Absolute input position whose output supplies each decision row.
    row_position_ids: Vec<usize>,
    /// Argmax vocabulary index for each decision row.
    row_top1_token_ids: Vec<u32>,
}

/// Cached and stateless top-1 evidence for 32 autoregressive prefixes.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceGeneration {
    /// Requested number of autoregressive decisions.
    max_new_tokens: usize,
    /// Token IDs treated as end-of-sequence by the reference run.
    eos_token_ids: Vec<u32>,
    /// Greedy continuation IDs excluding the source prompt.
    token_ids: Vec<u32>,
    /// Argmax token ID at every cached continuation prefix.
    cached_prefix_top1: Vec<u32>,
    /// Argmax token ID at the matching stateless continuation prefixes.
    stateless_prefix_top1: Vec<u32>,
    /// Top-two logits and binary16 stability evidence for each decision.
    decisions: Vec<MuseReferenceDecision>,
    /// Smallest top-two logit gap across all decisions.
    minimum_margin: f64,
    /// Smallest top-two gap divided by its elementwise binary16 perturbation budget.
    minimum_margin_to_epsilon_ratio: f64,
    /// Continuation text decoded by the official tokenizer.
    decoded_text: String,
    /// Cosine between stateless prompt logits and cached-prefill logits.
    prompt_cached_cosine: f64,
    /// Minimum cached-versus-stateless logit cosine over all continuation prefixes.
    minimum_cached_stateless_prefix_cosine: f64,
    /// Minimum cached-prefill-versus-stateless cosine across corresponding hidden-trace rows.
    minimum_cached_prefill_stateless_hidden_cosine: f64,
}

/// Binary16 perturbation evidence for one reference greedy decision.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceDecision {
    /// Zero-based position in the greedy continuation.
    step: usize,
    /// Vocabulary index of the largest decision logit.
    top1_token_id: u32,
    /// Vocabulary index of the runner-up decision logit.
    top2_token_id: u32,
    /// Largest decision logit.
    top1_logit: f64,
    /// Runner-up decision logit.
    top2_logit: f64,
    /// Difference between the highest and second-highest logits.
    margin: f64,
    /// One local binary16 ULP at the wider-spaced top-two logit.
    elementwise_f16_ulp: f64,
    /// Top-two logit gap divided by the elementwise binary16 perturbation budget.
    margin_to_epsilon_ratio: f64,
    /// Whether the top-two gap exceeds twice the one-ULP elementwise budget.
    stable_under_two_sided_one_ulp_budget: bool,
}

/// Synchronized wall-clock phases retained as non-acceptance diagnostics.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MuseReferenceTimings {
    /// Synchronized checkpoint loading duration in seconds.
    load: f64,
    /// Synchronized stateless prompt inference duration in seconds.
    stateless_prompt: f64,
    /// Synchronized cached continuation duration in seconds.
    cached_generation: f64,
    /// Synchronized teacher-forced stateless inference duration in seconds.
    teacher_forced_stateless: f64,
}

/// Returns all checkpoint identities as an independently compiled Rust inventory.
fn expected_reference_artifacts() -> Vec<MuseReferenceArtifact> {
    CONTROL_FILES
        .iter()
        .chain(SHARDS.iter())
        .map(|(relative_path, bytes, sha256)| MuseReferenceArtifact {
            relative_path: (*relative_path).to_owned(),
            bytes: *bytes,
            sha256: (*sha256).to_owned(),
        })
        .collect()
}

/// Reproduces the generator's canonical checkpoint-artifact digest.
fn reference_artifact_digest(artifacts: &[MuseReferenceArtifact]) -> String {
    let mut artifacts = artifacts.iter().collect::<Vec<_>>();
    artifacts.sort_unstable_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let mut canonical = Vec::new();
    for artifact in artifacts {
        canonical.extend_from_slice(artifact.relative_path.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(artifact.bytes.to_string().as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(artifact.sha256.as_bytes());
        canonical.push(b'\n');
    }
    sha256_hex(&canonical)
}

/// Returns the exact 54 semantic trace labels in execution order.
fn expected_hidden_trace_labels() -> Vec<String> {
    std::iter::once("embed_norm".to_owned())
        .chain((0..52).map(|index| format!("layers.{index}")))
        .chain(std::iter::once("norm".to_owned()))
        .collect()
}

/// Returns whether a value is one canonical lowercase SHA-256 digest.
fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Returns whether a reference payload name is one safe ordinary component.
fn is_safe_filename(value: &str) -> bool {
    let mut components = Path::new(value).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

/// Validates the strict record without trusting any output's self-declared hash.
fn validate_reference_schema(reference: &MuseOfficialReference) -> Result<(), String> {
    let ensure = |condition: bool, message: &'static str| {
        condition.then_some(()).ok_or_else(|| message.to_owned())
    };
    ensure(
        reference.schema_version == 3,
        "reference schema version changed",
    )?;
    ensure(
        reference.repository == REPOSITORY,
        "reference repository changed",
    )?;
    ensure(reference.revision == REVISION, "reference revision changed")?;
    ensure(
        reference.reference.repository == REFERENCE_REPOSITORY,
        "official implementation repository changed",
    )?;
    ensure(
        reference.reference.revision == REFERENCE_REVISION,
        "official implementation revision changed",
    )?;
    ensure(
        reference.reference.class == "transformers.MuseGlimmerForConditionalGeneration",
        "official reference class changed",
    )?;
    ensure(
        reference.reference.source_version == REFERENCE_SOURCE_VERSION
            && reference.reference.checkpoint_config_transformers_version
                == CHECKPOINT_CONFIG_TRANSFORMERS_VERSION,
        "official source or checkpoint producer version changed",
    )?;
    ensure(
        reference.reference.checkout_clean,
        "official checkout was dirty",
    )?;
    ensure(
        reference.reference.source.relative_path == REFERENCE_SOURCE
            && reference.reference.source.bytes == REFERENCE_SOURCE_BYTES
            && reference.reference.source.sha256 == REFERENCE_SOURCE_SHA256,
        "official Muse source identity changed",
    )?;
    ensure(
        reference.prompt == REFERENCE_PROMPT,
        "reference prompt changed",
    )?;
    ensure(
        reference.input_ids == REFERENCE_TOKEN_IDS,
        "reference tokenization changed",
    )?;

    let execution = &reference.execution;
    ensure(execution.dtype == "float16", "reference dtype changed")?;
    ensure(execution.device == "mps", "reference device changed")?;
    ensure(
        execution.attention_implementation == "eager",
        "reference attention implementation changed",
    )?;
    ensure(
        execution.attention_precision.score_equation
            == "(rms(q) * qk_scale_factor) @ rms(k)^T / sqrt(head_dim)"
            && execution.attention_precision.softmax_accumulator_dtype == "float32"
            && execution.attention_precision.probability_output_dtype == "float16",
        "reference attention precision contract changed",
    )?;
    ensure(
        !execution.hub_kernels_enabled,
        "reference execution enabled mutable Hub kernels",
    )?;
    ensure(
        execution.deterministic_algorithms && execution.seed == 0 && execution.use_cache,
        "deterministic execution controls changed",
    )?;
    ensure(
        execution.explicit_all_one_attention_masks
            && execution.float32_matmul_precision == "highest"
            && execution.mps_per_process_memory_fraction == 1.0,
        "reference precision or mask controls changed",
    )?;
    let expected_layer_types = (0..52)
        .map(|index| {
            if index % 4 == 3 {
                "full_attention"
            } else {
                "sliding_attention"
            }
            .to_owned()
        })
        .collect::<Vec<_>>();
    let expected_decision_lengths = (REFERENCE_TOKEN_IDS.len()
        ..REFERENCE_TOKEN_IDS.len() + REFERENCE_GENERATED_TOKEN_IDS.len())
        .collect::<Vec<_>>();
    let expected_decision_positions = (REFERENCE_TOKEN_IDS.len() - 1
        ..REFERENCE_TOKEN_IDS.len() - 1 + REFERENCE_GENERATED_TOKEN_IDS.len())
        .collect::<Vec<_>>();
    let cache = &execution.cache;
    ensure(
        cache.class == "transformers.cache_utils.DynamicCache"
            && cache.batch_size == 1
            && cache.api_reported_batch_size == -1
            && cache.batch_size_authority == "physical_key_value_tensor_axis_0"
            && cache.layer_count == 52
            && cache.layer_types == expected_layer_types
            && cache.sliding_layers == 39
            && cache.full_layers == 13
            && cache.sliding_window == 2_048
            && cache.key_value_heads == 2
            && cache.head_dimension == 128
            && cache.tensor_dtype == "float16"
            && cache.tensor_device == "mps"
            && cache.decision_logical_lengths == expected_decision_lengths
            && cache.decision_physical_lengths == expected_decision_lengths
            && cache.decision_position_ids == expected_decision_positions
            && cache.decode_input_position_ids == expected_decision_positions[1..]
            && cache.teacher_position_ids
                == (0..REFERENCE_TOKEN_IDS.len() + REFERENCE_GENERATED_TOKEN_IDS.len() - 1)
                    .collect::<Vec<_>>()
            && cache.implicit_position_ids_verified
            && cache.all_layers_validated_at_every_decision,
        "reference cache geometry, lengths, or positions changed",
    )?;
    ensure(
        execution.integrity_threat_model == REFERENCE_INTEGRITY_THREAT_MODEL,
        "reference integrity threat model changed",
    )?;
    let expected_package_keys = expected_reference_packages()
        .into_keys()
        .collect::<BTreeSet<_>>();
    let mut expected_resolution_keys = expected_package_keys.clone();
    expected_resolution_keys.insert("transformers-source".to_owned());
    ensure(
        execution
            .authenticated_import_resolutions
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
            == expected_resolution_keys
            && execution
                .imported_module_bindings
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>()
                == expected_package_keys
            && execution
                .authenticated_import_resolutions
                .values()
                .chain(execution.imported_module_bindings.values())
                .all(|path| Path::new(path).is_absolute()),
        "authenticated import bindings changed",
    )?;
    let bytecode = &execution.authenticated_import_bytecode_policy;
    ensure(
        bytecode.adjacent_bytecode_ignored
            && bytecode.writes_disabled
            && bytecode.private_cache_remained_empty
            && bytecode.restored,
        "authenticated bytecode policy changed",
    )?;
    let warmup = &execution.allocator_warmup_override;
    ensure(
        warmup.target == "transformers.modeling_utils.caching_allocator_warmup"
            && warmup.policy == "mps-official-noop-verified-during-from-pretrained"
            && warmup.scope == "MuseGlimmerForConditionalGeneration.from_pretrained"
            && warmup.reason == REFERENCE_WARMUP_REASON
            && warmup.call_count == 1
            && warmup.expanded_device_map_entries == PHYSICAL_TENSOR_COUNT
            && warmup.expanded_device_types == ["mps"]
            && !warmup.hf_quantizer_present
            && warmup.restored,
        "allocator warmup override changed",
    )?;

    let environment = &reference.environment;
    ensure(
        environment.generated_at_utc.ends_with('Z')
            && environment.python.version.starts_with("3.11.9 ")
            && Path::new(&environment.python.executable).is_absolute()
            && Path::new(&environment.python.resolved_executable).is_absolute()
            && Path::new(&environment.python.prefix).is_absolute()
            && Path::new(&environment.python.base_prefix).is_absolute()
            && environment.python.prefix != environment.python.base_prefix
            && environment.python.implementation == "CPython"
            && !environment.python.platform.is_empty()
            && environment.python.version_info.major == 3
            && environment.python.version_info.minor == 11
            && environment.python.version_info.micro == 9
            && environment.python.version_info.releaselevel == "final"
            && environment.python.version_info.serial == 0,
        "Python environment metadata changed",
    )?;
    ensure(
        environment.packages == expected_reference_packages()
            && environment.distribution_provenance == expected_reference_distributions()
            && reference_distribution_digest(&environment.distribution_provenance)
                == REFERENCE_DISTRIBUTION_MANIFEST_SHA256
            && environment.distribution_manifest_sha256 == REFERENCE_DISTRIBUTION_MANIFEST_SHA256
            && environment.absent_optional_distributions == ["flash-attn", "kernels", "torchao"],
        "reference dependency versions changed",
    )?;
    let primary_distributions = expected_reference_packages()
        .into_keys()
        .collect::<Vec<_>>();
    let trust = &environment.trust_boundary;
    ensure(
        trust.interpreter_and_stdlib == REFERENCE_INTERPRETER_TRUST_BOUNDARY
            && !trust.interpreter_and_stdlib_content_authenticated
            && trust.record_authenticated_primary_distributions == primary_distributions
            && trust.transitive_distributions == REFERENCE_TRANSITIVE_TRUST_BOUNDARY
            && !trust.transitive_distribution_content_authenticated
            && trust.all_module_origins_attributed,
        "runtime trust boundary changed",
    )?;
    let primary_keys = expected_reference_packages()
        .into_keys()
        .collect::<BTreeSet<_>>();
    ensure(
        environment
            .transitive_distributions
            .get("typing_extensions")
            .is_some_and(|version| !version.is_empty())
            && environment
                .transitive_distributions
                .iter()
                .all(|(name, version)| {
                    !name.is_empty() && !version.is_empty() && !primary_keys.contains(name)
                }),
        "recorded transitive distribution set changed",
    )?;
    ensure(
        environment.device.backend_available
            && environment.device.minimum_load_headroom_bytes == 2_000_000_000
            && environment.device.required_f16_load_bytes
                == CHECKPOINT_TENSOR_BYTES + environment.device.minimum_load_headroom_bytes
            && environment.device.recommended_max_memory_bytes
                >= environment.device.required_f16_load_bytes,
        "reference MPS memory preflight changed",
    )?;
    ensure(
        environment.offline_environment
            == BTreeMap::from([
                ("HF_DATASETS_OFFLINE".to_owned(), "1".to_owned()),
                ("HF_DEACTIVATE_ASYNC_LOAD".to_owned(), "1".to_owned()),
                ("HF_HUB_OFFLINE".to_owned(), "1".to_owned()),
                ("PYTORCH_ENABLE_MPS_FALLBACK".to_owned(), "0".to_owned()),
                ("TOKENIZERS_PARALLELISM".to_owned(), "false".to_owned()),
                ("TRANSFORMERS_OFFLINE".to_owned(), "1".to_owned()),
                ("USE_HUB_KERNELS".to_owned(), "0".to_owned()),
            ]),
        "offline environment controls changed",
    )?;

    let provenance = &reference.provenance;
    ensure(
        provenance.generator.relative_path == "scripts/generate_muse_glimmer_reference.py",
        "generator path changed",
    )?;
    let generator_bytes = include_bytes!("../scripts/generate_muse_glimmer_reference.py");
    ensure(
        provenance.generator.bytes == generator_bytes.len()
            && provenance.generator.sha256 == sha256_hex(generator_bytes)
            && provenance.generator.git_head.len() == 40
            && provenance.generator.git_tree.len() == 40
            && provenance
                .generator
                .git_head
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            && provenance
                .generator
                .git_tree
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            && !provenance.generator.git_dirty,
        "generator provenance changed",
    )?;
    let expected_artifacts = expected_reference_artifacts()
        .into_iter()
        .map(|artifact| (artifact.relative_path.clone(), artifact))
        .collect::<BTreeMap<_, _>>();
    let observed_artifacts = provenance
        .artifacts
        .iter()
        .cloned()
        .map(|artifact| (artifact.relative_path.clone(), artifact))
        .collect::<BTreeMap<_, _>>();
    ensure(
        observed_artifacts.len() == provenance.artifacts.len()
            && observed_artifacts == expected_artifacts,
        "reference checkpoint artifact inventory changed",
    )?;
    ensure(
        reference_artifact_digest(&provenance.artifacts) == REFERENCE_ARTIFACT_MANIFEST_SHA256
            && provenance.artifact_manifest_sha256 == REFERENCE_ARTIFACT_MANIFEST_SHA256,
        "checkpoint artifact manifest digest changed",
    )?;
    ensure(
        provenance.physical_tensor_count == PHYSICAL_TENSOR_COUNT
            && provenance.checkpoint_tensor_bytes == CHECKPOINT_TENSOR_BYTES
            && provenance.loaded_parameter_count == 29_776_626_688
            && provenance.parameter_dtype_elements
                == BTreeMap::from([("float16".to_owned(), 29_776_626_688)])
            && provenance.integrity_postflight_passed,
        "loaded model size or dtype inventory changed",
    )?;
    let empty_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    for diagnostic in [
        &provenance.loading_info.missing_keys,
        &provenance.loading_info.unexpected_keys,
        &provenance.loading_info.mismatched_keys,
        &provenance.loading_info.error_msgs,
    ] {
        ensure(
            diagnostic.count == 0 && diagnostic.sha256 == empty_sha,
            "official loading diagnostics were nonempty",
        )?;
    }

    let logits = &reference.outputs.prompt_logits;
    ensure(
        logits.relative_path == REFERENCE_LOGITS_FILENAME
            && is_safe_filename(&logits.relative_path)
            && logits.encoding == "ieee754-f32-little-endian"
            && logits.source_dtype == "float16"
            && logits.storage_dtype == "float32"
            && logits.shape == [202_048]
            && logits.elements == 202_048
            && logits.bytes == 202_048 * size_of::<f32>()
            && is_sha256(&logits.sha256)
            && logits.finite
            && logits.sum.is_finite()
            && logits.l2_norm.is_finite()
            && logits.l2_norm > 0.0
            && logits.top10.len() == 10
            && logits
                .top10
                .iter()
                .all(|entry| entry.token_id < 202_048 && entry.logit.is_finite()),
        "prompt-logit descriptor changed",
    )?;
    let hidden = &reference.outputs.hidden_last_trace;
    ensure(
        hidden.relative_path == REFERENCE_HIDDEN_FILENAME
            && is_safe_filename(&hidden.relative_path)
            && hidden.encoding == "ieee754-f32-little-endian"
            && hidden.source_dtype == "float16"
            && hidden.storage_dtype == "float32"
            && hidden.shape == [54, 6_656]
            && hidden.elements == 54 * 6_656
            && hidden.bytes == 54 * 6_656 * size_of::<f32>()
            && is_sha256(&hidden.sha256)
            && hidden.finite
            && hidden.labels == expected_hidden_trace_labels()
            && hidden.phase == "stateless_prompt"
            && hidden.token_position == REFERENCE_TOKEN_IDS.len() - 1,
        "hidden-trace descriptor changed",
    )?;
    let cached_prefill_hidden = &reference.outputs.cached_prefill_hidden_trace;
    ensure(
        cached_prefill_hidden.relative_path == REFERENCE_CACHED_PREFILL_HIDDEN_FILENAME
            && is_safe_filename(&cached_prefill_hidden.relative_path)
            && cached_prefill_hidden.encoding == "ieee754-f32-little-endian"
            && cached_prefill_hidden.source_dtype == "float16"
            && cached_prefill_hidden.storage_dtype == "float32"
            && cached_prefill_hidden.shape == [54, 6_656]
            && cached_prefill_hidden.elements == 54 * 6_656
            && cached_prefill_hidden.bytes == 54 * 6_656 * size_of::<f32>()
            && is_sha256(&cached_prefill_hidden.sha256)
            && cached_prefill_hidden.finite
            && cached_prefill_hidden.labels == expected_hidden_trace_labels()
            && cached_prefill_hidden.phase == "cached_prefill"
            && cached_prefill_hidden.token_position == REFERENCE_TOKEN_IDS.len() - 1,
        "cached-prefill hidden-trace descriptor changed",
    )?;
    let cached_decode_hidden = &reference.outputs.cached_decode_hidden_trace;
    ensure(
        cached_decode_hidden.relative_path == REFERENCE_CACHED_DECODE_HIDDEN_FILENAME
            && is_safe_filename(&cached_decode_hidden.relative_path)
            && cached_decode_hidden.encoding == "ieee754-f32-little-endian"
            && cached_decode_hidden.source_dtype == "float16"
            && cached_decode_hidden.storage_dtype == "float32"
            && cached_decode_hidden.shape == [54, 6_656]
            && cached_decode_hidden.elements == 54 * 6_656
            && cached_decode_hidden.bytes == 54 * 6_656 * size_of::<f32>()
            && is_sha256(&cached_decode_hidden.sha256)
            && cached_decode_hidden.finite
            && cached_decode_hidden.labels == expected_hidden_trace_labels()
            && cached_decode_hidden.phase == "cached_decode"
            && cached_decode_hidden.token_position == REFERENCE_TOKEN_IDS.len(),
        "cached-decode hidden-trace descriptor changed",
    )?;
    for (label, matrix, filename) in [
        (
            "cached-prefix",
            &reference.outputs.cached_prefix_logits,
            REFERENCE_CACHED_LOGITS_FILENAME,
        ),
        (
            "teacher-prefix",
            &reference.outputs.teacher_prefix_logits,
            REFERENCE_TEACHER_LOGITS_FILENAME,
        ),
    ] {
        ensure(
            matrix.relative_path == filename
                && is_safe_filename(&matrix.relative_path)
                && matrix.encoding == "ieee754-f32-little-endian"
                && matrix.source_dtype == "float16"
                && matrix.storage_dtype == "float32"
                && matrix.shape == [32, 202_048]
                && matrix.elements == 32 * 202_048
                && matrix.bytes == 32 * 202_048 * size_of::<f32>()
                && is_sha256(&matrix.sha256)
                && matrix.finite
                && matrix.row_position_ids == expected_decision_positions
                && matrix.row_top1_token_ids == REFERENCE_GENERATED_TOKEN_IDS,
            if label == "cached-prefix" {
                "cached-prefix logit descriptor changed"
            } else {
                "teacher-prefix logit descriptor changed"
            },
        )?;
    }

    let generation = &reference.generation;
    ensure(
        generation.max_new_tokens == 32
            && generation.eos_token_ids == [200_001, 200_008]
            && generation.token_ids == REFERENCE_GENERATED_TOKEN_IDS
            && generation.cached_prefix_top1 == generation.token_ids
            && generation.stateless_prefix_top1 == generation.token_ids
            && generation.decisions.len() == 32
            && !generation.decoded_text.is_empty()
            && (0.999..=1.0 + 1.0e-12).contains(&generation.prompt_cached_cosine)
            && (0.999..=1.0 + 1.0e-12).contains(&generation.minimum_cached_stateless_prefix_cosine)
            && (0.999..=1.0 + 1.0e-12)
                .contains(&generation.minimum_cached_prefill_stateless_hidden_cosine),
        "generation topology or cached/stateless parity changed",
    )?;
    ensure(
        generation
            .token_ids
            .iter()
            .all(|token| *token < 202_048 && !generation.eos_token_ids.contains(token)),
        "reference generation emitted an invalid or EOS token",
    )?;
    for (step, decision) in generation.decisions.iter().enumerate() {
        ensure(
            decision.step == step
                && decision.top1_token_id == generation.token_ids[step]
                && decision.top1_token_id != decision.top2_token_id
                && decision.top1_logit.is_finite()
                && decision.top2_logit.is_finite()
                && decision.margin.is_finite()
                && decision.margin > 0.0
                && decision.margin == decision.top1_logit - decision.top2_logit
                && decision.elementwise_f16_ulp.is_finite()
                && decision.elementwise_f16_ulp > 0.0
                && decision.margin_to_epsilon_ratio
                    == decision.margin / decision.elementwise_f16_ulp
                && decision.margin_to_epsilon_ratio > 2.0
                && decision.stable_under_two_sided_one_ulp_budget,
            "greedy stability evidence changed",
        )?;
    }
    let minimum_margin = generation
        .decisions
        .iter()
        .map(|decision| decision.margin)
        .reduce(f64::min)
        .ok_or_else(|| "missing greedy margins".to_owned())?;
    let minimum_ratio = generation
        .decisions
        .iter()
        .map(|decision| decision.margin_to_epsilon_ratio)
        .reduce(f64::min)
        .ok_or_else(|| "missing greedy ratios".to_owned())?;
    ensure(
        generation.minimum_margin == minimum_margin
            && generation.minimum_margin_to_epsilon_ratio == minimum_ratio
            && logits.top1_token_id == generation.token_ids[0]
            && logits.top10[0].token_id == logits.top1_token_id,
        "summary values do not match detailed evidence",
    )?;
    for timing in [
        reference.timings_seconds.load,
        reference.timings_seconds.stateless_prompt,
        reference.timings_seconds.cached_generation,
        reference.timings_seconds.teacher_forced_stateless,
    ] {
        ensure(timing.is_finite() && timing >= 0.0, "invalid timing")?;
    }
    Ok(())
}

/// Builds a compact, internally consistent record for weight-free schema tests.
fn synthetic_reference_json() -> serde_json::Value {
    let generator_bytes = include_bytes!("../scripts/generate_muse_glimmer_reference.py");
    let artifacts = expected_reference_artifacts()
        .into_iter()
        .map(|artifact| {
            serde_json::json!({
                "relative_path": artifact.relative_path,
                "bytes": artifact.bytes,
                "sha256": artifact.sha256,
            })
        })
        .collect::<Vec<_>>();
    let token_ids = REFERENCE_GENERATED_TOKEN_IDS.to_vec();
    let decisions = (0..32)
        .map(|step| {
            let top1 = token_ids[step];
            let top2 = if top1 == 8 { 9 } else { 8 };
            serde_json::json!({
                "step": step,
                "top1_token_id": top1,
                "top2_token_id": top2,
                "top1_logit": 2.0,
                "top2_logit": 1.0,
                "margin": 1.0,
                "elementwise_f16_ulp": 0.0009765625,
                "margin_to_epsilon_ratio": 1024.0,
                "stable_under_two_sided_one_ulp_budget": true,
            })
        })
        .collect::<Vec<_>>();
    let empty_diagnostic = serde_json::json!({
        "count": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    });
    let top10 = (0..10)
        .map(|rank| {
            serde_json::json!({
                "token_id": if rank == 0 { REFERENCE_GENERATED_TOKEN_IDS[0] } else { rank },
                "logit": 2.0 - f64::from(rank) * 0.125,
            })
        })
        .collect::<Vec<_>>();
    let packages = expected_reference_packages();
    let distributions = expected_reference_distributions();
    let bindings = packages
        .keys()
        .map(|name| (name.clone(), format!("/fixture/site-packages/{name}")))
        .collect::<BTreeMap<_, _>>();
    let mut resolutions = bindings.clone();
    resolutions.insert(
        "transformers-source".to_owned(),
        "/fixture/transformers/src/transformers/__init__.py".to_owned(),
    );

    serde_json::json!({
        "schema_version": 3,
        "repository": REPOSITORY,
        "revision": REVISION,
        "reference": {
            "repository": REFERENCE_REPOSITORY,
            "revision": REFERENCE_REVISION,
            "class": "transformers.MuseGlimmerForConditionalGeneration",
            "source_version": REFERENCE_SOURCE_VERSION,
            "checkpoint_config_transformers_version": CHECKPOINT_CONFIG_TRANSFORMERS_VERSION,
            "source": {
                "relative_path": REFERENCE_SOURCE,
                "bytes": REFERENCE_SOURCE_BYTES,
                "sha256": REFERENCE_SOURCE_SHA256,
            },
            "checkout_clean": true,
        },
        "prompt": REFERENCE_PROMPT,
        "input_ids": REFERENCE_TOKEN_IDS.as_slice(),
        "execution": {
            "dtype": "float16",
            "device": "mps",
            "attention_implementation": "eager",
            "attention_precision": {
                "score_equation": "(rms(q) * qk_scale_factor) @ rms(k)^T / sqrt(head_dim)",
                "softmax_accumulator_dtype": "float32",
                "probability_output_dtype": "float16",
            },
            "hub_kernels_enabled": false,
            "deterministic_algorithms": true,
            "seed": 0,
            "use_cache": true,
            "explicit_all_one_attention_masks": true,
            "float32_matmul_precision": "highest",
            "mps_per_process_memory_fraction": 1.0,
            "integrity_threat_model": REFERENCE_INTEGRITY_THREAT_MODEL,
            "authenticated_import_resolutions": resolutions,
            "imported_module_bindings": bindings,
            "authenticated_import_bytecode_policy": {
                "adjacent_bytecode_ignored": true,
                "writes_disabled": true,
                "private_cache_remained_empty": true,
                "restored": true,
            },
            "allocator_warmup_override": {
                "target": "transformers.modeling_utils.caching_allocator_warmup",
                "policy": "mps-official-noop-verified-during-from-pretrained",
                "scope": "MuseGlimmerForConditionalGeneration.from_pretrained",
                "reason": REFERENCE_WARMUP_REASON,
                "call_count": 1,
                "expanded_device_map_entries": PHYSICAL_TENSOR_COUNT,
                "expanded_device_types": ["mps"],
                "hf_quantizer_present": false,
                "restored": true,
            },
            "cache": {
                "class": "transformers.cache_utils.DynamicCache",
                "batch_size": 1,
                "api_reported_batch_size": -1,
                "batch_size_authority": "physical_key_value_tensor_axis_0",
                "layer_count": 52,
                "layer_types": (0..52).map(|index| if index % 4 == 3 { "full_attention" } else { "sliding_attention" }).collect::<Vec<_>>(),
                "sliding_layers": 39,
                "full_layers": 13,
                "sliding_window": 2_048,
                "key_value_heads": 2,
                "head_dimension": 128,
                "tensor_dtype": "float16",
                "tensor_device": "mps",
                "decision_logical_lengths": (67..99).collect::<Vec<_>>(),
                "decision_physical_lengths": (67..99).collect::<Vec<_>>(),
                "decision_position_ids": (66..98).collect::<Vec<_>>(),
                "decode_input_position_ids": (67..98).collect::<Vec<_>>(),
                "teacher_position_ids": (0..98).collect::<Vec<_>>(),
                "implicit_position_ids_verified": true,
                "all_layers_validated_at_every_decision": true,
            },
        },
        "environment": {
            "generated_at_utc": "2026-09-05T00:00:00Z",
            "python": {
                "version": "3.11.9 fixture",
                "version_info": {
                    "major": 3,
                    "minor": 11,
                    "micro": 9,
                    "releaselevel": "final",
                    "serial": 0,
                },
                "executable": "/fixture/bin/python",
                "resolved_executable": "/fixture/base/bin/python3.11",
                "prefix": "/fixture",
                "base_prefix": "/fixture/base",
                "implementation": "CPython",
                "platform": "fixture",
            },
            "packages": packages,
            "distribution_provenance": distributions,
            "distribution_manifest_sha256": REFERENCE_DISTRIBUTION_MANIFEST_SHA256,
            "absent_optional_distributions": ["flash-attn", "kernels", "torchao"],
            "transitive_distributions": {"typing_extensions": "4.15.0"},
            "trust_boundary": {
                "interpreter_and_stdlib": REFERENCE_INTERPRETER_TRUST_BOUNDARY,
                "interpreter_and_stdlib_content_authenticated": false,
                "record_authenticated_primary_distributions": expected_reference_packages().into_keys().collect::<Vec<_>>(),
                "transitive_distributions": REFERENCE_TRANSITIVE_TRUST_BOUNDARY,
                "transitive_distribution_content_authenticated": false,
                "all_module_origins_attributed": true,
            },
            "device": {
                "backend_available": true,
                "recommended_max_memory_bytes": 70_000_000_000_u64,
                "required_f16_load_bytes": CHECKPOINT_TENSOR_BYTES + 2_000_000_000,
                "minimum_load_headroom_bytes": 2_000_000_000_u64,
            },
            "offline_environment": {
                "HF_DATASETS_OFFLINE": "1",
                "HF_DEACTIVATE_ASYNC_LOAD": "1",
                "HF_HUB_OFFLINE": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
                "USE_HUB_KERNELS": "0",
            },
        },
        "provenance": {
            "generator": {
                "relative_path": "scripts/generate_muse_glimmer_reference.py",
                "bytes": generator_bytes.len(),
                "sha256": sha256_hex(generator_bytes),
                "git_head": "0000000000000000000000000000000000000000",
                "git_tree": "1111111111111111111111111111111111111111",
                "git_dirty": false,
            },
            "artifacts": artifacts,
            "artifact_manifest_sha256": REFERENCE_ARTIFACT_MANIFEST_SHA256,
            "physical_tensor_count": PHYSICAL_TENSOR_COUNT,
            "checkpoint_tensor_bytes": CHECKPOINT_TENSOR_BYTES,
            "loaded_parameter_count": 29_776_626_688_u64,
            "parameter_dtype_elements": {"float16": 29_776_626_688_u64},
            "loading_info": {
                "missing_keys": empty_diagnostic.clone(),
                "unexpected_keys": empty_diagnostic.clone(),
                "mismatched_keys": empty_diagnostic.clone(),
                "error_msgs": empty_diagnostic,
            },
            "integrity_postflight_passed": true,
        },
        "outputs": {
            "prompt_logits": {
                "relative_path": REFERENCE_LOGITS_FILENAME,
                "encoding": "ieee754-f32-little-endian",
                "source_dtype": "float16",
                "storage_dtype": "float32",
                "shape": [202_048],
                "elements": 202_048,
                "bytes": 202_048 * size_of::<f32>(),
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "finite": true,
                "sum": 0.0,
                "l2_norm": 1.0,
                "top10": top10,
                "top1_token_id": REFERENCE_GENERATED_TOKEN_IDS[0],
            },
            "hidden_last_trace": {
                "relative_path": REFERENCE_HIDDEN_FILENAME,
                "encoding": "ieee754-f32-little-endian",
                "source_dtype": "float16",
                "storage_dtype": "float32",
                "shape": [54, 6_656],
                "elements": 54 * 6_656,
                "bytes": 54 * 6_656 * size_of::<f32>(),
                "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
                "finite": true,
                "labels": expected_hidden_trace_labels(),
                "phase": "stateless_prompt",
                "token_position": 66,
            },
            "cached_prefix_logits": {
                "relative_path": REFERENCE_CACHED_LOGITS_FILENAME,
                "encoding": "ieee754-f32-little-endian",
                "source_dtype": "float16",
                "storage_dtype": "float32",
                "shape": [32, 202_048],
                "elements": 32 * 202_048,
                "bytes": 32 * 202_048 * size_of::<f32>(),
                "sha256": "2222222222222222222222222222222222222222222222222222222222222222",
                "finite": true,
                "row_position_ids": (66..98).collect::<Vec<_>>(),
                "row_top1_token_ids": token_ids.clone(),
            },
            "teacher_prefix_logits": {
                "relative_path": REFERENCE_TEACHER_LOGITS_FILENAME,
                "encoding": "ieee754-f32-little-endian",
                "source_dtype": "float16",
                "storage_dtype": "float32",
                "shape": [32, 202_048],
                "elements": 32 * 202_048,
                "bytes": 32 * 202_048 * size_of::<f32>(),
                "sha256": "3333333333333333333333333333333333333333333333333333333333333333",
                "finite": true,
                "row_position_ids": (66..98).collect::<Vec<_>>(),
                "row_top1_token_ids": token_ids.clone(),
            },
            "cached_prefill_hidden_trace": {
                "relative_path": REFERENCE_CACHED_PREFILL_HIDDEN_FILENAME,
                "encoding": "ieee754-f32-little-endian",
                "source_dtype": "float16",
                "storage_dtype": "float32",
                "shape": [54, 6_656],
                "elements": 54 * 6_656,
                "bytes": 54 * 6_656 * size_of::<f32>(),
                "sha256": "4444444444444444444444444444444444444444444444444444444444444444",
                "finite": true,
                "labels": expected_hidden_trace_labels(),
                "phase": "cached_prefill",
                "token_position": 66,
            },
            "cached_decode_hidden_trace": {
                "relative_path": REFERENCE_CACHED_DECODE_HIDDEN_FILENAME,
                "encoding": "ieee754-f32-little-endian",
                "source_dtype": "float16",
                "storage_dtype": "float32",
                "shape": [54, 6_656],
                "elements": 54 * 6_656,
                "bytes": 54 * 6_656 * size_of::<f32>(),
                "sha256": "5555555555555555555555555555555555555555555555555555555555555555",
                "finite": true,
                "labels": expected_hidden_trace_labels(),
                "phase": "cached_decode",
                "token_position": 67,
            },
        },
        "generation": {
            "max_new_tokens": 32,
            "eos_token_ids": [200_001, 200_008],
            "token_ids": token_ids.clone(),
            "cached_prefix_top1": token_ids.clone(),
            "stateless_prefix_top1": token_ids,
            "decisions": decisions,
            "minimum_margin": 1.0,
            "minimum_margin_to_epsilon_ratio": 1024.0,
            "decoded_text": "fixture",
            "prompt_cached_cosine": 1.0,
            "minimum_cached_stateless_prefix_cosine": 1.0,
            "minimum_cached_prefill_stateless_hidden_cosine": 1.0,
        },
        "timings_seconds": {
            "load": 0.0,
            "stateless_prompt": 0.0,
            "cached_generation": 0.0,
            "teacher_forced_stateless": 0.0,
        },
    })
}

/// Proves schema, digest, strictness, and tamper detection without model files.
#[test]
fn muse_official_reference_schema_is_weight_independent_and_strict() {
    let expected_artifacts = expected_reference_artifacts();
    assert_eq!(expected_artifacts.len(), CONTROL_FILES.len() + SHARDS.len());
    assert_eq!(
        reference_artifact_digest(&expected_artifacts),
        REFERENCE_ARTIFACT_MANIFEST_SHA256
    );
    assert_eq!(expected_hidden_trace_labels().len(), 54);
    assert_eq!(
        reference_distribution_digest(&expected_reference_distributions()),
        REFERENCE_DISTRIBUTION_MANIFEST_SHA256
    );

    let schema = synthetic_reference_json();
    let reference: MuseOfficialReference =
        serde_json::from_value(schema.clone()).expect("synthetic strict schema must deserialize");
    validate_reference_schema(&reference).expect("synthetic strict schema must validate");

    let mut unknown = schema.clone();
    unknown
        .as_object_mut()
        .expect("root object")
        .insert("unreviewed".to_owned(), serde_json::json!(true));
    assert!(
        serde_json::from_value::<MuseOfficialReference>(unknown).is_err(),
        "unknown fields must fail closed"
    );

    let mut tampered = schema.clone();
    tampered["outputs"]["cached_prefix_logits"]["bytes"] = serde_json::json!(1);
    let reference: MuseOfficialReference =
        serde_json::from_value(tampered).expect("structurally valid size tamper must deserialize");
    assert!(
        validate_reference_schema(&reference)
            .expect_err("unsafe payload size must fail")
            .contains("cached-prefix logit descriptor")
    );

    let mut tampered = schema.clone();
    tampered["execution"]["cache"]["decision_position_ids"][1] = serde_json::json!(999);
    let reference: MuseOfficialReference = serde_json::from_value(tampered)
        .expect("structurally valid position tamper must deserialize");
    assert!(
        validate_reference_schema(&reference)
            .expect_err("position tamper must fail")
            .contains("cache geometry")
    );

    let mut tampered = schema.clone();
    tampered["outputs"]["cached_decode_hidden_trace"]["source_dtype"] =
        serde_json::json!("float32");
    let reference: MuseOfficialReference =
        serde_json::from_value(tampered).expect("structurally valid dtype tamper must deserialize");
    assert!(
        validate_reference_schema(&reference)
            .expect_err("dtype tamper must fail")
            .contains("cached-decode hidden-trace")
    );

    let mut tampered = schema;
    tampered["generation"]["cached_prefix_top1"] = serde_json::json!([7]);
    let reference: MuseOfficialReference =
        serde_json::from_value(tampered).expect("structurally valid tamper must deserialize");
    assert!(
        validate_reference_schema(&reference)
            .expect_err("semantic tamper must fail")
            .contains("generation topology"),
        "semantic tampering must produce the expected failure class"
    );
}

/// Resolves the opt-in external parent containing the fixed reference bundle.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn pinned_reference_root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("TRANSFORMERS_MUSE_GLIMMER_REFERENCE_DIR").expect(
            "set TRANSFORMERS_MUSE_GLIMMER_REFERENCE_DIR to the external parent passed to \
             scripts/generate_muse_glimmer_reference.py",
        ),
    )
    .join(REFERENCE_BUNDLE_DIRECTORY)
}

/// Opens an exact-size reference file without following its final component.
#[cfg(all(feature = "metal", feature = "tokenizers", target_os = "macos"))]
fn open_reference_file_no_follow(path: &Path, expected_bytes: u64, maximum_bytes: u64) -> File {
    /// Darwin open flag rejecting a symlink in the final path component.
    const DARWIN_O_NOFOLLOW: i32 = 0x0000_0100;
    /// Darwin nonblocking open flag avoiding a hang on a substituted special file.
    const DARWIN_O_NONBLOCK: i32 = 0x0000_0004;

    assert!(
        expected_bytes <= maximum_bytes,
        "expected {expected_bytes} bytes exceeds the {maximum_bytes}-byte cap for {}",
        path.display()
    );
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(DARWIN_O_NOFOLLOW | DARWIN_O_NONBLOCK)
        .open(path)
        .unwrap_or_else(|error| {
            panic!(
                "open reference file {} without following links: {error}",
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
    assert_eq!(
        metadata.len(),
        expected_bytes,
        "unexpected byte length for {}",
        path.display()
    );
    file
}

/// Returns descriptor fields that expose replacement or mutation during a read.
#[cfg(all(feature = "metal", feature = "tokenizers", target_os = "macos"))]
fn reference_file_identity(metadata: &fs::Metadata) -> (u64, u64, u64, i64, i64, i64, i64) {
    (
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.mtime(),
        metadata.mtime_nsec(),
        metadata.ctime(),
        metadata.ctime_nsec(),
    )
}

/// Keeps unsupported host compilation explicit for the macOS-only Metal gate.
#[cfg(all(feature = "metal", feature = "tokenizers", not(target_os = "macos")))]
fn open_reference_file_no_follow(_path: &Path, _expected_bytes: u64, _maximum_bytes: u64) -> File {
    panic!("the Muse-Glimmer Metal reference gate requires macOS O_NOFOLLOW semantics")
}

/// Inspects one small record while retaining a hard allocation ceiling.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn bounded_reference_file_length(path: &Path, maximum_bytes: u64) -> u64 {
    let metadata = fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("inspect bounded file {}: {error}", path.display()));
    assert!(
        metadata.file_type().is_file(),
        "{} must be a non-symlink regular file",
        path.display()
    );
    assert!(
        metadata.len() <= maximum_bytes,
        "{} is {} bytes, exceeding the {maximum_bytes}-byte cap",
        path.display(),
        metadata.len()
    );
    metadata.len()
}

/// Reads one bounded regular file and rejects concurrent length changes.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn read_reference_file_exact(path: &Path, expected_bytes: u64, maximum_bytes: u64) -> Vec<u8> {
    let mut file = open_reference_file_no_follow(path, expected_bytes, maximum_bytes);
    #[cfg(target_os = "macos")]
    let before = file
        .metadata()
        .unwrap_or_else(|error| panic!("inspect opened file {}: {error}", path.display()));
    let capacity = usize::try_from(expected_bytes)
        .expect("bounded Muse reference length must fit the host address space");
    let mut payload = Vec::with_capacity(capacity);
    (&mut file)
        .take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut payload)
        .unwrap_or_else(|error| panic!("read reference file {}: {error}", path.display()));
    assert_eq!(
        payload.len(),
        capacity,
        "{} changed length while being read",
        path.display()
    );
    #[cfg(target_os = "macos")]
    {
        let after = file
            .metadata()
            .unwrap_or_else(|error| panic!("reinspect opened file {}: {error}", path.display()));
        assert_eq!(
            reference_file_identity(&before),
            reference_file_identity(&after),
            "{} changed identity or metadata while being read",
            path.display()
        );
    }
    payload
}

/// Fails closed until an independent reviewer pins one generated output hash.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn assert_trusted_reference_digest(label: &str, observed: &str, trusted: Option<&str>) {
    let trusted = trusted.unwrap_or_else(|| {
        panic!(
            "trusted {label} SHA-256 has not been pinned; independently review the generated \
             bundle, then set its narrowly scoped constant in tests/muse_glimmer_artifacts.rs \
             (observed {observed})"
        )
    });
    assert!(is_sha256(trusted), "trusted {label} SHA-256 is malformed");
    assert_eq!(observed, trusted, "generated {label} identity changed");
}

/// Decodes a complete little-endian binary32 payload without native casts.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn decode_f32le(label: &str, payload: &[u8]) -> Vec<f32> {
    assert_eq!(
        payload.len() % size_of::<f32>(),
        0,
        "{label} payload is not aligned to F32 elements"
    );
    payload
        .as_chunks::<{ size_of::<f32>() }>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

/// Computes deterministic descending token ranks with low-ID tie breaking.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn ranked_token_ids(values: &[f32], count: usize) -> Vec<usize> {
    assert!(count > 0 && count <= values.len());
    let mut ids = (0..values.len()).collect::<Vec<_>>();
    ids.sort_unstable_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    ids.truncate(count);
    ids
}

/// Selects one maximum in linear time while preserving low-ID tie breaking.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn top1_token_id(values: &[f32]) -> usize {
    assert!(!values.is_empty());
    assert!(values.iter().all(|value| value.is_finite()));
    let mut best = 0;
    for candidate in 1..values.len() {
        if values[candidate].total_cmp(&values[best]).is_gt() {
            best = candidate;
        }
    }
    best
}

/// Checks a recomputed floating summary with a conservative summation tolerance.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn assert_reference_close(label: &str, actual: f64, expected: f64) {
    let tolerance = expected.abs().max(1.0) * 1.0e-6;
    assert!(
        actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= tolerance,
        "{label} mismatch: expected {expected}, found {actual}, tolerance {tolerance}"
    );
}

/// Fully authenticated numerical tensors from one strict external bundle.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
struct LoadedMuseReference {
    /// Validated manifest describing every authenticated numerical payload.
    reference: MuseOfficialReference,
    /// Decoded complete-vocabulary prompt-final logit vector.
    prompt_logits: Vec<f32>,
    /// Decoded stateless prompt-final hidden trace in row-major order.
    stateless_prompt_hidden: Vec<f32>,
    /// Decoded cached decision-by-vocabulary logit matrix in row-major order.
    cached_prefix_logits: Vec<f32>,
    /// Decoded teacher-forced decision-by-vocabulary logit matrix in row-major order.
    teacher_prefix_logits: Vec<f32>,
    /// Decoded cached-prefill hidden trace in row-major order.
    cached_prefill_hidden: Vec<f32>,
    /// Decoded first recurrent-decode hidden trace in row-major order.
    cached_decode_hidden: Vec<f32>,
}

/// Reads, bounds, hashes, and decodes one exact F32LE tensor payload.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn read_reference_tensor(
    root: &Path,
    label: &str,
    relative_path: &str,
    expected_bytes: usize,
    expected_elements: usize,
    descriptor_sha256: &str,
    trusted_sha256: Option<&str>,
) -> Vec<f32> {
    let path = root.join(relative_path);
    let expected_bytes = u64::try_from(expected_bytes).expect("reference byte count fits u64");
    assert_eq!(
        bounded_reference_file_length(&path, expected_bytes),
        expected_bytes,
        "{label} length differs from its fixed schema bound"
    );
    let raw = read_reference_file_exact(&path, expected_bytes, expected_bytes);
    let digest = sha256_hex(&raw);
    assert_trusted_reference_digest(label, &digest, trusted_sha256);
    assert_eq!(
        digest, descriptor_sha256,
        "trusted {label} identity differs from the JSON descriptor"
    );
    let values = decode_f32le(label, &raw);
    assert_eq!(
        values.len(),
        expected_elements,
        "{label} element count changed"
    );
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} contains a non-finite value"
    );
    values
}

/// Reads and independently authenticates the complete external oracle bundle.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn read_pinned_muse_reference() -> LoadedMuseReference {
    /// Maximum accepted reference-manifest length in bytes before allocating a read buffer.
    const MAXIMUM_JSON_BYTES: u64 = 1024 * 1024;

    let supplied_root = pinned_reference_root();
    let metadata = fs::symlink_metadata(&supplied_root).unwrap_or_else(|error| {
        panic!(
            "inspect Muse reference bundle {}: {error}",
            supplied_root.display()
        )
    });
    assert!(
        metadata.file_type().is_dir(),
        "Muse reference bundle must be a non-symlink directory"
    );
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert!(
        !root.starts_with(env!("CARGO_MANIFEST_DIR")),
        "generated numerical references must remain outside the crate"
    );
    let expected_entries = BTreeSet::from([
        REFERENCE_JSON_FILENAME.to_owned(),
        REFERENCE_LOGITS_FILENAME.to_owned(),
        REFERENCE_HIDDEN_FILENAME.to_owned(),
        REFERENCE_CACHED_LOGITS_FILENAME.to_owned(),
        REFERENCE_TEACHER_LOGITS_FILENAME.to_owned(),
        REFERENCE_CACHED_PREFILL_HIDDEN_FILENAME.to_owned(),
        REFERENCE_CACHED_DECODE_HIDDEN_FILENAME.to_owned(),
    ]);
    let observed_entries = fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("enumerate reference bundle {}: {error}", root.display()))
        .map(|entry| {
            let entry = entry.unwrap_or_else(|error| {
                panic!(
                    "inspect reference bundle entry in {}: {error}",
                    root.display()
                )
            });
            let file_type = entry.file_type().unwrap_or_else(|error| {
                panic!(
                    "inspect reference entry {}: {error}",
                    entry.path().display()
                )
            });
            assert!(
                file_type.is_file() && !file_type.is_symlink(),
                "reference bundle entry {} must be a regular non-symlink file",
                entry.path().display()
            );
            entry
                .file_name()
                .into_string()
                .unwrap_or_else(|_| panic!("reference bundle contains a non-UTF-8 name"))
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        observed_entries, expected_entries,
        "reference bundle entry set changed"
    );

    let json_path = root.join(REFERENCE_JSON_FILENAME);
    let json_length = bounded_reference_file_length(&json_path, MAXIMUM_JSON_BYTES);
    let json = read_reference_file_exact(&json_path, json_length, MAXIMUM_JSON_BYTES);
    let json_digest = sha256_hex(&json);
    assert_trusted_reference_digest(
        "Muse reference JSON",
        &json_digest,
        TRUSTED_MUSE_REFERENCE_JSON_SHA256,
    );
    let reference: MuseOfficialReference = serde_json::from_slice(&json)
        .unwrap_or_else(|error| panic!("parse strict reference {}: {error}", json_path.display()));
    validate_reference_schema(&reference)
        .unwrap_or_else(|error| panic!("validate strict Muse reference: {error}"));

    let logits = read_reference_tensor(
        &root,
        "Muse reference prompt logits",
        &reference.outputs.prompt_logits.relative_path,
        reference.outputs.prompt_logits.bytes,
        reference.outputs.prompt_logits.elements,
        &reference.outputs.prompt_logits.sha256,
        TRUSTED_MUSE_REFERENCE_LOGITS_SHA256,
    );
    let ranked = ranked_token_ids(&logits, 10);
    for (record, token_id) in reference
        .outputs
        .prompt_logits
        .top10
        .iter()
        .zip(ranked.iter().copied())
    {
        assert_eq!(usize::try_from(record.token_id).unwrap(), token_id);
        assert_eq!(record.logit as f32, logits[token_id]);
    }
    assert_eq!(
        usize::try_from(reference.outputs.prompt_logits.top1_token_id).unwrap(),
        ranked[0]
    );
    let logits_sum = logits.iter().map(|value| f64::from(*value)).sum::<f64>();
    let logits_l2 = logits
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    assert_reference_close(
        "Muse prompt-logit sum",
        logits_sum,
        reference.outputs.prompt_logits.sum,
    );
    assert_reference_close(
        "Muse prompt-logit L2 norm",
        logits_l2,
        reference.outputs.prompt_logits.l2_norm,
    );

    let hidden = read_reference_tensor(
        &root,
        "Muse reference hidden trace",
        &reference.outputs.hidden_last_trace.relative_path,
        reference.outputs.hidden_last_trace.bytes,
        reference.outputs.hidden_last_trace.elements,
        &reference.outputs.hidden_last_trace.sha256,
        TRUSTED_MUSE_REFERENCE_HIDDEN_SHA256,
    );
    let cached_prefix_logits = read_reference_tensor(
        &root,
        "Muse reference cached-prefix logits",
        &reference.outputs.cached_prefix_logits.relative_path,
        reference.outputs.cached_prefix_logits.bytes,
        reference.outputs.cached_prefix_logits.elements,
        &reference.outputs.cached_prefix_logits.sha256,
        TRUSTED_MUSE_REFERENCE_CACHED_LOGITS_SHA256,
    );
    let teacher_prefix_logits = read_reference_tensor(
        &root,
        "Muse reference teacher-prefix logits",
        &reference.outputs.teacher_prefix_logits.relative_path,
        reference.outputs.teacher_prefix_logits.bytes,
        reference.outputs.teacher_prefix_logits.elements,
        &reference.outputs.teacher_prefix_logits.sha256,
        TRUSTED_MUSE_REFERENCE_TEACHER_LOGITS_SHA256,
    );
    let cached_prefill_hidden = read_reference_tensor(
        &root,
        "Muse reference cached-prefill hidden trace",
        &reference.outputs.cached_prefill_hidden_trace.relative_path,
        reference.outputs.cached_prefill_hidden_trace.bytes,
        reference.outputs.cached_prefill_hidden_trace.elements,
        &reference.outputs.cached_prefill_hidden_trace.sha256,
        TRUSTED_MUSE_REFERENCE_CACHED_PREFILL_HIDDEN_SHA256,
    );
    let cached_decode_hidden = read_reference_tensor(
        &root,
        "Muse reference cached-decode hidden trace",
        &reference.outputs.cached_decode_hidden_trace.relative_path,
        reference.outputs.cached_decode_hidden_trace.bytes,
        reference.outputs.cached_decode_hidden_trace.elements,
        &reference.outputs.cached_decode_hidden_trace.sha256,
        TRUSTED_MUSE_REFERENCE_CACHED_DECODE_HIDDEN_SHA256,
    );
    let prompt_cached_cosine = cosine_similarity(&logits, &cached_prefix_logits[..202_048]);
    assert!(
        prompt_cached_cosine >= 0.999,
        "official stateless/cached prompt cosine was {prompt_cached_cosine}"
    );
    for (row, (stateless, cached)) in hidden
        .as_chunks::<6_656>()
        .0
        .iter()
        .zip(cached_prefill_hidden.as_chunks::<6_656>().0.iter())
        .enumerate()
    {
        let cosine = cosine_similarity(stateless, cached);
        assert!(
            cosine >= 0.999,
            "official stateless/cached-prefill hidden cosine at row {row} was {cosine}"
        );
    }
    for (step, (cached_row, teacher_row)) in cached_prefix_logits
        .as_chunks::<202_048>()
        .0
        .iter()
        .zip(teacher_prefix_logits.as_chunks::<202_048>().0.iter())
        .enumerate()
    {
        assert_eq!(
            top1_token_id(cached_row),
            usize::try_from(REFERENCE_GENERATED_TOKEN_IDS[step]).unwrap(),
            "official cached-prefix top-1 changed at step {step}"
        );
        assert_eq!(
            top1_token_id(teacher_row),
            usize::try_from(REFERENCE_GENERATED_TOKEN_IDS[step]).unwrap(),
            "official teacher-prefix top-1 changed at step {step}"
        );
        let cosine = cosine_similarity(cached_row, teacher_row);
        assert!(
            cosine >= 0.999,
            "official cached/teacher cosine at step {step} was {cosine}"
        );
    }
    LoadedMuseReference {
        reference,
        prompt_logits: logits,
        stateless_prompt_hidden: hidden,
        cached_prefix_logits,
        teacher_prefix_logits,
        cached_prefill_hidden,
        cached_decode_hidden,
    }
}

/// Materializes one Metal tensor as synchronized F32 acceptance values.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn metal_values<const D: usize>(label: &str, tensor: &Tensor<Metal, D>) -> Vec<f32> {
    tensor
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .unwrap_or_else(|error| panic!("materialize {label}: {error}"))
}

/// Computes finite cosine similarity with F64 accumulation.
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

/// Creates the explicit all-valid token mask promised by the oracle schema.
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

/// Forces adapter discovery and one synchronized storage round trip.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn assert_metal_backend_available(device: &<Metal as burn::tensor::backend::BackendTypes>::Device) {
    let probe = Tensor::<Metal, 1>::zeros([1], device)
        .to_data()
        .to_vec::<f32>()
        .expect("the Metal acceptance backend must expose F32 storage");
    assert_eq!(probe, [0.0]);
}

/// Borrows the concrete Muse state selected by automatic model dispatch.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn muse_state(state: &AutoCausalDecodeState<Metal>) -> &MuseGlimmerDecodeState<Metal> {
    match state {
        AutoCausalDecodeState::MuseGlimmer(state) => state,
        _ => panic!("automatic decode state must be Muse-Glimmer"),
    }
}

/// Proves the complete Burn cache allocation, schedule, dtype, and retention state.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn assert_muse_cache_geometry(
    state: &AutoCausalDecodeState<Metal>,
    expected_length: usize,
    expected_capacity: usize,
) {
    assert_eq!(state.capacity(), expected_capacity);
    assert_eq!(state.lengths(), &[expected_length]);
    let state = muse_state(state);
    assert_eq!(state.sliding_window(), Some(expected_capacity));
    let tensors = state.tensor_state();
    assert_eq!(tensors.main().native_kv_count(), 2);
    assert_eq!(tensors.main().recurrent_count(), 0);
    let sliding = tensors.main().native_kv(0).expect("Muse sliding KV group");
    let full = tensors.main().native_kv(1).expect("Muse full KV group");
    let sliding_spec = sliding.spec();
    assert_eq!(sliding_spec.layers(), 39);
    assert_eq!(sliding_spec.key_value_heads(), 2);
    assert_eq!(sliding_spec.head_dimension(), 128);
    assert_eq!(sliding_spec.storage(), KvCacheStorage::MirroredSlidingRing);
    assert_eq!(sliding_spec.dtype(), DType::F16);
    assert_eq!(sliding_spec.staging_capacity(), 0);
    assert_eq!(sliding.committed_time_slots(), expected_capacity * 2);
    let full_spec = full.spec();
    assert_eq!(full_spec.layers(), 13);
    assert_eq!(full_spec.key_value_heads(), 2);
    assert_eq!(full_spec.head_dimension(), 128);
    assert_eq!(full_spec.storage(), KvCacheStorage::Full);
    assert_eq!(full_spec.dtype(), DType::F16);
    assert_eq!(full_spec.staging_capacity(), 0);
    assert_eq!(full.committed_time_slots(), expected_capacity);

    let mut sliding_layer = 0;
    let mut full_layer = 0;
    for model_layer in 0..52 {
        let expected = if model_layer % 4 == 3 {
            let slot = MuseGlimmerLayerCache::Full {
                group: 1,
                layer: full_layer,
            };
            full_layer += 1;
            slot
        } else {
            let slot = MuseGlimmerLayerCache::Sliding {
                group: 0,
                layer: sliding_layer,
            };
            sliding_layer += 1;
            slot
        };
        assert_eq!(state.layer_cache(model_layer), Some(expected));
    }
    assert_eq!(sliding_layer, 39);
    assert_eq!(full_layer, 13);

    for (label, cache, layers, time_slots) in [
        ("sliding", sliding, 39, expected_capacity * 2),
        ("full", full, 13, expected_capacity),
    ] {
        for layer in 0..layers {
            let keys = cache.keys(layer).expect("Muse cache keys");
            let values = cache.values(layer).expect("Muse cache values");
            assert_eq!(keys.dims(), [1, 2, time_slots, 128], "{label} keys");
            assert_eq!(values.dims(), [1, 2, time_slots, 128], "{label} values");
            assert_eq!(keys.dtype(), DType::F16, "{label} keys dtype");
            assert_eq!(values.dtype(), DType::F16, "{label} values dtype");
            let retained = cache
                .read_sequence(layer, 0, expected_length)
                .expect("read Muse cache retention");
            assert_eq!(retained.retained_positions(), 0..expected_length);
            assert_eq!(retained.segments().len(), usize::from(expected_length > 0));
            if let Some(segment) = retained.segments().first() {
                assert_eq!(segment.absolute_positions(), 0..expected_length);
                assert_eq!(segment.keys().dims(), [1, 2, expected_length, 128]);
                assert_eq!(segment.values().dims(), [1, 2, expected_length, 128]);
                assert_eq!(segment.keys().dtype(), DType::F16);
                assert_eq!(segment.values().dtype(), DType::F16);
            }
        }
    }
}

/// Captures the clean Rust source, exact trees, toolchain, target, and feature set.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
#[derive(Debug, serde::Serialize)]
struct MuseRustAcceptanceProvenance {
    /// Repository HEAD used for the Rust acceptance execution.
    git_head: String,
    /// Root Git tree of the repository containing the Rust crate.
    git_tree: String,
    /// Exact committed crate tree, including support for a historically nested checkout.
    transformers_tree: String,
    /// Whether the crate subtree has source changes; acceptance requires false.
    transformers_tree_dirty: bool,
    /// SHA-256 of the test source verified against its compiled-in bytes.
    test_source_sha256: String,
    /// SHA-256 of the quantized-quality helper verified against its compiled-in bytes.
    quantized_quality_source_sha256: String,
    /// Minimum Rust version declared by the compiled Cargo package.
    cargo_package_rust_version: String,
    /// Verbose compiler identity returned by `rustc -Vv`.
    rustc_verbose: String,
    /// Build profile recorded for this acceptance executable.
    profile: String,
    /// Target architecture and operating system of the acceptance executable.
    target: String,
    /// Enabled crate features recorded for the acceptance executable.
    features: Vec<String>,
}

/// Runs one bounded source-control or toolchain query and returns trimmed UTF-8.
fn checked_command_output(program: &str, arguments: &[&str], directory: &Path) -> String {
    let output = Command::new(program)
        .args(arguments)
        .current_dir(directory)
        .output()
        .unwrap_or_else(|error| panic!("run {program} {arguments:?}: {error}"));
    assert!(
        output.status.success(),
        "{program} {arguments:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("provenance command output must be UTF-8")
        .trim()
        .to_owned()
}

/// Resolves the committed crate tree in either a standalone or nested Git checkout.
///
/// Canonical paths identify the actual worktree rather than assuming a fixed
/// number of parent directories. A literal crate-relative pathspec excludes
/// unrelated parent-repository changes without letting pathspec metacharacters
/// broaden or suppress the clean-source check. The root crate uses the commit's
/// root tree; a nested crate uses its exact subtree, preserving the historical
/// meaning of `transformers_tree` without rewriting old evidence.
fn clean_git_crate_tree(crate_root: &Path) -> (PathBuf, String) {
    let crate_root = fs::canonicalize(crate_root).expect("resolve acceptance crate directory");
    let repository = checked_command_output("git", &["rev-parse", "--show-toplevel"], &crate_root);
    let repository = fs::canonicalize(repository).expect("resolve acceptance Git worktree");
    let relative = crate_root
        .strip_prefix(&repository)
        .expect("acceptance crate must be inside its Git worktree");
    let relative = relative
        .components()
        .map(|component| {
            component
                .as_os_str()
                .to_str()
                .expect("acceptance crate path must be UTF-8")
        })
        .collect::<Vec<_>>()
        .join("/");
    let pathspec = if relative.is_empty() { "." } else { &relative };
    let status = checked_command_output(
        "git",
        &[
            "--literal-pathspecs",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            pathspec,
        ],
        &repository,
    );
    assert!(
        status.is_empty(),
        "Muse acceptance requires a committed, clean crate subtree {pathspec}: {status}"
    );
    let tree_revision = if relative.is_empty() {
        "HEAD^{tree}".to_owned()
    } else {
        format!("HEAD:{relative}")
    };
    let tree = checked_command_output("git", &["rev-parse", &tree_revision], &repository);
    (repository, tree)
}

/// Creates a tiny committed checkout without reading or modifying the caller's Git index.
fn git_crate_fixture(relative: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let temporary = tempfile::tempdir().unwrap();
    let repository = temporary.path().join("repository with spaces");
    let crate_root = repository.join(relative);
    fs::create_dir_all(&crate_root).unwrap();
    fs::write(crate_root.join("Cargo.toml"), "fixture\n").unwrap();
    if !relative.is_empty() {
        fs::write(repository.join("unrelated.txt"), "outside crate\n").unwrap();
    }
    checked_command_output("git", &["init", "--quiet", "--template="], &repository);
    checked_command_output("git", &["add", "--", "."], &repository);
    checked_command_output(
        "git",
        &[
            "-c",
            "user.name=Transformer fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        &repository,
    );
    (temporary, repository, crate_root)
}

/// Requires the source-dirtiness diagnostic rather than accepting an unrelated Git failure.
fn assert_dirty_git_crate_rejected(crate_root: &Path) {
    let error = std::panic::catch_unwind(|| clean_git_crate_tree(crate_root)).unwrap_err();
    let message = error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .expect("source rejection must have a textual diagnostic");
    assert!(message.contains("requires a committed, clean crate subtree"));
}

/// Root, former nested and literal-metacharacter layouts bind the exact tree and reject source edits.
#[test]
fn git_crate_provenance_supports_standalone_and_nested_layouts() {
    for relative in ["", "crates/transformers", "nested [fixture]/transformers"] {
        let (_temporary, repository, crate_root) = git_crate_fixture(relative);
        let revision = if relative.is_empty() {
            "HEAD^{tree}".to_owned()
        } else {
            format!("HEAD:{relative}")
        };
        let expected_tree = checked_command_output("git", &["rev-parse", &revision], &repository);
        let (actual_repository, actual_tree) = clean_git_crate_tree(&crate_root);
        assert_eq!(actual_repository, fs::canonicalize(&repository).unwrap());
        assert_eq!(actual_tree, expected_tree);
        if !relative.is_empty() {
            assert_ne!(
                actual_tree,
                checked_command_output("git", &["rev-parse", "HEAD^{tree}"], &repository)
            );
            fs::write(repository.join("unrelated.txt"), "unrelated user edit\n").unwrap();
            assert_eq!(clean_git_crate_tree(&crate_root).1, expected_tree);
        }
        fs::write(crate_root.join("Cargo.toml"), "modified\n").unwrap();
        assert_dirty_git_crate_rejected(&crate_root);
        checked_command_output("git", &["add", "--", "."], &crate_root);
        assert_dirty_git_crate_rejected(&crate_root);
        fs::write(crate_root.join("Cargo.toml"), "fixture\n").unwrap();
        checked_command_output("git", &["add", "--", "."], &crate_root);
        assert_eq!(clean_git_crate_tree(&crate_root).1, expected_tree);
        fs::write(crate_root.join("untracked.rs"), "untracked source\n").unwrap();
        assert_dirty_git_crate_rejected(&crate_root);
        fs::remove_file(crate_root.join("untracked.rs")).unwrap();
        assert_eq!(clean_git_crate_tree(&crate_root).1, expected_tree);
    }
}

/// A directory outside Git or an unborn repository cannot supply committed source provenance.
#[test]
fn git_crate_provenance_requires_a_committed_repository() {
    let temporary = tempfile::tempdir().unwrap();
    assert!(std::panic::catch_unwind(|| clean_git_crate_tree(temporary.path())).is_err());
    checked_command_output("git", &["init", "--quiet", "--template="], temporary.path());
    assert!(std::panic::catch_unwind(|| clean_git_crate_tree(temporary.path())).is_err());
}

/// Requires a clean transformer subtree and builds the reproducibility record.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn rust_acceptance_provenance() -> MuseRustAcceptanceProvenance {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (repository, transformers_tree) = clean_git_crate_tree(crate_root);
    let repository = repository.as_path();
    let features = [
        ("std", cfg!(feature = "std")),
        ("metal", cfg!(feature = "metal")),
        ("tokenizers", cfg!(feature = "tokenizers")),
        ("autodiff", cfg!(feature = "autodiff")),
        ("cpu", cfg!(feature = "cpu")),
        ("flex", cfg!(feature = "flex")),
        ("cuda", cfg!(feature = "cuda")),
        ("rocm", cfg!(feature = "rocm")),
        ("chat-templates", cfg!(feature = "chat-templates")),
        ("kernel-tracing", cfg!(feature = "kernel-tracing")),
        (
            "attention-kernel-tracing",
            cfg!(feature = "attention-kernel-tracing"),
        ),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name.to_owned()))
    .collect::<Vec<_>>();
    assert_eq!(features, ["std", "metal", "tokenizers"]);
    let compiled_test_source = include_bytes!("muse_glimmer_artifacts.rs");
    let test_source_path = crate_root.join("tests/muse_glimmer_artifacts.rs");
    let test_source_metadata = fs::symlink_metadata(&test_source_path)
        .unwrap_or_else(|error| panic!("inspect {}: {error}", test_source_path.display()));
    assert!(
        test_source_metadata.is_file()
            && !test_source_metadata.file_type().is_symlink()
            && test_source_metadata.len() <= 1024 * 1024,
        "acceptance test source must be a bounded regular non-symlink file"
    );
    let current_test_source = fs::read(&test_source_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", test_source_path.display()));
    assert_eq!(
        sha256_hex(&current_test_source),
        sha256_hex(compiled_test_source),
        "running Muse acceptance binary was compiled from stale test source"
    );
    let compiled_quantized_quality_source = include_bytes!("support/quantized_quality.rs");
    let quantized_quality_source_path = crate_root.join("tests/support/quantized_quality.rs");
    let quantized_quality_source_metadata = fs::symlink_metadata(&quantized_quality_source_path)
        .unwrap_or_else(|error| {
            panic!(
                "inspect {}: {error}",
                quantized_quality_source_path.display()
            )
        });
    assert!(
        quantized_quality_source_metadata.is_file()
            && !quantized_quality_source_metadata.file_type().is_symlink()
            && quantized_quality_source_metadata.len() <= 1024 * 1024,
        "quantized-quality source must be a bounded regular non-symlink file"
    );
    let current_quantized_quality_source =
        fs::read(&quantized_quality_source_path).unwrap_or_else(|error| {
            panic!("read {}: {error}", quantized_quality_source_path.display())
        });
    assert_eq!(
        sha256_hex(&current_quantized_quality_source),
        sha256_hex(compiled_quantized_quality_source),
        "running Muse acceptance binary was compiled from stale quantized-quality source"
    );
    MuseRustAcceptanceProvenance {
        git_head: checked_command_output("git", &["rev-parse", "HEAD"], repository),
        git_tree: checked_command_output("git", &["rev-parse", "HEAD^{tree}"], repository),
        transformers_tree,
        transformers_tree_dirty: false,
        test_source_sha256: sha256_hex(compiled_test_source),
        quantized_quality_source_sha256: sha256_hex(compiled_quantized_quality_source),
        cargo_package_rust_version: env!("CARGO_PKG_RUST_VERSION").to_owned(),
        rustc_verbose: checked_command_output("rustc", &["-Vv"], repository),
        profile: if cfg!(debug_assertions) {
            "debug".to_owned()
        } else {
            "release".to_owned()
        },
        target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        features,
    }
}

/// Materializes exactly one last-token row from every F16 hidden diagnostic.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn metal_hidden_trace(
    label: &str,
    hidden_states: &[Tensor<Metal, 3>],
    sequence: usize,
) -> Vec<f32> {
    assert_eq!(hidden_states.len(), 54, "{label} diagnostic row count");
    for (row, hidden) in hidden_states.iter().enumerate() {
        assert_eq!(hidden.dims(), [1, sequence, 6_656], "{label} row {row}");
        assert_eq!(hidden.dtype(), DType::F16, "{label} row {row} dtype");
    }
    let trace = Tensor::cat(
        hidden_states
            .iter()
            .map(|hidden| {
                hidden
                    .clone()
                    .slice([0..1, sequence - 1..sequence, 0..6_656])
            })
            .collect(),
        0,
    );
    assert_eq!(trace.dtype(), DType::F16);
    metal_values(label, &trace)
}

/// Requires row-wise official/Burn hidden parity and returns the minimum cosine.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
fn assert_hidden_trace_parity(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    labels: &[String],
) -> f64 {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(labels.len(), 54);
    let mut minimum = f64::INFINITY;
    for (row, ((actual, expected), semantic_label)) in actual
        .as_chunks::<6_656>()
        .0
        .iter()
        .zip(expected.as_chunks::<6_656>().0.iter())
        .zip(labels)
        .enumerate()
    {
        let cosine = cosine_similarity(actual, expected);
        assert!(
            cosine >= 0.999,
            "{label} cosine for row {row} ({semantic_label}) was {cosine}"
        );
        minimum = minimum.min(cosine);
    }
    minimum
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

/// Keeps both immutable Muse artifact anchors aligned with their support record.
#[cfg(feature = "tokenizers")]
#[test]
fn committed_muse_burnpack_anchors_are_complete_and_non_claiming() {
    let q8_sha256 = Q8_BURNPACK_SHA256.expect("the completed Q8 conversion must be pinned");
    let q8_artifact_bytes =
        Q8_BURNPACK_ARTIFACT_BYTES.expect("the completed Q8 artifact length must be pinned");
    assert_eq!(q8_sha256.len(), 64);
    assert!(
        q8_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "Q8 identity must be a lowercase SHA-256"
    );
    assert_eq!(q8_artifact_bytes, 28_165_645_337);
    assert_eq!(
        q8_artifact_bytes - Q8_PACKED_PAYLOAD_BYTES,
        302_105,
        "manifest and alignment overhead changed"
    );
    let q4_sha256 = Q4_BURNPACK_SHA256.expect("the completed Q4 conversion must be pinned");
    let q4_artifact_bytes =
        Q4_BURNPACK_ARTIFACT_BYTES.expect("the completed Q4 artifact length must be pinned");
    assert_eq!(q4_sha256.len(), 64);
    assert!(
        q4_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "Q4 identity must be a lowercase SHA-256"
    );
    assert_eq!(q4_artifact_bytes, 14_911_372_485);
    assert_eq!(
        q4_artifact_bytes - Q4_PACKED_PAYLOAD_BYTES,
        308_421,
        "Q4 manifest and alignment overhead changed"
    );

    let record: serde_json::Value = serde_json::from_str(include_str!(
        "../acceptance/muse-glimmer-30b-a4e59da52a7bc87ae7251dd5545c0dd437c44b68.json"
    ))
    .expect("Muse acceptance record must remain valid JSON");
    assert_eq!(record["advertised_supported"].as_bool(), Some(false));
    let quantization = &record["native_quantization_scaffold"];
    assert_eq!(
        quantization["real_q8_sidecar_created"].as_bool(),
        Some(true)
    );
    assert_eq!(
        quantization["real_q4_sidecar_created"].as_bool(),
        Some(true)
    );
    let q8 = &quantization["real_q8_conversion"];
    assert_eq!(
        q8["artifact_filename"].as_str(),
        Some(Q8_BURNPACK_FILE_NAME)
    );
    assert_eq!(
        q8["artifact_path"].as_str(),
        Some(Q8_BURNPACK_OBSERVED_PATH)
    );
    assert_eq!(q8["artifact_bytes"].as_u64(), Some(q8_artifact_bytes));
    assert_eq!(q8["payload_bytes"].as_u64(), Some(Q8_PACKED_PAYLOAD_BYTES));
    assert_eq!(
        q8["largest_payload_bytes"].as_u64(),
        Some(Q8_LARGEST_PACKED_TENSOR_BYTES)
    );
    assert_eq!(q8["sha256"].as_str(), Some(q8_sha256));
    assert_eq!(q8["elapsed_seconds"].as_f64(), Some(1_570.845));
    assert_eq!(q8["format_version"].as_u64(), Some(2));
    assert_eq!(q8["packed_matrices"].as_u64(), Some(417));
    assert_eq!(q8["retained_f16_tensors"].as_u64(), Some(210));
    assert_eq!(q8["text_tensors_applied"].as_u64(), Some(627));
    assert_eq!(q8["renames_applied"].as_u64(), Some(626));
    assert_eq!(q8["bf16_to_f16_conversions"].as_u64(), Some(627));
    assert_eq!(q8["visual_tensors_allowlisted"].as_u64(), Some(809));
    assert_eq!(q8["source_shard_opens"].as_u64(), Some(2));
    assert_eq!(q8["maximum_open_source_shards"].as_u64(), Some(1));
    assert_eq!(q8["source_tensor_reads"].as_u64(), Some(627));
    assert_eq!(
        q8["source_payload_bytes"].as_u64(),
        Some(LANGUAGE_PARAMETER_BYTES)
    );
    for pass in ["provenance_preflight", "provenance_postflight"] {
        assert_eq!(q8[pass]["passes_completed"].as_u64(), Some(1));
        assert_eq!(q8[pass]["shards_hashed"].as_u64(), Some(2));
        assert_eq!(
            q8[pass]["bytes_hashed"].as_u64(),
            Some(CHECKPOINT_ARTIFACT_BYTES)
        );
        assert_eq!(q8[pass]["peak_buffer_bytes"].as_u64(), Some(1_048_576));
    }
    assert_eq!(
        q8["all_payloads_reopened_and_validated"].as_bool(),
        Some(true)
    );
    assert_eq!(q8["packed_metal_executed"].as_bool(), Some(true));
    let execution = &q8["packed_metal_execution"];
    assert_eq!(execution["status"].as_str(), Some("passed"));
    assert_eq!(execution["profile"].as_str(), Some("release"));
    assert_eq!(execution["backend"].as_str(), Some("metal"));
    assert_eq!(
        execution["features"].as_array().map(|features| {
            features
                .iter()
                .map(|feature| feature.as_str().expect("feature must be a string"))
                .collect::<Vec<_>>()
        }),
        Some(vec!["std", "metal", "tokenizers", "kernel-tracing"])
    );
    assert_eq!(
        execution["test"].as_str(),
        Some("smokes_and_audits_pinned_muse_glimmer_q8_burnpack_on_metal")
    );
    assert_eq!(execution["total_test_seconds"].as_f64(), Some(743.75));
    assert_eq!(execution["load_seconds"].as_f64(), Some(595.993));
    assert_eq!(execution["stateless_seconds"].as_f64(), Some(5.134));
    assert_eq!(execution["cached_prefill_seconds"].as_f64(), Some(0.561));
    assert_eq!(execution["recurrent_decode_seconds"].as_f64(), Some(0.247));
    assert_eq!(
        execution["cached_stateless_logit_cosine"].as_f64(),
        Some(1.0)
    );
    assert_eq!(execution["prompt_top1_token_id"].as_u64(), Some(85_068));
    assert_eq!(execution["decode_top1_token_id"].as_u64(), Some(90));
    assert_eq!(
        execution["packed_matmul_launches"].as_u64(),
        Some(QUANTIZED_TENSOR_COUNT as u64)
    );
    assert_eq!(
        execution["expected_packed_matmul_launches"].as_u64(),
        Some(QUANTIZED_TENSOR_COUNT as u64)
    );
    assert_eq!(
        execution["whole_weight_dequantization_launches"].as_u64(),
        Some(0)
    );
    for check in [
        "mixed_parameter_dtypes_validated",
        "sidecar_sha256_revalidated",
        "all_payloads_reopened_and_validated",
        "source_provenance_pre_and_postflight_revalidated",
    ] {
        assert_eq!(execution[check].as_bool(), Some(true), "{check}");
    }
    assert_eq!(execution["source_shard_opens"].as_u64(), Some(2));
    assert_eq!(execution["maximum_open_source_shards"].as_u64(), Some(1));
    assert_eq!(execution["source_tensor_reads"].as_u64(), Some(210));
    assert_eq!(
        execution["source_payload_bytes"].as_u64(),
        Some(RETAINED_F16_TENSOR_BYTES)
    );
    assert_eq!(q8["quality_evaluated"].as_bool(), Some(false));
    let q4 = &quantization["real_q4_conversion"];
    assert_eq!(
        q4["artifact_filename"].as_str(),
        Some(Q4_BURNPACK_FILE_NAME)
    );
    assert_eq!(
        q4["artifact_path"].as_str(),
        Some(Q4_BURNPACK_OBSERVED_PATH)
    );
    assert_eq!(q4["artifact_bytes"].as_u64(), Some(q4_artifact_bytes));
    assert_eq!(q4["payload_bytes"].as_u64(), Some(Q4_PACKED_PAYLOAD_BYTES));
    assert_eq!(
        q4["largest_payload_bytes"].as_u64(),
        Some(Q4_LARGEST_PACKED_TENSOR_BYTES)
    );
    assert_eq!(q4["sha256"].as_str(), Some(q4_sha256));
    assert_eq!(q4["elapsed_seconds"].as_f64(), Some(1_359.922));
    assert_eq!(q4["format_version"].as_u64(), Some(2));
    assert_eq!(q4["packed_matrices"].as_u64(), Some(417));
    assert_eq!(q4["retained_f16_tensors"].as_u64(), Some(210));
    assert_eq!(q4["text_tensors_applied"].as_u64(), Some(627));
    assert_eq!(q4["renames_applied"].as_u64(), Some(626));
    assert_eq!(q4["bf16_to_f16_conversions"].as_u64(), Some(627));
    assert_eq!(q4["visual_tensors_allowlisted"].as_u64(), Some(809));
    assert_eq!(q4["source_shard_opens"].as_u64(), Some(2));
    assert_eq!(q4["maximum_open_source_shards"].as_u64(), Some(1));
    assert_eq!(q4["source_tensor_reads"].as_u64(), Some(627));
    assert_eq!(
        q4["source_payload_bytes"].as_u64(),
        Some(LANGUAGE_PARAMETER_BYTES)
    );
    for pass in ["provenance_preflight", "provenance_postflight"] {
        assert_eq!(q4[pass]["passes_completed"].as_u64(), Some(1));
        assert_eq!(q4[pass]["shards_hashed"].as_u64(), Some(2));
        assert_eq!(
            q4[pass]["bytes_hashed"].as_u64(),
            Some(CHECKPOINT_ARTIFACT_BYTES)
        );
        assert_eq!(q4[pass]["peak_buffer_bytes"].as_u64(), Some(1_048_576));
    }
    assert_eq!(
        q4["all_payloads_reopened_and_validated"].as_bool(),
        Some(true)
    );
    assert_eq!(q4["packed_metal_executed"].as_bool(), Some(true));
    let q4_execution = &q4["packed_metal_execution"];
    assert_eq!(q4_execution["status"].as_str(), Some("passed"));
    assert_eq!(q4_execution["profile"].as_str(), Some("release"));
    assert_eq!(q4_execution["backend"].as_str(), Some("metal"));
    assert_eq!(
        q4_execution["features"].as_array().map(|features| {
            features
                .iter()
                .map(|feature| feature.as_str().expect("feature must be a string"))
                .collect::<Vec<_>>()
        }),
        Some(vec!["std", "metal", "tokenizers", "kernel-tracing"])
    );
    assert_eq!(
        q4_execution["test"].as_str(),
        Some("smokes_and_audits_pinned_muse_glimmer_q4_burnpack_on_metal")
    );
    assert_eq!(q4_execution["total_test_seconds"].as_f64(), Some(543.38));
    assert_eq!(q4_execution["load_seconds"].as_f64(), Some(459.031));
    assert_eq!(q4_execution["stateless_seconds"].as_f64(), Some(2.636));
    assert_eq!(q4_execution["cached_prefill_seconds"].as_f64(), Some(0.189));
    assert_eq!(
        q4_execution["recurrent_decode_seconds"].as_f64(),
        Some(0.248)
    );
    assert_eq!(
        q4_execution["cached_stateless_logit_cosine"].as_f64(),
        Some(1.0)
    );
    assert_eq!(q4_execution["prompt_top1_token_id"].as_u64(), Some(85_068));
    assert_eq!(q4_execution["decode_top1_token_id"].as_u64(), Some(90));
    assert_eq!(
        q4_execution["packed_matmul_launches"].as_u64(),
        Some(QUANTIZED_TENSOR_COUNT as u64)
    );
    assert_eq!(
        q4_execution["expected_packed_matmul_launches"].as_u64(),
        Some(QUANTIZED_TENSOR_COUNT as u64)
    );
    assert_eq!(
        q4_execution["whole_weight_dequantization_launches"].as_u64(),
        Some(0)
    );
    for check in [
        "mixed_parameter_dtypes_validated",
        "sidecar_sha256_revalidated",
        "all_payloads_reopened_and_validated",
        "source_provenance_pre_and_postflight_revalidated",
    ] {
        assert_eq!(q4_execution[check].as_bool(), Some(true), "{check}");
    }
    assert_eq!(q4_execution["source_shard_opens"].as_u64(), Some(2));
    assert_eq!(q4_execution["maximum_open_source_shards"].as_u64(), Some(1));
    assert_eq!(q4_execution["source_tensor_reads"].as_u64(), Some(210));
    assert_eq!(
        q4_execution["source_payload_bytes"].as_u64(),
        Some(RETAINED_F16_TENSOR_BYTES)
    );
    assert_eq!(q4["quality_evaluated"].as_bool(), Some(false));
    assert_eq!(
        record["checks"]["native_quantization_real_q8_conversion_validated"].as_bool(),
        Some(true)
    );
    assert_eq!(
        record["checks"]["native_quantization_real_q4_conversion_validated"].as_bool(),
        Some(true)
    );
    assert_eq!(
        record["checks"]["native_quantization_real_q8_packed_execution_validated"].as_bool(),
        Some(true)
    );
    assert_eq!(
        record["checks"]["native_quantization_real_q4_packed_execution_validated"].as_bool(),
        Some(true)
    );
    assert_eq!(
        record["checks"]["native_quantization_real_packed_execution_validated"].as_bool(),
        Some(true)
    );
    assert_eq!(
        record["checks"]["source_ordered_single_shard_streaming_validated"].as_bool(),
        Some(true)
    );
    assert_eq!(
        record["checks"]["quantization_validated"].as_bool(),
        Some(false)
    );
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
    let (_, eligible_elements) = assert_muse_quantization_partition(&plan);
    assert_eq!(eligible_elements * 68 / 64, Q8_PACKED_PAYLOAD_BYTES);
    assert_eq!(eligible_elements * 36 / 64, Q4_PACKED_PAYLOAD_BYTES);

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
    let reference_encoding = tokenizer
        .encode(
            REFERENCE_PROMPT,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("the rendered acceptance prompt must encode");
    assert_eq!(reference_encoding.token_ids()[0], REFERENCE_TOKEN_IDS);
    assert_eq!(
        reference_encoding.valid_tokens()[0],
        [true; REFERENCE_TOKEN_IDS.len()]
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

/// Streams one immutable Muse text plan into a create-only Q8/Q4 sidecar.
#[cfg(feature = "tokenizers")]
fn quantize_pinned_muse_glimmer(scheme: SymmetricQuantization, output_environment: &str) {
    let started = Instant::now();
    let supplied_root = pinned_root();
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert_control_files(&root);
    let (manifest, plan) = pinned_muse_f16_plan(&root);
    let (expected_quantized, _) = assert_muse_quantization_partition(&plan);
    let output = muse_burnpack_path(&root, output_environment, false);
    let (expected_payload_bytes, expected_largest_payload) = muse_packed_geometry(scheme);
    let outcome = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions {
            weights: WeightQuantizationConfig {
                scheme,
                group_size: QuantizationGroupSize::new(64).expect("supported G64"),
            },
            ..QuantizedCheckpointOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("stream Muse {scheme:?} Burnpack conversion: {error}"));
    assert_eq!(outcome.artifact.path, output);
    assert_eq!(outcome.artifact.tensor_count, QUANTIZED_TENSOR_COUNT);
    assert_eq!(
        outcome.artifact.quantized_payload_bytes,
        expected_payload_bytes
    );
    assert_eq!(
        outcome.artifact.largest_materialized_tensor_bytes,
        expected_largest_payload
    );
    assert_eq!(outcome.staging.tensor_count, QUANTIZED_TENSOR_COUNT);
    assert_eq!(outcome.staging.staged_payload_bytes, expected_payload_bytes);
    assert_eq!(
        outcome.staging.largest_payload_bytes,
        expected_largest_payload
    );
    assert!(outcome.report.is_strictly_valid());
    assert_eq!(outcome.report.applied.len(), CONSUMED_TENSOR_COUNT);
    assert_eq!(outcome.report.transformed.len(), RENAMED_TENSOR_COUNT);
    assert_eq!(outcome.report.dtype_converted.len(), CONSUMED_TENSOR_COUNT);
    assert_eq!(
        outcome.report.allowlisted_ignored.len(),
        VISUAL_TENSOR_COUNT
    );
    assert_eq!(outcome.report.quantized.len(), QUANTIZED_TENSOR_COUNT);
    assert_eq!(
        outcome
            .report
            .quantized
            .iter()
            .map(|record| record.target.clone())
            .collect::<Vec<_>>(),
        expected_quantized
    );
    assert_eq!(outcome.excluded.len(), RETAINED_F16_TENSOR_COUNT);
    assert_eq!(outcome.execution.tensors_applied, CONSUMED_TENSOR_COUNT);
    assert_eq!(outcome.execution.aliases_applied, 0);
    assert_eq!(outcome.execution.source_reads, CONSUMED_TENSOR_COUNT as u64);
    assert_eq!(
        outcome.execution.source_payload_bytes,
        LANGUAGE_PARAMETER_BYTES
    );
    assert_eq!(
        outcome.execution.delivered_tensor_bytes,
        LANGUAGE_PARAMETER_BYTES
    );
    assert_eq!(outcome.execution.source.shard_opens, SHARDS.len() as u64);
    assert_eq!(
        outcome.execution.source.tensor_reads,
        CONSUMED_TENSOR_COUNT as u64
    );
    assert_eq!(
        outcome.execution.source.payload_bytes_read,
        LANGUAGE_PARAMETER_BYTES
    );
    assert_eq!(outcome.execution.source.max_open_shards, 1);
    assert_eq!(
        outcome.execution.source.shard_open_order,
        SHARDS
            .iter()
            .map(|(name, _, _)| (*name).to_owned())
            .collect::<Vec<_>>()
    );
    for (phase, stats) in [
        ("pre-execution", outcome.provenance.pre_execution),
        ("post-execution", outcome.provenance.post_execution),
    ] {
        let stats = stats.unwrap_or_else(|| panic!("conversion omitted {phase} provenance"));
        assert_eq!(stats.passes_completed, 1, "{phase} pass count");
        assert_eq!(stats.shards_hashed, SHARDS.len(), "{phase} shard count");
        assert_eq!(
            stats.bytes_hashed, CHECKPOINT_ARTIFACT_BYTES,
            "{phase} byte count"
        );
        assert_eq!(stats.peak_buffer_bytes, PROVENANCE_HASH_BUFFER_BYTES);
    }

    let mut artifact = QuantizedBurnpackArtifact::open(&output)
        .unwrap_or_else(|error| panic!("reopen published Muse Burnpack: {error}"));
    assert_exact_muse_burnpack(&mut artifact, &plan, scheme, &expected_quantized);
    drop(artifact);
    let sha256 = sha256_file_hex(&output);
    eprintln!(
        "Muse-Glimmer-30B {scheme:?} Burnpack: path={}, tensors={}, retained={}, payload_bytes={}, artifact_bytes={}, largest_payload_bytes={}, sha256={}, elapsed_seconds={:.3}",
        output.display(),
        QUANTIZED_TENSOR_COUNT,
        RETAINED_F16_TENSOR_COUNT,
        expected_payload_bytes,
        outcome.artifact.artifact_bytes,
        expected_largest_payload,
        sha256,
        started.elapsed().as_secs_f64(),
    );
}

/// Converts the exact Muse language graph to a provenance-bound Q8/G64 sidecar.
#[cfg(feature = "tokenizers")]
#[test]
#[ignore = "streams the pinned 59.6 GB Muse checkpoint into a create-only external Q8 Burnpack"]
fn quantizes_pinned_muse_glimmer_weights_to_q8_burnpack() {
    quantize_pinned_muse_glimmer(SymmetricQuantization::Q8, Q8_BURNPACK_ENV);
}

/// Converts the exact Muse language graph to a provenance-bound Q4/G64 sidecar.
#[cfg(feature = "tokenizers")]
#[test]
#[ignore = "streams the pinned 59.6 GB Muse checkpoint into a create-only external Q4 Burnpack"]
fn quantizes_pinned_muse_glimmer_weights_to_q4_burnpack() {
    quantize_pinned_muse_glimmer(SymmetricQuantization::Q4, Q4_BURNPACK_ENV);
}

/// Requires one Muse linear map to retain packed QFloat storage without a bias.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
fn assert_muse_linear_qfloat(label: &str, linear: &Linear<Metal>) {
    assert!(
        matches!(linear.weight.val().dtype(), DType::QFloat(_)),
        "{label}.weight must retain packed QFloat storage"
    );
    assert!(
        linear.bias.is_none(),
        "the pinned bias-free Muse graph unexpectedly contains {label}.bias"
    );
}

/// Proves every architecture-eligible matrix and exclusion retained its dtype.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
fn assert_muse_mixed_parameter_dtypes(model: &MuseGlimmerForCausalLm<Metal>) {
    assert_eq!(model.model.embed_tokens.weight.val().dtype(), DType::F16);
    assert_eq!(model.model.norm.weight.val().dtype(), DType::F16);
    let mut packed = 0_usize;
    let mut retained = 2_usize;
    for (layer_index, layer) in model.model.layers.iter().enumerate() {
        for (name, linear) in [
            ("self_attn.q_proj", &layer.self_attn.q_proj),
            ("self_attn.k_proj", &layer.self_attn.k_proj),
            ("self_attn.v_proj", &layer.self_attn.v_proj),
            ("self_attn.gate_proj", &layer.self_attn.gate_proj),
            ("self_attn.o_proj", &layer.self_attn.o_proj),
            ("mlp.gate_proj", &layer.mlp.gate_proj),
            ("mlp.up_proj", &layer.mlp.up_proj),
            ("mlp.down_proj", &layer.mlp.down_proj),
        ] {
            assert_muse_linear_qfloat(&format!("layer {layer_index} {name}"), linear);
            packed += 1;
        }
        for (name, norm) in [
            ("input", &layer.input_layernorm),
            ("post_attention", &layer.post_attention_layernorm),
            ("pre_feedforward", &layer.pre_feedforward_layernorm),
            ("post_feedforward", &layer.post_feedforward_layernorm),
        ] {
            assert_eq!(
                norm.weight.val().dtype(),
                DType::F16,
                "layer {layer_index} {name} norm must remain F16"
            );
            retained += 1;
        }
    }
    assert_muse_linear_qfloat(
        "lm_head",
        model
            .lm_head
            .as_ref()
            .expect("the pinned Muse graph has an independent LM head"),
    );
    packed += 1;
    assert_eq!(packed, QUANTIZED_TENSOR_COUNT);
    assert_eq!(retained, RETAINED_F16_TENSOR_COUNT);
}

/// A fully authenticated mixed-precision graph and its cold load duration.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
struct AuthenticatedMuseBurnpack {
    /// Executable model plus the strict load and provenance reports it produced.
    loaded: LoadedModel<AutoCausalLm<Metal>>,
    /// Synchronized load duration retained only as a non-standardized diagnostic.
    load_seconds: f64,
}

/// Authenticates one reviewed sidecar and source checkpoint before returning a model.
///
/// This common prelude is intentionally shared by packed-kernel and quality
/// gates. It fully rehashes and decodes the sidecar, verifies its F16 load-plan
/// binding, lets the quantized loader perform source pre/postflight hashing,
/// and proves all 417 eligible matrices remain packed while every exclusion
/// remains F16.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
fn load_authenticated_muse_burnpack(
    scheme: SymmetricQuantization,
    output_environment: &str,
    device: &<Metal as burn::tensor::backend::BackendTypes>::Device,
) -> AuthenticatedMuseBurnpack {
    let supplied_root = pinned_root();
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    assert_metal_backend_available(device);
    assert_control_files(&root);
    let (_manifest, plan) = pinned_muse_f16_plan(&root);
    let (expected_quantized, _) = assert_muse_quantization_partition(&plan);
    let burnpack = muse_burnpack_path(&root, output_environment, true);
    let (expected_sha256, expected_artifact_bytes, expected_file_name) = match scheme {
        SymmetricQuantization::Q8 => (
            Q8_BURNPACK_SHA256,
            Q8_BURNPACK_ARTIFACT_BYTES,
            Q8_BURNPACK_FILE_NAME,
        ),
        SymmetricQuantization::Q4 => (
            Q4_BURNPACK_SHA256,
            Q4_BURNPACK_ARTIFACT_BYTES,
            Q4_BURNPACK_FILE_NAME,
        ),
    };
    assert_eq!(
        burnpack.file_name().and_then(|name| name.to_str()),
        Some(expected_file_name),
        "reviewed sidecar filename changed"
    );
    let expected_sha256 = expected_sha256
        .expect("review and pin the complete Muse Burnpack SHA-256 before Metal execution");
    let expected_artifact_bytes = expected_artifact_bytes
        .expect("review and pin the complete Muse Burnpack length before Metal execution");
    assert_eq!(
        fs::metadata(&burnpack)
            .expect("Muse Burnpack metadata")
            .len(),
        expected_artifact_bytes
    );
    assert_eq!(sha256_file_hex(&burnpack), expected_sha256);
    let mut artifact = QuantizedBurnpackArtifact::open(&burnpack)
        .expect("open content-authenticated Muse Burnpack");
    assert_exact_muse_burnpack(&mut artifact, &plan, scheme, &expected_quantized);
    drop(artifact);

    let load_started = Instant::now();
    let loaded = AutoCausalLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            load_tokenizer: false,
            quantized_burnpack: Some(burnpack),
            repository: Some(REPOSITORY.to_owned()),
            revision: Some(REVISION.to_owned()),
            ..AutoLoadOptions::default()
        },
        device,
    )
    .expect("strictly load mixed F16/packed Muse through the automatic registry");
    <Metal<f32, i32> as Backend>::sync(device).expect("complete mixed Muse parameter transfers");
    let load_seconds = load_started.elapsed().as_secs_f64();
    assert_eq!(loaded.source().model_type(), "muse_glimmer_text");
    assert_eq!(loaded.source().repository(), Some(REPOSITORY));
    assert_eq!(loaded.source().revision(), Some(REVISION));
    let report = loaded.load_report();
    assert!(report.is_strictly_valid());
    assert_eq!(report.applied.len(), CONSUMED_TENSOR_COUNT);
    assert_eq!(report.transformed.len(), RENAMED_TENSOR_COUNT);
    assert_eq!(report.dtype_converted.len(), CONSUMED_TENSOR_COUNT);
    assert_eq!(report.quantized.len(), QUANTIZED_TENSOR_COUNT);
    assert_eq!(report.allowlisted_ignored.len(), VISUAL_TENSOR_COUNT);
    let expected_scheme = match scheme {
        SymmetricQuantization::Q8 => "burn-qfloat-symmetric-q8-g64-block-1xg",
        SymmetricQuantization::Q4 => "burn-qfloat-symmetric-q4-g64-block-1xg",
    };
    assert!(
        report
            .quantized
            .iter()
            .all(|record| record.scheme == expected_scheme)
    );
    assert_eq!(
        report
            .quantized
            .iter()
            .map(|record| record.target.clone())
            .collect::<Vec<_>>(),
        expected_quantized
    );
    let (packed_payload_bytes, largest_packed_payload) = muse_packed_geometry(scheme);
    let execution = loaded
        .execution_stats()
        .expect("packed Muse load must retain source counters");
    assert_eq!(execution.tensors_applied, CONSUMED_TENSOR_COUNT);
    assert_eq!(execution.aliases_applied, 0);
    assert_eq!(execution.source_reads, RETAINED_F16_TENSOR_COUNT as u64);
    assert_eq!(execution.source_payload_bytes, RETAINED_F16_TENSOR_BYTES);
    assert_eq!(
        execution.delivered_tensor_bytes,
        RETAINED_F16_TENSOR_BYTES + packed_payload_bytes
    );
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
    assert_eq!(execution.source.shard_opens, SHARDS.len() as u64);
    assert_eq!(
        execution.source.tensor_reads,
        RETAINED_F16_TENSOR_COUNT as u64
    );
    assert_eq!(
        execution.source.payload_bytes_read,
        RETAINED_F16_TENSOR_BYTES
    );
    assert_eq!(execution.source.max_open_shards, 1);
    assert_eq!(
        execution.source.shard_open_order,
        SHARDS
            .iter()
            .map(|(name, _, _)| (*name).to_owned())
            .collect::<Vec<_>>()
    );
    let audit = loaded
        .quantized_load_audit()
        .expect("packed Muse load must retain sidecar audit counters");
    assert_eq!(
        audit.format_version(),
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    assert_eq!(audit.preflight().passes_started, 1);
    assert_eq!(audit.preflight().passes_completed, 1);
    assert_eq!(
        audit.preflight().tensor_reads,
        QUANTIZED_TENSOR_COUNT as u64
    );
    assert_eq!(audit.preflight().payload_bytes, packed_payload_bytes);
    assert_eq!(
        audit.preflight().largest_payload_bytes,
        largest_packed_payload
    );
    assert!(audit.preflight().peak_owned_payload_bytes >= largest_packed_payload);
    assert_eq!(audit.packed().tensor_reads, QUANTIZED_TENSOR_COUNT as u64);
    assert_eq!(audit.packed().tensors_applied, QUANTIZED_TENSOR_COUNT);
    assert_eq!(audit.packed().payload_bytes, packed_payload_bytes);
    assert_eq!(audit.packed().largest_payload_bytes, largest_packed_payload);
    for stats in [
        audit.provenance().pre_execution,
        audit.provenance().post_execution,
    ] {
        let stats = stats.expect("v2 Muse load must hash source before and after execution");
        assert_eq!(stats.passes_completed, 1);
        assert_eq!(stats.shards_hashed, SHARDS.len());
        assert_eq!(stats.bytes_hashed, CHECKPOINT_ARTIFACT_BYTES);
        assert_eq!(stats.peak_buffer_bytes, PROVENANCE_HASH_BUFFER_BYTES);
    }

    let model = match loaded.model() {
        AutoCausalLm::MuseGlimmer(model) => model,
        model => panic!("expected Muse automatic variant, found {model:?}"),
    };
    assert_muse_mixed_parameter_dtypes(model);
    AuthenticatedMuseBurnpack {
        loaded,
        load_seconds,
    }
}

/// Loads, executes, and optionally traces one exact Muse packed sidecar.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
fn smoke_and_audit_pinned_muse_glimmer_burnpack(
    scheme: SymmetricQuantization,
    output_environment: &str,
) {
    let device = Default::default();
    let AuthenticatedMuseBurnpack {
        loaded,
        load_seconds,
    } = load_authenticated_muse_burnpack(scheme, output_environment, &device);
    #[cfg(feature = "kernel-tracing")]
    let model = match loaded.model() {
        AutoCausalLm::MuseGlimmer(model) => model,
        model => panic!("expected Muse automatic variant, found {model:?}"),
    };
    let sequence = SMOKE_TOKEN_IDS.len();
    let prompt = CausalLmInput {
        token_ids: Tensor::<Metal, 2, Int>::from_data(
            TensorData::new(
                SMOKE_TOKEN_IDS
                    .iter()
                    .copied()
                    .map(i64::from)
                    .collect::<Vec<_>>(),
                [1, sequence],
            ),
            &device,
        ),
        token_mask: Some(all_valid_metal_mask(sequence, &device)),
        position_ids: None,
    };
    let diagnostics = ForwardOptions {
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let stateless_started = Instant::now();
    let stateless = loaded
        .model()
        .forward(prompt.clone(), &diagnostics)
        .expect("execute stateless mixed Muse graph");
    let stateless_logits = metal_values("quantized Muse stateless logits", &stateless.logits);
    let stateless_seconds = stateless_started.elapsed().as_secs_f64();
    assert_eq!(stateless.logits.dims(), [1, 1, 202_048]);
    assert_eq!(stateless.hidden_states.len(), 54);
    assert!(stateless_logits.iter().all(|value| value.is_finite()));

    let mut state = loaded
        .model()
        .new_decode_state(
            CacheConfig::new(1, sequence + 1).expect("packed smoke cache geometry"),
            1,
            &device,
        )
        .expect("allocate packed Muse KV state");
    let prefill_started = Instant::now();
    let cached = loaded
        .model()
        .prefill(prompt, &mut state, &diagnostics)
        .expect("execute cached mixed Muse prefill");
    let cached_logits = metal_values("quantized Muse cached logits", &cached.logits);
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    assert_eq!(cached.logits.dims(), [1, 1, 202_048]);
    assert_eq!(cached.hidden_states.len(), 54);
    assert!(cached_logits.iter().all(|value| value.is_finite()));
    let cached_cosine = cosine_similarity(&stateless_logits, &cached_logits);
    assert!(
        cached_cosine >= 0.999,
        "packed Muse cached/stateless cosine {cached_cosine} is below 0.999"
    );
    let stateless_top = top1_token_id(&stateless_logits);
    let cached_top = top1_token_id(&cached_logits);
    assert_eq!(cached_top, stateless_top);
    assert_eq!(state.lengths(), &[sequence]);

    let decode_input = DecodeInput {
        token_ids: Tensor::<Metal, 2, Int>::from_data(
            TensorData::new(
                vec![i64::try_from(stateless_top).expect("Muse token ID fits i64")],
                [1, 1],
            ),
            &device,
        ),
        token_mask: Some(all_valid_metal_mask(1, &device)),
        position_ids: None,
    };
    let decode_started = Instant::now();
    #[cfg(feature = "kernel-tracing")]
    let decoded_logits = {
        let calibration_weight = model.model.layers[0].self_attn.k_proj.weight.val();
        let (_, calibration_trace) =
            capture_kernel_launches(|| calibration_weight.dequantize().into_data());
        let dequantization_names = calibrated_dequantization_kernel_names(&calibration_trace);
        assert!(
            !dequantization_names.is_empty(),
            "installed Muse {scheme:?} weight must calibrate whole-tensor dequantization"
        );
        let (decoded_logits, decode_trace) = capture_kernel_launches(|| {
            let decoded = loaded
                .model()
                .decode(decode_input, &mut state, &ForwardOptions::default())
                .expect("execute recurrent mixed Muse decode");
            metal_values("quantized Muse decoded logits", &decoded.logits)
        });
        let packed = packed_matmul_launches(&decode_trace);
        assert_eq!(
            packed.len(),
            PACKED_DECODE_MATMUL_LAUNCHES,
            "Muse recurrent decode must execute every architecture-declared packed projection"
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
            "Muse {scheme:?} decode launched whole-weight dequantization: {dequantizations:#?}"
        );
        decoded_logits
    };
    #[cfg(not(feature = "kernel-tracing"))]
    let decoded_logits = {
        let decoded = loaded
            .model()
            .decode(decode_input, &mut state, &ForwardOptions::default())
            .expect("execute recurrent mixed Muse decode");
        metal_values("quantized Muse decoded logits", &decoded.logits)
    };
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    assert!(decoded_logits.iter().all(|value| value.is_finite()));
    assert_eq!(state.lengths(), &[sequence + 1]);
    eprintln!(
        "Muse-Glimmer-30B {scheme:?} packed Metal smoke: load_seconds={load_seconds:.3}, stateless_seconds={stateless_seconds:.3}, prefill_seconds={prefill_seconds:.3}, decode_seconds={decode_seconds:.3}, cached_cosine={cached_cosine:.9}, prompt_top1={stateless_top}, decode_top1={}",
        top1_token_id(&decoded_logits),
    );
}

/// Rejects an audit invocation whose binary lacks CubeCL kernel instrumentation.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
fn require_muse_packed_kernel_tracing() {
    #[cfg(not(feature = "kernel-tracing"))]
    panic!("Muse packed-kernel audits require the `kernel-tracing` feature");
}

/// Runs the content-pinned Q8 sidecar and audits all recurrent packed kernels.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
#[test]
#[ignore = "requires pinned Muse weights, reviewed Q8 Burnpack, Metal, and unified memory"]
fn smokes_and_audits_pinned_muse_glimmer_q8_burnpack_on_metal() {
    require_muse_packed_kernel_tracing();
    smoke_and_audit_pinned_muse_glimmer_burnpack(SymmetricQuantization::Q8, Q8_BURNPACK_ENV);
}

/// Runs the content-pinned Q4 sidecar and audits all recurrent packed kernels.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
#[test]
#[ignore = "requires pinned Muse weights, reviewed Q4 Burnpack, Metal, and unified memory"]
fn smokes_and_audits_pinned_muse_glimmer_q4_burnpack_on_metal() {
    require_muse_packed_kernel_tracing();
    smoke_and_audit_pinned_muse_glimmer_burnpack(SymmetricQuantization::Q4, Q4_BURNPACK_ENV);
}

/// Scores one reviewed sidecar against the matching official cached F16 rows.
///
/// After the authenticated prompt prefill, each decode consumes the committed
/// official token rather than the quantized model's prediction. Therefore row
/// `i` always represents the same prefix and target in both streams even when a
/// quantized top-one changes. Cached/stateless self-consistency is neither
/// computed nor accepted as quality evidence here.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
fn measure_pinned_muse_glimmer_quantized_quality(
    scheme: SymmetricQuantization,
    output_environment: &str,
    maximum_relative_nll_regression: f64,
) {
    let provenance = rust_acceptance_provenance();
    assert_eq!(provenance.cargo_package_rust_version, "1.92");
    assert_eq!(provenance.profile, "release");
    eprintln!(
        "muse_rust_acceptance_provenance={}",
        serde_json::to_string(&provenance).expect("serialize Rust acceptance provenance")
    );
    let oracle = read_pinned_muse_reference();
    let reference = &oracle.reference;
    let row_positions = &reference.outputs.cached_prefix_logits.row_position_ids;
    let targets = &reference.generation.token_ids;
    assert_eq!(targets.as_slice(), REFERENCE_GENERATED_TOKEN_IDS);
    assert_eq!(
        reference
            .outputs
            .cached_prefix_logits
            .row_top1_token_ids
            .as_slice(),
        targets.as_slice()
    );
    assert_eq!(
        row_positions,
        &reference.execution.cache.decision_position_ids
    );
    let mut quality = FixedSequenceQuality::new(
        &oracle.cached_prefix_logits,
        targets,
        row_positions,
        202_048,
    )
    .unwrap_or_else(|error| panic!("validate Muse official cached quality contract: {error}"));

    let device = Default::default();
    let AuthenticatedMuseBurnpack {
        loaded,
        load_seconds,
    } = load_authenticated_muse_burnpack(scheme, output_environment, &device);
    let prompt_length = reference.input_ids.len();
    assert_eq!(prompt_length, REFERENCE_TOKEN_IDS.len());
    let cache_capacity = prompt_length + targets.len();
    let mut state = loaded
        .model()
        .new_decode_state(
            CacheConfig::new(1, cache_capacity).expect("Muse quality cache geometry"),
            1,
            &device,
        )
        .expect("allocate Muse quantized-quality decode state");
    assert_muse_cache_geometry(&state, 0, cache_capacity);
    let quality_started = Instant::now();
    let mut cached = loaded
        .model()
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(
                        reference
                            .input_ids
                            .iter()
                            .copied()
                            .map(i64::from)
                            .collect::<Vec<_>>(),
                        [1, prompt_length],
                    ),
                    &device,
                ),
                token_mask: Some(all_valid_metal_mask(prompt_length, &device)),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("prefill the fixed Muse quality prompt");

    for (step, &target) in targets.iter().enumerate() {
        let expected_length = reference.execution.cache.decision_logical_lengths[step];
        let position_id = row_positions[step];
        assert_eq!(position_id + 1, expected_length);
        assert_muse_cache_geometry(&state, expected_length, cache_capacity);
        assert_eq!(cached.logits.dims(), [1, 1, 202_048]);
        assert_eq!(cached.logits.dtype(), DType::F16);
        let quantized_row = metal_values(
            &format!("Muse {scheme:?} quality cached logits row {step}"),
            &cached.logits,
        );
        quality
            .observe(position_id, target, &quantized_row)
            .unwrap_or_else(|error| panic!("score Muse {scheme:?} quality row {step}: {error}"));

        if step + 1 < targets.len() {
            assert_eq!(
                reference.execution.cache.decode_input_position_ids[step],
                row_positions[step + 1]
            );
            cached = loaded
                .model()
                .decode(
                    DecodeInput {
                        token_ids: Tensor::<Metal, 2, Int>::from_data(
                            TensorData::new(vec![i64::from(target)], [1, 1]),
                            &device,
                        ),
                        token_mask: Some(all_valid_metal_mask(1, &device)),
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("teacher-force Muse {scheme:?} fixed token at row {step}: {error}")
                });
            assert_muse_cache_geometry(&state, expected_length + 1, cache_capacity);
        }
    }
    let quality_seconds = quality_started.elapsed().as_secs_f64();
    let summary = quality
        .finish()
        .unwrap_or_else(|error| panic!("finalize Muse {scheme:?} quality: {error}"));
    eprintln!(
        "Muse-Glimmer-30B {scheme:?} official cached fixed-sequence quality: \
         rows={}, official_mean_nll={:.12}, quantized_mean_nll={:.12}, \
         relative_nll_regression={:.9}, maximum_relative_nll_regression={:.9}, \
         exact_top1_matches={}, minimum_full_vector_cosine={:.12}, \
         load_seconds={load_seconds:.3}, fixed_sequence_seconds={quality_seconds:.3}",
        summary.rows,
        summary.official_mean_nll,
        summary.quantized_mean_nll,
        summary.relative_nll_regression,
        maximum_relative_nll_regression,
        summary.exact_top1_matches,
        summary.minimum_full_vector_cosine,
    );
    enforce_regression_limit(&summary, maximum_relative_nll_regression)
        .unwrap_or_else(|error| panic!("Muse {scheme:?} quantized quality failed: {error}"));
}

/// Enforces the one-percent Q8 mean-NLL budget on all 32 official cached rows.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
#[test]
#[ignore = "requires pinned Muse weights, schema-3 oracle, reviewed Q8 Burnpack, Metal, and unified memory"]
fn measures_pinned_muse_glimmer_q8_official_cached_quality_on_metal() {
    measure_pinned_muse_glimmer_quantized_quality(
        SymmetricQuantization::Q8,
        Q8_BURNPACK_ENV,
        Q8_MAXIMUM_RELATIVE_NLL_REGRESSION,
    );
}

/// Enforces the five-percent Q4 mean-NLL budget on all 32 official cached rows.
#[cfg(all(
    feature = "metal",
    feature = "tokenizers",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
#[test]
#[ignore = "requires pinned Muse weights, schema-3 oracle, reviewed Q4 Burnpack, Metal, and unified memory"]
fn measures_pinned_muse_glimmer_q4_official_cached_quality_on_metal() {
    measure_pinned_muse_glimmer_quantized_quality(
        SymmetricQuantization::Q4,
        Q4_BURNPACK_ENV,
        Q4_MAXIMUM_RELATIVE_NLL_REGRESSION,
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
    let device = Default::default();
    assert_metal_backend_available(&device);
    assert_control_files(&root);
    assert_shard_file_hashes(&root);
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

/// Fails quickly when macOS does not expose a usable Metal adapter.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
#[test]
#[ignore = "requires an available Metal adapter"]
fn initializes_muse_metal_acceptance_backend() {
    assert_metal_backend_available(&Default::default());
}

/// Compares Burn Metal with the authenticated official eager/F16 MPS oracle.
///
/// In addition to full-vocabulary prompt logits, this checks every semantic
/// last-token hidden row, all 32 recurrent greedy choices, and the same 32
/// choices from one stateless teacher-forced graph. Checkpoint controls and both
/// 59.6 GB shard payloads are authenticated before Burn sees the artifacts.
#[cfg(all(feature = "metal", feature = "tokenizers"))]
#[test]
#[ignore = "requires reviewed Muse oracle hashes, pinned weights, Metal, and about 70 GiB free unified memory"]
fn matches_pinned_muse_glimmer_official_f16_reference_on_metal() {
    let oracle = read_pinned_muse_reference();
    let reference = &oracle.reference;
    let provenance = rust_acceptance_provenance();
    assert_eq!(provenance.cargo_package_rust_version, "1.92");
    assert_eq!(provenance.profile, "release");
    eprintln!(
        "muse_rust_acceptance_provenance={}",
        serde_json::to_string(&provenance).expect("serialize Rust acceptance provenance")
    );
    let supplied_root = pinned_root();
    let root = supplied_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", supplied_root.display()));
    let device = Default::default();
    assert_metal_backend_available(&device);
    assert_control_files(&root);
    assert_shard_file_hashes(&root);

    let loaded = AutoCausalLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            load_tokenizer: false,
            repository: Some(reference.repository.clone()),
            revision: Some(reference.revision.clone()),
            ..AutoLoadOptions::default()
        },
        &device,
    )
    .expect("strictly stream the pinned Muse text graph into Metal F16");
    <Metal<f32, i32> as Backend>::sync(&device)
        .expect("all Metal parameter transfers must complete");
    assert!(matches!(loaded.model(), AutoCausalLm::MuseGlimmer(_)));
    assert!(loaded.load_report().is_strictly_valid());
    assert_eq!(loaded.load_report().applied.len(), CONSUMED_TENSOR_COUNT);
    assert_eq!(loaded.load_report().transformed.len(), RENAMED_TENSOR_COUNT);
    assert_eq!(
        loaded.load_report().dtype_converted.len(),
        CONSUMED_TENSOR_COUNT
    );
    assert_eq!(loaded.load_report().allowlisted_ignored.len(), 809);

    let prompt_length = reference.input_ids.len();
    let prompt_ids = reference
        .input_ids
        .iter()
        .copied()
        .map(i64::from)
        .collect::<Vec<_>>();
    let prompt_tensor = Tensor::<Metal, 2, Int>::from_data(
        TensorData::new(prompt_ids.clone(), [1, prompt_length]),
        &device,
    );
    let prompt_mask = all_valid_metal_mask(prompt_length, &device);
    let diagnostics = ForwardOptions {
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let stateless = loaded
        .model()
        .forward(
            CausalLmInput {
                token_ids: prompt_tensor.clone(),
                token_mask: Some(prompt_mask.clone()),
                position_ids: None,
            },
            &diagnostics,
        )
        .expect("run stateless Burn Muse prompt inference");
    assert_eq!(stateless.logits.dims(), [1, 1, 202_048]);
    assert_eq!(stateless.logits.dtype(), DType::F16);
    assert_eq!(stateless.hidden_states.len(), 54);
    let actual_logits = metal_values("Muse official-reference logits", &stateless.logits);
    assert!(actual_logits.iter().all(|value| value.is_finite()));
    let prompt_cosine = cosine_similarity(&actual_logits, &oracle.prompt_logits);
    assert!(
        prompt_cosine >= 0.999,
        "official/Burn Muse prompt-logit cosine {prompt_cosine} is below 0.999"
    );
    let actual_top1 = stateless
        .logits
        .clone()
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("Metal prompt top-1 token must use i32 storage");
    assert_eq!(
        u32::try_from(actual_top1[0]).expect("nonnegative Muse token ID"),
        reference.outputs.prompt_logits.top1_token_id
    );

    let actual_hidden = metal_hidden_trace(
        "Muse stateless prompt hidden trace",
        &stateless.hidden_states,
        prompt_length,
    );
    let minimum_hidden_cosine = assert_hidden_trace_parity(
        "official/Burn Muse stateless prompt hidden",
        &actual_hidden,
        &oracle.stateless_prompt_hidden,
        &reference.outputs.hidden_last_trace.labels,
    );

    let mut state = loaded
        .model()
        .new_decode_state(
            CacheConfig::new(1, prompt_length + reference.generation.max_new_tokens)
                .expect("official reference cache geometry must validate"),
            1,
            &device,
        )
        .expect("allocate Muse reference decode state");
    let cache_capacity = prompt_length + reference.generation.max_new_tokens;
    assert_muse_cache_geometry(&state, 0, cache_capacity);
    let mut cached = loaded
        .model()
        .prefill(
            CausalLmInput {
                token_ids: prompt_tensor,
                token_mask: Some(prompt_mask),
                position_ids: None,
            },
            &mut state,
            &diagnostics,
        )
        .expect("prefill the official-reference Muse prompt");
    assert_muse_cache_geometry(&state, prompt_length, cache_capacity);
    assert_eq!(cached.logits.dims(), [1, 1, 202_048]);
    assert_eq!(cached.logits.dtype(), DType::F16);
    let cached_prompt_logits = metal_values("Muse cached prompt logits", &cached.logits);
    let cached_prompt_cosine = cosine_similarity(&cached_prompt_logits, &oracle.prompt_logits);
    assert!(
        cached_prompt_cosine >= 0.999,
        "official/Burn Muse cached prompt cosine {cached_prompt_cosine} is below 0.999"
    );
    let actual_cached_prefill_hidden = metal_hidden_trace(
        "Muse cached-prefill hidden trace",
        &cached.hidden_states,
        prompt_length,
    );
    let minimum_cached_prefill_hidden_cosine = assert_hidden_trace_parity(
        "official/Burn Muse cached-prefill hidden",
        &actual_cached_prefill_hidden,
        &oracle.cached_prefill_hidden,
        &reference.outputs.cached_prefill_hidden_trace.labels,
    );
    let minimum_burn_prefill_stateless_hidden_cosine = assert_hidden_trace_parity(
        "Burn cached-prefill/stateless hidden",
        &actual_cached_prefill_hidden,
        &actual_hidden,
        &reference.outputs.cached_prefill_hidden_trace.labels,
    );

    let mut generated = Vec::with_capacity(reference.generation.max_new_tokens);
    let mut burn_cached_logits = Vec::with_capacity(oracle.cached_prefix_logits.len());
    let mut minimum_cached_official_cosine = f64::INFINITY;
    let mut minimum_decode_hidden_cosine = None;
    for (step, expected_token) in reference
        .generation
        .cached_prefix_top1
        .iter()
        .copied()
        .enumerate()
    {
        let expected_length = reference.execution.cache.decision_logical_lengths[step];
        assert_eq!(
            reference.execution.cache.decision_position_ids[step],
            expected_length - 1
        );
        assert_muse_cache_geometry(&state, expected_length, cache_capacity);
        assert_eq!(cached.logits.dims(), [1, 1, 202_048]);
        assert_eq!(cached.logits.dtype(), DType::F16);
        let cached_row = metal_values(&format!("Muse cached logits step {step}"), &cached.logits);
        let official_cached_row =
            &oracle.cached_prefix_logits[step * 202_048..(step + 1) * 202_048];
        let cached_official_cosine = cosine_similarity(&cached_row, official_cached_row);
        assert!(
            cached_official_cosine >= 0.999,
            "official/Burn Muse cached cosine at step {step} was {cached_official_cosine}"
        );
        minimum_cached_official_cosine = minimum_cached_official_cosine.min(cached_official_cosine);
        let selected =
            u32::try_from(top1_token_id(&cached_row)).expect("Muse vocabulary index must fit u32");
        assert_eq!(
            selected, expected_token,
            "Burn Muse cached greedy token changed at step {step}"
        );
        burn_cached_logits.extend_from_slice(&cached_row);
        generated.push(selected);
        if step + 1 < reference.generation.max_new_tokens {
            assert_eq!(
                reference.execution.cache.decode_input_position_ids[step],
                expected_length
            );
            let decode_options = ForwardOptions {
                hidden_states: step == 0,
                ..ForwardOptions::default()
            };
            cached = loaded
                .model()
                .decode(
                    DecodeInput {
                        token_ids: Tensor::<Metal, 2, Int>::from_data(
                            TensorData::new(vec![i64::from(selected)], [1, 1]),
                            &device,
                        ),
                        token_mask: Some(all_valid_metal_mask(1, &device)),
                        position_ids: None,
                    },
                    &mut state,
                    &decode_options,
                )
                .unwrap_or_else(|error| panic!("Muse cached decode step {step}: {error}"));
            assert_muse_cache_geometry(&state, expected_length + 1, cache_capacity);
            if step == 0 {
                let actual_decode_hidden = metal_hidden_trace(
                    "Muse first recurrent-decode hidden trace",
                    &cached.hidden_states,
                    1,
                );
                minimum_decode_hidden_cosine = Some(assert_hidden_trace_parity(
                    "official/Burn Muse first recurrent-decode hidden",
                    &actual_decode_hidden,
                    &oracle.cached_decode_hidden,
                    &reference.outputs.cached_decode_hidden_trace.labels,
                ));
            } else {
                assert!(cached.hidden_states.is_empty());
            }
        }
    }
    assert_eq!(generated, REFERENCE_GENERATED_TOKEN_IDS);
    assert_eq!(generated, reference.generation.token_ids);
    assert_eq!(burn_cached_logits.len(), oracle.cached_prefix_logits.len());
    assert_muse_cache_geometry(
        &state,
        prompt_length + reference.generation.max_new_tokens - 1,
        cache_capacity,
    );
    let minimum_decode_hidden_cosine =
        minimum_decode_hidden_cosine.expect("first recurrent-decode trace must be checked");

    let mut teacher_ids = prompt_ids;
    teacher_ids.extend(
        generated[..generated.len() - 1]
            .iter()
            .copied()
            .map(i64::from),
    );
    let teacher_length = teacher_ids.len();
    let prediction_positions = (prompt_length - 1
        ..prompt_length - 1 + reference.generation.max_new_tokens)
        .collect::<Vec<_>>();
    assert_eq!(
        prediction_positions,
        reference.execution.cache.decision_position_ids
    );
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
                logits: LogitsSelection::Positions(prediction_positions),
                ..ForwardOptions::default()
            },
        )
        .expect("run Muse stateless teacher-forced prefix inference");
    assert_eq!(
        teacher.logits.dims(),
        [1, reference.generation.max_new_tokens, 202_048]
    );
    assert_eq!(teacher.logits.dtype(), DType::F16);
    let burn_teacher_logits = metal_values("Muse teacher-prefix logits", &teacher.logits);
    assert_eq!(
        burn_teacher_logits.len(),
        oracle.teacher_prefix_logits.len()
    );
    let mut stateless_top1 = Vec::with_capacity(reference.generation.max_new_tokens);
    let mut minimum_teacher_official_cosine = f64::INFINITY;
    let mut minimum_burn_cached_teacher_cosine = f64::INFINITY;
    for (step, ((teacher_row, official_teacher_row), cached_row)) in burn_teacher_logits
        .as_chunks::<202_048>()
        .0
        .iter()
        .zip(oracle.teacher_prefix_logits.as_chunks::<202_048>().0.iter())
        .zip(burn_cached_logits.as_chunks::<202_048>().0.iter())
        .enumerate()
    {
        let teacher_official_cosine = cosine_similarity(teacher_row, official_teacher_row);
        assert!(
            teacher_official_cosine >= 0.999,
            "official/Burn Muse teacher cosine at step {step} was {teacher_official_cosine}"
        );
        minimum_teacher_official_cosine =
            minimum_teacher_official_cosine.min(teacher_official_cosine);
        let cached_teacher_cosine = cosine_similarity(cached_row, teacher_row);
        assert!(
            cached_teacher_cosine >= 0.999,
            "Burn cached/teacher Muse cosine at step {step} was {cached_teacher_cosine}"
        );
        minimum_burn_cached_teacher_cosine =
            minimum_burn_cached_teacher_cosine.min(cached_teacher_cosine);
        stateless_top1.push(
            u32::try_from(top1_token_id(teacher_row)).expect("Muse vocabulary index must fit u32"),
        );
    }
    assert_eq!(
        stateless_top1, reference.generation.stateless_prefix_top1,
        "Burn Muse stateless predictions differ from the official reference"
    );
    assert_eq!(
        stateless_top1, generated,
        "Burn Muse cached and stateless predictions differ"
    );

    eprintln!(
        "Muse-Glimmer-30B official F16 parity: prompt_cosine={prompt_cosine:.9}, \
         cached_prompt_cosine={cached_prompt_cosine:.9}, \
         minimum_hidden_cosine={minimum_hidden_cosine:.9}, \
         minimum_cached_prefill_hidden_cosine={minimum_cached_prefill_hidden_cosine:.9}, \
         minimum_burn_prefill_stateless_hidden_cosine={minimum_burn_prefill_stateless_hidden_cosine:.9}, \
         minimum_decode_hidden_cosine={minimum_decode_hidden_cosine:.9}, \
         minimum_cached_official_cosine={minimum_cached_official_cosine:.9}, \
         minimum_teacher_official_cosine={minimum_teacher_official_cosine:.9}, \
         minimum_burn_cached_teacher_cosine={minimum_burn_cached_teacher_cosine:.9}, \
         greedy_tokens={}",
        generated.len()
    );
}
