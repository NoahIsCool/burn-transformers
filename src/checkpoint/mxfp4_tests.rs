//! MXFP4 decoding: the integer core, the loader hook, and bit-for-bit parity with Transformers.
//!
//! `mxfp4_goldens.json` records Transformers 5.2.0 / PyTorch 2.10.0 decodings
//! of three inputs, each with `convert_moe_packed_tensors(..., dtype=bfloat16)`
//! and `dtype=float32`, the F16 parameter `from_pretrained` produces (the BF16
//! decoding cast to F16), and `dtype=float16` called directly:
//!
//! * an *exhaustive* `[1, 256, 16, 16]` tensor in which row `s` uses scale
//!   code `s` and its 256 bytes are `0..=255`, so every `(scale, byte)` pair
//!   occurs once: the per-pair tables and the SHA-256 of each full output;
//! * a seeded `[2, 3, 4, 16]` tensor with scale codes `115..=136` (the pinned
//!   checkpoint's range), recorded element by element to check the layout;
//! * SHA-256 digests of three expert projections of the pinned gpt-oss-20b
//!   revision, checked by an ignored test that needs the local checkpoint.
//!
//! Regenerate from the repository root (the argument is a local
//! `openai/gpt-oss-20b` snapshot at the pinned revision):
//!
//! ```text
//! script=$(mktemp -d)/gen_mxfp4_goldens.py
//! sed -n '/^\/\/! ```python$/,/^\/\/! ```$/p' src/checkpoint/mxfp4_tests.rs \
//!   | sed -e '1d' -e '$d' -e 's|^//! \{0,1\}||' > "$script"
//! python3 -I "$script" /Volumes/T9/models/openai/gpt-oss-20b \
//!   > src/checkpoint/mxfp4_goldens.json
//! ```
//!
//! ```python
//! import hashlib, json, sys
//!
//! import torch
//! import transformers
//! from safetensors import safe_open
//! from transformers.integrations.mxfp4 import FP4_VALUES, convert_moe_packed_tensors
//!
//! #: Pinned gpt-oss-20b revision whose expert tensors the ignored real-tensor test decodes.
//! REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
//! #: Dequantized expert parameters checked against the pinned checkpoint.
//! PINNED = [
//!     "model.layers.0.mlp.experts.gate_up_proj",
//!     "model.layers.0.mlp.experts.down_proj",
//!     "model.layers.23.mlp.experts.down_proj",
//! ]
//!
//! BITS = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}
//! MASK = {torch.bfloat16: 0xFFFF, torch.float16: 0xFFFF, torch.float32: 0xFFFFFFFF}
//!
//!
//! def bits(tensor):
//!     """Raw IEEE bit patterns of a contiguous float tensor, as unsigned ints."""
//!     dtype = tensor.dtype
//!     return [value & MASK[dtype] for value in tensor.contiguous().view(BITS[dtype]).flatten().tolist()]
//!
//!
//! def digest(tensor):
//!     """SHA-256 of the tensor's little-endian contiguous bytes."""
//!     return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
//!
//!
//! def decodings(blocks, scales):
//!     """Transformers' decodings: direct BF16/F32, its F16 load path, and direct F16."""
//!     bf16 = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)
//!     return {
//!         "bfloat16": bf16,
//!         "float32": convert_moe_packed_tensors(blocks, scales, dtype=torch.float32),
//!         # `from_pretrained` decodes to BF16, then casts the parameter.
//!         "float16": bf16.to(torch.float16),
//!         "float16_direct": convert_moe_packed_tensors(blocks, scales, dtype=torch.float16),
//!     }
//!
//!
//! # Exhaustive input: row s uses scale s for every group; group g holds bytes 16g..16g+15,
//! # so the 256 x 16 x 16 bytes enumerate every (scale, byte) pair once.
//! blocks = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16).repeat(1, 256, 1, 1)
//! scales = torch.arange(256, dtype=torch.uint8).reshape(1, 256, 1).repeat(1, 1, 16)
//! exhaustive = decodings(blocks, scales)
//! tables = {}
//! for name, output in exhaustive.items():
//!     table = [[None] * 16 for _ in range(256)]
//!     flat = bits(output)
//!     columns, rows = output.shape[1], output.shape[2]
//!     for s in range(256):
//!         for column in range(columns):
//!             byte = column // 2  # group-major bytes: column = 32 g + 2 b + h
//!             code = (byte % 256 >> 4) if column % 2 else (byte % 256 & 15)
//!             value = flat[column * rows + s]
//!             assert table[s][code] in (None, value), (name, s, code)
//!             table[s][code] = value
//!     tables[name] = table
//! direct_differs = [
//!     [s, c] for s in range(256) for c in range(16) if tables["float16"][s][c] != tables["float16_direct"][s][c]
//! ]
//!
//! # Small seeded case with realistic exponents, recorded in full to check the layout.
//! generator = torch.Generator().manual_seed(20251015)
//! random_blocks = torch.randint(0, 256, (2, 3, 4, 16), generator=generator, dtype=torch.int64).to(torch.uint8)
//! random_scales = torch.randint(115, 137, (2, 3, 4), generator=generator, dtype=torch.int64).to(torch.uint8)
//! random = decodings(random_blocks, random_scales)
//!
//!
//! def pinned_digests(root):
//!     """Digests of Transformers' decodings of PINNED tensors in a local gpt-oss-20b checkpoint."""
//!     index = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
//!     tensors = []
//!     for name in PINNED:
//!         parts = []
//!         for suffix in ("_blocks", "_scales"):
//!             with safe_open(f"{root}/{index[name + suffix]}", "pt") as handle:
//!                 parts.append(handle.get_tensor(name + suffix))
//!         decoded = decodings(*parts)
//!         tensors.append({
//!             "name": name,
//!             "shape": list(decoded["bfloat16"].shape),
//!             "sha256": {key: digest(decoded[key]) for key in ("bfloat16", "float16", "float32")},
//!         })
//!     return {"revision": REVISION, "tensors": tensors}
//!
//! json.dump(
//!     {
//!         "transformers": transformers.__version__,
//!         "torch": torch.__version__,
//!         "fp4_values": FP4_VALUES,
//!         "exhaustive": {
//!             "shape": list(exhaustive["bfloat16"].shape),
//!             "sha256": {name: digest(output) for name, output in exhaustive.items()},
//!             "tables": tables,
//!             "float16_direct_differs": direct_differs,
//!         },
//!         "random": {
//!             "blocks": random_blocks.flatten().tolist(),
//!             "blocks_shape": list(random_blocks.shape),
//!             "scales": random_scales.flatten().tolist(),
//!             "scales_shape": list(random_scales.shape),
//!             "shape": list(random["bfloat16"].shape),
//!             "bits": {name: bits(output) for name, output in random.items()},
//!         },
//!         "pinned_checkpoint": pinned_digests(sys.argv[1]),
//!     },
//!     sys.stdout,
//!     indent=None,
//!     separators=(",", ":"),
//! )
//! sys.stdout.write("\n")
//! ```

use std::{collections::BTreeMap, fs::File, io::Write, path::Path};

use burn::tensor::TensorData;
use proptest::prelude::*;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use super::*;
use crate::checkpoint::{
    CheckpointError, CheckpointManifest, DTypePolicy, ExpectedTensor, LoadOptions,
    LocalShardSource, MappingPlan, PlannedSource, PlannedTransform, TensorDType, TensorOperation,
    TensorSink, TensorSource, execute_local_load_plan, runtime::dequantize_mxfp4,
};

/// Transformers' FP4 E2M1 values in code order (`integrations/mxfp4.py`).
const FP4_VALUES: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Every target dtype the loader decodes into, with the golden key of its reference.
const GOLDEN_DTYPES: [(TensorDType, &str); 3] = [
    (TensorDType::BF16, "bfloat16"),
    (TensorDType::F32, "float32"),
    (TensorDType::F16, "float16"),
];

/// Parses the committed goldens.
fn goldens() -> Value {
    serde_json::from_str(include_str!("mxfp4_goldens.json")).expect("valid golden JSON")
}

/// Reads a JSON array of unsigned integers.
fn unsigned(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|entry| entry.as_u64().expect("unsigned integer"))
        .collect()
}

/// Returns the exhaustive input: row `s` uses scale `s` and holds bytes `0..=255`.
fn exhaustive_input() -> (Vec<u8>, Vec<u8>) {
    let blocks = (0..256).flat_map(|_| 0..=u8::MAX).collect();
    let scales = (0..=u8::MAX)
        .flat_map(|scale| std::iter::repeat_n(scale, 16))
        .collect();
    (blocks, scales)
}

/// Shape of the exhaustive blocks.
const EXHAUSTIVE_BLOCKS: [u64; 4] = [1, 256, 16, 16];

/// Shape of the exhaustive scales.
const EXHAUSTIVE_SCALES: [u64; 3] = [1, 256, 16];

/// Splits native-endian decoded bytes into unsigned bit patterns of `dtype`.
fn decoded_bits(bytes: &[u8], dtype: TensorDType) -> Vec<u64> {
    match dtype.bits() {
        16 => bytes
            .chunks_exact(2)
            .map(|chunk| u64::from(u16::from_ne_bytes([chunk[0], chunk[1]])))
            .collect(),
        32 => bytes
            .chunks_exact(4)
            .map(|chunk| u64::from(u32::from_ne_bytes(chunk.try_into().unwrap())))
            .collect(),
        _ => bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()))
            .collect(),
    }
}

/// SHA-256 of decoded values serialized little-endian, as the goldens hash them.
fn little_endian_digest(bytes: &[u8], dtype: TensorDType) -> String {
    let width = (dtype.bits() / 8) as usize;
    let mut hasher = Sha256::new();
    if cfg!(target_endian = "little") {
        // Native bytes already are the little-endian serialization.
        hasher.update(bytes);
    } else {
        for value in decoded_bits(bytes, dtype) {
            hasher.update(&value.to_le_bytes()[..width]);
        }
    }
    format!("{:x}", hasher.finalize())
}

/// Decodes through the loader's production path.
fn decode(
    blocks: &[u8],
    blocks_shape: &[u64],
    scales: &[u8],
    scales_shape: &[u64],
    dtype: TensorDType,
) -> Vec<u8> {
    dequantize_mxfp4(
        "test",
        (blocks, blocks_shape),
        (scales, scales_shape),
        dtype,
    )
    .expect("valid MXFP4 pair")
}

/// Builds a geometry from extents, asserting the shapes are valid.
fn geometry(experts: u64, rows: u64, groups: u64) -> Mxfp4Geometry {
    mxfp4_geometry(&[experts, rows, groups, 16], &[experts, rows, groups]).unwrap()
}

/// Decodes one value independently of the kernel: binary64 of `fp4(code) · 2^(scale - 127)`.
fn reference_value(scale: u8, code: u8) -> f64 {
    let fp4 = FP4_VALUES[usize::from(code)];
    if scale == 255 {
        // Transformers' factor is 2^128 = +inf in every dtype it decodes to.
        return if fp4 == 0.0 {
            f64::NAN
        } else {
            fp4 * f64::INFINITY
        };
    }
    // 2^(scale - 127) is a normal binary64 power of two built from its exponent.
    let factor = f64::from_bits((u64::from(scale) + 1023 - 127) << 52);
    fp4 * factor
}

// The integer core.

/// Every byte splits into its low and high nibble codes.
#[test]
fn fp4_codes_split_every_byte_into_low_and_high_nibbles() {
    for byte in 0..=u8::MAX {
        let (low, high) = fp4_codes(byte);
        assert_eq!(low, byte & 0x0F);
        assert_eq!(high, byte >> 4);
        assert_eq!(low + 16 * high, byte);
    }
}

/// Every finite `(scale, code)` pair encodes `fp4(code) · 2^(scale - 127)` exactly.
#[test]
fn exact_value_bits_encode_every_finite_pair_exactly() {
    for scale in 0..255_u8 {
        for code in 0..16_u8 {
            let bits = exact_value_bits(scale, code);
            let expected = reference_value(scale, code);
            assert_eq!(
                bits,
                expected.to_bits(),
                "scale {scale}, code {code}: {} vs {expected}",
                f64::from_bits(bits)
            );
        }
    }
    // Signed zeros keep their sign; the smallest and largest magnitudes are normal binary64.
    assert_eq!(exact_value_bits(0, 8), (-0.0_f64).to_bits());
    assert_eq!(f64::from_bits(exact_value_bits(0, 1)), 2.0_f64.powi(-128));
    assert_eq!(
        f64::from_bits(exact_value_bits(254, 7)),
        6.0 * 2.0_f64.powi(127)
    );
}

/// Scale code 255 decodes zeros to the positive quiet NaN and every other code to ±inf.
#[test]
fn scale_code_255_decodes_like_transformers_infinite_factor() {
    for code in 0..16_u8 {
        let bits = exact_value_bits(255, code);
        match code % 8 {
            0 => assert_eq!(bits, CANONICAL_NAN_BITS, "code {code}"),
            _ => {
                let value = f64::from_bits(bits);
                assert!(value.is_infinite(), "code {code}");
                assert_eq!(value.is_sign_negative(), code >= 8, "code {code}");
            }
        }
    }
}

/// Valid shape pairs yield their extents and derived counts.
#[test]
fn geometry_accepts_matching_shapes_and_derives_counts() {
    let geometry = geometry(2, 3, 4);
    assert_eq!(
        geometry,
        Mxfp4Geometry {
            experts: 2,
            rows: 3,
            groups: 4,
            columns: 128,
        }
    );
    assert_eq!(geometry.values(), 2 * 3 * 128);
    assert_eq!(geometry.block_bytes(), 2 * 3 * 4 * 16);
    assert_eq!(geometry.scale_count(), 2 * 3 * 4);
    assert_eq!(
        decoded_shape(&[2, 3, 4, 16], &[2, 3, 4]).unwrap(),
        vec![2, 128, 3]
    );
}

/// Malformed shape pairs are rejected with the violated rule.
#[test]
fn geometry_rejects_malformed_shape_pairs() {
    let cases: [(&[u64], &[u64], Mxfp4ShapeError); 8] = [
        (&[2, 3, 4], &[2, 3, 4], Mxfp4ShapeError::Blocks),
        (&[2, 3, 4, 8], &[2, 3, 4], Mxfp4ShapeError::Blocks),
        (&[1, 2, 3, 4, 16], &[1, 2, 3, 4], Mxfp4ShapeError::Blocks),
        (&[2, 3, 4, 16], &[2, 3], Mxfp4ShapeError::Scales),
        (&[2, 3, 4, 16], &[2, 3, 5], Mxfp4ShapeError::Scales),
        (&[2, 3, 4, 16], &[3, 3, 4], Mxfp4ShapeError::Scales),
        (&[0, 3, 4, 16], &[0, 3, 4], Mxfp4ShapeError::Empty),
        (
            &[u64::MAX, 2, 1, 16],
            &[u64::MAX, 2, 1],
            Mxfp4ShapeError::Overflow,
        ),
    ];
    for (blocks, scales, expected) in cases {
        assert_eq!(
            mxfp4_geometry(blocks, scales),
            Err(expected),
            "{blocks:?} / {scales:?}"
        );
        assert!(!expected.to_string().is_empty());
    }
    // Overflow of the group extent alone and of the value count.
    assert_eq!(
        mxfp4_geometry(&[1, 1, u64::MAX / 16, 16], &[1, 1, u64::MAX / 16]),
        Err(Mxfp4ShapeError::Overflow)
    );
    assert_eq!(
        mxfp4_geometry(&[1 << 31, 1 << 31, 8, 16], &[1 << 31, 1 << 31, 8]),
        Err(Mxfp4ShapeError::Overflow)
    );
}

/// Block, scale, and output offsets follow their row-major formulas.
#[test]
fn offsets_follow_their_row_major_formulas() {
    let geometry = geometry(2, 3, 4);
    for expert in 0..2 {
        for row in 0..3 {
            for group in 0..4 {
                let scale = scale_offset(geometry, expert, row, group);
                assert_eq!(scale, (expert * 3 + row) * 4 + group);
                assert_eq!(block_offset(geometry, expert, row, group), scale * 16);
            }
            for column in 0..128 {
                assert_eq!(
                    transposed_offset(geometry, expert, column, row),
                    (expert * 128 + column) * 3 + row
                );
            }
        }
    }
    assert_eq!(block_offset(geometry, 1, 2, 3) + 16, geometry.block_bytes());
    assert_eq!(
        transposed_offset(geometry, 1, 127, 2) + 1,
        geometry.values()
    );
}

/// Table indices and coordinates are inverse bijections over all 4096 entries.
#[test]
fn table_index_and_coordinates_are_inverse_bijections() {
    let mut seen = vec![false; TABLE_ENTRIES];
    for scale in 0..=u8::MAX {
        for code in 0..=u8::MAX {
            let index = table_index(scale, code);
            assert_eq!(index, 16 * usize::from(scale) + usize::from(code & 15));
            assert_eq!(table_coordinates(index), (scale, code & 15));
            if code < 16 {
                assert!(!seen[index], "entry {index} claimed twice");
                seen[index] = true;
            }
        }
    }
    assert!(seen.iter().all(|&claimed| claimed));
    for index in 0..TABLE_ENTRIES {
        let (scale, code) = table_coordinates(index);
        assert_eq!(table_index(scale, code), index);
    }
}

/// The decode partition pins its thread count and covers every unit once.
#[test]
fn decode_partition_covers_every_unit_with_at_most_the_planned_threads() {
    for (experts, rows, groups) in [(1, 1, 1), (2, 3, 4), (32, 2880, 90), (1, 1 << 15, 64)] {
        let geometry = geometry(experts, rows, groups);
        let units = geometry.experts * geometry.groups;
        for available in [1, 2, 3, 7, 8, 64, 10_000] {
            for width in [2, 4, 8] {
                let partition = decode_partition(geometry, available, width);
                let expected_threads = if geometry.values() < PARALLEL_MIN_VALUES {
                    1
                } else {
                    available.min(units)
                };
                assert_eq!(partition.threads, expected_threads);
                assert_eq!(partition.unit_bytes, 32 * geometry.rows * width);
                assert_eq!(
                    partition.units_per_thread,
                    units.div_ceil(partition.threads)
                );
                assert_eq!(
                    partition.chunk_bytes,
                    partition.units_per_thread * partition.unit_bytes
                );
                assert_eq!(units * partition.unit_bytes, geometry.values() * width);
                let chunks = units.div_ceil(partition.units_per_thread);
                assert!(chunks <= partition.threads);
                assert!((chunks - 1) * partition.units_per_thread < units);
            }
        }
    }
}

/// Unit coordinates recover the unit and place its values contiguously.
#[test]
fn unit_coordinates_place_each_unit_in_its_output_slice() {
    let geometry = geometry(3, 5, 4);
    for unit in 0..geometry.experts * geometry.groups {
        let (expert, group) = unit_coordinates(geometry, unit);
        assert_eq!(expert * geometry.groups + group, unit);
        for column in 0..BLOCK_VALUES {
            for row in 0..geometry.rows {
                assert_eq!(
                    transposed_offset(geometry, expert, 32 * group + column, row),
                    unit * 32 * geometry.rows + column * geometry.rows + row
                );
            }
        }
    }
}

/// Each block byte's two values land in distinct in-unit slots, covering the unit once.
#[test]
fn unit_value_bytes_cover_each_unit_slot_once() {
    for rows in [1_usize, 2, 5] {
        for width in [2_usize, 4, 8] {
            let mut written = vec![0_u8; BLOCK_VALUES * rows * width];
            for row in 0..rows {
                for byte_index in 0..BLOCK_BYTES {
                    let (even, odd) = unit_value_bytes(rows, row, byte_index, width);
                    assert_eq!(even, (row + 2 * byte_index * rows) * width);
                    assert_eq!(odd, even + rows * width);
                    for slot in [even, odd] {
                        for byte in &mut written[slot..slot + width] {
                            *byte += 1;
                        }
                    }
                }
            }
            assert!(
                written.iter().all(|&count| count == 1),
                "{rows} rows, {width} bytes"
            );
        }
    }
}

/// Every rejected shape names exactly the first rule it violates.
#[test]
fn geometry_errors_name_the_first_violated_rule() {
    let cases: [(&[u64], &[u64], Mxfp4ShapeError); 5] = [
        (&[1, 1, 1, 15], &[0, 0], Mxfp4ShapeError::Blocks),
        (&[1, 1, 1, 16], &[1, 1, 2], Mxfp4ShapeError::Scales),
        (&[0, 1, 1, 16], &[0, 1, 1], Mxfp4ShapeError::Empty),
        (
            &[u64::MAX, 1, 1, 16],
            &[u64::MAX, 1, 1],
            Mxfp4ShapeError::Overflow,
        ),
        (
            &[1, 1, u64::MAX / 16, 16],
            &[1, 1, u64::MAX / 16],
            Mxfp4ShapeError::Overflow,
        ),
    ];
    for (blocks, scales, expected) in cases {
        assert_eq!(
            mxfp4_geometry(blocks, scales),
            Err(expected),
            "{blocks:?} {scales:?}"
        );
    }
}

/// The kernel refuses payloads whose lengths disagree with the geometry.
#[test]
fn decode_rejects_payload_lengths_that_disagree_with_the_geometry() {
    let geometry = geometry(1, 2, 1);
    let table = Mxfp4Table::<4>::new(|bits| (f64::from_bits(bits) as f32).to_ne_bytes());
    let mut output = vec![0; geometry.values() * 4];
    let blocks = vec![0; geometry.block_bytes()];
    let scales = vec![127; geometry.scale_count()];
    assert!(decode_transposed(&blocks, &scales, geometry, &table, &mut output).is_ok());
    assert_eq!(
        decode_transposed(&blocks[1..], &scales, geometry, &table, &mut output),
        Err(Mxfp4ShapeError::Overflow)
    );
    assert_eq!(
        decode_transposed(&blocks, &scales[1..], geometry, &table, &mut output),
        Err(Mxfp4ShapeError::Overflow)
    );
    assert_eq!(
        decode_transposed(&blocks, &scales, geometry, &table, &mut output[4..]),
        Err(Mxfp4ShapeError::Overflow)
    );
}

/// Decodes `(blocks, scales)` element by element through the verified offsets.
fn naive_decode(blocks: &[u8], scales: &[u8], geometry: Mxfp4Geometry) -> Vec<f64> {
    let mut output = vec![0.0; geometry.values()];
    for expert in 0..geometry.experts {
        for row in 0..geometry.rows {
            for column in 0..geometry.columns {
                let group = column / BLOCK_VALUES;
                let byte = blocks[block_offset(geometry, expert, row, group) + column % 32 / 2];
                let (low, high) = fp4_codes(byte);
                let code = if column % 2 == 0 { low } else { high };
                let scale = scales[scale_offset(geometry, expert, row, group)];
                output[transposed_offset(geometry, expert, column, row)] =
                    reference_value(scale, code);
            }
        }
    }
    output
}

/// Converts decoded F64 bytes to values.
fn as_f64(bytes: &[u8]) -> Vec<f64> {
    decoded_bits(bytes, TensorDType::F64)
        .into_iter()
        .map(f64::from_bits)
        .collect()
}

/// Asserts two decodings agree bit for bit (NaN included).
fn assert_same_values(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()),
            "{context}: element {index}: {a} vs {e}"
        );
    }
}

/// A tensor large enough for the multi-threaded path decodes like the element-wise reference.
#[test]
fn parallel_decode_matches_the_element_wise_reference() {
    let (experts, rows, groups) = (3_u64, 64_u64, 192_u64);
    let geometry = geometry(experts, rows, groups);
    assert!(geometry.values() >= PARALLEL_MIN_VALUES);
    let blocks: Vec<u8> = (0..geometry.block_bytes())
        .map(|index| (index.wrapping_mul(2_654_435_761) >> 7) as u8)
        .collect();
    let scales: Vec<u8> = (0..geometry.scale_count())
        .map(|index| 100 + (index % 50) as u8)
        .collect();
    let decoded = decode(
        &blocks,
        &[experts, rows, groups, 16],
        &scales,
        &[experts, rows, groups],
        TensorDType::F64,
    );
    assert_same_values(
        &as_f64(&decoded),
        &naive_decode(&blocks, &scales, geometry),
        "parallel",
    );
}

proptest! {
    /// Arbitrary small tensors decode like the element-wise reference, in every dtype.
    #[test]
    fn small_tensors_decode_like_the_element_wise_reference(
        experts in 1_u64..3,
        rows in 1_u64..5,
        groups in 1_u64..4,
        seed in any::<u64>(),
    ) {
        let geometry = geometry(experts, rows, groups);
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u8
        };
        let blocks: Vec<u8> = (0..geometry.block_bytes()).map(|_| next()).collect();
        let scales: Vec<u8> = (0..geometry.scale_count()).map(|_| next()).collect();
        let blocks_shape = [experts, rows, groups, 16];
        let scales_shape = [experts, rows, groups];
        let exact = as_f64(&decode(&blocks, &blocks_shape, &scales, &scales_shape, TensorDType::F64));
        assert_same_values(&exact, &naive_decode(&blocks, &scales, geometry), "f64");
        // Narrower dtypes round each exact value once.
        let narrow = decode(&blocks, &blocks_shape, &scales, &scales_shape, TensorDType::F32);
        let rounded: Vec<f64> = decoded_bits(&narrow, TensorDType::F32)
            .into_iter()
            .map(|bits| f64::from(f32::from_bits(bits as u32)))
            .collect();
        let expected: Vec<f64> = exact.iter().map(|&value| f64::from(value as f32)).collect();
        assert_same_values(&rounded, &expected, "f32");
    }
}

// Parity with Transformers.

/// Every `(scale, code)` decodes to the Transformers bit pattern in every dtype.
#[test]
fn every_scale_and_code_decodes_like_transformers() {
    let goldens = goldens();
    let (blocks, scales) = exhaustive_input();
    for (dtype, key) in GOLDEN_DTYPES {
        let decoded = decode(
            &blocks,
            &EXHAUSTIVE_BLOCKS,
            &scales,
            &EXHAUSTIVE_SCALES,
            dtype,
        );
        let table = &goldens["exhaustive"]["tables"][key];
        let bits = decoded_bits(&decoded, dtype);
        // Output element (column, row = scale) holds code of byte column / 2.
        for scale in 0..256_usize {
            let expected = unsigned(&table[scale]);
            for column in 0..512_usize {
                let byte = (column / 2) as u8;
                let code = if column % 2 == 0 {
                    byte & 15
                } else {
                    byte >> 4
                };
                assert_eq!(
                    bits[column * 256 + scale],
                    expected[usize::from(code)],
                    "{key}: scale {scale}, code {code}"
                );
            }
        }
        assert_eq!(
            little_endian_digest(&decoded, dtype),
            goldens["exhaustive"]["sha256"][key].as_str().unwrap(),
            "{key}: digest of the whole exhaustive tensor"
        );
    }
}

/// The seeded multi-expert tensor matches Transformers element by element.
#[test]
fn seeded_multi_expert_tensor_decodes_like_transformers() {
    let goldens = goldens();
    let random = &goldens["random"];
    let bytes = |key: &str| -> Vec<u8> {
        unsigned(&random[key])
            .into_iter()
            .map(|value| u8::try_from(value).unwrap())
            .collect()
    };
    let (blocks, scales) = (bytes("blocks"), bytes("scales"));
    let blocks_shape = unsigned(&random["blocks_shape"]);
    let scales_shape = unsigned(&random["scales_shape"]);
    assert_eq!(
        decoded_shape(&blocks_shape, &scales_shape).unwrap(),
        unsigned(&random["shape"])
    );
    for (dtype, key) in GOLDEN_DTYPES {
        let decoded = decode(&blocks, &blocks_shape, &scales, &scales_shape, dtype);
        assert_eq!(
            decoded_bits(&decoded, dtype),
            unsigned(&random["bits"][key]),
            "{key}"
        );
    }
}

/// Calling Transformers' decoder with `dtype=float16` differs only outside the pinned exponents.
///
/// That path rounds the factor `2^(s - 127)` to F16 first; the loader instead
/// rounds the exact value once, as `from_pretrained` does through BF16. The
/// pinned gpt-oss-20b scale codes are `115..=136`, where both agree.
#[test]
fn direct_f16_decoding_differs_only_outside_the_pinned_exponent_range() {
    let goldens = goldens();
    let exhaustive = &goldens["exhaustive"];
    let differing: Vec<(u64, u64)> = exhaustive["float16_direct_differs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| (pair[0].as_u64().unwrap(), pair[1].as_u64().unwrap()))
        .collect();
    assert!(!differing.is_empty());
    assert!(
        differing
            .iter()
            .all(|&(scale, _)| !(115..=136).contains(&scale)),
        "{differing:?}"
    );
    let ours = &exhaustive["tables"]["float16"];
    let direct = &exhaustive["tables"]["float16_direct"];
    for scale in 0..256_u64 {
        for code in 0..16_u64 {
            let same = ours[scale as usize][code as usize] == direct[scale as usize][code as usize];
            assert_eq!(!same, differing.contains(&(scale, code)), "{scale}, {code}");
        }
    }
}

/// Decoding rejects malformed shapes, short payloads, and non-float targets.
#[test]
fn production_decode_rejects_invalid_inputs() {
    let blocks = vec![0_u8; 32];
    let scales = vec![127_u8; 2];
    let shape = [1, 2, 1, 16];
    let scale_shape = [1, 2, 1];
    assert!(matches!(
        dequantize_mxfp4(
            "t",
            (&blocks, &[1, 2, 1, 8]),
            (&scales, &scale_shape),
            TensorDType::F32
        ),
        Err(CheckpointError::InvalidMapping { .. })
    ));
    assert!(matches!(
        dequantize_mxfp4(
            "t",
            (&blocks[1..], &shape),
            (&scales, &scale_shape),
            TensorDType::F32
        ),
        Err(CheckpointError::InvalidTensorPayload { .. })
    ));
    assert!(matches!(
        dequantize_mxfp4(
            "t",
            (&blocks, &shape),
            (&scales, &scale_shape),
            TensorDType::U8
        ),
        Err(CheckpointError::UnsupportedDTypeConversion { .. })
    ));
}

// The loader hook: planning, reporting, and streamed execution.

/// Writes one SafeTensors shard holding `tensors` in order.
fn write_shard(root: &Path, tensors: &[(&str, TensorDType, Vec<u64>, Vec<u8>)]) {
    let mut header = Map::new();
    let mut offset = 0_u64;
    for (name, dtype, shape, bytes) in tensors {
        let end = offset + bytes.len() as u64;
        header.insert(
            (*name).to_owned(),
            json!({"dtype": dtype.as_str(), "shape": shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    for (_, _, _, bytes) in tensors {
        file.write_all(bytes).unwrap();
    }
}

/// Collects every installed tensor.
#[derive(Default)]
struct Collect {
    /// Installed tensors by target.
    tensors: BTreeMap<String, TensorData>,
}

impl TensorSink for Collect {
    /// Never fails.
    type Error = std::convert::Infallible;

    /// Records one tensor.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        self.tensors.insert(target.to_owned(), data);
        Ok(())
    }

    /// Ignores aliases; none are planned here.
    fn alias_tensor(&mut self, _target: &str, _source_target: &str) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// A checkpoint with the seeded golden pair under gpt-oss names, plus its golden values.
fn seeded_checkpoint() -> (TempDir, Vec<u64>, Value) {
    let goldens = goldens();
    let random = goldens["random"].clone();
    let bytes = |key: &str| -> Vec<u8> {
        unsigned(&random[key])
            .into_iter()
            .map(|value| u8::try_from(value).unwrap())
            .collect()
    };
    let directory = TempDir::new().unwrap();
    write_shard(
        directory.path(),
        &[
            (
                "experts.down_proj_blocks",
                TensorDType::U8,
                unsigned(&random["blocks_shape"]),
                bytes("blocks"),
            ),
            (
                "experts.down_proj_scales",
                TensorDType::U8,
                unsigned(&random["scales_shape"]),
                bytes("scales"),
            ),
        ],
    );
    (directory, unsigned(&random["shape"]), random)
}

/// A planned dequantization streams into the sink with the requested dtype and a transform record.
#[test]
fn planned_dequantization_streams_decoded_values_into_the_sink() {
    let (directory, shape, random) = seeded_checkpoint();
    for (dtype, key) in GOLDEN_DTYPES {
        let options = LoadOptions {
            dtype_policy: DTypePolicy::Cast(dtype),
            ..LoadOptions::default()
        };
        let manifest = CheckpointManifest::discover(directory.path(), &options).unwrap();
        let mut mappings = MappingPlan::new();
        mappings.dequantize_mxfp4(
            "experts.down_proj_blocks",
            "experts.down_proj_scales",
            "experts.down_proj",
        );
        assert!(matches!(
            mappings.operations(),
            [TensorOperation::DequantizeMxfp4 { .. }]
        ));
        let expected = [ExpectedTensor::new("experts.down_proj", shape.clone())];
        let plan =
            crate::checkpoint::build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
        let [planned] = plan.tensors.as_slice() else {
            panic!("one planned tensor");
        };
        assert_eq!(planned.transform, PlannedTransform::DequantizeMxfp4);
        assert_eq!(planned.dtype, dtype);
        assert_eq!(planned.shape, shape);
        // Both payloads and the output are staged together.
        let staged = 2 * 3 * 4 * 16 + 2 * 3 * 4 + shape.iter().product::<u64>() * dtype.bits() / 8;
        assert_eq!(planned.staging_bytes().unwrap(), staged);
        assert!(plan.memory.peak_host_staging_bytes >= staged);
        assert!(plan.report.dtype_converted.is_empty(), "{key}");
        assert_eq!(plan.report.transformed.len(), 1);
        assert_eq!(plan.report.transformed[0].operation, "dequantize_mxfp4");
        let mut sink = Collect::default();
        let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();
        assert_eq!(outcome.stats.source_reads, 2);
        let data = &sink.tensors["experts.down_proj"];
        assert_eq!(data.shape.as_slice(), [2, 128, 3]);
        assert_eq!(
            decoded_bits(data.as_bytes(), dtype),
            unsigned(&random["bits"][key]),
            "{key}"
        );
    }
}

/// Planning rejects wrong source dtypes, malformed shapes, and reused sources.
#[test]
fn planning_rejects_invalid_mxfp4_sources() {
    let (directory, shape, _) = seeded_checkpoint();
    let options = LoadOptions::default();
    let manifest = CheckpointManifest::discover(directory.path(), &options).unwrap();
    let expected = [ExpectedTensor::new("experts.down_proj", shape)];
    let plan = |blocks: &str, scales: &str| {
        let mut mappings = MappingPlan::new();
        mappings.dequantize_mxfp4(blocks, scales, "experts.down_proj");
        crate::checkpoint::build_load_plan(&manifest, &expected, &mappings, &options)
    };
    // Swapped roles: the scales are not rank-4 U8 blocks.
    assert!(matches!(
        plan("experts.down_proj_scales", "experts.down_proj_blocks"),
        Err(CheckpointError::InvalidMapping { .. })
    ));
    // One tensor cannot serve as both sources.
    assert!(matches!(
        plan("experts.down_proj_blocks", "experts.down_proj_blocks"),
        Err(CheckpointError::MappingSourceCollision { .. })
    ));
    // Missing sources are reported by name.
    assert!(matches!(
        plan("missing_blocks", "experts.down_proj_scales"),
        Err(CheckpointError::MissingMappingSource { .. })
    ));
    // A target expectation of another shape is a shape mismatch.
    let mut mappings = MappingPlan::new();
    mappings.dequantize_mxfp4(
        "experts.down_proj_blocks",
        "experts.down_proj_scales",
        "experts.down_proj",
    );
    let wrong = [ExpectedTensor::new("experts.down_proj", vec![2, 3, 128])];
    assert!(matches!(
        crate::checkpoint::build_load_plan(&manifest, &wrong, &mappings, &options),
        Err(CheckpointError::ShapeMismatch { .. })
    ));
}

/// Runtime validation rejects a hand-edited MXFP4 work item before any read.
#[test]
fn runtime_rejects_tampered_mxfp4_work_items() {
    let (directory, shape, _) = seeded_checkpoint();
    let options = LoadOptions::default();
    let manifest = CheckpointManifest::discover(directory.path(), &options).unwrap();
    let mut mappings = MappingPlan::new();
    mappings.dequantize_mxfp4(
        "experts.down_proj_blocks",
        "experts.down_proj_scales",
        "experts.down_proj",
    );
    let expected = [ExpectedTensor::new("experts.down_proj", shape)];
    let plan =
        crate::checkpoint::build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
    let tamper = |edit: &dyn Fn(&mut crate::checkpoint::PlannedTensor)| {
        let mut plan = plan.clone();
        edit(&mut plan.tensors[0]);
        execute_local_load_plan(&manifest, &plan, &mut Collect::default()).is_err()
    };
    assert!(tamper(&|tensor| tensor.sources.truncate(1)));
    assert!(tamper(&|tensor| tensor.shape = vec![2, 3, 128]));
    assert!(tamper(&|tensor| tensor.dtype = TensorDType::U8));
    assert!(tamper(&|tensor| tensor.sources.swap(0, 1)));
}

// The pinned checkpoint.

/// Decodes expert projections of the pinned gpt-oss-20b revision exactly like Transformers.
///
/// Needs `TRANSFORMERS_GPT_OSS_DIR` pointing at a local `openai/gpt-oss-20b`
/// snapshot at revision `6cee5e81ee83917806bbde320786a8fb61efebee`; reads
/// about 410 MB of blocks and scales and decodes 796 M values per dtype.
#[test]
#[ignore = "needs the local gpt-oss-20b checkpoint (TRANSFORMERS_GPT_OSS_DIR)"]
fn pinned_checkpoint_experts_decode_like_transformers() {
    let root = std::env::var("TRANSFORMERS_GPT_OSS_DIR")
        .expect("set TRANSFORMERS_GPT_OSS_DIR to the gpt-oss-20b snapshot");
    let goldens = goldens();
    let pinned = &goldens["pinned_checkpoint"];
    let manifest = CheckpointManifest::discover(Path::new(&root), &LoadOptions::default()).unwrap();
    let mut source = LocalShardSource::new(&manifest);
    for tensor in pinned["tensors"].as_array().unwrap() {
        let name = tensor["name"].as_str().unwrap();
        let mut read = |suffix: &str| {
            let inventory = manifest.tensor(&format!("{name}{suffix}")).unwrap();
            source.read_tensor(&PlannedSource::from(inventory)).unwrap()
        };
        let (blocks, scales) = (read("_blocks"), read("_scales"));
        for (dtype, key) in GOLDEN_DTYPES {
            let started = std::time::Instant::now();
            let decoded = decode(
                blocks.bytes(),
                blocks.shape(),
                scales.bytes(),
                scales.shape(),
                dtype,
            );
            eprintln!("{name} {key}: decoded in {:?}", started.elapsed());
            assert_eq!(
                little_endian_digest(&decoded, dtype),
                tensor["sha256"][key].as_str().unwrap(),
                "{name} {key}"
            );
        }
        assert_eq!(
            decoded_shape(blocks.shape(), scales.shape()).unwrap(),
            unsigned(&tensor["shape"])
        );
    }
    source.close().unwrap();
}
