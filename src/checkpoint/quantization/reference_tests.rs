//! Parity, I/O-shape, and timing tests for the reference quantizer.

use std::io::{self, Write};

use burn::tensor::{TensorData, bf16};

use super::*;
use crate::checkpoint::quantization::QuantizationGroupSize;

/// Builds a configuration from a scheme and a supported group size.
fn config(scheme: SymmetricQuantization, group_size: usize) -> WeightQuantizationConfig {
    WeightQuantizationConfig {
        scheme,
        group_size: QuantizationGroupSize::new(group_size).unwrap(),
    }
}

/// Deterministic, smoothly varying weights with mixed signs and magnitudes.
fn weights(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| ((index as f32) * 0.618_034).sin() * (1.0 + (index % 97) as f32 / 13.0))
        .collect()
}

/// Times the reference quantizer and the staging write on large matrices.
///
/// Run with `cargo test --release --lib reference_quantizer_throughput -- --ignored --nocapture`.
#[test]
#[ignore = "timing benchmark; run explicitly in release mode"]
fn reference_quantizer_throughput() {
    use std::time::Instant;

    const ROWS: usize = 4096;
    const COLUMNS: usize = 4096;
    let values = weights(ROWS * COLUMNS);
    let f32_source = TensorData::new(values.clone(), [ROWS, COLUMNS]);
    let bf16_source = TensorData::new(
        values
            .iter()
            .map(|&value| bf16::from_f32(value))
            .collect::<Vec<_>>(),
        [ROWS, COLUMNS],
    );
    for (label, source) in [("F32", &f32_source), ("BF16", &bf16_source)] {
        for scheme in [SymmetricQuantization::Q8, SymmetricQuantization::Q4] {
            let started = Instant::now();
            let weight = quantize_tensor_data(source, config(scheme, 64)).unwrap();
            let quantize = started.elapsed();
            let directory = tempfile::tempdir().unwrap();
            let mut file = std::fs::File::create(directory.path().join("payload.bin")).unwrap();
            let started = Instant::now();
            weight.write_burnpack_payload(&mut file).unwrap();
            let write = started.elapsed();
            println!(
                "{label} {scheme:?}/G64: quantize {:.2} ns/element, stage {:.1} ms for {} scales",
                quantize.as_secs_f64() * 1e9 / (ROWS * COLUMNS) as f64,
                write.as_secs_f64() * 1e3,
                weight.scales().len(),
            );
        }
    }
}

/// Counts `write` calls and bytes while discarding data.
#[derive(Default)]
struct CountingWriter {
    /// Number of `write` invocations observed.
    calls: usize,
    /// Total bytes accepted.
    bytes: Vec<u8>,
}

impl Write for CountingWriter {
    /// Accepts the whole buffer and records one call.
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.calls += 1;
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    /// Nothing is buffered.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The staged payload is packed codes then little-endian scales, in bounded chunks.
#[test]
fn payload_writes_are_chunked_and_byte_identical() {
    let elements = 64 * 40_000;
    let weight = quantize_tensor_data(
        &TensorData::new(weights(elements), [40_000, 64]),
        config(SymmetricQuantization::Q4, 64),
    )
    .unwrap();
    let mut writer = CountingWriter::default();
    weight.write_burnpack_payload(&mut writer).unwrap();
    let mut expected = weight.packed_values().to_vec();
    for scale in weight.scales() {
        expected.extend_from_slice(&scale.to_le_bytes());
    }
    assert_eq!(writer.bytes, expected);
    // One write for the codes plus one per 64 KiB of scales, not one per scale.
    let scale_bytes = weight.scales().len() * 4;
    assert_eq!(writer.calls, 1 + scale_bytes.div_ceil(SCALE_CHUNK_BYTES));
    assert_eq!(
        super::super::provenance::quantized_payload_sha256(weight.packed_values(), weight.scales()),
        super::super::provenance::packed_payload_sha256(&expected)
    );
}

/// Straightforward two-pass reference quantizer over decoded `f32` values.
fn reference_codes(
    values: &[f32],
    scheme: SymmetricQuantization,
    group_size: usize,
) -> (Vec<i8>, Vec<f32>) {
    let qmax = f32::from(scheme.qmax());
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    for group in values.chunks(group_size) {
        let max_abs = group
            .iter()
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        let scale = if max_abs == 0.0 {
            1.0
        } else if max_abs / qmax == 0.0 {
            max_abs
        } else {
            max_abs / qmax
        };
        scales.push(scale);
        codes.extend(
            group
                .iter()
                .map(|value| (value / scale).round().clamp(-qmax, qmax) as i8),
        );
    }
    (codes, scales)
}

proptest::proptest! {
    /// The single-decode quantizer matches the two-pass reference for every source dtype.
    #[test]
    fn single_pass_quantizer_matches_two_pass_reference(
        seed in proptest::prelude::any::<u32>(),
        elements in 0_usize..400,
        dtype in 0_usize..4,
        q4 in proptest::prelude::any::<bool>(),
        group in proptest::sample::select(vec![32_usize, 64, 128]),
    ) {
        let scheme = if q4 { SymmetricQuantization::Q4 } else { SymmetricQuantization::Q8 };
        let raw = (0..elements)
            .map(|index| ((index as f32 + seed as f32) * 0.37).sin() * 3.0 * ((index % 5) as f32 - 2.0))
            .collect::<Vec<_>>();
        let (source, decoded) = match dtype {
            0 => (TensorData::new(raw.clone(), [elements]), raw.clone()),
            1 => {
                let values = raw.iter().map(|&value| bf16::from_f32(value)).collect::<Vec<_>>();
                let decoded = values.iter().map(|value| value.to_f32()).collect();
                (TensorData::new(values, [elements]), decoded)
            }
            2 => {
                let values = raw.iter().map(|&value| burn::tensor::f16::from_f32(value)).collect::<Vec<_>>();
                let decoded = values.iter().map(|value| value.to_f32()).collect();
                (TensorData::new(values, [elements]), decoded)
            }
            _ => {
                let values = raw.iter().map(|&value| f64::from(value) * 1.000_000_1).collect::<Vec<_>>();
                let decoded = values.iter().map(|&value| value as f32).collect();
                (TensorData::new(values, [elements]), decoded)
            }
        };
        let weight = quantize_tensor_data(&source, config(scheme, group)).unwrap();
        let (codes, scales) = reference_codes(&decoded, scheme, group);
        proptest::prop_assert_eq!(weight.scales(), scales.as_slice());
        let width = code_width(scheme);
        for (index, &code) in codes.iter().enumerate() {
            proptest::prop_assert_eq!(read_code(width, weight.packed_values(), index), Some(code));
        }
        // The persisted-payload validator accepts what the quantizer produced.
        let (metadata, packed, scales) = weight.into_parts();
        QuantizedWeight::from_parts(metadata, packed, scales).unwrap();
    }
}
