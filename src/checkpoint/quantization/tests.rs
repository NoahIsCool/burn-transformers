use std::{
    collections::BTreeMap,
    io::{self, Write},
};

use burn::{
    backend::Flex,
    tensor::{
        DType, Tensor, TensorData, bf16, f16,
        quantization::{QuantLevel, QuantStore, QuantValue},
    },
};
use proptest::prelude::*;

use super::*;

fn config(scheme: SymmetricQuantization, group_size: usize) -> WeightQuantizationConfig {
    WeightQuantizationConfig {
        scheme,
        group_size: QuantizationGroupSize::new(group_size).unwrap(),
    }
}

fn f32_values(data: &TensorData) -> Vec<f32> {
    data.as_slice::<f32>().unwrap().to_vec()
}

#[test]
fn group_sizes_are_closed_and_default_to_sixty_four() {
    assert_eq!(QuantizationGroupSize::default().get(), 64);
    for supported in [32, 64, 128] {
        assert_eq!(
            QuantizationGroupSize::new(supported).unwrap().get(),
            supported
        );
    }
    for invalid in [0, 1, 31, 33, 63, 65, 127, 129, usize::MAX] {
        assert!(matches!(
            QuantizationGroupSize::new(invalid),
            Err(QuantizationError::InvalidGroupSize { value }) if value == invalid
        ));
    }
}

#[test]
fn group_size_deserialization_cannot_bypass_validation() {
    let parsed: WeightQuantizationConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(parsed, WeightQuantizationConfig::default());
    assert!(
        serde_json::from_str::<WeightQuantizationConfig>(r#"{"scheme":"q4","group_size":16}"#)
            .is_err()
    );
    assert!(
        serde_json::from_str::<WeightQuantizationConfig>(
            r#"{"scheme":"q4","group_size":32,"surprise":true}"#
        )
        .is_err()
    );
}

#[test]
fn eligibility_policy_explains_every_exclusion_class() {
    let cases = [
        (
            "model.mtp.layers.0.q_proj.weight",
            &[8, 8][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::Mtp),
        ),
        (
            "next_token_head.attention.q_proj.weight",
            &[8, 8][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::Mtp),
        ),
        (
            "model.embed_tokens.weight",
            &[128, 8][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::Embedding),
        ),
        (
            "model.tok_embeddings.weight",
            &[128, 8][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::Embedding),
        ),
        (
            "shared.weight",
            &[128, 8][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::Embedding),
        ),
        (
            "model.layers.0.input_layernorm.weight",
            &[8][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::Normalization),
        ),
        (
            "model.layers.0.self_attn.q_proj.bias",
            &[8][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::Bias),
        ),
        (
            "model.layers.0.linear_attn.dt_bias",
            &[8][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::RecurrentScalar),
        ),
        (
            "model.layers.0.linear_attn.a_log",
            &[8][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::RecurrentScalar),
        ),
        (
            "model.layers.0.router.expert_indices",
            &[8, 8][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::Router),
        ),
        (
            "model.layers.0.scale.weight",
            &[8][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::ScalarOrVector),
        ),
        (
            "model.layers.0.linear_attn.conv1d.weight",
            &[32, 64][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::RecurrentConvolution),
        ),
        (
            "encoder.block.0.relative_attention_bias.weight",
            &[32, 8][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::BiasTable),
        ),
        (
            "model.layers.1.sparse_mlp.routed.router.weight",
            &[64, 2048][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::Router),
        ),
        (
            "model.layers.0.self_attn.kv_b_proj.weight",
            &[4096, 512][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::LatentExpansion),
        ),
        (
            "model.layers.0.experts.weight",
            &[8, 32, 64][..],
            QuantizationEligibility::Excluded(QuantizationExclusion::NonMatrix),
        ),
    ];
    for (name, shape, expected) in cases {
        assert_eq!(
            classify_weight_for_quantization(name, shape),
            expected,
            "{name}"
        );
    }

    for eligible in [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.experts.7.gate_proj.weight",
        "model.layers.1.moe.shared.up_proj.weight",
        "model.layers.1.moe.shared.down_proj.weight",
        "lm_head.weight",
    ] {
        assert_eq!(
            classify_weight_for_quantization(eligible, &[32, 64]),
            QuantizationEligibility::Eligible,
            "{eligible}"
        );
    }
}

#[test]
fn policy_is_applied_before_dtype_conversion() {
    let source = TensorData::new(vec![1_i8; 16], [4, 4]);
    let outcome = quantize_weight(
        "model.embed_tokens.weight",
        &source,
        WeightQuantizationConfig::default(),
    )
    .unwrap();
    assert_eq!(
        outcome,
        WeightQuantizationOutcome::Excluded(QuantizationExclusion::Embedding)
    );

    let eligible = TensorData::new(vec![1.0_f32; 32], [4, 8]);
    assert!(matches!(
        quantize_weight(
            "model.layers.0.q_proj.weight",
            &eligible,
            WeightQuantizationConfig::default()
        )
        .unwrap(),
        WeightQuantizationOutcome::Quantized(_)
    ));
}

#[test]
fn q8_fixture_has_exact_codes_scale_and_metadata() {
    let mut values = vec![0.0_f32; 32];
    values[..5].copy_from_slice(&[-127.0, -64.0, 0.0, 64.0, 127.0]);
    let quantized = quantize_tensor_data(
        &TensorData::new(values.clone(), [4, 8]),
        config(SymmetricQuantization::Q8, 32),
    )
    .unwrap();

    assert_eq!(
        &quantized.packed_values()[..5],
        &[0x81, 0xc0, 0, 0x40, 0x7f]
    );
    assert_eq!(quantized.scales(), &[1.0]);
    assert_eq!(quantized.metadata().shape, [4, 8]);
    assert_eq!(quantized.metadata().num_elements, 32);
    assert_eq!(quantized.metadata().group_count, 1);
    assert_eq!(quantized.metadata().packed_value_bytes, 32);
    assert_eq!(quantized.metadata().scale_bytes, 4);
    assert_eq!(f32_values(&quantized.dequantize_f32().unwrap()), values);
}

#[test]
fn q4_fixture_defines_nibble_order_and_twos_complement() {
    let mut values = vec![0.0_f32; 32];
    values[..5].copy_from_slice(&[-7.0, -1.0, 0.0, 1.0, 7.0]);
    let quantized = quantize_tensor_data(
        &TensorData::new(values.clone(), [4, 8]),
        config(SymmetricQuantization::Q4, 32),
    )
    .unwrap();

    assert_eq!(&quantized.packed_values()[..3], &[0xf9, 0x10, 0x07]);
    assert_eq!(quantized.packed_values().len(), 16);
    assert_eq!(quantized.scales(), &[1.0]);
    assert_eq!(quantized.metadata().packed_value_bytes, 16);
    assert_eq!(f32_values(&quantized.dequantize_f32().unwrap()), values);
}

#[test]
fn q4_odd_tail_uses_low_nibble_and_zeroes_unused_high_nibble() {
    let mut values = vec![0.0_f32; 33];
    values[32] = 7.0;
    let quantized = quantize_tensor_data(
        &TensorData::new(values, [3, 11]),
        config(SymmetricQuantization::Q4, 32),
    )
    .unwrap();
    assert_eq!(quantized.packed_values().len(), 17);
    assert_eq!(quantized.packed_values()[16], 0x07);
    assert_eq!(quantized.scales(), &[1.0, 1.0]);

    let (metadata, mut packed, scales) = quantized.into_parts();
    packed[16] |= 0x10;
    assert!(matches!(
        QuantizedWeight::from_parts(metadata, packed, scales),
        Err(QuantizationError::InvalidMetadata { .. })
    ));
}

#[test]
fn each_group_has_an_independent_scale_and_partial_final_group() {
    let mut values = vec![0.0_f32; 65];
    values[0] = 127.0;
    values[32] = 254.0;
    values[64] = 381.0;
    let quantized = quantize_tensor_data(
        &TensorData::new(values, [5, 13]),
        config(SymmetricQuantization::Q8, 32),
    )
    .unwrap();
    assert_eq!(quantized.scales(), &[1.0, 2.0, 3.0]);
    assert_eq!(quantized.metadata().group_count, 3);
    assert_eq!(quantized.stats().peak_group_elements, 32);
}

#[test]
fn zero_and_empty_tensors_have_canonical_representations() {
    let zeros = quantize_tensor_data(
        &TensorData::new(vec![0.0_f32; 32], [32]),
        config(SymmetricQuantization::Q4, 32),
    )
    .unwrap();
    assert_eq!(zeros.scales(), &[1.0]);
    assert!(zeros.packed_values().iter().all(|&byte| byte == 0));

    let empty = quantize_tensor_data(
        &TensorData::new(Vec::<f32>::new(), [0, 7]),
        WeightQuantizationConfig::default(),
    )
    .unwrap();
    assert!(empty.scales().is_empty());
    assert!(empty.packed_values().is_empty());
    assert_eq!(empty.metadata().num_elements, 0);
    assert_eq!(empty.stats().peak_group_elements, 0);
    assert_eq!(empty.dequantize_f32().unwrap().shape.as_slice(), &[0, 7]);
}

#[test]
fn smallest_subnormal_produces_a_positive_scale() {
    let mut values = vec![0.0_f32; 32];
    values[0] = f32::from_bits(1);
    let quantized = quantize_tensor_data(
        &TensorData::new(values, [32]),
        config(SymmetricQuantization::Q8, 32),
    )
    .unwrap();
    assert_eq!(quantized.scales(), &[f32::from_bits(1)]);
    assert_eq!(quantized.packed_values()[0], 1);
}

#[test]
fn source_float_dtypes_are_decoded_before_f32_math() {
    let f16_source = TensorData::new(vec![f16::from_f32(2.0); 32], [4, 8]);
    let bf16_source = TensorData::new(vec![bf16::from_f32(-2.0); 32], [4, 8]);
    let f64_source = TensorData::new(vec![3.0_f64; 32], [4, 8]);
    let flex_bytes: Vec<_> = vec![4.0_f32; 32]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
    let flex_source = TensorData::from_bytes_vec(flex_bytes, [4, 8], DType::Flex32);
    for (source, dtype, expected) in [
        (f16_source, SourceFloatDType::F16, 2.0),
        (bf16_source, SourceFloatDType::Bf16, -2.0),
        (f64_source, SourceFloatDType::F64, 3.0),
        (flex_source, SourceFloatDType::Flex32, 4.0),
    ] {
        let quantized =
            quantize_tensor_data(&source, config(SymmetricQuantization::Q8, 32)).unwrap();
        assert_eq!(quantized.metadata().source_dtype, dtype);
        assert_eq!(
            f32_values(&quantized.dequantize_f32().unwrap())[0],
            expected
        );
    }
}

#[test]
fn source_dtype_metadata_maps_every_supported_burn_storage_type() {
    let cases = [
        (SourceFloatDType::F16, DType::F16, 2),
        (SourceFloatDType::Bf16, DType::BF16, 2),
        (SourceFloatDType::F32, DType::F32, 4),
        (SourceFloatDType::Flex32, DType::Flex32, 4),
        (SourceFloatDType::F64, DType::F64, 8),
    ];
    for (source, burn, width) in cases {
        assert_eq!(source.burn_dtype(), burn);
        assert_eq!(source.byte_width(), width);
        assert_eq!(SourceFloatDType::try_from(burn).unwrap(), source);
    }
}

#[test]
fn malformed_sources_fail_without_panicking() {
    let unsupported = TensorData::new(vec![1_i8; 32], [4, 8]);
    assert!(matches!(
        quantize_tensor_data(&unsupported, WeightQuantizationConfig::default()),
        Err(QuantizationError::UnsupportedInputDType { dtype: DType::I8 })
    ));

    let malformed = TensorData::from_bytes_vec(vec![0; 7], [2], DType::F32);
    assert!(matches!(
        quantize_tensor_data(&malformed, WeightQuantizationConfig::default()),
        Err(QuantizationError::SourceByteLengthMismatch {
            expected: 8,
            actual: 7
        })
    ));

    let overflowing = TensorData::from_bytes_vec(Vec::new(), [usize::MAX, 2], DType::F32);
    assert!(matches!(
        quantize_tensor_data(&overflowing, WeightQuantizationConfig::default()),
        Err(QuantizationError::ShapeElementOverflow)
    ));

    let byte_overflow = TensorData::from_bytes_vec(Vec::new(), [usize::MAX], DType::F64);
    assert!(matches!(
        quantize_tensor_data(&byte_overflow, WeightQuantizationConfig::default()),
        Err(QuantizationError::ByteLengthOverflow {
            context: "source tensor byte length"
        })
    ));
}

#[test]
fn non_finite_values_and_f64_to_f32_overflow_are_rejected() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = vec![0.0_f32; 32];
        values[7] = bad;
        assert!(matches!(
            quantize_tensor_data(
                &TensorData::new(values, [4, 8]),
                WeightQuantizationConfig::default()
            ),
            Err(QuantizationError::NonFiniteValue { index: 7 })
        ));
    }

    let source = TensorData::new(vec![f64::MAX; 32], [4, 8]);
    assert!(matches!(
        quantize_tensor_data(&source, WeightQuantizationConfig::default()),
        Err(QuantizationError::NonFiniteValue { index: 0 })
    ));
}

#[test]
fn persisted_parts_validate_lengths_scales_and_reserved_codes() {
    let base = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 32], [4, 8]),
        config(SymmetricQuantization::Q8, 32),
    )
    .unwrap();
    let (metadata, packed, scales) = base.into_parts();

    assert!(matches!(
        QuantizedWeight::from_parts(metadata.clone(), vec![], scales.clone()),
        Err(QuantizationError::PackedLengthMismatch { .. })
    ));
    assert!(matches!(
        QuantizedWeight::from_parts(metadata.clone(), packed.clone(), vec![]),
        Err(QuantizationError::ScaleCountMismatch { .. })
    ));
    for bad_scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            QuantizedWeight::from_parts(metadata.clone(), packed.clone(), vec![bad_scale]),
            Err(QuantizationError::InvalidScale { group: 0, .. })
        ));
    }
    let mut reserved = packed;
    reserved[5] = 0x80;
    assert!(matches!(
        QuantizedWeight::from_parts(metadata, reserved, scales),
        Err(QuantizationError::InvalidQuantizedValue {
            index: 5,
            value: -128
        })
    ));

    let q4 = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 32], [4, 8]),
        config(SymmetricQuantization::Q4, 32),
    )
    .unwrap();
    let (metadata, mut packed, scales) = q4.into_parts();
    packed[0] = (packed[0] & 0xf0) | 0x08;
    assert!(matches!(
        QuantizedWeight::from_parts(metadata, packed, scales),
        Err(QuantizationError::InvalidQuantizedValue {
            index: 0,
            value: -8
        })
    ));
}

#[test]
fn digest_verified_constructor_skips_only_the_completed_integrity_check() {
    let weight = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 32], [4, 8]),
        config(SymmetricQuantization::Q8, 32),
    )
    .unwrap();
    let (mut metadata, packed, scales) = weight.into_parts();
    metadata.payload_sha256 = Some("0".repeat(64));

    assert!(matches!(
        QuantizedWeight::from_parts(metadata.clone(), packed.clone(), scales.clone()),
        Err(QuantizationError::PackedPayloadDigestMismatch { .. })
    ));

    let trusted = QuantizedWeight::from_payload_digest_verified_parts(
        metadata.clone(),
        packed.clone(),
        scales.clone(),
    )
    .unwrap();
    assert_eq!(trusted.packed_values(), packed);
    assert_eq!(trusted.scales(), scales);

    let mut invalid_codes = packed;
    invalid_codes[0] = 0x80;
    assert!(matches!(
        QuantizedWeight::from_payload_digest_verified_parts(metadata, invalid_codes, scales),
        Err(QuantizationError::InvalidQuantizedValue {
            index: 0,
            value: -128
        })
    ));
}

#[test]
fn metadata_redundancy_detects_tampering() {
    let mut metadata = QuantizedTensorMetadata::new(
        &[4, 8],
        SourceFloatDType::F32,
        config(SymmetricQuantization::Q4, 32),
    )
    .unwrap();
    metadata.num_elements = 31;
    assert!(matches!(
        metadata.validate(),
        Err(QuantizationError::InvalidMetadata { .. })
    ));

    metadata.num_elements = 32;
    metadata.group_count = 2;
    assert!(matches!(
        metadata.validate(),
        Err(QuantizationError::InvalidMetadata { .. })
    ));

    metadata.group_count = 1;
    metadata.packed_value_bytes = 15;
    assert!(matches!(
        metadata.validate(),
        Err(QuantizationError::InvalidMetadata { .. })
    ));

    metadata.packed_value_bytes = 16;
    metadata.scale_bytes = 8;
    assert!(matches!(
        metadata.validate(),
        Err(QuantizationError::InvalidMetadata { .. })
    ));

    metadata.scale_bytes = 4;
    metadata.packed_value_layout = PackedValueLayout::SignedI8TwosComplement;
    assert!(matches!(
        metadata.validate(),
        Err(QuantizationError::InvalidMetadata { .. })
    ));

    metadata.packed_value_layout = PackedValueLayout::SignedI4TwosComplementLowNibbleFirst;
    metadata.shape = vec![u64::MAX, 2];
    assert!(matches!(
        metadata.validate(),
        Err(QuantizationError::ShapeElementOverflow)
    ));
}

#[test]
fn manifest_json_is_deterministic_validated_and_sorted() {
    let q8 = QuantizedTensorMetadata::new(
        &[32, 64],
        SourceFloatDType::Bf16,
        config(SymmetricQuantization::Q8, 64),
    )
    .unwrap();
    let q4 = QuantizedTensorMetadata::new(
        &[64, 64],
        SourceFloatDType::F16,
        config(SymmetricQuantization::Q4, 128),
    )
    .unwrap();
    let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
    manifest.insert("z.weight", q8).unwrap();
    manifest.insert("a.weight", q4).unwrap();
    let json = manifest.to_json_pretty().unwrap();
    assert!(json.find("a.weight").unwrap() < json.find("z.weight").unwrap());
    assert_eq!(
        BurnpackQuantizationManifest::from_json(&json).unwrap(),
        manifest
    );
    assert_eq!(manifest.to_json_pretty().unwrap(), json);

    assert!(matches!(
        manifest.insert(
            "a.weight",
            QuantizedTensorMetadata::new(
                &[32, 64],
                SourceFloatDType::F32,
                WeightQuantizationConfig::default()
            )
            .unwrap()
        ),
        Err(QuantizationError::DuplicateManifestTensor { .. })
    ));
    assert!(matches!(
        manifest.insert(
            "",
            QuantizedTensorMetadata::new(
                &[32, 64],
                SourceFloatDType::F32,
                WeightQuantizationConfig::default()
            )
            .unwrap()
        ),
        Err(QuantizationError::EmptyTensorName)
    ));

    manifest.format_version = 99;
    assert!(matches!(
        manifest.validate(),
        Err(QuantizationError::UnsupportedFormatVersion { version: 99 })
    ));

    manifest.format_version = BURNPACK_QUANTIZATION_FORMAT_VERSION_V1;
    manifest.tensors.insert(
        String::new(),
        QuantizedTensorMetadata::new(
            &[32],
            SourceFloatDType::F32,
            WeightQuantizationConfig::default(),
        )
        .unwrap(),
    );
    assert!(matches!(
        manifest.validate(),
        Err(QuantizationError::EmptyTensorName)
    ));
}

#[test]
fn manifest_rejects_unknown_fields_and_invalid_derived_counts() {
    let metadata = QuantizedTensorMetadata::new(
        &[32],
        SourceFloatDType::F32,
        WeightQuantizationConfig::default(),
    )
    .unwrap();
    let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
    manifest.insert("x.weight", metadata).unwrap();
    let json = manifest.to_json_pretty().unwrap();
    let with_unknown = json.replacen('{', "{\"unknown\":true,", 1);
    assert!(BurnpackQuantizationManifest::from_json(&with_unknown).is_err());

    let invalid_count = json.replace("\"num_elements\": 32", "\"num_elements\": 31");
    assert!(matches!(
        BurnpackQuantizationManifest::from_json(&invalid_count),
        Err(QuantizationError::InvalidMetadata { .. })
    ));
}

/// Duplicate decoded tensor keys are ambiguous even when their payloads agree.
#[test]
fn manifest_duplicate_json_tensor_keys_are_rejected_before_map_collapse() {
    let metadata = QuantizedTensorMetadata::new(
        &[1, 64],
        SourceFloatDType::F32,
        WeightQuantizationConfig::default(),
    )
    .unwrap();
    let metadata = serde_json::to_string(&metadata).unwrap();
    for second_key in [r#""x.weight""#, r#""\u0078.weight""#] {
        let json = format!(
            r#"{{"format_version":1,"byte_order":"little_endian","packed_execution":"unavailable","tensors":{{"x.weight":{metadata},{second_key}:{metadata}}}}}"#
        );
        let error = BurnpackQuantizationManifest::from_json(&json).unwrap_err();
        assert!(matches!(error, QuantizationError::ManifestJson(_)));
        assert!(error.to_string().contains("duplicate map key `x.weight`"));
        let error = serde_json::from_str::<BurnpackQuantizationManifest>(&json).unwrap_err();
        assert!(error.to_string().contains("duplicate map key `x.weight`"));
    }
}

/// Checkpoint provenance cannot silently keep only the last duplicate shard.
#[test]
fn manifest_duplicate_json_provenance_keys_are_rejected_before_map_collapse() {
    let shard = serde_json::to_string(&BurnpackShardProvenance {
        file_bytes: 4096,
        sha256: "2".repeat(64),
    })
    .unwrap();
    for second_key in [r#""model.safetensors""#, r#""\u006dodel.safetensors""#] {
        let json = format!(
            r#"{{"load_plan_sha256":"{}","checkpoint_shards":{{"model.safetensors":{shard},{second_key}:{shard}}}}}"#,
            "1".repeat(64)
        );
        let error = serde_json::from_str::<BurnpackProvenance>(&json).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("duplicate map key `model.safetensors`")
        );
    }
}

/// Conflicting duplicates fail in either order, not only when values are equal.
#[test]
fn manifest_duplicate_conflicting_tensor_values_cannot_select_a_winner() {
    let q8 = serde_json::to_string(
        &QuantizedTensorMetadata::new(
            &[1, 64],
            SourceFloatDType::F32,
            config(SymmetricQuantization::Q8, 64),
        )
        .unwrap(),
    )
    .unwrap();
    let q4 = serde_json::to_string(
        &QuantizedTensorMetadata::new(
            &[2, 64],
            SourceFloatDType::F16,
            config(SymmetricQuantization::Q4, 64),
        )
        .unwrap(),
    )
    .unwrap();
    for (first, second) in [(&q8, &q4), (&q4, &q8)] {
        let json = format!(
            r#"{{"format_version":1,"byte_order":"little_endian","packed_execution":"unavailable","tensors":{{"x.weight":{first},"x.weight":{second}}}}}"#
        );
        assert!(
            BurnpackQuantizationManifest::from_json(&json)
                .unwrap_err()
                .to_string()
                .contains("duplicate map key `x.weight`")
        );
    }
    // Distinct decoded keys still produce a complete, deterministically sorted map.
    let json = format!(
        r#"{{"format_version":1,"byte_order":"little_endian","packed_execution":"unavailable","tensors":{{"z.weight":{q8},"a.weight":{q4}}}}}"#
    );
    let manifest = BurnpackQuantizationManifest::from_json(&json).unwrap();
    assert_eq!(manifest.tensors.len(), 2);
    assert_eq!(
        manifest.tensors["z.weight"].scheme,
        SymmetricQuantization::Q8
    );
    assert_eq!(
        manifest.tensors["a.weight"].scheme,
        SymmetricQuantization::Q4
    );
    assert!(
        manifest.to_json_pretty().unwrap().find("a.weight").unwrap()
            < manifest.to_json_pretty().unwrap().find("z.weight").unwrap()
    );
}

#[test]
fn manifest_v2_requires_provenance_while_v1_remains_compatible() {
    let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
    manifest
        .insert(
            "x.weight",
            QuantizedTensorMetadata::new(
                &[32, 32],
                SourceFloatDType::F32,
                WeightQuantizationConfig::default(),
            )
            .unwrap(),
        )
        .unwrap();
    let v1_json = manifest.to_json_pretty().unwrap();
    assert!(!v1_json.contains("provenance"));
    assert!(!v1_json.contains("payload_sha256"));
    assert_eq!(
        BurnpackQuantizationManifest::from_json(&v1_json).unwrap(),
        manifest
    );

    manifest.format_version = BURNPACK_QUANTIZATION_FORMAT_VERSION_V2;
    assert!(matches!(
        manifest.validate(),
        Err(QuantizationError::InvalidProvenance { .. })
    ));

    manifest.format_version = BURNPACK_QUANTIZATION_FORMAT_VERSION_V1;
    manifest.tensors.get_mut("x.weight").unwrap().payload_sha256 = Some("3".repeat(64));
    manifest
        .bind_provenance(BurnpackProvenance {
            load_plan_sha256: "1".repeat(64),
            checkpoint_shards: BTreeMap::from([(
                "model.safetensors".into(),
                BurnpackShardProvenance {
                    file_bytes: 4096,
                    sha256: "2".repeat(64),
                },
            )]),
        })
        .unwrap();
    assert_eq!(
        manifest.format_version,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    let v2_json = manifest.to_json_pretty().unwrap();
    assert_eq!(
        BurnpackQuantizationManifest::from_json(&v2_json).unwrap(),
        manifest
    );

    let mut missing_payload_digest = manifest.clone();
    missing_payload_digest
        .tensors
        .get_mut("x.weight")
        .unwrap()
        .payload_sha256 = None;
    assert!(matches!(
        missing_payload_digest.validate(),
        Err(QuantizationError::InvalidProvenance { .. })
    ));

    manifest.format_version = BURNPACK_QUANTIZATION_FORMAT_VERSION_V1;
    assert!(matches!(
        manifest.validate(),
        Err(QuantizationError::InvalidProvenance { .. })
    ));
}

/// Historical constructors and the unsuffixed version constant retain v1 semantics.
#[test]
fn compatibility_manifest_api_remains_v1_compatible_without_changing_current_v2() {
    assert_eq!(
        BURNPACK_QUANTIZATION_FORMAT_VERSION,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V1
    );
    assert_eq!(
        BURNPACK_QUANTIZATION_FORMAT_VERSION_CURRENT,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    assert_ne!(
        BURNPACK_QUANTIZATION_FORMAT_VERSION,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_CURRENT
    );

    let legacy = BurnpackQuantizationManifest::new();
    assert_eq!(legacy, BurnpackQuantizationManifest::new_legacy_v1());
    assert_eq!(legacy, BurnpackQuantizationManifest::default());
    assert_eq!(
        legacy.format_version,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V1
    );
    assert!(legacy.provenance.is_none());
    assert!(legacy.tensors.is_empty());

    let json = legacy.to_json_pretty().unwrap();
    assert!(!json.contains("provenance"));
    assert!(!json.contains("payload_sha256"));
    assert_eq!(
        BurnpackQuantizationManifest::from_json(&json).unwrap(),
        legacy
    );
}

#[test]
fn payload_writer_uses_packed_then_little_endian_scales() {
    let mut values = vec![0.0_f32; 32];
    values[0] = 127.0;
    let quantized = quantize_tensor_data(
        &TensorData::new(values, [4, 8]),
        config(SymmetricQuantization::Q8, 32),
    )
    .unwrap();
    let mut payload = Vec::new();
    quantized.write_burnpack_payload(&mut payload).unwrap();
    assert_eq!(&payload[..32], quantized.packed_values());
    assert_eq!(&payload[32..], &1.0_f32.to_le_bytes());
}

#[test]
fn payload_writer_preserves_context_for_io_failures() {
    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("fixture failure"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let quantized = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 32], [4, 8]),
        WeightQuantizationConfig::default(),
    )
    .unwrap();
    assert!(matches!(
        quantized.write_burnpack_payload(&mut FailingWriter),
        Err(QuantizationError::PayloadIo(_))
    ));
}

#[test]
fn compatible_q4_and_q8_matrices_map_exactly_to_burn_qfloat() {
    let device = Default::default();
    let values = (0..128)
        .map(|index| (index as f32 - 63.5) / 7.0)
        .collect::<Vec<_>>();
    for (precision, burn_value) in [
        (SymmetricQuantization::Q8, QuantValue::Q8S),
        (SymmetricQuantization::Q4, QuantValue::Q4S),
    ] {
        let quantized = quantize_tensor_data(
            &TensorData::new(values.clone(), [2, 64]),
            config(precision, 32),
        )
        .unwrap();
        let expected = f32_values(&quantized.dequantize_f32().unwrap());
        let data = quantized.into_burn_qfloat_data().unwrap();
        let DType::QFloat(scheme) = data.dtype else {
            panic!("adapter must produce QFloat data")
        };
        assert_eq!(scheme.value, burn_value);
        assert_eq!(scheme.store, QuantStore::PackedU32(0));
        assert_eq!(scheme.level, QuantLevel::block([1, 32]));

        let actual = Tensor::<Flex, 2>::from_data(data, &device)
            .dequantize()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn burn_qfloat_adapter_rejects_rank_and_cross_row_groups() {
    let rank_one = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 64], [64]),
        config(SymmetricQuantization::Q8, 32),
    )
    .unwrap();
    assert!(matches!(
        rank_one.into_burn_qfloat_data(),
        Err(QuantizationError::BurnQFloatIncompatible { .. })
    ));

    let crossing = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 96], [2, 48]),
        config(SymmetricQuantization::Q4, 32),
    )
    .unwrap();
    assert!(matches!(
        crossing.into_burn_qfloat_data(),
        Err(QuantizationError::BurnQFloatIncompatible { .. })
    ));
}

#[test]
fn packed_backend_execution_is_explicitly_unavailable() {
    let support = PackedExecutionSupport::Unavailable;
    assert!(!support.is_available());
    assert!(matches!(
        support.require_available(),
        Err(QuantizationError::PackedExecutionUnavailable)
    ));
    assert_eq!(
        BurnpackQuantizationManifest::new_legacy_v1().packed_execution,
        support
    );
}

/// Requires both the typed manifest error and its complete contextual fields.
#[track_caller]
fn assert_manifest_contract_error(actual: QuantizationError, expected: QuantizationError) {
    assert_eq!(
        core::mem::discriminant(&actual),
        core::mem::discriminant(&expected)
    );
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

/// Supplies valid, compact metadata independently of any packed payload allocation.
fn manifest_contract_metadata() -> QuantizedTensorMetadata {
    QuantizedTensorMetadata::new(
        &[2, 64],
        SourceFloatDType::F32,
        config(SymmetricQuantization::Q4, 64),
    )
    .unwrap()
}

/// Supplies syntactically valid provenance for transactional manifest tests.
fn manifest_contract_provenance() -> BurnpackProvenance {
    BurnpackProvenance {
        load_plan_sha256: "1".repeat(64),
        checkpoint_shards: BTreeMap::from([(
            "weights.safetensors".into(),
            BurnpackShardProvenance {
                file_bytes: 256,
                sha256: "2".repeat(64),
            },
        )]),
    }
}

/// V1 cannot advertise per-payload v2 digests, even without top-level provenance.
#[test]
fn manifest_contract_legacy_payload_digest_is_rejected_at_every_wire_boundary() {
    let mut metadata = manifest_contract_metadata();
    metadata.payload_sha256 = Some("3".repeat(64));
    let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
    // Insertion intentionally permits staging a digest before atomic v2 binding.
    manifest.insert("projection.weight", metadata).unwrap();
    let wire = serde_json::to_string(&manifest).unwrap();
    for error in [
        manifest.validate().unwrap_err(),
        manifest.to_json_pretty().unwrap_err(),
        BurnpackQuantizationManifest::from_json(&wire).unwrap_err(),
    ] {
        assert_manifest_contract_error(error, QuantizationError::InvalidProvenance {
            reason: "format version one tensor `projection.weight` must not claim a v2 payload digest".into(),
        });
    }
    assert_eq!(
        manifest.format_version,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V1
    );
    assert!(manifest.provenance.is_none());
}

/// Failed v2 binding preserves every field of the original legacy or v2 manifest.
#[test]
fn manifest_contract_provenance_binding_is_transactional_for_partial_digests() {
    let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
    let mut complete = manifest_contract_metadata();
    complete.payload_sha256 = Some("3".repeat(64));
    manifest.insert("a.weight", complete).unwrap();
    manifest
        .insert("z.weight", manifest_contract_metadata())
        .unwrap();
    let before = manifest.clone();
    assert_manifest_contract_error(
        manifest
            .bind_provenance(manifest_contract_provenance())
            .unwrap_err(),
        QuantizationError::InvalidProvenance {
            reason: "format version two tensor `z.weight` is missing its payload SHA-256".into(),
        },
    );
    assert_eq!(manifest, before);

    manifest.tensors.get_mut("z.weight").unwrap().payload_sha256 = Some("4".repeat(64));
    manifest
        .bind_provenance(manifest_contract_provenance())
        .unwrap();
    assert_eq!(
        manifest.format_version,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    assert_eq!(manifest.provenance, Some(manifest_contract_provenance()));
    let bound = manifest.clone();
    let mut invalid = manifest_contract_provenance();
    invalid.load_plan_sha256 = "INVALID".into();
    assert_manifest_contract_error(
        manifest.bind_provenance(invalid).unwrap_err(),
        QuantizationError::InvalidProvenance {
            reason: "load plan SHA-256 must be 64 lowercase hexadecimal characters".into(),
        },
    );
    assert_eq!(manifest, bound);
    assert_eq!(
        BurnpackQuantizationManifest::from_json(&manifest.to_json_pretty().unwrap()).unwrap(),
        bound
    );
}

/// Insertion validates names before metadata, then metadata before duplicate keys.
#[test]
fn manifest_contract_failed_insertions_preserve_existing_metadata() {
    let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
    manifest
        .insert("weight", manifest_contract_metadata())
        .unwrap();
    let before = manifest.clone();
    let mut invalid = manifest_contract_metadata();
    invalid.num_elements = 127;
    assert_manifest_contract_error(
        manifest.insert("", invalid.clone()).unwrap_err(),
        QuantizationError::EmptyTensorName,
    );
    assert_eq!(manifest, before);
    assert_manifest_contract_error(
        manifest.insert("weight", invalid).unwrap_err(),
        QuantizationError::InvalidMetadata {
            reason: "num_elements 127 does not match shape product 128".into(),
        },
    );
    assert_eq!(manifest, before);
    assert_manifest_contract_error(
        manifest
            .insert("weight", manifest_contract_metadata())
            .unwrap_err(),
        QuantizationError::DuplicateManifestTensor {
            name: "weight".into(),
        },
    );
    assert_eq!(manifest, before);
}

/// Wire readers report schema/provenance failures before examining tensor entries.
#[test]
fn manifest_contract_wire_validation_order_is_explicit() {
    let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
    let mut metadata = manifest_contract_metadata();
    metadata.num_elements = 127;
    manifest.tensors.insert(String::new(), metadata);
    for (version, error) in [
        (
            99,
            QuantizationError::UnsupportedFormatVersion { version: 99 },
        ),
        (
            2,
            QuantizationError::InvalidProvenance {
                reason: "format version two requires checkpoint and load-plan provenance".into(),
            },
        ),
        (1, QuantizationError::EmptyTensorName),
    ] {
        manifest.format_version = version;
        let wire = serde_json::to_string(&manifest).unwrap();
        assert_manifest_contract_error(
            BurnpackQuantizationManifest::from_json(&wire).unwrap_err(),
            error,
        );
    }
    let metadata = manifest.tensors.remove("").unwrap();
    manifest.tensors.insert("weight".into(), metadata);
    let wire = serde_json::to_string(&manifest).unwrap();
    assert_manifest_contract_error(
        BurnpackQuantizationManifest::from_json(&wire).unwrap_err(),
        QuantizationError::InvalidMetadata {
            reason: "num_elements 127 does not match shape product 128".into(),
        },
    );
}

/// Packed-layout checks first authenticate redundant counts, then rank and rows.
#[test]
fn manifest_contract_packed_layout_checks_metadata_before_backend_geometry() {
    let mut metadata = QuantizedTensorMetadata::new(
        &[2, 32],
        SourceFloatDType::F32,
        config(SymmetricQuantization::Q4, 64),
    )
    .unwrap();
    metadata.group_count = 0;
    assert_manifest_contract_error(
        metadata.validate_burn_qfloat_layout().unwrap_err(),
        QuantizationError::InvalidMetadata {
            reason: "group_count 0 does not match expected 1".into(),
        },
    );
    metadata.group_count = 1;
    assert_manifest_contract_error(
        metadata.validate_burn_qfloat_layout().unwrap_err(),
        QuantizationError::BurnQFloatIncompatible {
            reason: "innermost dimension 32 is not divisible by group size 64".into(),
        },
    );
    metadata.shape = vec![1, 2, 32];
    assert_manifest_contract_error(
        metadata.validate_burn_qfloat_layout().unwrap_err(),
        QuantizationError::BurnQFloatIncompatible {
            reason: "expected a rank-two matrix, found shape [1, 2, 32]".into(),
        },
    );
    for scheme in [SymmetricQuantization::Q4, SymmetricQuantization::Q8] {
        for group in [32, 64, 128] {
            for shape in [[0, group], [2, 0], [3, group * 2]] {
                let metadata = QuantizedTensorMetadata::new(
                    &shape,
                    SourceFloatDType::F32,
                    config(scheme, group),
                )
                .unwrap();
                metadata.validate_burn_qfloat_layout().unwrap();
                assert_eq!(metadata.host_shape().unwrap(), shape);
                let elements = shape[0] * shape[1];
                assert_eq!(
                    metadata.host_counts().unwrap(),
                    (
                        elements,
                        elements / group,
                        if scheme == SymmetricQuantization::Q4 {
                            elements / 2
                        } else {
                            elements
                        }
                    )
                );
                assert_eq!(
                    serde_json::from_str::<QuantizedTensorMetadata>(
                        &serde_json::to_string(&metadata).unwrap()
                    )
                    .unwrap(),
                    metadata
                );
            }
        }
    }
}

proptest! {
    #[test]
    fn finite_inputs_round_trip_within_half_a_quantization_step(
        values in prop::collection::vec(-10_000.0_f32..10_000.0_f32, 1..520),
        q4 in any::<bool>(),
        group_choice in 0_usize..3,
    ) {
        let scheme = if q4 {
            SymmetricQuantization::Q4
        } else {
            SymmetricQuantization::Q8
        };
        let group_size = [32, 64, 128][group_choice];
        let source = TensorData::new(values.clone(), [values.len()]);
        let first = quantize_tensor_data(&source, config(scheme, group_size)).unwrap();
        let second = quantize_tensor_data(&source, config(scheme, group_size)).unwrap();
        prop_assert_eq!(first.packed_values(), second.packed_values());
        prop_assert_eq!(first.scales(), second.scales());
        prop_assert_eq!(first.metadata(), second.metadata());

        let restored = f32_values(&first.dequantize_f32().unwrap());
        for (index, (&original, &actual)) in values.iter().zip(&restored).enumerate() {
            let scale = first.scales()[index / group_size];
            let tolerance = scale * 0.500_01 + f32::EPSILON * original.abs().max(1.0);
            prop_assert!((original - actual).abs() <= tolerance,
                "index={index}, original={original}, actual={actual}, scale={scale}");
        }
        prop_assert!(first.stats().peak_group_elements <= group_size);
        prop_assert_eq!(first.stats().source_elements, values.len());
        prop_assert_eq!(first.stats().group_count, values.len() / group_size + usize::from(values.len() % group_size != 0));
        prop_assert_eq!(first.metadata().packed_value_bytes as usize,
            if q4 { values.len() / 2 + values.len() % 2 } else { values.len() });
    }
}
