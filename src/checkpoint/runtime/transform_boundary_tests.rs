//! Test-only fixtures for transform.rs; excluded from non-test compiler inputs.

use super::*;
use crate::checkpoint::{LoadReport, MemoryEstimate};

/// Supplies immutable synthetic payloads while counting actual materializations.
struct FixtureSource {
    payloads: Vec<SourceTensor>,
    reads: usize,
}

impl TensorSource for FixtureSource {
    /// Clones the exact requested wire payload without performing conversion.
    fn read_tensor(&mut self, planned: &PlannedSource) -> Result<SourceTensor> {
        self.reads += 1;
        self.payloads
            .iter()
            .find(|payload| payload.name() == planned.name)
            .cloned()
            .ok_or_else(|| CheckpointError::InvalidTensorPayload {
                tensor: planned.name.clone(),
                reason: "fixture payload absent".into(),
            })
    }
}

/// Describes one standalone tensor with a precisely sized byte interval.
fn descriptor(name: &str, shape: &[u64], dtype: TensorDType) -> PlannedSource {
    PlannedSource {
        name: name.into(),
        shard: format!("{name}.safetensors"),
        shape: shape.to_vec(),
        dtype,
        data_offsets: [0, dtype.storage_bytes(shape.iter().product()).unwrap()],
    }
}

/// Creates one identity operation that individual tests mutate deliberately.
fn identity() -> PlannedTensor {
    PlannedTensor {
        target: "target".into(),
        sources: vec![descriptor("weight", &[2, 3], TensorDType::F32)],
        transform: PlannedTransform::Identity,
        shape: vec![2, 3],
        dtype: TensorDType::F32,
    }
}

/// Wraps a primary operation without hiding validation behind fixture setup.
fn plan(tensor: PlannedTensor) -> LoadPlan {
    LoadPlan {
        tensors: vec![tensor],
        aliases: Vec::new(),
        equivalences: Vec::new(),
        report: LoadReport::default(),
        memory: MemoryEstimate {
            peak_host_staging_bytes: 1024,
            ..MemoryEstimate::default()
        },
    }
}

/// Requires a stable contextual message, not merely any unrelated failure.
fn assert_mapping_error(plan: &LoadPlan, expected: &str) {
    let error = validate_plan(plan).unwrap_err();
    assert!(
        matches!(error, CheckpointError::InvalidMapping { ref message }
        if message.contains(expected)),
        "expected {expected:?}, got {error}"
    );
}

/// Rejects every malformed descriptor before layout processing can trust it.
#[test]
fn public_source_descriptors_require_names_extents_and_supported_dtypes() {
    for (case, message) in [
        ("name", "empty tensor name or shard"),
        ("shard", "empty tensor name or shard"),
        ("reversed", "reversed data offsets"),
        ("length", "offsets describe 23 bytes, expected 24"),
    ] {
        let mut tensor = identity();
        let source = &mut tensor.sources[0];
        match case {
            "name" => source.name.clear(),
            "shard" => source.shard.clear(),
            "reversed" => source.data_offsets = [24, 0],
            "length" => source.data_offsets[1] = 23,
            _ => unreachable!(),
        }
        assert_mapping_error(&plan(tensor), message);
    }
    for dtype in [
        TensorDType::F4,
        TensorDType::F6_E2M3,
        TensorDType::F6_E3M2,
        TensorDType::F8_E5M2,
        TensorDType::F8_E4M3,
        TensorDType::F8_E8M0,
        TensorDType::C64,
    ] {
        let mut tensor = identity();
        tensor.sources[0].dtype = dtype;
        assert!(matches!(validate_plan(&plan(tensor)),
            Err(CheckpointError::UnsupportedRuntimeDType { dtype: actual }) if actual == dtype));
    }
    let mut tensor = identity();
    tensor.sources[0].shape = vec![u64::MAX, 2];
    assert!(
        matches!(validate_plan(&plan(tensor)), Err(CheckpointError::InvalidTensorPayload { tensor, reason })
        if tensor == "weight" && reason.contains("shape element count overflows"))
    );
}

/// Keeps primary uniqueness, unary arity, and matrix orientation independent.
#[test]
fn primary_operations_reject_duplicate_targets_and_invalid_unary_contracts() {
    for (case, message) in [
        ("duplicate", "duplicate runtime target `target`"),
        ("no_source", "requires exactly one source, found 0"),
        ("two_sources", "requires exactly one source, found 2"),
        ("identity_shape", "identity target `target` shape differs"),
        ("transpose_rank", "does not reverse a rank-two source shape"),
        (
            "transpose_shape",
            "does not reverse a rank-two source shape",
        ),
        (
            "alias",
            "alias `target` occurred in the primary tensor list",
        ),
    ] {
        let mut plan = plan(identity());
        let tensor = &mut plan.tensors[0];
        match case {
            "no_source" => tensor.sources.clear(),
            "two_sources" => tensor.sources.push(tensor.sources[0].clone()),
            "identity_shape" => tensor.shape = vec![6],
            "transpose_rank" => {
                tensor.transform = PlannedTransform::Transpose2d;
                tensor.sources[0].shape = vec![6];
            }
            "transpose_shape" => tensor.transform = PlannedTransform::Transpose2d,
            "alias" => {
                tensor.transform = PlannedTransform::Alias {
                    source_target: "owner".into(),
                }
            }
            "duplicate" => plan.tensors.push(plan.tensors[0].clone()),
            _ => unreachable!(),
        }
        assert_mapping_error(&plan, message);
    }
}

/// Requires aliases to share a primary owner's exact logical storage contract.
#[test]
fn public_alias_entries_cannot_smuggle_sources_or_incompatible_owners() {
    for (case, message) in [
        ("empty", "alias target name cannot be empty"),
        ("duplicate", "duplicate runtime target `target`"),
        ("payload", "unexpectedly contains source payloads"),
        ("operation", "entry `alias` in alias list is not an alias"),
        ("owner", "must reference a storage-owning primary target"),
        ("shape", "metadata differs from storage owner"),
        ("dtype", "metadata differs from storage owner"),
    ] {
        let mut plan = plan(identity());
        let mut alias = plan.tensors[0].clone();
        alias.target = "alias".into();
        alias.sources.clear();
        alias.transform = PlannedTransform::Alias {
            source_target: "target".into(),
        };
        match case {
            "empty" => alias.target.clear(),
            "duplicate" => alias.target = "target".into(),
            "payload" => alias.sources = plan.tensors[0].sources.clone(),
            "operation" => alias.transform = PlannedTransform::Identity,
            "owner" => {
                alias.transform = PlannedTransform::Alias {
                    source_target: "absent".into(),
                }
            }
            "shape" => alias.shape = vec![6],
            "dtype" => alias.dtype = TensorDType::F16,
            _ => unreachable!(),
        }
        plan.aliases.push(alias);
        assert_mapping_error(&plan, message);
    }
}

/// Validates slice intervals using checked addition and the full output shape.
#[test]
fn slice_contract_rejects_axis_extent_shape_and_end_overflow() {
    for (axis, start, length, shape, message) in [
        (2, 0, 1, vec![2, 1], "slice axis 2 is out of bounds"),
        (1, 2, 2, vec![2, 2], "ends at 4, source axis has length 3"),
        (1, 0, 1, vec![1, 2], "shape [1, 2] should be [2, 1]"),
    ] {
        let mut tensor = identity();
        tensor.transform = PlannedTransform::Slice {
            axis,
            start,
            length,
        };
        tensor.shape = shape;
        assert_mapping_error(&plan(tensor), message);
    }
    let mut tensor = identity();
    tensor.transform = PlannedTransform::Slice {
        axis: 1,
        start: u64::MAX,
        length: 1,
    };
    tensor.shape = vec![2, 1];
    assert!(matches!(
        validate_plan(&plan(tensor)),
        Err(CheckpointError::ArithmeticOverflow {
            context: "computing runtime slice end"
        })
    ));
}

/// Requires concatenation to conserve rank, non-axis dimensions, and dtype.
#[test]
fn concatenation_contract_rejects_each_distinct_shape_violation() {
    for (case, message) in [
        ("empty", "concatenation `target` has no sources"),
        ("axis", "concatenation axis 2 is out of bounds"),
        ("dtype", "mixes source dtypes"),
        ("rank", "source `second` has incompatible shape"),
        ("non_axis", "source `second` has incompatible shape"),
        ("target", "shape [6] should be [2, 3]"),
    ] {
        let mut tensor = identity();
        tensor.transform = PlannedTransform::Concatenate { axis: 1 };
        tensor.sources = vec![
            descriptor("first", &[2, 2], TensorDType::F32),
            descriptor("second", &[2, 1], TensorDType::F32),
        ];
        match case {
            "empty" => tensor.sources.clear(),
            "axis" => tensor.transform = PlannedTransform::Concatenate { axis: 2 },
            "dtype" => tensor.sources[1] = descriptor("second", &[2, 1], TensorDType::F16),
            "rank" => tensor.sources[1] = descriptor("second", &[2, 1, 1], TensorDType::F32),
            "non_axis" => tensor.sources[1] = descriptor("second", &[3, 1], TensorDType::F32),
            "target" => tensor.shape = vec![6],
            _ => unreachable!(),
        }
        assert_mapping_error(&plan(tensor), message);
    }

    // Empty leading axes make each payload legitimately zero bytes, while
    // their declared concatenation dimensions must still add without wrap.
    let mut tensor = identity();
    tensor.sources = vec![
        descriptor("first", &[0, u64::MAX], TensorDType::F32),
        descriptor("second", &[0, 1], TensorDType::F32),
    ];
    tensor.shape = vec![0, 0];
    tensor.transform = PlannedTransform::Concatenate { axis: 1 };
    assert!(matches!(
        validate_plan(&plan(tensor)),
        Err(CheckpointError::ArithmeticOverflow {
            context: "summing runtime concatenation dimensions"
        })
    ));
}

/// Separates equal-byte-count aliases from actual shape and identity equality.
#[test]
fn physical_equivalence_rejects_self_alias_and_equal_size_shape_changes() {
    for self_alias in [false, true] {
        let mut plan = plan(identity());
        let owner = plan.tensors[0].sources[0].clone();
        let alias = if self_alias {
            owner.clone()
        } else {
            descriptor("copy", &[3, 2], TensorDType::F32)
        };
        plan.equivalences
            .push(PlannedTensorEquivalence { alias, owner });
        let error = validate_plan(&plan).unwrap_err();
        assert!(
            matches!(error, CheckpointError::TensorEquivalenceMismatch { alias, owner, reason }
            if owner == "weight" && alias == if self_alias { "weight" } else { "copy" }
            && reason == if self_alias { "alias and owner names must differ" } else { "alias and owner shapes differ" })
        );
    }
}

/// Encodes exactly representable fixture values in a chosen source byte order.
fn encode(values: &[f64], dtype: TensorDType, little_endian: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    for &value in values {
        let mut scalar = match dtype {
            TensorDType::F16 => f16::from_f64(value).to_bits().to_ne_bytes().to_vec(),
            TensorDType::BF16 => bf16::from_f64(value).to_bits().to_ne_bytes().to_vec(),
            TensorDType::F32 => (value as f32).to_ne_bytes().to_vec(),
            TensorDType::F64 => value.to_ne_bytes().to_vec(),
            _ => unreachable!(),
        };
        if little_endian && cfg!(target_endian = "big") {
            scalar.reverse();
        }
        bytes.extend(scalar);
    }
    bytes
}

/// Builds a wire payload independently of the production conversion helper.
fn float_payload(name: &str, shape: &[u64], dtype: TensorDType, values: &[f64]) -> SourceTensor {
    SourceTensor::new(name, shape.to_vec(), dtype, encode(values, dtype, true)).unwrap()
}

/// Checks every conventional float cast through four distinct layout paths.
#[test]
fn all_real_dtype_pairs_preserve_row_major_layout_and_signed_zero() {
    let dtypes = [
        TensorDType::F16,
        TensorDType::BF16,
        TensorDType::F32,
        TensorDType::F64,
    ];
    let values = [-2.0, -0.0, 0.0, 1.5, 7.0, f64::INFINITY];
    for source_dtype in dtypes {
        for target_dtype in dtypes {
            for operation in ["identity", "transpose", "slice", "concat"] {
                let mut tensor = identity();
                tensor.sources[0] = descriptor("weight", &[2, 3], source_dtype);
                tensor.dtype = target_dtype;
                let mut payloads = vec![float_payload("weight", &[2, 3], source_dtype, &values)];
                let expected = match operation {
                    "identity" => values.to_vec(),
                    "transpose" => {
                        tensor.transform = PlannedTransform::Transpose2d;
                        tensor.shape = vec![3, 2];
                        vec![-2.0, 1.5, -0.0, 7.0, 0.0, f64::INFINITY]
                    }
                    "slice" => {
                        tensor.transform = PlannedTransform::Slice {
                            axis: 1,
                            start: 1,
                            length: 1,
                        };
                        tensor.shape = vec![2, 1];
                        vec![-0.0, 7.0]
                    }
                    "concat" => {
                        tensor.transform = PlannedTransform::Concatenate { axis: 1 };
                        tensor.shape = vec![2, 4];
                        tensor
                            .sources
                            .push(descriptor("second", &[2, 1], source_dtype));
                        payloads.push(float_payload(
                            "second",
                            &[2, 1],
                            source_dtype,
                            &[100.0, 200.0],
                        ));
                        vec![-2.0, -0.0, 0.0, 100.0, 1.5, 7.0, f64::INFINITY, 200.0]
                    }
                    _ => unreachable!(),
                };
                validate_plan(&plan(tensor.clone())).unwrap();
                let mut source = FixtureSource { payloads, reads: 0 };
                let mut stats = ExecutionStats::default();
                let data = materialize(&tensor, &mut source, &mut stats).unwrap();
                assert_eq!(
                    data.as_bytes(),
                    encode(&expected, target_dtype, false),
                    "{source_dtype:?}->{target_dtype:?} {operation}"
                );
                assert_eq!(
                    data.shape.as_slice(),
                    tensor.shape.iter().map(|&n| n as usize).collect::<Vec<_>>()
                );
                assert_eq!(data.dtype, burn_dtype(target_dtype).unwrap());
                assert_eq!(source.reads, if operation == "concat" { 2 } else { 1 });
                assert_eq!(stats.source_reads as usize, source.reads);
            }
        }
    }
}

/// Validates shared source identity and proves reuse agrees with a fresh read.
#[test]
fn shared_slices_require_matching_payload_descriptors_and_slice_operations() {
    let mut tensor = identity();
    let payload = float_payload("weight", &[2, 3], TensorDType::F32, &[0.0; 6]);
    assert!(
        matches!(materialize_shared_slice(&tensor, &payload, &mut ExecutionStats::default()),
        Err(CheckpointError::InvalidMapping { message }) if message.contains("non-slice target `target`"))
    );
    tensor.transform = PlannedTransform::Slice {
        axis: 1,
        start: 1,
        length: 2,
    };
    tensor.shape = vec![2, 2];
    for payload in [
        float_payload("other", &[2, 3], TensorDType::F32, &[0.0; 6]),
        float_payload("weight", &[3, 2], TensorDType::F32, &[0.0; 6]),
        float_payload("weight", &[2, 3], TensorDType::F64, &[0.0; 6]),
    ] {
        assert!(
            matches!(materialize_shared_slice(&tensor, &payload, &mut ExecutionStats::default()),
            Err(CheckpointError::InvalidTensorPayload { tensor, reason })
            if tensor == "weight" && reason.contains("source returned"))
        );
    }
    let payload = float_payload(
        "weight",
        &[2, 3],
        TensorDType::F32,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let mut shared_stats = ExecutionStats::default();
    let shared = materialize_shared_slice(&tensor, &payload, &mut shared_stats).unwrap();
    let mut source = FixtureSource {
        payloads: vec![payload],
        reads: 0,
    };
    let mut read_stats = ExecutionStats::default();
    let fresh = materialize(&tensor, &mut source, &mut read_stats).unwrap();
    assert_eq!(shared, fresh);
    assert_eq!(shared.to_vec::<f32>().unwrap(), vec![2.0, 3.0, 5.0, 6.0]);
    assert_eq!(shared_stats.source_reads, 0);
    assert_eq!(read_stats.source_reads, 1);
    assert_eq!(shared_stats.peak_host_staging_bytes, 40);
    assert_eq!(read_stats.peak_host_staging_bytes, 40);
}

/// Rejects truncated contiguous spans before any destination byte is changed.
#[test]
fn copy_ranges_fail_contextually_for_equal_dtype_and_cast_paths() {
    for target_dtype in [TensorDType::F32, TensorDType::F16] {
        for source_short in [false, true] {
            let source = if source_short { vec![0; 3] } else { vec![0; 4] };
            let width = scalar_bytes(target_dtype).unwrap() as usize;
            let mut target = vec![0xa5; if source_short { width } else { width - 1 }];
            let before = target.clone();
            assert!(
                matches!(copy_elements("bounded", &source, TensorDType::F32, 0,
                &mut target, target_dtype, 0, 1), Err(CheckpointError::InvalidTensorPayload { tensor, reason })
                if tensor == "bounded" && reason.contains("outside the payload"))
            );
            assert_eq!(target, before);
        }
    }
    let mut target = [0xa5; 4];
    copy_elements(
        "empty",
        &[],
        TensorDType::F32,
        0,
        &mut target,
        TensorDType::F32,
        1,
        0,
    )
    .unwrap();
    assert_eq!(target, [0xa5; 4]);
}

/// Proves each fixed-width decoder and target encoder checks exact slot sizes.
#[test]
fn scalar_conversion_rejects_truncated_slots_and_non_real_dtypes() {
    for (dtype, bytes) in [
        (TensorDType::F16, 1),
        (TensorDType::BF16, 3),
        (TensorDType::F32, 3),
        (TensorDType::F64, 7),
    ] {
        assert!(
            matches!(convert_float_scalar("scalar", dtype, &vec![0; bytes],
            TensorDType::F32, &mut [0; 4]), Err(CheckpointError::InvalidTensorPayload { tensor, .. }) if tensor == "scalar")
        );
    }
    assert!(matches!(
        convert_float_scalar(
            "scalar",
            TensorDType::F32,
            &1.0_f32.to_le_bytes(),
            TensorDType::F64,
            &mut [0; 7]
        ),
        Err(CheckpointError::InvalidTensorPayload { .. })
    ));
    for (source_dtype, target_dtype) in [
        (TensorDType::I32, TensorDType::F32),
        (TensorDType::F32, TensorDType::I32),
        (TensorDType::U32, TensorDType::I32),
        (TensorDType::BOOL, TensorDType::U8),
    ] {
        assert!(matches!(validate_conversion(source_dtype, target_dtype),
            Err(CheckpointError::UnsupportedDTypeConversion { source_dtype: source, target_dtype: target })
            if source == source_dtype && target == target_dtype));
    }
    for (source_dtype, target_dtype) in [
        (TensorDType::I32, TensorDType::F32),
        (TensorDType::F32, TensorDType::I32),
    ] {
        assert!(matches!(
            convert_float_scalar("scalar", source_dtype, &[0; 4], target_dtype, &mut [0; 4]),
            Err(CheckpointError::UnsupportedDTypeConversion { .. })
        ));
    }
}

/// Rejects incomplete scalar storage before Burn receives a byte-backed tensor.
#[test]
fn final_data_and_endian_boundaries_reject_malformed_storage() {
    assert!(
        matches!(make_tensor_data("short", TensorDType::F32, &[1], vec![0; 3]),
        Err(CheckpointError::InvalidTensorPayload { tensor, reason })
        if tensor == "short" && reason == "materialized 3 bytes, expected 4")
    );
    assert!(
        matches!(make_tensor_data("mask", TensorDType::BOOL, &[3], vec![0, 1, 255]),
        Err(CheckpointError::InvalidTensorPayload { tensor, reason })
        if tensor == "mask" && reason.contains("element 2 has invalid byte value 255"))
    );
    assert_eq!(
        make_tensor_data("bytes", TensorDType::U8, &[3], vec![0, 1, 255])
            .unwrap()
            .to_vec::<u8>()
            .unwrap(),
        vec![0, 1, 255]
    );
    assert!(
        matches!(normalize_little_endian(&mut [0; 3], TensorDType::F16),
        Err(CheckpointError::InvalidTensorPayload { tensor, reason })
        if tensor == "<native-endian conversion>" && reason.contains("whole number of scalars"))
    );
    assert_eq!(
        make_tensor_data(
            "scalar",
            TensorDType::F32,
            &[],
            1.0_f32.to_ne_bytes().to_vec()
        )
        .unwrap()
        .shape
        .as_slice(),
        Vec::<usize>::new()
    );
    assert!(
        make_tensor_data("empty", TensorDType::F32, &[0, 3], vec![])
            .unwrap()
            .as_bytes()
            .is_empty()
    );
}

/// Exercises offset, extent, and end addition without allocating large buffers.
#[test]
fn checked_byte_ranges_and_staging_counters_reject_wraparound() {
    for (start, elements, width, context) in [
        (u64::MAX, 0, 2, "computing scalar byte offset"),
        (0, u64::MAX, 2, "computing scalar byte length"),
        (u64::MAX, 1, 1, "computing scalar byte range"),
    ] {
        assert!(matches!(byte_range(start, elements, width, "range"),
            Err(CheckpointError::ArithmeticOverflow { context: actual }) if actual == context));
    }
    assert_eq!(byte_range(2, 3, 4, "range").unwrap(), 8..20);
    assert_eq!(checked_elements(&[], "scalar").unwrap(), 1);
    assert_eq!(checked_elements(&[0, u64::MAX], "empty").unwrap(), 0);
    assert!(matches!(
        byte_len(&[u64::MAX], TensorDType::F64, "huge"),
        Err(CheckpointError::ArithmeticOverflow {
            context: "computing tensor storage bits"
        })
    ));
    assert!(matches!(
        byte_len(&[1], TensorDType::F4, "partial"),
        Err(CheckpointError::MisalignedTensorStorage { .. })
    ));
    #[cfg(target_pointer_width = "64")]
    {
        let mut stats = ExecutionStats::default();
        assert!(matches!(
            observe_peak(&mut stats, usize::MAX, 1, "peak"),
            Err(CheckpointError::ArithmeticOverflow {
                context: "recording loader staging high-water mark"
            })
        ));
        assert_eq!(stats, ExecutionStats::default());
    }
}

/// Rejects tiled transpose layouts before fixed-width workers index buffers.
#[cfg(target_endian = "little")]
#[test]
fn tiled_transpose_rejects_wrong_buffer_lengths_and_dimension_overflow() {
    for source_short in [false, true] {
        let source = vec![0; if source_short { 23 } else { 24 }];
        let mut target = vec![0xa5; if source_short { 24 } else { 23 }];
        let before = target.clone();
        assert!(
            matches!(transpose_little_endian_bytes("matrix", &source, &mut target, 2, 3, TensorDType::F32),
            Err(CheckpointError::InvalidTensorPayload { tensor, .. }) if tensor == "matrix")
        );
        assert_eq!(target, before);
    }
    #[cfg(target_pointer_width = "64")]
    assert!(matches!(
        transpose_little_endian_bytes("matrix", &[], &mut [], u64::MAX, 2, TensorDType::U8),
        Err(CheckpointError::ArithmeticOverflow {
            context: "computing tiled transpose byte length"
        })
    ));
}
