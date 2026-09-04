use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use burn::{
    backend::Flex,
    module::Param,
    nn::{LinearConfig, LinearLayout},
    tensor::{BoolStore, DType, Tensor, TensorData, bf16, f16},
};
use proptest::prelude::*;
use serde_json::{Map, Value, json};
use tempfile::TempDir;
use thiserror::Error;

use super::*;
use crate::checkpoint::quantization::{
    QuantizationGroupSize, SymmetricQuantization, WeightQuantizationConfig, quantize_tensor_data,
};
use crate::checkpoint::{
    ArtifactSelection, CheckpointError, DTypePolicy, ExpectedTensor, LoadOptions, MappingPlan,
    PlannedSource, PlannedTensor, PlannedTransform, TensorDType, build_load_plan,
    discover_checkpoint,
};

#[derive(Clone)]
struct FixtureTensor {
    name: String,
    dtype: TensorDType,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

impl FixtureTensor {
    fn new(
        name: impl Into<String>,
        dtype: TensorDType,
        shape: impl Into<Vec<u64>>,
        bytes: Vec<u8>,
    ) -> Self {
        let shape = shape.into();
        let elements = shape.iter().product();
        assert_eq!(dtype.storage_bytes(elements).unwrap() as usize, bytes.len());
        Self {
            name: name.into(),
            dtype,
            shape,
            bytes,
        }
    }
}

fn write_shard(root: &Path, relative: &str, tensors: &[FixtureTensor]) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let mut offset = 0_u64;
    let mut header = Map::new();
    for tensor in tensors {
        let end = offset + tensor.bytes.len() as u64;
        header.insert(
            tensor.name.clone(),
            json!({
                "dtype": tensor.dtype.as_str(),
                "shape": tensor.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    for tensor in tensors {
        file.write_all(&tensor.bytes).unwrap();
    }
}

fn write_index(root: &Path, assignments: &[(&str, &str)], total_size: u64) {
    let weight_map: BTreeMap<_, _> = assignments.iter().copied().collect();
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }))
        .unwrap(),
    )
    .unwrap();
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn tensor_f32(data: &TensorData) -> Vec<f32> {
    data.as_bytes()
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn tensor_u16(data: &TensorData) -> Vec<u16> {
    data.as_bytes()
        .chunks_exact(2)
        .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("synthetic sink rejected `{0}`")]
struct SinkError(String);

#[derive(Default)]
struct CollectingSink {
    tensors: BTreeMap<String, TensorData>,
    aliases: BTreeMap<String, String>,
    events: Vec<String>,
    fail_write: Option<String>,
    fail_alias: Option<String>,
}

impl TensorSink for CollectingSink {
    type Error = SinkError;

    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        if self.fail_write.as_deref() == Some(target) {
            return Err(SinkError(target.into()));
        }
        if self.tensors.insert(target.into(), data).is_some() {
            return Err(SinkError(format!("duplicate:{target}")));
        }
        self.events.push(format!("write:{target}"));
        Ok(())
    }

    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        if self.fail_alias.as_deref() == Some(target) {
            return Err(SinkError(target.into()));
        }
        if !self.tensors.contains_key(source_target) {
            return Err(SinkError(format!("missing-owner:{source_target}")));
        }
        self.aliases.insert(target.into(), source_target.into());
        self.events.push(format!("alias:{target}->{source_target}"));
        Ok(())
    }
}

/// Local fixture source that records each successful physical tensor read.
struct TracingLocalSource<'a> {
    inner: LocalShardSource<'a>,
    reads: Vec<(String, String, u64)>,
}

impl TensorSource for TracingLocalSource<'_> {
    fn read_tensor(&mut self, source: &PlannedSource) -> crate::checkpoint::Result<SourceTensor> {
        let payload = self.inner.read_tensor(source)?;
        self.reads.push((
            source.shard.clone(),
            source.name.clone(),
            source.data_offsets[0],
        ));
        Ok(payload)
    }

    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.inner.close()
    }

    fn stats(&self) -> SourceStats {
        self.inner.stats()
    }
}

fn single_manifest(tensors: &[FixtureTensor]) -> (TempDir, crate::checkpoint::CheckpointManifest) {
    let temp = TempDir::new().unwrap();
    write_shard(temp.path(), "model.safetensors", tensors);
    let manifest = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap();
    (temp, manifest)
}

fn identity_plan(
    manifest: &crate::checkpoint::CheckpointManifest,
    tensors: &[FixtureTensor],
) -> crate::checkpoint::LoadPlan {
    let expected: Vec<_> = tensors
        .iter()
        .map(|tensor| {
            ExpectedTensor::new(tensor.name.clone(), tensor.shape.clone()).with_dtype(tensor.dtype)
        })
        .collect();
    build_load_plan(
        manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap()
}

#[test]
fn identity_and_rename_deliver_native_burn_tensor_data() {
    let tensors = vec![
        FixtureTensor::new("flag", TensorDType::BOOL, [3], vec![1, 0, 1]),
        FixtureTensor::new("indices", TensorDType::I32, [3], i32_bytes(&[-7, 0, 42])),
        FixtureTensor::new(
            "external.weight",
            TensorDType::F32,
            [2, 2],
            f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        ),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let expected = [
        ExpectedTensor::new("flag", [3]).with_dtype(TensorDType::BOOL),
        ExpectedTensor::new("indices", [3]).with_dtype(TensorDType::I32),
        ExpectedTensor::new("internal.weight", [2, 2]).with_dtype(TensorDType::F32),
    ];
    let mut mapping = MappingPlan::new();
    mapping
        .map("external.weight", "internal.weight")
        .map("indices", "indices")
        .map("flag", "flag");
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    let mut sink = CollectingSink::default();

    let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    assert_eq!(
        tensor_f32(&sink.tensors["internal.weight"]),
        [1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(sink.tensors["internal.weight"].shape.as_slice(), [2, 2]);
    assert_eq!(sink.tensors["internal.weight"].dtype, DType::F32);
    assert_eq!(sink.tensors["indices"].dtype, DType::I32);
    assert_eq!(sink.tensors["flag"].dtype, DType::Bool(BoolStore::Native));
    assert_eq!(
        outcome.report.applied,
        ["flag", "indices", "internal.weight"]
    );
    assert_eq!(outcome.stats.tensors_applied, 3);
    assert_eq!(outcome.stats.source.shard_opens, 1);
    assert_eq!(outcome.stats.source.max_open_shards, 1);
    assert_eq!(outcome.stats.source.tensor_reads, 3);
    assert_eq!(outcome.stats.source.shard_open_order, ["model.safetensors"]);
}

#[test]
fn interleaved_targets_execute_once_per_shard_in_physical_offset_order() {
    let first = [
        FixtureTensor::new("source.zeta", TensorDType::F32, [1], f32_bytes(&[60.0])),
        FixtureTensor::new(
            "source.packed",
            TensorDType::F32,
            [2],
            f32_bytes(&[40.0, 50.0]),
        ),
        FixtureTensor::new(
            "source.matrix",
            TensorDType::F32,
            [1, 2],
            f32_bytes(&[20.0, 21.0]),
        ),
    ];
    let second = [
        FixtureTensor::new("source.alpha", TensorDType::F32, [1], f32_bytes(&[10.0])),
        FixtureTensor::new(
            "source.concat_left",
            TensorDType::F32,
            [1],
            f32_bytes(&[30.0]),
        ),
        FixtureTensor::new(
            "source.concat_right",
            TensorDType::F32,
            [1],
            f32_bytes(&[31.0]),
        ),
    ];
    let temp = TempDir::new().unwrap();
    write_shard(temp.path(), "part-01.safetensors", &first);
    write_shard(temp.path(), "part-02.safetensors", &second);
    write_index(
        temp.path(),
        &[
            ("source.alpha", "part-02.safetensors"),
            ("source.concat_left", "part-02.safetensors"),
            ("source.concat_right", "part-02.safetensors"),
            ("source.matrix", "part-01.safetensors"),
            ("source.packed", "part-01.safetensors"),
            ("source.zeta", "part-01.safetensors"),
        ],
        32,
    );
    let manifest = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap();
    let expected = [
        ExpectedTensor::new("aardvark_alias", [1]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("alpha", [1]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("beta", [2, 1]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("delta", [1]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("epsilon", [1]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("gamma", [2]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("zeta", [1]).with_dtype(TensorDType::F32),
    ];
    let mut mapping = MappingPlan::new();
    mapping
        .map("source.alpha", "alpha")
        .transpose_2d("source.matrix", "beta")
        .split(
            "source.packed",
            0,
            [("delta".into(), 1), ("epsilon".into(), 1)],
        )
        .concatenate(
            ["source.concat_left".into(), "source.concat_right".into()],
            "gamma",
            0,
        )
        .map("source.zeta", "zeta")
        .alias("zeta", "aardvark_alias");
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    assert_eq!(
        plan.tensors
            .iter()
            .map(|tensor| tensor.target.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "beta", "delta", "epsilon", "gamma", "zeta"]
    );
    assert_eq!(
        plan.tensors
            .iter()
            .map(|tensor| tensor.sources[0].shard.as_str())
            .collect::<Vec<_>>(),
        [
            "part-02.safetensors",
            "part-01.safetensors",
            "part-01.safetensors",
            "part-01.safetensors",
            "part-02.safetensors",
            "part-01.safetensors",
        ]
    );
    let planned_report = plan.report.clone();
    let mut source = TracingLocalSource {
        inner: LocalShardSource::new(&manifest),
        reads: Vec::new(),
    };
    let mut sink = CollectingSink::default();

    let outcome = execute_load_plan(&plan, &mut source, &mut sink).unwrap();

    assert_eq!(
        outcome.stats.source.shard_open_order,
        ["part-01.safetensors", "part-02.safetensors"]
    );
    assert_eq!(outcome.stats.source.shard_opens, 2);
    assert_eq!(outcome.stats.source.max_open_shards, 1);
    assert_eq!(outcome.stats.source.tensor_reads, 6);
    assert_eq!(outcome.stats.source_reads, 6);
    assert_eq!(
        source.reads,
        [
            ("part-01.safetensors".into(), "source.zeta".into(), 0),
            ("part-01.safetensors".into(), "source.packed".into(), 4),
            ("part-01.safetensors".into(), "source.matrix".into(), 12),
            ("part-02.safetensors".into(), "source.alpha".into(), 0),
            ("part-02.safetensors".into(), "source.concat_left".into(), 4,),
            (
                "part-02.safetensors".into(),
                "source.concat_right".into(),
                8,
            ),
        ]
    );
    assert_eq!(
        sink.events,
        [
            "write:zeta",
            "write:delta",
            "write:epsilon",
            "write:beta",
            "write:alpha",
            "write:gamma",
            "alias:aardvark_alias->zeta",
        ]
    );
    assert_eq!(tensor_f32(&sink.tensors["alpha"]), [10.0]);
    assert_eq!(tensor_f32(&sink.tensors["beta"]), [20.0, 21.0]);
    assert_eq!(tensor_f32(&sink.tensors["delta"]), [40.0]);
    assert_eq!(tensor_f32(&sink.tensors["epsilon"]), [50.0]);
    assert_eq!(tensor_f32(&sink.tensors["gamma"]), [30.0, 31.0]);
    assert_eq!(tensor_f32(&sink.tensors["zeta"]), [60.0]);
    assert_eq!(sink.aliases["aardvark_alias"], "zeta");
    let mut expected_report = planned_report;
    expected_report.applied = [
        "aardvark_alias",
        "alpha",
        "beta",
        "delta",
        "epsilon",
        "gamma",
        "zeta",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    expected_report.finalize();
    assert_eq!(outcome.report, expected_report);
}

#[test]
fn every_burn_representable_safetensors_dtype_is_delivered_without_reinterpretation() {
    let tensors = vec![
        FixtureTensor::new("bool", TensorDType::BOOL, [1], vec![1]),
        FixtureTensor::new("u8", TensorDType::U8, [1], vec![0xfe]),
        FixtureTensor::new("i8", TensorDType::I8, [1], vec![0xfe]),
        FixtureTensor::new("u16", TensorDType::U16, [1], u16_bytes(&[0xabcd])),
        FixtureTensor::new("i16", TensorDType::I16, [1], u16_bytes(&[0xfffe])),
        FixtureTensor::new("f16", TensorDType::F16, [1], u16_bytes(&[0x3c00])),
        FixtureTensor::new("bf16", TensorDType::BF16, [1], u16_bytes(&[0x3f80])),
        FixtureTensor::new(
            "u32",
            TensorDType::U32,
            [1],
            0x89ab_cdef_u32.to_le_bytes().to_vec(),
        ),
        FixtureTensor::new("i32", TensorDType::I32, [1], i32_bytes(&[-2])),
        FixtureTensor::new("f32", TensorDType::F32, [1], f32_bytes(&[1.5])),
        FixtureTensor::new(
            "u64",
            TensorDType::U64,
            [1],
            0x0123_4567_89ab_cdef_u64.to_le_bytes().to_vec(),
        ),
        FixtureTensor::new(
            "i64",
            TensorDType::I64,
            [1],
            (-2_i64).to_le_bytes().to_vec(),
        ),
        FixtureTensor::new(
            "f64",
            TensorDType::F64,
            [1],
            1.5_f64.to_bits().to_le_bytes().to_vec(),
        ),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let plan = identity_plan(&manifest, &tensors);
    let mut sink = CollectingSink::default();

    execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    let expected = [
        ("bool", DType::Bool(BoolStore::Native)),
        ("u8", DType::U8),
        ("i8", DType::I8),
        ("u16", DType::U16),
        ("i16", DType::I16),
        ("f16", DType::F16),
        ("bf16", DType::BF16),
        ("u32", DType::U32),
        ("i32", DType::I32),
        ("f32", DType::F32),
        ("u64", DType::U64),
        ("i64", DType::I64),
        ("f64", DType::F64),
    ];
    for (name, dtype) in expected {
        assert_eq!(sink.tensors[name].dtype, dtype, "{name}");
    }
}

struct RankTwoParamSink {
    parameter: Option<Param<Tensor<Flex, 2>>>,
}

impl TensorSink for RankTwoParamSink {
    type Error = SinkError;

    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        if target != "weight" || data.shape.as_slice().len() != 2 {
            return Err(SinkError(target.into()));
        }
        let device = Default::default();
        self.parameter = Some(Param::from_data(data, &device));
        Ok(())
    }

    fn alias_tensor(&mut self, target: &str, _source_target: &str) -> Result<(), Self::Error> {
        Err(SinkError(target.into()))
    }
}

#[test]
fn sink_boundary_can_construct_a_rank_typed_burn_parameter_immediately() {
    let tensor = FixtureTensor::new(
        "weight",
        TensorDType::F32,
        [2, 2],
        f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
    );
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let plan = identity_plan(&manifest, std::slice::from_ref(&tensor));
    let mut sink = RankTwoParamSink { parameter: None };

    execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    let data = sink.parameter.unwrap().val().to_data();
    assert_eq!(data.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn rank_two_transpose_changes_shape_and_row_major_order() {
    let tensor = FixtureTensor::new(
        "matrix",
        TensorDType::F32,
        [2, 3],
        f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let expected = [ExpectedTensor::new("transposed", [3, 2]).with_dtype(TensorDType::F32)];
    let mut mapping = MappingPlan::new();
    mapping.transpose_2d("matrix", "transposed");
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    let mut sink = CollectingSink::default();

    let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    assert_eq!(sink.tensors["transposed"].shape.as_slice(), [3, 2]);
    assert_eq!(
        tensor_f32(&sink.tensors["transposed"]),
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    assert_eq!(outcome.stats.peak_host_staging_bytes, 48);
}

#[test]
fn transpose_casts_bf16_directly_into_the_final_f16_layout() {
    let values = [
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
        bf16::from_f32(5.0),
        bf16::from_f32(6.0),
    ];
    let tensor = FixtureTensor::new(
        "matrix",
        TensorDType::BF16,
        [2, 3],
        u16_bytes(&values.map(bf16::to_bits)),
    );
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let expected = [ExpectedTensor::new("transposed", [3, 2]).with_dtype(TensorDType::F16)];
    let mut mapping = MappingPlan::new();
    mapping.transpose_2d("matrix", "transposed");
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    let mut sink = CollectingSink::default();

    execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    let actual = tensor_u16(&sink.tensors["transposed"]);
    let expected = [1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0].map(|value| f16::from_f32(value).to_bits());
    assert_eq!(actual, expected);
}

#[test]
fn splits_cover_first_middle_last_and_zero_length_axes() {
    let tensors = vec![
        FixtureTensor::new(
            "axis0",
            TensorDType::U16,
            [4, 2, 2],
            u16_bytes(&(0..16).collect::<Vec<_>>()),
        ),
        FixtureTensor::new(
            "axis1",
            TensorDType::U16,
            [2, 4, 2],
            u16_bytes(&(100..116).collect::<Vec<_>>()),
        ),
        FixtureTensor::new(
            "axis2",
            TensorDType::U16,
            [2, 2, 4],
            u16_bytes(&(200..216).collect::<Vec<_>>()),
        ),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let expected = [
        ExpectedTensor::new("a0", [1, 2, 2]).with_dtype(TensorDType::U16),
        ExpectedTensor::new("b0", [3, 2, 2]).with_dtype(TensorDType::U16),
        ExpectedTensor::new("a1", [2, 0, 2]).with_dtype(TensorDType::U16),
        ExpectedTensor::new("b1", [2, 4, 2]).with_dtype(TensorDType::U16),
        ExpectedTensor::new("a2", [2, 2, 3]).with_dtype(TensorDType::U16),
        ExpectedTensor::new("b2", [2, 2, 1]).with_dtype(TensorDType::U16),
    ];
    let mut mapping = MappingPlan::new();
    mapping
        .split("axis0", 0, [("a0".into(), 1), ("b0".into(), 3)])
        .split("axis1", 1, [("a1".into(), 0), ("b1".into(), 4)])
        .split("axis2", 2, [("a2".into(), 3), ("b2".into(), 1)]);
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    let mut sink = CollectingSink::default();

    let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    assert_eq!(tensor_u16(&sink.tensors["a0"]), (0..4).collect::<Vec<_>>());
    assert_eq!(tensor_u16(&sink.tensors["b0"]), (4..16).collect::<Vec<_>>());
    assert!(tensor_u16(&sink.tensors["a1"]).is_empty());
    assert_eq!(
        tensor_u16(&sink.tensors["b1"]),
        (100..116).collect::<Vec<_>>()
    );
    assert_eq!(
        tensor_u16(&sink.tensors["a2"]),
        [200, 201, 202, 204, 205, 206, 208, 209, 210, 212, 213, 214]
    );
    assert_eq!(tensor_u16(&sink.tensors["b2"]), [203, 207, 211, 215]);
    assert_eq!(outcome.stats.source_reads, 3);
    assert_eq!(outcome.stats.source_payload_bytes, 96);
    assert_eq!(outcome.stats.source.tensor_reads, 3);
    assert_eq!(outcome.stats.source.payload_bytes_read, 96);
    assert_eq!(outcome.stats.source.shard_opens, 1);
    assert_eq!(outcome.stats.source.max_open_shards, 1);
    assert_eq!(outcome.stats.peak_host_staging_bytes, 64);
    assert_eq!(
        sink.events,
        [
            "write:a0", "write:b0", "write:a1", "write:b1", "write:a2", "write:b2"
        ]
    );
    assert!(outcome.stats.peak_host_staging_bytes <= plan.memory.peak_host_staging_bytes);
}

#[test]
fn grouped_split_failure_reports_only_accepted_outputs_without_rereading_source() {
    let tensor = FixtureTensor::new(
        "packed",
        TensorDType::F32,
        [4],
        f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
    );
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let expected = [
        ExpectedTensor::new("first", [2]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("second", [2]).with_dtype(TensorDType::F32),
    ];
    let mut mapping = MappingPlan::new();
    mapping.split("packed", 0, [("first".into(), 2), ("second".into(), 2)]);
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    let mut sink = CollectingSink {
        fail_write: Some("second".into()),
        ..CollectingSink::default()
    };

    let error = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap_err();

    assert_eq!(error.failed_target.as_deref(), Some("second"));
    assert_eq!(error.report.applied, ["first"]);
    assert_eq!(error.stats.tensors_applied, 1);
    assert_eq!(error.stats.source_reads, 1);
    assert_eq!(error.stats.source_payload_bytes, 16);
    assert_eq!(error.stats.source.tensor_reads, 1);
    assert_eq!(error.stats.source.payload_bytes_read, 16);
    assert_eq!(error.stats.peak_host_staging_bytes, 24);
    assert_eq!(sink.events, ["write:first"]);
}

#[test]
fn concatenation_covers_every_axis_and_switches_one_shard_at_a_time() {
    let first = vec![
        FixtureTensor::new("a0", TensorDType::U16, [1, 2, 2], u16_bytes(&[1, 2, 3, 4])),
        FixtureTensor::new(
            "a1",
            TensorDType::U16,
            [2, 1, 2],
            u16_bytes(&[10, 11, 12, 13]),
        ),
        FixtureTensor::new(
            "a2",
            TensorDType::U16,
            [2, 2, 1],
            u16_bytes(&[20, 21, 22, 23]),
        ),
    ];
    let second = vec![
        FixtureTensor::new(
            "b0",
            TensorDType::U16,
            [2, 2, 2],
            u16_bytes(&[5, 6, 7, 8, 9, 10, 11, 12]),
        ),
        FixtureTensor::new(
            "b1",
            TensorDType::U16,
            [2, 2, 2],
            u16_bytes(&[14, 15, 16, 17, 18, 19, 20, 21]),
        ),
        FixtureTensor::new(
            "b2",
            TensorDType::U16,
            [2, 2, 2],
            u16_bytes(&[24, 25, 26, 27, 28, 29, 30, 31]),
        ),
    ];
    let temp = TempDir::new().unwrap();
    write_shard(temp.path(), "one.safetensors", &first);
    write_shard(temp.path(), "two.safetensors", &second);
    write_index(
        temp.path(),
        &[
            ("a0", "one.safetensors"),
            ("a1", "one.safetensors"),
            ("a2", "one.safetensors"),
            ("b0", "two.safetensors"),
            ("b1", "two.safetensors"),
            ("b2", "two.safetensors"),
        ],
        72,
    );
    let manifest = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap();
    let expected = [
        ExpectedTensor::new("cat0", [3, 2, 2]).with_dtype(TensorDType::U16),
        ExpectedTensor::new("cat1", [2, 3, 2]).with_dtype(TensorDType::U16),
        ExpectedTensor::new("cat2", [2, 2, 3]).with_dtype(TensorDType::U16),
    ];
    let mut mapping = MappingPlan::new();
    mapping
        .concatenate(["a0".into(), "b0".into()], "cat0", 0)
        .concatenate(["a1".into(), "b1".into()], "cat1", 1)
        .concatenate(["a2".into(), "b2".into()], "cat2", 2);
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    let mut sink = CollectingSink::default();

    let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    assert_eq!(
        tensor_u16(&sink.tensors["cat0"]),
        (1..=12).collect::<Vec<_>>()
    );
    assert_eq!(
        tensor_u16(&sink.tensors["cat1"]),
        [10, 11, 14, 15, 16, 17, 12, 13, 18, 19, 20, 21]
    );
    assert_eq!(
        tensor_u16(&sink.tensors["cat2"]),
        [20, 24, 25, 21, 26, 27, 22, 28, 29, 23, 30, 31]
    );
    assert_eq!(outcome.stats.source.max_open_shards, 1);
    assert_eq!(
        outcome.stats.source.shard_open_order,
        [
            "one.safetensors",
            "two.safetensors",
            "one.safetensors",
            "two.safetensors",
            "one.safetensors",
            "two.safetensors",
        ]
    );
    assert_eq!(outcome.stats.source.tensor_reads, 6);
    assert!(outcome.stats.peak_host_staging_bytes <= plan.memory.peak_host_staging_bytes);
}

#[test]
fn every_bf16_bit_pattern_casts_to_f16_with_half_semantics() {
    let bits: Vec<u16> = (0..=u16::MAX).collect();
    let tensor = FixtureTensor::new(
        "weight",
        TensorDType::BF16,
        [bits.len() as u64],
        u16_bytes(&bits),
    );
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let expected =
        [ExpectedTensor::new("weight", [bits.len() as u64]).with_dtype(TensorDType::F16)];
    let mut plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap();
    let payload_bytes = u64::try_from(bits.len() * 2).unwrap();
    plan.memory.peak_host_staging_bytes = payload_bytes;
    let mut sink = CollectingSink::default();

    let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    let actual = tensor_u16(&sink.tensors["weight"]);
    for (source, target) in bits.iter().zip(actual) {
        let expected = f16::from_f32(bf16::from_bits(*source).to_f32());
        let actual = f16::from_bits(target);
        if expected.is_nan() {
            assert!(actual.is_nan(), "BF16 bits {source:#06x}");
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "BF16 bits {source:#06x}"
            );
        }
    }
    assert_eq!(outcome.report.dtype_converted.len(), 1);
    assert_eq!(outcome.stats.peak_host_staging_bytes, payload_bytes);
    assert_eq!(outcome.stats.largest_source_payload_bytes, payload_bytes);
    assert_eq!(outcome.stats.largest_output_tensor_bytes, payload_bytes);
}

#[test]
fn bf16_to_f16_identity_rejects_a_bound_smaller_than_its_reused_buffer() {
    let tensor = FixtureTensor::new(
        "weight",
        TensorDType::BF16,
        [4],
        u16_bytes(&[0x0000, 0x3f80, 0x7f80, 0x7fc1]),
    );
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let expected = [ExpectedTensor::new("weight", [4]).with_dtype(TensorDType::F16)];
    let mut plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap();
    plan.memory.peak_host_staging_bytes = 7;
    let mut sink = CollectingSink::default();

    let error = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap_err();

    assert!(matches!(
        error.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::MemoryEstimateExceeded {
            required: 8,
            planned: 7,
            ..
        })
    ));
    assert!(sink.events.is_empty());
}

#[test]
fn every_f16_bit_pattern_casts_to_bf16_with_half_semantics() {
    let bits: Vec<u16> = (0..=u16::MAX).collect();
    let tensor = FixtureTensor::new(
        "weight",
        TensorDType::F16,
        [bits.len() as u64],
        u16_bytes(&bits),
    );
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let expected =
        [ExpectedTensor::new("weight", [bits.len() as u64]).with_dtype(TensorDType::BF16)];
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap();
    let mut sink = CollectingSink::default();

    let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    let actual = tensor_u16(&sink.tensors["weight"]);
    for (source, target) in bits.iter().zip(actual) {
        let expected = bf16::from_f32(f16::from_bits(*source).to_f32());
        let actual = bf16::from_bits(target);
        if expected.is_nan() {
            assert!(actual.is_nan(), "F16 bits {source:#06x}");
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "F16 bits {source:#06x}"
            );
        }
    }
    assert_eq!(outcome.report.dtype_converted.len(), 1);
    assert_eq!(outcome.stats.peak_host_staging_bytes, 262_144);
}

#[test]
fn preserve_float_identity_does_not_allocate_a_second_output_buffer() {
    let tensor = FixtureTensor::new(
        "weight",
        TensorDType::F32,
        [1024],
        f32_bytes(&vec![1.0; 1024]),
    );
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let plan = identity_plan(&manifest, std::slice::from_ref(&tensor));
    let mut sink = CollectingSink::default();

    let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    assert_eq!(outcome.stats.peak_host_staging_bytes, 4096);
    assert_eq!(outcome.stats.largest_output_tensor_bytes, 4096);
    assert!(outcome.stats.peak_host_staging_bytes <= plan.memory.peak_host_staging_bytes);
}

#[test]
fn staging_high_water_is_per_tensor_instead_of_complete_model_size() {
    let tensors = vec![
        FixtureTensor::new("a", TensorDType::F32, [8], f32_bytes(&[1.0; 8])),
        FixtureTensor::new("b", TensorDType::F32, [8], f32_bytes(&[2.0; 8])),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let expected = [
        ExpectedTensor::new("a", [8]).with_dtype(TensorDType::F16),
        ExpectedTensor::new("b", [8]).with_dtype(TensorDType::F16),
    ];
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap();
    let mut sink = CollectingSink::default();

    let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    assert_eq!(outcome.stats.peak_host_staging_bytes, 48);
    assert_eq!(outcome.stats.source_payload_bytes, 64);
    assert_eq!(outcome.stats.delivered_tensor_bytes, 32);
    assert!(outcome.stats.peak_host_staging_bytes < 64 + 32);
    assert!(outcome.stats.peak_host_staging_bytes <= plan.memory.peak_host_staging_bytes);
}

#[test]
fn aliases_are_flattened_and_installed_after_source_close() {
    let tensor = FixtureTensor::new("weight", TensorDType::F32, [1], f32_bytes(&[3.0]));
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let expected = [
        ExpectedTensor::new("weight", [1]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("embedding", [1]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("lm_head", [1]).with_dtype(TensorDType::F32),
    ];
    let mut mapping = MappingPlan::new();
    mapping
        .map("weight", "weight")
        .alias("weight", "embedding")
        .alias("embedding", "lm_head");
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    let mut sink = CollectingSink::default();

    let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    assert_eq!(
        sink.events,
        [
            "write:weight",
            "alias:embedding->weight",
            "alias:lm_head->weight"
        ]
    );
    assert_eq!(outcome.stats.aliases_applied, 2);
    assert_eq!(outcome.report.applied, ["embedding", "lm_head", "weight"]);
}

#[test]
fn sink_write_failure_returns_an_exact_partial_report() {
    let tensors = vec![
        FixtureTensor::new("a", TensorDType::F32, [1], f32_bytes(&[1.0])),
        FixtureTensor::new("b", TensorDType::F32, [1], f32_bytes(&[2.0])),
        FixtureTensor::new("c", TensorDType::F32, [1], f32_bytes(&[3.0])),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let plan = identity_plan(&manifest, &tensors);
    let mut sink = CollectingSink {
        fail_write: Some("b".into()),
        ..CollectingSink::default()
    };

    let error = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap_err();

    assert_eq!(error.failed_target.as_deref(), Some("b"));
    assert!(matches!(error.cause, ExecutionFailureCause::Sink(SinkError(ref key)) if key == "b"));
    assert_eq!(error.report.applied, ["a"]);
    assert_eq!(error.stats.tensors_applied, 1);
    assert_eq!(error.stats.source.tensor_reads, 2);
    assert_eq!(sink.events, ["write:a"]);
    assert!(error.to_string().contains("`b`"));
}

#[test]
fn alias_failure_does_not_claim_failed_or_later_aliases() {
    let tensor = FixtureTensor::new("weight", TensorDType::F32, [1], f32_bytes(&[1.0]));
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let expected = [
        ExpectedTensor::new("weight", [1]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("a_alias", [1]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("z_alias", [1]).with_dtype(TensorDType::F32),
    ];
    let mut mapping = MappingPlan::new();
    mapping
        .map("weight", "weight")
        .alias("weight", "a_alias")
        .alias("weight", "z_alias");
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    let mut sink = CollectingSink {
        fail_alias: Some("a_alias".into()),
        ..CollectingSink::default()
    };

    let error = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap_err();

    assert_eq!(error.failed_target.as_deref(), Some("a_alias"));
    assert_eq!(error.report.applied, ["weight"]);
    assert_eq!(error.stats.aliases_applied, 0);
    assert!(!sink.aliases.contains_key("z_alias"));
}

#[derive(Default)]
struct MemorySource {
    tensors: BTreeMap<String, SourceTensor>,
    close_calls: usize,
    fail_at: Option<String>,
}

struct CloseAwareSink {
    closed: Arc<AtomicBool>,
    owner_written: bool,
    alias_written: bool,
}

impl TensorSink for CloseAwareSink {
    type Error = SinkError;

    fn write_tensor(&mut self, _target: &str, _data: TensorData) -> Result<(), Self::Error> {
        self.owner_written = true;
        Ok(())
    }

    fn alias_tensor(&mut self, target: &str, _source_target: &str) -> Result<(), Self::Error> {
        if !self.closed.load(Ordering::SeqCst) {
            return Err(SinkError(format!("source-open:{target}")));
        }
        self.alias_written = true;
        Ok(())
    }
}

struct CloseAwareSource {
    inner: MemorySource,
    closed: Arc<AtomicBool>,
    fail_close: bool,
}

impl TensorSource for CloseAwareSource {
    fn read_tensor(&mut self, source: &PlannedSource) -> crate::checkpoint::Result<SourceTensor> {
        self.inner.read_tensor(source)
    }

    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        if self.fail_close {
            Err(CheckpointError::InvalidMapping {
                message: "synthetic close failure".into(),
            })
        } else {
            Ok(())
        }
    }
}

fn aliased_memory_fixture() -> (crate::checkpoint::LoadPlan, MemorySource) {
    let tensor = FixtureTensor::new("weight", TensorDType::F32, [1], f32_bytes(&[1.0]));
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let expected = [
        ExpectedTensor::new("weight", [1]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("alias", [1]).with_dtype(TensorDType::F32),
    ];
    let mut mapping = MappingPlan::new();
    mapping.map("weight", "weight").alias("weight", "alias");
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    let source = MemorySource {
        tensors: BTreeMap::from([(
            "weight".into(),
            SourceTensor::new("weight", [1], TensorDType::F32, tensor.bytes).unwrap(),
        )]),
        ..MemorySource::default()
    };
    (plan, source)
}

#[test]
fn aliases_observe_a_closed_source_boundary() {
    let (plan, inner) = aliased_memory_fixture();
    let closed = Arc::new(AtomicBool::new(false));
    let mut source = CloseAwareSource {
        inner,
        closed: closed.clone(),
        fail_close: false,
    };
    let mut sink = CloseAwareSink {
        closed,
        owner_written: false,
        alias_written: false,
    };

    execute_load_plan(&plan, &mut source, &mut sink).unwrap();

    assert!(sink.owner_written);
    assert!(sink.alias_written);
}

#[test]
fn close_failure_stops_before_aliases_with_primary_reported() {
    let (plan, inner) = aliased_memory_fixture();
    let closed = Arc::new(AtomicBool::new(false));
    let mut source = CloseAwareSource {
        inner,
        closed: closed.clone(),
        fail_close: true,
    };
    let mut sink = CloseAwareSink {
        closed,
        owner_written: false,
        alias_written: false,
    };

    let error = execute_load_plan(&plan, &mut source, &mut sink).unwrap_err();

    assert!(matches!(
        error.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::InvalidMapping { .. })
    ));
    assert_eq!(error.failed_target, None);
    assert_eq!(error.report.applied, ["weight"]);
    assert!(sink.owner_written);
    assert!(!sink.alias_written);
}

impl TensorSource for MemorySource {
    fn read_tensor(&mut self, source: &PlannedSource) -> crate::checkpoint::Result<SourceTensor> {
        if self.fail_at.as_deref() == Some(&source.name) {
            return Err(CheckpointError::InvalidTensorPayload {
                tensor: source.name.clone(),
                reason: "synthetic read failure".into(),
            });
        }
        self.tensors.get(&source.name).cloned().ok_or_else(|| {
            CheckpointError::InvalidTensorPayload {
                tensor: source.name.clone(),
                reason: "synthetic tensor is absent".into(),
            }
        })
    }

    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.close_calls += 1;
        Ok(())
    }
}

#[test]
fn custom_source_metadata_is_revalidated_and_closed_on_failure() {
    let tensor = FixtureTensor::new("a", TensorDType::F32, [2], f32_bytes(&[1.0, 2.0]));
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let plan = identity_plan(&manifest, std::slice::from_ref(&tensor));
    let mut source = MemorySource::default();
    source.tensors.insert(
        "a".into(),
        SourceTensor::new("wrong-name", [2], TensorDType::F32, tensor.bytes).unwrap(),
    );
    let mut sink = CollectingSink::default();

    let error = execute_load_plan(&plan, &mut source, &mut sink).unwrap_err();

    assert!(matches!(
        error.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::InvalidTensorPayload { .. })
    ));
    assert_eq!(source.close_calls, 1);
    assert!(error.report.applied.is_empty());
    assert!(sink.tensors.is_empty());
}

#[test]
fn source_read_failure_preserves_prior_sink_writes_and_closes() {
    let tensors = vec![
        FixtureTensor::new("a", TensorDType::F32, [1], f32_bytes(&[1.0])),
        FixtureTensor::new("b", TensorDType::F32, [1], f32_bytes(&[2.0])),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let plan = identity_plan(&manifest, &tensors);
    let mut source = MemorySource {
        tensors: tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    SourceTensor::new(
                        tensor.name.clone(),
                        tensor.shape.clone(),
                        tensor.dtype,
                        tensor.bytes.clone(),
                    )
                    .unwrap(),
                )
            })
            .collect(),
        fail_at: Some("b".into()),
        ..MemorySource::default()
    };
    let mut sink = CollectingSink::default();

    let error = execute_load_plan(&plan, &mut source, &mut sink).unwrap_err();

    assert_eq!(error.failed_target.as_deref(), Some("b"));
    assert_eq!(error.report.applied, ["a"]);
    assert_eq!(source.close_calls, 1);
    assert_eq!(sink.events, ["write:a"]);
}

#[test]
fn source_tensor_constructor_rejects_bad_lengths_and_overflowing_shapes() {
    assert!(matches!(
        SourceTensor::new("short", [2], TensorDType::F32, vec![0; 4]),
        Err(CheckpointError::InvalidTensorPayload { .. })
    ));
    assert!(matches!(
        SourceTensor::new("huge", [u64::MAX, 2], TensorDType::U8, Vec::new()),
        Err(CheckpointError::InvalidTensorPayload { .. })
    ));
}

#[test]
fn invalid_boolean_wire_values_are_rejected_before_the_sink() {
    let tensor = FixtureTensor::new("mask", TensorDType::BOOL, [3], vec![0, 2, 1]);
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let plan = identity_plan(&manifest, std::slice::from_ref(&tensor));
    let mut sink = CollectingSink::default();

    let error = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap_err();

    assert!(matches!(
        error.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::InvalidTensorPayload { .. })
    ));
    assert!(sink.tensors.is_empty());
}

#[test]
fn valid_but_unrepresentable_safetensors_dtype_fails_before_io() {
    let tensor = FixtureTensor::new("fp8", TensorDType::F8_E5M2, [2], vec![0, 1]);
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let plan = identity_plan(&manifest, std::slice::from_ref(&tensor));
    let mut source = LocalShardSource::new(&manifest);
    let mut sink = CollectingSink::default();

    let error = execute_load_plan(&plan, &mut source, &mut sink).unwrap_err();

    assert!(matches!(
        error.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::UnsupportedRuntimeDType {
            dtype: TensorDType::F8_E5M2
        })
    ));
    assert_eq!(error.stats.source.shard_opens, 0);
    assert!(error.failed_target.is_none());
}

#[test]
fn mutated_plan_is_rejected_before_any_source_or_sink_action() {
    let tensor = FixtureTensor::new("matrix", TensorDType::F32, [2, 2], f32_bytes(&[1.0; 4]));
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let mut plan = identity_plan(&manifest, std::slice::from_ref(&tensor));
    plan.tensors[0].shape = vec![4];
    let mut source = LocalShardSource::new(&manifest);
    let mut sink = CollectingSink::default();

    let error = execute_load_plan(&plan, &mut source, &mut sink).unwrap_err();

    assert!(matches!(
        error.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::InvalidMapping { .. })
    ));
    assert_eq!(error.stats.source.shard_opens, 0);
    assert!(sink.events.is_empty());
}

#[test]
fn understated_memory_plan_is_rejected_before_allocating() {
    let tensor = FixtureTensor::new("weight", TensorDType::F32, [8], f32_bytes(&[1.0; 8]));
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let expected = [ExpectedTensor::new("weight", [8]).with_dtype(TensorDType::F16)];
    let mut plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap();
    plan.memory.peak_host_staging_bytes = 47;
    let mut sink = CollectingSink::default();

    let error = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap_err();

    assert!(matches!(
        error.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::MemoryEstimateExceeded {
            required: 48,
            planned: 47,
            ..
        })
    ));
    assert!(sink.events.is_empty());
}

#[test]
fn non_float_dtype_reinterpretation_in_a_mutated_plan_is_rejected() {
    let tensor = FixtureTensor::new(
        "weight",
        TensorDType::U32,
        [1],
        7_u32.to_le_bytes().to_vec(),
    );
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let mut plan = identity_plan(&manifest, std::slice::from_ref(&tensor));
    plan.tensors[0].dtype = TensorDType::I32;
    let mut sink = CollectingSink::default();

    let error = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap_err();

    assert!(matches!(
        error.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::UnsupportedDTypeConversion {
            source_dtype: TensorDType::U32,
            target_dtype: TensorDType::I32,
        })
    ));
    assert!(sink.events.is_empty());
}

#[test]
fn file_size_change_after_discovery_is_detected() {
    let tensor = FixtureTensor::new("weight", TensorDType::F32, [1], f32_bytes(&[1.0]));
    let (temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let plan = identity_plan(&manifest, std::slice::from_ref(&tensor));
    OpenOptions::new()
        .append(true)
        .open(temp.path().join("model.safetensors"))
        .unwrap()
        .write_all(&[0])
        .unwrap();
    let mut sink = CollectingSink::default();

    let error = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap_err();

    assert!(matches!(
        error.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::ArtifactChanged { .. })
    ));
}

#[test]
fn header_length_change_after_discovery_is_detected() {
    let tensor = FixtureTensor::new("weight", TensorDType::F32, [1], f32_bytes(&[1.0]));
    let (temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let plan = identity_plan(&manifest, std::slice::from_ref(&tensor));
    let path = temp.path().join("model.safetensors");
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    file.write_all(&(manifest.shards[0].header_bytes + 8).to_le_bytes())
        .unwrap();
    let mut sink = CollectingSink::default();

    let error = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap_err();

    assert!(matches!(
        error.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::ArtifactChanged { .. })
    ));
}

#[test]
fn plan_source_descriptor_change_is_detected_by_local_source() {
    let tensor = FixtureTensor::new("weight", TensorDType::F32, [1], f32_bytes(&[1.0]));
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let mut plan = identity_plan(&manifest, std::slice::from_ref(&tensor));
    plan.tensors[0].sources[0].data_offsets = [0, 4];
    plan.tensors[0].sources[0].shard = "invented.safetensors".into();
    let mut sink = CollectingSink::default();

    let error = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap_err();

    assert!(matches!(
        error.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::PlanManifestMismatch { .. })
    ));
}

#[test]
fn explicit_artifact_selection_still_streams_the_selected_file() {
    let temp = TempDir::new().unwrap();
    let tensor = FixtureTensor::new("weight", TensorDType::F32, [1], f32_bytes(&[9.0]));
    write_shard(
        temp.path(),
        "custom.safetensors",
        std::slice::from_ref(&tensor),
    );
    let options = LoadOptions {
        artifact: ArtifactSelection::Single("custom.safetensors".into()),
        ..LoadOptions::default()
    };
    let manifest = discover_checkpoint(temp.path(), &options).unwrap();
    let plan = identity_plan(&manifest, std::slice::from_ref(&tensor));
    let mut sink = CollectingSink::default();

    let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    assert_eq!(tensor_f32(&sink.tensors["weight"]), [9.0]);
    assert_eq!(
        outcome.stats.source.shard_open_order,
        ["custom.safetensors"]
    );
}

fn in_memory_transpose(rows: u64, columns: u64, values: Vec<u16>) -> Vec<u16> {
    let source = PlannedSource {
        name: "source".into(),
        shard: "memory.safetensors".into(),
        shape: vec![rows, columns],
        dtype: TensorDType::U16,
        data_offsets: [0, rows * columns * 2],
    };
    let target = PlannedTensor {
        target: "target".into(),
        sources: vec![source],
        transform: PlannedTransform::Transpose2d,
        shape: vec![columns, rows],
        dtype: TensorDType::U16,
    };
    let report = crate::checkpoint::LoadReport {
        memory: Some(crate::checkpoint::MemoryEstimate {
            peak_host_staging_bytes: rows * columns * 4,
            ..crate::checkpoint::MemoryEstimate::default()
        }),
        ..crate::checkpoint::LoadReport::default()
    };
    let plan = crate::checkpoint::LoadPlan {
        tensors: vec![target],
        aliases: Vec::new(),
        report,
        memory: crate::checkpoint::MemoryEstimate {
            peak_host_staging_bytes: rows * columns * 4,
            ..crate::checkpoint::MemoryEstimate::default()
        },
    };
    let mut source = MemorySource::default();
    source.tensors.insert(
        "source".into(),
        SourceTensor::new(
            "source",
            [rows, columns],
            TensorDType::U16,
            u16_bytes(&values),
        )
        .unwrap(),
    );
    let mut sink = CollectingSink::default();
    execute_load_plan(&plan, &mut source, &mut sink).unwrap();
    tensor_u16(&sink.tensors["target"])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn transpose_matches_index_equation(
        rows in 0_u64..8,
        columns in 0_u64..8,
        seed in any::<u16>(),
    ) {
        let values: Vec<_> = (0..rows * columns)
            .map(|index| seed.wrapping_add(index as u16))
            .collect();
        let actual = in_memory_transpose(rows, columns, values.clone());
        let mut expected = Vec::with_capacity(values.len());
        for column in 0..columns {
            for row in 0..rows {
                expected.push(values[(row * columns + column) as usize]);
            }
        }
        prop_assert_eq!(actual, expected);
    }
}

#[test]
fn dtype_policy_can_explicitly_cast_f64_to_bf16() {
    let values = [1.0_f64, -2.5, f64::INFINITY, f64::NEG_INFINITY];
    let bytes = values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect();
    let tensor = FixtureTensor::new("weight", TensorDType::F64, [4], bytes);
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&tensor));
    let expected = [ExpectedTensor::new("weight", [4]).with_dtype(TensorDType::BF16)];
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::BF16),
        ..LoadOptions::default()
    };
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    let mut sink = CollectingSink::default();

    execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();

    let actual = tensor_u16(&sink.tensors["weight"]);
    let expected: Vec<_> = values
        .iter()
        .map(|value| bf16::from_f64(*value).to_bits())
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn parameter_install_uses_burns_checkpoint_layout_mapper() {
    let device = Default::default();
    let mut linear = LinearConfig::new(2, 3)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init::<Flex>(&device);
    assert!(!linear.weight.is_initialized());

    install_float_parameter(
        &mut linear.weight,
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]),
    )
    .unwrap();

    assert!(linear.weight.is_initialized());
    assert_eq!(linear.weight.val().dims(), [2, 3]);
    let output = linear
        .forward(Tensor::<Flex, 2>::from_floats([[10.0, 100.0]], &device))
        .to_data();
    assert_eq!(
        output,
        TensorData::from([[210.0_f32, 430.0_f32, 650.0_f32]])
    );
}

#[test]
fn parameter_install_preserves_native_quantized_float_data() {
    let device = Default::default();
    let mut linear = LinearConfig::new(64, 3)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init::<Flex>(&device);
    let source = TensorData::new(
        (0..192)
            .map(|index| (index as f32 - 95.5) / 16.0)
            .collect::<Vec<_>>(),
        [3, 64],
    );
    let packed = quantize_tensor_data(
        &source,
        WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q8,
            group_size: QuantizationGroupSize::new(32).unwrap(),
        },
    )
    .unwrap()
    .into_burn_qfloat_data()
    .unwrap();

    install_float_parameter(&mut linear.weight, packed).unwrap();

    assert!(linear.weight.is_initialized());
    assert_eq!(linear.weight.val().dims(), [64, 3]);
    assert!(matches!(linear.weight.val().dtype(), DType::QFloat(_)));
}

#[test]
fn parameter_install_rejects_bad_metadata_before_initialization() {
    let device = Default::default();
    let mut linear = LinearConfig::new(2, 3)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init::<Flex>(&device);

    let error = install_float_parameter(&mut linear.weight, TensorData::new(vec![0.0_f32; 6], [6]))
        .unwrap_err();
    assert_eq!(
        error,
        ParameterInstallError::Rank {
            expected: 2,
            actual: 1,
        }
    );
    assert!(!linear.weight.is_initialized());

    let error = install_float_parameter(
        &mut linear.weight,
        TensorData::new(vec![0.0_f32; 8], [4, 2]),
    )
    .unwrap_err();
    assert_eq!(
        error,
        ParameterInstallError::Shape {
            expected: vec![3, 2],
            actual: vec![4, 2],
        }
    );
    assert!(!linear.weight.is_initialized());

    let error = install_float_parameter(
        &mut linear.weight,
        TensorData::new(vec![true, false, true, false, true, false], [3, 2]),
    )
    .unwrap_err();
    assert!(matches!(error, ParameterInstallError::NonFloat { .. }));
    assert!(!linear.weight.is_initialized());
}
