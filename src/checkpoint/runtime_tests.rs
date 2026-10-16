//! Synthetic checkpoint execution tests for payloads, transforms, and sink atomicity.
//!
//! Byte-level fixtures preserve wire dtypes and row-major axes independently of
//! Burn construction. Instrumented sources and sinks expose read/write order,
//! cleanup, and exact partial progress when a later operation fails.

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
    ArtifactSelection, CheckpointEquivalenceRecord, CheckpointError, DTypePolicy, ExpectedTensor,
    IndexTensorAlias, LoadOptions, MappingPlan, PlannedSource, PlannedTensor, PlannedTransform,
    QuantizationRecord, TensorDType, build_load_plan, discover_checkpoint,
};

/// One tensor's descriptor and little-endian bytes for a synthetic shard.
#[derive(Clone)]
struct FixtureTensor {
    /// Tensor key written into the shard header.
    name: String,
    /// Safetensors scalar representation of the payload.
    dtype: TensorDType,
    /// Tensor dimensions in checkpoint axis order.
    shape: Vec<u64>,
    /// Contiguous payload bytes in the declared wire dtype.
    bytes: Vec<u8>,
}

impl FixtureTensor {
    /// Builds a fixture, asserting its payload length matches shape and dtype.
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

/// Writes a padded safetensors header followed by fixtures in physical order.
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

/// Writes the shard index mapping tensor names to paths and payload-byte totals.
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

/// Serializes F32 bit patterns into little-endian checkpoint bytes.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

/// Serializes U16 values, including raw half-precision bits, in wire order.
fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Serializes signed I32 checkpoint values in little-endian order.
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Decodes native-endian F32 sink bytes for numerical comparisons.
fn tensor_f32(data: &TensorData) -> Vec<f32> {
    data.as_bytes()
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

/// Decodes native-endian U16 sink bytes without changing their bit patterns.
fn tensor_u16(data: &TensorData) -> Vec<u16> {
    data.as_bytes()
        .chunks_exact(2)
        .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

/// Injected sink failure identifying the rejected target or ownership rule.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("synthetic sink rejected `{0}`")]
struct SinkError(
    /// Rejected target name, optionally prefixed by the violated sink invariant.
    String,
);

/// Records delivered data and aliases while allowing selected callback failures.
#[derive(Default)]
struct CollectingSink {
    /// Successfully inserted owned tensor payloads by target name.
    tensors: BTreeMap<String, TensorData>,
    /// Alias names mapped to their installed owner names.
    aliases: BTreeMap<String, String>,
    /// Successful write and alias callbacks in execution order.
    events: Vec<String>,
    /// Target whose write callback should reject its payload.
    fail_write: Option<String>,
    /// Alias whose callback should fail before recording installation.
    fail_alias: Option<String>,
}

impl TensorSink for CollectingSink {
    /// Synthetic callback rejection returned to the executor.
    type Error = SinkError;

    /// Stores one payload unless its target is rejected or already installed.
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

    /// Records an alias only after its owner exists and failure injection passes.
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
    /// Real local-file reader whose I/O and validation remain under test.
    inner: LocalShardSource<'a>,
    /// Successful `(shard, tensor, byte_offset)` reads in execution order.
    reads: Vec<(String, String, u64)>,
}

impl TensorSource for TracingLocalSource<'_> {
    /// Delegates the physical read and records its descriptor after success.
    fn read_tensor(&mut self, source: &PlannedSource) -> crate::checkpoint::Result<SourceTensor> {
        let payload = self.inner.read_tensor(source)?;
        self.reads.push((
            source.shard.clone(),
            source.name.clone(),
            source.data_offsets[0],
        ));
        Ok(payload)
    }

    /// Closes the delegated reader's current shard.
    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.inner.close()
    }

    /// Returns the delegated reader's physical I/O accounting.
    fn stats(&self) -> SourceStats {
        self.inner.stats()
    }
}

/// Creates a temporary single-shard checkpoint and discovers its manifest.
fn single_manifest(tensors: &[FixtureTensor]) -> (TempDir, crate::checkpoint::CheckpointManifest) {
    let temp = TempDir::new().unwrap();
    write_shard(temp.path(), "model.safetensors", tensors);
    let manifest = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap();
    (temp, manifest)
}

/// Builds a strict identity mapping retaining every fixture's shape and dtype.
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

/// A fresh execution replaces stale applied/quantized records with its actual progress.
#[test]
fn executor_replaces_stale_runtime_report_records() {
    let tensors = [FixtureTensor::new(
        "weight",
        TensorDType::F32,
        [1],
        f32_bytes(&[2.0]),
    )];
    let (_temp, manifest) = single_manifest(&tensors);
    let mut plan = identity_plan(&manifest, &tensors);
    plan.report.applied.push("stale.target".into());
    plan.report.quantized.push(QuantizationRecord {
        target: "stale.target".into(),
        scheme: "stale-scheme".into(),
        bytes: 1,
    });
    let mut source = LocalShardSource::new(&manifest);
    let mut sink = CollectingSink::default();

    let outcome = execute_load_plan(&plan, &mut source, &mut sink).unwrap();

    assert_eq!(outcome.report.applied, ["weight"]);
    assert!(outcome.report.quantized.is_empty());
    assert_eq!(tensor_f32(&sink.tensors["weight"]), [2.0]);
}

/// Byte-equivalent physical aliases are verified and accounted before installing their owner.
#[test]
fn physical_checkpoint_equivalence_is_verified_before_model_writes() {
    let tensors = vec![
        FixtureTensor::new(
            "shared.weight",
            TensorDType::F32,
            [2],
            f32_bytes(&[1.0, -2.0]),
        ),
        FixtureTensor::new(
            "decoder.embed_tokens.weight",
            TensorDType::F32,
            [2],
            f32_bytes(&[1.0, -2.0]),
        ),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let expected = [ExpectedTensor::new("shared.weight", [2]).with_dtype(TensorDType::F32)];
    let options = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new(
            "decoder.embed_tokens.weight",
            "shared.weight",
        )],
        ..LoadOptions::default()
    };
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    let record = CheckpointEquivalenceRecord::new("decoder.embed_tokens.weight", "shared.weight");
    assert_eq!(plan.equivalences.len(), 1);
    assert_eq!(
        plan.report.equivalence_planned,
        std::slice::from_ref(&record)
    );
    assert!(plan.report.equivalence_verified.is_empty());
    assert!(plan.report.allowlisted_ignored.is_empty());
    assert_eq!(plan.memory.largest_equivalence_pair_bytes, 16);

    let mut source = TracingLocalSource {
        inner: LocalShardSource::new(&manifest),
        reads: Vec::new(),
    };
    let mut sink = CollectingSink::default();
    let outcome = execute_load_plan(&plan, &mut source, &mut sink).unwrap();

    assert_eq!(sink.events, ["write:shared.weight"]);
    assert_eq!(
        outcome.report.equivalence_planned,
        std::slice::from_ref(&record)
    );
    assert_eq!(outcome.report.equivalence_verified, [record]);
    assert_eq!(outcome.stats.equivalences_verified, 1);
    assert_eq!(outcome.stats.equivalence_source_reads, 2);
    assert_eq!(outcome.stats.equivalence_source_payload_bytes, 16);
    assert_eq!(outcome.stats.source_reads, 3);
    assert_eq!(outcome.stats.source_payload_bytes, 24);
    assert_eq!(outcome.stats.largest_equivalence_pair_bytes, 16);
    assert_eq!(outcome.stats.peak_host_staging_bytes, 16);
    assert_eq!(
        source
            .reads
            .iter()
            .map(|(_, name, _)| name.as_str())
            .collect::<Vec<_>>(),
        [
            "shared.weight",
            "decoder.embed_tokens.weight",
            "shared.weight"
        ]
    );
}

/// Unequal alias payloads fail preflight with read accounting and an untouched sink.
#[test]
fn physical_checkpoint_equivalence_mismatch_fails_before_any_sink_mutation() {
    let tensors = vec![
        FixtureTensor::new(
            "shared.weight",
            TensorDType::F32,
            [2],
            f32_bytes(&[1.0, -2.0]),
        ),
        FixtureTensor::new(
            "decoder.embed_tokens.weight",
            TensorDType::F32,
            [2],
            f32_bytes(&[1.0, 3.0]),
        ),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let expected = [ExpectedTensor::new("shared.weight", [2])];
    let options = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new(
            "decoder.embed_tokens.weight",
            "shared.weight",
        )],
        ..LoadOptions::default()
    };
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    let mut source = TracingLocalSource {
        inner: LocalShardSource::new(&manifest),
        reads: Vec::new(),
    };
    let mut sink = CollectingSink::default();

    let failure = *execute_load_plan(&plan, &mut source, &mut sink).unwrap_err();

    assert_eq!(
        failure.failed_target.as_deref(),
        Some("decoder.embed_tokens.weight")
    );
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::TensorEquivalenceMismatch {
            ref alias,
            ref owner,
            ..
        }) if alias == "decoder.embed_tokens.weight" && owner == "shared.weight"
    ));
    assert!(sink.events.is_empty());
    assert!(sink.tensors.is_empty());
    assert!(failure.report.applied.is_empty());
    assert!(failure.report.equivalence_verified.is_empty());
    assert_eq!(failure.stats.equivalences_verified, 0);
    assert_eq!(failure.stats.equivalence_source_reads, 2);
    assert_eq!(failure.stats.equivalence_source_payload_bytes, 16);
    assert_eq!(failure.stats.source_reads, 2);
}

/// A later alias mismatch retains only earlier verified pairs and installs no tensors.
#[test]
fn equivalence_preflight_reports_only_verified_prefix_and_still_writes_nothing() {
    let tensors = vec![
        FixtureTensor::new("owner", TensorDType::F32, [1], f32_bytes(&[7.0])),
        FixtureTensor::new("a.copy", TensorDType::F32, [1], f32_bytes(&[7.0])),
        FixtureTensor::new("z.copy", TensorDType::F32, [1], f32_bytes(&[8.0])),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let expected = [ExpectedTensor::new("owner", [1])];
    let options = LoadOptions {
        index_aliases: vec![
            IndexTensorAlias::new("z.copy", "owner"),
            IndexTensorAlias::new("a.copy", "owner"),
        ],
        ..LoadOptions::default()
    };
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    let mut source = LocalShardSource::new(&manifest);
    let mut sink = CollectingSink::default();

    let failure = *execute_load_plan(&plan, &mut source, &mut sink).unwrap_err();

    assert!(sink.events.is_empty());
    assert_eq!(
        failure.report.equivalence_verified,
        [CheckpointEquivalenceRecord::new("a.copy", "owner")]
    );
    assert_eq!(failure.stats.equivalences_verified, 1);
    assert_eq!(failure.stats.equivalence_source_reads, 4);
    assert_eq!(failure.stats.source_reads, 4);
    assert_eq!(failure.stats.source_payload_bytes, 16);
}

/// Absent optional aliases are ignored; a present physical alias needs its declared owner.
#[test]
fn equivalence_declarations_are_optional_but_physical_aliases_require_an_owner() {
    let mapped = FixtureTensor::new("mapped", TensorDType::F32, [1], f32_bytes(&[1.0]));
    let (_temp, manifest) = single_manifest(std::slice::from_ref(&mapped));
    let expected = [ExpectedTensor::new("mapped", [1])];
    let absent_alias = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new("optional.copy", "mapped")],
        ..LoadOptions::default()
    };
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &absent_alias,
    )
    .unwrap();
    assert!(plan.equivalences.is_empty());
    assert!(plan.report.equivalence_planned.is_empty());

    let tensors = vec![
        mapped,
        FixtureTensor::new("physical.copy", TensorDType::F32, [1], f32_bytes(&[1.0])),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let absent_owner = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new("physical.copy", "missing.owner")],
        ..LoadOptions::default()
    };
    assert!(matches!(
        build_load_plan(
            &manifest,
            &expected,
            &MappingPlan::identity(&expected),
            &absent_owner,
        ),
        Err(CheckpointError::TensorEquivalenceMismatch {
            ref alias,
            ref owner,
            ..
        }) if alias == "physical.copy" && owner == "missing.owner"
    ));
}

/// Incompatible alias metadata and tampered plans reject before any source read or sink call.
#[test]
fn equivalence_shape_dtype_and_public_plan_mutations_fail_closed() {
    for alias in [
        FixtureTensor::new("copy", TensorDType::F32, [2], f32_bytes(&[1.0, 2.0])),
        FixtureTensor::new("copy", TensorDType::I32, [1], i32_bytes(&[1])),
    ] {
        let tensors = vec![
            FixtureTensor::new("owner", TensorDType::F32, [1], f32_bytes(&[1.0])),
            alias,
        ];
        let (_temp, manifest) = single_manifest(&tensors);
        let expected = [ExpectedTensor::new("owner", [1])];
        let options = LoadOptions {
            index_aliases: vec![IndexTensorAlias::new("copy", "owner")],
            ..LoadOptions::default()
        };
        assert!(matches!(
            build_load_plan(
                &manifest,
                &expected,
                &MappingPlan::identity(&expected),
                &options,
            ),
            Err(CheckpointError::TensorEquivalenceMismatch { .. })
        ));
    }

    let tensors = vec![
        FixtureTensor::new("owner", TensorDType::F32, [1], f32_bytes(&[1.0])),
        FixtureTensor::new("copy", TensorDType::F32, [1], f32_bytes(&[1.0])),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let expected = [ExpectedTensor::new("owner", [1])];
    let options = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new("copy", "owner")],
        ..LoadOptions::default()
    };
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    let assert_preflight_failure =
        |plan: &crate::checkpoint::LoadPlan, expected_error: fn(&CheckpointError) -> bool| {
            let mut source = LocalShardSource::new(&manifest);
            let mut sink = CollectingSink::default();
            let failure = *execute_load_plan(plan, &mut source, &mut sink).unwrap_err();
            let ExecutionFailureCause::Checkpoint(error) = &failure.cause else {
                panic!("plan mutation unexpectedly reached the sink")
            };
            assert!(expected_error(error), "unexpected preflight error: {error}");
            assert_eq!(failure.stats.source_reads, 0);
            assert_eq!(source.stats().tensor_reads, 0);
            assert!(sink.events.is_empty());
        };

    let mut missing_report = plan.clone();
    missing_report.report.equivalence_planned.clear();
    assert_preflight_failure(&missing_report, |error| {
        matches!(error, CheckpointError::InvalidMapping { .. })
    });

    let mut prematurely_verified = plan.clone();
    prematurely_verified.report.equivalence_verified =
        prematurely_verified.report.equivalence_planned.clone();
    assert_preflight_failure(&prematurely_verified, |error| {
        matches!(error, CheckpointError::InvalidMapping { .. })
    });

    let mut understated_memory = plan.clone();
    understated_memory.memory.largest_equivalence_pair_bytes = 0;
    assert_preflight_failure(&understated_memory, |error| {
        matches!(error, CheckpointError::InvalidMapping { .. })
    });

    let mut invalid_descriptor = plan.clone();
    invalid_descriptor.equivalences[0].alias.shape = vec![2];
    assert_preflight_failure(&invalid_descriptor, |error| {
        matches!(error, CheckpointError::InvalidMapping { .. })
    });

    let mut changed_dtype = plan.clone();
    changed_dtype.equivalences[0].alias.dtype = TensorDType::I32;
    assert_preflight_failure(&changed_dtype, |error| {
        matches!(error, CheckpointError::TensorEquivalenceMismatch { .. })
    });

    let mut duplicate_work = plan.clone();
    duplicate_work
        .equivalences
        .push(duplicate_work.equivalences[0].clone());
    assert_preflight_failure(&duplicate_work, |error| {
        matches!(error, CheckpointError::TensorEquivalenceMismatch { .. })
    });
}

/// Identity and renamed mappings retain numeric values, shapes, and native Burn dtypes.
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

/// Interleaved mappings read each shard in physical order while reporting targets consistently.
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

/// Each runtime-supported wire dtype becomes its matching Burn scalar representation.
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

/// Installs one named rank-two payload directly into a Flex parameter.
struct RankTwoParamSink {
    /// Parameter populated by a successful write callback.
    parameter: Option<Param<Tensor<Flex, 2>>>,
}

impl TensorSink for RankTwoParamSink {
    /// Rejection of a target name, tensor rank, or unsupported alias.
    type Error = SinkError;

    /// Constructs `weight` as a rank-two parameter after checking its metadata.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        if target != "weight" || data.shape.as_slice().len() != 2 {
            return Err(SinkError(target.into()));
        }
        let device = Default::default();
        self.parameter = Some(Param::from_data(data, &device));
        Ok(())
    }

    /// Rejects aliases because this fixture owns only one independent parameter.
    fn alias_tensor(&mut self, target: &str, _source_target: &str) -> Result<(), Self::Error> {
        Err(SinkError(target.into()))
    }
}

/// Delivered rank-two data can initialize a Flex parameter without another mapping layer.
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

/// A rectangular transpose swaps axes, reorders payload values, and records both live buffers.
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

/// Fused BF16-to-F16 transposition produces the expected half bits in destination axis order.
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

/// Splitting any rank-three axis preserves slice order, including empty slices, with one source read.
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

/// A rejected second slice preserves the first output and never rereads their shared source.
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

/// Concatenation preserves all rank-three axis layouts while limiting open shards to one.
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

/// A concatenation owns one complete output: a late input failure cannot publish its prefix.
#[test]
fn atomicity_concat_late_failures_preserve_only_prior_complete_targets() {
    for malformed_bool in [false, true] {
        let tensors = [
            FixtureTensor::new("prior", TensorDType::BOOL, [1], vec![1]),
            FixtureTensor::new("first", TensorDType::BOOL, [1], vec![0]),
            FixtureTensor::new(
                "last",
                TensorDType::BOOL,
                [2],
                vec![1, if malformed_bool { 2 } else { 0 }],
            ),
            FixtureTensor::new("later", TensorDType::BOOL, [1], vec![1]),
        ];
        let (_temp, manifest) = single_manifest(&tensors);
        let expected = [
            ExpectedTensor::new("prior", [1]).with_dtype(TensorDType::BOOL),
            ExpectedTensor::new("joined", [3]).with_dtype(TensorDType::BOOL),
            ExpectedTensor::new("later", [1]).with_dtype(TensorDType::BOOL),
            ExpectedTensor::new("tied", [1]).with_dtype(TensorDType::BOOL),
        ];
        let mut mapping = MappingPlan::new();
        mapping
            .map("prior", "prior")
            .concatenate(["first".into(), "last".into()], "joined", 0)
            .map("later", "later")
            .alias("prior", "tied");
        let plan =
            build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
        let mut source = MemorySource {
            tensors: tensors
                .iter()
                .map(|tensor| {
                    (
                        tensor.name.clone(),
                        SourceTensor::new(
                            &tensor.name,
                            tensor.shape.clone(),
                            tensor.dtype,
                            tensor.bytes.clone(),
                        )
                        .unwrap(),
                    )
                })
                .collect(),
            fail_at: (!malformed_bool).then(|| "last".into()),
            ..MemorySource::default()
        };
        let mut sink = CollectingSink::default();
        let failure = execute_load_plan(&plan, &mut source, &mut sink).unwrap_err();
        assert_eq!(failure.failed_target.as_deref(), Some("joined"));
        assert!(matches!(&failure.cause,
            ExecutionFailureCause::Checkpoint(CheckpointError::InvalidTensorPayload { tensor, reason })
            if tensor == if malformed_bool { "joined" } else { "last" }
                && reason == if malformed_bool { "boolean element 2 has invalid byte value 2" } else { "synthetic read failure" }));
        assert_eq!(failure.report.applied, ["prior"]);
        assert_eq!(failure.stats.tensors_applied, 1);
        assert_eq!(failure.stats.aliases_applied, 0);
        assert_eq!(
            failure.stats.source_reads,
            if malformed_bool { 3 } else { 2 }
        );
        assert_eq!(
            failure.stats.source_payload_bytes,
            if malformed_bool { 4 } else { 2 }
        );
        assert_eq!(failure.stats.delivered_tensor_bytes, 1);
        // The output's three bytes overlap only the current one- or two-byte input.
        assert_eq!(
            failure.stats.peak_host_staging_bytes,
            if malformed_bool { 5 } else { 4 }
        );
        assert_eq!(failure.stats.largest_output_tensor_bytes, 3);
        assert_eq!(source.close_calls, 1);
        assert_eq!(sink.events, ["write:prior"]);
        assert_eq!(sink.tensors.len(), 1);
        assert_eq!(sink.tensors["prior"].to_vec::<bool>().unwrap(), [true]);
        assert!(sink.aliases.is_empty());
    }
}

/// Physical-equivalence validation needs both raw payloads concurrently, including at the exact bound.
#[test]
fn atomicity_equivalence_pair_budget_rejects_before_reads_and_accepts_exactly() {
    let tensors = [
        FixtureTensor::new("owner", TensorDType::F32, [2], f32_bytes(&[1.0, -2.0])),
        FixtureTensor::new("copy", TensorDType::F32, [2], f32_bytes(&[1.0, -2.0])),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let expected = [ExpectedTensor::new("owner", [2])];
    let options = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new("copy", "owner")],
        ..LoadOptions::default()
    };
    let original = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    assert_eq!(original.memory.largest_equivalence_pair_bytes, 16);
    for bound in [15, 16] {
        let mut plan = original.clone();
        plan.memory.peak_host_staging_bytes = bound;
        let mut source = MemorySource {
            tensors: tensors
                .iter()
                .map(|tensor| {
                    (
                        tensor.name.clone(),
                        SourceTensor::new(
                            &tensor.name,
                            tensor.shape.clone(),
                            tensor.dtype,
                            tensor.bytes.clone(),
                        )
                        .unwrap(),
                    )
                })
                .collect(),
            ..MemorySource::default()
        };
        let mut sink = CollectingSink::default();
        let result = execute_load_plan(&plan, &mut source, &mut sink);
        assert_eq!(source.close_calls, 1);
        if bound == 15 {
            let failure = result.unwrap_err();
            assert!(matches!(failure.cause,
                ExecutionFailureCause::Checkpoint(CheckpointError::MemoryEstimateExceeded { ref tensor, required: 16, planned: 15 }) if tensor == "copy"));
            assert_eq!(failure.failed_target, None);
            assert_eq!(failure.stats, ExecutionStats::default());
            assert!(failure.report.applied.is_empty());
            assert!(failure.report.equivalence_verified.is_empty());
            assert!(sink.events.is_empty());
        } else {
            let outcome = result.unwrap();
            assert_eq!(outcome.stats.equivalences_verified, 1);
            assert_eq!(outcome.stats.equivalence_source_reads, 2);
            assert_eq!(outcome.stats.equivalence_source_payload_bytes, 16);
            assert_eq!(outcome.stats.source_reads, 3);
            assert_eq!(outcome.stats.source_payload_bytes, 24);
            assert_eq!(outcome.stats.largest_equivalence_pair_bytes, 16);
            assert_eq!(outcome.stats.peak_host_staging_bytes, 16);
            assert_eq!(
                outcome.report.equivalence_verified,
                [CheckpointEquivalenceRecord::new("copy", "owner")]
            );
            assert_eq!(outcome.report.applied, ["owner"]);
            assert_eq!(sink.events, ["write:owner"]);
            assert_eq!(tensor_f32(&sink.tensors["owner"]), [1.0, -2.0]);
        }
    }
}

/// All BF16 bit patterns match half conversion semantics, treating NaNs by classification.
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

/// A reused eight-byte cast buffer still requires an eight-byte staging budget.
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

/// All F16 bit patterns match BF16 conversion semantics with explicit NaN handling.
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
    // Equal-width F16 -> BF16 converts the 131,072-byte source in place.
    assert_eq!(outcome.stats.peak_host_staging_bytes, 131_072);
}

/// An unchanged F32 payload records one live buffer rather than source-plus-output storage.
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

/// Sequential casts retain only the current source/output pair in peak staging accounting.
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

/// Alias chains resolve directly to their owner and install after physical payload writes.
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

/// A failed write reports only previously accepted tensors and identifies its target.
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

/// Alias rejection retains prior owner progress without claiming uninstalled aliases.
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

/// In-memory payload source with injected read failures and cleanup accounting.
#[derive(Default)]
struct MemorySource {
    /// Available payloads keyed by planned source name.
    tensors: BTreeMap<String, SourceTensor>,
    /// Number of close callbacks performed by the executor.
    close_calls: usize,
    /// Source name whose read callback should fail.
    fail_at: Option<String>,
}

/// Sink that permits aliases only after the paired source reports closure.
struct CloseAwareSink {
    /// Shared closure signal written by the corresponding source.
    closed: Arc<AtomicBool>,
    /// Whether a primary tensor write has completed.
    owner_written: bool,
    /// Whether an alias callback passed the closure check.
    alias_written: bool,
}

impl TensorSink for CloseAwareSink {
    /// Synthetic rejection used when alias installation precedes source closure.
    type Error = SinkError;

    /// Records successful installation of the primary tensor.
    fn write_tensor(&mut self, _target: &str, _data: TensorData) -> Result<(), Self::Error> {
        self.owner_written = true;
        Ok(())
    }

    /// Rejects an alias if source cleanup has not yet signaled completion.
    fn alias_tensor(&mut self, target: &str, _source_target: &str) -> Result<(), Self::Error> {
        if !self.closed.load(Ordering::SeqCst) {
            return Err(SinkError(format!("source-open:{target}")));
        }
        self.alias_written = true;
        Ok(())
    }
}

/// In-memory source exposing when cleanup occurs and optionally failing it.
struct CloseAwareSource {
    /// Payload provider used for read operations.
    inner: MemorySource,
    /// Closure signal observed by the paired sink.
    closed: Arc<AtomicBool>,
    /// Whether the close callback returns a synthetic checkpoint error.
    fail_close: bool,
}

impl TensorSource for CloseAwareSource {
    /// Reads a planned payload through the wrapped in-memory source.
    fn read_tensor(&mut self, source: &PlannedSource) -> crate::checkpoint::Result<SourceTensor> {
        self.inner.read_tensor(source)
    }

    /// Signals closure and optionally reports the injected cleanup failure.
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

/// Creates one F32 owner and one alias backed by an in-memory source.
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

/// Alias callbacks observe completed source cleanup after the primary tensor write.
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

/// Source-close failure preserves the installed owner but prevents alias callbacks.
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
    /// Clones a named payload or returns an injected/missing-payload diagnostic.
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

    /// Counts cleanup calls without releasing the reusable in-memory fixtures.
    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.close_calls += 1;
        Ok(())
    }
}

/// Mismatched custom-source descriptors are rejected, and the source is still closed.
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

/// A later read failure retains earlier successful writes and executes source cleanup.
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

/// Payload construction rejects truncated bytes and shape products exceeding u64.
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

/// Boolean bytes outside zero and one cannot reach a tensor sink.
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

/// A valid FP8 wire dtype unsupported by this runtime is rejected before opening a shard.
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

/// A public plan with a changed destination shape fails preflight without I/O or writes.
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

/// A cast whose source/output pair exceeds its declared staging budget cannot reach the sink.
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

/// Changing a planned unsigned dtype to signed cannot reinterpret the same payload bytes.
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

/// Appending bytes after discovery invalidates the previously inspected checkpoint artifact.
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

/// Changing the on-disk header length after discovery invalidates the artifact.
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

/// A local reader rechecks planned metadata against the discovered shard descriptor.
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

/// Explicit selection loads the named nondefault shard and records that physical path.
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

/// Executes one equal-dtype transpose through the complete in-memory runtime.
fn in_memory_transpose_data(
    rows: u64,
    columns: u64,
    dtype: TensorDType,
    bytes: Vec<u8>,
) -> TensorData {
    in_memory_cast_transpose_data(rows, columns, dtype, dtype, bytes)
}

/// Executes one planned `Transpose2d` from `source_dtype` to `target_dtype`
/// through the complete in-memory runtime.
///
/// The plan's staging budget is the source payload plus the target buffer,
/// the two buffers `materialize_transpose` holds at once.
fn in_memory_cast_transpose_data(
    rows: u64,
    columns: u64,
    source_dtype: TensorDType,
    target_dtype: TensorDType,
    bytes: Vec<u8>,
) -> TensorData {
    let payload_bytes = source_dtype.storage_bytes(rows * columns).unwrap();
    let staging_bytes = payload_bytes + target_dtype.storage_bytes(rows * columns).unwrap();
    let source = PlannedSource {
        name: "source".into(),
        shard: "memory.safetensors".into(),
        shape: vec![rows, columns],
        dtype: source_dtype,
        data_offsets: [0, payload_bytes],
    };
    let target = PlannedTensor {
        target: "target".into(),
        sources: vec![source],
        transform: PlannedTransform::Transpose2d,
        shape: vec![columns, rows],
        dtype: target_dtype,
    };
    let report = crate::checkpoint::LoadReport {
        memory: Some(crate::checkpoint::MemoryEstimate {
            peak_host_staging_bytes: staging_bytes,
            ..crate::checkpoint::MemoryEstimate::default()
        }),
        ..crate::checkpoint::LoadReport::default()
    };
    let plan = crate::checkpoint::LoadPlan {
        equivalences: Vec::new(),
        tensors: vec![target],
        aliases: Vec::new(),
        report,
        memory: crate::checkpoint::MemoryEstimate {
            peak_host_staging_bytes: staging_bytes,
            ..crate::checkpoint::MemoryEstimate::default()
        },
    };
    let mut source = MemorySource::default();
    source.tensors.insert(
        "source".into(),
        SourceTensor::new("source", [rows, columns], source_dtype, bytes).unwrap(),
    );
    let mut sink = CollectingSink::default();
    execute_load_plan(&plan, &mut source, &mut sink).unwrap();
    sink.tensors.remove("target").unwrap()
}

/// A planned BF16-to-F32 `Transpose2d` above the threaded-transpose threshold
/// (`[1031, 1037]`, ragged final band) converts and places every scalar exactly.
///
/// Every finite BF16 value widens exactly to F32 by appending sixteen zero
/// bits. Patterns with an all-ones exponent are folded to finite ones (NaN
/// quieting is covered by the conversion tests), so the oracle is that shift
/// applied at the naively transposed position.
#[cfg(target_endian = "little")]
#[test]
fn planned_large_converting_transpose_is_exact_above_the_thread_threshold() {
    let (rows, columns) = (1031_usize, 1037_usize);
    let values = (0..rows * columns)
        .map(|index| {
            // A multiplicative hash spreads indices over the BF16 bit patterns.
            let bits = (index as u32).wrapping_mul(0x9E37_79B9).rotate_left(7) as u16;
            if bits & 0x7F80 == 0x7F80 {
                bits & !0x0080
            } else {
                bits
            }
        })
        .collect::<Vec<_>>();
    let data = in_memory_cast_transpose_data(
        rows as u64,
        columns as u64,
        TensorDType::BF16,
        TensorDType::F32,
        u16_bytes(&values),
    );
    assert_eq!(data.shape.as_slice(), [columns, rows]);
    let output = data.as_slice::<f32>().unwrap();
    for (index, value) in output.iter().enumerate() {
        let (column, row) = (index / rows, index % rows);
        let expected = u32::from(values[row * columns + column]) << 16;
        assert_eq!(value.to_bits(), expected, "target ({column}, {row})");
    }
}

/// Executes the numeric `u16` transpose used by the index-equation property test.
fn in_memory_transpose(rows: u64, columns: u64, values: Vec<u16>) -> Vec<u16> {
    let data = in_memory_transpose_data(rows, columns, TensorDType::U16, u16_bytes(&values));
    tensor_u16(&data)
}

#[cfg(target_endian = "little")]
/// Computes a deliberately simple scalar-by-scalar byte-transpose oracle.
fn reference_transpose_bytes(rows: usize, columns: usize, width: usize, bytes: &[u8]) -> Vec<u8> {
    let mut output = vec![0; bytes.len()];
    for row in 0..rows {
        for column in 0..columns {
            let source = (row * columns + column) * width;
            let target = (column * rows + row) * width;
            output[target..target + width].copy_from_slice(&bytes[source..source + width]);
        }
    }
    output
}

#[cfg(target_endian = "little")]
/// Encodes each scalar index into its own little-endian raw-bit pattern.
///
/// The odd multiplier is bijective modulo every supported power-of-two scalar
/// width. In particular, advancing by the 32-scalar tile edge changes the low
/// byte, unlike a byte-index pattern whose eight-byte scalars repeat every tile.
fn indexed_scalar_bytes(elements: usize, width: usize, seed: u8) -> Vec<u8> {
    assert!(matches!(width, 1 | 2 | 4 | 8));
    let mut bytes = Vec::with_capacity(elements.checked_mul(width).unwrap());
    let salt = u64::from(seed).wrapping_mul(0x0101_0101_0101_0101);
    for index in 0..elements {
        let encoded = (index as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(salt)
            .to_le_bytes();
        bytes.extend_from_slice(&encoded[..width]);
    }
    bytes
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Random small U16 matrices match the scalar row-major transpose equation.
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

    /// Random shapes and scalar widths match a byte-copy oracle across transpose tiles.
    #[cfg(target_endian = "little")]
    #[test]
    fn tiled_equal_dtype_transpose_is_bit_exact(
        rows in 0_usize..70,
        columns in 0_usize..70,
        width_selector in 0_usize..4,
        seed in any::<u8>(),
    ) {
        let (dtype, width) = [
            (TensorDType::U8, 1_usize),
            (TensorDType::U16, 2),
            (TensorDType::U32, 4),
            (TensorDType::U64, 8),
        ][width_selector];
        let bytes = indexed_scalar_bytes(rows * columns, width, seed);
        let expected = reference_transpose_bytes(rows, columns, width, &bytes);
        let actual = in_memory_transpose_data(
            rows as u64,
            columns as u64,
            dtype,
            bytes,
        );
        prop_assert_eq!(actual.as_bytes(), expected);
    }
}

/// Empty axes and sizes adjacent to each tile boundary preserve every scalar byte.
#[cfg(target_endian = "little")]
#[test]
fn tiled_transpose_covers_zero_and_every_tile_boundary() {
    let dimensions = [0_usize, 1, 31, 32, 33, 63, 64, 65];
    let dtypes = [
        (TensorDType::U8, 1_usize),
        (TensorDType::U16, 2),
        (TensorDType::U32, 4),
        (TensorDType::U64, 8),
    ];

    for rows in dimensions {
        for columns in dimensions {
            for (dtype, width) in dtypes {
                let bytes = indexed_scalar_bytes(rows * columns, width, 91);
                let expected = reference_transpose_bytes(rows, columns, width, &bytes);
                let actual = in_memory_transpose_data(rows as u64, columns as u64, dtype, bytes);

                assert_eq!(
                    actual.as_bytes(),
                    expected,
                    "dtype {dtype}, source shape [{rows}, {columns}]"
                );
            }
        }
    }
}

/// F32 transposition preserves NaN payloads, signed zero, infinities, and edge-tile bits.
#[cfg(target_endian = "little")]
#[test]
fn tiled_transpose_preserves_float_nan_payloads_signed_zero_and_tile_edges() {
    let rows = 35_usize;
    let columns = 67_usize;
    let patterns = [
        0x0000_0000_u32,
        0x8000_0000,
        0x7fc0_0001,
        0x7fa1_2345,
        0xffc5_4321,
        0x7f80_0000,
        0xff80_0000,
        0x0000_0001,
    ];
    let bytes: Vec<_> = (0..rows * columns)
        .flat_map(|index| patterns[index % patterns.len()].to_le_bytes())
        .collect();
    let expected = reference_transpose_bytes(rows, columns, 4, &bytes);

    let actual = in_memory_transpose_data(rows as u64, columns as u64, TensorDType::F32, bytes);

    assert_eq!(actual.as_bytes(), expected);
}

/// Every runtime dtype matches the byte-transpose oracle across partial tiles.
#[cfg(target_endian = "little")]
#[test]
fn tiled_transpose_is_bit_exact_for_every_runtime_dtype() {
    let rows = 33_usize;
    let columns = 35_usize;
    let dtypes = [
        (TensorDType::BOOL, 1_usize),
        (TensorDType::U8, 1),
        (TensorDType::I8, 1),
        (TensorDType::U16, 2),
        (TensorDType::I16, 2),
        (TensorDType::F16, 2),
        (TensorDType::BF16, 2),
        (TensorDType::U32, 4),
        (TensorDType::I32, 4),
        (TensorDType::F32, 4),
        (TensorDType::U64, 8),
        (TensorDType::I64, 8),
        (TensorDType::F64, 8),
    ];

    for (dtype, width) in dtypes {
        let bytes = if dtype == TensorDType::BOOL {
            (0..rows * columns)
                .map(|index| u8::from(index % 67 < 29))
                .collect()
        } else {
            indexed_scalar_bytes(rows * columns, width, 17)
        };
        let expected = reference_transpose_bytes(rows, columns, width, &bytes);
        let actual = in_memory_transpose_data(rows as u64, columns as u64, dtype, bytes);

        assert_eq!(actual.as_bytes(), expected, "dtype {dtype}");
    }
}

/// An explicit dtype policy converts finite and infinite F64 values to expected BF16 bits.
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

/// Column-layout installation applies the checkpoint mapper before the tested linear forward.
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

/// Loading applies the checkpoint transpose before restoring trainable-leaf policy.
#[cfg(feature = "autodiff")]
#[test]
fn parameter_install_preserves_autodiff_leaf_and_frozen_layouts() {
    use burn::{backend::Autodiff, module::Module};

    /// Flex backend with autodiff for checking parameter leaf and freeze policies.
    type AdBackend = Autodiff<Flex>;
    let device = Default::default();
    for layout in [LinearLayout::Row, LinearLayout::Col] {
        for frozen in [false, true] {
            let mut projection = crate::operations::init_linear::<AdBackend>(
                &LinearConfig::new(2, 3).with_bias(false).with_layout(layout),
                &device,
            );
            if frozen {
                projection.weight = projection.weight.set_require_grad(false);
            }
            assert!(!projection.weight.is_initialized());
            let source_shape = match layout {
                LinearLayout::Row => [2, 3],
                LinearLayout::Col => [3, 2],
            };
            let source = TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], source_shape);
            install_float_parameter(&mut projection.weight, source.clone()).unwrap();
            assert_eq!(projection.weight.val().dims(), [2, 3]);
            assert_eq!(projection.weight.val().is_require_grad(), !frozen);
            assert_eq!(projection.weight.clone().into_record().to_data(), source);

            let input = Tensor::<AdBackend, 2>::from_floats([[1.0, 2.0]], &device).require_grad();
            let gradients = crate::operations::linear(input.clone(), &projection)
                .unwrap()
                .sum()
                .backward();
            if frozen {
                assert!(projection.weight.grad(&gradients).is_none());
            } else {
                assert_eq!(
                    projection.weight.grad(&gradients).unwrap().to_data(),
                    TensorData::from([[1.0_f32, 1.0, 1.0], [2.0, 2.0, 2.0]])
                );
            }
            assert!(input.grad(&gradients).is_some());
        }
    }
}

/// Installing Q8 data retains packed storage while applying the parameter layout contract.
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

/// Wrong rank, shape, or nonfloat dtype leaves a lazy parameter uninitialized.
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

/// Public TensorData fields cannot smuggle truncated float storage past lazy-shape checks.
#[test]
fn parameter_install_rejects_truncated_float_storage_without_mutation() {
    let device = Default::default();
    let mut linear = LinearConfig::new(2, 3)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init::<Flex>(&device);
    for elements in [0, 5, 7] {
        let mut malformed = TensorData::new(vec![0.0_f32; elements], [elements]);
        malformed.shape = vec![3, 2].into();
        let error = install_float_parameter(&mut linear.weight, malformed)
            .expect_err("invalid byte length must be a recoverable error");
        assert_eq!(
            error,
            ParameterInstallError::PayloadLength {
                expected: 24,
                actual: elements * 4,
            }
        );
        assert!(
            error
                .to_string()
                .contains("expected 24 checkpoint float bytes")
        );
        assert!(!linear.weight.is_initialized());
    }
    let mut unsupported = TensorData::new(vec![0.0_f32; 6], [3, 2]);
    unsupported.dtype = DType::Flex32;
    let error = install_float_parameter(&mut linear.weight, unsupported)
        .expect_err("Flex does not advertise Flex32 storage");
    assert!(matches!(
        error,
        ParameterInstallError::UnsupportedStorage { .. }
    ));
    assert!(
        error
            .to_string()
            .contains("cannot store checkpoint dtype Flex32")
    );
    assert!(!linear.weight.is_initialized());
}

/// Dense byte equations cover scalar, empty, half, double, and overflow boundaries.
#[test]
fn dense_parameter_payload_equations_fail_before_allocation() {
    for dtype in [
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::Flex32,
        DType::F64,
    ] {
        for (shape, elements) in [(vec![], 1), (vec![0, 4], 0), (vec![2, 3], 6)] {
            let bytes = elements * dtype.size();
            validate_float_payload_length(&shape, dtype, bytes).unwrap();
            let error = validate_float_payload_length(&shape, dtype, bytes + 1).unwrap_err();
            assert_eq!(
                error,
                ParameterInstallError::PayloadLength {
                    expected: bytes,
                    actual: bytes + 1
                }
            );
        }
    }
    for shape in [vec![usize::MAX, 2], vec![usize::MAX]] {
        let error = validate_float_payload_length(&shape, DType::F32, 0).unwrap_err();
        assert_eq!(error, ParameterInstallError::PayloadSizeOverflow);
        assert!(error.to_string().contains("overflows usize"));
    }
}

/// Reinstallation must compare checkpoint layout, not an initialized runtime shape.
#[test]
fn runtime_coverage_initialized_column_parameter_keeps_checkpoint_shape_contract() {
    let device = Default::default();
    let mut projection = LinearConfig::new(2, 3)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init::<Flex>(&device);
    let original = TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
    install_float_parameter(&mut projection.weight, original.clone()).unwrap();
    let before = projection.weight.val().to_data();

    let error = install_float_parameter(
        &mut projection.weight,
        TensorData::new(vec![9.0_f32; 6], [2, 3]),
    )
    .expect_err("runtime-oriented metadata must not be treated as checkpoint layout");
    assert_eq!(
        error,
        ParameterInstallError::Shape {
            expected: vec![3, 2],
            actual: vec![2, 3],
        }
    );
    assert_eq!(projection.weight.val().to_data(), before);

    let replacement = TensorData::new(vec![6.0_f32, 5.0, 4.0, 3.0, 2.0, 1.0], [3, 2]);
    install_float_parameter(&mut projection.weight, replacement.clone()).unwrap();
    assert_eq!(projection.weight.val().dims(), [2, 3]);
    assert_eq!(
        projection.weight.clone().into_record().to_data(),
        replacement
    );
    assert_eq!(
        projection
            .forward(Tensor::<Flex, 2>::from_floats([[10.0, 100.0]], &device))
            .to_data(),
        TensorData::from([[560.0_f32, 340.0, 120.0]])
    );
}

/// Checkpoint installation uses runtime dtype without executing lazy initialization.
#[test]
fn runtime_coverage_lazy_parameter_install_preserves_each_native_float_dtype() {
    let device = Default::default();
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let initialized = Arc::new(AtomicBool::new(false));
        let observed = initialized.clone();
        let mut parameter = Param::<Tensor<Flex, 2>>::uninitialized(
            ParamId::new(),
            move |device, _| {
                observed.store(true, Ordering::SeqCst);
                Tensor::zeros([2, 2], device)
            },
            device,
            false,
            [2, 2].into(),
        );
        let expected = TensorData::new(
            vec![1.0_f64 + 2.0_f64.powi(-40), -0.0, -1.5, 65504.0],
            [2, 2],
        )
        .convert_dtype(dtype);
        install_float_parameter(&mut parameter, expected.clone()).unwrap();
        assert!(!initialized.load(Ordering::SeqCst));
        assert_eq!(parameter.val().dtype(), dtype);
        assert_eq!(parameter.val().to_data(), expected);
    }
}

/// Public shape overflow is recoverable before a lazy allocation or backend call.
#[test]
fn runtime_coverage_parameter_size_overflow_leaves_lazy_storage_uninitialized() {
    let initialized = Arc::new(AtomicBool::new(false));
    let observed = initialized.clone();
    let mut parameter = Param::<Tensor<Flex, 2>>::uninitialized(
        ParamId::new(),
        move |device, _| {
            observed.store(true, Ordering::SeqCst);
            Tensor::zeros([1, 1], device)
        },
        Default::default(),
        false,
        [usize::MAX, 2].into(),
    );
    let mut data = TensorData::new(Vec::<f32>::new(), [0, 0]);
    data.shape = vec![usize::MAX, 2].into();
    assert_eq!(
        install_float_parameter(&mut parameter, data).unwrap_err(),
        ParameterInstallError::PayloadSizeOverflow
    );
    assert!(!parameter.is_initialized());
    assert!(!initialized.load(Ordering::SeqCst));
}

/// A real malformed fused Boolean slice fails after only the valid sibling write.
#[test]
fn runtime_coverage_fused_transform_failure_preserves_successful_slice_prefix() {
    let tensors = [FixtureTensor::new(
        "fused",
        TensorDType::BOOL,
        [4],
        vec![1, 0, 2, 0],
    )];
    let (_temp, manifest) = single_manifest(&tensors);
    let expected = [
        ExpectedTensor::new("a", [2]).with_dtype(TensorDType::BOOL),
        ExpectedTensor::new("b", [2]).with_dtype(TensorDType::BOOL),
    ];
    let mut mapping = MappingPlan::new();
    mapping.split("fused", 0, [("a".into(), 2), ("b".into(), 2)]);
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    let mut source = MemorySource {
        tensors: BTreeMap::from([(
            "fused".into(),
            SourceTensor::new("fused", [4], TensorDType::BOOL, vec![1, 0, 2, 0]).unwrap(),
        )]),
        ..MemorySource::default()
    };
    let mut sink = CollectingSink::default();
    let failure = execute_load_plan(&plan, &mut source, &mut sink).unwrap_err();
    assert_eq!(failure.failed_target.as_deref(), Some("b"));
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::InvalidTensorPayload { ref tensor, .. })
        if tensor == "b"
    ));
    assert_eq!(failure.report.applied, ["a"]);
    assert_eq!(failure.stats.tensors_applied, 1);
    assert_eq!(failure.stats.source_reads, 1);
    assert_eq!(failure.stats.source_payload_bytes, 4);
    assert_eq!(source.close_calls, 1);
    assert_eq!(sink.events, ["write:a"]);
    assert_eq!(sink.tensors["a"].to_vec::<bool>().unwrap(), [true, false]);
}

/// Closing a failed source cannot mask the original payload failure or prior writes.
#[test]
fn runtime_coverage_payload_failure_keeps_primary_cause_when_close_also_fails() {
    let tensors = [
        FixtureTensor::new("a", TensorDType::F32, [1], f32_bytes(&[1.0])),
        FixtureTensor::new("b", TensorDType::F32, [1], f32_bytes(&[2.0])),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let plan = identity_plan(&manifest, &tensors);
    let closed = Arc::new(AtomicBool::new(false));
    let mut source = CloseAwareSource {
        inner: MemorySource {
            tensors: BTreeMap::from([(
                "a".into(),
                SourceTensor::new("a", [1], TensorDType::F32, f32_bytes(&[1.0])).unwrap(),
            )]),
            fail_at: Some("b".into()),
            ..MemorySource::default()
        },
        closed: closed.clone(),
        fail_close: true,
    };
    let mut sink = CollectingSink::default();
    let failure = execute_load_plan(&plan, &mut source, &mut sink).unwrap_err();
    assert!(closed.load(Ordering::SeqCst));
    assert_eq!(failure.failed_target.as_deref(), Some("b"));
    assert_eq!(failure.report.applied, ["a"]);
    assert_eq!(failure.stats.source_reads, 1);
    assert_eq!(sink.events, ["write:a"]);
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::InvalidTensorPayload { ref reason, .. })
        if reason == "synthetic read failure"
    ));
    assert!(!failure.to_string().contains("synthetic close failure"));
}

/// Empty explicit work still closes resources and clears stale execution records.
#[test]
fn runtime_coverage_empty_plan_runs_no_callbacks_and_finalizes_clean_progress() {
    let plan = LoadPlan {
        tensors: Vec::new(),
        aliases: Vec::new(),
        equivalences: Vec::new(),
        report: LoadReport {
            applied: vec!["stale".into()],
            quantized: vec![QuantizationRecord {
                target: "stale".into(),
                scheme: "stale".into(),
                bytes: 9,
            }],
            ..LoadReport::default()
        },
        memory: Default::default(),
    };
    let mut source = MemorySource::default();
    let mut sink = CollectingSink::default();
    let predicate_calls = std::cell::Cell::new(0);
    let provider_calls = std::cell::Cell::new(0);
    let outcome = execute_load_plan_with_overrides(
        &plan,
        &mut source,
        &mut sink,
        |_, _| {
            predicate_calls.set(predicate_calls.get() + 1);
            true
        },
        |_, _| {
            provider_calls.set(provider_calls.get() + 1);
            Ok(None)
        },
    )
    .unwrap();
    assert_eq!(predicate_calls.get(), 0);
    assert_eq!(provider_calls.get(), 0);
    assert_eq!(source.close_calls, 1);
    assert!(outcome.report.applied.is_empty());
    assert!(outcome.report.quantized.is_empty());
    assert_eq!(outcome.stats, ExecutionStats::default());
    assert!(sink.events.is_empty());
}

/// Equivalent physical owners must be fully verified before any override callback.
#[test]
fn runtime_coverage_equivalence_read_failures_preserve_verified_prefix_before_overrides() {
    let tensors = [
        FixtureTensor::new("owner", TensorDType::F32, [1], f32_bytes(&[7.0])),
        FixtureTensor::new("a.copy", TensorDType::F32, [1], f32_bytes(&[7.0])),
        FixtureTensor::new("z.copy", TensorDType::F32, [1], f32_bytes(&[7.0])),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let expected = [ExpectedTensor::new("owner", [1])];
    let options = LoadOptions {
        index_aliases: vec![
            IndexTensorAlias::new("a.copy", "owner"),
            IndexTensorAlias::new("z.copy", "owner"),
        ],
        ..LoadOptions::default()
    };
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    for failed in ["owner", "a.copy", "z.copy"] {
        let mut source = MemorySource {
            tensors: tensors
                .iter()
                .map(|tensor| {
                    (
                        tensor.name.clone(),
                        SourceTensor::new(
                            &tensor.name,
                            tensor.shape.clone(),
                            tensor.dtype,
                            tensor.bytes.clone(),
                        )
                        .unwrap(),
                    )
                })
                .collect(),
            fail_at: Some(failed.into()),
            ..MemorySource::default()
        };
        let mut sink = CollectingSink::default();
        let predicates = std::cell::Cell::new(0);
        let providers = std::cell::Cell::new(0);
        let failure = execute_load_plan_with_overrides(
            &plan,
            &mut source,
            &mut sink,
            |_, _| {
                predicates.set(predicates.get() + 1);
                true
            },
            |_, _| {
                providers.set(providers.get() + 1);
                Ok(None)
            },
        )
        .unwrap_err();
        let (target, reads, verified) = match failed {
            "owner" => ("a.copy", 0, 0),
            "a.copy" => ("a.copy", 1, 0),
            _ => ("z.copy", 3, 1),
        };
        assert_eq!(failure.failed_target.as_deref(), Some(target));
        assert_eq!(failure.stats.source_reads, reads);
        assert_eq!(failure.stats.equivalence_source_reads, reads);
        assert_eq!(failure.stats.equivalence_source_payload_bytes, reads * 4);
        assert_eq!(failure.stats.equivalences_verified, verified);
        assert_eq!(failure.report.equivalence_verified.len(), verified);
        if verified == 1 {
            assert_eq!(
                failure.report.equivalence_verified[0],
                CheckpointEquivalenceRecord::new("a.copy", "owner")
            );
        }
        assert_eq!(source.close_calls, 1);
        assert_eq!(predicates.get(), 0);
        assert_eq!(providers.get(), 0);
        assert!(sink.events.is_empty());
        assert!(failure.report.applied.is_empty());
    }
}

/// A failed shared read occurs before its override provider and preserves earlier groups.
#[test]
fn runtime_coverage_shared_source_failure_precedes_override_and_keeps_prior_group() {
    let tensors = [
        FixtureTensor::new("primary", TensorDType::F32, [1], f32_bytes(&[7.0])),
        FixtureTensor::new(
            "fused",
            TensorDType::F32,
            [4],
            f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        ),
    ];
    let (_temp, manifest) = single_manifest(&tensors);
    let expected = [
        ExpectedTensor::new("prior", [1]),
        ExpectedTensor::new("left", [2]),
        ExpectedTensor::new("right", [2]),
    ];
    let mut mapping = MappingPlan::new();
    mapping
        .map("primary", "prior")
        .split("fused", 0, [("left".into(), 2), ("right".into(), 2)]);
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    let mut source = MemorySource {
        tensors: BTreeMap::from([
            (
                "primary".into(),
                SourceTensor::new("primary", [1], TensorDType::F32, f32_bytes(&[7.0])).unwrap(),
            ),
            (
                "fused".into(),
                SourceTensor::new("fused", [2], TensorDType::F32, f32_bytes(&[1.0, 2.0])).unwrap(),
            ),
        ]),
        ..MemorySource::default()
    };
    let mut sink = CollectingSink::default();
    let providers = std::cell::Cell::new(0);
    let failure = execute_load_plan_with_overrides(
        &plan,
        &mut source,
        &mut sink,
        |_, tensor| tensor.target == "left",
        |_, _| {
            providers.set(providers.get() + 1);
            Ok(Some(TensorData::new(vec![9.0_f32; 2], [2])))
        },
    )
    .unwrap_err();
    assert_eq!(failure.failed_target.as_deref(), Some("left"));
    assert_eq!(failure.report.applied, ["prior"]);
    assert_eq!(failure.stats.source_reads, 1);
    assert_eq!(failure.stats.tensors_applied, 1);
    assert_eq!(providers.get(), 0);
    assert_eq!(source.close_calls, 1);
    assert_eq!(sink.events, ["write:prior"]);
    assert!(matches!(failure.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::InvalidTensorPayload { ref tensor, ref reason })
        if tensor == "fused" && reason.contains("shape")));
}

/// Repeated initialized Row/Col loads preserve autodiff leaf policy and reject mutation.
#[cfg(feature = "autodiff")]
#[test]
fn runtime_coverage_initialized_reload_preserves_autodiff_and_parameter_identity_on_error() {
    /// Flex backend with autodiff for checking parameter leaf and freeze policies.
    type AdBackend = burn::backend::Autodiff<Flex>;
    let device = Default::default();
    for layout in [LinearLayout::Row, LinearLayout::Col] {
        for frozen in [false, true] {
            let mut projection = crate::operations::init_linear::<AdBackend>(
                &LinearConfig::new(2, 3).with_bias(false).with_layout(layout),
                &device,
            );
            if frozen {
                projection.weight = projection.weight.set_require_grad(false);
            }
            let _initialized = projection.weight.val();
            let checkpoint_shape = match layout {
                LinearLayout::Row => [2, 3],
                LinearLayout::Col => [3, 2],
            };
            for value in [2.0_f32, 3.0] {
                let replacement = TensorData::new(vec![value; 6], checkpoint_shape);
                install_float_parameter(&mut projection.weight, replacement.clone()).unwrap();
                assert_eq!(projection.weight.val().dims(), [2, 3]);
                assert_eq!(
                    projection.weight.clone().into_record().to_data(),
                    replacement
                );
                let before_id = projection.weight.id;
                let before_value = projection.weight.val().to_data();
                let malformed = TensorData::new(vec![0.0_f32; 6], [1, 6]);
                let error = install_float_parameter(&mut projection.weight, malformed).unwrap_err();
                assert!(matches!(error, ParameterInstallError::Shape { .. }));
                assert_eq!(projection.weight.id, before_id);
                assert_eq!(projection.weight.val().to_data(), before_value);
                assert_eq!(projection.weight.val().is_require_grad(), !frozen);

                let input =
                    Tensor::<AdBackend, 2>::from_floats([[1.0, 2.0]], &device).require_grad();
                let gradients = crate::operations::linear(input.clone(), &projection)
                    .unwrap()
                    .sum()
                    .backward();
                assert_eq!(
                    input.grad(&gradients).unwrap().to_data(),
                    TensorData::from([[3.0 * value, 3.0 * value]])
                );
                if frozen {
                    assert!(projection.weight.grad(&gradients).is_none());
                } else {
                    assert_eq!(
                        projection.weight.grad(&gradients).unwrap().to_data(),
                        TensorData::from([[1.0_f32, 1.0, 1.0], [2.0, 2.0, 2.0]])
                    );
                }
            }
        }
    }
}

/// Initialized Q4/Q8 reloads fail before mapper work and preserve mapped/unmapped state.
#[test]
fn runtime_coverage_initialized_quantized_reload_rejects_before_mapper_without_mutation() {
    let device = Default::default();
    for scheme in [SymmetricQuantization::Q4, SymmetricQuantization::Q8] {
        let full = TensorData::new(
            (0..192)
                .map(|value| value as f32 / 32.0 - 3.0)
                .collect::<Vec<_>>(),
            [3, 64],
        );
        let packed = quantize_tensor_data(
            &full,
            WeightQuantizationConfig {
                scheme,
                group_size: QuantizationGroupSize::new(32).unwrap(),
            },
        )
        .unwrap()
        .into_burn_qfloat_data()
        .unwrap();
        for mapped in [false, true] {
            let mapper_called = Arc::new(AtomicBool::new(false));
            let mut parameter = if mapped {
                let sentinel = mapper_called.clone();
                LinearConfig::new(64, 3)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col)
                    .init::<Flex>(&device)
                    .weight
                    .save_mapper(move |value| {
                        sentinel.store(true, Ordering::SeqCst);
                        value.transpose()
                    })
            } else {
                Param::uninitialized(
                    ParamId::new(),
                    |device, _| Tensor::<Flex, 2>::zeros([3, 64], device),
                    device,
                    false,
                    [3, 64].into(),
                )
            };
            assert!(!parameter.is_initialized());
            install_float_parameter(&mut parameter, packed.clone()).unwrap();
            assert!(!mapper_called.load(Ordering::SeqCst));
            let before_id = parameter.id;
            let before = parameter.val().to_data();
            let before_dtype = parameter.val().dtype();
            assert!(matches!(before_dtype, DType::QFloat(_)));
            assert_eq!(
                parameter.val().dims(),
                if mapped { [64, 3] } else { [3, 64] }
            );

            // Replacing with floats must also reject based on existing storage,
            // not accidentally bypass the guard because the incoming dtype differs.
            for replacement in [packed.clone(), full.clone()] {
                let error = install_float_parameter(&mut parameter, replacement).unwrap_err();
                assert_eq!(error, ParameterInstallError::InitializedQuantized);
                assert!(error.to_string().contains("fresh lazy parameter"));
                assert!(!mapper_called.load(Ordering::SeqCst));
                assert_eq!(parameter.id, before_id);
                assert_eq!(parameter.val().dtype(), before_dtype);
                assert_eq!(parameter.val().to_data(), before);
            }
        }
    }
}

/// Inspecting a caller-defined save mapper must not mutate the initialized value.
#[test]
fn runtime_coverage_save_mapper_shape_inspection_preserves_original_on_error() {
    let device = Default::default();
    let source = Tensor::<Flex, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let mut parameter = Param::from_tensor(source.clone())
        .save_mapper(|value| (value * 2.0).reshape([1, 4]))
        .load_mapper(|value| (value / 2.0).reshape([2, 2]));
    let before_id = parameter.id;
    let error =
        install_float_parameter(&mut parameter, TensorData::from([[0.0_f32; 2]; 2])).unwrap_err();
    assert_eq!(
        error,
        ParameterInstallError::Shape {
            expected: vec![1, 4],
            actual: vec![2, 2]
        }
    );
    assert_eq!(parameter.id, before_id);
    assert_eq!(parameter.val().to_data(), source.to_data());
    install_float_parameter(&mut parameter, TensorData::from([[2.0_f32, 4.0, 6.0, 8.0]])).unwrap();
    assert_eq!(parameter.val().to_data(), source.to_data());
}

/// Structured install failures retain useful shape, rank, and dtype context in text.
#[test]
fn accounting_contract_parameter_error_messages_preserve_metadata() {
    let cases = [
        (
            ParameterInstallError::Rank {
                expected: 2,
                actual: 1,
            },
            "expected rank 2, received rank 1",
        ),
        (
            ParameterInstallError::Shape {
                expected: vec![3, 2],
                actual: vec![2, 3],
            },
            "expected shape [3, 2], received [2, 3]",
        ),
        (
            ParameterInstallError::NonFloat {
                dtype: "I32".into(),
            },
            "expected floating-point data, received I32",
        ),
        (
            ParameterInstallError::UnsupportedStorage {
                dtype: "BF16".into(),
            },
            "backend cannot store checkpoint dtype BF16",
        ),
        (
            ParameterInstallError::PayloadLength {
                expected: 24,
                actual: 20,
            },
            "expected 24 checkpoint float bytes, received 20",
        ),
        (
            ParameterInstallError::PayloadSizeOverflow,
            "checkpoint float payload size overflows usize",
        ),
    ];
    for (error, expected) in cases {
        assert_eq!(error.to_string(), expected);
        assert!(std::error::Error::source(&error).is_none());
    }
    assert!(
        ParameterInstallError::InitializedQuantized
            .to_string()
            .contains("fresh lazy parameter")
    );
}

/// Accounting overflow must be detected before handing ownership to a nontransactional sink.
#[test]
fn accounting_contract_install_overflow_precedes_sink_mutation() {
    let (plan, _) = aliased_memory_fixture();
    for (tensors_applied, delivered_tensor_bytes, expected_context) in [
        (usize::MAX, 8, "counting applied target tensors"),
        (
            2,
            u64::MAX - 3,
            "summing target bytes delivered to the sink",
        ),
    ] {
        let mut stats = ExecutionStats {
            tensors_applied,
            delivered_tensor_bytes,
            ..Default::default()
        };
        let mut report = LoadReport::default();
        report.record_applied("previous");
        let mut sink = CollectingSink::default();
        let result = install_tensor(
            &plan.tensors[0],
            TensorData::from([9.0_f32]),
            &mut sink,
            &mut report,
            &mut stats,
        );
        assert!(matches!(result, Err(InstallError::Checkpoint(
            CheckpointError::ArithmeticOverflow { context }
        )) if context == expected_context));
        assert!(sink.events.is_empty());
        assert!(sink.tensors.is_empty());
        assert_eq!(report.applied, ["previous"]);
        assert_eq!(stats.tensors_applied, tensors_applied);
        assert_eq!(stats.delivered_tensor_bytes, delivered_tensor_bytes);
        assert_eq!(stats.largest_output_tensor_bytes, 0);
    }
}

/// A destination rejection leaves audit counters reusable for a later successful write.
#[test]
fn accounting_contract_sink_failure_does_not_publish_install_counters() {
    let (plan, _) = aliased_memory_fixture();
    let tensor = &plan.tensors[0];
    let mut stats = ExecutionStats {
        tensors_applied: 2,
        delivered_tensor_bytes: 8,
        ..Default::default()
    };
    let mut report = LoadReport::default();
    report.record_applied("previous");
    let mut sink = CollectingSink {
        fail_write: Some(tensor.target.clone()),
        ..Default::default()
    };
    assert!(matches!(
        install_tensor(tensor, TensorData::from([9.0_f32]), &mut sink, &mut report, &mut stats),
        Err(InstallError::Sink(SinkError(target))) if target == tensor.target
    ));
    assert_eq!(stats.tensors_applied, 2);
    assert_eq!(stats.delivered_tensor_bytes, 8);
    assert_eq!(stats.largest_output_tensor_bytes, 0);
    assert_eq!(report.applied, ["previous"]);
    assert!(sink.events.is_empty());
    sink.fail_write = None;
    assert!(
        install_tensor(
            tensor,
            TensorData::from([9.0_f32]),
            &mut sink,
            &mut report,
            &mut stats
        )
        .is_ok()
    );
    assert_eq!(stats.tensors_applied, 3);
    assert_eq!(stats.delivered_tensor_bytes, 12);
    assert_eq!(stats.largest_output_tensor_bytes, 4);
    assert_eq!(report.applied, ["previous", "weight"]);
    assert_eq!(tensor_f32(&sink.tensors["weight"]), [9.0]);
}

/// Override memory includes the simultaneously live fused payload, not just the replacement.
#[test]
fn accounting_contract_override_peak_checks_shape_and_exact_combined_budget() {
    let (plan, _) = aliased_memory_fixture();
    let tensor = &plan.tensors[0];
    let data = TensorData::from([9.0_f32]);
    let source = SourceTensor::new("fused", [2], TensorDType::F32, f32_bytes(&[1.0, 2.0])).unwrap();
    for (shared, bound, required) in [(None, 4, 4), (Some(&source), 12, 12)] {
        let mut stats = ExecutionStats::default();
        observe_override_peak(tensor, &data, shared, bound, &mut stats).unwrap();
        assert_eq!(stats.peak_host_staging_bytes, required);
        assert_eq!(stats.largest_output_tensor_bytes, 4);
        let mut rejected = ExecutionStats::default();
        assert!(matches!(
            observe_override_peak(tensor, &data, shared, bound - 1, &mut rejected),
            Err(CheckpointError::MemoryEstimateExceeded { tensor, required: actual, planned })
            if tensor == "weight" && actual == required && planned == bound - 1
        ));
        // These buffers already exist: a failure report must retain the observed high-water.
        assert_eq!(rejected.peak_host_staging_bytes, required);
        assert_eq!(rejected.largest_output_tensor_bytes, 4);
        assert_eq!(rejected.tensors_applied, 0);
    }
    let mut stats = ExecutionStats::default();
    let wrong_shape = TensorData::from([[9.0_f32]]);
    assert_eq!(wrong_shape.shape.as_slice(), [1, 1]);
    let expected_reason = format!(
        "override shape {:?} differs from planned shape [1]",
        wrong_shape.shape
    );
    assert!(matches!(
        observe_override_peak(tensor, &wrong_shape, Some(&source), 12, &mut stats),
        Err(CheckpointError::InvalidTensorPayload { tensor, reason })
        if tensor == "weight" && reason == expected_reason
    ));
    assert_eq!(stats.peak_host_staging_bytes, 0);
}

/// Override failures cannot claim installation or execute aliases after source cleanup.
#[test]
fn accounting_contract_override_provider_failures_close_without_publication() {
    for case in 0..3 {
        let (plan, mut source) = aliased_memory_fixture();
        let mut sink = CollectingSink::default();
        let failure = execute_load_plan_with_overrides(
            &plan,
            &mut source,
            &mut sink,
            |_, _| true,
            |_, _| match case {
                0 => Ok(None),
                1 => Err(SinkError("override provider".into())),
                _ => Ok(Some(TensorData::from([[9.0_f32]]))),
            },
        )
        .unwrap_err();
        assert_eq!(failure.failed_target.as_deref(), Some("weight"));
        match (&failure.cause, case) {
            (ExecutionFailureCause::Checkpoint(CheckpointError::InvalidMapping { message }), 0) => {
                assert!(message.contains("materializer returned no tensor"))
            }
            (ExecutionFailureCause::Sink(SinkError(message)), 1) => {
                assert_eq!(message, "override provider")
            }
            (
                ExecutionFailureCause::Checkpoint(CheckpointError::InvalidTensorPayload {
                    reason,
                    ..
                }),
                2,
            ) => assert!(reason.contains("override shape")),
            _ => panic!("wrong override failure: {}", failure.cause),
        }
        assert_eq!(source.close_calls, 1);
        assert_eq!(failure.stats.source_reads, 1);
        assert_eq!(failure.stats.tensors_applied, 0);
        assert_eq!(failure.stats.aliases_applied, 0);
        assert!(failure.report.applied.is_empty());
        assert!(sink.events.is_empty());
    }
}

/// A packed-style override still preserves successful aliases when a later alias is rejected.
#[test]
fn checkpoint_standards_override_alias_failure_keeps_successful_owner_and_alias() {
    let (mut plan, mut source) = aliased_memory_fixture();
    plan.aliases[0].target = "a_alias".into();
    plan.aliases.push(PlannedTensor {
        target: "z_alias".into(),
        ..plan.aliases[0].clone()
    });
    let mut sink = CollectingSink {
        fail_alias: Some("z_alias".into()),
        ..Default::default()
    };
    let failure = execute_load_plan_with_overrides(
        &plan,
        &mut source,
        &mut sink,
        |_, _| true,
        |_, _| Ok(Some(TensorData::from([9.0_f32]))),
    )
    .unwrap_err();
    assert_eq!(failure.failed_target.as_deref(), Some("z_alias"));
    assert!(
        matches!(&failure.cause, ExecutionFailureCause::Sink(SinkError(target))
        if target == "z_alias")
    );
    assert_eq!(failure.report.applied, ["a_alias", "weight"]);
    assert_eq!(failure.stats.tensors_applied, 1);
    assert_eq!(failure.stats.aliases_applied, 1);
    assert_eq!(source.close_calls, 1);
    assert_eq!(tensor_f32(&sink.tensors["weight"]), [9.0]);
    assert_eq!(sink.aliases["a_alias"], "weight");
    assert!(!sink.aliases.contains_key("z_alias"));
}
