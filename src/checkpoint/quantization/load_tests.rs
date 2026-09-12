//! Test-only fixtures for load.rs; excluded from non-test compiler inputs.

use std::{
    cell::Cell,
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    rc::Rc,
};

use burn::tensor::{DType, TensorData};
use serde_json::{Map, Value, json};
use tempfile::TempDir;
use thiserror::Error;

use super::*;
use crate::checkpoint::quantization::{
    QuantizationGroupSize, QuantizedBurnpackSession, QuantizedCheckpointOptions,
    WeightQuantizationConfig, quantize_local_checkpoint_to_burnpack, quantize_tensor_data,
};
use crate::checkpoint::{
    CheckpointEquivalenceRecord, CheckpointError, DTypePolicy, ExpectedTensor, IndexTensorAlias,
    LoadOptions, LocalShardSource, MappingPlan, PlannedSource, SourceStats, SourceTensor,
    build_load_plan, discover_checkpoint, execute_local_load_plan,
};

/// Minimal architecture sink that retains delivered metadata and payloads.
#[derive(Default)]
struct RecordingSink {
    /// Final tensors delivered to each architecture target.
    tensors: BTreeMap<String, TensorData>,
    /// Ordered alias-to-owner installation requests.
    aliases: Vec<(String, String)>,
    /// Write/alias event order observed at the sink boundary.
    operations: Vec<String>,
}

/// Deterministic architecture rejection used to inspect error wrapping.
#[derive(Debug, Error)]
#[error("fixture architecture rejected `{0}`")]
struct SinkError(
    /// Target name or injected failure detail.
    String,
);

impl TensorSink for RecordingSink {
    /// Synthetic architecture error used by wrapper failure tests.
    type Error = SinkError;

    /// Retains the delivered tensor and records its write event.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        self.operations.push(format!("write:{target}"));
        self.tensors.insert(target.into(), data);
        Ok(())
    }

    /// Records the alias request and its position in the event stream.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        self.operations.push(format!("alias:{target}"));
        self.aliases
            .push((target.to_owned(), source_target.to_owned()));
        Ok(())
    }
}

/// Sink that persistently changes one checkpoint byte after its first write.
struct SameSizeMutatingSink {
    /// Recorder retaining successful architecture writes.
    inner: RecordingSink,
    /// Fixture file whose last byte is changed after the first accepted write.
    checkpoint: PathBuf,
    /// Whether the one-time corruption has already occurred.
    mutated: bool,
}

impl TensorSink for SameSizeMutatingSink {
    /// Wraps fixture mutation I/O failures and architecture diagnostics.
    type Error = SinkError;

    /// Accepts a tensor, then flips one source byte without changing file length.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        self.inner.write_tensor(target, data)?;
        if !self.mutated {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&self.checkpoint)
                .map_err(|error| SinkError(error.to_string()))?;
            let length = file
                .metadata()
                .map_err(|error| SinkError(error.to_string()))?
                .len();
            file.seek(SeekFrom::Start(length - 1))
                .map_err(|error| SinkError(error.to_string()))?;
            let mut byte = [0_u8; 1];
            file.read_exact(&mut byte)
                .map_err(|error| SinkError(error.to_string()))?;
            byte[0] ^= 1;
            file.seek(SeekFrom::Start(length - 1))
                .map_err(|error| SinkError(error.to_string()))?;
            file.write_all(&byte)
                .and_then(|()| file.sync_data())
                .map_err(|error| SinkError(error.to_string()))?;
            self.mutated = true;
        }
        Ok(())
    }

    /// Forwards alias installation without another source mutation.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        self.inner.alias_tensor(target, source_target)
    }
}

/// Changes a checkpoint pathname only after every fixture tensor is installed.
struct FinalWritePathMutatingSink {
    /// Recorder counting successfully installed tensors.
    inner: RecordingSink,
    /// Source pathname removed or replaced at the last write boundary.
    checkpoint: PathBuf,
    /// Number of successful writes triggering pathname mutation.
    final_write_count: usize,
    /// Replacement file size, or `None` to remove the pathname.
    replacement_length: Option<u64>,
    /// Whether the one-time pathname mutation has occurred.
    mutated: bool,
}

impl TensorSink for FinalWritePathMutatingSink {
    /// Wraps fixture filesystem or architecture failures.
    type Error = SinkError;

    /// Leaves an existing mapped inode intact while removing or replacing its path.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        self.inner.write_tensor(target, data)?;
        if self.inner.tensors.len() == self.final_write_count && !self.mutated {
            if let Some(length) = self.replacement_length {
                // Never truncate a possibly live mapping: retain the original inode
                // under a fixture-only backup name and create a distinct replacement.
                fs::rename(&self.checkpoint, self.checkpoint.with_extension("original"))
                    .map_err(|error| SinkError(error.to_string()))?;
                File::create(&self.checkpoint)
                    .and_then(|file| file.set_len(length))
                    .map_err(|error| SinkError(error.to_string()))?;
            } else {
                fs::remove_file(&self.checkpoint).map_err(|error| SinkError(error.to_string()))?;
            }
            self.mutated = true;
        }
        Ok(())
    }

    /// Forwards aliases without introducing another mutation boundary.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        self.inner.alias_tensor(target, source_target)
    }
}

/// Local source wrapper that deliberately cannot certify descriptors without I/O.
struct NonCertifyingLocalSource<'a> {
    /// Ordinary local source retained for payload reads and accounting.
    inner: LocalShardSource<'a>,
}

impl TensorSource for NonCertifyingLocalSource<'_> {
    /// Performs the ordinary payload read without descriptor-only certification.
    fn read_tensor(&mut self, source: &PlannedSource) -> crate::checkpoint::Result<SourceTensor> {
        self.inner.read_tensor(source)
    }

    /// Closes the underlying local shard handle.
    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.inner.close()
    }

    /// Returns the underlying source's read/open counters.
    fn stats(&self) -> SourceStats {
        self.inner.stats()
    }
}

/// Local source that records closure while retaining fail-closed descriptor validation.
struct CloseAwareLocalSource<'a> {
    /// Source supplying validated descriptors and payload reads.
    inner: LocalShardSource<'a>,
    /// Shared signal observed by the alias sink after closure.
    closed: Rc<Cell<bool>>,
    /// Whether closure should return the injected failure after closing the handle.
    fail_close: bool,
}

impl TensorSource for CloseAwareLocalSource<'_> {
    /// Delegates descriptor validation without changing certification semantics.
    fn validate_tensor_descriptor(
        &self,
        source: &PlannedSource,
    ) -> crate::checkpoint::Result<bool> {
        self.inner.validate_tensor_descriptor(source)
    }

    /// Reads the requested payload from the underlying source.
    fn read_tensor(&mut self, source: &PlannedSource) -> crate::checkpoint::Result<SourceTensor> {
        self.inner.read_tensor(source)
    }

    /// Records closure, releases the handle, then optionally injects an error.
    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.closed.set(true);
        self.inner.close()?;
        if self.fail_close {
            Err(CheckpointError::InvalidMapping {
                message: "synthetic close failure".into(),
            })
        } else {
            Ok(())
        }
    }

    /// Exposes underlying read and handle counters.
    fn stats(&self) -> SourceStats {
        self.inner.stats()
    }
}

/// Sink whose alias boundary proves that source closure precedes tied storage.
struct CloseAwareSink {
    /// Signal indicating the source has reached its closure boundary.
    closed: Rc<Cell<bool>>,
    /// Accepted tensor and alias event sequence.
    operations: Vec<String>,
}

impl TensorSink for CloseAwareSink {
    /// Diagnostic for aliases attempted before source closure.
    type Error = SinkError;

    /// Records a tensor write without retaining its payload.
    fn write_tensor(&mut self, target: &str, _data: TensorData) -> Result<(), Self::Error> {
        self.operations.push(format!("write:{target}"));
        Ok(())
    }

    /// Accepts an alias only after the shared source-closure signal is set.
    fn alias_tensor(&mut self, target: &str, _source_target: &str) -> Result<(), Self::Error> {
        if !self.closed.get() {
            return Err(SinkError(format!("source-open:{target}")));
        }
        self.operations.push(format!("alias:{target}"));
        Ok(())
    }
}

/// Source with a configurable bare descriptor boolean that rejects payload reads.
#[derive(Default)]
struct FailingSource {
    /// Bare descriptor-certification result returned by the synthetic source.
    certify: bool,
    /// Whether the executor called source closure.
    closed: bool,
    /// Number of attempted payload reads, all of which fail.
    reads: usize,
}

impl TensorSource for FailingSource {
    /// Returns the configured descriptor boolean without reading payload bytes.
    fn validate_tensor_descriptor(
        &self,
        _source: &PlannedSource,
    ) -> crate::checkpoint::Result<bool> {
        Ok(self.certify)
    }

    /// Counts the attempted read and returns the named synthetic payload failure.
    fn read_tensor(&mut self, source: &PlannedSource) -> crate::checkpoint::Result<SourceTensor> {
        self.reads += 1;
        Err(CheckpointError::InvalidTensorPayload {
            tensor: source.name.clone(),
            reason: "synthetic read failure".into(),
        })
    }

    /// Records successful closure without owning any real file handle.
    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.closed = true;
        Ok(())
    }
}

/// Writes a standards-compliant unsharded F32 SafeTensors fixture.
fn write_checkpoint(root: &Path, tensors: &[(&str, Vec<usize>)]) {
    let mut header = Map::new();
    let mut offset = 0_u64;
    for (name, shape) in tensors {
        let elements = shape.iter().product::<usize>();
        let bytes = u64::try_from(elements * size_of::<f32>()).unwrap();
        header.insert(
            (*name).into(),
            json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, offset + bytes],
            }),
        );
        offset += bytes;
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    for (_, shape) in tensors {
        let elements = shape.iter().product::<usize>();
        for index in 0..elements {
            file.write_all(&((index as f32 - 31.0) / 8.0).to_le_bytes())
                .unwrap();
        }
    }
}

/// Builds a strict plan with exactly one eligible projection.
fn fixture_plan(root: &Path, cast: Option<TensorDType>) -> (CheckpointManifest, LoadPlan) {
    let options = LoadOptions {
        dtype_policy: cast.map_or(DTypePolicy::Preserve, DTypePolicy::Cast),
        ..LoadOptions::default()
    };
    let manifest = discover_checkpoint(root, &options).unwrap();
    let expected = [
        ExpectedTensor::new("model.embed_tokens.weight", [2, 32]),
        ExpectedTensor::new("model.layers.0.input_layernorm.weight", [32]),
        ExpectedTensor::new("model.layers.0.self_attn.q_proj.weight", [32, 32]),
    ];
    let mappings = MappingPlan::identity(&expected);
    let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
    (manifest, plan)
}

/// Produces one executable Q8 sidecar from the fixture's F32 plan.
fn executable_sidecar(directory: &TempDir) -> (CheckpointManifest, LoadPlan, std::path::PathBuf) {
    write_checkpoint(
        directory.path(),
        &[
            ("model.embed_tokens.weight", vec![2, 32]),
            ("model.layers.0.input_layernorm.weight", vec![32]),
            ("model.layers.0.self_attn.q_proj.weight", vec![32, 32]),
        ],
    );
    let (manifest, plan) = fixture_plan(directory.path(), None);
    let output = directory.path().join("weights-q8.bpk");
    quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions {
            weights: WeightQuantizationConfig {
                group_size: QuantizationGroupSize::new(32).unwrap(),
                ..WeightQuantizationConfig::default()
            },
            ..QuantizedCheckpointOptions::default()
        },
    )
    .unwrap();
    (manifest, plan, output)
}

/// Writes an executable group-32 Q8 sidecar for explicit post-transform targets.
fn write_sidecar(path: &Path, tensors: &[(&str, Vec<usize>)]) {
    let mut session = crate::checkpoint::quantization::QuantizedBurnpackSession::new(path).unwrap();
    for (name, shape) in tensors {
        let elements = shape.iter().product::<usize>();
        let values = (0..elements)
            .map(|index| (index as f32 - 17.0) / 11.0)
            .collect::<Vec<_>>();
        session
            .append(
                name,
                &TensorData::new(values, shape.clone()),
                WeightQuantizationConfig {
                    group_size: QuantizationGroupSize::new(32).unwrap(),
                    ..WeightQuantizationConfig::default()
                },
            )
            .unwrap();
    }
    session.finalize().unwrap();
}

/// Returns the unique location of one complete byte sequence in a fixture.
fn unique_subslice(haystack: &[u8], needle: &[u8]) -> usize {
    let locations = haystack
        .windows(needle.len())
        .enumerate()
        .filter_map(|(index, candidate)| (candidate == needle).then_some(index))
        .collect::<Vec<_>>();
    assert_eq!(locations.len(), 1, "fixture byte sequence must be unique");
    locations[0]
}

/// Checks one complete bounded-memory hash pass over a fixture manifest.
fn assert_complete_hash_pass(
    stats: Option<crate::checkpoint::quantization::ProvenanceHashStats>,
    manifest: &CheckpointManifest,
) {
    let stats = stats.expect("v2 local execution must expose this hash pass");
    assert_eq!(stats.passes_completed, 1);
    assert_eq!(stats.shards_hashed, manifest.shards.len());
    assert_eq!(stats.bytes_hashed, manifest.artifact_bytes);
    assert_eq!(
        stats.peak_buffer_bytes,
        crate::checkpoint::quantization::PROVENANCE_HASH_BUFFER_BYTES
    );
}

/// Requires a legacy contract rejection to precede all payload I/O and model writes.
fn assert_unstarted_legacy_load(
    audited: &AuditedQuantizedExecutionFailure<SinkError>,
    sink: &RecordingSink,
) {
    assert_eq!(audited.format_version, 1);
    assert_eq!(audited.preflight, PackedPreflightStats::default());
    let failure = &audited.execution;
    assert_eq!(failure.failed_target, None);
    assert_eq!(failure.execution, ExecutionStats::default());
    assert_eq!(failure.packed, PackedLoadStats::default());
    assert_eq!(failure.provenance, ProvenanceExecutionStats::default());
    assert!(failure.report.applied.is_empty());
    assert!(failure.report.quantized.is_empty());
    assert!(sink.operations.is_empty());
    assert!(sink.tensors.is_empty());
    assert!(sink.aliases.is_empty());
}

/// Content-verified local overlays bypass packed sources after whole-shard hashing.
#[test]
fn local_overlay_installs_mixed_precision_and_reports_exact_io() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, output) = executable_sidecar(&directory);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    assert_eq!(
        artifact.manifest().packed_execution,
        PackedExecutionSupport::BurnQFloatMatmul
    );
    assert!(artifact.manifest().provenance.is_some());
    let mut sink = RecordingSink::default();

    let audited =
        execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap();
    assert_eq!(
        audited.format_version,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    assert_eq!(audited.preflight.passes_started, 1);
    assert_eq!(audited.preflight.passes_completed, 1);
    assert_eq!(audited.preflight.tensor_reads, 1);
    assert_eq!(audited.preflight.payload_bytes, 1_152);
    assert_eq!(audited.preflight.largest_payload_bytes, 1_152);
    assert!(audited.preflight.peak_owned_payload_bytes >= 1_280);
    let outcome = audited.execution;

    assert_eq!(outcome.execution.tensors_applied, 3);
    assert_eq!(outcome.execution.source_reads, 2);
    assert_eq!(outcome.execution.source_payload_bytes, 384);
    assert_eq!(outcome.execution.delivered_tensor_bytes, 1_536);
    assert_eq!(outcome.execution.largest_source_payload_bytes, 256);
    assert_eq!(outcome.execution.largest_output_tensor_bytes, 1_152);
    assert_eq!(outcome.execution.peak_host_staging_bytes, 1_152);
    assert_eq!(outcome.execution.source.tensor_reads, 2);
    assert_eq!(outcome.execution.source.payload_bytes_read, 384);
    assert_eq!(outcome.packed.tensor_reads, 1);
    assert_eq!(outcome.packed.payload_bytes, 1_152);
    assert_eq!(outcome.packed.largest_payload_bytes, 1_152);
    assert_eq!(outcome.packed.tensors_applied, 1);
    assert_complete_hash_pass(outcome.provenance.pre_execution, &manifest);
    assert_complete_hash_pass(outcome.provenance.post_execution, &manifest);
    assert_eq!(
        outcome.provenance.bytes_hashed(),
        u128::from(manifest.artifact_bytes) * 2
    );
    assert_eq!(
        outcome.provenance.peak_buffer_bytes(),
        crate::checkpoint::quantization::PROVENANCE_HASH_BUFFER_BYTES
    );
    assert_eq!(outcome.report.quantized.len(), 1);
    assert_eq!(
        outcome.report.quantized[0].target,
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert!(matches!(
        sink.tensors["model.layers.0.self_attn.q_proj.weight"].dtype,
        DType::QFloat(_)
    ));
    assert_eq!(sink.tensors["model.embed_tokens.weight"].dtype, DType::F32);
    assert_eq!(
        sink.tensors["model.layers.0.input_layernorm.weight"].dtype,
        DType::F32
    );
}

/// Packed overlays cannot bypass physical checkpoint equivalence preflight.
#[test]
fn local_overlay_verifies_redundant_physical_payloads_before_installation() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(
        directory.path(),
        &[
            ("model.embed_tokens.weight", vec![2, 32]),
            ("decoder.embed_tokens.weight", vec![2, 32]),
            ("model.layers.0.input_layernorm.weight", vec![32]),
            ("model.layers.0.self_attn.q_proj.weight", vec![32, 32]),
        ],
    );
    let options = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new(
            "decoder.embed_tokens.weight",
            "model.embed_tokens.weight",
        )],
        ..LoadOptions::default()
    };
    let manifest = discover_checkpoint(directory.path(), &options).unwrap();
    let expected = [
        ExpectedTensor::new("model.embed_tokens.weight", [2, 32]),
        ExpectedTensor::new("model.layers.0.input_layernorm.weight", [32]),
        ExpectedTensor::new("model.layers.0.self_attn.q_proj.weight", [32, 32]),
    ];
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    let record = CheckpointEquivalenceRecord::new(
        "decoder.embed_tokens.weight",
        "model.embed_tokens.weight",
    );
    assert_eq!(
        plan.report.equivalence_planned,
        std::slice::from_ref(&record)
    );

    let output = directory.path().join("weights-equivalence-q8.bpk");
    let conversion = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions {
            weights: WeightQuantizationConfig {
                group_size: QuantizationGroupSize::new(32).unwrap(),
                ..WeightQuantizationConfig::default()
            },
            ..QuantizedCheckpointOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        conversion.report.equivalence_verified,
        std::slice::from_ref(&record)
    );
    assert_eq!(conversion.execution.equivalences_verified, 1);

    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();
    let outcome =
        execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

    assert_eq!(outcome.report.equivalence_verified, [record]);
    assert_eq!(outcome.execution.equivalences_verified, 1);
    assert_eq!(outcome.execution.equivalence_source_reads, 2);
    assert_eq!(outcome.execution.equivalence_source_payload_bytes, 512);
    assert_eq!(outcome.execution.source_reads, 4);
    assert_eq!(outcome.execution.source_payload_bytes, 896);
    assert_eq!(outcome.execution.largest_equivalence_pair_bytes, 512);
    assert_eq!(sink.operations.len(), 3);
}

/// A sidecar cannot conceal a local source removed after header discovery.
#[test]
fn local_overlay_fails_closed_when_an_overridden_source_disappears() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, output) = executable_sidecar(&directory);
    fs::remove_file(directory.path().join("model.safetensors")).unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let audited =
        execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();
    assert_eq!(audited.preflight, PackedPreflightStats::default());
    let failure = &audited.execution;

    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.execution.source.shard_opens, 0);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert!(failure.report.applied.is_empty());
    assert!(sink.operations.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::ArtifactIo { .. }
        ))
    ));
}

/// Same-size source mutation is detected by its content digest before any model write.
#[test]
fn local_overlay_rejects_checkpoint_payload_digest_mismatch() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, output) = executable_sidecar(&directory);
    let checkpoint = directory.path().join("model.safetensors");
    let mut bytes = fs::read(&checkpoint).unwrap();
    let last = bytes.last_mut().unwrap();
    *last ^= 1;
    fs::write(&checkpoint, bytes).unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let audited =
        execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();
    assert_eq!(audited.preflight.passes_started, 1);
    assert_eq!(audited.preflight.passes_completed, 1);
    assert_eq!(audited.preflight.tensor_reads, 1);
    assert_eq!(audited.preflight.payload_bytes, 1_152);
    assert!(audited.preflight.peak_owned_payload_bytes >= 1_280);
    let failure = &audited.execution;

    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.execution.source.shard_opens, 0);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert!(failure.report.applied.is_empty());
    assert!(sink.operations.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::CheckpointProvenanceDigestMismatch { .. }
        ))
    ));
}

/// Shard-count and declared-length mismatches fail before allocating a hash buffer.
#[test]
fn local_overlay_preflights_shard_inventory_before_hashing() {
    for remove_shard in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let (mut manifest, plan, output) = executable_sidecar(&directory);
        if remove_shard {
            manifest.shards.clear();
        } else {
            manifest.shards[0].file_bytes += 1;
        }
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let failure = execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();

        assert_eq!(failure.provenance.pre_execution, None);
        assert_eq!(failure.provenance.post_execution, None);
        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert!(sink.operations.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::InvalidProvenance { .. }
                    | QuantizationError::CheckpointProvenanceLengthMismatch { .. }
            ))
        ));
    }
}

/// Persistent same-size mutation during execution is rejected after full progress.
#[test]
fn local_overlay_posthash_rejects_sink_driven_source_mutation() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, output) = executable_sidecar(&directory);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = SameSizeMutatingSink {
        inner: RecordingSink::default(),
        checkpoint: directory.path().join("model.safetensors"),
        mutated: false,
    };

    let audited =
        execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();
    assert_eq!(audited.preflight.passes_started, 1);
    assert_eq!(audited.preflight.passes_completed, 1);
    assert_eq!(audited.preflight.tensor_reads, 1);
    assert_eq!(audited.preflight.payload_bytes, 1_152);
    assert!(audited.preflight.peak_owned_payload_bytes >= 1_280);
    let failure = &audited.execution;

    assert!(sink.mutated);
    assert_eq!(failure.failed_target, None);
    assert_eq!(failure.execution.tensors_applied, 3);
    assert_eq!(failure.execution.source_reads, 2);
    assert_eq!(failure.execution.source_payload_bytes, 384);
    assert_eq!(failure.packed.tensor_reads, 1);
    assert_eq!(failure.packed.payload_bytes, 1_152);
    assert_eq!(failure.packed.tensors_applied, 1);
    assert_eq!(failure.report.applied.len(), 3);
    assert_eq!(failure.report.quantized.len(), 1);
    assert_eq!(sink.inner.operations.len(), 3);
    assert_complete_hash_pass(failure.provenance.pre_execution, &manifest);
    assert_complete_hash_pass(failure.provenance.post_execution, &manifest);
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::CheckpointProvenanceDigestMismatch { .. }
        ))
    ));
}

/// Post-execution path loss or size drift retains the complete successful load audit.
#[test]
fn local_overlay_postflight_rejects_source_path_changes_with_full_progress() {
    for length_delta in [None, Some(-1_i64), Some(1)] {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, plan, output) = executable_sidecar(&directory);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut unchanged_sink = RecordingSink::default();
        let unchanged = execute_local_quantized_load_plan_audited(
            &manifest,
            &plan,
            &mut artifact,
            &mut unchanged_sink,
        )
        .unwrap();
        let checkpoint = manifest.shards[0].path.clone();
        let original_length = fs::metadata(&checkpoint).unwrap().len();
        let replacement_length =
            length_delta.map(|delta| original_length.checked_add_signed(delta).unwrap());
        let mut sink = FinalWritePathMutatingSink {
            inner: RecordingSink::default(),
            checkpoint: checkpoint.clone(),
            final_write_count: plan.tensors.len(),
            replacement_length,
            mutated: false,
        };

        let audited =
            execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
                .unwrap_err();
        let failure = &audited.execution;

        assert!(sink.mutated);
        assert_eq!(audited.format_version, unchanged.format_version);
        assert_eq!(audited.preflight, unchanged.preflight);
        assert_eq!(failure.failed_target, None);
        assert_eq!(failure.execution, unchanged.execution.execution);
        assert_eq!(failure.packed, unchanged.execution.packed);
        assert_eq!(failure.report, unchanged.execution.report);
        assert_eq!(failure.execution.tensors_applied, 3);
        assert_eq!(failure.execution.source_reads, 2);
        assert_eq!(failure.execution.source_payload_bytes, 384);
        assert_eq!(failure.packed.tensors_applied, 1);
        assert_eq!(failure.report.applied.len(), 3);
        assert_eq!(failure.report.quantized.len(), 1);
        assert_eq!(sink.inner.operations, unchanged_sink.operations);
        assert_eq!(sink.inner.tensors, unchanged_sink.tensors);
        assert_complete_hash_pass(failure.provenance.pre_execution, &manifest);
        // The second pathname preflight fails before a post-execution hash
        // pass begins; retained pre-execution work must not imply a rehash.
        assert_eq!(failure.provenance.post_execution, None);
        assert_eq!(
            failure.provenance.bytes_hashed(),
            u128::from(manifest.artifact_bytes)
        );

        match (&failure.cause, replacement_length) {
            (
                ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                    QuantizationError::ArtifactIo {
                        operation,
                        path,
                        source,
                    },
                )),
                None,
            ) => {
                assert_eq!(
                    *operation,
                    "inspect checkpoint shard before provenance hashing"
                );
                assert_eq!(path, &checkpoint);
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
                assert!(!checkpoint.exists());
            }
            (
                ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                    QuantizationError::CheckpointProvenanceLengthMismatch {
                        shard,
                        expected,
                        actual,
                    },
                )),
                Some(length),
            ) => {
                assert_eq!(shard, "model.safetensors");
                assert_eq!(*expected, original_length);
                assert_eq!(*actual, length);
                assert_eq!(fs::metadata(&checkpoint).unwrap().len(), length);
                assert_eq!(
                    fs::metadata(checkpoint.with_extension("original"))
                        .unwrap()
                        .len(),
                    original_length
                );
            }
            (cause, _) => panic!("unexpected post-execution failure: {cause:?}"),
        }
    }
}

/// Packed size is rejected from metadata before source or sidecar payload I/O.
#[test]
fn packed_payload_exceeding_the_plan_bound_fails_in_preflight() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, mut plan, output) = executable_sidecar(&directory);
    plan.memory.peak_host_staging_bytes = 1_151;
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let audited =
        execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();
    assert_eq!(audited.preflight, PackedPreflightStats::default());
    let failure = &audited.execution;

    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert!(failure.report.applied.is_empty());
    assert!(sink.operations.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::SidecarContract { .. }
        ))
    ));
}

/// The deterministic preflight index rejects duplicate mutable plan targets.
#[test]
fn sidecar_preflight_rejects_duplicate_plan_targets() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, mut plan, _) = executable_sidecar(&directory);
    let output = directory.path().join("duplicate-plan-v1.bpk");
    write_sidecar(
        &output,
        &[("model.layers.0.self_attn.q_proj.weight", vec![32, 32])],
    );
    let duplicate = plan
        .tensors
        .iter()
        .find(|tensor| tensor.target.ends_with("q_proj.weight"))
        .unwrap()
        .clone();
    plan.tensors.push(duplicate);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let audited =
        execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();
    assert_eq!(audited.format_version, 1);
    assert_eq!(audited.preflight, PackedPreflightStats::default());
    let failure = &audited.execution;

    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert!(failure.report.applied.is_empty());
    assert!(sink.operations.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::SidecarContract { .. }
        ))
    ));
}

/// A source without artifact-bound evidence retains strict source reads.
#[test]
fn non_certifying_source_falls_back_to_full_materialization() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, converted_output) = executable_sidecar(&directory);
    fs::remove_file(converted_output).unwrap();
    let output = directory.path().join("legacy-generic-fallback.bpk");
    write_sidecar(
        &output,
        &[("model.layers.0.self_attn.q_proj.weight", vec![32, 32])],
    );
    let mut source = NonCertifyingLocalSource {
        inner: LocalShardSource::new(&manifest),
    };
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let outcome =
        execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap();

    assert_eq!(outcome.execution.tensors_applied, 3);
    assert_eq!(outcome.execution.source_reads, 3);
    assert_eq!(outcome.execution.source_payload_bytes, 4_480);
    assert_eq!(outcome.execution.source.tensor_reads, 3);
    assert_eq!(outcome.execution.source.payload_bytes_read, 4_480);
    assert_eq!(outcome.execution.delivered_tensor_bytes, 1_536);
    assert_eq!(outcome.execution.largest_source_payload_bytes, 4_096);
    assert_eq!(outcome.execution.largest_output_tensor_bytes, 1_152);
    assert_eq!(outcome.execution.peak_host_staging_bytes, 4_096);
    assert_eq!(outcome.packed.tensor_reads, 1);
    assert_eq!(outcome.packed.payload_bytes, 1_152);
    assert_eq!(outcome.packed.tensors_applied, 1);
    assert!(matches!(
        sink.tensors["model.layers.0.self_attn.q_proj.weight"].dtype,
        DType::QFloat(_)
    ));
}

/// Strict fallback must validate a failing source before decoding its override.
#[test]
fn non_certifying_override_source_failure_precedes_packed_read() {
    let directory = tempfile::tempdir().unwrap();
    let target = "model.layers.0.self_attn.q_proj.weight";
    write_checkpoint(directory.path(), &[(target, vec![32, 32])]);
    let options = LoadOptions::default();
    let manifest = discover_checkpoint(directory.path(), &options).unwrap();
    let expected = [ExpectedTensor::new(target, [32, 32])];
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    let output = directory.path().join("strict-fallback.bpk");
    write_sidecar(&output, &[(target, vec![32, 32])]);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut source = FailingSource::default();
    let mut sink = RecordingSink::default();

    let failure =
        execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();

    assert_eq!(failure.failed_target.as_deref(), Some(target));
    assert_eq!(source.reads, 1);
    assert!(source.closed);
    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.execution.tensors_applied, 0);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert_eq!(failure.packed.payload_bytes, 0);
    assert_eq!(failure.packed.tensors_applied, 0);
    assert!(failure.report.applied.is_empty());
    assert!(failure.report.quantized.is_empty());
    assert!(sink.operations.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::InvalidTensorPayload { .. })
    ));
}

/// Generic sources reject v2 before a bare descriptor claim or payload read.
#[test]
fn generic_source_certificate_is_never_treated_as_v2_artifact_binding() {
    let directory = tempfile::tempdir().unwrap();
    let target = "model.layers.0.self_attn.q_proj.weight";
    write_checkpoint(directory.path(), &[(target, vec![32, 32])]);
    let options = LoadOptions::default();
    let manifest = discover_checkpoint(directory.path(), &options).unwrap();
    let expected = [ExpectedTensor::new(target, [32, 32])];
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    let output = directory.path().join("generic-v2.bpk");
    quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions {
            weights: WeightQuantizationConfig {
                group_size: QuantizationGroupSize::new(32).unwrap(),
                ..WeightQuantizationConfig::default()
            },
            ..QuantizedCheckpointOptions::default()
        },
    )
    .unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    assert!(artifact.manifest().provenance.is_some());
    let mut source = FailingSource {
        certify: true,
        ..FailingSource::default()
    };
    let mut sink = RecordingSink::default();

    let failure =
        execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();

    assert_eq!(failure.failed_target, None);
    assert_eq!(source.reads, 0);
    assert!(source.closed);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert_eq!(failure.provenance, Default::default());
    assert!(sink.operations.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::ArtifactBoundSourceRequired
        ))
    ));
}

/// A same-layout checkpoint with different values cannot mix with a v2 sidecar.
#[test]
fn generic_source_rejects_same_plan_different_checkpoint_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, output) = executable_sidecar(&directory);
    let checkpoint = directory.path().join("model.safetensors");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(checkpoint)
        .unwrap();
    let final_offset = file.metadata().unwrap().len() - 1;
    file.seek(SeekFrom::Start(final_offset)).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    byte[0] ^= 1;
    file.seek(SeekFrom::Start(final_offset)).unwrap();
    file.write_all(&byte).unwrap();
    file.sync_data().unwrap();

    let mut source = LocalShardSource::new(&manifest);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();
    let failure =
        execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();

    assert_eq!(failure.failed_target, None);
    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.execution.source.shard_opens, 0);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert_eq!(failure.provenance, Default::default());
    assert!(failure.report.applied.is_empty());
    assert!(sink.operations.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::ArtifactBoundSourceRequired
        ))
    ));
}

/// Legacy packed weights remain reusable when excluded source tensors change dtype.
#[test]
fn preserved_sidecar_composes_with_a_cast_inference_plan() {
    let directory = tempfile::tempdir().unwrap();
    let (_, _, converted_output) = executable_sidecar(&directory);
    fs::remove_file(converted_output).unwrap();
    let output = directory.path().join("legacy-cast-compatible.bpk");
    write_sidecar(
        &output,
        &[("model.layers.0.self_attn.q_proj.weight", vec![32, 32])],
    );
    let (manifest, cast_plan) = fixture_plan(directory.path(), Some(TensorDType::F16));
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let outcome =
        execute_local_quantized_load_plan(&manifest, &cast_plan, &mut artifact, &mut sink).unwrap();

    assert_eq!(outcome.execution.tensors_applied, 3);
    assert_eq!(outcome.execution.source_reads, 3);
    assert_eq!(outcome.execution.source_payload_bytes, 4_480);
    assert_eq!(outcome.execution.delivered_tensor_bytes, 1_344);
    assert_eq!(outcome.execution.largest_source_payload_bytes, 4_096);
    assert_eq!(outcome.execution.largest_output_tensor_bytes, 2_048);
    assert_eq!(outcome.execution.peak_host_staging_bytes, 6_144);
    assert_eq!(outcome.report.quantized.len(), 1);
    assert_eq!(sink.tensors["model.embed_tokens.weight"].dtype, DType::F16);
    assert!(matches!(
        sink.tensors["model.layers.0.self_attn.q_proj.weight"].dtype,
        DType::QFloat(_)
    ));
}

/// V2 cannot use legacy source-dtype compatibility under a matching plan hash.
#[test]
fn v2_rejects_source_dtype_metadata_that_differs_from_the_planned_target() {
    let directory = tempfile::tempdir().unwrap();
    let target = "model.layers.0.self_attn.q_proj.weight";
    write_checkpoint(
        directory.path(),
        &[
            ("model.embed_tokens.weight", vec![2, 32]),
            ("model.layers.0.input_layernorm.weight", vec![32]),
            (target, vec![32, 32]),
        ],
    );
    let (manifest, cast_plan) = fixture_plan(directory.path(), Some(TensorDType::F16));
    let output = directory.path().join("crafted-v2-source-dtype.bpk");
    let mut session =
        crate::checkpoint::quantization::QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append(
            target,
            &TensorData::new(vec![1.0_f32; 32 * 32], [32, 32]),
            WeightQuantizationConfig {
                group_size: QuantizationGroupSize::new(32).unwrap(),
                ..WeightQuantizationConfig::default()
            },
        )
        .unwrap();
    let (provenance, _) =
        crate::checkpoint::quantization::hash_local_checkpoint_provenance(&manifest, &cast_plan)
            .unwrap();
    session.bind_provenance(provenance).unwrap();
    session.finalize().unwrap();

    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    assert_eq!(
        artifact.manifest().format_version,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    assert_eq!(
        artifact.manifest().tensors[target].source_dtype,
        SourceFloatDType::F32
    );
    let mut sink = RecordingSink::default();

    let failure =
        execute_local_quantized_load_plan(&manifest, &cast_plan, &mut artifact, &mut sink)
            .unwrap_err();

    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert_eq!(failure.provenance, Default::default());
    assert!(failure.report.applied.is_empty());
    assert!(sink.operations.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::SidecarContract { .. }
        ))
    ));
}

/// V2 rejects even shape-compatible plans whose destination dtype changed.
#[test]
fn provenance_rejects_a_different_cast_plan_before_checkpoint_io() {
    let directory = tempfile::tempdir().unwrap();
    let (_, _, output) = executable_sidecar(&directory);
    let (manifest, cast_plan) = fixture_plan(directory.path(), Some(TensorDType::F16));
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let failure =
        execute_local_quantized_load_plan(&manifest, &cast_plan, &mut artifact, &mut sink)
            .unwrap_err();

    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert!(failure.report.applied.is_empty());
    assert!(sink.operations.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::LoadPlanProvenanceMismatch { .. }
        ))
    ));
}

/// A deliberately partial sidecar still validates all local source tensors.
#[test]
fn partial_sidecar_revalidates_every_local_source() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(
        directory.path(),
        &[
            ("model.embed_tokens.weight", vec![2, 32]),
            ("model.layers.0.input_layernorm.weight", vec![32]),
            ("model.layers.0.self_attn.q_proj.weight", vec![32, 32]),
            ("model.layers.0.self_attn.k_proj.weight", vec![32, 32]),
        ],
    );
    let options = LoadOptions::default();
    let manifest = discover_checkpoint(directory.path(), &options).unwrap();
    let expected = [
        ExpectedTensor::new("model.embed_tokens.weight", [2, 32]),
        ExpectedTensor::new("model.layers.0.input_layernorm.weight", [32]),
        ExpectedTensor::new("model.layers.0.self_attn.q_proj.weight", [32, 32]),
        ExpectedTensor::new("model.layers.0.self_attn.k_proj.weight", [32, 32]),
    ];
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    let output = directory.path().join("partial.bpk");
    write_sidecar(
        &output,
        &[("model.layers.0.self_attn.q_proj.weight", vec![32, 32])],
    );
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let outcome =
        execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

    assert_eq!(outcome.execution.tensors_applied, 4);
    assert_eq!(outcome.execution.source_reads, 4);
    assert_eq!(outcome.execution.source_payload_bytes, 8_576);
    assert_eq!(outcome.execution.source.tensor_reads, 4);
    assert_eq!(outcome.execution.source.payload_bytes_read, 8_576);
    assert_eq!(outcome.execution.delivered_tensor_bytes, 5_632);
    assert_eq!(outcome.execution.largest_source_payload_bytes, 4_096);
    assert_eq!(outcome.execution.largest_output_tensor_bytes, 4_096);
    assert_eq!(outcome.execution.peak_host_staging_bytes, 4_096);
    assert_eq!(outcome.packed.tensor_reads, 1);
    assert_eq!(outcome.packed.payload_bytes, 1_152);
    assert_eq!(outcome.packed.tensors_applied, 1);
    assert!(matches!(
        sink.tensors["model.layers.0.self_attn.q_proj.weight"].dtype,
        DType::QFloat(_)
    ));
    assert_eq!(
        sink.tensors["model.layers.0.self_attn.k_proj.weight"].dtype,
        DType::F32
    );
}

/// A partial fused split reads its shared source once and preserves alias order.
#[test]
fn partial_fused_split_reuses_one_source_around_a_packed_sibling() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(directory.path(), &[("fused.weight", vec![64, 32])]);
    let options = LoadOptions::default();
    let manifest = discover_checkpoint(directory.path(), &options).unwrap();
    let q = "model.layers.0.self_attn.q_proj.weight";
    let k = "model.layers.0.self_attn.k_proj.weight";
    let alias = "lm_head.weight";
    let expected = [
        ExpectedTensor::new(q, [32, 32]),
        ExpectedTensor::new(k, [32, 32]),
        ExpectedTensor::new(alias, [32, 32]),
    ];
    let mut mappings = MappingPlan::new();
    mappings
        .split("fused.weight", 0, [(q.to_owned(), 32), (k.to_owned(), 32)])
        .alias(q, alias);
    let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
    let output = directory.path().join("partial-fused.bpk");
    write_sidecar(&output, &[(q, vec![32, 32])]);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let outcome =
        execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

    assert_eq!(outcome.execution.tensors_applied, 2);
    assert_eq!(outcome.execution.aliases_applied, 1);
    assert_eq!(outcome.execution.source_reads, 1);
    assert_eq!(outcome.execution.source_payload_bytes, 8_192);
    assert_eq!(outcome.execution.source.tensor_reads, 1);
    assert_eq!(outcome.execution.delivered_tensor_bytes, 5_248);
    assert_eq!(outcome.execution.largest_source_payload_bytes, 8_192);
    assert_eq!(outcome.execution.largest_output_tensor_bytes, 4_096);
    assert_eq!(outcome.execution.peak_host_staging_bytes, 12_288);
    assert_eq!(outcome.packed.tensor_reads, 1);
    assert_eq!(outcome.packed.payload_bytes, 1_152);
    assert_eq!(outcome.packed.tensors_applied, 1);
    assert_eq!(outcome.report.applied.len(), 3);
    assert_eq!(sink.aliases, [(alias.to_owned(), q.to_owned())]);
    assert_eq!(
        sink.operations,
        [
            format!("write:{k}"),
            format!("write:{q}"),
            format!("alias:{alias}"),
        ]
    );
    assert!(matches!(sink.tensors[q].dtype, DType::QFloat(_)));
    assert_eq!(sink.tensors[k].dtype, DType::F32);
}

/// Packed output contributes to the high-water mark while a fused source is live.
#[test]
fn asymmetric_partial_fused_split_accounts_for_concurrent_packed_output() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(directory.path(), &[("fused.weight", vec![129, 32])]);
    let options = LoadOptions::default();
    let manifest = discover_checkpoint(directory.path(), &options).unwrap();
    let packed = "model.layers.0.self_attn.k_proj.weight";
    let retained = "model.layers.0.self_attn.q_proj.weight";
    let expected = [
        ExpectedTensor::new(packed, [128, 32]),
        ExpectedTensor::new(retained, [1, 32]),
    ];
    let mut mappings = MappingPlan::new();
    mappings.split(
        "fused.weight",
        0,
        [(packed.to_owned(), 128), (retained.to_owned(), 1)],
    );
    let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
    let output = directory.path().join("asymmetric-fused.bpk");
    write_sidecar(&output, &[(packed, vec![128, 32])]);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let outcome =
        execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

    assert_eq!(outcome.execution.source_reads, 1);
    assert_eq!(outcome.execution.source_payload_bytes, 16_512);
    assert_eq!(outcome.execution.delivered_tensor_bytes, 4_736);
    assert_eq!(outcome.execution.largest_source_payload_bytes, 16_512);
    assert_eq!(outcome.execution.largest_output_tensor_bytes, 16_384);
    assert_eq!(outcome.execution.peak_host_staging_bytes, 32_896);
    assert_eq!(outcome.packed.tensor_reads, 1);
    assert_eq!(outcome.packed.payload_bytes, 4_608);
    assert_eq!(outcome.packed.tensors_applied, 1);
    assert_eq!(
        sink.operations,
        [format!("write:{packed}"), format!("write:{retained}")]
    );
}

/// A fully packed fused split validates its local source and applies aliases last.
#[test]
fn complete_fused_split_sidecar_reads_the_shared_local_source_once() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(directory.path(), &[("fused.weight", vec![64, 32])]);
    let options = LoadOptions::default();
    let manifest = discover_checkpoint(directory.path(), &options).unwrap();
    let q = "model.layers.0.self_attn.q_proj.weight";
    let k = "model.layers.0.self_attn.k_proj.weight";
    let alias = "lm_head.weight";
    let expected = [
        ExpectedTensor::new(q, [32, 32]),
        ExpectedTensor::new(k, [32, 32]),
        ExpectedTensor::new(alias, [32, 32]),
    ];
    let mut mappings = MappingPlan::new();
    mappings
        .split("fused.weight", 0, [(q.to_owned(), 32), (k.to_owned(), 32)])
        .alias(q, alias);
    let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
    let output = directory.path().join("complete-fused.bpk");
    write_sidecar(&output, &[(q, vec![32, 32]), (k, vec![32, 32])]);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let outcome =
        execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

    assert_eq!(outcome.execution.tensors_applied, 2);
    assert_eq!(outcome.execution.aliases_applied, 1);
    assert_eq!(outcome.execution.source_reads, 1);
    assert_eq!(outcome.execution.source_payload_bytes, 8_192);
    assert_eq!(outcome.execution.source.shard_opens, 1);
    assert_eq!(outcome.execution.source.tensor_reads, 1);
    assert_eq!(outcome.execution.source.payload_bytes_read, 8_192);
    assert_eq!(outcome.execution.delivered_tensor_bytes, 2_304);
    assert_eq!(outcome.execution.largest_source_payload_bytes, 8_192);
    assert_eq!(outcome.execution.largest_output_tensor_bytes, 4_096);
    assert_eq!(outcome.execution.peak_host_staging_bytes, 12_288);
    assert_eq!(outcome.packed.tensor_reads, 2);
    assert_eq!(outcome.packed.payload_bytes, 2_304);
    assert_eq!(outcome.packed.tensors_applied, 2);
    assert_eq!(outcome.report.applied.len(), 3);
    assert_eq!(sink.aliases, [(alias.to_owned(), q.to_owned())]);
    assert_eq!(
        sink.operations,
        [
            format!("write:{k}"),
            format!("write:{q}"),
            format!("alias:{alias}"),
        ]
    );
    assert!(matches!(sink.tensors[q].dtype, DType::QFloat(_)));
    assert!(matches!(sink.tensors[k].dtype, DType::QFloat(_)));
}

/// Source-validated packed execution closes its source before applying aliases.
#[test]
fn fully_packed_owner_closes_source_before_alias_installation() {
    let directory = tempfile::tempdir().unwrap();
    let owner = "model.layers.0.self_attn.q_proj.weight";
    let alias = "lm_head.weight";
    write_checkpoint(directory.path(), &[(owner, vec![32, 32])]);
    let options = LoadOptions::default();
    let manifest = discover_checkpoint(directory.path(), &options).unwrap();
    let expected = [
        ExpectedTensor::new(owner, [32, 32]),
        ExpectedTensor::new(alias, [32, 32]),
    ];
    let mut mappings = MappingPlan::new();
    mappings.map(owner, owner).alias(owner, alias);
    let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
    let output = directory.path().join("owner-alias.bpk");
    write_sidecar(&output, &[(owner, vec![32, 32])]);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let closed = Rc::new(Cell::new(false));
    let mut source = CloseAwareLocalSource {
        inner: LocalShardSource::new(&manifest),
        closed: closed.clone(),
        fail_close: false,
    };
    let mut sink = CloseAwareSink {
        closed,
        operations: Vec::new(),
    };

    let outcome =
        execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap();

    assert_eq!(outcome.execution.source_reads, 1);
    assert_eq!(outcome.execution.source.shard_opens, 1);
    assert_eq!(outcome.execution.tensors_applied, 1);
    assert_eq!(outcome.execution.aliases_applied, 1);
    assert_eq!(outcome.packed.tensor_reads, 1);
    assert_eq!(outcome.packed.tensors_applied, 1);
    assert_eq!(
        sink.operations,
        [format!("write:{owner}"), format!("alias:{alias}")]
    );
}

/// A close failure preserves its installed packed owner and suppresses aliases.
#[test]
fn source_close_failure_reports_packed_owner_without_pending_alias() {
    let directory = tempfile::tempdir().unwrap();
    let owner = "model.layers.0.self_attn.q_proj.weight";
    let alias = "lm_head.weight";
    write_checkpoint(directory.path(), &[(owner, vec![32, 32])]);
    let options = LoadOptions::default();
    let manifest = discover_checkpoint(directory.path(), &options).unwrap();
    let expected = [
        ExpectedTensor::new(owner, [32, 32]),
        ExpectedTensor::new(alias, [32, 32]),
    ];
    let mut mappings = MappingPlan::new();
    mappings.map(owner, owner).alias(owner, alias);
    let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
    let output = directory.path().join("owner-close-failure.bpk");
    write_sidecar(&output, &[(owner, vec![32, 32])]);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let closed = Rc::new(Cell::new(false));
    let mut source = CloseAwareLocalSource {
        inner: LocalShardSource::new(&manifest),
        closed: closed.clone(),
        fail_close: true,
    };
    let mut sink = CloseAwareSink {
        closed,
        operations: Vec::new(),
    };

    let failure =
        execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();

    assert_eq!(failure.failed_target, None);
    assert_eq!(failure.execution.source_reads, 1);
    assert_eq!(failure.execution.tensors_applied, 1);
    assert_eq!(failure.execution.aliases_applied, 0);
    assert_eq!(failure.execution.delivered_tensor_bytes, 1_152);
    assert_eq!(failure.packed.tensor_reads, 1);
    assert_eq!(failure.packed.payload_bytes, 1_152);
    assert_eq!(failure.packed.tensors_applied, 1);
    assert_eq!(failure.report.applied, [owner]);
    assert_eq!(failure.report.quantized.len(), 1);
    assert_eq!(failure.report.quantized[0].target, owner);
    assert_eq!(sink.operations, [format!("write:{owner}")]);
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::InvalidMapping { .. })
    ));
}

/// A generic source's bare positive certificate cannot authorize packed bypass.
#[test]
fn fused_source_failure_precedes_packed_first_target_installation() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(directory.path(), &[("fused.weight", vec![64, 32])]);
    let options = LoadOptions::default();
    let manifest = discover_checkpoint(directory.path(), &options).unwrap();
    let packed_first = "model.layers.0.self_attn.k_proj.weight";
    let uncovered_second = "model.layers.0.self_attn.q_proj.weight";
    let expected = [
        ExpectedTensor::new(packed_first, [32, 32]),
        ExpectedTensor::new(uncovered_second, [32, 32]),
    ];
    let mut mappings = MappingPlan::new();
    mappings.split(
        "fused.weight",
        0,
        [
            (packed_first.to_owned(), 32),
            (uncovered_second.to_owned(), 32),
        ],
    );
    let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
    assert_eq!(plan.tensors[0].target, packed_first);
    let output = directory.path().join("packed-first.bpk");
    write_sidecar(&output, &[(packed_first, vec![32, 32])]);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut source = FailingSource {
        certify: true,
        ..FailingSource::default()
    };
    let mut sink = RecordingSink::default();

    let failure =
        execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();

    assert_eq!(failure.failed_target.as_deref(), Some(packed_first));
    assert_eq!(source.reads, 1);
    assert!(source.closed);
    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.execution.tensors_applied, 0);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert_eq!(failure.packed.tensors_applied, 0);
    assert!(failure.report.applied.is_empty());
    assert!(failure.report.quantized.is_empty());
    assert!(sink.operations.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Checkpoint(CheckpointError::InvalidTensorPayload { .. })
    ));
}

/// Packed post-transform targets still validate transpose and concatenation sources.
#[test]
fn packed_transpose_and_concatenation_targets_validate_every_local_input() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(
        directory.path(),
        &[
            ("transpose.weight", vec![32, 64]),
            ("concat.a.weight", vec![16, 32]),
            ("concat.b.weight", vec![16, 32]),
        ],
    );
    let options = LoadOptions::default();
    let manifest = discover_checkpoint(directory.path(), &options).unwrap();
    let transpose = "model.layers.0.self_attn.o_proj.weight";
    let concatenate = "model.layers.0.mlp.down_proj.weight";
    let alias = "lm_head.weight";
    let expected = [
        ExpectedTensor::new(transpose, [64, 32]),
        ExpectedTensor::new(concatenate, [32, 32]),
        ExpectedTensor::new(alias, [64, 32]),
    ];
    let mut mappings = MappingPlan::new();
    mappings
        .transpose_2d("transpose.weight", transpose)
        .concatenate(
            ["concat.a.weight".to_owned(), "concat.b.weight".to_owned()],
            concatenate,
            0,
        )
        .alias(transpose, alias);
    let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
    let output = directory.path().join("transformed.bpk");
    let conversion = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions {
            weights: WeightQuantizationConfig {
                group_size: QuantizationGroupSize::new(32).unwrap(),
                ..WeightQuantizationConfig::default()
            },
            ..QuantizedCheckpointOptions::default()
        },
    )
    .unwrap();
    assert_eq!(conversion.execution.source_reads, 3);
    assert_eq!(conversion.execution.source_payload_bytes, 12_288);
    assert_eq!(conversion.staging.tensor_count, 2);

    let mut reference_sink = RecordingSink::default();
    execute_local_load_plan(&manifest, &plan, &mut reference_sink).unwrap();
    let mut parity_artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    for target in [transpose, concatenate] {
        let packed = parity_artifact.read_weight(target).unwrap();
        let tolerance = packed.scales().iter().copied().fold(0.0_f32, f32::max) / 2.0 + 1.0e-5;
        let restored = packed.dequantize_f32().unwrap();
        let reference = reference_sink.tensors[target].as_slice::<f32>().unwrap();
        let restored = restored.as_slice::<f32>().unwrap();
        assert_eq!(restored.len(), reference.len());
        assert!(
            restored
                .iter()
                .zip(reference)
                .all(|(restored, reference)| (restored - reference).abs() <= tolerance),
            "{target} quantized transform exceeded error tolerance {tolerance}"
        );
    }
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let outcome =
        execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

    assert_eq!(outcome.execution.tensors_applied, 2);
    assert_eq!(outcome.execution.aliases_applied, 1);
    assert_eq!(outcome.execution.source_reads, 0);
    assert_eq!(outcome.execution.source_payload_bytes, 0);
    assert_eq!(outcome.execution.source.shard_opens, 0);
    assert_eq!(outcome.execution.delivered_tensor_bytes, 3_456);
    assert_eq!(outcome.execution.largest_source_payload_bytes, 0);
    assert_eq!(outcome.execution.largest_output_tensor_bytes, 2_304);
    assert_eq!(outcome.execution.peak_host_staging_bytes, 2_304);
    assert_eq!(outcome.packed.tensor_reads, 2);
    assert_eq!(outcome.packed.payload_bytes, 3_456);
    assert_eq!(outcome.packed.largest_payload_bytes, 2_304);
    assert_eq!(outcome.packed.tensors_applied, 2);
    assert_eq!(outcome.report.quantized.len(), 2);
    assert_eq!(sink.aliases, [(alias.to_owned(), transpose.to_owned())]);
    assert!(matches!(sink.tensors[transpose].dtype, DType::QFloat(_)));
    assert!(matches!(sink.tensors[concatenate].dtype, DType::QFloat(_)));
}

/// An unknown packed target fails before either source or sink observes a tensor.
#[test]
fn sidecar_target_mismatch_is_preflighted_before_source_io() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, _) = executable_sidecar(&directory);
    let output = directory.path().join("unknown-target.bpk");
    let mut session =
        crate::checkpoint::quantization::QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append(
            "model.layers.0.self_attn.k_proj.weight",
            &TensorData::new(vec![1.0_f32; 32 * 32], [32, 32]),
            WeightQuantizationConfig {
                group_size: QuantizationGroupSize::new(32).unwrap(),
                ..WeightQuantizationConfig::default()
            },
        )
        .unwrap();
    session.finalize().unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let failure =
        execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap_err();

    assert_eq!(failure.execution.source.tensor_reads, 0);
    assert!(failure.report.applied.is_empty());
    assert!(sink.tensors.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::SidecarContract { .. }
        ))
    ));
}

/// Equal element counts do not excuse a wrong logical shape for a known target.
#[test]
fn sidecar_same_name_wrong_shape_is_rejected_before_payload_io() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, _) = executable_sidecar(&directory);
    let target = "model.layers.0.self_attn.q_proj.weight";
    let output = directory.path().join("wrong-shape-v1.bpk");
    write_sidecar(&output, &[(target, vec![16, 64])]);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    assert_eq!(artifact.manifest().tensors[target].shape, [16, 64]);
    let mut sink = RecordingSink::default();

    let audited =
        execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();

    assert_unstarted_legacy_load(&audited, &sink);
    assert!(matches!(
        &audited.execution.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::SidecarContract { name, reason }
        )) if name == target
            && reason == "logical shape [16, 64] does not match planned post-transform shape [32, 32]"
    ));
}

/// Low-level packed storage cannot bypass the embedding exclusion at model load time.
#[test]
fn sidecar_policy_excluded_storage_owner_is_rejected_before_payload_io() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, _) = executable_sidecar(&directory);
    let target = "model.embed_tokens.weight";
    let output = directory.path().join("excluded-owner-v1.bpk");
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    let weight = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 2 * 32], [2, 32]),
        WeightQuantizationConfig {
            group_size: QuantizationGroupSize::new(32).unwrap(),
            ..WeightQuantizationConfig::default()
        },
    )
    .unwrap();
    session.append_quantized(target, weight).unwrap();
    session.finalize().unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    assert_eq!(artifact.manifest().tensors[target].shape, [2, 32]);
    assert_eq!(
        artifact.manifest().packed_execution,
        PackedExecutionSupport::BurnQFloatMatmul
    );
    let mut sink = RecordingSink::default();

    let audited =
        execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();

    assert_unstarted_legacy_load(&audited, &sink);
    assert!(matches!(
        &audited.execution.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::SidecarContract { name, reason }
        )) if name == target
            && reason == "tensor is excluded by the conservative weight-only policy"
    ));
}

/// A valid empty storage artifact is not an executable model sidecar or a load no-op.
#[test]
fn sidecar_empty_contract_is_rejected_before_payload_io() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, _) = executable_sidecar(&directory);
    let output = directory.path().join("empty-v1.bpk");
    write_sidecar(&output, &[]);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    assert!(artifact.manifest().tensors.is_empty());
    assert_eq!(
        artifact.manifest().packed_execution,
        PackedExecutionSupport::Unavailable
    );
    let mut sink = RecordingSink::default();

    let audited =
        execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();

    assert_unstarted_legacy_load(&audited, &sink);
    // Empty artifacts are rejected by the earlier capability guard, before
    // the defensive empty-manifest check can be reached through this API.
    assert!(matches!(
        audited.execution.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::PackedExecutionUnavailable
        ))
    ));
}

/// A packed target cannot conceal a source descriptor mutated after planning.
#[test]
fn skipped_source_descriptor_is_still_checked_against_the_manifest() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, mut plan, converted_output) = executable_sidecar(&directory);
    fs::remove_file(converted_output).unwrap();
    let target = "model.layers.0.self_attn.q_proj.weight";
    let output = directory.path().join("descriptor-v1.bpk");
    write_sidecar(&output, &[(target, vec![32, 32])]);
    let tensor = plan
        .tensors
        .iter_mut()
        .find(|tensor| tensor.target == target)
        .unwrap();
    tensor.sources[0].data_offsets[0] += 4;
    tensor.sources[0].data_offsets[1] += 4;
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let failure =
        execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap_err();

    assert_eq!(failure.failed_target.as_deref(), Some(target));
    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.execution.source.shard_opens, 0);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert!(failure.report.applied.is_empty());
    assert!(sink.tensors.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Checkpoint(
            crate::checkpoint::CheckpointError::PlanManifestMismatch { .. }
        )
    ));
}

/// Reference-only flat groups can be stored but cannot enter packed execution.
#[test]
fn incompatible_artifact_is_marked_and_rejected_before_loading() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("incompatible.bpk");
    let source = TensorData::new(vec![1.0_f32; 64], [8, 8]);
    let mut session =
        crate::checkpoint::quantization::QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append(
            "model.layers.0.self_attn.q_proj.weight",
            &source,
            WeightQuantizationConfig {
                group_size: QuantizationGroupSize::new(32).unwrap(),
                ..WeightQuantizationConfig::default()
            },
        )
        .unwrap();
    session.finalize().unwrap();
    let artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    assert_eq!(
        artifact.manifest().packed_execution,
        PackedExecutionSupport::Unavailable
    );
    assert!(matches!(
        validate_sidecar_contract(
            &LoadPlan {
                equivalences: Vec::new(),
                tensors: Vec::new(),
                aliases: Vec::new(),
                report: LoadReport::default(),
                memory: Default::default(),
            },
            &artifact,
        ),
        Err(QuantizationError::PackedExecutionUnavailable)
    ));
}

/// An architecture rejection remains distinguishable from sidecar corruption.
#[test]
fn packed_sink_failure_preserves_partial_read_accounting() {
    /// Rejects query-projection installation while accepting other fixture targets.
    #[derive(Default)]
    struct RejectingSink;
    impl TensorSink for RejectingSink {
        /// Injected architecture rejection returned through the packed overlay.
        type Error = SinkError;

        /// Rejects only the configured query-projection target suffix.
        fn write_tensor(&mut self, target: &str, _data: TensorData) -> Result<(), Self::Error> {
            if target.ends_with("q_proj.weight") {
                Err(SinkError(target.into()))
            } else {
                Ok(())
            }
        }

        /// Accepts alias calls so the test isolates tensor-write failure accounting.
        fn alias_tensor(&mut self, _target: &str, _source_target: &str) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, output) = executable_sidecar(&directory);
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RejectingSink;
    let audited =
        execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();
    assert_eq!(
        audited.format_version,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    assert_eq!(audited.preflight.passes_started, 1);
    assert_eq!(audited.preflight.passes_completed, 1);
    assert_eq!(audited.preflight.tensor_reads, 1);
    assert_eq!(audited.preflight.payload_bytes, 1_152);
    assert_eq!(audited.preflight.largest_payload_bytes, 1_152);
    assert!(audited.preflight.peak_owned_payload_bytes >= 1_280);
    let failure = &audited.execution;

    assert_eq!(
        failure.failed_target.as_deref(),
        Some("model.layers.0.self_attn.q_proj.weight")
    );
    assert_eq!(failure.execution.source_reads, 2);
    assert_eq!(failure.execution.source_payload_bytes, 384);
    assert_eq!(failure.execution.source.tensor_reads, 2);
    assert_eq!(failure.execution.source.payload_bytes_read, 384);
    assert_eq!(failure.execution.tensors_applied, 2);
    assert_eq!(failure.execution.delivered_tensor_bytes, 384);
    assert_eq!(failure.execution.largest_source_payload_bytes, 256);
    assert_eq!(failure.execution.largest_output_tensor_bytes, 1_152);
    assert_eq!(failure.execution.peak_host_staging_bytes, 1_152);
    assert_eq!(failure.packed.tensor_reads, 1);
    assert_eq!(failure.packed.payload_bytes, 1_152);
    assert_eq!(failure.packed.largest_payload_bytes, 1_152);
    assert_eq!(failure.packed.tensors_applied, 0);
    assert_eq!(
        failure.report.applied,
        [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
        ]
    );
    assert!(failure.report.quantized.is_empty());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Architecture(_))
    ));
    let text = audited.to_string();
    assert!(text.contains("failed at `model.layers.0.self_attn.q_proj.weight`"));
    assert!(text.contains("fixture architecture rejected"));
    let execution_error = StdError::source(audited.as_ref()).unwrap();
    assert!(
        execution_error
            .downcast_ref::<QuantizedExecutionFailure<SinkError>>()
            .is_some()
    );
    let executor_cause = execution_error.source().unwrap();
    let overlay_error = executor_cause.source().unwrap();
    let architecture_error = overlay_error.source().unwrap();
    assert!(architecture_error.downcast_ref::<SinkError>().is_some());
    assert_eq!(overlay_error.to_string(), architecture_error.to_string());
}

/// A corrupt v2 packed payload fails before the first model sink mutation.
#[test]
fn corrupt_v2_packed_payload_cannot_create_partial_model_writes() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, output) = executable_sidecar(&directory);
    let target = "model.layers.0.self_attn.q_proj.weight";
    let payload = {
        let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
        artifact.tensor_payload(target).unwrap().as_bytes().to_vec()
    };
    let mut artifact_bytes = fs::read(&output).unwrap();
    let payload_offset = unique_subslice(&artifact_bytes, &payload);
    artifact_bytes[payload_offset] ^= 1;
    fs::write(&output, artifact_bytes).unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let audited =
        execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();
    assert_eq!(
        audited.format_version,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    assert_eq!(audited.preflight.passes_started, 1);
    assert_eq!(audited.preflight.passes_completed, 0);
    assert_eq!(audited.preflight.tensor_reads, 1);
    assert_eq!(audited.preflight.payload_bytes, 1_152);
    assert_eq!(audited.preflight.largest_payload_bytes, 1_152);
    assert!(audited.preflight.peak_owned_payload_bytes >= 1_152);
    let failure = &audited.execution;

    assert_eq!(failure.failed_target, None);
    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.execution.source_payload_bytes, 0);
    assert_eq!(failure.execution.tensors_applied, 0);
    assert_eq!(failure.execution.delivered_tensor_bytes, 0);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert_eq!(failure.packed.payload_bytes, 0);
    assert_eq!(failure.packed.tensors_applied, 0);
    assert!(failure.report.applied.is_empty());
    assert!(failure.report.quantized.is_empty());
    assert!(sink.tensors.is_empty());
    assert_eq!(failure.provenance.pre_execution, None);
    assert_eq!(failure.provenance.post_execution, None);
    assert!(matches!(
        &failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::PackedPayloadDigestMismatch { name, .. }
        )) if name == target
    ));
}

/// Legacy v1 semantic validation also completes before any model sink call.
#[test]
fn corrupt_v1_packed_payload_cannot_create_partial_model_writes() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(
        directory.path(),
        &[
            ("model.embed_tokens.weight", vec![2, 32]),
            ("model.layers.0.input_layernorm.weight", vec![32]),
            ("model.layers.0.self_attn.q_proj.weight", vec![32, 32]),
        ],
    );
    let (manifest, plan) = fixture_plan(directory.path(), None);
    let target = "model.layers.0.self_attn.q_proj.weight";
    let output = directory.path().join("legacy-corrupt.bpk");
    write_sidecar(&output, &[(target, vec![32, 32])]);
    let payload = {
        let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
        assert_eq!(artifact.manifest().format_version, 1);
        assert!(artifact.manifest().tensors[target].payload_sha256.is_none());
        artifact.tensor_payload(target).unwrap().as_bytes().to_vec()
    };
    let mut artifact_bytes = fs::read(&output).unwrap();
    let payload_offset = unique_subslice(&artifact_bytes, &payload);
    artifact_bytes[payload_offset] = 0x80;
    fs::write(&output, artifact_bytes).unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    let mut sink = RecordingSink::default();

    let audited =
        execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();
    assert_eq!(audited.format_version, 1);
    assert_eq!(audited.preflight.passes_started, 1);
    assert_eq!(audited.preflight.passes_completed, 0);
    assert_eq!(audited.preflight.tensor_reads, 1);
    assert_eq!(audited.preflight.payload_bytes, 1_152);
    assert!(audited.preflight.peak_owned_payload_bytes >= 1_280);
    let failure = &audited.execution;

    assert_eq!(failure.failed_target, None);
    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.packed.tensor_reads, 0);
    assert!(failure.report.applied.is_empty());
    assert!(sink.operations.is_empty());
    assert_eq!(failure.provenance, Default::default());
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::InvalidQuantizedValue {
                index: 0,
                value: -128,
            }
        ))
    ));
}

/// A generic source is closed before any read when a legacy sidecar names a non-owner.
#[test]
fn artifact_contract_generic_preflight_rejects_unknown_storage_without_writes() {
    let directory = tempfile::tempdir().unwrap();
    let (_, plan, _) = executable_sidecar(&directory);
    let output = directory.path().join("unknown-owner.bpk");
    write_sidecar(&output, &[("unknown.weight", vec![32, 32])]);
    let before = fs::read(&output).unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    let mut source = FailingSource::default();
    let mut sink = RecordingSink::default();
    let failure =
        execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();
    assert!(
        matches!(&failure.cause, ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(QuantizationError::SidecarContract { name, reason }))
        if name == "unknown.weight" && reason == "name is not a storage-owning target in the strict load plan")
    );
    assert!(source.closed);
    assert_eq!(source.reads, 0);
    assert_eq!(failure.failed_target, None);
    assert_eq!(failure.execution, ExecutionStats::default());
    assert_eq!(failure.packed, PackedLoadStats::default());
    assert!(failure.report.applied.is_empty());
    assert!(failure.report.quantized.is_empty());
    assert!(sink.operations.is_empty());
    assert_eq!(fs::read(output).unwrap(), before);

    // Diagnostics must retain the typed root error through every public wrapper.
    let text = failure.to_string();
    assert_eq!(
        text,
        "quantized checkpoint execution failed: tensor sink failed: invalid quantized sidecar tensor `unknown.weight`: name is not a storage-owning target in the strict load plan"
    );
    let cause = StdError::source(failure.as_ref()).unwrap();
    let sink_error = cause.source().unwrap();
    let root = sink_error.source().unwrap();
    assert!(root.downcast_ref::<QuantizationError>().is_some());
    assert_eq!(sink_error.to_string(), root.to_string());
}

/// Generic v1 execution validates every packed code before reading source storage.
#[test]
fn artifact_contract_generic_preflight_rejects_reserved_payload_before_source_io() {
    let directory = tempfile::tempdir().unwrap();
    let (_, plan, _) = executable_sidecar(&directory);
    let target = "model.layers.0.self_attn.q_proj.weight";
    let output = directory.path().join("generic-corrupt.bpk");
    write_sidecar(&output, &[(target, vec![32, 32])]);
    let payload = {
        let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
        artifact.tensor_payload(target).unwrap().as_bytes().to_vec()
    };
    // Destroy the fixture's mapping before editing bytes; never mutate a live mmap.
    let mut bytes = fs::read(&output).unwrap();
    let offset = unique_subslice(&bytes, &payload);
    bytes[offset] = 0x80;
    fs::write(&output, &bytes).unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    let mut source = FailingSource::default();
    let mut sink = RecordingSink::default();
    let failure =
        execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
            QuantizationError::InvalidQuantizedValue {
                index: 0,
                value: -128
            }
        ))
    ));
    assert!(source.closed);
    assert_eq!(source.reads, 0);
    assert_eq!(failure.failed_target, None);
    assert_eq!(failure.execution, ExecutionStats::default());
    assert_eq!(failure.packed, PackedLoadStats::default());
    assert_eq!(failure.provenance, ProvenanceExecutionStats::default());
    assert!(failure.report.applied.is_empty());
    assert!(failure.report.quantized.is_empty());
    assert!(sink.operations.is_empty());
    assert_eq!(fs::read(output).unwrap(), bytes);
}

/// Closing a rejected source is mandatory, and a close error takes lifecycle precedence.
#[test]
fn artifact_contract_generic_preflight_retains_source_close_failure() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, plan, _) = executable_sidecar(&directory);
    let output = directory.path().join("close-failure.bpk");
    write_sidecar(&output, &[("unknown.weight", vec![32, 32])]);
    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    let closed = Rc::new(Cell::new(false));
    let mut source = CloseAwareLocalSource {
        inner: LocalShardSource::new(&manifest),
        closed: closed.clone(),
        fail_close: true,
    };
    let mut sink = RecordingSink::default();
    let failure =
        execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();
    assert!(
        matches!(&failure.cause, ExecutionFailureCause::Checkpoint(CheckpointError::InvalidMapping { message }) if message == "synthetic close failure")
    );
    assert!(closed.get());
    assert_eq!(failure.failed_target, None);
    assert_eq!(failure.execution, ExecutionStats::default());
    assert_eq!(failure.packed, PackedLoadStats::default());
    assert!(failure.report.applied.is_empty());
    assert!(failure.report.quantized.is_empty());
    assert!(sink.operations.is_empty());
    let cause = StdError::source(failure.as_ref()).unwrap();
    assert!(
        cause
            .source()
            .unwrap()
            .downcast_ref::<CheckpointError>()
            .is_some()
    );
}

/// Legacy source-dtype tolerance does not admit unrelated F64 or synthetic Flex32 storage.
#[test]
fn artifact_contract_legacy_dtype_must_match_source_or_target() {
    let directory = tempfile::tempdir().unwrap();
    let (_, _, _) = executable_sidecar(&directory);
    let (_, plan) = fixture_plan(directory.path(), Some(TensorDType::F16));
    let target = "model.layers.0.self_attn.q_proj.weight";
    let flex_bytes = vec![1.0_f32; 1024]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
    let sources = [
        ("F64", TensorData::new(vec![1.0_f64; 1024], [32, 32])),
        (
            "Flex32",
            TensorData::from_bytes_vec(flex_bytes, [32, 32], DType::Flex32),
        ),
    ];
    for (dtype, data) in sources {
        let output = directory.path().join(format!("wrong-{dtype}.bpk"));
        let mut session = QuantizedBurnpackSession::new(&output).unwrap();
        session
            .append(
                target,
                &data,
                WeightQuantizationConfig {
                    group_size: QuantizationGroupSize::new(32).unwrap(),
                    ..WeightQuantizationConfig::default()
                },
            )
            .unwrap();
        session.finalize().unwrap();
        let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
        let mut source = FailingSource::default();
        let mut sink = RecordingSink::default();
        let failure =
            execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();
        assert!(
            matches!(&failure.cause, ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(QuantizationError::SidecarContract { name, reason }))
            if name == target && reason == &format!("legacy quantizer-input dtype {dtype} matches neither planned target dtype F16 nor its checkpoint source dtypes"))
        );
        assert!(source.closed);
        assert_eq!(source.reads, 0);
        assert!(sink.operations.is_empty());
    }
    // In contrast, the actual F32 checkpoint dtype remains valid for a cast-F16 plan.
    let valid = directory.path().join("original-source-dtype.bpk");
    write_sidecar(&valid, &[(target, vec![32, 32])]);
    let artifact = QuantizedBurnpackArtifact::open(valid).unwrap();
    validate_sidecar_contract(&plan, &artifact).unwrap();
}

/// Half-precision target plans preserve their quantizer source type through packed loading.
#[test]
fn checkpoint_standards_half_precision_sidecars_match_target_dtype_contracts() {
    for dtype in [TensorDType::F16, TensorDType::BF16] {
        let directory = tempfile::tempdir().unwrap();
        write_checkpoint(
            directory.path(),
            &[
                ("model.embed_tokens.weight", vec![2, 32]),
                ("model.layers.0.input_layernorm.weight", vec![32]),
                ("model.layers.0.self_attn.q_proj.weight", vec![32, 32]),
            ],
        );
        let (manifest, plan) = fixture_plan(directory.path(), Some(dtype));
        let output = directory.path().join("half-source.bpk");
        let target = "model.layers.0.self_attn.q_proj.weight";
        let data = match dtype {
            TensorDType::F16 => {
                TensorData::new(vec![burn::tensor::f16::from_f32(1.0); 1024], [32, 32])
            }
            TensorDType::BF16 => {
                TensorData::new(vec![burn::tensor::bf16::from_f32(1.0); 1024], [32, 32])
            }
            _ => unreachable!(),
        };
        let mut session = QuantizedBurnpackSession::new(&output).unwrap();
        session
            .append(
                target,
                &data,
                WeightQuantizationConfig {
                    group_size: QuantizationGroupSize::new(32).unwrap(),
                    ..Default::default()
                },
            )
            .unwrap();
        let (provenance, _) =
            crate::checkpoint::quantization::hash_local_checkpoint_provenance(&manifest, &plan)
                .unwrap();
        session.bind_provenance(provenance).unwrap();
        session.finalize().unwrap();
        let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
        assert_eq!(
            source_tensor_dtype(artifact.manifest().tensors[target].source_dtype),
            Some(dtype)
        );
        assert_eq!(
            artifact.manifest().packed_execution,
            burn_qfloat_execution_support()
        );
        let mut sink = RecordingSink::default();
        let outcome =
            execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
                .unwrap();
        assert_eq!(outcome.execution.execution.tensors_applied, 3);
        assert_eq!(outcome.execution.packed.tensors_applied, 1);
        assert!(outcome.execution.report.is_strictly_valid());
        assert_eq!(sink.tensors["model.embed_tokens.weight"].dtype, data.dtype);
        assert!(matches!(sink.tensors[target].dtype, DType::QFloat(_)));
    }
}

/// Standard error conversion retains the original typed quantization cause for callers.
#[test]
fn checkpoint_standards_quantization_error_conversion_retains_source() {
    let wrapped: QuantizedSinkError<SinkError> =
        QuantizationError::InvalidGroupSize { value: 17 }.into();
    let source = StdError::source(&wrapped).unwrap();
    assert!(matches!(
        source.downcast_ref::<QuantizationError>(),
        Some(QuantizationError::InvalidGroupSize { value: 17 })
    ));
    assert_eq!(wrapped.to_string(), source.to_string());
}
