//! Load-plan-aware SafeTensors-to-Burnpack conversion.
//!
//! Architecture checkpoint contracts may rename, transpose, split, concatenate,
//! or alias external tensors. Quantizing raw SafeTensors names would silently
//! bypass those semantics. This module instead reuses the audited pass-two load
//! executor and replaces its model sink with [`BurnpackQuantizationSink`]. Each
//! final architecture-owned tensor is materialized exactly as it would be for a
//! full-precision model, quantized if policy permits, appended to disk staging,
//! and dropped before the executor advances.
//!
//! Peak host ownership is therefore bounded by one load-plan source/output plus
//! one packed target. Excluded embeddings, norms, biases, recurrent scalars, and
//! MTP weights are validated and reported but remain in the source checkpoint.
//! Aliases consume no duplicate payload; their topology remains in the strict
//! checkpoint report.
//!
//! Architecture-aware conversion emits provenance-bearing format v2. It hashes
//! every complete source shard before and after pass two and refuses publication
//! if the identities differ, then binds the stable source inventory to a canonical
//! SHA-256 of the exact load-plan transformations. Direct low-level Burnpack write
//! sessions remain available for backward-compatible v1 artifacts.

use std::{
    error::Error as StdError,
    fmt,
    path::{Path, PathBuf},
};

use burn::tensor::TensorData;

use super::{
    BurnpackAppendOutcome, BurnpackArtifactSummary, BurnpackProvenance, BurnpackSessionLimits,
    BurnpackStagingStats, QuantizationError, QuantizationExclusion, QuantizedBurnpackSession,
    SymmetricQuantization, WeightQuantizationConfig,
    provenance::{ProvenanceExecutionStats, hash_local_checkpoint_provenance_audited},
};
use crate::checkpoint::{
    CheckpointError, CheckpointManifest, ExecutionFailureCause, ExecutionStats, LoadPlan,
    LoadReport, LocalShardSource, QuantizationRecord, TensorSink, execute_load_plan,
};

/// Controls one strict, architecture-aware checkpoint conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QuantizedCheckpointOptions {
    /// Symmetric integer precision and contiguous group size.
    pub weights: WeightQuantizationConfig,
    /// Bounded disk, tensor-count, and metadata resources for the writer.
    pub limits: BurnpackSessionLimits,
}

/// One load-plan target deliberately retained in source precision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizationExclusionRecord {
    /// Architecture-owned parameter name after checkpoint transformations.
    pub target: String,
    /// Conservative policy rule that kept the tensor unpacked.
    pub reason: QuantizationExclusion,
}

/// Successful result of streaming one checkpoint plan to an official Burnpack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointQuantizationOutcome {
    /// Atomically published artifact sizes and durability information.
    pub artifact: BurnpackArtifactSummary,
    /// Strict mapping report, augmented with every persisted quantized tensor.
    ///
    /// `applied` means the load-plan target was successfully consumed by the
    /// conversion sink. Consult `quantized` and [`Self::excluded`] to distinguish
    /// sidecar payloads from source-precision parameters.
    pub report: LoadReport,
    /// Architecture parameters intentionally absent from the weight-only sidecar.
    pub excluded: Vec<QuantizationExclusionRecord>,
    /// Source I/O and one-tensor transformation high-water counters.
    pub execution: ExecutionStats,
    /// Final writer staging counters before the private staging file was removed.
    pub staging: BurnpackStagingStats,
    /// Full-shard hashing performed before and after conversion execution.
    pub provenance: ProvenanceExecutionStats,
}

/// Root cause of a failed checkpoint conversion.
#[derive(Debug)]
#[non_exhaustive]
pub enum CheckpointQuantizationFailureCause {
    /// Strict plan validation, SafeTensors I/O, or transformation failed.
    Checkpoint(CheckpointError),
    /// Eligibility, numerical conversion, staging, or Burnpack publication failed.
    Quantization(QuantizationError),
}

impl fmt::Display for CheckpointQuantizationFailureCause {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Quantization(error) => error.fmt(formatter),
        }
    }
}

impl StdError for CheckpointQuantizationFailureCause {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Checkpoint(error) => Some(error),
            Self::Quantization(error) => Some(error),
        }
    }
}

/// Auditable failure with exact partial progress and no published output.
#[derive(Debug)]
pub struct CheckpointQuantizationFailure {
    /// Requested final artifact path.
    pub output_path: PathBuf,
    /// Target being transformed or staged, or `None` for setup/finalization.
    pub failed_target: Option<String>,
    /// Underlying checkpoint or quantization error.
    pub cause: CheckpointQuantizationFailureCause,
    /// Finalized strict report through the last successfully consumed target.
    pub report: LoadReport,
    /// Runtime source/transformation counters at failure.
    pub execution: ExecutionStats,
    /// Quantizer staging counters at failure.
    pub staging: BurnpackStagingStats,
    /// Targets intentionally excluded before the failure.
    pub excluded: Vec<QuantizationExclusionRecord>,
    /// Full-shard hashing attempted before and after conversion execution.
    pub provenance: ProvenanceExecutionStats,
}

/// Progress fields shared by every conversion failure boundary.
struct CheckpointQuantizationProgress {
    report: LoadReport,
    execution: ExecutionStats,
    staging: BurnpackStagingStats,
    excluded: Vec<QuantizationExclusionRecord>,
    provenance: ProvenanceExecutionStats,
}

impl fmt::Display for CheckpointQuantizationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(target) = &self.failed_target {
            write!(
                formatter,
                "checkpoint quantization failed at `{target}` for `{}`: {}",
                self.output_path.display(),
                self.cause
            )
        } else {
            write!(
                formatter,
                "checkpoint quantization failed for `{}`: {}",
                self.output_path.display(),
                self.cause
            )
        }
    }
}

impl StdError for CheckpointQuantizationFailure {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.cause)
    }
}

/// Result type retaining structured partial reports off the success-path stack.
pub type CheckpointQuantizationResult<T> =
    std::result::Result<T, Box<CheckpointQuantizationFailure>>;

/// Streams a validated architecture load plan into a quantized Burnpack sidecar.
///
/// The function is local-only and never contacts a model hub. The passed
/// [`CheckpointManifest`] remains the authority for containment and shard byte
/// ranges, while `plan` supplies the exact architecture mapping. Publication is
/// create-only and atomic; every error leaves `output_path` absent or preserves a
/// file that already existed before this call.
///
/// Provenance requires two complete sequential reads of every source shard. This
/// adds bounded-I/O work to offline conversion and prevents publication when an
/// ordinary persistent checkpoint modification spans execution. It is not a
/// hostile-filesystem authenticity guarantee: an adversarial ABA replacement
/// restored before the second observation remains outside the contract.
pub fn quantize_local_checkpoint_to_burnpack(
    manifest: &CheckpointManifest,
    plan: &LoadPlan,
    output_path: impl AsRef<Path>,
    options: QuantizedCheckpointOptions,
) -> CheckpointQuantizationResult<CheckpointQuantizationOutcome> {
    let output_path = output_path.as_ref().to_path_buf();
    let mut report = unstarted_report(plan);
    let mut provenance_stats = ProvenanceExecutionStats::default();
    let mut session =
        QuantizedBurnpackSession::with_limits(&output_path, options.limits).map_err(|cause| {
            conversion_failure(
                output_path.clone(),
                None,
                CheckpointQuantizationFailureCause::Quantization(cause),
                CheckpointQuantizationProgress {
                    report: report.clone(),
                    execution: ExecutionStats::default(),
                    staging: BurnpackStagingStats::default(),
                    excluded: Vec::new(),
                    provenance: provenance_stats,
                },
            )
        })?;

    // Hashing on both sides of pass two prevents a source mutation from spanning
    // conversion unnoticed. The one-MiB hash buffer is independent of model size.
    provenance_stats.pre_execution = Some(Default::default());
    let (initial_provenance, initial_hash_stats) =
        match hash_local_checkpoint_provenance_audited(manifest, plan) {
            Ok(result) => result,
            Err(failure) => {
                provenance_stats.pre_execution = Some(failure.stats);
                return Err(conversion_failure(
                    output_path,
                    None,
                    CheckpointQuantizationFailureCause::Quantization(failure.error),
                    CheckpointQuantizationProgress {
                        report,
                        execution: ExecutionStats::default(),
                        staging: BurnpackStagingStats::default(),
                        excluded: Vec::new(),
                        provenance: provenance_stats,
                    },
                ));
            }
        };
    provenance_stats.pre_execution = Some(initial_hash_stats);

    let mut source = LocalShardSource::new(manifest);
    let mut sink = BurnpackQuantizationSink::new(&mut session, options.weights);
    let execution_result = execute_load_plan(plan, &mut source, &mut sink);
    let quantized = std::mem::take(&mut sink.quantized);
    let mut excluded = std::mem::take(&mut sink.excluded);
    drop(sink);
    let mut staging = session.stats();

    let execution = match execution_result {
        Ok(execution) => execution,
        Err(failure) => {
            let failure = *failure;
            report = failure.report;
            extend_quantized_report(&mut report, quantized);
            normalize_exclusions(&mut excluded);
            let cause = match failure.cause {
                ExecutionFailureCause::Checkpoint(cause) => {
                    CheckpointQuantizationFailureCause::Checkpoint(cause)
                }
                ExecutionFailureCause::Sink(cause) => {
                    CheckpointQuantizationFailureCause::Quantization(cause)
                }
            };
            return Err(conversion_failure(
                output_path,
                failure.failed_target,
                cause,
                CheckpointQuantizationProgress {
                    report,
                    execution: failure.stats,
                    staging,
                    excluded,
                    provenance: provenance_stats,
                },
            ));
        }
    };

    report = execution.report;
    extend_quantized_report(&mut report, quantized);
    normalize_exclusions(&mut excluded);
    provenance_stats.post_execution = Some(Default::default());
    let (final_provenance, final_hash_stats) =
        match hash_local_checkpoint_provenance_audited(manifest, plan) {
            Ok(result) => result,
            Err(failure) => {
                provenance_stats.post_execution = Some(failure.stats);
                return Err(conversion_failure(
                    output_path,
                    None,
                    CheckpointQuantizationFailureCause::Quantization(failure.error),
                    CheckpointQuantizationProgress {
                        report,
                        execution: execution.stats,
                        staging,
                        excluded,
                        provenance: provenance_stats,
                    },
                ));
            }
        };
    provenance_stats.post_execution = Some(final_hash_stats);
    if initial_provenance != final_provenance {
        let shard = first_changed_shard(&initial_provenance, &final_provenance);
        return Err(conversion_failure(
            output_path,
            None,
            CheckpointQuantizationFailureCause::Quantization(
                QuantizationError::CheckpointChangedDuringQuantization { shard },
            ),
            CheckpointQuantizationProgress {
                report,
                execution: execution.stats,
                staging,
                excluded,
                provenance: provenance_stats,
            },
        ));
    }
    session.bind_provenance(final_provenance).map_err(|cause| {
        conversion_failure(
            output_path.clone(),
            None,
            CheckpointQuantizationFailureCause::Quantization(cause),
            CheckpointQuantizationProgress {
                report: report.clone(),
                execution: execution.stats.clone(),
                staging,
                excluded: excluded.clone(),
                provenance: provenance_stats,
            },
        )
    })?;
    staging = session.stats();
    let artifact = session.finalize().map_err(|cause| {
        conversion_failure(
            output_path.clone(),
            None,
            CheckpointQuantizationFailureCause::Quantization(cause),
            CheckpointQuantizationProgress {
                report: report.clone(),
                execution: execution.stats.clone(),
                staging,
                excluded: excluded.clone(),
                provenance: provenance_stats,
            },
        )
    })?;

    Ok(CheckpointQuantizationOutcome {
        artifact,
        report,
        excluded,
        execution: execution.stats,
        staging,
        provenance: provenance_stats,
    })
}

/// Identifies the first source shard changed across conversion's two hash passes.
fn first_changed_shard(before: &BurnpackProvenance, after: &BurnpackProvenance) -> String {
    before
        .checkpoint_shards
        .keys()
        .chain(after.checkpoint_shards.keys())
        .find(|name| before.checkpoint_shards.get(*name) != after.checkpoint_shards.get(*name))
        .cloned()
        .unwrap_or_else(|| "<load-plan>".into())
}

/// Tensor sink that applies quantization policy after load-plan transformations.
struct BurnpackQuantizationSink<'a> {
    session: &'a mut QuantizedBurnpackSession,
    config: WeightQuantizationConfig,
    quantized: Vec<QuantizationRecord>,
    excluded: Vec<QuantizationExclusionRecord>,
}

impl<'a> BurnpackQuantizationSink<'a> {
    /// Creates an empty reporting sink over one private write session.
    fn new(session: &'a mut QuantizedBurnpackSession, config: WeightQuantizationConfig) -> Self {
        Self {
            session,
            config,
            quantized: Vec::new(),
            excluded: Vec::new(),
        }
    }
}

impl TensorSink for BurnpackQuantizationSink<'_> {
    type Error = QuantizationError;

    /// Quantizes or explicitly excludes one final architecture parameter.
    fn write_tensor(
        &mut self,
        target: &str,
        data: TensorData,
    ) -> std::result::Result<(), Self::Error> {
        match self.session.append(target, &data, self.config)? {
            BurnpackAppendOutcome::Staged(staged) => {
                self.quantized.push(QuantizationRecord {
                    target: target.to_owned(),
                    scheme: quantization_scheme_name(self.config),
                    bytes: staged.payload_bytes,
                });
            }
            BurnpackAppendOutcome::Excluded(reason) => {
                self.excluded.push(QuantizationExclusionRecord {
                    target: target.to_owned(),
                    reason,
                });
            }
        }
        Ok(())
    }

    /// Records no duplicate payload for a tied architecture parameter.
    fn alias_tensor(
        &mut self,
        _target: &str,
        _source_target: &str,
    ) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

/// Produces the stable scheme spelling stored in [`LoadReport`].
fn quantization_scheme_name(config: WeightQuantizationConfig) -> String {
    let precision = match config.scheme {
        SymmetricQuantization::Q8 => "q8",
        SymmetricQuantization::Q4 => "q4",
    };
    format!(
        "symmetric-{precision}-g{}-flat-row-major",
        config.group_size.get()
    )
}

/// Resets progress-bearing report categories before the first source read.
fn unstarted_report(plan: &LoadPlan) -> LoadReport {
    let mut report = plan.report.clone();
    report.applied.clear();
    report.quantized.clear();
    report.equivalence_verified.clear();
    report.finalize();
    report
}

/// Adds exact sidecar records and restores deterministic report ordering.
fn extend_quantized_report(report: &mut LoadReport, quantized: Vec<QuantizationRecord>) {
    for record in quantized {
        report.record_quantized(record);
    }
    report.finalize();
}

/// Sorts exclusion records and removes exact duplicates defensively.
fn normalize_exclusions(excluded: &mut Vec<QuantizationExclusionRecord>) {
    excluded.sort_by(|left, right| left.target.cmp(&right.target));
    excluded.dedup();
}

/// Boxes a rich failure consistently across setup, execution, and publication.
fn conversion_failure(
    output_path: PathBuf,
    failed_target: Option<String>,
    cause: CheckpointQuantizationFailureCause,
    progress: CheckpointQuantizationProgress,
) -> Box<CheckpointQuantizationFailure> {
    Box::new(CheckpointQuantizationFailure {
        output_path,
        failed_target,
        cause,
        report: progress.report,
        execution: progress.execution,
        staging: progress.staging,
        excluded: progress.excluded,
        provenance: progress.provenance,
    })
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Write, path::Path};

    use serde_json::{Map, Value, json};

    use super::*;
    use crate::checkpoint::{
        ExpectedTensor, LoadOptions, MappingPlan, build_load_plan, discover_checkpoint,
    };

    /// Minimal input tensor for a standards-compliant SafeTensors fixture.
    struct FixtureTensor {
        name: &'static str,
        shape: Vec<u64>,
        values: Vec<f32>,
    }

    /// Writes one deterministic unsharded F32 SafeTensors artifact.
    fn write_checkpoint(root: &Path, tensors: &[FixtureTensor]) {
        let mut offset = 0_u64;
        let mut header = Map::new();
        for tensor in tensors {
            let bytes = u64::try_from(tensor.values.len() * size_of::<f32>()).unwrap();
            let end = offset + bytes;
            header.insert(
                tensor.name.into(),
                json!({
                    "dtype": "F32",
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
        let mut file = File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        for tensor in tensors {
            for value in &tensor.values {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
    }

    /// Builds a strict plan including a renamed weight, exclusions, and an alias.
    fn fixture_plan(root: &Path, tensors: &[FixtureTensor]) -> (CheckpointManifest, LoadPlan) {
        write_checkpoint(root, tensors);
        let manifest = discover_checkpoint(root, &LoadOptions::default()).unwrap();
        let expected = [
            ExpectedTensor::new("model.embed_tokens.weight", [4, 8]),
            ExpectedTensor::new("model.layers.0.input_layernorm.weight", [8]),
            ExpectedTensor::new("model.layers.0.self_attn.q_proj.weight", [4, 8]),
            ExpectedTensor::new("lm_head.weight", [4, 8]),
        ];
        let mut mappings = MappingPlan::new();
        mappings
            .map("external.embedding", "model.embed_tokens.weight")
            .map("external.norm", "model.layers.0.input_layernorm.weight")
            .map(
                "external.projection",
                "model.layers.0.self_attn.q_proj.weight",
            )
            .alias("model.embed_tokens.weight", "lm_head.weight");
        let plan =
            build_load_plan(&manifest, &expected, &mappings, &LoadOptions::default()).unwrap();
        (manifest, plan)
    }

    /// Architecture mappings run before policy and every outcome is auditable.
    #[test]
    fn local_plan_streams_quantized_targets_and_reports_exclusions() {
        let root = tempfile::tempdir().unwrap();
        let tensors = [
            FixtureTensor {
                name: "external.embedding",
                shape: vec![4, 8],
                values: vec![0.25; 32],
            },
            FixtureTensor {
                name: "external.norm",
                shape: vec![8],
                values: vec![1.0; 8],
            },
            FixtureTensor {
                name: "external.projection",
                shape: vec![4, 8],
                values: (0..32).map(|index| index as f32 - 15.5).collect(),
            },
        ];
        let (manifest, plan) = fixture_plan(root.path(), &tensors);
        let output = root.path().join("weights-q4.bpk");
        let options = QuantizedCheckpointOptions {
            weights: WeightQuantizationConfig {
                scheme: SymmetricQuantization::Q4,
                ..WeightQuantizationConfig::default()
            },
            ..QuantizedCheckpointOptions::default()
        };

        let outcome = quantize_local_checkpoint_to_burnpack(&manifest, &plan, &output, options)
            .expect("valid conversion");
        assert_eq!(outcome.artifact.tensor_count, 1);
        assert_eq!(outcome.report.quantized.len(), 1);
        assert_eq!(
            outcome.report.quantized[0].target,
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            outcome.report.quantized[0].scheme,
            "symmetric-q4-g64-flat-row-major"
        );
        assert_eq!(outcome.report.applied.len(), 4);
        assert_eq!(outcome.report.aliased.len(), 1);
        assert_eq!(
            outcome.excluded,
            [
                QuantizationExclusionRecord {
                    target: "model.embed_tokens.weight".into(),
                    reason: QuantizationExclusion::Embedding,
                },
                QuantizationExclusionRecord {
                    target: "model.layers.0.input_layernorm.weight".into(),
                    reason: QuantizationExclusion::Normalization,
                },
            ]
        );
        assert_eq!(outcome.execution.source.tensor_reads, 3);
        assert_eq!(outcome.staging.tensor_count, 1);
        assert!(outcome.staging.estimated_manifest_bytes >= outcome.artifact.manifest_bytes as u64);
        let pre_hash = outcome.provenance.pre_execution.unwrap();
        let post_hash = outcome.provenance.post_execution.unwrap();
        for stats in [pre_hash, post_hash] {
            assert_eq!(stats.passes_completed, 1);
            assert_eq!(stats.shards_hashed, 1);
            assert_eq!(stats.bytes_hashed, manifest.artifact_bytes);
            assert_eq!(
                stats.peak_buffer_bytes,
                crate::checkpoint::quantization::PROVENANCE_HASH_BUFFER_BYTES
            );
        }
        assert!(output.is_file());

        let mut artifact = super::super::QuantizedBurnpackArtifact::open(&output).unwrap();
        assert!(
            artifact
                .manifest()
                .tensors
                .values()
                .all(|metadata| metadata.payload_sha256.is_some())
        );
        artifact.validate_all().unwrap();
        assert_eq!(
            artifact
                .read_weight("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .metadata()
                .shape,
            [4, 8]
        );
    }

    /// Numerical failure preserves partial diagnostics and publishes no artifact.
    #[test]
    fn conversion_failure_is_structured_and_non_publishing() {
        let root = tempfile::tempdir().unwrap();
        let tensors = [
            FixtureTensor {
                name: "a.weight",
                shape: vec![4, 8],
                values: vec![1.0; 32],
            },
            FixtureTensor {
                name: "z.weight",
                shape: vec![4, 8],
                values: vec![f32::NAN; 32],
            },
        ];
        write_checkpoint(root.path(), &tensors);
        let manifest = discover_checkpoint(root.path(), &LoadOptions::default()).unwrap();
        let expected = [
            ExpectedTensor::new("a.weight", [4, 8]),
            ExpectedTensor::new("z.weight", [4, 8]),
        ];
        let plan = build_load_plan(
            &manifest,
            &expected,
            &MappingPlan::identity(&expected),
            &LoadOptions::default(),
        )
        .unwrap();
        let output = root.path().join("failed.bpk");

        let failure = quantize_local_checkpoint_to_burnpack(
            &manifest,
            &plan,
            &output,
            QuantizedCheckpointOptions::default(),
        )
        .unwrap_err();
        assert_eq!(failure.failed_target.as_deref(), Some("z.weight"));
        assert!(matches!(
            failure.cause,
            CheckpointQuantizationFailureCause::Quantization(
                QuantizationError::NonFiniteValue { .. }
            )
        ));
        assert_eq!(failure.report.applied, ["a.weight"]);
        assert_eq!(failure.report.quantized.len(), 1);
        assert_eq!(failure.staging.tensor_count, 1);
        let pre_hash = failure.provenance.pre_execution.unwrap();
        assert_eq!(pre_hash.passes_completed, 1);
        assert_eq!(pre_hash.shards_hashed, 1);
        assert_eq!(pre_hash.bytes_hashed, manifest.artifact_bytes);
        assert_eq!(failure.provenance.post_execution, None);
        assert!(!output.exists());
    }

    /// Existing outputs fail before the first source tensor is read or replaced.
    #[test]
    fn existing_output_is_preserved_before_checkpoint_io() {
        let root = tempfile::tempdir().unwrap();
        let tensors = [
            FixtureTensor {
                name: "external.embedding",
                shape: vec![4, 8],
                values: vec![0.25; 32],
            },
            FixtureTensor {
                name: "external.norm",
                shape: vec![8],
                values: vec![1.0; 8],
            },
            FixtureTensor {
                name: "external.projection",
                shape: vec![4, 8],
                values: vec![2.0; 32],
            },
        ];
        let (manifest, plan) = fixture_plan(root.path(), &tensors);
        let output = root.path().join("occupied.bpk");
        std::fs::write(&output, b"original").unwrap();

        let failure = quantize_local_checkpoint_to_burnpack(
            &manifest,
            &plan,
            &output,
            QuantizedCheckpointOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(
            failure.cause,
            CheckpointQuantizationFailureCause::Quantization(
                QuantizationError::OutputAlreadyExists { .. }
            )
        ));
        assert_eq!(failure.execution.source.tensor_reads, 0);
        assert_eq!(failure.provenance, Default::default());
        assert_eq!(std::fs::read(&output).unwrap(), b"original");
    }

    /// Creates a strict identity plan for arbitrary small conversion targets.
    fn identity_fixture(root: &Path, names: &[&'static str]) -> (CheckpointManifest, LoadPlan) {
        let tensors = names
            .iter()
            .map(|&name| FixtureTensor {
                name,
                shape: vec![4, 8],
                values: vec![1.0; 32],
            })
            .collect::<Vec<_>>();
        write_checkpoint(root, &tensors);
        let manifest = discover_checkpoint(root, &LoadOptions::default()).unwrap();
        let expected = names
            .iter()
            .map(|&name| ExpectedTensor::new(name, [4, 8]))
            .collect::<Vec<_>>();
        let plan = build_load_plan(
            &manifest,
            &expected,
            &MappingPlan::identity(&expected),
            &LoadOptions::default(),
        )
        .unwrap();
        (manifest, plan)
    }

    /// Seeds stale runtime categories to ensure failed conversions do not inherit them.
    fn stale_report(plan: &mut LoadPlan) {
        plan.report.applied.push("old".into());
        plan.report.quantized.push(QuantizationRecord {
            target: "old".into(),
            scheme: "old".into(),
            bytes: 1,
        });
        plan.report
            .equivalence_verified
            .push(crate::checkpoint::CheckpointEquivalenceRecord::new(
                "old", "owner",
            ));
    }

    /// Setup errors avoid hashing and preserve the source error's inspectable chain.
    #[test]
    fn invalid_conversion_limits_reset_stale_progress_before_checkpoint_access() {
        let root = tempfile::tempdir().unwrap();
        let (manifest, mut plan) = identity_fixture(root.path(), &["weight"]);
        stale_report(&mut plan);
        std::fs::remove_file(&manifest.shards[0].path).unwrap();
        let output = root.path().join("invalid-limit.bpk");
        let failure = quantize_local_checkpoint_to_burnpack(
            &manifest,
            &plan,
            &output,
            QuantizedCheckpointOptions {
                limits: BurnpackSessionLimits {
                    max_tensors: 0,
                    ..BurnpackSessionLimits::default()
                },
                ..QuantizedCheckpointOptions::default()
            },
        )
        .unwrap_err();
        assert!(failure.failed_target.is_none());
        assert_eq!(failure.provenance, ProvenanceExecutionStats::default());
        assert_eq!(failure.execution, ExecutionStats::default());
        assert!(failure.report.applied.is_empty());
        assert!(failure.report.quantized.is_empty());
        assert!(failure.report.equivalence_verified.is_empty());
        assert!(
            failure
                .to_string()
                .starts_with("checkpoint quantization failed for")
        );
        let cause = StdError::source(failure.as_ref()).unwrap();
        assert!(
            cause
                .source()
                .unwrap()
                .downcast_ref::<QuantizationError>()
                .is_some()
        );
        assert!(matches!(
            failure.cause,
            CheckpointQuantizationFailureCause::Quantization(
                QuantizationError::InvalidSessionLimit {
                    field: "max_tensors",
                    ..
                }
            )
        ));
        assert!(!output.exists());
    }

    /// A failed initial hash reports its completed prefix without staging a tensor.
    #[test]
    fn initial_provenance_failure_records_partial_hashing_without_publication() {
        let root = tempfile::tempdir().unwrap();
        let (mut manifest, plan) = identity_fixture(root.path(), &["weight"]);
        let mut absent = manifest.shards[0].clone();
        absent.relative_path = "missing.safetensors".into();
        absent.path = manifest.root.join(&absent.relative_path);
        manifest.shards.push(absent);
        let output = root.path().join("missing-source.bpk");
        let failure = quantize_local_checkpoint_to_burnpack(
            &manifest,
            &plan,
            &output,
            QuantizedCheckpointOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(
            failure.cause,
            CheckpointQuantizationFailureCause::Quantization(QuantizationError::ArtifactIo {
                operation: "canonicalize checkpoint shard for provenance",
                ..
            })
        ));
        let pre = failure.provenance.pre_execution.unwrap();
        assert_eq!(pre.shards_hashed, 1);
        assert_eq!(pre.bytes_hashed, manifest.shards[0].file_bytes);
        assert_eq!(pre.passes_completed, 0);
        assert_eq!(failure.provenance.post_execution, None);
        assert_eq!(failure.execution, ExecutionStats::default());
        assert_eq!(failure.staging.tensor_count, 0);
        assert!(failure.report.applied.is_empty());
        assert!(!output.exists());
    }

    /// Invalid public plans remain checkpoint errors after the initial source hash.
    #[test]
    fn malformed_conversion_plan_never_reaches_the_quantization_sink() {
        let root = tempfile::tempdir().unwrap();
        let (manifest, mut plan) = identity_fixture(root.path(), &["weight"]);
        plan.tensors[0].sources[0].data_offsets[1] -= 1;
        let output = root.path().join("bad-plan.bpk");
        let failure = quantize_local_checkpoint_to_burnpack(
            &manifest,
            &plan,
            &output,
            QuantizedCheckpointOptions::default(),
        )
        .unwrap_err();
        assert!(
            matches!(failure.cause, CheckpointQuantizationFailureCause::Checkpoint(
            CheckpointError::InvalidMapping { ref message }) if message.contains("offsets describe"))
        );
        assert!(failure.failed_target.is_none());
        assert!(failure.report.applied.is_empty());
        assert!(failure.report.quantized.is_empty());
        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.staging.tensor_count, 0);
        assert_eq!(
            failure.provenance.pre_execution.unwrap().bytes_hashed,
            manifest.artifact_bytes
        );
        assert_eq!(failure.provenance.post_execution, None);
        let cause = StdError::source(failure.as_ref()).unwrap();
        assert!(
            cause
                .source()
                .unwrap()
                .downcast_ref::<CheckpointError>()
                .is_some()
        );
        assert!(!output.exists());
    }

    /// A late checkpoint failure retains quantized and excluded successful targets.
    #[test]
    fn checkpoint_failure_keeps_exact_partial_quantized_and_exclusion_reports() {
        let root = tempfile::tempdir().unwrap();
        let (manifest, mut plan) = identity_fixture(
            root.path(),
            &["a.weight", "b.embed_tokens.weight", "z.weight"],
        );
        plan.tensors
            .iter_mut()
            .find(|tensor| tensor.target == "z.weight")
            .unwrap()
            .sources[0]
            .name = "missing".into();
        let output = root.path().join("partial.bpk");
        let failure = quantize_local_checkpoint_to_burnpack(
            &manifest,
            &plan,
            &output,
            QuantizedCheckpointOptions::default(),
        )
        .unwrap_err();
        assert!(
            matches!(failure.cause, CheckpointQuantizationFailureCause::Checkpoint(
            CheckpointError::PlanManifestMismatch { ref tensor, .. }) if tensor == "missing")
        );
        assert_eq!(failure.failed_target.as_deref(), Some("z.weight"));
        assert_eq!(
            failure.report.applied,
            ["a.weight", "b.embed_tokens.weight"]
        );
        assert_eq!(failure.report.quantized.len(), 1);
        assert_eq!(failure.report.quantized[0].target, "a.weight");
        assert_eq!(
            failure.excluded,
            [QuantizationExclusionRecord {
                target: "b.embed_tokens.weight".into(),
                reason: QuantizationExclusion::Embedding
            }]
        );
        assert_eq!(failure.execution.source_reads, 2);
        assert_eq!(failure.execution.tensors_applied, 2);
        assert_eq!(failure.staging.tensor_count, 1);
        assert_eq!(
            failure.staging.staged_payload_bytes,
            failure.report.quantized[0].bytes
        );
        assert_eq!(failure.provenance.post_execution, None);
        assert!(failure.to_string().contains("failed at `z.weight`"));
        assert!(failure.cause.to_string().contains("missing"));
        assert!(!output.exists());
    }

    /// Binding limits fail only after both hash passes and retain excluded targets.
    #[test]
    fn provenance_binding_limit_preserves_complete_conversion_progress() {
        let root = tempfile::tempdir().unwrap();
        let (manifest, plan) = identity_fixture(root.path(), &["embed_tokens.weight"]);
        let output = root.path().join("bound-limit.bpk");
        let failure = quantize_local_checkpoint_to_burnpack(
            &manifest,
            &plan,
            &output,
            QuantizedCheckpointOptions {
                limits: BurnpackSessionLimits {
                    max_manifest_bytes: 1,
                    ..BurnpackSessionLimits::default()
                },
                ..QuantizedCheckpointOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            failure.cause,
            CheckpointQuantizationFailureCause::Quantization(
                QuantizationError::SessionLimitExceeded {
                    kind: "estimated manifest",
                    limit: 1,
                    ..
                }
            )
        ));
        assert!(failure.failed_target.is_none());
        assert_eq!(failure.report.applied, ["embed_tokens.weight"]);
        assert!(failure.report.quantized.is_empty());
        assert_eq!(
            failure.excluded,
            [QuantizationExclusionRecord {
                target: "embed_tokens.weight".into(),
                reason: QuantizationExclusion::Embedding
            }]
        );
        assert_eq!(failure.execution.tensors_applied, 1);
        assert_eq!(failure.staging.tensor_count, 0);
        for stats in [
            failure.provenance.pre_execution.unwrap(),
            failure.provenance.post_execution.unwrap(),
        ] {
            assert_eq!(stats.passes_completed, 1);
            assert_eq!(stats.bytes_hashed, manifest.artifact_bytes);
        }
        assert!(!output.exists());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
    }

    /// Change diagnostics distinguish removed, added, modified, and plan-only state.
    #[test]
    fn changed_shard_diagnostic_identifies_membership_digest_and_plan_changes() {
        let root = tempfile::tempdir().unwrap();
        let (manifest, plan) = identity_fixture(root.path(), &["weight"]);
        let (before, _) = super::super::hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
        for case in ["remove", "add", "digest", "plan"] {
            let mut after = before.clone();
            match case {
                "remove" => after.checkpoint_shards.clear(),
                "add" => {
                    after.checkpoint_shards.insert(
                        "new.safetensors".into(),
                        before.checkpoint_shards["model.safetensors"].clone(),
                    );
                }
                "digest" => {
                    after
                        .checkpoint_shards
                        .get_mut("model.safetensors")
                        .unwrap()
                        .sha256 = "0".repeat(64)
                }
                "plan" => after.load_plan_sha256 = "0".repeat(64),
                _ => unreachable!(),
            }
            assert_eq!(
                first_changed_shard(&before, &after),
                match case {
                    "add" => "new.safetensors",
                    "plan" => "<load-plan>",
                    _ => "model.safetensors",
                }
            );
        }
    }
}
