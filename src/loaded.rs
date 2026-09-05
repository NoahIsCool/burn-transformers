//! Validated bundles returned by architecture-specific local model loaders.
//!
//! Loading has several outputs that must remain associated with the executable
//! module: tokenizer state, generation defaults, immutable checkpoint
//! provenance, and the exact tensor accounting report. [`LoadedModel`] keeps
//! those values together so an application cannot accidentally report one
//! revision while executing another model. The bundle performs no network I/O;
//! [`ModelSourceMetadata`] canonicalizes an already-present local directory.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    Error, GenerationConfig, Result,
    checkpoint::{
        ExecutionStats, LoadReport,
        quantization::{
            AuditedQuantizedExecutionOutcome, BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2, PackedLoadStats, PackedPreflightStats,
            ProvenanceExecutionStats, ProvenanceHashStats,
        },
    },
};

/// Tokenizer type carried by [`LoadedModel`] when tokenizer support is enabled.
#[cfg(feature = "tokenizers")]
pub type ModelTokenizer = crate::LocalTokenizer;

/// Uninhabited tokenizer marker used when the optional tokenizer feature is off.
///
/// Consequently, `LoadedModel::tokenizer()` can only return `None` in a build
/// without `tokenizers`, while the bundle's public shape remains unchanged.
#[cfg(not(feature = "tokenizers"))]
#[derive(Debug)]
pub enum ModelTokenizer {}

/// Immutable identity and local origin of a loaded model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSourceMetadata {
    /// Canonical local directory containing the model artifacts.
    root: PathBuf,
    /// Architecture discriminator read from the checkpoint configuration.
    model_type: String,
    /// Optional upstream repository name, such as `Qwen/Qwen3.5-0.8B`.
    repository: Option<String>,
    /// Optional immutable upstream commit or content revision.
    revision: Option<String>,
}

impl ModelSourceMetadata {
    /// Validates and canonicalizes one existing local artifact directory.
    pub fn local(root: impl AsRef<Path>, model_type: impl Into<String>) -> Result<Self> {
        let supplied_root = root.as_ref();
        let root = supplied_root
            .canonicalize()
            .map_err(|source| Error::io(supplied_root, source))?;
        if !root.is_dir() {
            return Err(Error::Artifact {
                path: root,
                reason: "model source root must be a directory".into(),
            });
        }
        let model_type = validate_label("model_type", model_type.into())?;
        Ok(Self {
            root,
            model_type,
            repository: None,
            revision: None,
        })
    }

    /// Attaches a non-empty upstream repository identifier.
    pub fn with_repository(mut self, repository: impl Into<String>) -> Result<Self> {
        self.repository = Some(validate_label("repository", repository.into())?);
        Ok(self)
    }

    /// Attaches a non-empty immutable revision identifier.
    pub fn with_revision(mut self, revision: impl Into<String>) -> Result<Self> {
        self.revision = Some(validate_label("revision", revision.into())?);
        Ok(self)
    }

    /// Returns the canonical local artifact root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the checkpoint architecture discriminator.
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    /// Returns the upstream repository identifier when recorded.
    pub fn repository(&self) -> Option<&str> {
        self.repository.as_deref()
    }

    /// Returns the immutable upstream revision when recorded.
    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }
}

/// Sidecar-specific counters retained after a successful quantized load.
///
/// A Burnpack overlay performs work outside the ordinary SafeTensors executor:
/// every packed payload is validated before model mutation, selected payloads
/// are decoded into Burn's quantized representation, and version-two artifacts
/// hash the complete source checkpoint before and after installation. These
/// counters stay separate from [`ExecutionStats`] because sidecar bytes and
/// provenance passes are not checkpoint tensor reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedLoadAudit {
    format_version: u32,
    preflight: PackedPreflightStats,
    packed: PackedLoadStats,
    provenance: ProvenanceExecutionStats,
}

impl QuantizedLoadAudit {
    /// Returns the exact Burnpack manifest format used for the successful load.
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Returns mandatory validation work completed before model mutation.
    pub fn preflight(&self) -> &PackedPreflightStats {
        &self.preflight
    }

    /// Returns packed decode and installation counters.
    pub fn packed(&self) -> &PackedLoadStats {
        &self.packed
    }

    /// Returns full-checkpoint hashing counters for a provenance-bound sidecar.
    ///
    /// Legacy version-one Burnpacks return an audit whose two hash passes are
    /// both `None`; a successful version-two local load records two completed
    /// passes.
    pub fn provenance(&self) -> &ProvenanceExecutionStats {
        &self.provenance
    }
}

/// Optional execution domains forwarded through the common bundle validator.
#[derive(Default)]
struct RetainedLoadAudit {
    /// Generic SafeTensors plan execution counters.
    execution: Option<ExecutionStats>,
    /// Burnpack-only preflight and provenance counters.
    quantized: Option<QuantizedLoadAudit>,
}

/// An executable model and all host-side state established by a strict load.
#[derive(Debug)]
pub struct LoadedModel<M> {
    model: M,
    tokenizer: Option<ModelTokenizer>,
    generation_defaults: GenerationConfig,
    source: ModelSourceMetadata,
    load_report: LoadReport,
    execution_stats: Option<ExecutionStats>,
    quantized_load_audit: Option<QuantizedLoadAudit>,
}

impl<M> LoadedModel<M> {
    /// Builds a bundle after validating generation and checkpoint invariants.
    ///
    /// Explicitly allowlisted omissions remain valid, but unexplained missing or
    /// unexpected tensors are rejected. The report is finalized here so every
    /// exposed category has deterministic ordering and no duplicates.
    pub fn new(
        model: M,
        tokenizer: Option<ModelTokenizer>,
        generation_defaults: GenerationConfig,
        source: ModelSourceMetadata,
        load_report: LoadReport,
        vocabulary_size: usize,
    ) -> Result<Self> {
        Self::build(
            model,
            tokenizer,
            generation_defaults,
            source,
            load_report,
            RetainedLoadAudit::default(),
            vocabulary_size,
        )
    }

    /// Builds a bundle with the exact counters from checkpoint-plan execution.
    ///
    /// Automatic loaders use this constructor after successful pass-two
    /// execution. Callers assembling a bundle from an architecture-specific
    /// [`crate::checkpoint::ExecutionOutcome`] should do the same so shard I/O,
    /// source reads, and bounded host-staging measurements remain attached to
    /// the model whose parameters they describe.
    pub fn new_with_execution_stats(
        model: M,
        tokenizer: Option<ModelTokenizer>,
        generation_defaults: GenerationConfig,
        source: ModelSourceMetadata,
        load_report: LoadReport,
        execution_stats: ExecutionStats,
        vocabulary_size: usize,
    ) -> Result<Self> {
        Self::build(
            model,
            tokenizer,
            generation_defaults,
            source,
            load_report,
            RetainedLoadAudit {
                execution: Some(execution_stats),
                quantized: None,
            },
            vocabulary_size,
        )
    }

    /// Builds a bundle from a complete successful quantized execution outcome.
    ///
    /// Unlike [`Self::new_with_execution_stats`], this constructor retains the
    /// Burnpack preflight/decode and source-provenance counters in addition to
    /// the generic checkpoint executor counters. Cross-domain relationships are
    /// validated before the model can be presented as successfully loaded.
    pub fn new_with_quantized_execution(
        model: M,
        tokenizer: Option<ModelTokenizer>,
        generation_defaults: GenerationConfig,
        source: ModelSourceMetadata,
        outcome: AuditedQuantizedExecutionOutcome,
        vocabulary_size: usize,
    ) -> Result<Self> {
        let AuditedQuantizedExecutionOutcome {
            format_version,
            preflight,
            execution: outcome,
        } = outcome;
        let report = outcome.report;
        let execution = outcome.execution;
        Self::build(
            model,
            tokenizer,
            generation_defaults,
            source,
            report,
            RetainedLoadAudit {
                execution: Some(execution),
                quantized: Some(QuantizedLoadAudit {
                    format_version,
                    preflight,
                    packed: outcome.packed,
                    provenance: outcome.provenance,
                }),
            },
            vocabulary_size,
        )
    }

    /// Validates shared invariants and constructs a bundle with optional counters.
    fn build(
        model: M,
        tokenizer: Option<ModelTokenizer>,
        generation_defaults: GenerationConfig,
        source: ModelSourceMetadata,
        mut load_report: LoadReport,
        retained_audit: RetainedLoadAudit,
        vocabulary_size: usize,
    ) -> Result<Self> {
        let RetainedLoadAudit {
            execution: execution_stats,
            quantized: quantized_load_audit,
        } = retained_audit;
        generation_defaults.validate(vocabulary_size)?;
        load_report.finalize();
        if !load_report.is_strictly_valid() {
            return Err(Error::Checkpoint(format!(
                "cannot construct LoadedModel: {} missing and {} unexpected tensors remain",
                load_report.missing.len(),
                load_report.unexpected.len()
            )));
        }
        if load_report.equivalence_verified != load_report.equivalence_planned {
            return Err(Error::Checkpoint(format!(
                "cannot construct LoadedModel: verified {} of {} planned tensor equivalences",
                load_report.equivalence_verified.len(),
                load_report.equivalence_planned.len()
            )));
        }
        if let Some(stats) = &execution_stats {
            validate_execution_stats(&load_report, stats)?;
        }
        if let Some(audit) = &quantized_load_audit {
            validate_quantized_load_audit(&load_report, audit)?;
        }
        Ok(Self {
            model,
            tokenizer,
            generation_defaults,
            source,
            load_report,
            execution_stats,
            quantized_load_audit,
        })
    }

    /// Returns the executable Burn module.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Returns mutable access to the executable module without touching metadata.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Returns the local tokenizer when this load attached one.
    pub fn tokenizer(&self) -> Option<&ModelTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Returns validated generation defaults normalized for this vocabulary.
    pub fn generation_defaults(&self) -> &GenerationConfig {
        &self.generation_defaults
    }

    /// Returns immutable local and upstream source metadata.
    pub fn source(&self) -> &ModelSourceMetadata {
        &self.source
    }

    /// Returns deterministic tensor-level load accounting.
    pub fn load_report(&self) -> &LoadReport {
        &self.load_report
    }

    /// Returns checkpoint-plan execution counters when they were retained.
    ///
    /// Values are present for models returned by the automatic local loaders.
    /// A bundle constructed through the compatibility [`Self::new`] API has no
    /// executor context and therefore returns `None`.
    pub fn execution_stats(&self) -> Option<&ExecutionStats> {
        self.execution_stats.as_ref()
    }

    /// Returns Burnpack and provenance counters for a quantized load.
    ///
    /// Full-precision loads and bundles built through the compatibility
    /// constructors return `None`.
    pub fn quantized_load_audit(&self) -> Option<&QuantizedLoadAudit> {
        self.quantized_load_audit.as_ref()
    }

    /// Consumes the bundle and returns the executable module.
    pub fn into_model(self) -> M {
        self.model
    }

    /// Consumes the bundle using the original five-component compatibility API.
    ///
    /// This method intentionally omits optional checkpoint execution counters.
    /// Use [`Self::into_parts_with_execution_stats`] when retaining that audit
    /// data matters.
    pub fn into_parts(
        self,
    ) -> (
        M,
        Option<ModelTokenizer>,
        GenerationConfig,
        ModelSourceMetadata,
        LoadReport,
    ) {
        (
            self.model,
            self.tokenizer,
            self.generation_defaults,
            self.source,
            self.load_report,
        )
    }

    /// Consumes the bundle and returns every component, including load counters.
    pub fn into_parts_with_execution_stats(
        self,
    ) -> (
        M,
        Option<ModelTokenizer>,
        GenerationConfig,
        ModelSourceMetadata,
        LoadReport,
        Option<ExecutionStats>,
    ) {
        (
            self.model,
            self.tokenizer,
            self.generation_defaults,
            self.source,
            self.load_report,
            self.execution_stats,
        )
    }

    /// Consumes the bundle and additionally returns its quantized-load audit.
    ///
    /// The first six elements intentionally match
    /// [`Self::into_parts_with_execution_stats`]. Full-precision bundles return
    /// `None` for the final element.
    pub fn into_parts_with_quantized_load_audit(
        self,
    ) -> (
        M,
        Option<ModelTokenizer>,
        GenerationConfig,
        ModelSourceMetadata,
        LoadReport,
        Option<ExecutionStats>,
        Option<QuantizedLoadAudit>,
    ) {
        (
            self.model,
            self.tokenizer,
            self.generation_defaults,
            self.source,
            self.load_report,
            self.execution_stats,
            self.quantized_load_audit,
        )
    }
}

/// Checks relationships between a completed packed pass and its strict report.
fn validate_quantized_load_audit(report: &LoadReport, audit: &QuantizedLoadAudit) -> Result<()> {
    let packed = audit.packed;
    let preflight = audit.preflight;
    if preflight.passes_started != 1 || preflight.passes_completed != 1 {
        return Err(Error::Checkpoint(format!(
            "successful quantized execution requires one completed Burnpack preflight, recorded {} started and {} completed",
            preflight.passes_started, preflight.passes_completed
        )));
    }
    if preflight.tensor_reads != packed.tensor_reads {
        return Err(Error::Checkpoint(format!(
            "Burnpack preflight read {} tensors but execution decoded {}",
            preflight.tensor_reads, packed.tensor_reads
        )));
    }
    if preflight.payload_bytes != packed.payload_bytes {
        return Err(Error::Checkpoint(format!(
            "Burnpack preflight read {} payload bytes but execution decoded {}",
            preflight.payload_bytes, packed.payload_bytes
        )));
    }
    if preflight.largest_payload_bytes != packed.largest_payload_bytes {
        return Err(Error::Checkpoint(format!(
            "Burnpack preflight largest payload is {}, but execution records {}",
            preflight.largest_payload_bytes, packed.largest_payload_bytes
        )));
    }
    if packed.tensors_applied > 0 {
        if packed.largest_payload_bytes == 0 || preflight.peak_owned_payload_bytes == 0 {
            return Err(Error::Checkpoint(
                "non-empty Burnpack execution requires non-zero largest-payload and owned-allocation peak counters".into(),
            ));
        }
        if packed.largest_payload_bytes > packed.payload_bytes {
            return Err(Error::Checkpoint(format!(
                "largest Burnpack payload {} exceeds all decoded payload bytes {}",
                packed.largest_payload_bytes, packed.payload_bytes
            )));
        }
        if preflight.peak_owned_payload_bytes < packed.largest_payload_bytes {
            return Err(Error::Checkpoint(format!(
                "Burnpack owned-payload peak {} is smaller than its largest decoded payload {}",
                preflight.peak_owned_payload_bytes, packed.largest_payload_bytes
            )));
        }
    }
    let tensor_reads = usize::try_from(packed.tensor_reads)
        .map_err(|_| Error::Checkpoint("Burnpack tensor read count exceeds usize".into()))?;
    if tensor_reads != packed.tensors_applied {
        return Err(Error::Checkpoint(format!(
            "Burnpack decoded {} tensors but installed {}",
            packed.tensor_reads, packed.tensors_applied
        )));
    }
    if packed.tensors_applied != report.quantized.len() {
        return Err(Error::Checkpoint(format!(
            "Burnpack installed {} tensors but the load report contains {} quantized records",
            packed.tensors_applied,
            report.quantized.len()
        )));
    }
    let reported_payload_bytes = report
        .quantized
        .iter()
        .try_fold(0_u64, |total, record| total.checked_add(record.bytes));
    let reported_payload_bytes = reported_payload_bytes.ok_or_else(|| {
        Error::Checkpoint("quantized load-report payload byte count overflowed".into())
    })?;
    if reported_payload_bytes != packed.payload_bytes {
        return Err(Error::Checkpoint(format!(
            "Burnpack decoded {} payload bytes but the load report accounts for {reported_payload_bytes}",
            packed.payload_bytes
        )));
    }
    let reported_largest_payload = report
        .quantized
        .iter()
        .map(|record| record.bytes)
        .max()
        .unwrap_or(0);
    if packed.largest_payload_bytes != reported_largest_payload {
        return Err(Error::Checkpoint(format!(
            "largest Burnpack payload is {}, but the load report records {reported_largest_payload}",
            packed.largest_payload_bytes
        )));
    }
    match audit.format_version {
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V1 => match (
            audit.provenance.pre_execution,
            audit.provenance.post_execution,
        ) {
            (None, None) => Ok(()),
            _ => Err(Error::Checkpoint(
                "legacy Burnpack v1 execution cannot claim checkpoint provenance hash passes".into(),
            )),
        },
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2 => match (
            audit.provenance.pre_execution,
            audit.provenance.post_execution,
        ) {
            (Some(before), Some(after)) => validate_provenance_pair(
                before,
                after,
                report.memory.map(|memory| memory.artifact_bytes),
            ),
            _ => Err(Error::Checkpoint(
                "successful Burnpack v2 execution requires both pre- and post-execution hash audits".into(),
            )),
        },
        format_version => Err(Error::Checkpoint(format!(
            "successful quantized execution recorded unsupported Burnpack format version {format_version}"
        ))),
    }
}

/// Checks that two successful provenance passes cover the same checkpoint bytes.
fn validate_provenance_pair(
    before: ProvenanceHashStats,
    after: ProvenanceHashStats,
    expected_artifact_bytes: Option<u64>,
) -> Result<()> {
    for (phase, stats) in [("pre-execution", before), ("post-execution", after)] {
        if stats.passes_completed != 1 {
            return Err(Error::Checkpoint(format!(
                "successful quantized execution requires one completed {phase} provenance pass, recorded {}",
                stats.passes_completed
            )));
        }
        if stats.shards_hashed == 0 || stats.bytes_hashed == 0 || stats.peak_buffer_bytes == 0 {
            return Err(Error::Checkpoint(format!(
                "successful {phase} provenance pass must hash non-empty checkpoint content"
            )));
        }
        if let Some(expected) = expected_artifact_bytes
            && stats.bytes_hashed != expected
        {
            return Err(Error::Checkpoint(format!(
                "{phase} provenance hashed {} bytes but the load plan records {expected} checkpoint artifact bytes",
                stats.bytes_hashed
            )));
        }
    }
    if before.shards_hashed != after.shards_hashed || before.bytes_hashed != after.bytes_hashed {
        return Err(Error::Checkpoint(format!(
            "pre/post provenance coverage differs: {} shards and {} bytes before, {} shards and {} bytes after",
            before.shards_hashed, before.bytes_hashed, after.shards_hashed, after.bytes_hashed
        )));
    }
    Ok(())
}

/// Checks the redundant invariants that make retained execution counters auditable.
fn validate_execution_stats(report: &LoadReport, stats: &ExecutionStats) -> Result<()> {
    let writes = stats
        .tensors_applied
        .checked_add(stats.aliases_applied)
        .ok_or_else(|| Error::Checkpoint("checkpoint execution write count overflowed".into()))?;
    if writes != report.applied.len() {
        return Err(Error::Checkpoint(format!(
            "checkpoint execution recorded {writes} successful writes but the load report contains {} applied tensors",
            report.applied.len()
        )));
    }
    if stats.source_reads != stats.source.tensor_reads {
        return Err(Error::Checkpoint(format!(
            "checkpoint executor recorded {} source reads but its source recorded {}",
            stats.source_reads, stats.source.tensor_reads
        )));
    }
    if stats.source_payload_bytes != stats.source.payload_bytes_read {
        return Err(Error::Checkpoint(format!(
            "checkpoint executor recorded {} source bytes but its source recorded {}",
            stats.source_payload_bytes, stats.source.payload_bytes_read
        )));
    }
    if report.equivalence_verified != report.equivalence_planned {
        return Err(Error::Checkpoint(format!(
            "checkpoint execution verified {} of {} planned tensor equivalences",
            report.equivalence_verified.len(),
            report.equivalence_planned.len()
        )));
    }
    if stats.equivalences_verified != report.equivalence_verified.len() {
        return Err(Error::Checkpoint(format!(
            "checkpoint executor counted {} verified equivalences but the load report contains {}",
            stats.equivalences_verified,
            report.equivalence_verified.len()
        )));
    }
    let expected_equivalence_reads = u64::try_from(stats.equivalences_verified)
        .ok()
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| Error::Checkpoint("equivalence read count overflowed".into()))?;
    if stats.equivalence_source_reads != expected_equivalence_reads {
        return Err(Error::Checkpoint(format!(
            "checkpoint executor recorded {} equivalence reads for {} verified pairs",
            stats.equivalence_source_reads, stats.equivalences_verified
        )));
    }
    if stats.equivalence_source_payload_bytes > stats.source_payload_bytes {
        return Err(Error::Checkpoint(format!(
            "equivalence source bytes {} exceed all checkpoint source bytes {}",
            stats.equivalence_source_payload_bytes, stats.source_payload_bytes
        )));
    }
    if stats.largest_equivalence_pair_bytes > stats.equivalence_source_payload_bytes {
        return Err(Error::Checkpoint(format!(
            "largest equivalence pair {} exceeds all equivalence source bytes {}",
            stats.largest_equivalence_pair_bytes, stats.equivalence_source_payload_bytes
        )));
    }
    if let Some(memory) = report.memory {
        if stats.largest_equivalence_pair_bytes != memory.largest_equivalence_pair_bytes {
            return Err(Error::Checkpoint(format!(
                "checkpoint executor observed a {}-byte largest equivalence pair but the load plan records {} bytes",
                stats.largest_equivalence_pair_bytes, memory.largest_equivalence_pair_bytes
            )));
        }
        if stats.peak_host_staging_bytes > memory.peak_host_staging_bytes {
            return Err(Error::Checkpoint(format!(
                "checkpoint executor observed a {}-byte host-staging peak above the planned {}-byte bound",
                stats.peak_host_staging_bytes, memory.peak_host_staging_bytes
            )));
        }
    }
    Ok(())
}

/// Rejects empty, surrounding-whitespace, and control-character provenance.
fn validate_label(field: &str, value: String) -> Result<String> {
    if value.is_empty() || value.trim() != value {
        return Err(Error::configuration(
            format!("model_source.{field}"),
            "must be non-empty and contain no surrounding whitespace",
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(Error::configuration(
            format!("model_source.{field}"),
            "must not contain control characters",
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use crate::checkpoint::{
        CheckpointEquivalenceRecord, ExecutionStats, LoadReport, MemoryEstimate,
        QuantizationRecord, SourceStats,
        quantization::{
            AuditedQuantizedExecutionOutcome, BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2, PackedLoadStats, PackedPreflightStats,
            ProvenanceExecutionStats, ProvenanceHashStats, QuantizedExecutionOutcome,
        },
    };

    use super::{LoadedModel, ModelSourceMetadata};

    /// Source metadata canonicalizes roots and preserves pinned provenance.
    #[test]
    fn local_source_is_canonical_and_pinned() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let source = ModelSourceMetadata::local(directory.path(), "qwen3_5_text")
            .expect("valid local source")
            .with_repository("Qwen/Qwen3.5-0.8B")
            .expect("valid repository")
            .with_revision("2fc06364715b967f1860aea9cf38778875588b17")
            .expect("valid revision");
        assert!(source.root().is_absolute());
        assert_eq!(source.model_type(), "qwen3_5_text");
        assert_eq!(source.repository(), Some("Qwen/Qwen3.5-0.8B"));
        assert_eq!(
            source.revision(),
            Some("2fc06364715b967f1860aea9cf38778875588b17")
        );
    }

    /// Invalid roots and ambiguous textual identifiers produce contextual errors.
    #[test]
    fn local_source_rejects_invalid_roots_and_labels() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let file = directory.path().join("config.json");
        std::fs::write(&file, b"{}").expect("fixture file");
        assert!(ModelSourceMetadata::local(&file, "t5").is_err());
        assert!(ModelSourceMetadata::local(directory.path().join("missing"), "t5").is_err());
        assert!(ModelSourceMetadata::local(directory.path(), "").is_err());

        let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
        assert!(
            source
                .clone()
                .with_repository(" google/flan-t5-small")
                .is_err()
        );
        assert!(source.with_revision("bad\nrevision").is_err());
    }

    /// Construction finalizes a strict report and round-trips every component.
    #[test]
    fn loaded_model_validates_and_preserves_components() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
        let mut report = LoadReport::new();
        report.applied = vec!["z.weight".into(), "a.weight".into(), "a.weight".into()];
        let loaded = LoadedModel::new(
            17_u8,
            None,
            Default::default(),
            source.clone(),
            report,
            32_128,
        )
        .expect("strict load should bundle");
        assert_eq!(*loaded.model(), 17);
        assert!(loaded.tokenizer().is_none());
        assert_eq!(loaded.source(), &source);
        assert_eq!(loaded.load_report().applied, ["a.weight", "z.weight"]);
        assert_eq!(loaded.generation_defaults().max_new_tokens, 256);
        assert!(loaded.execution_stats().is_none());
        assert!(loaded.quantized_load_audit().is_none());

        let (model, tokenizer, _, unpacked_source, unpacked_report) = loaded.into_parts();
        assert_eq!(model, 17);
        assert!(tokenizer.is_none());
        assert_eq!(unpacked_source, source);
        assert_eq!(unpacked_report.applied, ["a.weight", "z.weight"]);
    }

    /// Execution-aware construction and consumption retain exact audit counters.
    #[test]
    fn loaded_model_preserves_execution_stats_without_changing_legacy_parts() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let source =
            ModelSourceMetadata::local(directory.path(), "qwen3_moe").expect("valid local source");
        let stats = ExecutionStats {
            tensors_applied: 2,
            aliases_applied: 1,
            equivalences_verified: 0,
            equivalence_source_reads: 0,
            equivalence_source_payload_bytes: 0,
            source_reads: 2,
            source_payload_bytes: 96,
            delivered_tensor_bytes: 96,
            largest_source_payload_bytes: 64,
            largest_output_tensor_bytes: 64,
            largest_equivalence_pair_bytes: 0,
            peak_host_staging_bytes: 128,
            source: SourceStats {
                shard_opens: 1,
                tensor_reads: 2,
                payload_bytes_read: 96,
                max_open_shards: 1,
                shard_open_order: vec!["model.safetensors".into()],
            },
        };
        let mut report = LoadReport::new();
        report.applied = vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into(),
            "lm_head.weight".into(),
        ];
        let loaded = LoadedModel::new_with_execution_stats(
            23_u8,
            None,
            Default::default(),
            source.clone(),
            report,
            stats.clone(),
            32_128,
        )
        .expect("strict execution-aware bundle");

        assert_eq!(loaded.execution_stats(), Some(&stats));
        assert!(loaded.quantized_load_audit().is_none());
        let (model, tokenizer, _, unpacked_source, unpacked_report, unpacked_stats) =
            loaded.into_parts_with_execution_stats();
        assert_eq!(model, 23);
        assert!(tokenizer.is_none());
        assert_eq!(unpacked_source, source);
        assert!(unpacked_report.is_strictly_valid());
        assert_eq!(unpacked_stats, Some(stats));
    }

    /// Quantized construction retains all three execution counter domains.
    #[test]
    fn loaded_model_preserves_quantized_load_audit() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let source =
            ModelSourceMetadata::local(directory.path(), "qwen3_moe").expect("valid source");
        let mut report = LoadReport::new();
        report
            .applied
            .push("model.layers.0.mlp.down_proj.weight".into());
        report.quantized.push(QuantizationRecord {
            target: "model.layers.0.mlp.down_proj.weight".into(),
            scheme: "burn-qfloat-symmetric-q8-g64-block-1xg".into(),
            bytes: 72,
        });
        report.memory = Some(MemoryEstimate {
            artifact_bytes: 4_096,
            peak_host_staging_bytes: 256,
            ..MemoryEstimate::default()
        });
        let execution = ExecutionStats {
            tensors_applied: 1,
            delivered_tensor_bytes: 256,
            largest_output_tensor_bytes: 256,
            peak_host_staging_bytes: 256,
            ..ExecutionStats::default()
        };
        let hash_pass = ProvenanceHashStats {
            passes_completed: 1,
            shards_hashed: 2,
            bytes_hashed: 4_096,
            peak_buffer_bytes: 1_048_576,
        };
        let packed = PackedLoadStats {
            tensor_reads: 1,
            payload_bytes: 72,
            largest_payload_bytes: 72,
            tensors_applied: 1,
        };
        let preflight = PackedPreflightStats {
            passes_started: 1,
            passes_completed: 1,
            tensor_reads: 1,
            payload_bytes: 72,
            largest_payload_bytes: 72,
            peak_owned_payload_bytes: 144,
        };
        let provenance = ProvenanceExecutionStats {
            pre_execution: Some(hash_pass),
            post_execution: Some(hash_pass),
        };
        let loaded = LoadedModel::new_with_quantized_execution(
            29_u8,
            None,
            Default::default(),
            source.clone(),
            AuditedQuantizedExecutionOutcome {
                format_version: BURNPACK_QUANTIZATION_FORMAT_VERSION_V2,
                preflight,
                execution: QuantizedExecutionOutcome {
                    report,
                    execution: execution.clone(),
                    packed,
                    provenance,
                },
            },
            32_128,
        )
        .expect("consistent quantized outcome must bundle");

        assert_eq!(loaded.execution_stats(), Some(&execution));
        let audit = loaded
            .quantized_load_audit()
            .expect("quantized outcome must retain its sidecar audit");
        assert_eq!(
            audit.format_version(),
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
        );
        assert_eq!(audit.preflight(), &preflight);
        assert_eq!(audit.packed(), &packed);
        assert_eq!(audit.provenance(), &provenance);

        let (model, tokenizer, _, unpacked_source, unpacked_report, unpacked_stats, unpacked_audit) =
            loaded.into_parts_with_quantized_load_audit();
        assert_eq!(model, 29);
        assert!(tokenizer.is_none());
        assert_eq!(unpacked_source, source);
        assert_eq!(unpacked_report.quantized.len(), 1);
        assert_eq!(unpacked_stats, Some(execution));
        let unpacked_audit = unpacked_audit.expect("retained audit");
        assert_eq!(unpacked_audit.preflight(), &preflight);
        assert_eq!(unpacked_audit.packed(), &packed);
    }

    /// Contradictory packed or provenance counters cannot escape as success.
    #[test]
    fn loaded_model_rejects_inconsistent_quantized_load_audit() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let source =
            ModelSourceMetadata::local(directory.path(), "qwen3_moe").expect("valid source");
        let mut report = LoadReport::new();
        report
            .applied
            .push("model.layers.0.mlp.down_proj.weight".into());
        report.quantized.push(QuantizationRecord {
            target: "model.layers.0.mlp.down_proj.weight".into(),
            scheme: "burn-qfloat-symmetric-q8-g64-block-1xg".into(),
            bytes: 72,
        });
        report.memory = Some(MemoryEstimate {
            artifact_bytes: 4_096,
            ..MemoryEstimate::default()
        });
        let execution = ExecutionStats {
            tensors_applied: 1,
            ..ExecutionStats::default()
        };
        let packed = PackedLoadStats {
            tensor_reads: 1,
            payload_bytes: 72,
            largest_payload_bytes: 72,
            tensors_applied: 1,
        };
        let preflight = PackedPreflightStats {
            passes_started: 1,
            passes_completed: 1,
            tensor_reads: 1,
            payload_bytes: 72,
            largest_payload_bytes: 72,
            peak_owned_payload_bytes: 144,
        };
        let hash_pass = ProvenanceHashStats {
            passes_completed: 1,
            shards_hashed: 1,
            bytes_hashed: 4_096,
            peak_buffer_bytes: 1_048_576,
        };
        let construct = |format_version, report, preflight, packed, provenance| {
            LoadedModel::new_with_quantized_execution(
                (),
                None,
                Default::default(),
                source.clone(),
                AuditedQuantizedExecutionOutcome {
                    format_version,
                    preflight,
                    execution: QuantizedExecutionOutcome {
                        report,
                        execution: execution.clone(),
                        packed,
                        provenance,
                    },
                },
                32,
            )
        };

        for inconsistent in [
            PackedPreflightStats {
                passes_started: 0,
                ..preflight
            },
            PackedPreflightStats {
                passes_completed: 0,
                ..preflight
            },
            PackedPreflightStats {
                tensor_reads: 2,
                ..preflight
            },
            PackedPreflightStats {
                payload_bytes: 73,
                ..preflight
            },
            PackedPreflightStats {
                largest_payload_bytes: 73,
                ..preflight
            },
            PackedPreflightStats {
                peak_owned_payload_bytes: 0,
                ..preflight
            },
            PackedPreflightStats {
                peak_owned_payload_bytes: 71,
                ..preflight
            },
        ] {
            assert!(
                construct(
                    BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
                    report.clone(),
                    inconsistent,
                    packed,
                    ProvenanceExecutionStats::default(),
                )
                .is_err()
            );
        }

        for inconsistent in [
            PackedLoadStats {
                tensor_reads: 2,
                ..packed
            },
            PackedLoadStats {
                payload_bytes: 73,
                ..packed
            },
            PackedLoadStats {
                tensors_applied: 0,
                ..packed
            },
        ] {
            assert!(
                construct(
                    BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
                    report.clone(),
                    preflight,
                    inconsistent,
                    ProvenanceExecutionStats::default(),
                )
                .is_err()
            );
        }

        let zero_largest = PackedLoadStats {
            largest_payload_bytes: 0,
            ..packed
        };
        assert!(
            construct(
                BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
                report.clone(),
                PackedPreflightStats {
                    largest_payload_bytes: 0,
                    ..preflight
                },
                zero_largest,
                ProvenanceExecutionStats::default(),
            )
            .is_err()
        );
        let largest_above_total = PackedLoadStats {
            payload_bytes: 71,
            ..packed
        };
        assert!(
            construct(
                BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
                report.clone(),
                PackedPreflightStats {
                    payload_bytes: 71,
                    ..preflight
                },
                largest_above_total,
                ProvenanceExecutionStats::default(),
            )
            .is_err()
        );

        assert!(
            construct(
                BURNPACK_QUANTIZATION_FORMAT_VERSION_V2,
                report.clone(),
                preflight,
                packed,
                ProvenanceExecutionStats::default(),
            )
            .is_err()
        );
        assert!(
            construct(
                99,
                report.clone(),
                preflight,
                packed,
                ProvenanceExecutionStats::default(),
            )
            .is_err()
        );
        construct(
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
            report.clone(),
            preflight,
            packed,
            ProvenanceExecutionStats::default(),
        )
        .expect("v1 explicitly explains the absence of provenance hashes");

        assert!(
            construct(
                BURNPACK_QUANTIZATION_FORMAT_VERSION_V2,
                report.clone(),
                preflight,
                packed,
                ProvenanceExecutionStats {
                    pre_execution: Some(hash_pass),
                    post_execution: None,
                }
            )
            .is_err()
        );
        assert!(
            construct(
                BURNPACK_QUANTIZATION_FORMAT_VERSION_V2,
                report.clone(),
                preflight,
                packed,
                ProvenanceExecutionStats {
                    pre_execution: Some(ProvenanceHashStats {
                        passes_completed: 0,
                        ..hash_pass
                    }),
                    post_execution: Some(hash_pass),
                }
            )
            .is_err()
        );
        assert!(
            construct(
                BURNPACK_QUANTIZATION_FORMAT_VERSION_V2,
                report.clone(),
                preflight,
                packed,
                ProvenanceExecutionStats {
                    pre_execution: Some(hash_pass),
                    post_execution: Some(ProvenanceHashStats {
                        bytes_hashed: hash_pass.bytes_hashed + 1,
                        ..hash_pass
                    }),
                }
            )
            .is_err()
        );

        let after_wrong_bytes = construct(
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2,
            report.clone(),
            preflight,
            packed,
            ProvenanceExecutionStats {
                pre_execution: Some(hash_pass),
                post_execution: Some(ProvenanceHashStats {
                    bytes_hashed: 4_095,
                    ..hash_pass
                }),
            },
        )
        .expect_err("the post-execution pass must match planned artifact bytes");
        assert!(after_wrong_bytes.to_string().contains("post-execution"));

        let mut overflowing_report = report;
        overflowing_report.applied.push("second.weight".into());
        overflowing_report.quantized[0].bytes = u64::MAX;
        overflowing_report.quantized.push(QuantizationRecord {
            target: "second.weight".into(),
            scheme: "burn-qfloat-symmetric-q8-g64-block-1xg".into(),
            bytes: 1,
        });
        let overflowing_execution = ExecutionStats {
            tensors_applied: 2,
            ..ExecutionStats::default()
        };
        let overflowing = LoadedModel::new_with_quantized_execution(
            (),
            None,
            Default::default(),
            source,
            AuditedQuantizedExecutionOutcome {
                format_version: BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
                preflight: PackedPreflightStats {
                    passes_started: 1,
                    passes_completed: 1,
                    tensor_reads: 2,
                    payload_bytes: u64::MAX,
                    largest_payload_bytes: u64::MAX,
                    peak_owned_payload_bytes: u64::MAX,
                },
                execution: QuantizedExecutionOutcome {
                    report: overflowing_report,
                    execution: overflowing_execution,
                    packed: PackedLoadStats {
                        tensor_reads: 2,
                        payload_bytes: u64::MAX,
                        largest_payload_bytes: u64::MAX,
                        tensors_applied: 2,
                    },
                    provenance: ProvenanceExecutionStats::default(),
                },
            },
            32,
        )
        .expect_err("overflowing report byte sums must fail closed");
        assert!(overflowing.to_string().contains("overflowed"));
    }

    /// Invalid generation controls and non-strict reports cannot escape loaders.
    #[test]
    fn loaded_model_rejects_invalid_runtime_metadata() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
        let invalid_generation = crate::GenerationConfig {
            max_new_tokens: 0,
            ..Default::default()
        };
        assert!(
            LoadedModel::new(
                (),
                None,
                invalid_generation,
                source.clone(),
                LoadReport::new(),
                32
            )
            .is_err()
        );

        let mut report = LoadReport::new();
        report.missing = vec!["decoder.weight".into(), "decoder.weight".into()];
        report.unexpected.push("foreign.weight".into());
        let error = LoadedModel::new((), None, Default::default(), source, report, 32)
            .expect_err("non-strict report must fail");
        assert!(error.to_string().contains("1 missing and 1 unexpected"));
    }

    /// Contradictory retained counters cannot be presented as a successful audit.
    #[test]
    fn loaded_model_rejects_inconsistent_execution_stats() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
        let construct = |stats| {
            LoadedModel::new_with_execution_stats(
                (),
                None,
                Default::default(),
                source.clone(),
                LoadReport::new(),
                stats,
                32,
            )
        };

        let write_overflow = ExecutionStats {
            tensors_applied: usize::MAX,
            aliases_applied: 1,
            ..ExecutionStats::default()
        };
        assert!(
            construct(write_overflow)
                .unwrap_err()
                .to_string()
                .contains("overflowed")
        );

        let write_mismatch = ExecutionStats {
            tensors_applied: 1,
            ..ExecutionStats::default()
        };
        assert!(
            construct(write_mismatch)
                .unwrap_err()
                .to_string()
                .contains("1 successful writes")
        );

        let read_mismatch = ExecutionStats {
            source_reads: 1,
            ..ExecutionStats::default()
        };
        assert!(
            construct(read_mismatch)
                .unwrap_err()
                .to_string()
                .contains("1 source reads")
        );

        let byte_mismatch = ExecutionStats {
            source_payload_bytes: 1,
            ..ExecutionStats::default()
        };
        assert!(
            construct(byte_mismatch)
                .unwrap_err()
                .to_string()
                .contains("1 source bytes")
        );

        let equivalence = CheckpointEquivalenceRecord::new("copy", "owner");
        let equivalence_report = LoadReport {
            equivalence_planned: vec![equivalence.clone()],
            equivalence_verified: vec![equivalence],
            memory: Some(MemoryEstimate {
                largest_equivalence_pair_bytes: 8,
                peak_host_staging_bytes: 8,
                ..MemoryEstimate::default()
            }),
            ..LoadReport::default()
        };
        let equivalence_stats = ExecutionStats {
            equivalences_verified: 1,
            equivalence_source_reads: 2,
            equivalence_source_payload_bytes: 8,
            source_reads: 2,
            source_payload_bytes: 8,
            largest_source_payload_bytes: 4,
            largest_equivalence_pair_bytes: 8,
            peak_host_staging_bytes: 8,
            source: SourceStats {
                tensor_reads: 2,
                payload_bytes_read: 8,
                ..SourceStats::default()
            },
            ..ExecutionStats::default()
        };
        LoadedModel::new_with_execution_stats(
            (),
            None,
            Default::default(),
            source.clone(),
            equivalence_report.clone(),
            equivalence_stats.clone(),
            32,
        )
        .expect("consistent equivalence accounting must remain loadable");

        for stats in [
            ExecutionStats {
                equivalences_verified: 0,
                ..equivalence_stats.clone()
            },
            ExecutionStats {
                equivalence_source_reads: 1,
                ..equivalence_stats.clone()
            },
            ExecutionStats {
                equivalence_source_payload_bytes: 7,
                ..equivalence_stats.clone()
            },
            ExecutionStats {
                largest_equivalence_pair_bytes: 7,
                ..equivalence_stats.clone()
            },
            ExecutionStats {
                peak_host_staging_bytes: 9,
                ..equivalence_stats.clone()
            },
        ] {
            assert!(
                LoadedModel::new_with_execution_stats(
                    (),
                    None,
                    Default::default(),
                    source.clone(),
                    equivalence_report.clone(),
                    stats,
                    32,
                )
                .is_err()
            );
        }

        let incomplete_report = LoadReport {
            equivalence_verified: Vec::new(),
            ..equivalence_report
        };
        assert!(
            LoadedModel::new_with_execution_stats(
                (),
                None,
                Default::default(),
                source,
                incomplete_report,
                equivalence_stats,
                32,
            )
            .unwrap_err()
            .to_string()
            .contains("verified 0 of 1")
        );
    }

    /// The compatibility constructor cannot bless an unexecuted equivalence plan.
    #[test]
    fn loaded_model_rejects_unverified_equivalence_without_execution_stats() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
        let report = LoadReport {
            equivalence_planned: vec![CheckpointEquivalenceRecord::new(
                "decoder.embed_tokens.weight",
                "shared.weight",
            )],
            ..LoadReport::default()
        };

        let error = LoadedModel::new((), None, Default::default(), source, report, 32)
            .expect_err("unverified physical equivalence must not be presented as loaded");

        assert!(error.to_string().contains("0 of 1"));
    }
}
