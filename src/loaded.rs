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
    /// Burnpack manifest version used by this load.
    format_version: u32,
    /// Packed payload validation completed before installing model tensors.
    preflight: PackedPreflightStats,
    /// Packed tensor decoding and installation work.
    packed: PackedLoadStats,
    /// Full-checkpoint hashing passes that bind version-two sidecars to their source.
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

/// Every component of a [`LoadedModel`], moved out by name.
///
/// Returned by [`LoadedModel::into_named_parts`]. Destructure only what you
/// need, for example
/// `let LoadedModelParts { model, tokenizer, .. } = loaded.into_named_parts();`.
/// The struct is non-exhaustive so later audit domains can be added without
/// breaking such patterns.
#[derive(Debug)]
#[non_exhaustive]
pub struct LoadedModelParts<M> {
    /// Strictly loaded executable model.
    pub model: M,
    /// Local tokenizer when the load attached one.
    pub tokenizer: Option<ModelTokenizer>,
    /// Validated generation defaults normalized for the model vocabulary.
    pub generation_defaults: GenerationConfig,
    /// Local artifact path and recorded upstream identity.
    pub source: ModelSourceMetadata,
    /// Finalized tensor-level load accounting.
    pub load_report: LoadReport,
    /// SafeTensors plan-execution counters, when the loader retained them.
    pub execution_stats: Option<ExecutionStats>,
    /// Burnpack preflight, decode, and provenance counters of a quantized load.
    pub quantized_load_audit: Option<QuantizedLoadAudit>,
}

impl<M> From<LoadedModel<M>> for LoadedModelParts<M> {
    /// Moves every component out of the bundle by name.
    fn from(loaded: LoadedModel<M>) -> Self {
        loaded.into_named_parts()
    }
}

/// An executable model and all host-side state established by a strict load.
#[derive(Debug)]
pub struct LoadedModel<M> {
    /// Strictly loaded executable model.
    model: M,
    /// Local tokenizer when the bundle includes text-processing artifacts.
    tokenizer: Option<ModelTokenizer>,
    /// Validated checkpoint defaults used to initialize generation requests.
    generation_defaults: GenerationConfig,
    /// Local artifact path and recorded upstream identity.
    source: ModelSourceMetadata,
    /// Finalized accounting of all checkpoint tensor dispositions.
    load_report: LoadReport,
    /// SafeTensors execution counters when supplied by the loader.
    execution_stats: Option<ExecutionStats>,
    /// Additional Burnpack validation and provenance work when applicable.
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
        #[cfg(feature = "tokenizers")]
        if let Some(tokenizer) = &tokenizer {
            tokenizer.validate_model_vocabulary(vocabulary_size)?;
        }
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

    /// Consumes the bundle and returns every component in a named struct.
    pub fn into_named_parts(self) -> LoadedModelParts<M> {
        LoadedModelParts {
            model: self.model,
            tokenizer: self.tokenizer,
            generation_defaults: self.generation_defaults,
            source: self.source,
            load_report: self.load_report,
            execution_stats: self.execution_stats,
            quantized_load_audit: self.quantized_load_audit,
        }
    }

    /// Consumes the bundle using the original five-component compatibility API.
    ///
    /// This method intentionally omits optional checkpoint execution counters.
    #[deprecated(note = "use `into_named_parts`, which names every component")]
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
    #[deprecated(note = "use `into_named_parts`, which names every component")]
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
    /// The first six elements match the execution-stats tuple. Full-precision
    /// bundles return `None` for the final element.
    #[deprecated(note = "use `into_named_parts`, which names every component")]
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
#[path = "loaded_tests.rs"]
mod tests;
