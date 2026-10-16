//! Persistent quantized artifacts backed by Burn Store's official Burnpack format.
//!
//! Burnpack 0.21 does not expose its string metadata through [`BurnpackStore`],
//! while this module stores a portable flat-group wire layout rather than a
//! backend tensor primitive. Each quantized payload is therefore stored as an
//! ordinary U8 Burnpack tensor. When every matrix row is group-aligned, that
//! wire layout maps exactly to Burn block-QFloat without repacking values. A
//! reserved U8 tensor contains the JSON
//! [`BurnpackQuantizationManifest`] that restores logical shapes, source dtypes,
//! group boundaries, packing rules, and scale layout.
//!
//! A write session appends completed packed tensors to an adjacent staging file
//! and retains only offsets plus metadata. Finalization constructs lazy
//! [`TensorSnapshot`] closures over those ranges. Burn Store writes snapshots one
//! at a time, so finalization materializes no more than one packed tensor payload
//! in memory. The completed official `.bpk` file is synchronized and published
//! with a same-directory hard link, which is atomic and refuses to replace an
//! existing output.
//!
//! This is deliberately a quantized-weight sidecar, not a complete checkpoint:
//! policy-excluded embeddings, norms, biases, recurrent scalars, and MTP tensors
//! remain in their source checkpoint. A future full-checkpoint container can
//! compose those source-precision tensors with this independently versioned
//! packed layout.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};

use burn::module::ParamId;
use burn::tensor::{DType, Shape, TensorData};
use burn_store::{BurnpackStore, BurnpackWriter, ModuleStore, TensorSnapshot, TensorSnapshotError};

use super::{
    BurnpackProvenance, BurnpackQuantizationManifest, PackedExecutionSupport,
    QuantizationEligibility, QuantizationError, QuantizationExclusion, QuantizedTensorMetadata,
    QuantizedWeight, Result, SourceFloatDType, WeightQuantizationConfig,
    classify_weight_for_quantization,
    provenance::{DigestingWriter, packed_payload_sha256},
    quantize_tensor_data,
    staging::{
        BoundBreach, LimitViolation, TensorBreach, admit_tensor, bounded_add, first_limit_violation,
    },
};

/// Reserved Burnpack U8 tensor containing the quantization manifest JSON.
///
/// The `_v1` suffix versions this container slot, not the manifest format:
/// both v1 and v2 manifests are stored here, and the JSON's own
/// `format_version` field selects the schema. Renaming the slot would make
/// every existing sidecar unreadable.
pub const BURNPACK_MANIFEST_TENSOR: &str = "__transformers_quantization_manifest_v1__";

/// Burn Store's maximum tensor count, including the reserved manifest tensor.
const UPSTREAM_MAX_TENSORS: usize = 100_000;
/// Bounds staging below the address-space limit of a 32-bit host, in bytes.
#[cfg(target_pointer_width = "32")]
const DEFAULT_MAX_STAGED_BYTES: u64 = 3 * 1024 * 1024 * 1024;
/// Leaves serialized-header headroom below Burn Store's 100 GiB file limit.
#[cfg(not(target_pointer_width = "32"))]
const DEFAULT_MAX_STAGED_BYTES: u64 = 90 * 1024 * 1024 * 1024;
/// Explains the staging restriction on 32-bit hosts.
#[cfg(target_pointer_width = "32")]
const MAX_STAGED_REASON: &str = "must be at most 3 GiB on a 32-bit host";
/// Explains the staging restriction imposed by the upstream artifact format.
#[cfg(not(target_pointer_width = "32"))]
const MAX_STAGED_REASON: &str =
    "must be at most 90 GiB to leave room below Burnpack's 100 GiB file limit";
/// Maximum in-memory JSON manifest bytes accepted by default.
const DEFAULT_MAX_MANIFEST_BYTES: usize = 64 * 1024 * 1024;
/// Maximum UTF-8 bytes in one model tensor name.
const DEFAULT_MAX_TENSOR_NAME_BYTES: usize = 16 * 1024;
/// Session limit names, in validation order.
const SESSION_LIMIT_FIELDS: [&str; 5] = [
    "max_tensors",
    "max_staged_bytes",
    "max_tensor_payload_bytes",
    "max_manifest_bytes",
    "max_tensor_name_bytes",
];
/// Largest admissible value of each session limit, in validation order.
///
/// `max_tensors` stays below Burn Store's tensor count so one slot remains for
/// the manifest tensor.
const SESSION_LIMIT_CAPS: [u64; 5] = [
    UPSTREAM_MAX_TENSORS as u64 - 1,
    DEFAULT_MAX_STAGED_BYTES,
    UPSTREAM_MAX_TENSOR_BYTES,
    DEFAULT_MAX_MANIFEST_BYTES as u64,
    DEFAULT_MAX_TENSOR_NAME_BYTES as u64,
];
/// Why each session limit above its cap is refused, in validation order.
const SESSION_LIMIT_REASONS: [&str; 5] = [
    "must leave one of Burn Store's 100,000 tensor slots for the manifest",
    MAX_STAGED_REASON,
    "exceeds Burn Store's platform-specific per-tensor reader limit",
    "must be at most 64 MiB",
    "must be at most 16 KiB",
];
/// Position of `max_manifest_bytes` in the session limit tables.
const MANIFEST_LIMIT: usize = 3;
/// Maximum individually materialized payload bytes on a 32-bit host.
#[cfg(target_pointer_width = "32")]
const UPSTREAM_MAX_TENSOR_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Burn Store's maximum individually materialized tensor payload bytes.
#[cfg(not(target_pointer_width = "32"))]
const UPSTREAM_MAX_TENSOR_BYTES: u64 = 10 * 1024 * 1024 * 1024;
/// Conservative bytes for the pretty-printed manifest envelope.
///
/// The envelope (`format_version`, `byte_order`, `packed_execution`, and the
/// `tensors` braces) serializes to about 120 bytes; provenance is estimated
/// separately when it is bound.
const MANIFEST_BASE_ESTIMATE: u64 = 512;
/// Pretty-printing indentation added to each line of an entry nested in `tensors`.
const MANIFEST_ENTRY_INDENT: u64 = 4;
/// Key indentation, `": "` separator, and trailing `",\n"` around one entry.
const MANIFEST_ENTRY_FRAMING: u64 = 8;
/// The v2 `"payload_sha256": "<64 hex digits>"` line bound later, with framing.
const MANIFEST_DIGEST_LINE: u64 = 96;
/// Fixed estimated bytes for checkpoint provenance metadata.
const MANIFEST_PROVENANCE_BASE_OVERHEAD: u64 = 1024;
/// Estimated fixed bytes for each provenance shard record.
const MANIFEST_PROVENANCE_SHARD_OVERHEAD: u64 = 512;
/// Worst-case JSON escaping expands one source byte to six ASCII bytes.
const JSON_STRING_ESCAPE_FACTOR: u64 = 6;

/// Process-local suffix counter; exclusive file creation resolves cross-process collisions.
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Resource limits for an incremental Burnpack conversion session.
///
/// Defaults stay within Burn Store 0.21's public reader bounds: 100,000 total
/// tensors, 10 GiB per tensor on 64-bit hosts (2 GiB on 32-bit hosts), and a
/// 100 GiB artifact. The session reserves one tensor slot for its manifest and
/// extra artifact headroom for Burnpack metadata and 256-byte alignment. On a
/// 32-bit host, total staging defaults to 3 GiB so offsets remain representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BurnpackSessionLimits {
    /// Maximum number of quantized model tensors, excluding the reserved manifest.
    pub max_tensors: usize,
    /// Maximum sum of model payload bytes staged on disk.
    pub max_staged_bytes: u64,
    /// Maximum packed-plus-scale bytes for one model tensor.
    pub max_tensor_payload_bytes: u64,
    /// Maximum serialized JSON bytes in the reserved manifest tensor.
    pub max_manifest_bytes: usize,
    /// Maximum UTF-8 bytes in one architecture-owned tensor name.
    pub max_tensor_name_bytes: usize,
}

impl BurnpackSessionLimits {
    /// Validates limits against Burn Store 0.21's documented reader bounds.
    ///
    /// Each limit must be positive and at most its cap; the verified
    /// [`first_limit_violation`] names the first field that is not, checking
    /// fields in declaration order and each for zero before its cap.
    pub fn validate(self) -> Result<Self> {
        validate_session_limits(
            0,
            &[
                self.max_tensors as u64,
                self.max_staged_bytes,
                self.max_tensor_payload_bytes,
                self.max_manifest_bytes as u64,
                self.max_tensor_name_bytes as u64,
            ],
        )?;
        Ok(self)
    }
}

impl Default for BurnpackSessionLimits {
    /// Sets finite staging and manifest bounds suitable for incremental weight conversion.
    fn default() -> Self {
        Self {
            max_tensors: UPSTREAM_MAX_TENSORS - 1,
            max_staged_bytes: DEFAULT_MAX_STAGED_BYTES,
            max_tensor_payload_bytes: UPSTREAM_MAX_TENSOR_BYTES,
            max_manifest_bytes: DEFAULT_MAX_MANIFEST_BYTES,
            max_tensor_name_bytes: DEFAULT_MAX_TENSOR_NAME_BYTES,
        }
    }
}

/// Current bounded-resource accounting for a write session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BurnpackStagingStats {
    /// Number of quantized model tensors staged so far.
    pub tensor_count: usize,
    /// Sum of their packed-value and scale payload bytes.
    pub staged_payload_bytes: u64,
    /// Largest individual staged model payload.
    pub largest_payload_bytes: u64,
    /// Conservative upper estimate used to bound manifest serialization.
    pub estimated_manifest_bytes: u64,
}

/// Metadata and staging range for one successfully appended tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedTensorSummary {
    /// Architecture-owned model parameter name.
    pub name: String,
    /// Byte offset within the private staging file.
    pub staging_offset: u64,
    /// Exact packed-value-plus-scale byte count.
    pub payload_bytes: u64,
    /// Logical and physical quantization metadata.
    pub metadata: QuantizedTensorMetadata,
}

/// Result of applying eligibility and staging one source tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BurnpackAppendOutcome {
    /// Eligible matrix weight was quantized and appended to disk staging.
    Staged(
        /// Appended payload identity, quantization policy, and bounded resource use.
        StagedTensorSummary,
    ),
    /// Policy left the tensor in its separate source checkpoint.
    Excluded(
        /// Policy reason the source tensor must remain in its original precision.
        QuantizationExclusion,
    ),
}

/// Summary returned after atomically publishing an official Burnpack artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnpackArtifactSummary {
    /// Final `.bpk` path.
    pub path: PathBuf,
    /// Number of quantized model tensors, excluding the manifest tensor.
    pub tensor_count: usize,
    /// Sum of model payload bytes before Burnpack alignment.
    pub quantized_payload_bytes: u64,
    /// UTF-8 JSON byte count in the reserved manifest tensor.
    pub manifest_bytes: usize,
    /// Complete official Burnpack file size.
    pub artifact_bytes: u64,
    /// Largest payload materialized by a lazy snapshot during finalization.
    pub largest_materialized_tensor_bytes: u64,
    /// Whether every packed matrix maps exactly to Burn block-QFloat.
    ///
    /// [`PackedExecutionSupport::Unavailable`] artifacts are valid reference
    /// storage but every packed loader rejects them; see
    /// [`super::UnalignedMatrixPolicy`] to keep unaligned matrices unpacked.
    pub packed_execution: PackedExecutionSupport,
    /// Whether the platform accepted an output-directory durability sync.
    ///
    /// Publication remains atomic and successful when this is false, but a
    /// sudden power loss may lose the new directory entry on that filesystem.
    pub parent_directory_synchronized: bool,
}

/// Report produced by materializing and validating every quantized tensor once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BurnpackValidationReport {
    /// Number of model tensor payloads validated.
    pub tensor_count: usize,
    /// Sum of validated packed-value and scale bytes.
    pub payload_bytes: u64,
}

/// Detailed validation accounting used by audited packed-load entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct BurnpackValidationAudit {
    /// Payloads copied completely from mapped storage, including a failing payload.
    pub tensor_reads: u64,
    /// Payloads whose digest, structure, scales, and codes all validated.
    pub tensors_validated: usize,
    /// Sum of payload bytes copied, including a failing payload when fully copied.
    pub payload_bytes: u64,
    /// Largest individual payload copied completely from mapped storage.
    pub largest_payload_bytes: u64,
    /// Maximum simultaneous Rust-owned bytes used to decode one payload.
    pub peak_owned_payload_bytes: u64,
}

/// Partial sequential-validation audit retained when one payload is invalid.
#[derive(Debug)]
pub(super) struct BurnpackValidationFailure {
    /// Root payload, allocation, or semantic validation error.
    pub error: QuantizationError,
    /// Exact progress through and including any completely copied failing payload.
    pub audit: BurnpackValidationAudit,
}

/// Owned-buffer accounting for one packed payload read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct BurnpackPayloadReadStats {
    /// Whether the complete mapped payload reached owned memory.
    pub tensor_reads: u64,
    /// Complete mapped payload bytes copied into owned memory.
    pub payload_bytes: u64,
    /// Maximum simultaneous capacity of owned payload and scale vectors.
    pub peak_owned_payload_bytes: u64,
}

/// Failed packed read paired with allocation and byte progress through the error.
pub(super) struct BurnpackPayloadReadFailure {
    /// Root payload, allocation, digest, or semantic validation error.
    pub error: QuantizationError,
    /// Owned-buffer accounting completed before the error.
    pub stats: BurnpackPayloadReadStats,
}

/// Immutable byte-range descriptor for one completed tensor in the staging file.
#[derive(Debug, Clone)]
struct StagedTensor {
    /// Byte offset from the beginning of the staging file.
    offset: u64,
    /// Number of packed-code and scale bytes owned by this tensor.
    payload_bytes: u64,
    /// Hexadecimal SHA-256 of the exact staged payload bytes.
    payload_sha256: String,
}

/// Incremental, one-source-tensor-at-a-time quantized Burnpack writer.
///
/// Dropping an unfinished or completed session removes its private staging file.
/// The final output is never visible until Burn Store has written and synchronized
/// a complete temporary Burnpack file.
pub struct QuantizedBurnpackSession {
    /// Canonical parent and caller-preserved final artifact filename.
    output_path: PathBuf,
    /// Private adjacent payload spool, removed when the session closes.
    staging_path: PathBuf,
    /// Open spool handle, absent after staging has been closed.
    staging_file: Option<File>,
    /// Completed tensor byte ranges keyed by their unique parameter names.
    entries: BTreeMap<String, StagedTensor>,
    /// Logical shapes, quantization contracts, and optional source provenance.
    manifest: BurnpackQuantizationManifest,
    /// Validated upper bounds checked before allocation and append operations.
    limits: BurnpackSessionLimits,
    /// Cumulative I/O and high-water allocation diagnostics.
    stats: BurnpackStagingStats,
    /// Prevents further appends or publication once the output was finalized.
    finalized: bool,
}

impl QuantizedBurnpackSession {
    /// Creates a session with conservative defaults in the output's directory.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        Self::with_limits(path, BurnpackSessionLimits::default())
    }

    /// Creates a session with caller-specified validated resource limits.
    pub fn with_limits(path: impl AsRef<Path>, limits: BurnpackSessionLimits) -> Result<Self> {
        let output_path = validate_bpk_path(path.as_ref())?;
        let limits = limits.validate()?;
        if output_path.exists() {
            return Err(QuantizationError::OutputAlreadyExists { path: output_path });
        }
        let (staging_path, staging_file) = create_unique_file(&output_path, "quant-stage")?;
        #[cfg(unix)]
        fs::remove_file(&staging_path).map_err(|source| QuantizationError::ArtifactIo {
            operation: "unlink private quantization staging file",
            path: staging_path.clone(),
            source,
        })?;
        Ok(Self {
            output_path,
            staging_path,
            staging_file: Some(staging_file),
            entries: BTreeMap::new(),
            manifest: BurnpackQuantizationManifest::new_legacy_v1(),
            limits,
            stats: BurnpackStagingStats {
                tensor_count: 0,
                staged_payload_bytes: 0,
                largest_payload_bytes: 0,
                estimated_manifest_bytes: MANIFEST_BASE_ESTIMATE,
            },
            finalized: false,
        })
    }

    /// Returns the exact destination that will be atomically published.
    #[must_use]
    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    /// Returns the validated manifest accumulated so far.
    #[must_use]
    pub const fn manifest(&self) -> &BurnpackQuantizationManifest {
        &self.manifest
    }

    /// Returns current disk-staging and bounded-memory accounting.
    #[must_use]
    pub const fn stats(&self) -> BurnpackStagingStats {
        self.stats
    }

    /// Binds the in-progress artifact to exact checkpoint bytes, payloads, and a load plan.
    ///
    /// Low-level callers that do not own a checkpoint may omit this operation and
    /// continue producing readable v1 artifacts. Architecture-aware conversion
    /// always calls it after proving the source remained stable across execution.
    pub fn bind_provenance(&mut self, provenance: BurnpackProvenance) -> Result<()> {
        self.ensure_mutable()?;
        if self.manifest.provenance.is_some() {
            return Err(QuantizationError::ProvenanceAlreadyBound);
        }
        let bound_estimate = bounded_add(
            self.stats.estimated_manifest_bytes,
            estimated_provenance_bytes(&provenance)?,
            self.limits.max_manifest_bytes as u64,
        )
        .map_err(|breach| {
            limit_breach(
                breach,
                "bound manifest size estimate",
                "estimated manifest",
                self.limits.max_manifest_bytes as u64,
            )
        })?;

        // Assemble and validate a complete candidate before changing live state.
        // This keeps a manifest-size or digest error retryable as a v1 session.
        let mut candidate = self.manifest.clone();
        let manifest_tensor_count = candidate.tensors.len() as u64;
        for (name, entry) in &self.entries {
            let metadata =
                candidate
                    .tensors
                    .get_mut(name)
                    .ok_or(QuantizationError::StagingCorrupted {
                        expected: self.entries.len() as u64,
                        actual: manifest_tensor_count,
                    })?;
            metadata.payload_sha256 = Some(entry.payload_sha256.clone());
        }
        candidate.packed_execution = packed_execution_support(&candidate);
        candidate.bind_provenance(provenance)?;

        let actual_manifest_bytes = candidate.to_json_pretty()?.len();
        self.check_manifest_size(actual_manifest_bytes)?;
        let actual_manifest_bytes_u64 = u64::try_from(actual_manifest_bytes).map_err(|_| {
            QuantizationError::ByteLengthOverflow {
                context: "bound manifest byte length conversion",
            }
        })?;
        bounded_add(
            self.stats.staged_payload_bytes,
            actual_manifest_bytes_u64,
            self.limits.max_staged_bytes,
        )
        .map_err(|breach| {
            limit_breach(
                breach,
                "bound staging length with manifest",
                "staged payload",
                self.limits.max_staged_bytes,
            )
        })?;

        self.manifest = candidate;
        self.stats.estimated_manifest_bytes = self
            .stats
            .estimated_manifest_bytes
            .max(bound_estimate)
            .max(actual_manifest_bytes_u64);
        Ok(())
    }

    /// Applies policy, quantizes one source TensorData, and releases it before the next call.
    pub fn append(
        &mut self,
        name: &str,
        source: &TensorData,
        config: WeightQuantizationConfig,
    ) -> Result<BurnpackAppendOutcome> {
        self.ensure_mutable()?;
        self.ensure_provenance_unbound()?;
        self.validate_name(name)?;
        match classify_weight_for_quantization(name, source.shape.as_slice()) {
            QuantizationEligibility::Eligible => {
                let source_dtype = SourceFloatDType::try_from(source.dtype)?;
                let metadata =
                    QuantizedTensorMetadata::new(source.shape.as_slice(), source_dtype, config)?;
                let payload_bytes = metadata
                    .packed_value_bytes
                    .checked_add(metadata.scale_bytes)
                    .ok_or(QuantizationError::ByteLengthOverflow {
                        context: "preflight quantized tensor payload length",
                    })?;
                self.preflight_append(name, &metadata, payload_bytes)?;
                let weight = quantize_tensor_data(source, config)?;
                self.append_quantized_inner(name, weight)
                    .map(BurnpackAppendOutcome::Staged)
            }
            QuantizationEligibility::Excluded(reason) => {
                Ok(BurnpackAppendOutcome::Excluded(reason))
            }
        }
    }

    /// Checks a complete planned inventory against this session's limits.
    ///
    /// `planned` lists the `(name, metadata)` of every tensor the caller intends
    /// to append, and `provenance` the provenance it intends to bind (digests may
    /// be placeholders; only lengths matter). The tensor count, each payload,
    /// the staged total, and the manifest estimate, including the per-entry
    /// digest lines and provenance, are summed exactly as the incremental checks
    /// would, so a conversion that cannot fit fails before reading any source.
    /// The session itself is not modified.
    pub(crate) fn preflight_inventory<'a>(
        &self,
        planned: impl IntoIterator<Item = (&'a str, &'a QuantizedTensorMetadata)>,
        provenance: &BurnpackProvenance,
    ) -> Result<()> {
        let mut projected = self.stats;
        for (name, metadata) in planned {
            let payload_bytes = metadata
                .packed_value_bytes
                .checked_add(metadata.scale_bytes)
                .ok_or(QuantizationError::ByteLengthOverflow {
                    context: "preflight quantized tensor payload length",
                })?;
            let probe = Self::projection(self, projected);
            let (staged, estimate) = probe.preflight_append(name, metadata, payload_bytes)?;
            projected.tensor_count += 1;
            projected.staged_payload_bytes = staged;
            projected.estimated_manifest_bytes = estimate;
        }
        bounded_add(
            projected.estimated_manifest_bytes,
            estimated_provenance_bytes(provenance)?,
            self.limits.max_manifest_bytes as u64,
        )
        .map_err(|breach| {
            limit_breach(
                breach,
                "bound manifest size estimate",
                "estimated manifest",
                self.limits.max_manifest_bytes as u64,
            )
        })?;
        Ok(())
    }

    /// Borrows this session's limits with hypothetical staging counters.
    fn projection(&self, stats: BurnpackStagingStats) -> LimitProjection<'_> {
        LimitProjection {
            limits: &self.limits,
            stats,
        }
    }

    /// Appends an already validated reference-quantized tensor to disk staging.
    ///
    /// This consumes and drops the packed buffers after their bytes have been
    /// appended; only their range and cloned metadata remain in the session.
    pub fn append_quantized(
        &mut self,
        name: &str,
        weight: QuantizedWeight,
    ) -> Result<StagedTensorSummary> {
        self.ensure_mutable()?;
        self.ensure_provenance_unbound()?;
        self.validate_name(name)?;
        self.append_quantized_inner(name, weight)
    }

    /// Finalizes through [`BurnpackWriter`] and atomically publishes the `.bpk` file.
    ///
    /// Publication is create-only. A path that appears after session creation is
    /// never replaced, and a failed Burn Store write leaves no partial final path.
    pub fn finalize(&mut self) -> Result<BurnpackArtifactSummary> {
        self.ensure_mutable()?;
        if self.output_path.exists() {
            return Err(QuantizationError::OutputAlreadyExists {
                path: self.output_path.clone(),
            });
        }

        self.manifest.packed_execution = packed_execution_support(&self.manifest);

        let manifest_bytes = self.manifest.to_json_pretty()?.into_bytes();
        self.check_manifest_size(manifest_bytes.len())?;
        let manifest_len = u64::try_from(manifest_bytes.len()).map_err(|_| {
            QuantizationError::ByteLengthOverflow {
                context: "manifest byte length conversion",
            }
        })?;
        bounded_add(
            self.stats.staged_payload_bytes,
            manifest_len,
            self.limits.max_staged_bytes,
        )
        .map_err(|breach| {
            limit_breach(
                breach,
                "staging length with manifest",
                "staged payload",
                self.limits.max_staged_bytes,
            )
        })?;

        let manifest_offset = self.checked_staging_end()?;
        if let Err(error) = self.write_raw_staging(&manifest_bytes) {
            return self.rollback_after(manifest_offset, error);
        }
        if let Err(error) = self.sync_staging() {
            return self.rollback_after(manifest_offset, error);
        }

        let result = self.write_and_publish(manifest_offset, manifest_bytes.len());
        match result {
            Ok(mut summary) => {
                self.finalized = true;
                summary.largest_materialized_tensor_bytes =
                    summary.largest_materialized_tensor_bytes.max(manifest_len);
                self.close_staging();
                Ok(summary)
            }
            Err(error) => self.rollback_after(manifest_offset, error),
        }
    }

    // Staging transactions: validate, append, then commit metadata and counters.

    /// Stages one validated packed payload and commits its metadata only after exact writes.
    ///
    /// The payload digest covers packed codes followed by little-endian F32 scales.
    /// A write or validation failure truncates back to the previous committed offset;
    /// manifest entries and counters advance only after staging succeeds.
    fn append_quantized_inner(
        &mut self,
        name: &str,
        weight: QuantizedWeight,
    ) -> Result<StagedTensorSummary> {
        let mut metadata = weight.metadata().clone();
        metadata.validate()?;
        // A weight read from a v2 artifact carries its payload digest. The
        // session stages metadata in the v1 form and attaches its own staged
        // digests only when provenance is bound, so a carried digest is checked
        // against the staged bytes below and then removed; otherwise a v1
        // session would fail at `finalize` after every payload was staged.
        let carried_digest = metadata.payload_sha256.take();
        let payload_bytes = metadata
            .packed_value_bytes
            .checked_add(metadata.scale_bytes)
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "quantized tensor payload length",
            })?;
        let (next_staged, next_estimate) = self.preflight_append(name, &metadata, payload_bytes)?;
        let offset = self.checked_staging_end()?;

        // Hash the staged bytes in the same pass that writes them.
        let (write_result, payload_sha256) = {
            let mut writer = DigestingWriter::new(self.staging_file_mut()?);
            let result = weight.write_burnpack_payload(&mut writer);
            (result, writer.finish())
        };
        if let Err(error) = write_result {
            return self.rollback_after(offset, error);
        }
        if let Some(expected) = carried_digest
            && expected != payload_sha256
        {
            return self.rollback_after(
                offset,
                QuantizationError::PackedPayloadDigestMismatch {
                    name: name.to_owned(),
                    expected,
                    actual: payload_sha256,
                },
            );
        }
        let end = match self.staging_end() {
            Ok(end) => end,
            Err(error) => return self.rollback_after(offset, error),
        };
        let observed = match end.checked_sub(offset) {
            Some(observed) => observed,
            None => {
                return self.rollback_after(
                    offset,
                    QuantizationError::StagingCorrupted {
                        expected: offset,
                        actual: end,
                    },
                );
            }
        };
        if observed != payload_bytes {
            return self.rollback_after(
                offset,
                QuantizationError::StagingCorrupted {
                    expected: payload_bytes,
                    actual: observed,
                },
            );
        }

        if let Err(error) = self.manifest.insert(name, metadata.clone()) {
            return self.rollback_after(offset, error);
        }
        self.entries.insert(
            name.to_owned(),
            StagedTensor {
                offset,
                payload_bytes,
                payload_sha256,
            },
        );
        self.stats.tensor_count += 1;
        self.stats.staged_payload_bytes = next_staged;
        self.stats.largest_payload_bytes = self.stats.largest_payload_bytes.max(payload_bytes);
        self.stats.estimated_manifest_bytes = next_estimate;

        Ok(StagedTensorSummary {
            name: name.to_owned(),
            staging_offset: offset,
            payload_bytes,
            metadata,
        })
    }

    /// Checks prospective count, payload, staging, and manifest bounds before writing.
    fn preflight_append(
        &self,
        name: &str,
        metadata: &QuantizedTensorMetadata,
        payload_bytes: u64,
    ) -> Result<(u64, u64)> {
        self.projection(self.stats)
            .preflight_append(name, metadata, payload_bytes)
    }

    /// Rejects empty, reserved, overlong, or already staged tensor names.
    fn validate_name(&self, name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(QuantizationError::EmptyTensorName);
        }
        if name == BURNPACK_MANIFEST_TENSOR {
            return Err(QuantizationError::ReservedTensorCollision {
                name: name.to_owned(),
            });
        }
        if name.len() > self.limits.max_tensor_name_bytes {
            return Err(QuantizationError::SessionLimitExceeded {
                kind: "tensor name",
                attempted: name.len() as u64,
                limit: self.limits.max_tensor_name_bytes as u64,
            });
        }
        if self.entries.contains_key(name) {
            return Err(QuantizationError::DuplicateManifestTensor {
                name: name.to_owned(),
            });
        }
        Ok(())
    }

    /// Checks actual serialized manifest bytes against the configured hard bound.
    fn check_manifest_size(&self, actual: usize) -> Result<()> {
        if actual > self.limits.max_manifest_bytes {
            return Err(QuantizationError::SessionLimitExceeded {
                kind: "manifest",
                attempted: actual as u64,
                limit: self.limits.max_manifest_bytes as u64,
            });
        }
        Ok(())
    }

    /// Rejects changes after an output has been successfully published.
    fn ensure_mutable(&self) -> Result<()> {
        if self.finalized {
            Err(QuantizationError::SessionFinalized)
        } else {
            Ok(())
        }
    }

    /// Prevents late staging from invalidating an already bound v2 manifest.
    fn ensure_provenance_unbound(&self) -> Result<()> {
        if self.manifest.provenance.is_some() {
            Err(QuantizationError::ProvenanceAlreadyBound)
        } else {
            Ok(())
        }
    }

    /// Borrows the live staging handle or reports its absence as staging corruption.
    fn staging_file_mut(&mut self) -> Result<&mut File> {
        self.staging_file
            .as_mut()
            .ok_or(QuantizationError::StagingCorrupted {
                expected: self.stats.staged_payload_bytes,
                actual: 0,
            })
    }

    /// Requires the staging file length to match the last committed payload boundary.
    fn checked_staging_end(&mut self) -> Result<u64> {
        let expected = self.stats.staged_payload_bytes;
        let actual = self.staging_end()?;
        if actual != expected {
            return Err(QuantizationError::StagingCorrupted { expected, actual });
        }
        Ok(actual)
    }

    /// Seeks to the physical end of staging and returns its checked I/O result.
    fn staging_end(&mut self) -> Result<u64> {
        let path = self.staging_path.clone();
        self.staging_file_mut()?
            .seek(SeekFrom::End(0))
            .map_err(|source| QuantizationError::ArtifactIo {
                operation: "seek quantization staging file",
                path,
                source,
            })
    }

    /// Writes all supplied bytes to staging; the caller owns rollback on failure.
    fn write_raw_staging(&mut self, bytes: &[u8]) -> Result<()> {
        let path = self.staging_path.clone();
        self.staging_file_mut()?
            .write_all(bytes)
            .map_err(|source| QuantizationError::ArtifactIo {
                operation: "append quantization staging file",
                path,
                source,
            })
    }

    /// Flushes staged bytes and requests data durability before final materialization.
    fn sync_staging(&mut self) -> Result<()> {
        let path = self.staging_path.clone();
        let file = self.staging_file_mut()?;
        file.flush()
            .and_then(|()| file.sync_data())
            .map_err(|source| QuantizationError::ArtifactIo {
                operation: "synchronize quantization staging file",
                path,
                source,
            })
    }

    /// Truncates and rewinds to the last safe boundary after a failed append.
    ///
    /// If rollback itself fails, that failure takes precedence because the staging
    /// state can no longer be assumed reusable.
    fn rollback_after<T>(&mut self, offset: u64, error: QuantizationError) -> Result<T> {
        let path = self.staging_path.clone();
        let rollback = self.staging_file_mut().and_then(|file| {
            file.set_len(offset)
                .and_then(|()| file.seek(SeekFrom::Start(offset)).map(|_| ()))
                .map_err(|source| QuantizationError::ArtifactIo {
                    operation: "roll back quantization staging file",
                    path,
                    source,
                })
        });
        match rollback {
            Ok(()) => Err(error),
            Err(rollback_error) => Err(rollback_error),
        }
    }

    // Publication separates atomic visibility from best-effort directory durability.

    /// Writes, reopens, and validates a temporary Burnpack before create-only publication.
    ///
    /// The hard link is the atomic visibility boundary and cannot replace existing
    /// output. Directory synchronization is reported separately after publication,
    /// because a durability failure cannot undo an already visible successful link.
    fn write_and_publish(
        &self,
        manifest_offset: u64,
        manifest_bytes: usize,
    ) -> Result<BurnpackArtifactSummary> {
        let snapshots = self.snapshots(manifest_offset, manifest_bytes)?;
        let writer = BurnpackWriter::new(snapshots)
            .with_metadata("format", "burnpack")
            .with_metadata("producer", "transformers")
            .with_metadata("quantization_manifest_tensor", BURNPACK_MANIFEST_TENSOR);
        let temporary_directory = create_unique_directory(&self.output_path, "partial")?;
        let temporary_path = temporary_directory.join("artifact.bpk");
        let guard = TemporaryOutput::new(temporary_directory, temporary_path.clone());

        writer
            .write_to_file(&temporary_path)
            .map_err(|error| QuantizationError::BurnStore {
                operation: "write quantized Burnpack artifact",
                message: error.to_string(),
            })?;
        let artifact_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&temporary_path)
            .map_err(|source| QuantizationError::ArtifactIo {
                operation: "open completed temporary Burnpack artifact",
                path: temporary_path.clone(),
                source,
            })?;
        artifact_file
            .sync_all()
            .map_err(|source| QuantizationError::ArtifactIo {
                operation: "synchronize completed temporary Burnpack artifact",
                path: temporary_path.clone(),
                source,
            })?;
        let artifact_bytes = artifact_file
            .metadata()
            .map_err(|source| QuantizationError::ArtifactIo {
                operation: "inspect completed temporary Burnpack artifact",
                path: temporary_path.clone(),
                source,
            })?
            .len();
        drop(artifact_file);

        // Re-open through the production reader and validate every payload before
        // publication. This binds the bytes actually emitted by Burn Store—not
        // merely the in-memory values that were intended for staging—to the v2
        // manifest digests. Validation retains at most one payload at a time.
        let mut completed = QuantizedBurnpackArtifact::open(&temporary_path)?;
        completed.validate_all()?;
        drop(completed);

        fs::hard_link(&temporary_path, &self.output_path).map_err(|source| {
            if source.kind() == std::io::ErrorKind::AlreadyExists {
                QuantizationError::OutputAlreadyExists {
                    path: self.output_path.clone(),
                }
            } else {
                QuantizationError::ArtifactIo {
                    operation: "atomically publish completed Burnpack artifact",
                    path: self.output_path.clone(),
                    source,
                }
            }
        })?;
        // A failure after the hard link exists cannot truthfully be returned as
        // an unpublished transaction. Report durability in the success summary.
        let parent_directory_synchronized = sync_parent_directory(&self.output_path).is_ok();
        drop(guard);

        Ok(BurnpackArtifactSummary {
            path: self.output_path.clone(),
            tensor_count: self.stats.tensor_count,
            quantized_payload_bytes: self.stats.staged_payload_bytes,
            manifest_bytes,
            artifact_bytes,
            largest_materialized_tensor_bytes: self.stats.largest_payload_bytes,
            packed_execution: self.manifest.packed_execution,
            parent_directory_synchronized,
        })
    }

    /// Builds lazy U8 snapshots over staging offsets, reserving parameter zero for metadata.
    ///
    /// Snapshot closures share a private file handle and materialize one payload at a
    /// time; collecting descriptors therefore does not collect all model weights.
    fn snapshots(
        &self,
        manifest_offset: u64,
        manifest_bytes: usize,
    ) -> Result<Vec<TensorSnapshot>> {
        let snapshot_count =
            self.entries
                .len()
                .checked_add(1)
                .ok_or(QuantizationError::ByteLengthOverflow {
                    context: "Burnpack snapshot count",
                })?;
        let mut snapshots = Vec::new();
        snapshots.try_reserve_exact(snapshot_count).map_err(|_| {
            QuantizationError::AllocationFailed {
                context: "Burnpack lazy snapshots",
                requested: snapshot_count,
            }
        })?;

        let staging_file = self
            .staging_file
            .as_ref()
            .ok_or(QuantizationError::StagingCorrupted {
                expected: self.stats.staged_payload_bytes,
                actual: 0,
            })?
            .try_clone()
            .map_err(|source| QuantizationError::ArtifactIo {
                operation: "clone private quantization staging handle",
                path: self.staging_path.clone(),
                source,
            })?;
        let staging_file = Rc::new(staging_file);

        for (index, (name, entry)) in self.entries.iter().enumerate() {
            let parameter_id =
                u64::try_from(index + 1).map_err(|_| QuantizationError::ByteLengthOverflow {
                    context: "Burnpack tensor parameter id",
                })?;
            snapshots.push(staged_snapshot(
                Rc::clone(&staging_file),
                &self.staging_path,
                name,
                entry.offset,
                entry.payload_bytes,
                ParamId::from(parameter_id),
            )?);
        }
        snapshots.push(staged_snapshot(
            staging_file,
            &self.staging_path,
            BURNPACK_MANIFEST_TENSOR,
            manifest_offset,
            manifest_bytes as u64,
            ParamId::from(0),
        )?);
        Ok(snapshots)
    }

    /// Drops the staging handle and attempts to remove its private file without panicking.
    fn close_staging(&mut self) {
        if let Some(file) = self.staging_file.take() {
            drop(file);
        }
        let _ = fs::remove_file(&self.staging_path);
    }
}

impl Drop for QuantizedBurnpackSession {
    /// Performs best-effort staging cleanup when the session leaves scope.
    fn drop(&mut self) {
        self.close_staging();
    }
}

/// Memory-mapped, lazily materialized view of a quantized Burnpack artifact.
pub struct QuantizedBurnpackArtifact {
    /// Validated local artifact path used in contextual errors.
    path: PathBuf,
    /// Upstream memory-mapped store that owns lazy payload access.
    store: BurnpackStore,
    /// Validated decoded manifest describing the mapped payloads.
    manifest: BurnpackQuantizationManifest,
}

impl QuantizedBurnpackArtifact {
    /// Memory-maps an official Burnpack file and validates its manifest/payload contract.
    ///
    /// Opening first materializes the reserved manifest, then checks each model
    /// payload's mapped U8 range sequentially without decoding or retaining it.
    /// [`Self::validate_all`] additionally validates packed codes and scales.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_manifest_limit(path, DEFAULT_MAX_MANIFEST_BYTES)
    }

    /// Opens with a caller-selected upper bound for manifest tensor bytes.
    pub fn open_with_manifest_limit(
        path: impl AsRef<Path>,
        max_manifest_bytes: usize,
    ) -> Result<Self> {
        validate_session_limits(MANIFEST_LIMIT, &[max_manifest_bytes as u64])?;
        let path = validate_bpk_path(path.as_ref())?;
        let mut store = BurnpackStore::from_file(&path)
            .auto_extension(false)
            .zero_copy(true);
        let mut keys = store.keys().map_err(|error| QuantizationError::BurnStore {
            operation: "memory-map and enumerate Burnpack tensors",
            message: error.to_string(),
        })?;
        keys.sort();

        let manifest_len = validate_manifest_descriptor(&mut store, max_manifest_bytes)?;
        let manifest_data = materialize_store_tensor(&mut store, BURNPACK_MANIFEST_TENSOR)?.ok_or(
            QuantizationError::MissingManifestTensor {
                name: BURNPACK_MANIFEST_TENSOR,
            },
        )?;
        validate_u8_payload_descriptor(BURNPACK_MANIFEST_TENSOR, &manifest_data, manifest_len)?;
        let manifest_json = std::str::from_utf8(manifest_data.as_bytes())?;
        let manifest = BurnpackQuantizationManifest::from_json(manifest_json)?;
        if manifest.tensors.contains_key(BURNPACK_MANIFEST_TENSOR) {
            return Err(QuantizationError::ReservedTensorCollision {
                name: BURNPACK_MANIFEST_TENSOR.to_owned(),
            });
        }

        validate_key_sets(&keys, &manifest)?;
        validate_store_descriptors(&mut store, &manifest)?;
        Ok(Self {
            path,
            store,
            manifest,
        })
    }

    /// Returns the mapped artifact path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the validated logical quantization manifest.
    #[must_use]
    pub const fn manifest(&self) -> &BurnpackQuantizationManifest {
        &self.manifest
    }

    /// Lazily returns one U8 payload backed by Burn Store's mapped file storage.
    ///
    /// The returned `TensorData` owns a reference-counted slice of the mapping;
    /// no packed payload copy is made by this method.
    pub fn tensor_payload(&mut self, name: &str) -> Result<TensorData> {
        let metadata = self.manifest.tensors.get(name).ok_or_else(|| {
            QuantizationError::QuantizedTensorNotFound {
                name: name.to_owned(),
            }
        })?;
        let expected = payload_len_usize(metadata)?;
        let expected_sha256 = metadata.payload_sha256.clone();
        let data = materialize_store_tensor(&mut self.store, name)?.ok_or_else(|| {
            QuantizationError::InvalidArtifactTensor {
                name: name.to_owned(),
                reason: "manifest key is missing from Burnpack tensor map".into(),
            }
        })?;
        validate_u8_payload_descriptor(name, &data, expected)?;
        if let Some(expected_sha256) = expected_sha256 {
            let actual_sha256 = packed_payload_sha256(data.as_bytes());
            if actual_sha256 != expected_sha256 {
                return Err(QuantizationError::PackedPayloadDigestMismatch {
                    name: name.to_owned(),
                    expected: expected_sha256,
                    actual: actual_sha256,
                });
            }
        }
        Ok(data)
    }

    /// Materializes, decodes, and validates one packed tensor for reference execution.
    ///
    /// Version-two payloads are copied out of their mutable memory mapping before
    /// hashing. The digest therefore content-verifies the exact owned bytes decoded
    /// below, and is computed exactly once per call. The subsequent trusted
    /// constructor skips only the duplicate hash; it still validates metadata,
    /// lengths, scales, and every packed code.
    pub fn read_weight(&mut self, name: &str) -> Result<QuantizedWeight> {
        self.read_weight_audited(name)
            .map(|(weight, _)| weight)
            .map_err(|failure| failure.error)
    }

    /// Reads one weight while retaining exact owned-buffer progress on failure.
    pub(super) fn read_weight_audited(
        &mut self,
        name: &str,
    ) -> std::result::Result<(QuantizedWeight, BurnpackPayloadReadStats), BurnpackPayloadReadFailure>
    {
        let metadata =
            self.manifest
                .tensors
                .get(name)
                .cloned()
                .ok_or_else(|| BurnpackPayloadReadFailure {
                    error: QuantizationError::QuantizedTensorNotFound {
                        name: name.to_owned(),
                    },
                    stats: BurnpackPayloadReadStats::default(),
                })?;
        let expected = payload_len_usize(&metadata).map_err(empty_read_failure)?;
        let data = materialize_store_tensor(&mut self.store, name)
            .map_err(empty_read_failure)?
            .ok_or_else(|| BurnpackPayloadReadFailure {
                error: QuantizationError::InvalidArtifactTensor {
                    name: name.to_owned(),
                    reason: "manifest key is missing from Burnpack tensor map".into(),
                },
                stats: BurnpackPayloadReadStats::default(),
            })?;
        validate_u8_payload_descriptor(name, &data, expected).map_err(empty_read_failure)?;
        decode_payload(name, metadata, data.as_bytes())
    }

    /// Sequentially materializes and validates all payloads, dropping each before the next.
    pub fn validate_all(&mut self) -> Result<BurnpackValidationReport> {
        self.validate_all_audited()
            .map(|audit| BurnpackValidationReport {
                tensor_count: audit.tensors_validated,
                payload_bytes: audit.payload_bytes,
            })
            .map_err(|failure| failure.error)
    }

    /// Sequentially validates all payloads while preserving a failing prefix audit.
    pub(super) fn validate_all_audited(
        &mut self,
    ) -> std::result::Result<BurnpackValidationAudit, BurnpackValidationFailure> {
        let names: Vec<_> = self.manifest.tensors.keys().cloned().collect();
        let mut audit = BurnpackValidationAudit::default();
        for name in &names {
            let read = self.read_weight_audited(name);
            let read_stats = match &read {
                Ok((_, stats)) => *stats,
                Err(failure) => failure.stats,
            };
            if let Err(error) = record_validation_read(&mut audit, read_stats) {
                return Err(BurnpackValidationFailure { error, audit });
            }
            match read {
                Ok((_weight, _)) => {
                    let Some(tensors_validated) = audit.tensors_validated.checked_add(1) else {
                        return Err(BurnpackValidationFailure {
                            error: QuantizationError::ByteLengthOverflow {
                                context: "validated tensor count",
                            },
                            audit,
                        });
                    };
                    audit.tensors_validated = tensors_validated;
                }
                Err(failure) => {
                    return Err(BurnpackValidationFailure {
                        error: failure.error,
                        audit,
                    });
                }
            }
        }
        Ok(audit)
    }
}

/// Computes the exact persisted packed-execution discriminator.
fn packed_execution_support(manifest: &BurnpackQuantizationManifest) -> PackedExecutionSupport {
    if !manifest.tensors.is_empty()
        && manifest
            .tensors
            .values()
            .all(|metadata| metadata.validate_burn_qfloat_layout().is_ok())
    {
        PackedExecutionSupport::BurnQFloatMatmul
    } else {
        PackedExecutionSupport::Unavailable
    }
}

/// Session limits paired with actual or hypothetical staging counters.
struct LimitProjection<'a> {
    /// Limits the session enforces.
    limits: &'a BurnpackSessionLimits,
    /// Counters after the tensors staged (or planned) so far.
    stats: BurnpackStagingStats,
}

impl LimitProjection<'_> {
    /// Checks prospective count, payload, staging, and manifest bounds before writing.
    ///
    /// The manifest estimate charges each entry its exact pretty-printed size
    /// plus the digest line bound later (see [`manifest_entry_estimate`]); final
    /// serialization is still checked against the actual byte limit.
    fn preflight_append(
        &self,
        name: &str,
        metadata: &QuantizedTensorMetadata,
        payload_bytes: u64,
    ) -> Result<(u64, u64)> {
        // Verified: admitted counters stay within every limit (see `staging`).
        let admitted = admit_tensor(
            self.stats.tensor_count,
            self.stats.staged_payload_bytes,
            payload_bytes,
            self.limits.max_tensors,
            self.limits.max_tensor_payload_bytes,
            self.limits.max_staged_bytes,
        )
        .map_err(|breach| match breach {
            TensorBreach::CountOverflow => QuantizationError::ByteLengthOverflow {
                context: "staged tensor count",
            },
            TensorBreach::TooManyTensors { attempted } => QuantizationError::SessionLimitExceeded {
                kind: "tensor count",
                attempted: attempted as u64,
                limit: self.limits.max_tensors as u64,
            },
            TensorBreach::PayloadTooLarge => QuantizationError::SessionLimitExceeded {
                kind: "tensor payload",
                attempted: payload_bytes,
                limit: self.limits.max_tensor_payload_bytes,
            },
            TensorBreach::Staged(breach) => limit_breach(
                breach,
                "total staged payload length",
                "staged payload",
                self.limits.max_staged_bytes,
            ),
        })?;

        let entry_cost = manifest_entry_estimate(name, metadata)?;
        let next_estimate = bounded_add(
            self.stats.estimated_manifest_bytes,
            entry_cost,
            self.limits.max_manifest_bytes as u64,
        )
        .map_err(|breach| {
            limit_breach(
                breach,
                "manifest size estimate",
                "estimated manifest",
                self.limits.max_manifest_bytes as u64,
            )
        })?;
        Ok((admitted.staged, next_estimate))
    }
}

/// Maps a refused bounded total to its overflow or session-limit error.
fn limit_breach(
    breach: BoundBreach,
    overflow_context: &'static str,
    kind: &'static str,
    limit: u64,
) -> QuantizationError {
    match breach {
        BoundBreach::Overflow => QuantizationError::ByteLengthOverflow {
            context: overflow_context,
        },
        BoundBreach::Exceeded { attempted } => QuantizationError::SessionLimitExceeded {
            kind,
            attempted,
            limit,
        },
    }
}

/// Returns an upper bound on one tensor entry's bytes in the pretty manifest.
///
/// Inside the manifest's `tensors` map an entry is printed as
/// `    "<escaped name>": <metadata>,` where every line of the standalone pretty
/// metadata is indented [`MANIFEST_ENTRY_INDENT`] further. Version two adds one
/// digest line per entry when provenance is bound. The name's JSON escaping is
/// computed exactly rather than assumed to be worst case, so a long
/// many-expert inventory is not rejected for a manifest it would never write.
fn manifest_entry_estimate(name: &str, metadata: &QuantizedTensorMetadata) -> Result<u64> {
    let mut unbound = metadata.clone();
    unbound.payload_sha256 = None;
    let body = serde_json::to_vec_pretty(&unbound)?;
    let lines = body.iter().filter(|&&byte| byte == b'\n').count() + 1;
    let key = serde_json::to_string(name)?.len();
    let to_u64 = |value: usize| {
        u64::try_from(value).map_err(|_| QuantizationError::ByteLengthOverflow {
            context: "manifest entry estimate",
        })
    };
    to_u64(lines)?
        .checked_mul(MANIFEST_ENTRY_INDENT)
        .and_then(|indent| indent.checked_add(to_u64(body.len()).ok()?))
        .and_then(|bytes| bytes.checked_add(to_u64(key).ok()?))
        .and_then(|bytes| bytes.checked_add(MANIFEST_ENTRY_FRAMING + MANIFEST_DIGEST_LINE))
        .ok_or(QuantizationError::ByteLengthOverflow {
            context: "manifest entry estimate",
        })
}

/// Conservatively estimates late-bound checkpoint provenance before JSON allocation.
fn estimated_provenance_bytes(provenance: &BurnpackProvenance) -> Result<u64> {
    provenance.validate()?;
    provenance.checkpoint_shards.keys().try_fold(
        MANIFEST_PROVENANCE_BASE_OVERHEAD,
        |estimate, path| {
            let escaped_path = u64::try_from(path.len())
                .ok()
                .and_then(|length| length.checked_mul(JSON_STRING_ESCAPE_FACTOR))
                .ok_or(QuantizationError::ByteLengthOverflow {
                    context: "escaped provenance shard path estimate",
                })?;
            estimate
                .checked_add(MANIFEST_PROVENANCE_SHARD_OVERHEAD)
                .and_then(|estimate| estimate.checked_add(escaped_path))
                .ok_or(QuantizationError::ByteLengthOverflow {
                    context: "checkpoint provenance manifest estimate",
                })
        },
    )
}

/// Copies, content-verifies, decodes, and semantically validates one mapped payload.
///
/// Hashing occurs only after the complete canonical payload is owned. This order
/// prevents a concurrently mutated mapping from changing between the integrity
/// check and the copy consumed by reference execution.
fn decode_payload(
    name: &str,
    metadata: QuantizedTensorMetadata,
    bytes: &[u8],
) -> std::result::Result<(QuantizedWeight, BurnpackPayloadReadStats), BurnpackPayloadReadFailure> {
    let packed_len = usize::try_from(metadata.packed_value_bytes).map_err(|_| {
        empty_read_failure(QuantizationError::ByteLengthOverflow {
            context: "packed payload host length",
        })
    })?;
    let scale_len = usize::try_from(metadata.scale_bytes).map_err(|_| {
        empty_read_failure(QuantizationError::ByteLengthOverflow {
            context: "scale payload host length",
        })
    })?;
    let expected = packed_len.checked_add(scale_len).ok_or_else(|| {
        empty_read_failure(QuantizationError::ByteLengthOverflow {
            context: "combined payload host length",
        })
    })?;
    if bytes.len() != expected {
        return Err(empty_read_failure(
            QuantizationError::InvalidArtifactTensor {
                name: name.to_owned(),
                reason: format!("payload has {} bytes, expected {expected}", bytes.len()),
            },
        ));
    }

    let mut owned_payload = Vec::new();
    owned_payload.try_reserve_exact(expected).map_err(|_| {
        empty_read_failure(QuantizationError::AllocationFailed {
            context: "mapped quantized-payload copy",
            requested: expected,
        })
    })?;
    owned_payload.extend_from_slice(bytes);
    let mut stats = BurnpackPayloadReadStats {
        tensor_reads: 1,
        payload_bytes: u64::try_from(bytes.len()).map_err(|_| {
            empty_read_failure(QuantizationError::ByteLengthOverflow {
                context: "mapped quantized-payload byte accounting",
            })
        })?,
        peak_owned_payload_bytes: owned_payload_bytes(owned_payload.capacity(), 0)
            .map_err(empty_read_failure)?,
    };

    if let Some(expected_sha256) = &metadata.payload_sha256 {
        let actual_sha256 = packed_payload_sha256(&owned_payload);
        if actual_sha256 != *expected_sha256 {
            return Err(read_failure(
                QuantizationError::PackedPayloadDigestMismatch {
                    name: name.to_owned(),
                    expected: expected_sha256.clone(),
                    actual: actual_sha256,
                },
                stats,
            ));
        }
    }

    if !scale_len.is_multiple_of(4) {
        return Err(read_failure(
            QuantizationError::InvalidMetadata {
                reason: "scale byte count is not divisible by four".into(),
            },
            stats,
        ));
    }
    let scale_count = scale_len / 4;
    let mut scales = Vec::new();
    scales.try_reserve_exact(scale_count).map_err(|_| {
        read_failure(
            QuantizationError::AllocationFailed {
                context: "mapped quantization scales",
                requested: scale_count,
            },
            stats,
        )
    })?;
    stats.peak_owned_payload_bytes = stats.peak_owned_payload_bytes.max(
        owned_payload_bytes(owned_payload.capacity(), scales.capacity())
            .map_err(|error| read_failure(error, stats))?,
    );
    for chunk in owned_payload[packed_len..].chunks_exact(4) {
        scales.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    owned_payload.truncate(packed_len);
    let weight =
        QuantizedWeight::from_payload_digest_verified_parts(metadata, owned_payload, scales)
            .map_err(|error| read_failure(error, stats))?;
    Ok((weight, stats))
}

/// Wraps an error raised before a complete payload reaches owned memory.
fn empty_read_failure(error: QuantizationError) -> BurnpackPayloadReadFailure {
    read_failure(error, BurnpackPayloadReadStats::default())
}

/// Pairs a packed-read error with its exact completed allocation progress.
fn read_failure(
    error: QuantizationError,
    stats: BurnpackPayloadReadStats,
) -> BurnpackPayloadReadFailure {
    BurnpackPayloadReadFailure { error, stats }
}

/// Counts simultaneous capacity of the payload byte vector and decoded scales.
fn owned_payload_bytes(payload_capacity: usize, scale_capacity: usize) -> Result<u64> {
    let payload =
        u64::try_from(payload_capacity).map_err(|_| QuantizationError::ByteLengthOverflow {
            context: "owned packed-payload capacity",
        })?;
    let scales = scale_capacity
        .checked_mul(core::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(QuantizationError::ByteLengthOverflow {
            context: "owned packed-scale capacity",
        })?;
    payload
        .checked_add(scales)
        .ok_or(QuantizationError::ByteLengthOverflow {
            context: "owned packed decoder high-water mark",
        })
}

/// Adds one audited payload read to a sequential validation report atomically.
fn record_validation_read(
    audit: &mut BurnpackValidationAudit,
    read: BurnpackPayloadReadStats,
) -> Result<()> {
    let tensor_reads = audit.tensor_reads.checked_add(read.tensor_reads).ok_or(
        QuantizationError::ByteLengthOverflow {
            context: "Burnpack validation tensor reads",
        },
    )?;
    let payload_bytes = audit.payload_bytes.checked_add(read.payload_bytes).ok_or(
        QuantizationError::ByteLengthOverflow {
            context: "Burnpack validation payload bytes",
        },
    )?;
    audit.tensor_reads = tensor_reads;
    audit.payload_bytes = payload_bytes;
    audit.largest_payload_bytes = audit.largest_payload_bytes.max(read.payload_bytes);
    audit.peak_owned_payload_bytes = audit
        .peak_owned_payload_bytes
        .max(read.peak_owned_payload_bytes);
    Ok(())
}

/// Rejects mapped storage whose physical axes or dtype contradict the manifest.
fn validate_store_descriptors(
    store: &mut BurnpackStore,
    manifest: &BurnpackQuantizationManifest,
) -> Result<()> {
    for (name, metadata) in &manifest.tensors {
        let snapshot = store
            .get_snapshot(name)
            .map_err(|error| QuantizationError::BurnStore {
                operation: "inspect mapped Burnpack tensor descriptor",
                message: error.to_string(),
            })?
            .ok_or_else(|| QuantizationError::InvalidArtifactTensor {
                name: name.clone(),
                reason: "manifest key is missing from Burnpack tensor map".into(),
            })?;
        let expected = payload_len_usize(metadata)?;
        if snapshot.dtype != DType::U8 {
            return Err(QuantizationError::InvalidArtifactTensor {
                name: name.clone(),
                reason: format!("physical dtype is {:?}, expected U8", snapshot.dtype),
            });
        }
        if snapshot.shape.as_slice() != [expected] {
            return Err(QuantizationError::InvalidArtifactTensor {
                name: name.clone(),
                reason: format!(
                    "physical shape is {:?}, expected [{expected}]",
                    snapshot.shape.as_slice()
                ),
            });
        }
        let data = snapshot
            .to_data()
            .map_err(|error| QuantizationError::BurnStore {
                operation: "validate mapped Burnpack tensor range",
                message: error.to_string(),
            })?;
        validate_u8_payload_descriptor(name, &data, expected)?;
    }
    Ok(())
}

/// Bounds the manifest's one-dimensional U8 storage before reading its JSON.
fn validate_manifest_descriptor(
    store: &mut BurnpackStore,
    max_manifest_bytes: usize,
) -> Result<usize> {
    let snapshot = store
        .get_snapshot(BURNPACK_MANIFEST_TENSOR)
        .map_err(|error| QuantizationError::BurnStore {
            operation: "inspect quantization manifest descriptor",
            message: error.to_string(),
        })?
        .ok_or(QuantizationError::MissingManifestTensor {
            name: BURNPACK_MANIFEST_TENSOR,
        })?;
    if snapshot.dtype != DType::U8 {
        return Err(QuantizationError::InvalidArtifactTensor {
            name: BURNPACK_MANIFEST_TENSOR.into(),
            reason: format!("physical dtype is {:?}, expected U8", snapshot.dtype),
        });
    }
    let [manifest_bytes] = snapshot.shape.as_slice() else {
        return Err(QuantizationError::InvalidArtifactTensor {
            name: BURNPACK_MANIFEST_TENSOR.into(),
            reason: format!(
                "physical shape is {:?}, expected one dimension",
                snapshot.shape.as_slice()
            ),
        });
    };
    if *manifest_bytes > max_manifest_bytes {
        return Err(QuantizationError::SessionLimitExceeded {
            kind: "manifest",
            attempted: *manifest_bytes as u64,
            limit: max_manifest_bytes as u64,
        });
    }
    Ok(*manifest_bytes)
}

/// Requires exact physical/manifest key agreement apart from the reserved metadata tensor.
fn validate_key_sets(keys: &[String], manifest: &BurnpackQuantizationManifest) -> Result<()> {
    for name in manifest.tensors.keys() {
        if keys.binary_search(name).is_err() {
            return Err(QuantizationError::InvalidArtifactTensor {
                name: name.clone(),
                reason: "manifest key is missing from Burnpack tensor map".into(),
            });
        }
    }
    for name in keys {
        if name != BURNPACK_MANIFEST_TENSOR && !manifest.tensors.contains_key(name) {
            return Err(QuantizationError::InvalidArtifactTensor {
                name: name.clone(),
                reason: "Burnpack tensor is absent from quantization manifest".into(),
            });
        }
    }
    Ok(())
}

/// Materializes one mapped snapshot and preserves lookup/read operation context.
fn materialize_store_tensor(store: &mut BurnpackStore, name: &str) -> Result<Option<TensorData>> {
    store
        .get_snapshot(name)
        .map_err(|error| QuantizationError::BurnStore {
            operation: "look up mapped Burnpack tensor",
            message: error.to_string(),
        })?
        .map(|snapshot| {
            snapshot
                .to_data()
                .map_err(|error| QuantizationError::BurnStore {
                    operation: "materialize mapped Burnpack tensor",
                    message: error.to_string(),
                })
        })
        .transpose()
}

/// Checks dtype, logical byte-vector shape, and actual storage length independently.
fn validate_u8_payload_descriptor(name: &str, data: &TensorData, expected: usize) -> Result<()> {
    if data.dtype != DType::U8 {
        return Err(QuantizationError::InvalidArtifactTensor {
            name: name.to_owned(),
            reason: format!("physical dtype is {:?}, expected U8", data.dtype),
        });
    }
    if data.shape.as_slice() != [expected] {
        return Err(QuantizationError::InvalidArtifactTensor {
            name: name.to_owned(),
            reason: format!(
                "physical shape is {:?}, expected [{expected}]",
                data.shape.as_slice()
            ),
        });
    }
    if data.as_bytes().len() != expected {
        return Err(QuantizationError::InvalidArtifactTensor {
            name: name.to_owned(),
            reason: format!(
                "physical payload has {} bytes, expected {expected}",
                data.as_bytes().len()
            ),
        });
    }
    Ok(())
}

/// Validates portable metadata before converting its complete payload length to a host index.
fn payload_len_usize(metadata: &QuantizedTensorMetadata) -> Result<usize> {
    metadata.validate()?;
    let bytes = metadata
        .packed_value_bytes
        .checked_add(metadata.scale_bytes)
        .ok_or(QuantizationError::ByteLengthOverflow {
            context: "manifest tensor payload length",
        })?;
    usize::try_from(bytes).map_err(|_| QuantizationError::ByteLengthOverflow {
        context: "manifest tensor payload host conversion",
    })
}

// Lazy payload snapshots and private adjacent temporary-file lifecycles.

/// Creates a lazy one-dimensional U8 view of an exact staged byte interval.
///
/// The closure uses the retained file handle rather than reopening its pathname,
/// so replacing the path cannot redirect its payload read. Host length conversion
/// and allocation are checked when creating and materializing the snapshot.
fn staged_snapshot(
    staging_file: Rc<File>,
    staging_path: &Path,
    name: &str,
    offset: u64,
    payload_bytes: u64,
    parameter_id: ParamId,
) -> Result<TensorSnapshot> {
    let length =
        usize::try_from(payload_bytes).map_err(|_| QuantizationError::ByteLengthOverflow {
            context: "lazy snapshot payload host conversion",
        })?;
    let path_for_closure = staging_path.to_owned();
    let shape = Shape::from([length]);
    let closure_shape = shape.clone();
    let data_fn = Rc::new(move || {
        let mut file = staging_file.try_clone().map_err(|error| {
            TensorSnapshotError::IoError(format!(
                "failed to clone staging handle for `{}`: {error}",
                path_for_closure.display()
            ))
        })?;
        file.seek(SeekFrom::Start(offset)).map_err(|error| {
            TensorSnapshotError::IoError(format!(
                "failed to seek staging file `{}` to {offset}: {error}",
                path_for_closure.display()
            ))
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(|error| {
            TensorSnapshotError::DataError(format!(
                "failed to reserve {length} staged tensor bytes: {error}"
            ))
        })?;
        bytes.resize(length, 0);
        file.read_exact(&mut bytes).map_err(|error| {
            TensorSnapshotError::IoError(format!(
                "failed to read {length} bytes from staging file `{}`: {error}",
                path_for_closure.display()
            ))
        })?;
        Ok(TensorData::from_bytes_vec(
            bytes,
            closure_shape.clone(),
            DType::U8,
        ))
    });
    Ok(TensorSnapshot::from_closure(
        data_fn,
        DType::U8,
        shape,
        vec![name.to_owned()],
        vec!["QuantizedBurnpack".into()],
        parameter_id,
    ))
}

/// Validates the `.bpk` filename and canonicalizes its existing parent directory.
fn validate_bpk_path(path: &Path) -> Result<PathBuf> {
    let Some(filename) = path.file_name() else {
        return Err(QuantizationError::InvalidArtifactPath {
            path: path.to_owned(),
            reason: "expected a file path",
        });
    };
    if path.extension() != Some(OsStr::new("bpk")) {
        return Err(QuantizationError::InvalidArtifactPath {
            path: path.to_owned(),
            reason: "expected the `.bpk` extension",
        });
    }
    let parent = artifact_parent(path);
    if !parent.is_dir() {
        return Err(QuantizationError::InvalidArtifactPath {
            path: path.to_owned(),
            reason: "parent directory does not exist",
        });
    }
    let canonical_parent =
        fs::canonicalize(parent).map_err(|source| QuantizationError::ArtifactIo {
            operation: "canonicalize Burnpack artifact parent",
            path: parent.to_owned(),
            source,
        })?;
    Ok(canonical_parent.join(filename))
}

/// Returns an explicit parent or the current directory for a bare artifact filename.
fn artifact_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

/// Creates an exclusive adjacent staging file with bounded collision retries.
///
/// Unix permissions are owner-only; atomic `create_new` prevents overwriting an
/// existing file even if another process chooses the same candidate name.
fn create_unique_file(output: &Path, role: &'static str) -> Result<(PathBuf, File)> {
    let parent = artifact_parent(output);
    let filename = output
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("artifact.bpk");
    for _ in 0..128 {
        let unique = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{filename}.{}.{}.{}",
            std::process::id(),
            unique,
            role
        ));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&candidate) {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(QuantizationError::ArtifactIo {
                    operation: "create private Burnpack temporary file",
                    path: candidate,
                    source,
                });
            }
        }
    }
    Err(QuantizationError::ArtifactIo {
        operation: "create unique Burnpack temporary file",
        path: output.to_owned(),
        source: std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "exhausted 128 unique temporary names",
        ),
    })
}

/// Creates an exclusive adjacent output directory with bounded collision retries.
///
/// Unix directories are owner-only so the temporary artifact remains private until
/// the validated file is published.
fn create_unique_directory(output: &Path, role: &'static str) -> Result<PathBuf> {
    let parent = artifact_parent(output);
    let filename = output
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("artifact.bpk");
    for _ in 0..128 {
        let unique = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{filename}.{}.{}.{}",
            std::process::id(),
            unique,
            role
        ));
        let mut builder = DirBuilder::new();
        #[cfg(unix)]
        builder.mode(0o700);
        match builder.create(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(QuantizationError::ArtifactIo {
                    operation: "create private Burnpack temporary directory",
                    path: candidate,
                    source,
                });
            }
        }
    }
    Err(QuantizationError::ArtifactIo {
        operation: "create unique Burnpack temporary directory",
        path: output.to_owned(),
        source: std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "exhausted 128 unique temporary names",
        ),
    })
}

/// Requests durability of the published directory entry on Unix filesystems.
#[cfg(unix)]
fn sync_parent_directory(output: &Path) -> Result<()> {
    let parent = artifact_parent(output);
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| QuantizationError::ArtifactIo {
            operation: "synchronize Burnpack output directory",
            path: parent.to_owned(),
            source,
        })
}

/// Reports that directory-entry synchronization is unavailable on this platform.
#[cfg(not(unix))]
fn sync_parent_directory(output: &Path) -> Result<()> {
    Err(QuantizationError::ArtifactIo {
        operation: "synchronize Burnpack output directory",
        path: artifact_parent(output).to_owned(),
        source: std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "directory synchronization is not implemented on this platform",
        ),
    })
}

/// Validates `values` against the session limit tables starting at entry `first`.
///
/// The verified [`first_limit_violation`] finds the first limit outside
/// `[1, cap]`; this maps it to the contextual error of its field.
fn validate_session_limits(first: usize, values: &[u64]) -> Result<()> {
    let caps = &SESSION_LIMIT_CAPS[first..first + values.len()];
    match first_limit_violation(values, caps) {
        None => Ok(()),
        Some((index, violation)) => Err(QuantizationError::InvalidSessionLimit {
            field: SESSION_LIMIT_FIELDS[first + index],
            value: values[index],
            reason: match violation {
                LimitViolation::Zero => "must be greater than zero",
                LimitViolation::AboveCap => SESSION_LIMIT_REASONS[first + index],
            },
        }),
    }
}

/// Removes unpublished serialization output on success and every early-return path.
struct TemporaryOutput {
    /// Private adjacent directory created for the serialization attempt.
    directory: PathBuf,
    /// Temporary Burnpack output inside the private directory.
    path: PathBuf,
}

impl TemporaryOutput {
    /// Registers a temporary artifact and its directory for cleanup on every return path.
    fn new(directory: PathBuf, path: PathBuf) -> Self {
        Self { directory, path }
    }
}

impl Drop for TemporaryOutput {
    /// Removes the private temporary file and directory without touching published output.
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
        let _ = fs::remove_dir(&self.directory);
    }
}

#[cfg(test)]
#[path = "burnpack_validation_edge_tests.rs"]
mod validation_edge_tests;

#[cfg(test)]
#[path = "burnpack_test_support.rs"]
mod test_support;
