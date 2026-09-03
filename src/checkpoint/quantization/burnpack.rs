//! Persistent quantized artifacts backed by Burn Store's official Burnpack format.
//!
//! Burnpack 0.21 does not expose its string metadata through [`BurnpackStore`],
//! and this module's symmetric flat-group layout is intentionally distinct from
//! Burn's `DType::QFloat` layouts. Each quantized payload is therefore stored as
//! an ordinary U8 Burnpack tensor. A reserved U8 tensor contains the JSON
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
    BurnpackQuantizationManifest, QuantizationEligibility, QuantizationError,
    QuantizationExclusion, QuantizedTensorMetadata, QuantizedWeight, Result, SourceFloatDType,
    WeightQuantizationConfig, classify_weight_for_quantization, quantize_tensor_data,
};

/// Reserved Burnpack U8 tensor containing the quantization manifest JSON.
pub const BURNPACK_MANIFEST_TENSOR: &str = "__transformers_quantization_manifest_v1__";

const UPSTREAM_MAX_TENSORS: usize = 100_000;
#[cfg(target_pointer_width = "32")]
const DEFAULT_MAX_STAGED_BYTES: u64 = 3 * 1024 * 1024 * 1024;
#[cfg(not(target_pointer_width = "32"))]
const DEFAULT_MAX_STAGED_BYTES: u64 = 90 * 1024 * 1024 * 1024;
#[cfg(target_pointer_width = "32")]
const MAX_STAGED_REASON: &str = "must be at most 3 GiB on a 32-bit host";
#[cfg(not(target_pointer_width = "32"))]
const MAX_STAGED_REASON: &str =
    "must be at most 90 GiB to leave room below Burnpack's 100 GiB file limit";
const DEFAULT_MAX_MANIFEST_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_MAX_TENSOR_NAME_BYTES: usize = 16 * 1024;
#[cfg(target_pointer_width = "32")]
const UPSTREAM_MAX_TENSOR_BYTES: u64 = 2 * 1024 * 1024 * 1024;
#[cfg(not(target_pointer_width = "32"))]
const UPSTREAM_MAX_TENSOR_BYTES: u64 = 10 * 1024 * 1024 * 1024;
const MANIFEST_BASE_ESTIMATE: u64 = 2048;
const MANIFEST_ENTRY_OVERHEAD: u64 = 2048;
const MANIFEST_SHAPE_DIMENSION_OVERHEAD: u64 = 32;
const JSON_STRING_ESCAPE_FACTOR: u64 = 6;

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
    pub fn validate(self) -> Result<Self> {
        validate_nonzero_limit("max_tensors", self.max_tensors as u64)?;
        if self.max_tensors >= UPSTREAM_MAX_TENSORS {
            return Err(QuantizationError::InvalidSessionLimit {
                field: "max_tensors",
                value: self.max_tensors as u64,
                reason: "must leave one of Burn Store's 100,000 tensor slots for the manifest",
            });
        }
        validate_nonzero_limit("max_staged_bytes", self.max_staged_bytes)?;
        if self.max_staged_bytes > DEFAULT_MAX_STAGED_BYTES {
            return Err(QuantizationError::InvalidSessionLimit {
                field: "max_staged_bytes",
                value: self.max_staged_bytes,
                reason: MAX_STAGED_REASON,
            });
        }
        validate_nonzero_limit("max_tensor_payload_bytes", self.max_tensor_payload_bytes)?;
        if self.max_tensor_payload_bytes > UPSTREAM_MAX_TENSOR_BYTES {
            return Err(QuantizationError::InvalidSessionLimit {
                field: "max_tensor_payload_bytes",
                value: self.max_tensor_payload_bytes,
                reason: "exceeds Burn Store's platform-specific per-tensor reader limit",
            });
        }
        validate_nonzero_limit("max_manifest_bytes", self.max_manifest_bytes as u64)?;
        if self.max_manifest_bytes > DEFAULT_MAX_MANIFEST_BYTES {
            return Err(QuantizationError::InvalidSessionLimit {
                field: "max_manifest_bytes",
                value: self.max_manifest_bytes as u64,
                reason: "must be at most 64 MiB",
            });
        }
        validate_nonzero_limit("max_tensor_name_bytes", self.max_tensor_name_bytes as u64)?;
        if self.max_tensor_name_bytes > DEFAULT_MAX_TENSOR_NAME_BYTES {
            return Err(QuantizationError::InvalidSessionLimit {
                field: "max_tensor_name_bytes",
                value: self.max_tensor_name_bytes as u64,
                reason: "must be at most 16 KiB",
            });
        }
        Ok(self)
    }
}

impl Default for BurnpackSessionLimits {
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
    Staged(StagedTensorSummary),
    /// Policy left the tensor in its separate source checkpoint.
    Excluded(QuantizationExclusion),
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

#[derive(Debug, Clone)]
struct StagedTensor {
    offset: u64,
    payload_bytes: u64,
}

/// Incremental, one-source-tensor-at-a-time quantized Burnpack writer.
///
/// Dropping an unfinished or completed session removes its private staging file.
/// The final output is never visible until Burn Store has written and synchronized
/// a complete temporary Burnpack file.
pub struct QuantizedBurnpackSession {
    output_path: PathBuf,
    staging_path: PathBuf,
    staging_file: Option<File>,
    entries: BTreeMap<String, StagedTensor>,
    manifest: BurnpackQuantizationManifest,
    limits: BurnpackSessionLimits,
    stats: BurnpackStagingStats,
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
            manifest: BurnpackQuantizationManifest::new(),
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

    /// Applies policy, quantizes one source TensorData, and releases it before the next call.
    pub fn append(
        &mut self,
        name: &str,
        source: &TensorData,
        config: WeightQuantizationConfig,
    ) -> Result<BurnpackAppendOutcome> {
        self.ensure_mutable()?;
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

        let manifest_bytes = self.manifest.to_json_pretty()?.into_bytes();
        self.check_manifest_size(manifest_bytes.len())?;
        let manifest_len = u64::try_from(manifest_bytes.len()).map_err(|_| {
            QuantizationError::ByteLengthOverflow {
                context: "manifest byte length conversion",
            }
        })?;
        let staged_with_manifest = self
            .stats
            .staged_payload_bytes
            .checked_add(manifest_len)
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "staging length with manifest",
            })?;
        if staged_with_manifest > self.limits.max_staged_bytes {
            return Err(QuantizationError::SessionLimitExceeded {
                kind: "staged payload",
                attempted: staged_with_manifest,
                limit: self.limits.max_staged_bytes,
            });
        }

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

    fn append_quantized_inner(
        &mut self,
        name: &str,
        weight: QuantizedWeight,
    ) -> Result<StagedTensorSummary> {
        let metadata = weight.metadata().clone();
        metadata.validate()?;
        let payload_bytes = metadata
            .packed_value_bytes
            .checked_add(metadata.scale_bytes)
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "quantized tensor payload length",
            })?;
        let (next_staged, next_estimate) = self.preflight_append(name, &metadata, payload_bytes)?;
        let offset = self.checked_staging_end()?;

        let write_result = {
            let file = self.staging_file_mut()?;
            weight.write_burnpack_payload(file)
        };
        if let Err(error) = write_result {
            return self.rollback_after(offset, error);
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

    fn preflight_append(
        &self,
        name: &str,
        metadata: &QuantizedTensorMetadata,
        payload_bytes: u64,
    ) -> Result<(u64, u64)> {
        let next_count = self.stats.tensor_count.checked_add(1).ok_or(
            QuantizationError::ByteLengthOverflow {
                context: "staged tensor count",
            },
        )?;
        if next_count > self.limits.max_tensors {
            return Err(QuantizationError::SessionLimitExceeded {
                kind: "tensor count",
                attempted: next_count as u64,
                limit: self.limits.max_tensors as u64,
            });
        }
        if payload_bytes > self.limits.max_tensor_payload_bytes {
            return Err(QuantizationError::SessionLimitExceeded {
                kind: "tensor payload",
                attempted: payload_bytes,
                limit: self.limits.max_tensor_payload_bytes,
            });
        }
        let next_staged = self
            .stats
            .staged_payload_bytes
            .checked_add(payload_bytes)
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "total staged payload length",
            })?;
        if next_staged > self.limits.max_staged_bytes {
            return Err(QuantizationError::SessionLimitExceeded {
                kind: "staged payload",
                attempted: next_staged,
                limit: self.limits.max_staged_bytes,
            });
        }

        let shape_cost = u64::try_from(metadata.shape.len())
            .ok()
            .and_then(|rank| rank.checked_mul(MANIFEST_SHAPE_DIMENSION_OVERHEAD))
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "manifest shape estimate",
            })?;
        let name_cost = u64::try_from(name.len())
            .ok()
            .and_then(|length| length.checked_mul(JSON_STRING_ESCAPE_FACTOR))
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "escaped manifest tensor name length",
            })?;
        let entry_cost = MANIFEST_ENTRY_OVERHEAD
            .checked_add(shape_cost)
            .and_then(|cost| cost.checked_add(name_cost))
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "manifest entry estimate",
            })?;
        let next_estimate = self
            .stats
            .estimated_manifest_bytes
            .checked_add(entry_cost)
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "manifest size estimate",
            })?;
        if next_estimate > self.limits.max_manifest_bytes as u64 {
            return Err(QuantizationError::SessionLimitExceeded {
                kind: "estimated manifest",
                attempted: next_estimate,
                limit: self.limits.max_manifest_bytes as u64,
            });
        }
        Ok((next_staged, next_estimate))
    }

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

    fn ensure_mutable(&self) -> Result<()> {
        if self.finalized {
            Err(QuantizationError::SessionFinalized)
        } else {
            Ok(())
        }
    }

    fn staging_file_mut(&mut self) -> Result<&mut File> {
        self.staging_file
            .as_mut()
            .ok_or(QuantizationError::StagingCorrupted {
                expected: self.stats.staged_payload_bytes,
                actual: 0,
            })
    }

    fn checked_staging_end(&mut self) -> Result<u64> {
        let expected = self.stats.staged_payload_bytes;
        let actual = self.staging_end()?;
        if actual != expected {
            return Err(QuantizationError::StagingCorrupted { expected, actual });
        }
        Ok(actual)
    }

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
            parent_directory_synchronized,
        })
    }

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

    #[cfg(test)]
    pub(super) fn staging_path_for_test(&self) -> &Path {
        &self.staging_path
    }

    #[cfg(test)]
    pub(super) fn staging_len_for_test(&self) -> u64 {
        self.staging_file
            .as_ref()
            .unwrap()
            .metadata()
            .unwrap()
            .len()
    }

    #[cfg(test)]
    pub(super) fn truncate_staging_for_test(&mut self, length: u64) {
        self.staging_file.as_ref().unwrap().set_len(length).unwrap();
    }

    fn close_staging(&mut self) {
        if let Some(file) = self.staging_file.take() {
            drop(file);
        }
        let _ = fs::remove_file(&self.staging_path);
    }
}

impl Drop for QuantizedBurnpackSession {
    fn drop(&mut self) {
        self.close_staging();
    }
}

/// Memory-mapped, lazily materialized view of a quantized Burnpack artifact.
pub struct QuantizedBurnpackArtifact {
    path: PathBuf,
    store: BurnpackStore,
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
        validate_nonzero_limit("max_manifest_bytes", max_manifest_bytes as u64)?;
        if max_manifest_bytes > DEFAULT_MAX_MANIFEST_BYTES {
            return Err(QuantizationError::InvalidSessionLimit {
                field: "max_manifest_bytes",
                value: max_manifest_bytes as u64,
                reason: "must be at most 64 MiB",
            });
        }
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
        let data = materialize_store_tensor(&mut self.store, name)?.ok_or_else(|| {
            QuantizationError::InvalidArtifactTensor {
                name: name.to_owned(),
                reason: "manifest key is missing from Burnpack tensor map".into(),
            }
        })?;
        validate_u8_payload_descriptor(name, &data, expected)?;
        Ok(data)
    }

    /// Materializes, decodes, and validates one packed tensor for reference execution.
    pub fn read_weight(&mut self, name: &str) -> Result<QuantizedWeight> {
        let metadata = self.manifest.tensors.get(name).cloned().ok_or_else(|| {
            QuantizationError::QuantizedTensorNotFound {
                name: name.to_owned(),
            }
        })?;
        let data = self.tensor_payload(name)?;
        decode_payload(metadata, data.as_bytes())
    }

    /// Sequentially materializes and validates all payloads, dropping each before the next.
    pub fn validate_all(&mut self) -> Result<BurnpackValidationReport> {
        let names: Vec<_> = self.manifest.tensors.keys().cloned().collect();
        let mut payload_bytes = 0_u64;
        for name in &names {
            let weight = self.read_weight(name)?;
            let tensor_bytes = weight
                .metadata()
                .packed_value_bytes
                .checked_add(weight.metadata().scale_bytes)
                .ok_or(QuantizationError::ByteLengthOverflow {
                    context: "validated payload byte count",
                })?;
            payload_bytes = payload_bytes.checked_add(tensor_bytes).ok_or(
                QuantizationError::ByteLengthOverflow {
                    context: "validated artifact payload byte count",
                },
            )?;
        }
        Ok(BurnpackValidationReport {
            tensor_count: names.len(),
            payload_bytes,
        })
    }
}

fn decode_payload(metadata: QuantizedTensorMetadata, bytes: &[u8]) -> Result<QuantizedWeight> {
    let packed_len = usize::try_from(metadata.packed_value_bytes).map_err(|_| {
        QuantizationError::ByteLengthOverflow {
            context: "packed payload host length",
        }
    })?;
    let scale_len = usize::try_from(metadata.scale_bytes).map_err(|_| {
        QuantizationError::ByteLengthOverflow {
            context: "scale payload host length",
        }
    })?;
    let expected =
        packed_len
            .checked_add(scale_len)
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "combined payload host length",
            })?;
    if bytes.len() != expected {
        return Err(QuantizationError::InvalidArtifactTensor {
            name: "<decoded payload>".into(),
            reason: format!("payload has {} bytes, expected {expected}", bytes.len()),
        });
    }

    let mut packed = Vec::new();
    packed
        .try_reserve_exact(packed_len)
        .map_err(|_| QuantizationError::AllocationFailed {
            context: "mapped packed-value copy",
            requested: packed_len,
        })?;
    packed.extend_from_slice(&bytes[..packed_len]);

    if !scale_len.is_multiple_of(4) {
        return Err(QuantizationError::InvalidMetadata {
            reason: "scale byte count is not divisible by four".into(),
        });
    }
    let scale_count = scale_len / 4;
    let mut scales = Vec::new();
    scales
        .try_reserve_exact(scale_count)
        .map_err(|_| QuantizationError::AllocationFailed {
            context: "mapped quantization scales",
            requested: scale_count,
        })?;
    for chunk in bytes[packed_len..].chunks_exact(4) {
        scales.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    QuantizedWeight::from_parts(metadata, packed, scales)
}

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

fn validate_bpk_path(path: &Path) -> Result<PathBuf> {
    if path.as_os_str().is_empty() || path.file_name().is_none() {
        return Err(QuantizationError::InvalidArtifactPath {
            path: path.to_owned(),
            reason: "expected a file path",
        });
    }
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
    Ok(canonical_parent.join(
        path.file_name()
            .expect("validated artifact path always has a file name"),
    ))
}

fn artifact_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

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

fn validate_nonzero_limit(field: &'static str, value: u64) -> Result<()> {
    if value == 0 {
        Err(QuantizationError::InvalidSessionLimit {
            field,
            value,
            reason: "must be greater than zero",
        })
    } else {
        Ok(())
    }
}

struct TemporaryOutput {
    directory: PathBuf,
    path: PathBuf,
}

impl TemporaryOutput {
    fn new(directory: PathBuf, path: PathBuf) -> Self {
        Self { directory, path }
    }
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
        let _ = fs::remove_dir(&self.directory);
    }
}
