//! Content-integrity binding for Burnpack sidecars.
//!
//! A packed weight has meaning only relative to both the bytes from which it was
//! quantized and the architecture transformation that produced its final tensor.
//! Version-two manifests therefore bind two independent domains:
//!
//! - every complete SafeTensors shard, including its header, is identified by a
//!   SHA-256 digest and exact byte length; and
//! - the complete storage-owning, alias, and physical-equivalence topology of the
//!   validated [`LoadPlan`] is encoded with length-prefixed fields and hashed.
//!
//! Length prefixes make the plan encoding injective without depending on JSON
//! spelling or Rust's unstable debug representation. Hashing is streamed through
//! a fixed-size buffer, so checkpoint provenance requires constant host memory.
//!
//! These hashes detect accidental corruption and bind a sidecar to externally
//! trusted checkpoint and manifest contents. They are not signatures and do not
//! establish the origin authenticity of an artifact supplied by a hostile party: it can
//! replace its manifest and payload hashes together. Local pre/post verification
//! additionally assumes ordinary, non-adversarial filesystem concurrency. An
//! attacker able to restore the exact expected bytes between observations (an
//! ABA replacement) remains outside this mechanism's guarantees.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Read,
    path::{Component, Path},
};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{QuantizationError, Result};
use crate::checkpoint::{
    CheckpointManifest, LoadPlan, PlannedSource, PlannedTensor, PlannedTransform,
    artifact::deserialize_unique_map,
};

/// Domain separator for the canonical v2 load-plan digest.
const PLAN_DOMAIN: &[u8] = b"transformers.burnpack.load-plan.v2\0";
/// Fixed streaming buffer used while hashing large checkpoint shards.
pub const PROVENANCE_HASH_BUFFER_BYTES: usize = 1024 * 1024;

/// SHA-256 identity and exact length of one complete SafeTensors shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BurnpackShardProvenance {
    /// Complete file length, including SafeTensors header and data section.
    pub file_bytes: u64,
    /// Lowercase hexadecimal SHA-256 over the complete file contents.
    pub sha256: String,
}

impl BurnpackShardProvenance {
    /// Validates the fixed-width lowercase digest representation.
    pub fn validate(&self) -> Result<()> {
        validate_sha256(&self.sha256, "checkpoint shard")
    }
}

/// Content identity of the checkpoint and architecture plan behind a sidecar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BurnpackProvenance {
    /// Lowercase SHA-256 of the canonical, domain-separated [`LoadPlan`].
    pub load_plan_sha256: String,
    /// Complete shard identities keyed by validated root-relative path.
    #[serde(deserialize_with = "deserialize_unique_map")]
    pub checkpoint_shards: BTreeMap<String, BurnpackShardProvenance>,
}

impl BurnpackProvenance {
    /// Validates digest spelling, shard keys, and the non-empty source inventory.
    pub fn validate(&self) -> Result<()> {
        validate_sha256(&self.load_plan_sha256, "load plan")?;
        if self.checkpoint_shards.is_empty() {
            return Err(QuantizationError::InvalidProvenance {
                reason: "checkpoint shard digest map is empty".into(),
            });
        }
        for (relative_path, shard) in &self.checkpoint_shards {
            // Path::components removes internal `.` segments. Check the original
            // spelling as well, so one shard cannot have multiple lexical keys.
            if relative_path.is_empty()
                || relative_path.contains('\0')
                || relative_path.contains('\\')
                || has_windows_drive_prefix(relative_path)
                || relative_path
                    .split('/')
                    .any(|component| matches!(component, "" | "." | ".."))
                || Path::new(relative_path).is_absolute()
                || Path::new(relative_path)
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_)))
                || !relative_path.ends_with(".safetensors")
            {
                return Err(QuantizationError::InvalidProvenance {
                    reason: format!(
                        "checkpoint shard path `{relative_path}` is not a normalized relative .safetensors path"
                    ),
                });
            }
            if shard.file_bytes == 0 {
                return Err(QuantizationError::InvalidProvenance {
                    reason: format!("checkpoint shard `{relative_path}` has zero length"),
                });
            }
            shard.validate()?;
        }
        Ok(())
    }
}

/// Bounded I/O performed while hashing local checkpoint content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProvenanceHashStats {
    /// Complete hash passes that reached the end of every shard.
    pub passes_completed: u64,
    /// Complete checkpoint shards hashed successfully.
    pub shards_hashed: usize,
    /// Bytes read and fed to SHA-256, including a partial shard on failure.
    pub bytes_hashed: u64,
    /// Largest fixed read buffer allocated by any pass.
    ///
    /// This is either zero when no buffer was allocated or exactly
    /// [`PROVENANCE_HASH_BUFFER_BYTES`]. It is separate from model staging bounds.
    pub peak_buffer_bytes: usize,
}

/// Provenance hashing performed on the two sides of model execution.
///
/// `None` means that pass was not attempted, as for a v1 load or an earlier
/// contract failure. `Some` retains partial byte and shard progress even when
/// the corresponding pass failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProvenanceExecutionStats {
    /// Hashing completed or attempted before checkpoint execution.
    pub pre_execution: Option<ProvenanceHashStats>,
    /// Hashing completed or attempted after checkpoint execution.
    pub post_execution: Option<ProvenanceHashStats>,
}

impl ProvenanceExecutionStats {
    /// Returns total bytes fed to SHA-256 across both sequential passes.
    #[must_use]
    pub fn bytes_hashed(self) -> u128 {
        u128::from(self.pre_execution.map_or(0, |stats| stats.bytes_hashed))
            + u128::from(self.post_execution.map_or(0, |stats| stats.bytes_hashed))
    }

    /// Returns the maximum additional hash-buffer ownership across either pass.
    #[must_use]
    pub fn peak_buffer_bytes(self) -> usize {
        self.pre_execution
            .map_or(0, |stats| stats.peak_buffer_bytes)
            .max(
                self.post_execution
                    .map_or(0, |stats| stats.peak_buffer_bytes),
            )
    }
}

/// Internal hashing error that retains I/O performed before failure.
pub(crate) struct ProvenanceHashFailure {
    /// Root quantization or filesystem error.
    pub error: QuantizationError,
    /// Hashing counters through the failing read.
    pub stats: ProvenanceHashStats,
}

/// Computes the canonical, domain-separated v2 load-plan SHA-256.
#[must_use]
pub fn load_plan_sha256(plan: &LoadPlan) -> String {
    let mut hasher = Sha256::new();
    hasher.update(PLAN_DOMAIN);

    hash_len(&mut hasher, plan.equivalences.len());
    for equivalence in &plan.equivalences {
        hash_source(&mut hasher, &equivalence.alias);
        hash_source(&mut hasher, &equivalence.owner);
    }

    hash_len(&mut hasher, plan.tensors.len());
    for tensor in &plan.tensors {
        hash_tensor(&mut hasher, tensor);
    }

    hash_len(&mut hasher, plan.aliases.len());
    for alias in &plan.aliases {
        hash_tensor(&mut hasher, alias);
    }

    hex_digest(hasher.finalize())
}

/// Hashes every complete local shard and combines it with the exact load-plan digest.
///
/// Paths and opened files are revalidated against the discovery manifest before
/// bytes enter SHA-256. Callers doing conversion should hash once before and once
/// after quantization and require equality, preventing a checkpoint modification
/// from silently spanning the conversion interval.
pub fn hash_local_checkpoint_provenance(
    manifest: &CheckpointManifest,
    plan: &LoadPlan,
) -> Result<(BurnpackProvenance, ProvenanceHashStats)> {
    hash_local_checkpoint_provenance_audited(manifest, plan).map_err(|failure| failure.error)
}

/// Hashes local provenance while retaining partial I/O counters on failure.
pub(crate) fn hash_local_checkpoint_provenance_audited(
    manifest: &CheckpointManifest,
    plan: &LoadPlan,
) -> std::result::Result<(BurnpackProvenance, ProvenanceHashStats), ProvenanceHashFailure> {
    let mut checkpoint_shards = BTreeMap::new();
    let mut stats = ProvenanceHashStats::default();
    for shard in &manifest.shards {
        let digest = hash_shard(manifest, shard, &mut stats)
            .map_err(|error| ProvenanceHashFailure { error, stats })?;
        if checkpoint_shards
            .insert(shard.relative_path.clone(), digest)
            .is_some()
        {
            return Err(ProvenanceHashFailure {
                error: QuantizationError::InvalidProvenance {
                    reason: format!(
                        "checkpoint manifest repeats shard `{}`",
                        shard.relative_path
                    ),
                },
                stats,
            });
        }
    }
    stats.passes_completed = 1;
    let provenance = BurnpackProvenance {
        load_plan_sha256: load_plan_sha256(plan),
        checkpoint_shards,
    };
    provenance
        .validate()
        .map_err(|error| ProvenanceHashFailure { error, stats })?;
    Ok((provenance, stats))
}

/// Verifies an artifact's exact plan binding without accessing checkpoint bytes.
pub fn verify_load_plan_provenance(plan: &LoadPlan, provenance: &BurnpackProvenance) -> Result<()> {
    provenance.validate()?;
    let actual = load_plan_sha256(plan);
    if actual != provenance.load_plan_sha256 {
        return Err(QuantizationError::LoadPlanProvenanceMismatch {
            expected: provenance.load_plan_sha256.clone(),
            actual,
        });
    }
    Ok(())
}

/// Verifies every complete local checkpoint shard against persisted v2 provenance.
pub fn verify_local_checkpoint_provenance(
    manifest: &CheckpointManifest,
    plan: &LoadPlan,
    expected: &BurnpackProvenance,
) -> Result<ProvenanceHashStats> {
    verify_load_plan_provenance(plan, expected)?;
    preflight_local_checkpoint_provenance(manifest, expected)?;
    let (actual, stats) = hash_local_checkpoint_provenance(manifest, plan)?;
    compare_checkpoint_provenance(&actual, expected)?;
    Ok(stats)
}

/// Rejects mismatched shard sets and lengths before allocating a hash buffer.
pub(crate) fn preflight_local_checkpoint_provenance(
    manifest: &CheckpointManifest,
    expected: &BurnpackProvenance,
) -> Result<()> {
    expected.validate()?;
    if manifest.shards.len() != expected.checkpoint_shards.len() {
        return Err(QuantizationError::InvalidProvenance {
            reason: format!(
                "checkpoint shard count {} does not match sidecar count {}",
                manifest.shards.len(),
                expected.checkpoint_shards.len()
            ),
        });
    }

    let mut observed = BTreeMap::new();
    for shard in &manifest.shards {
        if observed.insert(shard.relative_path.as_str(), ()).is_some() {
            return Err(QuantizationError::InvalidProvenance {
                reason: format!(
                    "checkpoint manifest repeats shard `{}`",
                    shard.relative_path
                ),
            });
        }
        let expected_shard = expected
            .checkpoint_shards
            .get(&shard.relative_path)
            .ok_or_else(|| QuantizationError::InvalidProvenance {
                reason: format!(
                    "checkpoint manifest contains shard `{}` absent from sidecar provenance",
                    shard.relative_path
                ),
            })?;
        if shard.file_bytes != expected_shard.file_bytes {
            return Err(QuantizationError::CheckpointProvenanceLengthMismatch {
                shard: shard.relative_path.clone(),
                expected: expected_shard.file_bytes,
                actual: shard.file_bytes,
            });
        }
        let metadata =
            fs::symlink_metadata(&shard.path).map_err(|source| QuantizationError::ArtifactIo {
                operation: "inspect checkpoint shard before provenance hashing",
                path: shard.path.clone(),
                source,
            })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(QuantizationError::InvalidProvenance {
                reason: format!(
                    "checkpoint shard `{}` is not a regular non-symlink file",
                    shard.relative_path
                ),
            });
        }
        if metadata.len() != expected_shard.file_bytes {
            return Err(QuantizationError::CheckpointProvenanceLengthMismatch {
                shard: shard.relative_path.clone(),
                expected: expected_shard.file_bytes,
                actual: metadata.len(),
            });
        }
    }
    Ok(())
}

/// Compares a freshly hashed complete shard inventory with persisted provenance.
pub(crate) fn compare_checkpoint_provenance(
    actual: &BurnpackProvenance,
    expected: &BurnpackProvenance,
) -> Result<()> {
    if actual.checkpoint_shards.len() != expected.checkpoint_shards.len() {
        return Err(QuantizationError::InvalidProvenance {
            reason: format!(
                "checkpoint shard count {} does not match sidecar count {}",
                actual.checkpoint_shards.len(),
                expected.checkpoint_shards.len()
            ),
        });
    }
    for (relative_path, expected_shard) in &expected.checkpoint_shards {
        let actual_shard = actual.checkpoint_shards.get(relative_path).ok_or_else(|| {
            QuantizationError::InvalidProvenance {
                reason: format!("sidecar references absent checkpoint shard `{relative_path}`"),
            }
        })?;
        if actual_shard.file_bytes != expected_shard.file_bytes {
            return Err(QuantizationError::CheckpointProvenanceLengthMismatch {
                shard: relative_path.clone(),
                expected: expected_shard.file_bytes,
                actual: actual_shard.file_bytes,
            });
        }
        if actual_shard.sha256 != expected_shard.sha256 {
            return Err(QuantizationError::CheckpointProvenanceDigestMismatch {
                shard: relative_path.clone(),
                expected: expected_shard.sha256.clone(),
                actual: actual_shard.sha256.clone(),
            });
        }
    }
    Ok(())
}

/// Hashes a tensor work item including all post-transform and source semantics.
fn hash_tensor(hasher: &mut Sha256, tensor: &PlannedTensor) {
    hash_str(hasher, &tensor.target);
    hash_len(hasher, tensor.sources.len());
    for source in &tensor.sources {
        hash_source(hasher, source);
    }
    match &tensor.transform {
        PlannedTransform::Identity => hasher.update([0]),
        PlannedTransform::Transpose2d => hasher.update([1]),
        PlannedTransform::Slice {
            axis,
            start,
            length,
        } => {
            hasher.update([2]);
            hash_usize(hasher, *axis);
            hasher.update(start.to_le_bytes());
            hasher.update(length.to_le_bytes());
        }
        PlannedTransform::Concatenate { axis } => {
            hasher.update([3]);
            hash_usize(hasher, *axis);
        }
        PlannedTransform::Alias { source_target } => {
            hasher.update([4]);
            hash_str(hasher, source_target);
        }
    }
    hash_shape(hasher, &tensor.shape);
    hash_str(hasher, tensor.dtype.as_str());
}

/// Hashes one immutable SafeTensors descriptor with unambiguous field boundaries.
fn hash_source(hasher: &mut Sha256, source: &PlannedSource) {
    hash_str(hasher, &source.name);
    hash_str(hasher, &source.shard);
    hash_shape(hasher, &source.shape);
    hash_str(hasher, source.dtype.as_str());
    hasher.update(source.data_offsets[0].to_le_bytes());
    hasher.update(source.data_offsets[1].to_le_bytes());
}

/// Hashes one complete shard after path, inode, and length revalidation.
fn hash_shard(
    manifest: &CheckpointManifest,
    shard: &crate::checkpoint::ShardInventory,
    stats: &mut ProvenanceHashStats,
) -> Result<BurnpackShardProvenance> {
    let canonical =
        fs::canonicalize(&shard.path).map_err(|source| QuantizationError::ArtifactIo {
            operation: "canonicalize checkpoint shard for provenance",
            path: shard.path.clone(),
            source,
        })?;
    if canonical != shard.path || !canonical.starts_with(&manifest.root) {
        return Err(QuantizationError::InvalidProvenance {
            reason: format!(
                "checkpoint shard `{}` resolves outside its discovered canonical path",
                shard.relative_path
            ),
        });
    }
    let path_metadata =
        fs::symlink_metadata(&canonical).map_err(|source| QuantizationError::ArtifactIo {
            operation: "inspect checkpoint shard for provenance",
            path: canonical.clone(),
            source,
        })?;
    if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
        return Err(QuantizationError::InvalidProvenance {
            reason: format!(
                "checkpoint shard `{}` is not a regular non-symlink file",
                shard.relative_path
            ),
        });
    }
    let mut file = File::open(&canonical).map_err(|source| QuantizationError::ArtifactIo {
        operation: "open checkpoint shard for provenance",
        path: canonical.clone(),
        source,
    })?;
    let opened_metadata = file
        .metadata()
        .map_err(|source| QuantizationError::ArtifactIo {
            operation: "inspect opened checkpoint shard for provenance",
            path: canonical.clone(),
            source,
        })?;
    #[cfg(unix)]
    if path_metadata.dev() != opened_metadata.dev() || path_metadata.ino() != opened_metadata.ino()
    {
        return Err(QuantizationError::InvalidProvenance {
            reason: format!(
                "checkpoint shard `{}` changed between path inspection and open",
                shard.relative_path
            ),
        });
    }
    if opened_metadata.len() != shard.file_bytes {
        return Err(QuantizationError::CheckpointProvenanceLengthMismatch {
            shard: shard.relative_path.clone(),
            expected: shard.file_bytes,
            actual: opened_metadata.len(),
        });
    }

    let mut hasher = Sha256::new();
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(PROVENANCE_HASH_BUFFER_BYTES)
        .map_err(|_| QuantizationError::AllocationFailed {
            context: "checkpoint provenance hash buffer",
            requested: PROVENANCE_HASH_BUFFER_BYTES,
        })?;
    buffer.resize(PROVENANCE_HASH_BUFFER_BYTES, 0);
    stats.peak_buffer_bytes = stats.peak_buffer_bytes.max(PROVENANCE_HASH_BUFFER_BYTES);
    let mut bytes_read = 0_u64;
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|source| QuantizationError::ArtifactIo {
                operation: "hash checkpoint shard provenance",
                path: canonical.clone(),
                source,
            })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        stats.bytes_hashed = stats.bytes_hashed.checked_add(count as u64).ok_or(
            QuantizationError::ByteLengthOverflow {
                context: "provenance checkpoint byte count",
            },
        )?;
        bytes_read =
            bytes_read
                .checked_add(count as u64)
                .ok_or(QuantizationError::ByteLengthOverflow {
                    context: "hashed checkpoint shard bytes",
                })?;
    }
    if bytes_read != shard.file_bytes {
        return Err(QuantizationError::CheckpointProvenanceLengthMismatch {
            shard: shard.relative_path.clone(),
            expected: shard.file_bytes,
            actual: bytes_read,
        });
    }
    stats.shards_hashed =
        stats
            .shards_hashed
            .checked_add(1)
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "provenance shard count",
            })?;
    Ok(BurnpackShardProvenance {
        file_bytes: shard.file_bytes,
        sha256: hex_digest(hasher.finalize()),
    })
}

/// Hashes one length-prefixed UTF-8 string.
fn hash_str(hasher: &mut Sha256, value: &str) {
    hash_len(hasher, value.len());
    hasher.update(value.as_bytes());
}

/// Hashes one length-prefixed tensor shape.
fn hash_shape(hasher: &mut Sha256, shape: &[u64]) {
    hash_len(hasher, shape.len());
    for dimension in shape {
        hasher.update(dimension.to_le_bytes());
    }
}

/// Hashes a collection length in a platform-independent width.
fn hash_len(hasher: &mut Sha256, value: usize) {
    hash_usize(hasher, value);
}

/// Hashes a host integer after widening to the v2 wire width.
fn hash_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

/// Encodes a digest without adding a separate hexadecimal dependency.
fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    /// Lowercase hexadecimal digits indexed by a four-bit nibble.
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = digest.as_ref();
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

/// Computes the digest of a fully materialized packed tensor payload.
pub(crate) fn packed_payload_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_digest(hasher.finalize())
}

/// Computes the writer-side digest without allocating a combined payload.
///
/// Scales are hashed in the same bounded little-endian chunks that the staging
/// writer emits, so the digest covers exactly the canonical payload bytes.
pub(crate) fn quantized_payload_sha256(packed_values: &[u8], scales: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(packed_values);
    let _ = super::reference::for_each_scale_chunk(scales, |chunk| {
        hasher.update(chunk);
        Ok::<(), std::convert::Infallible>(())
    });
    hex_digest(hasher.finalize())
}

/// Writer adapter that SHA-256-hashes exactly the bytes its inner writer accepts.
///
/// Staging a payload through this adapter computes its digest in the same pass
/// as the write, and the digest then describes the staged bytes themselves
/// rather than a separate re-encoding of the in-memory weight.
pub(crate) struct DigestingWriter<W> {
    /// Destination receiving the payload.
    inner: W,
    /// Running digest of every accepted byte, in order.
    hasher: Sha256,
}

impl<W: std::io::Write> DigestingWriter<W> {
    /// Wraps a destination with an empty running digest.
    pub(crate) fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    /// Returns the lowercase hexadecimal digest of all accepted bytes.
    pub(crate) fn finish(self) -> String {
        hex_digest(self.hasher.finalize())
    }
}

impl<W: std::io::Write> std::io::Write for DigestingWriter<W> {
    /// Forwards one write and hashes only the prefix the destination accepted.
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buffer)?;
        self.hasher.update(&buffer[..written]);
        Ok(written)
    }

    /// Flushes the destination; the digest holds no buffered bytes.
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Enforces canonical lowercase SHA-256 spelling.
pub(crate) fn validate_sha256(digest: &str, subject: &str) -> Result<()> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(QuantizationError::InvalidProvenance {
            reason: format!("{subject} SHA-256 must be 64 lowercase hexadecimal characters"),
        });
    }
    Ok(())
}

/// Detects portable Windows drive prefixes even when running on Unix.
fn has_windows_drive_prefix(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

#[cfg(test)]
#[path = "provenance_tests.rs"]
mod tests;
