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
pub(crate) fn quantized_payload_sha256(packed_values: &[u8], scales: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(packed_values);
    for scale in scales {
        hasher.update(scale.to_le_bytes());
    }
    hex_digest(hasher.finalize())
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
mod tests {
    use std::io::Write;

    use super::*;
    use crate::checkpoint::{
        ExpectedTensor, LoadOptions, LoadReport, MappingPlan, MemoryEstimate,
        PlannedTensorEquivalence, TensorDType, build_load_plan, discover_checkpoint,
    };

    /// Constructs a plan exercising source, transform, equivalence, and alias domains.
    fn fixture_plan() -> LoadPlan {
        let owner = PlannedSource {
            name: "external.weight".into(),
            shard: "model-00001-of-00002.safetensors".into(),
            shape: vec![2, 4],
            dtype: TensorDType::BF16,
            data_offsets: [128, 144],
        };
        let redundant = PlannedSource {
            name: "external.tied_weight".into(),
            data_offsets: [144, 160],
            ..owner.clone()
        };
        LoadPlan {
            equivalences: vec![PlannedTensorEquivalence {
                alias: redundant,
                owner: owner.clone(),
            }],
            tensors: vec![PlannedTensor {
                target: "model.projection.weight".into(),
                sources: vec![owner],
                transform: PlannedTransform::Transpose2d,
                shape: vec![4, 2],
                dtype: TensorDType::F16,
            }],
            aliases: vec![PlannedTensor {
                target: "lm_head.weight".into(),
                sources: Vec::new(),
                transform: PlannedTransform::Alias {
                    source_target: "model.projection.weight".into(),
                },
                shape: vec![4, 2],
                dtype: TensorDType::F16,
            }],
            report: LoadReport::default(),
            memory: MemoryEstimate::default(),
        }
    }

    /// Canonical plan hashing is stable and covers mapping semantics, not reports.
    #[test]
    fn canonical_plan_digest_is_stable_and_semantically_sensitive() {
        let plan = fixture_plan();
        let digest = load_plan_sha256(&plan);
        assert_eq!(
            digest,
            "2ff631fc5de762e8499277fe3cfd817611d5091b8d601dddd87e74689426c126"
        );
        assert_eq!(load_plan_sha256(&plan.clone()), digest);

        let mut changed = plan.clone();
        changed.tensors[0].sources[0].data_offsets[0] += 1;
        assert_ne!(load_plan_sha256(&changed), digest);

        let mut changed = plan.clone();
        changed.tensors[0].transform = PlannedTransform::Identity;
        assert_ne!(load_plan_sha256(&changed), digest);

        let mut changed = plan.clone();
        changed.aliases[0].target = "different.weight".into();
        assert_ne!(load_plan_sha256(&changed), digest);

        let mut report_only = plan.clone();
        report_only.report.applied.push("diagnostic-only".into());
        assert_eq!(load_plan_sha256(&report_only), digest);
    }

    /// V2 digest spelling is canonical and rejects missing or malformed source identity.
    #[test]
    fn provenance_metadata_rejects_noncanonical_digests_and_empty_shards() {
        let mut provenance = BurnpackProvenance {
            load_plan_sha256: "0".repeat(64),
            checkpoint_shards: BTreeMap::from([(
                "model.safetensors".into(),
                BurnpackShardProvenance {
                    file_bytes: 128,
                    sha256: "a".repeat(64),
                },
            )]),
        };
        provenance.validate().unwrap();

        provenance.load_plan_sha256 = "A".repeat(64);
        assert!(matches!(
            provenance.validate(),
            Err(QuantizationError::InvalidProvenance { .. })
        ));
        provenance.load_plan_sha256 = "0".repeat(64);
        provenance.checkpoint_shards.clear();
        assert!(matches!(
            provenance.validate(),
            Err(QuantizationError::InvalidProvenance { .. })
        ));

        provenance.checkpoint_shards.insert(
            "../model.safetensors".into(),
            BurnpackShardProvenance {
                file_bytes: 128,
                sha256: "a".repeat(64),
            },
        );
        assert!(matches!(
            provenance.validate(),
            Err(QuantizationError::InvalidProvenance { .. })
        ));

        for invalid_path in [
            "nested//model.safetensors",
            "nested\\model.safetensors",
            "C:model.safetensors",
            "model\0.safetensors",
        ] {
            provenance.checkpoint_shards.clear();
            provenance.checkpoint_shards.insert(
                invalid_path.into(),
                BurnpackShardProvenance {
                    file_bytes: 128,
                    sha256: "a".repeat(64),
                },
            );
            assert!(matches!(
                provenance.validate(),
                Err(QuantizationError::InvalidProvenance { .. })
            ));
        }
    }

    /// Writes a real shard large enough, when requested, to span hash-buffer reads.
    fn local_fixture(elements: usize) -> (tempfile::TempDir, CheckpointManifest, LoadPlan) {
        let directory = tempfile::tempdir().unwrap();
        let header = serde_json::to_vec(&serde_json::json!({
            "weight": {"dtype": "F32", "shape": [elements], "data_offsets": [0, elements * 4]}
        }))
        .unwrap();
        let mut file = File::create(directory.path().join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        let bytes = (0..elements)
            .flat_map(|_| 1.0_f32.to_le_bytes())
            .collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        let manifest = discover_checkpoint(directory.path(), &LoadOptions::default()).unwrap();
        let expected = [ExpectedTensor::new("weight", [elements as u64])];
        let plan = build_load_plan(
            &manifest,
            &expected,
            &MappingPlan::identity(&expected),
            &LoadOptions::default(),
        )
        .unwrap();
        (directory, manifest, plan)
    }

    /// Covers every plan field and the slice/concatenation tags absent in the golden.
    #[test]
    fn canonical_digest_binds_every_executable_field_but_not_memory_estimates() {
        let plan = fixture_plan();
        let base = load_plan_sha256(&plan);
        for field in [
            "target",
            "source_name",
            "source_shard",
            "source_shape",
            "source_dtype",
            "source_start",
            "source_end",
            "shape",
            "dtype",
            "source_count",
            "tensor_count",
            "alias_owner",
            "equivalence_owner",
            "equivalence_order",
        ] {
            let mut changed = plan.clone();
            match field {
                "target" => changed.tensors[0].target.push('x'),
                "source_name" => changed.tensors[0].sources[0].name.push('x'),
                "source_shard" => changed.tensors[0].sources[0].shard.push('x'),
                "source_shape" => changed.tensors[0].sources[0].shape.reverse(),
                "source_dtype" => changed.tensors[0].sources[0].dtype = TensorDType::F16,
                "source_start" => changed.tensors[0].sources[0].data_offsets[0] += 1,
                "source_end" => changed.tensors[0].sources[0].data_offsets[1] += 1,
                "shape" => changed.tensors[0].shape.reverse(),
                "dtype" => changed.tensors[0].dtype = TensorDType::BF16,
                "source_count" => {
                    let source = changed.tensors[0].sources[0].clone();
                    changed.tensors[0].sources.push(source);
                }
                "tensor_count" => changed.tensors.push(changed.tensors[0].clone()),
                "alias_owner" => {
                    changed.aliases[0].transform = PlannedTransform::Alias {
                        source_target: "different".into(),
                    }
                }
                "equivalence_owner" => changed.equivalences[0].owner.name.push('x'),
                "equivalence_order" => {
                    let item = &mut changed.equivalences[0];
                    std::mem::swap(&mut item.owner, &mut item.alias);
                }
                _ => unreachable!(),
            }
            assert_ne!(load_plan_sha256(&changed), base, "field {field}");
        }
        let mut digests = std::collections::BTreeSet::new();
        for transform in [
            PlannedTransform::Identity,
            PlannedTransform::Transpose2d,
            PlannedTransform::Slice {
                axis: 0,
                start: 0,
                length: 1,
            },
            PlannedTransform::Slice {
                axis: 1,
                start: 0,
                length: 1,
            },
            PlannedTransform::Slice {
                axis: 0,
                start: 1,
                length: 1,
            },
            PlannedTransform::Slice {
                axis: 0,
                start: 0,
                length: 2,
            },
            PlannedTransform::Concatenate { axis: 0 },
            PlannedTransform::Concatenate { axis: 1 },
        ] {
            let mut changed = plan.clone();
            changed.tensors[0].transform = transform;
            assert!(digests.insert(load_plan_sha256(&changed)));
        }
        let mut diagnostic = plan.clone();
        diagnostic.memory.peak_host_staging_bytes = u64::MAX;
        diagnostic.report.unexpected.push("diagnostic".into());
        assert_eq!(load_plan_sha256(&diagnostic), base);

        // Length prefixes separate fields that would collide under concatenation.
        let mut left = plan.clone();
        left.tensors[0].sources[0].name = "ab".into();
        left.tensors[0].sources[0].shard = "c".into();
        let mut right = left.clone();
        right.tensors[0].sources[0].name = "a".into();
        right.tensors[0].sources[0].shard = "bc".into();
        assert_ne!(load_plan_sha256(&left), load_plan_sha256(&right));
    }

    /// Rejects lexical dot segments even where Path::components normalizes them.
    #[test]
    fn provenance_requires_lexically_normalized_paths_nonzero_lengths_and_shard_hashes() {
        let mut provenance = BurnpackProvenance {
            load_plan_sha256: "a".repeat(64),
            checkpoint_shards: BTreeMap::from([(
                "nested/model.safetensors".into(),
                BurnpackShardProvenance {
                    file_bytes: 8,
                    sha256: "0".repeat(64),
                },
            )]),
        };
        provenance.validate().unwrap();
        let shard = provenance
            .checkpoint_shards
            .values()
            .next()
            .unwrap()
            .clone();
        for path in [
            "",
            "/model.safetensors",
            "./model.safetensors",
            "nested/./model.safetensors",
            "nested/../model.safetensors",
            "nested/model.safetensors/",
            "model.bin",
        ] {
            provenance.checkpoint_shards = BTreeMap::from([(path.into(), shard.clone())]);
            assert!(
                matches!(provenance.validate(), Err(QuantizationError::InvalidProvenance { reason })
                if reason.contains("normalized relative")),
                "accepted {path:?}"
            );
        }
        provenance.checkpoint_shards = BTreeMap::from([("model.safetensors".into(), shard)]);
        provenance
            .checkpoint_shards
            .get_mut("model.safetensors")
            .unwrap()
            .file_bytes = 0;
        assert!(
            matches!(provenance.validate(), Err(QuantizationError::InvalidProvenance { reason }) if reason.contains("zero length"))
        );
        provenance
            .checkpoint_shards
            .get_mut("model.safetensors")
            .unwrap()
            .file_bytes = 8;
        provenance
            .checkpoint_shards
            .get_mut("model.safetensors")
            .unwrap()
            .sha256 = "z".repeat(64);
        assert!(
            matches!(provenance.validate(), Err(QuantizationError::InvalidProvenance { reason }) if reason.starts_with("checkpoint shard SHA-256"))
        );
    }

    /// Compares a multi-buffer streaming hash with a one-shot digest and re-verifies.
    #[test]
    fn streamed_shard_hash_matches_complete_file_bytes_and_exact_plan() {
        let (_directory, manifest, plan) = local_fixture(PROVENANCE_HASH_BUFFER_BYTES / 4 + 7);
        let (provenance, stats) = hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
        let bytes = fs::read(&manifest.shards[0].path).unwrap();
        assert_eq!(
            provenance.checkpoint_shards["model.safetensors"].sha256,
            hex_digest(Sha256::digest(&bytes))
        );
        assert_eq!(stats.bytes_hashed, bytes.len() as u64);
        assert_eq!(stats.shards_hashed, 1);
        assert_eq!(stats.passes_completed, 1);
        assert_eq!(stats.peak_buffer_bytes, PROVENANCE_HASH_BUFFER_BYTES);
        assert_eq!(
            verify_local_checkpoint_provenance(&manifest, &plan, &provenance).unwrap(),
            stats
        );
        let mut changed = plan.clone();
        changed.tensors[0].target.push('x');
        assert!(matches!(verify_load_plan_provenance(&changed, &provenance),
            Err(QuantizationError::LoadPlanProvenanceMismatch { expected, actual })
            if expected == provenance.load_plan_sha256 && actual == load_plan_sha256(&changed)));
        let mut changed_bytes = bytes;
        *changed_bytes.last_mut().unwrap() ^= 1;
        fs::write(&manifest.shards[0].path, changed_bytes).unwrap();
        assert!(
            matches!(verify_local_checkpoint_provenance(&manifest, &plan, &provenance),
            Err(QuantizationError::CheckpointProvenanceDigestMismatch { shard, .. }) if shard == "model.safetensors")
        );
    }

    /// Preserves completed-shard counters when a later file cannot be opened.
    #[test]
    fn audited_hash_failure_preserves_prefix_and_rejects_duplicate_shards() {
        let (_directory, mut manifest, plan) = local_fixture(2);
        let mut absent = manifest.shards[0].clone();
        absent.relative_path = "missing.safetensors".into();
        absent.path = manifest.root.join(&absent.relative_path);
        manifest.shards.push(absent);
        let failure = hash_local_checkpoint_provenance_audited(&manifest, &plan).unwrap_err();
        assert!(matches!(
            failure.error,
            QuantizationError::ArtifactIo {
                operation: "canonicalize checkpoint shard for provenance",
                ..
            }
        ));
        assert_eq!(failure.stats.shards_hashed, 1);
        assert_eq!(failure.stats.bytes_hashed, manifest.shards[0].file_bytes);
        assert_eq!(failure.stats.passes_completed, 0);
        assert_eq!(
            failure.stats.peak_buffer_bytes,
            PROVENANCE_HASH_BUFFER_BYTES
        );

        manifest.shards[1] = manifest.shards[0].clone();
        let failure = hash_local_checkpoint_provenance_audited(&manifest, &plan).unwrap_err();
        assert!(
            matches!(failure.error, QuantizationError::InvalidProvenance { reason } if reason.contains("repeats shard"))
        );
        assert_eq!(failure.stats.shards_hashed, 2);
        assert_eq!(
            failure.stats.bytes_hashed,
            2 * manifest.shards[0].file_bytes
        );
        assert_eq!(failure.stats.passes_completed, 0);
        manifest.shards.clear();
        let failure = hash_local_checkpoint_provenance_audited(&manifest, &plan).unwrap_err();
        assert!(
            matches!(failure.error, QuantizationError::InvalidProvenance { reason } if reason.contains("digest map is empty"))
        );
        assert_eq!(failure.stats.bytes_hashed, 0);
        assert_eq!(failure.stats.peak_buffer_bytes, 0);
    }

    /// Rejects inventory and filesystem changes before allocating a hash buffer.
    #[test]
    fn shard_preflight_and_hashing_fail_closed_on_changed_files() {
        for case in ["missing", "directory", "length", "root"] {
            let (_directory, mut manifest, plan) = local_fixture(2);
            let (expected, _) = hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
            let other = tempfile::tempdir().unwrap();
            let path = manifest.shards[0].path.clone();
            match case {
                "missing" => fs::remove_file(&path).unwrap(),
                "directory" => {
                    fs::remove_file(&path).unwrap();
                    fs::create_dir(&path).unwrap();
                }
                "length" => {
                    let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
                    file.set_len(manifest.shards[0].file_bytes + 1).unwrap();
                }
                "root" => manifest.root = fs::canonicalize(other.path()).unwrap(),
                _ => unreachable!(),
            }
            let failure = hash_local_checkpoint_provenance_audited(&manifest, &plan).unwrap_err();
            assert_eq!(failure.stats, ProvenanceHashStats::default());
            match case {
                "missing" => assert!(matches!(
                    failure.error,
                    QuantizationError::ArtifactIo { .. }
                )),
                "length" => assert!(matches!(
                    failure.error,
                    QuantizationError::CheckpointProvenanceLengthMismatch { .. }
                )),
                _ => assert!(matches!(
                    failure.error,
                    QuantizationError::InvalidProvenance { .. }
                )),
            }
            if case != "root" {
                assert!(preflight_local_checkpoint_provenance(&manifest, &expected).is_err());
            }
        }
    }

    /// Both preflight and hashing refuse replacement symlinks, even within the root.
    #[cfg(unix)]
    #[test]
    fn provenance_rejects_symlink_replacement_before_hashing() {
        let (_directory, manifest, plan) = local_fixture(2);
        let (expected, _) = hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
        let shard = &manifest.shards[0];
        let renamed = manifest.root.join("replacement.safetensors");
        fs::rename(&shard.path, &renamed).unwrap();
        std::os::unix::fs::symlink(&renamed, &shard.path).unwrap();
        assert!(
            matches!(preflight_local_checkpoint_provenance(&manifest, &expected),
            Err(QuantizationError::InvalidProvenance { reason }) if reason.contains("regular non-symlink"))
        );
        let failure = hash_local_checkpoint_provenance_audited(&manifest, &plan).unwrap_err();
        assert!(
            matches!(failure.error, QuantizationError::InvalidProvenance { reason } if reason.contains("discovered canonical path"))
        );
        assert_eq!(failure.stats, ProvenanceHashStats::default());
    }

    /// Distinguishes inventory cardinality, membership, and byte-length errors.
    #[test]
    fn provenance_inventory_comparisons_validate_counts_keys_and_lengths() {
        let (_directory, manifest, plan) = local_fixture(2);
        let (expected, _) = hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
        for case in ["count", "key", "length", "duplicate"] {
            let mut manifest = manifest.clone();
            let mut expected = expected.clone();
            match case {
                "count" => manifest.shards.clear(),
                "key" => manifest.shards[0].relative_path = "unknown.safetensors".into(),
                "length" => manifest.shards[0].file_bytes += 1,
                "duplicate" => {
                    manifest.shards.push(manifest.shards[0].clone());
                    expected.checkpoint_shards.insert(
                        "extra.safetensors".into(),
                        expected.checkpoint_shards["model.safetensors"].clone(),
                    );
                }
                _ => unreachable!(),
            }
            let error = preflight_local_checkpoint_provenance(&manifest, &expected).unwrap_err();
            if case == "length" {
                assert!(matches!(
                    error,
                    QuantizationError::CheckpointProvenanceLengthMismatch { .. }
                ));
            } else {
                let expected = match case {
                    "count" => "shard count",
                    "key" => "absent from sidecar",
                    _ => "repeats shard",
                };
                assert!(
                    matches!(error, QuantizationError::InvalidProvenance { reason } if reason.contains(expected))
                );
            }
        }
        for case in ["count", "key", "length", "digest"] {
            let mut actual = expected.clone();
            match case {
                "count" => actual.checkpoint_shards.clear(),
                "key" => {
                    let shard = actual
                        .checkpoint_shards
                        .remove("model.safetensors")
                        .unwrap();
                    actual
                        .checkpoint_shards
                        .insert("other.safetensors".into(), shard);
                }
                "length" => {
                    actual
                        .checkpoint_shards
                        .get_mut("model.safetensors")
                        .unwrap()
                        .file_bytes += 1
                }
                "digest" => {
                    actual
                        .checkpoint_shards
                        .get_mut("model.safetensors")
                        .unwrap()
                        .sha256 = "0".repeat(64)
                }
                _ => unreachable!(),
            }
            let error = compare_checkpoint_provenance(&actual, &expected).unwrap_err();
            match case {
                "length" => assert!(matches!(
                    error,
                    QuantizationError::CheckpointProvenanceLengthMismatch { .. }
                )),
                "digest" => assert!(matches!(
                    error,
                    QuantizationError::CheckpointProvenanceDigestMismatch { .. }
                )),
                _ => assert!(matches!(error, QuantizationError::InvalidProvenance { .. })),
            }
        }
    }

    /// Hashes scale wire bits identically without allocating the combined payload.
    #[test]
    fn packed_scale_hashing_and_two_pass_statistics_preserve_exact_boundaries() {
        assert_eq!(
            packed_payload_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let values = [0x12, 0xff, 0];
        let scales = [1.0_f32, -0.0, f32::from_bits(0x7fc0_0012)];
        let combined = values
            .into_iter()
            .chain(scales.into_iter().flat_map(f32::to_le_bytes))
            .collect::<Vec<_>>();
        assert_eq!(
            quantized_payload_sha256(&values, &scales),
            packed_payload_sha256(&combined)
        );
        let stats = ProvenanceExecutionStats {
            pre_execution: Some(ProvenanceHashStats {
                bytes_hashed: u64::MAX,
                peak_buffer_bytes: 16,
                ..ProvenanceHashStats::default()
            }),
            post_execution: Some(ProvenanceHashStats {
                bytes_hashed: u64::MAX,
                peak_buffer_bytes: 32,
                ..ProvenanceHashStats::default()
            }),
        };
        assert_eq!(stats.bytes_hashed(), 2 * u128::from(u64::MAX));
        assert_eq!(stats.peak_buffer_bytes(), 32);
        assert_eq!(ProvenanceExecutionStats::default().bytes_hashed(), 0);
        assert_eq!(ProvenanceExecutionStats::default().peak_buffer_bytes(), 0);
    }
}
