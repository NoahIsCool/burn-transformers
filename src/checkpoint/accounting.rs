//! Provenance-bound exceptions for defective checkpoint index byte totals.
//!
//! SafeTensors shard indexes normally satisfy
//! `metadata.total_size = sum(physical tensor payload bytes)`, after applying
//! the explicitly selected index-alias convention. That equality remains the
//! unconditional default. A published checkpoint can only deviate through a
//! [`PinnedIndexTotalSizePolicy`] supplied by its architecture contract. The
//! policy binds the deviation to exact control-file hashes, source labels,
//! physical totals, and tensor descriptors; a numeric delta alone is never
//! sufficient.

use std::collections::{BTreeMap, BTreeSet};

use super::{CheckpointError, Result, TensorDType, TensorInventory};

/// Immutable identity of one small checkpoint control file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PinnedControlFile {
    /// Normalized path relative to the checkpoint root.
    pub(crate) relative_path: String,
    /// Exact byte length of the pinned file.
    pub(crate) bytes: u64,
    /// Lowercase hexadecimal SHA-256 over the exact file bytes.
    pub(crate) sha256: String,
}

impl PinnedControlFile {
    /// Creates one exact control-file identity.
    pub(crate) fn new(
        relative_path: impl Into<String>,
        bytes: u64,
        sha256: impl Into<String>,
    ) -> Self {
        Self {
            relative_path: relative_path.into(),
            bytes,
            sha256: sha256.into(),
        }
    }
}

/// Shape and dtype of one tensor implicated in an upstream accounting defect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PinnedAccountingTensor {
    /// Exact checkpoint tensor key.
    pub(crate) name: String,
    /// Exact scalar dtype stored by SafeTensors.
    pub(crate) dtype: TensorDType,
    /// Exact logical row-major shape.
    pub(crate) shape: Vec<u64>,
}

impl PinnedAccountingTensor {
    /// Creates one exact tensor descriptor.
    pub(crate) fn new(
        name: impl Into<String>,
        dtype: TensorDType,
        shape: impl Into<Vec<u64>>,
    ) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.into(),
        }
    }
}

/// Selects a family-owned tensor namespace with a decimal layer component.
///
/// For example, prefix `backbone.layers.` and suffix `.mixer.gate.weight`
/// select `backbone.layers.17.mixer.gate.weight` but neither a non-decimal
/// layer nor `mtp.layers.1.mixer.gate.weight`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecimalLayerTensorSelector {
    /// Bytes before the canonical unsigned decimal layer index.
    pub(crate) prefix: String,
    /// Bytes after the canonical unsigned decimal layer index.
    pub(crate) suffix: String,
}

impl DecimalLayerTensorSelector {
    /// Creates a selector for one exact layered tensor namespace.
    pub(crate) fn new(prefix: impl Into<String>, suffix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            suffix: suffix.into(),
        }
    }

    /// Returns whether `name` has a canonical decimal component between both anchors.
    fn matches(&self, name: &str) -> bool {
        let Some(layer) = name
            .strip_prefix(&self.prefix)
            .and_then(|rest| rest.strip_suffix(&self.suffix))
        else {
            return false;
        };
        !layer.is_empty()
            && layer.bytes().all(|byte| byte.is_ascii_digit())
            && (layer == "0" || !layer.starts_with('0'))
    }
}

/// Exact observations presented to a pinned total-size policy.
///
/// Discovery produces this evidence from bounded control-file reads and fully
/// validated SafeTensors headers. Keeping the validator independent of I/O
/// makes every provenance and inventory near miss directly testable.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IndexTotalSizeEvidence<'a> {
    /// Root-relative index selected for discovery.
    pub(crate) index_relative_path: &'a str,
    /// Exact bytes read and parsed as the selected index.
    pub(crate) index_bytes: u64,
    /// SHA-256 of the exact parsed index bytes.
    pub(crate) index_sha256: &'a str,
    /// Root-relative configuration file authenticated by the policy.
    pub(crate) config_relative_path: &'a str,
    /// Exact configuration bytes read for authentication.
    pub(crate) config_bytes: u64,
    /// SHA-256 of the authenticated configuration bytes.
    pub(crate) config_sha256: &'a str,
    /// Byte total declared by `metadata.total_size`.
    pub(crate) declared_total_size: u64,
    /// Physical byte total derived from all validated shard headers.
    pub(crate) physical_tensor_bytes: u64,
    /// Complete physical tensor inventory derived from the shard headers.
    pub(crate) tensors: &'a BTreeMap<String, TensorInventory>,
    /// Number of index-only aliases resolved during discovery.
    pub(crate) index_alias_count: usize,
}

/// One architecture-owned, immutable exception to index total-size equality.
///
/// The `supplied_*` labels come from caller-provided source metadata, while the
/// `expected_*` labels and every remaining field are pinned by architecture
/// code. Validation requires all fields simultaneously; this type deliberately
/// has no permissive delta-only constructor or wildcard tensor matcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PinnedIndexTotalSizePolicy {
    /// Repository label claimed by the local model source.
    pub(crate) supplied_repository: String,
    /// Immutable revision claimed by the local model source.
    pub(crate) supplied_revision: String,
    /// Repository label tied to the pinned hashes.
    pub(crate) expected_repository: String,
    /// Revision tied to the pinned hashes.
    pub(crate) expected_revision: String,
    /// Pinned configuration identity.
    pub(crate) config: PinnedControlFile,
    /// Pinned shard-index identity.
    pub(crate) index: PinnedControlFile,
    /// Exact physical tensor bytes derived from official shard headers.
    pub(crate) expected_physical_tensor_bytes: u64,
    /// Exact defective total stored in the official index.
    pub(crate) expected_declared_total_size: u64,
    /// Namespace whose complete observed set must equal `duplicated_tensors`.
    pub(crate) duplicated_tensor_selector: DecimalLayerTensorSelector,
    /// Exact physical tensors counted a second time by the defective index total.
    pub(crate) duplicated_tensors: Vec<PinnedAccountingTensor>,
    /// Related physical tensors explicitly proven not to contribute to the delta.
    pub(crate) excluded_tensors: Vec<PinnedAccountingTensor>,
}

impl PinnedIndexTotalSizePolicy {
    /// Validates all provenance, inventory, and checked-byte-accounting evidence.
    pub(crate) fn validate(&self, evidence: IndexTotalSizeEvidence<'_>) -> Result<()> {
        self.validate_source_and_controls(evidence)?;
        if evidence.index_alias_count != 0 {
            return policy_mismatch(format!(
                "pinned accounting requires zero index-only aliases, observed {}",
                evidence.index_alias_count
            ));
        }
        if evidence.physical_tensor_bytes != self.expected_physical_tensor_bytes {
            return policy_mismatch(format!(
                "physical tensor bytes are {}, expected {}",
                evidence.physical_tensor_bytes, self.expected_physical_tensor_bytes
            ));
        }
        if evidence.declared_total_size != self.expected_declared_total_size {
            return policy_mismatch(format!(
                "declared total size is {}, expected {}",
                evidence.declared_total_size, self.expected_declared_total_size
            ));
        }

        let expected_names = self.validate_policy_tensor_specs()?;
        let observed_names = evidence
            .tensors
            .keys()
            .filter(|name| self.duplicated_tensor_selector.matches(name))
            .cloned()
            .collect::<BTreeSet<_>>();
        if observed_names != expected_names {
            return policy_mismatch(format!(
                "accounted tensor-name set differs: observed {observed_names:?}, expected {expected_names:?}"
            ));
        }

        let mut duplicated_bytes = 0_u64;
        for expected in &self.duplicated_tensors {
            let observed = evidence.tensors.get(&expected.name).ok_or_else(|| {
                CheckpointError::IndexTotalSizePolicyMismatch {
                    reason: format!("accounted tensor `{}` is absent", expected.name),
                }
            })?;
            validate_tensor_descriptor(observed, expected, "accounted")?;
            duplicated_bytes = duplicated_bytes.checked_add(observed.byte_len).ok_or(
                CheckpointError::ArithmeticOverflow {
                    context: "summing pinned duplicate-accounting tensor bytes",
                },
            )?;
        }
        for excluded in &self.excluded_tensors {
            if expected_names.contains(&excluded.name) {
                return policy_mismatch(format!(
                    "excluded tensor `{}` is also listed as duplicate-counted",
                    excluded.name
                ));
            }
            let observed = evidence.tensors.get(&excluded.name).ok_or_else(|| {
                CheckpointError::IndexTotalSizePolicyMismatch {
                    reason: format!("excluded tensor `{}` is absent", excluded.name),
                }
            })?;
            validate_tensor_descriptor(observed, excluded, "excluded")?;
        }

        let reconciled = evidence
            .physical_tensor_bytes
            .checked_add(duplicated_bytes)
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "reconciling pinned checkpoint index total size",
            })?;
        if reconciled != evidence.declared_total_size {
            return policy_mismatch(format!(
                "physical bytes {} plus exact duplicate-counted bytes {duplicated_bytes} equal {reconciled}, not declared bytes {}",
                evidence.physical_tensor_bytes, evidence.declared_total_size
            ));
        }
        Ok(())
    }

    /// Validates source labels and exact configuration/index byte identities.
    fn validate_source_and_controls(&self, evidence: IndexTotalSizeEvidence<'_>) -> Result<()> {
        for (label, supplied, expected) in [
            (
                "repository",
                self.supplied_repository.as_str(),
                self.expected_repository.as_str(),
            ),
            (
                "revision",
                self.supplied_revision.as_str(),
                self.expected_revision.as_str(),
            ),
        ] {
            if supplied != expected {
                return policy_mismatch(format!(
                    "supplied {label} `{supplied}` does not match pinned `{expected}`"
                ));
            }
        }
        validate_control(
            "configuration",
            evidence.config_relative_path,
            evidence.config_bytes,
            evidence.config_sha256,
            &self.config,
        )?;
        validate_control(
            "index",
            evidence.index_relative_path,
            evidence.index_bytes,
            evidence.index_sha256,
            &self.index,
        )
    }

    /// Validates policy construction and returns its unique accounted name set.
    fn validate_policy_tensor_specs(&self) -> Result<BTreeSet<String>> {
        if self.duplicated_tensors.is_empty() {
            return policy_mismatch("duplicate-counted tensor set is empty".into());
        }
        let mut names = BTreeSet::new();
        for tensor in &self.duplicated_tensors {
            if !self.duplicated_tensor_selector.matches(&tensor.name) {
                return policy_mismatch(format!(
                    "duplicate-counted tensor `{}` is outside its exact layered namespace",
                    tensor.name
                ));
            }
            if !names.insert(tensor.name.clone()) {
                return policy_mismatch(format!(
                    "duplicate-counted tensor `{}` appears more than once",
                    tensor.name
                ));
            }
        }
        Ok(names)
    }
}

/// Requires one observed tensor to match its pinned dtype, shape, and byte count.
fn validate_tensor_descriptor(
    observed: &TensorInventory,
    expected: &PinnedAccountingTensor,
    role: &'static str,
) -> Result<()> {
    if observed.dtype != expected.dtype || observed.shape != expected.shape {
        return policy_mismatch(format!(
            "{role} tensor `{}` has {:?} {:?}, expected {:?} {:?}",
            expected.name, observed.dtype, observed.shape, expected.dtype, expected.shape
        ));
    }
    let elements = super::layout::element_count(&expected.shape).ok_or(
        CheckpointError::ArithmeticOverflow {
            context: "computing pinned accounting tensor elements",
        },
    )?;
    let expected_bytes = expected.dtype.storage_bytes(elements)?;
    if observed.byte_len != expected_bytes {
        return policy_mismatch(format!(
            "{role} tensor `{}` stores {} bytes, expected {expected_bytes}",
            expected.name, observed.byte_len
        ));
    }
    Ok(())
}

/// Requires one observed control-file identity to match its pinned counterpart.
fn validate_control(
    role: &'static str,
    relative_path: &str,
    bytes: u64,
    sha256: &str,
    expected: &PinnedControlFile,
) -> Result<()> {
    if relative_path != expected.relative_path
        || bytes != expected.bytes
        || sha256 != expected.sha256
    {
        return policy_mismatch(format!(
            "{role} identity is path `{relative_path}`, {bytes} bytes, SHA-256 `{sha256}`; expected path `{}`, {} bytes, SHA-256 `{}`",
            expected.relative_path, expected.bytes, expected.sha256
        ));
    }
    Ok(())
}

/// Builds a contextual fail-closed policy error.
fn policy_mismatch<T>(reason: String) -> Result<T> {
    Err(CheckpointError::IndexTotalSizePolicyMismatch { reason })
}
