//! Explicit checkpoint-to-model key and layout mapping.
//!
//! A mapping plan is intentionally data, not filename-dependent string magic in a
//! generic loader. Each architecture declares how its external checkpoint tensors
//! produce internal parameters. This makes fused projections, transposes, aliases,
//! and ignored multimodal towers visible in validation reports and unit tests.
//!
//! Tensor axes are explicit: a matrix transpose exchanges its two dimensions;
//! split ranges partition a named axis; concatenation sums that axis while all
//! others match. Shape products and byte totals use checked integer arithmetic.
//! These operations are planned before materialization, so a malformed mapping
//! cannot allocate or mutate model parameters. Tied aliases resolve to one owner,
//! while dtype conversion follows each architecture's precision contract.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{
    AliasRecord, ArtifactSource, CheckpointEquivalenceRecord, CheckpointError, CheckpointManifest,
    DTypeConversion, DTypePolicy, IndexAliasSizeAccounting, IndexTensorAlias, LoadOptions,
    LoadReport, MemoryEstimate, Result, TensorDType, TensorInventory, TransformRecord,
};

// Declarative architecture mapping: names, axes, dtypes, and alias topology.

/// Exact or prefix-based exception for a checkpoint key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum KeyMatcher {
    /// Matches exactly one tensor name.
    Exact(
        /// Complete tensor key to match.
        String,
    ),
    /// Matches all tensor names beginning with a nonempty prefix.
    Prefix(
        /// Nonempty leading substring shared by permitted keys.
        String,
    ),
}

impl KeyMatcher {
    /// Creates an exact key matcher.
    #[must_use]
    pub fn exact(value: impl Into<String>) -> Self {
        Self::Exact(value.into())
    }

    /// Creates a prefix key matcher.
    #[must_use]
    pub fn prefix(value: impl Into<String>) -> Self {
        Self::Prefix(value.into())
    }

    /// Returns whether `key` is selected by this matcher.
    #[must_use]
    pub fn matches(&self, key: &str) -> bool {
        match self {
            Self::Exact(value) => key == value,
            Self::Prefix(value) => key.starts_with(value),
        }
    }

    /// Rejects empty matchers so an accidental empty prefix cannot exempt every key.
    fn validate(&self) -> Result<()> {
        let value = match self {
            Self::Exact(value) | Self::Prefix(value) => value,
        };
        if value.is_empty() {
            return Err(CheckpointError::InvalidMapping {
                message: "empty key matchers are forbidden; allowlists must be explicit".into(),
            });
        }
        Ok(())
    }
}

/// Shape and optional dtype expected by one model parameter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedTensor {
    /// Internal model parameter name.
    pub name: String,
    /// Required dimensions.
    pub shape: Vec<u64>,
    /// Required materialized dtype, or `None` when the model is dtype-generic.
    pub dtype: Option<TensorDType>,
}

impl ExpectedTensor {
    /// Creates a dtype-generic expected tensor.
    #[must_use]
    pub fn new(name: impl Into<String>, shape: impl Into<Vec<u64>>) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
            dtype: None,
        }
    }

    /// Requires a concrete materialized dtype.
    #[must_use]
    pub fn with_dtype(mut self, dtype: TensorDType) -> Self {
        self.dtype = Some(dtype);
        self
    }
}

/// A unary layout operation applied to one source tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TensorTransform {
    /// Preserve tensor element order and shape.
    Identity,
    /// Swap the two axes of a rank-two matrix.
    Transpose2d,
}

/// One explicit checkpoint-to-model mapping operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TensorOperation {
    /// Copy or transpose a single source into one target.
    Map {
        /// Checkpoint tensor name.
        source: String,
        /// Model parameter name.
        target: String,
        /// Layout transformation.
        transform: TensorTransform,
    },
    /// Split one source axis into consecutive, exhaustive target sections.
    Split {
        /// Fused checkpoint tensor name.
        source: String,
        /// Axis along which sections are taken.
        axis: usize,
        /// Ordered `(target, length)` sections. Lengths must sum to the source axis.
        targets: Vec<(String, u64)>,
    },
    /// Concatenate sources into one target along an axis.
    Concatenate {
        /// Ordered checkpoint tensor names.
        sources: Vec<String>,
        /// Model parameter name.
        target: String,
        /// Concatenation axis.
        axis: usize,
    },
    /// Reuse an already produced target parameter without allocating new storage.
    Alias {
        /// Existing target, which may itself be an alias.
        source_target: String,
        /// New model parameter alias.
        target: String,
    },
}

/// Architecture-owned sequence of explicit mapping operations.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MappingPlan {
    /// Ordered transformations and aliases requested by the architecture.
    operations: Vec<TensorOperation>,
}

impl MappingPlan {
    /// Creates an empty mapping plan.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all operations in declaration order.
    #[must_use]
    pub fn operations(&self) -> &[TensorOperation] {
        &self.operations
    }

    /// Appends a fully specified operation.
    ///
    /// Structural validation remains deferred until [`build_load_plan`] so plans
    /// can be deserialized or generated before a checkpoint is discovered.
    pub fn push(&mut self, operation: TensorOperation) -> &mut Self {
        self.operations.push(operation);
        self
    }

    /// Adds a direct identity mapping, including a pure key rename.
    pub fn map(&mut self, source: impl Into<String>, target: impl Into<String>) -> &mut Self {
        self.operations.push(TensorOperation::Map {
            source: source.into(),
            target: target.into(),
            transform: TensorTransform::Identity,
        });
        self
    }

    /// Adds a rank-two transpose mapping.
    pub fn transpose_2d(
        &mut self,
        source: impl Into<String>,
        target: impl Into<String>,
    ) -> &mut Self {
        self.operations.push(TensorOperation::Map {
            source: source.into(),
            target: target.into(),
            transform: TensorTransform::Transpose2d,
        });
        self
    }

    /// Adds an exhaustive split of one fused source tensor.
    pub fn split(
        &mut self,
        source: impl Into<String>,
        axis: usize,
        targets: impl IntoIterator<Item = (String, u64)>,
    ) -> &mut Self {
        self.operations.push(TensorOperation::Split {
            source: source.into(),
            axis,
            targets: targets.into_iter().collect(),
        });
        self
    }

    /// Adds a concatenation of multiple source tensors.
    pub fn concatenate(
        &mut self,
        sources: impl IntoIterator<Item = String>,
        target: impl Into<String>,
        axis: usize,
    ) -> &mut Self {
        self.operations.push(TensorOperation::Concatenate {
            sources: sources.into_iter().collect(),
            target: target.into(),
            axis,
        });
        self
    }

    /// Adds a tied target alias.
    pub fn alias(
        &mut self,
        source_target: impl Into<String>,
        target: impl Into<String>,
    ) -> &mut Self {
        self.operations.push(TensorOperation::Alias {
            source_target: source_target.into(),
            target: target.into(),
        });
        self
    }

    /// Creates identity operations for every expected tensor.
    #[must_use]
    pub fn identity(expected: &[ExpectedTensor]) -> Self {
        let mut plan = Self::new();
        for tensor in expected {
            plan.map(tensor.name.clone(), tensor.name.clone());
        }
        plan
    }
}

/// One source tensor and its validated byte location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedSource {
    /// Checkpoint tensor name.
    pub name: String,
    /// Root-relative shard filename.
    pub shard: String,
    /// Tensor dimensions.
    pub shape: Vec<u64>,
    /// Stored dtype.
    pub dtype: TensorDType,
    /// Half-open range relative to the shard data section.
    pub data_offsets: [u64; 2],
}

impl From<&TensorInventory> for PlannedSource {
    /// Copies a discovered tensor descriptor without changing its axes or byte range.
    fn from(tensor: &TensorInventory) -> Self {
        Self {
            name: tensor.name.clone(),
            shard: tensor.shard.clone(),
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
            data_offsets: tensor.data_offsets,
        }
    }
}

/// Concrete transformation used by the streaming pass-two loader.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlannedTransform {
    /// Direct tensor copy or key rename.
    Identity,
    /// Rank-two transpose.
    Transpose2d,
    /// Consecutive slice from a fused source.
    Slice {
        /// Source axis.
        axis: usize,
        /// Starting element index on that axis.
        start: u64,
        /// Number of elements on that axis.
        length: u64,
    },
    /// Concatenation of the listed sources.
    Concatenate {
        /// Target axis.
        axis: usize,
    },
    /// Storage alias of another target.
    Alias {
        /// Ultimate non-aliased target that owns storage.
        source_target: String,
    },
}

/// Fully validated work item for pass-two parameter loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTensor {
    /// Internal model parameter name.
    pub target: String,
    /// Source tensors; empty only for aliases.
    pub sources: Vec<PlannedSource>,
    /// Layout or alias operation.
    pub transform: PlannedTransform,
    /// Produced tensor dimensions.
    pub shape: Vec<u64>,
    /// Produced tensor dtype after policy conversion.
    pub dtype: TensorDType,
}

/// Two physical source descriptors requiring exact raw-payload equivalence.
///
/// These work items are executed before the first [`crate::checkpoint::TensorSink`]
/// mutation. The alias payload is redundant model state, while `owner` is the
/// canonical checkpoint tensor consumed by the architecture mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTensorEquivalence {
    /// Redundant physical checkpoint tensor.
    pub alias: PlannedSource,
    /// Canonical physical checkpoint tensor.
    pub owner: PlannedSource,
}

/// Strictly validated second-pass load plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadPlan {
    /// Physical checkpoint aliases verified before any parameter is installed.
    pub equivalences: Vec<PlannedTensorEquivalence>,
    /// Non-alias mappings in canonical target-name order.
    ///
    /// The runtime derives a separate source-location execution schedule without
    /// mutating this vector, so inspection and plan comparisons remain deterministic.
    pub tensors: Vec<PlannedTensor>,
    /// Alias operations, topologically resolved to storage-owning targets.
    pub aliases: Vec<PlannedTensor>,
    /// Deterministic pre-application report.
    pub report: LoadReport,
    /// Conservative memory requirements.
    pub memory: MemoryEstimate,
}

impl LoadPlan {
    /// Number of logical model parameters represented by this plan.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.tensors.len() + self.aliases.len()
    }
}

/// Validates architecture expectations against a checkpoint and compiles pass two.
///
/// Every observed tensor must be consumed once or explicitly allowlisted. Every
/// expected model parameter must be produced once or explicitly allowlisted.
pub fn build_load_plan(
    manifest: &CheckpointManifest,
    expected: &[ExpectedTensor],
    mappings: &MappingPlan,
    options: &LoadOptions,
) -> Result<LoadPlan> {
    build_load_plan_with_policy(manifest, expected, mappings, options, None)
}

/// Builds a plan with one architecture-owned, provenance-bound size policy.
pub(crate) fn build_load_plan_with_index_total_size_policy(
    manifest: &CheckpointManifest,
    expected: &[ExpectedTensor],
    mappings: &MappingPlan,
    options: &LoadOptions,
    policy: &super::accounting::PinnedIndexTotalSizePolicy,
) -> Result<LoadPlan> {
    build_load_plan_with_policy(manifest, expected, mappings, options, Some(policy))
}

/// Shared planner implementation with an optional pinned accounting policy.
fn build_load_plan_with_policy(
    manifest: &CheckpointManifest,
    expected: &[ExpectedTensor],
    mappings: &MappingPlan,
    options: &LoadOptions,
    policy: Option<&super::accounting::PinnedIndexTotalSizePolicy>,
) -> Result<LoadPlan> {
    validate_matchers(&options.allow_missing)?;
    validate_matchers(&options.allow_unexpected)?;
    validate_manifest_index_provenance(manifest, options, policy)?;
    if matches!(options.dtype_policy, DTypePolicy::Cast(dtype) if !dtype.is_float()) {
        return Err(CheckpointError::InvalidMapping {
            message: "DTypePolicy::Cast requires a real floating-point target dtype".into(),
        });
    }

    let expected = expected_inventory(expected)?;
    let mut consumed = BTreeSet::new();
    let mut produced = BTreeSet::new();
    let mut primary = Vec::new();
    let mut alias_specs = BTreeMap::new();

    for operation in mappings.operations() {
        match operation {
            TensorOperation::Map {
                source,
                target,
                transform,
            } => {
                consume(&mut consumed, source)?;
                let source_tensor = source_tensor(manifest, source)?;
                let shape = match transform {
                    TensorTransform::Identity => source_tensor.shape.clone(),
                    TensorTransform::Transpose2d => {
                        if source_tensor.shape.len() != 2 {
                            return Err(CheckpointError::InvalidMapping {
                                message: format!(
                                    "transpose source `{source}` has rank {}, expected rank 2",
                                    source_tensor.shape.len()
                                ),
                            });
                        }
                        vec![source_tensor.shape[1], source_tensor.shape[0]]
                    }
                };
                let planned_transform = match transform {
                    TensorTransform::Identity => PlannedTransform::Identity,
                    TensorTransform::Transpose2d => PlannedTransform::Transpose2d,
                };
                insert_primary(
                    &expected,
                    &mut produced,
                    &mut primary,
                    target,
                    vec![source_tensor.into()],
                    planned_transform,
                    shape,
                    source_tensor.dtype,
                    options.dtype_policy,
                )?;
            }
            TensorOperation::Split {
                source,
                axis,
                targets,
            } => {
                consume(&mut consumed, source)?;
                let source_tensor = source_tensor(manifest, source)?;
                let axis_len = *source_tensor.shape.get(*axis).ok_or_else(|| {
                    CheckpointError::InvalidMapping {
                        message: format!(
                            "split axis {axis} is out of bounds for `{source}` rank {}",
                            source_tensor.shape.len()
                        ),
                    }
                })?;
                if targets.is_empty() {
                    return Err(CheckpointError::InvalidMapping {
                        message: format!("split of `{source}` has no targets"),
                    });
                }
                let total = targets.iter().try_fold(0_u64, |sum, (_, length)| {
                    sum.checked_add(*length)
                        .ok_or(CheckpointError::ArithmeticOverflow {
                            context: "summing split section lengths",
                        })
                })?;
                if total != axis_len {
                    return Err(CheckpointError::InvalidMapping {
                        message: format!(
                            "split sections for `{source}` sum to {total}, axis {axis} has length {axis_len}"
                        ),
                    });
                }
                let mut start = 0_u64;
                for (target, length) in targets {
                    let mut shape = source_tensor.shape.clone();
                    shape[*axis] = *length;
                    insert_primary(
                        &expected,
                        &mut produced,
                        &mut primary,
                        target,
                        vec![source_tensor.into()],
                        PlannedTransform::Slice {
                            axis: *axis,
                            start,
                            length: *length,
                        },
                        shape,
                        source_tensor.dtype,
                        options.dtype_policy,
                    )?;
                    start += length;
                }
            }
            TensorOperation::Concatenate {
                sources,
                target,
                axis,
            } => {
                if sources.is_empty() {
                    return Err(CheckpointError::InvalidMapping {
                        message: format!("concatenation for `{target}` has no sources"),
                    });
                }
                let source_tensors = sources
                    .iter()
                    .map(|source| {
                        consume(&mut consumed, source)?;
                        source_tensor(manifest, source)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let first = source_tensors[0];
                if *axis >= first.shape.len() {
                    return Err(CheckpointError::InvalidMapping {
                        message: format!(
                            "concatenation axis {axis} is out of bounds for `{}` rank {}",
                            first.name,
                            first.shape.len()
                        ),
                    });
                }
                let mut shape = first.shape.clone();
                shape[*axis] = 0;
                for tensor in &source_tensors {
                    if tensor.dtype != first.dtype {
                        return Err(CheckpointError::InvalidMapping {
                            message: format!(
                                "concatenation for `{target}` mixes {} and {}",
                                first.dtype, tensor.dtype
                            ),
                        });
                    }
                    if tensor.shape.len() != shape.len()
                        || tensor
                            .shape
                            .iter()
                            .enumerate()
                            .any(|(index, value)| index != *axis && *value != shape[index])
                    {
                        return Err(CheckpointError::InvalidMapping {
                            message: format!(
                                "concatenation source `{}` has incompatible shape {:?}",
                                tensor.name, tensor.shape
                            ),
                        });
                    }
                    shape[*axis] = shape[*axis].checked_add(tensor.shape[*axis]).ok_or(
                        CheckpointError::ArithmeticOverflow {
                            context: "summing concatenated tensor dimensions",
                        },
                    )?;
                }
                insert_primary(
                    &expected,
                    &mut produced,
                    &mut primary,
                    target,
                    source_tensors
                        .into_iter()
                        .map(PlannedSource::from)
                        .collect(),
                    PlannedTransform::Concatenate { axis: *axis },
                    shape,
                    first.dtype,
                    options.dtype_policy,
                )?;
            }
            TensorOperation::Alias {
                source_target,
                target,
            } => {
                if produced.contains(target) || alias_specs.contains_key(target) {
                    return Err(CheckpointError::MappingTargetCollision {
                        tensor: target.clone(),
                    });
                }
                if !expected.contains_key(target) {
                    return Err(CheckpointError::UnknownMappingTarget {
                        tensor: target.clone(),
                    });
                }
                alias_specs.insert(target.clone(), source_target.clone());
            }
        }
    }

    let primary_by_target: BTreeMap<_, _> = primary
        .iter()
        .map(|tensor: &PlannedTensor| (tensor.target.clone(), tensor))
        .collect();
    let mut aliases = Vec::new();
    for target in alias_specs.keys() {
        let root_target = resolve_alias(target, &alias_specs, &primary_by_target)?;
        let root = primary_by_target[root_target];
        let expectation = expected[target];
        ensure_shape(target, &expectation.shape, &root.shape)?;
        if let Some(dtype) = expectation.dtype
            && dtype != root.dtype
        {
            return Err(CheckpointError::DTypeMismatch {
                tensor: target.clone(),
                observed: root.dtype,
                expected: dtype,
            });
        }
        produced.insert(target.clone());
        aliases.push(PlannedTensor {
            target: target.clone(),
            sources: Vec::new(),
            transform: PlannedTransform::Alias {
                source_target: root_target.clone(),
            },
            shape: root.shape.clone(),
            dtype: root.dtype,
        });
    }

    let equivalences = plan_tensor_equivalences(manifest, options, &mut consumed)?;
    let mut report = create_report(&primary, &aliases);
    report.checkpoint_aliases = manifest.index_aliases.clone();
    report.equivalence_planned = equivalences
        .iter()
        .map(|equivalence| {
            CheckpointEquivalenceRecord::new(&equivalence.alias.name, &equivalence.owner.name)
        })
        .collect();
    for key in expected.keys().filter(|key| !produced.contains(*key)) {
        if matches_any(&options.allow_missing, key) {
            report.allowlisted_missing.push(key.clone());
        } else {
            report.missing.push(key.clone());
        }
    }
    // Index-only storage aliases are part of the source checkpoint's logical
    // inventory even though they have no second payload. Requiring them to be
    // consumed or allowlisted prevents strict reports from silently dropping an
    // architecture-authorized exception after header discovery.
    for key in manifest
        .tensors
        .keys()
        .chain(manifest.index_aliases.iter().map(|alias| &alias.alias))
        .filter(|key| !consumed.contains(*key))
    {
        if matches_any(&options.allow_unexpected, key) {
            report.allowlisted_ignored.push(key.clone());
        } else {
            report.unexpected.push(key.clone());
        }
    }
    report.finalize();
    if !report.is_strictly_valid() {
        return Err(CheckpointError::StrictValidation {
            missing: report.missing.len(),
            unexpected: report.unexpected.len(),
            report: Box::new(report),
        });
    }

    primary.sort_by(|left, right| left.target.cmp(&right.target));
    aliases.sort_by(|left, right| left.target.cmp(&right.target));
    let memory = estimate_memory(manifest, &primary, &aliases, &equivalences)?;
    report.memory = Some(memory);
    Ok(LoadPlan {
        equivalences,
        tensors: primary,
        aliases,
        report,
        memory,
    })
}

/// Plans exact comparisons for declared aliases that own physical payloads.
///
/// A declaration is optional across artifact encodings: a key omitted from the
/// index entirely requires no work, an index-only key was already resolved by
/// discovery, and a physically present key becomes a pass-two comparison. Both
/// index-only and physical aliases count as consumed checkpoint entries rather
/// than generic allowlisted omissions.
fn plan_tensor_equivalences(
    manifest: &CheckpointManifest,
    options: &LoadOptions,
    consumed: &mut BTreeSet<String>,
) -> Result<Vec<PlannedTensorEquivalence>> {
    let declared = super::artifact::validated_index_alias_policy(&options.index_aliases)?;
    let index_only = manifest
        .index_aliases
        .iter()
        .map(|alias| (alias.alias.as_str(), alias.storage_key.as_str()))
        .collect::<BTreeMap<_, _>>();
    for alias in &manifest.index_aliases {
        consume(consumed, &alias.alias)?;
    }

    let mut planned = Vec::new();
    for (alias, owner) in declared {
        if index_only.get(alias.as_str()) == Some(&owner.as_str()) {
            continue;
        }
        let Some(alias_tensor) = manifest.tensors.get(&alias) else {
            continue;
        };
        let owner_tensor = manifest.tensors.get(&owner).ok_or_else(|| {
            CheckpointError::TensorEquivalenceMismatch {
                alias: alias.clone(),
                owner: owner.clone(),
                reason: "declared owner is absent from the physical tensor inventory".into(),
            }
        })?;
        if alias_tensor.shape != owner_tensor.shape {
            return Err(CheckpointError::TensorEquivalenceMismatch {
                alias,
                owner,
                reason: format!(
                    "shape {:?} differs from owner shape {:?}",
                    alias_tensor.shape, owner_tensor.shape
                ),
            });
        }
        if alias_tensor.dtype != owner_tensor.dtype {
            return Err(CheckpointError::TensorEquivalenceMismatch {
                alias,
                owner,
                reason: format!(
                    "dtype {} differs from owner dtype {}",
                    alias_tensor.dtype, owner_tensor.dtype
                ),
            });
        }
        if alias_tensor.byte_len != owner_tensor.byte_len {
            return Err(CheckpointError::TensorEquivalenceMismatch {
                alias,
                owner,
                reason: format!(
                    "payload length {} differs from owner length {}",
                    alias_tensor.byte_len, owner_tensor.byte_len
                ),
            });
        }
        consume(consumed, &alias)?;
        planned.push(PlannedTensorEquivalence {
            alias: alias_tensor.into(),
            owner: owner_tensor.into(),
        });
    }
    planned.sort_by(|left, right| {
        left.alias
            .name
            .cmp(&right.alias.name)
            .then_with(|| left.owner.name.cmp(&right.owner.name))
    });
    Ok(planned)
}

/// Rechecks index-only alias and size provenance before compiling a public manifest.
///
/// [`CheckpointManifest`] remains inspectable and constructible by downstream
/// code, so planning cannot assume its discovery-only invariants still hold. The
/// checks below bind every resolved alias to an authorized declaration, its
/// physical owner, the owner's shard inventory, and the exact byte-accounting
/// convention that accepted `metadata.total_size`.
fn validate_manifest_index_provenance(
    manifest: &CheckpointManifest,
    options: &LoadOptions,
    total_size_policy: Option<&super::accounting::PinnedIndexTotalSizePolicy>,
) -> Result<()> {
    let authorized = super::artifact::validated_index_alias_policy(&options.index_aliases)?;
    let declarations = manifest
        .index_aliases
        .iter()
        .map(|alias| IndexTensorAlias::new(&alias.alias, &alias.storage_key))
        .collect::<Vec<_>>();
    let resolved = super::artifact::validated_index_alias_policy(&declarations)?;
    if resolved.len() != manifest.index_aliases.len() {
        return Err(CheckpointError::InvalidIndexAlias {
            alias: "<duplicate>".into(),
            reason: "resolved manifest aliases must have unique alias keys".into(),
        });
    }

    for alias in &manifest.index_aliases {
        if authorized.get(&alias.alias) != Some(&alias.storage_key) {
            return Err(CheckpointError::InvalidIndexAlias {
                alias: alias.alias.clone(),
                reason: format!(
                    "resolved storage key `{}` is not authorized by this load plan",
                    alias.storage_key
                ),
            });
        }
        if alias.shard.is_empty() {
            return Err(CheckpointError::InvalidIndexAlias {
                alias: alias.alias.clone(),
                reason: "resolved shard provenance must not be empty".into(),
            });
        }
        if manifest.tensors.contains_key(&alias.alias) {
            return Err(CheckpointError::InvalidIndexAlias {
                alias: alias.alias.clone(),
                reason: "index-only alias is also present in the physical tensor inventory".into(),
            });
        }
        let owner = manifest.tensors.get(&alias.storage_key).ok_or_else(|| {
            CheckpointError::InvalidIndexAlias {
                alias: alias.alias.clone(),
                reason: format!(
                    "storage key `{}` is absent from the physical tensor inventory",
                    alias.storage_key
                ),
            }
        })?;
        if owner.shard != alias.shard {
            return Err(CheckpointError::InvalidIndexAlias {
                alias: alias.alias.clone(),
                reason: format!(
                    "resolved shard `{}` differs from storage owner shard `{}`",
                    alias.shard, owner.shard
                ),
            });
        }
        let shard =
            manifest
                .shard(&alias.shard)
                .ok_or_else(|| CheckpointError::InvalidIndexAlias {
                    alias: alias.alias.clone(),
                    reason: format!(
                        "resolved shard `{}` is absent from the manifest",
                        alias.shard
                    ),
                })?;
        if shard.tensors.get(&alias.storage_key) != Some(owner) {
            return Err(CheckpointError::InvalidIndexAlias {
                alias: alias.alias.clone(),
                reason: format!(
                    "storage key `{}` does not match its resolved shard inventory",
                    alias.storage_key
                ),
            });
        }
    }

    match &manifest.source {
        ArtifactSource::Single { .. } => {
            if !manifest.index_aliases.is_empty() {
                return Err(manifest_accounting_mismatch(
                    "an unsharded manifest cannot contain index-only aliases".into(),
                ));
            }
            if manifest.index_alias_size_accounting.is_some() {
                return Err(manifest_accounting_mismatch(
                    "an unsharded manifest cannot record index alias accounting".into(),
                ));
            }
        }
        ArtifactSource::Index {
            declared_total_size,
            ..
        } => {
            let physical_bytes = manifest.tensors.values().try_fold(0_u64, |total, tensor| {
                total
                    .checked_add(tensor.byte_len)
                    .ok_or(CheckpointError::ArithmeticOverflow {
                        context: "revalidating checkpoint manifest tensor bytes",
                    })
            })?;
            if manifest.tensor_bytes != physical_bytes {
                return Err(manifest_accounting_mismatch(format!(
                    "manifest records {} physical tensor bytes, inventory sums to {physical_bytes}",
                    manifest.tensor_bytes
                )));
            }
            let recorded = manifest.index_alias_size_accounting.ok_or_else(|| {
                manifest_accounting_mismatch(
                    "an indexed manifest is missing its alias-size convention".into(),
                )
            })?;
            if recorded != options.index_alias_size_accounting {
                return Err(manifest_accounting_mismatch(format!(
                    "manifest was discovered with {recorded:?} alias accounting, load plan requested {:?}",
                    options.index_alias_size_accounting
                )));
            }
            let expected_indexed_bytes = match recorded {
                IndexAliasSizeAccounting::Logical => {
                    manifest
                        .index_aliases
                        .iter()
                        .try_fold(physical_bytes, |total, alias| {
                            let owner = &manifest.tensors[&alias.storage_key];
                            total.checked_add(owner.byte_len).ok_or(
                                CheckpointError::ArithmeticOverflow {
                                    context: "revalidating logical index tensor bytes",
                                },
                            )
                        })?
                }
                IndexAliasSizeAccounting::Physical => physical_bytes,
            };
            if manifest.indexed_tensor_bytes != expected_indexed_bytes {
                return Err(manifest_accounting_mismatch(format!(
                    "manifest records {} indexed tensor bytes, validated convention requires {expected_indexed_bytes}",
                    manifest.indexed_tensor_bytes
                )));
            }
            if let Some(declared) = declared_total_size
                && *declared != expected_indexed_bytes
            {
                let Some(policy) = total_size_policy else {
                    return Err(CheckpointError::TotalSizeMismatch {
                        declared: *declared,
                        observed: expected_indexed_bytes,
                    });
                };
                super::artifact::validate_manifest_index_total_size_policy(manifest, policy)?;
            }
        }
    }
    Ok(())
}

/// Builds one contextual public-manifest accounting failure.
fn manifest_accounting_mismatch(reason: String) -> CheckpointError {
    CheckpointError::ManifestAccountingMismatch { reason }
}

// Validation helpers operate on temporary plans before parameter allocation.

/// Indexes expected targets after checking nonempty unique names and shape products.
fn expected_inventory(expected: &[ExpectedTensor]) -> Result<BTreeMap<String, &ExpectedTensor>> {
    let mut inventory = BTreeMap::new();
    for tensor in expected {
        if tensor.name.is_empty() {
            return Err(CheckpointError::InvalidMapping {
                message: "expected tensor names cannot be empty".into(),
            });
        }
        checked_elements(&tensor.shape)?;
        if inventory.insert(tensor.name.clone(), tensor).is_some() {
            return Err(CheckpointError::MappingTargetCollision {
                tensor: tensor.name.clone(),
            });
        }
    }
    Ok(inventory)
}

/// Adds one uniquely owned target only after its shape and dtype contract match.
///
/// This mutates temporary planning state, not model parameters. A rejected plan is
/// discarded before execution, so partial planning cannot install invalid tensors.
#[allow(clippy::too_many_arguments)]
fn insert_primary(
    expected: &BTreeMap<String, &ExpectedTensor>,
    produced: &mut BTreeSet<String>,
    planned: &mut Vec<PlannedTensor>,
    target: &str,
    sources: Vec<PlannedSource>,
    transform: PlannedTransform,
    shape: Vec<u64>,
    source_dtype: TensorDType,
    policy: DTypePolicy,
) -> Result<()> {
    let expectation =
        expected
            .get(target)
            .ok_or_else(|| CheckpointError::UnknownMappingTarget {
                tensor: target.to_owned(),
            })?;
    if !produced.insert(target.to_owned()) {
        return Err(CheckpointError::MappingTargetCollision {
            tensor: target.to_owned(),
        });
    }
    ensure_shape(target, &expectation.shape, &shape)?;
    let dtype = resolve_dtype(target, source_dtype, expectation.dtype, policy)?;
    planned.push(PlannedTensor {
        target: target.to_owned(),
        sources,
        transform,
        shape,
        dtype,
    });
    Ok(())
}

/// Marks a physical source consumed once; explicit split operations own their reuse.
fn consume(consumed: &mut BTreeSet<String>, source: &str) -> Result<()> {
    if !consumed.insert(source.to_owned()) {
        return Err(CheckpointError::MappingSourceCollision {
            tensor: source.to_owned(),
        });
    }
    Ok(())
}

/// Looks up a required physical source and reports its missing checkpoint key.
fn source_tensor<'a>(
    manifest: &'a CheckpointManifest,
    source: &str,
) -> Result<&'a TensorInventory> {
    manifest
        .tensor(source)
        .ok_or_else(|| CheckpointError::MissingMappingSource {
            tensor: source.to_owned(),
        })
}

/// Requires exact axis-by-axis equality rather than equal flattened element counts.
fn ensure_shape(target: &str, expected: &[u64], observed: &[u64]) -> Result<()> {
    if expected != observed {
        return Err(CheckpointError::ShapeMismatch {
            tensor: target.to_owned(),
            expected: expected.to_vec(),
            observed: observed.to_vec(),
        });
    }
    Ok(())
}

/// Resolves conversion policy while preserving concrete architecture dtype contracts.
///
/// Automatic conversion changes only real floating-point types. Blanket casts
/// cannot override a concrete target dtype or reinterpret integer parameters.
fn resolve_dtype(
    target: &str,
    source: TensorDType,
    expected: Option<TensorDType>,
    policy: DTypePolicy,
) -> Result<TensorDType> {
    let resolved = match policy {
        DTypePolicy::Preserve => source,
        DTypePolicy::Auto => match expected {
            Some(dtype) if source != dtype && source.is_float() && dtype.is_float() => dtype,
            _ => source,
        },
        // A concrete architecture contract denotes a numerically significant
        // storage dtype (for example an FP32 recurrent/router buffer), so the
        // blanket cast applies only to dtype-generic floating parameters.
        DTypePolicy::Cast(dtype) if source.is_float() => expected
            .filter(|candidate| candidate.is_float())
            .unwrap_or(dtype),
        DTypePolicy::Cast(_) => source,
    };
    if let Some(expected) = expected
        && resolved != expected
    {
        return Err(CheckpointError::DTypeMismatch {
            tensor: target.to_owned(),
            observed: resolved,
            expected,
        });
    }
    Ok(resolved)
}

/// Follows alias chains to a storage owner and rejects cycles or dangling targets.
fn resolve_alias<'a>(
    target: &'a str,
    aliases: &'a BTreeMap<String, String>,
    primary: &BTreeMap<String, &PlannedTensor>,
) -> Result<&'a String> {
    let mut cursor = target;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(cursor) {
            return Err(CheckpointError::InvalidMapping {
                message: format!("alias cycle contains `{cursor}`"),
            });
        }
        let source = aliases
            .get(cursor)
            .ok_or_else(|| CheckpointError::InvalidMapping {
                message: format!("alias `{target}` has no source target"),
            })?;
        if primary.contains_key(source) {
            return Ok(source);
        }
        if aliases.contains_key(source) {
            cursor = source;
        } else {
            return Err(CheckpointError::InvalidMapping {
                message: format!("alias `{target}` references unknown target `{source}`"),
            });
        }
    }
}

// Auditable transformations and conservative host/parameter storage accounting.

/// Builds deterministic provenance records for renames, layout changes, casts, and ties.
fn create_report(primary: &[PlannedTensor], aliases: &[PlannedTensor]) -> LoadReport {
    let mut report = LoadReport::new();
    for tensor in primary {
        let source_dtype = tensor.sources[0].dtype;
        if source_dtype != tensor.dtype {
            report.dtype_converted.push(DTypeConversion {
                target: tensor.target.clone(),
                source: source_dtype,
                target_dtype: tensor.dtype,
            });
        }
        let renamed = tensor.sources.len() == 1 && tensor.sources[0].name != tensor.target;
        let operation = match &tensor.transform {
            PlannedTransform::Identity if renamed => Some("rename".to_owned()),
            PlannedTransform::Identity => None,
            PlannedTransform::Transpose2d => Some("transpose_2d".to_owned()),
            PlannedTransform::Slice {
                axis,
                start,
                length,
            } => Some(format!("slice(axis={axis},start={start},length={length})")),
            PlannedTransform::Concatenate { axis } => Some(format!("concatenate(axis={axis})")),
            PlannedTransform::Alias { .. } => None,
        };
        if let Some(operation) = operation {
            report.transformed.push(TransformRecord {
                target: tensor.target.clone(),
                sources: tensor
                    .sources
                    .iter()
                    .map(|source| source.name.clone())
                    .collect(),
                operation,
            });
        }
    }
    for tensor in aliases {
        let PlannedTransform::Alias { source_target } = &tensor.transform else {
            continue;
        };
        report.aliased.push(AliasRecord {
            target: tensor.target.clone(),
            source_target: source_target.clone(),
        });
    }
    report
}

/// Estimates logical, uniquely owned, and conservative temporary storage separately.
///
/// Aliases add logical bytes without new parameter storage. The temporary bound is
/// the maximum of largest-shard-plus-output and the largest physical equivalence
/// pair; it does not include backend queues, activations, or allocator overhead.
fn estimate_memory(
    manifest: &CheckpointManifest,
    primary: &[PlannedTensor],
    aliases: &[PlannedTensor],
    equivalences: &[PlannedTensorEquivalence],
) -> Result<MemoryEstimate> {
    let largest_shard_bytes = manifest
        .shards
        .iter()
        .map(|shard| shard.file_bytes)
        .max()
        .unwrap_or(0);
    let largest_source_tensor_bytes = manifest
        .tensors
        .values()
        .map(|tensor| tensor.byte_len)
        .max()
        .unwrap_or(0);
    let mut target_parameter_bytes = 0_u64;
    let mut logical_model_bytes = 0_u64;
    let mut largest_target_tensor_bytes = 0_u64;
    for tensor in primary {
        let bytes = tensor_bytes(&tensor.shape, tensor.dtype)?;
        target_parameter_bytes = target_parameter_bytes.checked_add(bytes).ok_or(
            CheckpointError::ArithmeticOverflow {
                context: "summing target parameter bytes",
            },
        )?;
        logical_model_bytes =
            logical_model_bytes
                .checked_add(bytes)
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "summing logical model bytes",
                })?;
        largest_target_tensor_bytes = largest_target_tensor_bytes.max(bytes);
    }
    for tensor in aliases {
        logical_model_bytes = logical_model_bytes
            .checked_add(tensor_bytes(&tensor.shape, tensor.dtype)?)
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "summing aliased logical model bytes",
            })?;
    }
    let largest_equivalence_pair_bytes =
        equivalences
            .iter()
            .try_fold(0_u64, |largest, equivalence| {
                let alias_bytes = equivalence.alias.data_offsets[1]
                    .checked_sub(equivalence.alias.data_offsets[0])
                    .ok_or_else(|| CheckpointError::InvalidMapping {
                        message: format!(
                            "equivalence alias `{}` has reversed data offsets",
                            equivalence.alias.name
                        ),
                    })?;
                let owner_bytes = equivalence.owner.data_offsets[1]
                    .checked_sub(equivalence.owner.data_offsets[0])
                    .ok_or_else(|| CheckpointError::InvalidMapping {
                        message: format!(
                            "equivalence owner `{}` has reversed data offsets",
                            equivalence.owner.name
                        ),
                    })?;
                let pair = alias_bytes.checked_add(owner_bytes).ok_or(
                    CheckpointError::ArithmeticOverflow {
                        context: "estimating checkpoint equivalence staging bytes",
                    },
                )?;
                Ok::<u64, CheckpointError>(largest.max(pair))
            })?;
    let transform_staging_bytes = largest_shard_bytes
        .checked_add(largest_target_tensor_bytes)
        .ok_or(CheckpointError::ArithmeticOverflow {
            context: "estimating peak host staging bytes",
        })?;
    let peak_host_staging_bytes = transform_staging_bytes.max(largest_equivalence_pair_bytes);
    Ok(MemoryEstimate {
        artifact_bytes: manifest.artifact_bytes,
        checkpoint_tensor_bytes: manifest.tensor_bytes,
        target_parameter_bytes,
        logical_model_bytes,
        largest_shard_bytes,
        largest_source_tensor_bytes,
        largest_target_tensor_bytes,
        largest_equivalence_pair_bytes,
        peak_host_staging_bytes,
    })
}

/// Computes exact tensor storage after validating the dimension product and bit alignment.
fn tensor_bytes(shape: &[u64], dtype: TensorDType) -> Result<u64> {
    dtype.storage_bytes(checked_elements(shape)?)
}

/// Multiplies target dimensions without overflow, treating rank zero as one scalar.
fn checked_elements(shape: &[u64]) -> Result<u64> {
    shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "computing mapped tensor element count",
            })
    })
}

/// Tests whether any explicit exact-name or prefix exception selects a key.
fn matches_any(matchers: &[KeyMatcher], key: &str) -> bool {
    matchers.iter().any(|matcher| matcher.matches(key))
}

/// Validates every exception before it can relax strict missing/unexpected-key checks.
fn validate_matchers(matchers: &[KeyMatcher]) -> Result<()> {
    for matcher in matchers {
        matcher.validate()?;
    }
    Ok(())
}
