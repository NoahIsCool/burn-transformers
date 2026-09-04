//! Explicit checkpoint-to-model key and layout mapping.
//!
//! A mapping plan is intentionally data, not filename-dependent string magic in a
//! generic loader. Each architecture declares how its external checkpoint tensors
//! produce internal parameters. This makes fused projections, transposes, aliases,
//! and ignored multimodal towers visible in validation reports and unit tests.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{
    AliasRecord, CheckpointError, CheckpointManifest, DTypeConversion, DTypePolicy, LoadOptions,
    LoadReport, MemoryEstimate, Result, TensorDType, TensorInventory, TransformRecord,
};

/// Exact or prefix-based exception for a checkpoint key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum KeyMatcher {
    /// Matches exactly one tensor name.
    Exact(String),
    /// Matches all tensor names beginning with a nonempty prefix.
    Prefix(String),
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

/// Strictly validated second-pass load plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadPlan {
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
    validate_matchers(&options.allow_missing)?;
    validate_matchers(&options.allow_unexpected)?;
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

    let mut report = create_report(&primary, &aliases);
    for key in expected.keys().filter(|key| !produced.contains(*key)) {
        if matches_any(&options.allow_missing, key) {
            report.allowlisted_missing.push(key.clone());
        } else {
            report.missing.push(key.clone());
        }
    }
    for key in manifest
        .tensors
        .keys()
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
    let memory = estimate_memory(manifest, &primary, &aliases)?;
    report.memory = Some(memory);
    Ok(LoadPlan {
        tensors: primary,
        aliases,
        report,
        memory,
    })
}

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

fn consume(consumed: &mut BTreeSet<String>, source: &str) -> Result<()> {
    if !consumed.insert(source.to_owned()) {
        return Err(CheckpointError::MappingSourceCollision {
            tensor: source.to_owned(),
        });
    }
    Ok(())
}

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

fn estimate_memory(
    manifest: &CheckpointManifest,
    primary: &[PlannedTensor],
    aliases: &[PlannedTensor],
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
    let peak_host_staging_bytes = largest_shard_bytes
        .checked_add(largest_target_tensor_bytes)
        .ok_or(CheckpointError::ArithmeticOverflow {
            context: "estimating peak host staging bytes",
        })?;
    Ok(MemoryEstimate {
        artifact_bytes: manifest.artifact_bytes,
        checkpoint_tensor_bytes: manifest.tensor_bytes,
        target_parameter_bytes,
        logical_model_bytes,
        largest_shard_bytes,
        largest_source_tensor_bytes,
        largest_target_tensor_bytes,
        peak_host_staging_bytes,
    })
}

fn tensor_bytes(shape: &[u64], dtype: TensorDType) -> Result<u64> {
    dtype.storage_bytes(checked_elements(shape)?)
}

fn checked_elements(shape: &[u64]) -> Result<u64> {
    shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "computing mapped tensor element count",
            })
    })
}

fn matches_any(matchers: &[KeyMatcher], key: &str) -> bool {
    matchers.iter().any(|matcher| matcher.matches(key))
}

fn validate_matchers(matchers: &[KeyMatcher]) -> Result<()> {
    for matcher in matchers {
        matcher.validate()?;
    }
    Ok(())
}
