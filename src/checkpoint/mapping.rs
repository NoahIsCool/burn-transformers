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

mod aliases;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use aliases::{AliasLink, AliasWalk, walk_alias_chain};

use super::{
    AliasRecord, ArtifactSource, CheckpointEquivalenceRecord, CheckpointError, CheckpointManifest,
    DTypeConversion, DTypePolicy, IndexAliasSizeAccounting, IndexTensorAlias, LoadOptions,
    LoadReport, MemoryEstimate, Result, TensorDType, TensorInventory, TransformRecord,
    layout::{ConcatError, SectionError, extend_concatenation, section_starts},
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
    /// Decode an MXFP4 blocks/scales pair into one floating-point target.
    ///
    /// `blocks` is `u8 [experts, rows, groups, 16]` (two FP4 E2M1 codes per
    /// byte) and `scales` is `[experts, rows, groups]` E8M0 exponents (`U8` or
    /// `F8_E8M0`). The target is `[experts, 32 · groups, rows]`, exactly the
    /// tensor Transformers' `convert_moe_packed_tensors` returns; see
    /// [`crate::checkpoint`]'s `mxfp4` module for the decoding rules. The
    /// decoded values are nominally BF16 (Transformers' dequantized dtype),
    /// so the load plan's [`DTypePolicy`] selects the stored dtype, and each
    /// value is rounded once from its exact decoded value.
    DequantizeMxfp4 {
        /// Packed `*_blocks` checkpoint tensor name.
        blocks: String,
        /// Shared `*_scales` checkpoint tensor name.
        scales: String,
        /// Model parameter name.
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

    /// Adds an MXFP4 dequantization of a blocks/scales pair into one target.
    ///
    /// See [`TensorOperation::DequantizeMxfp4`] for the shapes and decoding.
    pub fn dequantize_mxfp4(
        &mut self,
        blocks: impl Into<String>,
        scales: impl Into<String>,
        target: impl Into<String>,
    ) -> &mut Self {
        self.operations.push(TensorOperation::DequantizeMxfp4 {
            blocks: blocks.into(),
            scales: scales.into(),
            target: target.into(),
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
    /// MXFP4 decoding of `sources = [blocks, scales]` into the target dtype.
    DequantizeMxfp4,
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

    /// Returns the tight host staging requirement of executing this plan.
    ///
    /// This is the maximum over primary operations of
    /// [`PlannedTensor::staging_bytes`] and over equivalence work items of the
    /// alias-plus-owner payload pair: the largest amount the generic executor
    /// holds at once. It never exceeds the conservative
    /// [`MemoryEstimate::peak_host_staging_bytes`] of a plan built by
    /// [`build_load_plan`], which additionally budgets for whole shard files.
    pub fn required_host_staging_bytes(&self) -> Result<u64> {
        let mut required = self.memory.largest_equivalence_pair_bytes;
        for tensor in &self.tensors {
            required = required.max(tensor.staging_bytes()?);
        }
        Ok(required)
    }
}

impl PlannedTensor {
    /// Returns the host bytes the executor stages while materializing this target.
    ///
    /// The executor holds one source payload at a time plus the output buffer:
    /// `largest source + output`. MXFP4 decoding holds both of its sources and
    /// the output: `blocks + scales + output`. An identity transfer that converts in place
    /// ([`TensorDType::converts_in_place_to`]) reuses its source buffer, so it
    /// needs only the source. The runtime enforces this same figure against
    /// [`MemoryEstimate::peak_host_staging_bytes`] before any source read.
    pub fn staging_bytes(&self) -> Result<u64> {
        let output = self.dtype.tensor_storage_bytes(&self.shape)?;
        if matches!(self.transform, PlannedTransform::DequantizeMxfp4) {
            // Blocks, scales, and the decoded output are resident together.
            let mut total = output;
            for source in &self.sources {
                total = total
                    .checked_add(source.dtype.tensor_storage_bytes(&source.shape)?)
                    .ok_or(CheckpointError::ArithmeticOverflow {
                        context: "computing MXFP4 decoding staging bound",
                    })?;
            }
            return Ok(total);
        }
        let mut largest_source = 0_u64;
        for source in &self.sources {
            largest_source = largest_source.max(source.dtype.tensor_storage_bytes(&source.shape)?);
        }
        let in_place = matches!(self.transform, PlannedTransform::Identity)
            && self
                .sources
                .first()
                .is_some_and(|source| source.dtype.converts_in_place_to(self.dtype));
        if in_place {
            return Ok(largest_source);
        }
        largest_source
            .checked_add(output)
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "computing runtime staging bound",
            })
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
    if matches!(options.dtype_policy, DTypePolicy::Cast(dtype) if !dtype.is_castable_float()) {
        return Err(CheckpointError::InvalidMapping {
            message: "DTypePolicy::Cast requires a castable floating-point target dtype \
                      (F16, BF16, F32, or F64)"
                .into(),
        });
    }

    let mut builder = PlanBuilder {
        manifest,
        expected: expected_inventory(expected)?,
        policy: options.dtype_policy,
        consumed: BTreeSet::new(),
        produced: BTreeSet::new(),
        primary: Vec::new(),
        alias_specs: BTreeMap::new(),
    };
    for operation in mappings.operations() {
        builder.apply(operation)?;
    }
    let mut aliases = builder.resolve_aliases()?;
    let PlanBuilder {
        expected,
        mut consumed,
        produced,
        mut primary,
        ..
    } = builder;

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

/// Temporary state while compiling mapping operations into planned tensors.
///
/// Every operation claims its targets through [`Self::claim_primary`] or
/// [`Self::declare_alias`], which both consult the primary *and* alias target
/// sets, so a target collision is rejected regardless of declaration order.
/// Each source is consumed through [`consume`], so no checkpoint tensor is used
/// twice. Nothing here touches model parameters: a rejected plan is discarded.
struct PlanBuilder<'a> {
    /// Discovered checkpoint inventory that sources are resolved against.
    manifest: &'a CheckpointManifest,
    /// Expected model parameters keyed by target name.
    expected: BTreeMap<String, &'a ExpectedTensor>,
    /// Dtype conversion policy applied to every primary target.
    policy: DTypePolicy,
    /// Checkpoint tensors already consumed by an operation.
    consumed: BTreeSet<String>,
    /// Targets claimed by primary operations, plus aliases once resolved.
    produced: BTreeSet<String>,
    /// Planned storage-owning targets in declaration order.
    primary: Vec<PlannedTensor>,
    /// Declared alias targets mapped to their (possibly aliased) sources.
    alias_specs: BTreeMap<String, String>,
}

impl PlanBuilder<'_> {
    /// Plans one declared mapping operation.
    fn apply(&mut self, operation: &TensorOperation) -> Result<()> {
        match operation {
            TensorOperation::Map {
                source,
                target,
                transform,
            } => self.plan_map(source, target, *transform),
            TensorOperation::Split {
                source,
                axis,
                targets,
            } => self.plan_split(source, *axis, targets),
            TensorOperation::Concatenate {
                sources,
                target,
                axis,
            } => self.plan_concatenate(sources, target, *axis),
            TensorOperation::Alias {
                source_target,
                target,
            } => self.declare_alias(source_target, target),
            TensorOperation::DequantizeMxfp4 {
                blocks,
                scales,
                target,
            } => self.plan_mxfp4(blocks, scales, target),
        }
    }

    /// Plans an identity copy, rename, or rank-two transpose of one source.
    fn plan_map(&mut self, source: &str, target: &str, transform: TensorTransform) -> Result<()> {
        consume(&mut self.consumed, source)?;
        let source_tensor = source_tensor(self.manifest, source)?;
        let (shape, planned_transform) = match transform {
            TensorTransform::Identity => (source_tensor.shape.clone(), PlannedTransform::Identity),
            TensorTransform::Transpose2d => {
                let [rows, columns] = source_tensor.shape.as_slice() else {
                    return Err(CheckpointError::InvalidMapping {
                        message: format!(
                            "transpose source `{source}` has rank {}, expected rank 2",
                            source_tensor.shape.len()
                        ),
                    });
                };
                (vec![*columns, *rows], PlannedTransform::Transpose2d)
            }
        };
        self.claim_primary(
            target,
            vec![source_tensor.into()],
            planned_transform,
            shape,
            source_tensor.dtype,
        )
    }

    /// Plans consecutive sections of one fused source along `axis`.
    ///
    /// Section starts come from the verified [`section_starts`], which requires
    /// the lengths to sum exactly to the axis length.
    fn plan_split(&mut self, source: &str, axis: usize, targets: &[(String, u64)]) -> Result<()> {
        consume(&mut self.consumed, source)?;
        let source_tensor = source_tensor(self.manifest, source)?;
        let axis_len =
            *source_tensor
                .shape
                .get(axis)
                .ok_or_else(|| CheckpointError::InvalidMapping {
                    message: format!(
                        "split axis {axis} is out of bounds for `{source}` rank {}",
                        source_tensor.shape.len()
                    ),
                })?;
        if targets.is_empty() {
            return Err(CheckpointError::InvalidMapping {
                message: format!("split of `{source}` has no targets"),
            });
        }
        let lengths = targets
            .iter()
            .map(|(_, length)| *length)
            .collect::<Vec<_>>();
        let starts = section_starts(&lengths, axis_len).map_err(|error| match error {
            SectionError::Overflow => CheckpointError::ArithmeticOverflow {
                context: "summing split section lengths",
            },
            SectionError::LengthMismatch { total } => CheckpointError::InvalidMapping {
                message: format!(
                    "split sections for `{source}` sum to {total}, axis {axis} has length {axis_len}"
                ),
            },
        })?;
        for ((target, length), start) in targets.iter().zip(starts) {
            let mut shape = source_tensor.shape.clone();
            shape[axis] = *length;
            self.claim_primary(
                target,
                vec![source_tensor.into()],
                PlannedTransform::Slice {
                    axis,
                    start,
                    length: *length,
                },
                shape,
                source_tensor.dtype,
            )?;
        }
        Ok(())
    }

    /// Plans a concatenation whose sources agree on dtype and every other axis.
    fn plan_concatenate(&mut self, sources: &[String], target: &str, axis: usize) -> Result<()> {
        if sources.is_empty() {
            return Err(CheckpointError::InvalidMapping {
                message: format!("concatenation for `{target}` has no sources"),
            });
        }
        let source_tensors = sources
            .iter()
            .map(|source| {
                consume(&mut self.consumed, source)?;
                source_tensor(self.manifest, source)
            })
            .collect::<Result<Vec<_>>>()?;
        let first = source_tensors[0];
        if axis >= first.shape.len() {
            return Err(CheckpointError::InvalidMapping {
                message: format!(
                    "concatenation axis {axis} is out of bounds for `{}` rank {}",
                    first.name,
                    first.shape.len()
                ),
            });
        }
        let mut shape = first.shape.clone();
        shape[axis] = 0;
        for tensor in &source_tensors {
            if tensor.dtype != first.dtype {
                return Err(CheckpointError::InvalidMapping {
                    message: format!(
                        "concatenation for `{target}` mixes {} and {}",
                        first.dtype, tensor.dtype
                    ),
                });
            }
            // Verified: agrees with `shape` off the axis and adds its axis length.
            match extend_concatenation(&mut shape, &tensor.shape, axis) {
                Ok(()) => {}
                Err(ConcatError::Incompatible) => {
                    return Err(CheckpointError::InvalidMapping {
                        message: format!(
                            "concatenation source `{}` has incompatible shape {:?}",
                            tensor.name, tensor.shape
                        ),
                    });
                }
                Err(ConcatError::Overflow) => {
                    return Err(CheckpointError::ArithmeticOverflow {
                        context: "summing concatenated tensor dimensions",
                    });
                }
            }
        }
        self.claim_primary(
            target,
            source_tensors
                .into_iter()
                .map(PlannedSource::from)
                .collect(),
            PlannedTransform::Concatenate { axis },
            shape,
            first.dtype,
        )
    }

    /// Plans an MXFP4 dequantization of `blocks` and `scales` into `target`.
    ///
    /// Blocks must be `U8` and scales `U8` or `F8_E8M0` (the same byte
    /// encoding); their shapes must pass [`super::mxfp4::mxfp4_geometry`]. The
    /// target is `[experts, 32 · groups, rows]` with the nominal source dtype
    /// BF16, so the dtype policy chooses the stored float dtype.
    fn plan_mxfp4(&mut self, blocks: &str, scales: &str, target: &str) -> Result<()> {
        consume(&mut self.consumed, blocks)?;
        consume(&mut self.consumed, scales)?;
        let blocks_tensor = source_tensor(self.manifest, blocks)?;
        let scales_tensor = source_tensor(self.manifest, scales)?;
        if blocks_tensor.dtype != TensorDType::U8
            || !matches!(scales_tensor.dtype, TensorDType::U8 | TensorDType::F8_E8M0)
        {
            return Err(CheckpointError::InvalidMapping {
                message: format!(
                    "MXFP4 target `{target}` needs U8 blocks and U8/F8_E8M0 scales, found {} and {}",
                    blocks_tensor.dtype, scales_tensor.dtype
                ),
            });
        }
        let shape = super::mxfp4::decoded_shape(&blocks_tensor.shape, &scales_tensor.shape)
            .map_err(|reason| CheckpointError::InvalidMapping {
                message: format!(
                    "MXFP4 target `{target}`: blocks {:?} and scales {:?} {reason}",
                    blocks_tensor.shape, scales_tensor.shape
                ),
            })?;
        self.claim_primary(
            target,
            vec![blocks_tensor.into(), scales_tensor.into()],
            PlannedTransform::DequantizeMxfp4,
            shape,
            TensorDType::BF16,
        )
    }

    /// Records a tied alias; resolution waits until every owner is planned.
    fn declare_alias(&mut self, source_target: &str, target: &str) -> Result<()> {
        if self.produced.contains(target) || self.alias_specs.contains_key(target) {
            return Err(CheckpointError::MappingTargetCollision {
                tensor: target.to_owned(),
            });
        }
        if !self.expected.contains_key(target) {
            return Err(CheckpointError::UnknownMappingTarget {
                tensor: target.to_owned(),
            });
        }
        self.alias_specs
            .insert(target.to_owned(), source_target.to_owned());
        Ok(())
    }

    /// Adds one uniquely owned target after its shape and dtype contract match.
    ///
    /// A target already claimed by a primary operation or declared as an alias
    /// is a collision, whichever was declared first.
    fn claim_primary(
        &mut self,
        target: &str,
        sources: Vec<PlannedSource>,
        transform: PlannedTransform,
        shape: Vec<u64>,
        source_dtype: TensorDType,
    ) -> Result<()> {
        let expectation =
            self.expected
                .get(target)
                .ok_or_else(|| CheckpointError::UnknownMappingTarget {
                    tensor: target.to_owned(),
                })?;
        if self.alias_specs.contains_key(target) || !self.produced.insert(target.to_owned()) {
            return Err(CheckpointError::MappingTargetCollision {
                tensor: target.to_owned(),
            });
        }
        ensure_shape(target, &expectation.shape, &shape)?;
        let dtype = resolve_dtype(target, source_dtype, expectation.dtype, self.policy)?;
        self.primary.push(PlannedTensor {
            target: target.to_owned(),
            sources,
            transform,
            shape,
            dtype,
        });
        Ok(())
    }

    /// Resolves every alias to its storage-owning primary target.
    ///
    /// Each alias inherits its owner's shape and dtype, which must satisfy the
    /// alias's own expectation. Resolved alias targets join `produced`.
    ///
    /// Aliases are numbered in target-name order and linked to their sources;
    /// the verified [`walk_alias_chain`] follows each chain to its first
    /// owner, dangling source, or repeated alias, so resolution always
    /// terminates and an owner is reached only along an acyclic chain.
    fn resolve_aliases(&mut self) -> Result<Vec<PlannedTensor>> {
        let primary_by_target: BTreeMap<_, _> = self
            .primary
            .iter()
            .map(|tensor| (tensor.target.as_str(), tensor))
            .collect();
        let names = self.alias_specs.keys().collect::<Vec<_>>();
        let sources = self.alias_specs.values().collect::<Vec<_>>();
        let index_of = names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect::<BTreeMap<_, _>>();
        // A primary source owns storage; it is checked before alias sources.
        let links = sources
            .iter()
            .map(|source| {
                if primary_by_target.contains_key(source.as_str()) {
                    AliasLink::Owner
                } else if let Some(&next) = index_of.get(source.as_str()) {
                    AliasLink::Next(next)
                } else {
                    AliasLink::Dangling
                }
            })
            .collect::<Vec<_>>();
        let mut aliases = Vec::with_capacity(self.alias_specs.len());
        for (start, &target) in names.iter().enumerate() {
            let root_target = match walk_alias_chain(&links, start) {
                AliasWalk::Owner(last) => sources[last],
                AliasWalk::Cycle(repeated) => {
                    return Err(CheckpointError::InvalidMapping {
                        message: format!("alias cycle contains `{}`", names[repeated]),
                    });
                }
                AliasWalk::Dangling(last) => {
                    return Err(CheckpointError::InvalidMapping {
                        message: format!(
                            "alias `{target}` references unknown target `{}`",
                            sources[last]
                        ),
                    });
                }
            };
            let root = primary_by_target[root_target.as_str()];
            let expectation = self.expected[target];
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
            if !self.produced.insert(target.clone()) {
                // Unreachable through `declare_alias`/`claim_primary`; kept as
                // defense in depth so a target can never be planned twice.
                return Err(CheckpointError::MappingTargetCollision {
                    tensor: target.clone(),
                });
            }
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
        Ok(aliases)
    }
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
/// A requested change between floats that the executor cannot convert (FP8 or
/// microscaling formats) fails here with
/// [`CheckpointError::UnsupportedDTypeConversion`], so a strictly valid plan
/// never records a conversion that execution would reject.
fn resolve_dtype(
    target: &str,
    source: TensorDType,
    expected: Option<TensorDType>,
    policy: DTypePolicy,
) -> Result<TensorDType> {
    let resolved = match policy {
        DTypePolicy::Preserve => source,
        DTypePolicy::Auto => match expected {
            Some(dtype) if source.is_float() && dtype.is_float() => dtype,
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
    if resolved != source && !(source.is_castable_float() && resolved.is_castable_float()) {
        return Err(CheckpointError::UnsupportedDTypeConversion {
            source_dtype: source,
            target_dtype: resolved,
        });
    }
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

// Auditable transformations and conservative host/parameter storage accounting.

/// Builds deterministic provenance records for renames, layout changes, casts, and ties.
fn create_report(primary: &[PlannedTensor], aliases: &[PlannedTensor]) -> LoadReport {
    let mut report = LoadReport::new();
    for tensor in primary {
        let source_dtype = tensor.sources[0].dtype;
        // MXFP4 decoding writes the target dtype directly; it is a transform,
        // not a cast of its integer sources.
        let decoded = matches!(tensor.transform, PlannedTransform::DequantizeMxfp4);
        if source_dtype != tensor.dtype && !decoded {
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
            PlannedTransform::DequantizeMxfp4 => Some("dequantize_mxfp4".to_owned()),
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
    // MXFP4 decoding holds two sources, which may live in different shards.
    let mut decoding_staging_bytes = 0_u64;
    for tensor in primary
        .iter()
        .filter(|tensor| matches!(tensor.transform, PlannedTransform::DequantizeMxfp4))
    {
        decoding_staging_bytes = decoding_staging_bytes.max(tensor.staging_bytes()?);
    }
    let peak_host_staging_bytes = transform_staging_bytes
        .max(largest_equivalence_pair_bytes)
        .max(decoding_staging_bytes);
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
    super::layout::element_count(shape).ok_or(CheckpointError::ArithmeticOverflow {
        context: "computing mapped tensor element count",
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
