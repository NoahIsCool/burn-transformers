//! Reuse selected parameters in an independently constructed Burn module.
//!
//! Names are ordinary Rust module paths, discoverable through [`parameters`].
//! Construct the architecture first, then copy only the parts the experiment needs:
//! ```
//! use burn::{backend::Flex, nn::LinearConfig};
//! use transformers::transfer::{TransferPlan, ShapeChangePolicy};
//!
//! let device = Default::default();
//! let pretrained = LinearConfig::new(8, 4).init::<Flex>(&device);
//! let replacement = LinearConfig::new(8, 3).init::<Flex>(&device);
//! let (experiment, report) = TransferPlan::new()
//!     .copy_matching()
//!     .on_shape_change(ShapeChangePolicy::Reinitialize)
//!     .apply(&pretrained, &replacement)?;
//! assert_eq!(report.newly_initialized.len(), 2);
//! # Ok::<(), transformers::Error>(())
//! ```
//! For a T5 encoder, select `.copy_prefix("shared", "shared")` and
//! `.copy_prefix("encoder", "encoder")` instead of copying every matching name.
//! The source and target are borrowed. Validation and explicit transforms finish
//! before a cloned target receives any replacement. Tensor payloads stay on device;
//! validating a tied source group synchronizes one Boolean equality reduction.
//! Copying neither casts dtype nor moves devices. Copied parameters start a new
//! gradient graph, preserve the target's trainability, and retain existing ties
//! by default (see [`TiePolicy`]). Copying one source path to several targets,
//! for example `.copy_prefix("layers.0", "layers.1")` next to
//! `.copy_prefix("layers.0", "layers.0")`, creates independent parameters.
//! This API transfers dense floating parameters of already constructed modules.
//! Checkpoint subset loaders use the same [`TransferReport`] while streaming files.

use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
};

use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId},
    tensor::{DType, ElementConversion, Tensor, backend::Backend},
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

mod plan;
use plan::{
    RuleMatch, RuleSelection, StagedKey, TargetTie, class_leaders, first_fanned_out_path,
    select_rule, target_tie,
};

/// Explicit handling of a selected source whose dimensions differ from its target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeChangePolicy {
    /// Reject the transfer; dimensions are never silently sliced or reshaped.
    #[default]
    Reject,
    /// Retain the target's initialization and record why the source was not reused.
    /// Construct a fresh target first when random initialization is desired.
    Reinitialize,
}

/// Whether selected parameter aliases remain tied in the resulting module.
///
/// A *tie* is sharing that already exists: two module paths holding one Burn
/// [`ParamId`]. Copying one source path to several targets, as in depth
/// up-scaling, layer duplication, or expert upcycling, is not a tie. Those copies
/// always receive independent identities, under either policy, so each can train.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TiePolicy {
    /// Keep existing sharing and reject mappings that would silently break it.
    ///
    /// Distinct source paths sharing one identity (for example tied embedding and
    /// output weights) map to targets that share one fresh identity, and existing
    /// target ties stay tied. A preserved tie is a set of duplicate module fields:
    /// Burn 0.21 optimizers update only one of them, so freeze such groups or check
    /// the trained module with [`crate::training::validate_trainable_module`].
    #[default]
    Preserve,
    /// Give each transferred target an independent parameter identity and gradient leaf.
    Untie,
}

/// A parameter copied without a numerical transformation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReusedParameter {
    /// Name in the source module or checkpoint.
    pub source: String,
    /// Name in the resulting module.
    pub target: String,
    /// Resulting runtime dimensions.
    pub shape: Vec<usize>,
}

impl ReusedParameter {
    /// Record one explicit source-to-target copy.
    pub fn new(source: impl Into<String>, target: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            shape,
        }
    }
}

/// A parameter produced by an explicitly named numerical or layout transformation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformedParameter {
    /// Names supplying the transformation.
    pub sources: Vec<String>,
    /// Name in the resulting module.
    pub target: String,
    /// Resulting runtime dimensions.
    pub shape: Vec<usize>,
    /// Human-readable transformation description.
    pub operation: String,
}

/// Explains why one source was omitted or one target retained its initialization.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterDisposition {
    /// Source or target parameter path, according to the containing report category.
    pub name: String,
    /// Explicit reason for the disposition.
    pub reason: String,
}

impl ParameterDisposition {
    /// Record a parameter with a human-readable explanation.
    pub fn new(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reason: reason.into(),
        }
    }
}

/// Auditable outcome of a successful module transfer or selective checkpoint load.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferReport {
    /// Explicit source-to-target copies.
    pub reused: Vec<ReusedParameter>,
    /// Explicit tensor, layout, or dtype transformations.
    pub transformed: Vec<TransformedParameter>,
    /// Source parameters not used by the resulting experiment.
    pub omitted: Vec<ParameterDisposition>,
    /// Target parameters retaining their deliberately constructed initialization.
    pub newly_initialized: Vec<ParameterDisposition>,
    /// Resulting parameter path groups that still share one parameter identity.
    pub preserved_ties: Vec<Vec<String>>,
    /// Affected path groups whose sharing was explicitly removed.
    pub untied: Vec<Vec<String>>,
}

impl TransferReport {
    /// Sort report entries for reproducible display and serialization.
    pub fn finalize(&mut self) {
        self.reused.sort_by(|a, b| a.target.cmp(&b.target));
        self.transformed.sort_by(|a, b| a.target.cmp(&b.target));
        self.omitted.sort_by(|a, b| a.name.cmp(&b.name));
        self.newly_initialized.sort_by(|a, b| a.name.cmp(&b.name));
        for groups in [&mut self.preserved_ties, &mut self.untied] {
            for group in groups.iter_mut() {
                group.sort();
                group.dedup();
            }
            groups.sort();
            groups.dedup();
        }
    }
}

/// Runtime metadata for one floating parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterInfo {
    /// Dot-separated Rust module path; a root `Param` is named `$`.
    pub name: String,
    /// Actual runtime dimensions, after any Burn parameter mapper.
    pub shape: Vec<usize>,
    /// Runtime floating-point dtype; `DType::QFloat` marks a packed checkpoint weight.
    pub dtype: DType,
    /// Whether the target parameter participates in differentiation.
    pub trainable: bool,
    /// First lexicographic path sharing this parameter, if this is an alias.
    pub tied_to: Option<String>,
}

impl ParameterInfo {
    /// Whether a [`TransferPlan`] may select this parameter as a source or target.
    ///
    /// Only positive-rank dense floating tensors are transferable. Packed
    /// quantized weights are listed for inspection but must be dequantized by an
    /// explicit conversion before they can be copied.
    pub fn is_transferable(&self) -> bool {
        !self.shape.is_empty() && !matches!(self.dtype, DType::QFloat(_))
    }
}

/// Discover parameter paths and ties before writing a transfer plan.
///
/// Inspection initializes lazy parameters to obtain their actual runtime layout,
/// but does not copy data to the host or change parameter values. Packed
/// quantized parameters are listed too, with [`ParameterInfo::is_transferable`]
/// false, so a loaded quantized checkpoint can be inspected before planning.
pub fn parameters<B: Backend, M: Module<B>>(module: &M) -> Result<Vec<ParameterInfo>> {
    Ok(collect(module)?
        .into_values()
        .map(|parameter| parameter.info)
        .collect())
}

/// Rank-independent dense tensor payload staged on its original device.
struct Value<B: Backend> {
    /// Flattened floating tensor; its element order is unchanged by staging.
    tensor: Tensor<B, 1>,
    /// Runtime dimensions restored before invoking transforms or installing values.
    shape: Vec<usize>,
}
/// Borrowable callback that explicitly transforms one staged tensor payload.
type Transform<'a, B> = Box<dyn Fn(&Value<B>) -> Result<Value<B>> + 'a>;

/// Parameter-selection rule before path validation and precedence resolution.
enum Rule<'a, B: Backend> {
    /// Copy one source path to one target path without a numerical change.
    Copy {
        /// Exact source parameter path.
        source: String,
        /// Exact target parameter path.
        target: String,
    },
    /// Match relative parameter paths below a pair of module prefixes.
    Prefix {
        /// Existing source module or parameter prefix.
        source: String,
        /// Existing target module or parameter prefix.
        target: String,
    },
    /// Copy parameters whose complete paths occur in both modules.
    Matching,
    /// Produce one target value through an explicit, auditable callback.
    Transform {
        /// Exact source parameter path consumed by the callback.
        source: String,
        /// Exact target parameter path receiving the callback result.
        target: String,
        /// Human-readable description recorded in the transfer report.
        operation: String,
        /// Rank-erased callback whose result must match the target layout.
        transform: Transform<'a, B>,
    },
}

// Selection precedence and transactional transfer planning.

/// Select parameters by discoverable module paths, then apply the plan transactionally.
/// Exact copy/transform rules override prefix rules; duplicate exact targets and
/// overlapping prefixes are rejected. Unselected target parameters remain initialized.
pub struct TransferPlan<'a, B: Backend> {
    /// Ordered rules; indices distinguish independently requested transforms.
    rules: Vec<Rule<'a, B>>,
    /// Target prefixes whose initialization takes precedence over all copy rules.
    keep: Vec<String>,
    /// Handling for direct copies with incompatible dimensions.
    shape_policy: ShapeChangePolicy,
    /// Whether transferred parameter identities retain their original sharing.
    tie_policy: TiePolicy,
}

impl<B: Backend> Default for TransferPlan<'_, B> {
    /// Creates an empty transfer plan with strict shape checks and alias preservation.
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, B: Backend> TransferPlan<'a, B> {
    /// Start with no selected parameters and strict shape/tie checks.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            keep: Vec::new(),
            shape_policy: ShapeChangePolicy::Reject,
            tie_policy: TiePolicy::Preserve,
        }
    }

    /// Copy one exact source parameter to one exact target parameter.
    pub fn copy(mut self, source: impl Into<String>, target: impl Into<String>) -> Self {
        self.rules.push(Rule::Copy {
            source: source.into(),
            target: target.into(),
        });
        self
    }

    /// Copy matching relative paths below two module prefixes.
    /// Prefixes match whole path components. Added target parameters with no
    /// corresponding source retain their initialization and appear in the report.
    pub fn copy_prefix(mut self, source: impl Into<String>, target: impl Into<String>) -> Self {
        self.rules.push(Rule::Prefix {
            source: source.into(),
            target: target.into(),
        });
        self
    }

    /// Copy every parameter whose complete path appears in both modules.
    pub fn copy_matching(mut self) -> Self {
        self.rules.push(Rule::Matching);
        self
    }

    /// Explicitly retain initialization below a target path, overriding copy rules.
    pub fn keep_initialized(mut self, target: impl Into<String>) -> Self {
        self.keep.push(target.into());
        self
    }

    /// Choose what happens when a direct copy encounters changed dimensions.
    pub fn on_shape_change(mut self, policy: ShapeChangePolicy) -> Self {
        self.shape_policy = policy;
        self
    }

    /// Preserve sharing by default, or explicitly make copied targets independent.
    pub fn with_ties(mut self, policy: TiePolicy) -> Self {
        self.tie_policy = policy;
        self
    }

    /// Apply an explicitly named, rank-checked tensor transformation.
    /// The callback may change shape, dtype, or device, but its result must exactly
    /// match the target. It runs before any target replacement and may borrow data.
    /// User callback side effects are outside the transactional parameter guarantee.
    pub fn transform<const I: usize, const O: usize>(
        mut self,
        source: impl Into<String>,
        target: impl Into<String>,
        operation: impl Into<String>,
        transform: impl Fn(Tensor<B, I>) -> Result<Tensor<B, O>> + 'a,
    ) -> Self {
        let operation = operation.into();
        self.rules.push(Rule::Transform {
            source: source.into(),
            target: target.into(),
            operation,
            transform: Box::new(move |value| {
                let shape: [usize; I] = value.shape.clone().try_into().map_err(|_| {
                    Error::configuration(
                        "transfer.transform",
                        "source rank differs from the callback rank",
                    )
                })?;
                if I == 0 || O == 0 {
                    return Err(Error::configuration(
                        "transfer.transform",
                        "parameter ranks must be positive",
                    ));
                }
                let tensor = transform(value.tensor.clone().reshape(shape))?;
                Ok(Value {
                    shape: tensor.dims().to_vec(),
                    tensor: tensor.flatten(0, O - 1),
                })
            }),
        });
        self
    }

    /// Validate every selected parameter, then return a modified target clone.
    /// Source and original target remain unchanged, including on transform failures.
    /// Dtype/device changes require an explicit [`Self::transform`] callback.
    pub fn apply<S: Module<B>, T: Module<B>>(
        &self,
        source: &S,
        target: &T,
    ) -> Result<(T, TransferReport)> {
        let source_names = parameter_names(source)?;
        let targets = collect(target)?;
        self.validate_rules(&source_names, &targets)?;
        let selections = targets
            .keys()
            .map(|name| {
                self.select(name, &source_names)
                    .map(|selection| (name.clone(), selection))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let selected_sources = selections
            .values()
            .filter_map(|selection| selection.as_ref().map(|(_, source)| source.clone()))
            .collect::<BTreeSet<_>>();
        let sources = collect_filtered(source, Some(&selected_sources))?;
        let mut report = TransferReport::default();
        let mut staged = BTreeMap::new();
        let mut used = std::collections::BTreeSet::new();
        for (name, target) in &targets {
            let Some((index, source_name)) = selections[name].clone() else {
                report.newly_initialized.push(ParameterDisposition::new(
                    name,
                    "not selected for transfer; retained target initialization",
                ));
                continue;
            };
            let Some(source) = sources.get(&source_name) else {
                report.newly_initialized.push(ParameterDisposition::new(
                    name,
                    format!("no source parameter `{source_name}`; retained target initialization"),
                ));
                continue;
            };
            let (Some(source_tensor), true) = (&source.tensor, target.info.is_transferable())
            else {
                return Err(Error::BackendCapability(format!(
                    "transfer.{name}: `{source_name}` -> `{name}` selects a packed or rank-zero parameter; only positive-rank dense floating parameters are supported, so dequantize explicitly before transferring"
                )));
            };
            let transformed = matches!(self.rules[index], Rule::Transform { .. });
            if !transformed && source.info.shape != target.info.shape {
                if self.shape_policy == ShapeChangePolicy::Reinitialize {
                    report.newly_initialized.push(ParameterDisposition::new(name, format!("source shape {:?} differs from target {:?}; retained target initialization", source.info.shape, target.info.shape)));
                    continue;
                }
                return Err(Error::shape(
                    format!("transfer.{name}"),
                    format!(
                        "{:?}; select Reinitialize or an explicit transform for changed dimensions",
                        target.info.shape
                    ),
                    format!("source {source_name}: {:?}", source.info.shape),
                ));
            }
            let mut value = Value {
                tensor: source_tensor.clone().detach(),
                shape: source.info.shape.clone(),
            };
            if let Rule::Transform {
                transform,
                operation,
                ..
            } = &self.rules[index]
            {
                value = transform(&value).map_err(|error| {
                    Error::configuration(
                        format!("transfer.{name}"),
                        format!("{operation}: {error}"),
                    )
                })?;
            }
            if value.shape != target.info.shape
                || value.tensor.dtype() != target.info.dtype
                || value.tensor.device() != target.device
            {
                return Err(Error::configuration(
                    format!("transfer.{name}"),
                    format!(
                        "replacement must match target shape {:?}, dtype {:?}, and device; found shape {:?}, dtype {:?}; dtype/device conversion must be explicit",
                        target.info.shape,
                        target.info.dtype,
                        value.shape,
                        value.tensor.dtype()
                    ),
                ));
            }
            used.insert(source_name.clone());
            if let Rule::Transform { operation, .. } = &self.rules[index] {
                report.transformed.push(TransformedParameter {
                    sources: vec![source_name.clone()],
                    target: name.clone(),
                    shape: value.shape.clone(),
                    operation: operation.clone(),
                });
            } else {
                report.reused.push(ReusedParameter::new(
                    &source_name,
                    name,
                    value.shape.clone(),
                ));
            }
            staged.insert(
                name.clone(),
                Staged {
                    value,
                    origin: source.id,
                    source: source_name,
                    transform: transformed.then_some(index),
                    id: ParamId::new(),
                    trainable: target.info.trainable,
                },
            );
        }
        for name in source_names.iter().filter(|name| !used.contains(*name)) {
            report.omitted.push(ParameterDisposition::new(
                name,
                "not reused by the selected target mappings",
            ));
        }
        resolve_ties(&targets, &mut staged, self.tie_policy, &mut report)?;
        let mut mapper = Install {
            path: Vec::new(),
            staged,
            shared: BTreeMap::new(),
            error: None,
        };
        let model = target.clone().map(&mut mapper);
        if let Some(error) = mapper.error {
            return Err(error);
        }
        if !mapper.staged.is_empty() {
            return Err(Error::configuration(
                "transfer.target",
                "module mapper did not visit all planned parameters",
            ));
        }
        report.finalize();
        Ok((model, report))
    }

    /// Validates source/target paths, explicit mappings, and transformation descriptions before staging.
    fn validate_rules(
        &self,
        sources: &BTreeSet<String>,
        targets: &BTreeMap<String, Parameter<B>>,
    ) -> Result<()> {
        let mut exact = std::collections::BTreeSet::new();
        for rule in &self.rules {
            match rule {
                Rule::Copy { source, target } | Rule::Transform { source, target, .. } => {
                    if !sources.contains(source) || !targets.contains_key(target) {
                        return Err(Error::configuration(
                            "transfer.mapping",
                            format!(
                                "exact mapping `{source}` -> `{target}` requires both parameter paths to exist"
                            ),
                        ));
                    }
                    if !exact.insert(target) {
                        return Err(Error::configuration(
                            target,
                            "duplicate exact transfer target",
                        ));
                    }
                }
                Rule::Prefix { source, target } => {
                    if source.is_empty()
                        || target.is_empty()
                        || !sources.iter().any(|name| under(name, source).is_some())
                        || !targets.keys().any(|name| under(name, target).is_some())
                    {
                        return Err(Error::configuration(
                            "transfer.prefix",
                            format!(
                                "prefix mapping `{source}` -> `{target}` requires nonempty existing module paths"
                            ),
                        ));
                    }
                }
                Rule::Matching => {
                    if !targets.keys().any(|name| sources.contains(name)) {
                        return Err(Error::configuration(
                            "transfer.copy_matching",
                            "the source and target have no matching parameter paths",
                        ));
                    }
                }
            }
            if let Rule::Prefix { source, target } = rule
                && !targets.keys().any(|name| {
                    under(name, target)
                        .is_some_and(|suffix| sources.contains(&format!("{source}{suffix}")))
                })
            {
                return Err(Error::configuration(
                    "transfer.prefix",
                    format!(
                        "prefix mapping `{source}` -> `{target}` has no matching relative parameter paths"
                    ),
                ));
            }
            if let Rule::Transform { operation, .. } = rule
                && operation.trim().is_empty()
            {
                return Err(Error::configuration(
                    "transfer.transform",
                    "describe the explicit transformation",
                ));
            }
        }
        for prefix in &self.keep {
            if prefix.is_empty() || !targets.keys().any(|name| under(name, prefix).is_some()) {
                return Err(Error::configuration(
                    "transfer.keep_initialized",
                    format!("unknown or empty target path `{prefix}`"),
                ));
            }
        }
        Ok(())
    }

    /// Resolves one target using keep-initialized, exact, then prefix/matching precedence.
    ///
    /// Multiple fallback matches are rejected, so rule order cannot silently choose
    /// between overlapping module prefixes. The precedence itself is the
    /// Verus-verified [`select_rule`]; this method classifies each rule by path.
    fn select(&self, target: &str, sources: &BTreeSet<String>) -> Result<Option<(usize, String)>> {
        let kept = self
            .keep
            .iter()
            .any(|prefix| under(target, prefix).is_some());
        let matches = self
            .rules
            .iter()
            .map(|rule| match rule {
                Rule::Copy { target: name, .. } | Rule::Transform { target: name, .. }
                    if name == target =>
                {
                    RuleMatch::Exact
                }
                Rule::Prefix { target: prefix, .. } if under(target, prefix).is_some() => {
                    RuleMatch::Fallback
                }
                Rule::Matching if sources.contains(target) => RuleMatch::Fallback,
                _ => RuleMatch::Unrelated,
            })
            .collect::<Vec<_>>();
        match select_rule(kept, &matches) {
            RuleSelection::Unselected => Ok(None),
            RuleSelection::Overlapping => Err(Error::configuration(
                target,
                "overlapping transfer prefix rules; use one prefix or an exact override",
            )),
            RuleSelection::Rule(index) => Ok(self
                .rule_source(index, target)
                .map(|source| (index, source))),
        }
    }

    /// The source path that rule `index` supplies for `target`.
    ///
    /// `None` only for a prefix rule that does not cover `target`, which
    /// [`select_rule`] never selects.
    fn rule_source(&self, index: usize, target: &str) -> Option<String> {
        match &self.rules[index] {
            Rule::Copy { source, .. } | Rule::Transform { source, .. } => Some(source.clone()),
            Rule::Prefix {
                source,
                target: prefix,
            } => under(target, prefix).map(|suffix| format!("{source}{suffix}")),
            Rule::Matching => Some(target.to_owned()),
        }
    }
}

/// Returns the suffix only for an exact module path or a dot-delimited descendant.
fn under<'a>(name: &'a str, prefix: &str) -> Option<&'a str> {
    name.strip_prefix(prefix)
        .filter(|suffix| suffix.is_empty() || suffix.starts_with('.'))
}

// Lazy discovery, value staging, and alias-aware installation.

/// Discovered parameter metadata and tensor handle used to validate a transfer.
struct Parameter<B: Backend> {
    /// Named runtime layout, differentiation status, and alias metadata.
    info: ParameterInfo,
    /// Burn identity used to distinguish aliases from numerically equal values.
    id: ParamId,
    /// Device on which any replacement must already reside.
    device: B::Device,
    /// Flattened dense value, without a host copy; absent for packed or rank-zero
    /// parameters, which are listed but never staged.
    tensor: Option<Tensor<B, 1>>,
}

/// Visitor that discovers parameter paths without initializing lazy tensors.
#[derive(Default)]
struct Names {
    /// Module segments from the root to the current parameter.
    path: Vec<String>,
    /// Distinct paths discovered so far, ordered for deterministic selection.
    names: BTreeSet<String>,
    /// Duplicate-path failure retained until module traversal completes.
    error: Option<Error>,
}

impl<B: Backend> ModuleVisitor<B> for Names {
    /// Pushes the module segment used to discover checkpoint-independent parameter names.
    fn enter_module(&mut self, name: &str, _: &str) {
        self.path.push(name.into());
    }
    /// Restores the parent path before visiting a sibling module.
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }
    /// Collects a parameter path without forcing lazy tensor initialization, rejecting duplicate paths.
    fn visit_float<const D: usize>(&mut self, _: &Param<Tensor<B, D>>) {
        let name = path_name(&self.path);
        if !self.names.insert(name.clone()) {
            self.error = Some(Error::configuration(
                name,
                "duplicate module parameter path",
            ));
        }
    }
}

/// Discovers unique float-parameter paths using Burn traversal and propagates path collisions.
fn parameter_names<B: Backend, M: Module<B>>(module: &M) -> Result<BTreeSet<String>> {
    let mut visitor = Names::default();
    module.visit(&mut visitor);
    match visitor.error {
        Some(error) => Err(error),
        None => Ok(visitor.names),
    }
}

/// Visitor that materializes only selected dense parameter handles and metadata.
struct Collect<'a, B: Backend> {
    /// Module segments from the root to the current parameter.
    path: Vec<String>,
    /// Validated parameter entries indexed by their complete paths.
    parameters: BTreeMap<String, Parameter<B>>,
    /// First capability or path error, after which collection stops.
    error: Option<Error>,
    /// Optional path subset; absent selection requests all floating parameters.
    selected: Option<&'a BTreeSet<String>>,
}

/// Formats a dot-delimited module path, using `$` for a parameter at the root.
fn path_name(path: &[String]) -> String {
    if path.is_empty() {
        "$".into()
    } else {
        path.join(".")
    }
}

impl<B: Backend> ModuleVisitor<B> for Collect<'_, B> {
    /// Pushes the module segment used to collect selected parameter metadata and values.
    fn enter_module(&mut self, name: &str, _: &str) {
        self.path.push(name.into());
    }
    /// Restores the parent path after collecting a nested module.
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }
    /// Collects selected float parameters with their runtime shape, dtype, and trainability.
    ///
    /// Positive-rank dense tensors are flattened on device for rank-independent
    /// staging. Packed quantized and rank-zero tensors keep only their metadata:
    /// they are never implicitly dequantized, and selecting one fails at staging.
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if self.error.is_some() {
            return;
        }
        let name = path_name(&self.path);
        if self.selected.is_some_and(|names| !names.contains(&name)) {
            return;
        }
        let value = param.val();
        let info = ParameterInfo {
            name: name.clone(),
            shape: value.dims().to_vec(),
            dtype: value.dtype(),
            trainable: value.is_require_grad(),
            tied_to: None,
        };
        let parameter = Parameter {
            id: param.id,
            device: value.device(),
            tensor: info
                .is_transferable()
                .then(|| value.flatten(0, D.saturating_sub(1))),
            info,
        };
        if self.parameters.insert(name.clone(), parameter).is_some() {
            self.error = Some(Error::configuration(
                name,
                "duplicate module parameter path",
            ));
        }
    }
}

/// Collects every dense float parameter and identifies shared parameter identities.
fn collect<B: Backend, M: Module<B>>(module: &M) -> Result<BTreeMap<String, Parameter<B>>> {
    collect_filtered(module, None)
}

/// Collects an optional path subset and labels aliases by their first sorted owner.
///
/// Unselected tensors remain lazy; identity tracking never infers ties merely from
/// equal numerical values.
fn collect_filtered<B: Backend, M: Module<B>>(
    module: &M,
    selected: Option<&BTreeSet<String>>,
) -> Result<BTreeMap<String, Parameter<B>>> {
    let mut visitor = Collect {
        path: Vec::new(),
        parameters: BTreeMap::new(),
        error: None,
        selected,
    };
    module.visit(&mut visitor);
    if let Some(error) = visitor.error {
        return Err(error);
    }
    let mut owners = BTreeMap::<ParamId, String>::new();
    for (name, parameter) in &mut visitor.parameters {
        if let Some(owner) = owners.get(&parameter.id) {
            parameter.info.tied_to = Some(owner.clone());
        } else {
            owners.insert(parameter.id, name.clone());
        }
    }
    Ok(visitor.parameters)
}

/// Validated replacement awaiting alias resolution and transactional installation.
struct Staged<B: Backend> {
    /// Replacement tensor and target-compatible runtime dimensions.
    value: Value<B>,
    /// Source parameter identity, retained to detect shared origins.
    origin: ParamId,
    /// Source parameter path; one path copied to several targets is fan-out, not a tie.
    source: String,
    /// Transform rule index, or `None` for a direct copy.
    transform: Option<usize>,
    /// Fresh output identity, unified only when tie validation succeeds.
    id: ParamId,
    /// Target differentiation policy applied to the new gradient leaf.
    trainable: bool,
}

/// Validates alias groups and assigns a common staged identity when preserving ties.
///
/// Two kinds of existing sharing are preserved under [`TiePolicy::Preserve`]:
///
/// 1. *Target ties*: target paths that already share one identity must all be
///    selected from the same source with the same transformation.
/// 2. *Source ties*: distinct source paths sharing one identity map to targets
///    that share one fresh identity. Their values must agree in shape, dtype,
///    device, trainability, and every element.
///
/// One source path copied to several targets is *fan-out*: each copy keeps its
/// independent fresh identity, may use its own transformation, and is never
/// reported as a tie. Fan-out of a path that is also source-tied is ambiguous
/// (joining the copies would contradict the request for several copies) and
/// is rejected unless the plan explicitly unties. Each preserved source tie
/// synchronizes one Boolean equality reduction; explicit untying synchronizes nothing.
fn resolve_ties<B: Backend>(
    targets: &BTreeMap<String, Parameter<B>>,
    staged: &mut BTreeMap<String, Staged<B>>,
    policy: TiePolicy,
    report: &mut TransferReport,
) -> Result<()> {
    validate_target_ties(targets, staged, policy, report)?;
    // Mutable map iteration yields disjoint entries. Keep those references in
    // each group through validation and identity assignment, avoiding a second
    // lookup whose success would otherwise have to be asserted.
    let mut source_groups = BTreeMap::<ParamId, Vec<StagedEntry<'_, B>>>::new();
    for (name, value) in staged.iter_mut() {
        source_groups
            .entry(value.origin)
            .or_default()
            .push((name, value));
    }
    for group in source_groups.into_values() {
        let paths = group
            .iter()
            .map(|(_, value)| value.source.as_str())
            .collect::<BTreeSet<_>>();
        if paths.len() == 1 {
            // Fan-out copies stay independent; only pre-existing target ties are shared.
            if policy == TiePolicy::Preserve {
                share_target_tie_classes(targets, group, report);
            }
            continue;
        }
        let names = group
            .iter()
            .map(|(name, _)| (*name).clone())
            .collect::<Vec<_>>();
        if policy == TiePolicy::Untie {
            report.untied.push(names);
            continue;
        }
        reject_fanned_out_source_tie(targets, &group, &names)?;
        share_source_tie(group, names, report)?;
    }
    Ok(())
}

/// One staged replacement together with its target path, borrowed disjointly.
type StagedEntry<'a, B> = (&'a String, &'a mut Staged<B>);

/// Requires every selected target alias to receive the same source and transformation.
///
/// A target tie group with no selected member keeps its identity and is reported
/// as preserved. Under [`TiePolicy::Untie`], partially or fully selected groups are
/// reported as untied; otherwise every member must be selected consistently. The
/// classification is the Verus-verified [`target_tie`].
fn validate_target_ties<B: Backend>(
    targets: &BTreeMap<String, Parameter<B>>,
    staged: &BTreeMap<String, Staged<B>>,
    policy: TiePolicy,
    report: &mut TransferReport,
) -> Result<()> {
    // Carry the checked lookup with each target path so alias validation cannot
    // accidentally assume that every original alias was selected for transfer.
    let mut original_groups = BTreeMap::<ParamId, Vec<_>>::new();
    for (name, target) in targets {
        original_groups
            .entry(target.id)
            .or_default()
            .push((name, staged.get(name)));
    }
    for group in original_groups
        .into_values()
        .filter(|group| group.len() > 1)
    {
        let names = group
            .iter()
            .map(|(name, _)| (*name).clone())
            .collect::<Vec<_>>();
        // `ParamId` equality is equality of its `val()`, so the verified
        // classification sees exactly the identities compared here.
        let members = group
            .iter()
            .map(|(_, selected)| {
                selected.map(|value| StagedKey {
                    origin: value.origin.val(),
                    transform: value.transform,
                    trainable: value.trainable,
                })
            })
            .collect::<Vec<_>>();
        match target_tie(&members, policy == TiePolicy::Untie) {
            TargetTie::Untouched => report.preserved_ties.push(names),
            TargetTie::Untied => report.untied.push(names),
            TargetTie::Consistent => {}
            TargetTie::Inconsistent => {
                return Err(Error::configuration(
                    "transfer.ties",
                    format!(
                        "target aliases {names:?} require the same source and transformation; select TiePolicy::Untie to separate them"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Shares one identity inside each pre-existing target tie of a fan-out group.
///
/// All entries come from one source path. [`validate_target_ties`] already
/// required each target tie to use one transformation, so members of a class
/// hold identical values and need no equality reduction. The verified
/// [`class_leaders`] gives each entry the first member of its target class:
/// every member takes that leader's fresh identity, so each class is joined
/// and distinct classes stay independent; a singleton keeps its own identity.
fn share_target_tie_classes<B: Backend>(
    targets: &BTreeMap<String, Parameter<B>>,
    group: Vec<StagedEntry<'_, B>>,
    report: &mut TransferReport,
) {
    let classes = group
        .iter()
        .map(|(name, _)| targets[*name].id.val())
        .collect::<Vec<_>>();
    let leaders = class_leaders(&classes);
    let identities = group.iter().map(|(_, value)| value.id).collect::<Vec<_>>();
    let mut members = BTreeMap::<usize, Vec<String>>::new();
    for ((name, value), leader) in group.into_iter().zip(leaders) {
        value.id = identities[leader];
        members.entry(leader).or_default().push(name.clone());
    }
    report
        .preserved_ties
        .extend(members.into_values().filter(|names| names.len() > 1));
}

/// Rejects a source tie whose member path is also copied to independent targets.
///
/// Preserving the source tie would join every target in the group, including
/// copies the plan requested as separate parameters. Each source path must
/// therefore reach exactly one target tie class (one target, or one existing
/// target tie group); the verified [`first_fanned_out_path`] checks that the
/// path-to-class map is a function.
fn reject_fanned_out_source_tie<B: Backend>(
    targets: &BTreeMap<String, Parameter<B>>,
    group: &[StagedEntry<'_, B>],
    names: &[String],
) -> Result<()> {
    // Dense indices into the sorted distinct paths make the verified core
    // report the lexicographically first fanned-out path.
    let distinct = group
        .iter()
        .map(|(_, value)| value.source.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let paths = group
        .iter()
        .map(|(_, value)| {
            distinct
                .binary_search(&value.source.as_str())
                .unwrap_or_else(|insertion| insertion)
        })
        .collect::<Vec<_>>();
    let classes = group
        .iter()
        .map(|(name, _)| targets[*name].id.val())
        .collect::<Vec<_>>();
    match first_fanned_out_path(&paths, &classes, distinct.len()).map(|path| distinct[path]) {
        Some(path) => Err(Error::configuration(
            "transfer.ties",
            format!(
                "source `{path}` is copied to several independent targets but also shares its parameter identity with another source path; preserving that tie would join the copies in {names:?}. Select TiePolicy::Untie, or copy from one source alias only"
            ),
        )),
        None => Ok(()),
    }
}

/// Validates a genuine source tie and assigns its targets one shared identity.
///
/// Members must agree in transformation, runtime layout, and trainability before
/// their values are compared. One fallible readback checks every element of
/// every distinct source path against the first member.
fn share_source_tie<B: Backend>(
    mut group: Vec<StagedEntry<'_, B>>,
    names: Vec<String>,
    report: &mut TransferReport,
) -> Result<()> {
    let Some(((_, first), aliases)) = group.split_first_mut() else {
        return Ok(());
    };
    if aliases.iter().any(|(_, value)| {
        value.transform != first.transform
            || value.value.shape != first.value.shape
            || value.value.tensor.dtype() != first.value.tensor.dtype()
            || value.value.tensor.device() != first.value.tensor.device()
            || value.trainable != first.trainable
    }) {
        return Err(Error::configuration(
            "transfer.ties",
            format!(
                "source aliases mapped to {names:?} have conflicting transforms, shapes, or trainability; select TiePolicy::Untie"
            ),
        ));
    }
    // Entries copied from the first member's own path are the same tensor; only
    // other source paths can hold diverged values.
    let all_equal = aliases
        .iter()
        .filter(|(_, value)| value.source != first.source)
        .map(|(_, value)| {
            first
                .value
                .tensor
                .clone()
                .equal(value.value.tensor.clone())
                .all()
        })
        .reduce(|left, right| left.bool_and(right));
    if let Some(all_equal) = all_equal {
        let equal = all_equal
            .int()
            .try_into_scalar()
            .map_err(|error| {
                Error::BackendCapability(format!(
                    "cannot compare tied source values for {names:?}: {error}"
                ))
            })?
            .elem::<i64>()
            != 0;
        if !equal {
            return Err(Error::configuration(
                "transfer.ties",
                format!(
                    "source parameters mapped to {names:?} share a parameter identity but have unequal values; select TiePolicy::Untie to retain both values"
                ),
            ));
        }
    }
    let shared_id = first.id;
    for (_, value) in aliases {
        value.id = shared_id;
    }
    report.preserved_ties.push(names);
    Ok(())
}

/// Mapper that installs staged tensors into a cloned target after validation.
struct Install<B: Backend> {
    /// Module segments identifying the parameter currently being mapped.
    path: Vec<String>,
    /// Validated replacements removed as the target mapper consumes them.
    staged: BTreeMap<String, Staged<B>>,
    /// First installed tensor for each preserved identity, erased over rank.
    shared: BTreeMap<ParamId, Box<dyn Any>>,
    /// Mapping failure retained so the partially modified clone can be discarded.
    error: Option<Error>,
}

impl<B: Backend> ModuleMapper<B> for Install<B> {
    /// Pushes the target path for transactional installation of prepared values.
    fn enter_module(&mut self, name: &str, _: &str) {
        self.path.push(name.into());
    }
    /// Restores the parent target path after mapping a nested module.
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }
    /// Installs one validated tensor while retaining the target parameter's record mapper.
    ///
    /// Values begin a fresh gradient graph with the target freezing policy. Shared
    /// identities reuse one tensor handle; a rank mismatch records an error and leaves
    /// that parameter unchanged so the caller can discard the staged clone.
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let name = path_name(&self.path);
        let Some(staged) = self.staged.remove(&name) else {
            return param;
        };
        let Ok(shape): std::result::Result<[usize; D], _> = staged.value.shape.clone().try_into()
        else {
            self.error = Some(Error::configuration(
                name,
                "module mapper rank changed after transfer validation",
            ));
            return param;
        };
        let tensor = if let Some(shared) = self.shared.get(&staged.id) {
            let Some(tensor) = shared.downcast_ref::<Tensor<B, D>>() else {
                self.error = Some(Error::configuration(name, "tied parameter ranks disagree"));
                return param;
            };
            tensor.clone()
        } else {
            let tensor = staged
                .value
                .tensor
                .reshape(shape)
                .detach()
                .set_require_grad(staged.trainable);
            self.shared.insert(staged.id, Box::new(tensor.clone()));
            tensor
        };
        let (_, _, mapper) = param.consume();
        Param::from_mapped_value(staged.id, tensor, mapper)
    }
}

#[cfg(test)]
#[path = "transfer_tests.rs"]
/// Transfer behavior, alias invariants, and transactional failure regressions.
mod tests;
