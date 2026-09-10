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
//! validating a tied group synchronizes a Boolean equality reduction.
//! copying neither casts dtype nor moves devices. Copied parameters start a new
//! gradient graph, preserve the target's trainability, and retain ties by default.
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TiePolicy {
    /// Preserve source sharing and existing target ties; reject conflicting mappings.
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

/// Runtime metadata for one dense floating parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterInfo {
    /// Dot-separated Rust module path; a root `Param` is named `$`.
    pub name: String,
    /// Actual runtime dimensions, after any Burn parameter mapper.
    pub shape: Vec<usize>,
    /// Runtime floating-point dtype.
    pub dtype: DType,
    /// Whether the target parameter participates in differentiation.
    pub trainable: bool,
    /// First lexicographic path sharing this parameter, if this is an alias.
    pub tied_to: Option<String>,
}

/// Discover parameter paths and ties before writing a transfer plan.
/// Inspection initializes lazy parameters to obtain their actual runtime layout,
/// but does not copy data to the host or change parameter values.
pub fn parameters<B: Backend, M: Module<B>>(module: &M) -> Result<Vec<ParameterInfo>> {
    Ok(collect(module)?
        .into_values()
        .map(|parameter| parameter.info)
        .collect())
}

struct Value<B: Backend> {
    tensor: Tensor<B, 1>,
    shape: Vec<usize>,
}
type Transform<'a, B> = Box<dyn Fn(&Value<B>) -> Result<Value<B>> + 'a>;

enum Rule<'a, B: Backend> {
    Copy {
        source: String,
        target: String,
    },
    Prefix {
        source: String,
        target: String,
    },
    Matching,
    Transform {
        source: String,
        target: String,
        operation: String,
        transform: Transform<'a, B>,
    },
}

/// Select parameters by discoverable module paths, then apply the plan transactionally.
/// Exact copy/transform rules override prefix rules; duplicate exact targets and
/// overlapping prefixes are rejected. Unselected target parameters remain initialized.
pub struct TransferPlan<'a, B: Backend> {
    rules: Vec<Rule<'a, B>>,
    keep: Vec<String>,
    shape_policy: ShapeChangePolicy,
    tie_policy: TiePolicy,
}

impl<B: Backend> Default for TransferPlan<'_, B> {
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
                tensor: source.tensor.clone().detach(),
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

    fn select(&self, target: &str, sources: &BTreeSet<String>) -> Result<Option<(usize, String)>> {
        if self
            .keep
            .iter()
            .any(|prefix| under(target, prefix).is_some())
        {
            return Ok(None);
        }
        let exact = self
            .rules
            .iter()
            .enumerate()
            .find_map(|(index, rule)| match rule {
                Rule::Copy {
                    source,
                    target: name,
                }
                | Rule::Transform {
                    source,
                    target: name,
                    ..
                } if name == target => Some((index, source.clone())),
                _ => None,
            });
        if exact.is_some() {
            return Ok(exact);
        }
        let matches = self
            .rules
            .iter()
            .enumerate()
            .filter_map(|(index, rule)| match rule {
                Rule::Prefix {
                    source,
                    target: prefix,
                } => under(target, prefix).map(|suffix| (index, format!("{source}{suffix}"))),
                Rule::Matching if sources.contains(target) => Some((index, target.to_owned())),
                _ => None,
            })
            .collect::<Vec<_>>();
        if matches.len() > 1 {
            return Err(Error::configuration(
                target,
                "overlapping transfer prefix rules; use one prefix or an exact override",
            ));
        }
        Ok(matches.into_iter().next())
    }
}

fn under<'a>(name: &'a str, prefix: &str) -> Option<&'a str> {
    name.strip_prefix(prefix)
        .filter(|suffix| suffix.is_empty() || suffix.starts_with('.'))
}

struct Parameter<B: Backend> {
    info: ParameterInfo,
    id: ParamId,
    device: B::Device,
    tensor: Tensor<B, 1>,
}

#[derive(Default)]
struct Names {
    path: Vec<String>,
    names: BTreeSet<String>,
    error: Option<Error>,
}

impl<B: Backend> ModuleVisitor<B> for Names {
    fn enter_module(&mut self, name: &str, _: &str) {
        self.path.push(name.into());
    }
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }
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

fn parameter_names<B: Backend, M: Module<B>>(module: &M) -> Result<BTreeSet<String>> {
    let mut visitor = Names::default();
    module.visit(&mut visitor);
    match visitor.error {
        Some(error) => Err(error),
        None => Ok(visitor.names),
    }
}

struct Collect<'a, B: Backend> {
    path: Vec<String>,
    parameters: BTreeMap<String, Parameter<B>>,
    error: Option<Error>,
    selected: Option<&'a BTreeSet<String>>,
}

fn path_name(path: &[String]) -> String {
    if path.is_empty() {
        "$".into()
    } else {
        path.join(".")
    }
}

impl<B: Backend> ModuleVisitor<B> for Collect<'_, B> {
    fn enter_module(&mut self, name: &str, _: &str) {
        self.path.push(name.into());
    }
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if self.error.is_some() {
            return;
        }
        let name = path_name(&self.path);
        if self.selected.is_some_and(|names| !names.contains(&name)) {
            return;
        }
        let value = param.val();
        if D == 0 || matches!(value.dtype(), DType::QFloat(_)) {
            self.error = Some(Error::BackendCapability(format!(
                "transfer.{name}: only positive-rank dense floating parameters are supported"
            )));
            return;
        }
        let parameter = Parameter {
            info: ParameterInfo {
                name: name.clone(),
                shape: value.dims().to_vec(),
                dtype: value.dtype(),
                trainable: value.is_require_grad(),
                tied_to: None,
            },
            id: param.id,
            device: value.device(),
            tensor: value.flatten(0, D - 1),
        };
        if self.parameters.insert(name.clone(), parameter).is_some() {
            self.error = Some(Error::configuration(
                name,
                "duplicate module parameter path",
            ));
        }
    }
}

fn collect<B: Backend, M: Module<B>>(module: &M) -> Result<BTreeMap<String, Parameter<B>>> {
    collect_filtered(module, None)
}

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

struct Staged<B: Backend> {
    value: Value<B>,
    origin: ParamId,
    transform: Option<usize>,
    id: ParamId,
    trainable: bool,
}

fn resolve_ties<B: Backend>(
    targets: &BTreeMap<String, Parameter<B>>,
    staged: &mut BTreeMap<String, Staged<B>>,
    policy: TiePolicy,
    report: &mut TransferReport,
) -> Result<()> {
    let mut original_groups = BTreeMap::<ParamId, Vec<String>>::new();
    for (name, target) in targets {
        original_groups
            .entry(target.id)
            .or_default()
            .push(name.clone());
    }
    for group in original_groups.values().filter(|group| group.len() > 1) {
        let selected = group
            .iter()
            .filter_map(|name| staged.get(name))
            .collect::<Vec<_>>();
        if selected.is_empty() {
            report.preserved_ties.push(group.clone());
            continue;
        }
        if policy == TiePolicy::Untie {
            report.untied.push(group.clone());
            continue;
        }
        if selected.len() != group.len()
            || selected.iter().any(|value| {
                value.origin != selected[0].origin
                    || value.transform != selected[0].transform
                    || value.trainable != selected[0].trainable
            })
        {
            return Err(Error::configuration(
                "transfer.ties",
                format!(
                    "target aliases {group:?} require the same source and transformation; select TiePolicy::Untie to separate them"
                ),
            ));
        }
    }
    let mut source_groups = BTreeMap::<ParamId, Vec<String>>::new();
    for (name, value) in staged.iter() {
        source_groups
            .entry(value.origin)
            .or_default()
            .push(name.clone());
    }
    for group in source_groups.values().filter(|group| group.len() > 1) {
        if policy == TiePolicy::Untie {
            report.untied.push(group.clone());
            continue;
        }
        let first = &staged[&group[0]];
        if group.iter().any(|name| {
            staged[name].transform != first.transform
                || staged[name].value.shape != first.value.shape
                || staged[name].value.tensor.dtype() != first.value.tensor.dtype()
                || staged[name].value.tensor.device() != first.value.tensor.device()
                || staged[name].trainable != first.trainable
        }) {
            return Err(Error::configuration(
                "transfer.ties",
                format!(
                    "source aliases mapped to {group:?} have conflicting transforms, shapes, or trainability; select TiePolicy::Untie"
                ),
            ));
        }
        for name in group.iter().skip(1) {
            let equal = first
                .value
                .tensor
                .clone()
                .equal(staged[name].value.tensor.clone())
                .all()
                .int()
                .into_scalar()
                .elem::<i64>()
                != 0;
            if !equal {
                return Err(Error::configuration(
                    "transfer.ties",
                    format!(
                        "source parameters mapped to {group:?} share a parameter identity but have unequal values; select TiePolicy::Untie to retain both values"
                    ),
                ));
            }
        }
        let shared_id = first.id;
        for name in group {
            staged.get_mut(name).unwrap().id = shared_id;
        }
        report.preserved_ties.push(group.clone());
    }
    Ok(())
}

struct Install<B: Backend> {
    path: Vec<String>,
    staged: BTreeMap<String, Staged<B>>,
    shared: BTreeMap<ParamId, Box<dyn Any>>,
    error: Option<Error>,
}

impl<B: Backend> ModuleMapper<B> for Install<B> {
    fn enter_module(&mut self, name: &str, _: &str) {
        self.path.push(name.into());
    }
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }
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
mod tests;
