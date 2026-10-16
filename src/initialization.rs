//! Explicit parameter initialization for randomly initialized experiments.
//!
//! Architecture comparisons from random initialization are only meaningful
//! when every candidate starts from a deliberate, documented distribution. This
//! module states that distribution as data ([`InitPolicy`]) and applies it to
//! any Burn module by walking its parameters.
//!
//! # Burn's defaults, and why tied embeddings need care
//!
//! Burn initializes `Linear` weights with Kaiming-uniform (gain `1/sqrt(3)`):
//! `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`, standard deviation `1/sqrt(3 fan_in)`.
//! `Embedding` weights are `N(0, 1)`.
//!
//! With tied input and output embeddings, the logits are `z_v = <h, E_v>`,
//! where `h` is the final RMS-normalized state. RMS normalization makes every
//! channel of `h` unit scale, so `||h||^2 ~ d`. If each row `E_v` has
//! independent `N(0, sigma^2)` entries, then `Var(z_v) = sigma^2 ||h||^2`, and
//!
//! ```text
//! std(z) ~ sigma * sqrt(d).
//! ```
//!
//! (The input token's own logit is larger still, since `h` contains its
//! embedding `E_t`; the estimate covers the other `V - 1` logits.)
//!
//! Burn's `sigma = 1` gives `std(z) ~ sqrt(d)`: about 22.6 at `d = 512`, and
//! the input token's own logit is an outlier of order `d`. The softmax is then
//! nearly one-hot, so the initial cross entropy is far above the
//! uniform-prediction value `ln V`. Measured on a one-layer modular model with
//! `d = 256` and `V = 512` (`ln V = 6.24`): Burn's defaults give logit std 15.9
//! and an initial loss of about 250 nats; `sigma = 0.02`, the common Hugging
//! Face `initializer_range`, gives std 0.32 and 6.33 nats. Training from the
//! defaults starts far from equilibrium, and architecture comparisons are
//! confounded by the recovery phase.
//!
//! # Depth-scaled residual projections
//!
//! Each block adds two branch outputs (mixer and feed-forward) to the residual
//! stream. With `L` blocks, `2L` roughly independent terms accumulate, so the
//! residual variance at initialization grows like `2L * sigma_out^2`. GPT-2
//! scales the projections that write into the residual stream by
//! `1/sqrt(2L)`, keeping that variance independent of depth.
//! [`ResidualScaling::DepthScaled`] applies this factor to the `Linear`
//! weights named in [`InitPolicy::residual_projections`] (for example
//! `o_proj` and `down_proj`).
//!
//! # Roles and precedence
//!
//! Each floating parameter receives a [`ParameterRole`] from the Burn module
//! that owns it: `Embedding.weight`, `Linear.weight` (a residual projection
//! when the `Linear` field's name is listed), `Linear.bias`, or anything else
//! (normalization scales, position biases, raw parameters). The role selects
//! an initializer; [`InitOverride`]s matching the parameter path then take
//! precedence, the last match winning. [`ParameterInit::Keep`] leaves a
//! parameter exactly as constructed (still lazy, if it was).
//!
//! Besides `N(0, std^2)` and zeros, a parameter can be set to ones or to any
//! finite constant ([`ParameterInit::Ones`], [`ParameterInit::Constant`]), so
//! a policy can also reset normalization scales, for example
//! `.with_override("final_norm", ParameterInit::Ones)`.
//!
//! Names are checked against the module before anything changes: every
//! override must match at least one floating parameter, and every listed
//! residual projection must name at least one `Linear` field. A misspelled
//! path or a family-specific name (`c_proj` against a model that calls it
//! `o_proj`) is therefore an error, not a silently skipped rule.
//!
//! # Contracts
//!
//! Replaced values keep their parameter identity, dtype, device, trainability
//! (the current tensor's `is_require_grad`), and Burn save/load mapper, so
//! records round-trip and optimizers see the same parameters. Each replacement
//! is a fresh autodiff leaf. Fields that share one identity receive one shared tensor.
//! Packed quantized parameters are always kept. Sampling uses the backend's
//! random generator: call `B::seed(&device, seed)` first for reproducible
//! draws.
//!
//! # Eager and deferred application
//!
//! [`InitPolicy::apply`] replaces every selected parameter immediately. A
//! freshly constructed Burn parameter is lazy, so `apply` first materializes
//! its constructed value, releases it, and then draws the replacement: every
//! replaced parameter is materialized afterwards (a [`ParameterInit::Keep`]
//! parameter stays lazy), and peak memory stays at one copy of each parameter.
//!
//! [`InitPolicy::apply_lazy`] instead keeps such parameters lazy and replaces
//! their deferred initializer, so nothing is computed at application time. A
//! parameter that is overwritten before its first read (for example by a
//! checkpoint record) is never drawn at all. At the first read, the deferred
//! initializer still runs Burn's constructed initializer first, because Burn
//! offers no public way to swap a lazy initializer while keeping the
//! parameter's save/load mapper: that draws and allocates Burn's default
//! value (consuming random numbers if the default is random), releases it,
//! and then draws the policy's value. Only the policy's value survives and
//! peak memory stays at one copy, but the default is still computed once. The
//! remaining tradeoffs are Burn's own lazy-initialization semantics: draws
//! happen in first-read order, and clones taken before the first read draw
//! independently. Tied identities are therefore always drawn at application
//! time, so every alias holds the same tensor.

use std::{
    any::Any,
    collections::{BTreeSet, HashMap, HashSet},
};

use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId},
    tensor::{DType, Distribution, Tensor, backend::Backend},
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// How one class of parameters is initialized.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParameterInit {
    /// Keep the constructed value (Burn's default or a loaded checkpoint).
    Keep,
    /// Independent `N(0, std^2)` entries.
    Normal {
        /// Positive, finite standard deviation.
        std: f64,
    },
    /// All zeros, for example for biases or zero-initialized output projections.
    Zeros,
    /// All ones, for example to reset normalization scales.
    Ones,
    /// Every entry equal to one finite value.
    Constant {
        /// Finite fill value.
        value: f64,
    },
}

impl ParameterInit {
    /// Checks that a normal standard deviation is positive and finite and a constant is finite.
    fn validate(self, field: &str) -> Result<()> {
        match self {
            Self::Normal { std } if !(std.is_finite() && std > 0.0) => Err(Error::configuration(
                field,
                format!(
                    "normal std must be finite and greater than zero, got {std}; use Zeros for a zero tensor"
                ),
            )),
            Self::Constant { value } if !value.is_finite() => Err(Error::configuration(
                field,
                format!("constant fill value must be finite, got {value}"),
            )),
            _ => Ok(()),
        }
    }

    /// Draws or fills a replacement with the shape, dtype, device, and trainability of `like`.
    ///
    /// `like` is released before the replacement is allocated, so replacing a
    /// parameter never holds two copies of it. `Keep` is resolved by callers
    /// and never reaches this function; it is treated as zeros only to keep
    /// the match total.
    fn draw<B: Backend, const D: usize>(self, like: Tensor<B, D>) -> Tensor<B, D> {
        let shape = like.shape();
        let device = like.device();
        let dtype = like.dtype();
        let trainable = like.is_require_grad();
        drop(like);
        let options = (&device, dtype);
        match self {
            Self::Normal { std } => Tensor::random(shape, Distribution::Normal(0.0, std), options),
            Self::Ones => Tensor::ones(shape, options),
            Self::Constant { value } => Tensor::full(shape, value, options),
            Self::Zeros | Self::Keep => Tensor::zeros(shape, options),
        }
        .set_require_grad(trainable)
    }

    /// Multiplies a normal standard deviation by `factor`; other variants are unchanged.
    ///
    /// Depth scaling shrinks the spread of a random draw; a deterministic fill
    /// (zeros, ones, or a constant) is never rescaled.
    fn scaled(self, factor: f64) -> Self {
        match self {
            Self::Normal { std } => Self::Normal { std: std * factor },
            other => other,
        }
    }
}

/// What a parameter does, inferred from the Burn module that owns it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterRole {
    /// The `weight` of a Burn `Embedding`.
    Embedding,
    /// The `weight` of a Burn `Linear` that does not write into the residual stream.
    Projection,
    /// The `weight` of a Burn `Linear` named in [`InitPolicy::residual_projections`].
    ResidualProjection,
    /// The `bias` of a Burn `Linear`.
    Bias,
    /// Any other parameter: normalization scales, position biases, raw tensors.
    Other,
}

/// Optional depth scaling for projections that write into the residual stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResidualScaling {
    /// Residual projections use the ordinary projection initializer.
    #[default]
    None,
    /// Multiply the residual projections' standard deviation by `1/sqrt(2 * layers)`.
    DepthScaled {
        /// Number of residual blocks `L`; must be positive.
        layers: usize,
    },
}

impl ResidualScaling {
    /// The multiplier applied to residual-projection standard deviations.
    pub fn factor(self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::DepthScaled { layers } => (2.0 * layers as f64).sqrt().recip(),
        }
    }
}

/// An initializer for one parameter path or every parameter below a module path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InitOverride {
    /// Exact parameter path (`lm_head.weight`) or module prefix (`layers.0`),
    /// matched on whole dot-separated components.
    pub path: String,
    /// Initializer used for matching parameters, before residual scaling.
    pub init: ParameterInit,
}

/// An explicit initialization policy for every floating parameter of a module.
///
/// ```
/// use burn::{backend::Flex, nn::LinearConfig};
/// use transformers::initialization::{InitPolicy, ParameterInit, ParameterRole};
///
/// let device = Default::default();
/// let linear = LinearConfig::new(64, 32).init::<Flex>(&device);
/// let (linear, report) = InitPolicy::normal(0.02).apply(linear)?;
/// assert_eq!(report.parameters.len(), 2);
/// let bias = report.parameters.iter().find(|p| p.role == ParameterRole::Bias).unwrap();
/// assert_eq!(bias.init, ParameterInit::Zeros);
/// # Ok::<(), transformers::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InitPolicy {
    /// Initializer for [`ParameterRole::Embedding`].
    pub embedding: ParameterInit,
    /// Initializer for [`ParameterRole::Projection`] and, before scaling,
    /// [`ParameterRole::ResidualProjection`].
    pub projection: ParameterInit,
    /// Initializer for [`ParameterRole::Bias`].
    pub bias: ParameterInit,
    /// Initializer for [`ParameterRole::Other`].
    pub other: ParameterInit,
    /// Scaling applied to residual projections' standard deviation.
    pub residual_scaling: ResidualScaling,
    /// Field names of `Linear` modules that write into the residual stream,
    /// such as `o_proj` and `down_proj`; each must exist in the module.
    pub residual_projections: Vec<String>,
    /// Path-specific initializers, applied after role rules; the last match
    /// wins, and each must match at least one parameter of the module.
    pub overrides: Vec<InitOverride>,
}

impl Default for InitPolicy {
    /// Keeps every constructed value, matching Burn's defaults.
    fn default() -> Self {
        Self::keep()
    }
}

impl InitPolicy {
    /// Keeps every parameter exactly as constructed.
    pub fn keep() -> Self {
        Self {
            embedding: ParameterInit::Keep,
            projection: ParameterInit::Keep,
            bias: ParameterInit::Keep,
            other: ParameterInit::Keep,
            residual_scaling: ResidualScaling::None,
            residual_projections: Vec::new(),
            overrides: Vec::new(),
        }
    }

    /// Hugging Face-style initialization: `N(0, std^2)` embeddings and
    /// projections, zero biases, and other parameters kept (normalization
    /// scales stay one). `std = 0.02` is the usual `initializer_range`.
    pub fn normal(std: f64) -> Self {
        Self {
            embedding: ParameterInit::Normal { std },
            projection: ParameterInit::Normal { std },
            bias: ParameterInit::Zeros,
            ..Self::keep()
        }
    }

    /// Adds GPT-2-style `1/sqrt(2 * layers)` scaling for the named residual projections.
    ///
    /// Each name is a `Linear` field name (`o_proj`, `down_proj`; GPT-2 uses
    /// `c_proj`); [`apply`](Self::apply) rejects a name that no `Linear` has.
    pub fn with_depth_scaling<I, S>(mut self, layers: usize, residual_projections: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.residual_scaling = ResidualScaling::DepthScaled { layers };
        self.residual_projections
            .extend(residual_projections.into_iter().map(Into::into));
        self
    }

    /// Initializes every parameter at or below `path` with `init`, overriding role rules.
    ///
    /// [`apply`](Self::apply) rejects a path that matches no floating parameter.
    pub fn with_override(mut self, path: impl Into<String>, init: ParameterInit) -> Self {
        self.overrides.push(InitOverride {
            path: path.into(),
            init,
        });
        self
    }

    /// Validates standard deviations, depth, and names before any parameter changes.
    pub fn validate(&self) -> Result<()> {
        for (field, init) in [
            ("init.embedding", self.embedding),
            ("init.projection", self.projection),
            ("init.bias", self.bias),
            ("init.other", self.other),
        ] {
            init.validate(field)?;
        }
        if self.residual_scaling == (ResidualScaling::DepthScaled { layers: 0 }) {
            return Err(Error::configuration(
                "init.residual_scaling.layers",
                "depth scaling needs at least one layer",
            ));
        }
        if self.residual_projections.iter().any(String::is_empty) {
            return Err(Error::configuration(
                "init.residual_projections",
                "names must be nonempty module field names",
            ));
        }
        for (index, entry) in self.overrides.iter().enumerate() {
            if entry.path.is_empty() {
                return Err(Error::configuration(
                    format!("init.overrides[{index}].path"),
                    "must name a parameter path or module prefix",
                ));
            }
            entry
                .init
                .validate(&format!("init.overrides[{index}].init"))?;
        }
        Ok(())
    }

    /// The initializer for a parameter at `path` with `role`, after overrides and scaling.
    ///
    /// The last matching override wins (the Verus-verified
    /// `experiment::precedence::last_match`); without one, the role rule applies.
    pub fn resolve(&self, path: &str, role: ParameterRole) -> ParameterInit {
        let matches = self
            .overrides
            .iter()
            .map(|entry| under(path, &entry.path))
            .collect::<Vec<_>>();
        let base = crate::experiment::precedence::last_match(&matches)
            .map_or_else(|| self.role_init(role), |index| self.overrides[index].init);
        if role == ParameterRole::ResidualProjection {
            base.scaled(self.residual_scaling.factor())
        } else {
            base
        }
    }

    /// The role-level initializer before overrides and residual scaling.
    fn role_init(&self, role: ParameterRole) -> ParameterInit {
        match role {
            ParameterRole::Embedding => self.embedding,
            ParameterRole::Projection | ParameterRole::ResidualProjection => self.projection,
            ParameterRole::Bias => self.bias,
            ParameterRole::Other => self.other,
        }
    }

    /// Validates the policy and checks that its names exist in `module`.
    ///
    /// Beyond [`validate`](Self::validate), every override path must match at
    /// least one floating parameter (exactly or as a component prefix), and
    /// every entry of [`residual_projections`](Self::residual_projections)
    /// must be the field name of at least one Burn `Linear`. The check only
    /// reads parameter paths, so lazy parameters stay unmaterialized and the
    /// module is left untouched, which makes it a cheap dry run before
    /// [`apply`](Self::apply) consumes the module.
    pub fn validate_for<B: Backend, M: Module<B>>(&self, module: &M) -> Result<()> {
        self.survey(module).map(|_| ())
    }

    /// Validates the policy against `module` and returns the parameter survey.
    ///
    /// This is the body of [`validate_for`](Self::validate_for); application
    /// reuses the survey's identity counts to recognize tied parameters.
    fn survey<B: Backend, M: Module<B>>(&self, module: &M) -> Result<Survey> {
        self.validate()?;
        let mut survey = Survey::default();
        module.visit(&mut survey);
        for (index, entry) in self.overrides.iter().enumerate() {
            if !survey.paths.iter().any(|path| under(path, &entry.path)) {
                return Err(Error::configuration(
                    format!("init.overrides[{index}].path"),
                    format!(
                        "`{}` matches no floating parameter of the module; paths are \
                         dot-separated field names such as `layers.0.o_proj.weight`",
                        entry.path
                    ),
                ));
            }
        }
        if let Some(missing) = self
            .residual_projections
            .iter()
            .find(|name| !survey.linear_fields.contains(*name))
        {
            let present = if survey.linear_fields.is_empty() {
                "none".to_owned()
            } else {
                survey
                    .linear_fields
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            return Err(Error::configuration(
                "init.residual_projections",
                format!(
                    "`{missing}` names no Linear module field; Linear fields present: {present}"
                ),
            ));
        }
        Ok(survey)
    }

    /// Checks the policy against `module`, then returns it with every parameter initialized.
    ///
    /// Errors from [`validate_for`](Self::validate_for) are raised before any
    /// parameter changes. The report lists every floating parameter with its
    /// role and the initializer actually used, sorted by path. Every replaced
    /// parameter is materialized on return (see the module documentation).
    pub fn apply<B: Backend, M: Module<B>>(&self, module: M) -> Result<(M, InitReport)> {
        self.initialize(module, false)
    }

    /// Like [`apply`](Self::apply), but parameters that are still lazy stay lazy.
    ///
    /// Use it to defer initialization to first use, for example
    /// `policy.apply_lazy(ModularCausalLm::init(spec, &device)?)` before
    /// loading a record. A lazy parameter's deferred initializer is replaced:
    /// a parameter that is overwritten (by a checkpoint record, say) before it
    /// is read is never drawn, and its first read computes Burn's constructed
    /// value, releases it, and draws the policy's value, so only the policy's
    /// value survives and peak memory is one copy of the parameter (see the
    /// module documentation). Already materialized parameters and tied
    /// identities are replaced immediately, exactly as by `apply`, and the
    /// report is the same.
    ///
    /// Draws happen in first-read order and clones taken before the first
    /// read draw independently, as with Burn's own lazy initialization; use
    /// [`apply`](Self::apply) when clones must be identical before first use.
    ///
    /// ```
    /// use burn::{backend::Flex, nn::LinearConfig};
    /// use transformers::initialization::{InitPolicy, ParameterInit};
    ///
    /// let device = Default::default();
    /// let linear = LinearConfig::new(4, 2).init::<Flex>(&device);
    /// let policy = InitPolicy::keep().with_override("weight", ParameterInit::Constant { value: 0.5 });
    /// let (linear, _) = policy.apply_lazy(linear)?;
    /// assert!(!linear.weight.is_initialized());
    /// assert_eq!(linear.weight.val().into_data().to_vec::<f32>().unwrap(), [0.5; 8]);
    /// # Ok::<(), transformers::Error>(())
    /// ```
    pub fn apply_lazy<B: Backend, M: Module<B>>(&self, module: M) -> Result<(M, InitReport)> {
        self.initialize(module, true)
    }

    /// Shared body of [`apply`](Self::apply) and [`apply_lazy`](Self::apply_lazy).
    ///
    /// With `defer_lazy`, untied parameters that are still lazy receive a
    /// replaced deferred initializer instead of an immediate draw.
    fn initialize<B: Backend, M: Module<B>>(
        &self,
        module: M,
        defer_lazy: bool,
    ) -> Result<(M, InitReport)> {
        let survey = self.survey(&module)?;
        let mut mapper = Initializer {
            policy: self,
            location: ModulePath::default(),
            shared: HashMap::new(),
            report: InitReport::default(),
            deferrable: if defer_lazy {
                survey.untied_identities()
            } else {
                HashSet::new()
            },
        };
        let module = module.map(&mut mapper);
        let mut report = mapper.report;
        report
            .parameters
            .sort_by(|left, right| left.path.cmp(&right.path));
        Ok((module, report))
    }
}

/// Whether `path` equals `prefix` or lies below it on a component boundary.
fn under(path: &str, prefix: &str) -> bool {
    path.strip_prefix(prefix)
        .is_some_and(|suffix| suffix.is_empty() || suffix.starts_with('.'))
}

/// One parameter's role and the initializer applied to it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InitializedParameter {
    /// Dot-separated module path; a root parameter is named `$`.
    pub path: String,
    /// Role inferred from the owning Burn module.
    pub role: ParameterRole,
    /// Initializer used, after overrides and residual scaling; packed weights are `Keep`.
    pub init: ParameterInit,
}

/// Auditable result of [`InitPolicy::apply`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InitReport {
    /// Every floating parameter, sorted by path.
    pub parameters: Vec<InitializedParameter>,
}

/// The initializer chosen for one parameter identity and its replacement tensor.
///
/// Fields sharing an identity are one parameter, so the first field visited
/// decides for all of them; otherwise an override could leave one alias kept
/// and another replaced under the same identity.
struct SharedChoice {
    /// Initializer applied to the identity.
    init: ParameterInit,
    /// Replacement tensor erased over rank, or `None` when the value was kept.
    tensor: Option<Box<dyn Any>>,
}

/// Field names and their container types from the module root to the current field.
///
/// Burn reports, on entering each field, the type of the struct that holds
/// it, so the last container of a parameter names its owning module type
/// (`Struct:Linear` for `linear.weight`).
#[derive(Debug, Default)]
struct ModulePath {
    /// Field names from the root to the current field.
    names: Vec<String>,
    /// Container type of each entered field, such as `Struct:Linear`.
    containers: Vec<String>,
}

impl ModulePath {
    /// Descends into field `name` held by a container of type `container`.
    fn enter(&mut self, name: &str, container: &str) {
        self.names.push(name.to_owned());
        self.containers.push(container.to_owned());
    }

    /// Returns to the parent field.
    fn exit(&mut self) {
        self.names.pop();
        self.containers.pop();
    }

    /// The dot-separated path of the current field; a root parameter is `$`.
    fn joined(&self) -> String {
        if self.names.is_empty() {
            "$".to_owned()
        } else {
            self.names.join(".")
        }
    }

    /// The field name of the `Linear` whose `weight` is the current field.
    ///
    /// `None` when the current field is not a `Linear` weight or when that
    /// `Linear` is the root module (it then has no field name).
    fn linear_weight_owner(&self) -> Option<&str> {
        let field = self.names.last().map(String::as_str);
        let owner = self.containers.last().map(String::as_str);
        match (owner, field) {
            (Some("Struct:Linear"), Some("weight")) => self
                .names
                .len()
                .checked_sub(2)
                .map(|index| self.names[index].as_str()),
            _ => None,
        }
    }

    /// Infers the current parameter's role from its owning module.
    fn role(&self, residual_projections: &[String]) -> ParameterRole {
        let field = self.names.last().map(String::as_str);
        let owner = self.containers.last().map(String::as_str);
        match (owner, field) {
            (Some("Struct:Embedding"), Some("weight")) => ParameterRole::Embedding,
            (Some("Struct:Linear"), Some("bias")) => ParameterRole::Bias,
            (Some("Struct:Linear"), Some("weight")) => {
                let residual = self
                    .linear_weight_owner()
                    .is_some_and(|name| residual_projections.iter().any(|listed| listed == name));
                if residual {
                    ParameterRole::ResidualProjection
                } else {
                    ParameterRole::Projection
                }
            }
            _ => ParameterRole::Other,
        }
    }
}

/// Read-only pre-pass that records which names a policy may refer to.
#[derive(Debug, Default)]
struct Survey {
    /// Position of the visitor in the module tree.
    location: ModulePath,
    /// Path of every floating parameter field, aliases included.
    paths: Vec<String>,
    /// Field names of the `Linear` modules that own a weight, sorted for messages.
    linear_fields: BTreeSet<String>,
    /// Number of fields holding each parameter identity; more than one means tied.
    identities: HashMap<ParamId, usize>,
}

impl Survey {
    /// Identities held by exactly one field, whose replacement may be deferred.
    fn untied_identities(&self) -> HashSet<ParamId> {
        self.identities
            .iter()
            .filter(|&(_, &fields)| fields == 1)
            .map(|(&id, _)| id)
            .collect()
    }
}

impl<B: Backend> ModuleVisitor<B> for Survey {
    /// Records the entered field.
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.location.enter(name, container_type);
    }

    /// Restores the parent field.
    fn exit_module(&mut self, _: &str, _: &str) {
        self.location.exit();
    }

    /// Records the parameter's path, identity, and, for a `Linear` weight, its owner's field name.
    ///
    /// Only names and identities are read, so a lazy parameter is not materialized.
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.paths.push(self.location.joined());
        *self.identities.entry(param.id).or_default() += 1;
        if let Some(field) = self.location.linear_weight_owner() {
            self.linear_fields.insert(field.to_owned());
        }
    }
}

/// Module mapper that classifies parameters by owner and replaces their values.
struct Initializer<'p> {
    /// Policy being applied.
    policy: &'p InitPolicy,
    /// Position of the mapper in the module tree.
    location: ModulePath,
    /// Decision taken at the first field of each identity, reused by its aliases.
    shared: HashMap<ParamId, SharedChoice>,
    /// Parameters visited so far.
    report: InitReport,
    /// Untied identities whose still-lazy parameters keep deferred initialization;
    /// empty for eager application.
    deferrable: HashSet<ParamId>,
}

impl<B: Backend> ModuleMapper<B> for Initializer<'_> {
    /// Records the field name and the type of the struct that holds it.
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.location.enter(name, container_type);
    }

    /// Restores the parent path after leaving a field.
    fn exit_module(&mut self, _: &str, _: &str) {
        self.location.exit();
    }

    /// Replaces one parameter according to its role, preserving identity and mapper.
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let path = self.location.joined();
        let role = self.location.role(&self.policy.residual_projections);
        if let Some(choice) = self.shared.get(&param.id) {
            let init = choice.init;
            let shared = choice
                .tensor
                .as_ref()
                .and_then(|tensor| tensor.downcast_ref::<Tensor<B, D>>())
                .cloned();
            self.report
                .parameters
                .push(InitializedParameter { path, role, init });
            return match shared {
                Some(tensor) => {
                    let (id, _, mapper) = param.consume();
                    Param::from_mapped_value(id, tensor, mapper)
                }
                None => param,
            };
        }
        let mut init = self.policy.resolve(&path, role);
        // Packed weights are always kept. A lazy parameter has not been
        // quantized (packing materializes it), so its dtype is not read.
        if init != ParameterInit::Keep
            && param.is_initialized()
            && matches!(param.val().dtype(), DType::QFloat(_))
        {
            init = ParameterInit::Keep;
        }
        self.report
            .parameters
            .push(InitializedParameter { path, role, init });
        if init == ParameterInit::Keep {
            self.shared
                .insert(param.id, SharedChoice { init, tensor: None });
            return param;
        }
        if !param.is_initialized() && self.deferrable.contains(&param.id) {
            // The constructed initializer still runs on first read, but its
            // output is released before the policy's value is drawn.
            return param.init_mapper(move |constructed| init.draw(constructed));
        }
        let (id, value, mapper) = param.consume();
        // The draw is a fresh leaf with the constructed value's trainability.
        let sampled = init.draw(value);
        self.shared.insert(
            id,
            SharedChoice {
                init,
                tensor: Some(Box::new(sampled.clone())),
            },
        );
        Param::from_mapped_value(id, sampled, mapper)
    }
}

#[cfg(all(test, feature = "flex"))]
#[path = "initialization_tests.rs"]
mod tests;
