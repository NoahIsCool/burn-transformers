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
//! # Contracts
//!
//! Replaced values keep their parameter identity, dtype, device, trainability
//! (the current tensor's `is_require_grad`), and Burn save/load mapper, so
//! records round-trip and optimizers see the same parameters. Each replacement
//! is a fresh autodiff leaf. Fields that share one identity receive one shared tensor.
//! Packed quantized parameters are always kept. Sampling uses the backend's
//! random generator: call `B::seed(&device, seed)` first for reproducible
//! draws. A replaced parameter is materialized once with its constructed value
//! before it is overwritten, so peak memory exceeds the model by at most one
//! parameter.

use std::{any::Any, collections::HashMap};

use burn::{
    module::{Module, ModuleMapper, Param, ParamId},
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
}

impl ParameterInit {
    /// Checks that a normal standard deviation is positive and finite.
    fn validate(self, field: &str) -> Result<()> {
        match self {
            Self::Normal { std } if !(std.is_finite() && std > 0.0) => Err(Error::configuration(
                field,
                format!(
                    "normal std must be finite and greater than zero, got {std}; use Zeros for a zero tensor"
                ),
            )),
            _ => Ok(()),
        }
    }

    /// Multiplies a normal standard deviation by `factor`; other variants are unchanged.
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
    /// such as `o_proj` and `down_proj`.
    pub residual_projections: Vec<String>,
    /// Path-specific initializers, applied after role rules; the last match wins.
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
    pub fn resolve(&self, path: &str, role: ParameterRole) -> ParameterInit {
        let base = self
            .overrides
            .iter()
            .rev()
            .find(|entry| under(path, &entry.path))
            .map_or_else(|| self.role_init(role), |entry| entry.init);
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

    /// Validates the policy, then returns `module` with every parameter initialized by it.
    ///
    /// The report lists every floating parameter with its role and the
    /// initializer actually used, sorted by path.
    pub fn apply<B: Backend, M: Module<B>>(&self, module: M) -> Result<(M, InitReport)> {
        self.validate()?;
        let mut mapper = Initializer {
            policy: self,
            path: Vec::new(),
            containers: Vec::new(),
            shared: HashMap::new(),
            report: InitReport::default(),
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

/// Module mapper that classifies parameters by owner and replaces their values.
struct Initializer<'p> {
    /// Policy being applied.
    policy: &'p InitPolicy,
    /// Field names from the root to the current parameter.
    path: Vec<String>,
    /// Container type of each entered field, such as `Struct:Linear`.
    containers: Vec<String>,
    /// Decision taken at the first field of each identity, reused by its aliases.
    shared: HashMap<ParamId, SharedChoice>,
    /// Parameters visited so far.
    report: InitReport,
}

impl Initializer<'_> {
    /// Infers the role of the parameter at the current path from its owning module.
    ///
    /// Burn reports the struct that holds a field when entering it, so a
    /// parameter's last container names its owning module type.
    fn role(&self) -> ParameterRole {
        let field = self.path.last().map(String::as_str);
        let owner = self.containers.last().map(String::as_str);
        let module_name = self
            .path
            .len()
            .checked_sub(2)
            .map(|index| self.path[index].as_str());
        match (owner, field) {
            (Some("Struct:Embedding"), Some("weight")) => ParameterRole::Embedding,
            (Some("Struct:Linear"), Some("bias")) => ParameterRole::Bias,
            (Some("Struct:Linear"), Some("weight"))
                if module_name.is_some_and(|name| {
                    self.policy
                        .residual_projections
                        .iter()
                        .any(|candidate| candidate == name)
                }) =>
            {
                ParameterRole::ResidualProjection
            }
            (Some("Struct:Linear"), Some("weight")) => ParameterRole::Projection,
            _ => ParameterRole::Other,
        }
    }
}

impl<B: Backend> ModuleMapper<B> for Initializer<'_> {
    /// Records the field name and the type of the struct that holds it.
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.path.push(name.to_owned());
        self.containers.push(container_type.to_owned());
    }

    /// Restores the parent path after leaving a field.
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
        self.containers.pop();
    }

    /// Replaces one parameter according to its role, preserving identity and mapper.
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let path = if self.path.is_empty() {
            "$".to_owned()
        } else {
            self.path.join(".")
        };
        let role = self.role();
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
        if init != ParameterInit::Keep && matches!(param.val().dtype(), DType::QFloat(_)) {
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
        let (id, value, mapper) = param.consume();
        let trainable = value.is_require_grad();
        let sampled = match init {
            ParameterInit::Normal { std } => value.random_like(Distribution::Normal(0.0, std)),
            ParameterInit::Zeros | ParameterInit::Keep => value.zeros_like(),
        }
        .detach()
        .set_require_grad(trainable);
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
