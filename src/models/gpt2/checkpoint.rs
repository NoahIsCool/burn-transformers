//! Strict local GPT-2 SafeTensors mapping. HF Conv1D matrices are `[in, out]`;
//! the fused QKV output axis is split without an implicit transpose.
//!
//! The contract separates source namespace selection, shape/alias preflight, and
//! parameter installation. A fused weight `[d, 3*d]` splits its output axis into
//! Q/K/V `[d, d]`; its `[3*d]` bias splits on axis zero. MLP weights remain
//! `[d, m]` and `[m, d]`. Affine LayerNorm vectors map to gamma/beta, and the
//! vocabulary head aliases the sole `[vocabulary, d]` token-embedding owner.
//!
//! All planned tensors must be floating point; conversion policy belongs to shared
//! load options. Exact legacy causal-mask buffers may be ignored because masks are
//! reconstructed from the graph. Metadata preflight precedes writes, but a later
//! streaming failure can leave earlier parameters installed: discard that model.

use std::path::Path;

use burn::{
    module::Param,
    nn::{LayerNorm, Linear},
    tensor::{Tensor, TensorData, backend::Backend},
};

use crate::{
    Error,
    checkpoint::{
        CheckpointError, CheckpointManifest, ExecutionFailure, ExecutionOutcome, ExpectedTensor,
        IndexTensorAlias, KeyMatcher, LoadOptions, LoadPlan, MappingPlan, ParameterInstallError,
        TensorSink, build_load_plan, discover_checkpoint, execute_local_load_plan,
        install_float_parameter,
    },
};

use super::{Gpt2Config, Gpt2ForCausalLm, Gpt2Model};

// Explicit source namespaces and metadata planning.

/// Exact external namespaces used by official GPT-2 backbone and LM checkpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gpt2CheckpointLayout {
    /// `wte.weight`, `h.0.attn.c_attn.weight`, and `ln_f.weight`.
    Backbone,
    /// `transformer.wte.weight`, `transformer.h.0.*`, and optional tied `lm_head.weight`.
    LanguageModel,
}

impl Gpt2CheckpointLayout {
    /// Returns the exact checkpoint namespace prefix for this embedding storage owner.
    fn prefix(self) -> &'static str {
        match self {
            Self::Backbone => "",
            Self::LanguageModel => "transformer.",
        }
    }

    /// Selects only an unambiguous storage owner; mixed namespaces are rejected.
    pub fn detect(manifest: &CheckpointManifest) -> Result<Self, Gpt2LoadError> {
        let backbone = manifest.tensors.contains_key("wte.weight");
        let lm = manifest.tensors.contains_key("transformer.wte.weight");
        match (backbone, lm) {
            (true, false) => Ok(Self::Backbone),
            (false, true) => Ok(Self::LanguageModel),
            _ => Err(Gpt2LoadError::Target(
                "expected exactly one GPT-2 token-embedding namespace".into(),
            )),
        }
    }
}

/// Auditable expected parameter inventory and HF Conv1D mapping.
#[derive(Clone, Debug)]
pub struct Gpt2CheckpointContract {
    /// Logical runtime parameters and their expected serialized shapes.
    expected: Vec<ExpectedTensor>,
    /// Explicit Conv1D splits and source-to-runtime tensor transformations.
    mappings: MappingPlan,
    /// Permitted index-only declarations of the tied vocabulary head.
    aliases: Vec<IndexTensorAlias>,
    /// Exact legacy mask-buffer names reconstructed by the executable graph.
    ignored_buffers: Vec<KeyMatcher>,
}

impl Gpt2CheckpointContract {
    /// Derives shapes from a validated graph, including the tied readout alias.
    pub fn new(config: &Gpt2Config, layout: Gpt2CheckpointLayout) -> Result<Self, Gpt2LoadError> {
        config.validate()?;
        let mut contract = Self {
            expected: Vec::new(),
            mappings: MappingPlan::new(),
            aliases: vec![IndexTensorAlias::new(
                "lm_head.weight",
                format!("{}wte.weight", layout.prefix()),
            )],
            ignored_buffers: Vec::new(),
        };
        let p = layout.prefix();
        let h = config.n_embd as u64;
        let inner = config.intermediate_size()? as u64;
        contract.map(
            format!("{p}wte.weight"),
            "token_embedding.weight".into(),
            vec![config.vocab_size as u64, h],
        );
        contract.map(
            format!("{p}wpe.weight"),
            "position_embedding.weight".into(),
            vec![config.n_positions as u64, h],
        );
        for i in 0..config.n_layer {
            let source = format!("{p}h.{i}");
            let target = format!("layers.{i}");
            for (external, internal) in [("ln_1", "attention_norm"), ("ln_2", "mlp_norm")] {
                contract.map(
                    format!("{source}.{external}.weight"),
                    format!("{target}.{internal}.gamma"),
                    vec![h],
                );
                contract.map(
                    format!("{source}.{external}.bias"),
                    format!("{target}.{internal}.beta"),
                    vec![h],
                );
            }
            let weights: Vec<_> = ["q_proj", "k_proj", "v_proj"]
                .into_iter()
                .map(|name| (format!("{target}.attention.{name}.weight"), h))
                .collect();
            let biases: Vec<_> = ["q_proj", "k_proj", "v_proj"]
                .into_iter()
                .map(|name| (format!("{target}.attention.{name}.bias"), h))
                .collect();
            for (name, _) in &weights {
                contract
                    .expected
                    .push(ExpectedTensor::new(name, vec![h, h]));
            }
            for (name, _) in &biases {
                contract.expected.push(ExpectedTensor::new(name, vec![h]));
            }
            contract
                .mappings
                .split(format!("{source}.attn.c_attn.weight"), 1, weights);
            contract
                .mappings
                .split(format!("{source}.attn.c_attn.bias"), 0, biases);
            for (external, internal, input, output) in [
                ("attn.c_proj", "attention.o_proj", h, h),
                ("mlp.c_fc", "mlp.up_proj", h, inner),
                ("mlp.c_proj", "mlp.down_proj", inner, h),
            ] {
                contract.map(
                    format!("{source}.{external}.weight"),
                    format!("{target}.{internal}.weight"),
                    vec![input, output],
                );
                contract.map(
                    format!("{source}.{external}.bias"),
                    format!("{target}.{internal}.bias"),
                    vec![output],
                );
            }
            // Historical checkpoints serialized deterministic causal-mask
            // buffers. The executable graph constructs its masks explicitly.
            contract.ignored_buffers.extend([
                KeyMatcher::exact(format!("{source}.attn.bias")),
                KeyMatcher::exact(format!("{source}.attn.masked_bias")),
            ]);
        }
        contract.map(
            format!("{p}ln_f.weight"),
            "final_norm.gamma".into(),
            vec![h],
        );
        contract.map(format!("{p}ln_f.bias"), "final_norm.beta".into(), vec![h]);
        contract.expected.push(ExpectedTensor::new(
            "lm_head.weight",
            vec![config.vocab_size as u64, h],
        ));
        contract
            .mappings
            .alias("token_embedding.weight", "lm_head.weight");
        Ok(contract)
    }

    /// Registers one source rename and its expected native runtime/record dimensions.
    fn map(&mut self, source: String, target: String, shape: Vec<u64>) {
        self.expected.push(ExpectedTensor::new(&target, shape));
        self.mappings.map(source, target);
    }

    /// Every logical parameter, with native runtime/record dimensions.
    pub fn expected_tensors(&self) -> &[ExpectedTensor] {
        &self.expected
    }

    /// Explicit QKV splits, renames, and tied readout alias.
    pub fn mappings(&self) -> &MappingPlan {
        &self.mappings
    }

    /// Adds only exact legacy-buffer exceptions and verified tied-weight aliases.
    pub fn configure_load_options(&self, options: &mut LoadOptions) {
        for ignored in &self.ignored_buffers {
            if !options.allow_unexpected.contains(ignored) {
                options.allow_unexpected.push(ignored.clone());
            }
        }
        for alias in &self.aliases {
            if !options.index_aliases.contains(alias) {
                options.index_aliases.push(alias.clone());
            }
        }
    }

    /// Validates complete expected/observed metadata before installing parameters.
    pub fn build_load_plan(
        &self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
    ) -> Result<LoadPlan, Gpt2LoadError> {
        let mut options = options.clone();
        self.configure_load_options(&mut options);
        let plan = build_load_plan(manifest, &self.expected, &self.mappings, &options)?;
        if plan.tensors.iter().any(|tensor| !tensor.dtype.is_float()) {
            return Err(Gpt2LoadError::Target(
                "GPT-2 parameters must be floating point".into(),
            ));
        }
        Ok(plan)
    }
}

/// Contextual configuration, discovery, mapping, or parameter-installation error.
#[derive(Debug, thiserror::Error)]
pub enum Gpt2LoadError {
    /// The declared graph is invalid.
    #[error(transparent)]
    Configuration(
        /// Invalid graph configuration or dimension contract.
        #[from]
        Error,
    ),
    /// Bounded local discovery or mapping failed.
    #[error(transparent)]
    Checkpoint(
        /// Local artifact discovery or strict metadata failure.
        #[from]
        CheckpointError,
    ),
    /// The target did not name a parameter in this exact graph.
    #[error("invalid GPT-2 checkpoint target: {0}")]
    Target(
        /// Rejected parameter target or namespace diagnostic.
        String,
    ),
    /// Rank, layout, dtype, device storage, or payload disagreed with a parameter.
    #[error("cannot install GPT-2 parameter `{target}`: {source}")]
    Parameter {
        /// Architecture-owned target name.
        target: String,
        /// Exact failed parameter contract.
        #[source]
        source: ParameterInstallError,
    },
    /// Streaming failed after zero or more writes; includes the partial report.
    #[error(transparent)]
    Execution(
        /// Partial execution report and underlying parameter failure.
        #[from]
        Box<ExecutionFailure<Gpt2LoadError>>,
    ),
}

impl<B: Backend> Gpt2Model<B> {
    /// Loads a local checkpoint after complete metadata preflight.
    ///
    /// Only the embedding owner determines the supported namespace. Legacy mask
    /// buffers are ignored explicitly; redundant LM-head bytes must equal the
    /// tied embedding. Discard the model after a streaming execution failure.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, Gpt2LoadError> {
        let root = root.as_ref();
        // An index can omit a physical tied head. Try each explicit namespace
        // with its alias policy in place before discovery; never allow one
        // alias to have two potential storage owners in a single manifest.
        let mut failure = None;
        for layout in [
            Gpt2CheckpointLayout::Backbone,
            Gpt2CheckpointLayout::LanguageModel,
        ] {
            let contract = Gpt2CheckpointContract::new(&self.config, layout)?;
            let mut configured = options.clone();
            contract.configure_load_options(&mut configured);
            let manifest = match discover_checkpoint(root, &configured) {
                Ok(manifest) => manifest,
                Err(error) => {
                    failure = Some(Gpt2LoadError::Checkpoint(error));
                    continue;
                }
            };
            if Gpt2CheckpointLayout::detect(&manifest)? != layout {
                continue;
            }
            let plan = contract.build_load_plan(&manifest, &configured)?;
            return Ok(execute_local_load_plan(&manifest, &plan, self)?);
        }
        Err(failure.unwrap_or_else(|| {
            Gpt2LoadError::Target("no supported GPT-2 checkpoint namespace".into())
        }))
    }
}

impl<B: Backend> Gpt2ForCausalLm<B> {
    /// Strictly loads the public backbone and verifies any tied vocabulary alias.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, Gpt2LoadError> {
        self.model.load_checkpoint(root, options)
    }
}

// Checked installation into existing parameter owners.

impl<B: Backend> TensorSink for Gpt2Model<B> {
    /// Contextual GPT-2 target, payload, or alias installation failure.
    type Error = Gpt2LoadError;

    /// Installs one preplanned embedding, affine normalization, or split projection target.
    ///
    /// Layer indices and suffixes remain checked even when the sink is invoked without a load plan.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        match target {
            "token_embedding.weight" => install(target, &mut self.token_embedding.weight, data),
            "position_embedding.weight" => {
                install(target, &mut self.position_embedding.weight, data)
            }
            "final_norm.gamma" => install(target, &mut self.final_norm.gamma, data),
            "final_norm.beta" => install_optional(target, &mut self.final_norm.beta, data),
            _ => {
                let (index, suffix) = target
                    .strip_prefix("layers.")
                    .and_then(|part| part.split_once('.'))
                    .ok_or_else(|| Gpt2LoadError::Target(target.into()))?;
                let index = index
                    .parse::<usize>()
                    .map_err(|_| Gpt2LoadError::Target(target.into()))?;
                let block = self
                    .layers
                    .get_mut(index)
                    .ok_or_else(|| Gpt2LoadError::Target(target.into()))?;
                if let Some(suffix) = suffix.strip_prefix("attention_norm.") {
                    return install_norm(target, &mut block.attention_norm, suffix, data);
                }
                if let Some(suffix) = suffix.strip_prefix("mlp_norm.") {
                    return install_norm(target, &mut block.mlp_norm, suffix, data);
                }
                let (path, parameter) = suffix
                    .rsplit_once('.')
                    .ok_or_else(|| Gpt2LoadError::Target(target.into()))?;
                let linear = match path {
                    "attention.q_proj" => &mut block.attention.q_proj,
                    "attention.k_proj" => &mut block.attention.k_proj,
                    "attention.v_proj" => &mut block.attention.v_proj,
                    "attention.o_proj" => &mut block.attention.o_proj,
                    "mlp.up_proj" => &mut block.mlp.up_proj,
                    "mlp.down_proj" => &mut block.mlp.down_proj,
                    _ => return Err(Gpt2LoadError::Target(target.into())),
                };
                install_linear(target, linear, parameter, data)
            }
        }
    }

    /// Verifies that the vocabulary head reuses an initialized token-embedding owner.
    ///
    /// No second parameter is created, so the tied weight has one optimizer registration.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        if target != "lm_head.weight"
            || source_target != "token_embedding.weight"
            || !self.config.tie_word_embeddings
            || !self.token_embedding.weight.is_initialized()
        {
            return Err(Gpt2LoadError::Target(format!(
                "invalid tied readout {target} -> {source_target}"
            )));
        }
        Ok(())
    }
}

/// Installs floating data using the parameter's layout contract and adds its target to errors.
fn install<B: Backend, const D: usize>(
    target: &str,
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> Result<(), Gpt2LoadError> {
    install_float_parameter(parameter, data).map_err(|source| Gpt2LoadError::Parameter {
        target: target.into(),
        source,
    })
}

/// Requires the declared bias to exist before installing its rank-one floating payload.
fn install_optional<B: Backend>(
    target: &str,
    parameter: &mut Option<Param<Tensor<B, 1>>>,
    data: TensorData,
) -> Result<(), Gpt2LoadError> {
    install(
        target,
        parameter
            .as_mut()
            .ok_or_else(|| Gpt2LoadError::Target(format!("missing bias at {target}")))?,
        data,
    )
}

/// Dispatches an exact `gamma` or `beta` normalization target without accepting extra fields.
fn install_norm<B: Backend>(
    target: &str,
    norm: &mut LayerNorm<B>,
    parameter: &str,
    data: TensorData,
) -> Result<(), Gpt2LoadError> {
    match parameter {
        "gamma" => install(target, &mut norm.gamma, data),
        "beta" => install_optional(target, &mut norm.beta, data),
        _ => Err(Gpt2LoadError::Target(target.into())),
    }
}

/// Installs a projection weight or bias in the existing checkpoint-native Row layout.
fn install_linear<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    parameter: &str,
    data: TensorData,
) -> Result<(), Gpt2LoadError> {
    match parameter {
        "weight" => install(target, &mut linear.weight, data),
        "bias" => install_optional(target, &mut linear.bias, data),
        _ => Err(Gpt2LoadError::Target(target.into())),
    }
}
