//! Editable Qwen backbones with explicit random or selective pretrained initialization.
//!
//! Recipes separate checkpoint layer identity from execution order. Each block can
//! change its mixer, head geometry, and SwiGLU width while retaining the stack's
//! residual axis `hidden_size`. Full attention has score axes
//! `[batch, query_heads, query, key]`; DeltaNet remains a causal recurrence and
//! cannot implement arbitrary softmax connectivity policies.
//!
//! Random initialization is artifact-free. Selective loading resolves the exact
//! source namespace and reports reused, converted, newly initialized, and omitted
//! parameters. Shape changes require explicit policy; new parameters adopt the
//! resolved checkpoint dtype and remain autodiff leaves. The result owns parameters
//! only, with no vocabulary head or mutable decode cache.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::Path,
};

use burn::{
    module::{Module, ModuleMapper, Param},
    nn::EmbeddingConfig,
    tensor::{DType, Tensor, backend::Backend},
};
use serde::{Deserialize, Serialize};

use crate::{
    ArchitectureConfig, Error, Result,
    checkpoint::{
        CheckpointManifest, ExecutionOutcome, ExpectedTensor, KeyMatcher, LanguageNamespace,
        LoadOptions, MappingPlan, TensorDType, TensorOperation, build_load_plan,
        execute_local_load_plan,
    },
    normalization::RmsNormConfig,
    transfer::{
        ParameterDisposition, ReusedParameter, ShapeChangePolicy, TransferReport,
        TransformedParameter,
    },
};

use super::{
    Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5DecoderLayer, Qwen3_5LayerType,
    Qwen3_5TextConfig, Qwen3_5TextModel,
};
use crate::models::config::ModelConfig;

// Consumer-editable topology, independent of checkpoint parameter ownership.

/// Editable dimensions and sequence mixer for one Qwen block.
///
/// Residual width and family numerical conventions come from the recipe's
/// `config`. Different residual widths can be joined with an explicit adapter
/// between independently executed block ranges.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qwen3_5LayerRecipe {
    /// Original checkpoint layer to reuse; `None` initializes this block anew.
    pub source_layer: Option<usize>,
    /// Full softmax attention or the original gated DeltaNet mixer.
    pub layer_type: Qwen3_5LayerType,
    /// Query heads in a full-attention block.
    pub num_attention_heads: usize,
    /// Native shared key/value heads in a full-attention block.
    pub num_key_value_heads: usize,
    /// Width of each full-attention head, independently of the residual width.
    pub head_dim: usize,
    /// SwiGLU gate/content intermediate width.
    pub intermediate_size: usize,
    /// DeltaNet query/key head count.
    pub linear_num_key_heads: usize,
    /// DeltaNet value head count.
    pub linear_num_value_heads: usize,
    /// DeltaNet query/key head width.
    pub linear_key_head_dim: usize,
    /// DeltaNet value head width.
    pub linear_value_head_dim: usize,
    /// Causal DeltaNet convolution kernel size.
    pub linear_conv_kernel_dim: usize,
}

impl Qwen3_5LayerRecipe {
    /// Copies one source layer into an editable recipe while preserving its original index.
    fn from_config(config: &Qwen3_5TextConfig, index: usize, layer_type: Qwen3_5LayerType) -> Self {
        Self {
            source_layer: Some(index),
            layer_type,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            intermediate_size: config.intermediate_size,
            linear_num_key_heads: config.linear_num_key_heads,
            linear_num_value_heads: config.linear_num_value_heads,
            linear_key_head_dim: config.linear_key_head_dim,
            linear_value_head_dim: config.linear_value_head_dim,
            linear_conv_kernel_dim: config.linear_conv_kernel_dim,
        }
    }

    /// Translates edited dimensions into a validated-later single-block family configuration.
    ///
    /// Residual width, rotary conventions, and dtype-independent numerical
    /// settings remain inherited; MTP is excluded from the headless block.
    fn config(&self, base: &Qwen3_5TextConfig) -> Qwen3_5TextConfig {
        let mut config = base.clone();
        config.num_hidden_layers = 1;
        config.layer_types = vec![self.layer_type];
        config.mtp_num_hidden_layers = 0;
        config.num_attention_heads = self.num_attention_heads;
        config.num_key_value_heads = self.num_key_value_heads;
        config.head_dim = self.head_dim;
        config.intermediate_size = self.intermediate_size;
        config.linear_num_key_heads = self.linear_num_key_heads;
        config.linear_num_value_heads = self.linear_num_value_heads;
        config.linear_key_head_dim = self.linear_key_head_dim;
        config.linear_value_head_dim = self.linear_value_head_dim;
        config.linear_conv_kernel_dim = self.linear_conv_kernel_dim;
        config
    }
}

/// A faithful Qwen starting point whose individual blocks can be edited.
///
/// `init_random` never reads weights. `load_pretrained` validates a local source
/// checkpoint and streams only the compatible parameters selected by this
/// recipe. Both return a stateless backbone with embeddings and final norm;
/// task heads and cached generation are separate choices.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qwen3_5BackboneRecipe {
    /// Shared residual width, vocabulary, norm, rotary, and family conventions.
    pub config: Qwen3_5TextConfig,
    /// Blocks in execution order. Remove, reorder, or edit entries directly.
    pub layers: Vec<Qwen3_5LayerRecipe>,
    /// Original checkpoint topology used to resolve stable parameter identities.
    source_config: Qwen3_5TextConfig,
}

/// Policy for selectively initializing a recipe from a local checkpoint.
#[derive(Clone, Debug, Default)]
pub struct Qwen3_5RecipeLoadOptions {
    /// Artifact bounds and dtype conversion policy.
    pub checkpoint: LoadOptions,
    /// Namespace override; `None` detects one unambiguous embedding namespace.
    pub source: Option<Qwen3_5CheckpointSource>,
    /// Changed shapes fail unless retaining random target initialization is explicit.
    /// New parameters use the checkpoint plan's resolved embedding dtype.
    pub shape_changes: ShapeChangePolicy,
}

/// A successfully initialized backbone and its complete transfer audit.
#[derive(Debug)]
pub struct LoadedQwen3_5Backbone<B: Backend> {
    /// The selected graph, ready for stateless execution and a custom task head.
    pub model: Qwen3_5TextModel<B>,
    /// Parameters reused, converted, omitted, or initialized anew.
    pub report: TransferReport,
    /// Source bytes read and the strict checkpoint executor's detailed report.
    pub checkpoint: ExecutionOutcome,
}

impl Qwen3_5BackboneRecipe {
    /// Starts with exactly the configured Qwen block schedule and conventions.
    pub fn from_config(config: Qwen3_5TextConfig) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        let layers = config
            .resolved_layer_types()
            .into_iter()
            .enumerate()
            .map(|(index, kind)| Qwen3_5LayerRecipe::from_config(&config, index, kind))
            .collect();
        Ok(Self {
            source_config: config.clone(),
            config,
            layers,
        })
    }

    /// Reads only `config.json` from a local model directory; no weights are loaded.
    pub fn from_local_config(root: impl AsRef<Path>) -> Result<Self> {
        Self::from_config(read_config(root.as_ref())?)
    }

    /// Retains the first `count` blocks, preserving their original source indices.
    pub fn prefix(mut self, count: usize) -> Result<Self> {
        if count == 0 || count > self.layers.len() {
            return Err(Error::configuration(
                "qwen.recipe.prefix",
                format!("count must be in 1..={}", self.layers.len()),
            ));
        }
        self.layers.truncate(count);
        Ok(self)
    }

    /// Returns the immutable source configuration used to interpret pretrained weights.
    pub fn source_config(&self) -> &Qwen3_5TextConfig {
        &self.source_config
    }

    /// Builds an editable backbone with new parameters and no checkpoint I/O.
    pub fn init_random<B: Backend>(&self, device: &B::Device) -> Result<Qwen3_5TextModel<B>> {
        let configurations = self.validate(false)?;
        let mut config = self.config.clone();
        config.num_hidden_layers = self.layers.len();
        config.layer_types = self.layers.iter().map(|layer| layer.layer_type).collect();
        config.mtp_num_hidden_layers = 0;
        let layer_rope = configurations
            .iter()
            .map(Qwen3_5TextConfig::rope_config)
            .collect::<Result<Vec<_>>>()?;
        let layers = configurations
            .iter()
            .zip(&self.layers)
            .map(|(config, layer)| Qwen3_5DecoderLayer::init(config, layer.layer_type, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Qwen3_5TextModel {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers,
            norm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_zero_centered(true)
                .init(device)?,
            rope: layer_rope[0],
            layer_rope,
            config,
            supports_family_cache: false,
        })
    }

    /// Streams compatible selected parameters from a local SafeTensors checkpoint.
    ///
    /// The original complete checkpoint is checked at the metadata level before
    /// allocating a target graph. Unselected blocks, MTP, and the language-model
    /// head are never materialized. The returned model is published only after
    /// every selected tensor has loaded successfully. Shape changes never slice
    /// or reshape pretrained weights implicitly. New parameters follow the
    /// resolved embedding dtype, including an explicit checkpoint dtype cast.
    pub fn load_pretrained<B: Backend>(
        &self,
        root: impl AsRef<Path>,
        options: &Qwen3_5RecipeLoadOptions,
        device: &B::Device,
    ) -> Result<LoadedQwen3_5Backbone<B>> {
        let configurations = self.validate(true)?;
        let root = root.as_ref();
        if read_config(root)? != self.source_config {
            return Err(Error::configuration(
                "qwen.recipe.source_config",
                "local checkpoint config differs from this recipe's original source configuration",
            ));
        }
        if !options.checkpoint.allow_missing.is_empty()
            || !options.checkpoint.allow_unexpected.is_empty()
        {
            return Err(Error::configuration(
                "qwen.recipe.checkpoint",
                "recipe loading owns its explicit parameter exceptions; remove broad missing/unexpected allowlists",
            ));
        }
        let manifest = CheckpointManifest::discover(root, &options.checkpoint)?;
        let source = source_namespace(&manifest, options.source)?;
        let source_contract = self
            .source_config
            .checkpoint_contract(Qwen3_5CheckpointOptions {
                source,
                include_mtp: false,
            })
            .map_err(|error| Error::Checkpoint(error.to_string()))?;
        let source_plan = source_contract.build_load_plan(&manifest, &options.checkpoint)?;
        let initialization_dtype = source_plan
            .tensors
            .iter()
            .find(|tensor| tensor.target == "model.embed_tokens.weight")
            .map(|tensor| tensor.dtype)
            .ok_or_else(|| {
                Error::Checkpoint("Qwen source plan lacks its embedding owner".into())
            })?;
        let initialization_dtype = match initialization_dtype {
            TensorDType::F16 => DType::F16,
            TensorDType::BF16 => DType::BF16,
            TensorDType::F32 => DType::F32,
            TensorDType::F64 => DType::F64,
            dtype => {
                return Err(Error::BackendCapability(format!(
                    "Qwen recipe initialization requires a supported floating dtype, found {dtype}"
                )));
            }
        };
        let source_operations = source_contract
            .mappings()
            .operations()
            .iter()
            .filter_map(|operation| match operation {
                TensorOperation::Map {
                    source,
                    target,
                    transform,
                } => Some((target.clone(), (source.clone(), *transform))),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let source_expected = source_contract
            .expected_tensors()
            .iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect::<BTreeMap<_, _>>();
        let mut expected = Vec::new();
        let mut mappings = MappingPlan::new();
        let mut consumed = BTreeSet::new();
        let mut report = TransferReport::default();
        for (target, source_target) in self.parameter_selection(&configurations)? {
            let selected_source = source_target
                .as_ref()
                .and_then(|name| source_expected.get(name).map(|entry| (name, entry)));
            let Some((source_target, source_entry)) = selected_source else {
                report.newly_initialized.push(ParameterDisposition::new(
                    &target.name,
                    "new block or component has no selected pretrained source",
                ));
                continue;
            };
            if source_entry.shape != target.shape {
                if options.shape_changes == ShapeChangePolicy::Reject {
                    return Err(Error::shape(
                        &target.name,
                        format!("checkpoint shape {:?}", source_entry.shape),
                        format!(
                            "edited shape {:?}; choose ShapeChangePolicy::Reinitialize explicitly",
                            target.shape
                        ),
                    ));
                }
                report.newly_initialized.push(ParameterDisposition::new(
                    &target.name,
                    "retained new initialization because the edited shape differs from the checkpoint",
                ));
                continue;
            }
            let Some((source, transform)) = source_operations.get(source_target) else {
                return Err(Error::configuration(
                    "qwen.recipe.mapping",
                    format!("missing physical source for {source_target}"),
                ));
            };
            if !consumed.insert(source.clone()) {
                return Err(Error::configuration(
                    "qwen.recipe.source_layer",
                    "a checkpoint layer may be selected once; duplicate a loaded module explicitly to choose sharing or independent copies",
                ));
            }
            mappings.push(TensorOperation::Map {
                source: source.clone(),
                target: target.name.clone(),
                transform: *transform,
            });
            report.reused.push(ReusedParameter::new(
                source,
                &target.name,
                target
                    .shape
                    .iter()
                    .map(|&dim| dim as usize)
                    .collect::<Vec<_>>(),
            ));
            expected.push(target);
        }
        let mut load_options = options.checkpoint.clone();
        source_contract.configure_load_options(&mut load_options);
        for source in manifest
            .tensors
            .keys()
            .filter(|name| !consumed.contains(*name))
        {
            load_options
                .allow_unexpected
                .push(KeyMatcher::exact(source));
            report.omitted.push(ParameterDisposition::new(source, "outside the selected backbone, or superseded by an explicitly reinitialized parameter"));
        }
        if self.source_config.tie_word_embeddings {
            report.omitted.push(ParameterDisposition::new("lm_head.weight", "tied vocabulary view omitted; its embedding storage owner remains available to the backbone"));
        }
        let plan = build_load_plan(&manifest, &expected, &mappings, &load_options)?;
        let new_parameters = report
            .newly_initialized
            .iter_mut()
            .map(|parameter| {
                parameter.reason.push_str(&format!(
                    "; initialized in the resolved embedding dtype {initialization_dtype:?}"
                ));
                module_parameter_path(&parameter.name)
            })
            .collect();
        let mut model = self.init_random(device)?.map(&mut InitializeDtype {
            path: Vec::new(),
            new_parameters,
            dtype: initialization_dtype,
        });
        let checkpoint = execute_local_load_plan(&manifest, &plan, &mut model)
            .map_err(Error::checkpoint_execution)?;
        for conversion in &checkpoint.report.dtype_converted {
            if let Some(index) = report
                .reused
                .iter()
                .position(|parameter| parameter.target == conversion.target)
            {
                let reused = report.reused.remove(index);
                report.transformed.push(TransformedParameter {
                    sources: vec![reused.source],
                    target: reused.target,
                    shape: reused.shape,
                    operation: format!(
                        "dtype conversion {:?} to {:?}",
                        conversion.source, conversion.target_dtype
                    ),
                });
            }
        }
        report.finalize();
        Ok(LoadedQwen3_5Backbone {
            model,
            report,
            checkpoint,
        })
    }

    /// Validates each edited block and resolves its executable family configuration.
    ///
    /// Pretrained reuse additionally requires unique, in-range source indices;
    /// random initialization does not read or require those source parameters.
    fn validate(&self, pretrained: bool) -> Result<Vec<Qwen3_5TextConfig>> {
        ArchitectureConfig::validate(&self.source_config)?;
        if self.layers.is_empty() {
            return Err(Error::configuration(
                "qwen.recipe.layers",
                "at least one block is required",
            ));
        }
        let mut seen = BTreeSet::new();
        let mut configurations = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            if pretrained
                && let Some(source) = layer.source_layer
                && (source >= self.source_config.num_hidden_layers || !seen.insert(source))
            {
                return Err(Error::configuration(
                    format!("qwen.recipe.layers.{index}.source_layer"),
                    "source index must be in range and selected only once; None selects new initialization",
                ));
            }
            let config = layer.config(&self.config);
            ArchitectureConfig::validate(&config)?;
            config.rope_config()?;
            configurations.push(config);
        }
        Ok(configurations)
    }

    /// Pairs each target parameter and checkpoint shape with an optional source owner.
    ///
    /// Block indices are rewritten independently of the checkpoint ordering.
    /// A missing source selects new initialization rather than implicit reuse.
    fn parameter_selection(
        &self,
        configurations: &[Qwen3_5TextConfig],
    ) -> Result<Vec<(ExpectedTensor, Option<String>)>> {
        let mut selection = vec![
            (
                ExpectedTensor::new(
                    "model.embed_tokens.weight",
                    vec![
                        self.config.vocab_size as u64,
                        self.config.hidden_size as u64,
                    ],
                ),
                Some("model.embed_tokens.weight".into()),
            ),
            (
                ExpectedTensor::new("model.norm.weight", vec![self.config.hidden_size as u64]),
                Some("model.norm.weight".into()),
            ),
        ];
        for (index, config) in configurations.iter().enumerate() {
            let contract = config
                .checkpoint_contract(Qwen3_5CheckpointOptions {
                    source: Qwen3_5CheckpointSource::TextOnly,
                    include_mtp: false,
                })
                .map_err(|error| Error::Checkpoint(error.to_string()))?;
            for tensor in contract.expected_tensors() {
                let Some(suffix) = tensor.name.strip_prefix("model.layers.0.") else {
                    continue;
                };
                let mut target = tensor.clone();
                target.name = format!("model.layers.{index}.{suffix}");
                let source = self.layers[index]
                    .source_layer
                    .map(|source| format!("model.layers.{source}.{suffix}"));
                selection.push((target, source));
            }
        }
        Ok(selection)
    }
}

// Selective installation: preserve lazy pretrained data and explicit dtype policy.

/// Checkpoint parameter suffixes whose Burn field names differ from Hugging Face's.
///
/// Burn's derived module paths use Rust field names, while the checkpoint
/// contract (and therefore the transfer report) uses Hugging Face tensor
/// names. These are the only Qwen3.5 backbone parameters where the two
/// disagree; every other segment, including `layers.{index}`, is identical,
/// and `Option` containers add no path segment. The family loader's
/// `install_layer_tensor` dispatch encodes the same correspondence.
const CHECKPOINT_TO_MODULE_SUFFIXES: [(&str, &str); 3] = [
    (".self_attn.q_norm.weight", ".self_attn.query_norm.weight"),
    (".self_attn.k_norm.weight", ".self_attn.key_norm.weight"),
    (".linear_attn.A_log", ".linear_attn.a_log"),
];

/// Translates a checkpoint parameter name into the Burn module path that owns it.
///
/// For example `model.layers.3.self_attn.q_norm.weight` becomes
/// `model.layers.3.self_attn.query_norm.weight`. Names without a differing
/// suffix are returned unchanged.
fn module_parameter_path(checkpoint_name: &str) -> String {
    for (checkpoint_suffix, module_suffix) in CHECKPOINT_TO_MODULE_SUFFIXES {
        if let Some(prefix) = checkpoint_name.strip_suffix(checkpoint_suffix) {
            return format!("{prefix}{module_suffix}");
        }
    }
    checkpoint_name.to_owned()
}

/// Cast only new parameters, leaving overwritten pretrained parameters lazy.
struct InitializeDtype {
    /// Current module components used to construct fully qualified parameter names.
    path: Vec<String>,
    /// Burn module paths (see [`module_parameter_path`]) of parameters whose
    /// checkpoint shapes cannot or should not be reused.
    new_parameters: BTreeSet<String>,
    /// Floating dtype selected by the source embedding's resolved load plan.
    dtype: DType,
}

impl<B: Backend> ModuleMapper<B> for InitializeDtype {
    /// Pushes a module name so parameter callbacks resolve fully qualified target paths.
    fn enter_module(&mut self, name: &str, _: &str) {
        self.path.push(name.into());
    }

    /// Removes the completed module from the parameter-path stack.
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }

    /// Casts only newly initialized parameters to the resolved checkpoint dtype.
    ///
    /// Detaching the cast restores a leaf while retaining its trainability flag;
    /// pretrained parameters remain lazy until the loader installs their data.
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let name = format!("model.{}", self.path.join("."));
        if !self.new_parameters.contains(&name) {
            return param;
        }
        param.map(|value| {
            let trainable = value.is_require_grad();
            value.cast(self.dtype).detach().set_require_grad(trainable)
        })
    }
}

/// Uses an explicit namespace or detects exactly one language namespace.
///
/// Detection is the shared [`LanguageNamespace`] classifier, so recipe,
/// concrete-family, and automatic loading choose the same layout for the same
/// checkpoint. Ambiguous or absent namespaces fail rather than guessing a
/// model layout, as [`crate::checkpoint::CheckpointError::AmbiguousLanguageNamespace`]
/// or [`crate::checkpoint::CheckpointError::LanguageNamespaceNotFound`].
fn source_namespace(
    manifest: &CheckpointManifest,
    explicit: Option<Qwen3_5CheckpointSource>,
) -> Result<Qwen3_5CheckpointSource> {
    if let Some(source) = explicit {
        return Ok(source);
    }
    Ok(match LanguageNamespace::of_manifest(manifest, "Qwen")? {
        LanguageNamespace::Wrapped => Qwen3_5CheckpointSource::Multimodal,
        LanguageNamespace::TextOnly => Qwen3_5CheckpointSource::TextOnly,
    })
}

/// Reads and parses local `config.json` under a 16 MiB bound.
///
/// Both metadata and the actual read are bounded so a growing file cannot
/// bypass the configuration-size limit.
fn read_config(root: &Path) -> Result<Qwen3_5TextConfig> {
    let path = root.join("config.json");
    let metadata = fs::metadata(&path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    if metadata.len() > 16 * 1024 * 1024 {
        return Err(Error::configuration(
            "qwen.recipe.config",
            "config.json exceeds 16 MiB",
        ));
    }
    let mut json = String::new();
    fs::File::open(&path)
        .map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?
        .take(16 * 1024 * 1024 + 1)
        .read_to_string(&mut json)
        .map_err(|source| Error::Io { path, source })?;
    if json.len() > 16 * 1024 * 1024 {
        return Err(Error::configuration(
            "qwen.recipe.config",
            "config.json exceeds 16 MiB",
        ));
    }
    Qwen3_5TextConfig::from_json_str(&json)
        .map_err(|error| Error::configuration("qwen.recipe.config", error.to_string()))
}

#[cfg(test)]
#[path = "recipe_tests.rs"]
mod tests;
