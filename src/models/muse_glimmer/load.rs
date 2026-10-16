//! Streamed Burn parameter installation for Muse-Glimmer checkpoints.
//!
//! Pass one validates every source key, shape, dtype, shard path, and explicit
//! visual exception before this module receives data. Pass two holds at most one
//! source tensor and one converted output at a time, then dispatches the owned
//! [`TensorData`] into the statically ranked Burn field. Matrix installation goes
//! through Burn's layout-aware parameter mapper, preserving differentiability and
//! converting official `[output, input]` records to runtime `[input, output]` once.
//!
//! Packed QFloat payloads from a certified Burnpack sidecar install exactly:
//! packed where the backend keeps the artifact's blocks through that transpose,
//! and as their exact dequantized values otherwise (see
//! [`install_packed_parameter`]), never as a requantized approximation.

use std::path::Path;

use burn::{
    module::Param,
    nn::Linear,
    tensor::{Tensor, TensorData, backend::Backend},
};
use thiserror::Error;

use crate::{
    checkpoint::{
        CheckpointError, CheckpointManifest, ExecutionFailure, ExecutionOutcome, LanguageNamespace,
        LoadOptions, PackedLayoutPolicy, ParameterInstallError, TensorSink,
        execute_local_load_plan, install_packed_parameter,
    },
    normalization::RmsNorm,
};

use super::{
    ConfigError, MuseGlimmerCheckpointOptions, MuseGlimmerCheckpointSource,
    MuseGlimmerDecoderLayer, MuseGlimmerForCausalLm,
};

/// Architecture-specific failure while applying one planned parameter.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MuseGlimmerLoadError {
    /// The target does not name an executable text-model parameter.
    #[error("unknown Muse-Glimmer parameter target `{target}`")]
    UnknownTarget {
        /// Unrecognized internal key.
        target: String,
    },
    /// A layer index or target structure is malformed or out of range.
    #[error("invalid Muse-Glimmer target `{target}`: {reason}")]
    InvalidIndex {
        /// Malformed internal key.
        target: String,
        /// Index or syntax diagnostic.
        reason: String,
    },
    /// The configured graph lacks a parameter required by the contract.
    #[error("Muse-Glimmer target `{target}` requires missing component `{component}`")]
    MissingComponent {
        /// Parameter being installed.
        target: String,
        /// Expected graph component.
        component: &'static str,
    },
    /// Tensor metadata disagrees with the statically ranked Burn field.
    #[error("cannot install Muse-Glimmer parameter `{target}`: {source}")]
    Parameter {
        /// Internal parameter target.
        target: String,
        /// Rank, shape, dtype, or backend storage failure.
        #[source]
        source: ParameterInstallError,
    },
    /// A tied parameter operation violated the embedding/head contract.
    #[error("invalid Muse-Glimmer alias `{target}` -> `{source_target}`: {reason}")]
    InvalidAlias {
        /// Alias target.
        target: String,
        /// Storage-owning target.
        source_target: String,
        /// Human-readable contract violation.
        reason: String,
    },
}

/// Failure from topology validation, strict planning, or streamed execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MuseGlimmerCheckpointLoadError {
    /// The model topology cannot describe a valid checkpoint.
    #[error(transparent)]
    Config(
        /// Invalid topology or dimension reported before loading.
        #[from]
        ConfigError,
    ),
    /// Artifact discovery or full expected/observed validation failed.
    #[error(transparent)]
    Checkpoint(
        /// Artifact discovery or strict load-plan failure.
        #[from]
        CheckpointError,
    ),
    /// Pass two failed after zero or more audited writes.
    #[error(transparent)]
    Execution(
        /// Partial write report and underlying parameter failure.
        #[from]
        Box<ExecutionFailure<MuseGlimmerLoadError>>,
    ),
}

impl MuseGlimmerCheckpointLoadError {
    /// Returns partial execution diagnostics when pass two reached the sink.
    #[must_use]
    pub fn execution_failure(&self) -> Option<&ExecutionFailure<MuseGlimmerLoadError>> {
        match self {
            Self::Execution(failure) => Some(failure),
            Self::Config(_) | Self::Checkpoint(_) => None,
        }
    }
}

impl<B: Backend> TensorSink for MuseGlimmerForCausalLm<B> {
    /// Family-specific parameter installation or alias failure.
    type Error = MuseGlimmerLoadError;

    /// Installs one storage-owning target from the validated language contract.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        match target {
            "model.embed_tokens.weight" => {
                install(target, &mut self.model.embed_tokens.weight, data)
            }
            "model.norm.weight" => install_norm(target, &mut self.model.norm, data),
            "lm_head.weight" => {
                let head = self.lm_head.as_mut().ok_or_else(|| {
                    MuseGlimmerLoadError::MissingComponent {
                        target: target.into(),
                        component: "independent LM head for an untied checkpoint",
                    }
                })?;
                install_linear_weight(target, head, data)
            }
            _ => {
                let (index, suffix) = parse_layer_target(target)?;
                let layer_count = self.model.layers.len();
                let layer = self.model.layers.get_mut(index).ok_or_else(|| {
                    MuseGlimmerLoadError::InvalidIndex {
                        target: target.into(),
                        reason: format!("layer {index} is outside 0..{layer_count}"),
                    }
                })?;
                install_layer_tensor(layer, target, suffix, data)
            }
        }
    }

    /// Validates the sole supported embedding-to-head storage alias.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        if target != "lm_head.weight" || source_target != "model.embed_tokens.weight" {
            return Err(MuseGlimmerLoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "only the tied embedding-to-LM-head alias is supported".into(),
            });
        }
        if !self.model.config.tie_word_embeddings || self.lm_head.is_some() {
            return Err(MuseGlimmerLoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the configured model declares an independent LM head".into(),
            });
        }
        if !self.model.embed_tokens.weight.is_initialized() {
            return Err(MuseGlimmerLoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the embedding storage owner has not been installed".into(),
            });
        }
        Ok(())
    }
}

impl<B: Backend> MuseGlimmerForCausalLm<B> {
    /// Strictly discovers, validates, and streams a local Muse checkpoint.
    ///
    /// The language namespace is detected from the tensor names with the
    /// shared classifier automatic loading uses: an official multimodal
    /// artifact (`model.language_model.*`, whose visual tensors are audited as
    /// explicitly ignored keys) or a standalone text checkpoint
    /// (`model.embed_tokens.weight`). A checkpoint with neither or both fails
    /// with [`CheckpointError::LanguageNamespaceNotFound`] or
    /// [`CheckpointError::AmbiguousLanguageNamespace`]; any unknown namespace
    /// remains a strict error. No network access occurs. After an execution
    /// failure the partially initialized model must be discarded.
    ///
    /// # Errors
    ///
    /// Returns a [`MuseGlimmerCheckpointLoadError`] naming the failing stage.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, MuseGlimmerCheckpointLoadError> {
        let manifest = CheckpointManifest::discover(root, options)?;
        let source = match LanguageNamespace::of_manifest(&manifest, "Muse")? {
            LanguageNamespace::Wrapped => MuseGlimmerCheckpointSource::Multimodal,
            LanguageNamespace::TextOnly => MuseGlimmerCheckpointSource::TextOnly,
        };
        self.load_discovered(&manifest, options, MuseGlimmerCheckpointOptions { source })
    }

    /// Loads a local checkpoint with an explicit multimodal or text-only namespace.
    ///
    /// Use this to force the `source` layout; [`Self::load_checkpoint`]
    /// detects it.
    ///
    /// # Errors
    ///
    /// As [`Self::load_checkpoint`], without the namespace-detection errors.
    pub fn load_checkpoint_with(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
        family_options: MuseGlimmerCheckpointOptions,
    ) -> Result<ExecutionOutcome, MuseGlimmerCheckpointLoadError> {
        let manifest = CheckpointManifest::discover(root, options)?;
        self.load_discovered(&manifest, options, family_options)
    }

    /// Former name of [`Self::load_checkpoint_with`], kept so 0.1.0 code compiles.
    ///
    /// # Errors
    ///
    /// As [`Self::load_checkpoint_with`], to which it forwards unchanged.
    #[deprecated(note = "use `load_checkpoint_with`, the name every family shares")]
    pub fn load_checkpoint_with_family_options(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
        family_options: MuseGlimmerCheckpointOptions,
    ) -> Result<ExecutionOutcome, MuseGlimmerCheckpointLoadError> {
        self.load_checkpoint_with(root, options, family_options)
    }

    /// Plans a discovered checkpoint under `family_options` and streams it into this model.
    fn load_discovered(
        &mut self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
        family_options: MuseGlimmerCheckpointOptions,
    ) -> Result<ExecutionOutcome, MuseGlimmerCheckpointLoadError> {
        let contract = self.config().checkpoint_contract(family_options)?;
        let plan = contract.build_load_plan(manifest, options)?;
        Ok(execute_local_load_plan(manifest, &plan, self)?)
    }
}

/// Parses `model.layers.<decimal>.<suffix>` without empty components.
fn parse_layer_target(target: &str) -> Result<(usize, &str), MuseGlimmerLoadError> {
    let remainder = target.strip_prefix("model.layers.").ok_or_else(|| {
        MuseGlimmerLoadError::UnknownTarget {
            target: target.into(),
        }
    })?;
    let (index, suffix) =
        remainder
            .split_once('.')
            .ok_or_else(|| MuseGlimmerLoadError::InvalidIndex {
                target: target.into(),
                reason: "expected `model.layers.<index>.<parameter>`".into(),
            })?;
    if suffix.is_empty() {
        return Err(MuseGlimmerLoadError::InvalidIndex {
            target: target.into(),
            reason: "parameter suffix must not be empty".into(),
        });
    }
    let index = index
        .parse::<usize>()
        .map_err(|error| MuseGlimmerLoadError::InvalidIndex {
            target: target.into(),
            reason: format!("`{index}` is not a decimal usize: {error}"),
        })?;
    Ok((index, suffix))
}

/// Dispatches a target within one resolved Muse decoder layer.
fn install_layer_tensor<B: Backend>(
    layer: &mut MuseGlimmerDecoderLayer<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), MuseGlimmerLoadError> {
    match suffix {
        "input_layernorm.weight" => install_norm(target, &mut layer.input_layernorm, data),
        "post_attention_layernorm.weight" => {
            install_norm(target, &mut layer.post_attention_layernorm, data)
        }
        "pre_feedforward_layernorm.weight" => {
            install_norm(target, &mut layer.pre_feedforward_layernorm, data)
        }
        "post_feedforward_layernorm.weight" => {
            install_norm(target, &mut layer.post_feedforward_layernorm, data)
        }
        "self_attn.q_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.q_proj, data)
        }
        "self_attn.k_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.k_proj, data)
        }
        "self_attn.v_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.v_proj, data)
        }
        "self_attn.gate_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.gate_proj, data)
        }
        "self_attn.o_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.o_proj, data)
        }
        "self_attn.q_proj.bias" => install_linear_bias(target, &mut layer.self_attn.q_proj, data),
        "self_attn.k_proj.bias" => install_linear_bias(target, &mut layer.self_attn.k_proj, data),
        "self_attn.v_proj.bias" => install_linear_bias(target, &mut layer.self_attn.v_proj, data),
        "self_attn.o_proj.bias" => install_linear_bias(target, &mut layer.self_attn.o_proj, data),
        "mlp.gate_proj.weight" => install_linear_weight(target, &mut layer.mlp.gate_proj, data),
        "mlp.up_proj.weight" => install_linear_weight(target, &mut layer.mlp.up_proj, data),
        "mlp.down_proj.weight" => install_linear_weight(target, &mut layer.mlp.down_proj, data),
        _ => Err(MuseGlimmerLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs a matrix through Burn's layout-aware record mapper.
fn install_linear_weight<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), MuseGlimmerLoadError> {
    install(target, &mut linear.weight, data)
}

/// Installs one enabled attention bias.
fn install_linear_bias<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), MuseGlimmerLoadError> {
    let bias = linear
        .bias
        .as_mut()
        .ok_or_else(|| MuseGlimmerLoadError::MissingComponent {
            target: target.into(),
            component: "configured attention projection bias",
        })?;
    install(target, bias, data)
}

/// Installs one learned RMSNorm scale vector.
fn install_norm<B: Backend>(
    target: &str,
    norm: &mut RmsNorm<B>,
    data: TensorData,
) -> Result<(), MuseGlimmerLoadError> {
    install(target, &mut norm.weight, data)
}

/// Validates and installs one statically ranked parameter, keeping packed weights exact.
///
/// Float data is installed unchanged, exactly as before packed loading existed.
/// Packed (QFloat) data from a quantized Burnpack sidecar stays packed only when
/// the backend reproduces the artifact's `[1, G]` blocks through the
/// parameter's load mapper (for example Metal with column-major linear
/// layouts). Otherwise, as on Burn Flex 0.21, whose quantized transpose
/// requantizes, the exact dequantized values `q * s` are installed in float
/// storage instead of a requantized approximation. The installed weight
/// therefore always equals the certified artifact; only its memory footprint
/// depends on the backend.
fn install<B: Backend, const D: usize>(
    target: &str,
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> Result<(), MuseGlimmerLoadError> {
    install_packed_parameter(parameter, data, PackedLayoutPolicy::DequantizeWhenInexact)
        .map(|_| ())
        .map_err(|source| MuseGlimmerLoadError::Parameter {
            target: target.into(),
            source,
        })
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
