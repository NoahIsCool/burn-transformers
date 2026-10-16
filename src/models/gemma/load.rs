//! Streamed parameter installation for Gemma checkpoints.
//!
//! The generic checkpoint runtime discovers every shard header and validates
//! the complete [`GemmaCheckpointContract`](super::GemmaCheckpointContract)
//! before this sink sees a payload; afterwards it holds at most one source
//! tensor and one converted tensor at a time. The sink only routes a model
//! field name (the contract's *target*) to the statically ranked Burn
//! parameter that owns it. Omitted vision-tower tensors never reach it.

use std::path::Path;

use burn::{
    module::Param,
    nn::Linear,
    tensor::{Tensor, TensorData, backend::Backend},
};
use thiserror::Error;

use crate::{
    checkpoint::{
        CheckpointError, CheckpointManifest, ExecutionFailure, ExecutionOutcome, LoadOptions,
        PackedLayoutPolicy, ParameterInstallError, TensorSink, execute_local_load_plan,
        install_packed_parameter,
    },
    feed_forward::GatedFeedForward,
    models::config::ConfigError,
    normalization::RmsNorm,
};

use super::{GemmaCheckpointOptions, GemmaCheckpointSource, GemmaDecoderLayer, GemmaForCausalLm};

/// Failure while installing one validated tensor into the graph.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GemmaLoadError {
    /// The target names no parameter of the Gemma graph.
    #[error("unknown Gemma parameter target `{target}`")]
    UnknownTarget {
        /// Unrecognized model field name.
        target: String,
    },
    /// A layer index was malformed or out of range.
    #[error("invalid Gemma target `{target}`: {reason}")]
    InvalidIndex {
        /// Malformed target name.
        target: String,
        /// Index diagnostic.
        reason: String,
    },
    /// The configured topology lacks the component a target addresses.
    #[error("Gemma target `{target}` requires missing component `{component}`")]
    MissingComponent {
        /// Parameter being installed.
        target: String,
        /// Component the configuration did not allocate.
        component: &'static str,
    },
    /// Tensor metadata disagreed with the statically ranked Burn parameter.
    #[error("cannot install Gemma parameter `{target}`: {source}")]
    Parameter {
        /// Model field name.
        target: String,
        /// Shape, rank, or element-kind failure.
        #[source]
        source: ParameterInstallError,
    },
    /// A tied-parameter alias violated the embedding/head contract.
    #[error("invalid Gemma alias `{target}` -> `{source_target}`: {reason}")]
    InvalidAlias {
        /// Alias target.
        target: String,
        /// Storage owner.
        source_target: String,
        /// Human-readable reason.
        reason: String,
    },
}

/// Failure from validation, namespace detection, strict planning, or streamed execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GemmaCheckpointLoadError {
    /// The configuration cannot describe a valid checkpoint.
    #[error(transparent)]
    Config(
        /// Field-level diagnostic.
        #[from]
        ConfigError,
    ),
    /// The manifest holds no (or several) Gemma language namespaces.
    #[error("{0}")]
    Namespace(
        /// Detection diagnostic naming the candidate embeddings.
        String,
    ),
    /// Discovery or expected/observed validation failed.
    #[error(transparent)]
    Checkpoint(
        /// Discovery or planning failure.
        #[from]
        CheckpointError,
    ),
    /// The sink failed after zero or more audited writes.
    #[error(transparent)]
    Execution(
        /// Sink failure with the writes completed so far.
        #[from]
        Box<ExecutionFailure<GemmaLoadError>>,
    ),
}

impl GemmaCheckpointLoadError {
    /// Returns partial execution diagnostics when the sink was reached.
    #[must_use]
    pub fn execution_failure(&self) -> Option<&ExecutionFailure<GemmaLoadError>> {
        match self {
            Self::Execution(failure) => Some(failure),
            Self::Config(_) | Self::Namespace(_) | Self::Checkpoint(_) => None,
        }
    }
}

impl<B: Backend> GemmaForCausalLm<B> {
    /// Strictly discovers, validates, and streams a local SafeTensors checkpoint.
    ///
    /// The published namespace (text-only, multimodal, or the original
    /// multimodal release) is detected from the manifest; a multimodal
    /// checkpoint's vision tower is reported as intentionally ignored. No
    /// network access is performed. A missing, unexpected, misshapen, or
    /// disallowed-dtype tensor fails before the first payload read. After
    /// an execution failure the partially initialized model must be
    /// discarded.
    ///
    /// # Errors
    ///
    /// Returns a [`GemmaCheckpointLoadError`] naming the failing stage.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, GemmaCheckpointLoadError> {
        let manifest = CheckpointManifest::discover(root, options)?;
        let source = GemmaCheckpointSource::detect(manifest.tensors.keys().map(String::as_str))
            .map_err(GemmaCheckpointLoadError::Namespace)?;
        self.load_discovered(&manifest, options, GemmaCheckpointOptions { source })
    }

    /// Loads a local checkpoint with an explicit parameter namespace.
    ///
    /// Use this to force the `source` layout; [`Self::load_checkpoint`]
    /// detects it.
    ///
    /// # Errors
    ///
    /// As [`Self::load_checkpoint`], without the namespace-detection error.
    pub fn load_checkpoint_with(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
        family_options: GemmaCheckpointOptions,
    ) -> Result<ExecutionOutcome, GemmaCheckpointLoadError> {
        let manifest = CheckpointManifest::discover(root, options)?;
        self.load_discovered(&manifest, options, family_options)
    }

    /// Plans a discovered checkpoint under `family_options` and streams it into this model.
    fn load_discovered(
        &mut self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
        family_options: GemmaCheckpointOptions,
    ) -> Result<ExecutionOutcome, GemmaCheckpointLoadError> {
        let contract = self.config().checkpoint_contract(family_options)?;
        let plan = contract.build_load_plan(manifest, options)?;
        Ok(execute_local_load_plan(manifest, &plan, self)?)
    }
}

impl<B: Backend> TensorSink for GemmaForCausalLm<B> {
    /// Contextual installation or alias failure.
    type Error = GemmaLoadError;

    /// Installs one storage-owning tensor named by the contract.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        match target {
            "model.embed_tokens.weight" => {
                install(target, &mut self.model.embed_tokens.weight, data)
            }
            "model.norm.weight" => install(target, &mut self.model.norm.weight, data),
            "lm_head.weight" => {
                let head = self
                    .lm_head
                    .as_mut()
                    .ok_or_else(|| missing(target, "independent LM head"))?;
                install(target, &mut head.weight, data)
            }
            _ => {
                let (index, suffix) = parse_layer_target(target)?;
                let count = self.model.layers.len();
                let layer = self.model.layers.get_mut(index).ok_or_else(|| {
                    GemmaLoadError::InvalidIndex {
                        target: target.into(),
                        reason: format!("layer {index} is outside 0..{count}"),
                    }
                })?;
                install_layer_tensor(layer, target, suffix, data)
            }
        }
    }

    /// Accepts only the tied embedding-to-head alias of a tied model.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        let reject = |reason: &str| GemmaLoadError::InvalidAlias {
            target: target.into(),
            source_target: source_target.into(),
            reason: reason.into(),
        };
        if target != "lm_head.weight" || source_target != "model.embed_tokens.weight" {
            return Err(reject(
                "only the tied embedding-to-LM-head alias is supported",
            ));
        }
        if !self.config().tie_word_embeddings || self.lm_head.is_some() {
            return Err(reject(
                "the configured model declares an independent LM head",
            ));
        }
        if !self.model.embed_tokens.weight.is_initialized() {
            return Err(reject("the embedding storage owner has not been installed"));
        }
        Ok(())
    }
}

/// Splits `model.layers.<index>.<suffix>` into a layer index and suffix.
fn parse_layer_target(target: &str) -> Result<(usize, &str), GemmaLoadError> {
    let invalid = |reason: String| GemmaLoadError::InvalidIndex {
        target: target.into(),
        reason,
    };
    let remainder =
        target
            .strip_prefix("model.layers.")
            .ok_or_else(|| GemmaLoadError::UnknownTarget {
                target: target.into(),
            })?;
    let (index, suffix) = remainder
        .split_once('.')
        .filter(|(_, suffix)| !suffix.is_empty())
        .ok_or_else(|| invalid("expected `model.layers.<index>.<parameter>`".into()))?;
    let index = index
        .parse::<usize>()
        .map_err(|error| invalid(format!("`{index}` is not a decimal usize: {error}")))?;
    Ok((index, suffix))
}

/// Routes a target inside one decoder block.
fn install_layer_tensor<B: Backend>(
    layer: &mut GemmaDecoderLayer<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), GemmaLoadError> {
    let attention = &mut layer.self_attn;
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
        "self_attn.q_proj.weight" => install(target, &mut attention.q_proj.weight, data),
        "self_attn.k_proj.weight" => install(target, &mut attention.k_proj.weight, data),
        "self_attn.v_proj.weight" => install(target, &mut attention.v_proj.weight, data),
        "self_attn.o_proj.weight" => install(target, &mut attention.o_proj.weight, data),
        "self_attn.q_proj.bias" => install_bias(target, &mut attention.q_proj, data),
        "self_attn.k_proj.bias" => install_bias(target, &mut attention.k_proj, data),
        "self_attn.v_proj.bias" => install_bias(target, &mut attention.v_proj, data),
        "self_attn.o_proj.bias" => install_bias(target, &mut attention.o_proj, data),
        "self_attn.query_norm.weight" => {
            let norm = attention
                .query_norm
                .as_mut()
                .ok_or_else(|| missing(target, "query RMSNorm"))?;
            install_norm(target, norm, data)
        }
        "self_attn.key_norm.weight" => {
            let norm = attention
                .key_norm
                .as_mut()
                .ok_or_else(|| missing(target, "key RMSNorm"))?;
            install_norm(target, norm, data)
        }
        _ => match suffix.strip_prefix("mlp.") {
            Some(projection) => install_mlp_tensor(&mut layer.mlp, target, projection, data),
            None => Err(GemmaLoadError::UnknownTarget {
                target: target.into(),
            }),
        },
    }
}

/// Routes one GeGLU projection weight.
fn install_mlp_tensor<B: Backend>(
    mlp: &mut GatedFeedForward<B>,
    target: &str,
    projection: &str,
    data: TensorData,
) -> Result<(), GemmaLoadError> {
    match projection {
        "gate_proj.weight" => install(target, &mut mlp.gate_proj.weight, data),
        "up_proj.weight" => install(target, &mut mlp.up_proj.weight, data),
        "down_proj.weight" => install(target, &mut mlp.down_proj.weight, data),
        _ => Err(GemmaLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs an enabled linear bias.
fn install_bias<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), GemmaLoadError> {
    let bias = linear
        .bias
        .as_mut()
        .ok_or_else(|| missing(target, "projection bias"))?;
    install(target, bias, data)
}

/// Installs one zero-centered RMSNorm weight `w` (scale `1 + w`).
fn install_norm<B: Backend>(
    target: &str,
    norm: &mut RmsNorm<B>,
    data: TensorData,
) -> Result<(), GemmaLoadError> {
    install(target, &mut norm.weight, data)
}

/// Validates and installs one statically ranked parameter, keeping packed weights exact.
///
/// Float data is installed unchanged. Packed (QFloat) data, which only a
/// quantized Burnpack sidecar produces, is installed as exactly the artifact:
/// packed where the backend keeps its blocks through the parameter's load
/// mapper, and otherwise as the exact dequantized values (see
/// [`install_packed_parameter`]), never as a requantized approximation.
fn install<B: Backend, const D: usize>(
    target: &str,
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> Result<(), GemmaLoadError> {
    install_packed_parameter(parameter, data, PackedLayoutPolicy::DequantizeWhenInexact)
        .map(|_| ())
        .map_err(|source| GemmaLoadError::Parameter {
            target: target.into(),
            source,
        })
}

/// Reports a target whose component the configuration did not allocate.
fn missing(target: &str, component: &'static str) -> GemmaLoadError {
    GemmaLoadError::MissingComponent {
        target: target.into(),
        component,
    }
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
