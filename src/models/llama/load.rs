//! Streamed parameter installation for Llama-family checkpoints.
//!
//! The generic checkpoint runtime discovers every shard header and validates
//! the complete [`LlamaCheckpointContract`](super::LlamaCheckpointContract)
//! before this sink sees a payload; afterwards it holds at most one source
//! tensor and one converted tensor at a time. The sink only routes a model
//! field name (the contract's *target*) to the statically ranked Burn
//! parameter that owns it.

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
        ParameterInstallError, TensorSink, execute_local_load_plan, install_float_parameter,
    },
    feed_forward::GatedFeedForward,
    models::config::ConfigError,
    normalization::RmsNorm,
};

use super::{LlamaCheckpointOptions, LlamaDecoderLayer, LlamaForCausalLm};

/// Failure while installing one validated tensor into the graph.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LlamaLoadError {
    /// The target names no parameter of the Llama-family graph.
    #[error("unknown Llama-family parameter target `{target}`")]
    UnknownTarget {
        /// Unrecognized model field name.
        target: String,
    },
    /// A layer index was malformed or out of range.
    #[error("invalid Llama-family target `{target}`: {reason}")]
    InvalidIndex {
        /// Malformed target name.
        target: String,
        /// Index diagnostic.
        reason: String,
    },
    /// The configured topology lacks the component a target addresses.
    #[error("Llama-family target `{target}` requires missing component `{component}`")]
    MissingComponent {
        /// Parameter being installed.
        target: String,
        /// Component the configuration did not allocate.
        component: &'static str,
    },
    /// Tensor metadata disagreed with the statically ranked Burn parameter.
    #[error("cannot install Llama-family parameter `{target}`: {source}")]
    Parameter {
        /// Model field name.
        target: String,
        /// Shape, rank, or element-kind failure.
        #[source]
        source: ParameterInstallError,
    },
    /// A tied-parameter alias violated the embedding/head contract.
    #[error("invalid Llama-family alias `{target}` -> `{source_target}`: {reason}")]
    InvalidAlias {
        /// Alias target.
        target: String,
        /// Storage owner.
        source_target: String,
        /// Human-readable reason.
        reason: String,
    },
}

/// Failure from validation, strict planning, or streamed execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LlamaCheckpointLoadError {
    /// The configuration cannot describe a valid checkpoint.
    #[error(transparent)]
    Config(
        /// Field-level diagnostic.
        #[from]
        ConfigError,
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
        Box<ExecutionFailure<LlamaLoadError>>,
    ),
}

impl LlamaCheckpointLoadError {
    /// Returns partial execution diagnostics when the sink was reached.
    #[must_use]
    pub fn execution_failure(&self) -> Option<&ExecutionFailure<LlamaLoadError>> {
        match self {
            Self::Execution(failure) => Some(failure),
            Self::Config(_) | Self::Checkpoint(_) => None,
        }
    }
}

impl<B: Backend> LlamaForCausalLm<B> {
    /// Strictly discovers, validates, and streams a local SafeTensors checkpoint.
    ///
    /// No network access is performed. A missing, unexpected, misshapen, or
    /// disallowed-dtype tensor fails before the first payload read. After an
    /// execution failure the partially initialized model must be discarded.
    ///
    /// # Errors
    ///
    /// Returns a [`LlamaCheckpointLoadError`] naming the failing stage.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, LlamaCheckpointLoadError> {
        self.load_checkpoint_with(root, options, LlamaCheckpointOptions::default())
    }

    /// Loads a local checkpoint under explicit family checkpoint options.
    ///
    /// [`LlamaCheckpointOptions`] has no fields yet, so this equals
    /// [`Self::load_checkpoint`]. It exists so every family offers the same
    /// `load_checkpoint` / `load_checkpoint_with` pair; see
    /// [concrete-family loading](crate::checkpoint#loading-a-concrete-family).
    ///
    /// # Errors
    ///
    /// As [`Self::load_checkpoint`].
    pub fn load_checkpoint_with(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
        family_options: LlamaCheckpointOptions,
    ) -> Result<ExecutionOutcome, LlamaCheckpointLoadError> {
        let manifest = CheckpointManifest::discover(root, options)?;
        let contract = self.config().checkpoint_contract(family_options)?;
        let plan = contract.build_load_plan(&manifest, options)?;
        Ok(execute_local_load_plan(&manifest, &plan, self)?)
    }
}

impl<B: Backend> TensorSink for LlamaForCausalLm<B> {
    /// Contextual installation or alias failure.
    type Error = LlamaLoadError;

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
                    LlamaLoadError::InvalidIndex {
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
        let reject = |reason: &str| LlamaLoadError::InvalidAlias {
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
fn parse_layer_target(target: &str) -> Result<(usize, &str), LlamaLoadError> {
    let invalid = |reason: String| LlamaLoadError::InvalidIndex {
        target: target.into(),
        reason,
    };
    let remainder =
        target
            .strip_prefix("model.layers.")
            .ok_or_else(|| LlamaLoadError::UnknownTarget {
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
    layer: &mut LlamaDecoderLayer<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), LlamaLoadError> {
    let attention = &mut layer.self_attn;
    match suffix {
        "attention_norm.weight" => install_norm(target, &mut layer.attention_norm, data),
        "mlp_norm.weight" => install_norm(target, &mut layer.mlp_norm, data),
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
            None => Err(LlamaLoadError::UnknownTarget {
                target: target.into(),
            }),
        },
    }
}

/// Routes one SwiGLU projection weight or bias.
fn install_mlp_tensor<B: Backend>(
    mlp: &mut GatedFeedForward<B>,
    target: &str,
    projection: &str,
    data: TensorData,
) -> Result<(), LlamaLoadError> {
    match projection {
        "gate_proj.weight" => install(target, &mut mlp.gate_proj.weight, data),
        "up_proj.weight" => install(target, &mut mlp.up_proj.weight, data),
        "down_proj.weight" => install(target, &mut mlp.down_proj.weight, data),
        "gate_proj.bias" => install_bias(target, &mut mlp.gate_proj, data),
        "up_proj.bias" => install_bias(target, &mut mlp.up_proj, data),
        "down_proj.bias" => install_bias(target, &mut mlp.down_proj, data),
        _ => Err(LlamaLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs an enabled linear bias.
fn install_bias<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), LlamaLoadError> {
    let bias = linear
        .bias
        .as_mut()
        .ok_or_else(|| missing(target, "projection bias"))?;
    install(target, bias, data)
}

/// Installs one RMSNorm scale.
fn install_norm<B: Backend>(
    target: &str,
    norm: &mut RmsNorm<B>,
    data: TensorData,
) -> Result<(), LlamaLoadError> {
    install(target, &mut norm.weight, data)
}

/// Validates and installs one statically ranked floating-point parameter.
fn install<B: Backend, const D: usize>(
    target: &str,
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> Result<(), LlamaLoadError> {
    install_float_parameter(parameter, data).map_err(|source| LlamaLoadError::Parameter {
        target: target.into(),
        source,
    })
}

/// Reports a target whose component the configuration did not allocate.
fn missing(target: &str, component: &'static str) -> LlamaLoadError {
    LlamaLoadError::MissingComponent {
        target: target.into(),
        component,
    }
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
