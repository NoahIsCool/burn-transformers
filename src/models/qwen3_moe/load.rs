//! Burn parameter installation for Qwen3 MoE checkpoints.
//!
//! The generic checkpoint runtime discovers all shard headers and validates the
//! complete contract before this sink receives a payload. It then holds at most
//! one source tensor and one converted output tensor at a time. This family
//! bridge installs statically ranked Burn parameters without materializing a
//! checkpoint-sized host record or executing repository code.

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
    normalization::RmsNorm,
};

use super::{ConfigError, Qwen3MoeCheckpointOptions, Qwen3MoeDecoderLayer, Qwen3MoeForCausalLm};

/// Architecture-specific failure while applying a validated load plan.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Qwen3MoeLoadError {
    /// The target is not part of the executable Qwen3 MoE graph.
    #[error("unknown Qwen3 MoE parameter target `{target}`")]
    UnknownTarget {
        /// Unrecognized internal parameter name.
        target: String,
    },
    /// A layer or expert index was malformed or out of range.
    #[error("invalid Qwen3 MoE target `{target}`: {reason}")]
    InvalidIndex {
        /// Malformed target name.
        target: String,
        /// Index diagnostic.
        reason: String,
    },
    /// The configured dense/MoE topology lacks a required component.
    #[error("Qwen3 MoE target `{target}` requires missing component `{component}`")]
    MissingComponent {
        /// Parameter being installed.
        target: String,
        /// Expected layer component.
        component: &'static str,
    },
    /// Tensor metadata disagreed with the statically ranked Burn field.
    #[error("cannot install Qwen3 MoE parameter `{target}`: {source}")]
    Parameter {
        /// Internal parameter target.
        target: String,
        /// Shape, rank, or element-kind failure.
        #[source]
        source: ParameterInstallError,
    },
    /// A tied-parameter operation violated the embedding/head contract.
    #[error("invalid Qwen3 MoE alias `{target}` -> `{source_target}`: {reason}")]
    InvalidAlias {
        /// Alias target.
        target: String,
        /// Storage owner.
        source_target: String,
        /// Human-readable reason.
        reason: String,
    },
}

/// Failure from topology validation, strict planning, or streamed execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Qwen3MoeCheckpointLoadError {
    /// The model topology cannot describe a valid Qwen3 MoE checkpoint.
    #[error(transparent)]
    Config(
        /// Invalid configuration and its field-level diagnostic.
        #[from]
        ConfigError,
    ),
    /// Artifact discovery or complete expected/observed validation failed.
    #[error(transparent)]
    Checkpoint(
        /// Discovery or load-plan validation failure.
        #[from]
        CheckpointError,
    ),
    /// Pass two failed after zero or more audited sink writes.
    #[error(transparent)]
    Execution(
        /// Sink failure together with the writes and diagnostics completed so far.
        #[from]
        Box<ExecutionFailure<Qwen3MoeLoadError>>,
    ),
}

impl Qwen3MoeCheckpointLoadError {
    /// Returns partial pass-two diagnostics when execution reached the sink.
    #[must_use]
    pub fn execution_failure(&self) -> Option<&ExecutionFailure<Qwen3MoeLoadError>> {
        match self {
            Self::Execution(failure) => Some(failure),
            Self::Config(_) | Self::Checkpoint(_) => None,
        }
    }
}

impl<B: Backend> TensorSink for Qwen3MoeForCausalLm<B> {
    /// Contextual parameter installation or alias validation failure.
    type Error = Qwen3MoeLoadError;

    /// Installs one storage-owning tensor named by the checkpoint contract.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        match target {
            "model.embed_tokens.weight" => {
                install(target, &mut self.model.embed_tokens.weight, data)
            }
            "model.norm.weight" => install_norm(target, &mut self.model.norm, data),
            "lm_head.weight" => {
                let head =
                    self.lm_head
                        .as_mut()
                        .ok_or_else(|| Qwen3MoeLoadError::MissingComponent {
                            target: target.into(),
                            component: "independent LM head for an untied checkpoint",
                        })?;
                install_linear_weight(target, head, data)
            }
            _ => {
                let (layer_index, suffix) = parse_layer_target(target)?;
                let layer_count = self.model.layers.len();
                let layer = self.model.layers.get_mut(layer_index).ok_or_else(|| {
                    Qwen3MoeLoadError::InvalidIndex {
                        target: target.into(),
                        reason: format!("layer {layer_index} is outside 0..{layer_count}"),
                    }
                })?;
                install_layer_tensor(layer, target, suffix, data)
            }
        }
    }

    /// Validates the sole supported embedding-to-head storage alias.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        if target != "lm_head.weight" || source_target != "model.embed_tokens.weight" {
            return Err(Qwen3MoeLoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "only the tied embedding-to-LM-head alias is supported".into(),
            });
        }
        if !self.model.config.tie_word_embeddings || self.lm_head.is_some() {
            return Err(Qwen3MoeLoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the configured model declares an independent LM head".into(),
            });
        }
        if !self.model.embed_tokens.weight.is_initialized() {
            return Err(Qwen3MoeLoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the embedding storage owner has not been installed".into(),
            });
        }
        Ok(())
    }
}

impl<B: Backend> Qwen3MoeForCausalLm<B> {
    /// Strictly discovers, validates, and streams a local SafeTensors checkpoint.
    ///
    /// No network access is performed. A missing, unexpected, wrong-shape, or
    /// disallowed-dtype tensor fails before the first payload read. On execution
    /// failure the partially initialized model should be discarded, as with all
    /// sinks in the generic checkpoint runtime.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, Qwen3MoeCheckpointLoadError> {
        let manifest = CheckpointManifest::discover(root, options)?;
        let contract = self
            .config()
            .checkpoint_contract(Qwen3MoeCheckpointOptions::default())?;
        let plan = contract.build_load_plan(&manifest, options)?;
        Ok(execute_local_load_plan(&manifest, &plan, self)?)
    }
}

/// Parses `model.layers.<decimal>.<suffix>` without accepting empty segments.
fn parse_layer_target(target: &str) -> Result<(usize, &str), Qwen3MoeLoadError> {
    let remainder =
        target
            .strip_prefix("model.layers.")
            .ok_or_else(|| Qwen3MoeLoadError::UnknownTarget {
                target: target.into(),
            })?;
    let (index, suffix) =
        remainder
            .split_once('.')
            .ok_or_else(|| Qwen3MoeLoadError::InvalidIndex {
                target: target.into(),
                reason: "expected `model.layers.<index>.<parameter>`".into(),
            })?;
    if suffix.is_empty() {
        return Err(Qwen3MoeLoadError::InvalidIndex {
            target: target.into(),
            reason: "parameter suffix must not be empty".into(),
        });
    }
    let index = index
        .parse::<usize>()
        .map_err(|error| Qwen3MoeLoadError::InvalidIndex {
            target: target.into(),
            reason: format!("`{index}` is not a decimal usize: {error}"),
        })?;
    Ok((index, suffix))
}

/// Dispatches a target within one decoder layer.
fn install_layer_tensor<B: Backend>(
    layer: &mut Qwen3MoeDecoderLayer<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), Qwen3MoeLoadError> {
    match suffix {
        "input_layernorm.weight" => install_norm(target, &mut layer.input_layernorm, data),
        "post_attention_layernorm.weight" => {
            install_norm(target, &mut layer.post_attention_layernorm, data)
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
        "self_attn.o_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.o_proj, data)
        }
        "self_attn.q_proj.bias" => install_linear_bias(target, &mut layer.self_attn.q_proj, data),
        "self_attn.k_proj.bias" => install_linear_bias(target, &mut layer.self_attn.k_proj, data),
        "self_attn.v_proj.bias" => install_linear_bias(target, &mut layer.self_attn.v_proj, data),
        "self_attn.o_proj.bias" => install_linear_bias(target, &mut layer.self_attn.o_proj, data),
        "self_attn.q_norm.weight" => {
            let norm = layer.self_attn.query_norm.as_mut().ok_or_else(|| {
                Qwen3MoeLoadError::MissingComponent {
                    target: target.into(),
                    component: "query RMSNorm",
                }
            })?;
            install_norm(target, norm, data)
        }
        "self_attn.k_norm.weight" => {
            let norm = layer.self_attn.key_norm.as_mut().ok_or_else(|| {
                Qwen3MoeLoadError::MissingComponent {
                    target: target.into(),
                    component: "key RMSNorm",
                }
            })?;
            install_norm(target, norm, data)
        }
        "sparse_mlp.router.weight" => {
            let moe =
                layer
                    .sparse_mlp
                    .as_mut()
                    .ok_or_else(|| Qwen3MoeLoadError::MissingComponent {
                        target: target.into(),
                        component: "routed MoE",
                    })?;
            install_linear_weight(target, &mut moe.router, data)
        }
        suffix if suffix.starts_with("sparse_mlp.routed.") => {
            let moe =
                layer
                    .sparse_mlp
                    .as_mut()
                    .ok_or_else(|| Qwen3MoeLoadError::MissingComponent {
                        target: target.into(),
                        component: "routed MoE",
                    })?;
            let (expert, projection) = parse_expert_target(target, suffix)?;
            let expert_count = moe.routed.len();
            let expert =
                moe.routed
                    .get_mut(expert)
                    .ok_or_else(|| Qwen3MoeLoadError::InvalidIndex {
                        target: target.into(),
                        reason: format!("expert index is outside 0..{expert_count}"),
                    })?;
            install_mlp_tensor(expert, target, projection, data)
        }
        suffix if suffix.starts_with("dense_mlp.") => {
            let dense =
                layer
                    .dense_mlp
                    .as_mut()
                    .ok_or_else(|| Qwen3MoeLoadError::MissingComponent {
                        target: target.into(),
                        component: "dense SwiGLU",
                    })?;
            install_mlp_tensor(dense, target, &suffix["dense_mlp.".len()..], data)
        }
        _ => Err(Qwen3MoeLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Parses `sparse_mlp.routed.<decimal>.<projection>`.
fn parse_expert_target<'a>(
    target: &str,
    suffix: &'a str,
) -> Result<(usize, &'a str), Qwen3MoeLoadError> {
    let remainder = suffix.strip_prefix("sparse_mlp.routed.").ok_or_else(|| {
        Qwen3MoeLoadError::UnknownTarget {
            target: target.into(),
        }
    })?;
    let (index, projection) =
        remainder
            .split_once('.')
            .ok_or_else(|| Qwen3MoeLoadError::InvalidIndex {
                target: target.into(),
                reason: "expected `sparse_mlp.routed.<index>.<projection>`".into(),
            })?;
    let index = index
        .parse::<usize>()
        .map_err(|error| Qwen3MoeLoadError::InvalidIndex {
            target: target.into(),
            reason: format!("invalid expert index `{index}`: {error}"),
        })?;
    Ok((index, projection))
}

/// Installs one gated-MLP projection.
fn install_mlp_tensor<B: Backend>(
    mlp: &mut GatedFeedForward<B>,
    target: &str,
    projection: &str,
    data: TensorData,
) -> Result<(), Qwen3MoeLoadError> {
    match projection {
        "gate_proj.weight" => install_linear_weight(target, &mut mlp.gate_proj, data),
        "up_proj.weight" => install_linear_weight(target, &mut mlp.up_proj, data),
        "down_proj.weight" => install_linear_weight(target, &mut mlp.down_proj, data),
        _ => Err(Qwen3MoeLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs a linear weight through Burn's layout-aware record mapper.
fn install_linear_weight<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), Qwen3MoeLoadError> {
    install(target, &mut linear.weight, data)
}

/// Installs an enabled linear bias.
fn install_linear_bias<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), Qwen3MoeLoadError> {
    let bias = linear
        .bias
        .as_mut()
        .ok_or_else(|| Qwen3MoeLoadError::MissingComponent {
            target: target.into(),
            component: "attention projection bias",
        })?;
    install(target, bias, data)
}

/// Installs one RMSNorm scale.
fn install_norm<B: Backend>(
    target: &str,
    norm: &mut RmsNorm<B>,
    data: TensorData,
) -> Result<(), Qwen3MoeLoadError> {
    install(target, &mut norm.weight, data)
}

/// Validates and installs one statically ranked floating-point parameter.
fn install<B: Backend, const D: usize>(
    target: &str,
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> Result<(), Qwen3MoeLoadError> {
    install_float_parameter(parameter, data).map_err(|source| Qwen3MoeLoadError::Parameter {
        target: target.into(),
        source,
    })
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
