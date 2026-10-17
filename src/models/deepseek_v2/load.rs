//! Burn parameter installation for strict DeepSeek-V2 SafeTensors plans.
//!
//! Pass one validates every shard header and the complete family inventory.
//! Pass two gives this sink one transformed tensor at a time. Linear parameters
//! retain Burn's column-layout mapper so checkpoint `[out, in]` matrices become
//! gradient-safe runtime `[in, out]` tensors without a full-model staging record.

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
    normalization::RmsNorm,
};

use super::{
    ConfigError, DeepseekV2CheckpointOptions, DeepseekV2DecoderLayer, DeepseekV2ForCausalLm,
};

/// Architecture-specific failure while applying a validated load plan.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeepseekV2LoadError {
    /// The target is not part of the executable graph.
    #[error("unknown DeepSeek-V2 parameter target `{target}`")]
    UnknownTarget {
        /// Unrecognized internal parameter name.
        target: String,
    },
    /// A layer or expert index was malformed or out of range.
    #[error("invalid DeepSeek-V2 target `{target}`: {reason}")]
    InvalidIndex {
        /// Malformed target name.
        target: String,
        /// Index diagnostic.
        reason: String,
    },
    /// The configured topology lacks a parameter required by the target.
    #[error("DeepSeek-V2 target `{target}` requires missing component `{component}`")]
    MissingComponent {
        /// Parameter being installed.
        target: String,
        /// Expected model component.
        component: &'static str,
    },
    /// Tensor metadata disagreed with the statically ranked Burn field.
    #[error("cannot install DeepSeek-V2 parameter `{target}`: {source}")]
    Parameter {
        /// Internal parameter target.
        target: String,
        /// Shape, rank, or dtype failure.
        #[source]
        source: ParameterInstallError,
    },
    /// A tied-parameter alias violated the embedding/head contract.
    #[error("invalid DeepSeek-V2 alias `{target}` -> `{source_target}`: {reason}")]
    InvalidAlias {
        /// Alias target.
        target: String,
        /// Storage-owning source target.
        source_target: String,
        /// Human-readable reason.
        reason: String,
    },
}

/// Failure from topology validation, strict planning, or streamed execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DeepseekV2CheckpointLoadError {
    /// Model topology is invalid.
    #[error(transparent)]
    Config(
        /// Topology or dimension validation failure.
        #[from]
        ConfigError,
    ),
    /// Artifact discovery or strict inventory validation failed.
    #[error(transparent)]
    Checkpoint(
        /// Artifact discovery or inventory-planning failure.
        #[from]
        CheckpointError,
    ),
    /// Pass two failed after zero or more audited writes.
    #[error(transparent)]
    Execution(
        /// Streamed execution failure with its partial write report.
        #[from]
        Box<ExecutionFailure<DeepseekV2LoadError>>,
    ),
}

impl DeepseekV2CheckpointLoadError {
    /// Returns partial pass-two diagnostics when execution reached the sink.
    #[must_use]
    pub fn execution_failure(&self) -> Option<&ExecutionFailure<DeepseekV2LoadError>> {
        match self {
            Self::Execution(failure) => Some(failure),
            Self::Config(_) | Self::Checkpoint(_) => None,
        }
    }
}

impl<B: Backend> TensorSink for DeepseekV2ForCausalLm<B> {
    /// Typed parameter-target, tensor-shape, and tied-alias installation failure.
    type Error = DeepseekV2LoadError;

    /// Installs one planned tensor into its uniquely typed model parameter.
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
                        .ok_or_else(|| DeepseekV2LoadError::MissingComponent {
                            target: target.into(),
                            component: "independent LM head for an untied checkpoint",
                        })?;
                install_linear_weight(target, head, data)
            }
            _ => {
                let (layer_index, suffix) = parse_layer_target(target)?;
                let layer_count = self.model.layers.len();
                let layer = self.model.layers.get_mut(layer_index).ok_or_else(|| {
                    DeepseekV2LoadError::InvalidIndex {
                        target: target.into(),
                        reason: format!("layer {layer_index} is outside 0..{layer_count}"),
                    }
                })?;
                install_layer_tensor(layer, target, suffix, data)
            }
        }
    }

    /// Verifies the sole logical embedding-to-head alias used by tied models.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        if target != "lm_head.weight" || source_target != "model.embed_tokens.weight" {
            return Err(DeepseekV2LoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "only the tied embedding-to-LM-head alias is supported".into(),
            });
        }
        if !self.model.config.tie_word_embeddings || self.lm_head.is_some() {
            return Err(DeepseekV2LoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the configured model declares an independent LM head".into(),
            });
        }
        if !self.model.embed_tokens.weight.is_initialized() {
            return Err(DeepseekV2LoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the embedding storage owner has not been installed".into(),
            });
        }
        Ok(())
    }
}

impl<B: Backend> DeepseekV2ForCausalLm<B> {
    /// Strictly discovers, validates, and streams a local checkpoint.
    ///
    /// The method performs no network access. On execution failure the partially
    /// initialized model should be discarded.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, DeepseekV2CheckpointLoadError> {
        self.load_checkpoint_with(root, options, DeepseekV2CheckpointOptions::default())
    }

    /// Loads a local checkpoint under explicit family checkpoint options.
    ///
    /// [`DeepseekV2CheckpointOptions`] has no fields yet, so this equals
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
        family_options: DeepseekV2CheckpointOptions,
    ) -> Result<ExecutionOutcome, DeepseekV2CheckpointLoadError> {
        let manifest = CheckpointManifest::discover(root, options)?;
        let contract = self.config().checkpoint_contract(family_options)?;
        let plan = contract.build_load_plan(&manifest, options)?;
        Ok(execute_local_load_plan(&manifest, &plan, self)?)
    }
}

/// Parses `model.layers.<decimal>.<suffix>`.
fn parse_layer_target(target: &str) -> Result<(usize, &str), DeepseekV2LoadError> {
    let remainder =
        target
            .strip_prefix("model.layers.")
            .ok_or_else(|| DeepseekV2LoadError::UnknownTarget {
                target: target.into(),
            })?;
    let (index, suffix) =
        remainder
            .split_once('.')
            .ok_or_else(|| DeepseekV2LoadError::InvalidIndex {
                target: target.into(),
                reason: "expected `model.layers.<index>.<parameter>`".into(),
            })?;
    if suffix.is_empty() {
        return Err(DeepseekV2LoadError::InvalidIndex {
            target: target.into(),
            reason: "parameter suffix must not be empty".into(),
        });
    }
    let index = index
        .parse::<usize>()
        .map_err(|error| DeepseekV2LoadError::InvalidIndex {
            target: target.into(),
            reason: format!("`{index}` is not a decimal usize: {error}"),
        })?;
    Ok((index, suffix))
}

/// Dispatches a target within one decoder layer.
fn install_layer_tensor<B: Backend>(
    layer: &mut DeepseekV2DecoderLayer<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), DeepseekV2LoadError> {
    match suffix {
        "input_layernorm.weight" => install_norm(target, &mut layer.input_layernorm, data),
        "post_attention_layernorm.weight" => {
            install_norm(target, &mut layer.post_attention_layernorm, data)
        }
        "self_attn.q_proj.weight" => install_optional_linear_weight(
            target,
            &mut layer.self_attn.q_proj,
            "direct query projection",
            data,
        ),
        "self_attn.q_a_proj.weight" => install_optional_linear_weight(
            target,
            &mut layer.self_attn.q_a_proj,
            "query down-projection",
            data,
        ),
        "self_attn.q_a_proj.bias" => install_optional_linear_bias(
            target,
            &mut layer.self_attn.q_a_proj,
            "query down-projection bias",
            data,
        ),
        "self_attn.q_a_layernorm.weight" => {
            let norm = layer.self_attn.q_a_layernorm.as_mut().ok_or_else(|| {
                DeepseekV2LoadError::MissingComponent {
                    target: target.into(),
                    component: "query latent RMSNorm",
                }
            })?;
            install_norm(target, norm, data)
        }
        "self_attn.q_b_proj.weight" => install_optional_linear_weight(
            target,
            &mut layer.self_attn.q_b_proj,
            "query up-projection",
            data,
        ),
        "self_attn.kv_a_proj_with_mqa.weight" => {
            install_linear_weight(target, &mut layer.self_attn.kv_a_proj_with_mqa, data)
        }
        "self_attn.kv_a_proj_with_mqa.bias" => {
            install_linear_bias(target, &mut layer.self_attn.kv_a_proj_with_mqa, data)
        }
        "self_attn.kv_a_layernorm.weight" => {
            install_norm(target, &mut layer.self_attn.kv_a_layernorm, data)
        }
        "self_attn.kv_b_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.kv_b_proj, data)
        }
        "self_attn.o_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.o_proj, data)
        }
        "self_attn.o_proj.bias" => install_linear_bias(target, &mut layer.self_attn.o_proj, data),
        "sparse_mlp.routed.router.weight" => {
            let sparse = sparse_mlp(layer, target)?;
            install_linear_weight(target, &mut sparse.routed.router, data)
        }
        "sparse_mlp.routed.correction_bias" => {
            let sparse = sparse_mlp(layer, target)?;
            let bias = sparse.routed.correction_bias.as_mut().ok_or_else(|| {
                DeepseekV2LoadError::MissingComponent {
                    target: target.into(),
                    component: "noaux_tc selection bias",
                }
            })?;
            install(target, bias, data)
        }
        _ if suffix.starts_with("sparse_mlp.routed.routed.") => {
            let remainder = &suffix["sparse_mlp.routed.routed.".len()..];
            let (expert_index, parameter) = parse_expert_suffix(target, remainder)?;
            let sparse = sparse_mlp(layer, target)?;
            let expert_count = sparse.routed.routed.len();
            let expert = sparse.routed.routed.get_mut(expert_index).ok_or_else(|| {
                DeepseekV2LoadError::InvalidIndex {
                    target: target.into(),
                    reason: format!("expert {expert_index} is outside 0..{expert_count}"),
                }
            })?;
            install_mlp_tensor(expert, target, parameter, data)
        }
        _ if suffix.starts_with("sparse_mlp.shared_experts.") => {
            let parameter = &suffix["sparse_mlp.shared_experts.".len()..];
            let sparse = sparse_mlp(layer, target)?;
            install_mlp_tensor(&mut sparse.shared_experts, target, parameter, data)
        }
        _ if suffix.starts_with("dense_mlp.") => {
            let parameter = &suffix["dense_mlp.".len()..];
            let dense =
                layer
                    .dense_mlp
                    .as_mut()
                    .ok_or_else(|| DeepseekV2LoadError::MissingComponent {
                        target: target.into(),
                        component: "dense SwiGLU block",
                    })?;
            install_mlp_tensor(dense, target, parameter, data)
        }
        _ => Err(DeepseekV2LoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Returns the sparse block or a contextual topology error.
fn sparse_mlp<'a, B: Backend>(
    layer: &'a mut DeepseekV2DecoderLayer<B>,
    target: &str,
) -> Result<&'a mut super::model::DeepseekV2Moe<B>, DeepseekV2LoadError> {
    layer
        .sparse_mlp
        .as_mut()
        .ok_or_else(|| DeepseekV2LoadError::MissingComponent {
            target: target.into(),
            component: "sparse routed/shared expert block",
        })
}

/// Parses `<expert>.<parameter>`.
fn parse_expert_suffix<'a>(
    target: &str,
    remainder: &'a str,
) -> Result<(usize, &'a str), DeepseekV2LoadError> {
    let (index, parameter) =
        remainder
            .split_once('.')
            .ok_or_else(|| DeepseekV2LoadError::InvalidIndex {
                target: target.into(),
                reason: "expected an expert index followed by a parameter".into(),
            })?;
    let index = index
        .parse::<usize>()
        .map_err(|error| DeepseekV2LoadError::InvalidIndex {
            target: target.into(),
            reason: format!("`{index}` is not a decimal expert index: {error}"),
        })?;
    Ok((index, parameter))
}

/// Installs one of the three SwiGLU matrices.
fn install_mlp_tensor<B: Backend>(
    mlp: &mut GatedFeedForward<B>,
    target: &str,
    parameter: &str,
    data: TensorData,
) -> Result<(), DeepseekV2LoadError> {
    match parameter {
        "gate_proj.weight" => install_linear_weight(target, &mut mlp.gate_proj, data),
        "up_proj.weight" => install_linear_weight(target, &mut mlp.up_proj, data),
        "down_proj.weight" => install_linear_weight(target, &mut mlp.down_proj, data),
        _ => Err(DeepseekV2LoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs a required rank-two linear weight.
fn install_linear_weight<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), DeepseekV2LoadError> {
    install(target, &mut linear.weight, data)
}

/// Installs a required rank-one linear bias.
fn install_linear_bias<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), DeepseekV2LoadError> {
    let bias = linear
        .bias
        .as_mut()
        .ok_or_else(|| DeepseekV2LoadError::MissingComponent {
            target: target.into(),
            component: "configured linear bias",
        })?;
    install(target, bias, data)
}

/// Installs an optional linear weight required by the mapped query topology.
fn install_optional_linear_weight<B: Backend>(
    target: &str,
    linear: &mut Option<Linear<B>>,
    component: &'static str,
    data: TensorData,
) -> Result<(), DeepseekV2LoadError> {
    let linear = linear
        .as_mut()
        .ok_or_else(|| DeepseekV2LoadError::MissingComponent {
            target: target.into(),
            component,
        })?;
    install_linear_weight(target, linear, data)
}

/// Installs an optional linear bias required by the mapped query topology.
fn install_optional_linear_bias<B: Backend>(
    target: &str,
    linear: &mut Option<Linear<B>>,
    component: &'static str,
    data: TensorData,
) -> Result<(), DeepseekV2LoadError> {
    let linear = linear
        .as_mut()
        .ok_or_else(|| DeepseekV2LoadError::MissingComponent {
            target: target.into(),
            component,
        })?;
    install_linear_bias(target, linear, data)
}

/// Installs one RMS scale parameter.
fn install_norm<B: Backend>(
    target: &str,
    norm: &mut RmsNorm<B>,
    data: TensorData,
) -> Result<(), DeepseekV2LoadError> {
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
) -> Result<(), DeepseekV2LoadError> {
    install_packed_parameter(parameter, data, PackedLayoutPolicy::DequantizeWhenInexact)
        .map(|_| ())
        .map_err(|source| DeepseekV2LoadError::Parameter {
            target: target.into(),
            source,
        })
}

#[cfg(test)]
#[path = "load_tests.rs"]
/// Successful streamed loading and malformed parameter-target regressions.
mod tests;
