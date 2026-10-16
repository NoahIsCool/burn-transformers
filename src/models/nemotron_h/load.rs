//! Strict, shard-streamed installation of NVIDIA Nemotron-H parameters.
//!
//! [`NemotronHCheckpointContract`](super::NemotronHCheckpointContract) performs
//! the metadata pass first. Consequently this module never guesses a topology
//! from parameter names: every payload reaching [`TensorSink`] already has a
//! unique, shape-checked destination in the executable graph. Linear records are
//! installed with Burn's layout-aware mapper, while the depthwise convolution
//! and Mamba recurrence scalars retain their native tensor ranks.

use std::path::Path;

use burn::{
    module::Param,
    nn::Linear,
    tensor::{Tensor, TensorData, backend::Backend},
};
use thiserror::Error;

use crate::{
    attention::GroupedQueryAttention,
    checkpoint::{
        CheckpointError, ExecutionFailure, ExecutionOutcome, LoadOptions, ParameterInstallError,
        TensorSink, execute_local_load_plan, install_float_parameter,
    },
    normalization::RmsNorm,
};

use super::{
    ConfigError, NemotronHBlock, NemotronHCheckpointOptions, NemotronHForCausalLm,
    NemotronHNextTokenHead,
    model::{NemotronHDense, NemotronHMoe, mamba2::NemotronMamba2},
};

/// Architecture-specific failure while applying a validated Nemotron-H plan.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NemotronHLoadError {
    /// The target is absent from the executable Nemotron-H graph.
    #[error("unknown Nemotron-H parameter target `{target}`")]
    UnknownTarget {
        /// Unrecognized internal parameter name.
        target: String,
    },
    /// A block or expert index is malformed or outside the configured range.
    #[error("invalid Nemotron-H target `{target}`: {reason}")]
    InvalidIndex {
        /// Malformed parameter target.
        target: String,
        /// Human-readable index diagnostic.
        reason: String,
    },
    /// The target names a mixer omitted by the declared heterogeneous schedule.
    #[error("Nemotron-H target `{target}` requires missing component `{component}`")]
    MissingComponent {
        /// Parameter being installed.
        target: String,
        /// Required runtime component.
        component: &'static str,
    },
    /// Tensor metadata disagrees with the statically ranked Burn parameter.
    #[error("cannot install Nemotron-H parameter `{target}`: {source}")]
    Parameter {
        /// Internal parameter target.
        target: String,
        /// Shape, rank, or scalar-kind failure.
        #[source]
        source: ParameterInstallError,
    },
    /// A tied-parameter operation violated the embedding/head invariant.
    #[error("invalid Nemotron-H alias `{target}` -> `{source_target}`: {reason}")]
    InvalidAlias {
        /// Alias destination.
        target: String,
        /// Storage-owning source target.
        source_target: String,
        /// Human-readable invariant violation.
        reason: String,
    },
}

/// Failure from topology validation, strict planning, or streamed execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NemotronHCheckpointLoadError {
    /// The family configuration cannot describe a supported checkpoint.
    #[error(transparent)]
    Config(
        /// Unsupported or invalid model configuration.
        #[from]
        ConfigError,
    ),
    /// Artifact discovery or complete expected/observed validation failed.
    #[error(transparent)]
    Checkpoint(
        /// Metadata, mapping, or artifact-integrity failure.
        #[from]
        CheckpointError,
    ),
    /// Pass two failed after zero or more audited sink writes.
    #[error(transparent)]
    Execution(
        /// Sink failure with the execution progress recorded before it occurred.
        #[from]
        Box<ExecutionFailure<NemotronHLoadError>>,
    ),
}

impl NemotronHCheckpointLoadError {
    /// Returns partial pass-two diagnostics when execution reached the sink.
    #[must_use]
    pub fn execution_failure(&self) -> Option<&ExecutionFailure<NemotronHLoadError>> {
        match self {
            Self::Execution(failure) => Some(failure),
            Self::Config(_) | Self::Checkpoint(_) => None,
        }
    }
}

impl<B: Backend> TensorSink for NemotronHForCausalLm<B> {
    /// Parameter-target, component, shape, or alias installation failure.
    type Error = NemotronHLoadError;

    /// Installs one storage-owning tensor named by the strict family contract.
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
                        .ok_or_else(|| NemotronHLoadError::MissingComponent {
                            target: target.into(),
                            component: "independent LM head for an untied checkpoint",
                        })?;
                install_linear_weight(target, head, data)
            }
            target if target.starts_with("model.layers.") => {
                let (layer_index, suffix) =
                    parse_block_target(target, &target["model.layers.".len()..])?;
                let layer_count = self.model.layers.len();
                let layer = self.model.layers.get_mut(layer_index).ok_or_else(|| {
                    NemotronHLoadError::InvalidIndex {
                        target: target.into(),
                        reason: format!("block {layer_index} is outside 0..{layer_count}"),
                    }
                })?;
                install_block_tensor(layer, target, suffix, data)
            }
            target if target.starts_with("next_token_head.") => {
                let head = self.next_token_head.as_mut().ok_or_else(|| {
                    NemotronHLoadError::MissingComponent {
                        target: target.into(),
                        component: "checkpoint-declared next-token head",
                    }
                })?;
                install_next_token_tensor(
                    head,
                    target,
                    target.trim_start_matches("next_token_head."),
                    data,
                )
            }
            _ => Err(NemotronHLoadError::UnknownTarget {
                target: target.into(),
            }),
        }
    }

    /// Validates the only supported storage alias: embedding to vocabulary head.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        if target != "lm_head.weight" || source_target != "model.embed_tokens.weight" {
            return Err(NemotronHLoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "only the tied embedding-to-LM-head alias is supported".into(),
            });
        }
        if !self.model.config.tie_word_embeddings || self.lm_head.is_some() {
            return Err(NemotronHLoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the configured model declares an independent LM head".into(),
            });
        }
        if !self.model.embed_tokens.weight.is_initialized() {
            return Err(NemotronHLoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the embedding storage owner has not been installed".into(),
            });
        }
        Ok(())
    }
}

impl<B: Backend> NemotronHForCausalLm<B> {
    /// Discovers, validates, and streams one local SafeTensors checkpoint.
    ///
    /// This method performs no network access. Missing, unexpected, duplicate,
    /// wrong-shape, and disallowed-dtype tensors fail during pass one, before a
    /// payload is read. If pass two fails, discard the partially initialized
    /// model; the returned execution failure contains the audited partial report.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, NemotronHCheckpointLoadError> {
        self.load_checkpoint_with_architecture_options(
            root,
            options,
            NemotronHCheckpointOptions::default(),
        )
    }

    /// Loads with explicit family provenance for pinned artifact policies.
    ///
    /// The default loader intentionally remains generic and strict. The
    /// provenance-bearing path is required for NVIDIA Lightning revision
    /// `a9904d24bcc1d289a1950fa9d2b978c47cf903b9`, whose index overstates its
    /// physical payload by counting 23 base routers twice. Every source label,
    /// control hash, tensor name, dtype, shape, and byte total is revalidated
    /// before that one exception is applied.
    pub fn load_checkpoint_with_architecture_options(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
        architecture_options: NemotronHCheckpointOptions,
    ) -> Result<ExecutionOutcome, NemotronHCheckpointLoadError> {
        let contract = self.config().checkpoint_contract(architecture_options)?;
        let manifest = contract.discover_checkpoint(root, options)?;
        let plan = contract.build_load_plan(&manifest, options)?;
        Ok(execute_local_load_plan(&manifest, &plan, self)?)
    }
}

/// Splits the `<decimal>.<suffix>` remainder of a `model.layers.` target.
///
/// `remainder` is the text after the prefix the caller already matched;
/// `target` is the full name reported by diagnostics. A missing separator, an
/// empty suffix, and a non-decimal index are invalid.
fn parse_block_target<'a>(
    target: &str,
    remainder: &'a str,
) -> Result<(usize, &'a str), NemotronHLoadError> {
    let (index, suffix) =
        remainder
            .split_once('.')
            .ok_or_else(|| NemotronHLoadError::InvalidIndex {
                target: target.into(),
                reason: "expected `model.layers.<index>.<parameter>`".into(),
            })?;
    if suffix.is_empty() {
        return Err(NemotronHLoadError::InvalidIndex {
            target: target.into(),
            reason: "parameter suffix must not be empty".into(),
        });
    }
    let index = index
        .parse::<usize>()
        .map_err(|error| NemotronHLoadError::InvalidIndex {
            target: target.into(),
            reason: format!("`{index}` is not a decimal usize: {error}"),
        })?;
    Ok((index, suffix))
}

/// Dispatches one target inside a schedule-selected residual block.
fn install_block_tensor<B: Backend>(
    block: &mut NemotronHBlock<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), NemotronHLoadError> {
    match suffix {
        "norm.weight" => install_norm(target, &mut block.norm, data),
        suffix if suffix.starts_with("mamba.") => {
            let mamba =
                block
                    .mamba
                    .as_mut()
                    .ok_or_else(|| NemotronHLoadError::MissingComponent {
                        target: target.into(),
                        component: "Mamba2 mixer selected by the block schedule",
                    })?;
            install_mamba_tensor(mamba, target, &suffix["mamba.".len()..], data)
        }
        suffix if suffix.starts_with("attention.") => {
            let attention =
                block
                    .attention
                    .as_mut()
                    .ok_or_else(|| NemotronHLoadError::MissingComponent {
                        target: target.into(),
                        component: "attention mixer selected by the block schedule",
                    })?;
            install_attention_tensor(attention, target, &suffix["attention.".len()..], data)
        }
        suffix if suffix.starts_with("mlp.") => {
            let mlp = block
                .mlp
                .as_mut()
                .ok_or_else(|| NemotronHLoadError::MissingComponent {
                    target: target.into(),
                    component: "dense MLP selected by the block schedule",
                })?;
            install_dense_tensor(mlp, target, &suffix["mlp.".len()..], data)
        }
        suffix if suffix.starts_with("moe.") => {
            let moe = block
                .moe
                .as_mut()
                .ok_or_else(|| NemotronHLoadError::MissingComponent {
                    target: target.into(),
                    component: "routed MoE selected by the block schedule",
                })?;
            install_moe_tensor(moe, target, &suffix["moe.".len()..], data)
        }
        _ => Err(NemotronHLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs one fused-projection, convolution, scalar, or norm Mamba parameter.
fn install_mamba_tensor<B: Backend>(
    mamba: &mut NemotronMamba2<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), NemotronHLoadError> {
    match suffix {
        "in_proj.weight" => install_linear_weight(target, &mut mamba.in_proj, data),
        "in_proj.bias" => install_linear_bias(target, &mut mamba.in_proj, data),
        "conv1d.weight" => install(target, &mut mamba.conv1d.weight, data),
        "conv1d.bias" => {
            let bias =
                mamba
                    .conv1d
                    .bias
                    .as_mut()
                    .ok_or_else(|| NemotronHLoadError::MissingComponent {
                        target: target.into(),
                        component: "Mamba depthwise-convolution bias",
                    })?;
            install(target, bias, data)
        }
        "dt_bias" => install(target, &mut mamba.dt_bias, data),
        "a_log" => install(target, &mut mamba.a_log, data),
        "d" => install(target, &mut mamba.d, data),
        "norm.weight" => install(target, &mut mamba.norm.weight, data),
        "out_proj.weight" => install_linear_weight(target, &mut mamba.out_proj, data),
        "out_proj.bias" => install_linear_bias(target, &mut mamba.out_proj, data),
        _ => Err(NemotronHLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs one Q/K/V/O attention projection parameter.
fn install_attention_tensor<B: Backend>(
    attention: &mut GroupedQueryAttention<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), NemotronHLoadError> {
    match suffix {
        "q_proj.weight" => install_linear_weight(target, &mut attention.q_proj, data),
        "k_proj.weight" => install_linear_weight(target, &mut attention.k_proj, data),
        "v_proj.weight" => install_linear_weight(target, &mut attention.v_proj, data),
        "o_proj.weight" => install_linear_weight(target, &mut attention.o_proj, data),
        "q_proj.bias" => install_linear_bias(target, &mut attention.q_proj, data),
        "k_proj.bias" => install_linear_bias(target, &mut attention.k_proj, data),
        "v_proj.bias" => install_linear_bias(target, &mut attention.v_proj, data),
        "o_proj.bias" => install_linear_bias(target, &mut attention.o_proj, data),
        _ => Err(NemotronHLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs one non-gated ReLU² expert/MLP projection.
fn install_dense_tensor<B: Backend>(
    dense: &mut NemotronHDense<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), NemotronHLoadError> {
    match suffix {
        "up_proj.weight" => install_linear_weight(target, &mut dense.up_proj, data),
        "up_proj.bias" => install_linear_bias(target, &mut dense.up_proj, data),
        "down_proj.weight" => install_linear_weight(target, &mut dense.down_proj, data),
        "down_proj.bias" => install_linear_bias(target, &mut dense.down_proj, data),
        _ => Err(NemotronHLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs router, selected-expert, shared-expert, or optional latent mapping.
fn install_moe_tensor<B: Backend>(
    moe: &mut NemotronHMoe<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), NemotronHLoadError> {
    match suffix {
        "router.weight" => install_linear_weight(target, &mut moe.router, data),
        "correction_bias" => install(target, &mut moe.correction_bias, data),
        suffix if suffix.starts_with("routed.") => {
            let (expert_index, projection) =
                parse_expert_target(target, &suffix["routed.".len()..])?;
            let count = moe.routed.len();
            let expert = moe.routed.get_mut(expert_index).ok_or_else(|| {
                NemotronHLoadError::InvalidIndex {
                    target: target.into(),
                    reason: format!("expert {expert_index} is outside 0..{count}"),
                }
            })?;
            install_dense_tensor(expert, target, projection, data)
        }
        suffix if suffix.starts_with("shared.") => {
            install_dense_tensor(&mut moe.shared, target, &suffix["shared.".len()..], data)
        }
        suffix if suffix.starts_with("fc1_latent_proj.") => {
            let projection = moe.fc1_latent_proj.as_mut().ok_or_else(|| {
                NemotronHLoadError::MissingComponent {
                    target: target.into(),
                    component: "first routed-expert latent projection",
                }
            })?;
            install_linear_tensor(
                projection,
                target,
                &suffix["fc1_latent_proj.".len()..],
                data,
            )
        }
        suffix if suffix.starts_with("fc2_latent_proj.") => {
            let projection = moe.fc2_latent_proj.as_mut().ok_or_else(|| {
                NemotronHLoadError::MissingComponent {
                    target: target.into(),
                    component: "second routed-expert latent projection",
                }
            })?;
            install_linear_tensor(
                projection,
                target,
                &suffix["fc2_latent_proj.".len()..],
                data,
            )
        }
        _ => Err(NemotronHLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Splits the `<decimal>.<projection>` remainder of a `routed.` expert target.
///
/// `remainder` is the text after the `routed.` prefix the caller already
/// matched; `target` is the full name reported by diagnostics. An empty
/// projection is left to the expert's own dispatch, which rejects it as
/// unknown.
fn parse_expert_target<'a>(
    target: &str,
    remainder: &'a str,
) -> Result<(usize, &'a str), NemotronHLoadError> {
    let (index, projection) =
        remainder
            .split_once('.')
            .ok_or_else(|| NemotronHLoadError::InvalidIndex {
                target: target.into(),
                reason: "expected `routed.<index>.<parameter>`".into(),
            })?;
    let index = index
        .parse::<usize>()
        .map_err(|error| NemotronHLoadError::InvalidIndex {
            target: target.into(),
            reason: format!("invalid expert index `{index}`: {error}"),
        })?;
    Ok((index, projection))
}

/// Installs a weight or optional bias into a generic linear projection.
fn install_linear_tensor<B: Backend>(
    linear: &mut Linear<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), NemotronHLoadError> {
    match suffix {
        "weight" => install_linear_weight(target, linear, data),
        "bias" => install_linear_bias(target, linear, data),
        _ => Err(NemotronHLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs one parameter of Lightning's checkpoint-declared MTP body.
fn install_next_token_tensor<B: Backend>(
    head: &mut NemotronHNextTokenHead<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), NemotronHLoadError> {
    match suffix {
        "embedding_norm.weight" => install_norm(target, &mut head.embedding_norm, data),
        "hidden_norm.weight" => install_norm(target, &mut head.hidden_norm, data),
        "eh_proj.weight" => install_linear_weight(target, &mut head.eh_proj, data),
        "attention_norm.weight" => install_norm(target, &mut head.attention_norm, data),
        suffix if suffix.starts_with("attention.") => install_attention_tensor(
            &mut head.attention,
            target,
            &suffix["attention.".len()..],
            data,
        ),
        "moe_norm.weight" => install_norm(target, &mut head.moe_norm, data),
        suffix if suffix.starts_with("moe.") => {
            install_moe_tensor(&mut head.moe, target, &suffix["moe.".len()..], data)
        }
        "final_layer_norm.weight" => install_norm(target, &mut head.final_layer_norm, data),
        _ => Err(NemotronHLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs a checkpoint matrix through Burn's column-layout record mapper.
fn install_linear_weight<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), NemotronHLoadError> {
    install(target, &mut linear.weight, data)
}

/// Installs one enabled linear bias and rejects topology disagreement.
fn install_linear_bias<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), NemotronHLoadError> {
    let bias = linear
        .bias
        .as_mut()
        .ok_or_else(|| NemotronHLoadError::MissingComponent {
            target: target.into(),
            component: "configured linear bias",
        })?;
    install(target, bias, data)
}

/// Installs one ordinary RMSNorm scale.
fn install_norm<B: Backend>(
    target: &str,
    norm: &mut RmsNorm<B>,
    data: TensorData,
) -> Result<(), NemotronHLoadError> {
    install(target, &mut norm.weight, data)
}

/// Validates and installs one statically ranked floating-point parameter.
fn install<B: Backend, const D: usize>(
    target: &str,
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> Result<(), NemotronHLoadError> {
    install_float_parameter(parameter, data).map_err(|source| NemotronHLoadError::Parameter {
        target: target.into(),
        source,
    })
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
