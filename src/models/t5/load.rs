//! Architecture-owned Burn parameter installation for T5 checkpoints.
//!
//! Checkpoint planning operates on dynamic names, ranks, and byte buffers, while
//! Burn encodes tensor rank and linear layouts in concrete Rust fields.
//! The [`TensorSink`] implementation below is therefore deliberately owned by
//! the T5 architecture. It accepts only names emitted by [`super::T5CheckpointContract`]
//! and installs weights through Burn's record-loading path. The generic load plan
//! has already transformed official T5 `[output, input]` matrices into Burn's
//! gradient-safe `[input, output]` runtime layout. Keeping layout conversion out
//! of this sink makes every transpose visible in [`crate::checkpoint::LoadReport`]
//! and ensures direct writes obey the same runtime-shape contract.
//!
//! Tied T5 heads require no second Burn parameter: the forward graph multiplies
//! by `shared.weight^T` directly. The sink still validates the logical alias after
//! the shared storage owner has been installed, allowing the generic loader's
//! report and memory accounting to describe the mathematical parameter sharing.

use std::path::Path;

use burn::{
    module::Param,
    nn::Linear,
    tensor::{Tensor, TensorData, backend::Backend},
};
use thiserror::Error;

use crate::{
    checkpoint::{
        CheckpointError, ExecutionFailure, ExecutionOutcome, LoadOptions, PackedLayoutPolicy,
        ParameterInstallError, TensorSink, execute_local_load_plan, install_packed_parameter,
    },
    normalization::RmsNorm,
};

use super::{
    ConfigError, T5Attention, T5Block, T5ForConditionalGeneration, T5LayerFeedForward,
    checkpoint::feed_forward_sublayer,
};

/// Architecture-specific failure while applying one validated T5 tensor.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum T5LoadError {
    /// A target was not part of the executable T5 graph.
    #[error("unknown T5 parameter target `{target}`")]
    UnknownTarget {
        /// Unrecognized checkpoint-contract target.
        target: String,
    },
    /// A stack target did not contain a valid in-range decimal block index.
    #[error("invalid T5 block target `{target}`: {reason}")]
    InvalidBlock {
        /// Malformed or out-of-range target.
        target: String,
        /// Contextual index diagnostic.
        reason: String,
    },
    /// A public model field disagreed with the validated checkpoint topology.
    #[error("T5 target `{target}` requires missing component `{component}`")]
    MissingComponent {
        /// Parameter whose owning module was absent.
        target: String,
        /// Required field or topology variant.
        component: &'static str,
    },
    /// Tensor metadata disagreed with a statically ranked Burn parameter.
    #[error("cannot install T5 parameter `{target}`: {source}")]
    Parameter {
        /// Internal parameter target.
        target: String,
        /// Rank, shape, or scalar-kind disagreement.
        #[source]
        source: ParameterInstallError,
    },
    /// A tied-weight operation violated T5's embedding/head contract.
    #[error("invalid T5 parameter alias `{target}` -> `{source_target}`: {reason}")]
    InvalidAlias {
        /// Logical alias target requested by the plan.
        target: String,
        /// Storage-owning target requested by the plan.
        source_target: String,
        /// Human-readable contract violation.
        reason: String,
    },
}

/// Failure from T5 contract derivation, strict planning, or streamed execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum T5CheckpointLoadError {
    /// The serialized T5 topology is not mathematically valid.
    #[error(transparent)]
    Config(
        /// Architecture validation error raised before checkpoint planning.
        #[from]
        ConfigError,
    ),
    /// Artifact discovery or strict expected-versus-observed planning failed.
    #[error(transparent)]
    Checkpoint(
        /// Artifact or mapping error raised before tensor installation.
        #[from]
        CheckpointError,
    ),
    /// Pass two failed after zero or more audited model writes.
    #[error(transparent)]
    Execution(
        /// Partial execution report and the cause of the failed streamed write.
        #[from]
        Box<ExecutionFailure<T5LoadError>>,
    ),
}

impl T5CheckpointLoadError {
    /// Returns partial execution diagnostics when pass two reached the tensor sink.
    #[must_use]
    pub fn execution_failure(&self) -> Option<&ExecutionFailure<T5LoadError>> {
        match self {
            Self::Execution(failure) => Some(failure),
            Self::Config(_) | Self::Checkpoint(_) => None,
        }
    }
}

impl<B: Backend> T5ForConditionalGeneration<B> {
    /// Strictly plans and streams a local SafeTensors checkpoint into this model.
    ///
    /// Pass one is [`super::T5Config::plan_checkpoint`]: it reports
    /// configuration errors, discovers the checkpoint, rejects one holding
    /// fewer tensors than encoder plus decoder blocks with
    /// [`CheckpointError::TooFewTensors`] before the per-parameter contract is
    /// built, and reads only bounded metadata to reject missing, unexpected, or
    /// shape-incompatible tensors before pass two initializes any model
    /// parameter. Pass two holds at most one source payload and one output
    /// buffer at a time. On an execution failure the returned error retains the
    /// exact partial [`crate::checkpoint::LoadReport`]; callers should discard
    /// the partially initialized model rather than continue inference.
    ///
    /// # Errors
    ///
    /// A [`T5CheckpointLoadError`] naming the failing stage.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, T5CheckpointLoadError> {
        let planned = self.config.plan_checkpoint(root, options)?;
        Ok(execute_local_load_plan(
            planned.manifest(),
            planned.load_plan(),
            self,
        )?)
    }
}

impl<B: Backend> TensorSink for T5ForConditionalGeneration<B> {
    /// Target-dispatch, parameter-installation, or tied-alias failure.
    type Error = T5LoadError;

    /// Installs one storage-owning parameter recognized by the T5 contract.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        match target {
            "shared.weight" => install(target, &mut self.shared.weight, data),
            "encoder.final_layer_norm.weight" => {
                install_norm(target, &mut self.encoder.final_layer_norm, data)
            }
            "decoder.final_layer_norm.weight" => {
                install_norm(target, &mut self.decoder.final_layer_norm, data)
            }
            "lm_head.weight" => {
                let head = self
                    .lm_head
                    .as_mut()
                    .ok_or_else(|| T5LoadError::MissingComponent {
                        target: target.into(),
                        component: "independent LM head for an untied checkpoint",
                    })?;
                install_linear_weight(target, head, data)
            }
            _ => {
                let (stack, index, suffix) = parse_block_target(target)?;
                let blocks = match stack {
                    Stack::Encoder => &mut self.encoder.block,
                    Stack::Decoder => &mut self.decoder.block,
                };
                let block_count = blocks.len();
                let block = blocks
                    .get_mut(index)
                    .ok_or_else(|| T5LoadError::InvalidBlock {
                        target: target.into(),
                        reason: format!("index {index} is outside 0..{block_count}"),
                    })?;
                install_block_tensor(block, stack, target, suffix, data)
            }
        }
    }

    /// Validates T5's single supported logical storage alias.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        if target != "lm_head.weight" || source_target != "shared.weight" {
            return Err(T5LoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "only the shared-embedding-to-LM-head alias is supported".into(),
            });
        }
        if !self.config.tie_word_embeddings || self.lm_head.is_some() {
            return Err(T5LoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the configured checkpoint declares an independent LM head".into(),
            });
        }
        if !self.shared.weight.is_initialized() {
            return Err(T5LoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the shared embedding storage owner has not been installed".into(),
            });
        }
        Ok(())
    }
}

/// Selects one of T5's structurally different stacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stack {
    /// Bidirectional encoder without cross-attention.
    Encoder,
    /// Causal decoder with encoder-decoder attention.
    Decoder,
}

/// Parses `<stack>.block.<index>.<suffix>` without accepting empty segments.
fn parse_block_target(target: &str) -> Result<(Stack, usize, &str), T5LoadError> {
    let (stack, remainder) = if let Some(remainder) = target.strip_prefix("encoder.block.") {
        (Stack::Encoder, remainder)
    } else if let Some(remainder) = target.strip_prefix("decoder.block.") {
        (Stack::Decoder, remainder)
    } else {
        return Err(T5LoadError::UnknownTarget {
            target: target.into(),
        });
    };
    let (index, suffix) = remainder
        .split_once('.')
        .ok_or_else(|| T5LoadError::InvalidBlock {
            target: target.into(),
            reason: "expected `<stack>.block.<index>.<parameter>`".into(),
        })?;
    if index.is_empty() || suffix.is_empty() {
        return Err(T5LoadError::InvalidBlock {
            target: target.into(),
            reason: "block index and parameter suffix must be non-empty".into(),
        });
    }
    let index = index
        .parse::<usize>()
        .map_err(|error| T5LoadError::InvalidBlock {
            target: target.into(),
            reason: format!("`{index}` is not a decimal usize: {error}"),
        })?;
    Ok((stack, index, suffix))
}

/// Dispatches an already resolved block target to its exact T5 sublayer.
fn install_block_tensor<B: Backend>(
    block: &mut T5Block<B>,
    stack: Stack,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), T5LoadError> {
    if let Some(parameter) = suffix.strip_prefix("layer.0.SelfAttention.") {
        return install_attention_tensor(
            &mut block.self_attention.self_attention,
            target,
            parameter,
            data,
        );
    }
    if suffix == "layer.0.layer_norm.weight" {
        return install_norm(target, &mut block.self_attention.layer_norm, data);
    }

    if stack == Stack::Decoder {
        if let Some(parameter) = suffix.strip_prefix("layer.1.EncDecAttention.") {
            let cross =
                block
                    .cross_attention
                    .as_mut()
                    .ok_or_else(|| T5LoadError::MissingComponent {
                        target: target.into(),
                        component: "decoder encoder-decoder attention",
                    })?;
            return install_attention_tensor(&mut cross.enc_dec_attention, target, parameter, data);
        }
        if suffix == "layer.1.layer_norm.weight" {
            let cross =
                block
                    .cross_attention
                    .as_mut()
                    .ok_or_else(|| T5LoadError::MissingComponent {
                        target: target.into(),
                        component: "decoder encoder-decoder attention",
                    })?;
            return install_norm(target, &mut cross.layer_norm, data);
        }
    }

    let feed_forward_layer = feed_forward_sublayer(stack == Stack::Decoder);
    let feed_forward_prefix = format!("layer.{feed_forward_layer}.DenseReluDense.");
    if let Some(parameter) = suffix.strip_prefix(&feed_forward_prefix) {
        return install_feed_forward_tensor(&mut block.feed_forward, target, parameter, data);
    }
    if suffix == format!("layer.{feed_forward_layer}.layer_norm.weight") {
        return install_norm(target, &mut block.feed_forward.layer_norm, data);
    }

    Err(T5LoadError::UnknownTarget {
        target: target.into(),
    })
}

/// Installs one bias-free T5 attention matrix or its learned relative-bias table.
fn install_attention_tensor<B: Backend>(
    attention: &mut T5Attention<B>,
    target: &str,
    parameter: &str,
    data: TensorData,
) -> Result<(), T5LoadError> {
    match parameter {
        "q.weight" => install_linear_weight(target, &mut attention.attention.q_proj, data),
        "k.weight" => install_linear_weight(target, &mut attention.attention.k_proj, data),
        "v.weight" => install_linear_weight(target, &mut attention.attention.v_proj, data),
        "o.weight" => install_linear_weight(target, &mut attention.attention.o_proj, data),
        "relative_attention_bias.weight" => {
            let bias = attention.relative_attention_bias.as_mut().ok_or_else(|| {
                T5LoadError::MissingComponent {
                    target: target.into(),
                    component: "block-zero relative-position bias table",
                }
            })?;
            install(target, &mut bias.weight, data)
        }
        _ => Err(T5LoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs an official `DenseReluDense` tensor into the selected FFN topology.
fn install_feed_forward_tensor<B: Backend>(
    feed_forward: &mut T5LayerFeedForward<B>,
    target: &str,
    parameter: &str,
    data: TensorData,
) -> Result<(), T5LoadError> {
    match parameter {
        "wi.weight" => {
            let dense =
                feed_forward
                    .dense
                    .as_mut()
                    .ok_or_else(|| T5LoadError::MissingComponent {
                        target: target.into(),
                        component: "dense `wi` feed-forward projection",
                    })?;
            install_linear_weight(target, &mut dense.up_proj, data)
        }
        "wi_0.weight" => {
            let gated =
                feed_forward
                    .gated
                    .as_mut()
                    .ok_or_else(|| T5LoadError::MissingComponent {
                        target: target.into(),
                        component: "gated `wi_0` feed-forward projection",
                    })?;
            install_linear_weight(target, &mut gated.gate_proj, data)
        }
        "wi_1.weight" => {
            let gated =
                feed_forward
                    .gated
                    .as_mut()
                    .ok_or_else(|| T5LoadError::MissingComponent {
                        target: target.into(),
                        component: "gated `wi_1` feed-forward projection",
                    })?;
            install_linear_weight(target, &mut gated.up_proj, data)
        }
        "wo.weight" => match (
            feed_forward.dense.as_mut(),
            feed_forward.gated.as_mut(),
            feed_forward.projection.is_gated(),
        ) {
            (Some(dense), None, false) => install_linear_weight(target, &mut dense.down_proj, data),
            (None, Some(gated), true) => install_linear_weight(target, &mut gated.down_proj, data),
            _ => Err(T5LoadError::MissingComponent {
                target: target.into(),
                component: "feed-forward output projection matching the configured topology",
            }),
        },
        _ => Err(T5LoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs the checkpoint-facing weight of one Burn linear layer.
fn install_linear_weight<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), T5LoadError> {
    install(target, &mut linear.weight, data)
}

/// Installs one T5 scale-only RMSNorm weight unchanged.
fn install_norm<B: Backend>(
    target: &str,
    norm: &mut RmsNorm<B>,
    data: TensorData,
) -> Result<(), T5LoadError> {
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
) -> Result<(), T5LoadError> {
    install_packed_parameter(parameter, data, PackedLayoutPolicy::DequantizeWhenInexact)
        .map(|_| ())
        .map_err(|source| T5LoadError::Parameter {
            target: target.into(),
            source,
        })
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
