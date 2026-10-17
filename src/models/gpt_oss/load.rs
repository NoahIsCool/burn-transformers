//! Streamed parameter installation for gpt-oss checkpoints.
//!
//! The generic checkpoint runtime discovers every shard header and validates
//! the complete [`GptOssCheckpointContract`](super::GptOssCheckpointContract)
//! before this sink sees a payload, decoding MXFP4 expert tensors on the way
//! (see `crate::checkpoint`'s `mxfp4` module). Afterwards the sink holds one
//! tensor at a time and routes each target name to the statically ranked
//! Burn parameter that owns it.
//!
//! Stacked expert tensors are split here. Expert `e` of a block receives
//!
//! ```text
//! gate_proj.weight = gate_up_proj[e, :, 0::2]   [d, f]   up_proj.weight = gate_up_proj[e, :, 1::2]
//! gate_proj.bias   = gate_up_proj_bias[e, 0::2] [f]      up_proj.bias   = gate_up_proj_bias[e, 1::2]
//! down_proj.weight = down_proj[e]               [f, d]   down_proj.bias = down_proj_bias[e]
//! ```
//!
//! in Burn's row layout (`[in, out]`, the reference's own orientation), so a
//! dense expert installs by one contiguous copy (`expert_slab`) and a gate/up
//! split by one strided pass (`deinterleave`), both Verus-verified to copy
//! exactly the reference's bytes from a payload whose length is checked first.

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
};

use super::{
    GptOssCheckpointOptions, GptOssDecoderLayer, GptOssForCausalLm,
    plan::{GluHalf, deinterleave, expert_slab, stacked_layout},
};

/// Failure while installing one validated tensor into the graph.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GptOssLoadError {
    /// The target names no parameter of the gpt-oss graph.
    #[error("unknown gpt-oss parameter target `{target}`")]
    UnknownTarget {
        /// Unrecognized target name.
        target: String,
    },
    /// A layer index was malformed or out of range.
    #[error("invalid gpt-oss target `{target}`: {reason}")]
    InvalidIndex {
        /// Malformed target name.
        target: String,
        /// Index diagnostic.
        reason: String,
    },
    /// The configured topology lacks the component a target addresses.
    #[error("gpt-oss target `{target}` requires missing component `{component}`")]
    MissingComponent {
        /// Parameter being installed.
        target: String,
        /// Component the configuration did not allocate.
        component: &'static str,
    },
    /// A stacked expert tensor does not match the block's expert geometry.
    #[error("gpt-oss expert tensor `{target}` has shape {actual:?}, expected {expected:?}")]
    ExpertShape {
        /// Stacked expert target.
        target: String,
        /// Shape implied by the block's experts.
        expected: Vec<usize>,
        /// Shape of the delivered tensor.
        actual: Vec<usize>,
    },
    /// Tensor metadata disagreed with the statically ranked Burn parameter.
    #[error("cannot install gpt-oss parameter `{target}`: {source}")]
    Parameter {
        /// Target name.
        target: String,
        /// Shape, rank, or element-kind failure.
        #[source]
        source: ParameterInstallError,
    },
    /// A tied-parameter alias violated the embedding/head contract.
    #[error("invalid gpt-oss alias `{target}` -> `{source_target}`: {reason}")]
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
pub enum GptOssCheckpointLoadError {
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
        Box<ExecutionFailure<GptOssLoadError>>,
    ),
}

impl GptOssCheckpointLoadError {
    /// Returns partial execution diagnostics when the sink was reached.
    #[must_use]
    pub fn execution_failure(&self) -> Option<&ExecutionFailure<GptOssLoadError>> {
        match self {
            Self::Execution(failure) => Some(failure),
            Self::Config(_) | Self::Checkpoint(_) => None,
        }
    }
}

impl<B: Backend> GptOssForCausalLm<B> {
    /// Strictly discovers, validates, and streams a local SafeTensors checkpoint.
    ///
    /// No network access is performed. A missing, unexpected, misshapen, or
    /// disallowed-dtype tensor fails before the first payload read. MXFP4
    /// expert tensors are decoded exactly into the plan's dtype. After an
    /// execution failure the partially initialized model must be discarded.
    ///
    /// # Errors
    ///
    /// Returns a [`GptOssCheckpointLoadError`] naming the failing stage.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, GptOssCheckpointLoadError> {
        self.load_checkpoint_with(root, options, GptOssCheckpointOptions::default())
    }

    /// Loads a local checkpoint under explicit family checkpoint options.
    ///
    /// [`GptOssCheckpointOptions`] has no fields yet, so this equals
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
        family_options: GptOssCheckpointOptions,
    ) -> Result<ExecutionOutcome, GptOssCheckpointLoadError> {
        let manifest = CheckpointManifest::discover(root, options)?;
        let contract = self.config().checkpoint_contract(family_options)?;
        let plan = contract.build_load_plan(&manifest, options)?;
        Ok(execute_local_load_plan(&manifest, &plan, self)?)
    }
}

impl<B: Backend> TensorSink for GptOssForCausalLm<B> {
    /// Contextual installation or alias failure.
    type Error = GptOssLoadError;

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
                    GptOssLoadError::InvalidIndex {
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
        let reject = |reason: &str| GptOssLoadError::InvalidAlias {
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
fn parse_layer_target(target: &str) -> Result<(usize, &str), GptOssLoadError> {
    let invalid = |reason: String| GptOssLoadError::InvalidIndex {
        target: target.into(),
        reason,
    };
    let remainder =
        target
            .strip_prefix("model.layers.")
            .ok_or_else(|| GptOssLoadError::UnknownTarget {
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
    layer: &mut GptOssDecoderLayer<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), GptOssLoadError> {
    let attention = &mut layer.self_attn;
    match suffix {
        "input_layernorm.weight" => install(target, &mut layer.input_layernorm.weight, data),
        "post_attention_layernorm.weight" => {
            install(target, &mut layer.post_attention_layernorm.weight, data)
        }
        "self_attn.q_proj.weight" => install(target, &mut attention.q_proj.weight, data),
        "self_attn.k_proj.weight" => install(target, &mut attention.k_proj.weight, data),
        "self_attn.v_proj.weight" => install(target, &mut attention.v_proj.weight, data),
        "self_attn.o_proj.weight" => install(target, &mut attention.o_proj.weight, data),
        "self_attn.q_proj.bias" => install_bias(target, &mut attention.q_proj, data),
        "self_attn.k_proj.bias" => install_bias(target, &mut attention.k_proj, data),
        "self_attn.v_proj.bias" => install_bias(target, &mut attention.v_proj, data),
        "self_attn.o_proj.bias" => install_bias(target, &mut attention.o_proj, data),
        "self_attn.sinks" => {
            let sinks = attention
                .sinks
                .as_mut()
                .ok_or_else(|| missing(target, "attention sinks"))?;
            install(target, sinks, data)
        }
        "mlp.router.weight" => install(target, &mut layer.mlp.router.weight, data),
        "mlp.router.bias" => install_bias(target, &mut layer.mlp.router, data),
        "mlp.experts.gate_up_proj" => install_gate_up(&mut layer.mlp.experts, target, data, false),
        "mlp.experts.gate_up_proj_bias" => {
            install_gate_up(&mut layer.mlp.experts, target, data, true)
        }
        "mlp.experts.down_proj" => install_down(&mut layer.mlp.experts, target, data, false),
        "mlp.experts.down_proj_bias" => install_down(&mut layer.mlp.experts, target, data, true),
        _ => Err(GptOssLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Reads the `[E, rows, columns]` (or bias `[E, columns]`) extents of a stacked expert tensor.
///
/// `rows` is 1 for biases. Rejects a tensor whose leading axis is not the
/// block's expert count or whose trailing extents differ from `expected`.
fn stacked_extents(
    target: &str,
    data: &TensorData,
    experts: usize,
    expected: &[usize],
) -> Result<(usize, usize), GptOssLoadError> {
    let shape = data.shape.to_vec();
    let mut full = vec![experts];
    full.extend_from_slice(expected);
    if shape != full {
        return Err(GptOssLoadError::ExpertShape {
            target: target.into(),
            expected: full,
            actual: shape,
        });
    }
    Ok(match expected {
        [columns] => (1, *columns),
        [rows, columns] => (*rows, *columns),
        _ => (0, 0),
    })
}

/// Checks that a stacked `[E, rows, columns]` payload holds exactly its scalars' bytes.
///
/// Returns one expert's byte span. The verified splitters read expert slabs
/// and interleaved columns inside a payload of exactly
/// `E · rows · columns · element` bytes, which every validated tensor has;
/// anything else is reported as a payload error rather than read.
fn stacked_payload(
    target: &str,
    bytes: &[u8],
    experts: usize,
    rows: usize,
    columns: usize,
    element: usize,
) -> Result<usize, GptOssLoadError> {
    let parameter = |source| GptOssLoadError::Parameter {
        target: target.into(),
        source,
    };
    let (span, total) = stacked_layout(experts, rows, columns, element)
        .ok_or_else(|| parameter(ParameterInstallError::PayloadSizeOverflow))?;
    if bytes.len() != total {
        return Err(parameter(ParameterInstallError::PayloadLength {
            expected: total,
            actual: bytes.len(),
        }));
    }
    Ok(span)
}

/// Splits a stacked, column-interleaved `gate_up_proj` (or its bias) into every expert.
///
/// The weight is `[E, d, 2f]` and the bias `[E, 2f]`; even columns go to
/// `gate_proj`, odd columns to `up_proj` (see the module documentation).
fn install_gate_up<B: Backend>(
    experts: &mut [GatedFeedForward<B>],
    target: &str,
    data: TensorData,
    bias: bool,
) -> Result<(), GptOssLoadError> {
    let Some(first) = experts.first() else {
        return Err(missing(target, "routed experts"));
    };
    let [hidden, width] = first.gate_proj.weight.lazy_shape().dims();
    let expected: Vec<usize> = if bias {
        vec![2 * width]
    } else {
        vec![hidden, 2 * width]
    };
    let count = experts.len();
    let (rows, columns) = stacked_extents(target, &data, count, &expected)?;
    let width = columns / 2;
    let element = data.dtype.size();
    let bytes = data.as_bytes();
    stacked_payload(target, bytes, count, rows, 2 * width, element)?;
    for (index, expert) in experts.iter_mut().enumerate() {
        let [gate, up] = [GluHalf::Gate, GluHalf::Up]
            .map(|half| deinterleave(bytes, count, rows, width, element, index, half));
        let shape: Vec<usize> = if bias { vec![width] } else { vec![rows, width] };
        let gate = TensorData::from_bytes_vec(gate, shape.clone(), data.dtype);
        let up = TensorData::from_bytes_vec(up, shape, data.dtype);
        if bias {
            install_bias(target, &mut expert.gate_proj, gate)?;
            install_bias(target, &mut expert.up_proj, up)?;
        } else {
            install(target, &mut expert.gate_proj.weight, gate)?;
            install(target, &mut expert.up_proj.weight, up)?;
        }
    }
    Ok(())
}

/// Splits a stacked `down_proj` `[E, f, d]` (or its bias `[E, d]`) into every expert.
fn install_down<B: Backend>(
    experts: &mut [GatedFeedForward<B>],
    target: &str,
    data: TensorData,
    bias: bool,
) -> Result<(), GptOssLoadError> {
    let Some(first) = experts.first() else {
        return Err(missing(target, "routed experts"));
    };
    let [width, hidden] = first.down_proj.weight.lazy_shape().dims();
    let expected: Vec<usize> = if bias {
        vec![hidden]
    } else {
        vec![width, hidden]
    };
    let count = experts.len();
    let (rows, columns) = stacked_extents(target, &data, count, &expected)?;
    let element = data.dtype.size();
    let bytes = data.as_bytes();
    let span = stacked_payload(target, bytes, count, rows, columns, element)?;
    for (index, expert) in experts.iter_mut().enumerate() {
        let slab = expert_slab(bytes, count, span, index);
        let tensor = TensorData::from_bytes_vec(slab, expected.clone(), data.dtype);
        if bias {
            install_bias(target, &mut expert.down_proj, tensor)?;
        } else {
            install(target, &mut expert.down_proj.weight, tensor)?;
        }
    }
    Ok(())
}

/// Installs an enabled linear bias.
fn install_bias<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), GptOssLoadError> {
    let bias = linear
        .bias
        .as_mut()
        .ok_or_else(|| missing(target, "projection bias"))?;
    install(target, bias, data)
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
) -> Result<(), GptOssLoadError> {
    install_packed_parameter(parameter, data, PackedLayoutPolicy::DequantizeWhenInexact)
        .map(|_| ())
        .map_err(|source| GptOssLoadError::Parameter {
            target: target.into(),
            source,
        })
}

/// Reports a target whose component the configuration did not allocate.
fn missing(target: &str, component: &'static str) -> GptOssLoadError {
    GptOssLoadError::MissingComponent {
        target: target.into(),
        component,
    }
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
