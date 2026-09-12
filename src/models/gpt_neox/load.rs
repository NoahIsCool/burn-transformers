//! Strict local GPT-NeoX SafeTensors loading.
//!
//! GPT-NeoX stores Q/K/V rows in `[head, component, head_channel, input]`
//! order. The architecture sink expands each validated packed tensor into the
//! three ordinary attention projections without changing its floating dtype.
//! The plan and load report name those component transformations explicitly.
//!
//! Unlike GPT-2 Conv1D storage, source linear weights use `[output, input]`.
//! Burn Col-layout parameters retain that checkpoint/record convention while their
//! runtime tensors use `[input, output]`. Affine norm vectors map directly; input
//! and vocabulary embeddings have independent owners. Packed component staging is
//! included in the host-memory plan, but backend transfer storage is not.
//!
//! Only exact configured legacy mask and rotary buffers are allowlisted. Strict
//! metadata validation precedes payload writes. A QKV source installs atomically as
//! a group; unrelated prior writes can remain after a later streaming failure.

use std::path::Path;

use burn::{
    module::Param,
    nn::{LayerNorm, Linear},
    tensor::{Tensor, TensorData, backend::Backend},
};
use thiserror::Error;

use crate::checkpoint::{
    CheckpointError, CheckpointManifest, DTypeConversion, ExecutionFailure, ExecutionOutcome,
    ExpectedTensor, KeyMatcher, LoadOptions, LoadPlan, LoadReport, MappingPlan,
    ParameterInstallError, TensorSink, TransformRecord, build_load_plan, execute_local_load_plan,
    install_float_parameter,
};

use super::{GptNeoXConfig, GptNeoXLayer, GptNeoXModel};

// Metadata contracts and deterministic shape accounting.

/// Expected source shapes, parameter mappings, and derived-buffer exceptions.
///
/// Packed `attention.query_key_value` targets are architecture-owned staging
/// targets. Their three resulting parameters appear in the plan's report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GptNeoXCheckpointContract {
    /// Logical parameters and packed staging targets with exact checkpoint shapes.
    expected: Vec<ExpectedTensor>,
    /// Explicit source-to-runtime target mappings before packed-QKV expansion.
    mappings: MappingPlan,
    /// Exact optional causal-mask and rotary-buffer names reconstructed by the graph.
    buffers: Vec<KeyMatcher>,
    /// Query-head count used to attribute packed-QKV component transformations.
    heads: usize,
}

impl GptNeoXCheckpointContract {
    /// Returns parameters and packed staging targets in deterministic order.
    #[must_use]
    pub fn expected_tensors(&self) -> &[ExpectedTensor] {
        &self.expected
    }

    /// Returns explicit source-to-runtime mappings.
    #[must_use]
    pub const fn mappings(&self) -> &MappingPlan {
        &self.mappings
    }

    /// Returns exact optional buffer names reconstructed from the configuration.
    ///
    /// Older checkpoints persist causal masks, masked-bias scalars, and rotary
    /// inverse frequencies. These are not learned parameters. The graph derives
    /// its mask and rotary frequencies from its explicit configuration; ignored
    /// buffers remain visible in `LoadReport::allowlisted_ignored`.
    #[must_use]
    pub fn allow_unexpected(&self) -> &[KeyMatcher] {
        &self.buffers
    }

    /// Validates every required tensor before any parameter payload is loaded.
    pub fn build_load_plan(
        &self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
    ) -> crate::checkpoint::Result<LoadPlan> {
        let mut options = options.clone();
        for buffer in &self.buffers {
            if !options.allow_unexpected.contains(buffer) {
                options.allow_unexpected.push(buffer.clone());
            }
        }
        let mut plan = build_load_plan(manifest, &self.expected, &self.mappings, &options)?;
        let mut largest_component = 0;
        for tensor in &plan.tensors {
            if !tensor.dtype.is_float() {
                return Err(CheckpointError::InvalidMapping {
                    message: format!(
                        "GPT-NeoX parameter `{}` requires floating-point storage, received {}",
                        tensor.target, tensor.dtype
                    ),
                });
            }
            let Some(targets) = component_targets(&tensor.target) else {
                continue;
            };
            plan.report
                .transformed
                .retain(|record| record.target != tensor.target);
            for (component, target) in targets.into_iter().enumerate() {
                plan.report.transformed.push(TransformRecord {
                    target,
                    sources: tensor
                        .sources
                        .iter()
                        .map(|source| source.name.clone())
                        .collect(),
                    operation: format!(
                        "gpt_neox_qkv_component(index={component},heads={})",
                        self.heads
                    ),
                });
            }
            let elements = tensor.shape.iter().try_fold(1_u64, |count, &size| {
                count
                    .checked_mul(size)
                    .ok_or(CheckpointError::ArithmeticOverflow {
                        context: "computing GPT-NeoX packed projection elements",
                    })
            })?;
            largest_component = largest_component.max(tensor.dtype.storage_bytes(elements)? / 3);
        }
        // The sink retains the packed TensorData while constructing one native
        // component buffer. Generic execution counters exclude sink allocations.
        plan.memory.peak_host_staging_bytes = plan
            .memory
            .peak_host_staging_bytes
            .checked_add(largest_component)
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "accounting for GPT-NeoX component staging",
            })?;
        plan.report.memory = Some(plan.memory);
        expand_report_targets(&mut plan.report);
        Ok(plan)
    }
}

impl GptNeoXConfig {
    /// Derives the complete GPT-NeoX checkpoint contract from validated dimensions.
    pub fn checkpoint_contract(
        &self,
    ) -> Result<GptNeoXCheckpointContract, GptNeoXCheckpointLoadError> {
        self.validate()?;
        if self.tie_word_embeddings {
            return Err(crate::Error::configuration(
                "gpt_neox.tie_word_embeddings",
                "checkpoint loading currently requires an independent output embedding",
            )
            .into());
        }
        let dimension = |value: usize| {
            u64::try_from(value).map_err(|_| CheckpointError::ArithmeticOverflow {
                context: "converting GPT-NeoX checkpoint dimensions",
            })
        };
        let hidden = dimension(self.hidden_size)?;
        let vocab = dimension(self.vocab_size)?;
        let intermediate = dimension(self.intermediate_size)?;
        let packed = hidden
            .checked_mul(3)
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "computing GPT-NeoX packed QKV width",
            })?;
        let mut contract = GptNeoXCheckpointContract {
            expected: Vec::new(),
            mappings: MappingPlan::new(),
            buffers: Vec::new(),
            heads: self.num_attention_heads,
        };
        contract.add(
            "gpt_neox.embed_in.weight",
            "embed_in.weight",
            vec![vocab, hidden],
        );
        contract.add("embed_out.weight", "embed_out.weight", vec![vocab, hidden]);
        for (source, target) in [("weight", "gamma"), ("bias", "beta")] {
            contract.add(
                &format!("gpt_neox.final_layer_norm.{source}"),
                &format!("final_layer_norm.{target}"),
                vec![hidden],
            );
        }
        for layer in 0..self.num_hidden_layers {
            let source = format!("gpt_neox.layers.{layer}");
            let target = format!("layers.{layer}");
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                for (source_suffix, target_suffix) in [("weight", "gamma"), ("bias", "beta")] {
                    contract.add(
                        &format!("{source}.{norm}.{source_suffix}"),
                        &format!("{target}.{norm}.{target_suffix}"),
                        vec![hidden],
                    );
                }
            }
            for (source_name, target_name, output, input) in [
                (
                    "attention.query_key_value",
                    "attention.query_key_value",
                    packed,
                    hidden,
                ),
                ("attention.dense", "attention.o_proj", hidden, hidden),
                ("mlp.dense_h_to_4h", "mlp.up_proj", intermediate, hidden),
                ("mlp.dense_4h_to_h", "mlp.down_proj", hidden, intermediate),
            ] {
                contract.add(
                    &format!("{source}.{source_name}.weight"),
                    &format!("{target}.{target_name}.weight"),
                    vec![output, input],
                );
                contract.add(
                    &format!("{source}.{source_name}.bias"),
                    &format!("{target}.{target_name}.bias"),
                    vec![output],
                );
            }
            for buffer in ["bias", "masked_bias", "rotary_emb.inv_freq"] {
                contract
                    .buffers
                    .push(KeyMatcher::exact(format!("{source}.attention.{buffer}")));
            }
        }
        Ok(contract)
    }

    /// Builds an audited load plan without reading parameter payloads.
    pub fn build_load_plan(
        &self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
    ) -> Result<LoadPlan, GptNeoXCheckpointLoadError> {
        Ok(self
            .checkpoint_contract()?
            .build_load_plan(manifest, options)?)
    }
}

impl GptNeoXCheckpointContract {
    /// Registers a source-to-runtime rename and its exact checkpoint-facing parameter shape.
    fn add(&mut self, source: &str, target: &str, shape: Vec<u64>) {
        self.expected.push(ExpectedTensor::new(target, shape));
        self.mappings.map(source, target);
    }
}

/// Contextual failure while installing one GPT-NeoX parameter or packed projection.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GptNeoXLoadError {
    /// A target is not an executable parameter in this architecture.
    #[error("unknown GPT-NeoX parameter target `{target}`")]
    UnknownTarget {
        /// Rejected target.
        target: String,
    },
    /// A layer index is malformed or outside the model.
    #[error("invalid GPT-NeoX target `{target}`: {reason}")]
    InvalidIndex {
        /// Rejected target.
        target: String,
        /// Index diagnostic.
        reason: String,
    },
    /// A public field lacks a required bias.
    #[error("GPT-NeoX target `{target}` requires a learned bias")]
    MissingBias {
        /// Rejected target.
        target: String,
    },
    /// Public attention dimensions cannot describe the packed projection.
    #[error("invalid GPT-NeoX projection `{target}`: {reason}")]
    InvalidProjection {
        /// Rejected packed target.
        target: String,
        /// Dimension diagnostic.
        reason: String,
    },
    /// Incoming shape, dtype, or byte count disagrees with a parameter.
    #[error("cannot install GPT-NeoX parameter `{target}`: {source}")]
    Parameter {
        /// Parameter or staging target.
        target: String,
        /// Underlying storage diagnostic.
        #[source]
        source: ParameterInstallError,
    },
    /// This graph has independent input and output embeddings.
    #[error("GPT-NeoX does not support checkpoint alias `{target}` -> `{source_target}`")]
    UnsupportedAlias {
        /// Requested alias.
        target: String,
        /// Requested storage owner.
        source_target: String,
    },
}

/// Failure from configuration, metadata validation, or local checkpoint execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GptNeoXCheckpointLoadError {
    /// Invalid or unsupported architecture configuration.
    #[error(transparent)]
    Configuration(
        /// Invalid architecture configuration or dimension contract.
        #[from]
        crate::Error,
    ),
    /// Discovery or strict metadata validation failed before execution.
    #[error(transparent)]
    Checkpoint(
        /// Local discovery or strict metadata validation failure.
        #[from]
        CheckpointError,
    ),
    /// Execution failed; discard the partially initialized model.
    #[error(transparent)]
    Execution(
        /// Partial execution report and contextual installation failure.
        #[from]
        Box<ExecutionFailure<GptNeoXLoadError>>,
    ),
}

impl GptNeoXCheckpointLoadError {
    /// Returns the exact partial report if loading reached execution.
    #[must_use]
    pub fn execution_failure(&self) -> Option<&ExecutionFailure<GptNeoXLoadError>> {
        match self {
            Self::Execution(failure) => Some(failure),
            Self::Configuration(_) | Self::Checkpoint(_) => None,
        }
    }
}

impl<B: Backend> GptNeoXModel<B> {
    /// Strictly imports a local SafeTensors checkpoint; performs no network access.
    ///
    /// All metadata is validated before parameter writes. Q/K/V installation is
    /// atomic for each packed source: a failed component leaves all three original
    /// projections intact. Other successful writes remain in the partial report.
    /// Generic execution counters count packed sink calls; the report names the
    /// separate installed parameters, and its memory bound includes component
    /// repacking. Backend transfers are outside the host-staging estimate.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, GptNeoXCheckpointLoadError> {
        let manifest = CheckpointManifest::discover(root, options)?;
        let plan = self.config().build_load_plan(&manifest, options)?;
        match execute_local_load_plan(&manifest, &plan, self) {
            Ok(mut outcome) => {
                expand_report_targets(&mut outcome.report);
                Ok(outcome)
            }
            Err(mut failure) => {
                expand_report_targets(&mut failure.report);
                Err(failure.into())
            }
        }
    }
}

// Exact target routing and atomic packed-projection installation.

impl<B: Backend> TensorSink for GptNeoXModel<B> {
    /// Contextual target or packed-component parameter installation failure.
    type Error = GptNeoXLoadError;

    /// Dispatches exact parameter targets, checking layer bounds before ordinary or packed installation.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        match target {
            "embed_in.weight" => install(target, &mut self.embed_in.weight, data),
            "embed_out.weight" => install(target, &mut self.embed_out.weight, data),
            "final_layer_norm.gamma" => install(target, &mut self.final_layer_norm.gamma, data),
            "final_layer_norm.beta" => install_norm_bias(target, &mut self.final_layer_norm, data),
            _ => {
                let heads = self.config().num_attention_heads;
                let hidden_size = self.config().hidden_size;
                let (index, suffix) = parse_layer_target(target)?;
                let count = self.layers.len();
                let layer =
                    self.layers
                        .get_mut(index)
                        .ok_or_else(|| GptNeoXLoadError::InvalidIndex {
                            target: target.into(),
                            reason: format!("layer {index} is outside 0..{count}"),
                        })?;
                install_layer(layer, target, suffix, data, heads, hidden_size)
            }
        }
    }

    /// Rejects checkpoint aliases because this graph requires independent input and output embeddings.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        Err(GptNeoXLoadError::UnsupportedAlias {
            target: target.into(),
            source_target: source_target.into(),
        })
    }
}

/// Parses only `layers.<decimal index>.<nonempty suffix>` with contextual malformed-index errors.
fn parse_layer_target(target: &str) -> Result<(usize, &str), GptNeoXLoadError> {
    let remainder = target
        .strip_prefix("layers.")
        .ok_or_else(|| unknown(target))?;
    let (index, suffix) =
        remainder
            .split_once('.')
            .ok_or_else(|| GptNeoXLoadError::InvalidIndex {
                target: target.into(),
                reason: "expected `layers.<decimal index>.<parameter>`".into(),
            })?;
    if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) || suffix.is_empty() {
        return Err(GptNeoXLoadError::InvalidIndex {
            target: target.into(),
            reason: "expected a decimal index and nonempty parameter suffix".into(),
        });
    }
    let index = index
        .parse()
        .map_err(|error| GptNeoXLoadError::InvalidIndex {
            target: target.into(),
            reason: format!("invalid layer index: {error}"),
        })?;
    Ok((index, suffix))
}

/// Routes a validated block suffix to norms, dense projections, or head-major packed QKV.
fn install_layer<B: Backend>(
    layer: &mut GptNeoXLayer<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
    heads: usize,
    hidden_size: usize,
) -> Result<(), GptNeoXLoadError> {
    match suffix {
        "input_layernorm.gamma" => install(target, &mut layer.input_layernorm.gamma, data),
        "input_layernorm.beta" => install_norm_bias(target, &mut layer.input_layernorm, data),
        "post_attention_layernorm.gamma" => {
            install(target, &mut layer.post_attention_layernorm.gamma, data)
        }
        "post_attention_layernorm.beta" => {
            install_norm_bias(target, &mut layer.post_attention_layernorm, data)
        }
        "attention.query_key_value.weight" => {
            install_packed(layer, target, data, false, heads, hidden_size)
        }
        "attention.query_key_value.bias" => {
            install_packed(layer, target, data, true, heads, hidden_size)
        }
        "attention.o_proj.weight" => install(target, &mut layer.attention.o_proj.weight, data),
        "attention.o_proj.bias" => install_bias(target, &mut layer.attention.o_proj, data),
        "mlp.up_proj.weight" => install(target, &mut layer.mlp.up_proj.weight, data),
        "mlp.up_proj.bias" => install_bias(target, &mut layer.mlp.up_proj, data),
        "mlp.down_proj.weight" => install(target, &mut layer.mlp.down_proj.weight, data),
        "mlp.down_proj.bias" => install_bias(target, &mut layer.mlp.down_proj, data),
        _ => Err(unknown(target)),
    }
}

/// Unpacks head-major QKV rows without floating conversion and commits all three projections together.
///
/// Source weights are `[heads, 3, head_channel, input]`; bias omits the input axis.
/// Selecting component `c` gathers rows `(head * 3 + c) * head_dim..+head_dim`.
/// Cloned targets prevent a failed component from partially replacing the live Q/K/V group.
fn install_packed<B: Backend>(
    layer: &mut GptNeoXLayer<B>,
    target: &str,
    data: TensorData,
    bias: bool,
    expected_heads: usize,
    expected_hidden: usize,
) -> Result<(), GptNeoXLoadError> {
    let heads = layer.attention.query_heads;
    let head_dim = layer.attention.head_dim;
    let hidden = layer.attention.hidden_size;
    if heads == 0
        || head_dim == 0
        || hidden == 0
        || heads != expected_heads
        || hidden != expected_hidden
        || heads.checked_mul(head_dim) != Some(expected_hidden)
        || layer.attention.key_value_heads != heads
    {
        return Err(GptNeoXLoadError::InvalidProjection {
            target: target.into(),
            reason: "head counts and widths must match the checkpoint configuration before loading"
                .into(),
        });
    }
    let overflow = || parameter_error(target, ParameterInstallError::PayloadSizeOverflow);
    let rows = heads
        .checked_mul(head_dim)
        .and_then(|width| width.checked_mul(3))
        .ok_or_else(overflow)?;
    let shape = if bias { vec![rows] } else { vec![rows, hidden] };
    if data.shape.as_slice() != shape {
        return Err(parameter_error(
            target,
            ParameterInstallError::Shape {
                expected: shape,
                actual: data.shape.to_vec(),
            },
        ));
    }
    if !data.dtype.is_float() {
        return Err(parameter_error(
            target,
            ParameterInstallError::NonFloat {
                dtype: format!("{:?}", data.dtype),
            },
        ));
    }
    let row_bytes = (if bias { 1 } else { hidden })
        .checked_mul(data.dtype.size())
        .ok_or_else(overflow)?;
    let expected_bytes = rows.checked_mul(row_bytes).ok_or_else(overflow)?;
    if data.bytes.len() != expected_bytes {
        return Err(parameter_error(
            target,
            ParameterInstallError::PayloadLength {
                expected: expected_bytes,
                actual: data.bytes.len(),
            },
        ));
    }
    let mut projections = [
        layer.attention.q_proj.clone(),
        layer.attention.k_proj.clone(),
        layer.attention.v_proj.clone(),
    ];
    let targets = component_targets(target).ok_or_else(|| unknown(target))?;
    for (component, projection) in projections.iter_mut().enumerate() {
        let mut bytes = Vec::with_capacity(expected_bytes / 3);
        for head in 0..heads {
            let start = (head * 3 + component) * head_dim * row_bytes;
            let end = start + head_dim * row_bytes;
            bytes.extend_from_slice(&data.bytes[start..end]);
        }
        let shape = if bias {
            vec![rows / 3]
        } else {
            vec![rows / 3, hidden]
        };
        let component_data = TensorData::from_bytes_vec(bytes, shape, data.dtype);
        if bias {
            install_bias(&targets[component], projection, component_data)?;
        } else {
            install(&targets[component], &mut projection.weight, component_data)?;
        }
    }
    let [query, key, value] = projections;
    layer.attention.q_proj = query;
    layer.attention.k_proj = key;
    layer.attention.v_proj = value;
    Ok(())
}

/// Requires an affine LayerNorm beta before installing its channelwise bias payload.
fn install_norm_bias<B: Backend>(
    target: &str,
    norm: &mut LayerNorm<B>,
    data: TensorData,
) -> Result<(), GptNeoXLoadError> {
    let bias = norm
        .beta
        .as_mut()
        .ok_or_else(|| GptNeoXLoadError::MissingBias {
            target: target.into(),
        })?;
    install(target, bias, data)
}

/// Requires an existing projection bias before installing its rank-one payload.
fn install_bias<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), GptNeoXLoadError> {
    let bias = linear
        .bias
        .as_mut()
        .ok_or_else(|| GptNeoXLoadError::MissingBias {
            target: target.into(),
        })?;
    install(target, bias, data)
}

/// Installs floating data with the parameter's save/load layout and adds the target to errors.
fn install<B: Backend, const D: usize>(
    target: &str,
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> Result<(), GptNeoXLoadError> {
    install_float_parameter(parameter, data).map_err(|source| parameter_error(target, source))
}

/// Attaches an architecture target to the shared parameter-installation diagnostic.
fn parameter_error(target: &str, source: ParameterInstallError) -> GptNeoXLoadError {
    GptNeoXLoadError::Parameter {
        target: target.into(),
        source,
    }
}

/// Constructs the contextual error for a target outside this architecture's exact inventory.
fn unknown(target: &str) -> GptNeoXLoadError {
    GptNeoXLoadError::UnknownTarget {
        target: target.into(),
    }
}

// Logical parameter reporting for packed staging targets.

/// Expands a packed QKV weight or bias target into ordered Q, K, and V parameter names.
fn component_targets(target: &str) -> Option<[String; 3]> {
    for suffix in ["weight", "bias"] {
        if let Some(prefix) = target.strip_suffix(&format!(".query_key_value.{suffix}")) {
            return Some(
                ["q_proj", "k_proj", "v_proj"]
                    .map(|component| format!("{prefix}.{component}.{suffix}")),
            );
        }
    }
    None
}

/// Replaces packed staging names with actual Q/K/V targets in applied and conversion records.
///
/// The executor's payload-write counters remain packed-source counts; report names describe
/// logical installed parameters and are finalized into deterministic order.
fn expand_report_targets(report: &mut LoadReport) {
    let expand_names = |names: &mut Vec<String>| {
        *names = std::mem::take(names)
            .into_iter()
            .flat_map(|name| component_targets(&name).map_or_else(|| vec![name], Vec::from))
            .collect();
    };
    expand_names(&mut report.applied);
    expand_names(&mut report.allowlisted_missing);
    report.dtype_converted = std::mem::take(&mut report.dtype_converted)
        .into_iter()
        .flat_map(|record| {
            component_targets(&record.target).map_or_else(
                || vec![record.clone()],
                |targets| {
                    targets
                        .into_iter()
                        .map(|target| DTypeConversion {
                            target,
                            ..record.clone()
                        })
                        .collect()
                },
            )
        })
        .collect();
    report.finalize();
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
