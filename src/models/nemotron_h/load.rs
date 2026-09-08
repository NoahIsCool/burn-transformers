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
    Config(#[from] ConfigError),
    /// Artifact discovery or complete expected/observed validation failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Pass two failed after zero or more audited sink writes.
    #[error(transparent)]
    Execution(#[from] Box<ExecutionFailure<NemotronHLoadError>>),
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
                let (layer_index, suffix) = parse_indexed_target(target, "model.layers.")?;
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

/// Parses `<prefix><decimal>.<suffix>` without accepting empty components.
fn parse_indexed_target<'a>(
    target: &'a str,
    prefix: &str,
) -> Result<(usize, &'a str), NemotronHLoadError> {
    let remainder =
        target
            .strip_prefix(prefix)
            .ok_or_else(|| NemotronHLoadError::UnknownTarget {
                target: target.into(),
            })?;
    let (index, suffix) =
        remainder
            .split_once('.')
            .ok_or_else(|| NemotronHLoadError::InvalidIndex {
                target: target.into(),
                reason: format!("expected `{prefix}<index>.<parameter>`"),
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
            install_mamba_tensor(mamba, target, &suffix[6..], data)
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
            install_attention_tensor(attention, target, &suffix[10..], data)
        }
        suffix if suffix.starts_with("mlp.") => {
            let mlp = block
                .mlp
                .as_mut()
                .ok_or_else(|| NemotronHLoadError::MissingComponent {
                    target: target.into(),
                    component: "dense MLP selected by the block schedule",
                })?;
            install_dense_tensor(mlp, target, &suffix[4..], data)
        }
        suffix if suffix.starts_with("moe.") => {
            let moe = block
                .moe
                .as_mut()
                .ok_or_else(|| NemotronHLoadError::MissingComponent {
                    target: target.into(),
                    component: "routed MoE selected by the block schedule",
                })?;
            install_moe_tensor(moe, target, &suffix[4..], data)
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
            let (expert_index, projection) = parse_indexed_suffix(target, suffix, "routed.")?;
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
            install_dense_tensor(&mut moe.shared, target, &suffix[7..], data)
        }
        suffix if suffix.starts_with("fc1_latent_proj.") => {
            let projection = moe.fc1_latent_proj.as_mut().ok_or_else(|| {
                NemotronHLoadError::MissingComponent {
                    target: target.into(),
                    component: "first routed-expert latent projection",
                }
            })?;
            install_linear_tensor(projection, target, &suffix[16..], data)
        }
        suffix if suffix.starts_with("fc2_latent_proj.") => {
            let projection = moe.fc2_latent_proj.as_mut().ok_or_else(|| {
                NemotronHLoadError::MissingComponent {
                    target: target.into(),
                    component: "second routed-expert latent projection",
                }
            })?;
            install_linear_tensor(projection, target, &suffix[16..], data)
        }
        _ => Err(NemotronHLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Parses an indexed fragment already isolated from its full target prefix.
fn parse_indexed_suffix<'a>(
    target: &str,
    suffix: &'a str,
    prefix: &str,
) -> Result<(usize, &'a str), NemotronHLoadError> {
    let remainder =
        suffix
            .strip_prefix(prefix)
            .ok_or_else(|| NemotronHLoadError::UnknownTarget {
                target: target.into(),
            })?;
    let (index, projection) =
        remainder
            .split_once('.')
            .ok_or_else(|| NemotronHLoadError::InvalidIndex {
                target: target.into(),
                reason: format!("expected `{prefix}<index>.<parameter>`"),
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
        suffix if suffix.starts_with("attention.") => {
            install_attention_tensor(&mut head.attention, target, &suffix[10..], data)
        }
        "moe_norm.weight" => install_norm(target, &mut head.moe_norm, data),
        suffix if suffix.starts_with("moe.") => {
            install_moe_tensor(&mut head.moe, target, &suffix[4..], data)
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
mod tests {
    use std::{fs::File, io::Write, path::Path};

    use burn::{
        backend::Flex,
        tensor::{Int, Tensor, TensorData},
    };
    use serde_json::{Map, Value, json};
    use tempfile::TempDir;

    use crate::{
        CausalLmInput, ForwardOptions,
        checkpoint::{CheckpointError, TensorOperation},
    };

    use super::*;
    use crate::models::nemotron_h::{
        NemotronHBlockType, NemotronHCheckpointContract, NemotronHConfig,
    };

    type TestBackend = Flex<f32, i32>;

    /// Reduced hybrid topology retaining every sink-dispatch branch and MTP.
    fn tiny_config(tied: bool) -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 16,
            hidden_size: 8,
            layers_block_type: vec![
                NemotronHBlockType::LinearAttention,
                NemotronHBlockType::FullAttention,
                NemotronHBlockType::Mlp,
                NemotronHBlockType::Moe,
            ],
            num_hidden_layers: Some(4),
            tie_word_embeddings: tied,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 16,
            intermediate_size: 12,
            ssm_state_size: 3,
            mamba_num_heads: 4,
            mamba_head_dim: 2,
            n_groups: 2,
            conv_kernel: 3,
            chunk_size: 2,
            n_routed_experts: 4,
            n_shared_experts: 1,
            moe_intermediate_size: 6,
            moe_shared_expert_intermediate_size: 8,
            num_experts_per_tok: 2,
            n_group: 2,
            topk_group: 1,
            num_nextn_predict_layers: 1,
            ..Default::default()
        }
    }

    /// Writes one complete, zero-valued F32 SafeTensors fixture.
    fn write_checkpoint(
        root: &Path,
        contract: &NemotronHCheckpointContract,
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>)>),
    ) {
        let mut tensors = contract
            .mappings()
            .operations()
            .iter()
            .filter_map(|operation| match operation {
                TensorOperation::Map { source, target, .. } => {
                    let shape = contract
                        .expected_tensors()
                        .iter()
                        .find(|expected| &expected.name == target)
                        .unwrap()
                        .shape
                        .clone();
                    Some((source.clone(), shape))
                }
                TensorOperation::Alias { .. }
                | TensorOperation::Split { .. }
                | TensorOperation::Concatenate { .. } => None,
            })
            .collect::<Vec<_>>();
        mutate(&mut tensors);

        let mut offset = 0_u64;
        let mut header = Map::new();
        for (name, shape) in tensors {
            let elements = shape.iter().copied().product::<u64>();
            let end = offset + elements * 4;
            header.insert(
                name,
                json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}),
            );
            offset = end;
        }
        let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&vec![0; offset as usize]).unwrap();
    }

    /// Every base and MTP contract target reaches the exact ranked Burn field.
    #[test]
    fn complete_contract_installs_and_runs() {
        let config = tiny_config(false);
        let contract = config
            .checkpoint_contract(NemotronHCheckpointOptions::default())
            .unwrap();
        let device = Default::default();
        let mut model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
        for expected in contract.expected_tensors() {
            let shape = expected
                .shape
                .iter()
                .map(|&dimension| dimension as usize)
                .collect::<Vec<_>>();
            model
                .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
                .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
        }
        let output = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap();
        assert!(
            output
                .logits
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|&value| value == 0.0)
        );
    }

    /// Pass two opens one local shard and applies tied and untied contracts.
    #[test]
    fn local_loader_streams_tied_and_untied_checkpoints() {
        let device = Default::default();
        for tied in [false, true] {
            let config = tiny_config(tied);
            let contract = config
                .checkpoint_contract(NemotronHCheckpointOptions::default())
                .unwrap();
            let temp = TempDir::new().unwrap();
            write_checkpoint(temp.path(), &contract, |_| {});
            let mut model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
            let outcome = model
                .load_checkpoint(temp.path(), &LoadOptions::default())
                .unwrap();
            assert_eq!(
                outcome.report.applied.len(),
                contract.expected_tensors().len()
            );
            assert_eq!(outcome.stats.source.shard_opens, 1);
            assert_eq!(outcome.stats.source.max_open_shards, 1);
            assert_eq!(outcome.stats.aliases_applied, usize::from(tied));
        }
    }

    /// Metadata mismatches fail before the first lazy parameter is initialized.
    #[test]
    fn strict_mismatches_fail_before_writes() {
        let device = Default::default();
        let config = tiny_config(false);
        let contract = config
            .checkpoint_contract(NemotronHCheckpointOptions::default())
            .unwrap();

        let missing = TempDir::new().unwrap();
        write_checkpoint(missing.path(), &contract, |tensors| {
            tensors.retain(|(name, _)| name != "backbone.layers.0.mixer.A_log");
        });
        let mut model = NemotronHForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
        let error = model
            .load_checkpoint(missing.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(
            error,
            NemotronHCheckpointLoadError::Checkpoint(CheckpointError::MissingMappingSource { .. })
        ));
        assert!(!model.model.embed_tokens.weight.is_initialized());

        let unexpected = TempDir::new().unwrap();
        write_checkpoint(unexpected.path(), &contract, |tensors| {
            tensors.push(("unexpected.weight".into(), vec![1]));
        });
        let mut model = NemotronHForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
        let error = model
            .load_checkpoint(unexpected.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(
            error,
            NemotronHCheckpointLoadError::Checkpoint(CheckpointError::StrictValidation {
                unexpected: 1,
                ..
            })
        ));
        assert!(!model.model.embed_tokens.weight.is_initialized());

        let wrong_shape = TempDir::new().unwrap();
        write_checkpoint(wrong_shape.path(), &contract, |tensors| {
            tensors
                .iter_mut()
                .find(|(name, _)| name == "backbone.embeddings.weight")
                .unwrap()
                .1 = vec![15, 8];
        });
        let mut model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
        let error = model
            .load_checkpoint(wrong_shape.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(
            error,
            NemotronHCheckpointLoadError::Checkpoint(CheckpointError::ShapeMismatch { .. })
        ));
        assert!(!model.model.embed_tokens.weight.is_initialized());
    }

    /// Malformed ranks, indexes, aliases, and absent mixers return typed errors.
    #[test]
    fn sink_rejects_invalid_targets_and_payloads() {
        let device = Default::default();
        let mut model =
            NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
        assert!(matches!(
            model.write_tensor(
                "model.embed_tokens.weight",
                TensorData::zeros::<i32, _>([16, 8])
            ),
            Err(NemotronHLoadError::Parameter { .. })
        ));
        assert!(matches!(
            model.write_tensor("model.layers.bad.weight", TensorData::from([0.0_f32])),
            Err(NemotronHLoadError::InvalidIndex { .. })
        ));
        assert!(matches!(
            model.write_tensor(
                "model.layers.3.moe.routed.9.up_proj.weight",
                TensorData::zeros::<f32, _>([6, 8])
            ),
            Err(NemotronHLoadError::InvalidIndex { .. })
        ));
        assert!(matches!(
            model.write_tensor(
                "model.layers.0.attention.q_proj.weight",
                TensorData::zeros::<f32, _>([8, 8])
            ),
            Err(NemotronHLoadError::MissingComponent { .. })
        ));
        assert!(matches!(
            model.alias_tensor("other.weight", "model.embed_tokens.weight"),
            Err(NemotronHLoadError::InvalidAlias { .. })
        ));
    }
}
