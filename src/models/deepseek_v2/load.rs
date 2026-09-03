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
        ParameterInstallError, TensorSink, execute_local_load_plan, install_float_parameter,
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
    Config(#[from] ConfigError),
    /// Artifact discovery or strict inventory validation failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Pass two failed after zero or more audited writes.
    #[error(transparent)]
    Execution(#[from] Box<ExecutionFailure<DeepseekV2LoadError>>),
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
        let manifest = CheckpointManifest::discover(root, options)?;
        let contract = self
            .config()
            .checkpoint_contract(DeepseekV2CheckpointOptions::default())?;
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

/// Installs a statically ranked float parameter with target context.
fn install<B: Backend, const D: usize>(
    target: &str,
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> Result<(), DeepseekV2LoadError> {
    install_float_parameter(parameter, data).map_err(|source| DeepseekV2LoadError::Parameter {
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

    use super::*;
    use crate::{
        CausalLmInput, ForwardOptions,
        checkpoint::{CheckpointError, TensorOperation, TensorSink},
        models::deepseek_v2::model::tests_support::tiny_config,
    };

    type TestBackend = Flex<f32, i32>;

    /// Writes a complete zero-valued F32 SafeTensors artifact.
    fn write_checkpoint(
        root: &Path,
        contract: &super::super::DeepseekV2CheckpointContract,
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
                _ => None,
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

    /// Invalid target syntax is rejected before model indexing.
    #[test]
    fn target_parser_rejects_unknown_and_malformed_indices() {
        assert!(matches!(
            parse_layer_target("not.a.layer"),
            Err(DeepseekV2LoadError::UnknownTarget { .. })
        ));
        assert!(matches!(
            parse_layer_target("model.layers.x.self_attn.q_proj.weight"),
            Err(DeepseekV2LoadError::InvalidIndex { .. })
        ));
    }

    /// A strict target cannot install a tensor with the wrong checkpoint shape.
    #[test]
    fn sink_rejects_wrong_parameter_shape() {
        let device = Default::default();
        let mut config = super::super::model::tests_support::tiny_config();
        config.tie_word_embeddings = false;
        let mut model = DeepseekV2ForCausalLm::<Flex>::init(config, &device).unwrap();
        let error = TensorSink::write_tensor(
            &mut model,
            "model.embed_tokens.weight",
            TensorData::zeros::<f32, _>([1, 1]),
        )
        .unwrap_err();
        assert!(matches!(error, DeepseekV2LoadError::Parameter { .. }));
    }

    /// Every dense, MLA, routed, and shared target installs and executes.
    #[test]
    fn complete_contract_installs_and_runs() {
        let config = tiny_config();
        let contract = config
            .checkpoint_contract(DeepseekV2CheckpointOptions::default())
            .unwrap();
        let device = Default::default();
        let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
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

    /// Streamed local loading opens one shard and applies every owner once.
    #[test]
    fn local_loader_streams_tied_and_untied_checkpoints() {
        let device = Default::default();
        for tied in [false, true] {
            let mut config = tiny_config();
            config.tie_word_embeddings = tied;
            let contract = config
                .checkpoint_contract(DeepseekV2CheckpointOptions::default())
                .unwrap();
            let temp = TempDir::new().unwrap();
            write_checkpoint(temp.path(), &contract, |_| {});
            let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
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

    /// Missing, unexpected, and wrong-shaped inventory fails before first write.
    #[test]
    fn strict_planning_rejects_inventory_mismatches_before_loading() {
        let device = Default::default();
        let config = tiny_config();
        let contract = config
            .checkpoint_contract(DeepseekV2CheckpointOptions::default())
            .unwrap();

        let missing = TempDir::new().unwrap();
        write_checkpoint(missing.path(), &contract, |tensors| {
            tensors.retain(|(name, _)| name != "model.layers.1.mlp.gate.weight");
        });
        let mut model =
            DeepseekV2ForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
        let error = model
            .load_checkpoint(missing.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(
            error,
            DeepseekV2CheckpointLoadError::Checkpoint(CheckpointError::MissingMappingSource { .. })
        ));
        assert!(!model.model.embed_tokens.weight.is_initialized());

        let unexpected = TempDir::new().unwrap();
        write_checkpoint(unexpected.path(), &contract, |tensors| {
            tensors.push(("unexpected.weight".into(), vec![1]));
        });
        let mut model =
            DeepseekV2ForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
        let error = model
            .load_checkpoint(unexpected.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(
            error,
            DeepseekV2CheckpointLoadError::Checkpoint(CheckpointError::StrictValidation {
                unexpected: 1,
                ..
            })
        ));
        assert!(!model.model.embed_tokens.weight.is_initialized());

        let wrong = TempDir::new().unwrap();
        write_checkpoint(wrong.path(), &contract, |tensors| {
            tensors
                .iter_mut()
                .find(|(name, _)| name == "model.embed_tokens.weight")
                .unwrap()
                .1 = vec![31, 8];
        });
        let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
        let error = model
            .load_checkpoint(wrong.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(
            error,
            DeepseekV2CheckpointLoadError::Checkpoint(CheckpointError::ShapeMismatch { .. })
        ));
        assert!(!model.model.embed_tokens.weight.is_initialized());
    }

    /// Query-LoRA targets and expert bounds dispatch to typed optional fields.
    #[test]
    fn query_lora_and_invalid_expert_targets_are_typed() {
        let device = Default::default();
        let mut config = tiny_config();
        config.q_lora_rank = Some(4);
        let contract = config.checkpoint_contract(Default::default()).unwrap();
        let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
        for expected in contract.expected_tensors() {
            let shape = expected
                .shape
                .iter()
                .map(|&value| value as usize)
                .collect::<Vec<_>>();
            model
                .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
                .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
        }
        assert!(matches!(
            model.write_tensor(
                "model.layers.1.sparse_mlp.routed.routed.9.gate_proj.weight",
                TensorData::zeros::<f32, _>([6, 8]),
            ),
            Err(DeepseekV2LoadError::InvalidIndex { .. })
        ));
        assert!(matches!(
            model.write_tensor(
                "model.layers.0.self_attn.q_proj.weight",
                TensorData::zeros::<f32, _>([10, 8]),
            ),
            Err(DeepseekV2LoadError::MissingComponent { .. })
        ));
    }
}
