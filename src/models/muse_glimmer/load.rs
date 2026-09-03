//! Streamed Burn parameter installation for Muse-Glimmer checkpoints.
//!
//! Pass one validates every source key, shape, dtype, shard path, and explicit
//! visual exception before this module receives data. Pass two holds at most one
//! source tensor and one converted output at a time, then dispatches the owned
//! [`TensorData`] into the statically ranked Burn field. Matrix installation goes
//! through Burn's layout-aware parameter mapper, preserving differentiability and
//! converting official `[output, input]` records to runtime `[input, output]` once.

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
    ConfigError, MuseGlimmerCheckpointOptions, MuseGlimmerDecoderLayer, MuseGlimmerForCausalLm,
};

/// Architecture-specific failure while applying one planned parameter.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MuseGlimmerLoadError {
    /// The target does not name an executable text-model parameter.
    #[error("unknown Muse-Glimmer parameter target `{target}`")]
    UnknownTarget {
        /// Unrecognized internal key.
        target: String,
    },
    /// A layer index or target structure is malformed or out of range.
    #[error("invalid Muse-Glimmer target `{target}`: {reason}")]
    InvalidIndex {
        /// Malformed internal key.
        target: String,
        /// Index or syntax diagnostic.
        reason: String,
    },
    /// The configured graph lacks a parameter required by the contract.
    #[error("Muse-Glimmer target `{target}` requires missing component `{component}`")]
    MissingComponent {
        /// Parameter being installed.
        target: String,
        /// Expected graph component.
        component: &'static str,
    },
    /// Tensor metadata disagrees with the statically ranked Burn field.
    #[error("cannot install Muse-Glimmer parameter `{target}`: {source}")]
    Parameter {
        /// Internal parameter target.
        target: String,
        /// Rank, shape, dtype, or backend storage failure.
        #[source]
        source: ParameterInstallError,
    },
    /// A tied parameter operation violated the embedding/head contract.
    #[error("invalid Muse-Glimmer alias `{target}` -> `{source_target}`: {reason}")]
    InvalidAlias {
        /// Alias target.
        target: String,
        /// Storage-owning target.
        source_target: String,
        /// Human-readable contract violation.
        reason: String,
    },
}

/// Failure from topology validation, strict planning, or streamed execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MuseGlimmerCheckpointLoadError {
    /// The model topology cannot describe a valid checkpoint.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// Artifact discovery or full expected/observed validation failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Pass two failed after zero or more audited writes.
    #[error(transparent)]
    Execution(#[from] Box<ExecutionFailure<MuseGlimmerLoadError>>),
}

impl MuseGlimmerCheckpointLoadError {
    /// Returns partial execution diagnostics when pass two reached the sink.
    #[must_use]
    pub fn execution_failure(&self) -> Option<&ExecutionFailure<MuseGlimmerLoadError>> {
        match self {
            Self::Execution(failure) => Some(failure),
            Self::Config(_) | Self::Checkpoint(_) => None,
        }
    }
}

impl<B: Backend> TensorSink for MuseGlimmerForCausalLm<B> {
    type Error = MuseGlimmerLoadError;

    /// Installs one storage-owning target from the validated language contract.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        match target {
            "model.embed_tokens.weight" => {
                install(target, &mut self.model.embed_tokens.weight, data)
            }
            "model.norm.weight" => install_norm(target, &mut self.model.norm, data),
            "lm_head.weight" => {
                let head = self.lm_head.as_mut().ok_or_else(|| {
                    MuseGlimmerLoadError::MissingComponent {
                        target: target.into(),
                        component: "independent LM head for an untied checkpoint",
                    }
                })?;
                install_linear_weight(target, head, data)
            }
            _ => {
                let (index, suffix) = parse_layer_target(target)?;
                let layer_count = self.model.layers.len();
                let layer = self.model.layers.get_mut(index).ok_or_else(|| {
                    MuseGlimmerLoadError::InvalidIndex {
                        target: target.into(),
                        reason: format!("layer {index} is outside 0..{layer_count}"),
                    }
                })?;
                install_layer_tensor(layer, target, suffix, data)
            }
        }
    }

    /// Validates the sole supported embedding-to-head storage alias.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        if target != "lm_head.weight" || source_target != "model.embed_tokens.weight" {
            return Err(MuseGlimmerLoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "only the tied embedding-to-LM-head alias is supported".into(),
            });
        }
        if !self.model.config.tie_word_embeddings || self.lm_head.is_some() {
            return Err(MuseGlimmerLoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the configured model declares an independent LM head".into(),
            });
        }
        if !self.model.embed_tokens.weight.is_initialized() {
            return Err(MuseGlimmerLoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the embedding storage owner has not been installed".into(),
            });
        }
        Ok(())
    }
}

impl<B: Backend> MuseGlimmerForCausalLm<B> {
    /// Strictly loads an official multimodal Muse artifact's language tower.
    ///
    /// No network access occurs. Visual tensors are audited as explicit ignored
    /// keys, while any unknown namespace remains a strict error.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, MuseGlimmerCheckpointLoadError> {
        self.load_checkpoint_with_family_options(
            root,
            options,
            MuseGlimmerCheckpointOptions::default(),
        )
    }

    /// Loads a local artifact using an explicit multimodal or text-only namespace.
    pub fn load_checkpoint_with_family_options(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
        family_options: MuseGlimmerCheckpointOptions,
    ) -> Result<ExecutionOutcome, MuseGlimmerCheckpointLoadError> {
        let manifest = CheckpointManifest::discover(root, options)?;
        let contract = self.config().checkpoint_contract(family_options)?;
        let plan = contract.build_load_plan(&manifest, options)?;
        Ok(execute_local_load_plan(&manifest, &plan, self)?)
    }
}

/// Parses `model.layers.<decimal>.<suffix>` without empty components.
fn parse_layer_target(target: &str) -> Result<(usize, &str), MuseGlimmerLoadError> {
    let remainder = target.strip_prefix("model.layers.").ok_or_else(|| {
        MuseGlimmerLoadError::UnknownTarget {
            target: target.into(),
        }
    })?;
    let (index, suffix) =
        remainder
            .split_once('.')
            .ok_or_else(|| MuseGlimmerLoadError::InvalidIndex {
                target: target.into(),
                reason: "expected `model.layers.<index>.<parameter>`".into(),
            })?;
    if suffix.is_empty() {
        return Err(MuseGlimmerLoadError::InvalidIndex {
            target: target.into(),
            reason: "parameter suffix must not be empty".into(),
        });
    }
    let index = index
        .parse::<usize>()
        .map_err(|error| MuseGlimmerLoadError::InvalidIndex {
            target: target.into(),
            reason: format!("`{index}` is not a decimal usize: {error}"),
        })?;
    Ok((index, suffix))
}

/// Dispatches a target within one resolved Muse decoder layer.
fn install_layer_tensor<B: Backend>(
    layer: &mut MuseGlimmerDecoderLayer<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), MuseGlimmerLoadError> {
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
        "self_attn.q_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.q_proj, data)
        }
        "self_attn.k_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.k_proj, data)
        }
        "self_attn.v_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.v_proj, data)
        }
        "self_attn.gate_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.gate_proj, data)
        }
        "self_attn.o_proj.weight" => {
            install_linear_weight(target, &mut layer.self_attn.o_proj, data)
        }
        "self_attn.q_proj.bias" => install_linear_bias(target, &mut layer.self_attn.q_proj, data),
        "self_attn.k_proj.bias" => install_linear_bias(target, &mut layer.self_attn.k_proj, data),
        "self_attn.v_proj.bias" => install_linear_bias(target, &mut layer.self_attn.v_proj, data),
        "self_attn.o_proj.bias" => install_linear_bias(target, &mut layer.self_attn.o_proj, data),
        "mlp.gate_proj.weight" => install_mlp_weight(target, &mut layer.mlp, "gate", data),
        "mlp.up_proj.weight" => install_mlp_weight(target, &mut layer.mlp, "up", data),
        "mlp.down_proj.weight" => install_mlp_weight(target, &mut layer.mlp, "down", data),
        _ => Err(MuseGlimmerLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs one named SwiGLU projection.
fn install_mlp_weight<B: Backend>(
    target: &str,
    mlp: &mut GatedFeedForward<B>,
    projection: &str,
    data: TensorData,
) -> Result<(), MuseGlimmerLoadError> {
    match projection {
        "gate" => install_linear_weight(target, &mut mlp.gate_proj, data),
        "up" => install_linear_weight(target, &mut mlp.up_proj, data),
        "down" => install_linear_weight(target, &mut mlp.down_proj, data),
        _ => Err(MuseGlimmerLoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs a matrix through Burn's layout-aware record mapper.
fn install_linear_weight<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), MuseGlimmerLoadError> {
    install(target, &mut linear.weight, data)
}

/// Installs one enabled attention bias.
fn install_linear_bias<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), MuseGlimmerLoadError> {
    let bias = linear
        .bias
        .as_mut()
        .ok_or_else(|| MuseGlimmerLoadError::MissingComponent {
            target: target.into(),
            component: "configured attention projection bias",
        })?;
    install(target, bias, data)
}

/// Installs one learned RMSNorm scale vector.
fn install_norm<B: Backend>(
    target: &str,
    norm: &mut RmsNorm<B>,
    data: TensorData,
) -> Result<(), MuseGlimmerLoadError> {
    install(target, &mut norm.weight, data)
}

/// Validates and installs one statically ranked floating-point parameter.
fn install<B: Backend, const D: usize>(
    target: &str,
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> Result<(), MuseGlimmerLoadError> {
    install_float_parameter(parameter, data).map_err(|source| MuseGlimmerLoadError::Parameter {
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
    use crate::models::muse_glimmer::{MuseGlimmerCheckpointSource, MuseGlimmerTextConfig};

    type TestBackend = Flex<f32, i32>;

    /// Builds a one-layer miniature suitable for sink and I/O tests.
    fn tiny_config(tied: bool) -> MuseGlimmerTextConfig {
        MuseGlimmerTextConfig {
            vocab_size: 16,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 16,
            sliding_window: Some(4),
            layer_types: vec![super::super::MuseGlimmerLayerType::FullAttention],
            layer_rope_theta: vec![0.0],
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: tied,
            ..Default::default()
        }
    }

    /// Writes a complete zero-valued F32 SafeTensors file from the mapping plan.
    fn write_checkpoint(
        root: &Path,
        contract: &super::super::MuseGlimmerCheckpointContract,
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

    /// Every family target installs into its exact field and the graph runs.
    #[test]
    fn complete_contract_installs_and_runs() {
        let config = tiny_config(false);
        let contract = config
            .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
            .unwrap();
        let device = Default::default();
        let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config, &device).unwrap();
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

    /// Local pass two streams all text tensors while preserving strict accounting.
    #[test]
    fn local_loader_streams_tied_and_untied_text_checkpoints() {
        let device = Default::default();
        for tied in [false, true] {
            let config = tiny_config(tied);
            let family_options = MuseGlimmerCheckpointOptions {
                source: MuseGlimmerCheckpointSource::TextOnly,
            };
            let contract = config.checkpoint_contract(family_options).unwrap();
            let temp = TempDir::new().unwrap();
            write_checkpoint(temp.path(), &contract, |_| {});
            let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config, &device).unwrap();
            let outcome = model
                .load_checkpoint_with_family_options(
                    temp.path(),
                    &LoadOptions::default(),
                    family_options,
                )
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

    /// Missing, unknown, and wrong-shaped tensors fail before any sink write.
    #[test]
    fn malformed_inventory_fails_before_parameter_initialization() {
        let device = Default::default();
        let config = tiny_config(false);
        let family_options = MuseGlimmerCheckpointOptions {
            source: MuseGlimmerCheckpointSource::TextOnly,
        };
        let contract = config.checkpoint_contract(family_options).unwrap();

        let missing = TempDir::new().unwrap();
        write_checkpoint(missing.path(), &contract, |tensors| {
            tensors.retain(|(name, _)| !name.ends_with("self_attn.gate_proj.weight"));
        });
        let mut model =
            MuseGlimmerForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
        let error = model
            .load_checkpoint_with_family_options(
                missing.path(),
                &LoadOptions::default(),
                family_options,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            MuseGlimmerCheckpointLoadError::Checkpoint(
                CheckpointError::MissingMappingSource { .. }
            )
        ));
        assert!(!model.model.embed_tokens.weight.is_initialized());

        let unexpected = TempDir::new().unwrap();
        write_checkpoint(unexpected.path(), &contract, |tensors| {
            tensors.push(("unknown.weight".into(), vec![1]));
        });
        let mut model =
            MuseGlimmerForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
        let error = model
            .load_checkpoint_with_family_options(
                unexpected.path(),
                &LoadOptions::default(),
                family_options,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            MuseGlimmerCheckpointLoadError::Checkpoint(CheckpointError::StrictValidation {
                unexpected: 1,
                ..
            })
        ));
        assert!(!model.model.embed_tokens.weight.is_initialized());

        let wrong = TempDir::new().unwrap();
        write_checkpoint(wrong.path(), &contract, |tensors| {
            tensors
                .iter_mut()
                .find(|(name, _)| name.ends_with("embed_tokens.weight"))
                .unwrap()
                .1 = vec![15, 4];
        });
        let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config, &device).unwrap();
        let error = model
            .load_checkpoint_with_family_options(
                wrong.path(),
                &LoadOptions::default(),
                family_options,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            MuseGlimmerCheckpointLoadError::Checkpoint(CheckpointError::ShapeMismatch { .. })
        ));
        assert!(!model.model.embed_tokens.weight.is_initialized());
    }

    /// Sink dispatch errors are typed rather than falling through to tensor panics.
    #[test]
    fn sink_rejects_bad_targets_payloads_and_aliases() {
        let device = Default::default();
        let mut model =
            MuseGlimmerForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
        assert!(matches!(
            model.write_tensor(
                "model.embed_tokens.weight",
                TensorData::zeros::<i32, _>([16, 4])
            ),
            Err(MuseGlimmerLoadError::Parameter { .. })
        ));
        assert!(matches!(
            model.write_tensor("model.layers.bad.weight", TensorData::from([0.0_f32])),
            Err(MuseGlimmerLoadError::InvalidIndex { .. })
        ));
        assert!(matches!(
            model.write_tensor(
                "model.layers.9.input_layernorm.weight",
                TensorData::zeros::<f32, _>([4])
            ),
            Err(MuseGlimmerLoadError::InvalidIndex { .. })
        ));
        assert!(matches!(
            model.alias_tensor("lm_head.weight", "model.embed_tokens.weight"),
            Err(MuseGlimmerLoadError::InvalidAlias { .. })
        ));
    }
}
