//! Architecture-owned Burn parameter installation for T5 checkpoints.
//!
//! Checkpoint planning operates on dynamic names, ranks, and byte buffers, while
//! Burn encodes tensor rank and linear-layout mappers in concrete Rust fields.
//! The [`TensorSink`] implementation below is therefore deliberately owned by
//! the T5 architecture. It accepts only names emitted by [`T5CheckpointContract`]
//! and installs weights through Burn's record-loading path. In particular, this
//! preserves the transpose mapper that converts serialized linear weights from
//! `[output, input]` to the runtime multiplication layout `[input, output]`.
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
        CheckpointError, ExecutionFailure, ExecutionOutcome, LoadOptions, ParameterInstallError,
        TensorSink, execute_local_load_plan, install_float_parameter,
    },
    normalization::RmsNorm,
};

use super::{ConfigError, T5Attention, T5Block, T5ForConditionalGeneration, T5LayerFeedForward};

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
    Config(#[from] ConfigError),
    /// Artifact discovery or strict expected-versus-observed planning failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Pass two failed after zero or more audited model writes.
    #[error(transparent)]
    Execution(#[from] Box<ExecutionFailure<T5LoadError>>),
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
    /// Pass one reads only bounded metadata and rejects missing, unexpected, or
    /// shape-incompatible tensors before pass two initializes any model parameter.
    /// Pass two holds at most one source payload and one output buffer at a time.
    /// On an execution failure the returned error retains the exact partial
    /// [`crate::checkpoint::LoadReport`]; callers should discard the partially
    /// initialized model rather than continue inference.
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

    let feed_forward_layer = match stack {
        Stack::Encoder => 1,
        Stack::Decoder => 2,
    };
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

/// Validates and installs one statically ranked floating-point parameter.
fn install<B: Backend, const D: usize>(
    target: &str,
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> Result<(), T5LoadError> {
    install_float_parameter(parameter, data).map_err(|source| T5LoadError::Parameter {
        target: target.into(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io::Write,
        path::{Path, PathBuf},
    };

    use burn::{
        backend::Flex,
        tensor::{Int, Tensor, TensorData},
    };
    use serde_json::{Map, Value, json};
    use tempfile::TempDir;

    use crate::{
        ForwardOptions,
        architecture::{CausalLmInput, Seq2SeqInput},
        checkpoint::{
            CheckpointError, ExecutionFailureCause, LoadOptions, TensorDType, TensorOperation,
        },
        models::config::ModelConfig,
    };

    use super::*;
    use crate::models::t5::{T5Config, T5FeedForwardProjection};

    type TestBackend = Flex<f32, i32>;

    /// Returns a minimal but complete asymmetric encoder-decoder topology.
    fn tiny(projection: T5FeedForwardProjection, tied: bool) -> T5Config {
        T5Config {
            vocab_size: 13,
            d_model: 4,
            d_kv: 2,
            d_ff: 6,
            num_layers: 2,
            num_decoder_layers: Some(1),
            num_heads: 2,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 16,
            feed_forward_proj: projection,
            dense_act_fn: None,
            is_gated_act: None,
            tie_word_embeddings: tied,
            ..T5Config::default()
        }
    }

    /// Serializes zero-valued tensors in the minimal SafeTensors wire format.
    fn write_checkpoint(
        root: &Path,
        contract: &super::super::T5CheckpointContract,
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>)>),
    ) {
        let alias_targets = contract
            .mappings()
            .operations()
            .iter()
            .filter_map(|operation| match operation {
                TensorOperation::Alias { target, .. } => Some(target.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut tensors = contract
            .expected_tensors()
            .iter()
            .filter(|tensor| !alias_targets.contains(&tensor.name.as_str()))
            .map(|tensor| (tensor.name.clone(), tensor.shape.clone()))
            .collect::<Vec<_>>();
        mutate(&mut tensors);

        let mut offset = 0_u64;
        let mut header = Map::new();
        for (name, shape) in &tensors {
            let elements = shape
                .iter()
                .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension));
            let bytes = elements.unwrap().checked_mul(4).unwrap();
            let end = offset.checked_add(bytes).unwrap();
            header.insert(
                name.clone(),
                json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, end],
                }),
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
        file.write_all(&vec![0_u8; offset as usize]).unwrap();
    }

    /// Writes every storage-owning target directly through the model sink.
    fn install_contract(model: &mut T5ForConditionalGeneration<TestBackend>, config: &T5Config) {
        let contract = config.checkpoint_contract().unwrap();
        for expected in contract.expected_tensors() {
            if expected.name == "lm_head.weight" && config.tie_word_embeddings {
                continue;
            }
            let shape = expected
                .shape
                .iter()
                .map(|dimension| usize::try_from(*dimension).unwrap())
                .collect::<Vec<_>>();
            model
                .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
                .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
        }
        if config.tie_word_embeddings {
            model
                .alias_tensor("lm_head.weight", "shared.weight")
                .unwrap();
        }
    }

    #[test]
    fn every_dense_gated_tied_and_untied_target_dispatches() {
        use T5FeedForwardProjection::*;

        let device = Default::default();
        for projection in [Relu, Gelu, GatedGelu, GatedGeluNew, GatedSilu] {
            for tied in [false, true] {
                let config = tiny(projection, tied);
                let mut model =
                    T5ForConditionalGeneration::<TestBackend>::init(config.clone(), &device)
                        .unwrap();
                install_contract(&mut model, &config);
            }
        }
    }

    #[test]
    fn zero_checkpoint_runs_a_complete_stateless_forward_pass() {
        let device = Default::default();
        let config = tiny(T5FeedForwardProjection::GatedGelu, false);
        let mut model =
            T5ForConditionalGeneration::<TestBackend>::init(config.clone(), &device).unwrap();
        install_contract(&mut model, &config);

        let output = model
            .forward(
                Seq2SeqInput {
                    encoder: CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[2, 3]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    decoder: CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0, 4]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                },
                &ForwardOptions::default(),
            )
            .unwrap();
        assert_eq!(output.decoder.logits.dims(), [1, 1, 13]);
        assert!(
            output
                .decoder
                .logits
                .to_data()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    #[test]
    fn invalid_dispatch_paths_are_contextual_and_non_panicking() {
        let device = Default::default();
        let mut tied = T5ForConditionalGeneration::<TestBackend>::init(
            tiny(T5FeedForwardProjection::GatedSilu, true),
            &device,
        )
        .unwrap();

        assert!(matches!(
            tied.alias_tensor("lm_head.weight", "shared.weight"),
            Err(T5LoadError::InvalidAlias { .. })
        ));
        assert!(matches!(
            tied.write_tensor("adapter.weight", TensorData::from([0.0_f32])),
            Err(T5LoadError::UnknownTarget { .. })
        ));
        assert!(matches!(
            tied.write_tensor(
                "encoder.block.nope.layer.0.layer_norm.weight",
                TensorData::zeros::<f32, _>([4])
            ),
            Err(T5LoadError::InvalidBlock { .. })
        ));
        assert!(matches!(
            tied.write_tensor(
                "decoder.block.9.layer.0.layer_norm.weight",
                TensorData::zeros::<f32, _>([4])
            ),
            Err(T5LoadError::InvalidBlock { .. })
        ));
        assert!(matches!(
            tied.write_tensor(
                "encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight",
                TensorData::zeros::<f32, _>([8, 2])
            ),
            Err(T5LoadError::MissingComponent { .. })
        ));
        assert!(matches!(
            tied.write_tensor(
                "encoder.block.0.layer.1.DenseReluDense.wi.weight",
                TensorData::zeros::<f32, _>([6, 4])
            ),
            Err(T5LoadError::MissingComponent { .. })
        ));
        assert!(matches!(
            tied.write_tensor("shared.weight", TensorData::zeros::<f32, _>([12, 4])),
            Err(T5LoadError::Parameter {
                source: ParameterInstallError::Shape { .. },
                ..
            })
        ));

        tied.write_tensor("shared.weight", TensorData::zeros::<f32, _>([13, 4]))
            .unwrap();
        assert!(matches!(
            tied.alias_tensor("decoder.embed_tokens.weight", "shared.weight"),
            Err(T5LoadError::InvalidAlias { .. })
        ));

        let mut untied = T5ForConditionalGeneration::<TestBackend>::init(
            tiny(T5FeedForwardProjection::Relu, false),
            &device,
        )
        .unwrap();
        untied
            .write_tensor("shared.weight", TensorData::zeros::<f32, _>([13, 4]))
            .unwrap();
        assert!(matches!(
            untied.alias_tensor("lm_head.weight", "shared.weight"),
            Err(T5LoadError::InvalidAlias { .. })
        ));
        assert!(matches!(
            untied.write_tensor(
                "decoder.block.0.layer.2.DenseReluDense.wi_0.weight",
                TensorData::zeros::<f32, _>([6, 4])
            ),
            Err(T5LoadError::MissingComponent { .. })
        ));
    }

    #[test]
    fn local_loader_plans_then_streams_tied_and_untied_checkpoints() {
        let device = Default::default();
        for tied in [false, true] {
            let config = tiny(T5FeedForwardProjection::GatedGelu, tied);
            let contract = config.checkpoint_contract().unwrap();
            let temp = TempDir::new().unwrap();
            write_checkpoint(temp.path(), &contract, |_| {});

            let planned = config
                .plan_checkpoint(temp.path(), &LoadOptions::default())
                .unwrap();
            assert_eq!(planned.load_plan().parameter_count(), 38);
            assert_eq!(planned.manifest().tensors.len(), if tied { 37 } else { 38 });
            let (_manifest, generic_plan) = planned.into_parts();
            assert_eq!(generic_plan.parameter_count(), 38);

            let mut model =
                T5ForConditionalGeneration::<TestBackend>::init(config, &device).unwrap();
            let outcome = model
                .load_checkpoint(temp.path(), &LoadOptions::default())
                .unwrap();
            assert_eq!(outcome.report.applied.len(), 38);
            assert_eq!(outcome.stats.tensors_applied, if tied { 37 } else { 38 });
            assert_eq!(outcome.stats.aliases_applied, usize::from(tied));
            assert_eq!(outcome.stats.source.shard_opens, 1);
            assert_eq!(outcome.stats.source.max_open_shards, 1);
        }
    }

    #[test]
    fn planning_failures_precede_all_model_parameter_writes() {
        let device = Default::default();
        let config = tiny(T5FeedForwardProjection::GatedGelu, false);
        let contract = config.checkpoint_contract().unwrap();

        let missing = TempDir::new().unwrap();
        write_checkpoint(missing.path(), &contract, |tensors| {
            tensors.retain(|(name, _)| name != "encoder.final_layer_norm.weight");
        });
        let mut missing_model =
            T5ForConditionalGeneration::<TestBackend>::init(config.clone(), &device).unwrap();
        let error = missing_model
            .load_checkpoint(missing.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(
            error,
            T5CheckpointLoadError::Checkpoint(CheckpointError::MissingMappingSource { .. })
        ));
        assert!(!missing_model.shared.weight.is_initialized());

        let unexpected = TempDir::new().unwrap();
        write_checkpoint(unexpected.path(), &contract, |tensors| {
            tensors.push(("unexpected.weight".into(), vec![1]));
        });
        let mut unexpected_model =
            T5ForConditionalGeneration::<TestBackend>::init(config.clone(), &device).unwrap();
        let error = unexpected_model
            .load_checkpoint(unexpected.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(
            error,
            T5CheckpointLoadError::Checkpoint(CheckpointError::StrictValidation {
                unexpected: 1,
                ..
            })
        ));
        assert!(!unexpected_model.shared.weight.is_initialized());

        let wrong_shape = TempDir::new().unwrap();
        write_checkpoint(wrong_shape.path(), &contract, |tensors| {
            tensors
                .iter_mut()
                .find(|(name, _)| name == "shared.weight")
                .unwrap()
                .1 = vec![12, 4];
        });
        let mut wrong_shape_model =
            T5ForConditionalGeneration::<TestBackend>::init(config, &device).unwrap();
        let error = wrong_shape_model
            .load_checkpoint(wrong_shape.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(
            error,
            T5CheckpointLoadError::Checkpoint(CheckpointError::ShapeMismatch { tensor, .. })
                if tensor == "shared.weight"
        ));
        assert!(!wrong_shape_model.shared.weight.is_initialized());
    }

    #[test]
    fn execution_failure_preserves_partial_report_and_sink_cause() {
        let device = Default::default();
        let config = tiny(T5FeedForwardProjection::Relu, true);
        let contract = config.checkpoint_contract().unwrap();
        let temp = TempDir::new().unwrap();
        write_checkpoint(temp.path(), &contract, |_| {});
        let planned = config
            .plan_checkpoint(temp.path(), &LoadOptions::default())
            .unwrap();
        let mut model = T5ForConditionalGeneration::<TestBackend>::init(config, &device).unwrap();

        // Corrupt a public topology field after pass one so the architecture sink,
        // rather than metadata planning, supplies the audited failure.
        model.encoder.block[0].feed_forward.dense = None;
        let error = execute_local_load_plan(planned.manifest(), planned.load_plan(), &mut model)
            .unwrap_err();
        assert!(matches!(
            error.cause,
            ExecutionFailureCause::Sink(T5LoadError::MissingComponent { .. })
        ));
        assert!(error.failed_target.is_some());
        assert!(!error.report.applied.is_empty());
        assert!(error.report.applied.len() < planned.load_plan().parameter_count());
    }

    #[test]
    fn checkpoint_error_helpers_expose_only_runtime_partial_reports() {
        let checkpoint = T5CheckpointLoadError::Checkpoint(CheckpointError::InvalidMapping {
            message: "test".into(),
        });
        assert!(checkpoint.execution_failure().is_none());
    }

    #[test]
    fn synthetic_writer_uses_expected_float_storage_size() {
        assert_eq!(TensorDType::F32.storage_bytes(13 * 4).unwrap(), 208);
        let temp = TempDir::new().unwrap();
        fs::write(temp.path().join("sentinel"), b"ok").unwrap();
        assert_eq!(fs::read(temp.path().join("sentinel")).unwrap(), b"ok");
    }

    /// Loads the pinned official FLAN-T5-small artifact and executes real weights.
    ///
    /// The test is ignored because it requires a 308 MB artifact outside Git and
    /// allocates the complete F32 model. Set `TRANSFORMERS_FLAN_T5_SMALL_DIR` when
    /// the repository's `models` symlink is unavailable from a worktree.
    #[test]
    #[ignore = "requires pinned google/flan-t5-small SafeTensors artifacts"]
    fn pinned_flan_t5_small_loads_strictly_and_runs_forward() {
        let root = std::env::var_os("TRANSFORMERS_FLAN_T5_SMALL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("models/google/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab")
            });
        let config = T5Config::from_json_str(
            &fs::read_to_string(root.join("config.json"))
                .unwrap_or_else(|error| panic!("cannot read pinned FLAN config: {error}")),
        )
        .unwrap();
        assert_eq!(config.feed_forward_proj, T5FeedForwardProjection::GatedGelu);
        assert!(!config.tie_word_embeddings);

        let plan = config
            .plan_checkpoint(&root, &LoadOptions::default())
            .unwrap();
        assert_eq!(plan.manifest().tensors.len(), 190);
        assert_eq!(plan.load_plan().parameter_count(), 190);
        assert!(plan.load_plan().report.allowlisted_ignored.is_empty());
        assert_eq!(plan.load_plan().memory.checkpoint_tensor_bytes, 307_844_608);

        let device = Default::default();
        let mut model = T5ForConditionalGeneration::<TestBackend>::init(config, &device).unwrap();
        let outcome = model
            .load_checkpoint(&root, &LoadOptions::default())
            .unwrap();
        assert_eq!(outcome.report.applied.len(), 190);
        assert_eq!(outcome.stats.source.shard_opens, 1);
        assert_eq!(outcome.stats.source.max_open_shards, 1);

        let output = model
            .forward(
                Seq2SeqInput {
                    encoder: CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[71, 8, 57]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    decoder: CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                },
                &ForwardOptions::default(),
            )
            .unwrap();
        assert_eq!(output.decoder.logits.dims(), [1, 1, 32_128]);
        assert!(
            output
                .decoder
                .logits
                .to_data()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }
}
