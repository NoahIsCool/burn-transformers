//! Burn parameter dispatch for Qwen3.5-family checkpoint plans.
//!
//! The generic checkpoint executor validates and materializes dynamic
//! [`burn::tensor::TensorData`], but a Rust model records tensor rank in each
//! concrete parameter field. This module is the deliberately architecture-owned
//! bridge between the two: it recognizes only names emitted by the Qwen
//! checkpoint contract and installs each tensor through Burn's ordinary record
//! mapper so column-layout projections retain their checkpoint transpose.

use burn::{
    module::Param,
    nn::Linear,
    tensor::{Tensor, TensorData, backend::Backend},
};
use thiserror::Error;

use crate::{
    checkpoint::{ParameterInstallError, TensorSink, install_float_parameter},
    normalization::RmsNorm,
};

use super::{Qwen3_5DecoderLayer, Qwen3_5ForCausalLm};

/// Architecture-specific failure while applying a validated Qwen load plan.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Qwen3_5LoadError {
    /// A target was not part of the executable base-language-model graph.
    #[error("unknown Qwen parameter target `{target}`")]
    UnknownTarget {
        /// Unrecognized internal parameter name.
        target: String,
    },
    /// A layer component disagreed with the configured hybrid schedule.
    #[error("Qwen target `{target}` requires missing component `{component}`")]
    MissingComponent {
        /// Internal parameter being installed.
        target: String,
        /// Expected model field or mixer kind.
        component: &'static str,
    },
    /// A layer target did not contain a valid in-range decimal index.
    #[error("invalid Qwen layer target `{target}`: {reason}")]
    InvalidLayer {
        /// Malformed or out-of-range target.
        target: String,
        /// Contextual index diagnostic.
        reason: String,
    },
    /// Tensor metadata disagreed with the statically ranked Burn field.
    #[error("cannot install Qwen parameter `{target}`: {source}")]
    Parameter {
        /// Internal parameter being installed.
        target: String,
        /// Rank, shape, or element-kind failure.
        #[source]
        source: ParameterInstallError,
    },
    /// A tied-parameter operation did not match Qwen's embedding/head contract.
    #[error("invalid Qwen parameter alias `{target}` -> `{source_target}`: {reason}")]
    InvalidAlias {
        /// Alias target requested by the plan.
        target: String,
        /// Storage-owning source requested by the plan.
        source_target: String,
        /// Human-readable contract violation.
        reason: String,
    },
}

impl<B: Backend> TensorSink for Qwen3_5ForCausalLm<B> {
    type Error = Qwen3_5LoadError;

    /// Installs one storage-owning base-model tensor by its contract target.
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
                        .ok_or_else(|| Qwen3_5LoadError::MissingComponent {
                            target: target.into(),
                            component: "independent lm_head for an untied checkpoint",
                        })?;
                install_linear_weight(target, head, data)
            }
            _ => {
                let (index, suffix) = parse_layer_target(target)?;
                let layer_count = self.model.layers.len();
                let layer = self.model.layers.get_mut(index).ok_or_else(|| {
                    Qwen3_5LoadError::InvalidLayer {
                        target: target.into(),
                        reason: format!("index {index} is outside 0..{layer_count}"),
                    }
                })?;
                install_layer_tensor(layer, target, suffix, data)
            }
        }
    }

    /// Validates the one storage alias supported by tied Qwen checkpoints.
    ///
    /// A tied model has no second `Param`: its forward path projects with the
    /// embedding table itself. Accepting the alias therefore records the exact
    /// storage relationship without creating a duplicate or differently mapped
    /// matrix.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        if target != "lm_head.weight" || source_target != "model.embed_tokens.weight" {
            return Err(Qwen3_5LoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "only the tied embedding-to-LM-head alias is supported".into(),
            });
        }
        if !self.model.config.tie_word_embeddings || self.lm_head.is_some() {
            return Err(Qwen3_5LoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the configured checkpoint declares an independent LM head".into(),
            });
        }
        if !self.model.embed_tokens.weight.is_initialized() {
            return Err(Qwen3_5LoadError::InvalidAlias {
                target: target.into(),
                source_target: source_target.into(),
                reason: "the embedding storage owner has not been installed".into(),
            });
        }
        Ok(())
    }
}

/// Parses `model.layers.<decimal>.<suffix>` without accepting empty segments.
fn parse_layer_target(target: &str) -> Result<(usize, &str), Qwen3_5LoadError> {
    let remainder =
        target
            .strip_prefix("model.layers.")
            .ok_or_else(|| Qwen3_5LoadError::UnknownTarget {
                target: target.into(),
            })?;
    let (index, suffix) =
        remainder
            .split_once('.')
            .ok_or_else(|| Qwen3_5LoadError::InvalidLayer {
                target: target.into(),
                reason: "expected `model.layers.<index>.<parameter>`".into(),
            })?;
    if suffix.is_empty() {
        return Err(Qwen3_5LoadError::InvalidLayer {
            target: target.into(),
            reason: "parameter suffix must not be empty".into(),
        });
    }
    let index = index
        .parse::<usize>()
        .map_err(|error| Qwen3_5LoadError::InvalidLayer {
            target: target.into(),
            reason: format!("`{index}` is not a decimal usize: {error}"),
        })?;
    Ok((index, suffix))
}

/// Dispatches a target within one already-resolved hybrid decoder layer.
fn install_layer_tensor<B: Backend>(
    layer: &mut Qwen3_5DecoderLayer<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), Qwen3_5LoadError> {
    match suffix {
        "input_layernorm.weight" => install_norm(target, &mut layer.input_layernorm, data),
        "post_attention_layernorm.weight" => {
            install_norm(target, &mut layer.post_attention_layernorm, data)
        }
        "mlp.gate_proj.weight" => install_linear_weight(target, &mut layer.mlp.gate_proj, data),
        "mlp.up_proj.weight" => install_linear_weight(target, &mut layer.mlp.up_proj, data),
        "mlp.down_proj.weight" => install_linear_weight(target, &mut layer.mlp.down_proj, data),
        suffix if suffix.starts_with("self_attn.") => {
            let attention =
                layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Qwen3_5LoadError::MissingComponent {
                        target: target.into(),
                        component: "full-attention mixer",
                    })?;
            match suffix {
                "self_attn.q_proj.weight" => {
                    install_linear_weight(target, &mut attention.q_proj, data)
                }
                "self_attn.k_proj.weight" => {
                    install_linear_weight(target, &mut attention.k_proj, data)
                }
                "self_attn.v_proj.weight" => {
                    install_linear_weight(target, &mut attention.v_proj, data)
                }
                "self_attn.o_proj.weight" => {
                    install_linear_weight(target, &mut attention.o_proj, data)
                }
                "self_attn.q_proj.bias" => install_linear_bias(target, &mut attention.q_proj, data),
                "self_attn.k_proj.bias" => install_linear_bias(target, &mut attention.k_proj, data),
                "self_attn.v_proj.bias" => install_linear_bias(target, &mut attention.v_proj, data),
                "self_attn.o_proj.bias" => install_linear_bias(target, &mut attention.o_proj, data),
                "self_attn.q_norm.weight" => {
                    let norm = attention.query_norm.as_mut().ok_or_else(|| {
                        Qwen3_5LoadError::MissingComponent {
                            target: target.into(),
                            component: "query RMSNorm",
                        }
                    })?;
                    install_norm(target, norm, data)
                }
                "self_attn.k_norm.weight" => {
                    let norm = attention.key_norm.as_mut().ok_or_else(|| {
                        Qwen3_5LoadError::MissingComponent {
                            target: target.into(),
                            component: "key RMSNorm",
                        }
                    })?;
                    install_norm(target, norm, data)
                }
                _ => Err(Qwen3_5LoadError::UnknownTarget {
                    target: target.into(),
                }),
            }
        }
        suffix if suffix.starts_with("linear_attn.") => {
            let delta =
                layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Qwen3_5LoadError::MissingComponent {
                        target: target.into(),
                        component: "Gated DeltaNet mixer",
                    })?;
            match suffix {
                "linear_attn.in_proj_qkv.weight" => {
                    install_linear_weight(target, &mut delta.in_proj_qkv, data)
                }
                "linear_attn.in_proj_z.weight" => {
                    install_linear_weight(target, &mut delta.in_proj_z, data)
                }
                "linear_attn.in_proj_b.weight" => {
                    install_linear_weight(target, &mut delta.in_proj_b, data)
                }
                "linear_attn.in_proj_a.weight" => {
                    install_linear_weight(target, &mut delta.in_proj_a, data)
                }
                "linear_attn.conv1d.weight" => install(target, &mut delta.conv1d.weight, data),
                "linear_attn.dt_bias" => install(target, &mut delta.dt_bias, data),
                "linear_attn.A_log" => install(target, &mut delta.a_log, data),
                "linear_attn.norm.weight" => install_norm(target, &mut delta.norm, data),
                "linear_attn.out_proj.weight" => {
                    install_linear_weight(target, &mut delta.out_proj, data)
                }
                _ => Err(Qwen3_5LoadError::UnknownTarget {
                    target: target.into(),
                }),
            }
        }
        _ => Err(Qwen3_5LoadError::UnknownTarget {
            target: target.into(),
        }),
    }
}

/// Installs the weight of a Burn linear layer through its layout mapper.
fn install_linear_weight<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), Qwen3_5LoadError> {
    install(target, &mut linear.weight, data)
}

/// Installs a configured Burn linear bias or reports a topology disagreement.
fn install_linear_bias<B: Backend>(
    target: &str,
    linear: &mut Linear<B>,
    data: TensorData,
) -> Result<(), Qwen3_5LoadError> {
    let bias = linear
        .bias
        .as_mut()
        .ok_or_else(|| Qwen3_5LoadError::MissingComponent {
            target: target.into(),
            component: "linear projection bias",
        })?;
    install(target, bias, data)
}

/// Installs one RMSNorm scale with its checkpoint convention unchanged.
fn install_norm<B: Backend>(
    target: &str,
    norm: &mut RmsNorm<B>,
    data: TensorData,
) -> Result<(), Qwen3_5LoadError> {
    install(target, &mut norm.weight, data)
}

/// Validates and installs one statically ranked floating-point parameter.
fn install<B: Backend, const D: usize>(
    target: &str,
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> Result<(), Qwen3_5LoadError> {
    install_float_parameter(parameter, data).map_err(|source| Qwen3_5LoadError::Parameter {
        target: target.into(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        tensor::{Int, Tensor, TensorData},
    };

    use crate::{
        ForwardOptions, architecture::CausalLmInput, checkpoint::TensorSink,
        models::config::ModelConfig,
    };

    use super::*;
    use crate::models::qwen3_5::{
        Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5TextConfig,
    };

    type TestBackend = Flex<f32, i32>;

    /// Loads every base-graph target from the synthetic contract and runs forward.
    #[test]
    fn tiny_contract_dispatches_every_executable_parameter() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/qwen3_5_checkpoint/tiny_text_inventory.json"
        ))
        .unwrap();
        let config =
            Qwen3_5TextConfig::from_json_str(&fixture.get("config").unwrap().to_string()).unwrap();
        let contract = config
            .checkpoint_contract(Qwen3_5CheckpointOptions {
                source: Qwen3_5CheckpointSource::TextOnly,
                include_mtp: false,
            })
            .unwrap();
        let device = Default::default();
        let mut model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();

        for expected in contract.expected_tensors() {
            if expected.name == "lm_head.weight" {
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
        model
            .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
            .unwrap();

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
        assert_eq!(output.logits.dims(), [1, 1, 32]);
        assert!(
            output
                .logits
                .to_data()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    /// Unknown, malformed, schedule-mismatched, and premature aliases are typed.
    #[test]
    fn invalid_dispatch_paths_are_contextual_and_non_panicking() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/qwen3_5_checkpoint/tiny_text_inventory.json"
        ))
        .unwrap();
        let config =
            Qwen3_5TextConfig::from_json_str(&fixture.get("config").unwrap().to_string()).unwrap();
        let device = Default::default();
        let mut model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();

        let alias = model
            .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
            .unwrap_err();
        assert!(matches!(alias, Qwen3_5LoadError::InvalidAlias { .. }));

        let malformed = model
            .write_tensor(
                "model.layers.not-an-index.weight",
                TensorData::from([0.0_f32]),
            )
            .unwrap_err();
        assert!(matches!(malformed, Qwen3_5LoadError::InvalidLayer { .. }));

        let wrong_mixer = model
            .write_tensor(
                "model.layers.0.self_attn.q_proj.weight",
                TensorData::zeros::<f32, _>([16, 8]),
            )
            .unwrap_err();
        assert!(matches!(
            wrong_mixer,
            Qwen3_5LoadError::MissingComponent { .. }
        ));

        let unknown = model
            .write_tensor("mtp.fc.weight", TensorData::zeros::<f32, _>([8, 16]))
            .unwrap_err();
        assert!(matches!(unknown, Qwen3_5LoadError::UnknownTarget { .. }));
    }
}
