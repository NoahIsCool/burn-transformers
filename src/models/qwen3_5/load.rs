//! Burn parameter dispatch for Qwen3.5-family checkpoint plans.
//!
//! The generic checkpoint executor validates and materializes dynamic
//! [`burn::tensor::TensorData`], but a Rust model records tensor rank in each
//! concrete parameter field. This module is the deliberately architecture-owned
//! bridge between the two: it recognizes only names emitted by the Qwen
//! checkpoint contract and installs each tensor through Burn's ordinary record
//! mapper so column-layout projections retain their checkpoint transpose.
//!
//! [`Qwen3_5ForCausalLm::load_checkpoint`] is the concrete-family entry point:
//! it detects the multimodal or text-only namespace with the crate's shared
//! classifier (the one automatic loading uses), expects the optional MTP
//! tensors exactly when the model owns an MTP graph, and streams the strict
//! plan into this sink. [`Qwen3_5ForCausalLm::load_checkpoint_with`] takes
//! both choices explicitly.

use std::path::Path;

use burn::{
    module::Param,
    nn::Linear,
    tensor::{Tensor, TensorData, backend::Backend},
};
use thiserror::Error;

use crate::{
    checkpoint::{
        CheckpointError, CheckpointManifest, ExecutionFailure, ExecutionOutcome, KeyMatcher,
        LanguageNamespace, LoadOptions, PackedLayoutPolicy, ParameterInstallError, TensorSink,
        execute_local_load_plan, install_packed_parameter,
    },
    normalization::RmsNorm,
};

use super::{
    ConfigError, Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLm, Qwen3_5Mtp, Qwen3_5TextModel,
};

/// Architecture-specific failure while applying a validated Qwen load plan.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Qwen3_5LoadError {
    /// A target was not part of the executable base or optional MTP graph.
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

/// Failure from topology validation, namespace detection, strict planning, or streamed execution.
#[derive(Debug, Error)]
#[non_exhaustive]
#[expect(
    unnameable_types,
    reason = "public load methods return this type, so src/models/qwen3_5.rs must re-export it; \
              delete this expectation in the change that adds that re-export"
)]
pub enum Qwen3_5CheckpointLoadError {
    /// The model topology cannot describe a valid Qwen3.5-family checkpoint.
    #[error(transparent)]
    Config(
        /// Invalid configuration and its field-level diagnostic.
        #[from]
        ConfigError,
    ),
    /// Artifact discovery, namespace detection, or complete expected/observed validation failed.
    #[error(transparent)]
    Checkpoint(
        /// Discovery, [`CheckpointError::LanguageNamespaceNotFound`] or
        /// [`CheckpointError::AmbiguousLanguageNamespace`], or load-plan failure.
        #[from]
        CheckpointError,
    ),
    /// The model owns an MTP graph that a base-only load would leave uninitialized.
    #[error(
        "the Qwen model owns a multi-token-prediction graph that a base-only load would leave \
         uninitialized; set `include_mtp` or initialize with `mtp_num_hidden_layers = 0`"
    )]
    UnloadedMtpGraph,
    /// MTP tensors were requested for a model initialized without an MTP graph.
    #[error(
        "MTP tensors were requested, but the Qwen model has no multi-token-prediction graph; \
         initialize it with `mtp_num_hidden_layers > 0`"
    )]
    MissingMtpGraph,
    /// Pass two failed after zero or more audited sink writes.
    #[error(transparent)]
    Execution(
        /// Sink failure together with the writes and diagnostics completed so far.
        #[from]
        Box<ExecutionFailure<Qwen3_5LoadError>>,
    ),
}

impl Qwen3_5CheckpointLoadError {
    /// Returns partial pass-two diagnostics when execution reached the sink.
    #[must_use]
    pub fn execution_failure(&self) -> Option<&ExecutionFailure<Qwen3_5LoadError>> {
        match self {
            Self::Execution(failure) => Some(failure),
            Self::Config(_)
            | Self::Checkpoint(_)
            | Self::UnloadedMtpGraph
            | Self::MissingMtpGraph => None,
        }
    }
}

impl<B: Backend> Qwen3_5ForCausalLm<B> {
    /// Strictly discovers, validates, and streams a local Qwen3.5-family checkpoint.
    ///
    /// The language namespace is detected from the tensor names: an official
    /// multimodal repository (`model.language_model.*`, whose vision tower is
    /// reported as intentionally ignored) or a standalone text checkpoint. The
    /// optional multi-token-prediction tensors are expected exactly when this
    /// model owns an MTP graph, that is, when it was initialized with
    /// `mtp_num_hidden_layers > 0`. A base-only model (initialized with
    /// `mtp_num_hidden_layers = 0`) reports any `mtp.*` tensors as
    /// intentionally ignored, as automatic loading without
    /// `AutoLoadOptions::qwen_mtp` does. No network access is performed. A missing,
    /// unexpected, misshapen, or disallowed-dtype tensor fails before the first
    /// payload read. After an execution failure the partially initialized
    /// model must be discarded.
    ///
    /// # Errors
    ///
    /// Returns a `Qwen3_5CheckpointLoadError` naming the failing stage.
    pub fn load_checkpoint(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<ExecutionOutcome, Qwen3_5CheckpointLoadError> {
        let manifest = CheckpointManifest::discover(root, options)?;
        let source = match LanguageNamespace::of_manifest(&manifest, "Qwen")? {
            LanguageNamespace::Wrapped => Qwen3_5CheckpointSource::Multimodal,
            LanguageNamespace::TextOnly => Qwen3_5CheckpointSource::TextOnly,
        };
        let family_options = Qwen3_5CheckpointOptions {
            source,
            include_mtp: self.mtp.is_some(),
        };
        self.load_discovered(&manifest, options, family_options)
    }

    /// Loads a local checkpoint with an explicit namespace and MTP policy.
    ///
    /// Use this to force the `source` layout or to request MTP tensors
    /// explicitly; [`Self::load_checkpoint`] detects both.
    ///
    /// # Errors
    ///
    /// As [`Self::load_checkpoint`]; additionally, before any planning,
    /// `Qwen3_5CheckpointLoadError::UnloadedMtpGraph` when `include_mtp` is
    /// false but the model owns an MTP graph, which would stay randomly
    /// initialized and reachable through the speculative API, and
    /// `Qwen3_5CheckpointLoadError::MissingMtpGraph` when `include_mtp` is
    /// true but the model has no MTP graph to receive the tensors.
    pub fn load_checkpoint_with(
        &mut self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
        family_options: Qwen3_5CheckpointOptions,
    ) -> Result<ExecutionOutcome, Qwen3_5CheckpointLoadError> {
        let manifest = CheckpointManifest::discover(root, options)?;
        self.load_discovered(&manifest, options, family_options)
    }

    /// Plans a discovered checkpoint under `family_options` and streams it into this model.
    fn load_discovered(
        &mut self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
        family_options: Qwen3_5CheckpointOptions,
    ) -> Result<ExecutionOutcome, Qwen3_5CheckpointLoadError> {
        match (self.mtp.is_some(), family_options.include_mtp) {
            (true, false) => return Err(Qwen3_5CheckpointLoadError::UnloadedMtpGraph),
            (false, true) => return Err(Qwen3_5CheckpointLoadError::MissingMtpGraph),
            _ => {}
        }
        let mut options = options.clone();
        if self.mtp.is_none() {
            // A base-only graph ignores the optional MTP subsystem exactly as
            // the contract's own exemption does for a configuration that
            // declares MTP layers: `mtp.*` is reported as intentionally
            // ignored, never as unexpected and never half-loaded.
            let mtp = KeyMatcher::prefix("mtp.");
            if !options.allow_unexpected.contains(&mtp) {
                options.allow_unexpected.push(mtp);
            }
        }
        let contract = self.config().checkpoint_contract(family_options)?;
        let plan = contract.build_load_plan(manifest, &options)?;
        Ok(execute_local_load_plan(manifest, &plan, self)?)
    }
}

impl<B: Backend> TensorSink for Qwen3_5TextModel<B> {
    /// Contextual parameter installation or alias validation failure.
    type Error = Qwen3_5LoadError;

    /// Installs a headless-backbone tensor after resolving its exact layer and mixer field.
    ///
    /// The shared installer validates rank, shape, and dtype before replacing
    /// the parameter and preserves Burn record transformations.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        match target {
            "model.embed_tokens.weight" => install(target, &mut self.embed_tokens.weight, data),
            "model.norm.weight" => install_norm(target, &mut self.norm, data),
            _ => {
                let (index, suffix) = parse_layer_target(target)?;
                let count = self.layers.len();
                let layer =
                    self.layers
                        .get_mut(index)
                        .ok_or_else(|| Qwen3_5LoadError::InvalidLayer {
                            target: target.into(),
                            reason: format!("index {index} is outside 0..{count}"),
                        })?;
                install_layer_tensor(layer, target, suffix, data)
            }
        }
    }

    /// Rejects aliases because a headless backbone owns only its embedding parameter.
    ///
    /// A tied vocabulary projection is an explicit reuse of that owner in the
    /// forward graph, not a second registered parameter field.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        Err(Qwen3_5LoadError::InvalidAlias { target: target.into(), source_target: source_target.into(),
            reason: "the headless backbone stores only the embedding owner; vocabulary views are explicit execution operations".into() })
    }
}

impl<B: Backend> TensorSink for Qwen3_5ForCausalLm<B> {
    /// Contextual parameter installation or alias validation failure.
    type Error = Qwen3_5LoadError;

    /// Installs one storage-owning base-model or MTP tensor by its contract target.
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
                if let Some(suffix) = target.strip_prefix("mtp.") {
                    let mtp =
                        self.mtp
                            .as_mut()
                            .ok_or_else(|| Qwen3_5LoadError::MissingComponent {
                                target: target.into(),
                                component: "configured multi-token-prediction graph",
                            })?;
                    return install_mtp_tensor(mtp, target, suffix, data);
                }
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

/// Dispatches one of the official `mtp.*` tensors into its concrete Burn field.
///
/// `suffix` is `target` without its `mtp.` namespace; `target` is kept whole
/// for diagnostics.
fn install_mtp_tensor<B: Backend>(
    mtp: &mut Qwen3_5Mtp<B>,
    target: &str,
    suffix: &str,
    data: TensorData,
) -> Result<(), Qwen3_5LoadError> {
    match suffix {
        "fc.weight" => install_linear_weight(target, &mut mtp.fc, data),
        "norm.weight" => install_norm(target, &mut mtp.norm, data),
        "pre_fc_norm_embedding.weight" => {
            install_norm(target, &mut mtp.pre_fc_norm_embedding, data)
        }
        "pre_fc_norm_hidden.weight" => install_norm(target, &mut mtp.pre_fc_norm_hidden, data),
        _ => {
            let (index, layer_suffix) = parse_indexed_target(target, suffix, "layers.", "mtp")?;
            let count = mtp.layers.len();
            let layer =
                mtp.layers
                    .get_mut(index)
                    .ok_or_else(|| Qwen3_5LoadError::InvalidLayer {
                        target: target.into(),
                        reason: format!("MTP index {index} is outside 0..{count}"),
                    })?;
            install_layer_tensor(layer, target, layer_suffix, data)
        }
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

/// Parses an indexed namespace after its architecture-specific prefix.
fn parse_indexed_target<'a>(
    target: &str,
    suffix: &'a str,
    indexed_prefix: &str,
    namespace: &str,
) -> Result<(usize, &'a str), Qwen3_5LoadError> {
    let remainder =
        suffix
            .strip_prefix(indexed_prefix)
            .ok_or_else(|| Qwen3_5LoadError::UnknownTarget {
                target: target.into(),
            })?;
    let (index, parameter) =
        remainder
            .split_once('.')
            .ok_or_else(|| Qwen3_5LoadError::InvalidLayer {
                target: target.into(),
                reason: format!("expected `{namespace}.layers.<index>.<parameter>`"),
            })?;
    if parameter.is_empty() {
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
    Ok((index, parameter))
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

/// Validates and installs one statically ranked parameter.
///
/// Float data is installed as is. Packed (QFloat) data stays packed when the
/// backend reproduces the artifact's blocks exactly through the parameter's
/// load mapper (for example Metal with column-major linear layouts);
/// otherwise its exact dequantized values are installed instead of a
/// requantized approximation, so a model's numerics never depend on the
/// backend's packed layout support.
fn install<B: Backend, const D: usize>(
    target: &str,
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> Result<(), Qwen3_5LoadError> {
    install_packed_parameter(parameter, data, PackedLayoutPolicy::DequantizeWhenInexact)
        .map(|_| ())
        .map_err(|source| Qwen3_5LoadError::Parameter {
            target: target.into(),
            source,
        })
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
