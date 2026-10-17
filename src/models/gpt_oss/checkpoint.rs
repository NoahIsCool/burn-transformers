//! The strict SafeTensors contract of a gpt-oss checkpoint.
//!
//! Attention, router, embedding, and head weights use the Hugging Face names
//! and row-major `[out_features, in_features]` layout, which Burn's
//! column-layout [`burn::nn::Linear`] consumes directly. The routed experts of
//! a block are stored *stacked* and input-major, exactly as the reference's
//! `GptOssExperts` parameters: the contract hands those stacked tensors to
//! the loader, which splits them into one row-layout
//! [`GatedFeedForward`](crate::feed_forward::GatedFeedForward) per expert
//! (de-interleaving gate and up columns, see the `plan` module). With
//! `d = hidden_size`, `f = intermediate_size`, `V = vocab_size`,
//! `h = head_dim`, `H` query heads, `K` key/value heads, and `E` experts:
//!
//! | target (model field) | shape | source |
//! | --- | --- | --- |
//! | `model.embed_tokens.weight` | `[V, d]` | identical |
//! | `model.layers.L.self_attn.{q,k,v,o}_proj.weight` | `[H·h, d]`, `[K·h, d]`, `[K·h, d]`, `[d, H·h]` | identical |
//! | `model.layers.L.self_attn.{q,k,v,o}_proj.bias` | `[H·h]`, `[K·h]`, `[K·h]`, `[d]` | identical, when `attention_bias` |
//! | `model.layers.L.self_attn.sinks` | `[H]` | identical |
//! | `model.layers.L.mlp.router.weight`, `.bias` | `[E, d]`, `[E]` | identical |
//! | `model.layers.L.mlp.experts.gate_up_proj` | `[E, d, 2f]` | dense: identical; MXFP4: decoded from `…gate_up_proj_blocks` `[E, 2f, d/32, 16]` and `…gate_up_proj_scales` `[E, 2f, d/32]` |
//! | `model.layers.L.mlp.experts.gate_up_proj_bias` | `[E, 2f]` | identical |
//! | `model.layers.L.mlp.experts.down_proj` | `[E, f, d]` | dense: identical; MXFP4: decoded from `…down_proj_blocks` `[E, d, f/32, 16]` and `…down_proj_scales` `[E, d, f/32]` |
//! | `model.layers.L.mlp.experts.down_proj_bias` | `[E, d]` | identical |
//! | `model.layers.L.{input,post_attention}_layernorm.weight` | `[d]` | identical |
//! | `model.norm.weight` | `[d]` | identical |
//! | `lm_head.weight` | `[V, d]` | identical, or an alias of the embedding when tied |
//!
//! MXFP4 decoding ([`MappingPlan::dequantize_mxfp4`]) reproduces Transformers'
//! `convert_moe_packed_tensors` exactly and produces the reference's dense
//! parameter, so both formats feed the loader the same tensors. The format
//! follows the configuration's `quantization_config`; a checkpoint whose
//! tensors disagree with it fails strict planning with the missing and
//! unexpected names.

use serde::{Deserialize, Serialize};

use crate::checkpoint::{
    CheckpointManifest, ExpectedTensor, LoadOptions, LoadPlan, MappingPlan, build_load_plan,
};
use crate::models::config::ConfigError;

use super::architecture::{GptOssArchitecture, GptOssWeightFormat};
use super::config::{FAMILY, GptOssConfig};
use crate::models::shared::decoder_plan::{ExtentError, checkpoint_extent, projection_extent};

/// Reserved family-specific checkpoint policy.
///
/// The released checkpoints use one namespace and the weight format is
/// declared by the configuration, so strict loading needs no options yet.
/// The value keeps future extensions additive.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GptOssCheckpointOptions {}

/// The complete expected tensor inventory and its source-to-target mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GptOssCheckpointContract {
    /// Required targets and shapes, in deterministic topology order.
    expected: Vec<ExpectedTensor>,
    /// Checkpoint-name to target mappings, including MXFP4 decodings and the tied-head alias.
    mappings: MappingPlan,
}

impl GptOssCheckpointContract {
    /// Returns every required target in topology order.
    #[must_use]
    pub fn expected_tensors(&self) -> &[ExpectedTensor] {
        &self.expected
    }

    /// Returns the exact checkpoint-name to target mapping.
    #[must_use]
    pub const fn mappings(&self) -> &MappingPlan {
        &self.mappings
    }

    /// Strictly compares a discovered manifest with the complete contract.
    ///
    /// # Errors
    ///
    /// Returns the checkpoint planning error for any mismatch.
    pub fn build_load_plan(
        &self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
    ) -> crate::checkpoint::Result<LoadPlan> {
        build_load_plan(manifest, &self.expected, &self.mappings, options)
    }
}

impl GptOssConfig {
    /// Derives the complete tensor contract from the validated topology.
    ///
    /// # Errors
    ///
    /// Returns the configuration's validation error, or a [`ConfigError`]
    /// when a dimension cannot be represented by the checkpoint format.
    pub fn checkpoint_contract(
        &self,
        _options: GptOssCheckpointOptions,
    ) -> Result<GptOssCheckpointContract, ConfigError> {
        let architecture = self.architecture()?;
        let mut contract = ContractBuilder::new(Dimensions::of(&architecture)?);
        contract.same(
            "model.embed_tokens.weight",
            vec![contract.dimensions.vocab, contract.dimensions.hidden],
        );
        for layer in 0..architecture.num_layers() {
            contract.add_layer(&architecture, layer);
        }
        contract.add_head(architecture.tie_word_embeddings);
        Ok(contract.finish())
    }
}

/// Checked shape terms of the contract, as checkpoint-format `u64` extents.
#[derive(Debug, Clone, Copy)]
struct Dimensions {
    /// Vocabulary rows `V`.
    vocab: u64,
    /// Residual width `d`.
    hidden: u64,
    /// Expert width `f`.
    intermediate: u64,
    /// Fused gate/up width `2f`.
    fused: u64,
    /// Query projection width `H·h`.
    query: u64,
    /// Key/value projection width `K·h`.
    key_value: u64,
    /// Query heads `H`.
    heads: u64,
    /// Experts `E`.
    experts: u64,
}

impl Dimensions {
    /// Converts and multiplies the architecture's extents without overflow.
    fn of(architecture: &GptOssArchitecture) -> Result<Self, ConfigError> {
        let format = |field: &'static str| {
            ConfigError::new(FAMILY, field, "dimension exceeds the checkpoint format")
        };
        let extent = |field: &'static str, value: usize| {
            checkpoint_extent(value).ok_or_else(|| format(field))
        };
        let product = |field: &'static str, heads: usize| {
            projection_extent(heads, architecture.head_dim).map_err(|error| match error {
                ExtentError::Overflow => {
                    ConfigError::new(FAMILY, field, "projection width overflows usize")
                }
                ExtentError::Format => format(field),
            })
        };
        Ok(Self {
            vocab: extent("vocab_size", architecture.vocab_size)?,
            hidden: extent("hidden_size", architecture.hidden_size)?,
            intermediate: extent("intermediate_size", architecture.intermediate_size)?,
            // The fused gate/up width `2 · f` is the same checked product as
            // a projection width: two "heads" of width `f`.
            fused: projection_extent(2, architecture.intermediate_size).map_err(
                |error| match error {
                    ExtentError::Overflow => ConfigError::new(
                        FAMILY,
                        "intermediate_size",
                        "the stacked expert projections overflow usize",
                    ),
                    ExtentError::Format => format("intermediate_size"),
                },
            )?,
            query: product("num_attention_heads", architecture.query_heads)?,
            key_value: product("num_key_value_heads", architecture.key_value_heads)?,
            heads: extent("num_attention_heads", architecture.query_heads)?,
            experts: extent("num_local_experts", architecture.experts)?,
        })
    }
}

/// Accumulates expected tensors and mappings in topology order.
struct ContractBuilder {
    /// Checked extents.
    dimensions: Dimensions,
    /// Expected targets so far.
    expected: Vec<ExpectedTensor>,
    /// Mappings so far.
    mappings: MappingPlan,
}

impl ContractBuilder {
    /// Starts an empty contract.
    fn new(dimensions: Dimensions) -> Self {
        Self {
            dimensions,
            expected: Vec::new(),
            mappings: MappingPlan::new(),
        }
    }

    /// Adds a tensor whose checkpoint name equals its target.
    fn same(&mut self, name: &str, shape: Vec<u64>) {
        self.expected.push(ExpectedTensor::new(name, shape));
        self.mappings.map(name, name);
    }

    /// Adds one decoder block: attention with sinks, the routed MLP, and two norms.
    fn add_layer(&mut self, architecture: &GptOssArchitecture, layer: usize) {
        let Dimensions {
            hidden,
            query,
            key_value,
            heads,
            ..
        } = self.dimensions;
        let prefix = format!("model.layers.{layer}");
        for (name, rows, columns) in [
            ("q_proj", query, hidden),
            ("k_proj", key_value, hidden),
            ("v_proj", key_value, hidden),
            ("o_proj", hidden, query),
        ] {
            self.same(
                &format!("{prefix}.self_attn.{name}.weight"),
                vec![rows, columns],
            );
            if architecture.attention_bias {
                self.same(&format!("{prefix}.self_attn.{name}.bias"), vec![rows]);
            }
        }
        self.same(&format!("{prefix}.self_attn.sinks"), vec![heads]);
        self.add_mlp(architecture.weight_format, &format!("{prefix}.mlp"));
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            self.same(&format!("{prefix}.{norm}.weight"), vec![hidden]);
        }
    }

    /// Adds the router and the stacked expert projections of one block.
    fn add_mlp(&mut self, format: GptOssWeightFormat, prefix: &str) {
        let Dimensions {
            hidden,
            intermediate,
            fused,
            experts,
            ..
        } = self.dimensions;
        self.same(&format!("{prefix}.router.weight"), vec![experts, hidden]);
        self.same(&format!("{prefix}.router.bias"), vec![experts]);
        for (name, shape) in [
            ("gate_up_proj", vec![experts, hidden, fused]),
            ("down_proj", vec![experts, intermediate, hidden]),
        ] {
            let target = format!("{prefix}.experts.{name}");
            self.expected.push(ExpectedTensor::new(&target, shape));
            match format {
                GptOssWeightFormat::Dense => {
                    self.mappings.map(&target, &target);
                }
                GptOssWeightFormat::Mxfp4 => {
                    self.mappings.dequantize_mxfp4(
                        format!("{target}_blocks"),
                        format!("{target}_scales"),
                        &target,
                    );
                }
            }
        }
        self.same(
            &format!("{prefix}.experts.gate_up_proj_bias"),
            vec![experts, fused],
        );
        self.same(
            &format!("{prefix}.experts.down_proj_bias"),
            vec![experts, hidden],
        );
    }

    /// Adds the final norm and the independent or tied vocabulary projection.
    fn add_head(&mut self, tied: bool) {
        let Dimensions { vocab, hidden, .. } = self.dimensions;
        self.same("model.norm.weight", vec![hidden]);
        self.expected
            .push(ExpectedTensor::new("lm_head.weight", vec![vocab, hidden]));
        if tied {
            self.mappings
                .alias("model.embed_tokens.weight", "lm_head.weight");
        } else {
            self.mappings.map("lm_head.weight", "lm_head.weight");
        }
    }

    /// Returns the finished contract.
    fn finish(self) -> GptOssCheckpointContract {
        GptOssCheckpointContract {
            expected: self.expected,
            mappings: self.mappings,
        }
    }
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
