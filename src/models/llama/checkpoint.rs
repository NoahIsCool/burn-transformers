//! The strict SafeTensors contract of a Llama-family checkpoint.
//!
//! Every supported reference stores its weights under the same Hugging Face
//! names, row-major `[out_features, in_features]` for projections; Burn's
//! column-layout [`burn::nn::Linear`] consumes that layout directly. The
//! contract lists every tensor the executable graph owns, with its exact
//! shape, so loading fails before the first payload read on any missing,
//! unexpected, or misshapen tensor. With `d = hidden_size`,
//! `f = intermediate_size`, `V = vocab_size`, `h = head_dim`, `H` query heads
//! and `K` key/value heads:
//!
//! | checkpoint tensor | shape | present when | model field |
//! | --- | --- | --- | --- |
//! | `model.embed_tokens.weight` | `[V, d]` | always | `model.embed_tokens.weight` |
//! | `model.layers.L.self_attn.q_proj.weight` | `[H·h, d]` | always | identical |
//! | `model.layers.L.self_attn.k_proj.weight` | `[K·h, d]` | always | identical |
//! | `model.layers.L.self_attn.v_proj.weight` | `[K·h, d]` | always | identical |
//! | `model.layers.L.self_attn.o_proj.weight` | `[d, H·h]` | always | identical |
//! | `model.layers.L.self_attn.{q,k,v}_proj.bias` | `[H·h]`, `[K·h]`, `[K·h]` | query/key/value bias | identical |
//! | `model.layers.L.self_attn.o_proj.bias` | `[d]` | output bias | identical |
//! | `model.layers.L.self_attn.q_norm.weight` | `[h]` per head, `[H·h]` full width | Q/K norm | `self_attn.query_norm.weight` |
//! | `model.layers.L.self_attn.k_norm.weight` | `[h]` per head, `[K·h]` full width | Q/K norm | `self_attn.key_norm.weight` |
//! | `model.layers.L.mlp.{gate,up}_proj.weight` | `[f, d]` | always | identical |
//! | `model.layers.L.mlp.down_proj.weight` | `[d, f]` | always | identical |
//! | `model.layers.L.mlp.{gate,up}_proj.bias`, `down_proj.bias` | `[f]`, `[d]` | MLP bias | identical |
//! | `model.layers.L.input_layernorm.weight` | `[d]` | input-normalized | `model.layers.L.attention_norm.weight` |
//! | `model.layers.L.post_attention_layernorm.weight` | `[d]` | input-normalized | `model.layers.L.mlp_norm.weight` |
//! | `model.layers.L.post_attention_layernorm.weight` | `[d]` | output-normalized (OLMo 2) | `model.layers.L.attention_norm.weight` |
//! | `model.layers.L.post_feedforward_layernorm.weight` | `[d]` | output-normalized (OLMo 2) | `model.layers.L.mlp_norm.weight` |
//! | `model.norm.weight` | `[d]` | always | identical |
//! | `lm_head.weight` | `[V, d]` | untied | identical |
//! | (alias of `model.embed_tokens.weight`) | `[V, d]` | tied | `lm_head.weight` |
//!
//! The two norm rows explain the only renames: Hugging Face names a block's
//! norms by *position*, and `post_attention_layernorm` normalizes the MLP
//! input in pre-norm references but the attention output in OLMo 2. The
//! model names them by *role* (`attention_norm`, `mlp_norm`) instead. Tied
//! checkpoints do not store `lm_head.weight`; a tied checkpoint that does is
//! rejected as unexpected rather than silently overwritten.

use serde::{Deserialize, Serialize};

use crate::checkpoint::{
    CheckpointManifest, ExpectedTensor, LoadOptions, LoadPlan, MappingPlan, build_load_plan,
};
use crate::models::config::ConfigError;

use super::architecture::{LlamaArchitecture, SublayerNorm};
use super::config::LlamaFamilyConfig;
use crate::attention::QueryKeyNormScope;
use crate::models::shared::decoder_plan::{ExtentError, checkpoint_extent, projection_extent};

/// Reserved family-specific checkpoint policy.
///
/// Every supported reference publishes one unambiguous text namespace, so
/// strict loading needs no options yet. The value keeps future extensions
/// (for example fused-projection layouts) additive.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaCheckpointOptions {}

/// The complete expected tensor inventory and its source-to-field mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaCheckpointContract {
    /// Required model fields and shapes, in deterministic topology order.
    expected: Vec<ExpectedTensor>,
    /// Checkpoint-name to model-field mappings, including the tied-head alias.
    mappings: MappingPlan,
}

impl LlamaCheckpointContract {
    /// Returns every required model field in topology order.
    #[must_use]
    pub fn expected_tensors(&self) -> &[ExpectedTensor] {
        &self.expected
    }

    /// Returns the exact checkpoint-name to model-field mapping.
    #[must_use]
    pub const fn mappings(&self) -> &MappingPlan {
        &self.mappings
    }

    /// Strictly compares a discovered manifest with the complete contract.
    ///
    /// Caller-provided missing/unexpected allowlists and dtype policy are
    /// honored; the family itself ignores no tensor.
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

impl LlamaFamilyConfig {
    /// Derives the complete parameter contract from the validated topology.
    ///
    /// # Errors
    ///
    /// Returns the configuration's validation error, or a [`ConfigError`]
    /// when a dimension cannot be represented by the checkpoint format.
    pub fn checkpoint_contract(
        &self,
        _options: LlamaCheckpointOptions,
    ) -> Result<LlamaCheckpointContract, ConfigError> {
        let architecture = self.architecture()?;
        let mut contract = ContractBuilder::new(Dimensions::of(&architecture)?);
        contract.add_embedding();
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
    /// SwiGLU width `f`.
    intermediate: u64,
    /// Query projection width `H·h`.
    query: u64,
    /// Key/value projection width `K·h`.
    key_value: u64,
    /// Head width `h`.
    head: u64,
}

impl Dimensions {
    /// Converts and multiplies the architecture's extents without overflow.
    fn of(architecture: &LlamaArchitecture) -> Result<Self, ConfigError> {
        let family = architecture.variant.model_type();
        let format = |field: &'static str| {
            ConfigError::new(family, field, "dimension exceeds the checkpoint format")
        };
        let extent = |field: &'static str, value: usize| {
            checkpoint_extent(value).ok_or_else(|| format(field))
        };
        let product = |field: &'static str, heads: usize| {
            projection_extent(heads, architecture.head_dim).map_err(|error| match error {
                ExtentError::Overflow => {
                    ConfigError::new(family, field, "projection width overflows usize")
                }
                ExtentError::Format => format(field),
            })
        };
        Ok(Self {
            vocab: extent("vocab_size", architecture.vocab_size)?,
            hidden: extent("hidden_size", architecture.hidden_size)?,
            intermediate: extent("intermediate_size", architecture.intermediate_size)?,
            query: product("num_attention_heads", architecture.query_heads)?,
            key_value: product("num_key_value_heads", architecture.key_value_heads)?,
            head: extent("head_dim", architecture.head_dim)?,
        })
    }
}

/// Accumulates expected tensors and mappings in topology order.
struct ContractBuilder {
    /// Checked extents.
    dimensions: Dimensions,
    /// Expected model fields so far.
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

    /// Adds a tensor whose checkpoint name equals its model field.
    fn same(&mut self, name: &str, shape: Vec<u64>) {
        self.renamed(name, name, shape);
    }

    /// Adds a tensor stored under `source` and installed into `target`.
    fn renamed(&mut self, source: &str, target: &str, shape: Vec<u64>) {
        self.expected.push(ExpectedTensor::new(target, shape));
        self.mappings.map(source, target);
    }

    /// Adds the token embedding table.
    fn add_embedding(&mut self) {
        let Dimensions { vocab, hidden, .. } = self.dimensions;
        self.same("model.embed_tokens.weight", vec![vocab, hidden]);
    }

    /// Adds one decoder block: attention, Q/K norms, MLP, and its two norms.
    fn add_layer(&mut self, architecture: &LlamaArchitecture, layer: usize) {
        let prefix = format!("model.layers.{layer}");
        self.add_attention(architecture, &format!("{prefix}.self_attn"));
        self.add_mlp(architecture.mlp_bias, &format!("{prefix}.mlp"));
        let (attention_norm, mlp_norm) = match architecture.sublayer_norm {
            SublayerNorm::Input => ("input_layernorm", "post_attention_layernorm"),
            SublayerNorm::Output => ("post_attention_layernorm", "post_feedforward_layernorm"),
        };
        let hidden = vec![self.dimensions.hidden];
        self.renamed(
            &format!("{prefix}.{attention_norm}.weight"),
            &format!("{prefix}.attention_norm.weight"),
            hidden.clone(),
        );
        self.renamed(
            &format!("{prefix}.{mlp_norm}.weight"),
            &format!("{prefix}.mlp_norm.weight"),
            hidden,
        );
    }

    /// Adds the four projections, their enabled biases, and optional Q/K norms.
    fn add_attention(&mut self, architecture: &LlamaArchitecture, prefix: &str) {
        let Dimensions {
            hidden,
            query,
            key_value,
            head,
            ..
        } = self.dimensions;
        let bias = architecture.projection_bias;
        for (name, rows, columns, biased) in [
            ("q_proj", query, hidden, bias.query),
            ("k_proj", key_value, hidden, bias.key),
            ("v_proj", key_value, hidden, bias.value),
            ("o_proj", hidden, query, bias.output),
        ] {
            self.same(&format!("{prefix}.{name}.weight"), vec![rows, columns]);
            if biased {
                self.same(&format!("{prefix}.{name}.bias"), vec![rows]);
            }
        }
        if let Some(scope) = architecture.query_key_norm {
            let (query_width, key_width) = match scope {
                QueryKeyNormScope::PerHead => (head, head),
                QueryKeyNormScope::FullWidth => (query, key_value),
            };
            self.renamed(
                &format!("{prefix}.q_norm.weight"),
                &format!("{prefix}.query_norm.weight"),
                vec![query_width],
            );
            self.renamed(
                &format!("{prefix}.k_norm.weight"),
                &format!("{prefix}.key_norm.weight"),
                vec![key_width],
            );
        }
    }

    /// Adds the three SwiGLU projections and their enabled biases.
    fn add_mlp(&mut self, biased: bool, prefix: &str) {
        let Dimensions {
            hidden,
            intermediate,
            ..
        } = self.dimensions;
        for (name, rows, columns) in [
            ("gate_proj", intermediate, hidden),
            ("up_proj", intermediate, hidden),
            ("down_proj", hidden, intermediate),
        ] {
            self.same(&format!("{prefix}.{name}.weight"), vec![rows, columns]);
            if biased {
                self.same(&format!("{prefix}.{name}.bias"), vec![rows]);
            }
        }
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
    fn finish(self) -> LlamaCheckpointContract {
        LlamaCheckpointContract {
            expected: self.expected,
            mappings: self.mappings,
        }
    }
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
