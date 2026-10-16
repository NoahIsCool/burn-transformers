//! Exact Qwen3 MoE SafeTensors contract.
//!
//! Official checkpoints use direct Hugging Face names. Every projection is
//! stored row-major as `[out_features, in_features]`; Burn's column-layout
//! [`burn::nn::Linear`] consumes that layout through its record mapper. Routed
//! experts remain independent tensors named
//! `model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight`; the loader never
//! concatenates all experts into a dense matrix.

use serde::{Deserialize, Serialize};

use crate::checkpoint::{
    CheckpointManifest, ExpectedTensor, LoadOptions, LoadPlan, MappingPlan, build_load_plan,
};

use super::{ConfigError, Qwen3MoeConfig, Validate};

/// Family label attached to checkpoint geometry errors.
const FAMILY: &str = "qwen3_moe";

/// Reserved family-specific checkpoint policy.
///
/// The official Qwen3 MoE artifact has one unambiguous text namespace and no
/// optional tower, so strict loading currently needs no exceptions. Keeping an
/// options value makes future explicit wrapper support additive rather than a
/// breaking API change.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qwen3MoeCheckpointOptions {}

/// Complete expected tensor inventory and explicit source mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeCheckpointContract {
    /// Required parameter names and shapes in deterministic topology order.
    expected: Vec<ExpectedTensor>,
    /// Explicit checkpoint source-to-parameter mappings, including tied aliases.
    mappings: MappingPlan,
}

impl Qwen3MoeCheckpointContract {
    /// Returns all required model parameters in deterministic topology order.
    #[must_use]
    pub fn expected_tensors(&self) -> &[ExpectedTensor] {
        &self.expected
    }

    /// Returns the exact checkpoint-key to model-field mapping.
    #[must_use]
    pub const fn mappings(&self) -> &MappingPlan {
        &self.mappings
    }

    /// Strictly compares a discovered manifest with the complete contract.
    ///
    /// Caller-provided explicit missing/unexpected allowlists and dtype policy
    /// are honored; the family itself silently ignores no tensor.
    pub fn build_load_plan(
        &self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
    ) -> crate::checkpoint::Result<LoadPlan> {
        build_load_plan(manifest, &self.expected, &self.mappings, options)
    }
}

impl Qwen3MoeConfig {
    /// Returns the exact length of [`Self::checkpoint_contract`]'s inventory, or `None` on overflow.
    ///
    /// With `L` blocks of which `S` are sparse, `E` routed experts, and
    /// `b = 4` attention biases when `attention_bias` (else `0`), the
    /// contract lists
    ///
    /// ```text
    /// 3 + L · (8 + b) + S · (1 + 3E) + (L - S) · 3
    /// ```
    ///
    /// tensors: the embedding, final norm, and LM head; two norms, four
    /// projections, and two query/key norms per block; a router and three
    /// matrices per expert in a sparse block; three dense-MLP matrices
    /// otherwise. Validation bounds it by
    /// [`MAX_CHECKPOINT_TENSORS`](crate::models::config::MAX_CHECKPOINT_TENSORS)
    /// before any contract or graph is built, using checked arithmetic
    /// because `L` and `E` come from untrusted metadata.
    pub(crate) fn checkpoint_tensor_count(&self) -> Option<usize> {
        let layers = self.num_hidden_layers;
        let sparse = self
            .moe_layer_flags()
            .into_iter()
            .filter(|&sparse| sparse)
            .count();
        let per_layer = if self.attention_bias { 12 } else { 8 };
        let per_sparse_layer = self.num_experts.checked_mul(3)?.checked_add(1)?;
        layers
            .checked_mul(per_layer)?
            .checked_add(sparse.checked_mul(per_sparse_layer)?)?
            .checked_add((layers - sparse).checked_mul(3)?)?
            .checked_add(3)
    }

    /// Derives the full parameter contract from validated topology metadata.
    ///
    /// Validation first bounds the inventory (see
    /// [`MAX_CHECKPOINT_TENSORS`](crate::models::config::MAX_CHECKPOINT_TENSORS)),
    /// so an absurd layer or expert count is an error rather than an
    /// allocation proportional to it.
    pub fn checkpoint_contract(
        &self,
        _options: Qwen3MoeCheckpointOptions,
    ) -> Result<Qwen3MoeCheckpointContract, ConfigError> {
        self.validate()?;
        let dimensions = Dimensions::try_from(self)?;
        // Validation proved the count fits the artifact limit.
        let mut expected = Vec::with_capacity(self.checkpoint_tensor_count().unwrap_or(0));
        let mut mappings = MappingPlan::new();
        let sparse_layers = self.moe_layer_flags();
        add(
            &mut expected,
            &mut mappings,
            "model.embed_tokens.weight",
            vec![dimensions.vocab, dimensions.hidden],
        );
        for (layer_index, &sparse) in sparse_layers.iter().enumerate() {
            let layer = format!("model.layers.{layer_index}");
            add(
                &mut expected,
                &mut mappings,
                &format!("{layer}.input_layernorm.weight"),
                vec![dimensions.hidden],
            );
            add(
                &mut expected,
                &mut mappings,
                &format!("{layer}.post_attention_layernorm.weight"),
                vec![dimensions.hidden],
            );
            let attention = format!("{layer}.self_attn");
            for (name, shape) in [
                ("q_proj.weight", vec![dimensions.query, dimensions.hidden]),
                (
                    "k_proj.weight",
                    vec![dimensions.key_value, dimensions.hidden],
                ),
                (
                    "v_proj.weight",
                    vec![dimensions.key_value, dimensions.hidden],
                ),
                ("o_proj.weight", vec![dimensions.hidden, dimensions.query]),
                ("q_norm.weight", vec![dimensions.head_dim]),
                ("k_norm.weight", vec![dimensions.head_dim]),
            ] {
                add(
                    &mut expected,
                    &mut mappings,
                    &format!("{attention}.{name}"),
                    shape,
                );
            }
            if self.attention_bias {
                for (name, shape) in [
                    ("q_proj.bias", vec![dimensions.query]),
                    ("k_proj.bias", vec![dimensions.key_value]),
                    ("v_proj.bias", vec![dimensions.key_value]),
                    ("o_proj.bias", vec![dimensions.hidden]),
                ] {
                    add(
                        &mut expected,
                        &mut mappings,
                        &format!("{attention}.{name}"),
                        shape,
                    );
                }
            }
            let mlp = format!("{layer}.mlp");
            if sparse {
                add_mapped(
                    &mut expected,
                    &mut mappings,
                    &format!("{mlp}.gate.weight"),
                    &format!("{layer}.sparse_mlp.router.weight"),
                    vec![dimensions.experts, dimensions.hidden],
                );
                for expert in 0..self.num_experts {
                    let source = format!("{mlp}.experts.{expert}");
                    let target = format!("{layer}.sparse_mlp.routed.{expert}");
                    add_gated_mlp_mapped(
                        &mut expected,
                        &mut mappings,
                        &source,
                        &target,
                        dimensions.hidden,
                        dimensions.moe_intermediate,
                    );
                }
            } else {
                add_gated_mlp_mapped(
                    &mut expected,
                    &mut mappings,
                    &mlp,
                    &format!("{layer}.dense_mlp"),
                    dimensions.hidden,
                    dimensions.intermediate,
                );
            }
        }
        add(
            &mut expected,
            &mut mappings,
            "model.norm.weight",
            vec![dimensions.hidden],
        );
        expected.push(ExpectedTensor::new(
            "lm_head.weight",
            vec![dimensions.vocab, dimensions.hidden],
        ));
        if self.tie_word_embeddings {
            mappings.alias("model.embed_tokens.weight", "lm_head.weight");
        } else {
            mappings.map("lm_head.weight", "lm_head.weight");
        }
        Ok(Qwen3MoeCheckpointContract { expected, mappings })
    }
}

mod dimensions; // Verified attention-width arithmetic of the tensor contract.

use dimensions::{DimensionInputs, Dimensions, checkpoint_dimensions};

impl TryFrom<&Qwen3MoeConfig> for Dimensions {
    /// Configuration field and reason for an overflowing dimension.
    type Error = ConfigError;

    /// Converts every dimension and checks projection-width products with the
    /// verified [`checkpoint_dimensions`].
    fn try_from(config: &Qwen3MoeConfig) -> Result<Self, Self::Error> {
        checkpoint_dimensions(DimensionInputs {
            vocab: config.vocab_size,
            hidden: config.hidden_size,
            intermediate: config.intermediate_size,
            moe_intermediate: config.moe_intermediate_size,
            experts: config.num_experts,
            attention_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
        })
        .map_err(|error| {
            ConfigError::new(FAMILY, error.field(), "dimension product overflows usize")
        })
    }
}

/// Adds the three matrices of a bias-free SwiGLU network.
fn add_gated_mlp_mapped(
    expected: &mut Vec<ExpectedTensor>,
    mappings: &mut MappingPlan,
    source_prefix: &str,
    target_prefix: &str,
    hidden: u64,
    intermediate: u64,
) {
    for (suffix, shape) in [
        ("gate_proj.weight", vec![intermediate, hidden]),
        ("up_proj.weight", vec![intermediate, hidden]),
        ("down_proj.weight", vec![hidden, intermediate]),
    ] {
        add_mapped(
            expected,
            mappings,
            &format!("{source_prefix}.{suffix}"),
            &format!("{target_prefix}.{suffix}"),
            shape,
        );
    }
}

/// Adds one identity-mapped parameter.
fn add(
    expected: &mut Vec<ExpectedTensor>,
    mappings: &mut MappingPlan,
    name: &str,
    shape: Vec<u64>,
) {
    expected.push(ExpectedTensor::new(name, shape));
    mappings.map(name, name);
}

/// Adds one source-to-field rename with a statically shaped target.
fn add_mapped(
    expected: &mut Vec<ExpectedTensor>,
    mappings: &mut MappingPlan,
    source: &str,
    target: &str,
    shape: Vec<u64>,
) {
    expected.push(ExpectedTensor::new(target, shape));
    mappings.map(source, target);
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
