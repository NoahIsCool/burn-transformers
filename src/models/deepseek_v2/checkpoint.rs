//! Strict checkpoint inventory for DeepSeek-V2 and V2-Lite.
//!
//! SafeTensors matrices are stored as `[output, input]`; Burn's column-layout
//! linear mapper consumes that checkpoint orientation directly. The contract
//! keeps every expert independent, maps the single widened shared expert to its
//! own three matrices, and never treats expert-bank tensors as fused storage.
//! `noaux_tc` routers additionally require the gate's F32
//! `e_score_correction_bias` (`[n_routed_experts]`), which selects experts
//! but never weights them.

use serde::{Deserialize, Serialize};

use crate::checkpoint::{
    CheckpointManifest, ExpectedTensor, LoadOptions, LoadPlan, MappingPlan, build_load_plan,
};

use super::{ConfigError, DeepseekTopKMethod, DeepseekV2Config, Validate};
use crate::models::config::{MoeInventory, moe_inventory};

/// Family identifier retained in checked checkpoint-dimension errors.
const FAMILY: &str = "deepseek_v2";

/// Reserved explicit policy for future wrapper namespaces.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV2CheckpointOptions {}

/// Complete expected tensor inventory and source-to-field mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepseekV2CheckpointContract {
    /// Required tensor names and shapes in deterministic topology order.
    expected: Vec<ExpectedTensor>,
    /// Checkpoint-to-runtime target mappings, including the optional tied-head alias.
    mappings: MappingPlan,
}

impl DeepseekV2CheckpointContract {
    /// Returns required model parameters in deterministic topology order.
    #[must_use]
    pub fn expected_tensors(&self) -> &[ExpectedTensor] {
        &self.expected
    }

    /// Returns the exact checkpoint-key mapping.
    #[must_use]
    pub const fn mappings(&self) -> &MappingPlan {
        &self.mappings
    }

    /// Strictly compares a discovered manifest with the complete contract.
    pub fn build_load_plan(
        &self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
    ) -> crate::checkpoint::Result<LoadPlan> {
        build_load_plan(manifest, &self.expected, &self.mappings, options)
    }
}

impl DeepseekV2Config {
    /// Returns the exact length of [`Self::checkpoint_contract`]'s inventory, or `None` on overflow.
    ///
    /// With `L` layers of which `S` are sparse, `E` routed experts, and `b = 1`
    /// when `attention_bias` (else `0`), each layer lists two norms, a query
    /// path (`1` tensor without `q_lora_rank`, `3 + b` with it), the latent
    /// key/value path and output projection (`4 + 2b`), and either a router
    /// (`1`, plus the `noaux_tc` correction bias), `3E` routed and `3` shared
    /// expert matrices, or a `3`-matrix dense MLP; the embedding, final norm,
    /// and LM head add `3`. Validation bounds the total by
    /// [`MAX_CHECKPOINT_TENSORS`](crate::models::config::MAX_CHECKPOINT_TENSORS)
    /// with checked arithmetic before any contract or graph is built.
    pub(crate) fn checkpoint_tensor_count(&self) -> Option<usize> {
        let layers = self.num_hidden_layers;
        let sparse_layers = (0..layers)
            .filter(|&layer| self.is_moe_layer(layer))
            .count();
        let bias = usize::from(self.attention_bias);
        let query = if self.q_lora_rank.is_some() {
            3 + bias
        } else {
            1
        };
        let router = 1 + usize::from(self.topk_method == DeepseekTopKMethod::NoAuxTc);
        // The Verus-verified total is exact or `None` on overflow.
        moe_inventory(MoeInventory {
            fixed: 3,
            layers,
            per_layer: 2 + query + 4 + 2 * bias,
            sparse_layers,
            per_sparse_layer: router + 3,
            experts: self.n_routed_experts,
            per_expert: 3,
            per_dense_layer: 3,
        })
    }

    /// Derives the exact tensor contract from validated architecture metadata.
    pub fn checkpoint_contract(
        &self,
        _options: DeepseekV2CheckpointOptions,
    ) -> Result<DeepseekV2CheckpointContract, ConfigError> {
        self.validate()?;
        let d = Dimensions::try_from(self)?;
        // Validation proved the count fits the artifact limit.
        let mut expected = Vec::with_capacity(self.checkpoint_tensor_count().unwrap_or(0));
        let mut mappings = MappingPlan::new();
        add(
            &mut expected,
            &mut mappings,
            "model.embed_tokens.weight",
            vec![d.vocab, d.hidden],
        );
        for layer_index in 0..self.num_hidden_layers {
            let layer = format!("model.layers.{layer_index}");
            for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                add(
                    &mut expected,
                    &mut mappings,
                    &format!("{layer}.{suffix}"),
                    vec![d.hidden],
                );
            }
            let attention = format!("{layer}.self_attn");
            match d.query_rank {
                None => add(
                    &mut expected,
                    &mut mappings,
                    &format!("{attention}.q_proj.weight"),
                    vec![d.query, d.hidden],
                ),
                Some(rank) => {
                    add(
                        &mut expected,
                        &mut mappings,
                        &format!("{attention}.q_a_proj.weight"),
                        vec![rank, d.hidden],
                    );
                    if self.attention_bias {
                        add(
                            &mut expected,
                            &mut mappings,
                            &format!("{attention}.q_a_proj.bias"),
                            vec![rank],
                        );
                    }
                    add(
                        &mut expected,
                        &mut mappings,
                        &format!("{attention}.q_a_layernorm.weight"),
                        vec![rank],
                    );
                    add(
                        &mut expected,
                        &mut mappings,
                        &format!("{attention}.q_b_proj.weight"),
                        vec![d.query, rank],
                    );
                }
            }
            add(
                &mut expected,
                &mut mappings,
                &format!("{attention}.kv_a_proj_with_mqa.weight"),
                vec![d.compressed, d.hidden],
            );
            if self.attention_bias {
                add(
                    &mut expected,
                    &mut mappings,
                    &format!("{attention}.kv_a_proj_with_mqa.bias"),
                    vec![d.compressed],
                );
            }
            add(
                &mut expected,
                &mut mappings,
                &format!("{attention}.kv_a_layernorm.weight"),
                vec![d.kv_rank],
            );
            add(
                &mut expected,
                &mut mappings,
                &format!("{attention}.kv_b_proj.weight"),
                vec![d.expanded_kv, d.kv_rank],
            );
            add(
                &mut expected,
                &mut mappings,
                &format!("{attention}.o_proj.weight"),
                vec![d.hidden, d.output_heads],
            );
            if self.attention_bias {
                add(
                    &mut expected,
                    &mut mappings,
                    &format!("{attention}.o_proj.bias"),
                    vec![d.hidden],
                );
            }

            let source_mlp = format!("{layer}.mlp");
            if self.is_moe_layer(layer_index) {
                add_mapped(
                    &mut expected,
                    &mut mappings,
                    &format!("{source_mlp}.gate.weight"),
                    &format!("{layer}.sparse_mlp.routed.router.weight"),
                    vec![d.experts, d.hidden],
                );
                // `noaux_tc` routers ship their selection-only bias with the gate.
                if self.topk_method == DeepseekTopKMethod::NoAuxTc {
                    add_mapped(
                        &mut expected,
                        &mut mappings,
                        &format!("{source_mlp}.gate.e_score_correction_bias"),
                        &format!("{layer}.sparse_mlp.routed.correction_bias"),
                        vec![d.experts],
                    );
                }
                for expert in 0..self.n_routed_experts {
                    add_gated_mlp_mapped(
                        &mut expected,
                        &mut mappings,
                        &format!("{source_mlp}.experts.{expert}"),
                        &format!("{layer}.sparse_mlp.routed.routed.{expert}"),
                        d.hidden,
                        d.moe_intermediate,
                    );
                }
                add_gated_mlp_mapped(
                    &mut expected,
                    &mut mappings,
                    &format!("{source_mlp}.shared_experts"),
                    &format!("{layer}.sparse_mlp.shared_experts"),
                    d.hidden,
                    d.shared_intermediate,
                );
            } else {
                add_gated_mlp_mapped(
                    &mut expected,
                    &mut mappings,
                    &source_mlp,
                    &format!("{layer}.dense_mlp"),
                    d.hidden,
                    d.intermediate,
                );
            }
        }
        add(
            &mut expected,
            &mut mappings,
            "model.norm.weight",
            vec![d.hidden],
        );
        expected.push(ExpectedTensor::new(
            "lm_head.weight",
            vec![d.vocab, d.hidden],
        ));
        if self.tie_word_embeddings {
            mappings.alias("model.embed_tokens.weight", "lm_head.weight");
        } else {
            mappings.map("lm_head.weight", "lm_head.weight");
        }
        Ok(DeepseekV2CheckpointContract { expected, mappings })
    }
}

mod dimensions; // Verified latent-attention width arithmetic of the tensor contract.

use dimensions::{DimensionInputs, Dimensions, checkpoint_dimensions};

impl TryFrom<&DeepseekV2Config> for Dimensions {
    /// Configuration failure from a dimension sum or product.
    type Error = ConfigError;

    /// Converts all host dimensions to checked checkpoint extents with the
    /// verified [`checkpoint_dimensions`].
    fn try_from(config: &DeepseekV2Config) -> Result<Self, Self::Error> {
        checkpoint_dimensions(DimensionInputs {
            vocab: config.vocab_size,
            hidden: config.hidden_size,
            intermediate: config.intermediate_size,
            moe_intermediate: config.moe_intermediate_size,
            shared_experts: config.n_shared_experts,
            routed_experts: config.n_routed_experts,
            attention_heads: config.num_attention_heads,
            qk_nope: config.qk_nope_head_dim,
            qk_rope: config.qk_rope_head_dim,
            value_head_dim: config.v_head_dim,
            query_rank: config.q_lora_rank,
            kv_rank: config.kv_lora_rank,
        })
        .map_err(|error| ConfigError::new(FAMILY, error.field(), error.reason()))
    }
}

/// Adds the three matrices of one bias-free SwiGLU network.
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

/// Adds an identity-mapped tensor.
fn add(
    expected: &mut Vec<ExpectedTensor>,
    mappings: &mut MappingPlan,
    name: &str,
    shape: Vec<u64>,
) {
    expected.push(ExpectedTensor::new(name, shape));
    mappings.map(name, name);
}

/// Adds a source-to-field rename.
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
/// Expected-inventory and strict checkpoint-planning regressions.
mod tests;
