//! Strict checkpoint inventory for DeepSeek-V2 and V2-Lite.
//!
//! SafeTensors matrices are stored as `[output, input]`; Burn's column-layout
//! linear mapper consumes that checkpoint orientation directly. The contract
//! keeps every expert independent, maps the single widened shared expert to its
//! own three matrices, and never treats expert-bank tensors as fused storage.

use serde::{Deserialize, Serialize};

use crate::checkpoint::{
    CheckpointManifest, ExpectedTensor, LoadOptions, LoadPlan, MappingPlan, build_load_plan,
};

use super::{ConfigError, DeepseekV2Config, Validate};

const FAMILY: &str = "deepseek_v2";

/// Reserved explicit policy for future wrapper namespaces.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV2CheckpointOptions {}

/// Complete expected tensor inventory and source-to-field mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepseekV2CheckpointContract {
    expected: Vec<ExpectedTensor>,
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
    /// Derives the exact tensor contract from validated architecture metadata.
    pub fn checkpoint_contract(
        &self,
        _options: DeepseekV2CheckpointOptions,
    ) -> Result<DeepseekV2CheckpointContract, ConfigError> {
        self.validate()?;
        let d = Dimensions::try_from(self)?;
        let mut expected = Vec::new();
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

/// Checked dimensions used by the contract builder.
#[derive(Debug, Clone, Copy)]
struct Dimensions {
    /// Token embedding and LM-head row count.
    vocab: u64,
    /// Residual-stream width.
    hidden: u64,
    /// Dense-prefix SwiGLU width.
    intermediate: u64,
    /// One routed expert's SwiGLU width.
    moe_intermediate: u64,
    /// Concatenated width of the checkpoint's shared-expert network.
    shared_intermediate: u64,
    /// Routed expert count.
    experts: u64,
    /// Concatenated per-head query width.
    query: u64,
    /// Optional low-rank query bottleneck width.
    query_rank: Option<u64>,
    /// Cacheable normalized-latent width.
    kv_rank: u64,
    /// Joint KV-A latent-plus-rotary output width.
    compressed: u64,
    /// Concatenated content-key and value expansion width.
    expanded_kv: u64,
    /// Concatenated value-head width consumed by the output projection.
    output_heads: u64,
}

impl TryFrom<&DeepseekV2Config> for Dimensions {
    type Error = ConfigError;

    /// Converts all host dimensions to checked checkpoint extents.
    fn try_from(config: &DeepseekV2Config) -> Result<Self, Self::Error> {
        let qk = sum(
            "qk_nope_head_dim + qk_rope_head_dim",
            config.qk_nope_head_dim,
            config.qk_rope_head_dim,
        )?;
        let kv_per_head = sum(
            "qk_nope_head_dim + v_head_dim",
            config.qk_nope_head_dim,
            config.v_head_dim,
        )?;
        Ok(Self {
            vocab: dimension("vocab_size", config.vocab_size)?,
            hidden: dimension("hidden_size", config.hidden_size)?,
            intermediate: dimension("intermediate_size", config.intermediate_size)?,
            moe_intermediate: dimension("moe_intermediate_size", config.moe_intermediate_size)?,
            shared_intermediate: product(
                "moe_intermediate_size * n_shared_experts",
                config.moe_intermediate_size,
                config.n_shared_experts,
            )?,
            experts: dimension("n_routed_experts", config.n_routed_experts)?,
            query: product(
                "num_attention_heads * qk_head_dim",
                config.num_attention_heads,
                qk,
            )?,
            query_rank: config
                .q_lora_rank
                .map(|rank| dimension("q_lora_rank", rank))
                .transpose()?,
            kv_rank: dimension("kv_lora_rank", config.kv_lora_rank)?,
            compressed: dimension(
                "kv_lora_rank + qk_rope_head_dim",
                sum(
                    "kv_lora_rank + qk_rope_head_dim",
                    config.kv_lora_rank,
                    config.qk_rope_head_dim,
                )?,
            )?,
            expanded_kv: product(
                "num_attention_heads * expanded_kv_head_dim",
                config.num_attention_heads,
                kv_per_head,
            )?,
            output_heads: product(
                "num_attention_heads * v_head_dim",
                config.num_attention_heads,
                config.v_head_dim,
            )?,
        })
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

/// Converts one dimension without truncation.
fn dimension(field: &'static str, value: usize) -> Result<u64, ConfigError> {
    u64::try_from(value).map_err(|_| {
        ConfigError::new(
            FAMILY,
            field,
            "dimension cannot be represented by the checkpoint format",
        )
    })
}

/// Adds two dimensions with overflow checking.
fn sum(field: &'static str, left: usize, right: usize) -> Result<usize, ConfigError> {
    left.checked_add(right)
        .ok_or_else(|| ConfigError::new(FAMILY, field, "dimension sum overflows usize"))
}

/// Multiplies two dimensions with overflow checking.
fn product(field: &'static str, left: usize, right: usize) -> Result<u64, ConfigError> {
    left.checked_mul(right)
        .ok_or_else(|| ConfigError::new(FAMILY, field, "dimension product overflows usize"))
        .and_then(|value| dimension(field, value))
}

#[cfg(test)]
mod tests {
    use crate::checkpoint::TensorOperation;

    use super::*;

    /// The pinned Lite topology has exactly the official 5,291 tensor entries.
    #[test]
    fn pinned_lite_inventory_count_and_shapes_are_exact() {
        let config = DeepseekV2Config::default();
        let contract = config
            .checkpoint_contract(DeepseekV2CheckpointOptions::default())
            .unwrap();
        assert_eq!(contract.expected_tensors().len(), 5_291);
        let shared = contract
            .expected_tensors()
            .iter()
            .find(|tensor| {
                tensor.name == "model.layers.1.sparse_mlp.shared_experts.gate_proj.weight"
            })
            .unwrap();
        assert_eq!(shared.shape, vec![2816, 2048]);
        let kv = contract
            .expected_tensors()
            .iter()
            .find(|tensor| tensor.name == "model.layers.0.self_attn.kv_b_proj.weight")
            .unwrap();
        assert_eq!(kv.shape, vec![4096, 512]);
    }

    /// Query-LoRA adds its three checkpoint components while direct Q uses one.
    #[test]
    fn optional_query_lora_changes_only_its_projection_contract() {
        let mut direct = DeepseekV2Config {
            num_hidden_layers: 2,
            n_routed_experts: 4,
            num_experts_per_tok: 2,
            first_k_dense_replace: 1,
            max_position_embeddings: 32,
            rope_scaling: None,
            ..Default::default()
        };
        let direct_count = direct
            .checkpoint_contract(Default::default())
            .unwrap()
            .expected_tensors()
            .len();
        direct.q_lora_rank = Some(6);
        let lora = direct.checkpoint_contract(Default::default()).unwrap();
        assert_eq!(lora.expected_tensors().len(), direct_count + 4);
        assert!(
            lora.mappings()
                .operations()
                .iter()
                .any(|operation| matches!(
                    operation,
                    TensorOperation::Map { source, target, .. }
                        if source.ends_with("q_a_layernorm.weight") && target == source
                ))
        );
    }
}
