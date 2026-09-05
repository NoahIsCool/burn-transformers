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
    expected: Vec<ExpectedTensor>,
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
    /// Derives the full parameter contract from validated topology metadata.
    pub fn checkpoint_contract(
        &self,
        _options: Qwen3MoeCheckpointOptions,
    ) -> Result<Qwen3MoeCheckpointContract, ConfigError> {
        self.validate()?;
        let dimensions = Dimensions::try_from(self)?;
        let mut expected = Vec::new();
        let mut mappings = MappingPlan::new();
        add(
            &mut expected,
            &mut mappings,
            "model.embed_tokens.weight",
            vec![dimensions.vocab, dimensions.hidden],
        );
        for layer_index in 0..self.num_hidden_layers {
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
            if self.is_moe_layer(layer_index) {
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

/// Checked shape terms used by the contract builder.
#[derive(Debug, Clone, Copy)]
struct Dimensions {
    vocab: u64,
    hidden: u64,
    intermediate: u64,
    moe_intermediate: u64,
    experts: u64,
    query: u64,
    key_value: u64,
    head_dim: u64,
}

impl TryFrom<&Qwen3MoeConfig> for Dimensions {
    type Error = ConfigError;

    /// Converts every dimension and checks projection-width products.
    fn try_from(config: &Qwen3MoeConfig) -> Result<Self, Self::Error> {
        Ok(Self {
            vocab: dimension("vocab_size", config.vocab_size)?,
            hidden: dimension("hidden_size", config.hidden_size)?,
            intermediate: dimension("intermediate_size", config.intermediate_size)?,
            moe_intermediate: dimension("moe_intermediate_size", config.moe_intermediate_size)?,
            experts: dimension("num_experts", config.num_experts)?,
            query: product(
                "num_attention_heads * head_dim",
                config.num_attention_heads,
                config.head_dim,
            )?,
            key_value: product(
                "num_key_value_heads * head_dim",
                config.num_key_value_heads,
                config.head_dim,
            )?,
            head_dim: dimension("head_dim", config.head_dim)?,
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

/// Converts one `usize` dimension without truncation.
fn dimension(field: &'static str, value: usize) -> Result<u64, ConfigError> {
    u64::try_from(value).map_err(|_| {
        ConfigError::new(
            FAMILY,
            field,
            "dimension cannot be represented by the checkpoint format",
        )
    })
}

/// Multiplies two projection dimensions without overflow.
fn product(field: &'static str, left: usize, right: usize) -> Result<u64, ConfigError> {
    left.checked_mul(right)
        .ok_or_else(|| ConfigError::new(FAMILY, field, "dimension product overflows usize"))
        .and_then(|value| dimension(field, value))
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::PathBuf};

    use crate::{
        checkpoint::{
            ArtifactSource, CheckpointError, CheckpointManifest, DTypePolicy, LoadOptions,
            TensorDType, TensorInventory,
        },
        models::config::ModelConfig,
    };

    use super::*;

    /// Tiny topology used to exercise both sparse and dense contract branches.
    fn tiny_config() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 32,
            max_window_layers: 2,
            decoder_sparse_step: 1,
            moe_intermediate_size: 6,
            num_experts: 4,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            ..Default::default()
        }
    }

    /// Constructs a metadata-only manifest matching identity mappings.
    fn manifest(contract: &Qwen3MoeCheckpointContract, dtype: TensorDType) -> CheckpointManifest {
        let mut offset = 0;
        let mut tensors = BTreeMap::new();
        for operation in contract.mappings().operations() {
            let crate::checkpoint::TensorOperation::Map { source, target, .. } = operation else {
                continue;
            };
            let expected = contract
                .expected_tensors()
                .iter()
                .find(|expected| &expected.name == target)
                .unwrap();
            let elements = expected.shape.iter().copied().product();
            let bytes = dtype.storage_bytes(elements).unwrap();
            tensors.insert(
                source.clone(),
                TensorInventory {
                    name: source.clone(),
                    shard: "metadata.safetensors".into(),
                    shape: expected.shape.clone(),
                    dtype,
                    data_offsets: [offset, offset + bytes],
                    byte_len: bytes,
                },
            );
            offset += bytes;
        }
        CheckpointManifest {
            root: PathBuf::from("metadata"),
            source: ArtifactSource::Single {
                relative_path: "metadata.safetensors".into(),
            },
            shards: Vec::new(),
            tensors,
            index_aliases: Vec::new(),
            index_alias_size_accounting: None,
            indexed_tensor_bytes: offset,
            tensor_bytes: offset,
            artifact_bytes: offset,
        }
    }

    /// Official pinned configuration yields the exact 128-by-eight expert inventory.
    #[test]
    fn official_a3b_contract_has_expected_cardinality_and_shapes() {
        let config = Qwen3MoeConfig::from_json_str(include_str!(
            "../../../tests/fixtures/qwen3_30b_a3b.json"
        ))
        .unwrap();
        let contract = config
            .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
            .unwrap();
        assert_eq!(contract.expected_tensors().len(), 18_867);
        let find = |name: &str| {
            contract
                .expected_tensors()
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap()
                .shape
                .clone()
        };
        assert_eq!(find("model.layers.0.self_attn.q_proj.weight"), [4096, 2048]);
        assert_eq!(find("model.layers.0.self_attn.k_proj.weight"), [512, 2048]);
        assert_eq!(find("model.layers.0.sparse_mlp.router.weight"), [128, 2048]);
        assert_eq!(
            find("model.layers.47.sparse_mlp.routed.127.down_proj.weight"),
            [2048, 768]
        );
        assert!(contract.mappings().operations().iter().any(|operation| {
            matches!(
                operation,
                crate::checkpoint::TensorOperation::Map { source, target, .. }
                    if source == "model.layers.0.mlp.gate.weight"
                        && target == "model.layers.0.sparse_mlp.router.weight"
            )
        }));
    }

    /// Strict planning accepts exactly the synthetic tiny inventory.
    #[test]
    fn exact_inventory_builds_identity_plan() {
        let contract = tiny_config()
            .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
            .unwrap();
        let plan = contract
            .build_load_plan(
                &manifest(&contract, TensorDType::F32),
                &LoadOptions::default(),
            )
            .unwrap();
        assert_eq!(plan.parameter_count(), 45);
        assert!(plan.report.missing.is_empty());
        assert!(plan.report.unexpected.is_empty());
        assert!(plan.aliases.is_empty());
    }

    /// Missing, unexpected, and wrong-shaped tensors all fail closed.
    #[test]
    fn strict_contract_rejects_inventory_mismatches() {
        let contract = tiny_config()
            .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
            .unwrap();

        let mut missing = manifest(&contract, TensorDType::F32);
        missing.tensors.remove("model.layers.0.mlp.gate.weight");
        let error = contract
            .build_load_plan(&missing, &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(
            error,
            CheckpointError::MissingMappingSource { .. }
        ));

        let mut unexpected = manifest(&contract, TensorDType::F32);
        let sample = unexpected.tensors.values().next().unwrap().clone();
        unexpected.tensors.insert(
            "model.layers.0.mlp.shared_expert.weight".into(),
            TensorInventory {
                name: "model.layers.0.mlp.shared_expert.weight".into(),
                ..sample
            },
        );
        let error = contract
            .build_load_plan(&unexpected, &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(
            error,
            CheckpointError::StrictValidation {
                missing: 0,
                unexpected: 1,
                ..
            }
        ));

        let mut wrong_shape = manifest(&contract, TensorDType::F32);
        wrong_shape
            .tensors
            .get_mut("model.layers.0.mlp.gate.weight")
            .unwrap()
            .shape = vec![3, 8];
        let error = contract
            .build_load_plan(&wrong_shape, &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(error, CheckpointError::ShapeMismatch { .. }));
    }

    /// BF16 checkpoint tensors can be streamed into an F16 Metal-oriented plan.
    #[test]
    fn dtype_policy_records_bf16_to_f16_conversion() {
        let contract = tiny_config()
            .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
            .unwrap();
        let options = LoadOptions {
            dtype_policy: DTypePolicy::Cast(TensorDType::F16),
            ..Default::default()
        };
        let plan = contract
            .build_load_plan(&manifest(&contract, TensorDType::BF16), &options)
            .unwrap();
        assert!(
            plan.tensors
                .iter()
                .all(|tensor| tensor.dtype == TensorDType::F16)
        );
        assert_eq!(plan.report.dtype_converted.len(), plan.tensors.len());
    }

    /// A tied head is represented as an alias with no second source tensor.
    #[test]
    fn tied_head_contract_uses_storage_alias() {
        let mut config = tiny_config();
        config.tie_word_embeddings = true;
        let contract = config
            .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
            .unwrap();
        let plan = contract
            .build_load_plan(
                &manifest(&contract, TensorDType::F32),
                &LoadOptions::default(),
            )
            .unwrap();
        assert_eq!(plan.aliases.len(), 1);
        assert_eq!(plan.aliases[0].target, "lm_head.weight");
    }
}
