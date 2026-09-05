//! Exact NVIDIA Nemotron-H SafeTensors inventory and field mapping.
//!
//! The pinned 3.5 Lightning checkpoint stores the base model below `backbone`,
//! an independent `lm_head`, and—because `num_nextn_predict_layers == 1`—a
//! separate `mtp.layers` body. Routed experts remain individual two-dimensional
//! tensors; no loader step stacks all 128 experts or evaluates them densely.
//!
//! Linear checkpoint tensors use `[out_features, in_features]`. Burn's column
//! [`burn::nn::Linear`] record mapper installs that data into runtime `[in, out]`
//! layout. The depthwise convolution remains
//! `[convolution_channels, 1, kernel]`. The pinned artifact stores router
//! correction buffers in FP32. Their explicit contract dtype keeps that storage
//! in FP32 even when dtype-generic projection tensors are cast to F16.

use serde::{Deserialize, Serialize};

use crate::checkpoint::{
    CheckpointManifest, ExpectedTensor, LoadOptions, LoadPlan, MappingPlan, TensorDType,
    build_load_plan,
};

use super::{ConfigError, NemotronHBlockType, NemotronHConfig, Validate};

const FAMILY: &str = "nemotron_h";

/// Reserved architecture-specific checkpoint controls.
///
/// The supported base and MTP namespaces are unambiguous, so v1 requires no
/// family exception flags. Keeping this value makes additive wrapper policies
/// possible without changing the contract-building signature.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NemotronHCheckpointOptions {}

/// Complete internal tensor inventory plus explicit source renames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemotronHCheckpointContract {
    expected: Vec<ExpectedTensor>,
    mappings: MappingPlan,
}

impl NemotronHCheckpointContract {
    /// Returns required internal parameter shapes in deterministic topology order.
    #[must_use]
    pub fn expected_tensors(&self) -> &[ExpectedTensor] {
        &self.expected
    }

    /// Returns every checkpoint-source to internal-field operation.
    #[must_use]
    pub const fn mappings(&self) -> &MappingPlan {
        &self.mappings
    }

    /// Strictly compares a discovered artifact with this complete contract.
    pub fn build_load_plan(
        &self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
    ) -> crate::checkpoint::Result<LoadPlan> {
        build_load_plan(manifest, &self.expected, &self.mappings, options)
    }
}

impl NemotronHConfig {
    /// Derives the exact base and optional MTP parameter contract.
    pub fn checkpoint_contract(
        &self,
        _options: NemotronHCheckpointOptions,
    ) -> Result<NemotronHCheckpointContract, ConfigError> {
        self.validate()?;
        let dimensions = Dimensions::try_from(self)?;
        let mut expected = Vec::new();
        let mut mappings = MappingPlan::new();
        add_mapped(
            &mut expected,
            &mut mappings,
            "backbone.embeddings.weight",
            "model.embed_tokens.weight",
            vec![dimensions.vocab, dimensions.hidden],
        );
        for (layer_index, kind) in self.layers_block_type.iter().copied().enumerate() {
            let source = format!("backbone.layers.{layer_index}");
            let target = format!("model.layers.{layer_index}");
            add_mapped(
                &mut expected,
                &mut mappings,
                &format!("{source}.norm.weight"),
                &format!("{target}.norm.weight"),
                vec![dimensions.hidden],
            );
            match kind {
                NemotronHBlockType::LinearAttention => add_mamba(
                    &mut expected,
                    &mut mappings,
                    &format!("{source}.mixer"),
                    &format!("{target}.mamba"),
                    dimensions,
                    self.use_bias,
                    self.use_conv_bias,
                ),
                NemotronHBlockType::FullAttention => add_attention(
                    &mut expected,
                    &mut mappings,
                    &format!("{source}.mixer"),
                    &format!("{target}.attention"),
                    dimensions,
                    self.attention_bias,
                ),
                NemotronHBlockType::Mlp => add_dense(
                    &mut expected,
                    &mut mappings,
                    &format!("{source}.mixer"),
                    &format!("{target}.mlp"),
                    dimensions.hidden,
                    dimensions.intermediate,
                    self.mlp_bias,
                ),
                NemotronHBlockType::Moe => add_moe(
                    &mut expected,
                    &mut mappings,
                    &format!("{source}.mixer"),
                    &format!("{target}.moe"),
                    dimensions,
                    self,
                ),
            }
        }
        add_mapped(
            &mut expected,
            &mut mappings,
            "backbone.norm_f.weight",
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
        if self.num_nextn_predict_layers > 0 {
            add_mtp(&mut expected, &mut mappings, dimensions, self)?;
        }
        Ok(NemotronHCheckpointContract { expected, mappings })
    }
}

/// Checked dimensions reused by contract sections.
#[derive(Debug, Clone, Copy)]
struct Dimensions {
    vocab: u64,
    hidden: u64,
    intermediate: u64,
    attention_query: u64,
    attention_kv: u64,
    mamba_intermediate: u64,
    mamba_convolution: u64,
    mamba_projection: u64,
    mamba_heads: u64,
    mamba_kernel: u64,
    routed_experts: u64,
    moe_intermediate: u64,
    shared_intermediate: u64,
    expert_input: u64,
}

impl TryFrom<&NemotronHConfig> for Dimensions {
    type Error = ConfigError;

    /// Converts runtime dimensions and rejects every overflowing product/sum.
    fn try_from(config: &NemotronHConfig) -> Result<Self, Self::Error> {
        let mamba_heads = dimension("mamba_num_heads", config.mamba_num_heads)?;
        let mamba_intermediate = product(
            "mamba_num_heads * mamba_head_dim",
            config.mamba_num_heads,
            config.mamba_head_dim,
        )?;
        let state_group = product(
            "n_groups * ssm_state_size",
            config.n_groups,
            config.ssm_state_size,
        )?;
        let convolution = mamba_intermediate
            .checked_add(state_group.checked_mul(2).ok_or_else(|| {
                ConfigError::new(
                    FAMILY,
                    "mamba convolution",
                    "dimension product overflows u64",
                )
            })?)
            .ok_or_else(|| {
                ConfigError::new(FAMILY, "mamba convolution", "dimension sum overflows u64")
            })?;
        let projection = mamba_intermediate
            .checked_add(convolution)
            .and_then(|value| value.checked_add(mamba_heads))
            .ok_or_else(|| {
                ConfigError::new(FAMILY, "mamba projection", "dimension sum overflows u64")
            })?;
        Ok(Self {
            vocab: dimension("vocab_size", config.vocab_size)?,
            hidden: dimension("hidden_size", config.hidden_size)?,
            intermediate: dimension("intermediate_size", config.intermediate_size)?,
            attention_query: product(
                "num_attention_heads * head_dim",
                config.num_attention_heads,
                config.head_dim,
            )?,
            attention_kv: product(
                "num_key_value_heads * head_dim",
                config.num_key_value_heads,
                config.head_dim,
            )?,
            mamba_intermediate,
            mamba_convolution: convolution,
            mamba_projection: projection,
            mamba_heads,
            mamba_kernel: dimension("conv_kernel", config.conv_kernel)?,
            routed_experts: dimension("n_routed_experts", config.n_routed_experts)?,
            moe_intermediate: dimension("moe_intermediate_size", config.moe_intermediate_size)?,
            shared_intermediate: dimension(
                "moe_shared_expert_intermediate_size",
                config.moe_shared_expert_intermediate_size,
            )?,
            expert_input: dimension(
                "moe_latent_size",
                config.moe_latent_size.unwrap_or(config.hidden_size),
            )?,
        })
    }
}

/// Adds one Mamba2 block's fused projections, convolution, scalars, and norm.
#[allow(clippy::too_many_arguments)]
fn add_mamba(
    expected: &mut Vec<ExpectedTensor>,
    mappings: &mut MappingPlan,
    source: &str,
    target: &str,
    dimensions: Dimensions,
    projection_bias: bool,
    convolution_bias: bool,
) {
    for (name, shape) in [
        (
            "in_proj.weight",
            vec![dimensions.mamba_projection, dimensions.hidden],
        ),
        (
            "conv1d.weight",
            vec![dimensions.mamba_convolution, 1, dimensions.mamba_kernel],
        ),
        ("dt_bias", vec![dimensions.mamba_heads]),
        ("A_log", vec![dimensions.mamba_heads]),
        ("D", vec![dimensions.mamba_heads]),
        ("norm.weight", vec![dimensions.mamba_intermediate]),
        (
            "out_proj.weight",
            vec![dimensions.hidden, dimensions.mamba_intermediate],
        ),
    ] {
        let internal = match name {
            "A_log" => "a_log",
            "D" => "d",
            other => other,
        };
        add_mapped(
            expected,
            mappings,
            &format!("{source}.{name}"),
            &format!("{target}.{internal}"),
            shape,
        );
    }
    if projection_bias {
        for (name, shape) in [
            ("in_proj.bias", vec![dimensions.mamba_projection]),
            ("out_proj.bias", vec![dimensions.hidden]),
        ] {
            add_mapped(
                expected,
                mappings,
                &format!("{source}.{name}"),
                &format!("{target}.{name}"),
                shape,
            );
        }
    }
    if convolution_bias {
        add_mapped(
            expected,
            mappings,
            &format!("{source}.conv1d.bias"),
            &format!("{target}.conv1d.bias"),
            vec![dimensions.mamba_convolution],
        );
    }
}

/// Adds one bias-configurable grouped-query attention block.
fn add_attention(
    expected: &mut Vec<ExpectedTensor>,
    mappings: &mut MappingPlan,
    source: &str,
    target: &str,
    dimensions: Dimensions,
    bias: bool,
) {
    for (name, shape) in [
        (
            "q_proj.weight",
            vec![dimensions.attention_query, dimensions.hidden],
        ),
        (
            "k_proj.weight",
            vec![dimensions.attention_kv, dimensions.hidden],
        ),
        (
            "v_proj.weight",
            vec![dimensions.attention_kv, dimensions.hidden],
        ),
        (
            "o_proj.weight",
            vec![dimensions.hidden, dimensions.attention_query],
        ),
    ] {
        add_mapped(
            expected,
            mappings,
            &format!("{source}.{name}"),
            &format!("{target}.{name}"),
            shape,
        );
    }
    if bias {
        for (name, shape) in [
            ("q_proj.bias", vec![dimensions.attention_query]),
            ("k_proj.bias", vec![dimensions.attention_kv]),
            ("v_proj.bias", vec![dimensions.attention_kv]),
            ("o_proj.bias", vec![dimensions.hidden]),
        ] {
            add_mapped(
                expected,
                mappings,
                &format!("{source}.{name}"),
                &format!("{target}.{name}"),
                shape,
            );
        }
    }
}

/// Adds one ordinary non-gated two-projection MLP.
fn add_dense(
    expected: &mut Vec<ExpectedTensor>,
    mappings: &mut MappingPlan,
    source: &str,
    target: &str,
    hidden: u64,
    intermediate: u64,
    bias: bool,
) {
    for (name, shape) in [
        ("up_proj.weight", vec![intermediate, hidden]),
        ("down_proj.weight", vec![hidden, intermediate]),
    ] {
        add_mapped(
            expected,
            mappings,
            &format!("{source}.{name}"),
            &format!("{target}.{name}"),
            shape,
        );
    }
    if bias {
        for (name, shape) in [
            ("up_proj.bias", vec![intermediate]),
            ("down_proj.bias", vec![hidden]),
        ] {
            add_mapped(
                expected,
                mappings,
                &format!("{source}.{name}"),
                &format!("{target}.{name}"),
                shape,
            );
        }
    }
}

/// Adds correction-biased router, independent experts, latent maps, and shared expert.
fn add_moe(
    expected: &mut Vec<ExpectedTensor>,
    mappings: &mut MappingPlan,
    source: &str,
    target: &str,
    dimensions: Dimensions,
    config: &NemotronHConfig,
) {
    add_mapped(
        expected,
        mappings,
        &format!("{source}.gate.weight"),
        &format!("{target}.router.weight"),
        vec![dimensions.routed_experts, dimensions.hidden],
    );
    let correction_source = format!("{source}.gate.e_score_correction_bias");
    let correction_target = format!("{target}.correction_bias");
    expected.push(
        ExpectedTensor::new(&correction_target, vec![dimensions.routed_experts])
            .with_dtype(TensorDType::F32),
    );
    mappings.map(correction_source, correction_target);
    for expert in 0..config.n_routed_experts {
        add_dense(
            expected,
            mappings,
            &format!("{source}.experts.{expert}"),
            &format!("{target}.routed.{expert}"),
            dimensions.expert_input,
            dimensions.moe_intermediate,
            config.mlp_bias,
        );
    }
    add_dense(
        expected,
        mappings,
        &format!("{source}.shared_experts"),
        &format!("{target}.shared"),
        dimensions.hidden,
        dimensions.shared_intermediate,
        config.mlp_bias,
    );
    if config.moe_latent_size.is_some() {
        for (name, shape) in [
            (
                "fc1_latent_proj.weight",
                vec![dimensions.expert_input, dimensions.hidden],
            ),
            (
                "fc2_latent_proj.weight",
                vec![dimensions.hidden, dimensions.expert_input],
            ),
        ] {
            add_mapped(
                expected,
                mappings,
                &format!("{source}.{name}"),
                &format!("{target}.{name}"),
                shape,
            );
        }
        if config.mlp_bias {
            for (name, shape) in [
                ("fc1_latent_proj.bias", vec![dimensions.expert_input]),
                ("fc2_latent_proj.bias", vec![dimensions.hidden]),
            ] {
                add_mapped(
                    expected,
                    mappings,
                    &format!("{source}.{name}"),
                    &format!("{target}.{name}"),
                    shape,
                );
            }
        }
    }
}

/// Adds Lightning's complete 270-tensor single-stage MTP body.
fn add_mtp(
    expected: &mut Vec<ExpectedTensor>,
    mappings: &mut MappingPlan,
    dimensions: Dimensions,
    config: &NemotronHConfig,
) -> Result<(), ConfigError> {
    if config.num_nextn_predict_layers != 1
        || config.mtp_layers_block_type
            != [NemotronHBlockType::FullAttention, NemotronHBlockType::Moe]
    {
        return Err(ConfigError::new(
            FAMILY,
            "mtp_layers_block_type",
            "v1 checkpoint loading supports one [attention, moe] Lightning MTP stage",
        ));
    }
    for (source, target) in [
        (
            "mtp.layers.0.enorm.weight",
            "next_token_head.embedding_norm.weight",
        ),
        (
            "mtp.layers.0.hnorm.weight",
            "next_token_head.hidden_norm.weight",
        ),
        (
            "mtp.layers.0.norm.weight",
            "next_token_head.attention_norm.weight",
        ),
        (
            "mtp.layers.1.norm.weight",
            "next_token_head.moe_norm.weight",
        ),
        (
            "mtp.layers.1.final_layernorm.weight",
            "next_token_head.final_layer_norm.weight",
        ),
    ] {
        add_mapped(expected, mappings, source, target, vec![dimensions.hidden]);
    }
    add_mapped(
        expected,
        mappings,
        "mtp.layers.0.eh_proj.weight",
        "next_token_head.eh_proj.weight",
        vec![dimensions.hidden, dimensions.hidden * 2],
    );
    add_attention(
        expected,
        mappings,
        "mtp.layers.0.mixer",
        "next_token_head.attention",
        dimensions,
        config.attention_bias,
    );
    add_moe(
        expected,
        mappings,
        "mtp.layers.1.mixer",
        "next_token_head.moe",
        dimensions,
        config,
    );
    Ok(())
}

/// Adds one dtype-generic source-to-target mapping.
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

/// Converts one runtime extent to SafeTensors' portable dimension type.
fn dimension(field: &'static str, value: usize) -> Result<u64, ConfigError> {
    u64::try_from(value).map_err(|_| {
        ConfigError::new(
            FAMILY,
            field,
            "dimension cannot be represented by the checkpoint format",
        )
    })
}

/// Multiplies two runtime extents without truncation or overflow.
fn product(field: &'static str, left: usize, right: usize) -> Result<u64, ConfigError> {
    left.checked_mul(right)
        .ok_or_else(|| ConfigError::new(FAMILY, field, "dimension product overflows usize"))
        .and_then(|value| dimension(field, value))
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::PathBuf};

    use crate::checkpoint::{
        ArtifactSource, CheckpointManifest, DTypePolicy, TensorDType, TensorInventory,
        TensorOperation,
    };

    use super::*;

    /// Reduced topology covering every mapping branch including MTP.
    fn tiny_config() -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 32,
            hidden_size: 8,
            layers_block_type: vec![
                NemotronHBlockType::LinearAttention,
                NemotronHBlockType::FullAttention,
                NemotronHBlockType::Mlp,
                NemotronHBlockType::Moe,
            ],
            num_hidden_layers: Some(4),
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 32,
            intermediate_size: 12,
            ssm_state_size: 3,
            mamba_num_heads: 4,
            mamba_head_dim: 2,
            n_groups: 2,
            conv_kernel: 3,
            chunk_size: 2,
            n_routed_experts: 4,
            n_shared_experts: 1,
            moe_intermediate_size: 6,
            moe_shared_expert_intermediate_size: 8,
            num_experts_per_tok: 2,
            n_group: 2,
            topk_group: 1,
            num_nextn_predict_layers: 1,
            ..Default::default()
        }
    }

    /// Constructs a complete metadata-only manifest for strict plan checks.
    fn manifest(contract: &NemotronHCheckpointContract) -> CheckpointManifest {
        manifest_with_dtype(contract, TensorDType::F32)
    }

    /// Constructs a manifest with one source dtype except for fixed-F32 buffers.
    fn manifest_with_dtype(
        contract: &NemotronHCheckpointContract,
        source_dtype: TensorDType,
    ) -> CheckpointManifest {
        let mut offset = 0;
        let mut tensors = BTreeMap::new();
        for operation in contract.mappings().operations() {
            let TensorOperation::Map { source, target, .. } = operation else {
                continue;
            };
            let expected = contract
                .expected_tensors()
                .iter()
                .find(|expected| &expected.name == target)
                .unwrap();
            let elements = expected.shape.iter().product::<u64>();
            let dtype = if target.ends_with("correction_bias") {
                TensorDType::F32
            } else {
                expected.dtype.unwrap_or(source_dtype)
            };
            let bytes = dtype.storage_bytes(elements).unwrap();
            tensors.insert(
                source.clone(),
                TensorInventory {
                    name: source.clone(),
                    shard: "synthetic.safetensors".into(),
                    shape: expected.shape.clone(),
                    dtype,
                    data_offsets: [offset, offset + bytes],
                    byte_len: bytes,
                },
            );
            offset += bytes;
        }
        CheckpointManifest {
            root: PathBuf::from("synthetic"),
            source: ArtifactSource::Single {
                relative_path: "synthetic.safetensors".into(),
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

    /// The strict plan consumes base and all 270 MTP tensor sources exactly once.
    #[test]
    fn complete_hybrid_contract_builds_strict_plan() {
        let contract = tiny_config()
            .checkpoint_contract(NemotronHCheckpointOptions::default())
            .unwrap();
        let plan = contract
            .build_load_plan(
                &manifest(&contract),
                &LoadOptions {
                    dtype_policy: DTypePolicy::Preserve,
                    ..LoadOptions::default()
                },
            )
            .unwrap();
        assert_eq!(plan.tensors.len(), contract.expected_tensors().len());
        assert!(plan.report.unexpected.is_empty());
        assert!(plan.report.missing.is_empty());
        assert!(contract.mappings().operations().iter().any(
            |operation| matches!(operation, TensorOperation::Map { source, target, .. }
                if source == "backbone.layers.0.mixer.A_log" && target.ends_with("mamba.a_log"))
        ));
    }

    /// Lightning's published 23/23/6 schedule yields its exact 6,513-key index.
    ///
    /// The fingerprint covers the sorted tensor names from NVIDIA's checkpoint
    /// index at revision `a9904d24bcc1d289a1950fa9d2b978c47cf903b9`.
    /// Keeping the compact digest here detects spelling and hierarchy drift
    /// without checking a multi-thousand-line generated fixture into Git.
    #[test]
    fn lightning_shape_contract_matches_pinned_inventory_count() {
        let mut config = tiny_config();
        config.vocab_size = 131_072;
        config.hidden_size = 2688;
        config.layers_block_type = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
            .chars()
            .map(|character| NemotronHBlockType::from_pattern(character).unwrap())
            .collect();
        config.num_hidden_layers = Some(52);
        config.num_attention_heads = 32;
        config.num_key_value_heads = 2;
        config.head_dim = 128;
        config.intermediate_size = 1856;
        config.mamba_num_heads = 64;
        config.mamba_head_dim = 64;
        config.ssm_state_size = 128;
        config.n_groups = 8;
        config.conv_kernel = 4;
        config.n_routed_experts = 128;
        config.num_experts_per_tok = 6;
        config.n_group = 1;
        config.topk_group = 1;
        config.moe_intermediate_size = 1856;
        config.moe_shared_expert_intermediate_size = 3712;
        let contract = config
            .checkpoint_contract(NemotronHCheckpointOptions::default())
            .unwrap();
        assert_eq!(contract.expected_tensors().len(), 6_513);
        let mut source_names = Vec::new();
        for operation in contract.mappings().operations() {
            match operation {
                TensorOperation::Map { source, .. } | TensorOperation::Split { source, .. } => {
                    source_names.push(source.as_str());
                }
                TensorOperation::Concatenate { sources, .. } => {
                    source_names.extend(sources.iter().map(String::as_str));
                }
                TensorOperation::Alias { .. } => {}
            }
        }
        source_names.sort_unstable();
        let fingerprint = source_names.into_iter().fold(
            14_695_981_039_346_656_037_u64,
            |mut fingerprint, source| {
                for byte in source.bytes().chain(std::iter::once(0)) {
                    fingerprint ^= u64::from(byte);
                    fingerprint = fingerprint.wrapping_mul(1_099_511_628_211);
                }
                fingerprint
            },
        );
        assert_eq!(fingerprint, 0x81d1_b113_e57a_5fbb);
        let expected = |name: &str| {
            contract
                .expected_tensors()
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap()
        };
        assert_eq!(
            expected("model.layers.0.mamba.in_proj.weight").shape,
            [10_304, 2_688]
        );
        assert_eq!(
            expected("model.layers.0.mamba.conv1d.weight").shape,
            [6_144, 1, 4]
        );
        assert_eq!(
            expected("model.layers.1.moe.router.weight").shape,
            [128, 2_688]
        );
        assert_eq!(
            expected("model.layers.1.moe.correction_bias").dtype,
            Some(TensorDType::F32)
        );
        assert_eq!(
            expected("next_token_head.eh_proj.weight").shape,
            [2_688, 5_376]
        );
    }

    /// A Metal policy casts generic weights while preserving router buffers in FP32.
    #[test]
    fn bf16_plan_preserves_fixed_fp32_router_buffers() {
        let contract = tiny_config()
            .checkpoint_contract(NemotronHCheckpointOptions::default())
            .unwrap();
        let plan = contract
            .build_load_plan(
                &manifest_with_dtype(&contract, TensorDType::BF16),
                &LoadOptions {
                    dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                    ..LoadOptions::default()
                },
            )
            .unwrap();
        for tensor in &plan.tensors {
            let expected = if tensor.target.ends_with("correction_bias") {
                TensorDType::F32
            } else {
                TensorDType::F16
            };
            assert_eq!(tensor.dtype, expected, "{}", tensor.target);
        }
        assert_eq!(
            plan.report
                .dtype_converted
                .iter()
                .filter(|conversion| conversion.target_dtype == TensorDType::F16)
                .count(),
            plan.tensors
                .iter()
                .filter(|tensor| !tensor.target.ends_with("correction_bias"))
                .count()
        );
    }

    /// Unsupported multi-stage MTP fails before constructing ambiguous mappings.
    #[test]
    fn unsupported_mtp_schedule_is_rejected() {
        let mut config = tiny_config();
        config.num_nextn_predict_layers = 2;
        assert!(
            config
                .checkpoint_contract(NemotronHCheckpointOptions::default())
                .is_err()
        );
    }
}
