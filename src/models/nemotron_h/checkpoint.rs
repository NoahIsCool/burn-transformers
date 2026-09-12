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

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::checkpoint::{
    CheckpointManifest, ExpectedTensor, LoadOptions, LoadPlan, MappingPlan, TensorDType,
    accounting::{
        DecimalLayerTensorSelector, PinnedAccountingTensor, PinnedControlFile,
        PinnedIndexTotalSizePolicy,
    },
    build_load_plan, build_load_plan_with_index_total_size_policy, discover_checkpoint,
    discover_checkpoint_with_index_total_size_policy,
};

use super::{ConfigError, NemotronHBlockType, NemotronHConfig, Validate};

/// Family label attached to checkpoint topology and dimension errors.
const FAMILY: &str = "nemotron_h";

/// Caller-supplied upstream identity for a local Nemotron-H checkpoint.
///
/// Labels do not establish authenticity by themselves. The sole pinned index
/// accounting policy additionally hashes exact `config.json` and shard-index
/// bytes before it accepts NVIDIA's published metadata defect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NemotronHCheckpointProvenance {
    /// Upstream repository identifier.
    pub repository: String,
    /// Immutable upstream Git or content revision.
    pub revision: String,
}

impl NemotronHCheckpointProvenance {
    /// Creates source labels that will be checked against pinned artifact hashes.
    #[must_use]
    pub fn new(repository: impl Into<String>, revision: impl Into<String>) -> Self {
        Self {
            repository: repository.into(),
            revision: revision.into(),
        }
    }
}

/// Architecture-specific checkpoint controls.
///
/// Without provenance, discovery retains generic strict equality for index
/// totals. Supplying provenance does not generally relax that rule: it merely
/// makes the exact pinned NVIDIA Lightning policy available if—and only if—the
/// index actually exhibits its authenticated 23-router overstatement.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NemotronHCheckpointOptions {
    /// Optional repository and immutable revision used by pinned artifact policy.
    #[serde(default)]
    pub provenance: Option<NemotronHCheckpointProvenance>,
}

impl NemotronHCheckpointOptions {
    /// Attaches source provenance to the otherwise strict family contract.
    #[must_use]
    pub fn with_provenance(mut self, provenance: NemotronHCheckpointProvenance) -> Self {
        self.provenance = Some(provenance);
        self
    }
}

/// Complete internal tensor inventory plus explicit source renames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemotronHCheckpointContract {
    /// Required internal parameter names, shapes, and dtype constraints.
    expected: Vec<ExpectedTensor>,
    /// Explicit source renames and parameter aliases for the complete graph.
    mappings: MappingPlan,
    /// Authenticated exception for the pinned Lightning index's router overcount.
    total_size_policy: Option<PinnedIndexTotalSizePolicy>,
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

    /// Discovers headers under this architecture's strict artifact policy.
    ///
    /// A contract without pinned provenance delegates to generic discovery.
    /// With provenance, only the exact NVIDIA Lightning source, control-file
    /// hashes, totals, and router inventory can use its known accounting quirk.
    pub fn discover_checkpoint(
        &self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> crate::checkpoint::Result<CheckpointManifest> {
        match &self.total_size_policy {
            Some(policy) => discover_checkpoint_with_index_total_size_policy(root, options, policy),
            None => discover_checkpoint(root, options),
        }
    }

    /// Strictly compares a discovered artifact with this complete contract.
    pub fn build_load_plan(
        &self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
    ) -> crate::checkpoint::Result<LoadPlan> {
        match &self.total_size_policy {
            Some(policy) => build_load_plan_with_index_total_size_policy(
                manifest,
                &self.expected,
                &self.mappings,
                options,
                policy,
            ),
            None => build_load_plan(manifest, &self.expected, &self.mappings, options),
        }
    }
}

impl NemotronHConfig {
    /// Derives the exact base and optional MTP parameter contract.
    pub fn checkpoint_contract(
        &self,
        options: NemotronHCheckpointOptions,
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
        let total_size_policy = options.provenance.as_ref().map(lightning_total_size_policy);
        Ok(NemotronHCheckpointContract {
            expected,
            mappings,
            total_size_policy,
        })
    }
}

/// Immutable NVIDIA repository whose index contains the reviewed overstatement.
const LIGHTNING_REPOSITORY: &str = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16";
/// Immutable checkpoint revision whose exact control bytes are reviewed.
const LIGHTNING_REVISION: &str = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9";
/// Exact byte length of the pinned raw `config.json`.
const LIGHTNING_CONFIG_BYTES: u64 = 2_352;
/// SHA-256 over the exact pinned raw `config.json`.
const LIGHTNING_CONFIG_SHA256: &str =
    "a3827a0f5e311547b40943dc081e3ff2f8a277466e8c1a3df2291e8db8a7617c";
/// Exact byte length of the pinned raw shard index.
const LIGHTNING_INDEX_BYTES: u64 = 612_207;
/// SHA-256 over the exact pinned raw shard index.
const LIGHTNING_INDEX_SHA256: &str =
    "67f21da80ce245a3e24967f54eef3fa10e67a63eba16ef19a2d0569f06103f50";
/// Physical tensor bytes proven by all fourteen official SafeTensors headers.
const LIGHTNING_PHYSICAL_TENSOR_BYTES: u64 = 65_826_538_624;
/// Defective total serialized by the pinned official shard index.
const LIGHTNING_DECLARED_TOTAL_SIZE: u64 = 65_842_365_568;
/// MoE layer indexes whose base router matrices were counted twice upstream.
const LIGHTNING_BASE_ROUTER_LAYERS: &[usize] = &[
    1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51,
];
/// MTP router that is physical and required but does not contribute to the defect.
const LIGHTNING_MTP_ROUTER: &str = "mtp.layers.1.mixer.gate.weight";

/// Builds the only nonstandard index-accounting policy owned by Nemotron-H.
fn lightning_total_size_policy(
    provenance: &NemotronHCheckpointProvenance,
) -> PinnedIndexTotalSizePolicy {
    let router =
        |name: String| PinnedAccountingTensor::new(name, TensorDType::BF16, vec![128, 2_688]);
    PinnedIndexTotalSizePolicy {
        supplied_repository: provenance.repository.clone(),
        supplied_revision: provenance.revision.clone(),
        expected_repository: LIGHTNING_REPOSITORY.into(),
        expected_revision: LIGHTNING_REVISION.into(),
        config: PinnedControlFile::new(
            "config.json",
            LIGHTNING_CONFIG_BYTES,
            LIGHTNING_CONFIG_SHA256,
        ),
        index: PinnedControlFile::new(
            "model.safetensors.index.json",
            LIGHTNING_INDEX_BYTES,
            LIGHTNING_INDEX_SHA256,
        ),
        expected_physical_tensor_bytes: LIGHTNING_PHYSICAL_TENSOR_BYTES,
        expected_declared_total_size: LIGHTNING_DECLARED_TOTAL_SIZE,
        duplicated_tensor_selector: DecimalLayerTensorSelector::new(
            "backbone.layers.",
            ".mixer.gate.weight",
        ),
        duplicated_tensors: LIGHTNING_BASE_ROUTER_LAYERS
            .iter()
            .map(|layer| router(format!("backbone.layers.{layer}.mixer.gate.weight")))
            .collect(),
        excluded_tensors: vec![router(LIGHTNING_MTP_ROUTER.into())],
    }
}

/// Checked dimensions reused by contract sections.
#[derive(Debug, Clone, Copy)]
struct Dimensions {
    /// Vocabulary rows in the embedding and output projection.
    vocab: u64,
    /// Residual-stream width.
    hidden: u64,
    /// Expanded width of each dense feed-forward block.
    intermediate: u64,
    /// Query projection width, `num_attention_heads * head_dim`.
    attention_query: u64,
    /// Key/value projection width, `num_key_value_heads * head_dim`.
    attention_kv: u64,
    /// Mamba value channels, `mamba_num_heads * mamba_head_dim`.
    mamba_intermediate: u64,
    /// Convolved channels: Mamba values plus both grouped B/C state projections.
    mamba_convolution: u64,
    /// Fused Mamba gate, convolution-input, and per-head timestep width.
    mamba_projection: u64,
    /// Independent Mamba heads with learned timestep, decay, and skip scalars.
    mamba_heads: u64,
    /// Temporal width of the depthwise convolution kernel.
    mamba_kernel: u64,
    /// Routed experts and router output channels.
    routed_experts: u64,
    /// Expanded width inside each routed expert.
    moe_intermediate: u64,
    /// Expanded width inside the shared expert.
    shared_intermediate: u64,
    /// Routed expert input/output width, using latent width when configured.
    expert_input: u64,
}

impl TryFrom<&NemotronHConfig> for Dimensions {
    /// Invalid or overflowing checkpoint geometry.
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
#[path = "checkpoint_tests.rs"]
mod tests;
