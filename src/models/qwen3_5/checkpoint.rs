//! Exact Qwen3.5/Qwen3.8 checkpoint tensor contract.
//!
//! Hugging Face's conditional-generation repositories wrap the text transformer
//! below `model.language_model`, while text-only causal-LM repositories place the
//! same parameters below `model`. This module makes that prefix distinction data
//! rather than loader heuristics. It also preserves Qwen's head-local gated-query
//! row layout and records every DeltaNet/MTP shape implied by the model config.

use serde::{Deserialize, Serialize};

use crate::checkpoint::{
    CheckpointManifest, ExpectedTensor, KeyMatcher, LoadOptions, LoadPlan, MappingPlan,
    TensorTransform, build_load_plan,
};

use super::{ConfigError, Qwen3_5LayerType, Qwen3_5TextConfig, Validate};

const FAMILY: &str = "qwen3_5";

/// Serialized namespace used by a Qwen3.5-family checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen3_5CheckpointSource {
    /// `Qwen3_5ForConditionalGeneration`: text lives below `model.language_model`
    /// and `model.visual.*` belongs to the intentionally skipped vision tower.
    Multimodal,
    /// `Qwen3_5ForCausalLM`: the language backbone lives directly below `model`.
    TextOnly,
}

impl Qwen3_5CheckpointSource {
    fn language_prefix(self) -> &'static str {
        match self {
            Self::Multimodal => "model.language_model",
            Self::TextOnly => "model",
        }
    }
}

/// Policy controlling which family-owned checkpoint tensors become model state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qwen3_5CheckpointOptions {
    /// Namespace layout exposed by the local checkpoint.
    pub source: Qwen3_5CheckpointSource,
    /// Include declared multi-token-prediction tensors in the expected inventory.
    ///
    /// This must be enabled only when the receiving model graph instantiates MTP
    /// parameters. The current base causal-LM graph does not, so the safe default
    /// reports them as an explicitly ignored optional subsystem.
    pub include_mtp: bool,
}

impl Default for Qwen3_5CheckpointOptions {
    /// Targets official multimodal repositories and the executable base LM only.
    fn default() -> Self {
        Self {
            source: Qwen3_5CheckpointSource::Multimodal,
            include_mtp: false,
        }
    }
}

/// Complete logical inventory, source mapping, and intentional-key exceptions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3_5CheckpointContract {
    expected: Vec<ExpectedTensor>,
    mappings: MappingPlan,
    allow_unexpected: Vec<KeyMatcher>,
}

impl Qwen3_5CheckpointContract {
    /// Returns the complete internal parameter inventory in topology order.
    #[must_use]
    pub fn expected_tensors(&self) -> &[ExpectedTensor] {
        &self.expected
    }

    /// Returns explicit source-to-target operations in topology order.
    #[must_use]
    pub fn mappings(&self) -> &MappingPlan {
        &self.mappings
    }

    /// Returns checkpoint keys intentionally outside the selected text graph.
    ///
    /// A multimodal source always contains the `model.visual.*` matcher. If MTP
    /// loading is disabled, `mtp.*` is present as a separate matcher so reports
    /// still distinguish optional prediction weights from unknown artifacts.
    #[must_use]
    pub fn allow_unexpected(&self) -> &[KeyMatcher] {
        &self.allow_unexpected
    }

    /// Adds family-owned exceptions to caller-supplied loader policy.
    ///
    /// Existing resource bounds, dtype policy, and caller exceptions are kept.
    pub fn configure_load_options(&self, options: &mut LoadOptions) {
        for matcher in &self.allow_unexpected {
            if !options.allow_unexpected.contains(matcher) {
                options.allow_unexpected.push(matcher.clone());
            }
        }
    }

    /// Strictly validates an observed local manifest against this family contract.
    pub fn build_load_plan(
        &self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
    ) -> crate::checkpoint::Result<LoadPlan> {
        let mut options = options.clone();
        self.configure_load_options(&mut options);
        build_load_plan(manifest, &self.expected, &self.mappings, &options)
    }
}

impl Qwen3_5TextConfig {
    /// Derives every expected tensor and checkpoint mapping from validated shape
    /// metadata, without opening or allocating checkpoint payloads.
    ///
    /// Gated full-attention query weights remain in their official head-local
    /// layout. The reference projection is reshaped as `[heads, 2, head_dim]`, so
    /// rows alternate one Q block and one gate block per head. The family model's
    /// attention primitive consumes that same layout directly.
    pub fn checkpoint_contract(
        &self,
        options: Qwen3_5CheckpointOptions,
    ) -> Result<Qwen3_5CheckpointContract, ConfigError> {
        self.validate()?;
        if options.include_mtp && self.mtp_use_dedicated_embeddings {
            return Err(ConfigError::new(
                FAMILY,
                "mtp_use_dedicated_embeddings",
                "dedicated MTP embedding checkpoints have no released Qwen3.5 tensor contract",
            ));
        }

        let dimensions = Dimensions::try_from(self)?;
        let mut builder = ContractBuilder::new(options.source);
        builder.language(
            "embed_tokens.weight",
            vec![dimensions.vocab, dimensions.hidden],
        );

        for (index, layer_type) in self.resolved_layer_types().into_iter().enumerate() {
            builder.decoder_layer(index, layer_type, self, &dimensions);
        }

        builder.language("norm.weight", vec![dimensions.hidden]);
        let head_shape = vec![dimensions.vocab, dimensions.hidden];
        builder
            .expected
            .push(ExpectedTensor::new("lm_head.weight", head_shape.clone()));
        if self.tie_word_embeddings {
            builder
                .mappings
                .alias("model.embed_tokens.weight", "lm_head.weight");
        } else {
            builder.mappings.map("lm_head.weight", "lm_head.weight");
        }

        if options.include_mtp {
            builder.mtp(self, &dimensions);
        } else if self.mtp_num_hidden_layers > 0 {
            builder.allow_unexpected.push(KeyMatcher::prefix("mtp."));
        }

        Ok(builder.finish())
    }
}

#[derive(Debug, Clone, Copy)]
struct Dimensions {
    vocab: u64,
    hidden: u64,
    intermediate: u64,
    query: u64,
    gated_query: u64,
    key_value: u64,
    delta_value: u64,
    delta_conv: u64,
    double_hidden: u64,
    head_dim: u64,
    delta_value_heads: u64,
    delta_value_head_dim: u64,
    conv_kernel: u64,
}

impl TryFrom<&Qwen3_5TextConfig> for Dimensions {
    type Error = ConfigError;

    fn try_from(config: &Qwen3_5TextConfig) -> Result<Self, Self::Error> {
        let hidden = dimension("hidden_size", config.hidden_size)?;
        let query = product(
            "num_attention_heads * head_dim",
            &[config.num_attention_heads, config.head_dim],
        )?;
        let key_value = product(
            "num_key_value_heads * head_dim",
            &[config.num_key_value_heads, config.head_dim],
        )?;
        let delta_key = product(
            "linear_num_key_heads * linear_key_head_dim",
            &[config.linear_num_key_heads, config.linear_key_head_dim],
        )?;
        let delta_value = product(
            "linear_num_value_heads * linear_value_head_dim",
            &[config.linear_num_value_heads, config.linear_value_head_dim],
        )?;
        let delta_conv = checked_sum(
            "2 * delta key width + delta value width",
            &[delta_key, delta_key, delta_value],
        )?;
        let gated_query = checked_sum("2 * query width", &[query, query])?;
        let double_hidden = checked_sum("2 * hidden_size", &[hidden, hidden])?;
        Ok(Self {
            vocab: dimension("vocab_size", config.vocab_size)?,
            hidden,
            intermediate: dimension("intermediate_size", config.intermediate_size)?,
            query,
            gated_query,
            key_value,
            delta_value,
            delta_conv,
            double_hidden,
            head_dim: dimension("head_dim", config.head_dim)?,
            delta_value_heads: dimension("linear_num_value_heads", config.linear_num_value_heads)?,
            delta_value_head_dim: dimension("linear_value_head_dim", config.linear_value_head_dim)?,
            conv_kernel: dimension("linear_conv_kernel_dim", config.linear_conv_kernel_dim)?,
        })
    }
}

struct ContractBuilder {
    source: Qwen3_5CheckpointSource,
    expected: Vec<ExpectedTensor>,
    mappings: MappingPlan,
    allow_unexpected: Vec<KeyMatcher>,
}

impl ContractBuilder {
    fn new(source: Qwen3_5CheckpointSource) -> Self {
        let allow_unexpected = match source {
            Qwen3_5CheckpointSource::Multimodal => {
                vec![KeyMatcher::prefix("model.visual.")]
            }
            Qwen3_5CheckpointSource::TextOnly => Vec::new(),
        };
        Self {
            source,
            expected: Vec::new(),
            mappings: MappingPlan::new(),
            allow_unexpected,
        }
    }

    fn finish(self) -> Qwen3_5CheckpointContract {
        Qwen3_5CheckpointContract {
            expected: self.expected,
            mappings: self.mappings,
            allow_unexpected: self.allow_unexpected,
        }
    }

    fn language(&mut self, suffix: &str, shape: Vec<u64>) {
        let source = format!("{}.{suffix}", self.source.language_prefix());
        let target = format!("model.{suffix}");
        self.tensor(source, target, shape, TensorTransform::Identity);
    }

    fn tensor(
        &mut self,
        source: String,
        target: String,
        shape: Vec<u64>,
        transform: TensorTransform,
    ) {
        self.expected.push(ExpectedTensor::new(&target, shape));
        self.mappings.push(crate::checkpoint::TensorOperation::Map {
            source,
            target,
            transform,
        });
    }

    fn decoder_layer(
        &mut self,
        index: usize,
        layer_type: Qwen3_5LayerType,
        config: &Qwen3_5TextConfig,
        dimensions: &Dimensions,
    ) {
        let layer = format!("layers.{index}");
        self.language(
            &format!("{layer}.input_layernorm.weight"),
            vec![dimensions.hidden],
        );
        match layer_type {
            Qwen3_5LayerType::LinearAttention => self.delta_layer(&layer, dimensions),
            Qwen3_5LayerType::FullAttention => self.attention_layer(&layer, config, dimensions),
        }
        self.language(
            &format!("{layer}.post_attention_layernorm.weight"),
            vec![dimensions.hidden],
        );
        self.mlp(&format!("{layer}.mlp"), dimensions, true);
    }

    fn delta_layer(&mut self, layer: &str, dimensions: &Dimensions) {
        let prefix = format!("{layer}.linear_attn");
        self.language(
            &format!("{prefix}.A_log"),
            vec![dimensions.delta_value_heads],
        );
        self.language(
            &format!("{prefix}.conv1d.weight"),
            vec![dimensions.delta_conv, 1, dimensions.conv_kernel],
        );
        self.language(
            &format!("{prefix}.dt_bias"),
            vec![dimensions.delta_value_heads],
        );
        for projection in ["in_proj_a", "in_proj_b"] {
            self.language(
                &format!("{prefix}.{projection}.weight"),
                vec![dimensions.delta_value_heads, dimensions.hidden],
            );
        }
        self.language(
            &format!("{prefix}.in_proj_qkv.weight"),
            vec![dimensions.delta_conv, dimensions.hidden],
        );
        self.language(
            &format!("{prefix}.in_proj_z.weight"),
            vec![dimensions.delta_value, dimensions.hidden],
        );
        self.language(
            &format!("{prefix}.norm.weight"),
            vec![dimensions.delta_value_head_dim],
        );
        self.language(
            &format!("{prefix}.out_proj.weight"),
            vec![dimensions.hidden, dimensions.delta_value],
        );
    }

    fn attention_layer(
        &mut self,
        layer: &str,
        config: &Qwen3_5TextConfig,
        dimensions: &Dimensions,
    ) {
        let prefix = format!("{layer}.self_attn");
        let query_width = if config.attn_output_gate {
            dimensions.gated_query
        } else {
            dimensions.query
        };
        self.language(
            &format!("{prefix}.q_proj.weight"),
            vec![query_width, dimensions.hidden],
        );
        self.language(
            &format!("{prefix}.k_proj.weight"),
            vec![dimensions.key_value, dimensions.hidden],
        );
        self.language(
            &format!("{prefix}.v_proj.weight"),
            vec![dimensions.key_value, dimensions.hidden],
        );
        self.language(
            &format!("{prefix}.o_proj.weight"),
            vec![dimensions.hidden, dimensions.query],
        );
        if config.attention_bias {
            self.language(&format!("{prefix}.q_proj.bias"), vec![query_width]);
            self.language(&format!("{prefix}.k_proj.bias"), vec![dimensions.key_value]);
            self.language(&format!("{prefix}.v_proj.bias"), vec![dimensions.key_value]);
            self.language(&format!("{prefix}.o_proj.bias"), vec![dimensions.hidden]);
        }
        self.language(
            &format!("{prefix}.q_norm.weight"),
            vec![dimensions.head_dim],
        );
        self.language(
            &format!("{prefix}.k_norm.weight"),
            vec![dimensions.head_dim],
        );
    }

    fn mlp(&mut self, prefix: &str, dimensions: &Dimensions, language: bool) {
        for (projection, shape) in [
            (
                "gate_proj",
                vec![dimensions.intermediate, dimensions.hidden],
            ),
            ("up_proj", vec![dimensions.intermediate, dimensions.hidden]),
            (
                "down_proj",
                vec![dimensions.hidden, dimensions.intermediate],
            ),
        ] {
            let suffix = format!("{prefix}.{projection}.weight");
            if language {
                self.language(&suffix, shape);
            } else {
                self.tensor(
                    format!("mtp.{suffix}"),
                    format!("mtp.{suffix}"),
                    shape,
                    TensorTransform::Identity,
                );
            }
        }
    }

    fn mtp(&mut self, config: &Qwen3_5TextConfig, dimensions: &Dimensions) {
        if config.mtp_num_hidden_layers == 0 {
            return;
        }
        self.mtp_tensor(
            "fc.weight",
            vec![dimensions.hidden, dimensions.double_hidden],
        );
        self.mtp_tensor("pre_fc_norm_embedding.weight", vec![dimensions.hidden]);
        self.mtp_tensor("pre_fc_norm_hidden.weight", vec![dimensions.hidden]);
        self.mtp_tensor("norm.weight", vec![dimensions.hidden]);

        for index in 0..config.mtp_num_hidden_layers {
            let layer = format!("layers.{index}");
            self.mtp_tensor(
                &format!("{layer}.input_layernorm.weight"),
                vec![dimensions.hidden],
            );
            self.mtp_attention_layer(&layer, config, dimensions);
            self.mtp_tensor(
                &format!("{layer}.post_attention_layernorm.weight"),
                vec![dimensions.hidden],
            );
            self.mlp(&format!("{layer}.mlp"), dimensions, false);
        }
    }

    fn mtp_attention_layer(
        &mut self,
        layer: &str,
        config: &Qwen3_5TextConfig,
        dimensions: &Dimensions,
    ) {
        let prefix = format!("{layer}.self_attn");
        let query_width = if config.attn_output_gate {
            dimensions.gated_query
        } else {
            dimensions.query
        };
        self.mtp_tensor(
            &format!("{prefix}.q_proj.weight"),
            vec![query_width, dimensions.hidden],
        );
        self.mtp_tensor(
            &format!("{prefix}.k_proj.weight"),
            vec![dimensions.key_value, dimensions.hidden],
        );
        self.mtp_tensor(
            &format!("{prefix}.v_proj.weight"),
            vec![dimensions.key_value, dimensions.hidden],
        );
        self.mtp_tensor(
            &format!("{prefix}.o_proj.weight"),
            vec![dimensions.hidden, dimensions.query],
        );
        if config.attention_bias {
            self.mtp_tensor(&format!("{prefix}.q_proj.bias"), vec![query_width]);
            self.mtp_tensor(&format!("{prefix}.k_proj.bias"), vec![dimensions.key_value]);
            self.mtp_tensor(&format!("{prefix}.v_proj.bias"), vec![dimensions.key_value]);
            self.mtp_tensor(&format!("{prefix}.o_proj.bias"), vec![dimensions.hidden]);
        }
        self.mtp_tensor(
            &format!("{prefix}.q_norm.weight"),
            vec![dimensions.head_dim],
        );
        self.mtp_tensor(
            &format!("{prefix}.k_norm.weight"),
            vec![dimensions.head_dim],
        );
    }

    fn mtp_tensor(&mut self, suffix: &str, shape: Vec<u64>) {
        self.tensor(
            format!("mtp.{suffix}"),
            format!("mtp.{suffix}"),
            shape,
            TensorTransform::Identity,
        );
    }
}

fn dimension(field: &'static str, value: usize) -> Result<u64, ConfigError> {
    u64::try_from(value).map_err(|_| {
        ConfigError::new(
            FAMILY,
            field,
            "does not fit the portable checkpoint dimension representation",
        )
    })
}

fn product(field: &'static str, values: &[usize]) -> Result<u64, ConfigError> {
    values.iter().try_fold(1_u64, |accumulator, value| {
        let value = dimension(field, *value)?;
        accumulator.checked_mul(value).ok_or_else(|| {
            ConfigError::new(FAMILY, field, "overflows checkpoint tensor dimensions")
        })
    })
}

fn checked_sum(field: &'static str, values: &[u64]) -> Result<u64, ConfigError> {
    values.iter().try_fold(0_u64, |accumulator, value| {
        accumulator.checked_add(*value).ok_or_else(|| {
            ConfigError::new(FAMILY, field, "overflows checkpoint tensor dimensions")
        })
    })
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::PathBuf, str::FromStr};

    use serde::Deserialize;
    use serde_json::Value;

    use crate::{
        checkpoint::{
            ArtifactSource, CheckpointError, CheckpointManifest, KeyMatcher, LoadOptions,
            PlannedTransform, TensorDType, TensorInventory, TensorOperation, TensorTransform,
        },
        models::config::ModelConfig,
    };

    use super::*;

    const QWEN_08_CONFIG: &str = include_str!("../../../tests/fixtures/qwen3_5_0_8b.json");
    const QWEN_27_CONFIG: &str = include_str!("../../../tests/fixtures/qwen3_8_27b.json");
    const QWEN_08_INVENTORY: &str =
        include_str!("../../../tests/fixtures/qwen3_5_checkpoint/qwen3_5_0_8b_inventory.json");
    const QWEN_27_INVENTORY: &str =
        include_str!("../../../tests/fixtures/qwen3_5_checkpoint/qwen3_8_27b_inventory.json");
    const TINY_INVENTORY: &str =
        include_str!("../../../tests/fixtures/qwen3_5_checkpoint/tiny_text_inventory.json");

    #[derive(Debug, Deserialize)]
    struct GoldenInventory {
        checkpoint: String,
        revision: String,
        #[serde(default)]
        config: Option<Value>,
        tensors: Vec<GoldenTensor>,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct GoldenTensor {
        name: String,
        shape: Vec<u64>,
        dtype: String,
    }

    fn golden(json: &str) -> GoldenInventory {
        serde_json::from_str(json).unwrap()
    }

    fn config(json: &str) -> Qwen3_5TextConfig {
        <Qwen3_5TextConfig as ModelConfig>::from_json_str(json).unwrap()
    }

    fn multimodal_with_mtp() -> Qwen3_5CheckpointOptions {
        Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::Multimodal,
            include_mtp: true,
        }
    }

    fn manifest(golden: &GoldenInventory) -> CheckpointManifest {
        let mut offset = 0_u64;
        let mut tensors = BTreeMap::new();
        for tensor in &golden.tensors {
            let dtype = TensorDType::from_str(&tensor.dtype).unwrap();
            let elements = tensor.shape.iter().copied().product();
            let byte_len = dtype.storage_bytes(elements).unwrap();
            let end = offset.checked_add(byte_len).unwrap();
            tensors.insert(
                tensor.name.clone(),
                TensorInventory {
                    name: tensor.name.clone(),
                    shard: "metadata-only-fixture.safetensors".into(),
                    shape: tensor.shape.clone(),
                    dtype,
                    data_offsets: [offset, end],
                    byte_len,
                },
            );
            offset = end;
        }
        CheckpointManifest {
            root: PathBuf::from("metadata-only-fixture"),
            source: ArtifactSource::Single {
                relative_path: "metadata-only-fixture.safetensors".into(),
            },
            shards: Vec::new(),
            tensors,
            tensor_bytes: offset,
            artifact_bytes: offset,
        }
    }

    fn expected_shape<'a>(contract: &'a Qwen3_5CheckpointContract, name: &str) -> &'a [u64] {
        &contract
            .expected_tensors()
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap()
            .shape
    }

    fn planned<'a>(plan: &'a LoadPlan, target: &str) -> &'a crate::checkpoint::PlannedTensor {
        plan.tensors
            .iter()
            .find(|tensor| tensor.target == target)
            .unwrap()
    }

    #[test]
    fn tiny_text_fixture_is_an_exact_strict_contract() {
        let fixture = golden(TINY_INVENTORY);
        assert_eq!(fixture.checkpoint, "synthetic/qwen3_5-tiny");
        assert_eq!(fixture.revision, "test-fixture-v1");
        let config = config(&fixture.config.as_ref().unwrap().to_string());
        let contract = config
            .checkpoint_contract(Qwen3_5CheckpointOptions {
                source: Qwen3_5CheckpointSource::TextOnly,
                include_mtp: true,
            })
            .unwrap();
        let plan = contract
            .build_load_plan(&manifest(&fixture), &LoadOptions::default())
            .unwrap();

        assert_eq!(fixture.tensors.len(), 42);
        assert_eq!(contract.expected_tensors().len(), 43);
        assert_eq!(plan.parameter_count(), 43);
        assert!(plan.report.missing.is_empty());
        assert!(plan.report.unexpected.is_empty());
        assert!(plan.report.allowlisted_ignored.is_empty());
        assert!(matches!(
            plan.aliases.as_slice(),
            [alias]
                if alias.target == "lm_head.weight"
                    && matches!(
                        &alias.transform,
                        PlannedTransform::Alias { source_target }
                            if source_target == "model.embed_tokens.weight"
                    )
        ));
        assert_eq!(
            expected_shape(&contract, "model.layers.0.linear_attn.in_proj_qkv.weight"),
            [12, 8]
        );
        assert_eq!(expected_shape(&contract, "mtp.fc.weight"), [8, 16]);
    }

    #[test]
    fn official_qwen35_point8_metadata_is_consumed_without_unknown_keys() {
        let fixture = golden(QWEN_08_INVENTORY);
        assert_eq!(fixture.checkpoint, "Qwen/Qwen3.5-0.8B");
        assert_eq!(fixture.revision, "2fc06364715b967f1860aea9cf38778875588b17");
        assert_eq!(fixture.tensors.len(), 488);
        let contract = config(QWEN_08_CONFIG)
            .checkpoint_contract(multimodal_with_mtp())
            .unwrap();
        let plan = contract
            .build_load_plan(&manifest(&fixture), &LoadOptions::default())
            .unwrap();

        assert_eq!(contract.expected_tensors().len(), 336);
        assert_eq!(plan.tensors.len(), 335);
        assert_eq!(plan.aliases.len(), 1);
        assert_eq!(plan.report.allowlisted_ignored.len(), 153);
        assert!(
            plan.report
                .allowlisted_ignored
                .iter()
                .all(|name| name.starts_with("model.visual."))
        );
        assert!(
            plan.report
                .allowlisted_ignored
                .iter()
                .all(|name| !name.starts_with("mtp."))
        );

        let delta = planned(&plan, "model.layers.0.linear_attn.in_proj_qkv.weight");
        assert_eq!(delta.shape, [6_144, 1_024]);
        assert_eq!(
            delta.sources[0].name,
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        );
        let query = planned(&plan, "model.layers.3.self_attn.q_proj.weight");
        assert_eq!(query.shape, [4_096, 1_024]);
        assert!(matches!(query.transform, PlannedTransform::Identity));
        assert!(matches!(
            planned(&plan, "mtp.layers.0.self_attn.q_proj.weight").transform,
            PlannedTransform::Identity
        ));
    }

    #[test]
    fn official_qwen38_27b_metadata_is_consumed_without_unknown_keys() {
        let fixture = golden(QWEN_27_INVENTORY);
        assert_eq!(fixture.checkpoint, "Qwen/Qwen3.8-27B");
        assert_eq!(fixture.revision, "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0");
        assert_eq!(fixture.tensors.len(), 1_199);
        let contract = config(QWEN_27_CONFIG)
            .checkpoint_contract(multimodal_with_mtp())
            .unwrap();
        let plan = contract
            .build_load_plan(&manifest(&fixture), &LoadOptions::default())
            .unwrap();

        assert_eq!(contract.expected_tensors().len(), 866);
        assert_eq!(plan.tensors.len(), 866);
        assert!(plan.aliases.is_empty());
        assert_eq!(plan.report.allowlisted_ignored.len(), 333);
        assert_eq!(
            expected_shape(&contract, "model.layers.0.linear_attn.conv1d.weight"),
            [10_240, 1, 4]
        );
        assert_eq!(
            expected_shape(&contract, "model.layers.0.linear_attn.in_proj_a.weight"),
            [48, 5_120]
        );
        assert_eq!(
            expected_shape(&contract, "model.layers.0.linear_attn.in_proj_z.weight"),
            [6_144, 5_120]
        );
        assert_eq!(
            expected_shape(&contract, "model.layers.3.self_attn.q_proj.weight"),
            [12_288, 5_120]
        );
        assert_eq!(
            expected_shape(&contract, "model.layers.3.self_attn.o_proj.weight"),
            [5_120, 6_144]
        );
        assert_eq!(expected_shape(&contract, "mtp.fc.weight"), [5_120, 10_240]);
    }

    #[test]
    fn safe_default_classifies_uninstantiated_mtp_tensors_separately() {
        let fixture = golden(QWEN_08_INVENTORY);
        let contract = config(QWEN_08_CONFIG)
            .checkpoint_contract(Qwen3_5CheckpointOptions::default())
            .unwrap();
        assert_eq!(contract.expected_tensors().len(), 321);
        assert!(
            contract
                .allow_unexpected()
                .contains(&KeyMatcher::prefix("mtp."))
        );

        let plan = contract
            .build_load_plan(&manifest(&fixture), &LoadOptions::default())
            .unwrap();
        assert_eq!(plan.report.allowlisted_ignored.len(), 168);
        assert_eq!(
            plan.report
                .allowlisted_ignored
                .iter()
                .filter(|name| name.starts_with("mtp."))
                .count(),
            15
        );
    }

    #[test]
    fn multimodal_exceptions_are_additive_and_deduplicated() {
        let contract = config(QWEN_08_CONFIG)
            .checkpoint_contract(Qwen3_5CheckpointOptions::default())
            .unwrap();
        let custom = KeyMatcher::exact("local.adapter.weight");
        let mut options = LoadOptions {
            allow_unexpected: vec![KeyMatcher::prefix("model.visual."), custom.clone()],
            ..LoadOptions::default()
        };
        contract.configure_load_options(&mut options);
        assert_eq!(
            options
                .allow_unexpected
                .iter()
                .filter(|matcher| **matcher == KeyMatcher::prefix("model.visual."))
                .count(),
            1
        );
        assert!(options.allow_unexpected.contains(&custom));
    }

    #[test]
    fn query_weight_and_bias_preserve_official_head_local_layout() {
        let fixture = golden(TINY_INVENTORY);
        let mut config = config(&fixture.config.unwrap().to_string());
        config.attention_bias = true;
        let contract = config
            .checkpoint_contract(Qwen3_5CheckpointOptions {
                source: Qwen3_5CheckpointSource::TextOnly,
                include_mtp: true,
            })
            .unwrap();
        for target in [
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.q_proj.bias",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.q_proj.bias",
        ] {
            let operation = contract
                .mappings()
                .operations()
                .iter()
                .find(|operation| {
                    matches!(operation, TensorOperation::Map { target: found, .. } if found == target)
                })
                .unwrap();
            assert!(matches!(
                operation,
                TensorOperation::Map {
                    transform: TensorTransform::Identity,
                    ..
                }
            ));
        }
        assert_eq!(
            expected_shape(&contract, "model.layers.1.self_attn.q_proj.bias"),
            [16]
        );
    }

    #[test]
    fn malformed_gated_query_shape_fails_inventory_validation() {
        let mut fixture = golden(TINY_INVENTORY);
        fixture
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == "model.layers.1.self_attn.q_proj.weight")
            .unwrap()
            .shape = vec![15, 8];
        let config = config(&fixture.config.as_ref().unwrap().to_string());
        let contract = config
            .checkpoint_contract(Qwen3_5CheckpointOptions {
                source: Qwen3_5CheckpointSource::TextOnly,
                include_mtp: true,
            })
            .unwrap();
        assert!(matches!(
            contract.build_load_plan(&manifest(&fixture), &LoadOptions::default()),
            Err(CheckpointError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn dedicated_mtp_embeddings_are_rejected_until_a_named_contract_exists() {
        let fixture = golden(TINY_INVENTORY);
        let mut config = config(&fixture.config.unwrap().to_string());
        config.mtp_use_dedicated_embeddings = true;
        let error = config
            .checkpoint_contract(Qwen3_5CheckpointOptions {
                source: Qwen3_5CheckpointSource::TextOnly,
                include_mtp: true,
            })
            .unwrap_err();
        assert_eq!(error.field, "mtp_use_dedicated_embeddings");
    }

    #[test]
    fn dimension_products_are_checked() {
        let fixture = golden(TINY_INVENTORY);
        let mut config = config(&fixture.config.unwrap().to_string());
        config.num_attention_heads = usize::MAX;
        config.num_key_value_heads = 1;
        let error = config
            .checkpoint_contract(Qwen3_5CheckpointOptions {
                source: Qwen3_5CheckpointSource::TextOnly,
                include_mtp: false,
            })
            .unwrap_err();
        assert!(error.reason.contains("overflows"));
    }
}
