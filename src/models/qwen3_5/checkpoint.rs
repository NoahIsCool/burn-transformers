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

/// Family label attached to checkpoint geometry errors.
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
    /// Returns the exact source namespace for the selected text checkpoint layout.
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
    /// Enable this when loading a model initialized with non-zero
    /// `mtp_num_hidden_layers`.  Leaving it disabled is the explicit base-only
    /// compatibility mode and reports MTP keys as an intentionally ignored
    /// optional subsystem.
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
    /// Required parameter names and shapes in deterministic topology order.
    expected: Vec<ExpectedTensor>,
    /// Explicit source-to-parameter transformations and tied-storage aliases.
    mappings: MappingPlan,
    /// Known vision or excluded MTP keys permitted outside the selected graph.
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

// Checked checkpoint geometry: no tensor data is allocated below.

mod dimensions; // Verified fused-width arithmetic of the tensor contract.

use dimensions::{DimensionInputs, Dimensions, checkpoint_dimensions, query_projection};

impl TryFrom<&Qwen3_5TextConfig> for Dimensions {
    /// Configuration field and reason for an overflowing dimension.
    type Error = ConfigError;

    /// Computes portable tensor dimensions with checked products and sums.
    ///
    /// Projection widths are head counts times head widths; DeltaNet convolution
    /// channels concatenate Q, K, and V. Overflow is rejected before allocation
    /// by the verified [`checkpoint_dimensions`].
    fn try_from(config: &Qwen3_5TextConfig) -> Result<Self, Self::Error> {
        checkpoint_dimensions(DimensionInputs {
            vocab: config.vocab_size,
            hidden: config.hidden_size,
            intermediate: config.intermediate_size,
            attention_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            delta_key_heads: config.linear_num_key_heads,
            delta_key_head_dim: config.linear_key_head_dim,
            delta_value_heads: config.linear_num_value_heads,
            delta_value_head_dim: config.linear_value_head_dim,
            conv_kernel: config.linear_conv_kernel_dim,
        })
        .map_err(|error| {
            ConfigError::new(
                FAMILY,
                error.field(),
                "overflows checkpoint tensor dimensions",
            )
        })
    }
}

// Topology-to-inventory translation and explicit source namespaces.

/// Accumulates required tensors and mappings for a selected source namespace.
struct ContractBuilder {
    /// Checkpoint wrapper whose language namespace supplies tensor names.
    source: Qwen3_5CheckpointSource,
    /// Required parameter names and shapes in deterministic topology order.
    expected: Vec<ExpectedTensor>,
    /// Explicit source-to-parameter transformations and tied-storage aliases.
    mappings: MappingPlan,
    /// Known vision or excluded MTP keys permitted outside the selected graph.
    allow_unexpected: Vec<KeyMatcher>,
}

impl ContractBuilder {
    /// Starts an empty contract and explicitly exempts the multimodal vision tower.
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

    /// Publishes the completed inventory, mappings, and intentional key exceptions.
    fn finish(self) -> Qwen3_5CheckpointContract {
        Qwen3_5CheckpointContract {
            expected: self.expected,
            mappings: self.mappings,
            allow_unexpected: self.allow_unexpected,
        }
    }

    /// Maps one source text tensor to its canonical `model.*` target without reshaping.
    fn language(&mut self, suffix: &str, shape: Vec<u64>) {
        let source = format!("{}.{suffix}", self.source.language_prefix());
        let target = format!("model.{suffix}");
        self.tensor(source, target, shape, TensorTransform::Identity);
    }

    /// Registers an expected target shape and its explicit source transformation.
    ///
    /// Shapes describe checkpoint axes; the runtime parameter mapper handles
    /// Burn linear storage layout separately.
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

    /// Adds pre-attention norm, the configured mixer, post-attention norm, and SwiGLU.
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

    /// Declares DeltaNet projection, convolution, decay, and gated-normalization tensors.
    ///
    /// Linear weights use `[output, input]`; the depthwise convolution uses
    /// `[2 * key_width + value_width, 1, kernel]`. Recurrent state is not a parameter.
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

    /// Declares native GQA projections and per-head Q/K normalization weights.
    ///
    /// Gated Q rows retain `[head, Q_or_gate, head_dim]` order; K/V stay at
    /// their native head count rather than being expanded to query heads.
    fn attention_layer(
        &mut self,
        layer: &str,
        config: &Qwen3_5TextConfig,
        dimensions: &Dimensions,
    ) {
        let prefix = format!("{layer}.self_attn");
        let query_width = query_projection(dimensions, config.attn_output_gate);
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

    /// Declares `down(SiLU(gate(x)) * up(x))` weights in text or MTP namespace.
    ///
    /// Gate and up weights are `[intermediate, hidden]`; down reverses those axes.
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

    /// Adds the shared embedding/hidden fusion and configured MTP decoder blocks.
    ///
    /// The fusion projection maps concatenated width `2 * hidden` back to
    /// `hidden`; zero MTP depth contributes no parameters.
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

    /// Declares MTP full-attention weights with the same gated-query layout as the base.
    ///
    /// MTP uses its own namespace and independent KV state during execution.
    fn mtp_attention_layer(
        &mut self,
        layer: &str,
        config: &Qwen3_5TextConfig,
        dimensions: &Dimensions,
    ) {
        let prefix = format!("{layer}.self_attn");
        let query_width = query_projection(dimensions, config.attn_output_gate);
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

    /// Registers one unchanged `mtp.*` tensor in both source and target inventories.
    fn mtp_tensor(&mut self, suffix: &str, shape: Vec<u64>) {
        self.tensor(
            format!("mtp.{suffix}"),
            format!("mtp.{suffix}"),
            shape,
            TensorTransform::Identity,
        );
    }
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
