//! Exact Muse-Glimmer language-tower SafeTensors contract.
//!
//! The released checkpoint is a multimodal conditional-generation repository.
//! Its text parameters live under `model.language_model`, while the independent
//! vocabulary head remains `lm_head.weight`. The Burn model uses the canonical
//! internal `model.*` namespace, so every language key is explicitly renamed.
//! Vision-tower, adapter, and projection keys are accepted only through three
//! narrow prefix matchers and therefore remain visible as allowlisted entries in
//! the final load report.
//!
//! All matrices retain official `[output, input]` checkpoint order. Burn's
//! `LinearLayout::Col` record mapper performs the single runtime transpose when
//! each streamed parameter is installed.

use serde::{Deserialize, Serialize};

use crate::checkpoint::{
    CheckpointManifest, ExpectedTensor, KeyMatcher, LoadOptions, LoadPlan, MappingPlan,
    TensorOperation, TensorTransform, build_load_plan,
};

use super::{ConfigError, MuseGlimmerTextConfig, Validate};

const FAMILY: &str = "muse_glimmer";

/// Serialized namespace containing a Muse text decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MuseGlimmerCheckpointSource {
    /// Official conditional-generation layout below `model.language_model`.
    Multimodal,
    /// Standalone causal-LM layout below `model` for converted local artifacts.
    TextOnly,
}

impl MuseGlimmerCheckpointSource {
    /// Returns the serialized prefix preceding language-model field names.
    const fn language_prefix(self) -> &'static str {
        match self {
            Self::Multimodal => "model.language_model",
            Self::TextOnly => "model",
        }
    }
}

/// Family-specific checkpoint selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MuseGlimmerCheckpointOptions {
    /// Namespace layout exposed by the local artifact.
    pub source: MuseGlimmerCheckpointSource,
}

impl Default for MuseGlimmerCheckpointOptions {
    /// Selects the released multimodal repository layout.
    fn default() -> Self {
        Self {
            source: MuseGlimmerCheckpointSource::Multimodal,
        }
    }
}

/// Complete expected language inventory, mapping plan, and visual exceptions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseGlimmerCheckpointContract {
    expected: Vec<ExpectedTensor>,
    mappings: MappingPlan,
    allow_unexpected: Vec<KeyMatcher>,
}

impl MuseGlimmerCheckpointContract {
    /// Returns every logical Burn parameter in deterministic topology order.
    #[must_use]
    pub fn expected_tensors(&self) -> &[ExpectedTensor] {
        &self.expected
    }

    /// Returns every explicit source-to-target mapping and tied alias.
    #[must_use]
    pub const fn mappings(&self) -> &MappingPlan {
        &self.mappings
    }

    /// Returns narrow exceptions for intentionally uninstantiated visual tensors.
    #[must_use]
    pub fn allow_unexpected(&self) -> &[KeyMatcher] {
        &self.allow_unexpected
    }

    /// Adds family-owned exceptions without removing caller policy.
    pub fn configure_load_options(&self, options: &mut LoadOptions) {
        for matcher in &self.allow_unexpected {
            if !options.allow_unexpected.contains(matcher) {
                options.allow_unexpected.push(matcher.clone());
            }
        }
    }

    /// Strictly validates an observed manifest before any payload allocation.
    pub fn build_load_plan(
        &self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
    ) -> crate::checkpoint::Result<LoadPlan> {
        let mut options = options.clone();
        self.configure_load_options(&mut options);
        let plan = build_load_plan(manifest, &self.expected, &self.mappings, &options)?;
        if let Some(tensor) = plan.tensors.iter().find(|tensor| !tensor.dtype.is_float()) {
            return Err(crate::checkpoint::CheckpointError::InvalidMapping {
                message: format!(
                    "Muse-Glimmer parameter `{}` must be floating point, planned dtype is {}",
                    tensor.target, tensor.dtype
                ),
            });
        }
        Ok(plan)
    }
}

impl MuseGlimmerTextConfig {
    /// Derives the exact text-side inventory from validated shape metadata.
    ///
    /// The official 52-layer checkpoint contains 627 consumed tensors: 626 in
    /// `model.language_model` plus the vocabulary head. No Q/K norm parameter is
    /// expected because Muse's per-head normalization is scale-free.
    pub fn checkpoint_contract(
        &self,
        options: MuseGlimmerCheckpointOptions,
    ) -> Result<MuseGlimmerCheckpointContract, ConfigError> {
        self.validate()?;
        let dimensions = Dimensions::try_from(self)?;
        let mut builder = ContractBuilder::new(options.source);
        builder.language(
            "embed_tokens.weight",
            vec![dimensions.vocab, dimensions.hidden],
        );
        for layer_index in 0..self.num_hidden_layers {
            builder.decoder_layer(layer_index, self, dimensions);
        }
        builder.language("norm.weight", vec![dimensions.hidden]);
        builder.expected.push(ExpectedTensor::new(
            "lm_head.weight",
            vec![dimensions.vocab, dimensions.hidden],
        ));
        if self.tie_word_embeddings {
            builder
                .mappings
                .alias("model.embed_tokens.weight", "lm_head.weight");
            builder
                .allow_unexpected
                .push(KeyMatcher::exact("lm_head.weight"));
        } else {
            builder.mappings.map("lm_head.weight", "lm_head.weight");
        }
        Ok(builder.finish())
    }
}

/// Checked shape terms shared across every layer mapping.
#[derive(Debug, Clone, Copy)]
struct Dimensions {
    vocab: u64,
    hidden: u64,
    intermediate: u64,
    query: u64,
    key_value: u64,
}

impl TryFrom<&MuseGlimmerTextConfig> for Dimensions {
    type Error = ConfigError;

    /// Converts portable dimensions and checks projection-width products.
    fn try_from(config: &MuseGlimmerTextConfig) -> Result<Self, Self::Error> {
        Ok(Self {
            vocab: dimension("vocab_size", config.vocab_size)?,
            hidden: dimension("hidden_size", config.hidden_size)?,
            intermediate: dimension("intermediate_size", config.intermediate_size)?,
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
        })
    }
}

/// Incrementally builds deterministic model and source inventories.
struct ContractBuilder {
    source: MuseGlimmerCheckpointSource,
    expected: Vec<ExpectedTensor>,
    mappings: MappingPlan,
    allow_unexpected: Vec<KeyMatcher>,
}

impl ContractBuilder {
    /// Creates an empty contract with source-specific visual exceptions.
    fn new(source: MuseGlimmerCheckpointSource) -> Self {
        let allow_unexpected = match source {
            MuseGlimmerCheckpointSource::Multimodal => vec![
                KeyMatcher::prefix("model.vision_tower."),
                KeyMatcher::prefix("model.vision_adapter."),
                KeyMatcher::prefix("model.vision_projection."),
            ],
            MuseGlimmerCheckpointSource::TextOnly => Vec::new(),
        };
        Self {
            source,
            expected: Vec::new(),
            mappings: MappingPlan::new(),
            allow_unexpected,
        }
    }

    /// Finalizes the immutable contract.
    fn finish(self) -> MuseGlimmerCheckpointContract {
        MuseGlimmerCheckpointContract {
            expected: self.expected,
            mappings: self.mappings,
            allow_unexpected: self.allow_unexpected,
        }
    }

    /// Adds one serialized language key and its canonical internal target.
    fn language(&mut self, suffix: &str, shape: Vec<u64>) {
        let source = format!("{}.{suffix}", self.source.language_prefix());
        let target = format!("model.{suffix}");
        self.expected.push(ExpectedTensor::new(&target, shape));
        self.mappings.push(TensorOperation::Map {
            source,
            target,
            transform: TensorTransform::Identity,
        });
    }

    /// Adds every tensor owned by one sandwich-normalized decoder layer.
    fn decoder_layer(
        &mut self,
        index: usize,
        config: &MuseGlimmerTextConfig,
        dimensions: Dimensions,
    ) {
        let layer = format!("layers.{index}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            self.language(&format!("{layer}.{norm}.weight"), vec![dimensions.hidden]);
        }
        let attention = format!("{layer}.self_attn");
        for (projection, shape) in [
            ("q_proj.weight", vec![dimensions.query, dimensions.hidden]),
            (
                "k_proj.weight",
                vec![dimensions.key_value, dimensions.hidden],
            ),
            (
                "v_proj.weight",
                vec![dimensions.key_value, dimensions.hidden],
            ),
            (
                "gate_proj.weight",
                vec![dimensions.query, dimensions.hidden],
            ),
            ("o_proj.weight", vec![dimensions.hidden, dimensions.query]),
        ] {
            self.language(&format!("{attention}.{projection}"), shape);
        }
        if config.attention_bias {
            for (projection, shape) in [
                ("q_proj.bias", vec![dimensions.query]),
                ("k_proj.bias", vec![dimensions.key_value]),
                ("v_proj.bias", vec![dimensions.key_value]),
                ("o_proj.bias", vec![dimensions.hidden]),
            ] {
                self.language(&format!("{attention}.{projection}"), shape);
            }
        }
        let mlp = format!("{layer}.mlp");
        for (projection, shape) in [
            (
                "gate_proj.weight",
                vec![dimensions.intermediate, dimensions.hidden],
            ),
            (
                "up_proj.weight",
                vec![dimensions.intermediate, dimensions.hidden],
            ),
            (
                "down_proj.weight",
                vec![dimensions.hidden, dimensions.intermediate],
            ),
        ] {
            self.language(&format!("{mlp}.{projection}"), shape);
        }
    }
}

/// Converts one runtime dimension to the SafeTensors shape representation.
fn dimension(field: &'static str, value: usize) -> Result<u64, ConfigError> {
    u64::try_from(value).map_err(|_| {
        ConfigError::new(
            FAMILY,
            field,
            "dimension cannot be represented by the checkpoint format",
        )
    })
}

/// Multiplies a projection width without truncation or overflow.
fn product(field: &'static str, left: usize, right: usize) -> Result<u64, ConfigError> {
    left.checked_mul(right)
        .ok_or_else(|| ConfigError::new(FAMILY, field, "dimension product overflows usize"))
        .and_then(|value| dimension(field, value))
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
