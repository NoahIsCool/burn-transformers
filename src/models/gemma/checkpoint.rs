//! The strict SafeTensors contract of a Gemma 2 or Gemma 3 text checkpoint.
//!
//! Every projection is stored row-major `[out_features, in_features]`;
//! Burn's column-layout [`burn::nn::Linear`] consumes that layout directly.
//! The contract lists every tensor the executable graph owns, with its exact
//! shape, so loading fails before the first payload read on any missing,
//! unexpected, or misshapen tensor. With `d = hidden_size`,
//! `f = intermediate_size`, `V = vocab_size`, `h = head_dim`, `H` query
//! heads, `K` key/value heads, and `P` the source's language prefix:
//!
//! | checkpoint tensor | shape | present when | model field |
//! | --- | --- | --- | --- |
//! | `P.embed_tokens.weight` | `[V, d]` | always | `model.embed_tokens.weight` |
//! | `P.layers.L.input_layernorm.weight` | `[d]` | always | `model.layers.L.input_layernorm.weight` |
//! | `P.layers.L.post_attention_layernorm.weight` | `[d]` | always | `model.layers.L.post_attention_layernorm.weight` |
//! | `P.layers.L.pre_feedforward_layernorm.weight` | `[d]` | always | `model.layers.L.pre_feedforward_layernorm.weight` |
//! | `P.layers.L.post_feedforward_layernorm.weight` | `[d]` | always | `model.layers.L.post_feedforward_layernorm.weight` |
//! | `P.layers.L.self_attn.q_proj.weight` | `[H·h, d]` | always | `model.layers.L.self_attn.q_proj.weight` |
//! | `P.layers.L.self_attn.{k,v}_proj.weight` | `[K·h, d]` | always | `model.layers.L.self_attn.{k,v}_proj.weight` |
//! | `P.layers.L.self_attn.o_proj.weight` | `[d, H·h]` | always | `model.layers.L.self_attn.o_proj.weight` |
//! | `P.layers.L.self_attn.{q,k,v,o}_proj.bias` | `[H·h]`, `[K·h]`, `[K·h]`, `[d]` | `attention_bias` | `…_proj.bias` |
//! | `P.layers.L.self_attn.q_norm.weight` | `[h]` | Gemma 3 | `model.layers.L.self_attn.query_norm.weight` |
//! | `P.layers.L.self_attn.k_norm.weight` | `[h]` | Gemma 3 | `model.layers.L.self_attn.key_norm.weight` |
//! | `P.layers.L.mlp.{gate,up}_proj.weight` | `[f, d]` | always | `model.layers.L.mlp.{gate,up}_proj.weight` |
//! | `P.layers.L.mlp.down_proj.weight` | `[d, f]` | always | `model.layers.L.mlp.down_proj.weight` |
//! | `P.norm.weight` | `[d]` | always | `model.norm.weight` |
//! | head (see below) | `[V, d]` | untied | `lm_head.weight` |
//! | (alias of `model.embed_tokens.weight`) | `[V, d]` | tied | `lm_head.weight` |
//!
//! Norm weights are the zero-centered `w` of the `1 + w` scale. Every name
//! is the reference's own; the only renames are the shared attention
//! module's Q/K norm fields.
//!
//! # Sources
//!
//! [`GemmaCheckpointSource`] names the three published layouts, told apart by
//! which embedding name the manifest holds (exactly one must be present):
//!
//! | source | `P` | untied head | omitted vision tensors |
//! | --- | --- | --- | --- |
//! | `TextOnly` (`Gemma2ForCausalLM`, `Gemma3ForCausalLM`) | `model` | `lm_head.weight` | none |
//! | `Multimodal` (`Gemma3ForConditionalGeneration`, Transformers ≥ 4.52) | `model.language_model` | `lm_head.weight` | `model.vision_tower.*`, `model.multi_modal_projector.*` |
//! | `LegacyMultimodal` (the original Gemma 3 release, Transformers 4.50) | `language_model.model` | `language_model.lm_head.weight` | `vision_tower.*`, `multi_modal_projector.*` |
//!
//! # The omitted vision tower
//!
//! This crate executes only Gemma 3's text tower. A multimodal checkpoint's
//! SigLIP vision encoder and multimodal projector are therefore *explicitly*
//! exempted by prefix: they are reported in the load report's
//! `allowlisted_ignored` list, never read, and any other unexpected tensor
//! still fails the load. Text-only checkpoints get no exemption.
//!
//! The multimodal references also differ from the text-only ones in one
//! numerical step: no Transformers 5.2 multimodal wrapper
//! (`Gemma3ForConditionalGeneration`, which `AutoModelForCausalLM` builds
//! for a `gemma3` checkpoint, or `PaliGemmaForConditionalGeneration`)
//! applies `final_logit_softcapping`, while `Gemma2ForCausalLM` and
//! `Gemma3ForCausalLM` do. A multimodal-namespace checkpoint with a non-null
//! final soft-cap would therefore disagree with its own reference, so its
//! contract is refused; every released multimodal Gemma 3 config sets `null`.

use serde::{Deserialize, Serialize};

use crate::checkpoint::{
    CheckpointError, CheckpointManifest, ExpectedTensor, KeyMatcher, LoadOptions, LoadPlan,
    MappingPlan, build_load_plan,
};
use crate::models::config::ConfigError;

use super::architecture::GemmaArchitecture;
use super::config::GemmaConfig;
use super::plan::{UniqueFlag, unique_flag};
use crate::models::shared::decoder_plan::{ExtentError, checkpoint_extent, projection_extent};

/// The published SafeTensors namespace a Gemma checkpoint uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmaCheckpointSource {
    /// A text-only causal LM: `model.*` and `lm_head.weight`.
    #[default]
    TextOnly,
    /// A current multimodal Gemma 3: `model.language_model.*` and `lm_head.weight`.
    Multimodal,
    /// The original Gemma 3 release: `language_model.model.*` and `language_model.lm_head.weight`.
    LegacyMultimodal,
}

impl GemmaCheckpointSource {
    /// Every source, in detection order.
    pub const ALL: [Self; 3] = [Self::TextOnly, Self::Multimodal, Self::LegacyMultimodal];

    /// Returns the prefix of the language model's parameters.
    #[must_use]
    pub const fn language_prefix(self) -> &'static str {
        match self {
            Self::TextOnly => "model",
            Self::Multimodal => "model.language_model",
            Self::LegacyMultimodal => "language_model.model",
        }
    }

    /// Returns the checkpoint name of an untied LM head.
    #[must_use]
    pub const fn head_name(self) -> &'static str {
        match self {
            Self::TextOnly | Self::Multimodal => "lm_head.weight",
            Self::LegacyMultimodal => "language_model.lm_head.weight",
        }
    }

    /// Returns the vision-tower prefixes exempted from strict loading.
    #[must_use]
    pub fn omitted_prefixes(self) -> &'static [&'static str] {
        match self {
            Self::TextOnly => &[],
            Self::Multimodal => &["model.vision_tower.", "model.multi_modal_projector."],
            Self::LegacyMultimodal => &["vision_tower.", "multi_modal_projector."],
        }
    }

    /// Returns the checkpoint name of the embedding table, which identifies the source.
    fn embedding_name(self) -> String {
        format!("{}.embed_tokens.weight", self.language_prefix())
    }

    /// Classifies a tensor-name set by its embedding name.
    ///
    /// Exactly one layout's embedding (see [`Self::language_prefix`]) must be
    /// present; vision-tower and other names are ignored.
    ///
    /// # Errors
    ///
    /// [`CheckpointError::LanguageEmbeddingNotFound`] with every candidate
    /// embedding when none is present, and
    /// [`CheckpointError::AmbiguousLanguageEmbedding`] with the present ones
    /// when more than one is.
    pub fn detect<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<Self, CheckpointError> {
        let embeddings: Vec<String> = Self::ALL.map(Self::embedding_name).into();
        let mut found = [false; 3];
        for name in names {
            if let Some(index) = embeddings.iter().position(|embedding| embedding == name) {
                found[index] = true;
            }
        }
        match unique_flag(&found) {
            UniqueFlag::Unique(index) => Ok(Self::ALL[index]),
            UniqueFlag::Absent => Err(CheckpointError::LanguageEmbeddingNotFound {
                family: "Gemma",
                embeddings,
            }),
            UniqueFlag::Ambiguous => Err(CheckpointError::AmbiguousLanguageEmbedding {
                family: "Gemma",
                embeddings: embeddings
                    .into_iter()
                    .zip(found)
                    .filter_map(|(embedding, found)| found.then_some(embedding))
                    .collect(),
            }),
        }
    }
}

/// Family-specific checkpoint policy: which published namespace to read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GemmaCheckpointOptions {
    /// Namespace of the language parameters (and of the omitted vision tower).
    pub source: GemmaCheckpointSource,
}

/// The complete expected tensor inventory, its mapping, and the vision exemptions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemmaCheckpointContract {
    /// Required model fields and shapes, in deterministic topology order.
    expected: Vec<ExpectedTensor>,
    /// Checkpoint-name to model-field mappings, including the tied-head alias.
    mappings: MappingPlan,
    /// Prefixes of intentionally unexecuted vision tensors.
    allow_unexpected: Vec<KeyMatcher>,
}

impl GemmaCheckpointContract {
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

    /// Returns the exemptions of the omitted vision tower (empty for text-only sources).
    #[must_use]
    pub fn allow_unexpected(&self) -> &[KeyMatcher] {
        &self.allow_unexpected
    }

    /// Strictly compares a discovered manifest with the complete contract.
    ///
    /// The vision exemptions are added to the caller's policy; the caller's
    /// own allowlists and dtype policy are honored.
    ///
    /// # Errors
    ///
    /// Returns the checkpoint planning error for any mismatch.
    pub fn build_load_plan(
        &self,
        manifest: &CheckpointManifest,
        options: &LoadOptions,
    ) -> crate::checkpoint::Result<LoadPlan> {
        let mut options = options.clone();
        for matcher in &self.allow_unexpected {
            if !options.allow_unexpected.contains(matcher) {
                options.allow_unexpected.push(matcher.clone());
            }
        }
        build_load_plan(manifest, &self.expected, &self.mappings, &options)
    }
}

impl GemmaConfig {
    /// Derives the complete parameter contract from the validated topology.
    ///
    /// # Errors
    ///
    /// Returns the configuration's validation error, a [`ConfigError`] at
    /// `final_logit_softcapping` for a multimodal source with a final
    /// soft-cap (see the module documentation), or a [`ConfigError`] when a
    /// dimension cannot be represented by the checkpoint format.
    pub fn checkpoint_contract(
        &self,
        options: GemmaCheckpointOptions,
    ) -> Result<GemmaCheckpointContract, ConfigError> {
        let architecture = self.architecture()?;
        reject_unreferenced_final_softcap(&architecture, options.source)?;
        let mut contract = ContractBuilder::new(Dimensions::of(&architecture)?, options.source);
        contract.add_embedding();
        for layer in 0..architecture.num_layers() {
            contract.add_layer(&architecture, layer);
        }
        contract.add_head(architecture.tie_word_embeddings);
        Ok(contract.finish())
    }
}

/// Refuses a final soft-cap that a multimodal source's reference never applies.
///
/// Text-only sources follow `Gemma{2,3}ForCausalLM`, which cap the logits;
/// the multimodal wrappers return the raw head logits, so executing the cap
/// for them would silently change every logit.
fn reject_unreferenced_final_softcap(
    architecture: &GemmaArchitecture,
    source: GemmaCheckpointSource,
) -> Result<(), ConfigError> {
    match (source, architecture.final_logit_softcap) {
        (
            GemmaCheckpointSource::Multimodal | GemmaCheckpointSource::LegacyMultimodal,
            Some(cap),
        ) => Err(ConfigError::new(
            architecture.variant.model_type(),
            "final_logit_softcapping",
            format!(
                "the {source:?} namespace belongs to a multimodal reference (Gemma3ForConditionalGeneration or PaliGemmaForConditionalGeneration) that never applies the final soft-cap {cap}; set it to null"
            ),
        )),
        _ => Ok(()),
    }
}

/// Checked shape terms of the contract, as checkpoint-format `u64` extents.
#[derive(Debug, Clone, Copy)]
struct Dimensions {
    /// Vocabulary rows `V`.
    vocab: u64,
    /// Residual width `d`.
    hidden: u64,
    /// GeGLU width `f`.
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
    fn of(architecture: &GemmaArchitecture) -> Result<Self, ConfigError> {
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
    /// Published namespace of the source.
    source: GemmaCheckpointSource,
    /// Expected model fields so far.
    expected: Vec<ExpectedTensor>,
    /// Mappings so far.
    mappings: MappingPlan,
}

impl ContractBuilder {
    /// Starts an empty contract for `source`.
    fn new(dimensions: Dimensions, source: GemmaCheckpointSource) -> Self {
        Self {
            dimensions,
            source,
            expected: Vec::new(),
            mappings: MappingPlan::new(),
        }
    }

    /// Adds `<prefix>.<source_suffix>` installed into `model.<target_suffix>`.
    fn language(&mut self, source_suffix: &str, target_suffix: &str, shape: Vec<u64>) {
        let target = format!("model.{target_suffix}");
        self.expected.push(ExpectedTensor::new(&target, shape));
        self.mappings.map(
            format!("{}.{source_suffix}", self.source.language_prefix()),
            target,
        );
    }

    /// Adds the token embedding table.
    fn add_embedding(&mut self) {
        let Dimensions { vocab, hidden, .. } = self.dimensions;
        self.language(
            "embed_tokens.weight",
            "embed_tokens.weight",
            vec![vocab, hidden],
        );
    }

    /// Adds one decoder block: its four norms, attention, Q/K norms, and GeGLU.
    fn add_layer(&mut self, architecture: &GemmaArchitecture, layer: usize) {
        let prefix = format!("layers.{layer}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            let name = format!("{prefix}.{norm}.weight");
            self.language(&name, &name, vec![self.dimensions.hidden]);
        }
        self.add_attention(architecture, &format!("{prefix}.self_attn"));
        self.add_mlp(&format!("{prefix}.mlp"));
    }

    /// Adds the four projections, their enabled biases, and Gemma 3's Q/K norms.
    fn add_attention(&mut self, architecture: &GemmaArchitecture, prefix: &str) {
        let Dimensions {
            hidden,
            query,
            key_value,
            head,
            ..
        } = self.dimensions;
        for (name, rows, columns) in [
            ("q_proj", query, hidden),
            ("k_proj", key_value, hidden),
            ("v_proj", key_value, hidden),
            ("o_proj", hidden, query),
        ] {
            let weight = format!("{prefix}.{name}.weight");
            self.language(&weight, &weight, vec![rows, columns]);
            if architecture.attention_bias {
                let bias = format!("{prefix}.{name}.bias");
                self.language(&bias, &bias, vec![rows]);
            }
        }
        if architecture.query_key_norm {
            self.language(
                &format!("{prefix}.q_norm.weight"),
                &format!("{prefix}.query_norm.weight"),
                vec![head],
            );
            self.language(
                &format!("{prefix}.k_norm.weight"),
                &format!("{prefix}.key_norm.weight"),
                vec![head],
            );
        }
    }

    /// Adds the three bias-free GeGLU projections.
    fn add_mlp(&mut self, prefix: &str) {
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
            let weight = format!("{prefix}.{name}.weight");
            self.language(&weight, &weight, vec![rows, columns]);
        }
    }

    /// Adds the final norm and the independent or tied vocabulary projection.
    fn add_head(&mut self, tied: bool) {
        let Dimensions { vocab, hidden, .. } = self.dimensions;
        self.language("norm.weight", "norm.weight", vec![hidden]);
        self.expected
            .push(ExpectedTensor::new("lm_head.weight", vec![vocab, hidden]));
        if tied {
            self.mappings
                .alias("model.embed_tokens.weight", "lm_head.weight");
        } else {
            self.mappings.map(self.source.head_name(), "lm_head.weight");
        }
    }

    /// Returns the finished contract with the source's vision exemptions.
    fn finish(self) -> GemmaCheckpointContract {
        GemmaCheckpointContract {
            expected: self.expected,
            mappings: self.mappings,
            allow_unexpected: self
                .source
                .omitted_prefixes()
                .iter()
                .map(|prefix| KeyMatcher::prefix(*prefix))
                .collect(),
        }
    }
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
