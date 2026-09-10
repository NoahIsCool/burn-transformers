//! Exact SafeTensors inventory and mapping contract for T5-family checkpoints.
//!
//! Hugging Face T5 checkpoints use the original TensorFlow module names even
//! when the selected activation is GELU or SiLU: every feed-forward container is
//! named `DenseReluDense`. Dense variants store `wi` and `wo`; gated variants
//! store `wi_0`, `wi_1`, and `wo`, where `wi_0` is the nonlinear gate. This
//! module makes those historical names explicit instead of inferring topology
//! from whichever keys happen to be present.
//!
//! T5 has one learned relative-position table per stack, owned by block zero.
//! Later blocks reuse its computed bias and therefore must not acquire their own
//! checkpoint parameters. Burn stores linear weights in `[input, output]` order,
//! so every official `[output, input]` projection is represented by an explicit
//! rank-two transpose in the generic load plan. The logical LM head is always
//! present in the expected inventory. For original tied T5 it aliases
//! `shared.weight`; T5 v1.1, FLAN, and UL2 instead map an independently stored
//! `lm_head.weight` through the same explicit layout conversion.

use std::path::Path;

use crate::checkpoint::{
    CheckpointManifest, ExpectedTensor, IndexTensorAlias, KeyMatcher, LoadOptions, LoadPlan,
    MappingPlan, build_load_plan, discover_checkpoint,
};

use super::{ConfigError, T5Config, Validate};

const FAMILY: &str = "t5";

/// Complete logical parameter inventory and explicit source mapping for T5.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct T5CheckpointContract {
    expected: Vec<ExpectedTensor>,
    mappings: MappingPlan,
    allow_unexpected: Vec<KeyMatcher>,
    index_aliases: Vec<IndexTensorAlias>,
}

impl T5CheckpointContract {
    /// Returns every logical parameter implied by the validated architecture.
    ///
    /// A tied model still contains the logical `lm_head.weight`; its mapping is
    /// an alias and therefore does not allocate a second tensor.
    #[must_use]
    pub fn expected_tensors(&self) -> &[ExpectedTensor] {
        &self.expected
    }

    /// Returns the complete checkpoint-key-to-model-target mapping.
    #[must_use]
    pub fn mappings(&self) -> &MappingPlan {
        &self.mappings
    }

    /// Returns architecture-owned checkpoint keys that may be ignored explicitly.
    ///
    /// Shared embedding copies are not allowlisted: [`Self::index_aliases`]
    /// resolves index-only names and requires physical copies to compare exactly.
    #[must_use]
    pub fn allow_unexpected(&self) -> &[KeyMatcher] {
        &self.allow_unexpected
    }

    /// Returns exact framework aliases that an index may deduplicate from headers.
    #[must_use]
    pub fn index_aliases(&self) -> &[IndexTensorAlias] {
        &self.index_aliases
    }

    /// Adds exact family-owned exceptions and equivalences to load policy.
    ///
    /// Existing resource bounds, dtype conversion, and caller exceptions are
    /// preserved. Duplicate matchers are not added.
    pub fn configure_load_options(&self, options: &mut LoadOptions) {
        for matcher in &self.allow_unexpected {
            if !options.allow_unexpected.contains(matcher) {
                options.allow_unexpected.push(matcher.clone());
            }
        }
        for alias in &self.index_aliases {
            if !options.index_aliases.contains(alias) {
                options.index_aliases.push(alias.clone());
            }
        }
    }

    /// Validates an already discovered manifest without reading tensor payloads.
    ///
    /// Validation is strict: every expected parameter must be produced and every
    /// observed checkpoint key must be consumed unless the caller supplied an
    /// explicit exception through [`LoadOptions`].
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
                    "T5 parameter `{}` must be floating point, planned dtype is {}",
                    tensor.target, tensor.dtype
                ),
            });
        }
        Ok(plan)
    }

    /// Discovers local SafeTensors headers and compiles a validated pass-two plan.
    ///
    /// The returned object keeps the manifest beside the plan so execution cannot
    /// accidentally pair validated offsets with a different repository root.
    pub fn plan_local(
        &self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> crate::checkpoint::Result<T5CheckpointPlan> {
        let mut options = options.clone();
        self.configure_load_options(&mut options);
        let manifest = discover_checkpoint(root, &options)?;
        let plan = self.build_load_plan(&manifest, &options)?;
        Ok(T5CheckpointPlan { manifest, plan })
    }
}

/// A pass-one T5 manifest paired with its strictly validated execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct T5CheckpointPlan {
    manifest: CheckpointManifest,
    plan: LoadPlan,
}

impl T5CheckpointPlan {
    /// Returns the validated local artifact inventory used by this plan.
    #[must_use]
    pub const fn manifest(&self) -> &CheckpointManifest {
        &self.manifest
    }

    /// Returns the validated streaming operations for pass two.
    #[must_use]
    pub const fn load_plan(&self) -> &LoadPlan {
        &self.plan
    }

    /// Decomposes the paired plan for callers that need owned generic values.
    #[must_use]
    pub fn into_parts(self) -> (CheckpointManifest, LoadPlan) {
        (self.manifest, self.plan)
    }
}

impl T5Config {
    /// Derives the exact official SafeTensors inventory from architecture metadata.
    ///
    /// Expected shapes describe Burn's executable parameter layout. Embeddings,
    /// norms, and relative-position tables preserve their serialized axes, while
    /// every independent linear projection maps official `[output, input]`
    /// storage into Burn's gradient-safe `[input, output]` layout through an
    /// auditable [`MappingPlan::transpose_2d`] operation.
    pub fn checkpoint_contract(&self) -> Result<T5CheckpointContract, ConfigError> {
        self.validate()?;
        let dimensions = Dimensions::try_from(self)?;
        let mut builder = ContractBuilder::new();

        builder.tensor("shared.weight", vec![dimensions.vocab, dimensions.model]);
        builder.stack("encoder", self.num_layers, false, self, &dimensions);
        builder.stack("decoder", self.decoder_layers(), true, self, &dimensions);

        if self.tie_word_embeddings {
            builder.expected.push(ExpectedTensor::new(
                "lm_head.weight",
                vec![dimensions.vocab, dimensions.model],
            ));
            builder.mappings.alias("shared.weight", "lm_head.weight");
            builder
                .index_aliases
                .push(IndexTensorAlias::new("lm_head.weight", "shared.weight"));
        } else {
            builder.linear("lm_head.weight", dimensions.model, dimensions.vocab);
        }
        Ok(builder.finish())
    }

    /// Discovers and strictly plans a local SafeTensors checkpoint for this config.
    pub fn plan_checkpoint(
        &self,
        root: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<T5CheckpointPlan, super::T5CheckpointLoadError> {
        let contract = self.checkpoint_contract()?;
        Ok(contract.plan_local(root, options)?)
    }
}

/// Portable checkpoint dimensions converted once after topology validation.
#[derive(Debug, Clone, Copy)]
struct Dimensions {
    vocab: u64,
    model: u64,
    attention: u64,
    feed_forward: u64,
    buckets: u64,
    heads: u64,
}

impl TryFrom<&T5Config> for Dimensions {
    type Error = ConfigError;

    /// Converts `usize` runtime extents to SafeTensors' portable `u64` shape form.
    fn try_from(config: &T5Config) -> Result<Self, Self::Error> {
        Ok(Self {
            vocab: dimension("vocab_size", config.vocab_size)?,
            model: dimension("d_model", config.d_model)?,
            attention: product("num_heads * d_kv", &[config.num_heads, config.d_kv])?,
            feed_forward: dimension("d_ff", config.d_ff)?,
            buckets: dimension(
                "relative_attention_num_buckets",
                config.relative_attention_num_buckets,
            )?,
            heads: dimension("num_heads", config.num_heads)?,
        })
    }
}

/// Accumulates one deterministic inventory and its one-to-one source mappings.
struct ContractBuilder {
    expected: Vec<ExpectedTensor>,
    mappings: MappingPlan,
    allow_unexpected: Vec<KeyMatcher>,
    index_aliases: Vec<IndexTensorAlias>,
}

impl ContractBuilder {
    /// Creates an empty architecture contract.
    fn new() -> Self {
        Self {
            expected: Vec::new(),
            mappings: MappingPlan::new(),
            allow_unexpected: Vec::new(),
            index_aliases: vec![
                IndexTensorAlias::new("encoder.embed_tokens.weight", "shared.weight"),
                IndexTensorAlias::new("decoder.embed_tokens.weight", "shared.weight"),
            ],
        }
    }

    /// Finishes construction without changing topology order.
    fn finish(self) -> T5CheckpointContract {
        T5CheckpointContract {
            expected: self.expected,
            mappings: self.mappings,
            allow_unexpected: self.allow_unexpected,
            index_aliases: self.index_aliases,
        }
    }

    /// Adds one identity-mapped official tensor.
    fn tensor(&mut self, name: impl Into<String>, shape: Vec<u64>) {
        let name = name.into();
        self.expected.push(ExpectedTensor::new(&name, shape));
        self.mappings.map(&name, &name);
    }

    /// Adds one official linear matrix with Burn's runtime `[input, output]` shape.
    ///
    /// SafeTensors stores the same scalar matrix in `[output, input]` order. A
    /// transpose preserves its element count and dtype but changes the stride
    /// interpretation required by Burn's row-layout linear primitive.
    fn linear(&mut self, name: impl Into<String>, input: u64, output: u64) {
        let name = name.into();
        self.expected
            .push(ExpectedTensor::new(&name, vec![input, output]));
        self.mappings.transpose_2d(&name, &name);
    }

    /// Adds every block and final normalization for one encoder or decoder stack.
    fn stack(
        &mut self,
        stack: &str,
        depth: usize,
        decoder: bool,
        config: &T5Config,
        dimensions: &Dimensions,
    ) {
        for index in 0..depth {
            self.self_attention(stack, index, dimensions);
            if index == 0 {
                self.tensor(
                    format!("{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
                    vec![dimensions.buckets, dimensions.heads],
                );
            }
            self.tensor(
                format!("{stack}.block.{index}.layer.0.layer_norm.weight"),
                vec![dimensions.model],
            );

            let feed_forward_layer = if decoder {
                self.cross_attention(stack, index, dimensions);
                2
            } else {
                1
            };
            self.feed_forward(stack, index, feed_forward_layer, config, dimensions);
        }
        self.tensor(
            format!("{stack}.final_layer_norm.weight"),
            vec![dimensions.model],
        );
    }

    /// Adds Q/K/V/O matrices for a stack's self-attention layer.
    fn self_attention(&mut self, stack: &str, index: usize, dimensions: &Dimensions) {
        self.attention(
            &format!("{stack}.block.{index}.layer.0.SelfAttention"),
            dimensions,
        );
    }

    /// Adds decoder cross-attention matrices and its scale-only norm.
    fn cross_attention(&mut self, stack: &str, index: usize, dimensions: &Dimensions) {
        self.attention(
            &format!("{stack}.block.{index}.layer.1.EncDecAttention"),
            dimensions,
        );
        self.tensor(
            format!("{stack}.block.{index}.layer.1.layer_norm.weight"),
            vec![dimensions.model],
        );
    }

    /// Adds one attention module in Burn's runtime `[input, output]` matrix order.
    fn attention(&mut self, prefix: &str, dimensions: &Dimensions) {
        for projection in ["q", "k", "v"] {
            self.linear(
                format!("{prefix}.{projection}.weight"),
                dimensions.model,
                dimensions.attention,
            );
        }
        self.linear(
            format!("{prefix}.o.weight"),
            dimensions.attention,
            dimensions.model,
        );
    }

    /// Adds the dense or gated feed-forward matrices and pre-FFN normalization.
    fn feed_forward(
        &mut self,
        stack: &str,
        index: usize,
        layer: usize,
        config: &T5Config,
        dimensions: &Dimensions,
    ) {
        let prefix = format!("{stack}.block.{index}.layer.{layer}.DenseReluDense");
        if config.feed_forward_proj.is_gated() {
            for input in ["wi_0", "wi_1"] {
                self.linear(
                    format!("{prefix}.{input}.weight"),
                    dimensions.model,
                    dimensions.feed_forward,
                );
            }
        } else {
            self.linear(
                format!("{prefix}.wi.weight"),
                dimensions.model,
                dimensions.feed_forward,
            );
        }
        self.linear(
            format!("{prefix}.wo.weight"),
            dimensions.feed_forward,
            dimensions.model,
        );
        self.tensor(
            format!("{stack}.block.{index}.layer.{layer}.layer_norm.weight"),
            vec![dimensions.model],
        );
    }
}

/// Converts one runtime extent into the SafeTensors shape representation.
fn dimension(field: &'static str, value: usize) -> Result<u64, ConfigError> {
    u64::try_from(value).map_err(|_| {
        ConfigError::new(
            FAMILY,
            field,
            "does not fit the portable checkpoint dimension representation",
        )
    })
}

/// Multiplies dimensions with a checked portable accumulator.
fn product(field: &'static str, values: &[usize]) -> Result<u64, ConfigError> {
    values.iter().try_fold(1_u64, |accumulator, value| {
        let value = dimension(field, *value)?;
        accumulator.checked_mul(value).ok_or_else(|| {
            ConfigError::new(FAMILY, field, "overflows checkpoint tensor dimensions")
        })
    })
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
