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
//! checkpoint parameters. The logical LM head is always present in the expected
//! inventory. For original tied T5 it aliases `shared.weight`; T5 v1.1, FLAN,
//! and UL2 instead map an independently stored `lm_head.weight`.

use std::path::Path;

use crate::checkpoint::{
    CheckpointManifest, ExpectedTensor, KeyMatcher, LoadOptions, LoadPlan, MappingPlan,
    build_load_plan, discover_checkpoint,
};

use super::{ConfigError, T5Config, Validate};

const FAMILY: &str = "t5";

/// Complete logical parameter inventory and explicit source mapping for T5.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct T5CheckpointContract {
    expected: Vec<ExpectedTensor>,
    mappings: MappingPlan,
    allow_unexpected: Vec<KeyMatcher>,
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

    /// Returns known serialized aliases that do not own model storage.
    ///
    /// Hugging Face exports may repeat `shared.weight` under the encoder and
    /// decoder embedding paths. Tied checkpoints may additionally serialize an
    /// LM-head copy. These exact keys are ignored because the executable model
    /// has one canonical shared table; they remain explicit in [`crate::checkpoint::LoadReport`].
    #[must_use]
    pub fn allow_unexpected(&self) -> &[KeyMatcher] {
        &self.allow_unexpected
    }

    /// Adds exact family-owned alias exceptions to caller-supplied load policy.
    ///
    /// Existing resource bounds, dtype conversion, and caller exceptions are
    /// preserved. Duplicate matchers are not added.
    pub fn configure_load_options(&self, options: &mut LoadOptions) {
        for matcher in &self.allow_unexpected {
            if !options.allow_unexpected.contains(matcher) {
                options.allow_unexpected.push(matcher.clone());
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
        let manifest = discover_checkpoint(root, options)?;
        let plan = self.build_load_plan(&manifest, options)?;
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
    /// Shapes use checkpoint order. Burn column-layout linear layers save weights
    /// as `[output, input]`, so the generic loader performs identity transforms and
    /// Burn's parameter record mapper applies the runtime transpose during install.
    pub fn checkpoint_contract(&self) -> Result<T5CheckpointContract, ConfigError> {
        self.validate()?;
        let dimensions = Dimensions::try_from(self)?;
        let mut builder = ContractBuilder::new();

        builder.tensor("shared.weight", vec![dimensions.vocab, dimensions.model]);
        builder.stack("encoder", self.num_layers, false, self, &dimensions);
        builder.stack("decoder", self.decoder_layers(), true, self, &dimensions);

        builder.expected.push(ExpectedTensor::new(
            "lm_head.weight",
            vec![dimensions.vocab, dimensions.model],
        ));
        if self.tie_word_embeddings {
            builder.mappings.alias("shared.weight", "lm_head.weight");
            builder
                .allow_unexpected
                .push(KeyMatcher::exact("lm_head.weight"));
        } else {
            builder.mappings.map("lm_head.weight", "lm_head.weight");
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
}

impl ContractBuilder {
    /// Creates an empty architecture contract.
    fn new() -> Self {
        Self {
            expected: Vec::new(),
            mappings: MappingPlan::new(),
            allow_unexpected: vec![
                KeyMatcher::exact("encoder.embed_tokens.weight"),
                KeyMatcher::exact("decoder.embed_tokens.weight"),
            ],
        }
    }

    /// Finishes construction without changing topology order.
    fn finish(self) -> T5CheckpointContract {
        T5CheckpointContract {
            expected: self.expected,
            mappings: self.mappings,
            allow_unexpected: self.allow_unexpected,
        }
    }

    /// Adds one identity-mapped official tensor.
    fn tensor(&mut self, name: impl Into<String>, shape: Vec<u64>) {
        let name = name.into();
        self.expected.push(ExpectedTensor::new(&name, shape));
        self.mappings.map(&name, &name);
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

    /// Adds one attention module in official `[output, input]` matrix order.
    fn attention(&mut self, prefix: &str, dimensions: &Dimensions) {
        for projection in ["q", "k", "v"] {
            self.tensor(
                format!("{prefix}.{projection}.weight"),
                vec![dimensions.attention, dimensions.model],
            );
        }
        self.tensor(
            format!("{prefix}.o.weight"),
            vec![dimensions.model, dimensions.attention],
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
                self.tensor(
                    format!("{prefix}.{input}.weight"),
                    vec![dimensions.feed_forward, dimensions.model],
                );
            }
        } else {
            self.tensor(
                format!("{prefix}.wi.weight"),
                vec![dimensions.feed_forward, dimensions.model],
            );
        }
        self.tensor(
            format!("{prefix}.wo.weight"),
            vec![dimensions.model, dimensions.feed_forward],
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
mod tests {
    use std::{collections::BTreeMap, path::PathBuf};

    use crate::{
        checkpoint::{
            ArtifactSource, CheckpointError, CheckpointManifest, KeyMatcher, LoadOptions,
            PlannedTransform, TensorDType, TensorInventory, TensorOperation,
        },
        models::config::ModelConfig,
    };

    use super::*;

    /// Returns a deliberately asymmetric tiny topology for ownership tests.
    fn tiny(projection: super::super::T5FeedForwardProjection, tied: bool) -> T5Config {
        T5Config {
            vocab_size: 11,
            d_model: 4,
            d_kv: 2,
            d_ff: 6,
            num_layers: 2,
            num_decoder_layers: Some(1),
            num_heads: 2,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 16,
            feed_forward_proj: projection,
            dense_act_fn: None,
            is_gated_act: None,
            tie_word_embeddings: tied,
            ..T5Config::default()
        }
    }

    /// Builds a metadata-only manifest matching the contract's non-alias sources.
    fn manifest(contract: &T5CheckpointContract) -> CheckpointManifest {
        let alias_targets = contract
            .mappings()
            .operations()
            .iter()
            .filter_map(|operation| match operation {
                TensorOperation::Alias { target, .. } => Some(target.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut tensors = BTreeMap::new();
        let mut offset = 0_u64;
        for expected in contract.expected_tensors() {
            if alias_targets.contains(&expected.name.as_str()) {
                continue;
            }
            let elements = expected.shape.iter().product();
            let byte_len = TensorDType::F32.storage_bytes(elements).unwrap();
            let end = offset.checked_add(byte_len).unwrap();
            tensors.insert(
                expected.name.clone(),
                TensorInventory {
                    name: expected.name.clone(),
                    shard: "metadata-only.safetensors".into(),
                    shape: expected.shape.clone(),
                    dtype: TensorDType::F32,
                    data_offsets: [offset, end],
                    byte_len,
                },
            );
            offset = end;
        }
        CheckpointManifest {
            root: PathBuf::from("metadata-only"),
            source: ArtifactSource::Single {
                relative_path: "metadata-only.safetensors".into(),
            },
            shards: Vec::new(),
            tensors,
            tensor_bytes: offset,
            artifact_bytes: offset,
        }
    }

    /// Finds one expected tensor and returns its checkpoint-order shape.
    fn shape<'a>(contract: &'a T5CheckpointContract, name: &str) -> &'a [u64] {
        &contract
            .expected_tensors()
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap()
            .shape
    }

    #[test]
    fn dense_and_gated_variants_have_exact_historical_keys() {
        use super::super::T5FeedForwardProjection::*;

        for projection in [Relu, Gelu] {
            let contract = tiny(projection, false).checkpoint_contract().unwrap();
            let names = contract
                .expected_tensors()
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>();
            assert!(names.contains(&"encoder.block.0.layer.1.DenseReluDense.wi.weight"));
            assert!(!names.iter().any(|name| name.ends_with("wi_0.weight")));
            assert_eq!(contract.expected_tensors().len(), 35);
        }

        for projection in [GatedGelu, GatedGeluNew, GatedSilu] {
            let contract = tiny(projection, false).checkpoint_contract().unwrap();
            let names = contract
                .expected_tensors()
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>();
            assert!(names.contains(&"encoder.block.0.layer.1.DenseReluDense.wi_0.weight"));
            assert!(names.contains(&"encoder.block.0.layer.1.DenseReluDense.wi_1.weight"));
            assert!(
                !names
                    .iter()
                    .any(|name| name.ends_with("DenseReluDense.wi.weight"))
            );
            assert_eq!(contract.expected_tensors().len(), 38);
        }
    }

    #[test]
    fn relative_bias_is_owned_only_by_each_stack_block_zero() {
        let contract = tiny(super::super::T5FeedForwardProjection::GatedGelu, false)
            .checkpoint_contract()
            .unwrap();
        let bias = contract
            .expected_tensors()
            .iter()
            .filter(|tensor| tensor.name.ends_with("relative_attention_bias.weight"))
            .collect::<Vec<_>>();
        assert_eq!(bias.len(), 2);
        assert_eq!(
            bias[0].name,
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        );
        assert_eq!(
            bias[1].name,
            "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        );
        assert_eq!(bias[0].shape, [8, 2]);
    }

    #[test]
    fn attention_and_feed_forward_shapes_follow_t5_projection_widths() {
        let contract = tiny(super::super::T5FeedForwardProjection::GatedGelu, false)
            .checkpoint_contract()
            .unwrap();
        assert_eq!(
            shape(&contract, "encoder.block.1.layer.0.SelfAttention.q.weight"),
            [4, 4]
        );
        assert_eq!(
            shape(
                &contract,
                "decoder.block.0.layer.1.EncDecAttention.o.weight"
            ),
            [4, 4]
        );
        assert_eq!(
            shape(
                &contract,
                "decoder.block.0.layer.2.DenseReluDense.wi_0.weight"
            ),
            [6, 4]
        );
        assert_eq!(
            shape(
                &contract,
                "decoder.block.0.layer.2.DenseReluDense.wo.weight"
            ),
            [4, 6]
        );
    }

    #[test]
    fn tied_head_is_a_logical_alias_while_untied_head_consumes_storage() {
        let tied = tiny(super::super::T5FeedForwardProjection::Relu, true)
            .checkpoint_contract()
            .unwrap();
        let tied_plan = tied
            .build_load_plan(&manifest(&tied), &LoadOptions::default())
            .unwrap();
        assert_eq!(tied_plan.parameter_count(), 35);
        assert_eq!(tied_plan.tensors.len(), 34);
        assert!(matches!(
            tied_plan.aliases.as_slice(),
            [alias]
                if alias.target == "lm_head.weight"
                    && matches!(
                        &alias.transform,
                        PlannedTransform::Alias { source_target }
                            if source_target == "shared.weight"
                    )
        ));

        let untied = tiny(super::super::T5FeedForwardProjection::Relu, false)
            .checkpoint_contract()
            .unwrap();
        let untied_plan = untied
            .build_load_plan(&manifest(&untied), &LoadOptions::default())
            .unwrap();
        assert_eq!(untied_plan.tensors.len(), 35);
        assert!(untied_plan.aliases.is_empty());
    }

    #[test]
    fn known_hugging_face_embedding_aliases_are_exact_and_reported() {
        for tied in [false, true] {
            let contract = tiny(super::super::T5FeedForwardProjection::GatedGelu, tied)
                .checkpoint_contract()
                .unwrap();
            let mut observed = manifest(&contract);
            let shared = observed.tensors["shared.weight"].clone();
            for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"] {
                observed.tensors.insert(
                    name.into(),
                    TensorInventory {
                        name: name.into(),
                        ..shared.clone()
                    },
                );
            }
            if tied {
                observed.tensors.insert(
                    "lm_head.weight".into(),
                    TensorInventory {
                        name: "lm_head.weight".into(),
                        ..shared
                    },
                );
            }

            let plan = contract
                .build_load_plan(&observed, &LoadOptions::default())
                .unwrap();
            assert_eq!(
                plan.report.allowlisted_ignored.len(),
                if tied { 3 } else { 2 }
            );
            assert!(
                plan.report
                    .allowlisted_ignored
                    .contains(&"encoder.embed_tokens.weight".into())
            );
            assert_eq!(plan.parameter_count(), 38);

            let custom = KeyMatcher::exact("local.optional.weight");
            let mut options = LoadOptions {
                allow_unexpected: vec![
                    KeyMatcher::exact("encoder.embed_tokens.weight"),
                    custom.clone(),
                ],
                ..LoadOptions::default()
            };
            contract.configure_load_options(&mut options);
            assert_eq!(
                options
                    .allow_unexpected
                    .iter()
                    .filter(|matcher| {
                        **matcher == KeyMatcher::exact("encoder.embed_tokens.weight")
                    })
                    .count(),
                1
            );
            assert!(options.allow_unexpected.contains(&custom));
        }
    }

    #[test]
    fn strict_planning_reports_missing_unexpected_and_shape_errors() {
        let contract = tiny(super::super::T5FeedForwardProjection::GatedSilu, false)
            .checkpoint_contract()
            .unwrap();

        let mut missing = manifest(&contract);
        missing.tensors.remove("encoder.final_layer_norm.weight");
        assert!(matches!(
            contract.build_load_plan(&missing, &LoadOptions::default()),
            Err(CheckpointError::MissingMappingSource { tensor })
                if tensor == "encoder.final_layer_norm.weight"
        ));

        let mut unexpected = manifest(&contract);
        let template = unexpected.tensors.values().next().unwrap().clone();
        unexpected.tensors.insert(
            "encoder.block.1.forbidden.weight".into(),
            TensorInventory {
                name: "encoder.block.1.forbidden.weight".into(),
                ..template
            },
        );
        assert!(matches!(
            contract.build_load_plan(&unexpected, &LoadOptions::default()),
            Err(CheckpointError::StrictValidation { unexpected: 1, .. })
        ));

        let mut wrong_shape = manifest(&contract);
        wrong_shape
            .tensors
            .get_mut("decoder.block.0.layer.1.EncDecAttention.k.weight")
            .unwrap()
            .shape = vec![5, 4];
        assert!(matches!(
            contract.build_load_plan(&wrong_shape, &LoadOptions::default()),
            Err(CheckpointError::ShapeMismatch { tensor, .. })
                if tensor == "decoder.block.0.layer.1.EncDecAttention.k.weight"
        ));

        let mut integer = manifest(&contract);
        let shared = integer.tensors.get_mut("shared.weight").unwrap();
        shared.dtype = TensorDType::I32;
        assert!(matches!(
            contract.build_load_plan(&integer, &LoadOptions::default()),
            Err(CheckpointError::InvalidMapping { message })
                if message.contains("shared.weight") && message.contains("I32")
        ));
    }

    #[test]
    fn official_reduced_flan_and_ul2_configs_generate_complete_contracts() {
        let flan = T5Config::from_json_str(include_str!("../../../tests/fixtures/flan_t5_xl.json"))
            .unwrap();
        let flan_contract = flan.checkpoint_contract().unwrap();
        assert_eq!(flan_contract.expected_tensors().len(), 558);
        assert!(
            flan_contract
                .mappings()
                .operations()
                .iter()
                .all(|operation| { !matches!(operation, TensorOperation::Alias { .. }) })
        );
        assert_eq!(
            shape(
                &flan_contract,
                "decoder.block.23.layer.2.DenseReluDense.wi_0.weight"
            ),
            [5120, 2048]
        );

        let ul2 =
            T5Config::from_json_str(include_str!("../../../tests/fixtures/ul2.json")).unwrap();
        let ul2_contract = ul2.checkpoint_contract().unwrap();
        assert_eq!(ul2_contract.expected_tensors().len(), 742);
        assert!(matches!(
            ul2_contract.mappings().operations().last(),
            Some(TensorOperation::Alias { source_target, target })
                if source_target == "shared.weight" && target == "lm_head.weight"
        ));
    }

    #[test]
    fn dimension_overflow_is_reported_before_inventory_allocation() {
        let config = T5Config {
            d_model: 1,
            d_kv: 2,
            d_ff: 1,
            vocab_size: 1,
            num_heads: usize::MAX,
            num_layers: 1,
            num_decoder_layers: Some(1),
            relative_attention_num_buckets: 4,
            relative_attention_max_distance: 2,
            ..T5Config::default()
        };
        let error = config.checkpoint_contract().unwrap_err();
        assert!(error.reason.contains("overflows"));
    }
}
