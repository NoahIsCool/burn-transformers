//! Small deterministic fixtures exercise recipe edits without external weights.

use std::{collections::BTreeSet, fs, path::Path};

use burn::{
    backend::Flex,
    tensor::{DType, Int, Tensor},
};
use serde_json::{Map, json};

use super::*;
use crate::{
    CausalLanguageModel, CausalLmInput, ForwardOptions, LogitsSelection,
    cache::CacheConfig,
    checkpoint::{TensorTransform, execute_local_load_plan},
    models::qwen3_5::Qwen3_5ForCausalLm,
};

/// Deterministic CPU tensor backend for small checkpoint fixtures.
type B = Flex<f32, i32>;

/// Builds one DeltaNet and one full-attention block with tied vocabulary weights.
fn tiny_config() -> Qwen3_5TextConfig {
    Qwen3_5TextConfig {
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        layer_types: vec![
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ],
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_conv_kernel_dim: 3,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        tie_word_embeddings: true,
        mtp_num_hidden_layers: 0,
        ..Default::default()
    }
}

/// Supplies three unpadded tokens to exercise both sequence-mixer types.
fn input() -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_ints([[1, 2, 3]], &Default::default()),
        token_mask: None,
        position_ids: None,
    }
}

/// Writes deterministic F32 SafeTensors and optionally one unexplained source key.
fn fixture(root: &Path, config: &Qwen3_5TextConfig, extra: bool) {
    fs::write(
        root.join("config.json"),
        serde_json::to_vec(config).unwrap(),
    )
    .unwrap();
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: false,
        })
        .unwrap();
    let mut header = Map::new();
    let mut payload = Vec::new();
    for (order, operation) in contract.mappings().operations().iter().enumerate() {
        let TensorOperation::Map {
            source,
            target,
            transform,
        } = operation
        else {
            continue;
        };
        assert_eq!(*transform, TensorTransform::Identity);
        let shape = &contract
            .expected_tensors()
            .iter()
            .find(|tensor| tensor.name == *target)
            .unwrap()
            .shape;
        let start = payload.len();
        for index in 0..shape.iter().product::<u64>() as usize {
            let value = (((index * 7 + order * 11) % 31) as f32 - 15.0) * 0.01;
            payload.extend(value.to_le_bytes());
        }
        header.insert(
            source.clone(),
            json!({"dtype":"F32", "shape":shape, "data_offsets":[start,payload.len()]}),
        );
    }
    if extra {
        let start = payload.len();
        payload.extend(0.0_f32.to_le_bytes());
        header.insert(
            "unexplained.weight".into(),
            json!({"dtype":"F32", "shape":[1], "data_offsets":[start,payload.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    fs::write(root.join("model.safetensors"), bytes).unwrap();
}

/// Loads every fixture parameter through the ordinary family contract for comparison.
fn original(root: &Path, config: Qwen3_5TextConfig) -> Qwen3_5ForCausalLm<B> {
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: false,
        })
        .unwrap();
    let manifest = CheckpointManifest::discover(root, &LoadOptions::default()).unwrap();
    let plan = contract
        .build_load_plan(&manifest, &LoadOptions::default())
        .unwrap();
    let mut model = Qwen3_5ForCausalLm::<B>::init(config, &Default::default()).unwrap();
    execute_local_load_plan(&manifest, &plan, &mut model).unwrap();
    model
}

/// Unedited recipes preserve backbone states and a vocabulary view tied to live embeddings.
#[test]
fn unchanged_recipe_matches_loaded_family_backbone_and_live_tied_logits() {
    let directory = tempfile::tempdir().unwrap();
    let config = tiny_config();
    fixture(directory.path(), &config, false);
    let family = original(directory.path(), config);
    let recipe = Qwen3_5BackboneRecipe::from_local_config(directory.path()).unwrap();
    let loaded = recipe
        .load_pretrained::<B>(directory.path(), &Default::default(), &Default::default())
        .unwrap();
    let (hidden, _) = loaded.model.forward(&input(), false).unwrap();
    let output = family
        .forward(
            input(),
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        hidden.to_data(),
        output.final_hidden_state.unwrap().to_data()
    );
    assert_eq!(
        loaded
            .model
            .tied_vocabulary_projection(hidden)
            .unwrap()
            .to_data(),
        output.logits.to_data()
    );
    assert!(loaded.report.newly_initialized.is_empty());
    assert!(
        loaded
            .report
            .omitted
            .iter()
            .any(|item| item.name == "lm_head.weight")
    );
}

/// Prefix loading reads exactly its selected payload bytes and permits a custom task head.
#[test]
fn prefix_reads_only_selected_backbone_parameters_and_can_use_a_custom_head() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), &tiny_config(), false);
    let recipe = Qwen3_5BackboneRecipe::from_local_config(directory.path())
        .unwrap()
        .prefix(1)
        .unwrap();
    let loaded = recipe
        .load_pretrained::<B>(directory.path(), &Default::default(), &Default::default())
        .unwrap();
    assert_eq!(loaded.model.layers.len(), 1);
    assert!(
        loaded
            .report
            .reused
            .iter()
            .all(|item| !item.target.starts_with("model.layers.1."))
    );
    assert!(
        loaded
            .report
            .omitted
            .iter()
            .any(|item| item.name.starts_with("model.layers.1."))
    );
    let manifest = CheckpointManifest::discover(directory.path(), &Default::default()).unwrap();
    let expected_bytes: u64 = loaded
        .report
        .reused
        .iter()
        .map(|item| manifest.tensor(&item.source).unwrap().byte_len)
        .sum();
    assert_eq!(loaded.checkpoint.stats.source_payload_bytes, expected_bytes);
    assert!(expected_bytes < manifest.tensor_bytes);
    let state = loaded.model.embed(&input()).unwrap();
    let state = loaded.model.run_layers(state, 0..1).unwrap();
    let state = loaded.model.normalize(state).unwrap();
    let head = burn::nn::LinearConfig::new(8, 2).init::<B>(&Default::default());
    assert_eq!(head.forward(state.hidden.mean_dim(1)).dims(), [1, 1, 2]);
}

/// Changed MLP shapes require explicit reinitialization while compatible norms retain weights.
#[test]
fn changed_feed_forward_width_requires_explicit_reinitialization_and_is_audited() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), &tiny_config(), false);
    let mut recipe = Qwen3_5BackboneRecipe::from_local_config(directory.path()).unwrap();
    recipe.layers[0].intermediate_size = 20;
    let error = match recipe.load_pretrained::<B>(
        directory.path(),
        &Default::default(),
        &Default::default(),
    ) {
        Ok(_) => panic!("shape change must be explicit"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("Reinitialize"));
    let loaded = recipe
        .load_pretrained::<B>(
            directory.path(),
            &Qwen3_5RecipeLoadOptions {
                shape_changes: ShapeChangePolicy::Reinitialize,
                ..Default::default()
            },
            &Default::default(),
        )
        .unwrap();
    assert_eq!(loaded.report.newly_initialized.len(), 3);
    assert!(
        loaded
            .report
            .newly_initialized
            .iter()
            .all(|item| item.name.starts_with("model.layers.0.mlp."))
    );
    assert!(
        loaded
            .report
            .reused
            .iter()
            .any(|item| item.target == "model.layers.0.input_layernorm.weight")
    );
    assert_eq!(
        loaded.model.layers[0].mlp.gate_proj.weight.val().dims(),
        [8, 20]
    );
    assert!(loaded.model.forward(&input(), false).is_ok());
}

/// Replacing DeltaNet with full attention initializes absent projections and reuses its MLP.
#[test]
fn changed_sequence_mixer_reuses_only_existing_compatible_source_parameters() {
    let directory = tempfile::tempdir().expect("fixture directory");
    fixture(directory.path(), &tiny_config(), false);
    let mut recipe = Qwen3_5BackboneRecipe::from_local_config(directory.path())
        .expect("fixture source topology");
    recipe.layers[0].layer_type = Qwen3_5LayerType::FullAttention;
    let loaded = recipe
        .load_pretrained::<B>(directory.path(), &Default::default(), &Default::default())
        .expect("a new mixer may initialize absent source components");
    assert!(!loaded.report.newly_initialized.is_empty());
    assert!(loaded.report.newly_initialized.iter().all(|item| {
        item.name.starts_with("model.layers.0.self_attn.")
            && item.reason.contains("no selected pretrained source")
    }));
    assert!(
        loaded
            .report
            .omitted
            .iter()
            .any(|item| { item.name.starts_with("model.layers.0.linear_attn.") })
    );
    let source = original(directory.path(), tiny_config());
    assert_eq!(
        loaded.model.layers[0].mlp.up_proj.weight.val().to_data(),
        source.model.layers[0].mlp.up_proj.weight.val().to_data(),
    );
    let (hidden, _) = loaded
        .model
        .forward(&input(), false)
        .expect("edited backbone");
    assert!(
        hidden
            .to_data()
            .to_vec::<f32>()
            .expect("F32 states")
            .iter()
            .all(|value| value.is_finite())
    );
}

/// Random recipes execute heterogeneous rotary geometries and an extra block without weights.
#[test]
fn random_recipe_supports_heterogeneous_attention_heads_rotary_widths_and_extra_blocks() {
    let mut recipe = Qwen3_5BackboneRecipe::from_config(tiny_config()).unwrap();
    recipe.layers[0].layer_type = Qwen3_5LayerType::FullAttention;
    recipe.layers[0].num_attention_heads = 3;
    recipe.layers[0].head_dim = 2;
    recipe.layers[0].intermediate_size = 19;
    let mut extra = recipe.layers[1].clone();
    extra.num_attention_heads = 4;
    recipe.layers.push(extra);
    let model = recipe.init_random::<B>(&Default::default()).unwrap();
    assert_eq!(model.layers.len(), 3);
    assert_eq!(model.layers[0].self_attn.as_ref().unwrap().query_heads, 3);
    assert_eq!(model.layers[2].self_attn.as_ref().unwrap().query_heads, 4);
    assert_ne!(model.layer_rope[0], model.layer_rope[1]);
    let (hidden, _) = model.forward(&input(), false).unwrap();
    assert_eq!(hidden.dims(), [1, 3, 8]);
}

/// Reordered execution loads weights from each block's explicit original source index.
#[test]
fn source_layer_reordering_reuses_the_explicit_original_parameters() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), &tiny_config(), false);
    let mut recipe = Qwen3_5BackboneRecipe::from_local_config(directory.path()).unwrap();
    recipe.layers.swap(0, 1);
    let loaded = recipe
        .load_pretrained::<B>(directory.path(), &Default::default(), &Default::default())
        .unwrap();
    assert!(loaded.report.reused.iter().any(|item| item.source
        == "model.layers.1.self_attn.q_proj.weight"
        && item.target == "model.layers.0.self_attn.q_proj.weight"));
    assert!(loaded.model.forward(&input(), false).is_ok());
}

/// Appended blocks with no source keep random initialization without requesting missing tensors.
#[test]
fn newly_appended_blocks_are_initialized_without_requesting_nonexistent_checkpoint_layers() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), &tiny_config(), false);
    let mut recipe = Qwen3_5BackboneRecipe::from_local_config(directory.path()).unwrap();
    let mut extra = recipe.layers[1].clone();
    extra.source_layer = None;
    recipe.layers.push(extra);
    let loaded = recipe
        .load_pretrained::<B>(directory.path(), &Default::default(), &Default::default())
        .unwrap();
    assert!(!loaded.report.newly_initialized.is_empty());
    assert!(
        loaded
            .report
            .newly_initialized
            .iter()
            .all(|item| item.name.starts_with("model.layers.2."))
    );
    assert!(loaded.model.forward(&input(), false).is_ok());
}

/// Rewrites fixture payloads and metadata to BF16 while retaining parameter shapes and names.
fn convert_fixture_to_bfloat16(root: &Path) {
    let path = root.join("model.safetensors");
    let original = fs::read(&path).unwrap();
    let header_length = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    let header: Map<String, serde_json::Value> =
        serde_json::from_slice(&original[8..8 + header_length]).unwrap();
    let source = &original[8 + header_length..];
    let mut converted_header = Map::new();
    let mut converted = Vec::new();
    for (name, mut entry) in header {
        let start = entry["data_offsets"][0].as_u64().unwrap() as usize;
        let end = entry["data_offsets"][1].as_u64().unwrap() as usize;
        let converted_start = converted.len();
        for scalar in source[start..end].chunks_exact(4) {
            let bits = u32::from_le_bytes(scalar.try_into().unwrap());
            converted.extend(((bits >> 16) as u16).to_le_bytes());
        }
        entry["dtype"] = json!("BF16");
        entry["data_offsets"] = json!([converted_start, converted.len()]);
        converted_header.insert(name, entry);
    }
    let mut header = serde_json::to_vec(&converted_header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(converted);
    fs::write(path, bytes).unwrap();
}

/// Requested checkpoint dtype changes move parameters from reuse to the transformation report.
#[test]
fn streamed_dtype_conversions_are_reported_as_transformations() {
    use crate::checkpoint::{DTypePolicy, TensorDType};

    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), &tiny_config(), false);
    convert_fixture_to_bfloat16(directory.path());

    let recipe = Qwen3_5BackboneRecipe::from_local_config(directory.path())
        .unwrap()
        .prefix(1)
        .unwrap();
    let loaded = recipe
        .load_pretrained::<B>(
            directory.path(),
            &Qwen3_5RecipeLoadOptions {
                checkpoint: LoadOptions {
                    dtype_policy: DTypePolicy::Cast(TensorDType::F32),
                    ..Default::default()
                },
                ..Default::default()
            },
            &Default::default(),
        )
        .unwrap();
    assert!(loaded.report.reused.is_empty());
    assert_eq!(
        loaded.report.transformed.len(),
        loaded.checkpoint.stats.tensors_applied
    );
    assert!(
        loaded
            .report
            .transformed
            .iter()
            .all(|item| item.operation.contains("dtype conversion"))
    );
    assert!(loaded.model.forward(&input(), false).is_ok());
}

/// Newly initialized owners follow either preserved BF16 or explicitly requested F32 source dtype.
#[test]
fn reinitialized_parameters_follow_preserved_or_explicit_checkpoint_dtype() {
    use crate::checkpoint::DTypePolicy;

    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), &tiny_config(), false);
    convert_fixture_to_bfloat16(directory.path());
    let mut recipe = Qwen3_5BackboneRecipe::from_local_config(directory.path())
        .unwrap()
        .prefix(1)
        .unwrap();
    recipe.layers[0].intermediate_size = 14;
    for (policy, expected_dtype) in [
        (DTypePolicy::Preserve, DType::BF16),
        (DTypePolicy::Cast(TensorDType::F64), DType::F64),
    ] {
        let loaded = recipe
            .load_pretrained::<B>(
                directory.path(),
                &Qwen3_5RecipeLoadOptions {
                    checkpoint: LoadOptions {
                        dtype_policy: policy,
                        ..Default::default()
                    },
                    shape_changes: ShapeChangePolicy::Reinitialize,
                    ..Default::default()
                },
                &Default::default(),
            )
            .unwrap();
        assert_eq!(loaded.report.newly_initialized.len(), 3);
        assert_eq!(
            loaded.model.embed_tokens.weight.val().dtype(),
            expected_dtype
        );
        assert_eq!(
            loaded.model.layers[0].mlp.gate_proj.weight.val().dtype(),
            expected_dtype
        );
        assert_eq!(
            loaded.model.layers[0].mlp.up_proj.weight.val().dtype(),
            expected_dtype
        );
        assert_eq!(
            loaded.model.layers[0].mlp.down_proj.weight.val().dtype(),
            expected_dtype
        );
        let state = loaded.model.embed(&input()).unwrap();
        let state = loaded.model.run_layers(state, 0..1).unwrap();
        assert_eq!(
            loaded.model.normalize(state).unwrap().hidden.dtype(),
            expected_dtype
        );
    }
}

/// Collects the fully qualified Burn path and dtype of every floating parameter.
///
/// Paths are prefixed with `model.` exactly as the recipe's dtype mapper and
/// the checkpoint contract name backbone parameters.
struct FloatParameterPaths {
    /// Module components entered so far.
    path: Vec<String>,
    /// `(path, dtype)` of each visited floating parameter, in visit order.
    parameters: Vec<(String, DType)>,
}

impl burn::module::ModuleVisitor<B> for FloatParameterPaths {
    /// Pushes a module name onto the current path.
    fn enter_module(&mut self, name: &str, _: &str) {
        self.path.push(name.into());
    }

    /// Pops the module that just finished.
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }

    /// Records the qualified path and dtype of one floating parameter.
    fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<B, D>>) {
        self.parameters.push((
            format!("model.{}", self.path.join(".")),
            param.val().dtype(),
        ));
    }
}

/// Returns `(path, dtype)` for every floating parameter of a backbone.
fn float_parameters(model: &Qwen3_5TextModel<B>) -> Vec<(String, DType)> {
    let mut visitor = FloatParameterPaths {
        path: Vec::new(),
        parameters: Vec::new(),
    };
    model.visit(&mut visitor);
    visitor.parameters
}

/// Every checkpoint name maps onto exactly one Burn parameter path, and back.
///
/// The recipe's dtype mapper identifies new parameters by Burn path while
/// the transfer report names them by checkpoint tensor; this pins the
/// translation table for both mixer kinds so a renamed field cannot silently
/// escape dtype selection again (QW-8).
#[test]
fn checkpoint_names_translate_to_every_backbone_parameter_path() {
    let config = tiny_config();
    let model = Qwen3_5TextModel::<B>::init(config.clone(), &Default::default()).unwrap();
    let module_paths = float_parameters(&model)
        .into_iter()
        .map(|(path, _)| path)
        .collect::<BTreeSet<_>>();
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: false,
        })
        .unwrap();
    let translated = contract
        .expected_tensors()
        .iter()
        .filter(|tensor| tensor.name.starts_with("model."))
        .map(|tensor| module_parameter_path(&tensor.name))
        .collect::<BTreeSet<_>>();
    assert_eq!(translated, module_paths);
    for (checkpoint, module) in [
        (
            "model.layers.1.self_attn.q_norm.weight",
            "model.layers.1.self_attn.query_norm.weight",
        ),
        (
            "model.layers.1.self_attn.k_norm.weight",
            "model.layers.1.self_attn.key_norm.weight",
        ),
        (
            "model.layers.0.linear_attn.A_log",
            "model.layers.0.linear_attn.a_log",
        ),
        (
            "model.layers.0.linear_attn.dt_bias",
            "model.layers.0.linear_attn.dt_bias",
        ),
    ] {
        assert_eq!(module_parameter_path(checkpoint), module);
    }
}

/// Fresh attention and DeltaNet blocks adopt the checkpoint dtype in every parameter (QW-8).
///
/// Q/K RMSNorm scales and `A_log` used to keep the backend default F32
/// because their checkpoint names differ from their Burn field names, while
/// the report claimed they were initialized in BF16.
#[test]
fn appended_attention_and_delta_blocks_initialize_every_parameter_in_checkpoint_dtype() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), &tiny_config(), false);
    convert_fixture_to_bfloat16(directory.path());
    let mut recipe = Qwen3_5BackboneRecipe::from_local_config(directory.path()).unwrap();
    for source in [0, 1] {
        let mut extra = recipe.layers[source].clone();
        extra.source_layer = None;
        recipe.layers.push(extra);
    }
    assert_eq!(
        recipe
            .layers
            .iter()
            .map(|layer| layer.layer_type)
            .collect::<Vec<_>>(),
        [
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ]
    );
    let loaded = recipe
        .load_pretrained::<B>(directory.path(), &Default::default(), &Default::default())
        .unwrap();
    let new_names = loaded
        .report
        .newly_initialized
        .iter()
        .map(|item| item.name.as_str())
        .collect::<BTreeSet<_>>();
    for name in [
        "model.layers.2.linear_attn.A_log",
        "model.layers.3.self_attn.q_norm.weight",
        "model.layers.3.self_attn.k_norm.weight",
    ] {
        assert!(new_names.contains(name), "{name} must be reported as new");
    }
    let mismatched = float_parameters(&loaded.model)
        .into_iter()
        .filter(|(_, dtype)| *dtype != DType::BF16)
        .collect::<Vec<_>>();
    assert!(
        mismatched.is_empty(),
        "parameters outside the checkpoint dtype: {mismatched:?}"
    );
    let output = loaded.model.forward(&input(), false).unwrap().0;
    assert_eq!(output.dtype(), DType::BF16);
}

#[cfg(feature = "autodiff")]
/// Dtype casting preserves trainable leaves for reinitialized MLP parameters.
#[test]
fn reinitialized_mlp_parameters_keep_gradients_after_dtype_selection() {
    /// Autodiff wrapper used to verify gradients through newly initialized owners.
    type Ad = burn::backend::Autodiff<B>;

    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), &tiny_config(), false);
    let mut recipe = Qwen3_5BackboneRecipe::from_local_config(directory.path())
        .unwrap()
        .prefix(1)
        .unwrap();
    recipe.layers[0].intermediate_size = 14;
    let model = recipe
        .load_pretrained::<Ad>(
            directory.path(),
            &Qwen3_5RecipeLoadOptions {
                shape_changes: ShapeChangePolicy::Reinitialize,
                ..Default::default()
            },
            &Default::default(),
        )
        .unwrap()
        .model;
    let input = CausalLmInput {
        token_ids: Tensor::<Ad, 2, Int>::from_ints([[1, 2, 3]], &Default::default()),
        token_mask: None,
        position_ids: None,
    };
    let gradients = model.forward(&input, false).unwrap().0.sum().backward();
    for parameter in [
        &model.layers[0].mlp.gate_proj.weight,
        &model.layers[0].mlp.up_proj.weight,
        &model.layers[0].mlp.down_proj.weight,
    ] {
        assert!(parameter.grad(&gradients).is_some());
    }
}

/// Strict loading rejects unexplained keys and a checkpoint config changed after recipe creation.
#[test]
fn unknown_source_parameters_and_changed_source_config_are_not_ignored() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), &tiny_config(), true);
    let recipe = Qwen3_5BackboneRecipe::from_local_config(directory.path())
        .unwrap()
        .prefix(1)
        .unwrap();
    assert!(
        recipe
            .load_pretrained::<B>(directory.path(), &Default::default(), &Default::default())
            .is_err()
    );
    fixture(directory.path(), &tiny_config(), false);
    let mut changed = tiny_config();
    changed.rms_norm_eps = 0.001;
    fs::write(
        directory.path().join("config.json"),
        serde_json::to_vec(&changed).unwrap(),
    )
    .unwrap();
    assert!(
        recipe
            .load_pretrained::<B>(directory.path(), &Default::default(), &Default::default())
            .is_err()
    );
}

/// Stateless heterogeneous backbones reject creation of homogeneous family decode caches.
#[test]
fn recipe_backbones_cannot_silently_enter_homogeneous_family_caches() {
    let model = Qwen3_5BackboneRecipe::from_config(tiny_config())
        .unwrap()
        .init_random::<B>(&Default::default())
        .unwrap();
    let wrapper = Qwen3_5ForCausalLm {
        model,
        lm_head: None,
        mtp: None,
    };
    let error = wrapper
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &Default::default())
        .unwrap_err();
    assert!(error.to_string().contains("stateless"));
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config {
        cases: 16,
        rng_seed: proptest::test_runner::RngSeed::Fixed(0x5157_454e_5245_4349),
        ..Default::default()
    })]

    /// Unique in-range selections preserve source identity; duplicates and invalid indices fail.
    #[test]
    fn qwen_recipe_source_selections_are_exact_or_rejected_before_loading(
        selected in proptest::collection::vec(0usize..5, 1..5),
    ) {
        let directory = tempfile::tempdir().unwrap();
        let mut config = tiny_config();
        config.num_hidden_layers = 3;
        config.layer_types = vec![Qwen3_5LayerType::FullAttention; 3];
        fixture(directory.path(), &config, false);
        let mut recipe = Qwen3_5BackboneRecipe::from_config(config.clone()).unwrap();
        let template = recipe.layers[0].clone();
        recipe.layers = selected.iter().map(|&index| {
            let mut layer = template.clone();
            layer.source_layer = Some(index);
            layer
        }).collect();
        let result = recipe.load_pretrained::<B>(directory.path(), &Default::default(), &Default::default());
        let valid = selected.iter().all(|&index| index < 3)
            && selected.iter().copied().collect::<std::collections::BTreeSet<_>>().len() == selected.len();
        if !valid {
            let error = result.unwrap_err();
            proptest::prop_assert!(error.to_string().contains("source_layer"), "{}", error);
        } else {
            let loaded = result.unwrap();
            let family = original(directory.path(), config);
            proptest::prop_assert!(loaded.report.newly_initialized.is_empty());
            for (target, &source) in selected.iter().enumerate() {
                proptest::prop_assert_eq!(
                    loaded.model.layers[target].mlp.up_proj.weight.val().to_data(),
                    family.model.layers[source].mlp.up_proj.weight.val().to_data(),
                );
                proptest::prop_assert!(loaded.report.reused.iter().any(|item| {
                    item.source == format!("model.layers.{source}.mlp.up_proj.weight")
                        && item.target == format!("model.layers.{target}.mlp.up_proj.weight")
                }), "transfer report must preserve the selected source identity");
            }
            let (hidden, _) = loaded.model.forward(&input(), false).unwrap();
            proptest::prop_assert!(hidden.to_data().to_vec::<f32>().unwrap().iter().all(|value| value.is_finite()));
        }
    }
}

/// Invalid prefixes, empty stacks, and broad checkpoint exceptions return contextual errors.
#[test]
fn qwen_recipe_empty_topology_prefix_and_source_contract_fail_contextually() {
    let recipe = Qwen3_5BackboneRecipe::from_config(tiny_config()).unwrap();
    assert_eq!(recipe.source_config(), &tiny_config());
    for count in [0, 3, usize::MAX] {
        let error = recipe.clone().prefix(count).unwrap_err();
        assert!(error.to_string().contains("qwen.recipe.prefix"));
    }
    let mut empty = recipe.clone();
    empty.layers.clear();
    let error = empty.init_random::<B>(&Default::default()).unwrap_err();
    assert!(error.to_string().contains("at least one block"));

    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), &tiny_config(), false);
    for missing in [true, false] {
        let mut options = Qwen3_5RecipeLoadOptions::default();
        if missing {
            options
                .checkpoint
                .allow_missing
                .push(KeyMatcher::prefix("model."));
        } else {
            options
                .checkpoint
                .allow_unexpected
                .push(KeyMatcher::prefix("model."));
        }
        let error = recipe
            .load_pretrained::<B>(directory.path(), &options, &Default::default())
            .unwrap_err();
        assert!(error.to_string().contains("allowlists"));
    }
}

/// Config-only recipes enforce the 16 MiB JSON bound and initialize without checkpoint payloads.
#[test]
fn qwen_local_recipe_config_is_bounded_and_does_not_require_weight_files() {
    let directory = tempfile::tempdir().unwrap();
    assert!(Qwen3_5BackboneRecipe::from_local_config(directory.path()).is_err());
    let path = directory.path().join("config.json");
    fs::write(&path, b"{invalid").unwrap();
    let error = Qwen3_5BackboneRecipe::from_local_config(directory.path()).unwrap_err();
    assert!(error.to_string().contains("qwen.recipe.config"));
    fs::File::create(&path)
        .unwrap()
        .set_len(16 * 1024 * 1024 + 1)
        .unwrap();
    let error = Qwen3_5BackboneRecipe::from_local_config(directory.path()).unwrap_err();
    assert!(error.to_string().contains("16 MiB"));
    fs::write(&path, serde_json::to_vec(&tiny_config()).unwrap()).unwrap();
    let recipe = Qwen3_5BackboneRecipe::from_local_config(directory.path()).unwrap();
    assert_eq!(recipe.source_config(), &tiny_config());
    assert!(!directory.path().join("model.safetensors").exists());
    assert!(recipe.init_random::<B>(&Default::default()).is_ok());
}

/// Explicit source layout loads successfully while two automatic embedding namespaces fail.
#[test]
fn qwen_recipe_namespace_ambiguity_is_rejected_and_explicit_layout_is_honored() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), &tiny_config(), false);
    let recipe = Qwen3_5BackboneRecipe::from_config(tiny_config()).unwrap();
    let explicit = Qwen3_5RecipeLoadOptions {
        source: Some(Qwen3_5CheckpointSource::TextOnly),
        ..Default::default()
    };
    assert!(
        recipe
            .load_pretrained::<B>(directory.path(), &explicit, &Default::default())
            .is_ok()
    );

    // Add a second, physically distinct embedding to create a valid SafeTensors
    // container whose model namespace is ambiguous before contract matching.
    let path = directory.path().join("model.safetensors");
    let bytes = fs::read(&path).unwrap();
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + header_length]).unwrap();
    let mut payload = bytes[8 + header_length..].to_vec();
    let mut embedding = header["model.embed_tokens.weight"].clone();
    let offsets = embedding["data_offsets"].as_array().unwrap();
    let duplicate = payload
        [offsets[0].as_u64().unwrap() as usize..offsets[1].as_u64().unwrap() as usize]
        .to_vec();
    let start = payload.len();
    payload.extend(duplicate);
    embedding["data_offsets"] = json!([start, payload.len()]);
    header.insert("model.language_model.embed_tokens.weight".into(), embedding);
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    fs::write(path, bytes).unwrap();
    let error = recipe
        .load_pretrained::<B>(directory.path(), &Default::default(), &Default::default())
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("exactly one Qwen embedding namespace")
    );
}
