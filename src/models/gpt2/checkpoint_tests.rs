//! GPT-2 namespace detection, contract inventory, and sink rejection paths.
//!
//! Synthetic SafeTensors files hold only the tensors each case needs; numerical
//! import parity lives in the model tests with the PyTorch reference fixture.

use std::path::Path;

use burn::backend::Flex;
use safetensors::tensor::{Dtype, TensorView, serialize};

use super::*;
use crate::checkpoint::{TensorDType, TensorOperation};

/// Deterministic CPU backend with F32 parameters.
type TestBackend = Flex<f32, i32>;

/// A one-block topology with a distinct MLP width.
fn tiny() -> Gpt2Config {
    Gpt2Config {
        vocab_size: 7,
        n_embd: 4,
        n_layer: 1,
        n_head: 2,
        n_positions: 8,
        n_inner: Some(6),
        ..Gpt2Config::default()
    }
}

/// Writes zero-valued F32 tensors with the given names and shapes as `model.safetensors`.
fn write(root: &Path, tensors: &[(&str, Vec<usize>)]) {
    let bytes = tensors
        .iter()
        .map(|(_, shape)| vec![0_u8; 4 * shape.iter().product::<usize>()])
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .zip(&bytes)
        .map(|((name, shape), bytes)| {
            (
                (*name).to_owned(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    std::fs::write(
        root.join("model.safetensors"),
        serialize(views, None).unwrap(),
    )
    .unwrap();
}

/// Discovers the manifest of a synthetic checkpoint directory.
fn manifest(root: &Path) -> CheckpointManifest {
    discover_checkpoint(root, &LoadOptions::default()).unwrap()
}

/// Exactly one token-embedding namespace selects the layout; neither or both is rejected.
#[test]
fn namespace_detection_requires_exactly_one_embedding_owner() {
    for (names, expected) in [
        (vec!["wte.weight"], Some(Gpt2CheckpointLayout::Backbone)),
        (
            vec!["transformer.wte.weight"],
            Some(Gpt2CheckpointLayout::LanguageModel),
        ),
        (vec!["wte.weight", "transformer.wte.weight"], None),
        (vec!["ln_f.weight"], None),
    ] {
        let root = tempfile::tempdir().unwrap();
        let tensors = names
            .iter()
            .map(|name| (*name, vec![2]))
            .collect::<Vec<_>>();
        write(root.path(), &tensors);
        let detected = Gpt2CheckpointLayout::detect(&manifest(root.path()));
        match expected {
            Some(layout) => assert_eq!(detected.unwrap(), layout, "{names:?}"),
            None => assert!(
                matches!(
                    &detected,
                    Err(Gpt2CheckpointLoadError::Namespace(reason))
                        if reason == "expected exactly one GPT-2 token-embedding namespace"
                ),
                "{names:?}: {detected:?}"
            ),
        }
    }
}

/// The contract lists runtime shapes, splits the fused projection, and aliases the head.
#[test]
fn contract_inventory_names_runtime_targets_and_fused_splits() {
    let config = tiny();
    for (layout, prefix) in [
        (Gpt2CheckpointLayout::Backbone, ""),
        (Gpt2CheckpointLayout::LanguageModel, "transformer."),
    ] {
        let contract = Gpt2CheckpointContract::new(&config, layout).unwrap();
        let shape = |name: &str| {
            contract
                .expected_tensors()
                .iter()
                .find(|tensor| tensor.name == name)
                .map(|tensor| tensor.shape.clone())
        };
        assert_eq!(shape("token_embedding.weight"), Some(vec![7, 4]));
        assert_eq!(shape("position_embedding.weight"), Some(vec![8, 4]));
        assert_eq!(shape("layers.0.attention.q_proj.weight"), Some(vec![4, 4]));
        assert_eq!(shape("layers.0.mlp.up_proj.weight"), Some(vec![4, 6]));
        assert_eq!(shape("layers.0.mlp.down_proj.weight"), Some(vec![6, 4]));
        assert_eq!(shape("lm_head.weight"), Some(vec![7, 4]));
        // 2 embeddings + 2 final-norm vectors + one block (4 norm, 6 QKV, 6 dense) + the head.
        assert_eq!(contract.expected_tensors().len(), 2 + 2 + 16 + 1);
        let fused = format!("{prefix}h.0.attn.c_attn.weight");
        assert!(
            contract.mappings().operations().iter().any(|operation| matches!(
                operation,
                TensorOperation::Split { source, axis: 1, targets } if *source == fused && targets.len() == 3
            )),
            "{layout:?}"
        );
        let mut options = LoadOptions::default();
        contract.configure_load_options(&mut options);
        contract.configure_load_options(&mut options);
        assert_eq!(
            options.allow_unexpected.len(),
            2,
            "configuring twice adds nothing"
        );
        assert_eq!(options.index_aliases.len(), 1);
    }
}

/// Lists every physical checkpoint source of `contract` with its serialized shape.
///
/// Fused Conv1D projections are stored once, three times as wide as each
/// split target; the tied head is an alias without its own source.
fn contract_sources(contract: &Gpt2CheckpointContract) -> Vec<(String, Vec<usize>)> {
    let shape_of = |target: &str| {
        contract
            .expected_tensors()
            .iter()
            .find(|tensor| tensor.name == target)
            .unwrap()
            .shape
            .clone()
    };
    let mut tensors = Vec::new();
    for operation in contract.mappings().operations() {
        let (source, shape) = match operation {
            TensorOperation::Map { source, target, .. } => (source.clone(), shape_of(target)),
            TensorOperation::Split {
                source,
                axis,
                targets,
            } => {
                let mut shape = shape_of(&targets[0].0);
                shape[*axis] *= 3;
                (source.clone(), shape)
            }
            _ => continue,
        };
        tensors.push((
            source,
            shape.iter().map(|&size| size as usize).collect::<Vec<_>>(),
        ));
    }
    tensors
}

/// Integer tensors cannot enter the floating GPT-2 graph even when shapes match.
#[test]
fn integer_storage_is_rejected_after_shape_preflight() {
    let config = tiny();
    let contract = Gpt2CheckpointContract::new(&config, Gpt2CheckpointLayout::Backbone).unwrap();
    let root = tempfile::tempdir().unwrap();
    let tensors = contract_sources(&contract);
    let borrowed = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.clone()))
        .collect::<Vec<_>>();
    write(root.path(), &borrowed);
    let mut manifest = manifest(root.path());
    let options = {
        let mut options = LoadOptions::default();
        contract.configure_load_options(&mut options);
        options
    };
    contract
        .build_load_plan(&manifest, &options)
        .expect("the floating inventory is complete");
    for shard in &mut manifest.shards {
        if let Some(tensor) = shard.tensors.get_mut("ln_f.weight") {
            tensor.dtype = TensorDType::I32;
        }
    }
    manifest.tensors.get_mut("ln_f.weight").unwrap().dtype = TensorDType::I32;
    let error = contract.build_load_plan(&manifest, &options).unwrap_err();
    assert!(
        matches!(
            &error,
            Gpt2CheckpointLoadError::NonFloatParameter { target, dtype: TensorDType::I32 }
                if target == "final_norm.gamma"
        ),
        "{error}"
    );
    assert_eq!(
        error.to_string(),
        "GPT-2 parameters must be floating point; `final_norm.gamma` is I32"
    );
    assert!(error.execution_failure().is_none());
}

/// Malformed targets, unknown fields, and absent biases are contextual target errors.
#[test]
fn sink_rejects_malformed_targets_and_absent_biases() {
    let mut model = Gpt2Model::<TestBackend>::init(tiny(), &Default::default()).unwrap();
    let vector = || TensorData::new(vec![0.0_f32; 4], [4]);
    for target in [
        "layers.0",
        "layers.zero.attention_norm.gamma",
        "layers.3.attention_norm.gamma",
        "layers.0.attention_norm.delta",
        "layers.0.attention",
        "layers.0.attention.c_attn.weight",
        "layers.0.mlp.up_proj.scale",
    ] {
        let error = model.write_tensor(target, vector()).unwrap_err();
        assert!(
            matches!(&error, Gpt2LoadError::Target(reported) if reported == target),
            "{target}: {error}"
        );
    }
    model.layers[0].mlp.up_proj.bias = None;
    model.final_norm.beta = None;
    for target in ["layers.0.mlp.up_proj.bias", "final_norm.beta"] {
        let error = model.write_tensor(target, vector()).unwrap_err();
        assert!(
            matches!(&error, Gpt2LoadError::Target(reason) if *reason == format!("missing bias at {target}")),
            "{error}"
        );
    }
    let error = model
        .write_tensor("final_norm.gamma", TensorData::new(vec![0.0_f32; 3], [3]))
        .unwrap_err();
    assert!(
        matches!(
            &error,
            Gpt2LoadError::Parameter { target, source: ParameterInstallError::Shape { .. } }
                if target == "final_norm.gamma"
        ),
        "{error}"
    );
}

/// Only the initialized token embedding of a tied graph may back the vocabulary head.
#[test]
fn tied_readout_alias_requires_the_exact_initialized_owner() {
    let mut model = Gpt2Model::<TestBackend>::init(tiny(), &Default::default()).unwrap();
    let invalid = |model: &mut Gpt2Model<TestBackend>, target: &str, source: &str| {
        let error = model.alias_tensor(target, source).unwrap_err();
        assert!(
            matches!(
                &error,
                Gpt2LoadError::Target(reason)
                    if *reason == format!("invalid tied readout {target} -> {source}")
            ),
            "{error}"
        );
    };
    // The lazily initialized embedding has no storage yet.
    invalid(&mut model, "lm_head.weight", "token_embedding.weight");
    model
        .write_tensor(
            "token_embedding.weight",
            TensorData::new(vec![0.0_f32; 28], [7, 4]),
        )
        .unwrap();
    model
        .alias_tensor("lm_head.weight", "token_embedding.weight")
        .unwrap();
    invalid(&mut model, "embed_out.weight", "token_embedding.weight");
    invalid(&mut model, "lm_head.weight", "position_embedding.weight");
    model.config.tie_word_embeddings = false;
    invalid(&mut model, "lm_head.weight", "token_embedding.weight");
}

/// Whole-load failures are typed by stage; only pass two carries a partial report.
///
/// A checkpoint with both embedding namespaces fails detection, an edited
/// configuration fails before planning, and a block removed after
/// initialization fails in pass two with the per-tensor [`Gpt2LoadError`]
/// inside the execution failure.
#[test]
fn load_checkpoint_errors_are_typed_by_stage() {
    let config = tiny();
    let contract = Gpt2CheckpointContract::new(&config, Gpt2CheckpointLayout::Backbone).unwrap();
    let sources = contract_sources(&contract);
    let mut borrowed = sources
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.clone()))
        .collect::<Vec<_>>();
    let complete = tempfile::tempdir().unwrap();
    write(complete.path(), &borrowed);
    let device = Default::default();

    let mut model = Gpt2ForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let outcome = model
        .load_checkpoint(complete.path(), &LoadOptions::default())
        .unwrap();
    assert!(outcome.report.is_strictly_valid());

    borrowed.push(("transformer.wte.weight", vec![7, 4]));
    let ambiguous = tempfile::tempdir().unwrap();
    write(ambiguous.path(), &borrowed);
    let mut model = Gpt2ForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(ambiguous.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(
        matches!(&error, Gpt2CheckpointLoadError::Namespace(_)),
        "{error}"
    );
    assert!(error.execution_failure().is_none());

    let mut model = Gpt2ForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    model.model.config.n_head = 3;
    let error = model
        .load_checkpoint(complete.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(
        matches!(&error, Gpt2CheckpointLoadError::Configuration(_)),
        "{error}"
    );
    assert!(error.execution_failure().is_none());

    let mut model = Gpt2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    model.model.layers.clear();
    let error = model
        .load_checkpoint(complete.path(), &LoadOptions::default())
        .unwrap_err();
    let failure = error
        .execution_failure()
        .expect("pass two reached the sink");
    assert!(
        matches!(
            &failure.cause,
            crate::checkpoint::ExecutionFailureCause::Sink(Gpt2LoadError::Target(target))
                if target.starts_with("layers.0.")
        ),
        "{error}"
    );
}
