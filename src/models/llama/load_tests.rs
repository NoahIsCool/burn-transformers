//! Strict loading: committed tiny checkpoints load exactly, malformed
//! checkpoints fail before any payload is installed, and the sink rejects
//! targets the configured graph does not own.

use std::path::{Path, PathBuf};

use burn::tensor::TensorData;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

use super::*;
use crate::models::llama::model::test_support::{B, tiny_model};
use crate::models::llama::{LlamaFamilyConfig, LlamaVariant};

/// Returns a committed tiny-golden fixture directory.
fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tiny_goldens")
        .join(case)
}

/// Loads a fixture's configuration and allocates its graph.
fn graph(root: &Path) -> LlamaForCausalLm<B> {
    let config = std::fs::read_to_string(root.join("config.json")).unwrap();
    let config =
        <LlamaFamilyConfig as crate::models::config::ModelConfig>::from_json_str(&config).unwrap();
    LlamaForCausalLm::init(config, &Default::default()).unwrap()
}

/// Copies `case` into a temporary directory, rewriting its tensors with `edit`.
fn rewritten(
    case: &str,
    edit: impl FnOnce(&mut Vec<(String, TensorView<'_>)>),
) -> tempfile::TempDir {
    let source = fixture(case);
    let directory = tempfile::tempdir().unwrap();
    std::fs::copy(
        source.join("config.json"),
        directory.path().join("config.json"),
    )
    .unwrap();
    let bytes = std::fs::read(source.join("model.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let mut entries: Vec<(String, TensorView<'_>)> = tensors.tensors().into_iter().collect();
    edit(&mut entries);
    let payload = safetensors::serialize(entries, None).unwrap();
    std::fs::write(directory.path().join("model.safetensors"), payload).unwrap();
    directory
}

/// Every committed fixture loads with zero unexplained tensors.
#[test]
fn committed_fixtures_load_strictly() {
    for case in [
        "llama3",
        "llama-bias-tied",
        "qwen2",
        "qwen3",
        "mistral-sliding",
        "smollm3",
        "olmo2",
    ] {
        let root = fixture(case);
        let mut model = graph(&root);
        let outcome = model
            .load_checkpoint(&root, &LoadOptions::default())
            .unwrap();
        assert!(outcome.report.is_strictly_valid(), "{case}");
    }
}

/// Missing, unexpected, and misshapen tensors fail during planning.
#[test]
fn malformed_checkpoints_fail_before_installation() {
    let missing = rewritten("qwen3", |entries| {
        entries.retain(|(name, _)| name != "model.layers.1.self_attn.k_norm.weight");
    });
    let unexpected = rewritten("llama-bias-tied", |entries| {
        let (_, embedding) = entries
            .iter()
            .find(|(name, _)| name == "model.embed_tokens.weight")
            .cloned()
            .unwrap();
        entries.push(("lm_head.weight".into(), embedding));
    });
    let misshapen = rewritten("olmo2", |entries| {
        let index = entries
            .iter()
            .position(|(name, _)| name == "model.layers.0.self_attn.q_norm.weight")
            .unwrap();
        let (name, view) = entries.remove(index);
        let data = &view.data()[..8 * 4];
        let narrow = TensorView::new(Dtype::F32, vec![8], data).unwrap();
        entries.push((name, narrow));
    });
    for (directory, context) in [
        (missing, "missing"),
        (unexpected, "stored tied head"),
        (misshapen, "per-head-sized full-width norm"),
    ] {
        let mut model = graph(directory.path());
        let error = model
            .load_checkpoint(directory.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(
            matches!(error, LlamaCheckpointLoadError::Checkpoint(_)),
            "{context}: {error}"
        );
        assert!(error.execution_failure().is_none(), "{context}");
    }
}

/// The sink rejects targets outside the configured graph.
#[test]
fn the_sink_rejects_foreign_targets() {
    let mut model = tiny_model(LlamaVariant::Qwen3, 1);
    let scalar = || TensorData::new(vec![0.0_f32; 4], [4]);
    let cases = [
        ("model.unknown.weight", "unknown"),
        ("model.layers.7.mlp_norm.weight", "outside"),
        ("model.layers.x.mlp_norm.weight", "decimal"),
        ("model.layers.0", "expected"),
        ("model.layers.0.", "expected"),
        ("model.layers.0.self_attn.o_proj.bias", "projection bias"),
        ("model.layers.0.mlp.gate_proj.bias", "projection bias"),
        ("model.layers.0.mlp.fused.weight", "unknown"),
        ("model.layers.0.self_attn.rotary.weight", "unknown"),
        ("lm_head.weight", "independent LM head"),
    ];
    let mut tied = tiny_model(LlamaVariant::Llama, 1);
    for (target, fragment) in cases {
        let sink: &mut LlamaForCausalLm<B> = if target == "lm_head.weight" {
            &mut tied
        } else {
            &mut model
        };
        let error = sink.write_tensor(target, scalar()).unwrap_err();
        assert!(error.to_string().contains(fragment), "{target}: {error}");
    }
    let shape = model
        .write_tensor("model.norm.weight", scalar())
        .unwrap_err();
    assert!(matches!(shape, LlamaLoadError::Parameter { .. }), "{shape}");
    let mut unnormalized = tiny_model(LlamaVariant::Mistral, 1);
    let error = unnormalized
        .write_tensor("model.layers.0.self_attn.query_norm.weight", scalar())
        .unwrap_err();
    assert!(error.to_string().contains("query RMSNorm"), "{error}");
    let error = unnormalized
        .write_tensor("model.layers.0.self_attn.key_norm.weight", scalar())
        .unwrap_err();
    assert!(error.to_string().contains("key RMSNorm"), "{error}");
}

/// Only a tied model accepts the embedding-to-head alias.
#[test]
fn the_tied_head_alias_is_validated() {
    let mut tied = tiny_model(LlamaVariant::Llama, 2);
    tied.alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap();
    assert!(
        tied.alias_tensor("lm_head.weight", "model.norm.weight")
            .is_err()
    );
    let mut untied = tiny_model(LlamaVariant::Qwen3, 2);
    let error = untied
        .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap_err();
    assert!(error.to_string().contains("independent LM head"), "{error}");
    let mut lazy = LlamaForCausalLm::<B>::init(
        crate::models::llama::model::test_support::tiny_config(LlamaVariant::Llama),
        &Default::default(),
    )
    .unwrap();
    let error = lazy
        .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap_err();
    assert!(error.to_string().contains("not been installed"), "{error}");
}

/// A graph edited after planning fails in the sink and reports the partial load.
#[test]
fn sink_failures_after_planning_keep_the_partial_report() {
    let root = fixture("qwen3");
    let mut model = graph(&root);
    model.model.layers[1].self_attn.query_norm = None;
    let error = model
        .load_checkpoint(&root, &LoadOptions::default())
        .unwrap_err();
    let failure = error
        .execution_failure()
        .expect("planning succeeded, so execution reports the failure");
    assert_eq!(
        failure.failed_target.as_deref(),
        Some("model.layers.1.self_attn.query_norm.weight")
    );
    assert!(matches!(
        &failure.cause,
        crate::checkpoint::ExecutionFailureCause::Sink(LlamaLoadError::MissingComponent { .. })
    ));
    assert!(!failure.report.applied.is_empty());
}

/// `load_checkpoint_with` under the (empty) family options equals `load_checkpoint`.
#[test]
fn load_checkpoint_with_default_options_matches_load_checkpoint() {
    let root = fixture("qwen3");
    let detected = graph(&root)
        .load_checkpoint(&root, &LoadOptions::default())
        .unwrap();
    let explicit = graph(&root)
        .load_checkpoint_with(
            &root,
            &LoadOptions::default(),
            crate::models::llama::LlamaCheckpointOptions::default(),
        )
        .unwrap();
    assert!(explicit.report.is_strictly_valid());
    assert_eq!(explicit.report, detected.report);
}
