//! Strict loading: committed tiny checkpoints load exactly from every
//! published namespace, the vision tower is reported as ignored, malformed
//! checkpoints fail before any payload is installed, and the sink rejects
//! targets the configured graph does not own.

use std::path::{Path, PathBuf};

use burn::tensor::TensorData;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

use super::*;
use crate::{
    CausalLmInput, ForwardOptions, LogitsSelection,
    models::{
        config::ModelConfig,
        gemma::{
            GemmaConfig, GemmaVariant,
            model::test_support::{B, host, ints, tiny_config, tiny_model},
        },
    },
};

/// Returns a committed tiny-golden fixture directory.
fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tiny_goldens")
        .join(case)
}

/// Loads a fixture's configuration and allocates its graph.
fn graph(root: &Path) -> GemmaForCausalLm<B> {
    let config = std::fs::read_to_string(root.join("config.json")).unwrap();
    let config = GemmaConfig::from_json_str(&config).unwrap();
    GemmaForCausalLm::init(config, &Default::default()).unwrap()
}

/// Copies `case` into a temporary directory, rewriting its tensors with `edit`.
///
/// `edit` receives owned `(name, dtype, shape, bytes)` entries and may
/// rename, drop, or add tensors.
fn rewritten(
    case: &str,
    edit: impl FnOnce(&mut Vec<(String, Dtype, Vec<usize>, Vec<u8>)>),
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
    let mut entries: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = tensors
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            (
                name,
                view.dtype(),
                view.shape().to_vec(),
                view.data().to_vec(),
            )
        })
        .collect();
    edit(&mut entries);
    let views: Vec<(String, TensorView<'_>)> = entries
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                name.clone(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        })
        .collect();
    let payload = safetensors::serialize(views, None).unwrap();
    std::fs::write(directory.path().join("model.safetensors"), payload).unwrap();
    directory
}

/// Renames every text-only `model.` tensor under `prefix` and adds a small vision tower.
fn multimodal(case: &str, prefix: &str, vision: &[&str]) -> tempfile::TempDir {
    let prefix = prefix.to_owned();
    let vision: Vec<String> = vision.iter().map(|name| (*name).to_owned()).collect();
    rewritten(case, move |entries| {
        for (name, ..) in entries.iter_mut() {
            *name = format!("{prefix}.{}", name.strip_prefix("model.").unwrap());
        }
        for name in vision {
            entries.push((name, Dtype::F32, vec![2], vec![0; 8]));
        }
    })
}

/// Last-position logits of a fixed prompt.
fn prompt_logits(model: &GemmaForCausalLm<B>) -> Vec<f32> {
    let output = model
        .forward(
            CausalLmInput {
                token_ids: ints([[5, 17, 3, 29, 8]]),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::Last,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    host(output.logits)
}

/// Every committed fixture loads with zero unexplained tensors.
#[test]
fn committed_fixtures_load_strictly() {
    for case in ["gemma2", "gemma3-text"] {
        let root = fixture(case);
        let mut model = graph(&root);
        let outcome = model
            .load_checkpoint(&root, &LoadOptions::default())
            .unwrap();
        assert!(outcome.report.is_strictly_valid(), "{case}");
        assert!(outcome.report.allowlisted_ignored.is_empty(), "{case}");
    }
}

/// Both multimodal namespaces load the same text tower and report the omitted vision tensors.
#[test]
fn multimodal_namespaces_load_the_text_tower_and_report_the_vision_tower() {
    let root = fixture("gemma3-text");
    let mut text_only = graph(&root);
    text_only
        .load_checkpoint(&root, &LoadOptions::default())
        .unwrap();
    let reference = prompt_logits(&text_only);
    for (prefix, vision) in [
        (
            "model.language_model",
            [
                "model.vision_tower.vision_model.post_layernorm.weight",
                "model.multi_modal_projector.mm_soft_emb_norm.weight",
            ],
        ),
        (
            "language_model.model",
            [
                "vision_tower.vision_model.post_layernorm.weight",
                "multi_modal_projector.mm_soft_emb_norm.weight",
            ],
        ),
    ] {
        let directory = multimodal("gemma3-text", prefix, &vision);
        let mut model = graph(directory.path());
        let outcome = model
            .load_checkpoint(directory.path(), &LoadOptions::default())
            .unwrap();
        assert!(outcome.report.is_strictly_valid(), "{prefix}");
        let mut ignored = outcome.report.allowlisted_ignored.clone();
        ignored.sort();
        let mut expected: Vec<String> = vision.iter().map(|name| (*name).to_owned()).collect();
        expected.sort();
        assert_eq!(ignored, expected, "{prefix}");
        assert_eq!(prompt_logits(&model), reference, "{prefix}");
    }
}

/// Missing, unexpected, misshapen, stray-vision, and ambiguous checkpoints fail before installation.
#[test]
fn malformed_checkpoints_fail_before_installation() {
    let missing = rewritten("gemma3-text", |entries| {
        entries.retain(|(name, ..)| name != "model.layers.1.self_attn.k_norm.weight");
    });
    let stored_tied_head = rewritten("gemma2", |entries| {
        let embedding = entries
            .iter()
            .find(|(name, ..)| name == "model.embed_tokens.weight")
            .cloned()
            .unwrap();
        entries.push((
            "lm_head.weight".into(),
            embedding.1,
            embedding.2,
            embedding.3,
        ));
    });
    let misshapen = rewritten("gemma3-text", |entries| {
        let entry = entries
            .iter_mut()
            .find(|(name, ..)| name == "model.layers.0.self_attn.q_norm.weight")
            .unwrap();
        entry.2 = vec![4];
        entry.3.truncate(16);
    });
    let vision_in_text_only = rewritten("gemma2", |entries| {
        entries.push((
            "vision_tower.vision_model.post_layernorm.weight".into(),
            Dtype::F32,
            vec![2],
            vec![0; 8],
        ));
    });
    let stray_in_multimodal = multimodal(
        "gemma3-text",
        "model.language_model",
        &["model.audio_tower.weight"],
    );
    for (directory, context) in [
        (missing, "missing"),
        (stored_tied_head, "stored tied head"),
        (misshapen, "misshapen Q norm"),
        (
            vision_in_text_only,
            "vision tensor in a text-only checkpoint",
        ),
        (stray_in_multimodal, "non-vision stray tensor"),
    ] {
        let mut model = graph(directory.path());
        let error = model
            .load_checkpoint(directory.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(
            matches!(error, GemmaCheckpointLoadError::Checkpoint(_)),
            "{context}: {error}"
        );
        assert!(error.execution_failure().is_none(), "{context}");
    }
    let ambiguous = rewritten("gemma2", |entries| {
        let embedding = entries
            .iter()
            .find(|(name, ..)| name == "model.embed_tokens.weight")
            .cloned()
            .unwrap();
        entries.push((
            "model.language_model.embed_tokens.weight".into(),
            embedding.1,
            embedding.2,
            embedding.3,
        ));
    });
    let mut model = graph(ambiguous.path());
    let error = model
        .load_checkpoint(ambiguous.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(
        matches!(error, GemmaCheckpointLoadError::Namespace(_)),
        "{error}"
    );
    assert!(error.to_string().contains("ambiguously"), "{error}");
}

/// The sink rejects targets outside the configured graph.
#[test]
fn the_sink_rejects_foreign_targets() {
    let mut gemma2 = tiny_model(GemmaVariant::Gemma2, 1);
    let scalar = || TensorData::new(vec![0.0_f32; 4], [4]);
    for (target, fragment) in [
        ("model.unknown.weight", "unknown"),
        ("model.layers.7.input_layernorm.weight", "outside"),
        ("model.layers.x.input_layernorm.weight", "decimal"),
        ("model.layers.0", "expected"),
        ("model.layers.0.", "expected"),
        ("model.layers.0.self_attn.o_proj.bias", "projection bias"),
        ("model.layers.0.mlp.gate_proj.bias", "unknown"),
        ("model.layers.0.self_attn.rotary.weight", "unknown"),
        (
            "model.layers.0.self_attn.query_norm.weight",
            "query RMSNorm",
        ),
        ("model.layers.0.self_attn.key_norm.weight", "key RMSNorm"),
        ("lm_head.weight", "independent LM head"),
    ] {
        let error = gemma2.write_tensor(target, scalar()).unwrap_err();
        assert!(error.to_string().contains(fragment), "{target}: {error}");
    }
    let shape = gemma2
        .write_tensor("model.norm.weight", scalar())
        .unwrap_err();
    assert!(matches!(shape, GemmaLoadError::Parameter { .. }), "{shape}");
    let mut gemma3 = tiny_model(GemmaVariant::Gemma3Text, 1);
    gemma3
        .write_tensor("model.layers.3.self_attn.key_norm.weight", scalar())
        .unwrap();
    gemma3
        .write_tensor("model.layers.3.post_feedforward_layernorm.weight", {
            TensorData::new(vec![0.0_f32; 16], [16])
        })
        .unwrap();
}

/// Only a tied model accepts the embedding-to-head alias.
#[test]
fn the_tied_head_alias_is_validated() {
    let mut tied = tiny_model(GemmaVariant::Gemma2, 2);
    tied.alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap();
    assert!(
        tied.alias_tensor("lm_head.weight", "model.norm.weight")
            .is_err()
    );
    let mut untied = crate::models::gemma::model::test_support::model_from(
        GemmaConfig {
            tie_word_embeddings: false,
            ..tiny_config(GemmaVariant::Gemma3Text)
        },
        2,
    );
    let error = untied
        .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap_err();
    assert!(error.to_string().contains("independent LM head"), "{error}");
    untied
        .write_tensor(
            "lm_head.weight",
            TensorData::new(vec![0.0_f32; 32 * 16], [32, 16]),
        )
        .unwrap();
    let mut lazy =
        GemmaForCausalLm::<B>::init(tiny_config(GemmaVariant::Gemma2), &Default::default())
            .unwrap();
    let error = lazy
        .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap_err();
    assert!(error.to_string().contains("not been installed"), "{error}");
}
