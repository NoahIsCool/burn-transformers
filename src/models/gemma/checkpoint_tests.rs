//! Checkpoint contracts: every tensor name and shape per variant and
//! source, the Q/K norm renames, optional biases, the tied-head alias, the
//! vision exemptions, and namespace detection.

use super::*;
use crate::checkpoint::{CheckpointError, TensorOperation};
use crate::models::gemma::GemmaVariant;

/// A small configuration of `variant`: `d = 8`, `f = 12`, two layers, `H = 4`, `K = 2`, `h = 2`.
fn small(variant: GemmaVariant) -> GemmaConfig {
    GemmaConfig {
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 2,
        max_position_embeddings: 32,
        sliding_window: 4,
        pad_token_id: None,
        bos_token_id: None,
        eos_token_id: None,
        ..GemmaConfig::new(variant)
    }
}

/// Returns `(target, shape)` pairs of a contract.
fn shapes(contract: &GemmaCheckpointContract) -> Vec<(String, Vec<u64>)> {
    contract
        .expected_tensors()
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.shape.clone()))
        .collect()
}

/// Returns `source -> target` renames and aliases of a contract.
fn sources(contract: &GemmaCheckpointContract) -> Vec<(String, String)> {
    contract
        .mappings()
        .operations()
        .iter()
        .map(|operation| match operation {
            TensorOperation::Map { source, target, .. } => (source.clone(), target.clone()),
            TensorOperation::Alias {
                source_target,
                target,
            } => (format!("alias:{source_target}"), target.clone()),
            other => panic!("unexpected operation {other:?}"),
        })
        .collect()
}

/// Returns the contract of `config` for `source`.
fn contract(config: &GemmaConfig, source: GemmaCheckpointSource) -> GemmaCheckpointContract {
    config
        .checkpoint_contract(GemmaCheckpointOptions { source })
        .unwrap()
}

/// A Gemma 2 layer has seven projections and four sandwich norms; the head is tied.
#[test]
fn gemma_2_contract_lists_every_tensor() {
    let contract = contract(
        &small(GemmaVariant::Gemma2),
        GemmaCheckpointSource::TextOnly,
    );
    let expected = shapes(&contract);
    assert_eq!(expected.len(), 1 + 2 * 11 + 2);
    let layer: Vec<(String, Vec<u64>)> = [
        ("input_layernorm.weight", vec![8]),
        ("post_attention_layernorm.weight", vec![8]),
        ("pre_feedforward_layernorm.weight", vec![8]),
        ("post_feedforward_layernorm.weight", vec![8]),
        ("self_attn.q_proj.weight", vec![8, 8]),
        ("self_attn.k_proj.weight", vec![4, 8]),
        ("self_attn.v_proj.weight", vec![4, 8]),
        ("self_attn.o_proj.weight", vec![8, 8]),
        ("mlp.gate_proj.weight", vec![12, 8]),
        ("mlp.up_proj.weight", vec![12, 8]),
        ("mlp.down_proj.weight", vec![8, 12]),
    ]
    .into_iter()
    .map(|(name, shape)| (format!("model.layers.1.{name}"), shape))
    .collect();
    assert_eq!(expected[12..23], layer[..]);
    assert_eq!(
        expected[0],
        ("model.embed_tokens.weight".into(), vec![16, 8])
    );
    assert_eq!(
        expected[expected.len() - 2..],
        [
            ("model.norm.weight".into(), vec![8]),
            ("lm_head.weight".into(), vec![16, 8])
        ]
    );
    let mappings = sources(&contract);
    assert!(mappings.iter().all(|(source, target)| source == target
        || source == "alias:model.embed_tokens.weight"));
    assert_eq!(
        mappings.last().unwrap(),
        &(
            "alias:model.embed_tokens.weight".into(),
            "lm_head.weight".into()
        )
    );
    assert!(contract.allow_unexpected().is_empty());
}

/// Gemma 3 adds per-head Q/K norms, renamed to the shared attention's fields.
#[test]
fn gemma_3_query_key_norms_are_renamed() {
    let contract = contract(
        &small(GemmaVariant::Gemma3Text),
        GemmaCheckpointSource::TextOnly,
    );
    assert_eq!(shapes(&contract).len(), 1 + 2 * 13 + 2);
    let mappings = sources(&contract);
    for (source, target) in [
        (
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.query_norm.weight",
        ),
        (
            "model.layers.0.self_attn.k_norm.weight",
            "model.layers.0.self_attn.key_norm.weight",
        ),
    ] {
        assert!(
            mappings.contains(&(source.into(), target.into())),
            "{source}"
        );
    }
    assert!(
        shapes(&contract).contains(&("model.layers.1.self_attn.query_norm.weight".into(), vec![2]))
    );
}

/// Biases appear on all four projections exactly when `attention_bias` is set.
#[test]
fn bias_tensors_follow_attention_bias() {
    let biased = GemmaConfig {
        attention_bias: true,
        ..small(GemmaVariant::Gemma2)
    };
    let expected = shapes(&contract(&biased, GemmaCheckpointSource::TextOnly));
    for (name, shape) in [
        ("q_proj", vec![8]),
        ("k_proj", vec![4]),
        ("v_proj", vec![4]),
        ("o_proj", vec![8]),
    ] {
        assert!(expected.contains(&(format!("model.layers.0.self_attn.{name}.bias"), shape)));
    }
    assert_eq!(expected.len(), 1 + 2 * 15 + 2);
}

/// Each source reads its own language prefix and head name and exempts its vision tower.
#[test]
fn sources_select_their_namespaces() {
    let untied = GemmaConfig {
        tie_word_embeddings: false,
        ..small(GemmaVariant::Gemma3Text)
    };
    for (source, prefix, head, omitted) in [
        (
            GemmaCheckpointSource::TextOnly,
            "model",
            "lm_head.weight",
            vec![],
        ),
        (
            GemmaCheckpointSource::Multimodal,
            "model.language_model",
            "lm_head.weight",
            vec!["model.vision_tower.", "model.multi_modal_projector."],
        ),
        (
            GemmaCheckpointSource::LegacyMultimodal,
            "language_model.model",
            "language_model.lm_head.weight",
            vec!["vision_tower.", "multi_modal_projector."],
        ),
    ] {
        let contract = contract(&untied, source);
        let mappings = sources(&contract);
        assert_eq!(
            mappings[0],
            (
                format!("{prefix}.embed_tokens.weight"),
                "model.embed_tokens.weight".into()
            )
        );
        assert!(mappings.contains(&(
            format!("{prefix}.layers.1.self_attn.k_norm.weight"),
            "model.layers.1.self_attn.key_norm.weight".into()
        )));
        assert_eq!(
            mappings.last().unwrap(),
            &(head.into(), "lm_head.weight".into())
        );
        let matchers: Vec<KeyMatcher> = omitted.into_iter().map(KeyMatcher::prefix).collect();
        assert_eq!(contract.allow_unexpected(), &matchers[..], "{source:?}");
        assert_eq!(source.head_name(), head);
        assert_eq!(source.language_prefix(), prefix);
    }
}

/// A multimodal source refuses a final soft-cap its reference never applies.
///
/// Regression: a multimodal Gemma 3 checkpoint with a non-null
/// `final_logit_softcapping` used to load and silently cap logits that
/// `Gemma3ForConditionalGeneration` leaves raw.
#[test]
fn multimodal_sources_refuse_a_final_soft_cap() {
    for variant in GemmaVariant::ALL {
        let capped = GemmaConfig {
            final_logit_softcapping: Some(30.0),
            ..small(variant)
        };
        let uncapped = GemmaConfig {
            final_logit_softcapping: None,
            ..capped.clone()
        };
        for source in GemmaCheckpointSource::ALL {
            let options = GemmaCheckpointOptions { source };
            assert!(uncapped.checkpoint_contract(options).is_ok(), "{source:?}");
            let result = capped.checkpoint_contract(options);
            if source == GemmaCheckpointSource::TextOnly {
                assert!(
                    result.is_ok(),
                    "{variant:?}: the text tower applies the cap"
                );
                continue;
            }
            let error = result.unwrap_err();
            assert_eq!(
                error.field, "final_logit_softcapping",
                "{variant:?} {source:?}"
            );
            assert!(
                error.to_string().contains("Gemma3ForConditionalGeneration"),
                "{error}"
            );
        }
    }
}

/// Exactly one language embedding identifies the source.
///
/// No embedding, or several, is a typed [`CheckpointError`] that lists the
/// candidate or present embeddings in detection order.
#[test]
fn sources_are_detected_from_the_embedding_name() {
    for source in GemmaCheckpointSource::ALL {
        let embedding = format!("{}.embed_tokens.weight", source.language_prefix());
        let names = [embedding.as_str(), "vision_tower.x", "lm_head.weight"];
        assert_eq!(GemmaCheckpointSource::detect(names).unwrap(), source);
    }
    let none = GemmaCheckpointSource::detect(["model.norm.weight"]).unwrap_err();
    assert_eq!(
        none.to_string(),
        "Gemma checkpoint contains none of the embeddings [\"model.embed_tokens.weight\", \
         \"model.language_model.embed_tokens.weight\", \
         \"language_model.model.embed_tokens.weight\"]"
    );
    assert!(
        matches!(
            &none,
            CheckpointError::LanguageEmbeddingNotFound { family: "Gemma", embeddings }
                if embeddings.len() == 3
        ),
        "{none:?}"
    );
    let both = GemmaCheckpointSource::detect([
        "language_model.model.embed_tokens.weight",
        "model.norm.weight",
        "model.embed_tokens.weight",
    ])
    .unwrap_err();
    assert!(
        matches!(
            &both,
            CheckpointError::AmbiguousLanguageEmbedding { family: "Gemma", embeddings }
                if embeddings == &[
                    "model.embed_tokens.weight".to_owned(),
                    "language_model.model.embed_tokens.weight".to_owned(),
                ]
        ),
        "{both:?}"
    );
    assert!(both.to_string().contains("ambiguously"), "{both}");
    assert_eq!(
        GemmaCheckpointSource::default(),
        GemmaCheckpointSource::TextOnly
    );
}

/// Invalid configurations have no contract.
#[test]
fn invalid_configurations_have_no_contract() {
    let invalid = GemmaConfig {
        num_key_value_heads: 3,
        ..small(GemmaVariant::Gemma2)
    };
    assert!(
        invalid
            .checkpoint_contract(GemmaCheckpointOptions::default())
            .is_err()
    );
    let overflowing = GemmaConfig {
        num_attention_heads: usize::MAX,
        num_key_value_heads: 1,
        head_dim: 2,
        ..small(GemmaVariant::Gemma2)
    };
    assert!(
        overflowing
            .checkpoint_contract(GemmaCheckpointOptions::default())
            .is_err()
    );
}
