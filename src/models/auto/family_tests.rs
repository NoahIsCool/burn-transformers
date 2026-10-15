//! Tests for the shared language-namespace classifier used by family checkpoint planning.

use proptest::prelude::*;

use super::*;

/// Exactly one official namespace is accepted; neither or both is a contextual error.
#[test]
fn language_namespace_classification_is_total_and_exclusive() {
    assert_eq!(
        classify_language_namespace(["model.language_model.embed_tokens.weight"], "Qwen").unwrap(),
        LanguageNamespace::Wrapped
    );
    assert_eq!(
        classify_language_namespace(["model.embed_tokens.weight", "lm_head.weight"], "Muse")
            .unwrap(),
        LanguageNamespace::TextOnly
    );
    for (names, expected) in [
        (
            vec!["lm_head.weight"],
            "Qwen checkpoint contains neither official language parameter namespace",
        ),
        (
            vec![],
            "Qwen checkpoint contains neither official language parameter namespace",
        ),
        (
            vec![
                "model.language_model.norm.weight",
                "model.embed_tokens.weight",
            ],
            "Qwen checkpoint ambiguously contains wrapped and text-only language namespaces",
        ),
    ] {
        assert!(matches!(
            classify_language_namespace(names, "Qwen").unwrap_err(),
            Error::Checkpoint(reason) if reason == expected
        ));
    }
    // Near misses of either namespace are not mistaken for it.
    assert!(
        classify_language_namespace(
            [
                "model.language_modelx.weight",
                "model.embed_tokens.weight.extra"
            ],
            "Muse"
        )
        .is_err()
    );
}

proptest! {
    /// Wrapped iff some name has the wrapped prefix and none is the text-only embedding.
    #[test]
    fn classification_matches_its_definition(
        names in proptest::collection::vec(
            prop_oneof![
                Just("model.language_model.layers.0.weight".to_owned()),
                Just("model.embed_tokens.weight".to_owned()),
                Just("lm_head.weight".to_owned()),
                "[a-z._]{0,24}",
            ],
            0..8,
        )
    ) {
        let wrapped = names.iter().any(|name| name.starts_with("model.language_model."));
        let text_only = names.iter().any(|name| name == "model.embed_tokens.weight");
        let result = classify_language_namespace(names.iter().map(String::as_str), "Fixture");
        match (wrapped, text_only) {
            (true, false) => prop_assert_eq!(result.unwrap(), LanguageNamespace::Wrapped),
            (false, true) => prop_assert_eq!(result.unwrap(), LanguageNamespace::TextOnly),
            _ => prop_assert!(result.is_err()),
        }
    }
}
